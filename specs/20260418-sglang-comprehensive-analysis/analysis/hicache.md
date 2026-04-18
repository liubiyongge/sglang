# T017: HiRadixCache (Hierarchical Cache) Analysis

**Source**: `python/sglang/srt/mem_cache/hiradix_cache.py`, `python/sglang/srt/mem_cache/evict_policy.py`, `python/sglang/srt/mem_cache/common.py`
**Cross-references**: [radix-cache.md](radix-cache.md), [memory-pools.md](memory-pools.md)

---

## Overview

HiRadixCache extends the base RadixCache with a three-tier memory hierarchy: GPU (L1) → CPU Host (L2) → External Storage (L3). It enables efficient long-context inference through intelligent data movement between tiers, configurable write policies, and async operations.

---

## Three-Tier Architecture

```
┌────────────────────────────────────────────────────────────────┐
│ L1: GPU Memory (Device)                                        │
│ - Access: ~microseconds                                        │
│ - Managed by: RadixCache (TreeNode.value = GPU indices)        │
│ - Purpose: Active inference, hot data                          │
├────────────────────────────────────────────────────────────────┤
│ L2: CPU Host Memory                                            │
│ - Access: ~milliseconds                                        │
│ - Managed by: HiCacheController.mem_pool_host                  │
│ - Data: TreeNode.host_value (CPU indices)                      │
│ - Purpose: Recently evicted data, warm cache                   │
├────────────────────────────────────────────────────────────────┤
│ L3: External Storage                                           │
│ - Access: ~seconds                                             │
│ - Backends: File, NIXL, Mooncake, HF3FS, EIC, AIBrix          │
│ - Data: TreeNode.hash_value (SHA256 page hashes)               │
│ - Purpose: Cold data, cross-session persistence                │
└────────────────────────────────────────────────────────────────┘
```

---

## Data Flow Between Tiers

### Eviction Path (L1 → L2 → L3)

```
GPU Full → evict() triggered
  ├── write_back policy: Copy to CPU during eviction (blocking)
  ├── write_through policy: Already in CPU (async earlier)
  └── Both: Free GPU memory after CPU confirmed

CPU Full → evict_host() triggered
  ├── Free CPU memory for oldest/lowest-priority nodes
  └── Storage backup may have already preserved data
```

### Load-Back Path (L3 → L2 → L1)

```
Request matches evicted node → init_load_back()
  ├── If host_value exists: CPU → GPU (async, ~ms)
  ├── If only in storage: prefetch_from_storage() → Storage → CPU (async, ~s)
  │                        then load_back() → CPU → GPU
  └── Returns device_indices immediately (future populated async)
```

---

## Write Policies

### write_back (Deferred)

- **Behavior**: No CPU backup on insert; backup only during eviction
- **Insert latency**: Low (GPU-only)
- **Eviction latency**: Higher (must transfer before freeing GPU)
- **Best for**: Write-heavy workloads, short-lived caches

### write_through (Immediate)

- **Behavior**: Backup to CPU on first cache hit (`hit_count >= 1`)
- **Insert latency**: Higher (async GPU→CPU transfer queued)
- **Eviction latency**: Low (already backed up)
- **Best for**: Read-heavy workloads, persistent data

### write_through_selective (Adaptive)

- **Behavior**: Backup after 2nd hit (`hit_count >= 2`)
- **Rationale**: Avoids backing up cold one-shot data
- **Best for**: Mixed workloads (default recommendation)

---

## Eviction Policies

All policies implemented via `EvictionStrategy.get_priority(node)`:

| Policy | Priority Key | Evicts First |
|--------|-------------|--------------|
| LRU | `last_access_time` | Least recently accessed |
| LFU | `(hit_count, access_time)` | Least frequently used |
| FIFO | `creation_time` | Oldest created |
| MRU | `-last_access_time` | Most recently used |
| FILO | `-creation_time` | Newest created |
| Priority | `(priority, access_time)` | Lowest explicit priority |

### Eviction Process

1. Build min-heap from `evictable_leaves`
2. Pop lowest-priority leaf
3. If write_back and not backed up: queue GPU→CPU transfer
4. If already backed up: free GPU directly
5. Promote parent to heap if all children now evicted
6. Repeat until `num_tokens` freed

---

## Lock/Reference Counting

### GPU Lock (`lock_ref`)

- `inc_lock_ref(node)`: Protects node + all ancestors from eviction
- `dec_lock_ref(node)`: Unprotects when request completes or transfer done
- **Used by**: Active requests, write_through async transfers, load_back

### Host Lock (`host_ref_counter`)

- `protect_host()`: Prevents CPU eviction during storage operations
- `release_host()`: Releases after storage read/write completes
- **Used by**: Storage prefetch and backup operations

### Leaf Status Tracking

```python
# GPU evictable leaves
evictable_leaves: Set[TreeNode]  # lock_ref==0, no non-evicted children

# CPU evictable leaves
evictable_host_leaves: Set[TreeNode]  # evicted from GPU, host_ref_counter==0
```

---

## Async Operations

### CUDA Stream Architecture

| Stream | Direction | Purpose |
|--------|-----------|---------|
| `write_stream` | GPU → CPU | KV cache backup |
| `load_stream` | CPU → GPU | KV cache load-back |

### Background Threads

| Thread | Direction | Purpose |
|--------|-----------|---------|
| `prefetch_thread` | Storage → CPU | Async page retrieval |
| `backup_thread` | CPU → Storage | Async page persistence |

### Synchronization

- **CUDA Events**: Track GPU↔CPU transfer completion (non-blocking query)
- **Layer-Done Counters**: Enable layer-by-layer pipeline overlap during load-back
- **TP All-Reduce**: Ensure consistent state across tensor-parallel ranks

---

## Storage Backend Abstraction

### Interface

```python
class HiCacheStorage(ABC):
    def get(self, key: str) -> Optional[Tensor]
    def set(self, key: str, value: Tensor) -> bool
    def batch_get_v1(self, keys, host_indices) -> List[bool]  # Zero-copy
    def batch_set_v1(self, keys, host_indices) -> List[bool]  # Zero-copy
    def batch_exists(self, keys) -> int  # Prefix existence check
```

### Available Backends

| Backend | Storage | Use Case |
|---------|---------|----------|
| File | Local filesystem | Development, single-node |
| NIXL | GPU-attached NVMe | High-performance, GPU-local |
| Mooncake | Distributed object store | Multi-node, cross-session |
| HF3FS | Distributed FS | Enterprise, throughput-optimized |
| EIC | Edge intelligent cache | Edge computing |
| AIBrix | KV cache service | Specialized storage service |

### Factory Pattern

```python
StorageBackendFactory.create_backend(backend_name, config, mem_pool_host)
```

Supports runtime attach/detach and lazy module loading.

---

## Scheduler Integration

### `init_load_back(last_node, host_hit_length)`

Called by scheduler after `match_prefix()` indicates CPU cache hit:

1. If `last_node.evicted` and host data available:
   - Allocate GPU memory
   - Queue async CPU→GPU transfer
   - Lock ancestor nodes
   - Return device_indices (populated async)
2. If load fails (insufficient GPU memory):
   - Walk up tree to find resident ancestor
   - Return empty indices

### `host_hit_length` Semantics

Accumulated tokens available in CPU tier below the GPU match point:

```python
match_result = tree_cache.match_prefix(key)
# match_result.device_indices → tokens in GPU (immediate use)
# match_result.host_hit_length → tokens in CPU (can load-back)
# match_result.last_host_node → node to start load-back from
```

### Event Checking

`check_hicache_events()` called each scheduler iteration:
- `writing_check()`: Drain completed GPU→CPU transfers, dec_lock_ref
- `loading_check()`: Drain completed CPU→GPU transfers, dec_lock_ref  
- `drain_storage_control_queues()`: Process prefetch/backup completions

---

## Configuration

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `--hicache-ratio` | 2.0 | CPU pool = GPU pool × ratio |
| `--hicache-size` | 0 | Override CPU pool size (bytes) |
| `--hicache-mem-layout` | page_first | page_first, layer_first, page_first_direct |
| `--hicache-write-policy` | write_through_selective | write_back, write_through, write_through_selective |
| `--hicache-io-backend` | kernel | kernel, direct, kernel_ascend |
| `--hicache-storage-backend` | None | file, nixl, mooncake, hf3fs, eic, aibrix |
| `--hicache-storage-prefetch-policy` | best_effort | best_effort, wait_complete, timeout |
