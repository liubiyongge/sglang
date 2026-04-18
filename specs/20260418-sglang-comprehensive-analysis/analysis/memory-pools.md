# T015: Memory Pools Analysis

**Source**: `python/sglang/srt/mem_cache/memory_pool.py`, `python/sglang/srt/mem_cache/allocator.py`
**Cross-references**: [radix-cache.md](radix-cache.md), [hicache.md](hicache.md), [batch-data-structures.md](batch-data-structures.md)

---

## Overview

SGLang uses a two-level memory pool architecture for KV cache management:
1. **ReqToTokenPool**: Maps requests to their token cache locations (logical layer)
2. **KVCache + Allocator**: Manages physical GPU memory for KV tensors (physical layer)

---

## ReqToTokenPool

### Purpose

Maintains a bidirectional mapping between requests and their KV cache slot indices.

### Data Structure

```python
class ReqToTokenPool:
    req_to_token: torch.Tensor  # [num_requests, max_context_len], int32, GPU
    free_slots: List[int]       # Available request pool slots
```

### Operations

- **alloc(reqs)**: Allocate pool indices for new requests. Chunked requests reuse previous slots.
- **free(req)**: Return slot to free list, set `req.req_pool_idx = None`
- **write()**: Write token-to-KV-cache mappings via Triton kernel

### Integration

```
ForwardBatch.req_pool_indices[i] → req_to_token[pool_idx, :seq_len] → KV cache indices
```

---

## KVCache Implementations

### MHATokenToKVPool (Standard Multi-Head Attention)

```python
k_buffer: List[torch.Tensor]  # [num_layers] → [size + page_size, head_num, head_dim]
v_buffer: List[torch.Tensor]  # [num_layers] → [size + page_size, head_num, v_head_dim]
```

**Memory per token:** `head_num × head_dim × dtype_bytes × 2` (K + V)
- Llama 70B (BF16): 64 × 128 × 2 × 2 = 32KB/token

### MLATokenToKVPool (Multi-Latent Attention — DeepSeek)

```python
kv_buffer: List[torch.Tensor]  # [num_layers] → [size + page_size, 1, kv_lora_rank + rope_dim]
```

**Memory per token:** `(kv_lora_rank + qk_rope_head_dim) × dtype_bytes`
- DeepSeek-V3 (BF16): (512 + 64) × 2 = 1.15KB/token (27× more efficient than MHA)

**Decomposed access:**
- `cache_k_nope`: KV lora components (quantizable)
- `cache_k_rope`: RoPE components (full precision)
- Set/get via Triton kernels with boundary handling

### FP4TokenToKVPool / FP8TokenToKVPool

```python
k_buffer: List[torch.Tensor]       # Packed quantized values
k_scale_buffer: List[torch.Tensor] # Per-block scaling factors
```

**Compression:**
- FP8: 8× compression (4KB/token for 64-head model)
- FP4: 16× compression (2KB/token), with per-16-element scales
- Transparent dequantization on read

### DoubleSparseTokenToKVPool

```python
k_buffer, v_buffer: standard layout
label_buffer: List[torch.Tensor]  # [size+1, head_num, heavy_channel_num]
```

Stores sparse attention pattern labels for selective computation.

---

## TokenToKVPoolAllocator

### Token-Level Allocation (page_size=1)

```python
class TokenToKVPoolAllocator:
    free_pages: torch.Tensor  # [num_free_tokens]

    def alloc(self, need_size):
        select = self.free_pages[:need_size]
        self.free_pages = self.free_pages[need_size:]
        return select

    def free(self, indices):
        self.release_pages = cat(self.release_pages, indices)
```

### Paged Allocation (page_size ≥ 64)

```python
class PagedTokenToKVPoolAllocator:
    free_pages: torch.Tensor  # [num_free_pages]

    def alloc(self, need_size):
        num_pages = need_size // page_size
        out_pages = self.free_pages[:num_pages]
        # Convert pages to token indices
        out_indices = (out_pages[:, None] * page_size + arange(page_size)).reshape(-1)
        return out_indices
```

### Triton Kernel Integration

For paged allocation during extend/decode, specialized Triton kernels handle:
- Three-part fill pattern (remaining current page → new full pages → partial last page)
- Page boundary alignment
- Batch-parallel allocation (one block per request)

### Free List Management

- **Deferred sorting**: Released pages accumulated, sorted only when allocation fails
- **Batch free**: Group multiple free operations for efficiency
- **Locality optimization**: Sorted free list improves cache locality

---

## Memory Sizing

### Typical Configurations

| Model | Pool Type | Per-Token | 8K Tokens × Layers | Total |
|-------|-----------|-----------|---------------------|-------|
| Llama 70B | MHA BF16 | 32KB | 32KB × 8K × 80 | 20.5GB |
| DeepSeek-V3 | MLA BF16 | 1.15KB | 1.15KB × 8K × 61 | 560MB |
| Llama 70B | FP8 MHA | 4KB | 4KB × 8K × 80 | 2.56GB |
| Llama 70B | FP4 MHA | 2KB | 2KB × 8K × 80 | 1.28GB |

### Pool Size Determination

```python
total_kv_size = (
    available_gpu_memory
    - model_weights_size
    - activation_memory
    - cuda_graph_memory
) / bytes_per_token_per_layer / num_layers
```

---

## Allocation Functions (common.py)

### `alloc_for_extend(batch)`

1. Evict SWA tokens if needed
2. Allocate request pool indices (`ReqToTokenPool.alloc`)
3. Allocate KV cache tokens (paged or token-level)
4. Write prefix + new token indices to `req_to_token` via Triton kernel
5. Returns `out_cache_loc` for forward pass

### `alloc_for_decode(batch, token_per_req=1)`

1. Evict SWA tokens if needed
2. Allocate 1 token per request (may trigger new page allocation)
3. Write to `req_to_token` at position `seq_lens`
4. Returns `out_cache_loc` for forward pass

### `evict_from_tree_cache(tree_cache, num_tokens)`

Triggers tree cache eviction when allocator reports insufficient free tokens.

### `release_kv_cache(req, tree_cache, is_insert=True)`

Cleanup after request completion:
1. Notify tree cache (`cache_finished_req`)
2. Release overallocated KV cache
3. Free mamba state if applicable
4. Free request pool slot

---

## Performance Optimizations

1. **Alternate stream for K/V writes**: Overlaps K and V cache stores during CUDA graph capture
2. **Store cache JIT kernel**: Optimized for dense indexing patterns
3. **Copy tiling**: Adaptive tile sizes (128-512 bytes) based on stride
4. **Batch operations**: Group alloc/free to reduce kernel launches
5. **Non-blocking transfers**: `to(device, non_blocking=True)` for CPU↔GPU
