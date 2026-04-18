# T014: RadixCache Tree Structure Analysis

**Source**: `python/sglang/srt/mem_cache/radix_cache.py`, `python/sglang/srt/mem_cache/base_prefix_cache.py`
**Cross-references**: [memory-pools.md](memory-pools.md), [hicache.md](hicache.md), [scheduling-policies.md](scheduling-policies.md)

---

## Overview

The RadixCache is a production-grade radix tree (Patricia tree) designed for efficient KV cache management. It enables prefix sharing across concurrent requests, significantly reducing memory footprint when requests share common prefixes (system prompts, retrieval contexts).

---

## Data Structure

### TreeNode

```python
class TreeNode:
    children: dict[Union[int, tuple], TreeNode]  # Child nodes by token/page key
    parent: TreeNode                              # Upward link
    key: RadixKey                                 # Token sequence this node represents
    value: Optional[torch.Tensor]                # GPU KV cache indices (None if evicted)

    # Reference counting
    lock_ref: int                                # >0 means protected from eviction

    # Eviction tracking
    last_access_time: float                      # LRU
    creation_time: float                         # FIFO
    hit_count: int                               # LFU
    priority: int                                # Explicit priority

    # Hierarchical cache
    host_value: Optional[torch.Tensor]          # CPU memory indices (L2)
    host_ref_counter: int                        # CPU protection count
    hash_value: Optional[List[str]]             # SHA256 per page (L3)
```

### RadixKey

```python
class RadixKey:
    token_ids: List[int]        # Token sequence
    extra_key: Optional[str]    # Namespace isolation (LoRA ID, cache version)
    is_bigram: bool             # EAGLE speculative decoding format
```

### Tree Organization

```
Root (empty key, always locked)
├── [token_1] → Node A (key=[1,2,3], value=[kv_indices])
│   ├── [token_4] → Node B (key=[4,5], value=[kv_indices])
│   └── [token_6] → Node C (key=[6,7,8], value=[kv_indices])
├── (lora_id=1, token_1) → Node D  # Namespace isolated
└── [token_9] → Node E
```

---

## Prefix Matching Algorithm

### `match_prefix()` — O(k) where k = query length

```python
def _match_prefix_helper(self, node, key):
    value = []
    while len(key) > 0 and child_key in node.children:
        child = node.children[child_key]
        prefix_len = self.key_match_fn(child.key, key)

        if prefix_len < len(child.key):
            # PARTIAL MATCH: Split node
            new_node = self._split_node(child.key, child, prefix_len)
            value.append(new_node.value)
            node = new_node
            break
        else:
            # FULL MATCH: Continue traversing
            value.append(child.value)
            node = child
            key = key[prefix_len:]

    return value, node
```

**Key behaviors:**
- Traverses tree following matching token prefixes
- Stops at first mismatch or evicted node
- Triggers lazy node splitting on partial matches
- Updates `last_access_time` and `hit_count` on accessed nodes

---

## Node Splitting

When a query partially matches a stored node, the node splits:

```
Before: parent → child(key=[1,2,3,4,5], value=[kv1..kv5])
Query:  [1,2,3,10,11] → prefix_len=3

After:  parent → new_node(key=[1,2,3], value=[kv1..kv3])
                   ├── child(key=[4,5], value=[kv4..kv5])
                   └── (new leaf for [10,11] on insert)
```

**Properties:**
- Tensor values are cloned (not aliased)
- Lock references inherited
- Hash values split appropriately
- Only one split per match operation (lazy)

---

## Reference Counting (Lock Mechanism)

### Protection: `inc_lock_ref(node)`

```python
def inc_lock_ref(self, node):
    while node != self.root_node:
        if node.lock_ref == 0:
            self.evictable_size_ -= len(node.key)
            self.protected_size_ += len(node.key)
        node.lock_ref += 1
        node = node.parent  # Walk ALL the way to root
```

### Unprotection: `dec_lock_ref(node)`

Symmetric operation: decrements lock_ref, re-adds to evictable if reaching 0.

### Invariants

1. If `lock_ref > 0`, all ancestors to root also have `lock_ref > 0`
2. `protected_size_ + evictable_size_ = total_cached`
3. Only nodes with `lock_ref == 0` and no non-evicted children are evictable

---

## Eviction

### Algorithm

```python
def evict(self, params):
    # Build min-heap from evictable leaves
    eviction_heap = [(strategy.get_priority(n), n) for n in self.evictable_leaves]
    heapq.heapify(eviction_heap)

    while num_evicted < num_tokens:
        _, node = heapq.heappop(eviction_heap)
        self.token_to_kv_pool_allocator.free(node.value)
        num_evicted += len(node.value)
        self._delete_leaf(node)

        # Promote parent if now a leaf
        if parent_is_now_leaf:
            heapq.heappush(eviction_heap, (priority(parent), parent))
```

### Supported Strategies

| Strategy | Priority | Evicts First |
|----------|----------|--------------|
| LRU | `last_access_time` | Least recently accessed |
| LFU | `(hit_count, last_access_time)` | Least frequently used |
| FIFO | `creation_time` | Oldest created |
| MRU | `-last_access_time` | Most recently used |
| Priority | `(priority, last_access_time)` | Lowest priority |

### Cascading

When a leaf is evicted and its parent has no other non-evicted children, the parent becomes a new eviction candidate.

---

## Namespace Isolation (`extra_key`)

Different `extra_key` values create completely separate subtrees:

```python
def get_child_key(key, page_size=1):
    if key.extra_key is None:
        return plain_key  # e.g., token_id or tuple
    else:
        return (key.extra_key, plain_key)  # Composite key
```

**Use cases:**
- LoRA adapter isolation
- Cache versioning
- Multi-tenant separation
- Sampling parameter isolation

---

## Simulated Tree (`create_simulated`)

For in-batch prefix caching and testing:

```python
@classmethod
def create_simulated(cls, disable=False, mock_allocator=None, page_size=1):
    return RadixCache(CacheInitParams(
        req_to_token_pool=None,  # No actual GPU memory
        token_to_kv_pool_allocator=mock_allocator,
    ))
```

Used by `SchedulePolicy.waiting_queue_radix_tree` to detect shared prefixes among waiting requests without touching GPU memory.

---

## Performance Characteristics

| Operation | Complexity | Notes |
|-----------|------------|-------|
| `match_prefix()` | O(k) | k = query tokens, linear tree scan |
| `insert()` | O(k) | Includes matching + leaf creation |
| `_split_node()` | O(k) | Tensor cloning |
| `evict()` | O(n + m log n) | n = evictable leaves, m = evictions |
| `inc/dec_lock_ref()` | O(d) | d = tree depth (typically 10-100) |

**Compression ratio:** With 80% shared prefix across 1000 requests, ~80% memory savings vs. independent storage.
