# T019: KV Cache Store Kernel Analysis

**Source**: `sgl-kernel/csrc/memory/store.cu`
**Cross-references**: [memory-pools.md](memory-pools.md), [model-execution.md](model-execution.md)

---

## Overview

The `store_kv_cache` kernel is a high-performance CUDA kernel for writing newly computed K and V tensors into the pre-allocated KV cache pool at scattered locations specified by `out_cache_loc`. It uses warp-level parallelism to achieve maximum memory bandwidth utilization.

---

## Kernel Variants

### `store_kv_cache_256x1` — 256 bytes per warp per iteration

```cuda
// Each warp processes one token's K and V sequentially
// 32 threads × 8 bytes (uint64_t) = 256 bytes per loop iteration
for (size_t i = 0; i < num_items; ++i) {
    k_dst[lane_id + i * 32] = k_src[lane_id + i * 32];
    v_dst[lane_id + i * 32] = v_src[lane_id + i * 32];
}
```

- **Used when**: `size_bytes % 256 == 0` (head_dim × elem_size divisible by 256)
- **Parallelism**: 1 warp per token, K and V written sequentially
- **Bandwidth**: Full 256B coalesced writes per iteration

### `store_kv_cache_128x2` — 128 bytes per half-warp, K and V in parallel

```cuda
// Split warp: lanes 0-15 handle K, lanes 16-31 handle V
const auto copy_k = lane_id < 16;
const auto copy_id = lane_id % 16;
const auto cache = copy_k ? k_cache : v_cache;
const auto input = copy_k ? k : v;
for (size_t i = 0; i < num_items; ++i) {
    dst[copy_id + i * 16] = src[copy_id + i * 16];
}
```

- **Used when**: `size_bytes % 128 == 0` (but not % 256)
- **Parallelism**: Half-warp for K, half-warp for V (concurrent)
- **Bandwidth**: 128B coalesced writes × 2 (K and V overlap)

---

## Interface

```cpp
void store_kv_cache(
    at::Tensor k_cache,   // [max_tokens, ...] — destination KV pool
    at::Tensor v_cache,   // [max_tokens, ...] — destination KV pool
    at::Tensor out_loc,   // [num_tokens] — scatter indices (int32/int64)
    at::Tensor k,         // [num_tokens, ...] — source K from forward pass
    at::Tensor v          // [num_tokens, ...] — source V from forward pass
)
```

### Requirements

- All tensors must be CUDA
- K/V cache and input must have matching last dimension (head_dim)
- Last dimension must be contiguous (`stride(-1) == 1`)
- `out_loc` must be 1D contiguous
- Head size in bytes must be divisible by 128

---

## Execution Configuration

```
threads_per_block = 256
warps_per_block = 8
blocks = ceil(num_tokens / warps_per_block)
```

Each warp independently processes one token:
1. Read `out_loc[warp_id]` → target cache slot
2. Compute source offset: `warp_id * kv_input_stride`
3. Compute destination offset: `out_loc[warp_id] * kv_cache_stride`
4. Copy `size_bytes` data using coalesced 8-byte transactions

---

## Performance Characteristics

- **Coalesced access**: All 32 threads in a warp access consecutive addresses
- **No bank conflicts**: Sequential uint64_t loads/stores
- **Scatter pattern**: `out_loc` provides arbitrary target indices (non-sequential)
- **Memory bandwidth bound**: Achieves near-peak bandwidth for large head dimensions
- **No shared memory**: Simple load-store pattern, no intermediate buffering needed

### Typical Sizes

| Model | head_dim × elem_size | Kernel | Items/Warp |
|-------|---------------------|--------|------------|
| Llama (BF16) | 128 × 2 = 256B | 256x1 | 1 |
| Llama (FP32) | 128 × 4 = 512B | 256x1 | 2 |
| GPT-NeoX (BF16) | 96 × 2 = 192B | 128x2 | 1.5 |

---

## Integration Point

Called from `MHATokenToKVPool.set_kv_buffer()` when `can_use_store_cache()` returns true:

```python
def set_kv_buffer(self, layer, loc, cache_k, cache_v):
    if can_use_store_cache(row_bytes):
        store_cache(cache_k, cache_v, self.k_buffer[layer], self.v_buffer[layer], loc)
    else:
        # Fallback: direct indexing
        self.k_buffer[layer][loc] = cache_k
        self.v_buffer[layer][loc] = cache_v
```

The kernel is faster than PyTorch's indexing for scattered writes because it avoids the overhead of PyTorch's dispatch and achieves guaranteed coalesced access patterns.
