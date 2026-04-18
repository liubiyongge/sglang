# INT4 Quantization (AWQ & GPTQ with Marlin)

## Overview

SGLang supports two primary INT4 quantization methods—AWQ and GPTQ—both accelerated via the Marlin kernel for high-performance inference on NVIDIA GPUs. Both use group quantization with packed INT32 weight storage but differ in zero-point handling, weight layout, and activation order support.

**Key Source Files:**
- `python/sglang/srt/layers/quantization/awq.py` - AWQ config and methods
- `python/sglang/srt/layers/quantization/gptq.py` - GPTQ config and methods
- `python/sglang/srt/layers/quantization/marlin_utils.py` - Marlin integration
- `python/sglang/srt/layers/quantization/awq_triton.py` - AWQ Triton fallback
- `sgl-kernel/csrc/gemm/marlin/` - Marlin CUDA kernels

---

## Weight Format

### INT4 Packing

Both methods pack 4-bit weights into 32-bit integers:
- **Pack factor**: 32 / 4 = 8 weights per int32
- Each int32: `[w7, w6, w5, w4, w3, w2, w1, w0]` (4 bits each)

### Storage Dimensions

| Aspect | AWQ | GPTQ |
|--------|-----|------|
| qweight shape | `[K, N/8]` | `[K/8, N]` |
| Packed dimension | Column (dim=1) | Row (dim=0) |
| Scales shape | `[K/G, N]` | `[K/G, N]` |
| Zero points shape | `[K/G, N/8]` | `[K/G, N/8]` |

Where `G = group_size`, `K = input_size`, `N = output_size`.

### Group Quantization

Supported group sizes: `[-1, 32, 64, 128]`
- `-1`: Channelwise (one group per output channel)
- Positive: Group size in input dimension

---

## AWQ vs GPTQ Differences

| Feature | AWQ | GPTQ |
|---------|-----|------|
| Weight range | Unsigned [0, 15] | Signed [-8, 7] |
| Zero-point | Asymmetric (runtime subtraction) | Symmetric (bias embedded) |
| Activation order (desc_act) | Not supported | Optional |
| Packing | Column (along N) | Row (along K) |
| g_idx | Not used | Used if desc_act=True |
| Min capability | SM75 (Turing) | SM60 (Pascal) |
| Marlin capability | SM80 (Ampere) | SM80 (Ampere) |

### AWQ Interleaving

AWQ checkpoints use reverse order: `[0, 4, 1, 5, 2, 6, 3, 7]`
- Must be undone during Marlin repacking
- Triton kernel handles this in dequantization

### GPTQ Activation Order (desc_act)

When `desc_act=True`:
- Input columns reordered by quantization difficulty
- Permutation stored in `g_idx` tensor
- Improves quantization quality but complicates computation
- Requires sorting and permutation during Marlin repack

---

## Marlin Kernel Acceleration

### When Marlin is Used

Requirements:
- SM 8.0+ (Ampere and newer)
- Output channels: `N % 64 == 0`
- Input channels: `K % 128 == 0`
- Group size in `[-1, 32, 64, 128]`

### Marlin Tile Format

After repacking: `[K/16, N*4]` int32 tensor
- 16×64 tiles optimized for tensor core operations
- Weights reorganized to match MMA instruction layouts
- 4-stage async pipeline in shared memory

### Scale Permutation

```python
# For 4-bit:
scale_perm = [0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15]
# For 8-bit:
scale_perm_single = [0, 2, 4, 6, 1, 3, 5, 7]
```

Ensures scales align with quantized weights during in-kernel dequantization.

### Workspace

```python
workspace = torch.zeros(sms * max_blocks_per_sm)
# Used for atomic operations and block synchronization
```

---

## Weight Loading

### AWQ → Marlin Repack

```python
# 1. Repack weights to Marlin format
marlin_qweight = awq_marlin_repack(qweight, ...)  # → [K/16, N*4]

# 2. Permute scales
marlin_scales = marlin_permute_scales(scales, ...)

# 3. Convert zero points
marlin_zp = awq_to_marlin_zero_points(qzeros, ...)
```

### GPTQ → Marlin Repack

```python
# 1. Sort g_idx (if desc_act)
sort_indices, perm = marlin_sort_g_idx(g_idx)

# 2. Repack with permutation
marlin_qweight = gptq_marlin_repack(qweight, perm, ...)  # → [K/16, N*4]

# 3. Permute scales
marlin_scales = marlin_permute_scales(scales, ...)

# 4. Handle zero points
marlin_zp = marlin_zero_points(qzeros, ...)  # if symmetric
```

### is_k_full Parameter

```python
is_k_full = (not act_order) or (act_order and not is_row_parallel)
```
- `True`: All input columns present (no permutation needed)
- `False`: Partial columns (desc_act permutation applied)

---

## Forward Pass

### AWQ Marlin Path

```python
def apply(self, layer, x, bias):
    return apply_awq_marlin_linear(
        input=x,
        weight=layer.qweight,          # Marlin format
        weight_scale=layer.scales,
        weight_zp=layer.qzeros,         # Marlin format
        g_idx=layer.g_idx,              # Empty for AWQ
        workspace=layer.workspace,
        quant_type=scalar_types.uint4,
        bias=bias)
```

### AWQ Non-Marlin Path (Fallback)

```python
def apply(self, layer, x, bias):
    # Step 1: Dequantize to full precision
    out = awq_dequantize(layer.qweight, layer.scales, layer.qzeros)  # [K, N]
    # Step 2: Standard matrix multiply
    out = torch.matmul(x, out)
```

### GPTQ Marlin Path

```python
def apply(self, layer, x, bias):
    return gptq_marlin_gemm(
        a=x,
        b_q_weight=layer.qweight,      # Marlin format
        b_scales=layer.scales,
        b_zeros=layer.qzeros,
        g_idx=layer.g_idx,              # Sorted if desc_act
        perm=layer.g_idx_sort_indices,
        workspace=layer.workspace,
        b_q_type=scalar_types.uint4b8,
        size_m=x.shape[0], size_n=N, size_k=K,
        is_k_full=layer.is_k_full)
```

### GPTQ Non-Marlin Path (Fallback)

```python
def apply(self, layer, x, bias):
    output = gptq_gemm(
        x, layer.qweight, layer.qzeros, layer.scales,
        layer.g_idx, self.use_shuffle, self.weight_bits)
```

---

## Marlin Kernel Architecture

### Tile-Based Computation

```
tile_k_size = 16
tile_n_size = 16 * 4 = 64
threads = 256 (8 warps × 32 threads)
```

### Warp-Level Operations

```
min_thread_n = 64   (N granularity)
min_thread_k = 64   (K granularity)
max_thread_n = 256  (max N per block)
```

### Tensor Core Operations

- MMA: 16×16×16 or 16×8×16 depending on datatype
- Accumulates in FP32, converts to FP16/BF16 at output
- 4-stage async pipeline hides memory latency

### Why Marlin is Faster

| Operation | Non-Marlin | Marlin |
|-----------|-----------|--------|
| Dequantization | Full explicit | Inside compute |
| Memory bandwidth | Very high | Low (4-bit weights) |
| Tensor core usage | Limited | Full |
| Estimated throughput | ~3-15 TFLOPS | ~50-80 TFLOPS |

---

## MoE Support

### AWQ MoE

```python
# Per-expert weights (fused gate+up and down projections)
w13_qweight: (num_experts, hidden, 2*intermediate/8)
w2_qweight: (num_experts, intermediate/8, hidden)

# Marlin repack for MoE
marlin_w13 = awq_marlin_moe_repack(layer.w13_qweight, ...)
marlin_w2 = awq_marlin_moe_repack(layer.w2_qweight, ...)
```

### GPTQ MoE

```python
marlin_w13 = gptq_marlin_moe_repack(
    layer.w13_qweight, layer.w13_g_idx_sort_indices, ...)
```

---

## Performance Characteristics

### Memory Savings

4-bit weights reduce memory by ~4× compared to FP16:
- FP16 model: 7B → ~14 GB
- INT4 model: 7B → ~3.5 GB

### Compute Efficiency

Marlin maintains near-FP16 throughput despite 4× memory reduction:
- Dequantization happens in registers (zero memory overhead)
- Tensor cores fully utilized
- Async pipeline hides global memory latency

---

## Configuration Summary

| Parameter | AWQ | GPTQ |
|-----------|-----|------|
| `bits` | 4 | 2, 3, 4, 8 |
| `group_size` | 32, 64, 128 | -1, 32, 64, 128 |
| `desc_act` | N/A | True/False |
| `sym` | False (asymmetric) | True/False |
| `quant_method` | "awq" | "gptq" |

---

## Kernel File Locations

| Component | Path |
|-----------|------|
| AWQ Marlin Repack | `sgl-kernel/csrc/gemm/marlin/awq_marlin_repack.cu` |
| GPTQ Marlin Repack | `sgl-kernel/csrc/gemm/marlin/gptq_marlin_repack.cu` |
| Marlin Header | `sgl-kernel/csrc/gemm/marlin/marlin.cuh` |
| GPTQ Marlin Kernel | `sgl-kernel/csrc/gemm/marlin/gptq_marlin.cuh` |
| Marlin Data Types | `sgl-kernel/csrc/gemm/marlin/marlin_dtypes.cuh` |
| Fused Marlin MoE | `python/sglang/srt/layers/moe/fused_moe_triton/fused_marlin_moe.py` |
