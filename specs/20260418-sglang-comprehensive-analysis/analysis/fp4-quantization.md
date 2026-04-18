# FP4/MXFP4 Quantization and Marlin/FP8 GEMM Kernels

## Part 1: MXFP4 Block-Scaled Quantization

### Overview

MXFP4 (Microscale FP4) uses the E2M1 format (2-bit exponent, 1-bit mantissa) with per-block UE8M0 scaling. Supported for MoE layers with multiple backend paths.

**Source:** `python/sglang/srt/layers/quantization/mxfp4.py`

### E2M1 Format

```
Representable values: [0, 0.5, 1, 1.5, 2, 3, 4, 6]
Decision boundaries:  [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5]
Block size: 32 elements per scale (OCP MX specification)
Storage: Two 4-bit values packed into one uint8
```

### Tensor Layout

```
Weights: uint8 (packed FP4 pairs)
  w13: [num_experts, 2*intermediate_size, hidden_size//2]
  w2:  [num_experts, hidden_size, intermediate_size//2]

Scales: uint8 (UE8M0 format)
  w13_scale: [num_experts, output_dim, hidden_size//32]
  w2_scale:  [num_experts, output_dim, intermediate_size//32]
```

### Backend Support

| Backend | Platform | Notes |
|---------|----------|-------|
| FlashInfer (TRT-LLM) | SM100+ | Epilogue tile shuffling (128×128) |
| AITer | AMD MI300 | Per 1×32 quant with e8m0 scale shuffling |
| Triton | Universal | MX-FP4 layouts, num_warps=8 |
| BF16 fallback | Any | Upcasts via `upcast_from_mxfp` |

### Dynamic Quantization (MoE)

```python
class Mxfp4DynamicQuantMoEMethod:
    # Per-tensor quantization for w1/w3 (2 scales) and w2 (1 scale)
    # On-device dynamic: dynamic_mxfp4_quant() + e8m0_shuffle()
    # Per 1×32 block granularity
```

---

## Part 2: Marlin GEMM Kernels

### Overview

Marlin is a highly optimized GEMM kernel for quantized weight inference on NVIDIA GPUs (SM80+), using tensor cores with in-register dequantization.

**Source:** `sgl-kernel/csrc/gemm/marlin/`

### Configuration

| Parameter | Value | Purpose |
|-----------|-------|---------|
| Threads | 256 (8 warps) | Thread-block size |
| Pipeline stages | 4 | Async global→shared pipeline |
| Tile K | 16 | K dimension per tile |
| Tile N | 64 (16×4) | N dimension per tile |
| Repack stages | 8 | Weight repacking pipeline |
| max_par | 16 | Loop unrolling factor |

### Tensor Core Instructions

```
MMA: mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 (FP16)
     mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 (BF16)
Output: 4× FP32 accumulators per thread
```

### Memory Operations

```
LDSM: ldmatrix.sync.aligned.m8n8.x4  (load 16×16 block)
      ldmatrix.sync.aligned.m8n8.x2  (load 8×16 block)
cp.async.cg.shared.global           (16-byte async copy)
cp.async.commit_group               (stage pipeline)
cp.async.wait_group                  (wait completion)
```

### Key Optimizations

1. **Stripe-based partitioning**: Distributes across SMs, minimizes global reductions
2. **Global reduction strategy**: Barrier sync with atomic operations, lock-based acquire/release
3. **4-stage pipeline**: Hides global memory latency with shared memory staging
4. **In-register dequantization**: No explicit dequantize pass, done inside compute
5. **Optional FP32 reduce**: Better precision for multi-SM accumulation
6. **Zero-point handling**: Per-element scale multiplication with act_order support

### Supported Weight Types

- U4 (unsigned 4-bit)
- U8 (unsigned 8-bit)
- U4B8 (4-bit with 8-bit bias)
- U8B128 (8-bit with 128-bit blocks)
- FE2M1f (MXFP4)

### Files

| File | Purpose |
|------|---------|
| `marlin.cuh` | Constants, async primitives |
| `marlin_template.h` | Full kernel with barrier logic |
| `gptq_marlin.cuh` | GPTQ-specific variant |
| `gptq_marlin_repack.cu` | GPTQ weight repacking |
| `awq_marlin_repack.cu` | AWQ weight repacking |
| `marlin_dtypes.cuh` | Data type definitions |

---

## Part 3: FP8 Blockwise GEMM Kernel

### Overview

CUTLASS-based FP8 blockwise GEMM for SM100+ (Blackwell) and SM120+, supporting block-scaled FP8 with per-128-element scale factors.

**Source:** `sgl-kernel/csrc/gemm/fp8_blockwise_gemm_kernel.cu`

### Architecture-Specific Configurations

**SM100 (Blackwell):**
```
M ≤ 128:
  MMA tile: 64×128×128
  Per-SM tile: 64×128×128
  Epilogue tile: 64×64
  Scale granularity: 64×1×1

M > 128:
  MMA tile: 128×128×128
  Per-SM tile: 128×128×128
  Epilogue tile: 128×64
  Scale granularity: 128×1×1
```

**SM120:**
```
  MMA tile: 128×128×128
  Per-SM tile: 128×128×128
  Epilogue tile: 128×64
  Scale granularity: 128×1×1
```

### Data Layout

```
Matrix A (Input):  Row-major, float8_e4m3
Matrix B (Weights): Column-major, float8_e4m3
Output D:          Row-major, FP16 or BF16
Scale A (SFA):     Column-major (MN-major)
Scale B (SFB):     Column-major (K-major)
```

### CUTLASS Integration

- `GemmUniversal` with TMA warp-specialized scheduler
- Collective builder for mainloop (FP8 blockwise)
- Collective builder for epilogue (cast to FP16/BF16)
- Persistent scheduler for efficient grid mapping
- Auto-carve-out for shared memory

### Alignment Requirements

- Input matrices: 16-byte alignment (128 bits)
- Scale tensors: contiguous, M-major for A, K-major for B
- Row padding to multiple of 4

---

## Part 4: Quantized Linear Layer Integration

### LinearBase Integration

```python
class LinearBase(torch.nn.Module):
    def __init__(self, ..., quant_config=None, prefix=""):
        if quant_config is None:
            self.quant_method = UnquantizedLinearMethod()
        else:
            self.quant_method = quant_config.get_quant_method(self, prefix)
```

### Weight Loader Mechanism

Two patterns:
- **weight_loader (legacy)**: Simple `param.data.copy_(loaded_weight)`
- **weight_loader_v2 (modern)**: Handles TP sharding, packed formats, merged layers

```python
# Modern quantization methods using v2:
WEIGHT_LOADER_V2_SUPPORTED = [
    "Fp8LinearMethod", "AWQMarlinLinearMethod", "GPTQMarlinLinearMethod",
    "BlockInt8LinearMethod", "MarlinLinearMethod", ...  # 20+ methods
]
```

### Parameter Classes

| Class | Purpose |
|-------|---------|
| `ModelWeightParameter` | Standard weights (input_dim, output_dim) |
| `PackedvLLMParameter` | Packed/quantized (pack_factor, packed_dim) |
| `PerTensorScaleParameter` | Per-tensor scales |
| `BlockQuantScaleParameter` | Block-wise scales (format_ue8m0 flag) |
| `RowvLLMParameter` | Row-parallel weights |
| `_ColumnvLLMParameter` | Column-parallel weights |

### Checkpoint Loading Flow

```
1. Model construction:
   - quant_config.get_quant_method(layer) → QuantizeMethodBase
   - quant_method.create_weights() → registers parameters with weight_loader

2. Checkpoint iteration:
   - For each (name, tensor) in checkpoint:
     - Find matching param in params_dict
     - Call param.weight_loader(param, loaded_weight, shard_id)
     - Handles TP sharding, packing, merging

3. Post-processing:
   - quant_method.process_weights_after_loading(layer)
   - Handles: online quantization, scale conversion, format normalization
```

---

## Format Comparison

| Feature | MXFP4 | FP8 Blockwise | Marlin INT4 |
|---------|--------|--------------|-------------|
| Mantissa bits | 1 | 3 | N/A (integer) |
| Block size | 32 | 128 | Group (32/64/128) |
| Scale storage | uint8 (UE8M0) | float32 | float16 |
| Primary use | MoE sparse | Dense GEMM | Dense GEMM |
| Min hardware | SM80 | SM100 | SM80 |
| Memory savings | ~8× vs FP16 | ~2× vs FP16 | ~4× vs FP16 |
