# FP8 Quantization

## Overview

SGLang's FP8 quantization implementation supports multiple modes (per-tensor, per-channel, block-wise, MXFP8) with automatic backend selection across multiple hardware platforms. The system handles both pre-quantized checkpoints and online quantization during weight loading.

**Key Source Files:**
- `python/sglang/srt/layers/quantization/fp8.py` (~1645 lines) - Config and methods
- `python/sglang/srt/layers/quantization/fp8_kernel.py` (~2051 lines) - Triton kernels
- `python/sglang/srt/layers/quantization/fp8_utils.py` (~1340 lines) - Dispatch utilities
- `python/sglang/srt/layers/quantization/marlin_utils_fp8.py` (~357 lines) - Marlin FP8

---

## FP8 Configuration

### Fp8Config

```python
class Fp8Config(QuantizationConfig):
    is_checkpoint_fp8_serialized: bool   # Pre-quantized checkpoint
    activation_scheme: str               # "static" or "dynamic"
    weight_block_size: Optional[List[int]]  # [block_n, block_k] or None
    use_mxfp8: bool                      # MXFP8 block-scaled (SM100+)
    ignored_layers: List[str]            # Layers to skip
```

### Minimum Hardware
- Standard FP8: SM80+ (A100)
- MXFP8: SM100+ (Blackwell)

---

## Quantization Modes

### A. Per-Tensor

Single scale per entire weight tensor.

```
Scale shape: (1,)
Usage: Simplest mode, lowest memory overhead
Kernel: torch._scaled_mm or sgl_kernel
```

### B. Per-Channel

One scale per output channel (column).

```
Scale shape: (output_channels,)
Usage: Default for CUTLASS/Marlin backends, better accuracy
Kernel: sgl_kernel.fp8_scaled_mm (CUTLASS)
```

### C. Per-Token-Group (Activations)

Groups of tokens share a scale (typically group_size=128).

```
Scale shape: (num_tokens / group_size,)
Usage: Dynamic activation quantization
Kernel: Per-token-group Triton kernel
```

### D. Block-Wise

2D grid of scales over weight matrix (typically [128, 128] blocks).

```
Scale shape: (N/block_n, K/block_k)
Usage: Higher accuracy for large models (DeepSeek, etc.)
Kernel: DeepGEMM, CUTLASS, FlashInfer, or Triton
```

### E. MXFP8 (Microscale FP8)

Block size fixed at [1, 32] with UE8M0 (unsigned exponent-only) scales.

```
Scale format: uint8 (UE8M0 = 2^exponent, no mantissa)
Block size: [1, 32] elements per scale
Usage: Blackwell (SM100+) with hardware support
Kernel: Triton MXFP8 or FlashInfer TRTLLM
```

---

## Weight Loading Pipeline

### Step 1: Weight Creation

```python
def create_weights(self, layer, input_size_per_partition, ...):
    weight_dtype = torch.float8_e4m3fn if serialized else params_dtype
    
    weight = ModelWeightParameter(
        data=torch.empty(out_size, in_size, dtype=weight_dtype))
    
    if self.block_quant:
        scale = BlockQuantScaleParameter(
            data=scale_init((n_blocks_h, n_blocks_w), dtype=scale_dtype))
        scale.format_ue8m0 = self.use_mxfp8
    else:
        scale = PerTensorScaleParameter(
            data=torch.empty(len(output_partition_sizes), dtype=torch.float32))
```

### Step 2: Post-Loading Processing

**Block-wise path:**
```
1. ROCm FP8FNUZ → normalize_e4m3fn_to_e4m3fnuz()
2. CPU AMX → _amx_process_weight_after_loading()
3. MXFP8 online → _quantize_mxfp8_weights()
4. DeepGEMM UE8M0 → requant_weight_ue8m0_inplace()
```

**Per-tensor/per-channel path:**
```
1. If not serialized:
   - CUTLASS/Marlin: per_token_group_quant_fp8() → per-channel
   - Fallback: input_to_float8() → per-tensor
2. If serialized:
   - CUTLASS/Marlin: convert_to_channelwise()
   - Fallback: requantize_with_max_scale()
```

---

## Kernel Dispatch Hierarchy

### Block-Wise GEMM Backend Selection

```
--fp8-gemm-backend auto (default):
  1. DeepGEMM (if ENABLE_JIT_DEEPGEMM and shape compatible)
  2. FlashInfer TRTLLM (Blackwell + available)
  3. CUTLASS (SM90+, N%128==0, K%128==0)
  4. AITER (AMD with SGLANG_USE_AITER=1)
  5. Triton (universal fallback)
```

### Backend Requirements

| Backend | Hardware | Shape Constraints | Output Type |
|---------|----------|-------------------|-------------|
| DeepGEMM | SM90+ | N%64==0, K%128==0 | bfloat16 |
| FlashInfer TRTLLM | SM100+ (Blackwell) | K>=256, N%64==0 | any |
| FlashInfer DeepGEMM | SM90+ (Hopper) | N%64==0, K%128==0 | bfloat16 |
| CUTLASS | SM90+ | N%128==0, K%128==0 | any |
| AITER | AMD MI300+ | any | bfloat16 |
| Triton | any | none | any |

### Per-Tensor Linear Dispatch

```python
def apply_fp8_linear(input, weight, weight_scale, input_scale, bias,
                     cutlass_fp8_supported, use_per_token_if_dynamic):
    if cutlass_fp8_supported and weight_scale.numel() == weight.shape[1]:
        # Per-channel: CUTLASS fp8_scaled_mm
        if weight.shape[0]%16==0 and weight.shape[1]%16==0:
            return fp8_scaled_mm(qinput, weight, x_scale, weight_scale)
        else:
            return triton_scaled_mm(...)  # Fallback
    else:
        # Per-tensor: torch._scaled_mm
        return torch._scaled_mm(qinput, weight, scale_a=x_scale, scale_b=weight_scale)
```

---

## Scale Format Handling

### E4M3FN vs E4M3FNUZ (AMD)

```python
def normalize_e4m3fn_to_e4m3fnuz(weight, weight_scale, input_scale):
    # ROCm MI300 uses E4M3FNUZ: bit pattern -128 = NaN (not zero)
    weight_as_int8 = weight.view(torch.int8)
    weight_as_int8[weight_as_int8 == -128] = 0  # Fix NaN
    weight = weight_as_int8.view(torch.float8_e4m3fnuz)
    # E4M3FNUZ values are half of E4M3FN → double scales
    weight_scale = weight_scale * 2.0
```

### UE8M0 Format (MXFP8/DeepGEMM)

```python
def ceil_to_ue8m0(x):
    # Unsigned exponent, zero mantissa: 2^ceil(log2(x))
    return torch.pow(2.0, torch.ceil(torch.log2(x.abs())))
```

**Packed UE8M0:**
- 4 uint8 exponents packed into 1 int32
- Shape transformation: `(scale_mn, scale_k)` → `(scale_mn_128x_aligned, scale_k//4)`

---

## Triton Kernels

### Block-Wise Matmul (`_w8a8_block_fp8_matmul`)

```python
@triton.jit
def _w8a8_block_fp8_matmul(A, B, C, As, Bs, M, N, K, group_n, group_k, ...):
    # For each output block (BLOCK_SIZE_M × BLOCK_SIZE_N):
    # accumulator += tl.dot(a, b) * a_s[:, None] * b_s[None, :]
```

### Per-Token-Group Quantization

```python
@triton.jit
def _per_token_group_quant_8bit_colmajor(y_ptr, y_q_ptr, y_s_ptr, group_size, ...):
    # 1. Compute max absolute value per group
    # 2. scale = amax / fp8_max
    # 3. If UE8M0: scale = 2^ceil(log2(scale))
    # 4. y_q = clamp(y / scale, fp8_min, fp8_max)
    # 5. Store scale (column-major for TMA)
```

### Configuration Lookup

Triton kernel configs are tuned per device and stored as JSON:
```
configs/N={N},K={K},device_name={device},dtype=fp8_w8a8,block_shape=[{bn},{bk}].json
```

Default config: `BLOCK_SIZE_M=64, BLOCK_SIZE_N=block_n, BLOCK_SIZE_K=block_k, num_warps=4`

---

## MoE Support

### Per-Expert Weights

```python
class Fp8MoEMethod(FusedMoEMethodBase):
    def create_weights(self, layer, num_experts, ...):
        w13_weight: (num_experts, 2*intermediate, hidden) FP8
        w2_weight: (num_experts, hidden, intermediate) FP8
        
        if block_quant:
            w13_weight_scale_inv: (num_experts, 2*n_blocks_h, n_blocks_w)
            w2_weight_scale_inv: (num_experts, n_blocks_h, n_blocks_w)
        else:
            w13_weight_scale: (num_experts, 2) or (num_experts,)
            w2_weight_scale: (num_experts,)
```

### MoE Processing

For per-tensor MoE: merge w1 and w3 scales, re-quantize with max scale per expert.

### MoE Runner Backends

- FlashInfer TRT-LLM (Blackwell)
- CUTLASS (Hopper+)
- DeepGEMM (DeepSeek-specific)
- Triton (universal)
- AITER (AMD)

---

## Data Flow Example (Block-Wise FP8)

### Weight Loading (Startup)
```
1. Load: layer.weight (N, K) FP8E4M3FN
         layer.weight_scale_inv (N//128, K//128) uint8 (UE8M0)
2. Process: requant_weight_ue8m0_inplace() if needed
3. Set: layer.weight_scale_inv.format_ue8m0 = True
```

### Forward Pass
```
1. Input: (batch, seq_len, hidden) FP16/BF16
2. Flatten: (batch*seq_len, hidden)
3. Quantize activations:
   - Group each 128 elements
   - scale = max(abs(group)) / 448
   - x_q = clamp(x / scale, -448, 448).to(FP8)
4. Select kernel:
   - DeepGEMM: deep_gemm.gemm_nt_f8f8bf16(x_q, scales, W, W_scales, out)
   - Triton: w8a8_block_fp8_matmul_triton(x_q, W, scales, W_scales)
5. Output: (batch*seq_len, hidden) BF16
6. Reshape + bias: (batch, seq_len, hidden)
```

---

## Environment Variables

| Variable | Effect |
|----------|--------|
| `SGLANG_FORCE_FP8_MARLIN` | Force Marlin backend |
| `SGLANG_ENABLE_JIT_DEEPGEMM` | Enable DeepGEMM |
| `SGLANG_SUPPORT_CUTLASS_BLOCK_FP8` | Enable CUTLASS |
| `SGLANG_USE_AITER_FP8_PER_TOKEN` | Per-token quant (AMD) |
| `--fp8-gemm-backend` | Backend selection (server arg) |
