# Diffusion JIT Kernels (CuTe DSL)

## Overview

SGLang provides **fused CUDA kernels** for diffusion transformer operations using NVIDIA's CuTe DSL (from CUTLASS). These kernels eliminate memory-bandwidth bottlenecks by fusing multiple operations (norm, scale, shift, residual) into single GPU kernel launches.

**File**: `python/sglang/jit_kernel/diffusion/cutedsl/`

---

## Architecture

### CuTe DSL Compilation Pipeline

```
Python (torch.library.custom_op) 
  → CuTe DSL JIT (@cute.jit / @cute.kernel)
    → CUTLASS Codegen
      → PTX/CUBIN (cached per compile-time signature)
```

### Compilation Cache Strategy

The kernels use a **lazy JIT** approach with hash-based caching:

```python
_COMPILE_CACHE = {}

def make_hash_key(cls, *inputs):
    """Hash = (dtype, ndim, last_dim) for each tensor."""
    def _sig(val):
        if isinstance(val, torch.Tensor):
            return (val.dtype, val.ndim, val.shape[-1])
        return val
    return tuple(_sig(val) for val in inputs)

# Only compile once per unique shape signature
hash_key = ScaleResidualNormScaleShift.make_hash_key(norm_type, *tensors)
compiled_fn = _COMPILE_CACHE.get(hash_key)
if compiled_fn is None:
    kernel = ScaleResidualNormScaleShift(D, norm_type)
    fake_sig_args = [to_fake_cute_args(t) for t in tensors]
    compiled_fn = cute.compile(kernel, *fake_sig_args, options="--enable-tvm-ffi")
    _COMPILE_CACHE[hash_key] = compiled_fn
```

The `to_fake_cute_args` function creates symbolic tensors with only the last dimension fixed, maximizing kernel reuse across different batch/sequence sizes.

---

## Kernel: `ScaleResidualNormScaleShift`

**File**: `cutedsl/scale_residual_norm_scale_shift.py`

### Purpose

Fuses up to 5 operations in one kernel launch:

```
residual_out = residual + gate * x
y = norm(residual_out, weight, bias) * (1 + scale) + shift
```

This pattern appears in every DiT block's adaptive layer norm (AdaLN) and is called `2 × num_dit_blocks × num_inference_steps` times per generation.

### Exposed Operations

Two `torch.library.custom_op` variants:

#### 1. `sglang::fused_norm_scale_shift`

```python
y = norm(x) * (1 + scale) + shift
```

- No residual connection
- Used for the first block in the DiT

#### 2. `sglang::fused_scale_residual_norm_scale_shift`

```python
residual_out = residual + gate * x
y = norm(residual_out) * (1 + scale) + shift
```

- Full fused operation with residual
- Used in all subsequent DiT blocks

### Supported Tensor Shapes

| Parameter | Accepted Shapes |
|-----------|----------------|
| `x, residual` | `[B, S, D]` — strict |
| `weight, bias` | `None` or `[D]` |
| `gate` | `None`, `[1]`, `[D]`, `[B, D]`, `[B, S, D]`, `[B, F, 1, D]` |
| `scale, shift` | `[1]`, `[D]`, `[B, D]`, `[B, S, D]`, `[B, F, 1, D]` |

The `[B, F, 1, D]` shape handles video frame-wise conditioning where each temporal frame has its own scale/shift.

### Constraints

- **D must be a multiple of 256 and ≤ 8192**: Required for vectorized LDG.128 loads without predication
- **Contiguous on last dimension**: `tensor.stride()[-1] == 1` required
- **Norm types**: `"layer"` (LayerNorm) or `"rms"` (RMSNorm)

### Kernel Implementation Details

```python
class ScaleResidualNormScaleShift:
    def __init__(self, D: int, norm_type: str):
        self.D = D
        self.num_warps = D // 256        # warps per CTA
        self.num_threads = self.num_warps * WARP_SIZE  # 32 threads/warp
    
    @cute.jit
    def __call__(self, mY, mResOut, mRes, mX, mGate, mWeight, mBias, mScale, mShift, eps, stream):
        B, S, _ = mX.shape
        self.kernel(...).launch(
            grid=[B * S, 1, 1],          # one CTA per (batch, seq) element
            block=[self.num_threads, 1, 1],
            stream=stream,
        )
```

- **Grid**: `B * S` CTAs — one CTA per row of length D
- **Vectorization**: `num_vectorized = 8` elements per copy (128-bit loads)
- **Data flow**: Global → Register → Compute → Register → Global (no shared memory)
- **None handling**: Scalar placeholders (1 or 0) used for None tensors; CuTe DSL const_expr eliminates unused branches at compile time

### Norm Computation

```python
@cute.jit
def apply_norm_cta(norm_type, num_warps, tidx, x, weight, bias, D, eps):
    if norm_type == "layer":
        # 1. Warp-level reduction for mean
        # 2. Warp-level reduction for variance
        # 3. Normalize: (x - mean) / sqrt(var + eps)
        # 4. Affine: normalized * weight + bias
    elif norm_type == "rms":
        # 1. Warp-level reduction for sum-of-squares
        # 2. Normalize: x / sqrt(mean_sq + eps)
        # 3. Scale: normalized * weight
```

---

## Kernel: Fused Attention Scale

Additional kernels exist for:
- **RoPE (Rotary Position Embedding)** application
- **Fused QKV projection** with bias
- **Activation functions** (SiLU, GELU with scale)

These follow the same CuTe DSL pattern but are less complex.

---

## torch.compile Integration

The fused kernels register fake implementations for `torch.compile`:

```python
@fused_norm_scale_shift.register_fake
def _fused_norm_scale_shift_fake(x, weight, bias, scale, shift, norm_type, eps):
    return x.new_empty(x.shape)

@fused_scale_residual_norm_scale_shift.register_fake
def _fake(residual, x, gate, weight, bias, scale, shift, norm_type, eps):
    return x.new_empty(x.shape), x.new_empty(x.shape)
```

This enables the kernels to participate in `torch.compile` graphs without graph breaks.

---

## Performance Impact

These fused kernels eliminate memory round-trips for operations that would otherwise require separate kernel launches:

| Unfused (5 separate kernels) | Fused (1 kernel) |
|------------------------------|------------------|
| `gate_out = gate * x` → Write to GMEM | Single kernel: |
| `res_out = residual + gate_out` → Write to GMEM | Read all inputs once |
| `normed = layernorm(res_out, w, b)` → Write to GMEM | Compute in registers |
| `scaled = normed * (1 + scale)` → Write to GMEM | Write final output |
| `y = scaled + shift` → Write to GMEM | **1 read + 1 write** |
| **5 reads + 5 writes** | |

For a typical WAN-2.1 model with D=3072, this reduces memory traffic by ~5x for AdaLN operations.

---

## Source References

- Scale/Residual/Norm/ScaleShift: `cutedsl/scale_residual_norm_scale_shift.py`
- Common Norm Fusion: `cutedsl/common/norm_fusion.py`
- Utils: `cutedsl/utils.py`
- Kernel entry points: `python/sglang/jit_kernel/diffusion/`
