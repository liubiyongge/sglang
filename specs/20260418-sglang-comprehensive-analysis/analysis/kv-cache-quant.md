# KV Cache Quantization

## Overview

SGLang supports FP8 and FP4 quantization of the KV cache to reduce memory footprint during inference. The `BaseKVCacheMethod` interface handles per-tensor scaling loaded from checkpoints, while FP4 uses dynamic block-based quantization.

**Key Source Files:**
- `python/sglang/srt/layers/quantization/kv_cache.py` - BaseKVCacheMethod interface
- `python/sglang/srt/mem_cache/memory_pool.py` - FP4/FP8 KV pool implementations
- Attention backend files (FlashInfer, TRTLLM) - Scale application

---

## Interface (BaseKVCacheMethod)

```python
class BaseKVCacheMethod(QuantizeMethodBase):
    create_weights(layer)  # Creates k_scale, v_scale parameters
    apply()                # Raises RuntimeError (not used directly)
    process_weights_after_loading()  # Processes scales from checkpoint
```

**Parameters added to attention layers:**
- `layer.k_scale` - float32 per-tensor scale for keys (init: -1.0)
- `layer.v_scale` - float32 per-tensor scale for values (init: -1.0)
- `layer.k_scale_float` - Python float version for kernel calls
- `layer.v_scale_float` - Python float version for kernel calls

---

## Supported Formats

| Format | Precision | Scale Type | Dynamic Range |
|--------|-----------|-----------|---------------|
| FP8 E4M3 | 4 exp, 3 mantissa | Per-tensor scalar | ±240 |
| FP8 E5M2 | 5 exp, 2 mantissa | Per-tensor scalar | ±57,344 |
| FP4 E2M1 | 2 exp, 1 mantissa | Per-block (16 elements) | Block-scaled |

---

## Scale Computation and Application

### Scale Loading Logic

```python
def process_weights_after_loading(self, layer):
    # Prefer separate K/V scales if both valid (> 0)
    if k_scale > 0 and v_scale > 0:
        layer.k_scale_float = k_scale
        layer.v_scale_float = v_scale
    # Single KV scale fallback
    elif k_scale > 0 or v_scale > 0:
        layer.k_scale_float = layer.v_scale_float = max(k_scale, v_scale)
    # Default: no quantization
    else:
        layer.k_scale_float = layer.v_scale_float = 1.0
    
    # FP8FNUZ (AMD): multiply scales by 2
    if _is_fp8_fnuz:
        layer.k_scale_float *= 2.0
        layer.v_scale_float *= 2.0
```

### Scale Application in Attention

**FlashInfer backend:**
```python
prefill_wrapper_paged.forward(..., k_scale=layer.k_scale_float, v_scale=layer.v_scale_float)
```

**TRTLLM MHA backend:**
```python
bmm1_scale = q_scale * k_scale * softmax_scale  # Composite scale
```

**Memory pool write (set_kv_buffer):**
```python
if cache_k.dtype != self.dtype:
    cache_k.div_(k_scale)  # De-scale before quantization
    cache_k = cache_k.to(self.dtype)  # Cast to FP8
```

---

## FP4 KV Cache Implementation

### Memory Pool (MHATokenToKVPoolFP4)

Allocates per layer:
- `k_buffer`: uint8, shape `[size, heads, head_dim//2]` (2 FP4 per byte)
- `v_buffer`: uint8, shape `[size, heads, head_dim//2]`
- `k_scale_buffer`: uint8, shape `[size, (heads*head_dim)//16]` (UE8M0)
- `v_scale_buffer`: uint8, shape `[size, (heads*head_dim)//16]`

### Quantization/Dequantization

```python
class KVFP4QuantizeUtil:
    @staticmethod
    def batched_quantize(tensor):
        # tensor: [B, M, N] → quant: [B, M, N/2], scales: [B, M*N/16]
        # Block size: 16 elements per scale
        # Scale format: UE8M0 (unsigned exponent only)
    
    @staticmethod
    def batched_dequantize(quant_tensor, scale_factors):
        # Reverse: quant [B, M, N/2] + scales → [B, M, N] float
```

### FP4 Values (E2M1)
```
Representable: [0, 0.5, 1, 1.5, 2, 3, 4, 6]
Boundaries:    [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5]
```

---

## Integration with Attention Backends

| Backend | FP8 | FP4 | Notes |
|---------|-----|-----|-------|
| FlashInfer | Y | Y | Passes k/v scale to paged attention |
| TRTLLM MHA | Y | - | Fused FP8 quantize+write kernel |
| TRTLLM MLA | Y | Y | De-quantizes before projection |
| Triton | Y | - | FP8 paged cache write kernel |

---

## Key Design Patterns

1. **Per-tensor only**: KV cache uses scalar scales (not per-channel/block for FP8)
2. **Checkpoint-driven**: Scales from model checkpoints, default 1.0 if absent
3. **Lazy quantization**: FP4 scales computed dynamically on cache write
4. **Scale composition**: Effective scale = `q_scale × k_scale × attention_scale`
5. **3-4× compression**: FP4 provides ~4× memory reduction for KV cache
