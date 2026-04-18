# Quantization Architecture

## Overview

SGLang implements a plugin-based quantization architecture supporting 25+ quantization methods. The system uses abstract base classes (`QuantizationConfig`, `QuantizeMethodBase`) to provide a clean dispatch mechanism from model configuration detection through kernel execution.

**Key Source Files:**
- `python/sglang/srt/layers/quantization/base_config.py` - Abstract base classes
- `python/sglang/srt/layers/quantization/__init__.py` - Method registry
- `python/sglang/srt/layers/quantization/*.py` - Individual method implementations

---

## Base Classes

### QuantizeMethodBase (ABC)

Root abstraction for all quantization implementations:

```python
class QuantizeMethodBase(ABC):
    create_weights(layer, *weight_args, **extra_weight_attrs)
    apply(layer, *args, **kwargs) -> torch.Tensor  # ABSTRACT
    process_weights_after_loading(layer) -> None
```

### LinearMethodBase (extends QuantizeMethodBase)

Specialized for linear layers:

```python
class LinearMethodBase(QuantizeMethodBase):
    create_weights(layer, input_size_per_partition, output_partition_sizes,
                   input_size, output_size, params_dtype, **extra_weight_attrs)
    apply(layer, x: torch.Tensor, bias: Optional[torch.Tensor]) -> torch.Tensor
```

### FusedMoEMethodBase (extends QuantizeMethodBase)

For Mixture of Experts layers:

```python
class FusedMoEMethodBase(QuantizeMethodBase):
    create_weights(layer, num_experts, hidden_size,
                   intermediate_size_per_partition, params_dtype, **extra_weight_attrs)
    create_moe_runner(layer, moe_runner_config: MoeRunnerConfig)
    apply(layer, dispatch_output: DispatchOutput) -> CombineInput
```

### QuantizationConfig (ABC)

Configuration class determining kernel/method dispatch:

```python
class QuantizationConfig(ABC):
    get_name() -> str                            # "fp8", "gptq", "awq", etc.
    get_supported_act_dtypes() -> List[dtype]    # fp16, bf16, etc.
    get_min_capability() -> int                  # SM70, SM80, etc.
    get_config_filenames() -> List[str]          # Checkpoint detection files
    from_config(config: Dict) -> QuantizationConfig  # Factory method
    get_quant_method(layer, prefix) -> Optional[QuantizeMethodBase]  # KEY DISPATCH
    get_scaled_act_names() -> List[str]          # Activations needing post-scaling
    override_quantization_method(hf_quant_cfg, user_quant) -> Optional[str]
```

---

## Plugin Registration

**Source:** `python/sglang/srt/layers/quantization/__init__.py`

```python
BASE_QUANTIZATION_METHODS: Dict[str, Type[QuantizationConfig]] = {
    "fp8": Fp8Config,
    "mxfp8": Fp8Config,
    "blockwise_int8": BlockInt8Config,
    "modelopt": ModelOptFp8Config,
    "modelopt_fp8": ModelOptFp8Config,
    "modelopt_fp4": ModelOptFp4Config,
    "w8a8_int8": W8A8Int8Config,
    "w8a8_fp8": W8A8Fp8Config,
    "awq": AWQConfig,
    "awq_marlin": AWQMarlinConfig,
    "bitsandbytes": BitsAndBytesConfig,
    "gguf": GGUFConfig,
    "gptq": GPTQConfig,
    "gptq_marlin": GPTQMarlinConfig,
    "moe_wna16": MoeWNA16Config,
    "compressed-tensors": CompressedTensorsConfig,
    "qoq": QoQConfig,
    "w4afp8": W4AFp8Config,
    "petit_nvfp4": PetitNvFp4Config,
    "fbgemm_fp8": FBGEMMFp8Config,
    "quark": QuarkConfig,
    "auto-round": AutoRoundConfig,
    "modelslim": ModelSlimConfig,
    "quark_int4fp8_moe": QuarkInt4Fp8Config,
}

# Platform-specific additions:
if is_cuda() or (_is_mxfp_supported and is_hip()):
    BASE_QUANTIZATION_METHODS["mxfp4"] = Mxfp4Config

def get_quantization_config(quantization: str) -> Type[QuantizationConfig]:
    return QUANTIZATION_METHODS[quantization]
```

---

## Dispatch Pattern

### Level 1: Config → Method Selection

When a layer is initialized, `QuantizationConfig.get_quant_method()` returns the appropriate method:

```python
class Fp8Config(QuantizationConfig):
    def get_quant_method(self, layer, prefix):
        if isinstance(layer, LinearBase):
            if is_layer_skipped(prefix, self.ignored_layers):
                return UnquantizedLinearMethod()
            return Fp8LinearMethod(self)
        elif isinstance(layer, FusedMoE):
            return Fp8MoEMethod(self)
        elif isinstance(layer, RadixAttention):
            return Fp8KVCacheMethod(self)
        return None
```

### Level 2: Method → Kernel Selection

Within each method's `apply()`, the actual compute kernel is selected based on hardware and tensor properties:

```python
class Fp8LinearMethod(LinearMethodBase):
    def apply(self, layer, x, bias):
        if self.use_marlin:
            return apply_fp8_marlin_linear(...)
        if self.use_mxfp8:
            return triton_mxfp8_blockscaled_linear(...)
        if self.block_quant:
            if use_intel_amx_backend(layer):
                return torch.ops.sgl_kernel.fp8_scaled_mm_cpu(...)
            return self.w8a8_block_fp8_linear(...)
        return apply_fp8_linear(...)
```

### Integration Point (LinearBase)

```python
class LinearBase(torch.nn.Module):
    def __init__(self, ..., quant_config=None, prefix=""):
        if quant_config is None:
            self.quant_method = UnquantizedLinearMethod()
        else:
            self.quant_method = quant_config.get_quant_method(self, prefix=prefix)
```

---

## Supported Quantization Methods

### FP8 Family
| Method | Description | Min Capability |
|--------|-------------|----------------|
| `fp8` | Standard FP8 (per-channel/per-tensor/block) | SM80 |
| `mxfp8` | MXFP8 block-scaled with UE8M0 scales | SM100 |
| `w8a8_fp8` | Weight-8 Activation-8 FP8 | SM80 |
| `modelopt_fp8` | ModelOpt FP8 | SM80 |
| `fbgemm_fp8` | FBGEMM FP8 kernel | SM80 |

### INT8 Family
| Method | Description | Min Capability |
|--------|-------------|----------------|
| `w8a8_int8` | Symmetric INT8 | SM70 |
| `blockwise_int8` | Block-wise INT8 | SM70 |

### INT4 Family
| Method | Description | Min Capability |
|--------|-------------|----------------|
| `awq` | AWQ 4-bit (Triton/CUDA dequant) | SM75 |
| `awq_marlin` | AWQ with Marlin kernel | SM80 |
| `gptq` | GPTQ 2/3/4/8-bit | SM60 |
| `gptq_marlin` | GPTQ with Marlin kernel | SM80 |
| `w4afp8` | Weight-4 Activation-FP8 | SM80 |
| `moe_wna16` | MoE Weights-N Activations-16 | SM80 |

### FP4 Family
| Method | Description | Min Capability |
|--------|-------------|----------------|
| `modelopt_fp4` | ModelOpt FP4 | SM89 |
| `petit_nvfp4` | PetitLM NV FP4 | SM89 |
| `mxfp4` | MXFP4 block-scaled | SM80 (CUDA/HIP) |

### Framework Integrations
| Method | Description |
|--------|-------------|
| `compressed-tensors` | CompressedTensors format |
| `bitsandbytes` | BnB 4/8-bit |
| `gguf` | GGUF file format |
| `auto-round` | AutoRound |
| `modelslim` | ModelSlim |
| `quark` | Quark (W4A4 MXFP4, W8A8 FP8) |
| `quark_int4fp8_moe` | Quark INT4-FP8 MoE |
| `qoq` | QoQ quantization |

---

## Configuration Flow

```
Model Loading
    |
    v
Detect quant format from checkpoint/config
    |
    v
QuantizationConfig.from_config(hf_quant_config)
    |
    v
Create Config instance (e.g., Fp8Config)
    |
    v
For each layer:
    |-- quant_config.get_quant_method(layer, prefix)
    |-- Returns QuantizeMethodBase subclass
    |
    v
create_weights(): Initialize weight tensors (correct dtype/shape)
    |
    v
Load checkpoint → process_weights_after_loading(): Post-process
    |
    v
Forward pass → apply(): Select and call kernel
```

---

## Key Design Patterns

1. **Config determines method type**: Each Config subclass dispatches different methods based on layer type (Linear, MoE, KVCache)
2. **Lazy kernel selection**: Config → Method at init time, actual kernel at inference time
3. **Layer type dispatch**: Different quantization for LinearBase, FusedMoE, RadixAttention
4. **Override and auto-detection**: `override_quantization_method()` supports format variants
5. **Platform abstraction**: Same config works across CUDA, HIP, NPU, CPU

---

## File Structure

```
python/sglang/srt/layers/quantization/
├── base_config.py                    # Abstract bases
├── __init__.py                       # Registry
├── fp8.py                           # FP8 config/methods
├── fp8_kernel.py                    # FP8 Triton kernels
├── fp8_utils.py                     # FP8 dispatch utilities
├── w8a8_fp8.py / w8a8_int8.py       # W8A8 variants
├── blockwise_int8.py                # Block INT8
├── awq.py / gptq.py                 # INT4 methods
├── marlin_utils.py / marlin_utils_fp8.py  # Marlin integration
├── awq_triton.py                    # AWQ Triton fallback
├── mxfp4.py                         # MXFP4
├── modelopt_quant.py                # ModelOpt
├── bitsandbytes.py / gguf.py        # Third-party
├── petit.py / fbgemm_fp8.py         # Specialized
├── kv_cache.py                      # KV cache quantization
├── unquant.py                       # Unquantized baseline
├── utils.py                         # Shared utilities
├── configs/                         # 159+ device-specific tuning JSONs
└── compressed_tensors/ / quark/ / modelslim/  # Framework dirs
```
