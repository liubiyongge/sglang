# SGLang Model Registry and Unified Model Interface Architecture Analysis

## Executive Summary

SGLang implements a sophisticated model registry and unified interface pattern that enables dynamic model discovery, transparent weight loading, and seamless integration of diverse model architectures with varying complexity levels. The system accommodates 150+ models across 167 Python files, spanning 89,271 lines of code, with support for 34 different quantization methods and special handling for MoE (Mixture of Experts) architectures.

The architecture achieves polymorphic model support through a convention-based discovery system using Python introspection, standardized forward interfaces, and pluggable weight loading strategies.

---

## 1. Registry Architecture and Core Components

### 1.1 `_ModelRegistry` Dataclass

The foundation of SGLang's model system is defined in `python/sglang/srt/models/registry.py`:

```python
@dataclass
class _ModelRegistry:
    # Keyed by model_arch
    models: Dict[str, Union[Type[nn.Module], str]] = field(default_factory=dict)

    def register(self, package_name: str, overwrite: bool = False, strict: bool = False):
        new_models = import_model_classes(package_name, strict=strict)
        if overwrite:
            self.models.update(new_models)
        else:
            for arch, cls in new_models.items():
                if arch in self.models:
                    raise ValueError(f"Model architecture {arch} already registered.")
                self.models[arch] = cls

    def get_supported_archs(self) -> AbstractSet[str]:
        return self.models.keys()

    def _normalize_archs(self, architectures: Union[str, List[str]]) -> List[str]:
        if isinstance(architectures, str):
            architectures = [architectures]
        normalized_arch = list(filter(lambda model: model in self.models, architectures))
        # Transformers backend as fallback
        if len(normalized_arch) != len(architectures):
            normalized_arch.append("TransformersForCausalLM")
        return normalized_arch

    def resolve_model_cls(self, architectures: Union[str, List[str]]) -> Tuple[Type[nn.Module], str]:
        architectures = self._normalize_archs(architectures)
        for arch in architectures:
            model_cls = self._try_load_model_cls(arch)
            if model_cls is not None:
                return (model_cls, arch)
        return self._raise_for_unsupported(architectures)
```

**Key Design Patterns:**

- **Lazy Registration**: Models are discovered and registered only when the registry is initialized
- **Fallback Strategy**: Automatically falls back to the Transformers backend if no native implementation exists
- **Conflict Resolution**: Duplicate registrations are detected and raise errors unless explicitly overwritten
- **Dictionary-Based Lookup**: O(1) model architecture resolution

### 1.2 Singleton Pattern with External Package Support

```python
ModelRegistry = _ModelRegistry()
ModelRegistry.register("sglang.srt.models")

if external_pkg := envs.SGLANG_EXTERNAL_MODEL_PACKAGE.get():
    ModelRegistry.register(external_pkg, overwrite=True)
```

The external package support enables users to:
- Register custom model implementations from external packages
- Override built-in implementations with custom versions
- Maintain complete separation between built-in and user-defined models

---

## 2. Dynamic Model Discovery via Python Introspection

### 2.1 The `import_model_classes()` Function

```python
@lru_cache()
def import_model_classes(package_name: str, strict: bool = False):
    model_arch_name_to_cls = {}
    package = importlib.import_module(package_name)
    for _, name, ispkg in pkgutil.iter_modules(package.__path__, package_name + "."):
        if not ispkg:
            if name.split(".")[-1] in envs.SGLANG_DISABLED_MODEL_ARCHS.get():
                continue
            try:
                module = importlib.import_module(name)
            except Exception as e:
                if strict:
                    raise
                logger.warning(f"Ignore import error when loading {name}: {e}")
                continue
            if hasattr(module, "EntryClass"):
                entry = module.EntryClass
                if isinstance(entry, list):
                    for tmp in entry:
                        assert tmp.__name__ not in model_arch_name_to_cls
                        model_arch_name_to_cls[tmp.__name__] = tmp
                else:
                    assert entry.__name__ not in model_arch_name_to_cls
                    model_arch_name_to_cls[entry.__name__] = entry
    return model_arch_name_to_cls
```

**Discovery Mechanism:**

1. **Module Iteration**: Uses `pkgutil.iter_modules()` to discover all modules without explicit listing
2. **Selective Disabling**: Supports disabling specific architectures via `SGLANG_DISABLED_MODEL_ARCHS`
3. **Graceful Degradation**: Logs warnings for import errors instead of crashing
4. **Multiple Architectures Per Module**: A single module can export multiple model classes via a list in `EntryClass`
5. **Duplicate Detection**: Prevents accidental registration of duplicate architectures

### 2.2 EntryClass Convention

Each model module must define an `EntryClass` attribute:

**Single Class** (most common):
```python
# In gemma.py
class GemmaForCausalLM(nn.Module):
    ...
EntryClass = GemmaForCausalLM
```

**Multiple Classes** (for architecture variants):
```python
# In llama.py
class LlamaForCausalLM(nn.Module): ...
class Phi3ForCausalLM(LlamaForCausalLM): ...
class InternLM3ForCausalLM(LlamaForCausalLM): ...

EntryClass = [LlamaForCausalLM, Phi3ForCausalLM, InternLM3ForCausalLM]
```

**Complex MoE Architectures**:
```python
# In deepseek_v2.py
class DeepseekV2ForCausalLM(nn.Module, DeepseekV2WeightLoaderMixin): ...
class DeepseekV3ForCausalLM(DeepseekV2ForCausalLM): ...
class DeepseekV32ForCausalLM(DeepseekV2ForCausalLM): ...

EntryClass = [DeepseekV2ForCausalLM, DeepseekV3ForCausalLM, DeepseekV32ForCausalLM]
```

---

## 3. Unified Forward Interface Pattern

### 3.1 Core Signature

All model implementations standardize on a consistent forward signature:

```python
@torch.no_grad()
def forward(
    self,
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    forward_batch: ForwardBatch,
    input_embeds: torch.Tensor = None,
) -> torch.Tensor:
    """
    Args:
        input_ids: [num_tokens] tensor of token indices
        positions: [num_tokens] tensor of position indices
        forward_batch: Batch-level metadata, KV cache locations, distributed info
        input_embeds: Optional pre-computed embeddings (for multimodal models)

    Returns:
        Logits or hidden states for downstream processing
    """
```

### 3.2 ForwardBatch: Unified Batch Information Container

```python
@dataclass
class ForwardBatch:
    input_ids: torch.Tensor           # [num_tokens]
    positions: torch.Tensor           # [num_tokens]
    seq_lens_cpu: torch.Tensor        # [batch_size] CPU tensor
    out_cache_loc: torch.Tensor       # [num_tokens] indices for KV cache writes
    token_to_kv_pool: KVCache         # Memory pool for KV caches
    forward_mode: ForwardMode         # EXTEND, DECODE, or MIXED
    attention_backend: AttentionBackend
    multimodal_inputs: Optional[MultimodalInputs]
    pp_proxy_tensors: Optional[PPProxyTensors]
```

### 3.3 Simple Model Example: Gemma

```python
class GemmaForCausalLM(nn.Module):
    def __init__(self, config, quant_config=None, prefix=""):
        super().__init__()
        self.config = config
        self.model = GemmaModel(config, quant_config=quant_config, prefix=add_prefix("model", prefix))
        self.logits_processor = LogitsProcessor(config)

    @torch.no_grad()
    def forward(self, input_ids, positions, forward_batch, input_embeds=None):
        hidden_states = self.model(input_ids, positions, forward_batch, input_embeds)
        return self.logits_processor(input_ids, hidden_states, self.model.embed_tokens, forward_batch)
```

### 3.4 Complex Model Example: DeepSeek V2 (MoE + Distributed)

```python
class DeepseekV2ForCausalLM(nn.Module, DeepseekV2WeightLoaderMixin):
    def forward(self, input_ids, positions, forward_batch, input_embeds=None, pp_proxy_tensors=None):
        # Handle context parallelism for NSA
        if self.nsa_enable_prefill_cp:
            if can_cp_split(len(input_ids), self.cp_size, self.use_nsa, forward_batch):
                forward_batch.nsa_cp_metadata = prepare_input_dp_with_cp_dsa(...)

        # Distributed forward with tensor model parallel
        with get_attn_tp_context().maybe_input_scattered(forward_batch):
            hidden_states = self.model(input_ids, positions, forward_batch, input_embeds, pp_proxy_tensors)

        # Pipeline parallelism: only last rank computes logits
        if self.pp_group.is_last_rank:
            return self.logits_processor(input_ids, hidden_states, self.lm_head, forward_batch)
        else:
            return hidden_states
```

---

## 4. Weight Loading System

### 4.1 Stacked Parameter Mapping

SGLang models combine multiple weight matrices from HuggingFace into single "stacked" parameters:

```python
def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
    stacked_params_mapping = [
        # (param_name, shard_name, shard_id)
        ("qkv_proj", "q_proj", "q"),
        ("qkv_proj", "k_proj", "k"),
        ("qkv_proj", "v_proj", "v"),
        ("gate_up_proj", "gate_proj", 0),
        ("gate_up_proj", "up_proj", 1),
    ]

    params_dict = dict(self.named_parameters())
    for name, loaded_weight in weights:
        for param_name, shard_name, shard_id in stacked_params_mapping:
            if shard_name not in name:
                continue
            name = name.replace(shard_name, param_name)
            if name not in params_dict:
                continue
            param = params_dict[name]
            weight_loader = param.weight_loader
            weight_loader(param, loaded_weight, shard_id)
            break
        else:
            if name not in params_dict:
                continue
            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)
```

**Loading Process:**

1. HuggingFace checkpoint has separate `q_proj`, `k_proj`, `v_proj` matrices
2. SGLang model has a single `qkv_proj` parameter combining all three
3. Weight loader concatenates the three matrices during loading
4. Shard ID specifies concatenation position

### 4.2 Per-Parameter Loaders

Custom weight loaders attached to individual parameters:

```python
class ColumnParallelLinear(nn.Module):
    def __init__(self, ...):
        self.weight = nn.Parameter(torch.empty(...))
        self.weight.weight_loader = stacked_weight_loader  # Custom loader
```

**Common Loader Types:**
- Default Loader: Direct tensor assignment
- Stacked Weight Loader: Handles parameter concatenation
- Quantization-Aware Loader: Handles quantized weights (FP8, INT4, etc.)
- Scale Loader: Loads quantization scales separately

### 4.3 Advanced Weight Loading: DeepSeek V2

Uses `DeepseekV2WeightLoaderMixin` for:
- MoE expert parameter slicing across multiple GPUs
- Complex attention backend-specific weight layouts
- FP8 quantization scale loading
- Distributed expert parallelism (EP) parameter mapping

---

## 5. Configuration to Implementation Mapping

### 5.1 `get_model_architecture()` Function

```python
def get_model_architecture(model_config: ModelConfig) -> Tuple[Type[nn.Module], str]:
    architectures = getattr(model_config.hf_config, "architectures", [])

    # Special handling for quantized Mixtral
    if model_config.quantization and model_config.quantization not in mixtral_supported and "MixtralForCausalLM" in architectures:
        architectures = ["QuantMixtralForCausalLM"]

    supported_archs = ModelRegistry.get_supported_archs()
    is_native_supported = any(arch in supported_archs for arch in architectures)

    if not is_native_supported or model_config.model_impl == ModelImpl.TRANSFORMERS:
        architectures = resolve_transformers_arch(model_config, architectures)

    return ModelRegistry.resolve_model_cls(architectures)
```

**Resolution Strategy:**
1. Read HuggingFace Config: Extract `architectures` field
2. Check Quantization: Special case certain quantized models
3. Check Native Support: Look for architecture in registry
4. Fallback to Transformers: If no native implementation, use Transformers backend
5. Return Model Class and architecture name

---

## 6. Quantization as an Orthogonal Concern

### 6.1 Integration Pattern

Quantization is applied through configuration rather than separate model classes:

```python
class GemmaForCausalLM(nn.Module):
    def __init__(self, config, quant_config=None, prefix=""):
        super().__init__()
        self.model = GemmaModel(config, quant_config=quant_config, prefix=add_prefix("model", prefix))
```

### 6.2 Supported Quantization Methods (34 total)

| Category | Methods |
|----------|---------|
| Float | FP8, MXFP8, MXFP4, NvFP4 |
| Integer | INT4, INT8, W8A8, W4AFP8 |
| Post-Training | GPTQ, AWQ, Marlin, AutoRound |
| Framework | BitsAndBytes, Compressed Tensors |
| Specialized | ModelOpt, Quark, Petit, QoQ |

---

## 7. Modular Layer Hierarchy

### Gemma Example

```
GemmaForCausalLM (top-level inference class)
+-- GemmaModel (transformer model)
|   +-- embed_tokens (VocabParallelEmbedding)
|   +-- layers (ModuleList of GemmaDecoderLayer)
|   |   +-- GemmaDecoderLayer
|   |       +-- self_attn (GemmaAttention)
|   |       |   +-- qkv_proj (QKVParallelLinear)
|   |       |   +-- o_proj (RowParallelLinear)
|   |       |   +-- rotary_emb (RotaryEmbedding)
|   |       |   +-- attn (RadixAttention)
|   |       +-- mlp (GemmaMLP)
|   |           +-- gate_up_proj (MergedColumnParallelLinear)
|   |           +-- down_proj (RowParallelLinear)
|   |           +-- act_fn (GeluAndMul)
|   +-- norm (RMSNorm)
+-- logits_processor (LogitsProcessor)
```

---

## 8. Distributed System Integration

### 8.1 Tensor Model Parallelism (TP)

All models handle TP through distributed linear layers:

```python
class QKVParallelLinear(nn.Module):
    """Partitions Q, K, V across multiple GPUs."""
    def __init__(self, input_size, head_dim, num_heads, num_kv_heads, ...):
        # Each GPU gets num_heads // tp_size query heads
        pass
```

### 8.2 Pipeline Parallelism (PP)

Models support layer partitioning:

```python
class DeepseekV2ForCausalLM(nn.Module):
    @property
    def start_layer(self):
        return self.model.start_layer

    @property
    def end_layer(self):
        return self.model.end_layer
```

### 8.3 Expert Parallelism (EP) for MoE

```python
class Qwen2MoeForCausalLM(nn.Module):
    def __init__(self, config):
        self.ep_size = get_moe_expert_parallel_world_size()
        self.ep_rank = get_moe_expert_parallel_rank()
        self.router = TopK(config.num_experts // self.ep_size, config.num_selected_experts)
```

### 8.4 Attention for Distributed Systems

```python
class GemmaAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, num_kv_heads, ...):
        tp_size = get_tensor_model_parallel_world_size()
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        if self.total_num_kv_heads >= tp_size:
            assert self.total_num_kv_heads % tp_size == 0
        else:
            assert tp_size % self.total_num_kv_heads == 0
```

---

## 9. Architecture Diagram

```
+---------------------------------------------------------------+
|                    Model Loading Pipeline                       |
+---------------------------------------------------------------+
|                                                                |
|  HuggingFace Model Config                                      |
|  +-- architectures: ["GemmaForCausalLM"]                      |
|         |                                                      |
|         v                                                      |
|  get_model_architecture(model_config)                          |
|  +-- Extract architectures from config                        |
|  +-- Query ModelRegistry.get_supported_archs()                |
|  +-- If native: Use SGLang implementation                     |
|  +-- Else: Fall back to TransformersForCausalLM               |
|         |                                                      |
|         v                                                      |
|  ModelRegistry.resolve_model_cls(architectures)                |
|  +-- Return (ModelClass, arch_name)                           |
|         |                                                      |
|         v                                                      |
|  model = ModelClass(config, quant_config)                     |
|  model.load_weights(weights_from_checkpoint)                  |
|         |                                                      |
|         v                                                      |
|  output = model.forward(input_ids, positions, forward_batch)  |
+---------------------------------------------------------------+

+---------------------------------------------------------------+
|              Dynamic Model Discovery at Startup                 |
+---------------------------------------------------------------+
|                                                                |
|  import_model_classes("sglang.srt.models")                    |
|  +-- Iterate all Python files in models/                      |
|  +-- For each file:                                           |
|  |   +-- importlib.import_module(file)                        |
|  |   +-- Check if hasattr(module, "EntryClass")              |
|  |   +-- EntryClass can be single class or list              |
|  +-- Return dict[arch_name] = model_class                    |
|                                                                |
|  ModelRegistry.register(mapping)                              |
|  +-- Store in _ModelRegistry.models dict                     |
|                                                                |
|  Optional: Load external package models                       |
|  if SGLANG_EXTERNAL_MODEL_PACKAGE env var set                |
|  +-- register(..., overwrite=True)                           |
+---------------------------------------------------------------+
```

---

## 10. Statistics and Metrics

| Metric | Count |
|--------|-------|
| Total Python Model Files | 167 |
| Files with EntryClass | 151 |
| Files with Multiple EntryClass | 73 |
| Total ForCausalLM Classes | 121 |
| MoE Model Classes | 50 |
| Total Lines of Code (models/) | 89,271 |
| Average File Size | 535 lines |
| Quantization Methods | 34 |

### Popular Model Families

| Family | Count | Examples |
|--------|-------|----------|
| LLaMA | 4+ | LlamaForCausalLM, Phi3, InternLM3 |
| DeepSeek | 10+ | V1, V2, V3, NextN, Janus |
| Qwen | 8+ | Qwen2, Qwen3, MoE variants |
| Gemma | 5+ | Gemma, Gemma2, Gemma3 |
| GLM | 6+ | GLM4, GLM4-MoE, GLM4-Vision |
| Others | 100+ | Mistral, Mixtral, Llava, MLLAMA |

---

## 11. Extension Guide: Adding a New Model

### Step 1: Create Model File

Create `python/sglang/srt/models/my_new_model.py`:

```python
from sglang.srt.layers.linear import QKVParallelLinear, RowParallelLinear, MergedColumnParallelLinear
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.model_executor.forward_batch_info import ForwardBatch

class MyNewModelForCausalLM(nn.Module):
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    def __init__(self, config, quant_config=None, prefix=""):
        super().__init__()
        self.model = MyNewModelBase(config, quant_config=quant_config, prefix=add_prefix("model", prefix))
        self.logits_processor = LogitsProcessor(config)

    @torch.no_grad()
    def forward(self, input_ids, positions, forward_batch, input_embeds=None):
        hidden_states = self.model(input_ids, positions, forward_batch, input_embeds)
        return self.logits_processor(input_ids, hidden_states, self.model.embed_tokens, forward_batch)

    def load_weights(self, weights):
        stacked_params_mapping = [
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        # ... standard weight loading ...

EntryClass = MyNewModelForCausalLM
```

### Step 2: Discovery is Automatic

Once the model file is in place with `EntryClass` defined:
1. `import_model_classes("sglang.srt.models")` automatically discovers it
2. Model is registered in `ModelRegistry`
3. `get_model_architecture()` can resolve it from HuggingFace config
4. Ready to use immediately without any configuration changes

---

## 12. Key Design Principles

1. **Convention Over Configuration**: EntryClass discovery eliminates boilerplate
2. **Orthogonal Concerns**: Quantization is separate from architecture
3. **Composition Over Inheritance**: Use mixins for weight loading
4. **Polymorphic Dispatch**: Single interface for 150+ architectures
5. **Graceful Degradation**: Fall back to Transformers if native not available
6. **Lazy Discovery**: Models are discovered only when registry initializes

---

## Conclusion

SGLang's model registry achieves remarkable flexibility while maintaining clean, maintainable code. Adding a new model requires only:
1. Creating one Python file
2. Defining `EntryClass` pointing to model class
3. Implementing standard interfaces: `forward()` and `load_weights()`

The registry automatically discovers and registers the model with zero configuration overhead. This enables rapid prototyping and scalable model integration across the entire SGLang ecosystem.
