# LoRA (Low-Rank Adaptation) Serving

## Overview

SGLang supports concurrent serving of multiple LoRA adapters with efficient GPU memory management, inspired by **S-LoRA** (Serving Thousands of Concurrent LoRA Adapters) and **Punica** (Multi-Tenant LoRA Serving). The system allows per-request adapter selection with minimal overhead.

**File**: `python/sglang/srt/lora/`

---

## Architecture

```
Request with lora_path
    ↓
Scheduler → LoRA Manager
    ├── Check if adapter is loaded (lora_registry)
    ├── If not: load from disk → LoRA Memory Pool
    ├── Prepare batch-level LoRA metadata
    └── Execute with LoRA backend kernels
```

---

## LoRAManager

**File**: `lora/lora_manager.py`

Central orchestrator for LoRA serving:

```python
class LoRAManager:
    def __init__(self, base_model, base_hf_config, max_loras_per_batch, ...):
        self.base_model: torch.nn.Module       # Original model weights
        self.max_loras_per_batch: int           # Max concurrent adapters
        self.lora_backend: BaseLoRABackend      # Kernel backend (triton/torch/chunked)
        self.mem_pool: LoRAMemoryPool           # GPU memory pool for adapter weights
        self.lora_registry: dict                # Loaded adapter registry
        self.eviction_policy: str               # LRU/FIFO for memory management
```

### Key Operations

| Operation | Method | Description |
|-----------|--------|-------------|
| **Load** | `init_state()`, dynamic loading | Load adapter weights from HF/local path |
| **Prepare Batch** | `prepare_lora_batch()` | Set up per-request adapter pointers for forward pass |
| **Forward** | via `BaseLayerWithLoRA` | Fused base + LoRA computation |
| **Evict** | `mem_pool.evict()` | Free GPU memory when pool full |
| **CUDA Graph** | `init_cuda_graph_batch_info()` | Pre-allocate for CUDA graph compatibility |

### Module Replacement

During initialization, the LoRA manager replaces target layers with LoRA-wrapped versions:

```python
# Original: model.layers[0].self_attn.q_proj = Linear(4096, 4096)
# After:    model.layers[0].self_attn.q_proj = QKVLoRALayer(Linear, lora_backend)
```

Target modules are identified by matching against `target_modules` patterns (e.g., `q_proj`, `k_proj`, `v_proj`, `gate_proj`, `up_proj`).

---

## Memory Pool

**File**: `lora/mem_pool.py`

The `LoRAMemoryPool` manages GPU memory allocation for adapter weights:

```python
class LoRAMemoryPool:
    def __init__(self, max_loras, max_rank, hidden_size, device):
        # Pre-allocated GPU buffers for A and B matrices
        self.lora_a_pool: torch.Tensor  # [max_slots, max_rank, hidden_size]
        self.lora_b_pool: torch.Tensor  # [max_slots, output_size, max_rank]
        
        # Slot management
        self.free_slots: List[int]
        self.adapter_to_slots: Dict[str, List[int]]
```

### Eviction Policy

When the memory pool is full and a new adapter is needed:

| Policy | Behavior |
|--------|----------|
| `lru` | Evict least-recently-used adapter |
| `fifo` | Evict first-loaded adapter |

### Overlapped Loading

**File**: `lora/lora_overlap_loader.py`

When `--enable-lora-overlap-loading` is set:
- Adapter weights are loaded in a background thread
- H2D transfer overlaps with ongoing computation on other streams
- Reduces adapter switch latency

---

## Backend Implementations

### Triton Backend (Default)

**File**: `lora/backend/triton_backend.py`

High-performance kernels using Triton for batched LoRA operations:

| Kernel | File | Operation |
|--------|------|-----------|
| `sgmv_expand` | `triton_ops/chunked_sgmv_expand.py` | LoRA B matrix multiply (expand) |
| `sgmv_shrink` | `triton_ops/chunked_sgmv_shrink.py` | LoRA A matrix multiply (shrink) |
| `sgemm_lora_a` | `triton_ops/sgemm_lora_a.py` | Batched GEMM for A matrices |
| `sgemm_lora_b` | `triton_ops/sgemm_lora_b.py` | Batched GEMM for B matrices |
| `qkv_lora_b` | `triton_ops/qkv_lora_b.py` | Fused Q/K/V LoRA B projection |
| `gate_up_lora_b` | `triton_ops/gate_up_lora_b.py` | Fused gate+up LoRA B projection |
| `embedding_lora_a` | `triton_ops/embedding_lora_a.py` | Embedding layer LoRA |

The key SGMV (Segmented Gather Matrix-Vector) kernel handles the multi-adapter batching:

```
For batch of N requests with adapters [A₁, A₂, A₃]:
  x_gathered[i] = x[seg_start[i]:seg_end[i]]  # per-request input
  output[i] = x_gathered[i] @ lora_a[adapter_id[i]]  # adapter-specific LoRA A
  output[i] = output[i] @ lora_b[adapter_id[i]]      # adapter-specific LoRA B
```

### Torch Backend

**File**: `lora/backend/torch_backend.py`

Pure PyTorch implementation for debugging and platforms without Triton.

### Chunked Backend

**File**: `lora/backend/chunked_backend.py`

Memory-efficient variant that processes LoRA in chunks to reduce peak memory.

---

## LoRA Configuration

**File**: `lora/lora_config.py`

Reads PEFT-compatible `adapter_config.json`:

```python
class LoRAConfig:
    r: int                          # LoRA rank
    lora_alpha: float               # Scaling factor
    target_modules: List[str]       # Which layers to adapt
    lora_dropout: float             # Dropout probability
    bias: str                       # "none", "all", "lora_only"
```

---

## LoRA Adapter

**File**: `lora/lora.py`

```python
class LoRAAdapter:
    def __init__(self, config: LoRAConfig, path: str):
        self.config = config
        self.lora_a_weights: Dict[str, torch.Tensor]  # {layer_name: weight}
        self.lora_b_weights: Dict[str, torch.Tensor]
        self.scaling: float  # lora_alpha / r
```

---

## LoRA Layer Wrappers

**File**: `lora/layers.py`

Each base layer type has a LoRA-aware wrapper:

```python
class QKVLoRALayer(BaseLayerWithLoRA):
    """Wraps QKV projection with multi-adapter LoRA."""
    
    def forward(self, x: torch.Tensor, forward_batch: ForwardBatch) -> torch.Tensor:
        # 1. Base computation
        base_output = self.base_layer(x)
        
        # 2. LoRA delta (batched across adapters)
        lora_a_output = self.backend.sgmv_shrink(x, lora_a_weights, batch_info)
        lora_b_output = self.backend.sgmv_expand(lora_a_output, lora_b_weights, batch_info)
        
        # 3. Add scaled delta
        return base_output + self.scaling * lora_b_output
```

---

## Batch Preparation

Before each forward pass, the LoRA manager prepares adapter metadata:

```python
def prepare_lora_batch(self, forward_batch: ForwardBatch):
    """Prepare LoRA indices and segment information for the batch."""
    # For each request in batch:
    #   - Look up adapter ID in registry
    #   - Record segment boundaries (which tokens use which adapter)
    # Produces: adapter_ids, segment_starts, segment_ends
    # These are passed to SGMV kernels
```

---

## Configuration

```bash
# Launch with LoRA support
python -m sglang.launch_server \
    --model-path meta-llama/Llama-3-8B \
    --lora-paths adapter1=/path/to/adapter1 adapter2=/path/to/adapter2 \
    --max-loras-per-batch 8 \
    --lora-backend triton \
    --lora-eviction-policy lru

# Dynamic loading via API
curl -X POST http://localhost:30000/update_lora \
    -d '{"lora_name": "adapter3", "lora_path": "/path/to/adapter3"}'
```

API usage:
```python
response = client.chat.completions.create(
    model="adapter1",  # Use adapter nickname as model name
    messages=[{"role": "user", "content": "..."}]
)
```

---

## Source References

- LoRAManager: `lora/lora_manager.py`
- Memory Pool: `lora/mem_pool.py`
- Adapter: `lora/lora.py`
- Config: `lora/lora_config.py`
- Triton Backend: `lora/backend/triton_backend.py`
- SGMV Kernels: `lora/triton_ops/`
- Layer Wrappers: `lora/layers.py`
- Registry: `lora/lora_registry.py`
- Overlap Loader: `lora/lora_overlap_loader.py`
