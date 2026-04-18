# Diffusion Distributed Computing

## Overview

The diffusion runtime implements its own distributed computing layer (separate from the LLM runtime's TP/PP system). It provides three orthogonal parallelism axes for video generation:

1. **Sequence Parallelism (SP)**: Splits the spatial-temporal token sequence across GPUs
2. **CFG Parallelism**: Distributes positive/negative classifier-free guidance passes
3. **Tensor Parallelism (TP)**: Standard weight-splitting for large models
4. **Data Parallelism (DP)**: Multiple pipeline replicas for throughput

**File**: `python/sglang/multimodal_gen/runtime/distributed/`

---

## Parallel State Management

**File**: `distributed/parallel_state.py`

A global process group registry providing getter functions for all parallelism types:

```python
# Global state (module-level singletons)
_SP_GROUP: SPGroup | None = None
_TP_GROUP: TPGroup | None = None
_CFG_GROUP: CFGGroup | None = None
_DP_GROUP: DPGroup | None = None

# SP Sub-types
_ULYSSES_GROUP: ProcessGroup | None = None
_RING_GROUP: ProcessGroup | None = None
```

### Process Group Hierarchy

```
World (all GPUs)
├── DP Group (data parallel replicas)
│   ├── SP Group (sequence parallel within pipeline)
│   │   ├── Ulysses Group (head-splitting all-to-all)
│   │   └── Ring Group (token-ring rotation)
│   ├── TP Group (tensor parallel weight-split)
│   └── CFG Group (classifier-free guidance split)
```

### Key Functions

| Function | Returns | Purpose |
|----------|---------|---------|
| `get_sp_group()` | `SPGroup` | Sequence parallelism group |
| `get_sp_rank()` | `int` | Rank within SP group |
| `get_sp_world_size()` | `int` | SP degree |
| `get_tp_group()` | `TPGroup` | Tensor parallelism group |
| `get_tp_rank()` | `int` | Rank within TP group |
| `get_cfg_group()` | `CFGGroup` | CFG parallelism group |
| `get_classifier_free_guidance_rank()` | `int` | 0=positive, 1=negative |
| `get_dp_group()` | `DPGroup` | Data parallelism group |
| `get_world_rank()` | `int` | Global rank |
| `get_world_size()` | `int` | Total GPUs |
| `get_local_torch_device()` | `torch.device` | This GPU's device |

---

## Initialization

**File**: `distributed/comm_utils.py`

```python
def maybe_init_distributed_environment_and_model_parallel(
    tp_size: int = 1,
    enable_cfg_parallel: bool = False,
    ulysses_degree: int = 1,
    ring_degree: int = 1,
    sp_size: int = 1,
    dp_size: int = 1,
):
    """Initialize all distributed groups for diffusion inference."""
    
    # 1. Init torch.distributed (NCCL backend)
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="nccl")
    
    world_size = torch.distributed.get_world_size()
    rank = torch.distributed.get_rank()
    
    # 2. Validate degree configuration
    assert sp_size * tp_size * (2 if enable_cfg_parallel else 1) * dp_size == world_size
    
    # 3. Compute rank-to-group mappings
    # Each GPU belongs to exactly one group in each dimension
    
    # 4. Create process groups
    _create_sp_group(sp_size, ulysses_degree, ring_degree, ...)
    _create_tp_group(tp_size, ...)
    _create_cfg_group(enable_cfg_parallel, ...)
    _create_dp_group(dp_size, ...)
```

---

## Sequence Parallelism (SP)

### Concept

Diffusion models process the entire image/video as one sequence:
- A 720p image → ~4K tokens  
- A 480p, 81-frame video → ~16K tokens
- A 720p, 121-frame video → ~60K tokens

SP shards this sequence across GPUs to distribute memory and computation.

### SP Modes

#### Ulysses SP

All-to-All communication for **head-splitting**:

```
Input: Each GPU has S/P tokens, H heads
All-to-All: Each GPU gets S tokens, H/P heads
Compute: Standard attention on full sequence, fewer heads
All-to-All: Restore original distribution
```

Best for: Moderate parallelism (2-4 GPUs), low communication overhead.

#### Ring SP

K/V chunks rotate through GPUs in a ring:

```
Step 0: GPU_i computes Q_i × K_i
Step 1: GPU_i computes Q_i × K_{(i+1)%P} (K rotated from neighbor)
...
Step P-1: Final accumulation
```

Best for: High parallelism (4-8+ GPUs), overlaps communication with computation.

#### Hybrid (Ulysses + Ring)

When `sp_size = ulysses_degree × ring_degree`:
- Inner dimension: Ulysses (head-split)
- Outer dimension: Ring (token-rotation)

```python
# Example: 8 GPU SP with ulysses=4, ring=2
# 4 GPUs do Ulysses all-to-all internally
# 2 groups of 4 do Ring attention between them
```

### SP Communication Primitives

```python
# Scatter: Split tensor along sequence dim
def sp_scatter(tensor, sp_group, scatter_dim=1):
    """Distribute sequence chunks to SP workers."""
    chunks = tensor.chunk(sp_group.world_size, dim=scatter_dim)
    return chunks[sp_group.rank].contiguous()

# Gather: Concatenate from all SP workers
def sp_gather(tensor, sp_group, gather_dim=1):
    """Collect sequence chunks from all SP workers."""
    gathered = [torch.empty_like(tensor) for _ in range(sp_group.world_size)]
    torch.distributed.all_gather(gathered, tensor, group=sp_group.group)
    return torch.cat(gathered, dim=gather_dim)
```

---

## CFG Parallelism

### Concept

Classifier-Free Guidance requires two forward passes:
1. Conditional (with text prompt)
2. Unconditional (without prompt)

CFG parallelism dedicates different GPUs to each branch:

```
CFG Group (2 ranks):
  Rank 0: positive prompt forward pass
  Rank 1: negative prompt forward pass

After both:
  all_reduce → combined noise prediction
```

### Implementation in DenoisingStage

```python
def _predict_noise_with_cfg(self, model, latents, timestep, ...):
    cfg_rank = get_classifier_free_guidance_rank()
    cfg_group = get_cfg_group()
    
    if cfg_rank == 0:
        # Process conditional (positive)
        noise_pred = model(latents, timestep, encoder_hidden_states=pos_embeds)
    else:
        # Process unconditional (negative)
        noise_pred = model(latents, timestep, encoder_hidden_states=neg_embeds)
    
    # Gather predictions across CFG group
    noise_preds = [torch.zeros_like(noise_pred) for _ in range(2)]
    torch.distributed.all_gather(noise_preds, noise_pred, group=cfg_group.group)
    
    # Apply guidance scale
    noise_pred = noise_preds[1] + guidance_scale * (noise_preds[0] - noise_preds[1])
    return noise_pred
```

### CFG + SP Interaction

When both are enabled:
- Total GPUs = `SP_degree × 2 (CFG)`
- Each CFG branch has its own SP sub-group
- The `ParallelExecutor` handles the interaction:
  - `CFG_PARALLEL` stages: Broadcast batch from rank 0, all execute, barrier
  - `MAIN_RANK_ONLY` stages (e.g., decoding): Only CFG rank 0 executes

---

## Tensor Parallelism (TP)

Standard column-parallel and row-parallel linear layers:

```python
class ColumnParallelLinear(nn.Module):
    """Column-parallel: weight split along output dim."""
    def forward(self, x):
        # Each GPU has W[:, start:end]
        output = F.linear(x, self.weight)
        return output  # [B, S, D/tp_size]

class RowParallelLinear(nn.Module):
    """Row-parallel: weight split along input dim."""
    def forward(self, x):
        # Each GPU has W[start:end, :]
        output = F.linear(x, self.weight)
        # All-reduce to sum partial results
        torch.distributed.all_reduce(output, group=tp_group)
        return output
```

---

## Communication Utilities

**File**: `distributed/comm_utils.py`

```python
# Broadcast Python objects (for request distribution)
def broadcast_pyobj(obj_list, rank, dist_group, src=0):
    """Broadcast arbitrary Python objects across ranks."""
    if rank == src:
        pickled = pickle.dumps(obj_list)
        size_tensor = torch.tensor([len(pickled)], device='cuda')
    else:
        size_tensor = torch.tensor([0], device='cuda')
    
    torch.distributed.broadcast(size_tensor, src=src, group=dist_group)
    
    # Broadcast actual data
    if rank != src:
        pickled = bytes(size_tensor.item())
    data_tensor = torch.frombuffer(pickled, dtype=torch.uint8).cuda()
    torch.distributed.broadcast(data_tensor, src=src, group=dist_group)
    
    if rank != src:
        obj_list = pickle.loads(data_tensor.cpu().numpy().tobytes())
    return obj_list
```

---

## GPU Topology Examples

### 2 GPUs: CFG Parallel

```
GPU 0: CFG rank 0 (positive) → Full SP=1 pipeline
GPU 1: CFG rank 1 (negative) → Full SP=1 pipeline
```

### 4 GPUs: SP=2, CFG=2

```
GPU 0: CFG rank 0, SP rank 0
GPU 1: CFG rank 0, SP rank 1
GPU 2: CFG rank 1, SP rank 0
GPU 3: CFG rank 1, SP rank 1
```

### 8 GPUs: SP=4 (Ulysses=2, Ring=2), CFG=2

```
CFG Positive:  GPU 0,1,2,3 → Ring[0,1] × Ulysses[0,1]
CFG Negative:  GPU 4,5,6,7 → Ring[0,1] × Ulysses[0,1]
```

---

## Source References

- Parallel State: `distributed/parallel_state.py`
- Comm Utils: `distributed/comm_utils.py`
- Initialization: `distributed/__init__.py`
- ParallelExecutor: `pipelines_core/executors/parallel_executor.py`
- DenoisingStage CFG: `pipelines_core/stages/denoising.py`
