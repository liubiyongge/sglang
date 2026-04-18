# Tensor Parallelism Implementation

## Overview

SGLang implements tensor parallelism (TP) to shard model weights and computation across multiple GPUs within a single layer. Each TP worker runs as a separate OS process with a dedicated GPU, coordinating through NCCL-based collective operations (all-reduce, all-gather) embedded within the forward pass.

**Key Source Files:**
- `python/sglang/srt/managers/tp_worker.py` - Core TP worker class
- `python/sglang/srt/model_executor/model_runner.py` - Model forward pass and weight loading
- `python/sglang/srt/layers/linear.py` - ColumnParallelLinear, RowParallelLinear
- `python/sglang/srt/layers/communicator.py` - Attention-specific TP operations
- `python/sglang/srt/distributed/parallel_state.py` - Group coordination
- `python/sglang/srt/entrypoints/engine.py` - Process spawning

---

## TpModelWorker Class

**Source:** `python/sglang/srt/managers/tp_worker.py`

### Initialization

```
Constructor receives:
  server_args     - Global server configuration
  gpu_id          - Assigned GPU device ID
  tp_rank         - Rank within tensor parallel group
  moe_ep_rank     - Rank for MOE expert parallelism
  pp_rank         - Pipeline parallel rank
  dp_rank         - Data parallel rank
  nccl_port       - NCCL communication port
```

**Setup steps:**
1. Creates a `ModelRunner` instance that handles actual forward passes
2. For multi-layer EAGLE (speculative decoding), creates multiple ModelRunner instances
3. Synchronizes random seed across all TP workers via `broadcast_pyobj()`
4. Initializes memory pools: `req_to_token_pool` and `token_to_kv_pool_allocator`
5. Gets communication groups: `pp_group` and `world_group`

### Key Methods

| Method | Purpose |
|--------|---------|
| `forward_batch_generation()` | Execute forward pass for a batch |
| `update_weights_from_disk()` | Load new weights (LoRA, checkpoints) |
| `update_weights_from_tensor()` | Update weights from in-memory tensors |
| `update_weights_from_distributed()` | Update weights from distributed source |
| `get_worker_info()` | Return capacity info (max tokens, max reqs, device) |

### Forward Batch Execution

```
forward_batch_generation():
  1. Call model_runner.forward() - compute next token logits
  2. If last PP rank:
     - Perform sampling (greedy, top-k, top-p, etc.)
     - Support delayed sampling with overlap mode
  3. Return sampled token IDs and metadata
```

---

## Process Spawning and Coordination

**Source:** `python/sglang/srt/entrypoints/engine.py` (lines 920-962)

### Spawning Pattern

```python
for pp_rank in pp_rank_range:
    for tp_rank in tp_rank_range:
        gpu_id = base_gpu_id + (pp_rank * tp_size_per_node) + (tp_rank * gpu_id_step)
        moe_ep_rank = tp_rank // (tp_size // ep_size)
        
        proc = mp.Process(
            target=run_scheduler_process_func,
            args=(server_args, port_args, gpu_id, tp_rank, moe_ep_rank, pp_rank, dp_rank, writer)
        )
        proc.start()
```

### Process Layout

- Each TP worker = independent OS process with dedicated GPU
- Within a node: TP workers assigned consecutive GPU IDs
- Multi-node: coordinated via NCCL over network
- Each scheduler process instantiates a `Scheduler` -> creates `TpModelWorker`
- Random seed synchronized across all TP ranks (broadcast from rank 0)

---

## Model Runner Initialization

**Source:** `python/sglang/srt/model_executor/model_runner.py` (lines 278-433)

```python
def __init__(self, tp_rank, tp_size, pp_rank, pp_size, ...):
    self.tp_rank = tp_rank
    self.tp_size = tp_size
    
    # Initialize distributed environment
    init_distributed_environment(
        world_size=tp_size * pp_size,
        rank=tp_size * pp_rank + tp_rank,
        distributed_init_method="env://",
        backend="nccl")
    
    # Initialize model parallel groups
    initialize_model_parallel(
        tensor_model_parallel_size=tp_size,
        pipeline_model_parallel_size=pp_size,
        backend="nccl")
    
    # Load model with TP sharding
    self.load_model()
    
    # Initialize attention backend
    attn_backend_wrapper(...)
    
    # Setup CUDA graphs
    self.cuda_graph_runner = CudaGraphRunner(...)
```

---

## Weight Sharding Strategies

**Source:** `python/sglang/srt/layers/linear.py`

### ColumnParallelLinear (Output Sharding)

Weight matrix `A` is split along the output dimension. Each rank stores `output_size / tp_size` columns.

**Loading:**
```python
shard_size = param_data.shape[output_dim]
start_idx = tp_rank * shard_size
loaded_weight = loaded_weight.narrow(output_dim, start_idx, shard_size)
```

**Forward:**
```python
def forward(self, input_):
    output_parallel = self.quant_method.apply(self, input_, bias)
    if self.gather_output:
        output = tensor_model_parallel_all_gather(output_parallel)
    else:
        output = output_parallel
    return output, output_bias
```

**Pattern:** Each rank computes `Y_i = X * A_i`, optional all-gather to reconstruct full output.

### RowParallelLinear (Input Sharding)

Weight matrix `A` is split along the input dimension. Each rank stores `input_size / tp_size` rows.

**Forward:**
```python
def forward(self, input_, skip_all_reduce=False):
    if not self.input_is_parallel:
        splitted_input = split_tensor_along_last_dim(input_, num_partitions=self.tp_size)
        input_parallel = splitted_input[self.tp_rank]
    
    output_parallel = self.quant_method.apply(self, input_parallel, bias=bias_)
    
    if self.reduce_results and self.tp_size > 1 and not skip_all_reduce:
        output = get_attention_tp_group().all_reduce(output_parallel)
    else:
        output = output_parallel
    
    return output, output_bias
```

**Pattern:** Each rank computes partial result, all-reduce aggregates across TP group.

### MoE Weight Sharding

For Mixture-of-Experts models, sharding involves both TP and EP dimensions:
- `moe_tp_size = tp_size // ep_size`
- Expert parameters sharded based on MOE architecture
- Different experts assigned to different EP ranks

---

## Communication Patterns in Forward Pass

### Transformer Layer Communication

For each transformer layer, the typical communication pattern is:

```
1. Attention QKV Projection (ColumnParallelLinear):
   - Each rank computes Q_i, K_i, V_i for its shard
   - No communication needed

2. Attention Computation:
   - Each rank processes its attention heads independently
   - No communication needed (head-parallel)

3. Attention Output Projection (RowParallelLinear):
   - Each rank computes partial output
   - ALL-REDUCE to aggregate across TP group

4. MLP Gate+Up Projection (ColumnParallelLinear):
   - Each rank computes its shard
   - No communication needed

5. MLP Down Projection (RowParallelLinear):
   - Each rank computes partial output
   - ALL-REDUCE to aggregate across TP group
```

**Result:** 2 all-reduce operations per transformer layer.

### Attention-Specific TP Operations

**Source:** `python/sglang/srt/layers/communicator.py`

```python
def tp_all_gather_hidden_states(self, hidden_states, forward_batch):
    total_tokens = forward_batch.input_ids.shape[0]
    output = hidden_states.new_empty((total_tokens, hidden_states.shape[-1]))
    get_tp_group().all_gather_into_tensor(output, hidden_states)
    return output
```

### DP-Attention with Reduce-Scatter

For distributed attention (dp_attention.py), the pattern uses reduce-scatter:
```python
get_tp_group().reduce_scatter_tensor(output, input_tensor)
```

---

## TP Group Creation

**Source:** `python/sglang/srt/distributed/parallel_state.py`

```
initialize_model_parallel():
  1. num_tp_groups = world_size / tp_size
  2. For each group i:
     ranks = [i*tp_size, i*tp_size+1, ..., (i+1)*tp_size-1]
     create GroupCoordinator(ranks, backend="nccl")
```

**Example:** 8 GPUs, TP=4, PP=2
- TP groups: [0,1,2,3], [4,5,6,7]
- PP groups: [0,4], [1,5], [2,6], [3,7]

---

## All-Reduce Backend Selection

Within a TP group, all-reduce operations dynamically select the optimal backend:

```
Dispatch Order (priority):
1. Deterministic mode (AMD) → Custom AR 1-stage
2. Custom AllReduce → Small tensors, single node, NVLink
3. Quick AllReduce → AMD MI300, quantized
4. PyMsccl++ → Graph mode, small messages (<=1MB)
5. Torch SymmMem → H100+, graph mode
6. PyNccl → Graph mode, any size
7. torch.distributed → Eager mode fallback
```

---

## CUDA Graph Integration

TP communication is captured in CUDA graphs for decode phase:

```python
with tp_group.graph_capture(stream=stream):
    # All collective operations within this block are captured
    # Includes: all-reduce (CA, PyNccl, PyMsccl++), all-gather
    forward_pass(batch)
```

**Benefits:**
- Zero kernel launch overhead for repeated decode steps
- Communication patterns pre-recorded
- Significant speedup for decode phase (same batch shapes)

---

## Overlap and Asynchronous Execution

### Communication-Computation Overlap

- Hidden state all-gather can overlap with subsequent computation
- Sampling done on CPU while GPU runs next forward pass
- Prefill and decode batches overlapped in "overlap mode"

### FutureMap for Overlapping

Circular buffer storing future token IDs before resolution:
- Allows prefill batches to proceed while decode batch computes
- For speculative decoding: stores multiple draft outputs

---

## Data Structures

### ModelWorkerBatch
Contains sequences for a batch:
- Input IDs, attention masks, sequence lengths
- Sampling parameters (temperature, top-k, top-p)
- LoRA adapter info

### ForwardBatch
GPU execution batch:
- Flattened input tokens (all tokens from all sequences)
- Position IDs, attention info
- Speculative decoding info

**TP Coordination:**
- Each TP worker receives the same ModelWorkerBatch (via IPC)
- Forward pass synchronized at collective operation points
- Results independently computed then aggregated

---

## Communication Flow Example (TP=4)

```
TP Rank 0       TP Rank 1       TP Rank 2       TP Rank 3
    |               |               |               |
    +-- Load Q0,K0,V0  Q1,K1,V1    Q2,K2,V2        Q3,K3,V3
    |               |               |               |
    +-- Attention   Attention       Attention       Attention
    |  (heads 0-7) (heads 8-15)   (heads 16-23)  (heads 24-31)
    |               |               |               |
    +-- Output proj (partial)       (partial)       (partial)
    |               |               |               |
    +===== ALL-REDUCE (sum partial outputs) ========+
    |               |               |               |
    +-- MLP gate+up (shard 0)      (shard 1)       (shard 2)      (shard 3)
    |               |               |               |
    +-- MLP down    (partial)       (partial)       (partial)
    |               |               |               |
    +===== ALL-REDUCE (sum partial outputs) ========+
    |               |               |               |
    v               v               v               v
  (identical hidden states on all ranks)
```

---

## File Organization

| Component | File | Purpose |
|-----------|------|---------|
| TP Worker | `managers/tp_worker.py` | Core TP worker lifecycle |
| Scheduler | `managers/scheduler.py` | Main event loop, batch scheduling |
| Model Runner | `model_executor/model_runner.py` | Forward pass, weight loading |
| Linear Layers | `layers/linear.py` | Column/RowParallelLinear |
| Communicator | `layers/communicator.py` | Attention-specific TP ops |
| Parallel State | `distributed/parallel_state.py` | Group coordination |
| Overlap Utils | `managers/overlap_utils.py` | Async token map |
| Engine | `entrypoints/engine.py` | Process spawning |
