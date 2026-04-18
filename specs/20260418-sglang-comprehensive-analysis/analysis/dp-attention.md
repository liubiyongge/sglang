# DP-Attention (Data Parallel Attention)

## Overview

DP-Attention is a distributed attention mechanism in SGLang that partitions work across both data parallel and tensor parallel dimensions within the attention layers. Unlike traditional TP where all ranks process the same tokens, DP-Attention allows each DP rank to process different tokens for attention, then gathers results for the MLP layers.

**Key Source Files:**
- `python/sglang/srt/layers/dp_attention.py` - Core DP-Attention implementation
- `python/sglang/srt/managers/data_parallel_controller.py` - DP request routing
- `python/sglang/srt/managers/scheduler_dp_attn_mixin.py` - Scheduler integration

---

## Architecture

### Attention TP Group vs Standard TP Group

DP-Attention creates a hierarchical group structure:

```
Standard TP Group: All ranks process same tokens (tensor parallel)
Attention TP Group: Subset of TP ranks that share attention work

Example with TP=4, DP=2:
  Global TP ranks: 0, 1, 2, 3
  Attention TP size: 4 / 2 = 2
  Group 0 (DP rank 0): ranks 0, 1 → process tokens A
  Group 1 (DP rank 1): ranks 2, 3 → process tokens B
  MLP: All 4 ranks process gathered tokens A+B together
```

### World Info Computation

```python
def compute_dp_attention_world_info(enable_dp_attention, tp_rank, tp_size, dp_size):
    attn_tp_size = tp_size // dp_size       # TP size within attention group
    attn_dp_rank = tp_rank // attn_tp_size  # Which DP group this rank belongs to
    attn_tp_rank = tp_rank % attn_tp_size   # Rank within attention group
    return attn_tp_rank, attn_tp_size, attn_dp_rank
```

---

## Padding Modes: MAX_LEN vs SUM_LEN

### Mode Selection Logic

```python
@classmethod
def get_dp_padding_mode(cls, is_extend_in_batch, global_num_tokens: List[int]):
    if is_extend_in_batch:
        return DpPaddingMode.SUM_LEN
    
    max_len = max(global_num_tokens)
    sum_len = sum(global_num_tokens)
    if sum_len * 2 > max_len * get_attention_dp_size():
        return cls.MAX_LEN
    else:
        return cls.SUM_LEN
```

### MAX_LEN Mode

- **Strategy**: Pad all tokens to the maximum sequence length across DP ranks
- **Communication**: `all_gather_into_tensor` for gathering
- **When Used**: Balanced workloads, decode batches, `sum_len * 2 <= max_len * dp_size`
- **Buffer Size**: `max_num_tokens * dp_size`
- **Advantages**:
  - Efficient for balanced workloads
  - Enables symmetric memory allocation
  - Compatible with CUDA graphs
  - Predictable buffer sizes

### SUM_LEN Mode

- **Strategy**: Pad tokens to their actual sum (no uniform length)
- **Communication**: `all_reduce` for gathering
- **When Used**: Mixed prefill/decode (`is_extend_in_batch`), skewed workloads
- **Buffer Size**: `sum(global_num_tokens)`
- **Advantages**:
  - Better memory efficiency for skewed workloads
  - Reduces communication for unbalanced batches
- **Overhead**: Requires manual token positioning via `memcpy_triton`

---

## Gather and Scatter Operations

### DP Gather (Attention → MLP)

After attention computes on local tokens, results must be gathered for MLP:

**MAX_LEN Path:**
```python
def _dp_gather_via_all_gather(global_tokens, local_tokens, forward_batch, is_partial):
    if get_attention_tp_rank() != 0:
        local_tokens.fill_(0)  # Only TP rank 0 has data
    
    # Reduce-scatter among attention TP group
    scattered = local_tokens.tensor_split(attn_tp_size)[attn_tp_rank]
    get_attention_tp_group().reduce_scatter_tensor(scattered, local_tokens)
    
    # All-gather from all DP ranks
    get_tp_group().all_gather_into_tensor(global_tokens, scattered)
```

**SUM_LEN Path:**
```python
def _dp_gather_via_all_reduce(global_tokens, local_tokens, forward_batch, is_partial):
    global_tokens.fill_(0)
    
    # Copy to correct position in global buffer
    if local_tokens.shape[0] > 0 and (is_partial or attn_tp_rank == 0):
        memcpy_triton(global_tokens, local_tokens, 0, local_start_pos, local_num_tokens, False)
    
    # All-reduce to combine
    global_tokens[:] = tensor_model_parallel_all_reduce(global_tokens)
```

### DP Scatter (MLP → Attention)

After MLP computes on all tokens, results are scattered back to local:

```python
def dp_scatter(local_tokens, global_tokens, forward_batch):
    local_start_pos, local_num_tokens = get_dp_local_info(forward_batch)
    local_tokens.fill_(0)
    if local_tokens.shape[0] > 0:
        memcpy_triton(local_tokens, global_tokens, 0, local_start_pos, local_num_tokens, True)
```

### Position Computation

```python
def get_dp_local_info(forward_batch):
    dp_rank = get_attention_dp_rank()
    cumtokens = torch.cumsum(forward_batch.global_num_tokens_gpu, dim=0)
    
    if dp_rank == 0:
        local_start_pos = torch.zeros_like(cumtokens[0])
    else:
        local_start_pos = cumtokens[dp_rank - 1]
    
    local_num_tokens = forward_batch.global_num_tokens_gpu[dp_rank]
    return local_start_pos, local_num_tokens
```

---

## Triton Kernel for Memory Copy

```python
@triton.jit
def memcpy_triton_kernel(dst_ptr, src_ptr, offset_ptr, sz_ptr, offset_src, chunk_size, BLOCK_SIZE):
    pid = tl.program_id(axis=0).to(tl.int64)
    offset = tl.load(offset_ptr).to(tl.int64) * chunk_size
    sz = tl.load(sz_ptr).to(tl.int64) * chunk_size
    
    start_index = pid * BLOCK_SIZE
    offs = tl.arange(0, BLOCK_SIZE)
    mask = start_index + offs < sz
    
    if offset_src:
        # Read from offset, write to start (scatter from global)
        data = tl.load(src_ptr + offset + start_index + offs, mask=mask)
        tl.store(dst_ptr + start_index + offs, data, mask=mask)
    else:
        # Read from start, write to offset (gather to global)
        data = tl.load(src_ptr + start_index + offs, mask=mask)
        tl.store(dst_ptr + offset + start_index + offs, data, mask=mask)
```

Block size: 8192 elements.

---

## Data Parallel Controller

**Source:** `python/sglang/srt/managers/data_parallel_controller.py`

### Request Routing

The controller manages request distribution across DP workers:

```python
class DPBudget:
    def dispatch(self, method: LoadBalanceMethod):
        if method == LoadBalanceMethod.TOTAL_REQUESTS:
            target_rank = self.total_requests.index(min(self.total_requests))
        elif method == LoadBalanceMethod.TOTAL_TOKENS:
            target_rank = min(
                range(self.dp_size),
                key=lambda i: (self.total_tokens[i], self.total_requests[i]))
        self.total_requests[target_rank] += 1
        return target_rank
```

### Load Balance Methods

| Method | Strategy |
|--------|----------|
| `TOTAL_REQUESTS` | Route to rank with fewest requests |
| `TOTAL_TOKENS` | Route to rank with fewest tokens (tie-break on requests) |

### DP-Attention Scheduler Launch

```python
def launch_dp_attention_schedulers(self, server_args, port_args):
    # All DP ranks share same NCCL port (they're in same TP group)
    for dp_rank in range(server_args.dp_size):
        port_and_socket = get_zmq_socket(self.context, zmq.PUSH)
        worker_ports.append(port_and_socket[0])
        self.workers[dp_rank] = port_and_socket[1]
    
    # Broadcast ports to all nodes
    broadcasted_ports = self._broadcast_worker_ports(server_args, worker_ports)
    # Launch single TP group (all DP ranks within)
    self.launch_tensor_parallel_group(server_args, port_args, 0, None, broadcasted_ports)
```

---

## Scheduler Integration (MLP Sync Batch)

**Source:** `python/sglang/srt/managers/scheduler_dp_attn_mixin.py`

### MLP Sync Batch Preparation

Each DP rank synchronizes batch metadata with all other DP ranks:

```python
def prepare_mlp_sync_batch_raw(local_batch, dp_size, attn_tp_size, tp_group, ...):
    # Gather batch info from all DP workers
    mlp_sync_info = MLPSyncBatchInfo(
        num_tokens=local_batch.extend_num_tokens if extend else batch_size,
        can_cuda_graph=local_batch.forward_mode.is_decode_or_idle(),
        is_extend_in_batch=local_batch.forward_mode.is_extend(),
    )
    
    # All-gather metadata across TP group
    mlp_sync_info.all_gather(device=device, group=group)
    
    # Extract info from TP rank 0 of each DP group
    tp0_info = global_info_tensor[:, 0, :]
    mlp_sync_info.global_num_tokens = tp0_info[:, 0].tolist()
    mlp_sync_info.is_extend_in_batch = bool(tp0_info[:, 3].max().item())
```

### Forward Batch DP Metadata

```python
@dataclass
class ForwardBatch:
    # DP-Attention specific fields
    global_num_tokens_cpu: Optional[List[int]]        # Per DP rank token counts
    global_num_tokens_gpu: Optional[torch.Tensor]     # GPU tensor
    dp_padding_mode: Optional[DpPaddingMode]          # MAX_LEN or SUM_LEN
    dp_local_start_pos: Optional[torch.Tensor]        # Local rank's start
    dp_local_num_tokens: Optional[torch.Tensor]       # Local rank's count
    global_dp_buffer_len: Optional[int]               # Gather buffer size
    is_extend_in_batch: bool                          # Mixed prefill/decode?
    can_run_dp_cuda_graph: bool                       # CUDA graph eligible?
```

---

## Layer Scatter Modes

The communicator layer manages data distribution state:

```python
class LayerScatterModes:
    layer_input_mode: ScatterMode     # SCATTERED, TP_ATTN_FULL, or FULL
    attn_mode: ScatterMode            # Usually TP_ATTN_FULL
    mlp_mode: ScatterMode             # Dense layers need FULL
    middle_residual_mode: ScatterMode
    layer_output_mode: ScatterMode
```

| Mode | Meaning |
|------|---------|
| `SCATTERED` | Data split only on this rank |
| `TP_ATTN_FULL` | All ranks in attention TP group share full data |
| `FULL` | All ranks in entire TP group share full data |

Transitions between modes trigger gather/scatter operations.

---

## Integration with Logits Processing

```python
def _gather_dp_attn_hidden_states(self, hidden_states, logits_metadata):
    # Gather hidden states from all DP ranks for final logits computation
    logits_metadata.compute_dp_attention_metadata()
    local_hidden_states = hidden_states
    hidden_states = logits_metadata.gathered_buffer
    dp_gather_replicate(hidden_states, local_hidden_states, logits_metadata)
    return hidden_states, local_hidden_states

def _scatter_dp_attn_logits(self, logits, local_hidden_states, logits_metadata):
    # Scatter logits back to local rank
    global_logits = logits
    logits = torch.empty((local_hidden_states.shape[0], global_logits.shape[1]), ...)
    dp_scatter(logits, global_logits, logits_metadata)
    return logits
```

---

## Communication Optimization Functions

```python
def dp_reduce_scatter_tensor(output, input):
    """Optimized reduce-scatter for DP attention groups"""
    if tp_world_size == dp_size:
        # Simple: TP size == DP size
        get_tp_group().reduce_scatter_tensor(output, input)
    else:
        # Complex: need intermediate all-gather
        scattered = input.tensor_split(tp_world_size)[tp_rank]
        get_tp_group().reduce_scatter_tensor(scattered, input)
        get_attention_tp_group().all_gather_into_tensor(output, scattered)

def attn_tp_reduce_scatter_tensor(output, input):
    """Reduce-scatter within attention TP group only"""
    return get_attention_tp_group().reduce_scatter_tensor(output, input)
```

---

## Symmetric Memory Allocation

For efficient all-gather in MAX_LEN mode:

```python
def get_global_dp_buffer():
    with use_symmetric_memory(get_tp_group(), disabled=not cls._dp_max_padding):
        buffer = torch.empty(
            (cls._global_dp_buffer_len, cls._hidden_size),
            dtype=cls._dtype, device=cls._device)
    return buffer
```

Symmetric memory enables direct GPU-to-GPU transfers without CPU bounce.

---

## CUDA Graph Compatibility

```python
@classmethod
def get_default_mode_in_cuda_graph(cls):
    if _USE_ROCM700A_WA:
        return cls.SUM_LEN    # Workaround for ROCm 7.0.0 alpha
    else:
        return cls.MAX_LEN    # Standard for NVIDIA
```

---

## Performance Characteristics

**MAX_LEN Mode:**
- Communication: O(max_len × dp_size × hidden_size)
- Memory: O(max_len × dp_size × hidden_size) for gather buffer
- Best when: Balanced token distribution across DP ranks

**SUM_LEN Mode:**
- Communication: O(sum_len × hidden_size)
- Memory: O(sum_len × hidden_size) for gather buffer
- Best when: Unbalanced token distribution (e.g., one rank has 10× tokens)

---

## Execution Flow

```
Per Transformer Layer:
  1. [SCATTERED] Input arrives at local DP rank
  2. [DP-Attention] Each DP rank processes its own tokens
     - Attention computation on local subset
     - KV cache local to this DP rank
  3. [DP-GATHER] Results gathered to all ranks
     - MAX_LEN: reduce_scatter + all_gather
     - SUM_LEN: memcpy + all_reduce
  4. [FULL] MLP processes all gathered tokens
     - Standard TP all-reduce for MLP
  5. [DP-SCATTER] Results scattered back to local rank
     - memcpy_triton extracts local portion
  6. [SCATTERED] Output ready for next layer
```

---

## Key Design Decisions

1. **Lazy Buffer Allocation**: DP buffers allocated only when needed
2. **Mode-aware Communication**: Different patterns for MAX_LEN vs SUM_LEN
3. **Position Caching**: `dp_local_start_pos` cached to avoid recomputation
4. **TP Rank 0 Convention**: Only attention TP rank 0 performs scatter operations
5. **Hierarchical Grouping**: World partitioned into attention groups reusing TP resources
6. **Scheduler Decoupling**: DP controller manages request routing, not compute
