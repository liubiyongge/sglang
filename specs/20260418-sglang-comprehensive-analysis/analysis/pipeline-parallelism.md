# Pipeline Parallelism Implementation

## Overview

SGLang implements pipeline parallelism (PP) to distribute model layers across multiple GPUs sequentially. The implementation uses a `SchedulerPPMixin` class that provides specialized event loops for pipeline execution, with support for async batch depth to overlap computation and communication. Each pipeline stage runs as an independent scheduler process.

**Key Source Files:**
- `python/sglang/srt/managers/scheduler_pp_mixin.py` - Main PP scheduling logic (~1390 lines)
- `python/sglang/srt/distributed/parallel_state.py` - P2P communication, tensor dict send/recv
- `python/sglang/srt/model_executor/forward_batch_info.py` - PPProxyTensors
- `python/sglang/srt/managers/schedule_batch.py` - ScheduleBatch with PP fields

---

## Design Principles

1. **Sequential stage ordering** - Each PP stage runs in fixed order
2. **Async sends, sync receives** - Minimizes desynchronization while reducing communication overhead
3. **Microbatch pipelining** - Multiple batches in flight to reduce pipeline bubble
4. **Overlapped execution** - GPU computation overlaps with CPU post-processing and communication

---

## Event Loop Implementations

The `SchedulerPPMixin` provides three event loop variants:

| Event Loop | Purpose |
|---|---|
| `event_loop_pp()` | Standard pipeline parallelism |
| `event_loop_pp_disagg_prefill()` | PP with disaggregated prefill |
| `event_loop_pp_disagg_decode()` | PP with disaggregated decode |

---

## Microbatch Formation

### Loop State Initialization

```python
def init_pp_loop_state(self):
    pp_loop_size = pp_size + pp_async_batch_depth
    mbs = [None] * pp_loop_size           # Current microbatches
    last_mbs = [None] * pp_loop_size       # Previous microbatches
    running_mbs = [ScheduleBatch(...)] * pp_loop_size  # Running batch tracking
    mb_metadata = [None] * pp_loop_size    # Metadata per microbatch
    last_rank_comm_queue = deque()          # Output buffer for last rank
```

### Key Parameters

| Parameter | Description | Example |
|-----------|-------------|---------|
| `pp_size` | Number of pipeline stages | 4 |
| `pp_async_batch_depth` | Additional buffered microbatches | 2 |
| `pp_loop_size` | Total microbatches in flight | 6 (4+2) |

### Loop Iteration

```
for mb_id in range(pp_loop_size):
    next_first_rank_mb_id = (mb_id + pp_size) % pp_loop_size
    next_mb_id = (mb_id + 1) % pp_loop_size
```

- `mb_id` - Current microbatch index
- `next_first_rank_mb_id` - Microbatch going to PP rank 0 after this completes
- `next_mb_id` - Microbatch for next stage

---

## Pipeline Execution Schedule

### Per-Stage Processing

```
Stage P (for each microbatch):
  1. recv_requests() from input
  2. process_input_requests()
  3. send_requests to next stage (async)
  4. get_next_batch_to_run()
  5. recv_proxy_tensors() from previous stage (hidden states)
  6. [IF async_batch_depth > 0] preprocess outputs in parallel
  7. launch_batch() - run forward pass
  8. [IF async_batch_depth == 0] recv outputs and preprocess
  9. wait for previous batch and process results
  10. send_proxy_tensors() to next stage (async)
  11. send_outputs to next stage for post-processing
```

### Timeline Overlapping

GPU computation of batch N happens in parallel with:
- CPU processing of batch N-1's results
- Communication of batch N+1 data
- Receiving data for batch N+2

---

## Communication Between Pipeline Stages

### Data Structures

#### PPProxyTensors

**Source:** `forward_batch_info.py` (line 972)

```python
@dataclass
class PPProxyTensors:
    tensors: Dict[str, torch.Tensor]
    
    def __getitem__(self, key: str) -> torch.Tensor
    def __setitem__(self, key: str, value: torch.Tensor)
```

Contents passed between stages:
- `hidden_states` - Layer outputs (intermediate representations)
- `residual` - Residual connections
- `next_token_ids` - Token predictions (last stage only)
- Logprobs (if requested)

#### P2PWork

**Source:** `parallel_state.py` (line 76)

```python
@dataclass
class P2PWork:
    work: Optional[torch.distributed.Work]  # NCCL/PyTorch comm handle
    payload: Optional[torch.Tensor]          # Tensor being sent
```

### Send/Receive Methods

**Source:** `parallel_state.py` (lines 1157-1270)

**Sending (asynchronous):**
```python
send_tensor_dict(
    tensor_dict: Dict[str, torch.Tensor],
    dst: Optional[int] = None,              # Default: next rank
    all_gather_group: Optional[GroupCoordinator] = None,
    async_send: bool = False
) -> List[P2PWork]
```

**Receiving (synchronous):**
```python
recv_tensor_dict(
    src: Optional[int] = None,              # Default: previous rank
    all_gather_group: Optional[GroupCoordinator] = None
) -> Dict[str, torch.Tensor]
```

### Communication Optimizations

1. **Metadata serialization** - Separates tensor metadata from actual data
2. **All-gather option** - Send-allgather to reduce redundant tensor sends:
   ```python
   if all_gather_group and tensor.numel() % all_gather_size == 0:
       tensor = tensor.reshape(all_gather_size, -1)[all_gather_rank]
   ```
3. **Async sends** - Non-blocking for continued CPU scheduling
4. **Empty tensor skipping** - Avoids communication for zero-sized tensors

---

## Communication Flow in event_loop_pp()

```python
# 1. REQUEST DISTRIBUTION (first stage only)
if not pp_group.is_last_rank:
    send_req_work = _pp_send_pyobj_to_next_stage(recv_reqs, async_send=True)

# 2. PROXY TENSOR RECEPTION (non-first stages)
if not pp_group.is_first_rank:
    pp_proxy_tensors = _pp_recv_proxy_tensors()

# 3. BATCH EXECUTION
result = _pp_launch_batch(mb_id, pp_proxy_tensors, ...)

# 4. OUTPUT SENDING (non-last stages)
if not pp_group.is_last_rank:
    send_proxy_work = _pp_send_dict_to_next_stage(
        result.pp_hidden_states_proxy_tensors.tensors, async_send=True)

# 5. EARLY OUTPUT SENDING (last stage)
if pp_group.is_last_rank:
    last_rank_comm_queue.append((event, pp_outputs))
```

---

## Rank Responsibilities

| Rank | Role | Behavior |
|------|------|----------|
| First (PP=0) | Entry | Receives from scheduler, sends requests + hidden states forward |
| Middle (0 < PP < N-1) | Relay | Receives from prev, processes layers, sends to next |
| Last (PP=N-1) | Exit | Processes final layers, samples, buffers outputs |

### Rank Identification (parallel_state.py)

```python
@property
def is_first_rank(self):
    return self.rank == self.first_rank

@property
def is_last_rank(self):
    return self.rank == self.last_rank

@property
def next_rank(self):
    return self.ranks[(rank_in_group + 1) % world_size]

@property
def prev_rank(self):
    return self.ranks[(rank_in_group - 1) % world_size]
```

---

## Async Batch Depth Optimization

**Parameter:** `pp_async_batch_depth` (default: 0)

### With `pp_async_batch_depth > 0`:

```python
# EARLY: Start output processing before current batch GPU computation
next_pp_outputs, next_batch_result, d2h_event = (
    _pp_commit_send_output_work_and_preprocess_output_tensors(...))

# Launch current batch while output processing happens
result = _pp_launch_batch(...)
```

### With `pp_async_batch_depth == 0`:

```python
# LATE: Process output after current batch GPU computation
result = _pp_launch_batch(...)
next_pp_outputs, next_batch_result, d2h_event = (
    _pp_commit_send_output_work_and_preprocess_output_tensors(...))
```

### Effect on Pipeline Bubble

- `depth=0`: Maximum dependency/stalling, minimal memory
- `depth>0`: Overlapping opportunities, reduced last-rank stalling, more memory

---

## Execution Flow Diagram

```
PP Stage 0 (First)       PP Stage 1 (Middle)      PP Stage 2 (Last)
    |                           |                        |
    +--recv_requests            |                        |
    +--get_batch()              |                        |
    |--forward(batch)--+        |                        |
    +--send_hidden-----+------->|--recv_hidden           |
                                +--get_batch()           |
                                |--forward(batch)--+     |
                                +--send_hidden-----+---->|--recv_hidden
                                                         +--get_batch()
                                                         |--forward(batch)
                                                         +--buffer_output
    +--recv_output<-------------|--recv_output<---------|
    +--process_result           |                        |
                                +--process_result        |
                                                         +--postprocess

Time ------------------------------------------------------------>
```

---

## Example: PP_SIZE=4, ASYNC_BATCH_DEPTH=2

`pp_loop_size = 6`

Microbatches in flight:
```
Time T:   [mb0@stage0, mb1@stage1, mb2@stage2, mb3@stage3, mb4@stage0, mb5@stage1]
Time T+1: [mb6@stage0, mb1@stage1, mb2@stage2, mb3@stage3, mb4@stage0, mb5@stage1]
          (mb0 exits after completing all 4 stages)
```

Each stage processes continuously without gaps, with up to 6 batches in the pipeline simultaneously.

---

## Key Methods

### Initialization
```python
init_pp_loop_state()
  - Creates microbatch arrays of size pp_loop_size
  - Initializes running_mbs with empty ScheduleBatch instances
  - Creates last_rank_comm_queue deque for output buffering
```

### Communication Primitives
```python
_pp_recv_pyobj_from_prev_stage()    # Receive Python objects (request IDs)
_pp_recv_dict_from_prev_stage()     # Receive tensor dictionary (hidden states)
_pp_recv_proxy_tensors()            # Wrapper for hidden state reception
_pp_send_pyobj_to_next_stage(data, async_send=True)  # Send objects async
_pp_send_dict_to_next_stage(tensor_dict, async_send=True)  # Send tensors async
_pp_commit_comm_work(work_list)     # Wait for all async work to complete
```

### Batch Execution
```python
_pp_launch_batch(mb_id, pp_proxy_tensors, mb_metadata, last_rank_comm_queue):
  - Execute forward pass on GPU
  - Record CUDA event for synchronization
  - On last rank: buffer output for async batch depth
  - Return (result, event)
```

### Output Handling
```python
_pp_send_recv_and_preprocess_output_tensors(
    next_first_rank_mb_id, next_mb_id, ...):
  - Send output from current batch (if not last stage)
  - Receive output for next batch (if available)
  - Preprocess output tensors (extract next_token_ids, logprobs)
  - Create D2H event for synchronization
  - Return (next_pp_outputs, batch_result, d2h_event, send_output_work)
```

---

## Disaggregation Support

Two additional event loops extend PP with disaggregated inference:

### event_loop_pp_disagg_prefill()

Adds to standard PP:
- Bootstrap request handling
- Consensus mechanism for KV transfer success
- Release tracking for transferred requests

### event_loop_pp_disagg_decode()

Adds to standard PP:
- Retract request consensus
- Preallocation tracking
- Transfer queue management

Both follow the same PP scheduling pattern with extra synchronization for disaggregation.

---

## PP Group Creation

**Source:** `parallel_state.py`, `initialize_model_parallel()`

```
PP groups use strided ranks:
  Example: 8 GPUs, PP=4, TP=2
  TP groups: [0,1], [2,3], [4,5], [6,7]
  PP groups: [0,2,4,6], [1,3,5,7]
```

Each PP group uses NCCL backend with `use_custom_allreduce=False` (P2P only, no collective needed).

---

## Performance Considerations

### Pipeline Bubble Mitigation
- Async batch depth buffering (multiple batches in flight)
- Overlapping communication with computation
- CUDA events and streams for async synchronization

### Communication Bottleneck
- Synchronous receive on critical path (hidden states needed before forward)
- Asynchronous send for non-blocking propagation
- Send-allgather optimization for TP+PP composition

### Last Rank Stalling
- Deque-based output buffering
- Early output processing with `async_batch_depth > 0`

### Memory Usage
- `pp_loop_size` buffers determine peak memory
- Larger async_batch_depth = more memory overhead for buffers
- Each microbatch's hidden states retained until received by next stage

---

## PPBatchMetadata

```python
@dataclass
class PPBatchMetadata:
    can_run_cuda_graph: bool  # Whether this batch can use CUDA graphs
```

CUDA graphs can be used for decode batches even in PP mode, as long as the batch shape is captured. The metadata flag indicates per-microbatch eligibility.
