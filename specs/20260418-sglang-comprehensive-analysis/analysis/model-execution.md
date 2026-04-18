# T013: Model Execution Pipeline Analysis

**Source**: `python/sglang/srt/model_executor/model_runner.py`, `python/sglang/srt/model_executor/cuda_graph_runner.py`, `python/sglang/srt/managers/tp_worker.py`
**Cross-references**: [batch-data-structures.md](batch-data-structures.md), [scheduler-event-loop.md](scheduler-event-loop.md)

---

## Overview

The model execution pipeline transforms scheduler batches into GPU computations and returns logits/sampled tokens. The pipeline consists of three key components:
1. **TpModelWorker**: Receives batches from scheduler, coordinates forward passes
2. **ModelRunner**: Manages model forward execution, CUDA graphs, and attention backends
3. **CudaGraphRunner**: Captures and replays CUDA graphs for decode efficiency

---

## Full GPU Execution Flow

```
Scheduler
  ↓ ModelWorkerBatch
TpModelWorker.forward_batch_generation()
  ↓ ForwardBatch.init_new() [CPU→GPU transfer]
ModelRunner.forward()
  ↓ _forward_raw()
  ├── CUDA Graph? → CudaGraphRunner.replay()
  ├── EXTEND? → forward_extend() [or piecewise graph]
  ├── DECODE? → forward_decode()
  └── IDLE? → forward_idle()
  ↓ LogitsProcessorOutput
TpModelWorker: sample() or compute_logprobs_only()
  ↓ GenerationBatchResult
Scheduler: process_batch_result()
```

---

## TpModelWorker: Scheduler-to-GPU Bridge

### Entry Point: `forward_batch_generation(ModelWorkerBatch)`

```python
def forward_batch_generation(self, model_worker_batch):
    # 1. Create ForwardBatch (transfers to GPU)
    forward_batch = ForwardBatch.init_new(model_worker_batch, self.model_runner)

    # 2. Execute model forward
    out = self.model_runner.forward(forward_batch, ...)
    logits_output = out.logits_output
    can_run_cuda_graph = out.can_run_graph

    # 3. Sample next tokens (for generation, non-prefill-only)
    if logits_output is not None and not model_worker_batch.is_prefill_only:
        batch_result.next_token_ids = self.model_runner.sample(logits_output, forward_batch)
    else:
        batch_result.next_token_ids = torch.zeros(len(seq_lens), ...)

    # 4. Compute logprobs if requested
    if return_logprob and logits_output.next_token_logits is not None:
        self.model_runner.compute_logprobs_only(logits_output, model_worker_batch)

    return batch_result
```

### Pipeline Parallelism (non-last ranks)

Non-last PP ranks return hidden states instead of logits:

```python
if not self.pp_group.is_last_rank:
    return GenerationBatchResult(
        pp_hidden_states_proxy_tensors=pp_proxy_tensors,
        can_run_cuda_graph=can_run_cuda_graph,
    )
```

---

## ModelRunner: Forward Dispatch

### Main Entry: `_forward_raw(forward_batch)`

```python
def _forward_raw(self, forward_batch, ...):
    # Check CUDA graph eligibility
    mode_check = forward_batch.forward_mode.is_cuda_graph()
    can_run_graph = bool(
        mode_check and self.graph_runner and self.graph_runner.can_run(forward_batch)
    )

    if can_run_graph:
        # FAST PATH: Graph replay
        ret = self.graph_runner.replay(forward_batch, ...)
        return ModelRunnerOutput(logits_output=ret, can_run_graph=True)

    # SLOW PATH: Mode-specific dispatch
    if forward_batch.forward_mode.is_decode():
        ret = self.forward_decode(forward_batch)
    elif forward_batch.forward_mode.is_extend():
        ret, can_run_graph = self.forward_extend(forward_batch)
    elif forward_batch.forward_mode.is_idle():
        ret = self.forward_idle(forward_batch)
```

### Extend (Prefill) Forward

```python
def forward_extend(self, forward_batch, ...):
    # Try piecewise CUDA graph (layer-by-layer capture)
    can_run_graph = (
        self.piecewise_cuda_graph_runner is not None and
        self.piecewise_cuda_graph_runner.can_run(forward_batch)
    )

    if can_run_graph:
        return self.piecewise_cuda_graph_runner.replay(forward_batch), True

    # Direct model forward
    self.attn_backend.init_forward_metadata(forward_batch)
    return self.model.forward(
        forward_batch.input_ids,
        forward_batch.positions,
        forward_batch,
    ), False
```

### Decode Forward

```python
def forward_decode(self, forward_batch, ...):
    self.attn_backend.init_forward_metadata(forward_batch)
    return self.model.forward(
        forward_batch.input_ids,
        forward_batch.positions,
        forward_batch,
    )
```

---

## CUDA Graph System

### Batch Size Bucketing

```python
def get_batch_sizes_to_capture():
    capture_bs = server_args.cuda_graph_bs  # Default: [1, 2, 4, 8, 16, 32]
    # Adjust for alignment requirements (attention TP size)
    # Separate torch.compile batches (≤ torch_compile_max_bs)
    return capture_bs, compile_bs
```

### Capture Process

```python
def capture(self):
    for bs in reversed(self.capture_bs):  # Large to small (memory sharing)
        with freeze_gc():
            with graph_capture() as ctx:
                # Warmup runs (2 iterations)
                for _ in range(2):
                    device_module.synchronize()
                    tp_group.barrier()
                    run_once()

                # Actual capture
                with device_module.graph(cuda_graph=graph, pool=global_memory_pool):
                    output_buffers = run_once()

                self.graphs[bs] = graph
                self.output_buffers[bs] = output_buffers
```

**Key design decisions:**
- Reverse capture order: Large→small enables memory reuse
- Global memory pool: Shared across all captured graphs
- GC frozen: Prevents interference during capture
- 2 warmup iterations: Populates CUDA caches

### Batch Size Selection (Bisect)

```python
def replay_prepare(self, forward_batch):
    raw_bs = forward_batch.batch_size
    index = bisect.bisect_left(self.capture_bs, raw_bs)
    bs = self.capture_bs[index]  # Smallest captured size >= actual
```

### Padding for Captured Size

```python
buffers.populate_from_forward_batch(
    forward_batch=forward_batch,
    raw_bs=raw_bs,           # Actual batch size
    bs=bs,                   # Captured (padded) batch size
    seq_len_fill_value=...,  # Fill padding positions
)
```

- Pad `seq_lens` with fill value (backend-specific, e.g., 1024)
- Pad `input_ids` with valid token ID
- Extend batch to captured size with dummy requests

### Replay

```python
def replay(self, forward_batch, ...):
    self.replay_prepare(forward_batch)  # Copy data into captured buffers
    self.graphs[bs].replay()            # Execute captured graph

    # Extract real results (discard padding)
    output = self.output_buffers[bs]
    return LogitsProcessorOutput(
        next_token_logits=output.next_token_logits[:raw_num_token],
        hidden_states=output.hidden_states[:raw_num_token] if ... else None,
    )
```

### Piecewise CUDA Graphs (for EXTEND)

Standard CUDA graphs don't work for EXTEND (variable lengths). Piecewise graphs capture individual model layers:

- Each layer captured as separate graph
- Flexible on total token count
- Captures attention metadata per-layer
- Slower than full graph but faster than no graph

---

## Extend vs Decode: Execution Differences

| Aspect | EXTEND (Prefill) | DECODE |
|--------|-------------------|--------|
| Tokens per request | Variable (extend_input_len) | 1 |
| Total tokens | Often large | = batch_size |
| KV computation | All new positions | Only last position |
| Attention pattern | Dense (all-to-all) | Query→full KV |
| Compute bound | FLOPs (compute) | Bandwidth (KV read) |
| CUDA graph | Piecewise or none | Standard (fast) |
| Position | `compute_position(prefix_lens, extend_lens)` | `clamp(seq_lens - 1)` |

---

## Memory Management During Forward

### Pre-allocated Buffers (GraphInputBuffers)

```python
GraphInputBuffers.create(
    max_bs=max_captured_batch_size,
    max_num_token=max_bs * num_tokens_per_bs,
    hidden_size=...,
    vocab_size=...,
)
```

Contains:
- `input_ids`: [max_num_token]
- `positions`: [max_num_token]
- `seq_lens`: [max_bs]
- `out_cache_loc`: [max_num_token]
- `next_token_logits_buffer`: [max_num_token, vocab_size] or [max_num_token]

### In-place Operations

CUDA graph replay uses in-place buffer copies:
```python
buffers.input_ids[:num_tokens].copy_(forward_batch.input_ids)
buffers.positions[:num_tokens].copy_(forward_batch.positions)
buffers.seq_lens[:bs].copy_(forward_batch.seq_lens)
buffers.out_cache_loc[:num_tokens].copy_(forward_batch.out_cache_loc)
```

Benefits:
- No allocation during inference
- Buffers allocated once at startup
- Same buffers reused every iteration

### KV Cache Access Pattern

```python
# ForwardBatch provides:
req_pool_indices[i]  → req_to_token_pool[pool_idx, :seq_len] = token_indices
out_cache_loc[j]     → token_to_kv_pool[cache_idx] = new K, V

# During attention:
# Read: All previous positions via req_to_token_pool
# Write: New positions via out_cache_loc
```

---

## Performance Optimizations Summary

### 1. CUDA Graph Capture/Replay
- Eliminates CPU overhead for decode (kernel launch, memory allocation)
- Batch size bucketing covers common sizes
- Padding overhead minimal (<5% extra compute)

### 2. Tensor Reuse
- Pre-allocated `GraphInputBuffers`
- In-place copy operations
- Double buffering in overlap mode

### 3. Attention Backend Selection
- 25+ backends (FlashAttention, FlashInfer, Triton, MLA variants)
- Backend-specific metadata initialization
- Per-mode optimization (decode vs extend)

### 4. Torch.compile Integration
- Selective compilation for small batch sizes (≤ `torch_compile_max_bs`)
- Kernel fusion opportunities
- Model patching during capture

### 5. Stream Management (Overlap Mode)
- Forward stream for GPU computation
- Default stream for CPU control
- Copy stream for GPU→CPU transfers
- Events for synchronization

### 6. Pipeline Parallelism
- Hidden state proxy tensors between PP ranks
- Only last rank computes logits
- Split prefill for memory-efficient long contexts

---

## Speculative Decoding Integration

### Skip Backend Reinitialization

```python
def replay(self, forward_batch, skip_attn_backend_init=False):
    if not skip_attn_backend_init:
        self.replay_prepare(forward_batch)
    else:
        # Fast path: only copy data, reuse metadata
        self.buffers.input_ids[:num_token].copy_(forward_batch.input_ids)
        self.buffers.positions[:num_token].copy_(forward_batch.positions)
    self.graphs[bs].replay()
```

For multi-step draft execution:
- Same graph replayed multiple times
- Only input data changes between steps
- Metadata (seq_lens, cache_loc) unchanged
- Significant speedup for speculative verification

---

## Distributed Coordination

### TP Synchronization

All TP ranks execute the same CUDA graph:
- Barriers during capture ensure consistency
- AllReduce embedded in captured graph
- Same batch broadcast to all ranks

### DP Attention

```python
# Set up global buffer for collective MLP
prepare_mlp_sync_batch():
    global_num_tokens = [tokens_per_rank...]
    buffer_len = max(global_num_tokens) * num_ranks  # or sum
    set_dp_buffer_len(buffer_len)
```

### Expert Load Balancing (EPLB)

```python
if elastic_ep_state.needs_rebalance():
    eplb_manager.rebalance()
    # Re-run forward with new expert distribution
    output = self._forward_raw(forward_batch)
```
