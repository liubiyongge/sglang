# T008: Scheduler Event Loop Analysis

**Source**: `python/sglang/srt/managers/scheduler.py` (lines 1066-1145, 1170-1282, 1824-1904, 2155-2219)
**Cross-references**: [engine-startup.md](engine-startup.md), [ipc-topology.md](ipc-topology.md)

---

## Overview

The SGLang scheduler operates as a continuous event loop orchestrating GPU utilization, request batching, and token generation. It supports two distinct modes: **normal** (synchronous) and **overlap** (asynchronous GPU/CPU pipelining). The scheduler is the central coordinator between the tokenizer manager (which sends requests) and the model executor (which runs GPU forward passes).

---

## Two Event Loop Modes

### Normal Mode (`event_loop_normal`, line 1066-1090)

```python
def event_loop_normal(self):
    while True:
        recv_reqs = self.recv_requests()
        self.process_input_requests(recv_reqs)
        if self._engine_paused:
            continue

        batch = self.get_next_batch_to_run()
        self.cur_batch = batch

        if batch:
            result = self.run_batch(batch)
            self.process_batch_result(batch, result)
        else:
            self.self_check_during_idle()

        self.last_batch = batch
```

**Characteristics:**
- Strictly sequential: GPU forward blocks until complete
- Lower TTFT: Results processed immediately
- Simpler reasoning: No deferred state
- Lower throughput: CPU idle during GPU compute

### Overlap Mode (`event_loop_overlap`, line 1093-1144)

```python
def event_loop_overlap(self):
    self.result_queue: Deque[Tuple[ScheduleBatch, Result]] = deque()

    while True:
        recv_reqs = self.recv_requests()
        self.process_input_requests(recv_reqs)
        if self._engine_paused:
            continue

        batch = self.get_next_batch_to_run()
        self.cur_batch = batch
        disable_overlap_for_batch = self.is_disable_overlap_for_batch(batch)

        if disable_overlap_for_batch:
            pop_and_process()  # Process previous result immediately

        if batch:
            batch_result = self.run_batch(batch)
            self.result_queue.append((batch.copy(), batch_result))
        else:
            batch_result = None

        if self.last_batch:
            if not disable_overlap_for_batch:
                pop_and_process()  # Process deferred result
        elif batch is None:
            self.self_check_during_idle()

        if self.is_generation:
            self.launch_batch_sample_if_needed(batch_result)

        self.last_batch = batch
```

**Characteristics:**
- Asynchronous: CPU processes previous results while GPU computes next batch
- Higher throughput: GPU/CPU overlap maximizes utilization
- Deferred processing via `result_queue`
- CUDA stream separation enables parallelism

---

## GPU/CPU Overlap Architecture

### Stream Management (initialized in `init_overlap`, line 965-991)

| Stream | Purpose |
|--------|---------|
| `default_stream` | CPU synchronization and control operations |
| `forward_stream` | GPU forward computation (runs independently) |
| `copy_stream` | Explicit GPU→CPU data transfers |

### Execution Flow

```
Iteration N:
  CPU: recv_requests → get_next_batch → prepare → submit to forward_stream
  GPU: forward_stream executes batch N
  
Iteration N+1:
  CPU: process batch N-1 results (from result_queue)
  GPU: continues executing batch N (overlaps with CPU processing)
  CPU: recv_requests → get_next_batch → prepare batch N+1
```

### Future Map (`FutureMap`)

- Manages deferred GPU tensor results
- Maps future indices to actual values populated by forward stream
- `resolve_future()`: Replaces futures with actual tensors before forward
- `store_to_map()`: Stores output for later retrieval

### Overlap Disabling Conditions (`is_disable_overlap_for_batch`, line 1146-1168)

1. **Consecutive prefills**: Two EXTEND batches in a row - disabled to improve TTFT
   - Controlled by `SGLANG_DISABLE_CONSECUTIVE_PREFILL_OVERLAP`
2. **Grammar + speculative decoding**: Grammar depends on previous batch sampling results

---

## Single Iteration Step-by-Step

### Step 1: Request Reception (`recv_requests`, line 1175-1282)

- Non-blocking ZMQ receive from tokenizer manager (only at `tp_rank=0`)
- Receives from both tokenizer channel and RPC control channel
- Bounded by `max_recv_per_poll` to prevent starvation
- Broadcasts received requests to all TP ranks

### Step 2: Request Processing (`process_input_requests`, line 1313-1332)

Type-based dispatch via `_request_dispatcher`:
- `TokenizedGenerateReqInput` → add to waiting queue
- `TokenizedEmbeddingReqInput` → add to waiting queue
- `AbortReq` → remove from running/waiting
- Control requests → weight updates, LoRA loading, metrics queries

### Step 3: Batch Selection (`get_next_batch_to_run`, line 1824-1904)

```
1. Merge last_batch (if EXTEND completed) into running_batch
2. Try get_new_batch_prefill() → returns new prefill batch if available
3. If prefill available → return prefill batch (PRIORITY: prefill first)
4. If no prefill → return running_batch for decode
5. If no work → return None (idle)
```

**Prefill priority rationale:**
- Ensures new requests make progress (fairness)
- Amortizes prefill cost over many decode tokens
- TTFT is critical for user experience

### Step 4: Batch Execution (`run_batch`, line 2230-2382)

- Creates `ModelWorkerBatch` from `ScheduleBatch`
- In overlap mode: submits to `forward_stream`, returns futures
- In normal mode: synchronous forward, returns actual tensors
- Records timing metrics for prefill batches

### Step 5: Result Processing (`process_batch_result`, line 2399-2419)

Dispatches by forward mode:
- **EXTEND**: Process prefill results, update cache, check completion
- **DECODE**: Process single generated token per request, check stop conditions
- **IDLE**: No-op (DP attention synchronization only)

---

## Decode Retraction Mechanism (line 2155-2219)

### Trigger

When `not batch.check_decode_mem()` — KV cache cannot allocate tokens for next decode step.

### Algorithm (`retract_decode` in schedule_batch.py)

1. Sort requests by retraction priority: `(output_tokens_count, -input_tokens_count)` descending
2. Pop requests from back of sorted list until `check_decode_mem()` passes
3. Always keep at least 1 request
4. Release KV cache of retracted requests (without reinserting to tree cache)
5. Return retracted requests to waiting queue for re-prefill

### Retraction Priority

Requests retracted first:
- Fewest output tokens (least progress lost)
- Longest input (most expensive to keep)

### Dynamic Token Ratio Adjustment

```python
# After retraction: increase ratio (conservative)
new_estimate_ratio = (total_decoded + RETRACT_DECODE_STEPS * batch_size) / (total_max_new + 1)
self.new_token_ratio = new_estimate_ratio

# During normal operation: gradually decrease toward minimum
self.new_token_ratio = max(
    self.new_token_ratio - self.new_token_ratio_decay,
    self.min_new_token_ratio,
)
```

**Effects:**
- High ratio (0.8-1.0): Conservative, reserves more decode memory, fewer prefill requests
- Low ratio (0.2-0.4): Aggressive, more prefill concurrency, risks retraction

**Initial values:**
- `init_new_token_ratio = 0.6 * schedule_conservativeness`
- `min_new_token_ratio = init * 0.5`
- `decay = (init - min) / 40` (over 40 iterations)

---

## Key State Variables

### Core State (`init_running_status`, line 721-735)

| Variable | Type | Purpose |
|----------|------|---------|
| `waiting_queue` | `List[Req]` | Queued requests waiting for prefill |
| `running_batch` | `ScheduleBatch` | Currently executing decode batch |
| `cur_batch` | `Optional[ScheduleBatch]` | Batch in current iteration |
| `last_batch` | `Optional[ScheduleBatch]` | Batch from previous iteration |
| `chunked_req` | `Optional[Req]` | Request being chunked across iterations |
| `new_token_ratio` | `float` | Estimated output token ratio for memory |
| `forward_ct` | `int` | Total forward passes executed |

### Overlap State (`init_overlap`, line 965-991)

| Variable | Type | Purpose |
|----------|------|---------|
| `result_queue` | `Deque[Tuple[Batch, Result]]` | Deferred forward results |
| `forward_stream` | `CudaStream` | GPU computation stream |
| `copy_stream` | `CudaStream` | Data transfer stream |
| `future_map` | `FutureMap` | Maps futures to actual GPU results |
| `batch_record_buf` | `List[ModelWorkerBatch]` | Double buffer (prevents GC) |

---

## Idle State Management

### Detection

Idle when: `waiting_queue` empty AND `running_batch` empty AND no `chunked_req`.

### Actions (`self_check_during_idle`)

- Memory cleanup and garbage collection
- Cache maintenance (evict stale entries)
- State reset for temporary buffers
- Optional sleep (`idle_sleeper.maybe_sleep()`) with ZMQ interrupt on new requests

---

## Trade-off Summary

| Aspect | Normal Mode | Overlap Mode |
|--------|-------------|--------------|
| TTFT | Lower (immediate) | Higher (deferred) |
| Throughput | Lower (serial) | Higher (pipelined) |
| Complexity | Simple | Complex (streams, futures) |
| Memory | Single batch | Double buffering |
| Best for | Short queries, latency SLAs | High throughput, long contexts |
