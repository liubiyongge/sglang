# T010: Batch Formation (PrefillAdder) Analysis

**Source**: `python/sglang/srt/managers/schedule_policy.py` (lines 372-890)
**Cross-references**: [scheduling-policies.md](scheduling-policies.md), [scheduler-event-loop.md](scheduler-event-loop.md)

---

## Overview

The `PrefillAdder` class implements the batch formation algorithm that determines which requests from the waiting queue can be admitted to a prefill batch. It manages multiple token budgets, handles chunked prefill for long sequences, and coordinates with the memory allocator to prevent OOM.

---

## Token Budget Model

The scheduler maintains four overlapping token budgets:

### Budget 1: `rem_total_tokens` (Total Available Capacity)

```python
rem_total_tokens = (
    token_to_kv_pool_allocator.available_size()
    + tree_cache.evictable_size()
    - rem_total_token_offset
)
```

- Represents total deployable KV cache tokens (available + evictable)
- Decremented as requests are admitted
- Most restrictive budget (prevents OOM)

### Budget 2: `rem_input_tokens` (Prefill-Specific)

- Maximum tokens allowed in a single prefill iteration
- Initialized from `max_prefill_tokens` (model worker configuration)
- Prevents prefill from consuming entire GPU memory
- Decremented by: `ceil_paged_tokens(extend_input_len)`

### Budget 3: `rem_chunk_tokens` (Chunked Prefill)

- Maximum tokens per chunk when splitting long sequences
- Initialized from `chunked_prefill_size`
- `None` when chunked prefill is disabled
- Enables multi-iteration prefill

### Budget 4: `rem_dllm_tokens` (Distributed LLM)

```python
rem_dllm_tokens = max_running_reqs * dllm_block_size
```

- Only active for DLLM mode
- Limits based on running requests * fixed block size

---

## Main Request Addition: `add_one_req()`

```python
def add_one_req(self, req, has_chunked_req, truncation_align_size):
    # 1. Context parallelism constraint
    if nsa_prefill_cp_in_seq_split and len(can_run_list) >= 1:
        return AddReqResult.OTHER

    # 2. Max requests per prefill batch
    if prefill_max_requests and len(can_run_list) >= prefill_max_requests:
        return AddReqResult.OTHER

    # 3. Estimate total tokens (input + estimated decode)
    total_tokens = (
        req.extend_input_len +
        min(req.sampling_params.max_new_tokens - len(req.output_ids),
            CLIP_MAX_NEW_TOKENS)
    )

    # 4. Check capacity
    if total_tokens >= rem_total_tokens:
        return AddReqResult.NO_TOKEN

    # 5. Lock cache node and re-check
    with _lock_node(req.last_node):
        # 6. Host page loading (HiCache)
        if req.host_hit_length > 0:
            new_indices, req.last_node = tree_cache.init_load_back(...)
            req.prefix_indices = cat([req.prefix_indices, new_indices])

        # 7. Dispatch to handler
        if rem_chunk_tokens is None or input_tokens <= rem_chunk_tokens:
            # Non-chunked: add entire request
            can_run_list.append(req)
            _update_prefill_budget(prefix_len, input_tokens, max_new_tokens)
        else:
            # Chunked: truncate to chunk size
            trunc_len = rem_chunk_tokens // page_size * page_size
            req.set_extend_input_len(trunc_len)
            can_run_list.append(req)
            new_chunked_req = req
            _update_prefill_budget(prefix_len, trunc_len, 0)

    return budget_state()
```

### Budget State Check

```python
def budget_state(self):
    if self.rem_total_tokens <= 0 or self.cur_rem_tokens <= 0:
        return AddReqResult.NO_TOKEN    # OOM
    if self.rem_input_tokens <= 0:
        return AddReqResult.OTHER       # Prefill budget exhausted
    if self.rem_chunk_tokens is not None and self.rem_chunk_tokens <= 0:
        return AddReqResult.OTHER       # Chunk budget exhausted
    return AddReqResult.CONTINUE
```

### Return Values

| Result | Meaning | Scheduler Action |
|--------|---------|------------------|
| `CONTINUE` | More requests can be added | Keep iterating |
| `NO_TOKEN` | KV cache exhausted | Stop batch formation |
| `OTHER` | Budget limit reached | Stop batch formation |

---

## Chunked Prefill

### Purpose

Split long-context requests across multiple GPU forward passes to prevent:
- Excessive TTFT for other requests
- GPU memory spikes from single large prefill
- Decode starvation

### State Machine

```
Request with 10,000 tokens (chunk_size=2,048):

Iteration 1: extend_input_len = 2,048 → chunked_req = req
Iteration 2: extend_input_len = 2,048 → chunked_req = req
Iteration 3: extend_input_len = 2,048 → chunked_req = req
Iteration 4: extend_input_len = 2,048 → chunked_req = req
Iteration 5: extend_input_len = 1,904 → chunked_req = None (complete)
```

### Continuation: `add_chunked_req()`

```python
def add_chunked_req(self, req):
    _rem_tokens = min(rem_chunk_tokens, int(rem_total_tokens))
    truncated = req.extend_input_len > _rem_tokens
    req.set_extend_input_len(min(req.extend_input_len, _rem_tokens))
    can_run_list.append(req)
    return req if truncated else None
```

### Alignment

```python
trunc_len = (rem_chunk_tokens // page_size) * page_size  # Page alignment
if truncation_align_size is not None:
    trunc_len = truncation_align_size * (trunc_len // truncation_align_size)
```

### Configuration

- `--chunked-prefill-size <size>`: 0 or -1 = disabled, >0 = chunk size
- Dynamic chunking: `predict_next_chunk_size(history_len)` for adaptive sizing

---

## Token Estimation: `new_token_ratio`

### Role

Estimates how many output tokens each request will generate, to reserve KV cache space during prefill.

```python
estimated_decode_tokens = min(
    req.sampling_params.max_new_tokens - len(req.output_ids),
    CLIP_MAX_NEW_TOKENS  # Default: 4096
) * new_token_ratio
```

### Clipping (`CLIP_MAX_NEW_TOKENS`)

Prevents overly conservative estimation for requests with very large `max_new_tokens`:
- Default clip: 4096 tokens
- Only clips the estimation, not the actual generation limit
- Configurable via `SGLANG_CLIP_MAX_NEW_TOKENS_ESTIMATION`

---

## Special Cases

### Ignore EOS Mode (`add_one_req_ignore_eos`)

For requests with `ignore_eos=True`:
- Uses `new_token_ratio = 1.0` (assumes full generation)
- Verifies sufficient memory for all requests to complete
- Conservative check: iterates all requests computing worst-case memory

### DLLM Mode (`_add_dllm_req`)

For distributed LLM prefill:
- Separate token budget (`rem_dllm_tokens`)
- Different truncation logic based on block size
- Don't reserve `max_new_tokens` if truncated

---

## Memory Protection

### Cache Node Locking

```python
with _lock_node(req.last_node):
    # Prevents eviction of cached prefix during admission
    ...
```

### Eviction During Admission

```python
rem_total_tokens = (
    allocator.available_size()
    + tree_cache.evictable_size()  # Can be reclaimed on demand
    - offset
)
```

The admission check accounts for evictable cache, which will be reclaimed when actual allocation occurs.

---

## Integration with Scheduler

### Full Pipeline

```
1. calc_priority(waiting_queue, running_batch)    # Apply scheduling policy
2. Create PrefillAdder(budgets, running_batch)    # Initialize budgets
3. add_chunked_req(chunked_req) if exists         # Continue previous chunk
4. For each req in sorted waiting_queue:
     add_one_req(req, ...)                        # Admit or reject
     if result != CONTINUE: break
5. can_run_list → ScheduleBatch.init_new()        # Create batch
```

### Key Invariants

1. `rem_total_tokens` monotonically decreases within an iteration
2. Cache nodes locked during admission prevent concurrent eviction
3. Chunked requests always take priority (continue before new requests)
4. At least 1 request admitted if memory available (liveness guarantee)
