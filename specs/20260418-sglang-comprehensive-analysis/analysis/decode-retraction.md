# T011: Decode Retraction Mechanism Analysis

**Source**: `python/sglang/srt/managers/scheduler.py` (lines 2155-2219), `python/sglang/srt/managers/schedule_batch.py`
**Cross-references**: [scheduler-event-loop.md](scheduler-event-loop.md), [batch-formation.md](batch-formation.md)

---

## Overview

Decode retraction is SGLang's mechanism for handling KV cache exhaustion during the decode phase. When the running batch cannot allocate memory for the next decode step, the scheduler retracts (preempts) requests from the running batch, releasing their KV cache. Retracted requests return to the waiting queue for later re-prefill. The system also dynamically adjusts the `new_token_ratio` to prevent future retraction events.

---

## Trigger Condition

In `update_running_batch` (scheduler.py, line 2164):

```python
def update_running_batch(self, batch: ScheduleBatch):
    batch.filter_batch(v1_spec_info_filtered=True)

    if not batch.check_decode_mem():
        # MEMORY PRESSURE: Initiate retraction
        retracted_reqs, new_token_ratio, reqs_to_abort = batch.retract_decode(self.server_args)
        ...
    else:
        # NORMAL: Gradually decrease token ratio
        self.new_token_ratio = max(
            self.new_token_ratio - self.new_token_ratio_decay,
            self.min_new_token_ratio,
        )
```

`check_decode_mem()` checks whether the KV pool has enough space for all requests to generate their next token, including eviction from tree cache.

---

## Retraction Algorithm

### Step 1: Sort by Retraction Priority

```python
sorted_indices = list(range(len(self.reqs)))

if not server_args.speculative_algorithm:
    sorted_indices.sort(
        key=lambda i: (
            len(self.reqs[i].output_ids),        # More output → higher priority to keep
            -len(self.reqs[i].origin_input_ids),  # More input → cheaper to retract
        ),
        reverse=True,
    )
```

**Retraction order (from back of sorted list):**
- Fewest output tokens retracted first (least progress lost)
- Among equal output lengths, longer inputs retracted first (more KV freed)

### Step 2: Pop Until Memory Sufficient

```python
retracted_reqs = []
first_iter = True

while first_iter or (not self.check_decode_mem(selected_indices=sorted_indices)):
    if len(sorted_indices) == 1:
        break  # Always keep at least 1 request

    first_iter = False
    idx = sorted_indices.pop()
    req = self.reqs[idx]
    retracted_reqs.append(req)
    self.release_req(idx, len(sorted_indices), server_args)
```

### Step 3: Release KV Cache

```python
def release_req(self, idx, remaining_req_count, server_args):
    req = self.reqs[idx]

    # Offload if disaggregated decode
    if server_args.disaggregation_mode == "decode":
        req.offload_kv_cache(...)

    # Release without re-insertion into cache tree
    release_kv_cache(req, tree_cache, is_insert=False)

    # Reclaim evictable memory for remaining requests
    num_tokens = remaining_req_count * RETRACT_DECODE_STEPS
    evict_from_tree_cache(tree_cache, num_tokens)

    # Reset request state for re-prefill
    req.reset_for_retract()
```

### Step 4: Compute New Token Ratio

```python
total_decoded_tokens = sum(len(r.output_ids) for r in self.reqs)
total_max_new_tokens = sum(r.sampling_params.max_new_tokens for r in self.reqs)

new_estimate_ratio = min(1.0,
    (total_decoded_tokens + RETRACT_DECODE_STEPS * len(self.reqs)) / (total_max_new_tokens + 1)
)

return retracted_reqs, new_estimate_ratio, []
```

---

## Dynamic `new_token_ratio` Adjustment

### Initialization

```python
init_new_token_ratio = min(SGLANG_INIT_NEW_TOKEN_RATIO * schedule_conservativeness, 1.0)
# Default: 0.6 * conservativeness

min_new_token_ratio = min(init_new_token_ratio * MIN_NEW_TOKEN_RATIO_FACTOR, 1.0)
# Default: init * 0.5

new_token_ratio_decay = (init_new_token_ratio - min_new_token_ratio) / NEW_TOKEN_RATIO_DECAY_STEPS
# Default: decay over 40 iterations
```

### Runtime Behavior

```
                ┌─── Retraction event: jump to new_estimate_ratio
                │
    1.0 ─ ─ ─ ─│─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─
               /│\
              / │ \
ratio       /  │  \  gradual decay
           /   │   \
    0.3 ──/────│────\──────────── min_new_token_ratio
          │    │    │
     time─────────────────────→
```

**Two dynamics:**
1. **After retraction**: Ratio jumps up (conservative, prevents immediate re-retraction)
2. **Normal operation**: Ratio decays linearly toward minimum (allows more prefill)

### Effects on Batch Formation

| Ratio | Behavior | Risk |
|-------|----------|------|
| High (0.8-1.0) | Few prefill requests admitted | Underutilization |
| Medium (0.4-0.6) | Balanced admission | Normal operation |
| Low (0.2-0.3) | Many prefill requests admitted | Retraction risk |

---

## Request State After Retraction

```python
def reset_for_retract(self):
    self.prefix_indices = torch.empty((0,), dtype=torch.int64)
    self.is_retracted = True
    self.retracted_stain = True
    self.retraction_count += 1
    self.extend_input_len = 0
    self.kv_allocated_len = 0
    self.kv_committed_len = 0
    # ... resets 20+ fields
```

**After retraction, the request:**
- Returns to `waiting_queue`
- Must be fully re-prefilled (prefix cache hit may still apply)
- Retains `output_ids` (can continue from where it left off)
- Marked with `is_retracted=True` and incremented `retraction_count`

---

## Failure Mode Protection

### Minimum Request Guarantee

```python
if len(sorted_indices) == 1:
    break  # Never retract the last request
```

If even 1 request cannot fit, raise `ValueError("OOM even after retracting all other requests")`.

### Liveness

- Retracted requests re-enter waiting queue with higher priority (FCFS ordering preserved)
- `retracted_stain` flag prevents double-counting of cache statistics
- `output_ids` preserved: generation continues from last decoded token

---

## Configuration

| Parameter | Default | Effect |
|-----------|---------|--------|
| `SGLANG_INIT_NEW_TOKEN_RATIO` | 0.6 | Initial conservativeness |
| `MIN_NEW_TOKEN_RATIO_FACTOR` | 0.5 | Floor as fraction of init |
| `NEW_TOKEN_RATIO_DECAY_STEPS` | 40 | Iterations to reach minimum |
| `RETRACT_DECODE_STEPS` | 8 | Lookahead for ratio estimation |
| `SGLANG_TEST_RETRACT` | false | Force periodic retraction |
| `schedule_conservativeness` | 1.0 | Multiplier on init ratio |
