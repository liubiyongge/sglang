# T012: Batch Data Structures Analysis

**Source**: `python/sglang/srt/managers/schedule_batch.py`, `python/sglang/srt/model_executor/forward_batch_info.py`
**Cross-references**: [scheduler-event-loop.md](scheduler-event-loop.md), [model-execution.md](model-execution.md)

---

## Overview

SGLang implements a three-layer batch hierarchy that progressively refines request data from high-level scheduling abstractions to low-level GPU tensors:

```
ScheduleBatch (CPU) → ModelWorkerBatch (CPU→GPU bridge) → ForwardBatch (GPU tensors)
```

Each layer serves a distinct purpose: scheduling decisions, data transport, and GPU execution.

---

## Layer 1: ScheduleBatch

### Purpose

Managed by `scheduler.py::Scheduler`. Contains high-level scheduling data, manages request lifecycle, and handles memory allocation. Most data resides on CPU.

### Core Fields

```python
@dataclasses.dataclass
class ScheduleBatch:
    reqs: List[Req]                         # Active requests in batch
    forward_mode: ForwardMode               # EXTEND, DECODE, MIXED, IDLE
    batch_is_full: bool                     # No more requests can be added

    # Memory infrastructure
    req_to_token_pool: ReqToTokenPool       # Maps req_idx → token indices
    token_to_kv_pool_allocator: Allocator   # KV cache allocator
    tree_cache: BasePrefixCache             # Radix prefix cache

    # Batched tensors (CPU-side)
    input_ids: torch.Tensor                 # [total_tokens], flattened input
    req_pool_indices: torch.Tensor          # [batch_size], req pool slots
    seq_lens: torch.Tensor                  # [batch_size], sequence lengths
    out_cache_loc: torch.Tensor             # [total_tokens], KV cache indices
    output_ids: torch.Tensor                # [batch_size], current output tokens
```

### Extend-Specific Fields

```python
    prefix_lens: List[int]                  # Cached prefix length per request
    extend_lens: List[int]                  # New tokens to process per request
    extend_num_tokens: int                  # Sum of extend_lens
    extend_logprob_start_lens: List[int]    # Logprob computation start per request
```

### Lifecycle Methods

| Method | Purpose |
|--------|---------|
| `init_new(reqs, ...)` | Create from list of requests |
| `prepare_for_extend()` | Set up for prefill phase (allocate KV) |
| `prepare_for_decode()` | Set up for decode phase (allocate 1 token/req) |
| `filter_batch(keep_indices)` | Remove finished requests |
| `merge_batch(other)` | Combine two batches (for mixed mode) |
| `mix_with_running(running)` | Combine extend + decode for chunked prefill |
| `get_model_worker_batch()` | Create ModelWorkerBatch for GPU workers |
| `retract_decode(server_args)` | Retract requests under memory pressure |
| `check_decode_mem()` | Check if decode can proceed |

---

## Layer 2: ModelWorkerBatch

### Purpose

Managed by `tp_worker.py::TpModelWorker`. Minimal data carrier containing only forward-pass-relevant data. Bridges CPU scheduler and GPU model runner.

### Fields (subset of ScheduleBatch)

```python
@dataclasses.dataclass
class ModelWorkerBatch:
    forward_mode: ForwardMode
    input_ids: torch.Tensor
    req_pool_indices: torch.Tensor
    seq_lens: torch.Tensor
    seq_lens_cpu: torch.Tensor
    out_cache_loc: torch.Tensor
    seq_lens_sum: int

    # Extend-specific
    extend_num_tokens: Optional[int]
    extend_seq_lens: Optional[List[int]]
    extend_prefix_lens: Optional[List[int]]

    # Sampling and constraints
    sampling_info: SamplingBatchInfo
    has_grammar: bool
    reqs: Optional[List[Req]]              # Only if grammar needed

    # Distributed
    global_num_tokens: Optional[List[int]]
    is_extend_in_batch: bool
    can_run_dp_cuda_graph: bool

    # Specialized
    spec_info: Optional[SpecInput]
    lora_ids: Optional[List[str]]
    multimodal_inputs: Optional[List[MultimodalInputs]]
```

### Creation

```python
# ScheduleBatch.get_model_worker_batch():
if self.forward_mode.is_decode_or_idle():
    extend_seq_lens = extend_prefix_lens = None
else:
    extend_seq_lens = self.extend_lens
    extend_prefix_lens = self.prefix_lens
```

---

## Layer 3: ForwardBatch

### Purpose

Managed by `model_runner.py::ModelRunner`. Contains GPU tensors ready for model forward passes. All data on GPU device.

### Core GPU Tensors

```python
@dataclass
class ForwardBatch:
    forward_mode: ForwardMode
    batch_size: int
    input_ids: torch.Tensor                 # [total_tokens], GPU
    positions: torch.Tensor                 # [total_tokens], GPU
    req_pool_indices: torch.Tensor          # [batch_size], GPU
    seq_lens: torch.Tensor                  # [batch_size], GPU
    out_cache_loc: torch.Tensor             # [total_tokens], GPU
    seq_lens_sum: int

    # Memory pool references
    req_to_token_pool: ReqToTokenPool
    token_to_kv_pool: KVCache
    attn_backend: AttentionBackend
```

### Extend-Specific GPU Tensors

```python
    extend_num_tokens: Optional[int]
    extend_seq_lens: Optional[torch.Tensor]     # [batch_size], int32, GPU
    extend_prefix_lens: Optional[torch.Tensor]  # [batch_size], int32, GPU
    extend_start_loc: Optional[torch.Tensor]    # [batch_size], cumsum offsets
```

### Position Computation

```python
# EXTEND mode:
positions, extend_start_loc = compute_position(
    backend, extend_prefix_lens, extend_seq_lens, extend_num_tokens
)

# DECODE mode:
positions = clamp_position(seq_lens - 1)  # Last position only
```

### Initialization

```python
@classmethod
def init_new(cls, batch: ModelWorkerBatch, model_runner: ModelRunner):
    # 1. Transfer CPU tensors to GPU (non_blocking=True)
    # 2. Compute positions from prefix/seq lens
    # 3. Set up LoRA and encoder-decoder
    # 4. Initialize hidden states for split prefill
    # 5. Handle speculative decoding metadata
```

---

## The Req Class

### Core Identity

```python
class Req:
    rid: str                                # Request ID
    origin_input_ids: List[int]             # Original tokenized input
    output_ids: List[int]                   # Generated tokens (grows)
    fill_ids: List[int]                     # origin_input_ids + output_ids
```

### Memory Management

```python
    prefix_indices: torch.Tensor            # KV cache indices for cached prefix
    last_node: Any                          # Deepest matched radix tree node
    req_pool_idx: Optional[int]             # Slot in req_to_token_pool
    kv_committed_len: int                   # KV cache committed length
    kv_allocated_len: int                   # KV cache allocated length
```

### Scheduling State

```python
    extend_input_len: int                   # Tokens to process this iteration
    extend_batch_idx: int                   # Times in extend batch
    decode_batch_idx: int                   # Times in decode batch
    is_retracted: bool                      # Currently preempted
    retraction_count: int                   # Total retractions
```

### Completion Tracking

```python
    finished_reason: Optional[BaseFinishReason]
    # FINISH_MATCHED_TOKEN, FINISH_MATCHED_STR, FINISH_LENGTH, FINISH_ABORT
```

### HiCache Support

```python
    last_host_node: Any                     # Host-side cache node
    host_hit_length: int                    # Tokens from host cache
    storage_hit_length: int                 # Tokens from L3 storage
    cached_tokens_device: int               # GPU cache hits
    cached_tokens_host: int                 # Host cache hits
    cached_tokens_storage: int              # Storage cache hits
```

---

## ForwardMode Enum

```python
class ForwardMode(IntEnum):
    EXTEND = 1          # Prefill: process new tokens with cached prefix
    DECODE = 2          # Generate: one token per request
    MIXED = 3           # Both EXTEND and DECODE (chunked prefill)
    IDLE = 4            # Empty batch for DP attention sync

    # Speculative decoding
    TARGET_VERIFY = 5   # Verify target model predictions
    DRAFT_EXTEND = 6    # Draft model prefill
    DRAFT_EXTEND_V2 = 7

    # Other
    PREBUILT = 8        # Disaggregated decode (KV ready)
    SPLIT_PREFILL = 9   # Pipeline parallel chunked prefill
    DLLM_EXTEND = 10    # Diffusion LLM prefill
```

### Key Predicates

| Predicate | Modes | Use |
|-----------|-------|-----|
| `is_extend()` | EXTEND, MIXED, DRAFT_EXTEND, TARGET_VERIFY, SPLIT_PREFILL, DLLM_EXTEND | Variable-length forward |
| `is_decode()` | DECODE | Single-token forward |
| `is_cuda_graph()` | DECODE, TARGET_VERIFY, IDLE, DLLM_EXTEND | Fixed-shape execution |
| `is_mixed()` | MIXED | Combined extend+decode |
| `is_idle()` | IDLE | Empty sync batch |

---

## Data Flow: Request Lifecycle

```
1. Request Created
   → origin_input_ids set, added to waiting_queue

2. Prefix Matching (init_next_round_input)
   → prefix_indices matched from radix cache
   → extend_input_len = len(fill_ids) - len(prefix_indices)

3. Batch Formation (PrefillAdder.add_one_req)
   → req added to can_run_list
   → KV budget reserved

4. Extend Phase (prepare_for_extend)
   → req_pool_idx allocated
   → KV cache allocated for extend_input_len tokens
   → kv_committed_len updated

5. Forward Pass
   → ModelWorkerBatch → ForwardBatch → GPU
   → Logits computed, tokens sampled

6. Result Processing
   → output_ids appended with new token
   → check_finished() evaluated

7. Decode Phase (prepare_for_decode) [if not finished]
   → input_ids = [last output token]
   → 1 token KV cache allocated
   → Loop back to step 5

8. Completion
   → filter_batch() removes from batch
   → KV cache released or inserted into tree cache
```

---

## Memory Tracking Through Layers

```
Req.req_pool_idx ─────────────────────────────────────────┐
                                                           │
ScheduleBatch.req_pool_indices[i] = req.req_pool_idx ─────┤
                                                           │
ModelWorkerBatch.req_pool_indices[i] ─────────────────────┤
                                                           │
ForwardBatch.req_pool_indices[i] ──────────────────────── ↓
                                                           
req_to_token_pool[pool_idx, :seq_len] = token_indices
                                           │
                                           ↓
token_to_kv_pool[token_indices] = [K, V] GPU tensors
```

---

## Mixed Batch Formation

When chunked prefill is active, EXTEND and DECODE requests can be combined:

```python
def mix_with_running(self, running_batch):
    # Update running requests: fill_ids, extend_input_len = 1
    for req in running_batch.reqs:
        req.fill_ids = req.origin_input_ids + req.output_ids
        req.extend_input_len = 1  # Decode 1 more token

    # Merge tensors
    self.input_ids = cat([extend_input_ids, decode_input_ids])
    self.out_cache_loc = cat([extend_cache_loc, decode_cache_loc])
    self.prefix_lens.extend(decode_seq_lens)
    self.extend_lens.extend([1] * len(running_batch.reqs))
    self.forward_mode = ForwardMode.MIXED
```
