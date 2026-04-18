# Prefill-Decode Disaggregation Architecture

## Overview

SGLang implements Prefill-Decode (PD) Disaggregation, an architecture that separates the prefill phase (processing input tokens) from the decode phase (generating output tokens) across separate server clusters. This allows independent scaling and optimization of each phase—prefill is compute-bound while decode is memory-bandwidth-bound.

**Key Source Files:**
- `python/sglang/srt/disaggregation/prefill.py` - Prefill server logic
- `python/sglang/srt/disaggregation/decode.py` - Decode server logic
- `python/sglang/srt/disaggregation/common/conn.py` - Bootstrap server
- `python/sglang/srt/disaggregation/base/conn.py` - Base interfaces
- `python/sglang/srt/disaggregation/utils.py` - Transfer backend registry
- `python/sglang/srt/disaggregation/{mooncake,mori,nixl,ascend,fake}/` - Transfer backends

---

## Architecture Overview

```
Client Request
      │
      ├── bootstrap_host, bootstrap_room assigned
      │
      ▼
┌─────────────────────┐                    ┌─────────────────────┐
│   PREFILL SERVER    │    KV Transfer     │   DECODE SERVER     │
│                     │ ──────────────────> │                     │
│ • Process input     │  (Mooncake/NIXL/   │ • Generate tokens   │
│ • Compute KV cache  │   Mori/Ascend)     │ • Auto-regressive   │
│ • Transfer KV out   │                    │ • Return output     │
└─────────────────────┘                    └─────────────────────┘
         │                                           │
         └──────────── Bootstrap Server ─────────────┘
                    (routing coordination)
```

---

## Bootstrap Mechanism

### Bootstrap Server (CommonKVBootstrapServer)

**Source:** `python/sglang/srt/disaggregation/common/conn.py`

Central coordinator maintaining routing information for prefill and decode servers.

**API Endpoints:**

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/route` | PUT | Register prefill server (IP, port, rank info) |
| `/route` | GET | Query prefill server info for connection |

**Registration Process:**
```
1. Prefill KVManager initializes:
   POST /route
   Body: {role: "Prefill", attn_tp_rank, pp_rank, system_dp_rank, rank_ip, rank_port}

2. Bootstrap stores in: prefill_port_table[dp_group][attn_tp_rank][pp_rank]

3. Decode receivers query:
   GET /route?engine_rank=X&target_dp_group=Y&target_pp_rank=Z
   Response: {rank_ip, rank_port} for direct connection
```

### Bootstrap Room

A unique session identifier (`req.bootstrap_room`) that:
- Correlates request across prefill/decode
- Enables context corruption detection via metadata validation
- Generated when request enters prefill, used throughout lifecycle

---

## Request Lifecycle

### Prefill Side

```
PrefillBootstrapQueue:
  1. add(req) → Create KVSender, register with bootstrap server
  2. pop_bootstrapped() → Poll sender status
     - Bootstrapping → WaitingForInput
     - Allocate metadata_buffer_index
     - sender.init(num_pages, metadata_buffer_idx)
     - Move to waiting_queue

Forward pass execution

process_batch_result_disagg_prefill():
  3. send_kv_chunk(req, last_chunk=True)
     - Extract kv_indices from req_to_token pool
     - Convert to page indices
     - Handle hybrid models (Mamba/SWA)
     - Set metadata_buffers
     - sender.send(page_indices, state_indices)
  4. Move to inflight_queue

process_disagg_prefill_inflight_queue():
  5. Poll sender → Success
     - Release KV cache lock
     - Finish request
```

### Decode Side

```
DecodePreallocQueue:
  1. add(req) → Create KVReceiver, connect via bootstrap
  2. pop_preallocated()
     - Poll receiver → WaitingForInput
     - Pre-allocate KV cache on decode GPU
     - receiver.init(page_indices, metadata_buffer_idx, state_indices)
     - Move to transfer_queue

DecodeTransferQueue:
  3. pop_transferred()
     - Poll receiver → Success
     - Validate bootstrap_room (corruption detection)
     - Get metadata (output_ids, cached_tokens)
     - Move to waiting_queue

Decode execution:
  4. get_new_prebuilt_batch() from waiting_queue
  5. Create fake-completed prefill batch
  6. Merge into running batch
  7. Execute decode forward
```

---

## KV Transfer Backends

### Backend Registry

```python
class TransferBackend(Enum):
    MOONCAKE = "mooncake"
    MORI = "mori"
    NIXL = "nixl"
    ASCEND = "ascend"
    FAKE = "fake"
```

Factory pattern returns appropriate sender/receiver classes per backend.

### Transfer Data Structures

**KVArgs (Transfer Configuration):**
```
Source buffers (prefill):
  kv_data_ptrs: List[int]     - GPU memory pointers to KV cache
  kv_data_lens: List[int]     - Total buffer sizes
  kv_item_lens: List[int]     - Per-token KV size

Destination buffers (decode):
  aux_data_ptrs: List[int]    - Metadata buffer pointers
  aux_data_lens: List[int]    - Metadata buffer sizes

State buffers (hybrid models):
  state_data_ptrs: List[int]  - Mamba/SWA state pointers
  state_type: str             - "mamba", "swa", "nsa", "none"
  state_dim_per_tensor: List[int]  - Cross-TP slicing info

Parallelism info:
  decode_tp_size, kv_head_num, page_size, prefill_pp_size, pp_rank
```

**MetadataBuffers (Decode-side Output):**
```
Per-request buffers for first token:
  output_ids: [size, 16] int32            - First output tokens
  cached_tokens: [size, 16] int32         - Cached token counts
  output_token_logprobs_val: [size, 16]   - Logprobs
  bootstrap_room: [size, 8] int64         - Corruption detection
  output_hidden_states: [size, hidden]    - For speculative decoding
```

### Mooncake Backend (Primary)

**Source:** `python/sglang/srt/disaggregation/mooncake/conn.py`

```python
class MooncakeKVManager(CommonKVManager):
    def __init__(self):
        self.init_engine()                    # Get transfer engine
        self.register_buffer_to_engine()       # Register GPU buffers
        self.transfer_queues = [FastQueue()]   # Lock-free queues
        self.executors = [ThreadPoolExecutor()]  # Thread pools

class MooncakeKVSender(CommonKVSender):
    def send(self, kv_indices, state_indices):
        # 1. Get destination buffer info from decode (ZMQ)
        # 2. Group contiguous memory regions
        transfer_blocks = group_concurrent_contiguous(
            prefill_kv_indices, dst_kv_indices)
        # 3. Execute parallel transfer
        for src_addr, dst_addr, length in transfer_blocks:
            engine.batch_transfer_sync(session_id, [src], [dst], [len])
        # 4. Send metadata and state indices
```

**MHA vs MLA Transfer:**
- MHA: Separate K and V transfers per layer
- MLA: Unified KV representation, single transfer per layer

### Transfer Optimizations

1. **Contiguous grouping**: Groups consecutive page indices for bulk transfer
2. **Parallel layer transfer**: Thread pool processes layers concurrently
3. **Lock-free queues**: `FastQueue` for inter-thread communication
4. **Pre-allocation**: Decode side pre-allocates before transfer starts

---

## Synchronization Mechanisms

### All-Reduce Polls (Gloo)

Collective decision-making across all ranks:
```python
polls = poll_and_all_reduce([sender.poll() for sender in senders], gloo_group)
# Takes minimum → collective decision
# Any Failed → all Failed
```

Used for:
- Bootstrap handshake coordination
- Transfer status tracking
- Failure detection

### Metadata Buffer Index Allocation

- Shared allocator across all ranks
- Maps request to output metadata location
- Limited size → backpressure mechanism
- Freed after request completion

### Context Corruption Detection

```python
if actual_bootstrap_room != expected_bootstrap_room:
    error_msg = f"Context corruption: {expected} vs {actual}"
    # Abort request
```

---

## Hybrid Model Support

### Mamba (HybridLinearKVPool)

```python
state_type = "mamba"
state_indices = [req_to_token_pool.req_index_to_mamba_index_mapping[idx]]
# Single state index per request
# Cross-TP: state_dim_per_tensor for slicing
```

### SWA (Sliding Window Attention)

```python
state_type = "swa"
window_start = max(0, seq_len - window_size)
state_indices = window_kv_indices_swa  # Last window only
```

### NSA (Next State Augmentation)

```python
state_type = "nsa"
state_indices = kv_indices_full  # All indices transferred
```

---

## Scheduler Integration

### Prefill Scheduler Mixin

```python
def event_loop_normal_disagg_prefill(self):
    while True:
        recv_reqs = recv_requests()
        process_input_requests(recv_reqs)
        
        # Bootstrap newly arrived requests
        self.waiting_queue.extend(
            self.disagg_prefill_bootstrap_queue.pop_bootstrapped())
        
        # Get and run batch
        batch = get_next_disagg_prefill_batch_to_run()
        if batch:
            result = run_batch(batch)
            process_batch_result_disagg_prefill(batch, result)
        
        # Poll inflight transfers
        process_disagg_prefill_inflight_queue()
```

### Decode Scheduler Mixin

```python
def event_loop_normal_disagg_decode(self):
    while True:
        recv_reqs = recv_requests()
        process_input_requests(recv_reqs)
        
        # Process decode queue (pre-alloc + transfers)
        process_decode_queue()
        
        # Get batch, merge transferred batches
        batch = get_next_disagg_decode_batch_to_run()
        if last_batch.forward_mode.is_prebuilt():
            running_batch.merge_batch(last_batch)
        
        # Create prebuilt batch from waiting queue
        new_batch = get_new_prebuilt_batch()
        
        if batch:
            result = run_batch(batch)
            process_batch_result(batch, result)
```

---

## Failure Handling

### Bootstrap Failures

```python
if poll == KVPoll.Failed:
    error_message = f"Prefill bootstrap failed for {req.rid}"
    sender.failure_exception()  # Get detailed error
    prepare_abort(req, error_message, HTTPStatus.INTERNAL_SERVER_ERROR)
```

### Transfer Failures

```python
if poll == KVPoll.Failed:
    error_message = f"Prefill transfer failed for {req.rid}"
    metrics_collector.increment_transfer_failed_reqs()
    # Abort request
```

### Heartbeat (Mooncake)

```python
class MooncakeKVManager:
    heartbeat_interval = 2.0s
    max_failures = 1
    # Track connection health, mark dead on failure
```

---

## KV Cache Offload (Decode Side)

**Source:** `python/sglang/srt/disaggregation/decode_kvcache_offload_manager.py`

Manages KV cache lifecycle for long sequences:

```python
def offload_kv_cache(self, req):
    # 1. Identify incremental part since prefill offload
    prefill_offloaded_len = len(origin_input_ids) // page_size * page_size
    incremental_len = len(all_tokens) - prefill_offloaded_len
    
    # 2. Early free prefill part from GPU
    free(token_indices[:prefill_offloaded_len])
    
    # 3. Async offload incremental to host (HiCache L2)
    host_indices = cache_controller.write(device_indices)
    
    # 4. Trigger async backup to storage (HiCache L3)
    cache_controller.write_storage(host_indices, tokens, hashes)
```

---

## Performance Optimizations

1. **Batch Transfer**: Groups contiguous page indices for bulk memory operations
2. **Thread Pooling**: Parallel layer transfers via ThreadPoolExecutor
3. **Lock-free Queues**: FastQueue for inter-thread communication
4. **Pre-allocation**: Decode allocates KV space before transfer starts
5. **Overlap Support**: Transfer can overlap with forward pass
6. **Symmetric Memory**: Efficient GPU-to-GPU buffer registration

---

## Directory Structure

```
python/sglang/srt/disaggregation/
├── base/
│   └── conn.py                    # BaseKVManager, BaseKVSender, BaseKVReceiver
├── common/
│   ├── conn.py                    # CommonKVManager, CommonKVBootstrapServer
│   └── utils.py                   # Helpers
├── prefill.py                     # PrefillBootstrapQueue, scheduler mixin
├── decode.py                      # DecodePreallocQueue, DecodeTransferQueue
├── encode_server.py               # Bootstrap server startup
├── kv_events.py                   # KV event publishing
├── utils.py                       # MetadataBuffers, TransferBackend registry
├── decode_schedule_batch_mixin.py # Decode batch preparation
├── decode_kvcache_offload_manager.py  # KV offloading
├── mooncake/conn.py               # Mooncake backend
├── mori/conn.py                   # Mori backend
├── nixl/conn.py                   # NIXL backend
├── ascend/conn.py                 # Ascend backend
└── fake/conn.py                   # No-op test backend
```
