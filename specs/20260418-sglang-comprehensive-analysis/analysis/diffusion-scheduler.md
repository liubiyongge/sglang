# Diffusion Runtime Scheduler & GPU Worker

## Overview

The diffusion runtime (`python/sglang/multimodal_gen/runtime/managers/`) provides a completely separate serving architecture from the LLM runtime. Adapted from [FastVideo](https://github.com/hao-ai-lab/FastVideo), it is designed for iterative denoising workloads (image/video generation) rather than autoregressive token generation.

**Key Difference from LLM Scheduler**: The diffusion scheduler processes one request at a time (FIFO, no continuous batching), since each diffusion request occupies the entire GPU memory during the multi-step denoising loop.

---

## Scheduler (`scheduler.py`)

### Architecture

```
HTTP Server → ZMQ (ROUTER) → Scheduler (Rank 0) → GPUWorker → Pipeline
                                    ↓ broadcast_pyobj
                              Scheduler (Rank N) → GPUWorker → Pipeline
```

### Class: `Scheduler`

**File**: `python/sglang/multimodal_gen/runtime/managers/scheduler.py`

The scheduler is the rank-0 event loop controller:

- **ZMQ Communication**: Uses `zmq.ROUTER` socket for multiplexed client connections with identity-based routing (lines 62-68)
- **FIFO Queue**: `waiting_queue: deque[tuple[bytes, Req]]` — simple deque, no priority scheduling (line 95)
- **Single-Request Processing**: `get_next_batch_to_run()` pops exactly one request at a time (lines 156-164)
- **Request Broadcasting**: For multi-GPU setups, received requests are broadcast via `broadcast_pyobj` to all ranks through SP, CFG, and TP groups (lines 264-288)

### Event Loop (`event_loop`)

```python
while self._running:
    # 1. Receive requests (ZMQ non-blocking poll)
    new_reqs = self.recv_reqs()
    new_reqs = self.process_received_reqs_with_req_based_warmup(new_reqs)
    self.waiting_queue.extend(new_reqs)
    
    # 2. Get next request (FIFO)
    items = self.get_next_batch_to_run()
    
    # 3. Dispatch to handler (type-based routing)
    handler = self.request_handlers.get(type(processed_req))
    output_batch = handler(reqs)
    
    # 4. Return result via ZMQ
    self.return_result(output_batch, identity, is_warmup)
```

### Request Handlers

| Request Type | Handler | Purpose |
|---|---|---|
| `Req` | `_handle_generation` | Execute diffusion pipeline forward pass |
| `SetLoraReq` | `_handle_set_lora` | Hot-swap LoRA adapters |
| `MergeLoraWeightsReq` | `_handle_merge_lora` | Merge LoRA into base weights |
| `UnmergeLoraWeightsReq` | `_handle_unmerge_lora` | Unmerge LoRA |
| `ListLorasReq` | `_handle_list_loras` | List loaded LoRA adapters |
| `ShutdownReq` | `_handle_shutdown` | Graceful shutdown |

### Warmup Strategy

Two warmup modes:

1. **Resolution-based warmup** (`warmup_resolutions`): Pre-generates dummy requests for each configured resolution at startup (lines 166-213)
2. **Request-based warmup**: Clones the first real request as a warmup run before processing it (lines 215-236)

### Error Handling

- **Consecutive error threshold**: After 3 consecutive ZMQ errors, the scheduler terminates (lines 106-107, 317-326)
- **Per-request error handling**: Errors during execution produce `OutputBatch(error=str(e))` without crashing the loop (lines 345-355)

---

## GPU Worker (`gpu_worker.py`)

### Class: `GPUWorker`

**File**: `python/sglang/multimodal_gen/runtime/managers/gpu_worker.py`

The worker is responsible for:
1. Initializing the distributed environment
2. Loading the diffusion pipeline
3. Executing forward passes

### Initialization Flow

```
GPUWorker.__init__()
├── init_device_and_model()
│   ├── torch.cuda.set_device(local_rank)
│   ├── Set MASTER_ADDR, MASTER_PORT, LOCAL_RANK, RANK, WORLD_SIZE
│   ├── maybe_init_distributed_environment_and_model_parallel(
│   │       tp_size, enable_cfg_parallel, ulysses_degree,
│   │       ring_degree, sp_size, dp_size)
│   ├── build_pipeline(server_args)  → ComposedPipelineBase
│   └── configure_layerwise_offload() [if enabled]
├── get_sp_group() → sp_group, sp_cpu_group
├── get_tp_group() → tp_group, tp_cpu_group
└── get_cfg_group() → cfg_group, cfg_cpu_group
```

### `execute_forward(batch: List[Req]) -> OutputBatch`

The core execution method:

1. **Reset memory stats** on rank 0
2. **Call pipeline forward**: `self.pipeline.forward(req, self.server_args)`
3. **Construct OutputBatch** with:
   - `output`: Generated frames/images
   - `audio`: Audio output (for audio models)
   - `timings`: Stage-level performance metrics
   - `trajectory_timesteps/latents/decoded`: Optional debug outputs
4. **Memory analysis** on rank 0: Reports peak GPU usage and suggests offload configurations
5. **Optional file save**: If `return_file_paths_only`, saves outputs to disk and returns paths

### Memory Analysis

The `do_mem_analysis()` method:
- Tracks `torch.cuda.max_memory_allocated()`
- Computes which components (VAE, text_encoder, etc.) could remain GPU-resident based on remaining memory
- Suggests which `--*-cpu-offload` flags could be disabled

### LoRA Management

The worker delegates to the underlying `LoRAPipeline`:
- `set_lora(nickname, path, target, strength)` — load adapters
- `merge_lora_weights(target, strength)` — bake into base weights
- `unmerge_lora_weights(target)` — restore base weights

---

## Process Model (`run_scheduler_process`)

```
run_scheduler_process()
├── configure_logger()
├── globally_suppress_loggers()
├── set_cuda_arch()
├── Scheduler(server_args, gpu_id=rank, port_args, pipes)
├── pipe_writer.send({"status": "ready"})  # signal parent
└── scheduler.event_loop()  # blocks until shutdown
```

**Multi-GPU Launch** (handled by `launch_server.py`):
- Rank 0: Acts as master, handles ZMQ requests
- Ranks > 0: Slave workers, receive tasks via `broadcast_pyobj`
- Each rank creates its own `Scheduler` + `GPUWorker` pair
- Communication: `task_pipes_to_slaves` / `result_pipes_from_slaves` (multiprocessing.Pipe)

---

## Comparison with LLM Scheduler

| Aspect | LLM Scheduler | Diffusion Scheduler |
|--------|--------------|-------------------|
| **Batching** | Continuous batching (dynamic) | Single-request FIFO |
| **Memory** | KV cache pool management | Full GPU per request |
| **Scheduling Policy** | LPM/DFS-Weight/FCFS/LOF | Simple FIFO deque |
| **Parallelism Coord** | TokenizerManager → Scheduler → Detokenizer | Scheduler → GPUWorker → Pipeline |
| **IPC** | Multi-process ZMQ topology | ZMQ + broadcast_pyobj |
| **Preemption** | Decode retraction, priority scheduling | None |
| **State** | Per-request KV state, radix cache | Stateless per-request |
| **Forward Mode** | Prefill/Extend/Decode | Single forward call per request |

---

## Source References

- Scheduler: `python/sglang/multimodal_gen/runtime/managers/scheduler.py`
- GPU Worker: `python/sglang/multimodal_gen/runtime/managers/gpu_worker.py`
- Process Entry: `gpu_worker.py:run_scheduler_process()` (line 358)
- Server Launch: `python/sglang/multimodal_gen/runtime/launch_server.py`
