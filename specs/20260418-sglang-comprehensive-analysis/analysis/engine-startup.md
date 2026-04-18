# SGLang Engine Startup and Process Spawning: Comprehensive Analysis

## Executive Summary

The SGLang Engine is a sophisticated multi-process inference system designed for distributed, high-performance language model serving. The architecture separates concerns into three main components: **TokenizerManager** (main process), **Scheduler** (subprocess), and **DetokenizerManager** (subprocess), communicating via ZMQ sockets over IPC or TCP connections. This document provides a detailed technical analysis of the engine startup sequence, process spawning mechanisms, inter-process communication setup, and the multi-node/multi-GPU coordination infrastructure.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Engine Class Initialization Sequence](#engine-class-initialization-sequence)
3. [Multi-Process Architecture](#multi-process-architecture)
4. [Process Spawning Mechanism](#process-spawning-mechanism)
5. [ZMQ Socket Setup and Port Allocation](#zmq-socket-setup-and-port-allocation)
6. [Process Synchronization via Pipes](#process-synchronization-via-pipes)
7. [Environment Configuration](#environment-configuration)
8. [Signal Handling and Error Cascading](#signal-handling-and-error-cascading)
9. [Shutdown and Cleanup](#shutdown-and-cleanup)
10. [Multi-Node Setup](#multi-node-setup)
11. [Data Parallel Architecture](#data-parallel-architecture)
12. [Configuration Through ServerArgs](#configuration-through-serverargs)

---

## Architecture Overview

### High-Level System Design

SGLang's inference engine follows a producer-consumer pipeline architecture optimized for throughput and latency:

```
+---------------------------------------------------------------------+
|                         Main Process (PID 1000)                      |
|  +----------------------------------------------------------------+ |
|  |  HTTP Server (Uvicorn/FastAPI)                                 | |
|  |  - Receives client requests                                    | |
|  |  - Routes to TokenizerManager                                  | |
|  +----------------------------------------------------------------+ |
|  +----------------------------------------------------------------+ |
|  |  Engine (Python API)                                           | |
|  |  - Manages lifecycle                                           | |
|  |  - Spawns subprocesses                                         | |
|  |  - Coordinates communication                                   | |
|  +----------------------------------------------------------------+ |
|  +----------------------------------------------------------------+ |
|  |  TokenizerManager                                              | |
|  |  - Text -> Token IDs (tokenization)                            | |
|  |  - Request queuing and batching                                | |
|  |  - Token IDs -> Text (detokenization output)                   | |
|  |  - Async event loop (asyncio)                                  | |
|  +----------------------------------------------------------------+ |
+---------------------------------------------------------------------+
                                  | (ZMQ PUSH)
                    scheduler_input_ipc_name
                                  v
         +----------------------------------------+
         |  Scheduler Process (PID 1001-100N)     |  For each GPU
         |  - Model inference execution           |
         |  - Request scheduling and batching     |
         |  - KV cache management                 |
         |  - Tensor Parallel (TP) coordination   |
         |  - Pipeline Parallel (PP) handling     |
         |  - Token generation                    |
         +----------------------------------------+
                                  | (ZMQ PUSH)
                      detokenizer_ipc_name
                                  v
+---------------------------------------------------------------------+
|              DetokenizerManager Process (PID 1002)                   |
|  - Token IDs -> Text strings (detokenization)                       |
|  - Incremental streaming support                                    |
|  - Return results to TokenizerManager                               |
+---------------------------------------------------------------------+
```

### Process Components

| Component | Process Type | Responsibilities | Communication |
|-----------|--------------|------------------|-----------------|
| **TokenizerManager** | Main | Request tokenization, batching, response streaming | ZMQ PULL from detokenizer, PUSH to scheduler |
| **Scheduler** | Subprocess | Model inference, scheduling, token generation | ZMQ PULL from tokenizer, PUSH to detokenizer |
| **DetokenizerManager** | Subprocess | Token decoding, streaming output | ZMQ PULL from scheduler, PUSH to tokenizer |

---

## Engine Class Initialization Sequence

### Phase 1: Constructor Initialization

The `Engine` class constructor (`engine.py`, lines 139-204) initiates the complete startup process:

```python
def __init__(self, **kwargs):
    """
    The arguments of this function is the same as `sglang/srt/server_args.py::ServerArgs`.
    """
    # Phase 1: Parse and validate ServerArgs
    if "server_args" in kwargs:
        server_args = kwargs["server_args"]
    else:
        if "log_level" not in kwargs:
            kwargs["log_level"] = "error"  # Default: suppress logs
        server_args = self.server_args_class(**kwargs)
    self.server_args = server_args
    logger.info(f"{server_args=}")

    # Phase 2: Register automatic shutdown
    atexit.register(self.shutdown)

    # Phase 3: Launch subprocesses
    tokenizer_manager, template_manager, scheduler_infos, port_args = (
        _launch_subprocesses(
            server_args=server_args,
            init_tokenizer_manager_func=self.init_tokenizer_manager_func,
            run_scheduler_process_func=self.run_scheduler_process_func,
            run_detokenizer_process_func=self.run_detokenizer_process_func,
        )
    )
    self.tokenizer_manager = tokenizer_manager
    self.template_manager = template_manager
    self.scheduler_info = scheduler_infos[0]
    self.port_args = port_args

    # Phase 4: Initialize ZMQ sockets for RPC (on primary node only)
    context = zmq.Context(2)
    if self.server_args.node_rank == 0:
        self.send_to_rpc = get_zmq_socket(
            context, zmq.DEALER, self.port_args.rpc_ipc_name, True
        )
    else:
        self.send_to_rpc = None

    # Phase 5: Setup tracing (optional)
    if server_args.enable_trace:
        process_tracing_init(server_args.otlp_traces_endpoint, "sglang")
        trace_set_thread_info("Tokenizer")

    # Phase 6: Create or get event loop
    try:
        self.loop = asyncio.get_running_loop()
    except RuntimeError:
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
```

---

## Multi-Process Architecture

### Three-Component Pipeline

The SGLang runtime uses a three-stage pipeline to maximize throughput:

#### 1. TokenizerManager (Main Process)

**Location**: `tokenizer_manager.py` (102KB file)

**Responsibilities**:
- Receives inference requests from HTTP/gRPC clients
- Tokenizes input text using HuggingFace transformers
- Batches requests for efficient GPU processing
- Manages request state and lifecycle tracking
- Handles multimodal input processing (images, audio, video)
- Streams output back to clients
- Manages LoRA adapter switching
- Implements graceful shutdown

**Key Data Structures**:
```python
@dataclasses.dataclass
class ReqState:
    """Store the state a request."""
    out_list: List[Dict[Any, Any]]          # Output accumulation
    finished: bool                           # Completion flag
    event: asyncio.Event                    # Synchronization primitive
    obj: Union[GenerateReqInput, EmbeddingReqInput]  # Request object

    # Metrics
    created_time: float
    finished_time: float
    first_token_time: float
    request_sent_to_scheduler_ts: float
    response_sent_to_client_ts: float
```

**Initialization Sequence** (lines 191-235):
```python
def __init__(self, server_args: ServerArgs, port_args: PortArgs):
    self.server_args = server_args
    self.init_model_config()
    self.init_tokenizer_and_processor()
    self.init_ipc_channels(port_args)       # Setup ZMQ sockets
    self.init_running_status()
    self.init_request_logging_and_dumping()
    self.init_weight_update()
    self.init_lora()
    self.init_disaggregation()
    self.init_metric_collector_watchdog()
    if self.enable_metrics:
        start_cpu_monitor_thread("tokenizer")
    self.init_request_dispatcher()          # Setup type-based request routing
```

#### 2. Scheduler (Subprocess per GPU/Device)

**Location**: `scheduler.py` (125.6KB file)

**Process Entry Point**: `run_scheduler_process()` (lines 2984-3100)

**Responsibilities**:
- Receives tokenized requests from TokenizerManager
- Manages KV cache and memory allocation
- Schedules requests into batches
- Executes forward passes on GPU/accelerator
- Generates output tokens
- Coordinates with other schedulers via torch.distributed (for TP/PP)
- Sends output tokens to DetokenizerManager

**Initialization** (lines 3008-3048):
```python
def run_scheduler_process(
    server_args: ServerArgs,
    port_args: PortArgs,
    gpu_id: int,
    tp_rank: int,
    moe_ep_rank: int,
    pp_rank: int,
    dp_rank: Optional[int],
    pipe_writer,
):
    # Configure process identity
    setproctitle.setproctitle(f"sglang::scheduler{prefix}")
    faulthandler.enable()
    kill_itself_when_parent_died()
    parent_process = psutil.Process().parent()

    # Configure logging
    configure_logger(server_args, prefix=prefix)

    # Create scheduler instance
    scheduler = Scheduler(
        server_args, port_args, gpu_id, tp_rank, moe_ep_rank, pp_rank, dp_rank,
    )

    # Signal readiness to parent
    result_dict = {
        "status": "ready",
        "max_total_num_tokens": scheduler.max_total_num_tokens,
        "max_req_input_len": scheduler.max_req_input_len,
    }
    pipe_writer.send(result_dict)

    # Run appropriate event loop based on disaggregation mode
    if disaggregation_mode == DisaggregationMode.NULL:
        if scheduler.enable_pdmux:
            scheduler.event_loop_pdmux()
        elif server_args.pp_size > 1:
            scheduler.event_loop_pp()
        elif scheduler.enable_overlap:
            scheduler.event_loop_overlap()
        else:
            scheduler.event_loop_normal()
```

#### 3. DetokenizerManager (Subprocess)

**Location**: `detokenizer_manager.py` (17.3KB file)

**Process Entry Point**: `run_detokenizer_process()` (lines 431-452)

```python
def run_detokenizer_process(
    server_args: ServerArgs,
    port_args: PortArgs,
    detokenizer_manager_class=DetokenizerManager,
):
    kill_itself_when_parent_died()
    setproctitle.setproctitle("sglang::detokenizer")
    configure_logger(server_args)
    parent_process = psutil.Process().parent()

    try:
        manager = detokenizer_manager_class(server_args, port_args)
        if server_args.tokenizer_worker_num == 1:
            manager.event_loop()
        else:
            manager.multi_http_worker_event_loop()
    except Exception:
        traceback = get_exception_traceback()
        logger.error(f"DetokenizerManager hit an exception: {traceback}")
        manager.maybe_clear_socket_mapping()
        parent_process.send_signal(signal.SIGQUIT)
```

---

## Process Spawning Mechanism

### Master Spawning Function: `_launch_subprocesses()`

**Location**: `engine.py`, lines 983-1060

```python
def _launch_subprocesses(
    server_args: ServerArgs,
    init_tokenizer_manager_func: Callable,
    run_scheduler_process_func: Callable,
    run_detokenizer_process_func: Callable,
    port_args: Optional[PortArgs] = None,
) -> Tuple[TokenizerManager, TemplateManager, Tuple[Dict], PortArgs]:
    # Step 1: Configure global environment
    configure_logger(server_args)
    _set_envs_and_config(server_args)
    server_args.check_server_args()

    # Step 2: Allocate ports for inter-process communication
    if port_args is None:
        port_args = PortArgs.init_new(server_args)

    # Step 3: Launch scheduler processes
    scheduler_procs, scheduler_pipe_readers = _launch_scheduler_processes(
        server_args=server_args, port_args=port_args,
        run_scheduler_process_func=run_scheduler_process_func,
    )

    # Step 4: Handle non-zero rank nodes in multi-node setup
    if server_args.node_rank >= 1:
        scheduler_infos = _wait_for_scheduler_ready(scheduler_pipe_readers, scheduler_procs)
        launch_dummy_health_check_server(...)
        for proc in scheduler_procs:
            proc.join()
        return None, None, scheduler_infos, port_args

    # Step 5: Launch detokenizer process
    detoken_proc = mp.Process(target=run_detokenizer_process_func, args=(server_args, port_args))
    detoken_proc.start()

    # Step 6: Initialize TokenizerManager in main process
    if server_args.tokenizer_worker_num == 1:
        tokenizer_manager, template_manager = init_tokenizer_manager_func(server_args, port_args)
    else:
        tokenizer_manager = MultiTokenizerRouter(server_args, port_args)
        template_manager = None

    # Step 7: Wait for model loading to complete
    scheduler_infos = _wait_for_scheduler_ready(scheduler_pipe_readers, scheduler_procs)

    # Step 8: Share scheduler info with tokenizer
    tokenizer_manager.max_req_input_len = scheduler_infos[0]["max_req_input_len"]

    return tokenizer_manager, template_manager, scheduler_infos, port_args
```

### Scheduler Process Spawning: `_launch_scheduler_processes()`

**Location**: `engine.py`, lines 905-981

Handles both single and multi-node scheduler spawning:

```python
def _launch_scheduler_processes(server_args, port_args, run_scheduler_process_func):
    scheduler_procs = []

    if server_args.dp_size == 1:
        # Tensor Parallel (TP) only - Direct process spawning
        pp_size_per_node = max(server_args.pp_size // server_args.nnodes, 1)
        tp_size_per_node = server_args.tp_size // nnodes_per_tp_group

        # Create one process per (PP rank, TP rank) combination
        for pp_rank in pp_rank_range:
            for tp_rank in tp_rank_range:
                reader, writer = mp.Pipe(duplex=False)
                gpu_id = server_args.base_gpu_id + (pp_rank % pp_size_per_node) * tp_size_per_node + (tp_rank % tp_size_per_node) * server_args.gpu_id_step
                moe_ep_rank = tp_rank // (server_args.tp_size // server_args.ep_size)

                proc = mp.Process(
                    target=run_scheduler_process_func,
                    args=(server_args, port_args, gpu_id, tp_rank, moe_ep_rank, pp_rank, None, writer),
                )
                proc.start()
                scheduler_procs.append(proc)
                scheduler_pipe_readers.append(reader)
    else:
        # Data Parallel (DP) - Launch DataParallelController
        reader, writer = mp.Pipe(duplex=False)
        proc = mp.Process(
            target=run_data_parallel_controller_process,
            kwargs=dict(server_args=server_args, port_args=port_args, pipe_writer=writer, run_scheduler_process_func=run_scheduler_process_func),
        )
        proc.start()
        scheduler_procs.append(proc)

    return scheduler_procs, scheduler_pipe_readers
```

### Process Creation Flow Diagram

```
Engine.__init__()
    |
_launch_subprocesses()
    |-- configure_logger()
    |-- _set_envs_and_config()
    |   +-- signal.signal(signal.SIGQUIT, handler)
    |
    |-- PortArgs.init_new()
    |   +-- Allocate ZMQ socket addresses/ports
    |
    |-- _launch_scheduler_processes()
    |   |-- If dp_size == 1 (TP only):
    |   |   |-- For each (pp_rank, tp_rank):
    |   |   |   |-- Create Pipe (duplex=False)
    |   |   |   |-- mp.Process() spawned
    |   |   |   |   +-- run_scheduler_process()
    |   |   |   +-- proc.start()
    |   |   +-- Return [procs], [readers]
    |   |
    |   +-- If dp_size > 1 (DP enabled):
    |       |-- Create Pipe for controller
    |       |-- mp.Process() spawned
    |       |   +-- run_data_parallel_controller_process()
    |       +-- Return [controller_proc], [reader]
    |
    |-- _wait_for_scheduler_ready()
    |   +-- reader.recv() x N (blocking on pipe)
    |
    |-- mp.Process(target=run_detokenizer_process_func)
    |   +-- detoken_proc.start()
    |
    |-- init_tokenizer_manager() [in main process]
    |   +-- TokenizerManager()
    |
    +-- Return tokenizer_manager, template_manager, scheduler_infos, port_args
```

---

## ZMQ Socket Setup and Port Allocation

### PortArgs: Port Configuration Management

**Location**: `server_args.py`, lines 5517-5627

```python
@dataclasses.dataclass
class PortArgs:
    tokenizer_ipc_name: str          # DetokenizerManager -> TokenizerManager
    scheduler_input_ipc_name: str    # TokenizerManager -> Scheduler (rank 0)
    detokenizer_ipc_name: str        # Scheduler -> DetokenizerManager
    nccl_port: int                   # torch.distributed initialization
    rpc_ipc_name: str                # Engine <-> Scheduler RPC
    metrics_ipc_name: str            # Scheduler -> Metrics
    tokenizer_worker_ipc_name: Optional[str]  # Multi-tokenizer routing
```

### Port Allocation Strategy

#### Single-Node Mode (IPC-based)

When running on a single node without DP attention, the system uses Unix domain sockets (IPC):

```python
return PortArgs(
    tokenizer_ipc_name=f"ipc://{tempfile.NamedTemporaryFile(delete=False).name}",
    scheduler_input_ipc_name=f"ipc://{tempfile.NamedTemporaryFile(delete=False).name}",
    detokenizer_ipc_name=f"ipc://{tempfile.NamedTemporaryFile(delete=False).name}",
    nccl_port=nccl_port,
    rpc_ipc_name=f"ipc://{tempfile.NamedTemporaryFile(delete=False).name}",
    metrics_ipc_name=f"ipc://{tempfile.NamedTemporaryFile(delete=False).name}",
    tokenizer_worker_ipc_name=tokenizer_worker_ipc_name,
)
```

#### Multi-Node Mode (TCP-based)

When DP attention is enabled across multiple nodes:

```python
port_base = dist_init_port + 1
detokenizer_port = port_base + 1
rpc_port = port_base + 2
metrics_port = port_base + 3
scheduler_input_port = port_base + 4

return PortArgs(
    tokenizer_ipc_name=f"tcp://{dist_init_host}:{port_base}",
    scheduler_input_ipc_name=f"tcp://{dist_init_host}:{scheduler_input_port}",
    detokenizer_ipc_name=f"tcp://{dist_init_host}:{detokenizer_port}",
    nccl_port=nccl_port,
    rpc_ipc_name=f"tcp://{dist_init_host}:{rpc_port}",
    metrics_ipc_name=f"tcp://{dist_init_host}:{metrics_port}",
    tokenizer_worker_ipc_name=tokenizer_worker_ipc_name,
)
```

### ZMQ Communication Patterns

| Pattern | Direction | Socket Type | Use Case |
|---------|-----------|-------------|----------|
| **PUSH-PULL** | Tokenizer -> Scheduler | PUSH | Async task queue |
| **PUSH-PULL** | Scheduler -> Detokenizer | PUSH | Load balancing |
| **PUSH-PULL** | Detokenizer -> Tokenizer | PUSH | Response streaming |
| **DEALER-REP** | Engine -> Scheduler (RPC) | DEALER/REP | Remote calls |

---

## Process Synchronization via Pipes

### Synchronization Mechanism

SGLang uses **unidirectional pipes** (`mp.Pipe(duplex=False)`) for synchronization between parent and child processes:

```python
# In parent process
reader, writer = mp.Pipe(duplex=False)

# Pass writer to child process
proc = mp.Process(target=run_scheduler_process, args=(..., writer))
proc.start()

# Parent waits for child readiness
data = reader.recv()  # Blocking until child signals ready
```

### Readiness Signal Protocol

**Function**: `_wait_for_scheduler_ready()` (lines 880-903)

```python
def _wait_for_scheduler_ready(scheduler_pipe_readers, scheduler_procs):
    scheduler_infos = []
    for i in range(len(scheduler_pipe_readers)):
        try:
            data = scheduler_pipe_readers[i].recv()
        except EOFError:
            logger.error(f"Rank {i} scheduler is dead.")
            scheduler_procs[i].join()
            logger.error(f"Exit code: {scheduler_procs[i].exitcode}")
            raise
        if data["status"] != "ready":
            raise RuntimeError("Initialization failed.")
        scheduler_infos.append(data)
    return scheduler_infos
```

### Readiness Payload Structure

```python
result_dict = {
    "status": "ready",
    "max_total_num_tokens": scheduler.max_total_num_tokens,
    "max_req_input_len": scheduler.max_req_input_len,
}
pipe_writer.send(result_dict)
```

### Synchronization Timeline

```
Time 0:     Parent spawns N processes
Time 1-N:   Children initialize, load model
Time N:     Each child sends readiness via pipe
Time N+1:   Parent unblocks on reader.recv()
Time N+2:   All children ready, system operational
```

---

## Environment Configuration

### Function: `_set_envs_and_config()` (lines 800-878)

```python
def _set_envs_and_config(server_args: ServerArgs):
    # NCCL Configuration
    os.environ["NCCL_CUMEM_ENABLE"] = str(int(server_args.enable_symm_mem))
    os.environ["NCCL_NVLS_ENABLE"] = str(int(server_args.enable_nccl_nvls or server_args.enable_symm_mem))

    # CUDA Configuration
    os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "8"
    os.environ["CUDA_MODULE_LOADING"] = "AUTO"

    # SGLang Run Identification
    os.environ["SGLANG_RUN_ID"] = f"sglang-run-{time.time()}-{random.randint(0, 100000000)}"

    # Prometheus Metrics
    if server_args.enable_metrics:
        set_prometheus_multiproc_dir()

    # Resource Limits
    set_ulimit()

    # Version Verification
    if not get_bool_env_var("SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK"):
        assert_pkg_version("flashinfer_python", "0.6.2", ...)
        assert_pkg_version("sgl-kernel", "0.3.21", ...)

    # Signal Handling
    signal.signal(signal.SIGQUIT, launch_phase_sigquit_handler)

    # Multiprocessing Start Method
    mp.set_start_method("spawn", force=True)
```

### Environment Variables Summary

| Variable | Purpose | Typical Value |
|----------|---------|---------------|
| `NCCL_CUMEM_ENABLE` | NCCL unified memory | 0 or 1 |
| `NCCL_NVLS_ENABLE` | NVLink collective ops | 0 or 1 |
| `CUDA_DEVICE_MAX_CONNECTIONS` | GPU stream limit | 8 |
| `CUDA_MODULE_LOADING` | Module loading mode | AUTO |
| `SGLANG_RUN_ID` | Unique run identifier | sglang-run-... |

---

## Signal Handling and Error Cascading

### SIGQUIT Error Propagation

The system uses SIGQUIT signals for error propagation from child processes to parent:

```python
def launch_phase_sigquit_handler(signum, frame):
    logger.error("Received sigquit from a child process. It usually means the child failed.")
    kill_process_tree(os.getpid())

signal.signal(signal.SIGQUIT, launch_phase_sigquit_handler)
```

### Error Cascade Diagram

```
Scheduler Exception
    |
send_signal(SIGQUIT)
    |
Parent Process
    |
launch_phase_sigquit_handler()
    |
kill_process_tree(parent_pid, include_parent=False)
    |-- Kill all scheduler processes
    |-- Kill detokenizer process
    |-- Kill tokenizer workers (if any)
    +-- Clean up ZMQ sockets
```

### Process Monitoring

Each subprocess registers itself to die if parent dies:

```python
# In scheduler.py
kill_itself_when_parent_died()

# In detokenizer_manager.py
kill_itself_when_parent_died()
```

This prevents orphaned processes from consuming GPU memory.

---

## Shutdown and Cleanup

### Automatic Shutdown Registration

```python
atexit.register(self.shutdown)
```

### Shutdown Implementation

```python
def shutdown(self):
    """Shutdown the engine and all subprocesses."""
    kill_process_tree(os.getpid(), include_parent=False)

def __enter__(self):
    return self

def __exit__(self, exc_type, exc_value, traceback):
    self.shutdown()
    return False
```

### Context Manager Pattern

```python
with Engine(model_path="meta-llama/Llama-2-7b-hf") as engine:
    output = engine.generate("Hello world")
# shutdown() called automatically
```

---

## Multi-Node Setup

### Node Configuration

- `nnodes: int = 1` - Total number of nodes
- `node_rank: int = 0` - This node's rank (0 = primary)
- `dist_init_addr: Optional[str] = None` - Head node address (host:port)

### Multi-Node Initialization Flow

Non-primary nodes (node_rank >= 1) only run scheduler processes:

```python
if server_args.node_rank >= 1:
    scheduler_infos = _wait_for_scheduler_ready(scheduler_pipe_readers, scheduler_procs)
    launch_dummy_health_check_server(server_args.host, server_args.port, ...)
    for proc in scheduler_procs:
        proc.join()
    return None, None, scheduler_infos, port_args
```

### Multi-Node Network Diagram

```
Node 0 (rank=0, master)              Node 1 (rank=1)
+---------------------------+         +---------------------------+
| Engine                    |         |                           |
|  +- TokenizerManager     |         |   (No TokenizerManager)   |
|  +- Scheduler TP0 PP0    |<------->| Scheduler TP0 PP0         |
|  +- Scheduler TP1 PP0    |<------->| Scheduler TP1 PP0         |
|  +- DetokenizerManager   |         |                           |
|  +- HTTP Server           |         | (Dummy Health Check)      |
+---------------------------+         +---------------------------+
       |                                     |
       +------- NCCL (torch.distributed) ----+
       |      (TP/PP communication)          |
       +---- TCP/IPC (ZMQ sockets) ----------+
            (Inference requests)
```

---

## Data Parallel Architecture

### Data Parallel Controller

When `dp_size > 1`, the system launches a `DataParallelController` instead of direct scheduler processes:

```python
class DataParallelController:
    """A controller that dispatches requests to multiple data parallel workers."""

    def __init__(self, server_args, port_args, run_scheduler_process_func):
        self.load_balance_method = LoadBalanceMethod.from_str(server_args.load_balance_method)
        self.workers: List[zmq.Socket] = [None] * server_args.dp_size
        self.status: List[bool] = [True] * server_args.dp_size
```

### Load Balancing Methods

```python
class LoadBalanceMethod(Enum):
    ROUND_ROBIN = auto()           # Rotate through workers
    FOLLOW_BOOTSTRAP_ROOM = auto() # Use bootstrap assignment
    TOTAL_REQUESTS = auto()        # Route to worker with fewest requests
    TOTAL_TOKENS = auto()          # Route to worker with fewest tokens
```

---

## Configuration Through ServerArgs

### Key Configuration Groups

#### Parallelism Configuration

```python
tp_size: int = 1      # Tensor Parallel (model sharding across GPUs)
pp_size: int = 1      # Pipeline Parallel (model stages on different GPUs)
dp_size: int = 1      # Data Parallel (different batches on different nodes)
ep_size: int = 1      # Expert Parallel (for MoE models)
```

#### Memory and Scheduling

```python
mem_fraction_static: Optional[float] = None
max_running_requests: Optional[int] = None
max_total_tokens: Optional[int] = None
schedule_policy: str = "fcfs"
max_prefill_tokens: int = 16384
chunked_prefill_size: Optional[int] = None
```

#### Multi-Node Configuration

```python
dist_init_addr: Optional[str] = None  # "head_ip:port"
nnodes: int = 1
node_rank: int = 0
```

---

## Summary

### Key Architectural Principles

1. **Process Separation**: Tokenization (main), Scheduling (GPU process), Detokenization (separate process)
2. **Asynchronous Communication**: ZMQ sockets for inter-process message passing
3. **Hierarchical Synchronization**: Pipes for initialization, ZMQ for runtime communication
4. **Distributed Coordination**: torch.distributed for GPU-to-GPU synchronization
5. **Failure Resilience**: SIGQUIT cascading and automatic cleanup

### Performance Optimizations

- **IPC instead of TCP** on single node (zero-copy)
- **Pipe-based sync** for initialization (efficient blocking)
- **Load balancing** across DP workers
- **Disaggregation support** for prefill/decode separation
- **Memory-aware scheduling** with configurable limits

### Extensibility Points

Users and developers can customize:
- `ServerArgs` subclasses for additional configuration
- Process entry functions (`run_scheduler_process_func`, etc.)
- Signal handlers for crash dumps
- Load balancing methods
- Disaggregation modes
