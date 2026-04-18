# GroupCoordinator & Communication Backend Hierarchy

## Overview

SGLang's distributed communication is orchestrated through the `GroupCoordinator` class, which serves as a unified abstraction over multiple communication backends. It provides automatic backend selection based on tensor size, hardware capabilities, and execution mode (eager vs. CUDA graph capture).

**Source:** `python/sglang/srt/distributed/parallel_state.py` (lines 173-1322)

---

## GroupCoordinator Architecture

### Core Attributes

```
Rank Information:
  rank              - Global rank across all processes
  ranks             - List of global ranks in this group
  world_size        - Size of the group
  rank_in_group     - Rank within this specific group
  local_rank        - Rank on the local node (device assignment)

Process Groups:
  device_group      - ProcessGroup for GPU communication (NCCL/Gloo/Mooncake)
  cpu_group         - ProcessGroup for CPU communication (always Gloo)
  local_size        - Number of processes on local node

Backend Selection Flags:
  use_pynccl                        - PyNccl communicator enabled
  use_pymscclpp                     - PyMsccl++ enabled
  use_custom_allreduce              - Custom all-reduce enabled
  use_torch_symm_mem_all_reduce     - Torch symmetric memory enabled
  use_hpu_communicator              - Habana HPU communicator
  use_xpu_communicator              - Intel XPU communicator
  use_npu_communicator              - Ascend NPU communicator
  use_message_queue_broadcaster     - Shared memory broadcaster

Communicator Instances:
  pynccl_comm       - PyNcclCommunicator
  pymscclpp_comm    - PyMscclppCommunicator
  ca_comm           - Custom AllReduce communicator
  qr_comm           - Quick AllReduce (AMD MI300)
  torch_symm_mem_comm - Torch symmetric memory communicator
  hpu_communicator  - Habana accelerator communicator
  xpu_communicator  - Intel GPU communicator
  npu_communicator  - Ascend NPU communicator
  mq_broadcaster    - Shared memory message queue broadcaster
```

### Initialization Flow

1. **Group hierarchy setup**: Creates rank info and determines group membership
2. **PyTorch process groups**:
   - `device_group` with configured backend (NCCL, Gloo, Mooncake)
   - `cpu_group` with Gloo backend (always, for CPU coordination)
3. **Device assignment**: Based on `is_cuda_alike()`, `is_npu()`, `is_xpu()`, `is_musa()`
4. **Lazy communicator initialization**: Created on demand based on flags

### Backend Selection for Device Groups

```python
if "mooncake" in torch_distributed_backend:
    device_group = torch.distributed.new_group(
        ranks, backend="mooncake",
        pg_options=MooncakeBackendOptions(active_ranks))
else:
    device_group = torch.distributed.new_group(
        ranks, backend=torch_distributed_backend)  # NCCL, Gloo, etc.

# CPU group (always Gloo):
cpu_group = torch.distributed.new_group(ranks, backend="gloo", timeout=gloo_timeout)
```

---

## Communication Backends

### 1. PyNccl (Python NCCL Wrapper)

**Source:** `python/sglang/srt/distributed/device_communicators/pynccl.py`

Direct Python bindings to the NCCL library, providing control over stream management and CUDA graph capture.

**Supported Operations:**
- `all_reduce()` - Sum aggregation across all ranks
- `all_gather()` / `cp_all_gather_into_tensor()` - Gather data from all ranks
- `reduce_scatter()` / `reduce_scatterv()` - Scatter reduced data
- `broadcast()` - One-to-all broadcast
- `send()` / `recv()` - Point-to-point async operations
- `group_start()` / `group_end()` - Batch collective calls
- `register_comm_window_raw()` / `deregister_comm_window()` - IPC memory windows

**Key Features:**
- Custom CUDA stream or current stream selection
- Disabled by default, enabled during CUDA graph capture
- Device-bound communicator
- `_resolve_stream()` method for stream determination

**Initialization:**
```python
# Rank 0 generates unique ID, others receive via broadcast
if rank == 0:
    unique_id = nccl.ncclGetUniqueId()
# Each rank creates communicator
comm = nccl.ncclCommInitRank(world_size, unique_id, rank)
# Warmup
data = torch.zeros(1, device=device)
all_reduce(data)
```

### 2. Custom All-Reduce (CAR)

**Source:** `python/sglang/srt/distributed/device_communicators/custom_all_reduce.py`

Optimized GPU peer-to-peer all-reduce for small to medium tensors using GPU direct access.

**Constraints:**
- Supported world sizes: 2, 4, 6, 8
- Max sizes: CUDA 8192 KB, HIP (AMD) 16384 KB, MUSA 131072 KB
- Single node only (no multi-node)
- Requires full NVLink connectivity (or PCIe for world_size=2)
- Requires GPU P2P access

**Implementation:**
- IPC memory registration for cross-GPU access
- Dual buffer strategy: `meta_ptrs` (metadata + sync) and `buffer_ptrs` (IPC buffer)
- Two modes: Registered (graph mode, tensor already in IPC buffer) and Unregistered (eager, copy first)
- AMD deterministic mode: 1-stage kernel for fixed ordering

### 3. PyMsccl++ (Multi-GPU Collective Communication)

**Source:** `python/sglang/srt/distributed/device_communicators/pymscclpp.py`

Optimized collective communication for A100+ GPUs, faster than NCCL for small messages.

**Constraints:**
- Supported world sizes: 8, 16 (consecutive ranks)
- Max message size: 1 MB (configurable via `SGLANG_MSCCLPP_MAX_BYTES`)
- Graph-only mode (disabled in eager)

**Context Selection:**
- 8 GPUs: `MSCCL1SHOT1NODELL` (1-shot, single node)
- 16 GPUs: `MSCCL1SHOT2NODELL` (2-shot)

**Features:**
- Pre-tuning: Benchmarks thread/block configs at initialization
- Binary search for closest pre-tuned message size
- Scratch buffers for intermediate results

### 4. Torch Symmetric Memory All-Reduce

**Source:** `python/sglang/srt/distributed/device_communicators/torch_symm_mem.py`

PyTorch's built-in symmetric memory collectives for Hopper+ GPUs.

**Requirements:**
- Device capability >= 9 (Hopper)
- Multicast support enabled
- Supported world sizes: CC 9: [4, 6, 8], CC 10: [6, 8]
- bfloat16 only

**Kernel Selection:**
- `multimem_all_reduce_()` for supported topologies
- `two_shot_all_reduce_()` as fallback
- Copy-and-reduce pattern: stage input into buffer, reduce, copy to output

### 5. Quick All-Reduce (AMD MI300 Series)

**Source:** `python/sglang/srt/distributed/device_communicators/quick_all_reduce.py`

Quantized all-reduce for AMD MI300 with ultra-high bandwidth.

**Supported:**
- Architectures: gfx94, gfx95 (MI300 series)
- World sizes: 2, 4, 8
- Data types: float16, bfloat16

**Quantization Regimes:**
- FP: Full precision
- INT8: 8-bit quantization
- INT6: 6-bit quantization
- INT4: 4-bit quantization

**Configuration (env vars):**
- `ROCM_QUICK_REDUCE_QUANTIZATION`: Level (FP, INT8, INT6, INT4, NONE)
- `ROCM_QUICK_REDUCE_CAST_BF16_TO_FP16`: Convert bfloat16 to float16
- `ROCM_QUICK_REDUCE_MAX_SIZE_BYTES_MB`: Max size in MB

### 6. Hardware-Specific Communicators

| Communicator | Source | Hardware | Notes |
|---|---|---|---|
| `HpuCommunicator` | `device_communicators/hpu_communicator.py` | Habana Gaudi | Uses `htorch.core.mark_step()` workaround |
| `XpuCommunicator` | `device_communicators/xpu_communicator.py` | Intel GPU | Uses all_gather instead of gather for Ray |
| `NpuCommunicator` | `device_communicators/npu_communicator.py` | Ascend NPU | Standard dist operations |

---

## All-Reduce Dispatch Logic

The central dispatcher in `GroupCoordinator.all_reduce()` (lines 527-622) implements a priority-based backend selection:

```
Priority Order:
1. Deterministic (AMD) → Custom AR 1-stage kernel
2. CPU tensor → SHM all-reduce or torch.distributed
3. Hardware-specific → HPU/XPU/NPU communicator
4. PyNccl + Symmetric Memory → PyNccl (symm_mem enabled)
5. Custom AR eligible → CA all-reduce (outplace)
6. Quick AR eligible → QR all-reduce (outplace)
7. PyMsccl++ eligible → MSCCL++ all-reduce (outplace)
8. Torch SymmMem eligible → SymmMem all-reduce (outplace)
9. Fallback → PyNccl inplace or torch.distributed
```

### Decision Criteria

| Backend | Eager Mode | Graph Mode | Size Constraint | Conditions |
|---------|-----------|-----------|-----------------|------------|
| CA (Custom AR) | Enabled | Enabled | <= 8/16/131 MB | Single node, full NVLink/P2P |
| QR (Quick AR) | Enabled | Enabled | Min~2GB | AMD MI300+, quantization |
| PyMsccl++ | Disabled | Enabled | <= 1 MB | 8/16 GPUs consecutive |
| TorchSymmMem | Disabled | Enabled | < max | H100+, supported topology |
| PyNccl | Disabled | Enabled | Any | Graph capture enabled |
| torch.distributed | Enabled | Disabled | Any | Fallback |

---

## CUDA Graph Capture Integration

Context manager for graph capture (lines 462-525):

```python
@contextmanager
def graph_capture(self, graph_capture_context=None, stream=None):
    stream = stream or self.device_module.Stream()
    curr_stream = get_current_device_stream_fast()
    if curr_stream != stream:
        stream.wait_stream(curr_stream)
    
    with self.device_module.stream(stream):
        maybe_ca_context = ca_comm.capture() if ca_comm else nullcontext()
        maybe_pynccl_context = pynccl_comm.change_state(enable=True, stream=stream) ...
        maybe_pymscclpp_context = pymscclpp_comm.change_state(enable=True) ...
        
        with maybe_ca_context, maybe_pynccl_context, maybe_pymscclpp_context:
            yield graph_capture_context
```

Backends supported in graph mode: Quick AR, Custom AR, PyNccl, PyMsccl++, TorchSymmMem.
Not supported: torch.distributed (requires eager).

---

## Communication Primitives Summary

| Operation | PyNccl | CA | QR | PyMsccl++ | TorchSymmMem | Gloo |
|-----------|--------|----|----|-----------|--------------|------|
| all_reduce | Y | Y | Y | Y | Y | Y |
| all_gather | Y | - | - | - | - | Y |
| all_gatherv | Y | - | - | - | - | - |
| gather | Y | - | - | - | - | Y |
| broadcast | Y | - | - | - | - | Y |
| reduce_scatter | Y | - | - | - | - | Y |
| reduce_scatterv | Y | - | - | - | - | - |
| send/recv | Y | - | - | - | - | Y |
| barrier | - | - | - | - | - | Y |

---

## Model Parallel Group Initialization

**Function:** `initialize_model_parallel()` (lines 1557-1698)

```python
def initialize_model_parallel(
    tensor_model_parallel_size: int,
    expert_model_parallel_size: int,
    pipeline_model_parallel_size: int,
    backend: Optional[str] = None,
    duplicate_tp_group: bool = False,
) -> None:
```

**Group creation order:**

1. **Tensor Parallel (TP) groups**: Consecutive ranks split into groups of `tp_size`
   - Example: 8 GPUs, TP=2 -> groups [0,1], [2,3], [4,5], [6,7]

2. **MOE Expert Parallel (EP) groups**: If MOE enabled, subdivide TP groups
   - `moe_tp_size = tp_size // ep_size`

3. **Pipeline Parallel (PP) groups**: Strided ranks
   - Example: 8 GPUs, PP=4 -> groups [0,2,4,6], [1,3,5,7]

4. **Optional duplicate TP group**: For PD-Multiplexing prefill

**Global group accessors:**
- `get_world_group()` - All processes
- `get_tp_group()` - Tensor parallel group
- `get_pp_group()` - Pipeline parallel group
- `get_moe_ep_group()` - MOE expert parallel group
- `get_moe_tp_group()` - MOE tensor parallel group

---

## Shared Memory Broadcast (MessageQueue)

**Source:** `python/sglang/srt/distributed/device_communicators/shm_broadcast.py`

Low-latency broadcast for TP groups using shared memory + ZMQ.

**Buffer Architecture (ShmRingBuffer):**
```
Memory Layout:
+-------------------------------------------------+
| Data Section        | Metadata Section          |
+-------------------------------------------------+
| chunk0 | ... | chunkN | meta0 | ... | metaN    |
+-------------------------------------------------+

Metadata per chunk:
+-----------+---------+---------+-----+---------+
| written_f | reader0 | reader1 | ... | readerN |
+-----------+---------+---------+-----+---------+
```

**State Transitions:**
- `0???...???` - Not written, writer can write
- `1000...000` - Just written, readers can read
- `1???...???` - Written, some readers have read
- `1111...111` - All readers done, writer can overwrite

**Communication model:** Single writer (rank 0), multiple readers (local ranks). Local via shared memory ring buffer, remote via ZMQ pub/sub.

---

## Symmetric Memory Allocator

**Source:** `python/sglang/srt/distributed/device_communicators/pynccl_allocator.py`

Allocates GPU memory via NCCL's symmetric memory allocator for IPC window operations.

**SymmetricMemoryContext:**
```python
class SymmetricMemoryContext:
    def __enter__(self):
        self._mem_pool_ctx = torch.cuda.use_mem_pool(get_nccl_mem_pool())
        if is_graph_capture:
            torch._C._cuda_endAllocateToPool(device, graph_pool_id)
        os.environ["SGLANG_TMP_NCCL_COMM_VALUE"] = str(comm.value)
    
    def __exit__(self, ...):
        if is_graph_capture:
            torch._C._cuda_beginAllocateCurrentThreadToPool(...)
```

---

## Custom Parallel Groups

**Function:** `create_custom_parallel_group()` (lines 1701-1747)

Creates ad-hoc process groups outside the standard TP/PP hierarchy:
1. Each rank passes the list of ranks it wants to join
2. All-gather configurations from all ranks
3. Identify unique group configurations
4. Create groups and return membership

---

## Cleanup and Lifecycle

```python
def cleanup_dist_env_and_memory(shutdown_ray: bool = False):
    destroy_model_parallel()              # TP, PP, MOE groups
    destroy_distributed_environment()     # World group
    # Clear CUDA cache across all device types
```

---

## Architecture Diagram

```
+------------------------------------------------------------------+
|                    GroupCoordinator                                |
|  (Ranks, Groups, Device Assignment, Communicator Management)     |
+------------------------------+-----------------------------------+
                               |
            +------------------+------------------+
            |                  |                  |
            v                  v                  v
       Device Group       CPU Group         Communicators
       (NCCL/Gloo)        (Gloo)            (Pluggable)
                                                 |
                +--------------------------------+-------------------+
                v                v                v                  v
           PyNccl        Custom/Quick AR    Hardware-Specific   Shared Memory
       all_reduce         CA (Small)         HPU/XPU/NPU        MessageQueue
       all_gather         QR (AMD MI300)                        (Broadcast)
       reduce_scatter     PyMsccl++ (Graph)
       P2P send/recv      Torch SymmMem
```
