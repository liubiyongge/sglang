# Custom All-Reduce Kernel

## Overview

SGLang implements a custom all-reduce CUDA kernel optimized for NVLink-equipped multi-GPU systems. The kernel provides lower latency than NCCL for small to medium tensors (up to 8-16 MB) by using direct GPU peer-to-peer memory access with minimal synchronization overhead.

**Key Source Files:**
- `sgl-kernel/csrc/allreduce/custom_all_reduce.cuh` - Main CUDA implementation
- `sgl-kernel/csrc/allreduce/custom_all_reduce.cu` - C++ bindings/Python interface
- `sgl-kernel/csrc/allreduce/custom_all_reduce_hip.cuh` - ROCm/HIP variant
- `sgl-kernel/csrc/allreduce/quick_all_reduce.cuh` - HIP 2-shot with quantization
- `sgl-kernel/csrc/allreduce/mscclpp_allreduce.cuh` - MSCCLPP kernels

---

## Algorithm Variants

### 1-Stage (cross_device_reduce_1stage)

Single synchronization pass where all ranks simultaneously reduce.

```
Step 1: Start barrier (all GPUs arrive)
Step 2: Each thread reads from ALL ranks and reduces locally
Step 3: End barrier (all GPUs complete)
```

**When Used:**
- world_size == 2 (always)
- world_size <= 4 AND bytes < 512 KB
- world_size <= 8 AND bytes < 256 KB

**Advantages:** Lowest latency, simplest synchronization.

### 2-Stage (cross_device_reduce_2stage)

Reduce-scatter followed by all-gather:

```
Stage 1: Reduce-Scatter
  - Partition data: part = size / ngpus
  - Each rank reduces its partition (reads from all, writes to temp)
  - Barrier with memory fence

Stage 2: All-Gather
  - Each rank reads reduced partition from all peers
  - Thread i gathers from rank i (cross-device visibility guarantee)
  - End barrier
```

**When Used:** Larger messages where bandwidth utilization matters more than latency.

**Visibility Guarantee:** GPU cross-device memory visibility is only guaranteed between threads with matching `threadIdx`. The all-gather stage exploits this by having thread `i` read from rank `i`.

---

## Grid/Block Configuration

| Platform | Blocks | Threads/Block | Occupancy | Notes |
|----------|--------|---------------|-----------|-------|
| CUDA | 36 | 512 | 1 block/SM | Careful grid search result |
| HIP/ROCm | 64 | 256 | - | 4 wavefronts |

The 36-block CUDA configuration balances SM utilization vs NVLink bus contention.

---

## Memory Layout

### Signal Structure (Synchronization Metadata)

```cpp
struct Signal {
    alignas(128) FlagType self_counter[kMaxBlocks][8];     // Per-block counters
    alignas(128) FlagType peer_counter[2][kMaxBlocks][8];  // Alternating peer counters
};
```

- `kMaxBlocks`: 36 (CUDA) / 64 (HIP)
- 8 entries per block (one per GPU, max world_size=8)
- 128-byte alignment prevents false sharing (cache-line boundary)
- Two peer_counter sets alternate to prevent race conditions

### Buffer Organization per Rank

```
Device Memory:
+----------------------------------------------+
| Signal (~128KB)                              |  Sync metadata
+----------------------------------------------+
| Temporary buffers (for 2-stage)              |  Reduce-scatter intermediate
+----------------------------------------------+
| Data buffer (actual reduction target)        |  Up to 2GB
+----------------------------------------------+
```

### RankData (Buffer Directory)

```cpp
struct RankData {
    const void* __restrict__ ptrs[8];  // Pointers to all ranks' data
};
```

Each rank maintains pointers to all peer buffers via IPC handles.

---

## Synchronization Mechanism

### Multi-GPU Barrier

```cpp
template <int ngpus, bool is_start, bool need_fence = false>
DINLINE void multi_gpu_barrier(const RankSignals& sg, Signal* self_sg, int rank)
```

**Flow:**
1. Increment own counter: `self_sg->self_counter[blockIdx.x][threadIdx.x] += 1`
2. Write to peer's peer_counter with release semantics
3. Spin-wait on own peer_counter with acquire semantics until all ranks signal

**Memory Ordering (Volta+):**
- Store: `st.release.sys.global.u32` (makes prior writes visible)
- Load: `ld.acquire.sys.global.u32` (sees prior stores from other GPUs)

**Two-Set Alternation:**
Uses `val % 2` to index into peer_counter arrays, preventing scenarios where a fast GPU overwrites counter N+1 while a slow GPU is still waiting for counter N.

---

## Packed Type System

```cpp
template <typename T>
struct packed_t {
    using P = array_t<T, 16 / sizeof(T)>;     // 128-bit packed load/store
    using A = array_t<float, 16 / sizeof(T)>;  // Float accumulator
};
```

| Data Type | Packed Elements | Load Size | Accumulator |
|-----------|----------------|-----------|-------------|
| float32 | 4 | 16B (ld.128) | float[4] |
| float16 | 8 | 16B (ld.128) | float[8] |
| bfloat16 | 8 | 16B (ld.128) | float[8] |

**Pattern:** Load 16B → unpack to float → reduce in float → pack back → store 16B

---

## IPC (Inter-Process Communication) Setup

1. **Handle creation**: `cudaIpcGetMemHandle()` creates shareable memory handle
2. **Handle opening**: `cudaIpcOpenMemHandle()` maps remote GPU memory into local address space
3. **Buffer registration**: All ranks exchange IPC handles and register peer memory
4. **Pointer table**: `RankData.ptrs[]` populated with peer buffer addresses

For CUDA graph capture, separate graph-specific buffers are registered via `register_graph_buffers()`.

---

## Kernel Dispatch Logic

```cpp
if (world_size_ == 2) {
    cross_device_reduce_1stage<T, 2><<<36, 512, 0, stream>>>(...);
} else if (full_nvlink_) {
    if ((world_size_ <= 4 && bytes < 512*1024) ||
        (world_size_ <= 8 && bytes < 256*1024)) {
        cross_device_reduce_1stage<T, ngpus><<<36, 512, 0, stream>>>(...);
    } else {
        cross_device_reduce_2stage<T, ngpus><<<36, 512, 0, stream>>>(...);
    }
}
```

**Environment Variable Override:**
```
SGLANG_CUSTOM_ALLREDUCE_ALGO=1stage|oneshot|2stage|twoshot
```

---

## HIP/ROCm Variant (Quick All-Reduce)

### Quantized All-Reduce

The HIP variant supports quantized reduction for bandwidth reduction:

| Codec | Bits | Description |
|-------|------|-------------|
| CodecFP | 16/32 | Full precision |
| CodecQ8 | 8 | 8-bit integer quantization |
| CodecQ6 | 6 | 6-bit integer quantization |
| CodecQ4 | 4 | 4-bit integer quantization |

### Algorithm (AllReduceTwoshot)

Always 2-stage on ROCm:
1. Quantize local data (optional)
2. Reduce-scatter with quantized communication
3. All-gather reduced partitions
4. Dequantize output

---

## MSCCLPP Integration

For multi-socket/multi-node:

| Component | Purpose |
|-----------|---------|
| MemoryChannel | Device-to-device communication |
| PortChannel | Host-to-device communication |
| LLPacket | Quantized communication packets |
| Proxy service | CPU-side cross-node forwarding |

---

## Supported Configurations

| World Size | CUDA | HIP | Notes |
|-----------|------|-----|-------|
| 2 | 1-stage | 2-shot | Full support |
| 4 | 1/2-stage | 2-shot | Size-dependent |
| 6 | 1/2-stage | 2-shot | Full support |
| 8 | 1/2-stage | 2-shot | Full support |
| >8 | Not supported | Not supported | Max 8 pointer slots |

### Data Type Support

| Type | CUDA | HIP |
|------|------|-----|
| float32 | Y | Y |
| float16 | Y | Y |
| bfloat16 | Y (SM>=80) | Y |
| int8 (quantized) | - | Y (CodecQ8) |
| int6 (quantized) | - | Y (CodecQ6) |
| int4 (quantized) | - | Y (CodecQ4) |

---

## Performance Optimizations

1. **128-bit vectorized loads/stores**: Generates `ld.128`/`st.128` instructions for maximum memory bandwidth
2. **Float accumulation**: FP16/BF16 reduced in float32 for precision, packed back on store
3. **Per-thread counters**: No shared memory synchronization bottleneck
4. **Release/acquire semantics**: Minimal fencing overhead vs full `membar.sys`
5. **Alternating counter sets**: Eliminates race conditions without extra barriers
6. **Launch bounds**: `__launch_bounds__(512, 1)` ensures register allocation for 1 block/SM occupancy
7. **Threshold-based algorithm selection**: 1-stage for low latency, 2-stage for bandwidth

---

## Constraints

| Constraint | Value | Reason |
|-----------|-------|--------|
| Max world size | 8 | Pointer array slots in RankData/Signal |
| Input alignment | Multiple of packed size | 128-bit loads require alignment |
| Max data size | ~2 GB | uint32_t element count |
| NVLink requirement | Recommended | PCIe P2P significantly slower |
| Single node only | Yes | IPC handles are node-local |

---

## API Functions

```python
# Python/C++ binding functions
init_custom_ar(fake_ipc_ptrs, rank_data, rank, full_nvlink)  -> handle
all_reduce(handle, inp, out, reg_buffer, reg_buffer_sz)       -> None
register_buffer(handle, fake_ipc_ptrs)                         -> None
get_graph_buffer_ipc_meta(handle)                             -> (handles, offsets)
register_graph_buffers(handle, handles, offsets)               -> None
dispose(handle)                                                -> None
meta_size()                                                    -> sizeof(Signal)
```

---

## Comparison with NCCL

| Aspect | Custom All-Reduce | NCCL |
|--------|------------------|------|
| Latency (small msg) | Lower | Higher (protocol overhead) |
| Bandwidth (large msg) | Similar | Similar or better |
| Multi-node | Not supported | Supported |
| Graph capture | Supported | Limited |
| Setup complexity | Higher (IPC handles) | Lower (auto-discovery) |
| World size limit | 8 GPUs | Unlimited |
| Quantized reduction | HIP only | Not available |
