# Expert Parallelism Implementation

## Overview

SGLang implements Expert Parallelism (EP) for Mixture-of-Experts (MoE) models, distributing experts across multiple GPUs and using all-to-all communication for token routing. The primary implementation uses the DeepEP library for efficient dispatch/combine operations, with support for load balancing via EPLB (Expert Parallel Load Balancing).

**Key Source Files:**
- `python/sglang/srt/layers/moe/ep_moe/layer.py` - DeepEPMoE, MoriEPMoE implementations
- `python/sglang/srt/layers/moe/ep_moe/kernels.py` - Triton dispatch/combine/fused kernels
- `python/sglang/srt/layers/moe/token_dispatcher/deepep.py` - DeepEP dispatcher
- `python/sglang/srt/eplb/` - Expert Parallel Load Balancing
- `python/sglang/srt/layers/moe/fused_moe_triton/layer.py` - FusedMoE base

---

## EP Layer Implementations

### DeepEPMoE (Primary)

**Source:** `python/sglang/srt/layers/moe/ep_moe/layer.py` (~714 lines)

Extends `FusedMoE` for DeepEP-based expert parallelism.

**Dispatch Modes:**
- **Normal mode**: Standard token routing for prefill
- **Low Latency (LL) mode**: Optimized for decode phase
- **AUTO mode**: Runtime selection based on batch type (`is_extend_in_batch`)

```python
class DeepEPMoE(FusedMoE):
    def __init__(self, num_experts, top_k, hidden_size, ...):
        self.deepep_mode: DeepEPMode  # NORMAL, LOW_LATENCY, AUTO
        self.deprecate_flag: bool     # Legacy DeepGEMM path
        self.expert_mask: Tensor      # Invalid expert ID masking
```

**Forward Flow:**
1. `dispatch()` - Route tokens to experts across GPUs
2. `run_moe_core()` - Execute GEMM on distributed experts
3. `combine()` - Gather results back to original token positions

### MoriEPMoE (AMD/ROCm)

Uses Mori library for EP dispatch/combine on AMD GPUs. Optimized for ROCm/HIP with aiter fused MoE operations.

### NpuFuseEPMoE (Ascend NPU)

NPU-specific weight reshaping and format casting for optimal Ascend performance.

---

## DeepEP Dispatch Mechanism

**Source:** `python/sglang/srt/layers/moe/token_dispatcher/deepep.py` (~872 lines)

### Normal Mode (Prefill)

Two-phase async dispatch:
```
dispatch_a() -> dispatch_b() (async)
  Input: hidden_states, topk_output (expert IDs + weights)
  Output: DeepEPNormalDispatchOutput
    - hidden_states: reshaped for expert processing
    - num_recv_tokens_per_expert: token counts per expert
    - topk_ids/weights: routing information
```

**Steps:**
1. Quantize activations (optional FP8, per-token-group with 128-byte groups)
2. Calculate dispatch layout (tokens per expert per GPU)
3. Send via NVL (intra-node) or RDMA (inter-node)
4. Receive at destination experts

### Low Latency Mode (Decode)

Optimized for batch_size < 256:
```
dispatch_a() -> dispatch_b() (faster)
  Output: DeepEPLLDispatchOutput
    - hidden_states: quantized format
    - masked_m: actual token count per expert
    - expected_m: padded token count for GEMM alignment
```

**Optimizations:**
- Packed token format for efficiency
- FP8 quantization for bandwidth reduction
- Silo computation: experts process local tokens first
- Minimal synchronization points

### Combine Phase

```
combine_a() -> combine_b()
  - Reverses dispatch: expert outputs → original token order
  - Weighted aggregation: sum(expert_output * router_weight)
  - Optional overlap with computation stream
```

### DeepEP Buffer Management

```python
class DeepEPBuffer:
    - Allocates NVL buffers for intra-node communication
    - Allocates RDMA buffers for inter-node communication
    - Manages QP (Queue Pairs) per rank
    - buffer.get_dispatch_layout(): tokens per rank/expert distribution
```

---

## Triton Kernels

**Source:** `python/sglang/srt/layers/moe/ep_moe/kernels.py` (~1384 lines)

### Key Kernels

| Kernel | Purpose |
|--------|---------|
| `deepep_permute_triton_kernel` | Routes tokens to assigned expert buffers |
| `deepep_post_reorder_triton_kernel` | Aggregates expert outputs back to token order |
| `compute_src2dst_triton_kernel` | Builds source-to-destination mapping |
| `ep_scatter()` | Routes quantized activations to experts |
| `ep_gather()` | Combines expert outputs with weights |
| `silu_and_mul_masked_post_quant_fwd()` | Fused gate+mul+quantization for LL mode |

### Fused SiLU+Mul+Quantization (LL Mode)

```
Input: [expert, token_padded, 2*hidden]
Process:
  gate = sigmoid(gate_input)
  output = gate * up_input
  output_fp8 = quantize(output)
Output: [expert, token_padded, hidden] in FP8 with per-token scales
```

---

## Expert Distribution Across GPUs

### Distribution Strategy

Each GPU/rank holds:
```
num_local_experts = (num_total_experts - num_shared) / ep_size + num_shared
```

For shared experts (fused): all ranks have all shared experts.

**Example (256 experts, EP=8, 32 shared):**
```
num_routed = 256 - 32 = 224
num_local_routed = 224 / 8 = 28
num_local_total = 28 + 32 = 60 experts/GPU
```

### Physical vs Logical Experts

- **Logical**: User-facing, model-defined (e.g., DeepSeek's 256)
- **Physical**: Actual distributed experts
  - Can equal logical (1:1 mapping)
  - Can exceed logical (redundant replicas)
  - Can be virtual (multiple logical → one physical)

---

## EPLB (Expert Parallel Load Balancing)

**Source:** `python/sglang/srt/eplb/`

### Components

#### EPLBManager (`eplb_manager.py`)

```python
class EPLBManager:
    - Triggered every N iterations (eplb_rebalance_num_iterations)
    - Collects expert usage statistics
    - Triggers rebalancing if utilization < threshold
    - Supports chunked layer updates (don't pause all layers)
```

#### ExpertLocationMetadata (`expert_location.py`)

```python
physical_to_logical_map: (layers, num_physical_experts)
  → logical_id = physical_to_logical_map[layer][physical_id]

logical_to_all_physical_map: (layers, num_logical_experts, max_replicas)
  → physical_ids = logical_to_all_physical_map[layer][logical_id]
```

#### Expert Distribution Recording (`expert_distribution.py`)

Three recorder implementations:

| Recorder | Mode | Description |
|----------|------|-------------|
| `_DetailSinglePassGatherer` | per-token | Exact routes per layer, highest accuracy |
| `_DeepepNormalSinglePassGatherer` | stat_approx | From DeepEP statistics, lower memory |
| `_DeepepLowLatencySinglePassGatherer` | LL mode | Tracks masked_m counts |

**Load Balancing Metrics:**
```
utilization_rate = avg_load / max_load
  → Lower deviation is better (more balanced)
  → Tracked over windows: [10, 100, 1000] iterations
  → Rebalancing triggered if rate < threshold (e.g., 0.8)
```

#### Expert Location Dispatch (`expert_location_dispatch.py`)

Converts logical expert selections to physical:

| Algorithm | Description |
|-----------|-------------|
| Static | Fixed mapping: `physical_id = logical_to_rank_dispatch_physical_map[logical_id]` |
| Dynamic/Random | Load-aware: `random_idx % num_replicas[logical_id]` |
| Fake | For testing |

---

## TP (Tensor Parallelism) Interaction

### Initialization

```python
FusedMoE.__init__():
    self.moe_tp_size = get_moe_tensor_parallel_world_size()
    self.moe_tp_rank = get_moe_tensor_parallel_rank()
    self.intermediate_size_per_partition = intermediate_size // moe_tp_size
```

### Communication Pattern

```
1. If FP4 quantization with TP:
   # All-gather FP4 hidden states across TP group
   topk_weights, topk_ids, x, x_sf = get_tp_group().all_gatherv(...)

2. After combine:
   # Reduce-scatter results across TP group
   get_tp_group().reduce_scatterv(...)
```

### Process Group Hierarchy

```
EP Group: Communication within expert-parallel dimension
TP Group: Communication within tensor-parallel dimension
Combined: EP size × TP size = total world size
```

---

## Data Flow Example

**1 Token, top_k=2, 4 Experts, 2 GPUs (EP=2):**

```
Input: hidden_state [1, 4096]
topk_ids = [0, 3]  → Expert 0 (GPU 0), Expert 3 (GPU 1)
topk_weights = [0.6, 0.4]

DISPATCH (all-to-all):
  GPU 0 → GPU 0: token for Expert 0
  GPU 0 → GPU 1: token for Expert 3

EXECUTE:
  GPU 0: output_0 = expert_0(token)  → [1, intermediate]
  GPU 1: output_3 = expert_3(token)  → [1, intermediate]

COMBINE (all-to-all reverse):
  GPU 0 ← GPU 0: output_0
  GPU 0 ← GPU 1: output_3
  Result = 0.6 * output_0 + 0.4 * output_3 → [1, 4096]
```

---

## Fused MoE Execution Pattern

```
1. Dispatch phase:
   - Token permutation to expert buffers
   - Optional quantization of activations

2. Core MoE (fused per-expert):
   - Gate projection (W1): input → gate
   - Up projection (W1): input → intermediate
   - Element-wise: gate = sigmoid(gate)
   - Multiply: output = gate * intermediate
   - Down projection (W2): intermediate → output

3. Combine phase:
   - Expert output gathering
   - Weight application (router weights)
   - Permutation back to token order
```

### Quantization Variants

| Config | Gate/Up | Down | Combine | Benefits |
|--------|---------|------|---------|----------|
| BF16 | BF16 | BF16 | BF16 | High precision |
| FP8 | FP8 | FP8 | BF16 | Reduced bandwidth |
| FP4 | FP4 | FP4 | FP8 | Very low bandwidth |
| W4A8 | INT8 | W4 | FP8 | Quantized weights |

---

## Dispatch/Combine Backends

| Backend | Dispatch | Combine | Library | Hardware |
|---------|----------|---------|---------|----------|
| DeepEP Normal | 2-phase async | 2-phase async | deep_ep | NVIDIA |
| DeepEP LL | 1-phase fast | 1-phase fast | deep_ep | NVIDIA |
| MoriEP | Single phase | Single phase | mori | AMD |
| FuseEP (NPU) | Custom | Custom | ascend | Ascend |
| Standard | Local only | Local only | N/A | Any |
| FlashInfer | Custom | Custom | flashinfer | NVIDIA |

---

## Configuration Parameters

| Parameter | Effect |
|-----------|--------|
| `SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK` | Max tokens for LL mode |
| `SGLANG_DEEPEP_BF16_DISPATCH` | Use BF16 instead of FP8 for dispatch |
| `SGLANG_USE_AITER` | Enable aiter fused MoE (AMD/HIP) |
| `eplb_rebalance_num_iterations` | Iterations between load balancing |
| `eplb_min_rebalancing_utilization_threshold` | Trigger threshold |
| `ep_dispatch_algorithm` | "static" or "dynamic" expert routing |

---

## Performance Optimizations

1. **Async Dispatch/Combine**: Separate stream overlaps communication with computation
2. **Quantization**: Reduces network bandwidth during all-to-all
3. **TMA-aligned Scales**: Tensor Memory Accelerator optimization
4. **Expert Masking**: Efficiently handles invalid expert IDs without branching
5. **Fused Kernels**: Combines gate, multiply, and quantization into single kernel
6. **Load Balancing**: Dynamically migrates experts to underutilized GPUs
7. **Streaming Combine**: Applies output weighting while receiving data

---

## File Structure

```
python/sglang/srt/layers/moe/
├── ep_moe/
│   ├── layer.py (714 lines)      # DeepEPMoE, MoriEPMoE
│   ├── kernels.py (1384 lines)   # Triton dispatch/combine kernels
│   └── __init__.py
├── token_dispatcher/
│   ├── deepep.py (872 lines)     # DeepEP dispatcher (normal + LL)
│   ├── base.py                   # BaseDispatcher protocol
│   ├── moriep.py                 # Mori EP dispatcher
│   ├── fuseep.py                 # Ascend FuseEP dispatcher
│   └── standard.py              # Standard (no EP) dispatcher
└── fused_moe_triton/
    └── layer.py                  # FusedMoE base integration

python/sglang/srt/eplb/
├── eplb_manager.py (118 lines)   # Rebalancing orchestration
├── expert_distribution.py        # Statistics collection
├── expert_location.py            # Expert metadata/mapping
├── expert_location_dispatch.py   # Routing algorithms
└── expert_location_updater.py    # Dynamic updates
```
