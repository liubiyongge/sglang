# Data Model: SGLang System Architecture

**Date**: 2026-04-18 | **Branch**: `20260418-sglang-comprehensive-analysis`

---

## Core Entities & Relationships

### 1. Request Lifecycle Entities

```
┌─────────────────────────────────────────────────────────────┐
│                    REQUEST LIFECYCLE                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  GenerateReqInput ──→ TokenizedGenerateReqInput ──→ Req    │
│  (user HTTP)          (after tokenizer)             (sched) │
│                                                             │
│  Req states:                                                │
│    WAITING → PREFILL → RUNNING (decode) → FINISHED         │
│                  ↑                  ↓                        │
│                  └── RETRACTED ←────┘ (memory pressure)     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

#### GenerateReqInput
- **Fields**: text, input_ids, sampling_params, stream, lora_path, image_data
- **Source**: HTTP/gRPC/Engine API
- **Transitions to**: TokenizedGenerateReqInput (via TokenizerManager)

#### Req (python/sglang/srt/managers/schedule_batch.py)
- **Fields**: 
  - `rid`: Request ID
  - `input_ids`: Token sequence
  - `req_pool_idx`: Index in ReqToTokenPool
  - `origin_input_ids`: Original tokens (for prefix cache match)
  - `prefix_indices`: Matched prefix KV indices
  - `output_ids`: Generated tokens
  - `sampling_params`: Temperature, top_p, max_tokens, etc.
  - `lora_path`: LoRA adapter path
  - `stream`: Whether to stream output
- **State**: waiting → extend (prefill) → decode → finished
- **Relationships**: 
  - Belongs to ScheduleBatch
  - References RadixCache nodes (via prefix_indices)
  - Allocated in ReqToTokenPool

#### ScheduleBatch (python/sglang/srt/managers/schedule_batch.py)
- **Fields**:
  - `reqs`: List[Req]
  - `forward_mode`: ForwardMode (EXTEND/DECODE/IDLE)
  - `batch_size`: Number of requests
  - `input_ids`: Flattened token tensor
  - `seq_lens`: Sequence lengths per request
  - `prefix_lens`: Prefix lengths (from cache)
  - `out_cache_loc`: GPU cache locations for new tokens
  - `spec_info`: Speculative decoding info
- **Transitions to**: ModelWorkerBatch (for GPU execution)
- **Operations**: filter(), merge(), retract()

---

### 2. Memory Management Entities

```
┌────────────────────────────────────────────────────────────────┐
│                   MEMORY HIERARCHY                               │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  RadixCache (Prefix Tree)                                      │
│    └── TreeNode                                                │
│          ├── key: RadixKey (token_ids + extra_key)             │
│          ├── value: Tensor (GPU KV indices) or None (evicted) │
│          ├── host_value: Tensor (CPU backup)                   │
│          ├── children: Dict[token, TreeNode]                   │
│          ├── lock_ref: int (protection counter)                │
│          └── hit_count: int (for LFU eviction)                 │
│                                                                │
│  ReqToTokenPool                                                │
│    └── req_to_token: Tensor[batch_size, max_context_len]      │
│                                                                │
│  TokenToKVPool (GPU Memory)                                    │
│    ├── MHATokenToKVPool                                        │
│    │     ├── k_buffer[layer]: Tensor[size, heads, head_dim]   │
│    │     └── v_buffer[layer]: Tensor[size, heads, head_dim]   │
│    ├── MLATokenToKVPool                                        │
│    │     └── kv_buffer[layer]: Tensor[size, 1, kv_dim]        │
│    └── FP4/FP8 Variants (quantized)                           │
│                                                                │
│  TokenToKVPoolAllocator                                        │
│    ├── free_pages: List[int]                                   │
│    ├── page_size: int (1 or power-of-2)                       │
│    └── Operations: alloc_extend(), alloc_decode(), free()      │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

#### TreeNode
- **Validation**: lock_ref >= 0, value is None iff evicted
- **State Transitions**: 
  - Created (new KV computed) → Active (lock_ref > 0, in use) → Evictable (lock_ref == 0) → Evicted (value=None, host_value retained)
- **Relationships**: Parent-child tree structure, references GPU memory indices

#### HiRadixCache (extends RadixCache)
- **Additional State**: 
  - host_ref_counter per node
  - write_policy (write_back/write_through/write_through_selective)
  - HiCacheController (async transfer manager)
  - Storage backend (L3)
- **Operations**: evict_to_host(), load_back(), prefetch_from_storage()

---

### 3. Parallelism Entities

```
┌──────────────────────────────────────────────────────────┐
│               DISTRIBUTED TOPOLOGY                        │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  GroupCoordinator                                         │
│    ├── rank: int                                         │
│    ├── world_size: int                                   │
│    ├── process_group: ProcessGroup                       │
│    ├── pynccl_comm: Optional[PyNcclCommunicator]        │
│    ├── custom_ar: Optional[CustomAllReduce]             │
│    └── Operations: all_reduce(), all_gather(), send()    │
│                                                          │
│  Parallel Groups (4 simultaneous):                       │
│    ├── TP group (tensor parallel)                        │
│    ├── PP group (pipeline parallel)                      │
│    ├── EP group (expert parallel)                        │
│    └── DP-Attention group                                │
│                                                          │
│  ExpertLocation (for EPLB)                               │
│    ├── logical_to_physical: Dict[int, List[int]]        │
│    ├── per_layer_distribution: List[List[int]]          │
│    └── replica_count: Dict[int, int]                    │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

---

### 4. Model Execution Entities

```
┌──────────────────────────────────────────────────────────┐
│               MODEL EXECUTION PIPELINE                    │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  ModelRunner                                              │
│    ├── model: nn.Module (the actual LLM)                 │
│    ├── cuda_graph_runner: CudaGraphRunner                │
│    ├── attn_backend: AttentionBackend                    │
│    └── Forward: input_metadata → hidden_states → logits │
│                                                          │
│  ForwardBatchInfo                                         │
│    ├── forward_mode: ForwardMode                         │
│    ├── batch_size: int                                   │
│    ├── seq_lens: Tensor                                  │
│    ├── positions: Tensor                                 │
│    ├── req_pool_indices: Tensor                          │
│    ├── out_cache_loc: Tensor                             │
│    └── attn_metadata: backend-specific                   │
│                                                          │
│  CudaGraphRunner                                          │
│    ├── captured_graphs: Dict[batch_size, CUDAGraph]      │
│    ├── input_buffers: Dict[str, Tensor]                  │
│    ├── output_buffers: Dict[str, Tensor]                 │
│    └── Operations: capture(), replay(), can_run()        │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

---

### 5. Quantization Entities

```
┌──────────────────────────────────────────────────────────┐
│              QUANTIZATION SYSTEM                          │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  QuantizationConfig (abstract)                           │
│    ├── get_quant_method(layer, prefix) → QuantMethod     │
│    ├── get_min_capability() → int                        │
│    └── from_config(hf_config) → Self                     │
│                                                          │
│  QuantizeMethodBase (abstract)                           │
│    ├── create_weights(layer, ...)                        │
│    ├── apply(layer, x, bias) → Tensor                   │
│    └── process_weights_after_loading(layer)              │
│                                                          │
│  Weight Parameters:                                      │
│    ├── ModelWeightParameter (base quantized weight)      │
│    ├── PerTensorScaleParameter (single scale)            │
│    ├── BlockQuantScaleParameter (block-wise scales)      │
│    ├── ChannelQuantScaleParameter (per-channel)          │
│    └── GroupQuantScaleParameter (per-group, e.g. AWQ)    │
│                                                          │
│  KV Cache Quantization:                                  │
│    ├── k_scale: Parameter (per-layer)                    │
│    ├── v_scale: Parameter (per-layer)                    │
│    └── Applied in: RadixAttention.forward()              │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

---

### 6. Speculative Decoding Entities

```
┌──────────────────────────────────────────────────────────┐
│           SPECULATIVE DECODING                           │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  EAGLEWorker                                              │
│    ├── draft_model: nn.Module (lightweight)              │
│    ├── target_worker: TpModelWorker                      │
│    ├── topk: int (candidates per step)                   │
│    ├── num_steps: int (draft depth)                      │
│    └── Operations: draft() → verify() → accept/reject   │
│                                                          │
│  NgramCache (C++ implementation)                         │
│    ├── capacity: int (max tokens stored)                 │
│    ├── window_sizes: (min, max)                          │
│    └── Operations: batch_get(tokens) → predictions      │
│                                                          │
│  SpecInfo (per-batch)                                    │
│    ├── draft_token_ids: Tensor                           │
│    ├── draft_tree_mask: Tensor                           │
│    ├── accept_indices: Tensor                            │
│    └── accepted_length: int                              │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

---

### 7. Diffusion Runtime Entities

```
┌──────────────────────────────────────────────────────────┐
│           DIFFUSION RUNTIME                              │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  DiffusionReq                                             │
│    ├── sampling_params: SamplingParams                    │
│    ├── prompt_embeds: List[Tensor]                       │
│    ├── latents: Tensor (noise/state)                     │
│    ├── timesteps: Tensor                                 │
│    ├── step_index: int                                   │
│    ├── teacache_params: TeaCacheParams                   │
│    └── output: Tensor (final image/video)                │
│                                                          │
│  Pipeline (composed stages)                              │
│    ├── stages: List[PipelineStage]                       │
│    ├── text_encoder: TextEncoder                         │
│    ├── transformer: DiT                                  │
│    ├── vae: VAE                                          │
│    └── scheduler: DiffusionScheduler                     │
│                                                          │
│  TeaCacheContext                                           │
│    ├── thresh: float (L1 distance threshold)             │
│    ├── coefficients: List[float]                         │
│    ├── accumulated_distance: float                       │
│    └── cached_output: Optional[Tensor]                   │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

---

### 8. LoRA Entities

```
┌──────────────────────────────────────────────────────────┐
│               LoRA MANAGEMENT                            │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  LoRAManager                                              │
│    ├── loaded_adapters: Dict[str, LoRAAdapter]           │
│    ├── memory_pool: LoRAMemoryPool                       │
│    ├── eviction_policy: LRU/LFU                          │
│    └── max_loras_per_batch: int                          │
│                                                          │
│  LoRAAdapter                                              │
│    ├── lora_A: Dict[module_name, Tensor]                 │
│    ├── lora_B: Dict[module_name, Tensor]                 │
│    ├── rank: int                                         │
│    ├── alpha: float                                      │
│    └── target_modules: List[str]                         │
│                                                          │
│  LoRABatchInfo                                            │
│    ├── seg_lens: List[int] (tokens per adapter)          │
│    ├── weight_indices: List[int]                         │
│    └── lora_ranks: List[int]                             │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

---

## Entity Relationship Diagram

```
                    ┌────────────────┐
                    │   Engine       │
                    └───────┬────────┘
                            │ spawns
            ┌───────────────┼───────────────┐
            ↓               ↓               ↓
   ┌────────────────┐  ┌────────┐  ┌──────────────────┐
   │TokenizerManager│  │Scheduler│  │DetokenizerManager│
   └────────────────┘  └───┬────┘  └──────────────────┘
                           │ manages
              ┌────────────┼────────────┐
              ↓            ↓            ↓
    ┌──────────────┐ ┌──────────┐ ┌──────────────┐
    │ScheduleBatch │ │RadixCache│ │TpModelWorker │
    │  (reqs)      │ │ (memory) │ │  (GPU exec)  │
    └──────┬───────┘ └────┬─────┘ └──────┬───────┘
           │              │              │
           ↓              ↓              ↓
    ┌─────────────┐ ┌──────────────┐ ┌──────────────┐
    │     Req     │ │  TreeNode    │ │  ModelRunner │
    │ (request)   │ │ (cache node) │ │ (forward)    │
    └─────────────┘ └──────────────┘ └──────────────┘
           │                                │
           ↓                                ↓
    ┌─────────────┐                  ┌──────────────┐
    │SamplingParams│                  │CudaGraphRunner│
    └─────────────┘                  └──────────────┘
```

---

## State Machines

### Request State Machine
```
                    ┌─────────┐
                    │ WAITING │←──────────────────────┐
                    └────┬────┘                       │
                         │ scheduled                  │ retracted
                         ↓                           │
                    ┌─────────┐                      │
                    │ PREFILL │                       │
                    └────┬────┘                       │
                         │ prefill done              │
                         ↓                           │
                    ┌─────────┐                      │
                    │ DECODE  │──────────────────────┘
                    └────┬────┘
                         │ EOS/max_tokens
                         ↓
                    ┌─────────┐
                    │FINISHED │
                    └─────────┘
```

### TreeNode State Machine
```
    ┌─────────┐     lock_ref++     ┌────────┐
    │EVICTABLE│────────────────────→│ LOCKED │
    └────┬────┘←───────────────────┘└────────┘
         │          lock_ref--
         │ evict()
         ↓
    ┌─────────┐     load_back()    ┌─────────┐
    │ EVICTED │────────────────────→│EVICTABLE│
    └─────────┘                     └─────────┘
```
