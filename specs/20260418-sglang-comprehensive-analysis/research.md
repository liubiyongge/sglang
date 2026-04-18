# Research: Comprehensive SGLang Architecture Analysis

**Date**: 2026-04-18 | **Branch**: `20260418-sglang-comprehensive-analysis`

---

## 1. Framework Architecture Overview

### Decision: Multi-Process Architecture with ZMQ Communication
**Rationale**: SGLang uses a multi-process architecture with ZeroMQ (ZMQ) for inter-process communication, enabling CPU-bound tokenization to run independently from GPU-bound inference.

**Process Model:**
```
User Request
    ↓
[TokenizerManager Process] ─── tokenize + template ──→ ZMQ ───→ [Scheduler Process]
    ↑                                                              ↓
    │                                                     [TpModelWorker (GPU)]
    │                                                              ↓
[DetokenizerManager] ←── ZMQ ←── results ─────────────────────────┘
    ↓
User Response
```

**Key Components:**
1. **Engine** (`entrypoints/engine.py`): Top-level entry point, spawns all processes
2. **TokenizerManager**: Handles tokenization, chat templates, multimodal preprocessing
3. **Scheduler**: Core event loop - request queuing, batching, memory management
4. **TpModelWorker**: GPU execution, model forward passes, CUDA graphs
5. **DetokenizerManager**: Streaming detokenization of output tokens
6. **DataParallelController**: Routes requests across DP workers with load balancing

**Alternatives Considered:**
- Single-process (simpler but CPU-bound tokenization blocks GPU)
- gRPC (higher overhead than ZMQ for internal communication)
- Ray-based (used in vLLM, but SGLang prefers direct process management for lower overhead)

---

## 2. Scheduling & Continuous Batching

### Decision: Zero-Overhead Continuous Batching with Overlap Scheduling
**Rationale**: The scheduler implements a continuous batching system that dynamically adds/removes requests from running batches, achieving near-zero scheduling overhead through overlap with GPU computation.

**Two Event Loop Modes:**
1. **Normal** (`event_loop_normal`): Sequential receive→schedule→forward→process
2. **Overlap** (`event_loop_overlap`): CPU scheduling overlapped with GPU forward pass

**Batch Formation:**
- **PrefillAdder**: Incrementally builds prefill batches respecting token budgets
- Token budget split: `rem_input_tokens` (prefill) vs `rem_total_tokens` (total)
- Chunked prefill: Long sequences split into manageable chunks
- Mixed-chunk: Simultaneous prefill + decode in one batch

**Scheduling Policies:**
| Policy | Description | Best For |
|--------|-------------|----------|
| LPM (Longest Prefix Match) | Prioritize cache hits | Multi-turn chat |
| DFS-Weight | Depth-first tree traversal | Tree-structured workloads |
| FCFS | First-come first-served | Fairness |
| LOF | Longest-output-first | Balanced throughput |
| ROUTING_KEY | Route by key | DP attention |

**Decode Retraction**: When GPU memory is exhausted, running decode requests are paused and re-queued, with `new_token_ratio` dynamically adjusted.

---

## 3. Memory Management & RadixAttention

### Decision: Radix Tree + Paged KV Cache + 3-Tier Hierarchical Storage
**Rationale**: The radix tree enables O(prefix_length) prefix matching for KV cache reuse, while paged allocation minimizes fragmentation. HiCache extends this to CPU/storage for capacity.

**RadixAttention Architecture:**
```
Radix Tree (prefix cache)
    ├── Node: token_ids → GPU KV cache indices (torch.Tensor)
    ├── Node splitting on partial match
    ├── Lock reference counting (protect in-use nodes)
    └── Eviction: LRU/LFU/FIFO/Priority policies

Memory Pools:
    ├── ReqToTokenPool: request_id → token positions
    └── TokenToKVPool: token positions → actual KV buffers
        ├── MHATokenToKVPool: [size, num_heads, head_dim]
        ├── MLATokenToKVPool: [size, 1, kv_dim] (compressed)
        └── FP4/FP8 variants (quantized)
```

**HiCache (Hierarchical Caching):**
- **L1 (GPU)**: Hot data, microsecond access
- **L2 (CPU)**: Warm data, millisecond access, configurable ratio
- **L3 (Storage)**: Cold data, persistent (file/NIXL/Mooncake/HF3FS backends)
- Write policies: write_back, write_through, write_through_selective
- Async load-back with prefetch from storage

**Alternatives Considered:**
- Flat hash map (no prefix sharing, higher memory)
- vLLM PagedAttention only (no prefix tree, less efficient for multi-turn)
- Fixed-size block cache (less flexible than radix tree)

---

## 4. Parallelism Strategies

### Decision: Five Composable Parallelism Dimensions
**Rationale**: Different model sizes and architectures require different parallelism strategies. SGLang supports all five composable strategies to handle from single-GPU to rack-scale deployments.

### 4.1 Tensor Parallelism (TP)
- Splits hidden dimensions across GPUs within a layer
- Communication: all-reduce after each linear layer
- Backend hierarchy: Custom AllReduce (small) → PyNCCL (large) → Symmetric Memory (very large)
- Separate TP groups for attention (DP-aware)

### 4.2 Pipeline Parallelism (PP)
- Splits sequential layers across GPUs
- Communication: P2P send/recv between stages
- Microbatch scheduling with circular buffer
- Async depth configurable (`pp_async_batch_depth`)

### 4.3 Expert Parallelism (EP)
- Distributes MoE experts across GPUs
- Communication: All-to-All dispatch/combine
- EPLB (Expert Parallel Load Balancing): Dynamic rebalancing
- DeepEP backend for optimized dispatch

### 4.4 Data Parallelism for Attention (DP-Attention)
- Splits sequence dimension across ranks
- Two modes: MAX_LEN (all_gather) vs SUM_LEN (all_reduce)
- Reduces KV cache memory per GPU
- Enables larger batch sizes

### 4.5 Prefill-Decode Disaggregation (PD)
- Separate server groups for prefill and decode
- KV cache transfer via NIXL/Mooncake/TCP
- Bootstrap + handshake protocol for coordination
- Optimizes for different compute characteristics

**Composition Example (96 H100 GPUs):**
```
- 48 GPUs for Prefill (TP=8, PP=1, EP=6)
- 48 GPUs for Decode (TP=8, PP=1, EP=6)
- DataParallelController routes between groups
```

**Communication Backends:**
- PyNCCL (default, most workloads)
- Custom AllReduce (tensors < 100KB)
- PyMsccl++ (CUDA graph mode)
- Torch Symmetric Memory (large tensors)
- Gloo (CPU barriers)

---

## 5. Quantization System

### Decision: Pluggable Quantization with 35+ Methods and Multi-Backend Dispatch
**Rationale**: Different deployment scenarios require different precision/performance tradeoffs. SGLang supports the widest range of quantization formats with automatic kernel dispatch.

**Supported Formats:**

| Category | Methods | Precision |
|----------|---------|-----------|
| Float8 | FP8, FP8-block, FBGEMM-FP8 | E4M3FN/E4M3FNUZ |
| Float4 | FP4, MXFP4, NvFP4 | 4-bit float |
| Int8 | W8A8-INT8, blockwise-INT8 | 8-bit integer |
| Int4 | W4A8, W4A16 | 4-bit weights |
| PTQ | AWQ, GPTQ, AutoRound, Quark | Various |
| Other | BitsAndBytes, GGUF, QoQ | Various |
| KV Cache | FP8 KV, FP4 KV | Reduced precision |

**Architecture:**
```
QuantizationConfig (abstract)
    ├── get_quant_method() → QuantizeMethodBase
    │   ├── create_weights()     # Initialize quantized tensors
    │   ├── apply()              # Quantized matmul forward
    │   └── process_weights_after_loading()  # Post-load transforms
    │
    └── Implementations:
        ├── Fp8Config → Fp8LinearMethod
        ├── GPTQConfig → GPTQLinearMethod (+ Marlin acceleration)
        ├── AWQConfig → AWQLinearMethod (+ Marlin acceleration)
        ├── W8A8Int8Config → W8A8Int8LinearMethod
        ├── Mxfp4Config → Mxfp4LinearMethod
        └── ... (35+ total)
```

**Kernel Dispatch Logic:**
```
FP8 → if SM90: CUTLASS native FP8
     → elif Marlin available: Marlin FP8
     → elif block_quant: Triton block kernel
     → else: fallback FP8 linear

INT4/AWQ/GPTQ → if Marlin: Marlin kernel (fastest)
              → else: custom CUDA kernel
```

**KV Cache Quantization:**
- Per-layer K/V scales (float32)
- Applied before storage, dequantized on fetch
- Reduces KV cache memory by 50% (FP8) or 75% (FP4)

---

## 6. Attention Backends

### Decision: 25+ Pluggable Attention Backends via Registry
**Rationale**: Different hardware, model architectures, and optimization strategies require different attention implementations.

**Key Backends:**
| Backend | Use Case |
|---------|----------|
| FlashInfer | Default for NVIDIA, supports paged KV, GQA, MLA |
| FlashAttention | Dense attention, prefill optimization |
| CUTLASS MLA | Multi-Latent Attention (DeepSeek) |
| FlashMLA | Flash-based MLA implementation |
| Triton | Portable, custom patterns |
| TBO (Token-Based Overlap) | Overlaps attention with MoE dispatch |
| Double Sparsity | DeepSeek sparse attention |
| NSA (Native Sparse Attention) | Quantized sparse patterns |
| Intel AMX | CPU inference on Xeon |
| Wave | AMD-specific optimizations |
| Torch Flex | PyTorch native SDPA |

**Attention Registry:**
- Backend selection based on hardware, model type, features required
- Fallback chain for unsupported configurations
- Unified interface: `init_forward_metadata()` + `forward()`

---

## 7. Diffusion Model Support

### Decision: Separate Runtime with Stage-Based Pipeline Architecture
**Rationale**: Diffusion inference has fundamentally different characteristics from autoregressive LLM serving (fixed-length iterative denoising vs variable-length token generation).

**Architecture Comparison:**
| Aspect | LLM Serving | Diffusion Serving |
|--------|-------------|-------------------|
| Generation | Token-by-token | Fixed N-step denoising |
| Batching | Dynamic continuous | Per-request pipeline |
| Memory | Growing KV cache | Fixed latent tensors |
| Optimization | Prefix caching, speculative | TeaCache, block skipping |

**Supported Models:**
- WAN (1.3B-14B, T2V, I2V, 480P/720P)
- Flux (1/2/Klein)
- Hunyuan Video
- LTX-2 (video + audio)
- Qwen-Image (generation + editing)
- GLM-Image
- LLaDA2 (Diffusion LLM)

**Pipeline Stages:**
```
Input Validation → Text Encoding → Conditioning → Timestep Prep
    → Latent Prep → Denoising Loop (N steps) → VAE Decode → Post-Process
```

**Optimizations:**
- TeaCache: Skip redundant steps when consecutive timesteps are similar
- Cache-DiT: Block-level caching with TaylorSeer calibration
- Sequence Parallelism (Ulysses/Ring) for long video
- CFG Parallelism: Dedicate GPUs to positive/negative branches

---

## 8. Speculative Decoding

### Decision: EAGLE (v1/v2/Multi-Layer) + N-Gram with Tree Verification
**Rationale**: Speculative decoding provides 2-5x speedup for autoregressive generation by drafting multiple tokens in parallel and verifying in batch.

**EAGLE Architecture:**
```
Target Model (full) ←── verify ←── Draft Model (lightweight)
     ↓                                    ↓
  Accept/reject              Generate tree of candidates
     ↓                                    ↓
  Output tokens              topk^depth total candidates
```

**Versions:**
- **V1**: Sequential draft→verify
- **V2**: Overlapped execution (draft and verify in pipeline)
- **Multi-Layer**: Multiple draft heads at different transformer layers

**N-Gram Speculative:**
- Statistical predictions from sequence history
- C++ n-gram cache for performance
- BFS tree construction with configurable branching
- Best for repetitive/copying tasks

**CUDA Graph Integration:**
- Pre-captured draft forward passes
- Replay mechanism for 0-copy execution
- Separate graph runners for draft extend operations

---

## 9. Structured Output Generation

### Decision: XGrammar Backend with Triton-Accelerated Mask Application
**Rationale**: Grammar-constrained decoding ensures outputs conform to JSON schemas, regex patterns, or custom EBNF grammars.

**Pipeline:**
```
Token Generation → XGrammar fills vocab mask → Triton kernel applies mask → Constrained sampling
```

**Backends:**
- XGrammar (primary): JSON schema, regex, EBNF, structural tags
- Outlines: Traditional FSM-based approach
- LLGuidance: Complex grammar support
- ReasonerGrammar: Reasoning-specific schemas

**Jump Forward**: Accelerates repetitive sequences (e.g., JSON structural tokens).

---

## 10. LoRA Adapter Management

### Decision: S-LoRA Style Multi-Tenant with Triton Kernels
**Rationale**: Enables efficient serving of many fine-tuned models from a single base model deployment.

**Architecture:**
- LoRAManager: Lifecycle management (load, unload, evict)
- Memory pool allocation for adapter weights
- Batched LoRA computation via segmented GEMM
- Max adapters per batch configurable
- Eviction policies: LRU, LFU

**Triton Kernels:**
- `sgemm_lora_a_fwd`: LoRA-A projection
- `sgemm_lora_b_fwd`: LoRA-B + base output fusion
- `qkv_lora_b_fwd`: QKV-specific optimization
- `chunked_sgmv_expand/shrink`: Token routing

---

## 11. sgl-kernel (Compiled CUDA Kernels)

### Decision: Separate Compiled Kernel Library for Performance-Critical Operations
**Rationale**: Ahead-of-time compiled CUDA kernels provide maximum performance for operations that cannot be efficiently expressed in Triton.

**Kernel Categories:**
| Category | Kernels |
|----------|---------|
| attention/ | MLA (CUTLASS SM100), cascade, merge states, vertical-slash index |
| gemm/ | Marlin (AWQ/GPTQ), FP8 blockwise, INT8, per-token quant, nvfp4 |
| moe/ | Marlin MoE WNA16, CUTLASS MoE, W4A8 MoE |
| quantization/ | GGUF dequant |
| memory/ | KV cache store kernel |
| allreduce/ | Custom all-reduce for small tensors |
| speculative/ | Tree verification kernels |
| elementwise/ | Fused activation, normalization |
| mamba/ | Causal conv1d for state-space models |

---

## 12. JIT Kernel System

### Decision: Triton + CuTe DSL for Portable High-Performance Kernels
**Rationale**: JIT compilation provides performance close to hand-written CUDA while being more maintainable and portable across GPU architectures.

**Components:**
- **Triton kernels**: RoPE, normalization, quantization, attention
- **CuTe DSL kernels**: Diffusion-specific fused ops (scale-residual-norm-scale-shift)
- **Compilation cache**: Hash-based kernel specialization cache
- **Benchmark suite**: Per-kernel performance testing

---

## 13. Model Support Breadth

### Decision: 160+ Model Implementations via Unified Model Interface
**Rationale**: SGLang supports the widest range of models through a consistent `forward()` interface with model-specific optimizations.

**Model Categories:**
- **Dense LLMs**: Llama, Qwen, Mistral, Gemma, GPT, DeepSeek, etc.
- **MoE LLMs**: Mixtral, DeepSeek-V2/V3, Qwen-MoE, GLM-MoE
- **Multimodal**: LLaVA, InternVL, Qwen-VL, Gemma3-MM, mLLaMA
- **Embedding**: e5-mistral, gte, mcdse
- **Reward Models**: Skywork, Gemma2-Reward
- **Diffusion LLM**: LLaDA2
- **State-Space**: Falcon-H1 (Mamba), NemotronH

**Model Registry** (`models/registry.py`): Maps model architecture names to implementation classes.

---

## 14. Hardware Support

### Decision: Multi-Platform with Hardware-Specific Optimizations
**Rationale**: Production deployments span diverse hardware. SGLang provides optimized code paths per platform.

**Platforms:**
| Platform | Optimizations |
|----------|--------------|
| NVIDIA (SM80+) | CUTLASS, FlashInfer, Custom AllReduce, CUDA Graphs |
| NVIDIA (SM90/SM100) | MLA kernels, FP8 native, NvFP4, DeepGEMM |
| AMD MI300x/MI355 | AITER, Wave attention, ROCm MoE, Quick AllReduce |
| Intel Xeon | AMX backend, CPU graph runner, INT8 AMX GEMM |
| Google TPU | SGLang-Jax backend |
| Huawei Ascend | NPU backend, Ascend communicators |

---

## 15. Frontend Language (SGLang DSL)

### Decision: Python-Embedded DSL with IR Compilation
**Rationale**: Provides a high-level programming model for complex LLM programs (multi-turn, branching, tool use) that compiles to efficient backend calls.

**Architecture:**
```
User Code (Python + SGLang decorators)
    ↓
Tracer (captures execution)
    ↓
IR (intermediate representation)
    ↓
Interpreter (executes against backend)
    ↓
Backend (RuntimeEndpoint / OpenAI / Anthropic / etc.)
```

**Key APIs:**
- `gen()`: Generate text with constraints
- `select()`: Choose from options
- `function()`: Define reusable SGLang programs
- `image()`, `video()`: Multimodal inputs
- `separate_reasoning()`: Reasoning extraction

---

## Summary of Architectural Decisions

| Component | Decision | Key Metric |
|-----------|----------|------------|
| IPC | ZeroMQ (not gRPC/Ray) | Lower latency |
| Scheduling | Continuous batching + overlap | Zero overhead |
| Caching | RadixAttention + 3-tier | 5x inference speedup |
| Parallelism | 5 composable dimensions | Rack-scale support |
| Quantization | 35+ methods + auto dispatch | Maximum flexibility |
| Diffusion | Separate stage-based runtime | Model-specific optimization |
| Speculation | EAGLE v2 overlapped | 2-5x decode speedup |
| Kernels | sgl-kernel + Triton JIT | Platform portability |
| Models | 160+ unified interface | Broad compatibility |
