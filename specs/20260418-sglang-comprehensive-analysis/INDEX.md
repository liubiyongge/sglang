# SGLang Comprehensive Code Analysis — Master Index

## About This Analysis

This directory contains a comprehensive deep-dive analysis of the SGLang codebase, covering the LLM serving runtime, diffusion model runtime, and all major subsystems. Each document provides architectural analysis, key class descriptions, data flow diagrams, and source file references.

**Repository**: https://github.com/sgl-project/sglang  
**Analysis Date**: 2026-04-18

---

## Document Index by Category

### Foundational Architecture

| Document | Description |
|----------|-------------|
| [architecture-overview.md](./architecture-overview.md) | Process model, startup flow, multi-process topology |
| [engine-startup.md](./engine-startup.md) | Engine initialization sequence and component wiring |
| [ipc-topology.md](./ipc-topology.md) | ZeroMQ IPC architecture, message routing |
| [server-args-catalog.md](./server-args-catalog.md) | Complete ServerArgs reference |
| [model-interface.md](./model-interface.md) | Model registration, EntryClass, forward() contract |
| [configuration.md](./configuration.md) | Environment variable system, 150+ typed flags |

---

### US1: Request Scheduling & Batch Formation

| Document | Description |
|----------|-------------|
| [scheduler-event-loop.md](./scheduler-event-loop.md) | Scheduler event loop, forward modes, overlap scheduling |
| [scheduling-policies.md](./scheduling-policies.md) | LPM, DFS-Weight, FCFS, LOF, random policies |
| [batch-formation.md](./batch-formation.md) | PrefillAdder, continuous batching, chunked prefill |
| [batch-data-structures.md](./batch-data-structures.md) | Req, ScheduleBatch, ForwardBatch, ModelWorkerBatch |
| [model-execution.md](./model-execution.md) | ModelRunner, CUDA graphs, forward pipeline |
| [decode-retraction.md](./decode-retraction.md) | Preemption, priority scheduling, retraction |

---

### US2: Memory Management & KV Caching

| Document | Description |
|----------|-------------|
| [memory-pools.md](./memory-pools.md) | ReqToTokenPool, TokenToKVPool, MHATokenToKVPool |
| [radix-cache.md](./radix-cache.md) | RadixCache, prefix sharing, eviction policies |
| [hicache.md](./hicache.md) | Hierarchical caching (GPU → CPU → Disk) |
| [kv-cache-quant.md](./kv-cache-quant.md) | KV cache quantization (FP8, INT4) |
| [kv-store-kernel.md](./kv-store-kernel.md) | Triton kernels for KV store/load operations |

---

### US3: Parallelism & Distribution

| Document | Description |
|----------|-------------|
| [tensor-parallelism.md](./tensor-parallelism.md) | TP implementation, column/row parallel layers |
| [dp-attention.md](./dp-attention.md) | Data-parallel attention, DP workers |
| [pipeline-parallelism.md](./pipeline-parallelism.md) | PP across nodes, micro-batching |
| [expert-parallelism.md](./expert-parallelism.md) | MoE expert distribution, EP groups |
| [pd-disaggregation.md](./pd-disaggregation.md) | Prefill-decode disaggregation, KV transfer |
| [group-coordinator.md](./group-coordinator.md) | Process group management, rank mapping |
| [custom-allreduce-kernel.md](./custom-allreduce-kernel.md) | Custom NCCL-free all-reduce via IPC |

---

### US4: Quantization

| Document | Description |
|----------|-------------|
| [quant-architecture.md](./quant-architecture.md) | Quantization system architecture, method registry |
| [fp8-quantization.md](./fp8-quantization.md) | FP8 formats, per-tensor/per-channel/block scaling |
| [fp4-quantization.md](./fp4-quantization.md) | FP4 (NF4/E2M1) with double quantization |
| [int4-quantization.md](./int4-quantization.md) | GPTQ, AWQ, Marlin kernels |

---

### US5: Diffusion Model Support

| Document | Description |
|----------|-------------|
| [us5-diffusion-summary.md](./us5-diffusion-summary.md) | User story summary and key insights |
| [diffusion-scheduler.md](./diffusion-scheduler.md) | Diffusion scheduler, GPU worker, process model |
| [diffusion-pipeline.md](./diffusion-pipeline.md) | Stage-based pipeline, ComposedPipelineBase, all 7 stages |
| [teacache.md](./teacache.md) | Temporal similarity caching for step-skipping |
| [diffusion-attention.md](./diffusion-attention.md) | STA, VSA, VMoBA, Sage attention backends |
| [diffusion-distributed.md](./diffusion-distributed.md) | SP (Ulysses/Ring), CFG parallel, topologies |
| [diffusion-jit-kernels.md](./diffusion-jit-kernels.md) | CuTe DSL fused kernels for DiT blocks |
| [dllm.md](./dllm.md) | Diffusion LLM (LLaDA2) scheduling and execution |

---

### US6: Advanced Features

| Document | Description |
|----------|-------------|
| [us6-advanced-features-summary.md](./us6-advanced-features-summary.md) | User story summary |
| [speculative-decoding.md](./speculative-decoding.md) | EAGLE, Standalone, N-gram algorithms |
| [structured-output.md](./structured-output.md) | Grammar backends, constrained generation |
| [lora.md](./lora.md) | Multi-tenant LoRA serving, SGMV kernels |
| [multimodal.md](./multimodal.md) | VLM support, 40+ model processors |
| [embedding-reward.md](./embedding-reward.md) | Embedding models, reward scoring |

---

### Cross-Cutting Concerns

| Document | Description |
|----------|-------------|
| [observability.md](./observability.md) | Prometheus metrics, request tracing |
| [profiling.md](./profiling.md) | Torch profiler, DeviceTimer, distributed merging |
| [class-index.md](./class-index.md) | Key class reference index |

---

## Quick Navigation by Question

| If you want to understand... | Read... |
|------------------------------|---------|
| How a request flows from API to response | [architecture-overview.md](./architecture-overview.md) → [scheduler-event-loop.md](./scheduler-event-loop.md) |
| How memory is allocated for KV cache | [memory-pools.md](./memory-pools.md) → [radix-cache.md](./radix-cache.md) |
| How multi-GPU inference works | [tensor-parallelism.md](./tensor-parallelism.md) → [group-coordinator.md](./group-coordinator.md) |
| How to add a new model | [model-interface.md](./model-interface.md) |
| How diffusion models are served | [diffusion-scheduler.md](./diffusion-scheduler.md) → [diffusion-pipeline.md](./diffusion-pipeline.md) |
| How speculative decoding accelerates generation | [speculative-decoding.md](./speculative-decoding.md) |
| How JSON schema constraints work | [structured-output.md](./structured-output.md) |
| How multiple LoRA adapters are served | [lora.md](./lora.md) |
| How to profile performance | [profiling.md](./profiling.md) |
| What metrics are available | [observability.md](./observability.md) |
| How configuration works | [configuration.md](./configuration.md) → [server-args-catalog.md](./server-args-catalog.md) |

---

## Key Code Entry Points

| Component | Primary File |
|-----------|-------------|
| Server Launch | `python/sglang/srt/entrypoints/engine.py` |
| Scheduler | `python/sglang/srt/managers/scheduler.py` |
| Model Runner | `python/sglang/srt/model_executor/model_runner.py` |
| Radix Cache | `python/sglang/srt/mem_cache/radix_cache.py` |
| Attention Backend | `python/sglang/srt/layers/attention/` |
| Diffusion Runtime | `python/sglang/multimodal_gen/runtime/managers/scheduler.py` |
| Gateway | `sgl-model-gateway/src/main.rs` |

---

## Statistics

- **Total Analysis Documents**: 45
- **Lines of Analysis**: ~5,000+
- **Codebase Coverage**: LLM runtime, diffusion runtime, gateway, kernels, testing
- **Key Systems Analyzed**: Scheduling, memory, parallelism, quantization, diffusion, advanced features, observability
