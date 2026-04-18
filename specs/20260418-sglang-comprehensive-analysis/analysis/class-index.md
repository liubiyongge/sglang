# Cross-Reference Index: Key Classes to Source Files

**Date**: 2026-04-18

---

## Entry Points & Server

| Class/Function | File | Lines | Description |
|---------------|------|-------|-------------|
| `Engine` | `python/sglang/srt/entrypoints/engine.py` | 1-600+ | Top-level API, process spawning |
| `EngineBase` | `python/sglang/srt/entrypoints/EngineBase.py` | - | Abstract engine interface |
| `launch_server` | `python/sglang/launch_server.py` | - | CLI entry point |
| `HTTPServer` | `python/sglang/srt/entrypoints/http_server.py` | - | FastAPI HTTP server |
| `GRPCServer` | `python/sglang/srt/entrypoints/grpc_server.py` | - | gRPC server |
| `ServerArgs` | `python/sglang/srt/server_args.py` | 1-6500+ | All configuration parameters |

## Managers (Orchestration)

| Class/Function | File | Lines | Description |
|---------------|------|-------|-------------|
| `Scheduler` | `python/sglang/srt/managers/scheduler.py` | ~2400 | Core scheduler event loop |
| `TokenizerManager` | `python/sglang/srt/managers/tokenizer_manager.py` | - | Tokenization process |
| `DetokenizerManager` | `python/sglang/srt/managers/detokenizer_manager.py` | - | Output detokenization |
| `DataParallelController` | `python/sglang/srt/managers/data_parallel_controller.py` | - | DP request routing |
| `TpModelWorker` | `python/sglang/srt/managers/tp_worker.py` | - | Tensor parallel GPU worker |
| `ScheduleBatch` | `python/sglang/srt/managers/schedule_batch.py` | - | Batch data container |
| `SchedulePolicy` | `python/sglang/srt/managers/schedule_policy.py` | - | Scheduling algorithms |
| `PrefillAdder` | `python/sglang/srt/managers/schedule_policy.py` | 372-890 | Batch formation logic |

## Model Execution

| Class/Function | File | Lines | Description |
|---------------|------|-------|-------------|
| `ModelRunner` | `python/sglang/srt/model_executor/model_runner.py` | - | GPU forward orchestration |
| `CudaGraphRunner` | `python/sglang/srt/model_executor/cuda_graph_runner.py` | - | CUDA graph capture/replay |
| `ForwardBatchInfo` | `python/sglang/srt/model_executor/forward_batch_info.py` | - | Batch metadata for GPU |
| `PiecewiseCudaGraphRunner` | `python/sglang/srt/model_executor/piecewise_cuda_graph_runner.py` | - | Piecewise graph compilation |

## Memory & Caching

| Class/Function | File | Lines | Description |
|---------------|------|-------|-------------|
| `RadixCache` | `python/sglang/srt/mem_cache/radix_cache.py` | - | Prefix tree cache |
| `HiRadixCache` | `python/sglang/srt/mem_cache/hiradix_cache.py` | - | 3-tier hierarchical cache |
| `TreeNode` | `python/sglang/srt/mem_cache/radix_cache.py` | - | Radix tree node |
| `ReqToTokenPool` | `python/sglang/srt/mem_cache/memory_pool.py` | - | Request→token mapping |
| `MHATokenToKVPool` | `python/sglang/srt/mem_cache/memory_pool.py` | - | MHA KV cache storage |
| `MLATokenToKVPool` | `python/sglang/srt/mem_cache/memory_pool.py` | - | MLA compressed KV |
| `TokenToKVPoolAllocator` | `python/sglang/srt/mem_cache/allocator.py` | - | Paged KV allocation |
| `PagedTokenToKVPoolAllocator` | `python/sglang/srt/mem_cache/allocator.py` | - | Page-aligned allocator |

## Distributed & Parallelism

| Class/Function | File | Lines | Description |
|---------------|------|-------|-------------|
| `GroupCoordinator` | `python/sglang/srt/distributed/parallel_state.py` | ~2000 | Unified comm abstraction |
| `DPAttentionManager` | `python/sglang/srt/layers/dp_attention.py` | - | DP attention coordination |
| `EPMoELayer` | `python/sglang/srt/layers/moe/ep_moe/layer.py` | - | Expert parallel MoE |
| `EPLBManager` | `python/sglang/srt/eplb/eplb_manager.py` | - | Expert load balancing |
| `ExpertLocation` | `python/sglang/srt/eplb/expert_location.py` | - | Expert placement metadata |

## Layers & Attention

| Class/Function | File | Lines | Description |
|---------------|------|-------|-------------|
| `RadixAttention` | `python/sglang/srt/layers/radix_attention.py` | - | Main attention layer |
| `FlashInferBackend` | `python/sglang/srt/layers/attention/flashinfer_backend.py` | - | FlashInfer attention |
| `CutlassMLABackend` | `python/sglang/srt/layers/attention/cutlass_mla_backend.py` | - | CUTLASS MLA |
| `LinearBase` | `python/sglang/srt/layers/linear.py` | - | Quantization-aware linear |
| `FusedMoE` | `python/sglang/srt/layers/moe/fused_moe_triton/` | - | Triton MoE kernel |

## Quantization

| Class/Function | File | Lines | Description |
|---------------|------|-------|-------------|
| `QuantizationConfig` | `python/sglang/srt/layers/quantization/base_config.py` | - | Abstract quant config |
| `Fp8Config` | `python/sglang/srt/layers/quantization/fp8.py` | - | FP8 quantization |
| `Fp8LinearMethod` | `python/sglang/srt/layers/quantization/fp8.py` | - | FP8 forward method |
| `GPTQConfig` | `python/sglang/srt/layers/quantization/gptq.py` | - | GPTQ quantization |
| `AWQConfig` | `python/sglang/srt/layers/quantization/awq.py` | - | AWQ quantization |
| `Mxfp4Config` | `python/sglang/srt/layers/quantization/mxfp4.py` | - | MXFP4 block-scaled |
| `BaseKVCacheMethod` | `python/sglang/srt/layers/quantization/kv_cache.py` | - | KV cache quant |

## Speculative Decoding

| Class/Function | File | Lines | Description |
|---------------|------|-------|-------------|
| `EAGLEWorker` | `python/sglang/srt/speculative/eagle_worker.py` | - | EAGLE v1 worker |
| `EAGLEWorkerV2` | `python/sglang/srt/speculative/eagle_worker_v2.py` | - | EAGLE v2 overlapped |
| `MultiLayerEagleWorker` | `python/sglang/srt/speculative/multi_layer_eagle_worker.py` | - | Multi-layer draft |
| `NGRAMWorker` | `python/sglang/srt/speculative/ngram_worker.py` | - | N-gram speculation |
| `EAGLEDraftCudaGraphRunner` | `python/sglang/srt/speculative/eagle_draft_cuda_graph_runner.py` | - | Draft CUDA graphs |

## Structured Output

| Class/Function | File | Lines | Description |
|---------------|------|-------|-------------|
| `GrammarManager` | `python/sglang/srt/constrained/grammar_manager.py` | - | Grammar lifecycle |
| `XGrammarGrammar` | `python/sglang/srt/constrained/xgrammar_backend.py` | - | XGrammar backend |
| `OutlinesGrammar` | `python/sglang/srt/constrained/outlines_backend.py` | - | Outlines backend |

## LoRA

| Class/Function | File | Lines | Description |
|---------------|------|-------|-------------|
| `LoRAManager` | `python/sglang/srt/lora/lora_manager.py` | - | Adapter lifecycle |
| `TritonLoRABackend` | `python/sglang/srt/lora/backend/triton_backend.py` | - | Triton LoRA kernels |
| `LoRABatchInfo` | `python/sglang/srt/lora/` | - | Batch routing info |

## Diffusion Runtime

| Class/Function | File | Lines | Description |
|---------------|------|-------|-------------|
| `DiffScheduler` | `python/sglang/multimodal_gen/runtime/managers/scheduler.py` | - | Diffusion scheduler |
| `GPUWorker` | `python/sglang/multimodal_gen/runtime/managers/gpu_worker.py` | - | Diffusion GPU execution |
| `Pipeline` | `python/sglang/multimodal_gen/runtime/pipelines_core/` | - | Stage-based pipeline |
| `TeaCacheContext` | `python/sglang/multimodal_gen/runtime/cache/teacache.py` | - | Temporal caching |
| `DiffGenerator` | `python/sglang/multimodal_gen/entrypoints/diffusion_generator.py` | - | Python SDK entry |

## Frontend Language

| Class/Function | File | Lines | Description |
|---------------|------|-------|-------------|
| `gen()` | `python/sglang/lang/api.py` | - | Text generation primitive |
| `select()` | `python/sglang/lang/api.py` | - | Choice primitive |
| `SglFunction` | `python/sglang/lang/ir.py` | - | IR function node |
| `StreamExecutor` | `python/sglang/lang/interpreter.py` | - | IR interpreter |
| `RuntimeEndpoint` | `python/sglang/lang/backend/runtime_endpoint.py` | - | SGLang backend |

## Model Implementations (Selected)

| Class/Function | File | Description |
|---------------|------|-------------|
| `LlamaForCausalLM` | `python/sglang/srt/models/llama.py` | Reference implementation |
| `DeepseekV2ForCausalLM` | `python/sglang/srt/models/deepseek_v2.py` | MoE model |
| `Qwen2ForCausalLM` | `python/sglang/srt/models/qwen2.py` | Qwen2 model |
| `LLaDA2MoeModelLM` | `python/sglang/srt/models/llada2.py` | Diffusion LLM |
| `ModelRegistry` | `python/sglang/srt/models/registry.py` | Model→class mapping |

## sgl-kernel (CUDA)

| Kernel | File | Description |
|--------|------|-------------|
| MLA Attention | `sgl-kernel/csrc/attention/cutlass_mla_kernel.cu` | CUTLASS MLA |
| Merge States | `sgl-kernel/csrc/attention/merge_attn_states.cu` | Attention state merge |
| Marlin GEMM | `sgl-kernel/csrc/gemm/marlin/gptq_marlin.cu` | Marlin quantized GEMM |
| FP8 Block GEMM | `sgl-kernel/csrc/gemm/fp8_blockwise_gemm_kernel.cu` | FP8 block-wise |
| KV Store | `sgl-kernel/csrc/memory/store.cu` | KV cache write |
| Custom AllReduce | `sgl-kernel/csrc/allreduce/` | Small tensor AR |
| Grammar Mask | `sgl-kernel/csrc/grammar/apply_token_bitmask_inplace_cuda.cu` | Vocab masking |
| MoE Dispatch | `sgl-kernel/csrc/moe/` | MoE token routing |
