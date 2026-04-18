# User Story 5: Diffusion Model Support — Summary

## Scope

This user story covers SGLang's support for diffusion-based generation, spanning two sub-systems:

1. **Multimodal Generation Runtime** (`sglang.multimodal_gen`): Image/video diffusion (Flux, WAN, HunyuanVideo, etc.)
2. **Diffusion LLM (dLLM)** (`sglang.srt.dllm`): Text generation via iterative denoising (LLaDA2)

---

## Documents Produced

| Document | Covers |
|----------|--------|
| [diffusion-scheduler.md](./diffusion-scheduler.md) | Scheduler event loop, GPU Worker, process model, comparison with LLM scheduler |
| [diffusion-pipeline.md](./diffusion-pipeline.md) | Stage-based pipeline architecture, ComposedPipelineBase, all 7 stages, data flow |
| [teacache.md](./teacache.md) | Temporal similarity caching, L1 distance + polynomial rescaling, CFG-aware state |
| [diffusion-attention.md](./diffusion-attention.md) | STA, VSA, VMoBA, Sage, Flash backends; SP integration |
| [diffusion-distributed.md](./diffusion-distributed.md) | SP (Ulysses/Ring/Hybrid), CFG Parallel, TP, GPU topologies |
| [diffusion-jit-kernels.md](./diffusion-jit-kernels.md) | CuTe DSL fused kernels, ScaleResidualNormScaleShift, compile cache |
| [dllm.md](./dllm.md) | Diffusion LLM scheduling, DllmManager, LLaDA2 support |

---

## Key Architectural Insights

### The Two Diffusion Systems

| Aspect | Multimodal Gen (Image/Video) | dLLM (Text) |
|--------|------------------------------|-------------|
| **Runtime** | Separate process (`multimodal_gen/runtime/`) | Integrated into LLM runtime (`srt/`) |
| **Scheduling** | FIFO, single-request | Mixin on standard scheduler |
| **Parallelism** | SP + CFG + TP | Standard TP/PP |
| **Forward** | Pipeline stages (text_encode → denoise → decode) | DLLM_EXTEND full-sequence |
| **KV Cache** | None | Allocated but fully rewritten each step |
| **Models** | Flux, WAN, Hunyuan, Qwen, GLM, LTX, MoVA | LLaDA2 |

### Performance Optimization Stack

```
Level 0: Fused CuTe DSL Kernels (norm/scale/shift elimination)
Level 1: torch.compile (graph-level fusion)
Level 2: Attention Backends (STA/VSA/VMoBA sparse patterns)
Level 3: TeaCache (step-level computation skipping)
Level 4: Cache-DiT (block-level caching within steps)
Level 5: Sequence Parallelism (memory distribution across GPUs)
Level 6: CFG Parallelism (guidance computation distribution)
Level 7: LayerWise Offload (model memory management)
```

### Stage-Based Pipeline Design

The diffusion runtime's key architectural choice is composable, verifiable stages:
- Each stage has `verify_input` / `forward` / `verify_output`
- Stages declare their parallelism type (`REPLICATED`, `MAIN_RANK_ONLY`, `CFG_PARALLEL`)
- The executor orchestrates communication based on stage declarations
- New models are added by creating a new pipeline config + stage composition

---

## Key Code Locations

```
python/sglang/multimodal_gen/runtime/
├── managers/scheduler.py          # Event loop, ZMQ, request handling
├── managers/gpu_worker.py         # Device init, pipeline execution
├── pipelines_core/
│   ├── composed_pipeline_base.py  # Pipeline ABC
│   ├── stages/                    # All 7 pipeline stages
│   └── executors/                 # ParallelExecutor
├── cache/teacache.py              # Temporal similarity caching
├── layers/attention/              # Backend selector + implementations
├── distributed/                   # SP, CFG, TP groups
└── pipelines/                     # Concrete pipelines (WAN, Flux, etc.)

python/sglang/srt/dllm/
├── config.py                      # DllmConfig
├── mixin/scheduler.py             # SchedulerDllmMixin, DllmManager
└── mixin/req.py                   # DllmReqPhase enum

python/sglang/jit_kernel/diffusion/cutedsl/
└── scale_residual_norm_scale_shift.py  # Fused AdaLN kernels
```
