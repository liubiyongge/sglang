# Implementation Plan: Comprehensive SGLang Codebase Analysis

**Branch**: `20260418-sglang-comprehensive-analysis` | **Date**: 2026-04-18 | **Spec**: [spec.md](specs/20260418-sglang-comprehensive-analysis/spec.md)
**Input**: Feature specification from `/specs/20260418-sglang-comprehensive-analysis/spec.md`

## Summary

A comprehensive architectural analysis of SGLang - a high-performance serving framework for LLMs and diffusion models. This analysis covers the complete system from entry points to CUDA kernels, including the scheduler, memory management (RadixAttention), five parallelism strategies (TP/PP/EP/DP/PD disaggregation), extensive quantization support (FP8/FP4/INT8/INT4/AWQ/GPTQ/GGUF), diffusion model serving, speculative decoding (EAGLE), structured outputs, and LoRA adaptation.

## Technical Context

**Language/Version**: Python 3.10+ (runtime), C++17/CUDA 12.x (kernels), Triton (JIT kernels)
**Primary Dependencies**: PyTorch 2.x, FlashInfer, CUTLASS, Triton, ZeroMQ, uvloop, FastAPI/uvicorn, xgrammar
**Storage**: GPU HBM (KV cache), CPU DRAM (HiCache L2), File/NIXL/Mooncake (HiCache L3)
**Testing**: pytest (unit + integration), custom benchmark scripts
**Target Platform**: NVIDIA GPUs (GB200/H100/A100), AMD MI300x/MI355, Intel Xeon (AMX), Google TPU, Ascend NPU
**Project Type**: High-performance inference engine (library + server)
**Performance Goals**: Millions of tokens/sec throughput, sub-100ms TTFT, 400k+ GPU deployments
**Constraints**: GPU memory bound, latency-sensitive, multi-tenant serving
**Scale/Scope**: 160+ model implementations, 30+ attention backends, 35+ quantization methods, 3 parallelism dimensions composable

## Constitution Check

*GATE: Passed - This is an analysis/documentation task. No code modifications.*

No violations applicable for analysis-only work.

## Project Structure

### Documentation (this feature)

```text
specs/20260418-sglang-comprehensive-analysis/
├── plan.md              # This file
├── research.md          # Phase 0 output - comprehensive findings
├── data-model.md        # Phase 1 output - entity/component model
├── quickstart.md        # Phase 1 output - navigation guide
└── contracts/           # Phase 1 output - API contracts
    └── api-surface.md   # Public API documentation
```

### Source Code (repository root - analyzed)

```text
python/sglang/
├── __init__.py              # Public API exports
├── lang/                    # Frontend language (SGLang DSL)
│   ├── api.py              # User-facing API (gen, select, etc.)
│   ├── ir.py               # Intermediate representation
│   ├── interpreter.py      # IR interpreter
│   └── backend/            # Backend adapters (OpenAI, Anthropic, etc.)
├── srt/                     # SGLang Runtime (core engine)
│   ├── entrypoints/        # Server entry points (HTTP, gRPC, Engine)
│   ├── managers/           # Core orchestration
│   │   ├── scheduler.py   # Main scheduler loop (~2400 lines)
│   │   ├── schedule_batch.py  # Batch data structures
│   │   ├── schedule_policy.py # Scheduling policies (LPM, DFS, FCFS)
│   │   ├── tokenizer_manager.py  # Tokenization orchestration
│   │   ├── data_parallel_controller.py  # DP dispatch
│   │   └── tp_worker.py   # Tensor parallel worker
│   ├── model_executor/     # GPU execution
│   │   ├── model_runner.py # Forward pass orchestration
│   │   ├── cuda_graph_runner.py  # CUDA graph capture/replay
│   │   └── forward_batch_info.py # Batch metadata
│   ├── models/             # 160+ model implementations
│   ├── layers/             # Neural network layers
│   │   ├── attention/      # 25+ attention backends
│   │   ├── moe/            # Mixture-of-Experts layers
│   │   ├── quantization/   # 35+ quantization methods
│   │   └── linear.py      # Quantized linear layers
│   ├── mem_cache/          # Memory & cache management
│   │   ├── radix_cache.py # RadixAttention prefix cache
│   │   ├── memory_pool.py # GPU KV cache pools
│   │   └── hiradix_cache.py  # Hierarchical 3-tier cache
│   ├── distributed/        # Communication primitives
│   ├── disaggregation/     # Prefill-Decode separation
│   ├── speculative/        # EAGLE, N-gram speculative decoding
│   ├── constrained/        # Structured output (FSM/grammar)
│   ├── lora/               # LoRA adapter management
│   ├── compilation/        # torch.compile integration
│   └── configs/            # Model configurations
├── multimodal_gen/          # Diffusion model runtime
│   ├── runtime/            # Separate scheduler/worker
│   ├── configs/            # Pipeline configurations
│   └── registry.py        # Model registry
└── jit_kernel/              # JIT-compiled kernels
    ├── diffusion/          # CuTe DSL diffusion kernels
    ├── rope.py             # Rotary embeddings
    ├── norm.py             # Layer normalization
    └── flash_attention_v4.py  # Custom FA kernel

sgl-kernel/                  # Compiled CUDA/CPU kernels
├── csrc/
│   ├── attention/          # MLA, cascade, merge states
│   ├── gemm/              # Marlin, GPTQ, FP8, INT8 GEMM
│   ├── moe/               # MoE dispatch/combine kernels
│   ├── quantization/      # GGUF, quant utilities
│   ├── memory/            # KV cache store kernel
│   ├── allreduce/         # Custom all-reduce
│   ├── speculative/       # Tree verification kernels
│   └── elementwise/       # Fused element-wise ops
└── python/                 # Python bindings
```

**Structure Decision**: Monorepo with Python runtime + compiled kernel library. The `python/sglang/srt/` directory is the core runtime engine, while `sgl-kernel/` provides optimized CUDA/CPU kernel implementations compiled separately.

## Complexity Tracking

> No violations - analysis task does not introduce complexity.

N/A
