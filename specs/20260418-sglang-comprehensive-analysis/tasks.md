# Tasks: Comprehensive SGLang Codebase Analysis

**Input**: Design documents from `/specs/20260418-sglang-comprehensive-analysis/`
**Prerequisites**: plan.md (required), spec.md (required), research.md, data-model.md, contracts/, quickstart.md

**Tests**: Not applicable - this is an analysis/documentation project.

**Organization**: Tasks are grouped by analysis domain (mapped to user stories from spec.md objectives).

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which analysis domain this task belongs to (US1-US6)
- Include exact file paths in descriptions

## Path Conventions

- **Analysis outputs**: `specs/20260418-sglang-comprehensive-analysis/analysis/`
- **Source being analyzed**: `python/sglang/` and `sgl-kernel/`

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Create output structure and establish analysis methodology

- [x] T001 Create analysis output directory structure at specs/20260418-sglang-comprehensive-analysis/analysis/
- [x] T002 [P] Generate high-level architecture diagram documenting process model in specs/20260418-sglang-comprehensive-analysis/analysis/architecture-overview.md
- [x] T003 [P] Create cross-reference index mapping key classes to source files in specs/20260418-sglang-comprehensive-analysis/analysis/class-index.md

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core understanding that ALL domain analyses depend on

- [x] T004 Analyze the Engine startup and process spawning flow in python/sglang/srt/entrypoints/engine.py, documenting the multi-process architecture in specs/20260418-sglang-comprehensive-analysis/analysis/engine-startup.md
- [x] T005 [P] Analyze the ZMQ communication topology between TokenizerManager, Scheduler, and DetokenizerManager in python/sglang/srt/managers/, documenting message formats in specs/20260418-sglang-comprehensive-analysis/analysis/ipc-topology.md
- [x] T006 [P] Analyze ServerArgs configuration surface (~246K lines) in python/sglang/srt/server_args.py, documenting all configuration categories in specs/20260418-sglang-comprehensive-analysis/analysis/server-args-catalog.md
- [x] T007 Analyze the model registry and unified model interface pattern in python/sglang/srt/models/registry.py, documenting how 160+ models integrate in specs/20260418-sglang-comprehensive-analysis/analysis/model-interface.md

**Checkpoint**: Core architecture understood - domain-specific analyses can now proceed in parallel

---

## Phase 3: User Story 1 - Framework & Scheduling (Priority: P1)

**Goal**: Complete analysis of the scheduler, continuous batching, and request lifecycle

**Independent Test**: Can explain the full path of a request from HTTP to token output with all scheduling decisions

### Implementation for User Story 1

- [x] T008 [P] [US1] Analyze the main scheduler event loop (normal + overlap modes) in python/sglang/srt/managers/scheduler.py lines 1066-1145, documenting in specs/20260418-sglang-comprehensive-analysis/analysis/scheduler-event-loop.md
- [x] T009 [P] [US1] Analyze scheduling policies (LPM, DFS-Weight, FCFS, LOF, ROUTING_KEY) in python/sglang/srt/managers/schedule_policy.py, documenting tradeoffs in specs/20260418-sglang-comprehensive-analysis/analysis/scheduling-policies.md
- [x] T010 [US1] Analyze PrefillAdder batch formation logic including chunked prefill and token budgets in python/sglang/srt/managers/schedule_policy.py lines 372-890, documenting in specs/20260418-sglang-comprehensive-analysis/analysis/batch-formation.md
- [x] T011 [US1] Analyze decode retraction mechanism and dynamic new_token_ratio adjustment in python/sglang/srt/managers/scheduler.py lines 2164-2219, documenting in specs/20260418-sglang-comprehensive-analysis/analysis/decode-retraction.md
- [x] T012 [P] [US1] Analyze ScheduleBatch and ForwardBatchInfo data structures in python/sglang/srt/managers/schedule_batch.py and python/sglang/srt/model_executor/forward_batch_info.py, documenting in specs/20260418-sglang-comprehensive-analysis/analysis/batch-data-structures.md
- [x] T013 [US1] Analyze the model execution pipeline (ModelRunner, CudaGraphRunner) in python/sglang/srt/model_executor/, documenting GPU execution flow in specs/20260418-sglang-comprehensive-analysis/analysis/model-execution.md

**Checkpoint**: Scheduler and batching system fully documented

---

## Phase 4: User Story 2 - Memory & Caching (Priority: P2)

**Goal**: Complete analysis of RadixAttention, memory pools, and hierarchical caching

**Independent Test**: Can explain how prefix caching reduces computation and how memory pressure is handled

### Implementation for User Story 2

- [x] T014 [P] [US2] Analyze RadixCache tree structure, node splitting, and prefix matching algorithms in python/sglang/srt/mem_cache/radix_cache.py, documenting in specs/20260418-sglang-comprehensive-analysis/analysis/radix-cache.md
- [x] T015 [P] [US2] Analyze memory pool implementations (ReqToTokenPool, MHA/MLA/FP4/FP8 TokenToKVPool) in python/sglang/srt/mem_cache/memory_pool.py, documenting in specs/20260418-sglang-comprehensive-analysis/analysis/memory-pools.md
- [x] T016 [US2] Analyze TokenToKVPoolAllocator paged allocation with Triton kernels in python/sglang/srt/mem_cache/allocator.py, documenting in specs/20260418-sglang-comprehensive-analysis/analysis/memory-pools.md (merged with T015)
- [x] T017 [US2] Analyze HiRadixCache 3-tier hierarchy (GPU/CPU/Storage) including write policies and async load-back in python/sglang/srt/mem_cache/hiradix_cache.py, documenting in specs/20260418-sglang-comprehensive-analysis/analysis/hicache.md
- [x] T018 [P] [US2] Analyze eviction policies (LRU/LFU/FIFO/Priority) and lock reference counting in python/sglang/srt/mem_cache/evict_policy.py, documenting in specs/20260418-sglang-comprehensive-analysis/analysis/hicache.md (merged with T017)
- [x] T019 [US2] Analyze KV cache store kernel in sgl-kernel/csrc/memory/store.cu, documenting GPU memory operations in specs/20260418-sglang-comprehensive-analysis/analysis/kv-store-kernel.md

**Checkpoint**: Memory management system fully documented

---

## Phase 5: User Story 3 - Parallelism & Distribution (Priority: P3)

**Goal**: Complete analysis of all 5 parallelism dimensions and their composition

**Independent Test**: Can explain how TP+PP+EP+DP compose for a 96-GPU DeepSeek deployment

### Implementation for User Story 3

- [x] T020 [P] [US3] Analyze GroupCoordinator and communication backend hierarchy (PyNCCL, Custom AR, PyMsccl++, Symmetric Memory) in python/sglang/srt/distributed/parallel_state.py, documenting in specs/20260418-sglang-comprehensive-analysis/analysis/group-coordinator.md
- [x] T021 [P] [US3] Analyze Tensor Parallelism implementation including TP worker coordination in python/sglang/srt/managers/tp_worker.py, documenting in specs/20260418-sglang-comprehensive-analysis/analysis/tensor-parallelism.md
- [x] T022 [P] [US3] Analyze Pipeline Parallelism microbatch scheduling in python/sglang/srt/managers/scheduler_pp_mixin.py, documenting in specs/20260418-sglang-comprehensive-analysis/analysis/pipeline-parallelism.md
- [x] T023 [US3] Analyze Expert Parallelism for MoE models (DeepEP dispatch, EPLB load balancing) in python/sglang/srt/layers/moe/ep_moe/ and python/sglang/srt/eplb/, documenting in specs/20260418-sglang-comprehensive-analysis/analysis/expert-parallelism.md
- [x] T024 [US3] Analyze DP-Attention (MAX_LEN vs SUM_LEN modes) in python/sglang/srt/layers/dp_attention.py, documenting in specs/20260418-sglang-comprehensive-analysis/analysis/dp-attention.md
- [x] T025 [US3] Analyze Prefill-Decode Disaggregation architecture (bootstrap, KV transfer backends) in python/sglang/srt/disaggregation/, documenting in specs/20260418-sglang-comprehensive-analysis/analysis/pd-disaggregation.md
- [x] T026 [US3] Analyze custom all-reduce kernel in sgl-kernel/csrc/allreduce/, documenting in specs/20260418-sglang-comprehensive-analysis/analysis/custom-allreduce-kernel.md

**Checkpoint**: All parallelism strategies documented with composition model

---

## Phase 6: User Story 4 - Quantization (Priority: P4)

**Goal**: Complete analysis of the quantization system covering all 35+ methods

**Independent Test**: Can trace the full path from model checkpoint loading through quantized inference kernel dispatch

### Implementation for User Story 4

- [x] T027 [P] [US4] Analyze QuantizationConfig/QuantizeMethodBase plugin architecture in python/sglang/srt/layers/quantization/base_config.py, documenting in specs/20260418-sglang-comprehensive-analysis/analysis/quant-architecture.md
- [x] T028 [P] [US4] Analyze FP8 quantization (per-tensor, per-channel, block-wise) including kernel dispatch logic in python/sglang/srt/layers/quantization/fp8.py and fp8_kernel.py, documenting in specs/20260418-sglang-comprehensive-analysis/analysis/fp8-quantization.md
- [x] T029 [P] [US4] Analyze INT4 quantization methods (AWQ, GPTQ, Marlin acceleration) in python/sglang/srt/layers/quantization/awq.py and gptq.py, documenting in specs/20260418-sglang-comprehensive-analysis/analysis/int4-quantization.md
- [x] T030 [US4] Analyze KV cache quantization (FP8/FP4 KV) including per-layer scaling in python/sglang/srt/layers/quantization/kv_cache.py, documenting in specs/20260418-sglang-comprehensive-analysis/analysis/kv-cache-quant.md
- [x] T031 [US4] Analyze MXFP4/NvFP4 block-scaled formats in python/sglang/srt/layers/quantization/mxfp4.py and fp4_utils.py, documenting in specs/20260418-sglang-comprehensive-analysis/analysis/fp4-quantization.md
- [x] T032 [US4] Analyze quantized linear layer integration and weight loading in python/sglang/srt/layers/linear.py and python/sglang/srt/model_loader/, documenting in specs/20260418-sglang-comprehensive-analysis/analysis/fp4-quantization.md (merged)
- [x] T033 [P] [US4] Analyze Marlin GEMM kernels in sgl-kernel/csrc/gemm/marlin/, documenting kernel optimizations in specs/20260418-sglang-comprehensive-analysis/analysis/fp4-quantization.md (merged)
- [x] T034 [P] [US4] Analyze FP8 blockwise GEMM kernel in sgl-kernel/csrc/gemm/fp8_blockwise_gemm_kernel.cu, documenting in specs/20260418-sglang-comprehensive-analysis/analysis/fp4-quantization.md (merged)

**Checkpoint**: Full quantization pipeline documented from config detection to kernel execution

---

## Phase 7: User Story 5 - Diffusion Model Support (Priority: P5)

**Goal**: Complete analysis of the diffusion runtime architecture and how it differs from LLM serving

**Independent Test**: Can explain how a WAN video generation request flows through the diffusion pipeline

### Implementation for User Story 5

- [ ] T035 [P] [US5] Analyze diffusion runtime scheduler and GPU worker in python/sglang/multimodal_gen/runtime/managers/, documenting in specs/20260418-sglang-comprehensive-analysis/analysis/diffusion-scheduler.md
- [ ] T036 [P] [US5] Analyze stage-based pipeline architecture (text encoding, conditioning, denoising, VAE decode) in python/sglang/multimodal_gen/runtime/pipelines_core/, documenting in specs/20260418-sglang-comprehensive-analysis/analysis/diffusion-pipeline.md
- [ ] T037 [US5] Analyze TeaCache temporal similarity caching in python/sglang/multimodal_gen/runtime/cache/teacache.py, documenting optimization strategy in specs/20260418-sglang-comprehensive-analysis/analysis/teacache.md
- [ ] T038 [US5] Analyze diffusion attention backends (FlashAttn, Sparse Video, VMoBA) in python/sglang/multimodal_gen/runtime/layers/attention/, documenting in specs/20260418-sglang-comprehensive-analysis/analysis/diffusion-attention.md
- [ ] T039 [P] [US5] Analyze CuTe DSL JIT kernels for diffusion in python/sglang/jit_kernel/diffusion/, documenting in specs/20260418-sglang-comprehensive-analysis/analysis/diffusion-jit-kernels.md
- [ ] T040 [US5] Analyze Diffusion LLM (LLaDA2/DLLM) integration in python/sglang/srt/dllm/, documenting hybrid text-diffusion approach in specs/20260418-sglang-comprehensive-analysis/analysis/diffusion-llm.md
- [ ] T041 [US5] Analyze diffusion distributed support (Ulysses/Ring SP, CFG parallel) in python/sglang/multimodal_gen/runtime/distributed/, documenting in specs/20260418-sglang-comprehensive-analysis/analysis/diffusion-distributed.md

**Checkpoint**: Diffusion runtime fully documented with LLM serving comparison

---

## Phase 8: User Story 6 - Advanced Features (Priority: P6)

**Goal**: Complete analysis of speculative decoding, structured outputs, LoRA, and compilation

**Independent Test**: Can explain EAGLE v2 overlapped draft-verify pipeline and XGrammar constrained generation

### Implementation for User Story 6

- [ ] T042 [P] [US6] Analyze EAGLE speculative decoding (v1/v2/multi-layer, tree verification) in python/sglang/srt/speculative/, documenting in specs/20260418-sglang-comprehensive-analysis/analysis/eagle-speculative.md
- [ ] T043 [P] [US6] Analyze N-gram speculative decoding with C++ cache in python/sglang/srt/speculative/ngram_worker.py and cpp_ngram/, documenting in specs/20260418-sglang-comprehensive-analysis/analysis/ngram-speculative.md
- [ ] T044 [P] [US6] Analyze structured output generation (XGrammar, Outlines, LLGuidance) in python/sglang/srt/constrained/, documenting in specs/20260418-sglang-comprehensive-analysis/analysis/structured-outputs.md
- [ ] T045 [US6] Analyze LoRA multi-tenant adapter management with Triton kernels in python/sglang/srt/lora/, documenting in specs/20260418-sglang-comprehensive-analysis/analysis/lora-management.md
- [ ] T046 [US6] Analyze torch.compile integration and piecewise CUDA graph compilation in python/sglang/srt/compilation/, documenting in specs/20260418-sglang-comprehensive-analysis/analysis/compilation.md
- [ ] T047 [P] [US6] Analyze batch overlap optimization (TBO, single-batch overlap for MoE) in python/sglang/srt/batch_overlap/, documenting in specs/20260418-sglang-comprehensive-analysis/analysis/batch-overlap.md
- [ ] T048 [P] [US6] Analyze attention backend registry and 25+ backend implementations in python/sglang/srt/layers/attention/, documenting in specs/20260418-sglang-comprehensive-analysis/analysis/attention-backends.md

**Checkpoint**: All advanced features documented

---

## Phase 9: Polish & Cross-Cutting Concerns

**Purpose**: Synthesize findings across all domains into unified reference documents

- [ ] T049 [P] Create unified architecture diagram showing all component interactions in specs/20260418-sglang-comprehensive-analysis/analysis/unified-architecture.md
- [ ] T050 [P] Create performance optimization catalog (all optimization techniques across subsystems) in specs/20260418-sglang-comprehensive-analysis/analysis/optimization-catalog.md
- [ ] T051 Create hardware compatibility matrix (which features work on which platforms) in specs/20260418-sglang-comprehensive-analysis/analysis/hardware-matrix.md
- [ ] T052 [P] Create data flow diagrams for key scenarios (single request, multi-turn, MoE, diffusion) in specs/20260418-sglang-comprehensive-analysis/analysis/data-flow-scenarios.md
- [ ] T053 Validate all analysis documents against source code for accuracy, update quickstart.md with any new findings in specs/20260418-sglang-comprehensive-analysis/quickstart.md
- [ ] T054 Create executive summary synthesizing all findings in specs/20260418-sglang-comprehensive-analysis/analysis/executive-summary.md

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational (Phase 2)**: Depends on Setup completion - BLOCKS all user stories
- **User Stories (Phase 3-8)**: All depend on Foundational phase completion
  - US1 (Framework/Scheduling) and US2 (Memory/Caching) can proceed in parallel
  - US3 (Parallelism) can proceed in parallel with US1/US2
  - US4 (Quantization) can proceed in parallel with all above
  - US5 (Diffusion) can proceed in parallel with all above
  - US6 (Advanced Features) can proceed in parallel with all above
- **Polish (Phase 9)**: Depends on all user stories being complete

### User Story Dependencies

- **US1 (Framework/Scheduling)**: Can start after Foundational - No other story dependencies
- **US2 (Memory/Caching)**: Can start after Foundational - Complements US1 but independent
- **US3 (Parallelism)**: Can start after Foundational - References scheduler concepts from US1
- **US4 (Quantization)**: Can start after Foundational - Independent of other stories
- **US5 (Diffusion)**: Can start after Foundational - Completely independent runtime
- **US6 (Advanced Features)**: Can start after Foundational - May reference US1 scheduler integration

### Within Each User Story

- Tasks marked [P] within a story can run in parallel
- Kernel analysis tasks can run parallel to Python code analysis
- Synthesis/integration tasks run after all parallel tasks complete

### Parallel Opportunities

- All [P] tasks within Setup (Phase 1) can run in parallel
- All [P] tasks within Foundational (Phase 2) can run in parallel
- Once Foundational completes: ALL 6 user stories can start in parallel
- Within each story: All [P] tasks can execute simultaneously
- Maximum parallelism: 6 stories x 2-4 parallel tasks = 12-24 concurrent analysis tasks

---

## Parallel Example: User Story 4 (Quantization)

```bash
# Launch parallel analysis tasks:
Task: "Analyze QuantizationConfig plugin architecture in base_config.py"
Task: "Analyze FP8 quantization including kernel dispatch in fp8.py"
Task: "Analyze INT4 quantization (AWQ, GPTQ, Marlin) in awq.py/gptq.py"
Task: "Analyze Marlin GEMM kernels in sgl-kernel/csrc/gemm/marlin/"
Task: "Analyze FP8 blockwise GEMM kernel in sgl-kernel/csrc/gemm/"

# Then sequential (depends on above):
Task: "Analyze KV cache quantization per-layer scaling in kv_cache.py"
Task: "Analyze quantized linear layer integration in linear.py"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup (output structure)
2. Complete Phase 2: Foundational (core architecture understanding)
3. Complete Phase 3: US1 - Framework & Scheduling
4. **STOP and VALIDATE**: Verify scheduler analysis covers full request lifecycle
5. Sufficient for understanding the "heart" of SGLang

### Incremental Delivery

1. Complete Setup + Foundational -> Architecture overview ready
2. Add US1 (Scheduling) -> Core runtime understood
3. Add US2 (Memory) -> Performance story complete
4. Add US3 (Parallelism) -> Scale story complete
5. Add US4 (Quantization) -> Efficiency story complete
6. Add US5 (Diffusion) -> Multi-modal story complete
7. Add US6 (Advanced) -> Full feature set documented
8. Polish -> Unified reference complete

### Parallel Team Strategy

With multiple analysts:
1. Team completes Setup + Foundational together
2. Once Foundational is done:
   - Analyst A: US1 (Framework) + US2 (Memory)
   - Analyst B: US3 (Parallelism) + US4 (Quantization)
   - Analyst C: US5 (Diffusion) + US6 (Advanced Features)
3. All analysts contribute to Polish phase

---

## Notes

- [P] tasks = independent analysis targets, no cross-dependencies
- [Story] label maps task to specific analysis domain
- Each user story produces independently useful documentation
- Source code references must include exact file paths and line ranges where applicable
- Diagrams should use ASCII/mermaid for portability
- All analysis should cross-reference the existing research.md, data-model.md, and quickstart.md
