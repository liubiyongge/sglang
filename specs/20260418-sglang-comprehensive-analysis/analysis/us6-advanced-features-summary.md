# User Story 6: Advanced Features — Summary

## Scope

This user story covers SGLang's advanced serving capabilities beyond basic LLM inference, including speculative decoding, constrained generation, adapter serving, multi-modal support, and non-generative model types.

---

## Documents Produced

| Document | Covers |
|----------|--------|
| [speculative-decoding.md](./speculative-decoding.md) | EAGLE, EAGLE3, Standalone, N-gram algorithms; V1/V2 workers; CUDA graphs |
| [structured-output.md](./structured-output.md) | GrammarManager, XGrammar/Outlines/LLGuidance backends; bitmask operations |
| [lora.md](./lora.md) | LoRAManager, memory pool, SGMV kernels, multi-tenant adapter serving |
| [multimodal.md](./multimodal.md) | BaseMultimodalProcessor, 40+ model families, ViT CUDA graphs, EVS |
| [embedding-reward.md](./embedding-reward.md) | Embedding models, reward models, disaggregated encoding |

---

## Architecture Pattern Summary

All advanced features follow a consistent integration pattern:

```
Feature Module (standalone logic)
    ↓ integrates via
Scheduler/ModelRunner hooks
    ↓ using
Shared infrastructure (KV cache, CUDA graphs, TP, etc.)
```

### Integration Points

| Feature | Scheduler Hook | ModelRunner Hook | Output Modification |
|---------|---------------|-----------------|-------------------|
| **Speculative** | `forward_batch_speculative()` | Draft + Verify forward | Multiple tokens accepted |
| **Grammar** | `grammar_queue`, pre-sampling | `fill_next_token_bitmask()` | Logit masking |
| **LoRA** | Request → adapter mapping | `prepare_lora_batch()` | Weight delta addition |
| **Multi-modal** | Media preprocessing | Vision encoder forward | Embedding fusion |
| **Embedding** | Prefill-only scheduling | Hidden state pooling | Vector output |
| **Reward** | Prefill-only scheduling | Score head forward | Scalar output |

---

## Key Architectural Decisions

### 1. Speculative Decoding: Overlap by Default

V2 workers overlap draft(N+1) with verify(N), maximizing GPU utilization. This is enabled by default (`--disable-overlap-schedule=false`).

### 2. Grammar: Async Compilation + Distributed Sync

Grammar compilation is async (thread pool) with distributed consensus (all-gather + intersection) ensuring deterministic multi-GPU behavior.

### 3. LoRA: S-LoRA Memory Pool + SGMV Kernels

Pre-allocated GPU memory pools with eviction policies allow thousands of adapters to coexist. SGMV kernels handle batched multi-adapter forward in a single kernel call.

### 4. Multi-Modal: Processor Abstraction

40+ model families are supported through a single abstract processor interface. Model-specific logic is encapsulated in per-model processor files.

### 5. Embedding/Reward: Runtime Reuse

Non-generative models reuse the full LLM serving stack (scheduler, KV management, TP) but terminate after prefill and produce vector/scalar outputs.

---

## Key Code Locations

```
python/sglang/srt/
├── speculative/                  # All speculative decoding algorithms
│   ├── spec_info.py              # Algorithm enum + factory
│   ├── eagle_worker.py           # EAGLE V1
│   ├── eagle_worker_v2.py        # EAGLE V2 (overlap)
│   ├── standalone_worker.py      # Standard spec decoding
│   └── ngram_worker.py           # N-gram speculation
├── constrained/                  # Structured output
│   ├── grammar_manager.py        # Orchestrator
│   ├── xgrammar_backend.py       # Primary backend
│   └── outlines_backend.py       # Alternative backend
├── lora/                         # LoRA serving
│   ├── lora_manager.py           # Central manager
│   ├── mem_pool.py               # GPU memory pool
│   ├── backend/triton_backend.py # Kernel backend
│   └── triton_ops/               # SGMV kernels
├── multimodal/                   # Multi-modal support
│   ├── processors/               # 40+ model processors
│   ├── mm_utils.py               # Utilities
│   └── vit_cuda_graph_runner.py  # ViT optimization
└── disaggregation/               # Embedding/distributed encoding
    ├── encode_server.py          # Dedicated encode server
    └── encode_receiver.py        # Embedding aggregation
```
