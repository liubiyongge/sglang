# Quickstart: Navigating the SGLang Codebase

**Date**: 2026-04-18 | **Branch**: `20260418-sglang-comprehensive-analysis`

---

## Architecture at a Glance

```
SGLang = Frontend DSL + Runtime Engine + CUDA Kernels + Diffusion Runtime
         (lang/)        (srt/)           (sgl-kernel/)   (multimodal_gen/)
```

---

## Entry Points (Start Here)

| Goal | Start At |
|------|----------|
| Understand the server startup | `python/sglang/srt/entrypoints/engine.py` |
| Understand HTTP API | `python/sglang/srt/entrypoints/http_server.py` |
| Understand request flow | `python/sglang/srt/managers/tokenizer_manager.py` → `scheduler.py` |
| Understand GPU execution | `python/sglang/srt/model_executor/model_runner.py` |
| Add a new model | `python/sglang/srt/models/registry.py` → copy `llama.py` template |
| Understand memory/caching | `python/sglang/srt/mem_cache/radix_cache.py` |
| Understand parallelism | `python/sglang/srt/distributed/parallel_state.py` |
| Understand quantization | `python/sglang/srt/layers/quantization/__init__.py` |
| Understand attention | `python/sglang/srt/layers/attention/attention_registry.py` |
| Understand diffusion | `python/sglang/multimodal_gen/runtime/managers/` |
| Understand frontend DSL | `python/sglang/lang/api.py` → `ir.py` → `interpreter.py` |

---

## Request Lifecycle (LLM)

```
1. HTTP Request arrives
   └── python/sglang/srt/entrypoints/http_server.py

2. Tokenization + Template Application
   └── python/sglang/srt/managers/tokenizer_manager.py

3. Scheduling (via ZMQ)
   └── python/sglang/srt/managers/scheduler.py
       ├── schedule_policy.py (LPM/DFS/FCFS policy)
       ├── RadixCache.match_prefix() (cache lookup)
       └── PrefillAdder (batch formation)

4. GPU Forward Pass
   └── python/sglang/srt/managers/tp_worker.py
       └── python/sglang/srt/model_executor/model_runner.py
           ├── cuda_graph_runner.py (graph replay)
           └── forward_batch_info.py (metadata)

5. Token Sampling
   └── python/sglang/srt/layers/sampler.py

6. Detokenization + Streaming
   └── python/sglang/srt/managers/detokenizer_manager.py
```

---

## Key Subsystem Deep Dives

### Scheduler (The Heart)
**File**: `python/sglang/srt/managers/scheduler.py` (~2400 lines)

Key methods:
- `event_loop_overlap()`: Main loop with GPU/CPU overlap (line ~1066)
- `get_next_batch_to_run()`: Select prefill or decode batch (line ~1824)
- `get_new_batch_prefill()`: Build prefill batch from waiting queue (line ~1912)
- `run_batch()`: Execute forward pass (line ~2230)
- `process_batch_result()`: Handle outputs (line ~2399)

### Memory Management
**Files**:
- `python/sglang/srt/mem_cache/radix_cache.py`: Prefix tree
- `python/sglang/srt/mem_cache/memory_pool.py`: GPU KV pools
- `python/sglang/srt/mem_cache/hiradix_cache.py`: 3-tier hierarchy

Key concepts:
- `TreeNode.lock_ref`: Prevents eviction of in-use cache
- `TokenToKVPoolAllocator`: Page-aligned GPU memory allocation
- `evict()`: Frees least-valuable cache entries under memory pressure

### Parallelism
**Files**:
- `python/sglang/srt/distributed/parallel_state.py`: GroupCoordinator (2000+ lines)
- `python/sglang/srt/layers/dp_attention.py`: DP attention implementation
- `python/sglang/srt/layers/moe/ep_moe/layer.py`: Expert parallelism
- `python/sglang/srt/disaggregation/`: PD disaggregation

### Quantization
**File**: `python/sglang/srt/layers/quantization/`

Pattern:
```python
# Each quantization method follows:
class XxxConfig(QuantizationConfig):        # Config + factory
class XxxLinearMethod(QuantizeMethodBase):   # Forward implementation
    def create_weights(...)                  # Weight initialization
    def apply(layer, x, bias)               # Quantized matmul
    def process_weights_after_loading(...)   # Post-load transforms
```

### Model Implementation
**File**: `python/sglang/srt/models/` (160+ files)

Pattern (using llama.py as template):
```python
class LlamaForCausalLM(nn.Module):
    def __init__(self, config, quant_config, cache_config):
        # Build layers with quantization support
        
    def forward(self, input_ids, positions, forward_batch):
        # Standard forward: embed → layers → norm → lm_head
        
    def load_weights(self, weights):
        # Custom weight loading (handles sharding, quantization)
```

---

## Common Development Workflows

### Adding a New Model
1. Create `python/sglang/srt/models/my_model.py`
2. Register in `python/sglang/srt/models/registry.py`
3. Implement `forward()` and `load_weights()`
4. Add config in `python/sglang/srt/configs/` if needed
5. Test: `python -m sglang.launch_server --model-path my-model`

### Adding a New Quantization Method
1. Create `python/sglang/srt/layers/quantization/my_quant.py`
2. Implement `QuantizationConfig` + `QuantizeMethodBase`
3. Register in `python/sglang/srt/layers/quantization/__init__.py`
4. Add kernel in `sgl-kernel/csrc/gemm/` if needed

### Adding a New Attention Backend
1. Create `python/sglang/srt/layers/attention/my_backend.py`
2. Implement `AttentionBackend` interface
3. Register in `attention_registry.py`
4. Handle both `init_forward_metadata()` and `forward()`

---

## Performance-Critical Paths

| Component | Hot Path | Optimization |
|-----------|----------|-------------|
| Scheduler | `get_next_batch_to_run()` | Zero-overhead overlap |
| KV Cache | `alloc_extend()/free()` | Triton kernel allocation |
| Attention | `forward()` per layer | FlashInfer/CUTLASS |
| Sampling | `sample()` | GPU kernels |
| Cache lookup | `match_prefix()` | Radix tree O(n) |
| AllReduce | TP communication | Custom kernel < 100KB |

---

## Testing Strategy

```bash
# Unit tests
pytest test/unit/

# Integration tests (require GPU)
pytest test/srt/

# Model-specific tests
pytest test/srt/models/

# Kernel tests
cd sgl-kernel && pytest tests/

# Benchmarks
python -m sglang.bench_serving --backend sglang --model <model>
python -m sglang.bench_one_batch --model <model>
```

---

## Key Configuration Knobs

| Knob | Effect | Tune When |
|------|--------|-----------|
| `--mem-fraction-static` | GPU memory for KV cache | OOM errors |
| `--chunked-prefill-size` | Max prefill chunk | Long prompts stall |
| `--max-running-requests` | Concurrent decodes | Throughput tuning |
| `--schedule-policy` | Scheduling strategy | Cache hit rates |
| `--cuda-graph-max-bs` | Graph capture limit | Memory vs latency |
| `--new-token-ratio` | Decode memory reserve | Retraction frequency |
| `--enable-overlap-schedule` | CPU/GPU overlap | CPU-bound scenarios |

---

## Code Statistics

| Component | Files | Lines (approx) |
|-----------|-------|----------------|
| srt/managers/ | 33 | ~20,000 |
| srt/models/ | 161 | ~80,000 |
| srt/layers/ | 100+ | ~40,000 |
| srt/mem_cache/ | 21 | ~8,000 |
| srt/distributed/ | 6 | ~3,000 |
| srt/speculative/ | 20 | ~6,000 |
| sgl-kernel/csrc/ | 50+ | ~30,000 (C++/CUDA) |
| multimodal_gen/ | 100+ | ~25,000 |
| **Total** | **500+** | **~210,000** |
