# Speculative Decoding

## Overview

SGLang supports speculative decoding to accelerate autoregressive generation. The key idea: a small/fast "draft" model proposes multiple tokens, and the main "target" model verifies them in a single forward pass. Accepted tokens avoid sequential target model calls.

**File**: `python/sglang/srt/speculative/`

---

## Supported Algorithms

| Algorithm | Class | Draft Model | Key Idea |
|-----------|-------|-------------|----------|
| **EAGLE** | `EAGLEWorker` / `EAGLEWorkerV2` | Trained draft head | Feature-level prediction with tree-based verification |
| **EAGLE3** | Same workers (variant) | Enhanced draft | Third-gen EAGLE with improved acceptance |
| **STANDALONE** | `StandaloneWorker` / `StandaloneWorkerV2` | Separate small LLM | Standard speculative decoding with independent model |
| **NGRAM** | `NGRAMWorker` | N-gram lookup table | No neural draft; uses cached n-gram patterns |

---

## Architecture

```
Scheduler
├── get_new_batch_prefill()    # Regular prefill
├── get_new_batch_decode()     # Regular decode
│   └── if spec_enabled:
│       └── SpecWorker.forward_batch_speculative()
│           ├── Draft Phase: Generate K candidate tokens
│           ├── Verify Phase: Batch-verify all candidates
│           └── Accept Phase: Accept longest valid prefix
```

### Worker Selection (Factory Pattern)

**File**: `spec_info.py`

```python
class SpeculativeAlgorithm(Enum):
    def create_worker(self, server_args) -> Type[BaseSpecWorker]:
        if self.is_eagle() and server_args.enable_multi_layer_eagle:
            return MultiLayerEagleWorkerV2 if enable_overlap else MultiLayerEagleWorker
        elif self.is_eagle():
            return EAGLEWorkerV2 if enable_overlap else EAGLEWorker
        elif self.is_standalone():
            return StandaloneWorkerV2 if enable_overlap else StandaloneWorker
        elif self.is_ngram():
            return NGRAMWorker
```

### V1 vs V2 Workers

- **V1** (`EAGLEWorker`, `StandaloneWorker`): Sequential draft → verify
- **V2** (`EAGLEWorkerV2`, `StandaloneWorkerV2`): **Overlapped scheduling** — draft of batch N+1 overlaps with verify of batch N. Requires `--disable-overlap-schedule=false` (default enabled).

---

## EAGLE Algorithm

### Draft Phase

EAGLE uses a lightweight trained head that predicts the next token's hidden state features:

```
Target model hidden states at position t → EAGLE head → predicted features at t+1
```

The EAGLE head is much cheaper than the full target model (1-2 transformer layers).

### Tree-Based Verification

EAGLE builds a **speculation tree** (not just a chain):

```
Root token (accepted)
├── Branch A: [tok_1a, tok_2a, tok_3a]
├── Branch B: [tok_1a, tok_2b, tok_3b]  (diverges at position 2)
└── Branch C: [tok_1c, tok_2c]           (different first prediction)
```

All branches are verified in a single target model forward pass using **tree attention** masks.

### Key Components

| Component | File | Purpose |
|-----------|------|---------|
| `EagleDraftInput` | `eagle_info.py` | Input structure for draft forward |
| `EagleVerifyInput` | `eagle_info.py` | Input structure for verify forward |
| `build_tree_kernel` | `eagle_utils.py` | Constructs speculation tree |
| `EagleDraftCudaGraphRunner` | `eagle_draft_cuda_graph_runner.py` | CUDA graph capture for draft |

### Multi-Layer EAGLE

For larger models, multi-layer EAGLE uses multiple draft layers with extended context:
- `MultiLayerEagleWorker`: Multi-layer draft model
- More accurate drafts at the cost of slightly more expensive draft computation
- Net benefit for models where single-layer acceptance is low

---

## STANDALONE Algorithm

Traditional speculative decoding with a separate (smaller) LLM:

```
Draft Model (e.g., 1B params): Generate K tokens autoregressively
Target Model (e.g., 70B params): Verify all K tokens in one forward pass
Accept: Longest prefix where draft matches target sampling
```

The standalone worker handles:
- Loading a separate model instance
- Managing separate KV caches for draft and target
- Token synchronization between models

---

## N-gram Algorithm

Zero-cost speculation using cached patterns:

```python
class NGRAMWorker:
    """Uses historical n-gram patterns for speculation."""
    # Maintains a C++ n-gram cache of previously seen sequences
    # Proposes next tokens based on frequency of seen n-grams
    # No neural network inference needed for drafting
```

Best for: Repetitive generation tasks (translation, summarization with repeated patterns).

---

## SpecInput Interface

**File**: `spec_info.py`

```python
class SpecInput(ABC):
    def __init__(self, spec_input_type: SpecInputType):
        self.spec_input_type = spec_input_type
    
    @abstractmethod
    def get_spec_adjust_token_coefficient(self) -> Tuple[int, int]:
        """Returns coefficients for token count adjustment in verify mode."""
        pass
```

The `SpecInputType` distinguishes `EAGLE_DRAFT`, `EAGLE_VERIFY`, and `NGRAM_VERIFY` for attention backend routing.

---

## CUDA Graph Integration

Speculation benefits heavily from CUDA graphs since draft forward passes are small and repetitive:

- `EagleDraftCudaGraphRunner`: Captures draft computation
- `EagleDraftExtendCudaGraphRunner`: Captures extended context drafts
- `MultiLayerEagleDraftExtendCudaGraphRunner`: Multi-layer variant

These runners pre-allocate fixed-size buffers and replay captured graphs for zero-overhead draft execution.

---

## Configuration

```bash
# EAGLE speculative decoding
python -m sglang.launch_server \
    --model-path meta-llama/Llama-3-70B \
    --speculative-algorithm EAGLE \
    --speculative-draft-model-path path/to/eagle-head \
    --speculative-num-steps 5 \
    --speculative-eagle-topk 10

# Standalone speculative decoding
python -m sglang.launch_server \
    --model-path meta-llama/Llama-3-70B \
    --speculative-algorithm STANDALONE \
    --speculative-draft-model-path meta-llama/Llama-3-8B

# N-gram speculation
python -m sglang.launch_server \
    --model-path meta-llama/Llama-3-70B \
    --speculative-algorithm NGRAM \
    --speculative-num-steps 3
```

---

## Source References

- Algorithm enum + factory: `speculative/spec_info.py`
- EAGLE worker: `speculative/eagle_worker.py` (V1), `eagle_worker_v2.py` (V2)
- Multi-layer EAGLE: `speculative/multi_layer_eagle_worker.py`
- Standalone: `speculative/standalone_worker.py`
- N-gram: `speculative/ngram_worker.py`
- CUDA graphs: `speculative/eagle_draft_cuda_graph_runner.py`
- Tree utilities: `speculative/eagle_utils.py`, `spec_utils.py`
