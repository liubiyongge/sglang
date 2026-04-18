# Diffusion LLM (dLLM) Support

## Overview

SGLang's **Diffusion LLM (dLLM)** module provides support for diffusion-based language models — a new paradigm where text generation is framed as iterative denoising rather than autoregressive next-token prediction. The primary supported model is **LLaDA2** (Large Language Diffusion with Mixture of Experts).

Unlike the multimodal generation runtime (which handles image/video diffusion), dLLM operates within the **LLM runtime** (`sglang.srt`) but with specialized scheduling, forward modes, and request lifecycle management.

**File**: `python/sglang/srt/dllm/`

---

## Architecture

```
Standard LLM Path:  Scheduler → ModelRunner → Prefill/Decode → TokenSampling
dLLM Path:          Scheduler + DllmMixin → DllmManager → DLLM_EXTEND forward → BlockDecoding
```

### Key Difference from Standard LLM

| Aspect | Autoregressive LLM | Diffusion LLM (dLLM) |
|--------|-------------------|---------------------|
| **Generation** | Left-to-right, 1 token/step | Parallel denoising, block-at-a-time |
| **Forward Mode** | Prefill → Decode | DLLM_EXTEND (always full sequence) |
| **State** | KV cache grows incrementally | Full sequence reprocessed each step |
| **Output** | Single next token | Block of unmasked tokens |
| **Mask Token** | N/A | Special `[MASK]` token (ID 156895 for LLaDA2) |
| **Scheduling** | Continuous batching | Block-chunked scheduling |

---

## Configuration

**File**: `srt/dllm/config.py`

```python
class DllmConfig:
    algorithm: str                # e.g., "llada2_baseline", "llada2_corrector"
    algorithm_config: dict        # YAML-loaded algorithm-specific params
    block_size: int               # tokens per denoising block (default: 32)
    mask_id: int                  # [MASK] token ID (156895 for LLaDA2)
    max_running_requests: int     # max concurrent dLLM requests
```

**Model Detection**:
```python
if model_config.hf_config.architectures[0] == "LLaDA2MoeModelLM":
    block_size = 32
    mask_id = 156895
else:
    raise RuntimeError(f"Unknown diffusion LLM: ...")
```

---

## Scheduler Integration

**File**: `srt/dllm/mixin/scheduler.py`

The `SchedulerDllmMixin` extends the standard LLM scheduler with dLLM-specific batch creation:

### `DllmManager`

Manages the dLLM request lifecycle with two queues:

```python
class DllmManager:
    waiting_queue: List[Req]     # Requests waiting for resources
    staging_queue: List[Req]     # Requests with allocated KV resources
    
    def get_prefill_requests(self) -> List[Req]:
        """Requests needing initial full-sequence encoding."""
        return [req for req in self.waiting_queue if req.is_dllm_prefill()]
    
    def get_decode_requests(self) -> List[Req]:
        """Requests in iterative denoising phase."""
        return [req for req in self.waiting_queue if not req.is_dllm_prefill()]
```

### Request Phases

**File**: `srt/dllm/mixin/req.py`

```python
class DllmReqPhase(Enum):
    INCOMING_PREFILL   # New request, needs first forward pass
    STAGING_PREFILL    # Has KV resources allocated for prefill
    INCOMING_DECODE    # In denoising loop, waiting for next step
    STAGING_DECODE     # Has resources for next denoising step
```

### Scheduling Flow

```python
def get_new_batch_dllm(self) -> Optional[ScheduleBatch]:
    # 1. Transfer requests from scheduler waiting_queue → dllm_manager
    self._fetch_waiting_reqs()
    
    # 2. Init DLLM manager for next round
    self.dllm_manager.init_next_round()
    
    # 3. Separate prefill vs decode requests
    prefill_reqs = self.dllm_manager.get_prefill_requests()
    if prefill_reqs:
        # Process staging + incoming prefills
        self._process_batch_by_phase(adder, prefill_reqs, ...)
    else:
        # Process decode requests
        decode_reqs = self.dllm_manager.get_decode_requests()
        self._process_batch_by_phase(adder, decode_reqs, ...)
    
    # 4. Create batch with DLLM_EXTEND forward mode
    new_batch = self._create_dllm_batch(can_run_list, ForwardMode.DLLM_EXTEND)
    return new_batch
```

### Resource Management

The `_create_dllm_prefill_adder` creates a `PrefillAdder` that respects dLLM constraints:
- `dllm_config=self.dllm_config` passed to adder
- Block-aligned token allocation
- Max running requests limit (default: 1, since full-sequence reprocessing is expensive)

---

## Forward Mode

The dLLM forward mode (`ForwardMode.DLLM_EXTEND`) processes the **entire sequence** each step:

```python
# In model_executor/forward_batch_info.py
class ForwardMode(IntEnum):
    PREFILL = 0
    EXTEND = 1
    DECODE = 2
    DLLM_EXTEND = 3  # Always full-sequence, no KV cache optimization
```

Unlike standard LLM decode (which processes 1 new token using KV cache), dLLM re-processes the full sequence because:
1. Masked tokens change each step → attention patterns change
2. The model predicts probabilities for ALL masked positions simultaneously
3. No causal masking — bidirectional attention

---

## Generation Algorithm

### LLaDA2 Baseline Algorithm

```
Input:  prompt tokens [t₁, t₂, ..., tₙ] + [MASK, MASK, ..., MASK] (output slots)
                                              ↑ block_size tokens

Step 1: Forward pass → logits for all positions
Step 2: For masked positions, sample based on confidence
Step 3: Unmask highest-confidence positions (progressive unmasking)
Step 4: If all unmasked → done, else repeat from Step 1
```

### Block-Based Generation

The `block_size` parameter (default 32) determines how many output tokens are generated per generation "chunk":

```
Sequence: [prompt | generated_block_1 | generated_block_2 | ... | current_block]
                                                              ↑ all [MASK]

Each block goes through multiple denoising iterations until all positions are unmasked.
```

---

## Integration with Standard Scheduler

The dLLM mixin coexists with the standard scheduler:

```python
class Scheduler:
    # Standard LLM scheduling
    def get_new_batch_prefill(self): ...
    def get_new_batch_decode(self): ...
    
    # dLLM scheduling (from SchedulerDllmMixin)
    def init_diffusion_llm(self): ...
    def get_new_batch_dllm(self): ...
```

When `--dllm-algorithm` is specified at server launch:
1. `init_diffusion_llm()` creates the `DllmManager` 
2. The scheduler routes to `get_new_batch_dllm()` instead of standard batching
3. `ForwardMode.DLLM_EXTEND` tells the model runner to use full-sequence attention

---

## Configuration at Launch

```bash
python -m sglang.launch_server \
    --model-path meta-llama/LLaDA2-8B-MoE \
    --dllm-algorithm llada2_baseline \
    --dllm-algorithm-config config.yaml \
    --max-running-requests 1
```

YAML config example:
```yaml
block_size: 32
num_denoising_steps: 64
temperature: 0.7
top_p: 0.9
```

---

## Source References

- Config: `python/sglang/srt/dllm/config.py`
- Scheduler Mixin: `python/sglang/srt/dllm/mixin/scheduler.py`
- Request Phase: `python/sglang/srt/dllm/mixin/req.py`
- Forward Mode: `python/sglang/srt/model_executor/forward_batch_info.py`
- Algorithm implementations: `python/sglang/srt/dllm/algorithms/`
