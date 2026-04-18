# TeaCache: Temporal Similarity Caching for Diffusion Models

## Overview

TeaCache accelerates diffusion inference by selectively **skipping redundant transformer computation** when consecutive denoising steps produce similar outputs. It tracks the L1 distance between modulated inputs across timesteps and uses cached residuals when the accumulated distance is below a threshold.

**Reference Paper**: [TeaCache: Accelerating Diffusion Models with Temporal Similarity](https://arxiv.org/abs/2411.14324)

**File**: `python/sglang/multimodal_gen/runtime/cache/teacache.py`

---

## Key Concepts

| Concept | Description |
|---------|-------------|
| **Modulated Input** | The DiT input after timestep conditioning (e.g., after adaptive norm) |
| **L1 Distance** | `abs(current - previous).mean() / previous.abs().mean()` — relative change |
| **Polynomial Rescaling** | Model-specific calibration: `np.poly1d(coefficients)(l1_distance)` |
| **Threshold** | When accumulated rescaled L1 exceeds threshold → force computation |
| **Residual Caching** | `output - input` is cached; on cache hit, add cached residual to current input |
| **CFG Separation** | Positive and negative CFG branches maintain independent caches |

---

## Architecture

### `TeaCacheMixin`

A mixin class that DiT models inherit to gain TeaCache optimization:

```python
class WanDiT(TeaCacheMixin, BaseDiT):
    def __init__(self, config):
        super().__init__(config)
        self._init_teacache_state()
    
    def forward(self, hidden_states, timestep, ...):
        ctx = self._get_teacache_context()
        if ctx is not None:
            modulated_input = self._compute_modulated_input(hidden_states, timestep)
            is_boundary = (ctx.current_timestep == 0 or 
                          ctx.current_timestep >= ctx.num_inference_steps - 1)
            
            should_calc = self._compute_teacache_decision(
                modulated_inp=modulated_input,
                is_boundary_step=is_boundary,
                coefficients=ctx.coefficients,
                teacache_thresh=ctx.teacache_thresh,
            )
            
            if not should_calc:
                return self.retrieve_cached_states(hidden_states)
        
        output = self._transformer_forward(hidden_states, timestep, ...)
        
        if ctx is not None:
            self.maybe_cache_states(output, hidden_states)
        
        return output
```

### `TeaCacheContext`

A dataclass extracted from the forward context each step:

```python
@dataclass
class TeaCacheContext:
    current_timestep: int          # 0-indexed step in denoising loop
    num_inference_steps: int        # total steps
    do_cfg: bool                   # classifier-free guidance enabled
    is_cfg_negative: bool          # currently processing negative branch
    teacache_thresh: float         # accumulated distance threshold
    coefficients: list[float]      # polynomial rescaling coefficients
    teacache_params: TeaCacheParams # full model-specific parameters
```

---

## Decision Algorithm

### Step-by-Step Flow

```
Step i:
  1. Compute modulated_input from (hidden_states, timestep_embedding)
  2. Check boundary: if step==0 or step==last → force compute
  3. Compute relative L1:
       diff = modulated_input - previous_modulated_input
       rel_l1 = diff.abs().mean() / previous_modulated_input.abs().mean()
  4. Apply polynomial rescaling:
       rescaled = np.poly1d(coefficients)(rel_l1)
  5. Accumulate:
       accumulated += rescaled
  6. Decision:
       if accumulated >= threshold:
           → COMPUTE (reset accumulated = 0)
       else:
           → USE CACHE (keep accumulated)
  7. Update previous_modulated_input = modulated_input.clone()
```

### `_compute_l1_and_decide` Implementation

```python
def _compute_l1_and_decide(self, modulated_inp, coefficients, teacache_thresh):
    prev = self.previous_modulated_input_negative if self.is_cfg_negative 
           else self.previous_modulated_input
    
    if prev is None:
        return 0.0, True  # Force compute on first step
    
    # Relative L1 distance
    diff = modulated_inp - prev
    rel_l1 = (diff.abs().mean() / prev.abs().mean()).cpu().item()
    
    # Polynomial rescaling (model-specific calibration)
    rescale_func = np.poly1d(coefficients)
    accumulated = self.accumulated_rel_l1_distance + rescale_func(rel_l1)
    
    if accumulated >= teacache_thresh:
        return 0.0, True   # Threshold exceeded: force compute
    return accumulated, False  # Cache hit
```

---

## CFG-Aware Caching

TeaCache maintains **separate state** for positive and negative CFG branches:

| State Variable | Positive Branch | Negative Branch |
|---|---|---|
| `previous_modulated_input` | ✓ | `previous_modulated_input_negative` |
| `previous_residual` | ✓ | `previous_residual_negative` |
| `accumulated_rel_l1_distance` | ✓ | `accumulated_rel_l1_distance_negative` |

**CFG Support Matrix**:

| Model Prefix | CFG Cache Supported |
|---|---|
| `wan` | ✓ |
| `hunyuan` | ✓ |
| `zimage` | ✓ |
| `flux` | ✗ (TeaCache auto-disabled when CFG enabled) |
| `qwen` | ✗ |

The `_supports_cfg_cache` flag is determined by:
```python
self._supports_cfg_cache = self.config.prefix.lower() in {"wan", "hunyuan", "zimage"}
```

---

## State Management

### Initialization

```python
def _init_teacache_state(self):
    self.cnt = 0
    self.enable_teacache = True
    self._supports_cfg_cache = ...
    
    # Always present (positive branch)
    self.previous_modulated_input = None
    self.previous_residual = None
    self.accumulated_rel_l1_distance = 0.0
    
    # CFG-specific (only when _supports_cfg_cache)
    if self._supports_cfg_cache:
        self.previous_modulated_input_negative = None
        self.previous_residual_negative = None
        self.accumulated_rel_l1_distance_negative = 0.0
```

### Reset (per generation request)

```python
def reset_teacache_state(self):
    self.cnt = 0
    self.previous_modulated_input = None
    self.previous_residual = None
    self.accumulated_rel_l1_distance = 0.0
    # + negative branch reset if applicable
```

### Context Retrieval

The context is obtained from the `ForwardContext` which is set per-timestep by `DenoisingStage`:

```python
def _get_teacache_context(self):
    forward_context = get_forward_context()
    forward_batch = forward_context.forward_batch
    
    if not forward_batch.enable_teacache:
        return None
    
    # Reset at first timestep
    if forward_context.current_timestep == 0 and not self.is_cfg_negative:
        self.reset_teacache_state()
    
    return TeaCacheContext(...)
```

---

## Integration with Cache-DiT

TeaCache (step-level caching) can be combined with **Cache-DiT** (block-level caching within a single step):

- **TeaCache**: Skips the entire transformer forward for a timestep
- **Cache-DiT**: Caches intermediate block outputs within a single forward pass

Both are controlled through environment variables and the `DenoisingStage._maybe_enable_cache_dit()` method.

---

## Configuration

TeaCache parameters are passed per-request through `TeaCacheParams` in sampling params:

```python
# configs/sample/teacache.py
class TeaCacheParams:
    teacache_thresh: float        # e.g., 0.15
    coefficients: list[float]     # e.g., [2.14, -1.75, 0.58]
```

The polynomial coefficients are model-specific and must be calibrated to map raw L1 distances to meaningful similarity scores.

---

## Performance Impact

| Scenario | Steps Computed | Steps Cached | Speedup |
|----------|---------------|-------------|---------|
| Typical WAN T2V (50 steps) | ~20-30 | ~20-30 | ~1.5-2x |
| Low threshold (aggressive) | ~15 | ~35 | ~2-3x (quality loss) |
| High threshold (conservative) | ~40 | ~10 | ~1.2x |

The threshold trades off generation quality against speed — higher thresholds are more conservative (more computation, better quality).

---

## Source References

- TeaCacheMixin: `python/sglang/multimodal_gen/runtime/cache/teacache.py`
- Cache-DiT integration: `python/sglang/multimodal_gen/runtime/cache/cache_dit_integration.py`
- TeaCacheParams config: `python/sglang/multimodal_gen/configs/sample/teacache.py`
- DenoisingStage integration: `pipelines_core/stages/denoising.py` (lines 1058-1075)
