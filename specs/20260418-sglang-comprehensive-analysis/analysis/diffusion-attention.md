# Diffusion Attention Backends

## Overview

The diffusion runtime supports multiple attention computation strategies through a backend selector pattern. Unlike the LLM runtime (which uses PagedAttention), diffusion models use **full sequence attention** with model-aware sparsity patterns for efficient long-sequence video generation.

**File**: `python/sglang/multimodal_gen/runtime/layers/attention/`

---

## Backend Selector Architecture

**File**: `layers/attention/selector.py`

```python
class AttentionSelector:
    """Routes attention requests to the appropriate backend."""
    
    @staticmethod
    def select_backend(attn_metadata: AttnMetadata, server_args: ServerArgs) -> str:
        # Priority order:
        # 1. Explicit override from server_args
        # 2. Model-specific default from pipeline config
        # 3. Auto-detection based on GPU capabilities
        ...
```

---

## Available Backends

### 1. Flash Attention (`flash_attn`)

Standard dense attention using Flash Attention 2 (Tri Dao).

- **Use case**: Short sequences, dense attention patterns
- **Shape**: `[B, S, H, D]` → standard Q/K/V format
- **Limitations**: Quadratic in sequence length; unsuitable for long video

### 2. STA (Sliding Tile Attention)

**File**: `layers/attention/backends/sta.py`

A block-sparse attention pattern for video DiTs that exploits **spatial-temporal locality**:

```
Video frame layout:   [Temporal tiles] × [Height tiles] × [Width tiles]
                        T₁ T₂ T₃        H₁ H₂ H₃        W₁ W₂ W₃
```

Key idea: Each token attends to tokens within its **local tile** plus a set of **global tokens** — providing both local context and global coherence at sub-quadratic cost.

- **Metadata Construction**: Per-timestep tile indexing computed in `_build_sta_metadata()`
- **Configuration**: `sta_tile_size`, `sta_num_global_tokens` in server args
- **Speedup**: ~2-4x for long video sequences (256+ frames)

### 3. VSA (Variable Sparse Attention)

**File**: `layers/attention/backends/vsa.py`

A dynamic sparse attention pattern that adapts based on timestep:

- Early steps (high noise): Sparser attention (less detail needed)
- Later steps (low noise): Denser attention (fine detail refinement)
- **Metadata**: Sparsity mask built per-step in `_build_vsa_metadata()`

### 4. VMoBA (Video Mixture of Block Attention)

**File**: `layers/attention/backends/vmoba.py`

Block attention with mixture routing for video generation:

- Divides the sequence into blocks
- Each block selects which other blocks to attend to via a routing mechanism
- Efficient for very long sequences (1000+ frames)

### 5. Sage Attention

**File**: `layers/attention/backends/sage_attn.py`

Quantized attention (INT4/INT8 matmuls) with smoothing:

- Quantizes Q/K to lower precision for memory bandwidth savings
- Applies per-head smoothing factors to preserve accuracy
- Best for memory-bound scenarios

---

## Attention Layer Implementation

**File**: `layers/attention/layer.py`

The `Attention` layer wraps backend dispatch:

```python
class Attention(nn.Module):
    def __init__(self, head_dim, num_heads, num_kv_heads=None, ...):
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.scaling = head_dim ** -0.5
        
    def forward(self, q, k, v, attn_metadata=None, cu_seqlens=None, ...):
        # 1. Get forward context for current timestep
        ctx = get_forward_context()
        
        # 2. Handle sequence parallel: gather/scatter
        if self.sp_enabled:
            q, k, v = self._sp_gather(q, k, v)
        
        # 3. Select and run backend
        backend = self._select_backend(ctx.attn_metadata)
        output = backend(q, k, v, cu_seqlens=cu_seqlens)
        
        # 4. Scatter for SP
        if self.sp_enabled:
            output = self._sp_scatter(output)
        
        return output
```

### Attention Metadata

Per-step metadata controls backend behavior:

```python
@dataclass
class AttnMetadata:
    backend: str                    # "flash", "sta", "vsa", "vmoba", "sage"
    timestep_index: int             # current denoising step
    total_steps: int                # total denoising steps
    seq_len: int                    # total sequence length
    tile_indices: Optional[Tensor]  # for STA: tile assignments
    sparsity_mask: Optional[Tensor] # for VSA: dynamic mask
    block_routing: Optional[Tensor] # for VMoBA: block routing
    cu_seqlens: Optional[Tensor]    # cumulative sequence lengths (variable-len)
```

---

## Sequence Parallelism Integration

Attention backends integrate with Ulysses and Ring parallelism:

### Ulysses SP

```
Before attention:
  Per-GPU tokens: [S/P, H, D]  (P = SP degree)
  
All-to-All → Each GPU gets all tokens but subset of heads:
  Per-GPU: [S, H/P, D]

Attention computation (full sequence, subset heads)

All-to-All → Restore original distribution:
  Per-GPU: [S/P, H, D]
```

### Ring SP

Ring attention splits the sequence and rotates K/V chunks:

```
GPU 0: Q₀ × [K₀, K₁, K₂, K₃]  (receives K chunks in ring)
GPU 1: Q₁ × [K₁, K₂, K₃, K₀]
GPU 2: Q₂ × [K₂, K₃, K₀, K₁]
GPU 3: Q₃ × [K₃, K₀, K₁, K₂]
```

### CFG Parallelism

When `enable_cfg_parallel=True`:
- Positive prompts: Even-ranked CFG workers
- Negative prompts: Odd-ranked CFG workers
- After attention: `all_reduce` to combine predictions

---

## Comparison with LLM Attention

| Aspect | LLM Runtime | Diffusion Runtime |
|--------|------------|-------------------|
| **Core Pattern** | PagedAttention (KV cache blocks) | Full/Sparse Attention |
| **KV Management** | Token pool, paged blocks, radix cache | None (no KV cache) |
| **Sequence Growth** | Incremental (decode adds 1 token) | Fixed per timestep |
| **Backends** | FlashInfer, Triton | Flash, STA, VSA, VMoBA, Sage |
| **Parallelism** | TP head-splitting | SP (Ulysses/Ring) + CFG |
| **Memory** | Proportional to max_total_tokens | Proportional to sequence length |

---

## Source References

- Selector: `layers/attention/selector.py`
- Layer: `layers/attention/layer.py`
- STA Backend: `layers/attention/backends/sta.py`
- VSA Backend: `layers/attention/backends/vsa.py`
- VMoBA Backend: `layers/attention/backends/vmoba.py`
- Sage Backend: `layers/attention/backends/sage_attn.py`
- Flash Backend: `layers/attention/backends/flash_attn.py`
