# Structured Output & Constrained Decoding

## Overview

SGLang provides grammar-based constrained generation that ensures model outputs conform to specified formats (JSON schemas, regular expressions, EBNF grammars, or structural tags). The system integrates with the scheduler's sampling phase to apply vocabulary-level masks during token selection.

**File**: `python/sglang/srt/constrained/`

---

## Architecture

```
Request arrives with constraint spec (json_schema / regex / ebnf / structural_tag)
    ↓
GrammarManager.process_req_with_grammar(req)
    ↓ (async compilation, possibly cached)
GrammarBackend creates GrammarObject → req.grammar
    ↓
During sampling: grammar.fill_next_token_bitmask(token_ids)
    ↓ (masks invalid tokens to -inf logit)
Token sampled from valid vocabulary only
    ↓
grammar.accept_token(sampled_token)  → advances FSM state
```

---

## GrammarManager

**File**: `constrained/grammar_manager.py`

Orchestrates grammar lifecycle within the scheduler:

```python
class GrammarManager:
    def __init__(self, scheduler):
        self.grammar_queue: List[Req] = []
        self.grammar_backend = create_grammar_backend(
            server_args, tokenizer, vocab_size, eos_token_id
        )
        # Distributed sync group for multi-GPU consistency
        self.grammar_sync_group = scheduler.dp_tp_cpu_group
```

### Request Processing Flow

1. **Constraint Detection**: Check `req.sampling_params` for `json_schema`, `regex`, `ebnf`, or `structural_tag`
2. **Cache Lookup**: `grammar_backend.get_cached_or_future_value(key, require_reasoning)`
   - Cache hit: Assign grammar object immediately
   - Cache miss: Add to `grammar_queue` for async compilation
3. **Async Polling**: `get_ready_grammar_requests()` polls futures until compilation completes
4. **Distributed Sync**: Multi-GPU ensures all ranks have consistent grammar state:
   - Rank 0 determines ready set → broadcasts to other ranks
   - `all_gather` for ready/failed indices, then `intersect(ready)` / `union(failed)`

### Constraint Types

| Type | Key Format | Backend Method |
|------|-----------|----------------|
| `json_schema` | `("json", schema_str)` | `create_json_grammar(schema)` |
| `regex` | `("regex", pattern_str)` | `create_regex_grammar(pattern)` |
| `ebnf` | `("ebnf", grammar_str)` | `create_ebnf_grammar(grammar)` |
| `structural_tag` | `("structural_tag", tag_str)` | `create_structural_tag_grammar(tag)` |

---

## Grammar Backends

### XGrammar (Primary)

**File**: `constrained/xgrammar_backend.py`

Uses the [XGrammar](https://github.com/mlc-ai/xgrammar) library:

```python
class XGrammarGrammarBackend(BaseGrammarBackend):
    def __init__(self, tokenizer, vocab_size, ...):
        self.grammar_compiler = xgr.GrammarCompiler(
            tokenizer_info, max_threads=num_workers
        )
    
    def create_json_grammar(self, json_schema: str) -> XGrammarGrammarObject:
        compiled = self.grammar_compiler.compile_json_schema(json_schema)
        return XGrammarGrammarObject(compiled, vocab_size)
```

Key features:
- **ThreadPoolExecutor** for async compilation
- **LRU cache** for compiled grammars
- **Bitmask operations**: Uses Triton kernels for efficient vocab masking
- **Jump-forward optimization**: Skips deterministic token sequences

### Outlines

**File**: `constrained/outlines_backend.py`

Integration with the [Outlines](https://github.com/dottxt-ai/outlines) library:

- FSM-based constraint enforcement
- Supports regex and JSON schema via Outlines' guide system
- `jump_forward.py`: Optimized multi-token acceptance when the grammar dictates a unique continuation

### LLGuidance

**File**: `constrained/llguidance_backend.py`

Integration with Microsoft's LLGuidance for constraint specification.

### Reasoner Grammar

**File**: `constrained/reasoner_grammar_backend.py`

Specialized grammar that wraps reasoning-specific patterns (e.g., chain-of-thought formatting with structured output).

---

## Grammar Object Interface

**File**: `constrained/base_grammar_backend.py`

```python
class BaseGrammarObject(ABC):
    @abstractmethod
    def fill_next_token_bitmask(self, vocab_mask: torch.Tensor, idx: int):
        """Fill a bitmask indicating valid next tokens."""
        pass
    
    @abstractmethod
    def accept_token(self, token: int) -> bool:
        """Advance the grammar state with accepted token."""
        pass
    
    @abstractmethod
    def is_terminated(self) -> bool:
        """Check if grammar has reached accepting state."""
        pass
    
    def try_jump_forward(self, tokenizer) -> Optional[str]:
        """Return deterministic continuation if grammar forces unique tokens."""
        return None
```

### Bitmask Application

During sampling, the grammar object produces a bitmask over the vocabulary:

```python
# In sampling phase (logits_processor.py)
if req.grammar is not None:
    # Fill bitmask: 1 = valid token, 0 = invalid
    req.grammar.fill_next_token_bitmask(vocab_bitmask, batch_idx)
    # Apply: invalid tokens get -inf logit
    logits = apply_bitmask(logits, vocab_bitmask)
```

### Triton Bitmask Operations

**File**: `constrained/triton_ops/bitmask_ops.py`

Optimized GPU kernels for applying vocabulary bitmasks:
- Bit-packed representation (32 tokens per int32)
- Triton kernel for parallel mask application across batch

---

## Distributed Grammar Synchronization

For multi-GPU (TP/DP) setups, all ranks must agree on grammar state:

```python
def get_ready_grammar_requests(self):
    # Each rank determines its own ready/failed sets
    ready_req_idxs, failed_req_idxs = self._local_poll()
    
    if self.grammar_sync_size > 1:
        # All-gather to synchronize across ranks
        all_ready = torch.distributed.all_gather(ready_tensor, group=self.grammar_sync_group)
        all_failed = torch.distributed.all_gather(failed_tensor, group=self.grammar_sync_group)
        
        # Intersection of ready (all must agree) 
        ready_req_idxs = intersect(all_ready)
        # Union of failed (any failure counts)
        failed_req_idxs = union(all_failed)
```

This ensures deterministic behavior: a request only proceeds when ALL ranks have successfully compiled its grammar.

---

## Performance Considerations

1. **Async Compilation**: Grammar compilation happens in background thread pool; requests wait in `grammar_queue` until ready
2. **Caching**: Compiled grammars are cached by (type, spec) key — repeated schemas compile once
3. **Jump-Forward**: When grammar forces a unique token sequence, tokens are accepted without model inference
4. **Bitmask Fusion**: GPU-based bitmask operations avoid CPU-GPU synchronization per token

---

## Configuration

```bash
# Enable XGrammar backend (default)
python -m sglang.launch_server \
    --model-path meta-llama/Llama-3-8B \
    --grammar-backend xgrammar

# Disable grammar support
python -m sglang.launch_server \
    --grammar-backend none
```

API usage:
```python
# JSON schema constraint
response = client.chat.completions.create(
    model="meta-llama/Llama-3-8B",
    messages=[{"role": "user", "content": "..."}],
    extra_body={"json_schema": '{"type": "object", "properties": {"name": {"type": "string"}}}'}
)

# Regex constraint
response = client.chat.completions.create(
    model="meta-llama/Llama-3-8B",
    messages=[{"role": "user", "content": "..."}],
    extra_body={"regex": r"\d{4}-\d{2}-\d{2}"}
)
```

---

## Source References

- GrammarManager: `constrained/grammar_manager.py`
- Base Interface: `constrained/base_grammar_backend.py`
- XGrammar: `constrained/xgrammar_backend.py`
- Outlines: `constrained/outlines_backend.py`
- LLGuidance: `constrained/llguidance_backend.py`
- Triton Bitmask: `constrained/triton_ops/bitmask_ops.py`
