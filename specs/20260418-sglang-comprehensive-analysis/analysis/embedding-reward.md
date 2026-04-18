# Embedding & Reward Models

## Overview

SGLang supports serving embedding models (for text/image retrieval and similarity) and reward models (for RLHF scoring) within the same infrastructure as generative LLMs. These models share the scheduling, memory management, and parallelism systems but produce vector/scalar outputs rather than token sequences.

**File**: `python/sglang/srt/` (integrated into main runtime)

---

## Embedding Models

### Architecture

```
Request: {"input": "text to embed", "model": "intfloat/e5-mistral-7b"}
    ↓
Scheduler → allocate KV, run forward (prefill only)
    ↓
ModelRunner → extract last hidden state / pooled output
    ↓
Response: {"embedding": [0.123, -0.456, ...], "dimensions": 4096}
```

### Key Differences from Generative Models

| Aspect | Generative LLM | Embedding Model |
|--------|---------------|-----------------|
| **Output** | Token sequence | Fixed-size vector |
| **Phases** | Prefill + Decode | Prefill only |
| **KV Cache** | Retained for decode | Freed immediately |
| **Scheduling** | Continuous batching | Batch prefill only |
| **Token Sampling** | Top-p, temperature | None |

### Implementation

Embedding models reuse the standard model runner but terminate after prefill:

```python
# In model forward:
class EmbeddingModel(nn.Module):
    def forward(self, input_ids, positions, ...):
        hidden_states = self.model(input_ids, positions)  # Standard transformer
        
        # Pooling strategy
        if self.pooling_type == "last":
            embeddings = hidden_states[last_token_indices]
        elif self.pooling_type == "mean":
            embeddings = hidden_states.mean(dim=1)
        elif self.pooling_type == "cls":
            embeddings = hidden_states[:, 0]
        
        # Optional normalization
        if self.normalize:
            embeddings = F.normalize(embeddings, p=2, dim=-1)
        
        return embeddings
```

### Supported Embedding Models

Common architectures supported:
- Mistral-based embeddings (E5-Mistral, GTE-Qwen)
- BERT/RoBERTa encoders
- Custom embedding heads on decoder models

---

## Reward Models

### Architecture

```
Request: {"input": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
    ↓
Scheduler → prefill with full conversation
    ↓
ModelRunner → extract reward score from classification head
    ↓
Response: {"score": 0.85}
```

### Key Differences from Generative Models

| Aspect | Generative LLM | Reward Model |
|--------|---------------|--------------|
| **Output** | Token sequence | Scalar score |
| **Head** | LM head (vocab projection) | Classification head (hidden → 1) |
| **Input** | Single turn or conversation | Full conversation for scoring |

### Implementation

Reward models use a score head instead of the language modeling head:

```python
class RewardModel(nn.Module):
    def __init__(self, config):
        self.model = TransformerModel(config)  # Shared backbone
        self.score_head = nn.Linear(config.hidden_size, 1)  # Scalar output
    
    def forward(self, input_ids, positions, ...):
        hidden_states = self.model(input_ids, positions)
        # Score from last token position
        reward = self.score_head(hidden_states[last_token_indices])
        return reward.squeeze(-1)
```

---

## Disaggregated Encoding

**File**: `srt/disaggregation/encode_server.py`

For high-throughput embedding deployments, SGLang supports disaggregated encoding:

### Encode Server

A dedicated server for embedding computation:

```python
class EncodeServer:
    """Dedicated embedding computation server."""
    
    def __init__(self, server_args):
        self.model = load_embedding_model(server_args)
        self.batch_processor = BatchProcessor(max_batch_size=...)
    
    async def encode(self, texts: List[str]) -> List[torch.Tensor]:
        # Tokenize
        input_ids = self.tokenizer(texts)
        # Batch forward
        embeddings = self.model(input_ids)
        return embeddings
```

### Encode Receiver

**File**: `srt/disaggregation/encode_receiver.py`

Manages distributed embedding aggregation:

```python
class EmbeddingData:
    """Manages partial embeddings from distributed encoding."""
    
    def __init__(self, num_parts: int):
        self.partial_embeddings: List[Optional[torch.Tensor]] = [None] * num_parts
        self.complete: bool = False
    
    def add_partial(self, rank: int, embedding: torch.Tensor):
        self.partial_embeddings[rank] = embedding
        if all(e is not None for e in self.partial_embeddings):
            self.complete = True
    
    def get_full_embedding(self) -> torch.Tensor:
        return torch.cat(self.partial_embeddings, dim=-1)
```

### Static Embedding Cache

**File**: `srt/disaggregation/mem_cache/multimodal_cache.py`

Caches computed embeddings for reuse:

```python
class MultiModalStaticCache:
    """Cache for computed multimodal embeddings."""
    
    def get(self, key: str) -> Optional[torch.Tensor]:
        """Retrieve cached embedding by content hash."""
    
    def put(self, key: str, embedding: torch.Tensor):
        """Store embedding with LRU eviction."""
```

---

## Logits Processing for Non-Generative Models

**File**: `srt/layers/logits_processor.py`

The logits processor handles different output types:

```python
class LogitsProcessor:
    def forward(self, hidden_states, forward_batch):
        if self.is_embedding_model:
            # Return hidden states directly (pooled)
            return self.pool_embeddings(hidden_states, forward_batch)
        elif self.is_reward_model:
            # Apply score head
            return self.score_head(hidden_states)
        else:
            # Standard LM head for generative models
            return self.lm_head(hidden_states)
```

---

## Multi-Platform Support

The disaggregation system supports multiple hardware backends:

| Platform | Location | Purpose |
|----------|----------|---------|
| Standard (CUDA) | `disaggregation/` | Default GPU execution |
| Ascend (NPU) | `disaggregation/ascend/` | Huawei Ascend accelerators |
| Mooncake | `disaggregation/mooncake/` | Alibaba Mooncake TPUs |
| NIxL | `disaggregation/nixl/` | NIxL accelerator |
| Fake | `disaggregation/fake/` | Testing/simulation |

---

## Configuration

```bash
# Serve embedding model
python -m sglang.launch_server \
    --model-path intfloat/e5-mistral-7b-instruct \
    --is-embedding-model

# Serve reward model
python -m sglang.launch_server \
    --model-path Nexusflow/Starling-RM-34B \
    --is-reward-model

# Disaggregated embedding server
python -m sglang.launch_server \
    --model-path BAAI/bge-large-en-v1.5 \
    --is-embedding-model \
    --enable-disaggregation
```

API usage:
```python
# Embedding
response = client.embeddings.create(
    model="intfloat/e5-mistral-7b-instruct",
    input=["Hello world", "Semantic search"]
)
# response.data[0].embedding → [0.123, -0.456, ...]

# Reward scoring
response = client.chat.completions.create(
    model="Nexusflow/Starling-RM-34B",
    messages=[...],
    extra_body={"return_reward_score": True}
)
```

---

## Source References

- Encode Server: `srt/disaggregation/encode_server.py`
- Encode Receiver: `srt/disaggregation/encode_receiver.py`
- Embedding Cache: `srt/disaggregation/mem_cache/multimodal_cache.py`
- Logits Processor: `srt/layers/logits_processor.py`
- Model Runner: `srt/model_executor/model_runner.py`
