# API Surface & Contracts: SGLang

**Date**: 2026-04-18 | **Branch**: `20260418-sglang-comprehensive-analysis`

---

## 1. Python SDK API (Public)

### Engine API (`sglang.Engine`)

```python
from sglang import Engine, ServerArgs

# Initialize engine
engine = Engine(model_path="meta-llama/Llama-3-8B-Instruct", **kwargs)

# Synchronous generation
result = engine.generate(
    prompt: str | List[int],
    sampling_params: dict = None,  # temperature, top_p, max_new_tokens, etc.
    stream: bool = False,
    lora_path: str = None,
)

# Async generation
async for chunk in engine.async_generate(...):
    yield chunk

# Embedding
embedding = engine.encode(prompt: str | List[int])

# LoRA management
engine.load_lora_adapter(lora_path: str, name: str)
engine.unload_lora_adapter(name: str)

# Weight updates
engine.update_weights_from_disk(model_path: str, load_format: str)
engine.update_weights_from_tensor(named_tensors: Dict[str, torch.Tensor])

# Memory management
engine.release_memory_occupation()
engine.resume_memory_occupation()

# Session management (multi-turn)
session_id = engine.open_session(capacity: int)
engine.close_session(session_id)

# Shutdown
engine.shutdown()
```

### Frontend Language API (`sglang.lang`)

```python
import sglang as sgl

@sgl.function
def multi_turn_chat(s, question):
    s += sgl.system("You are a helpful assistant.")
    s += sgl.user(question)
    s += sgl.assistant(sgl.gen("answer", max_tokens=256))

# Execute
state = multi_turn_chat.run(question="Hello")
print(state["answer"])

# Key primitives
sgl.gen(name, max_tokens, temperature, regex, json_schema)
sgl.select(name, choices: List[str])
sgl.image(url_or_path)
sgl.video(url_or_path)
sgl.separate_reasoning(name)
```

---

## 2. HTTP Server API (OpenAI-Compatible)

### Chat Completions
```
POST /v1/chat/completions
Content-Type: application/json

{
  "model": "meta-llama/Llama-3-8B-Instruct",
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."}
  ],
  "temperature": 0.7,
  "top_p": 0.9,
  "max_tokens": 256,
  "stream": true,
  "n": 1,
  "response_format": {"type": "json_object"},  // structured output
  "tools": [...],                               // function calling
  "lora_path": "path/to/adapter"               // LoRA selection
}
```

### Completions
```
POST /v1/completions
{
  "model": "...",
  "prompt": "...",
  "max_tokens": 256,
  "temperature": 0.0,
  "regex": "[0-9]+",         // constrained output
  "json_schema": {...}       // JSON schema constraint
}
```

### Embeddings
```
POST /v1/embeddings
{
  "model": "...",
  "input": "text to embed"
}
```

### Model Management
```
GET  /v1/models                    # List models
GET  /health                       # Health check
GET  /get_server_info              # Server info + metrics
POST /flush_cache                  # Clear KV cache

# LoRA
POST /lora/load    {"lora_path": "...", "lora_name": "..."}
POST /lora/unload  {"lora_name": "..."}

# Weight update
POST /update_weights_from_disk  {"model_path": "..."}
```

### SGLang Extensions (beyond OpenAI)
```
POST /generate     # Raw generation with full parameter control
{
  "text": "...",
  "sampling_params": {
    "max_new_tokens": 256,
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": 50,
    "min_p": 0.05,
    "frequency_penalty": 0.0,
    "presence_penalty": 0.0,
    "repetition_penalty": 1.0,
    "regex": null,
    "json_schema": null,
    "n": 1
  },
  "stream": false,
  "lora_path": null,
  "session_params": null
}
```

---

## 3. CLI Interface

### Server Launch
```bash
# Basic launch
python -m sglang.launch_server --model-path meta-llama/Llama-3-8B-Instruct

# Full options
python -m sglang.launch_server \
  --model-path <path> \
  --port 30000 \
  --host 0.0.0.0 \
  --tp 4 \                          # Tensor parallelism
  --dp 2 \                          # Data parallelism
  --pp 2 \                          # Pipeline parallelism
  --ep 8 \                          # Expert parallelism
  --mem-fraction-static 0.85 \      # GPU memory fraction
  --max-running-requests 512 \
  --max-total-tokens 131072 \
  --chunked-prefill-size 8192 \
  --quantization fp8 \              # Quantization method
  --kv-cache-dtype fp8_e5m2 \       # KV cache quantization
  --enable-torch-compile \
  --cuda-graph-max-bs 32 \
  --lora-modules adapter1=path1 \
  --speculative-algorithm EAGLE \
  --speculative-num-steps 5 \
  --schedule-policy lpm \           # LPM/DFS/FCFS/LOF
  --enable-overlap-schedule \
  --enable-pd-disagg \              # PD disaggregation
  --grammar-backend xgrammar        # Structured output
```

### Offline Benchmarking
```bash
python -m sglang.bench_serving \
  --backend sglang \
  --model meta-llama/Llama-3-8B-Instruct \
  --num-prompts 1000 \
  --request-rate 10
```

---

## 4. gRPC Interface

```protobuf
service SGLangService {
  rpc Generate(GenerateRequest) returns (stream GenerateResponse);
  rpc GetServerInfo(Empty) returns (ServerInfo);
}
```

---

## 5. Internal Communication (ZMQ)

### Tokenizer → Scheduler
```python
# Message format (msgpack serialized)
TokenizedGenerateReqInput:
  rid: str              # Request ID
  input_ids: List[int]  # Tokenized input
  sampling_params: dict
  lora_path: Optional[str]
  pixel_values: Optional[Tensor]  # Multimodal
```

### Scheduler → Detokenizer
```python
# Batch result message
BatchTokenIDOut:
  rids: List[str]
  output_tokens: List[List[int]]
  finished: List[bool]
  meta_info: List[dict]  # logprobs, usage stats
```

---

## 6. Diffusion API

### HTTP (multimodal_gen)
```
POST /v1/images/generations
{
  "model": "wan-2.1-t2v",
  "prompt": "A cat playing piano",
  "size": "720x480",
  "n": 1,
  "num_inference_steps": 50
}

POST /generate
{
  "prompt": "...",
  "negative_prompt": "...",
  "height": 720,
  "width": 480,
  "num_frames": 81,
  "num_inference_steps": 50,
  "guidance_scale": 7.5,
  "seed": 42
}
```

### Python SDK (DiffGenerator)
```python
from sglang.multimodal_gen import DiffGenerator

gen = DiffGenerator(model_path="wan-2.1-t2v", tp_size=4)
output = gen.generate(
    prompt="A sunset over mountains",
    height=720, width=480,
    num_frames=81,
    num_inference_steps=50
)
output.save("output.mp4")
```

---

## 7. Weight Sync / RL Integration API

```python
# For RL frameworks (verl, AReaL, etc.)
engine.init_weights_update_group(
    master_address: str,
    master_port: int,
    rank_offset: int,
    world_size: int,
    group_name: str
)

engine.update_weights_from_distributed(
    name: str,
    dtype: torch.dtype,
    shape: List[int]
)

engine.update_weights_from_ipc(
    name: str,
    handle: bytes,  # CUDA IPC handle
    shape: List[int]
)
```

---

## 8. Configuration Contract (ServerArgs)

Key server arguments that constitute the configuration contract:

| Category | Key Arguments | Default |
|----------|--------------|---------|
| Model | model_path, tokenizer_path, dtype | required, same, auto |
| Memory | mem_fraction_static, max_total_tokens | 0.88, auto |
| Scheduling | schedule_policy, max_running_requests | lpm, auto |
| Parallelism | tp_size, dp_size, pp_size, ep_size | 1,1,1,1 |
| Quantization | quantization, kv_cache_dtype | None, auto |
| Speculative | speculative_algorithm, num_steps | None, 5 |
| LoRA | lora_modules, max_loras_per_batch | None, 8 |
| Compilation | enable_torch_compile, cuda_graph_max_bs | False, 160 |
| Overlap | enable_overlap_schedule | True |
| Disagg | enable_pd_disagg | False |
