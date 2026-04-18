# SGLang ServerArgs Configuration Reference Guide

## Executive Summary

The `ServerArgs` dataclass in SGLang is a comprehensive configuration interface for the inference server. It manages over 200 configuration parameters spanning model loading, memory management, distributed execution, performance optimization, and advanced features like speculative decoding, disaggregation, and multi-modal support. This document provides a complete reference guide organized by functional domain.

---

## 1. Model and Tokenizer Configuration

### Core Model Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_path` | `str` | *Required* | Path to model weights (local folder or Hugging Face repo ID) |
| `tokenizer_path` | `Optional[str]` | Defaults to `model_path` | Path to the tokenizer |
| `tokenizer_mode` | `str` | `"auto"` | `"auto"` uses fast tokenizer if available, `"slow"` forces slow tokenizer |
| `tokenizer_worker_num` | `int` | `1` | Number of tokenizer worker processes |
| `skip_tokenizer_init` | `bool` | `False` | Skip tokenizer initialization; input_ids passed directly |
| `load_format` | `str` | `"auto"` | Weight loading format: `"auto"`, `"pt"`, `"safetensors"`, `"npcache"`, `"dummy"`, `"sharded_state"`, `"gguf"`, `"bitsandbytes"`, `"layered"`, `"flash_rl"`, `"remote"`, `"remote_instance"`, `"fastsafetensors"`, `"private"` |
| `model_loader_extra_config` | `str` | `"{}"` | JSON-formatted extra config for model loader |
| `trust_remote_code` | `bool` | `False` | Allow custom models with custom modeling files |
| `context_length` | `Optional[int]` | `None` | Override model's max context length |
| `is_embedding` | `bool` | `False` | Use a CausalLM as an embedding model |
| `enable_multimodal` | `Optional[bool]` | `None` | Enable multimodal functionality |
| `revision` | `Optional[str]` | `None` | Model version: branch name, tag, or commit ID |
| `model_impl` | `str` | `"auto"` | Model implementation: `"auto"`, `"sglang"`, `"transformers"`, `"mindspore"` |

### Tokenizer Batching Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enable_tokenizer_batch_encode` | `bool` | `False` | Batch tokenization for multiple inputs |
| `disable_tokenizer_batch_decode` | `bool` | `False` | Disable batch decoding |
| `enable_dynamic_batch_tokenizer` | `bool` | `False` | Enable async dynamic batch tokenizer |
| `dynamic_batch_tokenizer_batch_size` | `int` | `32` | Max batch size for dynamic batch tokenizer |
| `dynamic_batch_tokenizer_batch_timeout` | `float` | `0.002` | Timeout (seconds) for batching tokenization |

---

## 2. HTTP Server Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `host` | `str` | `"127.0.0.1"` | HTTP server host address |
| `port` | `int` | `30000` | HTTP server port number |
| `fastapi_root_path` | `str` | `""` | App root path for path-based routing proxy |
| `grpc_mode` | `bool` | `False` | Use gRPC server instead of HTTP |
| `nccl_port` | `Optional[int]` | `None` | Port for NCCL distributed backend |
| `skip_server_warmup` | `bool` | `False` | Skip server warmup phase on startup |
| `warmups` | `Optional[str]` | `None` | CSV of custom warmup function names |

---

## 3. Quantization and Data Type Configuration

### Data Type Settings

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dtype` | `str` | `"auto"` | Model weights/activations dtype: `"auto"`, `"half"`, `"float16"`, `"bfloat16"`, `"float"`, `"float32"` |
| `quantization` | `Optional[str]` | `None` | Quantization method |
| `quantization_param_path` | `Optional[str]` | `None` | Path to JSON with KV cache scaling factors |
| `kv_cache_dtype` | `str` | `"auto"` | KV cache dtype: `"auto"`, `"fp8_e5m2"`, `"fp8_e4m3"`, `"bf16"`, `"fp4_e2m1"` |
| `enable_fp32_lm_head` | `bool` | `False` | LM head outputs in FP32 |

### Quantization Methods Supported

- **NVIDIA**: `awq`, `gptq`, `marlin`, `gptq_marlin`, `awq_marlin`, `bitsandbytes`
- **FP8/FP4**: `fp8`, `mxfp8`, `modelopt`, `modelopt_fp8`, `modelopt_fp4`, `petit_nvfp8`, `mxfp4`
- **Integer**: `w8a8_int8`, `w8a8_fp8`, `w4afp8`
- **Specialized**: `gguf`, `qoq`, `compressed-tensors`, `auto-round`, `quark_int4fp8_moe`, `modelslim`

### ModelOpt Quantization

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `modelopt_quant` | `Optional[Union[str, Dict]]` | `None` | Config: `'fp8'`, `'int4_awq'`, `'w4a8_awq'`, `'nvfp4'`, `'nvfp4_awq'` |
| `modelopt_checkpoint_restore_path` | `Optional[str]` | `None` | Restore previously saved checkpoint |
| `modelopt_checkpoint_save_path` | `Optional[str]` | `None` | Save checkpoint after quantization |
| `modelopt_export_path` | `Optional[str]` | `None` | Export in HuggingFace format |
| `quantize_and_serve` | `bool` | `False` | Quantize and immediately serve |

---

## 4. Memory and Scheduling Configuration

### Memory Management

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `mem_fraction_static` | `Optional[float]` | Auto-calculated | Fraction of GPU memory for static allocation |
| `max_total_tokens` | `Optional[int]` | `None` | Maximum tokens in memory pool |
| `page_size` | `Optional[int]` | `1` | Number of tokens per page in paged attention |

### Request Queuing and Batch Size

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_running_requests` | `Optional[int]` | `None` | Maximum running requests |
| `max_queued_requests` | `Optional[int]` | `None` | Maximum queued requests |
| `max_prefill_tokens` | `int` | `16384` | Maximum tokens in prefill batch |
| `prefill_max_requests` | `Optional[int]` | `None` | Maximum requests in prefill batch |

### Chunked Prefill Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `chunked_prefill_size` | `Optional[int]` | Auto-calculated by GPU type | Max tokens in chunk; -1 disables |
| `enable_dynamic_chunking` | `bool` | `False` | Dynamic chunk size adjustment for PP |
| `enable_mixed_chunk` | `bool` | `False` | Mix prefill and decode in a batch |

### Scheduling Policy

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `schedule_policy` | `str` | `"fcfs"` | Policy: `"fcfs"`, `"lpm"`, `"random"`, `"dfs-weight"`, `"lof"`, `"priority"`, `"routing-key"` |
| `schedule_conservativeness` | `float` | `1.0` | Conservativeness (> 1.0 = more conservative) |
| `enable_priority_scheduling` | `bool` | `False` | Enable priority scheduling |
| `priority_scheduling_preemption_threshold` | `int` | `10` | Min priority difference for preemption |

### Caching

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `disable_radix_cache` | `bool` | `False` | Disable RadixAttention prefix caching |
| `radix_eviction_policy` | `str` | `"lru"` | Eviction policy: `"lru"`, `"lfu"` |

---

## 5. Runtime and Parallelism Configuration

### Parallelism

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `tp_size` | `int` | `1` | Tensor parallelism size |
| `pp_size` | `int` | `1` | Pipeline parallelism size |
| `dp_size` | `int` | `1` | Data parallelism size |
| `pp_max_micro_batch_size` | `Optional[int]` | `None` | Max micro batch size in PP |
| `pp_async_batch_depth` | `int` | `0` | Async batch depth of PP |

### Distributed Setup

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dist_init_addr` | `Optional[str]` | `None` | Host address for distributed init |
| `nnodes` | `int` | `1` | Number of nodes |
| `node_rank` | `int` | `0` | Node rank in distributed setup |
| `dist_timeout` | `Optional[int]` | `None` | Timeout for torch.distributed initialization |

### GPU Management

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `device` | `Optional[str]` | Auto-detected | `"cuda"`, `"xpu"`, `"hpu"`, `"npu"`, `"cpu"` |
| `base_gpu_id` | `int` | `0` | Base GPU ID for allocation |
| `gpu_id_step` | `int` | `1` | Delta between consecutive GPU IDs |

### Watchdog and Safety

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `watchdog_timeout` | `float` | `300` | Hard watchdog timeout (seconds) |
| `soft_watchdog_timeout` | `Optional[float]` | `None` | Soft watchdog for debug info dumps |
| `sleep_on_idle` | `bool` | `False` | Reduce CPU usage when idle |

---

## 6. Logging and Observability Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `log_level` | `str` | `"info"` | Logging level: `"debug"`, `"info"`, `"warning"`, `"error"` |
| `log_requests` | `bool` | `False` | Log all request metadata/inputs/outputs |
| `log_requests_level` | `int` | `2` | Verbosity (0: metadata, 1: +params, 2: +partial I/O, 3: all) |
| `enable_metrics` | `bool` | `False` | Enable Prometheus metrics |
| `enable_trace` | `bool` | `False` | Enable OpenTelemetry tracing |
| `otlp_traces_endpoint` | `str` | `"localhost:4317"` | OpenTelemetry collector endpoint |
| `show_time_cost` | `bool` | `False` | Show time cost of custom marks |
| `crash_dump_folder` | `Optional[str]` | `None` | Folder for crash dumps |

---

## 7. API and Integration Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `api_key` | `Optional[str]` | `None` | Server API key for OpenAI-compatible server |
| `admin_api_key` | `Optional[str]` | `None` | Admin API key for management endpoints |
| `served_model_name` | `Optional[str]` | Defaults to `model_path` | Override model name in v1/models |
| `chat_template` | `Optional[str]` | `None` | Built-in chat template name or path |
| `reasoning_parser` | `Optional[str]` | `None` | Reasoning model parser |
| `tool_call_parser` | `Optional[str]` | `None` | Tool-call parser |
| `sampling_defaults` | `str` | `"model"` | Default sampling params source: `"openai"` or `"model"` |
| `load_balance_method` | `str` | `"auto"` | Strategy: `"auto"`, `"round_robin"`, `"total_requests"`, `"total_tokens"` |

---

## 8. LoRA Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enable_lora` | `Optional[bool]` | `None` | Enable LoRA support |
| `lora_paths` | `Optional[Union[dict, List]]` | `None` | LoRA adapter paths |
| `lora_target_modules` | `Optional[Union[set, List[str]]]` | `None` | Target modules for LoRA |
| `max_lora_rank` | `Optional[int]` | `None` | Maximum LoRA rank |
| `max_loaded_loras` | `Optional[int]` | `None` | Max adapters in CPU memory |
| `max_loras_per_batch` | `int` | `8` | Max adapters per batch |
| `lora_backend` | `str` | `"csgmv"` | Backend: `"triton"`, `"csgmv"`, `"ascend"`, `"torch_native"` |
| `lora_eviction_policy` | `str` | `"lru"` | Eviction policy: `"lru"`, `"fifo"` |

---

## 9. Kernel Backend Configuration

### Attention Backend

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `attention_backend` | `Optional[str]` | Auto-selected | Unified attention backend |
| `prefill_attention_backend` | `Optional[str]` | `None` | Prefill-specific backend |
| `decode_attention_backend` | `Optional[str]` | `None` | Decode-specific backend |

**Common backends**: `"triton"`, `"torch_native"`, `"flex_attention"`, `"flashinfer"`, `"flashmla"`, `"cutlass_mla"`, `"fa3"`, `"fa4"`, `"trtllm_mla"`

### Other Backends

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `sampling_backend` | `Optional[str]` | Auto | `"flashinfer"`, `"pytorch"`, `"ascend"` |
| `grammar_backend` | `Optional[str]` | `"xgrammar"` | `"xgrammar"`, `"outlines"`, `"llguidance"`, `"none"` |
| `fp8_gemm_runner_backend` | `str` | `"auto"` | FP8 GEMM runner selection |
| `fp4_gemm_runner_backend` | `str` | `"flashinfer_cutlass"` | NVFP4 GEMM runner |

---

## 10. Speculative Decoding Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `speculative_algorithm` | `Optional[str]` | `None` | `"EAGLE"`, `"EAGLE3"`, `"STANDALONE"`, `"NGRAM"` |
| `speculative_draft_model_path` | `Optional[str]` | Auto-set for certain models | Path to draft model |
| `speculative_num_steps` | `Optional[int]` | Auto-chosen | Tokens predicted from draft |
| `speculative_eagle_topk` | `Optional[int]` | Auto-chosen | Tokens sampled per step |
| `speculative_num_draft_tokens` | `Optional[int]` | Auto-chosen | Total draft tokens |
| `speculative_accept_threshold_single` | `float` | `1.0` | Accept probability threshold |
| `speculative_attention_mode` | `str` | `"prefill"` | `"prefill"` or `"decode"` |

### N-Gram Speculative

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `speculative_ngram_min_match_window_size` | `int` | `1` | Minimum window size |
| `speculative_ngram_max_match_window_size` | `int` | `12` | Maximum window size |
| `speculative_ngram_branch_length` | `int` | `18` | Branch length |
| `speculative_ngram_capacity` | `int` | `10,000,000` | Cache capacity |

---

## 11. Expert Parallelism (MoE) Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `ep_size` | `int` | `1` | Expert parallelism size |
| `moe_a2a_backend` | `str` | `"none"` | `"none"`, `"deepep"`, `"mooncake"`, `"mori"`, `"ascend_fuseep"`, `"flashinfer"` |
| `moe_runner_backend` | `str` | `"auto"` | MoE runner backend |
| `deepep_mode` | `str` | `"auto"` | DeepEP mode: `"auto"`, `"normal"`, `"low_latency"` |
| `enable_eplb` | `bool` | `False` | Enable expert load balancing |
| `eplb_algorithm` | `str` | `"auto"` | EPLB algorithm |
| `eplb_rebalance_num_iterations` | `int` | `1000` | Iterations between rebalances |
| `ep_num_redundant_experts` | `int` | `0` | Redundant experts to allocate |

---

## 12. Advanced Caching Configuration

### Hierarchical Cache (HiCache)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enable_hierarchical_cache` | `bool` | `False` | Enable hierarchical cache |
| `hicache_ratio` | `float` | `2.0` | Host/device pool size ratio |
| `hicache_size` | `int` | `0` | Host pool size in GB |
| `hicache_write_policy` | `str` | `"write_through"` | `"write_back"`, `"write_through"`, `"write_through_selective"` |
| `hicache_io_backend` | `str` | `"kernel"` | `"direct"`, `"kernel"`, `"kernel_ascend"` |
| `hicache_storage_backend` | `Optional[str]` | `None` | `"file"`, `"mooncake"`, `"hf3fs"`, `"nixl"`, `"aibrix"`, `"dynamic"`, `"eic"` |

---

## 13. Disaggregation Configuration

### Prefill-Decode Disaggregation

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `disaggregation_mode` | `str` | `"null"` | `"null"`, `"prefill"`, `"decode"` |
| `disaggregation_transfer_backend` | `str` | `"mooncake"` | `"mooncake"`, `"nixl"`, `"ascend"`, `"fake"`, `"mori"` |
| `disaggregation_bootstrap_port` | `int` | `8998` | Bootstrap server port |
| `disaggregation_decode_tp` | `Optional[int]` | Defaults to prefill tp_size | Decode TP size |
| `disaggregation_decode_dp` | `Optional[int]` | Defaults to prefill dp_size | Decode DP size |

### Encoder-Language Disaggregation

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `encoder_only` | `bool` | `False` | Launch encoder-only server for MLLM |
| `language_only` | `bool` | `False` | Load language model weights only |
| `encoder_transfer_backend` | `str` | `"zmq_to_scheduler"` | Transfer backend |

---

## 14. Optimization Configuration

### CUDA Graph

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `cuda_graph_max_bs` | `Optional[int]` | Auto-calculated | Max batch size for CUDA graph |
| `cuda_graph_bs` | `Optional[List[int]]` | Auto-generated | Batch sizes to capture |
| `disable_cuda_graph` | `bool` | `False` | Disable CUDA graph entirely |

### Torch Compilation

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enable_torch_compile` | `bool` | `False` | Optimize with torch.compile |
| `torch_compile_max_bs` | `int` | `32` | Max batch size for torch.compile |
| `torchao_config` | `str` | `""` | TorchAO quantization config |

### Collective Communication

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enable_custom_all_reduce` | `bool` | True (default) | Custom all-reduce kernel |
| `disable_custom_all_reduce` | `bool` | `False` | Disable custom all-reduce |
| `enable_mscclpp` | `bool` | `False` | Use mscclpp for small messages |
| `enable_torch_symm_mem` | `bool` | `False` | Torch symmetric memory (SM90+) |

### Scheduling and Overlapping

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `disable_overlap_schedule` | `bool` | False (default) | Disable overlap scheduler |
| `enable_two_batch_overlap` | `bool` | `False` | Two micro-batch overlap |
| `enable_single_batch_overlap` | `bool` | `False` | Computation-communication overlap |
| `num_continuous_decode_steps` | `int` | `1` | Continuous decode steps |

### DP Attention

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enable_dp_attention` | `bool` | `False` | DP for attention + TP for FFN |
| `enable_dp_lm_head` | `bool` | `False` | Vocab parallel to avoid all-gather |

---

## 15. Multi-Modal Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `mm_max_concurrent_calls` | `int` | `32` | Max concurrent MM processing calls |
| `mm_per_request_timeout` | `float` | `10.0` | Timeout per MM request |
| `mm_enable_dp_encoder` | `bool` | `False` | Enable DP for MM encoder |
| `limit_mm_data_per_request` | `Optional[Dict]` | `None` | Limit modalities per request |
| `disable_fast_image_processor` | `bool` | `False` | Use base image processor |

---

## 16. Post-Initialization Processing

The `__post_init__()` method performs critical configuration orchestration through 17 major processing stages:

1. **Load Balance Normalization**: Auto-selects strategy based on mode
2. **Deprecated Arguments**: Handles backward compatibility
3. **Default Values**: Sets tokenizer_path, served_model_name, device, random_seed
4. **Device-Specific Backends**: HPU, CPU, NPU-specific defaults
5. **GPU Memory Settings**: Calculates chunked_prefill_size, cuda_graph_max_bs, mem_fraction_static
6. **Model-Specific Adjustments**: DeepSeek DSA, GPT-OSS, Llama4, Gemma, Mamba
7. **Data Parallelism**: Adjusts chunked_prefill_size and conservativeness
8. **MoE Configuration**: Validates and auto-sets ep_size
9. **Pipeline Parallelism**: Disables overlap when pp > 1
10. **Speculative Decoding**: Auto-chooses parameters, sets max_running_requests
11. **Load Format Handling**: GGUF detection, remote URLs
12. **Disaggregation Setup**: Validates PD configuration
13. **Cache Compatibility**: Validates mutual exclusivity
14. **Deterministic Inference**: Sets backends for reproducibility
15. **Environment Variables**: Sets SGLANG_* vars
16. **Kernel Backend Handling**: Attention/sampling compatibility
17. **CUDA Graph Generation**: Creates batch size lists

### GPU Memory-Based Defaults

| GPU Type | Memory | chunked_prefill_size | cuda_graph_max_bs |
|----------|--------|---------------------|-------------------|
| T4, 4080 | <20GB | 2048 | 8 |
| A10, 4090 | <35GB | 2048 | 24 (TP<4) / 80 |
| A100 40GB | <60GB | 4096 | 32 (TP<4) / 160 |
| H100 | <90GB | 8192 | 256 (TP<4) / 512 |
| H20, H200 | <160GB | 8192 | 256 (TP<4) / 512 |
| B200, MI300 | >=160GB | 16384 | 512 |

---

## 17. Parameter Interactions and Constraints

### Critical Mutual Exclusivity

| Parameters | Constraint |
|-----------|-----------|
| `enable_hierarchical_cache` + `disable_radix_cache` | Mutually exclusive |
| `encoder_only` + `language_only` | Cannot set both |
| `enable_tokenizer_batch_encode` + `enable_dynamic_batch_tokenizer` | Cannot enable both |

### Conditional Requirements

| Condition | Requirement |
|-----------|-----------|
| `pp_size > 1` | `disable_overlap_schedule` auto-set to True |
| `enable_pdmux = True` | pp_size==1, chunked_prefill_size==-1, disagg==null |
| `disaggregation_mode = "decode"` | `disable_radix_cache` auto-set to True |
| `enable_eplb = True` | `ep_size > 1` required |
| `moe_a2a_backend != "none"` | `ep_size` auto-set to `tp_size` |
| `speculative_algorithm` set | `max_running_requests` auto-set to 48 |
| `enable_dp_lm_head = True` | `enable_dp_attention` must be True |

### Validation Constraints

| Parameter | Constraint |
|-----------|-----------|
| `tokenizer_worker_num` | Must be > 0 |
| `tp_size * pp_size` | Must be divisible by nnodes |
| `base_gpu_id` | Must be >= 0 |
| `gpu_id_step` | Must be >= 1 |
| `schedule_conservativeness` | Must be >= 0 |
| `swa_full_tokens_ratio` | Must be in (0, 1.0] |
| `max_loras_per_batch` | Must be > 0 |
| `page_size` | Must divide chunked_prefill_size |

---

## Summary Statistics

- **Total Parameters**: 200+
- **Functional Domains**: 15 (model, network, quantization, memory, parallelism, logging, API, LoRA, kernels, speculative, MoE, caching, disaggregation, optimization, multimodal)
- **Post-Init Processing Stages**: 17
- **Supported GPU Types**: 10+ with memory-based auto-configuration
- **Quantization Methods**: 25+
- **Attention Backends**: 15+
- **MoE A2A Backends**: 6
