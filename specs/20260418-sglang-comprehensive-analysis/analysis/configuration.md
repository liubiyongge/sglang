# Configuration & Environment System

## Overview

SGLang uses a centralized, type-safe environment variable system for runtime configuration, supplemented by CLI arguments (`ServerArgs`) for server-level settings. The system manages 150+ configuration flags with type validation, deprecation warnings, and context-manager overrides.

**Key File**: `python/sglang/srt/environ.py`

---

## Environment Variable System

### Architecture

```python
# srt/environ.py

class EnvField:
    """Base descriptor for typed environment variables."""
    def __init__(self, name: str, default, doc: str = ""):
        self.name = name
        self.default = default
    
    def get(self):
        raw = os.environ.get(self.name)
        if raw is None:
            return self.default
        return self._parse(raw)
    
    def set(self, value):
        os.environ[self.name] = str(value)

class EnvBool(EnvField): ...   # Parses true/1/yes vs false/0/no
class EnvInt(EnvField): ...    # Integer with validation
class EnvFloat(EnvField): ...  # Float parsing
class EnvStr(EnvField): ...    # String passthrough
class EnvTuple(EnvField): ...  # Comma-separated tuple
```

### Envs Singleton

All configuration flags are defined on the `Envs` class:

```python
class Envs:
    # Scheduler tuning
    SGLANG_NEW_TOKEN_RATIO = EnvFloat("SGLANG_NEW_TOKEN_RATIO", 0.7)
    SGLANG_RECV_INTERVAL = EnvInt("SGLANG_RECV_INTERVAL", 100)
    SGLANG_CHECK_MEM_INTERVAL = EnvInt("SGLANG_CHECK_MEM_INTERVAL", 4)
    
    # Logging
    SGLANG_LOG_GC = EnvBool("SGLANG_LOG_GC", False)
    SGLANG_LOG_MS = EnvInt("SGLANG_LOG_MS", 0)
    SGLANG_LOG_REQUEST_EXCEEDED_MS = EnvInt("SGLANG_LOG_REQUEST_EXCEEDED_MS", 0)
    
    # Profiling
    SGLANG_PROFILE_V2 = EnvBool("SGLANG_PROFILE_V2", False)
    SGLANG_TORCH_PROFILER_DIR = EnvStr("SGLANG_TORCH_PROFILER_DIR", "/tmp/sglang_profile")
    SGLANG_ENABLE_METRICS_DEVICE_TIMER = EnvBool("SGLANG_ENABLE_METRICS_DEVICE_TIMER", False)
    
    # Metrics
    SGLANG_OTLP_EXPORTER_ENDPOINT = EnvStr("SGLANG_OTLP_EXPORTER_ENDPOINT", "")
    
    # Quantization
    SGLANG_INT4_WEIGHT_ONLY_QUANTIZATION = EnvBool(...)
    
    # ... 150+ more flags
    
envs = Envs()  # Singleton instance
```

### Usage Pattern

```python
from sglang.srt.environ import envs

# Read
interval = envs.SGLANG_RECV_INTERVAL.get()

# Override (context manager)
with envs.SGLANG_LOG_GC.override(True):
    # GC logging enabled within this block
    ...

# Check if explicitly set
if envs.SGLANG_PROFILE_V2.is_set():
    ...
```

---

## Configuration Categories

### Scheduler Tuning

| Variable | Default | Purpose |
|----------|---------|---------|
| `SGLANG_NEW_TOKEN_RATIO` | 0.7 | Reserve ratio for new tokens in KV cache |
| `SGLANG_RECV_INTERVAL` | 100 | Scheduler recv polling interval (iterations) |
| `SGLANG_CHECK_MEM_INTERVAL` | 4 | Memory check frequency |
| `SGLANG_MAX_PREFILL_TOKENS` | 16384 | Max tokens per prefill batch |
| `SGLANG_CHUNKED_PREFILL_SIZE` | 8192 | Max chunk size for chunked prefill |

### Memory & Cache

| Variable | Default | Purpose |
|----------|---------|---------|
| `SGLANG_MEM_FRACTION_STATIC` | 0.88 | Static KV cache memory fraction |
| `SGLANG_KV_CACHE_DTYPE` | `"auto"` | Override KV cache dtype |
| `SGLANG_RADIX_CACHE_EVICT_POLICY` | `"lru"` | Cache eviction strategy |

### Logging & Debug

| Variable | Default | Purpose |
|----------|---------|---------|
| `SGLANG_LOG_GC` | `false` | Log garbage collection |
| `SGLANG_LOG_MS` | 0 | Log operations exceeding N ms |
| `SGLANG_LOG_REQUEST_EXCEEDED_MS` | 0 | Log slow requests |
| `SGLANG_VERBOSE` | 0 | Verbosity level |

### Device & Platform

| Variable | Default | Purpose |
|----------|---------|---------|
| `SGLANG_USE_MODELSCOPE` | `false` | Use ModelScope instead of HuggingFace |
| `SGLANG_AMD_HIPBLASLT_TUNING` | `false` | Enable AMD hipBLASLt auto-tuning |
| `SGLANG_NPU_BACKEND` | `""` | NPU backend selection |

### Grammar & Constraints

| Variable | Default | Purpose |
|----------|---------|---------|
| `SGLANG_GRAMMAR_POLL_INTERVAL` | 0.01 | Grammar compilation poll interval (s) |
| `SGLANG_GRAMMAR_MAX_POLL_ITERATIONS` | 100 | Max poll iterations |

---

## Deprecation Management

```python
def _convert_SGL_to_SGLANG():
    """Convert legacy SGL_* env vars to SGLANG_* with deprecation warning."""
    for key in list(os.environ):
        if key.startswith("SGL_") and not key.startswith("SGLANG_"):
            new_key = "SGLANG_" + key[4:]
            warnings.warn(f"{key} is deprecated, use {new_key}")
            os.environ[new_key] = os.environ[key]

def _warn_deprecated_env_to_cli_flag():
    """Warn when env vars that have been moved to CLI flags are still set."""
    # e.g., SGLANG_TP_SIZE → --tp-size CLI arg
```

---

## Relationship to ServerArgs

Environment variables provide **runtime tuning** that applies globally.  
`ServerArgs` provides **instance-level configuration** that varies per server launch.

```
ServerArgs (CLI):  --model-path, --tp-size, --port, --max-running-requests
Envs (runtime):    SGLANG_NEW_TOKEN_RATIO, SGLANG_LOG_MS, SGLANG_MEM_FRACTION_STATIC
```

ServerArgs are passed explicitly through the call chain. Envs are accessed globally via `envs.VARIABLE.get()`.

---

## Testing Configuration

**File**: `python/sglang/test/test_utils.py`

Test infrastructure provides:
- Model constants for different test scenarios
- `popen_launch_server()` for launching test servers
- Default timeouts and URLs
- CI registration decorators: `@register_cuda_ci()`, `@register_amd_ci()`

```python
# test/ci/ci_register.py
@register_cuda_ci(est_time=60, suite="unit")
def test_radix_cache():
    ...

@register_npu_ci(est_time=120, suite="integration", nightly=True)
def test_npu_inference():
    ...
```

---

## Source References

- Environment System: `python/sglang/srt/environ.py`
- Server Args: `python/sglang/srt/server_args.py`
- Test Utils: `python/sglang/test/test_utils.py`
- CI Registration: `python/sglang/test/ci/ci_register.py`
- Logging Config: `python/sglang/srt/managers/configure_logging.py`
