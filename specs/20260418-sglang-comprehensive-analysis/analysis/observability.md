# Observability & Metrics

## Overview

SGLang provides comprehensive observability through Prometheus metrics, structured logging, and request-level tracing. The system tracks throughput, latency, memory utilization, cache efficiency, and per-request lifecycle timing.

**Key Files**:
- `python/sglang/srt/metrics/collector.py` — Core metrics collection (Prometheus)
- `python/sglang/srt/managers/scheduler_metrics_mixin.py` — Scheduler metric integration
- `python/sglang/srt/managers/request_metrics_exporter.py` — Per-request export

---

## Prometheus Metrics

### SchedulerMetricsCollector

**File**: `srt/metrics/collector.py`

Central metrics aggregation point exposed on `/metrics` endpoint:

| Metric | Type | Description |
|--------|------|-------------|
| `sglang_num_running_reqs` | Gauge | Currently executing requests |
| `sglang_num_waiting_reqs` | Gauge | Requests in waiting queue |
| `sglang_token_usage` | Gauge | KV cache token utilization |
| `sglang_gen_throughput` | Gauge | Generation tokens/second |
| `sglang_prompt_throughput` | Gauge | Input tokens/second |
| `sglang_time_to_first_token_seconds` | Histogram | TTFT distribution |
| `sglang_time_per_output_token_seconds` | Histogram | TPOT distribution |
| `sglang_e2e_request_latency_seconds` | Histogram | End-to-end latency |
| `sglang_num_queue_time_seconds` | Histogram | Queue wait time |
| `sglang_cache_hit_rate` | Gauge | Radix cache hit rate |
| `sglang_spec_accept_rate` | Gauge | Speculative decoding acceptance |
| `sglang_lora_pool_utilization` | Gauge | LoRA memory pool usage |

### Histogram Configuration

Histogram buckets are configurable via environment variables:
- `SGLANG_METRICS_TTFT_BUCKETS`
- `SGLANG_METRICS_TPOT_BUCKETS`
- `SGLANG_METRICS_E2E_BUCKETS`

---

## Scheduler Metrics Mixin

**File**: `srt/managers/scheduler_metrics_mixin.py`

Integrates metrics collection into the scheduler event loop:

```python
class SchedulerMetricsMixin:
    def log_prefill_stats(self, batch):
        """Log prefill batch statistics."""
        # Input tokens processed, cache hits, new token ratio
        
    def log_decode_stats(self, batch):
        """Log decode batch statistics."""
        # Batch size, tokens generated, throughput
    
    def publish_kv_cache_events(self):
        """Emit cache utilization metrics."""
        # Token pool usage, free blocks, eviction counts
```

### DeviceTimer Integration

GPU execution timing for accurate per-phase metrics:
```python
class DeviceTimer:
    """CUDA event-based GPU timing."""
    def start(self): ...  # Record start event
    def stop(self): ...   # Record end event, compute elapsed
```

---

## Request-Level Metrics Export

**File**: `srt/managers/request_metrics_exporter.py`

Per-request telemetry with pluggable backends:

```python
class RequestMetricsExporter(ABC):
    @abstractmethod
    def export(self, request_metrics: dict): ...

class FileRequestMetricsExporter(RequestMetricsExporter):
    """Writes hourly JSON log files with request metrics."""
    def export(self, metrics):
        # Appends to /path/to/logs/YYYY-MM-DD-HH.jsonl
```

Exported fields per request:
- `rid`: Request ID
- `ttft`: Time to first token
- `tpot`: Time per output token
- `total_latency`: End-to-end latency
- `input_tokens`, `output_tokens`: Token counts
- `queue_time`, `prefill_time`, `decode_time`: Phase breakdown
- `cache_hit_tokens`: Radix cache hits

---

## Rust Gateway Metrics

**File**: `sgl-model-gateway/src/observability/metrics.rs`

High-performance Prometheus metrics for the load-balancing gateway:
- Lock-free string interning (DashMap) for label efficiency
- HTTP method and status code constants
- Per-model, per-worker routing metrics
- Request/response latency histograms

---

## Logging System

### Structured Logging

- Configurable log levels per component
- JSON format option for log aggregation
- Key environment variables:
  - `SGLANG_LOG_GC`: Log garbage collection events
  - `SGLANG_LOG_MS`: Log slow operations exceeding threshold
  - `SGLANG_LOG_REQUEST_EXCEEDED_MS`: Log requests exceeding latency

### Metrics-Driven Alerts

The metrics system enables alerting on:
- KV cache approaching capacity (`token_usage > 0.95`)
- Queue buildup (`num_waiting_reqs > threshold`)
- Throughput degradation
- Speculative acceptance rate drops

---

## Source References

- Collector: `python/sglang/srt/metrics/collector.py`
- Scheduler Mixin: `python/sglang/srt/managers/scheduler_metrics_mixin.py`
- Request Export: `python/sglang/srt/managers/request_metrics_exporter.py`
- Gateway Metrics: `sgl-model-gateway/src/observability/metrics.rs`
