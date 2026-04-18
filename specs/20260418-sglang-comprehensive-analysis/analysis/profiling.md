# Profiling Infrastructure

## Overview

SGLang provides multi-level profiling capabilities ranging from PyTorch-level tracing to custom stage-based timing. The system supports both ad-hoc HTTP-triggered profiling and always-on DeviceTimer metrics.

**Key Files**:
- `python/sglang/srt/managers/scheduler_profiler_mixin.py` — Server-side profiler orchestration
- `python/sglang/srt/utils/profile_utils.py` — ProfileManager and DeviceTimer
- `python/sglang/profiler.py` — HTTP client for triggering profiles

---

## Profiling Modes

### Profile V1 (Legacy)

Torch profiler triggered via HTTP API:

```bash
# Trigger from client
curl -X POST http://localhost:30000/start_profile \
    -d '{"num_steps": 10, "output_dir": "/tmp/profiles"}'
```

Server-side (`scheduler_profiler_mixin.py`):
```python
class SchedulerProfilerMixin:
    def handle_start_profile(self, req):
        self.profiler = torch.profiler.profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            with_stack=True,
            profile_memory=True,
        )
        self.profiler.start()
        self.profile_steps_remaining = req.num_steps
    
    def handle_stop_profile(self):
        self.profiler.stop()
        self.profiler.export_chrome_trace(output_path)
```

### Profile V2 (Modern)

**File**: `srt/utils/profile_utils.py`

State-machine based profiler with better distributed support:

```python
class ProfileManager:
    """Modern profiling state machine."""
    
    States: IDLE → WARMUP → RECORDING → EXPORTING → IDLE
    
    def start(self, config: ProfileConfig):
        """Start profiling with warmup."""
        self.state = ProfileState.WARMUP
        self.warmup_steps = config.warmup_steps
    
    def step(self):
        """Called each scheduler iteration."""
        if self.state == WARMUP:
            if self.warmup_counter >= self.warmup_steps:
                self.state = RECORDING
                self._start_recording()
        elif self.state == RECORDING:
            if self.record_counter >= self.record_steps:
                self._stop_and_export()
```

Features:
- Warmup phase (avoids cold-start artifacts)
- Per-rank trace export
- Cross-rank profile merging (`ProfileMerger`)
- Chrome trace format output
- Stage-based separation (prefill vs decode)

---

## DeviceTimer

**File**: `srt/utils/profile_utils.py`

GPU-accurate timing using CUDA events:

```python
class DeviceTimer:
    def __init__(self, name: str):
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)
    
    def start(self):
        self.start_event.record()
    
    def stop(self) -> float:
        self.end_event.record()
        torch.cuda.synchronize()
        return self.start_event.elapsed_time(self.end_event)  # ms
```

Used in metrics when `SGLANG_ENABLE_METRICS_DEVICE_TIMER=1` for accurate GPU execution time.

---

## Profiling Activities

The system supports multiple activity types:

| Activity | Description | Platform |
|----------|-------------|----------|
| `CPU` | Python/C++ call stacks | All |
| `CUDA` | GPU kernel execution | NVIDIA |
| `MEM` | Memory allocation tracking | All |
| `RPD` | ROCm Profiling Data | AMD |
| `CUDA_PROFILER` | NSight integration | NVIDIA |

---

## CUDA Memory Profiling

```python
# Track memory history for leak detection
torch.cuda.memory._record_memory_history(
    max_entries=100000,
    context="all"
)
# ... run workload ...
torch.cuda.memory._dump_snapshot(output_path)
```

---

## HTTP Profiling Client

**File**: `python/sglang/profiler.py`

```python
def run_profile(
    server_url: str,
    num_steps: int = 10,
    output_dir: str = None,
    activities: List[str] = ["CPU", "CUDA"],
):
    """Trigger profiling via server HTTP API."""
    requests.post(f"{server_url}/start_profile", json={
        "num_steps": num_steps,
        "activities": activities,
        "output_dir": output_dir,
    })
    # Wait for completion
    while True:
        status = requests.get(f"{server_url}/profile_status")
        if status.json()["state"] == "done":
            break
    # Download merged trace
    requests.post(f"{server_url}/stop_profile")
```

---

## Distributed Profile Merging

For multi-GPU setups, profiles from each rank are merged:

```python
class ProfileMerger:
    def merge(self, per_rank_traces: List[str]) -> str:
        """Merge Chrome traces from multiple ranks into unified view."""
        # Aligns timestamps across ranks
        # Preserves rank information in thread names
        # Outputs single trace viewable in chrome://tracing
```

---

## Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `SGLANG_PROFILE_V2` | `false` | Enable Profile V2 state machine |
| `SGLANG_TORCH_PROFILER_DIR` | `/tmp/sglang_profile` | Output directory |
| `SGLANG_ENABLE_METRICS_DEVICE_TIMER` | `false` | GPU-accurate timing in metrics |
| `SGLANG_PROFILE_WARMUP_STEPS` | `2` | Warmup iterations before recording |

---

## Source References

- Profiler Mixin: `srt/managers/scheduler_profiler_mixin.py`
- Profile Utils: `srt/utils/profile_utils.py`
- HTTP Client: `python/sglang/profiler.py`
