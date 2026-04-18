# T009: Scheduling Policies Analysis

**Source**: `python/sglang/srt/managers/schedule_policy.py`
**Cross-references**: [scheduler-event-loop.md](scheduler-event-loop.md), [batch-formation.md](batch-formation.md)

---

## Overview

SGLang implements six scheduling policies organized into two categories: **cache-aware** (exploit the radix tree structure for KV reuse) and **cache-agnostic** (general-purpose ordering). The policy determines the order in which waiting requests are considered for batch formation.

---

## Policy Taxonomy

### Cache-Aware Policies

#### LPM (Longest Prefix Match)

- **Algorithm**: Sort waiting queue by `len(prefix_indices)` descending
- **Mechanism**: Match each request's tokens against the radix tree cache, schedule those with longest cache hits first
- **Optimization**: Falls back to FCFS when `len(waiting_queue) > 128` (expensive tree traversal)
- **Best for**: High request diversity with shared prefixes (multi-tenant, common system prompts)

```python
waiting_queue.sort(
    key=lambda r: (
        -len(r.prefix_indices) if r.rid not in temporary_deprioritized
        else float('inf')
    )
)
```

#### DFS-Weight (Depth-First Search Weighting)

- **Algorithm**:
  1. Cluster requests by their `last_node` in the radix tree
  2. Compute bottom-up weights: `weight[node] = requests_at_node + sum(weight[children])`
  3. DFS traversal prioritizing heaviest branches
- **Mechanism**: Schedules requests that share cache paths together, maintaining cache coherency
- **Best for**: Complex prompt hierarchies (multi-turn conversations, branching prompts)

```
Tree traversal order:
    root (weight=8)
    ├─ "system_A" (weight=5) → scheduled first (heavier)
    │   ├─ "system_A/q1" (weight=3)
    │   └─ "system_A/q2" (weight=2)
    └─ "system_B" (weight=3) → scheduled second
```

### Cache-Agnostic Policies

#### FCFS (First Come First Serve)

- **Algorithm**: Sort by `time_stats.wait_queue_entry_time` (arrival order)
- **With priority**: Sort by `(priority * priority_sign, arrival_time)`
- **Best for**: Strict SLA/fairness requirements, deterministic ordering

#### LOF (Longest Output First)

- **Algorithm**: Sort by `max_new_tokens` descending
- **With priority**: Sort by `(priority * priority_sign, -max_new_tokens)`
- **Best for**: Mixed generation length workloads (amortize long-sequence prefill)

#### RANDOM

- **Algorithm**: Shuffle waiting queue randomly
- **Best for**: Testing, breaking pathological patterns

#### ROUTING_KEY

- **Algorithm**: Count routing key frequency in running batch, prioritize matching keys
- **Sort key**: `(0 if key_in_running else 1, -frequency, routing_key)`
- **Best for**: Multi-model/multi-expert serving, load-aware scheduling

---

## Policy Selection and Fallback

```python
class SchedulePolicy:
    def _determine_active_policy(self, waiting_queue):
        # Expensive prefix matching disabled for large queues
        if self.policy == CacheAwarePolicy.LPM and len(waiting_queue) > 128:
            return CacheAgnosticPolicy.FCFS
        return self.policy

    def _validate_and_adjust_policy(self, policy, tree_cache):
        # If tree cache disabled, force FCFS
        if getattr(tree_cache, "disable", True):
            return CacheAgnosticPolicy.FCFS
```

---

## In-Batch Prefix Caching

### Mechanism

Prevents redundant computation within a single prefill batch by detecting shared prefixes among waiting requests.

### Algorithm

1. **Phase 1**: Match each request against persistent `tree_cache`
2. **Phase 2**: For requests with short cache hits (`< IN_BATCH_PREFIX_CACHING_CHECK_THRESHOLD`), check against `waiting_queue_radix_tree`
3. **Deprioritization**: If in-batch match `>= IN_BATCH_PREFIX_CACHING_DEPRIORITIZE_THRESHOLD`, move to end of queue

### Configuration

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `IN_BATCH_PREFIX_CACHING_CHECK_THRESHOLD` | 32 | When to check for in-batch sharing |
| `IN_BATCH_PREFIX_CACHING_DEPRIORITIZE_THRESHOLD` | 32 | When to deprioritize |

### Rationale

If multiple requests share a prefix not yet in cache, schedule only one first. After its prefill completes and the KV is cached, subsequent requests benefit from cache hits.

---

## Priority Scheduling

### Priority Sign Convention

```python
self.priority_sign = 1 if schedule_low_priority_values_first else -1
```

- `priority_sign = 1`: Lower values scheduled first (VIP=0, standard=1)
- `priority_sign = -1`: Higher values scheduled first (urgent=10, normal=1)

### Priority Preemption

High-priority waiting requests can preempt running low-priority requests when:
1. `priority_diff > priority_scheduling_preemption_threshold`
2. Freed tokens sufficient for new request
3. At least 1 request remains in batch

---

## Policy Selection Guide

| Workload Type | Recommended Policy | Rationale |
|---------------|-------------------|-----------|
| High-diversity prompts | LPM | Maximize cache reuse |
| Multi-turn conversations | DFS-Weight | Exploit conversation trees |
| Strict SLA/fairness | FCFS + Priority | Deterministic ordering |
| Mixed generation lengths | LOF | Amortize long sequences |
| Multi-expert/model routing | ROUTING_KEY | Improve batching efficiency |
| Testing/benchmarking | RANDOM | Eliminate bias |
