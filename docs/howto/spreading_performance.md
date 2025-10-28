---
title: Spreading Performance Guide
outline: deep
---

# Spreading Performance Guide

Translate Task 010 benchmark findings and Task 012 monitoring into concrete operating modes. Each preset references fields in `ParallelSpreadingConfig` (`engram-core/src/activation/mod.rs`) and associated metrics.

## Preset Overview

| Mode | Goal | Key Settings | Expected P95 | Notes |
| --- | --- | --- | --- | --- |
| Low Latency | Sub-5 ms response | `max_depth = 2`, `threshold = 0.25`, `batch_size = 32`, `deterministic = false` | 3–4 ms | Trades recall breadth for responsiveness; monitor fallback rate |
| Balanced | Default production | `max_depth = 3`, `threshold = 0.12`, `batch_size = 48`, `deterministic = false`, `priority_hot_tier = true` | 6–8 ms | Keeps quality while respecting budgets |
| High Recall | Maximize coverage | `max_depth = 4`, `threshold = 0.05`, `batch_size = 64`, `deterministic = true`, `phase_sync_interval = 15ms` | 10–12 ms | Requires higher pool utilization; ensure circuit breaker guardrails |

Apply presets through the runtime endpoint or configuration file:

```bash
curl -X POST http://localhost:7432/api/v1/system/runtime \
  -H "Content-Type: application/json" \
  -d '{
        "spreading": {
          "max_depth": 3,
          "threshold": 0.12,
          "batch_size": 48,
          "priority_hot_tier": true
        }
      }'

```

> Runtime overrides are tracked in Task 014. Until the HTTP endpoint lands, apply equivalent settings via deployment configuration or by rebuilding the CLI with updated `ParallelSpreadingConfig` defaults.

## Metrics to Watch

### Latency

- `engram_spreading_latency_hot_seconds_bucket` (and warm/cold) – histogram buckets by storage tier

- `SpreadingMetrics::average_latency` – internal EWMA (available via the metrics snapshot API)

- `SpreadingMetrics::latency_budget_violations` – incremented when spreads exceed time budgets

### Pool Utilization

`SpreadingMetrics::record_pool_snapshot` exports gauges:

| Metric | Description | Healthy Range |
| --- | --- | --- |
| `engram_spreading_pool_utilization` | Ratio of in-flight records to total capacity | 0.35–0.7 |
| `engram_spreading_pool_hit_rate` | Reuse percentage from the pool | >0.85 |
| `activation_pool_miss_count` | Fresh allocations per sampling window | <5 |

### Adaptive Batcher

Task 010’s adaptive batcher exposes counters to confirm convergence:

- `adaptive_batch_updates` – increases when EWMA recomputes sizes

- `adaptive_guardrail_hits` – spikes indicate oscillation; widen hysteresis if frequent

- `adaptive_latency_ewma_ns` – trending upward signals the need to reduce `batch_size`

## Troubleshooting Playbook

| Symptom | Likely Cause | Action |
| --- | --- | --- |
| `latency_budget_violations` climbing | Hop depth too high or downstream system back pressure | Drop `max_depth` or raise `threshold`; verify external dependencies |
| `activation_pool_hit_rate` < 0.7 | Pool under-provisioned | Increase `pool_chunk_size` or enable adaptive batch guardrails |
| Circuit breaker opening | Sustained high latency or spreading errors | Switch to `RecallMode::Hybrid`, investigate `engram_spreading_breaker_state` |
| GPU fallbacks | Device busy or threshold too low | Increase `gpu_threshold` or disable GPU until Task 011 GPU support lands |

## Dashboards

Import the Grafana JSON from Task 012 (`docs/operations/spreading_dashboard.json`) and focus on:

- Latency heatmap per tier

- Pool utilization overlayed with `batch_size`

- Circuit breaker state panel paired with fallback counts

## Rollout Strategy

1. Start in Hybrid mode with the Low Latency preset.

2. Monitor fallback ratios (`RecallMetrics::fallback_rate`) for 24 hours.

3. Enable the Balanced preset once latency stabilizes.

4. Use High Recall mode for dedicated analytics tenants or offline backfills.

5. Document every preset change in `docs/changelog.md` and notify operators via release notes.

## Further Reading

- [Spreading Monitoring How-To](spreading_monitoring.md)

- [Cognitive Spreading Explanation](../explanation/cognitive_spreading.md)

- [Spreading API Reference](../reference/spreading_api.md)
