---
title: Spreading Monitoring Runbook
outline: deep
---

# Spreading Monitoring Runbook

Ensure activation spreading stays healthy in production using Task 012’s instrumentation.

## Metrics Endpoints

- `/metrics` – Prometheus scrape target with spreading histograms and gauges
- `/api/v1/monitoring/events` – Server-sent events stream including activation traces when `include_spreading=true`
- `/api/v1/system/health` – Health checker that exercises a five-node spread for liveness

## Essential Prometheus Series

| Metric | Type | Purpose |
| --- | --- | --- |
| `engram_spreading_latency_hot_seconds` / `_warm_` / `_cold_` | Histogram | Tier-specific latency distribution |
| `engram_spreading_activations_total` | Gauge | Volume of spreading operations per interval |
| `engram_spreading_pool_utilization` | Gauge | Watch for saturation of the activation pool |
| `engram_spreading_autotune_changes_total` | Counter | Track auto-tuner adjustments |
| `engram_spreading_gpu_launch_total` / `_fallback_total` | Counter | GPU engagement vs. fallback frequency |
| `engram_spreading_breaker_state` | Gauge | Circuit breaker position (0 = closed, 1 = open) |

## Alerts

Ship the sample rules in `deploy/observability/prometheus/spreading.rules.yaml` and tune thresholds:

- **High Latency:** `histogram_quantile(0.95, sum(rate(engram_spreading_latency_hot_seconds_bucket[5m])) by (le)) > 0.010`
- **Circuit Breaker Open:** `engram_spreading_breaker_state == 1`
- **Pool Exhausted:** `engram_spreading_pool_utilization > 0.8` for 10 minutes

## SSE Debug Stream

Use SSE traces for real-time spreads:

```bash
curl "http://localhost:7432/api/v1/monitoring/events?event_types=activation,spreading&include_causality=true" \
  --no-buffer | jq '.spreading | select(. != null)'
```

Each payload contains:

- `spread_count`
- Top target nodes with activation/confidence
- Optional deterministic trace (when enabled)

## Health Checker

`SpreadingHealthChecker` injects a deterministic spread across a cycle graph and fails if:

- Activation mass is zero
- Runtime exceeds 50 ms
- Cycle detection fails to dampen activation

Monitor `/api/v1/system/health` and include it in readiness probes.

## Integrating Visualizations

1. Enable deterministic traces (`trace_activation_flow = true`).
2. Capture a trace from SSE or the recall API.
3. Run `cargo run -p engram-cli --example spreading_visualizer -- --input-trace trace.json --output spread.dot`.
4. Render DOT with GraphViz (`dot -Tpng spread.dot -o docs/assets/spreading_example.png`).

## Incident Checklist

1. Confirm whether the circuit breaker is open.
2. Compare `engram_spreading_latency_hot_seconds` (and warm/cold) with service-level objectives.
3. Inspect recent auto-tuner changes for regressions.
4. Dump a deterministic trace and visualize it to pinpoint bottleneck tiers.
5. If metrics continue to degrade, toggle `spreading_api_beta` off to fall back to similarity-only recall, log the incident, and escalate.

## References

- [Performance Guide](spreading_performance.md)
- [Cognitive Spreading Explanation](../explanation/cognitive_spreading.md)
- [Spreading API Reference](../reference/spreading_api.md)
