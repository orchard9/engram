# Spreading Activation Operations Runbook

Spreading activation drives cognitive recall and must remain inside a tight
latency/error envelope. This runbook summarises the probes, dashboards, alert
rules, and manual levers required to keep production healthy.

## Health Probes

| Endpoint | Purpose |
| --- | --- |
| `GET /health/spreading` | Returns the cached result from the synthetic five-node probe. Includes latency, activation mass, status, and hysteresis counters. |
| `GET /api/v1/system/health` | Aggregates all registered probes (including spreading). Responses contain timestamps, consecutive failure counts, and free-form messages per probe. |

Health tasks run every 10 seconds. Degraded status triggers a warning, while
three consecutive probe failures transition the system to `unhealthy` and flip
`/health/spreading` to HTTP 503.

## Circuit Breaker Operations

Spreading recalls are guarded by a three-state breaker (`Closed`, `HalfOpen`,
`Open`). Operators can monitor state transitions via:

- Streaming metric `engram_spreading_breaker_state` (0=Closed, 1=HalfOpen, 2=Open)
- Counter `engram_spreading_breaker_transitions_total`
- Log lines tagged `engram::recall` when the breaker opens, reopens, or closes

When the breaker opens the system automatically falls back to similarity-only
recall. Restore service by eliminating the root latency/failure condition and
waiting for the HalfOpen probes to succeed; no manual reset is required.

## Auto-Tune Controls & Audit Log

Every five minutes the auto-tuner evaluates the latest streaming snapshot. If a
tier exceeds its latency target by more than 10%, the tuner proposes a config
update (batch size, maximum depth, tier timeout) and applies the change when
an improvement is predicted.

- Audit log endpoint: `GET /api/v1/system/spreading/config`
- Payload contains before/after values, tier, timestamp, predicted benefit, and
  the reason captured during evaluation.
- Streaming metrics: `engram_spreading_autotune_changes_total` and
  `engram_spreading_autotune_last_improvement`

Operators should review the audit log when latency budget alerts fire, and roll
back changes by supplying a manual config override through the CLI (future
work).

## Observability Assets

- Grafana dashboard: [`docs/operations/spreading_dashboard.json`](spreading_dashboard.json)
  - Import into Grafana to chart per-tier latency, breaker behaviour, fallback
    rates, pool utilisation, and auto-tune activity.
- Prometheus rules: [`deploy/observability/prometheus/spreading.rules.yaml`](../../deploy/observability/prometheus/spreading.rules.yaml)
  - Recording rules provide smoothed latency and failure ratios.
  - Alerts cover latency SLO breaches, breaker openings, elevated failure rate,
    latency budget regressions, and fallback spikes.

## Chaos & Validation

Use the chaos harness to validate alert thresholds and breaker behaviour before
rollouts:

```bash
cargo run --bin fuzz-spreading-latency -- --duration 120s --latency-spike 15ms --failure-rate 0.15
```

The tool injects synthetic latency/failure patterns into the spreading engine,
records resulting metrics, and exits with a non-zero status if alerts/breakers
fail to trigger.

Recent verification runs:

- 60s spike (`--latency-spike 15ms --failure-rate 0.12`)
  - `engram_spreading_breaker_state` jumped to `2` (open) and
    `SpreadingBreakerOpen` would page after the 2m hold period.
  - Hot tier p95 settled at 15.1 ms with 324 samples, easily breaching the 200 µs
    objective that feeds `SpreadingHotLatencySLOBreach`.
  - Failure volume raised `engram_spreading_latency_budget_violations_total`
    confirming the budget regression alert path.
- 90s spike (`--latency-spike 30ms --failure-rate 0.25`)
  - Breaker opened immediately (state `2`, transitions counter incremented).
  - Auto-tuner pushed an audit entry for the hot tier: batch `64→8`, max depth
    `4→2`, timeout `0.0001s→0.0002s`, predicted improvement `~99.7%`.
  - Hot tier p95 latency reached 30.1 ms across 292 samples, keeping the breaker
    open long enough for alert windows and validating the Prometheus rule deck.

Capture the console output (breaker warnings, latency summaries, audit change)
and attach it to incident tickets or runbooks so reviewers can see which alerts
fired and which auto-tune adjustments occurred.

## Operational Checklist

1. Check `/health/spreading` and `/api/v1/system/health` for probe status.
2. Review Grafana dashboard for latency spikes or breaker churn.
3. Inspect Prometheus alerts and correlate with the auto-tune audit log.
4. If latency remains elevated, capture metrics, disable auto-tune via CLI flag,
   and execute the chaos harness to reproduce before filing an incident report.
