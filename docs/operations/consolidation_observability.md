# Consolidation Observability Playbook

Engram's consolidation system now streams semantic beliefs from the background scheduler. This playbook documents how snapshots are cached, which metrics are persisted, and how operators should react to alert thresholds.

## Scheduler-Backed Snapshots

- `ConsolidationScheduler` writes `ConsolidationSnapshot` instances into the global cache after each run.

- `/api/v1/consolidations*` and `/api/v1/stream/consolidation` serve the cached snapshot, falling back to on-demand generation only while the scheduler warms up.

- Cached snapshots include the contributing episode timestamps: `observed_at`, `stored_at`, and `last_access` for every citation.

- **Health Contract**:
  - Target run cadence: every 300s (scheduler default). If no run completes within 1.5× the configured interval, treat the scheduler as degraded.
  - Failover: API automatically regenerates on-demand snapshots when `consolidation_cache` is empty; operators should still investigate root cause within one hour.
  - Success criteria: each run must publish a snapshot within 5s of completion and refresh metrics gauges (`engram_consolidation_freshness_seconds`, `engram_consolidation_novelty_gauge`).
  - Triggers for incident: three consecutive failures (see `engram_consolidation_failures_total`) or freshness exceeding SLA (default 900s).

## Persisted Metrics

The following metrics are exported through `MetricsRegistry` (Prometheus/SSE):

| Metric | Type | Description |
| --- | --- | --- |
| `engram_consolidation_runs_total` | counter | Successful background consolidations persisted to the cache. |
| `engram_consolidation_failures_total` | counter | Consolidation runs that errored or failed validation. |
| `engram_consolidation_novelty_gauge` | gauge | Latest belief novelty delta emitted by the scheduler. |
| `engram_consolidation_freshness_seconds` | gauge | Age of the cached snapshot (seconds since `generated_at`). |
| `engram_consolidation_citations_current` | gauge | Total citations included in the most recent snapshot. |

Use the streaming metrics endpoint or Prometheus scrape target to verify these values. All metrics share the `consolidation="scheduler"` label to differentiate from on-demand runs.

## Alert Thresholds

| Condition | Threshold | Operator Response |
| --- | --- | --- |
| Failed consolidation streak | `engram_consolidation_failures_total` increases for three consecutive runs | Inspect scheduler logs, failover to on-demand consolidation, file Task 006 incident report. |
| Snapshot staleness | `engram_consolidation_freshness_seconds > 900` | Trigger manual scheduler run, verify background worker health. |
| Novelty stagnation | `engram_consolidation_novelty_gauge < 0.01` for five runs | Review consolidation inputs, ensure pattern detection Task 002 thresholds are still valid. |
| Health contract breach | No scheduler snapshot within 450s (1.5× interval) | Initiate failover run, capture diagnostics, escalate to reliability lead. |

## Belief Update Log

- Consolidation runs append JSONL entries to the belief update log under `data/consolidation/alerts/`.

- Each entry records the semantic pattern ID, confidence delta, citation churn, novelty score, and `generated_at` timestamp.

- Operators can tail the log for near-real-time diagnostics or load it into the observability stack for historical analysis.

## Runbook

1. **Verify**: Check `engram_consolidation_runs_total` against `failures_total` to confirm the scheduler is making progress.

2. **Inspect**: Tail the belief update log for spikes in `confidence_delta` or dropped citations.

3. **Respond**: Use the alert thresholds above to choose between manual reruns, parameter adjustments, or incident escalation.

4. **Document**: Update the relevant milestone task file with findings and create follow-up tasks when remediation requires code changes.

5. **Failover**: If the cache is empty during an outage, trigger `MemoryStore::consolidation_snapshot` manually to keep API responses live, then remediate scheduler state before returning to normal cadence.

6. **Dashboard Review**: Consult the consolidation dashboard (see `consolidation_dashboard.md`) to confirm SLA visualizations and annotations remain accurate after each incident.

## Soak Harness & Baseline Artifacts

- Run the soak harness to capture long-form metrics and belief deltas:

  ```bash
  cargo run --bin consolidation-soak \
    --duration-secs 3600 \
    --scheduler-interval-secs 60 \
    --sample-interval-secs 60 \
    --output-dir ./docs/assets/consolidation/baseline
  ```

- The harness seeds episodic traffic, drives the background scheduler, and emits:
  - `docs/assets/consolidation/baseline/metrics.jsonl` — rolling `AggregatedMetrics` snapshots.
  - `docs/assets/consolidation/baseline/snapshots.jsonl` — belief summaries (`pattern_count`, replay stats).
  - `docs/assets/consolidation/baseline/belief_updates.jsonl` — persisted deltas for alerting pipelines.

- A 30-second reference capture (scheduler interval 5s) is checked in as a smoke baseline; regenerate with a full 1h run before publishing dashboards or tuning SLAs.
