# Streaming Metrics Operations Guide

> **Status**: Canonical live payloads updated for adaptive batching telemetry on 2025-10-15 – see `docs/assets/metrics/2025-10-15-adaptive-update/` for HTTP snapshot, SSE stream, and structured log samples (the 2025-10-10 capture remains for historical comparison).

## Overview

- Engram now ships a streaming-first observability pipeline; the legacy Prometheus exporter is deprecated.
- Operators consume metrics through three surfaces: HTTP `GET /metrics`, gRPC `metrics_snapshot_json`, and structured logs tagged `engram::metrics::stream`.
- Current payloads expose activation pool health (available/in-flight/high-water, hit rate, utilization). Adaptive batching and hardware counters will land in upcoming milestones; placeholders are noted below.

## Streaming Snapshot API

- The HTTP endpoint returns a JSON document with three top-level keys:
  - `schema_version`: semantic version string (e.g., "1.0.0") for backward compatibility tracking. Always check this field before parsing to ensure compatibility with your monitoring tools.
  - `snapshot`: object containing `schema_version` plus four rolling windows (`one_second`, `ten_seconds`, `one_minute`, `five_minutes`). Each window is keyed by metric name (e.g., `activation_pool_available_records`) with per-metric aggregates including `count`, `sum`, `mean`, `min`, `max`, `p50`, `p90`, and `p99`.
  - `export`: exporter health (`exported`, `dropped`, `queue_depth`). Non-zero `dropped` indicates backpressure.
- Reset semantics: `SpreadingMetrics::reset` zeroes the gauges before each spread or engine shutdown. The streaming aggregator retains the last-seen values until they age out of the window, so `/metrics` keeps reporting the most recent pool state even if no new updates have arrived yet.
- Current counter/gauge identifiers exported through streaming metrics:
  - `activation_pool_available_records`, `activation_pool_in_flight_records`, `activation_pool_high_water_mark` (units = records).
  - `activation_pool_total_created`, `activation_pool_total_reused`, `activation_pool_miss_count`, `activation_pool_release_failures` (monotonic counts represented as gauges for rolling deltas).
  - `activation_pool_hit_rate`, `activation_pool_utilization` (bounded 0.0–1.0 floating point values).
  - `adaptive_batch_updates_total`, `adaptive_guardrail_hits_total`, `adaptive_topology_changes_total`, `adaptive_fallback_activations_total` (monotonic counters summarising adaptive controller activity).
  - `adaptive_batch_latency_ewma_ns` (smoothed latency EWMA in nanoseconds captured from recent spreads).
  - `adaptive_batch_hot_size`, `adaptive_batch_warm_size`, `adaptive_batch_cold_size` (latest recommended batch sizes per tier after guardrail adjustments).
  - `adaptive_batch_hot_confidence`, `adaptive_batch_warm_confidence`, `adaptive_batch_cold_confidence` (0.0–1.0 convergence confidence for each tier’s controller).
- Sample payload: `docs/assets/metrics/2025-10-15-adaptive-update/http_metrics.json` (captured from a live CLI session after a `remember` operation). The synthetic generator remains in `docs/assets/metrics/sample_metrics.json` for regression harness parity.
- Long-run references: `docs/assets/metrics/2025-10-12-longrun/start_snapshot.json`, `mid_snapshot.json`, `end_snapshot.json` (synthetic soak using `generate_pool_soak_metrics`; augment with future live soak captures once the batch harness is stable).
- Schema changes: see `docs/metrics-schema-changelog.md` for version history and migration guides.
- Curl example (assumes `engram start` is serving on localhost):

  ```bash
  HTTP_PORT=3928 # replace with the port your daemon is listening on
  curl --silent "http://127.0.0.1:${HTTP_PORT}/metrics" |
    jq '.snapshot.one_second.activation_pool_available_records.mean'
  ```

## Structured Log Consumption

- Logs tagged `engram::metrics::stream` emit the same snapshot payload the HTTP endpoint returns. Tail them with `rg --json` or `jq` to monitor pool behavior without polling HTTP.
- Suggested workflow: `journalctl -u engram --follow | rg 'metrics::stream'` and pipe into `jq '.snapshot.one_second.activation_pool_available_records.mean'` for quick trend checks. Add TODO once final jq script is agreed.
- Rotation cadence: logs adhere to standard CLI retention (7d rolling); increase retention if you rely on log-based dashboards.
- Example entry: `docs/assets/metrics/2025-10-15-adaptive-update/metrics_stream.log` (30-second cadence snapshot from `engram::metrics::stream`).
- Live SSE capture: `docs/assets/metrics/2025-10-10-live-session/activities_stream.log` recorded while encoding a memory.
- Long-run log sample: `docs/assets/metrics/2025-10-12-longrun/stream.log` (synthetic soak). Replace with live CLI output after the 10‑minute soak validation.

## Reset Cadence & Expectations

- Clarify when engine triggers `SpreadingMetrics::reset` (before each spread, on shutdown).
- Add checklist for confirming zeroed gauges after maintenance.
- Reference regression tests (`engram-core/tests/metrics_reset.rs`, `engram-cli/tests/http_api_tests.rs`).

## Schema Versioning & Compatibility

- **Current version**: 1.0.0 (as of 2025-10-13)
- All metrics exports include a `schema_version` field for tracking breaking changes
- **Backward compatibility**: Missing `schema_version` indicates pre-1.0.0 format
- **Version policy**:
  - Major version: breaking changes (field removals, type changes)
  - Minor version: backward-compatible additions (new fields)
  - Patch version: non-functional changes (docs, refactoring)
- **Migration**: see `docs/metrics-schema-changelog.md` for upgrade paths

## Troubleshooting

- Symptoms when reset fails (stale pool utilization, diverging HTTP/log payloads).
- Suggested diagnostics: rerun regression tests, capture streaming snapshot, inspect activation pool stats.
- Planned alerts/dashboard updates (TODO).
- **Schema incompatibility**: If `schema_version` is missing or incompatible, check `docs/metrics-schema-changelog.md` for migration instructions.

## Artifact Staging (Live Capture)

- `docs/assets/metrics/2025-10-15-adaptive-update/http_metrics.json` – canonical `/metrics` HTTP payload (schema version 1.0.0 with adaptive batching fields).
- `docs/assets/metrics/2025-10-15-adaptive-update/metrics_stream.log` – structured log export showing rolling windows pre/post activity.
- `docs/assets/metrics/2025-10-15-adaptive-update/activities_stream.log` – SSE activity sample for storage events (copied from the latest live run).
- `docs/assets/metrics/2025-10-15-adaptive-update/remember_request.json` / `remember_response.json` – request/response pair used to drive the capture.
- Legacy baseline (pre-adaptive capture): `docs/assets/metrics/2025-10-10-live-session/` retained for regression diffs.

## Legacy Consumer Shim

- Existing dashboards that still read `snapshot.one_second.pool_available` can apply `docs/assets/metrics/per_metric_compat_shim.jq` to rebuild the legacy field aliases from the new per-metric map:

  ```bash
  jq -f docs/assets/metrics/per_metric_compat_shim.jq docs/assets/metrics/2025-10-15-adaptive-update/http_metrics.json \
    > docs/assets/metrics/2025-10-15-adaptive-update/http_metrics.legacy.json
  ```

- The shim copies activation pool metrics to their historical aliases (`pool_available`, `pool_in_flight`, etc.) without mutating the per-metric data, buying time while dashboards migrate to the richer schema.
