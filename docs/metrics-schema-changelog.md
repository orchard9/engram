# Metrics Schema Changelog

This document tracks changes to the Engram metrics export schema to ensure backward compatibility and smooth upgrades.

## Schema Versioning

All metrics exports include a `schema_version` field following semantic versioning (semver):

- **Major version**: Backward-incompatible changes (field removals, type changes)

- **Minor version**: Backward-compatible additions (new fields, new metrics)

- **Patch version**: Non-functional changes (documentation, internal refactoring)

## Version History

### 1.0.0 (2025-10-13)

**Initial versioned release**

Added schema version tracking to all metrics exports.

**Structure: `AggregatedMetrics`**

```json
{
  "schema_version": "1.0.0",
  "one_second": { "<metric_name>": MetricAggregate, ... },
  "ten_seconds": { "<metric_name>": MetricAggregate, ... },
  "one_minute": { "<metric_name>": MetricAggregate, ... },
  "five_minutes": { "<metric_name>": MetricAggregate, ... }
}

```

**Structure: `MetricAggregate`**

```json
{
  "count": usize,
  "sum": f64,
  "mean": f64,
  "min": f64,
  "max": f64,
  "p50": f64,
  "p90": f64,
  "p99": f64
}

```

**Structure: `ExportStats`**

```json
{
  "exported": u64,
  "dropped": u64,
  "queue_depth": usize
}

```

**Breaking changes from pre-1.0.0**:

- Added `schema_version` field to `AggregatedMetrics`

- Field is optional (`skip_serializing_if = "None"`) for backward compatibility

**Migration guide**:

- Consumers should handle missing `schema_version` field as pre-1.0.0 format

- New consumers should verify `schema_version` is "1.0.0" or compatible

## Future Changes

### Planned for 1.1.0

- Add adaptive batcher metrics to aggregated output

- Add per-tier batch size recommendations

- Add topology fingerprint information

### Planned for 2.0.0 (Breaking)

- Restructure window aggregates to include timestamp ranges

- Add metric-specific aggregations (counter sums, gauge latest values)

- Separate spreading metrics from general metrics

## 2025-10-20 — WAL Observability Metrics (schema 1.3.0)

- Added WAL (Write-Ahead Log) metrics for persistence observability.

- `engram_wal_recovery_successes_total`: Episodes successfully recovered from WAL during startup.

- `engram_wal_recovery_failures_total`: WAL entries that failed deserialization during recovery.

- `engram_wal_recovery_duration_seconds`: Time taken to recover WAL entries on startup (histogram).

- `engram_wal_compaction_runs_total`: WAL compaction operations performed.

- `engram_wal_compaction_bytes_reclaimed`: Bytes saved by WAL compaction.

- Enables monitoring of WAL health, corruption rates, and automatic compaction effectiveness.

- No breaking changes; minor version bump signals new optional WAL metrics.

## 2025-10-19 — Consolidation Quality Metrics (schema 1.2.0)

- Added consolidation quality gauges (`engram_consolidation_novelty_variance`, `engram_consolidation_citation_churn`).

- `engram_consolidation_novelty_variance`: Measures diversity of pattern changes across consolidation run (variance of novelty deltas).

- `engram_consolidation_citation_churn`: Percentage of patterns with citation changes (0-100%), indicates consolidation volatility.

- High variance (>0.1) suggests heterogeneous pattern updates; low variance (<0.01) indicates uniform changes.

- High churn (>50%) indicates volatile consolidation; low churn (<10%) suggests stability.

- No breaking changes; minor version bump signals new optional quality metrics.

## 2025-10-18 — Consolidation Scheduler Metrics (schema 1.1.0)

- Added consolidation counters/gauges (`engram_consolidation_runs_total`, `engram_consolidation_failures_total`, `engram_consolidation_novelty_gauge`, `engram_consolidation_freshness_seconds`, `engram_consolidation_citations_current`).

- Established label contract (`consolidation="scheduler"`) and forward-compatible schema requirements documented in `docs/operations/metrics_streaming.md`.

- No breaking changes; minor version bump signals new optional fields.
