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
