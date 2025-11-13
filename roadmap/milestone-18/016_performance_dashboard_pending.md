# Task 016: Performance Dashboard

**Status**: Pending
**Estimated Duration**: 4-5 days
**Priority**: High - Operational visibility

## Objective

Build Grafana dashboard for real-time performance metrics visualization with historical baseline comparison, automated anomaly detection, and <5min time-to-root-cause for performance issues.

## Dashboard Panels

### 1. Latency Overview
- **P50/P95/P99 time series** (last 24h, 7d, 30d)
- **Baseline comparison**: Show M17 baseline as reference line
- **Anomaly highlights**: Red background when >5% regression

### 2. Throughput Metrics
- **Ops/sec by operation type** (store, recall, search, complete)
- **Success rate** (100% - error_rate)
- **Queue depth** (pending operations)

### 3. Resource Utilization
- **CPU usage** (user/system/iowait breakdown)
- **Memory usage** (RSS, heap allocated, cache size)
- **Disk I/O** (read/write MB/s, queue depth)

### 4. Competitive Positioning
- **Engram vs Neo4j** (P99 latency comparison)
- **Engram vs Qdrant** (throughput comparison)
- **Hybrid advantage** (operations Neo4j/Qdrant can't do)

### 5. Anomaly Detection
- **Z-score alerts**: >3σ deviation from baseline
- **Trend alerts**: >5% regression over 24h
- **Error spikes**: >2x normal error rate

## Implementation

```json
// grafana/dashboards/performance_overview.json
{
  "dashboard": {
    "title": "Engram Performance Overview",
    "panels": [
      {
        "title": "P99 Latency vs Baseline",
        "targets": [
          {
            "expr": "histogram_quantile(0.99, engram_operation_duration_seconds_bucket)",
            "legendFormat": "Current P99"
          },
          {
            "expr": "0.000458",  // M17 baseline
            "legendFormat": "M17 Baseline"
          }
        ],
        "thresholds": [
          { "value": 0.000481, "color": "yellow" },  // +5%
          { "value": 0.000550, "color": "red" }      // +20%
        ]
      }
      // ... more panels
    ]
  }
}
```

## Anomaly Detection Rules

```yaml
# prometheus/rules/performance_anomalies.yml
groups:
  - name: performance_regressions
    interval: 1m
    rules:
      - alert: P99LatencyRegression
        expr: |
          (
            histogram_quantile(0.99, rate(engram_operation_duration_seconds_bucket[5m]))
            / 0.000458  # M17 baseline
          ) > 1.05
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "P99 latency >5% above baseline"
          description: "Current P99: {{ $value }}ms, Baseline: 0.458ms"

      - alert: ThroughputDegradation
        expr: |
          rate(engram_operations_total[5m]) < 800
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Throughput below 800 ops/s"
```

## Success Criteria

- **Real-Time Updates**: <10s lag from metric to dashboard
- **Anomaly Detection**: Detect >5% regression within 5min
- **Root Cause**: <5min from alert to identifying bottleneck
- **Historical Context**: 90 days of retention for trend analysis

## Files

- `grafana/dashboards/performance_overview.json` (800 lines)
- `prometheus/rules/performance_anomalies.yml` (150 lines)
- `docs/operations/performance_dashboard.md` (250 lines)
- `scripts/setup_performance_dashboard.sh` (120 lines)
