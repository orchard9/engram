# Completion Metrics & Observability: Research Foundations

## Observability vs Monitoring

### Honeycomb's Three Pillars (2018)
**Monitoring:** Known unknowns. Predefined metrics, alerts for expected failures.
**Observability:** Unknown unknowns. Arbitrary queries to understand emergent behavior.

**Application to Completion:**
- Monitoring: Track latency P95, error rate, completion rate
- Observability: Why did this specific completion take 50ms? Which patterns contributed?

Task 008 implements both: Prometheus metrics (monitoring) + structured logs (observability).

## The Four Golden Signals (Google SRE)

### Latency
Time to serve a request. Track distribution (P50, P95, P99), not just average.

**Completion Latency Breakdown:**
- Pattern retrieval: ~3ms
- CA3 convergence: ~14ms
- Evidence integration: ~1ms
- Source attribution: ~500μs
- Confidence computation: ~200μs
- Total: ~19ms

Track each component to identify bottlenecks.

### Traffic
Request rate. Completions per second, per memory space.

**Capacity Planning:** If traffic 2x, can we handle it? Where's the limit?

### Errors
Error rate and types. Distinguish client errors (4xx) from server errors (5xx).

**Completion-Specific Errors:**
- Insufficient evidence (422): Not a failure, semantic limit
- Convergence failure (500): Actual error, needs investigation

### Saturation
How "full" is the service? CPU, memory, disk usage approaching limits.

**Completion Saturation:**
- Pattern cache full (evictions per second)
- CA3 weight matrix memory usage
- Working memory for reconstruction

## Prometheus Metric Types

### Counter
Monotonically increasing value. Reset on restart.

**Example:** engram_completion_operations_total

Usage: Rate of change = operations/sec.

### Gauge
Value that can go up or down.

**Example:** engram_pattern_cache_size_bytes

Usage: Current resource utilization.

### Histogram
Distribution of values. Pre-defined buckets.

**Example:** engram_completion_duration_seconds with buckets [0.001, 0.005, 0.01, 0.02, 0.05, 0.1]

Usage: P50, P95, P99 latency from bucket counts.

### Summary
Distribution with client-side quantile calculation.

**Histogram vs Summary:**
- Histogram: Server-side aggregation, allows PromQL quantile()
- Summary: Client-side quantiles, lower server cost

**Choice:** Histogram for completion latency (need aggregation across instances).

## Calibration Monitoring

### Drift Detection
Calibration degrades over time as data distribution changes. Need continuous monitoring.

**Strategy:**
1. Track accuracy per confidence bin (real-time sampling)
2. Compute calibration error every hour
3. Alert if error >10% for 3 consecutive hours
4. Trigger recalibration pipeline

**Metric:**
```
engram_completion_confidence_calibration_error{memory_space, bin}
```

Each bin (0.0-0.1, 0.1-0.2, ..., 0.9-1.0) tracked separately. Identifies which confidence ranges are miscalibrated.

## Grafana Dashboard Design Principles

### Dashboard Hierarchy
**Top-Level:** Red/Amber/Green indicators (RPS, error rate, P95 latency)
**Mid-Level:** Time series graphs (request rate, latency distribution)
**Detail-Level:** Heatmaps (calibration, source attribution)

**Navigation:** Click metric → drill down to details.

### Panel Layout
**Golden Ratio:** 1.618:1 width:height for time series
**Heatmaps:** Square aspect ratio
**Single Stats:** Large font, color-coded thresholds

**Cognitive Load:** <7 panels per row. Group related metrics.

## Alerting Strategy

### Alert Fatigue Prevention
Too many alerts → ignored alerts → missed real issues.

**Principles:**
1. Alert on symptoms (user impact), not causes
2. Make alerts actionable (runbook linked)
3. Tune thresholds to minimize false positives

**Example Alert:**
```yaml
alert: CompletionLatencyHigh
expr: histogram_quantile(0.95, engram_completion_duration_seconds) > 0.025
for: 5m
annotations:
  summary: Completion P95 latency >25ms for 5 minutes
  runbook: docs/operations/completion_latency_high.md
```

Fires after sustained high latency (not transient spikes). Links to remediation steps.

## Structured Logging for Debugging

Metrics tell you "what" and "how much". Logs tell you "why" and "which one".

**Structured Format:**
```json
{
  "timestamp": "2025-10-23T10:30:45Z",
  "level": "INFO",
  "event": "completion_success",
  "memory_space": "user_alice",
  "completion_confidence": 0.82,
  "ca3_iterations": 5,
  "patterns_used": ["pattern_breakfast", "pattern_morning"],
  "latency_ms": 18.3
}
```

**Query Example:** Find all completions for user_alice with latency >30ms and convergence failures.

```
level=ERROR event=completion_failed memory_space=user_alice latency_ms>30
```

Machine-parseable. Aggregate across instances. Powerful filtering.

## References

1. Beyer, B., et al. (2016). Site Reliability Engineering. O'Reilly Media.
2. Prometheus Documentation: Metric Types. https://prometheus.io/docs/concepts/metric_types/
3. Grafana Best Practices. https://grafana.com/docs/grafana/latest/best-practices/
4. Honeycomb Observability Guide. https://www.honeycomb.io/what-is-observability
