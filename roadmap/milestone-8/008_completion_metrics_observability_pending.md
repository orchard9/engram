# Task 008: Completion Metrics & Observability

**Status:** Pending
**Priority:** P1 (Important)
**Estimated Effort:** 2 days
**Dependencies:** Task 007 (Complete API)

## Objective

Implement comprehensive Prometheus metrics and Grafana dashboard for pattern completion monitoring. Track accuracy, latency, confidence calibration, source attribution precision, and resource usage to enable production tuning and anomaly detection.

## Integration Points

**Uses:**
- `/engram-core/src/metrics/mod.rs` - Metrics registry from M6
- `/engram-core/src/completion/*.rs` - Completion components
- `/grafana/dashboards/` - Dashboard JSON definitions

**Creates:**
- `/engram-core/src/metrics/completion_metrics.rs` - Completion-specific metrics
- `/grafana/dashboards/pattern_completion.json` - Grafana dashboard
- `/docs/operations/completion_monitoring.md` - Operations runbook

## Theoretical Foundations from Research

### Observability vs Monitoring (Honeycomb, 2018)

**Monitoring:** Known unknowns. Predefined metrics, alerts for expected failures.
- Example: Track latency P95, error rate, completion rate

**Observability:** Unknown unknowns. Arbitrary queries to understand emergent behavior.
- Example: Why did this specific completion take 50ms? Which patterns contributed?

**Application to Completion (Task 008):**
- **Monitoring:** Prometheus metrics (quantitative, time-series)
- **Observability:** Structured logs (qualitative, exploratory)

**Both Required:** Metrics tell "what" and "how much". Logs tell "why" and "which one".

### The Four Golden Signals (Google SRE)

**1. Latency:** Time to serve a request. Track distribution (P50, P95, P99), not just average.

**Completion Latency Breakdown (from research):**
```
engram_pattern_retrieval_duration_seconds:  ~3ms   (Task 003)
engram_ca3_convergence_duration_seconds:    ~14ms  (Task 002)
engram_evidence_integration_duration_ms:    ~1ms   (Task 004)
engram_source_attribution_duration_us:      ~500μs (Task 005)
engram_confidence_computation_duration_us:  ~200μs (Task 006)
------------------------------------------------------
Total: ~19ms
```

Track each component to identify bottlenecks.

**Prometheus Histogram:**
```rust
engram_completion_duration_seconds{memory_space, quantile}
buckets: [0.001, 0.005, 0.01, 0.02, 0.05, 0.1]
```

**2. Traffic:** Request rate. Completions per second, per memory space.

**Capacity Planning (from research):** If traffic 2x, can we handle it? Where's the limit?

**Prometheus Counter:**
```rust
engram_completion_operations_total{memory_space, result="success|failure"}
```

**3. Errors:** Error rate and types. Distinguish client errors (4xx) from server errors (5xx).

**Completion-Specific Errors (from Task 007 research):**
- **Insufficient evidence (422):** Not a failure, semantic limit reached
- **Convergence failure (500):** Actual error, needs investigation (CA3 didn't converge in 7 iterations)

**Prometheus Counters:**
```rust
engram_completion_insufficient_evidence_total{memory_space}
engram_completion_convergence_failures_total{memory_space}
```

**4. Saturation:** How "full" is the service? CPU, memory, disk usage approaching limits.

**Completion Saturation (from research):**
```rust
engram_pattern_cache_size_bytes{memory_space}      // Pattern cache from Task 003
engram_ca3_weight_matrix_bytes{memory_space}       // CA3 weights from Task 002
engram_completion_working_memory_bytes{memory_space} // Reconstruction buffers
```

Alert when approaching memory limits (e.g., cache >80% capacity).

### Prometheus Metric Types (Prometheus Documentation)

**Counter:** Monotonically increasing value. Reset on restart.
- Example: `engram_completion_operations_total`
- Usage: Rate of change = operations/sec via `rate(engram_completion_operations_total[5m])`

**Gauge:** Value that can go up or down.
- Example: `engram_pattern_cache_size_bytes`
- Usage: Current resource utilization

**Histogram:** Distribution of values. Pre-defined buckets.
- Example: `engram_completion_duration_seconds` with buckets [0.001, 0.005, 0.01, 0.02, 0.05, 0.1]
- Usage: P50, P95, P99 latency from bucket counts via `histogram_quantile(0.95, ...)`

**Histogram vs Summary:**
- **Histogram:** Server-side aggregation, allows PromQL `quantile()` across instances
- **Summary:** Client-side quantiles, lower server cost but can't aggregate

**Engram Choice (from research):** Histogram for completion latency (need aggregation across instances).

### Calibration Monitoring (Task 006 Integration)

Calibration degrades over time as data distribution changes. Need continuous monitoring.

**Strategy (from research):**
1. Track accuracy per confidence bin (real-time sampling)
2. Compute calibration error every hour
3. Alert if error >10% for 3 consecutive hours
4. Trigger recalibration pipeline (isotonic regression re-training)

**Metric:**
```rust
engram_completion_confidence_calibration_error{memory_space, bin}
```

Each bin (0.0-0.1, 0.1-0.2, ..., 0.9-1.0) tracked separately. Identifies which confidence ranges are miscalibrated.

**Grafana Heatmap Panel:** Visualize calibration drift over time (bin vs time, color = error).

### Grafana Dashboard Design Principles

**Dashboard Hierarchy (from research):**
- **Top-Level:** Red/Amber/Green indicators (RPS, error rate, P95 latency)
- **Mid-Level:** Time series graphs (request rate, latency distribution)
- **Detail-Level:** Heatmaps (calibration, source attribution)

**Navigation:** Click metric → drill down to details.

**Panel Layout:**
- **Golden Ratio:** 1.618:1 width:height for time series
- **Heatmaps:** Square aspect ratio
- **Single Stats:** Large font, color-coded thresholds

**Cognitive Load:** <7 panels per row. Group related metrics.

**Engram Dashboard Sections:**
1. **Overview:** Completion rate, success/failure ratio, P95 latency
2. **Performance:** Latency breakdown (pattern retrieval, CA3, integration)
3. **Accuracy:** Calibration heatmap, source attribution precision
4. **Resources:** Memory usage, cache hit rate, CA3 energy

### Alerting Strategy: Preventing Alert Fatigue

**Principles (Google SRE, from research):**
1. **Alert on symptoms (user impact), not causes**
2. **Make alerts actionable (runbook linked)**
3. **Tune thresholds to minimize false positives**

**Example Alert:**
```yaml
alert: CompletionLatencyHigh
expr: histogram_quantile(0.95, engram_completion_duration_seconds) > 0.025
for: 5m
annotations:
  summary: Completion P95 latency >25ms for 5 minutes
  runbook: docs/operations/completion_latency_high.md
```

Fires after **sustained** high latency (not transient spikes). Links to remediation steps.

**Runbook Contents (from docs/operations/completion_monitoring.md):**
1. Check CA3 convergence iterations (Task 002 metric)
2. Check pattern cache hit rate (Task 003 metric)
3. Check memory saturation (swap activity)
4. Adjust CA3 max_iterations or pattern cache size

### Structured Logging for Debugging

Metrics tell you "what" and "how much". Logs tell you "why" and "which one".

**Structured Format (from research):**
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

**Integration with Metrics:** Logs provide context when metrics show anomaly. P95 latency spike at 10:30? Query logs for that time window to see which completions were slow and why.

## Prometheus Metrics

```rust
// Completion operations
engram_completion_operations_total{memory_space, result="success|failure"}
engram_completion_insufficient_evidence_total{memory_space}

// Latency
engram_completion_duration_seconds{memory_space, quantile="0.5|0.95|0.99"}
engram_ca3_convergence_iterations{memory_space, quantile="0.5|0.95"}

// Accuracy (sampled)
engram_completion_accuracy_ratio{memory_space, confidence_bin}
engram_source_attribution_precision{memory_space, source_type}
engram_reconstruction_plausibility_score{memory_space, quantile="0.5|0.95"}

// Confidence calibration
engram_completion_confidence_calibration_error{memory_space, bin}
engram_metacognitive_correlation{memory_space}

// Pattern usage
engram_pattern_retrieval_duration_seconds{memory_space, quantile="0.5|0.95"}
engram_pattern_cache_hit_ratio{memory_space}
engram_patterns_used_per_completion{memory_space, quantile="0.5|0.95"}

// Resource usage
engram_completion_memory_bytes{memory_space, component="cache|working_memory"}
engram_ca3_attractor_energy{memory_space, quantile="0.5|0.95"}
```

## Grafana Dashboard

**Panels:**
1. Completion Operations: Request rate, success/failure ratio, insufficient evidence rate
2. Performance: P50/P95/P99 latency, CA3 convergence iterations
3. Accuracy: Calibration heatmap, source attribution precision, plausibility scores
4. Resource Usage: Memory consumption, cache hit rate, CA3 energy

## Acceptance Criteria

1. **Metrics Coverage:** All key completion aspects instrumented (operations, latency, accuracy, resources)
2. **Dashboard Usability:** Grafana dashboard provides actionable insights; <30s to diagnose issues
3. **Calibration Monitoring:** Drift detection alerts when calibration error exceeds 10%
4. **Operations Runbook:** Documentation covers common issues and remediation steps
5. **Performance:** Metrics overhead <1% of completion latency

## Success Criteria Validation

- [ ] All metrics implemented and validated
- [ ] Grafana dashboard deployed and functional
- [ ] Calibration drift alerts configured
- [ ] Operations runbook complete
- [ ] Metrics overhead <1%
