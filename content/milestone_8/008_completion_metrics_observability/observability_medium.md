# If You Can't Measure It, You Can't Trust It: Observability for Pattern Completion

Your pattern completion system works in development. 87% accuracy. Sub-20ms latency. Calibration error 6.2%. Beautiful.

Now you deploy to production. Traffic is unpredictable. Data distribution shifts. Edge cases emerge. Completion starts failing in ways you never tested.

How do you know what's happening? How do you diagnose issues? How do you know if users are getting good results?

Answer: Observability. Comprehensive metrics, Grafana dashboards, calibration drift detection, structured logging.

Task 008 implements production monitoring for pattern completion. The difference between "it works on my machine" and "it works in production at scale."

## The Observability Gap

Traditional monitoring asks: "Is the service up?"

Observability asks: "Why is completion latency spiking for user_alice? Which patterns are being used? Is CA3 converging? Is confidence calibrated?"

**The Gap:**
- Metrics tell you "what" (latency is high)
- Logs tell you "why" (CA3 iterations exceeded 7, convergence failed)
- Traces tell you "where" (bottleneck in pattern retrieval, not CA3)

Task 008 focuses on metrics + logs. Distributed tracing is future work.

## The Four Golden Signals for Completion

Google's SRE book defines four signals for monitoring any service:

### 1. Latency: How Long Does Completion Take?

**Naive:** Track average latency.
**Problem:** Average hides distribution. 90% fast, 10% slow → average looks fine.

**Correct:** Track histogram. P50, P95, P99 latencies.

```rust
// Prometheus histogram with pre-defined buckets
lazy_static! {
    static ref COMPLETION_DURATION: Histogram = register_histogram!(
        "engram_completion_duration_seconds",
        "Completion latency in seconds",
        vec![0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.5]
    ).unwrap();
}

// Record latency
let start = Instant::now();
let result = completion.complete(&partial);
COMPLETION_DURATION.observe(start.elapsed().as_secs_f64());
```

**Query:** What's P95 latency over last hour?
```promql
histogram_quantile(0.95, rate(engram_completion_duration_seconds_bucket[1h]))
```

**Alert:** Fire if P95 >25ms for 5+ minutes (sustained high latency, not transient spike).

### 2. Traffic: How Many Completions Per Second?

**Metric:**
```rust
static ref COMPLETION_OPS: IntCounterVec = register_int_counter_vec!(
    "engram_completion_operations_total",
    "Total completion operations",
    &["memory_space", "result"]  // Labels for grouping
).unwrap();

COMPLETION_OPS.with_label_values(&[space_id, "success"]).inc();
```

**Query:** Completion rate per memory space?
```promql
rate(engram_completion_operations_total[5m])
```

**Capacity Planning:** If rate doubles, can we handle it? Where's the bottleneck?

### 3. Errors: What Failures Are Occurring?

**Categories:**
- Insufficient evidence (422): Semantic limit, not a failure
- Convergence failure (500): Actual error, needs investigation
- Invalid request (400): Client error, fix the caller

```rust
match completion.complete(&partial) {
    Ok(result) => {
        COMPLETION_OPS.with_label_values(&[space_id, "success"]).inc();
    }
    Err(CompletionError::InsufficientPattern) => {
        COMPLETION_OPS.with_label_values(&[space_id, "insufficient_evidence"]).inc();
    }
    Err(CompletionError::ConvergenceFailed(_)) => {
        COMPLETION_OPS.with_label_values(&[space_id, "convergence_failure"]).inc();
    }
    Err(_) => {
        COMPLETION_OPS.with_label_values(&[space_id, "error"]).inc();
    }
}
```

**Query:** Error rate over time?
```promql
rate(engram_completion_operations_total{result="error"}[5m])
```

**Alert:** Fire if error rate >5% for 10 minutes.

### 4. Saturation: How Close to Capacity?

**Resources to Monitor:**
- Pattern cache: Memory usage, eviction rate
- CA3 weights: Memory consumption
- Working memory: Buffers, temporary allocations

```rust
static ref CACHE_MEMORY: IntGauge = register_int_gauge!(
    "engram_completion_memory_bytes",
    "Memory used by completion components",
    &["component"]
).unwrap();

// Update periodically
CACHE_MEMORY.with_label_values(&["pattern_cache"]).set(cache.memory_bytes());
```

**Alert:** Fire if pattern cache >80% capacity (approaching eviction churn).

## Calibration Drift Detection

Confidence calibration degrades over time. Data distribution shifts. Model needs recalibration.

**Continuous Monitoring Strategy:**

1. **Sample Completions:** 10% of completions verified against ground truth (where available).

2. **Track Accuracy Per Bin:**
```rust
// For each completion with ground truth
let bin = (confidence * 10.0).floor() as usize;
CALIBRATION_ACCURACY.with_label_values(&[space_id, &bin.to_string()])
    .observe(if accurate { 1.0 } else { 0.0 });
```

3. **Compute Calibration Error:**
```promql
# Per-bin calibration error
abs(
  avg_over_time(engram_calibration_accuracy{bin="7"}[1h]) - 0.75
)
```

Expected: Bin [0.7-0.8] has ~75% accuracy. If actual is 60%, error is 15%.

4. **Alert on Drift:**
```yaml
alert: CalibrationDrift
expr: abs(avg_over_time(engram_calibration_accuracy[1h]) - bin_midpoint) > 0.10
for: 3h
annotations:
  summary: Calibration error >10% for bin {{ $labels.bin }}
  action: Trigger recalibration pipeline
```

5. **Automatic Recalibration:** Alert triggers pipeline that:
   - Collects fresh validation set
   - Recomputes calibration curve
   - Updates confidence computation
   - Deploys new calibration parameters

**Result:** Self-correcting system. Calibration drift detected and remediated automatically.

## Grafana Dashboard: At-a-Glance Health

Production systems need dashboards that answer: "Is everything okay?" in <5 seconds.

**Top Row: Red/Amber/Green Indicators**
- Request rate (green if steady, amber if spiking)
- P95 latency (green if <20ms, amber <25ms, red >25ms)
- Error rate (green if <1%, amber <5%, red >5%)
- Calibration status (green if error <8%, amber <10%, red >10%)

**Second Row: Time Series Graphs**
- Completion operations per second (stacked by result: success, insufficient_evidence, error)
- Latency distribution (P50, P95, P99 lines)
- CA3 convergence iterations (histogram)

**Third Row: Accuracy & Confidence**
- Calibration heatmap: Rows=bins, Columns=time, Color=accuracy
  - Diagonal (perfect calibration) = green
  - Off-diagonal = red
- Source attribution precision per type (Recalled, Reconstructed, Consolidated, Imagined)

**Fourth Row: Resource Usage**
- Memory consumption (pattern cache, CA3 weights, working memory)
- Pattern cache hit rate (should be >60%)
- CA3 energy landscape (deeper = better convergence)

**Interaction:** Click spike in latency graph → drill down to logs for that time period → find which completions were slow → investigate pattern retrieval or CA3 convergence.

## Structured Logging for Root Cause Analysis

Metrics tell you "completion latency spiked at 10:30am."

Logs tell you "which user, which partial episode, which patterns, why convergence failed."

**Structured Log Format:**
```rust
info!(
    event = "completion_success",
    memory_space = space_id,
    completion_confidence = result.confidence,
    ca3_iterations = result.stats.iterations,
    patterns_used = ?result.stats.pattern_sources,
    latency_ms = start.elapsed().as_millis(),
    "Pattern completion succeeded"
);
```

JSON output:
```json
{
  "timestamp": "2025-10-23T10:30:45Z",
  "level": "INFO",
  "event": "completion_success",
  "memory_space": "user_alice",
  "completion_confidence": 0.82,
  "ca3_iterations": 5,
  "patterns_used": ["pattern_breakfast", "pattern_morning"],
  "latency_ms": 18
}
```

**Query Examples:**

Find all failed completions for user_alice:
```
level=ERROR event=completion_failed memory_space=user_alice
```

Find completions with >7 CA3 iterations (convergence struggles):
```
ca3_iterations>7
```

Find slow completions (latency >30ms):
```
latency_ms>30
```

**Aggregation:** How many completions used pattern_breakfast in the last hour?
```
event=completion_success patterns_used~"pattern_breakfast"
| count() by memory_space
```

Machine-readable logs enable powerful querying. Root cause analysis becomes structured search, not log grep.

## Performance Impact: <1% Overhead

Metrics collection must not slow down completion.

**Overhead Sources:**
- Counter increment: ~20ns (atomic add)
- Histogram observation: ~100ns (bucket lookup + atomic add)
- Gauge update: ~30ns (atomic store)
- Structured log: ~2μs (format + write to buffer)

**Per Completion:**
- 5 counter increments: 100ns
- 3 histogram observations: 300ns
- 2 gauge updates: 60ns
- 1 log line: 2μs
- Total: ~2.5μs

Completion latency: ~19ms = 19,000μs

Metrics overhead: 2.5μs / 19,000μs = 0.013% = negligible.

**Design Principle:** Metrics are fast. Logging is cheap. Profile to verify overhead <1%.

## Alerts vs Dashboards

**Dashboards:** Human-in-loop monitoring. Check periodically. Good for exploration.

**Alerts:** Automated monitoring. Fire when thresholds crossed. Good for known failure modes.

**Balance:**
- Don't alert on everything (alert fatigue → ignored alerts)
- Alert on user impact (high latency, high error rate, calibration drift)
- Link alerts to runbooks (actionable remediation steps)

**Example Runbook:**

```markdown
# Runbook: Completion Latency High

## Symptom
P95 completion latency >25ms for 5+ minutes.

## Investigation
1. Check Grafana: Which component is slow? (pattern retrieval, CA3, integration)
2. If CA3: Check convergence iterations. >7 iterations? Tune sparsity.
3. If pattern retrieval: Check cache hit rate. <50%? Increase cache size.
4. If consistent: Check traffic spike. 2x normal? Scale horizontally.

## Remediation
- Short-term: Increase CA1 threshold (fewer completions, lower latency)
- Medium-term: Tune parameters (ca3_sparsity, pattern_weight)
- Long-term: Optimize hot paths (SIMD, caching)
```

Alert → Runbook → Resolution. Minimize time to mitigation.

## Conclusion

Observability transforms "works in development" to "works in production at scale."

Comprehensive metrics (latency, traffic, errors, saturation). Calibration drift detection. Grafana dashboards. Structured logging. Automated alerts with runbooks.

The result: Production pattern completion you can trust.

Next: Accuracy validation and parameter tuning (Task 009) to optimize for real-world workloads.

---

**Citations:**
- Beyer, B., et al. (2016). Site Reliability Engineering: How Google Runs Production Systems
- Prometheus Documentation: Best Practices
- Grafana Dashboard Design Guide
