# Completion Metrics & Observability - Twitter Thread

1/9 Your pattern completion works in development.

87% accuracy. Sub-20ms latency. Calibration error 6.2%.

Then production happens. Traffic spikes. Data shifts. Edge cases emerge.

How do you know what's happening?

Task 008: Production observability.

2/9 Four Golden Signals (Google SRE):

1. Latency: How long? (P50/P95/P99, not average)
2. Traffic: How many? (ops/sec per memory space)
3. Errors: What's failing? (categorized by type)
4. Saturation: How close to limits? (cache, memory)

Complete service health picture.

3/9 Latency: Track distribution, not average

Average: 90% fast, 10% slow → looks fine
Reality: 10% of users suffering

Solution: Histogram with buckets [1ms, 5ms, 10ms, 20ms, 50ms, 100ms]

Query: P95 latency over last hour?
```promql
histogram_quantile(0.95, rate(engram_completion_duration_seconds_bucket[1h]))
```

4/9 Errors: Categorize by type

Not all errors equal:
- Insufficient evidence (422): Semantic limit, expected
- Convergence failure (500): Actual error, investigate
- Invalid request (400): Client error, fix caller

Track separately. Different remediation for each.

5/9 Calibration drift detection:

Confidence degrades over time (data distribution shifts).

Solution: Continuous monitoring
1. Sample 10% completions
2. Track accuracy per confidence bin
3. Alert if error >10% for 3 hours
4. Trigger automatic recalibration

Self-correcting system.

6/9 Grafana dashboard design:

Top row: Red/Amber/Green indicators (RPS, P95, error rate, calibration)
Answer "Is everything okay?" in <5 seconds.

Second row: Time series (ops/sec, latency distribution)
Third row: Accuracy heatmaps (calibration, source attribution)
Fourth row: Resources (memory, cache hit rate, CA3 energy)

7/9 Structured logging for root cause:

Metrics: "Latency spiked at 10:30"
Logs: "user_alice, 12 patterns, CA3 convergence failed, latency 47ms"

JSON format:
```json
{
  "event": "completion_failed",
  "memory_space": "user_alice",
  "ca3_iterations": 8,
  "convergence": false,
  "latency_ms": 47
}
```

Machine-parseable. Powerful queries.

8/9 Performance: <1% overhead

Counter increment: 20ns
Histogram observation: 100ns
Per completion: ~2.5μs

Completion latency: ~19ms = 19,000μs

Overhead: 2.5 / 19,000 = 0.013%

Negligible. Comprehensive instrumentation for free.

9/9 Alerts with runbooks:

Don't alert on everything (alert fatigue).
Alert on user impact (latency, errors, drift).
Link to runbooks (actionable steps).

Example:
"P95 latency >25ms → Check Grafana → If CA3 slow, tune sparsity → If cache cold, increase size"

Production pattern completion you can trust.

github.com/[engram]/milestone-8/008

---

## Calibration Drift Thread

1/5 Problem: Confidence calibration degrades over time.

Initially: 70% confidence → 70% accuracy
After 3 months: 70% confidence → 55% accuracy

Data distribution shifted. Model miscalibrated.

Solution: Automated drift detection + recalibration.

2/5 Continuous monitoring:

Sample 10% of completions with ground truth.
Track accuracy per confidence bin [0.0-0.1, 0.1-0.2, ..., 0.9-1.0].

Metric:
```
engram_calibration_accuracy{bin="7"} = 0.55
```

Expected: 0.75 for bin [0.7-0.8].
Error: 20%.

3/5 Alert on sustained drift:

```yaml
alert: CalibrationDrift
expr: abs(avg(calibration_accuracy) - bin_midpoint) > 0.10
for: 3h
```

Fires if error >10% for 3 consecutive hours (not transient spike).

4/5 Automatic recalibration:

Alert triggers pipeline:
1. Collect fresh validation set (1000+ completions)
2. Recompute calibration curve (isotonic regression)
3. Update confidence computation parameters
4. Deploy new calibration
5. Verify error <8%

Zero manual intervention.

5/5 Result: Self-correcting system

Calibration maintained automatically.
Confidence scores stay reliable.
70% means 70% accuracy, always.

Production-grade AI requires production-grade ops.

github.com/[engram]/milestone-8/008
