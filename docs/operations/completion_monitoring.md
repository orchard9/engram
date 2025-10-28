# Pattern Completion Monitoring Operations Guide

## Overview

This guide covers monitoring, troubleshooting, and tuning pattern completion operations in production. Pattern completion is a critical path operation that reconstructs missing parts of episodes using CA3 autoassociative dynamics.

## Key Metrics (Four Golden Signals)

### 1. Latency

**Target:** P95 < 25ms, P99 < 50ms

**Key Metrics:**

- `engram_completion_duration_seconds` - Total completion latency

- `engram_pattern_retrieval_duration_seconds` - Pattern retrieval phase (~3ms target)

- `engram_ca3_convergence_duration_seconds` - CA3 convergence phase (~14ms target)

- `engram_evidence_integration_duration_seconds` - Integration phase (~1ms target)

**Alert Thresholds:**

```yaml

- alert: CompletionLatencyHigh
  expr: histogram_quantile(0.95, engram_completion_duration_seconds) > 0.025
  for: 5m
  severity: warning
  annotations:
    summary: "P95 completion latency >25ms for 5 minutes"
    runbook: "See 'High Latency Troubleshooting' section"

```

### 2. Traffic

**Capacity:** ~1000 completions/sec per instance

**Key Metrics:**

- `engram_completion_operations_total{result="success|failure"}` - Operation count by result

- `engram_patterns_used_per_completion` - Pattern usage per completion

**Monitoring:**

```promql
# Completion rate per memory space
sum(rate(engram_completion_operations_total{memory_space="$space"}[5m])) by (result)

# Pattern usage efficiency
histogram_quantile(0.95, engram_patterns_used_per_completion)

```

### 3. Errors

**Target:** Error rate < 1%

**Key Metrics:**

- `engram_completion_insufficient_evidence_total` - Semantic limit reached (not a failure)

- `engram_completion_convergence_failures_total` - CA3 convergence failures (actual error)

**Alert Thresholds:**

```yaml

- alert: CompletionErrorRateHigh
  expr: |
    sum(rate(engram_completion_convergence_failures_total[5m])) /
    sum(rate(engram_completion_operations_total[5m])) > 0.01
  for: 10m
  severity: critical
  annotations:
    summary: "Completion error rate >1% for 10 minutes"
    runbook: "See 'Error Rate Troubleshooting' section"

```

### 4. Saturation

**Target:** Memory usage < 80% of limits

**Key Metrics:**

- `engram_completion_memory_bytes{component="cache|ca3_weights|working_memory"}` - Memory by component

- `engram_pattern_cache_hit_ratio` - Cache effectiveness

- `engram_ca3_attractor_energy` - Convergence energy

**Monitoring:**

```promql
# Total memory usage
sum(engram_completion_memory_bytes) by (memory_space)

# Cache saturation
1 - engram_pattern_cache_hit_ratio

```

## Calibration Monitoring

Confidence calibration degrades over time as data distribution shifts. Monitor and trigger recalibration when needed.

### Calibration Metrics

- `engram_completion_confidence_calibration_error{bin}` - Error per confidence bin

- `engram_metacognitive_correlation` - Correlation between predicted and actual accuracy

### Calibration Alert

```yaml

- alert: CalibrationDriftHigh
  expr: avg(engram_completion_confidence_calibration_error) > 0.1
  for: 3h
  severity: warning
  annotations:
    summary: "Calibration error >10% for 3 hours"
    action: "Trigger recalibration pipeline"

```

### Recalibration Process

1. Collect recent completion results (last 24h)

2. Group by confidence bins (0.0-0.1, 0.1-0.2, ..., 0.9-1.0)

3. Compute actual accuracy per bin

4. Train isotonic regression model

5. Deploy new calibration model with blue-green deployment

## Common Issues and Remediation

### High Latency Troubleshooting

#### Symptom: P95 latency > 25ms

**1. Check CA3 Convergence**

```promql
# Check convergence iterations
histogram_quantile(0.95, engram_ca3_convergence_iterations)

```

**If iterations > 5:**

- Increase `ca3_sparsity` from 0.05 to 0.07

- Reduce `convergence_threshold` from 0.01 to 0.02

- Check for noisy input patterns

**2. Check Pattern Cache**

```promql
# Cache hit ratio
avg(engram_pattern_cache_hit_ratio)

```

**If hit ratio < 0.8:**

- Increase cache size: `ENGRAM_PATTERN_CACHE_SIZE=10000`

- Enable cache prewarming for hot patterns

- Review cache eviction policy (LRU vs LFU)

**3. Check Memory Pressure**

```bash
# Check for swap activity
vmstat 1 10 | grep -E "si|so"

```

**If swapping detected:**

- Increase instance memory

- Reduce CA3 weight matrix size

- Enable memory-mapped files for cold patterns

### Error Rate Troubleshooting

#### Symptom: Convergence failures > 1%

**1. Check Input Quality**

```promql
# Patterns with low cue strength
histogram_quantile(0.1, engram_patterns_cue_strength)

```

**If P10 cue strength < 0.3:**

- Improve embedding quality

- Increase pattern separation in DG

- Add more training data

**2. Check CA3 Stability**

```promql
# Energy distribution
histogram_quantile(0.95, engram_ca3_attractor_energy)

```

**If P95 energy > 1.0:**

- Reduce learning rate: `ca3_learning_rate=0.01`

- Increase weight decay: `ca3_weight_decay=0.001`

- Check for catastrophic interference

### Insufficient Evidence Issues

#### Symptom: High rate of insufficient evidence responses

**1. Check Pattern Coverage**

```sql
-- Query pattern database
SELECT memory_space, COUNT(*) as pattern_count
FROM semantic_patterns
WHERE created_at > NOW() - INTERVAL '7 days'
GROUP BY memory_space;

```

**If pattern count < 1000:**

- Trigger consolidation run

- Lower consolidation threshold

- Increase episode retention

**2. Check Evidence Threshold**

```promql
# Evidence accumulation
histogram_quantile(0.5, engram_evidence_accumulated)

```

**If median evidence < 0.5:**

- Reduce evidence threshold from 0.6 to 0.5

- Increase spreading activation radius

- Enable transitive evidence paths

## Performance Tuning

### CA3 Optimization

**Sparse Operations:**

```rust
// Use sparse matrix multiplication for CA3
ca3_sparse_ratio: 0.95  // 95% zeros
ca3_batch_size: 32      // Process in batches

```

**GPU Acceleration:**

```yaml
# Enable for >100 concurrent completions
ENGRAM_CA3_GPU_ENABLED: true
ENGRAM_CA3_GPU_THRESHOLD: 100

```

### Pattern Cache Tuning

**Cache Configuration:**

```yaml
# Optimal settings for 16GB instances
pattern_cache:
  size: 10000
  ttl_seconds: 3600
  eviction: lfu  # Least Frequently Used
  prefetch: true
  prefetch_count: 10

```

**Monitoring Cache Performance:**

```promql
# Cache efficiency score
engram_pattern_cache_hit_ratio * (1 - (engram_pattern_cache_size_bytes / 1073741824))

```

### Memory Management

**Tiered Storage:**

1. **Hot Tier:** Recent patterns (last 1h) - In-memory

2. **Warm Tier:** Active patterns (last 24h) - Memory-mapped

3. **Cold Tier:** Historical patterns - Disk with async loading

**Configuration:**

```yaml
storage_tiers:
  hot:
    max_size: 1GB
    max_age: 1h
  warm:
    max_size: 10GB
    max_age: 24h
  cold:
    location: /data/patterns
    compression: zstd

```

## Capacity Planning

### Resource Requirements

**Per 1000 completions/sec:**

- CPU: 4 cores (8 with hyperthreading)

- Memory: 16GB

- Network: 100Mbps

- Disk: 100GB SSD (for pattern storage)

### Scaling Triggers

**Horizontal Scaling:**

```yaml

- metric: completion_rate
  threshold: 800/sec
  action: add_instance
  cooldown: 5m

- metric: p95_latency
  threshold: 30ms
  action: add_instance
  cooldown: 10m

```

**Vertical Scaling:**

```yaml

- metric: memory_usage
  threshold: 85%
  action: increase_memory
  amount: 8GB

```

## Monitoring Dashboard

Access the Grafana dashboard at: `/grafana/dashboards/pattern_completion.json`

### Dashboard Sections

1. **Overview (Row 1)**
   - Completion rate
   - Error rate
   - P95 latency
   - Memory saturation

2. **Performance (Row 2)**
   - Latency distribution (P50, P95, P99)
   - Component latency breakdown
   - CA3 convergence iterations
   - Cache hit ratio

3. **Accuracy (Row 3)**
   - Calibration error heatmap
   - Source attribution precision
   - Metacognitive correlation

4. **Resources (Row 4)**
   - Memory usage by component
   - CA3 attractor energy
   - Pattern cache size

## Alert Response Procedures

### Critical Alerts

**1. Completion Service Down**

```bash
# Check service health
curl http://localhost:8080/health/completion

# Check logs
journalctl -u engram-completion -n 100

# Restart if needed
systemctl restart engram-completion

```

**2. Memory Exhaustion**

```bash
# Check memory usage
free -h
ps aux | grep engram | awk '{sum+=$6} END {print sum/1024 " MB"}'

# Clear pattern cache
curl -X POST http://localhost:8080/admin/cache/clear

# Trigger GC
kill -USR1 $(pgrep engram)

```

### Warning Alerts

**1. High Latency**

- Check CA3 convergence stats

- Review recent pattern complexity

- Consider cache warming

**2. Calibration Drift**

- Schedule recalibration job

- Review recent completion accuracy

- Check for distribution shift

## Structured Logging

All completion operations emit structured logs for debugging:

```json
{
  "timestamp": "2025-10-24T10:30:45Z",
  "level": "INFO",
  "event": "completion_success",
  "memory_space": "user_alice",
  "completion_confidence": 0.82,
  "ca3_iterations": 5,
  "patterns_used": 3,
  "latency_ms": 18.3,
  "component_latencies": {
    "pattern_retrieval_ms": 3.1,
    "ca3_convergence_ms": 13.8,
    "evidence_integration_ms": 0.9,
    "source_attribution_us": 320,
    "confidence_computation_us": 180
  }
}

```

### Log Queries

**Find slow completions:**

```bash
jq 'select(.latency_ms > 30)' /var/log/engram/completion.json

```

**Analyze convergence failures:**

```bash
grep "convergence_failure" /var/log/engram/completion.json | \
  jq -r '[.memory_space, .iterations] | @csv' | \
  sort | uniq -c

```

## Health Checks

### Endpoint: `/health/completion`

```json
{
  "status": "healthy",
  "checks": {
    "pattern_cache": "ok",
    "ca3_weights": "ok",
    "calibration": "warning",
    "memory": "ok"
  },
  "metrics": {
    "p95_latency_ms": 22.5,
    "error_rate": 0.003,
    "cache_hit_ratio": 0.85,
    "calibration_error": 0.08
  }
}

```

### Automated Health Monitoring

```yaml
# Kubernetes liveness probe
livenessProbe:
  httpGet:
    path: /health/completion
    port: 8080
  periodSeconds: 10
  failureThreshold: 3

# Readiness probe
readinessProbe:
  httpGet:
    path: /health/completion
    port: 8080
  initialDelaySeconds: 30
  periodSeconds: 5

```

## Best Practices

1. **Monitor Calibration Weekly** - Review calibration drift trends

2. **Cache Warming** - Prewarm cache for known hot patterns

3. **Gradual Rollouts** - Use blue-green deployments for config changes

4. **Load Testing** - Test at 2x expected peak load

5. **Observability** - Correlate metrics with logs for root cause analysis

6. **Capacity Buffer** - Maintain 30% capacity headroom

## Support Escalation

1. **Level 1:** Check dashboard, review recent alerts

2. **Level 2:** Analyze logs, check configuration

3. **Level 3:** Review CA3 weights, debug convergence

4. **Engineering:** Code-level debugging, algorithm tuning

## References

- [Pattern Completion Architecture](../explanation/pattern_completion.md)

- [CA3 Attractor Dynamics](../reference/ca3_dynamics.md)

- [Confidence Calibration](../howto/calibrate_confidence.md)

- [Performance Tuning Guide](../howto/tune_performance.md)
