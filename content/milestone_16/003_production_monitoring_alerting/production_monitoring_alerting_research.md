# Production Monitoring and Alerting - Research

## Research Objectives

Design comprehensive observability stack for probabilistic graph databases to enable proactive issue detection and <5 minute MTTD (Mean Time To Detect).

## Key Findings

### Prometheus Metrics for Graph Databases

**Source: Prometheus Best Practices, Google SRE Book**

Four golden signals for monitoring:
- Latency: Request duration distributions (p50, p90, p99, p99.9)
- Traffic: Request rate (ops/sec)
- Errors: Error rate by type
- Saturation: Resource utilization (CPU, memory, disk, network)

**Engram-Specific Metrics:**

```
# Latency
engram_operation_duration_seconds{operation="activate",tier="fast"} histogram
engram_operation_duration_seconds{operation="consolidate",tier="warm"} histogram

# Traffic
engram_operations_total{operation="create_memory"} counter
engram_operations_total{operation="activate"} counter
engram_activations_per_query{} histogram

# Errors
engram_errors_total{type="out_of_memory"} counter
engram_errors_total{type="corruption_detected"} counter

# Saturation
engram_memory_tier_utilization{tier="fast"} gauge
engram_cache_hit_rate{tier="fast"} gauge
engram_consolidation_queue_depth{} gauge
```

### Alert Rule Design

**Source: Site Reliability Engineering Workbook, Chapter 5-6**

Alert on symptoms, not causes. User-facing impact over system metrics.

**Bad Alert:** CPU usage >80%
**Good Alert:** P99 latency >10ms for 5 minutes

Why? CPU spike might not affect users. Latency definitely does.

**Engram Alert Rules:**

```yaml
groups:
- name: engram_slo
  rules:
  # Latency SLO: P99 <10ms
  - alert: HighLatency
    expr: histogram_quantile(0.99, rate(engram_operation_duration_seconds_bucket[5m])) > 0.010
    for: 5m
    severity: warning

  # Error rate SLO: <0.1%
  - alert: HighErrorRate
    expr: rate(engram_errors_total[5m]) / rate(engram_operations_total[5m]) > 0.001
    for: 2m
    severity: critical

  # Cache hit rate: >70%
  - alert: LowCacheHitRate
    expr: engram_cache_hit_rate{tier="fast"} < 0.7
    for: 10m
    severity: warning
```

### Grafana Dashboard Design

**Source: Grafana Dashboard Best Practices**

Dashboard hierarchy:
1. Overview: Single pane health check
2. Detail: Per-component deep dive
3. Debug: Low-level system metrics

**Engram Dashboard Structure:**

**Overview Dashboard:**
- Request rate (ops/sec)
- Latency heatmap (p50, p99, p99.9)
- Error rate
- Cache hit rate
- Memory tier utilization

**Memory Operations Dashboard:**
- Create/update/delete rates
- Activation patterns
- Consolidation metrics
- Pattern completion stats

**Storage Dashboard:**
- Tier utilization (fast/warm/cold)
- I/O latency by tier
- Compaction metrics
- Backup status

**API Dashboard:**
- Endpoint latency by path
- Request rate by endpoint
- Error rate by status code
- Client distribution

### Log Aggregation with Loki

**Source: Grafana Loki Documentation**

Loki philosophy: Index labels, not content. Query when needed.

**Structured Logging:**

```rust
use tracing::{info, warn, error};

info!(
    operation = "activate",
    latency_ms = 3.2,
    node_count = 150,
    cache_hit = true,
    "Activation complete"
);
```

**Loki Query:**

```
{app="engram"} 
| json 
| latency_ms > 10 
| line_format "High latency: {{.latency_ms}}ms for {{.operation}}"
```

### Performance Impact of Observability

**Source: OpenTelemetry Performance Benchmarks**

Metric collection overhead:
- Counter increment: 5-10ns
- Histogram sample: 50-100ns
- Trace span: 200-500ns

For 10K ops/sec:
- Metrics: 1ms/sec total (0.1% overhead)
- Traces (1% sampling): 1ms/sec (0.1% overhead)
- Logs (structured JSON): 10ms/sec (1% overhead)

Total overhead: <2% with proper sampling.

**Sampling Strategy:**
- Metrics: 100% (cheap)
- Traces: 1% normal, 100% if error or >100ms latency
- Logs: INFO for all, DEBUG for errors only

## Implementation Checklist

- [ ] Prometheus instrumentation in Rust code
- [ ] ServiceMonitor for auto-discovery
- [ ] Alert rules for SLO violations
- [ ] Grafana dashboards (overview, operations, storage, API)
- [ ] Loki for log aggregation
- [ ] Alert routing (PagerDuty, Slack, email)
- [ ] Runbook links in alerts
- [ ] Dashboard templates in Helm chart

## Citations

1. Beyer, B., et al. (2016). Site Reliability Engineering. O'Reilly Media.
2. Beyer, B., et al. (2018). The Site Reliability Workbook. O'Reilly Media.
3. Grafana Labs (2024). Loki Documentation.
4. Grafana Labs (2024). Dashboard Best Practices.
5. Prometheus Authors (2024). Best Practices.
