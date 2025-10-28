# Production Monitoring Guide

Comprehensive guide to monitoring Engram in production using Prometheus, Grafana, and Loki.

## Overview

Engram uses a streaming-first observability model that supports both real-time monitoring and historical analysis:

- **Prometheus** - Metrics collection and alerting

- **Grafana** - Visualization dashboards

- **Loki** - Structured log aggregation

- **Streaming API** - Real-time metrics via HTTP/SSE

### Architecture

```
┌──────────────┐
│    Engram    │
│   Instance   │
└──────┬───────┘
       │
       ├─────► /metrics/prometheus (Prometheus text format)
       │
       ├─────► /metrics (JSON format for SSE)
       │
       └─────► stdout (JSON logs)
              │
              ▼
       ┌──────────────┐
       │  Promtail    │
       └──────┬───────┘
              │
              ▼
       ┌──────────────┐
       │     Loki     │
       └──────────────┘

┌──────────────┐
│  Prometheus  │◄───── Scrapes /metrics/prometheus every 15s
└──────┬───────┘
       │
       ▼
┌──────────────┐
│   Grafana    │◄───── Queries Prometheus & Loki
└──────────────┘

```

## Quick Start

### Kubernetes Deployment

Deploy the complete monitoring stack:

```bash
./scripts/setup_monitoring.sh --kubernetes --validate

```

Access Grafana:

```bash
kubectl port-forward -n monitoring svc/grafana 3000:3000

```

Open http://localhost:3000 (admin/admin)

### Docker Compose Deployment

```bash
./scripts/setup_monitoring.sh --docker --validate

```

Access:

- Grafana: http://localhost:3000

- Prometheus: http://localhost:9090

- Loki: http://localhost:3100

## Metrics Reference

### Spreading Activation Metrics

#### engram_spreading_activations_total

**Type:** Counter
**Description:** Total activation operations across all tiers
**Expected Range:** Increases with query load, rate depends on workload
**Alert:** None (informational)

#### engram_spreading_latency_hot_seconds

**Type:** Summary
**Description:** Hot tier activation latency distribution
**Expected Range:** P90 < 100ms for cognitive plausibility
**Alert:** SpreadingLatencySLOBreach if P90 > 100ms for 5m
**Quantiles:** 0.5 (median), 0.9, 0.99

Example query:

```promql
# P90 spreading latency trend
engram_spreading_latency_hot_seconds{quantile="0.9"}

```

#### engram_spreading_breaker_state

**Type:** Gauge
**Description:** Circuit breaker state
**Values:**

- 0 = Closed (healthy)

- 1 = Open (failing fast)

- 2 = Half-open (testing recovery)
**Alert:** SpreadingCircuitBreakerOpen if state=1 for 5m

#### activation_pool_hit_rate

**Type:** Gauge (0.0-1.0)
**Description:** Percentage of activation records reused from pool
**Expected Range:** >0.80 is healthy, <0.50 indicates inefficiency
**Alert:** ActivationPoolLowHitRate if <0.50 for 15m

### Consolidation Metrics

#### engram_consolidation_freshness_seconds

**Type:** Gauge
**Description:** Age of cached consolidation snapshot in seconds
**Expected Range:** <450s (1.5x scheduler interval)
**Alert:** ConsolidationStaleness if >900s for 5m

#### engram_consolidation_novelty_gauge

**Type:** Gauge
**Description:** Latest novelty delta from consolidation run
**Expected Range:** 0.01-0.50 during active learning, <0.01 at steady state
**Alert:** ConsolidationNoveltyStagnation if <0.01 for 30m

#### engram_consolidation_runs_total

**Type:** Counter
**Description:** Total successful consolidation runs
**Expected Range:** Increases every 5 minutes (default scheduler interval)
**Alert:** ConsolidationFailureStreak if 3+ failures in 15m

### Storage Metrics

#### engram_compaction_success_total

**Type:** Counter
**Description:** Number of successful storage compaction operations
**Expected Range:** Increases when episode count exceeds thresholds
**Alert:** None

#### engram_wal_recovery_duration_seconds

**Type:** Summary
**Description:** Time to recover WAL entries on startup
**Expected Range:** <10s for normal startup, <60s after crash
**Alert:** None (monitored for capacity planning)

### Adaptive Batching Metrics

#### adaptive_batch_hot_confidence

**Type:** Gauge (0.0-1.0)
**Description:** Convergence confidence for hot tier batch size
**Expected Range:** >0.50 after warmup, >0.80 during stable operation
**Alert:** AdaptiveBatchingNotConverging if <0.30 for 30m

#### adaptive_batch_latency_ewma_ns

**Type:** Gauge
**Description:** Exponentially weighted moving average of batch processing latency
**Expected Range:** Tracks workload characteristics, no fixed range
**Alert:** None (informational for tuning)

## Common Prometheus Queries

### Spreading Activation Performance

**Request rate:**

```promql
rate(engram_spreading_activations_total[5m])

```

**P99 latency by tier:**

```promql
engram_spreading_latency_hot_seconds{quantile="0.99"}
engram_spreading_latency_warm_seconds{quantile="0.99"}
engram_spreading_latency_cold_seconds{quantile="0.99"}

```

**Failure rate:**

```promql
rate(engram_spreading_failures_total[5m]) /
rate(engram_spreading_activations_total[5m])

```

### Consolidation Health

**Consolidation success ratio:**

```promql
rate(engram_consolidation_runs_total[5m]) /
(rate(engram_consolidation_runs_total[5m]) + rate(engram_consolidation_failures_total[5m]))

```

**Novelty trend (5-minute average):**

```promql
avg_over_time(engram_consolidation_novelty_gauge[5m])

```

### Activation Pool Efficiency

**Pool efficiency score (weighted average of hit rate and utilization):**

```promql
activation_pool_hit_rate * 0.7 + activation_pool_utilization * 0.3

```

**Reuse ratio:**

```promql
activation_pool_total_reused /
(activation_pool_total_created + activation_pool_total_reused)

```

## Grafana Dashboards

Engram includes pre-built dashboards for common monitoring scenarios.

### System Overview Dashboard

**Purpose:** High-level health monitoring
**Panels:**

- Service availability status

- Request rate and error rate

- P50/P99 latency trends

- Resource utilization (CPU, memory)

**Use When:** Daily health checks, incident triage

### Memory Operations Dashboard

**Purpose:** Track memory store activity
**Panels:**

- Store/recall/delete operation rates

- Operation latency distributions

- Success vs error ratios

- Active memory count growth

**Use When:** Investigating performance issues, capacity planning

### Storage Tiers Dashboard

**Purpose:** Monitor storage tier health
**Panels:**

- Tier utilization gauges

- Migration flow diagram

- WAL size and lag

- Compaction activity

**Use When:** Storage performance tuning, capacity management

### Spreading Activation Dashboard

**Purpose:** Deep dive into spreading activation performance
**Panels:**

- Activation latency heatmap

- Circuit breaker state timeline

- Pool hit rate and utilization

- GPU vs CPU operation ratio

**Use When:** Performance optimization, debugging latency issues

## Log Aggregation with Loki

### Log Format

Engram emits structured JSON logs:

```json
{
  "timestamp": "2025-10-27T12:34:56.789Z",
  "level": "INFO",
  "target": "engram_core::engine",
  "message": "Memory stored successfully",
  "memory_space": "agent-123",
  "operation": "store",
  "duration_ms": 12.5,
  "memory_id": "mem_abc123"
}

```

### Common LogQL Queries

**All errors in the last hour:**

```logql
{job="engram", level="ERROR"} |= "" | json

```

**Slow operations (>1s):**

```logql
{job="engram"} | json | duration_ms > 1000

```

**Consolidation failures with context:**

```logql
{job="engram", target="engram_core::consolidation"} |= "failed" | json

```

**Memory operations by memory_space:**

```logql
{job="engram"} | json | memory_space="agent-123" | operation="recall"

```

### Log Retention

Default retention: 30 days (720 hours)

Configure in `deployments/loki/loki-config.yml`:

```yaml
limits_config:
  retention_period: 720h

```

## Alert Configuration

Alerts are defined in `deployments/prometheus/alerts.yml`.

### Critical Alerts (Severity: critical)

**EngramDown** - Service unreachable for 1 minute
**Response:** Check pod status, container logs, liveness probe

**ActivationPoolExhaustion** - <10 available pool records
**Response:** Scale up pool size, investigate memory leaks

**ConsolidationFailureStreak** - 3+ failures in 15 minutes
**Response:** Check storage tier health, review consolidation logs

### Warning Alerts (Severity: warning)

**SpreadingLatencySLOBreach** - P90 latency >100ms for 5m
**Response:** Review query patterns, check GPU utilization, tune batch sizes

**ConsolidationStaleness** - Snapshot >15 minutes old
**Response:** Check consolidation scheduler, verify background worker health

**ActivationPoolLowHitRate** - Hit rate <50% for 15m
**Response:** Increase pool size, investigate workload changes

### Informational Alerts (Severity: info)

**ConsolidationNoveltyStagnation** - Novelty <0.01 for 30m
**Response:** Verify input activity, check for steady state

**AdaptiveBatchingNotConverging** - Confidence <30% for 30m
**Response:** Review workload stability, adjust controller parameters

## Troubleshooting

### No metrics appearing in Prometheus

**Check Engram metrics endpoint:**

```bash
curl http://localhost:7432/metrics/prometheus

```

**Verify Prometheus scrape config:**

```bash
kubectl logs -n monitoring deployment/prometheus | grep "scrape"

```

**Check Prometheus targets:**
Open http://localhost:9090/targets

### Grafana shows "No data"

**Verify Prometheus datasource:**
Grafana > Configuration > Data Sources > Prometheus > Test

**Check Prometheus has data:**

```promql
up{job="engram"}

```

**Verify time range:**
Ensure dashboard time range covers period when Engram was running

### Logs not appearing in Loki

**Check Promtail is running:**

```bash
kubectl logs -n monitoring daemonset/promtail

```

**Verify Loki ingestion:**

```bash
curl http://localhost:3100/loki/api/v1/label/__name__/values

```

**Test log query:**

```bash
curl -G -s "http://localhost:3100/loki/api/v1/query_range" \
  --data-urlencode 'query={job="engram"}' \
  --data-urlencode 'limit=10'

```

### High cardinality warnings

**Check series count:**

```promql
count({__name__=~".+"})

```

**Identify high-cardinality metrics:**

```bash
curl http://localhost:9090/api/v1/status/tsdb | jq '.data.seriesCountByMetricName'

```

**Solution:** Review label usage, ensure `memory_space` label is used judiciously

## Performance Tuning

### Metrics Collection Overhead

Target: <1% CPU overhead

**Measure overhead:**

```bash
# Compare CPU usage with metrics export enabled vs disabled
./scripts/profile_hotspots.sh --metrics-overhead

```

**Reduce overhead:**

- Increase scrape interval (15s → 30s)

- Disable detailed histogram buckets for low-priority metrics

- Use recording rules for expensive aggregations

### Prometheus Resource Usage

Expected resource consumption:

- CPU: <5% for 10,000 series

- RAM: <500MB for 15-day retention

- Disk: ~1GB per million samples

**Optimize storage:**

```yaml
# prometheus.yml
global:
  scrape_interval: 30s  # Increase from 15s
storage:
  tsdb:
    retention.time: 7d  # Reduce from 15d if not needed

```

### Loki Query Performance

**Enable query result caching:**

```yaml
# loki-config.yml
query_range:
  results_cache:
    cache:
      enable_fifocache: true
      fifocache:
        max_size_mb: 1024
        validity: 24h

```

**Use indexed labels:**
Only add frequently-queried fields as labels (level, target, memory_space)

## Validation

Validate monitoring stack deployment:

```bash
# Check all required metrics are exposed
./scripts/validate_metric_coverage.sh

# Verify Prometheus is scraping
curl http://localhost:9090/api/v1/targets | jq '.data.activeTargets[] | select(.health!="up")'

# Test Grafana dashboards load
curl -u admin:admin http://localhost:3000/api/dashboards/uid/engram-system-overview

# Verify Loki is ingesting logs
curl "http://localhost:3100/loki/api/v1/query_range?query={job=\"engram\"}&limit=1"

```

## Best Practices

1. **Set up alerts before going to production** - Configure PagerDuty/Slack integration

2. **Baseline your metrics** - Run soak tests to understand normal ranges

3. **Use recording rules** - Pre-compute expensive queries for dashboards

4. **Monitor the monitors** - Set up alerts for Prometheus/Grafana/Loki health

5. **Version dashboards** - Store dashboard JSON in git for reproducibility

6. **Tune scrape intervals** - Balance freshness vs resource overhead

7. **Use labels sparingly** - Each unique label combination creates a new series

8. **Correlate metrics with logs** - Use Grafana Explore to link metrics and logs

9. **Regular retention review** - Adjust retention based on actual query patterns

10. **Test alert rules** - Use chaos engineering to validate alerts fire correctly

## Next Steps

- [Alerting Guide](alerting.md) - Detailed alert response procedures

- [Performance Tuning](performance-tuning.md) - Optimize based on metrics

- [Troubleshooting](troubleshooting.md) - Diagnose issues using metrics and logs

- [Capacity Planning](scaling.md) - Plan scaling based on metric trends
