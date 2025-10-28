# Grafana Dashboard Setup Guide

This guide walks through setting up the Engram Consolidation Monitoring dashboard in Grafana.

## Prerequisites

1. **Grafana** (v9.0+)

   ```bash
   # macOS
   brew install grafana
   brew services start grafana

   # Linux
   sudo apt-get install -y grafana
   sudo systemctl start grafana-server
   ```

2. **Prometheus** (v2.30+)

   ```bash
   # macOS
   brew install prometheus

   # Linux
   sudo apt-get install -y prometheus
   ```

3. **Loki** (v2.0+) - For JSONL log tailing

   ```bash
   # macOS
   brew install loki

   # Linux
   wget https://github.com/grafana/loki/releases/download/v2.8.0/loki-linux-amd64.zip
   unzip loki-linux-amd64.zip
   chmod +x loki-linux-amd64
   ```

4. **Engram server** running with metrics enabled

## Quick Start

### 1. Configure Prometheus

Create or update `prometheus.yml`:

```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'engram'
    static_configs:
      - targets: ['localhost:9090']  # Adjust to your Engram metrics port
    metrics_path: '/metrics'

```

Start Prometheus:

```bash
prometheus --config.file=prometheus.yml

```

### 2. Configure Loki

Create `loki-config.yaml`:

```yaml
auth_enabled: false

server:
  http_listen_port: 3100

ingester:
  lifecycler:
    address: 127.0.0.1
    ring:
      kvstore:
        store: inmemory
      replication_factor: 1
  chunk_idle_period: 5m
  chunk_retain_period: 30s

schema_config:
  configs:
    - from: 2020-05-15
      store: boltdb
      object_store: filesystem
      schema: v11
      index:
        prefix: index_
        period: 168h

storage_config:
  boltdb:
    directory: /tmp/loki/index
  filesystem:
    directory: /tmp/loki/chunks

limits_config:
  enforce_metric_name: false
  reject_old_samples: true
  reject_old_samples_max_age: 168h

chunk_store_config:
  max_look_back_period: 0s

table_manager:
  retention_deletes_enabled: false
  retention_period: 0s

```

Start Loki:

```bash
loki -config.file=loki-config.yaml

```

### 3. Configure Promtail (Loki Log Shipper)

Create `promtail-config.yaml`:

```yaml
server:
  http_listen_port: 9080
  grpc_listen_port: 0

positions:
  filename: /tmp/positions.yaml

clients:
  - url: http://localhost:3100/loki/api/v1/push

scrape_configs:
  - job_name: engram_consolidation
    static_configs:
      - targets:
          - localhost
        labels:
          job: engram
          __path__: /Users/jordanwashburn/Workspace/orchard9/engram/engram-core/data/consolidation/alerts/belief_updates.jsonl

```

Start Promtail:

```bash
promtail -config.file=promtail-config.yaml

```

### 4. Import Dashboard to Grafana

1. Open Grafana at `http://localhost:3000` (default credentials: admin/admin)

2. Add Data Sources:
   - Navigate to **Configuration** > **Data Sources**
   - Add Prometheus:
     - Name: `Prometheus`
     - URL: `http://localhost:9090`
     - Click **Save & Test**
   - Add Loki:
     - Name: `Loki`
     - URL: `http://localhost:3100`
     - Click **Save & Test**

3. Import Dashboard:
   - Navigate to **Dashboards** > **Import**
   - Click **Upload JSON file**
   - Select `docs/operations/grafana/consolidation-dashboard.json`
   - Select Prometheus as the data source
   - Click **Import**

### 5. Run Baseline Soak Test

Before production deployment, capture a fresh 1-hour baseline:

```bash
# From the engram project root:
./target/debug/consolidation-soak \
  --duration-secs 3600 \
  --scheduler-interval-secs 300 \
  --sample-interval-secs 60 \
  --output-dir docs/assets/consolidation/baseline

```

**Options Explained:**

- `--duration-secs 3600`: Run for 1 hour (3600 seconds)

- `--scheduler-interval-secs 300`: Consolidation runs every 5 minutes (default SLA)

- `--sample-interval-secs 60`: Sample metrics every minute

- `--output-dir`: Where to write baseline artifacts

**Baseline Artifacts Generated:**

- `metrics.jsonl`: Time-series metrics (consolidation runs, novelty, freshness, citations)

- `snapshots.jsonl`: Consolidated belief snapshots

- `belief_updates.jsonl`: Detailed confidence/citation changes per pattern

**Expected Results:**

- 12 consolidation runs (1 hour / 5 minutes)

- Novelty gauge: Typically 0.05-0.15 (heterogeneous updates)

- Freshness: <10s average (with 300s scheduler interval)

- Citation churn: <30% (stable consolidation)

## Dashboard Widgets Reference

### 1. Run Cadence & Failures

- **Purpose**: Monitor consolidation scheduler health

- **Metrics**: `rate(engram_consolidation_runs_total[5m])`, `rate(engram_consolidation_failures_total[5m])`

- **Alert**: Triggers when failure rate >0.005/sec sustained over 5 minutes

- **Healthy State**: Consistent run cadence every 300s, zero failures

### 2. Snapshot Freshness Heatmap

- **Purpose**: Visualize cache staleness distribution

- **Metric**: `engram_consolidation_freshness_seconds`

- **Percentile Bands**: p50, p90, p99

- **Healthy State**: <450s (1.5x scheduler interval)

- **Failover Threshold**: >900s indicates health contract breach

### 3. Novelty Trend & Stagnation Detection

- **Purpose**: Track belief update diversity

- **Metric**: `engram_consolidation_novelty_gauge`

- **Thresholds**:
  - Stagnation: <0.01 (uniform updates, potential issue)
  - Heterogeneous: >0.1 (diverse pattern changes)

- **Healthy State**: 0.05-0.15 range

### 4. Belief Update Feed

- **Purpose**: Real-time confidence/citation change stream

- **Data Source**: Loki tailing `data/consolidation/alerts/belief_updates.jsonl`

- **Fields**: timestamp, pattern_id, old_confidence, new_confidence, citation_delta

- **Use Case**: Debug unexpected pattern changes, trace belief evolution

### 5. Failover Indicator & Health Status

- **Purpose**: Boolean health check for production SLA

- **Conditions**:
  - Health Contract Breach: No snapshot within 450s (1.5x interval)
  - Snapshot Stale: Freshness >900s

- **States**: HEALTHY (green) / FAILOVER (red)

- **Action**: Red state triggers on-demand snapshot regeneration

### 6. Citation Count Trend

- **Purpose**: Track total citation count over time

- **Metric**: `engram_consolidation_citations_current`

- **Healthy State**: Gradually increasing or stable

- **Alert**: Sudden drops may indicate compaction issues

### 7. Storage Metrics

- **Purpose**: High-level consolidation statistics

- **Metrics**:
  - Runs/hour: `sum(rate(engram_consolidation_runs_total[1h]))`
  - Avg novelty (1h): `avg(engram_consolidation_novelty_gauge)`
  - Avg freshness (s): `avg(engram_consolidation_freshness_seconds)`

## Alerting Rules

Create `alerts.yml` for Prometheus Alertmanager:

```yaml
groups:
  - name: engram_consolidation
    interval: 30s
    rules:
      - alert: ConsolidationHighFailureRate
        expr: rate(engram_consolidation_failures_total[5m]) > 0.005
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "High consolidation failure rate detected"
          description: "Failure rate {{ $value }}/sec sustained over 5 minutes"

      - alert: ConsolidationStaleSnapshot
        expr: engram_consolidation_freshness_seconds > 900
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "Consolidation snapshot is stale"
          description: "Snapshot age {{ $value }}s exceeds 900s threshold"

      - alert: ConsolidationNoveltyStagnation
        expr: engram_consolidation_novelty_gauge < 0.01
        for: 15m
        labels:
          severity: warning
        annotations:
          summary: "Consolidation novelty stagnation detected"
          description: "Novelty {{ $value }} below 0.01 for 15 minutes"

      - alert: ConsolidationHealthContractBreach
        expr: (time() - (engram_consolidation_runs_total > 0)) > 450
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Consolidation health contract breached"
          description: "No consolidation run within 450s (1.5x interval)"

```

## Troubleshooting

### Dashboard shows "No Data"

**Prometheus Issues:**

1. Verify Engram metrics endpoint: `curl http://localhost:9090/metrics | grep consolidation`

2. Check Prometheus targets: `http://localhost:9090/targets`

3. Ensure scrape interval matches dashboard refresh rate

**Loki Issues:**

1. Verify Promtail is tailing belief update log:

   ```bash
   curl http://localhost:9080/metrics | grep promtail_read_bytes_total
   ```

2. Check Loki ingestion:

   ```bash
   curl -G -s "http://localhost:3100/loki/api/v1/query" \
     --data-urlencode 'query={job="engram"}' | jq
   ```

### Freshness Widget Shows Old Data

1. Check consolidation scheduler is running:

   ```bash
   curl http://localhost:9090/api/v1/consolidations
   ```

2. Verify scheduler interval configuration matches Prometheus scrape interval

3. Review engram server logs for scheduler errors

### Belief Update Feed is Empty

1. Ensure `data/consolidation/alerts/belief_updates.jsonl` exists and has recent entries:

   ```bash
   tail -f data/consolidation/alerts/belief_updates.jsonl
   ```

2. Verify Promtail is configured with correct file path (update `promtail-config.yaml` with absolute path)

3. Check Promtail logs: `journalctl -u promtail -f`

### Alert Rules Not Firing

1. Verify Alertmanager is running: `http://localhost:9093`

2. Check alert rule syntax:

   ```bash
   promtool check rules alerts.yml
   ```

3. Reload Prometheus configuration:

   ```bash
   curl -X POST http://localhost:9090/-/reload
   ```

## Production Checklist

Before deploying to production:

- [ ] Run full 1-hour soak test to establish baseline

- [ ] Configure alert notification channels (PagerDuty, Slack, email)

- [ ] Set up Grafana user authentication and access control

- [ ] Enable Prometheus remote write for long-term storage

- [ ] Configure Loki retention policies for belief update logs

- [ ] Document remediation runbooks in dashboard annotations

- [ ] Test failover scenarios (scheduler failure, network partition)

- [ ] Validate dashboard performance under production load

## References

- [Consolidation Observability Playbook](../consolidation_observability.md)

- [Metrics Schema Changelog](../../metrics-schema-changelog.md)

- [Prometheus Query Documentation](https://prometheus.io/docs/prometheus/latest/querying/basics/)

- [Grafana Dashboard Best Practices](https://grafana.com/docs/grafana/latest/best-practices/dashboard-design/)

- [Loki LogQL Reference](https://grafana.com/docs/loki/latest/logql/)
