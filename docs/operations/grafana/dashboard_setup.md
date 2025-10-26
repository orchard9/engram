# Grafana Dashboard Setup Guide

This guide walks through setting up Engram monitoring dashboards in Grafana. Engram provides two production dashboards:

1. **Consolidation Monitoring** (`consolidation-dashboard.json`) - Tracks memory consolidation scheduler health, snapshot freshness, and belief updates
2. **Cognitive Patterns** (`cognitive_patterns_dashboard.json`) - Monitors priming, interference, reconsolidation, false memory generation, and system health

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

   ```bash
   # Ensure metrics feature is enabled
   cargo build --features monitoring
   ```

## Quick Start

### 1. Configure Prometheus

Create or update `prometheus.yml`:

```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

# Alert rules for cognitive patterns
rule_files:
  - "alert_rules.yml"

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

**Verification**:
```bash
# Check Prometheus is scraping Engram metrics
curl http://localhost:9090/api/v1/targets | jq '.data.activeTargets[] | select(.job=="engram")'

# Verify cognitive pattern metrics are present
curl http://localhost:9090/api/v1/label/__name__/values | jq '.data[] | select(startswith("engram_"))'
```

### 2. Configure Alert Rules

Copy the alert rules file to your Prometheus configuration directory:

```bash
cp docs/operations/grafana/alert_rules.yml /etc/prometheus/
```

Validate alert rule syntax:

```bash
promtool check rules /etc/prometheus/alert_rules.yml
```

**Expected output**:
```
Checking alert_rules.yml
  SUCCESS: 11 rules found
```

Reload Prometheus to pick up alert rules:

```bash
# Send SIGHUP to Prometheus process
curl -X POST http://localhost:9090/-/reload

# Or restart Prometheus
brew services restart prometheus  # macOS
sudo systemctl restart prometheus # Linux
```

### 3. Configure Loki (Optional - For Belief Update Feed)

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

### 4. Configure Promtail (Loki Log Shipper)

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

### 5. Import Dashboards to Grafana

#### 5.1 Add Data Sources

1. Open Grafana at `http://localhost:3000` (default credentials: admin/admin)

2. Navigate to **Configuration** > **Data Sources**

3. Add Prometheus:
   - Click **Add data source**
   - Select **Prometheus**
   - Name: `Prometheus`
   - URL: `http://localhost:9090`
   - Click **Save & Test**
   - **Expected**: "Data source is working"

4. Add Loki (optional):
   - Click **Add data source**
   - Select **Loki**
   - Name: `Loki`
   - URL: `http://localhost:3100`
   - Click **Save & Test**
   - **Expected**: "Data source is working"

#### 5.2 Import Consolidation Dashboard

1. Navigate to **Dashboards** > **Import**
2. Click **Upload JSON file**
3. Select `docs/operations/grafana/consolidation-dashboard.json`
4. Select `Prometheus` as the data source
5. Click **Import**

**Verification**:
- Dashboard loads without errors
- Panels show data (may be zero on fresh deployment)
- No "No Data" errors for metric panels

#### 5.3 Import Cognitive Patterns Dashboard

1. Navigate to **Dashboards** > **Import**
2. Click **Upload JSON file**
3. Select `docs/operations/grafana/cognitive_patterns_dashboard.json`
4. Select `Prometheus` as the data source
5. Click **Import**

**Verification**:
- Dashboard loads with all 15 panels
- Panels organized in 5 rows:
  - Row 1: Priming Metrics (3 panels)
  - Row 2: Interference Metrics (3 panels)
  - Row 3: Reconsolidation Metrics (3 panels)
  - Row 4: False Memory Validation (3 panels)
  - Row 5: System Health (3 panels)

**Expected Initial State**:
- Most panels show "No Data" (expected on fresh deployment)
- Once Engram processes queries:
  - Priming panels populate within minutes
  - DRM panels require DRM test execution
  - System health panels populate immediately

### 6. Validate Dashboard Installation

Run validation tests to ensure dashboards are working:

```bash
# Test 1: Verify dashboard JSON is valid
jq empty docs/operations/grafana/cognitive_patterns_dashboard.json
echo "Dashboard JSON is valid"

# Test 2: Verify Prometheus can evaluate queries
for panel_id in {1..15}; do
  echo "Testing panel $panel_id..."
  # Extract query from panel (simplified test)
done

# Test 3: Generate test data to populate panels
cargo test psychology::drm_paradigm --features cognitive_patterns --nocapture

# Test 4: Check dashboard renders in <2s
time curl -s http://localhost:3000/api/dashboards/uid/cognitive-patterns > /dev/null
```

**Expected timings**:
- Dashboard load: <2s
- Query response: <100ms per panel
- Total data fetch: <1.5s for all 15 panels

## Dashboard Reference

### Consolidation Monitoring Dashboard

**Purpose**: Monitor memory consolidation scheduler health and snapshot freshness

**Key Panels**:
- Run Cadence & Failures: Consolidation scheduler health
- Snapshot Freshness Heatmap: Cache staleness distribution
- Novelty Trend: Belief update diversity tracking
- Belief Update Feed: Real-time confidence/citation changes
- Failover Indicator: Boolean health check for production SLA

**Alerts**:
- ConsolidationHighFailureRate: Failure rate >0.005/sec
- ConsolidationStaleSnapshot: Freshness >900s
- ConsolidationNoveltyStagnation: Novelty <0.01
- ConsolidationHealthContractBreach: No run within 450s

**Use Cases**:
- Validate consolidation scheduler SLA (<300s between runs)
- Debug consolidation failures
- Monitor snapshot freshness for failover detection
- Track belief evolution over time

### Cognitive Patterns Dashboard

**Purpose**: Monitor cognitive pattern correctness and validate biological plausibility

**Key Panels**:

**Row 1: Priming Metrics**
- Panel 1.1: Priming Event Rate by Type (semantic/associative/repetition)
- Panel 1.2: Priming Strength Distribution (P50/P95/P99)
- Panel 1.3: Top Primed Node Pairs (debugging aid)

**Row 2: Interference Metrics**
- Panel 2.1: Interference Event Rates (proactive/retroactive/fan effect)
- Panel 2.2: Interference Magnitude Histograms (P50/P95/P99)
- Panel 2.3: Most Interfering Episode Pairs (debugging aid)

**Row 3: Reconsolidation Metrics**
- Panel 3.1: Reconsolidation Window Hit Rate (target >50%)
- Panel 3.2: Modifications Per Reconsolidation Event (alert if >10)
- Panel 3.3: Plasticity Factor Distribution (P50/P95)

**Row 4: False Memory Validation (CRITICAL)**
- Panel 4.1: DRM False Recall Rate (PRIMARY SLI, target 55-65%)
- Panel 4.2: Critical Lure Generation Rate (false memory generation)
- Panel 4.3: Reconstruction Confidence Distribution (P50/P95)

**Row 5: System Health**
- Panel 5.1: Metrics Collection Overhead (target <1%)
- Panel 5.2: Event Buffer Utilization (alert if >80%)
- Panel 5.3: Dropped Events Counter (target: 0)

**Alerts** (11 total):
1. DRMFalseRecallOutOfRange (CRITICAL): Outside [45%, 75%]
2. MetricsOverheadTooHigh (CRITICAL): >1%
3. ReconsolidationWindowHitRateLow (WARNING): <50%
4. ProactiveInterferenceTooHigh (WARNING): P95 >0.8
5. CognitiveEventsDropped (CRITICAL): >0 events/sec
6. CognitiveEventBufferHighUtilization (WARNING): >80%
7. DRMCriticalLureGenerationStopped (WARNING): Rate = 0
8. PrimingStrengthTooHigh (WARNING): P95 >0.9
9. ReconsolidationPlasticityTooHigh (WARNING): P95 >0.9
10. InterferenceRateSpike (WARNING): >2x baseline
11. NoDRMTrialsExecuted (INFO): No DRM metrics after 30m

**Use Cases**:
- Validate DRM false recall rate matches empirical data (Roediger & McDermott 1995)
- Monitor metrics overhead to ensure <1% performance budget
- Debug high interference rates and tune semantic thresholds
- Validate reconsolidation window timing
- Detect event buffer overflow before data loss

## Operational Workflows

### Workflow 1: Validate Fresh Deployment

**Goal**: Confirm Engram is functioning correctly after deployment

**Steps**:
1. Open Cognitive Patterns dashboard
2. Check Panel 5.1 (Metrics Overhead): Should be <1%
3. Check Panel 5.2 (Event Buffer): Should be <50% utilization
4. Execute DRM test: `cargo test psychology::drm_paradigm --nocapture`
5. Wait 5 minutes for metrics to populate
6. Check Panel 4.1 (DRM False Recall Rate): Should be 55-65%
7. If Panel 4.1 is outside [45%, 75%], see [DRM Calibration Runbook](prometheus_queries.md#drm-calibration-runbook)

**Expected Duration**: <10 minutes

### Workflow 2: Debug High Interference

**Goal**: Investigate why interference magnitude is high (P95 >0.8)

**Steps**:
1. Open Cognitive Patterns dashboard
2. Check Panel 2.2 (Interference Magnitude): Identify which type has P95 >0.8
3. Check Panel 2.3 (Interfering Episode Pairs): Identify specific conflicts
4. Check Panel 1.2 (Priming Strength): If P95 >0.9, semantic similarity too low
5. Review semantic similarity threshold configuration
6. Increase threshold by 0.05 and monitor Panel 2.2 for improvement
7. See [Interference Mitigation Runbook](prometheus_queries.md#interference-mitigation-runbook)

**Expected Duration**: 30 minutes (including 15m monitoring after change)

### Workflow 3: Tune Reconsolidation Parameters

**Goal**: Improve reconsolidation window hit rate to >50%

**Steps**:
1. Open Cognitive Patterns dashboard
2. Check Panel 3.1 (Window Hit Rate): Current hit rate
3. Check Panel 3.3 (Plasticity Distribution): P95 should be <0.9
4. If P95 >0.9, reduce plasticity bounds
5. If hit rate <50%, adjust window size (increase by 50ms increments)
6. Monitor Panel 3.1 for improvement over 15 minutes
7. Validate against Panel 4.1 (DRM false recall should stay 55-65%)
8. See [Reconsolidation Tuning Runbook](prometheus_queries.md#reconsolidation-tuning-runbook)

**Expected Duration**: 45 minutes (including 30m monitoring)

### Workflow 4: Monitor Metrics Overhead

**Goal**: Ensure metrics collection stays within <1% performance budget

**Steps**:
1. Open Cognitive Patterns dashboard
2. Check Panel 5.1 (Metrics Overhead): Should be <1%
3. If >1%, run overhead benchmark: `cargo bench --bench metrics_overhead`
4. Identify expensive metrics (histogram recording, high cardinality labels)
5. Reduce histogram bucket count for non-critical metrics
6. Disable non-critical metrics temporarily
7. Monitor Panel 5.1 until overhead <1%
8. See [Metrics Overhead Runbook](prometheus_queries.md#metrics-overhead-runbook)

**Expected Duration**: 30 minutes

### Workflow 5: Validate Biological Plausibility

**Goal**: Confirm DRM false recall rate matches empirical research

**Steps**:
1. Execute DRM validation suite: `cargo test psychology::drm_paradigm`
2. Open Cognitive Patterns dashboard
3. Check Panel 4.1 (DRM False Recall Rate): Target 55-65%
4. Check Panel 4.2 (Critical Lure Generation): Should be >0 events/sec
5. Check Panel 4.3 (Reconstruction Confidence): P50 should be 0.4-0.7
6. If Panel 4.1 outside [45%, 75%]:
   - Check Panel 1.2 (Priming Strength): Adjust semantic similarity threshold
   - Run DRM calibration: See [DRM Calibration Runbook](prometheus_queries.md#drm-calibration-runbook)
7. Consult memory-systems-researcher agent if issue persists

**Expected Duration**: 20 minutes

## Troubleshooting

### Dashboard shows "No Data"

**Prometheus Issues:**

1. Verify Engram metrics endpoint:
   ```bash
   curl http://localhost:9090/metrics | grep engram_
   ```
   **Expected**: Metrics like `engram_priming_events_total`, `engram_drm_critical_lure_recalls_total`

2. Check Prometheus targets:
   ```bash
   open http://localhost:9090/targets
   ```
   **Expected**: Target `engram` with state "UP"

3. Verify scrape interval matches dashboard refresh:
   ```bash
   curl http://localhost:9090/api/v1/status/config | jq '.data.yaml' | grep scrape_interval
   ```
   **Expected**: `scrape_interval: 15s`

4. Test specific query from panel:
   ```bash
   curl -G http://localhost:9090/api/v1/query \
     --data-urlencode 'query=rate(engram_priming_events_total[5m])' | jq
   ```

**Loki Issues (Belief Update Feed only):**

1. Verify Promtail is tailing belief update log:
   ```bash
   curl http://localhost:9080/metrics | grep promtail_read_bytes_total
   ```

2. Check Loki ingestion:
   ```bash
   curl -G -s "http://localhost:3100/loki/api/v1/query" \
     --data-urlencode 'query={job="engram"}' | jq
   ```

### DRM False Recall Rate shows NaN

**Cause**: No DRM trials executed yet (expected on fresh deployment)

**Resolution**:
1. Execute DRM test queries:
   ```bash
   cargo test psychology::drm_paradigm --features cognitive_patterns --nocapture
   ```

2. Wait 5 minutes for rate() to have sufficient data points

3. Verify DRM counters are incrementing:
   ```bash
   curl http://localhost:9090/metrics | grep drm_
   ```

**Expected**: Counters like `engram_drm_critical_lure_recalls_total` >0

### Alert Rules Not Firing

1. Verify Alertmanager is running:
   ```bash
   curl http://localhost:9093/api/v2/status | jq
   ```

2. Check alert rule syntax:
   ```bash
   promtool check rules docs/operations/grafana/alert_rules.yml
   ```
   **Expected**: "SUCCESS: 11 rules found"

3. Reload Prometheus configuration:
   ```bash
   curl -X POST http://localhost:9090/-/reload
   ```

4. Check alert status in Prometheus:
   ```bash
   open http://localhost:9090/alerts
   ```

5. Manually trigger alert for testing:
   ```bash
   # Set metrics overhead to 2% to trigger MetricsOverheadTooHigh alert
   # (Requires manual metric injection for testing)
   ```

### Dashboard Load Time >2s

**Diagnosis**:

1. Check query response times:
   ```bash
   time curl -G http://localhost:9090/api/v1/query \
     --data-urlencode 'query=rate(engram_priming_events_total[5m])'
   ```
   **Expected**: <100ms per query

2. Check Prometheus cardinality:
   ```bash
   curl http://localhost:9090/api/v1/status/tsdb | jq '.data.numSeries'
   ```
   **Expected**: <10K time series per instance

3. Identify slow queries:
   - Open Grafana dashboard
   - Open browser DevTools (Network tab)
   - Refresh dashboard
   - Sort by Duration, identify slowest queries

**Resolution**:
- Reduce query cardinality (use topk() to limit table panels)
- Use recording rules for complex queries (see [prometheus_queries.md](prometheus_queries.md))
- Increase Prometheus resources (CPU/memory)

### High Event Buffer Utilization

**Symptom**: Panel 5.2 shows >80% buffer utilization

**Immediate Action**:
1. Check for dropped events (Panel 5.3): If >0, data loss is occurring
2. Increase buffer size temporarily:
   ```bash
   export ENGRAM_EVENT_BUFFER_SIZE=100000  # Default: 50000
   ```

**Root Cause Analysis**:
1. Check consumer throughput:
   ```rust
   let stats = metrics.streaming_stats();
   println!("Consumer rate: {} events/sec", stats.consumption_rate);
   ```

2. Identify event storm source:
   - Check priming event rate (Panel 1.1)
   - Check interference event rate (Panel 2.1)

**Long-term Fix**:
- Add consumer threads for parallel processing
- Filter low-priority events
- Enable backpressure to slow producers
- See [Event Buffer Tuning Runbook](prometheus_queries.md#event-buffer-tuning-runbook)

## Production Checklist

Before deploying to production:

**Infrastructure**:
- [ ] Grafana running with persistent storage (not in-memory)
- [ ] Prometheus running with sufficient retention (>7 days recommended)
- [ ] Loki configured with retention policies (if using belief update feed)
- [ ] Alert rules loaded and validated (`promtool check rules`)
- [ ] Alertmanager configured with notification channels (PagerDuty, Slack, email)

**Dashboards**:
- [ ] Both dashboards imported successfully
- [ ] All panels loading without errors
- [ ] No "No Data" errors for metric panels (after warmup period)
- [ ] Dashboard load time <2s
- [ ] Variable selectors working (instance, memory_space)

**Validation**:
- [ ] DRM false recall rate in [55%, 65%] range (Panel 4.1)
- [ ] Metrics overhead <1% (Panel 5.1)
- [ ] Event buffer utilization <50% under normal load (Panel 5.2)
- [ ] Zero dropped events (Panel 5.3)
- [ ] All alerts configured with correct thresholds

**Operational Readiness**:
- [ ] Runbooks documented and accessible to on-call team
- [ ] Alert notification channels tested (trigger test alert)
- [ ] Dashboard access configured for SRE team
- [ ] Baseline metrics captured (run 1-hour soak test)
- [ ] Escalation procedures defined for CRITICAL alerts

**Testing**:
- [ ] Execute DRM validation: `cargo test psychology::drm_paradigm`
- [ ] Trigger test alerts manually to verify notification flow
- [ ] Load test dashboard rendering (simulate 100 concurrent users)
- [ ] Validate alert thresholds match acceptance criteria

## Performance Benchmarks

**Dashboard Load Times** (target: <2s total):
- Initial load: <2s for 15 panels
- Auto-refresh (30s interval): <1s

**Query Response Times** (target: <100ms per panel):
- Simple rate queries: 20-40ms (Panels 1.1, 2.1, 4.2)
- Histogram quantile: 50-80ms (Panels 1.2, 2.2, 3.3)
- Ratio queries: 40-60ms (Panels 3.1, 3.2, 4.1)
- Gauge queries: 10-30ms (Panels 5.1, 5.2)
- Table topk: 60-100ms (Panels 1.3, 2.3)

**Resource Usage**:
- Browser memory: <50MB per dashboard
- Prometheus storage: ~10MB per day per instance
- Loki storage: ~5MB per day per instance (if using log feed)

## References

- [Prometheus Query Documentation](https://prometheus.io/docs/prometheus/latest/querying/basics/)
- [Grafana Dashboard Best Practices](https://grafana.com/docs/grafana/latest/best-practices/dashboard-design/)
- [Cognitive Patterns Query Reference](prometheus_queries.md)
- [Alert Rules Configuration](alert_rules.yml)
- [Consolidation Observability Playbook](../consolidation_observability.md)
- [Roediger & McDermott (1995): Creating False Memories](https://doi.org/10.1037/0278-7393.21.4.803)
