# Engram Grafana Dashboards

Production-grade Grafana dashboards for monitoring Engram memory system operations.

## Dashboard Overview

### 1. System Overview (`system-overview.json`)
**Purpose**: High-level health and performance monitoring across the entire Engram system.

**Key Metrics**:
- Service health status (up/down)
- Pool utilization and available slots
- Circuit breaker state
- Overall operation rates (activations, consolidations, compactions)
- Error rates across all operations
- Resource utilization (pool hit rates, cache efficiency)
- Activation pool status (in-flight, available, high water mark)

**Use Cases**:
- First-line health check for operations team
- Capacity planning and resource allocation
- Identifying system-wide anomalies
- SLA compliance monitoring

### 2. Memory Operations (`memory-operations.json`)
**Purpose**: Deep dive into memory operation performance and cognitive dynamics.

**Key Metrics**:
- Hot/warm/cold tier activation latencies (P50, P90, P99)
- Operation throughput (ops/sec)
- Episode count and growth rate
- Confidence score distributions by tier
- Consolidation quality metrics (novelty delta, variance, citation churn)
- Adaptive batch sizes per tier
- SLO violations and guardrail hits
- Consolidation snapshot freshness

**Use Cases**:
- Performance tuning for spreading activation
- Memory consolidation effectiveness analysis
- Citation network evolution tracking
- Adaptive algorithm validation

### 3. Storage Tiers (`storage-tiers.json`)
**Purpose**: Monitor tiered storage management, capacity, and data lifecycle.

**Key Metrics**:
- Tier capacity usage (absolute bytes and percentage)
- Hot/warm/cold tier utilization
- Migration rates between tiers
- Cache hit rates (spreading pool, activation pool)
- WAL size and compaction efficiency
- Storage savings through compaction
- Episode removal through compaction
- Tier promotion/demotion rates (adaptive controller)

**Use Cases**:
- Capacity planning for each tier
- Storage optimization and cost management
- WAL health monitoring
- Data lifecycle management validation

### 4. API Performance (`api-performance.json`)
**Purpose**: Track API endpoint performance, errors, and request patterns.

**Key Metrics**:
- Service availability status
- Request rates per operation type
- Endpoint latencies by tier (P50, P90, P99)
- Error rates by endpoint
- Concurrent connection count
- Circuit breaker activity
- Rate limiting and throttling stats
- GPU launch and fallback rates
- Resource pool efficiency

**Use Cases**:
- API SLA compliance monitoring
- Client performance troubleshooting
- Load balancing optimization
- Rate limiting effectiveness

### 5. Cluster Health & Membership (`cluster-health.json`)
**Purpose**: Provide a single-pane-of-glass view of multi-node availability, breaker status, and latency budgets.

**Key Metrics**:
- Node heartbeat via `up{job="engram"}`
- Hot/warm/cold tier latency percentiles per instance
- Activation pool utilization/hit rate
- Consolidation freshness and guardrail violations
- Active Engram alerts surfaced from Prometheus `ALERTS`

### 6. Tenant Utilization (`tenant-utilization.json`)
**Purpose**: Break down throughput, latency, and storage usage per `memory_space` to catch noisy neighbors and enforce SLAs.

**Key Metrics**:
- Top-k activation throughput per tenant
- Latency percentiles per tier and memory_space
- Storage footprint (`engram_tier_size_bytes`) per tenant
- Error/SLO budget consumption per tenant
- Episode volumes for capacity planning

### 7. Replication & WAL (`replication-wal.json`)
**Purpose**: Track durability pipeline health including WAL lag, compaction efficiency, and recovery latency.

**Key Metrics**:
- `engram_wal_lag_seconds` (stat)
- WAL recovery duration quantiles
- Compaction attempts/successes and bytes reclaimed
- Storage saved via compaction
- WAL recovery failures per instance

### 8. Chaos Readiness (`chaos-readiness.json`)
**Purpose**: Validate alert coverage during chaos drills (network partitions, CPU spikes, slow queries, memory pressure).

**Key Metrics**:
- Live Prometheus alert states for critical chaos scenarios
- Operation latency histograms under stress
- Resource utilization (pool hit rates, breaker state)
- Failure counters for consolidation/WAL

### 9. Capacity & Cost Forecast (`capacity-forecast.json`)
**Purpose**: Forecast tier growth, utilization, and expected storage spend for upcoming weeks.

**Key Metrics**:
- Current tier usage by bytes
- `predict_linear` capacity forecast (7-day horizon)
- Estimated spend using per-tier $/byte multipliers
- Utilization ratio vs. tier capacity

### 10. API Client Insights (`api-clients.json`)
**Purpose**: Give customer success and security teams per-client throughput, latency, and quota visibility.

**Key Metrics**:
- `engram_api_requests_total` per client & operation
- Error-rate ratio using `engram_api_errors_total`
- Client latency percentiles (`engram_api_latency_seconds`)
- Quota utilization (`engram_client_quota_utilization`)
- Client/operation heatmap for traffic mix

## Metric Mappings

All dashboards use metrics exported by Engram's Prometheus exporter at `/metrics/prometheus`. Key metric categories:

### Spreading Activation
- `engram_spreading_activations_total` - Counter of total activations
- `engram_spreading_latency_hot_seconds` - Hot tier latency summary
- `engram_spreading_latency_warm_seconds` - Warm tier latency summary
- `engram_spreading_latency_cold_seconds` - Cold tier latency summary
- `engram_spreading_latency_budget_violations_total` - SLO violations
- `engram_spreading_failures_total` - Failed operations
- `engram_spreading_breaker_state` - Circuit breaker (0=closed, 1=open, 2=half-open)

### Consolidation
- `engram_consolidation_runs_total` - Successful consolidation runs
- `engram_consolidation_failures_total` - Failed consolidations
- `engram_consolidation_novelty_gauge` - Novelty delta
- `engram_consolidation_novelty_variance` - Pattern variance
- `engram_consolidation_citation_churn` - Citation change rate
- `engram_consolidation_freshness_seconds` - Snapshot age
- `engram_consolidation_citations_current` - Total citations

### Storage & Compaction
- `engram_compaction_attempts_total` - Compaction initiations
- `engram_compaction_success_total` - Successful compactions
- `engram_compaction_episodes_removed` - Episodes removed
- `engram_compaction_storage_saved_bytes` - Bytes reclaimed
- `engram_wal_recovery_duration_seconds` - WAL recovery latency
- `engram_wal_compaction_bytes_reclaimed` - WAL space reclaimed

### Activation Pool
- `activation_pool_available_records` - Available slots
- `activation_pool_in_flight_records` - Active operations
- `activation_pool_high_water_mark` - Peak usage
- `activation_pool_hit_rate` - Cache efficiency

### Adaptive Batching
- `adaptive_batch_hot_size` / `warm_size` / `cold_size` - Batch sizes per tier
- `adaptive_batch_hot_confidence` / `warm_confidence` / `cold_confidence` - Convergence confidence
- `adaptive_guardrail_hits_total` - Guardrail activations
- `adaptive_topology_changes_total` - Tier promotion/demotion

## Installation

### Docker Compose
Dashboards are automatically provisioned when using the monitoring stack:

```bash
docker-compose --profile monitoring up
```

Grafana will be available at `http://localhost:3000` (default credentials: admin/admin).

### Manual Import
1. Open Grafana UI
2. Navigate to Dashboards > Import
3. Upload JSON file or paste JSON content
4. Select Prometheus datasource
5. Click Import

## Dashboard Features

### Template Variables
All dashboards include the following template variables:
- `datasource` - Select the Prometheus data source (defaults to the provisioned Engram source)
- `instance` - Regex filter across Prometheus `instance` labels so you can compare individual Engram nodes
- `memory_space` - Regex filter for memory-space-aware metrics (tenancy dashboards rely on Engram's `memory_space` label; on boards without that label the filter harmlessly returns "All")
- `client` - Available on the API Client Insights dashboard for per-client slicing

Future enhancements will add a `tier` template variable for storage dashboards once tier labels are exposed directly in the Prometheus exporter.

### Refresh Intervals
- Default: 10 seconds auto-refresh
- Available: 5s, 10s, 30s, 1m, 5m, 15m, 30m, 1h

### Time Ranges
- Default: Last 1 hour
- Customizable via time picker

### Alert Annotations
All dashboards support Grafana alert annotations, which will appear as vertical lines on graphs when alerts fire.

## Customization

### Adding Panels
Each dashboard follows Grafana's standard panel structure:
- Panel IDs must be unique within a dashboard
- Grid positions use a 24-column layout
- Heights are in grid units (1 unit = ~30px)

### Modifying Queries
All PromQL queries use 5-minute rate windows (`[5m]`) aligned with the Prometheus scrape interval. Adjust based on your scrape configuration.

### Threshold Tuning
Color thresholds are based on production SLOs:
- Latency: Green < 50ms, Yellow < 100ms, Red >= 100ms
- Error Rate: Green < 1%, Yellow < 5%, Red >= 5%
- Utilization: Green < 70%, Yellow < 90%, Red >= 90%

Adjust these in the field config for each panel based on your requirements.

## Troubleshooting

### No Data Appearing
1. Verify Prometheus is scraping Engram: Check `up{job="engram"}` metric
2. Confirm metrics endpoint is accessible: `curl http://engram:7432/metrics/prometheus`
3. Check Prometheus targets: `http://prometheus:9090/targets`
4. Verify time range covers available data

### Missing Metrics
Some panels may show "No data" if certain features are not yet instrumented:
- API-specific metrics (REST/gRPC endpoint labels)
- Per-tenant metrics (memory_space labels)
- Storage tier capacity metrics (if not configured)

These will be added as the instrumentation evolves.

### Performance Issues
If dashboards are slow:
1. Increase Prometheus retention and query timeout
2. Use recording rules for expensive queries (see `recording_rules.yml`)
3. Reduce auto-refresh interval
4. Narrow time range

## Future Enhancements

### Planned Metrics
- Per-endpoint API metrics with method/path labels
- Per-tenant resource usage (memory_space label)
- GPU memory utilization and kernel launch latencies
- Network I/O by tier
- Query complexity distributions

### Planned Dashboards
- **GPU Acceleration**: CUDA kernel performance, memory transfers, fallback analysis
- **Multi-Tenancy**: Per-tenant resource usage, isolation metrics, quota enforcement
- **Cognitive Dynamics**: Memory decay curves, pattern completion success rates, system 2 reasoning

### Advanced Features
- Anomaly detection using Grafana ML
- Predictive capacity alerts
- Custom SLO tracking with error budgets
- Distributed tracing integration (Tempo)
- Log correlation (Loki)

## Validation

All JSON files have been validated:
- Valid JSON syntax
- Unique panel IDs within each dashboard
- Proper Grafana schema version (38)
- Prometheus datasource compatibility

## Support

For issues or questions:
1. Check Engram metrics documentation: `docs/operations/monitoring.md`
2. Verify Prometheus configuration: `deployments/prometheus/prometheus.yml`
3. Review alert rules: `deployments/prometheus/alerts.yml`
4. Inspect recording rules: `deployments/prometheus/recording_rules.yml`
