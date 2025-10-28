# Task 003: Production Monitoring & Alerting — pending

**Priority:** P0 (Critical Path)
**Estimated Effort:** 3 days
**Dependencies:** Task 001 (Container Deployment)
**Reviewed By:** verification-testing-lead (Professor John Regehr)
**Review Date:** 2025-10-24

## Objective

Implement comprehensive production monitoring and alerting infrastructure with Prometheus, Grafana, and Loki. Enable operators to observe cognitive system health, diagnose issues through correlated metrics, and respond to alerts within 30 seconds of threshold breach. Build upon existing streaming metrics infrastructure (see `docs/operations/metrics_streaming.md`) to provide complete observability coverage beyond consolidation-specific monitoring.

**Key Design Principles:**
1. **Metric Cardinality Control** - Label combinations must remain <10,000 series to prevent memory explosion in multi-tenant deployments
2. **Cognitive-Aligned SLIs** - Metrics track biologically-inspired properties (consolidation freshness, confidence decay, activation spread latency) not just system resource utilization
3. **Zero-Overhead Monitoring** - Leverage existing lock-free `MetricsRegistry` infrastructure (<1% overhead proven in Milestone 13)
4. **Differential Validation** - Alert thresholds derived from empirical baseline captures and soak test distributions
5. **Chaos-Tested Alerts** - Every alert rule must have a corresponding fault injection scenario to prove it fires correctly

## Integration Points

**Uses:**
- `/docs/operations/grafana/` - Existing consolidation dashboard (baseline for new panels)
- `/docs/operations/consolidation_observability.md` - Consolidation metrics contract and SLA definitions
- `/docs/operations/metrics_streaming.md` - SSE streaming architecture and schema versioning (current: 1.0.0)
- `/docs/operations/spreading.md` - Spreading activation SLO definitions and tuning parameters
- `/engram-cli/src/api.rs` - HTTP `/metrics` endpoint implementation (lines 2512, 3654)
- `/engram-core/src/metrics/mod.rs` - Lock-free `MetricsRegistry` with multi-tenant label support (lines 32-97, 256-363)
- `/engram-core/src/features/monitoring.rs` - Monitoring provider abstraction and streaming implementation
- `/engram-core/src/activation/health_checks.rs` - Spreading health probe with hysteresis (used in SystemHealth)
- `/engram-core/src/consolidation/service.rs` - Background consolidation scheduler emitting consolidation metrics
- `/engram-core/src/storage/` - Storage tier metrics (tiers.rs, wal.rs, persistence.rs)
- `/deployments/kubernetes/` - K8s manifests from Task 001
- Existing metric constants in `engram-core/src/metrics/mod.rs`:
  - Spreading: `SPREADING_ACTIVATIONS_TOTAL`, `SPREADING_LATENCY_*`, `SPREADING_BREAKER_STATE` (lines 32-45)
  - Consolidation: `CONSOLIDATION_RUNS_TOTAL`, `CONSOLIDATION_FAILURES_TOTAL`, `CONSOLIDATION_NOVELTY_GAUGE` (lines 58-72)
  - Storage: `COMPACTION_*_TOTAL`, `WAL_RECOVERY_*`, `WAL_COMPACTION_*` (lines 76-96)
  - Activation pool: `activation_pool_available_records`, `activation_pool_hit_rate`, `activation_pool_utilization` (streaming metrics)
  - Adaptive batching: `adaptive_batch_*_size`, `adaptive_batch_*_confidence`, `adaptive_batch_latency_ewma_ns` (streaming metrics)

**Creates:**
- `/deployments/prometheus/prometheus.yml` - Prometheus configuration
- `/deployments/prometheus/alerts.yml` - Alert rules
- `/deployments/prometheus/recording_rules.yml` - Recording rules
- `/deployments/grafana/dashboards/system-overview.json` - System dashboard
- `/deployments/grafana/dashboards/memory-operations.json` - Memory ops dashboard
- `/deployments/grafana/dashboards/storage-tiers.json` - Storage dashboard
- `/deployments/grafana/dashboards/api-performance.json` - API dashboard
- `/deployments/grafana/provisioning/datasources.yml` - Data source config
- `/deployments/grafana/provisioning/dashboards.yml` - Dashboard provisioning
- `/deployments/loki/loki-config.yml` - Loki configuration
- `/deployments/promtail/promtail-config.yml` - Log collection config
- `/deployments/kubernetes/monitoring-stack.yaml` - Complete monitoring K8s manifest
- `/scripts/setup_monitoring.sh` - Monitoring stack deployment script

**Updates:**
- `/docs/operations/monitoring.md` - Complete monitoring guide
- `/docs/operations/alerting.md` - Alert definitions and response

## Technical Specifications

### Prometheus Exporter Implementation

**Architecture Decision:**
Engram uses a **streaming-first** observability model. The HTTP `/metrics` endpoint returns JSON snapshots from the internal `StreamingAggregator` rather than Prometheus text format. This design:
1. Supports both Prometheus scraping (via JSON-to-Prometheus conversion) and real-time SSE streaming
2. Preserves metric schema versioning for backward compatibility
3. Enables efficient multi-window aggregation (1s, 10s, 1m, 5m) without duplicate storage
4. Allows operators to consume metrics via HTTP polling, SSE streams, or structured logs

**Implementation Path:**
1. Add Prometheus text format exporter at `/metrics/prometheus` that converts `AggregatedMetrics` snapshot to Prometheus exposition format
2. Keep existing `/metrics` JSON endpoint for SSE consumers and schema-versioned monitoring
3. Configure Prometheus to scrape `/metrics/prometheus` with 15s interval
4. Validate conversion preserves all label information and metric types

**Metric Naming Convention:**
- Prefix: `engram_` (consistent with existing metrics)
- Multi-word metrics: snake_case (e.g., `engram_memory_operation_duration_seconds`)
- Label names: snake_case (e.g., `memory_space`, `operation`, `tier`)
- Units as suffix: `_total` (counters), `_seconds` (time), `_bytes` (size), `_ratio` (0.0-1.0 gauges)

### Existing Metrics Coverage Audit

**Already Implemented (verify exposure via Prometheus exporter):**

1. **Spreading Activation Metrics** (`engram-core/src/metrics/mod.rs:32-45`):
   - `engram_spreading_activations_total` (counter) - Total activation operations
   - `engram_spreading_latency_hot_seconds` (histogram) - Hot tier activation latency
   - `engram_spreading_latency_warm_seconds` (histogram) - Warm tier activation latency
   - `engram_spreading_latency_cold_seconds` (histogram) - Cold tier activation latency
   - `engram_spreading_latency_budget_violations_total` (counter) - Activations exceeding latency SLO
   - `engram_spreading_fallback_total` (counter) - GPU->CPU fallback count
   - `engram_spreading_failures_total` (counter) - Failed activation operations
   - `engram_spreading_breaker_state` (gauge, 0=closed, 1=open, 2=half-open) - Circuit breaker state
   - `engram_spreading_breaker_transitions_total` (counter) - Circuit breaker state changes
   - `engram_spreading_gpu_launch_total` (counter) - GPU kernel launches
   - `engram_spreading_gpu_fallback_total` (counter) - GPU fallback to CPU
   - `engram_spreading_pool_utilization` (gauge, 0.0-1.0) - Activation pool utilization
   - `engram_spreading_pool_hit_rate` (gauge, 0.0-1.0) - Pool cache hit rate

2. **Consolidation Metrics** (`engram-core/src/metrics/mod.rs:58-72`):
   - `engram_consolidation_runs_total` (counter) - Successful consolidation runs
   - `engram_consolidation_failures_total` (counter) - Failed consolidation runs
   - `engram_consolidation_novelty_gauge` (gauge) - Latest novelty delta from scheduler
   - `engram_consolidation_novelty_variance` (gauge) - Novelty variance across patterns
   - `engram_consolidation_citation_churn` (gauge, 0.0-1.0) - Citation change rate
   - `engram_consolidation_freshness_seconds` (gauge) - Snapshot age in seconds
   - `engram_consolidation_citations_current` (gauge) - Total citations in snapshot

3. **Storage/Compaction Metrics** (`engram-core/src/metrics/mod.rs:76-96`):
   - `engram_compaction_attempts_total` (counter) - Compaction initiation count
   - `engram_compaction_success_total` (counter) - Successful compactions
   - `engram_compaction_rollback_total` (counter) - Rolled-back compactions
   - `engram_compaction_episodes_removed` (counter) - Episodes removed via compaction
   - `engram_compaction_storage_saved_bytes` (counter) - Bytes reclaimed
   - `engram_wal_recovery_successes_total` (counter) - WAL recovery successes
   - `engram_wal_recovery_failures_total` (counter) - WAL recovery failures
   - `engram_wal_recovery_duration_seconds` (histogram) - WAL recovery latency
   - `engram_wal_compaction_runs_total` (counter) - WAL compaction operations
   - `engram_wal_compaction_bytes_reclaimed` (counter) - Bytes reclaimed from WAL

4. **Streaming Metrics** (exported via `AggregatedMetrics` snapshot):
   - `activation_pool_available_records` (gauge) - Available pool slots
   - `activation_pool_in_flight_records` (gauge) - In-use pool slots
   - `activation_pool_high_water_mark` (gauge) - Peak pool usage
   - `activation_pool_total_created` (gauge) - Total records created
   - `activation_pool_total_reused` (gauge) - Total reused records
   - `activation_pool_miss_count` (gauge) - Cache misses
   - `activation_pool_release_failures` (gauge) - Failed releases
   - `activation_pool_hit_rate` (gauge, 0.0-1.0) - Cache hit rate
   - `activation_pool_utilization` (gauge, 0.0-1.0) - Pool utilization ratio
   - `adaptive_batch_updates_total` (counter) - Adaptive controller updates
   - `adaptive_guardrail_hits_total` (counter) - Guardrail constraint activations
   - `adaptive_topology_changes_total` (counter) - Topology change detections
   - `adaptive_fallback_activations_total` (counter) - Adaptive fallback activations
   - `adaptive_batch_latency_ewma_ns` (gauge) - Smoothed latency EWMA
   - `adaptive_batch_hot_size` (gauge) - Hot tier batch size
   - `adaptive_batch_warm_size` (gauge) - Warm tier batch size
   - `adaptive_batch_cold_size` (gauge) - Cold tier batch size
   - `adaptive_batch_hot_confidence` (gauge, 0.0-1.0) - Hot tier convergence confidence
   - `adaptive_batch_warm_confidence` (gauge, 0.0-1.0) - Warm tier convergence confidence
   - `adaptive_batch_cold_confidence` (gauge, 0.0-1.0) - Cold tier convergence confidence

### New Metrics Required for Production Monitoring

**Critical Gaps to Fill:**

**Memory Operations (missing from current implementation):**
```
engram_memories_stored_total{memory_space} - Counter
  Rationale: Track write throughput per tenant, detect write storms
  Implementation: Increment in MemoryStore::store() after successful insertion

engram_memories_recalled_total{memory_space} - Counter
  Rationale: Track read throughput per tenant, identify hot tenants
  Implementation: Increment in CognitiveRecall after successful query

engram_memories_deleted_total{memory_space} - Counter
  Rationale: Track deletion rate, detect mass deletions
  Implementation: Increment in MemoryStore::remove() after successful deletion

engram_memory_operation_duration_seconds{operation,memory_space} - Histogram
  Rationale: Per-operation latency distributions for SLO tracking
  Operations: store, recall, delete, consolidate
  Buckets: [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0] (aligned with cognitive latency targets)
  Implementation: Use existing timer pattern from StreamingMonitoringImpl::start_timer()

engram_memory_operation_errors_total{operation,error_type,memory_space} - Counter
  Rationale: Error classification for debugging, detect systemic failures
  Error types: validation_failed, storage_error, timeout, capacity_exceeded
  Implementation: Increment in error handling paths before returning ApiError

engram_active_memories_count{memory_space} - Gauge
  Rationale: Track memory space growth, detect unbounded growth
  Implementation: Update in MemoryStore after store/delete operations (use DashMap::len())
```

**Storage Tiers:**
```
engram_storage_tier_size_bytes{tier,memory_space} - Gauge
engram_storage_tier_utilization_ratio{tier,memory_space} - Gauge (0.0-1.0)
engram_storage_migration_duration_seconds{from_tier,to_tier} - Histogram
engram_storage_migration_errors_total{from_tier,to_tier,error_type} - Counter
engram_wal_size_bytes{memory_space} - Gauge
engram_wal_lag_seconds{memory_space} - Gauge
```

**API Performance:**
```
engram_http_requests_total{method,endpoint,status} - Counter
engram_http_request_duration_seconds{method,endpoint} - Histogram
engram_grpc_requests_total{service,method,status} - Counter
engram_grpc_request_duration_seconds{service,method} - Histogram
engram_active_connections{protocol} - Gauge
```

**System Resources:**
```
engram_process_cpu_seconds_total - Counter
engram_process_memory_bytes{type} - Gauge (rss, vms, heap)
engram_process_open_fds - Gauge
engram_process_threads - Gauge
engram_go_info - Info metric (binary version, commit, build date)
```

**Graph Operations:**
```
engram_graph_nodes_total{memory_space} - Gauge
engram_graph_edges_total{memory_space} - Gauge
engram_activation_spread_duration_seconds{memory_space} - Histogram
engram_pattern_completion_duration_seconds{memory_space} - Histogram
```

### Prometheus Configuration

**/deployments/prometheus/prometheus.yml:**
```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s
  external_labels:
    cluster: engram-prod
    environment: production

rule_files:
  - alerts.yml
  - recording_rules.yml

scrape_configs:
  - job_name: engram
    static_configs:
      - targets:
          - engram:7432
    metrics_path: /metrics
    scrape_interval: 10s

  - job_name: kubernetes-pods
    kubernetes_sd_configs:
      - role: pod
    relabel_configs:
      - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_scrape]
        action: keep
        regex: true
      - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_path]
        action: replace
        target_label: __metrics_path__
        regex: (.+)
      - source_labels: [__address__, __meta_kubernetes_pod_annotation_prometheus_io_port]
        action: replace
        regex: ([^:]+)(?::\d+)?;(\d+)
        replacement: $1:$2
        target_label: __address__

alerting:
  alertmanagers:
    - static_configs:
        - targets:
            - alertmanager:9093
```

### Alert Rules with Justified Thresholds

**/deployments/prometheus/alerts.yml:**

Alert thresholds are derived from:
1. **Empirical baselines** - Soak test distributions from `docs/assets/consolidation/baseline/` and `docs/assets/metrics/`
2. **Cognitive constraints** - Biological plausibility requirements (consolidation cadence, spreading latency SLOs)
3. **Differential testing** - Prometheus query validation against synthetic workloads with known failure modes
4. **Production experience** - Thresholds updated based on false-positive rates from staging deployments

```yaml
global:
  evaluation_interval: 30s

groups:
  # ========== Service Availability ==========
  - name: engram_availability
    interval: 30s
    rules:
      - alert: EngramDown
        expr: up{job="engram"} == 0
        for: 1m
        labels:
          severity: critical
          component: service
        annotations:
          summary: "Engram instance is down"
          description: "Engram has been unreachable for {{ $value | humanizeDuration }}. Check container logs and liveness probe."
          runbook: "https://docs.engram.io/operations/troubleshooting#engram-down"
          threshold_rationale: "1m delay prevents flapping during rolling restarts while ensuring rapid detection"

      - alert: HealthProbeFailure
        expr: engram_health_status{probe="spreading"} == 2  # 0=healthy, 1=degraded, 2=critical
        for: 2m
        labels:
          severity: warning
          component: spreading
        annotations:
          summary: "Spreading activation health probe reporting critical state"
          description: "Spreading health probe has been critical for {{ $value | humanizeDuration }}. Check activation pool metrics."
          runbook: "https://docs.engram.io/operations/spreading#health-probe-critical"
          threshold_rationale: "2m allows hysteresis to filter transient spikes (see SpreadingHealthProbe::hysteresis)"

  # ========== Cognitive Performance SLOs ==========
  - name: engram_cognitive_slos
    interval: 30s
    rules:
      - alert: SpreadingLatencySLOBreach
        expr: |
          histogram_quantile(0.95,
            rate(engram_spreading_latency_hot_seconds_bucket[5m])
          ) > 0.100
        for: 5m
        labels:
          severity: warning
          component: spreading
          tier: hot
        annotations:
          summary: "Hot tier spreading P95 latency exceeds 100ms SLO"
          description: "P95 spreading latency is {{ $value | humanizeDuration }}. Target: <100ms for cognitive plausibility."
          runbook: "https://docs.engram.io/operations/spreading#latency-tuning"
          threshold_rationale: "100ms aligns with hippocampal retrieval timescales (see docs/operations/spreading.md:34). P95 chosen over P99 to reduce alert noise while catching systematic slowdowns."
          validation: "Validated via chaos test: inject 150ms delays in spreading path, confirm alert fires within 5m30s"

      - alert: ConsolidationStaleness
        expr: engram_consolidation_freshness_seconds > 900
        for: 5m
        labels:
          severity: warning
          component: consolidation
        annotations:
          summary: "Consolidation snapshot is stale (>15 minutes old)"
          description: "Last consolidation snapshot is {{ $value | humanizeDuration }} old. Target: <450s (1.5x scheduler interval)."
          runbook: "https://docs.engram.io/operations/consolidation#stale-snapshots"
          threshold_rationale: "900s = 2x health contract threshold (450s). Allows one missed consolidation cycle before alerting (scheduler default: 300s interval)."
          validation: "Stop consolidation scheduler, confirm alert fires after 15m"

      - alert: ConsolidationNoveltyStagnation
        expr: |
          avg_over_time(engram_consolidation_novelty_gauge[30m]) < 0.01
          and
          deriv(engram_consolidation_novelty_gauge[30m]) == 0
        for: 30m
        labels:
          severity: info
          component: consolidation
        annotations:
          summary: "Consolidation novelty has stagnated (<0.01 for 30 minutes)"
          description: "Novelty gauge: {{ $value }}. System may have reached steady state or inputs have stopped."
          runbook: "https://docs.engram.io/operations/consolidation#novelty-stagnation"
          threshold_rationale: "<0.01 indicates minimal belief updates. 30m window filters out short quiescent periods. Info-level to avoid false alarms during normal steady-state operation."

      - alert: ConsolidationFailureStreak
        expr: |
          increase(engram_consolidation_failures_total[15m]) >= 3
        for: 0m
        labels:
          severity: critical
          component: consolidation
        annotations:
          summary: "3 consecutive consolidation failures in 15 minutes"
          description: "Consolidation has failed {{ $value }} times. Check scheduler logs and storage tier health."
          runbook: "https://docs.engram.io/operations/consolidation#failure-streak"
          threshold_rationale: "3 failures = systematic issue, not transient error. 15m window captures multiple consolidation cycles (5min default interval). Fire immediately (for: 0m) to enable fast remediation."
          validation: "Inject storage write failures, confirm alert fires after 3rd consecutive failure"

  # ========== Memory Operation Performance ==========
  - name: engram_operation_performance
    interval: 30s
    rules:
      - alert: HighMemoryOperationLatency
        expr: |
          histogram_quantile(0.99,
            sum(rate(engram_memory_operation_duration_seconds_bucket{operation="recall"}[5m])) by (le, memory_space)
          ) > 1.0
        for: 10m
        labels:
          severity: warning
          component: recall
        annotations:
          summary: "P99 recall latency exceeds 1s for memory_space={{ $labels.memory_space }}"
          description: "P99 recall latency: {{ $value }}s. Target: <1s for interactive queries."
          runbook: "https://docs.engram.io/operations/performance-tuning#high-recall-latency"
          threshold_rationale: "1s = maximum acceptable latency for interactive LLM retrieval augmentation. 10m for: period filters transient spikes from cold starts."
          validation: "Inject artificial 1.5s delays in recall path, confirm alert fires after 10m"

      - alert: HighErrorRate
        expr: |
          (
            sum(rate(engram_memory_operation_errors_total[5m])) by (operation, memory_space)
            /
            sum(rate(engram_memory_operation_duration_seconds_count[5m])) by (operation, memory_space)
          ) > 0.05
        for: 5m
        labels:
          severity: critical
          component: "{{ $labels.operation }}"
        annotations:
          summary: "Error rate >5% for {{ $labels.operation }} in {{ $labels.memory_space }}"
          description: "Error rate: {{ $value | humanizePercentage }}. Investigate error_type labels."
          runbook: "https://docs.engram.io/operations/troubleshooting#high-error-rate"
          threshold_rationale: "5% error rate indicates systematic failure (not rare edge cases). 5m window balances fast detection with noise reduction."
          validation: "Return validation errors for 10% of requests, confirm alert fires within 5m30s"

  # ========== Storage and Capacity ==========
  - name: engram_storage_capacity
    interval: 60s
    rules:
      - alert: StorageTierNearCapacity
        expr: engram_storage_tier_utilization_ratio{tier="hot"} > 0.85
        for: 10m
        labels:
          severity: warning
          component: storage
          tier: "{{ $labels.tier }}"
        annotations:
          summary: "Storage tier {{ $labels.tier }} utilization >85%"
          description: "Utilization: {{ $value | humanizePercentage }}. Consider tier migration or capacity expansion."
          runbook: "https://docs.engram.io/operations/scaling#storage-capacity"
          threshold_rationale: "85% provides ~15% headroom for write bursts before hitting hard capacity limits. 10m for: prevents alerts during transient tier migrations."

      - alert: WALLagHigh
        expr: engram_wal_lag_seconds > 10
        for: 5m
        labels:
          severity: warning
          component: wal
        annotations:
          summary: "WAL replay lag exceeds 10 seconds"
          description: "Current lag: {{ $value }}s. May impact durability guarantees."
          runbook: "https://docs.engram.io/operations/troubleshooting#wal-lag"
          threshold_rationale: "10s lag = risk of data loss beyond durability SLO (target: <1s lag). 5m for: filters transient lag spikes during compaction."

      - alert: ActiveMemoryGrowthUnbounded
        expr: |
          deriv(engram_active_memories_count{memory_space!=""}[1h]) > 1000
        for: 30m
        labels:
          severity: info
          component: storage
        annotations:
          summary: "Memory space {{ $labels.memory_space }} growing rapidly (>1000/hour)"
          description: "Growth rate: {{ $value }}/hour. Monitor for potential memory leak or unbounded writes."
          runbook: "https://docs.engram.io/operations/capacity-planning#memory-growth"
          threshold_rationale: "1000 memories/hour sustained growth indicates potential issue. Info-level to avoid false alarms during legitimate high-write workloads."

  # ========== Activation Pool Health ==========
  - name: engram_activation_pool
    interval: 30s
    rules:
      - alert: ActivationPoolExhaustion
        expr: activation_pool_available_records < 10
        for: 2m
        labels:
          severity: critical
          component: activation_pool
        annotations:
          summary: "Activation pool nearly exhausted (<10 available records)"
          description: "Available records: {{ $value }}. Spreading operations may block or fail."
          runbook: "https://docs.engram.io/operations/spreading#pool-exhaustion"
          threshold_rationale: "<10 records = imminent resource exhaustion. 2m for: allows brief exhaustion during burst traffic without alerting."
          validation: "Trigger concurrent spreading activations until pool exhausted, confirm alert fires"

      - alert: ActivationPoolLowHitRate
        expr: activation_pool_hit_rate < 0.50
        for: 15m
        labels:
          severity: warning
          component: activation_pool
        annotations:
          summary: "Activation pool hit rate below 50%"
          description: "Hit rate: {{ $value | humanizePercentage }}. May indicate pool sizing issue or workload change."
          runbook: "https://docs.engram.io/operations/spreading#low-hit-rate"
          threshold_rationale: "50% hit rate = inefficient pool utilization (target: >80%). 15m window filters cold-start periods."

  # ========== Circuit Breaker Health ==========
  - name: engram_circuit_breakers
    interval: 30s
    rules:
      - alert: SpreadingCircuitBreakerOpen
        expr: engram_spreading_breaker_state == 1
        for: 5m
        labels:
          severity: warning
          component: spreading
        annotations:
          summary: "Spreading activation circuit breaker is open"
          description: "Circuit breaker has been open for {{ $value | humanizeDuration }}. Spreading operations are failing fast."
          runbook: "https://docs.engram.io/operations/spreading#circuit-breaker-open"
          threshold_rationale: "5m open = sustained failures, not transient spike. Breaker auto-recovers to half-open, so prolonged open state indicates root cause issue."

      - alert: SpreadingCircuitBreakerFlapping
        expr: |
          rate(engram_spreading_breaker_transitions_total[10m]) > 0.5
        for: 10m
        labels:
          severity: warning
          component: spreading
        annotations:
          summary: "Spreading circuit breaker is flapping (>3 transitions in 10 minutes)"
          description: "Transition rate: {{ $value }}/min. Indicates unstable spreading layer."
          runbook: "https://docs.engram.io/operations/spreading#breaker-flapping"
          threshold_rationale: "0.5 transitions/min = ~5 state changes in 10m = flapping behavior. Suggests threshold tuning needed."

  # ========== Adaptive Batching Performance ==========
  - name: engram_adaptive_batching
    interval: 30s
    rules:
      - alert: AdaptiveBatchingNotConverging
        expr: |
          avg_over_time(adaptive_batch_hot_confidence[10m]) < 0.3
        for: 30m
        labels:
          severity: info
          component: adaptive_batching
          tier: hot
        annotations:
          summary: "Adaptive batch controller not converging (hot tier confidence <30%)"
          description: "Convergence confidence: {{ $value | humanizePercentage }}. May indicate unstable workload or misconfiguration."
          runbook: "https://docs.engram.io/operations/adaptive-batching#low-confidence"
          threshold_rationale: "<30% confidence after 30m = controller unable to find stable batch size. Info-level to monitor without paging."

      - alert: AdaptiveGuardrailHitRateHigh
        expr: |
          rate(adaptive_guardrail_hits_total[5m]) > 0.1
        for: 15m
        labels:
          severity: info
          component: adaptive_batching
        annotations:
          summary: "Adaptive guardrails triggering frequently (>0.1/sec)"
          description: "Hit rate: {{ $value }}/sec. Controller may be hitting configuration limits."
          runbook: "https://docs.engram.io/operations/adaptive-batching#frequent-guardrails"
          threshold_rationale: "0.1 hits/sec = controller constrained by guardrails. Info-level for capacity planning, not immediate action."
```

### Grafana Dashboards

**System Overview Dashboard:**
- Health status (green/yellow/red indicator)
- Request rate (HTTP + gRPC)
- P50/P99 latency charts
- Error rate chart
- Active connections
- CPU and memory usage
- Storage utilization across tiers

**Memory Operations Dashboard:**
- Store/Recall/Delete rates per memory space
- Operation latency distributions
- Success/error ratios
- Active memories count over time
- Memory space comparison view

**Storage Tiers Dashboard:**
- Tier utilization gauges
- Migration flow diagram
- WAL size and lag
- Compaction activity
- Disk I/O metrics

**API Performance Dashboard:**
- Endpoint latency heatmap
- Request volume by endpoint
- Status code distribution
- gRPC vs HTTP comparison
- Client connection tracking

### Loki Configuration

**/deployments/loki/loki-config.yml:**
```yaml
auth_enabled: false

server:
  http_listen_port: 3100

ingester:
  lifecycler:
    ring:
      kvstore:
        store: inmemory
      replication_factor: 1
  chunk_idle_period: 5m
  chunk_retain_period: 30s

schema_config:
  configs:
    - from: 2024-01-01
      store: boltdb-shipper
      object_store: filesystem
      schema: v11
      index:
        prefix: index_
        period: 24h

storage_config:
  boltdb_shipper:
    active_index_directory: /loki/index
    cache_location: /loki/cache
    shared_store: filesystem
  filesystem:
    directory: /loki/chunks

limits_config:
  enforce_metric_name: false
  reject_old_samples: true
  reject_old_samples_max_age: 168h
  retention_period: 720h  # 30 days

chunk_store_config:
  max_look_back_period: 0s

table_manager:
  retention_deletes_enabled: true
  retention_period: 720h
```

### Structured Logging

**Log levels:**
- ERROR: Errors requiring immediate attention
- WARN: Potential issues, degraded state
- INFO: Normal operations, state changes
- DEBUG: Detailed diagnostics (disabled in production by default)

**Log format (JSON):**
```json
{
  "timestamp": "2025-10-24T12:34:56.789Z",
  "level": "INFO",
  "target": "engram_core::engine",
  "message": "Memory stored successfully",
  "memory_space": "agent-123",
  "operation": "store",
  "duration_ms": 12.5,
  "memory_id": "mem_abc123"
}
```

**Required log labels for Loki:**
- `job`: Service name (engram)
- `level`: Log level
- `memory_space`: Memory space ID (if applicable)
- `host`: Hostname

## Monitoring Stack Deployment

### Kubernetes Manifest

**/deployments/kubernetes/monitoring-stack.yaml:**
```yaml
---
apiVersion: v1
kind: Namespace
metadata:
  name: monitoring

---
# Prometheus
apiVersion: apps/v1
kind: Deployment
metadata:
  name: prometheus
  namespace: monitoring
spec:
  replicas: 1
  selector:
    matchLabels:
      app: prometheus
  template:
    metadata:
      labels:
        app: prometheus
    spec:
      containers:
      - name: prometheus
        image: prom/prometheus:v2.45.0
        ports:
        - containerPort: 9090
        volumeMounts:
        - name: config
          mountPath: /etc/prometheus
        - name: data
          mountPath: /prometheus
      volumes:
      - name: config
        configMap:
          name: prometheus-config
      - name: data
        emptyDir: {}

---
apiVersion: v1
kind: Service
metadata:
  name: prometheus
  namespace: monitoring
spec:
  selector:
    app: prometheus
  ports:
  - port: 9090
    targetPort: 9090

---
# Grafana
apiVersion: apps/v1
kind: Deployment
metadata:
  name: grafana
  namespace: monitoring
spec:
  replicas: 1
  selector:
    matchLabels:
      app: grafana
  template:
    metadata:
      labels:
        app: grafana
    spec:
      containers:
      - name: grafana
        image: grafana/grafana:10.0.0
        ports:
        - containerPort: 3000
        env:
        - name: GF_SECURITY_ADMIN_PASSWORD
          value: admin  # Change in production
        volumeMounts:
        - name: data
          mountPath: /var/lib/grafana
        - name: dashboards
          mountPath: /etc/grafana/provisioning/dashboards
        - name: datasources
          mountPath: /etc/grafana/provisioning/datasources
      volumes:
      - name: data
        emptyDir: {}
      - name: dashboards
        configMap:
          name: grafana-dashboards
      - name: datasources
        configMap:
          name: grafana-datasources

---
apiVersion: v1
kind: Service
metadata:
  name: grafana
  namespace: monitoring
spec:
  selector:
    app: grafana
  ports:
  - port: 3000
    targetPort: 3000
  type: LoadBalancer

---
# Loki
apiVersion: apps/v1
kind: Deployment
metadata:
  name: loki
  namespace: monitoring
spec:
  replicas: 1
  selector:
    matchLabels:
      app: loki
  template:
    metadata:
      labels:
        app: loki
    spec:
      containers:
      - name: loki
        image: grafana/loki:2.8.0
        ports:
        - containerPort: 3100
        volumeMounts:
        - name: config
          mountPath: /etc/loki
        - name: data
          mountPath: /loki
      volumes:
      - name: config
        configMap:
          name: loki-config
      - name: data
        emptyDir: {}

---
apiVersion: v1
kind: Service
metadata:
  name: loki
  namespace: monitoring
spec:
  selector:
    app: loki
  ports:
  - port: 3100
    targetPort: 3100
```

## Testing and Validation Framework

### Differential Testing Strategy

**Principle:** Every alert rule must be validated against both synthetic failure injection and empirical baseline data to prove it fires correctly and avoids false positives.

**Validation Harness:** `/scripts/validate_monitoring_stack.sh`
- Runs synthetic workload generators with known failure modes
- Injects chaos scenarios to trigger each alert
- Queries Prometheus to verify alert fired within expected time window
- Compares alert thresholds against baseline distributions from soak tests
- Validates that alerts clear when failure condition resolves

### Property-Based Alert Validation

**Invariants to Verify:**
1. **No False Negatives** - Every injected failure triggers corresponding alert within `for:` duration + 30s evaluation delay
2. **No False Positives** - Healthy baseline workloads (from `docs/assets/consolidation/baseline/`) never trigger alerts
3. **Alert Hysteresis** - Alerts don't flap when metric oscillates near threshold (validated via sinusoidal injection)
4. **Cardinality Bounds** - Label combinations remain <10,000 series across all multi-tenant test scenarios
5. **Query Performance** - All alert rule queries execute in <1s at P95 (measured via Prometheus query stats)

### Chaos Engineering Test Scenarios

Create `/engram-cli/tests/chaos_monitoring_tests.rs` with the following scenarios:

```rust
#[test]
fn chaos_spreading_latency_breach() {
    // Inject 150ms artificial delay in spreading path
    // Validate SpreadingLatencySLOBreach fires within 5m30s
    // Validate alert clears when delay removed
}

#[test]
fn chaos_consolidation_failure_streak() {
    // Inject storage write failures to cause 3 consolidation failures
    // Validate ConsolidationFailureStreak fires immediately
    // Validate alert clears after successful consolidation
}

#[test]
fn chaos_activation_pool_exhaustion() {
    // Launch concurrent spreading operations to exhaust pool
    // Validate ActivationPoolExhaustion fires within 2m30s
    // Validate alert clears when operations complete
}

#[test]
fn chaos_circuit_breaker_flapping() {
    // Inject intermittent failures to trigger breaker state transitions
    // Validate SpreadingCircuitBreakerFlapping fires after >5 transitions in 10m
}

#[test]
fn chaos_high_error_rate() {
    // Return validation errors for 10% of recall requests
    // Validate HighErrorRate fires within 5m30s
    // Validate error_type label correctly identifies validation_failed
}
```

### Metric Coverage Testing

**Script:** `/scripts/validate_metric_coverage.sh`

```bash
#!/bin/bash
# Validate all defined metrics are exposed via Prometheus exporter

PROMETHEUS_URL="${PROMETHEUS_URL:-http://localhost:9090}"

# Critical metrics that MUST be present
REQUIRED_METRICS=(
  # Spreading activation
  "engram_spreading_activations_total"
  "engram_spreading_latency_hot_seconds"
  "engram_spreading_latency_warm_seconds"
  "engram_spreading_latency_cold_seconds"
  "engram_spreading_breaker_state"

  # Consolidation
  "engram_consolidation_runs_total"
  "engram_consolidation_failures_total"
  "engram_consolidation_freshness_seconds"
  "engram_consolidation_novelty_gauge"

  # Storage
  "engram_compaction_success_total"
  "engram_wal_recovery_successes_total"
  "engram_storage_tier_utilization_ratio"

  # Memory operations (new)
  "engram_memories_stored_total"
  "engram_memories_recalled_total"
  "engram_memory_operation_duration_seconds"
  "engram_active_memories_count"

  # Activation pool
  "activation_pool_available_records"
  "activation_pool_hit_rate"
  "activation_pool_utilization"

  # Adaptive batching
  "adaptive_batch_hot_size"
  "adaptive_batch_latency_ewma_ns"
)

MISSING_METRICS=()

for metric in "${REQUIRED_METRICS[@]}"; do
  result=$(curl -s -G "${PROMETHEUS_URL}/api/v1/query" \
    --data-urlencode "query=${metric}" | jq -r '.data.result | length')

  if [[ "$result" == "0" ]]; then
    MISSING_METRICS+=("$metric")
    echo "FAIL: Metric '${metric}' not found in Prometheus"
  else
    echo "PASS: Metric '${metric}' found (${result} series)"
  fi
done

if [[ ${#MISSING_METRICS[@]} -gt 0 ]]; then
  echo ""
  echo "ERROR: ${#MISSING_METRICS[@]} required metrics missing"
  exit 1
fi

echo ""
echo "SUCCESS: All ${#REQUIRED_METRICS[@]} required metrics present"
```

### Alert Rule Query Validation

**Script:** `/scripts/validate_alert_queries.sh`

```bash
#!/bin/bash
# Validate all alert rule queries execute successfully

PROMETHEUS_URL="${PROMETHEUS_URL:-http://localhost:9090}"
ALERTS_FILE="/deployments/prometheus/alerts.yml"

# Extract PromQL expressions from alerts.yml
promql_queries=$(yq eval '.groups[].rules[].expr' "$ALERTS_FILE")

FAILED_QUERIES=()

while IFS= read -r query; do
  if [[ -z "$query" ]]; then
    continue
  fi

  # Test query execution
  response=$(curl -s -G "${PROMETHEUS_URL}/api/v1/query" \
    --data-urlencode "query=${query}")

  status=$(echo "$response" | jq -r '.status')

  if [[ "$status" != "success" ]]; then
    error=$(echo "$response" | jq -r '.error')
    FAILED_QUERIES+=("$query => $error")
    echo "FAIL: Query failed"
    echo "  Query: $query"
    echo "  Error: $error"
  else
    # Validate query latency
    exec_time=$(echo "$response" | jq -r '.data.resultType')
    echo "PASS: Query executed successfully"
  fi
done <<< "$promql_queries"

if [[ ${#FAILED_QUERIES[@]} -gt 0 ]]; then
  echo ""
  echo "ERROR: ${#FAILED_QUERIES[@]} alert queries failed validation"
  exit 1
fi

echo ""
echo "SUCCESS: All alert queries validated"
```

### Baseline Comparison Testing

**Validate Thresholds Against Empirical Data:**

```bash
#!/bin/bash
# Compare alert thresholds against baseline metric distributions
# Uses soak test data from docs/assets/consolidation/baseline/

BASELINE_DIR="docs/assets/consolidation/baseline"

# Extract P95 spreading latency from baseline
baseline_p95=$(jq -r '.snapshot.five_minutes.engram_spreading_latency_hot_seconds.p90' \
  "${BASELINE_DIR}/metrics.jsonl" | tail -1)

# Alert threshold is 100ms (0.100s)
ALERT_THRESHOLD=0.100

if (( $(echo "$baseline_p95 < $ALERT_THRESHOLD" | bc -l) )); then
  echo "PASS: Baseline P95 ($baseline_p95s) below alert threshold (${ALERT_THRESHOLD}s)"
else
  echo "FAIL: Baseline P95 ($baseline_p95s) exceeds alert threshold (${ALERT_THRESHOLD}s)"
  echo "  Threshold may be too aggressive, causing false positives in healthy systems"
  exit 1
fi

# Validate consolidation freshness threshold
baseline_freshness=$(jq -r '.snapshot.five_minutes.engram_consolidation_freshness_seconds.mean' \
  "${BASELINE_DIR}/metrics.jsonl" | tail -1)

CONSOLIDATION_ALERT_THRESHOLD=900  # 15 minutes

if (( $(echo "$baseline_freshness < $CONSOLIDATION_ALERT_THRESHOLD" | bc -l) )); then
  echo "PASS: Baseline consolidation freshness ($baseline_freshness s) below threshold (${CONSOLIDATION_ALERT_THRESHOLD}s)"
else
  echo "FAIL: Baseline consolidation freshness exceeds threshold"
  exit 1
fi
```

### Dashboard Validation Testing

**Validate Grafana Dashboard Provisioning:**

```bash
#!/bin/bash
# Verify Grafana dashboards provision correctly and render without errors

GRAFANA_URL="${GRAFANA_URL:-http://admin:admin@localhost:3000}"

REQUIRED_DASHBOARDS=(
  "engram-system-overview"
  "engram-memory-operations"
  "engram-storage-tiers"
  "engram-api-performance"
)

for dashboard_uid in "${REQUIRED_DASHBOARDS[@]}"; do
  response=$(curl -s "${GRAFANA_URL}/api/dashboards/uid/${dashboard_uid}")

  if echo "$response" | jq -e '.dashboard' > /dev/null; then
    title=$(echo "$response" | jq -r '.dashboard.title')
    panel_count=$(echo "$response" | jq '.dashboard.panels | length')
    echo "PASS: Dashboard '${title}' found with ${panel_count} panels"

    # Validate all panels have data sources configured
    missing_ds=$(echo "$response" | jq '[.dashboard.panels[] | select(.datasource == null)] | length')
    if [[ "$missing_ds" -gt 0 ]]; then
      echo "  WARN: $missing_ds panels missing datasource configuration"
    fi
  else
    echo "FAIL: Dashboard '${dashboard_uid}' not found"
    exit 1
  fi
done

echo ""
echo "SUCCESS: All ${#REQUIRED_DASHBOARDS[@]} dashboards validated"
```

### Log Aggregation Testing

**Validate Loki Ingestion and Query Performance:**

```bash
#!/bin/bash
# Test Loki log ingestion and query performance

LOKI_URL="${LOKI_URL:-http://localhost:3100}"

# Generate test log entries via Engram operations
echo "Generating test logs..."
curl -s -X POST http://localhost:7432/api/v1/memories/remember \
  -H "Content-Type: application/json" \
  -d '{"content":"test memory for log validation","confidence":0.9}' > /dev/null

# Wait for log ingestion
sleep 5

# Query logs from last 5 minutes
query_start=$(date -u -d '5 minutes ago' +%s)000000000
query_result=$(curl -s -G "${LOKI_URL}/loki/api/v1/query_range" \
  --data-urlencode 'query={job="engram"}' \
  --data-urlencode "start=${query_start}")

result_count=$(echo "$query_result" | jq '.data.result | length')

if [[ "$result_count" -gt 0 ]]; then
  echo "PASS: Found $result_count log streams"

  # Validate query latency
  exec_time=$(echo "$query_result" | jq -r '.status')
  echo "  Query status: $exec_time"

  # Validate structured log format
  first_log=$(echo "$query_result" | jq -r '.data.result[0].values[0][1]')
  if echo "$first_log" | jq -e '.level' > /dev/null 2>&1; then
    echo "  PASS: Logs are structured JSON"
  else
    echo "  WARN: Logs may not be properly structured"
  fi
else
  echo "FAIL: No logs found in Loki"
  exit 1
fi

# Test log retention enforcement
echo ""
echo "Validating log retention configuration..."
config=$(curl -s "${LOKI_URL}/config")
retention=$(echo "$config" | yq eval '.limits_config.retention_period' -)

if [[ "$retention" == "720h" ]]; then
  echo "PASS: Log retention set to 720h (30 days)"
else
  echo "WARN: Log retention is $retention (expected 720h)"
fi
```

### Integration Test Coverage Matrix

| Alert Rule | Chaos Test | Baseline Validation | Query Test | Dashboard Panel |
|-----------|-----------|---------------------|-----------|----------------|
| EngramDown | ✓ Stop container | ✓ 100% uptime baseline | ✓ | System Overview |
| SpreadingLatencySLOBreach | ✓ Inject 150ms delay | ✓ P95 < 100ms baseline | ✓ | Spreading Performance |
| ConsolidationStaleness | ✓ Stop scheduler | ✓ Freshness < 450s baseline | ✓ | Consolidation Health |
| ConsolidationFailureStreak | ✓ Inject storage failures | ✓ 0 failures baseline | ✓ | Consolidation Health |
| HighMemoryOperationLatency | ✓ Inject 1.5s delay | ✓ P99 < 1s baseline | ✓ | Memory Operations |
| HighErrorRate | ✓ Return 10% errors | ✓ <0.1% error baseline | ✓ | API Performance |
| StorageTierNearCapacity | ✓ Fill to 90% | ✓ <50% util baseline | ✓ | Storage Tiers |
| WALLagHigh | ✓ Block WAL writes | ✓ <1s lag baseline | ✓ | Storage Tiers |
| ActivationPoolExhaustion | ✓ Exhaust pool | ✓ >50% avail baseline | ✓ | Spreading Performance |
| ActivationPoolLowHitRate | ✓ Evict pool entries | ✓ >80% hit baseline | ✓ | Spreading Performance |
| SpreadingCircuitBreakerOpen | ✓ Trigger breaker | ✓ Closed baseline | ✓ | Circuit Breakers |
| SpreadingCircuitBreakerFlapping | ✓ Intermittent failures | ✓ Stable baseline | ✓ | Circuit Breakers |
| AdaptiveBatchingNotConverging | ✓ Unstable workload | ✓ >50% confidence baseline | ✓ | Adaptive Batching |

### Performance Validation Criteria

**Alert Query Performance:**
- P95 query latency < 1s (measured via Prometheus query stats)
- No alert queries cause evaluator lag (monitor `prometheus_rule_evaluation_duration_seconds`)
- Label cardinality stays <10,000 series after 24h multi-tenant workload

**Metrics Collection Overhead:**
- <1% CPU overhead from metrics collection (validated via profiling)
- <500MB RAM for Prometheus with 10,000 series and 15-day retention
- Streaming aggregator queue depth stays <1000 pending updates at P99

## Documentation Requirements

### /docs/operations/monitoring.md

**Sections:**
1. Overview - Monitoring architecture, components
2. Metrics - All metrics explained with expected values
3. Prometheus Setup - Install, configure, verify
4. Grafana Setup - Install, configure dashboards
5. Loki Setup - Log aggregation configuration
6. Dashboard Guide - How to use each dashboard
7. Query Examples - Common PromQL queries
8. Troubleshooting - Monitoring stack issues

### /docs/operations/alerting.md

**Sections:**
1. Alert Philosophy - When to alert, severity levels
2. Alert Rules - All alerts with thresholds and rationale
3. Response Procedures - What to do for each alert
4. Alert Tuning - Adjusting thresholds for your workload
5. Integration - PagerDuty, Slack, email configuration
6. Silencing Alerts - When and how to silence
7. Alert History - Reviewing past alerts

### /docs/howto/metrics-interpretation.md

**Sections:**
1. Key Metrics - Most important metrics to watch
2. Normal Baselines - Expected metric ranges
3. Anomaly Detection - Spotting unusual patterns
4. Correlation Analysis - Related metrics to check together
5. Metric-Based Debugging - Using metrics to diagnose issues

## Acceptance Criteria

### Metrics Implementation and Exposure

**Prometheus Exporter:**
- [ ] New endpoint `/metrics/prometheus` returns valid Prometheus text format (verified via `promtool check metrics`)
- [ ] Existing `/metrics` JSON endpoint continues to work (backward compatibility)
- [ ] Prometheus scrapes target successfully every 15s with <5% scrape failures
- [ ] All 60+ defined metrics exposed (spreading: 13, consolidation: 7, storage: 10, memory ops: 6, activation pool: 9, adaptive: 11, API: 7, system: 4)
- [ ] Metric cardinality validated: <10,000 unique series after 24h multi-tenant soak test (measured via `prometheus_tsdb_symbol_table_size_bytes`)

**Label Coverage:**
- [ ] `memory_space` label present on all per-tenant metrics (memories_stored_total, memory_operation_duration_seconds, active_memories_count, storage_tier_*)
- [ ] `operation` label correctly identifies all operation types: store, recall, delete, consolidate
- [ ] `error_type` label classifies errors: validation_failed, storage_error, timeout, capacity_exceeded
- [ ] `tier` label distinguishes storage tiers: hot, warm, cold
- [ ] No label combinations create >100 series per metric (validated via cardinality test)

**Histogram Bucket Configuration:**
- [ ] `engram_memory_operation_duration_seconds` buckets: [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0] (cognitively-aligned latency targets)
- [ ] `engram_spreading_latency_*_seconds` buckets: [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0] (sub-100ms resolution for SLO tracking)
- [ ] `engram_wal_recovery_duration_seconds` buckets: [0.1, 0.5, 1.0, 5.0, 10.0, 30.0, 60.0] (startup/recovery timescales)
- [ ] All histogram buckets validated against baseline distributions (no empty buckets, no >90% in single bucket)

### Alert Rule Validation

**Chaos Engineering Coverage:**
- [ ] All 13 alert rules have corresponding chaos injection tests in `/engram-cli/tests/chaos_monitoring_tests.rs`
- [ ] Each chaos test validates:
  - Alert fires within `for:` duration + 30s evaluation window
  - Alert includes correct labels (severity, component)
  - Alert clears when failure condition resolves
  - No flapping when metric oscillates ±10% around threshold
- [ ] False positive rate validated: 0 alerts during 1h baseline soak test (no injected failures)

**Alert Metadata Completeness:**
- [ ] Every alert includes `threshold_rationale` annotation explaining threshold derivation
- [ ] Every alert includes `validation` annotation describing chaos test scenario
- [ ] Every alert includes `runbook` link (even if placeholder during development)
- [ ] Alert summary uses template variables for dynamic context (e.g., `{{ $labels.memory_space }}`, `{{ $value }}`)

**Alert Query Performance:**
- [ ] All 13 alert rule queries execute in <1s at P95 (measured via Prometheus query stats API)
- [ ] No alert rule queries cause rule evaluation lag >5s (monitor `prometheus_rule_evaluation_duration_seconds`)
- [ ] Alert queries validated via `/scripts/validate_alert_queries.sh` (exits 0)

**Threshold Empirical Validation:**
- [ ] Spreading latency threshold (100ms) validated against baseline: P95 <80ms in healthy workload
- [ ] Consolidation freshness threshold (900s) validated: mean <300s in healthy workload
- [ ] Error rate threshold (5%) validated: <0.1% errors in baseline
- [ ] Activation pool exhaustion threshold (<10 available) validated: >50% available in baseline
- [ ] All thresholds tested via `/scripts/validate_baseline_thresholds.sh`

### Dashboard Provisioning and Rendering

**Dashboard Availability:**
- [ ] All 4 required dashboards provision successfully: system-overview, memory-operations, storage-tiers, api-performance
- [ ] Dashboard UIDs match naming convention: `engram-{dashboard-name}`
- [ ] All dashboards accessible via Grafana API within 30s of startup
- [ ] Validation script `/scripts/validate_dashboards.sh` passes (exits 0)

**Panel Configuration:**
- [ ] Every panel has datasource configured (no `null` datasources)
- [ ] Panels use appropriate visualization types: Time series for histograms, Gauge for ratios, Stat for counters
- [ ] Color thresholds align with alert rules (e.g., spreading latency panel turns yellow at 80ms, red at 100ms)
- [ ] All panels include descriptions explaining metric meaning
- [ ] Mobile responsive layout validated (viewport widths: 375px, 768px, 1024px)

**Interactive Features:**
- [ ] Dashboards auto-refresh every 30s (configurable via UI)
- [ ] Time range selector works correctly (Last 15m, 1h, 6h, 24h, 7d)
- [ ] `memory_space` variable filters all panels correctly (multi-tenant isolation)
- [ ] Drill-down links from overview panels to detail dashboards
- [ ] Export to PNG/PDF functionality works

### Log Aggregation and Query Performance

**Loki Ingestion:**
- [ ] Promtail collects logs from all Engram pods with <10s delay (measured via timestamp delta)
- [ ] Structured JSON logs parsed correctly (`.level`, `.target`, `.message` fields extracted)
- [ ] Log labels correctly applied: `job=engram`, `level={ERROR,WARN,INFO}`, `memory_space={id}`
- [ ] Validation: `/scripts/validate_loki_ingestion.sh` passes

**Query Performance:**
- [ ] Simple queries (`{job="engram"}`) return in <500ms for 24h window
- [ ] Filtered queries (`{job="engram", level="ERROR"}`) return in <1s for 24h window
- [ ] LogQL aggregations (`rate({job="engram"}[5m])`) return in <2s for 24h window
- [ ] Query performance validated via automated test suite

**Retention and Compaction:**
- [ ] Log retention configured to 720h (30 days) in loki-config.yml
- [ ] Retention enforcement validated: logs older than 30d automatically deleted
- [ ] Compaction runs successfully, reducing storage size (monitor `loki_ingester_chunk_stored_bytes_total`)
- [ ] Old chunks deleted after retention period (no unbounded growth)

### Documentation Completeness

**Operator Onboarding:**
- [ ] External operator (with no Engram experience) sets up monitoring in <30 minutes following `/docs/operations/monitoring.md`
- [ ] Setup guide includes:
  - Prerequisites and dependency installation
  - Step-by-step deployment instructions
  - Validation commands to verify each component
  - Troubleshooting section for common issues
- [ ] Tested by having reviewer complete setup from scratch

**Metric Documentation:**
- [ ] Every metric documented in `/docs/operations/monitoring.md` with:
  - Description of what it measures
  - Expected healthy range (e.g., "activation_pool_hit_rate: >0.80 is healthy, <0.50 needs investigation")
  - Example PromQL queries for common analyses
  - Related alerts that use this metric
- [ ] Metric schema changelog updated in `/docs/metrics-schema-changelog.md`

**Alert Response Procedures:**
- [ ] Every alert has runbook page in `/docs/operations/alerting.md` with:
  - Symptom description and impact
  - Diagnostic steps (which metrics to check, which logs to query)
  - Remediation procedures (immediate actions, long-term fixes)
  - Escalation criteria (when to page on-call vs self-resolve)
- [ ] Runbook procedures tested during chaos injection tests

### Integration and Performance

**Deployment Automation:**
- [ ] Monitoring stack deploys successfully via `/scripts/setup_monitoring.sh` (<5 minutes)
- [ ] Kubernetes deployment via `kubectl apply -f /deployments/kubernetes/monitoring-stack.yaml` succeeds
- [ ] All components reach `Ready` state within 2 minutes
- [ ] Health checks validate connectivity: Prometheus→Engram, Grafana→Prometheus, Loki→Promtail

**Data Persistence:**
- [ ] Prometheus data persists across pod restarts (PersistentVolume configured)
- [ ] Grafana dashboards persist across restarts (ConfigMap provisioning)
- [ ] Loki chunks persist across restarts (persistent storage configured)
- [ ] Test: restart all monitoring pods, verify data continuity with no gaps

**Resource Overhead:**
- [ ] Metrics collection overhead measured via profiling: <1% CPU, <50MB RAM in Engram process
- [ ] Prometheus resource usage: <5% CPU, <500MB RAM with 10,000 series and 15-day retention
- [ ] Grafana resource usage: <2% CPU, <200MB RAM
- [ ] Loki resource usage: <5% CPU, <300MB RAM with 30-day retention
- [ ] Total monitoring overhead: <10% system resources

### End-to-End Validation

**Smoke Test Scenario:**
1. Deploy monitoring stack from scratch
2. Deploy Engram with default configuration
3. Generate synthetic workload (1000 recall operations, 500 store operations)
4. Verify all metrics appear in Prometheus within 30s
5. Verify dashboards render correctly with real data
6. Inject failure (stop consolidation scheduler)
7. Verify ConsolidationStaleness alert fires within 15m
8. Resolve failure (restart scheduler)
9. Verify alert clears within 5m
10. All steps complete successfully without manual intervention

**Multi-Tenant Validation:**
- [ ] Create 10 memory spaces with unique IDs
- [ ] Generate mixed workload across all spaces
- [ ] Verify metrics segregated by `memory_space` label
- [ ] Verify dashboard `memory_space` variable filters correctly
- [ ] Verify cardinality stays within limits (<1,000 series per metric)

**Failure Mode Coverage:**
- [ ] Network partition between Prometheus and Engram: queued metrics don't cause memory leak
- [ ] Grafana restart: dashboards reload from provisioning without data loss
- [ ] Loki disk full: graceful degradation, logs dropped with error counter increment
- [ ] Alert rule evaluation timeout: alert enters `pending` state, doesn't block other rules

## Implementation Summary

This task transforms Engram's existing streaming metrics infrastructure into a production-ready observability stack with Prometheus, Grafana, and Loki. Key architectural decisions:

1. **Streaming-First Architecture** - Preserve existing `/metrics` JSON endpoint while adding `/metrics/prometheus` for compatibility. This maintains backward compatibility with SSE consumers while enabling Prometheus scraping.

2. **Cognitive SLI Alignment** - Alert thresholds derived from biological plausibility requirements (hippocampal retrieval: <100ms, consolidation cadence: ~5min) rather than arbitrary percentiles.

3. **Zero-Overhead Metrics** - Leverage existing lock-free `MetricsRegistry` (<1% overhead proven in Milestone 13) rather than introducing new instrumentation that could impact query latency.

4. **Differential Validation** - Every alert rule backed by chaos injection test and empirical baseline comparison to prove correctness and avoid false positives.

5. **Multi-Tenant Label Strategy** - Controlled label cardinality (<10,000 series) through disciplined use of `memory_space`, `operation`, `tier`, `error_type` labels only where operationally necessary.

## Critical Path Dependencies

**Blocks:**
- Task 004 (Performance Tuning & Profiling) - Profiling tools depend on metrics infrastructure
- Task 005 (Comprehensive Troubleshooting) - Runbooks reference alert rules and metric queries
- Task 011 (Load Testing & Validation) - Load tests validate metrics under stress

**Blocked By:**
- Task 001 (Container Orchestration & Deployment) - K8s manifests needed for monitoring stack deployment

**Parallel Work:**
- Task 002 (Backup & Disaster Recovery) - Can develop backup metrics in parallel, integrate later

## Risk Mitigation

**Risk: Prometheus exporter implementation complex**
- Mitigation: Start with minimal text format conversion (counters/gauges), defer histogram _bucket exposition to iteration 2
- Fallback: Continue using JSON-to-Prometheus bridge tools (e.g., json_exporter) until native implementation stabilized

**Risk: Alert threshold tuning requires production data**
- Mitigation: Use existing soak test baselines from `docs/assets/consolidation/baseline/` and `docs/assets/metrics/` for initial thresholds
- Plan: Adjust thresholds during staging deployment based on false-positive rates
- Escape hatch: All alerts start at `severity: info` until validated in staging

**Risk: Cardinality explosion in multi-tenant deployment**
- Mitigation: Hard limit of 4 labels per metric, automated cardinality tests in CI
- Validation: 24h soak test with 100 memory spaces must stay <10,000 series
- Monitoring: Alert on `prometheus_tsdb_symbol_table_size_bytes` exceeding threshold

**Risk: Chaos tests flaky in CI**
- Mitigation: Chaos tests run serially with deterministic seeds, isolated network namespaces
- Fallback: Tag as `#[ignore]` for CI, run manually before production deployment
- Documentation: Record expected runtime (chaos tests may take 15-20min total)

## Validation Checkpoints

**Checkpoint 1: Prometheus Exporter (Day 1)**
- [ ] `/metrics/prometheus` endpoint returns valid text format
- [ ] All 60+ metrics appear in Prometheus UI
- [ ] Scrape succeeds every 15s with <5% failures
- [ ] `promtool check metrics` passes

**Checkpoint 2: Alert Rules (Day 2)**
- [ ] All 13 alert rules load without syntax errors
- [ ] Alert queries execute in <1s (measured via query stats)
- [ ] Baseline soak test triggers 0 alerts (no false positives)
- [ ] At least 3 chaos tests validate alert firing

**Checkpoint 3: Dashboards (Day 2)**
- [ ] All 4 dashboards provision and render
- [ ] Panels display real data from Prometheus
- [ ] `memory_space` variable filters correctly
- [ ] Mobile layout validated (375px, 768px viewports)

**Checkpoint 4: Integration (Day 3)**
- [ ] `/scripts/setup_monitoring.sh` deploys full stack in <5min
- [ ] End-to-end smoke test passes (10-step scenario)
- [ ] Multi-tenant validation passes (10 spaces, mixed workload)
- [ ] Resource overhead validated: <10% CPU/RAM

**Checkpoint 5: Documentation (Day 3)**
- [ ] Operator onboarding tested: reviewer sets up monitoring in <30min
- [ ] All metrics documented with expected ranges
- [ ] All alerts have runbook pages with remediation steps
- [ ] Troubleshooting section covers failure modes

## Follow-Up Tasks

**Milestone 16 Dependencies:**
- Task 002: Add backup metrics and alerts (backup_success_total, backup_duration_seconds, restore_validation_errors_total)
- Task 004: Add performance profiling metrics (cpu_profile_samples, heap_alloc_bytes, goroutine_count)
- Task 005: Reference metrics in troubleshooting runbook (correlate symptoms with metric patterns)
- Task 008: Add security audit logging (auth_attempts_total, access_denied_total, rate_limit_exceeded_total)
- Task 011: Use monitoring for load test validation (validate SLOs hold under 10x baseline traffic)

**Future Enhancements (Milestone 17+):**
- Distributed tracing integration (OpenTelemetry spans for spreading activation)
- Anomaly detection via machine learning (train models on baseline distributions)
- Predictive alerting (forecast consolidation staleness 30min before breach)
- Cross-region metrics aggregation (federated Prometheus for distributed deployments)

## References

**Existing Documentation:**
- `/docs/operations/consolidation_observability.md` - Consolidation metrics contract, SLA definitions, health contract
- `/docs/operations/metrics_streaming.md` - Streaming architecture, schema versioning, SSE endpoint design
- `/docs/operations/spreading.md` - Spreading activation SLOs, tuning parameters, health probe design
- `/docs/operations/grafana/SETUP.md` - Existing consolidation dashboard setup guide
- `/docs/metrics-schema-changelog.md` - Metric schema version history and migration guides

**Baseline Data:**
- `/docs/assets/consolidation/baseline/metrics.jsonl` - 1h soak test metrics from consolidation scheduler
- `/docs/assets/metrics/2025-10-15-adaptive-update/http_metrics.json` - Live HTTP /metrics snapshot
- `/docs/assets/metrics/2025-10-12-longrun/` - Long-run synthetic soak test captures

**Code References:**
- `/engram-core/src/metrics/mod.rs` - Lock-free MetricsRegistry, metric constants, label helpers
- `/engram-core/src/features/monitoring.rs` - Monitoring provider abstraction, streaming implementation
- `/engram-core/src/activation/health_checks.rs` - Spreading health probe with hysteresis
- `/engram-cli/src/api.rs` - HTTP /metrics endpoint (lines 2512, 3654)
- `/engram-cli/tests/monitoring_tests.rs` - SSE monitoring endpoint tests

**External Standards:**
- Prometheus Exposition Format: https://prometheus.io/docs/instrumenting/exposition_formats/
- PromQL Best Practices: https://prometheus.io/docs/practices/naming/
- Grafana Dashboard JSON Schema: https://grafana.com/docs/grafana/latest/dashboards/json-model/
- LogQL Query Language: https://grafana.com/docs/loki/latest/logql/
