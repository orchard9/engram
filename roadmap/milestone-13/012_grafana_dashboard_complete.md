# Task 012: Grafana Dashboard for Cognitive Metrics

**Status:** PENDING
**Priority:** P2 (Observability)
**Estimated Duration:** 2 days
**Dependencies:** Task 001 (Zero-Overhead Metrics), Task 011 (Tracing Infrastructure)
**Agent Review Required:** systems-product-planner

## Overview

Create production-ready Grafana dashboard for monitoring cognitive patterns in production deployments. Provides real-time visibility into priming, interference, reconsolidation, and false memory generation with actionable alerts for out-of-range conditions.

**Critical requirement**: Dashboard must enable operations teams to debug high interference rates, tune priming parameters, monitor reconsolidation window hit rate, detect metrics overhead issues, and validate DRM false recall stays within [45%, 75%] acceptance range.

## Research Foundation

Traditional observability focuses on infrastructure metrics (CPU, memory, latency). For a cognitive graph database, we need psychology-aware monitoring that surfaces human memory phenomena as operational metrics. This dashboard bridges neuroscience research and SRE practice.

**Key insight**: Cognitive patterns are production SLIs. DRM false recall rate 55-65% is not a bug, it is correct behavior matching human memory research (Roediger & McDermott 1995). Operations teams need to recognize this.

**Prometheus query design**:
- Use `rate()` for counters to show events/second
- Use `histogram_quantile()` for latency distributions (P50, P95, P99)
- Use ratio queries for percentage metrics (DRM false recall rate)
- Query response time target: <100ms (critical for dashboard refresh)
- Aggregation period: 5m for real-time, 1h for trends

**Alert threshold calibration**:
- DRM false recall: [45%, 75%] (Roediger & McDermott ±10% tolerance)
- Metrics overhead: <1% (performance budget from Task 001)
- Reconsolidation window hit rate: >50% (indicates correct temporal dynamics)
- Interference magnitude: P95 < 0.8 (extreme interference detection)

**Operational scenarios supported**:
1. **High interference debugging**: Panel showing interference type breakdown (proactive/retroactive/fan) with drill-down to episode pairs
2. **Priming tuning**: Histogram of priming strength distribution to identify semantic threshold misconfiguration
3. **Reconsolidation monitoring**: Window hit rate vs miss rate to validate timing parameters
4. **Overhead validation**: Metrics collection latency tracking to ensure <1% overhead target met
5. **False memory validation**: DRM critical lure rate tracking to ensure biological plausibility maintained

## Dashboard Requirements

### Panel Groups

#### 1. Priming Metrics (Row 1)

**Panel 1.1: Priming Event Rate by Type**
- **Purpose**: Monitor relative frequency of semantic vs associative vs repetition priming
- **Query**:
  ```promql
  rate(engram_priming_events_total{priming_type="semantic"}[5m])
  rate(engram_priming_events_total{priming_type="associative"}[5m])
  rate(engram_priming_events_total{priming_type="repetition"}[5m])
  ```
- **Visualization**: Time series graph with stacked areas
- **Alert**: None (informational only)
- **Operational use**: Identify if one priming type dominates unexpectedly

**Panel 1.2: Priming Strength Distribution**
- **Purpose**: Validate priming strength calibration (target: 0.3-0.8 range)
- **Query**:
  ```promql
  histogram_quantile(0.50, engram_priming_strength_bucket)
  histogram_quantile(0.95, engram_priming_strength_bucket)
  histogram_quantile(0.99, engram_priming_strength_bucket)
  ```
- **Visualization**: Heatmap showing strength distribution over time
- **Alert**: P95 > 0.9 indicates excessive priming (may cause false memories)
- **Operational use**: Tune semantic similarity threshold if distribution skewed

**Panel 1.3: Top Primed Node Pairs**
- **Purpose**: Identify most frequently primed associations
- **Query**:
  ```promql
  topk(10, engram_priming_node_pairs_total)
  ```
- **Visualization**: Table with columns: source_node, target_node, count
- **Alert**: None (debugging aid)
- **Operational use**: Validate semantic network structure, identify unexpected associations

#### 2. Interference Metrics (Row 2)

**Panel 2.1: Interference Event Rates Over Time**
- **Purpose**: Monitor interference detection frequency by type
- **Query**:
  ```promql
  rate(engram_proactive_interference_total[5m])
  rate(engram_retroactive_interference_total[5m])
  rate(engram_fan_effect_interference_total[5m])
  ```
- **Visualization**: Multi-line time series
- **Alert**: Sudden spike (>2x baseline) indicates memory conflict storm
- **Operational use**: Correlate interference spikes with recall failures

**Panel 2.2: Interference Magnitude Histograms**
- **Purpose**: Validate interference strength distribution (target: most <0.5)
- **Query**:
  ```promql
  histogram_quantile(0.50, engram_proactive_interference_magnitude_bucket)
  histogram_quantile(0.95, engram_proactive_interference_magnitude_bucket)
  histogram_quantile(0.99, engram_proactive_interference_magnitude_bucket)
  ```
- **Visualization**: Bar chart showing P50/P95/P99 for each interference type
- **Alert**: P95 > 0.8 indicates extreme interference (may degrade recall)
- **Operational use**: Identify if certain interference types systematically high

**Panel 2.3: Most Interfering Episode Pairs**
- **Purpose**: Identify specific memory conflicts
- **Query**:
  ```promql
  topk(10, engram_interference_episode_pairs_total)
  ```
- **Visualization**: Table with columns: target_episode, competing_episode, interference_count
- **Alert**: None (debugging aid)
- **Operational use**: Trace interference to specific memories, validate conflict resolution

#### 3. Reconsolidation Metrics (Row 3)

**Panel 3.1: Reconsolidation Window Hit Rate**
- **Purpose**: Monitor temporal window accuracy (target: >50% hit rate)
- **Query**:
  ```promql
  rate(engram_reconsolidation_window_hits_total[5m]) /
  (rate(engram_reconsolidation_window_hits_total[5m]) +
   rate(engram_reconsolidation_window_misses_total[5m]))
  ```
- **Visualization**: Gauge showing percentage (0-100%)
- **Alert**: Hit rate <50% for >10m indicates window misconfiguration
- **Operational use**: Tune reconsolidation timing parameters

**Panel 3.2: Modifications Per Reconsolidation Event**
- **Purpose**: Track memory update frequency during reconsolidation
- **Query**:
  ```promql
  rate(engram_reconsolidation_modifications_total[5m]) /
  rate(engram_reconsolidation_events_total[5m])
  ```
- **Visualization**: Line graph showing modifications/event ratio
- **Alert**: Ratio >10 indicates excessive modification (plasticity too high)
- **Operational use**: Validate plasticity factor calibration

**Panel 3.3: Plasticity Factor Distribution**
- **Purpose**: Monitor memory malleability during reconsolidation
- **Query**:
  ```promql
  histogram_quantile(0.50, engram_reconsolidation_plasticity_factor_bucket)
  histogram_quantile(0.95, engram_reconsolidation_plasticity_factor_bucket)
  ```
- **Visualization**: Heatmap showing plasticity distribution
- **Alert**: P95 > 0.9 indicates memories too malleable (stability risk)
- **Operational use**: Tune plasticity bounds based on use case requirements

#### 4. False Memory Metrics (Row 4) - CRITICAL VALIDATION

**Panel 4.1: DRM False Recall Rate (PRIMARY SLI)**
- **Purpose**: Validate biological plausibility (target: 55-65%, tolerance: [45%, 75%])
- **Query**:
  ```promql
  rate(engram_drm_critical_lure_recalls_total[5m]) /
  (rate(engram_drm_critical_lure_recalls_total[5m]) +
   rate(engram_drm_list_item_recalls_total[5m]))
  ```
- **Visualization**: Gauge with green zone (55-65%), yellow (45-55%, 65-75%), red (<45%, >75%)
- **Alert**: CRITICAL if outside [45%, 75%] for >10m
- **Operational use**: Primary indicator of cognitive architecture correctness
- **Thresholds**:
  - Nominal: 55-65% (matches Roediger & McDermott 1995)
  - Warning: 45-55% or 65-75% (acceptable but investigate)
  - Critical: <45% or >75% (cognitive mechanisms broken)

**Panel 4.2: Critical Lure Generation Rate**
- **Purpose**: Monitor false memory generation frequency
- **Query**:
  ```promql
  rate(engram_drm_critical_lure_generations_total[5m])
  ```
- **Visualization**: Line graph
- **Alert**: Rate = 0 for >5m indicates pattern completion failure
- **Operational use**: Validate semantic priming + pattern completion pipeline

**Panel 4.3: Reconstruction Confidence Distribution**
- **Purpose**: Validate confidence calibration for false memories
- **Query**:
  ```promql
  histogram_quantile(0.50, engram_reconstruction_confidence_bucket)
  histogram_quantile(0.95, engram_reconstruction_confidence_bucket)
  ```
- **Visualization**: Bar chart showing confidence percentiles
- **Alert**: P50 <0.3 indicates low-confidence false memories (may be filtered out)
- **Operational use**: Tune confidence threshold for reconstruction

#### 5. System Health (Row 5)

**Panel 5.1: Metrics Collection Overhead**
- **Purpose**: Validate <1% overhead requirement (Task 001)
- **Query**:
  ```promql
  engram_metrics_overhead_percent
  ```
- **Visualization**: Gauge (0-5% range, red zone >1%)
- **Alert**: CRITICAL if >1% for >5m
- **Operational use**: Detect metrics instrumentation performance issues

**Panel 5.2: Event Buffer Utilization**
- **Purpose**: Monitor tracing buffer capacity (Task 011)
- **Query**:
  ```promql
  engram_cognitive_event_buffer_utilization
  ```
- **Visualization**: Gauge (0-100%)
- **Alert**: >80% for >2m indicates event storm or slow consumer
- **Operational use**: Tune buffer size or increase consumer throughput

**Panel 5.3: Dropped Events Counter**
- **Purpose**: Detect data loss in tracing pipeline
- **Query**:
  ```promql
  rate(engram_cognitive_events_dropped_total[5m])
  ```
- **Visualization**: Single stat showing total dropped (target: 0)
- **Alert**: CRITICAL if >0 (data loss unacceptable)
- **Operational use**: Investigate buffer overflow or consumer failures

## Implementation Specifications

### File Structure
```
docs/operations/grafana/
├── cognitive_patterns_dashboard.json (new) - Main dashboard definition
├── prometheus_queries.md (new) - Query reference with explanations
├── alert_rules.yml (new) - Prometheus alerting rules
└── dashboard_setup.md (update) - Installation and configuration guide
```

### Prometheus Queries Reference

**File:** `/docs/operations/grafana/prometheus_queries.md`

Document each query with:
1. **Purpose**: What operational question does this answer?
2. **Query**: Full PromQL with explanations
3. **Expected range**: Nominal/warning/critical thresholds
4. **Performance**: Expected query response time
5. **Troubleshooting**: Common issues and fixes

Example:
```markdown
### DRM False Recall Rate

**Purpose**: Validate biological plausibility of false memory generation

**Query**:
```promql
rate(engram_drm_critical_lure_recalls_total[5m]) /
(rate(engram_drm_critical_lure_recalls_total[5m]) +
 rate(engram_drm_list_item_recalls_total[5m]))
```

**Explanation**:
- Numerator: Critical lure recalls (false memories)
- Denominator: Total recalls (false + true)
- `rate()` converts counter to events/second over 5m window
- Division yields percentage (0.0-1.0 range)

**Expected Range**:
- Nominal: 0.55-0.65 (55-65%, matches empirical data)
- Warning: 0.45-0.55 or 0.65-0.75 (acceptable but investigate)
- Critical: <0.45 or >0.75 (cognitive mechanisms broken)

**Performance**: <50ms (simple counter division)

**Troubleshooting**:
- Query returns NaN: No DRM trials executed yet (expected on fresh deployment)
- Rate = 0: Pattern completion disabled or semantic priming too weak
- Rate >80%: Semantic priming too strong, generating excessive false memories
```

### Alert Rules Configuration

**File:** `/docs/operations/grafana/alert_rules.yml`

```yaml
groups:
  - name: engram_cognitive_patterns
    interval: 30s
    rules:
      # CRITICAL: Biological plausibility validation
      - alert: DRMFalseRecallOutOfRange
        expr: |
          (
            rate(engram_drm_critical_lure_recalls_total[5m]) /
            (rate(engram_drm_critical_lure_recalls_total[5m]) +
             rate(engram_drm_list_item_recalls_total[5m]))
          ) < 0.45 OR
          (
            rate(engram_drm_critical_lure_recalls_total[5m]) /
            (rate(engram_drm_critical_lure_recalls_total[5m]) +
             rate(engram_drm_list_item_recalls_total[5m]))
          ) > 0.75
        for: 10m
        labels:
          severity: critical
          component: cognitive_patterns
        annotations:
          summary: "DRM false recall rate outside [45%, 75%] acceptance range"
          description: "False recall rate {{ $value | humanizePercentage }} violates biological plausibility (target: 55-65%)"
          runbook_url: "https://docs.engram.dev/operations/troubleshooting#drm-calibration"

      # CRITICAL: Metrics overhead budget violation
      - alert: MetricsOverheadTooHigh
        expr: engram_metrics_overhead_percent > 1.0
        for: 5m
        labels:
          severity: critical
          component: metrics
        annotations:
          summary: "Metrics overhead exceeds 1% threshold"
          description: "Overhead {{ $value }}% violates <1% performance budget (Task 001)"
          runbook_url: "https://docs.engram.dev/operations/troubleshooting#metrics-overhead"

      # WARNING: Reconsolidation window misconfiguration
      - alert: ReconsolidationWindowHitRateLow
        expr: |
          rate(engram_reconsolidation_window_hits_total[5m]) /
          (rate(engram_reconsolidation_window_hits_total[5m]) +
           rate(engram_reconsolidation_window_misses_total[5m])) < 0.50
        for: 10m
        labels:
          severity: warning
          component: reconsolidation
        annotations:
          summary: "Reconsolidation window hit rate below 50%"
          description: "Hit rate {{ $value | humanizePercentage }} indicates timing parameter misconfiguration"
          runbook_url: "https://docs.engram.dev/operations/troubleshooting#reconsolidation-tuning"

      # WARNING: Extreme interference detection
      - alert: ProactiveInterferenceTooHigh
        expr: histogram_quantile(0.95, engram_proactive_interference_magnitude_bucket) > 0.8
        for: 10m
        labels:
          severity: warning
          component: interference
        annotations:
          summary: "P95 proactive interference magnitude exceeds 0.8"
          description: "High interference {{ $value }} may degrade recall performance"
          runbook_url: "https://docs.engram.dev/operations/troubleshooting#interference-mitigation"

      # CRITICAL: Event buffer overflow (data loss)
      - alert: CognitiveEventsDropped
        expr: rate(engram_cognitive_events_dropped_total[5m]) > 0
        for: 1m
        labels:
          severity: critical
          component: tracing
        annotations:
          summary: "Cognitive events being dropped"
          description: "Dropping {{ $value }} events/sec - buffer overflow or slow consumer"
          runbook_url: "https://docs.engram.dev/operations/troubleshooting#event-buffer-tuning"

      # WARNING: Event buffer approaching capacity
      - alert: CognitiveEventBufferHighUtilization
        expr: engram_cognitive_event_buffer_utilization > 80
        for: 2m
        labels:
          severity: warning
          component: tracing
        annotations:
          summary: "Cognitive event buffer utilization high"
          description: "Buffer {{ $value }}% full - may drop events soon"
          runbook_url: "https://docs.engram.dev/operations/troubleshooting#event-buffer-tuning"

      # WARNING: Pattern completion not generating false memories
      - alert: DRMCriticalLureGenerationStopped
        expr: rate(engram_drm_critical_lure_generations_total[5m]) == 0
        for: 5m
        labels:
          severity: warning
          component: pattern_completion
        annotations:
          summary: "No DRM critical lure generations detected"
          description: "Pattern completion may be disabled or semantic priming too weak"
          runbook_url: "https://docs.engram.dev/operations/troubleshooting#pattern-completion"
```

### Dashboard JSON Structure

**File:** `/docs/operations/grafana/cognitive_patterns_dashboard.json`

Structure:
```json
{
  "dashboard": {
    "title": "Engram Cognitive Patterns",
    "tags": ["engram", "cognitive", "memory", "psychology"],
    "timezone": "utc",
    "refresh": "30s",
    "time": {
      "from": "now-1h",
      "to": "now"
    },
    "rows": [
      {
        "title": "Priming Metrics",
        "panels": [/* Panel 1.1, 1.2, 1.3 */]
      },
      {
        "title": "Interference Metrics",
        "panels": [/* Panel 2.1, 2.2, 2.3 */]
      },
      {
        "title": "Reconsolidation Metrics",
        "panels": [/* Panel 3.1, 3.2, 3.3 */]
      },
      {
        "title": "False Memory Validation (CRITICAL)",
        "panels": [/* Panel 4.1, 4.2, 4.3 */]
      },
      {
        "title": "System Health",
        "panels": [/* Panel 5.1, 5.2, 5.3 */]
      }
    ],
    "templating": {
      "list": [
        {
          "name": "instance",
          "type": "query",
          "query": "label_values(engram_priming_events_total, instance)"
        },
        {
          "name": "memory_space",
          "type": "query",
          "query": "label_values(engram_priming_events_total, memory_space)"
        }
      ]
    }
  }
}
```

### Dashboard Setup Documentation

**File:** `/docs/operations/grafana/dashboard_setup.md`

Update existing consolidation dashboard setup guide to include cognitive patterns dashboard. Follow Diátaxis operations pattern:

**Structure**:
1. **Context**: Prerequisites and assumptions
2. **Action**: Step-by-step installation
3. **Verification**: How to confirm dashboard working

**Format**:
- Use numbered steps
- Include exact commands with expected output
- Provide troubleshooting for each step
- Link to runbooks for common issues

## Integration Points

**Task 001 (Zero-Overhead Metrics):** Data source for dashboard
- Cognitive pattern metrics exposed via `CognitivePatternMetrics` API
- Prometheus scrapes `/metrics` endpoint
- Metrics follow naming convention: `engram_<component>_<metric>_<unit>`

**Task 011 (Tracing Infrastructure):** Event details for drill-down
- Dashboard links to trace viewer for detailed event inspection
- JSON export button for offline analysis
- Future: Direct Grafana panel integration with tracing backend

**Task 008 (DRM Paradigm):** Validation target
- DRM false recall rate (Panel 4.1) is primary validation metric
- Alert thresholds derived from Task 008 acceptance criteria
- Dashboard confirms DRM implementation correctness in production

## Acceptance Criteria

### Must Have (Production Blocking)
- [ ] Dashboard JSON file created with all 15 panels
- [ ] All panel groups implemented (Priming, Interference, Reconsolidation, False Memory, Health)
- [ ] Prometheus queries validated (test against mock data, <100ms response time)
- [ ] Alert rules configured with correct thresholds (7 rules minimum)
- [ ] Dashboard setup documentation written (Diátaxis operations format)
- [ ] Screenshots included in docs showing nominal/warning/critical states
- [ ] Queries use appropriate aggregation functions (rate, histogram_quantile)
- [ ] Alert thresholds aligned with acceptance criteria (DRM 45-75%, overhead <1%)

### Should Have (Post-Launch Enhancement)
- [ ] Variable selectors (time range, instance, memory_space)
- [ ] Dashboard imported into Grafana Cloud (optional for local deployments)
- [ ] Drill-down links to traces (requires Task 011 completion)
- [ ] Comparison with empirical baselines (overlay psychology research data)

### Nice to Have (Future Iteration)
- [ ] Auto-refresh intervals configured (30s default, configurable)
- [ ] Annotation overlays for consolidation runs
- [ ] Export to PDF functionality for SRE reports
- [ ] Multi-dashboard navigation (link to consolidation dashboard)

## Implementation Checklist

**Phase 1: Dashboard Structure (Day 1 Morning)**
- [ ] Create `cognitive_patterns_dashboard.json` skeleton
- [ ] Define 5 row structure (Priming, Interference, Reconsolidation, False Memory, Health)
- [ ] Configure dashboard variables (instance, memory_space)
- [ ] Set refresh interval and time range defaults

**Phase 2: Panel Implementation (Day 1 Afternoon)**
- [ ] Implement Row 1: Priming panels (1.1, 1.2, 1.3)
- [ ] Implement Row 2: Interference panels (2.1, 2.2, 2.3)
- [ ] Implement Row 3: Reconsolidation panels (3.1, 3.2, 3.3)
- [ ] Implement Row 4: False Memory panels (4.1, 4.2, 4.3) - CRITICAL
- [ ] Implement Row 5: Health panels (5.1, 5.2, 5.3)

**Phase 3: Query Validation (Day 1 Evening)**
- [ ] Test each query against mock Prometheus data
- [ ] Measure query response times (<100ms requirement)
- [ ] Validate histogram buckets align with metric definitions (Task 001)
- [ ] Test queries with missing data (verify graceful degradation)

**Phase 4: Alert Configuration (Day 2 Morning)**
- [ ] Create `alert_rules.yml` with 7 core alerts
- [ ] Test alert expressions using `promtool`
- [ ] Validate alert thresholds match acceptance criteria
- [ ] Document runbook URLs for each alert

**Phase 5: Documentation (Day 2 Afternoon)**
- [ ] Write `prometheus_queries.md` (all 15 queries documented)
- [ ] Update `dashboard_setup.md` with installation steps
- [ ] Add troubleshooting section for common issues
- [ ] Capture screenshots (3 states: nominal, warning, critical)

**Phase 6: Integration Testing (Day 2 Evening)**
- [ ] Test dashboard with live metrics from Task 001 implementation
- [ ] Verify variable selectors work (filter by instance/memory_space)
- [ ] Trigger alerts manually and verify firing
- [ ] Run `make quality` to ensure no documentation issues

## Performance Requirements

**Dashboard rendering**: <2s load time for 15 panels
**Query response time**: <100ms per panel (total <1.5s for all queries)
**Memory footprint**: <50MB browser memory for dashboard
**Prometheus cardinality**: <10K unique time series per instance

**Query optimization techniques**:
- Use `rate()` with 5m window for counters (balances accuracy vs performance)
- Use `histogram_quantile()` with pre-computed buckets (avoid raw samples)
- Leverage recording rules for complex queries (future optimization)
- Use `topk()` to limit table panel cardinality

## Operational Runbooks

Dashboard links to troubleshooting runbooks for each alert:

**DRM Calibration Runbook** (`/operations/troubleshooting#drm-calibration`):
1. Check semantic priming strength distribution (Panel 1.2)
2. Review pattern completion threshold configuration
3. Validate DRM word lists match Roediger & McDermott (1995)
4. Run DRM validation suite: `cargo test psychology::drm_paradigm`
5. Consult memory-systems-researcher agent if issue persists

**Metrics Overhead Runbook** (`/operations/troubleshooting#metrics-overhead`):
1. Verify conditional compilation: `cargo build --no-default-features`
2. Profile metrics recording: `cargo bench --bench metrics_overhead`
3. Check histogram bucket count (reduce if excessive)
4. Disable non-critical metrics temporarily
5. Review Task 001 zero-overhead guarantees

**Reconsolidation Tuning Runbook** (`/operations/troubleshooting#reconsolidation-tuning`):
1. Measure actual reconsolidation window timing
2. Review memory access patterns (window triggered by recent access)
3. Adjust plasticity factor bounds
4. Validate window hit rate correlates with recall performance
5. Consult cognitive-architecture-designer agent for parameter tuning

**Interference Mitigation Runbook** (`/operations/troubleshooting#interference-mitigation`):
1. Identify interfering episode pairs (Panel 2.3)
2. Check fan-out distribution (high fan = more interference)
3. Review similarity threshold (lower threshold = more conflicts)
4. Consider episode consolidation to reduce overlap
5. Validate interference aligns with psychology research (some interference is correct)

**Event Buffer Tuning Runbook** (`/operations/troubleshooting#event-buffer-tuning`):
1. Check consumer throughput: `metrics.streaming_stats()`
2. Increase buffer size: `ENGRAM_EVENT_BUFFER_SIZE=100000`
3. Add consumer threads for parallel processing
4. Filter low-priority events to reduce volume
5. Enable backpressure to slow producers

## Testing Strategy

```bash
# Validate dashboard JSON syntax
jq empty docs/operations/grafana/cognitive_patterns_dashboard.json

# Test Prometheus alert rules
promtool check rules docs/operations/grafana/alert_rules.yml

# Validate queries against test Prometheus instance
for query in $(jq -r '.dashboard.rows[].panels[].targets[].expr' \
    docs/operations/grafana/cognitive_patterns_dashboard.json); do
  curl -G http://localhost:9090/api/v1/query \
    --data-urlencode "query=$query" | jq .status
done

# Load test dashboard rendering
wrk -t4 -c100 -d30s --latency \
  'http://localhost:3000/api/dashboards/uid/cognitive-patterns'

# Verify alert firing (manual trigger)
curl -X POST http://localhost:9090/api/v1/admin/tsdb/snapshot

# Integration test with live metrics
cargo test --features monitoring cognitive_patterns -- --nocapture
```

## Follow-ups

- Task 001: Metrics infrastructure implementation (provides data source)
- Task 011: Tracing infrastructure (provides drill-down capability)
- Task 008: DRM paradigm validation (provides acceptance criteria)
- Task 014: Operational runbook documentation (provides troubleshooting guides)

## References

1. **Grafana Dashboard Best Practices**: https://grafana.com/docs/grafana/latest/dashboards/
   - Panel design patterns
   - Variable usage
   - Performance optimization

2. **Prometheus Querying**: https://prometheus.io/docs/prometheus/latest/querying/basics/
   - PromQL syntax
   - Aggregation functions
   - Recording rules

3. **Roediger & McDermott (1995)**: "Creating False Memories: Remembering Words Not Presented in Lists"
   - DRM paradigm baseline: 55-65% false recall
   - Critical lure design principles
   - Statistical validation methodology

4. **Psychology Research on Memory Interference**:
   - Proactive interference: earlier learning disrupts later recall
   - Retroactive interference: later learning disrupts earlier recall
   - Fan effect: more associations = slower retrieval

5. **Diátaxis Framework**: https://diataxis.fr/
   - Operations documentation pattern: Context → Action → Verification
   - Tutorial vs How-to vs Reference distinction
   - User-goal oriented structure
