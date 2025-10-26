# Prometheus Queries Reference - Cognitive Patterns Dashboard

This document provides detailed specifications for all PromQL queries used in the Engram Cognitive Patterns dashboard. Each query is documented with purpose, expected range, performance characteristics, and troubleshooting guidance.

## Query Performance Requirements

- Response time target: <100ms per query
- Total dashboard load: <1.5s for all 15 panels
- Prometheus cardinality: <10K unique time series per instance
- Query window: 5m for real-time metrics, 1h for trend analysis

## Row 1: Priming Metrics

### Panel 1.1: Priming Event Rate by Type

**Purpose**: Monitor relative frequency of semantic vs associative vs repetition priming to identify if one priming type dominates unexpectedly.

**Queries**:
```promql
rate(engram_priming_events_total{priming_type="semantic"}[5m])
rate(engram_priming_events_total{priming_type="associative"}[5m])
rate(engram_priming_events_total{priming_type="repetition"}[5m])
```

**Explanation**:
- `engram_priming_events_total`: Counter tracking priming events
- `priming_type` label: Distinguishes semantic/associative/repetition
- `rate()[5m]`: Converts counter to events/second over 5-minute window
- Visualization: Stacked area chart showing relative contribution

**Expected Range**:
- Semantic: Typically 40-60% of total priming events
- Associative: 20-40% of total
- Repetition: 10-30% of total
- Total rate: Depends on workload (0.1-100 events/sec)

**Performance**: <30ms (simple counter rate calculation)

**Troubleshooting**:
- All rates = 0: No priming events detected (check semantic similarity threshold)
- Semantic dominates >90%: Similarity threshold may be too high
- Repetition = 0: Memory access patterns may lack repeated retrieval
- High cardinality warning: Limit `priming_type` label to 3 values

---

### Panel 1.2: Priming Strength Distribution

**Purpose**: Validate priming strength calibration to ensure values fall within 0.3-0.8 range, preventing excessive priming that causes false memories.

**Queries**:
```promql
histogram_quantile(0.50, engram_priming_strength_bucket)
histogram_quantile(0.95, engram_priming_strength_bucket)
histogram_quantile(0.99, engram_priming_strength_bucket)
```

**Explanation**:
- `engram_priming_strength_bucket`: Histogram tracking priming strength distribution
- `histogram_quantile(0.95, ...)`: Computes 95th percentile from pre-computed buckets
- Buckets: [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

**Expected Range**:
- P50: 0.4-0.6 (median priming strength)
- P95: 0.6-0.8 (strong but not excessive)
- P99: <0.9 (extreme priming detection)

**Alert Thresholds**:
- Warning: P95 > 0.85 (investigate semantic similarity threshold)
- Critical: P95 > 0.9 (excessive priming, may cause false memories)

**Performance**: <50ms (histogram quantile computation)

**Troubleshooting**:
- P95 > 0.9: Semantic similarity threshold too low (increase cosine threshold)
- P50 < 0.3: Priming too weak (may not trigger pattern completion)
- Flat distribution: Check histogram bucket configuration
- Query returns NaN: No priming events recorded yet

---

### Panel 1.3: Top Primed Node Pairs

**Purpose**: Identify most frequently primed associations to validate semantic network structure and detect unexpected associations.

**Query**:
```promql
topk(10, engram_priming_node_pairs_total)
```

**Explanation**:
- `engram_priming_node_pairs_total`: Counter tracking priming between specific node pairs
- Labels: `source_node`, `target_node`
- `topk(10, ...)`: Returns 10 highest-value time series
- Visualization: Table with columns [Source Node, Target Node, Count]

**Expected Range**:
- Top pair: 100-10000 priming events (depends on deployment duration)
- Long tail: Many pairs with 1-10 events (sparse semantic network)
- Cardinality: <10K unique pairs per instance

**Performance**: <80ms (topk requires sorting, higher than simple aggregation)

**Troubleshooting**:
- Empty table: No priming events or label missing
- High cardinality warning: Limit tracking to top N pairs (use recording rule)
- Unexpected pairs: Indicates semantic similarity miscalibration
- All pairs have count = 1: Memory access patterns lack repetition

---

## Row 2: Interference Metrics

### Panel 2.1: Interference Event Rates Over Time

**Purpose**: Monitor interference detection frequency by type to correlate spikes with recall failures.

**Queries**:
```promql
rate(engram_proactive_interference_total[5m])
rate(engram_retroactive_interference_total[5m])
rate(engram_fan_effect_interference_total[5m])
```

**Explanation**:
- Proactive: Earlier learning disrupts later recall
- Retroactive: Later learning disrupts earlier recall
- Fan effect: High fan-out associations slow retrieval
- `rate()[5m]`: Interference events per second

**Expected Range**:
- Proactive: 0.01-1.0 events/sec (depends on memory update patterns)
- Retroactive: Similar to proactive (balanced interference)
- Fan effect: 0.1-2.0 events/sec (more frequent due to associative networks)

**Alert Thresholds**:
- Warning: Any interference type >2x baseline for >5m
- Critical: Sustained spike >5x baseline (memory conflict storm)

**Performance**: <40ms (three independent rate calculations)

**Troubleshooting**:
- All interference = 0: Detection disabled or similarity threshold too high
- Proactive >> Retroactive: Update patterns favor newer memories
- Fan effect dominates: High-degree nodes causing retrieval slowdown
- Sudden spike: Correlate with memory write bursts

---

### Panel 2.2: Interference Magnitude Histograms

**Purpose**: Validate interference strength distribution to ensure most interference is moderate (<0.5) and detect extreme cases.

**Queries**:
```promql
histogram_quantile(0.50, engram_proactive_interference_magnitude_bucket)
histogram_quantile(0.95, engram_proactive_interference_magnitude_bucket)
histogram_quantile(0.99, engram_proactive_interference_magnitude_bucket)
```

**Explanation**:
- `engram_proactive_interference_magnitude_bucket`: Histogram of interference strength [0.0-1.0]
- P50: Median interference (should be <0.5 for healthy system)
- P95: High interference threshold (alert if >0.8)
- P99: Extreme interference (rare, indicates severe memory conflict)

**Expected Range**:
- P50: 0.2-0.4 (moderate interference, biologically plausible)
- P95: 0.5-0.7 (strong interference, acceptable)
- P99: 0.7-0.8 (extreme but rare)

**Alert Thresholds**:
- Warning: P95 > 0.75 for >10m
- Critical: P95 > 0.8 (may degrade recall performance)

**Performance**: <60ms (histogram quantile on 3 percentiles)

**Troubleshooting**:
- P95 > 0.8: Check semantic similarity threshold (too low causes excessive overlap)
- P50 > 0.6: Systemic interference issue, review memory consolidation
- Flat distribution: Histogram bucket misconfiguration

---

### Panel 2.3: Most Interfering Episode Pairs

**Purpose**: Identify specific memory conflicts to trace interference to concrete episodes and validate conflict resolution.

**Query**:
```promql
topk(10, engram_interference_episode_pairs_total)
```

**Explanation**:
- `engram_interference_episode_pairs_total`: Counter tracking interference between specific episodes
- Labels: `target_episode`, `competing_episode`
- `topk(10, ...)`: Returns 10 most interfering pairs

**Expected Range**:
- Top pair: 50-5000 interference events
- Cardinality: <5K unique episode pairs

**Performance**: <80ms (topk with sorting)

**Troubleshooting**:
- Same pair dominates: Investigate why these specific episodes conflict
- High cardinality: Limit tracking with sampling or aggregation
- Empty table: Interference detection disabled or no conflicts detected

---

## Row 3: Reconsolidation Metrics

### Panel 3.1: Reconsolidation Window Hit Rate

**Purpose**: Monitor temporal window accuracy to validate reconsolidation timing parameters. Target: >50% hit rate.

**Query**:
```promql
rate(engram_reconsolidation_window_hits_total[5m]) /
(rate(engram_reconsolidation_window_hits_total[5m]) +
 rate(engram_reconsolidation_window_misses_total[5m]))
```

**Explanation**:
- Numerator: Successful reconsolidation within temporal window
- Denominator: Total reconsolidation attempts (hits + misses)
- Result: Hit rate as percentage (0.0-1.0)
- Visualization: Gauge showing percentage

**Expected Range**:
- Nominal: 50-70% (indicates correct temporal dynamics)
- Acceptable: 40-50% (suboptimal but functional)
- Critical: <40% (window misconfiguration)

**Alert Thresholds**:
- Warning: <50% for >10m (tune window parameters)
- Critical: <30% for >5m (window configuration broken)

**Performance**: <50ms (ratio of two rate calculations)

**Troubleshooting**:
- Hit rate = 0: Reconsolidation window not configured or all attempts outside window
- Hit rate = 1.0: Window too wide (captures all events, defeats purpose)
- Query returns NaN: No reconsolidation events yet
- Unstable hit rate: Memory access patterns highly variable

---

### Panel 3.2: Modifications Per Reconsolidation Event

**Purpose**: Track memory update frequency during reconsolidation to validate plasticity factor calibration.

**Query**:
```promql
rate(engram_reconsolidation_modifications_total[5m]) /
rate(engram_reconsolidation_events_total[5m])
```

**Explanation**:
- Numerator: Total memory modifications during reconsolidation
- Denominator: Total reconsolidation events
- Result: Average modifications per event
- Values >10 indicate excessive plasticity

**Expected Range**:
- Nominal: 1-5 modifications per event (targeted updates)
- Acceptable: 5-10 (moderate plasticity)
- Critical: >10 (excessive modification, stability risk)

**Alert Thresholds**:
- Warning: >10 for >10m (reduce plasticity factor)
- Critical: >20 (memories too malleable, may lose information)

**Performance**: <50ms (ratio calculation)

**Troubleshooting**:
- Ratio >10: Plasticity factor too high, reduce bounds
- Ratio <1: Reconsolidation events not triggering modifications (check thresholds)
- Query returns NaN: No reconsolidation events or divide-by-zero

---

### Panel 3.3: Plasticity Factor Distribution

**Purpose**: Monitor memory malleability during reconsolidation to ensure memories don't become unstable.

**Queries**:
```promql
histogram_quantile(0.50, engram_reconsolidation_plasticity_factor_bucket)
histogram_quantile(0.95, engram_reconsolidation_plasticity_factor_bucket)
```

**Explanation**:
- `engram_reconsolidation_plasticity_factor_bucket`: Histogram of plasticity values [0.0-1.0]
- P50: Median plasticity (typical malleability)
- P95: High plasticity threshold (alert if >0.9)

**Expected Range**:
- P50: 0.3-0.6 (moderate plasticity)
- P95: 0.6-0.8 (strong but bounded)

**Alert Thresholds**:
- Warning: P95 > 0.85 for >10m
- Critical: P95 > 0.9 (memories too malleable, stability risk)

**Performance**: <60ms (histogram quantile)

**Troubleshooting**:
- P95 > 0.9: Reduce plasticity factor bounds or increase decay rate
- P50 < 0.2: Reconsolidation not effective, increase plasticity
- Flat distribution: Histogram bucket misconfiguration

---

## Row 4: False Memory Validation (CRITICAL)

### Panel 4.1: DRM False Recall Rate (PRIMARY SLI)

**Purpose**: Validate biological plausibility of false memory generation. This is the MOST CRITICAL metric for cognitive architecture correctness.

**Query**:
```promql
rate(engram_drm_critical_lure_recalls_total[5m]) /
(rate(engram_drm_critical_lure_recalls_total[5m]) +
 rate(engram_drm_list_item_recalls_total[5m]))
```

**Explanation**:
- Numerator: Critical lure recalls (false memories)
- Denominator: Total recalls (false + true)
- `rate()`: Converts counters to recalls/second over 5m window
- Division: Yields false recall percentage (0.0-1.0 range)

**Expected Range**:
- Nominal: 0.55-0.65 (55-65%, matches Roediger & McDermott 1995)
- Warning: 0.45-0.55 or 0.65-0.75 (acceptable but investigate)
- Critical: <0.45 or >0.75 (cognitive mechanisms broken)

**Alert Thresholds**:
- Warning: Outside [0.50, 0.70] for >5m
- Critical: Outside [0.45, 0.75] for >10m (biological plausibility violated)

**Performance**: <50ms (simple counter division)

**Troubleshooting**:
- Query returns NaN: No DRM trials executed yet (expected on fresh deployment)
- Rate = 0: Pattern completion disabled or semantic priming too weak
- Rate >80%: Semantic priming too strong, generating excessive false memories
- Rate <30%: Semantic priming too weak, not triggering pattern completion
- Unstable rate: Insufficient DRM trials (increase sample size)

**Runbook**: [DRM Calibration Runbook](#drm-calibration-runbook)

---

### Panel 4.2: Critical Lure Generation Rate

**Purpose**: Monitor false memory generation frequency to validate semantic priming + pattern completion pipeline.

**Query**:
```promql
rate(engram_drm_critical_lure_generations_total[5m])
```

**Explanation**:
- `engram_drm_critical_lure_generations_total`: Counter tracking false memory generation events
- `rate()[5m]`: False memories generated per second

**Expected Range**:
- Nominal: 0.01-1.0 generations/sec (depends on query rate)
- Warning: 0 for >5m (pattern completion failure)
- Critical: 0 for >10m (semantic priming or pattern completion broken)

**Alert Thresholds**:
- Warning: Rate = 0 for >5m
- Critical: Rate = 0 for >10m (investigate semantic priming strength)

**Performance**: <30ms (simple rate calculation)

**Troubleshooting**:
- Rate = 0: Pattern completion disabled, semantic priming too weak, or no DRM queries
- Rate very high: May indicate excessive false memory generation (check DRM false recall rate)

---

### Panel 4.3: Reconstruction Confidence Distribution

**Purpose**: Validate confidence calibration for false memories to ensure they're not filtered out by low-confidence thresholds.

**Queries**:
```promql
histogram_quantile(0.50, engram_reconstruction_confidence_bucket)
histogram_quantile(0.95, engram_reconstruction_confidence_bucket)
```

**Explanation**:
- `engram_reconstruction_confidence_bucket`: Histogram of confidence values [0.0-1.0]
- P50: Median confidence for reconstructed memories
- P95: High-confidence reconstructions

**Expected Range**:
- P50: 0.4-0.7 (moderate confidence, matches human phenomenology)
- P95: 0.7-0.9 (high but not perfect confidence)

**Alert Thresholds**:
- Warning: P50 <0.3 (low-confidence false memories may be filtered)
- Critical: P95 <0.4 (confidence calibration broken)

**Performance**: <60ms (histogram quantile)

**Troubleshooting**:
- P50 <0.3: Increase base confidence for pattern completion
- P95 >0.95: False memories too confident (may mislead users)
- Query returns NaN: No reconstruction events yet

---

## Row 5: System Health

### Panel 5.1: Metrics Collection Overhead

**Purpose**: Validate <1% overhead requirement from Task 001 to ensure metrics don't degrade performance.

**Query**:
```promql
engram_metrics_overhead_percent
```

**Explanation**:
- `engram_metrics_overhead_percent`: Gauge tracking metrics collection CPU overhead as percentage
- Direct gauge read (not rate or histogram)
- Performance budget: <1% from Task 001

**Expected Range**:
- Nominal: 0.1-0.5% (efficient metrics collection)
- Warning: 0.5-1.0% (approaching budget)
- Critical: >1.0% (performance budget violation)

**Alert Thresholds**:
- Warning: >0.8% for >5m
- Critical: >1.0% for >5m (Task 001 requirement violated)

**Performance**: <20ms (direct gauge read, fastest query type)

**Troubleshooting**:
- Overhead >1%: Reduce histogram bucket count, disable non-critical metrics
- Overhead = 0: Metrics overhead tracking not enabled (check feature flag)
- Overhead spikes: Correlate with histogram recording or label cardinality explosion

**Runbook**: [Metrics Overhead Runbook](#metrics-overhead-runbook)

---

### Panel 5.2: Event Buffer Utilization

**Purpose**: Monitor tracing buffer capacity to detect event storms or slow consumers before data loss occurs.

**Query**:
```promql
engram_cognitive_event_buffer_utilization
```

**Explanation**:
- `engram_cognitive_event_buffer_utilization`: Gauge tracking buffer fill percentage [0-100]
- Direct gauge read
- Alerts trigger before buffer overflow (at 80%)

**Expected Range**:
- Nominal: 10-50% (healthy buffer headroom)
- Warning: 50-80% (monitor closely)
- Critical: >80% (may drop events soon)

**Alert Thresholds**:
- Warning: >80% for >2m
- Critical: >95% for >30s (imminent data loss)

**Performance**: <20ms (direct gauge read)

**Troubleshooting**:
- Utilization >80%: Increase buffer size (ENGRAM_EVENT_BUFFER_SIZE env var)
- Sustained high utilization: Consumer too slow (add consumer threads)
- Utilization = 100%: Events being dropped (check Panel 5.3)

**Runbook**: [Event Buffer Tuning Runbook](#event-buffer-tuning-runbook)

---

### Panel 5.3: Dropped Events Counter

**Purpose**: Detect data loss in tracing pipeline. Target: 0 dropped events (data loss unacceptable).

**Query**:
```promql
rate(engram_cognitive_events_dropped_total[5m])
```

**Explanation**:
- `engram_cognitive_events_dropped_total`: Counter tracking dropped events due to buffer overflow
- `rate()[5m]`: Dropped events per second
- Target: 0 (any data loss is critical)

**Expected Range**:
- Nominal: 0 events/sec (no data loss)
- Critical: >0 (immediate investigation required)

**Alert Thresholds**:
- Critical: >0 for >1m (data loss detected)

**Performance**: <30ms (simple rate calculation)

**Troubleshooting**:
- Rate >0: Buffer overflow (check Panel 5.2), increase buffer size
- Sustained drops: Consumer throughput insufficient (add threads)
- Sporadic drops: Event burst exceeded buffer capacity (increase size)

**Runbook**: [Event Buffer Tuning Runbook](#event-buffer-tuning-runbook)

---

## Query Optimization Techniques

### Recording Rules (Future)

For complex queries with high evaluation cost, use Prometheus recording rules:

```yaml
groups:
  - name: engram_cognitive_patterns_recording
    interval: 30s
    rules:
      - record: engram:drm_false_recall_rate:5m
        expr: |
          rate(engram_drm_critical_lure_recalls_total[5m]) /
          (rate(engram_drm_critical_lure_recalls_total[5m]) +
           rate(engram_drm_list_item_recalls_total[5m]))

      - record: engram:reconsolidation_window_hit_rate:5m
        expr: |
          rate(engram_reconsolidation_window_hits_total[5m]) /
          (rate(engram_reconsolidation_window_hits_total[5m]) +
           rate(engram_reconsolidation_window_misses_total[5m]))
```

### Cardinality Management

Limit high-cardinality labels:
- Use `topk()` to limit table panels to top N entries
- Aggregate rare labels into "other" category
- Use recording rules to pre-aggregate high-cardinality metrics

### Query Window Selection

- 5m window: Real-time dashboards (balance accuracy vs responsiveness)
- 1h window: Trend analysis (smooth out short-term fluctuations)
- Avoid windows <1m (excessive Prometheus load)

---

## Operational Runbooks

### DRM Calibration Runbook

**Symptom**: DRM false recall rate outside [45%, 75%] range

**Diagnosis Steps**:
1. Check semantic priming strength distribution (Panel 1.2)
   - If P95 >0.9: Priming too strong, increase semantic similarity threshold
   - If P95 <0.4: Priming too weak, decrease semantic similarity threshold

2. Review pattern completion threshold configuration
   - Ensure threshold matches cognitive architecture parameters

3. Validate DRM word lists match Roediger & McDermott (1995)
   - Lists must have high associative strength to critical lure

4. Run DRM validation suite:
   ```bash
   cargo test psychology::drm_paradigm --features cognitive_patterns
   ```

5. If issue persists, consult memory-systems-researcher agent for parameter tuning

**Resolution**: Adjust semantic similarity threshold in increments of 0.05 until false recall rate returns to [55%, 65%]

---

### Metrics Overhead Runbook

**Symptom**: Metrics overhead >1% (Panel 5.1)

**Diagnosis Steps**:
1. Verify conditional compilation (ensure metrics only enabled in production builds):
   ```bash
   cargo build --no-default-features  # Should exclude metrics
   ```

2. Profile metrics recording to identify expensive operations:
   ```bash
   cargo bench --bench metrics_overhead
   ```

3. Check histogram bucket count (excessive buckets increase overhead)
   - Reduce buckets for non-critical histograms

4. Disable non-critical metrics temporarily:
   - Comment out priming node pair tracking (high cardinality)
   - Reduce histogram precision

5. Review Task 001 zero-overhead guarantees and ensure implementation matches spec

**Resolution**: Reduce metrics granularity or sampling rate until overhead <1%

---

### Reconsolidation Tuning Runbook

**Symptom**: Reconsolidation window hit rate <50% (Panel 3.1)

**Diagnosis Steps**:
1. Measure actual reconsolidation window timing:
   - Check if window size matches expected temporal dynamics

2. Review memory access patterns:
   - Window triggered by recent access, verify access frequency

3. Adjust plasticity factor bounds (Panel 3.3):
   - If P95 >0.9: Reduce plasticity bounds
   - If P50 <0.3: Increase plasticity to make reconsolidation more effective

4. Validate window hit rate correlates with recall performance:
   - Compare hit rate with DRM false recall rate (Panel 4.1)

5. Consult cognitive-architecture-designer agent for parameter tuning

**Resolution**: Adjust reconsolidation window size in increments of 50ms until hit rate >50%

---

### Interference Mitigation Runbook

**Symptom**: P95 interference magnitude >0.8 (Panel 2.2)

**Diagnosis Steps**:
1. Identify interfering episode pairs (Panel 2.3)
   - Check if specific episodes systematically conflict

2. Check fan-out distribution:
   - High-degree nodes cause more fan-effect interference

3. Review similarity threshold:
   - Lower threshold = more semantic overlap = more conflicts

4. Consider episode consolidation:
   - Merge overlapping episodes to reduce interference

5. Validate interference aligns with psychology research:
   - Some interference is correct behavior (not a bug)

**Resolution**: If interference >0.8 is systemic, increase semantic similarity threshold by 0.05

---

### Event Buffer Tuning Runbook

**Symptom**: Event buffer utilization >80% (Panel 5.2) or dropped events >0 (Panel 5.3)

**Diagnosis Steps**:
1. Check consumer throughput:
   ```rust
   let stats = metrics.streaming_stats();
   println!("Consumer rate: {} events/sec", stats.consumption_rate);
   ```

2. Increase buffer size via environment variable:
   ```bash
   export ENGRAM_EVENT_BUFFER_SIZE=100000  # Default: 50000
   ```

3. Add consumer threads for parallel processing:
   - Configure tracing backend to use multiple consumers

4. Filter low-priority events to reduce volume:
   - Disable tracing for non-critical priming events

5. Enable backpressure to slow producers:
   - Configure tracing to block on buffer full (prevents drops but adds latency)

**Resolution**: Increase buffer size until utilization <50% under peak load

---

## References

1. **Prometheus Querying Best Practices**: https://prometheus.io/docs/practices/querying/
   - Query performance optimization
   - Cardinality management
   - Recording rules

2. **Grafana Dashboard Performance**: https://grafana.com/docs/grafana/latest/best-practices/best-practices-for-managing-dashboards/
   - Panel query optimization
   - Variable usage patterns
   - Dashboard load time targets

3. **Roediger & McDermott (1995)**: "Creating False Memories: Remembering Words Not Presented in Lists"
   - DRM paradigm baseline: 55-65% false recall
   - Experimental methodology for validation

4. **Psychology of Memory Interference**: Anderson & Neely (1996)
   - Proactive vs retroactive interference
   - Fan effect quantification
   - Retrieval-induced forgetting

5. **Cognitive Architecture Validation**: ACT-R theory (Anderson et al., 2004)
   - Activation spreading dynamics
   - Interference calculation methods
   - Temporal decay functions
