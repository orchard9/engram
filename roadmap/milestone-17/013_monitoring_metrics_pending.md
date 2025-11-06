# Task 013: Monitoring and Metrics

## Objective
Implement comprehensive monitoring for dual memory operations, tracking concept formation, binding dynamics, and system health.

## Background
Production visibility is critical for understanding dual memory behavior and diagnosing issues.

## Requirements
1. Design metrics for concept lifecycle
2. Track binding formation and decay
3. Monitor fan effect impact
4. Add performance counters
5. Create Grafana dashboards

## Technical Specification

### Files to Create
- `engram-core/src/metrics/dual_memory.rs` - Metric definitions
- `monitoring/dashboards/dual_memory.json` - Grafana dashboard

### Metric Categories
```rust
pub struct DualMemoryMetrics {
    // Concept formation metrics
    concepts_formed: Counter,
    concept_formation_duration: Histogram,
    avg_cluster_size: Gauge,
    coherence_distribution: Histogram,
    
    // Binding metrics
    bindings_created: Counter,
    bindings_strengthened: Counter,
    bindings_weakened: Counter,
    avg_binding_strength: Gauge,
    
    // Fan effect metrics
    fan_out_distribution: Histogram,
    avg_fan_penalty: Gauge,
    high_fan_nodes: Counter, // Nodes with >50 bindings
    
    // Recall metrics
    episodic_recalls: Counter,
    semantic_recalls: Counter,
    blended_recalls: Counter,
    recall_confidence_dist: Histogram,
    
    // Performance metrics
    concept_lookup_latency: Histogram,
    binding_traversal_latency: Histogram,
    clustering_duration: Histogram,
}

impl DualMemoryMetrics {
    pub fn record_concept_formation(&self, result: &ConceptFormationResult) {
        self.concepts_formed.inc_by(result.concepts_formed as u64);
        self.concept_formation_duration.observe(result.duration.as_secs_f64());
        self.avg_cluster_size.set(result.avg_cluster_size as f64);
        
        for coherence in &result.coherence_scores {
            self.coherence_distribution.observe(*coherence as f64);
        }
    }
    
    pub fn record_fan_effect(&self, node_id: &NodeId, fan_out: usize, penalty: f32) {
        self.fan_out_distribution.observe(fan_out as f64);
        
        if fan_out > 50 {
            self.high_fan_nodes.inc();
            warn!("High fan-out node detected: {} with {} bindings", node_id, fan_out);
        }
        
        // Update running average
        let current_avg = self.avg_fan_penalty.get();
        let new_avg = (current_avg * 0.95) + (penalty as f64 * 0.05);
        self.avg_fan_penalty.set(new_avg);
    }
}
```

### Integration Points
```rust
// In concept formation
let start = Instant::now();
let concepts = self.form_concepts(episodes);
let duration = start.elapsed();

self.metrics.record_concept_formation(&ConceptFormationResult {
    concepts_formed: concepts.len(),
    duration,
    avg_cluster_size: calculate_avg_size(&concepts),
    coherence_scores: concepts.iter().map(|c| c.coherence).collect(),
});

// In spreading engine
if let MemoryNodeType::Concept { .. } = node_type {
    let fan_out = self.graph.get_binding_count(&node_id);
    let penalty = calculate_fan_penalty(fan_out);
    self.metrics.record_fan_effect(&node_id, fan_out, penalty);
}
```

### Alerting Rules
```yaml
groups:
  - name: dual_memory_alerts
    rules:
      - alert: HighConceptFormationRate
        expr: rate(concepts_formed[5m]) > 100
        annotations:
          summary: "Excessive concept formation detected"
          
      - alert: LowCoherenceScores
        expr: histogram_quantile(0.5, coherence_distribution) < 0.5
        annotations:
          summary: "Poor concept quality - low coherence"
          
      - alert: ExcessiveFanOut
        expr: high_fan_nodes > 10
        annotations:
          summary: "Too many high fan-out nodes affecting performance"
```

## Implementation Notes
- Use Prometheus format for metrics
- Consider cardinality of labels
- Add trace sampling for detailed diagnostics
- Include memory usage tracking

## Testing Approach
1. Unit tests for metric recording
2. Integration tests with mock data
3. Load tests to verify metric overhead
4. Dashboard validation

## Acceptance Criteria
- [ ] All dual memory operations have metrics
- [ ] Grafana dashboard shows key indicators
- [ ] Alerts trigger on anomalies
- [ ] Metric overhead <2%
- [ ] Histograms have appropriate buckets

## Dependencies
- Existing metrics infrastructure
- Prometheus/Grafana setup

## Estimated Time
2 days