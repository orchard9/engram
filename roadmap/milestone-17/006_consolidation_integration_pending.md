# Task 006: Consolidation Integration

## Objective
Integrate concept formation and binding creation into the existing consolidation scheduler, ensuring smooth operation with current functionality.

## Background
The consolidation system currently handles pattern detection and compaction. We need to add concept formation without disrupting these operations.

## Requirements
1. Extend ConsolidationEngine with concept formation
2. Add configuration for concept formation parameters
3. Implement gradual rollout with feature flags
4. Track concept formation metrics
5. Ensure consolidation timing remains consistent

## Technical Specification

### Files to Modify
- `engram-core/src/consolidation/mod.rs` - Add concept formation
- `engram-core/src/config.rs` - Concept formation config
- `engram-core/src/metrics/mod.rs` - Concept metrics

### Extended Configuration
```toml
[consolidation]
enable_concepts = true
concept_sample_rate = 0.1  # Start with 10% of consolidations

[consolidation.concepts]
min_cluster_size = 3
coherence_threshold = 0.7
max_concepts_per_cycle = 10
formation_strategy = "density_based"
```

### Integration Points
```rust
impl ConsolidationEngine {
    pub async fn run_consolidation(&self) -> Result<ConsolidationStats> {
        let mut stats = ConsolidationStats::default();
        
        // Existing consolidation
        stats.patterns_detected = self.detect_patterns().await?;
        stats.memories_compacted = self.compact_memories().await?;
        
        // New concept formation
        if self.config.enable_concepts && self.should_form_concepts() {
            let concept_stats = self.form_concepts_from_episodes().await?;
            stats.concepts_formed = concept_stats.concepts_formed;
            stats.bindings_created = concept_stats.bindings_created;
        }
        
        self.metrics.record_consolidation(&stats);
        Ok(stats)
    }
    
    fn should_form_concepts(&self) -> bool {
        // Sample-based rollout
        rand::random::<f32>() < self.config.concept_sample_rate
    }
}
```

### Metrics Integration
```rust
pub struct ConceptFormationMetrics {
    concepts_formed: Counter,
    bindings_created: Counter,
    avg_cluster_size: Histogram,
    coherence_scores: Histogram,
    formation_duration: Histogram,
}
```

## Implementation Notes
- Start with low sample rate for safety
- Monitor consolidation latency impact
- Implement circuit breaker for formation failures
- Log all concept formation events

## Testing Approach
1. Integration tests with mock consolidation
2. Feature flag toggling tests
3. Metric recording validation
4. Performance impact measurement

## Acceptance Criteria
- [ ] Concept formation integrates without breaking existing consolidation
- [ ] Feature flags enable/disable cleanly
- [ ] Metrics properly track all operations
- [ ] Consolidation latency increase <10%
- [ ] Gradual rollout controls work correctly

## Dependencies
- Task 004 (Concept Formation Engine)
- Task 005 (Binding Formation)
- Existing consolidation system

## Estimated Time
2 days