# Task 008: Integrated Recall Implementation

## Objective
Implement the core `recall()` operation that integrates all spreading components with the existing MemoryStore.

## Priority
P0 (Critical Path)

## Effort Estimate
2 days

## Dependencies
- Task 002: Vector-Similarity Activation Seeding
- Task 003: Tier-Aware Spreading Scheduler
- Task 004: Confidence Aggregation Engine
- Task 005: Cyclic Graph Protection

## Technical Approach

### Implementation Details
- Extend `MemoryStore::recall()` to use activation spreading instead of simple similarity
- Implement cue-to-activation-to-results pipeline
- Add spreading result ranking based on final activation levels
- Create recall confidence that combines similarity and spreading confidence

### Files to Create/Modify
- `engram-core/src/store.rs` - Extend `recall()` method with spreading
- `engram-core/src/activation/recall.rs` - New file for integrated recall logic
- `engram-core/src/activation/mod.rs` - Export recall functionality

### Integration Points
- Extends existing `MemoryStore::recall()` around line 400
- Uses all spreading components from previous tasks
- Maintains backward compatibility with similarity-based recall
- Integrates with confidence calibration from Milestone 2

## Implementation Details

### CognitiveRecall Structure
```rust
pub struct CognitiveRecall {
    vector_seeder: VectorActivationSeeder,
    scheduler: TierAwareSpreadingScheduler,
    aggregator: ConfidenceAggregator,
    cycle_detector: CycleDetector,
    result_ranker: SpreadingResultRanker,
    spreading_config: SpreadingConfig,
}

impl CognitiveRecall {
    pub async fn recall(
        &self,
        cue: &Cue,
        memory_store: &MemoryStore,
    ) -> Result<Vec<(Episode, Confidence)>, EnggramError> {
        // 1. Seed activation from vector similarity
        let initial_activations = self.vector_seeder
            .seed_from_cue(cue, memory_store)
            .await?;

        // 2. Spread activation through memory graph
        let spreading_results = self.scheduler
            .spread_activation(initial_activations, &self.cycle_detector)
            .await?;

        // 3. Aggregate confidence from multiple paths
        let aggregated_results = self.aggregator
            .aggregate_paths(spreading_results)
            .await?;

        // 4. Rank and return results
        let ranked_results = self.result_ranker
            .rank_by_activation(aggregated_results)
            .await?;

        Ok(ranked_results)
    }
}
```

### MemoryStore Integration
```rust
impl MemoryStore {
    pub async fn recall(&self, cue: &Cue) -> Result<Vec<(Episode, Confidence)>, EnggramError> {
        match self.config.recall_mode {
            RecallMode::Similarity => self.recall_similarity(cue).await,
            RecallMode::Spreading => self.recall_spreading(cue).await,
            RecallMode::Hybrid => self.recall_hybrid(cue).await,
        }
    }

    async fn recall_spreading(&self, cue: &Cue) -> Result<Vec<(Episode, Confidence)>, EnggramError> {
        self.cognitive_recall
            .recall(cue, self)
            .await
    }
}
```

### Result Ranking Strategy
1. **Activation Level**: Primary ranking by final activation value
2. **Confidence Score**: Secondary ranking by aggregated confidence
3. **Similarity Score**: Tertiary ranking by original vector similarity
4. **Recency Bias**: Slight boost for recently accessed memories

### Backward Compatibility
- Feature flag `spreading_enabled` to enable/disable spreading
- Fallback to similarity-based recall if spreading fails
- Configuration migration for existing deployments

## Acceptance Criteria
- [ ] `recall()` uses spreading activation instead of simple similarity search
- [ ] All spreading components properly integrated in recall pipeline
- [ ] Results ranked by activation level and confidence
- [ ] Backward compatibility maintained with similarity-based recall
- [ ] Performance meets <10ms P95 latency target
- [ ] Error handling gracefully falls back to similarity search
- [ ] Configuration allows enabling/disabling spreading behavior

## Testing Approach
- Integration tests comparing spreading vs similarity recall
- Performance tests validating latency targets
- A/B tests measuring recall quality improvements
- Backward compatibility tests with existing datasets
- Error injection tests validating graceful fallback

## Risk Mitigation
- **Risk**: Spreading recall slower than similarity recall
- **Mitigation**: Performance optimization, hybrid mode with time budgets
- **Testing**: Comprehensive performance benchmarking across workload types

- **Risk**: Breaking existing MemoryStore behavior
- **Mitigation**: Feature flags, extensive regression testing, gradual rollout
- **Validation**: Full test suite passes with both recall modes

- **Risk**: Spreading produces worse recall quality than similarity
- **Mitigation**: Hybrid mode, A/B testing, quality metrics validation
- **Monitoring**: Track recall quality metrics and user satisfaction

## Implementation Strategy

### Phase 1: Basic Integration
- Implement `CognitiveRecall` with basic spreading pipeline
- Add feature flag for enabling spreading recall
- Basic integration tests

### Phase 2: Performance Optimization
- Optimize recall pipeline for latency targets
- Add performance monitoring and metrics
- Comprehensive performance testing

### Phase 3: Production Readiness
- Add error handling and graceful fallback
- Configuration management and migration
- Full backward compatibility validation

## Configuration Options
```rust
pub struct SpreadingConfig {
    pub max_hop_count: u16,           // Default: 5
    pub activation_threshold: f32,     // Default: 0.01
    pub time_budget: Duration,         // Default: 10ms
    pub max_results: usize,           // Default: 50
    pub enable_hybrid_fallback: bool, // Default: true
}
```

## Notes
This task represents the culmination of Milestone 3 - the moment when Engram transforms from a vector database into a cognitive database. The quality of this integration determines whether spreading activation enhances recall (cognitive behavior) or merely adds complexity (performance overhead). Success here validates the core hypothesis that memory-oriented computation requires fundamentally different primitives than transaction-oriented storage.