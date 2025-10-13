# Task 003: Lazy Decay Integration with Recall Pipeline

## Objective
Integrate `BiologicalDecaySystem` with `CognitiveRecall` to apply temporal decay lazily during recall operations, without requiring background threads.

## Priority
P0 (blocking - core Milestone 4 requirement)

## Effort Estimate
2 days

## Dependencies
- Task 002: Last Access Tracking

## Technical Approach

### Files to Modify
- `engram-core/src/activation/recall.rs` - Integrate decay evaluation
- `engram-core/src/decay/mod.rs` - Add lazy evaluation helpers
- `engram-core/src/store.rs` - Provide decay system to recall

### Design

**Decay Application During Ranking**:
```rust
// engram-core/src/activation/recall.rs
pub struct CognitiveRecall {
    // ... existing fields ...
    decay_system: Option<Arc<BiologicalDecaySystem>>,
}

impl CognitiveRecall {
    /// Rank and filter results with decay applied
    fn rank_results(
        &self,
        aggregated: Vec<(String, f32, Confidence, Option<f32>)>,
        store: &MemoryStore,
    ) -> Vec<RankedMemory> {
        let now = Utc::now();

        let mut ranked: Vec<RankedMemory> = aggregated
            .into_iter()
            .filter_map(|(node_id, activation, confidence, similarity)| {
                store.get_episode(&node_id).map(|mut episode| {
                    // Apply temporal decay lazily
                    let elapsed = now.signed_duration_since(episode.last_access);
                    let decayed_confidence = if let Some(decay_system) = &self.decay_system {
                        decay_system.apply_to_episode(&mut episode, elapsed.to_std().unwrap())
                    } else {
                        confidence
                    };

                    RankedMemory::new(episode, activation, decayed_confidence, similarity, &self.config)
                })
            })
            .filter(|r| r.confidence.raw() >= self.config.min_confidence)
            .collect();

        // ... existing sorting and truncation ...
        ranked
    }
}
```

**Builder Integration**:
```rust
impl CognitiveRecallBuilder {
    pub fn decay_system(mut self, system: Arc<BiologicalDecaySystem>) -> Self {
        self.decay_system = Some(system);
        self
    }
}
```

**Decay System Lazy Evaluation Helper**:
```rust
// engram-core/src/decay/mod.rs
impl BiologicalDecaySystem {
    /// Apply decay to confidence based on elapsed time
    /// This is the lazy evaluation entry point called during recall
    pub fn compute_decayed_confidence(
        &self,
        base_confidence: Confidence,
        elapsed_time: Duration,
        decay_rate: f32,
        access_count: u64,
    ) -> Confidence {
        // Use appropriate decay function based on memory characteristics
        let decay_factor = if access_count > 3 {
            // Frequently accessed -> slower neocortical decay
            self.neocortical.compute_decay(elapsed_time, decay_rate)
        } else {
            // Infrequently accessed -> faster hippocampal decay
            self.hippocampal.compute_decay(elapsed_time, decay_rate)
        };

        Confidence::from_raw((base_confidence.raw() * decay_factor).max(0.0).min(1.0))
    }
}
```

## Acceptance Criteria

- [ ] `CognitiveRecall` accepts optional `BiologicalDecaySystem`
- [ ] Decay applied lazily during result ranking, not background threads
- [ ] Confidence scores correctly decayed based on `last_access` and elapsed time
- [ ] Decay system selection based on access patterns (hippocampal vs neocortical)
- [ ] Zero performance impact when decay system not configured
- [ ] Decay application takes <1ms per memory on average
- [ ] Integration tests showing decay working end-to-end
- [ ] No memory leaks during long-running decay operations

## Testing Approach

**Integration Tests**:
```rust
#[tokio::test]
async fn test_decay_applied_during_recall() {
    let store = MemoryStore::new_temp();
    let decay_system = Arc::new(BiologicalDecaySystem::default());

    let recall = CognitiveRecallBuilder::new()
        .vector_seeder(seeder)
        .spreading_engine(engine)
        .decay_system(decay_system)
        .build()
        .unwrap();

    // Store memory with high confidence
    let episode = Episode::new("test content").with_confidence(Confidence::from_raw(0.9));
    let episode_id = store.insert_episode(episode);

    // Simulate time passing
    sleep(Duration::from_secs(86400)); // 1 day

    // Recall should return decayed confidence
    let results = recall.recall(&cue, &store).unwrap();
    assert!(results[0].confidence.raw() < 0.9, "Confidence should decay over time");
}

#[tokio::test]
async fn test_decay_respects_access_patterns() {
    // Test that frequently accessed memories decay slower (neocortical)
    // vs infrequently accessed (hippocampal)
}

#[test]
fn test_decay_performance() {
    // Benchmark decay computation time
    // Must be <1ms average
}
```

**Performance Tests**:
- Measure latency overhead of decay computation
- Verify no memory leaks with long-running recall operations
- Profile decay system to identify hot paths

## Risk Mitigation

**Risk**: Decay computation adds latency to recall operations
**Mitigation**: Profile decay functions. Cache decay factors for common time intervals. Use SIMD for batch decay computation if needed.

**Risk**: Decay modifies episode confidence in-place causing side effects
**Mitigation**: Apply decay to result confidence only, don't mutate stored episode confidence. Store original confidence separately.

**Risk**: Different decay rates for different memories causes complexity
**Mitigation**: Start with single decay rate per system (hippocampal vs neocortical). Add per-memory rates in future task if needed.

**Risk**: Background consolidation conflicts with lazy decay
**Mitigation**: Consolidation uses stored confidence. Lazy decay only affects query results. Clear separation of concerns.

## Notes

This task implements the core lazy decay mechanism required by Milestone 4. The key insight is that decay is computed *during recall* based on time since `last_access`, not by background threads updating stored values.

**Design Principle**: Decay is a view-time transformation, not a storage-time mutation. This keeps storage immutable and recall deterministic.

**Performance Target**: <1ms p95 for decay computation per memory. With 100 results, adds <100ms overhead.
