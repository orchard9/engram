# Task 002: Vector-Similarity Activation Seeding

## Objective
Implement activation seeding from HNSW similarity search results, bridging vector operations with graph spreading.

## Priority
P0 (Critical Path)

## Effort Estimate
1.5 days

## Dependencies
- Task 001: Storage-Aware Activation Interface

## Technical Approach

### Implementation Details
- Create `VectorActivationSeeder` that converts HNSW results to initial activation
- Map cosine similarity scores to activation levels using sigmoid normalization
- Implement multi-vector cue seeding for complex queries
- Add seeding confidence based on HNSW index confidence metrics

### Files to Create/Modify
- `engram-core/src/activation/seeding.rs` - New file for vector-based activation seeding
- `engram-core/src/activation/mod.rs` - Export seeding functionality
- `engram-core/src/store.rs` - Integrate seeding with HNSW results

### Integration Points
- Uses existing HNSW implementation from Milestone 1
- Integrates with `HnswIndex::search()` results
- Leverages SIMD similarity computations from `compute` module
- Connects to storage-aware activation from Task 001

## Implementation Details

### VectorActivationSeeder Structure
```rust
pub struct VectorActivationSeeder {
    similarity_threshold: f32,
    activation_decay: f32,
    max_seeds: usize,
}

impl VectorActivationSeeder {
    pub fn seed_from_cue(&self, cue: &Cue) -> Vec<StorageAwareActivation> {
        // Convert HNSW similarity to activation level
        // Apply sigmoid normalization
        // Create storage-aware activation records
    }
}
```

### Activation Mapping Function
- Sigmoid normalization: `activation = 1.0 / (1.0 + exp(-k * (similarity - threshold)))`
- Confidence inheritance from HNSW index confidence
- Distance-based decay for multiple vector cues

## Acceptance Criteria
- [ ] `VectorActivationSeeder` converts HNSW results to activation records
- [ ] Sigmoid normalization produces reasonable activation distributions
- [ ] Multi-vector cues properly aggregate into combined activation
- [ ] Seeding confidence correctly reflects HNSW index confidence
- [ ] Integration with existing HNSW search maintains performance
- [ ] Activation levels correlate with vector similarity scores

## Testing Approach
- Unit tests for sigmoid normalization and activation mapping
- Integration tests with various vector similarity ranges
- Performance tests ensuring seeding adds <1ms overhead
- Property tests validating activation bounds [0,1]
- A/B tests comparing seeded vs non-seeded spreading results

## Risk Mitigation
- **Risk**: Poor activation normalization leads to ineffective spreading
- **Mitigation**: Empirical tuning of sigmoid parameters against test datasets
- **Monitoring**: Track activation distribution statistics and spreading effectiveness

- **Risk**: HNSW integration performance degradation
- **Mitigation**: Batch processing, reuse existing HNSW results
- **Testing**: Benchmark HNSW + seeding vs HNSW alone

## Notes
This task bridges vector similarity (the foundation of modern vector databases) with cognitive activation spreading (the distinctive feature of Engram). The quality of this mapping directly impacts how well the system performs cognitive recall vs simple similarity search.