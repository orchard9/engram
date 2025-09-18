# Task 007: SIMD-Optimized Batch Spreading

## Objective
Integrate SIMD vector operations into spreading for batch activation processing and similarity computations.

## Priority
P1 (Performance Critical)

## Effort Estimate
1.5 days

## Dependencies
- Task 004: Confidence Aggregation Engine

## Technical Approach

### Implementation Details
- Create `SIMDBatchSpreader` using existing `compute::cosine_similarity_768`
- Implement vectorized activation accumulation for multiple targets
- Add batch confidence aggregation using SIMD operations
- Optimize memory layout for cache-efficient SIMD access patterns

### Files to Create/Modify
- `engram-core/src/activation/simd_spreading.rs` - New file for SIMD batch operations
- `engram-core/src/activation/mod.rs` - Export SIMD spreading functionality
- `engram-core/src/compute/mod.rs` - Add batch activation operations

### Integration Points
- Uses existing SIMD operations from `engram-core/src/compute/`
- Integrates with confidence aggregation from Task 004
- Leverages columnar storage layout from Milestone 2
- Connects to tier-aware scheduler for batch processing

## Implementation Details

### SIMDBatchSpreader Structure
```rust
pub struct SIMDBatchSpreader {
    batch_size: usize,
    similarity_threshold: f32,
    vector_ops: Arc<dyn VectorOps>,
}

impl SIMDBatchSpreader {
    pub fn spread_batch_simd(
        &self,
        source_embedding: &[f32; 768],
        target_embeddings: &[[f32; 768]],
        target_activations: &mut [f32],
    ) -> SIMDSpreadingResults {
        // Use existing SIMD cosine similarity
        // Vectorized activation accumulation
        // Batch confidence propagation
    }
}
```

### SIMD Operations Integration
- **Cosine Similarity**: Use `compute::cosine_similarity_768` for batch similarity
- **Activation Accumulation**: Vectorized addition using AVX2/AVX-512 instructions
- **Confidence Multiplication**: SIMD confidence decay and aggregation
- **Memory Layout**: Structure-of-Arrays (SoA) for optimal SIMD access

### Batch Processing Strategy
1. **Gather Phase**: Collect target embeddings into SIMD-friendly layout
2. **Compute Phase**: Batch similarity computation using SIMD operations
3. **Accumulate Phase**: Vectorized activation accumulation and confidence updates
4. **Scatter Phase**: Write results back to individual activation records

### Cache Optimization
- **Prefetch**: Prefetch next batch while processing current batch
- **Alignment**: Ensure embeddings aligned for SIMD operations
- **Blocking**: Process in cache-friendly blocks of 8-16 vectors
- **Temporal Locality**: Reuse computed similarities for confidence aggregation

## Acceptance Criteria
- [ ] Batch similarity computation uses existing SIMD operations
- [ ] Vectorized activation accumulation properly implemented
- [ ] SIMD confidence aggregation maintains mathematical correctness
- [ ] Memory layout optimized for cache efficiency
- [ ] Performance improvement >2x over scalar implementation
- [ ] Batch size auto-tuning based on CPU capabilities
- [ ] Correctness validation against scalar reference implementation

## Testing Approach
- Unit tests comparing SIMD vs scalar batch operations
- Property tests ensuring mathematical equivalence
- Performance benchmarks measuring SIMD speedup
- Cache performance analysis using hardware counters
- Stress tests with various batch sizes and vector patterns

## Risk Mitigation
- **Risk**: SIMD operations introduce numerical differences vs scalar
- **Mitigation**: Extensive validation against scalar reference, tolerance bounds
- **Testing**: Bit-level comparison for simple cases, statistical validation for complex

- **Risk**: Memory alignment issues causing SIMD faults
- **Mitigation**: Proper alignment assertions, fallback to scalar for unaligned data
- **Monitoring**: Track SIMD fault rates and fallback frequency

- **Risk**: Cache thrashing with large batch sizes
- **Mitigation**: Adaptive batch sizing, cache-aware blocking strategy
- **Testing**: Performance testing with various batch sizes and memory patterns

## Implementation Strategy

### Phase 1: Basic SIMD Integration
- Integrate existing `cosine_similarity_768` into spreading
- Simple batch processing with fixed batch sizes
- Validation against scalar implementation

### Phase 2: Vectorized Accumulation
- Implement SIMD activation accumulation
- Add vectorized confidence operations
- Cache-friendly memory layout

### Phase 3: Performance Optimization
- Adaptive batch sizing based on CPU capabilities
- Prefetching and cache optimization
- Auto-tuning for different vector distributions

## Performance Targets
- **Speedup**: >2x improvement over scalar spreading
- **Batch Size**: Auto-tune between 8-64 vectors per batch
- **Cache Efficiency**: >90% L1 cache hit rate for embeddings
- **Throughput**: Process >10K activations/second per core

## Notes
This task leverages the SIMD foundations from Milestone 1 to dramatically accelerate the core computational kernel of cognitive spreading. Unlike traditional databases that process records individually, cognitive spreading benefits enormously from batch vector operations, making SIMD optimization critical for achieving the performance targets needed for real-time cognitive applications.