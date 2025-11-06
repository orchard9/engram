# Task 012: Performance Optimization

## Objective
Optimize critical paths in the dual memory system to minimize overhead and maintain sub-5% performance regression compared to single-type memory.

## Background
The dual memory architecture adds complexity. We need targeted optimizations to maintain acceptable performance.

## Requirements
1. Profile dual memory operations
2. Optimize hot paths with SIMD where applicable
3. Implement caching for frequent lookups
4. Reduce memory allocations
5. Parallelize independent operations

## Technical Specification

### Files to Create
- `engram-core/src/optimization/dual_memory_cache.rs` - Caching layer

### Files to Modify
- `engram-core/src/activation/simd_optimization.rs` - SIMD for concepts
- `engram-core/src/memory/dual_types.rs` - Zero-copy optimizations

### Key Optimizations
```rust
/// Cache for concept lookups
pub struct ConceptCache {
    // LRU cache for concept embeddings
    embedding_cache: Mutex<LruCache<NodeId, Arc<[f32; 768]>>>,
    // Pre-computed fan-out counts
    fan_counts: DashMap<NodeId, AtomicUsize>,
    // Binding strength cache
    binding_cache: DashMap<(NodeId, NodeId), f32>,
}

impl ConceptCache {
    pub fn get_embedding(&self, concept_id: &NodeId) -> Option<Arc<[f32; 768]>> {
        // Fast path: check cache first
        if let Some(embedding) = self.embedding_cache.lock().get(concept_id) {
            return Some(embedding.clone());
        }
        None
    }
    
    pub fn get_fan_count(&self, concept_id: &NodeId) -> usize {
        self.fan_counts
            .get(concept_id)
            .map(|count| count.load(Ordering::Relaxed))
            .unwrap_or_else(|| self.compute_fan_count(concept_id))
    }
}

/// SIMD-optimized concept operations
pub mod simd_concepts {
    use std::arch::x86_64::*;
    
    #[target_feature(enable = "avx2")]
    pub unsafe fn batch_concept_similarity(
        concept_centroid: &[f32; 768],
        episode_embeddings: &[[f32; 768]],
    ) -> Vec<f32> {
        let mut results = Vec::with_capacity(episode_embeddings.len());
        
        // Process 8 episodes at a time with AVX2
        for chunk in episode_embeddings.chunks(8) {
            let similarities = compute_batch_cosine_similarity_avx2(
                concept_centroid,
                chunk,
            );
            results.extend_from_slice(&similarities);
        }
        
        results
    }
}

/// Zero-copy node type checking
impl DualMemoryNode {
    #[inline(always)]
    pub fn is_concept(&self) -> bool {
        matches!(self.node_type, MemoryNodeType::Concept { .. })
    }
    
    #[inline(always)]
    pub fn as_concept_embedding(&self) -> Option<&[f32; 768]> {
        match &self.node_type {
            MemoryNodeType::Concept { centroid, .. } => Some(centroid),
            _ => None,
        }
    }
}
```

### Parallel Operations
```rust
use rayon::prelude::*;

impl ConceptFormationEngine {
    pub fn parallel_clustering(&self, episodes: &[Episode]) -> Vec<Cluster> {
        // Parallel pairwise similarity computation
        let similarities = episodes
            .par_iter()
            .enumerate()
            .flat_map(|(i, ep1)| {
                episodes[i + 1..].par_iter().enumerate().map(move |(j, ep2)| {
                    let similarity = cosine_similarity(&ep1.embedding, &ep2.embedding);
                    (i, i + j + 1, similarity)
                })
            })
            .collect::<Vec<_>>();
        
        // Use similarity matrix for clustering
        self.cluster_from_similarities(episodes.len(), similarities)
    }
}
```

## Implementation Notes
- Profile before optimizing
- Maintain correctness over speed
- Use feature flags for SIMD code
- Monitor memory usage growth

## Testing Approach
1. Micro-benchmarks for each optimization
2. End-to-end performance tests
3. Memory usage profiling
4. Correctness validation

## Acceptance Criteria
- [ ] Overall performance regression <5%
- [ ] Concept lookup latency <100μs
- [ ] Memory overhead <20%
- [ ] SIMD paths 2x faster
- [ ] No correctness regressions

## Dependencies
- All previous dual memory tasks
- Performance profiling tools

## Estimated Time
3 days