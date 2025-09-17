# Task 004: Columnar Cold Storage with SIMD

## Status: Pending
## Priority: P2 - Optimization
## Estimated Effort: 1 day
## Dependencies: Task 001 (cold tier implementation)

## Objective
Optimize cold tier storage with columnar layout for SIMD batch operations on large vector datasets.

## Current State Analysis
- **Existing**: SIMD operations in `compute/` module
- **Existing**: Cold tier structure from Task 001
- **Missing**: Columnar data layout for vectors
- **Missing**: SIMD batch similarity operations

## Implementation Plan

### Enhance Cold Tier (engram-core/src/storage/cold_tier.rs from Task 001)
```rust
use crate::compute::VectorOps;

pub struct ColdTier {
    // Columnar layout: each dimension as separate vector
    columns: Vec<Vec<f32>>,     // 768 columns of f32 values
    row_metadata: Vec<MemoryMetadata>,
    id_to_row: DashMap<String, usize>,
    vector_ops: Box<dyn VectorOps>,
}

impl ColdTier {
    pub fn append_batch(&mut self, memories: &[Memory]) -> Result<(), StorageError> {
        let start_row = self.row_metadata.len();
        
        // Transpose vectors into columnar format
        for dim in 0..768 {
            for memory in memories {
                self.columns[dim].push(memory.episode.embedding[dim]);
            }
        }
        
        // Update metadata and index
        for (i, memory) in memories.iter().enumerate() {
            self.row_metadata.push(MemoryMetadata::from_memory(memory));
            self.id_to_row.insert(memory.episode.id.clone(), start_row + i);
        }
        
        Ok(())
    }
    
    pub fn simd_similarity_search(
        &self,
        query: &[f32; 768],
        k: usize,
    ) -> Vec<(String, f32)> {
        let num_vectors = self.row_metadata.len();
        let mut similarities = vec![0.0f32; num_vectors];
        
        // SIMD dot product across all vectors simultaneously
        for dim in 0..768 {
            let query_val = query[dim];
            let column = &self.columns[dim];
            
            // Use existing SIMD implementation from compute module
            self.vector_ops.fma_accumulate(column, query_val, &mut similarities);
        }
        
        // Convert to similarities and get top-k
        let mut results: Vec<_> = similarities.iter()
            .enumerate()
            .map(|(idx, &sim)| {
                let memory_id = self.get_memory_id_by_row(idx);
                (memory_id, sim)
            })
            .collect();
            
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        results.truncate(k);
        results
    }
}

impl StorageTier for ColdTier {
    async fn recall(&self, cue: &Cue) -> Result<Vec<(Episode, Confidence)>, Self::Error> {
        match &cue.cue_type {
            CueType::Embedding { vector, threshold } => {
                let results = self.simd_similarity_search(vector, 100);
                
                Ok(results.into_iter()
                    .filter(|(_, sim)| *sim >= *threshold)
                    .map(|(id, sim)| {
                        let episode = self.get_episode_by_id(&id);
                        (episode, Confidence::exact(sim))
                    })
                    .collect())
            }
            _ => self.scan_recall(cue) // Fallback for other cue types
        }
    }
}
```

## Acceptance Criteria
- [ ] Columnar layout reduces memory bandwidth >40%
- [ ] SIMD operations achieve >4x speedup over scalar
- [ ] Batch operations process >50K vectors/second
- [ ] Memory overhead <20% vs row storage

## Performance Targets
- SIMD similarity search: <10ms for 100K vectors
- Batch append: <1ms for 1000 vectors
- Memory layout: <10% overhead for metadata

## Risk Mitigation
- Reuse existing SIMD operations from compute module
- Fallback to row storage if performance doesn't improve
- Gradual migration from row to columnar format