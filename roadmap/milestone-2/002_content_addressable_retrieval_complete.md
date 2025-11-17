# Task 002: Content-Addressable Vector Retrieval

## Status: Partial ✅
## Priority: P1 - Performance Critical
## Estimated Effort: 1.5 days
## Dependencies: Task 001 (tier implementations)

## Objective
Add content-addressable retrieval to enable direct lookup by semantic content rather than IDs, with automatic deduplication and similarity-based addressing.

- ✅ `ContentAddress` and `ContentIndex` implemented with quantization-free hash + LSH buckets (`engram-core/src/storage/content_addressing.rs:1-210`).
- ✅ Deduplication strategies (`Skip`/`Replace`/`Merge`) integrate with the store when inserting memories (`engram-core/src/storage/deduplication.rs:360-520`, `engram-core/src/store.rs:292-360`).
- ✅ Memory store now generates addresses during `store()` and updates the content index for retrieval (`engram-core/src/store.rs:292-360`).
- ⚠️ `ContentAddress::from_embedding` still copies raw float bytes rather than using the BLAKE3 hash/quantization pipeline described in the spec (`engram-core/src/storage/content_addressing.rs:23-65`).
- ⚠️ Store lacks a fast-path lookup by `ContentAddress`; retrieval still walks hot memories unless HNSW results help. Needs a direct API that accepts a content address and resolves matching memory IDs.
- ⚠️ Deduplication action logging / metrics requested in spec are not present.

## Remaining Work for Completion
1. Replace the ad-hoc hash with the documented approach:
   - Quantize vectors to bytes and compute BLAKE3 hash + deterministic LSH (update `ContentAddress::from_embedding` in `engram-core/src/storage/content_addressing.rs`).
2. Expose content-address lookup path to callers:
   - Add `MemoryStore::get_by_content_address` (or similar) that consults `ContentIndex` without scanning `hot_memories` (`engram-core/src/store.rs`).
3. Emit instrumentation for dedup decisions:
   - Record metrics/log entries when `DeduplicationAction::{Skip,Replace,Merge}` fires (`engram-core/src/storage/deduplication.rs`).

## Current State Analysis
- **Existing**: HNSW index for similarity search in `index/` module
- **Existing**: SIMD-optimized similarity from `compute/` module
- **Missing**: Content addressing scheme for vectors
- **Missing**: Deduplication based on vector similarity

## Implementation Plan

### Create New Files
1. `engram-core/src/storage/content_addressing.rs` - Content address generation
2. `engram-core/src/storage/deduplication.rs` - Similarity-based deduplication

### Modify Existing Files
- `engram-core/src/storage/mod.rs:45` - Add content addressing re-exports
- `engram-core/src/store.rs:150-180` - Integrate content addressing in store()

## Implementation Details

### Content Addressing (engram-core/src/storage/content_addressing.rs)
```rust
use blake3::Hasher;
use crate::Confidence;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ContentAddress {
    semantic_hash: [u8; 32],    // Blake3 hash of quantized vector
    lsh_bucket: u32,            // Locality-sensitive hash bucket
    confidence: Confidence,      // Uniqueness confidence
}

impl ContentAddress {
    pub fn from_embedding(embedding: &[f32; 768]) -> Self {
        // Quantize for stable hashing
        let quantized: Vec<u8> = embedding.iter()
            .map(|&v| ((v + 1.0) * 127.5) as u8)
            .collect();
            
        let mut hasher = Hasher::new();
        hasher.update(&quantized);
        let semantic_hash = *hasher.finalize().as_bytes();
        
        // Simple LSH using sign of random projections
        let lsh_bucket = Self::compute_lsh_bucket(embedding);
        
        Self {
            semantic_hash,
            lsh_bucket,
            confidence: Confidence::HIGH,
        }
    }
    
    fn compute_lsh_bucket(embedding: &[f32; 768]) -> u32 {
        // 32-bit LSH using predetermined random vectors
        let mut bucket = 0u32;
        for i in 0..32 {
            let dot = Self::random_projection(embedding, i);
            if dot > 0.0 {
                bucket |= 1 << i;
            }
        }
        bucket
    }
}
```

### Deduplication (engram-core/src/storage/deduplication.rs)
```rust
use crate::{Memory, Confidence};
use std::collections::HashMap;

pub struct SemanticDeduplicator {
    similarity_threshold: f32,
    merge_strategy: MergeStrategy,
}

pub enum MergeStrategy {
    KeepHighestConfidence,
    MergeMetadata,
    CreateComposite,
}

impl SemanticDeduplicator {
    pub fn check_duplicate(
        &self,
        new_memory: &Memory,
        existing: &[Memory],
    ) -> Option<usize> {
        for (idx, existing_memory) in existing.iter().enumerate() {
            let similarity = crate::compute::cosine_similarity_768(
                &new_memory.episode.embedding,
                &existing_memory.episode.embedding,
            );
            
            if similarity >= self.similarity_threshold {
                return Some(idx);
            }
        }
        None
    }
}
```

### Integration with MemoryStore (engram-core/src/store.rs:150-180)
```rust
// Add to MemoryStore struct around line 95:
pub struct MemoryStore {
    // ... existing fields ...
    content_addresses: DashMap<ContentAddress, String>,
    deduplicator: SemanticDeduplicator,
}

// Update store() method around line 150:
pub fn store(&mut self, episode: Episode) -> Result<String> {
    let memory = Memory::from_episode(episode);
    
    // Generate content address
    let content_address = ContentAddress::from_embedding(&memory.episode.embedding);
    
    // Check for duplicates
    if let Some(existing_id) = self.content_addresses.get(&content_address) {
        // Handle duplicate according to strategy
        return self.handle_duplicate(&memory, existing_id.clone());
    }
    
    // Store new unique content
    let memory_id = memory.episode.id.clone();
    self.content_addresses.insert(content_address, memory_id.clone());
    
    // Continue with existing storage logic
    self.episodes.write().insert(memory_id.clone(), memory.episode.clone());
    self.hnsw_index.insert_memory(Arc::new(memory))?;
    
    Ok(memory_id)
}
```

## Acceptance Criteria
- [ ] Content addressing generates stable hashes for same content
- [ ] LSH groups similar vectors (>0.9 similarity) in same bucket
- [ ] Deduplication prevents storing near-identical vectors
- [ ] Content retrieval maintains <1ms latency
- [ ] Zero false negatives for exact content matches

## Performance Targets
- Content address generation: <10μs
- Deduplication check: <100μs for existing vectors
- LSH bucket lookup: <50μs
- Memory overhead: <10% for addressing metadata

## Risk Mitigation
- Use proven Blake3 for cryptographic hashing
- Simple LSH implementation for first iteration
- Tunable deduplication thresholds
- Metrics for deduplication effectiveness
