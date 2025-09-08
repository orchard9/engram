# Task 002: Content-Addressable Vector Retrieval System

## Status: Pending
## Priority: P0 - Critical Path
## Estimated Effort: 4 days
## Dependencies: Task 001 (three-tier storage)

## Objective
Implement content-addressable retrieval using vector embeddings as addresses, enabling direct lookup by semantic content rather than IDs, with automatic deduplication and similarity-based addressing.

## Current State Analysis
- **Existing**: HNSW index for similarity search (milestone-1/task-002)
- **Existing**: SIMD-optimized cosine similarity (milestone-1/task-001)
- **Missing**: Content-based addressing scheme
- **Missing**: Semantic hashing for fast lookup
- **Missing**: Deduplication based on vector similarity

## Technical Specification

### 1. Content Addressing Scheme

```rust
// engram-core/src/storage/content_addressing.rs

use blake3::Hasher;

/// Content address derived from vector embedding
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ContentAddress {
    /// Semantic hash of the vector
    semantic_hash: [u8; 32],
    /// Locality-sensitive hash for similarity
    lsh_bucket: u32,
    /// Confidence in address uniqueness
    uniqueness_confidence: Confidence,
}

impl ContentAddress {
    /// Generate address from embedding
    pub fn from_embedding(embedding: &[f32; 768]) -> Self {
        // Semantic hash using quantization + blake3
        let semantic_hash = Self::semantic_hash(embedding);
        
        // LSH for similarity-based bucketing
        let lsh_bucket = Self::lsh_hash(embedding);
        
        // Calculate uniqueness based on local density
        let uniqueness_confidence = Self::estimate_uniqueness(embedding);
        
        Self {
            semantic_hash,
            lsh_bucket,
            uniqueness_confidence,
        }
    }
    
    fn semantic_hash(embedding: &[f32; 768]) -> [u8; 32] {
        // Quantize to 8-bit for stable hashing
        let quantized: Vec<u8> = embedding.iter()
            .map(|&v| ((v + 1.0) * 127.5) as u8)
            .collect();
            
        let mut hasher = Hasher::new();
        hasher.update(&quantized);
        *hasher.finalize().as_bytes()
    }
    
    fn lsh_hash(embedding: &[f32; 768]) -> u32 {
        // Random projection LSH
        const NUM_HYPERPLANES: usize = 32;
        let mut bits = 0u32;
        
        for i in 0..NUM_HYPERPLANES {
            let dot = Self::random_projection(embedding, i);
            if dot > 0.0 {
                bits |= 1 << i;
            }
        }
        bits
    }
}
```

### 2. Content-Addressable Store

```rust
// engram-core/src/storage/content_store.rs

pub struct ContentAddressableStore {
    /// Primary content index
    content_index: DashMap<ContentAddress, VectorRecord>,
    
    /// LSH buckets for similarity queries
    lsh_buckets: DashMap<u32, Vec<ContentAddress>>,
    
    /// Deduplication threshold
    dedup_threshold: f32,
    
    /// Backend storage tiers
    storage: TieredStorage,
}

impl ContentAddressableStore {
    /// Store by content with automatic deduplication
    pub fn store_by_content(
        &self,
        embedding: [f32; 768],
        metadata: VectorMetadata,
    ) -> Result<ContentAddress> {
        let address = ContentAddress::from_embedding(&embedding);
        
        // Check for near-duplicates
        if let Some(similar) = self.find_similar(&embedding, self.dedup_threshold)? {
            // Merge metadata instead of storing duplicate
            self.merge_metadata(&similar, metadata)?;
            return Ok(similar);
        }
        
        // Store new unique content
        let record = VectorRecord {
            embedding,
            metadata,
            storage_tier: StorageTier::Hot,
        };
        
        self.content_index.insert(address.clone(), record);
        self.lsh_buckets.entry(address.lsh_bucket)
            .or_default()
            .push(address.clone());
            
        Ok(address)
    }
    
    /// Retrieve by content address
    pub fn retrieve_by_content(
        &self,
        address: &ContentAddress,
    ) -> Result<Option<(Vec<f32>, Confidence)>> {
        if let Some(record) = self.content_index.get(address) {
            let confidence = self.calculate_retrieval_confidence(&record);
            Ok(Some((record.embedding.to_vec(), confidence)))
        } else {
            Ok(None)
        }
    }
    
    /// Find similar content within threshold
    pub fn find_similar(
        &self,
        embedding: &[f32; 768],
        threshold: f32,
    ) -> Result<Option<ContentAddress>> {
        let query_address = ContentAddress::from_embedding(embedding);
        
        // Check LSH bucket for candidates
        if let Some(bucket) = self.lsh_buckets.get(&query_address.lsh_bucket) {
            for candidate_addr in bucket.iter() {
                if let Some(record) = self.content_index.get(candidate_addr) {
                    let similarity = crate::compute::cosine_similarity_768(
                        embedding,
                        &record.embedding,
                    );
                    
                    if similarity >= threshold {
                        return Ok(Some(candidate_addr.clone()));
                    }
                }
            }
        }
        
        Ok(None)
    }
}
```

### 3. Semantic Deduplication

```rust
// engram-core/src/storage/deduplication.rs

pub struct SemanticDeduplicator {
    /// Similarity threshold for considering duplicates
    similarity_threshold: f32,
    
    /// Merge strategy for duplicate metadata
    merge_strategy: MergeStrategy,
    
    /// Statistics tracking
    stats: DeduplicationStats,
}

pub enum MergeStrategy {
    /// Keep the most recent
    KeepNewest,
    /// Keep the highest confidence
    KeepHighestConfidence,
    /// Merge metadata fields
    MergeMetadata,
    /// Create composite with references
    CreateComposite,
}

impl SemanticDeduplicator {
    /// Check if vector is duplicate
    pub fn is_duplicate(
        &self,
        new_vector: &[f32; 768],
        existing: &[f32; 768],
    ) -> bool {
        let similarity = crate::compute::cosine_similarity_768(new_vector, existing);
        similarity >= self.similarity_threshold
    }
    
    /// Merge duplicate records
    pub fn merge_records(
        &self,
        existing: &mut VectorRecord,
        new: VectorRecord,
    ) -> Result<()> {
        match self.merge_strategy {
            MergeStrategy::KeepNewest => {
                if new.metadata.timestamp > existing.metadata.timestamp {
                    *existing = new;
                }
            }
            MergeStrategy::KeepHighestConfidence => {
                if new.metadata.confidence > existing.metadata.confidence {
                    *existing = new;
                }
            }
            MergeStrategy::MergeMetadata => {
                existing.metadata.merge(new.metadata);
                // Average embeddings for composite representation
                for i in 0..768 {
                    existing.embedding[i] = 
                        (existing.embedding[i] + new.embedding[i]) / 2.0;
                }
            }
            MergeStrategy::CreateComposite => {
                // Create reference to both
                existing.metadata.add_reference(new.metadata.id);
            }
        }
        
        self.stats.record_deduplication();
        Ok(())
    }
}
```

### 4. Integration with Cue Operations

```rust
// Modify engram-core/src/cue.rs

impl Cue {
    /// Create cue from content address
    pub fn from_content_address(address: ContentAddress) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            cue_type: CueType::ContentAddress(address),
            threshold: Confidence::HIGH,
            max_results: 10,
            spread_activation: true,
            activation_decay: 0.8,
        }
    }
    
    /// Query by semantic content
    pub fn query_by_content(
        &self,
        store: &ContentAddressableStore,
    ) -> Result<Vec<(Episode, Confidence)>> {
        match &self.cue_type {
            CueType::ContentAddress(address) => {
                store.retrieve_by_content(address)
                    .map(|opt| opt.map_or(Vec::new(), |v| vec![v]))
            }
            CueType::Embedding { vector, threshold } => {
                // Find all similar content
                store.find_all_similar(vector, *threshold)
            }
            _ => Ok(Vec::new()),
        }
    }
}
```

## Integration Points

### Modify MemoryStore (store.rs)
```rust
// Add around line 95:
pub struct MemoryStore {
    // ... existing fields ...
    
    // Add content-addressable store
    content_store: Arc<ContentAddressableStore>,
    
    // Deduplication settings
    deduplicator: SemanticDeduplicator,
}

// Update store() method around line 160:
pub fn store(&mut self, episode: Episode) -> Result<String> {
    // Generate content address
    let content_address = ContentAddress::from_embedding(&episode.embedding);
    
    // Check for duplicates
    if let Some(existing_addr) = self.content_store.find_similar(
        &episode.embedding,
        self.deduplicator.similarity_threshold,
    )? {
        // Handle duplicate according to strategy
        self.deduplicator.handle_duplicate(existing_addr, episode)?;
        return Ok(existing_addr.to_string());
    }
    
    // Store new unique content
    self.content_store.store_by_content(
        episode.embedding,
        VectorMetadata::from_episode(&episode),
    )?;
    
    // Rest of existing logic...
}
```

## Testing Strategy

### Unit Tests
```rust
#[test]
fn test_content_addressing() {
    let embedding = [0.5f32; 768];
    let address1 = ContentAddress::from_embedding(&embedding);
    let address2 = ContentAddress::from_embedding(&embedding);
    
    // Same content produces same address
    assert_eq!(address1, address2);
    
    // Similar content produces nearby addresses
    let mut similar = embedding;
    similar[0] = 0.51;
    let address3 = ContentAddress::from_embedding(&similar);
    
    // Should have same LSH bucket
    assert_eq!(address1.lsh_bucket, address3.lsh_bucket);
}

#[test]
fn test_semantic_deduplication() {
    let store = ContentAddressableStore::new();
    
    let vec1 = [0.5f32; 768];
    let vec2 = {
        let mut v = vec1;
        v[0] = 0.501; // Very similar
        v
    };
    
    let addr1 = store.store_by_content(vec1, Default::default()).unwrap();
    let addr2 = store.store_by_content(vec2, Default::default()).unwrap();
    
    // Should deduplicate to same address
    assert_eq!(addr1, addr2);
}
```

### Property Tests
```rust
use proptest::prelude::*;

proptest! {
    #[test]
    fn content_address_stability(
        embedding in prop::array::uniform768(any::<f32>())
    ) {
        let addr1 = ContentAddress::from_embedding(&embedding);
        let addr2 = ContentAddress::from_embedding(&embedding);
        
        prop_assert_eq!(addr1, addr2);
    }
    
    #[test]
    fn lsh_preserves_similarity(
        base in prop::array::uniform768(0.0f32..1.0),
        noise_level in 0.0f32..0.1
    ) {
        let similar = add_noise(&base, noise_level);
        
        let addr1 = ContentAddress::from_embedding(&base);
        let addr2 = ContentAddress::from_embedding(&similar);
        
        if noise_level < 0.05 {
            // Very similar vectors should hash to same bucket
            prop_assert_eq!(addr1.lsh_bucket, addr2.lsh_bucket);
        }
    }
}
```

## Acceptance Criteria
- [ ] Content addressing generates stable hashes for same content
- [ ] LSH bucketing groups similar vectors (>0.9 similarity)
- [ ] Deduplication prevents storing near-identical vectors
- [ ] Retrieval by content address maintains <1ms latency
- [ ] Merge strategies preserve important metadata
- [ ] Zero false negatives for exact content matches

## Performance Targets
- Content address generation: <10μs
- Deduplication check: <100μs for 1M vectors
- Content retrieval: <1ms including similarity check
- LSH bucket lookup: <50μs
- Memory overhead: <100 bytes per vector for indexing

## Risk Mitigation
- Use proven LSH techniques (random projection)
- Implement tunable deduplication thresholds
- Add metrics for deduplication effectiveness
- Provide bypass for critical unique content