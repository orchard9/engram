# Task 001: Three-Tier Storage Architecture for Vector-Native Operations

## Status: Pending
## Priority: P0 - Critical Path
## Estimated Effort: 5 days
## Dependencies: None (builds on milestone-1 memory-mapped persistence)

## Objective
Implement the three-tier storage architecture (hot/warm/cold) optimized for vector operations with automatic migration between tiers based on activation frequency and access patterns.

## Current State Analysis
- **Existing**: Basic memory-mapped persistence from milestone-1/task-003
- **Missing**: Tiered storage with different performance characteristics
- **Missing**: Automatic migration policies based on activation
- **Missing**: Lock-free concurrent hashmap for hot tier

## Technical Specification

### 1. Storage Tier Definitions

```rust
// engram-core/src/storage/tiered.rs

pub trait StorageTier: Send + Sync {
    /// Store a vector with metadata
    fn store(&self, id: &str, vector: &[f32; 768], metadata: StorageMetadata) -> Result<()>;
    
    /// Retrieve a vector with confidence
    fn retrieve(&self, id: &str) -> Result<(Vec<f32>, Confidence)>;
    
    /// Batch retrieval for efficiency
    fn retrieve_batch(&self, ids: &[String]) -> Result<Vec<(Vec<f32>, Confidence)>>;
    
    /// Migration support
    fn can_accept(&self, access_pattern: &AccessPattern) -> bool;
    fn migrate_to(&self, next_tier: &dyn StorageTier, ids: &[String]) -> Result<()>;
}

pub struct TieredStorage {
    hot: HotTier,    // Lock-free concurrent hashmap
    warm: WarmTier,  // Append-only log with compression
    cold: ColdTier,  // Columnar storage with SIMD
    migration_policy: MigrationPolicy,
}
```

### 2. Hot Tier Implementation

```rust
// engram-core/src/storage/hot_tier.rs

use dashmap::DashMap;
use std::sync::atomic::{AtomicU64, Ordering};

pub struct HotTier {
    /// Lock-free concurrent hashmap
    data: DashMap<String, HotVector>,
    /// Total memory usage tracking
    memory_usage: AtomicU64,
    /// Maximum memory before migration
    max_memory: u64,
}

#[repr(align(64))] // Cache-line aligned
struct HotVector {
    embedding: [f32; 768],
    activation: AtomicF32,
    last_access: AtomicU64,
    access_count: AtomicU32,
    confidence: Confidence,
}
```

### 3. Warm Tier Implementation

```rust
// engram-core/src/storage/warm_tier.rs

pub struct WarmTier {
    /// Append-only log for sequential writes
    log: AppendOnlyLog,
    /// Index for fast lookups
    index: BTreeMap<String, LogOffset>,
    /// Compression settings
    compression: CompressionStrategy,
}

impl WarmTier {
    /// Batch append with compression
    pub fn append_batch(&mut self, vectors: Vec<(String, [f32; 768])>) -> Result<()> {
        let compressed = self.compression.compress_batch(&vectors)?;
        let offset = self.log.append(&compressed)?;
        for (id, _) in vectors {
            self.index.insert(id, offset);
        }
        Ok(())
    }
}
```

### 4. Cold Tier Implementation

```rust
// engram-core/src/storage/cold_tier.rs

pub struct ColdTier {
    /// Columnar storage for SIMD operations
    columns: ColumnarVectorStorage,
    /// Memory-mapped files
    mmap: MmapVectorStore,
    /// Metadata index
    metadata: MetadataIndex,
}

/// Columnar layout for efficient SIMD
struct ColumnarVectorStorage {
    /// Each dimension stored contiguously
    dimensions: Vec<MmapVec<f32>>, // 768 columns
    /// Row mapping
    id_to_row: HashMap<String, usize>,
}
```

### 5. Migration Policies

```rust
// engram-core/src/storage/migration.rs

pub struct MigrationPolicy {
    /// Hot to warm threshold
    hot_to_warm_threshold: AccessThreshold,
    /// Warm to cold threshold
    warm_to_cold_threshold: AccessThreshold,
    /// Promotion thresholds
    cold_to_warm_threshold: AccessThreshold,
    warm_to_hot_threshold: AccessThreshold,
}

struct AccessThreshold {
    min_idle_time: Duration,
    max_access_count: u32,
    activation_threshold: f32,
}

impl MigrationPolicy {
    pub fn evaluate(&self, tier: StorageTier, stats: &AccessStats) -> MigrationDecision {
        match tier {
            StorageTier::Hot => {
                if stats.idle_time > self.hot_to_warm_threshold.min_idle_time {
                    MigrationDecision::Demote
                } else {
                    MigrationDecision::Stay
                }
            }
            // ... other tiers
        }
    }
}
```

## Integration Points

### Modify MemoryStore (store.rs)
```rust
// Replace line 89-92 with:
pub struct MemoryStore {
    episodes: Arc<RwLock<HashMap<String, Episode>>>,
    
    // Add tiered vector storage
    vector_storage: TieredStorage,
    
    // Keep existing fields
    index: Arc<RwLock<MemoryIndex>>,
    hnsw_index: Arc<RwLock<HnswIndex>>,
    decay_config: DecayConfiguration,
}

// Update store() method around line 150:
pub fn store(&mut self, episode: Episode) -> Result<String> {
    let id = episode.id.clone();
    
    // Store vector in tiered storage
    let metadata = StorageMetadata {
        timestamp: episode.when,
        confidence: episode.encoding_confidence,
        activation: 1.0, // Initial activation
    };
    self.vector_storage.store(&id, &episode.embedding, metadata)?;
    
    // Rest of existing logic...
}
```

## Testing Strategy

### Unit Tests
```rust
#[test]
fn test_hot_tier_concurrent_access() {
    let hot = HotTier::new(1_000_000);
    
    // Spawn multiple threads
    let handles: Vec<_> = (0..10).map(|i| {
        let hot = hot.clone();
        thread::spawn(move || {
            for j in 0..1000 {
                let id = format!("vec_{}_{}", i, j);
                let vector = [0.1 * i as f32; 768];
                hot.store(&id, &vector, Default::default()).unwrap();
            }
        })
    }).collect();
    
    for h in handles { h.join().unwrap(); }
    assert_eq!(hot.len(), 10_000);
}

#[test]
fn test_migration_policy() {
    let policy = MigrationPolicy::default();
    let stats = AccessStats {
        idle_time: Duration::from_secs(3600),
        access_count: 1,
        activation: 0.01,
    };
    
    assert_eq!(
        policy.evaluate(StorageTier::Hot, &stats),
        MigrationDecision::Demote
    );
}
```

### Performance Benchmarks
```rust
#[bench]
fn bench_tiered_storage_retrieval(b: &mut Bencher) {
    let storage = setup_tiered_storage_with_data();
    
    b.iter(|| {
        let id = black_box("test_vector_42");
        storage.retrieve(id).unwrap()
    });
}
```

## Acceptance Criteria
- [ ] Lock-free hot tier handles 100K concurrent operations/sec
- [ ] Warm tier compression achieves >50% space reduction
- [ ] Cold tier SIMD operations match milestone-1 performance
- [ ] Automatic migration triggers based on access patterns
- [ ] Zero data loss during tier migrations
- [ ] Retrieval maintains <1ms latency for hot tier

## Performance Targets
- Hot tier: <100ns retrieval latency
- Warm tier: <1μs retrieval latency  
- Cold tier: <10μs retrieval latency
- Migration: <100ms for 10K vectors
- Memory usage: <10GB for 1M vectors in hot tier

## Risk Mitigation
- Use proven lock-free libraries (dashmap)
- Implement gradual migration to prevent thundering herd
- Add circuit breakers for migration storms
- Monitor memory pressure and adapt thresholds