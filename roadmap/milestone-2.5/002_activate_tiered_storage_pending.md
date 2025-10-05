# Task 002: Activate Tiered Storage

## Status: Pending
## Priority: P0 - Critical (Blocks Performance)
## Estimated Effort: 2 days
## Dependencies: Task 001 (WAL activation)

## Objective

Wire warm and cold storage tiers into MemoryStore query path so recall operations search across all tiers with automatic migration based on access patterns.

## Current State

**Infrastructure EXISTS but INACTIVE:**
- ✅ `engram-core/src/storage/hot_tier.rs:1-150` - DashMap-based hot tier
- ✅ `engram-core/src/storage/warm_tier.rs:1-200` - Compressed warm tier
- ✅ `engram-core/src/storage/cold_tier.rs:1-180` - Columnar cold tier
- ✅ `engram-core/src/storage/tiers.rs:1-250` - TierCoordinator scaffolding
- ❌ MemoryStore only queries `hot_memories` DashMap, ignores tiers

**Relevant Code:**
```rust
// engram-core/src/store.rs:862-877 (CURRENT)
fn recall_from_memory(&self, cue: &Cue) -> Vec<(Episode, Confidence)> {
    // Only searches hot tier!
    let memories: Vec<_> = self.hot_memories.iter()
        .map(|entry| entry.value().clone())
        .collect();
    // ... filters memories ...
}
```

## Implementation Steps

### Step 1: Add Tier Coordinator to MemoryStore (1 hour)

**File**: `engram-core/src/store.rs`

**Line 105** - Add tier coordinator field:
```rust
/// Three-tier storage coordinator for hot/warm/cold data
#[cfg(feature = "memory_mapped_persistence")]
tier_coordinator: Option<Arc<CognitiveTierArchitecture>>,
```

**Line 246** - Initialize in constructor:
```rust
#[cfg(feature = "memory_mapped_persistence")]
tier_coordinator: None,
```

**Line 315-340** - Add tier initialization method:
```rust
/// Initialize three-tier storage architecture
#[cfg(feature = "memory_mapped_persistence")]
#[must_use]
pub fn with_tiered_storage<P: AsRef<Path>>(mut self, data_dir: P) -> Result<Self, StorageError> {
    use crate::storage::{CognitiveTierArchitecture, HotTier, WarmTier, ColdTier};

    let data_path = data_dir.as_ref();
    std::fs::create_dir_all(data_path)?;

    // Create tier instances
    let hot = Arc::new(HotTier::new(10_000)); // 10k memories in hot tier
    let warm = Arc::new(WarmTier::new(data_path.join("warm"))?);
    let cold = Arc::new(ColdTier::new(data_path.join("cold"))?);

    // Create coordinator
    let coordinator = CognitiveTierArchitecture::new(
        hot,
        warm,
        cold,
        Arc::clone(&self.storage_metrics),
    );

    self.tier_coordinator = Some(Arc::new(coordinator));
    Ok(self)
}
```

### Step 2: Replace Hot-Only Recall with Tier Search (2 hours)

**File**: `engram-core/src/store.rs`

**Line 862-877** - Replace `recall_from_memory` implementation:
```rust
fn recall_from_memory(&self, cue: &Cue) -> Vec<(Episode, Confidence)> {
    #[cfg(feature = "memory_mapped_persistence")]
    if let Some(ref coordinator) = self.tier_coordinator {
        // Search all tiers with budget constraints
        return self.tier_aware_recall(coordinator, cue);
    }

    // Fallback: hot tier only (current behavior)
    let memories: Vec<_> = self.hot_memories.iter()
        .map(|entry| entry.value().clone())
        .collect();

    self.filter_and_rank_memories(memories, cue)
}

#[cfg(feature = "memory_mapped_persistence")]
fn tier_aware_recall(
    &self,
    coordinator: &CognitiveTierArchitecture,
    cue: &Cue,
) -> Vec<(Episode, Confidence)> {
    use crate::storage::TierSearchStrategy;

    let strategy = match cue.confidence.raw() {
        c if c > 0.7 => TierSearchStrategy::HotFirst,     // High confidence: likely recent
        c if c > 0.4 => TierSearchStrategy::HotAndWarm,   // Medium: search 2 tiers
        _ => TierSearchStrategy::AllTiers,                 // Low: exhaustive search
    };

    match coordinator.search(cue, strategy) {
        Ok(results) => results,
        Err(e) => {
            tracing::warn!("Tier search failed: {:?}, falling back to hot tier", e);
            self.recall_from_memory(cue) // Recursive fallback
        }
    }
}
```

### Step 3: Implement Tier Search in Coordinator (3 hours)

**File**: `engram-core/src/storage/tiers.rs`

**Line 80-150** - Implement search method:
```rust
impl CognitiveTierArchitecture {
    /// Search tiers based on strategy
    pub fn search(
        &self,
        cue: &Cue,
        strategy: TierSearchStrategy,
    ) -> StorageResult<Vec<(Episode, Confidence)>> {
        use TierSearchStrategy::*;

        let start = Instant::now();
        let mut all_results = Vec::new();

        match strategy {
            HotFirst => {
                // Try hot tier first
                if let Ok(results) = self.hot.recall(cue).await {
                    if !results.is_empty() {
                        all_results = results;
                    }
                }

                // If not enough results, check warm tier
                if all_results.len() < cue.max_results.unwrap_or(10) {
                    if let Ok(warm_results) = self.warm.recall(cue).await {
                        all_results.extend(warm_results);
                    }
                }
            }

            HotAndWarm => {
                // Parallel search of hot and warm
                let (hot_res, warm_res) = tokio::join!(
                    self.hot.recall(cue),
                    self.warm.recall(cue),
                );

                if let Ok(results) = hot_res {
                    all_results.extend(results);
                }
                if let Ok(results) = warm_res {
                    all_results.extend(results);
                }
            }

            AllTiers => {
                // Parallel search all three tiers
                let (hot_res, warm_res, cold_res) = tokio::join!(
                    self.hot.recall(cue),
                    self.warm.recall(cue),
                    self.cold.recall(cue),
                );

                if let Ok(results) = hot_res {
                    all_results.extend(results);
                }
                if let Ok(results) = warm_res {
                    all_results.extend(results);
                }
                if let Ok(results) = cold_res {
                    all_results.extend(results);
                }
            }
        }

        // Sort by confidence (highest first)
        all_results.sort_by(|a, b| b.1.raw().partial_cmp(&a.1.raw()).unwrap_or(std::cmp::Ordering::Equal));

        // Take top N results
        let max_results = cue.max_results.unwrap_or(10);
        all_results.truncate(max_results);

        // Record metrics
        self.metrics.record_tier_search(start.elapsed(), all_results.len());

        Ok(all_results)
    }
}
```

**Line 20** - Add strategy enum:
```rust
/// Strategy for searching across tiers
#[derive(Debug, Clone, Copy)]
pub enum TierSearchStrategy {
    /// Search hot tier first, then warm if needed
    HotFirst,
    /// Search hot and warm tiers in parallel
    HotAndWarm,
    /// Search all three tiers in parallel
    AllTiers,
}
```

### Step 4: Implement Automatic Tier Migration (4 hours)

**File**: `engram-core/src/storage/tiers.rs`

**Line 200-280** - Add migration logic:
```rust
impl CognitiveTierArchitecture {
    /// Migrate memories between tiers based on access patterns
    pub async fn migrate_based_on_access(&self) -> StorageResult<MigrationStats> {
        let mut stats = MigrationStats::default();

        // Hot → Warm: Low activation, old access
        let hot_candidates = self.identify_hot_to_warm_candidates().await?;
        for memory in hot_candidates {
            if let Err(e) = self.migrate_memory(&memory, TierLevel::Hot, TierLevel::Warm).await {
                tracing::warn!("Migration failed for {}: {:?}", memory.id, e);
            } else {
                stats.hot_to_warm += 1;
            }
        }

        // Warm → Cold: Very old, infrequent access
        let warm_candidates = self.identify_warm_to_cold_candidates().await?;
        for memory in warm_candidates {
            if let Err(e) = self.migrate_memory(&memory, TierLevel::Warm, TierLevel::Cold).await {
                tracing::warn!("Migration failed for {}: {:?}", memory.id, e);
            } else {
                stats.warm_to_cold += 1;
            }
        }

        // Warm → Hot: Recently accessed
        let promote_candidates = self.identify_promotion_candidates().await?;
        for memory in promote_candidates {
            if let Err(e) = self.migrate_memory(&memory, TierLevel::Warm, TierLevel::Hot).await {
                tracing::warn!("Promotion failed for {}: {:?}", memory.id, e);
            } else {
                stats.warm_to_hot += 1;
            }
        }

        Ok(stats)
    }

    async fn identify_hot_to_warm_candidates(&self) -> StorageResult<Vec<Arc<Memory>>> {
        let threshold = 0.3; // Activation threshold
        let age_threshold = Duration::from_secs(3600); // 1 hour

        let now = Utc::now();
        let mut candidates = Vec::new();

        // Scan hot tier for cold memories
        for entry in self.hot.iter() {
            let memory = entry.value();
            let age = now.signed_duration_since(memory.last_access)
                .to_std()
                .unwrap_or_default();

            if memory.activation < threshold && age > age_threshold {
                candidates.push(Arc::clone(memory));
            }
        }

        Ok(candidates)
    }

    async fn migrate_memory(
        &self,
        memory: &Memory,
        from: TierLevel,
        to: TierLevel,
    ) -> StorageResult<()> {
        let id = &memory.id;

        // Write to destination tier
        match to {
            TierLevel::Hot => self.hot.store(Arc::clone(memory)).await?,
            TierLevel::Warm => self.warm.store(Arc::clone(memory)).await?,
            TierLevel::Cold => self.cold.store(Arc::clone(memory)).await?,
        }

        // Remove from source tier
        match from {
            TierLevel::Hot => self.hot.remove(id).await?,
            TierLevel::Warm => self.warm.remove(id).await?,
            TierLevel::Cold => self.cold.remove(id).await?,
        }

        tracing::debug!("Migrated {} from {:?} to {:?}", id, from, to);
        Ok(())
    }
}

#[derive(Debug, Clone, Copy)]
enum TierLevel {
    Hot,
    Warm,
    Cold,
}

#[derive(Default, Debug)]
pub struct MigrationStats {
    pub hot_to_warm: usize,
    pub warm_to_cold: usize,
    pub warm_to_hot: usize,
}
```

### Step 5: Add Background Migration Task (1 hour)

**File**: `engram-core/src/store.rs`

**Line 500-530** - Start migration background task:
```rust
/// Start background tier migration task
#[cfg(feature = "memory_mapped_persistence")]
pub fn start_tier_migration(&self) -> Option<std::thread::JoinHandle<()>> {
    let coordinator = self.tier_coordinator.as_ref()?.clone();

    let handle = std::thread::Builder::new()
        .name("tier-migration".to_string())
        .spawn(move || {
            let runtime = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("Failed to create migration runtime");

            runtime.block_on(async {
                let mut interval = tokio::time::interval(Duration::from_secs(300)); // Every 5 min

                loop {
                    interval.tick().await;

                    match coordinator.migrate_based_on_access().await {
                        Ok(stats) => {
                            tracing::info!(
                                "Tier migration: hot→warm={}, warm→cold={}, warm→hot={}",
                                stats.hot_to_warm, stats.warm_to_cold, stats.warm_to_hot
                            );
                        }
                        Err(e) => {
                            tracing::error!("Tier migration failed: {:?}", e);
                        }
                    }
                }
            });
        })
        .ok();

    handle
}
```

### Step 6: Wire into Main Startup (30 min)

**File**: `engram-cli/src/main.rs`

**Line 158-175** - Add tiered storage initialization:
```rust
// Enable WAL persistence
#[cfg(feature = "memory_mapped_persistence")]
{
    use engram_core::storage::FsyncMode;

    memory_store = memory_store
        .with_wal("./data/wal", FsyncMode::PerBatch)
        .expect("Failed to initialize WAL");

    // Add tiered storage
    memory_store = memory_store
        .with_tiered_storage("./data")
        .expect("Failed to initialize tiered storage");

    // Recover from WAL on startup
    match memory_store.recover_from_wal() {
        Ok(count) => tracing::info!("Recovered {} memories from WAL", count),
        Err(e) => tracing::warn!("WAL recovery failed: {:?}", e),
    }
}

let memory_store = Arc::new(memory_store);

// Start background tier migration
#[cfg(feature = "memory_mapped_persistence")]
if let Some(handle) = memory_store.start_tier_migration() {
    tracing::info!("Started tier migration background task");
    // Store handle for graceful shutdown if needed
}
```

## Testing Strategy

### Unit Tests

**File**: `engram-core/src/storage/tiers.rs`

Add at line 400:
```rust
#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[tokio::test]
    async fn test_tier_search_strategy() {
        let dir = tempdir().unwrap();
        let metrics = Arc::new(StorageMetrics::new());

        let hot = Arc::new(HotTier::new(100));
        let warm = Arc::new(WarmTier::new(dir.path().join("warm")).unwrap());
        let cold = Arc::new(ColdTier::new(dir.path().join("cold")).unwrap());

        let coordinator = CognitiveTierArchitecture::new(hot, warm, cold, metrics);

        // Store memory in warm tier
        let memory = create_test_memory("test_1", 0.5);
        coordinator.warm.store(Arc::new(memory)).await.unwrap();

        // Search with AllTiers strategy
        let cue = Cue::semantic("test_1".to_string(), "test".to_string(), Confidence::LOW);
        let results = coordinator.search(&cue, TierSearchStrategy::AllTiers).await.unwrap();

        assert_eq!(results.len(), 1);
    }

    #[tokio::test]
    async fn test_tier_migration() {
        // Test migration logic
        let dir = tempdir().unwrap();
        let coordinator = create_test_coordinator(&dir);

        // Store old memory in hot tier
        let old_memory = create_old_memory("old_1", 0.2);
        coordinator.hot.store(Arc::new(old_memory)).await.unwrap();

        // Run migration
        let stats = coordinator.migrate_based_on_access().await.unwrap();

        assert_eq!(stats.hot_to_warm, 1);
    }
}
```

### Integration Test

**File**: `engram-core/tests/tier_integration_test.rs` (create new)

```rust
use engram_core::{MemoryStore, Episode, Confidence, Cue};
use chrono::Utc;
use tempfile::tempdir;
use std::time::Duration;

#[tokio::test]
async fn test_cross_tier_recall() {
    let data_dir = tempdir().unwrap();

    let mut store = MemoryStore::new(100)
        .with_tiered_storage(data_dir.path())
        .unwrap();

    // Store multiple episodes
    for i in 0..20 {
        let episode = Episode::new(
            format!("ep_{}", i),
            Utc::now(),
            format!("Content {}", i),
            [i as f32 / 20.0; 768],
            Confidence::MEDIUM,
        );
        store.store(episode);
    }

    // Trigger migration (simulate time passing)
    std::thread::sleep(Duration::from_secs(2));

    // Search should find memories across all tiers
    let cue = Cue::semantic("search".to_string(), "Content".to_string(), Confidence::LOW);
    let results = store.recall(&cue);

    assert!(results.len() > 0, "Should recall from multiple tiers");
}
```

## Acceptance Criteria

- [ ] `MemoryStore::with_tiered_storage()` initializes hot/warm/cold tiers
- [ ] `recall()` searches all tiers based on cue confidence
- [ ] Automatic migration runs every 5 minutes
- [ ] Hot→Warm migration for activation <0.3, age >1 hour
- [ ] Warm→Hot promotion for recently accessed memories
- [ ] Warm→Cold migration for very old, infrequent access
- [ ] Cross-tier recall test passes
- [ ] Metrics track tier search latency and migration counts

## Performance Targets

- Hot tier: <100μs retrieval latency
- Warm tier: <1ms retrieval with compression
- Cold tier: <10ms batch operations
- Migration overhead: <5% CPU at 300s interval
- Cross-tier search P95: <5ms (parallel tier search)

## Files to Modify

1. `engram-core/src/store.rs` - Add tier_coordinator, tier-aware recall, migration task
2. `engram-core/src/storage/tiers.rs` - Implement search, migration logic
3. `engram-cli/src/main.rs` - Initialize tiers and start migration task
4. `engram-core/tests/tier_integration_test.rs` - Create integration test

## Files to Read First

- `engram-core/src/storage/tiers.rs:1-100` - Existing TierCoordinator structure
- `engram-core/src/storage/hot_tier.rs` - Hot tier implementation
- `engram-core/src/storage/warm_tier.rs` - Warm tier with compression
- `engram-core/src/storage/cold_tier.rs` - Cold columnar storage
- `engram-core/src/store.rs:862-900` - Current recall implementation
