# Phase 2 Fixes Required

**Priority:** HIGH (Production Blocker)
**Target:** Make warm/cold tier iteration production-ready
**Estimated Effort:** 3-5 days

---

## High Priority Fixes (BLOCKING)

### Fix 1: Content Persistence in Warm Tier

**Severity:** CRITICAL
**File:** `/Users/jordanwashburn/Workspace/orchard9/engram/engram-core/src/storage/mapped.rs`
**Issue:** Content field populated with placeholder "Memory {id}" instead of actual content
**Impact:** User data corruption - original content permanently lost

**Root Cause:**
`EmbeddingBlock` struct (lines 41-117) doesn't include a content field:
```rust
#[repr(C, align(32))]
pub struct EmbeddingBlock {
    pub embedding: [f32; 768],
    pub confidence: f32,
    pub activation: f32,
    pub creation_time: u64,
    pub last_access: u64,
    pub recall_count: u16,
    pub decay_rate: f32,
    // ❌ NO CONTENT FIELD
}
```

**Current Code (Line 547):**
```rust
memory.content = Some(format!("Memory {memory_id}")); // Content not stored in EmbeddingBlock
```

**Fix Required:**

1. **Update EmbeddingBlock struct (mapped.rs:41-117):**
```rust
#[repr(C, align(32))]
pub struct EmbeddingBlock {
    pub embedding: [f32; 768],
    pub confidence: f32,
    pub activation: f32,
    pub creation_time: u64,
    pub last_access: u64,
    pub recall_count: u16,
    pub decay_rate: f32,
    pub content_length: u32,        // ← ADD: Length of content string
    pub content_offset: u64,        // ← ADD: Offset to content in separate section
}
```

2. **Update MappedWarmStorage to persist content:**

Add content storage section:
```rust
pub struct MappedWarmStorage {
    // ... existing fields ...
    content_data: Vec<u8>,          // ← ADD: Separate storage for variable-length content
    content_offsets: DashMap<u64, (u64, u32)>, // ← ADD: Maps block offset to (content_offset, length)
}
```

3. **Update store logic (mapped.rs:~559):**
```rust
async fn store(&self, memory: Arc<Memory>) -> Result<(), Self::Error> {
    let mut block = EmbeddingBlock::new(&memory);

    // Persist content to separate storage
    if let Some(content) = &memory.content {
        let content_bytes = content.as_bytes();
        let content_offset = self.content_data.len();
        self.content_data.extend_from_slice(content_bytes);

        block.content_offset = content_offset as u64;
        block.content_length = content_bytes.len() as u32;
    }

    let offset = self.find_next_offset();
    self.store_embedding_block(&block, offset)?;
    // ... rest of store logic
}
```

4. **Update get logic (mapped.rs:520-552):**
```rust
pub fn get(&self, memory_id: &str) -> Result<Option<Arc<Memory>>, StorageError> {
    // ... existing block read logic ...

    // Restore content from separate storage
    let content = if block.content_length > 0 {
        let start = block.content_offset as usize;
        let end = start + block.content_length as usize;
        let content_bytes = &self.content_data[start..end];
        Some(String::from_utf8_lossy(content_bytes).to_string())
    } else {
        None
    };

    memory.content = content;  // ← FIX: Restore actual content
    // ... rest of conversion
}
```

**Testing:**
```rust
#[tokio::test]
async fn test_warm_tier_content_persistence() {
    let warm_tier = WarmTier::new(/* ... */);

    let original_content = "This is important user content";
    let memory = create_test_memory("test1", original_content, 0.6);
    warm_tier.store(memory).await.unwrap();

    // Retrieve and verify content preserved
    let memories: Vec<_> = warm_tier.iter_memories().collect();
    assert_eq!(memories[0].1.what, original_content);  // ← Must not be "Memory test1"
}
```

**Alternative Fix (Simpler but Less Efficient):**

Store content inline in EmbeddingBlock using fixed-size buffer:
```rust
pub struct EmbeddingBlock {
    // ... existing fields ...
    pub content: [u8; 256],  // ← Fixed 256-byte content buffer
    pub content_length: u16,
}
```

Pros: Simpler implementation
Cons: Content truncated to 256 bytes, wastes space for short content

---

### Fix 2: Cold Tier Per-Item Lock Acquisition

**Severity:** CRITICAL (Performance)
**File:** `/Users/jordanwashburn/Workspace/orchard9/engram/engram-core/src/storage/cold_tier.rs`
**Lines:** 657-689
**Issue:** RwLock acquired per-item during iteration → 10-100x slower than optimal
**Impact:** 950K production memories → 10+ second iteration (likely timeout)

**Current Code:**
```rust
pub fn iter_memories(&self) -> impl Iterator<Item = (String, Episode)> + '_ {
    self.id_index.iter().filter_map(|entry| {
        let memory_id = entry.key().clone();
        let index = *entry.value();

        // ❌ LOCK ACQUIRED PER-ITEM
        let data = match self.data.read() {
            Ok(guard) => guard,
            Err(_poisoned) => {
                tracing::warn!(memory_id = %memory_id, "Failed to acquire read lock");
                return None;
            }
        };

        match Self::index_to_episode(&data, index) {
            Some(episode) => Some((memory_id, episode)),
            None => None,
        }
    })
}
```

**Performance Impact:**
| Memories | Current (Per-Item Lock) | Optimal (Single Lock) | Regression |
|----------|-------------------------|----------------------|------------|
| 10K      | ~100ms                  | ~10ms                | 10x        |
| 100K     | ~1-2s                   | ~100ms               | 10-20x     |
| 1M       | ~10s                    | ~1s                  | 10x        |

**Fix Option A: Eager Collection (Simple, Production-Ready)**

Change return type to Vec:
```rust
pub fn iter_memories(&self) -> Vec<(String, Episode)> {
    let data = match self.data.read() {
        Ok(guard) => guard,
        Err(poisoned) => {
            tracing::error!("Cold tier RwLock poisoned");
            poisoned.into_inner()
        }
    };

    // Single lock held for entire collection
    self.id_index
        .iter()
        .filter_map(|entry| {
            let memory_id = entry.key().clone();
            let index = *entry.value();
            Self::index_to_episode(&data, index)
                .map(|episode| (memory_id, episode))
        })
        .collect()
}
```

**Update call sites:**
```rust
// store.rs:1857 - iter_cold_memories
pub fn iter_cold_memories(&self) -> Option<Box<dyn Iterator<Item = (String, Episode)> + '_>> {
    #[cfg(feature = "memory_mapped_persistence")]
    {
        self.persistent_backend
            .as_ref()
            .map(|backend| {
                let vec = backend.iter_cold_tier();  // Now returns Vec
                Box::new(vec.into_iter()) as Box<dyn Iterator<Item = (String, Episode)> + '_>
            })
    }
    // ...
}
```

**Pros:**
- Simple 5-line change
- Immediately production-ready
- 10-100x performance improvement

**Cons:**
- Not lazy (all memories loaded eagerly)
- Memory overhead for large cold tiers (100K memories ≈ 100MB)

**Fix Option B: Redesign with Lock-Free Columns (Optimal)**

Replace RwLock with Arc for immutable columns:
```rust
pub struct ColdTier {
    data: Arc<ColumnarData>,  // ← Remove RwLock
    id_index: DashMap<String, usize>,
    // ... other fields
}

pub struct ColumnarData {
    // Use Arc<Vec<T>> for concurrent read access
    embedding_columns: Arc<[Vec<f32>; 768]>,  // ← Arc for zero-cost sharing
    confidences: Arc<Vec<f32>>,
    activations: Arc<Vec<f32>>,
    // ... other columns as Arc<Vec<T>>
    count: AtomicUsize,  // ← Atomic for concurrent reads
}
```

**Iteration becomes lock-free:**
```rust
pub fn iter_memories(&self) -> impl Iterator<Item = (String, Episode)> + '_ {
    let data = Arc::clone(&self.data);  // ← Clone Arc, not data

    self.id_index.iter().filter_map(move |entry| {
        let memory_id = entry.key().clone();
        let index = *entry.value();
        Self::index_to_episode(&data, index)  // ← No lock!
            .map(|episode| (memory_id, episode))
    })
}
```

**Pros:**
- Lock-free reads
- True lazy iteration
- Optimal performance

**Cons:**
- Requires architectural redesign
- Mutations need different approach (copy-on-write)
- Higher implementation complexity

**Recommendation:** Use **Fix Option A** for immediate production deployment. Consider **Fix Option B** for future optimization.

---

### Fix 3: Test Compilation Errors

**Severity:** HIGH
**File:** `/Users/jordanwashburn/Workspace/orchard9/engram/engram-core/tests/tier_iteration_integration_tests.rs`
**Issue:** Tests import non-existent types and use wrong API
**Impact:** Cannot validate implementation

**Error 1: MemoryStoreConfig doesn't exist (Line 13)**
```rust
// ❌ WRONG:
use engram_core::{
    Confidence, Episode, EpisodeBuilder, Memory, MemoryStore, MemoryStoreConfig,  // ← Doesn't exist
    storage::{CognitiveTierArchitecture, StorageMetrics},
};
```

**Fix:**
```rust
// ✓ CORRECT:
use engram_core::{
    Confidence, Episode, EpisodeBuilder, Memory, MemoryStore,  // ← Remove MemoryStoreConfig
    storage::{CognitiveTierArchitecture, StorageMetrics},
};
use tempfile::TempDir;
```

**Error 2: MemoryStore::new() doesn't return Result (Lines 40, 165, 200, 245, 278, 353)**
```rust
// ❌ WRONG:
let config = MemoryStoreConfig::default().with_persistence(temp_dir.path());
let store = MemoryStore::new(config).context("Failed to create memory store")?;
```

**Actual API (store.rs:431):**
```rust
pub fn new(max_memories: usize) -> Self  // ← Returns Self, not Result
```

**Fix:**
```rust
// ✓ CORRECT:
let store = MemoryStore::new(1000);  // ← No config, no .context()
```

**Complete Fixed Test:**
```rust
#[tokio::test]
async fn test_hot_tier_iteration() -> Result<()> {
    let store = MemoryStore::new(1000);  // ← FIXED

    // Store memories in hot tier
    for i in 0..5 {
        let episode = create_test_episode(&format!("hot_{i}"), &format!("Hot memory {i}"));
        store.store(episode).context("Failed to store episode")?;
    }

    // Iterate hot tier
    let memories: Vec<(String, Episode)> = store.iter_hot_memories().collect();

    ensure!(
        memories.len() == 5,
        "Expected 5 memories in hot tier, got {}",
        memories.len()
    );

    Ok(())
}
```

**Tests Affected:**
- `test_hot_tier_iteration` (line 40)
- `test_all_tiers_iteration` (line 165)
- `test_tier_iteration_without_persistence` (line 200)
- `test_tier_iteration_pagination` (line 245)
- `test_tier_counts` (line 278)
- `test_empty_tier_iteration` (line 353)

**Apply same fix pattern to all 6 tests.**

**Persistence Tests (Lines 75, 129, 162, 306):**

These tests need `CognitiveTierArchitecture` but current MemoryStore doesn't expose persistence config.

**Option 1:** Skip these tests until persistence configuration exposed:
```rust
#[tokio::test]
#[cfg(feature = "memory_mapped_persistence")]
#[ignore = "Persistence configuration not yet exposed in MemoryStore API"]
async fn test_warm_tier_iteration() -> Result<()> {
    // ...
}
```

**Option 2:** Create architecture directly and test:
```rust
#[tokio::test]
#[cfg(feature = "memory_mapped_persistence")]
async fn test_warm_tier_iteration() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let metrics = Arc::new(StorageMetrics::new());

    let architecture = CognitiveTierArchitecture::new(
        temp_dir.path(),
        100,   // hot capacity
        1000,  // warm capacity
        10000, // cold capacity
        metrics,
    )?;  // ← This works because architecture exposes persistence

    // Test iteration directly on architecture
    let memories: Vec<_> = architecture.iter_warm_tier().collect();
    // ...
}
```

**Recommendation:** Use Option 2 - test architecture directly until MemoryStore exposes persistence.

---

## Medium Priority Fixes (SHOULD DO SOON)

### Fix 4: Deduplication in All Tier Iteration

**Severity:** MEDIUM
**File:** `/Users/jordanwashburn/Workspace/orchard9/engram/engram-core/src/store.rs`
**Lines:** 1883-1897
**Issue:** Same memory may appear multiple times if present in multiple tiers
**Impact:** Duplicate results in GET /api/v1/memories?tier=all

**Current Code:**
```rust
pub fn iter_all_memories(&self) -> Box<dyn Iterator<Item = (String, Episode)> + '_> {
    let hot = self.iter_hot_memories();

    #[cfg(feature = "memory_mapped_persistence")]
    {
        if let Some(ref backend) = self.persistent_backend {
            let warm = backend.iter_warm_tier();
            let cold = backend.iter_cold_tier();
            return Box::new(hot.chain(warm).chain(cold));  // ← No deduplication
        }
    }

    Box::new(hot)
}
```

**Fix Option A: Add Deduplication**
```rust
pub fn iter_all_memories(&self) -> Box<dyn Iterator<Item = (String, Episode)> + '_> {
    let hot = self.iter_hot_memories();

    #[cfg(feature = "memory_mapped_persistence")]
    {
        if let Some(ref backend) = self.persistent_backend {
            let warm = backend.iter_warm_tier();
            let cold = backend.iter_cold_tier();

            // Track seen IDs to deduplicate
            let mut seen = std::collections::HashSet::new();
            return Box::new(
                hot.chain(warm).chain(cold)
                    .filter(move |(id, _)| seen.insert(id.clone()))
            );
        }
    }

    Box::new(hot)
}
```

**Problem:** HashSet grows unbounded - defeats lazy iteration.

**Fix Option B: Document Behavior**

Add documentation that deduplication is caller's responsibility:
```rust
/// Iterate over all memories across all tiers (hot → warm → cold)
///
/// # Deduplication
///
/// This iterator does NOT deduplicate memories. If a memory exists in multiple
/// tiers (e.g., during migration), it will appear multiple times in results.
/// Callers should deduplicate by ID if needed:
///
/// ```ignore
/// let mut seen = HashSet::new();
/// for (id, episode) in store.iter_all_memories() {
///     if seen.insert(id.clone()) {
///         // Process unique memory
///     }
/// }
/// ```
///
/// In practice, tier migration moves memories (not copies), so duplicates
/// should not occur under normal operation.
pub fn iter_all_memories(&self) -> Box<dyn Iterator<Item = (String, Episode)> + '_> {
    // ... existing code
}
```

**Fix Option C: Ensure Migration Moves, Not Copies**

Verify tier migration logic removes from source tier:
```rust
// In tier migration code
async fn migrate_to_warm(&self, memory_id: &str) {
    let memory = self.hot_tier.get(memory_id)?;
    self.warm_tier.store(memory).await?;
    self.hot_tier.remove(memory_id).await?;  // ← Ensure removal
}
```

**Recommendation:** Use **Fix Option C** (ensure move semantics) + **Fix Option B** (document). Avoid Option A (defeats laziness).

---

### Fix 5: Improve Error Logging for Lock Poisoning

**Severity:** MEDIUM
**File:** `/Users/jordanwashburn/Workspace/orchard9/engram/engram-core/src/storage/cold_tier.rs`
**Lines:** 665-673
**Issue:** Warns per-item if lock poisoned - log spam
**Impact:** Debugging difficulty, log pollution

**Current Code:**
```rust
let data = match self.data.read() {
    Ok(guard) => guard,
    Err(_poisoned) => {
        tracing::warn!(  // ← Per-item warning
            memory_id = %memory_id,
            "Failed to acquire read lock on cold tier data, skipping"
        );
        return None;
    }
};
```

**Fix:**
```rust
use std::sync::atomic::{AtomicBool, Ordering};

// Add to ColdTier struct
struct ColdTier {
    // ... existing fields ...
    lock_poison_logged: AtomicBool,  // ← ADD
}

// Update iterator
let data = match self.data.read() {
    Ok(guard) => guard,
    Err(poisoned) => {
        // Log ERROR once when poisoning first detected
        if !self.lock_poison_logged.swap(true, Ordering::Relaxed) {
            tracing::error!(
                "Cold tier RwLock poisoned - all iteration will fail. \
                 This indicates a panic occurred while holding write lock. \
                 Data integrity may be compromised."
            );
        }
        // Continue with poisoned data or skip silently
        tracing::debug!(memory_id = %memory_id, "Skipping due to poisoned lock");
        return None;
    }
};
```

**Improvement:** Single ERROR log when poisoning detected, DEBUG for skipped items.

---

### Fix 6: Add Content Round-Trip Test

**Severity:** MEDIUM
**File:** Create new test in `/Users/jordanwashburn/Workspace/orchard9/engram/engram-core/tests/tier_content_persistence_test.rs`
**Issue:** No test validates content survives warm/cold tier storage
**Impact:** Content loss bug not caught by tests

**New Test:**
```rust
#[tokio::test]
#[cfg(feature = "memory_mapped_persistence")]
async fn test_warm_tier_content_round_trip() {
    let temp_dir = TempDir::new().unwrap();
    let metrics = Arc::new(StorageMetrics::new());

    let warm_tier = WarmTier::new(
        temp_dir.path().join("warm.dat"),
        1000,
        metrics,
    ).unwrap();

    // Store memory with specific content
    let original_content = "This is important user data that must be preserved";
    let episode = EpisodeBuilder::new()
        .id("test_content".to_string())
        .when(Utc::now())
        .what(original_content.to_string())
        .embedding([0.5f32; 768])
        .confidence(Confidence::HIGH)
        .build();

    let memory = Arc::new(Memory::from_episode(episode, 0.6));
    warm_tier.store(memory).await.unwrap();

    // Retrieve via iteration
    let memories: Vec<_> = warm_tier.iter_memories().collect();
    assert_eq!(memories.len(), 1);

    let (id, retrieved_episode) = &memories[0];
    assert_eq!(id, "test_content");

    // ❌ THIS WILL FAIL with current code:
    assert_eq!(
        retrieved_episode.what,
        original_content,
        "Content must be preserved, not replaced with placeholder"
    );

    // ✓ Should NOT be:
    assert_ne!(
        retrieved_episode.what,
        "Memory test_content",
        "Content must not be a placeholder"
    );
}

#[tokio::test]
#[cfg(feature = "memory_mapped_persistence")]
async fn test_cold_tier_content_round_trip() {
    let cold_tier = ColdTier::new(1000);

    let original_content = "Cold tier content preservation test";
    let episode = EpisodeBuilder::new()
        .id("cold_test".to_string())
        .when(Utc::now())
        .what(original_content.to_string())
        .embedding([0.7f32; 768])
        .confidence(Confidence::MEDIUM)
        .build();

    let memory = Arc::new(Memory::from_episode(episode, 0.3));
    cold_tier.store(memory).await.unwrap();

    // Retrieve via iteration
    let memories: Vec<_> = cold_tier.iter_memories().collect();
    assert_eq!(memories.len(), 1);

    let (id, retrieved_episode) = &memories[0];
    assert_eq!(id, "cold_test");
    assert_eq!(retrieved_episode.what, original_content);  // ✓ This should pass
}
```

**Expected Outcome:**
- Cold tier test: PASS (content persisted correctly)
- Warm tier test: FAIL (content replaced with placeholder)

**After Fix 1 Applied:**
- Both tests: PASS

---

### Fix 7: Add Cold Tier Scalability Test

**Severity:** MEDIUM
**File:** `/Users/jordanwashburn/Workspace/orchard9/engram/engram-core/tests/tier_iteration_performance_test.rs`
**Issue:** No test validates cold tier performance at scale
**Impact:** Performance regression not caught

**New Test:**
```rust
#[tokio::test]
#[cfg(feature = "memory_mapped_persistence")]
async fn test_cold_tier_iteration_scalability() {
    let cold_tier = ColdTier::new(50_000);

    // Store 10K memories (representative of production scale)
    for i in 0..10_000 {
        let episode = EpisodeBuilder::new()
            .id(format!("cold_{i:05}"))
            .when(Utc::now())
            .what(format!("Content {i}"))
            .embedding([0.1f32; 768])
            .confidence(Confidence::LOW)
            .build();

        let memory = Arc::new(Memory::from_episode(episode, 0.1));
        cold_tier.store(memory).await.unwrap();
    }

    // Measure full iteration time
    let start = std::time::Instant::now();
    let count = cold_tier.iter_memories().count();
    let duration = start.elapsed();

    assert_eq!(count, 10_000);

    // Performance target: <200ms for 10K memories
    // Current (per-item lock): ~100ms
    // After Fix 2: ~10ms
    assert!(
        duration.as_millis() < 200,
        "Cold tier iteration too slow: {:?} for 10K memories (expected <200ms)",
        duration
    );

    // Log performance for monitoring
    println!("Cold tier 10K iteration: {:?}", duration);
}
```

**Expected Outcome:**
- Before Fix 2: ~100ms (marginal pass)
- After Fix 2 (eager Vec): ~10ms (excellent)
- If per-item lock kept: May fail at 100K+ scale

---

## Low Priority Improvements (TECH DEBT)

### Improvement 1: Optimize String Allocations

**File:** `/Users/jordanwashburn/Workspace/orchard9/engram/engram-core/src/storage/warm_tier.rs`
**Line:** 166
**Issue:** Clones memory_id per iteration

**Current:**
```rust
let memory_id = entry.key().clone();  // String clone
```

**Optimization:**
```rust
let memory_id: &str = entry.key();  // Borrow
// ... but requires lifetime changes throughout
```

**Impact:** Minimal (string clone is ~10-50ns)
**Effort:** Medium (requires lifetime refactoring)
**Recommendation:** Defer until profiling shows bottleneck

---

### Improvement 2: Extract Common Conversion Logic

**Files:**
- `/Users/jordanwashburn/Workspace/orchard9/engram/engram-core/src/storage/warm_tier.rs:172-180`
- `/Users/jordanwashburn/Workspace/orchard9/engram/engram-core/src/storage/cold_tier.rs:371-378`

**Issue:** Duplicate Episode construction code

**Current:**
```rust
// Warm tier
let episode = Episode::new(
    memory.id.clone(),
    memory.created_at,
    memory.content.clone().unwrap_or_else(|| format!("Memory {id}", id = memory.id)),
    memory.embedding,
    memory.confidence,
);

// Cold tier
EpisodeBuilder::new()
    .id(data.memory_ids[index].clone())
    .when(datetime)
    .what(data.contents[index].clone())
    .embedding(embedding)
    .confidence(Confidence::exact(data.confidences[index]))
    .build()
```

**Refactor:**
```rust
// In storage/mod.rs or conversions.rs
fn memory_to_episode(
    id: String,
    content: String,
    embedding: [f32; 768],
    confidence: f32,
    created_at: DateTime<Utc>,
) -> Episode {
    EpisodeBuilder::new()
        .id(id)
        .when(created_at)
        .what(content)
        .embedding(embedding)
        .confidence(Confidence::exact(confidence))
        .build()
}
```

**Impact:** Minor code quality improvement
**Effort:** Low
**Recommendation:** Do when refactoring nearby code

---

### Improvement 3: Document Tier Iterator Performance

**File:** `/Users/jordanwashburn/Workspace/orchard9/engram/engram-core/src/store.rs`
**Lines:** 1828-1897
**Issue:** Performance characteristics not documented

**Add Documentation:**
```rust
/// Iterate over warm tier memories (persistent storage)
///
/// # Performance
///
/// - Per-memory: ~10-50μs (memory-mapped I/O with page faults)
/// - Typical: 10-50ms for thousands of memories
/// - Scalability: O(n) with low constant factor
/// - Memory: Lazy iterator, minimal heap allocation
///
/// # Production Considerations
///
/// - Suitable for datasets up to ~100K memories
/// - For larger datasets, use pagination with offset/limit
/// - First access triggers page faults (slower), subsequent accesses cached
/// - Consider madvise(MADV_SEQUENTIAL) for large scans
pub fn iter_warm_memories(&self) -> Option<Box<dyn Iterator<Item = (String, Episode)> + '_>> {
    // ...
}

/// Iterate over cold tier memories (archived storage)
///
/// # Performance
///
/// - Per-memory: ~1-10μs (columnar read from RAM)
/// - Typical: 10-100ms for 10K memories, 100ms-1s for 100K memories
/// - Scalability: O(n) - see note on lock overhead below
/// - Memory: Lazy iterator, minimal heap allocation
///
/// # Lock Overhead (BEFORE Fix 2)
///
/// Current implementation acquires read lock per-item during iteration.
/// This adds ~50-200ns overhead per memory.
/// For large cold tiers (>10K memories), consider:
/// - Using tier-specific pagination
/// - Upgrading to eager collection (Vec return)
/// - Redesigning with lock-free columns
///
/// # Lock Overhead (AFTER Fix 2)
///
/// Returns Vec with single lock acquisition. Fast for all scales.
/// Memory overhead: ~1KB per memory (100K memories ≈ 100MB).
pub fn iter_cold_memories(&self) -> Option<Box<dyn Iterator<Item = (String, Episode)> + '_>> {
    // ...
}
```

---

## Testing Checklist

After applying fixes, verify:

- [ ] Warm tier content persistence test passes
- [ ] Cold tier content persistence test passes
- [ ] All tier iteration tests compile
- [ ] Cold tier scalability test passes (<200ms for 10K memories)
- [ ] Integration test: Store → Migrate → Retrieve → Verify content
- [ ] API test: GET /api/v1/memories?tier=warm returns actual content
- [ ] API test: GET /api/v1/memories?tier=cold returns actual content
- [ ] Load test: 100K cold tier memories iterate in <1s (after Fix 2)
- [ ] Concurrent test: Iteration + store/remove operations don't deadlock
- [ ] Error test: Lock poisoning logs ERROR once, not per-item

---

## Deployment Checklist

Before merging Phase 2:

**Critical:**
- [ ] Fix 1 applied and tested (content persistence)
- [ ] Fix 2 applied and tested (cold tier lock)
- [ ] Fix 3 applied (tests compile and pass)
- [ ] Content round-trip tests added and passing
- [ ] API tested with actual content (not placeholders)

**Important:**
- [ ] Fix 4 or documented (deduplication)
- [ ] Fix 5 applied (error logging)
- [ ] Scalability test added and passing
- [ ] Performance documented

**Nice to Have:**
- [ ] Improvements 1-3 (optimizations and refactoring)

**Sign-Off:**
- [ ] Code review by second engineer
- [ ] QA tested with production-like dataset (100K+ memories)
- [ ] Performance validated (<5% regression)
- [ ] Documentation updated

---

## Implementation Priority

**Day 1:**
1. Fix 1 (content persistence) - 4-6 hours
2. Fix 3 (test compilation) - 1-2 hours
3. Fix 6 (content tests) - 1 hour

**Day 2:**
4. Fix 2 (cold tier lock) - 2-4 hours
5. Fix 7 (scalability test) - 1 hour
6. Validation and debugging - 2-4 hours

**Day 3:**
7. Fix 4 (deduplication) - 2-3 hours
8. Fix 5 (error logging) - 1 hour
9. Documentation updates - 2 hours
10. Final testing and review - 2-3 hours

**Total: 18-26 hours (2.5-3.5 days)**

---

## Conclusion

Phase 2 requires **3 critical fixes** before production deployment:
1. Content persistence (data corruption)
2. Cold tier locking (performance)
3. Test compilation (validation)

All fixes are well-understood with clear implementation paths. Estimated effort: 3-5 days for production-ready state.

**Recommendation:** Block merge until critical fixes complete and tested.
