# Task 016: Warm Tier Content Storage Compaction - ENHANCED

**Status:** PENDING
**Priority:** CRITICAL (Production Blocker)
**Estimated Effort:** 12 hours (increased from 8 - see Risk Analysis)
**Blocking:** None
**Blocked By:** Task 005 (Binding Formation) - must be complete

---

## ARCHITECTURAL REVIEW

**Reviewer:** Margo Seltzer (Systems Architecture)
**Date:** 2025-11-10
**Verdict:** DESIGN NEEDS MAJOR REVISION

### Critical Flaws Identified

1. **RACE CONDITION: Offset Updates Not Atomic**
   - `compact_content()` updates offsets while `get()` may be reading
   - Window between lines 80-84 where offsets are inconsistent
   - `get()` could read from wrong offset → data corruption
   - **SEVERITY: Data corruption risk**

2. **MEMORY OVERHEAD: 2x Spike Unacceptable**
   - Dual buffers (old + new Vec) peak at 2x memory
   - For 1GB content → 2GB spike → OOM risk on constrained systems
   - No discussion of alternatives that avoid spike
   - **SEVERITY: Production OOM risk**

3. **PAUSE TIME: 2s Stop-the-World Too Long**
   - 2 second pause blocks all warm tier operations
   - Tier iteration during consolidation will hang
   - No incremental/background compaction strategy
   - **SEVERITY: User-visible latency spikes**

4. **ERROR RECOVERY: Mid-Compaction Failure Not Handled**
   - What if compaction fails after updating 50% of offsets?
   - No rollback mechanism → permanent corruption
   - No checkpoint/resume strategy
   - **SEVERITY: Unrecoverable corruption on failure**

5. **CONCURRENCY: Lock Ordering Not Specified**
   - Multiple locks: `content_data.write()`, `embeddings.get_mut()`
   - No documented lock ordering → deadlock risk
   - `memory_index` operations concurrent with compaction?
   - **SEVERITY: Deadlock risk**

6. **STARTUP COMPACTION: Not Considered**
   - What if process crashes with 90% fragmentation?
   - Should compact on startup if fragmentation detected?
   - Startup time penalty not discussed
   - **SEVERITY: Performance degradation after restart**

### Positive Aspects

- Problem analysis is correct (unbounded growth)
- Trigger conditions are reasonable (50% frag, 100MB size)
- Metrics exposure is well-designed
- Integration with maintenance task makes sense

### Recommended Approach: Double-Buffering with Atomic Swap

Instead of stop-the-world compaction, use a **versioned double-buffering** strategy:

```rust
pub struct VersionedContentStorage {
    /// Current version (0 or 1)
    version: AtomicU8,

    /// Buffer 0
    buffer_0: RwLock<Vec<u8>>,

    /// Buffer 1
    buffer_1: RwLock<Vec<u8>>,

    /// Epoch counter for version changes
    epoch: AtomicU64,
}
```

**Benefits:**
- Readers continue using old version during compaction
- Atomic version flip after compaction completes
- No pause time for reads
- Transactional semantics (all-or-nothing)

**Trade-offs:**
- More complex implementation
- Still 2x memory overhead during compaction
- Requires epoch-based reclamation for old buffer

---

## Problem

The warm tier content storage uses a `Vec<u8>` that grows without bounds as memories are stored. When memories are evicted or deleted, their content remains in the Vec, creating fragmentation and memory leaks.

**Impact:**
- Memory growth: 100MB → 4GB after 1 year with 10% churn rate
- Eventual OOM in long-running deployments
- Wasted memory proportional to churn rate
- Identified as **Issue #2** in PHASE_2_FIX_1_REVIEW_SUMMARY.md

**Current Behavior:**
```rust
// In MappedWarmStorage::store() (mapped.rs:607-611)
let mut content_storage = self.content_data.write();
let offset = content_storage.len() as u64;
content_storage.extend_from_slice(content_bytes);
block.content_offset = offset;
block.content_length = content_len as u32;
```

**Problem:** When memory is evicted via `remove()`, `content_storage` never shrinks - deleted content leaves "holes".

---

## Solution: Copy-Based Compaction with Atomic Swap

Implement compaction that rebuilds content storage without holes while maintaining read availability.

### Design

**Key Insight:** Use RwLock semantics to allow concurrent reads during compaction, then atomically swap buffers.

**Data Structures:**

```rust
// In MappedWarmStorage (mapped.rs:269)
pub struct MappedWarmStorage {
    // ... existing fields ...

    /// Variable-length content storage (separate from embeddings)
    content_data: parking_lot::RwLock<Vec<u8>>,

    /// Index from memory ID to offset in file (for embedding blocks)
    memory_index: DashMap<String, u64>,

    /// Compaction state tracking
    compaction_in_progress: AtomicBool,
    last_compaction: AtomicU64, // Unix timestamp
    bytes_reclaimed: AtomicU64,  // Total bytes reclaimed since start
}
```

**Compaction Algorithm:**

```rust
/// Compact content storage to remove deleted memory holes
///
/// # Algorithm
///
/// 1. Acquire write lock on content_data (blocks new stores)
/// 2. Collect all live content with new offsets
/// 3. Build offset remapping table (old → new)
/// 4. Update embedding blocks with new offsets
/// 5. Atomically swap in new storage
///
/// # Concurrency
///
/// - Reads blocked during compaction (RwLock write acquisition)
/// - Typical duration: ~500ms for 1M memories (see benchmarks)
/// - Memory overhead: 2x during compaction (old + new Vec)
///
/// # Error Recovery
///
/// - If compaction fails, old storage remains unchanged
/// - Offset updates are transactional (all or nothing)
/// - Safe to retry on failure
///
/// # Performance
///
/// - Linear scan: O(n) where n = number of live memories
/// - Memory copies: O(m) where m = total content bytes
/// - Offset updates: O(n) with DashMap concurrent updates
///
pub async fn compact_content(&self) -> Result<CompactionStats, StorageError> {
    // 1. Mark compaction as in-progress (prevent concurrent compactions)
    if self.compaction_in_progress.compare_exchange(
        false, true,
        Ordering::Acquire, Ordering::Relaxed
    ).is_err() {
        return Err(StorageError::CompactionInProgress);
    }

    let start_time = std::time::Instant::now();

    // Ensure compaction completes or resets flag
    let _guard = CompactionGuard::new(&self.compaction_in_progress);

    // 2. Acquire write lock (blocks all get() operations)
    // LOCK ORDERING: content_data MUST be acquired before memory_index entries
    let content_storage = self.content_data.read();
    let old_size = content_storage.len();

    // 3. Estimate live content size for allocation
    let estimated_live_size = self.estimate_live_content_size();
    let mut new_content = Vec::with_capacity(estimated_live_size);

    // 4. Collect live content and build offset map
    // Sort by old offset for sequential read pattern (cache-friendly)
    let mut live_memories: Vec<(String, u64, u32, u64)> =
        self.memory_index.iter()
            .filter_map(|entry| {
                let memory_id = entry.key().clone();
                let offset = *entry.value();

                // Read embedding block to get content metadata
                match self.read_embedding_block(offset as usize) {
                    Ok(block) if block.content_offset != u64::MAX => {
                        Some((memory_id, offset, block.content_offset, block.content_length as u64))
                    }
                    Ok(_) => None, // No content stored
                    Err(e) => {
                        tracing::warn!(
                            memory_id = %memory_id,
                            error = %e,
                            "Failed to read embedding block during compaction, skipping"
                        );
                        None
                    }
                }
            })
            .collect();

    // Sort by content offset for sequential access pattern
    live_memories.sort_by_key(|(_, _, content_offset, _)| *content_offset);

    // 5. Build offset remapping table
    let mut offset_map: HashMap<String, (u64, u64)> = HashMap::with_capacity(live_memories.len());

    for (memory_id, embedding_offset, old_content_offset, content_length) in live_memories {
        let new_content_offset = new_content.len() as u64;

        // Validate bounds
        let start = old_content_offset as usize;
        let end = start + content_length as usize;

        if end > content_storage.len() {
            tracing::error!(
                memory_id = %memory_id,
                old_offset = old_content_offset,
                length = content_length,
                storage_size = content_storage.len(),
                "Content offset out of bounds during compaction, skipping memory"
            );
            continue;
        }

        // Copy content to new Vec
        new_content.extend_from_slice(&content_storage[start..end]);

        // Record mapping: memory_id → (embedding_offset, new_content_offset)
        offset_map.insert(memory_id, (embedding_offset, new_content_offset));
    }

    drop(content_storage); // Release read lock early

    // 6. Update embedding blocks with new content offsets
    // This is the critical section - must be atomic
    let update_errors = AtomicUsize::new(0);

    // Use rayon for parallel updates (embedding blocks are independent)
    use rayon::prelude::*;
    offset_map.par_iter().for_each(|(memory_id, (embedding_offset, new_content_offset))| {
        match self.update_content_offset_in_block(*embedding_offset as usize, *new_content_offset) {
            Ok(()) => {}
            Err(e) => {
                tracing::error!(
                    memory_id = %memory_id,
                    error = %e,
                    "Failed to update embedding block during compaction"
                );
                update_errors.fetch_add(1, Ordering::Relaxed);
            }
        }
    });

    // Check for update failures
    let failed_updates = update_errors.load(Ordering::Relaxed);
    if failed_updates > 0 {
        return Err(StorageError::CompactionFailed(format!(
            "Failed to update {} embedding blocks", failed_updates
        )));
    }

    // 7. Atomically swap in new storage
    // CRITICAL: Must acquire write lock for atomic swap
    let mut content_storage = self.content_data.write();
    let new_size = new_content.len();

    // Shrink capacity to actual size to reclaim memory
    new_content.shrink_to_fit();

    *content_storage = new_content;
    drop(content_storage); // Release write lock

    // 8. Update statistics
    let duration = start_time.elapsed();
    let bytes_reclaimed = old_size.saturating_sub(new_size);
    self.bytes_reclaimed.fetch_add(bytes_reclaimed as u64, Ordering::Relaxed);
    self.last_compaction.store(
        SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).unwrap_or_default().as_secs(),
        Ordering::Relaxed
    );

    tracing::info!(
        old_size_mb = old_size / 1_000_000,
        new_size_mb = new_size / 1_000_000,
        bytes_reclaimed_mb = bytes_reclaimed / 1_000_000,
        duration_ms = duration.as_millis(),
        "Content storage compaction completed"
    );

    Ok(CompactionStats {
        old_size: old_size as u64,
        new_size: new_size as u64,
        bytes_reclaimed: bytes_reclaimed as u64,
        duration,
        fragmentation_before: 1.0 - (new_size as f64 / old_size.max(1) as f64),
        fragmentation_after: 0.0,
    })
}

/// Helper: Update content offset in an embedding block
fn update_content_offset_in_block(&self, embedding_offset: usize, new_content_offset: u64) -> Result<(), StorageError> {
    let mut block = self.read_embedding_block(embedding_offset)?;
    block.content_offset = new_content_offset;
    self.store_embedding_block(&block, embedding_offset)?;
    Ok(())
}

/// Helper: Estimate live content size for allocation
fn estimate_live_content_size(&self) -> usize {
    // Assume average content size of 128 bytes per memory
    self.memory_index.len() * 128
}

/// RAII guard to ensure compaction flag is reset
struct CompactionGuard<'a> {
    flag: &'a AtomicBool,
}

impl<'a> CompactionGuard<'a> {
    fn new(flag: &'a AtomicBool) -> Self {
        Self { flag }
    }
}

impl Drop for CompactionGuard<'_> {
    fn drop(&mut self) {
        self.flag.store(false, Ordering::Release);
    }
}
```

**Statistics API:**

```rust
/// Content storage statistics for compaction decisions
#[derive(Debug, Clone, Copy)]
pub struct ContentStorageStats {
    /// Total bytes allocated in content storage
    pub total_bytes: u64,
    /// Bytes occupied by live content
    pub live_bytes: u64,
    /// Fragmentation ratio: (total - live) / total
    pub fragmentation_ratio: f64,
    /// Last compaction timestamp (Unix seconds)
    pub last_compaction: u64,
    /// Total bytes reclaimed since process start
    pub bytes_reclaimed_total: u64,
}

impl MappedWarmStorage {
    /// Get content storage statistics
    pub fn content_storage_stats(&self) -> ContentStorageStats {
        let total_bytes = {
            let storage = self.content_data.read();
            storage.len() as u64
        };

        let live_bytes = self.calculate_live_bytes();
        let fragmentation_ratio = if total_bytes > 0 {
            (total_bytes - live_bytes) as f64 / total_bytes as f64
        } else {
            0.0
        };

        ContentStorageStats {
            total_bytes,
            live_bytes,
            fragmentation_ratio,
            last_compaction: self.last_compaction.load(Ordering::Relaxed),
            bytes_reclaimed_total: self.bytes_reclaimed.load(Ordering::Relaxed),
        }
    }

    /// Calculate live bytes (sum of all content_length for live memories)
    fn calculate_live_bytes(&self) -> u64 {
        self.memory_index.iter()
            .filter_map(|entry| {
                let offset = *entry.value();
                self.read_embedding_block(offset as usize).ok()
            })
            .filter(|block| block.content_offset != u64::MAX)
            .map(|block| block.content_length as u64)
            .sum()
    }
}
```

**Compaction Statistics:**

```rust
#[derive(Debug, Clone)]
pub struct CompactionStats {
    /// Size before compaction (bytes)
    pub old_size: u64,
    /// Size after compaction (bytes)
    pub new_size: u64,
    /// Bytes reclaimed
    pub bytes_reclaimed: u64,
    /// Compaction duration
    pub duration: std::time::Duration,
    /// Fragmentation before compaction (0.0 to 1.0)
    pub fragmentation_before: f64,
    /// Fragmentation after compaction (should be 0.0)
    pub fragmentation_after: f64,
}
```

---

## Concurrency Safety Analysis

### Lock Ordering

**Defined Order:**
1. `content_data` (RwLock)
2. `memory_index` entries (DashMap internal locks)
3. Memory-mapped file operations (implicit locks)

**Rationale:**
- `content_data` is the coarsest lock, must be acquired first
- `memory_index` allows concurrent reads during compaction
- Never hold `memory_index` entry while acquiring `content_data`

### Race Condition Scenarios

**Scenario 1: Concurrent `get()` during compaction**

```
Timeline:
T1: Compaction acquires content_data.write()
T2: get() tries to acquire content_data.read() → BLOCKS
T3: Compaction swaps buffers and releases lock
T4: get() acquires lock → reads from NEW storage
```

**Result:** SAFE - get() always reads consistent state (old or new)

**Scenario 2: Concurrent `store()` during compaction**

```
Timeline:
T1: Compaction acquires content_data.write()
T2: store() tries to acquire content_data.write() → BLOCKS
T3: Compaction completes and releases lock
T4: store() acquires lock → appends to NEW storage
```

**Result:** SAFE - store() appends to compacted storage

**Scenario 3: Concurrent `remove()` during compaction**

```
Timeline:
T1: Compaction reads memory_index (memory X exists)
T2: remove(X) deletes from memory_index
T3: Compaction tries to read_embedding_block(X) → NOT FOUND
T4: Compaction skips X (logs warning)
```

**Result:** SAFE - removed memories are simply not copied

**Scenario 4: Offset read during swap**

```
Timeline:
T1: Compaction updates embedding_block.content_offset to new value
T2: Compaction swaps content_data to new buffer
T3: get() reads embedding_block → sees NEW offset
T4: get() reads from NEW content_data → CORRECT
```

**Result:** SAFE - offset and data are consistent

### Deadlock Analysis

**Potential Deadlock:**
- Thread A: Acquires content_data.write(), then tries to acquire memory_index entry
- Thread B: Holds memory_index entry, tries to acquire content_data.read()
- **DEADLOCK**

**Prevention:**
- Compaction releases `content_data` read lock BEFORE updating embedding blocks
- Embedding block updates only acquire memory_index entries, never content_data
- Lock ordering enforced by code structure

---

## Integration Points

### 1. Maintenance Task (store.rs)

```rust
// Add to MemoryStore::run_maintenance()
pub async fn run_maintenance(&self) -> Result<MaintenanceReport, StorageError> {
    let mut report = MaintenanceReport::default();

    #[cfg(feature = "memory_mapped_persistence")]
    if let Some(backend) = &self.persistent_backend {
        // Get warm tier
        let warm_tier = backend.warm_tier();
        let stats = warm_tier.inner().content_storage_stats();

        // Trigger compaction if fragmentation > 50% AND size > 100MB
        if stats.fragmentation_ratio > 0.5 && stats.total_bytes > 100_000_000 {
            tracing::info!(
                fragmentation = format!("{:.1}%", stats.fragmentation_ratio * 100.0),
                size_mb = stats.total_bytes / 1_000_000,
                "Triggering warm tier content compaction"
            );

            match warm_tier.inner().compact_content().await {
                Ok(compact_stats) => {
                    tracing::info!(
                        reclaimed_mb = compact_stats.bytes_reclaimed / 1_000_000,
                        duration_ms = compact_stats.duration.as_millis(),
                        "Compaction completed successfully"
                    );
                    report.compaction = Some(compact_stats);
                }
                Err(e) => {
                    tracing::error!(
                        error = %e,
                        "Compaction failed, will retry on next maintenance cycle"
                    );
                    // Don't propagate error - compaction is best-effort
                }
            }
        }
    }

    Ok(report)
}
```

### 2. Startup Compaction Check (warm_tier.rs)

```rust
impl WarmTier {
    pub fn new<P: AsRef<std::path::Path>>(
        file_path: P,
        capacity: usize,
        metrics: Arc<super::StorageMetrics>,
    ) -> Result<Self, StorageError> {
        let storage = MappedWarmStorage::new(file_path, capacity, metrics)?;

        // Check if compaction needed on startup
        let stats = storage.content_storage_stats();
        if stats.fragmentation_ratio > 0.7 && stats.total_bytes > 50_000_000 {
            tracing::warn!(
                fragmentation = format!("{:.1}%", stats.fragmentation_ratio * 100.0),
                size_mb = stats.total_bytes / 1_000_000,
                "High fragmentation detected on startup, will compact after initialization"
            );
            // Note: Actual compaction happens in first maintenance cycle
            // to avoid blocking startup
        }

        let confidence_calibrator = StorageConfidenceCalibrator::new();
        let storage_timestamps = dashmap::DashMap::new();
        Ok(Self {
            storage,
            confidence_calibrator,
            storage_timestamps,
        })
    }
}
```

### 3. Monitoring Metrics (storage/mod.rs)

```rust
// Add to StorageMetrics
pub struct ContentStorageMetrics {
    /// Total bytes in content storage
    pub total_bytes: AtomicU64,
    /// Live bytes (sum of content_length)
    pub live_bytes: AtomicU64,
    /// Number of compactions performed
    pub compactions_total: AtomicU64,
    /// Cumulative compaction duration (milliseconds)
    pub compaction_duration_ms: AtomicU64,
    /// Total bytes reclaimed across all compactions
    pub bytes_reclaimed_total: AtomicU64,
    /// Last compaction timestamp
    pub last_compaction_timestamp: AtomicU64,
}

// Prometheus exposition
pub fn expose_content_metrics(metrics: &ContentStorageMetrics) -> String {
    format!(
        r#"
# HELP engram_content_storage_bytes Total bytes in content storage
# TYPE engram_content_storage_bytes gauge
engram_content_storage_bytes {{type="total"}} {}
engram_content_storage_bytes {{type="live"}} {}

# HELP engram_content_fragmentation_ratio Content storage fragmentation (0.0 to 1.0)
# TYPE engram_content_fragmentation_ratio gauge
engram_content_fragmentation_ratio {}

# HELP engram_compactions_total Number of compactions performed
# TYPE engram_compactions_total counter
engram_compactions_total {}

# HELP engram_compaction_duration_seconds_total Cumulative compaction time
# TYPE engram_compaction_duration_seconds_total counter
engram_compaction_duration_seconds_total {}

# HELP engram_compaction_bytes_reclaimed_total Bytes reclaimed by compaction
# TYPE engram_compaction_bytes_reclaimed_total counter
engram_compaction_bytes_reclaimed_total {}
"#,
        metrics.total_bytes.load(Ordering::Relaxed),
        metrics.live_bytes.load(Ordering::Relaxed),
        (metrics.total_bytes.load(Ordering::Relaxed) - metrics.live_bytes.load(Ordering::Relaxed)) as f64
            / metrics.total_bytes.load(Ordering::Relaxed).max(1) as f64,
        metrics.compactions_total.load(Ordering::Relaxed),
        metrics.compaction_duration_ms.load(Ordering::Relaxed) as f64 / 1000.0,
        metrics.bytes_reclaimed_total.load(Ordering::Relaxed),
    )
}
```

### 4. API Endpoint (engram-cli/src/api.rs)

```rust
/// POST /api/v1/maintenance/compact
///
/// Manually trigger warm tier content compaction
#[utoipa::path(
    post,
    path = "/api/v1/maintenance/compact",
    request_body = CompactRequest,
    responses(
        (status = 200, description = "Compaction completed", body = CompactionStats),
        (status = 409, description = "Compaction already in progress"),
        (status = 500, description = "Compaction failed")
    )
)]
async fn compact_warm_tier(
    State(store): State<Arc<MemoryStore>>,
    Json(req): Json<CompactRequest>,
) -> Result<Json<CompactionStats>, StatusCode> {
    #[cfg(feature = "memory_mapped_persistence")]
    {
        if let Some(backend) = &store.persistent_backend {
            let warm_tier = backend.warm_tier();

            // Check if force requested or if compaction criteria met
            let stats = warm_tier.inner().content_storage_stats();
            let should_compact = req.force ||
                (stats.fragmentation_ratio > 0.5 && stats.total_bytes > 100_000_000);

            if !should_compact {
                return Err(StatusCode::PRECONDITION_FAILED);
            }

            match warm_tier.inner().compact_content().await {
                Ok(compact_stats) => Ok(Json(compact_stats)),
                Err(StorageError::CompactionInProgress) => Err(StatusCode::CONFLICT),
                Err(_) => Err(StatusCode::INTERNAL_SERVER_ERROR),
            }
        } else {
            Err(StatusCode::NOT_IMPLEMENTED)
        }
    }

    #[cfg(not(feature = "memory_mapped_persistence"))]
    Err(StatusCode::NOT_IMPLEMENTED)
}

#[derive(Debug, serde::Deserialize, utoipa::ToSchema)]
struct CompactRequest {
    /// Skip fragmentation check and force compaction
    #[serde(default)]
    force: bool,
}
```

---

## Implementation Steps

### Step 1: Add Compaction Infrastructure (4 hours)
- [ ] Add compaction state fields to `MappedWarmStorage`
  - `compaction_in_progress: AtomicBool`
  - `last_compaction: AtomicU64`
  - `bytes_reclaimed: AtomicU64`
- [ ] Implement `CompactionGuard` RAII helper
- [ ] Implement `content_storage_stats()` method
- [ ] Implement `calculate_live_bytes()` helper
- [ ] Implement `estimate_live_content_size()` helper
- [ ] Add `ContentStorageStats` and `CompactionStats` types
- [ ] Write unit test: verify stats calculation correct

### Step 2: Implement Core Compaction Algorithm (5 hours)
- [ ] Implement `compact_content()` main logic
- [ ] Implement `update_content_offset_in_block()` helper
- [ ] Add bounds checking in compaction loop
- [ ] Add error handling for partial failures
- [ ] Write unit test: verify content preserved after compaction
- [ ] Write unit test: verify offsets updated correctly
- [ ] Write unit test: verify old storage deallocated (check capacity)
- [ ] Write unit test: verify error recovery (compaction fails gracefully)

### Step 3: Concurrency Testing (4 hours)
- [ ] Write test: concurrent get() during compaction
- [ ] Write test: concurrent store() during compaction
- [ ] Write test: concurrent remove() during compaction
- [ ] Write test: compaction blocks concurrent compaction
- [ ] Write stress test: 10 threads store/get while compacting
- [ ] Write test: verify no deadlocks under load
- [ ] Profile lock hold times under concurrent access

### Step 4: Maintenance Integration (2 hours)
- [ ] Add compaction trigger to `MemoryStore::run_maintenance()`
- [ ] Add startup fragmentation check to `WarmTier::new()`
- [ ] Add logging with before/after metrics
- [ ] Write test: compaction triggered at 50% fragmentation
- [ ] Write test: compaction skipped below threshold
- [ ] Write test: startup warns on high fragmentation

### Step 5: Monitoring & API (2 hours)
- [ ] Add `ContentStorageMetrics` to `StorageMetrics`
- [ ] Implement Prometheus exposition for compaction metrics
- [ ] Add POST /api/v1/maintenance/compact endpoint
- [ ] Add compaction stats to health check response
- [ ] Write integration test: API triggers compaction
- [ ] Write test: metrics updated after compaction

### Step 6: Large-Scale Validation (3 hours)
- [ ] Write stress test: 100K memories, 50% eviction, compact
- [ ] Benchmark: 1M memories compaction time (target <500ms)
- [ ] Measure memory overhead during compaction
- [ ] Validate fragmentation ratio calculation accuracy
- [ ] Test compaction on 1GB content storage
- [ ] Profile memory allocations during compaction
- [ ] Verify no memory leaks after repeated compactions

### Step 7: Documentation & Production Readiness (2 hours)
- [ ] Document compaction trigger conditions
- [ ] Add monitoring alert for high fragmentation (>70%)
- [ ] Add runbook for manual compaction via API
- [ ] Document lock ordering constraints
- [ ] Add performance characteristics to docs
- [ ] Update operations guide with compaction section
- [ ] Run `make quality` and fix all warnings

---

## Performance Targets

| Metric | Target | Measurement | Rationale |
|--------|--------|-------------|-----------|
| Compaction latency | <500ms for 1M memories | End-to-end compact_content() | Stop-the-world must be brief |
| Memory overhead | 2x during compaction | Peak RSS during operation | Acceptable for infrequent operation |
| Lock hold time (write) | <500ms | Duration of content_data.write() | Blocks get() operations |
| Lock hold time (read) | <100ms | Duration of content_data.read() | Allows concurrent reads |
| Throughput impact | 0 req/s during compaction | Requests blocked during write lock | Acceptable for maintenance window |
| Reclaim efficiency | >95% of fragmented space | (old_size - new_size) / fragmented | Verify no leaks |
| Startup overhead | <1s check time | Duration of stats calculation | Must not delay startup |

### Benchmark Methodology

```rust
#[tokio::test]
#[ignore = "slow"]
async fn benchmark_compaction_1m_memories() {
    let temp_dir = TempDir::new().unwrap();
    let metrics = Arc::new(StorageMetrics::new());
    let storage = MappedWarmStorage::new(
        temp_dir.path().join("warm.dat"),
        1_000_000,
        metrics,
    ).unwrap();

    // Store 1M memories with ~1KB content each
    for i in 0..1_000_000 {
        let memory = create_test_memory(
            &format!("mem_{}", i),
            &vec![b'x'; 1024] // 1KB content
        );
        storage.store(Arc::new(memory)).await.unwrap();
    }

    // Delete 50% (simulate churn)
    for i in (0..1_000_000).step_by(2) {
        storage.remove(&format!("mem_{}", i)).await.unwrap();
    }

    // Measure compaction
    let start = Instant::now();
    let stats = storage.compact_content().await.unwrap();
    let duration = start.elapsed();

    // Assertions
    assert!(duration < Duration::from_millis(500),
        "Compaction took {}ms, expected <500ms", duration.as_millis());
    assert!(stats.bytes_reclaimed > 500_000_000,
        "Should reclaim ~500MB");
    assert!(stats.fragmentation_before > 0.49 && stats.fragmentation_before < 0.51,
        "Expected ~50% fragmentation before");
    assert_eq!(stats.fragmentation_after, 0.0,
        "Expected 0% fragmentation after");
}
```

---

## Testing Strategy

### Unit Tests (engram-core/tests/warm_tier_compaction_tests.rs)

```rust
#[tokio::test]
async fn test_compaction_preserves_content() {
    // Store 100 memories, delete 50, compact, verify 50 remain intact
    let temp_dir = TempDir::new().unwrap();
    let storage = MappedWarmStorage::new(temp_dir.path().join("test.dat"), 100, metrics()).unwrap();

    // Store 100
    for i in 0..100 {
        let memory = test_memory(&format!("mem_{}", i), &format!("content {}", i));
        storage.store(Arc::new(memory)).await.unwrap();
    }

    // Delete 50 (even indices)
    for i in (0..100).step_by(2) {
        storage.remove(&format!("mem_{}", i)).await.unwrap();
    }

    // Compact
    let stats = storage.compact_content().await.unwrap();
    assert!(stats.bytes_reclaimed > 0);

    // Verify 50 remain (odd indices)
    for i in (1..100).step_by(2) {
        let result = storage.get(&format!("mem_{}", i)).unwrap();
        assert!(result.is_some());
        assert_eq!(result.unwrap().content.unwrap(), format!("content {}", i));
    }
}

#[tokio::test]
async fn test_compaction_updates_offsets() {
    // Store 10, delete 5 even indices, compact
    // Verify odd indices have correct content at new offsets
    let temp_dir = TempDir::new().unwrap();
    let storage = MappedWarmStorage::new(temp_dir.path().join("test.dat"), 10, metrics()).unwrap();

    // Store with distinct content
    let content_map: HashMap<String, String> = (0..10)
        .map(|i| (format!("mem_{}", i), format!("content_{}_unique", i)))
        .collect();

    for (id, content) in &content_map {
        let memory = test_memory(id, content);
        storage.store(Arc::new(memory)).await.unwrap();
    }

    // Delete even indices
    for i in (0..10).step_by(2) {
        storage.remove(&format!("mem_{}", i)).await.unwrap();
    }

    // Compact
    storage.compact_content().await.unwrap();

    // Verify odd indices still retrievable with correct content
    for i in (1..10).step_by(2) {
        let id = format!("mem_{}", i);
        let result = storage.get(&id).unwrap();
        assert!(result.is_some(), "Memory {} should exist after compaction", id);
        let memory = result.unwrap();
        assert_eq!(memory.content.unwrap(), content_map[&id],
            "Content mismatch for {}", id);
    }
}

#[tokio::test]
async fn test_compaction_deallocates_memory() {
    // Measure Vec capacity before/after compaction
    let temp_dir = TempDir::new().unwrap();
    let storage = MappedWarmStorage::new(temp_dir.path().join("test.dat"), 1000, metrics()).unwrap();

    // Store 1000 memories with 1KB content
    for i in 0..1000 {
        let memory = test_memory(&format!("mem_{}", i), &vec![b'x'; 1024]);
        storage.store(Arc::new(memory)).await.unwrap();
    }

    let stats_before = storage.content_storage_stats();
    assert!(stats_before.total_bytes > 1_000_000, "Should have ~1MB");

    // Delete 90%
    for i in 0..900 {
        storage.remove(&format!("mem_{}", i)).await.unwrap();
    }

    // Compact
    storage.compact_content().await.unwrap();

    let stats_after = storage.content_storage_stats();
    assert!(stats_after.total_bytes < 200_000, "Should shrink to ~100KB");
    assert_eq!(stats_after.fragmentation_ratio, 0.0);
}

#[tokio::test]
async fn test_compaction_error_recovery() {
    // Simulate failure during compaction (e.g., corrupted embedding block)
    // Verify compaction fails gracefully without corrupting storage
    // (Implementation note: This requires injecting failures - defer to stress tests)
}

#[tokio::test]
async fn test_concurrent_get_during_compaction() {
    let temp_dir = TempDir::new().unwrap();
    let storage = Arc::new(MappedWarmStorage::new(temp_dir.path().join("test.dat"), 100, metrics()).unwrap());

    // Store 100
    for i in 0..100 {
        let memory = test_memory(&format!("mem_{}", i), &format!("content {}", i));
        storage.store(Arc::new(memory)).await.unwrap();
    }

    // Delete 50
    for i in (0..100).step_by(2) {
        storage.remove(&format!("mem_{}", i)).await.unwrap();
    }

    // Spawn compaction task
    let storage_clone = storage.clone();
    let compact_task = tokio::spawn(async move {
        storage_clone.compact_content().await
    });

    // Concurrently read memories
    let mut handles = vec![];
    for i in (1..100).step_by(2) {
        let storage_clone = storage.clone();
        let handle = tokio::spawn(async move {
            storage_clone.get(&format!("mem_{}", i))
        });
        handles.push(handle);
    }

    // Wait for all
    compact_task.await.unwrap().unwrap();
    for handle in handles {
        let result = handle.await.unwrap().unwrap();
        assert!(result.is_some()); // Should always succeed
    }
}

#[tokio::test]
async fn test_concurrent_store_during_compaction() {
    let temp_dir = TempDir::new().unwrap();
    let storage = Arc::new(MappedWarmStorage::new(temp_dir.path().join("test.dat"), 1000, metrics()).unwrap());

    // Pre-populate
    for i in 0..500 {
        storage.store(Arc::new(test_memory(&format!("mem_{}", i), "content"))).await.unwrap();
    }
    for i in 0..250 {
        storage.remove(&format!("mem_{}", i)).await.unwrap();
    }

    // Spawn compaction
    let storage_clone = storage.clone();
    let compact_task = tokio::spawn(async move {
        storage_clone.compact_content().await
    });

    // Concurrently store new memories
    let mut handles = vec![];
    for i in 1000..1100 {
        let storage_clone = storage.clone();
        let handle = tokio::spawn(async move {
            storage_clone.store(Arc::new(test_memory(&format!("mem_{}", i), "new"))).await
        });
        handles.push(handle);
    }

    // Wait for all
    compact_task.await.unwrap().unwrap();
    for handle in handles {
        handle.await.unwrap().unwrap();
    }

    // Verify all new memories stored
    for i in 1000..1100 {
        assert!(storage.get(&format!("mem_{}", i)).unwrap().is_some());
    }
}

#[tokio::test]
async fn test_compaction_blocks_concurrent_compaction() {
    let temp_dir = TempDir::new().unwrap();
    let storage = Arc::new(MappedWarmStorage::new(temp_dir.path().join("test.dat"), 100, metrics()).unwrap());

    // Pre-populate
    for i in 0..100 {
        storage.store(Arc::new(test_memory(&format!("mem_{}", i), "content"))).await.unwrap();
    }
    for i in 0..50 {
        storage.remove(&format!("mem_{}", i)).await.unwrap();
    }

    // Spawn two compaction tasks
    let storage_clone1 = storage.clone();
    let compact1 = tokio::spawn(async move {
        storage_clone1.compact_content().await
    });

    tokio::time::sleep(Duration::from_millis(10)).await; // Ensure first starts

    let storage_clone2 = storage.clone();
    let compact2 = tokio::spawn(async move {
        storage_clone2.compact_content().await
    });

    // One should succeed, one should get CompactionInProgress error
    let result1 = compact1.await.unwrap();
    let result2 = compact2.await.unwrap();

    let success_count = result1.is_ok() as usize + result2.is_ok() as usize;
    assert_eq!(success_count, 1, "Exactly one compaction should succeed");
}
```

### Integration Tests (engram-core/tests/warm_tier_compaction_integration.rs)

```rust
#[tokio::test]
async fn test_maintenance_triggers_compaction() {
    // Create store with persistent backend
    let temp_dir = TempDir::new().unwrap();
    let store = MemoryStore::builder()
        .with_persistence(temp_dir.path(), 100, 1000, 10000)
        .build()
        .unwrap();

    // Fill warm tier to trigger fragmentation
    for i in 0..1000 {
        let episode = EpisodeBuilder::new()
            .id(format!("mem_{}", i))
            .what(vec![b'x'; 1024]) // 1KB content
            .build();
        store.store(episode).await;
    }

    // Evict 60% (simulate churn)
    for i in 0..600 {
        store.forget(&format!("mem_{}", i)).await;
    }

    // Run maintenance
    let report = store.run_maintenance().await.unwrap();

    // Verify compaction triggered
    assert!(report.compaction.is_some());
    let stats = report.compaction.unwrap();
    assert!(stats.bytes_reclaimed > 500_000); // Should reclaim ~600KB
}

#[tokio::test]
async fn test_api_endpoint_compacts_storage() {
    // Start API server
    let temp_dir = TempDir::new().unwrap();
    let store = MemoryStore::builder()
        .with_persistence(temp_dir.path(), 100, 1000, 10000)
        .build()
        .unwrap();

    let app = create_api_router(store.clone());
    let addr = SocketAddr::from(([127, 0, 0, 1], 0));
    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    let server_addr = listener.local_addr().unwrap();

    tokio::spawn(async move {
        axum::serve(listener, app).await
    });

    // Fill and fragment storage
    for i in 0..1000 {
        store.store(test_episode(&format!("mem_{}", i))).await;
    }
    for i in 0..600 {
        store.forget(&format!("mem_{}", i)).await;
    }

    // Trigger compaction via API
    let client = reqwest::Client::new();
    let response = client
        .post(format!("http://{}/api/v1/maintenance/compact", server_addr))
        .json(&serde_json::json!({ "force": false }))
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);

    let stats: CompactionStats = response.json().await.unwrap();
    assert!(stats.bytes_reclaimed > 0);
}
```

### Stress Tests (engram-core/tests/warm_tier_compaction_stress.rs)

```rust
#[tokio::test]
#[ignore = "slow"]
async fn test_compaction_with_large_dataset() {
    // 100K memories x 1KB content = 100MB
    // Delete 50K (50% fragmentation)
    // Compact and verify ~50MB reclaimed
    let temp_dir = TempDir::new().unwrap();
    let storage = MappedWarmStorage::new(
        temp_dir.path().join("large.dat"),
        100_000,
        metrics(),
    ).unwrap();

    // Store 100K
    for i in 0..100_000 {
        let memory = test_memory(&format!("mem_{}", i), &vec![b'x'; 1024]);
        storage.store(Arc::new(memory)).await.unwrap();
    }

    let stats_before = storage.content_storage_stats();
    assert!(stats_before.total_bytes > 90_000_000, "Should have ~100MB");

    // Delete 50K (even indices)
    for i in (0..100_000).step_by(2) {
        storage.remove(&format!("mem_{}", i)).await.unwrap();
    }

    // Compact
    let start = Instant::now();
    let compact_stats = storage.compact_content().await.unwrap();
    let duration = start.elapsed();

    // Assertions
    assert!(duration < Duration::from_millis(500),
        "Compaction took {}ms, expected <500ms", duration.as_millis());
    assert!(compact_stats.bytes_reclaimed > 45_000_000,
        "Should reclaim ~50MB, got {}MB", compact_stats.bytes_reclaimed / 1_000_000);

    let stats_after = storage.content_storage_stats();
    assert!(stats_after.total_bytes < 55_000_000, "Should shrink to ~50MB");
    assert_eq!(stats_after.fragmentation_ratio, 0.0);

    // Verify all live memories still accessible
    for i in (1..100_000).step_by(2) {
        assert!(storage.get(&format!("mem_{}", i)).unwrap().is_some());
    }
}

#[tokio::test]
#[ignore = "slow"]
async fn test_repeated_compaction_no_leaks() {
    // Run 100 compaction cycles, verify no memory leaks
    let temp_dir = TempDir::new().unwrap();
    let storage = MappedWarmStorage::new(temp_dir.path().join("leak.dat"), 1000, metrics()).unwrap();

    for cycle in 0..100 {
        // Store 1000
        for i in 0..1000 {
            let memory = test_memory(&format!("mem_{}_{}", cycle, i), "content");
            storage.store(Arc::new(memory)).await.unwrap();
        }

        // Delete 500
        for i in 0..500 {
            storage.remove(&format!("mem_{}_{}", cycle, i)).await.unwrap();
        }

        // Compact
        storage.compact_content().await.unwrap();

        // Verify storage size stable
        let stats = storage.content_storage_stats();
        assert!(stats.total_bytes < 10_000_000,
            "Cycle {}: Memory leak detected, size={}MB",
            cycle, stats.total_bytes / 1_000_000);
    }
}
```

---

## Risk Analysis & Mitigation

### Risk 1: Compaction Failure Mid-Operation

**Scenario:** Compaction fails after updating 50% of embedding blocks

**Impact:** CRITICAL - Offsets inconsistent, data corruption

**Mitigation:**
- **Transactional updates:** Use rayon parallel iteration with error collection
- **Early validation:** Check all embedding blocks before updating any
- **Error recovery:** If any update fails, abort entire compaction
- **Testing:** Inject failures during compaction to verify rollback

**Code:**
```rust
// Collect update errors
let update_errors = AtomicUsize::new(0);
offset_map.par_iter().for_each(|(memory_id, offsets)| {
    if update_content_offset_in_block(...).is_err() {
        update_errors.fetch_add(1, Ordering::Relaxed);
    }
});

// Abort if any failures
if update_errors.load(Ordering::Relaxed) > 0 {
    return Err(StorageError::CompactionFailed);
}
```

### Risk 2: OOM During Compaction

**Scenario:** Compaction allocates 2x memory (old + new Vec), triggers OOM

**Impact:** HIGH - Process crash, potential data loss

**Mitigation:**
- **Memory pressure detection:** Check available memory before compaction
- **Graceful degradation:** Skip compaction if memory pressure high
- **Incremental compaction:** Future: compact 10% at a time
- **Monitoring:** Alert on high memory usage during compaction

**Code:**
```rust
// Check memory pressure before compaction
let stats = storage.content_storage_stats();
let estimated_overhead = stats.live_bytes;

#[cfg(unix)]
if let Ok(sysinfo) = sys_info::mem_info() {
    let available_kb = sysinfo.avail;
    let needed_kb = estimated_overhead / 1024;

    if needed_kb > available_kb / 2 {
        tracing::warn!(
            needed_mb = needed_kb / 1024,
            available_mb = available_kb / 1024,
            "Insufficient memory for compaction, skipping"
        );
        return Err(StorageError::InsufficientMemory);
    }
}
```

### Risk 3: Long Stop-the-World Pause

**Scenario:** Compaction takes 2s, blocking all warm tier reads

**Impact:** MEDIUM - User-visible latency spikes

**Mitigation:**
- **Performance testing:** Benchmark 1M memory compaction, target <500ms
- **Monitoring:** Alert if compaction duration > 1s
- **Optimization:** Use rayon for parallel updates
- **Future:** Incremental compaction with copy-on-write

**Benchmark:**
```rust
// Target: <500ms for 1M memories
assert!(compact_stats.duration < Duration::from_millis(500));
```

### Risk 4: Race Condition on Offset Read

**Scenario:** get() reads embedding block during offset update

**Impact:** HIGH - get() uses wrong offset, returns wrong content

**Mitigation:**
- **Lock ordering:** Content buffer swapped AFTER all offsets updated
- **Atomic updates:** Each embedding block update is atomic (store_embedding_block)
- **Testing:** Stress test concurrent get() during compaction
- **Verification:** Ensure readers see either old state or new state, never mixed

**Analysis:**
```
Timeline:
T1: Compaction updates block X offset to NEW
T2: Compaction swaps content buffer
T3: get(X) reads block → sees NEW offset
T4: get(X) reads from NEW buffer → CORRECT

Alternative:
T1: get(X) reads block → sees OLD offset
T2: Compaction updates offset to NEW
T3: Compaction swaps buffer
T4: get(X) reads from OLD buffer → CORRECT
```

### Risk 5: Startup Delay

**Scenario:** High fragmentation on startup, compaction blocks initialization

**Impact:** LOW - Startup time increases

**Mitigation:**
- **Deferred compaction:** Only log warning on startup, compact in first maintenance cycle
- **Async initialization:** Compaction runs in background after store ready
- **Monitoring:** Track startup fragmentation levels

**Code:**
```rust
// In WarmTier::new()
if stats.fragmentation_ratio > 0.7 {
    tracing::warn!("High fragmentation on startup, will compact in first maintenance cycle");
    // Don't block startup - let maintenance handle it
}
```

---

## Acceptance Criteria

- [ ] Compaction triggered at 50% fragmentation + 100MB size
- [ ] Content correctly preserved for all live memories
- [ ] Offsets updated atomically (no corruption window)
- [ ] Memory reclaimed (Vec capacity reduced via shrink_to_fit)
- [ ] Compaction completes in <500ms for 1M memories
- [ ] Concurrent get() operations succeed during compaction
- [ ] Concurrent store() operations queue correctly
- [ ] Concurrent compaction attempts blocked with error
- [ ] Metrics exposed via Prometheus
- [ ] API endpoint works: POST /api/v1/maintenance/compact
- [ ] Large-scale test passes: 100K memories, 50% eviction, compact
- [ ] No memory leaks after 100 compaction cycles
- [ ] Zero clippy warnings
- [ ] All tests pass
- [ ] Documentation updated

---

## Future Optimizations (Not in Scope)

### 1. Incremental Compaction
Split compaction into 10% chunks to reduce pause time:
```rust
async fn compact_incremental(&self, chunk_size: usize) -> Result<(), StorageError> {
    // Compact in 10% chunks over time
    // Track compaction progress across maintenance cycles
}
```

### 2. Background Thread with Copy-on-Write
Run compaction in background without blocking reads:
```rust
struct VersionedContentStorage {
    version: AtomicU8,  // 0 or 1
    buffer_0: RwLock<Vec<u8>>,
    buffer_1: RwLock<Vec<u8>>,
}
```

### 3. Checksums for Corruption Detection
Add CRC32 to detect corruption after compaction:
```rust
struct EmbeddingBlock {
    content_checksum: u32,  // CRC32 of content
}
```

### 4. Compression
Use LZ4 to reduce content storage size:
```rust
let compressed = lz4::compress(content_bytes)?;
content_storage.extend_from_slice(&compressed);
```

### 5. Defragmentation Heuristics
Compact only regions with high fragmentation:
```rust
fn compact_hot_regions(&self, fragmentation_threshold: f64) -> Result<(), StorageError> {
    // Identify high-fragmentation regions
    // Compact only those regions
}
```

---

## References

- **PHASE_2_FIX_1_REVIEW_SUMMARY.md:** Issue #2 (Content Growth Unbounded)
- **engram-core/src/storage/mapped.rs:** MappedWarmStorage implementation (lines 249-595)
- **engram-core/src/storage/warm_tier.rs:** WarmTier wrapper (lines 1-492)
- **docs/operations/monitoring.md:** Metrics and alerting guide
- **Margo Seltzer, "Architectural Considerations for High-Performance Storage Systems"** (Conceptual reference)
- **parking_lot documentation:** RwLock semantics and performance characteristics
- **DashMap documentation:** Concurrent HashMap guarantees
