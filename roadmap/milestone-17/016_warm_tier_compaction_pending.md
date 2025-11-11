# Task 016: Warm Tier Content Storage Compaction

**Status:** PENDING
**Priority:** HIGH
**Estimated Effort:** 8 hours
**Blocking:** None
**Blocked By:** Task 005 (Binding Formation) - must be complete

---

## Problem

The warm tier content storage uses a `Vec<u8>` that grows without bounds as memories are stored. When memories are evicted or deleted, their content remains in the Vec, creating fragmentation and memory leaks.

**Impact:**
- Memory growth: 100MB → 4GB after 1 year with 10% churn rate
- Eventual OOM in long-running deployments
- Wasted memory proportional to churn rate

**Current Behavior:**
```rust
// Content appended to Vec<u8>, offset stored in EmbeddingBlock
let offset = content_storage.len() as u64;
content_storage.extend_from_slice(content_bytes);
block.content_offset = offset;
block.content_length = content_len as u32;
```

**Problem:** When memory is evicted, `content_storage` never shrinks - deleted content leaves "holes".

---

## Solution: Stop-the-World Compaction

Implement periodic compaction that rebuilds the content storage without holes.

### Design

**Trigger Conditions:**
```rust
pub struct ContentStorageStats {
    pub total_bytes: u64,
    pub live_bytes: u64,
    pub fragmentation_ratio: f64,  // (total - live) / total
}

// Trigger compaction when:
// 1. Fragmentation > 50%
// 2. Total size > 100MB
// 3. Manual trigger via maintenance API
```

**Compaction Algorithm:**
```rust
pub async fn compact_content(&self) -> Result<CompactionStats, StorageError> {
    // 1. Acquire write lock (blocks all stores)
    let mut content_storage = self.content_data.write();

    // 2. Collect all live content with new offsets
    let mut new_content = Vec::with_capacity(estimate_live_size());
    let mut offset_map = HashMap::new();  // old_offset -> new_offset

    for (memory_id, block) in self.embeddings.iter() {
        if block.content_offset == u64::MAX {
            continue;  // Skip None content
        }

        let old_offset = block.content_offset;
        let new_offset = new_content.len() as u64;

        // Copy content to new Vec
        let start = old_offset as usize;
        let end = start + block.content_length as usize;
        new_content.extend_from_slice(&content_storage[start..end]);

        offset_map.insert(memory_id.clone(), new_offset);
    }

    // 3. Update all embedding blocks with new offsets
    for (memory_id, new_offset) in offset_map {
        if let Some(mut block) = self.embeddings.get_mut(&memory_id) {
            block.content_offset = new_offset;
        }
    }

    // 4. Atomically swap in new storage
    let old_size = content_storage.len();
    *content_storage = new_content;
    let new_size = content_storage.len();

    Ok(CompactionStats {
        old_size,
        new_size,
        bytes_reclaimed: old_size - new_size,
        duration: /* ... */,
    })
}
```

### Integration Points

**1. Maintenance Task:**
```rust
// In store.rs
pub async fn run_maintenance(&self) -> Result<MaintenanceReport, StorageError> {
    let mut report = MaintenanceReport::default();

    if let Some(backend) = &self.persistent_backend {
        let stats = backend.content_storage_stats();

        if stats.fragmentation_ratio > 0.5 && stats.total_bytes > 100_000_000 {
            tracing::info!(
                "Triggering compaction: fragmentation={:.2}%, size={}MB",
                stats.fragmentation_ratio * 100.0,
                stats.total_bytes / 1_000_000
            );

            let compact_stats = backend.compact_content().await?;
            report.compaction = Some(compact_stats);
        }
    }

    Ok(report)
}
```

**2. Monitoring Metrics:**
```rust
// Add to StorageMetrics
pub struct ContentStorageMetrics {
    pub total_bytes: AtomicU64,
    pub live_bytes: AtomicU64,
    pub compactions_total: AtomicU64,
    pub compaction_duration_ms: AtomicU64,
    pub bytes_reclaimed_total: AtomicU64,
}
```

**3. API Endpoint:**
```http
POST /api/v1/maintenance/compact
{
  "force": true  // Skip fragmentation check
}

Response:
{
  "old_size": 4294967296,
  "new_size": 104857600,
  "bytes_reclaimed": 4190109696,
  "duration_ms": 1234,
  "fragmentation_before": 0.975,
  "fragmentation_after": 0.0
}
```

---

## Implementation Steps

### Step 1: Add Compaction Method (3 hours)
- [ ] Implement `compact_content()` in `MappedWarmStorage`
- [ ] Add `content_storage_stats()` method
- [ ] Write unit test: verify content preserved after compaction
- [ ] Write unit test: verify offsets updated correctly
- [ ] Write unit test: verify old storage deallocated

### Step 2: Maintenance Integration (2 hours)
- [ ] Add `run_maintenance()` to MemoryStore
- [ ] Implement fragmentation threshold checks
- [ ] Add logging with before/after metrics
- [ ] Write test: compaction triggered at 50% fragmentation
- [ ] Write test: compaction skipped below threshold

### Step 3: Monitoring & API (2 hours)
- [ ] Add ContentStorageMetrics to StorageMetrics
- [ ] Expose metrics via Prometheus endpoint
- [ ] Add POST /api/v1/maintenance/compact endpoint
- [ ] Add compaction stats to health check response
- [ ] Write integration test: API triggers compaction

### Step 4: Documentation & Validation (1 hour)
- [ ] Document compaction trigger conditions
- [ ] Add monitoring alert for high fragmentation
- [ ] Add performance baseline: 1M memories compacts in <2s
- [ ] Run stress test: 10K stores + 50% eviction + compact
- [ ] Verify memory reclaimed correctly

---

## Performance Targets

| Metric | Target | Measurement |
|--------|--------|-------------|
| Compaction latency | <2s for 1M memories | 1M x 1KB content = 1GB |
| Memory overhead | <10% during compaction | Dual buffers |
| Throughput impact | 0% (stops writes) | Stop-the-world |
| Reclaim efficiency | >95% of fragmented space | Measure delta |

---

## Testing Strategy

### Unit Tests
```rust
#[tokio::test]
async fn test_compaction_preserves_content() {
    // Store 100 memories, delete 50, compact, verify 50 remain
}

#[tokio::test]
async fn test_compaction_updates_offsets() {
    // Store 10 memories, delete 5 even indices, compact
    // Verify odd indices have correct content at new offsets
}

#[tokio::test]
async fn test_compaction_deallocates_memory() {
    // Measure Vec capacity before/after compaction
}
```

### Integration Tests
```rust
#[tokio::test]
async fn test_maintenance_triggers_compaction() {
    // Fill warm tier, evict 60%, run maintenance
    // Verify compaction triggered and memory reclaimed
}

#[tokio::test]
async fn test_api_endpoint_compacts_storage() {
    // POST /api/v1/maintenance/compact
    // Verify response contains compaction stats
}
```

### Stress Tests
```rust
#[tokio::test]
#[ignore = "slow"]
async fn test_compaction_with_large_dataset() {
    // 100K memories x 1KB content = 100MB
    // Delete 50K (50% fragmentation)
    // Compact and verify ~50MB reclaimed
}
```

---

## Acceptance Criteria

- [ ] Compaction triggered at 50% fragmentation + 100MB size
- [ ] Content correctly preserved for all live memories
- [ ] Offsets updated atomically (no corruption)
- [ ] Memory reclaimed (Vec capacity reduced)
- [ ] Compaction completes in <2s for 1M memories
- [ ] Metrics exposed via Prometheus
- [ ] API endpoint works: POST /api/v1/maintenance/compact
- [ ] Zero clippy warnings
- [ ] All tests pass
- [ ] Documentation updated

---

## Future Optimizations (Not in Scope)

1. **Incremental Compaction:** Split into smaller chunks to reduce stop-the-world pause
2. **Background Thread:** Run compaction in background with copy-on-write
3. **Checksums:** Add CRC32 to detect corruption after compaction
4. **Compression:** Use LZ4 to reduce content storage size

---

## References

- PHASE_2_FIX_1_REVIEW_SUMMARY.md: Issue #2 (Content Growth Unbounded)
- engram-core/src/storage/mapped.rs: MappedWarmStorage implementation
- docs/operations/monitoring.md: Metrics and alerting
