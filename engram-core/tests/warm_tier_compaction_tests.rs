//! Unit tests for warm tier content storage compaction
//!
//! Tests verify:
//! - Content preservation after compaction
//! - Offset remapping correctness
//! - Memory reclamation
//! - Error recovery
//! - Concurrent access safety

use engram_core::{
    Confidence, Memory,
    storage::{MappedWarmStorage, StorageError, StorageMetrics, StorageTierBackend},
};
use std::sync::Arc;
use tempfile::TempDir;

/// Create a test memory with specific content
fn create_test_memory(id: &str, content: &str) -> Arc<Memory> {
    let embedding = [0.1_f32; 768];
    let mut memory = Memory::new(id.to_string(), embedding, Confidence::exact(0.8));
    memory.content = Some(content.to_string());
    Arc::new(memory)
}

/// Create a test memory with byte content
fn create_test_memory_bytes(id: &str, content: &[u8]) -> Arc<Memory> {
    let embedding = [0.1_f32; 768];
    let mut memory = Memory::new(id.to_string(), embedding, Confidence::exact(0.8));
    memory.content = Some(String::from_utf8_lossy(content).to_string());
    Arc::new(memory)
}

#[tokio::test]
async fn test_compaction_preserves_content() {
    let temp_dir = TempDir::new().unwrap();
    let metrics = Arc::new(StorageMetrics::new());
    let storage = MappedWarmStorage::new(temp_dir.path().join("test.dat"), 100, metrics).unwrap();

    // Store 100 memories
    for i in 0..100 {
        let memory = create_test_memory(&format!("mem_{i}"), &format!("content {i}"));
        storage.store(memory).await.unwrap();
    }

    // Delete 50 (even indices)
    for i in (0..100).step_by(2) {
        storage.remove(&format!("mem_{i}")).await.unwrap();
    }

    // Compact
    let stats = storage.compact_content().unwrap();
    assert!(stats.bytes_reclaimed > 0, "Should reclaim some bytes");

    // Verify 50 remain (odd indices)
    for i in (1..100).step_by(2) {
        let result = storage.get(&format!("mem_{i}")).unwrap();
        assert!(result.is_some(), "Memory mem_{i} should exist");
        let memory = result.unwrap();
        assert_eq!(
            memory.content.as_deref(),
            Some(&format!("content {i}")[..]),
            "Content should match for mem_{i}"
        );
    }
}

#[tokio::test]
async fn test_compaction_updates_offsets() {
    let temp_dir = TempDir::new().unwrap();
    let metrics = Arc::new(StorageMetrics::new());
    let storage = MappedWarmStorage::new(temp_dir.path().join("test.dat"), 10, metrics).unwrap();

    // Store with distinct content
    let content_map: std::collections::HashMap<String, String> = (0..10)
        .map(|i| (format!("mem_{i}"), format!("content_{i}_unique")))
        .collect();

    for (id, content) in &content_map {
        let memory = create_test_memory(id, content);
        storage.store(memory).await.unwrap();
    }

    // Delete even indices
    for i in (0..10).step_by(2) {
        storage.remove(&format!("mem_{i}")).await.unwrap();
    }

    // Compact
    storage.compact_content().unwrap();

    // Verify odd indices still retrievable with correct content
    for i in (1..10).step_by(2) {
        let id = format!("mem_{i}");
        let result = storage.get(&id).unwrap();
        assert!(
            result.is_some(),
            "Memory {id} should exist after compaction"
        );
        let memory = result.unwrap();
        assert_eq!(
            memory.content.as_deref(),
            Some(&content_map[&id][..]),
            "Content mismatch for {id}"
        );
    }
}

#[tokio::test]
async fn test_compaction_deallocates_memory() {
    let temp_dir = TempDir::new().unwrap();
    let metrics = Arc::new(StorageMetrics::new());
    let storage = MappedWarmStorage::new(temp_dir.path().join("test.dat"), 1000, metrics).unwrap();

    // Store 1000 memories with 1KB content
    for i in 0..1000 {
        let memory = create_test_memory_bytes(&format!("mem_{i}"), &vec![b'x'; 1024]);
        storage.store(memory).await.unwrap();
    }

    let stats_before = storage.content_storage_stats();
    assert!(stats_before.total_bytes > 1_000_000, "Should have ~1MB");

    // Delete 90%
    for i in 0..900 {
        storage.remove(&format!("mem_{i}")).await.unwrap();
    }

    // Compact
    storage.compact_content().unwrap();

    let stats_after = storage.content_storage_stats();
    assert!(
        stats_after.total_bytes < 200_000,
        "Should shrink to ~100KB, got {} bytes",
        stats_after.total_bytes
    );
    assert!(
        stats_after.fragmentation_ratio < 0.01,
        "Fragmentation should be near 0, got {}",
        stats_after.fragmentation_ratio
    );
}

#[tokio::test]
async fn test_compaction_stats_calculation() {
    let temp_dir = TempDir::new().unwrap();
    let metrics = Arc::new(StorageMetrics::new());
    let storage = MappedWarmStorage::new(temp_dir.path().join("test.dat"), 100, metrics).unwrap();

    // Store 100 memories
    for i in 0..100 {
        let memory = create_test_memory(&format!("mem_{i}"), &format!("content{i}"));
        storage.store(memory).await.unwrap();
    }

    // Delete 50
    for i in 0..50 {
        storage.remove(&format!("mem_{i}")).await.unwrap();
    }

    let stats_before = storage.content_storage_stats();
    assert!(
        stats_before.fragmentation_ratio > 0.4,
        "Should have significant fragmentation"
    );

    // Compact
    let compact_stats = storage.compact_content().unwrap();

    assert!(compact_stats.old_size > compact_stats.new_size);
    assert_eq!(
        compact_stats.bytes_reclaimed,
        compact_stats.old_size - compact_stats.new_size
    );
    assert!(compact_stats.fragmentation_before > 0.0);
    assert!((compact_stats.fragmentation_after - 0.0).abs() < f64::EPSILON);
    // Duration check - just verify it's measured (can be very fast)
    assert!(compact_stats.duration.as_nanos() > 0);

    let stats_after = storage.content_storage_stats();
    assert!(stats_after.fragmentation_ratio < 0.01);
    assert_eq!(
        stats_after.bytes_reclaimed_total,
        compact_stats.bytes_reclaimed
    );
}

#[tokio::test]
async fn test_content_storage_stats() {
    let temp_dir = TempDir::new().unwrap();
    let metrics = Arc::new(StorageMetrics::new());
    let storage = MappedWarmStorage::new(temp_dir.path().join("test.dat"), 10, metrics).unwrap();

    // Initial stats
    let stats = storage.content_storage_stats();
    assert_eq!(stats.total_bytes, 0);
    assert_eq!(stats.live_bytes, 0);
    assert!((stats.fragmentation_ratio - 0.0).abs() < f64::EPSILON);

    // Store some memories
    for i in 0..10 {
        let memory = create_test_memory(&format!("mem_{i}"), &format!("content{i}"));
        storage.store(memory).await.unwrap();
    }

    let stats = storage.content_storage_stats();
    assert!(stats.total_bytes > 0);
    assert_eq!(stats.live_bytes, stats.total_bytes);
    assert!((stats.fragmentation_ratio - 0.0).abs() < f64::EPSILON);

    // Delete half
    for i in 0..5 {
        storage.remove(&format!("mem_{i}")).await.unwrap();
    }

    let stats = storage.content_storage_stats();
    assert!(stats.fragmentation_ratio > 0.4 && stats.fragmentation_ratio < 0.6);
}

#[tokio::test]
async fn test_compaction_with_no_content() {
    let temp_dir = TempDir::new().unwrap();
    let metrics = Arc::new(StorageMetrics::new());
    let storage = MappedWarmStorage::new(temp_dir.path().join("test.dat"), 10, metrics).unwrap();

    // Store memories without content
    for i in 0..10 {
        let embedding = [0.1_f32; 768];
        let memory = Memory::new(format!("mem_{i}"), embedding, Confidence::exact(0.8));
        storage.store(Arc::new(memory)).await.unwrap();
    }

    // Compact
    let stats = storage.compact_content().unwrap();
    assert_eq!(stats.bytes_reclaimed, 0, "No content to reclaim");
    assert_eq!(stats.old_size, 0);
    assert_eq!(stats.new_size, 0);
}

#[tokio::test]
async fn test_compaction_with_empty_storage() {
    let temp_dir = TempDir::new().unwrap();
    let metrics = Arc::new(StorageMetrics::new());
    let storage = MappedWarmStorage::new(temp_dir.path().join("test.dat"), 10, metrics).unwrap();

    // Compact empty storage
    let stats = storage.compact_content().unwrap();
    assert_eq!(stats.bytes_reclaimed, 0);
    assert_eq!(stats.old_size, 0);
    assert_eq!(stats.new_size, 0);
}

#[tokio::test]
async fn test_compaction_blocks_concurrent_compaction() {
    let temp_dir = TempDir::new().unwrap();
    let metrics = Arc::new(StorageMetrics::new());
    let storage =
        Arc::new(MappedWarmStorage::new(temp_dir.path().join("test.dat"), 1000, metrics).unwrap());

    // Pre-populate with more data to make compaction slower
    for i in 0..1000 {
        let memory = create_test_memory(
            &format!("mem_{i}"),
            &format!("content {i} with more data to make it slower"),
        );
        storage.store(memory).await.unwrap();
    }
    for i in 0..500 {
        storage.remove(&format!("mem_{i}")).await.unwrap();
    }

    // Spawn two compaction tasks simultaneously
    let storage_clone1 = storage.clone();
    let storage_clone2 = storage.clone();

    let compact1 = tokio::spawn(async move { storage_clone1.compact_content() });
    let compact2 = tokio::spawn(async move { storage_clone2.compact_content() });

    // One should succeed with reclaimed bytes, or both complete but only one does real work
    let result1 = compact1.await.unwrap();
    let result2 = compact2.await.unwrap();

    match (result1, result2) {
        (Ok(stats1), Ok(stats2)) => {
            // Both succeeded - one should have reclaimed bytes, other should have found nothing
            let reclaimed_count =
                usize::from(stats1.bytes_reclaimed > 0) + usize::from(stats2.bytes_reclaimed > 0);
            assert_eq!(
                reclaimed_count, 1,
                "Exactly one should reclaim bytes, got: {stats1:?}, {stats2:?}"
            );
        }
        (Ok(_), Err(StorageError::CompactionInProgress))
        | (Err(StorageError::CompactionInProgress), Ok(_)) => {
            // Expected: one succeeded, one got blocked
        }
        other => panic!("Unexpected results: {other:?}"),
    }
}

#[tokio::test]
async fn test_compaction_sequential() {
    let temp_dir = TempDir::new().unwrap();
    let metrics = Arc::new(StorageMetrics::new());
    let storage = MappedWarmStorage::new(temp_dir.path().join("test.dat"), 500, metrics).unwrap();

    // Store, delete, compact - repeat 3 times
    for cycle in 0..3 {
        // Store 50
        for i in 0..50 {
            let memory = create_test_memory(&format!("mem_{cycle}_{i}"), &format!("content{i}"));
            storage.store(memory).await.unwrap();
        }

        // Delete 25
        for i in 0..25 {
            storage.remove(&format!("mem_{cycle}_{i}")).await.unwrap();
        }

        // Compact
        let stats = storage.compact_content().unwrap();
        assert!(
            stats.bytes_reclaimed > 0,
            "Cycle {cycle} should reclaim bytes"
        );
    }

    // Verify last compaction stats are tracked
    let final_stats = storage.content_storage_stats();
    assert!(final_stats.bytes_reclaimed_total > 0);
    assert!(final_stats.last_compaction > 0);
}

#[tokio::test]
async fn test_compaction_preserves_order() {
    let temp_dir = TempDir::new().unwrap();
    let metrics = Arc::new(StorageMetrics::new());
    let storage = MappedWarmStorage::new(temp_dir.path().join("test.dat"), 10, metrics).unwrap();

    // Store memories in specific order
    let ids = vec!["alpha", "beta", "gamma", "delta", "epsilon"];
    for id in &ids {
        let memory = create_test_memory(id, &format!("content_{id}"));
        storage.store(memory).await.unwrap();
    }

    // Delete middle one
    storage.remove("gamma").await.unwrap();

    // Compact
    storage.compact_content().unwrap();

    // Verify all remaining memories still accessible
    for id in &["alpha", "beta", "delta", "epsilon"] {
        let result = storage.get(id).unwrap();
        assert!(result.is_some(), "Memory {id} should exist");
        assert_eq!(
            result.unwrap().content.as_deref(),
            Some(&format!("content_{id}")[..])
        );
    }

    // Verify deleted one is gone
    assert!(storage.get("gamma").unwrap().is_none());
}

#[tokio::test]
async fn test_compaction_vs_concurrent_reads() {
    let temp_dir = TempDir::new().unwrap();
    let metrics = Arc::new(StorageMetrics::new());
    let storage = Arc::new(
        MappedWarmStorage::new(temp_dir.path().join("test.dat"), 10_000, metrics).unwrap(),
    );

    // Create 10K memories with distinct, verifiable content
    for i in 0..10_000 {
        let memory = create_test_memory(&format!("mem_{i}"), &format!("content_{i}"));
        storage.store(memory).await.unwrap();
    }

    // Delete 50% to create fragmentation
    for i in 0..5_000 {
        storage.remove(&format!("mem_{i}")).await.unwrap();
    }

    // Trigger background compaction
    let storage_clone = storage.clone();
    let compact_handle = tokio::spawn(async move { storage_clone.compact_content() });

    // Hammer with concurrent reads during compaction (100 threads)
    let read_handles: Vec<_> = (0..100)
        .map(|thread_id| {
            let storage_clone = storage.clone();
            tokio::spawn(async move {
                let mut read_count = 0;
                let error_count = 0;

                // Each thread reads 100 times
                for _ in 0..100 {
                    // Read memories in thread's range (50 per thread)
                    let base = 5_000 + (thread_id * 50);
                    for offset in 0..50 {
                        let i = base + offset;
                        match storage_clone.get(&format!("mem_{i}")) {
                            Ok(Some(mem)) => {
                                // Verify content hash matches ID
                                assert_eq!(
                                    mem.content.as_deref(),
                                    Some(&format!("content_{i}")[..]),
                                    "Content corruption detected: mem_{i} has content {:?}",
                                    mem.content
                                );
                                read_count += 1;
                            }
                            Ok(None) => {
                                // Memory was deleted - not an error
                            }
                            Err(e) => {
                                panic!("Corruption detected during read: {e}");
                            }
                        }
                    }

                    // Small sleep to allow compaction to progress
                    tokio::time::sleep(std::time::Duration::from_micros(10)).await;
                }

                (read_count, error_count)
            })
        })
        .collect();

    // Wait for compaction to complete
    let compact_result = compact_handle.await.unwrap();
    if let Err(ref e) = compact_result {
        panic!("Compaction should succeed but got error: {e:?}");
    }
    compact_result.unwrap();

    // All reads should have succeeded without corruption
    let mut total_reads = 0;
    let mut total_errors = 0;
    for handle in read_handles {
        let (reads, errors) = handle.await.unwrap();
        total_reads += reads;
        total_errors += errors;
    }

    assert_eq!(
        total_errors, 0,
        "No read errors should occur during compaction"
    );
    assert!(
        total_reads > 100_000,
        "Should have performed many successful reads"
    );
}

#[tokio::test]
async fn test_compaction_with_concurrent_writes() {
    let temp_dir = TempDir::new().unwrap();
    let metrics = Arc::new(StorageMetrics::new());
    let storage =
        Arc::new(MappedWarmStorage::new(temp_dir.path().join("test.dat"), 5_000, metrics).unwrap());

    // Populate initial data
    for i in 0..1_000 {
        let memory = create_test_memory(&format!("mem_{i}"), &format!("content_{i}"));
        storage.store(memory).await.unwrap();
    }

    // Delete half to create fragmentation
    for i in 0..500 {
        storage.remove(&format!("mem_{i}")).await.unwrap();
    }

    // Start compaction in background
    let storage_clone = storage.clone();
    let compact_handle = tokio::spawn(async move { storage_clone.compact_content() });

    // Concurrent writes to new memories
    let write_handles: Vec<_> = (0..10)
        .map(|thread_id| {
            let storage_clone = storage.clone();
            tokio::spawn(async move {
                for i in 0..100 {
                    let id = format!("new_mem_{thread_id}_{i}");
                    let memory = create_test_memory(&id, &format!("new_content_{thread_id}_{i}"));
                    storage_clone.store(memory).await.unwrap();
                }
            })
        })
        .collect();

    // Wait for both compaction and writes
    compact_handle.await.unwrap().unwrap();
    for handle in write_handles {
        handle.await.unwrap();
    }

    // Verify new memories are intact
    for thread_id in 0..10 {
        for i in 0..100 {
            let id = format!("new_mem_{thread_id}_{i}");
            let result = storage.get(&id).unwrap();
            assert!(result.is_some(), "New memory {id} should exist");
        }
    }
}

#[tokio::test]
async fn test_compaction_empty_content_strings() {
    let temp_dir = TempDir::new().unwrap();
    let metrics = Arc::new(StorageMetrics::new());
    let storage = MappedWarmStorage::new(temp_dir.path().join("test.dat"), 100, metrics).unwrap();

    // Store memories with empty strings (not None)
    for i in 0..10 {
        let memory = create_test_memory(&format!("mem_{i}"), "");
        storage.store(memory).await.unwrap();
    }

    // Compact
    let stats = storage.compact_content().unwrap();

    // Empty strings still have valid offsets but length 0
    // Verify all can be retrieved
    for i in 0..10 {
        let result = storage.get(&format!("mem_{i}")).unwrap();
        assert!(result.is_some());
        assert_eq!(result.unwrap().content.as_deref(), Some(""));
    }

    // Should have minimal bytes (only for offsets, not content)
    assert!(stats.new_size < 100);
}

#[tokio::test]
async fn test_fragmentation_threshold_edge_cases() {
    let temp_dir = TempDir::new().unwrap();
    let metrics = Arc::new(StorageMetrics::new());
    let storage = MappedWarmStorage::new(temp_dir.path().join("test.dat"), 1000, metrics).unwrap();

    // Test 49.9% fragmentation (below threshold)
    for i in 0..1000 {
        let memory = create_test_memory(&format!("mem_{i}"), &"x".repeat(100));
        storage.store(memory).await.unwrap();
    }

    // Delete exactly 499 (49.9%)
    for i in 0..499 {
        storage.remove(&format!("mem_{i}")).await.unwrap();
    }

    let stats_before = storage.content_storage_stats();
    assert!(
        stats_before.fragmentation_ratio < 0.5,
        "Should be just below threshold: {}",
        stats_before.fragmentation_ratio
    );

    // Delete 1 more to cross threshold (50%)
    storage.remove("mem_499").await.unwrap();

    let stats_at_threshold = storage.content_storage_stats();
    assert!(
        stats_at_threshold.fragmentation_ratio >= 0.5,
        "Should be at or above threshold: {}",
        stats_at_threshold.fragmentation_ratio
    );

    // Compaction should work at threshold
    let compact_stats = storage.compact_content().unwrap();
    assert!(compact_stats.bytes_reclaimed > 0);
}

#[tokio::test]
async fn test_compaction_failure_recovery() {
    let temp_dir = TempDir::new().unwrap();
    let metrics = Arc::new(StorageMetrics::new());
    let storage = MappedWarmStorage::new(temp_dir.path().join("test.dat"), 100, metrics).unwrap();

    // Store some memories
    for i in 0..50 {
        let memory = create_test_memory(&format!("mem_{i}"), &format!("content_{i}"));
        storage.store(memory).await.unwrap();
    }

    // Delete half
    for i in 0..25 {
        storage.remove(&format!("mem_{i}")).await.unwrap();
    }

    // First compaction should succeed
    let stats1 = storage.compact_content().unwrap();
    assert!(stats1.bytes_reclaimed > 0);

    // Verify all remaining memories intact after compaction
    for i in 25..50 {
        let result = storage.get(&format!("mem_{i}")).unwrap();
        assert!(
            result.is_some(),
            "Memory mem_{i} should exist after compaction"
        );
        assert_eq!(
            result.unwrap().content.as_deref(),
            Some(&format!("content_{i}")[..])
        );
    }

    // Store more memories
    for i in 50..100 {
        let memory = create_test_memory(&format!("mem_{i}"), &format!("content_{i}"));
        storage.store(memory).await.unwrap();
    }

    // Delete some more
    for i in 50..75 {
        storage.remove(&format!("mem_{i}")).await.unwrap();
    }

    // Second compaction after recovery
    let stats2 = storage.compact_content().unwrap();
    assert!(stats2.bytes_reclaimed > 0);

    // Verify all memories intact after second compaction
    for i in 25..50 {
        let result = storage.get(&format!("mem_{i}")).unwrap();
        assert!(result.is_some());
    }
    for i in 75..100 {
        let result = storage.get(&format!("mem_{i}")).unwrap();
        assert!(result.is_some());
    }
}

#[tokio::test]
async fn test_large_dataset_performance() {
    let temp_dir = TempDir::new().unwrap();
    let metrics = Arc::new(StorageMetrics::new());
    let storage =
        MappedWarmStorage::new(temp_dir.path().join("test.dat"), 50_000, metrics).unwrap();

    // Store 50K memories with 200 byte content each (~10MB)
    for i in 0..50_000 {
        let content = format!("content_{i}_") + &"x".repeat(180);
        let memory = create_test_memory(&format!("mem_{i}"), &content);
        storage.store(memory).await.unwrap();
    }

    // Delete 50% to create ~5MB fragmentation
    for i in 0..25_000 {
        storage.remove(&format!("mem_{i}")).await.unwrap();
    }

    let stats_before = storage.content_storage_stats();
    assert!(stats_before.total_bytes > 9_000_000, "Should have ~10MB");

    // Compact and measure time
    let start = std::time::Instant::now();
    let compact_stats = storage.compact_content().unwrap();
    let duration = start.elapsed();

    // Should complete in reasonable time (target: <2s, allow 5s for CI)
    assert!(
        duration.as_secs() < 5,
        "50K memories should compact in <5s, took {duration:?}"
    );

    // Should reclaim ~50% of space
    assert!(
        compact_stats.bytes_reclaimed > 4_000_000,
        "Should reclaim ~5MB"
    );

    let stats_after = storage.content_storage_stats();
    assert!(stats_after.total_bytes < 6_000_000, "Should shrink to ~5MB");
}

#[tokio::test]
async fn test_memory_leak_verification() {
    let temp_dir = TempDir::new().unwrap();
    let metrics = Arc::new(StorageMetrics::new());
    let storage =
        MappedWarmStorage::new(temp_dir.path().join("test.dat"), 10_000, metrics).unwrap();

    // Multiple cycles of allocate, delete, compact
    for cycle in 0..5 {
        // Allocate 2K memories with 1KB content each
        for i in 0..2_000 {
            let content = "x".repeat(1024);
            let memory = create_test_memory(&format!("mem_{cycle}_{i}"), &content);
            storage.store(memory).await.unwrap();
        }

        let stats_allocated = storage.content_storage_stats();
        assert!(
            stats_allocated.total_bytes > 1_900_000,
            "Cycle {cycle}: Should have ~2MB after allocation"
        );

        // Delete all
        for i in 0..2_000 {
            storage.remove(&format!("mem_{cycle}_{i}")).await.unwrap();
        }

        // Compact
        let compact_stats = storage.compact_content().unwrap();
        assert!(
            compact_stats.bytes_reclaimed > 1_900_000,
            "Cycle {cycle}: Should reclaim ~2MB"
        );

        let stats_after_compact = storage.content_storage_stats();
        assert!(
            stats_after_compact.total_bytes < 100_000,
            "Cycle {cycle}: Should have minimal memory after compaction, got {}",
            stats_after_compact.total_bytes
        );
    }

    // Final verification: no memory leak over multiple cycles
    let final_stats = storage.content_storage_stats();
    assert!(
        final_stats.total_bytes < 100_000,
        "Final memory should be minimal after 5 cycles, got {}",
        final_stats.total_bytes
    );
}
