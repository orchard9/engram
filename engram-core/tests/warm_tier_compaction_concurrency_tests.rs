//! Concurrency tests for warm tier compaction
//!
//! Tests verify correct behavior under concurrent access:
//! - get() during compaction
//! - store() during compaction
//! - remove() during compaction
//! - Multiple concurrent operations

use engram_core::{
    Confidence, Memory,
    storage::{MappedWarmStorage, StorageMetrics, StorageTierBackend},
};
use std::sync::Arc;
use tempfile::TempDir;

fn create_test_memory(id: &str, content: &str) -> Arc<Memory> {
    let embedding = [0.1_f32; 768];
    let mut memory = Memory::new(id.to_string(), embedding, Confidence::exact(0.8));
    memory.content = Some(content.to_string());
    Arc::new(memory)
}

#[tokio::test]
async fn test_concurrent_get_during_compaction() {
    let temp_dir = TempDir::new().unwrap();
    let metrics = Arc::new(StorageMetrics::new());
    let storage =
        Arc::new(MappedWarmStorage::new(temp_dir.path().join("test.dat"), 100, metrics).unwrap());

    // Store 100
    for i in 0..100 {
        let memory = create_test_memory(&format!("mem_{i}"), &format!("content {i}"));
        storage.store(memory).await.unwrap();
    }

    // Delete 50
    for i in (0..100).step_by(2) {
        storage.remove(&format!("mem_{i}")).await.unwrap();
    }

    // Spawn compaction task
    let storage_clone = storage.clone();
    let compact_task = tokio::spawn(async move { storage_clone.compact_content() });

    // Concurrently read memories
    let mut handles = vec![];
    for i in (1..100).step_by(2) {
        let storage_clone = storage.clone();
        let handle = tokio::spawn(async move { storage_clone.get(&format!("mem_{i}")) });
        handles.push(handle);
    }

    // Wait for all
    compact_task.await.unwrap().unwrap();
    for handle in handles {
        let result = handle.await.unwrap().unwrap();
        assert!(result.is_some(), "get() should always succeed");
    }
}

#[tokio::test]
async fn test_concurrent_store_during_compaction() {
    let temp_dir = TempDir::new().unwrap();
    let metrics = Arc::new(StorageMetrics::new());
    let storage =
        Arc::new(MappedWarmStorage::new(temp_dir.path().join("test.dat"), 1000, metrics).unwrap());

    // Pre-populate
    for i in 0..500 {
        storage
            .store(create_test_memory(&format!("mem_{i}"), "content"))
            .await
            .unwrap();
    }
    for i in 0..250 {
        storage.remove(&format!("mem_{i}")).await.unwrap();
    }

    // Spawn compaction
    let storage_clone = storage.clone();
    let compact_task = tokio::spawn(async move { storage_clone.compact_content() });

    // Concurrently store new memories
    let mut handles = vec![];
    for i in 1000..1100 {
        let storage_clone = storage.clone();
        let handle = tokio::spawn(async move {
            storage_clone
                .store(create_test_memory(&format!("mem_{i}"), "new"))
                .await
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
        assert!(storage.get(&format!("mem_{i}")).unwrap().is_some());
    }
}

#[tokio::test]
async fn test_concurrent_remove_during_compaction() {
    let temp_dir = TempDir::new().unwrap();
    let metrics = Arc::new(StorageMetrics::new());
    let storage =
        Arc::new(MappedWarmStorage::new(temp_dir.path().join("test.dat"), 100, metrics).unwrap());

    // Store 100
    for i in 0..100 {
        storage
            .store(create_test_memory(&format!("mem_{i}"), "content"))
            .await
            .unwrap();
    }

    // Delete first 50 to create fragmentation
    for i in 0..50 {
        storage.remove(&format!("mem_{i}")).await.unwrap();
    }

    // Spawn compaction
    let storage_clone = storage.clone();
    let compact_task = tokio::spawn(async move { storage_clone.compact_content() });

    // Concurrently remove more memories
    let mut handles = vec![];
    for i in 50..75 {
        let storage_clone = storage.clone();
        let handle = tokio::spawn(async move { storage_clone.remove(&format!("mem_{i}")).await });
        handles.push(handle);
    }

    // Wait for all
    compact_task.await.unwrap().unwrap();
    for handle in handles {
        handle.await.unwrap().unwrap();
    }

    // Verify correct memories remain
    for i in 0..75 {
        assert!(storage.get(&format!("mem_{i}")).unwrap().is_none());
    }
    for i in 75..100 {
        assert!(storage.get(&format!("mem_{i}")).unwrap().is_some());
    }
}

#[tokio::test]
async fn test_stress_concurrent_operations() {
    let temp_dir = TempDir::new().unwrap();
    let metrics = Arc::new(StorageMetrics::new());
    let storage =
        Arc::new(MappedWarmStorage::new(temp_dir.path().join("test.dat"), 500, metrics).unwrap());

    // Pre-populate with fragmentation
    for i in 0..300 {
        storage
            .store(create_test_memory(
                &format!("mem_{i}"),
                &format!("content{i}"),
            ))
            .await
            .unwrap();
    }
    for i in (0..300).step_by(2) {
        storage.remove(&format!("mem_{i}")).await.unwrap();
    }

    // Spawn compaction
    let storage_clone = storage.clone();
    let compact_task = tokio::spawn(async move { storage_clone.compact_content() });

    // Spawn 10 readers
    let mut reader_handles = vec![];
    for thread_id in 0..10 {
        let storage_clone = storage.clone();
        let handle = tokio::spawn(async move {
            for i in 0..100 {
                let id = format!("mem_{}", (thread_id * 100 + i) % 300);
                let _ = storage_clone.get(&id);
            }
        });
        reader_handles.push(handle);
    }

    // Spawn 5 writers
    let mut writer_handles = vec![];
    for thread_id in 0..5 {
        let storage_clone = storage.clone();
        let handle = tokio::spawn(async move {
            for i in 0..50 {
                let id = format!("new_mem_{thread_id}_{i}");
                let _ = storage_clone
                    .store(create_test_memory(&id, "content"))
                    .await;
            }
        });
        writer_handles.push(handle);
    }

    // Wait for compaction
    compact_task.await.unwrap().unwrap();

    // Wait for all workers
    for handle in reader_handles {
        handle.await.unwrap();
    }
    for handle in writer_handles {
        handle.await.unwrap();
    }

    // Verify storage is in consistent state
    let stats = storage.content_storage_stats();
    assert!(
        stats.fragmentation_ratio < 0.5,
        "Compaction should reduce fragmentation"
    );
}

#[tokio::test]
async fn test_compaction_after_concurrent_modifications() {
    let temp_dir = TempDir::new().unwrap();
    let metrics = Arc::new(StorageMetrics::new());
    let storage =
        Arc::new(MappedWarmStorage::new(temp_dir.path().join("test.dat"), 200, metrics).unwrap());

    // Fill storage with concurrent writes
    let mut handles = vec![];
    for thread_id in 0..10 {
        let storage_clone = storage.clone();
        let handle = tokio::spawn(async move {
            for i in 0..20 {
                let id = format!("mem_{thread_id}_{i}");
                storage_clone
                    .store(create_test_memory(&id, &format!("content{i}")))
                    .await
                    .unwrap();
            }
        });
        handles.push(handle);
    }

    for handle in handles {
        handle.await.unwrap();
    }

    // Delete half concurrently
    let mut handles = vec![];
    for thread_id in 0..10 {
        let storage_clone = storage.clone();
        let handle = tokio::spawn(async move {
            for i in (0..20).step_by(2) {
                let id = format!("mem_{thread_id}_{i}");
                storage_clone.remove(&id).await.unwrap();
            }
        });
        handles.push(handle);
    }

    for handle in handles {
        handle.await.unwrap();
    }

    // Now compact
    let stats = storage.compact_content().unwrap();
    assert!(stats.bytes_reclaimed > 0);

    // Verify remaining memories accessible
    for thread_id in 0..10 {
        for i in (1..20).step_by(2) {
            let id = format!("mem_{thread_id}_{i}");
            let result = storage.get(&id).unwrap();
            assert!(result.is_some(), "Memory {id} should exist");
        }
    }
}

#[tokio::test]
async fn test_repeated_compaction_no_leaks() {
    let temp_dir = TempDir::new().unwrap();
    let metrics = Arc::new(StorageMetrics::new());
    let storage = MappedWarmStorage::new(temp_dir.path().join("leak.dat"), 1000, metrics).unwrap();

    for cycle in 0..10 {
        // Store 100
        for i in 0..100 {
            let memory = create_test_memory(&format!("mem_{cycle}_{i}"), "content");
            storage.store(memory).await.unwrap();
        }

        // Delete 50
        for i in 0..50 {
            storage.remove(&format!("mem_{cycle}_{i}")).await.unwrap();
        }

        // Compact
        storage.compact_content().unwrap();

        // Verify storage size is reasonable
        let stats = storage.content_storage_stats();
        assert!(
            stats.total_bytes < 10_000_000,
            "Cycle {cycle}: Memory leak detected, size={}MB",
            stats.total_bytes / 1_000_000
        );
    }

    // Verify cumulative bytes reclaimed is tracked
    let final_stats = storage.content_storage_stats();
    assert!(final_stats.bytes_reclaimed_total > 0);
}
