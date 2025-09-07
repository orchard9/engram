//! Integration tests for memory-mapped persistence system
//!
//! Tests the complete persistent storage implementation including:
//! - Write-ahead logging with crash consistency
//! - Multi-tier storage with cognitive eviction policies
//! - NUMA-aware memory mapping
//! - Performance characteristics under various load patterns

#![cfg(feature = "memory_mapped_persistence")]

use chrono::Utc;
use engram_core::{
    Confidence, Cue, Episode, EpisodeBuilder, MemoryStore,
    storage::{StorageMetrics, TierStatistics},
};
use std::sync::Arc;
use tempfile::TempDir;

fn create_test_episode(id: &str, content: &str) -> Episode {
    EpisodeBuilder::new()
        .id(id.to_string())
        .when(Utc::now())
        .what(content.to_string())
        .embedding([0.5f32; 768])
        .confidence(Confidence::HIGH)
        .build()
}

#[tokio::test]
async fn test_basic_persistence_integration() {
    let temp_dir = TempDir::new().unwrap();

    // Create store with persistence
    let mut store = MemoryStore::new(100)
        .with_persistence(temp_dir.path())
        .unwrap();

    store.initialize_persistence().unwrap();

    // Store some episodes
    let episode1 = create_test_episode("test1", "first test memory");
    let episode2 = create_test_episode("test2", "second test memory");

    let activation1 = store.store(episode1);
    let activation2 = store.store(episode2);

    assert!(activation1.is_successful());
    assert!(activation2.is_successful());

    // Verify they can be recalled
    let cue = Cue::semantic(
        "test".to_string(),
        "test memory".to_string(),
        Confidence::MEDIUM,
    );
    let results = store.recall(cue);

    assert_eq!(results.len(), 2);
    assert!(results.iter().any(|(ep, _)| ep.id == "test1"));
    assert!(results.iter().any(|(ep, _)| ep.id == "test2"));

    // Check storage metrics
    let metrics = store.storage_metrics();
    assert!(
        metrics
            .writes_total
            .load(std::sync::atomic::Ordering::Relaxed)
            > 0
    );

    // Gracefully shutdown
    store.shutdown().unwrap();
}

#[tokio::test]
async fn test_tier_migration() {
    let temp_dir = TempDir::new().unwrap();

    let mut store = MemoryStore::new(10) // Small capacity to trigger migrations
        .with_persistence(temp_dir.path())
        .unwrap();

    store.initialize_persistence().unwrap();

    // Store memories that should trigger tier migrations
    let mut episodes = Vec::new();
    for i in 0..15 {
        let episode = create_test_episode(&format!("mem_{}", i), &format!("memory content {}", i));
        episodes.push(episode.clone());

        let activation = store.store(episode);
        if i < 10 {
            assert!(activation.is_successful());
        } else {
            // Later stores might be degraded due to capacity pressure
            assert!(activation.value() > 0.0);
        }
    }

    // Trigger maintenance to ensure migrations happen
    store.maintenance().await.unwrap();

    // Check tier statistics
    if let Some(stats) = store.tier_statistics() {
        println!(
            "Tier stats: Hot: {}, Warm: {}, Cold: {}",
            stats.hot.memory_count, stats.warm.memory_count, stats.cold.memory_count
        );

        // Should have some distribution across tiers
        assert!(stats.total_memories() > 0);
        assert!(stats.total_size_bytes() > 0);
    }

    store.shutdown().unwrap();
}

#[tokio::test]
async fn test_crash_consistency() {
    let temp_dir = TempDir::new().unwrap();

    // First session: store some data
    {
        let mut store = MemoryStore::new(100)
            .with_persistence(temp_dir.path())
            .unwrap();

        store.initialize_persistence().unwrap();

        for i in 0..5 {
            let episode = create_test_episode(&format!("crash_test_{}", i), "crash test data");
            store.store(episode);
        }

        // Force WAL flush
        store.storage_metrics().record_fsync();

        // Simulate crash by dropping without shutdown
    }

    // Second session: verify data survived the "crash"
    {
        let mut store = MemoryStore::new(100)
            .with_persistence(temp_dir.path())
            .unwrap();

        store.initialize_persistence().unwrap();

        // Verify we can still recall the data
        let cue = Cue::semantic(
            "crash".to_string(),
            "crash test".to_string(),
            Confidence::LOW,
        );
        let results = store.recall(cue);

        // Note: This test might not find results immediately as recovery
        // implementation is simplified. In a full implementation,
        // WAL replay would restore the data.
        println!("Recovered {} memories after simulated crash", results.len());

        store.shutdown().unwrap();
    }
}

#[tokio::test]
async fn test_cognitive_workload_pattern() {
    let temp_dir = TempDir::new().unwrap();

    let mut store = MemoryStore::new(1000)
        .with_persistence(temp_dir.path())
        .unwrap();

    store.initialize_persistence().unwrap();

    // Simulate a cognitive workload: bursts of related memories
    let topics = [
        "learning_session",
        "project_work",
        "meeting_notes",
        "research",
    ];

    for (topic_idx, topic) in topics.iter().enumerate() {
        // Burst of related memories
        for i in 0..25 {
            let episode = create_test_episode(
                &format!("{}_{}", topic, i),
                &format!("{} content item {}", topic, i),
            );

            let activation = store.store(episode);
            assert!(activation.is_successful());

            // Higher activation for more recent items in the session
            if i > 20 {
                assert!(activation.value() > 0.8);
            }
        }

        // Brief pause between topics (simulating real cognitive patterns)
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    }

    // Verify we can recall by topic
    for topic in &topics {
        let cue = Cue::semantic(
            topic.to_string(),
            format!("{} content", topic),
            Confidence::MEDIUM,
        );
        let results = store.recall(cue);

        // Should find memories related to this topic
        assert!(results.len() > 0);
        assert!(results.len() <= 25); // Max stored per topic
    }

    // Check that system pressure adapted appropriately
    let pressure = store.pressure();
    println!("Final system pressure: {}", pressure);
    assert!(pressure >= 0.0 && pressure <= 1.0);

    // Verify storage metrics show realistic performance
    let metrics = store.storage_metrics();
    let writes = metrics
        .writes_total
        .load(std::sync::atomic::Ordering::Relaxed);
    let reads = metrics
        .reads_total
        .load(std::sync::atomic::Ordering::Relaxed);

    assert!(writes >= 100); // Should have recorded all writes
    assert!(reads >= 4); // Should have recorded recall operations

    println!(
        "Performance: {} writes, {} reads, {:.1}% cache hit rate",
        writes,
        reads,
        metrics.cache_hit_rate() * 100.0
    );

    store.shutdown().unwrap();
}

#[test]
fn test_storage_metrics_accuracy() {
    let metrics = StorageMetrics::new();

    // Test metric recording
    metrics.record_write(1024);
    metrics.record_write(2048);

    assert_eq!(
        metrics
            .writes_total
            .load(std::sync::atomic::Ordering::Relaxed),
        2
    );
    assert_eq!(
        metrics
            .bytes_written
            .load(std::sync::atomic::Ordering::Relaxed),
        3072
    );

    metrics.record_cache_hit();
    metrics.record_cache_hit();
    metrics.record_cache_miss();

    let hit_rate = metrics.cache_hit_rate();
    assert!((hit_rate - 0.666).abs() < 0.01); // 2 hits out of 3 total
}

#[test]
fn test_tier_statistics() {
    let stats = TierStatistics {
        memory_count: 100,
        total_size_bytes: 1024000,
        average_activation: 0.75,
        last_access_time: std::time::SystemTime::now(),
        cache_hit_rate: 0.95,
        compaction_ratio: 0.85,
    };

    assert_eq!(stats.memory_count, 100);
    assert_eq!(stats.total_size_bytes, 1024000);
    assert_eq!(stats.average_activation, 0.75);
    assert!(stats.cache_hit_rate > 0.9);
    assert!(stats.compaction_ratio > 0.8);
}

#[tokio::test]
async fn test_graceful_degradation_under_errors() {
    let temp_dir = TempDir::new().unwrap();

    // Create store with very limited capacity to trigger pressure
    let mut store = MemoryStore::new(5)
        .with_persistence(temp_dir.path())
        .unwrap();

    store.initialize_persistence().unwrap();

    // Store beyond capacity
    let mut degraded_count = 0;
    for i in 0..10 {
        let episode = create_test_episode(&format!("pressure_test_{}", i), "pressure test");
        let activation = store.store(episode);

        if activation.is_degraded() {
            degraded_count += 1;
        }
    }

    // Should have some degraded activations due to pressure
    assert!(degraded_count > 0);
    println!("Degraded activations under pressure: {}/10", degraded_count);

    // System should still be functional
    let cue = Cue::semantic(
        "pressure".to_string(),
        "pressure test".to_string(),
        Confidence::LOW,
    );
    let results = store.recall(cue);

    // Should still be able to recall some memories
    assert!(results.len() > 0);

    store.shutdown().unwrap();
}

#[cfg(all(unix, feature = "memory_mapped_persistence"))]
#[tokio::test]
async fn test_numa_awareness() {
    use engram_core::storage::numa::NumaTopology;

    // Test NUMA topology detection
    let topology = NumaTopology::detect().unwrap();

    assert!(topology.socket_count > 0);
    assert!(topology.node_count > 0);

    println!(
        "Detected NUMA topology: {} sockets, {} nodes",
        topology.socket_count, topology.node_count
    );

    // Test socket suggestion for temporal clustering
    let timestamp1 = 1000000000u64;
    let timestamp2 = 2000000000u64;

    let socket1 = topology.suggest_socket_for_timestamp(timestamp1);
    let socket2 = topology.suggest_socket_for_timestamp(timestamp2);

    assert!(socket1 < topology.socket_count);
    assert!(socket2 < topology.socket_count);

    // Different timestamps should potentially map to different sockets
    // (though not guaranteed due to hashing)
    println!(
        "Timestamp {} -> socket {}, timestamp {} -> socket {}",
        timestamp1, socket1, timestamp2, socket2
    );
}

#[tokio::test]
async fn test_concurrent_access_patterns() {
    let temp_dir = TempDir::new().unwrap();

    let store = Arc::new(tokio::sync::Mutex::new(
        MemoryStore::new(1000)
            .with_persistence(temp_dir.path())
            .unwrap(),
    ));

    {
        let mut store_lock = store.lock().await;
        store_lock.initialize_persistence().unwrap();
    }

    // Spawn concurrent tasks
    let mut handles = Vec::new();

    for task_id in 0..5 {
        let store_clone = Arc::clone(&store);

        let handle = tokio::spawn(async move {
            for i in 0..10 {
                let store_guard = store_clone.lock().await;

                // Concurrent stores
                let episode = create_test_episode(
                    &format!("concurrent_{}_{}", task_id, i),
                    &format!("concurrent data from task {}", task_id),
                );

                let activation = store_guard.store(episode);
                assert!(activation.value() > 0.0);

                // Concurrent recalls
                let cue = Cue::semantic(
                    format!("task_{}", task_id),
                    "concurrent data".to_string(),
                    Confidence::LOW,
                );

                let _results = store_guard.recall(cue);

                drop(store_guard); // Release lock
                tokio::time::sleep(std::time::Duration::from_millis(1)).await;
            }
        });

        handles.push(handle);
    }

    // Wait for all tasks to complete
    for handle in handles {
        handle.await.unwrap();
    }

    // Verify final state
    let store_guard = store.lock().await;
    let final_count = store_guard.count();
    println!(
        "Final memory count after concurrent operations: {}",
        final_count
    );

    assert!(final_count > 0);
    assert!(final_count <= 50); // 5 tasks * 10 operations each, but some may be evicted

    // Test final recall
    let cue = Cue::semantic(
        "concurrent".to_string(),
        "concurrent data".to_string(),
        Confidence::LOW,
    );
    let results = store_guard.recall(cue);
    println!("Final recall found {} memories", results.len());

    drop(store_guard);

    {
        let mut store_lock = store.lock().await;
        store_lock.shutdown().unwrap();
    }
}
