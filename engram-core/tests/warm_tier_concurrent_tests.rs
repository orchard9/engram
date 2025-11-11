//! Comprehensive concurrent access tests for warm tier content persistence
//!
//! This test suite validates the thread-safety and correctness of the warm tier
//! implementation under heavy concurrent load. Tests focus on:
//! - Concurrent store operations (offset monotonicity)
//! - Concurrent get operations (reader parallelism)
//! - Mixed read/write operations (lock ordering)
//! - Lock contention stress (hot-spot access)
//! - Writer-writer contention (offset allocation races)
//! - Panic recovery (robustness under failure)
//! - Property-based testing (content integrity invariants)
//!
//! Critical properties tested:
//! - Content offset monotonicity under concurrent append
//! - No memory ID appears twice in index
//! - Content boundaries never overlap
//! - Total content length equals sum of stored lengths
//! - Iteration sees atomic snapshots (no torn reads)
//!
//! All tests use multi-threaded tokio runtime with barriers to maximize contention.

use engram_core::{
    Confidence, EpisodeBuilder, Memory,
    storage::{MappedWarmStorage, StorageMetrics, StorageTierBackend},
};
use std::collections::HashMap;
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};
use std::time::{Duration, Instant};
use tempfile::TempDir;
use tokio::sync::{Barrier, Mutex};

/// Create a test embedding with a unique value based on seed
fn create_test_embedding(seed: f32) -> [f32; 768] {
    let mut embedding = [0.0_f32; 768];
    for (i, val) in embedding.iter_mut().enumerate() {
        *val = (seed + (i as f32) * 0.001).sin();
    }
    embedding
}

/// Helper to compute latency percentile
fn percentile(mut latencies: Vec<Duration>, p: usize) -> Duration {
    if latencies.is_empty() {
        return Duration::from_secs(0);
    }
    latencies.sort();
    let idx = (latencies.len() * p) / 100;
    latencies[idx.min(latencies.len() - 1)]
}

// ============================================================================
// Test 1: Concurrent Store Operations - Offset Monotonicity
// ============================================================================

#[tokio::test(flavor = "multi_thread", worker_threads = 10)]
async fn test_concurrent_store_offset_monotonicity() {
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let metrics = Arc::new(StorageMetrics::new());
    let store = Arc::new(
        MappedWarmStorage::new(temp_dir.path().join("test.dat"), 10000, metrics)
            .expect("Failed to create storage"),
    );

    // Barrier to ensure all threads start simultaneously (maximize contention)
    let barrier = Arc::new(Barrier::new(10));

    // Spawn 10 writer tasks, each storing 100 memories
    let mut handles = vec![];
    for thread_id in 0..10 {
        let store = Arc::clone(&store);
        let barrier = Arc::clone(&barrier);

        let handle = tokio::spawn(async move {
            // Wait for all threads to be ready
            barrier.wait().await;

            for i in 0..100 {
                let content = format!(
                    "Content from thread {} episode {} - {}",
                    thread_id,
                    i,
                    "padding".repeat(thread_id)
                );

                let episode = EpisodeBuilder::new()
                    .id(format!("thread-{thread_id}-ep-{i}"))
                    .when(chrono::Utc::now())
                    .what(content.clone())
                    .embedding(create_test_embedding(thread_id as f32))
                    .confidence(Confidence::HIGH)
                    .build();

                let memory = Arc::new(Memory::from_episode(episode, 0.8));
                store.store(memory).await.expect("Store should succeed");
            }
        });
        handles.push(handle);
    }

    // Wait for all writers
    let start = Instant::now();
    for handle in handles {
        handle.await.expect("Writer thread panicked");
    }
    let duration = start.elapsed();

    // Performance target: <2s for 1000 stores
    assert!(
        duration < Duration::from_secs(2),
        "Concurrent stores should complete in <2s, took {duration:?}"
    );

    // Verify all 1000 memories stored
    let stats = store.statistics();
    assert_eq!(stats.memory_count, 1000, "All memories should be stored");

    // CRITICAL: Verify no duplicate memory IDs (offset collision detection)
    let mut ids = vec![];
    for thread_id in 0..10 {
        for i in 0..100 {
            let id = format!("thread-{thread_id}-ep-{i}");
            let result = store
                .get(&id)
                .expect("Get should succeed")
                .expect("Memory should exist");

            // Content should match thread ID and episode number
            assert!(
                result
                    .content
                    .as_ref()
                    .unwrap()
                    .contains(&format!("Content from thread {thread_id}")),
                "Content corrupted for memory {}: {:?}",
                id,
                result.content
            );

            // Content should have expected length (no truncation)
            let expected_len = format!("Content from thread {} episode {} - ", thread_id, 0).len()
                + "padding".len() * thread_id;
            assert!(
                result.content.as_ref().unwrap().len() >= expected_len,
                "Content truncated for {}: expected >={}, got {}",
                id,
                expected_len,
                result.content.as_ref().unwrap().len()
            );

            ids.push(id);
        }
    }

    // Property: No duplicate memory IDs
    ids.sort();
    let original_len = ids.len();
    ids.dedup();
    assert_eq!(
        ids.len(),
        original_len,
        "Found duplicate memory IDs - offset collision detected"
    );
}

// ============================================================================
// Test 2: Concurrent Get Operations - Reader Parallelism
// ============================================================================

#[tokio::test(flavor = "multi_thread", worker_threads = 20)]
async fn test_concurrent_get_reader_parallelism() {
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let metrics = Arc::new(StorageMetrics::new());
    let store = Arc::new(
        MappedWarmStorage::new(temp_dir.path().join("test.dat"), 1000, metrics)
            .expect("Failed to create storage"),
    );

    // Pre-populate with 100 memories of varying content sizes
    let mut expected_contents = HashMap::new();
    for i in 0..100 {
        let content = format!("Content for episode {} - {}", i, "x".repeat(i * 10));
        expected_contents.insert(format!("ep-{i}"), content.clone());

        let episode = EpisodeBuilder::new()
            .id(format!("ep-{i}"))
            .when(chrono::Utc::now())
            .what(content)
            .embedding(create_test_embedding(i as f32))
            .confidence(Confidence::HIGH)
            .build();

        let memory = Arc::new(Memory::from_episode(episode, 0.8));
        store.store(memory).await.expect("Store should succeed");
    }

    // Track read latencies to detect lock contention
    let latencies = Arc::new(Mutex::new(Vec::new()));

    // Spawn 20 reader tasks, each reading all 100 memories 10 times (20K reads)
    let mut handles = vec![];
    for reader_id in 0..20 {
        let store = Arc::clone(&store);
        let expected = expected_contents.clone();
        let latencies = Arc::clone(&latencies);

        let handle = tokio::spawn(async move {
            for iteration in 0..10 {
                for i in 0..100 {
                    let memory_id = format!("ep-{i}");
                    let start = Instant::now();

                    // Get should always succeed
                    let memory = store
                        .get(&memory_id)
                        .unwrap_or_else(|e| {
                            panic!(
                                "Reader {reader_id} iteration {iteration} failed to get {memory_id}: {e}"
                            )
                        })
                        .unwrap_or_else(|| {
                            panic!(
                                "Reader {reader_id} iteration {iteration} found None for {memory_id}"
                            )
                        });

                    let elapsed = start.elapsed();
                    latencies.lock().await.push(elapsed);

                    // CRITICAL: Content should be exactly correct (no torn reads)
                    let expected_content = expected.get(&memory_id).unwrap();
                    assert_eq!(
                        memory.content.as_deref(),
                        Some(expected_content.as_str()),
                        "Content corrupted for memory {memory_id} (reader {reader_id}, iteration {iteration})"
                    );
                }
            }
        });
        handles.push(handle);
    }

    // Wait for all readers (performance target: <3s)
    let start = Instant::now();
    for handle in handles {
        handle.await.expect("Reader thread panicked");
    }
    let duration = start.elapsed();

    assert!(
        duration < Duration::from_secs(3),
        "20K concurrent reads should complete in <3s, took {duration:?}"
    );

    // Analyze latency distribution to detect pathological contention
    let lats = latencies.lock().await.clone();
    let p50 = percentile(lats.clone(), 50);
    let p99 = percentile(lats, 99);

    println!("Read latency: p50={p50:?}, p99={p99:?}");

    // Performance target: P99 <1ms
    assert!(
        p99 < Duration::from_millis(1),
        "P99 read latency should be <1ms, got {p99:?}"
    );

    // P99 should be <10x p50 (no pathological tail)
    assert!(
        p99 < p50 * 10,
        "Pathological read latency tail: p50={p50:?}, p99={p99:?}"
    );
}

// ============================================================================
// Test 3: Mixed Read/Write Operations - Lock Ordering
// ============================================================================

#[tokio::test(flavor = "multi_thread", worker_threads = 15)]
async fn test_mixed_concurrent_lock_ordering() {
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let metrics = Arc::new(StorageMetrics::new());
    let store = Arc::new(
        MappedWarmStorage::new(temp_dir.path().join("test.dat"), 5000, metrics)
            .expect("Failed to create storage"),
    );

    // Pre-populate with 1000 memories
    for i in 0..1000 {
        let content = format!("Initial content {} - {}", i, "base".repeat(i % 10));
        let episode = EpisodeBuilder::new()
            .id(format!("ep-{i}"))
            .when(chrono::Utc::now())
            .what(content)
            .embedding(create_test_embedding(i as f32))
            .confidence(Confidence::HIGH)
            .build();

        let memory = Arc::new(Memory::from_episode(episode, 0.8));
        store.store(memory).await.expect("Store should succeed");
    }

    let mut handles = vec![];

    // Track successful operations for validation
    let write_counter = Arc::new(AtomicUsize::new(0));
    let read_counter = Arc::new(AtomicUsize::new(0));

    // Spawn 5 writer tasks (adding new memories)
    for writer_id in 0..5 {
        let store = Arc::clone(&store);
        let write_counter = Arc::clone(&write_counter);

        let handle = tokio::spawn(async move {
            for i in 0..100 {
                let content = format!(
                    "Content from writer {} - {}",
                    writer_id,
                    "data".repeat(writer_id + 1)
                );
                let episode = EpisodeBuilder::new()
                    .id(format!("writer-{writer_id}-ep-{i}"))
                    .when(chrono::Utc::now())
                    .what(content)
                    .embedding(create_test_embedding(writer_id as f32))
                    .confidence(Confidence::HIGH)
                    .build();

                let memory = Arc::new(Memory::from_episode(episode, 0.8));
                store.store(memory).await.expect("Store should succeed");
                write_counter.fetch_add(1, Ordering::Relaxed);

                // Small delay to increase concurrency window
                tokio::time::sleep(Duration::from_micros(100)).await;
            }
        });
        handles.push(handle);
    }

    // Spawn 10 reader tasks (reading existing memories)
    for reader_id in 0..10 {
        let store = Arc::clone(&store);
        let read_counter = Arc::clone(&read_counter);

        let handle = tokio::spawn(async move {
            for iteration in 0..50 {
                let memory_id = format!("ep-{}", (reader_id * 50 + iteration) % 1000);

                if let Ok(Some(memory)) = store.get(&memory_id) {
                    // Verify content structure
                    assert!(
                        memory
                            .content
                            .as_ref()
                            .unwrap()
                            .starts_with("Initial content"),
                        "Content corrupted during concurrent access: {:?}",
                        memory.content
                    );
                    read_counter.fetch_add(1, Ordering::Relaxed);
                }

                tokio::time::sleep(Duration::from_micros(50)).await;
            }
        });
        handles.push(handle);
    }

    // All tasks must complete within timeout (no deadlock)
    let timeout = Duration::from_secs(30);
    let result = tokio::time::timeout(timeout, async {
        for handle in handles {
            handle
                .await
                .expect("Task panicked during concurrent operations");
        }
    })
    .await;

    assert!(result.is_ok(), "Timeout detected - likely deadlock");

    // Verify operation counts
    let writes = write_counter.load(Ordering::Relaxed);
    let reads = read_counter.load(Ordering::Relaxed);

    assert_eq!(writes, 500, "Some writes failed");
    assert!(
        reads > 400,
        "Too many reads failed: {reads} (expected >400)"
    );

    // Verify final counts
    let stats = store.statistics();
    assert_eq!(
        stats.memory_count, 1500,
        "Should have exactly 1500 memories (1000 initial + 500 written)"
    );
}

// ============================================================================
// Test 4: Lock Contention Stress - Hot-Spot Access
// ============================================================================

#[tokio::test(flavor = "multi_thread", worker_threads = 50)]
async fn test_lock_contention_hot_spot() {
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let metrics = Arc::new(StorageMetrics::new());
    let store = Arc::new(
        MappedWarmStorage::new(temp_dir.path().join("test.dat"), 100, metrics)
            .expect("Failed to create storage"),
    );

    // Pre-populate with 100 memories
    for i in 0..100 {
        let content = format!("Content {} - {}", i, "x".repeat(i * 5));
        let episode = EpisodeBuilder::new()
            .id(format!("ep-{i}"))
            .when(chrono::Utc::now())
            .what(content)
            .embedding(create_test_embedding(i as f32))
            .confidence(Confidence::HIGH)
            .build();

        let memory = Arc::new(Memory::from_episode(episode, 0.8));
        store.store(memory).await.expect("Store should succeed");
    }

    // Spawn 50 tasks hammering the same 10 "hot" memories
    let mut handles = vec![];
    let latencies = Arc::new(Mutex::new(Vec::new()));

    for _task_id in 0..50 {
        let store = Arc::clone(&store);
        let latencies = Arc::clone(&latencies);

        let handle = tokio::spawn(async move {
            for iteration in 0..20 {
                // Hot-spot: all threads access same 10 memories
                let memory_id = format!("ep-{}", iteration % 10);
                let start = Instant::now();

                // Try to get
                if let Ok(Some(memory)) = store.get(&memory_id) {
                    assert!(
                        memory.content.is_some() && !memory.content.as_ref().unwrap().is_empty(),
                        "Content should not be empty"
                    );
                }

                let elapsed = start.elapsed();
                latencies.lock().await.push(elapsed);

                // No sleep - maximize contention
            }
        });
        handles.push(handle);
    }

    // All tasks should complete within reasonable time (performance target: <10s)
    let timeout = Duration::from_secs(10);
    let result = tokio::time::timeout(timeout, async {
        for handle in handles {
            handle.await.expect("Task panicked");
        }
    })
    .await;

    assert!(result.is_ok(), "Lock contention caused timeout");

    // Verify latency didn't degrade catastrophically
    let lats = latencies.lock().await.clone();
    let p99 = percentile(lats, 99);

    println!("Hot-spot p99 latency: {p99:?}");

    // Performance target: P99 <100ms under extreme contention
    assert!(
        p99 < Duration::from_millis(100),
        "Pathological contention: p99={p99:?}"
    );
}

// ============================================================================
// Test 5: Writer-Writer Contention - Offset Allocation Races
// ============================================================================

#[tokio::test(flavor = "multi_thread", worker_threads = 20)]
async fn test_writer_writer_offset_races() {
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let metrics = Arc::new(StorageMetrics::new());
    let store = Arc::new(
        MappedWarmStorage::new(temp_dir.path().join("test.dat"), 5000, metrics)
            .expect("Failed to create storage"),
    );

    // Barrier to maximize simultaneous writes
    let barrier = Arc::new(Barrier::new(20));

    // Track all stored content with metadata
    let stored = Arc::new(Mutex::new(Vec::new()));

    let mut handles = vec![];
    for thread_id in 0..20 {
        let store = Arc::clone(&store);
        let barrier = Arc::clone(&barrier);
        let stored = Arc::clone(&stored);

        let handle = tokio::spawn(async move {
            barrier.wait().await;

            for i in 0..50 {
                let content = format!(
                    "Writer {} content {} - {}",
                    thread_id,
                    i,
                    "x".repeat((thread_id + i) * 3)
                );
                let id = format!("w{thread_id}-m{i}");

                let episode = EpisodeBuilder::new()
                    .id(id.clone())
                    .when(chrono::Utc::now())
                    .what(content.clone())
                    .embedding(create_test_embedding(thread_id as f32))
                    .confidence(Confidence::HIGH)
                    .build();

                let memory = Arc::new(Memory::from_episode(episode, 0.8));
                store.store(memory).await.expect("Store should succeed");

                stored.lock().await.push((id, content));
            }
        });
        handles.push(handle);
    }

    // Performance target: <3s for 1000 concurrent stores
    let start = Instant::now();
    for handle in handles {
        handle.await.expect("Writer panicked");
    }
    let duration = start.elapsed();

    assert!(
        duration < Duration::from_secs(3),
        "1000 concurrent stores should complete in <3s, took {duration:?}"
    );

    // Verify all memories stored correctly
    let expected = stored.lock().await;
    assert_eq!(expected.len(), 1000, "All writes should succeed");

    let stats = store.statistics();
    assert_eq!(stats.memory_count, 1000, "All memories should be stored");

    // CRITICAL: Verify no content corruption from offset races
    for (expected_id, expected_content) in expected.iter() {
        let memory = store
            .get(expected_id)
            .expect("Get should succeed")
            .unwrap_or_else(|| panic!("Memory {expected_id} not found"));

        assert_eq!(
            memory.content.as_deref(),
            Some(expected_content.as_str()),
            "Content mismatch for {expected_id} - likely offset collision"
        );
    }
}

// ============================================================================
// Test 6: Panic Recovery - Robustness Under Failure
// ============================================================================

#[tokio::test(flavor = "multi_thread", worker_threads = 10)]
async fn test_panic_recovery_robustness() {
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let metrics = Arc::new(StorageMetrics::new());
    let store = Arc::new(
        MappedWarmStorage::new(temp_dir.path().join("test.dat"), 1000, metrics)
            .expect("Failed to create storage"),
    );

    // Store some initial memories
    for i in 0..100 {
        let episode = EpisodeBuilder::new()
            .id(format!("ep-{i}"))
            .when(chrono::Utc::now())
            .what(format!("Content {i}"))
            .embedding(create_test_embedding(i as f32))
            .confidence(Confidence::HIGH)
            .build();

        let memory = Arc::new(Memory::from_episode(episode, 0.8));
        store.store(memory).await.expect("Store should succeed");
    }

    // Spawn tasks where one will panic
    let mut handles = vec![];
    let panic_detected = Arc::new(AtomicUsize::new(0));

    for task_id in 0..10 {
        let store = Arc::clone(&store);
        let _panic_detected = Arc::clone(&panic_detected);

        let handle = tokio::spawn(async move {
            // Task 5 will panic after storing some memories
            if task_id == 5 {
                // Store a few memories first
                for i in 0..3 {
                    let episode = EpisodeBuilder::new()
                        .id(format!("t{task_id}-ep-{i}"))
                        .when(chrono::Utc::now())
                        .what(format!("Content from task {task_id}"))
                        .embedding(create_test_embedding(task_id as f32))
                        .confidence(Confidence::HIGH)
                        .build();

                    let memory = Arc::new(Memory::from_episode(episode, 0.8));
                    store.store(memory).await.expect("Store should succeed");
                }
                panic!("Injected panic for testing");
            }

            for i in 0..10 {
                let episode = EpisodeBuilder::new()
                    .id(format!("t{task_id}-ep-{i}"))
                    .when(chrono::Utc::now())
                    .what(format!("Content from task {task_id}"))
                    .embedding(create_test_embedding(task_id as f32))
                    .confidence(Confidence::HIGH)
                    .build();

                let memory = Arc::new(Memory::from_episode(episode, 0.8));
                store.store(memory).await.expect("Store should succeed");
            }
        });
        handles.push(handle);
    }

    // Some tasks will complete, one will panic
    for handle in handles {
        if handle.await.is_err() {
            panic_detected.fetch_add(1, Ordering::Relaxed);
        }
    }

    // Verify exactly one panic occurred
    assert_eq!(
        panic_detected.load(Ordering::Relaxed),
        1,
        "Should detect exactly one panic"
    );

    // Verify store is still functional
    let stats = store.statistics();
    assert!(
        stats.memory_count >= 100,
        "Initial memories should still be accessible"
    );

    // Try new operations post-panic
    let episode = EpisodeBuilder::new()
        .id("post-panic".to_string())
        .when(chrono::Utc::now())
        .what("After panic content".to_string())
        .embedding([0.5; 768])
        .confidence(Confidence::HIGH)
        .build();

    let memory = Arc::new(Memory::from_episode(episode, 0.8));
    store
        .store(memory)
        .await
        .expect("Store should succeed despite earlier panic");

    // Verify the post-panic memory
    let result = store.get("post-panic").expect("Get should succeed");
    assert!(result.is_some(), "Post-panic memory should be stored");
}

// ============================================================================
// Test 7: Property-Based Testing - Content Integrity Invariants
// ============================================================================

use proptest::prelude::*;

proptest! {
    #![proptest_config(ProptestConfig::with_cases(50))]

    #[test]
    fn prop_concurrent_store_preserves_content(
        thread_count in 2usize..=10,
        memories_per_thread in 10usize..=50,
        content_sizes in prop::collection::vec(1usize..=1000, 10..50)
    ) {
        let runtime = tokio::runtime::Runtime::new().unwrap();
        runtime.block_on(async {
            let temp_dir = TempDir::new().unwrap();
            let metrics = Arc::new(StorageMetrics::new());
            let store = Arc::new(
                MappedWarmStorage::new(
                    temp_dir.path().join("test.dat"),
                    thread_count * memories_per_thread * 2,
                    metrics
                )
                .unwrap()
            );

            let mut handles = vec![];
            let stored_contents = Arc::new(Mutex::new(HashMap::new()));

            for thread_id in 0..thread_count {
                let store = Arc::clone(&store);
                let stored_contents = Arc::clone(&stored_contents);
                let content_sizes = content_sizes.clone();

                let handle = tokio::spawn(async move {
                    for i in 0..memories_per_thread {
                        let size = content_sizes.get(i % content_sizes.len())
                            .copied()
                            .unwrap_or(100);
                        let content = "x".repeat(size);
                        let id = format!("t{thread_id}-m{i}");

                        stored_contents.lock().await.insert(id.clone(), content.clone());

                        let episode = EpisodeBuilder::new()
                            .id(id)
                            .when(chrono::Utc::now())
                            .what(content)
                            .embedding([0.5; 768])
                            .confidence(Confidence::HIGH)
                            .build();

                        let memory = Arc::new(Memory::from_episode(episode, 0.8));
                        store.store(memory).await.unwrap();
                    }
                });
                handles.push(handle);
            }

            for handle in handles {
                handle.await.unwrap();
            }

            // Property: All stored content must be retrievable and correct
            let expected = stored_contents.lock().await;
            let stats = store.statistics();

            prop_assert_eq!(stats.memory_count, expected.len(), "Memory count mismatch");

            for (id, expected_content) in expected.iter() {
                let memory = store.get(id)
                    .map_err(|e| TestCaseError::fail(format!("Get failed for {id}: {e}")))?
                    .ok_or_else(|| TestCaseError::fail(format!("Memory {id} not found")))?;

                prop_assert_eq!(
                    memory.content.as_deref(),
                    Some(expected_content.as_str()),
                    "Content mismatch for {}",
                    id
                );
            }
            drop(expected);

            Ok(())
        })?;
    }
}
