//! Load testing for dual-memory architecture
//!
//! Stress tests under concurrent operations: reads, writes, consolidation.

#![cfg(feature = "dual_memory_types")]

mod support;

use chrono::Utc;
use engram_core::{Confidence, Cue, CueBuilder, MemoryStore};
use std::time::Duration;
use support::dual_memory_fixtures::generate_test_episodes;

/// Test 1: Concurrent episode storage
///
/// Validates that multiple threads can store episodes simultaneously
/// without data loss or corruption.
#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn test_concurrent_episode_storage() {
    let store = Arc::new(MemoryStore::new(16384));

    // Spawn 8 threads, each storing 1000 episodes
    let mut handles = Vec::new();

    for thread_id in 0..8 {
        let store_clone = store.clone();
        let handle = tokio::spawn(async move {
            let episodes = generate_test_episodes(1000, thread_id * 1000);

            for episode in episodes {
                let result = store_clone.store(episode);
                assert!(
                    result.activation.is_successful(),
                    "Thread {} failed to store episode",
                    thread_id
                );
            }

            println!("Thread {} completed 1000 stores", thread_id);
        });
        handles.push(handle);
    }

    // Wait for all threads to complete
    for handle in handles {
        handle
            .await
            .expect("Thread should complete without panic");
    }

    // Verify 8000 episodes stored
    let final_count = store.count();
    assert_eq!(
        final_count, 8000,
        "Should have 8000 episodes after concurrent stores"
    );
}

/// Test 2: Concurrent recall queries
///
/// Validates read-heavy workload with 100 concurrent queries.
#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn test_concurrent_recall_queries() {
    let episodes = generate_test_episodes(10_000, 42);
    let store = Arc::new(MemoryStore::new(16384));

    // Populate store
    for episode in episodes {
        store.store(episode);
    }

    println!("Store populated with 10K episodes");

    // Spawn 100 concurrent recall queries
    let mut handles = Vec::new();

    for query_id in 0..100 {
        let store_clone = store.clone();
        let handle = tokio::spawn(async move {
            for iteration in 0..100 {
                let query_embedding = [0.5f32; 768];
                let cue = CueBuilder::new()
                    .id(format!("concurrent_cue_{}_{}", query_id, iteration))
                    .embedding_search(query_embedding, Confidence::LOW)
                    .cue_confidence(Confidence::HIGH)
                    .build();

                let results = store_clone.recall(&cue);
                assert!(
                    !results.results.is_empty(),
                    "Query {}-{} should return results",
                    query_id,
                    iteration
                );
            }
        });
        handles.push(handle);
    }

    // Wait for all queries to complete
    for handle in handles {
        handle
            .await
            .expect("Query thread should complete without panic");
    }

    println!("All 10K queries completed successfully");
}

/// Test 3: Mixed read-write workload
///
/// Simulates realistic production scenario with 70% reads, 30% writes.
#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn test_mixed_read_write_workload() {
    let store = Arc::new(MemoryStore::new(8192));

    // Pre-populate with some data
    let initial_episodes = generate_test_episodes(1000, 999);
    for episode in initial_episodes {
        store.store(episode);
    }

    let mut handles = Vec::new();

    // Spawn 4 read workers
    for worker_id in 0..4 {
        let store_clone = store.clone();
        let handle = tokio::spawn(async move {
            for iteration in 0..500 {
                let query_embedding = [0.5f32; 768];
                let cue = CueBuilder::new()
                    .id(format!("read_worker_{}_iter_{}", worker_id, iteration))
                    .embedding_search(query_embedding, Confidence::LOW)
                    .cue_confidence(Confidence::HIGH)
                    .build();

                let _results = store_clone.recall(&cue);

                tokio::time::sleep(Duration::from_micros(100)).await;
            }
            println!("Read worker {} completed", worker_id);
        });
        handles.push(handle);
    }

    // Spawn 2 write workers
    for worker_id in 0..2 {
        let store_clone = store.clone();
        let handle = tokio::spawn(async move {
            let episodes = generate_test_episodes(500, worker_id * 500 + 10_000);

            for episode in episodes {
                let result = store_clone.store(episode);
                assert!(result.activation.is_successful());

                tokio::time::sleep(Duration::from_micros(200)).await;
            }
            println!("Write worker {} completed", worker_id);
        });
        handles.push(handle);
    }

    // Wait for all workers
    for handle in handles {
        handle.await.expect("Worker should complete");
    }

    let final_count = store.count();
    assert!(
        final_count >= 1000,
        "Should have at least initial episodes"
    );

    println!("Mixed workload completed: final_count={}", final_count);
}

/// Test 4: Consolidation during writes
///
/// Validates that background consolidation doesn't interfere with writes.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[cfg(feature = "pattern_completion")]
async fn test_consolidation_during_writes() {
    let store = Arc::new(MemoryStore::new(4096));

    // Pre-populate for consolidation
    let initial_episodes = generate_test_episodes(2000, 123);
    for episode in initial_episodes {
        store.store(episode);
    }

    // Spawn consolidation worker
    let store_clone = store.clone();
    let consolidation_handle = tokio::spawn(async move {
        for iteration in 0..10 {
            let _patterns = store_clone.consolidate();
            println!("Consolidation iteration {} completed", iteration);
            tokio::time::sleep(Duration::from_secs(1)).await;
        }
    });

    // Spawn write worker
    let store_clone2 = store.clone();
    let write_handle = tokio::spawn(async move {
        let episodes = generate_test_episodes(1000, 456);

        for (i, episode) in episodes.into_iter().enumerate() {
            let result = store_clone2.store(episode);
            assert!(
                result.activation.is_successful(),
                "Write {} failed during consolidation",
                i
            );

            if i % 100 == 0 {
                println!("Stored {} episodes during consolidation", i);
            }

            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    });

    // Wait for both workers
    let consolidation_result = consolidation_handle.await;
    let write_result = write_handle.await;

    assert!(consolidation_result.is_ok());
    assert!(write_result.is_ok());

    println!("Consolidation + writes test passed");
}

/// Test 5: Burst traffic handling
///
/// Tests system behavior under sudden traffic spike.
#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn test_burst_traffic_handling() {
    let store = Arc::new(MemoryStore::new(8192));

    // Phase 1: Normal load (100 episodes)
    let normal_episodes = generate_test_episodes(100, 111);
    for episode in normal_episodes {
        store.store(episode);
    }

    println!("Phase 1: Normal load completed");

    // Phase 2: Burst (1000 concurrent stores)
    let mut handles = Vec::new();

    for batch_id in 0..10 {
        let store_clone = store.clone();
        let handle = tokio::spawn(async move {
            let episodes = generate_test_episodes(100, batch_id * 100 + 1000);

            for episode in episodes {
                store_clone.store(episode);
            }
        });
        handles.push(handle);
    }

    // Wait for burst to complete
    for handle in handles {
        handle.await.expect("Burst batch should complete");
    }

    println!("Phase 2: Burst traffic completed");

    // Phase 3: Verify system still responsive
    let query_embedding = [0.5f32; 768];
    let cue = Cue::embedding("post_burst_cue".to_string(), query_embedding, Confidence::MEDIUM);

    let results = store.recall(&cue);
    assert!(
        !results.results.is_empty(),
        "System should be responsive after burst"
    );

    println!("Phase 3: Post-burst verification passed");
}

/// Test 6: Memory pressure simulation
///
/// Tests behavior when approaching capacity limits.
#[test]
fn test_memory_pressure() {
    // Small store to hit capacity quickly
    let store = MemoryStore::new(512);

    // Store episodes until capacity
    let episodes = generate_test_episodes(1000, 789);

    for (i, episode) in episodes.into_iter().enumerate() {
        let result = store.store(episode);

        // Store may degrade but should never panic
        if i % 100 == 0 {
            println!(
                "Stored {} episodes, activation={}",
                i,
                result.activation.value()
            );
        }
    }

    // System should still be responsive
    let query_embedding = [0.5f32; 768];
    let cue = Cue::embedding("pressure_cue".to_string(), query_embedding, Confidence::MEDIUM);

    let results = store.recall(&cue);

    println!(
        "Under pressure: count={}, recall_results={}",
        store.count(),
        results.results.len()
    );

    // Should return some results even under pressure
    assert!(
        !results.results.is_empty() || store.count() == 0,
        "Should return results under pressure"
    );
}

/// Test 7: Long-running stability test
///
/// Runs mixed workload for extended period to detect memory leaks or degradation.
#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
#[ignore = "Long-running test, use --ignored"]
async fn test_long_running_stability() {
    let store = Arc::new(MemoryStore::new(8192));
    let duration = Duration::from_secs(300); // 5 minutes
    let start = std::time::Instant::now();

    println!("Starting 5-minute stability test");

    let mut handles = Vec::new();

    // Spawn continuous write worker
    let store_clone = store.clone();
    let start_clone = start;
    let duration_clone = duration;
    let write_handle = tokio::spawn(async move {
        let mut episode_counter = 0;

        while start_clone.elapsed() < duration_clone {
            let episodes = generate_test_episodes(10, episode_counter);

            for episode in episodes {
                store_clone.store(episode);
            }

            episode_counter += 10;
            tokio::time::sleep(Duration::from_millis(100)).await;
        }

        println!("Write worker stored {} episodes", episode_counter);
        episode_counter
    });
    handles.push(write_handle);

    // Spawn continuous read worker
    let store_clone2 = store.clone();
    let start_clone2 = start;
    let duration_clone2 = duration;
    let read_handle = tokio::spawn(async move {
        let mut query_counter = 0;

        while start_clone2.elapsed() < duration_clone2 {
            let query_embedding = [0.5f32; 768];
            let cue = CueBuilder::new()
                .id(format!("stability_cue_{}", query_counter))
                .embedding_search(query_embedding, Confidence::LOW)
                .cue_confidence(Confidence::HIGH)
                .build();

            let _results = store_clone2.recall(&cue);

            query_counter += 1;
            tokio::time::sleep(Duration::from_millis(50)).await;
        }

        println!("Read worker executed {} queries", query_counter);
        query_counter
    });
    handles.push(read_handle);

    // Wait for all workers
    for handle in handles {
        handle.await.expect("Worker should complete");
    }

    println!(
        "Stability test completed: duration={:?}, final_count={}",
        start.elapsed(),
        store.count()
    );
}
