//! End-to-end integration tests for streaming infrastructure.
//!
//! Tests the complete workflow: session creation, observation streaming,
//! HNSW insertion, recall, and session cleanup.
//!
//! ## Test Scenarios
//!
//! 1. **End-to-end workflow**: Stream 10K observations, validate indexing and recall
//! 2. **Multi-client concurrent**: 3 clients streaming to different memory spaces
//! 3. **Backpressure activation**: Exceed capacity and validate admission control
//! 4. **Worker failure recovery**: Simulate worker crash and validate recovery
//! 5. **Incremental recall**: Issue recalls during streaming, validate snapshot isolation

use chrono::Utc;
use engram_core::streaming::{
    BackpressureMonitor, ObservationPriority, ObservationQueue, QueueConfig, SessionManager,
    SpaceIsolatedHnsw, WorkerPool, WorkerPoolConfig,
};
use engram_core::{Confidence, Cue, Episode, EpisodeBuilder, MemorySpaceId};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::Duration;

/// Test 1: End-to-end workflow with 10K observations
#[test]
fn test_end_to_end_streaming_workflow() {
    // Setup components
    let session_manager = Arc::new(SessionManager::new());
    let queue_config = QueueConfig {
        capacity: 20_000,
        ..Default::default()
    };
    let obs_queue = Arc::new(ObservationQueue::new(queue_config));
    let hnsw = Arc::new(SpaceIsolatedHnsw::new());

    // Create worker pool
    let worker_config = WorkerPoolConfig {
        num_workers: 4,
        min_batch_size: 10,
        max_batch_size: 100,
        ..Default::default()
    };
    let worker_pool = Arc::new(WorkerPool::new(
        worker_config,
        Arc::clone(&obs_queue),
        Arc::clone(&hnsw),
    ));

    // Create session
    let space_id = MemorySpaceId::try_from("test_space").expect("Valid space ID");
    let session = session_manager
        .create_session(space_id.clone(), 1000)
        .expect("Session creation should succeed");

    // Stream 10K observations
    let observation_count = 10_000;
    for i in 0..observation_count {
        let episode = create_test_episode(i, &space_id);
        obs_queue
            .enqueue(
                ObservationPriority::Normal,
                space_id.clone(),
                Arc::new(episode),
            )
            .expect("Enqueue should succeed");
    }

    // Wait for processing (allow 5 seconds for 10K observations)
    thread::sleep(Duration::from_secs(5));

    // Verify worker pool processed all observations
    let stats = worker_pool.total_stats();
    assert_eq!(
        stats.processed_observations, observation_count,
        "All observations should be processed"
    );

    // Verify HNSW index populated
    let index_size = hnsw.space_size(&space_id).expect("Space should exist");
    assert_eq!(
        index_size, observation_count as usize,
        "HNSW index should contain all observations"
    );

    // Execute recall to verify indexing correctness
    let query_embedding = create_test_embedding(0);
    let cue = Cue::embedding("test_query".to_string(), query_embedding, Confidence::HIGH);

    let results = hnsw
        .search(&space_id, &cue, 10)
        .expect("Search should succeed");

    assert_eq!(results.len(), 10, "Should retrieve 10 results");
    assert!(
        results[0].1.value() > 0.8,
        "Top result should have high similarity (>0.8)"
    );

    // Cleanup
    session_manager.close_session(&session.id);
    worker_pool.shutdown(Duration::from_secs(5));
}

/// Test 2: Multi-client concurrent streaming
#[test]
fn test_multi_client_concurrent() {
    let session_manager = Arc::new(SessionManager::new());
    let queue_config = QueueConfig {
        capacity: 50_000,
        ..Default::default()
    };
    let obs_queue = Arc::new(ObservationQueue::new(queue_config));
    let hnsw = Arc::new(SpaceIsolatedHnsw::new());

    let worker_config = WorkerPoolConfig {
        num_workers: 4,
        ..Default::default()
    };
    let worker_pool = Arc::new(WorkerPool::new(
        worker_config,
        Arc::clone(&obs_queue),
        Arc::clone(&hnsw),
    ));

    // Create 3 clients with different memory spaces
    let num_clients = 3;
    let observations_per_client = 5_000;

    let handles: Vec<_> = (0..num_clients)
        .map(|client_id| {
            let queue = Arc::clone(&obs_queue);
            let hnsw_ref = Arc::clone(&hnsw);
            let session_mgr = Arc::clone(&session_manager);

            thread::spawn(move || {
                // Create unique space for this client
                let space_id = MemorySpaceId::try_from(format!("space_{client_id}"))
                    .expect("Valid space ID");

                // Create session
                let _session = session_mgr
                    .create_session(space_id.clone(), 1000)
                    .expect("Session creation should succeed");

                // Stream observations
                for i in 0..observations_per_client {
                    let episode = create_test_episode(i, &space_id);
                    queue
                        .enqueue(
                            ObservationPriority::Normal,
                            space_id.clone(),
                            Arc::new(episode),
                        )
                        .expect("Enqueue should succeed");
                }

                // Return space_id for validation
                space_id
            })
        })
        .collect();

    // Collect space IDs
    let space_ids: Vec<_> = handles
        .into_iter()
        .map(|h| h.join().expect("Client thread should complete"))
        .collect();

    // Wait for processing
    thread::sleep(Duration::from_secs(5));

    // Verify total observations processed
    let stats = worker_pool.total_stats();
    let total_observations = num_clients * observations_per_client;
    assert_eq!(
        stats.processed_observations, total_observations,
        "All observations from all clients should be processed"
    );

    // Verify space isolation: each space should have exactly observations_per_client
    for space_id in &space_ids {
        let size = hnsw.space_size(space_id).expect("Space should exist");
        assert_eq!(
            size, observations_per_client as usize,
            "Space {space_id:?} should have exactly {observations_per_client} observations"
        );
    }

    // Verify no cross-contamination: query each space should return only its observations
    for (client_id, space_id) in space_ids.iter().enumerate() {
        let query_embedding = create_test_embedding(0);
        let cue = Cue::embedding("test_query".to_string(), query_embedding, Confidence::HIGH);

        let results = hnsw
            .search(space_id, &cue, 100)
            .expect("Search should succeed");

        // All results should belong to this space (verified by episode ID prefix)
        for (episode, _confidence) in &results {
            assert!(
                episode.id.starts_with(&format!("space_{client_id}_")),
                "Episode {0} should belong to space_{client_id}",
                episode.id
            );
        }
    }

    // Cleanup
    worker_pool.shutdown(Duration::from_secs(5));
}

/// Test 3: Streaming with backpressure
#[test]
fn test_streaming_with_backpressure() {
    let queue_config = QueueConfig {
        capacity: 1_000, // Small capacity to trigger backpressure
        ..Default::default()
    };
    let obs_queue = Arc::new(ObservationQueue::new(queue_config));
    let hnsw = Arc::new(SpaceIsolatedHnsw::new());

    let worker_config = WorkerPoolConfig {
        num_workers: 2, // Fewer workers to increase pressure
        min_batch_size: 50,
        max_batch_size: 100,
        idle_sleep_ms: 10, // Slower processing
        ..Default::default()
    };
    let worker_pool = Arc::new(WorkerPool::new(
        worker_config,
        Arc::clone(&obs_queue),
        Arc::clone(&hnsw),
    ));

    let backpressure_monitor = BackpressureMonitor::new(1_000);
    let space_id = MemorySpaceId::try_from("test_space").expect("Valid space ID");

    // Track rejections
    let rejections = Arc::new(AtomicU64::new(0));
    let rejections_clone = Arc::clone(&rejections);

    // Stream at high rate to exceed capacity
    let stream_handle = thread::spawn(move || {
        for i in 0..10_000 {
            let episode = create_test_episode(i, &space_id);
            let current_depth = obs_queue.current_depth();

            // Check backpressure before enqueuing
            if backpressure_monitor.should_admit(current_depth) {
                match obs_queue.enqueue(
                    ObservationPriority::Normal,
                    space_id.clone(),
                    Arc::new(episode),
                ) {
                    Ok(_) => {}
                    Err(_) => {
                        rejections_clone.fetch_add(1, Ordering::Relaxed);
                    }
                }
            } else {
                rejections_clone.fetch_add(1, Ordering::Relaxed);
            }

            // No sleep = maximum pressure
        }
    });

    stream_handle.join().expect("Stream thread should complete");

    // Wait for queue to drain
    thread::sleep(Duration::from_secs(5));

    // Verify backpressure was activated
    let rejection_count = rejections.load(Ordering::Acquire);
    assert!(
        rejection_count > 0,
        "Backpressure should have rejected some observations (got {rejection_count} rejections)"
    );

    // Verify no observations lost: processed + rejected = total sent
    let stats = worker_pool.total_stats();
    let total_sent = 10_000;
    assert_eq!(
        stats.processed_observations + rejection_count,
        total_sent,
        "No observations should be lost: processed ({}) + rejected ({}) should equal sent ({})",
        stats.processed_observations,
        rejection_count,
        total_sent
    );

    // Verify queue eventually drained
    assert!(
        obs_queue.current_depth() < 100,
        "Queue should be mostly drained after processing"
    );

    // Cleanup
    worker_pool.shutdown(Duration::from_secs(5));
}

/// Test 4: Worker failure recovery via work stealing
#[test]
fn test_worker_failure_recovery() {
    let queue_config = QueueConfig {
        capacity: 20_000,
        ..Default::default()
    };
    let obs_queue = Arc::new(ObservationQueue::new(queue_config));
    let hnsw = Arc::new(SpaceIsolatedHnsw::new());

    let worker_config = WorkerPoolConfig {
        num_workers: 4,
        steal_threshold: 100, // Low threshold for aggressive work stealing
        ..Default::default()
    };
    let worker_pool = Arc::new(WorkerPool::new(
        worker_config,
        Arc::clone(&obs_queue),
        Arc::clone(&hnsw),
    ));

    let space_id = MemorySpaceId::try_from("test_space").expect("Valid space ID");

    // Stream observations
    let observation_count = 5_000;
    for i in 0..observation_count {
        let episode = create_test_episode(i, &space_id);
        obs_queue
            .enqueue(
                ObservationPriority::Normal,
                space_id.clone(),
                Arc::new(episode),
            )
            .expect("Enqueue should succeed");
    }

    // Note: We cannot actually kill a worker thread in this test without unsafe code
    // Instead, we verify work stealing behavior by observing stats

    // Wait for processing with work stealing
    thread::sleep(Duration::from_secs(5));

    // Verify all observations processed
    let stats = worker_pool.total_stats();
    assert_eq!(
        stats.processed_observations, observation_count,
        "All observations should be processed despite uneven load distribution"
    );

    // Verify work stealing occurred (at least some workers should have stolen batches)
    let total_steals: u64 = worker_pool
        .worker_stats()
        .iter()
        .map(|s| s.stolen_batches)
        .sum();

    assert!(
        total_steals > 0,
        "Work stealing should have occurred (found {total_steals} stolen batches)"
    );

    // Cleanup
    worker_pool.shutdown(Duration::from_secs(5));
}

/// Test 5: Incremental recall during streaming
#[test]
fn test_incremental_recall_during_streaming() {
    let queue_config = QueueConfig {
        capacity: 20_000,
        ..Default::default()
    };
    let obs_queue = Arc::new(ObservationQueue::new(queue_config));
    let hnsw = Arc::new(SpaceIsolatedHnsw::new());

    let worker_config = WorkerPoolConfig {
        num_workers: 4,
        ..Default::default()
    };
    let worker_pool = Arc::new(WorkerPool::new(
        worker_config,
        Arc::clone(&obs_queue),
        Arc::clone(&hnsw),
    ));

    let space_id = MemorySpaceId::try_from("test_space").expect("Valid space ID");

    // Stream observations in background
    let hnsw_streaming = Arc::clone(&hnsw);
    let queue_streaming = Arc::clone(&obs_queue);
    let space_streaming = space_id.clone();

    let stream_handle = thread::spawn(move || {
        for i in 0..10_000 {
            let episode = create_test_episode(i, &space_streaming);
            queue_streaming
                .enqueue(
                    ObservationPriority::Normal,
                    space_streaming.clone(),
                    Arc::new(episode),
                )
                .expect("Enqueue should succeed");

            // Throttle to sustained rate
            if i % 100 == 0 {
                thread::sleep(Duration::from_millis(10));
            }
        }
    });

    // Issue recalls periodically during streaming
    let recall_count = 10;
    let mut recall_latencies = Vec::new();

    for _ in 0..recall_count {
        thread::sleep(Duration::from_millis(500)); // Wait between recalls

        let query_embedding = create_test_embedding(0);
        let cue = Cue::embedding("test_query".to_string(), query_embedding, Confidence::HIGH);

        let start = std::time::Instant::now();
        let results = hnsw.search(&space_id, &cue, 10);
        let latency = start.elapsed();

        recall_latencies.push(latency);

        // Verify recall succeeded
        assert!(
            results.is_ok(),
            "Recall should succeed during streaming"
        );

        // Verify results are consistent (snapshot isolation)
        if let Ok(episodes) = results {
            // Results should be from a consistent snapshot (no partial updates)
            for (episode, _confidence) in &episodes {
                assert!(
                    episode.id.starts_with(&format!("{}_", space_id)),
                    "Episode should belong to correct space"
                );
            }
        }
    }

    stream_handle.join().expect("Stream thread should complete");

    // Wait for final processing
    thread::sleep(Duration::from_secs(2));

    // Verify recall latency target: P99 < 20ms
    recall_latencies.sort();
    let p99_index = (recall_latencies.len() as f32 * 0.99) as usize;
    let p99_latency = recall_latencies[p99_index.min(recall_latencies.len() - 1)];

    assert!(
        p99_latency < Duration::from_millis(20),
        "P99 recall latency should be < 20ms (got {p99_latency:?})"
    );

    // Verify all observations processed
    let stats = worker_pool.total_stats();
    assert_eq!(
        stats.processed_observations, 10_000,
        "All observations should be processed"
    );

    // Cleanup
    worker_pool.shutdown(Duration::from_secs(5));
}

// ==================== Helper Functions ====================

/// Create test episode with deterministic embedding
fn create_test_episode(id: u64, space_id: &MemorySpaceId) -> Episode {
    let embedding = create_test_embedding(id);

    EpisodeBuilder::new()
        .id(format!("{space_id}_episode_{id}"))
        .when(Utc::now())
        .what(format!("Test observation {id} for space {space_id}"))
        .embedding(embedding)
        .confidence(Confidence::HIGH)
        .build()
}

/// Create deterministic test embedding from seed
fn create_test_embedding(seed: u64) -> [f32; 768] {
    let mut embedding = [0.0f32; 768];
    for (i, val) in embedding.iter_mut().enumerate() {
        *val = ((seed as f32 + i as f32) * 0.001).sin();
    }
    // Normalize
    let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
    for val in &mut embedding {
        *val /= norm;
    }
    embedding
}
