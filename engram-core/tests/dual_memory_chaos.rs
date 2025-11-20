//! Chaos testing for dual-memory architecture
//!
//! Injects failures and validates graceful degradation, recovery,
//! and data consistency under adverse conditions.

#![cfg(feature = "dual_memory_types")]

mod support;

use chrono::Utc;
use engram_core::{Confidence, Cue, CueBuilder, MemoryStore};
use rand::{Rng, SeedableRng, rngs::StdRng};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::Duration;
use support::dual_memory_fixtures::generate_test_episodes;

/// Test 1: Random operation failures
///
/// Simulates random failures (5% of operations) and verifies system
/// remains consistent and eventually recovers.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_random_operation_failures() {
    let store = Arc::new(MemoryStore::new(2048));
    let failure_injector = Arc::new(ChaosInjector::new(0.05, 123));

    // Populate initial data
    let episodes = generate_test_episodes(500, 42);
    for episode in episodes {
        store.store(episode);
    }

    println!("Initial population: {} episodes", store.count());

    // Run mixed workload with injected failures
    let mut handles = Vec::new();

    // Write worker with failure injection
    let store_clone = store.clone();
    let injector_clone = failure_injector.clone();
    let write_handle = tokio::spawn(async move {
        let episodes = generate_test_episodes(500, 999);
        let mut success_count = 0;
        let mut failure_count = 0;

        for episode in episodes {
            if injector_clone.should_fail() {
                failure_count += 1;
                // Skip this operation to simulate failure
                continue;
            }

            let result = store_clone.store(episode);
            if result.activation.is_successful() {
                success_count += 1;
            } else {
                failure_count += 1;
            }
        }

        println!(
            "Write worker: success={}, failures={}",
            success_count, failure_count
        );
        (success_count, failure_count)
    });
    handles.push(write_handle);

    // Read worker with failure injection
    let store_clone2 = store.clone();
    let injector_clone2 = failure_injector.clone();
    let read_handle = tokio::spawn(async move {
        let mut success_count = 0;
        let mut failure_count = 0;

        for i in 0..200 {
            if injector_clone2.should_fail() {
                failure_count += 1;
                continue;
            }

            let query_embedding = [0.5f32; 768];
            let cue = CueBuilder::new()
                .id(format!("chaos_cue_{}", i))
                .embedding_search(query_embedding, Confidence::LOW)
                .cue_confidence(Confidence::HIGH)
                .build();

            let results = store_clone2.recall(&cue);
            if !results.results.is_empty() {
                success_count += 1;
            } else {
                failure_count += 1;
            }

            tokio::time::sleep(Duration::from_millis(5)).await;
        }

        println!(
            "Read worker: success={}, failures={}",
            success_count, failure_count
        );
        (success_count, failure_count)
    });
    handles.push(read_handle);

    // Wait for workers
    for handle in handles {
        handle.await.expect("Worker should complete");
    }

    // Verify system is still consistent
    let final_count = store.count();
    println!(
        "Chaos test completed: final_count={}, total_failures={}",
        final_count,
        failure_injector.failure_count()
    );

    assert!(final_count >= 500, "Should retain at least initial data");
}

/// Test 2: Concurrent failure scenarios
///
/// Tests behavior when multiple failures occur simultaneously.
#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn test_concurrent_failure_scenarios() {
    let store = Arc::new(MemoryStore::new(4096));
    let failure_flag = Arc::new(AtomicBool::new(false));

    // Populate data
    let episodes = generate_test_episodes(1000, 456);
    for episode in episodes {
        store.store(episode);
    }

    let mut handles = Vec::new();

    // Spawn multiple workers
    for worker_id in 0..8 {
        let store_clone = store.clone();
        let failure_flag_clone = failure_flag.clone();

        let handle = tokio::spawn(async move {
            let episodes = generate_test_episodes(100, worker_id * 100 + 5000);

            for (i, episode) in episodes.into_iter().enumerate() {
                // Inject synchronized failure at iteration 50
                if i == 50 {
                    failure_flag_clone.store(true, Ordering::Release);
                    tokio::time::sleep(Duration::from_millis(100)).await;
                    failure_flag_clone.store(false, Ordering::Release);
                }

                // Skip operations during failure window
                if failure_flag_clone.load(Ordering::Acquire) {
                    continue;
                }

                store_clone.store(episode);
            }
        });

        handles.push(handle);
    }

    // Wait for all workers
    for handle in handles {
        handle.await.expect("Worker should complete");
    }

    println!(
        "Concurrent failure test completed: final_count={}",
        store.count()
    );
}

/// Test 3: Recovery after transient failures
///
/// Validates that system recovers and continues normal operation
/// after transient failure period.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_recovery_after_transient_failures() {
    let store = Arc::new(MemoryStore::new(2048));
    let failure_window = Arc::new(AtomicBool::new(false));

    // Phase 1: Normal operation
    let episodes_phase1 = generate_test_episodes(500, 111);
    for episode in episodes_phase1 {
        store.store(episode);
    }

    let count_phase1 = store.count();
    println!("Phase 1 (normal): count={}", count_phase1);

    // Phase 2: Failure window
    failure_window.store(true, Ordering::Release);

    let store_clone = store.clone();
    let failure_window_clone = failure_window.clone();
    let failure_handle = tokio::spawn(async move {
        let episodes_phase2 = generate_test_episodes(200, 222);
        let mut skipped = 0;

        for episode in episodes_phase2 {
            if failure_window_clone.load(Ordering::Acquire) {
                skipped += 1;
                continue;
            }

            store_clone.store(episode);
        }

        println!("Phase 2 (failure): skipped={} episodes", skipped);
        skipped
    });

    // Let failure window run for 100ms
    tokio::time::sleep(Duration::from_millis(100)).await;

    // Phase 3: Recovery
    failure_window.store(false, Ordering::Release);

    let skipped_count = failure_handle.await.expect("Failure phase should complete");

    // Continue normal operations - track success rate
    let episodes_phase3 = generate_test_episodes(300, 333);
    let mut successful_stores = 0;
    for episode in episodes_phase3 {
        let result = store.store(episode);
        if result.activation.is_successful() {
            successful_stores += 1;
        }
    }

    // Should have reasonable success rate (>50%) after recovery
    assert!(
        successful_stores > 150,
        "Post-recovery stores should have >50% success rate (got {}/300)",
        successful_stores
    );

    let count_phase3 = store.count();
    println!(
        "Phase 3 (recovery): count={}, skipped_during_failure={}",
        count_phase3, skipped_count
    );

    // Verify recovery: should have phase1 + phase3 data (phase2 was skipped)
    assert!(
        count_phase3 >= count_phase1,
        "Should have recovered to at least pre-failure count"
    );
}

/// Test 4: Data consistency under failures
///
/// Ensures that failures don't corrupt stored data or create inconsistencies.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_data_consistency_under_failures() {
    let store = Arc::new(MemoryStore::new(2048));
    let chaos = Arc::new(ChaosInjector::new(0.10, 789));

    // Store episodes and track which succeeded
    let episodes = generate_test_episodes(1000, 42);
    let mut stored_ids = Vec::new();

    for episode in episodes {
        if chaos.should_fail() {
            continue; // Skip to simulate failure
        }

        let episode_id = episode.id.clone();
        let result = store.store(episode);

        if result.activation.is_successful() {
            stored_ids.push(episode_id);
        }
    }

    println!(
        "Stored {} episodes with {}% failure rate",
        stored_ids.len(),
        chaos.failure_rate() * 100.0
    );

    // Verify all stored episodes are retrievable and uncorrupted
    let mut verified_count = 0;

    for episode_id in &stored_ids {
        let retrieved = store.get_episode(episode_id);

        assert!(
            retrieved.is_some(),
            "Episode {} should be retrievable",
            episode_id
        );

        let episode = retrieved.unwrap();
        assert_eq!(&episode.id, episode_id, "Episode ID should match");

        // Verify embedding integrity (not all zeros)
        let has_data = episode.embedding.iter().any(|&x| x.abs() > 1e-6);
        assert!(
            has_data,
            "Episode {} should have valid embedding",
            episode_id
        );

        verified_count += 1;
    }

    println!("Verified {} episodes for consistency", verified_count);
    assert_eq!(verified_count, stored_ids.len());
}

/// Test 5: Graceful degradation under sustained failures
///
/// Tests that system degrades gracefully rather than failing catastrophically
/// when failures persist.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_graceful_degradation() {
    let store = Arc::new(MemoryStore::new(1024));
    let high_failure_chaos = Arc::new(ChaosInjector::new(0.30, 999)); // 30% failure rate

    let mut handles = Vec::new();

    // Spawn workers with high failure rate
    for worker_id in 0..4 {
        let store_clone = store.clone();
        let chaos_clone = high_failure_chaos.clone();

        let handle = tokio::spawn(async move {
            let episodes = generate_test_episodes(250, worker_id * 250);
            let mut success_count = 0;
            let mut degraded_count = 0;
            let mut failure_count = 0;

            for episode in episodes {
                if chaos_clone.should_fail() {
                    failure_count += 1;
                    continue;
                }

                let result = store_clone.store(episode);

                if result.activation.is_successful() {
                    success_count += 1;
                } else if result.activation.is_degraded() {
                    degraded_count += 1;
                } else {
                    failure_count += 1;
                }
            }

            println!(
                "Worker {}: success={}, degraded={}, failures={}",
                worker_id, success_count, degraded_count, failure_count
            );

            (success_count, degraded_count, failure_count)
        });

        handles.push(handle);
    }

    // Collect results
    let mut total_success = 0;
    let mut total_degraded = 0;
    let mut total_failures = 0;

    for handle in handles {
        let (success, degraded, failures) = handle.await.expect("Worker should complete");
        total_success += success;
        total_degraded += degraded;
        total_failures += failures;
    }

    println!(
        "Graceful degradation test: success={}, degraded={}, failures={}",
        total_success, total_degraded, total_failures
    );

    // System should maintain some level of service even under high failure rate
    assert!(
        total_success > 0 || total_degraded > 0,
        "System should complete some operations despite failures"
    );

    // Verify system is still responsive
    let query_embedding = [0.5f32; 768];
    let cue = Cue::embedding(
        "degradation_cue".to_string(),
        query_embedding,
        Confidence::MEDIUM,
    );

    let results = store.recall(&cue);
    println!(
        "Post-degradation query returned {} results",
        results.results.len()
    );
}

/// Chaos injector utility for simulating failures
struct ChaosInjector {
    failure_rate: f32,
    rng: std::sync::Mutex<StdRng>,
    failure_count: AtomicUsize,
}

impl ChaosInjector {
    fn new(failure_rate: f32, seed: u64) -> Self {
        Self {
            failure_rate: failure_rate.clamp(0.0, 1.0),
            rng: std::sync::Mutex::new(StdRng::seed_from_u64(seed)),
            failure_count: AtomicUsize::new(0),
        }
    }

    fn should_fail(&self) -> bool {
        let mut rng = self.rng.lock().expect("Lock should not be poisoned");
        let roll: f32 = rng.sample(rand::distributions::Standard);

        let should_fail = roll < self.failure_rate;

        if should_fail {
            self.failure_count.fetch_add(1, Ordering::Relaxed);
        }

        should_fail
    }

    fn failure_count(&self) -> usize {
        self.failure_count.load(Ordering::Relaxed)
    }

    fn failure_rate(&self) -> f32 {
        self.failure_rate
    }
}
