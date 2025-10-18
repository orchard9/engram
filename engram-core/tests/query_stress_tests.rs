//! Stress tests for probabilistic query system
//!
//! Tests high-volume query operations, memory usage, concurrent execution,
//! and system stability under load.

#![allow(missing_docs)]

use chrono::Utc;
use engram_core::query::ProbabilisticQueryResult;
use engram_core::query::executor::ProbabilisticQueryExecutor;
use engram_core::{Confidence, EpisodeBuilder};
use std::sync::Arc;
use std::thread;

/// Create a test episode with deterministic embedding based on index
fn create_test_episode_for_stress(index: usize, confidence: Confidence) -> engram_core::Episode {
    let mut embedding = [0.0f32; 768];
    for (i, val) in embedding.iter_mut().enumerate() {
        *val = ((index + i) as f32 * 0.001).sin();
    }

    EpisodeBuilder::new()
        .id(format!("stress_ep_{index}"))
        .when(Utc::now())
        .what(format!("Stress test episode {index}"))
        .embedding(embedding)
        .confidence(confidence)
        .build()
}

#[test]
fn test_high_volume_query_operations() {
    let executor = ProbabilisticQueryExecutor::default();

    // Create 1000 episodes
    let episodes: Vec<_> = (0..1000)
        .map(|i| {
            let confidence = if i % 3 == 0 {
                Confidence::HIGH
            } else if i % 3 == 1 {
                Confidence::MEDIUM
            } else {
                Confidence::LOW
            };
            (create_test_episode_for_stress(i, confidence), confidence)
        })
        .collect();

    // Execute query
    let result = executor.execute(episodes, &[], vec![]);

    // Validate result
    assert_eq!(result.len(), 1000);
    assert!(result.confidence_interval.point.raw() > 0.0);
    assert!(result.confidence_interval.lower.raw() <= result.confidence_interval.upper.raw());
}

#[test]
fn test_repeated_query_operations_no_panics() {
    let executor = ProbabilisticQueryExecutor::default();

    // Run 10K queries
    for iteration in 0..10_000 {
        // Use HIGH confidence so is_successful() works
        let episodes = vec![(
            create_test_episode_for_stress(iteration, Confidence::HIGH),
            Confidence::HIGH,
        )];

        let result = executor.execute(episodes, &[], vec![]);

        // Basic validation
        assert_eq!(result.len(), 1);

        // Every 1000th iteration, verify more thoroughly
        if iteration % 1000 == 0 {
            assert!(result.is_successful());
            assert!(result.confidence_interval.point.raw() > 0.0);
        }
    }
}

#[test]
fn test_query_combinator_stress() {
    let executor = ProbabilisticQueryExecutor::default();

    // Create two base results
    let episodes_a = vec![(
        create_test_episode_for_stress(0, Confidence::HIGH),
        Confidence::HIGH,
    )];

    let episodes_b = vec![(
        create_test_episode_for_stress(1, Confidence::MEDIUM),
        Confidence::MEDIUM,
    )];

    let result_a = executor.execute(episodes_a, &[], vec![]);
    let result_b = executor.execute(episodes_b, &[], vec![]);

    // Perform 1000 query operations
    for _ in 0..1000 {
        let and_result = result_a.and(&result_b);
        assert!(and_result.confidence_interval.point.raw() >= 0.0);

        let or_result = result_a.or(&result_b);
        assert!(or_result.confidence_interval.point.raw() <= 1.0);

        let not_result = result_a.not();
        assert!(not_result.confidence_interval.point.raw() <= 1.0);
    }
}

#[test]
fn test_concurrent_query_execution() {
    let executor = Arc::new(ProbabilisticQueryExecutor::default());

    // Create test episodes outside the threads to avoid excessive memory usage
    let test_episodes: Vec<_> = (0..100)
        .map(|i| {
            vec![(
                create_test_episode_for_stress(i, Confidence::MEDIUM),
                Confidence::MEDIUM,
            )]
        })
        .collect();

    let test_episodes = Arc::new(test_episodes);

    // Spawn 10 threads executing 100 queries each and wait for all to complete
    (0..10)
        .map(|_thread_id| {
            let executor = Arc::clone(&executor);
            let episodes = Arc::clone(&test_episodes);

            thread::spawn(move || {
                for i in 0..100 {
                    let result = executor.execute(episodes[i].clone(), &[], vec![]);
                    assert_eq!(result.len(), 1);

                    // Periodically verify query operations work
                    if i % 10 == 0 {
                        let empty_result = ProbabilisticQueryResult::from_episodes(vec![]);
                        let combined = result.or(&empty_result);
                        assert_eq!(combined.len(), 1);
                    }
                }
            })
        })
        .for_each(|h| {
            h.join().unwrap();
        });
}

#[test]
fn test_memory_bounded_large_query() {
    let executor = ProbabilisticQueryExecutor::default();

    // Create large batch of episodes (simulating large query result)
    let large_batch_size = 5000;
    let episodes: Vec<_> = (0..large_batch_size)
        .map(|i| {
            (
                create_test_episode_for_stress(i, Confidence::MEDIUM),
                Confidence::MEDIUM,
            )
        })
        .collect();

    // Execute query
    let result = executor.execute(episodes, &[], vec![]);

    assert_eq!(result.len(), large_batch_size);

    // Perform operations on large result
    let not_result = result.not();
    assert_eq!(not_result.len(), large_batch_size);

    // Combine with empty result (should not significantly increase memory)
    let empty = ProbabilisticQueryResult::from_episodes(vec![]);
    let or_result = result.or(&empty);
    assert_eq!(or_result.len(), large_batch_size);
}

#[test]
fn test_query_chain_stress() {
    let executor = ProbabilisticQueryExecutor::default();

    // Create initial result
    let mut current_result = executor.execute(
        vec![(
            create_test_episode_for_stress(0, Confidence::HIGH),
            Confidence::HIGH,
        )],
        &[],
        vec![],
    );

    // Chain 100 operations
    for i in 1..=100 {
        let next_episodes = vec![(
            create_test_episode_for_stress(i, Confidence::MEDIUM),
            Confidence::MEDIUM,
        )];

        let next_result = executor.execute(next_episodes, &[], vec![]);

        // Alternate between AND and OR operations
        if i % 2 == 0 {
            current_result = current_result.and(&next_result);
        } else {
            current_result = current_result.or(&next_result);
        }

        // Verify probability axioms still hold
        assert!(current_result.confidence_interval.point.raw() >= 0.0);
        assert!(current_result.confidence_interval.point.raw() <= 1.0);
        assert!(
            current_result.confidence_interval.lower.raw()
                <= current_result.confidence_interval.upper.raw()
        );
    }
}

#[test]
fn test_executor_reuse_stress() {
    // Verify executor can be reused many times without issues
    let executor = ProbabilisticQueryExecutor::default();

    for iteration in 0..1000 {
        let episodes = vec![(
            create_test_episode_for_stress(iteration, Confidence::HIGH),
            Confidence::HIGH,
        )];

        let result = executor.execute(episodes, &[], vec![]);
        assert_eq!(result.len(), 1);

        // Verify result can be used in query operations
        let not_result = result.not();
        // NOT inverts confidence: if result is high (>0.5), NOT will be low (<0.5)
        let expected_not = 1.0 - result.confidence_interval.point.raw();
        assert!((not_result.confidence_interval.point.raw() - expected_not).abs() < 1e-5);
    }
}

#[test]
fn test_confidence_interval_operations_stress() {
    use engram_core::query::ConfidenceInterval;

    // Test thousands of confidence interval operations
    for _ in 0..10_000 {
        let interval_a =
            ConfidenceInterval::from_confidence_with_uncertainty(Confidence::exact(0.7), 0.1);
        let interval_b =
            ConfidenceInterval::from_confidence_with_uncertainty(Confidence::exact(0.5), 0.15);

        let and_result = interval_a.and(&interval_b);
        let or_result = interval_a.or(&interval_b);
        let not_result = interval_a.not();

        // Verify all results are valid probabilities
        assert!(and_result.point.raw() >= 0.0 && and_result.point.raw() <= 1.0);
        assert!(or_result.point.raw() >= 0.0 && or_result.point.raw() <= 1.0);
        assert!(not_result.point.raw() >= 0.0 && not_result.point.raw() <= 1.0);
    }
}

#[test]
#[ignore = "Run with --ignored flag - takes ~30 seconds"]
fn test_100k_query_operations() {
    let executor = ProbabilisticQueryExecutor::default();

    println!("Running 100K query operations...");

    for iteration in 0..100_000 {
        let episodes = vec![(
            create_test_episode_for_stress(iteration % 1000, Confidence::MEDIUM),
            Confidence::MEDIUM,
        )];

        let result = executor.execute(episodes, &[], vec![]);
        assert_eq!(result.len(), 1);

        // Log progress every 10K queries
        if iteration % 10_000 == 0 {
            println!("  Completed {iteration} queries");
        }
    }

    println!("100K query operations completed successfully");
}

#[test]
fn test_deep_query_nesting() {
    let executor = ProbabilisticQueryExecutor::default();

    // Create deeply nested query structure
    // Use OR instead of AND so results don't become empty through intersections
    let mut results: Vec<ProbabilisticQueryResult> = (0..10)
        .map(|i| {
            let episodes = vec![(
                create_test_episode_for_stress(i, Confidence::MEDIUM),
                Confidence::MEDIUM,
            )];
            executor.execute(episodes, &[], vec![])
        })
        .collect();

    // Combine results in a tree structure using OR (depth 3)
    while results.len() > 1 {
        let mut next_level = Vec::new();

        for chunk in results.chunks(2) {
            if chunk.len() == 2 {
                // Use OR to union episodes instead of AND which intersects
                next_level.push(chunk[0].or(&chunk[1]));
            } else {
                next_level.push(chunk[0].clone());
            }
        }

        results = next_level;
    }

    // Verify final result
    let final_result = &results[0];
    assert!(!final_result.is_empty());
    assert!(final_result.confidence_interval.point.raw() >= 0.0);
    assert!(final_result.confidence_interval.point.raw() <= 1.0);
}
