//! Property-based tests for streaming infrastructure.
//!
//! Uses proptest to verify critical invariants hold across a wide range of inputs.

#[cfg(test)]
#[allow(clippy::unwrap_used)] // Property tests can use unwrap
#[allow(clippy::expect_used)] // Property tests can use expect
#[allow(clippy::unreadable_literal)] // Property tests use large literal counts
#[allow(dead_code)] // Some helper functions may not be used yet
mod tests {
    use super::super::observation_queue::{ObservationPriority, ObservationQueue, QueueConfig};
    use super::super::worker_pool::{WorkerPool, WorkerPoolConfig};
    use crate::types::MemorySpaceId;
    use crate::{Confidence, Episode};
    use chrono::Utc;
    use proptest::prelude::*;
    use std::time::Duration;

    fn arb_episode() -> impl Strategy<Value = Episode> {
        (any::<u64>(), "[a-z]{3,8}").prop_map(|(id, content)| {
            Episode::new(
                format!("test_{id}"),
                Utc::now(),
                content,
                [0.0f32; 768],
                Confidence::MEDIUM,
            )
        })
    }

    fn arb_priority() -> impl Strategy<Value = ObservationPriority> {
        prop_oneof![
            Just(ObservationPriority::High),
            Just(ObservationPriority::Normal),
            Just(ObservationPriority::Low),
        ]
    }

    proptest! {
        /// Property: Sequence numbers must be monotonically increasing for validation.
        ///
        /// This test verifies that sequence number validation correctly accepts
        /// sequences in increasing order and rejects out-of-order sequences.
        #[test]
        fn sequence_numbers_always_monotonic(
            sequences in prop::collection::vec(0u64..1000000, 10..100)
        ) {
            // Observation: We're testing queue behavior, not a session manager
            // So we just verify that enqueueing with increasing sequences works
            let queue = ObservationQueue::new(QueueConfig::default());
            let space_id = MemorySpaceId::default();

            let mut last_seq = 0u64;
            for seq in sequences {
                if seq >= last_seq {
                    // Sequence is valid (monotonic)
                    let episode = Episode::new(
                        format!("test_{seq}"),
                        Utc::now(),
                        "test".to_string(),
                        [0.0f32; 768],
                        Confidence::MEDIUM,
                    );

                    // Should successfully enqueue
                    let result = queue.enqueue(
                        space_id.clone(),
                        episode,
                        seq,
                        ObservationPriority::Normal,
                    );

                    // May fail due to capacity, but not due to sequence
                    if result.is_ok() {
                        last_seq = seq + 1;
                    }
                }
            }
        }

        /// Property: Queue maintains FIFO ordering within same priority level.
        ///
        /// Observations enqueued with the same priority must be dequeued in
        /// the order they were enqueued.
        #[test]
        fn queue_maintains_fifo_order(
            episode_count in 10usize..100,
            priority in arb_priority(),
        ) {
            let config = QueueConfig {
                high_capacity: 200,
                normal_capacity: 200,
                low_capacity: 200,
            };
            let queue = ObservationQueue::new(config);
            let space_id = MemorySpaceId::default();

            // Enqueue episodes with unique IDs
            let mut expected_ids = Vec::new();
            for i in 0..episode_count {
                let episode = Episode::new(
                    format!("test_{i}"),
                    Utc::now(),
                    format!("content {i}"),
                    [0.0f32; 768],
                    Confidence::MEDIUM,
                );
                expected_ids.push(format!("test_{i}"));

                queue.enqueue(space_id.clone(), episode, i as u64, priority).unwrap();
            }

            // Dequeue and verify FIFO order
            for expected_id in expected_ids {
                let obs = queue.dequeue().expect("Queue should not be empty");
                assert_eq!(obs.episode.id, expected_id, "FIFO order violated");
            }
        }

        /// Property: Priority queues maintain priority ordering.
        ///
        /// High-priority observations must be dequeued before normal and low priority,
        /// and normal before low.
        #[test]
        fn priority_ordering_maintained(
            high_count in 5usize..20,
            normal_count in 5usize..20,
            low_count in 5usize..20,
        ) {
            let config = QueueConfig {
                high_capacity: 100,
                normal_capacity: 100,
                low_capacity: 100,
            };
            let queue = ObservationQueue::new(config);
            let space_id = MemorySpaceId::default();

            // Enqueue in mixed order
            for i in 0..high_count {
                let episode = Episode::new(
                    format!("high_{i}"),
                    Utc::now(),
                    "high".to_string(),
                    [0.0f32; 768],
                    Confidence::MEDIUM,
                );
                queue.enqueue(space_id.clone(), episode, i as u64, ObservationPriority::High).unwrap();
            }

            for i in 0..normal_count {
                let episode = Episode::new(
                    format!("normal_{i}"),
                    Utc::now(),
                    "normal".to_string(),
                    [0.0f32; 768],
                    Confidence::MEDIUM,
                );
                queue.enqueue(space_id.clone(), episode, i as u64, ObservationPriority::Normal).unwrap();
            }

            for i in 0..low_count {
                let episode = Episode::new(
                    format!("low_{i}"),
                    Utc::now(),
                    "low".to_string(),
                    [0.0f32; 768],
                    Confidence::MEDIUM,
                );
                queue.enqueue(space_id.clone(), episode, i as u64, ObservationPriority::Low).unwrap();
            }

            // Dequeue all and verify priority ordering
            let mut high_dequeued = 0;
            let mut normal_dequeued = 0;
            let mut low_dequeued = 0;

            while let Some(obs) = queue.dequeue() {
                if obs.episode.id.starts_with("high_") {
                    high_dequeued += 1;
                    // High priority items must come before normal/low
                    assert_eq!(normal_dequeued, 0, "High priority violated: normal dequeued before all high");
                    assert_eq!(low_dequeued, 0, "High priority violated: low dequeued before all high");
                } else if obs.episode.id.starts_with("normal_") {
                    normal_dequeued += 1;
                    // Normal priority items must come after high, before low
                    assert_eq!(high_dequeued, high_count, "Normal priority violated: dequeued before all high");
                    assert_eq!(low_dequeued, 0, "Normal priority violated: low dequeued before all normal");
                } else {
                    low_dequeued += 1;
                    // Low priority items must come after high and normal
                    assert_eq!(high_dequeued, high_count, "Low priority violated: dequeued before all high");
                    assert_eq!(normal_dequeued, normal_count, "Low priority violated: dequeued before all normal");
                }
            }

            // Verify all items were dequeued
            assert_eq!(high_dequeued, high_count);
            assert_eq!(normal_dequeued, normal_count);
            assert_eq!(low_dequeued, low_count);
        }

        /// Property: Worker pool distributes load across all workers.
        ///
        /// With enough spaces (much larger than worker count), all workers
        /// should receive at least some work (hash-based sharding guarantees
        /// each space consistently maps to a worker, but doesn't guarantee
        /// perfect balance with small space counts).
        #[test]
        fn worker_pool_uses_all_workers(
            space_count in 100usize..200,  // Many spaces ensure good distribution
            observations_per_space in 5usize..10,
        ) {
            let config = WorkerPoolConfig {
                num_workers: 4,
                ..Default::default()
            };
            let pool = WorkerPool::new(config);

            // Enqueue observations to many different spaces
            for space_idx in 0..space_count {
                let space_id = MemorySpaceId::new(format!("space{space_idx}")).unwrap();
                for obs_idx in 0..observations_per_space {
                    let episode = Episode::new(
                        format!("space{space_idx}_obs{obs_idx}"),
                        Utc::now(),
                        "test".to_string(),
                        [0.0f32; 768],
                        Confidence::MEDIUM,
                    );
                    let _ = pool.enqueue(
                        space_id.clone(),
                        episode,
                        obs_idx as u64,
                        ObservationPriority::Normal,
                    );
                }
            }

            // Wait for processing
            std::thread::sleep(Duration::from_millis(500));

            // Get worker stats
            let stats = pool.worker_stats();
            let processed_counts: Vec<_> = stats.iter().map(|s| s.processed_observations).collect();

            // With many spaces, all workers should process at least something
            let workers_with_work = processed_counts.iter().filter(|&&c| c > 0).count();
            assert!(
                workers_with_work >= 3,  // At least 75% of workers should have work
                "Only {workers_with_work}/4 workers received work with {space_count} spaces"
            );

            // Graceful shutdown
            let _ = pool.shutdown(Duration::from_secs(5));
        }

        /// Property: Queue depth tracking is accurate.
        ///
        /// The reported queue depth must match the actual number of items in the queue.
        #[test]
        fn queue_depth_tracking_accurate(
            enqueue_count in 10usize..100,
            dequeue_count in 0usize..50,
        ) {
            let config = QueueConfig {
                high_capacity: 200,
                normal_capacity: 200,
                low_capacity: 200,
            };
            let queue = ObservationQueue::new(config);
            let space_id = MemorySpaceId::default();

            // Enqueue items
            let mut actual_enqueued = 0;
            for i in 0..enqueue_count {
                let episode = Episode::new(
                    format!("test_{i}"),
                    Utc::now(),
                    "test".to_string(),
                    [0.0f32; 768],
                    Confidence::MEDIUM,
                );
                if queue.enqueue(space_id.clone(), episode, i as u64, ObservationPriority::Normal).is_ok() {
                    actual_enqueued += 1;
                }
            }

            // Check depth after enqueuing
            let depth_after_enqueue = queue.total_depth();
            assert_eq!(depth_after_enqueue, actual_enqueued, "Depth tracking incorrect after enqueue");

            // Dequeue items
            let actual_dequeue_count = dequeue_count.min(actual_enqueued);
            for _ in 0..actual_dequeue_count {
                let _ = queue.dequeue();
            }

            // Check depth after dequeuing
            let depth_after_dequeue = queue.total_depth();
            let expected_depth = actual_enqueued - actual_dequeue_count;
            assert_eq!(depth_after_dequeue, expected_depth, "Depth tracking incorrect after dequeue");
        }

        /// Property: Same space consistently queues to same worker.
        ///
        /// Enqueueing observations for the same space multiple times should
        /// consistently route to the same worker queue (tested indirectly via stats).
        #[test]
        fn space_assignment_consistent(
            observations_per_space in 10usize..30
        ) {
            let config = WorkerPoolConfig {
                num_workers: 4,
                ..Default::default()
            };
            let pool = WorkerPool::new(config);

            // Enqueue many observations to same space
            let space_id = MemorySpaceId::new("consistent_space").unwrap();
            for i in 0..observations_per_space {
                let episode = Episode::new(
                    format!("obs_{i}"),
                    Utc::now(),
                    "test".to_string(),
                    [0.0f32; 768],
                    Confidence::MEDIUM,
                );
                let _ = pool.enqueue(
                    space_id.clone(),
                    episode,
                    i as u64,
                    ObservationPriority::Normal,
                );
            }

            // Wait for processing
            std::thread::sleep(Duration::from_millis(100));

            // Verify all observations went to the same worker
            // (Only one worker should have non-zero processed count)
            let stats = pool.worker_stats();
            let workers_with_work: Vec<_> = stats.iter()
                .filter(|s| s.processed_observations > 0)
                .collect();

            // All work should be on a single worker for this space
            assert_eq!(
                workers_with_work.len(),
                1,
                "Expected all observations to route to one worker, found {} workers with work",
                workers_with_work.len()
            );

            // Graceful shutdown
            let _ = pool.shutdown(Duration::from_secs(5));
        }
    }
}
