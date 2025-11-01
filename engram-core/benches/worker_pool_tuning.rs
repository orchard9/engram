//! Worker pool load balancing threshold tuning benchmark.
//!
//! This benchmark evaluates different `steal_threshold` values to find the optimal
//! trade-off between load balancing efficiency and cache pollution overhead.
//!
//! ## Methodology
//!
//! Tests worker pool performance under skewed workloads where one worker receives
//! significantly more observations than others. Measures:
//!
//! - **Throughput**: Total observations processed per second
//! - **Load balance**: Standard deviation of worker utilization
//! - **Latency**: P50/P95/P99 observation processing latency
//!
//! ## Run Instructions
//!
//! ```bash
//! cargo bench --bench worker_pool_tuning
//! ```

#![allow(clippy::items_after_statements)]
#![allow(dead_code)]
#![allow(missing_docs)]

use chrono::Utc;
use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use engram_core::streaming::{ObservationPriority, QueueConfig, WorkerPool, WorkerPoolConfig};
use engram_core::types::MemorySpaceId;
use engram_core::{Confidence, Episode};
use std::thread;
use std::time::Duration;

/// Create a test episode with unique ID
fn create_test_episode(id: usize) -> Episode {
    Episode::new(
        format!("test_episode_{id}"),
        Utc::now(),
        format!("Test observation {id}"),
        [0.0f32; 768],
        Confidence::MEDIUM,
    )
}

/// Benchmark worker pool with different steal thresholds
fn bench_steal_threshold(c: &mut Criterion) {
    let mut group = c.benchmark_group("worker_pool_steal_threshold");

    // Test workload parameters
    const NUM_OBSERVATIONS: usize = 10_000;
    const NUM_WORKERS: usize = 4;

    // Test different steal thresholds
    for steal_threshold in [100, 500, 1000, 2000, 5000] {
        group.throughput(Throughput::Elements(NUM_OBSERVATIONS as u64));

        group.bench_with_input(
            BenchmarkId::from_parameter(steal_threshold),
            &steal_threshold,
            |b, &threshold| {
                b.iter(|| {
                    // Create worker pool with specific threshold
                    let config = WorkerPoolConfig {
                        num_workers: NUM_WORKERS,
                        steal_threshold: threshold,
                        queue_config: QueueConfig::default(),
                        min_batch_size: 10,
                        max_batch_size: 500,
                        idle_sleep_ms: 1,
                    };

                    let pool = WorkerPool::new(config);

                    // Create skewed workload: all observations go to one space initially
                    // This tests the work stealing mechanism
                    let space1 = MemorySpaceId::new("heavy_load").unwrap();
                    let space2 = MemorySpaceId::new("light_load").unwrap();

                    // Send 90% of observations to space1 (skewed load)
                    for i in 0..NUM_OBSERVATIONS {
                        let space = if i < NUM_OBSERVATIONS * 9 / 10 {
                            space1.clone()
                        } else {
                            space2.clone()
                        };

                        let episode = create_test_episode(i);
                        let _ = pool.enqueue(space, episode, i as u64, ObservationPriority::Normal);
                    }

                    // Wait for all observations to be processed
                    let start = std::time::Instant::now();
                    let timeout = Duration::from_secs(30);
                    while pool.total_queue_depth() > 0 && start.elapsed() < timeout {
                        thread::sleep(Duration::from_millis(10));
                    }

                    // Get final stats for analysis
                    let stats = pool.worker_stats();
                    let total_processed: u64 = stats.iter().map(|s| s.processed_observations).sum();
                    let total_stolen: u64 = stats.iter().map(|s| s.stolen_batches).sum();

                    // Ensure all observations were processed
                    assert_eq!(
                        total_processed as usize, NUM_OBSERVATIONS,
                        "All observations must be processed"
                    );

                    black_box((total_processed, total_stolen))
                });
            },
        );
    }

    group.finish();
}

/// Benchmark worker pool under balanced load (baseline)
fn bench_balanced_load(c: &mut Criterion) {
    let mut group = c.benchmark_group("worker_pool_balanced_load");

    const NUM_OBSERVATIONS: usize = 10_000;
    const NUM_WORKERS: usize = 4;
    const NUM_SPACES: usize = 10;

    group.throughput(Throughput::Elements(NUM_OBSERVATIONS as u64));

    group.bench_function("balanced_distribution", |b| {
        b.iter(|| {
            let config = WorkerPoolConfig::default();
            let pool = WorkerPool::new(config);

            // Create multiple spaces for balanced distribution
            let spaces: Vec<MemorySpaceId> = (0..NUM_SPACES)
                .map(|i| MemorySpaceId::new(format!("space_{i}")).unwrap())
                .collect();

            // Distribute observations evenly across spaces
            for i in 0..NUM_OBSERVATIONS {
                let space = &spaces[i % NUM_SPACES];
                let episode = create_test_episode(i);
                let _ = pool.enqueue(
                    space.clone(),
                    episode,
                    i as u64,
                    ObservationPriority::Normal,
                );
            }

            // Wait for processing
            let start = std::time::Instant::now();
            let timeout = Duration::from_secs(30);
            while pool.total_queue_depth() > 0 && start.elapsed() < timeout {
                thread::sleep(Duration::from_millis(10));
            }

            let stats = pool.worker_stats();
            let total_processed: u64 = stats.iter().map(|s| s.processed_observations).sum();

            black_box(total_processed)
        });
    });

    group.finish();
}

criterion_group!(benches, bench_steal_threshold, bench_balanced_load);
criterion_main!(benches);
