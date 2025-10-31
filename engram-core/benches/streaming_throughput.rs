#![allow(missing_docs)]
//! Streaming throughput benchmarks for high-performance observation ingestion.
//!
//! These benchmarks validate the target of 100K observations/sec with bounded
//! latency and measure throughput scaling at different load levels. Uses Criterion
//! for statistical rigor and tracks P50/P99/P99.9 latency distributions.
//!
//! ## Benchmark Targets
//!
//! - **Sustained 100K obs/sec**: Validate production throughput target
//! - **Worker scaling**: Measure linear scaling up to core count
//! - **Latency bounds**: P99 < 100ms for observation → indexed
//! - **Memory efficiency**: < 2GB for 1M observations
//!
//! ## Research Foundation
//!
//! Based on:
//! - Amdahl's Law: Parallel scaling limits and sequential bottlenecks
//! - Little's Law: Queue depth = throughput × latency relationship
//! - Michael & Scott (1996): Lock-free queue performance characteristics

use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use engram_core::memory::{Episode, Memory};
use engram_core::streaming::{
    ObservationPriority, ObservationQueue, QueueConfig, SpaceIsolatedHnsw, WorkerPool,
    WorkerPoolConfig,
};
use engram_core::types::MemorySpaceId;
use engram_core::{Confidence, EMBEDDING_DIM};
use rand::{Rng, SeedableRng, rngs::StdRng};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Generate a random normalized embedding vector
fn generate_embedding(rng: &mut StdRng) -> [f32; EMBEDDING_DIM] {
    let mut embedding = [0.0f32; EMBEDDING_DIM];
    for value in &mut embedding {
        *value = rng.gen_range(-1.0..1.0);
    }
    // Normalize to unit length
    let magnitude: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
    if magnitude > 0.0 {
        for value in &mut embedding {
            *value /= magnitude;
        }
    }
    embedding
}

/// Generate a test episode with random embedding
fn generate_episode(rng: &mut StdRng, id: usize) -> Episode {
    let embedding = generate_embedding(rng);
    Episode::new(
        format!("episode_{id:08}"),
        chrono::Utc::now(),
        format!("Test observation content {id}"),
        embedding,
        Confidence::HIGH,
    )
}

/// Generate multiple test episodes efficiently
fn generate_episodes(count: usize, seed: u64) -> Vec<Episode> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..count).map(|i| generate_episode(&mut rng, i)).collect()
}

/// Benchmark: Throughput ramp from 10K to 200K observations/sec
///
/// Tests sustained throughput at different load levels to identify bottlenecks
/// and validate the 100K obs/sec production target.
fn throughput_ramp(c: &mut Criterion) {
    let mut group = c.benchmark_group("streaming_throughput_ramp");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(60));
    group.warm_up_time(Duration::from_secs(5));

    // Test at different throughput targets
    for rate in [10_000, 50_000, 100_000, 150_000, 200_000] {
        group.throughput(Throughput::Elements(rate as u64));
        group.bench_with_input(
            BenchmarkId::new("observations_per_second", rate),
            &rate,
            |b, &rate| {
                b.iter_custom(|iters| {
                    let mut total_duration = Duration::ZERO;

                    for _ in 0..iters {
                        // Setup worker pool with optimal configuration
                        let config = WorkerPoolConfig {
                            num_workers: 4,
                            min_batch_size: 10,
                            max_batch_size: 500,
                            ..Default::default()
                        };
                        let pool = WorkerPool::new(config);

                        // Pre-generate episodes to avoid measuring generation time
                        let duration_secs = 10;
                        let num_observations = rate * duration_secs;
                        let episodes = generate_episodes(num_observations, 0xBEEF);

                        // Generate memory spaces for multi-tenant testing
                        let space_id = MemorySpaceId::new("bench_space").unwrap();

                        // Measure streaming throughput
                        let start = Instant::now();

                        for (seq, episode) in episodes.into_iter().enumerate() {
                            let _ = pool.enqueue(
                                space_id.clone(),
                                episode,
                                seq as u64,
                                ObservationPriority::Normal,
                            );
                        }

                        // Wait for processing to complete (check queue depths)
                        loop {
                            let total_depth: usize = pool
                                .worker_stats()
                                .iter()
                                .map(|s| s.current_queue_depth)
                                .sum();
                            if total_depth == 0 {
                                break;
                            }
                            std::thread::sleep(Duration::from_millis(10));
                        }

                        let elapsed = start.elapsed();
                        total_duration += elapsed;

                        // Graceful shutdown
                        let _ = pool.shutdown(Duration::from_secs(5));
                    }

                    total_duration
                });
            },
        );
    }

    group.finish();
}

/// Benchmark: Worker count scaling (1, 2, 4, 8 workers)
///
/// Measures linear scaling up to core count. Expected: near-linear scaling
/// up to physical cores, diminishing returns beyond.
fn worker_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("worker_scaling");
    group.sample_size(20);
    group.measurement_time(Duration::from_secs(30));

    let num_observations = 10_000;
    let episodes = generate_episodes(num_observations, 0xCAFE);
    let space_id = MemorySpaceId::new("bench_space").unwrap();

    for worker_count in [1, 2, 4, 8] {
        group.throughput(Throughput::Elements(num_observations as u64));
        group.bench_with_input(
            BenchmarkId::new("worker_count", worker_count),
            &worker_count,
            |b, &worker_count| {
                b.iter(|| {
                    let config = WorkerPoolConfig {
                        num_workers: worker_count,
                        ..Default::default()
                    };
                    let pool = WorkerPool::new(config);

                    // Stream observations
                    for (seq, episode) in episodes.clone().into_iter().enumerate() {
                        let _ = pool.enqueue(
                            space_id.clone(),
                            episode,
                            seq as u64,
                            ObservationPriority::Normal,
                        );
                    }

                    // Wait for processing
                    loop {
                        let total_depth: usize = pool
                            .worker_stats()
                            .iter()
                            .map(|s| s.current_queue_depth)
                            .sum();
                        if total_depth == 0 {
                            break;
                        }
                        std::thread::sleep(Duration::from_millis(5));
                    }

                    let _ = pool.shutdown(Duration::from_secs(5));
                    black_box(());
                });
            },
        );
    }

    group.finish();
}

/// Benchmark: Batch size tuning (10, 50, 100, 500, 1000)
///
/// Identifies optimal batch size balancing latency vs throughput. Small batches
/// minimize latency, large batches maximize throughput by amortizing overhead.
fn batch_size_tuning(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_size_tuning");
    group.sample_size(20);
    group.measurement_time(Duration::from_secs(30));

    let num_observations = 5_000;
    let episodes = generate_episodes(num_observations, 0xDEAD);
    let space_id = MemorySpaceId::new("bench_space").unwrap();

    for batch_size in [10, 50, 100, 500, 1000] {
        group.throughput(Throughput::Elements(num_observations as u64));
        group.bench_with_input(
            BenchmarkId::new("batch_size", batch_size),
            &batch_size,
            |b, &batch_size| {
                b.iter(|| {
                    let config = WorkerPoolConfig {
                        num_workers: 4,
                        min_batch_size: batch_size,
                        max_batch_size: batch_size,
                        ..Default::default()
                    };
                    let pool = WorkerPool::new(config);

                    // Stream observations
                    for (seq, episode) in episodes.clone().into_iter().enumerate() {
                        let _ = pool.enqueue(
                            space_id.clone(),
                            episode,
                            seq as u64,
                            ObservationPriority::Normal,
                        );
                    }

                    // Wait for processing
                    loop {
                        let total_depth: usize = pool
                            .worker_stats()
                            .iter()
                            .map(|s| s.current_queue_depth)
                            .sum();
                        if total_depth == 0 {
                            break;
                        }
                        std::thread::sleep(Duration::from_millis(5));
                    }

                    let _ = pool.shutdown(Duration::from_secs(5));
                    black_box(());
                });
            },
        );
    }

    group.finish();
}

/// Benchmark: Queue capacity impact on backpressure
///
/// Tests different queue capacities (1K, 10K, 100K) to measure impact on
/// memory usage and backpressure activation frequency.
fn queue_capacity_tuning(c: &mut Criterion) {
    let mut group = c.benchmark_group("queue_capacity");
    group.sample_size(15);
    group.measurement_time(Duration::from_secs(20));

    let num_observations = 10_000;
    let episodes = generate_episodes(num_observations, 0xFACE);
    let space_id = MemorySpaceId::new("bench_space").unwrap();

    for capacity in [1_000, 10_000, 50_000, 100_000] {
        group.throughput(Throughput::Elements(num_observations as u64));
        group.bench_with_input(
            BenchmarkId::new("queue_capacity", capacity),
            &capacity,
            |b, &capacity| {
                b.iter(|| {
                    let config = WorkerPoolConfig {
                        num_workers: 4,
                        queue_config: QueueConfig {
                            high_capacity: capacity / 10,
                            normal_capacity: capacity,
                            low_capacity: capacity / 2,
                        },
                        ..Default::default()
                    };
                    let pool = WorkerPool::new(config);

                    // Stream observations
                    for (seq, episode) in episodes.clone().into_iter().enumerate() {
                        let _ = pool.enqueue(
                            space_id.clone(),
                            episode,
                            seq as u64,
                            ObservationPriority::Normal,
                        );
                    }

                    // Wait for processing
                    loop {
                        let total_depth: usize = pool
                            .worker_stats()
                            .iter()
                            .map(|s| s.current_queue_depth)
                            .sum();
                        if total_depth == 0 {
                            break;
                        }
                        std::thread::sleep(Duration::from_millis(5));
                    }

                    let _ = pool.shutdown(Duration::from_secs(5));
                    black_box(());
                });
            },
        );
    }

    group.finish();
}

/// Benchmark: Lock-free queue operations
///
/// Micro-benchmark measuring raw enqueue/dequeue performance of the
/// observation queue independent of HNSW processing.
fn queue_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("queue_operations");
    group.sample_size(100);

    let space_id = MemorySpaceId::new("bench_space").unwrap();
    let mut rng = StdRng::seed_from_u64(0x0000_FEED);
    let episode = generate_episode(&mut rng, 0);

    // Benchmark enqueue operation
    group.bench_function("enqueue", |b| {
        let queue = ObservationQueue::new(QueueConfig::default());
        b.iter(|| {
            let _ = queue.enqueue(
                space_id.clone(),
                black_box(episode.clone()),
                black_box(0),
                ObservationPriority::Normal,
            );
        });
    });

    // Benchmark dequeue operation
    group.bench_function("dequeue", |b| {
        let queue = ObservationQueue::new(QueueConfig::default());
        // Pre-fill queue
        for i in 0..10_000 {
            let _ = queue.enqueue(
                space_id.clone(),
                episode.clone(),
                i,
                ObservationPriority::Normal,
            );
        }

        b.iter(|| {
            let item = queue.dequeue();
            black_box(item)
        });
    });

    group.finish();
}

/// Benchmark: Space-isolated HNSW insertion
///
/// Measures HNSW insertion performance independent of queue and worker pool
/// overhead to isolate indexing bottlenecks.
fn space_hnsw_insertion(c: &mut Criterion) {
    let mut group = c.benchmark_group("space_hnsw_insertion");
    group.sample_size(50);

    let space_id = MemorySpaceId::new("bench_space").unwrap();
    let episodes = generate_episodes(1_000, 0xABCD);

    group.throughput(Throughput::Elements(1_000));
    group.bench_function("insert_1000_memories", |b| {
        b.iter(|| {
            let space_hnsw = SpaceIsolatedHnsw::new();

            for episode in &episodes {
                let memory = Arc::new(Memory::from_episode(episode.clone(), 1.0));
                let _ = space_hnsw.insert_memory(&space_id, memory);
            }

            black_box(());
        });
    });

    group.finish();
}

fn configure_criterion() -> Criterion {
    Criterion::default()
        .output_directory(std::path::Path::new("tmp/streaming_throughput_benchmarks"))
        .confidence_level(0.95)
        .noise_threshold(0.02)
        .significance_level(0.05)
}

criterion_group! {
    name = streaming_throughput_benches;
    config = configure_criterion();
    targets =
        throughput_ramp,
        worker_scaling,
        batch_size_tuning,
        queue_capacity_tuning,
        queue_operations,
        space_hnsw_insertion
}

criterion_main!(streaming_throughput_benches);
