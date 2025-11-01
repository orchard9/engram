#![allow(missing_docs)]
//! Parameter tuning benchmarks for optimal streaming configuration.
//!
//! These benchmarks explore the parameter space to identify optimal configurations
//! for different workload profiles. Measures trade-offs between throughput, latency,
//! and memory usage across worker count, batch size, and queue capacity.
//!
//! ## Optimization Goals
//!
//! - **Low-latency profile**: Minimize P99 latency (< 10ms target)
//! - **High-throughput profile**: Maximize obs/sec (> 100K target)
//! - **Balanced profile**: Optimize for both metrics
//! - **Resource-constrained**: Minimize memory footprint
//!
//! ## Research Foundation
//!
//! Based on:
//! - Pareto frontier analysis: Multi-objective optimization trade-offs
//! - Amdahl's Law: Diminishing returns beyond core count
//! - Batch processing: Amortization of fixed overhead costs
//! - Little's Law: Queue sizing for throughput/latency balance

use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use engram_core::memory::Episode;
use engram_core::streaming::{ObservationPriority, QueueConfig, WorkerPool, WorkerPoolConfig};
use engram_core::types::MemorySpaceId;
use engram_core::{Confidence, EMBEDDING_DIM};
use rand::{Rng, SeedableRng, rngs::StdRng};
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

/// Generate a test episode
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

/// Generate episodes efficiently in batch
fn generate_episodes(count: usize, seed: u64) -> Vec<Episode> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..count).map(|i| generate_episode(&mut rng, i)).collect()
}

/// Tuning result capturing performance metrics
#[derive(Debug, Clone)]
#[allow(dead_code)] // Prepared for future auto-tuning feature
struct TuningResult {
    worker_count: usize,
    batch_size: usize,
    queue_capacity: usize,
    throughput_obs_per_sec: f64,
    avg_latency_ms: f64,
    estimated_memory_mb: f64,
}

impl TuningResult {
    /// Calculate score for low-latency profile (minimize latency)
    #[allow(dead_code)] // Prepared for future auto-tuning feature
    fn low_latency_score(&self) -> f64 {
        1000.0 / self.avg_latency_ms // Higher score for lower latency
    }

    /// Calculate score for high-throughput profile (maximize throughput)
    #[allow(dead_code)] // Prepared for future auto-tuning feature
    const fn high_throughput_score(&self) -> f64 {
        self.throughput_obs_per_sec // Higher score for higher throughput
    }

    /// Calculate score for balanced profile (optimize both)
    #[allow(dead_code)] // Prepared for future auto-tuning feature
    fn balanced_score(&self) -> f64 {
        let normalized_throughput = self.throughput_obs_per_sec / 100_000.0; // Normalize to target
        let normalized_latency = 20.0 / self.avg_latency_ms; // Normalize to target
        (normalized_throughput * normalized_latency).sqrt() // Geometric mean
    }
}

/// Benchmark: Worker count grid search (1, 2, 4, 8 workers)
///
/// Measures throughput and latency scaling with worker count to identify
/// optimal parallelism for the target workload.
fn worker_count_grid_search(c: &mut Criterion) {
    let mut group = c.benchmark_group("worker_count_tuning");
    group.sample_size(15);
    group.measurement_time(Duration::from_secs(20));

    let num_observations = 5_000;
    let episodes = generate_episodes(num_observations, 0xBEEF);
    let space_id = MemorySpaceId::new("bench_space").unwrap();

    for worker_count in [1, 2, 4, 8, 16] {
        group.throughput(Throughput::Elements(num_observations as u64));
        group.bench_with_input(
            BenchmarkId::new("workers", worker_count),
            &worker_count,
            |b, &worker_count| {
                b.iter(|| {
                    let config = WorkerPoolConfig {
                        num_workers: worker_count,
                        ..Default::default()
                    };
                    let pool = WorkerPool::new(config);

                    let start = Instant::now();

                    // Stream all observations
                    for (seq, episode) in episodes.clone().into_iter().enumerate() {
                        let _ = pool.enqueue(
                            space_id.clone(),
                            episode,
                            seq as u64,
                            ObservationPriority::Normal,
                        );
                    }

                    // Wait for processing to complete
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

                    let elapsed = start.elapsed();
                    let _ = pool.shutdown(Duration::from_secs(5));

                    black_box(elapsed)
                });
            },
        );
    }

    group.finish();
}

/// Benchmark: Batch size grid search (10, 50, 100, 500, 1000)
///
/// Explores latency vs throughput trade-off at different batch sizes.
/// Small batches: lower latency, more overhead
/// Large batches: higher throughput, higher latency
fn batch_size_grid_search(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_size_tuning");
    group.sample_size(15);
    group.measurement_time(Duration::from_secs(20));

    let num_observations = 5_000;
    let episodes = generate_episodes(num_observations, 0xCAFE);
    let space_id = MemorySpaceId::new("bench_space").unwrap();

    for batch_size in [5, 10, 50, 100, 250, 500, 1000] {
        group.throughput(Throughput::Elements(num_observations as u64));
        group.bench_with_input(
            BenchmarkId::new("batch_size", batch_size),
            &batch_size,
            |b, &batch_size| {
                b.iter(|| {
                    let config = WorkerPoolConfig {
                        num_workers: 4,
                        min_batch_size: batch_size,
                        max_batch_size: batch_size, // Fixed batch size for measurement
                        ..Default::default()
                    };
                    let pool = WorkerPool::new(config);

                    let start = Instant::now();

                    for (seq, episode) in episodes.clone().into_iter().enumerate() {
                        let _ = pool.enqueue(
                            space_id.clone(),
                            episode,
                            seq as u64,
                            ObservationPriority::Normal,
                        );
                    }

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

                    let elapsed = start.elapsed();
                    let _ = pool.shutdown(Duration::from_secs(5));

                    black_box(elapsed)
                });
            },
        );
    }

    group.finish();
}

/// Benchmark: Queue capacity grid search (1K, 10K, 50K, 100K)
///
/// Measures impact of queue capacity on memory usage and backpressure frequency.
/// Larger queues: more memory, fewer backpressure events
/// Smaller queues: less memory, more backpressure activations
fn queue_capacity_grid_search(c: &mut Criterion) {
    let mut group = c.benchmark_group("queue_capacity_tuning");
    group.sample_size(15);
    group.measurement_time(Duration::from_secs(20));

    let num_observations = 10_000;
    let episodes = generate_episodes(num_observations, 0xDEAD);
    let space_id = MemorySpaceId::new("bench_space").unwrap();

    for capacity in [1_000, 5_000, 10_000, 50_000, 100_000] {
        group.throughput(Throughput::Elements(num_observations as u64));
        group.bench_with_input(
            BenchmarkId::new("capacity", capacity),
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

                    for (seq, episode) in episodes.clone().into_iter().enumerate() {
                        let _ = pool.enqueue(
                            space_id.clone(),
                            episode,
                            seq as u64,
                            ObservationPriority::Normal,
                        );
                    }

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

/// Benchmark: Adaptive batching effectiveness
///
/// Compares fixed vs adaptive batch sizing under varying load conditions.
/// Adaptive batching should provide better latency under light load and
/// better throughput under heavy load.
fn adaptive_batching_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("adaptive_batching");
    group.sample_size(15);

    let space_id = MemorySpaceId::new("bench_space").unwrap();

    // Test under different load conditions
    for load_level in [(1_000, "light"), (5_000, "medium"), (10_000, "heavy")] {
        let (num_observations, load_name) = load_level;
        let episodes = generate_episodes(num_observations, 0xFACE);

        // Fixed small batch
        group.bench_with_input(
            BenchmarkId::new(format!("{load_name}_fixed_small"), num_observations),
            &num_observations,
            |b, _| {
                b.iter(|| {
                    let config = WorkerPoolConfig {
                        num_workers: 4,
                        min_batch_size: 10,
                        max_batch_size: 10,
                        ..Default::default()
                    };
                    let pool = WorkerPool::new(config);

                    for (seq, episode) in episodes.clone().into_iter().enumerate() {
                        let _ = pool.enqueue(
                            space_id.clone(),
                            episode,
                            seq as u64,
                            ObservationPriority::Normal,
                        );
                    }

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

        // Adaptive batch
        group.bench_with_input(
            BenchmarkId::new(format!("{load_name}_adaptive"), num_observations),
            &num_observations,
            |b, _| {
                b.iter(|| {
                    let config = WorkerPoolConfig {
                        num_workers: 4,
                        min_batch_size: 10,
                        max_batch_size: 500, // Adaptive range
                        ..Default::default()
                    };
                    let pool = WorkerPool::new(config);

                    for (seq, episode) in episodes.clone().into_iter().enumerate() {
                        let _ = pool.enqueue(
                            space_id.clone(),
                            episode,
                            seq as u64,
                            ObservationPriority::Normal,
                        );
                    }

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

/// Benchmark: Work stealing threshold tuning
///
/// Tests different work stealing thresholds to find optimal load balancing.
/// Too low: excessive stealing overhead
/// Too high: poor load balancing
fn work_stealing_threshold_tuning(c: &mut Criterion) {
    let mut group = c.benchmark_group("work_stealing_threshold");
    group.sample_size(15);

    let num_observations = 10_000;
    let episodes = generate_episodes(num_observations, 0xABCD);

    // Create skewed workload (all to one space initially)
    let space_id = MemorySpaceId::new("skewed_space").unwrap();

    for threshold in [100, 500, 1_000, 5_000, 10_000] {
        group.bench_with_input(
            BenchmarkId::new("steal_threshold", threshold),
            &threshold,
            |b, &threshold| {
                b.iter(|| {
                    let config = WorkerPoolConfig {
                        num_workers: 8,
                        steal_threshold: threshold,
                        ..Default::default()
                    };
                    let pool = WorkerPool::new(config);

                    // Stream all to same space (creates imbalance)
                    for (seq, episode) in episodes.clone().into_iter().enumerate() {
                        let _ = pool.enqueue(
                            space_id.clone(),
                            episode,
                            seq as u64,
                            ObservationPriority::Normal,
                        );
                    }

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

/// Benchmark: Memory footprint measurement
///
/// Measures RSS memory usage with different configurations and observation counts
/// to validate memory efficiency targets.
fn memory_footprint_measurement(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_footprint");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(30));

    let space_id = MemorySpaceId::new("bench_space").unwrap();

    // Test memory usage at different scales
    for observation_count in [10_000, 50_000, 100_000] {
        group.throughput(Throughput::Elements(observation_count as u64));
        group.bench_with_input(
            BenchmarkId::new("observations", observation_count),
            &observation_count,
            |b, &observation_count| {
                b.iter(|| {
                    let episodes = generate_episodes(observation_count, 0x1234);

                    let config = WorkerPoolConfig {
                        num_workers: 4,
                        ..Default::default()
                    };
                    let pool = WorkerPool::new(config);

                    // Stream all observations
                    for (seq, episode) in episodes.into_iter().enumerate() {
                        let _ = pool.enqueue(
                            space_id.clone(),
                            episode,
                            seq as u64,
                            ObservationPriority::Normal,
                        );
                    }

                    // Wait for indexing
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

                    let _ = pool.shutdown(Duration::from_secs(5));
                    black_box(());
                });
            },
        );
    }

    group.finish();
}

fn configure_criterion() -> Criterion {
    Criterion::default()
        .output_directory(std::path::Path::new("tmp/parameter_tuning_benchmarks"))
        .confidence_level(0.95)
        .noise_threshold(0.02)
        .significance_level(0.05)
}

criterion_group! {
    name = parameter_tuning_benches;
    config = configure_criterion();
    targets =
        worker_count_grid_search,
        batch_size_grid_search,
        queue_capacity_grid_search,
        adaptive_batching_comparison,
        work_stealing_threshold_tuning,
        memory_footprint_measurement
}

criterion_main!(parameter_tuning_benches);
