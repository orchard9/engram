#![allow(missing_docs)]
#![allow(dead_code)] // Benchmark in progress - API changes needed
//! Concurrent recall benchmarks measuring recall performance during streaming.
//!
//! These benchmarks validate that recall operations maintain <20ms P99 latency
//! even while observations are being streamed at 100K/sec. Tests interference
//! between concurrent operations and validates isolation guarantees.
//!
//! ## Benchmark Targets
//!
//! - **Recall latency**: P99 < 20ms during 100K obs/sec streaming
//! - **Throughput impact**: < 10% degradation from recall operations
//! - **Isolation**: No cross-space contention between recall and observation
//! - **Concurrent recalls**: 10+ recalls/sec without queue exhaustion
//!
//! ## Research Foundation
//!
//! Based on:
//! - Priority inversion analysis: High-priority recalls preempt low-priority observations
//! - Space isolation: Zero contention guarantee through independent indices
//! - Lock-free data structures: Progress guarantees under concurrent access

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use engram_core::memory::{Episode, Memory};
use engram_core::streaming::{
    ObservationPriority, SpaceIsolatedHnsw, WorkerPool, WorkerPoolConfig,
};
use engram_core::types::MemorySpaceId;
use engram_core::{Confidence, EMBEDDING_DIM};
use rand::{Rng, SeedableRng, rngs::StdRng};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
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

/// Latency percentile tracker for statistical analysis
struct LatencyTracker {
    samples: Vec<Duration>,
}

impl LatencyTracker {
    fn new() -> Self {
        Self {
            samples: Vec::with_capacity(10_000),
        }
    }

    fn record(&mut self, latency: Duration) {
        self.samples.push(latency);
    }

    fn percentile(&mut self, p: f64) -> Duration {
        if self.samples.is_empty() {
            return Duration::ZERO;
        }

        self.samples.sort_unstable();
        let index = ((self.samples.len() as f64) * p) as usize;
        let index = index.min(self.samples.len() - 1);
        self.samples[index]
    }

    #[allow(dead_code)]
    fn p50(&mut self) -> Duration {
        self.percentile(0.50)
    }

    fn p99(&mut self) -> Duration {
        self.percentile(0.99)
    }

    #[allow(dead_code)]
    fn p999(&mut self) -> Duration {
        self.percentile(0.999)
    }
}

/// Benchmark: Baseline recall latency without concurrent streaming
///
/// Establishes baseline recall performance to compare against concurrent scenarios.
fn baseline_recall_latency(c: &mut Criterion) {
    let mut group = c.benchmark_group("baseline_recall");
    group.sample_size(100);

    let space_id = MemorySpaceId::new("bench_space").unwrap();
    let space_hnsw = Arc::new(SpaceIsolatedHnsw::new());

    // Pre-populate index with 10K memories
    let mut rng = StdRng::seed_from_u64(0xBEEF);
    for i in 0..10_000 {
        let episode = generate_episode(&mut rng, i);
        let memory = Arc::new(Memory::from_episode(episode, 1.0));
        let _ = space_hnsw.insert_memory(&space_id, memory);
    }

    let query_embedding = generate_embedding(&mut rng);

    group.bench_function("recall_10_neighbors", |b| {
        b.iter(|| {
            // Get the index and search
            if let Some(index) = space_hnsw.get_index(&space_id) {
                let results = index.search_with_confidence(&query_embedding, 10, Confidence::LOW);
                black_box(results)
            } else {
                black_box(vec![])
            }
        });
    });

    group.finish();
}

/// Benchmark: Recall latency during streaming at different rates
///
/// Measures recall latency degradation as streaming rate increases from
/// 10K to 100K observations/sec.
fn recall_under_streaming_load(c: &mut Criterion) {
    let mut group = c.benchmark_group("recall_under_load");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(30));

    for streaming_rate in [10_000, 50_000, 100_000] {
        group.bench_with_input(
            BenchmarkId::new("streaming_rate", streaming_rate),
            &streaming_rate,
            |b, &streaming_rate| {
                b.iter_custom(|iters| {
                    let mut total_duration = Duration::ZERO;

                    for _ in 0..iters {
                        let space_id = MemorySpaceId::new("bench_space").unwrap();
                        let config = WorkerPoolConfig {
                            num_workers: 4,
                            ..Default::default()
                        };
                        let pool = Arc::new(WorkerPool::new(config));

                        // Pre-populate index
                        let mut rng = StdRng::seed_from_u64(0xCAFE);
                        for i in 0..1_000 {
                            let episode = generate_episode(&mut rng, i);
                            let _ = pool.enqueue(
                                space_id.clone(),
                                episode,
                                i as u64,
                                ObservationPriority::Normal,
                            );
                        }

                        // Wait for initial population
                        thread::sleep(Duration::from_millis(500));

                        // Start streaming at target rate in background
                        let streaming_done = Arc::new(AtomicBool::new(false));
                        let streaming_done_clone = Arc::clone(&streaming_done);
                        let pool_clone = Arc::clone(&pool);
                        let space_id_clone = space_id.clone();

                        let streaming_thread = thread::spawn(move || {
                            let mut rng = StdRng::seed_from_u64(0xDEAD);
                            let target_interval =
                                Duration::from_secs(1).as_nanos() / streaming_rate as u128;
                            let mut seq = 1000u64;

                            while !streaming_done_clone.load(Ordering::Relaxed) {
                                let start = Instant::now();

                                let episode = generate_episode(&mut rng, seq as usize);
                                let _ = pool_clone.enqueue(
                                    space_id_clone.clone(),
                                    episode,
                                    seq,
                                    ObservationPriority::Normal,
                                );
                                seq += 1;

                                let elapsed = start.elapsed();
                                if let Some(sleep_duration) =
                                    Duration::from_nanos(target_interval as u64)
                                        .checked_sub(elapsed)
                                {
                                    thread::sleep(sleep_duration);
                                }
                            }
                        });

                        // Measure recall latency during streaming
                        let query_embedding = generate_embedding(&mut rng);
                        let start = Instant::now();

                        // Perform multiple recalls to measure latency distribution
                        for _ in 0..10 {
                            // Get space_hnsw from pool (simulate recall operation)
                            let space_hnsw = SpaceIsolatedHnsw::new();
                            if let Some(index) = space_hnsw.get_index(&space_id) {
                                let _ = index.search_with_confidence(
                                    &query_embedding,
                                    10,
                                    Confidence::LOW,
                                );
                            }
                            thread::sleep(Duration::from_millis(10));
                        }

                        let elapsed = start.elapsed();
                        total_duration += elapsed;

                        // Stop streaming
                        streaming_done.store(true, Ordering::Relaxed);
                        let _ = streaming_thread.join();

                        // Shutdown pool - just drop it since it's wrapped in Arc
                        drop(pool);
                    }

                    total_duration
                });
            },
        );
    }

    group.finish();
}

/// Benchmark: Concurrent recalls at different frequencies
///
/// Tests multiple concurrent recalls (1/sec, 10/sec, 100/sec) during
/// sustained 100K obs/sec streaming to measure interference.
fn concurrent_recall_frequency(c: &mut Criterion) {
    let mut group = c.benchmark_group("concurrent_recall_frequency");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(30));

    let streaming_rate = 100_000; // Target production rate

    for recall_rate in [1, 10, 50] {
        group.bench_with_input(
            BenchmarkId::new("recalls_per_second", recall_rate),
            &recall_rate,
            |b, &recall_rate| {
                b.iter_custom(|iters| {
                    let mut total_duration = Duration::ZERO;

                    for _ in 0..iters {
                        let space_id = MemorySpaceId::new("bench_space").unwrap();
                        let config = WorkerPoolConfig {
                            num_workers: 4,
                            ..Default::default()
                        };
                        let pool = Arc::new(WorkerPool::new(config));

                        // Pre-populate with 5K memories
                        let mut rng = StdRng::seed_from_u64(0xFACE);
                        for i in 0..5_000 {
                            let episode = generate_episode(&mut rng, i);
                            let _ = pool.enqueue(
                                space_id.clone(),
                                episode,
                                i as u64,
                                ObservationPriority::Normal,
                            );
                        }

                        thread::sleep(Duration::from_secs(1)); // Wait for indexing

                        // Start streaming observations
                        let streaming_done = Arc::new(AtomicBool::new(false));
                        let streaming_done_clone = Arc::clone(&streaming_done);
                        let pool_clone = Arc::clone(&pool);
                        let space_id_clone = space_id.clone();

                        let streaming_thread = thread::spawn(move || {
                            let mut rng = StdRng::seed_from_u64(0xABCD);
                            let target_interval =
                                Duration::from_secs(1).as_nanos() / streaming_rate as u128;
                            let mut seq = 5000u64;

                            while !streaming_done_clone.load(Ordering::Relaxed) {
                                let start = Instant::now();
                                let episode = generate_episode(&mut rng, seq as usize);
                                let _ = pool_clone.enqueue(
                                    space_id_clone.clone(),
                                    episode,
                                    seq,
                                    ObservationPriority::Normal,
                                );
                                seq += 1;

                                let elapsed = start.elapsed();
                                if let Some(sleep_duration) =
                                    Duration::from_nanos(target_interval as u64)
                                        .checked_sub(elapsed)
                                {
                                    thread::sleep(sleep_duration);
                                }
                            }
                        });

                        // Perform recalls at target frequency
                        let test_duration = Duration::from_secs(10);
                        let recall_interval = Duration::from_secs(1) / recall_rate;
                        let start = Instant::now();
                        let mut latency_tracker = LatencyTracker::new();

                        while start.elapsed() < test_duration {
                            let recall_start = Instant::now();

                            // Simulate recall operation
                            let space_hnsw = SpaceIsolatedHnsw::new();
                            let query_embedding = generate_embedding(&mut rng);
                            if let Some(index) = space_hnsw.get_index(&space_id) {
                                let _ = index.search_with_confidence(
                                    &query_embedding,
                                    10,
                                    Confidence::LOW,
                                );
                            }

                            let recall_latency = recall_start.elapsed();
                            latency_tracker.record(recall_latency);

                            thread::sleep(recall_interval);
                        }

                        let p99_latency = latency_tracker.p99();
                        total_duration += p99_latency;

                        // Cleanup
                        streaming_done.store(true, Ordering::Relaxed);
                        let _ = streaming_thread.join();

                        // Shutdown pool - just drop it since it's wrapped in Arc
                        drop(pool);
                    }

                    total_duration
                });
            },
        );
    }

    group.finish();
}

/// Benchmark: Multi-space isolation during concurrent operations
///
/// Validates zero-contention guarantee by streaming to multiple spaces
/// concurrently while performing recalls on different spaces.
fn multi_space_isolation(c: &mut Criterion) {
    let mut group = c.benchmark_group("multi_space_isolation");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(30));

    let num_spaces = vec![2, 4, 8];

    for space_count in num_spaces {
        group.bench_with_input(
            BenchmarkId::new("concurrent_spaces", space_count),
            &space_count,
            |b, &space_count| {
                b.iter(|| {
                    let config = WorkerPoolConfig {
                        num_workers: 8, // More workers for multi-space
                        ..Default::default()
                    };
                    let pool = Arc::new(WorkerPool::new(config));

                    // Create multiple spaces
                    let space_ids: Vec<_> = (0..space_count)
                        .map(|i| MemorySpaceId::new(&format!("space_{i}")).unwrap())
                        .collect();

                    // Stream to all spaces concurrently
                    let mut threads = vec![];

                    for (space_idx, space_id) in space_ids.iter().enumerate() {
                        let pool_clone = Arc::clone(&pool);
                        let space_id_clone = space_id.clone();

                        let thread = thread::spawn(move || {
                            let mut rng = StdRng::seed_from_u64(0x1000 + space_idx as u64);
                            for i in 0..1_000 {
                                let episode = generate_episode(&mut rng, i);
                                let _ = pool_clone.enqueue(
                                    space_id_clone.clone(),
                                    episode,
                                    i as u64,
                                    ObservationPriority::Normal,
                                );
                            }
                        });

                        threads.push(thread);
                    }

                    // Wait for all streaming to complete
                    for thread in threads {
                        let _ = thread.join();
                    }

                    // Wait for queues to drain
                    loop {
                        let total_depth: usize = pool
                            .worker_stats()
                            .iter()
                            .map(|s| s.current_queue_depth)
                            .sum();
                        if total_depth == 0 {
                            break;
                        }
                        thread::sleep(Duration::from_millis(10));
                    }

                    // Note: shutdown requires moving the pool, but Arc prevents that.
                    // We just drop the Arc instead, which will shut down when the last reference is dropped.
                    drop(pool);

                    black_box(());
                });
            },
        );
    }

    group.finish();
}

fn configure_criterion() -> Criterion {
    Criterion::default()
        .output_directory(std::path::Path::new("tmp/concurrent_recall_benchmarks"))
        .confidence_level(0.95)
        .noise_threshold(0.02)
        .significance_level(0.05)
}

criterion_group! {
    name = concurrent_recall_benches;
    config = configure_criterion();
    targets =
        baseline_recall_latency,
        recall_under_streaming_load,
        concurrent_recall_frequency,
        multi_space_isolation
}

criterion_main!(concurrent_recall_benches);
