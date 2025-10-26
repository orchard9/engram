//! Comprehensive benchmarks for HNSW batch insertion performance
//!
//! This benchmark suite validates the performance assumptions for Milestone 11 Task 003 (Worker Pool).
//! The critical concurrent benchmark determines if the worker pool can proceed as designed.

#![cfg(feature = "hnsw_index")]
#![allow(missing_docs)]

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use engram_core::index::CognitiveHnswIndex;
use engram_core::{Confidence, Memory};
use std::sync::Arc;
use std::time::Instant;

/// Helper function to create a random memory with unique ID
fn create_random_memory(id: u32) -> Memory {
    let mut embedding = [0.0f32; 768];

    // Generate pseudo-random but deterministic embedding
    for (i, elem) in embedding.iter_mut().enumerate() {
        let seed = id
            .wrapping_mul(769)
            .wrapping_add(u32::try_from(i).unwrap_or(0));
        #[allow(clippy::cast_precision_loss)]
        let normalized = (seed as f32 / u32::MAX as f32) * 2.0 - 1.0;
        *elem = normalized;
    }

    // Normalize to unit vector
    let magnitude: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
    if magnitude > 0.0 {
        for elem in &mut embedding {
            *elem /= magnitude;
        }
    }

    Memory::new(format!("memory_{id}"), embedding, Confidence::MEDIUM)
}

/// Benchmark: Sequential single inserts (baseline)
fn bench_sequential_inserts(c: &mut Criterion) {
    let mut group = c.benchmark_group("sequential_inserts");
    group.sample_size(20);

    for count in [10, 100, 500] {
        group.bench_with_input(
            BenchmarkId::new("individual", count),
            &count,
            |b, &count| {
                b.iter_custom(|iters| {
                    let mut total_duration = std::time::Duration::ZERO;

                    for _ in 0..iters {
                        let index = CognitiveHnswIndex::new();

                        let start = Instant::now();
                        for i in 0..count {
                            let memory = Arc::new(create_random_memory(i));
                            let _ = black_box(index.insert_memory(memory));
                        }
                        total_duration += start.elapsed();
                    }

                    total_duration
                });
            },
        );
    }

    group.finish();
}

/// Benchmark: Batch inserts with various batch sizes
fn bench_batch_inserts(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_inserts");
    group.sample_size(20);

    for count in [10, 100, 500] {
        group.bench_with_input(BenchmarkId::new("batch", count), &count, |b, &count| {
            b.iter_custom(|iters| {
                let mut total_duration = std::time::Duration::ZERO;

                for _ in 0..iters {
                    let index = CognitiveHnswIndex::new();

                    // Pre-create batch
                    let memories: Vec<Arc<Memory>> = (0..count)
                        .map(|i| Arc::new(create_random_memory(i)))
                        .collect();

                    let start = Instant::now();
                    let _ = black_box(index.insert_batch(&memories));
                    total_duration += start.elapsed();
                }

                total_duration
            });
        });
    }

    group.finish();
}

/// Benchmark: Measure per-item latency for batch operations
fn bench_per_item_latency(c: &mut Criterion) {
    let mut group = c.benchmark_group("per_item_latency");
    group.sample_size(20);

    for count in [10, 100, 500, 1000] {
        // Sequential baseline
        group.bench_with_input(
            BenchmarkId::new("sequential_per_item", count),
            &count,
            |b, &count| {
                b.iter_custom(|iters| {
                    let mut total_duration = std::time::Duration::ZERO;

                    for _ in 0..iters {
                        let index = CognitiveHnswIndex::new();

                        let start = Instant::now();
                        for i in 0..count {
                            let memory = Arc::new(create_random_memory(i));
                            let _ = index.insert_memory(memory);
                        }
                        let elapsed = start.elapsed();
                        total_duration += elapsed / count;
                    }

                    total_duration
                });
            },
        );

        // Batch per-item
        group.bench_with_input(
            BenchmarkId::new("batch_per_item", count),
            &count,
            |b, &count| {
                b.iter_custom(|iters| {
                    let mut total_duration = std::time::Duration::ZERO;

                    for _ in 0..iters {
                        let index = CognitiveHnswIndex::new();

                        let memories: Vec<Arc<Memory>> = (0..count)
                            .map(|i| Arc::new(create_random_memory(i)))
                            .collect();

                        let start = Instant::now();
                        let _ = index.insert_batch(&memories);
                        let elapsed = start.elapsed();
                        total_duration += elapsed / count;
                    }

                    total_duration
                });
            },
        );
    }

    group.finish();
}

/// CRITICAL BENCHMARK: Concurrent insertion with 8 threads
///
/// This benchmark validates the lock-free performance assumptions for Task 003.
/// Target: 80K ops/sec (10K per thread)
/// Minimum acceptable: 60K ops/sec
/// Fallback trigger: < 60K ops/sec indicates high lock contention
fn bench_concurrent_hnsw_insert(c: &mut Criterion) {
    let mut group = c.benchmark_group("concurrent_inserts");
    group.sample_size(10); // Fewer samples due to high cost

    for thread_count in [1, 2, 4, 8] {
        group.bench_with_input(
            BenchmarkId::new("concurrent", thread_count),
            &thread_count,
            |b, &thread_count| {
                b.iter_custom(|iters| {
                    let mut total_duration = std::time::Duration::ZERO;

                    for _ in 0..iters {
                        let index = Arc::new(CognitiveHnswIndex::new());
                        let inserts_per_thread = 1000;

                        let start = Instant::now();

                        let handles: Vec<_> = (0..thread_count)
                            .map(|thread_id| {
                                let idx = Arc::clone(&index);
                                std::thread::spawn(move || {
                                    for i in 0..inserts_per_thread {
                                        let memory_id = thread_id * inserts_per_thread + i;
                                        let memory = Arc::new(create_random_memory(memory_id));
                                        let _ = idx.insert_memory(memory);
                                    }
                                })
                            })
                            .collect();

                        for handle in handles {
                            let _ = handle.join();
                        }

                        total_duration += start.elapsed();
                    }

                    total_duration
                });
            },
        );
    }

    group.finish();
}

/// Benchmark: Measure lock contention under concurrent batch operations
fn bench_concurrent_batch_insert(c: &mut Criterion) {
    let mut group = c.benchmark_group("concurrent_batch");
    group.sample_size(10);

    for (thread_count, batch_size) in [(2, 100), (4, 100), (8, 100), (8, 500)] {
        group.bench_with_input(
            BenchmarkId::new(
                format!("threads_{thread_count}_batch_{batch_size}"),
                format!("{thread_count}x{batch_size}"),
            ),
            &(thread_count, batch_size),
            |b, &(thread_count, batch_size)| {
                b.iter_custom(|iters| {
                    let mut total_duration = std::time::Duration::ZERO;

                    for _ in 0..iters {
                        let index = Arc::new(CognitiveHnswIndex::new());

                        let start = Instant::now();

                        let handles: Vec<_> = (0..thread_count)
                            .map(|thread_id| {
                                let idx = Arc::clone(&index);
                                std::thread::spawn(move || {
                                    let base_id = thread_id * batch_size;
                                    let memories: Vec<Arc<Memory>> = (0..batch_size)
                                        .map(|i| {
                                            let id = base_id + i;
                                            Arc::new(create_random_memory(
                                                u32::try_from(id).unwrap_or(0),
                                            ))
                                        })
                                        .collect();

                                    let _ = idx.insert_batch(&memories);
                                })
                            })
                            .collect();

                        for handle in handles {
                            let _ = handle.join();
                        }

                        total_duration += start.elapsed();
                    }

                    total_duration
                });
            },
        );
    }

    group.finish();
}

/// Benchmark: Speedup ratio - batch vs sequential
fn bench_speedup_ratio(c: &mut Criterion) {
    c.bench_function("speedup_ratio_100", |b| {
        b.iter_custom(|iters| {
            let mut sequential_total = std::time::Duration::ZERO;
            let mut batch_total = std::time::Duration::ZERO;

            for _ in 0..iters {
                // Sequential
                {
                    let index = CognitiveHnswIndex::new();
                    let start = Instant::now();
                    for i in 0..100 {
                        let memory = Arc::new(create_random_memory(i));
                        let _ = index.insert_memory(memory);
                    }
                    sequential_total += start.elapsed();
                }

                // Batch
                {
                    let index = CognitiveHnswIndex::new();
                    let memories: Vec<Arc<Memory>> = (0..100)
                        .map(|i| Arc::new(create_random_memory(i)))
                        .collect();

                    let start = Instant::now();
                    let _ = index.insert_batch(&memories);
                    batch_total += start.elapsed();
                }
            }

            // Return batch time (we'll compare manually in results)
            black_box(batch_total);
            batch_total
        });
    });
}

criterion_group!(
    benches,
    bench_sequential_inserts,
    bench_batch_inserts,
    bench_per_item_latency,
    bench_concurrent_hnsw_insert,
    bench_concurrent_batch_insert,
    bench_speedup_ratio,
);
criterion_main!(benches);
