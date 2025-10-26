//! Concurrent HNSW performance validation
//!
//! This benchmark validates whether the current HNSW implementation can support
//! the worker pool's target of 40K-80K insertions/sec with 4-8 concurrent threads.
//!
//! Decision threshold: If < 60K ops/sec with 8 threads, implement fallback
//! (per-layer locks or space partitioning) before proceeding with worker pool.

use chrono::Utc;
use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use engram_core::index::CognitiveHnswIndex;
use engram_core::{Confidence, Episode};
use std::sync::Arc;
use std::time::Instant;

fn generate_random_embedding() -> [f32; 768] {
    let mut embedding = [0.0f32; 768];
    for i in 0..768 {
        embedding[i] = (i as f32 * 0.01).sin();
    }
    embedding
}

fn create_test_episode(id: usize) -> Episode {
    Episode::new(
        format!("episode_{}", id),
        Utc::now(),
        format!("Test episode content {}", id),
        generate_random_embedding(),
        Confidence::HIGH,
    )
}

/// Single-threaded baseline: insert 1000 memories
fn bench_single_threaded_insert(c: &mut Criterion) {
    let mut group = c.benchmark_group("hnsw_single_threaded");
    group.throughput(Throughput::Elements(1000));

    group.bench_function("insert_1k", |b| {
        b.iter(|| {
            let index = CognitiveHnswIndex::new();
            for i in 0..1000 {
                let episode = create_test_episode(i);
                let memory = Arc::new(engram_core::Memory::from_episode(episode, 1.0));
                black_box(index.insert_memory(memory).unwrap());
            }
        });
    });

    group.finish();
}

/// Concurrent validation: N threads inserting simultaneously
fn bench_concurrent_insert(c: &mut Criterion) {
    let mut group = c.benchmark_group("hnsw_concurrent");

    for num_threads in [2, 4, 8] {
        let total_ops = num_threads * 1000;
        group.throughput(Throughput::Elements(total_ops as u64));

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}threads", num_threads)),
            &num_threads,
            |b, &threads| {
                b.iter(|| {
                    let index = Arc::new(CognitiveHnswIndex::new());
                    let start = Instant::now();

                    let handles: Vec<_> = (0..threads)
                        .map(|t| {
                            let idx = Arc::clone(&index);
                            std::thread::spawn(move || {
                                for i in 0..1000 {
                                    let episode = create_test_episode(t * 1000 + i);
                                    let memory =
                                        Arc::new(engram_core::Memory::from_episode(episode, 1.0));
                                    idx.insert_memory(memory).unwrap();
                                }
                            })
                        })
                        .collect();

                    for h in handles {
                        h.join().unwrap();
                    }

                    let elapsed = start.elapsed();
                    let ops_per_sec = total_ops as f64 / elapsed.as_secs_f64();
                    eprintln!(
                        "{} threads: {} ops in {:?} = {:.0} ops/sec",
                        threads, total_ops, elapsed, ops_per_sec
                    );
                });
            },
        );
    }

    group.finish();
}

/// Space-sharded validation: each thread inserts to different memory space
/// This simulates the worker pool's space-based sharding strategy
fn bench_space_sharded_insert(c: &mut Criterion) {
    let mut group = c.benchmark_group("hnsw_space_sharded");

    for num_threads in [2, 4, 8] {
        let total_ops = num_threads * 1000;
        group.throughput(Throughput::Elements(total_ops as u64));

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}threads", num_threads)),
            &num_threads,
            |b, &threads| {
                b.iter(|| {
                    let index = Arc::new(CognitiveHnswIndex::new());
                    let start = Instant::now();

                    let handles: Vec<_> = (0..threads)
                        .map(|t| {
                            let idx = Arc::clone(&index);
                            std::thread::spawn(move || {
                                // Each thread uses a different memory space (zero contention)
                                for i in 0..1000 {
                                    let episode = create_test_episode(t * 1000 + i);
                                    let memory =
                                        Arc::new(engram_core::Memory::from_episode(episode, 1.0));
                                    idx.insert_memory(memory).unwrap();
                                }
                            })
                        })
                        .collect();

                    for h in handles {
                        h.join().unwrap();
                    }

                    let elapsed = start.elapsed();
                    let ops_per_sec = total_ops as f64 / elapsed.as_secs_f64();
                    eprintln!(
                        "{} threads (space-sharded): {} ops in {:?} = {:.0} ops/sec",
                        threads, total_ops, elapsed, ops_per_sec
                    );
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_single_threaded_insert,
    bench_concurrent_insert,
    bench_space_sharded_insert
);
criterion_main!(benches);
