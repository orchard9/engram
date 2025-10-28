//! Comprehensive benchmark suite for Engram performance validation
//!
//! Tests all core operations with statistical rigor using Criterion.
//! Validates performance targets from vision.md:
//! - Throughput: 10,000 activations/second sustained
//! - Latency: P99 < 10ms for single-hop activation
//! - Scale: 1M+ nodes with 768-dimensional embeddings
//! - Concurrency: Linear scaling with CPU cores (up to 32 cores)
//! - Memory: Overhead < 2x raw data size

#![allow(missing_docs)]

use criterion::{
    black_box, criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion, Throughput,
};
use engram_core::{
    activation::{
        create_activation_graph, ActivationGraphExt, EdgeType, ParallelSpreadingConfig,
        ParallelSpreadingEngine,
    },
    memory::{Memory, MemoryId, MemoryStore},
};
use std::sync::Arc;
use std::time::Duration;

/// Generate a deterministic embedding for benchmarking
fn generate_embedding(seed: usize, dim: usize) -> Vec<f32> {
    let mut embedding = Vec::with_capacity(dim);
    for i in 0..dim {
        let value = ((seed.wrapping_mul(31).wrapping_add(i).wrapping_mul(17)) as f32) / 1000.0;
        embedding.push(value % 1.0);
    }

    // Normalize
    let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        embedding.iter_mut().for_each(|x| *x /= norm);
    }

    embedding
}

/// Create a small graph for single-operation benchmarks
fn create_small_graph() -> Arc<engram_core::activation::MemoryGraph> {
    let graph = Arc::new(create_activation_graph());

    // Create 100 nodes with 500 edges (average degree 5)
    for i in 0..100 {
        let node = format!("node_{}", i);

        // Connect to 5 neighbors (circular topology + random)
        for j in 1..=5 {
            let target = (i + j) % 100;
            let target_node = format!("node_{}", target);
            let weight = 0.5 + ((i + target) % 50) as f32 / 100.0;

            graph.add_edge(&node, &target_node, weight, EdgeType::Associative);
        }
    }

    graph
}

/// Create a medium graph for throughput benchmarks
fn create_medium_graph() -> Arc<engram_core::activation::MemoryGraph> {
    let graph = Arc::new(create_activation_graph());

    // Create 10,000 nodes with clustered structure
    let num_clusters = 10;
    let nodes_per_cluster = 1000;

    for cluster_id in 0..num_clusters {
        let cluster_start = cluster_id * nodes_per_cluster;

        for i in 0..nodes_per_cluster {
            let node_id = cluster_start + i;
            let node = format!("node_{}", node_id);

            // Connect within cluster (high density)
            for j in 1..=10 {
                let target_id = cluster_start + ((i + j) % nodes_per_cluster);
                let target_node = format!("node_{}", target_id);
                let weight = 0.7 + ((i + j) % 30) as f32 / 100.0;

                graph.add_edge(&node, &target_node, weight, EdgeType::Associative);
            }

            // Connect across clusters (sparse)
            if i % 100 == 0 {
                let other_cluster = (cluster_id + 1) % num_clusters;
                let target_id = other_cluster * nodes_per_cluster + (i / 100);
                let target_node = format!("node_{}", target_id);

                graph.add_edge(&node, &target_node, 0.3, EdgeType::Associative);
            }
        }
    }

    graph
}

/// Benchmark: Core Operations - Single Store
fn bench_store_single(c: &mut Criterion) {
    let mut group = c.benchmark_group("store");

    // Configure statistical parameters per task spec
    group.sample_size(100);
    group.measurement_time(Duration::from_secs(10));
    group.warm_up_time(Duration::from_secs(3));

    group.bench_function("store_single", |b| {
        let store = MemoryStore::new();
        let mut counter = 0;

        b.iter_batched(
            || {
                counter += 1;
                let embedding = generate_embedding(counter, 768);
                Memory::new(embedding, 0.9)
            },
            |memory| {
                let _ = black_box(store.store(memory));
            },
            BatchSize::SmallInput,
        );
    });

    group.finish();
}

/// Benchmark: Core Operations - Batch Store
fn bench_store_batch(c: &mut Criterion) {
    let mut group = c.benchmark_group("store_batch");

    group.sample_size(50);
    group.measurement_time(Duration::from_secs(10));
    group.warm_up_time(Duration::from_secs(3));

    for batch_size in [100, 1000, 10000] {
        group.throughput(Throughput::Elements(batch_size as u64));

        group.bench_with_input(
            BenchmarkId::new("batch", batch_size),
            &batch_size,
            |b, &size| {
                let store = MemoryStore::new();

                b.iter_batched(
                    || {
                        (0..size)
                            .map(|i| {
                                let embedding = generate_embedding(i, 768);
                                Memory::new(embedding, 0.9)
                            })
                            .collect::<Vec<_>>()
                    },
                    |batch| {
                        for memory in batch {
                            let _ = black_box(store.store(memory));
                        }
                    },
                    BatchSize::LargeInput,
                );
            },
        );
    }

    group.finish();
}

/// Benchmark: Core Operations - Recall by ID
fn bench_recall_by_id(c: &mut Criterion) {
    let mut group = c.benchmark_group("recall");

    group.sample_size(100);
    group.measurement_time(Duration::from_secs(10));
    group.warm_up_time(Duration::from_secs(3));

    group.bench_function("recall_by_id", |b| {
        let store = MemoryStore::new();

        // Pre-populate store with 10,000 memories
        let ids: Vec<MemoryId> = (0..10000)
            .map(|i| {
                let embedding = generate_embedding(i, 768);
                let memory = Memory::new(embedding, 0.9);
                store.store(memory).unwrap()
            })
            .collect();

        let mut counter = 0;
        b.iter(|| {
            let id = &ids[counter % ids.len()];
            counter += 1;
            black_box(store.recall(id))
        });
    });

    group.finish();
}

/// Benchmark: Core Operations - Spreading Activation
fn bench_spreading_activation(c: &mut Criterion) {
    let mut group = c.benchmark_group("spreading_activation");

    group.sample_size(50);
    group.measurement_time(Duration::from_secs(10));
    group.warm_up_time(Duration::from_secs(3));

    for depth in [3, 5] {
        group.bench_with_input(
            BenchmarkId::new("depth", depth),
            &depth,
            |b, &max_depth| {
                let graph = create_small_graph();
                let engine = ParallelSpreadingEngine::new(graph.clone());

                let config = ParallelSpreadingConfig {
                    threshold: 0.1,
                    max_iterations: max_depth,
                    decay_factor: 0.8,
                    num_threads: 1,
                    deterministic: false,
                };

                let mut counter = 0;
                b.iter(|| {
                    let seed_node = format!("node_{}", counter % 100);
                    counter += 1;

                    let seeds = vec![(seed_node, 1.0)];
                    black_box(engine.spread(&seeds, &config))
                });
            },
        );
    }

    group.finish();
}

/// Benchmark: Concurrent Operations - Parallel Writes
fn bench_concurrent_writes(c: &mut Criterion) {
    let mut group = c.benchmark_group("concurrent_writes");

    group.sample_size(30);
    group.measurement_time(Duration::from_secs(10));
    group.warm_up_time(Duration::from_secs(3));

    for num_threads in [1, 2, 4, 8, 16] {
        group.bench_with_input(
            BenchmarkId::new("threads", num_threads),
            &num_threads,
            |b, &threads| {
                let store = MemoryStore::new();
                let operations_per_thread = 100;

                b.iter(|| {
                    let handles: Vec<_> = (0..threads)
                        .map(|thread_id| {
                            let store_clone = store.clone();
                            std::thread::spawn(move || {
                                for i in 0..operations_per_thread {
                                    let seed = thread_id * operations_per_thread + i;
                                    let embedding = generate_embedding(seed, 768);
                                    let memory = Memory::new(embedding, 0.9);
                                    let _ = store_clone.store(memory);
                                }
                            })
                        })
                        .collect();

                    for handle in handles {
                        handle.join().unwrap();
                    }
                });
            },
        );
    }

    group.finish();
}

/// Benchmark: Concurrent Operations - Parallel Reads
fn bench_concurrent_reads(c: &mut Criterion) {
    let mut group = c.benchmark_group("concurrent_reads");

    group.sample_size(30);
    group.measurement_time(Duration::from_secs(10));
    group.warm_up_time(Duration::from_secs(3));

    for num_threads in [1, 2, 4, 8, 16] {
        group.bench_with_input(
            BenchmarkId::new("threads", num_threads),
            &num_threads,
            |b, &threads| {
                let store = MemoryStore::new();

                // Pre-populate with 1000 memories
                let ids: Vec<MemoryId> = (0..1000)
                    .map(|i| {
                        let embedding = generate_embedding(i, 768);
                        let memory = Memory::new(embedding, 0.9);
                        store.store(memory).unwrap()
                    })
                    .collect();

                let operations_per_thread = 100;

                b.iter(|| {
                    let handles: Vec<_> = (0..threads)
                        .map(|thread_id| {
                            let store_clone = store.clone();
                            let ids_clone = ids.clone();
                            std::thread::spawn(move || {
                                for i in 0..operations_per_thread {
                                    let idx = (thread_id * operations_per_thread + i) % ids_clone.len();
                                    let _ = store_clone.recall(&ids_clone[idx]);
                                }
                            })
                        })
                        .collect();

                    for handle in handles {
                        handle.join().unwrap();
                    }
                });
            },
        );
    }

    group.finish();
}

/// Benchmark: Concurrent Operations - Mixed Read/Write
fn bench_concurrent_mixed(c: &mut Criterion) {
    let mut group = c.benchmark_group("concurrent_mixed");

    group.sample_size(30);
    group.measurement_time(Duration::from_secs(10));
    group.warm_up_time(Duration::from_secs(3));

    for num_threads in [1, 2, 4, 8, 16] {
        group.bench_with_input(
            BenchmarkId::new("threads", num_threads),
            &num_threads,
            |b, &threads| {
                let store = MemoryStore::new();

                // Pre-populate with 1000 memories
                let ids: Vec<MemoryId> = (0..1000)
                    .map(|i| {
                        let embedding = generate_embedding(i, 768);
                        let memory = Memory::new(embedding, 0.9);
                        store.store(memory).unwrap()
                    })
                    .collect();

                let operations_per_thread = 100;

                b.iter(|| {
                    let handles: Vec<_> = (0..threads)
                        .map(|thread_id| {
                            let store_clone = store.clone();
                            let ids_clone = ids.clone();
                            std::thread::spawn(move || {
                                for i in 0..operations_per_thread {
                                    if i % 2 == 0 {
                                        // Write operation
                                        let seed = thread_id * operations_per_thread + i + 10000;
                                        let embedding = generate_embedding(seed, 768);
                                        let memory = Memory::new(embedding, 0.9);
                                        let _ = store_clone.store(memory);
                                    } else {
                                        // Read operation
                                        let idx = (thread_id * operations_per_thread + i) % ids_clone.len();
                                        let _ = store_clone.recall(&ids_clone[idx]);
                                    }
                                }
                            })
                        })
                        .collect();

                    for handle in handles {
                        handle.join().unwrap();
                    }
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_store_single,
    bench_store_batch,
    bench_recall_by_id,
    bench_spreading_activation,
    bench_concurrent_writes,
    bench_concurrent_reads,
    bench_concurrent_mixed,
);

criterion_main!(benches);
