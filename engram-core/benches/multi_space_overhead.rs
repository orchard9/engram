//! Multi-space performance overhead benchmarks for Milestone 7 Task 007
//!
//! Measures the performance impact of multi-space architecture:
//! - Recall latency: 1 space vs 5 spaces vs 10 spaces
//! - Registry lookup overhead
//! - Concurrent space creation
//! - Single-space vs multi-space throughput
//! - Space routing overhead

#![allow(missing_docs)]

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use engram_core::{MemorySpaceId, MemorySpaceRegistry, MemoryStore};
use std::sync::Arc;
use tempfile::tempdir;

/// Create a test registry with N spaces pre-populated
async fn create_populated_registry(
    num_spaces: usize,
    memories_per_space: usize,
) -> (Arc<MemorySpaceRegistry>, Vec<MemorySpaceId>) {
    let temp_dir = tempdir().expect("temp dir");

    let registry = Arc::new(
        MemorySpaceRegistry::new(temp_dir.path(), |space_id, _| {
            Ok(Arc::new(MemoryStore::for_space(space_id.clone(), 1000)))
        })
        .expect("registry"),
    );

    let mut space_ids = Vec::new();

    // Create and populate spaces
    for i in 0..num_spaces {
        let space_id = MemorySpaceId::new(format!("bench_space_{i}")).expect("valid space id");
        let handle = registry
            .create_or_get(&space_id)
            .await
            .expect("space handle");
        let store = handle.store();

        // Populate with memories
        for j in 0..memories_per_space {
            let mut emb = [0.0f32; 768];
            emb.fill(i as f32 * 0.1 + j as f32 * 0.01);
            let episode = engram_core::Episode::new(
                format!("space_{i}_mem_{j}"),
                chrono::Utc::now(),
                format!("Benchmark content {j}"),
                emb,
                engram_core::Confidence::exact(0.9),
            );
            let _ = store.store(episode);
        }

        space_ids.push(space_id);
    }

    (registry, space_ids)
}

/// Benchmark recall latency across different numbers of spaces
fn bench_recall_latency(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();

    let mut group = c.benchmark_group("recall_latency");

    for num_spaces in [1, 5, 10] {
        let (registry, space_ids) = rt.block_on(create_populated_registry(num_spaces, 100));

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{num_spaces}_spaces")),
            &num_spaces,
            |b, _| {
                b.iter(|| {
                    rt.block_on(async {
                        // Pick a random space
                        let target_idx = black_box(num_spaces / 2);
                        let space_id = &space_ids[target_idx];

                        // Get the store
                        let handle = registry.get(space_id).expect("space handle");
                        let store = handle.store();

                        // Perform recall operation
                        let mem_id = format!("space_{target_idx}_mem_50");
                        let result = store.get(&mem_id);
                        black_box(result)
                    })
                });
            },
        );
    }

    group.finish();
}

/// Benchmark registry lookup overhead
fn bench_registry_lookup(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();

    let mut group = c.benchmark_group("registry_lookup");

    for num_spaces in [1, 5, 10, 20] {
        let (registry, space_ids) = rt.block_on(create_populated_registry(num_spaces, 10));

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{num_spaces}_spaces")),
            &num_spaces,
            |b, _| {
                b.iter(|| {
                    // Lookup random space
                    let target_idx = black_box(num_spaces / 2);
                    let space_id = &space_ids[target_idx];
                    let result = registry.get(space_id);
                    black_box(result)
                });
            },
        );
    }

    group.finish();
}

/// Benchmark concurrent space creation
fn bench_concurrent_space_creation(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();

    let mut group = c.benchmark_group("concurrent_space_creation");

    for num_concurrent in [1, 5, 10, 20] {
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{num_concurrent}_concurrent")),
            &num_concurrent,
            |b, &num| {
                b.iter(|| {
                    rt.block_on(async {
                        let temp_dir = tempdir().expect("temp dir");
                        let registry = Arc::new(
                            MemorySpaceRegistry::new(temp_dir.path(), |space_id, _| {
                                Ok(Arc::new(MemoryStore::for_space(space_id.clone(), 100)))
                            })
                            .expect("registry"),
                        );

                        // Spawn concurrent creation tasks
                        let mut handles = vec![];
                        for i in 0..num {
                            let registry = Arc::clone(&registry);
                            let handle = tokio::spawn(async move {
                                let space_id = MemorySpaceId::new(format!("concurrent_{i}"))
                                    .expect("valid space id");
                                registry.create_or_get(&space_id).await.expect("create")
                            });
                            handles.push(handle);
                        }

                        // Wait for all to complete
                        let results = futures::future::join_all(handles).await;
                        black_box(results)
                    });
                });
            },
        );
    }

    group.finish();
}

/// Benchmark single-space vs multi-space throughput
fn bench_throughput_comparison(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();

    let mut group = c.benchmark_group("throughput_comparison");

    // Single space baseline
    let temp_dir = tempdir().expect("temp dir");
    let single_store = Arc::new(MemoryStore::new(1000));
    let single_registry = Arc::new(
        MemorySpaceRegistry::new(temp_dir.path(), {
            let store = Arc::clone(&single_store);
            move |_, _| Ok(Arc::clone(&store))
        })
        .expect("registry"),
    );
    let single_space = MemorySpaceId::default();
    rt.block_on(single_registry.create_or_get(&single_space))
        .expect("default space");

    group.bench_function("single_space_baseline", |b| {
        b.iter(|| {
            let handle = single_registry.get(&single_space).expect("space");
            let store = handle.store();

            for i in 0..100 {
                let mut emb = [0.0f32; 768];
                emb.fill(i as f32 * 0.01);
                let episode = engram_core::Episode::new(
                    format!("single_{i}"),
                    chrono::Utc::now(),
                    format!("Content {i}"),
                    emb,
                    engram_core::Confidence::exact(0.9),
                );
                let _ = store.store(episode);
            }

            black_box(store.count())
        });
    });

    // Multi-space comparison
    for num_spaces in [5, 10] {
        let (registry, space_ids) = rt.block_on(create_populated_registry(num_spaces, 0));

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{num_spaces}_spaces")),
            &num_spaces,
            |b, _| {
                b.iter(|| {
                    rt.block_on(async {
                        // Write 100 memories distributed across spaces
                        for i in 0..100 {
                            let target_idx = i % num_spaces;
                            let space_id = &space_ids[target_idx];
                            let handle = registry.get(space_id).expect("space handle");
                            let store = handle.store();

                            let mut emb = [0.0f32; 768];
                            emb.fill(i as f32 * 0.01);
                            let episode = engram_core::Episode::new(
                                format!("multi_{i}"),
                                chrono::Utc::now(),
                                format!("Content {i}"),
                                emb,
                                engram_core::Confidence::exact(0.9),
                            );
                            let _ = store.store(episode);
                        }

                        // Aggregate counts
                        let mut total = 0;
                        for space_id in &space_ids {
                            let handle = registry.get(space_id).expect("space handle");
                            total += handle.store().count();
                        }
                        black_box(total)
                    })
                });
            },
        );
    }

    group.finish();
}

/// Benchmark space routing overhead (registry → store)
fn bench_space_routing(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();

    let mut group = c.benchmark_group("space_routing");

    for num_spaces in [1, 5, 10, 20, 50] {
        let (registry, space_ids) = rt.block_on(create_populated_registry(num_spaces, 10));

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{num_spaces}_spaces")),
            &num_spaces,
            |b, _| {
                b.iter(|| {
                    rt.block_on(async {
                        // Route to middle space
                        let target_idx = black_box(num_spaces / 2);
                        let space_id = &space_ids[target_idx];

                        // Full routing path: registry → handle → store
                        let handle = registry.get(space_id).expect("space handle");
                        let store = handle.store();
                        black_box(store.count())
                    })
                });
            },
        );
    }

    group.finish();
}

/// Benchmark concurrent reads from different spaces
fn bench_concurrent_reads(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();

    let mut group = c.benchmark_group("concurrent_reads");

    for num_spaces in [1, 5, 10] {
        let (registry, space_ids) = rt.block_on(create_populated_registry(num_spaces, 100));

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{num_spaces}_spaces")),
            &num_spaces,
            |b, _| {
                b.iter(|| {
                    rt.block_on(async {
                        // Spawn concurrent readers
                        let mut handles = vec![];
                        for (idx, space_id) in space_ids.iter().enumerate() {
                            let registry = Arc::clone(&registry);
                            let space_id = space_id.clone();
                            let handle = tokio::spawn(async move {
                                let handle = registry.get(&space_id).expect("space handle");
                                let store = handle.store();

                                // Read 10 memories
                                let mut results = vec![];
                                for i in 0..10 {
                                    let mem_id = format!("space_{idx}_mem_{i}");
                                    if let Some(memory) = store.get(&mem_id) {
                                        results.push(memory);
                                    }
                                }
                                results.len()
                            });
                            handles.push(handle);
                        }

                        // Wait for all reads
                        let results = futures::future::join_all(handles).await;
                        black_box(results)
                    });
                });
            },
        );
    }

    group.finish();
}

/// Benchmark space list operation
fn bench_space_list(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();

    let mut group = c.benchmark_group("space_list");

    for num_spaces in [1, 10, 50, 100] {
        let (registry, _) = rt.block_on(create_populated_registry(num_spaces, 10));

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{num_spaces}_spaces")),
            &num_spaces,
            |b, _| {
                b.iter(|| {
                    let list = registry.list();
                    black_box(list)
                });
            },
        );
    }

    group.finish();
}

/// Benchmark memory isolation overhead (cross-space lookup miss)
fn bench_isolation_overhead(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();

    let mut group = c.benchmark_group("isolation_overhead");

    let (registry, space_ids) = rt.block_on(create_populated_registry(2, 100));

    group.bench_function("cross_space_lookup", |b| {
        b.iter(|| {
            rt.block_on(async {
                // Try to find space_0 memory in space_1 (should fail)
                let handle = registry.get(&space_ids[1]).expect("space handle");
                let store = handle.store();

                let result = store.get("space_0_mem_50");
                black_box(result)
            })
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_recall_latency,
    bench_registry_lookup,
    bench_concurrent_space_creation,
    bench_throughput_comparison,
    bench_space_routing,
    bench_concurrent_reads,
    bench_space_list,
    bench_isolation_overhead
);
criterion_main!(benches);
