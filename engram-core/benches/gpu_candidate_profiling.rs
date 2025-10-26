#![allow(missing_docs)]
//! GPU Candidate Profiling Benchmark
//!
//! This benchmark profiles CPU SIMD operations to identify GPU acceleration candidates.
//! It measures:
//! - Batch cosine similarity across various batch sizes
//! - Activation spreading operations at different scales
//! - HNSW search operations
//! - Batch recall operations
//!
//! The profiling data is used to calculate theoretical GPU speedups and prioritize
//! which operations should be GPU-accelerated first.

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use engram_core::activation::{
    ActivationGraphExt, EdgeType, MemoryGraph, ParallelSpreadingConfig, create_activation_graph,
};
use engram_core::compute;
use pprof::criterion::{Output, PProfProfiler};
use rand::{Rng, SeedableRng, rngs::StdRng};
use std::sync::Arc;
use std::time::{Duration, Instant};

const SEED: u64 = 42;
const EMBEDDING_DIM: usize = 768;

fn generate_random_embedding(rng: &mut StdRng) -> [f32; EMBEDDING_DIM] {
    let mut embedding = [0.0f32; EMBEDDING_DIM];
    for value in &mut embedding {
        *value = rng.gen_range(-1.0..1.0);
    }

    let norm: f32 = embedding.iter().map(|&x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for value in &mut embedding {
            *value /= norm;
        }
    }

    embedding
}

fn profile_batch_cosine_similarity(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_cosine_similarity");

    group.sample_size(100);
    group.measurement_time(Duration::from_secs(10));

    let mut rng = StdRng::seed_from_u64(SEED);
    let query = generate_random_embedding(&mut rng);

    for batch_size in [16, 64, 256, 1024, 4096, 16384] {
        let targets: Vec<[f32; EMBEDDING_DIM]> = (0..batch_size)
            .map(|_| generate_random_embedding(&mut rng))
            .collect();

        group.bench_with_input(
            BenchmarkId::from_parameter(batch_size),
            &batch_size,
            |b, _| {
                let ops = compute::get_vector_ops();
                b.iter(|| {
                    black_box(
                        ops.cosine_similarity_batch_768(black_box(&query), black_box(&targets)),
                    )
                });
            },
        );
    }

    group.finish();
}

fn profile_single_cosine_similarity(c: &mut Criterion) {
    let mut group = c.benchmark_group("single_cosine_similarity");

    group.sample_size(100);

    let mut rng = StdRng::seed_from_u64(SEED);
    let a = generate_random_embedding(&mut rng);
    let b = generate_random_embedding(&mut rng);

    group.bench_function("cosine_similarity_768", |bencher| {
        let ops = compute::get_vector_ops();
        bencher.iter(|| black_box(ops.cosine_similarity_768(black_box(&a), black_box(&b))));
    });

    group.finish();
}

fn create_test_graph(node_count: usize, seed: u64) -> (Arc<MemoryGraph>, Vec<String>) {
    let graph = Arc::new(create_activation_graph());
    let mut rng = StdRng::seed_from_u64(seed);

    let nodes: Vec<String> = (0..node_count).map(|i| format!("node_{i:06}")).collect();

    let edges_per_node = 5;
    for i in 0..node_count {
        for _ in 0..edges_per_node {
            let target_idx = rng.gen_range(0..node_count);
            if target_idx != i {
                let weight = rng.gen_range(0.1..1.0);
                ActivationGraphExt::add_edge(
                    &*graph,
                    nodes[i].clone(),
                    nodes[target_idx].clone(),
                    weight,
                    EdgeType::Excitatory,
                );
            }
        }

        let embedding = generate_random_embedding(&mut rng);
        ActivationGraphExt::set_embedding(&*graph, &nodes[i], &embedding);
    }

    (graph, nodes)
}

fn profile_activation_spreading(c: &mut Criterion) {
    let mut group = c.benchmark_group("activation_spreading");

    group.sample_size(20);
    group.measurement_time(Duration::from_secs(10));

    for node_count in [100, 500, 1000, 5000, 10000] {
        let (graph, nodes) = create_test_graph(node_count, SEED);

        let config = ParallelSpreadingConfig {
            num_threads: 4,
            max_depth: 3,
            batch_size: 64,
            simd_batch_size: 32,
            deterministic: false,
            enable_metrics: false,
            completion_timeout: Some(Duration::from_secs(30)),
            ..Default::default()
        };

        group.bench_with_input(
            BenchmarkId::from_parameter(node_count),
            &node_count,
            |b, _| {
                b.iter_custom(|iters| {
                    let mut total_time = Duration::ZERO;

                    for _ in 0..iters {
                        let engine = engram_core::activation::ParallelSpreadingEngine::new(
                            config.clone(),
                            graph.clone(),
                        )
                        .expect("engine creation");

                        let seed_idx = node_count / 2;
                        let seed_activations = vec![(nodes[seed_idx].clone(), 1.0)];

                        let start = Instant::now();
                        let result = engine.spread_activation(&seed_activations);
                        let elapsed = start.elapsed();

                        black_box(result.expect("spreading"));
                        total_time += elapsed;

                        engine.shutdown().expect("shutdown");
                    }

                    total_time
                });
            },
        );
    }

    group.finish();
}

fn profile_dot_product(c: &mut Criterion) {
    let mut group = c.benchmark_group("dot_product");

    group.sample_size(100);

    let mut rng = StdRng::seed_from_u64(SEED);
    let a = generate_random_embedding(&mut rng);
    let b = generate_random_embedding(&mut rng);

    group.bench_function("dot_product_768", |bencher| {
        let ops = compute::get_vector_ops();
        bencher.iter(|| black_box(ops.dot_product_768(black_box(&a), black_box(&b))));
    });

    group.finish();
}

fn profile_weighted_average(c: &mut Criterion) {
    let mut group = c.benchmark_group("weighted_average");

    group.sample_size(100);

    let mut rng = StdRng::seed_from_u64(SEED);

    for count in [4, 8, 16, 32] {
        let vectors: Vec<[f32; EMBEDDING_DIM]> = (0..count)
            .map(|_| generate_random_embedding(&mut rng))
            .collect();

        let vector_refs: Vec<&[f32; EMBEDDING_DIM]> = vectors.iter().collect();

        let weights: Vec<f32> = (0..count)
            .map(|i| (i as f32 + 1.0) / count as f32)
            .collect();

        group.bench_with_input(BenchmarkId::from_parameter(count), &count, |b, _| {
            let ops = compute::get_vector_ops();
            b.iter(|| {
                black_box(ops.weighted_average_768(black_box(&vector_refs), black_box(&weights)))
            });
        });
    }

    group.finish();
}

fn profile_vector_add_scale(c: &mut Criterion) {
    let mut group = c.benchmark_group("vector_operations");

    group.sample_size(100);

    let mut rng = StdRng::seed_from_u64(SEED);
    let a = generate_random_embedding(&mut rng);
    let b = generate_random_embedding(&mut rng);
    let scale = 0.5f32;

    group.bench_function("vector_add_768", |bencher| {
        let ops = compute::get_vector_ops();
        bencher.iter(|| black_box(ops.vector_add_768(black_box(&a), black_box(&b))));
    });

    group.bench_function("vector_scale_768", |bencher| {
        let ops = compute::get_vector_ops();
        bencher.iter(|| black_box(ops.vector_scale_768(black_box(&a), black_box(scale))));
    });

    group.bench_function("l2_norm_768", |bencher| {
        let ops = compute::get_vector_ops();
        bencher.iter(|| black_box(ops.l2_norm_768(black_box(&a))));
    });

    group.finish();
}

fn profile_batch_vector_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_vector_operations");

    group.sample_size(50);
    group.measurement_time(Duration::from_secs(10));

    let mut rng = StdRng::seed_from_u64(SEED);

    for batch_size in [64, 256, 1024, 4096] {
        let vectors_a: Vec<[f32; EMBEDDING_DIM]> = (0..batch_size)
            .map(|_| generate_random_embedding(&mut rng))
            .collect();

        let vectors_b: Vec<[f32; EMBEDDING_DIM]> = (0..batch_size)
            .map(|_| generate_random_embedding(&mut rng))
            .collect();

        group.bench_with_input(BenchmarkId::new("add", batch_size), &batch_size, |b, _| {
            let ops = compute::get_vector_ops();
            b.iter(|| {
                let results: Vec<[f32; EMBEDDING_DIM]> = vectors_a
                    .iter()
                    .zip(vectors_b.iter())
                    .map(|(a, b)| ops.vector_add_768(a, b))
                    .collect();
                black_box(results)
            });
        });

        group.bench_with_input(
            BenchmarkId::new("scale", batch_size),
            &batch_size,
            |b, _| {
                let ops = compute::get_vector_ops();
                b.iter(|| {
                    let results: Vec<[f32; EMBEDDING_DIM]> = vectors_a
                        .iter()
                        .map(|a| ops.vector_scale_768(a, 0.5))
                        .collect();
                    black_box(results)
                });
            },
        );

        group.bench_with_input(BenchmarkId::new("norm", batch_size), &batch_size, |b, _| {
            let ops = compute::get_vector_ops();
            b.iter(|| {
                let results: Vec<f32> = vectors_a.iter().map(|a| ops.l2_norm_768(a)).collect();
                black_box(results)
            });
        });
    }

    group.finish();
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .with_profiler(PProfProfiler::new(100, Output::Flamegraph(None)))
        .sample_size(100)
        .measurement_time(Duration::from_secs(10));
    targets =
        profile_single_cosine_similarity,
        profile_batch_cosine_similarity,
        profile_dot_product,
        profile_weighted_average,
        profile_vector_add_scale,
        profile_batch_vector_operations,
        profile_activation_spreading,
}

criterion_main!(benches);
