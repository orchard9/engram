#![allow(missing_docs)]
//! Baseline performance benchmarks for critical hot paths.
//!
//! These micro-benchmarks establish performance baselines for:
//! - Vector similarity comparisons
//! - Activation spreading iterations
//! - Memory decay calculations
//!
//! Baselines are used for regression detection and tracking optimization progress.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use engram_core::activation::{
    ActivationGraphExt, DecayFunction, EdgeType, MemoryGraph, ParallelSpreadingConfig,
    create_activation_graph,
};
use engram_core::activation::test_support::run_spreading;
use engram_core::{Confidence, Memory};
use rand::{Rng, SeedableRng, rngs::StdRng};
use std::sync::Arc;
use std::time::Duration;

/// Generate a random normalized 768-dimensional embedding
fn generate_embedding(rng: &mut StdRng) -> [f32; 768] {
    let mut embedding = [0.0f32; 768];
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

/// Compute cosine similarity between two embeddings
#[inline]
fn cosine_similarity(a: &[f32; 768], b: &[f32; 768]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Baseline: Vector similarity against 1000 candidates
fn vector_similarity_baseline(c: &mut Criterion) {
    let mut group = c.benchmark_group("vector_similarity_baseline");
    group.sample_size(100);
    group.warm_up_time(Duration::from_secs(3));

    let mut rng = StdRng::seed_from_u64(0x0000_5EED);
    let query = generate_embedding(&mut rng);

    // Test against different candidate set sizes
    for candidate_count in [100, 500, 1000, 5000] {
        let candidates: Vec<[f32; 768]> = (0..candidate_count)
            .map(|_| generate_embedding(&mut rng))
            .collect();

        group.throughput(Throughput::Elements(candidate_count as u64));
        group.bench_with_input(
            BenchmarkId::new("cosine_similarity", candidate_count),
            &candidates,
            |b, candidates| {
                b.iter(|| {
                    let mut best_score = -1.0f32;
                    for candidate in candidates {
                        let score = cosine_similarity(black_box(&query), black_box(candidate));
                        best_score = best_score.max(score);
                    }
                    black_box(best_score)
                });
            },
        );
    }

    group.finish();
}

/// Baseline: Single vector similarity computation
fn single_vector_similarity_baseline(c: &mut Criterion) {
    let mut group = c.benchmark_group("single_vector_similarity");
    group.sample_size(1000);

    let mut rng = StdRng::seed_from_u64(0x0000_5EED);
    let query = generate_embedding(&mut rng);
    let candidate = generate_embedding(&mut rng);

    group.bench_function("cosine_similarity_768d", |b| {
        b.iter(|| {
            let score = cosine_similarity(black_box(&query), black_box(&candidate));
            black_box(score)
        });
    });

    group.finish();
}

/// Create a test graph with specified size
fn create_test_graph(node_count: usize, edge_count: usize, seed: u64) -> Arc<MemoryGraph> {
    let graph = Arc::new(create_activation_graph());
    let mut rng = StdRng::seed_from_u64(seed);

    let nodes: Vec<String> = (0..node_count)
        .map(|i| format!("node_{i:04}"))
        .collect();

    // Add edges randomly
    for _ in 0..edge_count {
        let source_idx = rng.gen_range(0..node_count);
        let mut target_idx = rng.gen_range(0..node_count);
        while target_idx == source_idx {
            target_idx = rng.gen_range(0..node_count);
        }

        let weight = rng.gen_range(0.3..1.0);
        ActivationGraphExt::add_edge(
            &*graph,
            nodes[source_idx].clone(),
            nodes[target_idx].clone(),
            weight,
            EdgeType::Excitatory,
        );
    }

    graph
}

/// Baseline: Spreading activation on different graph sizes
fn spreading_activation_baseline(c: &mut Criterion) {
    let mut group = c.benchmark_group("spreading_activation_baseline");
    group.sample_size(50);
    group.warm_up_time(Duration::from_secs(2));
    group.measurement_time(Duration::from_secs(10));

    // Test on different graph sizes
    let test_cases = [
        (100, 500, "small_100n_500e"),
        (500, 2500, "medium_500n_2500e"),
        (1000, 5000, "large_1000n_5000e"),
    ];

    for (node_count, edge_count, name) in test_cases {
        let graph = create_test_graph(node_count, edge_count, 0x0000_BA5E);
        let all_nodes = graph.get_all_nodes();

        let config = ParallelSpreadingConfig {
            max_depth: 5,
            decay_function: DecayFunction::Exponential { rate: 0.3 },
            num_threads: 4,
            cycle_detection: true,
            ..Default::default()
        };

        group.throughput(Throughput::Elements(node_count as u64));
        group.bench_with_input(BenchmarkId::new("spread", name), &graph, |b, graph| {
            b.iter(|| {
                let seed_node = &all_nodes[0];
                let seeds = vec![(seed_node.clone(), 1.0)];
                let result = run_spreading(graph, &seeds, config.clone());
                black_box(result)
            });
        });
    }

    group.finish();
}

/// Baseline: Spreading activation with different iteration counts
fn spreading_iteration_baseline(c: &mut Criterion) {
    let mut group = c.benchmark_group("spreading_iterations");
    group.sample_size(100);

    let graph = create_test_graph(1000, 5000, 0x0000_BA5E);
    let all_nodes = graph.get_all_nodes();
    let seed_node = &all_nodes[0];
    let seeds = vec![(seed_node.clone(), 1.0)];

    // Test with different max_depth values (controls iteration count)
    for max_depth in [3, 5, 7, 10] {
        let config = ParallelSpreadingConfig {
            max_depth,
            decay_function: DecayFunction::Exponential { rate: 0.3 },
            num_threads: 4,
            cycle_detection: true,
            ..Default::default()
        };

        group.bench_with_input(
            BenchmarkId::new("depth", max_depth),
            &config,
            |b, config| {
                b.iter(|| {
                    let result = run_spreading(&graph, black_box(&seeds), config.clone());
                    black_box(result)
                });
            },
        );
    }

    group.finish();
}

/// Baseline: Memory decay calculations
fn decay_calculation_baseline(c: &mut Criterion) {
    let mut group = c.benchmark_group("decay_calculations");
    group.sample_size(100);

    // Test different decay functions
    let decay_functions = [
        (DecayFunction::Exponential { rate: 0.3 }, "exponential"),
        (DecayFunction::PowerLaw { exponent: 0.5 }, "power_law"),
        (DecayFunction::Linear { slope: 0.1 }, "linear"),
    ];

    for (decay_fn, name) in decay_functions {
        group.bench_with_input(
            BenchmarkId::new("decay_type", name),
            &decay_fn,
            |b, decay_fn| {
                b.iter(|| {
                    let initial_activation = 1.0f32;
                    let mut activation = initial_activation;

                    // Simulate decay over 100 time steps
                    for depth in 0..100 {
                        activation = match decay_fn {
                            DecayFunction::Exponential { rate } => {
                                activation * (-rate * depth as f32).exp()
                            }
                            DecayFunction::PowerLaw { exponent } => {
                                activation / (1.0 + depth as f32).powf(*exponent)
                            }
                            DecayFunction::Linear { slope } => {
                                (activation - slope * depth as f32).max(0.0)
                            }
                            _ => activation * 0.95,
                        };
                    }
                    black_box(activation)
                });
            },
        );
    }

    group.finish();
}

/// Baseline: Batch decay on multiple memories
fn batch_decay_baseline(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_decay");
    group.sample_size(50);

    let graph = create_test_graph(10_000, 0, 0x0000_DECA);
    let all_nodes = graph.get_all_nodes();

    group.throughput(Throughput::Elements(10_000));
    group.bench_function("decay_10k_memories", |b| {
        b.iter(|| {
            // Simulate exponential decay calculation for all nodes
            let decay_rate = 0.05f32;
            for _ in &all_nodes {
                let activation = 1.0f32;
                let decayed = activation * (-decay_rate).exp();
                black_box(decayed);
            }
        });
    });

    group.finish();
}

/// Baseline: Graph traversal operations
fn graph_traversal_baseline(c: &mut Criterion) {
    let mut group = c.benchmark_group("graph_traversal");
    group.sample_size(100);

    let graph = create_test_graph(1000, 5000, 0x0000_07EA);
    let all_nodes = graph.get_all_nodes();

    group.bench_function("get_neighbors_1000_nodes", |b| {
        b.iter(|| {
            for node_id in &all_nodes {
                let neighbors = graph.get_neighbors(node_id);
                black_box(neighbors);
            }
        });
    });

    group.bench_function("get_all_nodes", |b| {
        b.iter(|| {
            let nodes = graph.get_all_nodes();
            black_box(nodes)
        });
    });

    group.finish();
}

/// Baseline: Memory allocation and deallocation patterns
fn memory_allocation_baseline(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_allocation");
    group.sample_size(100);

    let mut rng = StdRng::seed_from_u64(0x0000_A110C);

    group.bench_function("allocate_embedding", |b| {
        b.iter(|| {
            let embedding = generate_embedding(&mut rng);
            black_box(embedding)
        });
    });

    group.bench_function("allocate_memory_struct", |b| {
        b.iter(|| {
            let embedding = generate_embedding(&mut rng);
            let memory = Memory::new(
                "test_memory".to_string(),
                embedding,
                Confidence::HIGH,
            );
            black_box(memory)
        });
    });

    group.finish();
}

fn configure_criterion() -> Criterion {
    Criterion::default()
        .output_directory(std::path::Path::new("tmp/baseline_benchmarks"))
        .confidence_level(0.95)
        .noise_threshold(0.02)
        .significance_level(0.05)
}

criterion_group! {
    name = baseline_group;
    config = configure_criterion();
    targets =
        vector_similarity_baseline,
        single_vector_similarity_baseline,
        spreading_activation_baseline,
        spreading_iteration_baseline,
        decay_calculation_baseline,
        batch_decay_baseline,
        graph_traversal_baseline,
        memory_allocation_baseline
}

criterion_main!(baseline_group);
