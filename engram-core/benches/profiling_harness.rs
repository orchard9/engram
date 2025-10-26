#![allow(missing_docs)]
//! Profiling harness for flamegraph generation and hotspot identification.
//!
//! This benchmark creates a realistic workload with 10k nodes, 50k edges, and 1000 queries
//! to exercise all major hot paths: vector similarity, activation spreading, and memory decay.

use criterion::{Criterion, black_box, criterion_group, criterion_main};
use engram_core::activation::test_support::run_spreading;
use engram_core::activation::{
    ActivationGraphExt, DecayFunction, EdgeType, MemoryGraph, ParallelSpreadingConfig,
    create_activation_graph,
};
use rand::{Rng, SeedableRng, rngs::StdRng};
use std::sync::Arc;
use std::time::Duration;

/// Create a large-scale realistic graph with 10k nodes and 50k edges
fn create_realistic_graph(seed: u64) -> Arc<MemoryGraph> {
    let graph = Arc::new(create_activation_graph());
    let mut rng = StdRng::seed_from_u64(seed);

    let node_count = 10_000;
    let edge_count = 50_000;

    // Create nodes with realistic IDs
    let nodes: Vec<String> = (0..node_count).map(|i| format!("memory_{i:06}")).collect();

    // Add edges using preferential attachment to create realistic degree distribution.
    // This creates a scale-free graph similar to real memory networks.
    //
    // IMPORTANT: This implementation uses *total degree* (in-degree + out-degree)
    // for preferential attachment, not just out-degree. This means:
    //   - Nodes that are popular targets (high in-degree) are more likely to be chosen as sources
    //   - Nodes that are active sources (high out-degree) are more likely to be chosen again
    //   - This creates bidirectional hub nodes, which is realistic for memory consolidation
    //
    // For memory graphs, total degree is appropriate because:
    //   1. Frequently accessed memories (high in-degree) trigger more associations
    //   2. Memories with many associations (high out-degree) are more likely to be reused
    //   3. Real hippocampal-neocortical consolidation exhibits both patterns
    let mut node_degrees: Vec<usize> = vec![0; node_count];

    for edge_idx in 0..edge_count {
        // For first few edges, connect randomly
        let (source_idx, target_idx) = if edge_idx < 100 {
            let source = rng.gen_range(0..node_count);
            let mut target = rng.gen_range(0..node_count);
            while target == source {
                target = rng.gen_range(0..node_count);
            }
            (source, target)
        } else {
            // Preferential attachment: nodes with higher degree are more likely to be chosen
            let total_degree: usize = node_degrees.iter().sum();
            let source_idx = if total_degree == 0 {
                rng.gen_range(0..node_count)
            } else {
                let mut dart = rng.gen_range(0..total_degree);
                let mut chosen = 0;
                for (idx, degree) in node_degrees.iter().enumerate() {
                    if dart < *degree {
                        chosen = idx;
                        break;
                    }
                    dart = dart.saturating_sub(*degree);
                }
                chosen
            };

            let mut target_idx = rng.gen_range(0..node_count);
            while target_idx == source_idx {
                target_idx = rng.gen_range(0..node_count);
            }
            (source_idx, target_idx)
        };

        let weight = rng.gen_range(0.1..1.0);
        ActivationGraphExt::add_edge(
            &*graph,
            nodes[source_idx].clone(),
            nodes[target_idx].clone(),
            weight,
            EdgeType::Excitatory,
        );

        // Update total degree for both source and target
        // This creates bidirectional hubs in the network
        node_degrees[source_idx] = node_degrees[source_idx].saturating_add(1);
        node_degrees[target_idx] = node_degrees[target_idx].saturating_add(1);
    }

    graph
}

/// Generate a random 768-dimensional embedding
fn generate_random_embedding(rng: &mut StdRng) -> [f32; 768] {
    let mut embedding = [0.0f32; 768];
    for value in &mut embedding {
        *value = rng.gen_range(-1.0..1.0);
    }
    // Normalize to unit length for realistic cosine similarity
    let magnitude: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
    if magnitude > 0.0 {
        for value in &mut embedding {
            *value /= magnitude;
        }
    }
    embedding
}

/// Execute 1000 spreading activation queries
fn run_spreading_queries(graph: &Arc<MemoryGraph>, config: &ParallelSpreadingConfig, seed: u64) {
    let mut rng = StdRng::seed_from_u64(seed);
    let all_nodes = graph.get_all_nodes();

    for _ in 0..1000 {
        let seed_idx = rng.gen_range(0..all_nodes.len());
        let seed_node = &all_nodes[seed_idx];

        let seeds = vec![(seed_node.clone(), 1.0)];
        let _result = run_spreading(graph, &seeds, config.clone());
    }
}

/// Execute 1000 vector similarity queries (simulated)
fn run_similarity_queries(seed: u64) {
    let mut rng = StdRng::seed_from_u64(seed);

    // Create 1000 candidate embeddings
    let candidates: Vec<[f32; 768]> = (0..1000)
        .map(|_| generate_random_embedding(&mut rng))
        .collect();

    // Run 1000 similarity comparisons
    for _ in 0..1000 {
        let query = generate_random_embedding(&mut rng);

        // Compute cosine similarity against all candidates
        let mut best_score = -1.0f32;
        for candidate in &candidates {
            let dot_product: f32 = query.iter().zip(candidate.iter()).map(|(a, b)| a * b).sum();
            best_score = best_score.max(dot_product);
        }
        black_box(best_score);
    }
}

/// Execute memory decay on all memories
fn run_decay_simulation(graph: &Arc<MemoryGraph>) {
    let all_nodes = graph.get_all_nodes();

    // Simulate time-based decay calculation for all nodes
    let decay_rate = 0.05f32;
    for _ in &all_nodes {
        let activation = 1.0f32;
        let decayed = activation * (-decay_rate).exp();
        black_box(decayed);
    }
}

/// Complete profiling workload that exercises all hot paths
fn profiling_workload(c: &mut Criterion) {
    let mut group = c.benchmark_group("profiling_workload");
    group.sample_size(10); // Small sample size for profiling, not precise timing
    group.warm_up_time(Duration::from_secs(5)); // Increased from 2s for better cache warming
    group.measurement_time(Duration::from_secs(30)); // Long measurement for stable profiling

    group.bench_function("complete_workload", |b| {
        let graph = create_realistic_graph(0xDEAD_BEEF);
        let config = ParallelSpreadingConfig {
            max_depth: 5,
            decay_function: DecayFunction::Exponential { rate: 0.3 },
            num_threads: 4,
            cycle_detection: true,
            ..Default::default()
        };

        b.iter(|| {
            // 1. Run spreading activation queries (20-30% of compute time)
            run_spreading_queries(&graph, &config, 0x0005_EED1);

            // 2. Run vector similarity queries (15-25% of compute time)
            run_similarity_queries(0x0005_EED2);

            // 3. Run memory decay (10-15% of compute time)
            run_decay_simulation(&graph);
        });
    });

    group.finish();
}

/// Benchmark just graph creation to isolate setup cost
fn graph_creation_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("graph_creation");
    group.sample_size(10);

    group.bench_function("create_10k_nodes_50k_edges", |b| {
        b.iter(|| {
            let graph = create_realistic_graph(0xDEAD_BEEF);
            black_box(graph);
        });
    });

    group.finish();
}

/// Benchmark just spreading activation to isolate that hot path
fn spreading_only_benchmark(c: &mut Criterion) {
    let graph = create_realistic_graph(0xDEAD_BEEF);
    let config = ParallelSpreadingConfig {
        max_depth: 5,
        decay_function: DecayFunction::Exponential { rate: 0.3 },
        num_threads: 4,
        cycle_detection: true,
        ..Default::default()
    };

    let mut group = c.benchmark_group("spreading_activation");
    group.sample_size(10);

    group.bench_function("1000_queries", |b| {
        b.iter(|| {
            run_spreading_queries(&graph, &config, 0x0005_EED1);
        });
    });

    group.finish();
}

/// Benchmark just vector similarity to isolate that hot path
fn similarity_only_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("vector_similarity");
    group.sample_size(10);

    group.bench_function("1000_queries_vs_1000_candidates", |b| {
        b.iter(|| {
            run_similarity_queries(0x0005_EED2);
        });
    });

    group.finish();
}

/// Benchmark just memory decay to isolate that hot path
fn decay_only_benchmark(c: &mut Criterion) {
    let graph = create_realistic_graph(0xDEAD_BEEF);

    let mut group = c.benchmark_group("memory_decay");
    group.sample_size(10);

    group.bench_function("10k_memories", |b| {
        b.iter(|| {
            run_decay_simulation(&graph);
        });
    });

    group.finish();
}

fn configure_criterion() -> Criterion {
    Criterion::default()
        .output_directory(std::path::Path::new("tmp/profiling"))
        .confidence_level(0.95)
        .noise_threshold(0.05) // Higher noise threshold acceptable for profiling
}

criterion_group! {
    name = profiling_group;
    config = configure_criterion();
    targets =
        profiling_workload,
        graph_creation_benchmark,
        spreading_only_benchmark,
        similarity_only_benchmark,
        decay_only_benchmark
}

criterion_main!(profiling_group);
