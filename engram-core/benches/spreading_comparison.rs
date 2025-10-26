#![allow(missing_docs)]
//! Performance comparison: Zig spreading activation kernel vs. Rust baseline
//!
//! Benchmarks activation spreading across various graph topologies to measure
//! the performance improvement from the cache-optimized Zig implementation.
//!
//! Target: 20-35% faster than Rust baseline

use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
#[cfg(feature = "zig-kernels")]
use engram_core::zig_kernels;
use rand::{Rng, SeedableRng, rngs::StdRng};
use std::time::Duration;

/// Generate random graph in edge-list format
fn generate_random_graph(
    num_nodes: usize,
    edge_probability: f64,
    seed: u64,
) -> (Vec<u32>, Vec<f32>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut adjacency = Vec::new();
    let mut weights = Vec::new();

    // Generate edges with given probability
    for source in 0..num_nodes {
        for target in 0..num_nodes {
            if source != target && rng.gen_bool(edge_probability) {
                adjacency.push(target as u32);
                weights.push(rng.gen_range(0.1..1.0));
            }
        }
    }

    (adjacency, weights)
}

/// Rust baseline implementation of spreading activation
fn rust_spread_activation(
    adjacency: &[u32],
    weights: &[f32],
    activations: &mut [f32],
    num_nodes: usize,
    iterations: u32,
) {
    assert_eq!(adjacency.len(), weights.len());

    let num_edges = adjacency.len();
    if num_edges == 0 {
        return;
    }

    // Use a heuristic to infer source nodes from edge structure
    let edges_per_node = num_edges.div_ceil(num_nodes);

    // Main spreading loop
    for _ in 0..iterations {
        let current_activations = activations.to_vec();

        // Accumulate activations along edges
        for (edge_idx, (&target, &weight)) in adjacency.iter().zip(weights.iter()).enumerate() {
            let source = edge_idx / edges_per_node;
            if source >= num_nodes || target >= (num_nodes as u32) {
                continue;
            }

            activations[target as usize] += current_activations[source] * weight;
        }

        // Normalize to prevent explosion
        let max_activation = activations.iter().copied().fold(0.0_f32, f32::max);
        if max_activation > 1.0 {
            for act in activations.iter_mut() {
                *act /= max_activation;
            }
        }
    }
}

/// Benchmark spreading activation on different graph sizes
fn spreading_activation_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("spreading_activation_comparison");
    group.sample_size(50);
    group.warm_up_time(Duration::from_secs(2));
    group.measurement_time(Duration::from_secs(10));

    // Test different graph sizes
    let test_cases = [
        (100, 0.1, "sparse_100n"),
        (500, 0.05, "sparse_500n"),
        (1000, 0.03, "sparse_1000n"),
        (100, 0.3, "dense_100n"),
        (500, 0.2, "dense_500n"),
    ];

    for (num_nodes, edge_prob, name) in test_cases {
        let (adjacency, weights) = generate_random_graph(num_nodes, edge_prob, 0xBEEF);

        // Initialize activations
        let initial_activations = {
            let mut acts = vec![0.0f32; num_nodes];
            acts[0] = 1.0; // Activate first node
            acts
        };

        group.throughput(Throughput::Elements(num_nodes as u64));

        // Rust baseline
        group.bench_with_input(
            BenchmarkId::new("rust_baseline", name),
            &(&adjacency, &weights, &initial_activations),
            |b, (adj, w, init_act)| {
                b.iter(|| {
                    let mut activations = (*init_act).clone();
                    rust_spread_activation(
                        black_box(adj),
                        black_box(w),
                        black_box(&mut activations),
                        num_nodes,
                        10, // 10 iterations
                    );
                    black_box(activations)
                });
            },
        );

        // Zig kernel (if feature enabled)
        #[cfg(feature = "zig-kernels")]
        group.bench_with_input(
            BenchmarkId::new("zig_kernel", name),
            &(&adjacency, &weights, &initial_activations),
            |b, (adj, w, init_act)| {
                b.iter(|| {
                    let mut activations = (*init_act).clone();
                    zig_kernels::spread_activation(
                        black_box(adj),
                        black_box(w),
                        black_box(&mut activations),
                        num_nodes,
                        10, // 10 iterations
                    );
                    black_box(activations)
                });
            },
        );
    }

    group.finish();
}

/// Benchmark spreading with varying iteration counts
fn spreading_iteration_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("spreading_iteration_scaling");
    group.sample_size(50);

    let num_nodes = 1000;
    let (adjacency, weights) = generate_random_graph(num_nodes, 0.03, 0xCAFE);
    let initial_activations = {
        let mut acts = vec![0.0f32; num_nodes];
        acts[0] = 1.0;
        acts
    };

    for iterations in [5, 10, 20, 50] {
        // Rust baseline
        group.bench_with_input(
            BenchmarkId::new("rust_baseline", iterations),
            &(&adjacency, &weights, &initial_activations, iterations),
            |b, (adj, w, init_act, iters)| {
                b.iter(|| {
                    let mut activations = (*init_act).clone();
                    rust_spread_activation(
                        black_box(adj),
                        black_box(w),
                        black_box(&mut activations),
                        num_nodes,
                        *iters,
                    );
                    black_box(activations)
                });
            },
        );

        // Zig kernel
        #[cfg(feature = "zig-kernels")]
        group.bench_with_input(
            BenchmarkId::new("zig_kernel", iterations),
            &(&adjacency, &weights, &initial_activations, iterations),
            |b, (adj, w, init_act, iters)| {
                b.iter(|| {
                    let mut activations = (*init_act).clone();
                    zig_kernels::spread_activation(
                        black_box(adj),
                        black_box(w),
                        black_box(&mut activations),
                        num_nodes,
                        *iters,
                    );
                    black_box(activations)
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    spreading_activation_comparison,
    spreading_iteration_scaling
);
criterion_main!(benches);
