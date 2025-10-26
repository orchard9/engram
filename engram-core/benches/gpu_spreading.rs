//! GPU spreading performance benchmarks
//!
//! Validates GPU acceleration targets:
//! - >5x speedup for graphs >512 nodes
//! - Measures P50/P90/P99 latencies across node counts
//! - Tests sparse vs dense graph performance

#![allow(missing_docs)]

use criterion::{Criterion, criterion_group, criterion_main};
use engram_core::activation::{ActivationGraphExt, MemoryGraph, create_activation_graph};

#[cfg(cuda_available)]
use criterion::{BenchmarkId, Throughput, black_box};

#[cfg(cuda_available)]
use std::collections::HashMap;

#[cfg(cuda_available)]
use engram_core::compute::cuda::spreading::{CsrGraph, GpuSpreadingEngine, SpreadingConfig};

/// Create a test graph with specified number of nodes and average degree
#[allow(dead_code)]
fn create_test_graph(num_nodes: usize, avg_degree: usize) -> MemoryGraph {
    let graph = create_activation_graph();

    // Create nodes with sequential IDs
    let nodes: Vec<String> = (0..num_nodes).map(|i| format!("node_{i}")).collect();

    // Create edges to achieve target average degree
    let edges_per_node = avg_degree;
    let mut rng_seed = 42u64;

    for (i, source) in nodes.iter().enumerate() {
        for _ in 0..edges_per_node {
            // Simple LCG for deterministic "random" targets
            rng_seed = rng_seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let target_idx = (rng_seed as usize) % num_nodes;

            if target_idx != i {
                // Avoid self-loops
                let target = &nodes[target_idx];
                let weight = 0.5 + 0.5 * ((rng_seed % 100) as f32 / 100.0); // Weights in [0.5, 1.0]

                ActivationGraphExt::add_edge(
                    &graph,
                    source.clone(),
                    target.clone(),
                    weight,
                    engram_core::activation::EdgeType::Excitatory,
                );
            }
        }
    }

    graph
}

/// Create CSR representation from test graph
#[cfg(cuda_available)]
fn create_csr_from_test_graph(num_nodes: usize, avg_degree: usize) -> CsrGraph {
    let nodes: Vec<String> = (0..num_nodes).map(|i| format!("node_{i}")).collect();

    let mut row_ptr = Vec::with_capacity(num_nodes + 1);
    let mut col_idx = Vec::new();
    let mut values = Vec::new();
    let mut node_to_idx = HashMap::with_capacity(num_nodes);
    let mut idx_to_node = Vec::with_capacity(num_nodes);

    for (idx, node) in nodes.iter().enumerate() {
        node_to_idx.insert(node.clone(), idx);
        idx_to_node.push(node.clone());
    }

    row_ptr.push(0);

    let mut rng_seed = 42u64;
    for i in 0..num_nodes {
        for _ in 0..avg_degree {
            rng_seed = rng_seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let target_idx = (rng_seed as usize) % num_nodes;

            if target_idx != i {
                col_idx.push(target_idx as i32);
                let weight = 0.5 + 0.5 * ((rng_seed % 100) as f32 / 100.0);
                values.push(weight);
            }
        }
        row_ptr.push(col_idx.len() as i32);
    }

    let num_edges = col_idx.len();

    CsrGraph {
        row_ptr,
        col_idx,
        values,
        num_nodes,
        num_edges,
        node_to_idx,
        idx_to_node,
    }
}

/// Benchmark GPU vs CPU spreading across different node counts
fn benchmark_spreading_scalability(_c: &Criterion) {
    #[cfg(cuda_available)]
    {
        use engram_core::compute::cuda;

        if !cuda::is_available() {
            eprintln!("GPU not available, skipping GPU spreading benchmarks");
            return;
        }

        let mut group = _c.benchmark_group("gpu_spreading_scalability");

        // Test configurations: (num_nodes, avg_degree)
        let configs = vec![
            (100, 5),    // Small, sparse graph (CPU should be faster)
            (500, 8),    // Break-even point
            (1000, 10),  // GPU target
            (5000, 10),  // Large, sparse
            (10000, 15), // Very large
        ];

        for (num_nodes, avg_degree) in configs {
            group.throughput(Throughput::Elements(num_nodes as u64));

            // Create test data
            let csr_graph = create_csr_from_test_graph(num_nodes, avg_degree);
            let input_activations: Vec<f32> = (0..num_nodes)
                .map(|i| i as f32 / num_nodes as f32)
                .collect();

            // GPU benchmark
            group.bench_with_input(
                BenchmarkId::new("gpu", num_nodes),
                &(&csr_graph, &input_activations),
                |b, (csr, input)| {
                    let config = SpreadingConfig::default();
                    let engine = GpuSpreadingEngine::new(config)
                        .expect("Failed to create GPU spreading engine");

                    b.iter(|| {
                        let result = engine
                            .spread_activation_gpu(black_box(csr), black_box(input))
                            .expect("GPU spreading failed");
                        black_box(result);
                    });
                },
            );

            // CPU baseline benchmark (simplified scalar implementation)
            group.bench_with_input(
                BenchmarkId::new("cpu_baseline", num_nodes),
                &(&csr_graph, &input_activations),
                |b, (csr, input)| {
                    b.iter(|| {
                        let mut output = vec![0.0f32; csr.num_nodes];
                        for node_id in 0..csr.num_nodes {
                            let start = csr.row_ptr[node_id] as usize;
                            let end = csr.row_ptr[node_id + 1] as usize;

                            for edge_idx in start..end {
                                let neighbor = csr.col_idx[edge_idx] as usize;
                                let weight = csr.values[edge_idx];
                                output[node_id] += weight * input[neighbor];
                            }
                        }
                        black_box(output);
                    });
                },
            );
        }

        group.finish();
    }

    #[cfg(not(cuda_available))]
    {
        eprintln!("CUDA not available at compile time, skipping GPU benchmarks");
    }
}

/// Benchmark sparse vs dense graph performance
fn benchmark_graph_density(_c: &Criterion) {
    #[cfg(cuda_available)]
    {
        use engram_core::compute::cuda;

        if !cuda::is_available() {
            eprintln!("GPU not available, skipping density benchmarks");
            return;
        }

        let mut group = _c.benchmark_group("gpu_spreading_density");

        let num_nodes = 1000;
        // Test different densities
        let densities = vec![
            (5, "sparse"),       // Average degree 5
            (10, "medium"),      // Average degree 10
            (50, "dense"),       // Average degree 50
            (100, "very_dense"), // Average degree 100
        ];

        for (avg_degree, label) in densities {
            group.throughput(Throughput::Elements(num_nodes as u64));

            let csr_graph = create_csr_from_test_graph(num_nodes, avg_degree);
            let input_activations: Vec<f32> = (0..num_nodes)
                .map(|i| i as f32 / num_nodes as f32)
                .collect();

            group.bench_with_input(
                BenchmarkId::new("gpu", label),
                &(&csr_graph, &input_activations),
                |b, (csr, input)| {
                    let config = SpreadingConfig::default();
                    let engine = GpuSpreadingEngine::new(config)
                        .expect("Failed to create GPU spreading engine");

                    b.iter(|| {
                        let result = engine
                            .spread_activation_gpu(black_box(csr), black_box(input))
                            .expect("GPU spreading failed");
                        black_box(result);
                    });
                },
            );
        }

        group.finish();
    }

    #[cfg(not(cuda_available))]
    {
        eprintln!("CUDA not available at compile time, skipping density benchmarks");
    }
}

// Benchmark configuration for GPU spreading tests
criterion_group!(
    name = benches;
    config = Criterion::default().sample_size(10);  // Reduced for GPU benchmarks
    targets = benchmark_spreading_scalability, benchmark_graph_density
);
criterion_main!(benches);
