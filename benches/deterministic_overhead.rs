//! Benchmark comparing deterministic vs performance mode spreading
//!
//! Measures the overhead of deterministic execution to ensure it stays below 15%.

#![allow(missing_docs)]

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use engram_core::activation::{
    create_activation_graph, ActivationGraphExt, EdgeType, ParallelSpreadingConfig,
    ParallelSpreadingEngine,
};
use std::sync::Arc;

/// Create a 1000-node graph with realistic connectivity
fn create_large_graph() -> Arc<engram_core::activation::MemoryGraph> {
    let graph = Arc::new(create_activation_graph());

    // Create 1000 nodes in 5 layers (depth 0-4)
    // Layer 0: 50 nodes (seed layer)
    // Layer 1: 200 nodes
    // Layer 2: 300 nodes
    // Layer 3: 300 nodes
    // Layer 4: 150 nodes

    let layer_sizes = [50, 200, 300, 300, 150];
    let mut node_id = 0;

    // Build nodes and edges
    for (layer_idx, &layer_size) in layer_sizes.iter().enumerate() {
        let layer_start = node_id;

        for _ in 0..layer_size {
            let current_node = format!("node_{node_id}");

            // Connect to next layer if not the last layer
            if layer_idx < layer_sizes.len() - 1 {
                let next_layer_start: usize = layer_sizes[..=layer_idx].iter().sum();
                let next_layer_end = next_layer_start + layer_sizes[layer_idx + 1];

                // Connect to 8 random nodes in next layer (average)
                let connections = 8.min(layer_sizes[layer_idx + 1]);
                let step = layer_sizes[layer_idx + 1] / connections.max(1);

                for i in 0..connections {
                    let target_id = next_layer_start + (i * step);
                    if target_id < next_layer_end {
                        let target_node = format!("node_{target_id}");
                        // Edge weight varies between 0.3 and 0.9
                        let weight = 0.3 + (((node_id + target_id) % 60) as f32 / 100.0);
                        graph.add_edge(
                            current_node.clone(),
                            target_node,
                            weight,
                            EdgeType::Excitatory,
                        );
                    }
                }
            }

            node_id += 1;
        }

        // Add some intra-layer connections for realism
        if layer_size > 10 {
            for i in 0..(layer_size / 4) {
                let source_id = layer_start + (i * 4);
                let target_id = layer_start + (i * 4) + 2;
                if source_id < layer_start + layer_size && target_id < layer_start + layer_size {
                    graph.add_edge(
                        format!("node_{source_id}"),
                        format!("node_{target_id}"),
                        0.5,
                        EdgeType::Excitatory,
                    );
                }
            }
        }
    }

    graph
}

fn bench_deterministic_mode(c: &mut Criterion) {
    let graph = create_large_graph();

    // Seed nodes from layer 0
    let seed_nodes: Vec<(String, f32)> = (0..5)
        .map(|i| (format!("node_{i}"), 1.0))
        .collect();

    c.bench_function("spreading_deterministic_1000_nodes", |b| {
        b.iter(|| {
            let config = ParallelSpreadingConfig::deterministic(42);
            let engine = ParallelSpreadingEngine::new(config, graph.clone()).unwrap();
            let results = engine.spread_activation(black_box(&seed_nodes)).unwrap();
            engine.shutdown().unwrap();
            black_box(results)
        });
    });
}

fn bench_performance_mode(c: &mut Criterion) {
    let graph = create_large_graph();

    // Seed nodes from layer 0
    let seed_nodes: Vec<(String, f32)> = (0..5)
        .map(|i| (format!("node_{i}"), 1.0))
        .collect();

    c.bench_function("spreading_performance_1000_nodes", |b| {
        b.iter(|| {
            let config = ParallelSpreadingConfig {
                deterministic: false,
                trace_activation_flow: false,
                ..ParallelSpreadingConfig::default()
            };
            let engine = ParallelSpreadingEngine::new(config, graph.clone()).unwrap();
            let results = engine.spread_activation(black_box(&seed_nodes)).unwrap();
            engine.shutdown().unwrap();
            black_box(results)
        });
    });
}

fn bench_thread_scaling(c: &mut Criterion) {
    let graph = create_large_graph();
    let seed_nodes: Vec<(String, f32)> = (0..5)
        .map(|i| (format!("node_{i}"), 1.0))
        .collect();

    let mut group = c.benchmark_group("deterministic_thread_scaling");

    for threads in [1, 2, 4] {
        group.bench_with_input(
            BenchmarkId::from_parameter(threads),
            &threads,
            |b, &threads| {
                b.iter(|| {
                    let config = ParallelSpreadingConfig {
                        num_threads: threads,
                        ..ParallelSpreadingConfig::deterministic(42)
                    };
                    let engine = ParallelSpreadingEngine::new(config, graph.clone()).unwrap();
                    let results = engine.spread_activation(black_box(&seed_nodes)).unwrap();
                    engine.shutdown().unwrap();
                    black_box(results)
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_deterministic_mode,
    bench_performance_mode,
    bench_thread_scaling
);
criterion_main!(benches);
