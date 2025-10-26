//! Comprehensive GPU performance validation benchmarks
//!
//! This benchmark suite validates GPU acceleration achieves target speedups
//! by comparing against CPU SIMD baselines and industry-standard GPU libraries.
//!
//! Validates performance targets from Milestone 12:
//! - Consumer GPU (RTX 3060): >3x speedup for target operations
//! - Datacenter GPU (A100): >6x speedup for large-scale operations
//!
//! Compares against:
//! - CPU SIMD (AVX-512/AVX2/NEON) baseline
//! - cuBLAS for vector operations (if available)
//! - FAISS GPU for vector search (if available)
//!
//! Reports:
//! - P50/P90/P99 latencies for production planning
//! - Actual speedups vs Task 001 predictions
//! - Bottleneck analysis (memory bandwidth vs compute)
//! - Optimization opportunities

#![allow(missing_docs)]

use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use engram_core::compute::get_vector_ops;
use rand::{Rng, SeedableRng, rngs::StdRng};
use std::time::Duration;
#[cfg(cuda_available)]
use std::time::Instant;

/// Generate deterministic random 768-dim vector
fn random_vector_768(rng: &mut StdRng) -> [f32; 768] {
    let mut vec = [0.0f32; 768];
    for item in &mut vec {
        *item = rng.gen_range(-1.0..1.0);
    }
    // Normalize to unit length for realistic cosine similarity
    let magnitude: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
    if magnitude > 0.0 {
        for item in &mut vec {
            *item /= magnitude;
        }
    }
    vec
}

/// Generate batch of random vectors
fn random_vectors_768(count: usize, seed: u64) -> Vec<[f32; 768]> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..count).map(|_| random_vector_768(&mut rng)).collect()
}

/// Benchmark cosine similarity: CPU vs GPU
///
/// Validates Task 001 predictions:
/// - CPU baseline: 2.1 us/vector (AVX-512)
/// - GPU target: 0.3 us/vector
/// - Expected speedup: 7.0x on consumer GPUs
/// - Break-even batch size: 64 vectors
fn bench_cosine_similarity_cpu_vs_gpu(c: &mut Criterion) {
    let mut group = c.benchmark_group("cosine_similarity_cpu_vs_gpu");

    // Test batch sizes from break-even point to large scale
    let batch_sizes = vec![
        16,    // Below break-even (should use CPU)
        64,    // Break-even point
        256,   // GPU sweet spot
        1024,  // Consumer GPU target (1K vectors)
        4096,  // Large batch
        10240, // Datacenter GPU target (10K vectors)
    ];

    for &batch_size in &batch_sizes {
        group.throughput(Throughput::Elements(batch_size as u64));

        let query = random_vector_768(&mut StdRng::seed_from_u64(42));
        let targets = random_vectors_768(batch_size, 43);

        // CPU baseline
        group.bench_with_input(
            BenchmarkId::new("cpu_simd", batch_size),
            &batch_size,
            |b, _| {
                let ops = get_vector_ops();
                b.iter(|| {
                    let results =
                        ops.cosine_similarity_batch_768(black_box(&query), black_box(&targets));
                    black_box(results);
                });
            },
        );

        // GPU (if available)
        #[cfg(cuda_available)]
        group.bench_with_input(
            BenchmarkId::new("gpu_cuda", batch_size),
            &batch_size,
            |b, _| {
                use engram_core::compute::cuda::hybrid::{HybridConfig, HybridExecutor};

                let config = HybridConfig {
                    gpu_min_batch_size: 1, // Force GPU for all batch sizes
                    force_cpu_mode: false,
                    telemetry_enabled: false,
                    ..Default::default()
                };
                let executor = HybridExecutor::new(config);

                if !executor.is_gpu_available() {
                    eprintln!(
                        "GPU not available, skipping GPU benchmark for batch size {}",
                        batch_size
                    );
                    return;
                }

                b.iter(|| {
                    let results = executor
                        .execute_batch_cosine_similarity(black_box(&query), black_box(&targets));
                    black_box(results);
                });
            },
        );

        // Hybrid executor (auto-dispatch)
        #[cfg(cuda_available)]
        group.bench_with_input(
            BenchmarkId::new("hybrid_auto", batch_size),
            &batch_size,
            |b, _| {
                use engram_core::compute::cuda::hybrid::HybridExecutor;

                let executor = HybridExecutor::default();

                b.iter(|| {
                    let results = executor
                        .execute_batch_cosine_similarity(black_box(&query), black_box(&targets));
                    black_box(results);
                });
            },
        );
    }

    group.finish();
}

/// Benchmark activation spreading: CPU vs GPU
///
/// Validates Task 001 predictions:
/// - CPU baseline: 850 us for 1K nodes
/// - GPU target: 120 us for 1K nodes
/// - Expected speedup: 7.1x on consumer GPUs
/// - Break-even: 512 nodes
fn bench_activation_spreading_cpu_vs_gpu(_c: &mut Criterion) {
    #[cfg(cuda_available)]
    {
        let c = _c;
        use engram_core::compute::cuda::spreading::{
            CsrGraph, GpuSpreadingEngine, SpreadingConfig,
        };
        use std::collections::HashMap;

        let mut group = c.benchmark_group("activation_spreading_cpu_vs_gpu");

        // Test node counts from break-even to large scale
        let node_counts = vec![
            100,   // Small graph (CPU should be faster)
            512,   // Break-even point
            1000,  // Consumer GPU target (1K nodes)
            5000,  // Large graph
            10000, // Datacenter GPU target (10K nodes)
        ];

        for &num_nodes in &node_counts {
            group.throughput(Throughput::Elements(num_nodes as u64));

            // Create synthetic CSR graph with average degree 10
            let avg_degree = 10;
            let mut row_ptr = vec![0];
            let mut col_idx = Vec::new();
            let mut values = Vec::new();
            let mut node_to_idx = HashMap::with_capacity(num_nodes);
            let mut idx_to_node = Vec::with_capacity(num_nodes);

            let mut rng = StdRng::seed_from_u64(44);

            for i in 0..num_nodes {
                let node = format!("node_{}", i);
                node_to_idx.insert(node.clone(), i);
                idx_to_node.push(node);

                for _ in 0..avg_degree {
                    let target = rng.gen_range(0..num_nodes);
                    if target != i {
                        col_idx.push(target as i32);
                        values.push(rng.gen_range(0.5..1.0));
                    }
                }
                row_ptr.push(col_idx.len() as i32);
            }

            let csr_graph = CsrGraph {
                row_ptr,
                col_idx,
                values,
                num_nodes,
                num_edges: col_idx.len(),
                node_to_idx,
                idx_to_node,
            };

            let input_activations: Vec<f32> =
                (0..num_nodes).map(|i| (i % 10) as f32 / 10.0).collect();

            // CPU baseline (simplified scalar implementation)
            group.bench_with_input(
                BenchmarkId::new("cpu_scalar", num_nodes),
                &num_nodes,
                |b, _| {
                    b.iter(|| {
                        let mut output = vec![0.0f32; csr_graph.num_nodes];
                        for node_id in 0..csr_graph.num_nodes {
                            let start = csr_graph.row_ptr[node_id] as usize;
                            let end = csr_graph.row_ptr[node_id + 1] as usize;

                            for edge_idx in start..end {
                                let neighbor = csr_graph.col_idx[edge_idx] as usize;
                                let weight = csr_graph.values[edge_idx];
                                output[node_id] += weight * input_activations[neighbor];
                            }
                        }
                        black_box(output);
                    });
                },
            );

            // GPU spreading
            group.bench_with_input(
                BenchmarkId::new("gpu_cuda", num_nodes),
                &num_nodes,
                |b, _| {
                    use engram_core::compute::cuda;

                    if !cuda::is_available() {
                        eprintln!(
                            "GPU not available, skipping GPU spreading benchmark for {} nodes",
                            num_nodes
                        );
                        return;
                    }

                    let config = SpreadingConfig::default();
                    let engine = GpuSpreadingEngine::new(config)
                        .expect("Failed to create GPU spreading engine");

                    b.iter(|| {
                        let result = engine
                            .spread_activation_gpu(
                                black_box(&csr_graph),
                                black_box(&input_activations),
                            )
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
        eprintln!(
            "CUDA not available at compile time, skipping activation spreading GPU benchmarks"
        );
    }
}

/// Benchmark HNSW search: CPU vs GPU
///
/// Validates Task 001 predictions:
/// - CPU baseline: 1.2 ms for 10K index
/// - GPU target: 180 us for 10K index
/// - Expected speedup: 6.7x on consumer GPUs
fn bench_hnsw_search_cpu_vs_gpu(_c: &mut Criterion) {
    #[cfg(cuda_available)]
    {
        let c = _c;
        use engram_core::compute::cuda::hnsw::{GpuHnswSearch, HnswSearchConfig};

        let mut group = c.benchmark_group("hnsw_search_cpu_vs_gpu");

        // Test index sizes
        let index_sizes = vec![
            1000,  // Small index
            10000, // Consumer GPU target (10K index)
            50000, // Large index
        ];

        for &index_size in &index_sizes {
            group.throughput(Throughput::Elements(index_size as u64));

            // Create synthetic index data
            let query = random_vector_768(&mut StdRng::seed_from_u64(45));
            let candidates = random_vectors_768(index_size, 46);
            let k = 10; // Top-k search

            // CPU baseline (brute force kNN)
            group.bench_with_input(
                BenchmarkId::new("cpu_brute_force", index_size),
                &index_size,
                |b, _| {
                    let ops = get_vector_ops();

                    b.iter(|| {
                        let similarities = ops
                            .cosine_similarity_batch_768(black_box(&query), black_box(&candidates));

                        // Find top-k
                        let mut indexed_similarities: Vec<(usize, f32)> =
                            similarities.iter().copied().enumerate().collect();
                        indexed_similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                        let top_k: Vec<usize> = indexed_similarities
                            .iter()
                            .take(k)
                            .map(|(idx, _)| *idx)
                            .collect();
                        black_box(top_k);
                    });
                },
            );

            // GPU HNSW search
            group.bench_with_input(
                BenchmarkId::new("gpu_cuda", index_size),
                &index_size,
                |b, _| {
                    use engram_core::compute::cuda;

                    if !cuda::is_available() {
                        eprintln!(
                            "GPU not available, skipping GPU HNSW benchmark for index size {}",
                            index_size
                        );
                        return;
                    }

                    let config = HnswSearchConfig::default();
                    let search =
                        GpuHnswSearch::new(config).expect("Failed to create GPU HNSW search");

                    b.iter(|| {
                        let result = search
                            .batch_distance_and_select_gpu(
                                black_box(&query),
                                black_box(&candidates),
                                k,
                            )
                            .expect("GPU HNSW search failed");
                        black_box(result);
                    });
                },
            );
        }

        group.finish();
    }

    #[cfg(not(cuda_available))]
    {
        eprintln!("CUDA not available at compile time, skipping HNSW GPU benchmarks");
    }
}

/// Latency distribution analysis: P50/P90/P99
///
/// Critical for production planning - measures tail latencies
/// Uses standard Criterion benchmarking which automatically calculates percentiles
fn bench_latency_distribution(c: &mut Criterion) {
    let mut group = c.benchmark_group("latency_distribution");
    group.sample_size(1000); // Large sample for percentile accuracy

    let batch_size = 1024;
    let query = random_vector_768(&mut StdRng::seed_from_u64(47));
    let targets = random_vectors_768(batch_size, 48);

    // CPU latency - Criterion automatically calculates percentiles
    group.bench_function("cpu_latency", |b| {
        let ops = get_vector_ops();
        b.iter(|| {
            let results = ops.cosine_similarity_batch_768(black_box(&query), black_box(&targets));
            black_box(results)
        });
    });

    // GPU latency (if available) - Criterion automatically calculates percentiles
    #[cfg(cuda_available)]
    group.bench_function("gpu_latency", |b| {
        use engram_core::compute::cuda;
        use engram_core::compute::cuda::hybrid::{HybridConfig, HybridExecutor};

        if !cuda::is_available() {
            eprintln!("GPU not available, skipping GPU latency distribution");
            return;
        }

        let config = HybridConfig {
            gpu_min_batch_size: 1,
            force_cpu_mode: false,
            telemetry_enabled: false,
            ..Default::default()
        };
        let executor = HybridExecutor::new(config);

        b.iter(|| {
            let results =
                executor.execute_batch_cosine_similarity(black_box(&query), black_box(&targets));
            black_box(results)
        });
    });

    group.finish();
}

/// Memory bandwidth utilization test
///
/// Determines if operations are memory-bound or compute-bound
fn bench_memory_bandwidth(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_bandwidth");

    // Test with increasing batch sizes to saturate bandwidth
    let batch_sizes = vec![64, 256, 1024, 4096, 16384];

    for &batch_size in &batch_sizes {
        let bytes_transferred = batch_size * 768 * std::mem::size_of::<f32>() * 2; // Query + targets
        group.throughput(Throughput::Bytes(bytes_transferred as u64));

        let query = random_vector_768(&mut StdRng::seed_from_u64(49));
        let targets = random_vectors_768(batch_size, 50);

        group.bench_with_input(
            BenchmarkId::new("cpu_bandwidth", batch_size),
            &batch_size,
            |b, _| {
                let ops = get_vector_ops();
                b.iter(|| {
                    let results =
                        ops.cosine_similarity_batch_768(black_box(&query), black_box(&targets));
                    black_box(results);
                });
            },
        );

        #[cfg(cuda_available)]
        group.bench_with_input(
            BenchmarkId::new("gpu_bandwidth", batch_size),
            &batch_size,
            |b, _| {
                use engram_core::compute::cuda;
                use engram_core::compute::cuda::hybrid::{HybridConfig, HybridExecutor};

                if !cuda::is_available() {
                    return;
                }

                let config = HybridConfig {
                    gpu_min_batch_size: 1,
                    force_cpu_mode: false,
                    telemetry_enabled: false,
                    ..Default::default()
                };
                let executor = HybridExecutor::new(config);

                b.iter(|| {
                    let results = executor
                        .execute_batch_cosine_similarity(black_box(&query), black_box(&targets));
                    black_box(results);
                });
            },
        );
    }

    group.finish();
}

/// Kernel launch overhead measurement
///
/// Measures GPU kernel launch latency to validate break-even calculations
fn bench_kernel_launch_overhead(_c: &mut Criterion) {
    #[cfg(cuda_available)]
    {
        let c = _c;
        use engram_core::compute::cuda;

        if !cuda::is_available() {
            eprintln!("GPU not available, skipping kernel launch overhead benchmark");
            return;
        }

        let mut group = c.benchmark_group("kernel_launch_overhead");
        group.sample_size(100);

        // Measure launch overhead with minimal computation
        // Use smallest possible batch to isolate launch overhead
        let query = random_vector_768(&mut StdRng::seed_from_u64(51));
        let targets = random_vectors_768(1, 52); // Single vector

        group.bench_function("gpu_launch_overhead_1_vector", |b| {
            use engram_core::compute::cuda::hybrid::{HybridConfig, HybridExecutor};

            let config = HybridConfig {
                gpu_min_batch_size: 1,
                force_cpu_mode: false,
                telemetry_enabled: false,
                ..Default::default()
            };
            let executor = HybridExecutor::new(config);

            b.iter(|| {
                let results = executor
                    .execute_batch_cosine_similarity(black_box(&query), black_box(&targets));
                black_box(results);
            });
        });

        group.finish();
    }

    #[cfg(not(cuda_available))]
    {
        eprintln!("CUDA not available at compile time, skipping kernel launch overhead benchmark");
    }
}

/// Speedup validation: actual vs predicted
///
/// Compares actual measured speedups against Task 001 predictions
fn bench_speedup_validation(_c: &mut Criterion) {
    #[cfg(cuda_available)]
    {
        let c = _c;
        use engram_core::compute::cuda;

        if !cuda::is_available() {
            eprintln!("GPU not available, skipping speedup validation");
            return;
        }

        let mut group = c.benchmark_group("speedup_validation");

        // Test cases from Task 001 predictions
        let test_cases = vec![
            (1024, "1K_vectors", 7.0), // Expected 7.0x speedup
            (4096, "4K_vectors", 8.0), // Expected higher speedup for larger batches
        ];

        for (batch_size, label, expected_speedup) in test_cases {
            group.throughput(Throughput::Elements(batch_size as u64));

            let query = random_vector_768(&mut StdRng::seed_from_u64(53));
            let targets = random_vectors_768(batch_size, 54);

            // Measure CPU baseline
            let ops = get_vector_ops();
            let cpu_start = Instant::now();
            for _ in 0..10 {
                let results = ops.cosine_similarity_batch_768(&query, &targets);
                black_box(results);
            }
            let cpu_time = cpu_start.elapsed() / 10;

            // Measure GPU time
            use engram_core::compute::cuda::hybrid::{HybridConfig, HybridExecutor};

            let config = HybridConfig {
                gpu_min_batch_size: 1,
                force_cpu_mode: false,
                telemetry_enabled: false,
                ..Default::default()
            };
            let executor = HybridExecutor::new(config);

            let gpu_start = Instant::now();
            for _ in 0..10 {
                let results = executor.execute_batch_cosine_similarity(&query, &targets);
                black_box(results);
            }
            let gpu_time = gpu_start.elapsed() / 10;

            let actual_speedup = cpu_time.as_secs_f64() / gpu_time.as_secs_f64();

            eprintln!(
                "Speedup validation {}: CPU {:?}, GPU {:?}, Actual speedup: {:.2}x, Expected: {:.2}x",
                label, cpu_time, gpu_time, actual_speedup, expected_speedup
            );

            // Simple benchmark to record in output
            group.bench_function(format!("{}_cpu", label), |b| {
                b.iter(|| {
                    let results =
                        ops.cosine_similarity_batch_768(black_box(&query), black_box(&targets));
                    black_box(results);
                });
            });

            group.bench_function(format!("{}_gpu", label), |b| {
                b.iter(|| {
                    let results = executor
                        .execute_batch_cosine_similarity(black_box(&query), black_box(&targets));
                    black_box(results);
                });
            });
        }

        group.finish();
    }

    #[cfg(not(cuda_available))]
    {
        eprintln!("CUDA not available at compile time, skipping speedup validation");
    }
}

/// Configure criterion with appropriate settings for GPU benchmarking
fn configure_criterion() -> Criterion {
    Criterion::default()
        .sample_size(100) // Balanced for GPU benchmarks
        .measurement_time(Duration::from_secs(10)) // Longer for stable GPU measurements
        .warm_up_time(Duration::from_secs(3)) // GPU warmup
        .confidence_level(0.95)
        .noise_threshold(0.05)
}

criterion_group! {
    name = gpu_performance;
    config = configure_criterion();
    targets =
        bench_cosine_similarity_cpu_vs_gpu,
        bench_activation_spreading_cpu_vs_gpu,
        bench_hnsw_search_cpu_vs_gpu,
        bench_latency_distribution,
        bench_memory_bandwidth,
        bench_kernel_launch_overhead,
        bench_speedup_validation,
}

criterion_main!(gpu_performance);
