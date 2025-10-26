//! GPU HNSW Candidate Scoring Benchmarks
//!
//! Compares GPU-accelerated HNSW top-k selection against CPU baseline
//! across various candidate set sizes and k values.
//!
//! Performance targets (from Task 001 profiling):
//! - CPU baseline: ~1.2 ms for 1K candidates
//! - GPU target: ~180 us for 1K candidates (6.7x speedup)
//! - Break-even: 1024 candidates

#![allow(missing_docs)]

#[cfg(cuda_available)]
use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};

#[cfg(not(cuda_available))]
use criterion::{Criterion, criterion_group, criterion_main};

#[cfg(cuda_available)]
use engram_core::compute::cuda::hnsw::{
    DistanceMetric, cpu_hnsw_top_k, gpu_hnsw_top_k, hybrid_hnsw_top_k,
};

#[cfg(cuda_available)]
use engram_core::compute::cuda;

// Generate deterministic pseudo-random vector for benchmarking
#[cfg(cuda_available)]
fn generate_vector(seed: usize) -> [f32; 768] {
    let mut vec = [0.0f32; 768];
    for (i, elem) in vec.iter_mut().enumerate() {
        // Use sine function for deterministic, varying values
        *elem = ((seed * 768 + i) as f32 * 0.001).sin();
    }
    vec
}

// Generate candidate vectors
#[cfg(cuda_available)]
fn generate_candidates(count: usize) -> Vec<[f32; 768]> {
    (0..count).map(generate_vector).collect()
}

#[cfg(cuda_available)]
fn bench_cpu_hnsw_top_k(c: &mut Criterion) {
    let mut group = c.benchmark_group("cpu_hnsw_top_k");

    let query = generate_vector(0);
    let k = 10;

    // Test across different candidate set sizes
    for &num_candidates in &[16, 64, 256, 1024, 4096] {
        let candidates = generate_candidates(num_candidates);

        group.throughput(Throughput::Elements(num_candidates as u64));
        group.bench_with_input(
            BenchmarkId::new("cosine", num_candidates),
            &num_candidates,
            |b, _| {
                b.iter(|| {
                    let results = cpu_hnsw_top_k(
                        black_box(&query),
                        black_box(&candidates),
                        black_box(k),
                        black_box(DistanceMetric::Cosine),
                    );
                    black_box(results);
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("l2", num_candidates),
            &num_candidates,
            |b, _| {
                b.iter(|| {
                    let results = cpu_hnsw_top_k(
                        black_box(&query),
                        black_box(&candidates),
                        black_box(k),
                        black_box(DistanceMetric::L2),
                    );
                    black_box(results);
                });
            },
        );
    }

    group.finish();
}

#[cfg(cuda_available)]
fn bench_gpu_hnsw_top_k(c: &mut Criterion) {
    if !cuda::is_available() {
        println!("GPU not available, skipping GPU benchmarks");
        return;
    }

    let mut group = c.benchmark_group("gpu_hnsw_top_k");

    let query = generate_vector(0);
    let k = 10;

    // Focus on sizes above break-even point
    for &num_candidates in &[1024, 2048, 4096, 8192] {
        let candidates = generate_candidates(num_candidates);

        group.throughput(Throughput::Elements(num_candidates as u64));
        group.bench_with_input(
            BenchmarkId::new("cosine", num_candidates),
            &num_candidates,
            |b, _| {
                b.iter(|| {
                    let results = gpu_hnsw_top_k(
                        black_box(&query),
                        black_box(&candidates),
                        black_box(k),
                        black_box(DistanceMetric::Cosine),
                    );
                    black_box(results);
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("l2", num_candidates),
            &num_candidates,
            |b, _| {
                b.iter(|| {
                    let results = gpu_hnsw_top_k(
                        black_box(&query),
                        black_box(&candidates),
                        black_box(k),
                        black_box(DistanceMetric::L2),
                    );
                    black_box(results);
                });
            },
        );
    }

    group.finish();
}

#[cfg(cuda_available)]
fn bench_hybrid_hnsw_top_k(c: &mut Criterion) {
    let mut group = c.benchmark_group("hybrid_hnsw_top_k");

    let query = generate_vector(0);
    let k = 10;

    // Test across break-even boundary
    for &num_candidates in &[512, 1024, 2048, 4096] {
        let candidates = generate_candidates(num_candidates);

        group.throughput(Throughput::Elements(num_candidates as u64));
        group.bench_with_input(
            BenchmarkId::new("cosine", num_candidates),
            &num_candidates,
            |b, _| {
                b.iter(|| {
                    let results = hybrid_hnsw_top_k(
                        black_box(&query),
                        black_box(&candidates),
                        black_box(k),
                        black_box(DistanceMetric::Cosine),
                    );
                    black_box(results);
                });
            },
        );
    }

    group.finish();
}

#[cfg(cuda_available)]
fn bench_varying_k(c: &mut Criterion) {
    let mut group = c.benchmark_group("hnsw_varying_k");

    let query = generate_vector(0);
    let num_candidates = 4096;
    let candidates = generate_candidates(num_candidates);

    // Test different k values
    for &k in &[1, 5, 10, 50, 100] {
        group.bench_with_input(BenchmarkId::new("cpu", k), &k, |b, &k| {
            b.iter(|| {
                let results = cpu_hnsw_top_k(
                    black_box(&query),
                    black_box(&candidates),
                    black_box(k),
                    black_box(DistanceMetric::Cosine),
                );
                black_box(results);
            });
        });

        if cuda::is_available() {
            group.bench_with_input(BenchmarkId::new("gpu", k), &k, |b, &k| {
                b.iter(|| {
                    let results = gpu_hnsw_top_k(
                        black_box(&query),
                        black_box(&candidates),
                        black_box(k),
                        black_box(DistanceMetric::Cosine),
                    );
                    black_box(results);
                });
            });
        }
    }

    group.finish();
}

#[cfg(cuda_available)]
fn bench_cpu_vs_gpu_speedup(c: &mut Criterion) {
    if !cuda::is_available() {
        println!("GPU not available, skipping speedup benchmark");
        return;
    }

    let mut group = c.benchmark_group("cpu_vs_gpu_speedup");

    let query = generate_vector(0);
    let k = 10;

    // Test at target size (1K candidates)
    let num_candidates = 1024;
    let candidates = generate_candidates(num_candidates);

    group.throughput(Throughput::Elements(num_candidates as u64));

    group.bench_function("cpu_1024_candidates", |b| {
        b.iter(|| {
            let results = cpu_hnsw_top_k(
                black_box(&query),
                black_box(&candidates),
                black_box(k),
                black_box(DistanceMetric::Cosine),
            );
            black_box(results);
        });
    });

    group.bench_function("gpu_1024_candidates", |b| {
        b.iter(|| {
            let results = gpu_hnsw_top_k(
                black_box(&query),
                black_box(&candidates),
                black_box(k),
                black_box(DistanceMetric::Cosine),
            );
            black_box(results);
        });
    });

    group.finish();
}

#[cfg(cuda_available)]
criterion_group!(
    benches,
    bench_cpu_hnsw_top_k,
    bench_gpu_hnsw_top_k,
    bench_hybrid_hnsw_top_k,
    bench_varying_k,
    bench_cpu_vs_gpu_speedup
);

#[cfg(not(cuda_available))]
#[allow(clippy::missing_const_for_fn)]
fn placeholder_bench(_c: &mut Criterion) {
    // Placeholder when CUDA not available
}

// Benchmark configuration for GPU HNSW tests
#[cfg(not(cuda_available))]
criterion_group!(benches, placeholder_bench);

criterion_main!(benches);
