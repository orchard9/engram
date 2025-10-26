//! Performance benchmarks for GPU cosine similarity
//!
//! This benchmark suite measures GPU vs CPU performance across different batch sizes
//! and validates the speedup predictions from Task 001 profiling data.
//!
//! # Baseline Predictions (Task 001)
//!
//! - CPU AVX-512: ~2.1 µs/vector (305 µs for 256 vectors)
//! - GPU target: ~0.3 µs/vector (<60 µs for 256 vectors)
//! - Expected speedup: 5-7x for batches >=64 vectors
//! - Break-even point: 64 vectors
//!
//! # Success Criteria
//!
//! - Achieves >3x speedup for batches >=64 vectors
//! - P50/P90/P99 latencies validate Task 001 predictions (within 30%)
//! - Memory bandwidth utilization >70% of theoretical peak

#![allow(missing_docs)]

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use engram_core::compute::{create_vector_ops, detect_cpu_features};

#[cfg(all(feature = "gpu", cuda_available))]
use engram_core::compute::cuda::cosine_similarity::GpuCosineSimilarity;

#[cfg(all(feature = "gpu", cuda_available))]
use engram_core::compute::cuda;

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

/// Generate random 768-dimensional vector
fn random_vector_768(rng: &mut impl Rng) -> [f32; 768] {
    let mut vec = [0.0f32; 768];
    for val in &mut vec {
        *val = rng.gen_range(-1.0..1.0);
    }
    vec
}

/// Generate batch of random vectors
fn random_batch_768(count: usize, rng: &mut impl Rng) -> Vec<[f32; 768]> {
    (0..count).map(|_| random_vector_768(rng)).collect()
}

/// Benchmark CPU cosine similarity across different batch sizes
fn bench_cpu_cosine_similarity(c: &mut Criterion) {
    let mut group = c.benchmark_group("cpu_cosine_similarity");
    let cpu_ops = create_vector_ops();
    let mut rng = ChaCha8Rng::seed_from_u64(42);

    let cpu_capability = detect_cpu_features();
    println!("CPU capability: {cpu_capability:?}");

    // Benchmark batch sizes from 16 to 4096
    for size in [16, 64, 128, 256, 512, 1024, 2048, 4096] {
        let query = random_vector_768(&mut rng);
        let targets = random_batch_768(size, &mut rng);

        group.throughput(criterion::Throughput::Elements(size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, _| {
            b.iter(|| {
                let results =
                    cpu_ops.cosine_similarity_batch_768(black_box(&query), black_box(&targets));
                black_box(results)
            });
        });
    }

    group.finish();
}

/// Benchmark GPU cosine similarity and compare with CPU
#[cfg(all(feature = "gpu", cuda_available))]
fn bench_gpu_cosine_similarity(c: &mut Criterion) {
    if !cuda::is_available() {
        println!("GPU not available, skipping GPU benchmarks");
        return;
    }

    // Print GPU info
    let devices = cuda::get_device_info();
    for device in &devices {
        println!("GPU: {}", device);
    }

    let mut group = c.benchmark_group("gpu_cosine_similarity");
    let gpu_ops = GpuCosineSimilarity::new();
    let cpu_ops = create_vector_ops();
    let mut rng = ChaCha8Rng::seed_from_u64(42);

    // Benchmark batch sizes from 16 to 4096
    for size in [16, 64, 128, 256, 512, 1024, 2048, 4096] {
        let query = random_vector_768(&mut rng);
        let targets = random_batch_768(size, &mut rng);

        group.throughput(criterion::Throughput::Elements(size as u64));

        // GPU benchmark
        group.bench_with_input(BenchmarkId::new("gpu", size), &size, |b, _| {
            b.iter(|| {
                let results = gpu_ops
                    .batch_cosine_similarity_batch_768(black_box(&query), black_box(&targets));
                black_box(results)
            });
        });

        // CPU benchmark for comparison
        group.bench_with_input(BenchmarkId::new("cpu", size), &size, |b, _| {
            b.iter(|| {
                let results =
                    cpu_ops.cosine_similarity_batch_768(black_box(&query), black_box(&targets));
                black_box(results)
            });
        });
    }

    group.finish();

    // Print performance metrics
    println!("\nGPU Performance Metrics:");
    println!("  GPU calls: {}", gpu_ops.gpu_call_count());
    println!("  CPU fallbacks: {}", gpu_ops.cpu_fallback_count());
}

/// Benchmark GPU vs CPU speedup analysis
#[cfg(all(feature = "gpu", cuda_available))]
fn bench_gpu_vs_cpu_speedup(c: &mut Criterion) {
    if !cuda::is_available() {
        println!("GPU not available, skipping speedup benchmarks");
        return;
    }

    let mut group = c.benchmark_group("gpu_vs_cpu_speedup");
    group.sample_size(50); // More samples for accurate measurement

    let gpu_ops = GpuCosineSimilarity::new();
    let cpu_ops = create_vector_ops();
    let mut rng = ChaCha8Rng::seed_from_u64(42);

    // Focus on the critical batch size range
    for size in [64, 128, 256, 512, 1024] {
        let query = random_vector_768(&mut rng);
        let targets = random_batch_768(size, &mut rng);

        group.throughput(criterion::Throughput::Elements(size as u64));

        group.bench_with_input(BenchmarkId::new("gpu_optimized", size), &size, |b, _| {
            b.iter(|| {
                let results = gpu_ops
                    .batch_cosine_similarity_batch_768(black_box(&query), black_box(&targets));
                black_box(results)
            });
        });
    }

    group.finish();
}

/// Benchmark memory transfer overhead
#[cfg(all(feature = "gpu", cuda_available))]
fn bench_gpu_memory_overhead(c: &mut Criterion) {
    if !cuda::is_available() {
        println!("GPU not available, skipping memory overhead benchmarks");
        return;
    }

    let mut group = c.benchmark_group("gpu_memory_overhead");
    let gpu_ops = GpuCosineSimilarity::new();
    let mut rng = ChaCha8Rng::seed_from_u64(42);

    // Test different batch sizes to measure memory transfer impact
    for size in [64, 256, 1024, 4096] {
        let query = random_vector_768(&mut rng);
        let targets = random_batch_768(size, &mut rng);

        let data_size_mb = (size * 768 * std::mem::size_of::<f32>()) as f64 / (1024.0 * 1024.0);

        group.throughput(criterion::Throughput::Bytes(
            (size * 768 * std::mem::size_of::<f32>()) as u64,
        ));

        group.bench_with_input(BenchmarkId::new("transfer_compute", size), &size, |b, _| {
            b.iter(|| {
                let results = gpu_ops
                    .batch_cosine_similarity_batch_768(black_box(&query), black_box(&targets));
                black_box(results)
            });
        });

        println!("Batch size {}: {:.2} MB transferred", size, data_size_mb);
    }

    group.finish();
}

/// Benchmark Task 001 validation: 256-vector batch
#[cfg(all(feature = "gpu", cuda_available))]
fn bench_task_001_validation(c: &mut Criterion) {
    if !cuda::is_available() {
        println!("GPU not available, skipping Task 001 validation");
        return;
    }

    println!("\n=== Task 001 Validation Benchmark ===");
    println!("Expected CPU: ~305 µs (P50) for 256 vectors");
    println!("Target GPU: <60 µs (5x speedup)");
    println!("Acceptance: >3x speedup");

    let mut group = c.benchmark_group("task_001_validation");
    group.sample_size(100); // High sample count for accurate P50/P90/P99

    let gpu_ops = GpuCosineSimilarity::new();
    let cpu_ops = create_vector_ops();
    let mut rng = ChaCha8Rng::seed_from_u64(42);

    let query = random_vector_768(&mut rng);
    let targets = random_batch_768(256, &mut rng);

    group.throughput(criterion::Throughput::Elements(256));

    group.bench_function("cpu_256_vectors", |b| {
        b.iter(|| {
            let results =
                cpu_ops.cosine_similarity_batch_768(black_box(&query), black_box(&targets));
            black_box(results)
        });
    });

    group.bench_function("gpu_256_vectors", |b| {
        b.iter(|| {
            let results =
                gpu_ops.batch_cosine_similarity_batch_768(black_box(&query), black_box(&targets));
            black_box(results)
        });
    });

    group.finish();
}

// Register benchmarks

// CPU benchmark group for cosine similarity
criterion_group!(cpu_benches, bench_cpu_cosine_similarity);

// GPU benchmark group for cosine similarity
#[cfg(all(feature = "gpu", cuda_available))]
criterion_group!(
    gpu_benches,
    bench_gpu_cosine_similarity,
    bench_gpu_vs_cpu_speedup,
    bench_gpu_memory_overhead,
    bench_task_001_validation
);

#[cfg(all(feature = "gpu", cuda_available))]
criterion_main!(cpu_benches, gpu_benches);

#[cfg(not(all(feature = "gpu", cuda_available)))]
criterion_main!(cpu_benches);
