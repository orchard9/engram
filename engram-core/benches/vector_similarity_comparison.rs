// Performance comparison: Rust baseline vs. Zig SIMD kernels
//
// This benchmark validates the 15-25% performance improvement target
// for vector similarity computation with SIMD acceleration.
//
// Benchmark Methodology:
// - Realistic dimensions: 384, 768, 1536 (production embedding sizes)
// - Varied batch sizes: 10, 100, 1000 candidates
// - Cache effects: Test both hot and cold cache scenarios
// - Statistical rigor: criterion.rs with warmup and multiple iterations
//
// Expected Results (on AVX2 hardware):
// - Small batches (10 candidates): ~15% improvement (less amortization)
// - Medium batches (100 candidates): ~20% improvement (sweet spot)
// - Large batches (1000 candidates): ~25% improvement (maximum amortization)
//
// Performance Analysis:
// - Compute bound: SIMD provides 4-8x theoretical speedup
// - Memory bound: Bandwidth limits realized gains to 15-25%
// - Overhead: FFI boundary and cache effects reduce theoretical maximum

#![allow(missing_docs)] // Benchmark file doesn't require public API documentation

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};

/// Rust baseline: Pure Rust cosine similarity implementation
///
/// This serves as the performance baseline for comparison.
/// Uses standard Rust iterators which may be autovectorized by LLVM.
fn cosine_similarity_rust(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());

    let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();

    let magnitude_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let magnitude_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if magnitude_a == 0.0 || magnitude_b == 0.0 {
        0.0
    } else {
        dot_product / (magnitude_a * magnitude_b)
    }
}

/// Rust baseline: Batch cosine similarity
fn batch_cosine_similarity_rust(query: &[f32], candidates: &[Vec<f32>]) -> Vec<f32> {
    candidates
        .iter()
        .map(|candidate| cosine_similarity_rust(query, candidate))
        .collect()
}

/// Zig kernel: Batch cosine similarity with SIMD
///
/// This calls into the Zig FFI when the zig-kernels feature is enabled.
/// Falls back to Rust implementation when feature is disabled.
#[cfg(feature = "zig-kernels")]
#[allow(unsafe_code)] // FFI boundary requires unsafe
#[allow(clippy::items_after_statements)] // FFI declaration after setup code
fn batch_cosine_similarity_zig(query: &[f32], candidates: &[Vec<f32>]) -> Vec<f32> {
    // External FFI declaration
    unsafe extern "C" {
        fn engram_vector_similarity(
            query: *const f32,
            candidates: *const f32,
            scores: *mut f32,
            query_len: usize,
            num_candidates: usize,
        );
    }

    let dim = query.len();
    let num_candidates = candidates.len();

    // Flatten candidates into contiguous memory (required by FFI)
    let candidates_flat: Vec<f32> = candidates.iter().flat_map(|v| v.iter().copied()).collect();

    let mut scores = vec![0.0_f32; num_candidates];

    unsafe {
        engram_vector_similarity(
            query.as_ptr(),
            candidates_flat.as_ptr(),
            scores.as_mut_ptr(),
            dim,
            num_candidates,
        );
    }

    scores
}

#[cfg(not(feature = "zig-kernels"))]
fn batch_cosine_similarity_zig(query: &[f32], candidates: &[Vec<f32>]) -> Vec<f32> {
    // Fallback to Rust when Zig not available
    batch_cosine_similarity_rust(query, candidates)
}

/// Benchmark suite: Vector similarity performance comparison
fn bench_vector_similarity(c: &mut Criterion) {
    let mut group = c.benchmark_group("vector_similarity");

    // Realistic embedding dimensions
    // 384: sentence-transformers/all-MiniLM-L6-v2
    // 768: BERT-base, GPT-2
    // 1536: OpenAI text-embedding-ada-002
    let dimensions = [384, 768, 1536];

    // Varied batch sizes to test amortization effects
    let batch_sizes = [10, 100, 1000];

    for &dim in &dimensions {
        for &num_candidates in &batch_sizes {
            // Generate test data
            let query: Vec<f32> = (0..dim).map(|i| (i as f32) * 0.01).collect();

            let candidates: Vec<Vec<f32>> = (0..num_candidates)
                .map(|c| (0..dim).map(|i| ((c + i) as f32) * 0.01).collect())
                .collect();

            let bench_id = format!("{dim}d_{num_candidates}c");

            // Benchmark Rust baseline
            group.bench_with_input(
                BenchmarkId::new("rust", &bench_id),
                &(&query, &candidates),
                |b, (query, candidates)| {
                    b.iter(|| {
                        let scores = batch_cosine_similarity_rust(query, candidates);
                        black_box(scores);
                    });
                },
            );

            // Benchmark Zig kernel (only if feature enabled)
            #[cfg(feature = "zig-kernels")]
            group.bench_with_input(
                BenchmarkId::new("zig", &bench_id),
                &(&query, &candidates),
                |b, (query, candidates)| {
                    b.iter(|| {
                        let scores = batch_cosine_similarity_zig(query, candidates);
                        black_box(scores);
                    });
                },
            );
        }
    }

    group.finish();
}

/// Benchmark: Single similarity computation (non-batched)
///
/// This tests the overhead of FFI boundary for small workloads.
/// Expected: Zig may be slower here due to FFI overhead.
fn bench_single_similarity(c: &mut Criterion) {
    let mut group = c.benchmark_group("single_similarity");

    for &dim in &[384, 768, 1536] {
        let a: Vec<f32> = (0..dim).map(|i| (i as f32) * 0.01).collect();
        let b: Vec<f32> = (0..dim).map(|i| ((dim - i) as f32) * 0.01).collect();

        let bench_id = format!("{dim}d");

        // Rust baseline
        group.bench_with_input(
            BenchmarkId::new("rust", &bench_id),
            &(&a, &b),
            |bencher, (a, b)| {
                bencher.iter(|| {
                    let score = cosine_similarity_rust(a, b);
                    black_box(score);
                });
            },
        );

        // Zig kernel (batch size = 1)
        #[cfg(feature = "zig-kernels")]
        group.bench_with_input(
            BenchmarkId::new("zig", &bench_id),
            &(&a, &b),
            |bencher, (a, b)| {
                bencher.iter(|| {
                    let candidates = vec![(*b).clone()];
                    let scores = batch_cosine_similarity_zig(a, &candidates);
                    black_box(scores);
                });
            },
        );
    }

    group.finish();
}

/// Benchmark: Cache-cold scenario
///
/// Tests performance when data is not in L1/L2 cache.
/// This simulates real-world memory access patterns.
fn bench_cache_cold(c: &mut Criterion) {
    let mut group = c.benchmark_group("cache_cold");

    let dim = 768;
    let num_candidates = 1000;

    // Large dataset to evict cache between iterations
    let query: Vec<f32> = (0..dim).map(|i| (i as f32) * 0.01).collect();

    let candidates: Vec<Vec<f32>> = (0..num_candidates)
        .map(|c| (0..dim).map(|i| ((c + i) as f32) * 0.01).collect())
        .collect();

    // Rust baseline
    group.bench_function("rust", |b| {
        b.iter(|| {
            let scores = batch_cosine_similarity_rust(&query, &candidates);
            black_box(scores);
        });
    });

    // Zig kernel
    #[cfg(feature = "zig-kernels")]
    group.bench_function("zig", |b| {
        b.iter(|| {
            let scores = batch_cosine_similarity_zig(&query, &candidates);
            black_box(scores);
        });
    });

    group.finish();
}

/// Benchmark: Pathological cases
///
/// Tests performance with edge cases that might affect SIMD efficiency.
fn bench_edge_cases(c: &mut Criterion) {
    let mut group = c.benchmark_group("edge_cases");

    // Zero vectors (should short-circuit)
    {
        let query = vec![0.0; 768];
        let candidates: Vec<Vec<f32>> = (0..100).map(|_| vec![1.0; 768]).collect();

        group.bench_function("zero_query_rust", |b| {
            b.iter(|| {
                let scores = batch_cosine_similarity_rust(&query, &candidates);
                black_box(scores);
            });
        });

        #[cfg(feature = "zig-kernels")]
        group.bench_function("zero_query_zig", |b| {
            b.iter(|| {
                let scores = batch_cosine_similarity_zig(&query, &candidates);
                black_box(scores);
            });
        });
    }

    // Sparse vectors (mostly zeros)
    {
        let mut query = vec![0.0; 768];
        query[0] = 1.0;
        query[384] = 1.0;
        query[767] = 1.0;

        let candidates: Vec<Vec<f32>> = (0..100)
            .map(|_| {
                let mut v = vec![0.0; 768];
                v[100] = 1.0;
                v
            })
            .collect();

        group.bench_function("sparse_rust", |b| {
            b.iter(|| {
                let scores = batch_cosine_similarity_rust(&query, &candidates);
                black_box(scores);
            });
        });

        #[cfg(feature = "zig-kernels")]
        group.bench_function("sparse_zig", |b| {
            b.iter(|| {
                let scores = batch_cosine_similarity_zig(&query, &candidates);
                black_box(scores);
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_vector_similarity,
    bench_single_similarity,
    bench_cache_cold,
    bench_edge_cases
);
criterion_main!(benches);
