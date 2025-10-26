// Performance comparison: Rust baseline vs. Zig SIMD kernels for memory decay
//
// This benchmark validates the 20-30% performance improvement target
// for Ebbinghaus decay computation with SIMD acceleration.
//
// Benchmark Methodology:
// - Realistic batch sizes: 100, 1000, 10000 memories
// - Varied age distributions: uniform, linear, exponential, mixed
// - Cache effects: Test both hot and cold cache scenarios
// - Statistical rigor: criterion.rs with warmup and multiple iterations
//
// Expected Results (on AVX2 hardware):
// - Small batches (100 memories): ~20% improvement
// - Medium batches (1000 memories): ~25% improvement (sweet spot)
// - Large batches (10000 memories): ~30% improvement (maximum SIMD benefit)
//
// Performance Analysis:
// - Compute bound: exp() is transcendental function, SIMD helps but not 8x
// - Memory bound: Sequential access pattern maximizes cache efficiency
// - Overhead: FFI boundary minimal for batch operations
//
// Comparison with Vector Similarity:
// - Decay is MORE compute-bound (exp vs. multiply-add)
// - Expects slightly higher speedup from SIMD (~25% vs ~20%)

#![allow(missing_docs)] // Benchmark file doesn't require public API documentation

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};

/// Time constant for Ebbinghaus decay: 1 day in seconds
const TAU: f64 = 86400.0;

/// Rust baseline: Ebbinghaus decay implementation
///
/// This serves as the performance baseline for comparison.
/// Uses standard library exp() which may use SIMD internally on some platforms.
fn ebbinghaus_decay_rust(strengths: &mut [f32], ages_seconds: &[u64]) {
    assert_eq!(strengths.len(), ages_seconds.len());

    for (strength, &age) in strengths.iter_mut().zip(ages_seconds.iter()) {
        let age_f64 = age as f64;
        let decay_factor = (-age_f64 / TAU).exp();
        *strength *= decay_factor as f32;
    }
}

/// Zig kernel: Batch decay with SIMD
///
/// This calls into the Zig FFI when the zig-kernels feature is enabled.
/// Falls back to Rust implementation when feature is disabled.
#[cfg(feature = "zig-kernels")]
use engram_core::zig_kernels;

#[cfg(not(feature = "zig-kernels"))]
mod zig_kernels {
    /// Mock apply_decay for testing without zig-kernels feature
    pub fn apply_decay(strengths: &mut [f32], ages_seconds: &[u64]) {
        super::ebbinghaus_decay_rust(strengths, ages_seconds);
    }
}

/// Benchmark suite: Memory decay performance comparison
fn bench_decay_calculation(c: &mut Criterion) {
    let mut group = c.benchmark_group("decay_calculation");

    // Realistic batch sizes
    let batch_sizes = [100, 1000, 10_000];

    for &num_memories in &batch_sizes {
        // Generate test data with random ages
        let mut rng_state = 12345_u64;
        let ages: Vec<u64> = (0..num_memories)
            .map(|_| {
                // Simple LCG for reproducible "random" ages
                rng_state = rng_state.wrapping_mul(1_103_515_245).wrapping_add(12345);
                rng_state % 1_000_000 // Ages up to ~11 days
            })
            .collect();

        let strengths: Vec<f32> = (0..num_memories)
            .map(|i| (i as f32) / (num_memories as f32))
            .collect();

        let bench_id = format!("{num_memories}_memories");

        // Benchmark Rust baseline
        group.bench_with_input(
            BenchmarkId::new("rust", &bench_id),
            &(&strengths, &ages),
            |b, (strengths, ages)| {
                b.iter(|| {
                    let mut strengths_copy = (*strengths).clone();
                    ebbinghaus_decay_rust(&mut strengths_copy, ages);
                    black_box(strengths_copy);
                });
            },
        );

        // Benchmark Zig kernel
        group.bench_with_input(
            BenchmarkId::new("zig", &bench_id),
            &(&strengths, &ages),
            |b, (strengths, ages)| {
                b.iter(|| {
                    let mut strengths_copy = (*strengths).clone();
                    zig_kernels::apply_decay(&mut strengths_copy, ages);
                    black_box(strengths_copy);
                });
            },
        );
    }

    group.finish();
}

/// Benchmark: Uniform age distribution
///
/// All memories have the same age - tests SIMD efficiency with uniform inputs.
fn bench_uniform_ages(c: &mut Criterion) {
    let mut group = c.benchmark_group("uniform_ages");

    let num_memories = 10_000;
    let ages = vec![86400_u64; num_memories]; // All 1 day old
    let strengths: Vec<f32> = vec![0.8; num_memories];

    // Rust baseline
    group.bench_function("rust", |b| {
        b.iter(|| {
            let mut strengths_copy = strengths.clone();
            ebbinghaus_decay_rust(&mut strengths_copy, &ages);
            black_box(strengths_copy);
        });
    });

    // Zig kernel
    group.bench_function("zig", |b| {
        b.iter(|| {
            let mut strengths_copy = strengths.clone();
            zig_kernels::apply_decay(&mut strengths_copy, &ages);
            black_box(strengths_copy);
        });
    });

    group.finish();
}

/// Benchmark: Linear age progression
///
/// Ages increase linearly - tests cache prefetching with predictable pattern.
fn bench_linear_ages(c: &mut Criterion) {
    let mut group = c.benchmark_group("linear_ages");

    let num_memories = 10_000;
    let ages: Vec<u64> = (0..num_memories).map(|i| i as u64 * 1000).collect();
    let strengths: Vec<f32> = vec![1.0; num_memories];

    // Rust baseline
    group.bench_function("rust", |b| {
        b.iter(|| {
            let mut strengths_copy = strengths.clone();
            ebbinghaus_decay_rust(&mut strengths_copy, &ages);
            black_box(strengths_copy);
        });
    });

    // Zig kernel
    group.bench_function("zig", |b| {
        b.iter(|| {
            let mut strengths_copy = strengths.clone();
            zig_kernels::apply_decay(&mut strengths_copy, &ages);
            black_box(strengths_copy);
        });
    });

    group.finish();
}

/// Benchmark: Exponential age distribution
///
/// Ages distributed exponentially - simulates real-world memory access patterns.
fn bench_exponential_ages(c: &mut Criterion) {
    let mut group = c.benchmark_group("exponential_ages");

    let num_memories = 10_000;
    let ages: Vec<u64> = (0..num_memories)
        .map(|i| 2_u64.pow((i % 20) as u32))
        .collect();
    let strengths: Vec<f32> = vec![0.5; num_memories];

    // Rust baseline
    group.bench_function("rust", |b| {
        b.iter(|| {
            let mut strengths_copy = strengths.clone();
            ebbinghaus_decay_rust(&mut strengths_copy, &ages);
            black_box(strengths_copy);
        });
    });

    // Zig kernel
    group.bench_function("zig", |b| {
        b.iter(|| {
            let mut strengths_copy = strengths.clone();
            zig_kernels::apply_decay(&mut strengths_copy, &ages);
            black_box(strengths_copy);
        });
    });

    group.finish();
}

/// Benchmark: Mixed ages (realistic workload)
///
/// Combination of brand new, recent, old, and ancient memories.
fn bench_mixed_ages(c: &mut Criterion) {
    let mut group = c.benchmark_group("mixed_ages");

    let num_memories = 10_000;
    let ages: Vec<u64> = (0..num_memories)
        .map(|i| {
            match i % 5 {
                0 => 0,         // Brand new
                1 => 3600,      // 1 hour
                2 => 86400,     // 1 day
                3 => 604_800,   // 1 week
                _ => 2_592_000, // 1 month
            }
        })
        .collect();
    let strengths: Vec<f32> = (0..num_memories)
        .map(|i| (i as f32) / (num_memories as f32))
        .collect();

    // Rust baseline
    group.bench_function("rust", |b| {
        b.iter(|| {
            let mut strengths_copy = strengths.clone();
            ebbinghaus_decay_rust(&mut strengths_copy, &ages);
            black_box(strengths_copy);
        });
    });

    // Zig kernel
    group.bench_function("zig", |b| {
        b.iter(|| {
            let mut strengths_copy = strengths.clone();
            zig_kernels::apply_decay(&mut strengths_copy, &ages);
            black_box(strengths_copy);
        });
    });

    group.finish();
}

/// Benchmark: Cache-cold scenario
///
/// Large dataset to evict cache between iterations.
fn bench_cache_cold(c: &mut Criterion) {
    let mut group = c.benchmark_group("cache_cold");

    let num_memories = 100_000;
    let ages: Vec<u64> = (0..num_memories)
        .map(|i| (i as u64 * 100) % 1_000_000)
        .collect();
    let strengths: Vec<f32> = (0..num_memories)
        .map(|i| (i as f32) / (num_memories as f32))
        .collect();

    // Rust baseline
    group.bench_function("rust", |b| {
        b.iter(|| {
            let mut strengths_copy = strengths.clone();
            ebbinghaus_decay_rust(&mut strengths_copy, &ages);
            black_box(strengths_copy);
        });
    });

    // Zig kernel
    group.bench_function("zig", |b| {
        b.iter(|| {
            let mut strengths_copy = strengths.clone();
            zig_kernels::apply_decay(&mut strengths_copy, &ages);
            black_box(strengths_copy);
        });
    });

    group.finish();
}

/// Benchmark: Edge cases
///
/// Tests performance with pathological inputs.
fn bench_edge_cases(c: &mut Criterion) {
    let mut group = c.benchmark_group("edge_cases");

    // All zero ages (no decay)
    {
        let num_memories = 10_000;
        let ages = vec![0_u64; num_memories];
        let strengths: Vec<f32> = vec![1.0; num_memories];

        group.bench_function("zero_ages_rust", |b| {
            b.iter(|| {
                let mut strengths_copy = strengths.clone();
                ebbinghaus_decay_rust(&mut strengths_copy, &ages);
                black_box(strengths_copy);
            });
        });

        group.bench_function("zero_ages_zig", |b| {
            b.iter(|| {
                let mut strengths_copy = strengths.clone();
                zig_kernels::apply_decay(&mut strengths_copy, &ages);
                black_box(strengths_copy);
            });
        });
    }

    // All zero strengths
    {
        let num_memories = 10_000_usize;
        let ages: Vec<u64> = (0..num_memories).map(|i| (i as u64) * 1000).collect();
        let strengths: Vec<f32> = vec![0.0; num_memories];

        group.bench_function("zero_strengths_rust", |b| {
            b.iter(|| {
                let mut strengths_copy = strengths.clone();
                ebbinghaus_decay_rust(&mut strengths_copy, &ages);
                black_box(strengths_copy);
            });
        });

        group.bench_function("zero_strengths_zig", |b| {
            b.iter(|| {
                let mut strengths_copy = strengths.clone();
                zig_kernels::apply_decay(&mut strengths_copy, &ages);
                black_box(strengths_copy);
            });
        });
    }

    // Ancient memories (very large ages)
    {
        let num_memories = 10_000;
        let ages: Vec<u64> = vec![1_000_000_000; num_memories]; // ~31 years
        let strengths: Vec<f32> = vec![1.0; num_memories];

        group.bench_function("ancient_rust", |b| {
            b.iter(|| {
                let mut strengths_copy = strengths.clone();
                ebbinghaus_decay_rust(&mut strengths_copy, &ages);
                black_box(strengths_copy);
            });
        });

        group.bench_function("ancient_zig", |b| {
            b.iter(|| {
                let mut strengths_copy = strengths.clone();
                zig_kernels::apply_decay(&mut strengths_copy, &ages);
                black_box(strengths_copy);
            });
        });
    }

    group.finish();
}

/// Benchmark: Small batch sizes
///
/// Tests FFI overhead with small workloads.
fn bench_small_batches(c: &mut Criterion) {
    let mut group = c.benchmark_group("small_batches");

    for &size in &[1_usize, 5, 10, 50] {
        let ages: Vec<u64> = (0..size).map(|i| (i as u64) * 1000).collect();
        let strengths: Vec<f32> = vec![0.8; size];

        let bench_id = format!("{size}_memories");

        group.bench_with_input(
            BenchmarkId::new("rust", &bench_id),
            &(&strengths, &ages),
            |b, (strengths, ages)| {
                b.iter(|| {
                    let mut strengths_copy = (*strengths).clone();
                    ebbinghaus_decay_rust(&mut strengths_copy, ages);
                    black_box(strengths_copy);
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("zig", &bench_id),
            &(&strengths, &ages),
            |b, (strengths, ages)| {
                b.iter(|| {
                    let mut strengths_copy = (*strengths).clone();
                    zig_kernels::apply_decay(&mut strengths_copy, ages);
                    black_box(strengths_copy);
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_decay_calculation,
    bench_uniform_ages,
    bench_linear_ages,
    bench_exponential_ages,
    bench_mixed_ages,
    bench_cache_cold,
    bench_edge_cases,
    bench_small_batches
);
criterion_main!(benches);
