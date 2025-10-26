//! Overhead validation benchmark for zero-cost metrics infrastructure
//!
//! This benchmark validates that:
//! 1. When monitoring disabled, there is literally zero overhead (code eliminated)
//! 2. When monitoring enabled, overhead is <1% of realistic workload
//! 3. Single metric operations meet performance budgets (<25ns for counters, <80ns for histograms)
//!
//! Run with:
//! ```bash
//! # Without monitoring (baseline - should be identical to uninstrumented)
//! cargo bench --bench metrics_overhead --no-default-features
//!
//! # With monitoring (should show <1% overhead)
//! cargo bench --bench metrics_overhead --features monitoring
//! ```

#![allow(missing_docs)]

use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use std::time::Duration;

#[cfg(feature = "monitoring")]
use engram_core::metrics::cognitive_patterns::{
    CognitivePatternMetrics, InterferenceType, PrimingType,
};

/// Baseline: operation without any metrics
fn baseline_operation() -> f32 {
    // Simulate a typical spreading activation step
    let mut sum = 0.0f32;
    for i in 0..100 {
        sum += (i as f32).sin();
    }
    sum
}

/// Operation with metrics recording (only compiles if monitoring enabled)
#[cfg(feature = "monitoring")]
fn instrumented_operation(metrics: &CognitivePatternMetrics) -> f32 {
    let mut sum = 0.0f32;
    for i in 0..100 {
        sum += (i as f32).sin();

        // Record priming event every 10 iterations (realistic pattern)
        if i % 10 == 0 {
            metrics.record_priming(PrimingType::Semantic, sum / 100.0);
        }
    }
    sum
}

fn benchmark_metrics_overhead(c: &mut Criterion) {
    let mut group = c.benchmark_group("metrics_overhead");
    group.sample_size(1000);
    group.warm_up_time(Duration::from_secs(3));

    // Baseline without metrics
    group.bench_function("baseline_no_metrics", |b| {
        b.iter(|| {
            black_box(baseline_operation());
        });
    });

    // With metrics enabled (only compiles if monitoring feature enabled)
    #[cfg(feature = "monitoring")]
    {
        let metrics = CognitivePatternMetrics::new();

        group.bench_function("with_metrics_recording", |b| {
            b.iter(|| {
                black_box(instrumented_operation(&metrics));
            });
        });
    }

    group.finish();
}

#[cfg(feature = "monitoring")]
fn benchmark_single_record_latency(c: &mut Criterion) {
    let mut group = c.benchmark_group("single_record_latency");
    group.sample_size(10000);

    let metrics = CognitivePatternMetrics::new();

    // Counter increment (target: <25ns hot path)
    group.bench_function("record_priming_hot", |b| {
        b.iter(|| {
            metrics.record_priming(black_box(PrimingType::Semantic), black_box(0.75));
        });
    });

    // Histogram record (target: <80ns hot path)
    group.bench_function("record_interference_hot", |b| {
        b.iter(|| {
            metrics.record_interference(black_box(InterferenceType::Proactive), black_box(0.65));
        });
    });

    // Reconsolidation (counter + conditional, target: <25ns)
    group.bench_function("record_reconsolidation", |b| {
        b.iter(|| {
            metrics.record_reconsolidation(black_box(0.5));
        });
    });

    // Simple counter increment
    group.bench_function("record_false_memory", |b| {
        b.iter(|| {
            metrics.record_false_memory();
        });
    });

    group.finish();
}

#[cfg(not(feature = "monitoring"))]
fn benchmark_single_record_latency(_c: &mut Criterion) {
    // When monitoring disabled, skip this benchmark group
}

#[cfg(feature = "monitoring")]
fn benchmark_throughput_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("throughput_scaling");

    let metrics = CognitivePatternMetrics::new();

    for count in [100, 1000, 10_000, 100_000] {
        group.throughput(Throughput::Elements(count));
        group.bench_with_input(
            BenchmarkId::new("record_priming", count),
            &count,
            |b, &count| {
                b.iter(|| {
                    for i in 0..count {
                        metrics.record_priming(PrimingType::Semantic, (i as f32) / (count as f32));
                    }
                });
            },
        );
    }

    group.finish();
}

#[cfg(not(feature = "monitoring"))]
fn benchmark_throughput_scaling(_c: &mut Criterion) {
    // When monitoring disabled, skip this benchmark group
}

#[cfg(feature = "monitoring")]
fn benchmark_concurrent_contention(c: &mut Criterion) {
    let mut group = c.benchmark_group("concurrent_contention");
    group.sample_size(100);

    let metrics = std::sync::Arc::new(CognitivePatternMetrics::new());

    // Single-threaded baseline
    group.bench_function("single_thread", |b| {
        b.iter(|| {
            for _ in 0..1000 {
                metrics.record_priming(PrimingType::Semantic, 0.5);
            }
        });
    });

    // Multi-threaded contention
    for num_threads in [2, 4, 8] {
        group.bench_function(BenchmarkId::new("threads", num_threads), |b| {
            b.iter(|| {
                let handles: Vec<_> = (0..num_threads)
                    .map(|_| {
                        let metrics = std::sync::Arc::clone(&metrics);
                        std::thread::spawn(move || {
                            for _ in 0..1000 {
                                metrics.record_priming(PrimingType::Semantic, 0.5);
                            }
                        })
                    })
                    .collect();

                for h in handles {
                    h.join().expect("thread panicked");
                }
            });
        });
    }

    group.finish();
}

#[cfg(not(feature = "monitoring"))]
fn benchmark_concurrent_contention(_c: &mut Criterion) {
    // When monitoring disabled, skip this benchmark group
}

#[cfg(feature = "monitoring")]
fn benchmark_query_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("query_operations");
    group.sample_size(10000);

    let metrics = CognitivePatternMetrics::new();

    // Pre-populate with data
    for i in 0..10000 {
        metrics.record_priming(PrimingType::Semantic, (i % 100) as f32 / 100.0);
        metrics.record_interference(InterferenceType::Proactive, (i % 50) as f32 / 50.0);
    }

    // Counter read (target: <10ns)
    group.bench_function("read_counter", |b| {
        b.iter(|| {
            black_box(metrics.priming_events_total());
        });
    });

    // Type-specific counter read
    group.bench_function("read_type_counter", |b| {
        b.iter(|| {
            black_box(metrics.priming_type_count(PrimingType::Semantic));
        });
    });

    // Histogram mean (target: <100ns)
    group.bench_function("read_histogram_mean", |b| {
        b.iter(|| {
            black_box(metrics.priming_mean_strength());
        });
    });

    // Computed metric (hit rate)
    group.bench_function("read_computed_metric", |b| {
        b.iter(|| {
            black_box(metrics.reconsolidation_window_hit_rate());
        });
    });

    group.finish();
}

#[cfg(not(feature = "monitoring"))]
fn benchmark_query_operations(_c: &mut Criterion) {
    // When monitoring disabled, skip this benchmark group
}

#[cfg(feature = "monitoring")]
fn benchmark_cache_effects(c: &mut Criterion) {
    let mut group = c.benchmark_group("cache_effects");

    let metrics = CognitivePatternMetrics::new();

    // Hot path (L1 cached) - rapid repeated access to same metric
    group.bench_function("hot_path_l1", |b| {
        b.iter(|| {
            for _ in 0..1000 {
                metrics.record_priming(PrimingType::Semantic, 0.5);
            }
        });
    });

    // Warm path (L3 cached) - access multiple metrics in rotation
    group.bench_function("warm_path_l3", |b| {
        b.iter(|| {
            for i in 0..1000 {
                match i % 3 {
                    0 => metrics.record_priming(PrimingType::Semantic, 0.5),
                    1 => metrics.record_interference(InterferenceType::Proactive, 0.6),
                    2 => metrics.record_reconsolidation(0.7),
                    _ => unreachable!(),
                }
            }
        });
    });

    group.finish();
}

#[cfg(not(feature = "monitoring"))]
fn benchmark_cache_effects(_c: &mut Criterion) {
    // When monitoring disabled, skip this benchmark group
}

criterion_group!(
    benches,
    benchmark_metrics_overhead,
    benchmark_single_record_latency,
    benchmark_throughput_scaling,
    benchmark_concurrent_contention,
    benchmark_query_operations,
    benchmark_cache_effects
);
criterion_main!(benches);
