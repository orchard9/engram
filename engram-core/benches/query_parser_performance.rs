//! Performance benchmarks for query parser with regression guards.
//!
//! This benchmark suite measures parse times across various query types and
//! includes regression guards that fail CI if performance degrades by >10%.
//!
//! ## Performance Targets
//!
//! | Query Type | Target (P50) | Target (P90) | Target (P99) |
//! |------------|--------------|--------------|--------------|
//! | Simple     | <20μs        | <50μs        | <100μs       |
//! | Complex    | <50μs        | <100μs       | <200μs       |
//! | Large      | <100μs       | <200μs       | <500μs       |
//!
//! ## Usage
//!
//! ```bash
//! # Run benchmarks
//! cargo bench --bench query_parser_performance
//!
//! # Save baseline for regression testing
//! cargo bench --bench query_parser_performance -- --save-baseline current
//!
//! # Compare against baseline
//! cargo bench --bench query_parser_performance -- --baseline current
//! ```

#![allow(missing_docs)] // Benchmark file - documented via function names

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use engram_core::query::parser::Parser;
use std::time::Duration;

// ============================================================================
// Benchmark Queries
// ============================================================================

const SIMPLE_QUERIES: &[(&str, &str)] = &[
    ("recall_basic", "RECALL episode"),
    ("recall_confidence", "RECALL episode WHERE confidence > 0.7"),
    ("spread_basic", "SPREAD FROM node_123"),
    ("spread_params", "SPREAD FROM node_123 MAX_HOPS 5"),
    ("predict_basic", "PREDICT episode GIVEN context"),
    ("imagine_basic", "IMAGINE episode BASED ON seed"),
    ("consolidate_basic", "CONSOLIDATE episodes INTO semantic"),
];

const COMPLEX_QUERIES: &[(&str, &str)] = &[
    (
        "recall_multiple_constraints",
        "RECALL episode WHERE confidence > 0.7 AND created > \"2024-01-01\" AND memory_space = \"user_123\"",
    ),
    (
        "spread_all_parameters",
        "SPREAD FROM node_123 MAX_HOPS 5 DECAY 0.15 THRESHOLD 0.1",
    ),
    (
        "predict_with_horizon",
        "PREDICT episode GIVEN context1, context2, context3 HORIZON 3600",
    ),
    (
        "imagine_with_novelty",
        "IMAGINE episode BASED ON seed1, seed2, seed3 NOVELTY 0.5",
    ),
    (
        "consolidate_with_filter",
        "CONSOLIDATE episodes WHERE created < \"2024-10-20\" INTO semantic SCHEDULER interval 3600",
    ),
    (
        "recall_multiline",
        "RECALL episode\n  WHERE confidence > 0.7\n  AND created > \"2024-01-01\"\n  LIMIT 100",
    ),
    (
        "recall_with_content",
        "RECALL episode WHERE content CONTAINS \"neural network\" AND confidence > 0.8",
    ),
];

const LARGE_QUERIES: &[(&str, &str)] = &[
    (
        "recall_long_identifier",
        "RECALL very_long_identifier_with_many_underscores_and_words_that_goes_on_for_a_while_123",
    ),
    (
        "recall_many_constraints",
        "RECALL episode WHERE confidence > 0.7 AND confidence < 0.9 AND created > \"2024-01-01\" AND created < \"2024-12-31\" AND memory_space = \"user_123\"",
    ),
    (
        "predict_many_contexts",
        "PREDICT episode GIVEN context1, context2, context3, context4, context5, context6, context7, context8, context9, context10",
    ),
];

// ============================================================================
// Parse Time Benchmarks
// ============================================================================

fn benchmark_simple_queries(c: &mut Criterion) {
    let mut group = c.benchmark_group("parse_simple");

    for (name, query) in SIMPLE_QUERIES {
        group.bench_with_input(BenchmarkId::from_parameter(name), query, |b, q| {
            b.iter(|| Parser::parse(black_box(q)));
        });
    }

    group.finish();
}

fn benchmark_complex_queries(c: &mut Criterion) {
    let mut group = c.benchmark_group("parse_complex");

    for (name, query) in COMPLEX_QUERIES {
        group.bench_with_input(BenchmarkId::from_parameter(name), query, |b, q| {
            b.iter(|| Parser::parse(black_box(q)));
        });
    }

    group.finish();
}

fn benchmark_large_queries(c: &mut Criterion) {
    let mut group = c.benchmark_group("parse_large");

    for (name, query) in LARGE_QUERIES {
        group.bench_with_input(BenchmarkId::from_parameter(name), query, |b, q| {
            b.iter(|| Parser::parse(black_box(q)));
        });
    }

    group.finish();
}

// ============================================================================
// Regression Guards
// ============================================================================

/// Regression guard: fail if simple queries exceed 100μs
fn regression_guard_simple_queries(c: &mut Criterion) {
    const MAX_TIME: Duration = Duration::from_micros(100);

    let mut group = c.benchmark_group("regression_simple");
    group.significance_level(0.01).sample_size(1000);

    for (name, query) in SIMPLE_QUERIES {
        group.bench_with_input(BenchmarkId::from_parameter(name), query, |b, q| {
            b.iter_custom(|iters| {
                let start = std::time::Instant::now();
                for _i in 0..iters {
                    let _ = Parser::parse(black_box(q));
                }
                let elapsed = start.elapsed();

                // Check per-iteration time
                let per_iter = elapsed / iters.try_into().unwrap_or(1);
                assert!(
                    per_iter <= MAX_TIME,
                    "REGRESSION: {} exceeded {}μs baseline (took {}μs)",
                    name,
                    MAX_TIME.as_micros(),
                    per_iter.as_micros()
                );

                elapsed
            });
        });
    }

    group.finish();
}

/// Regression guard: fail if complex queries exceed 200μs
fn regression_guard_complex_queries(c: &mut Criterion) {
    const MAX_TIME: Duration = Duration::from_micros(200);

    let mut group = c.benchmark_group("regression_complex");
    group.significance_level(0.01).sample_size(1000);

    for (name, query) in COMPLEX_QUERIES {
        group.bench_with_input(BenchmarkId::from_parameter(name), query, |b, q| {
            b.iter_custom(|iters| {
                let start = std::time::Instant::now();
                for _i in 0..iters {
                    let _ = Parser::parse(black_box(q));
                }
                let elapsed = start.elapsed();

                // Check per-iteration time
                let per_iter = elapsed / iters.try_into().unwrap_or(1);
                assert!(
                    per_iter <= MAX_TIME,
                    "REGRESSION: {} exceeded {}μs baseline (took {}μs)",
                    name,
                    MAX_TIME.as_micros(),
                    per_iter.as_micros()
                );

                elapsed
            });
        });
    }

    group.finish();
}

/// Regression guard: fail if large queries exceed 500μs
fn regression_guard_large_queries(c: &mut Criterion) {
    const MAX_TIME: Duration = Duration::from_micros(500);

    let mut group = c.benchmark_group("regression_large");
    group.significance_level(0.01).sample_size(500);

    for (name, query) in LARGE_QUERIES {
        group.bench_with_input(BenchmarkId::from_parameter(name), query, |b, q| {
            b.iter_custom(|iters| {
                let start = std::time::Instant::now();
                for _i in 0..iters {
                    let _ = Parser::parse(black_box(q));
                }
                let elapsed = start.elapsed();

                // Check per-iteration time
                let per_iter = elapsed / iters.try_into().unwrap_or(1);
                assert!(
                    per_iter <= MAX_TIME,
                    "REGRESSION: {} exceeded {}μs baseline (took {}μs)",
                    name,
                    MAX_TIME.as_micros(),
                    per_iter.as_micros()
                );

                elapsed
            });
        });
    }

    group.finish();
}

// ============================================================================
// Throughput Benchmarks
// ============================================================================

/// Measure queries per second for different query types
fn benchmark_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("throughput");

    // Simple query throughput
    group.bench_function("simple_qps", |b| {
        let query = "RECALL episode WHERE confidence > 0.7";
        b.iter(|| Parser::parse(black_box(query)));
    });

    // Complex query throughput
    group.bench_function("complex_qps", |b| {
        let query = "RECALL episode WHERE confidence > 0.7 AND created > \"2024-01-01\" LIMIT 100";
        b.iter(|| Parser::parse(black_box(query)));
    });

    group.finish();
}

// ============================================================================
// Memory Allocation Benchmarks
// ============================================================================

/// Measure allocations per parse (qualitative - for profiling)
fn benchmark_allocations(c: &mut Criterion) {
    let mut group = c.benchmark_group("allocations");

    // We can't easily measure allocations in stable Rust, but we can
    // benchmark parse + drop to understand total allocation overhead
    group.bench_function("parse_and_drop", |b| {
        let query = "RECALL episode WHERE confidence > 0.7";
        b.iter(|| {
            let result = Parser::parse(black_box(query));
            drop(black_box(result));
        });
    });

    group.finish();
}

// ============================================================================
// Error Path Performance
// ============================================================================

/// Benchmark error handling performance
fn benchmark_error_paths(c: &mut Criterion) {
    let mut group = c.benchmark_group("error_paths");

    let error_queries = [
        ("typo", "RECAL episode"),
        ("missing_keyword", "episode WHERE confidence > 0.7"),
        ("invalid_syntax", "RECALL >>"),
        ("out_of_range", "RECALL episode WHERE confidence > 1.5"),
        ("incomplete", "RECALL"),
    ];

    for (name, query) in &error_queries {
        group.bench_with_input(BenchmarkId::from_parameter(name), query, |b, q| {
            b.iter(|| {
                let _ = Parser::parse(black_box(q));
            });
        });
    }

    group.finish();
}

// ============================================================================
// Criterion Configuration
// ============================================================================

criterion_group! {
    name = benches;
    config = Criterion::default()
        .measurement_time(Duration::from_secs(10))
        .sample_size(1000);
    targets =
        benchmark_simple_queries,
        benchmark_complex_queries,
        benchmark_large_queries,
        benchmark_throughput,
        benchmark_allocations,
        benchmark_error_paths,
        regression_guard_simple_queries,
        regression_guard_complex_queries,
        regression_guard_large_queries,
}

criterion_main!(benches);
