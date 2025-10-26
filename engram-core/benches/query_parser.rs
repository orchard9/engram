//! Parser performance benchmark suite.
//!
//! Validates that parsing meets <100μs latency targets through:
//! - Arena allocation for AST nodes
//! - Hot-path inlining
//! - Lazy error construction
//! - Zero-copy string handling
//!
//! Performance targets:
//! - Simple RECALL: <50μs P90
//! - Complex multi-constraint: <100μs P90
//! - Large embedding (1536 floats): <200μs P90

#![allow(clippy::uninlined_format_args)]
#![allow(missing_docs)]

use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use engram_core::query::parser::{Parser, Query};

/// Parse a query string and return the AST.
fn parse_query(source: &str) -> Result<Query<'_>, Box<dyn std::error::Error>> {
    Ok(Parser::parse(source)?)
}

fn bench_parse_simple(c: &mut Criterion) {
    let query = "RECALL episode WHERE confidence > 0.7";

    c.bench_function("parse_simple_recall", |b| {
        b.iter(|| parse_query(black_box(query)).unwrap());
    });
}

fn bench_parse_complex(c: &mut Criterion) {
    let query = "SPREAD FROM node_123 MAX_HOPS 5 DECAY 0.15 THRESHOLD 0.1";

    c.bench_function("parse_complex_spread", |b| {
        b.iter(|| parse_query(black_box(query)).unwrap());
    });
}

fn bench_parse_with_constraints(c: &mut Criterion) {
    let query =
        "RECALL episode WHERE confidence > 0.7 AND timestamp < 1234567890 AND decay_factor >= 0.5";

    c.bench_function("parse_multi_constraint", |b| {
        b.iter(|| parse_query(black_box(query)).unwrap());
    });
}

fn bench_parse_embedding(c: &mut Criterion) {
    // Generate a large embedding literal (768 dimensions)
    let embedding_values: Vec<String> = (0..768)
        .map(|i| format!("{:.6}", i as f32 * 0.001))
        .collect();
    let embedding_literal = format!("[{}]", embedding_values.join(", "));
    let query = format!("RECALL {} THRESHOLD 0.8", embedding_literal);

    c.bench_function("parse_embedding_768d", |b| {
        b.iter(|| parse_query(black_box(&query)).unwrap());
    });
}

fn bench_parse_large_embedding(c: &mut Criterion) {
    // Generate an extra large embedding literal (1536 dimensions for GPT-4)
    let embedding_values: Vec<String> = (0..1536)
        .map(|i| format!("{:.6}", i as f32 * 0.001))
        .collect();
    let embedding_literal = format!("[{}]", embedding_values.join(", "));
    let query = format!("RECALL {} THRESHOLD 0.8", embedding_literal);

    c.bench_function("parse_embedding_1536d", |b| {
        b.iter(|| parse_query(black_box(&query)).unwrap());
    });
}

fn bench_parse_predict(c: &mut Criterion) {
    let query = "PREDICT pattern GIVEN context_node HORIZON 5m CONFIDENCE 0.95";

    c.bench_function("parse_predict", |b| {
        b.iter(|| parse_query(black_box(query)).unwrap());
    });
}

fn bench_parse_imagine(c: &mut Criterion) {
    let query = "IMAGINE creative_pattern BASED ON seed1 seed2 seed3 NOVELTY 0.8 CONFIDENCE 0.6";

    c.bench_function("parse_imagine", |b| {
        b.iter(|| parse_query(black_box(query)).unwrap());
    });
}

fn bench_parse_consolidate(c: &mut Criterion) {
    let query = "CONSOLIDATE recent_episodes INTO long_term_memory SCHEDULER ripple";

    c.bench_function("parse_consolidate", |b| {
        b.iter(|| parse_query(black_box(query)).unwrap());
    });
}

fn bench_parse_error_handling(c: &mut Criterion) {
    // Benchmark the error path to ensure lazy construction
    let invalid_query = "RECALL WHERE >"; // Missing pattern and malformed constraint

    c.bench_function("parse_error_path", |b| {
        b.iter(|| {
            let _ = parse_query(black_box(invalid_query));
        });
    });
}

fn bench_parse_varying_complexity(c: &mut Criterion) {
    let mut group = c.benchmark_group("parse_complexity");

    // Different complexity levels
    let queries = [
        ("minimal", "RECALL node_123"),
        ("simple", "RECALL episode WHERE confidence > 0.7"),
        ("medium", "SPREAD FROM node_123 MAX_HOPS 3 DECAY 0.15"),
        (
            "complex",
            "RECALL pattern WHERE confidence > 0.7 AND timestamp < 1234567890 AND decay_factor >= 0.5 CONFIDENCE 0.9 BASE_RATE 0.1",
        ),
    ];

    for (name, query) in &queries {
        group.throughput(Throughput::Bytes(query.len() as u64));
        group.bench_with_input(BenchmarkId::from_parameter(name), query, |b, q| {
            b.iter(|| parse_query(black_box(q)).unwrap());
        });
    }

    group.finish();
}

/// Performance regression test to ensure we meet targets.
/// This is a quick smoke test - Criterion provides the statistical analysis.
#[test]
fn test_performance_targets() {
    use std::time::Instant;

    // Simple query target: <50μs
    let simple = "RECALL episode WHERE confidence > 0.7";
    let start = Instant::now();
    for _ in 0..1000 {
        let _ = parse_query(simple);
    }
    let elapsed = start.elapsed().as_micros() / 1000;
    assert!(
        elapsed < 50,
        "Simple RECALL query too slow: {}μs (target: <50μs)",
        elapsed
    );

    // Complex query target: <100μs
    let complex = "SPREAD FROM node_123 MAX_HOPS 5 DECAY 0.15 THRESHOLD 0.1";
    let start = Instant::now();
    for _ in 0..1000 {
        let _ = parse_query(complex);
    }
    let elapsed = start.elapsed().as_micros() / 1000;
    assert!(
        elapsed < 100,
        "Complex SPREAD query too slow: {}μs (target: <100μs)",
        elapsed
    );

    // Large embedding target: <200μs
    let embedding_values: Vec<String> = (0..1536)
        .map(|i| format!("{:.6}", i as f32 * 0.001))
        .collect();
    let embedding_literal = format!("[{}]", embedding_values.join(", "));
    let large_embedding = format!("RECALL {} THRESHOLD 0.8", embedding_literal);
    let start = Instant::now();
    for _ in 0..100 {
        let _ = parse_query(&large_embedding);
    }
    let elapsed = start.elapsed().as_micros() / 100;
    assert!(
        elapsed < 200,
        "Large embedding query too slow: {}μs (target: <200μs)",
        elapsed
    );
}

criterion_group!(
    benches,
    bench_parse_simple,
    bench_parse_complex,
    bench_parse_with_constraints,
    bench_parse_embedding,
    bench_parse_large_embedding,
    bench_parse_predict,
    bench_parse_imagine,
    bench_parse_consolidate,
    bench_parse_error_handling,
    bench_parse_varying_complexity
);
criterion_main!(benches);
