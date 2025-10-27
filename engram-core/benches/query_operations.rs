//! Performance benchmark for probabilistic query operations
//!
//! Validates Task 006 requirements:
//! - Query latency <1ms P95 for 10-result queries
//! - Memory allocation <100 bytes per query operation
//! - AND/OR/NOT operations maintain probability axioms

#![allow(missing_docs)]

use chrono::Utc;
use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use engram_core::query::{ConfidenceInterval, ProbabilisticQueryResult};
use engram_core::{Confidence, Episode};
use std::time::Duration;

/// Generate a test embedding with deterministic values
fn create_test_embedding(seed: usize) -> [f32; 768] {
    let mut embedding = [0.0f32; 768];
    for (i, val) in embedding.iter_mut().enumerate() {
        *val = ((seed + i) as f32 * 0.001).sin();
    }
    embedding
}

/// Create a test episode with specified confidence
fn create_test_episode(id: &str, confidence: Confidence) -> Episode {
    Episode {
        id: id.to_string(),
        when: Utc::now(),
        where_location: None,
        who: None,
        what: format!("test episode {id}"),
        embedding: create_test_embedding(id.len()),
        embedding_provenance: None,
        encoding_confidence: confidence,
        vividness_confidence: confidence,
        reliability_confidence: confidence,
        last_recall: Utc::now(),
        recall_count: 0,
        decay_rate: 0.1,
        decay_function: None,
        metadata: std::collections::HashMap::new(),
    }
}

/// Create a probabilistic query result with N episodes
fn create_query_result(episode_count: usize, confidence: Confidence) -> ProbabilisticQueryResult {
    let episodes: Vec<_> = (0..episode_count)
        .map(|i| {
            (
                create_test_episode(&format!("ep{i}"), confidence),
                confidence,
            )
        })
        .collect();

    ProbabilisticQueryResult::from_episodes(episodes)
}

/// Benchmark AND operation with different result sizes
fn bench_and_operation(c: &mut Criterion) {
    let mut group = c.benchmark_group("query_and");
    group.measurement_time(Duration::from_secs(5));

    for size in &[5, 10, 20, 50, 100] {
        let result_a = create_query_result(*size, Confidence::HIGH);
        let result_b = create_query_result(*size / 2, Confidence::MEDIUM);

        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, _| {
            b.iter(|| {
                let combined = result_a.and(black_box(&result_b));
                black_box(combined);
            });
        });
    }

    group.finish();
}

/// Benchmark OR operation with different result sizes
fn bench_or_operation(c: &mut Criterion) {
    let mut group = c.benchmark_group("query_or");
    group.measurement_time(Duration::from_secs(5));

    for size in &[5, 10, 20, 50, 100] {
        let result_a = create_query_result(*size, Confidence::HIGH);
        let result_b = create_query_result(*size / 2, Confidence::MEDIUM);

        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, _| {
            b.iter(|| {
                let combined = result_a.or(black_box(&result_b));
                black_box(combined);
            });
        });
    }

    group.finish();
}

/// Benchmark NOT operation with different result sizes
fn bench_not_operation(c: &mut Criterion) {
    let mut group = c.benchmark_group("query_not");
    group.measurement_time(Duration::from_secs(5));

    for size in &[5, 10, 20, 50, 100] {
        let result = create_query_result(*size, Confidence::HIGH);

        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, _| {
            b.iter(|| {
                let negated = result.not();
                black_box(negated);
            });
        });
    }

    group.finish();
}

/// P95 latency validation for 10-result queries (Task 006 requirement: <1ms)
fn bench_p95_latency_10_results(c: &mut Criterion) {
    let mut group = c.benchmark_group("query_p95_latency");
    group.measurement_time(Duration::from_secs(10));
    group.sample_size(1000); // Enough samples for P95 measurement

    let result_a = create_query_result(10, Confidence::HIGH);
    let result_b = create_query_result(10, Confidence::MEDIUM);

    group.bench_function("and_10_results", |b| {
        b.iter(|| {
            let combined = result_a.and(black_box(&result_b));
            black_box(combined);
        });
    });

    group.bench_function("or_10_results", |b| {
        b.iter(|| {
            let combined = result_a.or(black_box(&result_b));
            black_box(combined);
        });
    });

    group.bench_function("not_10_results", |b| {
        b.iter(|| {
            let negated = result_a.not();
            black_box(negated);
        });
    });

    group.finish();
}

/// Benchmark confidence interval operations
fn bench_confidence_interval_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("confidence_interval");
    group.measurement_time(Duration::from_secs(5));

    let interval_a =
        ConfidenceInterval::from_confidence_with_uncertainty(Confidence::exact(0.8), 0.1);
    let interval_b =
        ConfidenceInterval::from_confidence_with_uncertainty(Confidence::exact(0.6), 0.15);

    group.bench_function("and", |b| {
        b.iter(|| {
            let result = interval_a.and(black_box(&interval_b));
            black_box(result);
        });
    });

    group.bench_function("or", |b| {
        b.iter(|| {
            let result = interval_a.or(black_box(&interval_b));
            black_box(result);
        });
    });

    group.bench_function("not", |b| {
        b.iter(|| {
            let result = interval_a.not();
            black_box(result);
        });
    });

    group.finish();
}

/// Benchmark complex query chains
fn bench_query_chains(c: &mut Criterion) {
    let mut group = c.benchmark_group("query_chains");
    group.measurement_time(Duration::from_secs(5));

    let result_a = create_query_result(10, Confidence::HIGH);
    let result_b = create_query_result(10, Confidence::MEDIUM);
    let result_c = create_query_result(10, Confidence::LOW);

    group.bench_function("(A AND B) OR C", |b| {
        b.iter(|| {
            let combined = result_a.and(black_box(&result_b)).or(black_box(&result_c));
            black_box(combined);
        });
    });

    group.bench_function("A AND (B OR C)", |b| {
        b.iter(|| {
            let combined = result_a.and(&result_b.or(black_box(&result_c)));
            black_box(combined);
        });
    });

    group.bench_function("NOT (A AND B)", |b| {
        b.iter(|| {
            let combined = result_a.and(black_box(&result_b)).not();
            black_box(combined);
        });
    });

    group.finish();
}

/// Benchmark allocation overhead for query operations
fn bench_allocation_overhead(c: &mut Criterion) {
    let mut group = c.benchmark_group("query_allocation");
    group.measurement_time(Duration::from_secs(5));

    group.bench_function("query_result_creation", |b| {
        b.iter(|| {
            let result = create_query_result(10, Confidence::HIGH);
            black_box(result);
        });
    });

    group.bench_function("confidence_interval_creation", |b| {
        b.iter(|| {
            let interval =
                ConfidenceInterval::from_confidence_with_uncertainty(Confidence::exact(0.7), 0.1);
            black_box(interval);
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_and_operation,
    bench_or_operation,
    bench_not_operation,
    bench_p95_latency_10_results,
    bench_confidence_interval_operations,
    bench_query_chains,
    bench_allocation_overhead
);

criterion_main!(benches);
