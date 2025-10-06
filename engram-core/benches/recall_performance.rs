//! Performance benchmark for cognitive recall
//!
//! Validates P95 latency < 10ms requirement from Task 008 specification

#![allow(missing_docs)]

use chrono::Utc;
use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use engram_core::{Confidence, Cue, EpisodeBuilder, MemoryStore};
use std::time::Duration;

/// Generate a test embedding with deterministic values
fn create_test_embedding(seed: usize) -> [f32; 768] {
    let mut embedding = [0.0f32; 768];
    for (i, val) in embedding.iter_mut().enumerate() {
        *val = ((seed + i) as f32 * 0.001).sin();
    }
    embedding
}

/// Create a memory store with N episodes
fn create_test_store(episode_count: usize) -> MemoryStore {
    let store = MemoryStore::new(episode_count * 2);

    for i in 0..episode_count {
        let episode = EpisodeBuilder::new()
            .id(format!("episode_{i:05}"))
            .when(Utc::now())
            .what(format!("Test episode {i} content"))
            .embedding(create_test_embedding(i))
            .confidence(Confidence::HIGH)
            .build();

        store.store(episode);
    }

    store
}

/// Benchmark similarity-based recall (baseline)
fn bench_similarity_recall(c: &mut Criterion) {
    let mut group = c.benchmark_group("recall_similarity");
    group.measurement_time(Duration::from_secs(10));

    for size in &[100, 1000, 10_000] {
        let store = create_test_store(*size);
        let cue = Cue::embedding(
            "recall_cue".to_string(),
            create_test_embedding(42),
            Confidence::HIGH,
        );

        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, _| {
            b.iter(|| {
                let results = store.recall(black_box(&cue));
                black_box(results);
            });
        });
    }

    group.finish();
}

/// Benchmark with different result limits
fn bench_result_limits(c: &mut Criterion) {
    let mut group = c.benchmark_group("recall_limits");
    group.measurement_time(Duration::from_secs(10));

    let store = create_test_store(10_000);

    for limit in &[5, 10, 20, 50, 100] {
        let cue = Cue::embedding(
            format!("limit_cue_{limit}"),
            create_test_embedding(123),
            Confidence::HIGH,
        );

        group.bench_with_input(BenchmarkId::from_parameter(limit), limit, |b, _| {
            b.iter(|| {
                let results = store.recall(black_box(&cue));
                black_box(results);
            });
        });
    }

    group.finish();
}

/// P95 latency validation benchmark
fn bench_p95_latency(c: &mut Criterion) {
    let mut group = c.benchmark_group("p95_latency");
    group.measurement_time(Duration::from_secs(20));

    // Configure for P95 measurement
    group.sample_size(1000); // Enough samples for P95

    let store = create_test_store(10_000);
    let cue = Cue::embedding(
        "p95_cue".to_string(),
        create_test_embedding(999),
        Confidence::HIGH,
    );

    group.bench_function("10k_episodes", |b| {
        b.iter(|| {
            let results = store.recall(black_box(&cue));
            black_box(results);
        });
    });

    group.finish();
}

/// Benchmark memory allocation overhead
fn bench_allocation_overhead(c: &mut Criterion) {
    let mut group = c.benchmark_group("allocation");
    group.measurement_time(Duration::from_secs(5));

    group.bench_function("cue_creation", |b| {
        b.iter(|| {
            let embedding = create_test_embedding(42);
            let cue = Cue::embedding("alloc_cue".to_string(), embedding, Confidence::HIGH);
            black_box(cue);
        });
    });

    group.bench_function("episode_creation", |b| {
        b.iter(|| {
            let episode = EpisodeBuilder::new()
                .id("test".to_string())
                .when(Utc::now())
                .what("test".to_string())
                .embedding(create_test_embedding(1))
                .confidence(Confidence::HIGH)
                .build();
            black_box(episode);
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_similarity_recall,
    bench_result_limits,
    bench_p95_latency,
    bench_allocation_overhead
);

criterion_main!(benches);
