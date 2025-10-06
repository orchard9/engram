//! Benchmark for HNSW update latency
//!
//! Compares performance of async queue-based HNSW updates vs synchronous updates.

#![cfg(feature = "hnsw_index")]

use chrono::Utc;
use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use engram_core::{Confidence, EpisodeBuilder, MemoryStore};
use std::sync::Arc;

/// Benchmark store operations with HNSW async queue
fn bench_store_with_hnsw_async(c: &mut Criterion) {
    let mut group = c.benchmark_group("hnsw_async_updates");
    group.sample_size(50);

    for batch_size in [10, 100, 1000] {
        group.bench_with_input(
            BenchmarkId::new("async_queue", batch_size),
            &batch_size,
            |b, &batch_size| {
                let store = MemoryStore::new(100_000).with_hnsw_index();
                let store = Arc::new(store);
                store.start_hnsw_worker();

                b.iter(|| {
                    for i in 0..batch_size {
                        let episode = EpisodeBuilder::new()
                            .id(format!("ep_{}", rand::random::<u32>()))
                            .when(Utc::now())
                            .what(format!("Test content {i}"))
                            .embedding([0.5f32; 768])
                            .confidence(Confidence::MEDIUM)
                            .build();

                        let _ = black_box(store.store(episode));
                    }
                });

                store.shutdown_hnsw_worker();
            },
        );
    }

    group.finish();
}

/// Benchmark queue statistics overhead
fn bench_queue_stats(c: &mut Criterion) {
    let store = MemoryStore::new(100_000).with_hnsw_index();

    c.bench_function("hnsw_queue_stats", |b| {
        b.iter(|| {
            let stats = black_box(store.hnsw_queue_stats());
            black_box(stats);
        });
    });
}

/// Benchmark queue utilization under load
fn bench_queue_utilization(c: &mut Criterion) {
    let mut group = c.benchmark_group("queue_utilization");
    group.sample_size(20);

    for load_percent in [25, 50, 75, 100] {
        group.bench_with_input(
            BenchmarkId::new("utilization", load_percent),
            &load_percent,
            |b, &load_percent| {
                let store = MemoryStore::new(100_000).with_hnsw_index();
                let store = Arc::new(store);
                store.start_hnsw_worker();

                // Pre-fill queue to target utilization
                let target_depth = (10_000 * load_percent) / 100;
                for i in 0..target_depth {
                    let episode = EpisodeBuilder::new()
                        .id(format!("prefill_{i}"))
                        .when(Utc::now())
                        .what(format!("Prefill {i}"))
                        .embedding([0.5f32; 768])
                        .confidence(Confidence::LOW)
                        .build();
                    let _ = store.store(episode);
                }

                b.iter(|| {
                    let episode = EpisodeBuilder::new()
                        .id(format!("test_{}", rand::random::<u32>()))
                        .when(Utc::now())
                        .what("Benchmark test".to_string())
                        .embedding([0.5f32; 768])
                        .confidence(Confidence::MEDIUM)
                        .build();

                    let _ = black_box(store.store(episode));
                });

                store.shutdown_hnsw_worker();
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_store_with_hnsw_async,
    bench_queue_stats,
    bench_queue_utilization
);
criterion_main!(benches);
