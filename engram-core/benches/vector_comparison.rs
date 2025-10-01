//! Main vector comparison benchmark
//!
//! Compares Engram, FAISS, and Annoy on standard ANN datasets

#![allow(missing_docs)]

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use std::convert::TryFrom;
use std::time::Duration;

mod support;

use support::ann_common::{AnnDataset, AnnIndex, BenchmarkFramework, BenchmarkResults};
use support::datasets::DatasetLoader;
use support::engram_ann::EngramOptimizedAnnIndex;
use support::mock_annoy::MockAnnoyIndex;
use support::mock_faiss::MockFaissIndex;

fn dataset_or_panic(label: &str, loader: impl FnOnce() -> AnnDataset) -> AnnDataset {
    let dataset = loader();
    assert!(
        !dataset.vectors.is_empty(),
        "Dataset {label} produced no vectors"
    );
    dataset
}

fn faiss_or_panic(dims: usize, neighbors: usize) -> MockFaissIndex {
    MockFaissIndex::new_hnsw(dims, neighbors)
}

const fn annoy_or_panic(dims: usize, trees: usize) -> MockAnnoyIndex {
    MockAnnoyIndex::new(dims, trees)
}

fn build_index_or_panic(index: &mut dyn AnnIndex, vectors: &[[f32; 768]], label: &str) {
    if let Err(err) = index.build(vectors) {
        panic!("Failed to build {label}: {err}");
    }
}

/// Run comprehensive comparison
fn run_full_comparison() -> BenchmarkResults {
    let mut framework = BenchmarkFramework::new();

    // Add implementations
    framework.add_implementation(Box::new(EngramOptimizedAnnIndex::new()));
    framework.add_implementation(Box::new(faiss_or_panic(768, 16)));
    framework.add_implementation(Box::new(annoy_or_panic(768, 10)));

    // Add datasets
    framework.add_dataset(dataset_or_panic("synthetic", || {
        DatasetLoader::generate_synthetic(1000, 100)
    }));
    framework.add_dataset(dataset_or_panic("sift1m", DatasetLoader::load_sift1m_mock));
    framework.add_dataset(dataset_or_panic("glove", DatasetLoader::load_glove_mock));

    let results = match framework.run_comparison() {
        Ok(results) => results,
        Err(err) => panic!("Comparison failed: {err}"),
    };

    results.print_summary();

    if let Err(err) = results.export_csv("benchmark_results.csv") {
        eprintln!("Failed to export CSV: {err}");
    }

    results
}

/// Benchmark recall performance
fn benchmark_recall(c: &mut Criterion) {
    let dataset = dataset_or_panic("synthetic", || DatasetLoader::generate_synthetic(1000, 10));

    let mut group = c.benchmark_group("recall_at_10");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(10));

    // Build indexes once
    let mut engram = EngramOptimizedAnnIndex::new();
    build_index_or_panic(&mut engram, &dataset.vectors, "Engram");

    let mut faiss = faiss_or_panic(768, 16);
    build_index_or_panic(&mut faiss, &dataset.vectors, "FAISS");

    let mut annoy = annoy_or_panic(768, 10);
    build_index_or_panic(&mut annoy, &dataset.vectors, "Annoy");

    // Benchmark Engram
    group.bench_function("engram", |b| {
        b.iter(|| {
            for query in &dataset.queries {
                black_box(engram.search(query, 10));
            }
        });
    });

    // Benchmark FAISS
    group.bench_function("faiss", |b| {
        b.iter(|| {
            for query in &dataset.queries {
                black_box(faiss.search(query, 10));
            }
        });
    });

    // Benchmark Annoy
    group.bench_function("annoy", |b| {
        b.iter(|| {
            for query in &dataset.queries {
                black_box(annoy.search(query, 10));
            }
        });
    });

    group.finish();
}

/// Benchmark build time
fn benchmark_build_time(c: &mut Criterion) {
    let mut group = c.benchmark_group("build_time");
    group.sample_size(10);

    for size in [100, 500, 1000] {
        let dataset = dataset_or_panic("synthetic", || DatasetLoader::generate_synthetic(size, 10));

        group.bench_with_input(BenchmarkId::new("engram", size), &size, |b, _| {
            b.iter(|| {
                let mut index = EngramOptimizedAnnIndex::new();
                build_index_or_panic(&mut index, &dataset.vectors, "Engram");
            });
        });

        group.bench_with_input(BenchmarkId::new("faiss", size), &size, |b, _| {
            b.iter(|| {
                let mut index = faiss_or_panic(768, 16);
                build_index_or_panic(&mut index, &dataset.vectors, "FAISS");
            });
        });

        group.bench_with_input(BenchmarkId::new("annoy", size), &size, |b, _| {
            b.iter(|| {
                let mut index = annoy_or_panic(768, 10);
                build_index_or_panic(&mut index, &dataset.vectors, "Annoy");
            });
        });
    }

    group.finish();
}

/// Benchmark memory usage
fn benchmark_memory(c: &mut Criterion) {
    let dataset = dataset_or_panic("synthetic", || DatasetLoader::generate_synthetic(1000, 10));

    let mut group = c.benchmark_group("memory_usage");
    group.sample_size(10);

    let mut engram = EngramOptimizedAnnIndex::new();
    build_index_or_panic(&mut engram, &dataset.vectors, "Engram");

    let mut faiss = faiss_or_panic(768, 16);
    build_index_or_panic(&mut faiss, &dataset.vectors, "FAISS");

    let mut annoy = annoy_or_panic(768, 10);
    build_index_or_panic(&mut annoy, &dataset.vectors, "Annoy");

    group.bench_function("engram", |b| {
        b.iter(|| {
            black_box(engram.memory_usage());
        });
    });

    group.bench_function("faiss", |b| {
        b.iter(|| {
            black_box(faiss.memory_usage());
        });
    });

    group.bench_function("annoy", |b| {
        b.iter(|| {
            black_box(annoy.memory_usage());
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    benchmark_recall,
    benchmark_build_time,
    benchmark_memory
);
criterion_main!(benches);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_full_comparison() {
        let results = run_full_comparison();

        // Check that all implementations were tested
        assert!(results.data().contains_key("Engram-Optimized"));
        assert!(results.data().contains_key("FAISS (Mock)"));
        assert!(results.data().contains_key("Annoy (Mock)"));
    }

    #[test]
    fn test_engram_meets_requirements() {
        let dataset = DatasetLoader::generate_synthetic(1000, 100);

        let mut engram = EngramOptimizedAnnIndex::new();
        engram.build(&dataset.vectors).expect("Failed to build");

        let mut recalls = Vec::new();
        let mut latencies = Vec::new();

        for (query_idx, query) in dataset.queries.iter().enumerate().take(10) {
            let start = std::time::Instant::now();
            let results = engram.search(query, 10);
            let latency = start.elapsed();

            let recall = support::ann_common::calculate_recall(
                &results,
                &dataset.ground_truth[query_idx],
                10,
            );
            recalls.push(recall);
            latencies.push(latency);
        }

        let avg_recall: f32 = if recalls.is_empty() {
            0.0
        } else {
            let count = u64::try_from(recalls.len()).unwrap_or(0);
            if count == 0 {
                0.0
            } else {
                let sum = recalls.iter().map(|&value| f64::from(value)).sum::<f64>();
                #[allow(clippy::cast_precision_loss)]
                let average = sum / count as f64;
                #[allow(clippy::cast_possible_truncation)]
                let result = average as f32;
                result
            }
        };
        assert!(avg_recall > 0.8);

        let avg_latency = if latencies.is_empty() {
            0
        } else {
            let total: u128 = latencies.iter().map(std::time::Duration::as_micros).sum();
            let count = u128::try_from(latencies.len()).unwrap_or(1);
            total / count
        };
        assert!(avg_latency < 5_000);
    }
}
