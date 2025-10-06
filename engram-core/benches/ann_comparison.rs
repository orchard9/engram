//! Criterion harness comparing baseline ANN behaviours.
//!
//! Compares Engram's HNSW implementation against industry-standard libraries (FAISS, Annoy).
//! Requires the `ann_benchmarks` feature to be enabled for real library comparisons.
#![allow(missing_docs)]

// Ensure benchmarks are only run with the ann_benchmarks feature
#[cfg(not(feature = "ann_benchmarks"))]
compile_error!(
    "ANN benchmarks require 'ann_benchmarks' feature. Run with: cargo bench --features ann_benchmarks"
);

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};

mod support;

use support::ann_common::AnnIndex;
use support::datasets::DatasetLoader;
use support::engram_ann::EngramOptimizedAnnIndex;

#[cfg(feature = "ann_benchmarks")]
use support::annoy_ann::AnnoyAnnIndex;
#[cfg(feature = "ann_benchmarks")]
use support::faiss_ann::FaissAnnIndex;

/// Benchmark search performance across all ANN implementations
fn benchmark_ann_search(c: &mut Criterion) {
    let mut group = c.benchmark_group("ann_search_comparison");
    group.sample_size(50);

    // Use a moderately-sized dataset for benchmarking
    let dataset = DatasetLoader::generate_synthetic(1000, 100);

    // Build Engram index
    let mut engram = EngramOptimizedAnnIndex::new();
    engram
        .build(&dataset.vectors)
        .expect("Failed to build Engram index");

    // Build FAISS-HNSW index
    #[cfg(feature = "ann_benchmarks")]
    let mut faiss_hnsw = FaissAnnIndex::new_hnsw(768, 16).expect("Failed to create FAISS index");
    #[cfg(feature = "ann_benchmarks")]
    faiss_hnsw
        .build(&dataset.vectors)
        .expect("Failed to build FAISS index");

    // Build Annoy index
    #[cfg(feature = "ann_benchmarks")]
    let mut annoy = AnnoyAnnIndex::new(768, 10).expect("Failed to create Annoy index");
    #[cfg(feature = "ann_benchmarks")]
    annoy
        .build(&dataset.vectors)
        .expect("Failed to build Annoy index");

    // Benchmark Engram search
    group.bench_function("engram_search", |b| {
        b.iter(|| {
            for query in &dataset.queries {
                let results = engram.search(query, 10);
                black_box(results);
            }
        });
    });

    // Benchmark FAISS-HNSW search
    #[cfg(feature = "ann_benchmarks")]
    group.bench_function("faiss_hnsw_search", |b| {
        b.iter(|| {
            for query in &dataset.queries {
                let results = faiss_hnsw.search(query, 10);
                black_box(results);
            }
        });
    });

    // Benchmark Annoy search
    #[cfg(feature = "ann_benchmarks")]
    group.bench_function("annoy_search", |b| {
        b.iter(|| {
            for query in &dataset.queries {
                let results = annoy.search(query, 10);
                black_box(results);
            }
        });
    });

    group.finish();
}

/// Benchmark index build time across implementations
fn benchmark_ann_build(c: &mut Criterion) {
    let mut group = c.benchmark_group("ann_build_comparison");
    group.sample_size(10);

    let dataset = DatasetLoader::generate_synthetic(1000, 10);

    group.bench_function("engram_build", |b| {
        b.iter(|| {
            let mut index = EngramOptimizedAnnIndex::new();
            index.build(&dataset.vectors).expect("Build failed");
            black_box(index);
        });
    });

    #[cfg(feature = "ann_benchmarks")]
    group.bench_function("faiss_hnsw_build", |b| {
        b.iter(|| {
            let mut index = FaissAnnIndex::new_hnsw(768, 16).expect("Create failed");
            index.build(&dataset.vectors).expect("Build failed");
            black_box(index);
        });
    });

    #[cfg(feature = "ann_benchmarks")]
    group.bench_function("annoy_build", |b| {
        b.iter(|| {
            let mut index = AnnoyAnnIndex::new(768, 10).expect("Create failed");
            index.build(&dataset.vectors).expect("Build failed");
            black_box(index);
        });
    });

    group.finish();
}

/// Benchmark scalability with different dataset sizes
fn benchmark_ann_scalability(c: &mut Criterion) {
    let mut group = c.benchmark_group("ann_scalability");

    for size in [100, 500, 1000, 5000] {
        let dataset = DatasetLoader::generate_synthetic(size, 10);

        group.bench_with_input(BenchmarkId::new("engram", size), &dataset, |b, dataset| {
            let mut index = EngramOptimizedAnnIndex::new();
            index.build(&dataset.vectors).expect("Build failed");

            b.iter(|| {
                let results = index.search(&dataset.queries[0], 10);
                black_box(results);
            });
        });

        #[cfg(feature = "ann_benchmarks")]
        group.bench_with_input(
            BenchmarkId::new("faiss_hnsw", size),
            &dataset,
            |b, dataset| {
                let mut index = FaissAnnIndex::new_hnsw(768, 16).expect("Create failed");
                index.build(&dataset.vectors).expect("Build failed");

                b.iter(|| {
                    let results = index.search(&dataset.queries[0], 10);
                    black_box(results);
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    benchmark_ann_search,
    benchmark_ann_build,
    benchmark_ann_scalability
);
criterion_main!(benches);

#[cfg(test)]
mod tests {
    use super::support::ann_common::{
        AnnDataset, BenchmarkFramework, BenchmarkMetrics, BenchmarkResults, calculate_recall,
    };
    use std::time::Duration;

    #[test]
    fn test_recall_calculation() {
        let results = vec![(0, 0.9), (1, 0.8), (3, 0.7), (5, 0.6), (7, 0.5)];
        let ground_truth = vec![0, 1, 2, 3, 4];

        let recall = calculate_recall(&results, &ground_truth, 5);
        assert!((recall - 0.6).abs() < 0.01); // 3 out of 5
    }

    #[test]
    fn test_results_recording() {
        let mut results = BenchmarkResults::default();

        results.record(
            "test_impl",
            "test_dataset".to_string(),
            BenchmarkMetrics {
                build_time: Duration::from_secs(1),
                avg_recall: 0.95,
                avg_latency: Duration::from_micros(500),
                p95_latency: Duration::from_micros(900),
                p99_latency: Duration::from_micros(1200),
                memory_usage: 1024 * 1024 * 10,
            },
        );

        assert!(results.data().contains_key("test_impl"));
        assert!(
            results
                .data()
                .get("test_impl")
                .is_some_and(|datasets| datasets.contains_key("test_dataset"))
        );
    }

    #[test]
    fn test_framework_runs() {
        let mut framework = BenchmarkFramework::new();
        framework.add_implementation(Box::new(MockIndex));
        framework.add_dataset(AnnDataset {
            name: "mock".to_string(),
            vectors: vec![[0.0; 768]; 5],
            queries: vec![[0.0; 768]; 2],
            ground_truth: vec![vec![0], vec![1]],
        });

        let result = framework.run_comparison();
        assert!(result.is_ok());
    }

    #[derive(Default)]
    struct MockIndex;

    impl super::support::ann_common::AnnIndex for MockIndex {
        fn build(&mut self, _vectors: &[[f32; 768]]) -> anyhow::Result<()> {
            Ok(())
        }

        fn search(&mut self, _query: &[f32; 768], _k: usize) -> Vec<(usize, f32)> {
            vec![(0, 1.0)]
        }

        fn memory_usage(&self) -> usize {
            0
        }

        fn name(&self) -> &'static str {
            "mock"
        }
    }
}
