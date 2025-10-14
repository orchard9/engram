//! ANN comparison benchmarks for Engram vs FAISS.

#![cfg(feature = "ann_benchmarks")]
#![allow(missing_docs)]

mod support;

use criterion::{Criterion, black_box};
use support::ann_common::BenchmarkFramework;
use support::datasets::DatasetLoader;
use support::engram_ann::EngramAnnIndex;
use support::faiss_ann::FaissAnnIndex;

#[cfg(feature = "ann_benchmarks")]
use support::annoy_ann::AnnoyAnnIndex;

pub fn run_benchmarks(c: &mut Criterion) {
    c.bench_function("ann_comparison_search_latency", |b| {
        b.iter(|| {
            let dataset = DatasetLoader::generate_synthetic(10_000, 100);

            let mut framework = BenchmarkFramework::new();
            framework.add_dataset(dataset);

            framework.add_implementation(Box::new(EngramAnnIndex::new()));
            framework.add_implementation(Box::new(
                FaissAnnIndex::new_hnsw(768, 16).expect("Failed to create FAISS index"),
            ));
            framework.add_implementation(Box::new(
                AnnoyAnnIndex::new(768, 50).expect("Failed to create Annoy index"),
            ));

            black_box(
                framework
                    .run_comparison()
                    .expect("Benchmark comparison failed"),
            );
        });
    });
}

#[cfg(feature = "ann_benchmarks")]
criterion::criterion_group!(ann_benchmarks, run_benchmarks);
#[cfg(feature = "ann_benchmarks")]
criterion::criterion_main!(ann_benchmarks);
