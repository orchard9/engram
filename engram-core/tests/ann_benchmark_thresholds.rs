//! ANN benchmark threshold validation tests
#![allow(missing_docs)]
#[cfg(feature = "ann_benchmarks")]
#[path = "../benches/support/mod.rs"]
mod bench_support;

#[cfg(feature = "ann_benchmarks")]
use bench_support::ann_common::BenchmarkFramework;
#[cfg(feature = "ann_benchmarks")]
use bench_support::annoy_ann::AnnoyAnnIndex;
#[cfg(feature = "ann_benchmarks")]
use bench_support::datasets::DatasetLoader;
#[cfg(feature = "ann_benchmarks")]
use bench_support::engram_ann::EngramAnnIndex;
#[cfg(feature = "ann_benchmarks")]
use bench_support::faiss_ann::FaissAnnIndex;
#[cfg(feature = "ann_benchmarks")]
use std::time::Duration;

#[cfg(feature = "ann_benchmarks")]
#[test]
fn ann_benchmarks_meet_thresholds() {
    let dataset = DatasetLoader::generate_synthetic(1_000, 32);

    let mut framework = BenchmarkFramework::new();
    framework.add_dataset(dataset);
    framework.add_implementation(Box::new(EngramAnnIndex::new()));
    framework.add_implementation(Box::new(
        FaissAnnIndex::new_hnsw(768, 16).expect("failed to create FAISS index"),
    ));
    framework.add_implementation(Box::new(
        AnnoyAnnIndex::new(768, 25).expect("failed to create Annoy index"),
    ));

    let results = framework
        .run_comparison()
        .expect("ANN comparison failed in test");
    results
        .assert_thresholds(0.90, Duration::from_micros(1_000))
        .expect("ANN benchmark thresholds violated");
}
