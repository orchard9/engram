//! Criterion harness comparing baseline ANN behaviours.
#![allow(missing_docs)]

use criterion::{Criterion, black_box, criterion_group, criterion_main};

mod support;

use support::ann_common::AnnDataset;

fn benchmark_ann_framework(c: &mut Criterion) {
    let mut group = c.benchmark_group("ann_framework");

    // Create a small test dataset
    let dataset = AnnDataset {
        name: "test".to_string(),
        vectors: vec![[0.1f32; 768]; 100],
        queries: vec![[0.1f32; 768]; 10],
        ground_truth: vec![vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9]; 10],
    };

    group.bench_function("framework_test", |b| {
        b.iter(|| {
            black_box(&dataset.vectors[0]);
        });
    });

    group.finish();
}

criterion_group!(benches, benchmark_ann_framework);
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

        fn search(&self, _query: &[f32; 768], _k: usize) -> Vec<(usize, f32)> {
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
