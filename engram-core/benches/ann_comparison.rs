//! ANN (Approximate Nearest Neighbor) benchmark framework
//!
//! Compares Engram's vector search performance against FAISS and Annoy
//! on standard datasets like SIFT1M and GloVe.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use std::collections::HashMap;
use std::time::{Duration, Instant};
use anyhow::Result;

/// Trait for ANN index implementations
pub trait AnnIndex: Send + Sync {
    /// Build index from vectors
    fn build(&mut self, vectors: &[[f32; 768]]) -> Result<()>;

    /// Search for k nearest neighbors
    fn search(&self, query: &[f32; 768], k: usize) -> Vec<(usize, f32)>;

    /// Get index size in bytes
    fn memory_usage(&self) -> usize;

    /// Name for reporting
    fn name(&self) -> &str;
}

/// Standard ANN dataset for benchmarking
#[derive(Debug, Clone)]
pub struct AnnDataset {
    pub name: String,
    pub vectors: Vec<[f32; 768]>,
    pub queries: Vec<[f32; 768]>,
    pub ground_truth: Vec<Vec<usize>>, // True k-NN for each query
}

/// Benchmark metrics for a single run
#[derive(Debug, Clone)]
pub struct BenchmarkMetrics {
    pub build_time: Duration,
    pub avg_recall: f32,
    pub avg_latency: Duration,
    pub p95_latency: Duration,
    pub p99_latency: Duration,
    pub memory_usage: usize,
}

/// Results aggregator
#[derive(Debug, Clone, Default)]
pub struct BenchmarkResults {
    results: HashMap<String, HashMap<String, BenchmarkMetrics>>,
}

impl BenchmarkResults {
    pub fn record(&mut self, impl_name: &str, dataset_name: String, metrics: BenchmarkMetrics) {
        self.results
            .entry(impl_name.to_string())
            .or_insert_with(HashMap::new)
            .insert(dataset_name, metrics);
    }

    pub fn print_summary(&self) {
        println!("\n=== Benchmark Results ===\n");

        for (impl_name, datasets) in &self.results {
            println!("Implementation: {}", impl_name);
            for (dataset, metrics) in datasets {
                println!("  Dataset: {}", dataset);
                println!("    Build time: {:?}", metrics.build_time);
                println!("    Avg recall@10: {:.3}", metrics.avg_recall);
                println!("    Avg latency: {:?}", metrics.avg_latency);
                println!("    P95 latency: {:?}", metrics.p95_latency);
                println!("    P99 latency: {:?}", metrics.p99_latency);
                println!("    Memory usage: {} MB", metrics.memory_usage / (1024 * 1024));
                println!();
            }
        }
    }

    pub fn export_csv(&self, path: &str) -> Result<()> {
        use std::fs::File;
        use std::io::Write;

        let mut file = File::create(path)?;
        writeln!(file, "implementation,dataset,build_time_ms,avg_recall,avg_latency_us,p95_latency_us,p99_latency_us,memory_mb")?;

        for (impl_name, datasets) in &self.results {
            for (dataset, metrics) in datasets {
                writeln!(
                    file,
                    "{},{},{},{:.3},{},{},{},{}",
                    impl_name,
                    dataset,
                    metrics.build_time.as_millis(),
                    metrics.avg_recall,
                    metrics.avg_latency.as_micros(),
                    metrics.p95_latency.as_micros(),
                    metrics.p99_latency.as_micros(),
                    metrics.memory_usage / (1024 * 1024)
                )?;
            }
        }

        Ok(())
    }
}

/// Main benchmark framework
pub struct BenchmarkFramework {
    implementations: Vec<Box<dyn AnnIndex>>,
    datasets: Vec<AnnDataset>,
    results: BenchmarkResults,
}

impl BenchmarkFramework {
    pub fn new() -> Self {
        Self {
            implementations: Vec::new(),
            datasets: Vec::new(),
            results: BenchmarkResults::default(),
        }
    }

    pub fn add_implementation(&mut self, implementation: Box<dyn AnnIndex>) {
        self.implementations.push(implementation);
    }

    pub fn add_dataset(&mut self, dataset: AnnDataset) {
        self.datasets.push(dataset);
    }

    pub fn run_comparison(&mut self) -> BenchmarkResults {
        for dataset in &self.datasets {
            for implementation in &mut self.implementations {
                println!("Benchmarking {} on {}...", implementation.name(), dataset.name);
                self.benchmark_implementation(implementation.as_mut(), dataset);
            }
        }

        self.results.clone()
    }

    fn benchmark_implementation(
        &mut self,
        index: &mut dyn AnnIndex,
        dataset: &AnnDataset,
    ) {
        // Build index
        let build_start = Instant::now();
        index.build(&dataset.vectors).expect("Failed to build index");
        let build_time = build_start.elapsed();

        // Warm-up
        for _ in 0..10 {
            let _ = index.search(&dataset.queries[0], 10);
        }

        // Measure recall and latency
        let mut recalls = Vec::new();
        let mut latencies = Vec::new();

        for (query_idx, query) in dataset.queries.iter().enumerate() {
            let search_start = Instant::now();
            let results = index.search(query, 10);
            let latency = search_start.elapsed();

            let recall = self.calculate_recall(
                &results,
                &dataset.ground_truth[query_idx],
                10,
            );

            recalls.push(recall);
            latencies.push(latency);
        }

        // Calculate percentiles
        latencies.sort();
        let p95_idx = (latencies.len() as f64 * 0.95) as usize;
        let p99_idx = (latencies.len() as f64 * 0.99) as usize;

        // Record results
        self.results.record(
            index.name(),
            dataset.name.clone(),
            BenchmarkMetrics {
                build_time,
                avg_recall: recalls.iter().sum::<f32>() / recalls.len() as f32,
                avg_latency: latencies.iter().sum::<Duration>() / latencies.len() as u32,
                p95_latency: latencies[p95_idx.min(latencies.len() - 1)],
                p99_latency: latencies[p99_idx.min(latencies.len() - 1)],
                memory_usage: index.memory_usage(),
            },
        );
    }

    fn calculate_recall(&self, results: &[(usize, f32)], ground_truth: &[usize], k: usize) -> f32 {
        let result_set: std::collections::HashSet<usize> =
            results.iter().take(k).map(|(idx, _)| *idx).collect();

        let ground_truth_set: std::collections::HashSet<usize> =
            ground_truth.iter().take(k).cloned().collect();

        let intersection = result_set.intersection(&ground_truth_set).count();

        intersection as f32 / k.min(ground_truth.len()) as f32
    }
}

// Dummy benchmark for testing the framework
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
    use super::*;

    #[test]
    fn test_recall_calculation() {
        let framework = BenchmarkFramework::new();

        let results = vec![(0, 0.9), (1, 0.8), (3, 0.7), (5, 0.6), (7, 0.5)];
        let ground_truth = vec![0, 1, 2, 3, 4];

        let recall = framework.calculate_recall(&results, &ground_truth, 5);
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

        assert!(results.results.contains_key("test_impl"));
        assert!(results.results["test_impl"].contains_key("test_dataset"));
    }
}