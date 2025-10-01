use anyhow::Result;
use std::collections::{HashMap, HashSet};
use std::convert::TryFrom;
use std::time::{Duration, Instant};

/// Trait for ANN index implementations
pub trait AnnIndex: Send + Sync {
    /// Build index from vectors
    fn build(&mut self, vectors: &[[f32; 768]]) -> Result<()>;

    /// Search for k nearest neighbors
    fn search(&self, query: &[f32; 768], k: usize) -> Vec<(usize, f32)>;

    /// Get index size in bytes
    fn memory_usage(&self) -> usize;

    /// Name for reporting
    fn name(&self) -> &'static str;
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
#[allow(dead_code)]
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
            .or_default()
            .insert(dataset_name, metrics);
    }

    #[allow(dead_code)]
    pub fn print_summary(&self) {
        println!("\n=== Benchmark Results ===\n");

        for (impl_name, datasets) in &self.results {
            println!("Implementation: {impl_name}");
            for (dataset, metrics) in datasets {
                println!("  Dataset: {dataset}");
                println!("    Build time: {:?}", metrics.build_time);
                println!("    Avg recall@10: {:.3}", metrics.avg_recall);
                println!("    Avg latency: {:?}", metrics.avg_latency);
                println!("    P95 latency: {:?}", metrics.p95_latency);
                println!("    P99 latency: {:?}", metrics.p99_latency);
                let memory_mb = metrics.memory_usage / (1024 * 1024);
                println!("    Memory usage: {memory_mb} MB");
                println!();
            }
        }
    }

    #[allow(dead_code)]
    pub fn export_csv(&self, path: &str) -> Result<()> {
        use std::fs::File;
        use std::io::Write;

        let mut file = File::create(path)?;
        writeln!(
            file,
            "implementation,dataset,build_time_ms,avg_recall,avg_latency_us,p95_latency_us,p99_latency_us,memory_mb"
        )?;

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

    pub const fn data(&self) -> &HashMap<String, HashMap<String, BenchmarkMetrics>> {
        &self.results
    }
}

/// Benchmark coordination framework
pub struct BenchmarkFramework {
    implementations: Vec<Box<dyn AnnIndex>>,
    datasets: Vec<AnnDataset>,
}

impl BenchmarkFramework {
    pub fn new() -> Self {
        Self {
            implementations: Vec::new(),
            datasets: Vec::new(),
        }
    }

    pub fn add_implementation(&mut self, implementation: Box<dyn AnnIndex>) {
        self.implementations.push(implementation);
    }

    pub fn add_dataset(&mut self, dataset: AnnDataset) {
        self.datasets.push(dataset);
    }

    pub fn run_comparison(&mut self) -> Result<BenchmarkResults> {
        let mut results = BenchmarkResults::default();

        for dataset in &self.datasets {
            for implementation in &mut self.implementations {
                let metrics = Self::benchmark_implementation(implementation.as_mut(), dataset)?;
                results.record(implementation.name(), dataset.name.clone(), metrics);
            }
        }

        Ok(results)
    }

    fn benchmark_implementation(
        index: &mut dyn AnnIndex,
        dataset: &AnnDataset,
    ) -> Result<BenchmarkMetrics> {
        let build_start = Instant::now();
        index.build(&dataset.vectors)?;
        let build_time = build_start.elapsed();

        // Warm-up phase
        if !dataset.queries.is_empty() {
            for query in dataset.queries.iter().take(10) {
                let _ = index.search(query, 10);
            }
        }

        let mut recalls = Vec::new();
        let mut latencies = Vec::new();

        for (query_idx, query) in dataset.queries.iter().enumerate() {
            let search_start = Instant::now();
            let results = index.search(query, 10);
            let latency = search_start.elapsed();

            let recall = calculate_recall(&results, &dataset.ground_truth[query_idx], 10);
            recalls.push(recall);
            latencies.push(latency);
        }

        latencies.sort();

        let avg_latency = if latencies.is_empty() {
            Duration::ZERO
        } else {
            let total_ns: u128 = latencies.iter().map(Duration::as_nanos).sum();
            let avg_ns = total_ns / u128::try_from(latencies.len()).unwrap_or(1);
            let clamped = avg_ns.min(u128::from(u64::MAX));
            let clamped_u64 = u64::try_from(clamped).unwrap_or(u64::MAX);
            Duration::from_nanos(clamped_u64)
        };

        let percentile_index = |percent: u32| -> usize {
            if latencies.is_empty() {
                return 0;
            }
            let len_minus_one = latencies.len().saturating_sub(1);
            let len_minus_one_u128 = u128::try_from(len_minus_one).unwrap_or(0);
            let numerator = len_minus_one_u128 * u128::from(percent) + 50;
            let idx_u128 = (numerator / 100).min(len_minus_one_u128);
            let idx = usize::try_from(idx_u128).unwrap_or(len_minus_one);
            idx.min(len_minus_one)
        };

        let p95_latency = if latencies.is_empty() {
            Duration::ZERO
        } else {
            latencies[percentile_index(95)]
        };
        let p99_latency = if latencies.is_empty() {
            Duration::ZERO
        } else {
            latencies[percentile_index(99)]
        };

        let avg_recall = if recalls.is_empty() {
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

        Ok(BenchmarkMetrics {
            build_time,
            avg_recall,
            avg_latency,
            p95_latency,
            p99_latency,
            memory_usage: index.memory_usage(),
        })
    }
}

pub fn calculate_recall(results: &[(usize, f32)], ground_truth: &[usize], k: usize) -> f32 {
    if k == 0 || ground_truth.is_empty() {
        return 0.0;
    }

    let limit = k.min(results.len());
    let result_set: HashSet<usize> = results.iter().take(limit).map(|(idx, _)| *idx).collect();
    let truth_limit = k.min(ground_truth.len());
    let ground_truth_set: HashSet<usize> = ground_truth.iter().take(truth_limit).copied().collect();

    let intersection = result_set.intersection(&ground_truth_set).count();
    #[allow(clippy::cast_precision_loss)]
    let numerator = intersection as f64;
    #[allow(clippy::cast_precision_loss)]
    let denominator = truth_limit as f64;
    if denominator == 0.0 {
        0.0
    } else {
        let ratio = numerator / denominator;
        #[allow(clippy::cast_possible_truncation)]
        let result = ratio as f32;
        result
    }
}
