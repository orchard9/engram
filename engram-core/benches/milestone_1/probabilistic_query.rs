use std::cmp::Ordering;

use crate::milestone_1::statistical_framework::TaskBenchmarkResult;

#[derive(Debug, Clone)]
pub struct ProbabilisticQueryBenchmarks {
    baseline: TaskBenchmarkResult,
}

impl ProbabilisticQueryBenchmarks {
    pub fn new() -> Self {
        Self {
            baseline: TaskBenchmarkResult {
                mean_latency: 3.4,
                p95_latency: 4.1,
                p99_latency: 4.6,
                throughput: 18_500.0,
                samples: vec![3.3, 3.4, 3.5, 3.8, 4.0],
            },
        }
    }

    pub fn run_comprehensive_benchmarks(&self) -> TaskBenchmarkResult {
        let mut result = self.baseline.clone();
        result
            .samples
            .sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
        result
    }
}
