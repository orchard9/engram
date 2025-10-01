use crate::milestone_1::ComprehensiveBenchmarkResults;
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct RegressionDetector {
    sensitivity_threshold: f64,
}

impl RegressionDetector {
    pub const fn new() -> Self {
        Self {
            sensitivity_threshold: 1.25,
        }
    }

    pub fn analyze_performance_trends(
        &self,
        results: &ComprehensiveBenchmarkResults,
    ) -> RegressionAnalysisResults {
        let mut regressions = HashMap::new();

        if let Some(benchmark_results) = &results.benchmark_results {
            for (task, metrics) in benchmark_results.task_results() {
                let baseline = metrics.mean_latency.max(f64::EPSILON);
                let normalized_p95 = metrics.p95_latency / baseline;

                if normalized_p95 > self.sensitivity_threshold {
                    regressions.insert(
                        task.clone(),
                        RegressionInfo {
                            metric: "p95_latency".to_string(),
                            magnitude: normalized_p95,
                            confidence: (normalized_p95 / self.sensitivity_threshold).min(1.0),
                        },
                    );
                }
            }
        }

        RegressionAnalysisResults { regressions }
    }
}

#[derive(Debug, Clone)]
pub struct RegressionAnalysisResults {
    pub regressions: HashMap<String, RegressionInfo>,
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct RegressionInfo {
    pub metric: String,
    pub magnitude: f64,
    pub confidence: f64,
}
