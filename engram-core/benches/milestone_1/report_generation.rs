use crate::milestone_1::ComprehensiveBenchmarkResults;

#[derive(Debug, Clone)]
pub struct BenchmarkReportGenerator {
    template_header: String,
}

impl BenchmarkReportGenerator {
    pub fn new() -> Self {
        Self {
            template_header: "Comprehensive Benchmark Report\n========================="
                .to_string(),
        }
    }

    pub fn generate_comprehensive_report(&self, results: &ComprehensiveBenchmarkResults) -> String {
        let benchmark_summary = results
            .benchmark_results
            .as_ref()
            .map(|r| r.task_results().len())
            .unwrap_or_default();

        let detected_regressions = results
            .regression_analysis
            .as_ref()
            .map(|analysis| analysis.regressions.len())
            .unwrap_or_default();

        format!(
            "{}\nBenchmarked tasks: {}\nDetected regressions: {}\n",
            self.template_header, benchmark_summary, detected_regressions
        )
    }
}
