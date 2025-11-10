//! Report generation and validation

use crate::metrics_collector::{MetricsCollector, OperationStatistics};
use crate::workload_generator::{OperationType, WorkloadConfig};
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::path::Path;

pub struct ReportGenerator {
    metrics: MetricsCollector,
}

impl ReportGenerator {
    pub fn new(metrics: MetricsCollector) -> Self {
        Self { metrics }
    }

    pub fn generate(&self, config: &WorkloadConfig) -> Result<Report> {
        let stats = self.metrics.get_statistics();

        let overall_throughput = if stats.elapsed_seconds > 0.0 {
            stats.total_operations as f64 / stats.elapsed_seconds
        } else {
            0.0
        };

        let overall_error_rate = if stats.total_operations > 0 {
            stats.total_errors as f64 / stats.total_operations as f64
        } else {
            0.0
        };

        // Calculate overall latency percentiles (max across all operation types)
        let mut max_p99 = 0.0f64;
        let mut max_p95 = 0.0f64;
        let mut max_p50 = 0.0f64;

        for op_stats in stats.per_operation.values() {
            max_p99 = max_p99.max(op_stats.p99_latency_ms());
            max_p95 = max_p95.max(op_stats.p95_latency_ms());
            max_p50 = max_p50.max(op_stats.p50_latency_ms());
        }

        Ok(Report {
            config_name: config.name().to_string(),
            elapsed_seconds: stats.elapsed_seconds,
            total_operations: stats.total_operations,
            total_errors: stats.total_errors,
            overall_throughput,
            overall_error_rate,
            p50_latency_ms: max_p50,
            p95_latency_ms: max_p95,
            p99_latency_ms: max_p99,
            per_operation_stats: stats.per_operation.clone(),
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Report {
    pub config_name: String,
    pub elapsed_seconds: f64,
    pub total_operations: u64,
    pub total_errors: u64,
    pub overall_throughput: f64,
    pub overall_error_rate: f64,
    pub p50_latency_ms: f64,
    pub p95_latency_ms: f64,
    pub p99_latency_ms: f64,
    pub per_operation_stats: std::collections::HashMap<OperationType, OperationStatistics>,
}

impl Report {
    pub fn summary(&self) -> String {
        let mut s = String::new();
        s.push_str(&format!("Load Test Report: {}\n", self.config_name));
        s.push_str(&format!(
            "Duration: {:.2}s | Total Operations: {}\n",
            self.elapsed_seconds, self.total_operations
        ));
        s.push_str("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

        s.push_str(&format!(
            "Overall Throughput: {:.0} ops/sec\n",
            self.overall_throughput
        ));
        s.push_str(&format!(
            "Overall Error Rate: {:.2}%\n",
            self.overall_error_rate * 100.0
        ));
        s.push_str(&format!("P50 Latency: {:.2}ms\n", self.p50_latency_ms));
        s.push_str(&format!("P95 Latency: {:.2}ms\n", self.p95_latency_ms));
        s.push_str(&format!("P99 Latency: {:.2}ms\n", self.p99_latency_ms));

        s.push_str("\nPer-Operation Statistics:\n");
        s.push_str("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

        for (op_type, stats) in &self.per_operation_stats {
            s.push_str(&format!("\n{:?}:\n", op_type));
            s.push_str(&format!("  Count: {}\n", stats.count));
            s.push_str(&format!(
                "  Throughput: {:.0} ops/sec\n",
                stats.throughput(self.elapsed_seconds)
            ));
            s.push_str(&format!("  P50: {:.2}ms\n", stats.p50_latency_ms()));
            s.push_str(&format!("  P95: {:.2}ms\n", stats.p95_latency_ms()));
            s.push_str(&format!("  P99: {:.2}ms\n", stats.p99_latency_ms()));
            s.push_str(&format!("  Errors: {}\n", stats.error_count));
        }

        s
    }

    pub fn save_json(&self, path: &Path) -> Result<()> {
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(path, json)?;
        Ok(())
    }

    pub fn meets_validation_criteria(&self, config: &WorkloadConfig) -> bool {
        let mut passes = true;

        if let Some(expected_p99) = config.validation.expected_p99_latency_ms
            && self.p99_latency_ms > expected_p99
        {
            tracing::warn!(
                "P99 latency {:.2}ms exceeds target {:.2}ms",
                self.p99_latency_ms,
                expected_p99
            );
            passes = false;
        }

        if let Some(expected_throughput) = config.validation.expected_throughput_ops_sec
            && self.overall_throughput < expected_throughput
        {
            tracing::warn!(
                "Throughput {:.0} ops/sec below target {:.0} ops/sec",
                self.overall_throughput,
                expected_throughput
            );
            passes = false;
        }

        if let Some(max_error_rate) = config.validation.max_error_rate
            && self.overall_error_rate > max_error_rate
        {
            tracing::warn!(
                "Error rate {:.2}% exceeds maximum {:.2}%",
                self.overall_error_rate * 100.0,
                max_error_rate * 100.0
            );
            passes = false;
        }

        passes
    }
}
