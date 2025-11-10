//! Real-time metrics collection and aggregation

use crate::workload_generator::OperationType;
use hdrhistogram::Histogram;
use serde::Serialize;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Duration;

#[derive(Clone)]
pub struct MetricsCollector {
    inner: Arc<Mutex<MetricsInner>>,
}

struct MetricsInner {
    /// Per-operation-type histograms
    latency_histograms: HashMap<OperationType, Histogram<u64>>,
    /// Error counts
    error_counts: HashMap<OperationType, u64>,
    /// Success counts
    success_counts: HashMap<OperationType, u64>,
    /// Start time
    start_time: std::time::Instant,
}

impl MetricsCollector {
    pub fn new() -> Self {
        let mut latency_histograms = HashMap::new();
        latency_histograms.insert(
            OperationType::Store,
            Histogram::<u64>::new_with_bounds(1, 60_000, 3).expect("Invalid histogram bounds"),
        );
        latency_histograms.insert(
            OperationType::Recall,
            Histogram::<u64>::new_with_bounds(1, 60_000, 3).expect("Invalid histogram bounds"),
        );
        latency_histograms.insert(
            OperationType::Search,
            Histogram::<u64>::new_with_bounds(1, 60_000, 3).expect("Invalid histogram bounds"),
        );
        latency_histograms.insert(
            OperationType::PatternCompletion,
            Histogram::<u64>::new_with_bounds(1, 60_000, 3).expect("Invalid histogram bounds"),
        );

        Self {
            inner: Arc::new(Mutex::new(MetricsInner {
                latency_histograms,
                error_counts: HashMap::new(),
                success_counts: HashMap::new(),
                start_time: std::time::Instant::now(),
            })),
        }
    }

    pub fn record_operation(&mut self, op_type: OperationType, latency: Duration, success: bool) {
        let mut inner = self.inner.lock().expect("Mutex poisoned");

        // Record latency in microseconds
        let latency_us = latency.as_micros().min(u64::MAX as u128) as u64;
        if let Some(histogram) = inner.latency_histograms.get_mut(&op_type) {
            let _ = histogram.record(latency_us);
        }

        // Record success/error
        if success {
            *inner.success_counts.entry(op_type).or_insert(0) += 1;
        } else {
            *inner.error_counts.entry(op_type).or_insert(0) += 1;
        }
    }

    pub fn current_throughput(&self) -> f64 {
        let inner = self.inner.lock().expect("Mutex poisoned");
        let elapsed = inner.start_time.elapsed().as_secs_f64();
        if elapsed > 0.0 {
            let total_ops: u64 = inner.success_counts.values().sum::<u64>()
                + inner.error_counts.values().sum::<u64>();
            total_ops as f64 / elapsed
        } else {
            0.0
        }
    }

    pub fn p99_latency_ms(&self) -> f64 {
        let inner = self.inner.lock().expect("Mutex poisoned");

        // Aggregate across all operation types
        let mut total_count = 0u64;
        let mut max_p99 = 0.0f64;

        for histogram in inner.latency_histograms.values() {
            if !histogram.is_empty() {
                total_count += histogram.len();
                let p99_us = histogram.value_at_quantile(0.99);
                let p99_ms = p99_us as f64 / 1000.0;
                max_p99 = max_p99.max(p99_ms);
            }
        }

        if total_count > 0 { max_p99 } else { 0.0 }
    }

    pub fn error_rate(&self) -> f64 {
        let inner = self.inner.lock().expect("Mutex poisoned");
        let total_errors: u64 = inner.error_counts.values().sum();
        let total_success: u64 = inner.success_counts.values().sum();
        let total = total_errors + total_success;

        if total > 0 {
            total_errors as f64 / total as f64
        } else {
            0.0
        }
    }

    /// Get all latency samples for hypothesis testing
    pub fn all_latency_samples(&self) -> Vec<f64> {
        let inner = self.inner.lock().expect("Mutex poisoned");
        let mut samples = Vec::new();

        for histogram in inner.latency_histograms.values() {
            for value in histogram.iter_recorded() {
                // Convert from microseconds to milliseconds
                samples.push(value.value_iterated_to() as f64 / 1000.0);
            }
        }

        samples
    }

    /// Get throughput for a specific time window (used by hypothesis tests)
    pub fn window_throughput_ops_sec(&self, _window_size_secs: u64) -> f64 {
        // Simplified implementation - returns overall throughput
        // In a full implementation, this would track per-window metrics
        self.current_throughput()
    }

    /// Get detailed statistics for report generation
    pub fn get_statistics(&self) -> MetricsStatistics {
        let inner = self.inner.lock().expect("Mutex poisoned");

        let mut per_operation = HashMap::new();

        for op_type in [
            OperationType::Store,
            OperationType::Recall,
            OperationType::Search,
            OperationType::PatternCompletion,
        ] {
            if let Some(histogram) = inner.latency_histograms.get(&op_type) {
                let success_count = inner.success_counts.get(&op_type).copied().unwrap_or(0);
                let error_count = inner.error_counts.get(&op_type).copied().unwrap_or(0);

                if success_count + error_count > 0 {
                    per_operation.insert(
                        op_type,
                        OperationStatistics {
                            count: success_count + error_count,
                            success_count,
                            error_count,
                            p50_latency_us: histogram.value_at_quantile(0.50),
                            p95_latency_us: histogram.value_at_quantile(0.95),
                            p99_latency_us: histogram.value_at_quantile(0.99),
                            p999_latency_us: histogram.value_at_quantile(0.999),
                            mean_latency_us: histogram.mean(),
                            max_latency_us: histogram.max(),
                        },
                    );
                }
            }
        }

        let total_operations: u64 = per_operation.values().map(|s| s.count).sum();
        let total_errors: u64 = per_operation.values().map(|s| s.error_count).sum();

        MetricsStatistics {
            elapsed_seconds: inner.start_time.elapsed().as_secs_f64(),
            total_operations,
            total_errors,
            per_operation,
        }
    }
}

impl Default for MetricsCollector {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone)]
pub struct MetricsStatistics {
    pub elapsed_seconds: f64,
    pub total_operations: u64,
    pub total_errors: u64,
    pub per_operation: HashMap<OperationType, OperationStatistics>,
}

#[derive(Debug, Clone, Serialize, serde::Deserialize)]
pub struct OperationStatistics {
    pub count: u64,
    pub success_count: u64,
    pub error_count: u64,
    pub p50_latency_us: u64,
    pub p95_latency_us: u64,
    pub p99_latency_us: u64,
    pub p999_latency_us: u64,
    pub mean_latency_us: f64,
    pub max_latency_us: u64,
}

impl OperationStatistics {
    pub fn p50_latency_ms(&self) -> f64 {
        self.p50_latency_us as f64 / 1000.0
    }

    pub fn p95_latency_ms(&self) -> f64 {
        self.p95_latency_us as f64 / 1000.0
    }

    pub fn p99_latency_ms(&self) -> f64 {
        self.p99_latency_us as f64 / 1000.0
    }

    #[allow(dead_code)] // Useful for detailed analysis
    pub fn p999_latency_ms(&self) -> f64 {
        self.p999_latency_us as f64 / 1000.0
    }

    #[allow(dead_code)] // Useful for detailed analysis
    pub fn mean_latency_ms(&self) -> f64 {
        self.mean_latency_us / 1000.0
    }

    pub fn throughput(&self, elapsed_seconds: f64) -> f64 {
        if elapsed_seconds > 0.0 {
            self.count as f64 / elapsed_seconds
        } else {
            0.0
        }
    }
}
