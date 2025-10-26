//! Performance tracking for CPU/GPU hybrid execution
//!
//! This module tracks historical performance of CPU and GPU operations to enable
//! adaptive dispatch decisions. It maintains ring buffers of recent latencies and
//! computes moving averages to smooth out measurement noise.
//!
//! # Architecture
//!
//! - Ring buffers for CPU and GPU latencies (bounded memory usage)
//! - Success rate tracking for GPU reliability
//! - Per-operation metrics (cosine similarity, activation spreading, HNSW)
//! - Thread-safe concurrent updates
//!
//! # Usage
//!
//! ```rust,ignore
//! use engram_core::compute::cuda::performance_tracker::PerformanceTracker;
//!
//! let tracker = PerformanceTracker::new(100); // 100 sample window
//!
//! // Record measurements
//! tracker.record_cpu_latency(Operation::CosineSimilarity, Duration::from_micros(100));
//! tracker.record_gpu_success(Operation::CosineSimilarity, Duration::from_micros(20));
//!
//! // Query metrics
//! let speedup = tracker.gpu_speedup(Operation::CosineSimilarity); // ~5.0x
//! let success_rate = tracker.gpu_success_rate(Operation::CosineSimilarity); // ~1.0
//! ```

// Allow expect on lock() - if a Mutex is poisoned, we want to panic
// This is intentional behavior for correctness
#![allow(clippy::expect_used)]
// We intentionally hold locks for the full scope to ensure atomicity
#![allow(clippy::significant_drop_tightening)]

use std::collections::{HashMap, VecDeque};
use std::sync::Mutex;
use std::time::Duration;

/// Operation types for performance tracking
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Operation {
    /// Cosine similarity batch operations
    CosineSimilarity,
    /// Activation spreading graph traversal
    ActivationSpreading,
    /// HNSW neighbor search and scoring
    HnswSearch,
}

impl std::fmt::Display for Operation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::CosineSimilarity => write!(f, "CosineSimilarity"),
            Self::ActivationSpreading => write!(f, "ActivationSpreading"),
            Self::HnswSearch => write!(f, "HnswSearch"),
        }
    }
}

/// Performance tracker for adaptive CPU/GPU dispatch
///
/// Maintains ring buffers of recent latencies for CPU and GPU operations.
/// Computes moving averages and success rates to inform dispatch decisions.
///
/// Thread-safe: Uses Mutex for interior mutability.
pub struct PerformanceTracker {
    /// Ring buffer of recent CPU latencies per operation
    cpu_latencies: Mutex<HashMap<Operation, VecDeque<Duration>>>,

    /// Ring buffer of recent GPU latencies per operation
    gpu_latencies: Mutex<HashMap<Operation, VecDeque<Duration>>>,

    /// Count of GPU failures per operation
    gpu_failures: Mutex<HashMap<Operation, usize>>,

    /// Count of successful GPU executions per operation
    gpu_successes: Mutex<HashMap<Operation, usize>>,

    /// Count of OOM events per operation (subset of failures)
    oom_events: Mutex<HashMap<Operation, usize>>,

    /// Maximum window size for moving averages
    window_size: usize,
}

impl PerformanceTracker {
    /// Create new performance tracker
    ///
    /// # Arguments
    ///
    /// * `window_size` - Maximum number of samples to keep in ring buffers
    #[must_use]
    pub fn new(window_size: usize) -> Self {
        Self {
            cpu_latencies: Mutex::new(HashMap::new()),
            gpu_latencies: Mutex::new(HashMap::new()),
            gpu_failures: Mutex::new(HashMap::new()),
            gpu_successes: Mutex::new(HashMap::new()),
            oom_events: Mutex::new(HashMap::new()),
            window_size,
        }
    }

    /// Record CPU latency for an operation
    ///
    /// Adds latency to ring buffer, evicting oldest sample if at capacity.
    pub fn record_cpu_latency(&self, operation: Operation, latency: Duration) {
        let mut latencies = self
            .cpu_latencies
            .lock()
            .expect("CPU latencies lock poisoned");
        let queue = latencies.entry(operation).or_default();

        if queue.len() >= self.window_size {
            queue.pop_front();
        }
        queue.push_back(latency);
    }

    /// Record successful GPU execution
    ///
    /// Increments success counter and records latency.
    pub fn record_gpu_success(&self, operation: Operation, latency: Duration) {
        // Record latency
        {
            let mut latencies = self
                .gpu_latencies
                .lock()
                .expect("GPU latencies lock poisoned");
            let queue = latencies.entry(operation).or_default();

            if queue.len() >= self.window_size {
                queue.pop_front();
            }
            queue.push_back(latency);
        }

        // Increment success counter
        let mut successes = self
            .gpu_successes
            .lock()
            .expect("GPU successes lock poisoned");
        *successes.entry(operation).or_default() += 1;
    }

    /// Record GPU execution failure
    ///
    /// Increments failure counter. Does not record latency.
    pub fn record_gpu_failure(&self, operation: Operation) {
        let mut failures = self
            .gpu_failures
            .lock()
            .expect("GPU failures lock poisoned");
        *failures.entry(operation).or_default() += 1;
    }

    /// Record OOM (out-of-memory) event
    ///
    /// Tracks OOM events separately from general failures for telemetry.
    /// This helps distinguish memory pressure issues from other GPU errors.
    pub fn record_oom_event(&self, operation: Operation) {
        let mut oom_events = self.oom_events.lock().expect("OOM events lock poisoned");
        *oom_events.entry(operation).or_default() += 1;
    }

    /// Get total OOM event count for an operation
    ///
    /// Returns the number of times OOM was encountered for this operation.
    #[must_use]
    pub fn oom_count(&self, operation: Operation) -> usize {
        let oom_events = self.oom_events.lock().expect("OOM events lock poisoned");
        *oom_events.get(&operation).unwrap_or(&0)
    }

    /// Record GPU unavailable event
    ///
    /// This is tracked separately from failures - indicates GPU is in error state.
    #[allow(clippy::unused_self, clippy::missing_const_for_fn)]
    pub fn record_gpu_unavailable(&self) {
        // For now, this is informational only
        // Could be extended to track unavailability patterns
    }

    /// Record CPU fallback latency
    ///
    /// When GPU fails and we fall back to CPU, record the CPU latency.
    /// This is tracked as CPU latency but not counted as a "normal" CPU execution.
    #[allow(clippy::unused_self, clippy::missing_const_for_fn)]
    pub fn record_cpu_fallback(&self, _latency: Duration) {
        // For now, this is informational only
        // Could be used to track overhead of failed GPU attempts
    }

    /// Compute GPU speedup vs CPU
    ///
    /// Returns ratio of average CPU latency to average GPU latency.
    /// Values > 1.0 indicate GPU is faster.
    ///
    /// Returns 0.0 if insufficient data to compute speedup.
    #[must_use]
    pub fn gpu_speedup(&self, operation: Operation) -> f64 {
        let cpu_avg = self.average_cpu_latency(operation);
        let gpu_avg = self.average_gpu_latency(operation);

        if gpu_avg.is_zero() || cpu_avg.is_zero() {
            return 0.0; // Insufficient data
        }

        cpu_avg.as_secs_f64() / gpu_avg.as_secs_f64()
    }

    /// Compute GPU success rate
    ///
    /// Returns fraction of GPU attempts that succeeded (0.0 to 1.0).
    /// Returns 1.0 if no GPU executions have been attempted yet.
    #[must_use]
    pub fn gpu_success_rate(&self, operation: Operation) -> f64 {
        let failures = self
            .gpu_failures
            .lock()
            .expect("GPU failures lock poisoned");
        let successes = self
            .gpu_successes
            .lock()
            .expect("GPU successes lock poisoned");

        let fail_count = failures.get(&operation).copied().unwrap_or(0);
        let success_count = successes.get(&operation).copied().unwrap_or(0);
        let total = fail_count + success_count;

        if total == 0 {
            return 1.0; // No data, assume success
        }

        success_count as f64 / total as f64
    }

    /// Get average CPU latency for an operation
    ///
    /// Returns Duration::ZERO if no measurements available.
    #[must_use]
    pub fn average_cpu_latency(&self, operation: Operation) -> Duration {
        let latencies = self
            .cpu_latencies
            .lock()
            .expect("CPU latencies lock poisoned");
        let queue = match latencies.get(&operation) {
            Some(q) if !q.is_empty() => q,
            _ => return Duration::ZERO,
        };

        let sum: Duration = queue.iter().sum();
        sum / queue.len() as u32
    }

    /// Get average GPU latency for an operation
    ///
    /// Returns Duration::ZERO if no measurements available.
    #[must_use]
    pub fn average_gpu_latency(&self, operation: Operation) -> Duration {
        let latencies = self
            .gpu_latencies
            .lock()
            .expect("GPU latencies lock poisoned");
        let queue = match latencies.get(&operation) {
            Some(q) if !q.is_empty() => q,
            _ => return Duration::ZERO,
        };

        let sum: Duration = queue.iter().sum();
        sum / queue.len() as u32
    }

    /// Get number of GPU failures across all operations
    ///
    /// Used for testing and monitoring.
    #[must_use]
    pub fn total_gpu_failures(&self) -> usize {
        let failures = self
            .gpu_failures
            .lock()
            .expect("GPU failures lock poisoned");
        failures.values().sum()
    }

    /// Get telemetry snapshot for monitoring
    ///
    /// Returns human-readable statistics for all tracked operations.
    #[must_use]
    pub fn telemetry(&self) -> String {
        use std::fmt::Write;

        let operations = [
            Operation::CosineSimilarity,
            Operation::ActivationSpreading,
            Operation::HnswSearch,
        ];

        let mut output = String::from("Performance Tracker Telemetry:\n");

        for op in &operations {
            let cpu_avg = self.average_cpu_latency(*op);
            let gpu_avg = self.average_gpu_latency(*op);
            let speedup = self.gpu_speedup(*op);
            let success_rate = self.gpu_success_rate(*op);
            let oom_events = self.oom_count(*op);

            writeln!(
                &mut output,
                "  {}: CPU={:.2}us, GPU={:.2}us, Speedup={:.2}x, Success={:.1}%, OOM={}",
                op,
                cpu_avg.as_secs_f64() * 1_000_000.0,
                gpu_avg.as_secs_f64() * 1_000_000.0,
                speedup,
                success_rate * 100.0,
                oom_events,
            )
            .expect("String write should not fail");
        }

        output
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tracker_creation() {
        let tracker = PerformanceTracker::new(100);
        assert_eq!(tracker.window_size, 100);

        // No data yet, should return sensible defaults
        let speedup = tracker.gpu_speedup(Operation::CosineSimilarity);
        assert!(
            (speedup - 0.0).abs() < f64::EPSILON,
            "No data should give 0.0 speedup"
        );
        let success_rate = tracker.gpu_success_rate(Operation::CosineSimilarity);
        assert!(
            (success_rate - 1.0).abs() < f64::EPSILON,
            "No data should assume success"
        );
    }

    #[test]
    fn test_cpu_latency_tracking() {
        let tracker = PerformanceTracker::new(100);

        tracker.record_cpu_latency(Operation::CosineSimilarity, Duration::from_micros(100));
        tracker.record_cpu_latency(Operation::CosineSimilarity, Duration::from_micros(200));

        let avg = tracker.average_cpu_latency(Operation::CosineSimilarity);
        assert!(
            (avg.as_micros() as i64 - 150).abs() < 5,
            "Average of 100 and 200 should be ~150, got {}",
            avg.as_micros()
        );
    }

    #[test]
    fn test_gpu_latency_tracking() {
        let tracker = PerformanceTracker::new(100);

        tracker.record_gpu_success(Operation::CosineSimilarity, Duration::from_micros(20));
        tracker.record_gpu_success(Operation::CosineSimilarity, Duration::from_micros(30));

        let avg = tracker.average_gpu_latency(Operation::CosineSimilarity);
        assert!(
            (avg.as_micros() as i64 - 25).abs() < 5,
            "Average of 20 and 30 should be ~25, got {}",
            avg.as_micros()
        );
    }

    #[test]
    fn test_speedup_calculation() {
        let tracker = PerformanceTracker::new(100);

        // CPU: 100us average
        tracker.record_cpu_latency(Operation::CosineSimilarity, Duration::from_micros(100));

        // GPU: 20us average
        tracker.record_gpu_success(Operation::CosineSimilarity, Duration::from_micros(20));

        let speedup = tracker.gpu_speedup(Operation::CosineSimilarity);
        assert!(
            (speedup - 5.0).abs() < 0.1,
            "100us / 20us should be ~5.0x speedup, got {speedup}"
        );
    }

    #[test]
    fn test_success_rate_tracking() {
        let tracker = PerformanceTracker::new(100);

        // 3 successes
        tracker.record_gpu_success(Operation::CosineSimilarity, Duration::from_micros(20));
        tracker.record_gpu_success(Operation::CosineSimilarity, Duration::from_micros(20));
        tracker.record_gpu_success(Operation::CosineSimilarity, Duration::from_micros(20));

        // 1 failure
        tracker.record_gpu_failure(Operation::CosineSimilarity);

        let success_rate = tracker.gpu_success_rate(Operation::CosineSimilarity);
        assert!(
            (success_rate - 0.75).abs() < 0.01,
            "3/4 success should be 0.75, got {success_rate}"
        );
    }

    #[test]
    fn test_ring_buffer_eviction() {
        let tracker = PerformanceTracker::new(3); // Small window for testing

        // Add 5 samples (should evict first 2)
        for i in 1..=5 {
            tracker.record_cpu_latency(Operation::CosineSimilarity, Duration::from_micros(i * 10));
        }

        // Should have kept only last 3: 30, 40, 50
        let avg = tracker.average_cpu_latency(Operation::CosineSimilarity);
        assert!(
            (avg.as_micros() as i64 - 40).abs() < 5,
            "Average of last 3 samples (30, 40, 50) should be ~40, got {}",
            avg.as_micros()
        );
    }

    #[test]
    fn test_operation_isolation() {
        let tracker = PerformanceTracker::new(100);

        // Different operations should have independent metrics
        tracker.record_cpu_latency(Operation::CosineSimilarity, Duration::from_micros(100));
        tracker.record_cpu_latency(Operation::ActivationSpreading, Duration::from_micros(200));

        let avg_cosine = tracker.average_cpu_latency(Operation::CosineSimilarity);
        let avg_spreading = tracker.average_cpu_latency(Operation::ActivationSpreading);

        assert_eq!(avg_cosine.as_micros(), 100);
        assert_eq!(avg_spreading.as_micros(), 200);
    }

    #[test]
    fn test_telemetry_output() {
        let tracker = PerformanceTracker::new(100);

        tracker.record_cpu_latency(Operation::CosineSimilarity, Duration::from_micros(100));
        tracker.record_gpu_success(Operation::CosineSimilarity, Duration::from_micros(20));

        let telemetry = tracker.telemetry();
        assert!(telemetry.contains("CosineSimilarity"));
        assert!(telemetry.contains("Speedup"));
        assert!(telemetry.contains("Success"));
    }

    #[test]
    fn test_zero_latency_handling() {
        let tracker = PerformanceTracker::new(100);

        // No GPU data, speedup should be 0.0 (not panic)
        let speedup = tracker.gpu_speedup(Operation::CosineSimilarity);
        assert!((speedup - 0.0).abs() < f64::EPSILON);

        // Add CPU data but no GPU data
        tracker.record_cpu_latency(Operation::CosineSimilarity, Duration::from_micros(100));
        let speedup = tracker.gpu_speedup(Operation::CosineSimilarity);
        assert!((speedup - 0.0).abs() < f64::EPSILON);
    }
}
