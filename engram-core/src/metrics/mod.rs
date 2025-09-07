//! High-performance, low-overhead monitoring system for Engram's cognitive architecture
//!
//! Provides lock-free metrics collection with <1% overhead, NUMA-aware monitoring,
//! and deep insights into cognitive performance through atomic operations and
//! wait-free data structures.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;
use crossbeam_utils::CachePadded;

pub mod lockfree;
pub mod cognitive;
pub mod hardware;
pub mod streaming;
pub mod health;

#[cfg(feature = "monitoring")]
pub mod numa_aware;

#[cfg(feature = "monitoring")]
pub mod prometheus;

pub use lockfree::{LockFreeCounter, LockFreeHistogram, LockFreeGauge};
pub use cognitive::{CognitiveMetrics, ConsolidationState, CognitiveInsight};
pub use hardware::{HardwareMetrics, CacheLevel, SimdOperation};
pub use streaming::{StreamingAggregator, MetricUpdate};
pub use health::{SystemHealth, HealthCheck, HealthStatus};

/// Global metrics registry for the Engram system
pub struct MetricsRegistry {
    /// Lock-free counters for operation counts
    counters: Arc<LockFreeCounters>,
    
    /// Lock-free histograms for latency distributions
    histograms: Arc<LockFreeHistograms>,
    
    /// Cognitive architecture specific metrics
    cognitive: Arc<CognitiveMetrics>,
    
    /// Hardware performance metrics
    hardware: Arc<HardwareMetrics>,
    
    /// Streaming aggregation for real-time export
    streaming: Arc<StreamingAggregator>,
    
    /// System health monitoring
    health: Arc<SystemHealth>,
    
    // NUMA collectors temporarily disabled due to hwloc Send/Sync issues
    // TODO: Wrap hwloc types properly for thread safety
    // #[cfg(feature = "monitoring")]
    // numa_collectors: Option<Arc<numa_aware::NumaCollectors>>,
}

impl MetricsRegistry {
    /// Create a new metrics registry with default configuration
    pub fn new() -> Self {
        Self {
            counters: Arc::new(LockFreeCounters::new()),
            histograms: Arc::new(LockFreeHistograms::new()),
            cognitive: Arc::new(CognitiveMetrics::new()),
            hardware: Arc::new(HardwareMetrics::new()),
            streaming: Arc::new(StreamingAggregator::new()),
            health: Arc::new(SystemHealth::new()),
            // NUMA collectors temporarily disabled
            // #[cfg(feature = "monitoring")]
            // numa_collectors: numa_aware::NumaCollectors::new().map(Arc::new),
        }
    }
    
    /// Record a counter increment with <50ns overhead
    #[inline(always)]
    pub fn increment_counter(&self, name: &'static str, value: u64) {
        self.counters.increment(name, value);
        
        // Queue for streaming export
        self.streaming.queue_update(MetricUpdate::Counter {
            name,
            value,
            timestamp: Instant::now(),
        });
    }
    
    /// Record a histogram observation with <100ns overhead
    #[inline(always)]
    pub fn observe_histogram(&self, name: &'static str, value: f64) {
        self.histograms.observe(name, value);
        
        // Queue for streaming export
        self.streaming.queue_update(MetricUpdate::Histogram {
            name,
            value,
            timestamp: Instant::now(),
        });
    }
    
    /// Record cognitive architecture metrics
    #[inline(always)]
    pub fn record_cognitive(&self, metric: cognitive::CognitiveMetric) {
        self.cognitive.record(metric);
    }
    
    /// Record hardware performance metrics
    #[inline(always)]
    pub fn record_hardware(&self, metric: hardware::HardwareMetric) {
        self.hardware.record(metric);
    }
    
    /// Get current system health status
    pub fn health_status(&self) -> HealthStatus {
        self.health.check_all()
    }
    
    /// Export metrics in Prometheus format
    #[cfg(feature = "monitoring")]
    pub fn export_prometheus(&self) -> String {
        prometheus::export_all(self)
    }
}

/// Lock-free counter collection with cache-line alignment
pub struct LockFreeCounters {
    /// Counters aligned to cache lines to prevent false sharing
    counters: dashmap::DashMap<&'static str, CachePadded<AtomicU64>>,
}

impl LockFreeCounters {
    pub fn new() -> Self {
        Self {
            counters: dashmap::DashMap::new(),
        }
    }
    
    #[inline(always)]
    pub fn increment(&self, name: &'static str, value: u64) {
        self.counters
            .entry(name)
            .or_insert_with(|| CachePadded::new(AtomicU64::new(0)))
            .fetch_add(value, Ordering::Relaxed);
    }
    
    pub fn get(&self, name: &'static str) -> u64 {
        self.counters
            .get(name)
            .map(|c| c.load(Ordering::Acquire))
            .unwrap_or(0)
    }
}

/// Lock-free histogram collection with exponential buckets
pub struct LockFreeHistograms {
    /// Histogram buckets using atomic operations
    histograms: dashmap::DashMap<&'static str, Arc<lockfree::LockFreeHistogram>>,
}

impl LockFreeHistograms {
    pub fn new() -> Self {
        Self {
            histograms: dashmap::DashMap::new(),
        }
    }
    
    #[inline(always)]
    pub fn observe(&self, name: &'static str, value: f64) {
        self.histograms
            .entry(name)
            .or_insert_with(|| Arc::new(lockfree::LockFreeHistogram::new()))
            .record(value);
    }
    
    pub fn quantiles(&self, name: &'static str, quantiles: &[f64]) -> Vec<f64> {
        self.histograms
            .get(name)
            .map(|h| h.quantiles(quantiles))
            .unwrap_or_else(|| vec![0.0; quantiles.len()])
    }
}

/// Metric recording macros for zero-overhead when disabled
#[macro_export]
macro_rules! record_counter {
    ($registry:expr, $name:literal, $value:expr) => {
        #[cfg(feature = "monitoring")]
        {
            $registry.increment_counter($name, $value);
        }
    };
}

#[macro_export]
macro_rules! record_histogram {
    ($registry:expr, $name:literal, $value:expr) => {
        #[cfg(feature = "monitoring")]
        {
            $registry.observe_histogram($name, $value);
        }
    };
}

#[macro_export]
macro_rules! record_cognitive {
    ($registry:expr, $metric:expr) => {
        #[cfg(feature = "monitoring")]
        {
            $registry.record_cognitive($metric);
        }
    };
}

#[macro_export]
macro_rules! record_hardware {
    ($registry:expr, $metric:expr) => {
        #[cfg(feature = "monitoring")]
        {
            $registry.record_hardware($metric);
        }
    };
}

/// Global metrics instance (lazy-initialized)
static METRICS: std::sync::OnceLock<Arc<MetricsRegistry>> = std::sync::OnceLock::new();

/// Initialize the global metrics registry
pub fn init() -> Arc<MetricsRegistry> {
    Arc::clone(METRICS.get_or_init(|| Arc::new(MetricsRegistry::new())))
}

/// Get the global metrics registry
pub fn metrics() -> Option<Arc<MetricsRegistry>> {
    METRICS.get().cloned()
}

/// Increment a counter metric
pub fn increment_counter(name: &'static str, value: u64) {
    if let Some(metrics) = metrics() {
        metrics.increment_counter(name, value);
    }
}

/// Observe a value in a histogram
pub fn observe_histogram(name: &'static str, value: f64) {
    if let Some(metrics) = metrics() {
        metrics.observe_histogram(name, value);
    }
}

/// Record a cognitive metric
pub fn record_cognitive(metric: cognitive::CognitiveMetric) {
    if let Some(metrics) = metrics() {
        metrics.record_cognitive(metric);
    }
}

impl Default for MetricsRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for LockFreeCounters {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for LockFreeHistograms {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_metrics_registry_creation() {
        let registry = MetricsRegistry::new();
        assert_eq!(registry.counters.get("test"), 0);
    }
    
    #[test]
    fn test_counter_increment() {
        let registry = MetricsRegistry::new();
        registry.increment_counter("test_counter", 1);
        registry.increment_counter("test_counter", 2);
        assert_eq!(registry.counters.get("test_counter"), 3);
    }
    
    #[test]
    fn test_histogram_observation() {
        let registry = MetricsRegistry::new();
        registry.observe_histogram("test_histogram", 10.0);
        registry.observe_histogram("test_histogram", 20.0);
        registry.observe_histogram("test_histogram", 30.0);
        
        let quantiles = registry.histograms.quantiles("test_histogram", &[0.5, 0.9, 0.99]);
        assert_eq!(quantiles.len(), 3);
    }
    
    #[test]
    fn test_global_metrics_initialization() {
        let metrics1 = init();
        let metrics2 = init();
        assert!(Arc::ptr_eq(&metrics1, &metrics2));
    }
}