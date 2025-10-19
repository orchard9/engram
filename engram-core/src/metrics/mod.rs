//! High-performance, low-overhead monitoring system for Engram's cognitive architecture
//!
//! Provides lock-free metrics collection with <1% overhead, NUMA-aware monitoring,
//! and deep insights into cognitive performance through atomic operations and
//! wait-free data structures.

use crossbeam_utils::CachePadded;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

pub mod cognitive;
pub mod hardware;
pub mod health;
pub mod lockfree;
pub mod streaming;

#[cfg(feature = "monitoring")]
pub mod numa_aware;

pub use cognitive::{CognitiveInsight, CognitiveMetrics, ConsolidationState};
pub use hardware::{CacheLevel, HardwareMetrics, SimdOperation};
use health::HealthProbe;
pub use health::{HealthCheck, HealthStatus, SystemHealth};
pub use lockfree::{LockFreeCounter, LockFreeGauge, LockFreeHistogram};
pub use streaming::{
    AggregatedMetrics, ExportStats, MetricAggregate, MetricUpdate, SpreadingSummary,
    StreamingAggregator, TierLatencySummary,
};

const SPREADING_ACTIVATIONS_TOTAL: &str = "engram_spreading_activations_total";
const SPREADING_LATENCY_BUDGET_VIOLATIONS_TOTAL: &str =
    "engram_spreading_latency_budget_violations_total";
const SPREADING_FALLBACK_TOTAL: &str = "engram_spreading_fallback_total";
const SPREADING_FAILURE_TOTAL: &str = "engram_spreading_failures_total";
const SPREADING_BREAKER_STATE: &str = "engram_spreading_breaker_state";
const SPREADING_BREAKER_TRANSITIONS_TOTAL: &str = "engram_spreading_breaker_transitions_total";
const SPREADING_GPU_LAUNCH_TOTAL: &str = "engram_spreading_gpu_launch_total";
const SPREADING_GPU_FALLBACK_TOTAL: &str = "engram_spreading_gpu_fallback_total";
const SPREADING_POOL_UTILIZATION: &str = "engram_spreading_pool_utilization";
const SPREADING_POOL_HIT_RATE: &str = "engram_spreading_pool_hit_rate";
const SPREADING_LATENCY_HOT: &str = "engram_spreading_latency_hot_seconds";
const SPREADING_LATENCY_WARM: &str = "engram_spreading_latency_warm_seconds";
const SPREADING_LATENCY_COLD: &str = "engram_spreading_latency_cold_seconds";

// Embedding infrastructure metrics
/// Counter tracking episodes stored with embedding provenance.
///
/// This metric enables monitoring of embedding coverage across the memory store.
/// Label `model_version` allows tracking which embedding models are in use.
///
/// Usage: `increment_counter("engram_episodes_with_embeddings_total", 1)` when
/// storing an episode with `embedding_provenance.is_some()`.
const EMBEDDING_COVERAGE_TOTAL: &str = "engram_episodes_with_embeddings_total";

/// Total number of consolidation runs completed by the background scheduler.
pub const CONSOLIDATION_RUNS_TOTAL: &str = "engram_consolidation_runs_total";
/// Count of consolidation runs that failed due to validation or runtime errors.
pub const CONSOLIDATION_FAILURES_TOTAL: &str = "engram_consolidation_failures_total";
/// Latest observed novelty delta emitted by the scheduler, used for stagnation detection.
pub const CONSOLIDATION_NOVELTY_GAUGE: &str = "engram_consolidation_novelty_gauge";
/// Variance of novelty across all patterns in the current consolidation run.
/// Measures diversity of pattern changes, high variance indicates heterogeneous updates.
pub const CONSOLIDATION_NOVELTY_VARIANCE: &str = "engram_consolidation_novelty_variance";
/// Citation churn rate: percentage of patterns with citation changes between snapshots.
/// High churn (>50%) indicates volatile consolidation, low churn (<10%) suggests stability.
pub const CONSOLIDATION_CITATION_CHURN: &str = "engram_consolidation_citation_churn";
/// Age in seconds of the cached consolidation snapshot.
pub const CONSOLIDATION_FRESHNESS_SECONDS: &str = "engram_consolidation_freshness_seconds";
/// Total citations captured in the most recent consolidation snapshot.
pub const CONSOLIDATION_CITATIONS_CURRENT: &str = "engram_consolidation_citations_current";

// Storage compaction metrics
/// Total number of storage compaction attempts initiated.
pub const COMPACTION_ATTEMPTS_TOTAL: &str = "engram_compaction_attempts_total";
/// Count of successful storage compactions where episodes were replaced with semantic patterns.
pub const COMPACTION_SUCCESS_TOTAL: &str = "engram_compaction_success_total";
/// Count of compactions that were rolled back due to verification failure.
pub const COMPACTION_ROLLBACK_TOTAL: &str = "engram_compaction_rollback_total";
/// Number of episodes removed during successful compaction operations.
pub const COMPACTION_EPISODES_REMOVED: &str = "engram_compaction_episodes_removed";
/// Total bytes of storage space reclaimed through compaction.
pub const COMPACTION_STORAGE_SAVED_BYTES: &str = "engram_compaction_storage_saved_bytes";

/// Global metrics registry for the Engram system
pub struct MetricsRegistry {
    /// Lock-free counters for operation counts
    counters: Arc<LockFreeCounters>,

    /// Lock-free histograms for latency distributions
    histograms: Arc<LockFreeHistograms>,

    /// Lock-free gauges for instantaneous measurements
    gauges: Arc<LockFreeGauges>,

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
    #[must_use]
    pub fn new() -> Self {
        let health = Arc::new(SystemHealth::new());

        let registry = Self {
            counters: Arc::new(LockFreeCounters::new()),
            histograms: Arc::new(LockFreeHistograms::new()),
            gauges: Arc::new(LockFreeGauges::new()),
            cognitive: Arc::new(CognitiveMetrics::new()),
            hardware: Arc::new(HardwareMetrics::new()),
            streaming: Arc::new(StreamingAggregator::new()),
            health: Arc::clone(&health),
            // NUMA collectors temporarily disabled
            // #[cfg(feature = "monitoring")]
            // numa_collectors: numa_aware::NumaCollectors::new().map(Arc::new),
        };

        if let Ok(probe) = crate::activation::health_checks::SpreadingHealthProbe::default_probe() {
            let hysteresis = probe.hysteresis();
            health.register_probe_with_hysteresis(probe, hysteresis);
        } else {
            tracing::warn!(
                target = "engram::health",
                "Failed to initialise spreading health probe"
            );
        }

        registry
    }

    /// Record a counter increment with <50ns overhead
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
    pub fn observe_histogram(&self, name: &'static str, value: f64) {
        self.histograms.observe(name, value);

        // Queue for streaming export
        self.streaming.queue_update(MetricUpdate::Histogram {
            name,
            value,
            timestamp: Instant::now(),
        });
    }

    /// Record an instantaneous gauge measurement.
    pub fn record_gauge(&self, name: &'static str, value: f64) {
        self.gauges.set(name, value);

        self.streaming.queue_update(MetricUpdate::Gauge {
            name,
            value,
            timestamp: Instant::now(),
        });
    }

    /// Record cognitive architecture metrics
    pub fn record_cognitive(&self, metric: &cognitive::CognitiveMetric) {
        self.cognitive.record(metric);
    }

    /// Record hardware performance metrics
    pub fn record_hardware(&self, metric: &hardware::HardwareMetric) {
        self.hardware.record(metric);
    }

    /// Get current system health status
    #[must_use]
    pub fn health_status(&self) -> HealthStatus {
        self.health.check_all()
    }

    /// Obtain a handle to the health registry for direct probe management.
    #[must_use]
    pub fn health_registry(&self) -> Arc<SystemHealth> {
        Arc::clone(&self.health)
    }

    /// Retrieve the current value of a counter metric if recorded.
    #[must_use]
    pub fn counter_value(&self, name: &'static str) -> u64 {
        self.counters.get(name)
    }

    /// Retrieve the latest gauge reading if available.
    #[must_use]
    pub fn gauge_value(&self, name: &'static str) -> Option<f64> {
        self.gauges.get(name)
    }

    /// Retrieve histogram quantiles for the provided buckets.
    #[must_use]
    pub fn histogram_quantiles(&self, name: &'static str, quantiles: &[f64]) -> Vec<f64> {
        self.histograms.quantiles(name, quantiles)
    }

    /// Drain pending streaming updates into aggregated windows.
    #[must_use]
    pub fn streaming_snapshot(&self) -> AggregatedMetrics {
        let mut snapshot = self.streaming.process_updates();
        snapshot.spreading = Self::summarise_spreading(&snapshot);
        snapshot
    }

    /// Get statistics about streaming export throughput.
    #[must_use]
    pub fn streaming_stats(&self) -> ExportStats {
        self.streaming.export_stats()
    }

    /// Clone the underlying streaming aggregator for external polling.
    #[must_use]
    pub fn streaming_aggregator(&self) -> Arc<StreamingAggregator> {
        Arc::clone(&self.streaming)
    }

    /// Emit a structured log line with the latest streaming snapshot.
    pub fn log_streaming_snapshot(&self, label: &str) {
        let snapshot = self.streaming_snapshot();
        match serde_json::to_string(&snapshot) {
            Ok(payload) => {
                tracing::info!(target = "engram::metrics::stream", label, snapshot = %payload, "streaming metrics snapshot");
            }
            Err(err) => {
                tracing::warn!(target = "engram::metrics::stream", label, error = %err, "failed to serialize metrics snapshot");
            }
        }
    }
}

impl MetricsRegistry {
    fn summarise_spreading(snapshot: &AggregatedMetrics) -> Option<SpreadingSummary> {
        let mut summary = SpreadingSummary::default();

        for (label, metric) in [
            ("hot", SPREADING_LATENCY_HOT),
            ("warm", SPREADING_LATENCY_WARM),
            ("cold", SPREADING_LATENCY_COLD),
        ] {
            if let Some(aggregate) = metric_from_latency_windows(snapshot, metric)
                && aggregate.count > 0
            {
                let p95 = percentile_interpolate(aggregate.p90, aggregate.p99, 0.5);
                summary.per_tier.insert(
                    label.to_string(),
                    TierLatencySummary {
                        samples: aggregate.count,
                        mean_seconds: aggregate.mean,
                        p50_seconds: aggregate.p50,
                        p95_seconds: p95,
                        p99_seconds: aggregate.p99,
                    },
                );
            }
        }

        summary.activations_total = metric_from_any_window(snapshot, SPREADING_ACTIVATIONS_TOTAL)
            .map(metric_as_u64)
            .filter(|value| *value > 0);
        summary.latency_budget_violations_total =
            metric_from_any_window(snapshot, SPREADING_LATENCY_BUDGET_VIOLATIONS_TOTAL)
                .map(metric_as_u64);
        summary.fallback_total =
            metric_from_any_window(snapshot, SPREADING_FALLBACK_TOTAL).map(metric_as_u64);
        summary.failure_total =
            metric_from_any_window(snapshot, SPREADING_FAILURE_TOTAL).map(metric_as_u64);
        summary.breaker_state =
            metric_from_any_window(snapshot, SPREADING_BREAKER_STATE).map(metric_as_u64);
        summary.breaker_transitions_total =
            metric_from_any_window(snapshot, SPREADING_BREAKER_TRANSITIONS_TOTAL)
                .map(metric_as_u64);
        summary.gpu_launch_total =
            metric_from_any_window(snapshot, SPREADING_GPU_LAUNCH_TOTAL).map(metric_as_u64);
        summary.gpu_fallback_total =
            metric_from_any_window(snapshot, SPREADING_GPU_FALLBACK_TOTAL).map(metric_as_u64);

        summary.pool_utilization = metric_from_any_window(snapshot, SPREADING_POOL_UTILIZATION)
            .and_then(metric_as_f64)
            .filter(|value| value.is_finite());
        summary.pool_hit_rate = metric_from_any_window(snapshot, SPREADING_POOL_HIT_RATE)
            .and_then(metric_as_f64)
            .filter(|value| value.is_finite());

        let has_any = !summary.per_tier.is_empty()
            || summary.activations_total.is_some()
            || summary.latency_budget_violations_total.is_some()
            || summary.fallback_total.is_some()
            || summary.failure_total.is_some()
            || summary.breaker_state.is_some()
            || summary.gpu_launch_total.is_some()
            || summary.gpu_fallback_total.is_some()
            || summary.pool_utilization.is_some()
            || summary.pool_hit_rate.is_some();

        if has_any { Some(summary) } else { None }
    }
}

fn metric_from_any_window<'a>(
    snapshot: &'a AggregatedMetrics,
    name: &str,
) -> Option<&'a MetricAggregate> {
    snapshot
        .one_second
        .get(name)
        .or_else(|| snapshot.ten_seconds.get(name))
        .or_else(|| snapshot.one_minute.get(name))
        .or_else(|| snapshot.five_minutes.get(name))
}

fn metric_from_latency_windows<'a>(
    snapshot: &'a AggregatedMetrics,
    name: &str,
) -> Option<&'a MetricAggregate> {
    snapshot
        .ten_seconds
        .get(name)
        .or_else(|| snapshot.one_minute.get(name))
        .or_else(|| snapshot.five_minutes.get(name))
        .or_else(|| snapshot.one_second.get(name))
}

fn metric_as_u64(metric: &MetricAggregate) -> u64 {
    if metric.max.is_finite() && metric.max >= 0.0 {
        metric.max.round() as u64
    } else {
        0
    }
}

const fn metric_as_f64(metric: &MetricAggregate) -> Option<f64> {
    if metric.max.is_finite() {
        Some(metric.max)
    } else {
        None
    }
}

fn percentile_interpolate(lower: f64, upper: f64, weight: f64) -> f64 {
    if !lower.is_finite() {
        return 0.0;
    }
    if !upper.is_finite() {
        return lower;
    }
    let clamped_weight = weight.clamp(0.0, 1.0);
    lower + (upper - lower) * clamped_weight
}

/// Lock-free counter collection with cache-line alignment
pub struct LockFreeCounters {
    /// Counters aligned to cache lines to prevent false sharing
    counters: dashmap::DashMap<&'static str, CachePadded<AtomicU64>>,
}

impl LockFreeCounters {
    /// Create new lock-free counter collection
    #[must_use]
    pub fn new() -> Self {
        Self {
            counters: dashmap::DashMap::new(),
        }
    }

    /// Increment a named counter by the given value
    pub fn increment(&self, name: &'static str, value: u64) {
        self.counters
            .entry(name)
            .or_insert_with(|| CachePadded::new(AtomicU64::new(0)))
            .fetch_add(value, Ordering::Relaxed);
    }

    /// Get the current value of a named counter
    #[must_use]
    pub fn get(&self, name: &'static str) -> u64 {
        self.counters
            .get(name)
            .map_or(0, |c| c.load(Ordering::Acquire))
    }
}

/// Lock-free histogram collection with exponential buckets
pub struct LockFreeHistograms {
    /// Histogram buckets using atomic operations
    histograms: dashmap::DashMap<&'static str, Arc<lockfree::LockFreeHistogram>>,
}

impl LockFreeHistograms {
    /// Create new lock-free histogram collection
    #[must_use]
    pub fn new() -> Self {
        Self {
            histograms: dashmap::DashMap::new(),
        }
    }

    /// Record an observation in the named histogram
    pub fn observe(&self, name: &'static str, value: f64) {
        self.histograms
            .entry(name)
            .or_insert_with(|| Arc::new(lockfree::LockFreeHistogram::new()))
            .record(value);
    }

    /// Get quantile values from the named histogram
    #[must_use]
    pub fn quantiles(&self, name: &'static str, quantiles: &[f64]) -> Vec<f64> {
        self.histograms
            .get(name)
            .map_or_else(|| vec![0.0; quantiles.len()], |h| h.quantiles(quantiles))
    }
}

/// Lock-free collection of gauges storing the latest observation per metric.
pub struct LockFreeGauges {
    gauges: dashmap::DashMap<&'static str, CachePadded<AtomicU64>>,
}

impl Default for LockFreeGauges {
    fn default() -> Self {
        Self::new()
    }
}

impl LockFreeGauges {
    /// Create a new gauge collection.
    #[must_use]
    pub fn new() -> Self {
        Self {
            gauges: dashmap::DashMap::new(),
        }
    }

    /// Set the gauge to the provided floating-point value.
    pub fn set(&self, name: &'static str, value: f64) {
        let bits = value.to_bits();
        self.gauges
            .entry(name)
            .or_insert_with(|| CachePadded::new(AtomicU64::new(0)))
            .store(bits, Ordering::Release);
    }

    /// Retrieve the gauge value if present.
    #[must_use]
    pub fn get(&self, name: &'static str) -> Option<f64> {
        self.gauges
            .get(name)
            .map(|g| f64::from_bits(g.load(Ordering::Acquire)))
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

/// Macro to record histogram metrics
#[macro_export]
macro_rules! record_histogram {
    ($registry:expr, $name:literal, $value:expr) => {
        #[cfg(feature = "monitoring")]
        {
            $registry.observe_histogram($name, $value);
        }
    };
}

/// Macro to record gauge metrics
#[macro_export]
macro_rules! record_gauge {
    ($registry:expr, $name:literal, $value:expr) => {
        #[cfg(feature = "monitoring")]
        {
            $registry.record_gauge($name, $value);
        }
    };
}

/// Macro to record cognitive metrics
#[macro_export]
macro_rules! record_cognitive {
    ($registry:expr, $metric:expr) => {
        #[cfg(feature = "monitoring")]
        {
            let metric = $metric;
            $registry.record_cognitive(&metric);
        }
    };
}

/// Macro to record hardware metrics
#[macro_export]
macro_rules! record_hardware {
    ($registry:expr, $metric:expr) => {
        #[cfg(feature = "monitoring")]
        {
            let metric = $metric;
            $registry.record_hardware(&metric);
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

/// Record a gauge metric value
pub fn record_gauge(name: &'static str, value: f64) {
    if let Some(metrics) = metrics() {
        metrics.record_gauge(name, value);
    }
}

/// Record a cognitive metric
pub fn record_cognitive(metric: &cognitive::CognitiveMetric) {
    if let Some(metrics) = metrics() {
        metrics.record_cognitive(metric);
    }
}

/// Record when an episode is stored with embedding provenance.
///
/// This increments the `engram_episodes_with_embeddings_total` counter to track
/// embedding coverage across the memory store.
///
/// # Example
/// ```ignore
/// use engram_core::metrics;
///
/// // When storing an episode with embedding provenance
/// if episode.embedding_provenance.is_some() {
///     metrics::record_embedding_coverage();
/// }
/// ```
pub fn record_embedding_coverage() {
    increment_counter(EMBEDDING_COVERAGE_TOTAL, 1);
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

        let quantiles = registry
            .histograms
            .quantiles("test_histogram", &[0.5, 0.9, 0.99]);
        assert_eq!(quantiles.len(), 3);
    }

    #[test]
    fn test_global_metrics_initialization() {
        let metrics1 = init();
        let metrics2 = init();
        assert!(Arc::ptr_eq(&metrics1, &metrics2));
    }
}
