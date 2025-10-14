//! Streaming metrics aggregation for real-time export

use crossbeam_queue::ArrayQueue;
use crossbeam_utils::CachePadded;
use parking_lot::RwLock;
use serde::Serialize;
use std::collections::{BTreeMap, VecDeque};
use std::convert::TryFrom;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant};

/// Streaming aggregator for real-time metrics export
pub struct StreamingAggregator {
    /// Lock-free queue for metric updates
    update_queue: ArrayQueue<MetricUpdate>,

    /// Aggregation windows
    windows: RwLock<AggregationWindows>,

    /// Export state
    export_enabled: CachePadded<AtomicBool>,
    exported_count: CachePadded<AtomicU64>,
    dropped_count: CachePadded<AtomicU64>,
}

impl StreamingAggregator {
    /// Create a streaming aggregation pipeline with the default buffer size.
    #[must_use]
    pub fn new() -> Self {
        Self::with_capacity(65536) // 64K updates buffer
    }

    /// Create a streaming aggregator sized for the expected update volume.
    #[must_use]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            update_queue: ArrayQueue::new(capacity),
            windows: RwLock::new(AggregationWindows::new()),
            export_enabled: CachePadded::new(AtomicBool::new(true)),
            exported_count: CachePadded::new(AtomicU64::new(0)),
            dropped_count: CachePadded::new(AtomicU64::new(0)),
        }
    }

    /// Queue a metric update for aggregation
    pub fn queue_update(&self, update: MetricUpdate) {
        if !self.export_enabled.load(Ordering::Acquire) {
            return;
        }

        if self.update_queue.push(update.clone()).is_err() {
            // Queue full, drop oldest and retry
            self.dropped_count.fetch_add(1, Ordering::Relaxed);
            let _ = self.update_queue.pop();
            let _ = self.update_queue.push(update);
        }
    }

    /// Process pending updates and aggregate into windows
    pub fn process_updates(&self) -> AggregatedMetrics {
        let mut updates = Vec::with_capacity(1024);

        // Drain queue into local buffer
        while let Some(update) = self.update_queue.pop() {
            updates.push(update);
            if updates.len() >= 1024 {
                break; // Process in batches
            }
        }

        let mut windows = self.windows.write();
        if updates.is_empty() {
            return windows.compute_aggregates();
        }

        // Aggregate updates
        for update in &updates {
            windows.add_update(update);
            self.exported_count.fetch_add(1, Ordering::Relaxed);
        }

        windows.compute_aggregates()
    }

    /// Enable or disable export
    pub fn set_export_enabled(&self, enabled: bool) {
        self.export_enabled.store(enabled, Ordering::Release);
    }

    /// Get export statistics
    pub fn export_stats(&self) -> ExportStats {
        ExportStats {
            exported: self.exported_count.load(Ordering::Acquire),
            dropped: self.dropped_count.load(Ordering::Acquire),
            queue_depth: self.update_queue.len(),
        }
    }
}

/// Metric update types
#[derive(Debug, Clone)]
pub enum MetricUpdate {
    /// Discrete counter increment emitted by producers.
    Counter {
        /// Counter identifier used in registries and dashboards.
        name: &'static str,
        /// Increment amount contributed by this update.
        value: u64,
        /// Capture time so aggregators can maintain order.
        timestamp: Instant,
    },
    /// Scalar gauge reading such as temperature or queue depth.
    Gauge {
        /// Gauge identifier.
        name: &'static str,
        /// Recorded value for the gauge.
        value: f64,
        /// When the gauge value was captured.
        timestamp: Instant,
    },
    /// Single observation fed into a histogram distribution.
    Histogram {
        /// Histogram identifier.
        name: &'static str,
        /// Observation magnitude.
        value: f64,
        /// Capture time of the sample.
        timestamp: Instant,
    },
    /// Observation destined for percentile/summary calculations.
    Summary {
        /// Summary series identifier.
        name: &'static str,
        /// Sample value recorded for the summary.
        value: f64,
        /// Capture time associated with the sample.
        timestamp: Instant,
    },
}

/// Aggregated statistics keyed by metric name for a single window.
pub type WindowSnapshot = BTreeMap<&'static str, MetricAggregate>;

/// Time-based aggregation windows
struct AggregationWindows {
    /// 1-second window for real-time monitoring
    one_second: TimeWindow,

    /// 10-second window for short-term trends
    ten_seconds: TimeWindow,

    /// 1-minute window for medium-term analysis
    one_minute: TimeWindow,

    /// 5-minute window for stability monitoring
    five_minutes: TimeWindow,
}

impl AggregationWindows {
    const fn new() -> Self {
        Self {
            one_second: TimeWindow::new(Duration::from_secs(1)),
            ten_seconds: TimeWindow::new(Duration::from_secs(10)),
            one_minute: TimeWindow::new(Duration::from_secs(60)),
            five_minutes: TimeWindow::new(Duration::from_secs(300)),
        }
    }

    fn add_update(&mut self, update: &MetricUpdate) {
        self.one_second.add_update(update);
        self.ten_seconds.add_update(update);
        self.one_minute.add_update(update);
        self.five_minutes.add_update(update);
    }

    fn compute_aggregates(&mut self) -> AggregatedMetrics {
        // Clean expired data
        let now = Instant::now();
        self.one_second.clean_expired(now);
        self.ten_seconds.clean_expired(now);
        self.one_minute.clean_expired(now);
        self.five_minutes.clean_expired(now);

        AggregatedMetrics {
            schema_version: Some(AggregatedMetrics::SCHEMA_VERSION.to_string()),
            one_second: self.one_second.aggregate(),
            ten_seconds: self.ten_seconds.aggregate(),
            one_minute: self.one_minute.aggregate(),
            five_minutes: self.five_minutes.aggregate(),
            spreading: None,
        }
    }
}

/// Time window for aggregation
struct TimeWindow {
    duration: Duration,
    series: BTreeMap<&'static str, VecDeque<(Instant, f64)>>,
}

impl TimeWindow {
    const fn new(duration: Duration) -> Self {
        Self {
            duration,
            series: BTreeMap::new(),
        }
    }

    fn add_update(&mut self, update: &MetricUpdate) {
        match update {
            MetricUpdate::Counter {
                name,
                value,
                timestamp,
            } => self.push_sample(name, *timestamp, u64_to_f64(*value)),
            MetricUpdate::Gauge {
                name,
                value,
                timestamp,
            }
            | MetricUpdate::Histogram {
                name,
                value,
                timestamp,
            }
            | MetricUpdate::Summary {
                name,
                value,
                timestamp,
            } => self.push_sample(name, *timestamp, *value),
        }
    }

    fn push_sample(&mut self, name: &'static str, timestamp: Instant, value: f64) {
        let samples = self
            .series
            .entry(name)
            .or_insert_with(|| VecDeque::with_capacity(128));
        samples.push_back((timestamp, value));
    }

    fn clean_expired(&mut self, now: Instant) {
        let Some(cutoff) = now.checked_sub(self.duration) else {
            return;
        };

        let mut empty_keys = Vec::new();
        for (name, samples) in &mut self.series {
            while let Some((timestamp, _)) = samples.front() {
                if *timestamp < cutoff {
                    samples.pop_front();
                } else {
                    break;
                }
            }

            if samples.is_empty() {
                empty_keys.push(*name);
            }
        }

        for name in empty_keys {
            self.series.remove(&name);
        }
    }

    fn aggregate(&self) -> WindowSnapshot {
        let mut aggregates: WindowSnapshot = BTreeMap::new();

        for (name, samples) in &self.series {
            if samples.is_empty() {
                continue;
            }

            let values: Vec<f64> = samples.iter().map(|(_, v)| *v).collect();
            let mut sorted = values.clone();
            sorted.sort_by(f64::total_cmp);

            let count = values.len();
            let sum: f64 = values.iter().copied().sum();
            let mean = if count == 0 {
                0.0
            } else {
                sum / usize_to_f64(count)
            };

            let aggregate = MetricAggregate {
                count,
                sum,
                mean,
                min: sorted.first().copied().unwrap_or(0.0),
                max: sorted.last().copied().unwrap_or(0.0),
                p50: percentile(&sorted, 0.5),
                p90: percentile(&sorted, 0.9),
                p99: percentile(&sorted, 0.99),
            };

            aggregates.insert(*name, aggregate);
        }

        aggregates
    }
}

/// Calculate percentile from sorted values
fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }

    let len = sorted.len();
    let clamped = p.clamp(0.0, 1.0);
    let max_index = len.saturating_sub(1);
    let scaled = usize_to_f64(max_index) * clamped;
    let index = round_f64_to_usize(scaled, max_index);

    sorted[index]
}

/// Per-metric aggregated statistics within a single window.
#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct MetricAggregate {
    /// Number of samples included in this window.
    pub count: usize,
    /// Sum of all sample values to assist with rate calculations.
    pub sum: f64,
    /// Arithmetic mean of the observed values.
    pub mean: f64,
    /// Minimum value recorded during the window.
    pub min: f64,
    /// Maximum value recorded during the window.
    pub max: f64,
    /// 50th percentile (median) estimate.
    pub p50: f64,
    /// 90th percentile estimate.
    pub p90: f64,
    /// 99th percentile estimate.
    pub p99: f64,
}

/// Aggregated metrics across all windows
#[derive(Debug, Clone, Serialize)]
pub struct AggregatedMetrics {
    /// Schema version for backward compatibility tracking (semver format).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub schema_version: Option<String>,
    /// 1-second rolling aggregate for real-time monitoring.
    pub one_second: WindowSnapshot,
    /// 10-second aggregate for short trend analysis.
    pub ten_seconds: WindowSnapshot,
    /// 1-minute aggregate for medium-term trend detection.
    pub one_minute: WindowSnapshot,
    /// 5-minute aggregate for stability assessment.
    pub five_minutes: WindowSnapshot,
    /// Derived spreading summary constructed by higher-level metrics consumers.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub spreading: Option<SpreadingSummary>,
}

impl AggregatedMetrics {
    /// Current schema version for metrics export format (follows semver).
    /// Update this when making backward-incompatible changes to the structure.
    pub const SCHEMA_VERSION: &'static str = "1.1.0";

    /// Create an empty metrics aggregate with schema version.
    #[must_use]
    pub fn empty_with_version() -> Self {
        Self {
            schema_version: Some(Self::SCHEMA_VERSION.to_string()),
            one_second: BTreeMap::new(),
            ten_seconds: BTreeMap::new(),
            one_minute: BTreeMap::new(),
            five_minutes: BTreeMap::new(),
            spreading: None,
        }
    }
}

/// Latency summary derived for a storage tier.
#[derive(Debug, Clone, Serialize, Default, PartialEq)]
pub struct TierLatencySummary {
    /// Number of samples contributing to the summary.
    pub samples: usize,
    /// Mean latency in seconds.
    pub mean_seconds: f64,
    /// 50th percentile latency (seconds).
    pub p50_seconds: f64,
    /// 95th percentile latency (seconds).
    pub p95_seconds: f64,
    /// 99th percentile latency (seconds).
    pub p99_seconds: f64,
}

/// High-level spreading telemetry exposed alongside window snapshots.
#[derive(Debug, Clone, Serialize, Default, PartialEq)]
pub struct SpreadingSummary {
    /// Latency summaries keyed by tier label.
    pub per_tier: BTreeMap<String, TierLatencySummary>,
    /// Total recalls routed through spreading (gauge snapshot).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub activations_total: Option<u64>,
    /// Total latency budget violations observed.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub latency_budget_violations_total: Option<u64>,
    /// Total fallback count recorded.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fallback_total: Option<u64>,
    /// Total spreading failures observed.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub failure_total: Option<u64>,
    /// Current breaker state (0=Closed,1=HalfOpen,2=Open).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub breaker_state: Option<u64>,
    /// Total breaker transitions recorded.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub breaker_transitions_total: Option<u64>,
    /// Total GPU launch attempts recorded.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gpu_launch_total: Option<u64>,
    /// Total GPU fallbacks recorded.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gpu_fallback_total: Option<u64>,
    /// Current activation pool utilisation (0.0 – 1.0).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pool_utilization: Option<f64>,
    /// Current activation pool hit rate (0.0 – 1.0).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pool_hit_rate: Option<f64>,
}

fn round_f64_to_usize(value: f64, max: usize) -> usize {
    if max == 0 {
        return 0;
    }

    if !value.is_finite() {
        return 0;
    }

    let clamped = value.clamp(0.0, usize_to_f64(max));
    let rounded = clamped.round();
    if rounded == 0.0 {
        return 0;
    }

    let bits = rounded.to_bits();
    let exponent_bits = (bits >> 52) & 0x7FF;
    let exponent = i32::try_from(exponent_bits).unwrap_or(0);
    if exponent < 1023 {
        return 0;
    }

    let mantissa = bits & 0x000F_FFFF_FFFF_FFFF;
    let mut value_int = u128::from(mantissa | (1_u64 << 52));
    let shift = exponent - 1023 - 52;
    match shift.cmp(&0) {
        std::cmp::Ordering::Greater => value_int <<= shift.unsigned_abs(),
        std::cmp::Ordering::Less => value_int >>= shift.unsigned_abs(),
        std::cmp::Ordering::Equal => {}
    }

    usize::try_from(value_int).unwrap_or(max).min(max)
}

fn usize_to_f64(value: usize) -> f64 {
    u64::try_from(value).map_or_else(|_| u64_to_f64(u64::MAX), u64_to_f64)
}

fn u64_to_f64(value: u64) -> f64 {
    let high_part = u32::try_from(value >> 32).unwrap_or(u32::MAX);
    let low_part = u32::try_from(value & 0xFFFF_FFFF).unwrap_or(u32::MAX);
    f64::from(high_part).mul_add(4_294_967_296.0, f64::from(low_part))
}

/// Export statistics
#[derive(Debug, Clone, Serialize)]
pub struct ExportStats {
    /// Total updates successfully exported since startup.
    pub exported: u64,
    /// Number of updates dropped due to backpressure.
    pub dropped: u64,
    /// Current depth of the streaming queue.
    pub queue_depth: usize,
}

impl Default for StreamingAggregator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;

    fn assert_close(actual: f64, expected: f64) {
        let diff = (actual - expected).abs();
        assert!(
            diff <= f64::EPSILON,
            "expected {expected}, got {actual} (diff {diff})"
        );
    }
    use std::time::Duration;

    #[test]
    fn test_streaming_aggregation() {
        let aggregator = StreamingAggregator::new();

        // Queue some updates
        for i in 0..100 {
            aggregator.queue_update(MetricUpdate::Counter {
                name: "test_counter",
                value: 1,
                timestamp: Instant::now(),
            });

            aggregator.queue_update(MetricUpdate::Histogram {
                name: "test_histogram",
                value: f64::from(i),
                timestamp: Instant::now(),
            });
        }

        // Process updates
        let metrics = aggregator.process_updates();
        let counter = metrics
            .one_second
            .get("test_counter")
            .cloned()
            .expect("counter aggregate present");
        assert!(counter.count > 0);

        let histogram = metrics
            .one_second
            .get("test_histogram")
            .cloned()
            .expect("histogram aggregate present");
        assert!(histogram.max > 0.0);

        // Calling without new updates should retain prior aggregates.
        let cached = aggregator.process_updates();
        assert_eq!(cached.one_second, metrics.one_second);

        // Check stats
        let stats = aggregator.export_stats();
        assert_eq!(stats.exported, 200);
        assert_eq!(stats.dropped, 0);
    }

    #[test]
    fn test_window_aggregation() {
        let mut window = TimeWindow::new(Duration::from_secs(1));
        let now = Instant::now();

        // Add some data points
        for i in 0..10 {
            window.add_update(&MetricUpdate::Histogram {
                name: "test",
                value: f64::from(i),
                timestamp: now,
            });
        }

        let aggregate = window.aggregate();
        let metrics = aggregate
            .get("test")
            .cloned()
            .expect("aggregate for 'test'");
        assert_eq!(metrics.count, 10);
        assert_close(metrics.min, 0.0);
        assert_close(metrics.max, 9.0);
        assert_close(metrics.mean, 4.5);

        // After the window expires all samples should be removed.
        window.clean_expired(now + Duration::from_secs(2));
        assert!(window.aggregate().is_empty());
    }
}
