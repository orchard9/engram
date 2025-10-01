//! Streaming metrics aggregation for real-time export

use crossbeam_queue::ArrayQueue;
use crossbeam_utils::CachePadded;
use parking_lot::RwLock;
use std::collections::VecDeque;
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

        if updates.is_empty() {
            return AggregatedMetrics::empty();
        }

        // Aggregate updates
        let mut windows = self.windows.write();
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
    fn new() -> Self {
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
            one_second: self.one_second.aggregate(),
            ten_seconds: self.ten_seconds.aggregate(),
            one_minute: self.one_minute.aggregate(),
            five_minutes: self.five_minutes.aggregate(),
        }
    }
}

/// Time window for aggregation
struct TimeWindow {
    duration: Duration,
    data: VecDeque<(Instant, f64)>,
    counter_sums: dashmap::DashMap<&'static str, u64>,
}

impl TimeWindow {
    fn new(duration: Duration) -> Self {
        Self {
            duration,
            data: VecDeque::with_capacity(1024),
            counter_sums: dashmap::DashMap::new(),
        }
    }

    fn add_update(&mut self, update: &MetricUpdate) {
        match update {
            MetricUpdate::Counter {
                name,
                value,
                timestamp,
            } => {
                *self.counter_sums.entry(name).or_insert(0) += value;
                let value_f64 = u64_to_f64(*value);
                self.data.push_back((*timestamp, value_f64));
            }
            MetricUpdate::Gauge {
                name: _,
                value,
                timestamp,
            }
            | MetricUpdate::Histogram {
                name: _,
                value,
                timestamp,
            }
            | MetricUpdate::Summary {
                name: _,
                value,
                timestamp,
            } => {
                self.data.push_back((*timestamp, *value));
            }
        }
    }

    fn clean_expired(&mut self, now: Instant) {
        let Some(cutoff) = now.checked_sub(self.duration) else {
            return;
        };
        while let Some((timestamp, _)) = self.data.front() {
            if *timestamp < cutoff {
                self.data.pop_front();
            } else {
                break;
            }
        }
    }

    fn aggregate(&self) -> WindowAggregate {
        if self.data.is_empty() {
            return WindowAggregate::empty();
        }

        let values: Vec<f64> = self.data.iter().map(|(_, v)| *v).collect();
        let mut sorted = values.clone();
        sorted.sort_by(f64::total_cmp);

        let count = values.len();
        let sum: f64 = values.iter().sum();
        let mean = sum / usize_to_f64(count);

        WindowAggregate {
            count,
            sum,
            mean,
            min: sorted.first().copied().unwrap_or(0.0),
            max: sorted.last().copied().unwrap_or(0.0),
            p50: percentile(&sorted, 0.5),
            p90: percentile(&sorted, 0.9),
            p99: percentile(&sorted, 0.99),
        }
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

/// Aggregated metrics for a time window
#[derive(Debug, Clone)]
pub struct WindowAggregate {
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

impl WindowAggregate {
    const fn empty() -> Self {
        Self {
            count: 0,
            sum: 0.0,
            mean: 0.0,
            min: 0.0,
            max: 0.0,
            p50: 0.0,
            p90: 0.0,
            p99: 0.0,
        }
    }
}

/// Aggregated metrics across all windows
#[derive(Debug, Clone)]
pub struct AggregatedMetrics {
    /// 1-second rolling aggregate for real-time monitoring.
    pub one_second: WindowAggregate,
    /// 10-second aggregate for short trend analysis.
    pub ten_seconds: WindowAggregate,
    /// 1-minute aggregate for medium-term trend detection.
    pub one_minute: WindowAggregate,
    /// 5-minute aggregate for stability assessment.
    pub five_minutes: WindowAggregate,
}

impl AggregatedMetrics {
    const fn empty() -> Self {
        Self {
            one_second: WindowAggregate::empty(),
            ten_seconds: WindowAggregate::empty(),
            one_minute: WindowAggregate::empty(),
            five_minutes: WindowAggregate::empty(),
        }
    }
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
#[derive(Debug, Clone)]
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
        assert!(metrics.one_second.count > 0);

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
        assert_eq!(aggregate.count, 10);
        assert_close(aggregate.min, 0.0);
        assert_close(aggregate.max, 9.0);
        assert_close(aggregate.mean, 4.5);
    }
}
