//! Streaming metrics aggregation for real-time export

use crossbeam_queue::ArrayQueue;
use crossbeam_utils::CachePadded;
use parking_lot::RwLock;
use std::collections::VecDeque;
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
    #[must_use]
    pub fn new() -> Self {
        Self::with_capacity(65536) // 64K updates buffer
    }

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
    #[inline(always)]
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
        for update in updates {
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
    Counter {
        name: &'static str,
        value: u64,
        timestamp: Instant,
    },
    Gauge {
        name: &'static str,
        value: f64,
        timestamp: Instant,
    },
    Histogram {
        name: &'static str,
        value: f64,
        timestamp: Instant,
    },
    Summary {
        name: &'static str,
        value: f64,
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

    fn add_update(&mut self, update: MetricUpdate) {
        self.one_second.add_update(&update);
        self.ten_seconds.add_update(&update);
        self.one_minute.add_update(&update);
        self.five_minutes.add_update(&update);
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
                self.data.push_back((*timestamp, *value as f64));
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
        let cutoff = now.checked_sub(self.duration).unwrap();
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
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

        WindowAggregate {
            count: values.len(),
            sum: values.iter().sum(),
            mean: values.iter().sum::<f64>() / values.len() as f64,
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

    let index = ((sorted.len() - 1) as f64 * p) as usize;
    sorted[index]
}

/// Aggregated metrics for a time window
#[derive(Debug, Clone)]
pub struct WindowAggregate {
    pub count: usize,
    pub sum: f64,
    pub mean: f64,
    pub min: f64,
    pub max: f64,
    pub p50: f64,
    pub p90: f64,
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
    pub one_second: WindowAggregate,
    pub ten_seconds: WindowAggregate,
    pub one_minute: WindowAggregate,
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

/// Export statistics
#[derive(Debug, Clone)]
pub struct ExportStats {
    pub exported: u64,
    pub dropped: u64,
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
        assert_eq!(aggregate.min, 0.0);
        assert_eq!(aggregate.max, 9.0);
        assert_eq!(aggregate.mean, 4.5);
    }
}
