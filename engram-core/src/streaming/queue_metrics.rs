//! Queue metrics tracking and reporting for observation queue monitoring.
//!
//! Provides detailed metrics for queue health, throughput, latency, and
//! backpressure detection. Designed for integration with Prometheus and
//! other monitoring systems.

use super::observation_queue::{ObservationQueue, QueueDepths, QueueMetrics};
use std::time::Instant;

/// Detailed queue statistics for monitoring and alerting.
#[derive(Debug, Clone)]
pub struct QueueStatistics {
    /// Queue depths by priority
    pub depths: QueueDepths,
    /// Total observations enqueued (lifetime)
    pub total_enqueued: u64,
    /// Total observations dequeued (lifetime)
    pub total_dequeued: u64,
    /// Backpressure events triggered
    pub backpressure_events: u64,
    /// Current queue utilization (0.0-1.0)
    pub utilization: f32,
    /// Whether backpressure is currently active
    pub backpressure_active: bool,
    /// Enqueue rate (ops/sec) - requires periodic sampling
    pub enqueue_rate: Option<f32>,
    /// Dequeue rate (ops/sec) - requires periodic sampling
    pub dequeue_rate: Option<f32>,
}

/// Tracks queue metrics over time for rate calculation.
///
/// Maintains historical snapshots to calculate throughput rates
/// and detect trends.
pub struct QueueMetricsTracker {
    /// Reference to the queue being tracked
    queue: std::sync::Arc<ObservationQueue>,
    /// Last snapshot timestamp
    last_snapshot_time: Instant,
    /// Last enqueued count
    last_enqueued: u64,
    /// Last dequeued count
    last_dequeued: u64,
}

impl QueueMetricsTracker {
    /// Create a new metrics tracker for the given queue.
    ///
    /// # Arguments
    ///
    /// * `queue` - Shared reference to observation queue
    ///
    /// # Example
    ///
    /// ```rust
    /// use engram_core::streaming::observation_queue::{ObservationQueue, QueueConfig};
    /// use engram_core::streaming::queue_metrics::QueueMetricsTracker;
    /// use std::sync::Arc;
    ///
    /// let queue = Arc::new(ObservationQueue::new(QueueConfig::default()));
    /// let tracker = QueueMetricsTracker::new(queue);
    /// ```
    #[must_use]
    pub fn new(queue: std::sync::Arc<ObservationQueue>) -> Self {
        let metrics = queue.metrics();
        Self {
            queue,
            last_snapshot_time: Instant::now(),
            last_enqueued: metrics.total_enqueued,
            last_dequeued: metrics.total_dequeued,
        }
    }

    /// Take a metrics snapshot and calculate current statistics.
    ///
    /// This method should be called periodically (e.g., every 1-5 seconds)
    /// to update rate calculations.
    ///
    /// # Returns
    ///
    /// Current queue statistics including rates calculated from the
    /// time elapsed since last snapshot.
    ///
    /// # Example
    ///
    /// ```rust
    /// use engram_core::streaming::observation_queue::{ObservationQueue, QueueConfig};
    /// use engram_core::streaming::queue_metrics::QueueMetricsTracker;
    /// use std::sync::Arc;
    /// use std::thread;
    /// use std::time::Duration;
    ///
    /// let queue = Arc::new(ObservationQueue::new(QueueConfig::default()));
    /// let mut tracker = QueueMetricsTracker::new(queue);
    ///
    /// // Simulate some activity...
    /// thread::sleep(Duration::from_millis(100));
    ///
    /// let stats = tracker.snapshot();
    /// println!("Utilization: {:.2}%", stats.utilization * 100.0);
    /// ```
    pub fn snapshot(&mut self) -> QueueStatistics {
        let now = Instant::now();
        let elapsed = now.duration_since(self.last_snapshot_time);
        let elapsed_secs = elapsed.as_secs_f32();

        let metrics = self.queue.metrics();
        let depths = metrics.depths;
        let total_capacity = self.queue.total_capacity();

        // Calculate rates
        let (enqueue_rate, dequeue_rate) = if elapsed_secs > 0.0 {
            let enqueue_delta = metrics.total_enqueued.saturating_sub(self.last_enqueued);
            let dequeue_delta = metrics.total_dequeued.saturating_sub(self.last_dequeued);

            #[allow(clippy::cast_precision_loss)]
            let enq_rate = (enqueue_delta as f32) / elapsed_secs;
            #[allow(clippy::cast_precision_loss)]
            let deq_rate = (dequeue_delta as f32) / elapsed_secs;

            (Some(enq_rate), Some(deq_rate))
        } else {
            (None, None)
        };

        // Calculate utilization
        #[allow(clippy::cast_precision_loss)]
        let utilization = if total_capacity > 0 {
            (depths.total() as f32) / (total_capacity as f32)
        } else {
            0.0
        };

        // Update last snapshot
        self.last_snapshot_time = now;
        self.last_enqueued = metrics.total_enqueued;
        self.last_dequeued = metrics.total_dequeued;

        QueueStatistics {
            depths,
            total_enqueued: metrics.total_enqueued,
            total_dequeued: metrics.total_dequeued,
            backpressure_events: metrics.backpressure_events,
            utilization,
            backpressure_active: self.queue.should_apply_backpressure(),
            enqueue_rate,
            dequeue_rate,
        }
    }

    /// Get current metrics without updating snapshot state.
    ///
    /// Use this for quick health checks without affecting rate calculations.
    #[must_use]
    pub fn current_metrics(&self) -> QueueMetrics {
        self.queue.metrics()
    }

    /// Check if queue is healthy (below backpressure threshold).
    #[must_use]
    pub fn is_healthy(&self) -> bool {
        !self.queue.should_apply_backpressure()
    }

    /// Get queue utilization as percentage (0.0-1.0).
    #[must_use]
    pub fn utilization(&self) -> f32 {
        let depths = self.queue.depths();
        let total_capacity = self.queue.total_capacity();

        #[allow(clippy::cast_precision_loss)]
        if total_capacity > 0 {
            (depths.total() as f32) / (total_capacity as f32)
        } else {
            0.0
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)] // Test code - unwrap is acceptable for clarity
mod tests {
    use super::*;
    use crate::Confidence;
    use crate::memory::EpisodeBuilder;
    use crate::streaming::observation_queue::{ObservationPriority, QueueConfig};
    use crate::types::MemorySpaceId;
    use chrono::Utc;
    use std::sync::Arc;
    use std::thread;
    use std::time::Duration;

    fn test_episode(content: &str) -> crate::memory::Episode {
        EpisodeBuilder::new()
            .id(format!("test_{content}"))
            .when(Utc::now())
            .what(content.to_string())
            .embedding([0.0f32; 768])
            .confidence(Confidence::MEDIUM)
            .build()
    }

    fn test_space_id() -> MemorySpaceId {
        MemorySpaceId::default()
    }

    #[test]
    fn test_metrics_tracker_basic() {
        let queue = Arc::new(ObservationQueue::new(QueueConfig::default()));
        let mut tracker = QueueMetricsTracker::new(Arc::clone(&queue));

        // Enqueue some observations
        for i in 0..10 {
            queue
                .enqueue(
                    test_space_id(),
                    test_episode("test"),
                    i,
                    ObservationPriority::Normal,
                )
                .unwrap();
        }

        // Wait a bit for rate calculation
        thread::sleep(Duration::from_millis(100));

        let stats = tracker.snapshot();
        assert_eq!(stats.total_enqueued, 10);
        assert_eq!(stats.depths.normal, 10);
        assert!(stats.enqueue_rate.is_some());
        assert!(stats.enqueue_rate.unwrap() > 0.0);
    }

    #[test]
    fn test_utilization_calculation() {
        let config = QueueConfig {
            high_capacity: 100,
            normal_capacity: 100,
            low_capacity: 100,
        };
        let queue = Arc::new(ObservationQueue::new(config));
        let tracker = QueueMetricsTracker::new(Arc::clone(&queue));

        // Empty queue - 0% utilization
        assert!((tracker.utilization() - 0.0).abs() < f32::EPSILON);

        // Fill to 50% total (150 items across 300 capacity)
        // Distribute 50 items to each lane
        for i in 0..50 {
            queue
                .enqueue(
                    test_space_id(),
                    test_episode("test"),
                    i,
                    ObservationPriority::High,
                )
                .unwrap();
            queue
                .enqueue(
                    test_space_id(),
                    test_episode("test"),
                    i + 100,
                    ObservationPriority::Normal,
                )
                .unwrap();
            queue
                .enqueue(
                    test_space_id(),
                    test_episode("test"),
                    i + 200,
                    ObservationPriority::Low,
                )
                .unwrap();
        }

        let util = tracker.utilization();
        assert!((util - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_backpressure_detection() {
        let config = QueueConfig {
            high_capacity: 100,
            normal_capacity: 100,
            low_capacity: 100,
        };
        let queue = Arc::new(ObservationQueue::new(config));
        let mut tracker = QueueMetricsTracker::new(Arc::clone(&queue));

        // Fill to 70% total - no backpressure (210 items)
        // Distribute 70 items to each lane
        for i in 0..70 {
            queue
                .enqueue(
                    test_space_id(),
                    test_episode("test"),
                    i,
                    ObservationPriority::High,
                )
                .unwrap();
            queue
                .enqueue(
                    test_space_id(),
                    test_episode("test"),
                    i + 100,
                    ObservationPriority::Normal,
                )
                .unwrap();
            queue
                .enqueue(
                    test_space_id(),
                    test_episode("test"),
                    i + 200,
                    ObservationPriority::Low,
                )
                .unwrap();
        }

        let stats = tracker.snapshot();
        assert!(!stats.backpressure_active);

        // Fill to 85% total - backpressure (255 items)
        // Add 15 more items to each lane
        for i in 70..85 {
            queue
                .enqueue(
                    test_space_id(),
                    test_episode("test"),
                    i,
                    ObservationPriority::High,
                )
                .unwrap();
            queue
                .enqueue(
                    test_space_id(),
                    test_episode("test"),
                    i + 100,
                    ObservationPriority::Normal,
                )
                .unwrap();
            queue
                .enqueue(
                    test_space_id(),
                    test_episode("test"),
                    i + 200,
                    ObservationPriority::Low,
                )
                .unwrap();
        }

        let stats = tracker.snapshot();
        assert!(stats.backpressure_active);
    }

    #[test]
    fn test_rate_calculation() {
        let queue = Arc::new(ObservationQueue::new(QueueConfig::default()));
        let mut tracker = QueueMetricsTracker::new(Arc::clone(&queue));

        // First snapshot establishes baseline
        // Wait a bit to ensure elapsed_secs > 0 for rate calculation
        thread::sleep(Duration::from_millis(10));
        let stats1 = tracker.snapshot();
        // First snapshot may or may not have a rate depending on timing,
        // but it should be 0 since nothing has been enqueued yet
        if let Some(rate) = stats1.enqueue_rate {
            assert!(rate.abs() < f32::EPSILON);
        }

        // Enqueue 100 observations
        for i in 0..100 {
            queue
                .enqueue(
                    test_space_id(),
                    test_episode("test"),
                    i,
                    ObservationPriority::Normal,
                )
                .unwrap();
        }

        // Wait 100ms
        thread::sleep(Duration::from_millis(100));

        let stats2 = tracker.snapshot();
        assert!(stats2.enqueue_rate.is_some());
        let rate = stats2.enqueue_rate.unwrap();

        // Should be roughly 100 / 0.1 = 1000 ops/sec
        // Allow wide tolerance for CI timing variance
        assert!(rate > 500.0 && rate < 2000.0);
    }

    #[test]
    fn test_dequeue_rate() {
        let queue = Arc::new(ObservationQueue::new(QueueConfig::default()));
        let mut tracker = QueueMetricsTracker::new(Arc::clone(&queue));

        // Enqueue 1000
        for i in 0..1_000 {
            queue
                .enqueue(
                    test_space_id(),
                    test_episode("test"),
                    i,
                    ObservationPriority::Normal,
                )
                .unwrap();
        }

        tracker.snapshot(); // Reset baseline

        // Dequeue 100
        for _ in 0..100 {
            let _ = queue.dequeue();
        }

        thread::sleep(Duration::from_millis(100));

        let stats = tracker.snapshot();
        assert!(stats.dequeue_rate.is_some());
        let rate = stats.dequeue_rate.unwrap();

        // Should be roughly 100 / 0.1 = 1000 ops/sec
        assert!(rate > 500.0 && rate < 2000.0);
    }

    #[test]
    fn test_current_metrics_no_mutation() {
        let queue = Arc::new(ObservationQueue::new(QueueConfig::default()));
        let tracker = QueueMetricsTracker::new(Arc::clone(&queue));

        // Enqueue some items
        for i in 0..10 {
            queue
                .enqueue(
                    test_space_id(),
                    test_episode("test"),
                    i,
                    ObservationPriority::Normal,
                )
                .unwrap();
        }

        let metrics1 = tracker.current_metrics();
        thread::sleep(Duration::from_millis(10));
        let metrics2 = tracker.current_metrics();

        // Should be same since we didn't modify queue
        assert_eq!(metrics1.total_enqueued, metrics2.total_enqueued);
    }
}
