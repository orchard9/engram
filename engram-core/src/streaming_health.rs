//! Event Streaming Health Tracking
//!
//! Provides visibility into the health of the event streaming subsystem.
//! Critical for detecting when the keepalive subscriber has died or
//! when event delivery is fundamentally broken.

use parking_lot::RwLock;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

/// Health status of the event streaming subsystem
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StreamingHealthStatus {
    /// Streaming is healthy - events being delivered
    Healthy,
    /// Streaming is degraded - some events dropping but keepalive alive
    Degraded,
    /// Streaming is broken - no subscribers (keepalive died)
    Broken,
    /// Streaming is disabled - not initialized
    Disabled,
}

/// Detailed health metrics for event streaming
#[derive(Debug, Clone)]
pub struct StreamingHealthMetrics {
    /// Current health status
    pub status: StreamingHealthStatus,
    /// Total events broadcast (successful + failed)
    pub events_attempted: u64,
    /// Total events successfully delivered
    pub events_delivered: u64,
    /// Total events dropped (no subscribers)
    pub events_dropped: u64,
    /// Current subscriber count
    pub subscriber_count: usize,
    /// Timestamp of last successful delivery (Unix timestamp)
    pub last_successful_delivery: Option<u64>,
    /// Timestamp of last delivery failure (Unix timestamp)
    pub last_failure: Option<u64>,
    /// Duration since last successful delivery
    pub time_since_last_success: Option<Duration>,
    /// Whether keepalive subscriber is detected
    pub keepalive_present: bool,
}

/// Event streaming health tracker
///
/// Thread-safe tracker that monitors event delivery health and detects
/// when the streaming subsystem is broken (e.g., keepalive subscriber died).
#[derive(Debug)]
pub struct StreamingHealthTracker {
    /// Current health status
    status: Arc<RwLock<StreamingHealthStatus>>,
    /// Total attempted broadcasts
    events_attempted: Arc<AtomicU64>,
    /// Successfully delivered events
    events_delivered: Arc<AtomicU64>,
    /// Dropped events (no subscribers)
    events_dropped: Arc<AtomicU64>,
    /// Last subscriber count seen
    last_subscriber_count: Arc<AtomicUsize>,
    /// Timestamp of last successful delivery
    last_success_timestamp: Arc<AtomicU64>,
    /// Timestamp of last failure
    last_failure_timestamp: Arc<AtomicU64>,
}

impl Default for StreamingHealthTracker {
    fn default() -> Self {
        Self::new()
    }
}

impl StreamingHealthTracker {
    /// Create a new health tracker
    #[must_use]
    pub fn new() -> Self {
        Self {
            status: Arc::new(RwLock::new(StreamingHealthStatus::Disabled)),
            events_attempted: Arc::new(AtomicU64::new(0)),
            events_delivered: Arc::new(AtomicU64::new(0)),
            events_dropped: Arc::new(AtomicU64::new(0)),
            last_subscriber_count: Arc::new(AtomicUsize::new(0)),
            last_success_timestamp: Arc::new(AtomicU64::new(0)),
            last_failure_timestamp: Arc::new(AtomicU64::new(0)),
        }
    }

    /// Mark streaming as enabled
    pub fn enable(&self) {
        let mut status = self.status.write();
        *status = StreamingHealthStatus::Healthy;
    }

    /// Record a successful event delivery
    pub fn record_success(&self, subscriber_count: usize) {
        self.events_attempted.fetch_add(1, Ordering::Relaxed);
        self.events_delivered.fetch_add(1, Ordering::Relaxed);
        self.last_subscriber_count
            .store(subscriber_count, Ordering::Relaxed);

        // Update timestamp
        if let Ok(duration) = SystemTime::now().duration_since(UNIX_EPOCH) {
            self.last_success_timestamp
                .store(duration.as_secs(), Ordering::Relaxed);
        }

        // Update status to healthy if we had successful delivery
        let mut status = self.status.write();
        if *status != StreamingHealthStatus::Disabled {
            *status = StreamingHealthStatus::Healthy;
        }
    }

    /// Record a failed event delivery (no subscribers)
    ///
    /// This is a CRITICAL failure - the keepalive subscriber should prevent this.
    /// If this happens, it means the invariant is violated.
    pub fn record_failure(&self, subscriber_count: usize) {
        self.events_attempted.fetch_add(1, Ordering::Relaxed);
        self.events_dropped.fetch_add(1, Ordering::Relaxed);
        self.last_subscriber_count
            .store(subscriber_count, Ordering::Relaxed);

        // Update timestamp
        if let Ok(duration) = SystemTime::now().duration_since(UNIX_EPOCH) {
            self.last_failure_timestamp
                .store(duration.as_secs(), Ordering::Relaxed);
        }

        // Update status based on subscriber count
        let mut status = self.status.write();
        if *status == StreamingHealthStatus::Disabled {
            return;
        }

        *status = if subscriber_count == 0 {
            // CRITICAL: No subscribers means keepalive died
            StreamingHealthStatus::Broken
        } else {
            // Some subscribers exist but delivery failed (buffer full?)
            StreamingHealthStatus::Degraded
        };
    }

    /// Get current health status
    #[must_use]
    pub fn status(&self) -> StreamingHealthStatus {
        *self.status.read()
    }

    /// Get comprehensive health metrics
    #[must_use]
    pub fn metrics(&self) -> StreamingHealthMetrics {
        let status = *self.status.read();
        let events_attempted = self.events_attempted.load(Ordering::Relaxed);
        let events_delivered = self.events_delivered.load(Ordering::Relaxed);
        let events_dropped = self.events_dropped.load(Ordering::Relaxed);
        let subscriber_count = self.last_subscriber_count.load(Ordering::Relaxed);

        let last_success = self.last_success_timestamp.load(Ordering::Relaxed);
        let last_failure = self.last_failure_timestamp.load(Ordering::Relaxed);

        let last_successful_delivery = if last_success > 0 {
            Some(last_success)
        } else {
            None
        };

        let last_failure_ts = if last_failure > 0 {
            Some(last_failure)
        } else {
            None
        };

        let time_since_last_success = if last_success > 0 {
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .ok()
                .and_then(|now| {
                    let last = Duration::from_secs(last_success);
                    now.checked_sub(last)
                })
        } else {
            None
        };

        // Keepalive is present if we have at least one subscriber
        let keepalive_present = subscriber_count > 0;

        StreamingHealthMetrics {
            status,
            events_attempted,
            events_delivered,
            events_dropped,
            subscriber_count,
            last_successful_delivery,
            last_failure: last_failure_ts,
            time_since_last_success,
            keepalive_present,
        }
    }

    /// Check if streaming is healthy
    #[must_use]
    pub fn is_healthy(&self) -> bool {
        matches!(self.status(), StreamingHealthStatus::Healthy)
    }

    /// Check if streaming is broken (CRITICAL state)
    #[must_use]
    pub fn is_broken(&self) -> bool {
        matches!(self.status(), StreamingHealthStatus::Broken)
    }

    /// Get delivery success rate (0.0 to 1.0)
    #[must_use]
    pub fn success_rate(&self) -> f32 {
        let attempted = self.events_attempted.load(Ordering::Relaxed);
        if attempted == 0 {
            return 1.0;
        }

        let delivered = self.events_delivered.load(Ordering::Relaxed);
        delivered as f32 / attempted as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_initial_state() {
        let tracker = StreamingHealthTracker::new();
        assert_eq!(tracker.status(), StreamingHealthStatus::Disabled);
        assert!((tracker.success_rate() - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_enable_streaming() {
        let tracker = StreamingHealthTracker::new();
        tracker.enable();
        assert_eq!(tracker.status(), StreamingHealthStatus::Healthy);
    }

    #[test]
    fn test_success_recording() {
        let tracker = StreamingHealthTracker::new();
        tracker.enable();

        tracker.record_success(1);
        let metrics = tracker.metrics();

        assert_eq!(metrics.events_attempted, 1);
        assert_eq!(metrics.events_delivered, 1);
        assert_eq!(metrics.events_dropped, 0);
        assert_eq!(metrics.subscriber_count, 1);
        assert!(metrics.keepalive_present);
        assert_eq!(tracker.status(), StreamingHealthStatus::Healthy);
    }

    #[test]
    fn test_failure_with_zero_subscribers() {
        let tracker = StreamingHealthTracker::new();
        tracker.enable();

        tracker.record_failure(0);
        let metrics = tracker.metrics();

        assert_eq!(metrics.events_attempted, 1);
        assert_eq!(metrics.events_delivered, 0);
        assert_eq!(metrics.events_dropped, 1);
        assert_eq!(metrics.subscriber_count, 0);
        assert!(!metrics.keepalive_present);
        assert_eq!(tracker.status(), StreamingHealthStatus::Broken);
    }

    #[test]
    fn test_failure_with_subscribers() {
        let tracker = StreamingHealthTracker::new();
        tracker.enable();

        tracker.record_failure(2); // Subscribers exist but delivery failed
        assert_eq!(tracker.status(), StreamingHealthStatus::Degraded);
    }

    #[test]
    fn test_success_rate() {
        let tracker = StreamingHealthTracker::new();
        tracker.enable();

        tracker.record_success(1);
        tracker.record_success(1);
        tracker.record_failure(1);

        assert!((tracker.success_rate() - 2.0 / 3.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_recovery_after_failure() {
        let tracker = StreamingHealthTracker::new();
        tracker.enable();

        tracker.record_failure(0);
        assert_eq!(tracker.status(), StreamingHealthStatus::Broken);

        tracker.record_success(1);
        assert_eq!(tracker.status(), StreamingHealthStatus::Healthy);
    }
}
