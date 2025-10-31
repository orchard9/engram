//! Backpressure monitoring and adaptive flow control for streaming observations.
//!
//! This module implements adaptive backpressure with:
//! - Periodic queue depth monitoring
//! - State-based pressure levels (Normal/Warning/Critical/Overloaded)
//! - Recommended batch sizes based on current pressure
//! - Retry-after calculation for admission control
//!
//! ## Architecture
//!
//! The `BackpressureMonitor` runs as a background task, periodically sampling
//! queue depth and broadcasting state changes to active subscribers (streaming
//! handlers, worker pools).
//!
//! ```text
//! BackpressureMonitor (100ms check interval)
//!         |
//!         ├─> broadcast::Sender<BackpressureState>
//!         |        ↓
//!         |   StreamingHandlers (forward to clients)
//!         |   WorkerPool (adjust batch size)
//!         |
//!         └─> ObservationQueue (read-only depth sampling)
//! ```

use crate::streaming::ObservationQueue;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::broadcast;

/// Backpressure state based on queue pressure (depth / capacity).
///
/// State transitions determine:
/// - Batch size recommendations (latency vs throughput)
/// - Flow control messages to clients
/// - Admission control decisions
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackpressureState {
    /// Normal operation: < 50% capacity
    ///
    /// Optimize for low latency with small batches.
    Normal,

    /// Warning: 50-80% capacity
    ///
    /// Balanced mode - moderate batching.
    Warning,

    /// Critical: 80-95% capacity
    ///
    /// High pressure - large batches for throughput.
    Critical,

    /// Overloaded: > 95% capacity
    ///
    /// Maximum batching, approaching admission control rejection.
    Overloaded,
}

impl BackpressureState {
    /// Determine backpressure state from queue pressure.
    ///
    /// # Arguments
    ///
    /// * `pressure` - Queue depth / capacity ratio (0.0 - 1.0)
    ///
    /// # Examples
    ///
    /// ```
    /// use engram_core::streaming::backpressure::BackpressureState;
    ///
    /// assert_eq!(BackpressureState::from_pressure(0.3), BackpressureState::Normal);
    /// assert_eq!(BackpressureState::from_pressure(0.6), BackpressureState::Warning);
    /// assert_eq!(BackpressureState::from_pressure(0.85), BackpressureState::Critical);
    /// assert_eq!(BackpressureState::from_pressure(0.98), BackpressureState::Overloaded);
    /// ```
    #[must_use]
    pub fn from_pressure(pressure: f32) -> Self {
        match pressure {
            p if p < 0.5 => Self::Normal,
            p if p < 0.8 => Self::Warning,
            p if p < 0.95 => Self::Critical,
            _ => Self::Overloaded,
        }
    }

    /// Get recommended batch size for current pressure level.
    ///
    /// Batch size trades latency for throughput:
    /// - Normal: 10 (low latency ~10ms)
    /// - Warning: 100 (balanced ~20ms)
    /// - Critical: 500 (high throughput ~50ms)
    /// - Overloaded: 1000 (maximum throughput ~100ms)
    ///
    /// Worker pools should use this to adaptively adjust batch sizes.
    #[must_use]
    pub const fn recommended_batch_size(self) -> usize {
        match self {
            Self::Normal => 10,
            Self::Warning => 100,
            Self::Critical => 500,
            Self::Overloaded => 1000,
        }
    }

    /// Check if admission control should reject new observations.
    ///
    /// Only reject in Overloaded state (> 95% capacity).
    #[must_use]
    pub const fn should_reject(self) -> bool {
        matches!(self, Self::Overloaded)
    }
}

/// Backpressure monitor that periodically samples queue depth.
///
/// Runs as a background task, broadcasting state changes to subscribers.
pub struct BackpressureMonitor {
    /// Observation queue to monitor
    observation_queue: Arc<ObservationQueue>,

    /// Broadcast channel for state changes
    state_tx: broadcast::Sender<BackpressureState>,

    /// Check interval for sampling queue depth
    check_interval: Duration,
}

impl BackpressureMonitor {
    /// Default check interval: 100ms
    ///
    /// Balances responsiveness vs CPU overhead.
    pub const DEFAULT_CHECK_INTERVAL: Duration = Duration::from_millis(100);

    /// Create a new backpressure monitor.
    ///
    /// # Arguments
    ///
    /// * `observation_queue` - Queue to monitor
    /// * `check_interval` - How often to sample queue depth
    ///
    /// # Examples
    ///
    /// ```
    /// use engram_core::streaming::{ObservationQueue, QueueConfig};
    /// use engram_core::streaming::backpressure::BackpressureMonitor;
    /// use std::sync::Arc;
    /// use std::time::Duration;
    ///
    /// let queue = Arc::new(ObservationQueue::new(QueueConfig::default()));
    /// let monitor = BackpressureMonitor::new(queue, Duration::from_millis(100));
    /// ```
    #[must_use]
    pub fn new(observation_queue: Arc<ObservationQueue>, check_interval: Duration) -> Self {
        let (state_tx, _) = broadcast::channel(32);
        Self {
            observation_queue,
            state_tx,
            check_interval,
        }
    }

    /// Subscribe to backpressure state changes.
    ///
    /// Returns a receiver that will get notified whenever the pressure state changes.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use engram_core::streaming::{ObservationQueue, QueueConfig};
    /// # use engram_core::streaming::backpressure::BackpressureMonitor;
    /// # use std::sync::Arc;
    /// # use std::time::Duration;
    /// # #[tokio::main]
    /// # async fn main() {
    /// # let queue = Arc::new(ObservationQueue::new(QueueConfig::default()));
    /// let monitor = BackpressureMonitor::new(queue, Duration::from_millis(100));
    /// let mut rx = monitor.subscribe();
    ///
    /// tokio::spawn(async move { monitor.run().await });
    ///
    /// while let Ok(state) = rx.recv().await {
    ///     println!("Backpressure state: {state:?}");
    /// }
    /// # }
    /// ```
    #[must_use]
    pub fn subscribe(&self) -> broadcast::Receiver<BackpressureState> {
        self.state_tx.subscribe()
    }

    /// Run the backpressure monitoring loop.
    ///
    /// This is a long-running task that should be spawned with `tokio::spawn`.
    /// It will continuously monitor queue depth and broadcast state changes.
    ///
    /// # Cancellation
    ///
    /// This loop runs indefinitely. Cancel by dropping the spawned task handle.
    pub async fn run(&self) {
        let mut current_state = BackpressureState::Normal;
        let mut interval = tokio::time::interval(self.check_interval);

        loop {
            interval.tick().await;

            // Sample queue depth
            let total_depth = self.observation_queue.total_depth();
            let total_capacity = self.observation_queue.total_capacity();

            // Avoid division by zero
            if total_capacity == 0 {
                continue;
            }

            let pressure = total_depth as f32 / total_capacity as f32;
            let new_state = BackpressureState::from_pressure(pressure);

            // Only broadcast on state change
            if new_state != current_state {
                tracing::info!(
                    "Backpressure state: {new_state:?} (pressure: {:.1}%)",
                    pressure * 100.0
                );

                // Record backpressure state metrics
                let state_value = match new_state {
                    BackpressureState::Normal => 0,
                    BackpressureState::Warning => 1,
                    BackpressureState::Critical => 2,
                    BackpressureState::Overloaded => 3,
                };
                let space_id = crate::types::MemorySpaceId::default();
                super::stream_metrics::update_backpressure_state(&space_id, state_value);

                // Record state transition
                let level = match new_state {
                    BackpressureState::Normal => "normal",
                    BackpressureState::Warning => "warning",
                    BackpressureState::Critical => "critical",
                    BackpressureState::Overloaded => "overloaded",
                };
                super::stream_metrics::record_backpressure_activation(&space_id, level);

                // Broadcast to all subscribers (ignore send errors - means no active subscribers)
                let _ = self.state_tx.send(new_state);
                current_state = new_state;
            }
        }
    }
}

/// Calculate retry-after duration based on queue depth and dequeue rate.
///
/// Used for admission control to provide clients with accurate retry guidance.
///
/// # Algorithm
///
/// Time to drain to 50% capacity = (current_depth - target_depth) / dequeue_rate
///
/// # Arguments
///
/// * `queue_depth` - Current total queue depth
/// * `dequeue_rate` - Observations processed per second
///
/// # Returns
///
/// Estimated duration until queue reaches 50% capacity (capped at 5 minutes).
///
/// # Examples
///
/// ```
/// use engram_core::streaming::backpressure::calculate_retry_after;
/// use std::time::Duration;
///
/// // Queue depth: 10000, dequeue rate: 1000 obs/sec
/// // Excess: 5000, time: 5 seconds
/// let retry = calculate_retry_after(10_000, 1000.0);
/// assert_eq!(retry, Duration::from_secs(5));
///
/// // Low dequeue rate - fallback to 60s
/// let retry = calculate_retry_after(10_000, 0.5);
/// assert_eq!(retry, Duration::from_secs(60));
/// ```
#[must_use]
pub fn calculate_retry_after(queue_depth: usize, dequeue_rate: f32) -> Duration {
    if dequeue_rate < 1.0 {
        // Pessimistic fallback for very low dequeue rate
        return Duration::from_secs(60);
    }

    // Calculate time to drain to 50% capacity
    let target_depth = queue_depth / 2;
    let excess = queue_depth.saturating_sub(target_depth);
    let drain_seconds = excess as f32 / dequeue_rate;

    // Cap at 5 minutes to avoid unreasonable waits
    Duration::from_secs_f32(drain_seconds.min(300.0))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::streaming::QueueConfig;

    #[test]
    fn test_backpressure_state_thresholds() {
        assert_eq!(
            BackpressureState::from_pressure(0.0),
            BackpressureState::Normal
        );
        assert_eq!(
            BackpressureState::from_pressure(0.3),
            BackpressureState::Normal
        );
        assert_eq!(
            BackpressureState::from_pressure(0.49),
            BackpressureState::Normal
        );
        assert_eq!(
            BackpressureState::from_pressure(0.5),
            BackpressureState::Warning
        );
        assert_eq!(
            BackpressureState::from_pressure(0.6),
            BackpressureState::Warning
        );
        assert_eq!(
            BackpressureState::from_pressure(0.79),
            BackpressureState::Warning
        );
        assert_eq!(
            BackpressureState::from_pressure(0.8),
            BackpressureState::Critical
        );
        assert_eq!(
            BackpressureState::from_pressure(0.85),
            BackpressureState::Critical
        );
        assert_eq!(
            BackpressureState::from_pressure(0.94),
            BackpressureState::Critical
        );
        assert_eq!(
            BackpressureState::from_pressure(0.95),
            BackpressureState::Overloaded
        );
        assert_eq!(
            BackpressureState::from_pressure(0.98),
            BackpressureState::Overloaded
        );
        assert_eq!(
            BackpressureState::from_pressure(1.0),
            BackpressureState::Overloaded
        );
    }

    #[test]
    fn test_adaptive_batch_sizing() {
        assert_eq!(BackpressureState::Normal.recommended_batch_size(), 10);
        assert_eq!(BackpressureState::Warning.recommended_batch_size(), 100);
        assert_eq!(BackpressureState::Critical.recommended_batch_size(), 500);
        assert_eq!(BackpressureState::Overloaded.recommended_batch_size(), 1000);
    }

    #[test]
    fn test_admission_control_rejection() {
        assert!(!BackpressureState::Normal.should_reject());
        assert!(!BackpressureState::Warning.should_reject());
        assert!(!BackpressureState::Critical.should_reject());
        assert!(BackpressureState::Overloaded.should_reject());
    }

    #[test]
    fn test_retry_after_calculation() {
        // Normal case: 10K queue, 1K obs/sec dequeue rate
        // Target: 5K, excess: 5K, time: 5s
        let retry = calculate_retry_after(10_000, 1000.0);
        assert_eq!(retry, Duration::from_secs(5));

        // Higher dequeue rate - allow for floating point precision
        let retry = calculate_retry_after(10_000, 2000.0);
        assert!(retry >= Duration::from_secs(2) && retry <= Duration::from_secs(3));

        // Low dequeue rate - fallback
        let retry = calculate_retry_after(10_000, 0.5);
        assert_eq!(retry, Duration::from_secs(60));

        // Very high queue depth - capped at 5 minutes
        let retry = calculate_retry_after(1_000_000, 1000.0);
        assert_eq!(retry, Duration::from_secs(300));
    }

    #[tokio::test]
    async fn test_backpressure_monitor_creation() {
        let queue = Arc::new(ObservationQueue::new(QueueConfig::default()));
        let monitor = BackpressureMonitor::new(queue, Duration::from_millis(10));

        // Can subscribe before run()
        let _rx = monitor.subscribe();
    }

    #[tokio::test]
    async fn test_backpressure_monitor_no_panic_empty_queue() {
        let queue = Arc::new(ObservationQueue::new(QueueConfig {
            high_capacity: 0,
            normal_capacity: 0,
            low_capacity: 0,
        }));

        let monitor = BackpressureMonitor::new(Arc::clone(&queue), Duration::from_millis(1));

        // Spawn monitor - should handle zero capacity gracefully
        let handle = tokio::spawn(async move { monitor.run().await });

        // Let it run a few cycles
        tokio::time::sleep(Duration::from_millis(10)).await;

        // Should still be running (not panicked)
        assert!(!handle.is_finished());

        handle.abort();
    }
}
