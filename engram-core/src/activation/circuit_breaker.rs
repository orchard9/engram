use super::SpreadingMetrics;
use parking_lot::Mutex;
use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::atomic::{AtomicU8, Ordering};
use std::time::{Duration, Instant};
use tracing::{info, warn};

const STATE_CLOSED: u8 = 0;
const STATE_HALF_OPEN: u8 = 1;
const STATE_OPEN: u8 = 2;

/// Current state of the spreading circuit breaker
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BreakerState {
    /// Normal operation, spreading requests are allowed
    Closed,
    /// Recovery mode, probing for service restoration
    HalfOpen,
    /// Failure mode, spreading requests are rejected
    Open,
}

impl From<u8> for BreakerState {
    fn from(value: u8) -> Self {
        match value {
            STATE_HALF_OPEN => Self::HalfOpen,
            STATE_OPEN => Self::Open,
            _ => Self::Closed,
        }
    }
}

impl From<BreakerState> for u8 {
    fn from(value: BreakerState) -> Self {
        match value {
            BreakerState::Closed => STATE_CLOSED,
            BreakerState::HalfOpen => STATE_HALF_OPEN,
            BreakerState::Open => STATE_OPEN,
        }
    }
}

/// Configuration for circuit breaker behavior
#[derive(Clone, Debug)]
pub struct BreakerSettings {
    /// Fraction of failures that triggers breaker opening (0.0 to 1.0)
    pub failure_rate_threshold: f32,
    /// Number of recent samples used for failure rate calculation
    pub sample_window: usize,
    /// Latency budget multiplier that triggers breaker opening
    pub latency_multiplier: f64,
    /// Duration to wait before transitioning from Open to HalfOpen
    pub cooldown: Duration,
    /// Number of consecutive successes required to close the breaker from HalfOpen
    pub half_open_probe_count: usize,
}

impl Default for BreakerSettings {
    fn default() -> Self {
        Self {
            failure_rate_threshold: 0.05,
            sample_window: 50,
            latency_multiplier: 1.5,
            cooldown: Duration::from_secs(30),
            half_open_probe_count: 5,
        }
    }
}

struct BreakerSample {
    success: bool,
    #[allow(dead_code)]
    latency: Duration,
}

struct BreakerInner {
    samples: VecDeque<BreakerSample>,
    last_state_change: Instant,
    probes_remaining: usize,
}

/// Circuit breaker for protecting the spreading engine from cascading failures
pub struct SpreadingCircuitBreaker {
    metrics: Arc<SpreadingMetrics>,
    settings: BreakerSettings,
    state: AtomicU8,
    inner: Mutex<BreakerInner>,
}

impl SpreadingCircuitBreaker {
    /// Create a new circuit breaker with the given metrics handle and settings
    pub fn new(metrics: Arc<SpreadingMetrics>, settings: &BreakerSettings) -> Self {
        let initial_state = BreakerState::Closed;
        metrics.record_breaker_state(initial_state as u64);
        Self {
            metrics,
            settings: settings.clone(),
            state: AtomicU8::new(initial_state.into()),
            inner: Mutex::new(BreakerInner {
                samples: VecDeque::with_capacity(settings.sample_window),
                last_state_change: Instant::now(),
                probes_remaining: settings.half_open_probe_count,
            }),
        }
    }

    /// Check if a spreading operation should be attempted based on current breaker state
    pub fn should_attempt(&self) -> bool {
        let mut guard = self.inner.lock();
        match BreakerState::from(self.state.load(Ordering::Acquire)) {
            BreakerState::Closed | BreakerState::HalfOpen => true,
            BreakerState::Open => {
                if guard.last_state_change.elapsed() >= self.settings.cooldown {
                    self.transition(&mut guard, BreakerState::HalfOpen);
                    true
                } else {
                    false
                }
            }
        }
    }

    /// Record the result of a spreading operation to update breaker state
    pub fn on_result(&self, success: bool, latency: Duration, budget: Duration) {
        let mut guard = self.inner.lock();

        if guard.samples.len() >= self.settings.sample_window {
            guard.samples.pop_front();
        }
        guard.samples.push_back(BreakerSample { success, latency });

        let current_state = BreakerState::from(self.state.load(Ordering::Acquire));

        if success {
            if matches!(current_state, BreakerState::HalfOpen) {
                guard.probes_remaining = guard.probes_remaining.saturating_sub(1);
                if guard.probes_remaining == 0 {
                    info!("Circuit breaker closed after successful probes");
                    self.transition(&mut guard, BreakerState::Closed);
                }
            } else {
                // Reset probe counter for steady state operations.
                guard.probes_remaining = self.settings.half_open_probe_count;
            }
            return;
        }

        // Failure handling
        let failure_rate = failure_rate(&guard.samples);
        drop(guard);
        let latency_limit = budget.mul_f64(self.settings.latency_multiplier);
        let latency_breach = latency > latency_limit;

        if matches!(current_state, BreakerState::HalfOpen) {
            warn!("Circuit breaker reopened after half-open failure");
            let mut guard = self.inner.lock();
            self.transition(&mut guard, BreakerState::Open);
            return;
        }

        if failure_rate >= self.settings.failure_rate_threshold || latency_breach {
            warn!(
                failure_rate,
                latency_seconds = latency.as_secs_f64(),
                latency_limit_seconds = latency_limit.as_secs_f64(),
                "Circuit breaker opening due to failure rate/latency"
            );
            let mut guard = self.inner.lock();
            self.transition(&mut guard, BreakerState::Open);
        }
    }

    /// Get the current state of the circuit breaker
    pub fn state(&self) -> BreakerState {
        BreakerState::from(self.state.load(Ordering::Acquire))
    }

    fn transition(&self, inner: &mut BreakerInner, new_state: BreakerState) {
        let previous = BreakerState::from(self.state.swap(new_state.into(), Ordering::AcqRel));
        inner.last_state_change = Instant::now();
        if matches!(new_state, BreakerState::HalfOpen) {
            inner.probes_remaining = self.settings.half_open_probe_count;
        }
        self.metrics.record_breaker_transition(new_state as u64);
        self.metrics.record_breaker_state(new_state as u64);
        if new_state != previous {
            info!(?previous, ?new_state, "Circuit breaker transitioned");
        }
    }
}

fn failure_rate(samples: &VecDeque<BreakerSample>) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let failures = samples.iter().filter(|sample| !sample.success).count();
    failures as f32 / samples.len() as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_settings() -> BreakerSettings {
        BreakerSettings {
            failure_rate_threshold: 0.25,
            sample_window: 8,
            latency_multiplier: 1.1,
            cooldown: Duration::from_millis(0),
            half_open_probe_count: 2,
        }
    }

    #[test]
    fn breaker_cycles_through_states() {
        let metrics = Arc::new(SpreadingMetrics::default());
        let breaker = SpreadingCircuitBreaker::new(metrics, &test_settings());
        assert_eq!(breaker.state(), BreakerState::Closed);

        let budget = Duration::from_millis(5);
        let slow = Duration::from_millis(8);

        // Feed enough failures to trip the breaker.
        for _ in 0..6 {
            breaker.on_result(false, slow, budget);
        }
        assert_eq!(breaker.state(), BreakerState::Open);

        // Cooldown is zero, so the next attempt should transition to HalfOpen.
        assert!(breaker.should_attempt());
        assert_eq!(breaker.state(), BreakerState::HalfOpen);

        // First probe succeeds but requires two successes to fully close.
        breaker.on_result(true, Duration::from_millis(1), budget);
        assert_eq!(breaker.state(), BreakerState::HalfOpen);

        breaker.on_result(true, Duration::from_millis(1), budget);
        assert_eq!(breaker.state(), BreakerState::Closed);

        // Another failure reopens the breaker immediately.
        breaker.on_result(false, slow, budget);
        assert_eq!(breaker.state(), BreakerState::Open);
    }
}
