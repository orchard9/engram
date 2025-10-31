//! Fault injection utilities for chaos testing streaming memory operations.
//!
//! This module provides tools to inject various failure modes into the streaming
//! pipeline to validate correctness under adverse conditions.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::time::sleep;

/// Network delay injector for simulating variable latency.
///
/// Injects random delays between min and max milliseconds to test temporal
/// ordering guarantees under network latency.
pub struct DelayInjector {
    min_delay_ms: u64,
    max_delay_ms: u64,
    rng: Arc<Mutex<StdRng>>,
}

impl DelayInjector {
    /// Create a new delay injector with specified range.
    ///
    /// # Arguments
    /// * `min_delay_ms` - Minimum delay in milliseconds
    /// * `max_delay_ms` - Maximum delay in milliseconds
    /// * `seed` - RNG seed for reproducibility
    #[must_use]
    pub fn new(min_delay_ms: u64, max_delay_ms: u64, seed: u64) -> Self {
        Self {
            min_delay_ms,
            max_delay_ms,
            rng: Arc::new(Mutex::new(StdRng::seed_from_u64(seed))),
        }
    }

    /// Inject a random delay within configured range.
    ///
    /// This is an async function that suspends execution for a random duration.
    pub async fn inject_delay(&self) {
        let delay_ms = {
            let mut rng = self.rng.lock().unwrap();
            rng.gen_range(self.min_delay_ms..=self.max_delay_ms)
        };
        sleep(Duration::from_millis(delay_ms)).await;
    }

    /// Get the configured delay range.
    #[must_use]
    pub fn delay_range(&self) -> (u64, u64) {
        (self.min_delay_ms, self.max_delay_ms)
    }
}

/// Packet loss simulator for testing retry logic.
///
/// Randomly drops operations with specified probability to validate that
/// clients properly retry and eventual consistency is maintained.
pub struct PacketLossSimulator {
    drop_rate: f64,
    rng: Arc<Mutex<StdRng>>,
    drops_total: Arc<std::sync::atomic::AtomicU64>,
    attempts_total: Arc<std::sync::atomic::AtomicU64>,
}

impl PacketLossSimulator {
    /// Create a new packet loss simulator.
    ///
    /// # Arguments
    /// * `drop_rate` - Probability of dropping (0.0 = never, 1.0 = always, 0.01 = 1%)
    /// * `seed` - RNG seed for reproducibility
    #[must_use]
    pub fn new(drop_rate: f64, seed: u64) -> Self {
        assert!((0.0..=1.0).contains(&drop_rate), "drop_rate must be in [0.0, 1.0]");
        Self {
            drop_rate,
            rng: Arc::new(Mutex::new(StdRng::seed_from_u64(seed))),
            drops_total: Arc::new(std::sync::atomic::AtomicU64::new(0)),
            attempts_total: Arc::new(std::sync::atomic::AtomicU64::new(0)),
        }
    }

    /// Determine if this operation should be dropped.
    ///
    /// Returns `true` if the operation should be dropped (packet loss),
    /// `false` if it should proceed normally.
    pub fn should_drop(&self) -> bool {
        self.attempts_total.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        let drop = {
            let mut rng = self.rng.lock().unwrap();
            rng.gen_bool(self.drop_rate)
        };

        if drop {
            self.drops_total.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }

        drop
    }

    /// Get statistics about packet loss.
    #[must_use]
    pub fn stats(&self) -> PacketLossStats {
        let drops = self.drops_total.load(std::sync::atomic::Ordering::Relaxed);
        let attempts = self.attempts_total.load(std::sync::atomic::Ordering::Relaxed);

        PacketLossStats {
            drops_total: drops,
            attempts_total: attempts,
            effective_drop_rate: if attempts > 0 {
                drops as f64 / attempts as f64
            } else {
                0.0
            },
        }
    }

    /// Reset statistics counters.
    pub fn reset_stats(&self) {
        self.drops_total.store(0, std::sync::atomic::Ordering::Relaxed);
        self.attempts_total.store(0, std::sync::atomic::Ordering::Relaxed);
    }
}

/// Statistics from packet loss simulation.
#[derive(Debug, Clone, Copy)]
pub struct PacketLossStats {
    /// Total number of drops
    pub drops_total: u64,
    /// Total number of attempts
    pub attempts_total: u64,
    /// Effective drop rate observed
    pub effective_drop_rate: f64,
}

/// Clock skew simulator for testing timestamp handling.
///
/// Injects time offsets to simulate clock drift, NTP corrections, and
/// other temporal anomalies.
pub struct ClockSkewSimulator {
    offset_ms: Arc<std::sync::atomic::AtomicI64>,
}

impl ClockSkewSimulator {
    /// Create a new clock skew simulator.
    #[must_use]
    pub fn new() -> Self {
        Self {
            offset_ms: Arc::new(std::sync::atomic::AtomicI64::new(0)),
        }
    }

    /// Inject a time offset in milliseconds.
    ///
    /// Positive values simulate clock drift forward, negative values simulate
    /// clock drift backward.
    pub fn inject_skew(&self, offset_ms: i64) {
        self.offset_ms.store(offset_ms, std::sync::atomic::Ordering::SeqCst);
    }

    /// Get current simulated time with offset applied.
    #[must_use]
    pub fn now(&self) -> std::time::Instant {
        let offset_ms = self.offset_ms.load(std::sync::atomic::Ordering::SeqCst);
        let base = std::time::Instant::now();

        if offset_ms >= 0 {
            base + Duration::from_millis(offset_ms.unsigned_abs())
        } else {
            // For negative offsets, we can't actually go backward in time with Instant,
            // so we just return the base time. In real usage, this would be tracked
            // separately in timestamps.
            base
        }
    }

    /// Get current offset in milliseconds.
    #[must_use]
    pub fn current_offset_ms(&self) -> i64 {
        self.offset_ms.load(std::sync::atomic::Ordering::SeqCst)
    }

    /// Reset clock skew to zero.
    pub fn reset(&self) {
        self.offset_ms.store(0, std::sync::atomic::Ordering::SeqCst);
    }
}

impl Default for ClockSkewSimulator {
    fn default() -> Self {
        Self::new()
    }
}

/// Burst load generator for testing queue overflow and admission control.
///
/// Generates controlled bursts of load to trigger backpressure and validate
/// the system's response to overload conditions.
pub struct BurstLoadGenerator {
    burst_size: usize,
    burst_interval: Duration,
    active: Arc<std::sync::atomic::AtomicBool>,
}

impl BurstLoadGenerator {
    /// Create a new burst load generator.
    ///
    /// # Arguments
    /// * `burst_size` - Number of items per burst
    /// * `burst_interval` - Time between bursts
    #[must_use]
    pub fn new(burst_size: usize, burst_interval: Duration) -> Self {
        Self {
            burst_size,
            burst_interval,
            active: Arc::new(std::sync::atomic::AtomicBool::new(false)),
        }
    }

    /// Start the burst generator.
    pub fn start(&self) {
        self.active.store(true, std::sync::atomic::Ordering::SeqCst);
    }

    /// Stop the burst generator.
    pub fn stop(&self) {
        self.active.store(false, std::sync::atomic::Ordering::SeqCst);
    }

    /// Check if generator is active.
    #[must_use]
    pub fn is_active(&self) -> bool {
        self.active.load(std::sync::atomic::Ordering::SeqCst)
    }

    /// Get burst configuration.
    #[must_use]
    pub fn config(&self) -> (usize, Duration) {
        (self.burst_size, self.burst_interval)
    }
}

/// Chaos scenario combining multiple fault types.
///
/// Coordinates multiple fault injectors to create complex failure scenarios.
pub struct ChaosScenario {
    delay_injector: Option<DelayInjector>,
    packet_loss: Option<PacketLossSimulator>,
    clock_skew: Option<ClockSkewSimulator>,
    burst_load: Option<BurstLoadGenerator>,
}

impl ChaosScenario {
    /// Create a new chaos scenario builder.
    #[must_use]
    pub fn builder() -> ChaosScenarioBuilder {
        ChaosScenarioBuilder::default()
    }

    /// Apply all enabled fault injections.
    pub async fn apply_faults(&self) {
        // Apply delay if configured
        if let Some(ref injector) = self.delay_injector {
            injector.inject_delay().await;
        }
    }

    /// Check if packet should be dropped.
    #[must_use]
    pub fn should_drop_packet(&self) -> bool {
        self.packet_loss.as_ref().map_or(false, |sim| sim.should_drop())
    }

    /// Get current clock skew offset.
    #[must_use]
    pub fn clock_offset_ms(&self) -> i64 {
        self.clock_skew.as_ref().map_or(0, |sim| sim.current_offset_ms())
    }

    /// Check if burst load is active.
    #[must_use]
    pub fn is_burst_active(&self) -> bool {
        self.burst_load.as_ref().map_or(false, |gen| gen.is_active())
    }
}

/// Builder for constructing chaos scenarios.
#[derive(Default)]
pub struct ChaosScenarioBuilder {
    delay_injector: Option<DelayInjector>,
    packet_loss: Option<PacketLossSimulator>,
    clock_skew: Option<ClockSkewSimulator>,
    burst_load: Option<BurstLoadGenerator>,
}

impl ChaosScenarioBuilder {
    /// Add network delay injection.
    #[must_use]
    pub fn with_delay(mut self, min_ms: u64, max_ms: u64, seed: u64) -> Self {
        self.delay_injector = Some(DelayInjector::new(min_ms, max_ms, seed));
        self
    }

    /// Add packet loss simulation.
    #[must_use]
    pub fn with_packet_loss(mut self, drop_rate: f64, seed: u64) -> Self {
        self.packet_loss = Some(PacketLossSimulator::new(drop_rate, seed));
        self
    }

    /// Add clock skew simulation.
    #[must_use]
    pub fn with_clock_skew(mut self) -> Self {
        self.clock_skew = Some(ClockSkewSimulator::new());
        self
    }

    /// Add burst load generation.
    #[must_use]
    pub fn with_burst_load(mut self, burst_size: usize, interval: Duration) -> Self {
        self.burst_load = Some(BurstLoadGenerator::new(burst_size, interval));
        self
    }

    /// Build the chaos scenario.
    #[must_use]
    pub fn build(self) -> ChaosScenario {
        ChaosScenario {
            delay_injector: self.delay_injector,
            packet_loss: self.packet_loss,
            clock_skew: self.clock_skew,
            burst_load: self.burst_load,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn delay_injector_range() {
        let injector = DelayInjector::new(10, 100, 42);
        assert_eq!(injector.delay_range(), (10, 100));
    }

    #[test]
    fn packet_loss_statistics() {
        let simulator = PacketLossSimulator::new(0.5, 42);

        // Simulate 100 attempts
        for _ in 0..100 {
            let _ = simulator.should_drop();
        }

        let stats = simulator.stats();
        assert_eq!(stats.attempts_total, 100);
        assert!(stats.drops_total > 0);
        assert!(stats.effective_drop_rate > 0.0 && stats.effective_drop_rate < 1.0);
    }

    #[test]
    fn clock_skew_offset() {
        let simulator = ClockSkewSimulator::new();
        assert_eq!(simulator.current_offset_ms(), 0);

        simulator.inject_skew(5000);
        assert_eq!(simulator.current_offset_ms(), 5000);

        simulator.inject_skew(-3000);
        assert_eq!(simulator.current_offset_ms(), -3000);

        simulator.reset();
        assert_eq!(simulator.current_offset_ms(), 0);
    }

    #[test]
    fn burst_load_control() {
        let generator = BurstLoadGenerator::new(1000, Duration::from_secs(5));
        assert!(!generator.is_active());

        generator.start();
        assert!(generator.is_active());

        generator.stop();
        assert!(!generator.is_active());
    }

    #[test]
    fn chaos_scenario_builder() {
        let scenario = ChaosScenario::builder()
            .with_delay(0, 100, 42)
            .with_packet_loss(0.01, 43)
            .with_clock_skew()
            .with_burst_load(10_000, Duration::from_secs(5))
            .build();

        assert!(scenario.delay_injector.is_some());
        assert!(scenario.packet_loss.is_some());
        assert!(scenario.clock_skew.is_some());
        assert!(scenario.burst_load.is_some());
    }
}
