//! Adaptive batch sizing for parallel activation spreading
//!
//! Provides EWMA-based batch size tuning that adapts to CPU topology and
//! observed spreading latency without requiring hardware performance counters.

use crate::activation::storage_aware::StorageTier;
use atomic_float::AtomicF32;
use crossbeam_queue::ArrayQueue;
use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::Instant;

/// Maximum observations tracked in variance window
/// Ring buffer size for observation history (power of 2 for fast modulo)
const OBSERVATION_RING_SIZE: usize = 256;
const VARIANCE_WINDOW_SIZE: usize = 20;
const LATENCY_EWMA_ALPHA: f64 = 0.25;

/// Topology fingerprint for detecting system changes
#[derive(Debug, Clone, Copy, Hash, Eq, PartialEq)]
pub struct TopologyFingerprint {
    /// Number of logical CPU cores detected
    pub logical_cores: usize,
    /// Number of physical CPU cores (approximate)
    pub physical_cores: usize,
    /// Whether system has heterogeneous cores (P-cores + E-cores)
    pub has_heterogeneous_cores: bool,
    /// Empirically determined memory bandwidth classification
    pub bandwidth_class: BandwidthClass,
}

/// Empirically determined memory bandwidth class
#[derive(Debug, Clone, Copy, Hash, Eq, PartialEq)]
pub enum BandwidthClass {
    /// Low bandwidth: < 10 GB/s
    Low,
    /// Medium bandwidth: 10-30 GB/s
    Medium,
    /// High bandwidth: > 30 GB/s
    High,
}

impl TopologyFingerprint {
    /// Detect system topology characteristics
    #[must_use]
    pub fn detect() -> Self {
        let logical = std::thread::available_parallelism()
            .map(std::num::NonZero::get)
            .unwrap_or(4);

        // Approximate physical cores (simplified heuristic)
        let physical = if logical > 8 { logical / 2 } else { logical };

        // Detect heterogeneous architectures (P-cores + E-cores)
        let has_heterogeneous_cores = logical > physical * 2;

        // Quick bandwidth probe (simplified)
        let bandwidth_class = Self::probe_bandwidth();

        Self {
            logical_cores: logical,
            physical_cores: physical,
            has_heterogeneous_cores,
            bandwidth_class,
        }
    }

    /// Simplified memory bandwidth probe
    fn probe_bandwidth() -> BandwidthClass {
        // Allocate 16MB buffer
        const PROBE_SIZE: usize = 16 * 1024 * 1024;
        let buffer = vec![0u8; PROBE_SIZE];

        // Measure sequential access time
        let start = Instant::now();
        let mut sum = 0u64;

        for chunk in buffer.chunks(64) {
            sum = sum.wrapping_add(u64::from(chunk[0]));
        }

        let elapsed = start.elapsed();
        std::hint::black_box(sum);

        // Rough classification based on access time
        let ns_per_mb = elapsed.as_nanos() / 16;

        if ns_per_mb < 10_000 {
            BandwidthClass::High
        } else if ns_per_mb < 30_000 {
            BandwidthClass::Medium
        } else {
            BandwidthClass::Low
        }
    }

    /// Compute stable hash for change detection
    #[must_use]
    pub fn fingerprint_hash(&self) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        self.hash(&mut hasher);
        hasher.finish()
    }
}

/// Observation of spreading performance for EWMA input
#[derive(Debug, Clone, Copy)]
pub struct Observation {
    /// Batch size used for this spreading operation
    pub batch_size: usize,
    /// Latency observed in nanoseconds
    pub latency_ns: u64,
    /// Number of hops (depth) in the spreading operation
    pub hop_count: usize,
    /// Storage tier where this activation occurred
    pub tier: StorageTier,
}

/// Dual-rate EWMA controller with variance tracking
pub struct EwmaController {
    /// Fast alpha for initial convergence (first N observations)
    fast_alpha: f32,
    /// Slow alpha for steady state
    slow_alpha: f32,
    /// Current EWMA estimate
    current_estimate: AtomicU64,
    /// Number of observations processed
    observation_count: AtomicUsize,
    /// Threshold for switching from fast to slow alpha
    convergence_threshold: usize,
    /// Rolling window for variance calculation
    variance_window: parking_lot::Mutex<VecDeque<f64>>,
}

impl EwmaController {
    /// Create new EWMA controller with dual-rate adaptation
    #[must_use]
    pub fn new(initial_batch_size: usize) -> Self {
        Self {
            fast_alpha: 0.35,
            slow_alpha: 0.20,
            current_estimate: AtomicU64::new(initial_batch_size as u64),
            observation_count: AtomicUsize::new(0),
            convergence_threshold: 10,
            variance_window: parking_lot::Mutex::new(VecDeque::with_capacity(VARIANCE_WINDOW_SIZE)),
        }
    }

    /// Update EWMA with new observation
    pub fn update(&self, observed_batch_size: f64) -> usize {
        let count = self.observation_count.fetch_add(1, Ordering::Relaxed);

        // Choose alpha based on convergence state
        let alpha = if count < self.convergence_threshold {
            self.fast_alpha
        } else {
            self.slow_alpha
        };

        // Compute new EWMA estimate
        let current = self.current_estimate.load(Ordering::Relaxed) as f64;
        #[allow(clippy::cast_possible_truncation)]
        let new_estimate =
            (f64::from(alpha)).mul_add(observed_batch_size, (1.0 - f64::from(alpha)) * current);

        // Update variance window
        {
            let mut window = self.variance_window.lock();
            if window.len() >= VARIANCE_WINDOW_SIZE {
                window.pop_front();
            }
            window.push_back(observed_batch_size);
        }

        // Store new estimate
        let clamped = (new_estimate as usize).clamp(4, 128);
        self.current_estimate
            .store(clamped as u64, Ordering::Relaxed);

        clamped
    }

    /// Get current batch size estimate
    #[must_use]
    pub fn current_estimate(&self) -> usize {
        self.current_estimate.load(Ordering::Relaxed) as usize
    }

    /// Calculate coefficient of variation (σ/μ) for stability assessment
    #[must_use]
    pub fn coefficient_of_variation(&self) -> f32 {
        let window = self.variance_window.lock();
        if window.len() < 2 {
            drop(window);
            return 1.0; // Maximum uncertainty
        }

        let sample_count = window.len() as f64;
        let mean = window.iter().sum::<f64>() / sample_count;
        let variance = window
            .iter()
            .map(|value| {
                let diff = value - mean;
                diff * diff
            })
            .sum::<f64>()
            / sample_count;
        let std_dev = variance.sqrt();
        drop(window);

        if mean < f64::EPSILON {
            1.0
        } else {
            #[allow(clippy::cast_possible_truncation)]
            {
                (std_dev / mean) as f32
            }
        }
    }

    /// Reset EWMA state on topology change
    pub fn reset(&self, initial_batch_size: usize) {
        self.current_estimate
            .store(initial_batch_size as u64, Ordering::Relaxed);
        self.observation_count.store(0, Ordering::Relaxed);
        let mut window = self.variance_window.lock();
        window.clear();
    }
}

/// Configuration for adaptive batching behavior
#[derive(Debug, Clone)]
pub struct AdaptiveBatcherConfig {
    /// Minimum allowed batch size
    pub min_batch_size: usize,
    /// Maximum allowed batch size
    pub max_batch_size: usize,
    /// Number of spreads to wait between batch size adjustments
    pub cooldown_interval: usize,
    /// Minimum delta to trigger batch size change (anti-oscillation)
    pub oscillation_threshold: usize,
    /// Variance/mean ratio threshold for increasing cooldown
    pub instability_threshold: f32,
}

impl Default for AdaptiveBatcherConfig {
    fn default() -> Self {
        Self {
            min_batch_size: 4,
            max_batch_size: 128,
            cooldown_interval: 50,
            oscillation_threshold: 2,
            instability_threshold: 0.30, // 30% coefficient of variation
        }
    }
}

/// Adaptive mode configuration
#[derive(Debug, Clone)]
pub enum AdaptiveMode {
    /// Adaptive batching disabled, use static batch size
    Disabled,
    /// Adaptive with fallback to static on instability
    EnabledWithFallback {
        /// Coefficient of variation threshold triggering fallback (default: 0.30)
        instability_threshold: f32,
        /// Static batch size to use when fallback is triggered
        fallback_batch_size: usize,
    },
    /// Always use adaptive (after validation)
    AlwaysOn,
}

impl Default for AdaptiveMode {
    fn default() -> Self {
        Self::EnabledWithFallback {
            instability_threshold: 0.30,
            fallback_batch_size: 64,
        }
    }
}

/// Per-tier adaptive batch controller
pub struct AdaptiveBatcher {
    /// Topology fingerprint for change detection (reserved for diagnostics)
    #[allow(dead_code)]
    topology: TopologyFingerprint,
    /// Hash of current topology for quick comparison
    topology_hash: u64,
    /// Per-tier EWMA controllers
    tier_controllers: [EwmaController; 3],
    /// Configuration parameters
    config: AdaptiveBatcherConfig,
    /// Adaptive mode setting
    mode: AdaptiveMode,
    /// Spreads since last batch size adjustment
    cooldown_counter: AtomicUsize,
    /// Observation ring buffer (lock-free MPSC)
    observation_queue: Arc<ArrayQueue<Observation>>,
    /// Metrics counters
    metrics: AdaptiveBatcherMetrics,
}

/// Metrics for adaptive batching telemetry
#[derive(Default)]
pub struct AdaptiveBatcherMetrics {
    /// Total batch size updates performed
    pub update_count: AtomicU64,
    /// Guardrail activations (oscillation prevention)
    pub guardrail_hits: AtomicU64,
    /// Topology changes detected
    pub topology_changes: AtomicU64,
    /// Fallback activations due to instability
    pub fallback_activations: AtomicU64,
    /// Smoothed latency (EWMA) derived from recent observations (nanoseconds)
    pub latency_ewma_ns: AtomicU64,
    /// Last recommended batch size for the hot tier
    pub hot_batch_size: AtomicU64,
    /// Last recommended batch size for the warm tier
    pub warm_batch_size: AtomicU64,
    /// Last recommended batch size for the cold tier
    pub cold_batch_size: AtomicU64,
    /// Convergence confidence for the hot tier (0.0–1.0)
    pub hot_confidence: AtomicF32,
    /// Convergence confidence for the warm tier (0.0–1.0)
    pub warm_confidence: AtomicF32,
    /// Convergence confidence for the cold tier (0.0–1.0)
    pub cold_confidence: AtomicF32,
}

impl AdaptiveBatcherMetrics {
    /// Persist the most recent recommendation for a tier so streaming metrics can export it.
    pub fn record_tier_state(&self, tier: StorageTier, batch_size: usize, confidence: f32) {
        let batch_size_u64 = batch_size as u64;
        match tier {
            StorageTier::Hot => {
                self.hot_batch_size.store(batch_size_u64, Ordering::Relaxed);
                self.hot_confidence.store(confidence, Ordering::Relaxed);
            }
            StorageTier::Warm => {
                self.warm_batch_size
                    .store(batch_size_u64, Ordering::Relaxed);
                self.warm_confidence.store(confidence, Ordering::Relaxed);
            }
            StorageTier::Cold => {
                self.cold_batch_size
                    .store(batch_size_u64, Ordering::Relaxed);
                self.cold_confidence.store(confidence, Ordering::Relaxed);
            }
        }
    }

    /// Update the latency EWMA tracked for observability.
    pub fn record_latency_ewma(&self, latency_ns: u64) {
        self.latency_ewma_ns.store(latency_ns, Ordering::Relaxed);
    }
}

/// Telemetry snapshot used by `SpreadingMetrics` to export adaptive batching state.
#[derive(Debug, Clone, Copy, Default)]
pub struct AdaptiveBatcherSnapshot {
    /// Total batch size updates performed
    pub update_count: u64,
    /// Guardrail activations that clamped the recommendation
    pub guardrail_hits: u64,
    /// Topology changes detected since startup
    pub topology_changes: u64,
    /// Fallback activations triggered due to instability
    pub fallback_activations: u64,
    /// Smoothed latency derived from recent observations (nanoseconds)
    pub latency_ewma_ns: u64,
    /// Recommended batch size for the hot tier
    pub hot_batch_size: u64,
    /// Recommended batch size for the warm tier
    pub warm_batch_size: u64,
    /// Recommended batch size for the cold tier
    pub cold_batch_size: u64,
    /// Convergence confidence for the hot tier (0.0–1.0)
    pub hot_confidence: f32,
    /// Convergence confidence for the warm tier (0.0–1.0)
    pub warm_confidence: f32,
    /// Convergence confidence for the cold tier (0.0–1.0)
    pub cold_confidence: f32,
}

impl AdaptiveBatcher {
    /// Create new adaptive batcher
    #[must_use]
    pub fn new(config: AdaptiveBatcherConfig, mode: AdaptiveMode) -> Self {
        let topology = TopologyFingerprint::detect();
        let topology_hash = topology.fingerprint_hash();

        // Initialize per-tier controllers with tier-appropriate defaults
        let tier_controllers = [
            EwmaController::new(config.max_batch_size), // Hot: start high
            EwmaController::new(config.max_batch_size * 3 / 4), // Warm: 75%
            EwmaController::new(config.max_batch_size / 2), // Cold: 50%
        ];

        Self {
            topology,
            topology_hash,
            tier_controllers,
            config,
            mode,
            cooldown_counter: AtomicUsize::new(0),
            observation_queue: Arc::new(ArrayQueue::new(OBSERVATION_RING_SIZE)),
            metrics: AdaptiveBatcherMetrics::default(),
        }
    }

    /// Record a spreading observation for adaptive learning
    pub fn record_observation(&self, observation: Observation) {
        // Non-blocking push to ring buffer
        let _ = self.observation_queue.push(observation);
    }

    /// Process pending observations and update batch size estimates
    pub fn process_observations(&self) {
        let mut latency_ewma = self.metrics.latency_ewma_ns.load(Ordering::Relaxed);
        while let Some(obs) = self.observation_queue.pop() {
            let tier_idx = obs.tier as usize;

            // Calculate normalized batch efficiency
            let latency_per_item = obs.latency_ns as f64 / obs.batch_size as f64;

            // Inverse latency as signal (lower latency → higher batch size)
            let efficiency_signal = 1e9 / latency_per_item.max(1.0);

            // Map efficiency to batch size recommendation
            let recommended_batch = efficiency_signal / 1e6;

            // Update tier-specific EWMA
            self.tier_controllers[tier_idx].update(recommended_batch);
            self.metrics.update_count.fetch_add(1, Ordering::Relaxed);

            // Update latency EWMA with a light smoothing factor
            if latency_ewma == 0 {
                latency_ewma = obs.latency_ns;
            } else {
                let ewma_f64 = (1.0 - LATENCY_EWMA_ALPHA) * latency_ewma as f64
                    + LATENCY_EWMA_ALPHA * obs.latency_ns as f64;
                latency_ewma = ewma_f64.max(1.0).round() as u64;
            }
        }

        if latency_ewma != 0 {
            self.metrics.record_latency_ewma(latency_ewma);
        }

        // Refresh per-tier telemetry so streaming metrics see the latest recommendations.
        for tier in [StorageTier::Hot, StorageTier::Warm, StorageTier::Cold] {
            let controller = &self.tier_controllers[tier as usize];
            let mut recommended = controller.current_estimate();
            recommended = stabilize_to_power_of_two(recommended);
            recommended = match tier {
                StorageTier::Hot => recommended,
                StorageTier::Warm => recommended * 3 / 4,
                StorageTier::Cold => recommended / 2,
            };
            let recommended =
                recommended.clamp(self.config.min_batch_size, self.config.max_batch_size);
            let confidence = self.convergence_confidence(tier);
            self.metrics
                .record_tier_state(tier, recommended, confidence);
        }
    }

    /// Export a read-only snapshot of the adaptive batcher telemetry.
    #[must_use]
    pub fn snapshot(&self) -> AdaptiveBatcherSnapshot {
        let metrics = &self.metrics;
        AdaptiveBatcherSnapshot {
            update_count: metrics.update_count.load(Ordering::Relaxed),
            guardrail_hits: metrics.guardrail_hits.load(Ordering::Relaxed),
            topology_changes: metrics.topology_changes.load(Ordering::Relaxed),
            fallback_activations: metrics.fallback_activations.load(Ordering::Relaxed),
            latency_ewma_ns: metrics.latency_ewma_ns.load(Ordering::Relaxed),
            hot_batch_size: metrics.hot_batch_size.load(Ordering::Relaxed),
            warm_batch_size: metrics.warm_batch_size.load(Ordering::Relaxed),
            cold_batch_size: metrics.cold_batch_size.load(Ordering::Relaxed),
            hot_confidence: metrics.hot_confidence.load(Ordering::Relaxed),
            warm_confidence: metrics.warm_confidence.load(Ordering::Relaxed),
            cold_confidence: metrics.cold_confidence.load(Ordering::Relaxed),
        }
    }

    /// Get recommended batch size for a tier
    #[must_use]
    pub fn recommend_batch_size(&self, tier: StorageTier) -> usize {
        // Check if in cooldown
        let cooldown = self.cooldown_counter.load(Ordering::Relaxed);
        if cooldown < self.config.cooldown_interval {
            // Still in cooldown, use current estimate
            return self.tier_controllers[tier as usize].current_estimate();
        }

        // Check for topology changes
        let current_topology = TopologyFingerprint::detect();
        let current_hash = current_topology.fingerprint_hash();

        if current_hash != self.topology_hash {
            // Topology changed, reset all controllers
            for controller in &self.tier_controllers {
                controller.reset(self.config.max_batch_size);
            }
            self.metrics
                .topology_changes
                .fetch_add(1, Ordering::Relaxed);
        }

        let tier_idx = tier as usize;
        let controller = &self.tier_controllers[tier_idx];

        // Check for instability
        let cv = controller.coefficient_of_variation();
        if cv > self.config.instability_threshold {
            // High variance detected
            if let AdaptiveMode::EnabledWithFallback {
                instability_threshold: _,
                fallback_batch_size,
            } = self.mode
            {
                self.metrics
                    .fallback_activations
                    .fetch_add(1, Ordering::Relaxed);
                self.metrics.guardrail_hits.fetch_add(1, Ordering::Relaxed);
                let fallback = fallback_batch_size
                    .clamp(self.config.min_batch_size, self.config.max_batch_size);
                self.metrics
                    .record_tier_state(tier, fallback, self.convergence_confidence(tier));
                return fallback;
            }
        }

        let mut recommended = controller.current_estimate();

        // Snap to power-of-2 boundaries for cache alignment
        recommended = stabilize_to_power_of_two(recommended);

        // Apply tier-specific adjustments
        recommended = match tier {
            StorageTier::Hot => recommended,          // No adjustment
            StorageTier::Warm => recommended * 3 / 4, // 75%
            StorageTier::Cold => recommended / 2,     // 50%
        };

        // Clamp to configured bounds
        let clamped = recommended.clamp(self.config.min_batch_size, self.config.max_batch_size);

        if clamped == self.config.min_batch_size || clamped == self.config.max_batch_size {
            self.metrics.guardrail_hits.fetch_add(1, Ordering::Relaxed);
        }

        self.metrics
            .record_tier_state(tier, clamped, self.convergence_confidence(tier));

        clamped
    }

    /// Increment cooldown counter after using a batch size
    pub fn mark_batch_used(&self) {
        self.cooldown_counter.fetch_add(1, Ordering::Relaxed);
    }

    /// Reset cooldown counter when batch size changes
    pub fn reset_cooldown(&self) {
        self.cooldown_counter.store(0, Ordering::Relaxed);
    }

    /// Get current metrics snapshot
    #[must_use]
    pub const fn metrics(&self) -> &AdaptiveBatcherMetrics {
        &self.metrics
    }

    /// Get convergence confidence (0.0-1.0)
    #[must_use]
    pub fn convergence_confidence(&self, tier: StorageTier) -> f32 {
        let controller = &self.tier_controllers[tier as usize];
        let updates = controller.observation_count.load(Ordering::Relaxed);

        // Exponential convergence function
        let convergence_window = 5.0;
        #[allow(clippy::cast_precision_loss)]
        let updates_f32 = updates as f32;
        1.0 - (-updates_f32 / convergence_window).exp()
    }
}

/// Snap batch size to nearby power of 2 for cache alignment
const fn stabilize_to_power_of_two(size: usize) -> usize {
    if size < 4 {
        return 4;
    }

    let lower = size.next_power_of_two() / 2;
    let upper = size.next_power_of_two();

    // Snap if within 10% of a power of 2 boundary
    if size - lower < lower / 10 {
        lower
    } else if upper - size < upper / 10 {
        upper
    } else {
        size
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_topology_fingerprint_detection() {
        let topo = TopologyFingerprint::detect();

        assert!(topo.logical_cores > 0);
        assert!(topo.physical_cores > 0);
        assert!(topo.physical_cores <= topo.logical_cores);

        let hash1 = topo.fingerprint_hash();
        let hash2 = topo.fingerprint_hash();
        assert_eq!(hash1, hash2, "Hash should be deterministic");
    }

    #[test]
    fn test_ewma_controller_convergence() {
        let controller = EwmaController::new(64);

        // Feed stable observations
        for _ in 0..20 {
            controller.update(64.0);
        }

        let estimate = controller.current_estimate();
        assert!((estimate as i32 - 64).abs() <= 2, "Should converge to 64");

        let cv = controller.coefficient_of_variation();
        assert!(cv < 0.1, "Should have low variance: {cv}");
    }

    #[test]
    fn test_ewma_controller_adaptation() {
        let controller = EwmaController::new(32);

        // Feed increasing observations
        for i in 0..10 {
            controller.update(32.0 + f64::from(i) * 4.0);
        }

        let estimate = controller.current_estimate();
        assert!(estimate > 32, "Should adapt upward");
        assert!(estimate < 72, "Should not overshoot");
    }

    #[test]
    fn test_stabilize_to_power_of_two() {
        assert_eq!(stabilize_to_power_of_two(63), 64);
        assert_eq!(stabilize_to_power_of_two(65), 64);
        assert_eq!(stabilize_to_power_of_two(50), 50);
        assert_eq!(stabilize_to_power_of_two(31), 32);
        assert_eq!(stabilize_to_power_of_two(33), 32);
    }

    #[test]
    fn test_adaptive_batcher_per_tier_sizing() {
        let config = AdaptiveBatcherConfig::default();
        let batcher = AdaptiveBatcher::new(config, AdaptiveMode::AlwaysOn);

        let hot_size = batcher.recommend_batch_size(StorageTier::Hot);
        let warm_size = batcher.recommend_batch_size(StorageTier::Warm);
        let cold_size = batcher.recommend_batch_size(StorageTier::Cold);

        // Tiers should have decreasing batch sizes
        assert!(hot_size >= warm_size);
        assert!(warm_size >= cold_size);
    }

    #[test]
    fn test_adaptive_batcher_fallback_on_instability() {
        let config = AdaptiveBatcherConfig {
            instability_threshold: 0.1, // Very low threshold
            ..Default::default()
        };

        let mode = AdaptiveMode::EnabledWithFallback {
            instability_threshold: 0.1,
            fallback_batch_size: 42,
        };

        let batcher = AdaptiveBatcher::new(config, mode);

        // Feed highly variable observations to trigger instability
        for i in 0..20 {
            let obs = Observation {
                batch_size: if i % 2 == 0 { 10 } else { 100 },
                latency_ns: 1_000_000,
                hop_count: 2,
                tier: StorageTier::Hot,
            };
            batcher.record_observation(obs);
        }

        batcher.process_observations();

        // Should fall back due to high variance
        let recommended = batcher.recommend_batch_size(StorageTier::Hot);

        // Might hit fallback or clamped bounds
        assert!((4..=128).contains(&recommended));
    }

    #[test]
    fn test_cooldown_prevents_thrashing() {
        let config = AdaptiveBatcherConfig {
            cooldown_interval: 10,
            ..Default::default()
        };

        let batcher = AdaptiveBatcher::new(config, AdaptiveMode::AlwaysOn);

        let initial = batcher.recommend_batch_size(StorageTier::Hot);

        // Mark several batches used
        for _ in 0..5 {
            batcher.mark_batch_used();
        }

        let during_cooldown = batcher.recommend_batch_size(StorageTier::Hot);

        // Should return same value during cooldown
        assert_eq!(initial, during_cooldown);
    }
}
