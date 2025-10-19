//! Parallel Activation Spreading with Cognitive Architecture
//!
//! This module implements high-performance lock-free parallel activation spreading
//! that replaces the simple temporal-proximity spreading with sophisticated
//! work-stealing graph traversal engine following biological principles.

use atomic_float::AtomicF32;
use dashmap::DashMap;
use serde::Serialize;
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

const METRIC_SPREADING_LATENCY_HOT: &str = "engram_spreading_latency_hot_seconds";
const METRIC_SPREADING_LATENCY_WARM: &str = "engram_spreading_latency_warm_seconds";
const METRIC_SPREADING_LATENCY_COLD: &str = "engram_spreading_latency_cold_seconds";
const METRIC_SPREADING_ACTIVATIONS_TOTAL: &str = "engram_spreading_activations_total";
const METRIC_SPREADING_LATENCY_BUDGET_VIOLATIONS_TOTAL: &str =
    "engram_spreading_latency_budget_violations_total";
const METRIC_SPREADING_FALLBACK_TOTAL: &str = "engram_spreading_fallback_total";
const METRIC_SPREADING_FAILURES_TOTAL: &str = "engram_spreading_failures_total";
const METRIC_SPREADING_BREAKER_STATE: &str = "engram_spreading_breaker_state";
const METRIC_SPREADING_BREAKER_TRANSITIONS_TOTAL: &str =
    "engram_spreading_breaker_transitions_total";
const METRIC_SPREADING_GPU_LAUNCH_TOTAL: &str = "engram_spreading_gpu_launch_total";
const METRIC_SPREADING_GPU_FALLBACK_TOTAL: &str = "engram_spreading_gpu_fallback_total";
const METRIC_SPREADING_POOL_UTILIZATION: &str = "engram_spreading_pool_utilization";
const METRIC_SPREADING_POOL_HIT_RATE: &str = "engram_spreading_pool_hit_rate";
const METRIC_SPREADING_AUTOTUNE_CHANGES_TOTAL: &str = "engram_spreading_autotune_changes_total";
const METRIC_SPREADING_AUTOTUNE_LAST_IMPROVEMENT: &str =
    "engram_spreading_autotune_last_improvement";

/// Latency budget management for activation spreading
pub mod latency_budget;
/// Storage-aware activation spreading algorithms
pub mod storage_aware;

pub mod accumulator;
/// Adaptive batch sizing for spreading optimization
pub mod adaptive_batcher;
/// Auto-tuning utilities for spreading configuration
pub mod auto_tuning;
/// Circuit breaker for spreading resilience
pub mod circuit_breaker;
pub mod confidence_aggregation;
pub mod cycle_detector;
/// GPU abstraction layer for acceleration
pub mod gpu_interface;
/// Health monitoring probes for activation subsystems
pub mod health_checks;
/// Multi-cue activation spreading support
pub mod multi_cue;
/// Lock-free parallel activation spreading engine
pub mod parallel;
pub mod queue;
/// Integrated cognitive recall pipeline
pub mod recall;
pub mod scheduler;
#[cfg(feature = "hnsw_index")]
/// Vector similarity-based activation seeding
pub mod seeding;
#[cfg(feature = "hnsw_index")]
/// Semantic search integration with spreading activation
pub mod semantic_seeder;
/// SIMD-optimized batch activation spreading
pub mod simd_optimization;
/// Configuration for similarity-based activation
pub mod similarity_config;
pub mod test_support;
pub mod traversal;

pub mod visualization;

pub use auto_tuning::{AutoTuneAuditEntry, SpreadingAutoTuner};
pub use circuit_breaker::{BreakerSettings, BreakerState, SpreadingCircuitBreaker};
pub use confidence_aggregation::{
    ConfidenceAggregationOutcome, ConfidenceAggregator, ConfidenceContribution, ConfidencePath,
};
pub use gpu_interface::{
    AdaptiveConfig, AdaptiveSpreadingEngine, BatchMetadata, CpuFallback, GPUActivationBatch,
    GPUSpreadingInterface, GpuCapabilities, GpuLaunchFuture, MockGpuInterface,
};
pub use health_checks::SpreadingHealthProbe;
pub use multi_cue::CueAggregationStrategy;
pub use parallel::ParallelSpreadingEngine;
pub use recall::{
    CognitiveRecall, CognitiveRecallBuilder, RankedMemory, RecallConfig, RecallMetrics, RecallMode,
};
pub use scheduler::{SchedulerSnapshot, TierAwareSpreadingScheduler, TierQueueStateSnapshot};
#[cfg(feature = "hnsw_index")]
pub use seeding::{
    ActivationTier, SeededActivation, SeedingError, SeedingOutcome, VectorActivationSeeder,
};
#[cfg(feature = "hnsw_index")]
pub use semantic_seeder::{
    ActivationSource, FigurativeInterpreter, SemanticActivationSeeder, SemanticError,
};
pub use similarity_config::SimilarityConfig;
pub mod memory_pool;
pub use adaptive_batcher::{
    AdaptiveBatcher, AdaptiveBatcherConfig, AdaptiveBatcherMetrics, AdaptiveBatcherSnapshot,
    AdaptiveMode, BandwidthClass, Observation, TopologyFingerprint,
};
pub use memory_pool::{
    ActivationMemoryPool, ActivationRecordPool, ActivationRecordPoolStats, LocalMemoryPool,
    PoolStats,
};

// HNSW integration
#[cfg(feature = "hnsw_index")]
pub mod hnsw_integration;
#[cfg(feature = "hnsw_index")]
pub use hnsw_integration::{
    HierarchicalActivation, HnswActivationEngine, SpreadingConfig as HnswSpreadingConfig,
};

/// Node identifier for graph traversal
pub type NodeId = String;

/// Cache-aligned hot fields reused across activation records.
#[repr(C, align(64))]
#[derive(Debug)]
pub struct CacheOptimizedNode {
    activation: AtomicF32,
    confidence: AtomicF32,
    visits: AtomicUsize,
    source_count: AtomicUsize,
}

impl Default for CacheOptimizedNode {
    fn default() -> Self {
        Self::new()
    }
}

impl CacheOptimizedNode {
    /// Create a cache-aligned hot field container with zeroed state.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            activation: AtomicF32::new(0.0),
            confidence: AtomicF32::new(0.0),
            visits: AtomicUsize::new(0),
            source_count: AtomicUsize::new(0),
        }
    }

    /// Reset hot fields to their initial state.
    pub fn reset(&self) {
        self.activation.store(0.0, Ordering::Relaxed);
        self.confidence.store(0.0, Ordering::Relaxed);
        self.visits.store(0, Ordering::Relaxed);
        self.source_count.store(0, Ordering::Relaxed);
    }

    /// Access the activation atomic for this node.
    #[must_use]
    pub const fn activation(&self) -> &AtomicF32 {
        &self.activation
    }

    /// Access the confidence atomic for this node.
    #[must_use]
    pub const fn confidence(&self) -> &AtomicF32 {
        &self.confidence
    }

    /// Access the visit counter for this node.
    #[must_use]
    pub const fn visits(&self) -> &AtomicUsize {
        &self.visits
    }

    /// Access the source count atomic for this node.
    #[must_use]
    pub const fn source_count(&self) -> &AtomicUsize {
        &self.source_count
    }
}

/// Lock-free activation record that shares cache-optimized hot fields.
#[repr(align(64))]
pub struct ActivationRecord {
    /// Unique identifier for this node
    pub node_id: NodeId,
    hot: Arc<CacheOptimizedNode>,
    /// Last update timestamp for ordering
    pub timestamp: AtomicU64,
    /// Node-specific decay coefficient
    pub decay_rate: f32,
    /// Optional storage tier classification for this activation.
    pub storage_tier: Option<storage_aware::StorageTier>,
}

impl ActivationRecord {
    /// Create a new activation record with zero initial activation
    #[must_use]
    pub fn new(node_id: NodeId, decay_rate: f32) -> Self {
        Self {
            node_id,
            hot: Arc::new(CacheOptimizedNode::new()),
            timestamp: AtomicU64::new(0),
            decay_rate,
            storage_tier: None,
        }
    }

    /// Borrow the cache-optimized hot fields for external consumers.
    #[must_use]
    pub fn hot_fields(&self) -> &CacheOptimizedNode {
        &self.hot
    }

    /// Clone the shared handle to the hot fields for caching elsewhere.
    #[must_use]
    pub fn hot_handle(&self) -> Arc<CacheOptimizedNode> {
        Arc::clone(&self.hot)
    }

    /// Atomically accumulate activation with memory ordering, returning both the
    /// updated activation level and the delta actually applied.
    pub fn accumulate_activation_with_result(&self, contribution: f32) -> Option<(f32, f32)> {
        let threshold = self
            .storage_tier
            .map_or(0.01, storage_aware::StorageTier::activation_threshold);
        let activation = self.hot.activation();
        loop {
            let current = activation.load(Ordering::Relaxed);
            let new_activation = (current + contribution).min(1.0);

            // Only update if above threshold to reduce contention
            if new_activation < threshold {
                return None;
            }

            if activation
                .compare_exchange_weak(
                    current,
                    new_activation,
                    Ordering::Relaxed,
                    Ordering::Relaxed,
                )
                .is_ok()
            {
                self.hot
                    .confidence()
                    .store(new_activation, Ordering::Relaxed);
                // Update timestamp for cycle detection
                let now = Instant::now()
                    .elapsed()
                    .as_nanos()
                    .try_into()
                    .unwrap_or(u64::MAX);
                self.timestamp.store(now, Ordering::Relaxed);
                let applied_delta = new_activation - current;
                return Some((new_activation, applied_delta));
            }
        }
    }

    /// Atomically accumulate activation with memory ordering
    pub fn accumulate_activation(&self, contribution: f32) -> bool {
        self.accumulate_activation_with_result(contribution)
            .is_some()
    }

    /// Get current activation level
    pub fn get_activation(&self) -> f32 {
        self.hot.activation().load(Ordering::Relaxed)
    }

    /// Reset activation and counters to zero
    pub fn reset(&self) {
        self.hot.reset();
        self.timestamp.store(0, Ordering::Relaxed);
    }

    /// Reinitialise the record for a new node before reuse.
    pub fn reinitialize(
        &mut self,
        node_id: NodeId,
        decay_rate: f32,
        storage_tier: Option<storage_aware::StorageTier>,
    ) {
        self.node_id = node_id;
        self.decay_rate = decay_rate;
        self.storage_tier = storage_tier;
        self.reset();
    }

    /// Prepare the record for returning to the pool.
    pub fn prepare_for_pool(&mut self) {
        self.reset();
        self.node_id.clear();
        self.storage_tier = None;
    }

    /// Set activation level directly
    pub fn set_activation(&self, value: f32) {
        let clamped = value.clamp(0.0, 1.0);
        self.hot.activation().store(clamped, Ordering::Relaxed);
        self.hot.confidence().store(clamped, Ordering::Relaxed);
    }

    /// Retrieve the current confidence score associated with the record.
    pub fn get_confidence(&self) -> f32 {
        self.hot.confidence().load(Ordering::Relaxed)
    }

    /// Apply a multiplicative penalty to activation and confidence when a cycle is detected.
    pub fn apply_cycle_penalty(&self, factor: f32) {
        let penalty = factor.clamp(0.0, 1.0);
        let activation = self.hot.activation();

        // Scale activation
        loop {
            let current = activation.load(Ordering::Relaxed);
            let updated = (current * penalty).clamp(0.0, 1.0);
            if activation
                .compare_exchange_weak(current, updated, Ordering::Relaxed, Ordering::Relaxed)
                .is_ok()
            {
                break;
            }
        }

        let confidence = self.hot.confidence();
        // Scale confidence to mirror activation decay.
        loop {
            let current = confidence.load(Ordering::Relaxed);
            let updated = (current * penalty).clamp(0.0, 1.0);
            if confidence
                .compare_exchange_weak(current, updated, Ordering::Relaxed, Ordering::Relaxed)
                .is_ok()
            {
                break;
            }
        }
    }

    /// Access the activation atomic for downstream consumers.
    #[must_use]
    pub fn activation_atomic(&self) -> &AtomicF32 {
        self.hot.activation()
    }

    /// Access the confidence atomic for downstream consumers.
    #[must_use]
    pub fn confidence_atomic(&self) -> &AtomicF32 {
        self.hot.confidence()
    }

    /// Access the visit counter for downstream consumers.
    #[must_use]
    pub fn visits_atomic(&self) -> &AtomicUsize {
        self.hot.visits()
    }

    /// Access the source-count atomic for downstream consumers.
    #[must_use]
    pub fn source_count_atomic(&self) -> &AtomicUsize {
        self.hot.source_count()
    }

    /// Assign a storage tier to the activation record.
    pub const fn set_storage_tier(&mut self, tier: storage_aware::StorageTier) {
        self.storage_tier = Some(tier);
    }

    /// Remove any storage tier association from the activation record.
    pub const fn clear_storage_tier(&mut self) {
        self.storage_tier = None;
    }

    /// Convert the activation record into a storage-aware activation snapshot.
    #[must_use]
    pub fn to_storage_aware(&self) -> storage_aware::StorageAwareActivation {
        let tier = self.storage_tier.unwrap_or(storage_aware::StorageTier::Hot);
        storage_aware::StorageAwareActivation::from_activation_record(self, tier)
    }

    /// Access the configured storage tier, if present.
    #[must_use]
    pub const fn storage_tier(&self) -> Option<storage_aware::StorageTier> {
        self.storage_tier
    }
}
/// Work-stealing task for parallel activation spreading
#[derive(Clone, Debug)]
pub struct ActivationTask {
    /// Target node to spread activation to
    pub target_node: NodeId,
    /// Activation level from source node
    pub source_activation: f32,
    /// Edge weight connecting source to target
    pub edge_weight: f32,
    /// Decay factor for this spreading step
    pub decay_factor: f32,
    /// Current spreading depth
    pub depth: u16,
    /// Maximum allowed depth
    pub max_depth: u16,
    /// Storage tier of the originating activation, if known.
    pub storage_tier: Option<storage_aware::StorageTier>,
    /// Path of node identifiers traversed to reach this task.
    pub path: Vec<NodeId>,
}

impl ActivationTask {
    /// Create a new activation spreading task
    #[must_use]
    pub fn new(
        target_node: NodeId,
        source_activation: f32,
        edge_weight: f32,
        decay_factor: f32,
        depth: u16,
        max_depth: u16,
    ) -> Self {
        let path = vec![target_node.clone()];
        Self {
            target_node,
            source_activation,
            edge_weight,
            decay_factor,
            depth,
            max_depth,
            storage_tier: None,
            path,
        }
    }

    /// Calculate the activation contribution this task will provide
    #[must_use]
    pub fn contribution(&self) -> f32 {
        self.source_activation * self.edge_weight * self.decay_factor
    }

    /// Check if this task should continue spreading based on depth and threshold
    #[must_use]
    pub fn should_continue(&self) -> bool {
        let threshold = self
            .storage_tier
            .map_or(0.01, storage_aware::StorageTier::activation_threshold);
        self.depth < self.max_depth && self.contribution() > threshold
    }

    /// Associate a storage tier with this activation task.
    pub const fn set_storage_tier(&mut self, tier: storage_aware::StorageTier) {
        self.storage_tier = Some(tier);
    }

    /// Builder-style helper to attach storage tier metadata.
    #[must_use]
    pub const fn with_storage_tier(mut self, tier: storage_aware::StorageTier) -> Self {
        self.set_storage_tier(tier);
        self
    }

    /// Attach a traversal path to the task.
    #[must_use]
    pub fn with_path(mut self, path: Vec<NodeId>) -> Self {
        self.path = path;
        self
    }

    /// Contribution adjusted by tier confidence factor.
    #[must_use]
    pub fn tier_adjusted_contribution(&self) -> f32 {
        let factor = self
            .storage_tier
            .map_or(1.0, storage_aware::StorageTier::confidence_factor);
        (self.contribution() * factor).clamp(0.0, 1.0)
    }
}

/// Cache-optimized weighted edge with compression
#[derive(Clone, Debug)]
pub struct WeightedEdge {
    /// Target node identifier
    pub target: NodeId,
    /// Edge weight (0.0 to 1.0)
    pub weight: f32,
    /// Type of neural connection
    pub edge_type: EdgeType,
    /// Cache-optimized hot fields for the target when available.
    pub hot_handle: Option<Arc<CacheOptimizedNode>>,
}

/// Edge types for Dale's law compliance
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EdgeType {
    /// Positive influence (increases activation)
    Excitatory,
    /// Negative influence (decreases activation)
    Inhibitory,
    /// Context-dependent influence (depends on pattern)
    Modulatory,
}

impl Default for EdgeType {
    fn default() -> Self {
        Self::Excitatory
    }
}

/// Decay function implementations for different spreading dynamics
#[derive(Clone, Debug)]
pub enum DecayFunction {
    /// Exponential decay with configurable rate
    Exponential {
        /// Decay rate parameter
        rate: f32,
    },
    /// Power law decay with configurable exponent
    PowerLaw {
        /// Power law exponent
        exponent: f32,
    },
    /// Linear decay with configurable slope
    Linear {
        /// Linear decay slope
        slope: f32,
    },
    /// Custom decay function
    Custom {
        /// Custom decay function
        func: fn(u16) -> f32,
    },
}

impl DecayFunction {
    /// Apply the decay function at the given depth
    #[must_use]
    pub fn apply(&self, depth: u16) -> f32 {
        match self {
            Self::Exponential { rate } => (-rate * f32::from(depth)).exp(),
            Self::PowerLaw { exponent } => (f32::from(depth) + 1.0).powf(-exponent),
            Self::Linear { slope } => (1.0 - slope * f32::from(depth)).max(0.0),
            Self::Custom { func } => func(depth),
        }
    }
}

impl Default for DecayFunction {
    fn default() -> Self {
        Self::Exponential { rate: 0.7 }
    }
}

/// Performance metrics collection for optimization
#[derive(Debug)]
pub struct SpreadingMetrics {
    /// Total number of activation operations performed
    pub total_activations: AtomicU64,
    /// Number of cache hits during spreading
    pub cache_hits: AtomicU64,
    /// Number of cache misses during spreading
    pub cache_misses: AtomicU64,
    /// Number of work stealing operations
    pub work_steals: AtomicU64,
    /// Number of cycles detected in the graph
    pub cycles_detected: AtomicU64,
    /// Average latency in nanoseconds
    pub average_latency: AtomicU64,
    /// Count of operations exceeding tier latency budget
    pub latency_budget_violations: AtomicU64,
    /// Total number of latency budget violations tracked for export
    pub latency_budget_violations_total: AtomicU64,
    /// Peak memory usage in bytes
    pub peak_memory_usage: AtomicU64,
    /// Parallel efficiency ratio (0.0 to 1.0)
    pub parallel_efficiency: AtomicF32,
    /// Tier-specific cycle detection counts
    pub cycle_counts: dashmap::DashMap<storage_aware::StorageTier, AtomicU64>,
    /// Rolling latency accumulators per tier (nanoseconds)
    tier_latency: dashmap::DashMap<storage_aware::StorageTier, TierLatencyStats>,
    /// Sampling counter used to throttle telemetry in high-throughput scenarios
    telemetry_sample_counter: AtomicU64,
    /// Current sampling rate (1 = sample every event, 10 = every tenth, etc.)
    telemetry_sample_rate: AtomicU64,
    /// Whether we have already detected high-throughput mode to avoid repeated toggles
    high_throughput_mode: AtomicBool,
    /// Total number of times recall fell back from spreading
    pub fallback_total: AtomicU64,
    /// Total number of spreading failures recorded
    pub failure_total: AtomicU64,
    /// Current state of the spreading circuit breaker (0=Closed,1=HalfOpen,2=Open)
    pub breaker_state: AtomicU64,
    /// Total breaker state transitions observed
    pub breaker_transitions: AtomicU64,
    /// Total GPU launch attempts
    pub gpu_launch_total: AtomicU64,
    /// Total GPU fallbacks to CPU
    pub gpu_fallback_total: AtomicU64,
    /// Total number of auto-tuner adjustments recorded
    pub autotune_changes: AtomicU64,
    /// Last recorded improvement estimate from auto-tuner (0.0-1.0)
    pub autotune_last_improvement: AtomicF32,
    /// Records currently available for reuse in the activation pool
    pub pool_available: AtomicU64,
    /// Records currently checked out of the activation pool
    pub pool_in_flight: AtomicU64,
    /// Highest observed availability for the activation pool
    pub pool_high_water_mark: AtomicU64,
    /// Total activation records created by the pool
    pub pool_total_created: AtomicU64,
    /// Total activation records reused from the pool
    pub pool_total_reused: AtomicU64,
    /// Number of misses that required allocating fresh activation records
    pub pool_miss_count: AtomicU64,
    /// Pool release attempts that failed due to outstanding references
    pub pool_release_failures: AtomicU64,
    /// Current activation pool hit rate (0.0 – 1.0)
    pub pool_hit_rate: AtomicF32,
    /// Activation pool utilization ratio (0.0 – 1.0)
    pub pool_utilization: AtomicF32,
    /// Adaptive batcher: total batch size update operations performed
    pub adaptive_batch_updates: AtomicU64,
    /// Adaptive batcher: guardrail activations preventing oscillation
    pub adaptive_guardrail_hits: AtomicU64,
    /// Adaptive batcher: topology changes detected (CPU core count, bandwidth class)
    pub adaptive_topology_changes: AtomicU64,
    /// Adaptive batcher: fallback activations due to high variance
    pub adaptive_fallback_count: AtomicU64,
    /// Adaptive batcher: latency EWMA (nanoseconds)
    pub adaptive_latency_ewma_ns: AtomicU64,
    /// Adaptive batcher: last recommended batch size for the hot tier
    pub adaptive_hot_batch_size: AtomicU64,
    /// Adaptive batcher: last recommended batch size for the warm tier
    pub adaptive_warm_batch_size: AtomicU64,
    /// Adaptive batcher: last recommended batch size for the cold tier
    pub adaptive_cold_batch_size: AtomicU64,
    /// Adaptive batcher: convergence confidence for the hot tier
    pub adaptive_hot_confidence: AtomicF32,
    /// Adaptive batcher: convergence confidence for the warm tier
    pub adaptive_warm_confidence: AtomicF32,
    /// Adaptive batcher: convergence confidence for the cold tier
    pub adaptive_cold_confidence: AtomicF32,
}

/// Rolling statistics for tier-specific latency tracking.
#[derive(Debug)]
struct TierLatencyStats {
    total_ns: AtomicU64,
    samples: AtomicU64,
    max_ns: AtomicU64,
}

impl TierLatencyStats {
    const fn new() -> Self {
        Self {
            total_ns: AtomicU64::new(0),
            samples: AtomicU64::new(0),
            max_ns: AtomicU64::new(0),
        }
    }

    fn record(&self, latency_ns: u64) {
        self.total_ns.fetch_add(latency_ns, Ordering::Relaxed);
        self.samples.fetch_add(1, Ordering::Relaxed);
        self.max_ns.fetch_max(latency_ns, Ordering::Relaxed);
    }
}

impl Default for TierLatencyStats {
    fn default() -> Self {
        Self::new()
    }
}

/// Snapshot view of tier latency aggregates for summaries and auto-tuning.
#[derive(Debug, Clone, Serialize)]
pub struct TierLatencySnapshot {
    /// Total latency accumulated for the tier (nanoseconds).
    pub total_ns: u64,
    /// Number of samples contributing to the total.
    pub samples: u64,
    /// Maximum latency observed for the tier (nanoseconds).
    pub max_ns: u64,
}

impl SpreadingMetrics {
    #[inline]
    const fn histogram_name_for_tier(tier: storage_aware::StorageTier) -> &'static str {
        match tier {
            storage_aware::StorageTier::Hot => METRIC_SPREADING_LATENCY_HOT,
            storage_aware::StorageTier::Warm => METRIC_SPREADING_LATENCY_WARM,
            storage_aware::StorageTier::Cold => METRIC_SPREADING_LATENCY_COLD,
        }
    }

    #[inline]
    fn should_sample_for_streaming(&self) -> bool {
        let count = self
            .telemetry_sample_counter
            .fetch_add(1, Ordering::Relaxed)
            + 1;
        let rate = self.telemetry_sample_rate.load(Ordering::Relaxed).max(1);

        if rate == 1
            && count >= 32_768
            && self
                .high_throughput_mode
                .compare_exchange(false, true, Ordering::Relaxed, Ordering::Relaxed)
                .is_ok()
        {
            self.telemetry_sample_rate.store(10, Ordering::Relaxed);
        }

        count.is_multiple_of(rate)
    }

    /// Record an activation latency observation for a tier and export telemetry.
    pub fn record_activation_latency(&self, tier: storage_aware::StorageTier, duration: Duration) {
        let duration_ns = duration.as_nanos().min(u128::from(u64::MAX));
        #[allow(clippy::cast_possible_truncation)]
        let duration_u64 = duration_ns as u64;

        self.total_activations.fetch_add(1, Ordering::Relaxed);
        self.average_latency.store(duration_u64, Ordering::Relaxed);

        self.tier_latency
            .entry(tier)
            .or_default()
            .record(duration_u64);

        if self.should_sample_for_streaming() {
            crate::metrics::observe_histogram(
                Self::histogram_name_for_tier(tier),
                duration.as_secs_f64(),
            );
            let total = self.total_activations.load(Ordering::Relaxed);
            crate::metrics::record_gauge(METRIC_SPREADING_ACTIVATIONS_TOTAL, total as f64);
        }
    }

    /// Record a latency budget violation, updating both counters and export gauges.
    pub fn record_latency_budget_violation(&self) {
        self.latency_budget_violations
            .fetch_add(1, Ordering::Relaxed);
        let total = self
            .latency_budget_violations_total
            .fetch_add(1, Ordering::Relaxed)
            + 1;
        crate::metrics::record_gauge(
            METRIC_SPREADING_LATENCY_BUDGET_VIOLATIONS_TOTAL,
            total as f64,
        );
    }

    /// Record a spreading failure for observability.
    pub fn record_spread_failure(&self) {
        let total = self.failure_total.fetch_add(1, Ordering::Relaxed) + 1;
        crate::metrics::record_gauge(METRIC_SPREADING_FAILURES_TOTAL, total as f64);
    }

    /// Record a fallback from spreading to similarity.
    pub fn record_fallback(&self) {
        let total = self.fallback_total.fetch_add(1, Ordering::Relaxed) + 1;
        crate::metrics::record_gauge(METRIC_SPREADING_FALLBACK_TOTAL, total as f64);
    }

    /// Record an auto-tuner adjustment for observability.
    pub fn record_autotune_change(&self, improvement: f64) {
        let total = self.autotune_changes.fetch_add(1, Ordering::Relaxed) + 1;
        crate::metrics::record_gauge(METRIC_SPREADING_AUTOTUNE_CHANGES_TOTAL, total as f64);
        let clamped = improvement.max(0.0);
        self.autotune_last_improvement
            .store(clamped as f32, Ordering::Relaxed);
        crate::metrics::record_gauge(METRIC_SPREADING_AUTOTUNE_LAST_IMPROVEMENT, clamped);
    }

    /// Record the current breaker state (0 = Closed, 1 = HalfOpen, 2 = Open).
    pub fn record_breaker_state(&self, state: u64) {
        self.breaker_state.store(state, Ordering::Relaxed);
        crate::metrics::record_gauge(METRIC_SPREADING_BREAKER_STATE, state as f64);
    }

    /// Record a breaker state transition to feed dashboards and audits.
    pub fn record_breaker_transition(&self, new_state: u64) {
        let transitions = self.breaker_transitions.fetch_add(1, Ordering::Relaxed) + 1;
        crate::metrics::record_gauge(
            METRIC_SPREADING_BREAKER_TRANSITIONS_TOTAL,
            transitions as f64,
        );
        self.record_breaker_state(new_state);
    }

    /// Record that a GPU launch path was taken.
    pub fn record_gpu_launch(&self) {
        let total = self.gpu_launch_total.fetch_add(1, Ordering::Relaxed) + 1;
        crate::metrics::record_gauge(METRIC_SPREADING_GPU_LAUNCH_TOTAL, total as f64);
    }

    /// Record that a GPU path fell back to CPU execution.
    pub fn record_gpu_fallback(&self) {
        let total = self.gpu_fallback_total.fetch_add(1, Ordering::Relaxed) + 1;
        crate::metrics::record_gauge(METRIC_SPREADING_GPU_FALLBACK_TOTAL, total as f64);
    }

    /// Snapshot tier latency accumulators for external consumers.
    pub fn tier_latency_snapshot(&self) -> Vec<(storage_aware::StorageTier, TierLatencySnapshot)> {
        self.tier_latency
            .iter()
            .map(|entry| {
                let stats = entry.value();
                (
                    *entry.key(),
                    TierLatencySnapshot {
                        total_ns: stats.total_ns.load(Ordering::Relaxed),
                        samples: stats.samples.load(Ordering::Relaxed),
                        max_ns: stats.max_ns.load(Ordering::Relaxed),
                    },
                )
            })
            .collect()
    }

    /// Current breaker state snapshot.
    pub fn breaker_state(&self) -> u64 {
        self.breaker_state.load(Ordering::Relaxed)
    }

    /// Total breaker transitions observed.
    pub fn breaker_transitions(&self) -> u64 {
        self.breaker_transitions.load(Ordering::Relaxed)
    }

    /// Calculate cache hit rate (0.0 to 1.0)
    pub fn cache_hit_rate(&self) -> f32 {
        let hits = self.cache_hits.load(Ordering::Relaxed);
        let misses = self.cache_misses.load(Ordering::Relaxed);
        let total = hits.saturating_add(misses);
        if total > 0 {
            let ratio = {
                #[allow(clippy::cast_precision_loss)]
                let hits_f64 = hits as f64;
                #[allow(clippy::cast_precision_loss)]
                let total_f64 = total as f64;
                hits_f64 / total_f64
            };
            #[allow(clippy::cast_possible_truncation)]
            {
                ratio as f32
            }
        } else {
            0.0
        }
    }

    /// Calculate work stealing rate (0.0 to 1.0)
    pub fn work_stealing_rate(&self) -> f32 {
        let steals = self.work_steals.load(Ordering::Relaxed);
        let total = self.total_activations.load(Ordering::Relaxed);
        if total > 0 {
            let ratio = {
                #[allow(clippy::cast_precision_loss)]
                let steals_f64 = steals as f64;
                #[allow(clippy::cast_precision_loss)]
                let total_f64 = total as f64;
                steals_f64 / total_f64
            };
            #[allow(clippy::cast_possible_truncation)]
            {
                ratio as f32
            }
        } else {
            0.0
        }
    }

    /// Reset all metrics to zero
    pub fn reset(&self) {
        self.total_activations.store(0, Ordering::Relaxed);
        self.cache_hits.store(0, Ordering::Relaxed);
        self.cache_misses.store(0, Ordering::Relaxed);
        self.work_steals.store(0, Ordering::Relaxed);
        self.cycles_detected.store(0, Ordering::Relaxed);
        self.average_latency.store(0, Ordering::Relaxed);
        self.latency_budget_violations.store(0, Ordering::Relaxed);
        self.peak_memory_usage.store(0, Ordering::Relaxed);
        self.parallel_efficiency.store(0.0, Ordering::Relaxed);
        self.cycle_counts.clear();
        self.tier_latency.clear();
        self.telemetry_sample_counter.store(0, Ordering::Relaxed);
        self.telemetry_sample_rate.store(1, Ordering::Relaxed);
        self.high_throughput_mode.store(false, Ordering::Relaxed);
        self.pool_available.store(0, Ordering::Relaxed);
        self.pool_in_flight.store(0, Ordering::Relaxed);
        self.pool_high_water_mark.store(0, Ordering::Relaxed);
        self.pool_total_created.store(0, Ordering::Relaxed);
        self.pool_total_reused.store(0, Ordering::Relaxed);
        self.pool_miss_count.store(0, Ordering::Relaxed);
        self.pool_release_failures.store(0, Ordering::Relaxed);
        self.pool_hit_rate.store(0.0, Ordering::Relaxed);
        self.pool_utilization.store(0.0, Ordering::Relaxed);
        self.adaptive_batch_updates.store(0, Ordering::Relaxed);
        self.adaptive_guardrail_hits.store(0, Ordering::Relaxed);
        self.adaptive_topology_changes.store(0, Ordering::Relaxed);
        self.adaptive_fallback_count.store(0, Ordering::Relaxed);
        self.adaptive_latency_ewma_ns.store(0, Ordering::Relaxed);
        self.adaptive_hot_batch_size.store(0, Ordering::Relaxed);
        self.adaptive_warm_batch_size.store(0, Ordering::Relaxed);
        self.adaptive_cold_batch_size.store(0, Ordering::Relaxed);
        self.adaptive_hot_confidence.store(0.0, Ordering::Relaxed);
        self.adaptive_warm_confidence.store(0.0, Ordering::Relaxed);
        self.adaptive_cold_confidence.store(0.0, Ordering::Relaxed);
    }

    /// Increment a per-tier cycle counter and return the updated total.
    pub fn increment_cycle_for_tier(&self, tier: storage_aware::StorageTier) -> u64 {
        let entry = self.cycle_counts.entry(tier).or_default();
        entry.fetch_add(1, Ordering::Relaxed) + 1
    }

    /// Retrieve the total number of cycles recorded for a tier.
    pub fn cycle_count_for_tier(&self, tier: storage_aware::StorageTier) -> u64 {
        self.cycle_counts
            .get(&tier)
            .map_or(0, |count| count.load(Ordering::Relaxed))
    }

    /// Record a snapshot of activation pool statistics for observability.
    pub fn record_pool_snapshot(&self, stats: &ActivationRecordPoolStats) {
        self.pool_available
            .store(stats.available as u64, Ordering::Relaxed);
        self.pool_in_flight
            .store(stats.in_flight, Ordering::Relaxed);
        self.pool_high_water_mark
            .store(stats.high_water_mark as u64, Ordering::Relaxed);
        self.pool_total_created
            .store(stats.total_created, Ordering::Relaxed);
        self.pool_total_reused
            .store(stats.total_reused, Ordering::Relaxed);
        self.pool_miss_count.store(stats.misses, Ordering::Relaxed);
        self.pool_release_failures
            .store(stats.release_failures, Ordering::Relaxed);
        self.pool_hit_rate.store(stats.hit_rate, Ordering::Relaxed);
        self.pool_utilization
            .store(stats.utilization, Ordering::Relaxed);

        #[allow(clippy::cast_precision_loss)]
        {
            crate::metrics::record_gauge(
                "activation_pool_available_records",
                stats.available as f64,
            );
            crate::metrics::record_gauge(
                "activation_pool_in_flight_records",
                stats.in_flight as f64,
            );
            crate::metrics::record_gauge(
                "activation_pool_high_water_mark",
                stats.high_water_mark as f64,
            );
            crate::metrics::record_gauge(
                "activation_pool_total_created",
                stats.total_created as f64,
            );
            crate::metrics::record_gauge("activation_pool_total_reused", stats.total_reused as f64);
            crate::metrics::record_gauge("activation_pool_miss_count", stats.misses as f64);
            crate::metrics::record_gauge(
                "activation_pool_release_failures",
                stats.release_failures as f64,
            );
        }
        crate::metrics::record_gauge("activation_pool_hit_rate", f64::from(stats.hit_rate));
        crate::metrics::record_gauge("activation_pool_utilization", f64::from(stats.utilization));
        crate::metrics::record_gauge(METRIC_SPREADING_POOL_HIT_RATE, f64::from(stats.hit_rate));
        crate::metrics::record_gauge(
            METRIC_SPREADING_POOL_UTILIZATION,
            f64::from(stats.utilization),
        );
    }

    /// Latest activation pool hit rate (0.0 – 1.0).
    pub fn pool_hit_rate(&self) -> f32 {
        self.pool_hit_rate.load(Ordering::Relaxed)
    }

    /// Latest activation pool utilization (0.0 – 1.0).
    pub fn pool_utilization(&self) -> f32 {
        self.pool_utilization.load(Ordering::Relaxed)
    }

    /// Record a snapshot of adaptive batcher metrics for observability.
    pub fn record_adaptive_batcher_snapshot(&self, snapshot: &AdaptiveBatcherSnapshot) {
        self.adaptive_batch_updates
            .store(snapshot.update_count, Ordering::Relaxed);
        self.adaptive_guardrail_hits
            .store(snapshot.guardrail_hits, Ordering::Relaxed);
        self.adaptive_topology_changes
            .store(snapshot.topology_changes, Ordering::Relaxed);
        self.adaptive_fallback_count
            .store(snapshot.fallback_activations, Ordering::Relaxed);
        self.adaptive_latency_ewma_ns
            .store(snapshot.latency_ewma_ns, Ordering::Relaxed);
        self.adaptive_hot_batch_size
            .store(snapshot.hot_batch_size, Ordering::Relaxed);
        self.adaptive_warm_batch_size
            .store(snapshot.warm_batch_size, Ordering::Relaxed);
        self.adaptive_cold_batch_size
            .store(snapshot.cold_batch_size, Ordering::Relaxed);
        self.adaptive_hot_confidence
            .store(snapshot.hot_confidence, Ordering::Relaxed);
        self.adaptive_warm_confidence
            .store(snapshot.warm_confidence, Ordering::Relaxed);
        self.adaptive_cold_confidence
            .store(snapshot.cold_confidence, Ordering::Relaxed);

        // Export to global metrics pipeline
        #[allow(clippy::cast_precision_loss)]
        {
            crate::metrics::record_gauge(
                "adaptive_batch_updates_total",
                snapshot.update_count as f64,
            );
            crate::metrics::record_gauge(
                "adaptive_guardrail_hits_total",
                snapshot.guardrail_hits as f64,
            );
            crate::metrics::record_gauge(
                "adaptive_topology_changes_total",
                snapshot.topology_changes as f64,
            );
            crate::metrics::record_gauge(
                "adaptive_fallback_activations_total",
                snapshot.fallback_activations as f64,
            );
            crate::metrics::record_gauge(
                "adaptive_batch_latency_ewma_ns",
                snapshot.latency_ewma_ns as f64,
            );
            crate::metrics::record_gauge("adaptive_batch_hot_size", snapshot.hot_batch_size as f64);
            crate::metrics::record_gauge(
                "adaptive_batch_warm_size",
                snapshot.warm_batch_size as f64,
            );
            crate::metrics::record_gauge(
                "adaptive_batch_cold_size",
                snapshot.cold_batch_size as f64,
            );
            crate::metrics::record_gauge(
                "adaptive_batch_hot_confidence",
                f64::from(snapshot.hot_confidence),
            );
            crate::metrics::record_gauge(
                "adaptive_batch_warm_confidence",
                f64::from(snapshot.warm_confidence),
            );
            crate::metrics::record_gauge(
                "adaptive_batch_cold_confidence",
                f64::from(snapshot.cold_confidence),
            );
        }
    }

    /// Get current adaptive batch update count.
    pub fn adaptive_batch_updates(&self) -> u64 {
        self.adaptive_batch_updates.load(Ordering::Relaxed)
    }

    /// Get current adaptive guardrail hit count.
    pub fn adaptive_guardrail_hits(&self) -> u64 {
        self.adaptive_guardrail_hits.load(Ordering::Relaxed)
    }

    /// Get current adaptive topology change count.
    pub fn adaptive_topology_changes(&self) -> u64 {
        self.adaptive_topology_changes.load(Ordering::Relaxed)
    }

    /// Get current adaptive fallback count.
    pub fn adaptive_fallback_count(&self) -> u64 {
        self.adaptive_fallback_count.load(Ordering::Relaxed)
    }
}

impl Default for SpreadingMetrics {
    fn default() -> Self {
        Self {
            total_activations: AtomicU64::new(0),
            cache_hits: AtomicU64::new(0),
            cache_misses: AtomicU64::new(0),
            work_steals: AtomicU64::new(0),
            cycles_detected: AtomicU64::new(0),
            average_latency: AtomicU64::new(0),
            latency_budget_violations: AtomicU64::new(0),
            latency_budget_violations_total: AtomicU64::new(0),
            peak_memory_usage: AtomicU64::new(0),
            parallel_efficiency: AtomicF32::new(0.0),
            cycle_counts: DashMap::new(),
            tier_latency: DashMap::new(),
            telemetry_sample_counter: AtomicU64::new(0),
            telemetry_sample_rate: AtomicU64::new(1),
            high_throughput_mode: AtomicBool::new(false),
            fallback_total: AtomicU64::new(0),
            failure_total: AtomicU64::new(0),
            breaker_state: AtomicU64::new(0),
            breaker_transitions: AtomicU64::new(0),
            gpu_launch_total: AtomicU64::new(0),
            gpu_fallback_total: AtomicU64::new(0),
            autotune_changes: AtomicU64::new(0),
            autotune_last_improvement: AtomicF32::new(0.0),
            pool_available: AtomicU64::new(0),
            pool_in_flight: AtomicU64::new(0),
            pool_high_water_mark: AtomicU64::new(0),
            pool_total_created: AtomicU64::new(0),
            pool_total_reused: AtomicU64::new(0),
            pool_miss_count: AtomicU64::new(0),
            pool_release_failures: AtomicU64::new(0),
            pool_hit_rate: AtomicF32::new(0.0),
            pool_utilization: AtomicF32::new(0.0),
            adaptive_batch_updates: AtomicU64::new(0),
            adaptive_guardrail_hits: AtomicU64::new(0),
            adaptive_topology_changes: AtomicU64::new(0),
            adaptive_fallback_count: AtomicU64::new(0),
            adaptive_latency_ewma_ns: AtomicU64::new(0),
            adaptive_hot_batch_size: AtomicU64::new(0),
            adaptive_warm_batch_size: AtomicU64::new(0),
            adaptive_cold_batch_size: AtomicU64::new(0),
            adaptive_hot_confidence: AtomicF32::new(0.0),
            adaptive_warm_confidence: AtomicF32::new(0.0),
            adaptive_cold_confidence: AtomicF32::new(0.0),
        }
    }
}

/// Summary statistics for a specific storage tier after spreading completes.
#[derive(Debug, Clone)]
pub struct TierSummary {
    /// Tier that the summary references.
    pub tier: storage_aware::StorageTier,
    /// Number of memories activated within this tier.
    pub node_count: usize,
    /// Sum of activation values observed for nodes in the tier.
    pub total_activation: f32,
    /// Average confidence across nodes in the tier.
    pub average_confidence: f32,
    /// Whether the tier exceeded its scheduling deadline or timeout.
    pub deadline_missed: bool,
}

/// Trace entry capturing activation flow for deterministic debugging
#[derive(Debug, Clone, PartialEq)]
pub struct TraceEntry {
    /// Depth at which this activation occurred
    pub depth: u16,
    /// Node identifier receiving activation
    pub target_node: NodeId,
    /// Activation level achieved at this step
    pub activation: f32,
    /// Confidence score for this activation
    pub confidence: f32,
    /// Source node that triggered this activation
    pub source_node: Option<NodeId>,
}

impl Default for TierSummary {
    fn default() -> Self {
        Self {
            tier: storage_aware::StorageTier::Hot,
            node_count: 0,
            total_activation: 0.0,
            average_confidence: 0.0,
            deadline_missed: false,
        }
    }
}

/// Results produced by a spreading operation including per-tier summaries.
#[derive(Debug, Default)]
pub struct SpreadingResults {
    /// Final storage-aware activations captured at the end of spreading.
    pub activations: Vec<storage_aware::StorageAwareActivation>,
    /// Aggregated statistics keyed by storage tier.
    pub tier_summaries: HashMap<storage_aware::StorageTier, TierSummary>,
    /// Detected cycle paths encountered during spreading.
    pub cycle_paths: Vec<Vec<NodeId>>,
    /// Deterministic trace of activation flow (only populated when trace_activation_flow enabled)
    pub deterministic_trace: Vec<TraceEntry>,
}

impl SpreadingResults {
    /// Lookup convenience for tier summaries.
    #[must_use]
    pub fn tier_summary(&self, tier: storage_aware::StorageTier) -> Option<&TierSummary> {
        self.tier_summaries.get(&tier)
    }
}

#[doc = include_str!("doc/parallel_spreading_config.md")]
/// Configuration for high-performance parallel spreading
#[derive(Clone, Debug)]
#[allow(clippy::struct_excessive_bools)]
pub struct ParallelSpreadingConfig {
    // Parallelism control
    /// Worker thread count for parallel processing
    pub num_threads: usize,
    /// Probability of stealing work vs processing local work
    pub work_stealing_ratio: f32,
    /// Number of tasks processed per batch
    pub batch_size: usize,

    // Memory management
    /// Initial size of activation pool for memory reuse
    pub pool_initial_size: usize,
    /// Target cache line alignment in bytes
    pub cache_line_size: usize,
    /// Enable NUMA-aware memory allocation
    pub numa_aware: bool,

    // Spreading dynamics
    /// Maximum spreading depth allowed
    pub max_depth: u16,
    /// Decay function for activation spreading
    pub decay_function: DecayFunction,
    /// Minimum activation threshold for processing
    pub threshold: f32,
    /// Enable cycle detection during spreading
    pub cycle_detection: bool,
    /// Multiplicative penalty applied when cycle detected
    pub cycle_penalty_factor: f32,
    /// Tier-specific hop budgets before treating a revisit as a cycle
    pub tier_cycle_budgets: HashMap<storage_aware::StorageTier, usize>,

    // Integration parameters
    /// SIMD vector width for bulk operations
    pub simd_batch_size: usize,
    /// Cache prefetch lookahead distance
    pub prefetch_distance: usize,

    // Tier-aware scheduling
    /// Timeouts for hot, warm, and cold tiers respectively.
    pub tier_timeouts: [Duration; 3],
    /// Maximum concurrent tasks allowed per tier before queueing new work.
    pub max_concurrent_per_tier: usize,
    /// Whether the scheduler always prioritises the hot tier when available.
    pub priority_hot_tier: bool,

    // Determinism and reproducibility
    /// Enable deterministic execution mode
    pub deterministic: bool,
    /// RNG seed for reproducible results
    pub seed: Option<u64>,
    /// Phase barrier synchronization interval
    pub phase_sync_interval: Duration,

    // Performance monitoring
    /// Enable performance metrics collection
    pub enable_metrics: bool,
    /// Enable detailed activation flow tracing
    pub trace_activation_flow: bool,

    // GPU acceleration
    /// Enable GPU acceleration for batch spreading
    pub enable_gpu: bool,
    /// Minimum batch size to use GPU acceleration
    pub gpu_threshold: usize,

    // Memory pool configuration
    /// Enable memory pool for efficient allocation
    pub enable_memory_pool: bool,
    /// Size of each memory pool chunk in bytes
    pub pool_chunk_size: usize,
    /// Maximum number of memory pool chunks
    pub pool_max_chunks: usize,

    // Adaptive batching configuration
    /// Adaptive batching mode and configuration
    pub adaptive_batcher_config: Option<AdaptiveBatcherConfig>,

    // Completion timeout configuration
    /// Maximum duration to wait for spreading completion before timing out.
    /// If None, uses a computed timeout based on available parallelism.
    pub completion_timeout: Option<Duration>,
}

impl Default for ParallelSpreadingConfig {
    fn default() -> Self {
        Self {
            num_threads: std::thread::available_parallelism()
                .map(std::num::NonZero::get)
                .unwrap_or(4),
            work_stealing_ratio: 0.1,
            batch_size: 64,
            pool_initial_size: 10000,
            cache_line_size: 64,
            numa_aware: true,
            max_depth: 4,
            decay_function: DecayFunction::default(),
            threshold: 0.01,
            cycle_detection: true,
            cycle_penalty_factor: 0.93,
            tier_cycle_budgets: HashMap::from([
                (storage_aware::StorageTier::Hot, 2),
                (storage_aware::StorageTier::Warm, 3),
                (storage_aware::StorageTier::Cold, 4),
            ]),
            simd_batch_size: 8,
            prefetch_distance: 64,
            tier_timeouts: [
                Duration::from_micros(100),
                Duration::from_millis(1),
                Duration::from_millis(10),
            ],
            max_concurrent_per_tier: 32,
            priority_hot_tier: true,
            deterministic: false,
            seed: None,
            phase_sync_interval: Duration::from_millis(10),
            enable_metrics: true,
            trace_activation_flow: false,
            enable_gpu: false,
            gpu_threshold: 64,
            enable_memory_pool: true,
            pool_chunk_size: 8192,         // 8KB per chunk
            pool_max_chunks: 16,           // Max 128KB total
            adaptive_batcher_config: None, // Disabled by default
            completion_timeout: None,      // Use computed timeout by default
        }
    }
}

impl ParallelSpreadingConfig {
    /// Create a deterministic configuration with a specific seed
    ///
    /// Enables deterministic execution mode with canonical task ordering,
    /// seeded randomness, and hop-level synchronization for reproducible results.
    ///
    /// # Example
    /// ```
    /// use engram_core::activation::ParallelSpreadingConfig;
    /// let config = ParallelSpreadingConfig::deterministic(42);
    /// assert!(config.deterministic);
    /// assert_eq!(config.seed, Some(42));
    /// ```
    #[must_use]
    pub fn deterministic(seed: u64) -> Self {
        Self {
            deterministic: true,
            seed: Some(seed),
            trace_activation_flow: true,
            ..Self::default()
        }
    }
}

/// Error types for activation spreading
#[derive(Debug, thiserror::Error)]
pub enum ActivationError {
    /// Invalid configuration provided
    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),
    /// Memory allocation failed during activation spreading
    #[error("Memory allocation failed")]
    AllocationFailed,
    /// Cycle detected in activation graph
    #[error("Cycle detected in graph: {0:?}")]
    CycleDetected(Vec<NodeId>),
    /// Threading or concurrency error occurred
    #[error("Threading error: {0}")]
    ThreadingError(String),
}

/// Result type for activation operations
pub type ActivationResult<T> = Result<T, ActivationError>;

/// Memory graph for activation spreading using the unified backend architecture.
///
/// This uses `DashMapBackend` for lock-free concurrent access patterns needed
/// for parallel activation spreading.
pub use crate::memory_graph::{DashMapBackend, GraphConfig, UnifiedMemoryGraph};

/// Activation graph wrapper with instance-level storage for UUID mappings and embeddings.
///
/// This eliminates global state and provides proper test isolation.
pub struct ActivationGraph {
    graph: UnifiedMemoryGraph<DashMapBackend>,
    uuid_mappings: DashMap<uuid::Uuid, NodeId>,
    embeddings: DashMap<NodeId, [f32; 768]>,
    hot_handles: DashMap<NodeId, Arc<CacheOptimizedNode>>,
}

impl std::ops::Deref for ActivationGraph {
    type Target = UnifiedMemoryGraph<DashMapBackend>;

    fn deref(&self) -> &Self::Target {
        &self.graph
    }
}

/// Default memory graph type for activation spreading
pub type MemoryGraph = ActivationGraph;

/// Create a new memory graph optimized for parallel activation spreading
#[must_use]
pub fn create_activation_graph() -> MemoryGraph {
    let config = GraphConfig {
        max_results: 1000,
        enable_spreading: true,
        decay_rate: 0.8,
        max_depth: 5,
        activation_threshold: 0.01,
    };
    ActivationGraph {
        graph: UnifiedMemoryGraph::new(DashMapBackend::default(), config),
        uuid_mappings: DashMap::new(),
        embeddings: DashMap::new(),
        hot_handles: DashMap::new(),
    }
}

/// Extension trait to add activation-specific methods to the unified graph
pub trait ActivationGraphExt {
    /// Add an edge between two nodes with specified weight and type
    fn add_edge(&self, source: NodeId, target: NodeId, weight: f32, edge_type: EdgeType);
    /// Get neighbors of a specific node
    fn get_neighbors(&self, node_id: &NodeId) -> Option<Vec<WeightedEdge>>;
    /// Get the total number of nodes in the graph
    fn node_count(&self) -> usize;
    /// Get the total number of edges in the graph
    fn edge_count(&self) -> usize;

    /// Store embedding for a node
    ///
    /// Embeddings are cached per-node and do not automatically sync with the HNSW index.
    /// Call this method when:
    /// - Adding a new node to the graph
    /// - An episode's embedding is updated in the HNSW index
    /// - Rebuilding the graph after memory consolidation
    ///
    /// # Memory Usage
    /// Each embedding consumes 3KB (768 × f32). Memory grows linearly:
    /// - 100K nodes = 300MB
    /// - 1M nodes = 3GB
    /// - 10M nodes = 30GB
    ///
    /// # Thread Safety
    /// This method is thread-safe and can be called concurrently. Last write wins.
    fn set_embedding(&self, node_id: &NodeId, embedding: &[f32; 768]);

    /// Retrieve embedding for a node
    ///
    /// Returns a copy of the embedding (3KB) if present, or `None` if the node
    /// has no cached embedding. When this returns `None`, SIMD batch spreading
    /// will gracefully fall back to scalar operations.
    ///
    /// # Performance Note
    /// This method copies the 3KB embedding array. For batch operations processing
    /// 8 neighbors, this results in ~27KB of copies (current node + 8 neighbors).
    /// This overhead is typically negligible compared to the 2-3× SIMD speedup gain.
    fn get_embedding(&self, node_id: &NodeId) -> Option<[f32; 768]>;

    /// Get all node IDs in the graph
    ///
    /// Returns a vector of all node IDs that have been added to the graph.
    /// This is useful for testing and diagnostics.
    ///
    /// # Performance Note
    /// This method performs a full scan of the graph's UUID mappings and allocates
    /// a new vector. Use sparingly in performance-critical paths.
    fn get_all_nodes(&self) -> Vec<NodeId>;

    /// Register a cache-optimized hot node handle for reuse.
    fn register_hot_handle(&self, node_id: NodeId, handle: Arc<CacheOptimizedNode>);

    /// Remove a cached hot node handle when the activation record is released.
    fn clear_hot_handle(&self, node_id: &NodeId);

    /// Retrieve the cached hot node handle for a node if present.
    fn get_hot_handle(&self, node_id: &NodeId) -> Option<Arc<CacheOptimizedNode>>;
}

impl ActivationGraphExt for ActivationGraph {
    fn add_edge(&self, source: NodeId, target: NodeId, weight: f32, _edge_type: EdgeType) {
        // Convert NodeId (String) to Uuid for the backend
        use crate::memory_graph::traits::MemoryBackend;
        use uuid::Uuid;
        let source_id = Uuid::new_v5(&Uuid::NAMESPACE_OID, source.as_bytes());
        let target_id = Uuid::new_v5(&Uuid::NAMESPACE_OID, target.as_bytes());

        // Store reverse mappings for NodeId recovery in instance storage
        self.uuid_mappings.insert(source_id, source.clone());
        self.uuid_mappings.insert(target_id, target.clone());

        // Ensure both nodes exist in backend for spreading activation to work
        // The backend's spread_activation() only propagates to nodes in the memories DashMap
        if !self.graph.contains(&source_id) {
            let memory = crate::Memory::new(
                source,
                [0.0; 768], // Zero embedding - will be set separately if needed
                crate::Confidence::LOW,
            );
            let _ = self.graph.backend().store(source_id, memory);
        }
        if !self.graph.contains(&target_id) {
            let memory = crate::Memory::new(
                target,
                [0.0; 768], // Zero embedding - will be set separately if needed
                crate::Confidence::LOW,
            );
            let _ = self.graph.backend().store(target_id, memory);
        }

        // Add edge using the UnifiedMemoryGraph's add_edge method
        let _ = self.graph.add_edge(source_id, target_id, weight);

        // Edge type metadata will be handled in a future iteration
    }

    fn get_neighbors(&self, node_id: &NodeId) -> Option<Vec<WeightedEdge>> {
        use uuid::Uuid;
        let id = Uuid::new_v5(&Uuid::NAMESPACE_OID, node_id.as_bytes());

        self.graph.get_neighbors(&id).ok().map(|neighbors| {
            neighbors
                .into_iter()
                .map(|(neighbor_id, weight)| {
                    // Convert Uuid back to original NodeId using instance storage
                    let target = self
                        .uuid_mappings
                        .get(&neighbor_id)
                        .map_or_else(|| neighbor_id.to_string(), |entry| entry.value().clone());

                    let hot_handle = self
                        .hot_handles
                        .get(&target)
                        .map(|entry| entry.value().clone());

                    WeightedEdge {
                        target,
                        weight,
                        edge_type: EdgeType::Excitatory, // Default for now
                        hot_handle,
                    }
                })
                .collect()
        })
    }

    fn node_count(&self) -> usize {
        // Use uuid_mappings for consistency with get_all_nodes()
        // This counts all nodes that have been added via add_edge()
        self.uuid_mappings.len()
    }

    fn edge_count(&self) -> usize {
        // Use the all_edges method from UnifiedMemoryGraph
        self.graph.all_edges().map_or(0, |edges| edges.len())
    }

    fn set_embedding(&self, node_id: &NodeId, embedding: &[f32; 768]) {
        self.embeddings.insert(node_id.clone(), *embedding);
    }

    fn get_embedding(&self, node_id: &NodeId) -> Option<[f32; 768]> {
        self.embeddings.get(node_id).map(|entry| *entry.value())
    }

    fn get_all_nodes(&self) -> Vec<NodeId> {
        self.uuid_mappings
            .iter()
            .map(|entry| entry.value().clone())
            .collect()
    }

    fn register_hot_handle(&self, node_id: NodeId, handle: Arc<CacheOptimizedNode>) {
        self.hot_handles.insert(node_id, handle);
    }

    fn clear_hot_handle(&self, node_id: &NodeId) {
        self.hot_handles.remove(node_id);
    }

    fn get_hot_handle(&self, node_id: &NodeId) -> Option<Arc<CacheOptimizedNode>> {
        self.hot_handles
            .get(node_id)
            .map(|entry| entry.value().clone())
    }
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use super::storage_aware::StorageTier;
    use super::{
        ActivationGraphExt, ActivationRecord, ActivationRecordPoolStats, create_activation_graph,
    };

    #[test]
    fn test_embedding_storage_and_retrieval() {
        let graph = create_activation_graph();
        let node_id = "test_node".to_string();
        let embedding = [0.5f32; 768];

        // Store embedding
        graph.set_embedding(&node_id, &embedding);

        // Retrieve embedding
        let retrieved = graph.get_embedding(&node_id);
        assert!(retrieved.is_some(), "Embedding should be retrievable");

        let retrieved_embedding = retrieved.expect("embedding should exist");
        for (i, (&expected, &actual)) in
            embedding.iter().zip(retrieved_embedding.iter()).enumerate()
        {
            assert!(
                (expected - actual).abs() < f32::EPSILON,
                "Mismatch at index {i}"
            );
        }
    }

    #[test]
    fn test_hot_handle_registration_lifecycle() {
        let graph = create_activation_graph();
        let mut record = ActivationRecord::new("hot-node".to_string(), 0.1);
        record.set_storage_tier(StorageTier::Hot);
        let handle = record.hot_handle();

        graph.register_hot_handle("hot-node".to_string(), handle);
        let retrieved = graph.get_hot_handle(&"hot-node".to_string());
        assert!(retrieved.is_some(), "handle should be retrievable");

        graph.clear_hot_handle(&"hot-node".to_string());
        assert!(
            graph.get_hot_handle(&"hot-node".to_string()).is_none(),
            "handle should be cleared"
        );
    }

    #[test]
    fn test_embedding_retrieval_nonexistent() {
        let graph = create_activation_graph();
        let node_id = "nonexistent_node".to_string();

        let retrieved = graph.get_embedding(&node_id);
        assert!(retrieved.is_none(), "Non-existent node should return None");
    }

    #[test]
    fn test_embedding_update() {
        let graph = create_activation_graph();
        let node_id = "update_test".to_string();

        // Store initial embedding
        let embedding1 = [1.0f32; 768];
        graph.set_embedding(&node_id, &embedding1);

        // Update with new embedding
        let embedding2 = [2.0f32; 768];
        graph.set_embedding(&node_id, &embedding2);

        // Verify updated embedding
        let retrieved = graph
            .get_embedding(&node_id)
            .expect("embedding should exist");
        assert!(
            (retrieved[0] - 2.0).abs() < f32::EPSILON,
            "Embedding should be updated"
        );
    }

    #[test]
    fn test_multiple_node_embeddings() {
        let graph = create_activation_graph();

        // Store embeddings for multiple nodes
        for i in 0..10 {
            let node_id = format!("node_{i}");
            let mut embedding = [0.0f32; 768];
            embedding[0] = i as f32;
            graph.set_embedding(&node_id, &embedding);
        }

        // Verify all embeddings
        for i in 0..10 {
            let node_id = format!("node_{i}");
            let retrieved = graph
                .get_embedding(&node_id)
                .expect("embedding should exist");
            assert!(
                (retrieved[0] - i as f32).abs() < f32::EPSILON,
                "Embedding for node_{i} should match"
            );
        }
    }
    use super::*;
    use std::sync::atomic::Ordering as AtomicOrdering;

    type TestResult<T = ()> = Result<T, String>;

    #[test]
    fn test_activation_record_creation() {
        let record = ActivationRecord::new("test_node".to_string(), 0.1);
        assert_eq!(record.node_id, "test_node");
        assert!((record.decay_rate - 0.1).abs() < f32::EPSILON);
        assert!(record.get_activation().abs() < f32::EPSILON);
    }

    #[test]
    fn test_activation_accumulation() {
        let record = ActivationRecord::new("test_node".to_string(), 0.1);

        assert!(record.accumulate_activation(0.5));
        assert!((record.get_activation() - 0.5).abs() < 1e-6);

        assert!(record.accumulate_activation(0.3));
        assert!((record.get_activation() - 0.8).abs() < 1e-6);

        // Test clamping to 1.0
        assert!(record.accumulate_activation(0.5));
        assert!((record.get_activation() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_activation_threshold() {
        let record = ActivationRecord::new("test_node".to_string(), 0.1);

        // Below threshold should not update
        assert!(!record.accumulate_activation(0.005));
        assert!(record.get_activation().abs() < f32::EPSILON);

        // Above threshold should update
        assert!(record.accumulate_activation(0.02));
        assert!((record.get_activation() - 0.02).abs() < 1e-6);
    }

    #[test]
    fn test_activation_task() {
        let task = ActivationTask::new("target".to_string(), 0.8, 0.5, 0.7, 2, 5);

        let expected = (0.8_f32 * 0.5).mul_add(0.7, 0.0);
        assert!((task.contribution() - expected).abs() < 1e-6);
        assert!(task.should_continue());

        let task_too_deep = ActivationTask::new("target".to_string(), 0.8, 0.5, 0.7, 5, 5);

        assert!(!task_too_deep.should_continue());

        let warm_task = ActivationTask::new("warm".to_string(), 0.4, 0.5, 0.7, 1, 5)
            .with_storage_tier(StorageTier::Warm);
        assert!(warm_task.should_continue());

        let cold_task = ActivationTask::new("cold".to_string(), 0.05, 0.4, 0.6, 2, 5)
            .with_storage_tier(StorageTier::Cold);
        assert!(!cold_task.should_continue());
    }

    #[test]
    fn test_activation_record_tier_thresholds() {
        let mut record = ActivationRecord::new("tiered".to_string(), 0.1);
        record.set_storage_tier(StorageTier::Warm);
        assert!(!record.accumulate_activation(0.03));
        assert!(record.accumulate_activation(0.06));
        assert!((record.get_activation() - 0.06).abs() < 1e-6);
        record.visits_atomic().fetch_add(2, AtomicOrdering::Relaxed);

        let storage_aware = record.to_storage_aware();
        assert_eq!(storage_aware.storage_tier, StorageTier::Warm);
        assert!((storage_aware.tier_threshold() - 0.05).abs() < f32::EPSILON);
        let hops = storage_aware.hop_count.load(AtomicOrdering::Relaxed);
        assert!(hops >= 2);
        assert!(storage_aware.adjust_confidence_for_tier(0.5) < 0.5);
    }

    #[test]
    fn cycle_penalty_scales_activation_and_confidence() {
        let record = ActivationRecord::new("penalty".to_string(), 0.1);
        record.set_activation(0.8);
        assert!((record.get_confidence() - 0.8).abs() < f32::EPSILON);

        record.apply_cycle_penalty(0.5);
        let activation = record.get_activation();
        let confidence = record.get_confidence();

        assert!((activation - 0.4).abs() < 1e-6);
        assert!((confidence - 0.4).abs() < 1e-6);
    }

    #[test]
    fn test_decay_functions() {
        let exp = DecayFunction::Exponential { rate: 0.5 };
        let power = DecayFunction::PowerLaw { exponent: 1.5 };
        let linear = DecayFunction::Linear { slope: 0.2 };

        assert!((exp.apply(0) - 1.0).abs() < 1e-6);
        assert!(exp.apply(1) < exp.apply(0));
        assert!(exp.apply(2) < exp.apply(1));

        assert!((power.apply(0) - 1.0).abs() < 1e-6);
        assert!(power.apply(1) < power.apply(0));

        assert!((linear.apply(0) - 1.0).abs() < 1e-6);
        assert!(linear.apply(1) < linear.apply(0));
        assert!(linear.apply(5).abs() < f32::EPSILON); // Should clamp to 0
    }

    #[test]
    fn test_memory_graph() -> TestResult {
        let graph = create_activation_graph();

        ActivationGraphExt::add_edge(
            &graph,
            "node1".to_string(),
            "node2".to_string(),
            0.8,
            EdgeType::Excitatory,
        );

        ActivationGraphExt::add_edge(
            &graph,
            "node1".to_string(),
            "node3".to_string(),
            0.3,
            EdgeType::Inhibitory,
        );

        let neighbors = ActivationGraphExt::get_neighbors(&graph, &"node1".to_string())
            .ok_or_else(|| "expected neighbors for node1".to_string())?;
        if neighbors.len() != 2 {
            return Err("expected two neighbors for node1".to_string());
        }

        // Note: The targets are UUID strings converted from the original node names
        // We just need to verify we have 2 neighbors with the correct weights
        let weights: Vec<f32> = neighbors.iter().map(|n| n.weight).collect();
        if !weights.contains(&0.8f32) {
            return Err("expected excitatory edge weight 0.8".to_string());
        }
        if !weights.contains(&0.3f32) {
            return Err("expected inhibitory edge weight 0.3".to_string());
        }

        // The graph has edges but may not track nodes separately
        // Just verify we have the edges
        if graph.edge_count() != 2 {
            return Err("expected graph to contain two edges".to_string());
        }

        Ok(())
    }

    #[test]
    fn test_spreading_metrics() {
        let metrics = SpreadingMetrics::default();

        metrics.cache_hits.store(80, Ordering::Relaxed);
        metrics.cache_misses.store(20, Ordering::Relaxed);
        metrics.work_steals.store(15, Ordering::Relaxed);
        metrics.total_activations.store(100, Ordering::Relaxed);

        assert!((metrics.cache_hit_rate() - 0.8).abs() < 1e-6);
        assert!((metrics.work_stealing_rate() - 0.15).abs() < 1e-6);

        let stats = ActivationRecordPoolStats {
            available: 12,
            in_flight: 8,
            high_water_mark: 20,
            total_created: 32,
            total_reused: 24,
            misses: 8,
            hit_rate: 0.75,
            utilization: 0.4,
            release_failures: 1,
        };

        metrics.record_pool_snapshot(&stats);
        assert_eq!(
            metrics.pool_available.load(Ordering::Relaxed),
            stats.available as u64
        );
        assert_eq!(
            metrics.pool_in_flight.load(Ordering::Relaxed),
            stats.in_flight
        );
        assert_eq!(
            metrics.pool_high_water_mark.load(Ordering::Relaxed),
            stats.high_water_mark as u64
        );
        assert_eq!(
            metrics.pool_total_created.load(Ordering::Relaxed),
            stats.total_created
        );
        assert_eq!(
            metrics.pool_total_reused.load(Ordering::Relaxed),
            stats.total_reused
        );
        assert_eq!(
            metrics.pool_miss_count.load(Ordering::Relaxed),
            stats.misses
        );
        assert_eq!(
            metrics.pool_release_failures.load(Ordering::Relaxed),
            stats.release_failures
        );
        assert!((metrics.pool_hit_rate() - stats.hit_rate).abs() < f32::EPSILON);
        assert!((metrics.pool_utilization() - stats.utilization).abs() < f32::EPSILON);

        metrics.reset();
        assert!(metrics.cache_hit_rate().abs() < f32::EPSILON);
        assert!(metrics.work_stealing_rate().abs() < f32::EPSILON);
        assert!(metrics.pool_hit_rate().abs() < f32::EPSILON);
        assert!(metrics.pool_utilization().abs() < f32::EPSILON);
        assert_eq!(metrics.pool_available.load(Ordering::Relaxed), 0);
        assert_eq!(metrics.pool_in_flight.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn test_parallel_spreading_config() {
        let config = ParallelSpreadingConfig::default();

        assert!(config.num_threads > 0);
        assert!(config.max_depth > 0);
        assert!(config.threshold > 0.0);
        assert!(config.work_stealing_ratio >= 0.0 && config.work_stealing_ratio <= 1.0);

        // Test custom config
        let custom_config = ParallelSpreadingConfig {
            max_depth: 6,
            threshold: 0.05,
            deterministic: true,
            seed: Some(42),
            ..config
        };

        assert_eq!(custom_config.max_depth, 6);
        assert!((custom_config.threshold - 0.05).abs() < f32::EPSILON);
        assert!(custom_config.deterministic);
        assert_eq!(custom_config.seed, Some(42));
    }
}
