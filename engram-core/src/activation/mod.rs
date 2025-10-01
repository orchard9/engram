//! Parallel Activation Spreading with Cognitive Architecture
//!
//! This module implements high-performance lock-free parallel activation spreading
//! that replaces the simple temporal-proximity spreading with sophisticated
//! work-stealing graph traversal engine following biological principles.

use atomic_float::AtomicF32;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

/// Latency budget management for activation spreading
pub mod latency_budget;
/// Storage-aware activation spreading algorithms
pub mod storage_aware;

pub mod accumulator;
pub mod confidence_aggregation;
pub mod cycle_detector;
/// GPU abstraction layer for acceleration
pub mod gpu_interface;
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
/// SIMD-optimized batch activation spreading
pub mod simd_optimization;
/// Configuration for similarity-based activation
pub mod similarity_config;
pub mod traversal;

pub use confidence_aggregation::{
    ConfidenceAggregationOutcome, ConfidenceAggregator, ConfidenceContribution, ConfidencePath,
};
pub use gpu_interface::{
    AdaptiveConfig, AdaptiveSpreadingEngine, BatchMetadata, CpuFallback, GPUActivationBatch,
    GPUSpreadingInterface, GpuCapabilities, GpuLaunchFuture, MockGpuInterface,
};
pub use multi_cue::CueAggregationStrategy;
pub use parallel::ParallelSpreadingEngine;
pub use recall::{CognitiveRecall, CognitiveRecallBuilder, RankedMemory, RecallConfig, RecallMetrics, RecallMode};
pub use scheduler::{SchedulerSnapshot, TierAwareSpreadingScheduler, TierQueueStateSnapshot};
#[cfg(feature = "hnsw_index")]
pub use seeding::{
    ActivationTier, SeededActivation, SeedingError, SeedingOutcome, VectorActivationSeeder,
};
pub use similarity_config::SimilarityConfig;
pub mod memory_pool;
pub use memory_pool::{ActivationMemoryPool, LocalMemoryPool, PoolStats};

// HNSW integration
// #[cfg(feature = "hnsw_index")]
// pub mod hnsw_integration; // TODO: Implement HNSW integration module

/// Node identifier for graph traversal
pub type NodeId = String;

/// Lock-free activation record with atomic operations  
#[repr(align(64))] // Cache line alignment
pub struct ActivationRecord {
    /// Unique identifier for this node
    pub node_id: NodeId,
    /// Current activation level (0.0 to 1.0)
    pub activation: AtomicF32,
    /// Confidence score accumulated for this activation
    pub confidence: AtomicF32,
    /// Last update timestamp for ordering
    pub timestamp: AtomicU64,
    /// Node-specific decay coefficient
    pub decay_rate: f32,
    /// Visit count for cycle detection
    pub visits: AtomicUsize,
    /// Number of pending source updates
    pub source_count: AtomicUsize,
    /// Optional storage tier classification for this activation.
    pub storage_tier: Option<storage_aware::StorageTier>,
}

impl ActivationRecord {
    /// Create a new activation record with zero initial activation
    #[must_use]
    pub const fn new(node_id: NodeId, decay_rate: f32) -> Self {
        Self {
            node_id,
            activation: AtomicF32::new(0.0),
            confidence: AtomicF32::new(0.0),
            timestamp: AtomicU64::new(0),
            decay_rate,
            visits: AtomicUsize::new(0),
            source_count: AtomicUsize::new(0),
            storage_tier: None,
        }
    }

    /// Atomically accumulate activation with memory ordering
    pub fn accumulate_activation(&self, contribution: f32) -> bool {
        let threshold = self
            .storage_tier
            .map_or(0.01, storage_aware::StorageTier::activation_threshold);
        loop {
            let current = self.activation.load(Ordering::Relaxed);
            let new_activation = (current + contribution).min(1.0);

            // Only update if above threshold to reduce contention
            if new_activation < threshold {
                return false;
            }

            if self
                .activation
                .compare_exchange_weak(
                    current,
                    new_activation,
                    Ordering::Relaxed,
                    Ordering::Relaxed,
                )
                .is_ok()
            {
                self.confidence.store(new_activation, Ordering::Relaxed);
                // Update timestamp for cycle detection
                let now = Instant::now()
                    .elapsed()
                    .as_nanos()
                    .try_into()
                    .unwrap_or(u64::MAX);
                self.timestamp.store(now, Ordering::Relaxed);
                return true;
            }
        }
    }

    /// Get current activation level
    pub fn get_activation(&self) -> f32 {
        self.activation.load(Ordering::Relaxed)
    }

    /// Reset activation and counters to zero
    pub fn reset(&self) {
        self.activation.store(0.0, Ordering::Relaxed);
        self.confidence.store(0.0, Ordering::Relaxed);
        self.visits.store(0, Ordering::Relaxed);
        self.source_count.store(0, Ordering::Relaxed);
    }

    /// Set activation level directly
    pub fn set_activation(&self, value: f32) {
        self.activation
            .store(value.clamp(0.0, 1.0), Ordering::Relaxed);
        self.confidence
            .store(value.clamp(0.0, 1.0), Ordering::Relaxed);
    }

    /// Retrieve the current confidence score associated with the record.
    pub fn get_confidence(&self) -> f32 {
        self.confidence.load(Ordering::Relaxed)
    }

    /// Apply a multiplicative penalty to activation and confidence when a cycle is detected.
    pub fn apply_cycle_penalty(&self, factor: f32) {
        let penalty = factor.clamp(0.0, 1.0);

        // Scale activation
        loop {
            let current = self.activation.load(Ordering::Relaxed);
            let updated = (current * penalty).clamp(0.0, 1.0);
            if self
                .activation
                .compare_exchange_weak(current, updated, Ordering::Relaxed, Ordering::Relaxed)
                .is_ok()
            {
                break;
            }
        }

        // Scale confidence to mirror activation decay.
        loop {
            let current = self.confidence.load(Ordering::Relaxed);
            let updated = (current * penalty).clamp(0.0, 1.0);
            if self
                .confidence
                .compare_exchange_weak(current, updated, Ordering::Relaxed, Ordering::Relaxed)
                .is_ok()
            {
                break;
            }
        }
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
#[derive(Default, Debug)]
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
    /// Peak memory usage in bytes
    pub peak_memory_usage: AtomicU64,
    /// Parallel efficiency ratio (0.0 to 1.0)
    pub parallel_efficiency: AtomicF32,
    /// Tier-specific cycle detection counts
    pub cycle_counts: dashmap::DashMap<storage_aware::StorageTier, AtomicU64>,
}

impl SpreadingMetrics {
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
            pool_chunk_size: 8192,    // 8KB per chunk
            pool_max_chunks: 16,       // Max 128KB total
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

/// Default memory graph type for activation spreading
pub type MemoryGraph = UnifiedMemoryGraph<DashMapBackend>;

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
    UnifiedMemoryGraph::new(DashMapBackend::default(), config)
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
}

impl ActivationGraphExt for MemoryGraph {
    fn add_edge(&self, source: NodeId, target: NodeId, weight: f32, _edge_type: EdgeType) {
        // Convert NodeId (String) to Uuid for the backend
        use uuid::Uuid;
        let source_id = Uuid::new_v5(&Uuid::NAMESPACE_OID, source.as_bytes());
        let target_id = Uuid::new_v5(&Uuid::NAMESPACE_OID, target.as_bytes());

        // Add edge using the UnifiedMemoryGraph's add_edge method
        let _ = Self::add_edge(self, source_id, target_id, weight);

        // Edge type metadata will be handled in a future iteration
    }

    fn get_neighbors(&self, node_id: &NodeId) -> Option<Vec<WeightedEdge>> {
        use uuid::Uuid;
        let id = Uuid::new_v5(&Uuid::NAMESPACE_OID, node_id.as_bytes());

        Self::get_neighbors(self, &id).ok().map(|neighbors| {
            neighbors
                .into_iter()
                .map(|(neighbor_id, weight)| {
                    // Convert Uuid back to NodeId
                    WeightedEdge {
                        target: neighbor_id.to_string(),
                        weight,
                        edge_type: EdgeType::Excitatory, // Default for now
                    }
                })
                .collect()
        })
    }

    fn node_count(&self) -> usize {
        Self::count(self)
    }

    fn edge_count(&self) -> usize {
        // Use the all_edges method from UnifiedMemoryGraph
        Self::all_edges(self).map_or(0, |edges| edges.len())
    }
}

#[cfg(test)]
mod tests {
    use super::storage_aware::StorageTier;
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
        record.visits.fetch_add(2, AtomicOrdering::Relaxed);

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

        metrics.reset();
        assert!(metrics.cache_hit_rate().abs() < f32::EPSILON);
        assert!(metrics.work_stealing_rate().abs() < f32::EPSILON);
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
