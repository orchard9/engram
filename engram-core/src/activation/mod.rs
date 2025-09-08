//! Parallel Activation Spreading with Cognitive Architecture
//!
//! This module implements high-performance lock-free parallel activation spreading
//! that replaces the simple temporal-proximity spreading with sophisticated
//! work-stealing graph traversal engine following biological principles.

use atomic_float::AtomicF32;
use dashmap::DashMap;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

pub mod accumulator;
pub mod queue;
pub mod simple_parallel;
pub mod traversal;
pub mod cycle_detector;
// pub mod memory_pool; // TODO: Implement memory pool module

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
    /// Last update timestamp for ordering
    pub timestamp: AtomicU64,
    /// Node-specific decay coefficient
    pub decay_rate: f32,
    /// Visit count for cycle detection
    pub visits: AtomicUsize,
    /// Number of pending source updates
    pub source_count: AtomicUsize,
}

impl ActivationRecord {
    /// Create a new activation record with zero initial activation
    #[must_use]
    pub const fn new(node_id: NodeId, decay_rate: f32) -> Self {
        Self {
            node_id,
            activation: AtomicF32::new(0.0),
            timestamp: AtomicU64::new(0),
            decay_rate,
            visits: AtomicUsize::new(0),
            source_count: AtomicUsize::new(0),
        }
    }

    /// Atomically accumulate activation with memory ordering
    pub fn accumulate_activation(&self, contribution: f32) -> bool {
        loop {
            let current = self.activation.load(Ordering::Relaxed);
            let new_activation = (current + contribution).min(1.0);

            // Only update if above threshold to reduce contention
            if new_activation < 0.01 {
                return false;
            }

            match self.activation.compare_exchange_weak(
                current,
                new_activation,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => {
                    // Update timestamp for cycle detection
                    let now = Instant::now().elapsed().as_nanos() as u64;
                    self.timestamp.store(now, Ordering::Relaxed);
                    return true;
                }
                Err(_) => {} // Retry on contention
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
        self.visits.store(0, Ordering::Relaxed);
        self.source_count.store(0, Ordering::Relaxed);
    }

    /// Set activation level directly
    pub fn set_activation(&self, value: f32) {
        self.activation.store(value.clamp(0.0, 1.0), Ordering::Relaxed);
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
}

impl ActivationTask {
    /// Create a new activation spreading task
    #[must_use]
    pub const fn new(
        target_node: NodeId,
        source_activation: f32,
        edge_weight: f32,
        decay_factor: f32,
        depth: u16,
        max_depth: u16,
    ) -> Self {
        Self {
            target_node,
            source_activation,
            edge_weight,
            decay_factor,
            depth,
            max_depth,
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
        self.depth < self.max_depth && self.contribution() > 0.01
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
    /// Peak memory usage in bytes
    pub peak_memory_usage: AtomicU64,
    /// Parallel efficiency ratio (0.0 to 1.0)
    pub parallel_efficiency: AtomicF32,
}

impl SpreadingMetrics {
    /// Calculate cache hit rate (0.0 to 1.0)
    pub fn cache_hit_rate(&self) -> f32 {
        let hits = self.cache_hits.load(Ordering::Relaxed) as f32;
        let misses = self.cache_misses.load(Ordering::Relaxed) as f32;
        if hits + misses > 0.0 {
            hits / (hits + misses)
        } else {
            0.0
        }
    }

    /// Calculate work stealing rate (0.0 to 1.0)
    pub fn work_stealing_rate(&self) -> f32 {
        let steals = self.work_steals.load(Ordering::Relaxed) as f32;
        let total = self.total_activations.load(Ordering::Relaxed) as f32;
        if total > 0.0 { steals / total } else { 0.0 }
    }

    /// Reset all metrics to zero
    pub fn reset(&self) {
        self.total_activations.store(0, Ordering::Relaxed);
        self.cache_hits.store(0, Ordering::Relaxed);
        self.cache_misses.store(0, Ordering::Relaxed);
        self.work_steals.store(0, Ordering::Relaxed);
        self.cycles_detected.store(0, Ordering::Relaxed);
        self.average_latency.store(0, Ordering::Relaxed);
        self.peak_memory_usage.store(0, Ordering::Relaxed);
        self.parallel_efficiency.store(0.0, Ordering::Relaxed);
    }
}

/// Configuration for high-performance parallel spreading
#[derive(Clone, Debug)]
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

    // Integration parameters
    /// SIMD vector width for bulk operations
    pub simd_batch_size: usize,
    /// Cache prefetch lookahead distance
    pub prefetch_distance: usize,

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
            simd_batch_size: 8,
            prefetch_distance: 64,
            deterministic: false,
            seed: None,
            phase_sync_interval: Duration::from_millis(10),
            enable_metrics: true,
            trace_activation_flow: false,
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

/// Simple memory graph for testing (will be replaced with HNSW integration)
#[derive(Default)]
pub struct MemoryGraph {
    adjacency_list: DashMap<NodeId, Vec<WeightedEdge>>,
}

impl MemoryGraph {
    /// Create a new empty memory graph
    #[must_use]
    pub fn new() -> Self {
        Self {
            adjacency_list: DashMap::new(),
        }
    }

    /// Add a weighted edge between two nodes in the memory graph
    pub fn add_edge(&self, source: NodeId, target: NodeId, weight: f32, edge_type: EdgeType) {
        let edge = WeightedEdge {
            target,
            weight,
            edge_type,
        };

        self.adjacency_list.entry(source).or_default().push(edge);
    }

    /// Get all neighbors of a given node
    #[must_use]
    pub fn get_neighbors(&self, node_id: &NodeId) -> Option<Vec<WeightedEdge>> {
        self.adjacency_list.get(node_id).map(|edges| edges.clone())
    }

    /// Get the total number of nodes in the graph
    #[must_use]
    pub fn node_count(&self) -> usize {
        self.adjacency_list.len()
    }

    /// Get the total number of edges in the graph
    #[must_use]
    pub fn edge_count(&self) -> usize {
        self.adjacency_list
            .iter()
            .map(|entry| entry.value().len())
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_activation_record_creation() {
        let record = ActivationRecord::new("test_node".to_string(), 0.1);
        assert_eq!(record.node_id, "test_node");
        assert_eq!(record.decay_rate, 0.1);
        assert_eq!(record.get_activation(), 0.0);
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
        assert_eq!(record.get_activation(), 0.0);

        // Above threshold should update
        assert!(record.accumulate_activation(0.02));
        assert!((record.get_activation() - 0.02).abs() < 1e-6);
    }

    #[test]
    fn test_activation_task() {
        let task = ActivationTask::new("target".to_string(), 0.8, 0.5, 0.7, 2, 5);

        assert_eq!(task.contribution(), 0.8 * 0.5 * 0.7);
        assert!(task.should_continue());

        let task_too_deep = ActivationTask::new("target".to_string(), 0.8, 0.5, 0.7, 5, 5);

        assert!(!task_too_deep.should_continue());
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
        assert_eq!(linear.apply(5), 0.0); // Should clamp to 0
    }

    #[test]
    fn test_memory_graph() {
        let graph = MemoryGraph::new();

        graph.add_edge(
            "node1".to_string(),
            "node2".to_string(),
            0.8,
            EdgeType::Excitatory,
        );

        graph.add_edge(
            "node1".to_string(),
            "node3".to_string(),
            0.3,
            EdgeType::Inhibitory,
        );

        let neighbors = graph.get_neighbors(&"node1".to_string()).unwrap();
        assert_eq!(neighbors.len(), 2);
        assert_eq!(neighbors[0].target, "node2");
        assert_eq!(neighbors[0].weight, 0.8);
        assert_eq!(neighbors[0].edge_type, EdgeType::Excitatory);

        assert_eq!(graph.node_count(), 1);
        assert_eq!(graph.edge_count(), 2);
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
        assert_eq!(metrics.cache_hit_rate(), 0.0);
        assert_eq!(metrics.work_stealing_rate(), 0.0);
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
        assert_eq!(custom_config.threshold, 0.05);
        assert!(custom_config.deterministic);
        assert_eq!(custom_config.seed, Some(42));
    }
}
