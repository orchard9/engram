//! Parallel Activation Spreading with Cognitive Architecture
//!
//! This module implements high-performance lock-free parallel activation spreading
//! that replaces the simple temporal-proximity spreading with sophisticated
//! work-stealing graph traversal engine following biological principles.

use crate::{Confidence, Memory};
use crossbeam_deque::{Injector, Stealer, Worker};
use crossbeam_queue::SegQueue;
use dashmap::DashMap;
use atomic_float::AtomicF32;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

// pub mod parallel;  // Complex work-stealing implementation - disabled for now
pub mod simple_parallel;
pub mod queue;
pub mod traversal;
pub mod accumulator;
// pub mod cycle_detector;
// pub mod memory_pool;

// Note: HNSW integration will be added once Task 002 is complete
// pub mod hnsw_integration;

// pub mod deterministic;

/// Node identifier for graph traversal
pub type NodeId = String;

/// Lock-free activation record with atomic operations  
#[repr(align(64))] // Cache line alignment
pub struct ActivationRecord {
    pub node_id: NodeId,
    pub activation: AtomicF32,        // Current activation level
    pub timestamp: AtomicU64,         // Last update timestamp for ordering
    pub decay_rate: f32,              // Node-specific decay coefficient
    pub visits: AtomicUsize,          // Visit count for cycle detection
    pub source_count: AtomicUsize,    // Number of pending source updates
}

impl ActivationRecord {
    pub fn new(node_id: NodeId, decay_rate: f32) -> Self {
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
                Err(_) => continue, // Retry on contention
            }
        }
    }
    
    pub fn get_activation(&self) -> f32 {
        self.activation.load(Ordering::Relaxed)
    }
    
    pub fn reset(&self) {
        self.activation.store(0.0, Ordering::Relaxed);
        self.visits.store(0, Ordering::Relaxed);
        self.source_count.store(0, Ordering::Relaxed);
    }
}

/// Work-stealing task for parallel activation spreading
#[derive(Clone, Debug)]
pub struct ActivationTask {
    pub target_node: NodeId,
    pub source_activation: f32,
    pub edge_weight: f32,
    pub decay_factor: f32,
    pub depth: u16,                   // Current spreading depth
    pub max_depth: u16,               // Maximum allowed depth
}

impl ActivationTask {
    pub fn new(
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
    
    pub fn contribution(&self) -> f32 {
        self.source_activation * self.edge_weight * self.decay_factor
    }
    
    pub fn should_continue(&self) -> bool {
        self.depth < self.max_depth && self.contribution() > 0.01
    }
}

/// Cache-optimized weighted edge with compression
#[derive(Clone, Debug)]
pub struct WeightedEdge {
    pub target: NodeId,
    pub weight: f32,
    pub edge_type: EdgeType,
}

/// Edge types for Dale's law compliance
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum EdgeType {
    Excitatory,   // Positive influence
    Inhibitory,   // Negative influence  
    Modulatory,   // Context-dependent
}

impl Default for EdgeType {
    fn default() -> Self {
        Self::Excitatory
    }
}

/// Decay function implementations for different spreading dynamics
#[derive(Clone, Debug)]
pub enum DecayFunction {
    Exponential { rate: f32 },
    PowerLaw { exponent: f32 },
    Linear { slope: f32 },
    Custom { func: fn(u16) -> f32 },
}

impl DecayFunction {
    pub fn apply(&self, depth: u16) -> f32 {
        match self {
            Self::Exponential { rate } => (-rate * depth as f32).exp(),
            Self::PowerLaw { exponent } => (depth as f32 + 1.0).powf(-exponent),
            Self::Linear { slope } => (1.0 - slope * depth as f32).max(0.0),
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
    pub total_activations: AtomicU64,
    pub cache_hits: AtomicU64,
    pub cache_misses: AtomicU64,
    pub work_steals: AtomicU64,
    pub cycles_detected: AtomicU64,
    pub average_latency: AtomicU64, // In nanoseconds
    pub peak_memory_usage: AtomicU64,
    pub parallel_efficiency: AtomicF32,
}

impl SpreadingMetrics {
    pub fn cache_hit_rate(&self) -> f32 {
        let hits = self.cache_hits.load(Ordering::Relaxed) as f32;
        let misses = self.cache_misses.load(Ordering::Relaxed) as f32;
        if hits + misses > 0.0 {
            hits / (hits + misses)
        } else {
            0.0
        }
    }
    
    pub fn work_stealing_rate(&self) -> f32 {
        let steals = self.work_steals.load(Ordering::Relaxed) as f32;
        let total = self.total_activations.load(Ordering::Relaxed) as f32;
        if total > 0.0 {
            steals / total
        } else {
            0.0
        }
    }
    
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
    pub num_threads: usize,              // Worker thread count
    pub work_stealing_ratio: f32,        // Probability of stealing vs local work
    pub batch_size: usize,               // Tasks processed per batch
    
    // Memory management
    pub pool_initial_size: usize,        // Initial activation pool size
    pub cache_line_size: usize,          // Target cache line alignment
    pub numa_aware: bool,                // Enable NUMA-local allocation
    
    // Spreading dynamics
    pub max_depth: u16,                  // Maximum spreading depth
    pub decay_function: DecayFunction,   // Exponential, power-law, or custom
    pub threshold: f32,                  // Minimum activation threshold
    pub cycle_detection: bool,           // Enable cycle detection
    
    // Integration parameters
    pub simd_batch_size: usize,          // SIMD vector width for bulk operations
    pub prefetch_distance: usize,        // Cache prefetch lookahead
    
    // Determinism and reproducibility
    pub deterministic: bool,             // Enable deterministic mode
    pub seed: Option<u64>,               // RNG seed for reproducible results
    pub phase_sync_interval: Duration,   // Phase barrier sync interval
    
    // Performance monitoring
    pub enable_metrics: bool,            // Collect performance metrics
    pub trace_activation_flow: bool,     // Detailed activation tracing
}

impl Default for ParallelSpreadingConfig {
    fn default() -> Self {
        Self {
            num_threads: std::thread::available_parallelism()
                .map(|n| n.get())
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
    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),
    #[error("Memory allocation failed")]
    AllocationFailed,
    #[error("Cycle detected in graph: {0:?}")]
    CycleDetected(Vec<NodeId>),
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
    pub fn new() -> Self {
        Self {
            adjacency_list: DashMap::new(),
        }
    }
    
    pub fn add_edge(&self, source: NodeId, target: NodeId, weight: f32, edge_type: EdgeType) {
        let edge = WeightedEdge {
            target,
            weight,
            edge_type,
        };
        
        self.adjacency_list
            .entry(source)
            .or_insert_with(Vec::new)
            .push(edge);
    }
    
    pub fn get_neighbors(&self, node_id: &NodeId) -> Option<Vec<WeightedEdge>> {
        self.adjacency_list
            .get(node_id)
            .map(|edges| edges.clone())
    }
    
    pub fn node_count(&self) -> usize {
        self.adjacency_list.len()
    }
    
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
        let task = ActivationTask::new(
            "target".to_string(),
            0.8,
            0.5,
            0.7,
            2,
            5,
        );
        
        assert_eq!(task.contribution(), 0.8 * 0.5 * 0.7);
        assert!(task.should_continue());
        
        let task_too_deep = ActivationTask::new(
            "target".to_string(),
            0.8,
            0.5,
            0.7,
            5,
            5,
        );
        
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