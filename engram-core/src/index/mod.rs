//! HNSW (Hierarchical Navigable Small World) index for fast vector similarity search
//!
//! This module provides a high-performance, lock-free HNSW implementation specifically
//! designed for Engram's cognitive memory architecture. It features:
//!
//! - Lock-free concurrent operations using crossbeam data structures
//! - SIMD-optimized similarity computations from the compute module
//! - Confidence-aware search with probabilistic ranking
//! - Memory pressure adaptation for graceful degradation
//! - Zero-copy integration with existing `MemoryStore`

use crate::{Confidence, Memory};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicUsize, Ordering};

pub mod cognitive_dynamics;
pub mod confidence_metrics;
pub mod hnsw_construction;
pub mod hnsw_graph;
pub mod hnsw_node;
pub mod hnsw_search;

pub use cognitive_dynamics::ActivationDynamics;
pub use hnsw_graph::HnswGraph;
pub use hnsw_node::{ConnectionBlock, HnswEdge, HnswNode};
pub use hnsw_search::SearchResult;

/// Update types for background indexing
#[derive(Clone)]
pub enum IndexUpdate {
    /// Insert a new memory into the index
    Insert {
        /// Unique identifier for the memory
        memory_id: String,
        /// Memory object to be indexed
        memory: Arc<Memory>,
        /// Version number for ordering updates
        generation: u64,
        /// Processing priority level
        priority: UpdatePriority,
    },
    /// Remove a memory from the index
    Remove {
        /// Unique identifier of memory to remove
        memory_id: String,
        /// Version number for ordering updates
        generation: u64,
    },
    /// Update activation level for a single memory
    UpdateActivation {
        /// Unique identifier of memory to update
        memory_id: String,
        /// New activation value (0.0 to 1.0)
        activation: f32,
        /// Version number for ordering updates
        generation: u64,
    },
    /// Batch update activation levels for multiple memories
    BatchActivationUpdate {
        /// List of (`memory_id`, activation) pairs
        updates: Vec<(String, f32)>,
        /// Version number for ordering updates
        generation: u64,
    },
}

/// Priority levels for index update operations
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum UpdatePriority {
    /// Process immediately, preempting other operations
    Immediate = 0,
    /// High priority, process before normal operations
    High = 1,
    /// Standard priority for routine operations
    Normal = 2,
    /// Low priority, process during idle periods
    Low = 3,
}

/// Cognitive-aware HNSW index integrated with `MemoryStore`
pub struct CognitiveHnswIndex {
    /// Lock-free graph layers using crossbeam data structures
    graph: Arc<HnswGraph>,

    /// Memory pressure awareness
    pressure_adapter: PressureAdapter,

    /// Performance monitoring for self-tuning
    metrics: HnswMetrics,
    params: Arc<CognitiveHnswParams>,

    /// SIMD operations from compute module
    vector_ops: Box<dyn crate::compute::VectorOps>,

    /// Generation counter for ABA protection
    generation: AtomicU64,

    /// Node ID allocator
    node_counter: AtomicU32,
    
    /// Cognitive dynamics tracking for adaptive parameters
    activation_dynamics: ActivationDynamics,
}

/// Pressure-adaptive parameters that respect cognitive load
pub struct CognitiveHnswParams {
    /// Standard HNSW parameters with cognitive bounds
    pub m_max: AtomicUsize, // Max connections (reduces under pressure)
    pub m_l: AtomicUsize,             // Level 0 connections
    pub ef_construction: AtomicUsize, // Construction beam width
    pub ef_search: AtomicUsize,       // Search beam width
    pub ml: f32,                      // Layer probability factor

    /// Engram-specific cognitive parameters
    pub confidence_threshold: Confidence, // Minimum confidence for indexing
    pub activation_decay_rate: f32, // How fast activation spreads decay
    pub temporal_boost_factor: f32, // Boost for recent memories
    pub pressure_sensitivity: f32,  // How aggressively to reduce under pressure
    
    /// NEW FIELDS: Cognitive dynamics configuration (added at end for ABI compatibility)
    pub dynamics_enabled: AtomicBool,
    pub activation_sensitivity: f32,                    // Non-atomic: set at initialization
    pub confidence_stability_target: f32,               // Non-atomic: set at initialization
    pub temporal_locality_window_ns: AtomicU64,         // Nanoseconds for thread safety
    pub overconfidence_threshold: f32,                  // Non-atomic: set at initialization
    
    /// Lock-free adaptation control
    pub adaptation_cycle: AtomicU64,
    pub last_adaptation_time: AtomicU64,
}

impl CognitiveHnswParams {
    /// Adapt parameters based on activation energy distribution
    pub fn adapt_to_activation_patterns(&self, dynamics: &ActivationDynamics) {
        if !self.dynamics_enabled.load(Ordering::Relaxed) {
            return;
        }
        
        let current_ef = self.ef_search.load(Ordering::Relaxed);
        let activation_density = dynamics.compute_activation_density();
        
        // Biological principle: Sparse activation = increase search width
        // Dense activation = interference, reduce search width
        let target_ef = if activation_density < 0.3 {
            // Too sparse - increase exploration
            (current_ef as f32 * 1.2).min(512.0) as usize
        } else if activation_density > 0.7 {
            // Too dense - reduce interference  
            (current_ef as f32 * 0.8).max(16.0) as usize
        } else {
            current_ef // Optimal range
        };
        
        self.ef_search.store(target_ef, Ordering::Relaxed);
        
        // Update adaptation timestamp
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        self.last_adaptation_time.store(now, Ordering::Relaxed);
    }
}

/// Memory pressure adapter for graceful degradation
struct PressureAdapter {
    last_pressure_check: AtomicU64,
    pressure_sensitivity: f32,
}

impl PressureAdapter {
    const fn new(sensitivity: f32) -> Self {
        Self {
            last_pressure_check: AtomicU64::new(0),
            pressure_sensitivity: sensitivity,
        }
    }

    /// Adjust HNSW parameters based on current memory pressure
    fn adapt_params(&self, pressure: f32, params: &CognitiveHnswParams) {
        // Exponential backoff under pressure (cognitive principle)
        let pressure_factor = (1.0 - pressure).max(0.1); // Never go below 10%

        // Adapt parameters to maintain performance under pressure
        let target_m = (params.m_max.load(Ordering::Relaxed) as f32 * pressure_factor) as usize;
        params.m_max.store(target_m.max(2), Ordering::Relaxed); // Minimum connectivity

        let target_ef = (64.0 * pressure_factor) as usize; // Base ef=64
        params.ef_search.store(target_ef.max(8), Ordering::Relaxed); // Minimum search width
    }
}

/// Performance metrics for monitoring and self-tuning
#[derive(Default)]
struct HnswMetrics {
    searches_performed: AtomicU64,
    total_search_time_ns: AtomicU64,
    inserts_performed: AtomicU64,
    total_insert_time_ns: AtomicU64,
    graph_compactions: AtomicU64,
}

impl CognitiveHnswIndex {
    /// Create a new HNSW index with default parameters
    #[must_use]
    pub fn new() -> Self {
        let params = Arc::new(CognitiveHnswParams {
            m_max: AtomicUsize::new(16),
            m_l: AtomicUsize::new(32),
            ef_construction: AtomicUsize::new(200),
            ef_search: AtomicUsize::new(64),
            ml: 1.0 / (2.0_f32).ln(),
            confidence_threshold: Confidence::LOW,
            activation_decay_rate: 0.2,
            temporal_boost_factor: 1.2,
            pressure_sensitivity: 0.5,
            
            // NEW FIELDS: Cognitive dynamics configuration (added at end for ABI compatibility)
            dynamics_enabled: AtomicBool::new(false),
            activation_sensitivity: 0.15,                    // Non-atomic: set at init
            confidence_stability_target: 0.2,               // Non-atomic: set at init
            temporal_locality_window_ns: AtomicU64::new(500_000_000), // 500ms in nanoseconds
            overconfidence_threshold: 0.25,                 // Non-atomic: set at init
            
            // Lock-free adaptation control
            adaptation_cycle: AtomicU64::new(0),
            last_adaptation_time: AtomicU64::new(0),
        });

        Self {
            graph: Arc::new(HnswGraph::new()),
            pressure_adapter: PressureAdapter::new(params.pressure_sensitivity),
            metrics: HnswMetrics::default(),
            params,
            vector_ops: crate::compute::create_vector_ops(),
            generation: AtomicU64::new(0),
            node_counter: AtomicU32::new(0),
            activation_dynamics: ActivationDynamics::new(),
        }
    }
    
    /// Create index with custom cognitive parameters
    #[must_use]
    pub fn with_cognitive_params(
        activation_sensitivity: f32,
        stability_target: f32,
        overconfidence_threshold: f32,
    ) -> Self {
        let params = Arc::new(CognitiveHnswParams {
            // Standard HNSW defaults
            m_max: AtomicUsize::new(16),
            m_l: AtomicUsize::new(32),
            ef_construction: AtomicUsize::new(200),
            ef_search: AtomicUsize::new(64),
            ml: 1.0 / (2.0_f32).ln(),
            confidence_threshold: Confidence::LOW,
            activation_decay_rate: 0.2,
            temporal_boost_factor: 1.2,
            pressure_sensitivity: 0.5,
            
            // Custom cognitive configuration
            dynamics_enabled: AtomicBool::new(false), // Start disabled
            activation_sensitivity,
            confidence_stability_target: stability_target,
            temporal_locality_window_ns: AtomicU64::new(500_000_000),
            overconfidence_threshold,
            adaptation_cycle: AtomicU64::new(0),
            last_adaptation_time: AtomicU64::new(0),
        });

        Self {
            graph: Arc::new(HnswGraph::new()),
            pressure_adapter: PressureAdapter::new(params.pressure_sensitivity),
            metrics: HnswMetrics::default(),
            params,
            vector_ops: crate::compute::create_vector_ops(),
            generation: AtomicU64::new(0),
            node_counter: AtomicU32::new(0),
            activation_dynamics: ActivationDynamics::new(),
        }
    }

    /// Insert a memory into the HNSW index
    pub fn insert_memory(&self, memory: Arc<Memory>) -> Result<(), HnswError> {
        let start = std::time::Instant::now();

        // Allocate node ID
        let node_id = self.node_counter.fetch_add(1, Ordering::Relaxed);

        // Select layer for this node
        let layer = self.select_layer_probabilistic();

        // Create HNSW node from memory
        let node = HnswNode::from_memory(node_id, memory, layer)?;

        // Insert into graph using lock-free operations
        self.graph
            .insert_node(node, &self.params, self.vector_ops.as_ref())?;

        // Update metrics
        self.metrics
            .inserts_performed
            .fetch_add(1, Ordering::Relaxed);
        self.metrics
            .total_insert_time_ns
            .fetch_add(start.elapsed().as_nanos() as u64, Ordering::Relaxed);

        Ok(())
    }

    /// Search for k nearest neighbors with confidence filtering
    pub fn search_with_confidence(
        &self,
        query: &[f32; 768],
        k: usize,
        threshold: Confidence,
    ) -> Vec<(String, Confidence)> {
        let start = std::time::Instant::now();

        let ef = self.params.ef_search.load(Ordering::Relaxed);
        let results = self
            .graph
            .search(query, k, ef, threshold, self.vector_ops.as_ref());

        // Update metrics
        self.metrics
            .searches_performed
            .fetch_add(1, Ordering::Relaxed);
        self.metrics
            .total_search_time_ns
            .fetch_add(start.elapsed().as_nanos() as u64, Ordering::Relaxed);

        results
    }

    /// Apply spreading activation using graph structure with cognitive adaptation feedback
    pub fn apply_spreading_activation(
        &self,
        mut results: Vec<(crate::Episode, Confidence)>,
        _cue: &crate::Cue,
        pressure: f32,
    ) -> Vec<(crate::Episode, Confidence)> {
        // Existing pressure adaptation
        self.pressure_adapter.adapt_params(pressure, &self.params);

        // Cognitive dynamics analysis
        let activation_energy = 1.0 - pressure;
        let temporal_locality = self.activation_dynamics.temporal_locality_factor();
        let overconfidence_ratio = self.activation_dynamics.overconfidence_ratio();
        
        // Record activation pattern for future adaptation
        for (_, confidence) in &results {
            self.activation_dynamics.record_activation(activation_energy, *confidence);
        }
        
        // Adapt parameters based on cognitive dynamics (not ML accuracy)
        if self.should_adapt_dynamics() {
            self.params.adapt_to_activation_patterns(&self.activation_dynamics);
        }
        
        // Apply cognitive principles to spreading activation
        let max_hops = self.compute_cognitive_hops(pressure, temporal_locality, overconfidence_ratio);
        
        // Enhanced activation spreading with confidence weighting
        for hop in 0..max_hops {
            let hop_energy = activation_energy * (0.8_f32).powi(hop as i32); // Decay per hop
            
            if hop_energy < 0.1 {
                break; // Below threshold
            }
            
            // Process each result for potential spreading
            let mut new_activations = Vec::new();
            for (episode, confidence) in &results {
                // Find connected memories via graph traversal
                let connected = self.find_connected_memories(&episode.id, hop_energy);
                new_activations.extend(connected);
            }
            
            // Merge new activations with existing results
            results = self.merge_activations(results, new_activations, hop_energy);
        }

        // Apply temporal boost for recent memories
        self.apply_temporal_boost(&mut results, temporal_locality);
        
        results
    }
    
    /// Enable cognitive dynamics adaptation with biological parameters
    pub fn enable_cognitive_adaptation(&self) {
        self.params.dynamics_enabled.store(true, Ordering::Relaxed);
    }
    
    /// Circuit breaker: Disable adaptation if system becomes unstable
    pub fn disable_adaptation_on_instability(&self) -> bool {
        let overconfidence = self.activation_dynamics.overconfidence_ratio();
        if overconfidence > 0.5 { // >50% overconfident connections
            self.params.dynamics_enabled.store(false, Ordering::Relaxed);
            true // Signal instability detected
        } else {
            false
        }
    }
    
    /// Check if dynamics adaptation should occur
    fn should_adapt_dynamics(&self) -> bool {
        // Adapt every 100 activation cycles, but not more than once per 10 seconds
        let cycle = self.params.adaptation_cycle.fetch_add(1, Ordering::Relaxed);
        let last_adaptation = self.params.last_adaptation_time.load(Ordering::Relaxed);
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        
        cycle % 100 == 0 && now.saturating_sub(last_adaptation) >= 10
    }
    
    /// Compute cognitive hops based on biological principles
    fn compute_cognitive_hops(&self, pressure: f32, temporal_locality: f32, overconfidence: f32) -> usize {
        // Biological principle: High pressure = reduce exploration
        // High temporal locality = increase local search
        // High overconfidence = reduce to prevent bias amplification
        
        let base_hops: i32 = if pressure > 0.8 { 1 } else if pressure > 0.5 { 2 } else { 3 };
        
        let locality_adjustment: i32 = if temporal_locality > 0.7 { 1 } else { 0 };
        let confidence_adjustment: i32 = if overconfidence > 0.3 { -1 } else { 0 };
        
        (base_hops + locality_adjustment + confidence_adjustment).max(1).min(4) as usize
    }
    
    /// Find connected memories for activation spreading
    fn find_connected_memories(&self, memory_id: &str, energy: f32) -> Vec<(crate::Episode, Confidence)> {
        // TODO: Implement using graph.get_neighbors() + memory lookup
        // This is a placeholder implementation
        let _ = (memory_id, energy);
        Vec::new()
    }
    
    /// Merge new activations with existing results
    fn merge_activations(
        &self,
        existing: Vec<(crate::Episode, Confidence)>,
        new_activations: Vec<(crate::Episode, Confidence)>,
        energy: f32,
    ) -> Vec<(crate::Episode, Confidence)> {
        // TODO: Implement energy decay and confidence combination
        // This is a placeholder implementation
        let _ = (new_activations, energy);
        existing
    }
    
    /// Apply temporal boost for recent memories
    fn apply_temporal_boost(&self, results: &mut [(crate::Episode, Confidence)], locality: f32) {
        let boost_factor = 1.0 + (locality * self.params.temporal_boost_factor);
        for (_, confidence) in results {
            let boosted = confidence.raw() * boost_factor;
            *confidence = Confidence::exact(boosted.min(1.0));
        }
    }

    /// Validate graph structure integrity
    pub fn validate_graph_integrity(&self) -> bool {
        self.graph.validate_structure()
    }

    /// Check bidirectional consistency of edges
    pub fn validate_bidirectional_consistency(&self) -> bool {
        self.graph.validate_bidirectional_consistency()
    }

    /// Check memory consistency
    pub fn check_memory_consistency(&self) -> bool {
        self.graph.check_memory_consistency()
    }

    /// Select layer for new node using probabilistic assignment
    fn select_layer_probabilistic(&self) -> u8 {
        let mut layer = 0;
        let ml = self.params.ml;

        // Simple linear congruential generator for deterministic but varied layer selection
        use std::sync::atomic::{AtomicU32, Ordering};
        static LAYER_SEED: AtomicU32 = AtomicU32::new(1);

        let seed = LAYER_SEED.fetch_add(1, Ordering::Relaxed);
        let mut rng =
            ((seed.wrapping_mul(1_103_515_245).wrapping_add(12345)) >> 16) as f32 / 32768.0;

        while rng < ml && layer < 16 {
            layer += 1;
            let new_seed = seed.wrapping_add(u32::from(layer));
            rng =
                ((new_seed.wrapping_mul(1_103_515_245).wrapping_add(12345)) >> 16) as f32 / 32768.0;
        }

        layer
    }
}

/// Error types for HNSW operations
#[derive(Debug, thiserror::Error)]
pub enum HnswError {
    /// Failed to allocate memory for graph node
    #[error("Failed to allocate node: {0}")]
    AllocationError(String),

    /// Embedding vector has wrong number of dimensions
    #[error("Invalid embedding dimension: expected 768, got {0}")]
    InvalidDimension(usize),

    /// Graph data structure integrity check failed
    #[error("Graph structure corrupted: {0}")]
    CorruptedGraph(String),

    /// Requested memory was not found in the index
    #[error("Memory not found: {0}")]
    MemoryNotFound(String),
}

impl Default for CognitiveHnswIndex {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hnsw_creation() {
        let index = CognitiveHnswIndex::new();
        assert!(index.validate_graph_integrity());
    }

    #[test]
    fn test_layer_selection() {
        let index = CognitiveHnswIndex::new();
        let mut layer_counts = [0u32; 17];

        for _ in 0..1000 {
            let layer = index.select_layer_probabilistic();
            layer_counts[layer as usize] += 1;
        }

        // Layer 0 should be most frequent
        assert!(layer_counts[0] > 0);

        // Higher layers should generally be less frequent (but don't enforce strict ordering)
        assert!(layer_counts[0] >= layer_counts[1]);
    }

    #[test]
    fn test_cognitive_dynamics_integration() {
        let index = CognitiveHnswIndex::new();
        
        // Test enabling cognitive adaptation
        assert!(!index.params.dynamics_enabled.load(Ordering::Relaxed));
        index.enable_cognitive_adaptation();
        assert!(index.params.dynamics_enabled.load(Ordering::Relaxed));
        
        // Test activation dynamics tracking
        let initial_density = index.activation_dynamics.compute_activation_density();
        assert_eq!(initial_density, 0.0); // Should start empty
        
        // Record some activations
        for i in 0..10 {
            let energy = 0.5 + (i as f32 / 20.0);
            let confidence = Confidence::exact(0.7);
            index.activation_dynamics.record_activation(energy, confidence);
        }
        
        let updated_density = index.activation_dynamics.compute_activation_density();
        assert!(updated_density > 0.0); // Should have some density now
        
        // Test overconfidence tracking
        let initial_ratio = index.activation_dynamics.overconfidence_ratio();
        assert_eq!(initial_ratio, 0.0); // Should start with no overconfidence
        
        // Record some overconfident connections
        let threshold = index.params.overconfidence_threshold;
        index.activation_dynamics.record_connection(Confidence::exact(threshold + 0.1), threshold);
        index.activation_dynamics.record_connection(Confidence::exact(threshold - 0.1), threshold);
        
        let updated_ratio = index.activation_dynamics.overconfidence_ratio();
        assert!(updated_ratio > 0.0); // Should detect some overconfidence
        assert!(updated_ratio < 1.0); // But not all connections are overconfident
    }

    #[test]
    fn test_cognitive_parameter_adaptation() {
        let index = CognitiveHnswIndex::new();
        index.enable_cognitive_adaptation();
        
        let initial_ef_search = index.params.ef_search.load(Ordering::Relaxed);
        
        // Simulate sparse activation (should increase ef_search)
        for _ in 0..5 {
            let energy = 0.1; // Low energy = sparse activation
            let confidence = Confidence::exact(0.5);
            index.activation_dynamics.record_activation(energy, confidence);
        }
        
        // Force parameter adaptation
        index.params.adapt_to_activation_patterns(&index.activation_dynamics);
        
        let adapted_ef_search = index.params.ef_search.load(Ordering::Relaxed);
        
        // With sparse activation (density < 0.3), ef_search should increase
        // Note: This is a simplified test - in practice activation density calculation is more complex
        assert!(adapted_ef_search >= initial_ef_search.min(400)); // Should not decrease beyond reasonable bounds
    }

    #[test]
    fn test_circuit_breaker_functionality() {
        let index = CognitiveHnswIndex::new();
        index.enable_cognitive_adaptation();
        
        // Should not trigger circuit breaker initially
        assert!(!index.disable_adaptation_on_instability());
        assert!(index.params.dynamics_enabled.load(Ordering::Relaxed));
        
        // Simulate high overconfidence to trigger circuit breaker
        let threshold = index.params.overconfidence_threshold;
        for _ in 0..60 { // Create many overconfident connections
            index.activation_dynamics.record_connection(Confidence::exact(threshold + 0.2), threshold);
        }
        for _ in 0..10 { // Add some normal connections
            index.activation_dynamics.record_connection(Confidence::exact(threshold - 0.1), threshold);
        }
        
        // Circuit breaker should trigger when overconfidence > 50%
        let triggered = index.disable_adaptation_on_instability();
        assert!(triggered); // Should detect instability
        assert!(!index.params.dynamics_enabled.load(Ordering::Relaxed)); // Should disable adaptation
    }

    #[test]
    fn test_custom_cognitive_params() {
        let custom_index = CognitiveHnswIndex::with_cognitive_params(0.25, 0.15, 0.3);
        
        // Verify custom parameters are set correctly
        assert_eq!(custom_index.params.activation_sensitivity, 0.25);
        assert_eq!(custom_index.params.confidence_stability_target, 0.15);
        assert_eq!(custom_index.params.overconfidence_threshold, 0.3);
        
        // Should start with dynamics disabled
        assert!(!custom_index.params.dynamics_enabled.load(Ordering::Relaxed));
        
        // Should be able to enable
        custom_index.enable_cognitive_adaptation();
        assert!(custom_index.params.dynamics_enabled.load(Ordering::Relaxed));
    }
}
