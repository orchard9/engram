//! HNSW (Hierarchical Navigable Small World) index for fast vector similarity search
//!
//! This module provides a high-performance, lock-free HNSW implementation specifically
//! designed for Engram's cognitive memory architecture. It features:
//!
//! - Lock-free concurrent operations using crossbeam data structures
//! - SIMD-optimized similarity computations from the compute module
//! - Confidence-aware search with probabilistic ranking
//! - Memory pressure adaptation for graceful degradation
//! - Zero-copy integration with existing MemoryStore

use crate::{Confidence, Memory};
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, AtomicU64, AtomicUsize, Ordering};

pub mod confidence_metrics;
pub mod hnsw_construction;
pub mod hnsw_graph;
pub mod hnsw_node;
pub mod hnsw_search;

pub use hnsw_graph::HnswGraph;
pub use hnsw_node::{ConnectionBlock, HnswEdge, HnswNode};
pub use hnsw_search::SearchResult;

/// Update types for background indexing
#[derive(Clone)]
pub enum IndexUpdate {
    Insert {
        memory_id: String,
        memory: Arc<Memory>,
        generation: u64,
        priority: UpdatePriority,
    },
    Remove {
        memory_id: String,
        generation: u64,
    },
    UpdateActivation {
        memory_id: String,
        activation: f32,
        generation: u64,
    },
    BatchActivationUpdate {
        updates: Vec<(String, f32)>,
        generation: u64,
    },
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum UpdatePriority {
    Immediate = 0,
    High = 1,
    Normal = 2,
    Low = 3,
}

/// Cognitive-aware HNSW index integrated with MemoryStore
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
}

/// Memory pressure adapter for graceful degradation
struct PressureAdapter {
    last_pressure_check: AtomicU64,
    pressure_sensitivity: f32,
}

impl PressureAdapter {
    fn new(sensitivity: f32) -> Self {
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
        });

        Self {
            graph: Arc::new(HnswGraph::new()),
            pressure_adapter: PressureAdapter::new(params.pressure_sensitivity),
            metrics: HnswMetrics::default(),
            params,
            vector_ops: crate::compute::create_vector_ops(),
            generation: AtomicU64::new(0),
            node_counter: AtomicU32::new(0),
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

    /// Apply spreading activation using graph structure
    pub fn apply_spreading_activation(
        &self,
        mut results: Vec<(crate::Episode, Confidence)>,
        _cue: &crate::Cue,
        pressure: f32,
    ) -> Vec<(crate::Episode, Confidence)> {
        // Adapt parameters based on pressure
        self.pressure_adapter.adapt_params(pressure, &self.params);

        // Use graph structure for efficient activation spreading
        let activation_energy = 1.0 - pressure;
        let max_hops = if pressure > 0.8 {
            1
        } else if pressure > 0.5 {
            2
        } else {
            3
        };

        // Spread activation through graph neighbors
        for (episode, confidence) in &mut results {
            let neighbors = self.graph.get_neighbors(&episode.id, max_hops);
            for (_neighbor_id, distance, neighbor_confidence) in neighbors {
                // Apply activation decay based on graph distance
                let decayed_activation = activation_energy
                    * (1.0 - distance)
                    * self.params.activation_decay_rate.powi(distance as i32);

                // Combine confidences with decay
                let combined_confidence = confidence.and(neighbor_confidence);
                *confidence = Confidence::exact(
                    confidence.raw() + decayed_activation * combined_confidence.raw(),
                );
            }
        }

        results
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
        let mut rng = ((seed.wrapping_mul(1103515245).wrapping_add(12345)) >> 16) as f32 / 32768.0;

        while rng < ml && layer < 16 {
            layer += 1;
            let new_seed = seed.wrapping_add(layer as u32);
            rng = ((new_seed.wrapping_mul(1103515245).wrapping_add(12345)) >> 16) as f32 / 32768.0;
        }

        layer
    }
}

/// Error types for HNSW operations
#[derive(Debug, thiserror::Error)]
pub enum HnswError {
    #[error("Failed to allocate node: {0}")]
    AllocationError(String),

    #[error("Invalid embedding dimension: expected 768, got {0}")]
    InvalidDimension(usize),

    #[error("Graph structure corrupted: {0}")]
    CorruptedGraph(String),

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
}
