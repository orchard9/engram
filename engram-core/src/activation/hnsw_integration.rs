//! Integration between HNSW index and activation spreading
//!
//! This module connects the HNSW graph structure with activation spreading,
//! allowing activation to flow through the hierarchical navigable small world
//! graph based on vector similarity.

use super::{storage_aware::StorageTier, ActivationRecord, NodeId};
use crate::index::{HnswGraph, HnswNode, SearchResult};
use crate::Confidence;
use dashmap::DashMap;
use std::sync::Arc;

/// HNSW-guided activation spreading engine
pub struct HnswActivationEngine {
    /// Reference to the HNSW index
    hnsw_graph: Arc<HnswGraph>,
    /// Activation records for nodes
    activations: Arc<DashMap<NodeId, ActivationRecord>>,
    /// Spreading parameters
    config: SpreadingConfig,
}

/// Configuration for HNSW-based activation spreading
#[derive(Debug, Clone)]
pub struct SpreadingConfig {
    /// Minimum similarity threshold for activation to spread
    pub similarity_threshold: f32,
    /// Decay factor based on graph distance
    pub distance_decay: f32,
    /// Maximum number of hops to spread activation
    pub max_hops: usize,
    /// Whether to use hierarchical layers for spreading
    pub use_hierarchical: bool,
    /// Confidence threshold for spreading
    pub confidence_threshold: Confidence,
}

impl Default for SpreadingConfig {
    fn default() -> Self {
        Self {
            similarity_threshold: 0.5,
            distance_decay: 0.8,
            max_hops: 3,
            use_hierarchical: true,
            confidence_threshold: Confidence::LOW,
        }
    }
}

impl HnswActivationEngine {
    /// Create a new HNSW activation engine
    pub fn new(hnsw_graph: Arc<HnswGraph>, config: SpreadingConfig) -> Self {
        Self {
            hnsw_graph,
            activations: Arc::new(DashMap::new()),
            config,
        }
    }

    /// Spread activation from a source node through HNSW neighbors
    pub fn spread_activation(&self, source_id: &NodeId, initial_activation: f32) -> Vec<(NodeId, f32)> {
        let mut results = Vec::new();
        let mut visited = dashmap::DashSet::new();
        
        // Initialize source activation
        let mut base_record = ActivationRecord::new(source_id.clone(), 0.1);
        base_record.set_storage_tier(StorageTier::Hot);
        self.activations.insert(source_id.clone(), base_record);
        
        if let Some(record) = self.activations.get(source_id) {
            record.accumulate_activation(initial_activation);
        }
        
        // Spread through HNSW neighbors
        self.spread_recursive(source_id, initial_activation, 0, &mut visited, &mut results);
        
        results
    }

    /// Recursive activation spreading through graph
    fn spread_recursive(
        &self,
        node_id: &NodeId,
        activation: f32,
        depth: usize,
        visited: &mut dashmap::DashSet<NodeId>,
        results: &mut Vec<(NodeId, f32)>,
    ) {
        // Check termination conditions
        if depth >= self.config.max_hops || activation < 0.01 {
            return;
        }
        
        // Mark as visited
        if !visited.insert(node_id.clone()) {
            return; // Already visited
        }
        
        // Get neighbors from HNSW graph
        let neighbors = self.get_hnsw_neighbors(node_id);
        
        for (neighbor_id, similarity) in neighbors {
            // Check similarity threshold
            if similarity < self.config.similarity_threshold {
                continue;
            }
            
            // Calculate propagated activation
            let decay = self.config.distance_decay.powi(depth as i32 + 1);
            let propagated = activation * similarity * decay;
            
            // Update neighbor activation
            let tier = StorageTier::from_depth(depth as u16 + 1);
            let record = self.activations.entry(neighbor_id.clone())
                .or_insert_with(|| {
                    let mut base = ActivationRecord::new(neighbor_id.clone(), 0.1);
                    base.set_storage_tier(tier);
                    base
                });
            
            if record.accumulate_activation(propagated) {
                results.push((neighbor_id.clone(), propagated));
                
                // Continue spreading
                self.spread_recursive(&neighbor_id, propagated, depth + 1, visited, results);
            }
        }
    }

    /// Get neighbors from HNSW index with similarity scores
    fn get_hnsw_neighbors(&self, node_id: &NodeId) -> Vec<(NodeId, f32)> {
        // Get neighbors from the HNSW graph
        self.hnsw_graph
            .get_neighbors(node_id, self.config.max_hops)
            .into_iter()
            .map(|(memory_id, distance, _confidence)| {
                // Convert distance to similarity (1.0 - distance)
                let similarity = 1.0 - distance;
                (memory_id, similarity)
            })
            .collect()
    }

    /// Query-based activation: activate nodes similar to query
    pub fn activate_by_query(&self, query_embedding: &[f32; 768], k: usize) -> Vec<(NodeId, f32)> {
        // Search HNSW for similar nodes
        let search_results = self.search_similar(query_embedding, k);
        
        let mut activated = Vec::new();
        
        for result in search_results {
            let activation = result.confidence.raw();
            
            // Create or update activation record
            let record = self.activations.entry(result.memory_id.clone())
                .or_insert_with(|| ActivationRecord::new(result.memory_id.clone(), 0.1));
            
            record.accumulate_activation(activation);
            activated.push((result.memory_id, activation));
        }
        
        activated
    }

    /// Search for similar nodes using HNSW
    fn search_similar(&self, query: &[f32; 768], k: usize) -> Vec<SearchResult> {
        // Use SIMD vector operations for search
        use crate::compute::get_vector_ops;
        let vector_ops = get_vector_ops();

        // Search with ef=k*2 for better quality, applying confidence threshold
        let results = self.hnsw_graph.search_with_details(
            query,
            k,
            k * 2,
            self.config.confidence_threshold,
            vector_ops.as_ref(),
        );

        results.hits
    }

    /// Get current activation levels for all nodes
    pub fn get_activations(&self) -> Vec<(NodeId, f32)> {
        self.activations
            .iter()
            .map(|entry| (entry.key().clone(), entry.value().get_activation()))
            .filter(|(_, activation)| *activation > 0.01)
            .collect()
    }

    /// Reset all activations
    pub fn reset(&self) {
        for entry in self.activations.iter() {
            entry.value().reset();
        }
    }

    /// Apply time-based decay to all activations
    pub fn apply_decay(&self, decay_factor: f32) {
        for entry in self.activations.iter() {
            let current = entry.value().get_activation();
            entry.value().set_activation(current * decay_factor);
        }
    }
}

/// Bidirectional activation flow between HNSW layers
pub struct HierarchicalActivation {
    /// Activation engines for each HNSW layer
    layer_engines: Vec<HnswActivationEngine>,
    /// Inter-layer connection strength
    layer_coupling: f32,
}

impl HierarchicalActivation {
    /// Create hierarchical activation across HNSW layers
    pub fn new(hnsw_graph: Arc<HnswGraph>, layer_coupling: f32) -> Self {
        // Create an engine for each layer
        let layer_engines = vec![
            HnswActivationEngine::new(hnsw_graph.clone(), SpreadingConfig::default()),
        ];
        
        Self {
            layer_engines,
            layer_coupling,
        }
    }

    /// Spread activation across all layers
    pub fn spread_hierarchical(&self, source_id: &NodeId, initial_activation: f32) -> Vec<(NodeId, f32)> {
        let mut all_results = Vec::new();
        
        // Spread through each layer
        for (layer_idx, engine) in self.layer_engines.iter().enumerate() {
            let layer_activation = initial_activation * (1.0 - layer_idx as f32 * 0.2);
            let layer_results = engine.spread_activation(source_id, layer_activation);
            
            // Collect results
            for (node_id, activation) in layer_results {
                all_results.push((node_id, activation * self.layer_coupling.powi(layer_idx as i32)));
            }
        }
        
        all_results
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spreading_config() {
        let config = SpreadingConfig::default();
        assert_eq!(config.similarity_threshold, 0.5);
        assert_eq!(config.max_hops, 3);
        assert!(config.use_hierarchical);
    }

    #[test]
    fn test_activation_engine_creation() {
        // This would need a mock HNSW graph
        // For now, just test that the types compile correctly
        let config = SpreadingConfig {
            similarity_threshold: 0.6,
            distance_decay: 0.9,
            max_hops: 5,
            use_hierarchical: false,
            confidence_threshold: Confidence::MEDIUM,
        };
        
        // Verify config is set correctly
        assert_eq!(config.similarity_threshold, 0.6);
        assert_eq!(config.distance_decay, 0.9);
    }
}
