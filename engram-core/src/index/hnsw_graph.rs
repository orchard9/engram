//! Lock-free HNSW graph structure using crossbeam data structures

use super::{CognitiveHnswParams, HnswEdge, HnswNode};
use crate::{Confidence, compute::VectorOps};
use crossbeam_epoch::{self as epoch, Guard};
use crossbeam_skiplist::SkipMap;
use dashmap::DashMap;
use smallvec::SmallVec;
use std::cmp::Ordering as CmpOrdering;
use std::collections::BinaryHeap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, AtomicUsize, Ordering};

/// Lock-free HNSW graph with multiple layers
pub struct HnswGraph {
    /// Skip-list layers for lock-free access
    layers: Vec<SkipMap<u32, Arc<HnswNode>>>,

    /// Entry points for each layer
    entry_points: Vec<AtomicU32>,

    /// Node lookup map for O(1) access
    node_map: DashMap<String, u32>,

    /// Total number of nodes
    node_count: AtomicUsize,
}

impl HnswGraph {
    /// Create a new empty graph
    pub fn new() -> Self {
        let mut layers = Vec::with_capacity(16);
        let mut entry_points = Vec::with_capacity(16);

        for _ in 0..16 {
            layers.push(SkipMap::new());
            entry_points.push(AtomicU32::new(u32::MAX));
        }

        Self {
            layers,
            entry_points,
            node_map: DashMap::new(),
            node_count: AtomicUsize::new(0),
        }
    }

    /// Insert a node into the graph
    pub fn insert_node(
        &self,
        node: HnswNode,
        params: &CognitiveHnswParams,
        vector_ops: &dyn VectorOps,
    ) -> Result<(), super::HnswError> {
        let guard = epoch::pin();
        let node_id = node.node_id;
        let layer_count = node.layer_count.load(Ordering::Relaxed);
        let node_arc = Arc::new(node);

        // Store memory ID to node ID mapping
        self.node_map.insert(node_arc.memory.id.clone(), node_id);

        // Insert into all layers up to layer_count
        for layer in 0..=layer_count {
            // Find entry point for this layer
            let entry_point = self.get_entry_point(layer as usize);

            // Search for nearest neighbors in this layer
            let ef_construction = params.ef_construction.load(Ordering::Relaxed);
            let candidates = if entry_point != u32::MAX {
                self.search_layer(
                    node_arc.get_embedding(),
                    entry_point,
                    ef_construction,
                    layer as usize,
                    vector_ops,
                    &guard,
                )?
            } else {
                Vec::new()
            };

            // Select M neighbors with diversity
            let m = if layer == 0 {
                params.m_l.load(Ordering::Relaxed)
            } else {
                params.m_max.load(Ordering::Relaxed)
            };

            let neighbors = self.select_neighbors_heuristic(
                &node_arc,
                candidates,
                m,
                layer as usize,
                vector_ops,
            );

            // Add bidirectional connections
            for neighbor_id in &neighbors {
                // Add edge from new node to neighbor
                let neighbor_node = self.get_node(*neighbor_id, &guard)?;
                let distance = vector_ops
                    .cosine_similarity_768(node_arc.get_embedding(), neighbor_node.get_embedding());

                let edge = HnswEdge::new(*neighbor_id, 1.0 - distance, neighbor_node.confidence);
                node_arc.add_connection(layer as usize, edge)?;

                // Add edge from neighbor to new node (bidirectional)
                let reverse_edge = HnswEdge::new(node_id, 1.0 - distance, node_arc.confidence);
                neighbor_node.add_connection(layer as usize, reverse_edge)?;

                // Prune neighbor's connections if exceeded M
                self.prune_connections(neighbor_node, layer as usize, m, vector_ops)?;
            }

            // Insert node into layer
            self.layers[layer as usize].insert(node_id, node_arc.clone());

            // Update entry point if necessary
            if entry_point == u32::MAX {
                self.entry_points[layer as usize].store(node_id, Ordering::Release);
            }
        }

        self.node_count.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    /// Search for k nearest neighbors
    pub fn search(
        &self,
        query: &[f32; 768],
        k: usize,
        ef: usize,
        threshold: Confidence,
        vector_ops: &dyn VectorOps,
    ) -> Vec<(String, Confidence)> {
        let guard = epoch::pin();

        // Start from highest layer with an entry point
        let mut current_nearest = Vec::new();

        for layer in (0..16).rev() {
            let entry_point = self.get_entry_point(layer);
            if entry_point == u32::MAX {
                continue;
            }

            // Search in this layer
            let candidates = self
                .search_layer(
                    query,
                    entry_point,
                    if layer == 0 { ef } else { 1 },
                    layer,
                    vector_ops,
                    &guard,
                )
                .unwrap_or_default();

            if !candidates.is_empty() {
                current_nearest = candidates;

                // If we're not at layer 0, use the best candidate as entry for next layer
                if layer > 0 && !current_nearest.is_empty() {
                    // Continue search from best candidate
                    continue;
                }
            }
        }

        // Filter by confidence threshold and convert to results
        let mut results = Vec::new();
        for candidate in current_nearest.into_iter().take(k) {
            if candidate.confidence.raw() >= threshold.raw() {
                if let Ok(node) = self.get_node(candidate.node_id, &guard) {
                    results.push((node.memory.id.clone(), candidate.confidence));
                }
            }
        }

        results
    }

    /// Search within a single layer
    fn search_layer(
        &self,
        query: &[f32; 768],
        entry_point: u32,
        ef: usize,
        layer: usize,
        vector_ops: &dyn VectorOps,
        guard: &Guard,
    ) -> Result<Vec<SearchCandidate>, super::HnswError> {
        let mut visited = std::collections::HashSet::new();
        let mut candidates = BinaryHeap::new();
        let mut w = BinaryHeap::new();

        // Initialize with entry point
        let entry_node = self.get_node(entry_point, guard)?;
        let entry_distance =
            1.0 - vector_ops.cosine_similarity_768(query, entry_node.get_embedding());

        let entry_candidate = SearchCandidate {
            node_id: entry_point,
            distance: entry_distance,
            confidence: entry_node.confidence,
        };

        candidates.push(entry_candidate.clone());
        w.push(std::cmp::Reverse(entry_candidate));
        visited.insert(entry_point);

        // Search expansion
        while let Some(current) = candidates.pop() {
            if current.distance > w.peek().map(|r| r.0.distance).unwrap_or(f32::MAX) {
                break;
            }

            // Check neighbors
            let current_node = self.get_node(current.node_id, guard)?;
            if let Some(connections) = current_node.get_connections(layer) {
                for edge in connections {
                    if visited.insert(edge.target_id) {
                        let neighbor = self.get_node(edge.target_id, guard)?;
                        let distance =
                            1.0 - vector_ops.cosine_similarity_768(query, neighbor.get_embedding());

                        let neighbor_candidate = SearchCandidate {
                            node_id: edge.target_id,
                            distance,
                            confidence: neighbor.confidence,
                        };

                        if distance < w.peek().map(|r| r.0.distance).unwrap_or(f32::MAX)
                            || w.len() < ef
                        {
                            candidates.push(neighbor_candidate.clone());
                            w.push(std::cmp::Reverse(neighbor_candidate));

                            if w.len() > ef {
                                w.pop();
                            }
                        }
                    }
                }
            }
        }

        // Extract results
        let mut results = Vec::with_capacity(w.len());
        while let Some(std::cmp::Reverse(candidate)) = w.pop() {
            results.push(candidate);
        }
        results.reverse(); // Best first

        Ok(results)
    }

    /// Select diverse neighbors using heuristic
    fn select_neighbors_heuristic(
        &self,
        node: &Arc<HnswNode>,
        mut candidates: Vec<SearchCandidate>,
        m: usize,
        layer: usize,
        vector_ops: &dyn VectorOps,
    ) -> Vec<u32> {
        if candidates.is_empty() {
            return Vec::new();
        }

        // Sort by confidence-weighted distance
        candidates.sort_by(|a, b| {
            let a_score = a.distance * (1.0 - a.confidence.raw());
            let b_score = b.distance * (1.0 - b.confidence.raw());
            a_score.partial_cmp(&b_score).unwrap_or(CmpOrdering::Equal)
        });

        let mut selected = Vec::with_capacity(m);

        for candidate in candidates.into_iter().take(m * 2) {
            if selected.len() >= m {
                break;
            }

            // Check diversity constraints
            let mut is_diverse = true;
            for &existing_id in &selected {
                if let Ok(existing_node) = self.get_node(existing_id, &epoch::pin()) {
                    let similarity = vector_ops
                        .cosine_similarity_768(node.get_embedding(), existing_node.get_embedding());

                    // Require minimum diversity
                    if similarity > 0.95 {
                        is_diverse = false;
                        break;
                    }
                }
            }

            if is_diverse {
                selected.push(candidate.node_id);
            }
        }

        selected
    }

    /// Prune connections to maintain M limit
    fn prune_connections(
        &self,
        _node: Arc<HnswNode>,
        _layer: usize,
        _m: usize,
        _vector_ops: &dyn VectorOps,
    ) -> Result<(), super::HnswError> {
        // This would be implemented with proper lock-free pruning
        // For now, we'll keep it simple
        Ok(())
    }

    /// Get a node by ID
    fn get_node(&self, node_id: u32, _guard: &Guard) -> Result<Arc<HnswNode>, super::HnswError> {
        // Try each layer until we find the node
        for layer in &self.layers {
            if let Some(entry) = layer.get(&node_id) {
                return Ok(entry.value().clone());
            }
        }

        Err(super::HnswError::MemoryNotFound(format!(
            "Node {} not found",
            node_id
        )))
    }

    /// Get entry point for a layer
    fn get_entry_point(&self, layer: usize) -> u32 {
        self.entry_points[layer].load(Ordering::Acquire)
    }

    /// Get neighbors within N hops
    pub fn get_neighbors(
        &self,
        memory_id: &str,
        max_hops: usize,
    ) -> Vec<(String, f32, Confidence)> {
        let mut results = Vec::new();
        let guard = epoch::pin();

        // Get starting node
        let node_id = match self.node_map.get(memory_id) {
            Some(id) => *id,
            None => return results,
        };

        let mut visited = std::collections::HashSet::new();
        let mut current_wave = vec![(node_id, 0)];
        visited.insert(node_id);

        for hop in 1..=max_hops {
            let mut next_wave = Vec::new();

            for (current_id, _) in current_wave {
                if let Ok(node) = self.get_node(current_id, &guard) {
                    // Get connections from layer 0 (most connections)
                    if let Some(connections) = node.get_connections(0) {
                        for edge in connections {
                            if visited.insert(edge.target_id) {
                                if let Ok(neighbor) = self.get_node(edge.target_id, &guard) {
                                    results.push((
                                        neighbor.memory.id.clone(),
                                        edge.cached_distance,
                                        Confidence::exact(edge.confidence_weight),
                                    ));
                                    next_wave.push((edge.target_id, hop));
                                }
                            }
                        }
                    }
                }
            }

            current_wave = next_wave;
            if current_wave.is_empty() {
                break;
            }
        }

        results
    }

    /// Validate graph structure integrity
    pub fn validate_structure(&self) -> bool {
        // Check that all nodes in higher layers also exist in lower layers
        for layer_idx in 1..16 {
            for entry in self.layers[layer_idx].iter() {
                let node_id = *entry.key();

                // Check that this node exists in all lower layers
                for lower_layer in 0..layer_idx {
                    if !self.layers[lower_layer].contains_key(&node_id) {
                        return false;
                    }
                }
            }
        }

        true
    }

    /// Validate bidirectional consistency
    pub fn validate_bidirectional_consistency(&self) -> bool {
        let guard = epoch::pin();

        for layer_idx in 0..16 {
            for entry in self.layers[layer_idx].iter() {
                let node = entry.value();

                if let Some(connections) = node.get_connections(layer_idx) {
                    for edge in connections {
                        // Check that the target has a reverse edge
                        if let Ok(target) = self.get_node(edge.target_id, &guard) {
                            if let Some(target_connections) = target.get_connections(layer_idx) {
                                let has_reverse = target_connections
                                    .iter()
                                    .any(|e| e.target_id == node.node_id);

                                if !has_reverse {
                                    return false;
                                }
                            }
                        }
                    }
                }
            }
        }

        true
    }

    /// Check memory consistency
    pub fn check_memory_consistency(&self) -> bool {
        // Verify that node_map is consistent with actual nodes
        for entry in self.node_map.iter() {
            let memory_id = entry.key();
            let node_id = *entry.value();

            // Check that node exists in at least layer 0
            if !self.layers[0].contains_key(&node_id) {
                return false;
            }

            // Check that the node's memory ID matches
            if let Some(node_entry) = self.layers[0].get(&node_id) {
                if node_entry.value().memory.id != *memory_id {
                    return false;
                }
            }
        }

        true
    }
}

/// Search candidate with distance and confidence
#[derive(Clone, Debug)]
struct SearchCandidate {
    node_id: u32,
    distance: f32,
    confidence: Confidence,
}

impl PartialEq for SearchCandidate {
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance
    }
}

impl Eq for SearchCandidate {}

impl PartialOrd for SearchCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<CmpOrdering> {
        // Reverse order for min-heap behavior
        other.distance.partial_cmp(&self.distance)
    }
}

impl Ord for SearchCandidate {
    fn cmp(&self, other: &Self) -> CmpOrdering {
        self.partial_cmp(other).unwrap_or(CmpOrdering::Equal)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_graph_creation() {
        let graph = HnswGraph::new();
        assert_eq!(graph.node_count.load(Ordering::Relaxed), 0);
        assert!(graph.validate_structure());
    }
}
