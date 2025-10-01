//! Lock-free HNSW graph structure using crossbeam data structures

use super::{CognitiveHnswParams, HnswEdge, HnswNode, SearchResult, SearchResults, SearchStats};
use crate::{Confidence, compute::VectorOps};
use crossbeam_epoch::{self as epoch, Guard};
use crossbeam_skiplist::SkipMap;
use dashmap::DashMap;
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

impl Default for HnswGraph {
    fn default() -> Self {
        Self::new()
    }
}

impl HnswGraph {
    /// Create a new empty graph
    #[must_use]
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
    ///
    /// # Errors
    ///
    /// Returns an error if required neighbor nodes cannot be loaded or if
    /// connection updates fail while building bidirectional edges.
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
            let candidates = if entry_point == u32::MAX {
                Vec::new()
            } else {
                self.search_layer(
                    node_arc.get_embedding(),
                    entry_point,
                    ef_construction,
                    layer as usize,
                    vector_ops,
                    &guard,
                )?
                .candidates
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
                Self::prune_connections(neighbor_node, layer as usize, m, vector_ops);
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
        let detailed = self.search_with_details(query, k, ef, threshold, vector_ops);
        detailed
            .hits
            .into_iter()
            .map(|result| (result.memory_id, result.confidence))
            .collect()
    }

    /// Search with detailed statistics and threshold filtering
    pub fn search_with_details(
        &self,
        query: &[f32; 768],
        k: usize,
        ef: usize,
        threshold: Confidence,
        vector_ops: &dyn VectorOps,
    ) -> SearchResults {
        let guard = epoch::pin();
        let mut stats = SearchStats::with_ef(ef);

        // Start from highest layer with an entry point
        let mut current_nearest = Vec::new();

        for layer in (0..16).rev() {
            let entry_point = self.get_entry_point(layer);
            if entry_point == u32::MAX {
                continue;
            }

            let layer_result = self
                .search_layer(
                    query,
                    entry_point,
                    if layer == 0 { ef } else { 1 },
                    layer,
                    vector_ops,
                    &guard,
                )
                .unwrap_or_default();

            stats.record_layer(layer_result.nodes_visited);

            if !layer_result.candidates.is_empty() {
                current_nearest = layer_result.candidates;
            }
        }

        let mut hits = Vec::new();
        let mut distances = Vec::new();

        for candidate in current_nearest.into_iter().take(k) {
            if candidate.confidence.raw() >= threshold.raw() {
                if let Ok(node) = self.get_node(candidate.node_id, &guard) {
                    distances.push(candidate.distance);
                    hits.push(SearchResult::new(
                        candidate.node_id,
                        candidate.distance,
                        candidate.confidence,
                        node.memory.id.clone(),
                    ));
                }
            }
        }

        stats.finalize(&distances);

        SearchResults { hits, stats }
    }

    /// Search within a single layer
    /// Search within a single layer
    ///
    /// # Errors
    ///
    /// Propagates errors when node lookups fail during the beam search.
    fn search_layer(
        &self,
        query: &[f32; 768],
        entry_point: u32,
        ef: usize,
        layer: usize,
        vector_ops: &dyn VectorOps,
        guard: &Guard,
    ) -> Result<LayerSearchResult, super::HnswError> {
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
            if current.distance > w.peek().map_or(f32::MAX, |r| r.0.distance) {
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

                        if distance < w.peek().map_or(f32::MAX, |r| r.0.distance) || w.len() < ef {
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

        let visited_count = visited.len();

        Ok(LayerSearchResult {
            candidates: results,
            nodes_visited: visited_count,
        })
    }

    /// Select diverse neighbors using heuristic
    fn select_neighbors_heuristic(
        &self,
        node: &Arc<HnswNode>,
        mut candidates: Vec<SearchCandidate>,
        m: usize,
        _layer: usize,
        vector_ops: &dyn VectorOps,
    ) -> Vec<u32> {
        if candidates.is_empty() {
            return Vec::new();
        }

        // Sort by confidence-weighted distance
        candidates.sort_by(|a, b| {
            let a_score = a.distance * (1.0 - a.confidence.raw());
            let b_score = b.distance * (1.0 - b.confidence.raw());
            a_score.total_cmp(&b_score)
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
        _node: Arc<HnswNode>,
        _layer: usize,
        _m: usize,
        _vector_ops: &dyn VectorOps,
    ) {
        // Placeholder for lock-free pruning implementation
    }

    /// Get a node by ID
    ///
    /// # Errors
    ///
    /// Returns `HnswError::MemoryNotFound` if the requested node is absent from
    /// every layer.
    fn get_node(&self, node_id: u32, _guard: &Guard) -> Result<Arc<HnswNode>, super::HnswError> {
        // Try each layer until we find the node
        for layer in &self.layers {
            if let Some(entry) = layer.get(&node_id) {
                return Ok(entry.value().clone());
            }
        }

        Err(super::HnswError::MemoryNotFound(format!(
            "Node {node_id} not found"
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
            for entry in &self.layers[layer_idx] {
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
            for entry in &self.layers[layer_idx] {
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
        for entry in &self.node_map {
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

#[derive(Clone, Debug, Default)]
struct LayerSearchResult {
    candidates: Vec<SearchCandidate>,
    nodes_visited: usize,
}

impl PartialEq for SearchCandidate {
    fn eq(&self, other: &Self) -> bool {
        self.distance.total_cmp(&other.distance) == CmpOrdering::Equal
    }
}

impl Eq for SearchCandidate {}

impl Ord for SearchCandidate {
    fn cmp(&self, other: &Self) -> CmpOrdering {
        // Reverse order for min-heap behavior
        other.distance.total_cmp(&self.distance)
    }
}

impl PartialOrd for SearchCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<CmpOrdering> {
        Some(self.cmp(other))
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
