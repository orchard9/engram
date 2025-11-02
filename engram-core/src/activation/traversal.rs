//! Graph traversal algorithms for activation spreading
//!
//! Implements cache-optimized breadth-first and depth-first traversal
//! with biological constraints and NUMA awareness.

use crate::activation::{
    ActivationError, ActivationResult, DecayFunction, EdgeType, MemoryGraph, NodeId,
    storage_aware::StorageTier,
};
use dashmap::DashMap;
use std::collections::{HashSet, VecDeque};
use std::convert::TryFrom;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use uuid::Uuid;

/// Cache-optimized breadth-first traversal for activation spreading
pub struct BreadthFirstTraversal {
    visited: DashMap<NodeId, AtomicUsize>, // Visit count per node
    tier_budgets: std::collections::HashMap<StorageTier, usize>,
    default_budget: usize,
    prefetch_distance: usize, // Cache prefetch lookahead
}

impl BreadthFirstTraversal {
    /// Create new breadth-first traversal
    #[must_use]
    pub fn new(
        tier_budgets: std::collections::HashMap<StorageTier, usize>,
        prefetch_distance: usize,
    ) -> Self {
        let default_budget = tier_budgets
            .get(&StorageTier::Hot)
            .copied()
            .or_else(|| tier_budgets.values().copied().min())
            .unwrap_or(3);
        Self {
            visited: DashMap::new(),
            tier_budgets,
            default_budget,
            prefetch_distance,
        }
    }

    /// Helper to create traversal with the same budget for every tier.
    #[must_use]
    pub fn with_uniform_budget(max_visits: usize, prefetch_distance: usize) -> Self {
        Self::new(
            std::collections::HashMap::from([
                (StorageTier::Hot, max_visits),
                (StorageTier::Warm, max_visits.saturating_add(1)),
                (StorageTier::Cold, max_visits.saturating_add(2)),
            ]),
            prefetch_distance,
        )
    }

    /// Execute breadth-first traversal from seed nodes
    ///
    /// # Errors
    ///
    /// Currently never returns an error but maintains Result for future extensibility
    pub fn traverse<F>(
        &self,
        graph: &MemoryGraph,
        seed_nodes: &[(NodeId, f32)],
        max_depth: u16,
        decay_function: &DecayFunction,
        mut process_node: F,
    ) -> ActivationResult<()>
    where
        F: FnMut(&NodeId, f32, u16) -> bool, // Returns true to continue spreading
    {
        let mut current_level = VecDeque::new();
        let mut next_level = VecDeque::new();

        // Initialize with seed nodes
        for (node_id, activation) in seed_nodes {
            current_level.push_back((node_id.clone(), *activation, 0));
        }

        let mut current_depth = 0;

        while !current_level.is_empty() && current_depth < max_depth {
            // Process current level
            while let Some((node_id, activation, depth)) = current_level.pop_front() {
                // Check visit count for cycle detection
                let tier = StorageTier::from_depth(depth);
                let budget = self
                    .tier_budgets
                    .get(&tier)
                    .copied()
                    .unwrap_or(self.default_budget);

                let (_current_visits, should_continue) = {
                    let visits_entry = self
                        .visited
                        .entry(node_id.clone())
                        .or_insert_with(|| AtomicUsize::new(0));

                    let current_visits = visits_entry.load(Ordering::Relaxed);
                    let should_skip = if current_visits >= budget {
                        true
                    } else {
                        // Increment visit count since we're processing this node
                        visits_entry.fetch_add(1, Ordering::Relaxed);
                        false
                    };
                    drop(visits_entry);
                    (current_visits, should_skip)
                };

                if should_continue {
                    continue;
                }

                // Process the node
                if !process_node(&node_id, activation, depth) {
                    continue; // Skip spreading from this node
                }

                // Get neighbors and add to next level
                let node_uuid = Uuid::new_v5(&Uuid::NAMESPACE_OID, node_id.as_bytes());
                let neighbors = graph
                    .get_neighbors(&node_uuid)
                    .map_err(|err| ActivationError::ThreadingError(err.to_string()))?;
                let decay_factor = decay_function.apply(depth + 1);

                for (neighbor_id, weight) in neighbors {
                    let neighbor_node_id = neighbor_id.to_string();
                    let new_activation = Self::calculate_activation(
                        activation,
                        weight,
                        decay_factor,
                        EdgeType::Excitatory, // Default edge type
                    );

                    if new_activation > 0.01 {
                        // Threshold check
                        next_level.push_back((neighbor_node_id, new_activation, depth + 1));
                    }
                }

                // Prefetch cache for upcoming nodes
                self.prefetch_neighbors(graph, &current_level);
            }

            // Move to next level
            std::mem::swap(&mut current_level, &mut next_level);
            current_depth += 1;
        }

        Ok(())
    }

    /// Calculate activation considering edge type (Dale's law)
    fn calculate_activation(
        source_activation: f32,
        edge_weight: f32,
        decay_factor: f32,
        edge_type: EdgeType,
    ) -> f32 {
        let base_activation = source_activation * edge_weight * decay_factor;

        match edge_type {
            EdgeType::Excitatory => base_activation.max(0.0),
            EdgeType::Inhibitory => -base_activation.abs(),
            EdgeType::Modulatory => base_activation * 0.5, // Reduced influence
        }
    }

    /// Prefetch memory for cache optimization
    fn prefetch_neighbors(&self, graph: &MemoryGraph, queue: &VecDeque<(NodeId, f32, u16)>) {
        // Prefetch neighbors for upcoming nodes to improve cache locality
        for (i, (node_id, _, _)) in queue.iter().enumerate() {
            if i >= self.prefetch_distance {
                break;
            }

            // This would use CPU prefetch instructions in optimized version
            let node_uuid = Uuid::new_v5(&Uuid::NAMESPACE_OID, node_id.as_bytes());
            let _ = graph.get_neighbors(&node_uuid);
        }
    }

    /// Reset traversal state
    pub fn reset(&self) {
        self.visited.clear();
    }

    /// Get visit statistics
    #[must_use]
    pub fn get_visit_stats(&self) -> Vec<(NodeId, usize)> {
        self.visited
            .iter()
            .map(|entry| (entry.key().clone(), entry.value().load(Ordering::Relaxed)))
            .collect()
    }
}

/// Depth-first traversal with stack-based implementation
pub struct DepthFirstTraversal {
    visited: DashMap<NodeId, bool>,
    recursion_limit: usize,
}

impl DepthFirstTraversal {
    /// Create new depth-first traversal
    #[must_use]
    pub fn new(recursion_limit: usize) -> Self {
        Self {
            visited: DashMap::new(),
            recursion_limit,
        }
    }

    /// Execute depth-first traversal
    ///
    /// # Errors
    ///
    /// Currently never returns an error but maintains Result for future extensibility
    pub fn traverse<F>(
        &self,
        graph: &MemoryGraph,
        seed_nodes: &[(NodeId, f32)],
        max_depth: u16,
        decay_function: &DecayFunction,
        mut process_node: F,
    ) -> ActivationResult<()>
    where
        F: FnMut(&NodeId, f32, u16) -> bool,
    {
        // Use explicit stack to avoid stack overflow
        let mut stack = Vec::new();

        // Initialize with seed nodes
        for (node_id, activation) in seed_nodes {
            stack.push((node_id.clone(), *activation, 0));
        }

        while let Some((node_id, activation, depth)) = stack.pop() {
            if depth >= max_depth {
                continue;
            }

            // Check if already visited
            if self.visited.contains_key(&node_id) {
                continue;
            }

            self.visited.insert(node_id.clone(), true);

            // Process the node
            if !process_node(&node_id, activation, depth) {
                continue;
            }

            // Add neighbors to stack (in reverse order for proper DFS ordering)
            let node_uuid = Uuid::new_v5(&Uuid::NAMESPACE_OID, node_id.as_bytes());
            let neighbors = graph
                .get_neighbors(&node_uuid)
                .map_err(|err| ActivationError::ThreadingError(err.to_string()))?;
            let decay_factor = decay_function.apply(depth + 1);

            for (neighbor_id, weight) in neighbors.iter().rev() {
                let neighbor_node_id = neighbor_id.to_string();
                let new_activation = BreadthFirstTraversal::calculate_activation(
                    activation,
                    *weight,
                    decay_factor,
                    EdgeType::Excitatory, // Default edge type
                );

                if new_activation > 0.01 {
                    stack.push((neighbor_node_id, new_activation, depth + 1));
                }
            }

            // Check stack size to prevent excessive memory usage
            if stack.len() > self.recursion_limit {
                return Err(ActivationError::InvalidConfig(
                    "Recursion limit exceeded in depth-first traversal".to_string(),
                ));
            }
        }

        Ok(())
    }

    /// Reset traversal state
    pub fn reset(&self) {
        self.visited.clear();
    }

    /// Get visited nodes
    #[must_use]
    pub fn get_visited(&self) -> HashSet<NodeId> {
        self.visited
            .iter()
            .map(|entry| entry.key().clone())
            .collect()
    }
}

/// Adaptive traversal that switches between BFS and DFS based on graph structure
pub struct AdaptiveTraversal {
    bfs: BreadthFirstTraversal,
    dfs: DepthFirstTraversal,
    branching_threshold: f32, // Switch to DFS when branching factor is low
}

impl AdaptiveTraversal {
    /// Create new adaptive traversal
    #[must_use]
    pub fn new(
        max_visits: usize,
        prefetch_distance: usize,
        recursion_limit: usize,
        branching_threshold: f32,
    ) -> Self {
        Self {
            bfs: BreadthFirstTraversal::with_uniform_budget(max_visits, prefetch_distance),
            dfs: DepthFirstTraversal::new(recursion_limit),
            branching_threshold,
        }
    }

    /// Execute adaptive traversal
    ///
    /// # Errors
    ///
    /// Currently never returns an error but maintains Result for future extensibility
    pub fn traverse<F>(
        &self,
        graph: &MemoryGraph,
        seed_nodes: &[(NodeId, f32)],
        max_depth: u16,
        decay_function: &DecayFunction,
        process_node: F,
    ) -> ActivationResult<()>
    where
        F: FnMut(&NodeId, f32, u16) -> bool,
    {
        // Analyze graph structure to choose traversal method
        let avg_branching_factor = Self::estimate_branching_factor(graph, seed_nodes);

        if avg_branching_factor > self.branching_threshold {
            // High branching factor - use BFS for better cache locality
            self.bfs
                .traverse(graph, seed_nodes, max_depth, decay_function, process_node)
        } else {
            // Low branching factor - use DFS for better memory usage
            self.dfs
                .traverse(graph, seed_nodes, max_depth, decay_function, process_node)
        }
    }

    /// Estimate average branching factor
    fn estimate_branching_factor(graph: &MemoryGraph, seed_nodes: &[(NodeId, f32)]) -> f32 {
        let sample_size = std::cmp::min(seed_nodes.len(), 10);
        let mut total_neighbors = 0;
        let mut sampled_nodes = 0;

        for (node_id, _) in seed_nodes.iter().take(sample_size) {
            let node_uuid = Uuid::new_v5(&Uuid::NAMESPACE_OID, node_id.as_bytes());
            if let Ok(neighbors) = graph.get_neighbors(&node_uuid) {
                total_neighbors += neighbors.len();
                sampled_nodes += 1;
            }
        }

        if sampled_nodes > 0 {
            let total_neighbors = u64::try_from(total_neighbors).unwrap_or(u64::MAX);
            let sampled_nodes = u64::try_from(sampled_nodes).unwrap_or(1);
            let average = u64_to_f64(total_neighbors) / u64_to_f64(sampled_nodes);
            f64_to_f32(average)
        } else {
            0.0
        }
    }

    /// Reset both traversal methods
    pub fn reset(&self) {
        self.bfs.reset();
        self.dfs.reset();
    }
}

const fn u64_to_f64(value: u64) -> f64 {
    #[allow(clippy::cast_precision_loss)]
    {
        value as f64
    }
}

const fn f64_to_f32(value: f64) -> f32 {
    #[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
    {
        value as f32
    }
}

/// Parallel breadth-first traversal with work distribution
pub struct ParallelBreadthFirstTraversal {
    _num_workers: usize,
    chunk_size: usize,
}

impl ParallelBreadthFirstTraversal {
    /// Create new parallel BFS traversal
    #[must_use]
    pub const fn new(num_workers: usize, chunk_size: usize) -> Self {
        Self {
            _num_workers: num_workers,
            chunk_size,
        }
    }

    /// Execute parallel traversal using work distribution with rayon
    ///
    /// This implementation uses rayon's parallel iterator support to distribute
    /// traversal work across threads while maintaining proper lifetime management.
    ///
    /// # Performance
    ///
    /// - Uses work-stealing thread pool for dynamic load balancing
    /// - Chunk size controls granularity of parallel work distribution
    /// - Best performance on graphs with high branching factors
    ///
    /// # Arguments
    ///
    /// * `graph` - Thread-safe graph reference (Arc allows sharing across workers)
    /// * `seed_nodes` - Starting nodes for traversal
    /// * `max_depth` - Maximum traversal depth
    /// * `decay_function` - Activation decay function applied at each level
    ///
    /// # Returns
    ///
    /// Vector of visited nodes with their activations and depths
    ///
    /// # Panics
    ///
    /// Panics if internal mutex becomes poisoned (rare, indicates thread panic during traversal)
    #[must_use]
    #[allow(clippy::expect_used)] // expect is appropriate for mutex poisoning detection
    pub fn traverse_parallel(
        &self,
        graph: &Arc<MemoryGraph>,
        seed_nodes: &[(NodeId, f32)],
        max_depth: u16,
        decay_function: &DecayFunction,
    ) -> Vec<(NodeId, f32, u16)> {
        use rayon::prelude::*;
        use std::sync::Mutex;

        let results = Arc::new(Mutex::new(Vec::new()));
        let mut current_level: Vec<_> = seed_nodes
            .iter()
            .map(|(id, act)| (id.clone(), *act, 0))
            .collect();

        for _depth in 0..max_depth {
            if current_level.is_empty() {
                break;
            }

            // Process current level in parallel chunks
            let next_level = Arc::new(Mutex::new(Vec::new()));

            current_level
                .par_chunks(self.chunk_size.max(1))
                .for_each(|chunk| {
                    let mut local_results = Vec::with_capacity(chunk.len());
                    let mut local_next = Vec::new();

                    for (node_id, activation, depth) in chunk {
                        local_results.push((node_id.clone(), *activation, *depth));

                        let node_uuid = Uuid::new_v5(&Uuid::NAMESPACE_OID, node_id.as_bytes());
                        if let Ok(neighbors) = graph.get_neighbors(&node_uuid) {
                            let decay_factor = decay_function.apply(*depth + 1);

                            for (neighbor_id, weight) in neighbors {
                                let neighbor_node_id = neighbor_id.to_string();
                                let new_activation = BreadthFirstTraversal::calculate_activation(
                                    *activation,
                                    weight,
                                    decay_factor,
                                    EdgeType::Excitatory,
                                );

                                if new_activation > 0.01 {
                                    local_next.push((neighbor_node_id, new_activation, *depth + 1));
                                }
                            }
                        }
                    }

                    // Batch append to shared collections
                    if !local_results.is_empty()
                        && let Ok(mut results_guard) = results.lock()
                    {
                        results_guard.extend(local_results);
                    }

                    if !local_next.is_empty()
                        && let Ok(mut next_guard) = next_level.lock()
                    {
                        next_guard.extend(local_next);
                    }
                });

            // Extract next level for next iteration
            current_level = Arc::try_unwrap(next_level).map_or_else(
                |arc| {
                    arc.lock()
                        .expect("Parallel traversal: failed to lock next_level")
                        .clone()
                },
                |mutex| {
                    mutex
                        .into_inner()
                        .expect("Parallel traversal: mutex poisoned for next_level")
                },
            );
        }

        // Extract final results
        Arc::try_unwrap(results).map_or_else(
            |arc| {
                arc.lock()
                    .expect("Parallel traversal: failed to lock results")
                    .clone()
            },
            |mutex| {
                mutex
                    .into_inner()
                    .expect("Parallel traversal: mutex poisoned for results")
            },
        )
    }

    /// Execute traversal using single-threaded implementation
    ///
    /// Simpler implementation without parallel overhead, suitable for
    /// small graphs or when deterministic ordering is required.
    ///
    /// # Returns
    ///
    /// Vector of visited nodes with their activations and depths
    #[must_use]
    pub fn traverse_simple(
        graph: &Arc<MemoryGraph>,
        seed_nodes: &[(NodeId, f32)],
        max_depth: u16,
        decay_function: &DecayFunction,
    ) -> Vec<(NodeId, f32, u16)> {
        let mut results = Vec::new();
        let mut current_level: VecDeque<_> = seed_nodes
            .iter()
            .map(|(id, act)| (id.clone(), *act, 0))
            .collect();

        for _depth in 0..max_depth {
            if current_level.is_empty() {
                break;
            }

            let mut next_level = VecDeque::new();

            while let Some((node_id, activation, depth)) = current_level.pop_front() {
                results.push((node_id.clone(), activation, depth));

                let node_uuid = Uuid::new_v5(&Uuid::NAMESPACE_OID, node_id.as_bytes());
                if let Ok(neighbors) = graph.get_neighbors(&node_uuid) {
                    let decay_factor = decay_function.apply(depth + 1);

                    for (neighbor_id, weight) in neighbors {
                        let neighbor_node_id = neighbor_id.to_string();
                        let new_activation = BreadthFirstTraversal::calculate_activation(
                            activation,
                            weight,
                            decay_factor,
                            EdgeType::Excitatory,
                        );

                        if new_activation > 0.01 {
                            next_level.push_back((neighbor_node_id, new_activation, depth + 1));
                        }
                    }
                }
            }

            current_level = next_level;
        }

        results
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use super::*;
    use crate::activation::{ActivationGraphExt, EdgeType};

    fn create_test_graph() -> MemoryGraph {
        let graph = crate::activation::create_activation_graph();

        // Create test graph: A -> B -> C, A -> D, B -> D
        ActivationGraphExt::add_edge(
            &graph,
            "A".to_string(),
            "B".to_string(),
            0.8,
            EdgeType::Excitatory,
        );
        ActivationGraphExt::add_edge(
            &graph,
            "B".to_string(),
            "C".to_string(),
            0.6,
            EdgeType::Excitatory,
        );
        ActivationGraphExt::add_edge(
            &graph,
            "A".to_string(),
            "D".to_string(),
            0.4,
            EdgeType::Excitatory,
        );
        ActivationGraphExt::add_edge(
            &graph,
            "B".to_string(),
            "D".to_string(),
            0.5,
            EdgeType::Excitatory,
        );
        ActivationGraphExt::add_edge(
            &graph,
            "D".to_string(),
            "E".to_string(),
            0.3,
            EdgeType::Inhibitory,
        );

        graph
    }

    #[test]
    fn test_breadth_first_traversal() {
        let graph = create_test_graph();
        let bfs = BreadthFirstTraversal::with_uniform_budget(3, 2);
        let decay_fn = DecayFunction::Exponential { rate: 0.5 };

        let mut visited_nodes = Vec::new();
        let seed_nodes = vec![("A".to_string(), 1.0)];

        bfs.traverse(
            &graph,
            &seed_nodes,
            3,
            &decay_fn,
            |node_id, activation, depth| {
                visited_nodes.push((node_id.clone(), activation, depth));
                true
            },
        )
        .unwrap();

        assert!(!visited_nodes.is_empty());

        // Should visit A at depth 0
        assert!(
            visited_nodes
                .iter()
                .any(|(id, _, depth)| id == "A" && *depth == 0)
        );

        bfs.reset();
        assert!(bfs.get_visit_stats().is_empty());
    }

    #[test]
    fn test_depth_first_traversal() {
        let graph = create_test_graph();
        let dfs = DepthFirstTraversal::new(100);
        let decay_fn = DecayFunction::Linear { slope: 0.2 };

        let mut visited_nodes = Vec::new();
        let seed_nodes = vec![("A".to_string(), 1.0)];

        dfs.traverse(
            &graph,
            &seed_nodes,
            3,
            &decay_fn,
            |node_id, activation, depth| {
                visited_nodes.push((node_id.clone(), activation, depth));
                true
            },
        )
        .unwrap();

        assert!(!visited_nodes.is_empty());

        let visited_set = dfs.get_visited();
        assert!(visited_set.contains("A"));

        dfs.reset();
        assert!(dfs.get_visited().is_empty());
    }

    #[test]
    fn test_edge_type_calculation() {
        let excitatory =
            BreadthFirstTraversal::calculate_activation(1.0, 0.5, 0.8, EdgeType::Excitatory);
        assert!(excitatory > 0.0);
        assert!((excitatory - 0.4).abs() < 1e-6);

        let inhibitory =
            BreadthFirstTraversal::calculate_activation(1.0, 0.5, 0.8, EdgeType::Inhibitory);
        assert!(inhibitory < 0.0);
        assert!((inhibitory + 0.4).abs() < 1e-6);

        let modulatory =
            BreadthFirstTraversal::calculate_activation(1.0, 0.5, 0.8, EdgeType::Modulatory);
        assert!((modulatory - 0.2).abs() < 1e-6);
    }

    #[test]
    fn test_adaptive_traversal() {
        let graph = create_test_graph();
        let adaptive = AdaptiveTraversal::new(3, 2, 100, 2.0);
        let decay_fn = DecayFunction::PowerLaw { exponent: 1.5 };

        let mut visited_nodes = Vec::new();
        let seed_nodes = vec![("A".to_string(), 1.0)];

        adaptive
            .traverse(
                &graph,
                &seed_nodes,
                2,
                &decay_fn,
                |node_id, activation, depth| {
                    visited_nodes.push((node_id.clone(), activation, depth));
                    true
                },
            )
            .unwrap();

        assert!(!visited_nodes.is_empty());

        adaptive.reset();
    }

    #[test]
    fn test_branching_factor_estimation() {
        let graph = create_test_graph();

        let seed_nodes = vec![("A".to_string(), 1.0)];
        let branching_factor = AdaptiveTraversal::estimate_branching_factor(&graph, &seed_nodes);

        // Node A has 2 neighbors (B and D)
        assert!((branching_factor - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_cycle_detection() {
        use crate::activation::{ActivationGraphExt, create_activation_graph};
        let graph = create_activation_graph();

        // Create cycle: A -> B -> A
        // Use the ActivationGraphExt trait for String-based node IDs
        ActivationGraphExt::add_edge(
            &graph,
            "A".to_string(),
            "B".to_string(),
            0.8,
            EdgeType::Excitatory,
        );
        ActivationGraphExt::add_edge(
            &graph,
            "B".to_string(),
            "A".to_string(),
            0.8,
            EdgeType::Excitatory,
        );

        let bfs = BreadthFirstTraversal::with_uniform_budget(2, 1); // Max 2 visits
        let decay_fn = DecayFunction::Exponential { rate: 0.3 };

        let mut visit_count = 0;
        let seed_nodes = vec![("A".to_string(), 1.0)];

        bfs.traverse(
            &graph,
            &seed_nodes,
            5, // Deep enough to trigger cycle detection
            &decay_fn,
            |_, _, _| {
                visit_count += 1;
                true
            },
        )
        .unwrap();

        // Should have limited visits due to cycle detection
        let visit_stats = bfs.get_visit_stats();
        for (_, visits) in visit_stats {
            assert!(visits <= 2);
        }
    }
}
