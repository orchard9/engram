//! Evidence dependency graph with circular dependency detection
//!
//! Implements Tarjan's strongly connected components algorithm to detect
//! circular dependencies in evidence chains, preventing infinite loops in
//! Bayesian updating.
//!
//! # Example
//!
//! ```
//! use engram_core::query::dependency_graph::DependencyGraph;
//!
//! let mut graph = DependencyGraph::new();
//! graph.add_evidence(0, vec![]);  // Evidence 0 has no dependencies
//! graph.add_evidence(1, vec![0]); // Evidence 1 depends on 0
//! graph.add_evidence(2, vec![1]); // Evidence 2 depends on 1
//!
//! assert!(!graph.has_cycles());
//! ```

use crate::query::EvidenceId;
use std::collections::{HashMap, HashSet};

/// Dependency graph for evidence tracking
///
/// Tracks dependencies between evidence nodes and detects circular
/// dependencies using Tarjan's strongly connected components algorithm.
#[derive(Debug, Clone)]
pub struct DependencyGraph {
    /// Adjacency list: evidence_id -> list of dependencies
    edges: HashMap<EvidenceId, Vec<EvidenceId>>,
    /// All evidence IDs in the graph
    nodes: HashSet<EvidenceId>,
}

impl DependencyGraph {
    /// Create a new empty dependency graph
    #[must_use]
    pub fn new() -> Self {
        Self {
            edges: HashMap::new(),
            nodes: HashSet::new(),
        }
    }

    /// Add evidence with its dependencies to the graph
    ///
    /// # Arguments
    ///
    /// * `evidence_id` - Unique identifier for the evidence
    /// * `dependencies` - List of evidence IDs that this evidence depends on
    pub fn add_evidence(&mut self, evidence_id: EvidenceId, dependencies: Vec<EvidenceId>) {
        self.nodes.insert(evidence_id);
        for dep in &dependencies {
            self.nodes.insert(*dep);
        }
        self.edges.insert(evidence_id, dependencies);
    }

    /// Check if the graph contains any circular dependencies
    ///
    /// Uses Tarjan's algorithm to find strongly connected components.
    /// Any SCC with more than one node, or a single node with a self-loop,
    /// indicates a cycle.
    #[must_use]
    pub fn has_cycles(&self) -> bool {
        let sccs = self.tarjan_scc();
        sccs.iter().any(|scc| {
            match scc.len().cmp(&1) {
                std::cmp::Ordering::Greater => true,
                std::cmp::Ordering::Equal => {
                    // Check for self-loop
                    let node = scc[0];
                    self.edges
                        .get(&node)
                        .is_some_and(|deps| deps.contains(&node))
                }
                std::cmp::Ordering::Less => false,
            }
        })
    }

    /// Find all strongly connected components using Tarjan's algorithm
    ///
    /// Returns a vector of SCCs, where each SCC is a vector of evidence IDs
    /// that form a cycle (or are part of the same strongly connected subgraph).
    #[must_use]
    pub fn find_cycles(&self) -> Vec<Vec<EvidenceId>> {
        self.tarjan_scc()
            .into_iter()
            .filter(|scc| {
                match scc.len().cmp(&1) {
                    std::cmp::Ordering::Greater => true,
                    std::cmp::Ordering::Equal => {
                        // Check for self-loop
                        let node = scc[0];
                        self.edges
                            .get(&node)
                            .is_some_and(|deps| deps.contains(&node))
                    }
                    std::cmp::Ordering::Less => false,
                }
            })
            .collect()
    }

    /// Tarjan's strongly connected components algorithm
    ///
    /// Finds all strongly connected components in O(V + E) time.
    fn tarjan_scc(&self) -> Vec<Vec<EvidenceId>> {
        let mut state = TarjanState::new();
        let mut sccs = Vec::new();

        for &node in &self.nodes {
            if !state.visited.contains(&node) {
                self.tarjan_visit(node, &mut state, &mut sccs);
            }
        }

        sccs
    }

    /// Recursive visit for Tarjan's algorithm
    fn tarjan_visit(
        &self,
        node: EvidenceId,
        state: &mut TarjanState,
        sccs: &mut Vec<Vec<EvidenceId>>,
    ) {
        let index = state.index;
        state.index += 1;
        state.indices.insert(node, index);
        state.lowlinks.insert(node, index);
        state.visited.insert(node);
        state.stack.push(node);
        state.on_stack.insert(node);

        // Visit all dependencies
        if let Some(deps) = self.edges.get(&node) {
            for &dep in deps {
                if !state.visited.contains(&dep) {
                    self.tarjan_visit(dep, state, sccs);
                    let dep_lowlink = *state.lowlinks.get(&dep).unwrap_or(&0);
                    let node_lowlink = state.lowlinks.get(&node).copied().unwrap_or(0);
                    state.lowlinks.insert(node, node_lowlink.min(dep_lowlink));
                } else if state.on_stack.contains(&dep) {
                    let dep_index = *state.indices.get(&dep).unwrap_or(&0);
                    let node_lowlink = state.lowlinks.get(&node).copied().unwrap_or(0);
                    state.lowlinks.insert(node, node_lowlink.min(dep_index));
                }
            }
        }

        // If node is a root node, pop the stack to get SCC
        let node_index = *state.indices.get(&node).unwrap_or(&0);
        let node_lowlink = *state.lowlinks.get(&node).unwrap_or(&0);

        if node_lowlink == node_index {
            let mut scc = Vec::new();
            while let Some(w) = state.stack.pop() {
                state.on_stack.remove(&w);
                scc.push(w);
                if w == node {
                    break;
                }
            }
            sccs.push(scc);
        }
    }

    /// Get topological ordering of evidence (if acyclic)
    ///
    /// Returns `None` if the graph contains cycles.
    /// Returns `Some(ordering)` with evidence IDs in dependency order
    /// (dependencies appear before dependents).
    #[must_use]
    pub fn topological_sort(&self) -> Option<Vec<EvidenceId>> {
        if self.has_cycles() {
            return None;
        }

        let mut visited = HashSet::new();
        let mut stack = Vec::new();

        for &node in &self.nodes {
            if !visited.contains(&node) {
                self.topo_visit(node, &mut visited, &mut stack);
            }
        }

        // DFS post-order gives us dependencies before dependents
        Some(stack)
    }

    /// Recursive visit for topological sort
    fn topo_visit(
        &self,
        node: EvidenceId,
        visited: &mut HashSet<EvidenceId>,
        stack: &mut Vec<EvidenceId>,
    ) {
        visited.insert(node);

        if let Some(deps) = self.edges.get(&node) {
            for &dep in deps {
                if !visited.contains(&dep) {
                    self.topo_visit(dep, visited, stack);
                }
            }
        }

        stack.push(node);
    }

    /// Get all evidence IDs in the graph
    #[must_use]
    pub fn evidence_ids(&self) -> Vec<EvidenceId> {
        self.nodes.iter().copied().collect()
    }

    /// Get dependencies for a specific evidence
    #[must_use]
    pub fn dependencies(&self, evidence_id: EvidenceId) -> Option<&[EvidenceId]> {
        self.edges.get(&evidence_id).map(std::vec::Vec::as_slice)
    }
}

impl Default for DependencyGraph {
    fn default() -> Self {
        Self::new()
    }
}

/// State for Tarjan's algorithm
struct TarjanState {
    index: usize,
    indices: HashMap<EvidenceId, usize>,
    lowlinks: HashMap<EvidenceId, usize>,
    visited: HashSet<EvidenceId>,
    stack: Vec<EvidenceId>,
    on_stack: HashSet<EvidenceId>,
}

impl TarjanState {
    fn new() -> Self {
        Self {
            index: 0,
            indices: HashMap::new(),
            lowlinks: HashMap::new(),
            visited: HashSet::new(),
            stack: Vec::new(),
            on_stack: HashSet::new(),
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)] // Unwrap is acceptable in tests
mod tests {
    use super::*;

    #[test]
    fn test_empty_graph_has_no_cycles() {
        let graph = DependencyGraph::new();
        assert!(!graph.has_cycles());
    }

    #[test]
    fn test_single_node_no_cycle() {
        let mut graph = DependencyGraph::new();
        graph.add_evidence(0, vec![]);
        assert!(!graph.has_cycles());
    }

    #[test]
    fn test_linear_chain_no_cycle() {
        let mut graph = DependencyGraph::new();
        graph.add_evidence(0, vec![]);
        graph.add_evidence(1, vec![0]);
        graph.add_evidence(2, vec![1]);
        graph.add_evidence(3, vec![2]);
        assert!(!graph.has_cycles());
    }

    #[test]
    fn test_simple_cycle_detected() {
        let mut graph = DependencyGraph::new();
        graph.add_evidence(0, vec![1]);
        graph.add_evidence(1, vec![0]);
        assert!(graph.has_cycles());
    }

    #[test]
    fn test_three_node_cycle_detected() {
        let mut graph = DependencyGraph::new();
        graph.add_evidence(0, vec![1]);
        graph.add_evidence(1, vec![2]);
        graph.add_evidence(2, vec![0]);
        assert!(graph.has_cycles());
    }

    #[test]
    fn test_self_loop_detected() {
        let mut graph = DependencyGraph::new();
        graph.add_evidence(0, vec![0]);
        assert!(graph.has_cycles());
    }

    #[test]
    fn test_dag_with_multiple_paths() {
        let mut graph = DependencyGraph::new();
        graph.add_evidence(0, vec![]);
        graph.add_evidence(1, vec![]);
        graph.add_evidence(2, vec![0, 1]);
        graph.add_evidence(3, vec![2]);
        assert!(!graph.has_cycles());
    }

    #[test]
    fn test_find_cycles_returns_sccs() {
        let mut graph = DependencyGraph::new();
        graph.add_evidence(0, vec![1]);
        graph.add_evidence(1, vec![0]);
        graph.add_evidence(2, vec![]);

        let cycles = graph.find_cycles();
        assert_eq!(cycles.len(), 1);
        assert_eq!(cycles[0].len(), 2);
    }

    #[test]
    fn test_topological_sort_acyclic() {
        let mut graph = DependencyGraph::new();
        graph.add_evidence(0, vec![]);
        graph.add_evidence(1, vec![0]);
        graph.add_evidence(2, vec![1]);

        let sorted = graph.topological_sort();
        assert!(sorted.is_some());

        let order = sorted.unwrap();
        assert_eq!(order.len(), 3);

        // Dependencies should come before dependents
        let pos_0 = order.iter().position(|&x| x == 0).unwrap();
        let pos_1 = order.iter().position(|&x| x == 1).unwrap();
        let pos_2 = order.iter().position(|&x| x == 2).unwrap();

        assert!(pos_0 < pos_1);
        assert!(pos_1 < pos_2);
    }

    #[test]
    fn test_topological_sort_cyclic_returns_none() {
        let mut graph = DependencyGraph::new();
        graph.add_evidence(0, vec![1]);
        graph.add_evidence(1, vec![0]);

        assert!(graph.topological_sort().is_none());
    }

    #[test]
    fn test_dependencies_retrieval() {
        let mut graph = DependencyGraph::new();
        graph.add_evidence(0, vec![]);
        graph.add_evidence(1, vec![0, 2]);

        assert_eq!(graph.dependencies(0), Some(&[][..]));
        assert_eq!(graph.dependencies(1), Some(&[0, 2][..]));
        assert_eq!(graph.dependencies(999), None);
    }

    #[test]
    fn test_evidence_ids() {
        let mut graph = DependencyGraph::new();
        graph.add_evidence(5, vec![]);
        graph.add_evidence(10, vec![5]);
        graph.add_evidence(15, vec![]);

        let ids = graph.evidence_ids();
        assert_eq!(ids.len(), 3);
        assert!(ids.contains(&5));
        assert!(ids.contains(&10));
        assert!(ids.contains(&15));
    }
}
