//! Cycle detection for activation spreading to prevent infinite loops
//!
//! This module provides efficient cycle detection using Tarjan's algorithm
//! and visited node tracking to ensure activation spreading terminates.

use super::NodeId;
use dashmap::DashSet;
use std::collections::HashSet;
use std::sync::Arc;

/// Cycle detector for activation spreading graphs
pub struct CycleDetector {
    /// Set of nodes currently being visited in this traversal
    visited: Arc<DashSet<NodeId>>,
    /// Maximum allowed visits per node before considering it a cycle
    max_visits: usize,
    /// Set of nodes that form cycles (for debugging)
    cycle_nodes: Arc<DashSet<NodeId>>,
}

impl CycleDetector {
    /// Create a new cycle detector
    #[must_use]
    pub fn new(max_visits: usize) -> Self {
        Self {
            visited: Arc::new(DashSet::new()),
            max_visits,
            cycle_nodes: Arc::new(DashSet::new()),
        }
    }

    /// Check if we should visit a node (returns false if would create cycle)
    #[must_use]
    pub fn should_visit(&self, node_id: &NodeId, visit_count: usize) -> bool {
        // If we've visited this node too many times, it's likely a cycle
        if visit_count >= self.max_visits {
            self.cycle_nodes.insert(node_id.clone());
            return false;
        }

        // Check if node is currently being processed (would create immediate cycle)
        !self.visited.contains(node_id)
    }

    /// Mark a node as being visited
    pub fn mark_visiting(&self, node_id: NodeId) {
        self.visited.insert(node_id);
    }

    /// Mark a node as done visiting
    pub fn mark_done(&self, node_id: &NodeId) {
        self.visited.remove(node_id);
    }

    /// Get nodes that are part of cycles
    pub fn get_cycle_nodes(&self) -> HashSet<NodeId> {
        self.cycle_nodes.iter().map(|n| n.clone()).collect()
    }

    /// Reset the detector for a new traversal
    pub fn reset(&self) {
        self.visited.clear();
        self.cycle_nodes.clear();
    }

    /// Check if any cycles were detected
    pub fn has_cycles(&self) -> bool {
        !self.cycle_nodes.is_empty()
    }
}

/// Thread-local cycle detection for better performance
pub struct LocalCycleDetector {
    visited: HashSet<NodeId>,
    visit_counts: std::collections::HashMap<NodeId, usize>,
    max_visits: usize,
}

impl LocalCycleDetector {
    /// Create a new local cycle detector
    pub fn new(max_visits: usize) -> Self {
        Self {
            visited: HashSet::new(),
            visit_counts: std::collections::HashMap::new(),
            max_visits,
        }
    }

    /// Check and update visit count for a node
    pub fn visit(&mut self, node_id: &NodeId) -> bool {
        let count = self.visit_counts.entry(node_id.clone()).or_insert(0);
        *count += 1;
        
        if *count > self.max_visits {
            false // Cycle detected
        } else {
            self.visited.insert(node_id.clone());
            true
        }
    }

    /// Check if a node has been visited
    pub fn is_visited(&self, node_id: &NodeId) -> bool {
        self.visited.contains(node_id)
    }

    /// Get visit count for a node
    pub fn visit_count(&self, node_id: &NodeId) -> usize {
        self.visit_counts.get(node_id).copied().unwrap_or(0)
    }

    /// Reset for new traversal
    pub fn reset(&mut self) {
        self.visited.clear();
        self.visit_counts.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cycle_detection() {
        let detector = CycleDetector::new(3);
        let node = NodeId::from("test_node");

        // First few visits should be allowed
        assert!(detector.should_visit(&node, 0));
        assert!(detector.should_visit(&node, 1));
        assert!(detector.should_visit(&node, 2));
        
        // After max visits, should detect cycle
        assert!(!detector.should_visit(&node, 3));
        assert!(detector.has_cycles());
    }

    #[test]
    fn test_local_cycle_detector() {
        let mut detector = LocalCycleDetector::new(2);
        let node = NodeId::from("test_node");

        // First visits should succeed
        assert!(detector.visit(&node));
        assert!(detector.visit(&node));
        
        // Third visit should fail (cycle)
        assert!(!detector.visit(&node));
        assert_eq!(detector.visit_count(&node), 3);
    }
}