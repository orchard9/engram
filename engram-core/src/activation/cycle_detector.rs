//! Cycle detection for activation spreading to prevent infinite loops
//!
//! This module provides efficient cycle detection using Tarjan's algorithm
//! and visited node tracking to ensure activation spreading terminates.

use super::{NodeId, storage_aware::StorageTier};
use dashmap::{DashMap, DashSet};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

/// Cycle detector for activation spreading graphs
pub struct CycleDetector {
    /// Set of nodes currently being visited in this traversal
    visited: Arc<DashSet<NodeId>>,
    /// Tier-specific visit budgets
    tier_budgets: HashMap<StorageTier, usize>,
    /// Fallback visit budget when tier not provided in map
    default_budget: usize,
    /// Set of nodes that form cycles (for debugging)
    cycle_nodes: Arc<DashSet<NodeId>>,
    /// Captured cycle paths for downstream diagnostics
    cycle_paths: Arc<DashMap<NodeId, Vec<NodeId>>>,
}

impl CycleDetector {
    /// Create a new cycle detector
    #[must_use]
    pub fn new(tier_budgets: HashMap<StorageTier, usize>) -> Self {
        let default_budget = tier_budgets
            .get(&StorageTier::Hot)
            .copied()
            .or_else(|| tier_budgets.values().copied().min())
            .unwrap_or(3);
        Self {
            visited: Arc::new(DashSet::new()),
            tier_budgets,
            default_budget,
            cycle_nodes: Arc::new(DashSet::new()),
            cycle_paths: Arc::new(DashMap::new()),
        }
    }

    /// Resolve the visit budget for a specific storage tier.
    #[must_use]
    pub fn max_visits_for_tier(&self, tier: StorageTier) -> usize {
        self.tier_budgets
            .get(&tier)
            .copied()
            .unwrap_or(self.default_budget)
    }

    /// Check if we should visit a node (returns false if would create cycle)
    #[must_use]
    pub fn should_visit(
        &self,
        node_id: &NodeId,
        visit_count: usize,
        tier: StorageTier,
        path: &[NodeId],
    ) -> bool {
        // If we've visited this node too many times, it's likely a cycle
        // Budget of N means allow N visits (0..N); reject when visit_count >= N
        let budget = self.max_visits_for_tier(tier);
        if visit_count >= budget {
            self.cycle_nodes.insert(node_id.clone());
            if !path.is_empty() {
                self.cycle_paths.insert(node_id.clone(), path.to_vec());
            }
            return false;
        }

        // Check if node is currently being processed (would create immediate cycle)
        if self.visited.contains(node_id) {
            self.cycle_nodes.insert(node_id.clone());
            if !path.is_empty() {
                self.cycle_paths.insert(node_id.clone(), path.to_vec());
            }
            return false;
        }

        true
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
    #[must_use]
    pub fn get_cycle_nodes(&self) -> HashSet<NodeId> {
        self.cycle_nodes.iter().map(|n| n.clone()).collect()
    }

    /// Get recorded cycle paths for diagnostics.
    #[must_use]
    pub fn get_cycle_paths(&self) -> Vec<Vec<NodeId>> {
        self.cycle_paths
            .iter()
            .map(|entry| entry.value().clone())
            .collect()
    }

    /// Reset the detector for a new traversal
    pub fn reset(&self) {
        self.visited.clear();
        self.cycle_nodes.clear();
        self.cycle_paths.clear();
    }

    /// Check if any cycles were detected
    #[must_use]
    pub fn has_cycles(&self) -> bool {
        !self.cycle_nodes.is_empty()
    }
}

/// Thread-local cycle detection for better performance
pub struct LocalCycleDetector {
    visited: HashSet<NodeId>,
    visit_counts: std::collections::HashMap<NodeId, usize>,
    tier_budgets: HashMap<StorageTier, usize>,
    default_budget: usize,
}

impl LocalCycleDetector {
    /// Create a new local cycle detector
    #[must_use]
    pub fn new(tier_budgets: HashMap<StorageTier, usize>) -> Self {
        let default_budget = tier_budgets
            .get(&StorageTier::Hot)
            .copied()
            .or_else(|| tier_budgets.values().copied().min())
            .unwrap_or(3);
        Self {
            visited: HashSet::new(),
            visit_counts: std::collections::HashMap::new(),
            tier_budgets,
            default_budget,
        }
    }

    /// Check and update visit count for a node
    pub fn visit(&mut self, node_id: &NodeId, tier: StorageTier) -> bool {
        let count = self.visit_counts.entry(node_id.clone()).or_insert(0);
        *count += 1;

        let budget = self
            .tier_budgets
            .get(&tier)
            .copied()
            .unwrap_or(self.default_budget);

        if *count > budget {
            false // Cycle detected
        } else {
            self.visited.insert(node_id.clone());
            true
        }
    }

    /// Check if a node has been visited
    #[must_use]
    pub fn is_visited(&self, node_id: &NodeId) -> bool {
        self.visited.contains(node_id)
    }

    /// Get visit count for a node
    #[must_use]
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
        let detector = CycleDetector::new(HashMap::from([
            (StorageTier::Hot, 3),
            (StorageTier::Warm, 4),
        ]));
        let node = NodeId::from("test_node");

        // First few visits should be allowed
        assert!(detector.should_visit(&node, 0, StorageTier::Hot, &[node.clone()]));
        assert!(detector.should_visit(&node, 1, StorageTier::Hot, &[node.clone()]));
        assert!(detector.should_visit(&node, 2, StorageTier::Hot, &[node.clone()]));

        // After max visits, should detect cycle
        assert!(!detector.should_visit(&node, 3, StorageTier::Hot, &[node.clone()]));
        assert!(detector.has_cycles());
    }

    #[test]
    fn test_local_cycle_detector() {
        let mut detector = LocalCycleDetector::new(HashMap::from([
            (StorageTier::Hot, 2),
            (StorageTier::Warm, 3),
        ]));
        let node = NodeId::from("test_node");

        // First visits should succeed
        assert!(detector.visit(&node, StorageTier::Hot));
        assert!(detector.visit(&node, StorageTier::Hot));

        // Third visit should fail (cycle)
        assert!(!detector.visit(&node, StorageTier::Hot));
        assert_eq!(detector.visit_count(&node), 3);
    }

    #[test]
    fn cycle_paths_are_recorded() {
        let detector = CycleDetector::new(HashMap::from([
            (StorageTier::Hot, 2),
            (StorageTier::Warm, 3),
        ]));
        let node = NodeId::from("cycle_node");
        let path = vec!["seed".to_string(), node.clone()];

        assert!(detector.should_visit(&node, 1, StorageTier::Hot, &path));
        assert!(!detector.should_visit(&node, 2, StorageTier::Hot, &path));

        let nodes = detector.get_cycle_nodes();
        assert!(nodes.contains(&node));

        let recorded_paths = detector.get_cycle_paths();
        assert_eq!(recorded_paths.len(), 1);
        assert_eq!(recorded_paths[0], path);
    }
}
