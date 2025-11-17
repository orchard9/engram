use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;

use crate::MemorySpaceId;

use super::ClusterError;

/// Per-node logical clock used to track causal relationships between primaries.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct VectorClock {
    clocks: HashMap<String, u64>,
}

impl VectorClock {
    /// Create an empty vector clock.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Increment the local clock entry.
    pub fn tick(&mut self, node_id: &str) {
        *self.clocks.entry(node_id.to_string()).or_insert(0) += 1;
    }

    /// Merge another clock into this one, keeping the maximum for each node.
    pub fn merge(&mut self, other: &Self) {
        for (node, &clock) in &other.clocks {
            let entry = self.clocks.entry(node.clone()).or_insert(0);
            *entry = (*entry).max(clock);
        }
    }

    /// Compare two clocks to determine their causal relationship.
    #[must_use]
    pub fn compare(&self, other: &Self) -> CausalOrdering {
        let mut less = false;
        let mut greater = false;
        let nodes: HashSet<_> = self
            .clocks
            .keys()
            .chain(other.clocks.keys())
            .cloned()
            .collect();

        for node in nodes {
            let left = self.clocks.get(&node).copied().unwrap_or(0);
            let right = other.clocks.get(&node).copied().unwrap_or(0);
            if left < right {
                less = true;
            } else if left > right {
                greater = true;
            }
        }

        match (less, greater) {
            (false, false) => CausalOrdering::Equal,
            (true, false) => CausalOrdering::Less,
            (false, true) => CausalOrdering::Greater,
            (true, true) => CausalOrdering::Concurrent,
        }
    }
}

/// Relationship between two vector clocks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CausalOrdering {
    /// `self` happened-before `other`.
    Less,
    /// Clocks are identical.
    Equal,
    /// `self` happened-after `other`.
    Greater,
    /// Neither clock dominates, implying a concurrent update (potential split-brain).
    Concurrent,
}

/// Detects conflicting primaries using per-space vector clocks.
pub struct SplitBrainDetector {
    node_id: String,
    spaces: DashMap<String, Arc<RwLock<VectorClock>>>,
}

impl SplitBrainDetector {
    /// Create a detector bound to the provided node identifier.
    #[must_use]
    pub fn new(node_id: impl Into<String>) -> Self {
        Self {
            node_id: node_id.into(),
            spaces: DashMap::new(),
        }
    }

    /// Increment the local clock for a write handled on this node.
    pub async fn on_write(&self, space_id: &MemorySpaceId) {
        let clock = self.clock_for(space_id);
        let mut guard = clock.write().await;
        guard.tick(&self.node_id);
    }

    /// Merge remote clocks received via replication or anti-entropy.
    pub async fn on_receive(&self, space_id: &MemorySpaceId, remote_clock: &VectorClock) {
        let clock = self.clock_for(space_id);
        let mut guard = clock.write().await;
        guard.merge(remote_clock);
    }

    /// Check whether a remote primary conflicts with our local view.
    pub async fn check_for_split_brain(
        &self,
        space_id: &MemorySpaceId,
        remote_clock: &VectorClock,
    ) -> Result<(), ClusterError> {
        let local_clock = self.clock_for(space_id);
        let guard = local_clock.read().await;
        if guard.compare(remote_clock) == CausalOrdering::Concurrent {
            return Err(ClusterError::SplitBrain {
                space_id: space_id.clone(),
                local_clock: guard.clone(),
                remote_clock: remote_clock.clone(),
            });
        }
        drop(guard);
        Ok(())
    }

    /// Snapshot the current clock for diagnostics or replication metadata.
    pub async fn snapshot(&self, space_id: &MemorySpaceId) -> VectorClock {
        let clock = self.clock_for(space_id);
        clock.read().await.clone()
    }

    fn clock_for(&self, space_id: &MemorySpaceId) -> Arc<RwLock<VectorClock>> {
        let space = space_id.as_str().to_string();
        let entry = self
            .spaces
            .entry(space)
            .or_insert_with(|| Arc::new(RwLock::new(VectorClock::default())));
        Arc::clone(&*entry)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_concurrent_clocks() {
        let mut clock_a = VectorClock::new();
        let mut clock_b = VectorClock::new();
        clock_a.tick("node-a");
        clock_b.tick("node-b");
        assert_eq!(clock_a.compare(&clock_b), CausalOrdering::Concurrent);
    }
}
