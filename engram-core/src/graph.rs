//! Graph structures for cognitive memory operations.

use crate::MemoryNode;
use std::collections::HashMap;

/// Memory graph structure for storing and retrieving memories.
///
/// This is a placeholder implementation that will be expanded
/// with spreading activation, pattern completion, and consolidation.
pub struct MemoryGraph {
    nodes: HashMap<String, MemoryNode>,
}

impl MemoryGraph {
    /// Create a new empty memory graph.
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
        }
    }

    /// Store a memory node in the graph.
    pub fn store(&mut self, node: MemoryNode) {
        self.nodes.insert(node.id.clone(), node);
    }

    /// Retrieve a memory node by ID.
    pub fn get(&self, id: &str) -> Option<&MemoryNode> {
        self.nodes.get(id)
    }

    /// Get the number of nodes in the graph.
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Check if the graph is empty.
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }
}

impl Default for MemoryGraph {
    fn default() -> Self {
        Self::new()
    }
}
