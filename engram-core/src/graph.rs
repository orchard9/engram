//! Graph structures for cognitive memory operations.
//!
//! This module provides a migration path from the old MemoryGraph to the new
//! unified memory graph architecture.

use crate::MemoryNode;
use std::collections::HashMap;

// Re-export the new unified memory graph types
pub use crate::memory_graph::{
    UnifiedMemoryGraph, HashMapBackend, DashMapBackend, InfallibleBackend,
    GraphConfig, MemoryBackend, GraphBackend, MemoryError,
};

/// Legacy memory graph structure for backward compatibility.
///
/// This implementation is deprecated and will be removed in the next major version.
/// Please migrate to `UnifiedMemoryGraph<HashMapBackend>` for single-threaded use
/// or `UnifiedMemoryGraph<DashMapBackend>` for concurrent access.
#[deprecated(
    since = "0.2.0",
    note = "Use UnifiedMemoryGraph<HashMapBackend> for single-threaded or UnifiedMemoryGraph<DashMapBackend> for concurrent access"
)]
pub struct MemoryGraph {
    nodes: HashMap<String, MemoryNode>,
}

#[allow(deprecated)]
impl MemoryGraph {
    /// Create a new empty memory graph.
    #[must_use]
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
    #[must_use]
    pub fn get(&self, id: &str) -> Option<&MemoryNode> {
        self.nodes.get(id)
    }

    /// Get the number of nodes in the graph.
    #[must_use]
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Check if the graph is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }
}

#[allow(deprecated)]
impl Default for MemoryGraph {
    fn default() -> Self {
        Self::new()
    }
}

/// Create a single-threaded memory graph with default configuration.
///
/// This is the recommended replacement for the deprecated MemoryGraph.
pub fn create_simple_graph() -> UnifiedMemoryGraph<HashMapBackend> {
    UnifiedMemoryGraph::new(HashMapBackend::default(), GraphConfig::default())
}

/// Create a concurrent memory graph with default configuration.
///
/// Use this for parallel activation spreading and concurrent access patterns.
pub fn create_concurrent_graph() -> UnifiedMemoryGraph<DashMapBackend> {
    UnifiedMemoryGraph::new(DashMapBackend::default(), GraphConfig::default())
}
