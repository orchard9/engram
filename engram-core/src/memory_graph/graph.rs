//! Unified memory graph implementation

use crate::memory::{Episode, Memory};
use crate::memory_graph::{GraphBackend, GraphConfig, MemoryBackend, MemoryError};
use crate::{Confidence, Cue, CueType};
use std::sync::Arc;
use uuid::Uuid;

/// Unified memory graph that works with any backend
///
/// This is the main entry point for all memory graph operations.
/// The backend can be swapped without changing client code.
pub struct UnifiedMemoryGraph<B: MemoryBackend> {
    backend: Arc<B>,
    config: GraphConfig,
}

impl<B: MemoryBackend> UnifiedMemoryGraph<B> {
    /// Create a new memory graph with the specified backend and configuration
    #[must_use]
    pub fn new(backend: B, config: GraphConfig) -> Self {
        Self {
            backend: Arc::new(backend),
            config,
        }
    }

    /// Create with default configuration
    #[must_use]
    pub fn with_backend(backend: B) -> Self {
        Self::new(backend, GraphConfig::default())
    }

    /// Get a reference to the backend
    #[must_use]
    pub fn backend(&self) -> &B {
        &self.backend
    }

    /// Get the current configuration
    #[must_use]
    pub const fn config(&self) -> &GraphConfig {
        &self.config
    }

    /// Update the configuration
    #[allow(clippy::missing_const_for_fn)]
    pub fn set_config(&mut self, config: GraphConfig) {
        self.config = config;
    }

    /// Store a memory and return its ID
    ///
    /// # Errors
    ///
    /// Returns a [`MemoryError`] if the backend fails to persist the memory or
    /// if any backend preconditions are violated.
    #[must_use = "Handle the result to detect memory graph errors"]
    pub fn store_memory(&self, memory: Memory) -> Result<Uuid, MemoryError> {
        let id = Uuid::new_v4();
        self.backend.store(id, memory)?;
        Ok(id)
    }

    /// Store an episode as a memory
    ///
    /// # Errors
    ///
    /// Returns a [`MemoryError`] if storing the converted memory fails.
    #[must_use = "Handle the result to detect memory graph errors"]
    pub fn store_episode(&self, episode: Episode) -> Result<Uuid, MemoryError> {
        let memory = Memory::from_episode(episode, 1.0); // Default activation
        self.store_memory(memory)
    }

    /// Retrieve a memory by ID
    ///
    /// # Errors
    ///
    /// Returns a [`MemoryError`] if the backend encounters storage failures
    /// while fetching the memory.
    #[must_use = "Handle the result to detect memory graph errors"]
    pub fn retrieve(&self, id: &Uuid) -> Result<Option<Arc<Memory>>, MemoryError> {
        self.backend.retrieve(id)
    }

    /// Remove a memory by ID
    ///
    /// # Errors
    ///
    /// Returns a [`MemoryError`] when the backend cannot complete the remove
    /// operation, for example due to locking or storage issues.
    #[must_use = "Handle the result to detect memory graph errors"]
    pub fn remove(&self, id: &Uuid) -> Result<Option<Arc<Memory>>, MemoryError> {
        self.backend.remove(id)
    }

    /// Recall memories based on a cue
    ///
    /// # Errors
    ///
    /// Returns a [`MemoryError`] if the backend fails during search,
    /// retrieval, or any graph-specific processing.
    #[must_use = "Handle the result to detect memory graph errors"]
    pub fn recall(&self, cue: &Cue) -> Result<Vec<Arc<Memory>>, MemoryError> {
        let memories = match &cue.cue_type {
            CueType::Embedding { vector, .. } => {
                // Search by embedding similarity
                let results = self.backend.search(vector, cue.max_results)?;

                // Apply spreading activation if enabled and backend supports it
                // Note: This requires the backend to implement GraphBackend trait
                // We'll handle this in the GraphBackend-specific implementation below

                // Retrieve memories that meet threshold
                let mut memories = Vec::new();
                for (id, score) in results {
                    if score >= cue.result_threshold.raw()
                        && let Some(memory) = self.backend.retrieve(&id)?
                    {
                        memories.push(memory);
                    }
                }
                memories
            }
            CueType::Context { location, .. } => {
                // For context cues, we search based on location if provided
                let all_ids = self.backend.all_ids();
                let mut matches = Vec::new();

                for id in all_ids {
                    if let Some(memory) = self.backend.retrieve(&id)? {
                        // Simple location matching if location is specified
                        let matches_location = location.as_ref().is_none_or(|loc| {
                            memory
                                .content
                                .as_ref()
                                .is_some_and(|content| content.contains(loc))
                        });
                        if matches_location {
                            matches.push(memory);
                        }
                    }

                    if matches.len() >= cue.max_results {
                        break;
                    }
                }
                matches
            }
            CueType::Temporal { .. } => {
                // For temporal pattern matching
                // This is a simplified implementation - real pattern matching would be more complex
                let all_ids = self.backend.all_ids();
                let mut matches = Vec::new();

                for id in all_ids {
                    if let Some(memory) = self.backend.retrieve(&id)? {
                        // For now, just return all memories as pattern matching is not implemented
                        matches.push(memory);
                    }

                    if matches.len() >= cue.max_results {
                        break;
                    }
                }
                matches
            }
            CueType::Semantic { content, .. } => {
                // For semantic content search
                // Supports both content-based matching (neocortical semantic retrieval) and
                // direct ID matching (hippocampal episodic indexing).
                let all_ids = self.backend.all_ids();
                let mut matches = Vec::new();

                for id in all_ids {
                    if let Some(memory) = self.backend.retrieve(&id)? {
                        // Match against memory ID (for direct episodic retrieval)
                        // or content (for semantic search).
                        //
                        // Direct ID matches enable hippocampal-style pattern completion
                        // where node IDs act as sparse indices for rapid, deterministic recall.
                        let matches_id = memory.id == *content;
                        let matches_content = memory
                            .content
                            .as_ref()
                            .is_some_and(|mem_content| mem_content.contains(content));

                        if matches_id || matches_content {
                            matches.push(memory);
                        }
                    }

                    if matches.len() >= cue.max_results {
                        break;
                    }
                }
                matches
            }
        };

        Ok(memories)
    }

    /// Search for similar memories by embedding
    ///
    /// # Errors
    ///
    /// Returns a [`MemoryError`] if the backend cannot execute the search or
    /// retrieve the matching memories.
    #[must_use = "Handle the result to detect memory graph errors"]
    pub fn similarity_search(
        &self,
        embedding: &[f32],
        k: usize,
        threshold: Confidence,
    ) -> Result<Vec<(Arc<Memory>, f32)>, MemoryError> {
        let results = self.backend.search(embedding, k)?;
        let mut memories = Vec::new();

        for (id, score) in results {
            if score >= threshold.raw()
                && let Some(memory) = self.backend.retrieve(&id)?
            {
                memories.push((memory, score));
            }
        }

        Ok(memories)
    }

    /// Get the current memory count
    #[must_use]
    pub fn count(&self) -> usize {
        self.backend.count()
    }

    /// Clear all memories
    ///
    /// # Errors
    ///
    /// Returns a [`MemoryError`] if the backend fails to clear its state.
    #[must_use = "Handle the result to detect memory graph errors"]
    pub fn clear(&self) -> Result<(), MemoryError> {
        self.backend.clear()
    }

    /// Check if a memory exists
    #[must_use]
    pub fn contains(&self, id: &Uuid) -> bool {
        self.backend.contains(id)
    }
}

/// Extension methods for backends that support graph operations
impl<B: GraphBackend> UnifiedMemoryGraph<B> {
    /// Add an edge between two memories
    ///
    /// # Errors
    ///
    /// Returns a [`MemoryError`] if the backend fails to persist the new edge.
    #[must_use = "Handle the result to detect memory graph errors"]
    pub fn add_edge(&self, from: Uuid, to: Uuid, weight: f32) -> Result<(), MemoryError> {
        self.backend.add_edge(from, to, weight)
    }

    /// Remove an edge between two memories
    ///
    /// # Errors
    ///
    /// Returns a [`MemoryError`] if the backend encounters an issue while removing the edge.
    #[must_use = "Handle the result to detect memory graph errors"]
    pub fn remove_edge(&self, from: &Uuid, to: &Uuid) -> Result<bool, MemoryError> {
        self.backend.remove_edge(from, to)
    }

    /// Get neighbors of a memory node
    ///
    /// # Errors
    ///
    /// Returns a [`MemoryError`] if the backend cannot fetch the adjacency list.
    #[must_use = "Handle the result to detect memory graph errors"]
    pub fn get_neighbors(&self, id: &Uuid) -> Result<Vec<(Uuid, f32)>, MemoryError> {
        self.backend.get_neighbors(id)
    }

    /// Perform breadth-first traversal
    ///
    /// # Errors
    ///
    /// Returns a [`MemoryError`] if the backend traversal fails or access is denied.
    #[must_use = "Handle the result to detect memory graph errors"]
    pub fn traverse_bfs(&self, start: &Uuid, max_depth: usize) -> Result<Vec<Uuid>, MemoryError> {
        self.backend
            .traverse_bfs(start, max_depth.min(self.config.max_depth))
    }

    /// Spread activation from a source node
    ///
    /// # Errors
    ///
    /// Returns a [`MemoryError`] if activation propagation cannot be performed by the backend.
    #[must_use = "Handle the result to detect memory graph errors"]
    pub fn spread_activation(&self, source: &Uuid) -> Result<(), MemoryError> {
        self.backend
            .spread_activation(source, self.config.decay_rate)
    }

    /// Get all edges in the graph
    ///
    /// # Errors
    ///
    /// Returns a [`MemoryError`] if the backend fails to enumerate the stored edges.
    #[must_use = "Handle the result to detect memory graph errors"]
    pub fn all_edges(&self) -> Result<Vec<(Uuid, Uuid, f32)>, MemoryError> {
        self.backend.all_edges()
    }

    /// Get the count of outgoing edges from a node
    ///
    /// Used for computing fan effect (number of associations).
    ///
    /// # Errors
    ///
    /// Returns a [`MemoryError`] if the backend cannot access the node's edges.
    #[must_use = "Handle the result to detect memory graph errors"]
    pub fn get_outgoing_edge_count(&self, id: &Uuid) -> Result<usize, MemoryError> {
        self.backend.get_outgoing_edge_count(id)
    }
}

/// Methods for dual memory backends (episode/concept discrimination)
#[cfg(feature = "dual_memory_types")]
impl<B> UnifiedMemoryGraph<B>
where
    B: crate::memory_graph::DualMemoryBackend,
{
    /// Get the memory node type for a given node ID
    ///
    /// This provides efficient type discrimination without string parsing heuristics.
    /// Uses the backend's type index for O(1) lookup.
    ///
    /// # Returns
    ///
    /// - `Ok(Some(MemoryNodeType))` if the node exists and has type information
    /// - `Ok(None)` if the node doesn't exist
    /// - `Err(MemoryError)` if the backend encounters a storage error
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use engram_core::memory::MemoryNodeType;
    ///
    /// if let Some(node_type) = graph.get_node_type(&node_id)? {
    ///     match node_type {
    ///         MemoryNodeType::Episode { .. } => {
    ///             // Apply upward spreading strength
    ///         }
    ///         MemoryNodeType::Concept { .. } => {
    ///             // Apply downward spreading strength
    ///         }
    ///     }
    /// }
    /// ```
    #[must_use = "Handle the result to detect backend failures"]
    pub fn get_node_type(
        &self,
        id: &Uuid,
    ) -> Result<Option<crate::memory::MemoryNodeType>, MemoryError> {
        match self.backend.get_node_typed(id)? {
            Some(node) => Ok(Some(node.node_type)),
            None => Ok(None),
        }
    }
}

/// Convenience constructors for common backend configurations
impl UnifiedMemoryGraph<crate::memory_graph::backends::HashMapBackend> {
    /// Create a simple single-threaded memory graph
    #[must_use]
    pub fn simple() -> Self {
        Self::with_backend(crate::memory_graph::backends::HashMapBackend::new())
    }
}

impl UnifiedMemoryGraph<crate::memory_graph::backends::DashMapBackend> {
    /// Create a concurrent memory graph optimized for parallel access
    #[must_use]
    pub fn concurrent() -> Self {
        Self::with_backend(crate::memory_graph::backends::DashMapBackend::new())
    }
}

impl UnifiedMemoryGraph<crate::memory_graph::backends::InfallibleBackend> {
    /// Create an infallible memory graph that degrades gracefully
    #[must_use]
    pub fn infallible(max_capacity: usize) -> Self {
        Self::with_backend(crate::memory_graph::backends::InfallibleBackend::new(
            max_capacity,
        ))
    }
}
