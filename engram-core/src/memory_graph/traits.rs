//! Core traits for memory backend implementations

use crate::Cue;
use crate::memory::Memory;
use std::sync::Arc;
use thiserror::Error;
use uuid::Uuid;

#[cfg(feature = "dual_memory_types")]
use crate::memory::DualMemoryNode;

/// Errors that can occur in memory operations
#[derive(Error, Debug)]
pub enum MemoryError {
    #[error("Memory not found: {0}")]
    /// Occurs when a requested memory doesn't exist in the graph
    NotFound(Uuid),

    #[error("Storage error: {0}")]
    /// Occurs when underlying storage operations fail
    StorageError(String),

    #[error("Index error: {0}")]
    /// Occurs when index operations fail
    IndexError(String),

    #[error("Lock poisoned: {0}")]
    /// Occurs when a synchronization primitive is poisoned
    LockPoisoned(String),

    #[error("Capacity exceeded: current {current}, max {max}")]
    /// Occurs when memory graph capacity limits are exceeded
    CapacityExceeded {
        /// Current number of items
        current: usize,
        /// Maximum allowed items
        max: usize,
    },

    #[error("Invalid embedding dimension: expected 768, got {0}")]
    /// Occurs when embedding has wrong dimension
    InvalidEmbeddingDimension(usize),

    #[error("Type conversion error: {0}")]
    /// Occurs when converting between Memory and DualMemoryNode types fails
    TypeConversionError(String),

    #[error("Unsupported operation for memory type: {0}")]
    /// Occurs when an operation is not supported for the memory node type
    UnsupportedTypeOperation(String),
}

/// Core trait for memory storage backends
pub trait MemoryBackend: Send + Sync {
    /// Store a memory with the given ID
    ///
    /// # Errors
    ///
    /// Returns [`MemoryError`] if the backend cannot persist the memory or its
    /// associated metadata.
    #[must_use = "Handle the result to detect backend failures"]
    fn store(&self, id: Uuid, memory: Memory) -> Result<(), MemoryError>;

    /// Retrieve a memory by ID
    ///
    /// # Errors
    ///
    /// Returns [`MemoryError`] if the backend encounters a storage failure
    /// while loading the memory.
    #[must_use = "Handle the result to detect backend failures"]
    fn retrieve(&self, id: &Uuid) -> Result<Option<Arc<Memory>>, MemoryError>;

    /// Remove a memory by ID
    ///
    /// # Errors
    ///
    /// Returns [`MemoryError`] if the backend fails to remove the memory and
    /// clean up its related state.
    #[must_use = "Handle the result to detect backend failures"]
    fn remove(&self, id: &Uuid) -> Result<Option<Arc<Memory>>, MemoryError>;

    /// Search for memories by embedding similarity
    ///
    /// # Errors
    ///
    /// Returns [`MemoryError`] if similarity search fails, such as when
    /// embeddings are invalid or the backend cannot access indexing data.
    #[must_use = "Handle the result to detect backend failures"]
    fn search(&self, embedding: &[f32], k: usize) -> Result<Vec<(Uuid, f32)>, MemoryError>;

    /// Update the activation level of a memory
    ///
    /// # Errors
    ///
    /// Returns [`MemoryError`] if the targeted memory cannot be updated.
    #[must_use = "Handle the result to detect backend failures"]
    fn update_activation(&self, id: &Uuid, activation: f32) -> Result<(), MemoryError>;

    /// Get the current memory count
    #[must_use]
    fn count(&self) -> usize;

    /// Clear all memories
    ///
    /// # Errors
    ///
    /// Returns [`MemoryError`] if the backend cannot clear its state.
    #[must_use = "Handle the result to detect backend failures"]
    fn clear(&self) -> Result<(), MemoryError>;

    /// Check if a memory exists
    #[must_use]
    fn contains(&self, id: &Uuid) -> bool {
        self.retrieve(id).is_ok_and(|memory| memory.is_some())
    }

    /// Get all memory IDs
    #[must_use]
    fn all_ids(&self) -> Vec<Uuid>;

    /// Recall memories based on a cue
    ///
    /// # Errors
    ///
    /// Returns [`MemoryError`] if similarity search or retrieval fails while
    /// assembling the recalled memories.
    #[must_use = "Handle the result to detect backend failures"]
    fn recall(&self, cue: &Cue) -> Result<Vec<Arc<Memory>>, MemoryError> {
        match &cue.cue_type {
            crate::CueType::Embedding { vector, .. } => {
                let results = self.search(vector, cue.max_results)?;
                let mut memories = Vec::new();
                for (id, score) in results {
                    if score >= cue.result_threshold.raw()
                        && let Some(memory) = self.retrieve(&id)?
                    {
                        memories.push(memory);
                    }
                }
                Ok(memories)
            }
            _ => Ok(Vec::new()), // Other cue types handled by extensions
        }
    }
}

/// Extended trait for graph operations on memory backends
pub trait GraphBackend: MemoryBackend {
    /// Add an edge between two memory nodes
    ///
    /// # Errors
    ///
    /// Returns [`MemoryError`] if the backend cannot persist the edge.
    #[must_use = "Handle the result to detect backend failures"]
    fn add_edge(&self, from: Uuid, to: Uuid, weight: f32) -> Result<(), MemoryError>;

    /// Remove an edge between two memory nodes
    ///
    /// # Errors
    ///
    /// Returns [`MemoryError`] if the backend cannot access or modify the
    /// stored edge data.
    #[must_use = "Handle the result to detect backend failures"]
    fn remove_edge(&self, from: &Uuid, to: &Uuid) -> Result<bool, MemoryError>;

    /// Get all neighbors of a memory node
    ///
    /// # Errors
    ///
    /// Returns [`MemoryError`] if neighbor metadata cannot be retrieved.
    #[must_use = "Handle the result to detect backend failures"]
    fn get_neighbors(&self, id: &Uuid) -> Result<Vec<(Uuid, f32)>, MemoryError>;

    /// Perform breadth-first traversal from a starting node
    ///
    /// # Errors
    ///
    /// Returns [`MemoryError`] if traversal fails to read graph structure.
    #[must_use = "Handle the result to detect backend failures"]
    fn traverse_bfs(&self, start: &Uuid, max_depth: usize) -> Result<Vec<Uuid>, MemoryError>;

    /// Spread activation from a source node
    ///
    /// # Errors
    ///
    /// Returns [`MemoryError`] if neighbor retrieval or activation updates
    /// fail during the spread.
    #[must_use = "Handle the result to detect backend failures"]
    fn spread_activation(&self, source: &Uuid, decay: f32) -> Result<(), MemoryError> {
        // Default implementation - can be overridden for efficiency
        let neighbors = self.get_neighbors(source)?;
        for (neighbor_id, weight) in neighbors {
            if let Some(memory) = self.retrieve(&neighbor_id)? {
                let new_activation = weight.mul_add(decay, memory.activation_value);
                self.update_activation(&neighbor_id, new_activation.min(1.0))?;
            }
        }
        Ok(())
    }

    /// Get the edge weight between two nodes
    ///
    /// # Errors
    ///
    /// Returns [`MemoryError`] if the backend cannot access the stored edge.
    #[must_use = "Handle the result to detect backend failures"]
    fn get_edge_weight(&self, from: &Uuid, to: &Uuid) -> Result<Option<f32>, MemoryError>;

    /// Get all edges in the graph
    ///
    /// # Errors
    ///
    /// Returns [`MemoryError`] if the backend cannot iterate the stored edges.
    #[must_use = "Handle the result to detect backend failures"]
    fn all_edges(&self) -> Result<Vec<(Uuid, Uuid, f32)>, MemoryError>;

    /// Get the count of outgoing edges from a node
    ///
    /// Used for computing fan effect (number of associations).
    ///
    /// # Errors
    ///
    /// Returns [`MemoryError`] if the backend cannot access the node's edges.
    #[must_use = "Handle the result to detect backend failures"]
    fn get_outgoing_edge_count(&self, id: &Uuid) -> Result<usize, MemoryError> {
        // Default implementation using get_neighbors
        self.get_neighbors(id).map(|neighbors| neighbors.len())
    }

    /// Add a concept binding between episode and concept
    ///
    /// Creates a bidirectional binding with atomic strength tracking.
    ///
    /// # Arguments
    ///
    /// * `episode_id` - Source episode node
    /// * `concept_id` - Target concept node
    /// * `strength` - Initial binding strength (0.0-1.0)
    /// * `contribution` - Episode's contribution to concept formation (0.0-1.0)
    ///
    /// # Errors
    ///
    /// Returns [`MemoryError::NotFound`] if either node doesn't exist.
    /// Returns [`MemoryError::StorageError`] if binding creation fails.
    ///
    /// # Default Implementation
    ///
    /// Returns [`MemoryError::UnsupportedTypeOperation`] - backends must override.
    #[must_use = "Handle the result to detect backend failures"]
    fn add_concept_binding(
        &self,
        _episode_id: Uuid,
        _concept_id: Uuid,
        _strength: f32,
        _contribution: f32,
    ) -> Result<(), MemoryError> {
        Err(MemoryError::UnsupportedTypeOperation(
            "Concept bindings not supported by this backend".to_string(),
        ))
    }

    /// Get concepts for an episode (bottom-up access)
    ///
    /// Returns all concept bindings for the given episode.
    ///
    /// # Arguments
    ///
    /// * `episode_id` - Episode node UUID
    ///
    /// # Returns
    ///
    /// Vector of (concept_id, strength, contribution) tuples
    ///
    /// # Default Implementation
    ///
    /// Returns empty vector - backends should override for efficiency.
    #[must_use]
    fn get_episode_concepts(&self, _episode_id: &Uuid) -> Vec<(Uuid, f32, f32)> {
        Vec::new()
    }

    /// Get episodes for a concept (top-down access)
    ///
    /// Returns all episode bindings for the given concept.
    ///
    /// # Arguments
    ///
    /// * `concept_id` - Concept node UUID
    ///
    /// # Returns
    ///
    /// Vector of (episode_id, strength, contribution) tuples
    ///
    /// # Default Implementation
    ///
    /// Returns empty vector - backends should override for efficiency.
    #[must_use]
    fn get_concept_episodes(&self, _concept_id: &Uuid) -> Vec<(Uuid, f32, f32)> {
        Vec::new()
    }

    /// Spread activation through bindings
    ///
    /// Propagates activation from source node through its bindings.
    ///
    /// # Arguments
    ///
    /// * `source_id` - Source node UUID
    /// * `is_episode` - True if source is episode, false if concept
    /// * `decay` - Activation decay factor (0.0-1.0)
    ///
    /// # Errors
    ///
    /// Returns [`MemoryError::NotFound`] if source node doesn't exist.
    /// Returns [`MemoryError::StorageError`] if activation update fails.
    ///
    /// # Default Implementation
    ///
    /// Returns [`MemoryError::UnsupportedTypeOperation`] - backends must override.
    #[must_use = "Handle the result to detect backend failures"]
    fn spread_through_bindings(
        &self,
        _source_id: &Uuid,
        _is_episode: bool,
        _decay: f32,
    ) -> Result<(), MemoryError> {
        Err(MemoryError::UnsupportedTypeOperation(
            "Binding-based spreading not supported by this backend".to_string(),
        ))
    }
}

/// Extended trait for dual memory type operations.
///
/// This trait extends `MemoryBackend` to support type-aware storage and retrieval
/// of episodic vs semantic (concept) memory nodes. Backends implementing this trait
/// can optimize storage layouts, NUMA placement, and cache strategies based on the
/// distinct access patterns of episodes (temporal locality, frequent updates) vs
/// concepts (stable, semantic search).
///
/// # Design Philosophy
///
/// Following *A Philosophy of Software Design* principles:
///
/// - **Deep module**: The trait hides internal storage complexity (separate DashMaps,
///   NUMA allocations, cache tiers) behind a simple, general-purpose interface.
/// - **Information hiding**: Implementation details like shard counts, memory budgets,
///   and type indices are encapsulated within the backend.
/// - **Strategic design**: Methods are designed to work for any dual-memory backend
///   implementation (DashMap-based, B-tree, hybrid, etc.) without exposing storage details.
///
/// # When to Use
///
/// Use `DualMemoryBackend` when:
/// - You need to distinguish between episodic and semantic memory at the storage layer
/// - You want to optimize storage strategies based on memory type (e.g., episodes use
///   LRU eviction, concepts use consolidation)
/// - You need type-specific iteration without deserializing all nodes
/// - You want separate memory budgets or NUMA placement for different memory types
///
/// Use plain `MemoryBackend` when:
/// - You don't need to distinguish between memory types
/// - You have a homogeneous memory model
/// - You want maximum backward compatibility with existing code
///
/// # Performance Characteristics
///
/// Implementations should target:
/// - **Type-specific insertion**: <100μs P99 latency for episodes, <200μs for concepts
/// - **Type-filtered iteration**: >1M nodes/sec with zero allocations
/// - **Memory overhead**: <15% vs single-map storage (for type indexing)
/// - **NUMA placement**: >60% reduction in remote memory access on multi-socket systems
///
/// # Example
///
/// ```rust,ignore
/// use engram_core::memory_graph::{DualMemoryBackend, MemoryBackend};
/// use engram_core::memory::DualMemoryNode;
///
/// fn process_episodes<B: DualMemoryBackend>(backend: &B) {
///     // Iterate only episode nodes - zero allocation
///     for episode in backend.iter_episodes() {
///         // Process episodic memories with temporal context
///         if episode.is_episode() {
///             println!("Episode: {}", episode.id);
///         }
///     }
///
///     // Get separate counts for capacity planning
///     let (ep_count, concept_count) = backend.count_by_type();
///     println!("Episodes: {}, Concepts: {}", ep_count, concept_count);
/// }
/// ```
///
/// # Thread Safety
///
/// All methods are required to be thread-safe (through `Send + Sync` bound from
/// `MemoryBackend`). Concurrent type-specific operations must not cause data races.
#[cfg(feature = "dual_memory_types")]
pub trait DualMemoryBackend: MemoryBackend {
    /// Add a node with explicit type annotation.
    ///
    /// This method stores a `DualMemoryNode` with its memory type metadata,
    /// allowing the backend to route it to the appropriate storage tier
    /// (episode map vs concept map) and apply type-specific policies.
    ///
    /// # Implementation Notes
    ///
    /// Backends should:
    /// - Route episodes to high-churn storage (e.g., episode DashMap with 64 shards)
    /// - Route concepts to stable storage (e.g., concept DashMap with 16 shards)
    /// - Update type index for fast type lookups
    /// - Apply NUMA placement based on memory type
    /// - Enforce type-specific memory budgets
    ///
    /// # Errors
    ///
    /// Returns [`MemoryError::CapacityExceeded`] if type-specific budget is exhausted.
    /// Returns [`MemoryError::StorageError`] if the backend fails to persist the node.
    #[must_use = "Handle the result to detect backend failures"]
    fn add_node_typed(&self, node: DualMemoryNode) -> Result<Uuid, MemoryError>;

    /// Retrieve a node with type information preserved.
    ///
    /// Unlike `retrieve()` which returns `Memory`, this method returns the full
    /// `DualMemoryNode` with type metadata intact, allowing callers to query
    /// consolidation scores, instance counts, and other type-specific fields.
    ///
    /// # Performance
    ///
    /// Implementations should check the type index first to determine which
    /// storage tier to query, avoiding unnecessary lookups in both maps.
    ///
    /// # Errors
    ///
    /// Returns [`MemoryError::NotFound`] if the node doesn't exist.
    /// Returns [`MemoryError::StorageError`] if retrieval fails.
    #[must_use = "Handle the result to detect backend failures"]
    fn get_node_typed(&self, id: &Uuid) -> Result<Option<DualMemoryNode>, MemoryError>;

    /// Iterate only episode nodes with zero allocations.
    ///
    /// Returns an iterator over episode nodes, avoiding deserialization of
    /// concept nodes entirely. This is more efficient than filtering `all_ids()`
    /// when you only need one memory type.
    ///
    /// # Performance Contract
    ///
    /// - **Zero allocation**: Iterator should yield references or clones without
    ///   intermediate allocations (beyond DashMap internal iteration overhead).
    /// - **Cache efficiency**: Should iterate the episode storage tier directly
    ///   for better cache locality vs jumping between types.
    /// - **Throughput**: Target >1M nodes/sec on modern hardware.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// for episode in backend.iter_episodes() {
    ///     if let Some(score) = episode.node_type.consolidation_score() {
    ///         if score > 0.8 {
    ///             // Trigger concept formation
    ///         }
    ///     }
    /// }
    /// ```
    #[must_use]
    fn iter_episodes(&self) -> Box<dyn Iterator<Item = DualMemoryNode> + '_>;

    /// Iterate only concept nodes with zero allocations.
    ///
    /// Returns an iterator over concept nodes, avoiding deserialization of
    /// episode nodes entirely. Useful for consolidation and semantic search.
    ///
    /// # Performance Contract
    ///
    /// - **Zero allocation**: Iterator should yield references or clones without
    ///   intermediate allocations (beyond DashMap internal iteration overhead).
    /// - **Cache efficiency**: Should iterate the concept storage tier directly.
    /// - **Throughput**: Target >500K nodes/sec (lower than episodes due to
    ///   larger centroid embeddings).
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// for concept in backend.iter_concepts() {
    ///     if let Some(count) = concept.node_type.instance_count() {
    ///         println!("Concept {} has {} instances", concept.id, count);
    ///     }
    /// }
    /// ```
    #[must_use]
    fn iter_concepts(&self) -> Box<dyn Iterator<Item = DualMemoryNode> + '_>;

    /// Get counts by type: (episodes, concepts).
    ///
    /// Returns separate counts for capacity planning and monitoring. This is
    /// more efficient than filtering all nodes when you only need statistics.
    ///
    /// # Performance
    ///
    /// Implementations should cache these counts or compute them from separate
    /// DashMap `.len()` calls rather than iterating all nodes.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let (episodes, concepts) = backend.count_by_type();
    /// if episodes > 1_000_000 {
    ///     // Trigger consolidation to free up episode storage
    /// }
    /// ```
    #[must_use]
    fn count_by_type(&self) -> (usize, usize);

    /// Get memory usage by type in bytes: (episode_bytes, concept_bytes).
    ///
    /// Returns approximate memory consumption for each type, useful for
    /// monitoring and enforcing type-specific memory budgets.
    ///
    /// # Accuracy
    ///
    /// Implementations may return estimates based on:
    /// - `count * sizeof(DualMemoryNode)` for ballpark numbers
    /// - Atomic counters updated on insertion/removal for precise tracking
    /// - System allocator queries (less portable)
    ///
    /// Exact byte counts are not required - approximations within 10% are acceptable.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let (ep_bytes, concept_bytes) = backend.memory_usage_by_type();
    /// let ep_mb = ep_bytes / (1024 * 1024);
    /// let concept_mb = concept_bytes / (1024 * 1024);
    /// println!("Episodes: {}MB, Concepts: {}MB", ep_mb, concept_mb);
    /// ```
    #[must_use]
    fn memory_usage_by_type(&self) -> (usize, usize);
}
