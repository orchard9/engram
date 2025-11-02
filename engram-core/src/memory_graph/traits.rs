//! Core traits for memory backend implementations

use crate::Cue;
use crate::memory::Memory;
use std::sync::Arc;
use thiserror::Error;
use uuid::Uuid;

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
}
