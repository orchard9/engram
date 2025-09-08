//! Core traits for memory backend implementations

use crate::{Confidence, Cue};
use crate::memory::Memory;
use std::sync::Arc;
use thiserror::Error;
use uuid::Uuid;

/// Errors that can occur in memory operations
#[derive(Error, Debug)]
pub enum MemoryError {
    #[error("Memory not found: {0}")]
    NotFound(Uuid),
    
    #[error("Storage error: {0}")]
    StorageError(String),
    
    #[error("Index error: {0}")]
    IndexError(String),
    
    #[error("Lock poisoned: {0}")]
    LockPoisoned(String),
    
    #[error("Capacity exceeded: current {current}, max {max}")]
    CapacityExceeded { current: usize, max: usize },
    
    #[error("Invalid embedding dimension: expected 768, got {0}")]
    InvalidEmbeddingDimension(usize),
}

/// Core trait for memory storage backends
pub trait MemoryBackend: Send + Sync {
    /// Store a memory with the given ID
    fn store(&self, id: Uuid, memory: Memory) -> Result<(), MemoryError>;
    
    /// Retrieve a memory by ID
    fn retrieve(&self, id: &Uuid) -> Result<Option<Arc<Memory>>, MemoryError>;
    
    /// Remove a memory by ID
    fn remove(&self, id: &Uuid) -> Result<Option<Arc<Memory>>, MemoryError>;
    
    /// Search for memories by embedding similarity
    fn search(&self, embedding: &[f32], k: usize) -> Result<Vec<(Uuid, f32)>, MemoryError>;
    
    /// Update the activation level of a memory
    fn update_activation(&self, id: &Uuid, activation: f32) -> Result<(), MemoryError>;
    
    /// Get the current memory count
    fn count(&self) -> usize;
    
    /// Clear all memories
    fn clear(&self) -> Result<(), MemoryError>;
    
    /// Check if a memory exists
    fn contains(&self, id: &Uuid) -> bool {
        self.retrieve(id).map(|m| m.is_some()).unwrap_or(false)
    }
    
    /// Get all memory IDs
    fn all_ids(&self) -> Vec<Uuid>;
    
    /// Recall memories based on a cue
    fn recall(&self, cue: &Cue) -> Result<Vec<Arc<Memory>>, MemoryError> {
        match &cue.cue_type {
            crate::CueType::Embedding { vec, .. } => {
                let results = self.search(vec, cue.max_results)?;
                let mut memories = Vec::new();
                for (id, score) in results {
                    if score >= cue.result_threshold.raw() {
                        if let Some(memory) = self.retrieve(&id)? {
                            memories.push(memory);
                        }
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
    fn add_edge(&self, from: Uuid, to: Uuid, weight: f32) -> Result<(), MemoryError>;
    
    /// Remove an edge between two memory nodes
    fn remove_edge(&self, from: &Uuid, to: &Uuid) -> Result<bool, MemoryError>;
    
    /// Get all neighbors of a memory node
    fn get_neighbors(&self, id: &Uuid) -> Result<Vec<(Uuid, f32)>, MemoryError>;
    
    /// Perform breadth-first traversal from a starting node
    fn traverse_bfs(&self, start: &Uuid, max_depth: usize) -> Result<Vec<Uuid>, MemoryError>;
    
    /// Spread activation from a source node
    fn spread_activation(&self, source: &Uuid, decay: f32) -> Result<(), MemoryError> {
        // Default implementation - can be overridden for efficiency
        let neighbors = self.get_neighbors(source)?;
        for (neighbor_id, weight) in neighbors {
            if let Some(memory) = self.retrieve(&neighbor_id)? {
                let new_activation = memory.activation_value + (weight * decay);
                self.update_activation(&neighbor_id, new_activation.min(1.0))?;
            }
        }
        Ok(())
    }
    
    /// Get the edge weight between two nodes
    fn get_edge_weight(&self, from: &Uuid, to: &Uuid) -> Result<Option<f32>, MemoryError>;
    
    /// Get all edges in the graph
    fn all_edges(&self) -> Result<Vec<(Uuid, Uuid, f32)>, MemoryError>;
}