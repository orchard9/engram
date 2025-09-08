//! Unified memory graph implementation

use crate::memory_graph::{GraphBackend, GraphConfig, MemoryBackend, MemoryError};
use crate::{Confidence, Cue, CueType};
use crate::memory::{Episode, Memory};
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
    pub fn new(backend: B, config: GraphConfig) -> Self {
        Self {
            backend: Arc::new(backend),
            config,
        }
    }
    
    /// Create with default configuration
    pub fn with_backend(backend: B) -> Self {
        Self::new(backend, GraphConfig::default())
    }
    
    /// Get a reference to the backend
    pub fn backend(&self) -> &B {
        &self.backend
    }
    
    /// Get the current configuration
    pub fn config(&self) -> &GraphConfig {
        &self.config
    }
    
    /// Update the configuration
    pub fn set_config(&mut self, config: GraphConfig) {
        self.config = config;
    }
    
    /// Store a memory and return its ID
    pub fn store_memory(&self, memory: Memory) -> Result<Uuid, MemoryError> {
        let id = Uuid::new_v4();
        self.backend.store(id, memory)?;
        Ok(id)
    }
    
    /// Store an episode as a memory
    pub fn store_episode(&self, episode: Episode) -> Result<Uuid, MemoryError> {
        let memory = Memory::from_episode(episode, 1.0); // Default activation
        self.store_memory(memory)
    }
    
    /// Retrieve a memory by ID
    pub fn retrieve(&self, id: &Uuid) -> Result<Option<Arc<Memory>>, MemoryError> {
        self.backend.retrieve(id)
    }
    
    /// Remove a memory by ID
    pub fn remove(&self, id: &Uuid) -> Result<Option<Arc<Memory>>, MemoryError> {
        self.backend.remove(id)
    }
    
    /// Recall memories based on a cue
    pub fn recall(&self, cue: &Cue) -> Result<Vec<Arc<Memory>>, MemoryError> {
        let memories = match &cue.cue_type {
            CueType::Embedding(vec) => {
                // Search by embedding similarity
                let results = self.backend.search(vec, cue.max_results)?;
                
                // Apply spreading activation if enabled and backend supports it
                if self.config.enable_spreading {
                    if let Some(graph_backend) = (self.backend.as_ref() as &dyn std::any::Any)
                        .downcast_ref::<B>()
                        .and_then(|b| (b as &dyn std::any::Any).downcast_ref::<dyn GraphBackend>())
                    {
                        for (id, _score) in &results {
                            // Safe to ignore error as this is optional enhancement
                            let _ = graph_backend.spread_activation(id, self.config.decay_rate);
                        }
                    }
                }
                
                // Retrieve memories that meet threshold
                let mut memories = Vec::new();
                for (id, score) in results {
                    if score >= cue.result_threshold.raw() {
                        if let Some(memory) = self.backend.retrieve(&id)? {
                            memories.push(memory);
                        }
                    }
                }
                memories
            }
            CueType::Semantic { content, fuzzy_threshold } => {
                // For semantic cues, we need to search all memories
                // In a real implementation, this would use a text index
                let all_ids = self.backend.all_ids();
                let mut matches = Vec::new();
                
                for id in all_ids {
                    if let Some(memory) = self.backend.retrieve(&id)? {
                        // Simple content matching - could be enhanced with fuzzy matching
                        if memory.content.contains(content) {
                            matches.push(memory);
                        }
                    }
                    
                    if matches.len() >= cue.max_results {
                        break;
                    }
                }
                matches
            }
            CueType::Temporal { start, end } => {
                // Search memories within time range
                let all_ids = self.backend.all_ids();
                let mut matches = Vec::new();
                
                for id in all_ids {
                    if let Some(memory) = self.backend.retrieve(&id)? {
                        let in_range = match (start, end) {
                            (Some(s), Some(e)) => memory.timestamp >= *s && memory.timestamp <= *e,
                            (Some(s), None) => memory.timestamp >= *s,
                            (None, Some(e)) => memory.timestamp <= *e,
                            (None, None) => true,
                        };
                        
                        if in_range {
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
    pub fn similarity_search(&self, embedding: &[f32], k: usize, threshold: Confidence) 
        -> Result<Vec<(Arc<Memory>, f32)>, MemoryError> 
    {
        let results = self.backend.search(embedding, k)?;
        let mut memories = Vec::new();
        
        for (id, score) in results {
            if score >= threshold.raw() {
                if let Some(memory) = self.backend.retrieve(&id)? {
                    memories.push((memory, score));
                }
            }
        }
        
        Ok(memories)
    }
    
    /// Get the current memory count
    pub fn count(&self) -> usize {
        self.backend.count()
    }
    
    /// Clear all memories
    pub fn clear(&self) -> Result<(), MemoryError> {
        self.backend.clear()
    }
    
    /// Check if a memory exists
    pub fn contains(&self, id: &Uuid) -> bool {
        self.backend.contains(id)
    }
}

/// Extension methods for backends that support graph operations
impl<B: GraphBackend> UnifiedMemoryGraph<B> {
    /// Add an edge between two memories
    pub fn add_edge(&self, from: Uuid, to: Uuid, weight: f32) -> Result<(), MemoryError> {
        self.backend.add_edge(from, to, weight)
    }
    
    /// Remove an edge between two memories
    pub fn remove_edge(&self, from: &Uuid, to: &Uuid) -> Result<bool, MemoryError> {
        self.backend.remove_edge(from, to)
    }
    
    /// Get neighbors of a memory node
    pub fn get_neighbors(&self, id: &Uuid) -> Result<Vec<(Uuid, f32)>, MemoryError> {
        self.backend.get_neighbors(id)
    }
    
    /// Perform breadth-first traversal
    pub fn traverse_bfs(&self, start: &Uuid, max_depth: usize) -> Result<Vec<Uuid>, MemoryError> {
        self.backend.traverse_bfs(start, max_depth.min(self.config.max_depth))
    }
    
    /// Spread activation from a source node
    pub fn spread_activation(&self, source: &Uuid) -> Result<(), MemoryError> {
        self.backend.spread_activation(source, self.config.decay_rate)
    }
    
    /// Get all edges in the graph
    pub fn all_edges(&self) -> Result<Vec<(Uuid, Uuid, f32)>, MemoryError> {
        self.backend.all_edges()
    }
}

/// Convenience constructors for common backend configurations
impl UnifiedMemoryGraph<crate::memory::backends::HashMapBackend> {
    /// Create a simple single-threaded memory graph
    pub fn simple() -> Self {
        Self::with_backend(crate::memory::backends::HashMapBackend::new())
    }
}

impl UnifiedMemoryGraph<crate::memory::backends::DashMapBackend> {
    /// Create a concurrent memory graph optimized for parallel access
    pub fn concurrent() -> Self {
        Self::with_backend(crate::memory::backends::DashMapBackend::new())
    }
}

impl UnifiedMemoryGraph<crate::memory::backends::InfallibleBackend> {
    /// Create an infallible memory graph that degrades gracefully
    pub fn infallible(max_capacity: usize) -> Self {
        Self::with_backend(crate::memory::backends::InfallibleBackend::new(max_capacity))
    }
}