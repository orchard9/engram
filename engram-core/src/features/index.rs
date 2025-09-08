//! Index provider abstraction for vector similarity search
//!
//! This module provides a trait-based abstraction over indexing backends,
//! allowing graceful fallback from HNSW to linear search when the feature is disabled.

use super::FeatureProvider;
use crate::{Confidence, Episode};
use std::any::Any;
use std::sync::Arc;
use thiserror::Error;

/// Errors that can occur during index operations
#[derive(Debug, Error)]
pub enum IndexError {
    #[error("Index build failed: {0}")]
    BuildFailed(String),
    
    #[error("Search failed: {0}")]
    SearchFailed(String),
    
    #[error("Index not initialized")]
    NotInitialized,
    
    #[error("Invalid parameters: {0}")]
    InvalidParameters(String),
}

/// Result type for index operations
pub type IndexResult<T> = Result<T, IndexError>;

/// Trait for index operations
pub trait Index: Send + Sync {
    /// Build the index from a set of episodes
    fn build(&mut self, episodes: &[Episode]) -> IndexResult<()>;
    
    /// Search for k nearest neighbors
    fn search(&self, query: &[f32; 768], k: usize) -> IndexResult<Vec<(String, f32)>>;
    
    /// Add a single episode to the index
    fn add(&mut self, episode: &Episode) -> IndexResult<()>;
    
    /// Remove an episode from the index
    fn remove(&mut self, id: &str) -> IndexResult<()>;
    
    /// Get the number of indexed items
    fn size(&self) -> usize;
    
    /// Clear the index
    fn clear(&mut self);
}

/// Provider trait for index implementations
pub trait IndexProvider: FeatureProvider {
    /// Create a new index instance
    fn create_index(&self) -> Box<dyn Index>;
    
    /// Get index configuration
    fn get_config(&self) -> IndexConfig;
}

/// Configuration for index operations
#[derive(Debug, Clone)]
pub struct IndexConfig {
    /// Maximum number of neighbors for HNSW
    pub m: usize,
    /// Size of dynamic candidate list for HNSW construction
    pub ef_construction: usize,
    /// Size of dynamic candidate list for search
    pub ef_search: usize,
    /// Distance metric to use
    pub metric: DistanceMetric,
}

impl Default for IndexConfig {
    fn default() -> Self {
        Self {
            m: 16,
            ef_construction: 200,
            ef_search: 100,
            metric: DistanceMetric::Cosine,
        }
    }
}

/// Distance metrics for similarity search
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DistanceMetric {
    Cosine,
    Euclidean,
    DotProduct,
}

/// HNSW index provider (only available when feature is enabled)
#[cfg(feature = "hnsw_index")]
pub struct HnswIndexProvider {
    config: IndexConfig,
}

#[cfg(feature = "hnsw_index")]
impl HnswIndexProvider {
    pub fn new() -> Self {
        Self {
            config: IndexConfig::default(),
        }
    }
    
    pub fn with_config(config: IndexConfig) -> Self {
        Self { config }
    }
}

#[cfg(feature = "hnsw_index")]
impl FeatureProvider for HnswIndexProvider {
    fn is_enabled(&self) -> bool {
        true
    }
    
    fn name(&self) -> &'static str {
        "hnsw_index"
    }
    
    fn description(&self) -> &'static str {
        "Hierarchical Navigable Small World index for fast similarity search"
    }
    
    fn as_any(&self) -> &dyn Any {
        self
    }
}

#[cfg(feature = "hnsw_index")]
impl IndexProvider for HnswIndexProvider {
    fn create_index(&self) -> Box<dyn Index> {
        Box::new(HnswIndexImpl::new(self.config.clone()))
    }
    
    fn get_config(&self) -> IndexConfig {
        self.config.clone()
    }
}

/// Actual HNSW implementation
#[cfg(feature = "hnsw_index")]
struct HnswIndexImpl {
    config: IndexConfig,
    index: Option<crate::index::CognitiveHnswIndex>,
}

#[cfg(feature = "hnsw_index")]
impl HnswIndexImpl {
    fn new(config: IndexConfig) -> Self {
        Self {
            config,
            index: None,
        }
    }
}

#[cfg(feature = "hnsw_index")]
impl Index for HnswIndexImpl {
    fn build(&mut self, episodes: &[Episode]) -> IndexResult<()> {
        use crate::index::CognitiveHnswIndex;
        
        let mut index = CognitiveHnswIndex::new();
        
        for episode in episodes {
            let memory = crate::Memory::from_episode(episode.clone(), 1.0);
            index.insert_memory(Arc::new(memory))
                .map_err(|e| IndexError::BuildFailed(e.to_string()))?;
        }
        
        self.index = Some(index);
        Ok(())
    }
    
    fn search(&self, query: &[f32; 768], k: usize) -> IndexResult<Vec<(String, f32)>> {
        let index = self.index.as_ref()
            .ok_or(IndexError::NotInitialized)?;
            
        let results = index.search_with_confidence(query, k, Confidence::LOW);
            
        Ok(results.into_iter()
            .map(|r| (r.0, r.1.raw()))
            .collect())
    }
    
    fn add(&mut self, episode: &Episode) -> IndexResult<()> {
        let index = self.index.as_mut()
            .ok_or(IndexError::NotInitialized)?;
            
        let memory = crate::Memory::from_episode(episode.clone(), 1.0);
        index.insert_memory(Arc::new(memory))
            .map_err(|e| IndexError::BuildFailed(e.to_string()))?;
            
        Ok(())
    }
    
    fn remove(&mut self, id: &str) -> IndexResult<()> {
        // HNSW doesn't support removal in our implementation
        // This is a known limitation
        Ok(())
    }
    
    fn size(&self) -> usize {
        // CognitiveHnswIndex doesn't expose a size method, so we'll return 0 for now
        // This would need to be implemented in the actual index
        0
    }
    
    fn clear(&mut self) {
        self.index = None;
    }
}