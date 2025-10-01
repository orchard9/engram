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
    /// The backend failed while constructing the index.
    #[error("Index build failed: {0}")]
    BuildFailed(String),

    /// The backend failed while executing a similarity search.
    #[error("Search failed: {0}")]
    SearchFailed(String),

    /// The operation requires an index that has already been built.
    #[error("Index not initialized")]
    NotInitialized,

    /// The supplied parameters are invalid for the active backend.
    #[error("Invalid parameters: {0}")]
    InvalidParameters(String),
}

/// Result type for index operations
pub type IndexResult<T> = Result<T, IndexError>;

/// Trait for index operations
pub trait Index: Send + Sync {
    /// Build the index from a set of episodes
    ///
    /// # Errors
    /// Returns [`IndexError::BuildFailed`] if the backend fails during construction.
    fn build(&mut self, episodes: &[Episode]) -> IndexResult<()>;

    /// Search for k nearest neighbors
    ///
    /// # Errors
    /// Returns [`IndexError::NotInitialized`] when the backend has not been built or
    /// [`IndexError::SearchFailed`] if the backend cannot complete the query.
    fn search(&self, query: &[f32; 768], k: usize) -> IndexResult<Vec<(String, f32)>>;

    /// Add a single episode to the index
    ///
    /// # Errors
    /// Returns [`IndexError::NotInitialized`] when the backend has not been built or
    /// [`IndexError::BuildFailed`] if the backend rejects the episode.
    fn add(&mut self, episode: &Episode) -> IndexResult<()>;

    /// Remove an episode from the index
    ///
    /// # Errors
    /// Returns an error when the backend cannot remove the episode (for example when it
    /// has not been initialized or removal is unsupported).
    fn remove(&mut self, id: &str) -> IndexResult<()>;

    /// Get the number of indexed items
    #[must_use]
    fn size(&self) -> usize;

    /// Clear the index
    fn clear(&mut self);
}

/// Provider trait for index implementations
pub trait IndexProvider: FeatureProvider {
    /// Create a new index instance
    #[must_use]
    fn create_index(&self) -> Box<dyn Index>;

    /// Get index configuration
    #[must_use]
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
    /// Use cosine similarity for comparing embeddings.
    Cosine,
    /// Use Euclidean distance for comparing embeddings.
    Euclidean,
    /// Use raw dot-product similarity.
    DotProduct,
}

/// HNSW index provider (only available when feature is enabled)
#[cfg(feature = "hnsw_index")]
pub struct HnswIndexProvider {
    config: IndexConfig,
}

#[cfg(feature = "hnsw_index")]
impl HnswIndexProvider {
    /// Create a provider using default HNSW configuration.
    #[must_use]
    pub fn new() -> Self {
        Self {
            config: IndexConfig::default(),
        }
    }

    /// Create a provider using a caller-specified configuration.
    #[must_use]
    pub const fn with_config(config: IndexConfig) -> Self {
        Self { config }
    }
}

#[cfg(feature = "hnsw_index")]
impl Default for HnswIndexProvider {
    fn default() -> Self {
        Self::new()
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
    const fn new(config: IndexConfig) -> Self {
        Self {
            config,
            index: None,
        }
    }

    /// Validate configuration values before running index operations.
    ///
    /// # Errors
    /// Returns [`IndexError::InvalidParameters`] when any configured parameter would
    /// produce undefined behaviour (for example zero neighbors or search beam width).
    fn validate_config(&self) -> IndexResult<()> {
        if self.config.m == 0 {
            return Err(IndexError::InvalidParameters(
                "`m` must be greater than zero".to_string(),
            ));
        }

        if self.config.ef_construction == 0 || self.config.ef_search == 0 {
            return Err(IndexError::InvalidParameters(
                "`ef_construction` and `ef_search` must be greater than zero".to_string(),
            ));
        }

        Ok(())
    }
}

#[cfg(feature = "hnsw_index")]
impl Index for HnswIndexImpl {
    fn build(&mut self, episodes: &[Episode]) -> IndexResult<()> {
        use crate::index::CognitiveHnswIndex;

        self.validate_config()?;

        let index = CognitiveHnswIndex::new();

        for episode in episodes {
            let memory = crate::Memory::from_episode(episode.clone(), 1.0);
            index
                .insert_memory(Arc::new(memory))
                .map_err(|e| IndexError::BuildFailed(e.to_string()))?;
        }

        self.index = Some(index);
        Ok(())
    }

    fn search(&self, query: &[f32; 768], k: usize) -> IndexResult<Vec<(String, f32)>> {
        let index = self.index.as_ref().ok_or(IndexError::NotInitialized)?;
        self.validate_config()?;
        let effective_k = k.min(self.config.ef_search);

        let results = index.search_with_confidence(query, effective_k, Confidence::LOW);

        Ok(results.into_iter().map(|r| (r.0, r.1.raw())).collect())
    }

    fn add(&mut self, episode: &Episode) -> IndexResult<()> {
        self.validate_config()?;
        let index = self.index.as_mut().ok_or(IndexError::NotInitialized)?;

        let memory = crate::Memory::from_episode(episode.clone(), 1.0);
        index
            .insert_memory(Arc::new(memory))
            .map_err(|e| IndexError::BuildFailed(e.to_string()))?;

        Ok(())
    }

    fn remove(&mut self, _id: &str) -> IndexResult<()> {
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
