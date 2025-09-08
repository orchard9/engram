//! Unified memory graph abstractions
//!
//! This module provides a trait-based architecture for memory storage and graph operations,
//! allowing different backend implementations while maintaining a consistent API.

pub mod backends;
pub mod graph;
pub mod traits;

#[cfg(test)]
mod tests;

// Re-export core types
pub use graph::UnifiedMemoryGraph;
pub use traits::{GraphBackend, MemoryBackend, MemoryError};

// Re-export backend implementations
pub use backends::{DashMapBackend, HashMapBackend, InfallibleBackend};

// Configuration for memory graph operations
#[derive(Debug, Clone)]
pub struct GraphConfig {
    /// Maximum number of results to return from queries
    pub max_results: usize,
    /// Enable spreading activation on recall
    pub enable_spreading: bool,
    /// Decay rate for activation spreading
    pub decay_rate: f32,
    /// Maximum depth for graph traversal
    pub max_depth: usize,
    /// Minimum activation threshold
    pub activation_threshold: f32,
}

impl Default for GraphConfig {
    fn default() -> Self {
        Self {
            max_results: 100,
            enable_spreading: true,
            decay_rate: 0.8,
            max_depth: 3,
            activation_threshold: 0.01,
        }
    }
}