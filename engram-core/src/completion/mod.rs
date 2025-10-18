//! Pattern completion engine with biologically-inspired hippocampal dynamics.
//!
//! This module implements pattern completion for reconstructing missing parts of episodes
//! using CA3 autoassociative dynamics, DG pattern separation, and System 2 reasoning.

use crate::{Confidence, Episode};
use std::collections::HashMap;
use thiserror::Error;

#[cfg(feature = "pattern_completion")]
use nalgebra::DVector;

pub mod confidence;
pub mod consolidation;
pub mod context;
pub mod hippocampal;
pub mod hypothesis;
pub mod numeric;
pub mod reconstruction;
pub mod scheduler;

pub use confidence::MetacognitiveConfidence;
pub use consolidation::ConsolidationEngine;
pub use context::{EntorhinalContext, GridModule};
pub use hippocampal::HippocampalCompletion;
pub use hypothesis::{Hypothesis, System2Reasoner};
pub use reconstruction::PatternReconstructor;
pub use scheduler::{
    ConsolidationScheduler, ConsolidationStats as SchedulerStats, SchedulerConfig, SchedulerState,
};

/// Error types for pattern completion operations
#[derive(Debug, Error)]
pub enum CompletionError {
    /// Pattern lacks sufficient information for reliable completion
    #[error("Insufficient pattern information for completion")]
    InsufficientPattern,

    /// Iterative completion algorithm failed to converge within limit
    #[error("Pattern completion failed to converge after {0} iterations")]
    ConvergenceFailed(usize),

    /// Embedding vector has incorrect dimensions
    #[error("Invalid embedding dimension: expected 768, got {0}")]
    InvalidEmbeddingDimension(usize),

    /// Completion confidence is below acceptable threshold
    #[error("Confidence below threshold: {0}")]
    LowConfidence(f32),

    /// Linear algebra operation encountered an error
    #[error("Matrix operation failed: {0}")]
    MatrixError(String),
}

/// Result type for pattern completion operations
pub type CompletionResult<T> = Result<T, CompletionError>;

/// Represents a partial episode with missing information
#[derive(Debug, Clone)]
pub struct PartialEpisode {
    /// Available fields from the episode
    pub known_fields: HashMap<String, String>,

    /// Partial embedding (may have masked dimensions)
    pub partial_embedding: Vec<Option<f32>>,

    /// Cue strength for pattern completion
    pub cue_strength: Confidence,

    /// Context from surrounding episodes
    pub temporal_context: Vec<String>,
}

/// Represents a completed episode with biological plausibility
#[derive(Debug, Clone)]
pub struct CompletedEpisode {
    /// Reconstructed episode
    pub episode: Episode,

    /// Pattern completion confidence (CA1 output)
    pub completion_confidence: Confidence,

    /// Source monitoring: which parts are recalled vs reconstructed
    pub source_attribution: SourceMap,

    /// Alternative hypotheses from System 2 reasoning
    pub alternative_hypotheses: Vec<(Episode, Confidence)>,

    /// Metacognitive monitoring signal
    pub metacognitive_confidence: Confidence,

    /// Evidence from spreading activation
    pub activation_evidence: Vec<ActivationTrace>,
}

/// Maps episode fields to their source (recalled vs reconstructed)
#[derive(Debug, Clone, Default)]
pub struct SourceMap {
    /// Maps field names to their memory source
    pub field_sources: HashMap<String, MemorySource>,

    /// Confidence in source attribution
    pub source_confidence: HashMap<String, Confidence>,
}

/// Indicates the source of a memory field
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MemorySource {
    /// Original memory
    Recalled,
    /// Pattern-completed
    Reconstructed,
    /// Generated through System 2 reasoning
    Imagined,
    /// Retrieved from consolidated semantic memory
    Consolidated,
}

/// Activation trace for evidence accumulation
#[derive(Debug, Clone)]
pub struct ActivationTrace {
    /// Source memory ID
    pub source_memory: String,

    /// Activation strength
    pub activation_strength: f32,

    /// Pathway type
    pub pathway: ActivationPathway,

    /// Decay factor
    pub decay_factor: f32,
}

/// Type of activation pathway
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActivationPathway {
    /// Direct association
    Direct,
    /// Multi-hop activation
    Transitive,
    /// Semantic similarity
    Semantic,
    /// Temporal co-occurrence
    Temporal,
    /// Spatial proximity
    Spatial,
}

/// Core trait for pattern completion engines
pub trait PatternCompleter {
    /// Complete a partial episode using the engine's algorithm
    ///
    /// # Errors
    ///
    /// Returns an error when the implementation cannot complete the episode with the
    /// available evidence or encounters an internal failure.
    fn complete(&self, partial: &PartialEpisode) -> CompletionResult<CompletedEpisode>;

    /// Update the engine with new episodes for learning
    fn update(&mut self, episodes: &[Episode]);

    /// Get completion confidence for a partial pattern
    fn estimate_confidence(&self, partial: &PartialEpisode) -> Confidence;
}

/// Trait for biological dynamics simulation
pub trait BiologicalDynamics {
    /// Simulate one timestep of neural dynamics
    fn step(&mut self, input: &DVector<f32>) -> DVector<f32>;

    /// Check if dynamics have converged to attractor
    fn has_converged(&self) -> bool;

    /// Reset dynamics to initial state
    fn reset(&mut self);

    /// Get current energy of the system
    fn energy(&self) -> f32;
}

/// Configuration for pattern completion
#[derive(Debug, Clone)]
pub struct CompletionConfig {
    /// CA3 sparsity level (percentage of active neurons)
    pub ca3_sparsity: f32,

    /// DG expansion factor for pattern separation
    pub dg_expansion_factor: usize,

    /// CA1 confidence threshold for output gating
    pub ca1_threshold: Confidence,

    /// Maximum iterations for convergence
    pub max_iterations: usize,

    /// Convergence threshold
    pub convergence_threshold: f32,

    /// Working memory capacity for System 2 reasoning
    pub working_memory_capacity: usize,

    /// Number of alternative hypotheses to generate
    pub num_hypotheses: usize,

    /// Sharp-wave ripple frequency (Hz)
    pub ripple_frequency: f32,

    /// Ripple duration (ms)
    pub ripple_duration: f32,
}

impl Default for CompletionConfig {
    fn default() -> Self {
        Self {
            ca3_sparsity: 0.05, // 5% sparsity
            dg_expansion_factor: 10,
            ca1_threshold: Confidence::exact(0.7),
            max_iterations: 7, // Theta rhythm constraint
            convergence_threshold: 0.01,
            working_memory_capacity: 7, // Miller's magic number
            num_hypotheses: 3,
            ripple_frequency: 200.0, // 200 Hz
            ripple_duration: 75.0,   // 75 ms
        }
    }
}

/// Statistics for pattern completion performance
#[derive(Debug, Clone, Default)]
pub struct CompletionStats {
    /// Number of successful completions
    pub successful_completions: usize,

    /// Number of failed completions
    pub failed_completions: usize,

    /// Average iterations to convergence
    pub avg_iterations: f32,

    /// Average completion confidence
    pub avg_confidence: f32,

    /// Pattern separation index
    pub separation_index: f32,

    /// Source monitoring accuracy
    pub source_accuracy: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_completion_config_defaults() {
        let config = CompletionConfig::default();
        // Use tolerance to accommodate floating point representation differences
        assert!((config.ca3_sparsity - 0.05).abs() < 1e-6);
        assert_eq!(config.max_iterations, 7);
        assert_eq!(config.working_memory_capacity, 7);
    }

    #[test]
    fn test_memory_source_types() {
        let source = MemorySource::Recalled;
        assert_eq!(source, MemorySource::Recalled);

        let source = MemorySource::Reconstructed;
        assert_ne!(source, MemorySource::Recalled);
    }
}
