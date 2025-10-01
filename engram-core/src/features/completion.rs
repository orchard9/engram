//! Pattern completion provider abstraction
//!
//! This module provides a trait-based abstraction over pattern completion,
//! allowing graceful fallback from neural completion to simple retrieval.

#![allow(clippy::unused_self)]

use super::FeatureProvider;
use crate::{Confidence, Memory};
use std::any::Any;
use std::cmp::Ordering;
use std::sync::Arc;
use thiserror::Error;

/// Errors that can occur during completion operations
#[derive(Debug, Error)]
pub enum CompletionError {
    /// The backend failed while attempting to complete the pattern.
    #[error("Completion failed: {0}")]
    CompletionFailed(String),

    /// The provided input pattern or parameters were invalid.
    #[error("Invalid pattern: {0}")]
    InvalidPattern(String),

    /// No matches satisfied the requested threshold.
    #[error("No matches found")]
    NoMatches,
}

/// Result type for completion operations
pub type CompletionResult<T> = Result<T, CompletionError>;

/// Trait for pattern completion operations
pub trait Completion: Send + Sync {
    /// Complete a partial pattern
    ///
    /// # Errors
    /// Returns [`CompletionError::CompletionFailed`] when the backend cannot complete the
    /// pattern or [`CompletionError::InvalidPattern`] when the input is malformed.
    fn complete(
        &self,
        partial: &Memory,
        candidates: &[Arc<Memory>],
        threshold: f32,
    ) -> CompletionResult<Vec<CompletionMatch>>;

    /// Predict next element in a sequence
    ///
    /// # Errors
    /// Returns [`CompletionError::CompletionFailed`] when sequence prediction is disabled
    /// or [`CompletionError::InvalidPattern`] when the input sequence is empty.
    fn predict_next(
        &self,
        sequence: &[Memory],
        candidates: &[Arc<Memory>],
    ) -> CompletionResult<Vec<PredictionResult>>;

    /// Fill in missing parts of a pattern
    ///
    /// # Errors
    /// Returns [`CompletionError::CompletionFailed`] when the backend cannot reconstruct
    /// the pattern.
    fn fill_gaps(&self, pattern: &Memory, mask: &[bool]) -> CompletionResult<Memory>;
}

/// A pattern completion match
#[derive(Debug, Clone)]
pub struct CompletionMatch {
    /// Memory that matched the requested pattern.
    pub memory: Arc<Memory>,
    /// Confidence assigned to the match by the backend.
    pub confidence: Confidence,
    /// Raw similarity score between the candidate and the query pattern.
    pub similarity: f32,
}

/// A sequence prediction result
#[derive(Debug, Clone)]
pub struct PredictionResult {
    /// Memory predicted as the next element in a sequence.
    pub memory: Arc<Memory>,
    /// Probability assigned to the prediction.
    pub probability: f32,
    /// Confidence interval associated with the prediction.
    pub confidence: Confidence,
}

/// Provider trait for completion implementations
pub trait CompletionProvider: FeatureProvider {
    /// Create a new completion instance
    #[must_use]
    fn create_completion(&self) -> Box<dyn Completion>;

    /// Get completion configuration
    #[must_use]
    fn get_config(&self) -> CompletionConfig;
}

/// Configuration for completion operations
#[derive(Debug, Clone)]
pub struct CompletionConfig {
    /// Minimum similarity threshold
    pub min_similarity: f32,
    /// Maximum candidates to consider
    pub max_candidates: usize,
    /// Use neural networks for completion
    pub use_neural: bool,
    /// Enable sequence prediction
    pub sequence_prediction: bool,
}

impl Default for CompletionConfig {
    fn default() -> Self {
        Self {
            min_similarity: 0.7,
            max_candidates: 100,
            use_neural: true,
            sequence_prediction: true,
        }
    }
}

/// Pattern completion provider (only available when feature is enabled)
#[cfg(feature = "pattern_completion")]
pub struct PatternCompletionProvider {
    config: CompletionConfig,
}

#[cfg(feature = "pattern_completion")]
impl PatternCompletionProvider {
    /// Create a provider using the default completion configuration.
    #[must_use]
    pub fn new() -> Self {
        Self {
            config: CompletionConfig::default(),
        }
    }

    /// Create a provider using a caller-supplied configuration.
    #[must_use]
    pub const fn with_config(config: CompletionConfig) -> Self {
        Self { config }
    }
}

#[cfg(feature = "pattern_completion")]
impl Default for PatternCompletionProvider {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(feature = "pattern_completion")]
impl FeatureProvider for PatternCompletionProvider {
    fn is_enabled(&self) -> bool {
        true
    }

    fn name(&self) -> &'static str {
        "pattern_completion"
    }

    fn description(&self) -> &'static str {
        "Neural pattern completion and sequence prediction"
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

#[cfg(feature = "pattern_completion")]
impl CompletionProvider for PatternCompletionProvider {
    fn create_completion(&self) -> Box<dyn Completion> {
        Box::new(PatternCompletionImpl::new(self.config.clone()))
    }

    fn get_config(&self) -> CompletionConfig {
        self.config.clone()
    }
}

/// Actual pattern completion implementation
#[cfg(feature = "pattern_completion")]
struct PatternCompletionImpl {
    config: CompletionConfig,
}

#[cfg(feature = "pattern_completion")]
impl PatternCompletionImpl {
    const fn new(config: CompletionConfig) -> Self {
        Self { config }
    }
}

#[cfg(feature = "pattern_completion")]
impl Completion for PatternCompletionImpl {
    fn complete(
        &self,
        partial: &Memory,
        candidates: &[Arc<Memory>],
        threshold: f32,
    ) -> CompletionResult<Vec<CompletionMatch>> {
        use crate::compute::VectorOps;
        use crate::compute::dispatch::DispatchVectorOps;

        let processor = DispatchVectorOps::new();
        let mut matches = Vec::new();

        for candidate in candidates.iter().take(self.config.max_candidates) {
            let similarity =
                processor.cosine_similarity_768(&partial.embedding, &candidate.embedding);

            if similarity >= threshold.max(self.config.min_similarity) {
                matches.push(CompletionMatch {
                    memory: candidate.clone(),
                    confidence: Confidence::from_raw(similarity),
                    similarity,
                });
            }
        }

        matches.sort_by(|a, b| {
            b.similarity
                .partial_cmp(&a.similarity)
                .unwrap_or(Ordering::Equal)
        });
        Ok(matches)
    }

    fn predict_next(
        &self,
        sequence: &[Memory],
        candidates: &[Arc<Memory>],
    ) -> CompletionResult<Vec<PredictionResult>> {
        if !self.config.sequence_prediction {
            return Err(CompletionError::CompletionFailed(
                "Sequence prediction disabled".to_string(),
            ));
        }

        // Simple implementation: find patterns based on last element
        if let Some(last) = sequence.last() {
            let matches = self.complete(last, candidates, self.config.min_similarity)?;

            Ok(matches
                .into_iter()
                .map(|m| PredictionResult {
                    memory: m.memory,
                    probability: m.similarity,
                    confidence: m.confidence,
                })
                .collect())
        } else {
            Err(CompletionError::InvalidPattern(
                "Empty sequence".to_string(),
            ))
        }
    }

    fn fill_gaps(&self, pattern: &Memory, _mask: &[bool]) -> CompletionResult<Memory> {
        // Simple implementation: return the pattern as-is
        // A real implementation would use neural networks to fill gaps
        Ok(pattern.clone())
    }
}
