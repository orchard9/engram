//! Centralized cue dispatcher implementation

use crate::error::CueError;
use crate::{Confidence, Cue, CueType, Episode};
use std::collections::HashMap;

/// Centralized cue handling with strategy pattern
pub struct CueDispatcher {
    handlers: HashMap<String, Box<dyn CueHandler>>,
}

/// Strategy interface for different cue types
pub trait CueHandler: Send + Sync {
    /// Handle a cue using the strategy implementation.
    ///
    /// # Errors
    ///
    /// Returns an error when cue handling fails or the cue type is unsupported by the
    /// implementation.
    fn handle(
        &self,
        cue: &Cue,
        context: &dyn CueContext,
    ) -> Result<Vec<(Episode, Confidence)>, CueError>;

    /// Check if this handler supports the given cue type
    fn supports_cue_type(&self, cue_type: &CueType) -> bool;

    /// Return the name identifier for this handler
    fn handler_name(&self) -> &'static str;
}

/// Context interface for cue handling operations
pub trait CueContext: Send + Sync {
    /// Get all available episodes from the context
    fn get_episodes(&self) -> &std::collections::HashMap<String, Episode>;
    /// Compute similarity between two embedding vectors
    fn compute_similarity(&self, a: &[f32; 768], b: &[f32; 768]) -> f32;
    /// Apply decay to confidence based on episode age
    fn apply_decay(&self, episode: &Episode, confidence: Confidence) -> Confidence;
    /// Search for similar episodes using vector similarity
    fn search_similar_episodes(
        &self,
        vector: &[f32; 768],
        k: usize,
        threshold: Confidence,
    ) -> Vec<(String, Confidence)>;
}

impl CueDispatcher {
    /// Creates a new cue dispatcher with default handlers
    #[must_use]
    pub fn new() -> Self {
        let mut dispatcher = Self {
            handlers: HashMap::new(),
        };

        // Register default handlers
        dispatcher.register_handler(Box::new(super::handlers::EmbeddingCueHandler));
        dispatcher.register_handler(Box::new(super::handlers::ContextCueHandler));
        dispatcher.register_handler(Box::new(super::handlers::SemanticCueHandler));
        dispatcher.register_handler(Box::new(super::handlers::TemporalCueHandler));

        dispatcher
    }

    /// Register a new cue handler
    pub fn register_handler(&mut self, handler: Box<dyn CueHandler>) {
        self.handlers
            .insert(handler.handler_name().to_string(), handler);
    }

    /// Dispatch a cue to the appropriate handler implementation.
    ///
    /// # Errors
    ///
    /// Returns an error when no handler supports the cue type or the handler reports a
    /// domain-specific failure.
    pub fn handle_cue(
        &self,
        cue: &Cue,
        context: &dyn CueContext,
    ) -> Result<Vec<(Episode, Confidence)>, CueError> {
        // Find appropriate handler for this cue type
        for handler in self.handlers.values() {
            if handler.supports_cue_type(&cue.cue_type) {
                return handler.handle(cue, context);
            }
        }

        Err(CueError::UnsupportedCueType {
            cue_type: format!("{:?}", cue.cue_type),
            operation: "cue_dispatch".to_string(),
        })
    }
}

impl Default for CueDispatcher {
    fn default() -> Self {
        Self::new()
    }
}
