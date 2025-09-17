//! Centralized cue dispatcher implementation

use crate::{Cue, CueType, Episode, Confidence};
use crate::error::CueError;
use std::collections::HashMap;

/// Centralized cue handling with strategy pattern
pub struct CueDispatcher {
    handlers: HashMap<String, Box<dyn CueHandler>>,
}

/// Strategy interface for different cue types
pub trait CueHandler: Send + Sync {
    fn handle(
        &self,
        cue: &Cue,
        context: &dyn CueContext,
    ) -> Result<Vec<(Episode, Confidence)>, CueError>;
    
    fn supports_cue_type(&self, cue_type: &CueType) -> bool;
    
    fn handler_name(&self) -> &'static str;
}

/// Context interface for cue handling operations
pub trait CueContext: Send + Sync {
    fn get_episodes(&self) -> &std::collections::HashMap<String, Episode>;
    fn compute_similarity(&self, a: &[f32; 768], b: &[f32; 768]) -> f32;
    fn apply_decay(&self, episode: &Episode, confidence: Confidence) -> Confidence;
    fn search_similar_episodes(&self, vector: &[f32; 768], k: usize, threshold: Confidence) -> Vec<(String, Confidence)>;
}

impl CueDispatcher {
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
    
    pub fn register_handler(&mut self, handler: Box<dyn CueHandler>) {
        self.handlers.insert(handler.handler_name().to_string(), handler);
    }
    
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