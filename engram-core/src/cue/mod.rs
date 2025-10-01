//! Centralized cue handling with strategy pattern
//!
//! This module provides a unified approach to handling different types of cues
//! through a strategy pattern, eliminating duplicate pattern-matching logic
//! across the codebase.

pub mod dispatcher;
pub mod handlers;
#[cfg(test)]
mod tests;

pub use dispatcher::{CueContext, CueDispatcher, CueHandler};
pub use handlers::{
    ContextCueHandler, EmbeddingCueHandler, SemanticCueHandler, TemporalCueHandler,
};
