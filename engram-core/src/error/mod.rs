//! Error handling module with recovery strategies and cognitive principles
//!
//! Combines cognitive error design with production-ready recovery strategies

pub mod cognitive;
pub mod recovery;
pub mod cue;

// Re-export core error types from cognitive module
pub use cognitive::{
    CognitiveError, CognitiveErrorBuilder, ErrorContext, PartialResult, CognitiveContext,
};

// Re-export recovery types and utilities
pub use recovery::{ErrorRecovery, ResultExt, unreachable_pattern};

// Re-export the enhanced error types
pub use cognitive::{EngramError, RecoveryStrategy, Result};

// Re-export cue error types
pub use cue::CueError;