//! Cue-specific error types
//!
//! Provides error handling for cue processing operations

use thiserror::Error;

/// Errors that can occur during cue processing
#[derive(Error, Debug)]
pub enum CueError {
    #[error("Unsupported cue type: {cue_type} for operation {operation}")]
    UnsupportedCueType {
        cue_type: String,
        operation: String,
    },
    
    #[error("Invalid cue configuration: {reason}")]
    InvalidConfiguration { reason: String },
    
    #[error("Cue processing failed: {source}")]
    ProcessingFailed {
        #[from]
        source: Box<dyn std::error::Error + Send + Sync>,
    },
    
    #[error("Memory access error: {reason}")]
    MemoryAccessError { reason: String },
    
    #[error("Index operation failed: {reason}")]
    IndexError { reason: String },
}