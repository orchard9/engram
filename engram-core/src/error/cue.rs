//! Cue-specific error types
//!
//! Provides error handling for cue processing operations

use thiserror::Error;

/// Errors that can occur during cue processing
#[derive(Error, Debug)]
pub enum CueError {
    #[error("Unsupported cue type: {cue_type} for operation {operation}")]
    /// Occurs when attempting to use a cue type that isn't supported for the given operation
    UnsupportedCueType {
        /// The cue type that was attempted
        cue_type: String,
        /// The operation that doesn't support this cue type
        operation: String,
    },

    #[error("Invalid cue configuration: {reason}")]
    /// Occurs when cue configuration is invalid or malformed
    InvalidConfiguration {
        /// The reason why the configuration is invalid
        reason: String,
    },

    #[error("Cue processing failed: {source}")]
    /// Occurs when cue processing fails due to an underlying error
    ProcessingFailed {
        #[from]
        /// The underlying error that caused processing to fail
        source: Box<dyn std::error::Error + Send + Sync>,
    },

    #[error("Memory access error: {reason}")]
    /// Occurs when cue processing cannot access required memory
    MemoryAccessError {
        /// The reason why memory access failed
        reason: String,
    },

    #[error("Index operation failed: {reason}")]
    /// Occurs when cue-related index operations fail
    IndexError {
        /// The reason why the index operation failed
        reason: String,
    },
}
