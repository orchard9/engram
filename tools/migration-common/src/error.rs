//! Error types for migration operations

use thiserror::Error;

/// Errors that can occur during database migration
#[derive(Error, Debug)]
pub enum MigrationError {
    /// Data source connection error
    #[error("Failed to connect to data source: {0}")]
    ConnectionError(String),

    /// Error reading from source database
    #[error("Failed to read from source: {0}")]
    SourceReadError(String),

    /// Error transforming source data to Engram format
    #[error("Failed to transform data: {0}")]
    TransformationError(String),

    /// Embedding generation error
    #[error("Failed to generate embedding: {0}")]
    EmbeddingError(String),

    /// Error writing to Engram
    #[error("Failed to write to Engram: {0}")]
    EngramWriteError(String),

    /// Validation error
    #[error("Validation failed: {0}")]
    ValidationError(String),

    /// Checkpoint error
    #[error("Checkpoint operation failed: {0}")]
    CheckpointError(String),

    /// I/O error
    #[error("I/O error: {0}")]
    IoError(#[from] std::io::Error),

    /// JSON serialization error
    #[error("JSON error: {0}")]
    JsonError(#[from] serde_json::Error),

    /// Generic error
    #[error("{0}")]
    Other(String),
}

/// Result type for migration operations
pub type MigrationResult<T> = Result<T, MigrationError>;
