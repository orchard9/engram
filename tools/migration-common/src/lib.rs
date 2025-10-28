//! Shared migration infrastructure for database migrations to Engram
//!
//! Provides common functionality for migrating from various databases
//! (Neo4j, PostgreSQL, Redis) to Engram's cognitive memory graph.

pub mod checkpoint;
pub mod embedding_generator;
pub mod error;
pub mod progress;
pub mod streaming;
pub mod validator;

pub use checkpoint::{Checkpoint, CheckpointManager};
pub use embedding_generator::{ContentHash, EmbeddingCache, EmbeddingGenerator};
pub use error::{MigrationError, MigrationResult};
pub use progress::ProgressTracker;
pub use streaming::{DataSource, MemoryTransformer, MigrationPipeline, SourceRecord};
pub use validator::{
    CountReport, EdgeReport, EmbeddingReport, MigrationValidator, SampleReport, SourceStatistics,
};
