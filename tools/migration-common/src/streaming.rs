//! Streaming and batching infrastructure for migrations

use crate::checkpoint::Checkpoint;
use crate::error::MigrationResult;
use engram_core::{Confidence, Episode};
use serde_json::Value;
use std::collections::HashMap;

/// Source record from external database
#[derive(Debug, Clone)]
pub struct SourceRecord {
    /// Unique ID in source system
    pub id: String,
    /// Text content for embedding generation
    pub text_content: String,
    /// Additional properties as JSON
    pub properties: HashMap<String, Value>,
    /// Optional creation timestamp
    pub created_at: Option<chrono::DateTime<chrono::Utc>>,
}

/// Trait for data source extraction
pub trait DataSource: Send {
    /// Get next batch of records
    fn next_batch(&mut self) -> MigrationResult<Vec<SourceRecord>>;

    /// Get total record count (if known)
    fn total_records(&self) -> Option<u64>;

    /// Get checkpoint for current position
    fn checkpoint(&self) -> MigrationResult<Option<Checkpoint>>;

    /// Resume from a checkpoint
    fn resume_from(&mut self, checkpoint: &Checkpoint) -> MigrationResult<()>;
}

/// Trait for transforming source records to Engram memories
pub trait MemoryTransformer: Send {
    /// Transform a source record into an Engram episode
    fn transform(&self, record: &SourceRecord, embedding: &[f32; 768]) -> MigrationResult<Episode>;

    /// Get memory space ID for a record
    fn memory_space_id(&self, record: &SourceRecord) -> String;

    /// Get initial confidence for transformed memories
    fn initial_confidence(&self) -> Confidence;
}

/// Migration pipeline coordinating extraction, transformation, and loading
pub struct MigrationPipeline<S: DataSource, T: MemoryTransformer> {
    source: S,
    transformer: T,
    batch_size: usize,
}

impl<S: DataSource, T: MemoryTransformer> MigrationPipeline<S, T> {
    /// Create a new migration pipeline
    #[must_use]
    pub const fn new(source: S, transformer: T, batch_size: usize) -> Self {
        Self {
            source,
            transformer,
            batch_size,
        }
    }

    /// Get reference to data source
    pub const fn source(&self) -> &S {
        &self.source
    }

    /// Get mutable reference to data source
    pub const fn source_mut(&mut self) -> &mut S {
        &mut self.source
    }

    /// Get reference to transformer
    pub const fn transformer(&self) -> &T {
        &self.transformer
    }

    /// Get batch size
    #[must_use]
    pub const fn batch_size(&self) -> usize {
        self.batch_size
    }

    /// Process next batch from the pipeline
    pub fn next_batch(&mut self) -> MigrationResult<Vec<SourceRecord>> {
        self.source.next_batch()
    }
}

/// Report from a completed migration
#[derive(Debug, Clone)]
pub struct MigrationReport {
    /// Total records migrated
    pub total_records: u64,
    /// Total time elapsed
    pub elapsed_time: std::time::Duration,
    /// Average records per second
    pub avg_rate: f64,
    /// Number of errors encountered (non-fatal)
    pub error_count: u64,
}

impl MigrationReport {
    /// Create a new migration report
    #[must_use]
    pub fn new(total_records: u64, elapsed_time: std::time::Duration, error_count: u64) -> Self {
        #[allow(clippy::cast_precision_loss)]
        let avg_rate = if elapsed_time.as_secs() > 0 {
            total_records as f64 / elapsed_time.as_secs_f64()
        } else {
            0.0
        };

        Self {
            total_records,
            elapsed_time,
            avg_rate,
            error_count,
        }
    }

    /// Print summary to console
    pub fn print_summary(&self) {
        println!("\nMigration Complete!");
        println!("==================");
        println!("Total records: {}", self.total_records);
        println!("Elapsed time: {:?}", self.elapsed_time);
        println!("Average rate: {:.0} records/sec", self.avg_rate);
        println!("Errors: {}", self.error_count);
    }
}
