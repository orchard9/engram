//! Incremental recall with snapshot isolation for streaming queries.
//!
//! This module implements snapshot-isolated recall that returns committed observations
//! plus probabilistically-available recent observations, with incremental result streaming
//! for low first-result latency (< 10ms target, bounded staleness P99 < 100ms).
//!
//! ## Architecture
//!
//! ```text
//! Client Request → Capture Generation Snapshot
//!                          ↓
//!                  HNSW Search (async, blocking pool)
//!                          ↓
//!                  Filter by generation <= snapshot
//!                          ↓
//!                  Stream results in batches (10-100 items)
//!                          ↓
//!                  First batch < 10ms latency
//! ```
//!
//! ## Consistency Model
//!
//! - **Snapshot Isolation**: Observations committed before snapshot T are visible
//! - **Bounded Staleness**: Observations in queue may be visible (probabilistic)
//! - **Not Linearizable**: Acceptable for cognitive memory (eventual consistency)
//!
//! ## Performance Characteristics
//!
//! - First result latency: P99 < 10ms
//! - Visibility staleness: P99 < 100ms (observation → visible in recall)
//! - Memory overhead: < 1MB per active recall stream
//! - Throughput: 1000+ concurrent recall streams

use crate::index::{CognitiveHnswIndex, HnswError};
use crate::memory::{Cue, CueType};
use crate::streaming::ObservationQueue;
use crate::{Confidence, compute::VectorOps};
use std::sync::Arc;
use std::time::Instant;
use thiserror::Error;

/// Errors that can occur during recall operations.
#[derive(Debug, Error)]
pub enum RecallError {
    /// Invalid cue provided (e.g., malformed embedding)
    #[error("Invalid cue: {0}")]
    InvalidCue(String),

    /// Search operation failed
    #[error("Search failed: {0}")]
    SearchFailed(String),

    /// HNSW index error
    #[error("HNSW error: {0}")]
    HnswError(#[from] HnswError),

    /// Generation mismatch or invalid generation
    #[error("Invalid generation: {0}")]
    InvalidGeneration(String),
}

/// Configuration for snapshot-isolated recall.
#[derive(Clone, Debug)]
pub struct SnapshotRecallConfig {
    /// Snapshot generation (observations with generation <= this are visible).
    pub snapshot_generation: u64,

    /// Batch size for incremental result streaming.
    /// Default: 10 (optimized for low latency)
    /// Range: 1-100 (smaller = lower latency, larger = higher throughput)
    pub batch_size: usize,

    /// Include in-flight observations (bounded staleness mode).
    /// - `true`: Show observations between snapshot and current (probabilistic visibility)
    /// - `false`: Strict snapshot isolation (only committed observations)
    pub include_recent: bool,

    /// Confidence threshold for filtering results.
    pub confidence_threshold: Confidence,

    /// Maximum number of results to return.
    pub max_results: usize,
}

impl Default for SnapshotRecallConfig {
    fn default() -> Self {
        Self {
            snapshot_generation: 0,
            batch_size: 10,        // Low latency default
            include_recent: false, // Strict snapshot isolation by default
            confidence_threshold: Confidence::LOW,
            max_results: 100,
        }
    }
}

/// Incremental recall stream that yields batches of results.
///
/// Executes HNSW search once (async in blocking thread pool), then streams
/// results in batches for low first-result latency.
///
/// ## Example
///
/// ```ignore
/// use engram_core::streaming::recall::IncrementalRecallStream;
/// use engram_core::streaming::ObservationQueue;
/// use engram_core::memory::Cue;
/// use engram_core::index::CognitiveHnswIndex;
/// use std::sync::Arc;
///
/// let queue = ObservationQueue::new(Default::default());
/// let index = Arc::new(CognitiveHnswIndex::new());
/// let vector_ops = Arc::new(ScalarVectorOps::new());
///
/// let cue = Cue::embedding("query".to_string(), [0.5f32; 768], Confidence::HIGH);
///
/// let mut stream = IncrementalRecallStream::new(
///     index,
///     cue,
///     &queue,
///     vector_ops,
///     false, // strict snapshot isolation
/// );
///
/// // Execute search (async, runs in blocking thread pool)
/// stream.search().await?;
///
/// // Stream results incrementally
/// while let Some(batch) = stream.next_batch() {
///     for memory in batch {
///         println!("Memory: {:?}", memory.id);
///     }
/// }
/// ```
pub struct IncrementalRecallStream {
    /// HNSW index reference
    index: Arc<CognitiveHnswIndex>,

    /// Query cue
    cue: Cue,

    /// Snapshot configuration
    config: SnapshotRecallConfig,

    /// Vector operations for distance calculations
    vector_ops: Arc<dyn VectorOps>,

    /// Current position in results (for batching)
    position: usize,

    /// Cached results from HNSW search (Memory ID, similarity score)
    results: Vec<(String, f32)>,

    /// Search execution timestamp (for latency measurement)
    search_started_at: Option<Instant>,

    /// Search completion timestamp
    search_completed_at: Option<Instant>,
}

impl IncrementalRecallStream {
    /// Create a new incremental recall stream.
    ///
    /// Captures current generation from observation queue as snapshot point.
    ///
    /// # Arguments
    ///
    /// * `index` - HNSW index to search
    /// * `cue` - Query cue for recall
    /// * `observation_queue` - Queue for generation tracking
    /// * `vector_ops` - Vector operations implementation
    /// * `include_recent` - Include uncommitted observations (bounded staleness mode)
    #[must_use]
    pub fn new(
        index: Arc<CognitiveHnswIndex>,
        cue: Cue,
        observation_queue: &ObservationQueue,
        vector_ops: Arc<dyn VectorOps>,
        include_recent: bool,
    ) -> Self {
        let snapshot_generation = observation_queue.current_generation();

        // Extract configuration from cue
        let confidence_threshold = cue.result_threshold;
        let max_results = cue.max_results;

        Self {
            index,
            cue,
            config: SnapshotRecallConfig {
                snapshot_generation,
                batch_size: 10, // Low latency default
                include_recent,
                confidence_threshold,
                max_results,
            },
            vector_ops,
            position: 0,
            results: Vec::new(),
            search_started_at: None,
            search_completed_at: None,
        }
    }

    /// Create with explicit configuration (for testing/advanced use).
    #[must_use]
    pub fn with_config(
        index: Arc<CognitiveHnswIndex>,
        cue: Cue,
        config: SnapshotRecallConfig,
        vector_ops: Arc<dyn VectorOps>,
    ) -> Self {
        Self {
            index,
            cue,
            config,
            vector_ops,
            position: 0,
            results: Vec::new(),
            search_started_at: None,
            search_completed_at: None,
        }
    }

    /// Execute HNSW search with snapshot isolation.
    ///
    /// This is a potentially expensive operation - should run in blocking thread pool.
    /// Filters results by generation for snapshot isolation.
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - Cue is invalid (malformed embedding)
    /// - HNSW search fails
    /// - Generation filtering encounters invalid data
    pub async fn search(&mut self) -> Result<(), RecallError> {
        self.search_started_at = Some(Instant::now());

        // Extract query embedding from cue
        let query_embedding = self.extract_query_embedding()?;

        // Clone needed values for blocking task
        let index = Arc::clone(&self.index);
        let _vector_ops = Arc::clone(&self.vector_ops);
        let snapshot_gen = self.config.snapshot_generation;
        let include_recent = self.config.include_recent;
        let confidence_threshold = self.config.confidence_threshold;
        let max_results = self.config.max_results;

        // Execute HNSW search in blocking thread pool
        let search_results = tokio::task::spawn_blocking(move || {
            // Apply generation filtering based on snapshot mode
            let max_generation = if include_recent {
                None // No filtering - include all nodes (bounded staleness mode)
            } else {
                Some(snapshot_gen) // Strict snapshot isolation
            };

            index.search_with_generation(
                &query_embedding,
                max_results,
                confidence_threshold,
                max_generation,
            )
        })
        .await
        .map_err(|e| RecallError::SearchFailed(format!("Task join error: {e}")))?;

        // Convert to (memory_id, similarity) pairs
        self.results = search_results
            .into_iter()
            .map(|(memory_id, confidence)| (memory_id, confidence.raw()))
            .collect();

        self.search_completed_at = Some(Instant::now());

        Ok(())
    }

    /// Get next batch of results (for incremental streaming).
    ///
    /// Returns `None` when all results have been yielded.
    ///
    /// # Example
    ///
    /// ```ignore
    /// while let Some(batch) = stream.next_batch() {
    ///     for memory in batch {
    ///         println!("Memory: {:?}", memory.id);
    ///     }
    /// }
    /// ```
    #[must_use]
    pub fn next_batch(&mut self) -> Option<Vec<RecallBatchItem>> {
        if self.position >= self.results.len() {
            return None;
        }

        let end = (self.position + self.config.batch_size).min(self.results.len());

        let batch: Vec<RecallBatchItem> = self.results[self.position..end]
            .iter()
            .map(|(memory_id, similarity)| RecallBatchItem {
                memory_id: memory_id.clone(),
                similarity: *similarity,
                confidence: Confidence::exact(*similarity),
            })
            .collect();

        self.position = end;

        Some(batch)
    }

    /// Check if more results are available.
    #[must_use]
    pub const fn has_more(&self) -> bool {
        self.position < self.results.len()
    }

    /// Get total result count.
    #[must_use]
    pub const fn total_count(&self) -> usize {
        self.results.len()
    }

    /// Get search latency (time spent in HNSW search).
    #[must_use]
    pub fn search_latency(&self) -> Option<std::time::Duration> {
        match (self.search_started_at, self.search_completed_at) {
            (Some(start), Some(end)) => Some(end.duration_since(start)),
            _ => None,
        }
    }

    /// Get snapshot generation used for this recall.
    #[must_use]
    pub const fn snapshot_generation(&self) -> u64 {
        self.config.snapshot_generation
    }

    /// Get batch size configuration.
    #[must_use]
    pub const fn batch_size(&self) -> usize {
        self.config.batch_size
    }

    /// Extract query embedding from cue.
    ///
    /// # Errors
    ///
    /// Returns error if cue type is not supported or embedding is invalid.
    fn extract_query_embedding(&self) -> Result<[f32; 768], RecallError> {
        match &self.cue.cue_type {
            CueType::Embedding { vector, .. } => {
                // Validate embedding dimension
                if vector.len() != 768 {
                    return Err(RecallError::InvalidCue(format!(
                        "Invalid embedding dimension: {} (expected 768)",
                        vector.len()
                    )));
                }
                Ok(*vector)
            }
            CueType::Context { .. } => Err(RecallError::InvalidCue(
                "Context cues not yet supported for incremental recall".to_string(),
            )),
            CueType::Semantic { .. } => Err(RecallError::InvalidCue(
                "Semantic cues not yet supported for incremental recall".to_string(),
            )),
            CueType::Temporal { .. } => Err(RecallError::InvalidCue(
                "Temporal cues not yet supported for incremental recall".to_string(),
            )),
        }
    }
}

/// Single item in a recall batch.
#[derive(Clone, Debug)]
pub struct RecallBatchItem {
    /// Memory identifier
    pub memory_id: String,
    /// Similarity score (0.0-1.0, higher is better)
    pub similarity: f32,
    /// Confidence in this result
    pub confidence: Confidence,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Confidence;
    use crate::compute::scalar::ScalarVectorOps;
    use crate::streaming::{ObservationQueue, QueueConfig};

    #[test]
    #[allow(clippy::expect_used)]
    fn test_snapshot_config_default() {
        let config = SnapshotRecallConfig::default();
        assert_eq!(config.batch_size, 10);
        assert!(!config.include_recent);
        assert_eq!(config.confidence_threshold, Confidence::LOW);
        assert_eq!(config.max_results, 100);
    }

    #[test]
    #[allow(clippy::expect_used)]
    fn test_recall_stream_creation() {
        let queue = ObservationQueue::new(QueueConfig::default());
        let index = Arc::new(CognitiveHnswIndex::new());
        let vector_ops = Arc::new(ScalarVectorOps::new());

        let cue = Cue::embedding("test".to_string(), [0.5f32; 768], Confidence::HIGH);

        let stream = IncrementalRecallStream::new(index, cue, &queue, vector_ops, false);

        assert_eq!(stream.snapshot_generation(), 0);
        assert_eq!(stream.batch_size(), 10);
        assert!(!stream.has_more());
        assert_eq!(stream.total_count(), 0);
    }

    #[test]
    #[allow(clippy::expect_used)]
    fn test_generation_tracking() {
        let queue = ObservationQueue::new(QueueConfig::default());

        // Initial generation is 0
        assert_eq!(queue.current_generation(), 0);

        // Mark some generations as committed
        queue.mark_generation_committed(5);
        assert_eq!(queue.current_generation(), 5);

        queue.mark_generation_committed(10);
        assert_eq!(queue.current_generation(), 10);

        // Out-of-order commits should not regress
        queue.mark_generation_committed(7);
        assert_eq!(queue.current_generation(), 10);
    }

    #[test]
    #[allow(clippy::expect_used)]
    fn test_extract_query_embedding_valid() {
        let index = Arc::new(CognitiveHnswIndex::new());
        let vector_ops = Arc::new(ScalarVectorOps::new());
        let queue = ObservationQueue::new(QueueConfig::default());

        let cue = Cue::embedding("test".to_string(), [0.7f32; 768], Confidence::HIGH);

        let stream = IncrementalRecallStream::new(index, cue, &queue, vector_ops, false);

        let embedding = stream.extract_query_embedding();
        assert!(embedding.is_ok(), "Valid embedding should extract");
        #[allow(clippy::ok_expect)]
        let embedding = embedding.expect("checked above");
        assert_eq!(embedding.len(), 768);
        assert!((embedding[0] - 0.7f32).abs() < f32::EPSILON);
    }

    #[test]
    #[allow(clippy::expect_used)]
    fn test_extract_query_embedding_context_unsupported() {
        let index = Arc::new(CognitiveHnswIndex::new());
        let vector_ops = Arc::new(ScalarVectorOps::new());
        let queue = ObservationQueue::new(QueueConfig::default());

        let cue = Cue::context("test".to_string(), None, None, Confidence::HIGH);

        let stream = IncrementalRecallStream::new(index, cue, &queue, vector_ops, false);

        let result = stream.extract_query_embedding();
        assert!(result.is_err());
        assert!(matches!(result, Err(RecallError::InvalidCue(_))));
    }

    #[test]
    #[allow(clippy::expect_used)]
    fn test_batch_iteration_empty() {
        let index = Arc::new(CognitiveHnswIndex::new());
        let vector_ops = Arc::new(ScalarVectorOps::new());
        let queue = ObservationQueue::new(QueueConfig::default());

        let cue = Cue::embedding("test".to_string(), [0.5f32; 768], Confidence::HIGH);

        let mut stream = IncrementalRecallStream::new(index, cue, &queue, vector_ops, false);

        // No results yet (search not executed)
        assert!(stream.next_batch().is_none());
        assert!(!stream.has_more());
    }
}
