//! High-performance batch processing system for Engram's cognitive memory architecture
//!
//! Provides streaming batch operations with lock-free concurrency, SIMD acceleration,
//! and bounded memory usage while maintaining cognitive semantics.

use crate::{Activation, Confidence, Cue, Episode};

pub mod buffers;
pub mod engine;
pub mod operations;
// pub mod streaming; // Requires tokio, commented out for now
pub mod collector;

pub use buffers::BatchBuffer;
pub use engine::BatchEngine;
// Streaming functionality requires additional dependencies - disabled for now
// #[cfg(feature = "streaming")]
// pub use streaming::StreamingBatchProcessor;
pub use collector::AtomicResultCollector;

/// High-throughput batch operations with streaming interfaces and bounded memory usage
pub trait BatchOperations: Send + Sync {
    /// Batch store multiple episodes with graceful degradation under memory pressure
    /// Returns individual activation levels for each episode (maintains cognitive semantics)
    fn batch_store(&self, episodes: Vec<Episode>, config: BatchConfig) -> BatchStoreResult;

    /// Batch recall with parallel processing
    /// Leverages existing SIMD compute module and maintains confidence scoring
    fn batch_recall(&self, cues: Vec<Cue>, config: BatchConfig) -> BatchRecallResult;

    /// Batch similarity search using existing `compute::cosine_similarity_batch_768`
    /// Integrates with existing `MemoryStore` `hot_memories` for cache efficiency
    fn batch_similarity_search(
        &self,
        query_embeddings: &[[f32; 768]],
        k: usize,
        threshold: Confidence,
    ) -> BatchSimilarityResult;
}

/// Configuration for batch operations with adaptive sizing
#[derive(Debug, Clone)]
pub struct BatchConfig {
    /// Target batch size (adaptive based on system pressure)
    pub batch_size: usize,
    /// Memory limit per batch operation (bytes)
    pub memory_limit_mb: usize,
    /// Use SIMD acceleration when available
    pub use_simd: bool,
    /// Enable streaming results for large batches
    pub streaming_threshold: usize,
    /// Backpressure handling strategy
    pub backpressure_strategy: BackpressureStrategy,
}

impl Default for BatchConfig {
    fn default() -> Self {
        Self {
            batch_size: 1000,
            memory_limit_mb: 256,
            use_simd: true,
            streaming_threshold: 10000,
            backpressure_strategy: BackpressureStrategy::AdaptiveResize,
        }
    }
}

/// Batch operation types that mirror existing `MemoryStore` operations
#[derive(Debug, Clone)]
pub enum BatchOperation {
    /// Store an episode in memory
    Store(Episode),
    /// Recall memories using a cue
    Recall(Cue),
    /// Search for similar embeddings
    SimilaritySearch {
        /// 768-dimensional embedding vector to search with
        embedding: [f32; 768],
        /// Number of top results to return
        k: usize,
        /// Minimum confidence threshold for results
        threshold: Confidence,
    },
}

/// Streaming batch result maintaining cognitive semantics
#[derive(Debug, Clone)]
pub struct BatchResult {
    /// Operation ID for result tracking
    pub operation_id: usize,
    /// Result of the batch operation  
    pub result: BatchOperationResult,
    /// Processing metadata
    pub metadata: BatchMetadata,
}

/// Result of a batch operation
#[derive(Debug, Clone)]
pub enum BatchOperationResult {
    /// Store result with activation level (cognitive semantics preserved)
    Store {
        /// Activation level achieved during storage
        activation: Activation,
        /// Unique identifier for the stored memory
        memory_id: String,
    },
    /// Recall results with confidence scores
    Recall(Vec<(Episode, Confidence)>),
    /// Similarity search results
    SimilaritySearch(Vec<(String, Confidence)>),
}

/// Metadata about batch operation processing
#[derive(Debug, Clone)]
pub struct BatchMetadata {
    /// Processing time in microseconds
    pub processing_time_us: u64,
    /// Whether SIMD was used
    pub simd_used: bool,
    /// Memory pressure at time of processing
    pub memory_pressure: f32,
}

/// Batch operation result aggregation maintaining cognitive semantics
#[derive(Debug)]
pub struct BatchStoreResult {
    /// Individual activation levels for each stored episode (cognitive semantics)
    pub activations: Vec<Activation>,
    /// Successfully processed episodes
    pub successful_count: usize,
    /// Operations that encountered degradation
    pub degraded_count: usize,
    /// Total processing time
    pub processing_time_ms: u64,
    /// Memory pressure encountered during batch
    pub peak_memory_pressure: f32,
}

/// Result of batch recall operations
#[derive(Debug)]
pub struct BatchRecallResult {
    /// Results for each cue in the batch
    pub results: Vec<Vec<(Episode, Confidence)>>,
    /// Total processing time
    pub processing_time_ms: u64,
    /// Number of cues that used SIMD acceleration
    pub simd_accelerated_count: usize,
}

/// Result of batch similarity search operations
#[derive(Debug)]
pub struct BatchSimilarityResult {
    /// Top-k similar memories for each query
    pub results: Vec<Vec<(String, Confidence)>>,
    /// Total processing time
    pub processing_time_ms: u64,
    /// SIMD efficiency metric
    pub simd_efficiency: f32,
}

/// Backpressure handling strategies for streaming batch operations
#[derive(Debug, Clone)]
pub enum BackpressureStrategy {
    /// Drop oldest operations when buffer full
    DropOldest,
    /// Reduce batch sizes under pressure
    AdaptiveResize,
    /// Block until capacity available
    Block,
    /// Switch to degraded single-operation mode
    FallbackMode,
}

/// Batch processing errors
#[derive(Debug, thiserror::Error)]
pub enum BatchError {
    /// Batch operation capacity was exceeded
    #[error("Batch capacity exceeded")]
    CapacityExceeded,
    /// Memory limit was exceeded during batch processing
    #[error("Memory limit exceeded: {current_mb}MB > {limit_mb}MB")]
    MemoryLimitExceeded {
        /// Current memory usage in MB
        current_mb: usize,
        /// Memory limit in MB
        limit_mb: usize,
    },
    /// Invalid batch configuration provided
    #[error("Invalid batch configuration: {0}")]
    InvalidConfig(String),
}
