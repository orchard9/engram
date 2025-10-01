//! High-performance, tiered storage system with memory-mapped persistence
//!
//! This module provides a NUMA-aware, lock-free storage architecture optimized
//! for Engram's cognitive memory patterns. It features:
//!
//! - Crash-consistent write-ahead logging with sub-10ms durability
//! - Memory-mapped warm/cold tiers with cache-optimal layouts
//! - NUMA-aware allocation for multi-socket scalability
//! - SIMD-optimized batch operations on columnar cold storage
//! - Adaptive tier migration based on cognitive access patterns
//! - Zero-copy reads with hardware-accelerated checksums

#![allow(async_fn_in_trait)]

use crate::{Confidence, Cue, Episode, Memory};
use std::sync::Arc;
use thiserror::Error;

// Conditional imports based on feature flags
pub mod access_tracking;
#[cfg(feature = "memory_mapped_persistence")]
pub mod cache;
pub mod cold_tier;
#[cfg(feature = "memory_mapped_persistence")]
pub mod compact;
pub mod confidence;
pub mod content_addressing;
pub mod deduplication;
pub mod hot_tier;
#[cfg(feature = "memory_mapped_persistence")]
pub mod index;
#[cfg(feature = "memory_mapped_persistence")]
pub mod mapped;
#[cfg(feature = "memory_mapped_persistence")]
pub mod recovery;
#[cfg(feature = "memory_mapped_persistence")]
pub mod tiers;
#[cfg(feature = "memory_mapped_persistence")]
pub mod wal;
pub mod warm_tier;

#[cfg(all(feature = "memory_mapped_persistence", unix))]
pub mod numa;

// Re-exports for public API
pub use crate::activation::storage_aware::StorageTier;
pub use access_tracking::{
    AccessEvent, AccessPredictor, AccessStats, AccessTracker, GlobalAccessStats, PredictedAccess,
    PredictorStats,
};
#[cfg(feature = "memory_mapped_persistence")]
pub use cache::{CacheOptimalMemoryNode, CognitiveIndex};
pub use cold_tier::{ColdTier, ColdTierConfig, CompactionResult};
pub use confidence::{
    CalibrationStats, ConfidenceTier, StorageConfidenceCalibrator, TierConfidenceFactors,
};
pub use content_addressing::{ContentAddress, ContentIndex, ContentIndexStats};
pub use deduplication::{
    DeduplicationAction, DeduplicationResult, DeduplicationStats, MergeStrategy,
    SemanticDeduplicator,
};
pub use hot_tier::HotTier;
#[cfg(feature = "memory_mapped_persistence")]
pub use mapped::MappedWarmStorage;
#[cfg(feature = "memory_mapped_persistence")]
pub use tiers::{
    CognitiveTierArchitecture, MemoryPressure, MigrationCandidate, MigrationReport,
    TierArchitectureStats, TierCoordinator,
};
#[cfg(feature = "memory_mapped_persistence")]
pub use wal::{WalEntry, WalWriter};
pub use warm_tier::{CompactionStats, MemoryUsage, WarmTier, WarmTierConfig};

const MAX_SAFE_INTEGER: u64 = 9_007_199_254_740_992; // 2^53

/// Core storage traits for pluggable backends
pub trait StorageTierBackend: Send + Sync {
    /// Error type produced by the backend implementation.
    type Error: std::error::Error + Send + Sync + 'static;

    /// Store a memory with specified activation level
    async fn store(&self, memory: Arc<Memory>) -> Result<(), Self::Error>;

    /// Recall memories matching the given cue
    async fn recall(&self, cue: &Cue) -> Result<Vec<(Episode, Confidence)>, Self::Error>;

    /// Update activation level for a memory
    async fn update_activation(&self, memory_id: &str, activation: f32) -> Result<(), Self::Error>;

    /// Remove a memory from storage
    async fn remove(&self, memory_id: &str) -> Result<(), Self::Error>;

    /// Get statistics about this storage tier
    fn statistics(&self) -> TierStatistics;

    /// Perform background maintenance (compaction, cleanup, etc.)
    async fn maintenance(&self) -> Result<(), Self::Error>;
}

/// Statistics about a storage tier's state
#[derive(Debug, Clone)]
pub struct TierStatistics {
    /// Number of memories currently housed in the tier.
    pub memory_count: usize,
    /// Total bytes occupied by the tier.
    pub total_size_bytes: u64,
    /// Average activation score across stored memories.
    pub average_activation: f32,
    /// Time of the most recent access touching the tier.
    pub last_access_time: std::time::SystemTime,
    /// Cache hit ratio observed for lookups.
    pub cache_hit_rate: f32,
    /// Ratio of reclaimed space achieved through compaction.
    pub compaction_ratio: f32,
}

/// Persistent storage backend interface
pub trait PersistentBackend: Send + Sync {
    /// Error type produced by the persistent backend implementation.
    type Error: std::error::Error + Send + Sync + 'static;

    /// Initialize the backend with the given configuration
    async fn initialize(&self, config: &StorageConfig) -> Result<(), Self::Error>;

    /// Gracefully shutdown the backend
    async fn shutdown(&self) -> Result<(), Self::Error>;

    /// Force synchronization of all pending writes
    async fn fsync(&self) -> Result<(), Self::Error>;

    /// Recover from crash by replaying write-ahead log
    async fn recover(&self) -> Result<RecoveryReport, Self::Error>;

    /// Perform integrity check on stored data
    async fn validate_integrity(&self) -> Result<IntegrityReport, Self::Error>;
}

/// Configuration for storage backend
#[derive(Debug, Clone)]
pub struct StorageConfig {
    /// Filesystem directory where persisted data lives.
    pub data_directory: std::path::PathBuf,
    /// Maximum in-memory working set measured in MiB.
    pub max_memory_mb: usize,
    /// Whether to enable NUMA-aware scheduling and allocation.
    pub enable_numa_awareness: bool,
    /// Size of the write-ahead log buffer in MiB.
    pub wal_buffer_size_mb: usize,
    /// Compaction trigger threshold expressed as a ratio.
    pub compaction_threshold: f32,
    /// Cache capacity dedicated to hot reads in MiB.
    pub cache_size_mb: usize,
    /// Use huge pages for reduced TLB pressure.
    pub enable_huge_pages: bool,
    /// File synchronization strategy used by the backend.
    pub fsync_mode: FsyncMode,
}

impl Default for StorageConfig {
    fn default() -> Self {
        Self {
            data_directory: std::path::PathBuf::from("./engram_data"),
            max_memory_mb: 1024,
            enable_numa_awareness: true,
            wal_buffer_size_mb: 64,
            compaction_threshold: 0.7,
            cache_size_mb: 256,
            enable_huge_pages: true,
            fsync_mode: FsyncMode::PerBatch,
        }
    }
}

/// File synchronization modes
#[derive(Debug, Clone, Copy)]
pub enum FsyncMode {
    /// Sync every write (safest, slowest)
    PerWrite,
    /// Sync every batch of writes (balanced)
    PerBatch,
    /// Sync on timer (fastest, less safe)
    Timer(std::time::Duration),
    /// No explicit sync (testing only)
    None,
}

/// Recovery report after crash
#[derive(Debug)]
pub struct RecoveryReport {
    /// Number of entries successfully recovered from WAL
    pub recovered_entries: usize,
    /// Number of entries found to be corrupted
    pub corrupted_entries: usize,
    /// Total time taken for recovery process
    pub recovery_duration: std::time::Duration,
    /// Sequence number of last valid WAL entry
    pub last_valid_sequence: u64,
}

/// Data integrity validation report
#[derive(Debug)]
pub struct IntegrityReport {
    /// Total number of entries validated during integrity check
    pub total_entries_checked: usize,
    /// Number of entries that failed checksum validation
    pub checksum_failures: usize,
    /// List of missing entry IDs that were expected but not found
    pub missing_entries: Vec<String>,
    /// List of files that have been corrupted or are unreadable
    pub corrupted_files: Vec<std::path::PathBuf>,
}

/// Tier migration policy based on cognitive patterns
#[derive(Debug, Clone)]
pub struct CognitiveEvictionPolicy {
    /// Activation threshold for hot tier retention
    pub hot_activation_threshold: f32,
    /// Time window for recent access (warm tier)
    pub warm_access_window: std::time::Duration,
    /// Maximum age before cold tier migration
    pub cold_migration_age: std::time::Duration,
    /// Confidence boost factor for recent memories
    pub recency_boost_factor: f32,
}

impl Default for CognitiveEvictionPolicy {
    fn default() -> Self {
        Self {
            hot_activation_threshold: 0.7,
            warm_access_window: std::time::Duration::from_secs(3600), // 1 hour
            cold_migration_age: std::time::Duration::from_secs(86400), // 1 day
            recency_boost_factor: 1.2,
        }
    }
}

/// Errors that can occur in storage operations
#[derive(Error, Debug)]
pub enum StorageError {
    /// Underlying I/O operation failed
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// Memory allocation failed (out of memory or fragmentation)
    #[error("Memory allocation failed: {0}")]
    AllocationFailed(String),

    /// Data corruption detected during read or validation
    #[error("Corruption detected: {0}")]
    CorruptionDetected(String),

    /// Memory mapping operation failed (insufficient virtual memory)
    #[error("Memory mapping failed: {0}")]
    MmapFailed(String),

    /// Write-ahead log operation failed
    #[error("WAL operation failed: {0}")]
    WalFailed(String),

    /// Invalid configuration parameters provided
    #[error("Configuration error: {0}")]
    Configuration(String),

    /// NUMA topology detection or allocation failed
    #[error("NUMA topology error: {0}")]
    NumaError(String),

    /// Checksum verification failed during integrity check
    #[error("Checksum verification failed: expected {expected:x}, got {actual:x}")]
    ChecksumMismatch {
        /// Expected checksum value
        expected: u32,
        /// Actual checksum value found
        actual: u32,
    },

    /// Storage backend has not been initialized
    #[error("Storage backend not initialized: {0}")]
    NotInitialized(String),

    /// Operation exceeded the configured timeout
    #[error("Operation timeout after {duration:?}")]
    Timeout {
        /// Duration after which the operation timed out
        duration: std::time::Duration,
    },

    /// Memory or resource not found
    #[error("Resource not found: {0}")]
    NotFound(String),

    /// Feature not yet implemented
    #[error("Feature not implemented: {0}")]
    NotImplemented(String),

    /// Migration operation failed
    #[error("Migration failed: {0}")]
    MigrationFailed(String),
}

impl StorageError {
    /// Create a memory mapping failure error
    #[must_use]
    pub fn mmap_failed(msg: &str) -> Self {
        Self::MmapFailed(msg.to_string())
    }

    /// Create a memory allocation failure error
    #[must_use]
    pub fn allocation_failed(msg: &str) -> Self {
        Self::AllocationFailed(msg.to_string())
    }

    /// Create a data corruption detection error
    #[must_use]
    pub fn corruption_detected(msg: &str) -> Self {
        Self::CorruptionDetected(msg.to_string())
    }

    /// Create a WAL operation failure error
    #[must_use]
    pub fn wal_failed(msg: &str) -> Self {
        Self::WalFailed(msg.to_string())
    }
}

/// Result type for storage operations
pub type StorageResult<T> = Result<T, StorageError>;

/// Performance metrics for storage operations
#[derive(Debug, Default)]
pub struct StorageMetrics {
    /// Total number of write operations performed
    pub writes_total: std::sync::atomic::AtomicU64,
    /// Total number of read operations performed
    pub reads_total: std::sync::atomic::AtomicU64,
    /// Total bytes written to storage
    pub bytes_written: std::sync::atomic::AtomicU64,
    /// Total bytes read from storage
    pub bytes_read: std::sync::atomic::AtomicU64,
    /// Number of fsync operations performed for durability
    pub fsync_count: std::sync::atomic::AtomicU64,
    /// Number of cache hits (data found in memory)
    pub cache_hits: std::sync::atomic::AtomicU64,
    /// Number of cache misses (data loaded from disk)
    pub cache_misses: std::sync::atomic::AtomicU64,
    /// Number of page faults during memory-mapped operations
    pub page_faults: std::sync::atomic::AtomicU64,
    /// Number of compaction operations performed
    pub compactions: std::sync::atomic::AtomicU64,
}

impl StorageMetrics {
    /// Create a new `StorageMetrics` instance with zero counters
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a write operation with the number of bytes written
    pub fn record_write(&self, bytes: u64) {
        self.writes_total
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.bytes_written
            .fetch_add(bytes, std::sync::atomic::Ordering::Relaxed);
    }

    /// Record a read operation with the number of bytes read
    pub fn record_read(&self, bytes: u64) {
        self.reads_total
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.bytes_read
            .fetch_add(bytes, std::sync::atomic::Ordering::Relaxed);
    }

    /// Record an fsync operation for durability guarantees
    pub fn record_fsync(&self) {
        self.fsync_count
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }

    /// Record a cache hit (data found in memory)
    pub fn record_cache_hit(&self) {
        self.cache_hits
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }

    /// Record a cache miss (data loaded from disk)
    pub fn record_cache_miss(&self) {
        self.cache_misses
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }

    /// Calculate the cache hit rate as a percentage
    pub fn cache_hit_rate(&self) -> f32 {
        let hits = self.cache_hits.load(std::sync::atomic::Ordering::Relaxed);
        let misses = self.cache_misses.load(std::sync::atomic::Ordering::Relaxed);

        if hits + misses == 0 {
            0.0
        } else {
            let total = hits.saturating_add(misses).clamp(1, MAX_SAFE_INTEGER);
            let numerator = hits.clamp(0, MAX_SAFE_INTEGER);
            #[allow(clippy::cast_precision_loss)]
            let ratio = (numerator as f64) / (total as f64);
            #[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
            {
                ratio.clamp(0.0, 1.0) as f32
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_storage_config_defaults() {
        let config = StorageConfig::default();
        assert_eq!(config.max_memory_mb, 1024);
        assert!(config.enable_numa_awareness);
        assert_eq!(config.wal_buffer_size_mb, 64);
    }

    #[test]
    fn test_cognitive_eviction_policy_defaults() {
        let policy = CognitiveEvictionPolicy::default();
        assert!((policy.hot_activation_threshold - 0.7).abs() < f32::EPSILON);
        assert_eq!(policy.warm_access_window.as_secs(), 3600);
        assert_eq!(policy.cold_migration_age.as_secs(), 86400);
    }

    #[test]
    fn test_storage_metrics() {
        let metrics = StorageMetrics::new();

        metrics.record_cache_hit();
        metrics.record_cache_hit();
        metrics.record_cache_miss();

        let hit_rate = metrics.cache_hit_rate();
        assert!((hit_rate - 0.666).abs() < 0.01);
    }
}
