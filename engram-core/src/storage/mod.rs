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

use crate::{Confidence, Cue, Episode, Memory};
use std::sync::Arc;
use thiserror::Error;

// Conditional imports based on feature flags
#[cfg(feature = "memory_mapped_persistence")]
pub mod cache;
#[cfg(feature = "memory_mapped_persistence")]
pub mod compact;
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

#[cfg(all(feature = "memory_mapped_persistence", unix))]
pub mod numa;

// Re-exports for public API
#[cfg(feature = "memory_mapped_persistence")]
pub use cache::{CacheOptimalMemoryNode, CognitiveIndex};
#[cfg(feature = "memory_mapped_persistence")]
pub use mapped::{MappedStorage, NumaMemoryMap};
#[cfg(feature = "memory_mapped_persistence")]
pub use tiers::{CognitiveTierArchitecture, TierCoordinator};
#[cfg(feature = "memory_mapped_persistence")]
pub use wal::{WalEntry, WalWriter};

/// Core storage traits for pluggable backends
pub trait StorageTier: Send + Sync {
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
    pub memory_count: usize,
    pub total_size_bytes: u64,
    pub average_activation: f32,
    pub last_access_time: std::time::SystemTime,
    pub cache_hit_rate: f32,
    pub compaction_ratio: f32,
}

/// Persistent storage backend interface
pub trait PersistentBackend: Send + Sync {
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
    pub data_directory: std::path::PathBuf,
    pub max_memory_mb: usize,
    pub enable_numa_awareness: bool,
    pub wal_buffer_size_mb: usize,
    pub compaction_threshold: f32,
    pub cache_size_mb: usize,
    pub enable_huge_pages: bool,
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
    pub recovered_entries: usize,
    pub corrupted_entries: usize,
    pub recovery_duration: std::time::Duration,
    pub last_valid_sequence: u64,
}

/// Data integrity validation report
#[derive(Debug)]
pub struct IntegrityReport {
    pub total_entries_checked: usize,
    pub checksum_failures: usize,
    pub missing_entries: Vec<String>,
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
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Memory allocation failed: {0}")]
    AllocationFailed(String),

    #[error("Corruption detected: {0}")]
    CorruptionDetected(String),

    #[error("Memory mapping failed: {0}")]
    MmapFailed(String),

    #[error("WAL operation failed: {0}")]
    WalFailed(String),

    #[error("Configuration error: {0}")]
    Configuration(String),

    #[error("NUMA topology error: {0}")]
    NumaError(String),

    #[error("Checksum verification failed: expected {expected:x}, got {actual:x}")]
    ChecksumMismatch { expected: u32, actual: u32 },

    #[error("Storage backend not initialized")]
    NotInitialized,

    #[error("Operation timeout after {duration:?}")]
    Timeout { duration: std::time::Duration },
}

impl StorageError {
    pub fn mmap_failed(msg: &str) -> Self {
        Self::MmapFailed(msg.to_string())
    }

    pub fn allocation_failed(msg: &str) -> Self {
        Self::AllocationFailed(msg.to_string())
    }

    pub fn corruption_detected(msg: &str) -> Self {
        Self::CorruptionDetected(msg.to_string())
    }

    pub fn wal_failed(msg: &str) -> Self {
        Self::WalFailed(msg.to_string())
    }
}

/// Result type for storage operations
pub type StorageResult<T> = Result<T, StorageError>;

/// Performance metrics for storage operations
#[derive(Debug, Default)]
pub struct StorageMetrics {
    pub writes_total: std::sync::atomic::AtomicU64,
    pub reads_total: std::sync::atomic::AtomicU64,
    pub bytes_written: std::sync::atomic::AtomicU64,
    pub bytes_read: std::sync::atomic::AtomicU64,
    pub fsync_count: std::sync::atomic::AtomicU64,
    pub cache_hits: std::sync::atomic::AtomicU64,
    pub cache_misses: std::sync::atomic::AtomicU64,
    pub page_faults: std::sync::atomic::AtomicU64,
    pub compactions: std::sync::atomic::AtomicU64,
}

impl StorageMetrics {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn record_write(&self, bytes: u64) {
        self.writes_total
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.bytes_written
            .fetch_add(bytes, std::sync::atomic::Ordering::Relaxed);
    }

    pub fn record_read(&self, bytes: u64) {
        self.reads_total
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.bytes_read
            .fetch_add(bytes, std::sync::atomic::Ordering::Relaxed);
    }

    pub fn record_fsync(&self) {
        self.fsync_count
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }

    pub fn record_cache_hit(&self) {
        self.cache_hits
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }

    pub fn record_cache_miss(&self) {
        self.cache_misses
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }

    pub fn cache_hit_rate(&self) -> f32 {
        let hits = self.cache_hits.load(std::sync::atomic::Ordering::Relaxed);
        let misses = self.cache_misses.load(std::sync::atomic::Ordering::Relaxed);

        if hits + misses == 0 {
            0.0
        } else {
            hits as f32 / (hits + misses) as f32
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
        assert_eq!(policy.hot_activation_threshold, 0.7);
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
