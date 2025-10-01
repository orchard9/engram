//! Storage provider abstraction for persistence backends
//!
//! This module provides a trait-based abstraction over storage backends,
//! allowing graceful fallback from memory-mapped persistence to in-memory storage.

use super::FeatureProvider;
use crate::Episode;
use std::any::Any;
use std::path::Path;
use thiserror::Error;

/// Errors that can occur during storage operations
#[derive(Debug, Error)]
pub enum StorageError {
    #[error("Storage operation failed: {0}")]
    /// Occurs when a storage operation fails due to backend issues or resource constraints
    OperationFailed(String),

    #[error("IO error: {0}")]
    /// Occurs when underlying file system operations fail
    IoError(#[from] std::io::Error),

    #[error("Serialization error: {0}")]
    /// Occurs when data cannot be serialized or deserialized properly
    SerializationError(String),

    #[error("Storage not initialized")]
    /// Occurs when attempting to use storage that hasn't been properly initialized
    NotInitialized,
}

/// Result type for storage operations
pub type StorageResult<T> = Result<T, StorageError>;

/// Trait for storage operations
pub trait Storage: Send + Sync {
    /// Store an episode
    ///
    /// # Errors
    ///
    /// Returns an error when the underlying storage backend fails to persist the episode.
    fn store(&mut self, episode: &Episode) -> StorageResult<()>;

    /// Retrieve an episode by ID
    ///
    /// # Errors
    ///
    /// Propagates failures that occur while reading from the storage backend.
    fn retrieve(&self, id: &str) -> StorageResult<Option<Episode>>;

    /// Delete an episode
    ///
    /// # Errors
    ///
    /// Returns an error if the backend cannot remove the requested episode.
    fn delete(&mut self, id: &str) -> StorageResult<()>;

    /// List all episode IDs
    ///
    /// # Errors
    ///
    /// Returns an error when the backend cannot enumerate stored identifiers.
    fn list_ids(&self) -> StorageResult<Vec<String>>;

    /// Flush any pending writes
    ///
    /// # Errors
    ///
    /// Returns an error if the backend fails to sync pending operations to durable storage.
    fn flush(&mut self) -> StorageResult<()>;

    /// Get storage statistics
    fn stats(&self) -> StorageStats;
}

/// Statistics about storage usage
#[derive(Debug, Clone, Default)]
pub struct StorageStats {
    /// Total number of items stored
    pub total_items: usize,
    /// Total size in bytes
    pub total_bytes: usize,
    /// Compression ratio (compressed/uncompressed)
    pub compression_ratio: f32,
}

/// Provider trait for storage implementations
pub trait StorageProvider: FeatureProvider {
    /// Create a new storage instance
    fn create_storage(&self, path: &Path) -> Box<dyn Storage>;

    /// Get storage configuration
    fn get_config(&self) -> StorageConfig;
}

/// Configuration for storage operations
#[derive(Debug, Clone)]
pub struct StorageConfig {
    /// Maximum memory map size
    pub max_mmap_size: usize,
    /// Enable compression
    pub compression: bool,
    /// Sync mode for durability
    pub sync_mode: SyncMode,
}

impl StorageConfig {
    /// Create a new storage configuration with explicit parameters.
    #[must_use]
    pub const fn new(max_mmap_size: usize, compression: bool, sync_mode: SyncMode) -> Self {
        Self {
            max_mmap_size,
            compression,
            sync_mode,
        }
    }

    /// Default configuration used by in-process backends.
    #[must_use]
    pub const fn default_const() -> Self {
        Self::new(1024 * 1024 * 1024, true, SyncMode::Periodic)
    }
}

impl Default for StorageConfig {
    fn default() -> Self {
        Self::default_const()
    }
}

/// Sync modes for storage durability
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SyncMode {
    /// Sync on every write
    Immediate,
    /// Sync periodically
    Periodic,
    /// Never sync (fastest, least durable)
    Never,
}

/// Memory-mapped storage provider (only available when feature is enabled)
#[cfg(feature = "memory_mapped_persistence")]
pub struct MmapStorageProvider {
    config: StorageConfig,
}

#[cfg(feature = "memory_mapped_persistence")]
impl MmapStorageProvider {
    /// Create a new memory-mapped storage provider with default configuration
    #[must_use]
    pub const fn new() -> Self {
        Self {
            config: StorageConfig::default_const(),
        }
    }

    /// Create a new memory-mapped storage provider with custom configuration
    #[must_use]
    pub const fn with_config(config: StorageConfig) -> Self {
        Self { config }
    }
}

#[cfg(feature = "memory_mapped_persistence")]
impl Default for MmapStorageProvider {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(feature = "memory_mapped_persistence")]
impl FeatureProvider for MmapStorageProvider {
    fn is_enabled(&self) -> bool {
        true
    }

    fn name(&self) -> &'static str {
        "storage_memory_mapped"
    }

    fn description(&self) -> &'static str {
        "Memory-mapped file persistence for efficient storage"
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

#[cfg(feature = "memory_mapped_persistence")]
impl StorageProvider for MmapStorageProvider {
    fn create_storage(&self, path: &Path) -> Box<dyn Storage> {
        // Use the NullStorageProvider as a simple fallback for now
        use crate::features::null_impls::NullStorageProvider;
        let null_provider = NullStorageProvider::new();
        null_provider.create_storage(path)
    }

    fn get_config(&self) -> StorageConfig {
        self.config.clone()
    }
}
