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
    OperationFailed(String),
    
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
    
    #[error("Serialization error: {0}")]
    SerializationError(String),
    
    #[error("Storage not initialized")]
    NotInitialized,
}

/// Result type for storage operations
pub type StorageResult<T> = Result<T, StorageError>;

/// Trait for storage operations
pub trait Storage: Send + Sync {
    /// Store an episode
    fn store(&mut self, episode: &Episode) -> StorageResult<()>;
    
    /// Retrieve an episode by ID
    fn retrieve(&self, id: &str) -> StorageResult<Option<Episode>>;
    
    /// Delete an episode
    fn delete(&mut self, id: &str) -> StorageResult<()>;
    
    /// List all episode IDs
    fn list_ids(&self) -> StorageResult<Vec<String>>;
    
    /// Flush any pending writes
    fn flush(&mut self) -> StorageResult<()>;
    
    /// Get storage statistics
    fn stats(&self) -> StorageStats;
}

/// Statistics about storage usage
#[derive(Debug, Clone, Default)]
pub struct StorageStats {
    pub total_items: usize,
    pub total_bytes: usize,
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

impl Default for StorageConfig {
    fn default() -> Self {
        Self {
            max_mmap_size: 1024 * 1024 * 1024, // 1GB
            compression: true,
            sync_mode: SyncMode::Periodic,
        }
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
    pub fn new() -> Self {
        Self {
            config: StorageConfig::default(),
        }
    }
    
    pub fn with_config(config: StorageConfig) -> Self {
        Self { config }
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