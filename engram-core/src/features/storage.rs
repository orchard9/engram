//! Storage provider abstraction for persistence backends
//!
//! This module provides a trait-based abstraction over storage backends,
//! allowing graceful fallback from memory-mapped persistence to in-memory storage.

use super::FeatureProvider;
use crate::{Episode, Memory};
use std::any::Any;
use std::path::Path;
use std::sync::Arc;
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
        "memory_mapped_persistence"
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
        Box::new(MmapStorageImpl::new(path, self.config.clone()))
    }
    
    fn get_config(&self) -> StorageConfig {
        self.config.clone()
    }
}

/// Actual memory-mapped storage implementation
#[cfg(feature = "memory_mapped_persistence")]
struct MmapStorageImpl {
    path: std::path::PathBuf,
    config: StorageConfig,
    storage: Option<crate::storage::mapped::MappedWarmStorage>,
}

#[cfg(feature = "memory_mapped_persistence")]
impl MmapStorageImpl {
    fn new(path: &Path, config: StorageConfig) -> Self {
        use crate::storage::mapped::MappedWarmStorage;
        use crate::storage::StorageMetrics;
        use crate::storage::StorageTier;
        use std::sync::Arc;
        
        let metrics = Arc::new(StorageMetrics::default());
        let storage = MappedWarmStorage::new(
            path.to_path_buf(),
            config.max_mmap_size,
            metrics,
        ).ok();
        
        Self {
            path: path.to_path_buf(),
            config,
            storage,
        }
    }
}

#[cfg(feature = "memory_mapped_persistence")]
impl Storage for MmapStorageImpl {
    fn store(&mut self, episode: &Episode) -> StorageResult<()> {
        use crate::storage::StorageTier;
        let storage = self.storage.as_mut()
            .ok_or(StorageError::NotInitialized)?;
            
        let memory = Memory::from_episode(episode.clone(), 1.0);
        storage.store(&episode.id, Arc::new(memory), 1.0)
            .map_err(|e| StorageError::OperationFailed(e.to_string()))?;
            
        Ok(())
    }
    
    fn retrieve(&self, id: &str) -> StorageResult<Option<Episode>> {
        use crate::storage::StorageTier;
        let storage = self.storage.as_ref()
            .ok_or(StorageError::NotInitialized)?;
            
        let result = storage.retrieve(id)
            .map_err(|e| StorageError::OperationFailed(e.to_string()))?;
            
        Ok(result.map(|(memory, _)| {
            // Convert Memory back to Episode
            Episode {
                id: memory.id.clone(),
                when: memory.created_at,
                where_location: None,
                who: None,
                what: memory.content.clone().unwrap_or_default(),
                embedding: memory.embedding,
                encoding_confidence: memory.confidence,
                vividness_confidence: memory.confidence,
                reliability_confidence: memory.confidence,
                last_recall: memory.last_access,
                recall_count: 1,
                decay_rate: memory.decay_rate,
            }
        }))
    }
    
    fn delete(&mut self, id: &str) -> StorageResult<()> {
        // Memory-mapped storage doesn't support deletion in our implementation
        Ok(())
    }
    
    fn list_ids(&self) -> StorageResult<Vec<String>> {
        // Would need to maintain an index for this
        Ok(Vec::new())
    }
    
    fn flush(&mut self) -> StorageResult<()> {
        use crate::storage::StorageTier;
        let storage = self.storage.as_mut()
            .ok_or(StorageError::NotInitialized)?;
            
        storage.flush()
            .map_err(|e| StorageError::OperationFailed(e.to_string()))?;
            
        Ok(())
    }
    
    fn stats(&self) -> StorageStats {
        self.storage.as_ref()
            .map(|s| {
                let metrics = s.get_metrics();
                StorageStats {
                    total_items: metrics.hot_tier_count.load(std::sync::atomic::Ordering::Relaxed),
                    total_bytes: metrics.total_bytes_written.load(std::sync::atomic::Ordering::Relaxed),
                    compression_ratio: 1.0, // TODO: Calculate actual ratio
                }
            })
            .unwrap_or_default()
    }
}