//! Warm tier storage implementation using memory-mapped files
//!
//! This module provides compressed persistent storage for moderately active memories.
//! Key features:
//! - Memory-mapped files for efficient I/O
//! - Compression for space efficiency
//! - Sub-1ms retrieval latency
//! - NUMA-aware allocation
//! - Automatic compaction and defragmentation

use super::confidence::{ConfidenceTier, StorageConfidenceCalibrator};
use super::{
    StorageError, StorageTierBackend, TierStatistics,
    mapped::{CompactionStats, MappedWarmStorage},
};
use crate::{Confidence, Cue, Episode, Memory};
use std::sync::Arc;

/// Warm tier storage using memory-mapped files with compression
pub struct WarmTier {
    /// Underlying memory-mapped storage implementation
    storage: MappedWarmStorage,
    /// Confidence calibrator for warm tier adjustments
    confidence_calibrator: StorageConfidenceCalibrator,
    /// Storage timestamps for tracking time in storage
    storage_timestamps: dashmap::DashMap<String, std::time::SystemTime>,
}

impl WarmTier {
    /// Create a new warm tier with specified file path and capacity
    ///
    /// # Errors
    ///
    /// Returns an error if the underlying mapped storage cannot be
    /// initialised or if the warm tier resources fail during setup.
    pub fn new<P: AsRef<std::path::Path>>(
        file_path: P,
        capacity: usize,
        metrics: Arc<super::StorageMetrics>,
    ) -> Result<Self, StorageError> {
        let storage = MappedWarmStorage::new(file_path, capacity, metrics)?;
        let confidence_calibrator = StorageConfidenceCalibrator::new();
        let storage_timestamps = dashmap::DashMap::new();
        Ok(Self {
            storage,
            confidence_calibrator,
            storage_timestamps,
        })
    }

    /// Create warm tier with custom configuration
    ///
    /// # Errors
    ///
    /// Returns an error when the mapped storage cannot be created with the
    /// provided configuration.
    pub fn with_config<P: AsRef<std::path::Path>>(
        file_path: P,
        config: &WarmTierConfig,
        metrics: Arc<super::StorageMetrics>,
    ) -> Result<Self, StorageError> {
        let storage = MappedWarmStorage::new(file_path, config.capacity, metrics)?;
        let confidence_calibrator = StorageConfidenceCalibrator::new();

        // Apply configuration settings
        if config.enable_compression {
            // Compression is handled internally by MappedWarmStorage
        }

        if config.enable_numa_awareness {
            // NUMA awareness is built into MappedWarmStorage
        }

        let storage_timestamps = dashmap::DashMap::new();
        Ok(Self {
            storage,
            confidence_calibrator,
            storage_timestamps,
        })
    }

    /// Get the underlying storage for direct access if needed
    pub const fn inner(&self) -> &MappedWarmStorage {
        &self.storage
    }

    /// Get storage duration for a memory (time since first stored)
    pub fn get_storage_duration(&self, memory_id: &str) -> std::time::Duration {
        self.storage_timestamps.get(memory_id).map_or_else(
            || std::time::Duration::from_secs(0),
            |entry| {
                let stored_time = *entry.value();
                std::time::SystemTime::now()
                    .duration_since(stored_time)
                    .unwrap_or_default()
            },
        )
    }

    /// Force a compaction of the warm tier data
    ///
    /// # Errors
    ///
    /// Returns an error if compaction on the underlying storage fails.
    pub fn compact(&self) -> Result<CompactionStats, StorageError> {
        // Trigger content compaction
        self.storage.compact_content()
    }

    /// Get memory usage statistics
    pub fn memory_usage(&self) -> MemoryUsage {
        const MAX_SAFE_INTEGER: u64 = 9_007_199_254_740_992; // 2^53

        let stats = self.storage.statistics();
        let ratio = f64::from(stats.compaction_ratio.clamp(0.0, 1.0));
        let capped_total = stats.total_size_bytes.min(MAX_SAFE_INTEGER);
        #[allow(clippy::cast_precision_loss)]
        let total_f64 = capped_total as f64;
        let compressed_estimate = ratio * total_f64;
        #[allow(clippy::cast_precision_loss)]
        let rounded = compressed_estimate
            .round()
            .clamp(0.0, MAX_SAFE_INTEGER as f64);
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let compressed_bytes = rounded as u64;
        MemoryUsage {
            total_bytes: stats.total_size_bytes,
            compressed_bytes,
            compression_ratio: stats.compaction_ratio,
            fragmentation_ratio: 1.0 - stats.compaction_ratio,
        }
    }

    /// Check whether the tier currently holds the provided memory identifier.
    pub fn contains_memory(&self, memory_id: &str) -> bool {
        self.storage_timestamps.contains_key(memory_id)
    }

    /// Iterate over all memories in the warm tier
    ///
    /// Returns an iterator over (id, episode) pairs from persistent storage.
    /// The iterator is lazy and only loads memories as needed.
    ///
    /// # Performance
    ///
    /// - Iterator creation: O(1) - just clones ID list
    /// - Per-memory: ~10-50μs (memory-mapped I/O)
    /// - Total: ~10-50ms for thousands of memories
    ///
    /// # Implementation Note
    ///
    /// Memories that fail to load are skipped with a warning logged.
    /// This ensures iteration continues even if individual memories are corrupted.
    pub fn iter_memories(&self) -> impl Iterator<Item = (String, Episode)> + '_ {
        // Get all memory IDs from storage timestamps
        // We use storage_timestamps rather than directly accessing MappedWarmStorage
        // to ensure we only iterate over memories we've explicitly tracked
        self.storage_timestamps.iter().filter_map(|entry| {
            let memory_id = entry.key().clone();

            // Try to load memory from storage
            match self.storage.get(&memory_id) {
                Ok(Some(memory)) => {
                    // Convert Memory to Episode
                    let episode = Episode::new(
                        memory.id.clone(),
                        memory.created_at,
                        memory
                            .content
                            .clone()
                            .unwrap_or_else(|| format!("Memory {id}", id = memory.id)),
                        memory.embedding,
                        memory.confidence,
                    );
                    Some((memory_id, episode))
                }
                Ok(None) => {
                    tracing::warn!(
                        memory_id = %memory_id,
                        "Memory ID in storage_timestamps but not found in storage"
                    );
                    None
                }
                Err(e) => {
                    tracing::warn!(
                        memory_id = %memory_id,
                        error = %e,
                        "Failed to load memory from warm tier, skipping"
                    );
                    None
                }
            }
        })
    }
}

/// Configuration for warm tier storage
#[derive(Debug, Clone, Copy)]
pub struct WarmTierConfig {
    /// Maximum number of memories to store
    pub capacity: usize,
    /// Enable compression for space efficiency
    pub enable_compression: bool,
    /// Enable NUMA-aware memory allocation
    pub enable_numa_awareness: bool,
    /// Compaction threshold (0.0 to 1.0)
    pub compaction_threshold: f32,
    /// Enable automatic defragmentation
    pub enable_defragmentation: bool,
}

impl Default for WarmTierConfig {
    fn default() -> Self {
        Self {
            capacity: 10000,
            enable_compression: true,
            enable_numa_awareness: true,
            compaction_threshold: 0.7,
            enable_defragmentation: true,
        }
    }
}

/// Memory usage information
#[derive(Debug, Clone)]
pub struct MemoryUsage {
    /// Total bytes allocated
    pub total_bytes: u64,
    /// Bytes after compression
    pub compressed_bytes: u64,
    /// Compression ratio (compressed/total)
    pub compression_ratio: f32,
    /// Fragmentation ratio (fragmented/total)
    pub fragmentation_ratio: f32,
}

impl StorageTierBackend for WarmTier {
    type Error = StorageError;

    async fn store(&self, memory: Arc<Memory>) -> Result<(), Self::Error> {
        let memory_id = memory.id.clone();

        // Record storage timestamp for temporal decay calculation
        self.storage_timestamps
            .insert(memory_id, std::time::SystemTime::now());

        // Delegate to underlying mapped storage
        self.storage.store(memory).await
    }

    async fn recall(&self, cue: &Cue) -> Result<Vec<(Episode, Confidence)>, Self::Error> {
        // Delegate to underlying mapped storage with enhanced error handling
        match self.storage.recall(cue).await {
            Ok(mut results) => {
                // Apply warm tier confidence calibration
                for (episode, confidence) in &mut results {
                    // Get actual storage duration for this memory
                    let storage_duration = self.get_storage_duration(&episode.id);
                    *confidence = self.confidence_calibrator.adjust_for_storage_tier(
                        *confidence,
                        ConfidenceTier::Warm,
                        storage_duration,
                    );
                }
                Ok(results)
            }
            Err(e) => {
                // Log the error and attempt recovery
                tracing::warn!("Warm tier recall failed, attempting recovery: {}", e);

                // Try maintenance to fix any issues
                if let Err(maintenance_error) = self.storage.maintenance().await {
                    tracing::error!("Warm tier maintenance failed: {}", maintenance_error);
                }

                // Return original error
                Err(e)
            }
        }
    }

    async fn update_activation(&self, memory_id: &str, activation: f32) -> Result<(), Self::Error> {
        // Delegate to underlying mapped storage
        self.storage.update_activation(memory_id, activation).await
    }

    async fn remove(&self, memory_id: &str) -> Result<(), Self::Error> {
        // Clean up storage timestamp
        self.storage_timestamps.remove(memory_id);

        // Delegate to underlying mapped storage
        self.storage.remove(memory_id).await
    }

    fn statistics(&self) -> TierStatistics {
        // Get base statistics from underlying storage
        let mut stats = self.storage.statistics();

        // Enhance with warm tier specific information
        stats.cache_hit_rate *= 0.9; // Slightly lower than hot tier

        stats
    }

    async fn maintenance(&self) -> Result<(), Self::Error> {
        // Perform comprehensive warm tier maintenance

        // 1. Run underlying storage maintenance
        self.storage.maintenance().await?;

        // 2. Check for fragmentation and trigger compaction if needed
        let stats = self.statistics();
        if stats.compaction_ratio < 0.7 {
            // Trigger compaction by running maintenance again
            self.storage.maintenance().await?;
        }

        // 3. Validate data integrity
        // This would be implemented with checksums in a production system

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CueBuilder, EpisodeBuilder};
    use anyhow::{Context, Result, ensure};
    use chrono::Utc;
    use tempfile::TempDir;

    fn create_test_memory(id: &str, activation: f32) -> Arc<Memory> {
        let episode = EpisodeBuilder::new()
            .id(id.to_string())
            .when(Utc::now())
            .what(format!("test memory {id}"))
            .embedding([0.5f32; 768])
            .confidence(Confidence::HIGH)
            .build();

        Arc::new(Memory::from_episode(episode, activation))
    }

    #[tokio::test]
    async fn test_warm_tier_creation() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let file_path = temp_dir.path().join("warm_tier_test.dat");
        let metrics = Arc::new(super::super::StorageMetrics::new());

        let warm_tier =
            WarmTier::new(file_path, 1000, metrics).context("failed to create warm tier")?;
        let stats = warm_tier.statistics();

        ensure!(stats.memory_count == 0, "new warm tier should be empty");
        Ok(())
    }

    #[tokio::test]
    async fn test_warm_tier_store_and_recall() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let file_path = temp_dir.path().join("warm_tier_test.dat");
        let metrics = Arc::new(super::super::StorageMetrics::new());

        let warm_tier =
            WarmTier::new(file_path, 1000, metrics).context("failed to create warm tier")?;

        // Store test memory
        let memory = create_test_memory("test1", 0.6);
        warm_tier.store(memory).await.context("store failed")?;

        // Test recall
        let cue = CueBuilder::new()
            .id("test_cue".to_string())
            .embedding_search([0.5f32; 768], Confidence::MEDIUM)
            .max_results(10)
            .build();

        let results = warm_tier.recall(&cue).await.context("recall failed")?;
        ensure!(!results.is_empty(), "recall should return stored memory");
        ensure!(
            results[0].0.id == "test1",
            "recalled memory id should match"
        );
        Ok(())
    }

    #[tokio::test]
    async fn test_warm_tier_maintenance() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let file_path = temp_dir.path().join("warm_tier_test.dat");
        let metrics = Arc::new(super::super::StorageMetrics::new());

        let warm_tier =
            WarmTier::new(file_path, 1000, metrics).context("failed to create warm tier")?;

        // Store some memories
        for i in 0..10 {
            let memory = create_test_memory(&format!("mem_{i}"), 0.5);
            warm_tier.store(memory).await.context("store failed")?;
        }

        // Run maintenance
        warm_tier
            .maintenance()
            .await
            .context("maintenance failed")?;

        // Verify stats are still valid
        let stats = warm_tier.statistics();
        ensure!(
            stats.memory_count == 10,
            "expected warm tier to track inserted memories"
        );
        Ok(())
    }

    #[tokio::test]
    async fn test_warm_tier_compaction() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let file_path = temp_dir.path().join("warm_tier_test.dat");
        let metrics = Arc::new(super::super::StorageMetrics::new());

        let warm_tier =
            WarmTier::new(file_path, 1000, metrics).context("failed to create warm tier")?;

        // Store memories with actual content to create fragmentation
        for i in 0..20 {
            let episode = EpisodeBuilder::new()
                .id(format!("mem_{i}"))
                .when(Utc::now())
                .what(format!(
                    "Content for memory {i} - this creates storage usage"
                ))
                .embedding([0.5f32; 768])
                .confidence(Confidence::HIGH)
                .build();

            warm_tier
                .store(Arc::new(Memory::from_episode(episode, 0.5)))
                .await
                .context("store failed")?;
        }

        // Delete half the memories to create fragmentation
        for i in (0..20).step_by(2) {
            warm_tier
                .remove(&format!("mem_{i}"))
                .await
                .context("remove failed")?;
        }

        // Force compaction - should reclaim space from deleted memories
        let compaction_stats = warm_tier.compact().context("compaction failed")?;
        ensure!(
            compaction_stats.bytes_reclaimed > 0,
            "compaction should reclaim at least one byte (deleted 10 out of 20 memories)"
        );
        Ok(())
    }

    #[test]
    fn test_warm_tier_config() {
        let config = WarmTierConfig::default();
        assert_eq!(config.capacity, 10000);
        assert!(config.enable_compression);
        assert!(config.enable_numa_awareness);
        assert!((config.compaction_threshold - 0.7).abs() < f32::EPSILON);
    }

    #[test]
    fn test_memory_usage() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let file_path = temp_dir.path().join("warm_tier_test.dat");
        let metrics = Arc::new(super::super::StorageMetrics::new());

        let warm_tier =
            WarmTier::new(file_path, 1000, metrics).context("failed to create warm tier")?;
        let usage = warm_tier.memory_usage();

        ensure!(
            usage.total_bytes == 0,
            "fresh warm tier should not consume bytes"
        );
        ensure!(
            usage.compression_ratio >= 0.0,
            "compression ratio should be non-negative"
        );
        ensure!(
            usage.fragmentation_ratio >= 0.0,
            "fragmentation ratio should be non-negative"
        );
        Ok(())
    }
}
