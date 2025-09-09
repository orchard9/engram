//! Multi-tier storage coordinator with cognitive eviction policies
//!
//! This module orchestrates data movement between hot, warm, and cold storage tiers
//! based on cognitive memory patterns and access frequency.

use super::{
    CognitiveEvictionPolicy, StorageError, StorageMetrics, StorageResult, StorageTier,
    TierStatistics, mapped::MappedWarmStorage,
};
use crate::{Confidence, Cue, Episode, Memory};
use crossbeam_queue::SegQueue;
use dashmap::DashMap;
use std::path::Path;
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};
use std::time::{Duration, SystemTime};
use tokio::sync::RwLock as AsyncRwLock;

/// Migration task for moving memories between tiers
#[derive(Debug, Clone)]
pub struct MigrationTask {
    /// Unique identifier of the memory to migrate
    pub memory_id: String,
    /// Current tier where the memory is stored
    pub from_tier: TierType,
    /// Destination tier for the memory
    pub to_tier: TierType,
    /// Priority level for this migration task
    pub priority: MigrationPriority,
    /// When this migration task was created
    pub created_at: SystemTime,
}

/// Types of storage tiers
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TierType {
    /// In-memory DashMap for active memories
    Hot,  // In-memory DashMap for active memories
    /// Memory-mapped files for recent memories
    Warm, // Memory-mapped files for recent memories
    /// Columnar storage for archived memories
    Cold, // Columnar storage for archived memories
}

/// Priority levels for migration tasks
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum MigrationPriority {
    /// Memory pressure - move now
    Immediate = 0, // Memory pressure - move now
    /// Activation threshold crossed
    High = 1,      // Activation threshold crossed
    /// Regular maintenance
    Normal = 2,    // Regular maintenance
    /// Background optimization
    Low = 3,
}

/// Cognitive tier architecture with automatic migration
pub struct CognitiveTierArchitecture {
    /// Hot tier: DashMap for immediate access
    hot_tier: Arc<DashMap<String, Arc<Memory>>>,

    /// Warm tier: Memory-mapped storage
    #[cfg(feature = "memory_mapped_persistence")]
    warm_tier: Arc<MappedWarmStorage>,

    /// Cold tier: Placeholder for columnar storage
    cold_tier: Arc<DummyColdStorage>,

    /// Tier coordinator for migration
    coordinator: Arc<TierCoordinator>,

    /// Performance metrics
    metrics: Arc<StorageMetrics>,

    /// Configuration
    eviction_policy: CognitiveEvictionPolicy,

    /// Capacity limits per tier
    hot_capacity: usize,
    warm_capacity: usize,
    cold_capacity: usize,
}

impl CognitiveTierArchitecture {
    /// Create new tiered architecture
    pub fn new<P: AsRef<Path>>(
        data_dir: P,
        hot_capacity: usize,
        warm_capacity: usize,
        cold_capacity: usize,
        metrics: Arc<StorageMetrics>,
    ) -> StorageResult<Self> {
        let data_dir = data_dir.as_ref();

        // Create warm tier storage
        #[cfg(feature = "memory_mapped_persistence")]
        let warm_tier = Arc::new(MappedWarmStorage::new(
            data_dir.join("warm_tier.dat"),
            warm_capacity,
            metrics.clone(),
        )?);

        // Create tier coordinator
        let coordinator = Arc::new(TierCoordinator::new(
            CognitiveEvictionPolicy::default(),
            metrics.clone(),
        ));

        Ok(Self {
            hot_tier: Arc::new(DashMap::with_capacity(hot_capacity)),
            #[cfg(feature = "memory_mapped_persistence")]
            warm_tier,
            cold_tier: Arc::new(DummyColdStorage::new()),
            coordinator,
            metrics,
            eviction_policy: CognitiveEvictionPolicy::default(),
            hot_capacity,
            warm_capacity,
            cold_capacity,
        })
    }

    /// Store a memory in the appropriate tier
    pub async fn store(&self, memory: Arc<Memory>) -> StorageResult<()> {
        let activation = memory.activation();

        // Determine initial tier based on activation level
        if activation >= self.eviction_policy.hot_activation_threshold {
            // Store in hot tier
            self.hot_tier.insert(memory.id.clone(), memory.clone());

            // Check if hot tier is approaching capacity
            if self.hot_tier.len() > (self.hot_capacity as f32 * 0.8) as usize {
                self.coordinator
                    .schedule_migration(MigrationTask {
                        memory_id: memory.id.clone(),
                        from_tier: TierType::Hot,
                        to_tier: TierType::Warm,
                        priority: MigrationPriority::High,
                        created_at: SystemTime::now(),
                    })
                    .await;
            }
        } else {
            // Store directly in warm tier
            #[cfg(feature = "memory_mapped_persistence")]
            self.warm_tier.store(memory.clone()).await?;
        }

        self.metrics
            .record_write(std::mem::size_of::<Memory>() as u64);
        Ok(())
    }

    /// Recall memories from all tiers
    pub async fn recall(&self, cue: &Cue) -> StorageResult<Vec<(Episode, Confidence)>> {
        let mut results = Vec::new();

        // Search hot tier first (fastest)
        for entry in self.hot_tier.iter() {
            let memory = entry.value();
            if self.memory_matches_cue(memory, cue) {
                let episode = self.memory_to_episode(memory);
                results.push((episode, memory.confidence));
            }
        }

        // If we need more results, search warm tier
        if results.len() < cue.max_results {
            #[cfg(feature = "memory_mapped_persistence")]
            {
                let warm_results = self.warm_tier.recall(cue).await.unwrap_or_else(|e| {
                    tracing::warn!("Warm tier recall failed: {}", e);
                    Vec::new()
                });
                results.extend(warm_results);
            }
        }

        // If still need more, search cold tier
        if results.len() < cue.max_results {
            let cold_results = self.cold_tier.recall(cue).await.unwrap_or_else(|e| {
                tracing::warn!("Cold tier recall failed: {}", e);
                Vec::new()
            });
            results.extend(cold_results);
        }

        // Sort by confidence and limit results
        results.sort_by(|a, b| {
            b.1.raw()
                .partial_cmp(&a.1.raw())
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        results.truncate(cue.max_results);

        self.metrics
            .record_read(results.len() as u64 * std::mem::size_of::<Episode>() as u64);
        Ok(results)
    }

    /// Update activation level across tiers
    pub async fn update_activation(&self, memory_id: &str, activation: f32) -> StorageResult<()> {
        // Try hot tier first
        if let Some(mut memory_ref) = self.hot_tier.get_mut(memory_id) {
            let memory = Arc::make_mut(memory_ref.value_mut());
            memory.set_activation(activation);
            return Ok(());
        }

        // Try warm tier
        #[cfg(feature = "memory_mapped_persistence")]
        {
            if let Err(_) = self
                .warm_tier
                .update_activation(memory_id, activation)
                .await
            {
                // Try cold tier
                self.cold_tier
                    .update_activation(memory_id, activation)
                    .await?;
            }
        }

        Ok(())
    }

    /// Remove memory from all tiers
    pub async fn remove(&self, memory_id: &str) -> StorageResult<bool> {
        let mut found = false;

        // Remove from hot tier
        if self.hot_tier.remove(memory_id).is_some() {
            found = true;
        }

        // Remove from warm tier
        #[cfg(feature = "memory_mapped_persistence")]
        {
            if self.warm_tier.remove(memory_id).await.is_ok() {
                found = true;
            }
        }

        // Remove from cold tier
        if self.cold_tier.remove(memory_id).await.is_ok() {
            found = true;
        }

        Ok(found)
    }

    /// Get statistics for all tiers
    pub fn get_tier_statistics(&self) -> TierArchitectureStats {
        let hot_stats = TierStatistics {
            memory_count: self.hot_tier.len(),
            total_size_bytes: self.hot_tier.len() as u64 * std::mem::size_of::<Memory>() as u64,
            average_activation: self.calculate_hot_tier_avg_activation(),
            last_access_time: SystemTime::now(),
            cache_hit_rate: 1.0,   // Hot tier is always a hit
            compaction_ratio: 1.0, // No compaction needed
        };

        #[cfg(feature = "memory_mapped_persistence")]
        let warm_stats = self.warm_tier.statistics();
        #[cfg(not(feature = "memory_mapped_persistence"))]
        let warm_stats = TierStatistics {
            memory_count: 0,
            total_size_bytes: 0,
            average_activation: 0.0,
            last_access_time: SystemTime::UNIX_EPOCH,
            cache_hit_rate: 0.0,
            compaction_ratio: 0.0,
        };

        let cold_stats = self.cold_tier.statistics();

        TierArchitectureStats {
            hot: hot_stats,
            warm: warm_stats,
            cold: cold_stats,
        }
    }

    /// Trigger background maintenance across all tiers
    pub async fn maintenance(&self) -> StorageResult<()> {
        // Perform hot tier eviction if needed
        if self.hot_tier.len() > self.hot_capacity {
            self.evict_from_hot_tier().await?;
        }

        // Perform warm tier maintenance
        #[cfg(feature = "memory_mapped_persistence")]
        self.warm_tier.maintenance().await?;

        // Perform cold tier maintenance
        self.cold_tier.maintenance().await?;

        Ok(())
    }

    /// Evict memories from hot tier based on cognitive policy
    async fn evict_from_hot_tier(&self) -> StorageResult<()> {
        let current_time = SystemTime::now();
        let mut candidates = Vec::new();

        // Collect eviction candidates
        for entry in self.hot_tier.iter() {
            let memory = entry.value();
            let age = current_time
                .duration_since(memory.created_at.into())
                .unwrap_or_default();

            if memory.activation() < self.eviction_policy.hot_activation_threshold
                || age > self.eviction_policy.warm_access_window
            {
                candidates.push((entry.key().clone(), memory.clone(), memory.activation()));
            }
        }

        // Sort by activation (evict lowest first)
        candidates.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal));

        // Evict until we're under capacity
        let target_count = (self.hot_capacity as f32 * 0.7) as usize; // Target 70% capacity
        let evict_count = self.hot_tier.len().saturating_sub(target_count);

        for (memory_id, memory, _) in candidates.into_iter().take(evict_count) {
            // Move to warm tier
            #[cfg(feature = "memory_mapped_persistence")]
            {
                if let Err(e) = self.warm_tier.store(memory).await {
                    tracing::warn!("Failed to migrate {} to warm tier: {}", memory_id, e);
                    continue;
                }
            }

            // Remove from hot tier
            self.hot_tier.remove(&memory_id);
        }

        Ok(())
    }

    /// Calculate average activation in hot tier
    fn calculate_hot_tier_avg_activation(&self) -> f32 {
        if self.hot_tier.is_empty() {
            return 0.0;
        }

        let total: f32 = self
            .hot_tier
            .iter()
            .map(|entry| entry.value().activation())
            .sum();

        total / self.hot_tier.len() as f32
    }

    /// Check if memory matches the given cue
    fn memory_matches_cue(&self, memory: &Memory, cue: &Cue) -> bool {
        // Simple implementation - in practice would use sophisticated matching
        match &cue.cue_type {
            crate::CueType::Semantic { content, .. } => {
                memory.confidence.raw() >= cue.result_threshold.raw() && memory.id.contains(content) // Simplified matching
            }
            crate::CueType::Embedding { vector, .. } => {
                memory.confidence.raw() >= cue.result_threshold.raw()
                    && self.cosine_similarity(&memory.embedding, vector) > 0.8
            }
            crate::CueType::Context { .. } => {
                // TODO: Implement context-based matching
                memory.confidence.raw() >= cue.result_threshold.raw()
            }
            crate::CueType::Temporal { .. } => {
                // TODO: Implement temporal matching
                memory.confidence.raw() >= cue.result_threshold.raw()
            }
        }
    }

    /// Optimized cosine similarity using SIMD when available
    fn cosine_similarity(&self, a: &[f32; 768], b: &[f32; 768]) -> f32 {
        crate::compute::cosine_similarity_768(a, b)
    }

    /// Convert Memory to Episode
    fn memory_to_episode(&self, memory: &Memory) -> Episode {
        crate::EpisodeBuilder::new()
            .id(memory.id.clone())
            .when(chrono::DateTime::from_timestamp_nanos(
                SystemTime::from(memory.created_at)
                    .duration_since(SystemTime::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_nanos() as i64,
            ))
            .what(format!("Memory: {}", memory.id))
            .embedding(memory.embedding)
            .confidence(memory.confidence)
            .build()
    }
}

/// Statistics for all tiers
#[derive(Debug, Clone)]
pub struct TierArchitectureStats {
    /// Statistics for the hot tier (in-memory)
    pub hot: TierStatistics,
    /// Statistics for the warm tier (memory-mapped)
    pub warm: TierStatistics,
    /// Statistics for the cold tier (archived)
    pub cold: TierStatistics,
}

impl TierArchitectureStats {
    /// Calculate total number of memories across all tiers
    pub fn total_memories(&self) -> usize {
        self.hot.memory_count + self.warm.memory_count + self.cold.memory_count
    }

    /// Calculate total size in bytes across all tiers
    pub fn total_size_bytes(&self) -> u64 {
        self.hot.total_size_bytes + self.warm.total_size_bytes + self.cold.total_size_bytes
    }
}

/// Tier coordinator for managing migrations
pub struct TierCoordinator {
    migration_queue: Arc<SegQueue<MigrationTask>>,
    policy: CognitiveEvictionPolicy,
    metrics: Arc<StorageMetrics>,
    active_migrations: Arc<AtomicUsize>,
    worker_handle: AsyncRwLock<Option<tokio::task::JoinHandle<()>>>,
}

impl TierCoordinator {
    /// Create a new tier coordinator with the given policy and metrics
    pub fn new(policy: CognitiveEvictionPolicy, metrics: Arc<StorageMetrics>) -> Self {
        Self {
            migration_queue: Arc::new(SegQueue::new()),
            policy,
            metrics,
            active_migrations: Arc::new(AtomicUsize::new(0)),
            worker_handle: AsyncRwLock::new(None),
        }
    }

    /// Schedule a migration task
    pub async fn schedule_migration(&self, task: MigrationTask) {
        self.migration_queue.push(task);
    }

    /// Start background migration worker
    pub async fn start(&self) {
        let queue = self.migration_queue.clone();
        let metrics = Arc::clone(&self.metrics);
        let active_migrations = self.active_migrations.clone();

        let handle = tokio::spawn(async move {
            Self::migration_worker(queue, metrics, active_migrations).await;
        });

        *self.worker_handle.write().await = Some(handle);
    }

    /// Background migration worker
    async fn migration_worker(
        queue: Arc<SegQueue<MigrationTask>>,
        _metrics: Arc<StorageMetrics>,
        active_migrations: Arc<AtomicUsize>,
    ) {
        while let Some(task) = queue.pop() {
            active_migrations.fetch_add(1, Ordering::Relaxed);

            // Process migration task
            if let Err(e) = Self::process_migration(task).await {
                tracing::warn!("Migration failed: {}", e);
            }

            active_migrations.fetch_sub(1, Ordering::Relaxed);

            // Small delay to prevent overwhelming the system
            tokio::time::sleep(Duration::from_millis(1)).await;
        }
    }

    /// Process a single migration task
    async fn process_migration(task: MigrationTask) -> StorageResult<()> {
        tracing::debug!(
            "Migrating {} from {:?} to {:?}",
            task.memory_id,
            task.from_tier,
            task.to_tier
        );

        // Actual migration logic would go here
        // For now, just simulate the work
        tokio::time::sleep(Duration::from_millis(1)).await;

        Ok(())
    }
}

/// Dummy cold storage implementation for testing
pub struct DummyColdStorage {
    memories: DashMap<String, Arc<Memory>>,
}

impl DummyColdStorage {
    /// Create a new dummy cold storage implementation
    pub fn new() -> Self {
        Self {
            memories: DashMap::new(),
        }
    }
}

impl StorageTier for DummyColdStorage {
    type Error = StorageError;

    async fn store(&self, memory: Arc<Memory>) -> Result<(), Self::Error> {
        self.memories.insert(memory.id.clone(), memory);
        Ok(())
    }

    async fn recall(&self, cue: &Cue) -> Result<Vec<(Episode, Confidence)>, Self::Error> {
        let mut results = Vec::new();

        for entry in self.memories.iter() {
            let memory = entry.value();
            if memory.confidence.raw() >= cue.result_threshold.raw() {
                let episode = crate::EpisodeBuilder::new()
                    .id(memory.id.clone())
                    .when(chrono::DateTime::from_timestamp_nanos(
                        SystemTime::from(memory.created_at)
                            .duration_since(SystemTime::UNIX_EPOCH)
                            .unwrap_or_default()
                            .as_nanos() as i64,
                    ))
                    .what(format!("Cold memory: {}", memory.id))
                    .embedding(memory.embedding)
                    .confidence(memory.confidence)
                    .build();

                results.push((episode, memory.confidence));
            }
        }

        Ok(results)
    }

    async fn update_activation(&self, memory_id: &str, activation: f32) -> Result<(), Self::Error> {
        if let Some(mut memory_ref) = self.memories.get_mut(memory_id) {
            let memory = Arc::make_mut(memory_ref.value_mut());
            memory.set_activation(activation);
        }
        Ok(())
    }

    async fn remove(&self, memory_id: &str) -> Result<(), Self::Error> {
        self.memories.remove(memory_id);
        Ok(())
    }

    fn statistics(&self) -> TierStatistics {
        TierStatistics {
            memory_count: self.memories.len(),
            total_size_bytes: self.memories.len() as u64 * std::mem::size_of::<Memory>() as u64,
            average_activation: 0.3, // Cold memories have low activation
            last_access_time: SystemTime::now(),
            cache_hit_rate: 0.5,   // Medium hit rate for cold storage
            compaction_ratio: 0.9, // Good compaction in cold storage
        }
    }

    async fn maintenance(&self) -> Result<(), Self::Error> {
        // Perform cold storage maintenance like compaction
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::EpisodeBuilder;
    use chrono::Utc;
    use tempfile::TempDir;

    fn create_test_memory(id: &str, activation: f32) -> Arc<Memory> {
        let episode = EpisodeBuilder::new()
            .id(id.to_string())
            .when(Utc::now())
            .what(format!("test memory {}", id))
            .embedding([0.5f32; 768])
            .confidence(Confidence::HIGH)
            .build();

        let memory = Memory::from_episode(episode, activation);
        Arc::new(memory)
    }

    #[tokio::test]
    async fn test_tier_architecture_storage() {
        let temp_dir = TempDir::new().unwrap();
        let metrics = Arc::new(StorageMetrics::new());

        let architecture = CognitiveTierArchitecture::new(
            temp_dir.path(),
            100,   // hot capacity
            1000,  // warm capacity
            10000, // cold capacity
            metrics,
        )
        .unwrap();

        // Test storing high-activation memory (should go to hot tier)
        let hot_memory = create_test_memory("hot_memory", 0.9);
        architecture.store(hot_memory.clone()).await.unwrap();

        // Test storing low-activation memory (should go to warm tier)
        let warm_memory = create_test_memory("warm_memory", 0.3);
        architecture.store(warm_memory.clone()).await.unwrap();

        let stats = architecture.get_tier_statistics();
        assert!(stats.hot.memory_count > 0 || stats.warm.memory_count > 0);
    }

    #[tokio::test]
    async fn test_tier_coordinator() {
        let metrics = Arc::new(StorageMetrics::new());
        let coordinator = TierCoordinator::new(CognitiveEvictionPolicy::default(), metrics);

        coordinator.start().await;

        let task = MigrationTask {
            memory_id: "test_memory".to_string(),
            from_tier: TierType::Hot,
            to_tier: TierType::Warm,
            priority: MigrationPriority::Normal,
            created_at: SystemTime::now(),
        };

        coordinator.schedule_migration(task).await;

        // Give worker time to process
        tokio::time::sleep(Duration::from_millis(10)).await;
    }

    #[test]
    fn test_tier_statistics() {
        let stats = TierArchitectureStats {
            hot: TierStatistics {
                memory_count: 100,
                total_size_bytes: 1024,
                average_activation: 0.8,
                last_access_time: SystemTime::now(),
                cache_hit_rate: 0.95,
                compaction_ratio: 1.0,
            },
            warm: TierStatistics {
                memory_count: 500,
                total_size_bytes: 5120,
                average_activation: 0.5,
                last_access_time: SystemTime::now(),
                cache_hit_rate: 0.8,
                compaction_ratio: 0.9,
            },
            cold: TierStatistics {
                memory_count: 1000,
                total_size_bytes: 10240,
                average_activation: 0.2,
                last_access_time: SystemTime::now(),
                cache_hit_rate: 0.6,
                compaction_ratio: 0.95,
            },
        };

        assert_eq!(stats.total_memories(), 1600);
        assert_eq!(stats.total_size_bytes(), 16384);
    }
}
