//! Multi-tier storage coordinator with cognitive eviction policies
//!
//! This module orchestrates data movement between hot, warm, and cold storage tiers
//! based on cognitive memory patterns and access frequency.

use super::{
    access_tracking::{AccessTracker, AccessPredictor},
    CognitiveEvictionPolicy, StorageMetrics, StorageResult, StorageTier, StorageError,
    TierStatistics, HotTier, WarmTier, ColdTier,
};
use crate::{Confidence, Cue, Episode, Memory};
use crossbeam_queue::SegQueue;
use std::path::Path;
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};
use std::time::{Duration, SystemTime};
use tokio::sync::{RwLock as AsyncRwLock, Semaphore};

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

/// Memory pressure levels for emergency eviction
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub enum MemoryPressure {
    /// System is healthy
    Normal = 0,
    /// Moderate pressure, accelerate migrations
    Moderate = 1,
    /// High pressure, aggressive migrations
    High = 2,
    /// Critical pressure, emergency eviction
    Critical = 3,
}

/// Cognitive tier architecture with automatic migration
pub struct CognitiveTierArchitecture {
    /// Hot tier: DashMap for immediate access
    hot_tier: Arc<HotTier>,

    /// Warm tier: Memory-mapped storage
    warm_tier: Arc<WarmTier>,

    /// Cold tier: Columnar storage for archived memories
    cold_tier: Arc<ColdTier>,

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

        // Create tier instances
        let hot_tier = Arc::new(HotTier::new(hot_capacity));
        
        let warm_tier = Arc::new(WarmTier::new(
            data_dir.join("warm_tier.dat"),
            warm_capacity,
            metrics.clone(),
        )?);
        
        let cold_tier = Arc::new(ColdTier::new(cold_capacity));

        // Create tier coordinator
        let mut coordinator = TierCoordinator::new(
            CognitiveEvictionPolicy::default(),
            metrics.clone(),
        );

        // Set tier references for migration
        coordinator.set_tiers(
            hot_tier.clone(),
            warm_tier.clone(),
            cold_tier.clone(),
        );

        let coordinator = Arc::new(coordinator);

        Ok(Self {
            hot_tier,
            warm_tier,
            cold_tier,
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
            self.hot_tier.store(memory.clone()).await?;

            // Check if hot tier is approaching capacity
            if self.hot_tier.is_near_capacity() {
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
        let hot_results = self.hot_tier.recall(cue).await.unwrap_or_else(|e| {
            tracing::warn!("Hot tier recall failed: {}", e);
            Vec::new()
        });
        results.extend(hot_results);

        // If we need more results, search warm tier
        if results.len() < cue.max_results {
            let warm_results = self.warm_tier.recall(cue).await.unwrap_or_else(|e| {
                tracing::warn!("Warm tier recall failed: {}", e);
                Vec::new()
            });
            results.extend(warm_results);
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
        if let Ok(()) = self.hot_tier.update_activation(memory_id, activation).await {
            return Ok(());
        }

        // Try warm tier
        if let Ok(()) = self.warm_tier.update_activation(memory_id, activation).await {
            return Ok(());
        }

        // Try cold tier
        self.cold_tier.update_activation(memory_id, activation).await
    }

    /// Remove memory from all tiers
    pub async fn remove(&self, memory_id: &str) -> StorageResult<bool> {
        let mut found = false;

        // Remove from hot tier
        if self.hot_tier.remove(memory_id).await.is_ok() {
            found = true;
        }

        // Remove from warm tier
        if self.warm_tier.remove(memory_id).await.is_ok() {
            found = true;
        }

        // Remove from cold tier
        if self.cold_tier.remove(memory_id).await.is_ok() {
            found = true;
        }

        Ok(found)
    }

    /// Get statistics for all tiers
    pub fn get_tier_statistics(&self) -> TierArchitectureStats {
        let hot_stats = self.hot_tier.statistics();
        let warm_stats = self.warm_tier.statistics();

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
        // Check if eviction is needed
        if !self.hot_tier.is_near_capacity() {
            return Ok(());
        }

        // Get LRU candidates for eviction
        let target_count = (self.hot_capacity as f32 * 0.7) as usize; // Target 70% capacity
        let current_count = self.hot_tier.len();
        let evict_count = current_count.saturating_sub(target_count);

        if evict_count == 0 {
            return Ok(());
        }

        let candidates = self.hot_tier.get_lru_candidates(evict_count);

        for memory_id in candidates {
            // Get memory for migration
            if let Some(memory) = self.hot_tier.get_memory(&memory_id) {
                // Move to warm tier
                if let Err(e) = self.warm_tier.store(memory).await {
                    tracing::warn!("Failed to migrate {} to warm tier: {}", memory_id, e);
                    continue;
                }

                // Remove from hot tier
                if let Err(e) = self.hot_tier.remove(&memory_id).await {
                    tracing::warn!("Failed to remove {} from hot tier: {}", memory_id, e);
                }
            }
        }

        Ok(())
    }

}

/// Migration candidate for tier movement
#[derive(Debug, Clone)]
pub struct MigrationCandidate {
    /// Memory identifier
    pub memory_id: String,
    /// Current tier
    pub source_tier: TierType,
    /// Target tier for migration
    pub target_tier: TierType,
    /// Migration priority score (higher = more urgent)
    pub priority: f32,
    /// Current activation level
    pub activation: f32,
    /// Idle time since last access
    pub idle_time: Duration,
}

/// Report of completed migration cycle
#[derive(Debug, Default, Clone)]
pub struct MigrationReport {
    /// Memories moved from hot to warm
    pub hot_to_warm: usize,
    /// Memories moved from warm to cold
    pub warm_to_cold: usize,
    /// Memories moved from warm to hot
    pub warm_to_hot: usize,
    /// Memories moved from cold to warm
    pub cold_to_warm: usize,
    /// Total migrations performed
    pub total_migrations: usize,
    /// Duration of migration cycle
    pub duration: Duration,
    /// Errors encountered during migration
    pub errors: Vec<String>,
}

impl MigrationReport {
    /// Calculate total number of successful migrations
    pub fn successful_migrations(&self) -> usize {
        self.hot_to_warm + self.warm_to_cold + self.warm_to_hot + self.cold_to_warm
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
    /// Queue of pending migration tasks
    migration_queue: Arc<SegQueue<MigrationTask>>,
    /// Eviction and migration policy
    policy: CognitiveEvictionPolicy,
    /// Performance metrics
    metrics: Arc<StorageMetrics>,
    /// Count of active migrations
    active_migrations: Arc<AtomicUsize>,
    /// Background worker handle
    worker_handle: AsyncRwLock<Option<tokio::task::JoinHandle<()>>>,
    /// Access tracking for migration decisions
    access_tracker: Arc<AccessTracker>,
    /// Access pattern predictor for proactive migration
    access_predictor: Arc<AccessPredictor>,
    /// Rate limiter for migrations (max 100/second)
    rate_limiter: Arc<Semaphore>,
    /// References to storage tiers
    hot_tier: Option<Arc<HotTier>>,
    warm_tier: Option<Arc<WarmTier>>,
    cold_tier: Option<Arc<ColdTier>>,
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
            access_tracker: Arc::new(AccessTracker::new()),
            access_predictor: Arc::new(AccessPredictor::new()),
            rate_limiter: Arc::new(Semaphore::new(100)), // Max 100 concurrent migrations
            hot_tier: None,
            warm_tier: None,
            cold_tier: None,
        }
    }

    /// Set tier references for migration operations
    pub fn set_tiers(
        &mut self,
        hot: Arc<HotTier>,
        warm: Arc<WarmTier>,
        cold: Arc<ColdTier>,
    ) {
        self.hot_tier = Some(hot);
        self.warm_tier = Some(warm);
        self.cold_tier = Some(cold);
    }

    /// Calculate memory pressure for a tier
    pub fn calculate_memory_pressure(&self, tier: &HotTier) -> MemoryPressure {
        let usage_ratio = tier.len() as f32 / tier.max_capacity as f32;

        if usage_ratio > 0.95 {
            MemoryPressure::Critical
        } else if usage_ratio > 0.85 {
            MemoryPressure::High
        } else if usage_ratio > 0.70 {
            MemoryPressure::Moderate
        } else {
            MemoryPressure::Normal
        }
    }

    /// Handle memory pressure with emergency eviction
    pub async fn handle_memory_pressure(&self) -> Result<usize, StorageError> {
        let hot_tier = self.hot_tier.as_ref()
            .ok_or_else(|| StorageError::NotInitialized("Hot tier not set".to_string()))?;

        let pressure = self.calculate_memory_pressure(hot_tier);
        let mut evicted = 0;

        match pressure {
            MemoryPressure::Critical => {
                // Emergency eviction: remove 20% of memories immediately
                let target = (hot_tier.len() as f32 * 0.2) as usize;
                let candidates = hot_tier.get_lru_candidates(target);

                for memory_id in candidates {
                    if let Some(memory) = hot_tier.evict_memory(&memory_id) {
                        // Try to move to warm tier, drop if that fails
                        if let Some(warm) = &self.warm_tier {
                            let _ = warm.store(memory).await;
                        }
                        evicted += 1;
                    }
                }
            }
            MemoryPressure::High => {
                // Accelerated migration: move 10% to warm tier
                let target = (hot_tier.len() as f32 * 0.1) as usize;
                let candidates = hot_tier.get_lru_candidates(target);

                for memory_id in candidates {
                    if let Some(memory) = hot_tier.get_memory(&memory_id) {
                        if let Some(warm) = &self.warm_tier {
                            if warm.store(memory).await.is_ok() {
                                hot_tier.evict_memory(&memory_id);
                                evicted += 1;
                            }
                        }
                    }
                }
            }
            MemoryPressure::Moderate => {
                // Schedule normal migration
                let task = MigrationTask {
                    memory_id: "batch_migration".to_string(),
                    from_tier: TierType::Hot,
                    to_tier: TierType::Warm,
                    priority: MigrationPriority::High,
                    created_at: SystemTime::now(),
                };
                self.schedule_migration(task).await;
            }
            MemoryPressure::Normal => {
                // No action needed
            }
        }

        Ok(evicted)
    }

    /// Run a complete migration cycle
    pub async fn run_migration_cycle(&self) -> Result<MigrationReport, StorageError> {
        let start_time = SystemTime::now();
        let mut report = MigrationReport::default();

        // Check if tiers are configured
        if self.hot_tier.is_none() || self.warm_tier.is_none() || self.cold_tier.is_none() {
            return Err(StorageError::NotInitialized("Tiers not configured".to_string()));
        }

        // Check and handle memory pressure first
        if let Ok(evicted) = self.handle_memory_pressure().await {
            if evicted > 0 {
                report.hot_to_warm += evicted;
            }
        }

        // Evaluate hot tier for demotion
        match self.evaluate_hot_tier_migrations().await {
            Ok(candidates) => {
                for candidate in candidates {
                    // Apply rate limiting
                    let _permit = self.rate_limiter.acquire().await
                        .map_err(|e| StorageError::MigrationFailed(e.to_string()))?;

                    match self.migrate_memory(&candidate).await {
                        Ok(()) => {
                            if candidate.target_tier == TierType::Warm {
                                report.hot_to_warm += 1;
                            }
                        }
                        Err(e) => {
                            report.errors.push(format!("Failed to migrate {}: {}", candidate.memory_id, e));
                        }
                    }
                }
            }
            Err(e) => {
                report.errors.push(format!("Hot tier evaluation failed: {}", e));
            }
        }

        // Evaluate warm tier migrations
        match self.evaluate_warm_tier_migrations().await {
            Ok(candidates) => {
                for candidate in candidates {
                    let _permit = self.rate_limiter.acquire().await
                        .map_err(|e| StorageError::MigrationFailed(e.to_string()))?;

                    match self.migrate_memory(&candidate).await {
                        Ok(()) => {
                            match candidate.target_tier {
                                TierType::Hot => report.warm_to_hot += 1,
                                TierType::Cold => report.warm_to_cold += 1,
                                _ => {}
                            }
                        }
                        Err(e) => {
                            report.errors.push(format!("Failed to migrate {}: {}", candidate.memory_id, e));
                        }
                    }
                }
            }
            Err(e) => {
                report.errors.push(format!("Warm tier evaluation failed: {}", e));
            }
        }

        // Evaluate cold tier for promotion
        match self.evaluate_cold_tier_migrations().await {
            Ok(candidates) => {
                for candidate in candidates {
                    let _permit = self.rate_limiter.acquire().await
                        .map_err(|e| StorageError::MigrationFailed(e.to_string()))?;

                    match self.migrate_memory(&candidate).await {
                        Ok(()) => {
                            if candidate.target_tier == TierType::Warm {
                                report.cold_to_warm += 1;
                            }
                        }
                        Err(e) => {
                            report.errors.push(format!("Failed to migrate {}: {}", candidate.memory_id, e));
                        }
                    }
                }
            }
            Err(e) => {
                report.errors.push(format!("Cold tier evaluation failed: {}", e));
            }
        }

        report.duration = SystemTime::now()
            .duration_since(start_time)
            .unwrap_or_default();
        report.total_migrations = report.successful_migrations();

        Ok(report)
    }

    /// Evaluate hot tier for demotion candidates
    async fn evaluate_hot_tier_migrations(&self) -> Result<Vec<MigrationCandidate>, StorageError> {
        let mut candidates = Vec::new();
        let hot_tier = self.hot_tier.as_ref()
            .ok_or_else(|| StorageError::NotInitialized("Hot tier not set".to_string()))?;

        // Iterate through hot tier memories
        for entry in hot_tier.data.iter() {
            let memory_id = entry.key();
            let memory = entry.value();

            // Get access statistics
            let stats = self.access_tracker.get_access_stats(memory_id);
            let current_activation = memory.activation();

            // Check demotion criteria (research: activation < 0.3 OR idle_time > 5 minutes)
            let should_demote =
                current_activation < 0.3 ||
                stats.idle_time > Duration::from_secs(300) || // 5 minutes
                stats.activation_trend < -0.2; // Declining activation

            if should_demote {
                candidates.push(MigrationCandidate {
                    memory_id: memory_id.clone(),
                    source_tier: TierType::Hot,
                    target_tier: TierType::Warm,
                    priority: 1.0 - current_activation, // Lower activation = higher priority
                    activation: current_activation,
                    idle_time: stats.idle_time,
                });
            }
        }

        // Sort by priority (highest first)
        candidates.sort_by(|a, b| b.priority.partial_cmp(&a.priority).unwrap_or(std::cmp::Ordering::Equal));

        // Limit batch size to 100 memories per cycle (research recommendation)
        candidates.truncate(100);

        Ok(candidates)
    }

    /// Evaluate warm tier for promotion or demotion
    async fn evaluate_warm_tier_migrations(&self) -> Result<Vec<MigrationCandidate>, StorageError> {
        let mut candidates = Vec::new();

        // Note: Warm tier evaluation would require access to warm tier data
        // This is a placeholder implementation
        // In practice, you'd iterate through warm tier memories similar to hot tier

        Ok(candidates)
    }

    /// Evaluate cold tier for promotion candidates
    async fn evaluate_cold_tier_migrations(&self) -> Result<Vec<MigrationCandidate>, StorageError> {
        let mut candidates = Vec::new();

        // Note: Cold tier evaluation would check for recent access patterns
        // This is a placeholder implementation

        Ok(candidates)
    }

    /// Migrate a memory between tiers
    async fn migrate_memory(&self, candidate: &MigrationCandidate) -> Result<(), StorageError> {
        // Record the migration for tracking
        self.access_tracker.record_access(&candidate.memory_id, candidate.activation);

        // Get the memory from source tier
        let memory = match candidate.source_tier {
            TierType::Hot => {
                let hot = self.hot_tier.as_ref()
                    .ok_or_else(|| StorageError::NotInitialized("Hot tier not set".to_string()))?;
                hot.get_memory(&candidate.memory_id)
                    .ok_or_else(|| StorageError::NotFound(candidate.memory_id.clone()))?
            }
            TierType::Warm => {
                // Would retrieve from warm tier
                return Err(StorageError::NotImplemented("Warm tier retrieval not implemented".to_string()));
            }
            TierType::Cold => {
                // Would retrieve from cold tier
                return Err(StorageError::NotImplemented("Cold tier retrieval not implemented".to_string()));
            }
        };

        // Store in target tier
        match candidate.target_tier {
            TierType::Hot => {
                let hot = self.hot_tier.as_ref()
                    .ok_or_else(|| StorageError::NotInitialized("Hot tier not set".to_string()))?;
                hot.store(memory.clone()).await?;
            }
            TierType::Warm => {
                let warm = self.warm_tier.as_ref()
                    .ok_or_else(|| StorageError::NotInitialized("Warm tier not set".to_string()))?;
                warm.store(memory.clone()).await?;
            }
            TierType::Cold => {
                let cold = self.cold_tier.as_ref()
                    .ok_or_else(|| StorageError::NotInitialized("Cold tier not set".to_string()))?;
                cold.store(memory.clone()).await?;
            }
        }

        // Remove from source tier (only after successful store)
        match candidate.source_tier {
            TierType::Hot => {
                let hot = self.hot_tier.as_ref()
                    .ok_or_else(|| StorageError::NotInitialized("Hot tier not set".to_string()))?;
                hot.evict_memory(&candidate.memory_id);
            }
            TierType::Warm => {
                let warm = self.warm_tier.as_ref()
                    .ok_or_else(|| StorageError::NotInitialized("Warm tier not set".to_string()))?;
                warm.remove(&candidate.memory_id).await?;
            }
            TierType::Cold => {
                let cold = self.cold_tier.as_ref()
                    .ok_or_else(|| StorageError::NotInitialized("Cold tier not set".to_string()))?;
                cold.remove(&candidate.memory_id).await?;
            }
        }

        Ok(())
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

    #[tokio::test]
    async fn test_migration_cycle() {
        let temp_dir = TempDir::new().unwrap();
        let metrics = Arc::new(StorageMetrics::new());

        // Create coordinator with tiers
        let mut coordinator = TierCoordinator::new(
            CognitiveEvictionPolicy {
                hot_activation_threshold: 0.7,
                warm_access_window: Duration::from_secs(60),
                cold_migration_age: Duration::from_secs(3600),
                recency_boost_factor: 1.2,
            },
            metrics.clone(),
        );

        let hot_tier = Arc::new(HotTier::new(100));
        let warm_tier = Arc::new(WarmTier::new(
            temp_dir.path().join("warm.dat"),
            1000,
            metrics.clone(),
        ).unwrap());
        let cold_tier = Arc::new(ColdTier::new(10000));

        coordinator.set_tiers(
            hot_tier.clone(),
            warm_tier.clone(),
            cold_tier.clone(),
        );

        // Add test memories to hot tier (research: < 0.3 triggers migration)
        let low_activation_memory = create_test_memory("low_mem", 0.2);
        let high_activation_memory = create_test_memory("high_mem", 0.9);

        hot_tier.store(low_activation_memory).await.unwrap();
        hot_tier.store(high_activation_memory).await.unwrap();

        // Record access patterns
        coordinator.access_tracker.record_access("low_mem", 0.2);
        coordinator.access_tracker.record_access("high_mem", 0.9);

        // Wait to create idle time
        tokio::time::sleep(Duration::from_millis(100)).await;

        // Run migration cycle
        let report = coordinator.run_migration_cycle().await.unwrap();

        // Low activation memory should be migrated to warm tier
        assert!(report.hot_to_warm > 0, "Should migrate low activation memory");
        assert_eq!(report.errors.len(), 0, "Should have no errors");
    }

    #[tokio::test]
    async fn test_access_tracking_integration() {
        let metrics = Arc::new(StorageMetrics::new());
        let coordinator = TierCoordinator::new(CognitiveEvictionPolicy::default(), metrics);

        // Record multiple accesses
        coordinator.access_tracker.record_access("mem1", 0.8);
        coordinator.access_tracker.record_access("mem1", 0.9);
        coordinator.access_tracker.record_access("mem2", 0.3);

        // Check that access patterns are tracked
        let stats1 = coordinator.access_tracker.get_access_stats("mem1");
        let stats2 = coordinator.access_tracker.get_access_stats("mem2");

        assert_eq!(stats1.total_accesses, 2);
        assert_eq!(stats2.total_accesses, 1);
        assert!(stats1.average_activation > stats2.average_activation);
    }

    #[tokio::test]
    async fn test_migration_candidate_prioritization() {
        let metrics = Arc::new(StorageMetrics::new());
        let mut coordinator = TierCoordinator::new(CognitiveEvictionPolicy::default(), metrics.clone());

        let hot_tier = Arc::new(HotTier::new(100));
        let warm_tier = Arc::new(WarmTier::new(
            TempDir::new().unwrap().path().join("warm.dat"),
            1000,
            metrics.clone(),
        ).unwrap());
        let cold_tier = Arc::new(ColdTier::new(10000));

        coordinator.set_tiers(hot_tier.clone(), warm_tier, cold_tier);

        // Add memories with different activation levels
        for i in 0..5 {
            let memory = create_test_memory(&format!("mem_{}", i), 0.1 * i as f32);
            hot_tier.store(memory).await.unwrap();
            coordinator.access_tracker.record_access(&format!("mem_{}", i), 0.1 * i as f32);
        }

        // Get migration candidates
        let candidates = coordinator.evaluate_hot_tier_migrations().await.unwrap();

        // Candidates should be sorted by priority
        for i in 1..candidates.len() {
            assert!(
                candidates[i-1].priority >= candidates[i].priority,
                "Candidates should be sorted by priority"
            );
        }
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
