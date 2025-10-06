//! Infallible memory store with graceful degradation
//!
//! Implements cognitive design principles where operations never fail but
//! degrade gracefully under pressure, returning activation levels that
//! indicate store quality.

use crate::memory_graph::{GraphConfig, InfallibleBackend, UnifiedMemoryGraph};
use crate::numeric::{u64_to_f64, unit_ratio_to_f32};
use crate::storage::{
    ContentAddress, ContentIndex, DeduplicationAction, MergeStrategy, SemanticDeduplicator,
};
use crate::{Confidence, Cue, CueType, Episode, Memory, TemporalPattern};
use chrono::Utc;
use dashmap::DashMap;
use parking_lot::RwLock;
use std::collections::BTreeMap;
use std::convert::TryFrom;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

const MAX_SAFE_INTEGER: u64 = 9_007_199_254_740_992; // 2^53

#[cfg(feature = "monitoring")]
use std::time::Instant;

#[cfg(feature = "monitoring")]
use crate::metrics::cognitive::CognitiveMetric;

#[cfg(feature = "pattern_completion")]
use crate::completion::{
    CompletedEpisode, CompletionConfig, PartialEpisode, PatternCompleter, PatternReconstructor,
};

#[cfg(feature = "hnsw_index")]
use crate::index::CognitiveHnswIndex;

#[cfg(feature = "memory_mapped_persistence")]
use crate::storage::{
    CognitiveTierArchitecture, FsyncMode, StorageMetrics,
    wal::{WalEntryType, WalReader, WalWriter},
};

/// Activation level returned by store operations
///
/// Indicates the quality of a store operation from 0.0 to 1.0
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct Activation(f32);

/// HNSW index update for async processing
#[cfg(feature = "hnsw_index")]
#[derive(Clone)]
pub enum HnswUpdate {
    /// Insert a memory into the HNSW index
    Insert { memory: Arc<Memory> },
    /// Remove a memory from the HNSW index
    Remove { id: String },
    /// Rebuild the entire HNSW index
    Rebuild,
}

/// Statistics for HNSW update queue
#[cfg(feature = "hnsw_index")]
#[derive(Debug, Clone, Copy)]
pub struct HnswQueueStats {
    /// Current number of updates in queue
    pub queue_depth: usize,
    /// Maximum queue capacity
    pub queue_capacity: usize,
    /// Queue utilization (0.0 to 1.0)
    pub utilization: f32,
}

impl Activation {
    /// Create a new activation level
    #[must_use]
    pub const fn new(value: f32) -> Self {
        Self(value.clamp(0.0, 1.0))
    }

    /// Get the raw activation value
    #[must_use]
    pub const fn value(self) -> f32 {
        self.0
    }

    /// Check if activation indicates successful store
    #[must_use]
    pub fn is_successful(self) -> bool {
        self.0 > 0.5
    }

    /// Check if activation indicates degraded store
    #[must_use]
    pub fn is_degraded(self) -> bool {
        self.0 < 0.8
    }
}

/// Memory store that never fails, degrading gracefully under pressure
///
/// # Cognitive Design
///
/// This store follows human memory formation patterns:
/// - Store quality varies based on system state (like attention/fatigue)
/// - Returns activation levels instead of Result types
/// - Graceful degradation mirrors biological memory under stress
/// - Concurrent stores don't block (like parallel memory formation)
pub struct MemoryStore {
    /// Unified memory graph with infallible backend
    graph: UnifiedMemoryGraph<InfallibleBackend>,

    /// Legacy compatibility: hot memories reference
    pub(crate) hot_memories: DashMap<String, Arc<Memory>>,

    /// Write-ahead log for durability (non-blocking)
    pub(crate) wal_buffer: Arc<DashMap<String, Episode>>,

    /// Maximum number of memories before eviction
    max_memories: usize,

    /// Current memory count
    memory_count: AtomicUsize,

    /// System pressure indicator
    pressure: RwLock<f32>,

    /// Eviction queue for least-recently-used memories
    eviction_queue: RwLock<BTreeMap<(OrderedFloat, String), Arc<Memory>>>,

    /// HNSW index for fast similarity search
    #[cfg(feature = "hnsw_index")]
    hnsw_index: Option<Arc<CognitiveHnswIndex>>,

    /// HNSW index update queue for async processing
    #[cfg(feature = "hnsw_index")]
    hnsw_update_queue: Arc<crossbeam_queue::ArrayQueue<HnswUpdate>>,

    /// Background worker handle for HNSW updates
    #[cfg(feature = "hnsw_index")]
    hnsw_worker: Arc<parking_lot::Mutex<Option<std::thread::JoinHandle<()>>>>,

    /// Shutdown signal for HNSW worker
    #[cfg(feature = "hnsw_index")]
    hnsw_shutdown: Arc<std::sync::atomic::AtomicBool>,

    /// Persistent storage backend
    #[cfg(feature = "memory_mapped_persistence")]
    persistent_backend: Option<Arc<CognitiveTierArchitecture>>,

    /// Background worker handle for tier migration
    #[cfg(feature = "memory_mapped_persistence")]
    tier_migration_worker: Arc<parking_lot::Mutex<Option<std::thread::JoinHandle<()>>>>,

    /// Shutdown signal for tier migration worker
    #[cfg(feature = "memory_mapped_persistence")]
    tier_migration_shutdown: Arc<std::sync::atomic::AtomicBool>,

    /// Write-ahead log for durability
    #[cfg(feature = "memory_mapped_persistence")]
    wal_writer: Option<Arc<WalWriter>>,

    /// Storage metrics
    #[cfg(feature = "memory_mapped_persistence")]
    storage_metrics: Arc<StorageMetrics>,

    /// Pattern completion engine
    #[cfg(feature = "pattern_completion")]
    pattern_reconstructor: Option<Arc<RwLock<PatternReconstructor>>>,

    /// Content-addressable index for deduplication
    content_index: Arc<ContentIndex>,

    /// Semantic deduplicator for preventing near-duplicates
    deduplicator: Arc<RwLock<SemanticDeduplicator>>,

    /// Cognitive recall pipeline for spreading activation
    #[cfg(feature = "hnsw_index")]
    cognitive_recall: Option<Arc<crate::activation::CognitiveRecall>>,

    /// Configuration for recall operations
    recall_config: crate::activation::RecallConfig,
}

/// Wrapper for f32 that implements Ord for `BTreeMap`
#[derive(Clone, Copy, Debug)]
struct OrderedFloat(f32);

impl PartialEq for OrderedFloat {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl Eq for OrderedFloat {}

impl PartialOrd for OrderedFloat {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for OrderedFloat {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0
            .partial_cmp(&other.0)
            .unwrap_or(std::cmp::Ordering::Equal)
    }
}

impl MemoryStore {
    fn update_pressure_metric(&self, value: f32) {
        let mut pressure = self.pressure.write();
        *pressure = value;
    }

    fn pressure_ratio(current_count: usize, max_memories: usize) -> f32 {
        if max_memories == 0 {
            return 0.0;
        }

        let current_u64 = u64::try_from(current_count).unwrap_or(u64::MAX);
        let max_u64 = u64::try_from(max_memories).unwrap_or(u64::MAX);

        let capped_current = current_u64.clamp(0_u64, MAX_SAFE_INTEGER);
        let capped_max = max_u64.clamp(1, MAX_SAFE_INTEGER);

        unit_ratio_to_f32(capped_current, capped_max)
    }

    #[cfg(feature = "memory_mapped_persistence")]
    fn persist_episode(&self, memory_arc: &Arc<Memory>, episode: &Episode) {
        use std::convert::TryFrom;

        if let Some(ref wal_writer) = self.wal_writer {
            if let Ok(wal_entry) = crate::storage::wal::WalEntry::new_episode(episode) {
                wal_writer.write_async(wal_entry);
                let episode_size = std::mem::size_of::<Episode>() as u64
                    + u64::try_from(episode.id.len()).unwrap_or(0)
                    + u64::try_from(episode.what.len()).unwrap_or(0);
                self.storage_metrics.record_write(episode_size);
            }
        }

        if let Some(ref backend) = self.persistent_backend {
            let backend = Arc::clone(backend);
            let memory = Arc::clone(memory_arc);
            Self::spawn_persistence_task(backend, memory);
        }
    }

    #[cfg(feature = "memory_mapped_persistence")]
    fn spawn_persistence_task(backend: Arc<CognitiveTierArchitecture>, memory: Arc<Memory>) {
        if let Ok(handle) = tokio::runtime::Handle::try_current() {
            handle.spawn(async move {
                if let Err(error) = backend.store(memory).await {
                    tracing::warn!(?error, "failed to persist memory in tiered storage backend");
                }
            });
        } else if let Err(spawn_error) = std::thread::Builder::new()
            .name("engram-tier-persistence".to_string())
            .spawn(move || {
                match tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                {
                    Ok(runtime) => {
                        if let Err(error) = runtime.block_on(backend.store(memory)) {
                            tracing::warn!(
                                ?error,
                                "failed to persist memory in tiered storage backend"
                            );
                        }
                    }
                    Err(error) => {
                        tracing::warn!(?error, "failed to construct persistence runtime");
                    }
                }
            })
        {
            tracing::warn!(?spawn_error, "failed to spawn persistence worker thread");
        }
    }

    /// Create a new memory store with specified capacity
    #[must_use]
    pub fn new(max_memories: usize) -> Self {
        let config = GraphConfig {
            max_results: 100,
            enable_spreading: true,
            decay_rate: 0.8,
            max_depth: 3,
            activation_threshold: 0.01,
        };

        let backend = InfallibleBackend::new(max_memories);
        let graph = UnifiedMemoryGraph::new(backend, config);

        Self {
            graph,
            hot_memories: DashMap::new(),
            wal_buffer: Arc::new(DashMap::new()),
            max_memories,
            memory_count: AtomicUsize::new(0),
            pressure: RwLock::new(0.0),
            eviction_queue: RwLock::new(BTreeMap::new()),
            #[cfg(feature = "hnsw_index")]
            hnsw_index: None,
            #[cfg(feature = "hnsw_index")]
            hnsw_update_queue: Arc::new(crossbeam_queue::ArrayQueue::new(10_000)),
            #[cfg(feature = "hnsw_index")]
            hnsw_worker: Arc::new(parking_lot::Mutex::new(None)),
            #[cfg(feature = "hnsw_index")]
            hnsw_shutdown: Arc::new(std::sync::atomic::AtomicBool::new(false)),
            #[cfg(feature = "memory_mapped_persistence")]
            persistent_backend: None,
            #[cfg(feature = "memory_mapped_persistence")]
            tier_migration_worker: Arc::new(parking_lot::Mutex::new(None)),
            #[cfg(feature = "memory_mapped_persistence")]
            tier_migration_shutdown: Arc::new(std::sync::atomic::AtomicBool::new(false)),
            #[cfg(feature = "memory_mapped_persistence")]
            wal_writer: None,
            #[cfg(feature = "memory_mapped_persistence")]
            storage_metrics: Arc::new(StorageMetrics::new()),
            #[cfg(feature = "pattern_completion")]
            pattern_reconstructor: None,
            content_index: Arc::new(ContentIndex::new()),
            deduplicator: Arc::new(RwLock::new(SemanticDeduplicator::default())),
            #[cfg(feature = "hnsw_index")]
            cognitive_recall: None,
            recall_config: crate::activation::RecallConfig::default(),
        }
    }

    /// Create a memory store with HNSW index enabled
    ///
    /// Automatically starts the background worker thread to process HNSW updates.
    /// The worker will run until `shutdown_hnsw_worker()` is called.
    #[cfg(feature = "hnsw_index")]
    #[must_use]
    pub fn with_hnsw_index(mut self) -> Self {
        self.hnsw_index = Some(Arc::new(CognitiveHnswIndex::new()));
        self.start_hnsw_worker();
        self
    }

    /// Access the HNSW index when enabled
    ///
    /// Returns a reference to the HNSW index Arc if one is configured.
    /// This is useful for building cognitive recall pipelines that need
    /// to use the same index the store is using.
    #[cfg(feature = "hnsw_index")]
    #[must_use]
    pub fn hnsw_index(&self) -> Option<Arc<CognitiveHnswIndex>> {
        self.hnsw_index.clone()
    }

    /// Set cognitive recall pipeline for spreading activation
    #[cfg(feature = "hnsw_index")]
    #[must_use]
    pub fn with_cognitive_recall(
        mut self,
        recall: Arc<crate::activation::CognitiveRecall>,
    ) -> Self {
        self.cognitive_recall = Some(recall);
        self
    }

    /// Set recall configuration
    #[must_use]
    pub fn with_recall_config(mut self, config: crate::activation::RecallConfig) -> Self {
        self.recall_config = config;
        self
    }

    /// Create a memory store with persistent backend enabled
    #[cfg(feature = "memory_mapped_persistence")]
    ///
    /// # Errors
    ///
    /// Returns an error if the tiered storage backend or WAL writer cannot be
    /// initialised at the provided data directory.
    pub fn with_persistence<P: AsRef<std::path::Path>>(
        mut self,
        data_dir: P,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let storage_metrics = Arc::clone(&self.storage_metrics);

        // Create persistent backend
        let persistent_backend = Arc::new(CognitiveTierArchitecture::new(
            data_dir.as_ref(),
            self.max_memories,       // hot capacity
            self.max_memories * 10,  // warm capacity
            self.max_memories * 100, // cold capacity
            Arc::clone(&storage_metrics),
        )?);

        // Create WAL writer
        let wal_dir = data_dir.as_ref().join("wal");
        let wal_writer = Arc::new(WalWriter::new(
            wal_dir,
            FsyncMode::PerBatch,
            storage_metrics,
        )?);

        self.persistent_backend = Some(persistent_backend);
        self.wal_writer = Some(wal_writer);

        Ok(self)
    }

    /// Initialize the persistent backend (start background workers)
    #[cfg(feature = "memory_mapped_persistence")]
    ///
    /// # Errors
    ///
    /// Returns an error if the WAL writer is still shared elsewhere or if
    /// the underlying writer fails to start.
    pub fn initialize_persistence(
        &mut self,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        use crate::storage::StorageError;

        if let Some(ref mut wal_writer) = self.wal_writer {
            let writer = Arc::get_mut(wal_writer).ok_or_else(|| {
                StorageError::wal_failed("WAL writer already in use; cannot start")
            })?;
            writer
                .start()
                .map_err(|e| Box::new(e) as Box<dyn std::error::Error + Send + Sync>)?;
        }
        Ok(())
    }

    /// Start background tier migration task
    ///
    /// Returns a join handle for the background thread if migration is enabled.
    /// The task runs every 5 minutes and migrates memories between tiers based on
    /// access patterns (hot→warm→cold and warm→hot for recently accessed).
    #[cfg(feature = "memory_mapped_persistence")]
    #[must_use = "Join handle should be stored for graceful shutdown"]
    pub fn start_tier_migration(&self) {
        // Check if worker is already running
        if self.tier_migration_worker.lock().is_some() {
            tracing::debug!("Tier migration worker already running");
            return;
        }

        let backend = match &self.persistent_backend {
            Some(b) => b.clone(),
            None => {
                tracing::warn!("Cannot start tier migration: no persistent backend configured");
                return;
            }
        };

        let shutdown = Arc::clone(&self.tier_migration_shutdown);

        let handle = match std::thread::Builder::new()
            .name("tier-migration".to_string())
            .spawn(move || {
                // Create a tokio runtime for async operations
                let runtime = match tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                {
                    Ok(rt) => rt,
                    Err(e) => {
                        tracing::error!(error = ?e, "Failed to create runtime for tier migration");
                        return;
                    }
                };

                runtime.block_on(async {
                    let mut interval = tokio::time::interval(std::time::Duration::from_secs(300)); // 5 minutes

                    loop {
                        // Check shutdown signal
                        if shutdown.load(std::sync::atomic::Ordering::Relaxed) {
                            tracing::info!("Tier migration worker shutting down");
                            break;
                        }

                        interval.tick().await;

                        // Run migration cycle
                        match backend.maintenance().await {
                            Ok(()) => {
                                tracing::debug!("Tier maintenance completed successfully");
                            }
                            Err(e) => {
                                tracing::warn!(error = ?e, "Tier maintenance failed");
                            }
                        }
                    }
                });
            }) {
            Ok(h) => h,
            Err(e) => {
                tracing::error!(error = ?e, "Failed to spawn tier migration thread");
                return;
            }
        };

        *self.tier_migration_worker.lock() = Some(handle);
        tracing::info!("Started tier migration background task (5 minute interval)");
    }

    /// Graceful shutdown of tier migration worker
    ///
    /// Signals the worker to stop and joins the thread.
    #[cfg(feature = "memory_mapped_persistence")]
    pub fn shutdown_tier_migration(&self) {
        if let Some(handle) = self.tier_migration_worker.lock().take() {
            tracing::info!("Shutting down tier migration worker");

            // Signal shutdown
            self.tier_migration_shutdown
                .store(true, std::sync::atomic::Ordering::Relaxed);

            // Wait for worker to finish
            match handle.join() {
                Ok(()) => tracing::info!("Tier migration worker shut down gracefully"),
                Err(e) => {
                    tracing::error!("Tier migration worker panicked during shutdown: {:?}", e)
                }
            }

            // Reset shutdown flag for potential restart
            self.tier_migration_shutdown
                .store(false, std::sync::atomic::Ordering::Relaxed);
        }
    }

    /// Start HNSW update worker thread for async index updates
    ///
    /// Returns a join handle for the background thread if HNSW is enabled.
    /// The worker processes updates in batches of 100 with 50ms timeout.
    #[cfg(feature = "hnsw_index")]
    #[must_use = "Join handle should be stored for graceful shutdown"]
    pub fn start_hnsw_worker(&self) {
        // Check if worker is already running
        if self.hnsw_worker.lock().is_some() {
            tracing::debug!("HNSW worker already running");
            return;
        }

        let queue = Arc::clone(&self.hnsw_update_queue);
        let shutdown = Arc::clone(&self.hnsw_shutdown);
        let hnsw = match &self.hnsw_index {
            Some(index) => Arc::clone(index),
            None => {
                tracing::warn!("Cannot start HNSW worker: no HNSW index configured");
                return;
            }
        };

        let handle = std::thread::Builder::new()
            .name("hnsw-worker".to_string())
            .spawn(move || {
                Self::hnsw_worker_loop(queue, hnsw, shutdown);
            })
            .expect("Failed to spawn HNSW worker thread");

        *self.hnsw_worker.lock() = Some(handle);
        tracing::info!("Started HNSW update worker thread (batch size: 100, timeout: 50ms)");
    }

    /// Background worker loop for processing HNSW updates
    ///
    /// Runs until shutdown signal is received, processing updates in batches.
    #[cfg(feature = "hnsw_index")]
    fn hnsw_worker_loop(
        queue: Arc<crossbeam_queue::ArrayQueue<HnswUpdate>>,
        hnsw: Arc<CognitiveHnswIndex>,
        shutdown: Arc<std::sync::atomic::AtomicBool>,
    ) {
        const BATCH_SIZE: usize = 100;
        const BATCH_TIMEOUT: std::time::Duration = std::time::Duration::from_millis(50);

        let mut batch = Vec::with_capacity(BATCH_SIZE);
        let mut last_flush = std::time::Instant::now();

        loop {
            // Check shutdown signal
            if shutdown.load(std::sync::atomic::Ordering::Relaxed) {
                // Drain remaining updates before shutdown
                while let Some(update) = queue.pop() {
                    batch.push(update);
                }
                if !batch.is_empty() {
                    tracing::info!(
                        count = batch.len(),
                        "Processing remaining HNSW updates before shutdown"
                    );
                    Self::process_hnsw_batch(&hnsw, &batch);
                }
                tracing::info!("HNSW worker shutting down");
                break;
            }

            // Collect updates into batch
            while batch.len() < BATCH_SIZE {
                match queue.pop() {
                    Some(update) => batch.push(update),
                    None => break,
                }
            }

            // Flush batch if full or timeout reached
            let should_flush = batch.len() >= BATCH_SIZE
                || (last_flush.elapsed() > BATCH_TIMEOUT && !batch.is_empty());

            if should_flush {
                Self::process_hnsw_batch(&hnsw, &batch);
                batch.clear();
                last_flush = std::time::Instant::now();
            } else if batch.is_empty() {
                // Sleep briefly to avoid busy loop
                std::thread::sleep(std::time::Duration::from_millis(10));
            }
        }
    }

    /// Process a batch of HNSW updates
    #[cfg(feature = "hnsw_index")]
    fn process_hnsw_batch(hnsw: &CognitiveHnswIndex, batch: &[HnswUpdate]) {
        let start = std::time::Instant::now();
        let mut processed = 0;
        let mut failed = 0;
        let mut skipped = 0;

        for update in batch {
            match update {
                HnswUpdate::Insert { memory } => {
                    if let Err(e) = hnsw.insert_memory(Arc::clone(memory)) {
                        tracing::warn!(error = ?e, memory_id = %memory.id, "HNSW insert failed");
                        failed += 1;
                    } else {
                        processed += 1;
                    }
                }
                HnswUpdate::Remove { id } => {
                    // Remove operation not yet implemented in CognitiveHnswIndex
                    tracing::trace!(memory_id = %id, "HNSW remove operation not yet implemented");
                    skipped += 1;
                }
                HnswUpdate::Rebuild => {
                    // Rebuild operation not yet implemented in CognitiveHnswIndex
                    tracing::trace!("HNSW rebuild operation not yet implemented");
                    skipped += 1;
                }
            }
        }

        if processed > 0 || failed > 0 || skipped > 0 {
            tracing::debug!(
                processed,
                failed,
                skipped,
                duration_ms = start.elapsed().as_millis() as u64,
                "Processed HNSW update batch"
            );
        }
    }

    /// Graceful shutdown of HNSW worker
    ///
    /// Signals the worker to stop, drains any remaining updates, and joins the thread.
    /// This ensures all queued HNSW updates are processed before shutdown completes.
    #[cfg(feature = "hnsw_index")]
    pub fn shutdown_hnsw_worker(&self) {
        if let Some(handle) = self.hnsw_worker.lock().take() {
            tracing::info!("Shutting down HNSW worker");

            // Signal shutdown
            self.hnsw_shutdown
                .store(true, std::sync::atomic::Ordering::Relaxed);

            // Wait for worker to finish processing remaining updates
            match handle.join() {
                Ok(()) => tracing::info!("HNSW worker shut down gracefully"),
                Err(e) => tracing::error!("HNSW worker panicked during shutdown: {:?}", e),
            }

            // Reset shutdown flag for potential restart
            self.hnsw_shutdown
                .store(false, std::sync::atomic::Ordering::Relaxed);
        }
    }

    /// Get HNSW queue statistics
    #[cfg(feature = "hnsw_index")]
    #[must_use]
    pub fn hnsw_queue_stats(&self) -> HnswQueueStats {
        let queue_len = self.hnsw_update_queue.len();
        let queue_capacity = self.hnsw_update_queue.capacity();

        HnswQueueStats {
            queue_depth: queue_len,
            queue_capacity,
            utilization: if queue_capacity > 0 {
                queue_len as f32 / queue_capacity as f32
            } else {
                0.0
            },
        }
    }

    /// Recover previously written episodes from the WAL into the hot tier
    #[cfg(feature = "memory_mapped_persistence")]
    pub fn recover_from_wal(&mut self) -> Result<usize, crate::storage::StorageError> {
        let Some(wal_writer) = self.wal_writer.as_ref() else {
            return Ok(0);
        };

        let wal_reader = WalReader::new(
            wal_writer.wal_directory(),
            Arc::clone(&self.storage_metrics),
        );
        let entries = wal_reader.scan_all()?;

        let mut recovered = 0usize;

        for entry in entries {
            let entry_type = WalEntryType::from(entry.header.entry_type);
            match entry_type {
                WalEntryType::EpisodeStore => {
                    match bincode::deserialize::<Episode>(&entry.payload) {
                        Ok(episode) => {
                            if self.hot_memories.contains_key(&episode.id) {
                                continue;
                            }

                            let activation = episode.encoding_confidence.raw();
                            let memory = Memory::from_episode(episode.clone(), activation);
                            let memory_arc = Arc::new(memory);

                            let content_address =
                                ContentAddress::from_embedding(&memory_arc.embedding);
                            let _ = self
                                .content_index
                                .insert(content_address, episode.id.clone());

                            self.hot_memories
                                .insert(episode.id.clone(), Arc::clone(&memory_arc));
                            self.wal_buffer.insert(episode.id.clone(), episode.clone());

                            {
                                let mut queue = self.eviction_queue.write();
                                queue.insert(
                                    (OrderedFloat(memory_arc.activation()), episode.id.clone()),
                                    Arc::clone(&memory_arc),
                                );
                            }

                            self.memory_count.fetch_add(1, Ordering::Relaxed);
                            recovered += 1;
                        }
                        Err(error) => {
                            tracing::warn!(
                                ?error,
                                "Failed to deserialize WAL episode during recovery"
                            );
                        }
                    }
                }
                WalEntryType::MemoryDelete => {
                    if let Ok(memory_id) = String::from_utf8(entry.payload.clone()) {
                        if self.hot_memories.remove(&memory_id).is_some() {
                            self.memory_count.fetch_sub(1, Ordering::Relaxed);
                        }
                        self.wal_buffer.remove(&memory_id);
                        let _ = self.content_index.remove_by_memory_id(&memory_id);
                        {
                            let mut queue = self.eviction_queue.write();
                            queue.retain(|(_, id), _| id != &memory_id);
                        }
                    } else {
                        tracing::warn!("Invalid UTF-8 payload for WAL memory deletion entry");
                    }
                }
                WalEntryType::MemoryUpdate
                | WalEntryType::Consolidation
                | WalEntryType::Checkpoint
                | WalEntryType::CompactionMarker => {
                    tracing::debug!(
                        entry_type = ?entry_type,
                        "Skipping WAL entry type during recovery"
                    );
                }
            }
        }

        let current_count = self.memory_count.load(Ordering::Relaxed);
        let pressure = Self::pressure_ratio(current_count, self.max_memories);
        self.update_pressure_metric(pressure);

        Ok(recovered)
    }

    /// Store an episode, returning activation level indicating store quality
    ///
    /// # Returns
    ///
    /// Activation level from 0.0 to 1.0 indicating:
    /// - 1.0: Perfect store with full confidence
    /// - 0.8-0.9: Normal store with slight system pressure
    /// - 0.5-0.7: Degraded store under memory pressure
    /// - 0.3-0.4: Heavily degraded, may be evicted soon
    /// - < 0.3: Critical pressure, immediate eviction likely
    ///
    /// # Cognitive Design
    ///
    /// Never returns errors because human memory formation doesn't "fail" -
    /// it degrades. Under stress or fatigue, memories form with lower
    /// confidence and are more likely to be forgotten.
    pub fn store(&self, episode: Episode) -> Activation {
        #[cfg(feature = "monitoring")]
        let start = Instant::now();

        let current_count = self.memory_count.load(Ordering::Relaxed);
        let pressure = Self::pressure_ratio(current_count, self.max_memories);
        self.update_pressure_metric(pressure);

        let base_activation = episode.encoding_confidence.raw() * pressure.mul_add(-0.5, 1.0);

        if current_count >= self.max_memories {
            self.evict_lowest_activation();
        }

        let wal_episode = episode.clone();
        let graph_penalty = match self.graph.store_episode(wal_episode.clone()) {
            Ok(_) => 0.0,
            Err(error) => {
                tracing::warn!(?error, "failed to record episode in unified memory graph");
                0.1
            }
        };

        let activation = (base_activation * (1.0 - graph_penalty)).clamp(0.0, 1.0);
        let memory = Memory::from_episode(episode, activation);
        let content_address = ContentAddress::from_embedding(&memory.embedding);

        let existing_memories: Vec<Arc<Memory>> = self
            .hot_memories
            .iter()
            .map(|entry| Arc::clone(entry.value()))
            .collect();

        let dedup_result = {
            let mut dedup = self.deduplicator.write();
            dedup.check_duplicate(&memory, &existing_memories)
        };

        let mut memory_id = memory.id.clone();
        match dedup_result.action {
            DeduplicationAction::Skip => {
                if dedup_result.existing_id.is_some() {
                    return Activation::new(activation * 0.9);
                }
            }
            DeduplicationAction::Replace(existing_id) => {
                self.hot_memories.remove(&existing_id);
                let _ = self.content_index.remove(&content_address);
            }
            DeduplicationAction::Merge(existing_id) => {
                if let Some(existing) = self.hot_memories.get(&existing_id) {
                    let merged = {
                        let dedup = self.deduplicator.read();
                        dedup.merge_memories(&memory, existing.value())
                    };
                    let merged_arc = Arc::new(merged);
                    self.hot_memories
                        .insert(existing_id.clone(), Arc::clone(&merged_arc));
                    return Activation::new(activation * 0.95);
                }
                memory_id = existing_id;
            }
            _ => {}
        }

        let memory_arc = Arc::new(memory);

        let _ = self
            .content_index
            .insert(content_address, memory_id.clone());
        self.hot_memories
            .insert(memory_id.clone(), Arc::clone(&memory_arc));

        {
            let mut queue = self.eviction_queue.write();
            queue.insert(
                (OrderedFloat(activation), memory_id.clone()),
                Arc::clone(&memory_arc),
            );
        }

        #[cfg(feature = "memory_mapped_persistence")]
        self.persist_episode(&memory_arc, &wal_episode);

        self.wal_buffer.insert(memory_id.clone(), wal_episode);

        // Queue HNSW update for async processing instead of blocking
        #[cfg(feature = "hnsw_index")]
        {
            if self.hnsw_index.is_some() {
                let update = HnswUpdate::Insert {
                    memory: Arc::clone(&memory_arc),
                };

                if let Err(_) = self.hnsw_update_queue.push(update) {
                    tracing::warn!(
                        "HNSW update queue full for {}, falling back to synchronous insert",
                        memory_id
                    );
                    // Fallback: synchronous update if queue is full
                    if let Some(ref hnsw) = self.hnsw_index {
                        let _ = hnsw.insert_memory(Arc::clone(&memory_arc));
                    }
                }
            }
        }

        self.memory_count.fetch_add(1, Ordering::Relaxed);

        #[cfg(feature = "monitoring")]
        {
            use crate::metrics;
            metrics::increment_counter("memories_created_total", 1);
            metrics::observe_histogram("store_activation", f64::from(activation));
            metrics::observe_histogram("store_duration_seconds", start.elapsed().as_secs_f64());

            let metric = CognitiveMetric::CLSContribution {
                hippocampal: 1.0 - pressure,
                neocortical: pressure,
            };
            metrics::record_cognitive(&metric);
        }

        Activation::new(activation)
    }

    /// Evict the memory with lowest activation
    fn evict_lowest_activation(&self) {
        let candidate = {
            let mut queue = self.eviction_queue.write();
            let next_id = queue.iter().next().map(|((_, id), _)| id.clone());
            if let Some(ref id) = next_id {
                queue.retain(|k, _| k.1 != *id);
            }
            next_id
        };

        if let Some(id) = candidate {
            self.hot_memories.remove(&id);
            self.wal_buffer.remove(&id);
            self.memory_count.fetch_sub(1, Ordering::Relaxed);
        }
    }

    /// Get a memory by ID
    pub fn get(&self, id: &str) -> Option<Arc<Memory>> {
        self.hot_memories
            .get(id)
            .map(|entry| Arc::clone(entry.value()))
    }

    /// Get an episode by ID
    pub fn get_episode(&self, id: &str) -> Option<Episode> {
        // First check WAL buffer
        if let Some(episode) = self.wal_buffer.get(id) {
            return Some(episode.clone());
        }

        // If not in WAL buffer, try converting from hot_memories
        if let Some(memory) = self.hot_memories.get(id) {
            let episode = Episode::new(
                memory.id.clone(),
                memory.created_at,
                memory
                    .content
                    .clone()
                    .unwrap_or_else(|| "No content".to_string()),
                memory.embedding,
                memory.confidence,
            );
            return Some(episode);
        }

        None
    }

    /// Get current system pressure
    pub fn pressure(&self) -> f32 {
        *self.pressure.read()
    }

    /// Get storage metrics (if persistence enabled)
    #[cfg(feature = "memory_mapped_persistence")]
    pub fn storage_metrics(&self) -> Arc<StorageMetrics> {
        Arc::clone(&self.storage_metrics)
    }

    /// Get tier statistics (if persistence enabled)
    #[cfg(feature = "memory_mapped_persistence")]
    pub fn tier_statistics(&self) -> Option<crate::storage::TierArchitectureStats> {
        self.persistent_backend
            .as_ref()
            .map(|backend| backend.get_tier_statistics())
    }

    /// Perform maintenance on storage tiers
    #[cfg(feature = "memory_mapped_persistence")]
    pub fn maintenance(&self) {
        // Perform basic maintenance operations
        if self.persistent_backend.is_some() {
            let current_count = self.memory_count.load(Ordering::Relaxed);
            let pressure = Self::pressure_ratio(current_count, self.max_memories);
            self.update_pressure_metric(pressure);

            if pressure > 0.8 {
                tracing::info!(
                    "High memory pressure detected ({pressure}), maintenance recommended"
                );
            }
        }
    }

    /// Gracefully shutdown storage backend
    #[cfg(feature = "memory_mapped_persistence")]
    ///
    /// # Errors
    ///
    /// Returns an error if the WAL writer fails to shut down cleanly.
    pub fn shutdown(&mut self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        if let Some(ref mut wal_writer) = self.wal_writer {
            if let Some(writer) = Arc::get_mut(wal_writer) {
                writer.shutdown()?;
            }
        }
        Ok(())
    }

    /// Get current memory count
    pub fn count(&self) -> usize {
        self.memory_count.load(Ordering::Relaxed)
    }

    /// Get the number of memories (alias for count)
    pub fn len(&self) -> usize {
        self.count()
    }

    /// Check if the store currently holds no memories
    pub fn is_empty(&self) -> bool {
        self.count() == 0
    }

    /// Check if store can accept more memories without eviction
    pub fn has_capacity(&self) -> bool {
        self.count() < self.max_memories
    }

    /// Get all episodes stored in the memory store
    /// Returns an iterator over (id, episode) pairs
    pub fn get_all_episodes(&self) -> impl Iterator<Item = (String, Episode)> + '_ {
        self.wal_buffer
            .iter()
            .map(|entry| (entry.key().clone(), entry.value().clone()))
            .chain(self.hot_memories.iter().map(|entry| {
                let memory = entry.value();
                let episode = Episode::new(
                    memory.id.clone(),
                    memory.created_at,
                    memory
                        .content
                        .clone()
                        .unwrap_or_else(|| format!("Memory {}", memory.id)),
                    memory.embedding,
                    memory.confidence,
                );
                (memory.id.clone(), episode)
            }))
    }

    /// Recall memories based on a cue, returning episodes with confidence scores
    ///
    /// # Returns
    ///
    /// Vector of (Episode, Confidence) tuples, sorted by confidence
    /// - Empty vector if no matches found
    /// - Low confidence results for partial matches
    /// - Higher confidence for better matches
    ///
    /// # Cognitive Design
    ///
    /// Never returns errors because human recall doesn't "fail" - it returns
    /// nothing, partial matches, or reconstructed memories with varying confidence.
    pub fn recall(&self, cue: &Cue) -> Vec<(Episode, Confidence)> {
        #[cfg(feature = "monitoring")]
        let start = Instant::now();

        // Dispatch based on recall mode if cognitive recall is available
        #[cfg(feature = "hnsw_index")]
        {
            use crate::activation::RecallMode;

            match self.recall_config.recall_mode {
                RecallMode::Spreading if self.cognitive_recall.is_some() => {
                    // Use spreading activation pipeline
                    if let Some(ref cognitive_recall) = self.cognitive_recall {
                        return match cognitive_recall.recall(cue, self) {
                            Ok(ranked_memories) => ranked_memories
                                .into_iter()
                                .map(|r| (r.episode, r.confidence))
                                .collect(),
                            Err(e) => {
                                // Fallback to similarity on error
                                tracing::warn!(target: "engram::store", error = ?e, "Spreading activation failed, falling back to similarity");
                                self.recall_similarity_only(cue)
                            }
                        };
                    }
                }
                RecallMode::Hybrid if self.cognitive_recall.is_some() => {
                    // Try spreading, fallback to similarity
                    if let Some(ref cognitive_recall) = self.cognitive_recall {
                        match cognitive_recall.recall(cue, self) {
                            Ok(ranked_memories) if !ranked_memories.is_empty() => {
                                return ranked_memories
                                    .into_iter()
                                    .map(|r| (r.episode, r.confidence))
                                    .collect();
                            }
                            Ok(_) | Err(_) => {
                                // Empty results or error - fallback to similarity
                                return self.recall_similarity_only(cue);
                            }
                        }
                    }
                }
                RecallMode::Similarity | _ => {
                    // Use similarity-only recall (default path below)
                }
            }
        }

        // Try persistent backend first if available
        #[cfg(feature = "memory_mapped_persistence")]
        {
            if let Some(ref backend) = self.persistent_backend {
                let results = self.recall_with_persistence(cue, backend);

                // Record read operation in storage metrics
                let bytes_read = results
                    .iter()
                    .map(|(ep, _)| {
                        std::mem::size_of::<Episode>() as u64
                            + ep.id.len() as u64
                            + ep.what.len() as u64
                    })
                    .sum::<u64>();
                self.storage_metrics.record_read(bytes_read);

                // TODO: Add metrics when monitoring feature is properly integrated
                #[cfg(feature = "monitoring")]
                {
                    // metrics.increment_counter("queries_executed_total", 1);
                    // metrics.observe_histogram("query_duration_seconds", start.elapsed().as_secs_f64());
                    // metrics.observe_histogram("query_result_count", results.len() as f64);
                }

                return results;
            }
        }

        // Fallback to in-memory recall
        let results = self.recall_in_memory(cue);

        #[cfg(feature = "monitoring")]
        {
            use crate::metrics;
            metrics::increment_counter("queries_executed_total", 1);
            metrics::observe_histogram("query_duration_seconds", start.elapsed().as_secs_f64());
            if let Ok(result_len) = u64::try_from(results.len()) {
                metrics::observe_histogram("query_result_count", u64_to_f64(result_len));
            }
        }

        results
    }

    /// Recall with persistent backend integration
    #[cfg(feature = "memory_mapped_persistence")]
    fn recall_with_persistence(
        &self,
        cue: &Cue,
        backend: &Arc<CognitiveTierArchitecture>,
    ) -> Vec<(Episode, Confidence)> {
        // Search across all tiers (hot/warm/cold) using the tier architecture
        // We need to run async code from a sync context
        let tier_results = if let Ok(_handle) = tokio::runtime::Handle::try_current() {
            // We're already in a tokio runtime
            // Spawn blocking task to avoid nested runtime issues
            std::thread::scope(|s| {
                let backend = backend.clone();
                let cue = cue.clone();
                let thread_handle = s.spawn(move || {
                    // Create a new runtime in this thread
                    let rt = tokio::runtime::Builder::new_current_thread()
                        .enable_all()
                        .build()
                        .expect("Failed to create runtime");
                    rt.block_on(backend.recall(&cue))
                });
                thread_handle.join().expect("Thread panicked")
            })
        } else {
            // No runtime available, create a new one
            let runtime = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build();

            match runtime {
                Ok(rt) => rt.block_on(backend.recall(cue)),
                Err(e) => {
                    tracing::warn!(error = ?e, "Failed to create runtime for tier search, falling back to in-memory");
                    return self.recall_in_memory(cue);
                }
            }
        };

        match tier_results {
            Ok(results) => {
                // If tier search succeeded, return those results
                if !results.is_empty() {
                    return results;
                }

                // If tier search returned no results, fallback to in-memory
                tracing::debug!("Tier search returned no results, falling back to in-memory");
                self.recall_in_memory(cue)
            }
            Err(e) => {
                // On error, fallback to in-memory recall
                tracing::warn!(error = ?e, "Tier search failed, falling back to in-memory");
                self.recall_in_memory(cue)
            }
        }
    }

    /// Calculate confidence score for an episode matching a cue
    /// Reserved for future confidence calibration during tier-based recall
    #[cfg(feature = "memory_mapped_persistence")]
    #[allow(dead_code)]
    fn calculate_episode_confidence(episode: &Episode, cue: &Cue) -> Confidence {
        match &cue.cue_type {
            CueType::Embedding {
                vector,
                threshold: _,
            } => {
                let similarity = cosine_similarity(&episode.embedding, vector);
                Confidence::exact(similarity)
            }
            CueType::Context {
                confidence_threshold,
                ..
            } => {
                // Simple context matching based on text similarity
                *confidence_threshold
            }
            CueType::Semantic {
                content,
                fuzzy_threshold: threshold,
            } => {
                // Simple text matching
                let content_similarity = if episode.what.contains(content) {
                    0.9
                } else if episode
                    .what
                    .to_lowercase()
                    .contains(&content.to_lowercase())
                {
                    0.7
                } else {
                    threshold.raw() / 2.0 // Below threshold but non-zero
                };
                Confidence::exact(content_similarity)
            }
            CueType::Temporal {
                confidence_threshold,
                ..
            } => {
                // Basic temporal confidence
                *confidence_threshold
            }
        }
    }

    /// In-memory recall implementation
    fn recall_in_memory(&self, cue: &Cue) -> Vec<(Episode, Confidence)> {
        // Main orchestration method - delegates to focused methods

        // 1. Try HNSW index for embedding cues if available
        #[cfg(feature = "hnsw_index")]
        {
            if let CueType::Embedding { vector, threshold } = &cue.cue_type {
                if let Some(ref hnsw) = self.hnsw_index {
                    return self.recall_with_hnsw(cue, hnsw, vector, *threshold);
                }
            }
        }

        // 2. Get episodes and apply cue-specific filtering
        let episodes = self.get_episodes_from_buffer();
        let mut results = Self::apply_cue_filtering(episodes, cue);

        // 3. Apply spreading activation
        results = self.apply_spreading_activation(results, cue);

        // 4. Sort and limit results
        Self::finalize_results(results, cue.max_results)
    }

    /// Recall using similarity only, without spreading activation
    /// Used as fallback when spreading fails or times out
    fn recall_similarity_only(&self, cue: &Cue) -> Vec<(Episode, Confidence)> {
        // Try HNSW index for embedding cues if available
        #[cfg(feature = "hnsw_index")]
        {
            if let CueType::Embedding { vector, threshold } = &cue.cue_type {
                if let Some(ref hnsw) = self.hnsw_index {
                    return self.recall_with_hnsw(cue, hnsw, vector, *threshold);
                }
            }
        }

        // Fallback to basic filtering without spreading
        let episodes = self.get_episodes_from_buffer();
        let results = Self::apply_cue_filtering(episodes, cue);
        Self::finalize_results(results, cue.max_results)
    }

    /// Extract all episodes from the WAL buffer and hot memories
    fn get_episodes_from_buffer(&self) -> Vec<(String, Episode)> {
        let mut episodes = Vec::new();

        // Get episodes from WAL buffer
        for entry in self.wal_buffer.iter() {
            episodes.push((entry.key().clone(), entry.value().clone()));
        }

        // Also convert hot memories to episodes for recall
        // This ensures deduplication doesn't prevent recall
        for entry in &self.hot_memories {
            let memory = entry.value();
            let episode = Episode::new(
                memory.id.clone(),
                memory.created_at,
                memory
                    .content
                    .clone()
                    .unwrap_or_else(|| format!("Memory {}", memory.id)),
                memory.embedding,
                memory.confidence,
            );
            // Only add if not already in WAL buffer
            if !self.wal_buffer.contains_key(&memory.id) {
                episodes.push((memory.id.clone(), episode));
            }
        }

        episodes
    }

    /// Apply cue-specific filtering to episodes
    fn apply_cue_filtering(
        episodes: Vec<(String, Episode)>,
        cue: &Cue,
    ) -> Vec<(Episode, Confidence)> {
        match &cue.cue_type {
            CueType::Embedding { vector, threshold } => {
                Self::filter_by_embedding(episodes, vector, *threshold)
            }
            CueType::Context {
                time_range,
                location,
                confidence_threshold,
            } => Self::filter_by_context(
                episodes,
                time_range.as_ref(),
                location.as_deref(),
                *confidence_threshold,
            ),
            CueType::Semantic {
                content,
                fuzzy_threshold,
            } => Self::filter_by_semantic(episodes, content, *fuzzy_threshold),
            CueType::Temporal {
                pattern,
                confidence_threshold,
            } => Self::filter_by_temporal(episodes, pattern, *confidence_threshold),
        }
    }

    /// Filter episodes by embedding similarity
    fn filter_by_embedding(
        episodes: Vec<(String, Episode)>,
        vector: &[f32; 768],
        threshold: Confidence,
    ) -> Vec<(Episode, Confidence)> {
        let mut results = Vec::new();

        for (_id, episode) in episodes {
            let similarity = cosine_similarity(&episode.embedding, vector);
            let confidence = Confidence::exact(similarity);

            if confidence.raw() >= threshold.raw() {
                results.push((episode, confidence));
            }
        }

        results
    }

    /// Filter episodes by context (time range and location)
    fn filter_by_context(
        episodes: Vec<(String, Episode)>,
        time_range: Option<&(chrono::DateTime<chrono::Utc>, chrono::DateTime<chrono::Utc>)>,
        location: Option<&str>,
        confidence_threshold: Confidence,
    ) -> Vec<(Episode, Confidence)> {
        let mut results = Vec::new();

        for (_id, episode) in episodes {
            let (match_score, max_score) =
                Self::calculate_context_score(&episode, time_range, location);

            if max_score > 0.0 {
                let confidence_value = match_score / max_score;
                let confidence = Confidence::exact(confidence_value);

                if confidence.raw() >= confidence_threshold.raw() {
                    results.push((episode, confidence));
                }
            }
        }

        results
    }

    /// Calculate context matching score for an episode
    fn calculate_context_score(
        episode: &Episode,
        time_range: Option<&(chrono::DateTime<chrono::Utc>, chrono::DateTime<chrono::Utc>)>,
        location: Option<&str>,
    ) -> (f32, f32) {
        let mut match_score = 0.0;
        let mut max_score = 0.0;

        // Check time range if provided
        if let Some((start, end)) = time_range {
            max_score += 1.0;
            if episode.when >= *start && episode.when <= *end {
                match_score += 1.0;
            }
        }

        // Check location if provided
        if let Some(loc) = location {
            max_score += 1.0;
            if let Some(ep_loc) = &episode.where_location {
                if ep_loc.contains(loc) || loc.contains(ep_loc) {
                    match_score += 1.0;
                }
            }
        }

        (match_score, max_score)
    }

    /// Filter episodes by semantic content
    fn filter_by_semantic(
        episodes: Vec<(String, Episode)>,
        content: &str,
        fuzzy_threshold: Confidence,
    ) -> Vec<(Episode, Confidence)> {
        let mut results = Vec::new();

        for (_id, episode) in episodes {
            let match_score = Self::calculate_semantic_similarity(&episode.what, content);

            if match_score > 0.0 {
                let confidence = Confidence::exact(match_score);

                if confidence.raw() >= fuzzy_threshold.raw() {
                    results.push((episode, confidence));
                }
            }
        }

        results
    }

    /// Calculate semantic similarity between episode content and query
    fn calculate_semantic_similarity(episode_content: &str, query: &str) -> f32 {
        // Exact substring match
        if episode_content
            .to_lowercase()
            .contains(&query.to_lowercase())
        {
            return 1.0;
        }

        // Reverse substring match
        if query
            .to_lowercase()
            .contains(&episode_content.to_lowercase())
        {
            return 0.7;
        }

        // Word overlap calculation
        let query_words: Vec<&str> = query.split_whitespace().collect();
        let episode_words: Vec<&str> = episode_content.split_whitespace().collect();
        let matches = query_words
            .iter()
            .filter(|w| episode_words.contains(w))
            .count();

        if matches > 0 {
            let matches_u64 = u64::try_from(matches).unwrap_or(u64::MAX);
            let denominator_u64 = u64::try_from(query_words.len().max(1)).unwrap_or(u64::MAX);
            let overlap_ratio = unit_ratio_to_f32(matches_u64, denominator_u64);
            overlap_ratio * 0.5
        } else {
            0.0
        }
    }

    /// Filter episodes by temporal pattern
    fn filter_by_temporal(
        episodes: Vec<(String, Episode)>,
        pattern: &TemporalPattern,
        confidence_threshold: Confidence,
    ) -> Vec<(Episode, Confidence)> {
        let mut results = Vec::new();

        for (_id, episode) in episodes {
            if Self::matches_temporal_pattern(&episode, pattern) {
                results.push((episode, confidence_threshold));
            }
        }

        results
    }

    /// Check if episode matches temporal pattern
    fn matches_temporal_pattern(episode: &Episode, pattern: &TemporalPattern) -> bool {
        match pattern {
            TemporalPattern::Before(time) => episode.when < *time,
            TemporalPattern::After(time) => episode.when > *time,
            TemporalPattern::Between(start, end) => episode.when >= *start && episode.when <= *end,
            TemporalPattern::Recent(duration) => {
                let now = Utc::now();
                episode.when > now - *duration
            }
        }
    }

    /// Sort and limit results to requested max
    fn finalize_results(
        mut results: Vec<(Episode, Confidence)>,
        max_results: usize,
    ) -> Vec<(Episode, Confidence)> {
        // Sort by confidence (highest first)
        results.sort_by(|a, b| {
            b.1.raw()
                .partial_cmp(&a.1.raw())
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Limit results
        results.truncate(max_results);

        results
    }

    /// Complete a partial episode using pattern completion
    #[cfg(feature = "pattern_completion")]
    pub fn complete_pattern(&self, partial: &PartialEpisode) -> CompletedEpisode {
        #[cfg(feature = "monitoring")]
        let start = Instant::now();

        self.pattern_reconstructor.as_ref().map_or_else(
            || {
                Self::fallback_completion(
                    "Pattern completion not available",
                    "unavailable",
                    Confidence::exact(0.0),
                )
            },
            |reconstructor| {
                let episodes: Vec<Episode> = self
                    .wal_buffer
                    .iter()
                    .map(|entry| entry.value().clone())
                    .collect();

                let mut reconstructor = reconstructor.write();
                reconstructor.update(&episodes);

                match reconstructor.complete(partial) {
                    Ok(completed) => {
                        #[cfg(feature = "monitoring")]
                        {
                            use crate::metrics;
                            metrics::increment_counter("pattern_completions_total", 1);
                            metrics::observe_histogram(
                                "pattern_completion_duration_seconds",
                                start.elapsed().as_secs_f64(),
                            );
                            metrics::record_cognitive(&CognitiveMetric::PatternCompletion {
                                plausibility: 0.8,
                                is_false_memory: false,
                            });
                        }

                        completed
                    }
                    Err(error) => {
                        tracing::warn!("Pattern completion failed: {:?}", error);
                        Self::fallback_completion(
                            "Failed pattern completion",
                            "failed",
                            Confidence::exact(0.1),
                        )
                    }
                }
            },
        )
    }

    #[cfg(feature = "pattern_completion")]
    fn fallback_completion(
        message: &str,
        prefix: &str,
        confidence: Confidence,
    ) -> CompletedEpisode {
        let now = chrono::Utc::now();
        let identifier = format!("{prefix}_{}", now.timestamp());
        CompletedEpisode {
            episode: Episode::new(identifier, now, message.to_string(), [0.0; 768], confidence),
            completion_confidence: confidence,
            source_attribution: crate::completion::SourceMap::default(),
            alternative_hypotheses: Vec::new(),
            metacognitive_confidence: confidence,
            activation_evidence: Vec::new(),
        }
    }

    /// Enable pattern completion with default configuration
    #[cfg(feature = "pattern_completion")]
    pub fn enable_pattern_completion(&mut self) {
        let config = CompletionConfig::default();
        let reconstructor = PatternReconstructor::new(config);
        self.pattern_reconstructor = Some(Arc::new(RwLock::new(reconstructor)));
    }

    /// Enable pattern completion with custom configuration
    #[cfg(feature = "pattern_completion")]
    pub fn enable_pattern_completion_with_config(&mut self, config: CompletionConfig) {
        let reconstructor = PatternReconstructor::new(config);
        self.pattern_reconstructor = Some(Arc::new(RwLock::new(reconstructor)));
    }

    /// Apply spreading activation to enhance recall results
    fn apply_spreading_activation(
        &self,
        mut results: Vec<(Episode, Confidence)>,
        cue: &Cue,
    ) -> Vec<(Episode, Confidence)> {
        // Get system pressure to modulate activation spreading
        let pressure = self.pressure();
        let spread_factor = pressure.mul_add(-0.5, 1.0); // Reduce spreading under pressure

        // For each high-confidence result, boost related memories
        let high_confidence_results: Vec<_> = results
            .iter()
            .filter(|(_, conf)| conf.is_high())
            .cloned()
            .collect();

        for (ep, base_conf) in high_confidence_results {
            // Find related episodes based on temporal proximity
            let time_window = chrono::Duration::hours(1);
            let start = ep.when - time_window;
            let end = ep.when + time_window;

            for entry in self.wal_buffer.iter() {
                let related_ep = entry.value();
                if related_ep.id != ep.id && related_ep.when >= start && related_ep.when <= end {
                    // Calculate activation boost based on temporal proximity
                    let time_diff = (ep.when - related_ep.when).num_seconds().unsigned_abs();
                    let max_diff_seconds =
                        std::cmp::max(time_window.num_seconds().unsigned_abs(), 1);
                    let ratio = unit_ratio_to_f32(time_diff, max_diff_seconds);
                    let proximity = (1.0_f32 - ratio).clamp(0.0, 1.0);
                    let boost = proximity * base_conf.raw() * spread_factor * 0.3;

                    // Check if already in results
                    let existing_idx = results.iter().position(|(e, _)| e.id == related_ep.id);

                    if let Some(idx) = existing_idx {
                        // Boost existing result
                        let (ep, old_conf) = &results[idx];
                        let new_value = (old_conf.raw() + boost).min(1.0);
                        let new_conf = Confidence::exact(new_value);
                        results[idx] = (ep.clone(), new_conf);
                    } else if boost > cue.result_threshold.raw() * 0.5 {
                        // Add as new low-confidence result
                        let conf = Confidence::exact(boost);
                        results.push((related_ep.clone(), conf));
                    }
                }
            }
        }

        results
    }

    /// Recall using HNSW index for fast similarity search
    #[cfg(feature = "hnsw_index")]
    fn recall_with_hnsw(
        &self,
        cue: &Cue,
        hnsw: &CognitiveHnswIndex,
        vector: &[f32; 768],
        threshold: Confidence,
    ) -> Vec<(Episode, Confidence)> {
        // Use HNSW for fast similarity search
        let candidates = hnsw.search_with_confidence(
            vector,
            cue.max_results * 2, // Get more candidates for diversity
            threshold,
        );

        let mut results = Vec::new();
        for (memory_id, confidence) in candidates {
            // Try WAL buffer first
            if let Some(episode) = self.wal_buffer.get(&memory_id) {
                results.push((episode.clone(), confidence));
            }
            // If not in WAL buffer, try converting from hot_memories
            else if let Some(memory) = self.hot_memories.get(&memory_id) {
                let episode = Episode::new(
                    memory.id.clone(),
                    memory.created_at,
                    memory
                        .content
                        .clone()
                        .unwrap_or_else(|| format!("Memory {}", memory.id)),
                    memory.embedding,
                    memory.confidence,
                );
                results.push((episode, confidence));
            }
        }

        // Apply spreading activation using HNSW graph structure
        let pressure = *self.pressure.read();
        results = hnsw.apply_spreading_activation(results, cue, pressure);

        // Sort and limit results
        results.sort_by(|a, b| {
            b.1.raw()
                .partial_cmp(&a.1.raw())
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        results.truncate(cue.max_results);
        results
    }

    /// Recall memory by content address
    pub fn recall_by_content(&self, embedding: &[f32; 768]) -> Option<Episode> {
        let content_address = ContentAddress::from_embedding(embedding);

        // Look up memory ID from content index
        if let Some(memory_id) = self.content_index.get(&content_address) {
            // Retrieve memory from hot tier
            if let Some(memory) = self.hot_memories.get(&memory_id) {
                // Convert Memory to Episode
                let episode = Episode::new(
                    memory.id.clone(),
                    memory.created_at,
                    memory
                        .content
                        .clone()
                        .unwrap_or_else(|| format!("Memory {}", memory.id)),
                    memory.embedding,
                    memory.confidence,
                );
                return Some(episode);
            }
        }

        None
    }

    /// Find similar content using LSH buckets
    pub fn find_similar_content(
        &self,
        embedding: &[f32; 768],
        max_results: usize,
    ) -> Vec<(Episode, Confidence)> {
        let content_address = ContentAddress::from_embedding(embedding);

        // Find similar content addresses in same or nearby buckets
        let similar_addresses = self.content_index.find_nearby(&content_address, 5);

        let mut results = Vec::new();

        for addr in similar_addresses {
            if let Some(memory_id) = self.content_index.get(&addr) {
                if let Some(memory) = self.hot_memories.get(&memory_id) {
                    // Calculate actual similarity
                    let similarity =
                        crate::compute::cosine_similarity_768(embedding, &memory.embedding);

                    if similarity > 0.8 {
                        let episode = Episode::new(
                            memory.id.clone(),
                            memory.created_at,
                            memory
                                .content
                                .clone()
                                .unwrap_or_else(|| format!("Memory {}", memory.id)),
                            memory.embedding,
                            memory.confidence,
                        );

                        results.push((episode, Confidence::exact(similarity)));
                    }
                }
            }
        }

        // Sort by confidence
        results.sort_by(|a, b| {
            b.1.raw()
                .partial_cmp(&a.1.raw())
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        results.truncate(max_results);
        results
    }

    /// Get deduplication statistics
    pub fn deduplication_stats(&self) -> crate::storage::DeduplicationStats {
        self.deduplicator.read().stats()
    }

    /// Get content index statistics
    pub fn content_index_stats(&self) -> crate::storage::ContentIndexStats {
        self.content_index.stats()
    }

    /// Configure deduplication threshold
    pub fn set_deduplication_threshold(&self, threshold: f32) {
        self.deduplicator.write().set_threshold(threshold);
    }

    /// Configure deduplication merge strategy
    pub fn set_deduplication_strategy(&self, strategy: MergeStrategy) {
        self.deduplicator.write().set_strategy(strategy);
    }
}

/// Calculate cosine similarity between two vectors
fn cosine_similarity(a: &[f32; 768], b: &[f32; 768]) -> f32 {
    crate::compute::cosine_similarity_768(a, b)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Confidence, Cue, EpisodeBuilder, numeric};
    use chrono::Utc;
    use std::convert::TryFrom;
    use std::fmt::Debug;

    type TestResult<T = ()> = Result<T, String>;

    fn ensure(condition: bool, message: impl Into<String>) -> TestResult {
        if condition {
            Ok(())
        } else {
            Err(message.into())
        }
    }

    fn ensure_eq<T>(actual: &T, expected: &T, context: &str) -> TestResult
    where
        T: PartialEq + Debug,
    {
        if actual == expected {
            Ok(())
        } else {
            Err(format!("{context}: expected {expected:?}, got {actual:?}"))
        }
    }

    fn usize_to_f32(value: usize) -> f32 {
        debug_assert!(value < (1 << 24));
        let value_u64 = u64::try_from(value).unwrap_or(u64::MAX);
        numeric::saturating_f32_from_f64(numeric::u64_to_f64(value_u64))
    }

    fn usize_to_u64(value: usize) -> u64 {
        u64::try_from(value).unwrap_or(u64::MAX)
    }

    // Helper to create unique embeddings for tests
    fn create_test_embedding(seed: f32) -> [f32; 768] {
        use std::f32::consts::PI;

        let mut embedding = [0.0f32; 768];
        let freq = seed.mul_add(10.0, 1.0);
        let phase = seed * PI * 2.0;
        let amplitude_scale = (1.0 + seed).sqrt();

        // Create orthogonal basis vectors based on seed
        // Each seed creates a different direction in high-dimensional space
        for (idx, value) in embedding.iter_mut().enumerate() {
            let idx_f = usize_to_f32(idx);
            let base = phase.mul_add(1.0, idx_f * freq / 768.0);
            let first = base.sin();
            let second = phase.mul_add(1.5, idx_f * freq * 2.0 / 768.0).cos();
            let third = phase.mul_add(2.0, idx_f * freq * 3.0 / 768.0).sin();
            let mixed = third.mul_add(0.2, first.mul_add(0.5, second * 0.3));
            *value = mixed * amplitude_scale;
        }

        // Normalize to unit vector for consistent cosine similarity
        let norm = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for val in &mut embedding {
                *val /= norm;
            }
        }

        embedding
    }

    #[test]
    fn test_store_returns_activation() {
        let store = MemoryStore::new(10);

        let episode = EpisodeBuilder::new()
            .id("ep1".to_string())
            .when(Utc::now())
            .what("test episode".to_string())
            .embedding(create_test_embedding(0.1))
            .confidence(Confidence::HIGH)
            .build();

        let activation = store.store(episode);

        // Should return high activation with no pressure
        assert!(activation.value() > 0.8);
        assert!(activation.value() <= 1.0);
    }

    // test_store_never_panics removed as redundant with test_eviction_of_low_activation
    // The eviction behavior is already tested more thoroughly in the dedicated eviction test

    #[test]
    fn test_concurrent_stores_dont_block() -> TestResult {
        use std::sync::Arc;
        use std::thread;
        use std::time::{SystemTime, UNIX_EPOCH};

        let store = Arc::new(MemoryStore::new(100));
        let mut handles = vec![];

        // Spawn multiple threads storing concurrently
        for i in 0..10 {
            let store_clone = Arc::clone(&store);
            let handle = thread::spawn(move || {
                // Small delay to ensure unique timestamps
                let delay = usize_to_u64(i);
                thread::sleep(std::time::Duration::from_millis(delay));

                // Use thread ID + timestamp for truly unique embedding
                let elapsed = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap_or_else(|_| std::time::Duration::from_secs(0));
                let timestamp_secs = {
                    let secs = elapsed.as_secs_f64();
                    let nanos = f64::from(elapsed.subsec_nanos());
                    numeric::saturating_f32_from_f64(secs + nanos / 1_000_000_000.0)
                };
                let idx_f = usize_to_f32(i);
                let unique_seed = idx_f.mul_add(1000.0, timestamp_secs);

                let episode = EpisodeBuilder::new()
                    .id(format!("ep_thread_{i}"))
                    .when(Utc::now())
                    .what(format!("concurrent episode {i}"))
                    .embedding(create_test_embedding(unique_seed))
                    .confidence(Confidence::HIGH)
                    .build();

                store_clone.store(episode)
            });
            handles.push(handle);
        }

        // All stores should complete without blocking
        for handle in handles {
            let activation = handle
                .join()
                .map_err(|err| format!("store thread panicked: {err:?}"))?;
            ensure(activation.value() > 0.0, "activation should be positive")?;
        }

        // Should have stored 10 unique memories
        ensure_eq(&store.count(), &10_usize, "stored memory count")?;

        Ok(())
    }

    #[test]
    fn test_degraded_store_under_pressure() {
        let store = MemoryStore::new(5);

        // Fill store to capacity with unique memories
        for i in 0..5 {
            let seed = usize_to_f32(i);
            let episode = EpisodeBuilder::new()
                .id(format!("ep{i}"))
                .when(Utc::now())
                .what(format!("episode {i}"))
                .embedding(create_test_embedding(seed)) // Use seed derived from loop index for separation
                .confidence(Confidence::MEDIUM)
                .build();
            store.store(episode);
        }

        // Verify pressure is high
        assert!(
            store.pressure() > 0.6,
            "Pressure should be high when at capacity, got {}",
            store.pressure()
        );

        // Store under pressure should still work but with degraded activation
        let high_pressure_episode = EpisodeBuilder::new()
            .id("pressure_ep".to_string())
            .when(Utc::now())
            .what("high pressure episode".to_string())
            .embedding(create_test_embedding(0.99))
            .confidence(Confidence::HIGH)
            .build();

        let activation = store.store(high_pressure_episode);

        // Activation should be degraded due to pressure
        assert!(
            activation.value() < 0.5,
            "Activation should be degraded under pressure, got {}",
            activation.value()
        );
    }

    #[test]
    fn test_recall_returns_empty_for_no_matches() {
        let store = MemoryStore::new(10);

        // Store an episode
        let episode = EpisodeBuilder::new()
            .id("ep1".to_string())
            .when(Utc::now())
            .what("test episode".to_string())
            .embedding(create_test_embedding(0.1))
            .confidence(Confidence::HIGH)
            .build();

        store.store(episode);

        // Create cue that won't match
        let cue = Cue::semantic(
            "cue1".to_string(),
            "completely different content".to_string(),
            Confidence::HIGH,
        );

        let results = store.recall(&cue);

        // Should return empty vector, not error
        assert_eq!(results.len(), 0);
    }

    #[test]
    fn test_recall_with_embedding_cue() {
        let store = MemoryStore::new(10);

        // Store episodes with different embeddings
        let mut embedding1 = [0.0; 768];
        embedding1[0] = 1.0;

        let mut embedding2 = [0.0; 768];
        embedding2[1] = 1.0;

        let episode1 = EpisodeBuilder::new()
            .id("ep1".to_string())
            .when(Utc::now())
            .what("memory one".to_string())
            .embedding(embedding1)
            .confidence(Confidence::HIGH)
            .build();

        let episode2 = EpisodeBuilder::new()
            .id("ep2".to_string())
            .when(Utc::now())
            .what("memory two".to_string())
            .embedding(embedding2)
            .confidence(Confidence::HIGH)
            .build();

        store.store(episode1);
        store.store(episode2);

        // Search with embedding similar to first
        let cue = Cue::embedding("cue1".to_string(), embedding1, Confidence::exact(0.9));

        let results = store.recall(&cue);

        // Should find the matching episode
        assert!(!results.is_empty());
        assert_eq!(results[0].0.id, "ep1");
        assert!(results[0].1.raw() > 0.9);
    }

    #[test]
    fn test_recall_with_context_cue() {
        let store = MemoryStore::new(10);

        let now = Utc::now();
        let yesterday = now - chrono::Duration::days(1);
        let tomorrow = now + chrono::Duration::days(1);

        // Store episodes at different times
        let episode1 = EpisodeBuilder::new()
            .id("ep1".to_string())
            .when(now)
            .what("today's memory".to_string())
            .embedding(create_test_embedding(0.1))
            .confidence(Confidence::HIGH)
            .where_location("office".to_string())
            .build();

        let episode2 = EpisodeBuilder::new()
            .id("ep2".to_string())
            .when(yesterday)
            .what("yesterday's memory".to_string())
            .embedding(create_test_embedding(0.1))
            .confidence(Confidence::HIGH)
            .where_location("home".to_string())
            .build();

        store.store(episode1);
        store.store(episode2);

        // Search for today's memories in office
        let cue = Cue::context(
            "cue1".to_string(),
            Some((now - chrono::Duration::hours(1), tomorrow)),
            Some("office".to_string()),
            Confidence::MEDIUM,
        );

        let results = store.recall(&cue);

        // Should find only today's office memory
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0.id, "ep1");
        assert!((results[0].1.raw() - 1.0).abs() < 1e-6); // Perfect match on both criteria
    }

    #[test]
    fn test_recall_with_semantic_cue() {
        let store = MemoryStore::new(100);
        let now = Utc::now();

        // Store episodes with unique embeddings AND clear content
        let episode1 = EpisodeBuilder::new()
            .id("ep1".to_string())
            .when(now)
            .what("team standup meeting in the morning".to_string())
            .embedding(create_test_embedding(0.1))
            .confidence(Confidence::HIGH)
            .build();

        let episode2 = EpisodeBuilder::new()
            .id("ep2".to_string())
            .when(now - chrono::Duration::hours(2)) // Outside spreading activation window
            .what("lunch at the cafeteria with colleagues".to_string())
            .embedding(create_test_embedding(0.2))
            .confidence(Confidence::HIGH)
            .build();

        let episode3 = EpisodeBuilder::new()
            .id("ep3".to_string())
            .when(now)
            .what("project review meeting in conference room".to_string())
            .embedding(create_test_embedding(0.3))
            .confidence(Confidence::HIGH)
            .build();

        store.store(episode1);
        store.store(episode2);
        store.store(episode3);

        // Search for "meeting" should find ep1 and ep3
        let cue = Cue::semantic("cue1".to_string(), "meeting".to_string(), Confidence::LOW);

        let results = store.recall(&cue);

        // Should find both meeting-related episodes
        assert_eq!(results.len(), 2, "Should find both meeting episodes");
        let ids: Vec<String> = results.iter().map(|(e, _)| e.id.clone()).collect();
        assert!(ids.contains(&"ep1".to_string()));
        assert!(ids.contains(&"ep3".to_string()));
        assert!(!ids.contains(&"ep2".to_string()));
    }

    #[test]
    fn test_recall_confidence_normalization() {
        let store = MemoryStore::new(10);

        let episode = EpisodeBuilder::new()
            .id("ep1".to_string())
            .when(Utc::now())
            .what("test memory".to_string())
            .embedding(create_test_embedding(5.0))
            .confidence(Confidence::MEDIUM)
            .build();

        store.store(episode);

        // Create cue with similar but not identical embedding
        let stored_embedding = create_test_embedding(5.0);
        let mut partial_embedding = stored_embedding;
        // Modify it slightly to create partial match
        for value in partial_embedding.iter_mut().take(100) {
            *value *= 0.9;
        }
        // Renormalize
        let norm = partial_embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for val in &mut partial_embedding {
                *val /= norm;
            }
        }

        let cue = Cue::embedding("cue1".to_string(), partial_embedding, Confidence::LOW);

        let results = store.recall(&cue);

        // Should return with confidence in valid range
        assert!(!results.is_empty());
        let confidence = results[0].1.raw();
        assert!(confidence >= 0.0);
        assert!(confidence <= 1.0);
        assert!(confidence < 1.0); // Should be less than perfect match
    }

    #[test]
    fn test_concurrent_recall_doesnt_block() -> TestResult {
        use std::thread;

        let store = Arc::new(MemoryStore::new(100));

        // Store some episodes first
        for i in 0..20 {
            let factor = usize_to_f32(i) * 0.01;
            let episode = EpisodeBuilder::new()
                .id(format!("ep{i}"))
                .when(Utc::now())
                .what(format!("memory {i}"))
                .embedding([factor; 768])
                .confidence(Confidence::HIGH)
                .build();

            store.store(episode);
        }

        let mut handles = vec![];

        // Spawn multiple threads doing recalls concurrently
        for i in 0..10 {
            let store_clone = Arc::clone(&store);
            let handle = thread::spawn(move || {
                let cue = Cue::semantic(format!("cue{i}"), format!("memory {i}"), Confidence::LOW);

                store_clone.recall(&cue)
            });
            handles.push(handle);
        }

        // All recalls should complete without blocking
        for handle in handles {
            let results = handle
                .join()
                .map_err(|err| format!("recall thread panicked: {err:?}"))?;
            // Each should find at least one match
            ensure(!results.is_empty(), "recall results should not be empty")?;
        }

        Ok(())
    }

    #[test]
    fn test_spreading_activation() {
        let store = MemoryStore::new(10);

        let base_time = Utc::now();

        // Store related episodes close in time
        let episode1 = EpisodeBuilder::new()
            .id("ep1".to_string())
            .when(base_time)
            .what("meeting about project".to_string())
            .embedding(create_test_embedding(0.5))
            .confidence(Confidence::HIGH)
            .build();

        let episode2 = EpisodeBuilder::new()
            .id("ep2".to_string())
            .when(base_time + chrono::Duration::minutes(30))
            .what("discussion with John".to_string())
            .embedding(create_test_embedding(0.3))
            .confidence(Confidence::MEDIUM)
            .build();

        let episode3 = EpisodeBuilder::new()
            .id("ep3".to_string())
            .when(base_time + chrono::Duration::hours(2))
            .what("unrelated task".to_string())
            .embedding(create_test_embedding(0.1))
            .confidence(Confidence::LOW)
            .build();

        store.store(episode1);
        store.store(episode2);
        store.store(episode3);

        // Search for meeting - should also pull in related discussion
        let cue = Cue::semantic("cue1".to_string(), "meeting".to_string(), Confidence::LOW);

        let results = store.recall(&cue);

        // Should find meeting directly and possibly discussion through spreading
        assert!(!results.is_empty());
        assert_eq!(results[0].0.id, "ep1"); // Direct match should be first

        // If spreading activation worked, temporally related episode might be included
        // (This depends on implementation details and thresholds)
        if results.len() > 1 {
            // Related episode should have lower confidence due to spreading
            assert!(results[1].1.raw() < results[0].1.raw());
        }
    }

    #[test]
    fn test_eviction_of_low_activation() {
        let store = MemoryStore::new(3);

        // Store episodes with different confidence levels
        let low_conf = EpisodeBuilder::new()
            .id("low".to_string())
            .when(Utc::now())
            .what("low confidence episode".to_string())
            .embedding(create_test_embedding(1.0))
            .confidence(Confidence::LOW)
            .build();

        let med_conf = EpisodeBuilder::new()
            .id("med".to_string())
            .when(Utc::now())
            .what("medium confidence episode".to_string())
            .embedding(create_test_embedding(2.0))
            .confidence(Confidence::MEDIUM)
            .build();

        let high_conf = EpisodeBuilder::new()
            .id("high".to_string())
            .when(Utc::now())
            .what("high confidence episode".to_string())
            .embedding(create_test_embedding(3.0))
            .confidence(Confidence::HIGH)
            .build();

        store.store(low_conf);
        store.store(med_conf);
        store.store(high_conf);

        // Store one more to trigger eviction
        let new_episode = EpisodeBuilder::new()
            .id("new".to_string())
            .when(Utc::now())
            .what("new episode".to_string())
            .embedding(create_test_embedding(4.0))
            .confidence(Confidence::HIGH)
            .build();

        store.store(new_episode);

        // Low confidence memory should be evicted
        assert!(store.get("low").is_none());
        assert!(store.get("med").is_some());
        assert!(store.get("high").is_some());
        assert!(store.get("new").is_some());
    }
}
