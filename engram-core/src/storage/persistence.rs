//! Per-space persistence handle managing WAL, tier backend, and worker threads.

use crate::MemorySpaceId;
use crate::registry::SpaceDirectories;
use crate::storage::{
    CognitiveTierArchitecture, FsyncMode, RecoveryReport, StorageMetrics,
    wal::{WalReader, WalWriter},
};
use parking_lot::Mutex;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread::JoinHandle;
use thiserror::Error;

/// Configuration for per-space persistence.
#[derive(Clone, Debug)]
pub struct PersistenceConfig {
    /// Maximum number of memories in hot tier.
    pub hot_capacity: usize,
    /// Maximum number of memories in warm tier.
    pub warm_capacity: usize,
    /// Maximum number of memories in cold tier.
    pub cold_capacity: usize,
    /// File synchronization mode for WAL.
    pub fsync_mode: FsyncMode,
}

impl Default for PersistenceConfig {
    fn default() -> Self {
        Self {
            hot_capacity: 100_000,
            warm_capacity: 1_000_000,
            cold_capacity: 10_000_000,
            fsync_mode: FsyncMode::PerBatch,
        }
    }
}

/// Per-space persistence handle owning WAL writer, tier backend, and worker threads.
///
/// This struct ensures complete isolation of persistence resources across memory spaces,
/// preventing any cross-tenant contamination in WAL logs, tier storage, or background workers.
pub struct MemorySpacePersistence {
    space_id: MemorySpaceId,
    wal_writer: Arc<WalWriter>,
    storage_metrics: Arc<StorageMetrics>,
    tier_backend: Arc<CognitiveTierArchitecture>,
    tier_worker: Mutex<Option<JoinHandle<()>>>,
    tier_shutdown: Arc<AtomicBool>,
}

impl MemorySpacePersistence {
    /// Create a new persistence handle for the given memory space.
    ///
    /// # Errors
    ///
    /// Returns error if WAL writer or tier backend initialization fails.
    pub fn new(
        space_id: MemorySpaceId,
        config: &PersistenceConfig,
        directories: &SpaceDirectories,
    ) -> Result<Self, PersistenceError> {
        let storage_metrics = Arc::new(StorageMetrics::new());

        let wal_writer = WalWriter::new(
            &directories.wal,
            config.fsync_mode,
            Arc::clone(&storage_metrics),
        )
        .map_err(|e| PersistenceError::WalInit {
            space_id: space_id.clone(),
            path: directories.wal.clone(),
            source: e.into(),
        })?;

        let tier_backend = CognitiveTierArchitecture::new(
            &directories.root,
            config.hot_capacity,
            config.warm_capacity,
            config.cold_capacity,
            Arc::clone(&storage_metrics),
        )
        .map_err(|e| PersistenceError::TierInit {
            space_id: space_id.clone(),
            source: e.into(),
        })?;

        Ok(Self {
            space_id,
            wal_writer: Arc::new(wal_writer),
            storage_metrics,
            tier_backend: Arc::new(tier_backend),
            tier_worker: Mutex::new(None),
            tier_shutdown: Arc::new(AtomicBool::new(false)),
        })
    }

    /// Memory space identifier for this persistence handle.
    pub const fn space_id(&self) -> &MemorySpaceId {
        &self.space_id
    }

    /// Write-ahead log writer for this space.
    pub fn wal_writer(&self) -> Arc<WalWriter> {
        Arc::clone(&self.wal_writer)
    }

    /// Storage metrics for this space.
    pub fn storage_metrics(&self) -> Arc<StorageMetrics> {
        Arc::clone(&self.storage_metrics)
    }

    /// Tier backend for this space.
    pub fn tier_backend(&self) -> Arc<CognitiveTierArchitecture> {
        Arc::clone(&self.tier_backend)
    }

    /// Start background workers for tier migration.
    ///
    /// This spawns a thread that periodically migrates memories between tiers
    /// based on access patterns and capacity pressure.
    ///
    /// Note: Tier migration is currently managed at the MemoryStore level.
    /// This method is reserved for future per-space worker management.
    pub fn start_workers(&self) {
        tracing::debug!(space = %self.space_id, "persistence workers managed by MemoryStore");
    }

    /// Shutdown background workers.
    ///
    /// This method is idempotent and safe to call multiple times.
    ///
    /// Note: Worker shutdown is currently managed at the MemoryStore level.
    /// This method is reserved for future per-space worker management.
    pub fn shutdown(&self) {
        self.tier_shutdown.store(true, Ordering::Relaxed);

        let mut worker = self.tier_worker.lock();
        if let Some(handle) = worker.take() {
            drop(worker);
            let _ = handle.join();
        }

        tracing::debug!(space = %self.space_id, "persistence shutdown complete");
    }

    /// Recover from WAL by replaying entries.
    ///
    /// # Errors
    ///
    /// Returns error if WAL recovery fails or entries are corrupted.
    pub fn recover(&self, wal_dir: &Path) -> Result<RecoveryReport, PersistenceError> {
        let start = std::time::Instant::now();
        let reader = WalReader::new(wal_dir, Arc::clone(&self.storage_metrics));

        let entries = reader
            .scan_all()
            .map_err(|e| PersistenceError::WalRecovery {
                space_id: self.space_id.clone(),
                path: wal_dir.to_path_buf(),
                source: e.into(),
            })?;

        let recovered_entries = entries.len();
        let last_valid_sequence = entries.last().map_or(0, |e| e.header.sequence);

        tracing::info!(
            space = %self.space_id,
            recovered = recovered_entries,
            "WAL recovery completed"
        );

        Ok(RecoveryReport {
            recovered_entries,
            corrupted_entries: 0, // scan_all handles corrupt entries internally
            recovery_duration: start.elapsed(),
            last_valid_sequence,
        })
    }
}

impl Drop for MemorySpacePersistence {
    fn drop(&mut self) {
        self.shutdown();
    }
}

/// Errors that can occur during persistence operations.
#[derive(Error, Debug)]
pub enum PersistenceError {
    /// WAL writer initialization failed.
    #[error("failed to initialize WAL for space {space_id}: {source}")]
    WalInit {
        /// Memory space identifier
        space_id: MemorySpaceId,
        /// WAL directory path that failed
        path: std::path::PathBuf,
        /// Underlying error
        source: Box<dyn std::error::Error + Send + Sync>,
    },

    /// Tier backend initialization failed.
    #[error("failed to initialize tier backend for space {space_id}: {source}")]
    TierInit {
        /// Memory space identifier
        space_id: MemorySpaceId,
        /// Underlying error
        source: Box<dyn std::error::Error + Send + Sync>,
    },

    /// WAL recovery failed.
    #[error("failed to recover WAL for space {space_id} from {path:?}: {source}")]
    WalRecovery {
        /// Memory space identifier
        space_id: MemorySpaceId,
        /// WAL directory path that failed
        path: std::path::PathBuf,
        /// Underlying error
        source: Box<dyn std::error::Error + Send + Sync>,
    },
}
