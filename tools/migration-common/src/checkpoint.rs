//! Checkpoint management for resumable migrations

use crate::error::MigrationResult;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// Migration checkpoint for resume capability
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Checkpoint {
    /// Last processed record ID
    pub last_processed_id: String,
    /// Number of records migrated so far
    pub records_migrated: u64,
    /// Timestamp when checkpoint was created
    pub timestamp: DateTime<Utc>,
    /// Path to checkpoint file
    #[serde(skip)]
    pub checkpoint_file: PathBuf,
}

impl Checkpoint {
    /// Create a new checkpoint
    #[must_use]
    pub fn new(last_processed_id: String, records_migrated: u64, checkpoint_file: PathBuf) -> Self {
        Self {
            last_processed_id,
            records_migrated,
            timestamp: Utc::now(),
            checkpoint_file,
        }
    }

    /// Save checkpoint to disk
    pub fn save(&self) -> MigrationResult<()> {
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(&self.checkpoint_file, json)?;
        tracing::info!(
            path = ?self.checkpoint_file,
            records = self.records_migrated,
            "Checkpoint saved"
        );
        Ok(())
    }

    /// Load checkpoint from disk
    pub fn load(path: &Path) -> MigrationResult<Self> {
        let json = std::fs::read_to_string(path)?;
        let mut checkpoint: Checkpoint = serde_json::from_str(&json)?;
        checkpoint.checkpoint_file = path.to_path_buf();
        tracing::info!(
            path = ?path,
            records = checkpoint.records_migrated,
            last_id = checkpoint.last_processed_id,
            "Checkpoint loaded"
        );
        Ok(checkpoint)
    }
}

/// Manages checkpoint creation and recovery
pub struct CheckpointManager {
    checkpoint_file: PathBuf,
    checkpoint_interval: u64,
    last_checkpoint: Option<Checkpoint>,
}

impl CheckpointManager {
    /// Create a new checkpoint manager
    #[must_use]
    pub fn new(checkpoint_file: PathBuf, checkpoint_interval: u64) -> Self {
        Self {
            checkpoint_file,
            checkpoint_interval,
            last_checkpoint: None,
        }
    }

    /// Load existing checkpoint if available
    pub fn load_existing(&mut self) -> MigrationResult<Option<Checkpoint>> {
        if self.checkpoint_file.exists() {
            let checkpoint = Checkpoint::load(&self.checkpoint_file)?;
            self.last_checkpoint = Some(checkpoint.clone());
            Ok(Some(checkpoint))
        } else {
            Ok(None)
        }
    }

    /// Check if a checkpoint should be saved based on interval
    #[must_use]
    pub fn should_checkpoint(&self, records_migrated: u64) -> bool {
        if let Some(ref last) = self.last_checkpoint {
            records_migrated - last.records_migrated >= self.checkpoint_interval
        } else {
            records_migrated >= self.checkpoint_interval
        }
    }

    /// Save a checkpoint
    pub fn save_checkpoint(
        &mut self,
        last_processed_id: String,
        records_migrated: u64,
    ) -> MigrationResult<()> {
        let checkpoint = Checkpoint::new(
            last_processed_id,
            records_migrated,
            self.checkpoint_file.clone(),
        );
        checkpoint.save()?;
        self.last_checkpoint = Some(checkpoint);
        Ok(())
    }
}
