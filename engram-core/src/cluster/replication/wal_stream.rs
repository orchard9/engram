use std::sync::Arc;

use crate::MemorySpaceId;
use crate::cluster::replication::ReplicationMetadata;
use crate::cluster::replication::{ReplicationBatch, ReplicationEntry, ReplicationError};
use crate::registry::{MemorySpaceError, MemorySpaceRegistry};
use crate::storage::{
    StorageMetrics,
    wal::{WalEntry, WalEntryType, WalReader},
};

/// Configuration for chunking WAL entries into replication batches.
#[derive(Debug, Clone, Copy)]
pub struct WalStreamerConfig {
    /// Maximum payload bytes per batch before forcing a flush.
    pub max_batch_bytes: usize,
    /// Maximum number of WAL entries per batch.
    pub max_batch_entries: usize,
}

impl Default for WalStreamerConfig {
    fn default() -> Self {
        Self {
            max_batch_bytes: 1024 * 1024, // 1MiB
            max_batch_entries: 512,
        }
    }
}

/// Helper that reads WAL entries for a memory space and emits replication batches.
pub struct WalStreamer {
    registry: Arc<MemorySpaceRegistry>,
    metadata: Arc<ReplicationMetadata>,
    config: WalStreamerConfig,
}

impl WalStreamer {
    /// Create a new WAL streamer backed by the provided registry + metadata trackers.
    #[must_use]
    pub const fn new(
        registry: Arc<MemorySpaceRegistry>,
        metadata: Arc<ReplicationMetadata>,
        config: WalStreamerConfig,
    ) -> Self {
        Self {
            registry,
            metadata,
            config,
        }
    }

    /// Collect batches for `space` starting at `start_sequence`.
    pub async fn collect_batches(
        &self,
        primary_id: &str,
        space: &MemorySpaceId,
        start_sequence: u64,
    ) -> Result<Vec<ReplicationBatch>, ReplicationError> {
        let handle = match self.registry.get(space) {
            Ok(handle) => handle,
            Err(MemorySpaceError::NotFound { .. }) => {
                return Err(ReplicationError::MissingSpace(space.clone()));
            }
            Err(err) => return Err(ReplicationError::Registry(err)),
        };

        let wal_dir = handle.directories().wal.clone();
        let metrics = Arc::new(StorageMetrics::new());
        let entries = tokio::task::spawn_blocking(move || {
            let reader = WalReader::new(wal_dir, metrics);
            reader.scan_all()
        })
        .await??;

        let mut current_batch = Vec::new();
        let mut current_bytes = 0usize;
        let mut start_seq = None;
        let mut batches = Vec::new();

        for entry in entries
            .into_iter()
            .filter(|e| e.header.sequence >= start_sequence)
        {
            self.metadata.record_local_seq(space, entry.header.sequence);
            let entry_bytes =
                entry.payload.len() + std::mem::size_of::<crate::storage::wal::WalEntryHeader>();
            let replication_entry = convert_entry(entry);

            if start_seq.is_none() {
                start_seq = Some(replication_entry.sequence);
            }

            current_bytes += entry_bytes;
            current_batch.push(replication_entry);

            let batch_full = current_bytes >= self.config.max_batch_bytes
                || current_batch.len() >= self.config.max_batch_entries;
            if batch_full {
                flush_batch(
                    space,
                    primary_id,
                    &mut current_batch,
                    &mut current_bytes,
                    &mut start_seq,
                    &mut batches,
                );
            }
        }

        if !current_batch.is_empty() {
            flush_batch(
                space,
                primary_id,
                &mut current_batch,
                &mut current_bytes,
                &mut start_seq,
                &mut batches,
            );
        }

        Ok(batches)
    }
}

fn convert_entry(entry: WalEntry) -> ReplicationEntry {
    ReplicationEntry {
        sequence: entry.header.sequence,
        entry_type: WalEntryType::from(entry.header.entry_type),
        payload: entry.payload,
    }
}

fn flush_batch(
    space: &MemorySpaceId,
    primary_id: &str,
    current_batch: &mut Vec<ReplicationEntry>,
    current_bytes: &mut usize,
    start_seq: &mut Option<u64>,
    batches: &mut Vec<ReplicationBatch>,
) {
    if current_batch.is_empty() {
        return;
    }

    let start_sequence = start_seq.unwrap_or_else(|| current_batch[0].sequence);
    let end_sequence = current_batch
        .last()
        .map_or(start_sequence, |entry| entry.sequence);
    let checksum = batch_checksum(current_batch);
    let batch = ReplicationBatch {
        space: space.clone(),
        primary_id: primary_id.to_string(),
        start_sequence,
        end_sequence,
        entries: std::mem::take(current_batch),
        checksum,
    };
    *current_bytes = 0;
    *start_seq = None;
    batches.push(batch);
}

const FNV_OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x1000_0000_01b3;

fn batch_checksum(entries: &[ReplicationEntry]) -> u64 {
    let mut hash = FNV_OFFSET_BASIS;
    for entry in entries {
        hash ^= entry.sequence;
        hash = hash.wrapping_mul(FNV_PRIME);
        hash ^= entry.entry_type as u64;
        hash = hash.wrapping_mul(FNV_PRIME);
        hash ^= entry.payload.len() as u64;
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}
