use crate::MemorySpaceId;
use crate::storage::wal::WalEntryType;

/// Entry replicated to a remote node.
#[derive(Debug, Clone)]
pub struct ReplicationEntry {
    /// Monotonic WAL sequence number.
    pub sequence: u64,
    /// Entry type stored in the WAL.
    pub entry_type: WalEntryType,
    /// Raw payload copied from the WAL.
    pub payload: Vec<u8>,
}

/// Batch of WAL entries destined for a replica.
#[derive(Debug, Clone)]
pub struct ReplicationBatch {
    /// Memory space identifier for this batch.
    pub space: MemorySpaceId,
    /// Node identifier for the primary generating the batch.
    pub primary_id: String,
    /// Inclusive starting sequence number for the batch.
    pub start_sequence: u64,
    /// Inclusive ending sequence number for the batch.
    pub end_sequence: u64,
    /// Serialized entries that must be applied on the replica.
    pub entries: Vec<ReplicationEntry>,
    /// Lightweight checksum derived from the entry metadata for validation.
    pub checksum: u64,
}

impl ReplicationBatch {
    /// Total number of entries carried by the batch.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the batch contains any entries.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Approximate number of payload bytes carried in the batch.
    #[must_use]
    pub fn payload_bytes(&self) -> usize {
        self.entries.iter().map(|entry| entry.payload.len()).sum()
    }
}
