use crate::storage::StorageError;
use crate::{MemorySpaceError, MemorySpaceId};
use thiserror::Error;

/// Errors emitted by the replication subsystem.
#[derive(Debug, Error)]
pub enum ReplicationError {
    /// Requested memory space does not exist on the local node.
    #[error("memory space '{0}' not found for replication")]
    MissingSpace(MemorySpaceId),
    /// General storage error surfaced while reading WAL files.
    #[error("replication storage error: {0}")]
    Storage(#[from] StorageError),
    /// Internal async task failed.
    #[error("replication worker failed: {0}")]
    Join(#[from] tokio::task::JoinError),
    /// Memory space registry error surfaced when resolving handles.
    #[error("replication registry error: {0}")]
    Registry(#[from] MemorySpaceError),
}
