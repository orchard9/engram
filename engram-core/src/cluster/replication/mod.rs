//! Replication helpers for streaming WAL entries to cluster replicas.
//!
//! The implementation lives behind the `memory_mapped_persistence` feature flag because
//! it depends on the persistent write-ahead log for each memory space.

mod batch;
mod error;
mod metadata;
mod wal_stream;

pub use batch::{ReplicationBatch, ReplicationEntry};
pub use error::ReplicationError;
pub use metadata::{ReplicaLag, ReplicationMetadata, ReplicationSpaceSummary};
pub use wal_stream::{WalStreamer, WalStreamerConfig};
