use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use dashmap::DashMap;

use crate::MemorySpaceId;

/// Tracks replication progress for each memory space and replica.
pub struct ReplicationMetadata {
    local_sequences: DashMap<MemorySpaceId, Arc<AtomicU64>>,
    replica_sequences: DashMap<(MemorySpaceId, String), Arc<AtomicU64>>,
}

impl ReplicationMetadata {
    /// Create a new metadata tracker.
    #[must_use]
    pub fn new() -> Self {
        Self {
            local_sequences: DashMap::new(),
            replica_sequences: DashMap::new(),
        }
    }

    fn local_entry(&self, space: &MemorySpaceId) -> Arc<AtomicU64> {
        self.local_sequences
            .entry(space.clone())
            .or_insert_with(|| Arc::new(AtomicU64::new(0)))
            .clone()
    }

    fn replica_entry(&self, space: &MemorySpaceId, replica: &str) -> Arc<AtomicU64> {
        self.replica_sequences
            .entry((space.clone(), replica.to_string()))
            .or_insert_with(|| Arc::new(AtomicU64::new(0)))
            .clone()
    }

    /// Record the latest local sequence observed for a space.
    pub fn record_local_seq(&self, space: &MemorySpaceId, sequence: u64) {
        let entry = self.local_entry(space);
        entry.store(sequence, Ordering::Release);
    }

    /// Record the latest sequence replicated to a remote node.
    pub fn record_replica_seq(&self, space: &MemorySpaceId, replica: &str, sequence: u64) {
        let entry = self.replica_entry(space, replica);
        entry.store(sequence, Ordering::Release);
    }

    /// Compute the lag between local and replica sequences (if known).
    #[must_use]
    pub fn replica_lag(&self, space: &MemorySpaceId, replica: &str) -> Option<ReplicaLag> {
        let local_seq = {
            let guard = self.local_sequences.get(space)?;
            guard.load(Ordering::Acquire)
        };
        let replica_key = (space.clone(), replica.to_string());
        let remote_seq = {
            let guard = self.replica_sequences.get(&replica_key)?;
            guard.load(Ordering::Acquire)
        };
        Some(ReplicaLag {
            space: space.clone(),
            replica: replica.to_string(),
            local_sequence: local_seq,
            replica_sequence: remote_seq,
        })
    }

    /// Snapshot replication state for diagnostics.
    #[must_use]
    pub fn snapshot(&self) -> Vec<ReplicationSpaceSummary> {
        let mut summaries = Vec::new();
        for entry in &self.local_sequences {
            let space_id = entry.key().clone();
            let local_seq = entry.value().load(Ordering::Acquire);
            let mut replicas = Vec::new();
            for replica_entry in self
                .replica_sequences
                .iter()
                .filter(|r| r.key().0 == space_id)
            {
                let remote_seq = replica_entry.value().load(Ordering::Acquire);
                replicas.push(ReplicaLag {
                    space: space_id.clone(),
                    replica: replica_entry.key().1.clone(),
                    local_sequence: local_seq,
                    replica_sequence: remote_seq,
                });
            }
            summaries.push(ReplicationSpaceSummary {
                space: space_id,
                local_sequence: local_seq,
                replicas,
            });
        }
        summaries
    }
}

impl Default for ReplicationMetadata {
    fn default() -> Self {
        Self::new()
    }
}

/// Lag snapshot for a particular replica.
#[derive(Debug, Clone)]
pub struct ReplicaLag {
    /// Memory space identifier.
    pub space: MemorySpaceId,
    /// Remote node identifier.
    pub replica: String,
    /// Latest sequence observed on the primary.
    pub local_sequence: u64,
    /// Latest sequence confirmed on the replica.
    pub replica_sequence: u64,
}

impl ReplicaLag {
    /// Difference between the local and replica sequences.
    #[must_use]
    pub const fn sequences_behind(&self) -> u64 {
        self.local_sequence.saturating_sub(self.replica_sequence)
    }
}

/// Summary structure surfaced via diagnostics/health endpoints.
#[derive(Debug, Clone)]
pub struct ReplicationSpaceSummary {
    /// Memory space identifier.
    pub space: MemorySpaceId,
    /// Latest sequence persisted locally.
    pub local_sequence: u64,
    /// Replica lag information for this space.
    pub replicas: Vec<ReplicaLag>,
}

#[cfg(test)]
mod tests {
    #![allow(clippy::expect_used, clippy::unwrap_used)]

    use super::*;
    use crate::MemorySpaceId;

    #[test]
    fn metadata_tracks_lag() {
        let metadata = ReplicationMetadata::new();
        let space = MemorySpaceId::try_from("alpha").unwrap();
        metadata.record_local_seq(&space, 10);
        metadata.record_replica_seq(&space, "node-b", 5);
        let lag = metadata
            .replica_lag(&space, "node-b")
            .expect("lag entry should exist");
        assert_eq!(lag.local_sequence, 10);
        assert_eq!(lag.replica_sequence, 5);
        assert_eq!(lag.sequences_behind(), 5);
    }
}
