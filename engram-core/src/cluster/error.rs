use super::vector_clock::VectorClock;
use crate::MemorySpaceId;
use thiserror::Error;

/// Errors that occur while running the SWIM membership protocol.
#[derive(Debug, Error)]
pub enum ClusterError {
    /// Underlying network I/O failure.
    #[error("cluster transport error: {0}")]
    Io(#[from] std::io::Error),
    /// Serialization or deserialization failure for gossip payloads.
    #[error("cluster serialization error: {0}")]
    Serialization(String),
    /// Payload exceeded the configured MTU budget.
    #[error("cluster message too large: {0} bytes")]
    MessageTooLarge(usize),
    /// Too few healthy peers are available for the requested operation.
    #[error(
        "insufficient healthy nodes for replication (required {required}, available {available})"
    )]
    InsufficientHealthyNodes {
        /// Nodes needed to satisfy the request (primary + replicas).
        required: usize,
        /// Alive nodes currently available (including the local node).
        available: usize,
    },
    /// Local node attempted to serve a space it does not own.
    #[error(
        "memory space {space} is owned by {owner}; local node {local} must route the write remotely"
    )]
    NotPrimary {
        /// Memory space identifier.
        space: MemorySpaceId,
        /// Node assigned as the primary owner.
        owner: String,
        /// Identifier for the local node.
        local: String,
    },
    /// Generic configuration issue preventing cluster startup.
    #[error("cluster configuration error: {0}")]
    Configuration(String),
    /// Cluster is partitioned from the majority of nodes.
    #[error("cluster partitioned from majority (reachable {reachable_nodes} / {total_nodes})")]
    Partitioned {
        /// Nodes reachable from this partition.
        reachable_nodes: usize,
        /// Total nodes tracked at the time of detection.
        total_nodes: usize,
    },
    /// Concurrent primaries detected for a memory space.
    #[error(
        "split-brain detected for space {space_id}: local_clock={:?}, remote_clock={:?}",
        local_clock,
        remote_clock
    )]
    SplitBrain {
        /// Memory space that experienced conflicting primaries.
        space_id: MemorySpaceId,
        /// Local vector clock snapshot.
        local_clock: VectorClock,
        /// Remote vector clock snapshot.
        remote_clock: VectorClock,
    },
}
