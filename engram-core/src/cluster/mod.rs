//! Cluster coordination primitives (membership, discovery, configuration).

/// Anti-entropy sync scaffolding triggered after partitions heal.
pub mod anti_entropy;
/// Cached placement helpers tracking assignment versions.
pub mod assignment;
/// Partition-aware confidence helpers.
pub mod confidence;
/// Declarative configuration schema shared by CLI and runtime components.
pub mod config;
/// Node discovery adapters (static seed lists, DNS SRV, registries).
pub mod discovery;
/// Error types surfaced by cluster subsystems.
pub mod error;
/// Lock-free SWIM membership engine with gossip support.
pub mod membership;
/// Partition detection and monitoring utilities.
pub mod partition;
/// Deterministic placement planner for memory spaces.
pub mod placement;
/// Rebalance coordinator reacting to membership updates.
pub mod rebalance;
/// WAL-driven replication helpers (requires persistence).
#[cfg(feature = "memory_mapped_persistence")]
pub mod replication;
/// UDP transport for SWIM message exchange.
pub mod transport;
/// Split-brain detection via vector clocks.
pub mod vector_clock;

pub use anti_entropy::AntiEntropySync;
pub use assignment::{
    AssignmentSnapshot, CachedAssignment, NodeAssignmentLoad, SpaceAssignmentManager,
};
pub use confidence::PartitionAwareConfidence;
pub use error::ClusterError;
pub use membership::{
    MembershipStats, MembershipUpdate, NodeInfo, NodeState, ProbePlan, ProbeResult, SwimHandle,
    SwimMembership, SwimMessage, SwimObserver, SwimRuntime,
};
pub use partition::{PartitionDetector, PartitionState};
pub use placement::{SpaceAssignment, SpaceAssignmentPlanner};
pub use rebalance::{MigrationPlan, MigrationReason, RebalanceCoordinator, RebalanceStatus};
#[cfg(feature = "memory_mapped_persistence")]
pub use replication::{
    ReplicaLag, ReplicationBatch, ReplicationEntry, ReplicationError, ReplicationMetadata,
    ReplicationSpaceSummary, WalStreamer, WalStreamerConfig,
};
pub use transport::SwimTransport;
pub use vector_clock::{CausalOrdering, SplitBrainDetector, VectorClock};
