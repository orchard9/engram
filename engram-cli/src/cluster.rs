use std::{
    net::{SocketAddr, UdpSocket},
    str::FromStr,
    sync::Arc,
    time::Instant,
};

use anyhow::{Context, Result, anyhow};
use engram_core::MemorySpaceId;
#[cfg(feature = "memory_mapped_persistence")]
use engram_core::cluster::ReplicationMetadata;
use engram_core::cluster::{
    AssignmentSnapshot, ClusterError, PartitionDetector, PartitionState, SpaceAssignment,
    SpaceAssignmentManager, SpaceAssignmentPlanner, SplitBrainDetector,
    config::{ClusterConfig, DiscoveryConfig, ReplicationConfig},
    discovery::{DiscoveryError, build_discovery},
    membership::{NodeInfo, SwimMembership},
    rebalance::{MigrationPlan, RebalanceCoordinator, RebalanceStatus},
};
use tokio::sync::mpsc;
use tracing::{info, warn};
use uuid::Uuid;

/// Runtime cluster context used by the CLI/server entrypoint.
pub enum ClusterContext {
    /// Single-node mode with no membership/discovery.
    SingleNode,
    /// Distributed mode with SWIM membership state.
    Distributed {
        /// Stable node identifier.
        node_id: String,
        /// Membership engine keeping gossip state alive.
        membership: Arc<SwimMembership>,
        /// Cached assignment manager used for routing decisions.
        assignments: Arc<SpaceAssignmentManager>,
        /// Replication configuration applied to placements.
        replication: ReplicationConfig,
        /// Most recent set of discovered seed nodes.
        seeds: Vec<SocketAddr>,
        /// Address the SWIM transport is bound to.
        swim_addr: SocketAddr,
        /// Partition detector shared with API/routers.
        partition_detector: Arc<PartitionDetector>,
        /// Split-brain detector used to guard writes.
        split_brain: Arc<SplitBrainDetector>,
        /// Rebalance coordinator subscribed to membership updates.
        rebalance: Arc<RebalanceCoordinator>,
        /// Channel of queued migration plans for background orchestration.
        migration_rx: mpsc::Receiver<MigrationPlan>,
    },
}

/// Shared cluster handles surfaced to API/gRPC layers.
#[derive(Clone)]
pub struct ClusterState {
    /// Stable identifier assigned to the local node.
    pub node_id: String,
    /// Handle to the SWIM membership engine.
    pub membership: Arc<SwimMembership>,
    /// Cached assignment manager shared with API layers.
    pub assignments: Arc<SpaceAssignmentManager>,
    /// Replication policy enforced for each placement decision.
    pub replication: ReplicationConfig,
    /// Partition detector shared with routing layers.
    pub partition_detector: Arc<PartitionDetector>,
    /// Split-brain detector guarding concurrent primaries.
    pub split_brain: Arc<SplitBrainDetector>,
    /// Rebalance coordinator for admin operations.
    pub rebalance: Arc<RebalanceCoordinator>,
    /// Replication metadata shared with diagnostics.
    #[cfg(feature = "memory_mapped_persistence")]
    pub replication_metadata: Option<Arc<ReplicationMetadata>>,
}

/// Result of planning which node should handle an operation.
#[derive(Debug, Clone)]
pub enum RouteDecision {
    /// The current process owns the primary for the requested space.
    Local,
    /// Another node should handle the request (identified by its metadata).
    Remote {
        /// Node assigned as the primary owner.
        primary: NodeInfo,
        /// Replica candidates for failover or read-scaling.
        replicas: Vec<NodeInfo>,
    },
}

impl RouteDecision {
    /// Convenience helper for constructing a local decision.
    #[must_use]
    pub const fn local() -> Self {
        Self::Local
    }

    /// Convenience helper for constructing a remote decision with replicas.
    #[must_use]
    pub const fn remote(primary: NodeInfo, replicas: Vec<NodeInfo>) -> Self {
        Self::Remote { primary, replicas }
    }

    /// Returns `true` when the route targets the local node.
    #[must_use]
    pub const fn is_local(&self) -> bool {
        matches!(self, Self::Local)
    }

    /// Returns the current set of replicas for remote routes.
    #[must_use]
    pub fn replicas(&self) -> &[NodeInfo] {
        match self {
            Self::Local => &[],
            Self::Remote { replicas, .. } => replicas,
        }
    }

    /// Returns the primary node for remote routes.
    #[must_use]
    pub const fn primary(&self) -> Option<&NodeInfo> {
        match self {
            Self::Local => None,
            Self::Remote { primary, .. } => Some(primary),
        }
    }
}

impl ClusterState {
    /// Plan the placement for the provided memory space.
    pub fn plan_space(&self, space_id: &MemorySpaceId) -> Result<SpaceAssignment, ClusterError> {
        self.assignments.assign(space_id)
    }

    /// Ensure the local node is the primary owner for a memory space before writes.
    pub fn ensure_local_primary(&self, space_id: &MemorySpaceId) -> Result<(), ClusterError> {
        match self.route_for_space(space_id)? {
            RouteDecision::Local => Ok(()),
            RouteDecision::Remote { primary, .. } => Err(ClusterError::NotPrimary {
                space: space_id.clone(),
                owner: primary.id,
                local: self.node_id.clone(),
            }),
        }
    }

    /// Determine whether the local node should service the request or proxy it.
    pub fn route_for_space(&self, space_id: &MemorySpaceId) -> Result<RouteDecision, ClusterError> {
        let assignment = self.plan_space(space_id)?;
        if assignment.primary.id == self.node_id {
            return Ok(RouteDecision::Local);
        }

        if let PartitionState::Partitioned {
            reachable_nodes,
            total_nodes,
            ..
        } = self.partition_detector.current_state()
        {
            return Err(ClusterError::Partitioned {
                reachable_nodes,
                total_nodes,
            });
        }

        Ok(RouteDecision::Remote {
            primary: assignment.primary,
            replicas: assignment.replicas,
        })
    }

    /// Snapshot the latest partition state for diagnostics.
    #[must_use]
    pub fn partition_state(&self) -> PartitionState {
        self.partition_detector.current_state()
    }

    /// Tick the split-brain detector after a successful local write.
    pub async fn record_local_write(&self, space_id: &MemorySpaceId) {
        self.split_brain.on_write(space_id).await;
    }

    /// Snapshot current assignment cache metrics for diagnostics.
    #[must_use]
    pub fn assignment_snapshot(&self) -> AssignmentSnapshot {
        self.assignments.snapshot()
    }

    /// Fetch the latest rebalance status, if coordinated in this process.
    #[must_use]
    pub fn rebalance_status(&self) -> RebalanceStatus {
        self.rebalance.status()
    }

    /// Trigger a manual rebalance scan across cached spaces.
    pub async fn trigger_rebalance(&self) -> Result<usize, ClusterError> {
        self.rebalance.trigger_rebalance().await
    }

    /// Force a single space to migrate and return the plan generated for execution.
    pub async fn migrate_space(
        &self,
        space_id: &MemorySpaceId,
    ) -> Result<Option<MigrationPlan>, ClusterError> {
        self.rebalance.migrate_space(space_id).await
    }
}

/// Initialize cluster discovery + membership according to configuration.
pub async fn initialize_cluster(config: &ClusterConfig) -> Result<ClusterContext> {
    if !config.enabled {
        info!("cluster mode disabled; running single-node");
        return Ok(ClusterContext::SingleNode);
    }

    let swim_bind = parse_addr(&config.network.swim_bind, "cluster.network.swim_bind")?;
    let api_bind = parse_addr(&config.network.api_bind, "cluster.network.api_bind")?;
    let node_id = if config.node_id.trim().is_empty() {
        format!("engram-{:#x}", Uuid::new_v4())
    } else {
        config.node_id.clone()
    };

    let discovery = build_discovery(&config.discovery)
        .map_err(|err| map_discovery_error(err, &config.discovery))?;
    let seeds = discovery.discover().await?;

    let advertise_override = std::env::var("ENGRAM_CLUSTER_ADVERTISE_ADDR")
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
        .map(|value| parse_addr(&value, "ENGRAM_CLUSTER_ADVERTISE_ADDR"))
        .transpose()?;
    let advertise_addr = resolve_advertise_addr(
        advertise_override.or(config.network.advertise_addr),
        swim_bind,
        &seeds,
    );
    if advertise_addr.ip().is_unspecified() {
        return Err(anyhow!(
            "Unable to determine cluster advertise address for bind {swim_bind}. \
             Set `[cluster.network].advertise_addr` or the ENGRAM_CLUSTER_ADVERTISE_ADDR env var \
             to a routable host:port so peers can reach this node."
        ));
    }

    let local = NodeInfo::new(node_id.clone(), advertise_addr, api_bind, None, None);
    let membership = Arc::new(SwimMembership::new(local, config.swim.clone()));
    let planner = Arc::new(SpaceAssignmentPlanner::new(
        Arc::clone(&membership),
        &config.replication,
    ));
    let assignments = Arc::new(SpaceAssignmentManager::new(
        Arc::clone(&planner),
        &config.replication,
    ));
    let (rebalance, migration_rx) =
        RebalanceCoordinator::new(Arc::clone(&assignments), Arc::clone(&membership), 128);

    let now = Instant::now();
    for seed in &seeds {
        if *seed == swim_bind {
            continue;
        }
        let node = NodeInfo::new(seed.to_string(), *seed, *seed, None, None);
        membership.upsert_member(node, 0, now);
    }

    if seeds.is_empty() {
        warn!("cluster discovery returned zero peers");
    }

    let partition_detector = Arc::new(PartitionDetector::new(
        Arc::clone(&membership),
        config.partition.clone(),
    ));
    let split_brain = Arc::new(SplitBrainDetector::new(node_id.clone()));

    Ok(ClusterContext::Distributed {
        node_id,
        membership,
        assignments,
        replication: config.replication.clone(),
        seeds,
        swim_addr: swim_bind,
        partition_detector,
        split_brain,
        rebalance,
        migration_rx,
    })
}

fn parse_addr(input: &str, field: &str) -> Result<SocketAddr> {
    SocketAddr::from_str(input)
        .with_context(|| format!("invalid socket address for {field}: {input}"))
}

fn map_discovery_error(err: DiscoveryError, cfg: &DiscoveryConfig) -> anyhow::Error {
    match err {
        DiscoveryError::DnsUnavailable(reason) => {
            anyhow::anyhow!("discovery backend {:?} unavailable: {}", cfg, reason)
        }
        _ => anyhow::anyhow!(err),
    }
}

fn resolve_advertise_addr(
    configured: Option<SocketAddr>,
    bind_addr: SocketAddr,
    seeds: &[SocketAddr],
) -> SocketAddr {
    if let Some(addr) = configured {
        return addr;
    }
    if !bind_addr.ip().is_unspecified() {
        return bind_addr;
    }

    for seed in seeds {
        if let Ok(local) = probe_local_interface(*seed) {
            return SocketAddr::new(local.ip(), bind_addr.port());
        }
    }

    bind_addr
}

fn probe_local_interface(seed: SocketAddr) -> std::io::Result<SocketAddr> {
    let bind_any = if seed.is_ipv4() {
        "0.0.0.0:0"
    } else {
        "[::]:0"
    };
    let socket = UdpSocket::bind(bind_any)?;
    socket.connect(seed)?;
    socket.local_addr()
}
