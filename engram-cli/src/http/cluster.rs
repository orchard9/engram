#![allow(missing_docs)]

use std::time::Instant;

use axum::{Json, extract::State, response::IntoResponse};
use serde::Serialize;

use crate::api::{ApiError, ApiState};
use crate::router::{RouterBreakerState, RouterHealthSnapshot};
use engram_core::cluster::{NodeInfo, NodeState, PartitionState};

#[derive(Serialize)]
pub struct ClusterHealthSummary {
    pub node_id: String,
    pub stats: ClusterMemberStats,
    pub partition: PartitionStatusView,
    pub assignments: Option<AssignmentSummaryView>,
    pub replication: Option<ReplicationSummaryView>,
    pub router: Option<RouterHealthView>,
}

#[derive(Serialize)]
pub struct ClusterMemberStats {
    pub alive: usize,
    pub suspect: usize,
    pub dead: usize,
    pub left: usize,
    pub total: usize,
}

#[derive(Serialize)]
pub struct PartitionStatusView {
    pub state: String,
    pub reachable_nodes: usize,
    pub total_nodes: usize,
}

#[derive(Serialize)]
pub struct ClusterNodesResponse {
    pub nodes: Vec<ClusterNodeView>,
}

#[derive(Serialize)]
pub struct AssignmentSummaryView {
    pub cached_spaces: usize,
    pub per_node: Vec<NodeAssignmentView>,
}

#[derive(Serialize)]
pub struct NodeAssignmentView {
    pub node_id: String,
    pub primary_spaces: usize,
}

#[derive(Serialize)]
pub struct ReplicationSummaryView {
    pub replicas: Vec<ReplicaLagView>,
}

#[derive(Serialize)]
pub struct ReplicaLagView {
    pub space: String,
    pub replica: String,
    pub local_sequence: u64,
    pub replicated_sequence: u64,
    pub lag: u64,
}

#[derive(Serialize)]
pub struct RouterHealthView {
    pub requests_total: u64,
    pub retries_total: u64,
    pub replica_fallback_total: u64,
    pub open_breakers: usize,
    pub breakers: Vec<RouterBreakerView>,
}

#[derive(Serialize)]
pub struct RouterBreakerView {
    pub node_id: String,
    pub state: String,
    pub retry_after_ms: Option<u64>,
}

#[derive(Serialize)]
pub struct ClusterNodeView {
    pub id: String,
    pub swim_address: String,
    pub api_address: String,
    pub rack: Option<String>,
    pub zone: Option<String>,
    pub state: NodeState,
    pub incarnation: u64,
    pub last_change_ms_ago: u128,
    pub local: bool,
}

pub async fn cluster_health(State(state): State<ApiState>) -> Result<impl IntoResponse, ApiError> {
    let cluster = state
        .cluster
        .as_ref()
        .ok_or_else(|| ApiError::FeatureNotEnabled("Cluster mode disabled".to_string()))?;
    let membership_stats = cluster.membership.stats();
    let partition_snapshot = match cluster.partition_state() {
        PartitionState::Connected {
            reachable_nodes,
            total_nodes,
        } => PartitionStatusView {
            state: "connected".to_string(),
            reachable_nodes,
            total_nodes,
        },
        PartitionState::Partitioned {
            reachable_nodes,
            total_nodes,
            ..
        } => PartitionStatusView {
            state: "partitioned".to_string(),
            reachable_nodes,
            total_nodes,
        },
        PartitionState::Healing { .. } => PartitionStatusView {
            state: "healing".to_string(),
            reachable_nodes: membership_stats.alive + 1,
            total_nodes: membership_stats.total() + 1,
        },
    };
    let assignments = cluster.assignment_snapshot();
    let assignment_summary = AssignmentSummaryView {
        cached_spaces: assignments.cached_spaces,
        per_node: assignments
            .per_node
            .into_iter()
            .map(|load| NodeAssignmentView {
                node_id: load.node_id,
                primary_spaces: load.primary_spaces,
            })
            .collect(),
    };

    #[cfg(feature = "memory_mapped_persistence")]
    let replication_summary = cluster.replication_metadata.as_ref().and_then(|metadata| {
        let replicas: Vec<ReplicaLagView> = metadata
            .snapshot()
            .into_iter()
            .flat_map(|space_summary| {
                let space_name = space_summary.space.to_string();
                space_summary
                    .replicas
                    .into_iter()
                    .map(move |lag| ReplicaLagView {
                        space: space_name.clone(),
                        replica: lag.replica.clone(),
                        local_sequence: lag.local_sequence,
                        replicated_sequence: lag.replica_sequence,
                        lag: lag.sequences_behind(),
                    })
            })
            .collect();
        if replicas.is_empty() {
            None
        } else {
            Some(ReplicationSummaryView { replicas })
        }
    });

    #[cfg(not(feature = "memory_mapped_persistence"))]
    let replication_summary: Option<ReplicationSummaryView> = None;

    let router_summary = state
        .router
        .as_ref()
        .map(|router| RouterHealthView::from(router.health_snapshot()));

    Ok(Json(ClusterHealthSummary {
        node_id: cluster.node_id.clone(),
        stats: ClusterMemberStats {
            alive: membership_stats.alive,
            suspect: membership_stats.suspect,
            dead: membership_stats.dead,
            left: membership_stats.left,
            total: membership_stats.total(),
        },
        partition: partition_snapshot,
        assignments: Some(assignment_summary),
        replication: replication_summary,
        router: router_summary,
    }))
}

pub async fn cluster_nodes(State(state): State<ApiState>) -> Result<impl IntoResponse, ApiError> {
    let cluster = state
        .cluster
        .as_ref()
        .ok_or_else(|| ApiError::FeatureNotEnabled("Cluster mode disabled".to_string()))?;
    let now = Instant::now();
    let mut nodes = Vec::new();

    nodes.push(render_node(
        cluster.membership.local_node(),
        NodeState::Alive,
        cluster.membership.local_incarnation(),
        0,
        true,
    ));

    for snapshot in cluster.membership.snapshots() {
        let elapsed = now
            .saturating_duration_since(snapshot.last_update)
            .as_millis();
        nodes.push(render_node(
            &snapshot.node,
            snapshot.state,
            snapshot.incarnation,
            elapsed,
            false,
        ));
    }

    Ok(Json(ClusterNodesResponse { nodes }))
}

impl From<RouterHealthSnapshot> for RouterHealthView {
    fn from(snapshot: RouterHealthSnapshot) -> Self {
        let breakers = snapshot
            .breakers
            .into_iter()
            .map(|breaker| RouterBreakerView {
                node_id: breaker.node_id,
                state: match breaker.state {
                    RouterBreakerState::Open => "open".to_string(),
                    RouterBreakerState::HalfOpen => "half_open".to_string(),
                },
                retry_after_ms: breaker.retry_after_ms,
            })
            .collect();

        Self {
            requests_total: snapshot.requests_total,
            retries_total: snapshot.retries_total,
            replica_fallback_total: snapshot.replica_fallback_total,
            open_breakers: snapshot.open_breakers,
            breakers,
        }
    }
}

fn render_node(
    node: &NodeInfo,
    state: NodeState,
    incarnation: u64,
    last_change_ms_ago: u128,
    local: bool,
) -> ClusterNodeView {
    ClusterNodeView {
        id: node.id.clone(),
        swim_address: node.swim_addr.to_string(),
        api_address: node.api_addr.to_string(),
        rack: node.rack.clone(),
        zone: node.zone.clone(),
        state,
        incarnation,
        last_change_ms_ago,
        local,
    }
}
