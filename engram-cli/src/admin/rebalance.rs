//! Cluster rebalance HTTP/gRPC handlers.

use axum::{Json, extract::State, http::StatusCode};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use crate::api::{ApiError, ApiState};
use crate::cluster::ClusterState;
use engram_core::MemorySpaceId;
use engram_core::cluster::{MigrationPlan, MigrationReason};

/// JSON payload for `POST /cluster/migrate`.
#[derive(Deserialize)]
pub struct MigrateSpaceRequest {
    /// Memory space identifier to migrate.
    pub space: String,
}

/// View returned by admin endpoints describing a migration plan.
#[derive(Serialize)]
pub struct MigrationPlanView {
    /// Memory space identifier.
    pub space: String,
    /// Previous primary owner if cached locally.
    pub from: Option<String>,
    /// Target primary node identifier.
    pub to: String,
    /// Assignment version clients should wait for.
    pub version: u64,
    /// Textual reason (`membership_change` or `manual`).
    pub reason: String,
    /// RFC3339 timestamp when the plan was emitted.
    pub planned_at: String,
}

/// Response body for `POST /cluster/migrate`.
#[derive(Serialize)]
pub struct MigrateSpaceResponse {
    /// Planned migration when the space was cached locally.
    pub plan: Option<MigrationPlanView>,
}

/// Response body for `POST /cluster/rebalance`.
#[derive(Serialize)]
pub struct TriggerRebalanceResponse {
    /// Number of migrations queued by the scan.
    pub planned: usize,
}

/// Response body for `GET /cluster/rebalance`.
#[derive(Serialize)]
pub struct RebalanceStatusResponse {
    /// Number of plans currently buffered.
    pub pending: usize,
    /// Total plan count since startup.
    pub planned_total: u64,
    /// Timestamp of the most recent plan, if any.
    pub last_plan: Option<String>,
    /// Recent plan summaries.
    pub recent: Vec<MigrationPlanView>,
}

/// Force a single memory space to migrate to its latest assignment.
pub async fn migrate_space_handler(
    State(state): State<ApiState>,
    Json(request): Json<MigrateSpaceRequest>,
) -> Result<Json<MigrateSpaceResponse>, ApiError> {
    let cluster = require_cluster(&state)?;
    let space_id = MemorySpaceId::try_from(request.space.as_str())
        .map_err(|err| ApiError::InvalidInput(format!("invalid space id: {err}")))?;
    let plan = cluster
        .migrate_space(&space_id)
        .await
        .map_err(ApiError::from)?
        .map(plan_to_view);
    Ok(Json(MigrateSpaceResponse { plan }))
}

/// Trigger a full rebalance scan across cached spaces.
pub async fn trigger_rebalance_handler(
    State(state): State<ApiState>,
) -> Result<(StatusCode, Json<TriggerRebalanceResponse>), ApiError> {
    let cluster = require_cluster(&state)?;
    let planned = cluster.trigger_rebalance().await.map_err(ApiError::from)?;
    Ok((
        StatusCode::ACCEPTED,
        Json(TriggerRebalanceResponse { planned }),
    ))
}

/// Return cached migration plans and pending counts for operators.
pub async fn rebalance_status_handler(
    State(state): State<ApiState>,
) -> Result<Json<RebalanceStatusResponse>, ApiError> {
    let cluster = require_cluster(&state)?;
    let status = cluster.rebalance_status();
    let recent = status.recent.into_iter().map(plan_to_view).collect();
    let last_plan = status
        .last_event
        .map(|ts| DateTime::<Utc>::from(ts).to_rfc3339());
    Ok(Json(RebalanceStatusResponse {
        pending: status.pending_events,
        planned_total: status.planned_total,
        last_plan,
        recent,
    }))
}

fn require_cluster(state: &ApiState) -> Result<Arc<ClusterState>, ApiError> {
    state
        .cluster
        .clone()
        .ok_or_else(|| ApiError::FeatureNotEnabled("Cluster mode disabled".into()))
}

fn plan_to_view(plan: MigrationPlan) -> MigrationPlanView {
    let planned_at = DateTime::<Utc>::from(plan.planned_at).to_rfc3339();
    MigrationPlanView {
        space: plan.space.to_string(),
        from: plan.from.map(|node| node.id),
        to: plan.to.id,
        version: plan.version,
        reason: match plan.reason {
            MigrationReason::MembershipChange => "membership_change".to_string(),
            MigrationReason::Manual => "manual".to_string(),
        },
        planned_at,
    }
}
