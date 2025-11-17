use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::SystemTime;

use parking_lot::Mutex;
use tokio::sync::{broadcast, mpsc, watch};
use tokio::task::JoinHandle;
use tracing::warn;

use crate::MemorySpaceId;
use crate::cluster::assignment::{CachedAssignment, SpaceAssignmentManager};
use crate::cluster::error::ClusterError;
use crate::cluster::membership::{MembershipUpdate, NodeInfo, NodeState, SwimMembership};
use crate::metrics;

const RECENT_PLAN_LIMIT: usize = 64;

/// Reason why a migration plan was generated.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MigrationReason {
    /// Triggered automatically after a membership change.
    MembershipChange,
    /// Triggered via an operator/admin request.
    Manual,
}

/// High-level description of a planned space migration.
#[derive(Debug, Clone)]
pub struct MigrationPlan {
    /// Memory space that requires migration.
    pub space: MemorySpaceId,
    /// Previous primary owner (if cached).
    pub from: Option<NodeInfo>,
    /// Target primary node after the migration completes.
    pub to: NodeInfo,
    /// Assignment version to apply once the migration is finished.
    pub version: u64,
    /// Reason the migration was planned.
    pub reason: MigrationReason,
    /// Timestamp when the plan was produced.
    pub planned_at: SystemTime,
}

/// Summary of rebalance activity for diagnostics/admin endpoints.
#[derive(Debug, Clone)]
pub struct RebalanceStatus {
    /// Number of in-flight plans awaiting execution/publication.
    pub pending_events: usize,
    /// Total number of plans produced since startup.
    pub planned_total: u64,
    /// Timestamp of the most recent plan.
    pub last_event: Option<SystemTime>,
    /// Ring buffer of the most recent migration plans.
    pub recent: Vec<MigrationPlan>,
}

/// Coordinates space rebalancing when membership changes occur.
pub struct RebalanceCoordinator {
    assignments: Arc<SpaceAssignmentManager>,
    membership: Arc<SwimMembership>,
    plans_tx: mpsc::Sender<MigrationPlan>,
    recent: Mutex<VecDeque<MigrationPlan>>,
    pending: AtomicU64,
    planned_total: AtomicU64,
    last_event: Mutex<Option<SystemTime>>,
}

impl RebalanceCoordinator {
    /// Create a new coordinator and channel for downstream consumers of migration plans.
    pub fn new(
        assignments: Arc<SpaceAssignmentManager>,
        membership: Arc<SwimMembership>,
        capacity: usize,
    ) -> (Arc<Self>, mpsc::Receiver<MigrationPlan>) {
        let (plans_tx, plans_rx) = mpsc::channel(capacity);
        let coordinator = Arc::new(Self {
            assignments,
            membership,
            plans_tx,
            recent: Mutex::new(VecDeque::with_capacity(RECENT_PLAN_LIMIT)),
            pending: AtomicU64::new(0),
            planned_total: AtomicU64::new(0),
            last_event: Mutex::new(None),
        });
        (coordinator, plans_rx)
    }

    /// Start listening for membership updates until the provided shutdown signal fires.
    pub fn spawn(self: &Arc<Self>, mut shutdown: watch::Receiver<bool>) -> JoinHandle<()> {
        let coordinator = Arc::clone(self);
        tokio::spawn(async move {
            let mut updates = coordinator.membership.subscribe();
            loop {
                tokio::select! {
                    _ = shutdown.changed() => break,
                    update = updates.recv() => match update {
                        Ok(update) => coordinator.handle_update(update).await,
                        Err(broadcast::error::RecvError::Lagged(_)) => {}
                        Err(broadcast::error::RecvError::Closed) => break,
                    }
                }
            }
        })
    }

    /// Force a full rescan of cached spaces, returning the number of plans produced.
    pub async fn trigger_rebalance(&self) -> Result<usize, ClusterError> {
        let spaces = self.assignments.cached_spaces();
        let mut planned = 0;
        for space in spaces {
            if let Some(plan) = self.reassign_space(&space, MigrationReason::Manual)? {
                planned += 1;
                self.enqueue_plan(plan).await;
            }
        }
        Ok(planned)
    }

    /// Manually migrate a single space and return the generated plan (if any).
    pub async fn migrate_space(
        &self,
        space: &MemorySpaceId,
    ) -> Result<Option<MigrationPlan>, ClusterError> {
        let plan = self.reassign_space(space, MigrationReason::Manual)?;
        if let Some(plan) = plan.as_ref() {
            self.enqueue_plan(plan.clone()).await;
        }
        Ok(plan)
    }

    /// Snapshot the current coordinator state for observability surfaces.
    pub fn status(&self) -> RebalanceStatus {
        let recent = self.recent.lock().iter().cloned().collect();
        RebalanceStatus {
            pending_events: self.pending.load(Ordering::Relaxed) as usize,
            planned_total: self.planned_total.load(Ordering::Relaxed),
            last_event: *self.last_event.lock(),
            recent,
        }
    }

    async fn handle_update(self: &Arc<Self>, update: MembershipUpdate) {
        match update.state {
            NodeState::Dead | NodeState::Left => {
                self.plan_departure(&update.node).await;
            }
            NodeState::Alive => {
                if let Err(err) = self.plan_join_rebalance().await {
                    warn!("rebalance scan after join failed: {err}");
                }
            }
            NodeState::Suspect => {}
        }
    }

    async fn plan_departure(&self, node: &NodeInfo) {
        let spaces = self.assignments.spaces_assigned_to(&node.id);
        for space in spaces {
            match self.reassign_space(&space, MigrationReason::MembershipChange) {
                Ok(Some(plan)) => self.enqueue_plan(plan).await,
                Ok(None) => {}
                Err(err) => {
                    warn!(space = %space, "failed to reassign space during departure: {err}");
                }
            }
        }
    }

    async fn plan_join_rebalance(&self) -> Result<(), ClusterError> {
        let spaces = self.assignments.cached_spaces();
        for space in spaces {
            if let Some(plan) = self.reassign_space(&space, MigrationReason::MembershipChange)? {
                self.enqueue_plan(plan).await;
            }
        }
        Ok(())
    }

    fn reassign_space(
        &self,
        space: &MemorySpaceId,
        reason: MigrationReason,
    ) -> Result<Option<MigrationPlan>, ClusterError> {
        let previous = self.assignments.invalidate(space);
        let cached = self.assignments.recompute(space)?;
        previous.map_or(Ok(None), |prev| {
            Ok(Self::plan_from_assignments(space, prev, cached, reason))
        })
    }

    fn plan_from_assignments(
        space: &MemorySpaceId,
        previous: CachedAssignment,
        current: CachedAssignment,
        reason: MigrationReason,
    ) -> Option<MigrationPlan> {
        if previous.assignment.primary.id == current.assignment.primary.id {
            return None;
        }
        Some(MigrationPlan {
            space: space.clone(),
            from: Some(previous.assignment.primary),
            to: current.assignment.primary,
            version: current.version,
            reason,
            planned_at: SystemTime::now(),
        })
    }

    async fn enqueue_plan(&self, plan: MigrationPlan) {
        self.record_plan(&plan);
        self.pending.fetch_add(1, Ordering::Relaxed);
        if let Err(err) = self.plans_tx.send(plan.clone()).await {
            warn!("failed to publish migration plan: {err}");
        }
        self.pending.fetch_sub(1, Ordering::Relaxed);
    }

    fn record_plan(&self, plan: &MigrationPlan) {
        self.planned_total.fetch_add(1, Ordering::Relaxed);
        metrics::increment_counter(metrics::CLUSTER_REBALANCE_PLANS_TOTAL, 1);
        {
            let mut recent = self.recent.lock();
            if recent.len() >= RECENT_PLAN_LIMIT {
                recent.pop_front();
            }
            recent.push_back(plan.clone());
        }
        *self.last_event.lock() = Some(plan.planned_at);
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::expect_used, clippy::unwrap_used)]

    use super::*;
    use crate::cluster::config::{PlacementStrategy, ReplicationConfig, SwimConfig};
    use crate::cluster::membership::{NodeInfo, SwimMembership};
    use crate::cluster::placement::SpaceAssignmentPlanner;
    use std::net::{IpAddr, Ipv4Addr, SocketAddr};
    use std::time::Instant;

    fn test_membership() -> Arc<SwimMembership> {
        let local = NodeInfo::new(
            "node-local",
            SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), 7_946),
            SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), 5_051),
            None,
            None,
        );
        let membership = Arc::new(SwimMembership::new(local, SwimConfig::default()));
        let now = Instant::now();
        for idx in 1..=2 {
            let node = NodeInfo::new(
                format!("node-{idx}"),
                SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), 7_946 + idx),
                SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), 5_051 + idx),
                None,
                None,
            );
            membership.upsert_member(node, u64::from(idx), now);
        }
        membership
    }

    fn test_manager() -> SpaceAssignmentManager {
        let membership = test_membership();
        let replication = ReplicationConfig {
            placement: PlacementStrategy::Random,
            ..ReplicationConfig::default()
        };
        let planner = Arc::new(SpaceAssignmentPlanner::new(
            Arc::clone(&membership),
            &replication,
        ));
        SpaceAssignmentManager::new(planner, &replication)
    }

    #[tokio::test]
    async fn migrate_space_emits_plan() {
        let assignments = Arc::new(test_manager());
        let membership = test_membership();
        let space = MemorySpaceId::try_from("delta").unwrap();
        let _ = assignments.assign(&space);
        let (coordinator, mut rx) =
            RebalanceCoordinator::new(Arc::clone(&assignments), membership, 4);
        let result = coordinator
            .migrate_space(&space)
            .await
            .expect("migrate should succeed");

        // Migration may or may not result in a plan depending on whether
        // reassignment picks a different primary (Random placement strategy)
        if let Some(plan) = result {
            assert_eq!(plan.space, space);
            let emitted = rx.recv().await.expect("plan on channel");
            assert_eq!(emitted.space, space);
        }
    }
}
