use std::collections::HashSet;
use std::sync::Arc;
use std::time::{Duration, Instant};

use parking_lot::{Mutex, RwLock};
use tokio::sync::watch;
use tokio::time::{self, MissedTickBehavior};
use tracing::{debug, info, warn};

use super::config::PartitionConfig;
use super::membership::{NodeState, SwimMembership};

#[cfg(feature = "monitoring")]
use crate::metrics::{self, PartitionStatus, cluster_metrics_handle};

#[cfg(feature = "monitoring")]
const PARTITIONS_DETECTED_COUNTER: &str = "engram_cluster_partitions_detected_total";
#[cfg(feature = "monitoring")]
const PARTITIONS_HEALED_COUNTER: &str = "engram_cluster_partitions_healed_total";

/// Cluster connectivity snapshot shared with higher-level routing components.
#[derive(Debug, Clone)]
pub enum PartitionState {
    /// Majority of peers are reachable and the cluster is healthy.
    Connected {
        /// Number of peers (including the local node) considered reachable.
        reachable_nodes: usize,
        /// Total peers in the cluster (including the local node).
        total_nodes: usize,
    },
    /// Majority of peers are unreachable; remote routing must be disabled.
    Partitioned {
        /// Reachable nodes when the partition was detected.
        reachable_nodes: usize,
        /// Total nodes tracked for partition calculations.
        total_nodes: usize,
        /// Timestamp of the first detection event.
        partitioned_since: Instant,
    },
    /// Partition has healed but the cluster is still reconciling state.
    Healing {
        /// Nodes that became reachable during this healing period.
        newly_reachable: HashSet<String>,
        /// Timestamp when healing was first detected.
        healing_since: Instant,
    },
}

impl PartitionState {
    /// Returns a human-readable status label.
    #[must_use]
    pub const fn label(&self) -> &'static str {
        match self {
            Self::Connected { .. } => "connected",
            Self::Partitioned { .. } => "partitioned",
            Self::Healing { .. } => "healing",
        }
    }

    /// Helper returning the `(reachable, total)` pair for metrics and logging.
    #[must_use]
    pub const fn counts(&self) -> (usize, usize) {
        match self {
            Self::Connected {
                reachable_nodes,
                total_nodes,
            }
            | Self::Partitioned {
                reachable_nodes,
                total_nodes,
                ..
            } => (*reachable_nodes, *total_nodes),
            Self::Healing { .. } => (0, 0),
        }
    }
}

/// Periodically inspects SWIM membership state to detect network partitions.
pub struct PartitionDetector {
    membership: Arc<SwimMembership>,
    config: PartitionConfig,
    partition_state: Arc<RwLock<PartitionState>>,
    reachable_cache: Mutex<HashSet<String>>,
}

impl PartitionDetector {
    /// Create a detector around the provided membership engine.
    #[must_use]
    pub fn new(membership: Arc<SwimMembership>, config: PartitionConfig) -> Self {
        Self {
            membership,
            config,
            partition_state: Arc::new(RwLock::new(PartitionState::Connected {
                reachable_nodes: 1,
                total_nodes: 1,
            })),
            reachable_cache: Mutex::new(HashSet::new()),
        }
    }

    /// Returns a clone of the internal [`PartitionState`] handle for observers.
    #[must_use]
    pub fn state_handle(&self) -> Arc<RwLock<PartitionState>> {
        Arc::clone(&self.partition_state)
    }

    /// Background loop that refreshes the partition state until shutdown.
    pub async fn run(self: Arc<Self>, mut shutdown: watch::Receiver<bool>) {
        let mut interval = time::interval(self.config.check_interval.max(Duration::from_secs(1)));
        interval.set_missed_tick_behavior(MissedTickBehavior::Skip);

        loop {
            tokio::select! {
                _ = shutdown.changed() => {
                    debug!("partition detector shutting down");
                    break;
                }
                _ = interval.tick() => {
                    self.check_partition_status();
                }
            }
        }
    }

    /// Perform a single reachability check. Exposed for tests and manual triggers.
    pub fn check_partition_status(&self) {
        let reachability = self.compute_reachability();
        let newly_reachable = self.diff_reachable(&reachability.alive);
        let ratio = if reachability.total == 0 {
            1.0
        } else {
            reachability.reachable as f64 / reachability.total as f64
        };
        let majority_threshold = self.config.majority_threshold.clamp(0.0, 1.0);
        let is_partitioned = ratio < majority_threshold;

        let mut state_guard = self.partition_state.write();
        match &mut *state_guard {
            PartitionState::Connected { .. } => {
                if is_partitioned {
                    warn!(
                        reachable = reachability.reachable,
                        total = reachability.total,
                        "Network partition detected"
                    );
                    *state_guard = PartitionState::Partitioned {
                        reachable_nodes: reachability.reachable,
                        total_nodes: reachability.total,
                        partitioned_since: Instant::now(),
                    };
                    Self::on_partition_detected(reachability.reachable, reachability.total);
                } else {
                    *state_guard = PartitionState::Connected {
                        reachable_nodes: reachability.reachable,
                        total_nodes: reachability.total,
                    };
                }
            }
            PartitionState::Partitioned {
                partitioned_since, ..
            } => {
                if is_partitioned {
                    if partitioned_since.elapsed() > self.config.detection_window {
                        warn!(
                            duration = partitioned_since.elapsed().as_secs_f32(),
                            "Network partition still active"
                        );
                    }
                    *state_guard = PartitionState::Partitioned {
                        reachable_nodes: reachability.reachable,
                        total_nodes: reachability.total,
                        partitioned_since: *partitioned_since,
                    };
                } else {
                    info!(
                        reachable = reachability.reachable,
                        total = reachability.total,
                        "Partition healing detected"
                    );
                    let healing_nodes = if newly_reachable.is_empty() {
                        reachability.alive.clone()
                    } else {
                        newly_reachable.iter().cloned().collect()
                    };
                    *state_guard = PartitionState::Healing {
                        newly_reachable: healing_nodes,
                        healing_since: Instant::now(),
                    };
                    Self::on_partition_healing(reachability.reachable, reachability.total);
                }
            }
            PartitionState::Healing {
                newly_reachable: tracked,
                healing_since,
            } => {
                if is_partitioned {
                    warn!("Cluster re-entered partition during healing window");
                    *state_guard = PartitionState::Partitioned {
                        reachable_nodes: reachability.reachable,
                        total_nodes: reachability.total,
                        partitioned_since: Instant::now(),
                    };
                    Self::on_partition_detected(reachability.reachable, reachability.total);
                } else if healing_since.elapsed() > self.config.detection_window {
                    info!("Network partition fully healed");
                    *state_guard = PartitionState::Connected {
                        reachable_nodes: reachability.reachable,
                        total_nodes: reachability.total,
                    };
                    Self::on_partition_healed(reachability.reachable, reachability.total);
                } else if !newly_reachable.is_empty() {
                    tracked.extend(newly_reachable.iter().cloned());
                }
            }
        }

        #[cfg(feature = "monitoring")]
        {
            let state_snapshot = state_guard.clone();
            drop(state_guard);
            Self::record_partition_metrics(
                &state_snapshot,
                reachability.reachable,
                reachability.total,
            );
        }
        #[cfg(not(feature = "monitoring"))]
        drop(state_guard);
    }

    /// Returns `true` when the detector believes the node is partitioned.
    pub fn is_partitioned(&self) -> bool {
        matches!(
            *self.partition_state.read(),
            PartitionState::Partitioned { .. }
        )
    }

    /// Obtain a snapshot of the current partition state.
    #[must_use]
    pub fn current_state(&self) -> PartitionState {
        self.partition_state.read().clone()
    }

    /// Override the detector state to simplify testing of downstream components.
    #[cfg(test)]
    pub fn set_state_for_test(&self, state: PartitionState) {
        *self.partition_state.write() = state;
    }

    fn compute_reachability(&self) -> ReachabilityStats {
        let stats = self.membership.stats();
        let tracked_nodes = stats.alive + stats.suspect + stats.dead;
        let total = (tracked_nodes + 1).max(1);
        let alive = self
            .membership
            .snapshots()
            .into_iter()
            .filter(|snapshot| snapshot.state == NodeState::Alive)
            .map(|snapshot| snapshot.node.id)
            .collect();

        ReachabilityStats {
            total,
            reachable: stats.alive + 1,
            alive,
        }
    }

    fn diff_reachable(&self, current: &HashSet<String>) -> HashSet<String> {
        let mut cache = self.reachable_cache.lock();
        let delta = current.difference(&*cache).cloned().collect::<HashSet<_>>();
        cache.clone_from(current);
        delta
    }

    fn on_partition_detected(reachable: usize, total: usize) {
        debug!(
            reachable,
            total, "entering partitioned mode (local-only operations)"
        );
        #[cfg(feature = "monitoring")]
        metrics::increment_counter(PARTITIONS_DETECTED_COUNTER, 1);
    }

    fn on_partition_healing(reachable: usize, total: usize) {
        debug!(reachable, total, "partition healing in progress");
    }

    fn on_partition_healed(reachable: usize, total: usize) {
        debug!(
            reachable,
            total, "partition healed; resuming normal routing"
        );
        #[cfg(feature = "monitoring")]
        metrics::increment_counter(PARTITIONS_HEALED_COUNTER, 1);
    }

    #[cfg(feature = "monitoring")]
    fn record_partition_metrics(state: &PartitionState, reachable: usize, total: usize) {
        if let Some(handle) = cluster_metrics_handle() {
            let status = match state {
                PartitionState::Connected { .. } => PartitionStatus::Connected,
                PartitionState::Partitioned { .. } => PartitionStatus::Partitioned,
                PartitionState::Healing { .. } => PartitionStatus::Healing,
            };
            handle.record_partition_state(status, reachable, total);
        }
    }
}

struct ReachabilityStats {
    total: usize,
    reachable: usize,
    alive: HashSet<String>,
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use super::*;
    use crate::cluster::{
        config::{PartitionConfig, SwimConfig},
        membership::{MembershipUpdate, NodeInfo, SwimMembership},
    };

    #[tokio::test]
    async fn detector_transitions_between_states() {
        let local = NodeInfo::new(
            "node0",
            "127.0.0.1:7946".parse().unwrap(),
            "127.0.0.1:50051".parse().unwrap(),
            None,
            None,
        );
        let membership = Arc::new(SwimMembership::new(local, SwimConfig::default()));
        let now = Instant::now();
        for idx in 1..=4 {
            let node = NodeInfo::new(
                format!("node{idx}"),
                format!("127.0.0.1:{}", 8_000 + idx).parse().unwrap(),
                format!("127.0.0.1:{}", 50_100 + idx).parse().unwrap(),
                None,
                None,
            );
            membership.upsert_member(node, 0, now);
        }

        let config = PartitionConfig {
            majority_threshold: 0.6,
            detection_window: Duration::from_millis(50),
            check_interval: Duration::from_millis(10),
        };
        let detector = PartitionDetector::new(Arc::clone(&membership), config);

        detector.check_partition_status();
        assert!(matches!(
            detector.current_state(),
            PartitionState::Connected { .. }
        ));

        let updates = membership
            .snapshots()
            .into_iter()
            .map(|snapshot| MembershipUpdate {
                node: snapshot.node,
                state: NodeState::Suspect,
                incarnation: snapshot.incarnation + 1,
            })
            .collect::<Vec<_>>();
        membership.apply_updates(updates);

        detector.check_partition_status();
        assert!(detector.is_partitioned());

        let revive = membership
            .snapshots()
            .into_iter()
            .map(|snapshot| MembershipUpdate {
                node: snapshot.node,
                state: NodeState::Alive,
                incarnation: snapshot.incarnation + 1,
            })
            .collect::<Vec<_>>();
        membership.apply_updates(revive);

        detector.check_partition_status();
        tokio::time::sleep(Duration::from_millis(60)).await;
        detector.check_partition_status();
        assert!(matches!(
            detector.current_state(),
            PartitionState::Connected { .. }
        ));
    }
}
