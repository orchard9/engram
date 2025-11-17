use std::sync::Arc;
use std::time::Duration;

use super::{LockFreeGauges, LockFreeHistograms};

const CLUSTER_MEMBERSHIP_ALIVE: &str = "engram_cluster_membership_alive";
const CLUSTER_MEMBERSHIP_SUSPECT: &str = "engram_cluster_membership_suspect";
const CLUSTER_MEMBERSHIP_DEAD: &str = "engram_cluster_membership_dead";
const CLUSTER_MEMBERSHIP_LEFT: &str = "engram_cluster_membership_left";
const CLUSTER_MEMBERSHIP_TOTAL: &str = "engram_cluster_membership_total";
const CLUSTER_PROBE_LATENCY_SECONDS: &str = "engram_cluster_probe_latency_seconds";
const CLUSTER_PARTITION_STATE: &str = "engram_cluster_partition_state";
const CLUSTER_PARTITION_REACHABLE: &str = "engram_cluster_partition_reachable";
const CLUSTER_PARTITION_TOTAL: &str = "engram_cluster_partition_total";
const CLUSTER_PARTITION_RATIO: &str = "engram_cluster_partition_ratio";

/// Lock-free metrics covering SWIM membership health.
pub struct ClusterMetrics {
    gauges: Arc<LockFreeGauges>,
    histograms: Arc<LockFreeHistograms>,
}

impl ClusterMetrics {
    pub(crate) const fn new(
        gauges: Arc<LockFreeGauges>,
        histograms: Arc<LockFreeHistograms>,
    ) -> Self {
        Self { gauges, histograms }
    }

    /// Record membership counts for alive/suspect/dead/left states.
    pub fn record_membership(&self, alive: usize, suspect: usize, dead: usize, left: usize) {
        let alive_f = alive as f64;
        let suspect_f = suspect as f64;
        let dead_f = dead as f64;
        let left_f = left as f64;
        self.gauges.set(CLUSTER_MEMBERSHIP_ALIVE, alive_f);
        self.gauges.set(CLUSTER_MEMBERSHIP_SUSPECT, suspect_f);
        self.gauges.set(CLUSTER_MEMBERSHIP_DEAD, dead_f);
        self.gauges.set(CLUSTER_MEMBERSHIP_LEFT, left_f);
        self.gauges.set(
            CLUSTER_MEMBERSHIP_TOTAL,
            alive_f + suspect_f + dead_f + left_f,
        );
    }

    /// Record the latency of an individual SWIM probe.
    pub fn record_probe_latency(&self, latency: Duration) {
        self.histograms
            .observe(CLUSTER_PROBE_LATENCY_SECONDS, latency.as_secs_f64());
    }

    /// Record the current partition status and reachability counts.
    pub fn record_partition_state(&self, status: PartitionStatus, reachable: usize, total: usize) {
        let reachable_f = reachable as f64;
        let total_f = total.max(1) as f64;
        self.gauges.set(CLUSTER_PARTITION_STATE, status.as_value());
        self.gauges.set(CLUSTER_PARTITION_REACHABLE, reachable_f);
        self.gauges.set(CLUSTER_PARTITION_TOTAL, total_f);
        self.gauges.set(
            CLUSTER_PARTITION_RATIO,
            (reachable_f / total_f).clamp(0.0, 1.0),
        );
    }
}

/// Partition status mapped to a gauge value for dashboards.
#[derive(Debug, Clone, Copy)]
pub enum PartitionStatus {
    /// Majority reachable.
    Connected,
    /// Majority unreachable.
    Partitioned,
    /// Recovering from a partition.
    Healing,
}

impl PartitionStatus {
    const fn as_value(self) -> f64 {
        match self {
            Self::Connected => 0.0,
            Self::Partitioned => 1.0,
            Self::Healing => 2.0,
        }
    }
}
