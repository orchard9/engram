use std::sync::Arc;

use tokio::task::yield_now;
use tracing::{debug, info, warn};

use super::ClusterError;
use super::membership::{NodeInfo, SwimMembership};
use super::partition::{PartitionDetector, PartitionState};

/// Placeholder anti-entropy synchronizer invoked after partitions heal.
pub struct AntiEntropySync {
    membership: Arc<SwimMembership>,
    detector: Arc<PartitionDetector>,
}

impl AntiEntropySync {
    /// Create a new anti-entropy synchronizer tied to the membership engine and detector.
    #[must_use]
    pub const fn new(membership: Arc<SwimMembership>, detector: Arc<PartitionDetector>) -> Self {
        Self {
            membership,
            detector,
        }
    }

    /// Perform best-effort reconciliation against nodes that became reachable during healing.
    pub async fn sync_after_partition(&self) -> Result<(), ClusterError> {
        let state = self.detector.current_state();
        let newly_reachable = match state {
            PartitionState::Healing {
                ref newly_reachable,
                ..
            } if !newly_reachable.is_empty() => newly_reachable.clone(),
            _ => return Ok(()),
        };

        info!(
            targets = newly_reachable.len(),
            "Starting anti-entropy sync"
        );
        for node_id in newly_reachable {
            if let Some(node) = self.lookup_node(&node_id) {
                if let Err(err) = self.sync_with_node(&node).await {
                    warn!(node = %node.id, "Anti-entropy sync failed: {err}");
                }
            } else {
                warn!(node = %node_id, "Skipping anti-entropy sync for unknown node");
            }
        }
        info!("Anti-entropy sync finished");
        Ok(())
    }

    fn lookup_node(&self, node_id: &str) -> Option<NodeInfo> {
        self.membership
            .snapshots()
            .into_iter()
            .find(|snapshot| snapshot.node.id == node_id)
            .map(|snapshot| snapshot.node)
    }

    async fn sync_with_node(&self, node: &NodeInfo) -> Result<(), ClusterError> {
        debug!(peer = %node.id, "Synchronizing state with peer after partition");
        // Yield to avoid monopolising the runtime while future async plumbing is wired in.
        yield_now().await;
        // Follow-up tasks will stream actual diffs; for now we only emit diagnostic logs.
        Ok(())
    }
}
