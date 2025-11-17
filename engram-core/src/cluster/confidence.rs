use std::sync::Arc;

use super::partition::{PartitionDetector, PartitionState};

/// Applies dynamic confidence penalties during partitions or healing periods.
pub struct PartitionAwareConfidence {
    detector: Arc<PartitionDetector>,
    floor_multiplier: f32,
}

impl PartitionAwareConfidence {
    /// Create a confidence adjuster with the default 50% lower bound.
    #[must_use]
    pub const fn new(detector: Arc<PartitionDetector>) -> Self {
        Self::with_floor(detector, 0.5)
    }

    /// Create an adjuster with a custom floor multiplier.
    #[must_use]
    pub const fn with_floor(detector: Arc<PartitionDetector>, floor_multiplier: f32) -> Self {
        Self {
            detector,
            floor_multiplier: floor_multiplier.clamp(0.0, 1.0),
        }
    }

    /// Scale the provided confidence based on the current partition state.
    #[must_use]
    pub fn adjust_confidence(&self, base: f32) -> f32 {
        let state = self.detector.current_state();
        match state {
            PartitionState::Connected { .. } => base,
            PartitionState::Partitioned {
                reachable_nodes,
                total_nodes,
                ..
            } => {
                if total_nodes == 0 {
                    return (base * self.floor_multiplier).max(0.0);
                }
                let reachability = reachable_nodes as f32 / total_nodes as f32;
                let penalty = (1.0 - reachability).clamp(0.0, 1.0);
                let scaled = base * (1.0 - 0.5 * penalty);
                scaled.max(base * self.floor_multiplier)
            }
            PartitionState::Healing { .. } => base * 0.9,
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use super::*;
    use crate::cluster::config::SwimConfig;
    use crate::cluster::{
        config::PartitionConfig,
        membership::{NodeInfo, SwimMembership},
        partition::PartitionState,
    };
    use std::time::Instant;

    #[tokio::test]
    async fn applies_partition_penalty() {
        let local = NodeInfo::new(
            "node0",
            "127.0.0.1:7946".parse().unwrap(),
            "127.0.0.1:50051".parse().unwrap(),
            None,
            None,
        );
        let membership = Arc::new(SwimMembership::new(local, SwimConfig::default()));
        let detector = Arc::new(PartitionDetector::new(
            Arc::clone(&membership),
            PartitionConfig::default(),
        ));
        detector.set_state_for_test(PartitionState::Partitioned {
            reachable_nodes: 2,
            total_nodes: 8,
            partitioned_since: Instant::now(),
        });

        let adjuster = PartitionAwareConfidence::new(detector);
        let base = 0.8;
        let penalized = adjuster.adjust_confidence(base);
        assert!(penalized < base);
        assert!(penalized > base * 0.5);
    }
}
