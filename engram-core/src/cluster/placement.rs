use std::cmp::Ordering;
use std::collections::HashSet;
use std::sync::Arc;

use crate::MemorySpaceId;
use crate::cluster::config::{PlacementStrategy, ReplicationConfig};
use crate::cluster::error::ClusterError;
use crate::cluster::membership::{NodeInfo, SwimMembership};

/// Primary/replica mapping for a single memory space.
#[derive(Debug, Clone)]
pub struct SpaceAssignment {
    /// Primary node responsible for writes.
    pub primary: NodeInfo,
    /// Replica targets used for failover and read-scaling.
    pub replicas: Vec<NodeInfo>,
}

/// Computes deterministic space placements from SWIM membership snapshots.
pub struct SpaceAssignmentPlanner {
    membership: Arc<SwimMembership>,
    strategy: PlacementStrategy,
    jump_buckets: usize,
    rack_penalty: f32,
    zone_penalty: f32,
}

impl SpaceAssignmentPlanner {
    /// Create a planner backed by live membership state.
    pub fn new(membership: Arc<SwimMembership>, config: &ReplicationConfig) -> Self {
        Self {
            membership,
            strategy: config.placement,
            jump_buckets: config.jump_buckets.max(1),
            rack_penalty: config.rack_penalty.clamp(0.0, 1.0),
            zone_penalty: config.zone_penalty.clamp(0.0, 1.0),
        }
    }

    /// Compute the preferred placement for `space` using the configured strategy.
    pub fn plan(
        &self,
        space: &MemorySpaceId,
        replica_count: usize,
    ) -> Result<SpaceAssignment, ClusterError> {
        let mut nodes = self.membership.alive_nodes();
        let available = nodes.len();
        let required = replica_count.saturating_add(1);
        if available < required {
            return Err(ClusterError::InsufficientHealthyNodes {
                required,
                available,
            });
        }

        nodes.sort_by(|a, b| a.id.cmp(&b.id));
        let space_hash = stable_space_hash(space);
        let bucket = jump_consistent_hash(space_hash, self.jump_buckets.max(available));
        let primary_index = bucket % available;
        let primary = nodes.remove(primary_index);
        let replicas = self.select_replicas(space_hash, &nodes, replica_count, &primary);

        Ok(SpaceAssignment { primary, replicas })
    }

    fn select_replicas(
        &self,
        space_hash: u64,
        nodes: &[NodeInfo],
        replica_count: usize,
        primary: &NodeInfo,
    ) -> Vec<NodeInfo> {
        if replica_count == 0 {
            return Vec::new();
        }

        let mut used_racks = HashSet::new();
        let mut used_zones = HashSet::new();
        if let Some(rack) = primary.rack.as_deref() {
            used_racks.insert(rack.to_string());
        }
        if let Some(zone) = primary.zone.as_deref() {
            used_zones.insert(zone.to_string());
        }

        let mut candidates: Vec<_> = nodes
            .iter()
            .map(|node| {
                let base = rendezvous_score(space_hash, node);
                let penalty = self.diversity_penalty(node, &used_racks, &used_zones);
                (base * penalty, node.clone())
            })
            .collect();

        candidates.sort_by(|(score_a, _), (score_b, _)| {
            score_b.partial_cmp(score_a).unwrap_or(Ordering::Equal)
        });

        let mut replicas = Vec::with_capacity(replica_count);
        for (_, node) in candidates.into_iter().take(replica_count) {
            if let Some(rack) = node.rack.as_deref() {
                used_racks.insert(rack.to_string());
            }
            if let Some(zone) = node.zone.as_deref() {
                used_zones.insert(zone.to_string());
            }
            replicas.push(node);
        }
        replicas
    }

    fn diversity_penalty(
        &self,
        node: &NodeInfo,
        used_racks: &HashSet<String>,
        used_zones: &HashSet<String>,
    ) -> f64 {
        match self.strategy {
            PlacementStrategy::Random => 1.0,
            PlacementStrategy::RackAware => {
                if let Some(rack) = node.rack.as_ref()
                    && used_racks.contains(rack)
                {
                    f64::from(self.rack_penalty)
                } else {
                    1.0
                }
            }
            PlacementStrategy::ZoneAware => {
                let mut penalty = 1.0;
                if let Some(zone) = node.zone.as_ref()
                    && used_zones.contains(zone)
                {
                    penalty *= f64::from(self.zone_penalty);
                }
                if let Some(rack) = node.rack.as_ref()
                    && used_racks.contains(rack)
                {
                    penalty *= f64::from(self.rack_penalty);
                }
                penalty
            }
        }
    }
}

fn stable_space_hash(space: &MemorySpaceId) -> u64 {
    const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const FNV_PRIME: u64 = 0x1000_0000_01b3;
    space.as_str().bytes().fold(FNV_OFFSET, |hash, byte| {
        let hash = hash ^ u64::from(byte);
        hash.wrapping_mul(FNV_PRIME)
    })
}

fn rendezvous_score(space_hash: u64, node: &NodeInfo) -> f64 {
    let mut hash = space_hash ^ 0x9e37_79b9_7f4a_7c15;
    for byte in node.id.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0xff51_afd7_ed55_8ccd);
    }
    (hash as f64) / (u64::MAX as f64)
}

fn jump_consistent_hash(mut key: u64, buckets: usize) -> usize {
    let mut b: i64 = -1;
    let mut j: i64 = 0;
    let buckets = buckets as i64;
    while j < buckets {
        b = j;
        key = key.wrapping_mul(2_862_933_555_777_941_757).wrapping_add(1);
        let denom = ((key >> 33) + 1) as f64;
        j = (((b as f64) + 1.0) * (1u64 << 31) as f64 / denom) as i64;
    }
    b.max(0) as usize
}

#[cfg(test)]
mod tests {
    #![allow(clippy::expect_used, clippy::panic, clippy::unwrap_used)]
    use super::*;
    use crate::cluster::config::{ReplicationConfig, SwimConfig};
    use std::collections::HashMap;
    use std::net::{IpAddr, Ipv4Addr, SocketAddr};
    use std::time::Instant;

    fn make_node(idx: u16, rack: Option<&str>, zone: Option<&str>) -> NodeInfo {
        NodeInfo {
            id: format!("node-{idx}"),
            swim_addr: SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), 7_900 + idx),
            api_addr: SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), 5_000 + idx),
            rack: rack.map(str::to_string),
            zone: zone.map(str::to_string),
        }
    }

    fn init_membership() -> Arc<SwimMembership> {
        let local = make_node(0, Some("rack-a"), Some("zone-1"));
        Arc::new(SwimMembership::new(local, SwimConfig::default()))
    }

    fn planner_with_strategy(
        membership: Arc<SwimMembership>,
        placement: PlacementStrategy,
    ) -> SpaceAssignmentPlanner {
        let config = ReplicationConfig {
            placement,
            ..ReplicationConfig::default()
        };
        SpaceAssignmentPlanner::new(membership, &config)
    }

    #[test]
    fn planner_errors_when_not_enough_nodes() {
        let membership = init_membership();
        let planner = planner_with_strategy(Arc::clone(&membership), PlacementStrategy::Random);
        let space = MemorySpaceId::try_from("alpha").unwrap();
        match planner.plan(&space, 2) {
            Err(ClusterError::InsufficientHealthyNodes {
                required,
                available,
            }) => {
                assert_eq!(required, 3);
                assert_eq!(available, 1);
            }
            other => panic!("unexpected planner result: {other:?}"),
        }
    }

    #[test]
    fn planner_prefers_deterministic_primary() {
        let membership = init_membership();
        let now = Instant::now();
        membership.upsert_member(make_node(1, None, None), 0, now);
        membership.upsert_member(make_node(2, None, None), 0, now);
        membership.upsert_member(make_node(3, None, None), 0, now);

        let planner = planner_with_strategy(Arc::clone(&membership), PlacementStrategy::Random);
        let space = MemorySpaceId::try_from("beta").unwrap();
        let assignment = planner.plan(&space, 2).unwrap();
        let expected = planner.plan(&space, 2).unwrap();
        assert_eq!(assignment.primary.id, expected.primary.id);
        assert_eq!(assignment.replicas.len(), 2);
    }

    #[test]
    fn rack_awareness_spreads_nodes_when_possible() {
        let membership = init_membership();
        let now = Instant::now();
        membership.upsert_member(make_node(1, Some("rack-a"), Some("zone-1")), 0, now);
        membership.upsert_member(make_node(2, Some("rack-b"), Some("zone-1")), 0, now);
        membership.upsert_member(make_node(3, Some("rack-b"), Some("zone-2")), 0, now);

        let planner = planner_with_strategy(Arc::clone(&membership), PlacementStrategy::RackAware);
        let space = MemorySpaceId::try_from("gamma").unwrap();
        let assignment = planner.plan(&space, 2).unwrap();
        let mut racks: HashMap<&str, usize> = HashMap::new();
        if let Some(rack) = assignment.primary.rack.as_deref() {
            *racks.entry(rack).or_insert(0) += 1;
        }
        for replica in &assignment.replicas {
            if let Some(rack) = replica.rack.as_deref() {
                *racks.entry(rack).or_insert(0) += 1;
            }
        }
        assert_eq!(racks.get("rack-b").copied().unwrap_or_default(), 1);
    }
}
