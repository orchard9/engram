use std::sync::Arc;
use std::time::Instant;

use dashmap::DashMap;
use dashmap::mapref::entry::Entry;

use crate::MemorySpaceId;
use crate::cluster::config::ReplicationConfig;
use crate::cluster::error::ClusterError;
use crate::cluster::placement::{SpaceAssignment, SpaceAssignmentPlanner};
use crate::metrics;

/// Cached assignment with versioning metadata for migration planning.
#[derive(Debug, Clone)]
pub struct CachedAssignment {
    /// Current assignment for the memory space.
    pub assignment: SpaceAssignment,
    /// Monotonic version used to coordinate migrations.
    pub version: u64,
    /// Timestamp when the assignment was last computed.
    pub assigned_at: Instant,
}

/// Snapshot of cached placements used for diagnostics and health endpoints.
#[derive(Debug, Clone)]
pub struct AssignmentSnapshot {
    /// Number of spaces currently cached in the manager.
    pub cached_spaces: usize,
    /// Per-node summary of primary ownership counts.
    pub per_node: Vec<NodeAssignmentLoad>,
}

/// Per-node assignment counts exposed via diagnostics.
#[derive(Debug, Clone)]
pub struct NodeAssignmentLoad {
    /// Stable identifier for the cluster node.
    pub node_id: String,
    /// Number of primary spaces owned by the node according to the cache.
    pub primary_spaces: usize,
}

/// Thread-safe cache that memoises placement decisions and tracks versions for rebalancing.
pub struct SpaceAssignmentManager {
    planner: Arc<SpaceAssignmentPlanner>,
    replication: ReplicationConfig,
    cache: DashMap<MemorySpaceId, CachedAssignment>,
    versions: DashMap<MemorySpaceId, u64>,
    per_node_counts: DashMap<String, usize>,
}

impl SpaceAssignmentManager {
    /// Create a manager backed by the provided planner/config.
    #[must_use]
    pub fn new(planner: Arc<SpaceAssignmentPlanner>, replication: &ReplicationConfig) -> Self {
        Self {
            planner,
            replication: replication.clone(),
            cache: DashMap::new(),
            versions: DashMap::new(),
            per_node_counts: DashMap::new(),
        }
    }

    /// Return the cached assignment (if present) or compute a new one via the planner.
    pub fn assign(&self, space: &MemorySpaceId) -> Result<SpaceAssignment, ClusterError> {
        if let Some(entry) = self.cache.get(space) {
            return Ok(entry.assignment.clone());
        }
        let cached = self.recompute(space)?;
        Ok(cached.assignment)
    }

    /// Force a new assignment to be computed, invalidating any cached placement.
    pub fn recompute(&self, space: &MemorySpaceId) -> Result<CachedAssignment, ClusterError> {
        let planned = self.planner.plan(space, self.replication.factor)?;
        let cached = self.store_assignment(space, planned);
        metrics::increment_counter(metrics::CLUSTER_ASSIGNMENTS_TOTAL, 1);
        Ok(cached)
    }

    /// Remove a cached space assignment, returning it to callers (for diagnostics/rebalancing).
    #[must_use]
    pub fn invalidate(&self, space: &MemorySpaceId) -> Option<CachedAssignment> {
        let removed = self.cache.remove(space).map(|(_, entry)| entry);
        if let Some(entry) = &removed {
            self.update_counts(Some(&entry.assignment.primary.id), None);
        }
        removed
    }

    /// Enumerate spaces where the provided node is currently the cached primary.
    #[must_use]
    pub fn spaces_assigned_to(&self, node_id: &str) -> Vec<MemorySpaceId> {
        self.cache
            .iter()
            .filter_map(|entry| {
                (entry.value().assignment.primary.id == node_id).then(|| entry.key().clone())
            })
            .collect()
    }

    /// Lightweight snapshot of cached assignments for observability surfaces.
    #[must_use]
    pub fn snapshot(&self) -> AssignmentSnapshot {
        let per_node = self
            .per_node_counts
            .iter()
            .map(|entry| NodeAssignmentLoad {
                node_id: entry.key().clone(),
                primary_spaces: *entry.value(),
            })
            .collect();
        AssignmentSnapshot {
            cached_spaces: self.cache.len(),
            per_node,
        }
    }

    /// List of cached spaces currently tracked by the manager.
    #[must_use]
    pub fn cached_spaces(&self) -> Vec<MemorySpaceId> {
        self.cache.iter().map(|entry| entry.key().clone()).collect()
    }

    /// Number of cached entries (primaries + replicas share the same record).
    #[must_use]
    pub fn cached_len(&self) -> usize {
        self.cache.len()
    }

    /// Expose the replication configuration (needed for CLI timeouts/metadata).
    #[must_use]
    pub const fn replication(&self) -> &ReplicationConfig {
        &self.replication
    }

    fn store_assignment(
        &self,
        space: &MemorySpaceId,
        assignment: SpaceAssignment,
    ) -> CachedAssignment {
        let version = self.next_version(space);
        let cached = CachedAssignment {
            assignment,
            version,
            assigned_at: Instant::now(),
        };
        let previous = self.cache.insert(space.clone(), cached.clone());
        let previous_primary = previous.map(|entry| entry.assignment.primary.id);
        self.update_counts(
            previous_primary.as_deref(),
            Some(&cached.assignment.primary.id),
        );
        cached
    }

    fn next_version(&self, space: &MemorySpaceId) -> u64 {
        match self.versions.entry(space.clone()) {
            Entry::Occupied(mut entry) => {
                let next = entry.get().saturating_add(1);
                *entry.get_mut() = next;
                next
            }
            Entry::Vacant(entry) => {
                entry.insert(1);
                1
            }
        }
    }

    fn update_counts(&self, previous: Option<&str>, next: Option<&str>) {
        if let Some(node) = previous {
            let new_count = match self.per_node_counts.entry(node.to_string()) {
                Entry::Occupied(mut entry) => {
                    let updated = entry.get().saturating_sub(1);
                    if updated == 0 {
                        entry.remove();
                        0
                    } else {
                        *entry.get_mut() = updated;
                        updated
                    }
                }
                Entry::Vacant(_) => 0,
            };
            metrics::record_gauge_with_labels(
                metrics::CLUSTER_SPACES_PER_NODE,
                new_count as f64,
                &[("node", node.to_string())],
            );
        }

        if let Some(node) = next {
            let updated = match self.per_node_counts.entry(node.to_string()) {
                Entry::Occupied(mut entry) => {
                    let updated = entry.get().saturating_add(1);
                    *entry.get_mut() = updated;
                    updated
                }
                Entry::Vacant(entry) => {
                    entry.insert(1);
                    1
                }
            };
            metrics::record_gauge_with_labels(
                metrics::CLUSTER_SPACES_PER_NODE,
                updated as f64,
                &[("node", node.to_string())],
            );
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use super::*;
    use crate::cluster::config::{PlacementStrategy, ReplicationConfig, SwimConfig};
    use crate::cluster::membership::{NodeInfo, SwimMembership};
    use crate::cluster::placement::SpaceAssignmentPlanner;
    use std::net::{IpAddr, Ipv4Addr, SocketAddr};
    use std::time::Instant;

    fn membership_with_peers() -> Arc<SwimMembership> {
        let local = NodeInfo::new(
            "node-local",
            SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), 7_900),
            SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), 5_000),
            None,
            None,
        );
        let membership = Arc::new(SwimMembership::new(local, SwimConfig::default()));
        let now = Instant::now();
        for idx in 1..=3 {
            let node = NodeInfo::new(
                format!("node-{idx}"),
                SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), 7_900 + idx),
                SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), 5_000 + idx),
                None,
                None,
            );
            membership.upsert_member(node, u64::from(idx), now);
        }
        membership
    }

    fn make_manager(strategy: PlacementStrategy) -> SpaceAssignmentManager {
        let membership = membership_with_peers();
        let replication = ReplicationConfig {
            placement: strategy,
            ..ReplicationConfig::default()
        };
        let planner = Arc::new(SpaceAssignmentPlanner::new(
            Arc::clone(&membership),
            &replication,
        ));
        SpaceAssignmentManager::new(planner, &replication)
    }

    #[test]
    fn manager_memoises_assignments() {
        let manager = make_manager(PlacementStrategy::Random);
        let space = MemorySpaceId::try_from("alpha").unwrap();
        let first = manager.assign(&space).unwrap();
        let second = manager.assign(&space).unwrap();
        assert_eq!(first.primary.id, second.primary.id);
        assert_eq!(manager.snapshot().cached_spaces, 1);
    }

    #[test]
    fn invalidate_removes_primary_counts() {
        let manager = make_manager(PlacementStrategy::Random);
        let space = MemorySpaceId::try_from("beta").unwrap();
        let assignment = manager.assign(&space).unwrap();
        assert_eq!(manager.spaces_assigned_to(&assignment.primary.id).len(), 1);
        let _ = manager.invalidate(&space);
        assert!(
            manager
                .spaces_assigned_to(&assignment.primary.id)
                .is_empty()
        );
    }
}
