# Task 004: Memory Space Partitioning and Assignment

**Status**: Pending
**Estimated Duration**: 3 days
**Dependencies**: Task 001 (SWIM membership), Task 002 (Discovery)
**Owner**: TBD

## Objective

Implement consistent hashing-based memory space assignment to cluster nodes with topology-aware replica placement. This task provides the fundamental data partitioning layer that determines which nodes host which memory spaces, enabling horizontal scaling while maintaining fault tolerance.

## Research Foundation

### Consistent Hashing Algorithm Comparison

Distributed systems need to assign data partitions to nodes while minimizing reassignment on topology changes. Three primary algorithms exist:

**1. Karger's Consistent Hashing (Ring Hash)**
The original consistent hashing algorithm (Karger et al., 1997) maps both nodes and data to points on a circular ring using hash functions. Virtual nodes (vnodes) improve load balance: with 100 vnodes per physical node, the standard deviation of load is ~10%, with 99% confidence interval of [0.76, 1.28] times average load.

Pros: Minimal reassignment on node changes (only K/N keys move for K keys and N nodes), widely deployed (Cassandra, DynamoDB), excellent for stateful servers with names/IDs.

Cons: Requires storing ring structure (memory overhead scales with vnodes), complex implementation with virtual nodes, non-uniform distribution without sufficient vnodes.

**2. Jump Consistent Hash (Lamping & Veach, 2014)**
Google's stateless algorithm that computes bucket assignment directly via pseudorandom traversal. No memory overhead, virtually perfect distribution (standard deviation 0.000000764%), O(ln N) computation time.

Pros: Zero memory overhead, perfect load balance, extremely fast, deterministic.

Cons: Only returns integer bucket IDs (not arbitrary node names), nodes can only be added/removed at the end of range (no arbitrary node removal), requires mapping layer from bucket→node.

**3. Rendezvous Hashing (Highest Random Weight, 1996)**
For each key, compute hash(key, node_id) for all nodes, assign to highest weight node. Simple, fully distributed, no shared state required.

Pros: Perfect decentralization, minimal reassignment on changes, no virtual nodes needed, supports arbitrary node names, excellent for CDNs (Akamai).

Cons: O(N) lookup time (must evaluate all nodes), becomes bottleneck for large N (>1000 nodes).

### Choice for Engram: Jump Hash with Node Mapping Layer

**Rationale**: Memory spaces in Engram are identified by UUID strings, but the set of spaces is dynamic and potentially massive (millions). Jump Hash provides perfect load balance with zero memory overhead, critical for large-scale deployments. We layer a node mapping structure on top to translate integer buckets to node IDs.

**Trade-off**: We sacrifice arbitrary node removal (must add new node before removing old) for superior load balance and zero state. For planned capacity changes, this is acceptable: add new nodes, wait for rebalancing, then remove old nodes.

**Alternative considered**: Rendezvous hashing would allow arbitrary node removal but O(N) lookup per space assignment becomes prohibitive at 100+ nodes. Jump Hash O(ln N) scales better.

### Rack-Aware Replica Placement

Physical failure domains (racks, availability zones) require topology-aware replica placement to survive correlated failures. HDFS rack awareness policy: for replication factor R=3, place first replica locally, second replica on different rack, third replica on yet another rack (if available). Never place >2 replicas on same rack.

**Engram strategy**: Assign primary via Jump Hash, then select R-1 replica nodes using modified Rendezvous hashing with rack diversity penalties. Replica selection computes scores that penalize same-rack placement exponentially.

### Rebalancing on Topology Changes

When nodes join/leave, affected space assignments must migrate to new primaries. Jump Hash property: adding node N+1 causes exactly K/(N+1) keys to reassign from existing nodes to new node - optimal minimal disruption.

**Rebalancing phases**:
1. **Discovery**: Detect topology change via SWIM membership
2. **Planning**: Compute delta (which spaces move to/from which nodes)
3. **Synchronization**: Ship space data from old primary to new primary
4. **Cutover**: Atomic pointer swap, new primary serves requests
5. **Cleanup**: Old primary deletes migrated space data

**Zero-downtime requirement**: During migration, both old and new primaries serve reads. Writes go to new primary only after cutover. Replicas remain available throughout.

## Technical Specification

### Core Data Structures

```rust
// engram-core/src/cluster/space_assignment.rs

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use dashmap::DashMap;
use tokio::sync::RwLock;

use crate::cluster::membership::{SwimMembership, NodeInfo};
use crate::MemorySpaceId;

/// Manages assignment of memory spaces to cluster nodes
pub struct SpaceAssignmentManager {
    /// SWIM membership for node list
    membership: Arc<SwimMembership>,

    /// Current space assignments (space_id -> Assignment)
    assignments: DashMap<MemorySpaceId, SpaceAssignment>,

    /// Placement strategy (determines replicas)
    placement: Arc<PlacementStrategy>,

    /// Rebalancing coordinator
    rebalancer: Arc<RebalancingCoordinator>,

    /// Configuration
    config: AssignmentConfig,
}

#[derive(Debug, Clone)]
pub struct AssignmentConfig {
    /// Number of replicas per space (default: 3)
    pub replication_factor: usize,

    /// Number of virtual buckets for Jump Hash (default: 1024)
    /// More buckets = better load balance on small clusters
    pub num_buckets: u32,

    /// Enable rack-aware placement (default: true)
    pub rack_aware: bool,

    /// Enable zone-aware placement (default: true)
    pub zone_aware: bool,
}

impl Default for AssignmentConfig {
    fn default() -> Self {
        Self {
            replication_factor: 3,
            num_buckets: 1024,
            rack_aware: true,
            zone_aware: true,
        }
    }
}

/// Assignment of a memory space to nodes
#[derive(Debug, Clone)]
pub struct SpaceAssignment {
    /// Memory space ID
    pub space_id: MemorySpaceId,

    /// Primary node (serves writes and reads)
    pub primary_node_id: String,

    /// Replica nodes (serve reads, async replication)
    pub replica_node_ids: Vec<String>,

    /// Assignment version (increments on each reassignment)
    pub version: u64,

    /// Timestamp of last assignment change
    pub assigned_at: chrono::DateTime<chrono::Utc>,
}

impl SpaceAssignmentManager {
    pub fn new(
        membership: Arc<SwimMembership>,
        config: AssignmentConfig,
    ) -> Self {
        let placement = Arc::new(PlacementStrategy::new(
            membership.clone(),
            config.clone(),
        ));

        let rebalancer = Arc::new(RebalancingCoordinator::new(
            membership.clone(),
        ));

        Self {
            membership,
            assignments: DashMap::new(),
            placement,
            rebalancer,
            config,
        }
    }

    /// Assign a memory space to nodes (primary + replicas)
    pub fn assign_space(&self, space_id: &MemorySpaceId) -> Result<SpaceAssignment, ClusterError> {
        // Check if already assigned
        if let Some(existing) = self.assignments.get(space_id) {
            return Ok(existing.clone());
        }

        // 1. Hash space ID to bucket using Jump Hash
        let bucket = self.jump_hash(space_id);

        // 2. Map bucket to primary node
        let primary_node = self.bucket_to_node(bucket)?;

        // 3. Select replica nodes using topology-aware placement
        let replica_nodes = self.placement.select_replicas(
            space_id,
            &primary_node,
            self.config.replication_factor - 1,
        )?;

        // 4. Create assignment
        let assignment = SpaceAssignment {
            space_id: space_id.clone(),
            primary_node_id: primary_node,
            replica_node_ids: replica_nodes,
            version: 1,
            assigned_at: chrono::Utc::now(),
        };

        // 5. Store assignment
        self.assignments.insert(space_id.clone(), assignment.clone());

        info!(
            "Assigned space {} to primary {} with {} replicas",
            space_id,
            assignment.primary_node_id,
            assignment.replica_node_ids.len()
        );

        Ok(assignment)
    }

    /// Get assignment for a memory space
    pub fn get_assignment(&self, space_id: &MemorySpaceId) -> Option<SpaceAssignment> {
        self.assignments.get(space_id).map(|a| a.clone())
    }

    /// List all assignments
    pub fn list_assignments(&self) -> Vec<SpaceAssignment> {
        self.assignments
            .iter()
            .map(|entry| entry.value().clone())
            .collect()
    }

    /// Jump Consistent Hash implementation
    fn jump_hash(&self, space_id: &MemorySpaceId) -> u32 {
        // Use SipHash for space_id -> u64
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        space_id.hash(&mut hasher);
        let key = hasher.finish();

        // Jump Hash algorithm (Lamping & Veach, 2014)
        jump_hash_u64(key, self.config.num_buckets as i32) as u32
    }

    /// Map Jump Hash bucket to actual node
    fn bucket_to_node(&self, bucket: u32) -> Result<String, ClusterError> {
        // Get sorted list of alive nodes
        let mut nodes: Vec<_> = self.membership
            .members
            .iter()
            .filter(|entry| entry.value().state == NodeState::Alive)
            .map(|entry| entry.value().id.clone())
            .collect();

        if nodes.is_empty() {
            return Err(ClusterError::NoNodesAvailable);
        }

        nodes.sort(); // Deterministic ordering

        // Map bucket to node via modulo
        let node_idx = (bucket as usize) % nodes.len();
        Ok(nodes[node_idx].clone())
    }
}

/// Jump Hash algorithm implementation
/// Based on "A Fast, Minimal Memory, Consistent Hash Algorithm" (Lamping & Veach, 2014)
fn jump_hash_u64(mut key: u64, num_buckets: i32) -> i32 {
    let mut b: i64 = -1;
    let mut j: i64 = 0;

    while j < num_buckets as i64 {
        b = j;
        key = key.wrapping_mul(2862933555777941757).wrapping_add(1);
        j = ((b + 1) as f64 * (((1i64 << 31) as f64) / (((key >> 33) + 1) as f64))) as i64;
    }

    b as i32
}
```

### Topology-Aware Placement Strategy

```rust
// engram-core/src/cluster/placement.rs

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use crate::cluster::membership::{SwimMembership, NodeInfo};
use crate::MemorySpaceId;

/// Strategy for selecting replica nodes with topology awareness
pub struct PlacementStrategy {
    membership: Arc<SwimMembership>,
    config: AssignmentConfig,
}

impl PlacementStrategy {
    pub fn new(
        membership: Arc<SwimMembership>,
        config: AssignmentConfig,
    ) -> Self {
        Self {
            membership,
            config,
        }
    }

    /// Select N replica nodes for a space, avoiding topology conflicts
    pub fn select_replicas(
        &self,
        space_id: &MemorySpaceId,
        primary_node_id: &str,
        num_replicas: usize,
    ) -> Result<Vec<String>, ClusterError> {
        let mut candidates = self.get_candidate_nodes(primary_node_id)?;

        if candidates.is_empty() {
            return Err(ClusterError::InsufficientNodes {
                required: num_replicas,
                available: 0,
            });
        }

        // Get primary node topology
        let primary_topology = self.get_node_topology(primary_node_id)?;

        // Score each candidate using modified Rendezvous hashing with rack penalties
        let mut scored: Vec<_> = candidates
            .into_iter()
            .map(|node_id| {
                let score = self.compute_placement_score(
                    space_id,
                    &node_id,
                    &primary_topology,
                );
                (node_id, score)
            })
            .collect();

        // Sort by score descending (highest score = best placement)
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        // Select top N replicas with rack diversity
        let mut selected = Vec::new();
        let mut used_racks = HashSet::new();
        let mut used_zones = HashSet::new();

        // Primary's rack/zone are already "used"
        if let Some(rack) = &primary_topology.rack {
            used_racks.insert(rack.clone());
        }
        if let Some(zone) = &primary_topology.zone {
            used_zones.insert(zone.clone());
        }

        for (node_id, score) in scored {
            if selected.len() >= num_replicas {
                break;
            }

            let topology = self.get_node_topology(&node_id)?;

            // Check rack diversity constraint
            if self.config.rack_aware {
                if let Some(rack) = &topology.rack {
                    if used_racks.contains(rack) && used_racks.len() < num_replicas {
                        // Skip this node, prefer rack diversity
                        continue;
                    }
                    used_racks.insert(rack.clone());
                }
            }

            // Check zone diversity constraint
            if self.config.zone_aware {
                if let Some(zone) = &topology.zone {
                    if used_zones.contains(zone) && used_zones.len() < num_replicas {
                        // Skip this node, prefer zone diversity
                        continue;
                    }
                    used_zones.insert(zone.clone());
                }
            }

            selected.push(node_id);
        }

        if selected.len() < num_replicas {
            // Couldn't satisfy diversity constraints, relax and fill remaining
            for (node_id, _) in scored {
                if selected.len() >= num_replicas {
                    break;
                }
                if !selected.contains(&node_id) {
                    selected.push(node_id);
                }
            }
        }

        Ok(selected)
    }

    /// Compute placement score using Rendezvous hash with topology penalties
    fn compute_placement_score(
        &self,
        space_id: &MemorySpaceId,
        node_id: &str,
        primary_topology: &NodeTopology,
    ) -> f64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        // Base score: hash(space_id, node_id)
        let mut hasher = DefaultHasher::new();
        space_id.hash(&mut hasher);
        node_id.hash(&mut hasher);
        let hash = hasher.finish();

        let mut score = (hash as f64) / (u64::MAX as f64);

        // Apply topology penalties
        if let Ok(topology) = self.get_node_topology(node_id) {
            // Same rack penalty: reduce score by 50%
            if self.config.rack_aware {
                if topology.rack == primary_topology.rack {
                    score *= 0.5;
                }
            }

            // Same zone penalty: reduce score by 30%
            if self.config.zone_aware {
                if topology.zone == primary_topology.zone {
                    score *= 0.7;
                }
            }
        }

        score
    }

    fn get_candidate_nodes(&self, exclude_node_id: &str) -> Result<Vec<String>, ClusterError> {
        let candidates: Vec<_> = self.membership
            .members
            .iter()
            .filter(|entry| {
                let node = entry.value();
                node.state == NodeState::Alive && node.id != exclude_node_id
            })
            .map(|entry| entry.value().id.clone())
            .collect();

        Ok(candidates)
    }

    fn get_node_topology(&self, node_id: &str) -> Result<NodeTopology, ClusterError> {
        self.membership
            .members
            .get(node_id)
            .map(|node| NodeTopology {
                rack: node.topology.rack.clone(),
                zone: node.topology.zone.clone(),
            })
            .ok_or(ClusterError::NodeNotFound {
                node_id: node_id.to_string(),
            })
    }
}

#[derive(Debug, Clone, Default)]
pub struct NodeTopology {
    pub rack: Option<String>,
    pub zone: Option<String>,
}
```

### Rebalancing Coordinator

```rust
// engram-core/src/cluster/rebalancing.rs

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use tokio::sync::RwLock;

use crate::cluster::membership::{SwimMembership, NodeInfo, NodeState};
use crate::MemorySpaceId;

/// Coordinates rebalancing when cluster topology changes
pub struct RebalancingCoordinator {
    membership: Arc<SwimMembership>,
    active_rebalances: Arc<RwLock<HashSet<MemorySpaceId>>>,
}

impl RebalancingCoordinator {
    pub fn new(membership: Arc<SwimMembership>) -> Self {
        Self {
            membership,
            active_rebalances: Arc::new(RwLock::new(HashSet::new())),
        }
    }

    /// Detect spaces that need rebalancing due to membership change
    pub async fn compute_rebalancing_plan(
        &self,
        current_assignments: &HashMap<MemorySpaceId, SpaceAssignment>,
        assignment_manager: &SpaceAssignmentManager,
    ) -> RebalancingPlan {
        let mut plan = RebalancingPlan::default();

        for (space_id, current) in current_assignments {
            // Recompute assignment with current membership
            let ideal = match assignment_manager.assign_space(space_id) {
                Ok(a) => a,
                Err(e) => {
                    warn!("Failed to compute ideal assignment for {}: {}", space_id, e);
                    continue;
                }
            };

            // Check if primary changed
            if current.primary_node_id != ideal.primary_node_id {
                plan.primary_migrations.push(PrimaryMigration {
                    space_id: space_id.clone(),
                    from_node: current.primary_node_id.clone(),
                    to_node: ideal.primary_node_id.clone(),
                });
            }

            // Check if replicas changed
            let current_replicas: HashSet<_> = current.replica_node_ids.iter().collect();
            let ideal_replicas: HashSet<_> = ideal.replica_node_ids.iter().collect();

            let to_add: Vec<_> = ideal_replicas
                .difference(&current_replicas)
                .map(|s| s.to_string())
                .collect();

            let to_remove: Vec<_> = current_replicas
                .difference(&ideal_replicas)
                .map(|s| s.to_string())
                .collect();

            if !to_add.is_empty() || !to_remove.is_empty() {
                plan.replica_adjustments.push(ReplicaAdjustment {
                    space_id: space_id.clone(),
                    add_replicas: to_add,
                    remove_replicas: to_remove,
                });
            }
        }

        info!(
            "Rebalancing plan: {} primary migrations, {} replica adjustments",
            plan.primary_migrations.len(),
            plan.replica_adjustments.len()
        );

        plan
    }

    /// Execute rebalancing plan
    pub async fn execute_rebalancing(
        &self,
        plan: RebalancingPlan,
        assignment_manager: Arc<SpaceAssignmentManager>,
    ) -> Result<RebalancingReport, ClusterError> {
        let mut report = RebalancingReport::default();

        // Phase 1: Add new replicas (no disruption)
        for adjustment in &plan.replica_adjustments {
            for new_replica in &adjustment.add_replicas {
                info!(
                    "Adding replica for {} on node {}",
                    adjustment.space_id, new_replica
                );

                // Signal new replica node to start syncing
                match self.initiate_replica_sync(&adjustment.space_id, new_replica).await {
                    Ok(()) => report.replicas_added += 1,
                    Err(e) => {
                        warn!("Failed to add replica: {}", e);
                        report.failures.push(format!(
                            "Failed to add replica for {} on {}: {}",
                            adjustment.space_id, new_replica, e
                        ));
                    }
                }
            }
        }

        // Phase 2: Migrate primaries (atomic cutover)
        for migration in &plan.primary_migrations {
            info!(
                "Migrating primary for {} from {} to {}",
                migration.space_id, migration.from_node, migration.to_node
            );

            match self.migrate_primary(migration, &assignment_manager).await {
                Ok(()) => report.primaries_migrated += 1,
                Err(e) => {
                    error!("Failed to migrate primary: {}", e);
                    report.failures.push(format!(
                        "Failed to migrate primary for {}: {}",
                        migration.space_id, e
                    ));
                }
            }
        }

        // Phase 3: Remove old replicas (cleanup)
        for adjustment in &plan.replica_adjustments {
            for old_replica in &adjustment.remove_replicas {
                info!(
                    "Removing replica for {} from node {}",
                    adjustment.space_id, old_replica
                );

                match self.remove_replica(&adjustment.space_id, old_replica).await {
                    Ok(()) => report.replicas_removed += 1,
                    Err(e) => {
                        warn!("Failed to remove replica: {}", e);
                        report.failures.push(format!(
                            "Failed to remove replica for {} from {}: {}",
                            adjustment.space_id, old_replica, e
                        ));
                    }
                }
            }
        }

        info!(
            "Rebalancing complete: {} primaries migrated, {} replicas added, {} replicas removed, {} failures",
            report.primaries_migrated,
            report.replicas_added,
            report.replicas_removed,
            report.failures.len()
        );

        Ok(report)
    }

    async fn initiate_replica_sync(
        &self,
        space_id: &MemorySpaceId,
        target_node: &str,
    ) -> Result<(), ClusterError> {
        // Send RPC to target node to start syncing space data
        // Implementation will use gRPC client from Task 006
        todo!("Implement in Task 006: Distributed Routing Layer")
    }

    async fn migrate_primary(
        &self,
        migration: &PrimaryMigration,
        assignment_manager: &SpaceAssignmentManager,
    ) -> Result<(), ClusterError> {
        // 1. Ensure new primary has synced data
        // 2. Pause writes to old primary
        // 3. Wait for replication to catch up
        // 4. Atomic cutover: update assignment to new primary
        // 5. Resume writes on new primary
        todo!("Implement in Task 005: Replication Protocol")
    }

    async fn remove_replica(
        &self,
        space_id: &MemorySpaceId,
        node_id: &str,
    ) -> Result<(), ClusterError> {
        // Send RPC to node to delete space data
        todo!("Implement in Task 006: Distributed Routing Layer")
    }
}

#[derive(Debug, Default)]
pub struct RebalancingPlan {
    pub primary_migrations: Vec<PrimaryMigration>,
    pub replica_adjustments: Vec<ReplicaAdjustment>,
}

#[derive(Debug, Clone)]
pub struct PrimaryMigration {
    pub space_id: MemorySpaceId,
    pub from_node: String,
    pub to_node: String,
}

#[derive(Debug, Clone)]
pub struct ReplicaAdjustment {
    pub space_id: MemorySpaceId,
    pub add_replicas: Vec<String>,
    pub remove_replicas: Vec<String>,
}

#[derive(Debug, Default)]
pub struct RebalancingReport {
    pub primaries_migrated: usize,
    pub replicas_added: usize,
    pub replicas_removed: usize,
    pub failures: Vec<String>,
}
```

### Integration with Memory Space Registry

```rust
// engram-core/src/registry/memory_space.rs (modifications)

use crate::cluster::space_assignment::{SpaceAssignmentManager, SpaceAssignment};

pub struct MemorySpaceRegistry {
    // Existing fields...
    handles: DashMap<MemorySpaceId, Arc<SpaceHandle>>,

    // New: cluster space assignment (optional, only in cluster mode)
    space_assignment: Option<Arc<SpaceAssignmentManager>>,
}

impl MemorySpaceRegistry {
    /// Enable cluster mode with space assignment
    pub fn enable_cluster_mode(
        &mut self,
        assignment_manager: Arc<SpaceAssignmentManager>,
    ) {
        self.space_assignment = Some(assignment_manager);
    }

    /// Get or create space handle (cluster-aware)
    pub async fn get_or_create(
        &self,
        space_id: &MemorySpaceId,
    ) -> Result<Arc<SpaceHandle>, MemorySpaceError> {
        // Check if we have local handle
        if let Some(handle) = self.handles.get(space_id) {
            return Ok(handle.clone());
        }

        // In cluster mode, check if we're assigned this space
        if let Some(assignment_manager) = &self.space_assignment {
            let assignment = assignment_manager.assign_space(space_id)?;

            let local_node_id = get_local_node_id(); // From SWIM membership

            // Check if we're primary or replica for this space
            if assignment.primary_node_id != local_node_id
                && !assignment.replica_node_ids.contains(&local_node_id)
            {
                // Not assigned to us, don't create local handle
                return Err(MemorySpaceError::NotAssignedToThisNode {
                    space_id: space_id.clone(),
                    assigned_to: assignment.primary_node_id,
                });
            }
        }

        // Create local handle (same as single-node mode)
        self.create_space_handle(space_id).await
    }
}
```

## Files to Create

1. `engram-core/src/cluster/space_assignment.rs` - Space assignment manager
2. `engram-core/src/cluster/placement.rs` - Topology-aware placement strategy
3. `engram-core/src/cluster/rebalancing.rs` - Rebalancing coordinator
4. `engram-core/src/cluster/jump_hash.rs` - Jump Hash implementation
5. `engram-core/src/cluster/error.rs` - Cluster error types (extend)

## Files to Modify

1. `engram-core/src/cluster/mod.rs` - Export space assignment modules
2. `engram-core/src/cluster/membership.rs` - Add topology fields to NodeInfo
3. `engram-core/src/registry/memory_space.rs` - Integrate space assignment
4. `engram-cli/src/cluster.rs` - Initialize space assignment on startup
5. `engram-core/src/metrics/mod.rs` - Add assignment metrics

## Testing Strategy

### Unit Tests

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_jump_hash_distribution() {
        let num_buckets = 100;
        let num_keys = 100_000;
        let mut bucket_counts = vec![0; num_buckets];

        for i in 0..num_keys {
            let bucket = jump_hash_u64(i as u64, num_buckets as i32);
            bucket_counts[bucket as usize] += 1;
        }

        // Check distribution uniformity
        let avg = num_keys / num_buckets;
        let max_deviation = (avg as f64 * 0.2) as usize; // Allow 20% deviation

        for count in bucket_counts {
            assert!(
                (count as i32 - avg as i32).abs() < max_deviation as i32,
                "Bucket count {} deviates too much from average {}",
                count,
                avg
            );
        }
    }

    #[test]
    fn test_jump_hash_minimal_reassignment() {
        // Test that adding one bucket only reassigns ~1/N keys
        let num_keys = 10_000;
        let buckets_before = 10;
        let buckets_after = 11;

        let mut reassigned = 0;

        for i in 0..num_keys {
            let bucket_before = jump_hash_u64(i as u64, buckets_before);
            let bucket_after = jump_hash_u64(i as u64, buckets_after);

            if bucket_before != bucket_after {
                reassigned += 1;
            }
        }

        // Should reassign approximately num_keys / buckets_after
        let expected = num_keys / buckets_after;
        let tolerance = expected / 10; // 10% tolerance

        assert!(
            (reassigned as i32 - expected as i32).abs() < tolerance as i32,
            "Reassigned {} keys, expected ~{} (10% tolerance)",
            reassigned,
            expected
        );
    }

    #[tokio::test]
    async fn test_topology_aware_placement() {
        let membership = Arc::new(SwimMembership::new_test());

        // Add nodes in different racks
        membership.add_node(NodeInfo {
            id: "node1".into(),
            topology: NodeTopology {
                rack: Some("rack1".into()),
                zone: Some("zone1".into()),
            },
            ..Default::default()
        });

        membership.add_node(NodeInfo {
            id: "node2".into(),
            topology: NodeTopology {
                rack: Some("rack2".into()),
                zone: Some("zone1".into()),
            },
            ..Default::default()
        });

        membership.add_node(NodeInfo {
            id: "node3".into(),
            topology: NodeTopology {
                rack: Some("rack3".into()),
                zone: Some("zone2".into()),
            },
            ..Default::default()
        });

        let config = AssignmentConfig {
            replication_factor: 3,
            rack_aware: true,
            zone_aware: true,
            ..Default::default()
        };

        let placement = PlacementStrategy::new(membership, config);

        let space_id = MemorySpaceId::new();
        let primary = "node1";

        let replicas = placement.select_replicas(&space_id, primary, 2).unwrap();

        // Should select node2 and node3 (different racks)
        assert_eq!(replicas.len(), 2);
        assert!(replicas.contains(&"node2".to_string()));
        assert!(replicas.contains(&"node3".to_string()));
    }

    #[tokio::test]
    async fn test_assignment_manager_basic() {
        let membership = Arc::new(SwimMembership::new_test());

        // Add 3 nodes
        for i in 0..3 {
            membership.add_node(NodeInfo {
                id: format!("node{}", i),
                state: NodeState::Alive,
                ..Default::default()
            });
        }

        let manager = SpaceAssignmentManager::new(
            membership,
            AssignmentConfig::default(),
        );

        let space_id = MemorySpaceId::new();
        let assignment = manager.assign_space(&space_id).unwrap();

        // Should have 1 primary + 2 replicas
        assert!(!assignment.primary_node_id.is_empty());
        assert_eq!(assignment.replica_node_ids.len(), 2);

        // Primary should not be in replicas
        assert!(!assignment.replica_node_ids.contains(&assignment.primary_node_id));
    }
}
```

### Integration Tests

```rust
// engram-core/tests/space_assignment_integration.rs

#[tokio::test]
async fn test_space_assignment_across_cluster() {
    let cluster = TestCluster::new(5).await;

    let assignment_manager = cluster.node(0).space_assignment();

    // Assign 100 spaces
    let mut assignments = Vec::new();
    for _ in 0..100 {
        let space_id = MemorySpaceId::new();
        let assignment = assignment_manager.assign_space(&space_id).unwrap();
        assignments.push(assignment);
    }

    // Check load balance
    let mut node_loads = HashMap::new();
    for assignment in &assignments {
        *node_loads.entry(assignment.primary_node_id.clone()).or_insert(0) += 1;
    }

    // Each node should have ~20 spaces (100/5 = 20)
    for (node, count) in node_loads {
        assert!(
            (count as i32 - 20).abs() <= 5,
            "Node {} has {} spaces, expected ~20",
            node,
            count
        );
    }
}

#[tokio::test]
async fn test_rebalancing_on_node_join() {
    let cluster = TestCluster::new(3).await;
    let assignment_manager = cluster.node(0).space_assignment();

    // Assign 30 spaces to 3 nodes (10 each)
    let mut space_ids = Vec::new();
    for _ in 0..30 {
        let space_id = MemorySpaceId::new();
        assignment_manager.assign_space(&space_id).unwrap();
        space_ids.push(space_id);
    }

    // Add 4th node
    cluster.add_node().await;

    // Wait for membership propagation
    tokio::time::sleep(Duration::from_secs(2)).await;

    // Compute rebalancing plan
    let current_assignments: HashMap<_, _> = space_ids
        .iter()
        .map(|id| (id.clone(), assignment_manager.get_assignment(id).unwrap()))
        .collect();

    let rebalancer = cluster.node(0).rebalancer();
    let plan = rebalancer
        .compute_rebalancing_plan(&current_assignments, &assignment_manager)
        .await;

    // Should migrate ~7-8 spaces to new node (30/4 = 7.5)
    assert!(
        plan.primary_migrations.len() >= 6 && plan.primary_migrations.len() <= 9,
        "Expected 7-8 migrations, got {}",
        plan.primary_migrations.len()
    );
}

#[tokio::test]
async fn test_rebalancing_zero_downtime() {
    let cluster = TestCluster::new(3).await;

    // Write data to space
    let space_id = MemorySpaceId::new();
    cluster.store_memory(&space_id, test_memory()).await;

    // Trigger rebalancing by adding node
    cluster.add_node().await;

    // During rebalancing, continuously query space
    let mut handles = Vec::new();
    for _ in 0..10 {
        let cluster = cluster.clone();
        let space_id = space_id.clone();

        let handle = tokio::spawn(async move {
            for _ in 0..100 {
                let result = cluster.recall_memory(&space_id).await;
                assert!(result.is_ok(), "Query failed during rebalancing");
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        });

        handles.push(handle);
    }

    // Wait for all queries to complete
    for handle in handles {
        handle.await.unwrap();
    }
}
```

### Property-Based Tests

```rust
use proptest::prelude::*;

proptest! {
    #[test]
    fn test_jump_hash_always_in_range(key: u64, num_buckets in 1..1000i32) {
        let bucket = jump_hash_u64(key, num_buckets);
        prop_assert!(bucket >= 0 && bucket < num_buckets);
    }

    #[test]
    fn test_assignment_deterministic(
        space_id: String,
        num_nodes in 1..20usize,
    ) {
        let membership = Arc::new(SwimMembership::new_test());

        // Add nodes
        for i in 0..num_nodes {
            membership.add_node(NodeInfo {
                id: format!("node{}", i),
                state: NodeState::Alive,
                ..Default::default()
            });
        }

        let manager = SpaceAssignmentManager::new(
            membership,
            AssignmentConfig::default(),
        );

        let space_id = MemorySpaceId::from(space_id);

        // Assign twice, should be identical
        let assignment1 = manager.assign_space(&space_id).unwrap();
        manager.assignments.remove(&space_id); // Clear cache
        let assignment2 = manager.assign_space(&space_id).unwrap();

        prop_assert_eq!(assignment1.primary_node_id, assignment2.primary_node_id);
        prop_assert_eq!(
            assignment1.replica_node_ids.len(),
            assignment2.replica_node_ids.len()
        );
    }
}
```

## Dependencies

Add to `engram-core/Cargo.toml`:

```toml
[dependencies]
# Existing dependencies...
# No new dependencies required - uses existing dashmap, tokio, serde, chrono
```

## Acceptance Criteria

1. Jump Hash provides even distribution: max 20% imbalance across nodes
2. Topology-aware placement: no more than 1 replica per rack (when possible)
3. Assignment is deterministic: same space always maps to same nodes
4. Rebalancing on node join: only K/(N+1) spaces migrate
5. Rebalancing on node leave: affected spaces promote replicas to primary
6. Zero downtime: queries succeed during entire rebalancing process
7. Configuration: replication factor, rack/zone awareness are configurable
8. Metrics: track assignment distribution, rebalancing operations

## Performance Targets

- Assignment computation: <100μs per space (Jump Hash O(ln N))
- Replica selection: <1ms for 100-node cluster (Rendezvous O(N))
- Rebalancing plan generation: <1s for 10,000 spaces
- Memory overhead: <1KB per space assignment
- Distribution uniformity: standard deviation <5% of mean

## Next Steps

After completing this task:
- Task 005 will implement the actual data migration during rebalancing
- Task 006 will use assignments for routing requests to correct nodes
- Task 007 will use assignments to determine gossip targets for consolidation state
- Task 009 will query assignments to determine scatter-gather targets
