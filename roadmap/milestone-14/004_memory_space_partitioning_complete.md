# Task 004: Memory Space Partitioning and Assignment

**Status**: Complete
**Estimated Duration**: 3 days
**Dependencies**: Task 001 (SWIM membership), Task 002 (Discovery)
**Owner**: TBD

## Objective

Implement consistent hashing-based memory space assignment to cluster nodes with topology-aware replica placement. This work supplies the partitioning layer that determines which nodes host which memory spaces so Engram can scale horizontally without sacrificing durability.

## Research Foundation

### Consistent Hashing Algorithm Comparison

Distributed systems need to assign data partitions to nodes while minimizing reassignment on topology changes. Key algorithms:

1. **Karger's Consistent Hashing (Ring Hash)** – Minimal reassignment but requires storing vnode rings.
2. **Jump Consistent Hash (Lamping & Veach, 2014)** – Stateless, perfect balance, O(log N) lookup.
3. **Rendezvous Hashing (Highest Random Weight, 1996)** – Decentralized but O(N) per lookup.

### Choice for Engram: Jump Hash with Node Mapping Layer

Memory spaces are identified by UUID strings and can reach millions. Jump Hash gives perfect balance with zero per-space state; we map the resulting bucket to the sorted list of alive nodes reported by SWIM.

Replica placement still needs topology awareness, so we reuse the rack/zone labels already stored on `NodeInfo` (Task 001) and apply penalties when selecting replicas.

### Rebalancing Goal

When membership changes, only the affected spaces should move. Jump Hash guarantees ~K/N reassignment when a new node joins, but we still need a coordinator to invalidate cached placements, plan migrations, and expose progress.

## Technical Specification

### Current Implementation

- `SpaceAssignmentPlanner` (`engram-core/src/cluster/placement.rs`) deterministically sorts alive nodes using a per-space hash of `(space_id, node.id)` and applies rack/zone spreading via `spread_by_label`.
- The CLI wraps the planner in `ClusterState` (`engram-cli/src/cluster.rs`) and uses it for routing/gRPC proxy decisions.

This approach works but has gaps:
1. The ranking hash is not Jump Consistent Hash, so adding/removing nodes can reshuffle a large portion of spaces.
2. Results are recomputed per request; there is no assignment cache/versioning to drive migrations.
3. Replica selection only takes the next N nodes and doesn’t penalize same rack/zone beyond simple filtering.
4. No component listens to SWIM membership updates to trigger rebalancing or admin tooling.

### Enhancements Required

1. **Jump Hash integration**: Implement Jump Consistent Hash (Lamping & Veach) and use it to pick the primary node. Calculate the bucket ID from `MemorySpaceId` and map it to the sorted alive node list. When membership changes, the mapping changes predictably.
2. **Replica diversity**: Extend `SpaceAssignmentPlanner` to call a new `select_replicas` helper that penalizes same rack/zone placement (e.g., multiply Rendezvous scores by diversity weights) using the `NodeInfo.rack`/`zone` metadata added in Task 001.
3. **Assignment cache/manager**: Add `engram-core/src/cluster/assignment.rs` with a `SpaceAssignmentManager` wrapping the planner. It caches recent assignments in a `DashMap`, tracks a `version` counter per space, and exposes helpers like `spaces_assigned_to(node_id)` for rebalancing.
4. **Rebalancing coordinator**: Subscribe to SWIM membership updates (the planner already exposes `random_members`). On join/leave, invalidate affected assignments, plan migrations (old primary ➜ new primary), and emit `MigrationPlan` structs over an internal channel. CLI tooling will consume these plans to orchestrate WAL catch-up + cutover.
5. **CLI/admin hooks**: Add gRPC/HTTP admin endpoints for `migrate_space`/`rebalance_status` so operators can trigger or monitor migrations. Routing logic (`ApiState::route_for_write` and gRPC `plan_route`) should read from the new `SpaceAssignmentManager` instead of rehashing every request.
6. **Observability**: Record Prometheus metrics for `engram_cluster_assignments_total`, `engram_cluster_rebalance_plans`, and gauge of spaces per node. Update `/cluster/health` (Task 002) to include assignment summaries.
7. **Config wiring**: Extend `ClusterConfig.replication` with Jump-hash options (e.g., `buckets`) and exposure via CLI config files (`engram-cli/config/default.toml`).

### Data Structures (Sketch)

```rust
// engram-core/src/cluster/assignment.rs
pub struct SpaceAssignmentManager {
    membership: Arc<SwimMembership>,
    planner: SpaceAssignmentPlanner,
    cache: DashMap<MemorySpaceId, CachedAssignment>,
}

pub struct CachedAssignment {
    pub assignment: SpaceAssignment,
    pub version: u64,
    pub assigned_at: Instant,
}

impl SpaceAssignmentManager {
    pub fn assign(&self, space: &MemorySpaceId, replicas: usize) -> Result<SpaceAssignment, ClusterError> {
        if let Some(entry) = self.cache.get(space) {
            return Ok(entry.assignment.clone());
        }
        let assignment = self.planner.plan(space, replicas)?;
        self.cache.insert(space.clone(), CachedAssignment { assignment: assignment.clone(), version: 1, assigned_at: Instant::now() });
        Ok(assignment)
    }

    pub fn invalidate(&self, space: &MemorySpaceId) {
        self.cache.remove(space);
    }

    pub fn spaces_assigned_to(&self, node_id: &str) -> Vec<MemorySpaceId> {
        self.cache
            .iter()
            .filter_map(|entry| (entry.value().assignment.primary.id == node_id).then(|| entry.key().clone()))
            .collect()
    }
}
```

`SpaceAssignmentPlanner::plan` should be updated to:
- use Jump Hash to pick the primary (`jump_consistent_hash(space_id_hash, alive_nodes.len())`)
- call a new `select_replicas` helper that applies rack/zone penalties instead of simple truncation
- return `NodeInfo` for primary + replicas as today so upper layers don’t change

### Rebalancing Workflow

1. **Detect**: SWIM fires `MembershipUpdate`. The coordinator computes `spaces_assigned_to(update.node_id)` and invalidates them.
2. **Plan**: For each affected space, compare old vs new assignment and enqueue a `MigrationPlan { space_id, from, to, version }` to a background worker.
3. **Sync**: CLI (or a background task) reads the plan, streams WAL + content to the new primary, and waits for replica acknowledgements.
4. **Cutover**: Once the new primary has caught up, bump the cached version and start routing writes there. Reads may continue from old primary until acknowledgement.
5. **Cleanup**: Old primary removes the space when replicas confirm the cutover.

Zero-downtime remains the requirement: replicas always cover the desired factor, reads stay available throughout, and writes only pause for the cutover RPC.

### API / CLI Integration

- `ClusterState` should store `Arc<SpaceAssignmentManager>` so HTTP/gRPC handlers call `manager.assign(space_id, replication.factor)`.
- Add admin RPCs (`POST /cluster/rebalance`, gRPC `RebalanceSpaces`) that trigger a scan when nodes are added/removed.
- Update CLI docs and `engram status` output to show per-node space counts (from the manager cache) so operators can verify balance.

## Files to Create / Modify

**Create**
1. `engram-core/src/cluster/assignment.rs` – manager/cache definitions.
2. `engram-core/src/cluster/rebalance.rs` – migration planner + background worker.
3. `engram-cli/src/admin/rebalance.rs` – HTTP/gRPC handlers for triggering migrations.

**Modify**
1. `engram-core/src/cluster/placement.rs` – integrate Jump Hash + replica penalties.
2. `engram-core/src/cluster/mod.rs` – export new modules.
3. `engram-core/src/cluster/membership.rs` – emit membership change notifications for the coordinator.
4. `engram-cli/src/cluster.rs` – construct `SpaceAssignmentManager`/rebalance coordinator and pass into `ClusterState`.
5. `engram-cli/src/api.rs` & `engram-cli/src/grpc.rs` – use cached assignments instead of ad-hoc planning; surface partition/rebalance errors to clients.
6. `engram-cli/config/default.toml` – expose new `[cluster.replication]` knobs (jump hash buckets, rack awareness toggles).
7. Docs (`docs/operations/production-deployment.md`) – describe rebalance workflow.

## Testing Strategy

- **Unit tests**: `engram-core/src/cluster/placement.rs` covering Jump hash determinism, replica diversity, and error handling when nodes < replicas. Tests should hash deterministic spaces and assert only the expected nodes change when membership changes.
- **Assignment manager tests**: Ensure caching, invalidation, and `spaces_assigned_to` behave correctly with concurrent updates (use `DashMap` iterators carefully).
- **Rebalance tests**: Simulate membership changes via a fake `MembershipUpdates` channel and assert that migration plans are produced for the expected spaces.
- **Integration tests**: Extend `engram-core/tests/swim_membership.rs` or add a new `space_assignment_tests.rs` that spins up multiple nodes, assigns spaces, then adds/removes a node and verifies rebalancing occurs with no data loss.
- **CLI tests**: Exercise the admin endpoints via `engram-cli/tests` (e.g., `tests/api_complete_tests.rs`) to confirm JSON responses include the new assignment data.

## Acceptance Criteria

1. Space assignments use Jump Consistent Hash primaries and rack/zone-aware replica selection.
2. Assignments are cached/versioned so writes can be migrated without recomputing per request.
3. Rebalancing is triggered automatically on membership changes and exposes migration progress via admin APIs.
4. CLI status/health endpoints show per-node assignment counts and rebalance status.
5. Metrics (`engram_cluster_assignments_total`, `engram_cluster_rebalance_plans`) feed Grafana dashboards.
6. Unit/integration tests cover the new planner paths and rebalance flows.
7. Documentation explains how to add/remove cluster nodes and monitor rebalancing.

## Completion Summary (2025-11-16)

- `SpaceAssignmentPlanner` now uses FNV64 + jump-consistent hash to deterministically pick primaries, then applies rendezvous scores with rack/zone penalties driven by `[cluster.replication]` (`jump_buckets`, `rack_penalty`, `zone_penalty`).
- `SpaceAssignmentManager` caches placements, tracks monotonically increasing versions, and maintains per-node counts that surface through `/cluster/health` and the new `engram_cluster_spaces_per_node` gauge.
- `SwimMembership` publishes membership updates over `tokio::sync::broadcast`, enabling the `RebalanceCoordinator` to react instantly, plan migrations, increment `engram_cluster_rebalance_plans_total`, and stream `MigrationPlan`s to background workers.
- `ClusterState` on the CLI bundles the assignment manager, rebalance coordinator, partition detector, and router so HTTP handlers, admin APIs (`GET/POST /cluster/rebalance`, `POST /cluster/migrate`), and the new gRPC RPCs (`RebalanceSpaces`, `MigrateSpace`) all share cached routing decisions.
- Prometheus export includes `engram_cluster_assignments_total`, `engram_cluster_rebalance_plans_total`, and per-node assignment gauges; docs/configs (`docs/operations/production-deployment.md`, `docs/reference/configuration.md`, `engram-cli/config/cluster.toml`) walk operators through jump-hash tuning and rebalance workflows.

### Validation
- `cargo test -p engram-cli initialize_cluster_bootstraps_static_seeds -- --exact`
- `cargo test -p engram-cli cluster_rebalance_status_is_exposed -- --exact`
- `cargo test -p engram-core cluster::assignment::tests::manager_memoises_assignments -- --exact`
