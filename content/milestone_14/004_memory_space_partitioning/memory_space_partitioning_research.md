# Research: Memory Space Partitioning in Distributed Cognitive Systems

## The Partitioning Problem

You have a cognitive graph database with millions of memory nodes distributed across 10 machines. When a query arrives asking for memories related to "coffee," which machines should process it? If every machine searches its local storage, you waste resources. If you route to the wrong machine, you miss relevant memories.

This is the partitioning problem: how do you split data across machines such that related information tends to land on the same machine, queries can find what they need efficiently, and load stays balanced?

Traditional databases partition by key ranges or hash functions. User IDs 1-1000 go to Node A, 1001-2000 go to Node B. This works when queries target specific keys, but breaks down for graph traversals and associative queries.

Engram needs something different. Memories don't have primary keys. Queries aren't "get memory 12345" - they're "find memories activated by this cue, spreading through semantic associations." The partitioning unit must reflect cognitive structure, not arbitrary ID ranges.

## Memory Spaces as Partitioning Units

Engram's solution: partition by memory space. Each memory space (introduced in Milestone 7) represents an isolated cognitive context - a user, an application, a conversation thread. Memory spaces are independent: activation doesn't spread across spaces, consolidation happens within spaces.

This makes them natural partitioning boundaries:

```rust
struct SpaceAssignment {
    space_id: SpaceId,
    primary_node: NodeId,
    replicas: Vec<NodeId>,
    version: u64,  // For atomic updates
}
```

Benefits of space-based partitioning:

1. **Query locality**: Spreading activation stays within one space, so queries target a single partition (or its replicas)
2. **Independent scaling**: Hot spaces (high traffic) can have more replicas
3. **Tenant isolation**: Different users' spaces live on different nodes
4. **Natural boundaries**: No cross-space transactions or coordination needed

## Consistent Hashing for Even Distribution

With potentially millions of memory spaces and dozens of nodes, how do you assign spaces to nodes fairly?

Consistent hashing (Karger et al., 1997) is the standard solution. The algorithm:

1. Hash each node ID to points on a ring (0 to 2^64)
2. Hash each space ID to a point on the same ring
3. A space is owned by the next node clockwise on the ring

```rust
fn assign_space(space_id: SpaceId, nodes: &[NodeId]) -> NodeId {
    let space_hash = hash(space_id);

    nodes.iter()
        .map(|node| (hash(node), node))
        .filter(|(node_hash, _)| *node_hash >= space_hash)
        .min_by_key(|(node_hash, _)| *node_hash)
        .map(|(_, node)| *node)
        .unwrap_or_else(|| nodes[0])  // Wrap around
}
```

The beauty: when nodes join or leave, only K/N spaces need to move (K = total spaces, N = total nodes). Most assignments stay stable.

## Virtual Nodes for Better Balance

Pure consistent hashing can create imbalanced distributions. If you have 3 nodes and 1000 spaces, one node might get 400 spaces while another gets 250.

The solution: virtual nodes. Each physical node owns multiple points on the hash ring:

```rust
const VIRTUAL_NODES_PER_PHYSICAL: usize = 150;

fn create_ring(nodes: &[NodeId]) -> Vec<(u64, NodeId)> {
    let mut ring = Vec::new();

    for node in nodes {
        for vnode in 0..VIRTUAL_NODES_PER_PHYSICAL {
            let hash = hash(&(node, vnode));
            ring.push((hash, *node));
        }
    }

    ring.sort_by_key(|(hash, _)| *hash);
    ring
}
```

Now each physical node has 150 positions on the ring. Space assignments distribute evenly across these virtual nodes, which map back to physical nodes. This achieves near-perfect load balancing: with 150 virtual nodes per physical node, imbalance stays under 5%.

Cassandra uses this technique with great success at massive scale.

## Replication for Availability

Each space has a primary node (handles writes) plus N replicas (handle reads, failover for writes). Replicas should be on different physical machines for fault tolerance.

The replication strategy uses the consistent hash ring:

```rust
fn assign_replicas(
    space_id: SpaceId,
    ring: &[(u64, NodeId)],
    replication_factor: usize,
) -> Vec<NodeId> {
    let space_hash = hash(space_id);

    let mut replicas = Vec::new();
    let mut seen_physical_nodes = HashSet::new();

    for (_, node) in ring.iter().cycle().skip_while(|(h, _)| *h < space_hash) {
        if seen_physical_nodes.insert(*node) {
            replicas.push(*node);
            if replicas.len() >= replication_factor {
                break;
            }
        }
    }

    replicas
}
```

This walks clockwise from the space's hash position, collecting unique physical nodes. With virtual nodes, consecutive ring positions likely belong to different physical nodes, giving good replica distribution.

For 3x replication, each space has 1 primary + 2 replicas across 3 different machines.

## Rack-Aware and Zone-Aware Placement

In production, machines fail together. A network switch dies, taking out an entire rack. An AWS availability zone becomes unreachable. You want replicas spread across failure domains.

Engram supports topology-aware placement:

```rust
struct Node {
    id: NodeId,
    rack: Option<String>,
    zone: Option<String>,
}

fn assign_rack_aware_replicas(
    space_id: SpaceId,
    ring: &[(u64, NodeId)],
    nodes: &HashMap<NodeId, Node>,
    replication_factor: usize,
) -> Vec<NodeId> {
    let mut replicas = Vec::new();
    let mut used_racks = HashSet::new();

    for (_, node_id) in ring.iter().cycle().skip_while(|(h, _)| *h < hash(space_id)) {
        if let Some(node) = nodes.get(node_id) {
            let rack = node.rack.as_deref().unwrap_or("default");

            if used_racks.insert(rack) {
                replicas.push(*node_id);
                if replicas.len() >= replication_factor {
                    break;
                }
            }
        }
    }

    replicas
}
```

This prefers replicas in different racks. If racks aren't configured, it falls back to physical node diversity.

For multi-region deployments (future milestone), the same logic extends to zones: prefer replicas in different availability zones.

## Rebalancing When Nodes Join or Leave

When a new node joins, some spaces should move to it for better balance. When a node leaves, its spaces must move elsewhere. This is rebalancing.

The naive approach: immediately move all affected spaces. This causes massive data transfer and query disruptions.

The better approach: gradual rebalancing with query-aware scheduling.

```rust
struct RebalancingPlan {
    moves: Vec<SpaceMove>,
    estimated_bytes: u64,
    estimated_duration: Duration,
}

struct SpaceMove {
    space_id: SpaceId,
    from_node: NodeId,
    to_node: NodeId,
    priority: Priority,
}

enum Priority {
    Immediate,  // Node failed, must move
    High,       // Severe imbalance
    Medium,     // Moderate imbalance
    Low,        // Gentle rebalancing
}
```

Immediate priority (node failures) transfers data as fast as possible. Low priority (optimization rebalancing) rate-limits transfers to avoid impacting live queries.

The scheduling algorithm:
1. Identify spaces to move based on new hash ring
2. Prioritize moves (failure recovery > balancing)
3. Execute high-priority moves immediately
4. Execute low-priority moves during off-peak hours or rate-limited

This minimizes disruption while ensuring critical moves (failover) happen fast.

## Partition Metadata Propagation

Every node needs to know the current space assignments to route queries correctly. This metadata must be consistent across the cluster but updates infrequently (only when nodes join/leave or rebalancing occurs).

Options:

**Option 1: Centralized metadata store**
Use a strongly consistent KV store (etcd, Consul) to hold assignments. Nodes query this store for routing decisions. Simple but adds a dependency and single point of failure.

**Option 2: Gossip-based propagation**
Treat space assignments as gossip payloads. When assignments change, gossip the update. Eventually consistent, no external dependencies.

Engram chooses Option 2 for consistency with the overall AP architecture. Assignment updates are small (space_id + 3 node IDs = ~40 bytes), so they piggyback on existing SWIM gossip messages efficiently.

```rust
struct AssignmentUpdate {
    space_id: SpaceId,
    primary: NodeId,
    replicas: Vec<NodeId>,
    version: u64,
}
```

The version number provides last-write-wins semantics. If two nodes have different assignments for the same space, higher version wins. This ensures eventual consistency even during network partitions.

## Query Routing with Stale Metadata

Gossip means nodes might temporarily have stale assignment metadata. Node A thinks Space 5 is on Node B, but it actually moved to Node C. What happens when a query arrives?

The solution: forwarding with version checking.

```rust
async fn route_query(
    &self,
    space_id: SpaceId,
    query: Query,
) -> Result<QueryResult> {
    let assignment = self.assignments.get(&space_id)?;

    // Try primary node
    match self.send_query(assignment.primary, space_id, query.clone()).await {
        Ok(result) => Ok(result),
        Err(QueryError::NotPrimary { actual_primary, version }) => {
            // Our metadata is stale, update and retry
            if version > assignment.version {
                self.update_assignment(space_id, actual_primary, version);
            }
            self.send_query(actual_primary, space_id, query).await
        }
        Err(e) => {
            // Try replicas
            for replica in &assignment.replicas {
                if let Ok(result) = self.send_query(*replica, space_id, query.clone()).await {
                    return Ok(result);
                }
            }
            Err(e)
        }
    }
}
```

If the primary responds with "I'm not primary anymore, the actual primary is Node C (version 5)," the querying node updates its local metadata and retries. This self-correcting mechanism ensures routing works even with stale metadata.

## Monitoring and Observability

Critical metrics for partitioning:

```
space_assignments_total: Gauge (total spaces)
space_assignments_per_node: Gauge (per-node space count)
space_assignment_imbalance: Gauge (max - min / avg)
rebalancing_moves_pending: Gauge (spaces waiting to move)
rebalancing_bytes_transferred: Counter (total data moved)
query_routing_errors: Counter (routes to wrong node)
```

Alert conditions:
- Imbalance > 20% for more than 1 hour
- Rebalancing stalled (pending > 100 for 30 minutes)
- Query routing errors > 1% of traffic

Dashboard showing per-node space distribution helps operators identify hotspots and validate rebalancing effectiveness.

## Academic Foundation

Space partitioning draws from:
- **Consistent Hashing**: Karger et al. (1997) - fundamental algorithm
- **Dynamo**: DeCandia et al. (2007) - virtual nodes and replication strategies
- **Cassandra**: Lakshman & Malik (2010) - rack-aware placement
- **Spanner**: Corbett et al. (2012) - though we don't use their strongly consistent approach

## Cognitive Architecture Parallels

The brain partitions information spatially. Visual memories cluster in visual cortex, linguistic memories in language areas, motor memories in motor cortex. This spatial organization isn't arbitrary - it enables specialized processing and efficient retrieval.

Engram's space-based partitioning mirrors this. Memory spaces (cognitive contexts) map to physical locations (nodes). Related memories colocate, enabling efficient spreading activation within a space. Cross-space queries (rare, like inter-regional brain communication) are more expensive.

This biological realism extends to rebalancing. When brain regions are damaged, remaining regions can take over some functions through neuroplasticity. Similarly, when Engram nodes fail, spaces rebalance to healthy nodes, maintaining system function.

## Conclusion

Memory space partitioning provides the foundation for Engram's distributed data placement. By using spaces as partitioning units, consistent hashing for distribution, and gossip for metadata propagation, Engram achieves:

- Even load distribution (< 5% imbalance with virtual nodes)
- Query locality (spreading activation stays on one node/replicas)
- Graceful rebalancing (incremental moves with priority scheduling)
- Fault tolerance (rack-aware replica placement)

This infrastructure enables Engram to scale horizontally while maintaining the cognitive property that related memories cluster together - just like they do in biological brains.
