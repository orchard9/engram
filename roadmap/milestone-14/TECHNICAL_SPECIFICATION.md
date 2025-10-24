# Milestone 14: Distributed Architecture - Technical Specification

## Executive Summary

This document provides a complete technical specification for distributing Engram across multiple nodes while maintaining API transparency, partition tolerance, and eventual consistency. The design prioritizes availability over consistency (AP system), degrading gracefully during network partitions rather than failing.

**Status**: Final specification ready for implementation
**Duration**: 18-24 days
**Risk Level**: High (distributed systems complexity)
**Validation**: Jepsen-style testing with formal consistency verification

---

## 1. System Architecture

### 1.1 Architectural Principles

**AP System (Availability + Partition Tolerance)**
- Queries continue during network partitions using local data
- Writes fail fast if primary unreachable (no stale writes)
- Eventual consistency with bounded staleness (<60s convergence)

**Memory Space = Partitioning Unit**
- Each space assigned to one primary + N replicas
- Spaces are independent (no cross-space activation spreading)
- Enables per-tenant placement policies and isolation

**Gossip-Based Coordination**
- No external coordination services (ZooKeeper, etcd, Consul)
- SWIM protocol for membership and failure detection
- Anti-entropy gossip for consolidation state synchronization
- Scales to 100+ nodes without coordination bottleneck

### 1.2 Consistency Model

**Episodic Memories (Writes)**
- Primary-per-space with async replication
- Write acknowledgment from primary immediately (don't wait for replicas)
- Replication lag monitored, alerting on >1s lag
- Read-your-writes consistency on primary
- Stale reads possible on replicas (confidence penalty applied)

**Semantic Memories (Consolidation)**
- Local consolidation on each node independently
- Gossip protocol syncs consolidation results
- Conflict resolution via vector clocks + confidence voting
- Convergence guarantee: O(log N) gossip rounds

**Activation Spreading**
- Scatter-gather execution across partitions
- Partial results aggregated with confidence adjustment
- Missing partitions reduce confidence, don't block query
- Timeout prevents slow nodes from degrading performance

### 1.3 Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                         Client Request                          │
│                    (gRPC/HTTP - unchanged API)                   │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Any Node (Load Balanced)                    │
│                                                                   │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                      Router Layer                           │ │
│  │  - Determine target node(s) from SpaceAssignment           │ │
│  │  - Forward to primary (writes) or primary/replicas (reads) │ │
│  │  - Scatter-gather for multi-partition queries              │ │
│  └────────────────────┬───────────────────────┬────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                       │                       │
         ┌─────────────┴──────────┐    ┌──────┴─────────────┐
         │                        │    │                     │
         ▼                        ▼    ▼                     │
┌──────────────┐        ┌──────────────┐        ┌──────────────┐
│   Node A     │◄──────►│   Node B     │◄──────►│   Node C     │
│  (Primary)   │  SWIM  │  (Replica)   │  SWIM  │  (Replica)   │
│              │  +     │              │  +     │              │
│  Space 1,2   │ Gossip │  Space 1,3   │ Gossip │  Space 2,3   │
└──────────────┘        └──────────────┘        └──────────────┘
       │                        │                        │
       └────────────────────────┴────────────────────────┘
              Async Replication (WAL Shipping)
```

### 1.4 Component Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        Engram Node                               │
│                                                                   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │               gRPC/HTTP API (unchanged)                   │   │
│  └─────────────────────────┬────────────────────────────────┘   │
│                            │                                     │
│  ┌─────────────────────────▼────────────────────────────────┐   │
│  │                    Router Layer                           │   │
│  │  - SpaceAssignment (space -> nodes mapping)              │   │
│  │  - ConnectionPool (gRPC clients to remote nodes)         │   │
│  │  - RetryPolicy (exponential backoff, replica fallback)   │   │
│  └─────────────────────────┬────────────────────────────────┘   │
│                            │                                     │
│  ┌─────────────────────────▼────────────────────────────────┐   │
│  │              Local Memory Engine (existing)               │   │
│  │  - MemoryGraph, Consolidation, Activation Spreading      │   │
│  └─────────────────────────┬────────────────────────────────┘   │
│                            │                                     │
│  ┌─────────────────────────▼────────────────────────────────┐   │
│  │                Cluster Subsystems                         │   │
│  │  ┌──────────────────────────────────────────────────────┐│   │
│  │  │ SwimMembership: node discovery, failure detection    ││   │
│  │  └──────────────────────────────────────────────────────┘│   │
│  │  ┌──────────────────────────────────────────────────────┐│   │
│  │  │ PartitionDetector: network partition detection       ││   │
│  │  └──────────────────────────────────────────────────────┘│   │
│  │  ┌──────────────────────────────────────────────────────┐│   │
│  │  │ ReplicationManager: WAL shipping to replicas         ││   │
│  │  └──────────────────────────────────────────────────────┘│   │
│  │  ┌──────────────────────────────────────────────────────┐│   │
│  │  │ GossipProtocol: consolidation state synchronization  ││   │
│  │  └──────────────────────────────────────────────────────┘│   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. Detailed Design

### 2.1 SWIM Membership Protocol

**Failure Detection Algorithm**:
```
Every probe_interval (default 1s):
1. Select random alive node N
2. Send direct ping to N
3. If ack received within timeout (500ms):
     Mark N as alive
4. Else:
     Select K random nodes (default K=3)
     Send ping-req to each: "please ping N"
     If any indirect ack received:
        Mark N as alive
     Else:
        Mark N as suspect
        Start suspect timer (default 5s)
5. If suspect timer expires:
     Mark N as dead
```

**Gossip Dissemination**:
- Piggyback membership updates on ping/ack messages
- Infection-style propagation: O(log N) convergence
- Incarnation counter prevents false positives (refutation)

**Message Format** (UDP, max 1400 bytes):
```rust
enum SwimMessage {
    Ping { from: NodeId, incarnation: u64 },
    Ack { from: NodeId, incarnation: u64 },
    PingReq { from: NodeId, target: NodeId, incarnation: u64 },
    Gossip { updates: Vec<MembershipUpdate> },
}

struct MembershipUpdate {
    node_id: String,
    addr: SocketAddr,
    state: NodeState, // Alive, Suspect, Dead, Left
    incarnation: u64,
    spaces: Vec<String>, // Which spaces this node hosts
}
```

### 2.2 Space Assignment Strategy

**Consistent Hashing for Even Distribution**:
```rust
struct SpaceAssignment {
    ring: ConsistentHashRing<NodeId>,
    replication_factor: usize,
}

impl SpaceAssignment {
    fn assign_space(&self, space_id: &str) -> SpaceNodes {
        // Hash space ID to point on ring
        let hash = hash_space_id(space_id);

        // Find N successor nodes on ring
        let nodes = self.ring.successors(hash, self.replication_factor);

        SpaceNodes {
            primary: nodes[0].clone(),
            replicas: nodes[1..].to_vec(),
        }
    }
}
```

**Rebalancing on Node Join/Leave**:
- Affected spaces: those whose N successors changed
- Rebalance strategy: copy data to new node, wait for sync, update assignment
- No downtime: old assignment valid until new node ready

**Placement Strategies**:
1. **Random**: Simple consistent hashing (default)
2. **RackAware**: Avoid replicas on same rack (requires rack metadata)
3. **ZoneAware**: Avoid replicas in same availability zone

### 2.3 Replication Protocol

**Write-Ahead Log (WAL) Shipping**:
```rust
// On primary node
async fn handle_write(space_id: &str, memory: Memory) -> Result<()> {
    // 1. Append to local WAL
    let wal_entry = self.wal.append(space_id, &memory).await?;

    // 2. Return success immediately (don't wait for replicas)
    let write_id = wal_entry.id;

    // 3. Async: ship to replicas
    tokio::spawn({
        let replicas = self.assignment.get_replicas(space_id);
        async move {
            for replica in replicas {
                self.ship_wal_entry(replica, wal_entry.clone()).await;
            }
        }
    });

    Ok(())
}

// On replica node
async fn handle_wal_entry(entry: WalEntry) -> Result<()> {
    // 1. Validate entry (checksum, sequence number)
    entry.validate()?;

    // 2. Apply to local memory graph
    self.memory_graph.apply(entry.memory).await?;

    // 3. Send ack to primary
    self.send_ack(entry.primary_node, entry.id).await?;

    Ok(())
}
```

**Replication Lag Monitoring**:
```rust
struct LagMonitor {
    per_replica_lag: DashMap<(SpaceId, NodeId), Duration>,
}

impl LagMonitor {
    async fn record_ack(&self, space: &str, replica: &str, entry_id: u64) {
        let lag = self.compute_lag(entry_id);
        self.per_replica_lag.insert((space.to_string(), replica.to_string()), lag);

        if lag > Duration::from_secs(1) {
            warn!("Replication lag for space {} on replica {}: {:?}", space, replica, lag);
        }
    }
}
```

### 2.4 Partition Detection and Handling

**Partition Detection**:
```rust
async fn check_partition_status() {
    let alive_nodes = count_alive_nodes();
    let total_nodes = membership.members.len();
    let reachability_ratio = alive_nodes as f64 / total_nodes as f64;

    if reachability_ratio < 0.5 {
        // Partitioned from majority
        enter_partition_mode().await;
    }
}
```

**Partition Mode Behavior**:
- **Reads**: Serve from local data only, apply confidence penalty
- **Writes**: Only accept writes for spaces where we're primary
- **Consolidation**: Continue local consolidation, sync after healing
- **Queries**: Return partial results with clear confidence reduction

**Confidence Penalty Calculation**:
```rust
fn adjust_confidence(base: f32, partition_state: &PartitionState) -> f32 {
    match partition_state {
        Connected => base,
        Partitioned { reachable, total } => {
            let reachability = *reachable as f32 / *total as f32;
            let penalty = 1.0 - reachability;
            base * (1.0 - penalty * 0.5) // Max 50% penalty
        }
    }
}
```

### 2.5 Gossip Protocol for Consolidation

**Merkle Tree for Efficient Sync**:
```rust
struct ConsolidationState {
    // Merkle tree of semantic memories
    merkle: MerkleTree<SemanticMemory>,
}

async fn gossip_consolidation(peer: NodeId) {
    // 1. Exchange merkle roots
    let local_root = self.merkle.root();
    let remote_root = peer.get_merkle_root().await?;

    // 2. If roots match, we're in sync
    if local_root == remote_root {
        return Ok(());
    }

    // 3. Identify divergent subtrees
    let diff = self.merkle.diff(&remote_root).await?;

    // 4. Fetch only different data
    let remote_data = peer.get_consolidation_subset(diff.keys()).await?;

    // 5. Merge using conflict resolution
    for (key, remote_memory) in remote_data {
        self.merge_semantic_memory(key, remote_memory).await?;
    }
}
```

**Conflict Resolution**:
```rust
fn resolve_conflict(local: &SemanticMemory, remote: &SemanticMemory) -> SemanticMemory {
    // 1. Check vector clocks for causality
    match local.vector_clock.compare(&remote.vector_clock) {
        Less => return remote.clone(),    // Remote dominates
        Greater => return local.clone(),  // Local dominates
        Equal => return local.clone(),    // Identical
        Concurrent => {
            // 2. Concurrent updates (split-brain scenario)
            // Use confidence-based voting
            if remote.confidence > local.confidence {
                remote.clone()
            } else if local.confidence > remote.confidence {
                local.clone()
            } else {
                // 3. Same confidence, merge patterns
                merge_patterns(local, remote)
            }
        }
    }
}
```

### 2.6 Distributed Query Execution

**Scatter-Gather Algorithm**:
```rust
async fn execute_distributed_query(query: Query) -> Result<QueryResult> {
    // 1. Determine which spaces are relevant
    let relevant_spaces = query.get_relevant_spaces()?;

    // 2. Find nodes hosting those spaces
    let target_nodes: HashSet<NodeId> = relevant_spaces
        .iter()
        .flat_map(|space| assignment.get_nodes(space))
        .collect();

    // 3. Scatter query to all target nodes (parallel)
    let mut handles = vec![];
    for node in target_nodes {
        let handle = tokio::spawn({
            let query = query.clone();
            async move {
                client.query(node, query).await
            }
        });
        handles.push(handle);
    }

    // 4. Gather results with timeout
    let timeout = Duration::from_secs(5);
    let results = futures::future::join_all(handles)
        .timeout(timeout)
        .await?;

    // 5. Aggregate partial results
    let aggregated = aggregate_results(results)?;

    // 6. Adjust confidence for missing nodes
    let missing_ratio = compute_missing_ratio(&target_nodes, &results);
    aggregated.confidence *= (1.0 - missing_ratio * 0.5);

    Ok(aggregated)
}
```

**Result Aggregation**:
```rust
fn aggregate_results(partial: Vec<QueryResult>) -> QueryResult {
    let mut merged = QueryResult::empty();

    for result in partial {
        // Merge memories (dedup by ID)
        merged.memories.extend(result.memories);

        // Combine confidence intervals
        merged.confidence = combine_confidence_intervals(
            merged.confidence,
            result.confidence
        );
    }

    // Sort by activation level (descending)
    merged.memories.sort_by(|a, b| b.activation.partial_cmp(&a.activation).unwrap());

    merged
}
```

---

## 3. API Transparency

### 3.1 Unchanged Client APIs

Clients use identical API for single-node and distributed deployments:

```protobuf
// engram/v1/service.proto (unchanged)

service EngramService {
    rpc Store(StoreRequest) returns (StoreResponse);
    rpc Recall(RecallRequest) returns (RecallResponse);
    rpc Spread(SpreadRequest) returns (SpreadResponse);
    rpc Consolidate(ConsolidateRequest) returns (ConsolidateResponse);
}
```

### 3.2 Internal Node-to-Node RPC

New internal RPC for node-to-node communication:

```protobuf
// engram/v1/cluster.proto (new)

service ClusterService {
    // Replication
    rpc ShipWalEntry(WalEntryRequest) returns (WalEntryResponse);

    // Gossip
    rpc ExchangeMerkleRoot(MerkleRootRequest) returns (MerkleRootResponse);
    rpc GetConsolidationSubset(SubsetRequest) returns (SubsetResponse);

    // Query forwarding
    rpc ForwardQuery(QueryRequest) returns (QueryResponse);

    // Health
    rpc Ping(PingRequest) returns (PingResponse);
}
```

### 3.3 Routing Logic

```rust
impl Router {
    async fn route_request(&self, req: Request) -> Result<Response> {
        match req.operation {
            Operation::Store(space, memory) => {
                let primary = self.assignment.get_primary(&space);
                if primary == self.node_id {
                    // We're the primary, handle locally
                    self.local_store(space, memory).await
                } else {
                    // Forward to primary
                    self.forward_to(primary, req).await
                }
            },

            Operation::Recall(space, cue) => {
                let primary = self.assignment.get_primary(&space);
                match self.query_with_fallback(primary, &space, req).await {
                    Ok(resp) => Ok(resp),
                    Err(_) => {
                        // Primary unavailable, try replica
                        let replicas = self.assignment.get_replicas(&space);
                        self.query_any_replica(replicas, req).await
                    }
                }
            },

            Operation::Consolidate(space) => {
                // Local operation, always handled locally
                self.local_consolidate(space).await
            },
        }
    }
}
```

---

## 4. Configuration

### 4.1 Single-Node Mode (Default)

```toml
# engram.toml
[cluster]
enabled = false  # Single-node by default
```

### 4.2 Distributed Mode

```toml
[cluster]
enabled = true
node_id = "node-1"  # Or auto-generated

[cluster.discovery]
type = "static"
seed_nodes = ["node-1:7946", "node-2:7946", "node-3:7946"]

[cluster.swim]
bind_addr = "0.0.0.0:7946"
probe_interval_ms = 1000
probe_timeout_ms = 500
indirect_probes = 3
suspect_timeout_ms = 5000

[cluster.replication]
factor = 2              # Primary + 1 replica
timeout_ms = 1000       # Async write timeout
placement = "random"    # Or "rack_aware", "zone_aware"

[cluster.network]
swim_bind = "0.0.0.0:7946"
api_bind = "0.0.0.0:50051"
max_message_size = 4194304  # 4MB
connection_pool_size = 4
```

### 4.3 Kubernetes Configuration

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: engram-config
data:
  engram.toml: |
    [cluster]
    enabled = true

    [cluster.discovery]
    type = "dns"
    service = "engram-cluster.default.svc.cluster.local"
    port = 7946
    refresh_interval_sec = 30
```

---

## 5. Observability

### 5.1 Metrics

**Cluster Membership**:
- `engram.cluster.size` (gauge): Total nodes in cluster
- `engram.cluster.alive_nodes` (gauge): Nodes in Alive state
- `engram.cluster.suspect_nodes` (gauge): Nodes in Suspect state
- `engram.cluster.dead_nodes` (gauge): Nodes in Dead state

**Replication**:
- `engram.replication.lag_ms` (histogram): Per-replica replication lag
- `engram.replication.wal_entries_shipped` (counter): WAL entries sent
- `engram.replication.wal_entries_applied` (counter): WAL entries applied

**Partition**:
- `engram.partition.detected` (counter): Partition events
- `engram.partition.healed` (counter): Healing events
- `engram.partition.duration_sec` (histogram): Time spent partitioned

**Distributed Query**:
- `engram.query.scatter_fanout` (histogram): Nodes queried per request
- `engram.query.gather_latency_ms` (histogram): Aggregation time
- `engram.query.partial_results` (counter): Queries with missing nodes

### 5.2 Logging

**Structured Logs**:
```rust
info!(
    node_id = %node_id,
    space_id = %space_id,
    primary = %primary_node,
    replicas = ?replica_nodes,
    "Space assignment updated"
);

warn!(
    node_id = %failed_node,
    state = "suspect",
    last_seen_sec = %last_seen,
    "Node suspected of failure"
);

error!(
    space_id = %space_id,
    local_clock = ?local_clock,
    remote_clock = ?remote_clock,
    "Split-brain detected"
);
```

### 5.3 Tracing

**Distributed Tracing**:
- Trace ID propagates across node boundaries
- Span per hop (client → node A → node B)
- Causality tracking via vector clocks

```rust
#[tracing::instrument(skip(self))]
async fn forward_to(&self, target: NodeId, req: Request) -> Result<Response> {
    let span = tracing::info_span!(
        "forward_request",
        target = %target,
        operation = ?req.operation
    );

    async move {
        self.client.send(target, req).await
    }
    .instrument(span)
    .await
}
```

---

## 6. Failure Modes and Recovery

### 6.1 Node Crash

**Detection**: SWIM failure detection (suspect timeout 5s)

**Recovery**:
1. Replica promoted to primary for affected spaces
2. New replica selected from remaining nodes
3. Data copied to new replica
4. Assignment updated and gossiped

**Data Loss**: None (replicas have all data)

### 6.2 Network Partition

**Detection**: Majority unreachable (partition detector)

**Behavior**:
- Enter partition mode (local-only operations)
- Continue serving local reads with confidence penalty
- Reject writes to non-local spaces
- Local consolidation continues

**Recovery**:
- Partition healing detected
- Anti-entropy sync triggered
- Normal operations resume
- Metrics track partition duration

**Data Loss**: None (eventual consistency guarantees convergence)

### 6.3 Split-Brain

**Detection**: Vector clock comparison shows concurrent primaries

**Prevention**:
- Quorum-based writes (require majority acknowledgment)
- Partition mode prevents concurrent primaries

**Recovery**:
- Refuse operations until operator intervention
- Manual resolution: choose authoritative primary
- Discard conflicting writes from other primary

**Data Loss**: Possible (conflicting writes discarded)

### 6.4 Slow Replica

**Detection**: Replication lag monitoring

**Behavior**:
- Alert at >1s lag (warning)
- Remove from query rotation at >10s lag (critical)
- Automatic catchup from primary

**Recovery**:
- Replica catches up via WAL replay
- Re-added to query rotation
- Metrics track catchup progress

**Data Loss**: None (eventual consistency)

---

## 7. Testing Strategy

### 7.1 Unit Tests (per task)

**Coverage**:
- SWIM protocol state machine
- Consistent hashing distribution
- Vector clock causality
- Conflict resolution determinism

**Tools**: `cargo test`, `proptest` for property-based tests

### 7.2 Integration Tests

**Scenarios**:
- 3-node cluster formation
- Node join/leave rebalancing
- Partition and healing workflow
- Primary failure and replica promotion

**Tools**: `tokio::test`, `TestCluster` harness

### 7.3 Chaos Tests

**Fault Injection**:
- Random node crashes
- Network partitions (symmetric, asymmetric, flapping)
- Packet loss and latency injection
- Cascading failures

**Tools**: `NetworkSimulator`, `chaos` module

### 7.4 Jepsen Tests

**Invariants**:
- Linearizability of writes (quorum-based)
- Eventual consistency of consolidation
- No data loss under failures
- Confidence bounds remain valid

**Tools**: Clojure Jepsen framework, custom checker

### 7.5 Performance Benchmarks

**Baselines**:
- Single-node throughput: 10K ops/sec
- Single-node latency: <10ms P99

**Targets**:
- Intra-partition: <2x latency
- Cross-partition: <5x latency
- Scaling: linear throughput to 16 nodes

**Tools**: `criterion`, custom workload generator

---

## 8. Operational Procedures

### 8.1 Deployment

**Single-Node to Distributed Migration**:
1. Deploy second node with `cluster.enabled = true`
2. Point to first node as seed
3. Wait for cluster formation (5s)
4. Deploy third node
5. Verify even space distribution
6. Update load balancer to all nodes

**Rolling Upgrade**:
1. Deploy new version to one node
2. Wait for SWIM convergence (10s)
3. Verify health metrics stable
4. Repeat for next node
5. No downtime (queries route to healthy nodes)

### 8.2 Scaling

**Adding Node**:
```bash
# 1. Deploy new node
engram --config engram.toml

# 2. Verify it joined cluster
curl http://engram-1:8080/cluster/nodes

# 3. Wait for automatic rebalancing (10min)
# 4. Verify even distribution
curl http://engram-1:8080/cluster/spaces
```

**Removing Node**:
```bash
# 1. Gracefully leave cluster
engram-ctl leave-cluster node-5

# 2. Wait for rebalancing (10min)
# 3. Verify no data loss
engram-ctl verify-consistency

# 4. Shutdown node
systemctl stop engram
```

### 8.3 Troubleshooting

**Symptom**: High replication lag

**Diagnosis**:
```bash
# Check lag metrics
curl http://engram:8080/metrics | grep replication_lag

# Check network between primary and replica
engram-ctl network-test node-1 node-2
```

**Resolution**:
- Network congestion: increase `replication.timeout_ms`
- Slow replica: remove from rotation, investigate resource constraints
- Primary overloaded: reduce write throughput or add more primaries

---

**Symptom**: Partition detected but network healthy

**Diagnosis**:
```bash
# Check SWIM suspicions
engram-ctl cluster-health --verbose

# Check ping latencies
engram-ctl swim-ping node-2
```

**Resolution**:
- Increase `swim.probe_timeout_ms` for high-latency networks
- Increase `swim.suspect_timeout_ms` to reduce false positives

---

**Symptom**: Split-brain detected

**Diagnosis**:
```bash
# Identify concurrent primaries
engram-ctl detect-split-brain space-123

# Check vector clocks
engram-ctl show-vector-clock space-123
```

**Resolution**:
1. Stop writes to affected space
2. Choose authoritative primary (highest incarnation)
3. Discard writes from other primary
4. Reset vector clock
5. Resume operations

---

## 9. Performance Targets

| Metric | Target | Rationale |
|--------|--------|-----------|
| Cluster Formation | <5s | SWIM gossip convergence |
| Failure Detection | <7s | Probe (1s) + suspect timeout (5s) |
| Partition Detection | <10s | 2 probe cycles + suspect timeout |
| Replication Lag | <1s | Async WAL shipping |
| Query Latency (Intra-partition) | <2x single-node | Network + routing overhead |
| Query Latency (Cross-partition) | <5x single-node | Scatter-gather overhead |
| Gossip Convergence | <60s | O(log N) rounds, 1s interval |
| Scaling Efficiency | Linear to 16 nodes | No coordination bottleneck |

---

## 10. Risks and Mitigations

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Gossip doesn't converge | Low | High | Formal proof, property testing |
| Replication lag degrades writes | Medium | Medium | Lag monitoring, auto-tuning |
| Partition tolerance insufficient | Low | High | Jepsen testing, chaos engineering |
| Split-brain causes corruption | Low | Critical | Vector clocks, operator alerts |
| Performance worse than single-node | Medium | High | Keep single-node fast path |
| Operational complexity too high | High | Medium | Comprehensive runbooks, auto-tuning |

---

## 11. Success Criteria

**Functional**:
- [ ] 100% of single-node tests pass against distributed cluster
- [ ] Partition tolerance: survive 50% node loss with graceful degradation
- [ ] Eventual consistency: convergence within 60s on 100-node cluster
- [ ] No data loss: Jepsen tests show 0% data loss under failures
- [ ] API transparency: zero client code changes required

**Performance**:
- [ ] Intra-partition queries <2x single-node latency
- [ ] Linear scaling to 16 nodes
- [ ] Replication lag <1s under normal load
- [ ] Cluster formation <5s
- [ ] Failure detection <7s

**Operational**:
- [ ] External operator deploys cluster from docs successfully
- [ ] All failure scenarios documented with recovery steps
- [ ] Metrics and alerts cover all key health indicators
- [ ] Rolling upgrade with zero downtime
- [ ] Troubleshooting guide resolves common issues

---

## 12. Future Work (Out of Scope)

**Multi-Region Deployment** (M17):
- Cross-region replication with conflict-free replicated data types (CRDTs)
- Geo-aware routing (prefer local region)
- WAN optimization (compression, batching)

**Strong Consistency Modes** (M18):
- Raft consensus for linearizable writes
- Tunable consistency (eventual, session, linearizable)
- Hybrid mode (strong for critical writes, eventual for bulk)

**Automatic Rebalancing** (M19):
- Load-aware placement (balance CPU/memory across nodes)
- Automatic partition migration (move spaces to underutilized nodes)
- Proactive scaling (add nodes based on metrics)

---

## Appendix A: Glossary

**AP System**: Availability + Partition Tolerance (CAP theorem)
**SWIM**: Scalable Weakly-consistent Infection-style Process Group Membership
**Gossip Protocol**: Epidemic-style information dissemination
**Vector Clock**: Causality tracking for concurrent events
**Merkle Tree**: Hash tree for efficient state comparison
**Anti-Entropy**: Periodic synchronization to repair divergence
**Split-Brain**: Multiple nodes believe they're primary for same partition
**Replication Lag**: Time delay between primary write and replica apply
**Scatter-Gather**: Query pattern distributing work across nodes

---

## Appendix B: References

**Distributed Systems**:
- "Designing Data-Intensive Applications" by Martin Kleppmann
- "Distributed Systems" by Maarten van Steen and Andrew S. Tanenbaum

**SWIM Protocol**:
- "SWIM: Scalable Weakly-consistent Infection-style Process Group Membership Protocol" (Das et al., 2002)

**Gossip Protocols**:
- "Epidemic Algorithms for Replicated Database Maintenance" (Demers et al., 1987)

**Consistency Testing**:
- "Jepsen: On the Perils of Network Partitions" (Kyle Kingsbury)
- "Testing Distributed Systems" (Peter Alvaro)

**Vector Clocks**:
- "Time, Clocks, and the Ordering of Events in a Distributed System" (Lamport, 1978)

---

**Document Status**: Final
**Last Updated**: 2025-10-23
**Next Review**: After Task 003 completion
