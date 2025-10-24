# Milestone 14: Distributed Architecture - Executive Summary

**Status**: Planning Complete - Ready for Implementation
**Duration**: 18-24 days (12 tasks)
**Risk**: High (distributed systems complexity)
**Dependencies**: M6 (Consolidation), M7 (Memory Spaces), M15 (Multi-interface)

---

## Objective

Extend Engram to support optional distributed deployment across multiple nodes while maintaining complete API transparency, partition tolerance, and eventual consistency. Single-node deployment remains first-class and default.

---

## Key Design Decisions

### 1. AP System (Availability + Partition Tolerance)

**Why**: Cognitive memory retrieval must remain available even with stale data. A human brain doesn't stop recalling during "network partitions."

**Tradeoff**: Accept eventual consistency (convergence <60s) instead of strong consistency. No linearizability guarantees.

### 2. SWIM Protocol for Coordination

**Why**: Scales to 100+ nodes without external coordination services (no ZooKeeper, etcd, Consul dependency).

**Tradeoff**: Probabilistic failure detection (configurable false positive rate) instead of deterministic consensus.

### 3. Memory Space = Partitioning Unit

**Why**: Spaces are already isolated (M7), natural boundary for sharding. Per-tenant placement policies.

**Tradeoff**: No cross-space activation spreading in distributed mode.

### 4. Gossip-Based Consolidation Sync

**Why**: Matches biological plausibility (distributed memory consolidation). Fault-tolerant by design.

**Tradeoff**: Eventual consistency for semantic memories instead of immediate synchronization.

---

## Architecture at a Glance

```
Client (unchanged API)
    │
    ▼
Any Node (Load Balanced) ──────► Router Layer
    │                              │
    │                              ├─► Primary Node (writes)
    │                              ├─► Primary/Replicas (reads)
    │                              └─► Scatter-Gather (multi-partition queries)
    │
    ▼
┌────────────────────────────────────────────────┐
│  Node A          Node B          Node C        │
│  (Primary)       (Replica)       (Replica)     │
│                                                 │
│  Space 1,2       Space 1,3       Space 2,3     │
└────────────────────────────────────────────────┘
         │              │              │
         └──────────────┴──────────────┘
           SWIM + Gossip Protocol
```

---

## Task Breakdown

### Phase 1: Foundation (Week 1)

**001: SWIM Membership** (3-4 days)
- Cluster membership and failure detection
- UDP-based gossip protocol
- Node discovery and health monitoring

**002: Discovery & Configuration** (2 days)
- Static seed lists, DNS SRV, Consul integration
- Single-node vs cluster mode configuration
- Health monitoring integration

**003: Partition Handling** (3 days)
- Network partition detection
- Local-only recall during partitions
- Confidence penalty calculation
- Vector clocks for split-brain detection

### Phase 2: Partitioning (Week 2)

**004: Space Assignment** (3 days)
- Consistent hashing for even distribution
- Primary node election per space
- Replica placement strategies
- Rebalancing on node join/leave

**005: Replication Protocol** (4 days)
- Async WAL shipping to replicas
- Replication lag monitoring
- Catchup mechanism for slow replicas
- Replica promotion on primary failure

**006: Routing Layer** (3 days)
- Route operations to correct nodes
- Connection pooling to remote nodes
- Retry logic with exponential backoff
- Fallback to replicas on primary failure

### Phase 3: Gossip & Queries (Week 3)

**007: Gossip for Consolidation** (4 days)
- Merkle tree for state fingerprinting
- Anti-entropy gossip protocol
- Delta synchronization
- Convergence guarantees

**008: Conflict Resolution** (2 days)
- Vector clock causality tracking
- Confidence-based voting
- Deterministic merge strategies
- Property-based testing for correctness

**009: Distributed Query** (3 days)
- Scatter-gather execution
- Parallel queries to multiple nodes
- Result aggregation with confidence
- Timeout handling for slow nodes

### Phase 4: Validation (Week 4)

**010: Test Framework** (3 days)
- Network simulator for partition scenarios
- Chaos testing harness
- Deterministic replay
- CI integration

**011: Jepsen Testing** (4 days)
- Linearizability checking
- Invariant verification
- Nemesis for random failures
- History analysis

**012: Runbook & Validation** (2 days)
- Operational procedures documentation
- Production validation checklist
- Load testing (100K ops/sec, 5-node cluster)
- SLO definitions

---

## Critical Path

```
001 → 002 → 004 → 005 → 009 → 011 → 012
(20 days total)
```

**Parallel Work**:
- 003 can start after 002
- 007-008 can start after 001
- 010 can start immediately

---

## Consistency Model

### Episodic Memories (Writes)
- **Write**: Append to primary's WAL, return immediately
- **Replication**: Async to N replicas (default N=1)
- **Read**: Primary serves fresh data, replicas may be stale (confidence penalty)
- **Guarantee**: Read-your-writes on primary, eventual consistency on replicas

### Semantic Memories (Consolidation)
- **Process**: Local consolidation on each node independently
- **Sync**: Gossip protocol exchanges state every 60s
- **Conflicts**: Resolved via vector clocks + confidence voting
- **Guarantee**: Eventual consistency, convergence within O(log N) rounds

### Activation Spreading (Queries)
- **Execution**: Scatter to nodes with relevant partitions
- **Aggregation**: Merge partial results, adjust confidence for missing nodes
- **Timeout**: 5s per node (prevents slow nodes from blocking)
- **Guarantee**: Best-effort retrieval with confidence reflecting completeness

---

## Performance Targets

| Metric | Single-Node | Distributed | Notes |
|--------|-------------|-------------|-------|
| Write Latency | 5ms P99 | 10ms P99 | Primary local write + async replication |
| Read Latency (Intra-partition) | 10ms P99 | 20ms P99 | Routing + network overhead |
| Read Latency (Cross-partition) | N/A | 50ms P99 | Scatter-gather across nodes |
| Throughput | 10K ops/sec | 50K ops/sec @ 5 nodes | Linear scaling |
| Cluster Formation | N/A | <5s | SWIM gossip convergence |
| Failure Detection | N/A | <7s | Probe interval + suspect timeout |
| Partition Detection | N/A | <10s | 2 probe cycles |
| Gossip Convergence | N/A | <60s | O(log N) rounds |

---

## Failure Modes

### Node Crash
**Detection**: SWIM marks dead after 7s
**Recovery**: Replica promoted to primary, new replica selected
**Data Loss**: None (replicas have data)

### Network Partition
**Detection**: <50% nodes reachable
**Behavior**: Local-only operations, confidence penalty
**Recovery**: Anti-entropy sync after healing
**Data Loss**: None (eventual consistency)

### Split-Brain
**Detection**: Vector clocks show concurrent primaries
**Prevention**: Partition mode prevents concurrent writes
**Recovery**: Manual resolution, choose authoritative primary
**Data Loss**: Possible (conflicting writes discarded)

### Slow Replica
**Detection**: Replication lag >1s
**Behavior**: Remove from read rotation, alert operator
**Recovery**: Catchup via WAL replay
**Data Loss**: None (eventual consistency)

---

## API Transparency

**Zero Client Changes**:
- Same gRPC/HTTP API for single-node and distributed
- Load balancer routes to any node
- Node handles routing internally
- Clients unaware of distribution

**Internal Changes**:
- New `ClusterService` RPC for node-to-node communication
- Router layer in each node
- Connection pool to remote nodes
- Metadata propagation for tracing

---

## Configuration

### Single-Node (Default)
```toml
[cluster]
enabled = false  # That's it!
```

### Distributed
```toml
[cluster]
enabled = true

[cluster.discovery]
type = "static"
seed_nodes = ["node-1:7946", "node-2:7946"]

[cluster.replication]
factor = 2  # Primary + 1 replica
```

---

## Observability

### Metrics
- `engram.cluster.size`: Total nodes in cluster
- `engram.cluster.alive_nodes`: Healthy nodes
- `engram.replication.lag_ms`: Per-replica lag
- `engram.partition.detected`: Partition events
- `engram.query.scatter_fanout`: Nodes per query

### Tracing
- Distributed tracing with trace ID propagation
- Span per hop (client → node A → node B)
- Causality tracking via vector clocks

### Logging
- Structured logs with node ID, space ID, operation
- Partition events, split-brain warnings
- Rebalancing progress, health changes

---

## Success Criteria

**Functional** (Must Have):
- [ ] Single-node tests pass against distributed cluster (100%)
- [ ] Survive 50% node loss with graceful degradation
- [ ] Convergence within 60s on 100-node cluster
- [ ] Zero data loss in Jepsen tests

**Performance** (Must Have):
- [ ] Intra-partition queries <2x single-node latency
- [ ] Linear scaling to 16 nodes
- [ ] Replication lag <1s under normal load

**Operational** (Should Have):
- [ ] External operator deploys cluster from docs
- [ ] All failure scenarios documented
- [ ] Rolling upgrade with zero downtime

---

## Risks

| Risk | Mitigation | Owner |
|------|------------|-------|
| Gossip doesn't converge | Formal proof, property testing | Task 007 |
| Partition tolerance insufficient | Jepsen testing, chaos engineering | Task 011 |
| Split-brain causes corruption | Vector clocks, operator alerts | Task 003 |
| Performance worse than single-node | Keep single-node fast path, measure overhead | All tasks |
| Operational complexity too high | Comprehensive runbooks, auto-tuning | Task 012 |

---

## Out of Scope

**Deferred to Future Milestones**:
- Multi-region deployment (requires geo-replication)
- Strong consistency modes (requires Raft/Paxos)
- Automatic rebalancing (manual partition migration only)
- Encryption in transit (TLS for node-to-node)
- Cross-space transactions (semantic dependencies)

---

## Next Steps

1. **Review** this plan with team for architectural soundness
2. **Assign** owners to each task (001-012)
3. **Set up** Jepsen environment (Task 010 baseline)
4. **Implement** SWIM membership (Task 001) as foundation
5. **Track** progress with weekly demos showing distributed functionality

---

## Files Created

### Documentation
- `README.md` - Milestone overview and context
- `TECHNICAL_SPECIFICATION.md` - Complete technical design (13,000 words)
- `SUMMARY.md` - This executive summary
- `004-012_remaining_tasks_pending.md` - Concise task descriptions

### Detailed Task Files
- `001_cluster_membership_swim_pending.md` - SWIM protocol (4,800 words)
- `002_node_discovery_configuration_pending.md` - Discovery and config (4,200 words)
- `003_network_partition_handling_pending.md` - Partition tolerance (5,100 words)

**Total**: 30,000+ words of detailed implementation specifications

---

**Document Status**: Final - Ready for Implementation
**Last Updated**: 2025-10-23
**Estimated Start**: After M13 completion
