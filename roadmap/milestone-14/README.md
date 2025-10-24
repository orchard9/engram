# Milestone 14: Distributed Architecture

**Objective**: Design partitioned memory across nodes with gossip-based consolidation sync. Enable transparent distribution without changing API semantics.

**Duration**: 18-24 days (12 tasks)

**Status**: Planning

## Context

Engram currently operates as a single-node cognitive graph database with:
- Multi-tenant memory spaces (M7)
- Consolidation system transforming episodic to semantic memories (M6)
- Activation spreading with parallel workers (M3)
- Three-tier storage: hot/warm/cold
- gRPC and HTTP APIs with streaming support (M15)

This milestone extends Engram to support optional distributed deployment while maintaining:
- **API Transparency**: Zero client code changes for distributed vs single-node
- **Partition Tolerance**: Graceful degradation during network splits
- **Eventual Consistency**: Consolidation converges across nodes
- **Optional Deployment**: Single-node remains first-class citizen

## CAP Theorem Position

Engram is an **AP system** (Availability + Partition Tolerance):

- **Availability**: Nodes continue serving local queries during partitions
- **Partition Tolerance**: Network splits cause graceful degradation, not failures
- **Consistency**: Eventual consistency with bounded staleness for consolidation

Rationale: Cognitive memory retrieval must remain available even with stale data. A human brain doesn't stop recalling memories during "network partitions" (e.g., isolation). However, it may recall with lower confidence or incomplete information.

## Consistency Model

**Eventually Consistent with Probabilistic Semantics**

1. **Episodic Memories**: Primary-per-space with async replication
   - Write to primary node returns immediately
   - Asynchronous replication to N replicas
   - Read-your-writes consistency on primary
   - Stale reads possible on replicas (marked with confidence penalty)

2. **Semantic Memories (Consolidation)**: Gossip-based anti-entropy
   - Each node performs local consolidation independently
   - Gossip protocol syncs consolidation results
   - Conflict resolution via vector clocks + confidence voting
   - Convergence guarantee: all nodes eventually agree on semantic patterns

3. **Activation Spreading**: Scatter-gather with degraded results
   - Query routes to nodes with relevant partitions
   - Partial results aggregated with confidence adjustment
   - Missing partitions reduce confidence, don't block query

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        Client Application                        │
└───────────────────────────┬─────────────────────────────────────┘
                            │ gRPC/HTTP (unchanged API)
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Engram Cluster                              │
│                                                                   │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐  │
│  │  Node A  │◄──►│  Node B  │◄──►│  Node C  │◄──►│  Node D  │  │
│  │          │    │          │    │          │    │          │  │
│  │ Spaces:  │    │ Spaces:  │    │ Spaces:  │    │ Spaces:  │  │
│  │  1,2,5   │    │  1,3,6   │    │  2,3,4   │    │  4,5,6   │  │
│  │          │    │          │    │          │    │          │  │
│  │ Primary: │    │ Primary: │    │ Primary: │    │ Primary: │  │
│  │  1,2     │    │  3       │    │  4       │    │  5,6     │  │
│  └──────────┘    └──────────┘    └──────────┘    └──────────┘  │
│       │               │               │               │          │
│       └───────────────┴───────────────┴───────────────┘          │
│                   Gossip Protocol Layer                          │
│         (Consolidation Sync, Membership, Failure Detection)      │
└─────────────────────────────────────────────────────────────────┘
```

### Key Design Decisions

1. **Memory Space = Partitioning Unit**
   - Each memory space has a primary node + N replicas
   - Spaces are independent partitioning units (no cross-space spreading)
   - Allows per-tenant placement policies

2. **Gossip for Coordination**
   - No external coordination service (no ZooKeeper, etcd, Consul)
   - SWIM-based membership and failure detection
   - Anti-entropy for consolidation state synchronization
   - Scales to 100+ nodes without coordination bottleneck

3. **Primary-Per-Space Model**
   - Each space has one primary node (writes)
   - Primary election via gossip consensus
   - Replicas serve stale reads with confidence penalty
   - Failover: replica promotes to primary on timeout

4. **Transparent Routing**
   - Load balancer routes to any node
   - Node routes to primary/replicas based on operation
   - Scatter-gather for queries spanning partitions
   - Client sees single logical database

## Technical Implementation Plan

### Phase 1: Foundation (Tasks 001-003)
- Cluster membership with SWIM protocol
- Node discovery and health monitoring
- Configuration for cluster mode vs single-node

### Phase 2: Partitioning (Tasks 004-006)
- Memory space assignment to primary nodes
- Replication protocol for episodic memories
- Routing layer for directing operations to correct nodes

### Phase 3: Gossip Synchronization (Tasks 007-009)
- Anti-entropy gossip for consolidation state
- Vector clock implementation for causality tracking
- Conflict resolution for divergent consolidations

### Phase 4: Distributed Operations (Tasks 010-011)
- Scatter-gather query execution
- Confidence aggregation for partial results
- Network partition handling and recovery

### Phase 5: Validation (Task 012)
- Jepsen-style consistency testing
- Performance benchmarking under various conditions
- Operational runbook for distributed deployment

## Risk Analysis

### Critical Risks

1. **Partitioning Breaks Activation Spreading**
   - Risk: Spreading across partitions introduces unbounded latency
   - Mitigation: Limit spreading to single partition, use confidence penalty for cross-partition
   - Validation: Benchmark shows <2x latency vs single-node for intra-partition queries

2. **Consolidation Divergence**
   - Risk: Nodes consolidate differently, never converge
   - Mitigation: Deterministic pattern detection algorithms, conflict resolution with confidence voting
   - Validation: Formal proof that gossip protocol converges under network asynchrony

3. **Split-Brain Scenarios**
   - Risk: Network partition creates two primaries for same space
   - Mitigation: Quorum-based writes, vector clock detection of concurrent primaries
   - Validation: Jepsen testing with network partitions, verify data loss bounds

4. **Performance Degradation**
   - Risk: Distributed overhead makes system slower than single-node
   - Mitigation: Keep single-node fast path, only pay distribution cost when needed
   - Validation: Single-node performance unchanged, distributed shows linear scaling

### Operational Risks

1. **Debugging Complexity**
   - Distributed systems are notoriously hard to debug
   - Mitigation: Comprehensive tracing with causality tracking, deterministic replay tools
   - Validation: Simulated partition scenarios with full trace analysis

2. **Configuration Complexity**
   - Many knobs to tune (replication factor, gossip intervals, timeouts)
   - Mitigation: Auto-tuning based on measured latencies, safe defaults
   - Validation: Default config works for 80% of deployments

3. **Operational Runbook Gaps**
   - Operators need playbooks for failure scenarios
   - Mitigation: Document every failure mode with recovery steps
   - Validation: Operations team reviews and approves runbooks

## Success Criteria

1. **API Transparency**: Existing single-node tests pass against distributed cluster
2. **Partition Tolerance**: 99.9% availability during simulated network partitions
3. **Consistency**: Consolidation converges within 1 hour on 100-node cluster
4. **Performance**: Intra-partition queries within 2x of single-node latency
5. **Jepsen Validation**: No data loss or corruption under partition scenarios
6. **Operational Readiness**: Complete runbooks for deployment, scaling, recovery

## Out of Scope

The following are explicitly NOT included in Milestone 14:

- **Multi-region deployment**: All nodes assumed in single data center (latency <10ms)
- **Cross-space transactions**: Memory spaces remain independent
- **Strong consistency**: No linearizability guarantees (eventual consistency only)
- **Automatic rebalancing**: Manual partition reassignment only
- **Byzantine fault tolerance**: Assumes non-malicious nodes
- **Dynamic schema migration**: Schema changes require cluster restart
- **Encryption in transit**: TLS for node-to-node communication deferred to M17
- **Multi-tenancy isolation**: Relies on M7 isolation, no additional distributed isolation

## Dependencies

- Milestone 6: Consolidation system (must understand consolidation semantics)
- Milestone 7: Memory space support (spaces are partitioning units)
- Milestone 15: Multi-interface layer (routing logic integrates with gRPC/HTTP)

## Next Steps

1. Review this plan with team for architectural soundness
2. Break down tasks 001-012 with specific file paths and integration points
3. Identify performance benchmarks to track throughout milestone
4. Create Jepsen test suite skeleton before implementation starts
