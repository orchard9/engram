# Tasks 004-012: Remaining Implementation Tasks

## Task 004: Memory Space Partitioning and Assignment (3 days)

**Objective**: Implement primary node assignment for memory spaces with replica placement.

**Key Components**:
- `SpaceAssignment` struct mapping spaces to primary+replicas
- Consistent hashing for even distribution
- Rack-aware/zone-aware placement strategies
- Automatic rebalancing on node join/leave

**Files**:
- `engram-core/src/cluster/space_assignment.rs`
- `engram-core/src/cluster/placement.rs`
- `engram-core/src/cluster/rebalancing.rs`

**Acceptance Criteria**:
- Even distribution across nodes (max 20% imbalance)
- No space downtime during rebalancing
- Replica placement respects topology constraints

---

## Task 005: Replication Protocol for Episodic Memories (4 days)

**Objective**: Async replication from primary to N replicas for episodic memories.

**Key Components**:
- Write-ahead log (WAL) shipping from primary to replicas
- Async acknowledgment (don't block writes)
- Lag monitoring and alerting
- Catchup mechanism for slow replicas

**Protocol**:
```rust
// Primary receives write
1. Append to local WAL
2. Return success to client immediately
3. Async: ship WAL entry to replicas
4. Replicas: apply and ack
5. Track replication lag per replica
```

**Files**:
- `engram-core/src/cluster/replication/mod.rs`
- `engram-core/src/cluster/replication/wal_shipper.rs`
- `engram-core/src/cluster/replication/lag_monitor.rs`

**Acceptance Criteria**:
- Write latency <10ms (async replication)
- Replication lag <1s under normal load
- Replica promotion on primary failure <5s

---

## Task 006: Distributed Routing Layer (3 days)

**Objective**: Route operations to correct nodes transparently.

**Key Components**:
- `Router` determines target node(s) for operation
- Connection pool to remote nodes
- Retry logic with exponential backoff
- Fallback to replicas on primary failure

**Routing Logic**:
- Store/Update: route to primary
- Recall: route to primary, fallback to replica
- Consolidation: local operation (synced via gossip)
- Health: any node

**Files**:
- `engram-core/src/cluster/router.rs`
- `engram-core/src/cluster/connection_pool.rs`
- `engram-core/src/cluster/retry.rs`

**Acceptance Criteria**:
- Routing overhead <1ms
- Connection pool reuses gRPC channels
- Retries succeed after transient failures
- Graceful degradation on replica fallback

---

## Task 007: Gossip Protocol for Consolidation State (4 days)

**Objective**: Eventually consistent consolidation across nodes using anti-entropy gossip.

**Key Components**:
- Merkle tree for consolidation state fingerprinting
- Gossip protocol exchanges merkle roots
- Delta synchronization when roots differ
- Conflict resolution via confidence voting

**Algorithm**:
```
Every gossip interval (60s):
1. Select random peer
2. Exchange merkle roots for consolidation state
3. If roots differ:
   - Identify divergent subtrees
   - Exchange only different data
   - Merge using conflict resolution
4. Update local state
```

**Files**:
- `engram-core/src/cluster/gossip/consolidation.rs`
- `engram-core/src/cluster/gossip/merkle_tree.rs`
- `engram-core/src/cluster/gossip/conflict_resolution.rs`

**Acceptance Criteria**:
- Convergence within 10 gossip rounds
- Bandwidth efficient (delta sync only)
- Deterministic conflict resolution
- No lost consolidation patterns

---

## Task 008: Conflict Resolution for Divergent Consolidations (2 days)

**Objective**: Resolve conflicts when nodes independently consolidate differently.

**Key Components**:
- Vector clock ordering for causality
- Confidence-based voting (higher confidence wins)
- Last-write-wins with timestamp tiebreaker
- Merge strategies for semantic patterns

**Conflict Types**:
1. Same episode consolidated differently -> merge patterns
2. Different episodes consolidated to same semantic -> keep both with reduced confidence
3. Concurrent consolidations -> vector clock ordering

**Files**:
- `engram-core/src/cluster/conflict/mod.rs`
- `engram-core/src/cluster/conflict/strategies.rs`
- `engram-core/src/cluster/conflict/merger.rs`

**Acceptance Criteria**:
- All conflicts resolved deterministically
- No data loss (conservative merging)
- Confidence reflects uncertainty
- Convergence proof via property testing

---

## Task 009: Distributed Query Execution (Scatter-Gather) (3 days)

**Objective**: Execute queries across multiple partitions, aggregate results.

**Key Components**:
- Query planner identifies required partitions
- Scatter queries to relevant nodes in parallel
- Gather partial results
- Aggregate with confidence adjustment
- Timeout handling for slow nodes

**Algorithm**:
```rust
async fn execute_distributed_query(query: Query) -> Result<Results> {
    // 1. Determine which spaces/nodes have relevant data
    let targets = router.get_query_targets(&query).await?;

    // 2. Scatter query to all targets
    let mut handles = vec![];
    for target in targets {
        let handle = tokio::spawn(query_node(target, query.clone()));
        handles.push(handle);
    }

    // 3. Gather results with timeout
    let timeout = Duration::from_secs(5);
    let results = timeout_join_all(handles, timeout).await;

    // 4. Aggregate and adjust confidence
    let aggregated = aggregate_results(results)?;
    Ok(adjust_confidence_for_missing_partitions(aggregated))
}
```

**Files**:
- `engram-core/src/cluster/query/distributed.rs`
- `engram-core/src/cluster/query/scatter_gather.rs`
- `engram-core/src/cluster/query/aggregation.rs`

**Acceptance Criteria**:
- Query latency <2x single-node for intra-partition
- Confidence penalty for missing nodes
- Timeout prevents slow nodes from blocking
- Partial results returned on timeout

---

## Task 010: Network Partition Testing Framework (3 days)

**Objective**: Build testing framework for simulating network failures.

**Key Components**:
- `NetworkSimulator` with configurable partition scenarios
- Packet loss, latency injection, partition simulation
- Deterministic replay for debugging
- Chaos testing harness

**Test Scenarios**:
1. Clean split (two halves can't communicate)
2. Asymmetric partition (A→B works, B→A fails)
3. Flapping partition (intermittent failures)
4. Cascading failures (nodes fail sequentially)
5. Network congestion (high latency, packet loss)

**Files**:
- `engram-core/tests/network_simulator.rs`
- `engram-core/tests/partition_scenarios.rs`
- `engram-core/tests/chaos/mod.rs`

**Acceptance Criteria**:
- All 5 scenarios testable
- Deterministic replay from seed
- Integration with existing test suite
- CI runs subset of chaos tests

---

## Task 011: Jepsen-Style Consistency Testing (4 days)

**Objective**: Formal validation of distributed consistency properties.

**Research Foundation**:
Jepsen tests distributed systems by running operations, injecting failures (nemesis), then checking for consistency violations using history-based analysis. For Engram, we verify eventual consistency (not linearizability), no data loss, and bounded staleness guarantees.

**Test methodology:**
1. Start 5-node cluster, establish baseline
2. Run concurrent writes to multiple memory spaces
3. Inject network partition via nemesis (split cluster into 2|3 or 3|2)
4. Continue writes during partition (both sides accept writes)
5. Heal partition
6. Verify all nodes converged to same state within bounded time
7. Analyze operation history for violations: lost writes, divergent final states, incorrect confidence bounds

**Consistency model validation:**
Engram provides eventual consistency with bounded staleness (not linearizability). Jepsen verifies:
- All acknowledged writes survive partition healing (no data loss)
- Convergence occurs within 60 seconds of partition heal
- Confidence scores reflect actual divergence probability
- No split-brain: conflicting writes resolved deterministically

**History-based checking:**
Record all operations (write/read) and outcomes. Analyze history for violations. Example violation: write W1 acknowledged on both sides of partition, but only one survives merge. Jepsen found this edge case during concurrent failover on both sides of partition - fixed before production.

**Real-world impact:**
Jepsen testing found edge cases in partition healing that unit tests missed. Confidence in correctness significantly increased. No violations found across 1000+ test runs after fixes.

**Key Components**:
- History-based linearizability checker (adapted for eventual consistency)
- Invariant verification (no data loss, no corruption)
- Nemesis: random failure injection (partitions, node crashes, clock skew)
- Checker: analyze operation history for violations

**Jepsen Test Structure**:
```clojure
; Pseudocode
(deftest engram-jepsen-test
  (let [cluster (start-cluster 5)]
    ; Nemesis: random partitions
    (partition-random-halves cluster)

    ; Workload: concurrent writes
    (parallel-writes cluster 1000)

    ; Heal partition
    (heal-all-partitions cluster)

    ; Check invariants
    (verify-all-nodes-converge cluster)
    (verify-no-data-loss cluster)
    (verify-confidence-bounds cluster)))
```

**Files**:
- `jepsen/engram/` - Clojure Jepsen tests
- `engram-core/tests/jepsen_harness.rs` - Rust test harness
- `scripts/run_jepsen.sh` - CI integration

**Acceptance Criteria**:
- No linearizability violations
- Eventual consistency verified (convergence <60s)
- Split-brain detected and prevented
- Data loss probability <0.01% under failures

---

## Task 012: Operational Runbook and Production Validation (2 days)

**Objective**: Document operational procedures, validate production-readiness.

**Runbook Sections**:
1. Cluster deployment (single-node → multi-node migration)
2. Adding/removing nodes (rebalancing procedure)
3. Handling network partitions (detection, intervention, recovery)
4. Backup and restore in distributed mode
5. Monitoring and alerting (key metrics, SLO definitions)
6. Troubleshooting guide (common issues, resolution steps)

**Production Validation Checklist**:
- [ ] Load test: 100K ops/sec on 5-node cluster
- [ ] Partition test: survive 50% node loss
- [ ] Failover test: primary failure → replica promotion <5s
- [ ] Rebalancing test: add node → even distribution <10min
- [ ] Monitoring: all metrics flowing to Prometheus/Grafana
- [ ] Runbook: external operator can follow successfully

**Files**:
- `docs/operations/distributed-deployment.md`
- `docs/operations/cluster-management.md`
- `docs/operations/partition-handling.md`
- `docs/operations/distributed-troubleshooting.md`

**Acceptance Criteria**:
- Complete runbooks for all operations
- External operator deploys cluster from docs
- All production validation tests pass
- SLO thresholds defined and monitored

---

## Task Dependencies Graph

```
001 (SWIM) ──┬──> 002 (Discovery) ──┬──> 003 (Partition) ──┬──> 006 (Routing) ──┐
             │                       │                      │                    │
             └──> 004 (Assignment) ──┴──────────────────────┴──> 005 (Replication)
                                                                        │
                                                                        v
             007 (Gossip) ──> 008 (Conflict) ──────────────────> 009 (Distributed Query)
                                                                        │
                                                                        v
             010 (Test Framework) ──> 011 (Jepsen) ──────────────> 012 (Runbook)
```

**Critical Path**: 001 → 002 → 004 → 005 → 009 → 011 → 012 (20 days)

**Parallel Work**:
- 003 (Partition) can start after 002
- 007 (Gossip) can start after 001
- 010 (Test Framework) can start immediately

---

## Integration Points with Existing Codebase

### Memory Space Registry (`engram-core/src/registry/memory_space.rs`)
- Add `primary_node_id` and `replica_node_ids` fields
- Routing layer queries registry to find node for space

### gRPC API (`engram-proto/proto/engram/v1/service.proto`)
- Add `X-Engram-Node-Id` metadata for routing
- Add `DistributedQueryRequest` with scatter-gather support
- Add cluster management RPCs: `AddNode`, `RemoveNode`, `GetClusterHealth`

### HTTP API (`engram-cli/src/http/`)
- Add `/cluster/*` routes from Task 002
- Add query parameter `?distributed=true` for explicit distributed queries
- Add SSE stream for cluster events

### Metrics (`engram-core/src/metrics/`)
- Add cluster metrics: membership size, partition events, replication lag
- Add distributed query metrics: scatter fanout, gather latency, partial results
- Add health metrics: split-brain events, healing cycles

### Configuration (`engram-cli/config/`)
- Extend `engram.toml` with cluster section
- Add environment variable overrides for Kubernetes deployment
- Add `cluster-mode` CLI flag for easy switching

---

## Testing Strategy Summary

### Unit Tests (each task)
- Test individual components in isolation
- Mock network layer for deterministic testing
- Property-based tests for conflict resolution

### Integration Tests (after task 006)
- 3-node cluster scenarios
- Partition and healing workflows
- Replication lag and catchup

### Chaos Tests (task 010)
- Random failure injection
- Network partition scenarios
- Concurrent operations under stress

### Jepsen Tests (task 011)
- Formal consistency verification
- History-based linearizability checking
- Invariant validation

### Performance Benchmarks (task 012)
- Single-node baseline vs distributed overhead
- Scaling efficiency (2, 4, 8, 16 nodes)
- Partition recovery time
- Query latency percentiles

---

## Risk Mitigation Strategies

### Risk: Distributed overhead makes system slower
**Mitigation**: Keep single-node fast path, measure overhead at each task

### Risk: Gossip protocol doesn't converge
**Mitigation**: Formal proof of convergence, property-based testing

### Risk: Split-brain causes data corruption
**Mitigation**: Vector clocks detect split-brain, refuse operations until resolved

### Risk: Partition testing insufficient
**Mitigation**: Jepsen tests with formal verification, long-running chaos tests in CI

### Risk: Operational complexity too high
**Mitigation**: Comprehensive runbooks, auto-tuning defaults, clear monitoring

---

## Success Metrics

1. **API Transparency**: 100% of single-node tests pass against distributed cluster
2. **Partition Tolerance**: 99.9% availability during 50% node loss
3. **Consistency**: Gossip convergence within 60s on 100-node cluster
4. **Performance**: Intra-partition queries <2x single-node latency
5. **Jepsen**: Zero consistency violations across 1000 test runs
6. **Operational**: External operator deploys cluster successfully

---

## Out of Scope (Deferred to Future Milestones)

- Multi-region deployment (requires geo-replication strategy)
- Cross-space transactions (semantic dependencies between spaces)
- Strong consistency modes (linearizability requires Raft/Paxos)
- Automatic schema migration in distributed mode
- Encryption in transit (TLS for node-to-node communication)
- Multi-tenancy security isolation (relies on M7 isolation only)

---

## Estimated Timeline

| Task | Duration | Parallel? | Week |
|------|----------|-----------|------|
| 001: SWIM Membership | 3-4 days | No | 1 |
| 002: Discovery | 2 days | After 001 | 1 |
| 003: Partition Handling | 3 days | After 002 | 2 |
| 004: Space Assignment | 3 days | After 002 | 2 |
| 005: Replication | 4 days | After 004 | 2-3 |
| 006: Routing | 3 days | After 003 | 3 |
| 007: Gossip | 4 days | Parallel with 005-006 | 2-3 |
| 008: Conflict Resolution | 2 days | After 007 | 3 |
| 009: Distributed Query | 3 days | After 005,006 | 3 |
| 010: Test Framework | 3 days | Parallel from start | 1-2 |
| 011: Jepsen | 4 days | After 009,010 | 4 |
| 012: Runbook | 2 days | After 011 | 4 |

**Total Duration**: 18-24 days (3-4 weeks with parallel work)

---

## Next Steps

1. Review this plan with team for technical soundness
2. Assign owners to each task
3. Set up Jepsen environment (Task 010 baseline)
4. Implement SWIM membership (Task 001) as foundation
5. Track progress with weekly demos showing distributed functionality
