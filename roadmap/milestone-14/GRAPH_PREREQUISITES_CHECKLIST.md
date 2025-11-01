# M14 Graph Distribution: Prerequisites Checklist

**Status**: NOT READY - 4/6 prerequisites incomplete

---

## Critical Blockers (Must Complete Before M14)

### 1. Deterministic HNSW Construction ❌

**Status**: NOT IMPLEMENTED (current construction is non-deterministic)

**Location**: `engram-core/src/index/hnsw_graph.rs`

**Problem**:
```rust
// Current: RANDOM layer assignment
fn select_layer(&self) -> u8 {
    let uniform = rand::thread_rng().gen::<f32>();
    (-uniform.ln() * self.ml).floor() as u8  // Non-deterministic!
}
```

**Required Fix**:
```rust
// Deterministic: content-based hash
fn select_layer_deterministic(&self, node_id: u32) -> u8 {
    let hash = seahash::hash(&node_id.to_le_bytes());
    let uniform = (hash as f64) / (u64::MAX as f64);
    (-uniform.ln() * self.ml).floor() as u8
}
```

**Validation Test**:
```rust
#[test]
fn test_hnsw_determinism() {
    let graph1 = HnswGraph::new();
    let graph2 = HnswGraph::new();

    let nodes = generate_test_nodes(1000);
    for (seq, node) in nodes.iter().enumerate() {
        graph1.insert_deterministic(node.clone(), seq);
        graph2.insert_deterministic(node.clone(), seq);
    }

    assert!(graph1.topology_equals(&graph2));  // Must be identical
}
```

**Effort**: 2 weeks
**Blocking**: HNSW replication, distributed search

---

### 2. Single-Node HNSW Performance Baselines ❌

**Status**: NOT MEASURED (no systematic benchmarks)

**Required Metrics**:

| Metric | Target | Measurement Method |
|--------|--------|-------------------|
| Insertion throughput | >10K vectors/sec | `criterion` benchmark |
| Search latency P50 | <1ms | HNSW search over 100K nodes |
| Search latency P99 | <5ms | 1000 queries, report P99 |
| Memory footprint | <5KB/vector | RSS measurement |
| Index build time | <1min for 100K | Sequential insertion |

**Validation**:
```bash
cargo bench --bench hnsw_baselines
```

**Output Required**:
```
HNSW Baseline Results:
  Insertion: 15,243 vectors/sec
  Search P50: 0.8ms
  Search P99: 3.2ms
  Memory: 4.1KB/vector
  Build 100K: 48s
```

**Effort**: 1 week
**Blocking**: Cannot validate "distributed <2x overhead" without baseline

---

### 3. Consolidation Determinism ❌

**Status**: NOT IMPLEMENTED (identified by systems planner)

**Location**: `engram-core/src/consolidation/pattern_detector.rs`

**Problem**: Hierarchical clustering with cosine similarity is order-dependent

**Required**: See `MILESTONE_14_CRITICAL_REVIEW.md` Section 3.1

**Effort**: 2-3 weeks
**Blocking**: Distributed consolidation convergence

---

### 4. Vector Clock Implementation ❌

**Status**: NOT IMPLEMENTED

**Required Type**:
```rust
pub struct VectorClock {
    clocks: HashMap<NodeId, u64>,
}

impl VectorClock {
    pub fn tick(&mut self, node: NodeId) {
        *self.clocks.entry(node).or_insert(0) += 1;
    }

    pub fn merge(&mut self, other: &VectorClock) {
        for (node, &clock) in &other.clocks {
            let entry = self.clocks.entry(*node).or_insert(0);
            *entry = (*entry).max(clock);
        }
    }

    pub fn happens_before(&self, other: &VectorClock) -> bool {
        // self < other iff ∀ node: self[node] ≤ other[node] AND ∃ node: self[node] < other[node]
    }
}
```

**Tests Required**:
- Happens-before transitivity
- Concurrent event detection
- Merge commutativity

**Effort**: 1 week
**Blocking**: Graph topology consistency, conflict resolution

---

### 5. M13 Completion ⚠️

**Status**: IN PROGRESS (15/21 tasks done, 71% complete)

**Blocking Tasks**:
- `006_reconsolidation_core_pending.md` (CRITICAL)
- 5 other pending tasks

**Impact**: Reconsolidation affects distributed conflict resolution semantics

**Effort**: 2-3 weeks
**Blocking**: Understanding complete memory consolidation behavior

---

### 6. 7-Day Single-Node Soak Test ❌

**Status**: NOT RUN (only 1-hour soak test from M6)

**Required**:
- 168+ hours continuous operation
- Multi-tenant workload (10+ spaces)
- Memory leak detection (valgrind, heaptrack)
- Consolidation convergence monitoring
- Performance stability (no degradation)

**Validation**:
```bash
./scripts/soak_test.sh --duration 168h --spaces 10 --memories 1M
```

**Metrics to Track**:
- RSS over time (detect memory leaks)
- Consolidation cycle timing (should remain stable)
- P99 latency (should not degrade)
- Connection pool stability

**Effort**: 1-2 weeks (setup + 1 week runtime)
**Blocking**: Production readiness, distributed stability assumptions

---

## Prerequisites Summary

| Prerequisite | Status | Effort | Priority |
|--------------|--------|--------|----------|
| Deterministic HNSW | ❌ NOT DONE | 2 weeks | **CRITICAL** |
| HNSW Baselines | ❌ NOT DONE | 1 week | **CRITICAL** |
| Consolidation Determinism | ❌ NOT DONE | 2-3 weeks | **CRITICAL** |
| Vector Clocks | ❌ NOT DONE | 1 week | **CRITICAL** |
| M13 Completion | ⚠️ IN PROGRESS | 2-3 weeks | **HIGH** |
| 7-Day Soak Test | ❌ NOT DONE | 1-2 weeks | **HIGH** |
| **TOTAL** | **0/6 Complete** | **9-12 weeks** | **BLOCKER** |

---

## Go/No-Go Decision Criteria

### MUST HAVE (before M14 implementation)

- [x] ~~All 1,035 tests passing~~ ✓ (99.6% passing, 5 failures)
- [ ] Deterministic HNSW construction proven (property tests pass)
- [ ] HNSW performance baselines documented
- [ ] Consolidation determinism proven
- [ ] Vector clock implementation complete
- [ ] M13 completion (21/21 tasks, including reconsolidation)

### SHOULD HAVE (before distributed graph work)

- [ ] 7-day single-node soak test passes
- [ ] Memory leak analysis clean (valgrind)
- [ ] Single-node performance regression framework
- [ ] Observability stack validated

**Current Decision**: **NO GO** (0/6 MUST HAVE complete)

---

## Next Steps

### Option A: Complete Prerequisites (Recommended)

**Timeline**: 9-12 weeks

**Order**:
1. Vector clocks (1 week) - parallel with other work
2. Deterministic HNSW (2 weeks) - CRITICAL PATH
3. HNSW baselines (1 week) - after deterministic construction
4. M13 completion (2-3 weeks) - parallel with HNSW work
5. Consolidation determinism (2-3 weeks) - depends on M13
6. 7-day soak test (1-2 weeks) - final validation

**Outcome**: Prerequisites met, ready for M14 implementation

### Option B: Defer M14 to M18+ (Also Recommended)

**Rationale**:
- Single-node supports production scale (millions of memories)
- Focus M16-M17 on production hardening instead
- Research distributed graphs in parallel
- Start M14 when prerequisites are met AND scale demands it

**Timeline**:
1. Complete M13 (2-3 weeks)
2. M14 prerequisites (6-7 weeks, in parallel with M16)
3. M16-M17: Production hardening (8-12 weeks)
4. M18: Distributed graph research (8-12 weeks)
5. M19: Production distributed graph (20-30 weeks)

**Outcome**: Production-ready distributed graph in 18-24 months

---

## Daily Standup Format

**Question**: "Are we ready to start M14?"

**Answer Template**:
```
Prerequisites Status: X/6 complete

Blockers:
- [ ] Deterministic HNSW (2 weeks remaining)
- [ ] HNSW baselines (1 week remaining)
- [ ] Vector clocks (1 week remaining)
- [ ] Consolidation determinism (3 weeks remaining)
- [ ] M13 completion (2 weeks remaining)
- [ ] Soak test (2 weeks remaining)

ETA to M14-ready: X weeks

Recommendation: [DEFER / PROCEED]
```

---

## References

- Full Analysis: `GRAPH_ENGINE_DISTRIBUTION_ANALYSIS.md`
- Executive Summary: `GRAPH_DISTRIBUTION_EXECUTIVE_SUMMARY.md`
- Systems Review: `MILESTONE_14_CRITICAL_REVIEW.md`

**Last Updated**: 2025-10-31
**Owner**: Graph Engine Architecture
