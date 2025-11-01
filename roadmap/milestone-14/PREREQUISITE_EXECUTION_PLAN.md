# Milestone 14 Prerequisites: Execution Plan

**Status**: Contingent - Execute if Option B (Prerequisites First) approved
**Duration**: 6-10 weeks
**Owner**: TBD
**Success Criteria**: All prerequisites met, M14 ready to start

---

## Overview

This plan details the execution of prerequisites required before Milestone 14 (Distributed Architecture) can begin. Each prerequisite is broken down into concrete tasks with acceptance criteria.

**Prerequisites**:
1. Test Health: 100% passing (1,035/1,035)
2. M13 Completion: 21/21 tasks done
3. Consolidation Determinism: Proven convergence
4. Single-Node Baselines: Performance documented
5. Production Soak Testing: 7+ days validated

---

## Prerequisite 1: Test Health (Week 1)

### Objective
Fix 5 failing tests to achieve 100% test health (1,035/1,035 passing)

### Current Status
```
test result: FAILED. 1030 passed; 5 failed; 5 ignored; 0 measured; 0 filtered out
```

### Tasks

**Task 1.1: Identify Failing Tests** (Day 1)
```bash
cargo test --workspace --lib 2>&1 | grep -A 10 "FAILED"
# Document:
# - Test names
# - Failure reasons
# - Affected modules
```

**Deliverable**: `tmp/failing_tests_analysis.md` with root cause analysis

**Task 1.2: Fix Failing Tests** (Days 2-4)
- Root cause each failure
- Implement fixes with regression tests
- Verify no new failures introduced

**Deliverable**: All tests passing locally

**Task 1.3: CI Validation** (Day 5)
```bash
make quality  # Must pass with zero warnings
cargo test --workspace --all-targets
```

**Deliverable**: Clean CI run, 1,035/1,035 passing

### Acceptance Criteria
- [ ] All 1,035 tests passing
- [ ] `make quality` produces zero clippy warnings
- [ ] CI green on main branch
- [ ] No ignored tests (unless documented exception)

### Risk Mitigation
- If failure is in consolidation system: blocks determinism work
- If failure is in spreading: may indicate race condition
- If failure is timeout: may indicate performance regression

**Escalation**: If any test cannot be fixed in 5 days, escalate for architectural review

---

## Prerequisite 2: M13 Completion (Weeks 1-3)

### Objective
Complete 6 pending M13 tasks (15/21 → 21/21)

### Current Status
```
roadmap/milestone-13/
  - 15 tasks complete
  - 6 tasks pending
```

### Tasks

**Task 2.1: Prioritize Pending Tasks** (Day 1)
Analyze dependencies:
- `001_zero_overhead_metrics_pending.md`
- `002_semantic_priming_pending.md`
- `005_retroactive_fan_effect_pending.md`
- `006_reconsolidation_core_pending.md` ← CRITICAL for M14
- Others

**Deliverable**: Dependency graph, critical path identified

**Task 2.2: Reconsolidation Core** (Week 1-2, PRIORITY)
File: `006_reconsolidation_core_pending.md` (23,266 lines)

Why critical: Reconsolidation affects consolidation semantics. Distributed conflict resolution depends on understanding complete behavior.

**Subtasks**:
1. Review 23K line spec (Days 1-2)
2. Implement core reconsolidation (Days 3-7)
3. Integration tests (Days 8-9)
4. Validate against cognitive psychology research (Day 10)

**Deliverable**: Reconsolidation system complete, tests passing

**Task 2.3: Remaining 5 Tasks** (Weeks 2-3, PARALLEL)
Assign owners to:
- `001_zero_overhead_metrics_pending.md`
- `002_semantic_priming_pending.md`
- `005_retroactive_fan_effect_pending.md`
- Others

Execute in parallel with weekly progress reviews

**Deliverable**: All tasks moved to `*_complete.md`

### Acceptance Criteria
- [ ] 21/21 M13 tasks complete
- [ ] Reconsolidation core validated (CRITICAL)
- [ ] All M13 tests passing
- [ ] Cognitive pattern behaviors documented

### Risk Mitigation
- Reconsolidation may reveal consolidation non-determinism
- If so: feeds into Prerequisite 3 (determinism work)

**Timeline Risk**: If reconsolidation takes >2 weeks, may need to defer some M13 tasks and proceed with determinism work

---

## Prerequisite 3: Consolidation Determinism (Weeks 2-4)

### Objective
Prove consolidation produces identical results across multiple runs

### Current Problem
```rust
// engram-core/src/consolidation/pattern_detector.rs
fn cluster_episodes(&self, episodes: &[Episode]) -> Vec<Vec<Episode>> {
    let mut clusters: Vec<Vec<Episode>> =
        episodes.iter().map(|ep| vec![ep.clone()]).collect();
    // DashMap iteration is non-deterministic
    // Clustering merge order undefined for equal similarities
}
```

### Tasks

**Task 3.1: Determinism Analysis** (Week 2, Days 1-3)
```rust
// Create property test to expose non-determinism
#[test]
fn test_consolidation_non_determinism() {
    let episodes = generate_test_episodes(100);
    let mut signatures = HashSet::new();

    for run in 0..1000 {
        let patterns = detector.detect_patterns(&episodes);
        let sig = compute_pattern_signature(&patterns);
        signatures.insert(sig);
    }

    println!("Unique signatures: {}", signatures.len());
    // Currently: signatures.len() > 1 (NON-DETERMINISTIC)
    // Goal: signatures.len() == 1 (DETERMINISTIC)
}
```

**Deliverable**: Quantified non-determinism (how many unique signatures in 1000 runs?)

**Task 3.2: Root Cause Analysis** (Week 2, Days 4-5)
Identify sources of non-determinism:
1. DashMap iteration order
2. Clustering merge order (equal similarities)
3. Floating-point precision differences
4. Random number generation (if any)

**Deliverable**: `docs/architecture/consolidation_determinism_analysis.md`

**Task 3.3: Implement Deterministic Clustering** (Week 3)

**Option A: Stable Sorting** (Recommended)
```rust
fn cluster_episodes_deterministic(&self, episodes: &[Episode]) -> Vec<Vec<Episode>> {
    // Sort episodes by deterministic key BEFORE clustering
    let mut sorted_episodes = episodes.to_vec();
    sorted_episodes.sort_by(|a, b| {
        a.id.cmp(&b.id)  // Lexicographic ID ordering
    });

    // Initialize clusters with sorted episodes
    let mut clusters: Vec<Vec<Episode>> =
        sorted_episodes.iter().map(|ep| vec![ep.clone()]).collect();

    // Break similarity ties using episode IDs
    while clusters.len() > 1 {
        let (i, j) = find_closest_cluster_pair_deterministic(&clusters);
        // Merge deterministically
    }
}

fn find_closest_cluster_pair_deterministic(
    clusters: &[Vec<Episode>]
) -> (usize, usize) {
    let mut best = (0, 1);
    let mut best_similarity = -1.0;

    for i in 0..clusters.len() {
        for j in (i+1)..clusters.len() {
            let sim = compute_similarity(&clusters[i], &clusters[j]);

            // Break ties using cluster representative IDs
            if sim > best_similarity ||
               (sim == best_similarity &&
                compare_cluster_ids(&clusters[i], &clusters[j]) == Ordering::Less)
            {
                best = (i, j);
                best_similarity = sim;
            }
        }
    }

    best
}
```

**Option B: CRDT-Based** (Future consideration)
- G-Set for patterns (grow-only set)
- Commutative and associative by construction
- Higher complexity, defer to M17+ (multi-region)

**Deliverable**: Deterministic clustering implementation

**Task 3.4: Property-Based Testing** (Week 4, Days 1-3)
```rust
use proptest::prelude::*;

proptest! {
    #[test]
    fn test_deterministic_consolidation(
        episodes in prop::collection::vec(arbitrary_episode(), 10..100)
    ) {
        let mut signatures = HashSet::new();

        // Run 1000 times with same input
        for seed in 0..1000 {
            let patterns = detector.detect_patterns_seeded(&episodes, seed);
            let sig = compute_pattern_signature(&patterns);
            signatures.insert(sig);
        }

        // MUST produce exactly 1 unique signature
        prop_assert_eq!(signatures.len(), 1);
    }
}
```

**Deliverable**: 1000+ property test runs, 1 unique signature

**Task 3.5: Performance Regression Check** (Week 4, Days 4-5)
Validate determinism doesn't slow down consolidation:
```rust
#[bench]
fn bench_consolidation_deterministic(b: &mut Bencher) {
    let episodes = generate_episodes(1000);
    b.iter(|| detector.detect_patterns_deterministic(&episodes));
}
```

**Target**: <10% performance degradation vs. non-deterministic version

**Deliverable**: Performance benchmarks, <10% regression

### Acceptance Criteria
- [ ] Property tests: 1000 runs → 1 unique signature (deterministic)
- [ ] Performance: <10% regression from non-deterministic version
- [ ] Integration tests: Deterministic consolidation in full system
- [ ] Documentation: Determinism guarantees documented

### Risk Mitigation
- If determinism impossible: Consider primary-only consolidation (gossip distributes)
- If performance degradation >10%: Optimize or accept tradeoff

**Escalation**: If determinism cannot be achieved in 3 weeks, escalate for architectural decision (primary-only vs. CRDT approach)

---

## Prerequisite 4: Single-Node Baselines (Weeks 4-5)

### Objective
Establish performance baselines for distributed overhead validation

### Current Problem
M14 plan claims:
- Single-node: 5ms P99 write, 10ms P99 read
- Distributed: <2x overhead

**These are assumptions, not measurements.**

### Tasks

**Task 4.1: Benchmarking Infrastructure** (Week 4, Days 1-2)
```rust
// engram-benchmarks/benches/baseline.rs
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn bench_store(c: &mut Criterion) {
    let runtime = tokio::runtime::Runtime::new().unwrap();
    let engram = runtime.block_on(async { Engram::new().await.unwrap() });

    c.bench_function("store_memory", |b| {
        b.to_async(&runtime).iter(|| async {
            engram.store(black_box(generate_memory())).await
        });
    });
}

criterion_group!(benches, bench_store, bench_recall, bench_spread);
criterion_main!(benches);
```

**Deliverable**: Criterion benchmark suite

**Task 4.2: Representative Workloads** (Week 4, Days 3-5)
Define workloads:
1. **Small**: 10K memories, single space
2. **Medium**: 100K memories, 5 spaces
3. **Large**: 1M memories, 10 spaces
4. **Mixed**: Store/recall/spread mix (70/20/10 ratio)

**Deliverable**: Workload generators

**Task 4.3: Baseline Measurements** (Week 5, Days 1-3)
Run benchmarks:
```bash
cargo bench --bench baseline

# Collect:
# - P50/P95/P99 latencies
# - Throughput (ops/sec)
# - Memory footprint (RSS)
# - CPU utilization
```

**Deliverable**: `docs/performance/single_node_baselines.md`

Example:
```markdown
## Single-Node Baselines (2025-10-31)

### Hardware
- CPU: AMD Ryzen 9 5950X (16C/32T)
- RAM: 64GB DDR4-3600
- Storage: NVMe SSD

### Store Operation
- P50: 2.3ms
- P95: 4.7ms
- P99: 8.1ms
- Throughput: 12,400 ops/sec

### Recall Operation
- P50: 5.1ms
- P95: 9.8ms
- P99: 15.2ms
- Throughput: 8,900 ops/sec

### Spread Operation (3 hops)
- P50: 12.4ms
- P95: 23.1ms
- P99: 34.7ms
- Throughput: 3,200 ops/sec
```

**Task 4.4: Distributed Targets** (Week 5, Day 4)
Define distributed targets based on baselines:
```markdown
## Distributed Performance Targets

### Intra-Partition (same node)
- Store: <2x baseline (P99 <16ms)
- Recall: <2x baseline (P99 <30ms)
- Spread: <2x baseline (P99 <70ms)

### Cross-Partition (network hop)
- Store: <3x baseline (P99 <25ms)
- Recall: <3x baseline (P99 <45ms)
- Spread: <5x baseline (P99 <175ms)

### Throughput
- 5 nodes: 5x baseline (62K ops/sec)
- Linear scaling to 16 nodes
```

**Deliverable**: `docs/performance/distributed_targets.md`

**Task 4.5: Continuous Benchmarking** (Week 5, Day 5)
Set up regression detection:
```yaml
# .github/workflows/benchmark.yml
name: Performance Regression

on:
  pull_request:
    branches: [main]

jobs:
  benchmark:
    runs-on: ubuntu-latest
    steps:
      - run: cargo bench
      - run: |
          if [ "$P99_LATENCY" -gt "$BASELINE_P99" ]; then
            echo "Performance regression detected"
            exit 1
          fi
```

**Deliverable**: CI integration for benchmark tracking

### Acceptance Criteria
- [ ] Criterion benchmarks for all core operations
- [ ] P50/P95/P99 documented for 3 workload sizes
- [ ] Distributed targets defined (<2x baseline intra-partition)
- [ ] CI tracks performance regressions

### Risk Mitigation
- If baselines worse than expected: Optimize before distributed work
- If hardware varies: Normalize to reference hardware

---

## Prerequisite 5: Production Soak Testing (Weeks 5-7)

### Objective
Validate 7+ days continuous operation, no memory leaks, stable performance

### Current Validation
M6: 1-hour soak test (insufficient for distributed systems)

### Tasks

**Task 5.1: Soak Test Design** (Week 5, Days 1-2)
```yaml
# soak_test.yaml
duration: 168h  # 7 days minimum
workload:
  - operation: store
    rate: 100 ops/sec
    spaces: 10
  - operation: recall
    rate: 50 ops/sec
    spaces: 10
  - operation: consolidation
    interval: 60s
    spaces: 10
  - operation: spread
    rate: 20 ops/sec
    max_hops: 5

monitoring:
  - memory_rss
  - memory_heap
  - cpu_percent
  - disk_io
  - consolidation_cadence
  - query_latency_p99

alerts:
  - memory_leak: "RSS increases >10% per day"
  - performance_degradation: "P99 latency increases >20%"
  - consolidation_failure: "Cadence drift >5s"
```

**Deliverable**: Soak test specification

**Task 5.2: Soak Test Infrastructure** (Week 5, Days 3-5)
```bash
# scripts/soak_test.sh
#!/bin/bash
set -euo pipefail

# Start Engram with observability
./target/release/engram-cli --config config/soak_test.toml &
ENGRAM_PID=$!

# Start Prometheus + Grafana
docker-compose -f docker/soak_test_observability.yml up -d

# Run workload generator
cargo run --release --bin workload_generator -- \
  --duration 168h \
  --config soak_test.yaml

# Continuous monitoring
while true; do
  # Check memory growth
  RSS=$(ps -p $ENGRAM_PID -o rss=)
  echo "$(date): RSS=${RSS}KB" >> tmp/soak_memory.log

  # Check for crashes
  if ! kill -0 $ENGRAM_PID 2>/dev/null; then
    echo "FATAL: Engram crashed during soak test"
    exit 1
  fi

  sleep 60
done
```

**Deliverable**: Automated soak test runner

**Task 5.3: Execute 7-Day Soak Test** (Weeks 6-7)
```bash
# Start soak test
./scripts/soak_test.sh

# Monitor:
# - Memory growth (valgrind, heaptrack)
# - Performance stability (P99 latency)
# - Consolidation cadence (60s ± drift)
# - Error rates (zero expected)
```

**Monitoring Plan**:
- **Hourly**: Check metrics dashboard
- **Daily**: Analyze memory growth trend
- **Day 3**: Mid-point validation (restart if issues)
- **Day 7**: Final validation, collect results

**Deliverable**: 7+ days continuous operation

**Task 5.4: Memory Leak Detection** (During soak)
```bash
# Run with valgrind (performance impact, shorter duration OK)
valgrind --leak-check=full --show-leak-kinds=all \
  ./target/release/engram-cli > tmp/valgrind_soak.log 2>&1

# Run with heaptrack (production-like performance)
heaptrack ./target/release/engram-cli
# Analyze after 24h:
heaptrack_gui heaptrack.engram.*.gz
```

**Acceptance**: Zero memory leaks detected

**Task 5.5: Soak Test Report** (Week 7, Day 5)
```markdown
# Soak Test Report (7-Day)

## Summary
- Duration: 168 hours (7 days)
- Workload: 100 stores/sec, 50 recalls/sec, 20 spreads/sec
- Spaces: 10 concurrent memory spaces
- Consolidation: 10,080 cycles (60s cadence)

## Results
- **Memory**: Stable at 1.2GB RSS (±50MB)
- **Performance**: P99 latency stable (<5% drift)
- **Consolidation**: 100% success rate, 60s ± 0.3s cadence
- **Errors**: 0 crashes, 0 data corruption events

## Memory Analysis
- Valgrind: 0 definite leaks, 0 possible leaks
- Heaptrack: Allocation rate stable (no growth trend)

## Conclusion
✅ PASS - System validated for continuous operation
```

**Deliverable**: `tmp/soak_test_report.md`

### Acceptance Criteria
- [ ] 168+ hours continuous operation (7+ days)
- [ ] Zero memory leaks (valgrind + heaptrack validation)
- [ ] Performance stable (P99 latency drift <5%)
- [ ] Consolidation stable (cadence 60s ± 5s)
- [ ] Zero crashes or data corruption

### Risk Mitigation
- If memory leak found: Must fix before distributed work
- If performance degrades: Investigate root cause (GC, fragmentation?)
- If crashes occur: Critical blocker, must resolve

**Escalation**: Any failure in 7-day soak test blocks M14 start

---

## Integration and Go/No-Go Decision (Week 8)

### Task 6.1: Prerequisites Validation

**Checklist**:
```markdown
## Prerequisites Met?

### Test Health
- [ ] 1,035/1,035 tests passing (100%)
- [ ] `make quality` zero warnings
- [ ] CI green on main branch

### M13 Completion
- [ ] 21/21 tasks complete
- [ ] Reconsolidation core implemented
- [ ] All M13 tests passing

### Consolidation Determinism
- [ ] Property tests: 1000 runs → 1 signature
- [ ] Performance: <10% regression
- [ ] Integration tests passing

### Single-Node Baselines
- [ ] P50/P95/P99 documented
- [ ] Distributed targets defined
- [ ] CI tracking regressions

### Production Soak Testing
- [ ] 168+ hours operation
- [ ] Zero memory leaks
- [ ] Performance stable
- [ ] Zero crashes

### Additional Checks
- [ ] Team capacity: 12-16 weeks available
- [ ] Infrastructure: Benchmark/monitoring tools ready
```

### Task 6.2: Go/No-Go Decision

**If ALL prerequisites met**:
→ **GO**: Proceed to M14 Phase 1 (Foundation)
- Start SWIM membership implementation
- Use realistic 12-16 week timeline
- Jepsen validation early (week 2)

**If ANY prerequisite NOT met**:
→ **NO-GO**: Defer M14, continue prerequisite work
- Identify blockers
- Create remediation plan
- Re-evaluate in 2 weeks

### Task 6.3: M14 Kickoff (If GO)

**Week 8, Day 5**: M14 Phase 1 starts
- Task 001: SWIM membership (7-10 days)
- Parallel: Test framework infrastructure
- Weekly reviews with prerequisite baseline comparisons

---

## Risk Management

### Critical Risks

| Risk | Probability | Mitigation | Contingency |
|------|-------------|------------|-------------|
| Determinism unsolvable | 10% | Option B: Primary-only consolidation | Defer M14, redesign gossip |
| Memory leak unfixable | 15% | Extended debugging, heaptrack profiling | Defer M14, Rust audit |
| M13 reconsolidation blocks | 20% | Parallel work on other prerequisites | Partial M13 completion acceptable |
| Timeline overrun | 40% | Weekly reviews, adjust scope | Extend timeline, maintain quality |

### Escalation Criteria

**Escalate immediately if**:
- Memory leak cannot be identified after 1 week debugging
- Determinism cannot be achieved after 3 weeks
- M13 reconsolidation reveals fundamental consolidation issues
- Soak test fails repeatedly (>2 failures)

---

## Success Metrics

### Prerequisites Complete
- [ ] All 5 prerequisites met (test health, M13, determinism, baselines, soak)
- [ ] Go/No-Go checklist: 100% complete
- [ ] Team capacity confirmed (12-16 weeks available)

### Ready for M14
- [ ] SWIM implementation can start immediately
- [ ] Baselines established for distributed validation
- [ ] Consolidation convergence mathematically provable
- [ ] Production stability demonstrated (7+ days)

### Timeline
- **Target**: 6-10 weeks to prerequisites complete
- **Best case**: 6 weeks (parallel execution, no blockers)
- **Worst case**: 10 weeks (serial execution, minor blockers)
- **Escalation threshold**: >12 weeks (architectural issues)

---

## Appendix: Weekly Progress Template

```markdown
# Prerequisites Progress: Week N

## Completed This Week
- [ ] Test health: X/1,035 passing
- [ ] M13: Y/21 complete
- [ ] Determinism: Z/1000 runs convergent
- [ ] Baselines: A/B operations benchmarked
- [ ] Soak test: C/168 hours complete

## Blockers
- (List any blockers)

## Next Week Plan
- (Tasks for next week)

## Go/No-Go Forecast
- On track: YES/NO
- Estimated completion: Week X
```

Use this template for weekly reviews to track progress toward M14 readiness.

---

**Document Status**: Execution Plan - Ready for Approval
**Owner**: TBD (assign after decision to execute prerequisites)
**Review Cadence**: Weekly
**Success Criteria**: All prerequisites met, M14 Phase 1 ready to start
