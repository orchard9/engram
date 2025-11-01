# Engram - Next Steps

**Date**: 2025-11-01
**Branch**: dev (commit 8e9c141)
**Tag**: baseline-2025-11-01
**Status**: Prerequisites Weeks 1-4 COMPLETE, ready for Week 5-10

---

## Summary of Completed Work

**Weeks 1-4 Status**: COMPLETE (finished early!)

- 100% test health (1,035/1,035 passing)
- M13 milestone 100% complete (6/6 tasks)
- Consolidation determinism fixed (CRITICAL M14 blocker resolved)
- Zero clippy warnings
- 12 parallel AI agents completed ~200 agent-hours in single session

**Baseline Tag**: `baseline-2025-11-01` pushed to remote

---

## Next Steps Overview

### Phase 1: Week 5-7 - Single-Node Performance Baselines (2-3 weeks)
### Phase 2: Week 7-10 - Production Soak Test (7+ days continuous)
### Phase 3: Week 8 - Go/No-Go Decision for M14

---

## Week 5-7: Single-Node Performance Baselines (NEXT)

**Objective**: Establish comprehensive performance baselines for comparing against future distributed architecture (M14).

**Environment**: Build machine with production-like hardware configuration

### Setup (Day 1)

1. **Deploy baseline tag to build machine**:
   ```bash
   git clone https://github.com/orchard9/engram.git
   cd engram
   git checkout baseline-2025-11-01
   cargo build --release --features "default,cognitive_tracing,pattern_completion"
   ```

2. **Configure for production workload**:
   - Enable all relevant features
   - Set production-level configuration (buffer sizes, thread pools, etc.)
   - Configure observability stack (Prometheus, Grafana)

3. **Create benchmark workloads**:
   - Synthetic workload generator (various query patterns)
   - Realistic workload based on expected production usage
   - Stress test scenarios

### Baseline Measurements (Day 2-10)

**Write Operations** (Day 2-3):
- Latency: P50, P95, P99, P99.9
- Throughput: Measure at 1K, 10K, 50K, 100K ops/sec
- Memory usage: Per-episode overhead, heap growth
- CPU utilization: Core usage patterns

**Read Operations - Intra-Partition** (Day 4-5):
- Query latency by complexity (simple recall, multi-hop spreading, pattern completion)
- Throughput under concurrent queries
- Cache hit rates and memory access patterns
- Activation spreading latency by hop count

**Read Operations - Cross-Partition Simulation** (Day 6-7):
- Simulate cross-partition queries (for M14 comparison)
- Multi-space queries with artificial latency injection
- Scatter-gather query patterns
- Confidence degradation under partial node availability

**Consolidation Performance** (Day 8-9):
- Pattern detection latency
- Memory consolidation throughput
- Semantic memory formation rate
- Background consolidation overhead

**Memory and Resource Profiling** (Day 10):
- Heap profiling with Valgrind/massif
- CPU profiling with perf/flamegraph
- NUMA memory access patterns
- Lock contention analysis

### Deliverables

1. **Baseline Performance Report** (`docs/operations/baseline-2025-11-01.md`):
   - All measurements with statistical confidence intervals
   - Hardware specifications
   - Configuration parameters
   - Comparison against target SLOs

2. **Performance Regression Test Suite**:
   - Automated benchmarks for CI
   - Alert thresholds for performance degradation
   - Baseline data for future comparison

3. **Optimization Opportunities**:
   - Identified bottlenecks
   - Low-hanging fruit optimizations
   - Recommendations for M14 architecture

---

## Week 7-10: Production Soak Test (7+ days)

**Objective**: Validate single-node stability and production readiness under sustained load.

**Duration**: Minimum 7 days continuous operation (168 hours)

### Setup (Day 1)

1. **Deploy to production-like environment**:
   - Dedicated machine or VM
   - Production OS configuration
   - Network conditions matching expected deployment
   - Full observability stack

2. **Configure continuous workload**:
   - Realistic query patterns (70% reads, 30% writes)
   - Variable load profile (peak hours, off-peak)
   - Background consolidation processes
   - Periodic memory cleanup

3. **Set up monitoring and alerting**:
   - Memory leak detection
   - Crash/panic alerts
   - Performance degradation alerts
   - Resource exhaustion warnings

### Monitoring (Day 2-8)

**Daily Checks**:
- Memory usage trend (detect leaks)
- CPU utilization stability
- Query latency drift
- Error rate and crash count
- Disk usage growth
- Log file analysis

**Metrics to Track**:
- Heap size over time
- RSS/VSS memory
- Thread count stability
- File descriptor usage
- Network connection count
- Query success rate
- P99 latency stability

**Failure Scenarios to Test** (scheduled during soak):
- Process restart (graceful shutdown)
- SIGTERM/SIGKILL handling
- Disk full condition
- Memory pressure
- High query load spikes
- Invalid query handling

### Analysis (Day 9-10)

1. **Performance Degradation Analysis**:
   - Compare Day 1 vs Day 7 metrics
   - Identify any drift or degradation
   - Root cause performance issues

2. **Memory Leak Investigation**:
   - Heap profiling snapshots
   - Memory growth trend analysis
   - Reference leak detection

3. **Crash/Panic Analysis**:
   - Core dump analysis
   - Panic log review
   - Bug fixes for any crashes

4. **Operational Learnings**:
   - Common failure modes
   - Recovery procedures
   - Monitoring gaps
   - Configuration tuning needed

### Deliverables

1. **Soak Test Report** (`docs/operations/soak-test-2025-11.md`):
   - 7-day stability summary
   - Performance metrics over time
   - Incidents and resolutions
   - Lessons learned

2. **Production Runbooks** (update existing):
   - Deployment procedures
   - Operational procedures (start, stop, restart)
   - Troubleshooting guides
   - Incident response playbooks

3. **Bug Fixes and Improvements**:
   - Any critical bugs found during soak
   - Performance optimizations
   - Configuration improvements
   - Monitoring enhancements

---

## Week 8: Go/No-Go Decision for M14

**Objective**: Make evidence-based decision on whether to proceed with M14 distributed architecture.

### Go/No-Go Checklist

**Prerequisites Review**:
- [ ] Consolidation determinism proven (property tests pass)
- [ ] Performance baselines established (comprehensive report exists)
- [ ] M13 100% complete (all 6 tasks done)
- [ ] 7-day soak test passed (no critical issues)
- [ ] 100% test health maintained (1,035/1,035 passing)

**Performance Targets Met**:
- [ ] Write latency: P99 < 10ms
- [ ] Read latency: P99 < 20ms (intra-partition)
- [ ] Throughput: 10K+ ops/sec sustained
- [ ] Memory leak: None detected over 7 days
- [ ] Crashes: Zero unhandled panics

**Operational Readiness**:
- [ ] Monitoring stack validated
- [ ] Runbooks complete and tested
- [ ] Deployment automation working
- [ ] Backup/restore procedures verified
- [ ] Performance regression tests integrated

**M14 Feasibility Assessment**:
- [ ] Cross-partition query latency targets achievable (3-7x baseline acceptable)
- [ ] Gossip convergence time reasonable (<60s)
- [ ] SWIM membership overhead acceptable
- [ ] Replication lag targets feasible (<1s)
- [ ] Distributed consolidation determinism validated

### Decision Outcomes

**GO (75-85% probability)**:
- All prerequisites met
- Performance baselines excellent
- No critical blockers identified
- Proceed to M14 Phase 1: SWIM Membership (Task 001)
- Timeline: 12-16 weeks to production-ready distributed Engram

**NO-GO (Remediation needed)**:
- Critical issues found during soak test
- Performance targets not met
- Operational gaps identified
- Action: Address blockers, re-evaluate in 2 weeks

**DEFER (Alternative path)**:
- Single-node performance exceptional
- M14 complexity vs value reassessed
- Action: Continue single-node optimization, defer M14 indefinitely

---

## Baseline Measurement Procedures

### Write Latency Benchmark

```bash
# Target: P99 < 10ms
cargo bench --bench write_latency -- --warm-up-time 10 --measurement-time 60

# Vary batch sizes: 1, 10, 100, 1000
# Measure P50, P90, P95, P99, P99.9
# Track memory allocation per write
```

### Read Latency Benchmark

```bash
# Target: P99 < 20ms (intra-partition)
cargo bench --bench read_latency -- --warm-up-time 10 --measurement-time 60

# Query types:
# - Simple recall (by ID)
# - Semantic query (embedding similarity)
# - Multi-hop spreading (1-hop, 2-hop, 3-hop)
# - Pattern completion (COMPLETE query)
# - IMAGINE query (confidence-based completion)
```

### Throughput Benchmark

```bash
# Target: 10K+ ops/sec sustained
# Ramp test: 1K -> 10K -> 50K -> 100K ops/sec
cargo bench --bench throughput -- --duration 300

# Measure:
# - Max sustainable throughput
# - Latency degradation under load
# - Resource utilization (CPU, memory, I/O)
```

### Memory Profiling

```bash
# Heap profiling
MALLOC_CONF=prof:true cargo build --release
./target/release/engram-cli start --profile-heap

# After 1 hour, 24 hours, 7 days:
jeprof --svg ./engram-cli prof.out > heap_profile.svg

# Look for:
# - Memory growth trend
# - Large allocations
# - Reference leaks
```

### Consolidation Performance

```bash
# Measure consolidation overhead
cargo bench --bench consolidation_overhead

# Metrics:
# - Pattern detection latency
# - Clustering algorithm performance
# - Memory usage during consolidation
# - Background consolidation CPU overhead
```

---

## Performance Metrics to Collect

### Latency Metrics

| Metric | Target | Baseline | Notes |
|--------|--------|----------|-------|
| Write P50 | < 2ms | TBD | Single episode insert |
| Write P99 | < 10ms | TBD | 99th percentile |
| Read P50 (simple) | < 5ms | TBD | Direct ID lookup |
| Read P99 (simple) | < 20ms | TBD | 99th percentile |
| Read P50 (spreading) | < 15ms | TBD | 3-hop activation |
| Read P99 (spreading) | < 50ms | TBD | 99th percentile |
| Consolidation P50 | < 100ms | TBD | Pattern detection |
| Consolidation P99 | < 500ms | TBD | Large cluster merging |

### Throughput Metrics

| Metric | Target | Baseline | Notes |
|--------|--------|----------|-------|
| Write throughput | 10K ops/sec | TBD | Sustained rate |
| Read throughput | 10K ops/sec | TBD | Concurrent queries |
| Mixed workload | 10K ops/sec | TBD | 70% read, 30% write |
| Peak throughput | 50K ops/sec | TBD | Burst capacity |

### Resource Metrics

| Metric | Target | Baseline | Notes |
|--------|--------|----------|-------|
| Memory per episode | < 10 KB | TBD | Average overhead |
| Heap growth rate | < 1 MB/hour | TBD | Over 7 days |
| CPU utilization | < 50% | TBD | At 10K ops/sec |
| Thread count | Stable | TBD | No thread leaks |
| File descriptors | Stable | TBD | No FD leaks |

---

## M14 Comparison Targets

Once baselines established, use these for M14 distributed architecture comparison:

### Intra-Partition Queries (Same Node)
- **Target**: < 2x single-node latency
- **Example**: If single-node P99 = 20ms, distributed intra-partition P99 should be < 40ms

### Cross-Partition Queries (Multi-Node)
- **Target**: < 7x single-node latency (realistic, not < 2x as in original M14 plan)
- **Example**: If single-node P99 = 20ms, distributed cross-partition P99 should be < 140ms
- **Rationale**: Network RTT dominates (3 hops × 5ms RTT = 15ms + processing)

### Throughput Scaling
- **Target**: 0.6×N scaling (60% efficiency at N nodes)
- **Example**: 5-node cluster should achieve 30K ops/sec (5 × 10K × 0.6)
- **Rationale**: Overhead from replication, gossip, coordination

### Consistency Metrics
- **Gossip convergence**: < 60s for semantic memories
- **Replication lag**: < 1s for episodic memories
- **Consolidation convergence**: Same patterns across all nodes within 5 minutes

---

## Success Criteria Summary

### Week 5-7 Success Criteria

- [ ] All baseline measurements completed
- [ ] Performance targets met or exceeded
- [ ] Baseline report written
- [ ] Regression test suite created
- [ ] Optimization recommendations documented

### Week 7-10 Success Criteria

- [ ] 7-day soak test completed
- [ ] Zero unhandled panics/crashes
- [ ] No memory leaks detected
- [ ] Performance stable over time
- [ ] Operational runbooks validated
- [ ] Soak test report written

### Week 8 Decision Criteria

- [ ] All prerequisites met (5/5)
- [ ] Go/No-Go decision made
- [ ] If GO: M14 Phase 1 kickoff planned
- [ ] If NO-GO: Remediation plan created
- [ ] Stakeholders informed

---

## Risk Mitigation

### Risk: Performance targets not met

**Mitigation**:
- Identify bottlenecks early via profiling
- Low-hanging optimizations (caching, batching)
- Adjust targets if necessary (with justification)
- Document why targets are challenging

### Risk: Memory leaks detected during soak

**Mitigation**:
- Heap profiling every 24 hours
- Reference cycle detection
- Fix critical leaks immediately
- Re-run soak test after fixes

### Risk: Critical bugs found during soak

**Mitigation**:
- Comprehensive logging for debugging
- Core dump analysis
- Fix bugs, re-run affected tests
- Extend soak test if needed

### Risk: Build machine unavailable

**Mitigation**:
- Use cloud VM as backup (AWS c5.4xlarge or similar)
- Document exact hardware specs
- Ensure reproducible environment

---

## Timeline Summary

| Phase | Duration | Dates (Estimated) | Deliverable |
|-------|----------|-------------------|-------------|
| **Week 5-7: Baselines** | 2-3 weeks | Nov 4-24 | Baseline report + regression tests |
| **Week 7-10: Soak Test** | 7-10 days | Nov 18-28 | Soak test report + runbooks |
| **Week 8: Decision** | 1 day | Nov 25 | Go/No-Go decision |
| **M14 Phase 1** (if GO) | 3-4 weeks | Dec 1-28 | SWIM membership protocol |

**Total Timeline to M14 Start**: 3-4 weeks from today (early December 2025)

---

## Contact and Escalation

**For Questions**:
- Review BASELINE_PREPARATION_COMPLETE.md for context
- Review roadmap/milestone-14/M14_PREPARATION_COMPLETE.md for M14 details
- Review roadmap/milestone-14/PREREQUISITE_EXECUTION_PLAN.md for original plan

**Blockers or Issues**:
- Document in NEXT_STEPS.md (this file)
- Create issue in GitHub with "M14-prerequisite" label
- Escalate critical blockers immediately

---

**Status**: READY FOR WEEK 5-7 BASELINES
**Next Action**: Deploy baseline-2025-11-01 tag to build machine
**Owner**: TBD
**Start Date**: 2025-11-04 (Monday)
