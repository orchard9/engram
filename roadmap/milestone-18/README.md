# Milestone 18: Performance & Correctness Infrastructure

**Status**: Reorganized 2025-11-12
**Duration**: 10-14 weeks (parallelizable)
**Prerequisites**: M17 at 40% (Phase 1), M17 at 60% (Phases 2-3)

---

## Overview

Milestone 18 establishes comprehensive performance testing and correctness validation for Engram's dual memory architecture (M17). This milestone answers three critical questions before production deployment:

1. **Does it scale?** (Phase 1: Performance Infrastructure, Tasks 001-016)
2. **Is it correct?** (Phase 2: Graph Correctness, Tasks 017-020)
3. **Is it plausible?** (Phase 3: Cognitive Validation, Tasks 021-028)

**Note**: Production deployment workflows moved to **M18.1** (separate milestone, Tasks 101-115)

---

## Milestone Structure

```
M18: Performance & Correctness Infrastructure (001-028)
├── Phase 1: Performance Infrastructure (001-016) - 10 weeks
│   ├── Production Load Testing (001-003)
│   ├── Scalability Validation (004-006)
│   ├── Concurrency & Contention (007-009)
│   ├── Hardware-Specific Testing (010-011)
│   ├── Cache Efficiency (012-013)
│   └── Regression Prevention (014-016)
├── Phase 2: Graph Correctness (017-020) - 3 weeks
│   ├── Concurrent Correctness (017) - loom testing
│   ├── Invariant Property Testing (018) - proptest
│   ├── Probabilistic Semantics (019) - Bayesian validation
│   └── Biological Plausibility (020) - cognitive phenomena
└── Phase 3: Cognitive Validation (021-028) - 5 weeks
    ├── Semantic Priming (021)
    ├── Anderson Fan Effect (022)
    ├── Consolidation Timeline (023)
    ├── Retrograde Amnesia (024)
    ├── DRM False Memory (025)
    ├── Spacing Effect (026)
    ├── Pattern Completion (027)
    └── Reconsolidation Dynamics (028)

M18.1: Production Readiness (101-115) - 6 weeks [Separate Milestone]
└── End-to-end workflows, chaos engineering, operational procedures
```

---

## Quick Reference

### Key Documents
- **MASTER_TASK_INDEX.md**: Complete task catalog with unified numbering
- **PERFORMANCE_TESTING_SUMMARY.md**: Phase 1 performance infrastructure details
- **GRAPH_VALIDATION_SUMMARY.md**: Phase 2 graph correctness details
- **COGNITIVE_VALIDATION_TASKS.md**: Phase 3 cognitive phenomena details
- **M18_PRODUCTION_READINESS_PLAN.md**: M18.1 production deployment plan

### Success Metrics at a Glance
| Category | Key Metric | Target | Phase |
|----------|------------|--------|-------|
| **Stability** | 24h soak test | <1% drift, zero leaks | 1 |
| **Scale** | 10M nodes P99 | <15ms | 1 |
| **Concurrency** | 32-core efficiency | >80% | 1 |
| **Correctness** | Loom tests | 100% interleaving coverage | 2 |
| **Invariants** | Proptest | 10K cases pass | 2 |
| **Probabilistic** | Bayesian validation | Within numerical precision | 2 |
| **Cognitive** | Psychology correlation | r > 0.75 for all 8 phenomena | 3 |

---

## Phase 1: Performance Infrastructure (001-016)

**Goal**: Validate system scales to production workloads with acceptable performance

**Duration**: 10 weeks (parallelizable)

**Prerequisites**: M17 at 40% (basic dual memory operational)

### Production Load Testing (Week 1-2)
- **001**: Production Workload Simulation - Diurnal cycles, bursts, temporal correlation
- **002**: Extended Soak Testing - 24h+ stability, leak/drift detection
- **003**: Burst Traffic Stress - 10x spike handling, <5s recovery

### Scalability Validation (Week 3-4)
- **004**: Dataset Scaling Tests - 100K → 10M nodes with performance curves
- **005**: Throughput Scaling Tests - Ramp-to-breaking-point capacity discovery
- **006**: Latency Tail Analysis - P99.9/P99.99 characterization

### Concurrency & Contention (Week 5-6)
- **007**: Thread Scalability - 1 → 128 threads, parallel efficiency
- **008**: Lock-Free Contention - DashMap hot-spot scenarios
- **009**: Multi-Tenant Isolation - Cross-space interference (<1% target)

### Hardware-Specific Testing (Week 7)
- **010**: NUMA Cross-Socket - Multi-socket validation [OPTIONAL - Tier 3]
- **011**: CPU Architecture Diversity - ARM/x86 SIMD validation

### Cache Efficiency (Week 8)
- **012**: Cache Alignment - False sharing detection (<1%)
- **013**: Prefetching Effectiveness - Software prefetch (>70% coverage)

### Regression Prevention (Week 9-10)
- **014**: CI/CD Performance Gates - Block merges on >5% regression
- **015**: Competitive Baseline Tracking - Neo4j/Qdrant weekly comparison
- **016**: Performance Dashboard - Grafana with anomaly detection

**Key Deliverables**:
- Production workload generator with realistic patterns
- CI/CD performance gates integrated into git hooks
- Grafana dashboard for real-time monitoring
- Scalability characterization: 100K, 1M, 10M node performance curves

---

## Phase 2: Graph Correctness (017-020)

**Goal**: Prove graph engine operations are correct under all conditions

**Duration**: 3 weeks

**Prerequisites**: M17 at 60% (spreading activation operational)

### Tasks

**017: Graph Concurrent Correctness** (3-4 days)
- **Tool**: loom 0.7+ (systematic concurrency testing)
- **Validates**: Atomic operations correct under all thread interleavings
- **Tests**: 6+ loom tests exploring 2-3 thread state spaces
- **Output**: Proof of lock-freedom, memory ordering correctness

**018: Graph Invariant Property Testing** (3 days)
- **Tool**: proptest 1.5+ (property-based testing)
- **Validates**: Bounds, conservation laws, connectivity preservation
- **Tests**: 11+ properties with 10,000 cases each
- **Output**: Invariant violations caught before production

**019: Probabilistic Semantics Validation** (3 days)
- **Tool**: approx 0.5+ (floating-point comparisons)
- **Validates**: Confidence propagation follows probability theory
- **Tests**: Bayesian updates, numerical stability, log-probabilities
- **Output**: Mathematically sound uncertainty quantification

**020: Biological Plausibility Validation** (4 days)
- **Validates**: Fan effect, priming, forgetting curve, concept formation
- **Tests**: Empirical psychology data reproduction
- **Output**: Confirmation that biological inspiration translates to function

**Key Deliverables**:
- Loom test suite proving concurrent correctness
- Proptest generators covering graph operation space
- Probabilistic validation against Bayesian theory
- Cognitive phenomena validation against literature

---

## Phase 3: Cognitive Validation (021-028)

**Goal**: Validate dual memory reproduces empirical cognitive psychology phenomena

**Duration**: 5 weeks

**Prerequisites**: M17 at 60% (consolidation operational)

### Core Phenomena (Week 1-2, Tasks 021-024)

**021: Semantic Priming** (5 days)
- **Phenomenon**: Neely 1977 - 40-80ms priming at 250ms SOA
- **Target**: r > 0.80 correlation with published data
- **Validates**: Spreading activation in concept network

**022: Anderson Fan Effect** (4 days)
- **Phenomenon**: Anderson 1974 - 70ms RT increase per association
- **Target**: r > 0.85 correlation, linear r² > 0.70
- **Validates**: M17 Task 007 fan effect spreading

**023: Consolidation Timeline** (5 days)
- **Phenomenon**: Takashima 2006 - Hippocampal-neocortical transfer over 90 days
- **Target**: r > 0.75 correlation for both trajectories
- **Validates**: Episodic-to-semantic transformation

**024: Retrograde Amnesia Gradient** (4 days)
- **Phenomenon**: Ribot's Law - Recent memories more vulnerable
- **Target**: r > 0.80 correlation, exponential decay
- **Validates**: Consolidation protects against disruption

### Emergent Properties (Week 3-4, Tasks 025-028)

**025: DRM False Memory** (4 days)
- **Phenomenon**: Roediger & McDermott 1995 - 40-55% false recall
- **Target**: r > 0.70 correlation
- **Validates**: Schema-based reconstruction

**026: Spacing Effect** (3 days)
- **Phenomenon**: Cepeda 2006 - 15-30% retention advantage
- **Target**: Inverted-U function, overnight consolidation boost
- **Validates**: Consolidation-rehearsal interaction

**027: Pattern Completion** (3 days)
- **Phenomenon**: Nakazawa 2002 - CA3 attractor dynamics
- **Target**: 60-70% threshold, >80% above-threshold accuracy
- **Validates**: M17 Task 004 coherence threshold

**028: Reconsolidation Dynamics** (4 days)
- **Phenomenon**: Lee 2008 - 3-6 hour reactivation window
- **Target**: 10-20% enhancement, 30-50% disruption
- **Validates**: Memory updating via retrieval

**Key Deliverables**:
- 8 cognitive validation tests replicating classic studies
- Correlation analysis showing r > 0.75 for all phenomena
- Proof that biological inspiration creates functional properties
- CI/CD integration for continuous cognitive validation

---

## M18.1: Production Readiness (Separate Milestone)

**Status**: Separate milestone document (M18_PRODUCTION_READINESS_PLAN.md)

**Tasks**: 101-115 (renumbered from original 001-005 to avoid conflicts)

**Phases**:
- End-to-end workflows (101-103)
- Chaos engineering (104-108)
- Operational readiness (109-111)
- Performance SLOs (112-114)
- API compatibility (115)

**Prerequisites**: M17 at 60%, M18 Phase 1 at 50%

**Duration**: 6 weeks

---

## Dependencies

### M17 Prerequisites by Phase

**Phase 1 (Performance Infrastructure)**:
- Start: M17 at 40% (basic dual memory operational)
- Required: M17 Tasks 001-002 (dual memory types, storage)
- Blockers: None - can start immediately

**Phase 2 (Graph Correctness)**:
- Start: M17 at 60% (spreading activation operational)
- Required: M17 Tasks 001-007 (through fan effect)
- Blockers: Phase 1 Tasks 001-003 (basic performance baseline)

**Phase 3 (Cognitive Validation)**:
- Start: M17 at 60% (consolidation operational)
- Required: M17 Tasks 001-009 (through blended recall)
- Blockers: Phase 2 complete (correctness validated first)

### External Dependencies

**Infrastructure**:
- Prometheus/Grafana monitoring stack
- Load test tool with chaos injection support
- Diagnostic scripts (engram_diagnostics.sh)

**Tools**:
- loom 0.7+ (Phase 2, Task 017)
- proptest 1.5+ (Phase 2, Task 018)
- approx 0.5+ (Phase 2, Task 019)
- statrs (Phase 3, statistical analysis)

---

## Hardware Requirements

### Tier 1: Minimum (All Core Tests)
- **CPU**: 8+ cores (Apple M1/M2, AMD Ryzen 5000+, Intel 12th gen+)
- **RAM**: 16GB minimum (32GB recommended)
- **Storage**: SSD required
- **OS**: macOS 12+, Linux 5.10+
- **Tests**: 001-009, 011-028

### Tier 2: Production Representative (Scalability)
- **CPU**: 16-32 cores
- **RAM**: 64GB minimum
- **Storage**: NVMe SSD
- **Tests**: 004 (10M scale), 005 (high throughput)

### Tier 3: NUMA Systems (Optional Hardware-Specific)
- **CPU**: Multi-socket (2+ NUMA nodes)
- **RAM**: 128GB+ distributed across sockets
- **Storage**: NVMe with NUMA awareness
- **OS**: Linux 5.10+ with numactl
- **Tests**: 010 only

---

## Timeline & Parallelization

### Sequential Critical Path (Minimum Duration)
1. **Week 1-2**: Tasks 001-003 (production load baseline)
2. **Week 3-4**: Tasks 004-006 (scalability characterization)
3. **Week 5-10**: Tasks 007-016 (concurrency, cache, CI/CD)
4. **Week 11-13**: Tasks 017-020 (graph correctness)
5. **Week 14-18**: Tasks 021-028 (cognitive validation)

**Total**: 18 weeks sequential

### Parallel Execution (Optimized Duration)
- **Phase 1**: 10 weeks (many tasks parallelizable)
- **Phase 2**: 3 weeks (overlap with Phase 1 end)
- **Phase 3**: 5 weeks (overlap with Phase 2)

**Total**: 10-14 weeks with parallelization

### Start Conditions
- **Phase 1**: Start when M17 reaches 40%
- **Phase 2**: Start when M17 reaches 60%
- **Phase 3**: Start when M17 reaches 60%

---

## Success Metrics

### Phase 1: Performance Infrastructure
- [ ] 24h soak test: Zero leaks, <1% latency drift
- [ ] 10M nodes: P99 <15ms, throughput >500 ops/s
- [ ] 32-core efficiency: >80%
- [ ] Multi-tenant isolation: <1% cross-space interference
- [ ] CI/CD gates: Block >5% regressions
- [ ] Dashboard: <5min time-to-root-cause

### Phase 2: Graph Correctness
- [ ] Loom tests: 100% interleaving coverage for 2-3 threads
- [ ] Property tests: 10,000 cases per property, all pass
- [ ] Probabilistic: Bayesian updates within numerical precision
- [ ] Biological: Fan effect, priming, forgetting curve match literature

### Phase 3: Cognitive Validation
- [ ] Correlation: r > 0.75 for all 8 phenomena
- [ ] Effect sizes: Within ±0.3 Cohen's d of published values
- [ ] Temporal accuracy: Timing windows match human data ±20%
- [ ] Reproducibility: <5% variance across runs

---

## Integration with Existing M17 Framework

All performance tasks extend (not replace) the M17 performance framework:

```bash
# Existing M17 workflow (60s regression tests)
./scripts/m17_performance_check.sh <task> before
./scripts/m17_performance_check.sh <task> after
./scripts/compare_m17_performance.sh <task>

# New M18 extensions
./scripts/run_scaling_tests.sh          # Task 004
./scripts/run_soak_test.sh 24           # Task 002
./scripts/ci_performance_gate.sh        # Task 014
```

**No breaking changes** - M18 is additive only.

---

## Competitive Positioning

M18 validates Engram's market differentiation:

**Target Competitive Positioning**:
- **vs Neo4j**: 46% faster (27.96ms → <15ms P99 traversal)
- **vs Qdrant**: 9-17% faster (22-24ms → <20ms P99 ANN search)
- **vs Both**: Unique hybrid capability (vector+graph+temporal in single system)

**Validation**: Task 015 automated weekly tracking ensures maintaining edge.

---

## Risk Mitigation

### Hardware Limitations
- **Risk**: No Tier 2/3 hardware for advanced tests
- **Mitigation**: Task 010 marked optional, graceful degradation

### Test Duration
- **Risk**: Long tests block development
- **Mitigation**: Soak tests run nightly, CI uses 5-min fast tests

### False Positives
- **Risk**: Performance gates block valid changes
- **Mitigation**: Statistical significance tests, conservative thresholds

### Platform Diversity
- **Risk**: Different results on ARM vs x86
- **Mitigation**: Task 011 validates equivalence, documents variance

---

## Next Steps

### Immediate Actions
1. **Review task organization**: Confirm new numbering scheme (001-028, 101-115)
2. **Validate M17 progress**: Check if M17 at 40% to start Phase 1
3. **Hardware assessment**: Confirm Tier 1/2/3 availability
4. **Tool installation**: Install loom, proptest, approx for Phase 2

### Phase 1 Start
1. Begin with Task 001 (Production Workload Simulation)
2. Establish baseline performance metrics
3. Create production traffic patterns
4. Validate M17 performance framework integration

### Long-term
- Tasks 014-016 provide ongoing value after M18 (CI/CD gates, monitoring)
- Cognitive validation (021-028) becomes part of regression suite
- Competitive tracking (015) informs quarterly strategy reviews

---

## References

- **M17 Overview**: `roadmap/milestone-17/000_milestone_overview_dual_memory.md`
- **M17 Performance Baseline**: `roadmap/milestone-17/PERFORMANCE_BASELINE.md`
- **M17.1 Competitive Framework**: `roadmap/milestone-17.1/README.md`
- **M18.1 Production Plan**: `roadmap/milestone-18/M18_PRODUCTION_READINESS_PLAN.md`
- **Vision**: `vision.md`
- **Operations Docs**: `docs/operations/`

---

**Document Status**: Active
**Last Updated**: 2025-11-12
**Next Review**: After M17 reaches 60% completion
