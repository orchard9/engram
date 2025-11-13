# Milestone 18: Master Task Index

**Status**: Reorganized 2025-11-12
**Purpose**: Unified task numbering across all M18 scopes to resolve conflicts

## Milestone Structure

- **M18: Performance & Correctness Infrastructure** (001-028) - This milestone
- **M18.1: Production Readiness & Deployment** (101-115) - Separate milestone, starts at M17 60%

---

## M18: Performance & Correctness Infrastructure (001-028)

### Phase 1: Performance Infrastructure (001-016)

#### Production Load Testing (Week 1-2)
| Task | Name | File | Status | Duration |
|------|------|------|--------|----------|
| 001 | Production Workload Simulation | 001_production_workload_simulation_pending.md | Pending | 3-4 days |
| 002 | Extended Soak Testing | 002_extended_soak_testing_pending.md | Pending | 4-5 days |
| 003 | Burst Traffic Stress | 003_burst_traffic_stress_pending.md | Pending | 2-3 days |

#### Scalability Validation (Week 3-4)
| Task | Name | File | Status | Duration |
|------|------|------|--------|----------|
| 004 | Dataset Scaling Tests (100K→10M) | 004_dataset_scaling_tests_pending.md | Pending | 5-6 days |
| 005 | Throughput Scaling Tests | 005_throughput_scaling_tests_pending.md | Pending | 3-4 days |
| 006 | Latency Tail Analysis | 006_latency_tail_analysis_pending.md | Pending | 3-4 days |

#### Concurrency & Contention (Week 5-6)
| Task | Name | File | Status | Duration |
|------|------|------|--------|----------|
| 007 | Thread Scalability Benchmarking | 007_thread_scalability_benchmarks_pending.md | Pending | 4-5 days |
| 008 | Lock-Free Contention Testing | 008_lockfree_contention_testing_pending.md | Pending | 3-4 days |
| 009 | Multi-Tenant Isolation Testing | 009_multitenant_isolation_testing_pending.md | Pending | 3-4 days |

#### Hardware-Specific Testing (Week 7)
| Task | Name | File | Status | Duration |
|------|------|------|--------|----------|
| 010 | NUMA Cross-Socket Performance | 010_numa_cross_socket_performance_pending.md | Pending | 5-6 days |
| 011 | CPU Architecture Diversity | 011_cpu_architecture_diversity_pending.md | Pending | 3-4 days |

#### Cache Efficiency (Week 8)
| Task | Name | File | Status | Duration |
|------|------|------|--------|----------|
| 012 | Cache-Line Alignment Validation | 012_cache_alignment_validation_pending.md | Pending | 4-5 days |
| 013 | Prefetching Effectiveness | 013_prefetching_effectiveness_pending.md | Pending | 3-4 days |

#### Regression Prevention (Week 9-10)
| Task | Name | File | Status | Duration |
|------|------|------|--------|----------|
| 014 | CI/CD Performance Gate Integration | 014_cicd_performance_gates_pending.md | Pending | 4-5 days |
| 015 | Competitive Baseline Tracking | 015_competitive_baseline_tracking_pending.md | Pending | 3-4 days |
| 016 | Performance Dashboard | 016_performance_dashboard_pending.md | Pending | 4-5 days |

**Phase 1 Total**: 48-66 days (can parallelize to ~10 weeks)

---

### Phase 2: Graph Engine Correctness (017-020)

| Task | Name | File | Status | Duration | Priority |
|------|------|------|--------|----------|----------|
| 017 | Graph Concurrent Correctness | 017_graph_concurrent_correctness_pending.md | Pending | 3-4 days | High |
| 018 | Graph Invariant Property Testing | 018_graph_invariant_validation_pending.md | Pending | 3 days | High |
| 019 | Probabilistic Semantics Validation | 019_probabilistic_semantics_validation_pending.md | Pending | 3 days | High |
| 020 | Biological Plausibility Validation | 020_biological_plausibility_validation_pending.md | Pending | 4 days | High |

**Phase 2 Total**: 13-14 days (2.5-3 weeks)

**Description**: Low-level correctness validation using loom (concurrency), proptest (invariants), Bayesian validation (probabilistic), and cognitive phenomena (bio-plausibility).

---

### Phase 3: Cognitive Dynamics Validation (021-028)

| Task | Name | File | Status | Duration | Priority |
|------|------|------|--------|----------|----------|
| 021 | Semantic Priming Validation | 021_semantic_priming_validation_pending.md | Pending | 5 days | High |
| 022 | Anderson Fan Effect Validation | 022_anderson_fan_effect_validation_pending.md | Pending | 4 days | High |
| 023 | Consolidation Timeline Validation | 023_consolidation_timeline_validation_pending.md | Pending | 5 days | High |
| 024 | Retrograde Amnesia Gradient | 024_retrograde_amnesia_gradient_validation_pending.md | Pending | 4 days | High |
| 025 | DRM False Memory Paradigm | 025_drm_false_memory_pending.md | Pending | 4 days | Medium |
| 026 | Spacing Effect Validation | 026_spacing_effect_pending.md | Pending | 3 days | Medium |
| 027 | Pattern Completion Accuracy | 027_pattern_completion_pending.md | Pending | 3 days | Medium |
| 028 | Reconsolidation Dynamics | 028_reconsolidation_dynamics_pending.md | Pending | 4 days | Medium |

**Phase 3 Total**: 32 days (6.5 weeks)

**Description**: Validates dual memory architecture reproduces empirical cognitive psychology phenomena (Neely 1977, Anderson 1974, Takashima 2006, etc.).

---

## M18 Total Timeline

- **Phase 1**: 10 weeks (parallelizable)
- **Phase 2**: 3 weeks (sequential after Phase 1 core tests)
- **Phase 3**: 6.5 weeks (can overlap with Phase 1)

**Total Duration**: 10-14 weeks depending on parallelization

**Prerequisites**: M17 at 40% for Phase 1, M17 at 60% for Phases 2-3

---

## M18.1: Production Readiness & Deployment (101-115)

**Separate milestone, documented in M18_PRODUCTION_READINESS_PLAN.md**

### Phase 1: End-to-End Workflows (101-105)

| Task | Name | File | Status | Duration |
|------|------|------|--------|----------|
| 101 | Knowledge Graph Construction | 101_knowledge_graph_construction_workflow_pending.md | Pending | 2 days |
| 102 | Recommendation Engine Workflow | 102_recommendation_engine_workflow_pending.md | Pending | 2 days |
| 103 | Fraud Detection Workflow | 103_fraud_detection_workflow_pending.md | Pending | 2 days |
| 104 | Consolidation Crash Recovery | 104_consolidation_crash_recovery_pending.md | Pending | 2 days |
| 105 | Resource Exhaustion Handling | 105_resource_exhaustion_handling_pending.md | Pending | 2 days |

### Phase 2: Chaos Engineering (106-108)

| Task | Name | File | Status | Duration |
|------|------|------|--------|----------|
| 106 | Network Partition Resilience | 106_network_partition_resilience_pending.md | Pending | 2 days |
| 107 | Data Corruption Detection | 107_data_corruption_detection_pending.md | Pending | 2 days |
| 108 | Cascading Failure Prevention | 108_cascading_failure_prevention_pending.md | Pending | 2 days |

### Phase 3: Operational Readiness (109-111)

| Task | Name | File | Status | Duration |
|------|------|------|--------|----------|
| 109 | Backup and Restore Workflows | 109_backup_restore_workflows_pending.md | Pending | 2 days |
| 110 | Monitoring and Alerting Validation | 110_monitoring_alerting_validation_pending.md | Pending | 2 days |
| 111 | Performance Troubleshooting Runbooks | 111_performance_troubleshooting_pending.md | Pending | 2 days |

### Phase 4: Performance SLOs (112-114)

| Task | Name | File | Status | Duration |
|------|------|------|--------|----------|
| 112 | Latency Budget Testing | 112_latency_budget_testing_pending.md | Pending | 2 days |
| 113 | Throughput Capacity Planning | 113_throughput_capacity_planning_pending.md | Pending | 2 days |
| 114 | Competitive Benchmark Validation | 114_competitive_benchmark_validation_pending.md | Pending | 2 days |

### Phase 5: API Compatibility (115)

| Task | Name | File | Status | Duration |
|------|------|------|--------|----------|
| 115 | Zero-Downtime Migration Testing | 115_zero_downtime_migration_pending.md | Pending | 2 days |

**M18.1 Total**: 30 days (6 weeks)

**Prerequisites**: M17 at 60% completion, M18 Phase 1 at 50%

---

## Success Metrics by Phase

### M18 Phase 1: Performance Infrastructure
- [ ] 24h soak test: Zero leaks, <1% latency drift
- [ ] 10M nodes: P99 <15ms, throughput >500 ops/s
- [ ] 32-core efficiency: >80%
- [ ] Multi-tenant isolation: <1% cross-space interference
- [ ] CI/CD gates: Block >5% regressions

### M18 Phase 2: Graph Correctness
- [ ] Loom tests: 100% interleaving coverage for 2-3 threads
- [ ] Property tests: 10,000 cases per property, all pass
- [ ] Probabilistic: Bayesian updates within numerical precision
- [ ] Biological: Fan effect, priming, forgetting curve match literature

### M18 Phase 3: Cognitive Validation
- [ ] Correlation: r > 0.75 for all 8 phenomena
- [ ] Effect sizes: Within ±0.3 Cohen's d of published values
- [ ] Temporal accuracy: Timing windows match human data ±20%
- [ ] Reproducibility: <5% variance across runs

### M18.1: Production Readiness
- [ ] Zero data loss across all chaos scenarios
- [ ] RTO <15min, RPO = 0 for all failures
- [ ] Runbooks executable by L1 without escalation
- [ ] Zero-downtime migration validated

---

## File Renaming Required

### Current → New Numbering

**Production Readiness (M18 → M18.1)**:
- `001_knowledge_graph_construction_workflow_pending.md` → `101_knowledge_graph_construction_workflow_pending.md`
- `002_recommendation_engine_workflow_pending.md` → `102_recommendation_engine_workflow_pending.md`
- `003_fraud_detection_workflow_pending.md` → `103_fraud_detection_workflow_pending.md`
- `004_consolidation_crash_recovery_pending.md` → `104_consolidation_crash_recovery_pending.md`
- `005_resource_exhaustion_handling_pending.md` → `105_resource_exhaustion_handling_pending.md`

**Cognitive Validation (017-020 → 021-024)**:
- `017_semantic_priming_validation_pending.md` → `021_semantic_priming_validation_pending.md`
- `018_anderson_fan_effect_validation_pending.md` → `022_anderson_fan_effect_validation_pending.md`
- `019_consolidation_timeline_validation_pending.md` → `023_consolidation_timeline_validation_pending.md`
- `020_retrograde_amnesia_gradient_validation_pending.md` → `024_retrograde_amnesia_gradient_validation_pending.md`

**Graph Validation (keep at 017-020)**:
- `017_graph_concurrent_correctness_pending.md` → Keep as is
- `018_graph_invariant_validation_pending.md` → Keep as is
- `019_probabilistic_semantics_validation_pending.md` → Keep as is
- `020_biological_plausibility_validation_pending.md` → Keep as is

---

## Dependencies

### M18 Phase 1 (Performance)
- **Start Condition**: M17 at 40% (basic dual memory operational)
- **M17 Dependencies**: Tasks 001-002 (dual memory types, storage)
- **Can run in parallel**: Yes, most tasks independent

### M18 Phase 2 (Graph Correctness)
- **Start Condition**: M17 at 60% (spreading activation operational)
- **M17 Dependencies**: Tasks 001-007 (through fan effect)
- **Blockers**: Phase 1 Tasks 001-003 (basic performance baseline)

### M18 Phase 3 (Cognitive)
- **Start Condition**: M17 at 60% (consolidation operational)
- **M17 Dependencies**: Tasks 001-009 (through blended recall)
- **Blockers**: Phase 2 complete (correctness validated first)

### M18.1 (Production Readiness)
- **Start Condition**: M17 at 60%, M18 Phase 1 at 50%
- **M17 Dependencies**: Tasks 001-009, 013 (monitoring)
- **M18 Dependencies**: Phase 1 Tasks 001-003 (load testing baseline)

---

## Task Creation Status

### Already Created (Need Renaming)
- [x] 001-003: Production load testing files exist
- [x] 004-016: Performance infrastructure files exist
- [x] 017-020: Cognitive validation files exist (need renaming to 021-024)
- [x] 017-020: Graph validation files exist (currently conflict with cognitive)
- [x] 101-105: Production workflows exist (currently 001-005, need renaming)

### Need Creation
- [ ] 021-024: Cognitive validation (files exist but need renaming)
- [ ] 025-028: Additional cognitive validation (DRM, spacing, pattern, reconsolidation)
- [ ] 106-115: Additional production readiness tasks

---

## Next Steps

1. Rename conflicting files to resolve numbering
2. Create missing task files (025-028, 106-115)
3. Update all cross-references in documentation
4. Create separate M18.1 directory structure
5. Update milestone overview documents

---

## References

- **M17 Overview**: `roadmap/milestone-17/000_milestone_overview_dual_memory.md`
- **M17 Performance Baseline**: `roadmap/milestone-17/PERFORMANCE_BASELINE.md`
- **M17.1 Competitive Framework**: `roadmap/milestone-17.1/README.md`
- **M18 Performance Plan**: `roadmap/milestone-18/README.md`
- **M18 Production Plan**: `roadmap/milestone-18/M18_PRODUCTION_READINESS_PLAN.md`
- **Vision**: `vision.md`
