# Milestone 18: Production Readiness Validation Plan

**Created**: 2025-11-12
**Author**: Denise Gosnell (Graph Systems Acceptance Tester Agent)
**Purpose**: Comprehensive production deployment validation for dual memory architecture

## Executive Summary

Milestone 18 validates Engram's readiness for production deployment through five phases of acceptance testing:

1. **End-to-End Workflows** (3 tasks) - Realistic user journeys with complete data flows
2. **Chaos Engineering** (3 tasks) - Failure injection and recovery validation
3. **Operational Readiness** (3 tasks) - Runbook validation by operators
4. **Performance SLOs** (3 tasks) - Load testing with production access patterns
5. **API Compatibility** (3 tasks) - Migration paths and client validation

This complements the existing M18 performance testing infrastructure (README.md) by adding production deployment concerns: failure recovery, operational procedures, migration paths, and real-world workflow validation.

## Relationship to Existing M18 Plan

The existing M18 README.md focuses on performance infrastructure (load testing, scalability, NUMA, cache efficiency, regression prevention). This plan adds production readiness validation:

**Existing M18 (Performance Infrastructure)**:
- Load testing (tasks 001-003)
- Scalability validation (tasks 004-006)
- Concurrency testing (tasks 007-009)
- NUMA/cache optimization (tasks 010-013)
- Regression prevention (tasks 014-016)

**This Plan (Production Readiness)**:
- End-to-end workflows (tasks 101-103)
- Chaos engineering (tasks 104-106)
- Operational procedures (tasks 107-109)
- Performance SLOs (tasks 110-112)
- API compatibility (tasks 113-115)

**Recommendation**: Integrate both plans, renumber production readiness tasks as 101-115 to avoid conflicts, or split into M18 (Performance) and M18.1 (Production Readiness).

## Task Catalog (Production Readiness)

### Phase 1: End-to-End Workflows

**Task 101: Knowledge Graph Construction Workflow**
- **File**: `roadmap/milestone-18/001_knowledge_graph_construction_workflow_pending.md`
- **Objective**: Validate streaming ingestion → consolidation → blended recall
- **Duration**: 2 days
- **Key Validation**: 10K Wikipedia articles, zero data loss, space isolation

**Task 102: Recommendation Engine Workflow**
- **File**: `roadmap/milestone-18/002_recommendation_engine_workflow_pending.md`
- **Objective**: Validate high-read workload with temporal decay and spreading activation
- **Duration**: 2 days
- **Key Validation**: MovieLens 100K, temporal bias 1.8-2.2x, P99 <100ms

**Task 103: Fraud Detection Workflow**
- **File**: `roadmap/milestone-18/003_fraud_detection_workflow_pending.md`
- **Objective**: Validate pattern matching, completion, temporal velocity checks
- **Duration**: 2 days
- **Key Validation**: 100K transactions, 90% precision, 85% recall, P99 <50ms

### Phase 2: Chaos Engineering

**Task 104: Consolidation Crash Recovery**
- **File**: `roadmap/milestone-18/004_consolidation_crash_recovery_pending.md`
- **Objective**: Validate WAL recovery at 5 crash points with zero data loss
- **Duration**: 2 days
- **Key Validation**: RTO <15min, RPO=0, automatic recovery

**Task 105: Resource Exhaustion Handling**
- **Objective**: Memory/disk pressure, graceful degradation, OOM prevention
- **Duration**: 2 days
- **Key Validation**: Admission control triggers, no crash under pressure

**Task 106: Network Partition Resilience**
- **Objective**: Single-node resilience, client reconnection, request replay
- **Duration**: 2 days
- **Key Validation**: Operations resume after partition heal

### Phase 3: Operational Readiness

**Task 107: Backup and Restore Workflows**
- **Objective**: Hot backup, point-in-time recovery, data integrity validation
- **Duration**: 2 days
- **Key Validation**: Backup <30min, restore <1hr, zero data loss

**Task 108: Monitoring and Alerting Validation**
- **Objective**: Chaos scenarios trigger alerts, telemetry captures failures
- **Duration**: 2 days
- **Key Validation**: 100% failure detection, <30s alert latency

**Task 109: Performance Troubleshooting Runbooks**
- **Objective**: Slow query diagnosis, capacity planning, optimization procedures
- **Duration**: 2 days
- **Key Validation**: Runbooks executable by L1 on-call without escalation

### Phase 4: Performance SLOs

**Task 110: Latency Budget Testing**
- **Objective**: P50/P95/P99 SLOs under sustained production load
- **Duration**: 2 days
- **Key Validation**: SLO compliance for 24 hours

**Task 111: Throughput Capacity Planning**
- **Objective**: Find breaking points, establish scaling curves
- **Duration**: 2 days
- **Key Validation**: Linear scaling to 16 cores, >80% efficiency to 32

**Task 112: Competitive Benchmark Validation**
- **Objective**: Neo4j/Qdrant parity check using M17.1 framework
- **Duration**: 2 days
- **Key Validation**: <10% competitive regression, document positioning

### Phase 5: API Compatibility

**Task 113: Zero-Downtime Migration**
- **Objective**: Episodic-only to dual memory upgrade without restart
- **Duration**: 2 days
- **Key Validation**: Rolling upgrade succeeds, client sessions preserved

**Task 114: API Versioning and Deprecation**
- **Objective**: Backward compatibility, graceful deprecation paths
- **Duration**: 2 days
- **Key Validation**: Old clients continue working, deprecation warnings

**Task 115: Client Library Integration**
- **Objective**: Python/TypeScript/Rust clients tested end-to-end
- **Duration**: 2 days
- **Key Validation**: All bindings pass integration tests

## Acceptance Criteria Framework

Every task follows this structure:

### Pass Criteria
- **Functional**: All specified behaviors work correctly
- **Performance**: Meets or exceeds latency/throughput targets
- **Reliability**: Zero data loss, defined RTO/RPO met
- **Observability**: Telemetry captures all tested scenarios
- **Quality**: Zero errors, zero clippy warnings

### Fail Criteria
- Any data loss detected
- Performance regression ≥5% (internal) or ≥10% (competitive)
- Errors during workflow execution
- Missing telemetry for failures
- Runbook procedures fail or incomplete

## Testing Infrastructure

### Required Components
1. **Test Fixtures**:
   - Wikipedia 10K embeddings (compressed JSONL)
   - MovieLens 100K dataset
   - Synthetic fraud dataset (100K transactions)

2. **Chaos Injection**:
   - Crash injection via conditional panics
   - Resource exhaustion via cgroups
   - Network partition via iptables

3. **Observability Stack**:
   - Prometheus metrics collection
   - Grafana dashboards
   - Log aggregation (structured logging)
   - Distributed tracing (if M11+ traces implemented)

4. **Automation Scripts**:
   - `scripts/acceptance/<task>_*.sh` - Individual test runners
   - `scripts/run_m18_production_acceptance.sh` - Full suite runner
   - `scripts/m17_performance_check.sh` - Regression detection

### Observability Validation

Every test must verify:
1. **Metrics**: Prometheus captures operation counts, latencies, errors
2. **Logs**: Structured logs contain context for debugging
3. **Alerts**: Monitoring triggers alerts within SLO thresholds
4. **Traces**: Operations correlated across components (if applicable)

## Performance Budget

All tests must maintain M17 baseline performance within 5% tolerance:
- **P99 Latency**: ≤0.526ms (0.501 × 1.05)
- **P95 Latency**: ≤0.458ms (0.436 × 1.05)
- **Throughput**: ≥949.9 ops/sec (999.9 × 0.95)
- **Error Rate**: 0.0% (zero tolerance)

Specialized operations have different budgets:
- **Consolidation P99**: <500ms (background operation)
- **Blended Recall P99**: <50ms (acceptable overhead)
- **Pattern Completion P99**: <100ms (complex operation)

## Dependencies

### M17 Prerequisites
- Tasks 001-006: Dual memory types, storage, formation, consolidation
- Task 009: Blended recall implementation
- Task 013: Monitoring and metrics

### M17.1 Prerequisites
- Complete competitive baseline framework
- Automated benchmark suite operational

### Infrastructure Prerequisites
- Prometheus/Grafana monitoring stack
- Load test tool with chaos injection support
- Diagnostic scripts (engram_diagnostics.sh)

### Documentation Prerequisites
- Operations guides (M16)
- Deployment procedures (production-deployment.md)
- Incident response procedures (incident-response.md)

## Timeline and Sequencing

**Total Duration**: 30 days (6 weeks) minimum

**Phase Sequencing** (can parallelize within phases):
- Week 1-2: Phase 1 (End-to-End Workflows)
- Week 2-3: Phase 2 (Chaos Engineering)
- Week 3-4: Phase 3 (Operational Readiness)
- Week 4-5: Phase 4 (Performance SLOs)
- Week 5-6: Phase 5 (API Compatibility)

**Critical Path**: All phases sequential, but tasks within phase can run parallel.

## Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Acceptance Tests Passing | 15/15 | Automated test results |
| Data Loss Incidents | 0 | Integrity checks across all tests |
| Performance Regression | <5% | M17 baseline comparison |
| Runbook Success Rate | 100% | External operator validation |
| Client Compatibility | 100% | All bindings pass integration |
| Observability Coverage | 100% | Telemetry validation during tests |

## Integration with Existing M18

This plan complements the existing M18 performance infrastructure:

**Existing M18 Tasks (001-016)**: Performance testing infrastructure
- Focus: Scalability, NUMA, cache efficiency, regression prevention
- Audience: Performance engineers, systems architects
- Deliverables: Profiling tools, CI/CD gates, competitive dashboards

**This Plan Tasks (101-115)**: Production deployment readiness
- Focus: Failure recovery, operational procedures, real-world workflows
- Audience: SREs, operators, application developers
- Deliverables: Runbooks, acceptance tests, migration guides

**Recommended Integration**:
1. Keep existing M18 (001-016) for performance infrastructure
2. Add M18.1 (101-115) for production readiness
3. Both milestones share M17 baseline and M17.1 competitive framework
4. Gate production deployment on BOTH M18 and M18.1 completion

## Out of Scope

**Deliberately excluded** (deferred to future work):
- Distributed architecture testing (M14 dependency)
- GPU-specific failure modes (validated in M12)
- Multi-datacenter deployment (M14+)
- Custom hardware optimization (post-M19)
- Third-party integration testing (M20+)

**Rationale**: Focus on single-node production readiness. Distributed concerns addressed after M14.

## Next Steps

1. **Review with team**: Validate approach, identify concerns
2. **Prioritize tasks**: Decide on M18 vs M18.1 split
3. **Generate fixtures**: Download/create test datasets
4. **Build chaos tooling**: Implement crash injection, resource limits
5. **Start Phase 1**: Begin with Task 101 (Knowledge Graph Construction)

## References

- M17 Dual Memory: `roadmap/milestone-17/000_milestone_overview_dual_memory.md`
- M17 Performance Baseline: `roadmap/milestone-17/PERFORMANCE_BASELINE.md`
- M17.1 Competitive Framework: `roadmap/milestone-17.1/README.md`
- Existing M18 Plan: `roadmap/milestone-18/README.md`
- Operations Documentation: `docs/operations/`
- Vision: `vision.md`

---

**Document Status**: Draft for review
**Next Review**: After M17 reaches 60% completion
**Owner**: Graph Systems Acceptance Tester (Denise Gosnell agent)
