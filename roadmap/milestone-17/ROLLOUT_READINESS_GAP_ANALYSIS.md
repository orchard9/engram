# Milestone 17 Rollout Readiness Gap Analysis

**Date:** 2025-11-20
**Task:** 015 - Production Validation and Rollout
**Status:** Documentation Complete, Implementation Gaps Identified
**Assessment:** NOT READY FOR PRODUCTION ROLLOUT

## Executive Summary

Task 015 has successfully produced a comprehensive production rollout runbook (`docs/operations/dual_memory_rollout.md`), validated by three specialized agents:
- **Technical-communication-lead**: Created operator-facing runbook (60KB)
- **Systems-product-planner**: Strategic review (verdict: NEEDS WORK)
- **Verification-testing-lead**: Testing methodology review (score: 6.5/10)

**Critical Finding:** The rollout strategy is well-designed, but **5 critical infrastructure gaps** prevent safe production deployment. These gaps must be closed before Phase 0 can begin.

**Estimated Effort to Close Gaps:** 9-13 days (2-3 weeks with dedicated team)

**Recommendation:** Treat gap closure as prerequisite work (Milestone 17.5) before proceeding with rollout.

---

## Gap Analysis

### Critical Gaps (BLOCKING)

#### Gap #1: Runtime Feature Flag System MISSING

**Severity:** CRITICAL (blocks all rollout phases)
**Identified By:** Systems-product-planner, Verification-testing-lead

**Problem:**
- Rollout plan extensively references runtime feature flag HTTP APIs
- Current implementation: all flags are compile-time only (`#[cfg(feature = "dual_memory_types")]`)
- Cannot perform canary rollout without runtime toggles
- Immediate rollback impossible (requires recompilation + redeployment = 30-60 min vs documented 2 min)

**Impact:**
- Phase 2 (Canary) cannot execute cohort-based enablement
- A/B testing methodology breaks without deterministic runtime flags
- Rollback procedures document 4 levels; only 2 are possible (slow ones)
- Recovery time objective violated (2 min target → 30+ min actual)

**Required Implementation:**
```rust
// File: engram-core/src/config/feature_flags.rs
pub struct FeatureFlagManager {
    flags: Arc<DashMap<String, bool>>,
    snapshot_path: PathBuf,
}

impl FeatureFlagManager {
    pub fn toggle(&self, flag: &str, enabled: bool) -> Result<()>;
    pub fn is_enabled(&self, flag: &str) -> bool;
    pub fn should_use_blended_recall(&self, user_id: &str, query_hash: u64, cohort_rate: f32) -> bool;
}

// File: engram-http/src/routes/admin.rs
pub async fn toggle_feature_flag(
    State(state): State<AppState>,
    Json(req): Json<FeatureFlagRequest>,
) -> Result<Json<FeatureFlagResponse>, ApiError> {
    state.flags.toggle(&req.flag, req.enabled)?;
    Ok(Json(FeatureFlagResponse { success: true }))
}
```

**Deliverables:**
1. `engram-core/src/config/feature_flags.rs` - Flag manager with persistence
2. HTTP endpoint `/api/v1/admin/features` for toggle operations
3. Modify activation/consolidation code to check runtime flags
4. Integration tests validating <5s propagation across cluster
5. Documentation in `docs/reference/api.md`

**Estimated Effort:** 2-3 days
**Owner:** Backend Lead

**Acceptance Criteria:**
- [ ] Flag toggle propagates to all instances within 5 seconds
- [ ] State persisted to disk (survives restart)
- [ ] Cohort assignment deterministic (same user_id → same cohort)
- [ ] HTTP API returns 200 with confirmation
- [ ] Integration test: toggle flag, verify activation path changes

---

#### Gap #2: Kubernetes Deployment Manifests INCOMPLETE

**Severity:** CRITICAL (blocks k8s rollout)
**Identified By:** Systems-product-planner

**Problem:**
- Rollout plan references 5 deployment variants that don't exist
- Traffic splitting mechanism undefined (service mesh? Istio?)
- Only base deployment exists (`engram-cluster.yaml`, `statefulset.yaml`)

**Missing Files:**
```
deployments/kubernetes/variants/shadow-mode.yaml         - NOT FOUND
deployments/kubernetes/variants/canary-5pct.yaml         - NOT FOUND
deployments/kubernetes/variants/ramp-25pct.yaml          - NOT FOUND
deployments/kubernetes/variants/ramp-50pct.yaml          - NOT FOUND
deployments/kubernetes/variants/ramp-75pct.yaml          - NOT FOUND
deployments/kubernetes/variants/full-100pct.yaml         - NOT FOUND
```

**Impact:**
- Cannot execute phased rollout as documented
- Rollback procedures reference non-existent manifests
- Traffic splitting for A/B testing not possible

**Required Implementation:**
```yaml
# File: deployments/kubernetes/variants/canary-5pct.yaml
apiVersion: networking.istio.io/v1alpha3
kind: VirtualService
metadata:
  name: engram-traffic-split
spec:
  hosts:
  - engram.svc.cluster.local
  http:
  - match:
    - headers:
        x-engram-cohort:
          exact: canary
    route:
    - destination:
        host: engram-canary
        subset: v2-dual-memory
      weight: 100
  - route:
    - destination:
        host: engram-stable
        subset: v1-single-type
      weight: 95
    - destination:
        host: engram-canary
        subset: v2-dual-memory
      weight: 5
```

**Deliverables:**
1. Create `deployments/kubernetes/variants/` directory
2. Implement 6 deployment manifests (shadow through full)
3. Choose traffic splitting: Istio VirtualService (recommended) OR native k8s Service
4. Document service mesh dependencies in Pre-Deployment Checklist
5. Test deployment transitions in staging cluster
6. Update rollout runbook with actual manifest paths

**Estimated Effort:** 2-3 days
**Owner:** DevOps Lead

**Acceptance Criteria:**
- [ ] All 6 deployment variants deployable to staging
- [ ] Traffic split verified (5%/95% observed in logs)
- [ ] Rollover from one phase to next < 5 minutes
- [ ] Rollback tested (full → canary → stable)
- [ ] Service mesh dependencies documented

---

#### Gap #3: Load Testing Infrastructure MISSING

**Severity:** CRITICAL (blocks performance validation)
**Identified By:** Systems-product-planner, Verification-testing-lead

**Problem:**
- `./target/release/loadtest` binary exists but no source code in repo
- Load test scenarios incomplete (only `m17_baseline.toml` exists)
- 6 automation scripts referenced in runbook don't exist
- Cannot execute Phase 0 baseline or any performance validation

**Missing Components:**
```
tools/loadtest/src/                              - NO SOURCE CODE
scenarios/canary.toml                            - NOT FOUND
scenarios/ramp-25.toml                           - NOT FOUND
scenarios/competitive.toml                       - NOT FOUND
scripts/run_soak_test.sh                         - NOT FOUND
scripts/toggle_feature_flag.sh                   - NOT FOUND
scripts/validate_monitoring.sh                   - NOT FOUND
scripts/compare_canary_control.sh                - NOT FOUND
scripts/analyze_canary_significance.py           - NOT FOUND
scripts/monitor_full_rollout.sh                  - NOT FOUND
```

**Impact:**
- Phase 0 (baseline establishment) cannot be executed
- 24-hour soak test cannot run
- Nightly canary comparisons impossible
- Performance regression detection blocked
- A/B testing methodology breaks

**Quality Issues with Existing `m17_baseline.toml`:**
- Pattern completion disabled (core dual-memory feature untested)
- 50% writes (unrealistic - should be 5-10%)
- Single memory space (doesn't test multi-tenancy)
- 60s duration too short for GC pressure testing
- No warm-up period (first 10s affected by cold start)

**Deliverables:**
1. Add `tools/loadtest/src/` to repository OR document binary provenance
2. Create load scenarios: canary, ramp-25/50/75, competitive
3. Implement 6 missing automation scripts
4. Improve baseline scenario (realistic workload)
5. Document loadtest usage in `tools/loadtest/README.md`
6. Add 30s warm-up period to all scenarios

**Estimated Effort:** 3-4 days
**Owner:** Performance Engineer

**Acceptance Criteria:**
- [ ] Loadtest source code in repo with build instructions
- [ ] All scenarios runnable with deterministic results
- [ ] Soak test runs unattended for 24 hours
- [ ] Scripts return exit code 0 on success, 1 on failure
- [ ] Performance comparison automated with statistical analysis
- [ ] Warm-up period excludes first 30s from metrics

---

#### Gap #4: Monitoring Dashboards PARTIALLY IMPLEMENTED

**Severity:** HIGH PRIORITY (reduces observability)
**Identified By:** Systems-product-planner

**Problem:**
- Only 1 of 4 referenced dashboards exists
- Task 013 marked as "complete" but actually "pending"
- Missing dashboards reduce visibility during rollout

**Dashboard Status:**
```
deployments/grafana/dashboards/dual-memory.json         - EXISTS (598 lines)
deployments/grafana/dashboards/cognitive-metrics.json   - NOT FOUND
deployments/grafana/dashboards/canary-comparison.json   - NOT FOUND
deployments/grafana/dashboards/slo-compliance.json      - NOT FOUND
```

**Missing Alert Rules:**
```
BlendedRecallLatencyP99Breach    - NOT FOUND in deployments/prometheus/alerts.yml
ConceptQualityViolation          - NOT FOUND in deployments/prometheus/alerts.yml
FanEffectExcessive               - NOT FOUND in deployments/prometheus/alerts.yml
```

**Impact:**
- Operators cannot access referenced dashboards during rollout
- Canary comparisons require manual Prometheus queries
- SLO violation detection relies on alerts only, no visual dashboards
- Reduced operational awareness

**Deliverables:**
1. Create `cognitive-metrics.json` (consolidation perf, fan effect, semantic health)
2. Create `canary-comparison.json` (side-by-side latency/throughput/quality)
3. Create `slo-compliance.json` (SLO % with red/yellow/green zones)
4. Add 3 missing Prometheus alert rules
5. Mark Task 013 as in-progress, complete implementation
6. Validate all dashboards render with test data

**Estimated Effort:** 1-2 days
**Owner:** SRE Lead

**Acceptance Criteria:**
- [ ] All 4 dashboards exist and render in Grafana
- [ ] Alert rules fire correctly in test scenarios
- [ ] Dashboard UIDs match runbook references
- [ ] README.md updated with dashboard descriptions
- [ ] Task 013 marked complete

---

#### Gap #5: Configuration Management INADEQUATE

**Severity:** MEDIUM PRIORITY (blocks consistent deployment)
**Identified By:** Systems-product-planner

**Problem:**
- Production config missing critical sections referenced in runbook
- Feature flag state undefined on startup
- Cohort assignment parameters missing

**Missing from `config/production.toml`:**
```toml
[features]                      # NOT PRESENT
dual_memory_types = false
blended_recall = false
fan_effect = false
monitoring = true

[blended_recall]               # NOT PRESENT
cohort_sampling_rate = 0.0
cohort_seed = 42
base_episodic_weight = 0.7
base_semantic_weight = 0.3
semantic_timeout_ms = 100
enable_quality_threshold = 0.65
```

**Impact:**
- Cannot deploy with documented configuration
- Undefined behavior on startup (defaults may not match expectations)
- Cohort assignment impossible without seed parameter

**Deliverables:**
1. Add `[features]` section to all config templates (development, staging, production)
2. Add `[blended_recall]` section with all parameters
3. Document config schema in `docs/reference/configuration.md`
4. Add config validation tests (ensure required fields present)
5. Update sample configs in repository

**Estimated Effort:** 1 day
**Owner:** Backend Lead

**Acceptance Criteria:**
- [ ] All config templates have required sections
- [ ] Config validation test catches missing fields
- [ ] Documentation updated with schema
- [ ] Sample production config works end-to-end

---

### Non-Critical Issues (Enhancements)

#### Issue #1: Statistical Testing Methodology

**Severity:** LOW (testing works but not optimal)
**Identified By:** Verification-testing-lead

**Issues:**
- Using two-tailed t-test when one-tailed appropriate (checking regression only)
- Cohort assignment hashes both user_id and query_hash (unstable per-user experience)
- Single 60s measurement has high variance (should run 3 trials, report median)

**Recommendations:**
1. Change to one-tailed test (H0: μ_canary ≤ μ_control)
2. Hash user_id only for consistent user experience
3. Run 3 trials per phase, report median with confidence intervals

**Estimated Effort:** 1 day

---

#### Issue #2: Chaos Engineering Integration

**Severity:** LOW (nice-to-have)
**Identified By:** Verification-testing-lead

**Issues:**
- Production chaos drills reference non-existent APIs
- Global failure injection unsafe (affects all traffic)
- No framework integration (Chaos Mesh, Toxiproxy)

**Recommendations:**
1. Use pre-production chaos testing only
2. Integrate Chaos Mesh for k8s deployments
3. Add network partition drill (test replication)

**Estimated Effort:** 2 days

---

#### Issue #3: Property-Based Testing

**Severity:** LOW (existing tests adequate)
**Identified By:** Verification-testing-lead

**Gap:**
- No property-based tests for core invariants
- Should have 20+ tests covering:
  - Episode retrieval determinism
  - Consolidation preservation of count
  - Confidence monotonicity during decay

**Recommendations:**
1. Add `proptest` crate
2. Create `tests/property_tests.rs`
3. Cover 5 key invariants

**Estimated Effort:** 2 days

---

## Summary of Required Work

### Critical Path (MUST COMPLETE BEFORE PHASE 0)

| Gap | Effort | Owner | Blocks |
|-----|--------|-------|--------|
| Runtime Feature Flags | 2-3 days | Backend Lead | ALL PHASES |
| K8s Deployment Manifests | 2-3 days | DevOps Lead | Phased rollout |
| Load Testing Infrastructure | 3-4 days | Performance Engineer | Performance validation |
| Monitoring Dashboards | 1-2 days | SRE Lead | Observability |
| Configuration Management | 1 day | Backend Lead | Consistent deployment |

**Total Critical Path:** 9-13 days

**Team Composition:**
- 1 Backend Lead (feature flags + config)
- 1 DevOps Lead (k8s manifests)
- 1 Performance Engineer (load testing)
- 1 SRE Lead (monitoring)

**Timeline:**
- With dedicated team: 2-3 weeks (parallel work)
- With part-time contributors: 4-6 weeks

---

## Rollout Readiness Decision

### Current State: NOT READY FOR PRODUCTION

**Blockers:**
1. Runtime feature flags missing (cannot rollback quickly)
2. K8s deployment manifests missing (cannot execute phased rollout)
3. Load testing infrastructure incomplete (cannot validate performance)
4. Monitoring dashboards incomplete (reduced observability)
5. Configuration management inadequate (deployment inconsistency)

### Go/No-Go: NO-GO

**Do not start Phase 0** until all five critical gaps are closed and validated in staging.

### Post-Gap-Closure Assessment

Once all critical gaps closed:
- **Strategic Completeness:** EXCELLENT (8.5/10)
- **Technical Soundness:** VERY GOOD (8/10)
- **Operational Readiness:** GOOD (7.5/10)
- **Risk Management:** VERY GOOD (8/10)

**Overall Rollout Plan Quality:** 8/10 after gaps closed

**Recommendation:** READY FOR PRODUCTION once all action items complete

---

## Recommended Path Forward

### Option 1: Milestone 17.5 (Recommended)

Treat gap closure as prerequisite milestone before rollout:

**Milestone 17.5: Rollout Infrastructure**
- Task 017: Runtime Feature Flag System (2-3 days)
- Task 018: K8s Deployment Variants (2-3 days)
- Task 019: Load Testing Infrastructure (3-4 days)
- Task 020: Monitoring Dashboard Completion (1-2 days)
- Task 021: Configuration Management (1 day)

**Total:** 9-13 days with dedicated team

**Benefits:**
- Clear dependency tracking
- Proper testing of each component
- Gradual integration validation

### Option 2: Extended Shadow Mode

If timeline critical, extend Phase 1 from 1 week to 3 weeks:

**Week 1:** Build missing infrastructure (feature flags, k8s manifests)
**Week 2:** Complete monitoring + load testing
**Week 3:** Shadow mode validation with all infrastructure complete

**Benefits:**
- Rollout starts sooner (shadow mode with reduced functionality)
- Infrastructure built during shadow mode
- Skip initial canary, go straight to 25% after infrastructure ready

**Risks:**
- Reduced validation quality
- Pressure to proceed before infrastructure complete
- Shadow mode less valuable without full feature set

**Recommendation:** Option 1 (Milestone 17.5) is safer and more thorough.

---

## Task 015 Status

**Deliverables Completed:**
- [x] Comprehensive rollout runbook (`docs/operations/dual_memory_rollout.md`)
- [x] Strategic review by systems-product-planner
- [x] Testing methodology review by verification-testing-lead
- [x] Gap analysis document (this file)

**Deliverables NOT Completed (out of scope for Task 015):**
- [ ] Runtime feature flag implementation (Milestone 17.5 Task 017)
- [ ] K8s deployment manifests (Milestone 17.5 Task 018)
- [ ] Load testing infrastructure (Milestone 17.5 Task 019)
- [ ] Monitoring dashboard completion (Milestone 17.5 Task 020)
- [ ] Configuration management (Milestone 17.5 Task 021)

**Task 015 Verdict:** COMPLETE (documentation and planning)

**Follow-Up Work:** Milestone 17.5 (rollout infrastructure)

---

## References

- Rollout Runbook: `docs/operations/dual_memory_rollout.md`
- Systems Planner Review: Agent report (2025-11-20)
- Testing Review: `tmp/DUAL_MEMORY_TESTING_STRATEGY_REVIEW.md`
- Task 015 Specification: `roadmap/milestone-17/015_production_validation_pending.md`
- Task 013 (Monitoring): `roadmap/milestone-17/013_monitoring_metrics_complete.md` (actually pending)
