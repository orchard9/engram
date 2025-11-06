# Milestone 13 Implementation Roadmap

**Compiled By:** Bryan Cantrill (systems-product-planner)
**Date:** 2025-10-26
**Status:** READY FOR IMPLEMENTATION (with critical corrections)

## Executive Summary

Milestone 13 implements cognitive patterns (priming, interference, reconsolidation) with psychology validation. Based on comprehensive agent reviews, we have:

**READY TO IMPLEMENT:** 5 tasks (with corrections applied)
**REQUIRES REDESIGN:** 1 task (Task 010 - skeletal spec)
**BLOCKED PENDING API:** 4 tasks (awaiting implementations from prior tasks)
**DOCUMENTATION:** 1 task (enhanced spec ready)

**CRITICAL PATH:** Task 001 → Task 002 → Task 008 → Documentation
**ESTIMATED DURATION:** 18-20 days (was 14 days - increased for quality)
**MAJOR RISKS:** API availability, psychology validation failure, time simulation artifacts

---

## 1. Task Status Summary

| Task | Name | Status | File to Use | Dependencies | Duration | Risk |
|------|------|--------|-------------|--------------|----------|------|
| 001 | Zero-Overhead Metrics | READY | `001_CORRECTED.md` | None | 3d | LOW |
| 002 | Semantic Priming | BLOCKED | `002_pending.md` | Task 001 | 2d | MEDIUM |
| 003 | Associative/Repetition | BLOCKED | `003_pending.md` | Task 002 | 1.5d | LOW |
| 004 | Proactive Interference | BLOCKED | `004_pending.md` | Task 002 | 1.5d | MEDIUM |
| 005 | Retroactive/Fan Effect | BLOCKED | `005_pending.md` | Task 004 | 1.5d | MEDIUM |
| 006 | Reconsolidation Core | BLOCKED | `006_pending.md` | Task 002 | 2d | MEDIUM |
| 007 | Reconsolidation Integration | BLOCKED | `007_pending.md` | Task 006 | 1d | LOW |
| 008 | DRM False Memory | READY | `008_ENHANCED.md` | Task 002, M8 | 2.5d | HIGH |
| 009 | Spacing Effect | READY | `009_ENHANCED.md` | M4 | 2d | HIGH |
| 010 | Interference Suite | REQUIRES REDESIGN | `010_ENHANCED.md` | Task 004, 005 | 3d | HIGH |
| 011 | Cognitive Tracing | READY | `011_CORRECTED.md` | Task 001 | 5d | MEDIUM |
| 012 | System Integration | BLOCKED | Not reviewed | All prior | 2d | MEDIUM |
| 013 | Integration Testing | BLOCKED | Not reviewed | Task 012 | 1.5d | LOW |
| 014 | Documentation | READY | `014_ENHANCED.md` | Task 013 | 1.5d | LOW |

### Status Definitions

**READY:** Corrected/enhanced spec exists, can begin immediately
**BLOCKED:** Awaiting dependency completion (correct spec exists)
**REQUIRES REDESIGN:** Spec incomplete, needs additional work

---

## 2. Critical Path Analysis

### Phase 1: Foundation (Days 1-3)
**Blocking:** Everything depends on this

**Task 001: Zero-Overhead Metrics (3 days)**
- **File:** `001_zero_overhead_metrics_CORRECTED.md`
- **Critical Fixes Applied:**
  - Removed Arc wrapper (eliminates pointer indirection)
  - Fixed histogram sum calculation (was broken, now uses atomic_float)
  - Added loom tests for lock-free correctness
  - Corrected assembly verification methodology
- **Deliverables:**
  - `/engram-core/src/metrics/cognitive_patterns.rs`
  - `/engram-core/benches/metrics_overhead.rs`
  - `/engram-core/tests/metrics/loom_tests.rs`
- **Acceptance:** All loom tests pass, <1% overhead, zero-cost when disabled
- **Risk:** Low - well-specified, no ambiguity

### Phase 2: Core Implementations (Days 4-11)
**Parallelizable after Phase 1**

**Sequential Track A: Priming → DRM Validation**

**Task 002: Semantic Priming (2 days)** [BLOCKED on 001]
- **File:** `002_semantic_priming_pending.md` (original - not reviewed)
- **Pre-Implementation:** Verify API compatibility
- **Critical:** This unblocks Task 008 (DRM validation)
- **Risk:** MEDIUM - needs spreading activation from M3

**Task 008: DRM False Memory Validation (2.5 days)** [BLOCKED on 002]
- **File:** `008_drm_false_memory_ENHANCED.md`
- **Critical Enhancements:**
  - Increased sample size to n=200 (was 100) for 80% power
  - Added embedding generation and validation
  - Replaced string matching with semantic similarity
  - Added Cohen's d effect size calculation
  - Added parameter sweep for failure recovery
- **Pre-Implementation Checklist:**
  ```bash
  # MUST RUN FIRST - verify API availability
  cargo test psychology::api_compatibility
  ```
- **Risk:** HIGH - acid test for cognitive architecture
- **If Fails:** Run parameter sweep, budget +1 day for tuning

**Independent Track B: Time-Based Validations**

**Task 009: Spacing Effect Validation (2 days)** [BLOCKED on M4]
- **File:** `009_spacing_effect_validation_ENHANCED.md`
- **Critical Enhancements:**
  - Increased sample size to n=200 (was 50) for 90% power
  - Added time simulation artifact detection
  - Implemented paired t-test for significance
  - Strengthened stability criterion (25/30 replications)
- **Pre-Implementation:**
  ```bash
  # MUST RUN FIRST - validate time simulation
  cargo test psychology::spacing_time_simulation
  ```
- **Risk:** HIGH - time simulation validity critical
- **Parallel to:** Task 002-008 (no dependencies)

**Independent Track C: Advanced Tracing** [Can start after 001]

**Task 011: Cognitive Tracing (5 days)** [BLOCKED on 001]
- **File:** `011_cognitive_tracing_infrastructure_CORRECTED.md`
- **Major Rewrite:**
  - Original spec FAILED review (unbounded memory, wrong tools)
  - Complete redesign with ring buffers
  - Instant timestamps (not DateTime - 40x faster)
  - Bounded memory strategy
- **Risk:** MEDIUM - complete respecification adds uncertainty
- **Parallel to:** Task 002-009 (only depends on 001)

### Phase 3: Interference (Days 12-15)
**Sequential after priming complete**

**Task 003: Associative/Repetition Priming (1.5 days)** [BLOCKED on 002]
- **File:** `003_associative_repetition_priming_pending.md` (not reviewed)
- **Risk:** LOW - extension of Task 002

**Task 004: Proactive Interference (1.5 days)** [BLOCKED on 002]
- **File:** `004_proactive_interference_detection_pending.md` (not reviewed)
- **Risk:** MEDIUM - uses priming mechanisms

**Task 005: Retroactive/Fan Effect (1.5 days)** [BLOCKED on 004]
- **File:** `005_retroactive_fan_effect_pending.md` (not reviewed)
- **Risk:** MEDIUM - needs RT measurement API

**Task 010: Interference Validation Suite (3 days)** [BLOCKED on 004, 005]
- **File:** `010_interference_validation_suite_ENHANCED.md`
- **STATUS: REQUIRES ADDITIONAL WORK**
- **Original Issue:** Completely underspecified - just stubs
- **Enhancement:** Full experimental protocols added
- **Critical Missing:**
  - Need to verify RT measurement API exists
  - Sample materials need generation
  - Statistical analysis needs implementation
- **Action Required Before Starting:**
  ```bash
  # Check if recall_with_latency() API exists
  grep -r "recall_with_latency" engram-core/src/
  ```
- **Risk:** HIGH - complex multi-paradigm validation
- **Budget:** 3 days (was 1 day) due to complexity

### Phase 4: Reconsolidation (Days 12-14, parallel with interference)
**Can parallelize with Task 003-005**

**Task 006: Reconsolidation Core (2 days)** [BLOCKED on 002]
- **File:** `006_reconsolidation_core_pending.md` (not reviewed)
- **Risk:** MEDIUM - window timing critical

**Task 007: Reconsolidation Integration (1 day)** [BLOCKED on 006]
- **File:** `007_reconsolidation_integration_pending.md` (not reviewed)
- **Risk:** LOW - integration task

### Phase 5: Integration & Documentation (Days 16-20)

**Task 012: System Integration (2 days)** [BLOCKED on all prior]
- **File:** Not reviewed
- **Risk:** MEDIUM - cross-system interactions

**Task 013: Integration Testing (1.5 days)** [BLOCKED on 012]
- **File:** Not reviewed
- **Risk:** LOW - test execution

**Task 014: Documentation (1.5 days)** [BLOCKED on 013]
- **File:** `014_documentation_operational_runbook_ENHANCED.md`
- **Enhancements:**
  - 15+ working code examples (validated via doc tests)
  - Complete runbooks with symptom → diagnosis → fix
  - Parameter decision trees for tuning
  - Quality gates for validation
- **Risk:** LOW - well-specified, quality gates defined

---

## 3. Implementation Order (Recommended)

### Week 1 (Days 1-5)
```
Day 1-3:  Task 001 (Metrics Foundation) [CRITICAL PATH]
Day 4-5:  Task 002 (Semantic Priming) [BLOCKS 008]
          Task 009 (Spacing Effect) [PARALLEL - independent]
          Task 011 (Tracing) [PARALLEL - only needs 001]
```

### Week 2 (Days 6-10)
```
Day 6-8:  Task 008 (DRM Validation) [ACID TEST]
          Task 009 continues (2 days total)
          Task 011 continues (5 days total)
Day 9-10: Task 003 (Assoc/Rep Priming)
          Task 006 (Reconsolidation Core) [PARALLEL]
```

### Week 3 (Days 11-15)
```
Day 11-12: Task 004 (Proactive Interference)
           Task 007 (Reconsolidation Integration) [PARALLEL]
Day 13-14: Task 005 (Retroactive/Fan Effect)
Day 15:    Task 010 (Interference Suite) [START - needs 3 days]
```

### Week 4 (Days 16-20)
```
Day 16-17: Task 010 continues (Interference Suite complete)
           Task 012 (System Integration) [PARALLEL after 010]
Day 18:    Task 013 (Integration Testing)
Day 19-20: Task 014 (Documentation)
```

### Parallelization Opportunities

**Maximum Parallelization (3 simultaneous tracks):**
- **Track A (Critical Path):** 001 → 002 → 008 → 003 → 004 → 005 → 010 → 012 → 013 → 014
- **Track B (Time-Based):** 009 (starts after M4, parallel to everything)
- **Track C (Tracing):** 011 (starts after 001, parallel to 002-010)
- **Track D (Reconsolidation):** 006 → 007 (parallel to 003-005)

**Minimum Parallelization (sequential):**
- All tasks in order: 20 days

**Realistic (2-3 simultaneous):**
- 18 days with overlap

---

## 4. Risk Matrix

### HIGH RISK (P0 - May Block Milestone)

#### Risk 1: DRM Validation Fails (Task 008)
**Probability:** 30%
**Impact:** CRITICAL - invalidates entire cognitive architecture
**Symptoms:**
- False recall rate outside [50%, 70%] range
- Chi-square test rejects empirical distribution
- Effect size too small (d < 0.3)

**Mitigation:**
1. **Pre-Implementation:**
   ```bash
   # Verify API compatibility FIRST
   cargo test psychology::api_compatibility
   # Validate embedding semantic structure
   cargo test psychology::drm_embeddings
   ```
2. **If Fails:** Run parameter sweep (already specified in enhanced spec)
3. **Budget:** +1 day for tuning
4. **Escalation:** Consult memory-systems-researcher agent if sweep doesn't resolve

**Early Warning Signs:**
- Veridical recall rate < 60% (baseline memory broken)
- Consolidation events = 0 (consolidation not triggering)
- All false recall confidences = 0 (pattern completion disabled)

#### Risk 2: Spacing Effect Validation Fails (Task 009)
**Probability:** 25%
**Impact:** HIGH - time-based learning broken
**Symptoms:**
- No improvement or reversed effect (massed > distributed)
- Statistical test p > 0.05 (not significant)
- Effect size d < 0.3 (too weak)

**Mitigation:**
1. **Pre-Implementation:**
   ```bash
   # CRITICAL: Validate time simulation first
   cargo test psychology::spacing_time_simulation
   ```
2. **Time Simulation Artifacts:** If linearity test fails, investigate M4 implementation
3. **Parameter Tuning:** Adjust consolidation depth, decay rates
4. **Budget:** +0.5 day for time simulation debugging

**Early Warning Signs:**
- Time simulation tests fail (non-linear decay)
- Confidence never decays (decay disabled)
- No difference between conditions (consolidation broken)

#### Risk 3: Interference Suite Incomplete API (Task 010)
**Probability:** 40%
**Impact:** HIGH - blocks completion
**Symptoms:**
- `recall_with_latency()` API doesn't exist
- RT measurement infeasible with current API

**Mitigation:**
1. **Pre-Implementation:**
   ```bash
   # Check API availability BEFORE starting
   grep -r "recall_with_latency\|with_latency" engram-core/src/
   ```
2. **If Missing:** Use external timing (acceptable for validation)
   ```rust
   let start = Instant::now();
   let results = store.recall_by_content(cue);
   let latency = start.elapsed();
   ```
3. **If External Timing Insufficient:** Add API (budget +0.5 day)

**Decision Point:** Day 15 - before starting Task 010

### MEDIUM RISK (P1 - May Cause Delays)

#### Risk 4: Task 010 Specification Gaps
**Probability:** 30%
**Impact:** MEDIUM - extended timeline
**Issue:** Enhanced spec added protocols, but implementation details may be missing

**Mitigation:**
1. **Review Protocol:** Read enhanced spec thoroughly before starting
2. **Stimulus Generation:** Test `generate_paired_associate_lists()` separately
3. **Statistical Analysis:** Implement regression functions in separate module first
4. **Budget:** Task 010 already increased to 3 days (was 1 day)

#### Risk 5: Task 011 Complexity Underestimated
**Probability:** 25%
**Impact:** MEDIUM - parallel track delayed
**Issue:** Complete rewrite of spec adds uncertainty

**Mitigation:**
1. **Incremental Implementation:**
   - Day 1: Ring buffer + event types (core)
   - Day 2: Collector thread + JSON export
   - Day 3: Configuration + sampling
   - Day 4: OpenTelemetry integration
   - Day 5: Testing + validation
2. **Early Decision Point:** Day 2 - if ring buffer proving difficult, simplify to channel-based
3. **Fallback:** Use existing `tracing` crate instead of custom (sacrifice performance for schedule)

#### Risk 6: Priming Task Dependencies (Tasks 002-007)
**Probability:** 20%
**Impact:** MEDIUM - cascading delays
**Issue:** Tasks 002-007 not reviewed, original specs may have issues

**Mitigation:**
1. **Fast-Track Review:** Request agent reviews for 002-007 after starting 001
2. **API Verification:** Check spreading activation API before Task 002
3. **Buffer Time:** Built 1 day buffer into schedule

### LOW RISK (P2 - Minor Issues)

#### Risk 7: Documentation Examples Break
**Probability:** 15%
**Impact:** LOW - easy to fix
**Mitigation:** Quality gates catch this, doc tests in CI

#### Risk 8: Loom Tests Take Too Long
**Probability:** 20%
**Impact:** LOW - can reduce coverage
**Mitigation:** Run in CI only, not required for local development

---

## 5. File Reference Guide

### For Each Task, Use This File:

| Task | Implementation File | Notes |
|------|-------------------|-------|
| 001 | `001_zero_overhead_metrics_CORRECTED.md` | MANDATORY - fixes critical Arc bug |
| 002 | `002_semantic_priming_pending.md` | Original spec (not reviewed) |
| 003 | `003_associative_repetition_priming_pending.md` | Original spec |
| 004 | `004_proactive_interference_detection_pending.md` | Original spec |
| 005 | `005_retroactive_fan_effect_pending.md` | Original spec |
| 006 | `006_reconsolidation_core_pending.md` | Original spec |
| 007 | `007_reconsolidation_integration_pending.md` | Original spec |
| 008 | `008_drm_false_memory_ENHANCED.md` | MANDATORY - critical fixes for power |
| 009 | `009_spacing_effect_validation_ENHANCED.md` | MANDATORY - fixes power + artifacts |
| 010 | `010_interference_validation_suite_ENHANCED.md` | Use enhanced, check RT API first |
| 011 | `011_cognitive_tracing_infrastructure_CORRECTED.md` | MANDATORY - complete rewrite |
| 012 | Original (not reviewed) | Review before starting |
| 013 | Original (not reviewed) | Review before starting |
| 014 | `014_documentation_operational_runbook_ENHANCED.md` | Use enhanced for quality |

### Quick Lookup: What to Read First

**Starting Task 001?** Read `SYSTEMS_ARCHITECTURE_REVIEW.md` first (understand why corrections needed)
**Starting Task 008?** Read `PSYCHOLOGY_VALIDATION_REVIEW.md` first (understand power issues)
**Starting Task 009?** Read enhanced spec + time simulation section carefully
**Starting Task 010?** Read enhanced spec + check API availability
**Starting Task 011?** Read `SYSTEMS_ARCHITECTURE_REVIEW.md` Task 011 section (major rewrite)
**Starting Task 014?** Read `DOCUMENTATION_TASK_014_REVIEW.md` first (quality requirements)

---

## 6. Quality Gates

### Gate 1: Foundation Complete (End of Day 3)
**Criteria:**
- [ ] Task 001 complete with all loom tests passing
- [ ] Metrics overhead benchmarks < 1%
- [ ] Zero-cost verification passes (size_of == 0 when disabled)
- [ ] No compiler warnings in metrics code

**Decision Point:** If fails, do NOT proceed to Task 002
**Rollback:** Fix Task 001 issues before unblocking dependencies

### Gate 2: DRM Validation Passes (End of Day 8)
**Criteria:**
- [ ] False recall rate: 55-65%
- [ ] Statistical power > 0.80
- [ ] Chi-square test: p > 0.01
- [ ] Effect size: d > 0.8

**Decision Point:** If fails, STOP and run parameter sweep
**Budget:** +1 day for tuning if needed
**Escalation:** If sweep fails, milestone needs redesign

### Gate 3: Spacing Effect Validates (End of Day 10)
**Criteria:**
- [ ] Improvement: 10-50%
- [ ] Statistical significance: p < 0.05
- [ ] Stability: ≥25/30 replications pass
- [ ] Time simulation tests pass

**Decision Point:** If fails, investigate time simulation
**Budget:** +0.5 day for debugging if needed

### Gate 4: Interference Suite Complete (End of Day 17)
**Criteria:**
- [ ] All three paradigms (PI, RI, fan) pass validation
- [ ] Sample sizes adequate (n ≥ 250 total)
- [ ] Statistical tests significant (p < 0.05)
- [ ] Effect sizes in expected ranges

**Decision Point:** If fails, individual paradigms may pass - document failures
**Budget:** +1 day if major issues

### Gate 5: Documentation Quality (End of Day 20)
**Criteria:**
- [ ] All code examples compile (doc tests pass)
- [ ] All parameters match actual code
- [ ] All operations commands work
- [ ] Bibliography accessible

**Decision Point:** If fails, documentation incomplete
**Rollback:** Fix issues before marking milestone complete

---

## 7. Rollback Procedures

### If Task 001 Fails Quality Gates

**Symptoms:**
- Loom tests reveal data races
- Overhead benchmarks exceed 1%
- Histogram calculations wrong

**Rollback Procedure:**
1. Do NOT merge Task 001
2. Review `SYSTEMS_ARCHITECTURE_REVIEW.md` corrections again
3. Consult systems-architecture-optimizer agent
4. Budget +1 day for fixes

### If Task 008 (DRM) Fails Validation

**Symptoms:**
- False recall rate outside [50%, 70%]
- Pattern completion not generating false memories
- Effect size too small

**Rollback Procedure:**
1. Run parameter sweep (already specified in enhanced spec):
   ```bash
   cargo test psychology::drm_parameter_sweep -- --nocapture --ignored
   ```
2. Analyze top configurations
3. Update Task 002 (semantic priming) parameters
4. Re-run DRM validation
5. If still fails after 3 sweep iterations, escalate to memory-systems-researcher

### If Task 009 (Spacing) Fails Validation

**Symptoms:**
- No improvement or reversed effect
- Time simulation artifacts detected

**Rollback Procedure:**
1. Run time simulation validation:
   ```bash
   cargo test psychology::spacing_time_simulation -- --nocapture
   ```
2. If artifacts found, investigate M4 (temporal dynamics) implementation
3. Check consolidation triggering (may need explicit API)
4. Budget +1 day for M4 fixes

### If Task 010 (Interference) API Missing

**Symptoms:**
- `recall_with_latency()` doesn't exist
- External timing insufficient

**Rollback Procedure:**
1. Use external timing (already specified in enhanced spec)
2. If accuracy insufficient, add RT measurement API:
   ```rust
   pub fn recall_with_latency(&self, cue: &str)
     -> (Vec<(Episode, Confidence)>, Duration)
   ```
3. Budget +0.5 day for API addition

---

## 8. Revised Timeline with Corrections

### Original Estimate: 14 days

**Breakdown:**
- Infrastructure: 3 days (Tasks 001, 011)
- Implementations: 7 days (Tasks 002-007)
- Validations: 3 days (Tasks 008-010)
- Integration: 1 day (Tasks 012-014)

### Revised Estimate: 18-20 days

**Changes:**
- Task 001: 2d → 3d (+1 for architectural corrections)
- Task 008: 2d → 2.5d (+0.5 for higher sample size)
- Task 009: 1d → 2d (+1 for power analysis fixes)
- Task 010: 1d → 3d (+2 for complete protocol specification)
- Task 011: 2d → 5d (+3 for complete redesign)
- Task 014: 1d → 1.5d (+0.5 for quality gates)
- Buffer: +1d for risk mitigation

**Phase-by-Phase Breakdown:**

**Phase 1: Foundation (3 days)**
- Task 001 complete with corrections

**Phase 2: Core + Validations (8 days, parallelized)**
- Track A: 002 (2d) → 008 (2.5d) → 003 (1.5d)
- Track B: 009 (2d, independent)
- Track C: 011 (5d, parallel after 001)
- Track D: 006 (2d) → 007 (1d)

**Phase 3: Interference (5 days)**
- Task 004 (1.5d) → 005 (1.5d) → 010 (3d)

**Phase 4: Integration + Docs (3 days)**
- Task 012 (2d) → 013 (1.5d) → 014 (1.5d, some overlap)

**Total: 18 days (best case) to 20 days (with contingency)**

### Critical Path Duration

**Longest dependency chain:**
001 (3d) → 002 (2d) → 008 (2.5d) → 003 (1.5d) → 004 (1.5d) → 005 (1.5d) → 010 (3d) → 012 (2d) → 013 (1.5d) → 014 (1.5d) = **20 days**

**With perfect parallelization:**
- Tracks B, C, D reduce effective duration
- Realistic: **18 days**

---

## 9. Resource Allocation

### Skill Requirements by Task

**Systems Programming (Rust):**
- Task 001: Advanced (lock-free, atomics, loom)
- Task 002-007: Intermediate
- Task 011: Advanced (ring buffers, unsafe code)

**Psychology/Statistics:**
- Task 008-010: Advanced (experimental design, power analysis)
- Task 009: Intermediate (t-tests, effect sizes)

**Technical Writing:**
- Task 014: Advanced (multiple audiences, quality gates)

### Recommended Assignments

**Track A (Critical Path):** Most experienced Rust engineer
- Tasks: 001 → 002 → 008 (requires both systems + stats knowledge)

**Track B (Validations):** Psychology/stats background
- Tasks: 009, assist with 008, 010

**Track C (Tracing):** Systems engineer
- Task: 011 (ring buffers, performance)

**Track D (Reconsolidation):** Mid-level Rust engineer
- Tasks: 006 → 007

**Integration/Docs:** Technical writer with coding experience
- Tasks: 012-014

### Minimum Viable Team

- 2 senior engineers (one systems, one with psych background)
- 1 mid-level engineer
- 1 technical writer

**Duration with 2-person team:** 25-30 days (tasks sequentialized)
**Duration with 4-person team:** 18-20 days (optimal parallelization)

---

## 10. Success Criteria

### Milestone Acceptance Criteria

**Technical Correctness:**
- [ ] All quality gates pass
- [ ] Zero clippy warnings (`make quality` clean)
- [ ] All loom tests pass (Task 001)
- [ ] All psychology validations within acceptance ranges (Tasks 008-010)

**Psychology Validation:**
- [ ] DRM: 55-65% false recall (p < 0.05)
- [ ] Spacing: 10-50% improvement (p < 0.05)
- [ ] Interference: All three paradigms validate (p < 0.05)

**Performance:**
- [ ] Metrics overhead < 1% (Task 001)
- [ ] Tracing overhead < 100ns/event (Task 011)
- [ ] Psychology tests complete in < 10 minutes

**Documentation:**
- [ ] All code examples compile
- [ ] All parameters accurate
- [ ] All runbooks tested

### Milestone 13 Complete When:

1. **All 14 tasks complete** with acceptance criteria met
2. **Psychology validations pass** (proof of cognitive plausibility)
3. **Documentation quality gates pass** (usable by developers)
4. **Integration tests pass** (system-level correctness)
5. **No P0/P1 issues open** (production-ready)

---

## 11. Communication Plan

### Daily Standups

**Focus:**
- Critical path progress (Task 001 → 002 → 008)
- Blockers (API availability, validation failures)
- Risk status (DRM validation, time simulation)

### Weekly Reviews

**Topics:**
- Quality gate status
- Timeline adjustments
- Risk mitigation effectiveness

### Decision Points

**Day 3:** Task 001 quality gate - proceed to Phase 2?
**Day 8:** DRM validation - run parameter sweep?
**Day 10:** Spacing validation - investigate time simulation?
**Day 15:** Task 010 API check - implement or use external timing?
**Day 20:** Milestone acceptance review

### Escalation Criteria

**Immediate Escalation:**
- Task 001 fails quality gates (blocks all work)
- DRM validation fails after parameter sweep (architecture issue)
- Time simulation artifacts detected (M4 bug)

**Same-Day Escalation:**
- Any HIGH risk materializes
- Critical path delayed > 1 day

**Next-Day Escalation:**
- MEDIUM risk materializes
- Non-critical task blocked

---

## 12. Agent Consultation Map

### When to Consult Which Agent

**systems-architecture-optimizer:**
- Task 001 issues (lock-free correctness)
- Task 011 issues (ring buffer design)
- Performance problems (overhead exceeds budget)

**memory-systems-researcher:**
- Task 008 validation failures (DRM parameters)
- Task 009 validation failures (spacing effect)
- Psychology interpretation questions

**verification-testing-lead:**
- Statistical power questions
- Experimental design issues
- Test methodology validation

**rust-graph-engine-architect:**
- Spreading activation API questions
- Graph algorithm issues
- Integration problems

**technical-communication-lead:**
- Documentation quality issues
- Example clarity problems
- Operations runbook validation

---

## 13. Post-Milestone Actions

### If Milestone Succeeds

**Celebrate:**
- Cognitive patterns validated against psychology literature
- Production-ready observability infrastructure
- Comprehensive documentation

**Follow-ups:**
- Publish validation results (academic paper?)
- Blog post about DRM replication in production system
- Conference talk submission

### If Validations Fail

**Root Cause Analysis:**
1. Which validation failed?
2. Was it statistical power issue or real problem?
3. Parameter sweep results?
4. Consult agent recommendations

**Options:**
1. **Parameter Tuning:** Adjust priming/consolidation parameters (budget +2-3 days)
2. **Architecture Fix:** If fundamental issue (e.g., pattern completion broken)
3. **Scope Reduction:** Mark failing validations as "known limitations"

**Decision Matrix:**

| Failure | Severity | Action |
|---------|----------|--------|
| DRM only | HIGH | Parameter sweep → tuning (+2d) |
| Spacing only | MEDIUM | Investigate M4 time simulation (+1d) |
| Interference only | MEDIUM | Individual paradigm fixes (+1d) |
| DRM + Spacing | CRITICAL | Architecture review (M8, M4) (+5d) |
| All three | BLOCKING | Milestone redesign required |

---

## Appendix A: Task Dependencies Graph

```
Foundation:
    001 (Metrics)
       ↓
    ┌──┴────────────┐
    ↓               ↓
Implementations:    Tracing:
  002 (Priming)     011 (Tracing 5d)
    ↓ ↓ ↓
    | | └→ 006 (Recons Core)
    | |         ↓
    | |     007 (Recons Integ)
    | |
    | └→ 003 (Assoc/Rep)
    |       ↓
    └→ 004 (Proactive)
          ↓
      005 (Retroactive)

Validations:
  008 (DRM) ← depends on 002
  009 (Spacing) ← depends on M4 (independent)
  010 (Interference) ← depends on 004, 005

Integration:
  012 (Integration) ← all above
    ↓
  013 (Testing)
    ↓
  014 (Documentation)
```

---

## Appendix B: Agent Review Summary

### Systems Architecture Review (Margo Seltzer)

**Task 001:**
- CONDITIONAL PASS
- Critical fixes: Remove Arc, fix histogram sum, add loom tests
- Estimated +1 day for corrections

**Task 011:**
- FAIL - requires complete rewrite
- Issues: Unbounded memory, DateTime overhead, wrong tools (Prometheus for events)
- Estimated +3 days for redesign

### Psychology Validation Review (Professor Regehr)

**Task 008:**
- PASS with minor improvements
- Fixes: Increase n=200, add Cohen's d, implement get_embedding()
- Estimated +0.5 day

**Task 009:**
- FAIL - insufficient power
- Fixes: Increase n=200, add t-test, time simulation validation
- Estimated +1 day

**Task 010:**
- FAIL - skeletal specification
- Fixes: Complete protocols for all three paradigms
- Estimated +2 days

### Documentation Review (Julia Evans)

**Task 014:**
- NEEDS ENHANCEMENT
- Improvements: 15+ examples, complete runbooks, quality gates
- Estimated +0.5 day for quality

---

## Appendix C: Pre-Implementation Checklist

### Before Starting Milestone 13

- [ ] Milestone 12 acceptance complete
- [ ] M3 (Spreading Activation) API stable
- [ ] M4 (Temporal Dynamics) time simulation working
- [ ] M8 (Pattern Completion) implemented
- [ ] Benchmark infrastructure ready (Criterion)
- [ ] CI pipeline supports doc tests
- [ ] Loom available for concurrency testing

### Before Starting Task 001

- [ ] Read `SYSTEMS_ARCHITECTURE_REVIEW.md` Task 001 section
- [ ] Understand Arc removal rationale
- [ ] Review atomic_float crate usage
- [ ] Set up loom test environment

### Before Starting Task 008

- [ ] Read `PSYCHOLOGY_VALIDATION_REVIEW.md` Task 008 section
- [ ] Run API compatibility test
- [ ] Generate pre-computed embeddings (Python script)
- [ ] Validate embedding semantic structure

### Before Starting Task 009

- [ ] Read enhanced spec time simulation section
- [ ] Run time simulation validation tests
- [ ] Verify M4 decay functions working
- [ ] Understand power analysis requirements

### Before Starting Task 010

- [ ] Read enhanced spec experimental protocols
- [ ] Check if `recall_with_latency()` API exists
- [ ] Test stimulus generation functions
- [ ] Review statistical analysis requirements

### Before Starting Task 011

- [ ] Read `SYSTEMS_ARCHITECTURE_REVIEW.md` Task 011 section
- [ ] Understand ring buffer SPSC design
- [ ] Review bounded memory strategy
- [ ] Set up profiling tools

---

## Appendix D: Validation Command Reference

### Quick Validation Commands

```bash
# Task 001: Metrics overhead
cargo bench --bench metrics_overhead -- --save-baseline m13
cargo test --lib --no-default-features metrics::zero_overhead

# Task 008: DRM validation
cargo test psychology::api_compatibility
cargo test psychology::drm_embeddings
cargo test psychology::drm_paradigm -- --nocapture

# Task 009: Spacing effect
cargo test psychology::spacing_time_simulation
cargo test psychology::spacing_effect -- --nocapture

# Task 010: Interference suite
cargo test psychology::interference_validation -- --nocapture

# Task 011: Tracing overhead
cargo bench --bench tracing_overhead --features tracing
cargo test --lib --features tracing tracing::memory_bounds

# Quality checks
make quality  # Must pass with zero warnings
cargo test --doc  # All examples compile
```

---

**This roadmap is actionable, unambiguous, and ready for implementation. No guesswork required.**
