# Milestone 13: Planning Summary

**Created:** 2025-10-23
**Planner:** Systems Product Planner (Bryan Cantrill mode)
**Status:** Ready for Implementation

## Executive Summary

Milestone 13 delivers **rigorously validated cognitive psychology phenomena** and **zero-overhead observability infrastructure**. This is not superficial "cognitive branding" - every implementation replicates published empirical research with measurable statistical acceptance criteria.

**Duration:** 16-18 days
**Tasks:** 14 (001-014)
**Critical Dependencies:** M1-M8 (foundation complete)

## Key Design Decisions

### 1. Zero-Cost Abstraction for Metrics

**Decision:** Use conditional compilation (`#[cfg(feature = "monitoring")]`) for metrics, not runtime flags.

**Rationale:**
- When disabled: Compiler eliminates all metrics code (0% overhead, provable via assembly)
- When enabled: Lock-free atomics with cache-line padding (<1% overhead, benchmark-verified)
- Runtime flags have non-zero cost even when "disabled" (branch prediction, cache pollution)

**Validation:**
- Assembly inspection: `objdump -d` shows zero cognitive_patterns symbols when disabled
- Benchmark suite: Criterion measures <1% P99 regression when enabled

### 2. Exact Boundary Conditions, Not Fuzzy Logic

**Decision:** Reconsolidation window boundaries are exact inequalities, not sigmoid curves.

**Rationale:**
- Nader et al. (2000) specify 1-6 hour window - this is experimental finding, not model parameter
- Biology has hard boundaries (protein synthesis windows are time-critical)
- Fuzzy boundaries introduce free parameters that hide poor modeling

**Implementation:**
```rust
if time_since_recall < self.window_start {
    return None; // Exact rejection, not degradation
}
```

### 3. Statistical Validation, Not Intuition

**Decision:** Every cognitive phenomenon has quantitative acceptance criteria from published research.

**Example:**
- DRM false recall: 60% ± 10% (Roediger & McDermott 1995)
- Not: "Should produce some false memories" (unmeasurable)
- Not: "Looks about right" (unscientific)

**Acceptance Process:**
1. Run experiment with n ≥ 100 trials
2. Compute 95% confidence interval
3. Check overlap with published empirical range
4. Reject if outside tolerance (no hand-waving)

### 4. Priming via Activation Spreading, Not Ad-Hoc Boosts

**Decision:** Semantic priming emerges from spreading activation (Collins & Loftus 1975), not arbitrary bonuses.

**Mechanism:**
1. Recall activates node N
2. Activation spreads to neighbors via embedding similarity
3. Spread activation persists with exponential decay (half-life: 500ms)
4. Subsequent recall of neighbor benefits from residual activation

**Why Not Ad-Hoc:**
- "Add 15% to similar concepts" lacks mechanistic grounding
- Spreading activation explains: priming, semantic relatedness, decay timing
- Single mechanism → fewer free parameters → harder to fit noise

## Critical Path Analysis

**Longest Path (Foundation → Validation):**
1. Task 001: Zero-Overhead Metrics (2 days) → Foundation
2. Task 002: Semantic Priming (2 days) → Uses metrics, enables DRM
3. Task 008: DRM Paradigm (2 days) → Uses priming, validates M8 pattern completion
4. Task 013: Integration/Performance (2 days) → Final validation

**Total Critical Path:** 8 days minimum

**Parallel Work Opportunities:**
- Tasks 003-005 (Priming/Interference) can proceed after Task 002
- Tasks 006-007 (Reconsolidation) independent of priming tasks
- Tasks 009-010 (Validation) blocked only by their implementation tasks
- Tasks 011-012 (Observability) parallel to validation

**Realistic Schedule with Parallelization:** 16-18 days

## Risk Assessment

### High-Impact Risks

**Risk 1: DRM Validation Failure (Impact: High, Likelihood: Medium)**

**Scenario:** False recall rate outside 50-70% range despite parameter tuning.

**Root Causes:**
- Semantic priming too weak/strong
- Pattern completion threshold miscalibrated
- Consolidation not extracting semantic patterns (M6 issue)

**Mitigation:**
1. **Early detection:** Implement Task 008 (DRM) immediately after Task 002 (priming)
2. **Parameter sweep:** Automated search over priming_strength ∈ [0.10, 0.25]
3. **Agent consultation:** memory-systems-researcher review if initial attempt fails
4. **Fallback plan:** +2 days budgeted for iterative tuning

**Acceptance:** If after 2 days of tuning still failing, escalate to architectural review (may indicate M8 pattern completion issue, not M13 issue)

**Risk 2: Metrics Overhead Exceeds 1% (Impact: High, Likelihood: Low)**

**Scenario:** Benchmark shows >1% P99 latency regression with monitoring enabled.

**Root Causes:**
- False sharing despite cache-line padding
- Atomic contention on high-frequency counters
- Histogram bucket allocation overhead

**Mitigation:**
1. **Pre-implementation validation:** Profile existing metrics (M6) to verify <1% overhead pattern
2. **Lock-free verification:** Loom tests catch synchronization bugs early
3. **Sampling fallback:** If overhead exceeds budget, implement 10% reservoir sampling
4. **Assembly inspection:** Verify inlining and register allocation via `cargo asm`

**Acceptance:** Sampling at 10% rate acceptable if necessary to meet <1% budget (reduces statistical power but preserves correctness)

### Medium-Impact Risks

**Risk 3: Reconsolidation Boundary Edge Cases (Impact: Medium, Likelihood: Medium)**

**Scenario:** Time zone changes, leap seconds, or negative durations cause boundary violations.

**Mitigation:**
- Property-based testing with quickcheck (arbitrary DateTime generation)
- Explicit handling of `Duration::zero()` and negative durations
- Use monotonic time (`Instant`) for decay, UTC time for boundaries

**Risk 4: Integration with M6 Consolidation (Impact: Medium, Likelihood: Low)**

**Scenario:** Reconsolidated memories conflict with consolidation pipeline.

**Mitigation:**
- Review M6 code before implementing Task 007
- Separate reconsolidation from initial consolidation (different code paths)
- Integration tests for: consolidate → recall → reconsolidate → consolidate

## Success Metrics

### Quantitative (Must Pass)

| Metric | Target | Tolerance | Source |
|--------|--------|-----------|--------|
| DRM false recall | 60% | ±10% | Roediger 1995 |
| Semantic priming RT reduction | 65ms | ±15ms | Neely 1977 |
| Proactive interference | 25% | ±10% | Underwood 1957 |
| Retroactive interference | 20% | ±10% | McGeoch 1942 |
| Fan effect | 50ms/assoc | ±25ms | Anderson 1974 |
| Spacing effect | 30% gain | ±10% | Cepeda 2006 |
| Metrics overhead (enabled) | <1% | Hard limit | Benchmark |
| Metrics overhead (disabled) | 0% | Exact | Assembly |

### Qualitative (Should Achieve)

- **Code quality:** Zero clippy warnings, `make quality` passes
- **Documentation:** All psychology papers cited in code comments
- **Biological plausibility:** memory-systems-researcher approval
- **API clarity:** rust-graph-engine-architect review
- **Operational readiness:** Grafana dashboard deployed

## Implementation Guidance

### For Each Task

**Before Starting:**
1. Read task markdown file completely
2. Review cited research papers (at least abstracts)
3. Understand acceptance criteria (what "done" means)
4. Check dependencies are complete

**During Implementation:**
1. Write tests FIRST (TDD for cognitive phenomena)
2. Use empirical parameters from research (not guesses)
3. Add `#[cfg(feature = "monitoring")]` for all metrics
4. Document WHY, not just WHAT (cite papers in comments)

**Before Marking Complete:**
1. All unit tests pass
2. Acceptance criteria verified
3. Clippy warnings fixed
4. Integration with existing systems tested
5. Performance budgets met (if applicable)

### Parameter Tuning Protocol

When empirical validation fails:

1. **Isolate variable:** Change ONE parameter at a time
2. **Sweep systematically:** Test 5-10 values across plausible range
3. **Document results:** Record all tested values and outcomes
4. **Justify selection:** Final parameter must have empirical or mechanistic justification
5. **Update docs:** If deviating from published values, explain why in code comments

**Anti-Pattern:** "I tried a bunch of values and 0.17 seemed to work" ← Unacceptable

**Correct Pattern:** "Swept priming_strength ∈ [0.10, 0.25] in 0.01 increments. DRM false recall peaked at 0.15 (62% false recall, SE=0.03). Selected 0.15 as default. Neely (1977) reports 10-20% RT reduction; our 15% is mid-range." ← Acceptable

### Agent Consultation Points

**memory-systems-researcher:**
- Before implementing reconsolidation (validate boundary conditions)
- After DRM validation (interpret results, compare to biological data)
- If any psychology validation fails (root cause analysis)

**rust-graph-engine-architect:**
- After Task 001 (review metrics API design)
- After Task 002 (review priming integration with spreading activation)
- Before Task 013 (performance validation strategy)

**verification-testing-lead:**
- Before Task 008 (DRM test design review)
- Before Task 010 (interference validation suite design)
- After Task 013 (interpret performance results)

## Delivered Artifacts

**Code (Primary):**
- `/engram-core/src/cognitive/` - Priming, interference, reconsolidation implementations
- `/engram-core/src/metrics/cognitive_patterns.rs` - Zero-overhead metrics
- `/engram-core/src/tracing/cognitive_events.rs` - Structured tracing
- `/engram-core/tests/psychology/` - Empirical validation test suite

**Documentation (Supporting):**
- `/docs/reference/cognitive_patterns.md` - API reference with citations
- `/docs/explanation/psychology_foundations.md` - Biological basis
- `/docs/operations/cognitive_metrics_tuning.md` - Operational guide
- `/docs/operations/grafana/cognitive_patterns_dashboard.json` - Grafana dashboard

**Benchmarks (Validation):**
- `/engram-core/benches/metrics_overhead.rs` - Overhead validation
- `/engram-core/benches/cognitive_patterns_performance.rs` - Performance suite

## Next Steps

1. **Create remaining task files:** (003-005, 007, 009-014)
   - Use existing tasks (001, 002, 006, 008) as templates
   - Maintain specification rigor and empirical grounding

2. **Begin implementation:**
   - Start with Task 001 (zero-overhead metrics foundation)
   - Validate metrics overhead before building on top
   - Use TodoWrite tool to track progress (14 tasks)

3. **Continuous validation:**
   - Run `make quality` after each task
   - Check diagnostics with `./scripts/engram_diagnostics.sh`
   - Track progress in task markdown files

4. **Milestone completion:**
   - All 14 tasks marked `_complete`
   - All psychology validations passing
   - Documentation merged
   - Grafana dashboard deployed

---

**This milestone is complete when we can hand a skeptical cognitive psychologist the DRM validation results and they say "Yes, this is real memory."**

No shortcuts. No hand-waving. No "close enough."

Science.
