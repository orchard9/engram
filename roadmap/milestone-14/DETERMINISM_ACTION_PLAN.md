# Determinism Resolution: 3-Week Action Plan

**Status**: Ready for Implementation
**Target Completion**: 3 weeks (21 days)
**Owner**: TBD
**Reviewer**: Randy O'Reilly (Memory Systems)
**Priority**: CRITICAL BLOCKER for M14

---

## Quick Summary

The consolidation determinism audit confirms the systems-product-planner's concern is valid. Current consolidation produces **non-deterministic clusters** due to:

1. No tie-breaking in similarity comparisons
2. Floating-point non-associativity
3. Unstable sorting with equal scores
4. Order-dependent pattern merging

**Good news**: Determinism is achievable in 3 weeks without sacrificing biological plausibility.

**Solution**: Implement 5 targeted fixes to make hierarchical agglomerative clustering fully deterministic.

---

## Week 1: Implementation

### Day 1-2: Core Clustering Determinism

**File**: `/Users/jordan/Workspace/orchard9/engram/engram-core/src/consolidation/pattern_detector.rs`

**Tasks**:
1. Implement stable episode sorting (sort by ID before clustering)
2. Add deterministic tie-breaking to `find_most_similar_clusters_centroid`
3. Write `cluster_tiebreaker` helper (lexicographic episode ID ordering)

**Code Changes**: ~40 lines

**Acceptance**:
- Clustering produces identical results on repeated runs
- Unit test: `test_cluster_determinism_basic` passes

---

### Day 3: Floating-Point Determinism

**File**: `/Users/jordan/Workspace/orchard9/engram/engram-core/src/consolidation/pattern_detector.rs`

**Tasks**:
1. Implement Kahan summation algorithm
2. Replace `average_embeddings` to use Kahan summation
3. Verify no accuracy regression (semantic pattern quality unchanged)

**Code Changes**: ~30 lines

**Acceptance**:
- Cross-platform test produces identical centroids (x86_64 vs ARM64)
- Performance overhead <10%

---

### Day 4: Stable Sorting

**Files**:
- `/Users/jordan/Workspace/orchard9/engram/engram-core/src/consolidation/dream.rs`
- `/Users/jordan/Workspace/orchard9/engram/engram-core/src/completion/consolidation.rs`

**Tasks**:
1. Grep for all `partial_cmp` + `unwrap_or(Equal)` in consolidation code
2. Add `.then_with(|| a.id.cmp(&b.id))` for deterministic tie-breaking
3. Update all sorting in episode selection

**Code Changes**: ~20 lines (multiple files)

**Acceptance**:
- Same-score episodes sort in deterministic order
- Test: `test_episode_selection_determinism` passes

---

### Day 5: Pattern Merging Determinism

**File**: `/Users/jordan/Workspace/orchard9/engram/engram-core/src/consolidation/pattern_detector.rs`

**Tasks**:
1. Sort patterns by ID before merging in `merge_similar_patterns`
2. Verify merge order is deterministic

**Code Changes**: ~10 lines

**Acceptance**:
- Pattern merging produces identical results on repeated runs
- Test: `test_pattern_merge_determinism` passes

---

## Week 2: Testing

### Day 6-7: Property-Based Tests

**File**: `/Users/jordan/Workspace/orchard9/engram/engram-core/src/consolidation/pattern_detector.rs` (test module)

**Tasks**:
1. Implement `test_consolidation_determinism_property` with proptest
2. Generate 100 arbitrary episodes, run consolidation 1000 times
3. Assert all runs produce identical pattern signatures
4. Add test fixtures for reproducibility

**Code Changes**: ~80 lines (tests)

**Dependencies**: Add `proptest` to dev-dependencies

**Acceptance**:
- Test passes with 1000 iterations in <60 seconds
- Zero variance in pattern signatures

---

### Day 8: Cross-Platform Validation

**File**: `/Users/jordan/Workspace/orchard9/engram/engram-core/src/consolidation/pattern_detector.rs` (test module)

**Tasks**:
1. Create reference episode dataset (save to `fixtures/determinism_test_episodes.json`)
2. Run consolidation on macOS ARM64, compute signature
3. Run on Linux x86_64 (CI), verify same signature
4. Document expected signature in test

**Code Changes**: ~40 lines (tests + fixtures)

**Acceptance**:
- Identical signatures across x86_64, ARM64
- CI test fails if signature changes (regression detection)

---

### Day 9: Gossip Convergence Simulation

**File**: `/Users/jordan/Workspace/orchard9/engram/engram-core/src/consolidation/pattern_detector.rs` (test module)

**Tasks**:
1. Implement `test_gossip_convergence_with_determinism`
2. Simulate 5 nodes with shuffled episode arrival order
3. Verify all nodes produce identical patterns

**Code Changes**: ~60 lines (tests)

**Acceptance**:
- All nodes converge to identical pattern sets
- Zero vector clock conflicts in simulation

---

### Day 10: Performance Regression Testing

**File**: `/Users/jordan/Workspace/orchard9/engram/benches/consolidation_determinism.rs` (NEW)

**Tasks**:
1. Create Criterion benchmark for pattern detection
2. Measure baseline (current non-deterministic implementation)
3. Measure deterministic implementation
4. Assert <10% overhead

**Code Changes**: ~100 lines (benchmarks)

**Dependencies**: Criterion already in project

**Acceptance**:
- Deterministic implementation <10% slower
- Benchmark runs in CI (performance tracking)

---

## Week 3: Validation & Integration

### Day 11-12: Production Data Validation

**Tasks**:
1. Export 1000 real episodes from production/staging MemoryStore
2. Run determinism tests on production data
3. Verify pattern quality unchanged (human review of semantic patterns)
4. Check for edge cases (empty episodes, NaN embeddings, etc.)

**Acceptance**:
- 1000-run determinism test passes on production data
- Semantic pattern quality equivalent to baseline
- No edge case failures

---

### Day 13: Edge Case Hardening

**Tasks**:
1. Fix any edge cases discovered on Day 11-12
2. Add specific tests for edge cases
3. Re-run full test suite

**Code Changes**: Variable (depends on findings)

**Acceptance**:
- All edge cases have regression tests
- Full test suite passes (1000+ iterations)

---

### Day 14-15: Documentation & Handoff

**Tasks**:
1. Update `/docs/architecture/consolidation.md` with determinism guarantees
2. Document tie-breaking semantics (biological justification)
3. Write runbook for validating determinism in distributed cluster
4. Code review with team
5. Merge to dev branch

**Deliverables**:
- Architecture documentation updated
- Runbook: "Validating Consolidation Determinism in Production"
- PR merged with full test coverage

**Acceptance**:
- Documentation approved by Randy O'Reilly (biological plausibility)
- Code review approved (2+ reviewers)
- All tests passing in CI

---

## Success Criteria

### Functional

- [ ] Property-based test: 1000 runs produce identical output
- [ ] Cross-platform test: x86_64 and ARM64 produce same signature
- [ ] Gossip simulation: 5 nodes converge to identical patterns
- [ ] Edge cases: All production data edge cases handled

### Performance

- [ ] Deterministic overhead <10% vs. baseline
- [ ] Memory footprint unchanged
- [ ] M6 consolidation benchmarks no regression

### Biological Plausibility

- [ ] Semantic pattern quality unchanged (human evaluation)
- [ ] Pattern coherence scores equivalent to baseline
- [ ] Neuroscience expert approval (Randy O'Reilly sign-off)

### Integration

- [ ] No breaking changes to existing APIs
- [ ] Backwards compatible with existing semantic patterns
- [ ] M13 cognitive patterns tests still pass

---

## Risk Mitigation

### Risk 1: Floating-Point Edge Cases (Prob: 30%)

**Mitigation**:
- Use battle-tested Kahan summation (not custom implementation)
- Extensive testing with edge values (NaN, Inf, subnormals)
- Fallback: Use fixed-point arithmetic if FP proves intractable (unlikely)

### Risk 2: Performance Regression (Prob: 15%)

**Mitigation**:
- Profile early (Day 3)
- Optimize hot paths (Kahan summation can be SIMD-accelerated)
- If >10% overhead: Use Kahan only for centroid computation, not all FP ops

### Risk 3: Missed Non-Determinism Sources (Prob: 25%)

**Mitigation**:
- Extensive property-based testing (10,000+ episodes)
- Fuzz testing with random inputs
- Production data validation (real-world edge cases)

### Risk 4: Biological Plausibility Concerns (Prob: 5%)

**Mitigation**:
- Document neuroscience justification in code comments
- Randy O'Reilly review and approval
- Semantic pattern quality validation (human evaluation)

---

## Rollout Plan

### Phase 1: Feature Flag (Day 16-17)

**After Week 3 completion**:
1. Merge deterministic consolidation behind feature flag `deterministic_consolidation`
2. Default: OFF (use existing non-deterministic implementation)
3. Enable in staging environment for validation

### Phase 2: Staging Validation (Week 4-5)

**Parallel with M13 completion**:
1. Run staging cluster with deterministic consolidation enabled
2. Monitor semantic pattern quality (human review)
3. Compare convergence behavior (single-node vs. future distributed)
4. Collect performance metrics (P50/P95/P99 latencies)

### Phase 3: Production Rollout (Week 6)

**After staging validation**:
1. Enable deterministic consolidation in production (default: ON)
2. Monitor for 7 days (consolidation quality, performance)
3. If issues: Rollback via feature flag (zero downtime)
4. If successful: Remove feature flag, make deterministic the only path

---

## Dependencies

### External

- [ ] proptest crate (already in dev-dependencies)
- [ ] Criterion benchmarking (already in project)
- [ ] Cross-platform CI (x86_64, ARM64) - verify available

### Internal

- [ ] M13 completion (reconsolidation core) - BLOCKING for integration testing
- [ ] Single-node performance baselines - needed for regression testing
- [ ] Staging environment - needed for validation

### Can Proceed Without

- M14 distributed implementation (determinism is prerequisite)
- 7-day soak test (validate determinism first, then soak)

---

## Deliverables

### Code

1. Deterministic pattern detector (`pattern_detector.rs` ~130 LOC changed)
2. Property-based tests (~180 LOC new)
3. Benchmarks (~100 LOC new)
4. Cross-platform validation tests (~100 LOC new)

### Documentation

1. Architecture doc: Consolidation Determinism Guarantees
2. Runbook: Validating Determinism in Production
3. Neuroscience justification (inline code comments)
4. Performance analysis report

### Artifacts

1. Reference episode dataset (`fixtures/determinism_test_episodes.json`)
2. Expected pattern signatures (per-platform)
3. Benchmark results (baseline vs. deterministic)

---

## Go/No-Go Decision Points

### End of Week 1 (Day 5)

**Question**: Is implementation complete and unit tests passing?

**Go Criteria**:
- All 5 code changes implemented
- Basic determinism tests passing
- No show-stopping bugs

**No-Go Action**: Extend implementation by 2-3 days, compress testing phase

---

### End of Week 2 (Day 10)

**Question**: Is determinism proven with high confidence?

**Go Criteria**:
- Property-based tests passing (1000 runs)
- Cross-platform tests passing
- Performance overhead <10%

**No-Go Action**: Deep-dive on failures, may require algorithm rework (add 1 week)

---

### End of Week 3 (Day 15)

**Question**: Is deterministic consolidation ready for staging?

**Go Criteria**:
- All success criteria met
- Documentation complete
- Code review approved

**No-Go Action**: Defer M14, continue iteration until criteria met

---

## Communication Plan

### Stakeholders

1. Engineering team (implementation updates)
2. Randy O'Reilly (biological plausibility review)
3. Product (M14 timeline implications)
4. Users (transparency on consolidation algorithm changes)

### Cadence

- **Daily standups**: Progress, blockers
- **End of Week 1**: Implementation demo
- **End of Week 2**: Testing results review
- **End of Week 3**: Go/no-go decision for staging

### Artifacts

- Weekly summary email (progress, risks, timeline)
- Final report: "Consolidation Determinism: Implementation and Validation"
- Blog post (optional): "How We Made Memory Consolidation Deterministic"

---

## Next Steps (Immediate)

1. **Review this plan** with engineering team (1 hour meeting)
2. **Assign owner** for 3-week sprint
3. **Set up tracking** (GitHub project board or similar)
4. **Kick off Week 1** (start implementation)
5. **Schedule reviews**:
   - End of Week 1: Implementation review
   - End of Week 2: Testing review
   - End of Week 3: Go/no-go decision

**Target Start Date**: ASAP (this week)
**Target Completion**: 3 weeks from start
**M14 Unblock Date**: 4-5 weeks from start (after staging validation)

---

**Document Owner**: Randy O'Reilly (Memory Systems Researcher)
**Last Updated**: 2025-10-31
**Status**: READY FOR APPROVAL
