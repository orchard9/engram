# Task 009: Accuracy Validation & Production Tuning

**Status:** Pending
**Priority:** P0 (Critical Path)
**Estimated Effort:** 2 days
**Dependencies:** Task 008 (Metrics & Observability)

## Objective

Validate pattern completion accuracy against ground truth datasets including deliberately corrupted episodes, DRM paradigm for false memories, and serial position curves. Tune production parameters (CA3 sparsity, CA1 threshold, pattern weights) based on empirical accuracy-latency tradeoffs.

## Integration Points

**Uses:**
- All completion components from Tasks 001-007
- `/engram-core/benches/pattern_completion.rs` - Existing benchmarks
- `/engram-core/tests/pattern_completion_tests.rs` - Existing tests

**Creates:**
- `/engram-core/tests/accuracy/` - Ground truth validation suites
- `/engram-core/tests/accuracy/drm_paradigm.rs` - False memory testing
- `/engram-core/tests/accuracy/serial_position.rs` - Position curve validation
- `/docs/tuning/completion_parameters.md` - Parameter tuning guide

## Validation Datasets

### 1. Corrupted Episodes Dataset
- 1000 complete episodes with ground truth
- Corrupt 30%, 50%, 70% of fields randomly
- Measure reconstruction accuracy per corruption level
- Target: >85% accuracy at 30% corruption, >70% at 50%, >50% at 70%

**Research Foundation (Task 001):**
Ensemble methods (Breiman, 1996) predict combining 3+ diverse temporal neighbors reduces error by 20-30%. Field consensus algorithm should achieve target accuracy by aggregating neighbor evidence weighted by similarity and recency.

**Validation Metrics:**
- Precision: Correct reconstructions / total reconstructions
- Recall: Fields reconstructed / fields corrupted
- F1 Score: Harmonic mean of precision and recall
- Per-field accuracy breakdown (what, when, where, who, why)

### 2. DRM Paradigm (False Memories)

**Research Foundation (Task 005):**
Lindsay & Johnson (2000) demonstrated false memory formation when suggested information is plausible. Source monitoring framework prevents confabulation by explicit source tracking and alternative hypothesis generation.

**Experimental Design:**
- Present semantically related episodes (e.g., bed, rest, awake, tired, dream)
- Test completion for critical lure "sleep" (never presented but semantically related)
- Measure false reconstruction rate at different confidence thresholds
- Target: <15% false lure completions at high confidence (>0.7)

**Validation:**
- Source attribution should label lure as "Consolidated" (from semantic patterns) not "Recalled"
- Alternative hypotheses (Task 005) should include non-lure options
- Metacognitive confidence (Task 006) should be lower for lure completions

**Success Criteria:**
- False lure rate <15% at confidence >0.7 (CA1 threshold filters implausible)
- Source attribution correctly identifies lures as consolidated/imagined (not recalled)
- Alternative hypotheses include ground truth >70% of time

### 3. Serial Position Curves

**Research Foundation:**
Temporal weighting from Task 001 implements recency bias. Recent neighbors receive higher weight due to quadratic temporal decay (recency_exponent = 2.0).

**Experimental Design:**
- Episodes in temporal sequence (1-20 items)
- Test completion for primacy (positions 1-3), middle (positions 8-12), and recency (positions 18-20) items
- Validate matches human serial position curve (Murdock, 1962)
- Target: Recency effect >10% accuracy boost, primacy effect >5%

**Expected Pattern:**
```
Accuracy:
Primacy (1-3):   75-80% (consolidation strengthens early items)
Middle (8-12):   65-70% (baseline)
Recency (18-20): 80-85% (temporal proximity)
```

**Mechanism Validation:**
- Recency effect driven by temporal_distance weighting in Task 001
- Primacy effect driven by consolidation strength (more rehearsal) in Task 003
- Curve shape should match human data (U-shaped with stronger recency)

## Parameter Tuning Strategy

### Critical Parameters

**CA3 Sparsity (default 0.05, from Task 002):**
- Sweep: [0.02, 0.03, 0.05, 0.07, 0.10]
- Measure: Convergence rate, reconstruction accuracy, false positive rate
- Select: Highest accuracy with >95% convergence rate

**Biological Constraint (Marr, 1971):** Sparse coding in CA3 for pattern separation. Literature suggests 2-10% sparsity optimal for balancing capacity and convergence.

**Expected Result:** 5% sparsity (default) achieves balance. Lower sparsity (2-3%) increases capacity but slower convergence. Higher sparsity (7-10%) faster convergence but more false positives.

**CA1 Threshold (default 0.7, from Task 002):**
- Sweep: [0.5, 0.6, 0.7, 0.8, 0.9]
- Measure: Precision (correct completions), recall (completion rate)
- Select: F1-optimal threshold (balance precision/recall)

**Trade-off (CA1 Gating, Task 002):**
- Low threshold (0.5): High recall, low precision (many completions, some incorrect)
- High threshold (0.9): Low recall, high precision (few completions, mostly correct)
- Target: F1-optimal around 0.7 for balanced performance

**Calibration Integration (Task 006):** Threshold choice affects calibration. Lower threshold increases over-confidence. Monitor calibration error across threshold sweep.

**Pattern Weight (default 0.4, from Task 004):**
- Sweep: [0.2, 0.3, 0.4, 0.5, 0.6]
- Measure: Reconstruction accuracy, confabulation rate
- Select: Highest accuracy with <10% confabulation

**Research Foundation (Task 004):** Adaptive weighting balances local (temporal neighbors) vs global (semantic patterns). Fixed pattern weight = 0.4 means 40% global, 60% local.

**Expected Result:** Higher pattern weight (0.5-0.6) improves accuracy for sparse cues but increases confabulation risk (DRM paradigm). Lower weight (0.2-0.3) reduces confabulation but decreases accuracy on degraded episodes.

**Monitoring (Task 008):** Track `engram_patterns_used_per_completion` to measure global pattern reliance.

**Num Hypotheses (default 3, from Task 005):**
- Sweep: [1, 2, 3, 5, 10]
- Measure: Top-K coverage, latency
- Select: Minimum K with >70% ground truth in top-K

**Metacognitive Trade-off (Task 005):** More hypotheses increase coverage but add latency. System 2 reasoning (Kahneman, 2011) requires diverse alternatives for deliberative checking.

**Expected Result:** 3 hypotheses achieves >70% coverage with minimal latency (<5ms, Task 005 acceptance criterion). Diminishing returns beyond 5 hypotheses.

### Tuning Process

1. **Baseline:** Run full validation suite with default parameters
   - Corrupted episodes dataset (30%, 50%, 70% corruption)
   - DRM paradigm (false memory rate)
   - Serial position curve (primacy/recency effects)
   - Collect 1000+ completions with ground truth for calibration

2. **Single-Variable Sweeps:** Vary each parameter independently, measure accuracy/latency
   - CA3 sparsity sweep: 5 values × 200 completions = 1000 trials
   - CA1 threshold sweep: 5 values × 200 completions = 1000 trials
   - Pattern weight sweep: 5 values × 200 completions = 1000 trials
   - Num hypotheses sweep: 5 values × 200 completions = 1000 trials

3. **Pareto Frontier:** Identify parameter sets on accuracy-latency Pareto frontier
   - Plot accuracy vs latency for all parameter combinations
   - Pareto-optimal: No parameter set achieves both higher accuracy AND lower latency
   - Select candidates: Top-3 points on frontier

4. **Workload-Specific:** Tune for sparse cues vs rich cues separately
   - **Sparse cues (30% complete):** May benefit from higher pattern weight (more global)
   - **Rich cues (70% complete):** May benefit from lower pattern weight (more local)
   - Validate both workloads separately

5. **Isotonic Regression Calibration (Task 006 research):**
   - Collect raw confidence scores and actual accuracy for 1000+ completions
   - Train isotonic regression mapping: raw_confidence → calibrated_probability
   - Ensure monotonicity: higher raw score → higher calibrated probability
   - Validate: Brier score <0.08, calibration error <8% per bin

6. **A/B Test:** Deploy top-2 parameter sets to production subsets
   - Group A: Default parameters (control)
   - Group B: Pareto-optimal parameters (treatment)
   - Monitor metrics (Task 008): accuracy, latency, calibration error, false memory rate
   - Duration: 1 week

7. **Production Selection:** Choose based on real-world accuracy metrics
   - Compare A/B test results
   - Select parameter set with highest accuracy and acceptable latency (<25ms P95)
   - Document decision in ADR format
   - Deploy to all production instances

## Acceptance Criteria

1. **Reconstruction Accuracy:**
   - >85% accuracy at 30% corruption
   - >70% accuracy at 50% corruption
   - >50% accuracy at 70% corruption

2. **False Memory Control:**
   - <15% false lure completions (DRM paradigm)
   - <10% confabulations at high confidence (>0.7)
   - Source attribution correctly labels reconstructions

3. **Biological Plausibility:**
   - Serial position curve matches human data (Murdock, 1962)
   - Recency effect >10% accuracy boost
   - Primacy effect >5% accuracy boost

4. **Production Parameters:**
   - Selected parameters on Pareto frontier (no accuracy loss for latency)
   - <8% calibration error with tuned parameters
   - >95% convergence rate

5. **Documentation:**
   - Parameter tuning guide with empirical results
   - Per-workload recommendations (sparse vs rich cues)
   - A/B test results and production selection rationale

## Testing Strategy

**Ground Truth Validation:** 1000+ episodes with known correct completions; measure accuracy per corruption level

**DRM Paradigm:** 100+ semantic association lists; measure false lure rate

**Serial Position:** 50+ temporal sequences; validate primacy/recency effects

**Parameter Sweeps:** Automated grid search across parameter space; Pareto frontier analysis

**Human Evaluation:** 100 random completions rated by humans (5-point plausibility scale); target >75% acceptable (≥3/5)

## Risk Mitigation

**Risk: Ground truth accuracy <80%**
- **Mitigation:** Tune CA3 sparsity and CA1 threshold based on empirical sweeps
- **Contingency:** Reduce CA1 threshold to 0.6 (higher recall, lower precision)

**Risk: False memory rate >20%**
- **Mitigation:** Strengthen statistical significance filters (p < 0.001)
- **Contingency:** Require minimum 5 source episodes for pattern contribution

**Risk: Serial position curve doesn't match human data**
- **Mitigation:** Adjust temporal weighting decay exponent
- **Contingency:** Document deviation and biological plausibility limits

## Implementation Notes

1. Use existing `tests/pattern_completion_tests.rs` as foundation
2. Generate corrupted datasets deterministically (seeded RNG)
3. Cache parameter sweep results for iterative tuning
4. Plot Pareto frontiers with Plotters crate
5. Document all tuning decisions in ADR format

## Success Criteria Validation

- [ ] Corruption accuracy targets met (85%/70%/50%)
- [ ] False memory rate <15% (DRM paradigm)
- [ ] Serial position curve matches human data
- [ ] Production parameters on Pareto frontier
- [ ] Calibration error <8% with tuned parameters
- [ ] Human evaluation >75% acceptable
- [ ] Tuning guide documented with empirical results
- [ ] All validation tests pass

---

*This task establishes production-ready pattern completion validated against cognitive psychology benchmarks and tuned for optimal accuracy-latency tradeoffs in real-world deployments.*
