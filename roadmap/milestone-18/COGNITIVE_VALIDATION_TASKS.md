# Milestone 18: Cognitive Validation Tasks for Dual Memory Architecture

## Overview

These tasks provide comprehensive psychological validation of M17's dual memory implementation, covering the core empirical phenomena that must emerge from a biologically-plausible episodic-semantic memory system. Each task replicates a classic study with quantitative correlation targets.

## Task List

### Task 017: Semantic Priming Validation (Neely 1977)
**Status**: Pending
**Duration**: 5 days
**Priority**: High (foundational spreading activation validation)

**Objective**: Validate concept-mediated spreading activation produces 40-80ms priming at 250ms SOA

**Key Metrics**:
- Overall correlation: r > 0.80 with Neely (1977) RT data
- Priming effect at 250ms SOA: 40-80ms facilitation
- Automatic spreading: <300ms SOA shows priming
- Fan effect interaction: High-fan concepts show 30-50% reduced priming

**Dataset**: Neely's 26 word pairs (DOCTOR→NURSE, BREAD→BUTTER, etc.)

**Validation Criteria**:
- Pearson correlation r > 0.80
- Cohen's d > 0.5 for related vs unrelated
- Temporal dynamics match automatic/controlled processing boundaries

---

### Task 018: Anderson Fan Effect Validation (Person-Location Paradigm)
**Status**: Pending
**Duration**: 4 days
**Priority**: High (validates fan effect implementation in M17 Task 007)

**Objective**: Replicate Anderson (1974) linear RT increase (~70ms per association) with r > 0.85 correlation

**Key Metrics**:
- Regression slope: 70ms ± 10ms per association
- Fan 1 baseline: 1159ms ± 50ms
- Fan 2: +77ms from baseline
- Fan 3: +69ms from fan 2

**Dataset**: 26 person-location facts with controlled fan 1-5

**Validation Criteria**:
- Correlation r > 0.85
- Linear r² > 0.70
- Asymmetric spreading: episode→concept 1.2x stronger

---

### Task 019: Consolidation Timeline Validation (Takashima 2006)
**Status**: Pending
**Duration**: 5 days
**Priority**: High (validates episodic-to-semantic transformation)

**Objective**: Validate hippocampal-neocortical transfer over 30min-90day timeline

**Key Metrics**:
- Hippocampal trajectory: r > 0.75 with Takashima fMRI data
- Neocortical trajectory: r > 0.75
- Day 1 neocortical increase: 2-5% from baseline
- Month 3 hippocampal reduction: 30-50%

**Dataset**: 60 face-location associations tested at 5 timepoints

**Validation Criteria**:
- Correlation r > 0.75 for both trajectories
- Semanticization ratio increases from 0.1 to 0.3+ over 90 days
- Positive neocortical slope (increasing consolidation strength)

---

### Task 020: Retrograde Amnesia Gradient (Ribot's Law)
**Status**: Pending
**Duration**: 4 days
**Priority**: High (validates consolidation protects against "lesion")

**Objective**: Replicate temporal gradient where recent memories show 70-90% impairment, remote 0-20%

**Key Metrics**:
- Correlation: r > 0.80 with Squire & Alvarez (1995) data
- Exponential decay τ: 40-80 days (expected ~60 days)
- Day 1 impairment: 75-95%
- Year 1 impairment: 0-15%

**Dataset**: Memories at Day 1, Week 1, Month 1, Month 3, Year 1

**Validation Criteria**:
- Exponential fit r² > 0.80
- Monotonic gradient: recent > remote
- Concepts more resistant than episodes

---

### Task 021: DRM False Memory Paradigm (Roediger & McDermott 1995)
**Status**: Pending (task file to be created)
**Duration**: 4 days
**Priority**: Medium (validates schema-based reconstruction)

**Objective**: Demonstrate 40-55% false recall of critical lures via semantic spreading

**Key Metrics**:
- False recall rate: 40-55% for critical lures
- Veridical recall: 60-75% for studied items
- Related lure false alarm: 20-30%
- Unrelated lure false alarm: <5%

**Dataset**: 15 DRM lists (sleep, chair, doctor, etc.)

**Validation Criteria**:
- Critical lure rate within 40-55% range
- Correlation r > 0.70 with Roediger & McDermott Table 2
- Schema-consistent bias evident in reconstruction

---

### Task 022: Spacing Effect Validation (Cepeda et al. 2006 Meta-Analysis)
**Status**: Pending (task file to be created)
**Duration**: 3 days
**Priority**: Medium (validates rehearsal and consolidation interaction)

**Objective**: Demonstrate distributed practice advantage over massed practice

**Key Metrics**:
- Spacing benefit: 15-30% retention advantage at 1-week delay
- Optimal spacing: ~10-20% of retention interval
- Inverted-U function: Too short or too long spacing suboptimal

**Dataset**: Word pairs with spacing intervals: 0s, 1m, 5m, 1h, 1d

**Validation Criteria**:
- Distributed > massed by at least 15% at 1-week test
- Inverted-U function matches Cepeda meta-analysis
- Consolidation benefit evident in overnight retention

---

### Task 023: Pattern Completion Accuracy (Nakazawa et al. 2002)
**Status**: Pending (task file to be created)
**Duration**: 3 days
**Priority**: Medium (validates CA3-like pattern completion in concepts)

**Objective**: Demonstrate >65% completion accuracy with 65% cue overlap

**Key Metrics**:
- Completion threshold: 60-70% cue overlap
- Above threshold accuracy: >80%
- Below threshold accuracy: <40%
- Graceful degradation: Linear decline 40-60% overlap

**Dataset**: Artificial patterns with controlled overlap

**Validation Criteria**:
- Matches CA3 attractor dynamics from Nakazawa (2002)
- Coherence threshold 0.65 from M17 Task 004 validated
- Sharp transition at pattern separation boundary

---

### Task 024: Reconsolidation Dynamics (Lee 2008)
**Status**: Pending (task file to be created)
**Duration**: 4 days
**Priority**: Medium (validates existing reconsolidation module with dual memory)

**Objective**: Demonstrate memory updating via retrieval-induced plasticity

**Key Metrics**:
- Reactivation window: 3-6 hours post-retrieval
- Enhancement: 10-20% retention boost with additional learning
- Interference: Disruption during window causes 30-50% impairment

**Dataset**: Word pairs with retrieval + interference/enhancement

**Validation Criteria**:
- Temporal window matches Lee (2008) behavioral data
- Integration with M17 consolidation preserves reconsolidation effects
- Binding strength updates during reactivation

---

## Implementation Strategy

### Phase 1: Core Spreading & Fan Effects (Tasks 017-018)
**Duration**: Week 1
**Rationale**: Validates M17 Task 007 (fan effect spreading) which is currently in progress

### Phase 2: Consolidation Dynamics (Tasks 019-020)
**Duration**: Week 2
**Rationale**: Validates M17 Tasks 004-006 (concept formation, binding, consolidation integration)

### Phase 3: Schema & Reconstruction (Tasks 021, 023)
**Duration**: Week 3
**Rationale**: Tests emergent properties of dual memory for false memory and pattern completion

### Phase 4: Learning Phenomena (Tasks 022, 024)
**Duration**: Week 4
**Rationale**: Validates interaction of consolidation with spacing and reconsolidation

### Phase 5: Integration & Reporting
**Duration**: Week 5
**Tasks**: Cross-validation, effect size analysis, dashboard creation

---

## Success Metrics

### Overall Validation Criteria

1. **Correlation Threshold**: r > 0.75 for all primary phenomena
2. **Effect Size Matching**: Cohen's d within ±0.3 of published values
3. **Temporal Accuracy**: Timing windows match human data (±20%)
4. **Statistical Power**: N > 100 trials per condition for stable estimates
5. **Reproducibility**: <5% variance across runs with fixed seeds

### Quantitative Targets by Phenomenon

| Phenomenon | Target Correlation | Effect Size | Timing Window |
|------------|-------------------|-------------|---------------|
| Semantic Priming | r > 0.80 | d > 0.5 | 250ms SOA |
| Fan Effect | r > 0.85 | d > 1.0 | 70ms/assoc |
| Consolidation | r > 0.75 (dual) | Linear | 30min-90d |
| Retrograde Amnesia | r > 0.80 | Exponential | 1d-1y gradient |
| DRM False Memory | r > 0.70 | 40-55% | Immediate |
| Spacing Effect | r > 0.70 | 15-30% | 1-week test |
| Pattern Completion | r > 0.75 | >80% @ 65% | Threshold |
| Reconsolidation | r > 0.70 | 10-20% | 3-6h window |

---

## CI/CD Integration

### Automated Validation Pipeline

1. **Nightly Runs**: Full validation suite runs every night (4-5 hours)
2. **PR Checks**: Smoke tests (1 trial per condition) on pull requests (<5 minutes)
3. **Release Gates**: Complete validation required before version tagging
4. **Performance Regression**: Validation tests also check <5% latency regression

### Failure Modes & Alerts

- **Correlation Drop**: Alert if any test drops below 0.70 (warning) or 0.60 (failure)
- **Effect Size Mismatch**: Alert if Cohen's d deviates by >0.5 from expected
- **Timing Violations**: Alert if temporal dynamics shift by >30%
- **Determinism Failure**: Alert if same-seed runs vary by >10%

---

## Testing Infrastructure

### Required Components

1. **Stimulus Datasets**: JSON files with published experimental stimuli
2. **Statistical Analysis**: Pearson correlation, linear/exponential regression, Cohen's d
3. **Visualization**: Matplotlib/Plotly graphs for each validation test
4. **Reporting**: Markdown reports with pass/fail status and deviation analysis
5. **CI Integration**: GitHub Actions workflow for nightly validation

### File Structure

```
engram-core/
├── tests/
│   └── cognitive_validation/
│       ├── datasets/
│       │   ├── neely_1977_stimuli.json
│       │   ├── anderson_1974_stimuli.json
│       │   ├── consolidation_schedule.json
│       │   ├── amnesia_gradients.json
│       │   └── drm_word_lists.json
│       ├── semantic_priming_neely.rs
│       ├── anderson_fan_effect.rs
│       ├── consolidation_timeline.rs
│       ├── retrograde_amnesia.rs
│       ├── drm_false_memory.rs
│       ├── spacing_effect.rs
│       ├── pattern_completion.rs
│       └── reconsolidation_dynamics.rs
└── src/
    └── cognitive/
        └── validation/
            ├── priming_metrics.rs
            ├── fan_validation.rs
            ├── consolidation_metrics.rs
            ├── lesion_simulator.rs
            └── statistical_analysis.rs
```

---

## Dependencies

### M17 Task Dependencies

- **Task 017**: Requires M17 Tasks 001-007 (dual memory types, spreading, fan effect)
- **Task 018**: Requires M17 Tasks 001-007 (dual memory types, bindings, fan effect)
- **Task 019**: Requires M17 Tasks 001-006 (dual memory types, consolidation)
- **Task 020**: Requires M17 Tasks 004-006 (concept formation, bindings, consolidation)

### External Dependencies

- **Statistical Libraries**: `statrs` or similar for correlation, regression
- **Embedding Generation**: Sentence-transformers or deterministic hash-based
- **Time Simulation**: Fast-forward time for consolidation cycles
- **Visualization**: `plotters` or export to Python matplotlib

---

## Risk Mitigation

### Technical Risks

1. **Stochastic Variability**: Mitigated by high trial counts (N>100) and fixed seeds
2. **Parameter Sensitivity**: Test robustness across ±20% parameter variation
3. **Computational Cost**: Optimize with parallel trial execution, cached embeddings
4. **Determinism**: Ensure M17 spreading activation is deterministic with seeds

### Scientific Risks

1. **Correlation Failure**: If r < 0.70, identify which parameters need tuning
2. **Effect Size Mismatch**: May indicate missing mechanisms (e.g., strategic factors)
3. **Timing Violations**: Could reveal incorrect decay rates or spreading speeds
4. **False Positives**: Use Bonferroni correction for multiple comparisons

---

## Deliverables

### Code Deliverables

1. 8 test files (`.rs`) implementing each validation task
2. 5 JSON stimulus datasets from published studies
3. Statistical analysis module with correlation, regression, effect size functions
4. Visualization module for generating validation graphs
5. CI/CD workflow for nightly validation runs

### Documentation Deliverables

1. Validation report generator (Markdown format)
2. Per-task detailed specifications (this document expands into 8 task files)
3. Troubleshooting guide for correlation failures
4. User guide for adding new validation tests
5. Publication-ready figures for cognitive validation

### Validation Deliverables

1. Proof of r > 0.75 correlation for all core phenomena
2. Effect size analysis matching published Cohen's d values
3. Temporal dynamics validation across ms-to-months timescales
4. Cross-phenomenon correlation matrix (e.g., fan effect predicts priming?)
5. Performance benchmarks: validation suite completes in <5 hours

---

## Timeline Summary

| Phase | Tasks | Duration | Key Milestone |
|-------|-------|----------|---------------|
| Phase 1 | 017-018 | Week 1 | Spreading activation validated |
| Phase 2 | 019-020 | Week 2 | Consolidation validated |
| Phase 3 | 021, 023 | Week 3 | Schema effects validated |
| Phase 4 | 022, 024 | Week 4 | Learning phenomena validated |
| Phase 5 | Integration | Week 5 | Complete validation suite |

**Total Duration**: 5 weeks (1.25 months)

---

## References

### Core Papers (Replicated in Tasks 017-024)

1. Neely, J. H. (1977). Semantic priming and retrieval from lexical memory. *Journal of Experimental Psychology: General*, 106(3), 226-254.
2. Anderson, J. R. (1974). Retrieval of propositional information from long-term memory. *Cognitive Psychology*, 6(4), 451-474.
3. Takashima, A., et al. (2006). Declarative memory consolidation in humans. *PNAS*, 103(3), 756-761.
4. Squire, L. R., & Alvarez, P. (1995). Retrograde amnesia and memory consolidation. *Current Opinion in Neurobiology*, 5(2), 169-177.
5. Roediger, H. L., & McDermott, K. B. (1995). Creating false memories. *Journal of Experimental Psychology: Learning, Memory, and Cognition*, 21(4), 803-814.
6. Cepeda, N. J., et al. (2006). Distributed practice in verbal recall tasks. *Psychological Bulletin*, 132(3), 354-380.
7. Nakazawa, K., et al. (2002). Requirement for hippocampal CA3 NMDA receptors. *Science*, 297(5579), 211-218.
8. Lee, J. L. (2008). Memory reconsolidation mediates the strengthening of memories. *Nature Neuroscience*, 11(11), 1264-1266.

### Theoretical Frameworks

- McClelland, J. L., et al. (1995). Why there are complementary learning systems. *Psychological Review*, 102(3), 419-457.
- O'Reilly, R. C., & Norman, K. A. (2002). Hippocampal and neocortical contributions to memory. *Trends in Cognitive Sciences*, 6(12), 505-510.

---

## Next Steps

1. **Immediate**: Complete M17 Task 007 (Fan Effect Spreading) to unblock Tasks 017-018
2. **Week 1**: Implement Task 017 (Semantic Priming) as foundational validation
3. **Week 2**: Implement Task 018 (Fan Effect) and validate M17 spreading activation
4. **Week 3**: Begin Task 019 (Consolidation Timeline) for longitudinal validation
5. **Month 2**: Complete remaining tasks and integrate into CI/CD pipeline

**Status**: 4 detailed task specifications complete, 4 more to be created
**Next Task**: Task 017 (Semantic Priming Validation) - awaits M17 Task 007 completion
