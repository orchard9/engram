# Psychology Validation Implementation Checklist

Quick reference for implementing Tasks 008, 009, 010 using enhanced specifications.

---

## Pre-Implementation (MANDATORY - Do First)

### API Compatibility Check
```bash
# Create and run this test FIRST
touch engram-core/tests/psychology/api_compatibility.rs
cargo test psychology::api_compatibility -- --nocapture
```

**Required APIs:**
- [ ] `MemoryStore::new(capacity)`
- [ ] `EpisodeBuilder` with `.id()`, `.what()`, `.when()`, `.confidence()`, `.embedding()`
- [ ] `store.store(episode)`
- [ ] `store.recall_by_id(id)` or `store.recall_by_content(content)`

**Optional APIs (check availability):**
- [ ] `store.consolidate()` - Task 008 needs this
- [ ] `store.advance_time(duration)` - Tasks 009, 010 need this
- [ ] `store.recall_with_latency(cue)` - Task 010 (fan effect) needs this

**Action if missing:** Update enhanced task specs with actual API before implementation.

### Embedding Generation (Task 008 only)
```bash
# Generate pre-computed embeddings
python scripts/generate_drm_embeddings.py

# Verify output
ls -lh engram-core/tests/psychology/drm_embeddings_precomputed.json
```

**Expected output:** JSON file with 64 word embeddings (4 lists × 16 words)

**Alternative:** Use sentence-transformers if OpenAI unavailable.

---

## Task 008: DRM False Memory Paradigm

### File Creation Checklist
- [ ] `engram-core/tests/psychology/drm_embeddings.rs`
- [ ] `engram-core/tests/psychology/drm_word_lists.json`
- [ ] `engram-core/tests/psychology/drm_embeddings_precomputed.json`
- [ ] `engram-core/tests/psychology/drm_paradigm.rs`
- [ ] `engram-core/tests/psychology/drm_parameter_sweep.rs` (optional - for failures)

### Implementation Steps
1. **Embeddings Module** (`drm_embeddings.rs`)
   - [ ] Implement `get_embedding(word: &str) -> Vec<f32>`
   - [ ] Implement `cosine_similarity(a: &[f32], b: &[f32]) -> f32`
   - [ ] Implement `validate_drm_semantic_structure()`
   - [ ] Test: Embeddings are normalized (norm ≈ 1.0)
   - [ ] Test: Semantic structure validation passes (BAS > 0.35)

2. **Word Lists** (`drm_word_lists.json`)
   - [ ] Create JSON with 4 standard DRM lists
   - [ ] Lists: sleep, chair, doctor, mountain
   - [ ] 15 study items per list
   - [ ] Critical lures never in study items

3. **Main Test** (`drm_paradigm.rs`)
   - [ ] Implement `run_drm_trial(list, seed) -> DrmTrialResult`
   - [ ] Use semantic similarity (NOT string matching) for false recall detection
   - [ ] Implement `DrmAnalysis::from_results()` with:
     - [ ] False recall rate + 95% CI
     - [ ] Cohen's d calculation
     - [ ] Chi-square goodness-of-fit test
   - [ ] Test: n=200 trials (50 per list)
   - [ ] Test: Determinism (same seed → same results)

4. **Validation**
   - [ ] Run: `cargo test psychology::drm_paradigm -- --nocapture`
   - [ ] False recall rate: 55-65% ✓
   - [ ] Chi-square p > 0.01 ✓
   - [ ] All lists show effect (>30%) ✓
   - [ ] Confidence paradox validated ✓

5. **If Test Fails**
   - [ ] Run parameter sweep: `cargo test psychology::drm_parameter_sweep --ignored -- --nocapture`
   - [ ] Identify best configuration
   - [ ] Tune semantic priming / pattern completion
   - [ ] Re-run validation

### Common Issues
| Issue | Cause | Fix |
|-------|-------|-----|
| Embeddings not found | Missing pre-computed file | Run `generate_drm_embeddings.py` |
| Semantic validation fails | BAS too low | Check embedding model, may need better embeddings |
| String matching fails | "chair" matches "chairman" | Use semantic similarity threshold (>0.85) |
| Low power warnings | n < 200 | Increase `TRIALS_PER_LIST` to 50 |

---

## Task 009: Spacing Effect Validation

### File Creation Checklist
- [ ] `engram-core/tests/psychology/spacing_helpers.rs`
- [ ] `engram-core/tests/psychology/spacing_time_simulation.rs`
- [ ] `engram-core/tests/psychology/spacing_effect.rs`
- [ ] `engram-core/tests/psychology/test_materials.json` (optional)

### Implementation Steps
1. **Helper Functions** (`spacing_helpers.rs`)
   - [ ] Implement `generate_random_facts(count, seed) -> Vec<TestFact>`
   - [ ] Implement `test_retention(store, facts) -> f32`
   - [ ] Implement `fact_to_episode(fact, timestamp) -> Episode`
   - [ ] Test: Facts are deterministic (same seed → same facts)
   - [ ] Test: Embeddings are normalized

2. **Time Simulation Validation** (`spacing_time_simulation.rs`)
   - [ ] Test: Linearity (1h + 1h == 2h)
   - [ ] Test: Consistency (multiple engines decay identically)
   - [ ] Test: Monotonicity (no spontaneous increase)
   - [ ] **CRITICAL:** All three tests must pass before main validation

3. **Statistical Functions** (add to `spacing_effect.rs`)
   - [ ] Implement `independent_t_test(group1, group2) -> StatisticalTest`
   - [ ] Implement `normal_cdf(x) -> f64`
   - [ ] Test: t-test returns correct p-values

4. **Main Test** (`spacing_effect.rs`)
   - [ ] Implement `run_single_spacing_trial(seed) -> SpacingTrialResult`
   - [ ] Test: n=200 (100 massed, 100 distributed)
   - [ ] Compute improvement, t-test, Cohen's d
   - [ ] Test: Stability (30 replications, ≥25 pass)
   - [ ] Test: Determinism

5. **Validation**
   - [ ] Run time simulation tests FIRST: `cargo test psychology::spacing_time_simulation -- --nocapture`
   - [ ] Run helpers: `cargo test psychology::spacing_helpers -- --nocapture`
   - [ ] Run main test: `cargo test psychology::spacing_effect::test_spacing_effect_replication -- --nocapture`
   - [ ] Run stability: `cargo test psychology::spacing_effect::test_spacing_effect_stability -- --nocapture`

6. **Success Criteria**
   - [ ] Improvement: 20-40% (±10%) ✓
   - [ ] Statistical significance: p < 0.05 ✓
   - [ ] Cohen's d ≥ 0.3 ✓
   - [ ] Stability: ≥25/30 pass ✓

### Common Issues
| Issue | Cause | Fix |
|-------|-------|-----|
| Time simulation fails | Artifacts in advance_time() | Investigate M4 implementation |
| Insufficient power | n < 200 | Increase `ITEMS_PER_CONDITION` to 100 |
| Unstable results | Random fluctuations | Check determinism test, verify seed control |
| No spacing effect | Decay parameters wrong | Check M4 forgetting curves, run parameter sweep |

---

## Task 010: Interference Validation Suite

### File Creation Checklist
- [ ] `engram-core/tests/psychology/interference_materials.rs`
- [ ] `engram-core/tests/psychology/interference_proactive.rs`
- [ ] `engram-core/tests/psychology/interference_retroactive.rs`
- [ ] `engram-core/tests/psychology/interference_fan_effect.rs`
- [ ] `engram-core/tests/psychology/interference_validation.rs` (main suite)

### Implementation Steps

#### 1. Stimulus Materials (`interference_materials.rs`)
- [ ] Implement `generate_paired_associate_lists(num_lists, pairs_per_list, seed, overlap) -> Vec<Vec<WordPair>>`
- [ ] Implement `generate_fan_facts(count, seed) -> Vec<FanFact>`
- [ ] Test: Deterministic generation
- [ ] Test: Semantic overlap controlled correctly

#### 2. Proactive Interference (`interference_proactive.rs`)
- [ ] Implement `run_proactive_interference_trial(prior_list_count, seed) -> ProactiveResult`
- [ ] Test conditions: 0, 1, 2, 5, 10, 20 prior lists
- [ ] 15 trials per condition (n=90 total)
- [ ] Implement `compute_linear_regression(results) -> RegressionResult`
- [ ] Success criteria:
  - [ ] Negative slope ✓
  - [ ] R² > 0.60 ✓
  - [ ] At 5 lists: 20-30% (±10%) reduction ✓

#### 3. Retroactive Interference (`interference_retroactive.rs`)
- [ ] Implement `run_retroactive_interference_trial(is_experimental, seed) -> RetroactiveResult`
- [ ] Control: Learn A → Rest → Test A
- [ ] Experimental: Learn A → Learn B → Test A
- [ ] 30 trials per condition (n=60 total)
- [ ] Implement t-test
- [ ] Success criteria:
  - [ ] Reduction: 15-25% (±10%) ✓
  - [ ] p < 0.05 ✓
  - [ ] Cohen's d > 0.5 ✓

#### 4. Fan Effect (`interference_fan_effect.rs`)
- [ ] Implement `run_fan_effect_trial(seed) -> Vec<FanEffectResult>`
- [ ] Fan levels: 1, 2, 3, 4 (20 trials per level, n=80 total)
- [ ] Measure RT externally (or add `recall_with_latency()` API)
- [ ] Implement RT regression
- [ ] Success criteria:
  - [ ] Slope: 50-150ms (±25ms) per association ✓
  - [ ] R² > 0.70 ✓

#### 5. Comprehensive Suite (`interference_validation.rs`)
- [ ] Run all three tests sequentially
- [ ] Generate comprehensive validation report
- [ ] Log all statistics

### Validation Commands
```bash
# Test stimulus generation
cargo test psychology::interference_materials -- --nocapture

# Test individual paradigms
cargo test psychology::interference_proactive -- --nocapture
cargo test psychology::interference_retroactive -- --nocapture
cargo test psychology::interference_fan_effect -- --nocapture

# Run comprehensive suite
cargo test psychology::interference_validation -- --nocapture
```

### Common Issues
| Issue | Cause | Fix |
|-------|-------|-----|
| No interference effect | Semantic overlap too low | Increase overlap to 0.5-0.6 |
| RT measurement unavailable | API missing | Use external timing (Instant::now()) |
| Weak PI effect | Prior lists not interfering | Check list similarity, adjust semantic overlap |
| Weak RI effect | List B not interfering with A | Ensure same cues, different targets |

---

## Final Validation Report

After all three tasks complete, create:

**File:** `roadmap/milestone-13/VALIDATION_RESULTS.md`

### Required Content
- [ ] **Executive Summary:** Pass/fail status for each task
- [ ] **DRM Results:**
  - [ ] False recall rate + 95% CI
  - [ ] Chi-square statistic + p-value
  - [ ] Per-list results table
  - [ ] Confidence comparison (false vs true)
- [ ] **Spacing Results:**
  - [ ] Improvement percentage
  - [ ] t-test results (t, df, p, Cohen's d)
  - [ ] Stability analysis (success rate)
- [ ] **Interference Results:**
  - [ ] PI: Regression (slope, R², p), reduction at 5 lists
  - [ ] RI: t-test (t, df, p, d), reduction percentage
  - [ ] Fan: RT regression (slope, R², p)
- [ ] **Comparison Tables:** Engram results vs published research
- [ ] **Parameter Tuning:** Document any parameters adjusted
- [ ] **Follow-up Tasks:** Create tasks for any issues found

### Visualization (Optional but Recommended)
- [ ] DRM: Bar chart (false recall rate by list)
- [ ] Spacing: Line graph (massed vs distributed retention curves)
- [ ] PI: Scatter plot (accuracy vs prior list count) with regression line
- [ ] RI: Bar chart (control vs experimental accuracy)
- [ ] Fan: Scatter plot (RT vs fan level) with regression line

---

## Quick Reference: Success Criteria

| Task | Metric | Target Range | Statistical Test |
|------|--------|--------------|------------------|
| 008 (DRM) | False recall rate | 55-65% (±10%) | χ² p > 0.01 |
| 008 (DRM) | Confidence interval | Overlaps [50%, 70%] | - |
| 008 (DRM) | Per-list effect | >30% all lists | - |
| 009 (Spacing) | Improvement | 20-40% (±10%) | t-test p < 0.05 |
| 009 (Spacing) | Effect size | Cohen's d ≥ 0.3 | - |
| 009 (Spacing) | Stability | ≥25/30 replications | Binomial test |
| 010 (PI) | Reduction at 5 lists | 20-30% (±10%) | - |
| 010 (PI) | Regression fit | R² > 0.60 | F-test |
| 010 (RI) | Reduction | 15-25% (±10%) | t-test p < 0.05 |
| 010 (RI) | Effect size | Cohen's d > 0.5 | - |
| 010 (Fan) | RT slope | 50-150ms (±25ms) | - |
| 010 (Fan) | Regression fit | R² > 0.70 | F-test |

---

## Emergency Contacts

**If validation fails and parameter sweep doesn't help:**

1. **Semantic/cognitive issues:** Consult `memory-systems-researcher` agent
2. **Statistical issues:** Re-review `PSYCHOLOGY_VALIDATION_REVIEW.md`
3. **Implementation bugs:** Check API compatibility, embedding generation
4. **Time simulation artifacts:** Investigate M4 temporal dynamics

**Remember:** These are empirical validations, not unit tests. Treat failures as research findings, not bugs. Document unexpected results thoroughly.

---

**Good luck! These validations will prove Engram has real cognitive memory.**
