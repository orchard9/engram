# Milestone-13 Cognitive Neuroscience Validation Report

**Reviewer:** Randy O'Reilly (memory-systems-researcher agent)
**Date:** 2025-10-26
**Scope:** Tasks 003-007 biological plausibility and empirical grounding

---

## Executive Summary

**Overall Assessment:** 4/5 tasks PASS with minor corrections, 1/5 task FAIL requires major revision

- **Task 003 (Associative/Repetition Priming):** PASS with parameter corrections
- **Task 004 (Proactive Interference):** PASS with temporal window adjustment
- **Task 005 (Retroactive/Fan Effect):** FAIL - critical theoretical errors
- **Task 006 (Reconsolidation Core):** PASS with boundary condition refinements
- **Task 007 (Reconsolidation Integration):** PASS - biologically sound

---

## Task 003: Associative and Repetition Priming

### Biological Plausibility: PASS (with corrections)

### Critical Issues

**Issue 1: Co-occurrence temporal window too restrictive**

The spec states 10-second window for co-occurrence counting (line 20), but this conflicts with empirical evidence on associative learning timescales.

**Empirical Basis:**
- McKoon & Ratcliff (1992) used trial-based paradigms with inter-trial intervals of 2-30 seconds
- Associative learning in hippocampus operates on timescales from 100ms (spike-timing dependent plasticity) to minutes (working memory span)
- Hebb's rule for synaptic strengthening: "Neurons that fire together, wire together" - effective window is ~100ms for STDP, but functional association extends to seconds for semantic co-occurrence

**Recommended Correction:**
Change co-occurrence window to **30 seconds** as default, make configurable:

```rust
/// Temporal window for co-occurrence (default: 30 seconds)
/// Empirical basis: Working memory span (~7±2 items × 3-4s per item)
cooccurrence_window: Duration,  // default: Duration::seconds(30)
```

**Justification:** Working memory span governs functional co-occurrence - items within the same working memory episode should count as co-activated. This aligns with:
- Baddeley's phonological loop duration (~2s without rehearsal)
- Visual-spatial sketchpad retention (~3-4s)
- Central executive integration window (~10-30s for task episodes)

**Issue 2: Minimum co-occurrence threshold lacks empirical grounding**

Line 21 specifies minimum 3 co-occurrences before association forms, but provides no citation.

**Empirical Evidence:**
- Rescorla-Wagner model: Association strength is continuous from first pairing
- Single-trial learning exists in hippocampus (e.g., fear conditioning can occur in 1 trial)
- Statistical learning literature (Saffran et al. 1996): Transitional probabilities detected after ~2 minutes exposure (~6-12 co-occurrences for common pairs)

**Recommended Correction:**
Lower minimum to **2 co-occurrences** OR make this an adaptive threshold based on baseline co-occurrence rate:

```rust
/// Minimum co-occurrence for reliable association (default: 2)
/// Empirical basis: Saffran et al. (1996) statistical learning
/// Prevents spurious associations from single random co-activation
min_cooccurrence: u64,  // default: 2
```

**Issue 3: Repetition priming parameters are accurate**

Lines 28-29 specify 5% boost per exposure with 30% ceiling - this aligns well with empirical data.

**Empirical Validation:**
- Tulving & Schacter (1990): Perceptual priming shows 20-50% RT reduction over 3-6 exposures
- 5% × 6 exposures = 30% total matches upper bound of perceptual fluency effects
- Ceiling prevents unrealistic over-priming from excessive repetitions
- **No changes needed** - parameters are biologically plausible

**Issue 4: Additive combination of priming types needs justification**

Line 156 specifies additive combination: `semantic + associative + repetition`

**Cognitive Neuroscience Evidence:**
Priming types operate through different neural mechanisms:
- **Semantic priming:** Spreading activation in neocortical association areas (temporal/parietal)
- **Associative priming:** Hippocampal binding and co-activation patterns
- **Repetition priming:** Perceptual processing fluency in sensory cortices

**Problem:** Different mechanisms suggest **independent contributions**, supporting additive model. However, empirical studies show **diminishing returns** when multiple priming types combine (e.g., Neely & Keefe 1989).

**Recommended Correction:**
Add **saturation function** to prevent over-priming:

```rust
pub fn compute_total_boost(&self, node_id: NodeId) -> f32 {
    let semantic = self.semantic.compute_priming_boost(node_id);
    let associative = self.associative.compute_association_strength(/* ... */);
    let repetition = self.repetition.compute_repetition_boost(node_id);

    // Additive combination with saturation (prevents >80% total boost)
    let linear_sum = semantic + associative + repetition;

    // Diminishing returns: boost_effective = 1 - exp(-linear_sum)
    // At linear_sum=0.3 → 26%, at 0.6 → 45%, at 1.0 → 63%
    1.0 - (-linear_sum).exp()
}
```

**Justification:** Log-linear saturation matches:
- Neural firing rate saturation (can't exceed max firing rate)
- Behavioral ceiling effects in RT facilitation studies
- Prevents unrealistic 100%+ boosts from perfect storm of all priming types

### Recommended Parameter Adjustments

```rust
// /engram-core/src/cognitive/priming/associative.rs
pub struct AssociativePrimingEngine {
    /// Temporal window for co-occurrence (default: 30 seconds, not 10)
    /// Empirical: Working memory span + central executive integration
    cooccurrence_window: Duration,  // Duration::seconds(30)

    /// Minimum co-occurrence (default: 2, not 3)
    /// Empirical: Saffran et al. (1996) statistical learning threshold
    min_cooccurrence: u64,  // 2
}

// /engram-core/src/cognitive/priming/mod.rs
impl PrimingCoordinator {
    /// Compute total boost with saturation (not linear additive)
    pub fn compute_total_boost(&self, node_id: NodeId) -> f32 {
        let linear_sum = semantic + associative + repetition;
        1.0 - (-linear_sum).exp()  // Diminishing returns
    }
}
```

### Additional Validation Tests Needed

```rust
#[test]
fn test_cooccurrence_window_matches_working_memory_span() {
    // Items within 30s window should co-activate
    // Items beyond 30s should not (working memory cleared)
}

#[test]
fn test_priming_saturation_prevents_overshoot() {
    // semantic=0.3, associative=0.3, repetition=0.3
    // linear_sum = 0.9 → should saturate to ~0.60, not 0.90
}
```

---

## Task 004: Proactive Interference Detection

### Biological Plausibility: PASS (with temporal window adjustment)

### Critical Issues

**Issue 1: 24-hour temporal window too long**

Line 41 specifies "Only memories within 24 hours prior count as interfering," but Underwood (1957) used **inter-list intervals of minutes to hours**, not days.

**Empirical Basis:**
- Underwood (1957): Lists learned in sequence during single experimental session (typically 30min - 2hr total)
- Proactive interference peaks when interfering material is learned **minutes to hours** before target list
- After 24 hours, consolidation reduces interference (complementary learning systems theory)

**Recommended Correction:**
Change temporal window to **6 hours** (not 24 hours):

```rust
/// Temporal window for "prior" memories (default: 6 hours before)
/// Empirical: Underwood (1957) session-based interference
prior_memory_window: Duration,  // Duration::hours(6)
```

**Justification:**
- 6 hours captures within-session interference without crossing consolidation boundary
- Aligns with synaptic consolidation timescale (~6 hours for protein synthesis)
- After 6 hours, memories transition from hippocampal-dependent to neocortical representations (reduced interference)

**Issue 2: Linear accumulation model is correct**

Lines 42-43 specify **5% per similar item, capped at 30%** - this matches Underwood's (1957) findings well.

**Empirical Validation:**
- Underwood (1957): Interference increases ~linearly with number of prior lists (1-5 lists)
- With 6 prior lists, recall dropped from ~70% to ~45% (25% reduction)
- 5% × 5 items = 25% matches empirical data
- **No changes needed** - linear model is appropriate

**Issue 3: Similarity threshold 0.7 is reasonable but should be validated**

Line 42 specifies embedding similarity ≥ 0.7 for interference.

**Cognitive Neuroscience Considerations:**
- Similarity threshold determines what counts as "confusable" items
- Too low (e.g., 0.5): Unrelated items interfere (biologically implausible)
- Too high (e.g., 0.9): Only near-duplicates interfere (misses category-level interference)

**Recommended Validation:**
The 0.7 threshold is reasonable but needs **empirical calibration**:

```rust
#[test]
fn test_similarity_threshold_calibration_with_category_structure() {
    // Within-category pairs (dog/cat): similarity should be ≥ 0.7
    // Across-category pairs (dog/car): similarity should be < 0.7
    // Validates threshold captures semantic category interference
}
```

**Issue 4: Temporal direction is correctly enforced**

Line 44 correctly states "Only forward-in-time interference (old → new)" - this is crucial for distinguishing proactive from retroactive interference.

**Biological Plausibility:** CORRECT
The implementation correctly checks `prior_episode.timestamp >= new_episode.timestamp` (line 150), ensuring only temporally-prior memories interfere.

### Recommended Parameter Adjustments

```rust
pub struct ProactiveInterferenceDetector {
    similarity_threshold: f32,        // 0.7 (KEEP)
    prior_memory_window: Duration,    // Duration::hours(6) - CHANGE from 24h
    interference_per_item: f32,       // 0.05 (KEEP)
    max_interference: f32,            // 0.30 (KEEP)
}
```

### Additional Validation Tests Needed

```rust
#[test]
fn test_consolidation_boundary_reduces_interference() {
    // Prior memory learned >6 hours ago: reduced/no interference
    // Prior memory learned <6 hours ago: full interference
    // Validates synaptic consolidation effect
}

#[test]
fn test_interference_magnitude_matches_underwood_1957() {
    // 5 similar prior lists within 6h window
    // Expected: 20-30% accuracy reduction (matches Underwood 1957 Fig 2)
}
```

---

## Task 005: Retroactive Interference and Fan Effect

### Biological Plausibility: FAIL - Critical theoretical errors

This task contains **fundamental misunderstandings** of cognitive neuroscience literature and conflates distinct phenomena.

### Critical Issues

**Issue 1: Retroactive interference temporal window is backwards**

Line 50 specifies 1-hour window **after** original memory, but this misunderstands the retroactive interference paradigm.

**Empirical Paradigm (McGeoch 1942):**
1. Learn List A
2. **Immediately** learn List B (interpolated list)
3. Test recall of List A

**Critical Point:** The interfering material (List B) is learned **during the retention interval**, not after retrieval. The spec confuses:
- **Retention interval:** Time between learning A and testing A
- **Interference window:** When interpolated material (B) is learned

**Recommended Correction:**
Retroactive interference should check if **subsequent material was learned during the retention interval before retrieval**, not after:

```rust
/// Check if subsequent learning occurred during retention interval
/// (between original encoding and current retrieval attempt)
fn is_retroactively_interfering(
    &self,
    target_episode: &Episode,      // List A (learned first)
    subsequent_episode: &Episode,  // List B (learned second)
    retrieval_time: DateTime<Utc>  // When we're testing recall
) -> bool {
    // List B must be learned AFTER List A (temporal ordering)
    if subsequent_episode.timestamp <= target_episode.timestamp {
        return false;
    }

    // List B must be learned BEFORE current retrieval (during retention interval)
    if subsequent_episode.timestamp >= retrieval_time {
        return false;
    }

    // Similarity check (same as proactive)
    let similarity = compute_similarity(target_episode, subsequent_episode);
    similarity >= self.similarity_threshold
}
```

**Issue 2: Quadratic similarity weighting lacks empirical support**

Line 52 specifies `similarity²` weighting for retroactive interference, claiming this comes from McGeoch (1942).

**Empirical Reality:**
- McGeoch (1942) predates modern similarity metrics (used categorical similarity: synonyms, opposites, unrelated)
- Modern research (e.g., Anderson & Neely 1996) shows **linear or logarithmic** relationship between similarity and interference magnitude
- Quadratic weighting would predict:
  - similarity=0.9 → interference = 0.81 (81% of maximum)
  - similarity=0.7 → interference = 0.49 (49% of maximum)
  - This creates **extremely steep threshold** - biologically implausible

**Recommended Correction:**
Use **linear similarity weighting** matching empirical data:

```rust
pub struct RetroactiveInterferenceDetector {
    base_interference: f32,       // 0.15 (15% base reduction)
    similarity_exponent: f32,     // 1.0 (LINEAR, not quadratic)
    max_interference: f32,        // 0.25 (25% max, matching McGeoch)
}

fn compute_interference_magnitude(
    &self,
    similarity: f32,
    num_interfering: usize
) -> f32 {
    // Linear weighting: magnitude = base × similarity × count
    let magnitude = self.base_interference * similarity * (num_interfering as f32);
    magnitude.min(self.max_interference)
}
```

**Issue 3: Fan effect implementation is fundamentally correct**

Lines 79-106 correctly implement Anderson (1974) fan effect:
- Linear retrieval time increase: RT = base + (fan × 50ms)
- Fan count = number of associations to a concept
- **No changes needed** - this aligns with ACT-R architecture and empirical data

**Validation:** Anderson (1974) Table 1 shows:
- Fan 1: 1159ms
- Fan 2: 1236ms (+77ms)
- Fan 3: 1305ms (+69ms from fan 2)
- Average: ~70ms per association

**Recommended Adjustment:**
Update default from 50ms to **70ms** to better match Anderson (1974):

```rust
pub struct FanEffectDetector {
    base_retrieval_time_ms: f32,    // 1150ms (matches Anderson fan=1 baseline)
    time_per_association_ms: f32,   // 70ms (not 50ms - better fit to data)
}
```

**Issue 4: Integration of three interference types is conceptually confused**

The spec treats proactive, retroactive, and fan effect as **independent** phenomena that should be tracked separately (line 176-180). This is correct in principle, but the implementation details are problematic.

**Cognitive Neuroscience Reality:**
- **Proactive interference:** Old memories compete during **encoding** of new information
- **Retroactive interference:** New learning disrupts **consolidation** of old memories
- **Fan effect:** High associative density slows **retrieval** due to competition

**Critical Problem:** These operate at **different memory stages** (encoding, consolidation, retrieval), so they shouldn't all apply simultaneously to the same retrieval operation.

**Recommended Correction:**
Apply interference types at appropriate processing stages:

```rust
// ENCODING STAGE (storing new episode)
fn encode_episode(&self, episode: Episode) -> Result<()> {
    // Check proactive interference from prior similar memories
    let proactive = self.proactive_detector.detect_interference(&episode, ...);
    let adjusted_confidence = episode.confidence * (1.0 - proactive.magnitude);

    self.store_with_confidence(episode, adjusted_confidence)
}

// CONSOLIDATION STAGE (overnight processing)
fn consolidate_episode(&self, episode: Episode) -> Result<()> {
    // Check retroactive interference from subsequently learned material
    let retroactive = self.retroactive_detector.detect_interference(&episode, ...);
    // Consolidation strength reduced by retroactive interference
}

// RETRIEVAL STAGE (recall operation)
fn retrieve_episode(&self, cue: Cue) -> Result<Vec<(Episode, f32)>> {
    // Apply fan effect to retrieval latency/confidence
    for candidate in candidates {
        let fan = self.fan_detector.compute_fan_effect(candidate.id, &graph);
        // Higher fan → lower activation per edge → slower retrieval
        candidate.activation /= fan.fan_count.max(1) as f32;
    }
}
```

### Recommended Major Revisions

**This task requires significant restructuring:**

1. **Split into separate tasks:**
   - Task 005a: Retroactive Interference (consolidation-stage phenomenon)
   - Task 005b: Fan Effect (retrieval-stage phenomenon)

2. **Fix retroactive interference temporal logic:**
   - Check for interpolated learning during retention interval
   - Use linear (not quadratic) similarity weighting

3. **Update fan effect parameters:**
   - Change 50ms → 70ms per association (better empirical fit)
   - Keep linear model (correct)

4. **Clarify integration points:**
   - Proactive → affects encoding confidence
   - Retroactive → affects consolidation strength
   - Fan → affects retrieval activation

### Additional Validation Tests Needed

```rust
#[test]
fn test_retroactive_interference_requires_interpolated_learning() {
    // Learn List A at T=0
    // Learn List B at T=30min (DURING retention interval)
    // Test List A recall at T=60min
    // List B should interfere (learned during retention)

    // Learn List C at T=90min (AFTER retrieval)
    // List C should NOT interfere (not interpolated)
}

#[test]
fn test_fan_effect_linear_scaling_matches_anderson_1974() {
    // Fan=1 → ~1150ms
    // Fan=2 → ~1220ms
    // Fan=3 → ~1290ms
    // Validate slope ~70ms per association ±15ms
}

#[test]
fn test_interference_types_applied_at_different_stages() {
    // Proactive checked during store()
    // Retroactive checked during consolidate()
    // Fan checked during recall()
    // Verify stages don't conflict or double-apply
}
```

---

## Task 006: Reconsolidation Engine Core

### Biological Plausibility: PASS (with boundary condition refinements)

### Critical Issues

**Issue 1: Reconsolidation window boundaries are well-specified**

Lines 56-68 specify:
- Window start: 1 hour post-recall
- Window end: 6 hours post-recall
- Minimum memory age: 24 hours
- Maximum memory age: 365 days

**Empirical Validation:**
These align well with Nader et al. (2000) and subsequent literature:
- **1-hour start:** Matches protein synthesis initiation window (Lee 2009)
- **6-hour end:** Matches protein synthesis completion (Nader & Einarsson 2010)
- **24-hour minimum:** Ensures memory is consolidated before reconsolidation (systems consolidation)
- **365-day maximum:** Remote memories show reduced plasticity (Frankland & Bontempi 2005)

**Minor Issue:** The 365-day boundary is less precise than claimed (line 75: "boundary less precise").

**Recommended Clarification:**
Make maximum age **configurable** with clear documentation:

```rust
/// Maximum memory age for reconsolidation (default: 365 days)
/// Empirical note: This boundary is less precise than others
/// Remote memories (>1 year) show reduced but not absent plasticity
/// Recommend tuning based on domain (semantic memories may remain plastic longer)
max_memory_age: Duration,  // Duration::days(365)
```

**Issue 2: Active recall requirement is correct**

Lines 99-100 correctly require `is_active_recall: bool` to distinguish active retrieval from passive re-exposure.

**Empirical Basis:**
- Nader et al. (2000): Reconsolidation triggered by memory **reactivation**, not mere re-exposure
- Passive exposure (e.g., seeing related cues without full recall) does NOT trigger reconsolidation
- Active recall requires **hippocampal replay** and **retrieval-induced reactivation**

**Validation:** The boundary check (lines 220-225) correctly enforces this. **No changes needed.**

**Issue 3: Linear plasticity decrease across window is oversimplified**

Lines 296-300 implement **linear** plasticity decrease: `plasticity = max × (1.0 - window_position)`

**Empirical Reality:**
Protein synthesis-dependent reconsolidation shows **non-linear kinetics**:
- Rapid rise to peak plasticity (0-2 hours)
- Plateau at peak (2-4 hours)
- Gradual decline (4-6 hours)

**Recommended Correction:**
Use **inverted-U function** matching protein synthesis dynamics:

```rust
/// Compute plasticity factor based on window position
///
/// Uses inverted-U function matching protein synthesis dynamics:
/// - Early phase (0-0.4): Rising plasticity
/// - Peak phase (0.4-0.6): Maximum plasticity
/// - Late phase (0.6-1.0): Declining plasticity
fn compute_plasticity(&self, window_position: f32) -> f32 {
    // Inverted-U: peaks at window_position = 0.5
    // f(x) = 4x(1-x) gives parabola with peak at 0.5
    let u_curve = 4.0 * window_position * (1.0 - window_position);

    self.reconsolidation_plasticity * u_curve
}
```

**Justification:**
- Matches protein synthesis time course (Nader & Einarsson 2010 Fig 3)
- Peak plasticity at 2-3 hours post-recall (middle of window)
- Degrading early or late attempts reflects biological reality

**Issue 4: Modification confidence calculation needs refinement**

Lines 342-356 compute modified confidence as:
```rust
reduction = modification_extent × plasticity × 0.2
new_confidence = original × (1.0 - reduction)
```

**Problem:** This assumes all modifications **reduce** confidence, but reconsolidation can also **strengthen** memories (Dudai 2006).

**Recommended Correction:**
Distinguish **updating** (neutral/positive) from **corrupting** (negative) modifications:

```rust
pub struct EpisodeModifications {
    pub field_changes: HashMap<String, String>,
    pub modification_extent: f32,
    pub modification_type: ModificationType,  // NEW
}

pub enum ModificationType {
    /// Updating with new accurate information (maintains/increases confidence)
    Update,
    /// Partial corruption or uncertainty (decreases confidence)
    Corruption,
    /// Complete replacement (resets confidence to modification quality)
    Replacement,
}

fn compute_modified_confidence(
    &self,
    original_confidence: Confidence,
    modifications: &EpisodeModifications,
    plasticity: f32
) -> Confidence {
    match modifications.modification_type {
        ModificationType::Update => {
            // Updating strengthens memory (like rehearsal)
            let boost = plasticity * 0.1;
            Confidence::new((original_confidence.value() * (1.0 + boost)).min(1.0))
        }
        ModificationType::Corruption => {
            // Current implementation (reduction)
            let reduction = modifications.modification_extent * plasticity * 0.2;
            Confidence::new((original_confidence.value() * (1.0 - reduction)).max(0.1))
        }
        ModificationType::Replacement => {
            // Confidence reset to quality of new information
            Confidence::new(modifications.replacement_confidence.unwrap_or(0.5))
        }
    }
}
```

### Recommended Parameter Adjustments

```rust
pub struct ReconsolidationEngine {
    window_start: Duration,    // Duration::hours(1) - KEEP
    window_end: Duration,      // Duration::hours(6) - KEEP
    min_memory_age: Duration,  // Duration::hours(24) - KEEP
    max_memory_age: Duration,  // Duration::days(365) - KEEP, but document uncertainty

    // CHANGE: Use inverted-U function, not linear
    reconsolidation_plasticity: f32,  // 0.5 max (applied via inverted-U)
}
```

### Additional Validation Tests Needed

```rust
#[test]
fn test_plasticity_peaks_mid_window() {
    // Plasticity at 1h < plasticity at 3h
    // Plasticity at 3h > plasticity at 5.5h
    // Validates inverted-U dynamics
}

#[test]
fn test_update_modifications_strengthen_memory() {
    // Modification type: Update (not corruption)
    // Original confidence: 0.7
    // After reconsolidation: confidence should be ≥0.7 (strengthened)
}

#[test]
fn test_remote_memory_boundary_gradual_not_sharp() {
    // Memory at 364 days: should be eligible
    // Memory at 366 days: currently rejected
    // Consider gradual plasticity reduction instead of hard cutoff
}
```

---

## Task 007: Reconsolidation Integration with Consolidation Pipeline

### Biological Plausibility: PASS - Excellent biological grounding

### Critical Issues

**No major issues.** This task correctly implements the **critical biological insight** that reconsolidation re-uses consolidation machinery.

**Empirical Validation:**
- Nader et al. (2000): Reconsolidation requires same protein synthesis as initial consolidation
- Lee (2009): "Reconsolidation is a re-engagement of consolidation processes"
- Task correctly tags reconsolidated memories and routes them through M6 consolidation pipeline

**Minor Enhancement:** Add metric to track **reconsolidation cycles**

```rust
pub struct ReconsolidationConsolidationBridge {
    /// Track how many times each episode has been reconsolidated
    reconsolidation_cycles: DashMap<String, u32>,
}

impl ReconsolidationConsolidationBridge {
    pub fn process_reconsolidated_memory(
        &self,
        modified_episode: Episode,
        reconsolidation_result: ReconsolidationResult
    ) -> Result<ConsolidationHandle> {
        // Increment reconsolidation cycle counter
        let cycles = self.reconsolidation_cycles
            .entry(modified_episode.id.clone())
            .or_insert(0);
        *cycles += 1;

        // Tag with cycle count (prevents infinite reconsolidation loops)
        let tagged = modified_episode
            .with_metadata("reconsolidation_event", reconsolidation_result.plasticity_factor)
            .with_metadata("reconsolidation_cycles", cycles.to_string());

        // Prevent excessive reconsolidation (>10 cycles may indicate pathology)
        if *cycles > 10 {
            warn!("Episode {} has been reconsolidated {} times - possible memory instability",
                  modified_episode.id, *cycles);
        }

        self.consolidation.consolidate(tagged)
    }
}
```

**Justification:**
- Excessive reconsolidation can create **memory instability** (Dudai & Eisenberg 2004)
- Tracking cycles enables detection of pathological patterns
- Aligns with boundary hypothesis (Finnie & Nader 2012): repeated reconsolidation may eventually render memory non-reconsolidable

### Recommended Enhancements

```rust
#[test]
fn test_repeated_reconsolidation_tracked_and_bounded() {
    // Reconsolidate same memory 3 times
    // Verify cycle counter increments correctly
    // Verify warning issued if >10 cycles
}

#[test]
fn test_reconsolidation_metrics_distinguish_cycles() {
    #[cfg(feature = "monitoring")]
    {
        // First reconsolidation: cycle=1
        // Second reconsolidation: cycle=2
        // Metrics should track distribution of cycle counts
    }
}
```

---

## Summary Recommendations

### Tasks Requiring Immediate Revision

**CRITICAL - Task 005 (Retroactive/Fan Effect):**
- [ ] Fix retroactive interference temporal logic (retention interval, not post-recall)
- [ ] Change quadratic → linear similarity weighting
- [ ] Update fan effect timing: 50ms → 70ms per association
- [ ] Clarify integration: different interference types at different memory stages
- [ ] Consider splitting into Task 005a (Retroactive) and 005b (Fan Effect)

### Tasks Requiring Minor Corrections

**Task 003 (Associative/Repetition Priming):**
- [ ] Change co-occurrence window: 10s → 30s
- [ ] Lower minimum co-occurrence: 3 → 2
- [ ] Add saturation to combined priming (diminishing returns)

**Task 004 (Proactive Interference):**
- [ ] Reduce temporal window: 24 hours → 6 hours
- [ ] Add empirical validation test matching Underwood (1957) Fig 2

**Task 006 (Reconsolidation Core):**
- [ ] Change plasticity function: linear → inverted-U
- [ ] Add modification type (Update vs Corruption vs Replacement)
- [ ] Document maximum age boundary uncertainty

**Task 007 (Reconsolidation Integration):**
- [ ] Add reconsolidation cycle tracking
- [ ] Add warning for excessive reconsolidation (>10 cycles)

---

## Validation Against Complementary Learning Systems Theory

All tasks align with **CLS theory** (McClelland et al. 1995) framework:

**Hippocampal System (Fast Learning):**
- Associative priming via rapid co-occurrence binding ✓
- Proactive interference from overlapping patterns ✓
- Reconsolidation as hippocampal replay and updating ✓

**Neocortical System (Slow Learning):**
- Semantic priming via distributed representations ✓
- Gradual pattern extraction during consolidation (M6) ✓
- Fan effect from distributed associative networks ✓

**Integration:**
- Task 007 correctly bridges reconsolidation → consolidation ✓
- M6 provides hippocampal-to-neocortical transfer ✓

**One Concern:** Retroactive interference (Task 005) needs clarification on whether it disrupts:
- Hippocampal consolidation (first 24 hours) - PRIMARY TARGET
- Neocortical consolidation (weeks-months) - SECONDARY TARGET

Recommend Task 005 focus on **synaptic consolidation** (first 24 hours, hippocampal-dependent) per McGeoch (1942) paradigm.

---

## References Cited

1. Anderson, J. R. (1974). Retrieval of propositional information from long-term memory. *Cognitive Psychology*, 6(4), 451-474.
2. Anderson, M. C., & Neely, J. H. (1996). Interference and inhibition in memory retrieval. *Memory*, 125-153.
3. Dudai, Y. (2006). Reconsolidation: the advantage of being refocused. *Current Opinion in Neurobiology*, 16(2), 174-178.
4. Dudai, Y., & Eisenberg, M. (2004). Rites of passage of the engram. *Neuroscience*, 7, 584-590.
5. Finnie, P. S., & Nader, K. (2012). The role of metaplasticity mechanisms in regulating memory destabilization and reconsolidation. *Neuroscience & Biobehavioral Reviews*, 36(7), 1667-1707.
6. Frankland, P. W., & Bontempi, B. (2005). The organization of recent and remote memories. *Nature Reviews Neuroscience*, 6(2), 119-130.
7. Lee, J. L. (2009). Reconsolidation: maintaining memory relevance. *Trends in Neurosciences*, 32(8), 413-420.
8. McClelland, J. L., McNaughton, B. L., & O'Reilly, R. C. (1995). Why there are complementary learning systems in the hippocampus and neocortex. *Psychological Review*, 102(3), 419.
9. McGeoch, J. A. (1942). The psychology of human learning: An introduction. New York: Longmans, Green.
10. McKoon, G., & Ratcliff, R. (1992). Spreading activation versus compound cue accounts of priming. *Psychological Review*, 99(1), 177.
11. Nader, K., Schafe, G. E., & Le Doux, J. E. (2000). Fear memories require protein synthesis in the amygdala for reconsolidation after retrieval. *Nature*, 406(6797), 722-726.
12. Nader, K., & Einarsson, E. Ö. (2010). Memory reconsolidation: an update. *Annals of the New York Academy of Sciences*, 1191(1), 27-41.
13. Neely, J. H., & Keefe, D. E. (1989). Semantic context effects on visual word processing: A hybrid prospective-retrospective processing theory. *Psychology of Learning and Motivation*, 24, 207-248.
14. Saffran, J. R., Aslin, R. N., & Newport, E. L. (1996). Statistical learning by 8-month-old infants. *Science*, 274(5294), 1926-1928.
15. Tulving, E., & Schacter, D. L. (1990). Priming and human memory systems. *Science*, 247(4940), 301-306.
16. Underwood, B. J. (1957). Interference and forgetting. *Psychological Review*, 64(1), 49.

---

## Conclusion

**4 of 5 tasks** demonstrate strong biological plausibility with minor parameter adjustments needed. **Task 005 requires major revision** due to fundamental misunderstanding of retroactive interference temporal dynamics and inappropriate quadratic similarity weighting.

**Overall Milestone-13 biological realism: 7.5/10**
- Strong grounding in empirical literature (Tasks 003, 004, 006, 007)
- Exact boundary conditions well-specified (Task 006)
- Critical failure in Task 005 temporal logic
- Minor parameter tuning needed across all tasks

**Recommended action:** Fix Task 005 before proceeding with implementation. All other tasks can proceed with minor corrections applied during development.
