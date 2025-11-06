# Task 005: Retroactive Interference and Fan Effect - Validation Notes

**Biological Plausibility:** FAIL - Critical theoretical errors
**Reviewer:** Randy O'Reilly (memory-systems-researcher)

## CRITICAL ERRORS REQUIRING MAJOR REVISION

### ERROR 1: Retroactive Interference Temporal Logic is Backwards

**Current (Line 50):** "1 hour window after original memory"
**Fundamental Problem:** This misunderstands the retroactive interference paradigm.

**Correct Paradigm (McGeoch 1942):**
1. Learn List A at T=0
2. **Immediately** learn List B (interpolated during retention interval)
3. Test List A recall at T=60min

**Critical Point:** List B must be learned **DURING the retention interval between A encoding and A retrieval**, NOT after retrieval.

**Required Complete Rewrite:**
```rust
/// Check if subsequent learning occurred during retention interval
fn is_retroactively_interfering(
    &self,
    target_episode: &Episode,      // List A (learned first)
    subsequent_episode: &Episode,  // List B (learned second)
    retrieval_time: DateTime<Utc>  // When we're testing A recall
) -> bool {
    // List B must be learned AFTER List A (temporal ordering)
    if subsequent_episode.timestamp <= target_episode.timestamp {
        return false;
    }

    // List B must be learned BEFORE current retrieval (interpolated)
    // This is the CRITICAL check missing from current spec
    if subsequent_episode.timestamp >= retrieval_time {
        return false;  // Not interpolated, can't interfere retroactively
    }

    // Standard similarity check
    let similarity = compute_similarity(target_episode, subsequent_episode);
    similarity >= self.similarity_threshold
}
```

### ERROR 2: Quadratic Similarity Weighting Lacks Empirical Support

**Current (Line 52):** `similarity²` weighting
**Problem:** McGeoch (1942) predates modern similarity metrics. Modern research shows **linear or logarithmic** relationships.

**Empirical Reality:**
- Anderson & Neely (1996): Linear relationship between similarity and interference magnitude
- Quadratic creates biologically implausible steep threshold:
  - similarity=0.9 → 81% weight (too high)
  - similarity=0.7 → 49% weight (too low)

**Required Change:**
```rust
pub struct RetroactiveInterferenceDetector {
    base_interference: f32,       // 0.15 (15% base reduction)
    similarity_exponent: f32,     // 1.0 (LINEAR, not 2.0 quadratic)
    max_interference: f32,        // 0.25 (25% max per McGeoch)
}

fn compute_interference_magnitude(
    &self,
    similarity: f32,
    num_interfering: usize
) -> f32 {
    // Linear weighting matches empirical data
    let magnitude = self.base_interference * similarity * (num_interfering as f32);
    magnitude.min(self.max_interference)
}
```

### ERROR 3: Integration Conflates Different Memory Stages

**Current (Lines 176-180):** Treats proactive, retroactive, and fan effect as simultaneously applied during retrieval.

**Cognitive Neuroscience Reality:**
- **Proactive:** Affects **encoding** (old memories compete during new learning)
- **Retroactive:** Affects **consolidation** (new learning disrupts old memory stabilization)
- **Fan Effect:** Affects **retrieval** (high associative density slows recall)

**Required Clarification:**
```rust
// ENCODING STAGE
fn encode_episode(&self, episode: Episode) -> Result<()> {
    // Apply proactive interference from prior similar memories
    let proactive = self.proactive_detector.detect_interference(&episode, ...);
    let adjusted_confidence = episode.confidence * (1.0 - proactive.magnitude);
    self.store_with_confidence(episode, adjusted_confidence)
}

// CONSOLIDATION STAGE (overnight/background)
fn consolidate_episode(&self, episode: Episode) -> Result<()> {
    // Apply retroactive interference from subsequently learned material
    let retroactive = self.retroactive_detector.detect_interference(&episode, ...);
    // Consolidation strength reduced by interpolated learning
}

// RETRIEVAL STAGE
fn retrieve_episode(&self, cue: Cue) -> Result<Vec<(Episode, f32)>> {
    // Apply fan effect to retrieval latency/activation
    for candidate in candidates {
        let fan = self.fan_detector.compute_fan_effect(candidate.id, &graph);
        candidate.activation /= fan.fan_count.max(1) as f32;
    }
}
```

## Fan Effect: MOSTLY CORRECT (Minor Adjustment)

**Current (Lines 79-106):** Linear retrieval time, RT = base + (fan × 50ms)
**Validation:** Anderson (1974) Table 1 shows ~70ms per association (not 50ms).

**Required Adjustment:**
```rust
pub struct FanEffectDetector {
    base_retrieval_time_ms: f32,    // 1150ms (Anderson 1974 fan=1 baseline)
    time_per_association_ms: f32,   // 70ms (CHANGE from 50ms)
}
```

**Empirical Data (Anderson 1974):**
- Fan 1: 1159ms
- Fan 2: 1236ms (+77ms)
- Fan 3: 1305ms (+69ms)
- Average: ~70ms per association

## RECOMMENDED: Split into Separate Tasks

**Task 005a: Retroactive Interference**
- Focus on consolidation-stage phenomenon
- Fix temporal logic (interpolated learning during retention interval)
- Use linear similarity weighting
- Integration: Apply during consolidate() operations

**Task 005b: Fan Effect**
- Keep current implementation (mostly correct)
- Update timing: 50ms → 70ms per association
- Integration: Apply during retrieve() operations

## Required Major Revisions

**Before Implementation:**
1. [ ] Fix retroactive interference temporal logic (retention interval paradigm)
2. [ ] Change similarity weighting: quadratic → linear
3. [ ] Update fan effect parameters: 50ms → 70ms
4. [ ] Clarify integration points: encoding vs consolidation vs retrieval stages
5. [ ] Consider task split (005a + 005b)

**New Validation Tests Required:**
```rust
#[test]
fn test_retroactive_interference_requires_interpolated_learning() {
    // Learn List A at T=0
    // Learn List B at T=30min (DURING retention interval)
    // Test List A at T=60min
    // Expected: List B interferes (learned during retention)

    // Learn List C at T=90min (AFTER retrieval)
    // Expected: List C does NOT interfere (not interpolated)
}

#[test]
fn test_fan_effect_matches_anderson_1974_timing() {
    // Fan=1 → ~1150ms
    // Fan=2 → ~1220ms
    // Fan=3 → ~1290ms
    // Validate slope ~70ms ±15ms per association
}

#[test]
fn test_interference_types_applied_at_correct_stages() {
    // Proactive checked during store()
    // Retroactive checked during consolidate()
    // Fan checked during recall()
    // Verify no double-application or conflicts
}
```

## Biological Plausibility Concerns

**Retroactive Interference Target:**
Should focus on **synaptic consolidation** disruption (first 24 hours, hippocampal-dependent), not systems consolidation. McGeoch (1942) paradigm involves immediate interpolated learning, not long-term (weeks-months) interference.

**Recommended Scope:**
- Primary: Interference during synaptic consolidation window (0-24h post-encoding)
- Secondary: May extend to early systems consolidation (24h-7 days)
- Out of scope: Remote memory interference (>1 month)

## References

- McGeoch, J. A. (1942). The psychology of human learning: An introduction. New York: Longmans, Green.
- Anderson, J. R. (1974). Retrieval of propositional information from long-term memory. *Cognitive Psychology*, 6(4), 451-474.
- Anderson, M. C., & Neely, J. H. (1996). Interference and inhibition in memory retrieval. *Memory*, 125-153.

## Conclusion

**DO NOT IMPLEMENT THIS TASK AS CURRENTLY SPECIFIED.** Requires fundamental redesign of retroactive interference logic and clarification of integration points before proceeding.
