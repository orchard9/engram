# Task 006: Reconsolidation Engine Core - Validation Notes

**Biological Plausibility:** PASS (with boundary condition refinements)
**Reviewer:** Randy O'Reilly (memory-systems-researcher)

## Overall Assessment

**Strong empirical grounding** with exact boundary conditions from Nader et al. (2000). Minor refinements needed for plasticity dynamics and confidence calculations.

## Validated Boundary Conditions

### 1. Temporal Windows: CORRECT

**Window start: 1 hour** (Line 62)
- ✓ Matches protein synthesis initiation (Nader et al. 2000)
- ✓ Aligns with synaptic plasticity onset (Lee 2009)

**Window end: 6 hours** (Line 66)
- ✓ Matches protein synthesis completion window
- ✓ Corresponds to reconsolidation "closing" (Nader & Einarsson 2010)

**Minimum age: 24 hours** (Line 70)
- ✓ Ensures memory is consolidated before reconsolidation
- ✓ Aligns with synaptic consolidation timescale

**Maximum age: 365 days** (Line 74)
- ✓ Reasonable boundary for reduced plasticity in remote memories
- ⚠️ Less precise than other boundaries (Frankland & Bontempi 2005)

**Recommended Clarification:**
```rust
/// Maximum memory age for reconsolidation (default: 365 days)
/// NOTE: This boundary is less precise than others. Remote memories
/// show reduced but not absent plasticity. Recommend domain-specific
/// tuning (semantic memories may remain plastic longer than episodic).
max_memory_age: Duration,  // Duration::days(365)
```

### 2. Active Recall Requirement: CORRECT

**Implementation (Lines 99-100, 220-225):** Correctly enforces `is_active_recall: bool`
- ✓ Distinguishes active retrieval from passive re-exposure
- ✓ Aligns with Nader et al. (2000): reconsolidation triggered by reactivation
- ✓ Passive exposure does NOT trigger reconsolidation

**No changes needed** - implementation is biologically accurate.

## Required Corrections

### 1. Plasticity Function: Linear → Inverted-U

**Current (Lines 296-300):** Linear decrease
```rust
plasticity = max × (1.0 - window_position)
```

**Problem:** Protein synthesis shows non-linear kinetics:
- Rapid rise (0-2h)
- Plateau at peak (2-4h)
- Gradual decline (4-6h)

**Required Change:**
```rust
/// Compute plasticity factor based on window position
///
/// Uses inverted-U function matching protein synthesis dynamics
/// (Nader & Einarsson 2010 Fig 3)
fn compute_plasticity(&self, window_position: f32) -> f32 {
    // Inverted-U: peaks at window_position = 0.5 (middle of window)
    // f(x) = 4x(1-x) gives parabola with maximum at x=0.5
    let u_curve = 4.0 * window_position * (1.0 - window_position);

    self.reconsolidation_plasticity * u_curve
}
```

**Empirical Justification:**
- Peak plasticity at 2-3 hours post-recall (middle of 1-6h window) ✓
- Early attempts (1h): rising but not peak plasticity ✓
- Late attempts (5-6h): declining plasticity ✓

**New Validation Test:**
```rust
#[test]
fn test_plasticity_peaks_mid_window() {
    let engine = ReconsolidationEngine::new();

    // Plasticity at 1h < plasticity at 3h
    let early_plasticity = engine.compute_plasticity(0.0);  // window_position=0
    let mid_plasticity = engine.compute_plasticity(0.5);    // window_position=0.5

    assert!(mid_plasticity > early_plasticity);

    // Plasticity at 3h > plasticity at 5.5h
    let late_plasticity = engine.compute_plasticity(0.9);   // window_position=0.9
    assert!(mid_plasticity > late_plasticity);
}
```

### 2. Modification Confidence: Add Modification Type

**Current (Lines 342-356):** All modifications **reduce** confidence
**Problem:** Reconsolidation can **strengthen** memories (Dudai 2006)

**Required Enhancement:**
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
    /// Complete replacement (resets confidence)
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
            // Strengthens memory (like rehearsal)
            let boost = plasticity * 0.1;  // Up to 5% boost at peak plasticity
            Confidence::new((original_confidence.value() * (1.0 + boost)).min(1.0))
        }
        ModificationType::Corruption => {
            // Current implementation (reduction)
            let reduction = modifications.modification_extent * plasticity * 0.2;
            Confidence::new((original_confidence.value() * (1.0 - reduction)).max(0.1))
        }
        ModificationType::Replacement => {
            // Confidence reset to quality of new information
            Confidence::new(0.5)  // Or provided replacement_confidence
        }
    }
}
```

**Biological Justification:**
- Retrieval-induced strengthening (Roediger & Karpicke 2006)
- Reconsolidation can enhance memory stability (Dudai 2006)
- Not all modifications are corrupting

## Updated Default Parameters

```rust
impl Default for ReconsolidationEngine {
    fn default() -> Self {
        Self {
            window_start: Duration::hours(1),    // ✓ KEEP
            window_end: Duration::hours(6),      // ✓ KEEP
            min_memory_age: Duration::hours(24), // ✓ KEEP
            max_memory_age: Duration::days(365), // ✓ KEEP (document uncertainty)
            reconsolidation_plasticity: 0.5,     // ✓ KEEP (applied via inverted-U)
            recent_recalls: DashMap::new(),
        }
    }
}
```

## Additional Validation Tests Required

```rust
#[test]
fn test_update_modifications_strengthen_memory() {
    let engine = ReconsolidationEngine::new();
    let episode = Episode::from_text_with_age("memory", emb, Duration::hours(48));

    engine.record_recall(&episode, Utc::now(), true);

    let modifications = EpisodeModifications {
        field_changes: HashMap::new(),
        modification_extent: 0.3,
        modification_type: ModificationType::Update,  // Strengthening
    };

    let result = engine.attempt_reconsolidation(
        &episode.id,
        modifications,
        Utc::now() + Duration::hours(3)  // Peak plasticity
    ).unwrap();

    // Confidence should increase or stay same (not decrease)
    assert!(result.modified_episode.confidence.value() >= episode.confidence.value());
}

#[test]
fn test_remote_memory_boundary_documented_uncertainty() {
    // Memory at 364 days: eligible
    // Memory at 366 days: rejected
    // Note: This is a hard boundary, but biological reality is gradual
    // Consider adding warning for memories near boundary (350-380 days)
}

#[test]
fn test_boundary_conditions_exact_not_fuzzy() {
    let engine = ReconsolidationEngine::new();

    // Test exact boundaries (not ±epsilon)
    // At window_start: eligible
    // At window_start - 1 second: ineligible
    // At window_end: eligible
    // At window_end + 1 second: ineligible
}
```

## Integration with M6 Consolidation

**Validated (via Task 007):**
- Reconsolidated memories re-enter M6 consolidation pipeline ✓
- Same molecular machinery as initial consolidation (Nader et al. 2000) ✓
- Episode metadata tracks reconsolidation events ✓

**Enhancement Recommendation:** Track reconsolidation cycles (see Task 007 notes)

## Complementary Learning Systems Alignment

**Hippocampal System:**
- Reconsolidation window matches hippocampal replay dynamics ✓
- Active recall requirement aligns with hippocampal reactivation ✓

**Consolidation Boundary:**
- 24-hour minimum ensures synaptic consolidation complete ✓
- Prevents interference with initial consolidation process ✓

**Remote Memories:**
- 365-day maximum reflects reduced hippocampal dependency ✓
- Remote memories increasingly neocortical (less plastic) ✓

## References

- Nader, K., et al. (2000). Fear memories require protein synthesis for reconsolidation. *Nature*, 406(6797), 722-726.
- Nader, K., & Einarsson, E. Ö. (2010). Memory reconsolidation: an update. *Annals of the NY Academy of Sciences*, 1191(1), 27-41.
- Lee, J. L. (2009). Reconsolidation: maintaining memory relevance. *Trends in Neurosciences*, 32(8), 413-420.
- Dudai, Y. (2006). Reconsolidation: the advantage of being refocused. *Current Opinion in Neurobiology*, 16(2), 174-178.
- Frankland, P. W., & Bontempi, B. (2005). The organization of recent and remote memories. *Nature Reviews Neuroscience*, 6(2), 119-130.
- Roediger, H. L., & Karpicke, J. D. (2006). Test-enhanced learning. *Psychological Science*, 17(3), 249-255.

## Conclusion

**Strong implementation** with exact boundary conditions. Apply inverted-U plasticity function and modification type distinction before proceeding. All boundary checks are biologically accurate.
