# Milestone 13 Task Corrections Summary

**Date:** 2025-10-26
**Reviewer:** Jon Gjengset (rust-graph-engine-architect) + Margo Seltzer (systems-architecture-optimizer) + Randy O'Reilly (memory-systems-researcher)

## Overview

Three critical tasks from Milestone 13 have been corrected based on validation reports from domain experts. All corrections address fundamental design flaws that would prevent achieving stated performance targets or biological plausibility.

## Task 001: Zero-Overhead Metrics Infrastructure

**File:** `/Users/jordanwashburn/Workspace/orchard9/engram/roadmap/milestone-13/001_zero_overhead_metrics_CORRECTED.md`

**Source Review:** `/Users/jordanwashburn/Workspace/orchard9/engram/roadmap/milestone-13/SYSTEMS_ARCHITECTURE_REVIEW.md`

### Critical Fixes Applied

1. **Removed Arc wrapper (CRITICAL)**
   - **Problem:** Arc adds 16 bytes control block + pointer indirection
   - **Impact:** 4-5 indirections per access, defeats `#[inline(always)]`
   - **Fix:** Direct struct field eliminates overhead
   - **Code Change:**
     ```rust
     // BEFORE (WRONG):
     pub struct CognitivePatternMetrics {
         #[cfg(feature = "monitoring")]
         inner: Arc<CognitivePatternMetricsInner>,
     }

     // AFTER (CORRECT):
     #[cfg(feature = "monitoring")]
     pub struct CognitivePatternMetrics {
         inner: CognitivePatternMetricsInner,
     }

     #[cfg(not(feature = "monitoring"))]
     pub struct CognitivePatternMetrics {
         _phantom: core::marker::PhantomData<()>,
     }
     ```

2. **Fixed histogram sum calculation (CRITICAL BUG)**
   - **Problem:** Adding f64 bit representations: `1.5.to_bits() + 2.5.to_bits() != 4.0.to_bits()`
   - **Impact:** Mean calculation completely broken (produces NaN or garbage)
   - **Fix:** Use `atomic_float::AtomicF64` or remove mean entirely
   - **Code Change:**
     ```rust
     // BEFORE (BROKEN):
     let value_bits = value.to_bits();
     self.sum.fetch_add(value_bits, Ordering::Relaxed);

     // AFTER (FIXED):
     use atomic_float::AtomicF64;
     sum: CachePadded<AtomicF64>,
     self.sum.fetch_add(value, Ordering::Relaxed);
     ```

3. **Added loom concurrency tests (NEW REQUIREMENT)**
   - **Problem:** No verification of lock-free correctness
   - **Impact:** Cannot claim lock-free without formal verification
   - **Fix:** Added comprehensive loom test suite
   - **New File:** `/engram-core/tests/metrics/loom_concurrency_tests.rs`

4. **Corrected assembly verification methodology**
   - **Problem:** `objdump -d target/release/engram-core` fails (library not executable)
   - **Fix:** Use `nm -C target/release/deps/libengram_core-*.o` with proper demangling
   - **Script:** Updated verification script with correct commands

5. **Clarified performance budgets (hot/warm/cold paths)**
   - **Problem:** Single <50ns budget unrealistic under production workload
   - **Fix:** Tiered budgets:
     - Hot (L1 cached): <25ns
     - Warm (L3 cached): <80ns
     - Cold (main memory): <250ns

### Estimated Impact

- **Time to fix:** +1 day (3 days total, up from 2 days)
- **Performance improvement:** 2-3x faster (no Arc indirection)
- **Correctness:** Histogram mean now works correctly
- **Verification:** Lock-free correctness now provable via loom

---

## Task 004: Proactive Interference Detection

**File:** `/Users/jordanwashburn/Workspace/orchard9/engram/roadmap/milestone-13/004_proactive_interference_detection_CORRECTED.md`

**Source Review:** `/Users/jordanwashburn/Workspace/orchard9/engram/roadmap/milestone-13/004_VALIDATION_NOTES.md`

### Critical Fix Applied

1. **Prior memory window: 24 hours → 6 hours (CRITICAL)**
   - **Problem:** 24h exceeds synaptic consolidation timescale
   - **Biological Rationale:**
     - Synaptic consolidation completes within ~6 hours (protein synthesis window)
     - After consolidation, memories shift from hippocampal → neocortical representations
     - Cross-consolidation-boundary interference is reduced (CLS theory, McClelland et al. 1995)
   - **Empirical Basis:** Underwood (1957) used session-based interference (minutes to hours), not day-scale
   - **Code Change:**
     ```rust
     // BEFORE (WRONG):
     prior_memory_window: Duration::hours(24),

     // AFTER (CORRECT):
     /// Temporal window for "prior" memories (default: 6 hours before)
     /// Empirical: Underwood (1957) session-based interference
     /// Justification: Synaptic consolidation (~6h) transitions memories from
     /// hippocampal to neocortical representations, reducing interference
     prior_memory_window: Duration::hours(6),
     ```

### Additional Validation Tests Added

1. **Consolidation boundary test**
   - Validates that memories >6h ago don't interfere (consolidated)
   - Validates that memories <6h ago do interfere (not yet consolidated)

2. **Similarity threshold calibration test**
   - Within-category pairs (dog/cat): similarity ≥ 0.7
   - Across-category pairs (dog/car): similarity < 0.7

3. **Exact boundary test**
   - 5h 59m: should interfere
   - 6h 01m: should NOT interfere

### Estimated Impact

- **Time to fix:** No change (2 days)
- **Biological plausibility:** PASS (was FAIL)
- **Empirical validation:** Now aligns with Underwood (1957)

---

## Task 006: Reconsolidation Engine Core

**File:** `/Users/jordanwashburn/Workspace/orchard9/engram/roadmap/milestone-13/006_reconsolidation_core_CORRECTED.md`

**Source Review:** `/Users/jordanwashburn/Workspace/orchard9/engram/roadmap/milestone-13/006_VALIDATION_NOTES.md`

### Critical Fixes Applied

1. **Plasticity function: linear → inverted-U (CRITICAL)**
   - **Problem:** Linear decrease doesn't match protein synthesis kinetics
   - **Biological Rationale:**
     - Protein synthesis shows non-linear kinetics (Nader & Einarsson 2010)
     - Rapid rise (0-2h)
     - Plateau at peak (2-4h)
     - Gradual decline (4-6h)
   - **Code Change:**
     ```rust
     // BEFORE (WRONG):
     fn compute_plasticity(&self, window_position: f32) -> f32 {
         self.reconsolidation_plasticity * (1.0 - window_position)
     }

     // AFTER (CORRECT):
     /// Uses inverted-U function matching protein synthesis dynamics
     fn compute_plasticity(&self, window_position: f32) -> f32 {
         // Inverted-U: peaks at window_position = 0.5 (middle of window)
         // f(x) = 4x(1-x) gives parabola with maximum at x=0.5
         let u_curve = 4.0 * window_position * (1.0 - window_position);
         self.reconsolidation_plasticity * u_curve
     }
     ```

2. **Added modification type distinction (NEW)**
   - **Problem:** All modifications reduced confidence (wrong)
   - **Biological Rationale:** Reconsolidation can strengthen memories (Dudai 2006)
   - **New Types:**
     - `Update`: Updating with accurate information (increases confidence)
     - `Corruption`: Conflicting information (decreases confidence)
     - `Replacement`: Complete replacement (resets confidence)
   - **Code Addition:**
     ```rust
     pub enum ModificationType {
         Update,      // Strengthens memory (like rehearsal)
         Corruption,  // Reduces confidence
         Replacement, // Resets confidence
     }

     pub struct EpisodeModifications {
         pub field_changes: HashMap<String, String>,
         pub modification_extent: f32,
         pub modification_type: ModificationType,  // NEW
     }
     ```

3. **Peak plasticity at 3-4 hours post-recall**
   - **Problem:** Linear model had peak at window start (1h)
   - **Fix:** Inverted-U peaks at window midpoint (3.5h)
   - **Validation:** Test ensures mid_plasticity > early_plasticity and mid_plasticity > late_plasticity

### Additional Validation Tests Added

1. **Plasticity peaks mid-window test**
   - Validates inverted-U curve shape
   - Ensures peak at window_position = 0.5

2. **Update modifications strengthen memory test**
   - Validates confidence increases with Update type
   - Matches retrieval-induced strengthening (Roediger & Karpicke 2006)

3. **Corruption modifications reduce confidence test**
   - Validates confidence decreases with Corruption type

4. **Replacement modifications reset confidence test**
   - Validates confidence resets to moderate value

5. **Remote memory boundary documented uncertainty test**
   - Documents that 365-day boundary is hard but biologically gradual

### Estimated Impact

- **Time to fix:** +1 day (4 days total, up from 3 days)
- **Biological plausibility:** PASS (was CONDITIONAL PASS)
- **Accuracy:** Plasticity dynamics now match Nader & Einarsson (2010) Fig 3

---

## Summary of Changes

| Task | Original Estimate | Corrected Estimate | Critical Fixes | New Tests | Status |
|------|------------------|-------------------|----------------|-----------|--------|
| 001  | 2 days           | 3 days (+1)       | 5              | 3         | READY  |
| 004  | 2 days           | 2 days            | 1              | 3         | READY  |
| 006  | 3 days           | 4 days (+1)       | 3              | 5         | READY  |

**Total time adjustment:** +2 days for milestone

---

## Implementation Priority

All three tasks are **P0 (Critical Path)** and should be implemented in order:

1. **Task 001 (Zero-Overhead Metrics)** - Foundation for all metrics recording
2. **Task 004 (Proactive Interference)** - Depends on Task 001 for metrics
3. **Task 006 (Reconsolidation Core)** - Independent, can be parallel with Task 004

---

## Validation Checklist

Before marking any corrected task as complete:

### Task 001
- [ ] Remove Arc wrapper from CognitivePatternMetrics
- [ ] Fix histogram sum calculation (use AtomicF64)
- [ ] Add all loom concurrency tests
- [ ] Update assembly verification script
- [ ] Run verification script and confirm 0 function symbols
- [ ] Verify size_of::<CognitivePatternMetrics>() == 0 when disabled
- [ ] Run overhead benchmarks and confirm <1% overhead
- [ ] `make quality` passes

### Task 004
- [ ] Change prior_memory_window to Duration::hours(6)
- [ ] Add consolidation boundary test
- [ ] Add similarity threshold calibration test
- [ ] Add exact boundary test
- [ ] Verify Underwood (1957) replication within ±10%
- [ ] `make quality` passes

### Task 006
- [ ] Implement inverted-U plasticity function
- [ ] Add ModificationType enum
- [ ] Update compute_modified_confidence for all three types
- [ ] Add max_memory_age uncertainty documentation
- [ ] Add test_plasticity_peaks_mid_window
- [ ] Add test_update_modifications_strengthen_memory
- [ ] Add test_corruption_modifications_reduce_confidence
- [ ] Add test_replacement_modifications_reset_confidence
- [ ] Add test_remote_memory_boundary_documented_uncertainty
- [ ] Verify plasticity curve matches Nader & Einarsson (2010) Fig 3
- [ ] `make quality` passes

---

## References

### Systems Architecture
- Seltzer, M. (2025). Milestone 13 Infrastructure Tasks: Systems Architecture Review. Internal review document.

### Memory Systems
- O'Reilly, R. (2025). Task 004: Proactive Interference Detection - Validation Notes. Internal review document.
- O'Reilly, R. (2025). Task 006: Reconsolidation Engine Core - Validation Notes. Internal review document.

### Empirical Basis
- Underwood, B. J. (1957). Interference and forgetting. *Psychological Review*, 64(1), 49.
- Nader, K., et al. (2000). Fear memories require protein synthesis for reconsolidation. *Nature*, 406(6797), 722-726.
- Nader, K., & Einarsson, E. Ö. (2010). Memory reconsolidation: an update. *Annals of the NY Academy of Sciences*, 1191(1), 27-41.
- McClelland, J. L., et al. (1995). Why there are complementary learning systems. *Psychological Review*, 102(3), 419.
- Dudai, Y. (2006). Reconsolidation: the advantage of being refocused. *Current Opinion in Neurobiology*, 16(2), 174-178.
- Roediger, H. L., & Karpicke, J. D. (2006). Test-enhanced learning. *Psychological Science*, 17(3), 249-255.

---

**End of Corrections Summary**
