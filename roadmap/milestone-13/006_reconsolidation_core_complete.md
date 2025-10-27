# Task 006: Reconsolidation Engine Core (CORRECTED)

**Status:** Pending
**Priority:** P0 (Critical Path)
**Estimated Effort:** 4 days (extended from 3 days for corrections)
**Dependencies:** None (independent cognitive system)

## Objective

Implement memory reconsolidation engine with exact boundary conditions from Nader et al. (2000). Memories become labile (modifiable) during 1-6 hour window post-recall, allowing updates before re-stabilization. Boundary conditions must be implemented exactly, not approximately.

**CRITICAL CORRECTIONS APPLIED:**
1. Plasticity function: linear → inverted-U (protein synthesis kinetics)
2. Add modification type distinction (Update/Corruption/Replacement)
3. Peak plasticity at 3-4 hours post-recall (not start of window)

## Theoretical Foundation

**Biology (Nader et al. 2000):**
Memories undergo reconsolidation when:
1. They are actively recalled (not passive exposure)
2. Within 1-6 hour window post-recall
3. Memory is consolidated (>24 hours old)
4. Memory is not too remote (< ~1 year, boundary less precise)

**Critical Insight:**
This is NOT gradual decay. Outside the reconsolidation window, memories are stable and resistant to modification. Inside the window, they exhibit heightened plasticity. Boundary violations must be rejected, not degraded.

**CORRECTED: Plasticity Dynamics (Nader & Einarsson 2010)**
Protein synthesis shows non-linear kinetics:
- Rapid rise (0-2h)
- Plateau at peak (2-4h) - MAXIMUM PLASTICITY
- Gradual decline (4-6h)

This requires inverted-U function, not linear decrease.

## Integration Points

**Creates:**
- `/engram-core/src/cognitive/reconsolidation/mod.rs` - Core reconsolidation engine
- `/engram-core/src/cognitive/reconsolidation/window.rs` - Temporal window logic
- `/engram-core/src/cognitive/reconsolidation/boundary.rs` - Boundary condition checks
- `/engram-core/tests/cognitive/reconsolidation_tests.rs` - Comprehensive validation

**Uses:**
- `/engram-core/src/memory.rs` - Episode modification
- `/engram-core/src/metrics/cognitive_patterns.rs` - Metrics recording

**Future Integration (Task 007):**
- `/engram-core/src/decay/consolidation.rs` - Re-enter consolidation pipeline

## Detailed Specification

### 1. Reconsolidation Engine (CORRECTED)

```rust
// /engram-core/src/cognitive/reconsolidation/mod.rs

use chrono::{DateTime, Utc, Duration};
use crate::{Episode, Confidence};
use dashmap::DashMap;
use std::collections::HashMap;

/// Memory reconsolidation engine with exact Nader et al. (2000) boundary conditions
///
/// Memories become labile during recall, allowing modification within specific
/// temporal window. Outside this window, modifications are rejected.
///
/// # Boundary Conditions (Exact)
/// 1. Reconsolidation window: 1-6 hours post-recall
/// 2. Minimum memory age: 24 hours (must be consolidated)
/// 3. Maximum memory age: 365 days (remote memories less plastic)
/// 4. Requires active recall (not passive re-exposure)
///
/// # Plasticity Dynamics (CORRECTED)
/// Uses inverted-U function matching protein synthesis kinetics:
/// - Early (1h): Rising plasticity
/// - Peak (3-4h): Maximum plasticity
/// - Late (5-6h): Declining plasticity
pub struct ReconsolidationEngine {
    /// Reconsolidation window start (default: 1 hour = 3600 seconds)
    /// Empirical basis: Nader et al. (2000) protein synthesis window
    window_start: Duration,

    /// Reconsolidation window end (default: 6 hours = 21600 seconds)
    /// Empirical basis: Lee (2009) reconsolidation boundaries
    window_end: Duration,

    /// Minimum memory age for reconsolidation (default: 24 hours)
    /// Memories must be consolidated before they can reconsolidate
    min_memory_age: Duration,

    /// Maximum memory age for reconsolidation (default: 365 days)
    /// Very old memories show reduced plasticity (boundary less precise)
    /// NOTE: This boundary is less precise than others. Remote memories
    /// show reduced but not absent plasticity. Recommend domain-specific
    /// tuning (semantic memories may remain plastic longer than episodic).
    max_memory_age: Duration,

    /// Maximum plasticity during reconsolidation window (default: 0.5)
    /// Applied via inverted-U function peaking at window midpoint
    /// 0.0 = no modification allowed, 1.0 = complete replacement
    reconsolidation_plasticity: f32,

    /// Tracking of recent recalls for reconsolidation opportunities
    /// Key: episode_id, Value: recall event metadata
    recent_recalls: DashMap<String, RecallEvent>,
}

/// Metadata for a recall event that may trigger reconsolidation
#[derive(Debug, Clone)]
struct RecallEvent {
    /// Episode ID that was recalled
    episode_id: String,

    /// Timestamp when recall occurred
    recall_timestamp: DateTime<Utc>,

    /// Original episode snapshot at time of recall
    original_episode: Episode,

    /// Whether this was an active recall (required for reconsolidation)
    is_active_recall: bool,
}

impl ReconsolidationEngine {
    /// Create new reconsolidation engine with Nader et al. (2000) parameters
    #[must_use]
    pub fn new() -> Self {
        Self {
            window_start: Duration::hours(1),    // 1 hour
            window_end: Duration::hours(6),      // 6 hours
            min_memory_age: Duration::hours(24), // 24 hours
            max_memory_age: Duration::days(365), // 1 year
            reconsolidation_plasticity: 0.5,
            recent_recalls: DashMap::new(),
        }
    }

    /// Record a recall event for potential reconsolidation
    ///
    /// Must be called immediately after recall to track timing accurately.
    ///
    /// # Arguments
    /// * `episode` - The episode that was recalled
    /// * `recall_time` - Timestamp of recall (use `Utc::now()` for current)
    /// * `is_active` - Whether this was active recall vs passive re-exposure
    pub fn record_recall(
        &self,
        episode: &Episode,
        recall_time: DateTime<Utc>,
        is_active: bool
    ) {
        self.recent_recalls.insert(
            episode.id.clone(),
            RecallEvent {
                episode_id: episode.id.clone(),
                recall_timestamp: recall_time,
                original_episode: episode.clone(),
                is_active_recall: is_active,
            }
        );

        // Prune old recall events outside maximum window
        self.prune_old_recalls(recall_time);
    }

    /// Attempt to modify a memory through reconsolidation
    ///
    /// Returns `Some(result)` if within reconsolidation window and boundary
    /// conditions met. Returns `None` if outside window or ineligible.
    ///
    /// # Boundary Condition Checks (Exact)
    /// 1. Time since recall: [window_start, window_end]
    /// 2. Memory age: [min_memory_age, max_memory_age]
    /// 3. Active recall: must be true
    ///
    /// # Arguments
    /// * `episode_id` - ID of episode to modify
    /// * `modifications` - Field-level changes to apply
    /// * `current_time` - Current timestamp for window calculation
    ///
    /// # Returns
    /// * `Some(result)` - Modification succeeded, returns modified episode
    /// * `None` - Outside window or boundary conditions not met
    pub fn attempt_reconsolidation(
        &self,
        episode_id: &str,
        modifications: EpisodeModifications,
        current_time: DateTime<Utc>
    ) -> Option<ReconsolidationResult> {
        // Retrieve recall event
        let recall_event = self.recent_recalls.get(episode_id)?;

        // Check boundary conditions (EXACT, not fuzzy)
        let eligibility = self.check_eligibility(&recall_event, current_time);
        if !eligibility.is_eligible {
            // Record rejection reason in metrics
            #[cfg(feature = "monitoring")]
            {
                crate::metrics::cognitive_patterns()
                    .record_reconsolidation_rejection(&eligibility.rejection_reason);
            }

            return None;
        }

        // Compute position within reconsolidation window [0, 1]
        let time_since_recall = current_time - recall_event.recall_timestamp;
        let window_position = self.compute_window_position(time_since_recall);

        // Apply modifications with plasticity modulation
        let modified_episode = self.apply_modifications(
            &recall_event.original_episode,
            &modifications,
            window_position
        );

        // Record successful reconsolidation
        #[cfg(feature = "monitoring")]
        {
            crate::metrics::cognitive_patterns()
                .record_reconsolidation(window_position);
        }

        Some(ReconsolidationResult {
            modified_episode,
            window_position,
            plasticity_factor: self.compute_plasticity(window_position),
            modification_confidence: self.compute_modification_confidence(window_position),
            original_episode: recall_event.original_episode.clone(),
        })
    }

    /// Check all boundary conditions for reconsolidation eligibility
    ///
    /// Returns detailed eligibility with rejection reason if ineligible.
    fn check_eligibility(
        &self,
        recall: &RecallEvent,
        current_time: DateTime<Utc>
    ) -> EligibilityResult {
        // Boundary 1: Must be active recall
        if !recall.is_active_recall {
            return EligibilityResult::ineligible(
                "Reconsolidation requires active recall, not passive re-exposure"
            );
        }

        // Boundary 2: Time since recall must be within window
        let time_since_recall = current_time - recall.recall_timestamp;

        if time_since_recall < self.window_start {
            return EligibilityResult::ineligible(
                &format!(
                    "Too soon after recall: {} < {} (window starts at {})",
                    format_duration(time_since_recall),
                    format_duration(self.window_start),
                    format_duration(self.window_start)
                )
            );
        }

        if time_since_recall > self.window_end {
            return EligibilityResult::ineligible(
                &format!(
                    "Too late after recall: {} > {} (window closed)",
                    format_duration(time_since_recall),
                    format_duration(self.window_end)
                )
            );
        }

        // Boundary 3: Memory must be consolidated (>24 hours old)
        let memory_age = recall.recall_timestamp - recall.original_episode.timestamp;

        if memory_age < self.min_memory_age {
            return EligibilityResult::ineligible(
                &format!(
                    "Memory too recent: {} < {} (must be consolidated)",
                    format_duration(memory_age),
                    format_duration(self.min_memory_age)
                )
            );
        }

        // Boundary 4: Memory must not be too remote (< ~1 year)
        if memory_age > self.max_memory_age {
            return EligibilityResult::ineligible(
                &format!(
                    "Memory too remote: {} > {} (reduced plasticity)",
                    format_duration(memory_age),
                    format_duration(self.max_memory_age)
                )
            );
        }

        // All boundaries satisfied
        EligibilityResult::eligible()
    }

    /// Compute position within reconsolidation window [0, 1]
    ///
    /// 0.0 = start of window (1 hour post-recall)
    /// 0.5 = middle of window (3.5 hours post-recall, near peak plasticity)
    /// 1.0 = end of window (6 hours post-recall)
    fn compute_window_position(&self, time_since_recall: Duration) -> f32 {
        let window_duration = self.window_end - self.window_start;
        let offset = time_since_recall - self.window_start;

        let position = offset.num_milliseconds() as f64
            / window_duration.num_milliseconds() as f64;

        position.clamp(0.0, 1.0) as f32
    }

    /// Compute plasticity factor based on window position
    ///
    /// CORRECTED: Uses inverted-U function matching protein synthesis dynamics
    /// (Nader & Einarsson 2010 Fig 3)
    ///
    /// - Early (position=0.0, 1h): Rising plasticity
    /// - Peak (position=0.5, 3.5h): Maximum plasticity
    /// - Late (position=1.0, 6h): Declining plasticity
    fn compute_plasticity(&self, window_position: f32) -> f32 {
        // Inverted-U: peaks at window_position = 0.5 (middle of window)
        // f(x) = 4x(1-x) gives parabola with maximum at x=0.5
        let u_curve = 4.0 * window_position * (1.0 - window_position);

        self.reconsolidation_plasticity * u_curve
    }

    /// Apply modifications during reconsolidation window
    ///
    /// Modifications are weighted by plasticity factor. Higher plasticity
    /// allows greater changes to original memory.
    fn apply_modifications(
        &self,
        original: &Episode,
        modifications: &EpisodeModifications,
        window_position: f32
    ) -> Episode {
        let plasticity = self.compute_plasticity(window_position);

        let mut modified = original.clone();

        // Apply field-level modifications weighted by plasticity
        for (field_name, new_value) in &modifications.field_changes {
            // Blend original and new value based on plasticity
            // plasticity = 0.0 → keep original
            // plasticity = 1.0 → full replacement
            modified.update_field_with_blending(field_name, new_value.clone(), plasticity);
        }

        // Update confidence based on modification type (CORRECTED)
        modified.confidence = self.compute_modified_confidence(
            original.confidence,
            modifications,
            plasticity
        );

        // Mark as reconsolidated
        modified.metadata.insert("reconsolidated".to_string(), "true".to_string());
        modified.metadata.insert(
            "reconsolidation_plasticity".to_string(),
            plasticity.to_string()
        );
        modified.metadata.insert(
            "modification_type".to_string(),
            format!("{:?}", modifications.modification_type)
        );

        modified
    }

    /// Compute confidence of modified memory (CORRECTED)
    ///
    /// Confidence change depends on modification type:
    /// - Update: Strengthens memory (like rehearsal)
    /// - Corruption: Reduces confidence
    /// - Replacement: Resets confidence
    fn compute_modified_confidence(
        &self,
        original_confidence: Confidence,
        modifications: &EpisodeModifications,
        plasticity: f32
    ) -> Confidence {
        match modifications.modification_type {
            ModificationType::Update => {
                // Strengthens memory (retrieval-induced strengthening)
                // Dudai (2006): Reconsolidation can enhance memory stability
                let boost = plasticity * 0.1;  // Up to 5% boost at peak plasticity (0.5 * 0.1)
                let new_value = (original_confidence.value() * (1.0 + boost)).min(1.0);
                Confidence::new(new_value)
            }
            ModificationType::Corruption => {
                // Reduces confidence proportional to modification extent
                let reduction = modifications.modification_extent * plasticity * 0.2;
                let new_value = (original_confidence.value() * (1.0 - reduction)).max(0.1);
                Confidence::new(new_value)
            }
            ModificationType::Replacement => {
                // Confidence reset to quality of new information
                // Use modification_extent as proxy for new information quality
                Confidence::new(modifications.modification_extent.clamp(0.3, 0.8))
            }
        }
    }

    fn compute_modification_confidence(&self, window_position: f32) -> Confidence {
        // Higher confidence near peak plasticity (window_position = 0.5)
        // Use inverted-U curve same as plasticity
        let u_curve = 4.0 * window_position * (1.0 - window_position);
        let confidence_value = 0.5 + (u_curve * 0.4); // Range [0.5, 0.9]
        Confidence::new(confidence_value)
    }

    /// Remove recall events outside maximum reconsolidation window
    fn prune_old_recalls(&self, current_time: DateTime<Utc>) {
        self.recent_recalls.retain(|_, recall| {
            let time_since = current_time - recall.recall_timestamp;
            time_since <= self.window_end
        });
    }
}

impl Default for ReconsolidationEngine {
    fn default() -> Self {
        Self::new()
    }
}

/// Episode modifications to apply during reconsolidation (CORRECTED)
pub struct EpisodeModifications {
    /// Field-level changes: field_name -> new_value
    pub field_changes: HashMap<String, String>,

    /// Overall modification extent [0, 1]
    /// 0.0 = minimal changes, 1.0 = extensive changes
    pub modification_extent: f32,

    /// Type of modification (ADDED)
    pub modification_type: ModificationType,
}

/// Type of memory modification (NEW)
#[derive(Debug, Clone, Copy)]
pub enum ModificationType {
    /// Updating with new accurate information (maintains/increases confidence)
    /// Example: Adding details to existing memory
    Update,

    /// Partial corruption or uncertainty (decreases confidence)
    /// Example: Conflicting information causing doubt
    Corruption,

    /// Complete replacement (resets confidence)
    /// Example: Correcting false memory with true information
    Replacement,
}

/// Result of reconsolidation attempt
pub struct ReconsolidationResult {
    /// Modified episode after reconsolidation
    pub modified_episode: Episode,

    /// Position within window [0, 1]
    pub window_position: f32,

    /// Plasticity factor applied [0, reconsolidation_plasticity]
    pub plasticity_factor: f32,

    /// Confidence in modification
    pub modification_confidence: Confidence,

    /// Original episode before modification (for auditing)
    pub original_episode: Episode,
}

struct EligibilityResult {
    is_eligible: bool,
    rejection_reason: String,
}

impl EligibilityResult {
    fn eligible() -> Self {
        Self {
            is_eligible: true,
            rejection_reason: String::new(),
        }
    }

    fn ineligible(reason: &str) -> Self {
        Self {
            is_eligible: false,
            rejection_reason: reason.to_string(),
        }
    }
}

fn format_duration(d: Duration) -> String {
    let hours = d.num_hours();
    let minutes = d.num_minutes() % 60;
    format!("{}h {}m", hours, minutes)
}
```

### 2. Comprehensive Validation Tests (CORRECTED)

```rust
// /engram-core/tests/cognitive/reconsolidation_tests.rs

use engram_core::cognitive::reconsolidation::{
    ReconsolidationEngine, EpisodeModifications, ModificationType
};
use engram_core::Episode;
use chrono::{Utc, Duration};
use std::collections::HashMap;

// ... [Previous boundary tests remain unchanged] ...

#[test]
fn test_plasticity_peaks_mid_window() {
    // ADDED: Validates inverted-U plasticity function
    let engine = ReconsolidationEngine::new();

    // Plasticity at 1h < plasticity at 3.5h
    let early_plasticity = engine.compute_plasticity(0.0);  // window_position=0
    let mid_plasticity = engine.compute_plasticity(0.5);    // window_position=0.5 (peak)

    assert!(
        mid_plasticity > early_plasticity,
        "Mid-window plasticity ({:.3}) should exceed early ({:.3})",
        mid_plasticity,
        early_plasticity
    );

    // Plasticity at 3.5h > plasticity at 5.5h
    let late_plasticity = engine.compute_plasticity(0.9);   // window_position=0.9
    assert!(
        mid_plasticity > late_plasticity,
        "Mid-window plasticity ({:.3}) should exceed late ({:.3})",
        mid_plasticity,
        late_plasticity
    );

    // Peak should be near 0.5 reconsolidation_plasticity (inverted-U maximum)
    let expected_peak = 0.5 * 1.0; // reconsolidation_plasticity * u_curve_max
    assert!(
        (mid_plasticity - expected_peak).abs() < 0.05,
        "Peak plasticity {:.3} should be near {:.3}",
        mid_plasticity,
        expected_peak
    );
}

#[test]
fn test_update_modifications_strengthen_memory() {
    // ADDED: Validates Update modification type increases confidence
    let engine = ReconsolidationEngine::new();
    let episode = Episode::from_text_with_age(
        "test memory",
        random_embedding(),
        Duration::hours(48)
    );

    let recall_time = Utc::now();
    engine.record_recall(&episode, recall_time, true);

    let modifications = EpisodeModifications {
        field_changes: HashMap::new(),
        modification_extent: 0.3,
        modification_type: ModificationType::Update,  // Strengthening
    };

    let result = engine.attempt_reconsolidation(
        &episode.id,
        modifications,
        recall_time + Duration::hours(3)  // Peak plasticity
    ).unwrap();

    // Confidence should increase or stay same (not decrease)
    assert!(
        result.modified_episode.confidence.value() >= episode.confidence.value(),
        "Update modification should not decrease confidence: {} -> {}",
        episode.confidence.value(),
        result.modified_episode.confidence.value()
    );
}

#[test]
fn test_corruption_modifications_reduce_confidence() {
    // ADDED: Validates Corruption modification type decreases confidence
    let engine = ReconsolidationEngine::new();
    let episode = Episode::from_text_with_age(
        "test memory",
        random_embedding(),
        Duration::hours(48)
    );

    let recall_time = Utc::now();
    engine.record_recall(&episode, recall_time, true);

    let modifications = EpisodeModifications {
        field_changes: HashMap::new(),
        modification_extent: 0.5,  // Moderate corruption
        modification_type: ModificationType::Corruption,
    };

    let result = engine.attempt_reconsolidation(
        &episode.id,
        modifications,
        recall_time + Duration::hours(3)
    ).unwrap();

    // Confidence should decrease
    assert!(
        result.modified_episode.confidence.value() < episode.confidence.value(),
        "Corruption modification should decrease confidence: {} -> {}",
        episode.confidence.value(),
        result.modified_episode.confidence.value()
    );
}

#[test]
fn test_replacement_modifications_reset_confidence() {
    // ADDED: Validates Replacement modification type resets confidence
    let engine = ReconsolidationEngine::new();
    let episode = Episode::from_text_with_age(
        "test memory",
        random_embedding(),
        Duration::hours(48)
    );

    let recall_time = Utc::now();
    engine.record_recall(&episode, recall_time, true);

    let modifications = EpisodeModifications {
        field_changes: HashMap::from([
            ("content".to_string(), "completely new content".to_string())
        ]),
        modification_extent: 0.7,  // High quality replacement
        modification_type: ModificationType::Replacement,
    };

    let result = engine.attempt_reconsolidation(
        &episode.id,
        modifications,
        recall_time + Duration::hours(3)
    ).unwrap();

    // Confidence should be reset to moderate value (not original)
    let confidence_value = result.modified_episode.confidence.value();
    assert!(
        (0.3..=0.8).contains(&confidence_value),
        "Replacement confidence {} should be in moderate range [0.3, 0.8]",
        confidence_value
    );
}

#[test]
fn test_remote_memory_boundary_documented_uncertainty() {
    // ADDED: Documents that 365-day boundary is hard but biologically gradual
    let engine = ReconsolidationEngine::new();

    // Memory at 364 days: eligible
    let eligible_episode = Episode::from_text_with_age(
        "old memory",
        random_embedding(),
        Duration::days(364)
    );

    let recall_time = Utc::now();
    engine.record_recall(&eligible_episode, recall_time, true);

    let result = engine.attempt_reconsolidation(
        &eligible_episode.id,
        minimal_modifications(),
        recall_time + Duration::hours(2)
    );
    assert!(result.is_some(), "Memory at 364 days should be eligible");

    // Memory at 366 days: rejected
    let remote_episode = Episode::from_text_with_age(
        "very old memory",
        random_embedding(),
        Duration::days(366)
    );

    engine.record_recall(&remote_episode, recall_time, true);

    let result = engine.attempt_reconsolidation(
        &remote_episode.id,
        minimal_modifications(),
        recall_time + Duration::hours(2)
    );
    assert!(result.is_none(), "Memory at 366 days should be rejected");

    // NOTE: This is a hard boundary, but biological reality is gradual.
    // Consider adding warning for memories near boundary (350-380 days).
}

fn minimal_modifications() -> EpisodeModifications {
    EpisodeModifications {
        field_changes: HashMap::new(),
        modification_extent: 0.1,
        modification_type: ModificationType::Update,
    }
}
```

## Acceptance Criteria

1. **Boundary Conditions (Exact per Nader et al. 2000):**
   - Window: 1-6 hours post-recall (exact)
   - Min age: 24 hours (exact)
   - Max age: 365 days (exact, with documented uncertainty)
   - Active recall required (exact)

2. **Plasticity Dynamics (CORRECTED):**
   - Inverted-U function with peak at window midpoint
   - Early window: rising plasticity
   - Peak (3-4h): maximum plasticity
   - Late window: declining plasticity
   - Test validates peak > early and peak > late

3. **Modification Types (NEW):**
   - Update: Strengthens memory (confidence increases)
   - Corruption: Weakens memory (confidence decreases)
   - Replacement: Resets memory (confidence reset)
   - Each type tested independently

4. **Functional Requirements:**
   - Modifications accepted only within window
   - Rejections include clear reason
   - Original episode preserved for auditing
   - Metadata tracks modification type

5. **Performance:**
   - Eligibility check: <50μs
   - Modification application: <100μs
   - Memory: <100KB for 1K tracked recalls

6. **Testing:**
   - All four boundary conditions tested at exact thresholds
   - Plasticity inverted-U validated
   - All three modification types tested
   - Edge cases (negative time, future dates) handled

## Follow-ups

- Task 007: Integration with consolidation system (M6)
- Task 013: Performance validation under load

## Implementation Checklist

- [ ] Implement inverted-U plasticity function (CRITICAL FIX)
- [ ] Add ModificationType enum (Update/Corruption/Replacement)
- [ ] Update compute_modified_confidence to handle modification types
- [ ] Add max_memory_age uncertainty documentation
- [ ] Write test_plasticity_peaks_mid_window (NEW)
- [ ] Write test_update_modifications_strengthen_memory (NEW)
- [ ] Write test_corruption_modifications_reduce_confidence (NEW)
- [ ] Write test_replacement_modifications_reset_confidence (NEW)
- [ ] Write test_remote_memory_boundary_documented_uncertainty (NEW)
- [ ] Update all existing tests to use ModificationType
- [ ] Run `make quality` and fix all clippy warnings
- [ ] Verify plasticity curve matches Nader & Einarsson (2010) Fig 3

## References

1. Nader, K., et al. (2000). Fear memories require protein synthesis for reconsolidation. *Nature*, 406(6797), 722-726.
2. Nader, K., & Einarsson, E. Ö. (2010). Memory reconsolidation: an update. *Annals of the NY Academy of Sciences*, 1191(1), 27-41.
3. Lee, J. L. (2009). Reconsolidation: maintaining memory relevance. *Trends in Neurosciences*, 32(8), 413-420.
4. Dudai, Y. (2006). Reconsolidation: the advantage of being refocused. *Current Opinion in Neurobiology*, 16(2), 174-178.
5. Roediger, H. L., & Karpicke, J. D. (2006). Test-enhanced learning. *Psychological Science*, 17(3), 249-255.
