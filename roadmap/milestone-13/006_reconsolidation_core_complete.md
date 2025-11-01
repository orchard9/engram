# Task 006: Reconsolidation Engine Core

**Status:** Pending
**Priority:** P0 (Critical Path)
**Estimated Effort:** 3 days
**Dependencies:** None (independent cognitive system)

## Objective

Implement memory reconsolidation engine with exact boundary conditions from Nader et al. (2000). Memories become labile (modifiable) during 1-6 hour window post-recall, allowing updates before re-stabilization. Boundary conditions must be implemented exactly, not approximately.

## Theoretical Foundation

**Biology (Nader et al. 2000):**
Memories undergo reconsolidation when:
1. They are actively recalled (not passive exposure)
2. Within 1-6 hour window post-recall
3. Memory is consolidated (>24 hours old)
4. Memory is not too remote (< ~1 year, boundary less precise)

**Critical Insight:**
This is NOT gradual decay. Outside the reconsolidation window, memories are stable and resistant to modification. Inside the window, they exhibit heightened plasticity. Boundary violations must be rejected, not degraded.

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

### 1. Reconsolidation Engine

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
    max_memory_age: Duration,

    /// Plasticity during reconsolidation window (default: 0.5)
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
            modifications,
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
    /// 0.0 = start of window (maximum plasticity)
    /// 0.5 = middle of window
    /// 1.0 = end of window (minimum plasticity)
    fn compute_window_position(&self, time_since_recall: Duration) -> f32 {
        let window_duration = self.window_end - self.window_start;
        let offset = time_since_recall - self.window_start;

        let position = offset.num_milliseconds() as f64
            / window_duration.num_milliseconds() as f64;

        position.clamp(0.0, 1.0) as f32
    }

    /// Compute plasticity factor based on window position
    ///
    /// Plasticity decreases linearly across window (empirical basis: Lee 2009)
    fn compute_plasticity(&self, window_position: f32) -> f32 {
        // Linear decrease: max plasticity at start, min at end
        self.reconsolidation_plasticity * (1.0 - window_position)
    }

    /// Apply modifications during reconsolidation window
    ///
    /// Modifications are weighted by plasticity factor. Higher plasticity
    /// allows greater changes to original memory.
    fn apply_modifications(
        &self,
        original: &Episode,
        modifications: EpisodeModifications,
        window_position: f32
    ) -> Episode {
        let plasticity = self.compute_plasticity(window_position);

        let mut modified = original.clone();

        // Apply field-level modifications weighted by plasticity
        for (field_name, new_value) in modifications.field_changes {
            // Blend original and new value based on plasticity
            // plasticity = 0.0 → keep original
            // plasticity = 1.0 → full replacement
            modified.update_field_with_blending(&field_name, new_value, plasticity);
        }

        // Update confidence based on modification extent
        modified.confidence = self.compute_modified_confidence(
            original.confidence,
            modifications.modification_extent,
            plasticity
        );

        // Mark as reconsolidated
        modified.metadata.insert("reconsolidated".to_string(), "true".to_string());
        modified.metadata.insert(
            "reconsolidation_plasticity".to_string(),
            plasticity.to_string()
        );

        modified
    }

    /// Compute confidence of modified memory
    ///
    /// Confidence decreases with extent of modification and plasticity
    fn compute_modified_confidence(
        &self,
        original_confidence: Confidence,
        modification_extent: f32,
        plasticity: f32
    ) -> Confidence {
        // Reduction proportional to modification extent and plasticity
        let reduction = modification_extent * plasticity * 0.2; // 20% max reduction

        let new_confidence = original_confidence.value() * (1.0 - reduction);

        Confidence::new(new_confidence.max(0.1)) // Floor at 10%
    }

    fn compute_modification_confidence(&self, window_position: f32) -> Confidence {
        // Higher confidence at start of window, lower at end
        let confidence_value = 0.9 - (window_position * 0.4);
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

/// Episode modifications to apply during reconsolidation
pub struct EpisodeModifications {
    /// Field-level changes: field_name -> new_value
    pub field_changes: HashMap<String, String>,

    /// Overall modification extent [0, 1]
    /// 0.0 = minimal changes, 1.0 = extensive changes
    pub modification_extent: f32,
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

### 2. Comprehensive Validation Tests

```rust
// /engram-core/tests/cognitive/reconsolidation_tests.rs

use engram_core::cognitive::reconsolidation::{ReconsolidationEngine, EpisodeModifications};
use engram_core::Episode;
use chrono::{Utc, Duration};
use std::collections::HashMap;

#[test]
fn test_reconsolidation_window_start_boundary() {
    let engine = ReconsolidationEngine::new();
    let episode = Episode::from_text("test memory", random_embedding());

    let recall_time = Utc::now();
    engine.record_recall(&episode, recall_time, true);

    // Attempt modification at 59 minutes (before window start)
    let too_early = recall_time + Duration::minutes(59);
    let result = engine.attempt_reconsolidation(
        &episode.id,
        minimal_modifications(),
        too_early
    );

    assert!(result.is_none(), "Should reject modification before window start");

    // Attempt at exactly 1 hour (window start)
    let at_start = recall_time + Duration::hours(1);
    let result = engine.attempt_reconsolidation(
        &episode.id,
        minimal_modifications(),
        at_start
    );

    assert!(result.is_some(), "Should accept modification at window start");
}

#[test]
fn test_reconsolidation_window_end_boundary() {
    let engine = ReconsolidationEngine::new();
    let episode = Episode::from_text("test memory", random_embedding());

    let recall_time = Utc::now();
    engine.record_recall(&episode, recall_time, true);

    // Attempt at exactly 6 hours (window end)
    let at_end = recall_time + Duration::hours(6);
    let result = engine.attempt_reconsolidation(
        &episode.id,
        minimal_modifications(),
        at_end
    );

    assert!(result.is_some(), "Should accept modification at window end");

    // Attempt at 6 hours 1 minute (after window end)
    let too_late = recall_time + Duration::hours(6) + Duration::minutes(1);
    let result = engine.attempt_reconsolidation(
        &episode.id,
        minimal_modifications(),
        too_late
    );

    assert!(result.is_none(), "Should reject modification after window end");
}

#[test]
fn test_minimum_memory_age_boundary() {
    let engine = ReconsolidationEngine::new();

    // Recent memory (23 hours old - too young)
    let recent_timestamp = Utc::now() - Duration::hours(23);
    let recent_episode = Episode::from_text_with_timestamp(
        "recent memory",
        random_embedding(),
        recent_timestamp
    );

    let recall_time = Utc::now();
    engine.record_recall(&recent_episode, recall_time, true);

    let mod_time = recall_time + Duration::hours(2); // Within window
    let result = engine.attempt_reconsolidation(
        &recent_episode.id,
        minimal_modifications(),
        mod_time
    );

    assert!(result.is_none(), "Should reject unconsolidated memory (<24h)");

    // Consolidated memory (25 hours old - eligible)
    let consolidated_timestamp = Utc::now() - Duration::hours(25);
    let consolidated_episode = Episode::from_text_with_timestamp(
        "consolidated memory",
        random_embedding(),
        consolidated_timestamp
    );

    engine.record_recall(&consolidated_episode, recall_time, true);

    let result = engine.attempt_reconsolidation(
        &consolidated_episode.id,
        minimal_modifications(),
        mod_time
    );

    assert!(result.is_some(), "Should accept consolidated memory (>24h)");
}

#[test]
fn test_maximum_memory_age_boundary() {
    let engine = ReconsolidationEngine::new();

    // Remote memory (366 days old - too old)
    let remote_timestamp = Utc::now() - Duration::days(366);
    let remote_episode = Episode::from_text_with_timestamp(
        "remote memory",
        random_embedding(),
        remote_timestamp
    );

    let recall_time = Utc::now();
    engine.record_recall(&remote_episode, recall_time, true);

    let mod_time = recall_time + Duration::hours(2); // Within window
    let result = engine.attempt_reconsolidation(
        &remote_episode.id,
        minimal_modifications(),
        mod_time
    );

    assert!(result.is_none(), "Should reject remote memory (>365 days)");

    // Old but eligible memory (364 days old)
    let eligible_timestamp = Utc::now() - Duration::days(364);
    let eligible_episode = Episode::from_text_with_timestamp(
        "eligible old memory",
        random_embedding(),
        eligible_timestamp
    );

    engine.record_recall(&eligible_episode, recall_time, true);

    let result = engine.attempt_reconsolidation(
        &eligible_episode.id,
        minimal_modifications(),
        mod_time
    );

    assert!(result.is_some(), "Should accept memory within age range");
}

#[test]
fn test_active_recall_requirement() {
    let engine = ReconsolidationEngine::new();
    let episode = Episode::from_text_with_age("test", random_embedding(), Duration::hours(48));

    let recall_time = Utc::now();

    // Passive re-exposure (is_active = false)
    engine.record_recall(&episode, recall_time, false);

    let mod_time = recall_time + Duration::hours(2);
    let result = engine.attempt_reconsolidation(
        &episode.id,
        minimal_modifications(),
        mod_time
    );

    assert!(result.is_none(), "Should reject passive re-exposure");

    // Active recall (is_active = true)
    engine.record_recall(&episode, recall_time, true);

    let result = engine.attempt_reconsolidation(
        &episode.id,
        minimal_modifications(),
        mod_time
    );

    assert!(result.is_some(), "Should accept active recall");
}

#[test]
fn test_plasticity_decreases_across_window() {
    let engine = ReconsolidationEngine::new();
    let episode = Episode::from_text_with_age("test", random_embedding(), Duration::hours(48));

    let recall_time = Utc::now();
    engine.record_recall(&episode, recall_time, true);

    // Early in window (1 hour post-recall)
    let early_time = recall_time + Duration::hours(1);
    let early_result = engine.attempt_reconsolidation(
        &episode.id,
        moderate_modifications(),
        early_time
    ).unwrap();

    // Late in window (5.5 hours post-recall)
    engine.record_recall(&episode, recall_time, true); // Re-record
    let late_time = recall_time + Duration::hours(5) + Duration::minutes(30);
    let late_result = engine.attempt_reconsolidation(
        &episode.id,
        moderate_modifications(),
        late_time
    ).unwrap();

    // Plasticity should decrease across window
    assert!(
        early_result.plasticity_factor > late_result.plasticity_factor,
        "Plasticity should be higher early in window ({:.3}) than late ({:.3})",
        early_result.plasticity_factor,
        late_result.plasticity_factor
    );
}

fn minimal_modifications() -> EpisodeModifications {
    EpisodeModifications {
        field_changes: HashMap::new(),
        modification_extent: 0.1,
    }
}

fn moderate_modifications() -> EpisodeModifications {
    let mut changes = HashMap::new();
    changes.insert("detail".to_string(), "modified detail".to_string());

    EpisodeModifications {
        field_changes: changes,
        modification_extent: 0.5,
    }
}
```

## Acceptance Criteria

1. **Boundary Conditions (Exact per Nader et al. 2000):**
   - Window: 1-6 hours post-recall (exact)
   - Min age: 24 hours (exact)
   - Max age: 365 days (exact)
   - Active recall required (exact)

2. **Functional Requirements:**
   - Modifications accepted only within window
   - Rejections include clear reason
   - Plasticity decreases linearly across window
   - Original episode preserved for auditing

3. **Performance:**
   - Eligibility check: <50μs
   - Modification application: <100μs
   - Memory: <100KB for 1K tracked recalls

4. **Testing:**
   - All four boundary conditions tested at exact thresholds
   - Plasticity gradient validated
   - Edge cases (negative time, future dates) handled

## Follow-ups

- Task 007: Integration with consolidation system (M6)
- Task 013: Performance validation under load
