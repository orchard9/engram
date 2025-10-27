//! Memory reconsolidation engine with exact boundary conditions from Nader et al. (2000).
//!
//! Memories become labile (modifiable) during 1-6 hour window post-recall, allowing
//! updates before re-stabilization. This module implements exact temporal boundaries
//! with inverted-U plasticity dynamics matching protein synthesis kinetics.
//!
//! # Theoretical Foundation
//!
//! **Biology (Nader et al. 2000):**
//! Memories undergo reconsolidation when:
//! 1. They are actively recalled (not passive exposure)
//! 2. Within 1-6 hour window post-recall
//! 3. Memory is consolidated (>24 hours old)
//! 4. Memory is not too remote (< ~1 year, boundary less precise)
//!
//! **Plasticity Dynamics (Nader & Einarsson 2010):**
//! Protein synthesis shows non-linear kinetics:
//! - Rapid rise (0-2h)
//! - Plateau at peak (2-4h) - MAXIMUM PLASTICITY
//! - Gradual decline (4-6h)
//!
//! # References
//!
//! 1. Nader, K., et al. (2000). Fear memories require protein synthesis for reconsolidation. *Nature*, 406(6797), 722-726.
//! 2. Nader, K., & Einarsson, E. Ö. (2010). Memory reconsolidation: an update. *Annals of the NY Academy of Sciences*, 1191(1), 27-41.
//! 3. Lee, J. L. (2009). Reconsolidation: maintaining memory relevance. *Trends in Neurosciences*, 32(8), 413-420.
//! 4. Dudai, Y. (2006). Reconsolidation: the advantage of being refocused. *Current Opinion in Neurobiology*, 16(2), 174-178.

pub mod consolidation_integration;

use crate::{Confidence, Episode};
use chrono::{DateTime, Duration, Utc};
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
/// # Plasticity Dynamics
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
    pub fn record_recall(&self, episode: &Episode, recall_time: DateTime<Utc>, is_active: bool) {
        self.recent_recalls.insert(
            episode.id.clone(),
            RecallEvent {
                recall_timestamp: recall_time,
                original_episode: episode.clone(),
                is_active_recall: is_active,
            },
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
    #[must_use]
    pub fn attempt_reconsolidation(
        &self,
        episode_id: &str,
        modifications: &EpisodeModifications,
        current_time: DateTime<Utc>,
    ) -> Option<ReconsolidationResult> {
        // Retrieve recall event
        let recall_event = self.recent_recalls.get(episode_id)?;

        // Check boundary conditions (EXACT, not fuzzy)
        let eligibility = self.check_eligibility(&recall_event, current_time);
        if !eligibility.is_eligible {
            // Record rejection reason in metrics
            #[cfg(feature = "monitoring")]
            {
                if let Some(metrics) = crate::metrics::cognitive_patterns::cognitive_patterns() {
                    metrics.record_reconsolidation_rejection(&eligibility.rejection_reason);
                }
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
            window_position,
        );

        // Record successful reconsolidation
        #[cfg(feature = "monitoring")]
        {
            if let Some(metrics) = crate::metrics::cognitive_patterns::cognitive_patterns() {
                metrics.record_reconsolidation(window_position);
            }
        }

        Some(ReconsolidationResult {
            modified_episode,
            window_position,
            plasticity_factor: self.compute_plasticity(window_position),
            modification_confidence: Self::compute_modification_confidence(window_position),
            original_episode: recall_event.original_episode.clone(),
        })
    }

    /// Check all boundary conditions for reconsolidation eligibility
    ///
    /// Returns detailed eligibility with rejection reason if ineligible.
    fn check_eligibility(
        &self,
        recall: &RecallEvent,
        current_time: DateTime<Utc>,
    ) -> EligibilityResult {
        // Boundary 1: Must be active recall
        if !recall.is_active_recall {
            return EligibilityResult::ineligible(
                "Reconsolidation requires active recall, not passive re-exposure",
            );
        }

        // Boundary 2: Time since recall must be within window
        let time_since_recall = current_time - recall.recall_timestamp;

        if time_since_recall < self.window_start {
            return EligibilityResult::ineligible(&format!(
                "Too soon after recall: {} < {} (window starts at {})",
                format_duration(time_since_recall),
                format_duration(self.window_start),
                format_duration(self.window_start)
            ));
        }

        if time_since_recall > self.window_end {
            return EligibilityResult::ineligible(&format!(
                "Too late after recall: {} > {} (window closed)",
                format_duration(time_since_recall),
                format_duration(self.window_end)
            ));
        }

        // Boundary 3: Memory must be consolidated (>24 hours old)
        let memory_age = recall.recall_timestamp - recall.original_episode.when;

        if memory_age < self.min_memory_age {
            return EligibilityResult::ineligible(&format!(
                "Memory too recent: {} < {} (must be consolidated)",
                format_duration(memory_age),
                format_duration(self.min_memory_age)
            ));
        }

        // Boundary 4: Memory must not be too remote (< ~1 year)
        if memory_age > self.max_memory_age {
            return EligibilityResult::ineligible(&format!(
                "Memory too remote: {} > {} (reduced plasticity)",
                format_duration(memory_age),
                format_duration(self.max_memory_age)
            ));
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

        #[allow(clippy::cast_precision_loss)]
        let position = offset.num_milliseconds() as f64 / window_duration.num_milliseconds() as f64;

        position.clamp(0.0, 1.0) as f32
    }

    /// Compute plasticity factor based on window position
    ///
    /// Uses inverted-U function matching protein synthesis dynamics
    /// (Nader & Einarsson 2010 Fig 3)
    ///
    /// - Early (position=0.0, 1h): Rising plasticity
    /// - Peak (position=0.5, 3.5h): Maximum plasticity
    /// - Late (position=1.0, 6h): Declining plasticity
    #[must_use]
    pub fn compute_plasticity(&self, window_position: f32) -> f32 {
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
        window_position: f32,
    ) -> Episode {
        let plasticity = self.compute_plasticity(window_position);

        let mut modified = original.clone();

        // Apply field-level modifications
        if let Some(new_what) = &modifications.what {
            // Blend original and new content based on plasticity
            // For simplicity, we just replace if plasticity is high enough
            if plasticity > 0.2 {
                modified.what.clone_from(new_what);
            }
        }

        if plasticity > 0.2 {
            if let Some(new_where) = &modifications.where_location {
                modified.where_location = Some(new_where.clone());
            }

            if let Some(new_who) = &modifications.who {
                modified.who = Some(new_who.clone());
            }
        }

        // Update confidence based on modification type
        modified.reliability_confidence =
            Self::compute_modified_confidence(original, modifications, plasticity);

        // Vividness may be affected by modifications
        if modifications.modification_type == ModificationType::Update {
            // Updates can enhance vividness (retrieval-induced strengthening)
            let boost = plasticity * 0.1;
            modified.vividness_confidence =
                Confidence::exact(original.vividness_confidence.raw() * (1.0 + boost));
        } else if modifications.modification_type == ModificationType::Corruption {
            // Corruption reduces vividness
            let reduction = plasticity * 0.15;
            modified.vividness_confidence =
                Confidence::exact(original.vividness_confidence.raw() * (1.0 - reduction));
        }

        modified
    }

    /// Compute confidence of modified memory
    ///
    /// Confidence change depends on modification type:
    /// - Update: Strengthens memory (like rehearsal)
    /// - Corruption: Reduces confidence
    /// - Replacement: Resets confidence
    fn compute_modified_confidence(
        original: &Episode,
        modifications: &EpisodeModifications,
        plasticity: f32,
    ) -> Confidence {
        match modifications.modification_type {
            ModificationType::Update => {
                // Strengthens memory (retrieval-induced strengthening)
                // Dudai (2006): Reconsolidation can enhance memory stability
                let boost = plasticity * 0.1; // Up to 5% boost at peak plasticity (0.5 * 0.1)
                let new_value = (original.reliability_confidence.raw() * (1.0 + boost)).min(1.0);
                Confidence::exact(new_value)
            }
            ModificationType::Corruption => {
                // Reduces confidence proportional to modification extent
                let reduction = modifications.modification_extent * plasticity * 0.2;
                let new_value =
                    (original.reliability_confidence.raw() * (1.0 - reduction)).max(0.1);
                Confidence::exact(new_value)
            }
            ModificationType::Replacement => {
                // Confidence reset to quality of new information
                // Use modification_extent as proxy for new information quality
                Confidence::exact(modifications.modification_extent.clamp(0.3, 0.8))
            }
        }
    }

    fn compute_modification_confidence(window_position: f32) -> Confidence {
        // Higher confidence near peak plasticity (window_position = 0.5)
        // Use inverted-U curve same as plasticity
        let u_curve = 4.0 * window_position * (1.0 - window_position);
        let confidence_value = 0.5 + (u_curve * 0.4); // Range [0.5, 0.9]
        Confidence::exact(confidence_value)
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
    /// New content for "what" field (None = no change)
    pub what: Option<String>,

    /// New location (None = no change)
    pub where_location: Option<String>,

    /// New participants (None = no change)
    pub who: Option<Vec<String>>,

    /// Overall modification extent [0, 1]
    /// 0.0 = minimal changes, 1.0 = extensive changes
    pub modification_extent: f32,

    /// Type of modification
    pub modification_type: ModificationType,
}

impl EpisodeModifications {
    /// Create minimal update modification
    #[must_use]
    pub const fn minimal_update() -> Self {
        Self {
            what: None,
            where_location: None,
            who: None,
            modification_extent: 0.1,
            modification_type: ModificationType::Update,
        }
    }

    /// Create from HashMap of field changes (for test compatibility)
    #[must_use]
    pub fn from_fields(
        field_changes: &HashMap<String, String>,
        modification_extent: f32,
        modification_type: ModificationType,
    ) -> Self {
        let what = field_changes.get("what").cloned();
        let where_location = field_changes.get("where").cloned();
        let who = field_changes
            .get("who")
            .map(|s| s.split(',').map(|s| s.trim().to_string()).collect());

        Self {
            what,
            where_location,
            who,
            modification_extent,
            modification_type,
        }
    }
}

/// Type of memory modification
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
    const fn eligible() -> Self {
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

#[must_use]
fn format_duration(d: Duration) -> String {
    let hours = d.num_hours();
    let minutes = d.num_minutes() % 60;
    format!("{hours}h {minutes}m")
}
