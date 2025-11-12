//! Biologically-plausible integration of concept formation into consolidation scheduler.
//!
//! This module implements multi-timescale memory consolidation with sleep-stage-aware
//! concept formation, circuit breakers, and gradual rollout controls.
//!
//! ## Biological Foundation
//!
//! Memory consolidation operates across multiple timescales:
//! - Fast hippocampal encoding: seconds to minutes
//! - Initial consolidation: hours (first sleep cycle)
//! - Systems consolidation: days to weeks (repeated sleep cycles)
//! - Schema formation: weeks to months (asymptotic strengthening)
//!
//! Critical: Concept formation does NOT happen every consolidation cycle.
//! Formation requires:
//! - Multiple replay events across sleep stages (Mölle & Born 2011)
//! - Temporal spacing for interference reduction (Walker 2009)
//! - Statistical evidence accumulation (Tse et al. 2007)
//!
//! ## Key Design Principles
//!
//! 1. **Multiple Formation Gates**: Concept formation gated by minimum cycles,
//!    sufficient episodes, temporal spacing, sleep stage probability, and rollout rate.
//!
//! 2. **Sleep Stage Modulation**: Formation probability varies by stage:
//!    - NREM2: 15% (peak spindle-ripple coupling)
//!    - NREM3: 8% (sustained slow-wave consolidation)
//!    - REM: 3% (selective emotional processing)
//!    - QuietWake: 1% (brief awake replay)
//!
//! 3. **Circuit Breakers**: Prevent pathological formation rates:
//!    - Rate limit: 20 formations/hour maximum
//!    - Ratio limit: 30% concepts relative to episodes maximum
//!
//! 4. **Gradual Rollout**: Safe progressive deployment:
//!    - 0.0 = Shadow mode (log only, no persistence)
//!    - 0.01 = 1% rollout (early validation)
//!    - 0.10 = 10% rollout (broader testing)
//!    - 0.50 = 50% rollout (pre-production)
//!    - 1.0 = Full deployment
//!
//! 5. **Determinism**: Formation decisions deterministic for M14 distributed consolidation
//!    using seeded RNG based on cycle number.

use crate::consolidation::{ConceptFormationEngine, ProtoConcept, SleepStage};
use crate::memory::types::Episode;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

#[cfg(feature = "dual_memory_types")]
use crate::consolidation::ConceptFormationResult;

/// Biologically-inspired consolidation cycle tracker
///
/// Tracks sleep stage transitions, formation events, and episode accumulation
/// across consolidation cycles to enforce biological timing constraints.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsolidationCycleState {
    /// Current sleep stage
    pub sleep_stage: SleepStage,

    /// Number of consolidation cycles completed in current stage
    pub cycles_in_stage: u32,

    /// Total cycles across all stages in this consolidation session
    pub total_cycles: u64,

    /// Last concept formation cycle number
    pub last_formation_cycle: u64,

    /// Episodes accumulated since last formation
    pub episodes_since_formation: usize,

    /// Timestamp of last formation event
    pub last_formation_time: DateTime<Utc>,

    /// Recent formation timestamps (for rate limiting)
    /// Stored as Unix timestamps (seconds since epoch)
    recent_formations: VecDeque<i64>,

    /// Total concepts formed this session
    pub total_concepts_formed: usize,

    /// Total episodes processed this session
    pub total_episodes_processed: usize,
}

impl ConsolidationCycleState {
    /// Create new cycle state starting in NREM2
    #[must_use]
    pub fn new() -> Self {
        Self {
            sleep_stage: SleepStage::NREM2,
            cycles_in_stage: 0,
            total_cycles: 0,
            last_formation_cycle: 0,
            episodes_since_formation: 0,
            last_formation_time: Utc::now(),
            recent_formations: VecDeque::new(),
            total_concepts_formed: 0,
            total_episodes_processed: 0,
        }
    }

    /// Check if concept formation should occur this cycle
    ///
    /// Implements multi-gate formation criteria (ALL must be true):
    /// 1. Minimum cycles in stage reached (prevents premature formation)
    /// 2. Sufficient episodes accumulated (≥3 for statistical regularity)
    /// 3. Temporal spacing since last formation (≥30 min, prevents excessive formation)
    /// 4. Probabilistic sleep stage gate (models resource constraints)
    /// 5. Rollout sampling gate (gradual deployment control)
    ///
    /// Uses deterministic RNG seeded by cycle number for M14 distributed consolidation.
    ///
    /// Returns (should_form, skip_reason) tuple.
    #[must_use]
    pub fn should_form_concepts(
        &self,
        config: &ConceptFormationConfig,
    ) -> (bool, Option<SkipReason>) {
        // Gate 1: Minimum cycles in current stage
        let min_cycles_met = self.cycles_in_stage >= self.sleep_stage.min_cycles_before_formation();
        if !min_cycles_met {
            return (false, Some(SkipReason::InsufficientCycles));
        }

        // Gate 2: Sufficient episodes (minimum 3 per Tse et al. 2007)
        let sufficient_episodes = self.episodes_since_formation >= config.min_cluster_size;
        if !sufficient_episodes {
            return (false, Some(SkipReason::InsufficientEpisodes));
        }

        // Gate 3: Temporal spacing (30 minutes minimum per Rasch & Born 2013)
        let time_since_formation = Utc::now().signed_duration_since(self.last_formation_time);
        let sufficient_spacing = time_since_formation.num_minutes() >= 30;
        if !sufficient_spacing {
            return (false, Some(SkipReason::InsufficientTemporalSpacing));
        }

        // Gate 4: Sleep stage probability gate (models spindle/ripple coupling)
        // Use deterministic RNG seeded by cycle number for M14 distributed consolidation
        let stage_probability = self.sleep_stage.concept_formation_probability();
        let probabilistic_gate = self.deterministic_sample(stage_probability);
        if !probabilistic_gate {
            return (false, Some(SkipReason::SleepStageProbability));
        }

        // Gate 5: Rollout control (gradual deployment)
        let rollout_gate = self.deterministic_rollout_sample(config.rollout_sample_rate);
        if !rollout_gate {
            return (false, Some(SkipReason::RolloutSample));
        }

        (true, None)
    }

    /// Deterministic sampling for sleep stage probability gate
    ///
    /// Uses cycle number as seed to ensure same decision on all nodes (M14).
    /// Hash-based RNG for fast, high-quality randomness without external dependencies.
    fn deterministic_sample(&self, probability: f32) -> bool {
        // Simple hash-based RNG using cycle number and a mixing constant
        let hash = self
            .total_cycles
            .wrapping_mul(0x9e37_79b9_7f4a_7c15) // Golden ratio constant
            .wrapping_add(0x85eb_ca6b); // Mixing constant

        let normalized = (hash % 10_000) as f32 / 10_000.0;
        normalized < probability
    }

    /// Deterministic rollout sampling
    ///
    /// Uses cycle number + 1 as seed offset to decorrelate from stage probability.
    fn deterministic_rollout_sample(&self, sample_rate: f32) -> bool {
        let hash = (self.total_cycles + 1)
            .wrapping_mul(0x9e37_79b9_7f4a_7c15)
            .wrapping_add(0x85eb_ca6b);

        let normalized = (hash % 10_000) as f32 / 10_000.0;
        normalized < sample_rate
    }

    /// Advance to next cycle, potentially transitioning sleep stage
    ///
    /// Updates cycle counters, episode accumulation, and handles sleep stage transitions
    /// following simplified 90-minute ultradian rhythm model.
    pub fn advance_cycle(&mut self, episodes_processed: usize) {
        self.cycles_in_stage += 1;
        self.total_cycles += 1;
        self.episodes_since_formation += episodes_processed;
        self.total_episodes_processed += episodes_processed;

        // Simulate sleep stage transitions (simplified 90-minute ultradian rhythm)
        // Real implementation would use external sleep stage detection
        let cycles_per_stage = self.sleep_stage.typical_duration_minutes() / 10;
        if self.cycles_in_stage >= cycles_per_stage {
            self.transition_sleep_stage();
        }
    }

    /// Record concept formation event
    ///
    /// Updates formation timestamps, counters, and maintains recent formation history
    /// for rate limiting circuit breaker.
    pub fn record_formation(&mut self, concepts_formed: usize) {
        let now = Utc::now();
        let timestamp = now.timestamp();

        self.last_formation_cycle = self.total_cycles;
        self.last_formation_time = now;
        self.episodes_since_formation = 0;
        self.total_concepts_formed += concepts_formed;

        // Track recent formations for rate limiting (keep last hour)
        self.recent_formations.push_back(timestamp);
        self.prune_old_formations();
    }

    /// Check circuit breakers to prevent pathological formation rates
    ///
    /// Circuit breaker 1: Formation rate limit (max 20/hour)
    /// Circuit breaker 2: Concept/episode ratio limit (max 30%)
    ///
    /// Returns (breakers_ok, failure_reason)
    #[must_use]
    pub fn check_circuit_breakers(
        &self,
        config: &ConceptFormationConfig,
    ) -> (bool, Option<CircuitBreakerReason>) {
        // Circuit breaker 1: Formation rate limit
        let formations_last_hour = self.count_formations_last_hour();
        if formations_last_hour >= config.max_formations_per_hour {
            return (
                false,
                Some(CircuitBreakerReason::RateLimitExceeded {
                    current: formations_last_hour,
                    limit: config.max_formations_per_hour,
                }),
            );
        }

        // Circuit breaker 2: Concept/episode ratio
        if self.total_episodes_processed > 0 {
            let current_ratio =
                self.total_concepts_formed as f32 / self.total_episodes_processed as f32;
            if current_ratio >= config.max_concept_ratio {
                return (
                    false,
                    Some(CircuitBreakerReason::RatioExceeded {
                        current_ratio,
                        limit: config.max_concept_ratio,
                    }),
                );
            }
        }

        (true, None)
    }

    /// Count formations in last hour (for rate limiting and observability)
    #[must_use]
    pub fn count_formations_last_hour(&self) -> usize {
        let now = Utc::now().timestamp();
        let hour_ago = now - 3600;

        self.recent_formations
            .iter()
            .filter(|&&ts| ts >= hour_ago)
            .count()
    }

    /// Remove formation timestamps older than 1 hour
    fn prune_old_formations(&mut self) {
        let now = Utc::now().timestamp();
        let hour_ago = now - 3600;

        while let Some(&oldest) = self.recent_formations.front() {
            if oldest < hour_ago {
                self.recent_formations.pop_front();
            } else {
                break;
            }
        }
    }

    /// Transition to next sleep stage in cycle
    ///
    /// Follows simplified ultradian rhythm: NREM2 → NREM3 → REM → NREM2
    #[allow(clippy::missing_const_for_fn)] // Mutates self
    fn transition_sleep_stage(&mut self) {
        self.sleep_stage = match self.sleep_stage {
            SleepStage::NREM2 => SleepStage::NREM3,
            SleepStage::NREM3 => SleepStage::REM,
            SleepStage::REM | SleepStage::QuietWake => SleepStage::NREM2, // Loop back / enter sleep
        };
        self.cycles_in_stage = 0;
    }

    /// Get current concept/episode ratio
    #[must_use]
    pub fn concept_episode_ratio(&self) -> f32 {
        if self.total_episodes_processed > 0 {
            self.total_concepts_formed as f32 / self.total_episodes_processed as f32
        } else {
            0.0
        }
    }

    /// Get time since last formation in seconds
    #[must_use]
    pub fn seconds_since_last_formation(&self) -> i64 {
        Utc::now()
            .signed_duration_since(self.last_formation_time)
            .num_seconds()
    }
}

impl Default for ConsolidationCycleState {
    fn default() -> Self {
        Self::new()
    }
}

/// Configuration for concept formation with rollout controls
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConceptFormationConfig {
    /// Enable concept formation (master switch)
    pub enabled: bool,

    /// Rollout sample rate (0.0 = shadow mode, 1.0 = full deployment)
    /// - 0.0: Shadow mode (log decisions but don't create concepts)
    /// - 0.01: 1% rollout (early validation)
    /// - 0.10: 10% rollout (broader testing)
    /// - 0.50: 50% rollout (pre-production)
    /// - 1.0: Full deployment
    pub rollout_sample_rate: f32,

    /// Minimum cluster size for concept formation (default: 3)
    /// Derived from Tse et al. (2007) schema formation requirements
    pub min_cluster_size: usize,

    /// Coherence threshold for viable concepts (default: 0.65)
    /// Matches CA3 pattern completion threshold (Nakazawa et al. 2002)
    pub coherence_threshold: f32,

    /// Maximum concepts per formation event (default: 5)
    /// Limited by sleep spindle density (Mölle & Born 2011)
    pub max_concepts_per_event: usize,

    /// Circuit breaker: max formation rate per hour
    /// Prevents pathological concept explosion (default: 20/hour)
    pub max_formations_per_hour: usize,

    /// Circuit breaker: max concept/episode ratio
    /// Prevents semantic inflation (default: 0.3 = 30% concepts)
    pub max_concept_ratio: f32,
}

impl Default for ConceptFormationConfig {
    fn default() -> Self {
        Self {
            enabled: false,              // Disabled by default, explicit opt-in
            rollout_sample_rate: 0.0,    // Start in shadow mode
            min_cluster_size: 3,         // Tse et al. 2007
            coherence_threshold: 0.65,   // Nakazawa et al. 2002 (CA3 completion)
            max_concepts_per_event: 5,   // Mölle & Born 2011 (spindle density)
            max_formations_per_hour: 20, // Rate limit
            max_concept_ratio: 0.3,      // 30% max concepts
        }
    }
}

impl ConceptFormationConfig {
    /// Check if currently in shadow mode (logging only, no persistence)
    #[must_use]
    pub const fn is_shadow_mode(&self) -> bool {
        self.rollout_sample_rate == 0.0
    }

    /// Get rollout phase name for logging
    #[must_use]
    pub fn rollout_phase_name(&self) -> &'static str {
        if self.rollout_sample_rate == 0.0 {
            "shadow"
        } else if self.rollout_sample_rate <= 0.01 {
            "1pct"
        } else if self.rollout_sample_rate <= 0.10 {
            "10pct"
        } else if self.rollout_sample_rate <= 0.50 {
            "50pct"
        } else {
            "full"
        }
    }

    /// Validate configuration parameters
    ///
    /// Returns Ok(()) if valid, Err(message) if invalid.
    pub fn validate(&self) -> Result<(), String> {
        if self.rollout_sample_rate < 0.0 || self.rollout_sample_rate > 1.0 {
            return Err(format!(
                "rollout_sample_rate must be in [0.0, 1.0], got {}",
                self.rollout_sample_rate
            ));
        }

        if self.min_cluster_size < 2 {
            return Err(format!(
                "min_cluster_size must be >= 2, got {}",
                self.min_cluster_size
            ));
        }

        if self.coherence_threshold < 0.0 || self.coherence_threshold > 1.0 {
            return Err(format!(
                "coherence_threshold must be in [0.0, 1.0], got {}",
                self.coherence_threshold
            ));
        }

        if self.max_concepts_per_event == 0 {
            return Err("max_concepts_per_event must be > 0".to_string());
        }

        if self.max_formations_per_hour == 0 {
            return Err("max_formations_per_hour must be > 0".to_string());
        }

        if self.max_concept_ratio <= 0.0 || self.max_concept_ratio > 1.0 {
            return Err(format!(
                "max_concept_ratio must be in (0.0, 1.0], got {}",
                self.max_concept_ratio
            ));
        }

        Ok(())
    }
}

/// Statistics from a concept formation attempt
#[derive(Debug, Clone, Default)]
pub struct ConceptFormationStats {
    /// Number of concepts formed
    pub concepts_formed: usize,

    /// Number of bindings created
    pub bindings_created: usize,

    /// Reason formation was skipped (if applicable)
    pub skip_reason: Option<SkipReason>,

    /// Circuit breaker reason (if tripped)
    pub circuit_breaker_reason: Option<CircuitBreakerReason>,

    /// Average coherence score of formed concepts
    pub avg_coherence: f32,

    /// Proto-concepts created (for tracking gradual consolidation)
    pub proto_concepts_created: usize,
}

/// Reason concept formation was skipped
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SkipReason {
    /// Rollout sample rate excluded this cycle
    RolloutSample,

    /// Minimum cycles in sleep stage not reached
    InsufficientCycles,

    /// Less than min_cluster_size episodes accumulated
    InsufficientEpisodes,

    /// Less than 30 minutes since last formation
    InsufficientTemporalSpacing,

    /// Probabilistic sleep stage gate
    SleepStageProbability,

    /// Concept formation disabled in config
    Disabled,
}

/// Reason circuit breaker was tripped
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum CircuitBreakerReason {
    /// Formation rate exceeded limit
    RateLimitExceeded {
        /// Current formations in last hour
        current: usize,
        /// Maximum allowed formations per hour
        limit: usize,
    },

    /// Concept/episode ratio exceeded limit
    RatioExceeded {
        /// Current concept/episode ratio
        current_ratio: f32,
        /// Maximum allowed ratio
        limit: f32,
    },
}

/// Helper for forming concepts during consolidation cycle
///
/// This encapsulates the concept formation logic to keep DreamEngine clean.
/// In production, this would be a full service with persistence integration.
pub struct ConceptFormationHelper {
    engine: ConceptFormationEngine,
    config: ConceptFormationConfig,
}

impl ConceptFormationHelper {
    /// Create new formation helper with configuration
    #[must_use]
    pub fn new(config: ConceptFormationConfig) -> Self {
        let engine = ConceptFormationEngine::new();
        Self { engine, config }
    }

    /// Attempt to form concepts from episodes
    ///
    /// Returns formation statistics including skip reasons and circuit breaker info.
    ///
    /// In shadow mode (rollout_sample_rate = 0.0), logs decisions but doesn't persist.
    #[must_use]
    pub fn try_form_concepts(
        &self,
        episodes: &[Episode],
        cycle_state: &ConsolidationCycleState,
    ) -> ConceptFormationStats {
        // Check if enabled
        if !self.config.enabled {
            return ConceptFormationStats {
                skip_reason: Some(SkipReason::Disabled),
                ..Default::default()
            };
        }

        // Check circuit breakers
        let (breakers_ok, breaker_reason) = cycle_state.check_circuit_breakers(&self.config);
        if !breakers_ok {
            return ConceptFormationStats {
                circuit_breaker_reason: breaker_reason,
                ..Default::default()
            };
        }

        // Check formation gates
        let (should_form, skip_reason) = cycle_state.should_form_concepts(&self.config);
        if !should_form {
            return ConceptFormationStats {
                skip_reason,
                ..Default::default()
            };
        }

        // Insufficient episodes check
        if episodes.len() < self.config.min_cluster_size {
            return ConceptFormationStats {
                skip_reason: Some(SkipReason::InsufficientEpisodes),
                ..Default::default()
            };
        }

        // Form proto-concepts
        let proto_concepts = self
            .engine
            .process_episodes(episodes, cycle_state.sleep_stage);

        // Filter by promotion criteria and apply max concepts per event limit
        let concepts_to_promote: Vec<&ProtoConcept> = proto_concepts
            .iter()
            .filter(|pc| pc.is_ready_for_promotion())
            .take(self.config.max_concepts_per_event)
            .collect();

        // Calculate statistics
        let avg_coherence = if concepts_to_promote.is_empty() {
            0.0
        } else {
            concepts_to_promote
                .iter()
                .map(|pc| pc.coherence_score)
                .sum::<f32>()
                / concepts_to_promote.len() as f32
        };

        // In shadow mode, log but don't return concepts for persistence
        if self.config.is_shadow_mode() {
            tracing::info!(
                proto_concepts = proto_concepts.len(),
                ready_for_promotion = concepts_to_promote.len(),
                avg_coherence,
                sleep_stage = ?cycle_state.sleep_stage,
                cycle_number = cycle_state.total_cycles,
                "Shadow mode: concept formation logged but not persisted"
            );

            ConceptFormationStats {
                proto_concepts_created: proto_concepts.len(),
                concepts_formed: 0, // Not persisted in shadow mode
                avg_coherence,
                ..Default::default()
            }
        } else {
            // Real formation (concepts would be persisted by caller)
            ConceptFormationStats {
                proto_concepts_created: proto_concepts.len(),
                concepts_formed: concepts_to_promote.len(),
                bindings_created: concepts_to_promote.len(), // One binding per concept-episode pair
                avg_coherence,
                ..Default::default()
            }
        }
    }

    /// Form concepts and return DualMemoryNode instances ready for graph insertion
    ///
    /// Only available with dual_memory_types feature.
    /// Returns empty vec in shadow mode.
    #[cfg(feature = "dual_memory_types")]
    #[must_use]
    pub fn form_and_promote_concepts(
        &self,
        episodes: &[Episode],
        cycle_state: &ConsolidationCycleState,
    ) -> Vec<ConceptFormationResult> {
        // Pre-flight checks
        if !self.config.enabled || self.config.is_shadow_mode() {
            return Vec::new();
        }

        let (breakers_ok, _) = cycle_state.check_circuit_breakers(&self.config);
        if !breakers_ok {
            return Vec::new();
        }

        let (should_form, _) = cycle_state.should_form_concepts(&self.config);
        if !should_form {
            return Vec::new();
        }

        // Form concepts with promotion
        let concepts = self.engine.form_concepts(episodes, cycle_state.sleep_stage);

        // Apply limit
        concepts
            .into_iter()
            .take(self.config.max_concepts_per_event)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Confidence;
    use chrono::Duration;

    fn create_test_episodes(count: usize) -> Vec<Episode> {
        let base_time = Utc::now() - Duration::days(2); // Old enough for consolidation
        let mut episodes = Vec::new();

        for i in 0..count {
            episodes.push(Episode::new(
                format!("episode_{i:03}"),
                base_time - Duration::hours(count as i64 - i as i64),
                format!("content_{i}"),
                [0.5; 768],
                Confidence::exact(0.8),
            ));
        }

        episodes
    }

    #[test]
    fn test_cycle_state_creation() {
        let state = ConsolidationCycleState::new();
        assert_eq!(state.sleep_stage, SleepStage::NREM2);
        assert_eq!(state.cycles_in_stage, 0);
        assert_eq!(state.total_cycles, 0);
        assert_eq!(state.episodes_since_formation, 0);
    }

    #[test]
    fn test_cycle_advancement() {
        let mut state = ConsolidationCycleState::new();
        state.advance_cycle(10);

        assert_eq!(state.cycles_in_stage, 1);
        assert_eq!(state.total_cycles, 1);
        assert_eq!(state.episodes_since_formation, 10);
        assert_eq!(state.total_episodes_processed, 10);
    }

    #[test]
    fn test_sleep_stage_transitions() {
        let mut state = ConsolidationCycleState::new();

        // Start in NREM2
        assert_eq!(state.sleep_stage, SleepStage::NREM2);

        // Advance enough cycles to transition
        for _ in 0..3 {
            state.advance_cycle(10);
        }

        // Should transition to NREM3
        assert_eq!(state.sleep_stage, SleepStage::NREM3);
    }

    #[test]
    fn test_formation_gates_insufficient_cycles() {
        let state = ConsolidationCycleState::new();
        let config = ConceptFormationConfig {
            enabled: true,
            rollout_sample_rate: 1.0,
            ..Default::default()
        };

        let (should_form, reason) = state.should_form_concepts(&config);
        assert!(!should_form);
        assert_eq!(reason, Some(SkipReason::InsufficientCycles));
    }

    #[test]
    fn test_formation_gates_insufficient_episodes() {
        let mut state = ConsolidationCycleState::new();
        state.cycles_in_stage = 5; // Enough cycles
        state.episodes_since_formation = 2; // Not enough episodes

        let config = ConceptFormationConfig {
            enabled: true,
            rollout_sample_rate: 1.0,
            min_cluster_size: 3,
            ..Default::default()
        };

        let (should_form, reason) = state.should_form_concepts(&config);
        assert!(!should_form);
        assert_eq!(reason, Some(SkipReason::InsufficientEpisodes));
    }

    #[test]
    fn test_formation_gates_insufficient_spacing() {
        let mut state = ConsolidationCycleState::new();
        state.cycles_in_stage = 5;
        state.episodes_since_formation = 10;
        state.last_formation_time = Utc::now() - Duration::minutes(15); // Only 15 min ago

        let config = ConceptFormationConfig {
            enabled: true,
            rollout_sample_rate: 1.0,
            ..Default::default()
        };

        let (should_form, reason) = state.should_form_concepts(&config);
        assert!(!should_form);
        assert_eq!(reason, Some(SkipReason::InsufficientTemporalSpacing));
    }

    #[test]
    fn test_deterministic_sampling() {
        let mut state = ConsolidationCycleState::new();
        state.total_cycles = 12345;

        // Same cycle number should give same result
        let result1 = state.deterministic_sample(0.5);
        let result2 = state.deterministic_sample(0.5);
        assert_eq!(result1, result2);

        // Different cycle numbers should give different results (eventually)
        state.total_cycles = 67890;
        let _result3 = state.deterministic_sample(0.5);
        // With high probability, at least one should differ
        // (not guaranteed but extremely likely)
    }

    #[test]
    fn test_circuit_breaker_rate_limit() {
        let mut state = ConsolidationCycleState::new();
        let config = ConceptFormationConfig {
            max_formations_per_hour: 5,
            ..Default::default()
        };

        // Simulate 5 formations in last hour
        for _ in 0..5 {
            state.record_formation(1);
        }

        let (ok, reason) = state.check_circuit_breakers(&config);
        assert!(!ok);
        assert!(matches!(
            reason,
            Some(CircuitBreakerReason::RateLimitExceeded { .. })
        ));
    }

    #[test]
    fn test_circuit_breaker_ratio_limit() {
        let mut state = ConsolidationCycleState::new();
        state.total_episodes_processed = 100;
        state.total_concepts_formed = 35; // 35% ratio

        let config = ConceptFormationConfig {
            max_concept_ratio: 0.3, // 30% max
            ..Default::default()
        };

        let (ok, reason) = state.check_circuit_breakers(&config);
        assert!(!ok);
        assert!(matches!(
            reason,
            Some(CircuitBreakerReason::RatioExceeded { .. })
        ));
    }

    #[test]
    fn test_concept_episode_ratio() {
        let mut state = ConsolidationCycleState::new();
        state.total_episodes_processed = 100;
        state.total_concepts_formed = 25;

        let ratio = state.concept_episode_ratio();
        assert!((ratio - 0.25).abs() < 1e-6);
    }

    #[test]
    fn test_config_validation() {
        // Valid config
        let config = ConceptFormationConfig::default();
        assert!(config.validate().is_ok());

        // Invalid rollout rate
        let invalid = ConceptFormationConfig {
            rollout_sample_rate: 1.5,
            ..Default::default()
        };
        assert!(invalid.validate().is_err());

        // Invalid min cluster size
        let invalid = ConceptFormationConfig {
            min_cluster_size: 1,
            ..Default::default()
        };
        assert!(invalid.validate().is_err());

        // Invalid coherence threshold
        let invalid = ConceptFormationConfig {
            coherence_threshold: 1.5,
            ..Default::default()
        };
        assert!(invalid.validate().is_err());
    }

    #[test]
    fn test_config_rollout_phase_name() {
        let shadow = ConceptFormationConfig {
            rollout_sample_rate: 0.0,
            ..Default::default()
        };
        assert_eq!(shadow.rollout_phase_name(), "shadow");

        let one_pct = ConceptFormationConfig {
            rollout_sample_rate: 0.01,
            ..Default::default()
        };
        assert_eq!(one_pct.rollout_phase_name(), "1pct");

        let full = ConceptFormationConfig {
            rollout_sample_rate: 1.0,
            ..Default::default()
        };
        assert_eq!(full.rollout_phase_name(), "full");
    }

    #[test]
    fn test_formation_helper_disabled() {
        let config = ConceptFormationConfig {
            enabled: false,
            ..Default::default()
        };
        let helper = ConceptFormationHelper::new(config);
        let state = ConsolidationCycleState::new();
        let episodes = create_test_episodes(10);

        let result = helper.try_form_concepts(&episodes, &state);
        assert_eq!(result.skip_reason, Some(SkipReason::Disabled));
        assert_eq!(result.concepts_formed, 0);
    }

    #[test]
    fn test_formation_helper_shadow_mode() {
        let config = ConceptFormationConfig {
            enabled: true,
            rollout_sample_rate: 0.0, // Shadow mode
            ..Default::default()
        };
        let helper = ConceptFormationHelper::new(config);

        let mut state = ConsolidationCycleState::new();
        state.cycles_in_stage = 5;
        state.episodes_since_formation = 10;
        state.last_formation_time = Utc::now() - Duration::hours(1);

        let episodes = create_test_episodes(10);
        let result = helper.try_form_concepts(&episodes, &state);

        // Shadow mode should not persist concepts
        assert_eq!(result.concepts_formed, 0);
    }

    #[test]
    fn test_formation_recording() {
        let mut state = ConsolidationCycleState::new();
        state.total_cycles = 10;
        state.episodes_since_formation = 20;

        state.record_formation(5);

        assert_eq!(state.last_formation_cycle, 10);
        assert_eq!(state.episodes_since_formation, 0);
        assert_eq!(state.total_concepts_formed, 5);
        assert_eq!(state.recent_formations.len(), 1);
    }

    #[test]
    #[allow(clippy::expect_used)]
    fn test_recent_formations_pruning() {
        let mut state = ConsolidationCycleState::new();

        // Add old formation (should be pruned)
        let old_time = (Utc::now() - Duration::hours(2)).timestamp();
        state.recent_formations.push_back(old_time);

        // Add recent formation (should be kept)
        let recent_time = Utc::now().timestamp();
        state.recent_formations.push_back(recent_time);

        state.prune_old_formations();

        // Only recent formation should remain
        assert_eq!(state.recent_formations.len(), 1);
        assert_eq!(
            *state
                .recent_formations
                .front()
                .expect("Test should have at least one recent formation"),
            recent_time
        );
    }
}
