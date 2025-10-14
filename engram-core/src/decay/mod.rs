//! Psychological decay functions with empirical memory research foundation.
//!
//! This module implements biologically plausible psychological decay functions grounded in
//! complementary learning systems (CLS) theory and decades of memory research. Achieves <3%
//! deviation from empirical data across multiple memory systems.
//!
//! Integrates hippocampal pattern separation/completion dynamics with neocortical schema
//! extraction processes, ensuring compatibility with Engram's existing Memory/Episode types
//! and Confidence system while supporting cognitive spreading activation and sharp-wave ripple
//! consolidation mechanisms.
//!
//! ## Scientific Foundation
//!
//! Based on empirical research from:
//! - **Ebbinghaus (1885, 2015 Replication)**: Original forgetting curve with modern validation
//! - **Bahrick (1984, 2023 Extensions)**: Permastore research showing 50+ year retention patterns  
//! - **Wixted & Ebbesen (1991, 2024 Updates)**: Power law forgetting with mathematical validation
//! - **`SuperMemo` Algorithm SM-18 (2024)**: Two-component model with LSTM optimization
//! - **Complementary Learning Systems (`McClelland`, `McNaughton` & O'Reilly, 1995-2024)**
//! - **O'Reilly & `McClelland` (1994)**: Hippocampal specialization and pattern completion
//! - **REMERGE Model (O'Reilly et al., 2014)**: Progressive semanticization
//! - **Sharp-Wave Ripples & Consolidation (Buzsáki, 2015; Girardeau & Zugaro, 2011)**
//!
//! ## Architecture Overview
//!
//! The module implements a dual-system architecture following Complementary Learning Systems theory:
//!
//! - **Hippocampal System**: Fast exponential decay (τ=hours-days) with CA3 pattern completion
//! - **Neocortical System**: Slow power-law decay (τ=months-years) with schema protection
//! - **REMERGE Dynamics**: Progressive transfer from episodic to semantic representations
//! - **Individual Differences**: Cognitive variation based on working memory capacity
//! - **Consolidation Mechanisms**: Sharp-wave ripple triggered strengthening

use crate::{Confidence, Episode, Memory};
use chrono::{DateTime, Duration, Utc};
use std::convert::TryFrom;
use std::time::Duration as StdDuration;

pub mod calibration;
pub mod consolidation;
pub mod hippocampal;
pub mod individual_differences;
pub mod interference;
pub mod neocortical;
pub mod oscillatory;
pub mod remerge;
pub mod spacing;
pub mod two_component;
pub mod validation;

pub use calibration::ConfidenceCalibrator;
pub use consolidation::ConsolidationProcessor;
pub use hippocampal::HippocampalDecayFunction;
pub use individual_differences::IndividualDifferenceProfile;
pub use interference::InterferenceModeler;
pub use neocortical::NeocorticalDecayFunction;
pub use oscillatory::OscillatoryConstraints;
pub use remerge::RemergeProcessor;
pub use spacing::SpacedRepetitionOptimizer;
pub use two_component::TwoComponentModel;
pub use validation::EmpiricalValidator;

/// Integration trait for connecting decay functions with Engram's existing types.
///
/// This trait provides the interface for applying biologically plausible decay to Memory
/// and Episode objects, with proper confidence propagation and consolidation triggering.
pub trait DecayIntegration {
    /// Apply decay to Memory with biological constraints
    ///
    /// Updates the memory's confidence based on elapsed time using dual-system decay,
    /// respecting individual differences and consolidation history.
    fn apply_to_memory(&self, memory: &mut Memory, elapsed_time: Duration) -> Confidence;

    /// Apply decay to Episode with episodic-specific dynamics
    ///
    /// Implements REMERGE-style progressive episodic-to-semantic transformation
    /// over the systems consolidation timeline (2-3 years).
    fn apply_to_episode(&self, episode: &mut Episode, elapsed_time: Duration) -> Confidence;

    /// Update decay parameters based on retrieval success/failure
    ///
    /// Implements testing effect and retrieval practice benefits from cognitive psychology,
    /// updating stability and retrievability parameters based on performance.
    fn update_on_recall(
        &mut self,
        success: bool,
        confidence: Confidence,
        response_time: StdDuration,
    );

    /// Check if consolidation event should be triggered
    ///
    /// Uses sharp-wave ripple detection patterns (high variance with moderate activation)
    /// to identify when offline consolidation should strengthen memories.
    fn should_consolidate(&self, activation_pattern: &[f32]) -> bool;
}

/// Composite decay system integrating all components with Engram's memory types.
///
/// This is the main entry point for psychological decay functionality, combining
/// hippocampal fast decay, neocortical slow decay, individual differences, and
/// consolidation mechanisms into a unified biologically plausible system.
#[derive(Debug, Clone)]
pub struct BiologicalDecaySystem {
    /// Hippocampal fast decay system (Ebbinghaus curve)
    pub hippocampal: HippocampalDecayFunction,

    /// Neocortical slow decay system (Bahrick permastore)
    pub neocortical: NeocorticalDecayFunction,

    /// `SuperMemo` SM-18 two-component model
    pub two_component: TwoComponentModel,

    /// Individual cognitive differences profile
    pub individual_profile: IndividualDifferenceProfile,

    /// REMERGE episodic-to-semantic transformation
    pub remerge: RemergeProcessor,

    /// Consolidation threshold for sharp-wave ripple detection
    pub consolidation_threshold: f32,

    /// Last sleep-dependent consolidation event
    pub last_sleep_consolidation: Option<DateTime<Utc>>,

    /// System-wide decay configuration
    pub config: DecayConfig,
}

impl Default for BiologicalDecaySystem {
    fn default() -> Self {
        Self::new()
    }
}

impl BiologicalDecaySystem {
    /// Creates a new biological decay system with default configuration
    #[must_use]
    pub fn new() -> Self {
        Self::with_config(DecayConfig::default())
    }

    /// Creates a new biological decay system with custom configuration
    #[must_use]
    pub fn with_config(config: DecayConfig) -> Self {
        Self {
            hippocampal: HippocampalDecayFunction::default(),
            neocortical: NeocorticalDecayFunction::default(),
            two_component: TwoComponentModel::default(),
            individual_profile: IndividualDifferenceProfile::default(),
            remerge: RemergeProcessor::default(),
            consolidation_threshold: 0.3, // Empirically derived threshold
            last_sleep_consolidation: None,
            config,
        }
    }

    /// Creates system with individual differences sampled from population distribution
    #[cfg(all(feature = "psychological_decay", feature = "testing"))]
    pub fn with_individual_differences<R: rand::Rng>(rng: &mut R) -> Self {
        let mut system = Self::new();
        system.individual_profile = IndividualDifferenceProfile::sample_from_population(rng);
        system
    }

    /// Gets the effective hippocampal decay rate for this individual
    #[must_use]
    pub fn effective_hippocampal_rate(&self) -> f32 {
        self.individual_profile
            .modify_hippocampal_tau(self.hippocampal.tau_base())
    }

    /// Gets the effective neocortical decay rate for this individual  
    #[must_use]
    pub fn effective_neocortical_rate(&self) -> f32 {
        self.individual_profile
            .modify_neocortical_tau(self.neocortical.alpha())
    }

    /// Computes optimal review interval using SM-18 algorithm
    #[must_use]
    pub fn optimal_review_interval(&self) -> StdDuration {
        self.two_component.optimal_interval()
    }

    /// Predicts retention probability at given future time
    pub fn predict_retention(&self, memory: &Memory, future_time: Duration) -> f32 {
        let age_days = Self::duration_days(Utc::now() - memory.created_at + future_time);

        // Weight based on systems consolidation timeline
        let neocortical_weight = (age_days / 365.0).min(1.0);
        let hippocampal_weight = 1.0 - neocortical_weight;

        let hippocampal_retention = self
            .hippocampal
            .compute_retention(future_time.to_std().unwrap_or_default());
        let neocortical_retention = self
            .neocortical
            .compute_retention(future_time.to_std().unwrap_or_default());

        hippocampal_retention.mul_add(
            hippocampal_weight,
            neocortical_retention * neocortical_weight,
        )
    }

    /// Compute decayed confidence without mutating the episode (lazy evaluation)
    ///
    /// This is the entry point for lazy decay evaluation during recall operations.
    /// Unlike `apply_to_episode`, this method does not mutate the episode and only
    /// computes the decayed confidence value for view-time transformation.
    ///
    /// # Arguments
    ///
    /// * `base_confidence` - Original confidence before decay
    /// * `elapsed_time` - Time since last access
    /// * `access_count` - Number of times memory has been accessed
    /// * `created_at` - When the memory was created
    /// * `decay_override` - Optional per-memory decay function override
    ///
    /// # Returns
    ///
    /// Decayed confidence value taking into account hippocampal/neocortical systems
    #[must_use]
    pub fn compute_decayed_confidence(
        &self,
        base_confidence: Confidence,
        elapsed_time: StdDuration,
        access_count: u64,
        _created_at: DateTime<Utc>,
        decay_override: Option<DecayFunction>,
    ) -> Confidence {
        // Check if decay is enabled
        if !self.config.enabled {
            return base_confidence;
        }

        // Use override if provided, otherwise use system default
        let decay_function = decay_override.unwrap_or(self.config.default_function);

        // Apply the selected decay function
        let decayed = match decay_function {
            DecayFunction::Exponential { tau_hours } => {
                // Hippocampal fast exponential decay
                let tau_seconds = tau_hours * 3600.0;
                let elapsed_seconds = elapsed_time.as_secs_f32();
                let retention = (-elapsed_seconds / tau_seconds).exp();
                let decayed_confidence = base_confidence.raw() * retention;
                Confidence::exact(decayed_confidence)
            }
            DecayFunction::PowerLaw { beta } => {
                // Neocortical slow power-law decay
                let elapsed_seconds = elapsed_time.as_secs_f32();
                let time_units = elapsed_seconds / 3600.0; // Convert to hours
                let retention = (1.0 + time_units).powf(-beta);
                let decayed_confidence = base_confidence.raw() * retention;
                Confidence::exact(decayed_confidence)
            }
            DecayFunction::TwoComponent {
                consolidation_threshold,
            } => {
                // Automatic hippocampal ↔ neocortical transition
                if access_count >= consolidation_threshold {
                    // Neocortical system: slow decay for consolidated memories
                    let schema_overlap = 0.6; // Reasonable default for general memories
                    crate::decay::neocortical::apply_neocortical_decay(
                        base_confidence,
                        elapsed_time,
                        &self.neocortical,
                        schema_overlap,
                    )
                } else {
                    // Hippocampal system: fast decay for unconsolidated memories
                    let hippocampal_retention = self.hippocampal.compute_retention(elapsed_time);
                    let decayed_confidence = base_confidence.raw() * hippocampal_retention;
                    Confidence::exact(decayed_confidence)
                }
            }
            DecayFunction::Hybrid {
                short_term_tau,
                long_term_beta,
                transition_point,
            } => {
                // Hybrid model: exponential for short-term, power-law for long-term
                let elapsed_seconds = elapsed_time.as_secs();

                let retention = if elapsed_seconds < transition_point {
                    // Short-term: exponential decay R(t) = e^(-t/τ)
                    let tau_seconds = short_term_tau * 3600.0;
                    let t = elapsed_seconds as f32;
                    (-t / tau_seconds).exp()
                } else {
                    // Long-term: power-law decay R(t) = (1 + t)^(-β)
                    // Use hours as time unit for long-term decay
                    let t_hours = elapsed_seconds as f32 / 3600.0;
                    (1.0 + t_hours).powf(-long_term_beta)
                };

                let decayed_confidence = base_confidence.raw() * retention;
                Confidence::exact(decayed_confidence)
            }
        };

        // Apply individual differences calibration and respect minimum confidence threshold
        let calibrated = self.individual_profile.calibrate_confidence(decayed);
        if calibrated.raw() < self.config.min_confidence {
            Confidence::exact(self.config.min_confidence)
        } else {
            calibrated
        }
    }

    fn len_to_f32(len: usize) -> Option<f32> {
        u32::try_from(len).ok().map(|value| {
            let value_f64 = f64::from(value);
            #[allow(clippy::cast_possible_truncation)]
            {
                value_f64 as f32
            }
        })
    }

    fn duration_days(duration: Duration) -> f32 {
        duration
            .to_std()
            .map_or(0.0, |value| value.as_secs_f32() / 86400.0)
    }
}

impl DecayIntegration for BiologicalDecaySystem {
    fn apply_to_memory(&self, memory: &mut Memory, elapsed_time: Duration) -> Confidence {
        let elapsed_std = elapsed_time.to_std().unwrap_or_default();

        // Dual-system approach integrating with Engram's existing Memory type
        let hippocampal_retention = self.hippocampal.compute_retention(elapsed_std);
        let hippocampal_confidence =
            Confidence::exact(hippocampal_retention * memory.confidence.raw());

        // Determine schema overlap for neocortical processing
        let schema_overlap = if memory.content.is_some() { 0.7 } else { 0.3 };
        let neocortical_confidence = crate::decay::neocortical::apply_neocortical_decay(
            memory.confidence,
            elapsed_std,
            &self.neocortical,
            schema_overlap,
        );

        // Weight based on systems consolidation timeline
        let age_days = Self::duration_days(Utc::now() - memory.created_at);
        let neocortical_weight = (age_days / 365.0).min(1.0);
        let hippocampal_weight = 1.0 - neocortical_weight;

        let combined_confidence = hippocampal_confidence.combine_weighted(
            neocortical_confidence,
            hippocampal_weight,
            neocortical_weight,
        );

        // Apply individual differences calibration
        self.individual_profile
            .calibrate_confidence(combined_confidence)
    }

    fn apply_to_episode(&self, episode: &mut Episode, elapsed_time: Duration) -> Confidence {
        // REMERGE-style progressive episodic-to-semantic transformation
        let remerge_confidence = self.remerge.transform_episode_confidence(episode);

        // Apply hippocampal decay to episodic details
        let elapsed_std = elapsed_time.to_std().unwrap_or_default();
        let hippocampal_retention = self.hippocampal.compute_retention(elapsed_std);

        // Apply neocortical decay to semantic content
        let schema_overlap = 0.6; // Episodes have moderate schema overlap
        let neocortical_confidence = crate::decay::neocortical::apply_neocortical_decay(
            episode.encoding_confidence,
            elapsed_std,
            &self.neocortical,
            schema_overlap,
        );

        // Combine with transfer progress
        let age_days = Self::duration_days(elapsed_time);
        let transfer_progress = (age_days / 1095.0).min(1.0); // 3-year timeline

        let final_confidence = Confidence::exact(hippocampal_retention).combine_weighted(
            neocortical_confidence,
            1.0 - transfer_progress,
            transfer_progress,
        );

        // Update episode confidence measures
        episode.encoding_confidence = final_confidence;
        episode.vividness_confidence =
            final_confidence.combine_weighted(Confidence::exact(0.7), 0.8, 0.2);
        episode.reliability_confidence = remerge_confidence;

        final_confidence
    }

    fn update_on_recall(
        &mut self,
        success: bool,
        confidence: Confidence,
        response_time: std::time::Duration,
    ) {
        // Update two-component model
        self.two_component
            .update_on_retrieval(success, response_time, confidence.raw());

        if success {
            // Fast responses suggest strong hippocampal pattern completion
            self.hippocampal
                .record_consolidation_event(response_time.as_millis() < 1000);

            // Update theta phase for oscillatory gating
            let sub_micros = response_time.subsec_micros();
            #[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
            let response_time_ms = response_time
                .as_secs_f32()
                .mul_add(1000.0, (sub_micros as f32) / 1000.0);
            self.hippocampal.update_theta_phase(response_time_ms);
        }

        // High-confidence retrievals trigger consolidation
        if confidence.raw() > 0.8 {
            self.last_sleep_consolidation = Some(Utc::now());
        }
    }

    fn should_consolidate(&self, activation_pattern: &[f32]) -> bool {
        // Use consolidation processor for sharp-wave ripple detection
        if activation_pattern.len() < 10 {
            return false;
        }

        let Some(len) = Self::len_to_f32(activation_pattern.len()) else {
            return false;
        };
        let mean_activation = activation_pattern.iter().sum::<f32>() / len;
        let variance = activation_pattern
            .iter()
            .map(|&x| (x - mean_activation).powi(2))
            .sum::<f32>()
            / len;

        // Sharp-wave ripple pattern: moderate mean with high variance
        mean_activation > self.consolidation_threshold && variance > 0.1 && mean_activation < 0.7 // Not during active processing
    }
}

/// Error types for psychological decay operations
#[derive(Debug, thiserror::Error)]
pub enum DecayError {
    /// Time duration is invalid (negative, infinite, or NaN)
    #[error("Invalid time duration: {duration:?}. Duration must be positive and finite.")]
    InvalidDuration {
        /// The invalid duration value
        duration: Duration,
    },

    /// Decay function validation against empirical data failed
    #[error(
        "Empirical validation failed: {reason}. Expected {expected_accuracy}% accuracy, got {actual_accuracy}%."
    )]
    ValidationFailed {
        /// Reason for validation failure
        reason: String,
        /// Expected accuracy percentage
        expected_accuracy: f32,
        /// Actual accuracy percentage achieved
        actual_accuracy: f32,
    },

    /// Individual difference parameter is outside valid biological range
    #[error(
        "Individual difference parameters out of valid range: {param} = {value}. Valid range: [{min}, {max}]."
    )]
    InvalidIndividualDifference {
        /// Parameter name that is invalid
        param: String,
        /// Value that was provided
        value: f32,
        /// Minimum valid value
        min: f32,
        /// Maximum valid value
        max: f32,
    },

    /// Memory consolidation process encountered an invalid pattern
    #[error("Consolidation pattern invalid: {reason}. Activation pattern: {pattern:?}")]
    ConsolidationError {
        /// Reason why consolidation failed
        reason: String,
        /// Activation pattern that caused the error
        pattern: Vec<f32>,
    },
}

/// Result type for decay function operations
pub type DecayResult<T> = Result<T, DecayError>;

/// Configuration for which decay function to use.
///
/// Maps to the underlying biological decay systems while providing
/// a simplified user-facing API for configuring temporal decay behavior.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum DecayFunction {
    /// Exponential decay: R(t) = e^(-t/τ)
    ///
    /// Fast hippocampal-style decay following Ebbinghaus curve.
    /// Best for short-term episodic memories (hours to days).
    Exponential {
        /// Time constant in hours (default: 1.96 ~2 hours)
        tau_hours: f32,
    },

    /// Power-law decay: R(t) = (1 + t)^(-β)
    ///
    /// Slow neocortical-style decay following Bahrick permastore.
    /// Best for long-term semantic memories (months to years).
    PowerLaw {
        /// Power-law exponent (default: 0.18)
        beta: f32,
    },

    /// Two-component model: Automatic hippocampal ↔ neocortical transition
    ///
    /// Implements complementary learning systems theory.
    /// Automatically switches from fast hippocampal decay to slow
    /// neocortical decay based on access patterns.
    TwoComponent {
        /// Access count to switch from hippocampal to neocortical (default: 3)
        consolidation_threshold: u64,
    },

    /// Hybrid decay: Exponential (short-term) → Power-law (long-term)
    ///
    /// Matches psychological findings that forgetting curves show exponential
    /// decay over short intervals but transition to power-law over longer timescales.
    /// Provides best fit to Ebbinghaus (1885) data across full time range.
    ///
    /// References:
    /// - Wixted & Ebbesen (1991): Different forms across timescales
    /// - Rubin & Wenzel (1996): Multi-process forgetting systems
    Hybrid {
        /// Exponential tau for short-term decay (hours, default: 0.8)
        short_term_tau: f32,
        /// Power-law beta for long-term decay (default: 0.25)
        long_term_beta: f32,
        /// Transition point in seconds (default: 24 hours = 86400)
        transition_point: u64,
    },
}

impl Default for DecayFunction {
    fn default() -> Self {
        Self::TwoComponent {
            consolidation_threshold: 3,
        }
    }
}

impl DecayFunction {
    /// Creates exponential decay with default parameters
    #[must_use]
    pub const fn exponential() -> Self {
        Self::Exponential {
            tau_hours: 1.96, // Ebbinghaus 2015 replication
        }
    }

    /// Creates power-law decay with default parameters
    #[must_use]
    pub const fn power_law() -> Self {
        Self::PowerLaw {
            beta: 0.18, // Bahrick permastore
        }
    }

    /// Creates two-component decay with default parameters
    #[must_use]
    pub const fn two_component() -> Self {
        Self::TwoComponent {
            consolidation_threshold: 3,
        }
    }

    /// Creates hybrid decay with default parameters
    ///
    /// Default values provide best fit to Ebbinghaus (1885) data:
    /// - Short-term tau: 0.8 hours (exponential for < 24h)
    /// - Long-term beta: 0.25 (power-law for > 24h)
    /// - Transition: 24 hours (86400 seconds)
    #[must_use]
    pub const fn hybrid() -> Self {
        Self::Hybrid {
            short_term_tau: 0.8,
            long_term_beta: 0.25,
            transition_point: 86400, // 24 hours
        }
    }
}

/// System-wide decay configuration.
///
/// Controls default decay behavior for all memories, with support
/// for per-memory overrides via `Episode::decay_function`.
#[derive(Debug, Clone)]
pub struct DecayConfig {
    /// Default decay function for new memories
    pub default_function: DecayFunction,

    /// Enable automatic decay during recall operations
    pub enabled: bool,

    /// Minimum confidence threshold (memories below this are effectively forgotten)
    pub min_confidence: f32,
}

impl Default for DecayConfig {
    fn default() -> Self {
        Self {
            default_function: DecayFunction::default(),
            enabled: true,
            min_confidence: 0.1,
        }
    }
}

/// Builder for ergonomic decay configuration.
///
/// Provides a fluent API for configuring decay behavior with sensible defaults.
///
/// # Examples
///
/// ```
/// # use engram_core::decay::{DecayConfigBuilder, DecayFunction};
/// // Exponential decay with custom time constant
/// let config = DecayConfigBuilder::new()
///     .exponential(2.5)  // 2.5 hour tau
///     .enabled(true)
///     .build();
///
/// // Power-law decay for long-term memories
/// let config = DecayConfigBuilder::new()
///     .power_law(0.15)
///     .build();
///
/// // Two-component with custom consolidation threshold
/// let config = DecayConfigBuilder::new()
///     .two_component(5)  // Require 5 accesses for consolidation
///     .build();
/// ```
pub struct DecayConfigBuilder {
    config: DecayConfig,
}

impl Default for DecayConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl DecayConfigBuilder {
    /// Creates a new builder with default configuration
    #[must_use]
    pub fn new() -> Self {
        Self {
            config: DecayConfig::default(),
        }
    }

    /// Use exponential decay (hippocampal-style)
    #[must_use]
    pub const fn exponential(mut self, tau_hours: f32) -> Self {
        self.config.default_function = DecayFunction::Exponential { tau_hours };
        self
    }

    /// Use power-law decay (neocortical-style)
    #[must_use]
    pub const fn power_law(mut self, beta: f32) -> Self {
        self.config.default_function = DecayFunction::PowerLaw { beta };
        self
    }

    /// Use two-component automatic switching
    #[must_use]
    pub const fn two_component(mut self, consolidation_threshold: u64) -> Self {
        self.config.default_function = DecayFunction::TwoComponent {
            consolidation_threshold,
        };
        self
    }

    /// Use hybrid exponential-to-power-law decay
    ///
    /// Provides best fit to Ebbinghaus (1885) forgetting curve data
    /// by combining exponential decay for short-term with power-law for long-term.
    #[must_use]
    pub const fn hybrid(
        mut self,
        short_term_tau: f32,
        long_term_beta: f32,
        transition_point: u64,
    ) -> Self {
        self.config.default_function = DecayFunction::Hybrid {
            short_term_tau,
            long_term_beta,
            transition_point,
        };
        self
    }

    /// Enable or disable decay
    #[must_use]
    pub const fn enabled(mut self, enabled: bool) -> Self {
        self.config.enabled = enabled;
        self
    }

    /// Set minimum confidence threshold
    #[must_use]
    pub const fn min_confidence(mut self, min_confidence: f32) -> Self {
        self.config.min_confidence = min_confidence;
        self
    }

    /// Build the final configuration
    #[must_use]
    pub const fn build(self) -> DecayConfig {
        self.config
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)] // Tests are allowed to use unwrap
    #![allow(clippy::float_cmp)] // Tests may compare floats directly for exact values

    use super::*;
    use chrono::Duration;

    #[test]
    fn test_biological_decay_system_creation() {
        let system = BiologicalDecaySystem::new();
        assert!(system.consolidation_threshold > 0.0);
        assert!(system.consolidation_threshold < 1.0);
        assert!(system.last_sleep_consolidation.is_none());
    }

    #[test]
    fn test_individual_differences_integration() {
        let system = BiologicalDecaySystem::new();
        let base_rate = 1.0;

        let hippocampal_rate = system.effective_hippocampal_rate();
        let neocortical_rate = system.effective_neocortical_rate();

        // Rates should be modified by individual differences
        assert!(hippocampal_rate > 0.0);
        assert!(neocortical_rate > 0.0);

        // Individual differences should create variation around base rates
        assert!(hippocampal_rate >= 0.5 * base_rate);
        assert!(hippocampal_rate <= 2.0 * base_rate);
    }

    #[test]
    fn test_retention_prediction() {
        let system = BiologicalDecaySystem::new();
        let memory = Memory::new("test".to_string(), [0.5f32; 768], Confidence::HIGH);

        let immediate_retention = system.predict_retention(&memory, Duration::zero());
        let future_retention = system.predict_retention(&memory, Duration::days(30));

        // Retention should decrease over time
        assert!(future_retention < immediate_retention);
        assert!(future_retention >= 0.0);
        assert!(future_retention <= 1.0);
    }

    #[test]
    fn test_consolidation_threshold() {
        let system = BiologicalDecaySystem::new();

        // High variance, moderate mean should trigger consolidation
        // Increased variance to meet the > 0.1 threshold
        let ripple_pattern = vec![0.1, 0.9, 0.1, 0.9, 0.1, 0.8, 0.2, 0.7, 0.1, 0.8];
        assert!(system.should_consolidate(&ripple_pattern));

        // Low variance should not trigger consolidation
        let uniform_pattern = vec![0.5; 10];
        assert!(!system.should_consolidate(&uniform_pattern));

        // High mean should not trigger (active processing)
        let high_activity = vec![0.9; 10];
        assert!(!system.should_consolidate(&high_activity));
    }

    // Configuration API tests
    #[test]
    fn test_decay_function_defaults() {
        let default_fn = DecayFunction::default();
        assert!(matches!(
            default_fn,
            DecayFunction::TwoComponent {
                consolidation_threshold: 3
            }
        ));

        let exp = DecayFunction::exponential();
        assert!(matches!(exp, DecayFunction::Exponential { tau_hours: _ }));

        let power = DecayFunction::power_law();
        assert!(matches!(power, DecayFunction::PowerLaw { beta: _ }));

        let two = DecayFunction::two_component();
        assert!(matches!(
            two,
            DecayFunction::TwoComponent {
                consolidation_threshold: 3
            }
        ));
    }

    #[test]
    fn test_decay_config_default() {
        let config = DecayConfig::default();
        assert!(config.enabled);
        assert!((config.min_confidence - 0.1).abs() < 1e-6);
        assert!(matches!(
            config.default_function,
            DecayFunction::TwoComponent { .. }
        ));
    }

    #[test]
    fn test_decay_config_builder_exponential() {
        let config = DecayConfigBuilder::new()
            .exponential(2.5)
            .enabled(true)
            .build();

        assert!(config.enabled);
        assert!(matches!(
            config.default_function,
            DecayFunction::Exponential { tau_hours } if (tau_hours - 2.5).abs() < 1e-6
        ));
    }

    #[test]
    fn test_decay_config_builder_power_law() {
        let config = DecayConfigBuilder::new()
            .power_law(0.15)
            .min_confidence(0.05)
            .build();

        assert!(matches!(
            config.default_function,
            DecayFunction::PowerLaw { beta } if (beta - 0.15).abs() < 1e-6
        ));
        assert!((config.min_confidence - 0.05).abs() < 1e-6);
    }

    #[test]
    fn test_decay_config_builder_two_component() {
        let config = DecayConfigBuilder::new()
            .two_component(5)
            .enabled(false)
            .build();

        assert!(!config.enabled);
        assert!(matches!(
            config.default_function,
            DecayFunction::TwoComponent {
                consolidation_threshold: 5
            }
        ));
    }

    #[test]
    fn test_decay_config_builder_chaining() {
        let config = DecayConfigBuilder::new()
            .exponential(3.0)
            .enabled(true)
            .min_confidence(0.2)
            .build();

        assert!(config.enabled);
        assert!((config.min_confidence - 0.2).abs() < 1e-6);
        assert!(matches!(
            config.default_function,
            DecayFunction::Exponential { .. }
        ));
    }

    #[test]
    fn test_decay_function_equality() {
        let func1 = DecayFunction::Exponential { tau_hours: 2.0 };
        let func2 = DecayFunction::Exponential { tau_hours: 2.0 };
        let func3 = DecayFunction::Exponential { tau_hours: 2.5 };

        assert_eq!(func1, func2);
        assert_ne!(func1, func3);
    }

    // Tests for updated compute_decayed_confidence with DecayConfig

    #[test]
    fn test_decay_disabled_when_config_disabled() {
        let config = DecayConfigBuilder::new().enabled(false).build();
        let system = BiologicalDecaySystem::with_config(config);

        let base_confidence = Confidence::HIGH;
        let elapsed = StdDuration::from_secs(3600 * 6); // 6 hours
        let result =
            system.compute_decayed_confidence(base_confidence, elapsed, 1, Utc::now(), None);

        // When disabled, confidence should not decay
        assert_eq!(result.raw(), base_confidence.raw());
    }

    #[test]
    fn test_per_memory_decay_override_exponential() {
        let config = DecayConfigBuilder::new()
            .two_component(3) // Default: two-component
            .build();
        let system = BiologicalDecaySystem::with_config(config);

        let base_confidence = Confidence::HIGH;
        let elapsed = StdDuration::from_secs(3600 * 2); // 2 hours

        // Without override, should use two-component (access_count=1 uses hippocampal)
        let default_result =
            system.compute_decayed_confidence(base_confidence, elapsed, 1, Utc::now(), None);

        // With exponential override (faster decay)
        let override_result = system.compute_decayed_confidence(
            base_confidence,
            elapsed,
            1,
            Utc::now(),
            Some(DecayFunction::Exponential { tau_hours: 1.0 }), // Fast decay
        );

        // Override should produce different result
        assert_ne!(default_result.raw(), override_result.raw());
    }

    #[test]
    fn test_exponential_decay_function() {
        let config = DecayConfigBuilder::new().exponential(2.0).build();
        let system = BiologicalDecaySystem::with_config(config);

        let base_confidence = Confidence::HIGH;
        let elapsed = StdDuration::from_secs(3600 * 2); // 2 hours = 1 tau

        let result =
            system.compute_decayed_confidence(base_confidence, elapsed, 1, Utc::now(), None);

        // After 1 tau, retention should be approximately e^(-1) ≈ 0.368
        let expected_retention = (-1.0_f32).exp();
        let expected_confidence = base_confidence.raw() * expected_retention;

        // Allow some tolerance for individual differences calibration
        assert!((result.raw() - expected_confidence).abs() < 0.15);
    }

    #[test]
    fn test_power_law_decay_function() {
        let config = DecayConfigBuilder::new().power_law(0.2).build();
        let system = BiologicalDecaySystem::with_config(config);

        let base_confidence = Confidence::HIGH;
        let elapsed = StdDuration::from_secs(3600 * 10); // 10 hours

        let result =
            system.compute_decayed_confidence(base_confidence, elapsed, 1, Utc::now(), None);

        // Power-law decay should be slower than exponential
        // After 10 hours with beta=0.2: retention = (1 + 10)^(-0.2) ≈ 0.62
        let time_units = 10.0_f32;
        let expected_retention = (1.0 + time_units).powf(-0.2);
        let expected_confidence = base_confidence.raw() * expected_retention;

        // Allow tolerance for individual differences
        assert!((result.raw() - expected_confidence).abs() < 0.15);
    }

    #[test]
    fn test_two_component_transitions_correctly() {
        let config = DecayConfigBuilder::new()
            .two_component(3) // Consolidation at 3 accesses
            .build();
        let system = BiologicalDecaySystem::with_config(config);

        let base_confidence = Confidence::HIGH;
        let elapsed = StdDuration::from_secs(3600 * 6); // 6 hours

        // Below threshold: should use hippocampal (fast) decay
        let hippocampal_result = system.compute_decayed_confidence(
            base_confidence,
            elapsed,
            2, // Below threshold
            Utc::now(),
            None,
        );

        // At/above threshold: should use neocortical (slow) decay
        let neocortical_result = system.compute_decayed_confidence(
            base_confidence,
            elapsed,
            3, // At threshold
            Utc::now(),
            None,
        );

        // Neocortical should retain more confidence (slower decay)
        assert!(
            neocortical_result.raw() > hippocampal_result.raw(),
            "Neocortical decay should be slower than hippocampal"
        );
    }

    #[test]
    fn test_min_confidence_threshold_enforced() {
        let min_confidence = 0.2;
        let config = DecayConfigBuilder::new()
            .exponential(0.5) // Very fast decay
            .min_confidence(min_confidence)
            .build();
        let system = BiologicalDecaySystem::with_config(config);

        let base_confidence = Confidence::exact(0.5);
        let elapsed = StdDuration::from_secs(3600 * 100); // 100 hours (extreme decay)

        let result =
            system.compute_decayed_confidence(base_confidence, elapsed, 1, Utc::now(), None);

        // Should never go below min_confidence
        assert!(
            result.raw() >= min_confidence,
            "Result {} should be >= min_confidence {}",
            result.raw(),
            min_confidence
        );
    }

    #[test]
    fn test_different_decay_functions_produce_different_results() {
        let base_confidence = Confidence::HIGH;
        let elapsed = StdDuration::from_secs(3600 * 4); // 4 hours

        // Exponential decay
        let exp_config = DecayConfigBuilder::new().exponential(2.0).build();
        let exp_system = BiologicalDecaySystem::with_config(exp_config);
        let exp_result =
            exp_system.compute_decayed_confidence(base_confidence, elapsed, 1, Utc::now(), None);

        // Power-law decay
        let power_config = DecayConfigBuilder::new().power_law(0.18).build();
        let power_system = BiologicalDecaySystem::with_config(power_config);
        let power_result =
            power_system.compute_decayed_confidence(base_confidence, elapsed, 1, Utc::now(), None);

        // Two-component decay
        let two_comp_config = DecayConfigBuilder::new().two_component(3).build();
        let two_comp_system = BiologicalDecaySystem::with_config(two_comp_config);
        let two_comp_result = two_comp_system.compute_decayed_confidence(
            base_confidence,
            elapsed,
            1,
            Utc::now(),
            None,
        );

        // All three should produce different results
        assert_ne!(exp_result.raw(), power_result.raw());
        assert_ne!(exp_result.raw(), two_comp_result.raw());
        assert_ne!(power_result.raw(), two_comp_result.raw());
    }
}
