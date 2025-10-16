//! Uncertainty Tracking System
//!
//! Provides system-wide tracking and quantification of uncertainty sources that affect
//! query confidence scores. Aggregates uncertainty from multiple sources (system pressure,
//! spreading activation noise, temporal decay, measurement errors) and computes their
//! combined impact on result confidence.
//!
//! # Architecture
//!
//! The tracker uses:
//! - Lightweight aggregation with minimal allocations
//! - Statistical summarization of uncertainty impacts
//! - Breakdown by source type for diagnostics
//! - Fast quantification methods (<1% overhead target)
//!
//! # Performance
//!
//! Target: <1% overhead added to base query latency
//!
//! # Example
//!
//! ```
//! use engram_core::query::uncertainty_tracker::UncertaintyTracker;
//! use engram_core::query::UncertaintySource;
//!
//! let mut tracker = UncertaintyTracker::new();
//!
//! // Add uncertainty sources from different parts of the system
//! tracker.add_source(UncertaintySource::SystemPressure {
//!     pressure_level: 0.3,
//!     effect_on_confidence: 0.05,
//! });
//!
//! tracker.add_source(UncertaintySource::SpreadingActivationNoise {
//!     activation_variance: 0.1,
//!     path_diversity: 0.8,
//! });
//!
//! // Get total uncertainty impact
//! let total_impact = tracker.total_uncertainty_impact();
//! assert!(total_impact > 0.0);
//!
//! // Get breakdown by source type
//! let summary = tracker.summarize();
//! assert!(summary.source_count > 0);
//! ```

use crate::Confidence;
use crate::query::UncertaintySource;

/// Tracks and quantifies uncertainty sources affecting query results
#[derive(Debug, Clone)]
pub struct UncertaintyTracker {
    /// Collection of uncertainty sources
    sources: Vec<UncertaintySource>,
    /// Cached total uncertainty impact (recomputed when sources change)
    cached_impact: Option<f32>,
}

impl UncertaintyTracker {
    /// Create a new empty uncertainty tracker
    #[must_use]
    pub const fn new() -> Self {
        Self {
            sources: Vec::new(),
            cached_impact: None,
        }
    }

    /// Add an uncertainty source to the tracker
    ///
    /// Invalidates cached computations.
    pub fn add_source(&mut self, source: UncertaintySource) {
        self.sources.push(source);
        self.cached_impact = None; // Invalidate cache
    }

    /// Add multiple uncertainty sources at once
    pub fn add_sources(&mut self, sources: impl IntoIterator<Item = UncertaintySource>) {
        self.sources.extend(sources);
        self.cached_impact = None;
    }

    /// Get the total number of uncertainty sources tracked
    #[must_use]
    pub const fn source_count(&self) -> usize {
        self.sources.len()
    }

    /// Check if any uncertainty sources are being tracked
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.sources.is_empty()
    }

    /// Clear all uncertainty sources
    pub fn clear(&mut self) {
        self.sources.clear();
        self.cached_impact = None;
    }

    /// Get reference to all tracked sources
    #[must_use]
    pub fn sources(&self) -> &[UncertaintySource] {
        &self.sources
    }

    /// Compute total uncertainty impact on confidence
    ///
    /// Aggregates the effect of all uncertainty sources using probabilistic combination.
    /// Uses cached value if available for performance.
    ///
    /// # Returns
    ///
    /// A value in [0, 1] representing the estimated reduction in confidence certainty.
    /// Higher values indicate more uncertainty.
    #[must_use]
    pub fn total_uncertainty_impact(&mut self) -> f32 {
        if let Some(cached) = self.cached_impact {
            return cached;
        }

        if self.sources.is_empty() {
            self.cached_impact = Some(0.0);
            return 0.0;
        }

        // Aggregate uncertainty impacts using probabilistic OR formula
        // Total uncertainty = 1 - ∏(1 - individual_uncertainty)
        let mut log_complement = 0.0f64;

        for source in &self.sources {
            let impact = source.uncertainty_impact();
            let complement = (1.0 - f64::from(impact)).clamp(f64::MIN_POSITIVE, 1.0);
            log_complement += complement.ln();
        }

        #[allow(clippy::cast_precision_loss)]
        #[allow(clippy::cast_possible_truncation)]
        let total_impact = (1.0 - log_complement.exp()).clamp(0.0, 1.0) as f32;

        self.cached_impact = Some(total_impact);
        total_impact
    }

    /// Apply uncertainty to a confidence value
    ///
    /// Reduces confidence based on tracked uncertainty sources.
    ///
    /// # Arguments
    ///
    /// * `confidence` - Original confidence value
    ///
    /// # Returns
    ///
    /// Adjusted confidence accounting for uncertainty
    #[must_use]
    pub fn apply_uncertainty(&mut self, confidence: Confidence) -> Confidence {
        if self.sources.is_empty() {
            return confidence;
        }

        let uncertainty = self.total_uncertainty_impact();
        let raw = confidence.raw();

        // Reduce confidence by uncertainty amount, but keep it above zero
        let adjusted = raw * (1.0 - uncertainty);

        Confidence::from_raw(adjusted.clamp(0.0, 1.0))
    }

    /// Compute maximum single-source uncertainty impact
    ///
    /// Finds the largest uncertainty contribution from any single source.
    #[must_use]
    pub fn max_single_source_impact(&self) -> f32 {
        self.sources
            .iter()
            .map(UncertaintySource::uncertainty_impact)
            .max_by(f32::total_cmp)
            .unwrap_or(0.0)
    }

    /// Get breakdown of uncertainty by source type
    #[must_use]
    pub fn summarize(&self) -> UncertaintySummary {
        let mut system_pressure_count = 0;
        let mut spreading_noise_count = 0;
        let mut temporal_decay_count = 0;
        let mut measurement_error_count = 0;

        let mut total_system_pressure_impact = 0.0f32;
        let mut total_spreading_impact = 0.0f32;
        let mut total_temporal_impact = 0.0f32;
        let mut total_measurement_impact = 0.0f32;

        for source in &self.sources {
            let impact = source.uncertainty_impact();

            match source {
                UncertaintySource::SystemPressure { .. } => {
                    system_pressure_count += 1;
                    total_system_pressure_impact += impact;
                }
                UncertaintySource::SpreadingActivationNoise { .. } => {
                    spreading_noise_count += 1;
                    total_spreading_impact += impact;
                }
                UncertaintySource::TemporalDecayUnknown { .. } => {
                    temporal_decay_count += 1;
                    total_temporal_impact += impact;
                }
                UncertaintySource::MeasurementError { .. } => {
                    measurement_error_count += 1;
                    total_measurement_impact += impact;
                }
            }
        }

        UncertaintySummary {
            source_count: self.sources.len(),
            system_pressure_count,
            spreading_noise_count,
            temporal_decay_count,
            measurement_error_count,
            total_system_pressure_impact,
            total_spreading_impact,
            total_temporal_impact,
            total_measurement_impact,
        }
    }

    /// Get all sources of a specific type
    #[must_use]
    pub fn sources_by_type(&self, source_type: UncertaintySourceType) -> Vec<&UncertaintySource> {
        self.sources
            .iter()
            .filter(|s| s.source_type() == source_type)
            .collect()
    }
}

impl Default for UncertaintyTracker {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistical summary of tracked uncertainty sources
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct UncertaintySummary {
    /// Total number of uncertainty sources
    pub source_count: usize,
    /// Number of system pressure sources
    pub system_pressure_count: usize,
    /// Number of spreading activation noise sources
    pub spreading_noise_count: usize,
    /// Number of temporal decay uncertainty sources
    pub temporal_decay_count: usize,
    /// Number of measurement error sources
    pub measurement_error_count: usize,
    /// Total impact from system pressure
    pub total_system_pressure_impact: f32,
    /// Total impact from spreading activation noise
    pub total_spreading_impact: f32,
    /// Total impact from temporal decay
    pub total_temporal_impact: f32,
    /// Total impact from measurement errors
    pub total_measurement_impact: f32,
}

impl UncertaintySummary {
    /// Check if any uncertainty is being tracked
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.source_count == 0
    }

    /// Get the dominant uncertainty source type
    ///
    /// Returns the type with the highest total impact.
    #[must_use]
    pub fn dominant_source_type(&self) -> Option<UncertaintySourceType> {
        if self.is_empty() {
            return None;
        }

        // Find the source type with maximum impact
        let impacts = [
            (
                self.total_system_pressure_impact,
                UncertaintySourceType::SystemPressure,
            ),
            (
                self.total_spreading_impact,
                UncertaintySourceType::SpreadingActivationNoise,
            ),
            (
                self.total_temporal_impact,
                UncertaintySourceType::TemporalDecayUnknown,
            ),
            (
                self.total_measurement_impact,
                UncertaintySourceType::MeasurementError,
            ),
        ];

        impacts
            .iter()
            .max_by(|(a, _), (b, _)| a.total_cmp(b))
            .map(|(_, source_type)| *source_type)
    }
}

/// Type classification for uncertainty sources
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum UncertaintySourceType {
    /// System pressure uncertainty
    SystemPressure,
    /// Spreading activation noise
    SpreadingActivationNoise,
    /// Temporal decay uncertainty
    TemporalDecayUnknown,
    /// Measurement error
    MeasurementError,
}

impl UncertaintySource {
    /// Get the type classification of this uncertainty source
    #[must_use]
    pub const fn source_type(&self) -> UncertaintySourceType {
        match self {
            Self::SystemPressure { .. } => UncertaintySourceType::SystemPressure,
            Self::SpreadingActivationNoise { .. } => {
                UncertaintySourceType::SpreadingActivationNoise
            }
            Self::TemporalDecayUnknown { .. } => UncertaintySourceType::TemporalDecayUnknown,
            Self::MeasurementError { .. } => UncertaintySourceType::MeasurementError,
        }
    }

    /// Compute the uncertainty impact of this source
    ///
    /// Returns a value in [0, 1] representing how much this source
    /// reduces confidence certainty.
    #[must_use]
    pub fn uncertainty_impact(&self) -> f32 {
        match self {
            Self::SystemPressure {
                effect_on_confidence,
                ..
            } => effect_on_confidence.clamp(0.0, 1.0),

            Self::SpreadingActivationNoise {
                activation_variance,
                path_diversity,
            } => {
                // Combine variance and diversity into uncertainty estimate
                // Higher variance and lower diversity mean more uncertainty
                let variance_impact = activation_variance.clamp(0.0, 1.0);
                let diversity_impact = (1.0 - path_diversity).clamp(0.0, 1.0);

                // Weight variance more heavily as it directly affects results
                (variance_impact * 0.7 + diversity_impact * 0.3).clamp(0.0, 1.0)
            }

            Self::TemporalDecayUnknown {
                decay_model_uncertainty,
                ..
            } => decay_model_uncertainty.clamp(0.0, 1.0),

            Self::MeasurementError {
                confidence_degradation,
                ..
            } => confidence_degradation.clamp(0.0, 1.0),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn test_empty_tracker() {
        let tracker = UncertaintyTracker::new();
        assert_eq!(tracker.source_count(), 0);
        assert!(tracker.is_empty());
    }

    #[test]
    fn test_add_single_source() {
        let mut tracker = UncertaintyTracker::new();
        tracker.add_source(UncertaintySource::SystemPressure {
            pressure_level: 0.5,
            effect_on_confidence: 0.1,
        });

        assert_eq!(tracker.source_count(), 1);
        assert!(!tracker.is_empty());
    }

    #[test]
    fn test_total_uncertainty_impact_single_source() {
        let mut tracker = UncertaintyTracker::new();
        tracker.add_source(UncertaintySource::SystemPressure {
            pressure_level: 0.5,
            effect_on_confidence: 0.2,
        });

        let impact = tracker.total_uncertainty_impact();
        assert!((impact - 0.2).abs() < 1e-6);
    }

    #[test]
    fn test_total_uncertainty_impact_multiple_sources() {
        let mut tracker = UncertaintyTracker::new();
        tracker.add_source(UncertaintySource::SystemPressure {
            pressure_level: 0.5,
            effect_on_confidence: 0.1,
        });
        tracker.add_source(UncertaintySource::MeasurementError {
            error_magnitude: 0.05,
            confidence_degradation: 0.15,
        });

        let impact = tracker.total_uncertainty_impact();

        // Combined: 1 - (1-0.1)*(1-0.15) = 1 - 0.9*0.85 = 1 - 0.765 = 0.235
        let expected = 0.235;
        assert!((impact - expected).abs() < 1e-5);
    }

    #[test]
    fn test_impact_caching() {
        let mut tracker = UncertaintyTracker::new();
        tracker.add_source(UncertaintySource::SystemPressure {
            pressure_level: 0.5,
            effect_on_confidence: 0.2,
        });

        // First call computes
        let impact1 = tracker.total_uncertainty_impact();
        // Second call uses cache
        let impact2 = tracker.total_uncertainty_impact();

        assert!((impact1 - impact2).abs() < f32::EPSILON);
    }

    #[test]
    fn test_cache_invalidation_on_add() {
        let mut tracker = UncertaintyTracker::new();
        tracker.add_source(UncertaintySource::SystemPressure {
            pressure_level: 0.5,
            effect_on_confidence: 0.1,
        });

        let impact_before = tracker.total_uncertainty_impact();

        // Add another source - should invalidate cache
        tracker.add_source(UncertaintySource::MeasurementError {
            error_magnitude: 0.05,
            confidence_degradation: 0.1,
        });

        let impact_after = tracker.total_uncertainty_impact();

        // Impact should increase
        assert!(impact_after > impact_before);
    }

    #[test]
    fn test_apply_uncertainty_to_confidence() {
        let mut tracker = UncertaintyTracker::new();
        tracker.add_source(UncertaintySource::SystemPressure {
            pressure_level: 0.5,
            effect_on_confidence: 0.2,
        });

        let original = Confidence::from_raw(0.8);
        let adjusted = tracker.apply_uncertainty(original);

        // Should reduce by 20%: 0.8 * (1 - 0.2) = 0.64
        let expected = 0.64;
        assert!((adjusted.raw() - expected).abs() < 1e-5);
    }

    #[test]
    fn test_apply_uncertainty_empty_tracker() {
        let mut tracker = UncertaintyTracker::new();
        let original = Confidence::from_raw(0.8);
        let adjusted = tracker.apply_uncertainty(original);

        // No uncertainty sources, so no change
        assert!((adjusted.raw() - original.raw()).abs() < f32::EPSILON);
    }

    #[test]
    fn test_max_single_source_impact() {
        let mut tracker = UncertaintyTracker::new();
        tracker.add_source(UncertaintySource::SystemPressure {
            pressure_level: 0.5,
            effect_on_confidence: 0.1,
        });
        tracker.add_source(UncertaintySource::MeasurementError {
            error_magnitude: 0.1,
            confidence_degradation: 0.3,
        });
        tracker.add_source(UncertaintySource::SystemPressure {
            pressure_level: 0.2,
            effect_on_confidence: 0.05,
        });

        let max_impact = tracker.max_single_source_impact();
        assert!((max_impact - 0.3).abs() < 1e-6);
    }

    #[test]
    fn test_summary_breakdown() {
        let mut tracker = UncertaintyTracker::new();
        tracker.add_source(UncertaintySource::SystemPressure {
            pressure_level: 0.5,
            effect_on_confidence: 0.1,
        });
        tracker.add_source(UncertaintySource::SpreadingActivationNoise {
            activation_variance: 0.2,
            path_diversity: 0.8,
        });
        tracker.add_source(UncertaintySource::SystemPressure {
            pressure_level: 0.3,
            effect_on_confidence: 0.05,
        });

        let summary = tracker.summarize();

        assert_eq!(summary.source_count, 3);
        assert_eq!(summary.system_pressure_count, 2);
        assert_eq!(summary.spreading_noise_count, 1);
        assert_eq!(summary.temporal_decay_count, 0);
        assert_eq!(summary.measurement_error_count, 0);
    }

    #[test]
    fn test_sources_by_type() {
        let mut tracker = UncertaintyTracker::new();
        tracker.add_source(UncertaintySource::SystemPressure {
            pressure_level: 0.5,
            effect_on_confidence: 0.1,
        });
        tracker.add_source(UncertaintySource::SpreadingActivationNoise {
            activation_variance: 0.2,
            path_diversity: 0.8,
        });
        tracker.add_source(UncertaintySource::SystemPressure {
            pressure_level: 0.3,
            effect_on_confidence: 0.05,
        });

        let pressure_sources = tracker.sources_by_type(UncertaintySourceType::SystemPressure);
        assert_eq!(pressure_sources.len(), 2);

        let spreading_sources =
            tracker.sources_by_type(UncertaintySourceType::SpreadingActivationNoise);
        assert_eq!(spreading_sources.len(), 1);
    }

    #[test]
    fn test_clear() {
        let mut tracker = UncertaintyTracker::new();
        tracker.add_source(UncertaintySource::SystemPressure {
            pressure_level: 0.5,
            effect_on_confidence: 0.1,
        });

        assert_eq!(tracker.source_count(), 1);

        tracker.clear();

        assert_eq!(tracker.source_count(), 0);
        assert!(tracker.is_empty());
    }

    #[test]
    fn test_spreading_activation_noise_impact() {
        let source = UncertaintySource::SpreadingActivationNoise {
            activation_variance: 0.4,
            path_diversity: 0.6,
        };

        let impact = source.uncertainty_impact();

        // Impact = 0.4 * 0.7 + (1-0.6) * 0.3 = 0.28 + 0.12 = 0.4
        let expected = 0.4;
        assert!((impact - expected).abs() < 1e-5);
    }

    #[test]
    fn test_temporal_decay_uncertainty_impact() {
        let source = UncertaintySource::TemporalDecayUnknown {
            time_since_encoding: Duration::from_secs(3600),
            decay_model_uncertainty: 0.25,
        };

        let impact = source.uncertainty_impact();
        assert!((impact - 0.25).abs() < 1e-6);
    }

    #[test]
    fn test_dominant_source_type() {
        let mut tracker = UncertaintyTracker::new();
        tracker.add_source(UncertaintySource::SystemPressure {
            pressure_level: 0.5,
            effect_on_confidence: 0.1,
        });
        tracker.add_source(UncertaintySource::MeasurementError {
            error_magnitude: 0.2,
            confidence_degradation: 0.3,
        });

        let summary = tracker.summarize();
        let dominant = summary.dominant_source_type();

        assert_eq!(dominant, Some(UncertaintySourceType::MeasurementError));
    }

    #[test]
    fn test_add_sources_batch() {
        let mut tracker = UncertaintyTracker::new();
        let sources = vec![
            UncertaintySource::SystemPressure {
                pressure_level: 0.5,
                effect_on_confidence: 0.1,
            },
            UncertaintySource::MeasurementError {
                error_magnitude: 0.1,
                confidence_degradation: 0.05,
            },
        ];

        tracker.add_sources(sources);

        assert_eq!(tracker.source_count(), 2);
    }
}
