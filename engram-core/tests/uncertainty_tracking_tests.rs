//! Integration tests for uncertainty tracking system
//!
//! Tests end-to-end uncertainty tracking with query executor integration,
//! multi-source aggregation, and confidence adjustment.

#![allow(clippy::panic)]
#![allow(clippy::field_reassign_with_default)]
#![allow(clippy::unwrap_used)] // Unwrap is acceptable in tests

use engram_core::Confidence;
use engram_core::query::UncertaintySource;
use engram_core::query::uncertainty_tracker::{UncertaintySourceType, UncertaintyTracker};
use std::time::Duration;

#[test]
fn test_end_to_end_uncertainty_tracking() {
    let mut tracker = UncertaintyTracker::new();

    // Simulate collecting uncertainty from multiple system components
    tracker.add_source(UncertaintySource::SystemPressure {
        pressure_level: 0.4,
        effect_on_confidence: 0.1,
    });

    tracker.add_source(UncertaintySource::SpreadingActivationNoise {
        activation_variance: 0.3,
        path_diversity: 0.7,
    });

    tracker.add_source(UncertaintySource::TemporalDecayUnknown {
        time_since_encoding: Duration::from_secs(7200), // 2 hours
        decay_model_uncertainty: 0.15,
    });

    // Verify all sources tracked
    assert_eq!(tracker.source_count(), 3);

    // Get total impact
    let total_impact = tracker.total_uncertainty_impact();
    assert!(total_impact > 0.0);
    assert!(total_impact < 1.0);

    // Get summary breakdown
    let summary = tracker.summarize();
    assert_eq!(summary.source_count, 3);
    assert_eq!(summary.system_pressure_count, 1);
    assert_eq!(summary.spreading_noise_count, 1);
    assert_eq!(summary.temporal_decay_count, 1);
}

#[test]
fn test_uncertainty_impact_on_high_confidence() {
    let mut tracker = UncertaintyTracker::new();

    tracker.add_source(UncertaintySource::SystemPressure {
        pressure_level: 0.5,
        effect_on_confidence: 0.2,
    });

    let original = Confidence::HIGH; // 0.9
    let adjusted = tracker.apply_uncertainty(original);

    // Should reduce confidence
    assert!(adjusted.raw() < original.raw());

    // Should reduce by approximately 20%
    // 0.9 * (1 - 0.2) = 0.72
    assert!((adjusted.raw() - 0.72).abs() < 0.01);
}

#[test]
fn test_uncertainty_impact_on_medium_confidence() {
    let mut tracker = UncertaintyTracker::new();

    tracker.add_source(UncertaintySource::MeasurementError {
        error_magnitude: 0.1,
        confidence_degradation: 0.15,
    });

    let original = Confidence::MEDIUM; // 0.5
    let adjusted = tracker.apply_uncertainty(original);

    // Should reduce: 0.5 * (1 - 0.15) = 0.425
    assert!((adjusted.raw() - 0.425).abs() < 0.01);
}

#[test]
fn test_multiple_uncertainty_sources_combine() {
    let mut tracker = UncertaintyTracker::new();

    // Add multiple sources that should combine
    tracker.add_source(UncertaintySource::SystemPressure {
        pressure_level: 0.3,
        effect_on_confidence: 0.1,
    });

    tracker.add_source(UncertaintySource::SystemPressure {
        pressure_level: 0.4,
        effect_on_confidence: 0.15,
    });

    tracker.add_source(UncertaintySource::MeasurementError {
        error_magnitude: 0.05,
        confidence_degradation: 0.05,
    });

    let total_impact = tracker.total_uncertainty_impact();

    // Combined using probabilistic OR:
    // 1 - (1-0.1)*(1-0.15)*(1-0.05) = 1 - 0.9*0.85*0.95 = 1 - 0.72675 = 0.27325
    let expected = 0.27325;
    assert!((total_impact - expected).abs() < 0.001);
}

#[test]
fn test_dominant_uncertainty_source_identification() {
    let mut tracker = UncertaintyTracker::new();

    tracker.add_source(UncertaintySource::SystemPressure {
        pressure_level: 0.2,
        effect_on_confidence: 0.05,
    });

    tracker.add_source(UncertaintySource::SpreadingActivationNoise {
        activation_variance: 0.5,
        path_diversity: 0.3,
    });

    tracker.add_source(UncertaintySource::MeasurementError {
        error_magnitude: 0.1,
        confidence_degradation: 0.02,
    });

    let summary = tracker.summarize();
    let dominant = summary.dominant_source_type();

    // Spreading activation should dominate with higher impact
    assert_eq!(
        dominant,
        Some(UncertaintySourceType::SpreadingActivationNoise)
    );
}

#[test]
fn test_filtering_by_source_type() {
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

    tracker.add_source(UncertaintySource::TemporalDecayUnknown {
        time_since_encoding: Duration::from_secs(3600),
        decay_model_uncertainty: 0.1,
    });

    let pressure_sources = tracker.sources_by_type(UncertaintySourceType::SystemPressure);
    assert_eq!(pressure_sources.len(), 2);

    let spreading_sources =
        tracker.sources_by_type(UncertaintySourceType::SpreadingActivationNoise);
    assert_eq!(spreading_sources.len(), 1);

    let temporal_sources = tracker.sources_by_type(UncertaintySourceType::TemporalDecayUnknown);
    assert_eq!(temporal_sources.len(), 1);

    let measurement_sources = tracker.sources_by_type(UncertaintySourceType::MeasurementError);
    assert_eq!(measurement_sources.len(), 0);
}

#[test]
fn test_uncertainty_tracker_reuse() {
    let mut tracker = UncertaintyTracker::new();

    // First query
    tracker.add_source(UncertaintySource::SystemPressure {
        pressure_level: 0.3,
        effect_on_confidence: 0.1,
    });

    let impact1 = tracker.total_uncertainty_impact();
    assert!((impact1 - 0.1).abs() < 0.001);

    // Clear for next query
    tracker.clear();
    assert!(tracker.is_empty());

    // Second query with different uncertainty
    tracker.add_source(UncertaintySource::MeasurementError {
        error_magnitude: 0.2,
        confidence_degradation: 0.25,
    });

    let impact2 = tracker.total_uncertainty_impact();
    assert!((impact2 - 0.25).abs() < 0.001);
}

#[test]
fn test_batch_uncertainty_addition() {
    let mut tracker = UncertaintyTracker::new();

    let sources = vec![
        UncertaintySource::SystemPressure {
            pressure_level: 0.4,
            effect_on_confidence: 0.1,
        },
        UncertaintySource::SpreadingActivationNoise {
            activation_variance: 0.3,
            path_diversity: 0.7,
        },
        UncertaintySource::TemporalDecayUnknown {
            time_since_encoding: Duration::from_secs(1800),
            decay_model_uncertainty: 0.12,
        },
    ];

    tracker.add_sources(sources);

    assert_eq!(tracker.source_count(), 3);

    let summary = tracker.summarize();
    assert_eq!(summary.system_pressure_count, 1);
    assert_eq!(summary.spreading_noise_count, 1);
    assert_eq!(summary.temporal_decay_count, 1);
}

#[test]
fn test_max_single_source_impact_identification() {
    let mut tracker = UncertaintyTracker::new();

    tracker.add_source(UncertaintySource::SystemPressure {
        pressure_level: 0.2,
        effect_on_confidence: 0.05,
    });

    tracker.add_source(UncertaintySource::MeasurementError {
        error_magnitude: 0.3,
        confidence_degradation: 0.35,
    });

    tracker.add_source(UncertaintySource::SystemPressure {
        pressure_level: 0.4,
        effect_on_confidence: 0.15,
    });

    let max_impact = tracker.max_single_source_impact();
    assert!((max_impact - 0.35).abs() < 0.001);
}

#[test]
fn test_zero_uncertainty_preserves_confidence() {
    let mut tracker = UncertaintyTracker::new();

    let original = Confidence::from_raw(0.75);
    let adjusted = tracker.apply_uncertainty(original);

    // No uncertainty sources, so confidence should be unchanged
    assert!((adjusted.raw() - original.raw()).abs() < f32::EPSILON);
}

#[test]
fn test_high_uncertainty_significantly_reduces_confidence() {
    let mut tracker = UncertaintyTracker::new();

    // Add high uncertainty sources
    tracker.add_source(UncertaintySource::SystemPressure {
        pressure_level: 0.8,
        effect_on_confidence: 0.4,
    });

    tracker.add_source(UncertaintySource::MeasurementError {
        error_magnitude: 0.5,
        confidence_degradation: 0.3,
    });

    tracker.add_source(UncertaintySource::SpreadingActivationNoise {
        activation_variance: 0.6,
        path_diversity: 0.2,
    });

    let original = Confidence::HIGH;
    let adjusted = tracker.apply_uncertainty(original);

    // Should significantly reduce confidence
    assert!(adjusted.raw() < original.raw() * 0.5);
}

#[test]
fn test_spreading_activation_noise_calculation() {
    let source = UncertaintySource::SpreadingActivationNoise {
        activation_variance: 0.4,
        path_diversity: 0.6,
    };

    let impact = source.uncertainty_impact();

    // Impact = 0.4 * 0.7 + (1-0.6) * 0.3 = 0.28 + 0.12 = 0.4
    let expected = 0.4;
    assert!((impact - expected).abs() < 0.001);
}

#[test]
fn test_spreading_activation_high_variance_low_diversity() {
    let source = UncertaintySource::SpreadingActivationNoise {
        activation_variance: 0.8,
        path_diversity: 0.2,
    };

    let impact = source.uncertainty_impact();

    // Impact = 0.8 * 0.7 + (1-0.2) * 0.3 = 0.56 + 0.24 = 0.8
    let expected = 0.8;
    assert!((impact - expected).abs() < 0.001);
}

#[test]
fn test_temporal_decay_uncertainty_proportional() {
    let recent = UncertaintySource::TemporalDecayUnknown {
        time_since_encoding: Duration::from_secs(300), // 5 minutes
        decay_model_uncertainty: 0.05,
    };

    let old = UncertaintySource::TemporalDecayUnknown {
        time_since_encoding: Duration::from_secs(86400), // 1 day
        decay_model_uncertainty: 0.3,
    };

    // Older memory has more uncertainty
    assert!(old.uncertainty_impact() > recent.uncertainty_impact());
}

#[test]
fn test_summary_statistics_accuracy() {
    let mut tracker = UncertaintyTracker::new();

    tracker.add_source(UncertaintySource::SystemPressure {
        pressure_level: 0.5,
        effect_on_confidence: 0.1,
    });

    tracker.add_source(UncertaintySource::SystemPressure {
        pressure_level: 0.3,
        effect_on_confidence: 0.05,
    });

    tracker.add_source(UncertaintySource::SpreadingActivationNoise {
        activation_variance: 0.2,
        path_diversity: 0.8,
    });

    let summary = tracker.summarize();

    // Check counts
    assert_eq!(summary.system_pressure_count, 2);
    assert_eq!(summary.spreading_noise_count, 1);

    // Check total impacts
    let expected_pressure_impact = 0.1 + 0.05;
    assert!((summary.total_system_pressure_impact - expected_pressure_impact).abs() < 0.001);

    // Spreading impact: 0.2 * 0.7 + (1-0.8) * 0.3 = 0.14 + 0.06 = 0.2
    let expected_spreading_impact = 0.2;
    assert!((summary.total_spreading_impact - expected_spreading_impact).abs() < 0.001);
}

#[test]
fn test_empty_summary() {
    let tracker = UncertaintyTracker::new();
    let summary = tracker.summarize();

    assert!(summary.is_empty());
    assert_eq!(summary.source_count, 0);
    assert_eq!(summary.dominant_source_type(), None);
}

#[test]
fn test_realistic_query_scenario() {
    // Simulate a realistic query with multiple uncertainty sources
    let mut tracker = UncertaintyTracker::new();

    // System under moderate load
    tracker.add_source(UncertaintySource::SystemPressure {
        pressure_level: 0.3,
        effect_on_confidence: 0.08,
    });

    // Spreading activation explored multiple paths with some variance
    tracker.add_source(UncertaintySource::SpreadingActivationNoise {
        activation_variance: 0.15,
        path_diversity: 0.75,
    });

    // Memory is a few hours old, some decay uncertainty
    tracker.add_source(UncertaintySource::TemporalDecayUnknown {
        time_since_encoding: Duration::from_secs(10800), // 3 hours
        decay_model_uncertainty: 0.1,
    });

    // Small measurement noise
    tracker.add_source(UncertaintySource::MeasurementError {
        error_magnitude: 0.03,
        confidence_degradation: 0.02,
    });

    // Apply to high confidence result
    let original = Confidence::from_raw(0.85);
    let adjusted = tracker.apply_uncertainty(original);

    // Should have modest reduction
    assert!(adjusted.raw() < original.raw());
    // Multiple uncertainty sources combine, so allow significant reduction
    assert!(adjusted.raw() > 0.5); // Still above medium confidence

    // Get diagnostic information
    let summary = tracker.summarize();
    assert_eq!(summary.source_count, 4);

    // All types represented
    assert!(summary.system_pressure_count > 0);
    assert!(summary.spreading_noise_count > 0);
    assert!(summary.temporal_decay_count > 0);
    assert!(summary.measurement_error_count > 0);
}
