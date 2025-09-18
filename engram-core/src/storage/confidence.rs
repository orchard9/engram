//! Confidence calibration for storage operations
//!
//! This module provides confidence adjustment for storage tier retrieval
//! to ensure uncertainty is properly tracked across different storage tiers.

use crate::Confidence;
use std::time::{Duration, SystemTime};

/// Storage tier types for confidence calibration
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConfidenceTier {
    /// Hot tier: in-memory, no degradation
    Hot,
    /// Warm tier: memory-mapped, slight degradation from compression
    Warm,
    /// Cold tier: columnar storage, degradation from quantization
    Cold,
}

/// Configuration for tier-specific confidence factors
#[derive(Debug, Clone)]
pub struct TierConfidenceFactors {
    /// Hot tier factor: 1.0 - no degradation
    pub hot_factor: f32,
    /// Warm tier factor: 0.95 - slight degradation from compression
    pub warm_factor: f32,
    /// Cold tier factor: 0.9 - degradation from quantization
    pub cold_factor: f32,
}

impl Default for TierConfidenceFactors {
    fn default() -> Self {
        Self {
            hot_factor: 1.0,   // No degradation
            warm_factor: 0.95, // Slight degradation from compression
            cold_factor: 0.9,  // Degradation from quantization
        }
    }
}

/// Calibrator for adjusting confidence scores based on storage characteristics
#[derive(Debug, Clone)]
pub struct StorageConfidenceCalibrator {
    /// Tier-specific confidence factors
    tier_confidence_factors: TierConfidenceFactors,
    /// Whether temporal decay is enabled
    enable_temporal_decay: bool,
    /// Temporal decay half-life in days (default: 10 years)
    temporal_half_life_days: f32,
}

impl StorageConfidenceCalibrator {
    /// Create a new calibrator with default settings
    pub fn new() -> Self {
        Self {
            tier_confidence_factors: TierConfidenceFactors::default(),
            enable_temporal_decay: true,
            temporal_half_life_days: 3650.0, // 10 years
        }
    }

    /// Create a calibrator with custom tier factors
    pub fn with_tier_factors(factors: TierConfidenceFactors) -> Self {
        Self {
            tier_confidence_factors: factors,
            enable_temporal_decay: true,
            temporal_half_life_days: 3650.0,
        }
    }

    /// Create a calibrator with custom temporal decay settings
    pub fn with_temporal_decay(mut self, enabled: bool, half_life_days: f32) -> Self {
        self.enable_temporal_decay = enabled;
        self.temporal_half_life_days = half_life_days.max(1.0);
        self
    }

    /// Adjust confidence for storage tier characteristics and time in storage
    pub fn adjust_for_storage_tier(
        &self,
        original_confidence: Confidence,
        tier: ConfidenceTier,
        time_in_storage: Duration,
    ) -> Confidence {
        // Get tier-specific confidence factor
        let tier_factor = match tier {
            ConfidenceTier::Hot => self.tier_confidence_factors.hot_factor,
            ConfidenceTier::Warm => self.tier_confidence_factors.warm_factor,
            ConfidenceTier::Cold => self.tier_confidence_factors.cold_factor,
        };

        let mut adjusted_confidence = original_confidence.raw() * tier_factor;

        // Apply temporal decay if enabled
        if self.enable_temporal_decay {
            let days_stored = time_in_storage.as_secs_f32() / 86400.0;
            let temporal_factor = 0.5_f32.powf(days_stored / self.temporal_half_life_days);
            adjusted_confidence *= temporal_factor;
        }

        // Ensure confidence is within valid range with minimum threshold
        Confidence::from_raw(adjusted_confidence.max(0.01).min(1.0))
    }

    /// Adjust confidence for storage tier only (no temporal decay)
    pub fn adjust_for_tier_only(&self, original_confidence: Confidence, tier: ConfidenceTier) -> Confidence {
        let tier_factor = match tier {
            ConfidenceTier::Hot => self.tier_confidence_factors.hot_factor,
            ConfidenceTier::Warm => self.tier_confidence_factors.warm_factor,
            ConfidenceTier::Cold => self.tier_confidence_factors.cold_factor,
        };

        let adjusted = original_confidence.raw() * tier_factor;
        Confidence::from_raw(adjusted.max(0.01).min(1.0))
    }

    /// Get the confidence factor for a specific tier
    pub fn get_tier_factor(&self, tier: ConfidenceTier) -> f32 {
        match tier {
            ConfidenceTier::Hot => self.tier_confidence_factors.hot_factor,
            ConfidenceTier::Warm => self.tier_confidence_factors.warm_factor,
            ConfidenceTier::Cold => self.tier_confidence_factors.cold_factor,
        }
    }

    /// Calculate temporal decay factor for given time in storage
    pub fn calculate_temporal_factor(&self, time_in_storage: Duration) -> f32 {
        if !self.enable_temporal_decay {
            return 1.0;
        }

        let days_stored = time_in_storage.as_secs_f32() / 86400.0;
        0.5_f32.powf(days_stored / self.temporal_half_life_days)
    }

    /// Batch adjustment for multiple results
    pub fn adjust_batch(
        &self,
        results: &mut [(Confidence, ConfidenceTier, Duration)],
    ) {
        for (confidence, tier, time_in_storage) in results.iter_mut() {
            *confidence = self.adjust_for_storage_tier(*confidence, *tier, *time_in_storage);
        }
    }

    /// Update tier factors
    pub fn update_tier_factors(&mut self, factors: TierConfidenceFactors) {
        self.tier_confidence_factors = factors;
    }
}

impl Default for StorageConfidenceCalibrator {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics about confidence calibration
#[derive(Debug, Clone, Default)]
pub struct CalibrationStats {
    /// Total adjustments performed
    pub total_adjustments: usize,
    /// Average adjustment factor applied
    pub average_adjustment_factor: f32,
    /// Minimum confidence after adjustment
    pub min_adjusted_confidence: f32,
    /// Maximum confidence after adjustment
    pub max_adjusted_confidence: f32,
}

impl CalibrationStats {
    /// Record an adjustment
    pub fn record_adjustment(&mut self, original: f32, adjusted: f32) {
        let adjustment_factor = if original > 0.0 { adjusted / original } else { 1.0 };

        self.total_adjustments += 1;
        self.average_adjustment_factor =
            (self.average_adjustment_factor * (self.total_adjustments - 1) as f32 + adjustment_factor)
            / self.total_adjustments as f32;

        self.min_adjusted_confidence = self.min_adjusted_confidence.min(adjusted);
        self.max_adjusted_confidence = self.max_adjusted_confidence.max(adjusted);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_calibrator() {
        let calibrator = StorageConfidenceCalibrator::new();
        let original = Confidence::from_raw(0.8);

        // Hot tier should have no degradation
        let hot_adjusted = calibrator.adjust_for_tier_only(original, ConfidenceTier::Hot);
        assert_eq!(hot_adjusted.raw(), 0.8);

        // Warm tier should have slight degradation (0.95 factor)
        let warm_adjusted = calibrator.adjust_for_tier_only(original, ConfidenceTier::Warm);
        assert!((warm_adjusted.raw() - 0.76).abs() < 0.001); // 0.8 * 0.95

        // Cold tier should have more degradation (0.9 factor)
        let cold_adjusted = calibrator.adjust_for_tier_only(original, ConfidenceTier::Cold);
        assert!((cold_adjusted.raw() - 0.72).abs() < 0.001); // 0.8 * 0.9
    }

    #[test]
    fn test_temporal_decay() {
        let calibrator = StorageConfidenceCalibrator::new()
            .with_temporal_decay(true, 365.0); // 1 year half-life

        let original = Confidence::from_raw(0.8);
        let one_year = Duration::from_secs(365 * 24 * 3600);

        // After one half-life, should be reduced by ~50%
        let adjusted = calibrator.adjust_for_storage_tier(original, ConfidenceTier::Hot, one_year);
        assert!(adjusted.raw() < 0.5);
        assert!(adjusted.raw() > 0.3); // Should still be reasonable
    }

    #[test]
    fn test_confidence_bounds() {
        let calibrator = StorageConfidenceCalibrator::new();

        // Very low original confidence
        let low_original = Confidence::from_raw(0.02);
        let adjusted = calibrator.adjust_for_tier_only(low_original, ConfidenceTier::Cold);
        assert!(adjusted.raw() >= 0.01); // Should be clamped to minimum

        // High original confidence
        let high_original = Confidence::from_raw(0.99);
        let adjusted = calibrator.adjust_for_tier_only(high_original, ConfidenceTier::Hot);
        assert!(adjusted.raw() <= 1.0); // Should not exceed 1.0
    }

    #[test]
    fn test_custom_tier_factors() {
        let custom_factors = TierConfidenceFactors {
            hot_factor: 1.0,
            warm_factor: 0.8,
            cold_factor: 0.6,
        };

        let calibrator = StorageConfidenceCalibrator::with_tier_factors(custom_factors);
        let original = Confidence::from_raw(0.5);

        let cold_adjusted = calibrator.adjust_for_tier_only(original, ConfidenceTier::Cold);
        assert!((cold_adjusted.raw() - 0.3).abs() < 0.001); // 0.5 * 0.6
    }

    #[test]
    fn test_batch_adjustment() {
        let calibrator = StorageConfidenceCalibrator::new();
        let mut results = vec![
            (Confidence::from_raw(0.8), ConfidenceTier::Hot, Duration::from_secs(0)),
            (Confidence::from_raw(0.8), ConfidenceTier::Warm, Duration::from_secs(0)),
            (Confidence::from_raw(0.8), ConfidenceTier::Cold, Duration::from_secs(0)),
        ];

        calibrator.adjust_batch(&mut results);

        assert_eq!(results[0].0.raw(), 0.8);     // Hot: no change
        assert!((results[1].0.raw() - 0.76).abs() < 0.001); // Warm: 0.8 * 0.95
        assert!((results[2].0.raw() - 0.72).abs() < 0.001); // Cold: 0.8 * 0.9
    }

    #[test]
    fn test_calibration_stats() {
        let mut stats = CalibrationStats::default();

        stats.record_adjustment(0.8, 0.76); // 0.95 factor
        stats.record_adjustment(0.6, 0.54); // 0.9 factor

        assert_eq!(stats.total_adjustments, 2);
        assert!((stats.average_adjustment_factor - 0.925).abs() < 0.001); // (0.95 + 0.9) / 2
    }

    #[test]
    fn test_tier_factor_retrieval() {
        let calibrator = StorageConfidenceCalibrator::new();

        assert_eq!(calibrator.get_tier_factor(ConfidenceTier::Hot), 1.0);
        assert_eq!(calibrator.get_tier_factor(ConfidenceTier::Warm), 0.95);
        assert_eq!(calibrator.get_tier_factor(ConfidenceTier::Cold), 0.9);
    }

    #[test]
    fn test_temporal_factor_calculation() {
        let calibrator = StorageConfidenceCalibrator::new()
            .with_temporal_decay(true, 365.0); // 1 year half-life

        let one_year = Duration::from_secs(365 * 24 * 3600);
        let factor = calibrator.calculate_temporal_factor(one_year);

        // Should be approximately 0.5 after one half-life
        assert!((factor - 0.5).abs() < 0.1);

        // With temporal decay disabled
        let no_decay = StorageConfidenceCalibrator::new()
            .with_temporal_decay(false, 365.0);
        let factor_no_decay = no_decay.calculate_temporal_factor(one_year);
        assert_eq!(factor_no_decay, 1.0);
    }
}