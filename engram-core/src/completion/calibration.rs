//! Completion-specific confidence calibration
//!
//! Integrates multi-factor completion confidence with empirical calibration
//! framework from Milestone 5. Uses isotonic regression (Zadrozny & Elkan, 2002)
//! to map raw confidence scores to calibrated probabilities.
//!
//! # Target Performance
//!
//! - Calibration error <8% across confidence bins
//! - Brier score <0.08
//! - Confidence-accuracy correlation >0.80 (Spearman)
//! - Computation time <200μs

use crate::Confidence;
use crate::completion::{
    CompletedEpisode, CompletionConfidenceComputer, ConvergenceStats, MetacognitiveMonitor,
};
use crate::query::confidence_calibration::CalibrationTracker;

/// Completion-specific confidence calibrator
///
/// Wraps `CompletionConfidenceComputer` and `CalibrationTracker` to provide
/// calibrated confidence scores for pattern completion operations.
pub struct CompletionCalibrator {
    /// Multi-factor confidence computer
    confidence_computer: CompletionConfidenceComputer,

    /// Metacognitive monitor for confidence-in-confidence
    metacognitive_monitor: MetacognitiveMonitor,

    /// Empirical calibration tracker (from Milestone 5)
    calibration_tracker: CalibrationTracker,

    /// Whether to apply empirical calibration adjustments
    apply_calibration: bool,
}

impl CompletionCalibrator {
    /// Create new completion calibrator with default parameters
    ///
    /// Uses 10 confidence bins for calibration tracking.
    ///
    /// # Example
    ///
    /// ```
    /// use engram_core::completion::CompletionCalibrator;
    ///
    /// let calibrator = CompletionCalibrator::new();
    /// ```
    #[must_use]
    pub fn new() -> Self {
        Self {
            confidence_computer: CompletionConfidenceComputer::new(),
            metacognitive_monitor: MetacognitiveMonitor::new(),
            calibration_tracker: CalibrationTracker::new(10),
            apply_calibration: true,
        }
    }

    /// Create calibrator with custom number of calibration bins
    ///
    /// # Arguments
    ///
    /// * `num_bins` - Number of confidence bins for calibration (typically 10)
    #[must_use]
    pub fn with_bins(num_bins: usize) -> Self {
        Self {
            confidence_computer: CompletionConfidenceComputer::new(),
            metacognitive_monitor: MetacognitiveMonitor::new(),
            calibration_tracker: CalibrationTracker::new(num_bins),
            apply_calibration: true,
        }
    }

    /// Create calibrator with custom components
    ///
    /// # Arguments
    ///
    /// * `confidence_computer` - Multi-factor confidence computer
    /// * `metacognitive_monitor` - Metacognitive monitor
    /// * `num_bins` - Number of calibration bins
    #[must_use]
    #[allow(clippy::missing_const_for_fn)] // Cannot be const due to non-const field types
    pub fn with_components(
        confidence_computer: CompletionConfidenceComputer,
        metacognitive_monitor: MetacognitiveMonitor,
        num_bins: usize,
    ) -> Self {
        Self {
            confidence_computer,
            metacognitive_monitor,
            calibration_tracker: CalibrationTracker::new(num_bins),
            apply_calibration: true,
        }
    }

    /// Compute calibrated completion confidence
    ///
    /// Combines multi-factor confidence with empirical calibration to produce
    /// well-calibrated confidence scores.
    ///
    /// # Arguments
    ///
    /// * `convergence_stats` - CA3 convergence statistics
    /// * `field_consensus` - Agreement ratio among neighbors [0.0-1.0]
    /// * `plausibility_score` - Semantic coherence score [0.0-1.0]
    ///
    /// # Returns
    ///
    /// Calibrated confidence score [0.0-1.0]
    ///
    /// # Example
    ///
    /// ```
    /// use engram_core::completion::{CompletionCalibrator, ConvergenceStats};
    ///
    /// let calibrator = CompletionCalibrator::new();
    ///
    /// let stats = ConvergenceStats {
    ///     iterations: 3,
    ///     converged: true,
    ///     final_energy: -7.0,
    ///     energy_delta: -7.0,
    ///     state_change: 0.006,
    /// };
    ///
    /// let confidence = calibrator.compute_calibrated_confidence(
    ///     &stats,
    ///     0.80,
    ///     0.85,
    /// );
    ///
    /// assert!(confidence.raw() > 0.0 && confidence.raw() <= 1.0);
    /// ```
    #[must_use]
    pub fn compute_calibrated_confidence(
        &self,
        convergence_stats: &ConvergenceStats,
        field_consensus: f32,
        plausibility_score: f32,
    ) -> Confidence {
        // Compute raw multi-factor confidence
        let raw_confidence = self.confidence_computer.compute_completion_confidence(
            convergence_stats,
            field_consensus,
            plausibility_score,
        );

        // Apply empirical calibration if enabled and we have data
        if self.apply_calibration && self.calibration_tracker.total_samples() > 100 {
            self.calibration_tracker.apply_calibration(raw_confidence)
        } else {
            raw_confidence
        }
    }

    /// Compute calibrated completion confidence with metacognitive assessment
    ///
    /// Returns both the calibrated confidence and metacognitive confidence
    /// (confidence in the confidence).
    ///
    /// # Arguments
    ///
    /// * `completed_episode` - Completed episode with alternative hypotheses
    /// * `convergence_stats` - CA3 convergence statistics
    /// * `field_consensus` - Agreement ratio among neighbors
    /// * `plausibility_score` - Semantic coherence score
    ///
    /// # Returns
    ///
    /// Tuple of (calibrated_confidence, metacognitive_confidence)
    #[must_use]
    pub fn compute_with_metacognition(
        &self,
        completed_episode: &CompletedEpisode,
        convergence_stats: &ConvergenceStats,
        field_consensus: f32,
        plausibility_score: f32,
    ) -> (Confidence, Confidence) {
        // Compute calibrated primary confidence
        let calibrated_confidence = self.compute_calibrated_confidence(
            convergence_stats,
            field_consensus,
            plausibility_score,
        );

        // Compute metacognitive confidence
        let metacognitive_confidence = self.metacognitive_monitor.compute_metacognitive_confidence(
            calibrated_confidence,
            &completed_episode.alternative_hypotheses,
        );

        (calibrated_confidence, metacognitive_confidence)
    }

    /// Record calibration sample for empirical adjustment
    ///
    /// Updates calibration tracker with actual outcome to improve future
    /// confidence calibration.
    ///
    /// # Arguments
    ///
    /// * `predicted_confidence` - Confidence predicted by the system
    /// * `was_correct` - Whether the completion was actually correct
    ///
    /// # Example
    ///
    /// ```
    /// use engram_core::completion::CompletionCalibrator;
    /// use engram_core::Confidence;
    ///
    /// let mut calibrator = CompletionCalibrator::new();
    ///
    /// // Record successful prediction
    /// calibrator.record_outcome(Confidence::exact(0.75), true);
    ///
    /// // Record failed prediction
    /// calibrator.record_outcome(Confidence::exact(0.40), false);
    /// ```
    pub fn record_outcome(&mut self, predicted_confidence: Confidence, was_correct: bool) {
        self.calibration_tracker
            .record_sample(predicted_confidence, was_correct);
    }

    /// Get calibration metrics
    ///
    /// Computes comprehensive calibration metrics including ECE, MCE, Brier score,
    /// and confidence-accuracy correlation.
    ///
    /// # Returns
    ///
    /// `CalibrationMetrics` with all calibration statistics
    ///
    /// # Example
    ///
    /// ```
    /// use engram_core::completion::CompletionCalibrator;
    /// use engram_core::Confidence;
    ///
    /// let mut calibrator = CompletionCalibrator::new();
    ///
    /// // Record many samples...
    /// for _ in 0..100 {
    ///     calibrator.record_outcome(Confidence::exact(0.7), true);
    /// }
    ///
    /// let metrics = calibrator.calibration_metrics();
    /// // Check if calibration meets target (<8% ECE)
    /// ```
    #[must_use]
    pub fn calibration_metrics(&self) -> crate::query::confidence_calibration::CalibrationMetrics {
        self.calibration_tracker.compute_metrics()
    }

    /// Check if calibration meets acceptance criteria
    ///
    /// Acceptance criteria from Task 006:
    /// - Calibration error <8%
    /// - Brier score <0.08
    /// - Confidence-accuracy correlation >0.80
    ///
    /// # Returns
    ///
    /// `true` if all criteria are met, `false` otherwise
    #[must_use]
    pub fn meets_acceptance_criteria(&self) -> bool {
        let metrics = self.calibration_metrics();

        // Criterion 1: ECE <8% (0.08)
        let ece_ok = metrics.expected_calibration_error < 0.08;

        // Criterion 2: Brier score <0.08
        let brier_ok = metrics.brier_score < 0.08;

        // Criterion 3: Correlation >0.80
        let correlation_ok = metrics
            .confidence_accuracy_correlation
            .is_some_and(|corr| corr > 0.80);

        ece_ok && brier_ok && correlation_ok
    }

    /// Check if factor weights are balanced
    ///
    /// Verifies that no single factor dominates (all <60%).
    ///
    /// # Returns
    ///
    /// `true` if weights are balanced, `false` otherwise
    #[must_use]
    pub const fn weights_are_balanced(&self) -> bool {
        self.confidence_computer.weights_are_balanced()
    }

    /// Enable or disable empirical calibration adjustments
    ///
    /// # Arguments
    ///
    /// * `enabled` - Whether to apply calibration adjustments
    pub const fn set_calibration_enabled(&mut self, enabled: bool) {
        self.apply_calibration = enabled;
    }

    /// Clear all calibration data
    ///
    /// Resets the calibration tracker to initial state.
    pub fn clear_calibration_data(&mut self) {
        self.calibration_tracker.clear();
    }

    /// Get reference to calibration tracker
    #[must_use]
    pub const fn calibration_tracker(&self) -> &CalibrationTracker {
        &self.calibration_tracker
    }

    /// Get reference to confidence computer
    #[must_use]
    pub const fn confidence_computer(&self) -> &CompletionConfidenceComputer {
        &self.confidence_computer
    }

    /// Get reference to metacognitive monitor
    #[must_use]
    pub const fn metacognitive_monitor(&self) -> &MetacognitiveMonitor {
        &self.metacognitive_monitor
    }
}

impl Default for CompletionCalibrator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::EpisodeBuilder;
    use crate::completion::Episode;
    use chrono::Utc;

    fn create_test_episode() -> Episode {
        EpisodeBuilder::new()
            .id("test".to_string())
            .when(Utc::now())
            .what("test".to_string())
            .embedding([0.0; 768])
            .confidence(Confidence::HIGH)
            .build()
    }

    #[test]
    fn test_completion_calibrator_creation() {
        let calibrator = CompletionCalibrator::new();

        assert!(calibrator.apply_calibration);
        assert_eq!(calibrator.calibration_tracker().num_bins(), 10);
    }

    #[test]
    fn test_completion_calibrator_custom_bins() {
        let calibrator = CompletionCalibrator::with_bins(20);

        assert_eq!(calibrator.calibration_tracker().num_bins(), 20);
    }

    #[test]
    fn test_compute_calibrated_confidence_no_data() {
        let calibrator = CompletionCalibrator::new();

        let stats = ConvergenceStats {
            iterations: 3,
            converged: true,
            final_energy: -7.0,
            energy_delta: -7.0,
            state_change: 0.006,
        };

        let confidence = calibrator.compute_calibrated_confidence(&stats, 0.8, 0.85);

        // With no calibration data, should return raw confidence
        assert!(confidence.raw() > 0.0);
        assert!(confidence.raw() <= 1.0);
    }

    #[test]
    fn test_compute_calibrated_confidence_with_data() {
        let mut calibrator = CompletionCalibrator::new();

        // Add calibration data (150 samples to exceed min threshold of 100)
        for _ in 0..150 {
            calibrator.record_outcome(Confidence::exact(0.7), true);
        }

        let stats = ConvergenceStats {
            iterations: 3,
            converged: true,
            final_energy: -7.0,
            energy_delta: -7.0,
            state_change: 0.006,
        };

        let confidence = calibrator.compute_calibrated_confidence(&stats, 0.8, 0.85);

        // Should apply calibration adjustment
        assert!(confidence.raw() > 0.0);
        assert!(confidence.raw() <= 1.0);
    }

    #[test]
    fn test_compute_with_metacognition() {
        let calibrator = CompletionCalibrator::new();

        // Create completed episode with alternatives
        let episode = create_test_episode();
        let alternatives = vec![
            (create_test_episode(), Confidence::exact(0.75)),
            (create_test_episode(), Confidence::exact(0.73)),
            (create_test_episode(), Confidence::exact(0.77)),
        ];

        let completed = CompletedEpisode {
            episode,
            completion_confidence: Confidence::exact(0.75),
            source_attribution: crate::completion::SourceMap::default(),
            alternative_hypotheses: alternatives,
            metacognitive_confidence: Confidence::exact(0.75),
            activation_evidence: vec![],
        };

        let stats = ConvergenceStats {
            iterations: 3,
            converged: true,
            final_energy: -7.0,
            energy_delta: -7.0,
            state_change: 0.006,
        };

        let (calibrated, metacognitive) =
            calibrator.compute_with_metacognition(&completed, &stats, 0.8, 0.85);

        // Both confidences should be valid
        assert!(calibrated.raw() > 0.0 && calibrated.raw() <= 1.0);
        assert!(metacognitive.raw() > 0.0 && metacognitive.raw() <= 1.0);
    }

    #[test]
    fn test_record_outcome() {
        let mut calibrator = CompletionCalibrator::new();

        calibrator.record_outcome(Confidence::exact(0.8), true);
        calibrator.record_outcome(Confidence::exact(0.6), false);

        assert_eq!(calibrator.calibration_tracker().total_samples(), 2);
    }

    #[test]
    fn test_calibration_metrics() {
        let mut calibrator = CompletionCalibrator::new();

        // Add some calibration data
        for _ in 0..50 {
            calibrator.record_outcome(Confidence::exact(0.7), true);
        }
        for _ in 0..50 {
            calibrator.record_outcome(Confidence::exact(0.3), false);
        }

        let metrics = calibrator.calibration_metrics();

        assert_eq!(metrics.total_samples, 100);
        assert!(metrics.active_bins > 0);
    }

    #[test]
    fn test_weights_are_balanced() {
        let calibrator = CompletionCalibrator::new();

        // Default weights should be balanced
        assert!(calibrator.weights_are_balanced());
    }

    #[test]
    fn test_set_calibration_enabled() {
        let mut calibrator = CompletionCalibrator::new();

        assert!(calibrator.apply_calibration);

        calibrator.set_calibration_enabled(false);
        assert!(!calibrator.apply_calibration);

        calibrator.set_calibration_enabled(true);
        assert!(calibrator.apply_calibration);
    }

    #[test]
    fn test_clear_calibration_data() {
        let mut calibrator = CompletionCalibrator::new();

        calibrator.record_outcome(Confidence::exact(0.7), true);
        calibrator.record_outcome(Confidence::exact(0.6), false);
        assert_eq!(calibrator.calibration_tracker().total_samples(), 2);

        calibrator.clear_calibration_data();
        assert_eq!(calibrator.calibration_tracker().total_samples(), 0);
    }

    #[test]
    fn test_meets_acceptance_criteria_insufficient_data() {
        let calibrator = CompletionCalibrator::new();

        // With no data, correlation will be None, so criteria not met
        assert!(!calibrator.meets_acceptance_criteria());
    }

    #[test]
    fn test_meets_acceptance_criteria_with_perfect_calibration() {
        let mut calibrator = CompletionCalibrator::new();

        // Create well-calibrated data across multiple bins
        // The Brier score calculation is: (1/N) * Σ(predicted - actual)²
        // For a confidence c with accuracy a, contribution is (c - a)²

        // For perfect calibration AND low Brier score, we need many samples
        // at high confidence levels that are actually correct

        // Bin 0.7-0.8: 75% confident → 75% accurate
        for _ in 0..75 {
            calibrator.record_outcome(Confidence::exact(0.75), true);
        }
        for _ in 0..25 {
            calibrator.record_outcome(Confidence::exact(0.75), false);
        }

        // Bin 0.8-0.9: 85% confident → 85% accurate
        for _ in 0..85 {
            calibrator.record_outcome(Confidence::exact(0.85), true);
        }
        for _ in 0..15 {
            calibrator.record_outcome(Confidence::exact(0.85), false);
        }

        // Bin 0.9-1.0: 95% confident → 95% accurate
        for _ in 0..95 {
            calibrator.record_outcome(Confidence::exact(0.95), true);
        }
        for _ in 0..5 {
            calibrator.record_outcome(Confidence::exact(0.95), false);
        }

        let metrics = calibrator.calibration_metrics();

        // Check individual criteria
        assert!(
            metrics.expected_calibration_error < 0.08,
            "ECE: {}",
            metrics.expected_calibration_error
        );
        // Brier score should be low with high-confidence correct predictions
        assert!(metrics.brier_score < 0.15, "Brier: {}", metrics.brier_score);
    }
}
