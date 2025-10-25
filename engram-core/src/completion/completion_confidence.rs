//! Multi-factor confidence computation for pattern completion
//!
//! Implements confidence assessment combining multiple cues from Koriat's
//! Cue-Utilization Framework (1997):
//! - Intrinsic cues: Pattern strength, cue overlap
//! - Extrinsic cues: Number of source episodes, pattern age
//! - Mnemonic cues: Convergence speed (fluency), consensus (ease)
//!
//! Confidence aggregates four factors with empirically-validated weights:
//! - CA3 convergence speed (weight: 0.3) - faster convergence = higher confidence
//! - Energy reduction/attractor depth (weight: 0.25) - deeper basin = higher confidence
//! - Field consensus from neighbor agreement (weight: 0.25) - more agreement = higher confidence
//! - Plausibility score from semantic coherence (weight: 0.2) - more plausible = higher confidence
//!
//! No single factor dominates (all <60%) to prevent overreliance on any single cue.

use crate::Confidence;
use crate::completion::{ConvergenceStats, Episode};
use std::time::Instant;

/// Multi-factor confidence computer for pattern completion
///
/// Combines convergence speed, energy reduction, field consensus, and plausibility
/// into a single calibrated confidence score. Factor weights are tuned to prevent
/// over-reliance on any single cue (Koriat, 1997).
pub struct CompletionConfidenceComputer {
    /// Convergence speed weight (default: 0.3)
    /// Measures retrieval fluency - faster convergence indicates stronger memory
    convergence_weight: f32,

    /// Energy reduction weight (default: 0.25)
    /// Measures attractor basin depth - deeper basin indicates more stable pattern
    energy_weight: f32,

    /// Field consensus weight (default: 0.25)
    /// Measures neighbor agreement - higher agreement indicates reliable reconstruction
    consensus_weight: f32,

    /// Plausibility weight (default: 0.2)
    /// Measures semantic coherence - more plausible patterns are more likely correct
    plausibility_weight: f32,

    /// Maximum iterations for normalization (default: 7, theta rhythm constraint)
    max_iterations: usize,
}

impl CompletionConfidenceComputer {
    /// Create new confidence computer with default weights
    ///
    /// Default weights are:
    /// - Convergence: 0.3 (mnemonic fluency)
    /// - Energy: 0.25 (attractor stability)
    /// - Consensus: 0.25 (social validation analog)
    /// - Plausibility: 0.2 (semantic coherence)
    ///
    /// # Example
    ///
    /// ```
    /// use engram_core::completion::CompletionConfidenceComputer;
    ///
    /// let computer = CompletionConfidenceComputer::new();
    /// assert!((computer.convergence_weight() - 0.3).abs() < 1e-6);
    /// ```
    #[must_use]
    pub const fn new() -> Self {
        Self {
            convergence_weight: 0.3,
            energy_weight: 0.25,
            consensus_weight: 0.25,
            plausibility_weight: 0.2,
            max_iterations: 7,
        }
    }

    /// Create confidence computer with custom weights
    ///
    /// # Arguments
    ///
    /// * `convergence_weight` - Weight for convergence speed factor (0.0-1.0)
    /// * `energy_weight` - Weight for energy reduction factor (0.0-1.0)
    /// * `consensus_weight` - Weight for field consensus factor (0.0-1.0)
    /// * `plausibility_weight` - Weight for plausibility score factor (0.0-1.0)
    ///
    /// # Panics
    ///
    /// Panics in debug mode if weights don't sum to approximately 1.0
    ///
    /// # Example
    ///
    /// ```
    /// use engram_core::completion::CompletionConfidenceComputer;
    ///
    /// // Custom weights (must sum to 1.0)
    /// let computer = CompletionConfidenceComputer::with_weights(0.4, 0.3, 0.2, 0.1);
    /// assert!((computer.convergence_weight() - 0.4).abs() < 1e-6);
    /// ```
    #[must_use]
    pub const fn with_weights(
        convergence_weight: f32,
        energy_weight: f32,
        consensus_weight: f32,
        plausibility_weight: f32,
    ) -> Self {
        // Verify weights sum to ~1.0 (const context prevents complex assertions)
        debug_assert!(
            (convergence_weight + energy_weight + consensus_weight + plausibility_weight - 1.0)
                .abs()
                < 0.01,
            "Weights must sum to approximately 1.0"
        );

        Self {
            convergence_weight,
            energy_weight,
            consensus_weight,
            plausibility_weight,
            max_iterations: 7,
        }
    }

    /// Compute multi-factor completion confidence
    ///
    /// Aggregates four factors using weighted combination:
    /// 1. Convergence factor: 1.0 - (iterations / max_iterations)
    /// 2. Energy factor: |energy_delta| / 10.0 (normalized)
    /// 3. Field consensus: weighted_agreement_ratio from neighbors
    /// 4. Plausibility: HNSW neighborhood consistency
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
    /// ```no_run
    /// use engram_core::completion::{CompletionConfidenceComputer, ConvergenceStats};
    /// use engram_core::Confidence;
    ///
    /// let computer = CompletionConfidenceComputer::new();
    ///
    /// // Fast convergence (3 iterations), deep attractor (energy delta = -8.0)
    /// let stats = ConvergenceStats {
    ///     iterations: 3,
    ///     converged: true,
    ///     final_energy: -8.0,
    ///     energy_delta: -8.0,
    ///     state_change: 0.005,
    /// };
    ///
    /// let confidence = computer.compute_completion_confidence(
    ///     &stats,
    ///     0.85, // High field consensus
    ///     0.90, // High plausibility
    /// );
    ///
    /// // High confidence expected (all factors favorable)
    /// println!("Confidence: {:.2}", confidence.raw());
    /// ```
    #[must_use]
    pub fn compute_completion_confidence(
        &self,
        convergence_stats: &ConvergenceStats,
        field_consensus: f32,
        plausibility_score: f32,
    ) -> Confidence {
        // If convergence failed, return low confidence immediately
        if !convergence_stats.converged {
            return Confidence::exact(0.3);
        }

        // Factor 1: Convergence speed (faster = higher confidence)
        let convergence_factor = self.convergence_factor(convergence_stats.iterations);

        // Factor 2: Energy reduction (deeper attractor = higher confidence)
        let energy_factor = Self::energy_factor(convergence_stats.energy_delta);

        // Factor 3: Field consensus (higher agreement = higher confidence)
        let consensus_factor = field_consensus.clamp(0.0, 1.0);

        // Factor 4: Plausibility (more plausible = higher confidence)
        let plausibility_factor = plausibility_score.clamp(0.0, 1.0);

        // Weighted combination
        let confidence = consensus_factor.mul_add(
            self.consensus_weight,
            plausibility_factor.mul_add(
                self.plausibility_weight,
                convergence_factor
                    .mul_add(self.convergence_weight, energy_factor * self.energy_weight),
            ),
        );

        Confidence::exact(confidence.clamp(0.0, 1.0))
    }

    /// Compute convergence speed factor
    ///
    /// Faster convergence (fewer iterations) indicates stronger memory trace
    /// and higher retrieval fluency (mnemonic cue from Koriat, 1997).
    ///
    /// Formula: 1.0 - (iterations / max_iterations)
    ///
    /// # Arguments
    ///
    /// * `iterations` - Number of CA3 iterations to convergence
    ///
    /// # Returns
    ///
    /// Convergence factor [0.0-1.0], where 1.0 = instant convergence, 0.0 = max iterations
    #[must_use]
    pub fn convergence_factor(&self, iterations: usize) -> f32 {
        1.0 - (iterations as f32 / self.max_iterations as f32)
    }

    /// Compute energy reduction factor
    ///
    /// Deeper attractor basin (larger energy reduction) indicates more stable
    /// pattern with higher confidence. Normalizes by typical energy range of 10.0.
    ///
    /// Formula: |energy_delta| / 10.0, clamped to [0.0, 1.0]
    ///
    /// # Arguments
    ///
    /// * `energy_delta` - Energy reduction during convergence (typically negative)
    ///
    /// # Returns
    ///
    /// Energy factor [0.0-1.0], where 1.0 = deepest basin, 0.0 = no energy change
    #[must_use]
    pub fn energy_factor(energy_delta: f32) -> f32 {
        (energy_delta.abs() / 10.0).min(1.0)
    }

    /// Compute timed confidence with performance tracking
    ///
    /// Same as `compute_completion_confidence` but returns computation time.
    ///
    /// # Returns
    ///
    /// Tuple of (confidence, computation_duration_micros)
    #[must_use]
    pub fn compute_timed(
        &self,
        convergence_stats: &ConvergenceStats,
        field_consensus: f32,
        plausibility_score: f32,
    ) -> (Confidence, u128) {
        let start = Instant::now();
        let confidence = self.compute_completion_confidence(
            convergence_stats,
            field_consensus,
            plausibility_score,
        );
        let elapsed = start.elapsed().as_micros();
        (confidence, elapsed)
    }

    /// Check if factor weights are balanced (no single factor >60%)
    ///
    /// Verifies that no single cue dominates confidence computation,
    /// preventing overreliance on any one signal.
    ///
    /// # Returns
    ///
    /// `true` if all weights ≤ 0.6, `false` otherwise
    #[must_use]
    pub const fn weights_are_balanced(&self) -> bool {
        self.convergence_weight <= 0.6
            && self.energy_weight <= 0.6
            && self.consensus_weight <= 0.6
            && self.plausibility_weight <= 0.6
    }

    /// Get convergence weight
    #[must_use]
    pub const fn convergence_weight(&self) -> f32 {
        self.convergence_weight
    }

    /// Get energy weight
    #[must_use]
    pub const fn energy_weight(&self) -> f32 {
        self.energy_weight
    }

    /// Get consensus weight
    #[must_use]
    pub const fn consensus_weight(&self) -> f32 {
        self.consensus_weight
    }

    /// Get plausibility weight
    #[must_use]
    pub const fn plausibility_weight(&self) -> f32 {
        self.plausibility_weight
    }
}

impl Default for CompletionConfidenceComputer {
    fn default() -> Self {
        Self::new()
    }
}

/// Metacognitive monitor for confidence-in-confidence assessment
///
/// Implements Fleming & Dolan (2012) metacognitive confidence framework.
/// Assesses reliability of completion confidence by checking internal
/// consistency across alternative hypotheses.
pub struct MetacognitiveMonitor {
    /// Internal consistency threshold for high metacognitive confidence (default: 0.8)
    consistency_threshold: f32,

    /// Minimum number of alternatives required for assessment (default: 2)
    min_alternatives: usize,
}

impl MetacognitiveMonitor {
    /// Create new metacognitive monitor with default parameters
    ///
    /// # Example
    ///
    /// ```
    /// use engram_core::completion::MetacognitiveMonitor;
    ///
    /// let monitor = MetacognitiveMonitor::new();
    /// assert!((monitor.consistency_threshold() - 0.8).abs() < 1e-6);
    /// ```
    #[must_use]
    pub const fn new() -> Self {
        Self {
            consistency_threshold: 0.8,
            min_alternatives: 2,
        }
    }

    /// Create monitor with custom parameters
    ///
    /// # Arguments
    ///
    /// * `consistency_threshold` - Minimum consistency for high metacognitive confidence
    /// * `min_alternatives` - Minimum alternatives required for assessment
    #[must_use]
    pub const fn with_params(consistency_threshold: f32, min_alternatives: usize) -> Self {
        Self {
            consistency_threshold,
            min_alternatives,
        }
    }

    /// Compute metacognitive confidence (confidence in our confidence)
    ///
    /// Assesses how reliable the primary completion confidence is by checking
    /// internal consistency across alternative hypotheses. High consistency
    /// across alternatives increases trust in the completion confidence.
    ///
    /// # Arguments
    ///
    /// * `completion_confidence` - Primary completion confidence
    /// * `alternative_hypotheses` - Alternative completions with their confidences
    ///
    /// # Returns
    ///
    /// Metacognitive confidence score [0.0-1.0]
    ///
    /// # Example
    ///
    /// ```
    /// use engram_core::completion::MetacognitiveMonitor;
    /// use engram_core::Confidence;
    ///
    /// let monitor = MetacognitiveMonitor::new();
    ///
    /// // Alternatives would be created from actual Episode instances
    /// // For documentation, showing just the concept
    /// assert_eq!(monitor.min_alternatives(), 2);
    /// ```
    #[must_use]
    pub fn compute_metacognitive_confidence(
        &self,
        completion_confidence: Confidence,
        alternative_hypotheses: &[(Episode, Confidence)],
    ) -> Confidence {
        // Need at least min_alternatives for meaningful assessment
        if alternative_hypotheses.len() < self.min_alternatives {
            // Low metacognitive confidence - insufficient alternatives
            return Confidence::exact(0.5);
        }

        // Measure consistency across alternatives
        let consistency = Self::check_consistency(alternative_hypotheses);

        // Higher consistency → trust completion_confidence more
        // Lower consistency → uncertain about completion quality
        let base_metacognitive = if consistency >= self.consistency_threshold {
            // High consistency - trust the primary confidence
            completion_confidence.raw() * 0.9 + 0.1 // Slight boost
        } else {
            // Low consistency - reduce trust
            completion_confidence.raw() * consistency
        };

        Confidence::exact(base_metacognitive.clamp(0.0, 1.0))
    }

    /// Check internal consistency of alternative hypotheses
    ///
    /// Measures how much alternatives agree with each other. High consistency
    /// indicates reliable pattern completion; low consistency suggests uncertainty.
    ///
    /// # Arguments
    ///
    /// * `alternatives` - Alternative completions with confidences
    ///
    /// # Returns
    ///
    /// Consistency score [0.0-1.0], where 1.0 = perfect agreement, 0.0 = complete disagreement
    #[must_use]
    pub fn check_consistency(alternatives: &[(Episode, Confidence)]) -> f32 {
        if alternatives.len() < 2 {
            return 1.0; // Single alternative = trivially consistent
        }

        // Extract confidence values
        let confidences: Vec<f32> = alternatives.iter().map(|(_, conf)| conf.raw()).collect();

        // Compute mean confidence
        let mean = confidences.iter().sum::<f32>() / confidences.len() as f32;

        // Compute coefficient of variation (CV = std_dev / mean)
        // Lower CV = higher consistency
        let variance = confidences
            .iter()
            .map(|&conf| (conf - mean).powi(2))
            .sum::<f32>()
            / confidences.len() as f32;

        let std_dev = variance.sqrt();

        // Avoid division by zero
        if mean < 1e-6 {
            return 0.5; // Uncertain consistency for near-zero confidences
        }

        let cv = std_dev / mean;

        // Convert CV to consistency score: lower CV = higher consistency
        // CV of 0.0 → consistency 1.0
        // CV of 0.5 → consistency 0.5
        // CV of 1.0+ → consistency 0.0
        (1.0 - cv.min(1.0)).max(0.0)
    }

    /// Get consistency threshold
    #[must_use]
    pub const fn consistency_threshold(&self) -> f32 {
        self.consistency_threshold
    }

    /// Get minimum alternatives requirement
    #[must_use]
    pub const fn min_alternatives(&self) -> usize {
        self.min_alternatives
    }
}

impl Default for MetacognitiveMonitor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::EpisodeBuilder;
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
    fn test_completion_confidence_computer_creation() {
        let computer = CompletionConfidenceComputer::new();

        assert!((computer.convergence_weight() - 0.3).abs() < 1e-6);
        assert!((computer.energy_weight() - 0.25).abs() < 1e-6);
        assert!((computer.consensus_weight() - 0.25).abs() < 1e-6);
        assert!((computer.plausibility_weight() - 0.2).abs() < 1e-6);
    }

    #[test]
    fn test_completion_confidence_computer_custom_weights() {
        let computer = CompletionConfidenceComputer::with_weights(0.4, 0.3, 0.2, 0.1);

        assert!((computer.convergence_weight() - 0.4).abs() < 1e-6);
        assert!((computer.energy_weight() - 0.3).abs() < 1e-6);
        assert!((computer.consensus_weight() - 0.2).abs() < 1e-6);
        assert!((computer.plausibility_weight() - 0.1).abs() < 1e-6);
    }

    #[test]
    fn test_weights_are_balanced() {
        let computer = CompletionConfidenceComputer::new();
        assert!(computer.weights_are_balanced());

        // Unbalanced weights (convergence dominates)
        let unbalanced = CompletionConfidenceComputer::with_weights(0.7, 0.1, 0.1, 0.1);
        assert!(!unbalanced.weights_are_balanced());
    }

    #[test]
    fn test_convergence_factor() {
        let computer = CompletionConfidenceComputer::new();

        // Fast convergence (1 iteration)
        assert!((computer.convergence_factor(1) - (1.0 - 1.0 / 7.0)).abs() < 1e-6);

        // Moderate convergence (4 iterations)
        assert!((computer.convergence_factor(4) - (1.0 - 4.0 / 7.0)).abs() < 1e-6);

        // Slow convergence (7 iterations, max)
        assert!((computer.convergence_factor(7) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_energy_factor() {
        // Deep attractor (energy delta = -10.0)
        assert!((CompletionConfidenceComputer::energy_factor(-10.0) - 1.0).abs() < 1e-6);

        // Moderate attractor (energy delta = -5.0)
        assert!((CompletionConfidenceComputer::energy_factor(-5.0) - 0.5).abs() < 1e-6);

        // Shallow attractor (energy delta = -2.0)
        assert!((CompletionConfidenceComputer::energy_factor(-2.0) - 0.2).abs() < 1e-6);

        // Very deep attractor (energy delta = -20.0, clamped to 1.0)
        assert!((CompletionConfidenceComputer::energy_factor(-20.0) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_compute_completion_confidence_high() {
        let computer = CompletionConfidenceComputer::new();

        // Fast convergence, deep attractor, high consensus, high plausibility
        let stats = ConvergenceStats {
            iterations: 2,
            converged: true,
            final_energy: -8.0,
            energy_delta: -8.0,
            state_change: 0.005,
        };

        let confidence = computer.compute_completion_confidence(&stats, 0.9, 0.85);

        // Should be high confidence
        assert!(confidence.raw() > 0.7);
        assert!(confidence.raw() <= 1.0);
    }

    #[test]
    fn test_compute_completion_confidence_moderate() {
        let computer = CompletionConfidenceComputer::new();

        // Moderate convergence, moderate attractor, moderate consensus, moderate plausibility
        let stats = ConvergenceStats {
            iterations: 4,
            converged: true,
            final_energy: -5.0,
            energy_delta: -5.0,
            state_change: 0.01,
        };

        let confidence = computer.compute_completion_confidence(&stats, 0.6, 0.55);

        // Should be moderate confidence
        assert!(confidence.raw() > 0.4);
        assert!(confidence.raw() < 0.7);
    }

    #[test]
    fn test_compute_completion_confidence_low() {
        let computer = CompletionConfidenceComputer::new();

        // Slow convergence, shallow attractor, low consensus, low plausibility
        let stats = ConvergenceStats {
            iterations: 7,
            converged: true,
            final_energy: -2.0,
            energy_delta: -2.0,
            state_change: 0.02,
        };

        let confidence = computer.compute_completion_confidence(&stats, 0.3, 0.25);

        // Should be low confidence
        assert!(confidence.raw() < 0.4);
        assert!(confidence.raw() >= 0.0);
    }

    #[test]
    fn test_compute_completion_confidence_failed_convergence() {
        let computer = CompletionConfidenceComputer::new();

        // Convergence failed
        let stats = ConvergenceStats {
            iterations: 7,
            converged: false,
            final_energy: 0.0,
            energy_delta: 0.0,
            state_change: 0.5,
        };

        let confidence = computer.compute_completion_confidence(&stats, 0.8, 0.8);

        // Should return fixed low confidence (0.3)
        assert!((confidence.raw() - 0.3).abs() < 1e-6);
    }

    #[test]
    fn test_compute_timed_performance() {
        let computer = CompletionConfidenceComputer::new();

        let stats = ConvergenceStats {
            iterations: 3,
            converged: true,
            final_energy: -6.0,
            energy_delta: -6.0,
            state_change: 0.008,
        };

        let (confidence, elapsed_micros) = computer.compute_timed(&stats, 0.75, 0.70);

        // Should compute confidence correctly
        assert!(confidence.raw() > 0.5);

        // Should be very fast (<200μs acceptance criterion)
        assert!(elapsed_micros < 200);
    }

    #[test]
    fn test_metacognitive_monitor_creation() {
        let monitor = MetacognitiveMonitor::new();

        assert!((monitor.consistency_threshold() - 0.8).abs() < 1e-6);
        assert_eq!(monitor.min_alternatives(), 2);
    }

    #[test]
    fn test_metacognitive_monitor_custom_params() {
        let monitor = MetacognitiveMonitor::with_params(0.75, 3);

        assert!((monitor.consistency_threshold() - 0.75).abs() < 1e-6);
        assert_eq!(monitor.min_alternatives(), 3);
    }

    #[test]
    fn test_check_consistency_high() {
        // Alternatives with very similar confidences (high consistency)
        let alternatives = vec![
            (create_test_episode(), Confidence::exact(0.75)),
            (create_test_episode(), Confidence::exact(0.73)),
            (create_test_episode(), Confidence::exact(0.77)),
            (create_test_episode(), Confidence::exact(0.74)),
        ];

        let consistency = MetacognitiveMonitor::check_consistency(&alternatives);

        // Should have high consistency (low coefficient of variation)
        assert!(consistency > 0.9);
    }

    #[test]
    fn test_check_consistency_moderate() {
        // Alternatives with moderate variation
        let alternatives = vec![
            (create_test_episode(), Confidence::exact(0.8)),
            (create_test_episode(), Confidence::exact(0.6)),
            (create_test_episode(), Confidence::exact(0.7)),
            (create_test_episode(), Confidence::exact(0.75)),
        ];

        let consistency = MetacognitiveMonitor::check_consistency(&alternatives);

        // Should have moderate consistency
        assert!(consistency > 0.5);
        assert!(consistency < 0.9);
    }

    #[test]
    fn test_check_consistency_low() {
        // Alternatives with high variation (low consistency)
        let alternatives = vec![
            (create_test_episode(), Confidence::exact(0.9)),
            (create_test_episode(), Confidence::exact(0.3)),
            (create_test_episode(), Confidence::exact(0.5)),
            (create_test_episode(), Confidence::exact(0.2)),
        ];

        let consistency = MetacognitiveMonitor::check_consistency(&alternatives);

        // Should have low consistency (high coefficient of variation)
        assert!(consistency < 0.6);
    }

    #[test]
    fn test_check_consistency_single_alternative() {
        // Single alternative (trivially consistent)
        let alternatives = vec![(create_test_episode(), Confidence::exact(0.7))];

        let consistency = MetacognitiveMonitor::check_consistency(&alternatives);

        // Should be perfect consistency
        assert!((consistency - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_compute_metacognitive_confidence_high_consistency() {
        let monitor = MetacognitiveMonitor::new();

        // High consistency across alternatives
        let alternatives = vec![
            (create_test_episode(), Confidence::exact(0.76)),
            (create_test_episode(), Confidence::exact(0.74)),
            (create_test_episode(), Confidence::exact(0.75)),
        ];

        let meta_confidence =
            monitor.compute_metacognitive_confidence(Confidence::exact(0.75), &alternatives);

        // Should have high metacognitive confidence (consistency > threshold)
        assert!(meta_confidence.raw() > 0.7);
    }

    #[test]
    fn test_compute_metacognitive_confidence_low_consistency() {
        let monitor = MetacognitiveMonitor::new();

        // Low consistency across alternatives
        let alternatives = vec![
            (create_test_episode(), Confidence::exact(0.9)),
            (create_test_episode(), Confidence::exact(0.4)),
            (create_test_episode(), Confidence::exact(0.6)),
        ];

        let meta_confidence =
            monitor.compute_metacognitive_confidence(Confidence::exact(0.7), &alternatives);

        // Should have moderate to low metacognitive confidence
        assert!(meta_confidence.raw() < 0.7);
    }

    #[test]
    fn test_compute_metacognitive_confidence_insufficient_alternatives() {
        let monitor = MetacognitiveMonitor::new();

        // Only one alternative (below min_alternatives = 2)
        let alternatives = vec![(create_test_episode(), Confidence::exact(0.8))];

        let meta_confidence =
            monitor.compute_metacognitive_confidence(Confidence::exact(0.8), &alternatives);

        // Should return moderate confidence (0.5) for insufficient data
        assert!((meta_confidence.raw() - 0.5).abs() < 1e-6);
    }
}
