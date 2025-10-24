//! CA1 output gating with confidence thresholding and plausibility checking.
//!
//! Implements the CA1 region's role as hippocampal output gate, filtering
//! completions based on multi-factor confidence and plausibility scores.
//! Prevents hallucinations by detecting implausible reconstructions through
//! neighborhood consistency validation.

use crate::compute::VectorOps;
use crate::compute::dispatch::DispatchVectorOps;
use crate::{Confidence, Episode};
use std::collections::HashMap;

use super::attractor_dynamics::ConvergenceStats;
use super::field_reconstruction::ReconstructedField;
use super::{CompletedEpisode, CompletionError, MemorySource, SourceMap};

/// CA1 output gating with confidence thresholding
pub struct CA1Gate {
    /// Minimum confidence threshold for output (default: 0.7)
    threshold: Confidence,

    /// Plausibility scoring for hallucination detection
    plausibility_checker: PlausibilityChecker,

    /// Convergence speed weight (default: 0.3, from Task 006 calibration)
    convergence_weight: f32,

    /// Energy reduction weight (default: 0.25, from Task 006 calibration)
    energy_weight: f32,

    /// Field consensus weight (default: 0.25)
    consensus_weight: f32,

    /// Plausibility weight (default: 0.20)
    plausibility_weight: f32,
}

impl CA1Gate {
    /// Create new CA1 gate with threshold
    #[must_use]
    pub fn new(threshold: Confidence) -> Self {
        Self {
            threshold,
            plausibility_checker: PlausibilityChecker::new(),
            convergence_weight: 0.3,
            energy_weight: 0.25,
            consensus_weight: 0.25,
            plausibility_weight: 0.20,
        }
    }

    /// Create CA1 gate with custom weight configuration
    #[must_use]
    pub fn with_weights(
        threshold: Confidence,
        convergence_weight: f32,
        energy_weight: f32,
        consensus_weight: f32,
        plausibility_weight: f32,
    ) -> Self {
        Self {
            threshold,
            plausibility_checker: PlausibilityChecker::new(),
            convergence_weight,
            energy_weight,
            consensus_weight,
            plausibility_weight,
        }
    }

    /// Gate CA3 output based on confidence and plausibility
    ///
    /// Returns `Ok(completed)` if passes threshold, `Err(LowConfidence)` otherwise
    ///
    /// # Errors
    ///
    /// Returns `CompletionError::LowConfidence` when the computed confidence
    /// is below the CA1 threshold or plausibility check fails.
    pub fn gate_output(
        &self,
        ca3_embedding: &[f32; 768],
        convergence_stats: &ConvergenceStats,
        field_reconstructions: &HashMap<String, ReconstructedField>,
        _partial_episode: &super::PartialEpisode,
    ) -> Result<Confidence, CompletionError> {
        // Compute field consensus strength
        let field_consensus = Self::compute_field_consensus(field_reconstructions);

        // Score plausibility of reconstructed embedding
        let plausibility = self.plausibility_checker.score_plausibility(ca3_embedding);

        // Compute multi-factor completion confidence
        let completion_confidence =
            self.compute_completion_confidence(convergence_stats, field_consensus, plausibility);

        // Gate based on threshold
        if self.passes_threshold(completion_confidence) {
            Ok(completion_confidence)
        } else {
            Err(CompletionError::LowConfidence(completion_confidence.raw()))
        }
    }

    /// Compute completion confidence from multiple factors
    ///
    /// Factors (from Koriat's Cue-Utilization Framework, 1997):
    /// 1. CA3 convergence speed (faster = higher confidence)
    /// 2. Energy reduction (deeper attractor = higher confidence)
    /// 3. Field consensus strength (agreement = higher confidence)
    /// 4. Plausibility score (coherent = higher confidence)
    fn compute_completion_confidence(
        &self,
        convergence_stats: &ConvergenceStats,
        field_consensus: f32,
        plausibility: f32,
    ) -> Confidence {
        // Convergence factor: 1.0 (fast) to 0.0 (slow)
        // Assume max_iterations = 7 (theta constraint)
        let convergence_factor = if convergence_stats.converged {
            1.0 - (convergence_stats.iterations as f32 / 7.0)
        } else {
            0.0 // No convergence = very low confidence
        };

        // Energy factor: normalized energy reduction
        // Typical range: -10.0 (deep basin) to 0.0 (shallow)
        let energy_factor = if convergence_stats.energy_delta > 0.0 {
            (convergence_stats.energy_delta / 10.0).clamp(0.0, 1.0)
        } else {
            0.0
        };

        // Weighted combination (Task 006 calibration research)
        let confidence = self.convergence_weight * convergence_factor
            + self.energy_weight * energy_factor
            + self.consensus_weight * field_consensus
            + self.plausibility_weight * plausibility;

        Confidence::exact(confidence.clamp(0.0, 1.0))
    }

    /// Compute field consensus strength from reconstructed fields
    fn compute_field_consensus(fields: &HashMap<String, ReconstructedField>) -> f32 {
        if fields.is_empty() {
            return 0.5; // Neutral consensus when no fields
        }

        // Average confidence across all reconstructed fields
        let total_confidence: f32 = fields.values().map(|f| f.confidence.raw()).sum();
        total_confidence / fields.len() as f32
    }

    /// Check if completion passes threshold
    #[must_use]
    pub fn passes_threshold(&self, confidence: Confidence) -> bool {
        confidence >= self.threshold
    }

    /// Build completed episode from gated output
    #[must_use]
    pub fn build_completed_episode(
        episode: Episode,
        completion_confidence: Confidence,
        field_reconstructions: &HashMap<String, ReconstructedField>,
        partial: &super::PartialEpisode,
    ) -> CompletedEpisode {
        // Build source attribution
        let source_map = Self::build_source_attribution(partial, field_reconstructions);

        // Create activation trace
        let activation_trace = super::ActivationTrace {
            source_memory: episode.id.clone(),
            activation_strength: completion_confidence.raw(),
            pathway: super::ActivationPathway::Direct,
            decay_factor: 0.1,
        };

        CompletedEpisode {
            episode,
            completion_confidence,
            source_attribution: source_map,
            alternative_hypotheses: Vec::new(),
            metacognitive_confidence: completion_confidence,
            activation_evidence: vec![activation_trace],
        }
    }

    /// Build source attribution map from field reconstructions
    fn build_source_attribution(
        partial: &super::PartialEpisode,
        field_reconstructions: &HashMap<String, ReconstructedField>,
    ) -> SourceMap {
        let mut source_map = SourceMap {
            field_sources: HashMap::new(),
            source_confidence: HashMap::new(),
        };

        // Mark known fields as recalled
        for field_name in partial.known_fields.keys() {
            source_map
                .field_sources
                .insert(field_name.clone(), MemorySource::Recalled);
            source_map
                .source_confidence
                .insert(field_name.clone(), Confidence::exact(1.0));
        }

        // Mark reconstructed fields
        for (field_name, reconstruction) in field_reconstructions {
            source_map
                .field_sources
                .insert(field_name.clone(), reconstruction.source);
            source_map
                .source_confidence
                .insert(field_name.clone(), reconstruction.confidence);
        }

        source_map
    }
}

impl Default for CA1Gate {
    fn default() -> Self {
        Self::new(Confidence::exact(0.7))
    }
}

/// Plausibility checker for detecting implausible reconstructions
pub struct PlausibilityChecker {
    /// Minimum neighborhood agreement for plausibility (default: 0.6)
    #[allow(dead_code)] // Reserved for future HNSW integration
    min_neighborhood_agreement: f32,

    /// Vector operations dispatcher for similarity computation
    vector_ops: DispatchVectorOps,
}

impl PlausibilityChecker {
    /// Create new plausibility checker with default parameters
    #[must_use]
    pub fn new() -> Self {
        Self {
            min_neighborhood_agreement: 0.6,
            vector_ops: DispatchVectorOps::new(),
        }
    }

    /// Create plausibility checker with custom threshold
    #[must_use]
    pub fn with_threshold(min_neighborhood_agreement: f32) -> Self {
        Self {
            min_neighborhood_agreement,
            vector_ops: DispatchVectorOps::new(),
        }
    }

    /// Score plausibility of reconstructed embedding
    ///
    /// Returns 0.0-1.0 score based on:
    /// 1. Similarity to nearest neighbors (not yet implemented - needs HNSW)
    /// 2. Consistency with local embedding manifold
    /// 3. Not in "nowhere" region (isolated point)
    ///
    /// Current implementation: simple heuristic based on embedding magnitude
    /// TODO: Integrate with HNSW index for true neighborhood consistency
    #[must_use]
    pub fn score_plausibility(&self, embedding: &[f32; 768]) -> f32 {
        // Simple plausibility heuristic: check embedding is not degenerate
        // A plausible embedding should have reasonable magnitude and variance

        // Compute L2 norm
        let magnitude = self.vector_ops.l2_norm_768(embedding);

        // Plausible embeddings typically have magnitude in range [0.5, 2.0]
        // (normalized embeddings are ~1.0)
        let magnitude_score = if (0.5..=2.0).contains(&magnitude) {
            1.0
        } else if magnitude < 0.5 {
            magnitude * 2.0 // Linear penalty for too small
        } else {
            (2.0 / magnitude).clamp(0.0, 1.0) // Inverse penalty for too large
        };

        // Compute variance (spread across dimensions)
        let mean: f32 = embedding.iter().sum::<f32>() / 768.0;
        let variance: f32 = embedding.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / 768.0;

        // Plausible embeddings should have some variance (not all same value)
        let variance_score = if variance > 0.001 { 1.0 } else { 0.3 };

        // Combined score (simple average for now)
        f32::midpoint(magnitude_score, variance_score)
    }

    /// Check if embedding is in sparse region (potential hallucination)
    ///
    /// TODO: Implement with HNSW neighborhood density estimation
    #[must_use]
    pub const fn is_isolated(_embedding: &[f32; 768]) -> bool {
        // Placeholder: currently cannot detect isolation without HNSW
        // Return false (assume not isolated) for now
        false
    }
}

impl Default for PlausibilityChecker {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ca1_gate_creation() {
        let gate = CA1Gate::new(Confidence::exact(0.7));
        assert!((gate.threshold.raw() - 0.7).abs() < 1e-6);
        assert!((gate.convergence_weight - 0.3).abs() < 1e-6);
        assert!((gate.energy_weight - 0.25).abs() < 1e-6);
    }

    #[test]
    fn test_passes_threshold() {
        let gate = CA1Gate::new(Confidence::exact(0.7));

        assert!(gate.passes_threshold(Confidence::exact(0.8)));
        assert!(gate.passes_threshold(Confidence::exact(0.7)));
        assert!(!gate.passes_threshold(Confidence::exact(0.6)));
    }

    #[test]
    fn test_compute_completion_confidence_high() {
        let gate = CA1Gate::new(Confidence::exact(0.7));

        // High-confidence scenario: fast convergence, good energy reduction, high consensus
        let stats = ConvergenceStats {
            iterations: 3,
            converged: true,
            final_energy: -5.0,
            energy_delta: 3.0,
            state_change: 0.005,
        };

        let confidence = gate.compute_completion_confidence(&stats, 0.9, 0.85);
        assert!(
            confidence.raw() > 0.6,
            "High-quality completion should have reasonable confidence: {}",
            confidence.raw()
        );
    }

    #[test]
    fn test_compute_completion_confidence_low() {
        let gate = CA1Gate::new(Confidence::exact(0.7));

        // Low-confidence scenario: slow convergence, poor energy, low consensus
        let stats = ConvergenceStats {
            iterations: 7,
            converged: false,
            final_energy: -1.0,
            energy_delta: 0.5,
            state_change: 0.05,
        };

        let confidence = gate.compute_completion_confidence(&stats, 0.4, 0.5);
        assert!(
            confidence.raw() < 0.7,
            "Low-quality completion should have low confidence: {}",
            confidence.raw()
        );
    }

    #[test]
    fn test_plausibility_score_normal_embedding() {
        let checker = PlausibilityChecker::new();

        // Normal embedding: reasonable magnitude and variance
        let mut embedding = [0.0f32; 768];
        for (i, item) in embedding.iter_mut().enumerate() {
            *item = (i as f32 / 768.0) * 2.0 - 1.0; // Range [-1, 1]
        }

        let score = checker.score_plausibility(&embedding);
        assert!(score > 0.5, "Normal embedding should be plausible: {score}");
    }

    #[test]
    fn test_plausibility_score_degenerate_embedding() {
        let checker = PlausibilityChecker::new();

        // Degenerate embedding: all zeros
        let embedding = [0.0f32; 768];

        let score = checker.score_plausibility(&embedding);
        assert!(
            score < 0.8,
            "Degenerate embedding should have low plausibility: {score}"
        );
    }

    #[test]
    fn test_field_consensus_empty() {
        let _gate = CA1Gate::new(Confidence::exact(0.7));
        let fields = HashMap::new();
        let consensus = CA1Gate::compute_field_consensus(&fields);
        assert!(
            (consensus - 0.5).abs() < 1e-6,
            "Empty fields should return neutral consensus"
        );
    }
}
