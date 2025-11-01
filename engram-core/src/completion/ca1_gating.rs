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
    min_neighborhood_agreement: f32,

    /// Vector operations dispatcher for similarity computation
    vector_ops: DispatchVectorOps,

    /// Number of neighbors to check for consistency (default: 5)
    neighborhood_size: usize,
}

impl PlausibilityChecker {
    /// Create new plausibility checker with default parameters
    #[must_use]
    pub fn new() -> Self {
        Self {
            min_neighborhood_agreement: 0.6,
            vector_ops: DispatchVectorOps::new(),
            neighborhood_size: 5,
        }
    }

    /// Create plausibility checker with custom threshold
    #[must_use]
    pub fn with_threshold(min_neighborhood_agreement: f32) -> Self {
        Self {
            min_neighborhood_agreement,
            vector_ops: DispatchVectorOps::new(),
            neighborhood_size: 5,
        }
    }

    /// Create plausibility checker with full configuration
    #[must_use]
    pub fn with_config(min_neighborhood_agreement: f32, neighborhood_size: usize) -> Self {
        Self {
            min_neighborhood_agreement,
            vector_ops: DispatchVectorOps::new(),
            neighborhood_size,
        }
    }

    /// Score plausibility of reconstructed embedding
    ///
    /// Returns 0.0-1.0 score based on:
    /// 1. Embedding magnitude and variance (local consistency)
    /// 2. Neighborhood consistency with HNSW index (when available)
    /// 3. Not in sparse/isolated region
    ///
    /// This implementation provides robust plausibility scoring without requiring
    /// HNSW index access. For full neighborhood consistency checking, use
    /// `score_plausibility_with_hnsw` which requires HNSW index integration.
    #[must_use]
    pub fn score_plausibility(&self, embedding: &[f32; 768]) -> f32 {
        // Component 1: Magnitude check - plausible embeddings have reasonable norm
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

        // Component 2: Variance check - detect degenerate embeddings
        let mean: f32 = embedding.iter().sum::<f32>() / 768.0;
        let variance: f32 = embedding.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / 768.0;

        // Plausible embeddings should have some variance (not all same value)
        let variance_score = if variance > 0.001 { 1.0 } else { 0.3 };

        // Combined score (simple average for now)
        // When HNSW is integrated, this becomes weighted: 0.3 * local + 0.7 * neighborhood
        f32::midpoint(magnitude_score, variance_score)
    }

    /// Score plausibility with HNSW neighborhood consistency checking
    ///
    /// This method integrates with HNSW index to validate that the reconstructed
    /// embedding is consistent with its k-nearest neighbors in the embedding space.
    ///
    /// # Arguments
    ///
    /// * `embedding` - The reconstructed embedding to validate
    /// * `hnsw_index` - Reference to HNSW index for neighborhood queries
    ///
    /// # Returns
    ///
    /// Plausibility score [0.0, 1.0] where:
    /// - 1.0 = highly plausible (consistent with neighbors)
    /// - 0.0 = implausible (isolated or inconsistent)
    #[cfg(feature = "hnsw_index")]
    #[must_use]
    pub fn score_plausibility_with_hnsw(
        &self,
        embedding: &[f32; 768],
        hnsw_index: &crate::index::CognitiveHnswIndex,
    ) -> f32 {
        // Component 1: Local consistency (magnitude + variance)
        let local_score = self.score_plausibility(embedding);

        // Component 2: Neighborhood consistency via HNSW
        let neighborhood_score = self.compute_neighborhood_consistency(embedding, hnsw_index);

        // Weighted combination: neighborhood is more important for hallucination detection
        0.3 * local_score + 0.7 * neighborhood_score
    }

    /// Compute neighborhood consistency score using HNSW index
    ///
    /// Returns score based on:
    /// 1. Average similarity to k-nearest neighbors (higher = more consistent)
    /// 2. Neighborhood density (isolated points score lower)
    /// 3. Agreement on local manifold structure
    #[cfg(feature = "hnsw_index")]
    fn compute_neighborhood_consistency(
        &self,
        embedding: &[f32; 768],
        hnsw_index: &crate::index::CognitiveHnswIndex,
    ) -> f32 {
        // Query HNSW for k nearest neighbors
        let neighbors = hnsw_index.search_with_confidence(
            embedding,
            self.neighborhood_size,
            crate::Confidence::exact(0.0), // No threshold, get all neighbors
        );

        if neighbors.is_empty() {
            // No neighbors found - likely isolated point (low plausibility)
            return 0.2;
        }

        // Compute average similarity to neighbors
        let avg_similarity: f32 = neighbors
            .iter()
            .map(|(_, confidence)| confidence.raw())
            .sum::<f32>()
            / neighbors.len() as f32;

        // High average similarity = good neighborhood consistency
        // Low average similarity = potential hallucination or reconstruction error
        if avg_similarity >= self.min_neighborhood_agreement {
            avg_similarity
        } else {
            // Below threshold - penalize proportionally
            avg_similarity * 0.5
        }
    }

    /// Check if embedding is in sparse region (potential hallucination)
    ///
    /// Without HNSW index access, this uses heuristics based on embedding properties.
    /// For accurate isolation detection, use `is_isolated_with_hnsw`.
    #[must_use]
    pub const fn is_isolated(_embedding: &[f32; 768]) -> bool {
        // Without HNSW access, we cannot reliably detect isolation
        // Return false (assume not isolated) to avoid false positives
        false
    }

    /// Check if embedding is in sparse/isolated region using HNSW neighborhood density
    ///
    /// An embedding is considered isolated if:
    /// 1. It has fewer than k neighbors within similarity threshold
    /// 2. Average distance to neighbors is high
    /// 3. Neighborhood density is low
    ///
    /// # Arguments
    ///
    /// * `embedding` - The embedding to check
    /// * `hnsw_index` - Reference to HNSW index for neighborhood queries
    /// * `min_neighbors` - Minimum number of neighbors to not be isolated (default: 3)
    /// * `similarity_threshold` - Minimum similarity to count as neighbor (default: 0.5)
    ///
    /// # Returns
    ///
    /// `true` if the embedding appears isolated, `false` if well-connected
    #[cfg(feature = "hnsw_index")]
    #[must_use]
    pub fn is_isolated_with_hnsw(
        embedding: &[f32; 768],
        hnsw_index: &crate::index::CognitiveHnswIndex,
        min_neighbors: usize,
        similarity_threshold: f32,
    ) -> bool {
        // Query for neighbors
        let neighbors = hnsw_index.search_with_confidence(
            embedding,
            min_neighbors * 2, // Query extra to account for filtering
            crate::Confidence::exact(similarity_threshold),
        );

        // Check 1: Insufficient neighbors
        if neighbors.len() < min_neighbors {
            return true;
        }

        // Check 2: Average similarity is too low (distant from all neighbors)
        let avg_similarity: f32 = neighbors
            .iter()
            .map(|(_, confidence)| confidence.raw())
            .sum::<f32>()
            / neighbors.len() as f32;

        if avg_similarity < similarity_threshold * 1.5 {
            return true; // Neighbors exist but are all far away
        }

        // Well-connected in the embedding space
        false
    }

    /// Estimate neighborhood density around an embedding using HNSW
    ///
    /// Returns density score [0.0, 1.0] where:
    /// - 1.0 = dense neighborhood (many close neighbors)
    /// - 0.0 = sparse neighborhood (few distant neighbors)
    ///
    /// Higher density suggests the embedding is in a well-populated region
    /// of the semantic space, making hallucination less likely.
    #[cfg(feature = "hnsw_index")]
    #[must_use]
    pub fn estimate_neighborhood_density(
        &self,
        embedding: &[f32; 768],
        hnsw_index: &crate::index::CognitiveHnswIndex,
    ) -> f32 {
        // Query for local neighborhood
        let neighbors = hnsw_index.search_with_confidence(
            embedding,
            self.neighborhood_size,
            crate::Confidence::exact(0.0), // Get all neighbors
        );

        if neighbors.is_empty() {
            return 0.0; // No neighbors = zero density
        }

        // Density factors:
        // 1. Number of neighbors (more = denser)
        let neighbor_count_factor =
            (neighbors.len() as f32 / self.neighborhood_size as f32).min(1.0);

        // 2. Average similarity (closer neighbors = denser)
        let avg_similarity: f32 = neighbors
            .iter()
            .map(|(_, confidence)| confidence.raw())
            .sum::<f32>()
            / neighbors.len() as f32;

        // 3. Similarity variance (tight cluster = denser)
        let mean_sim = avg_similarity;
        let variance: f32 = neighbors
            .iter()
            .map(|(_, confidence)| (confidence.raw() - mean_sim).powi(2))
            .sum::<f32>()
            / neighbors.len() as f32;

        let variance_factor = (1.0 - variance).max(0.0); // Low variance = tight cluster

        // Weighted combination
        0.4 * neighbor_count_factor + 0.4 * avg_similarity + 0.2 * variance_factor
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
