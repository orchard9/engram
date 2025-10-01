//! Hippocampal-inspired pattern completion with CA3/CA1/DG dynamics.

use super::numeric::{fraction_to_count, ratio, usize_to_f32};
use super::{
    ActivationPathway, ActivationTrace, BiologicalDynamics, CompletedEpisode, CompletionConfig,
    CompletionError, CompletionResult, MemorySource, PartialEpisode, PatternCompleter, SourceMap,
};
use crate::{Confidence, Episode};
use chrono::Utc;
use std::collections::HashMap;

#[cfg(feature = "pattern_completion")]
use nalgebra::{DMatrix, DVector};

/// Hippocampal-inspired pattern completion engine
pub struct HippocampalCompletion {
    /// Configuration parameters
    config: CompletionConfig,

    /// CA3 recurrent weights (autoassociative memory)
    #[cfg(feature = "pattern_completion")]
    ca3_weights: DMatrix<f32>,

    /// Current activation state
    #[cfg(feature = "pattern_completion")]
    current_state: DVector<f32>,

    /// Previous state for convergence check
    #[cfg(feature = "pattern_completion")]
    previous_state: DVector<f32>,

    /// Iteration counter
    iterations: usize,

    /// Has converged flag
    converged: bool,

    /// Stored patterns for retrieval
    stored_patterns: Vec<Episode>,
}

impl HippocampalCompletion {
    /// Create a new hippocampal completion engine
    #[must_use]
    pub fn new(config: CompletionConfig) -> Self {
        #[cfg(feature = "pattern_completion")]
        let size = 768;

        Self {
            config,
            #[cfg(feature = "pattern_completion")]
            ca3_weights: DMatrix::zeros(size, size),
            #[cfg(feature = "pattern_completion")]
            current_state: DVector::zeros(size),
            #[cfg(feature = "pattern_completion")]
            previous_state: DVector::zeros(size),
            iterations: 0,
            converged: false,
            stored_patterns: Vec::new(),
        }
    }

    /// Dentate Gyrus: Pattern separation through sparse encoding
    #[cfg(feature = "pattern_completion")]
    fn pattern_separate(&self, input: &DVector<f32>) -> DVector<f32> {
        let size = input.len();
        let expanded_size = size * self.config.dg_expansion_factor;
        let mut separated = DVector::zeros(expanded_size);

        // Competitive k-winner-take-all for sparsity
        let k = fraction_to_count(expanded_size, self.config.ca3_sparsity);
        let mut activations: Vec<(usize, f32)> = Vec::new();

        // Generate expanded representation
        for i in 0..expanded_size {
            let mut activation = 0.0;
            for j in 0..size {
                // Random projection for expansion
                let index = i
                    .checked_mul(size)
                    .and_then(|value| value.checked_add(j))
                    .unwrap_or_default();
                let weight = usize_to_f32(index).sin() * 0.1;
                activation += input[j] * weight;
            }
            activations.push((i, activation));
        }

        // Select top-k neurons
        activations.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        for i in 0..k.min(activations.len()) {
            separated[activations[i].0] = activations[i].1;
        }

        // Compress back to original size
        let mut output = DVector::zeros(size);
        let expansion_factor = usize_to_f32(self.config.dg_expansion_factor).max(1.0);
        for i in 0..size {
            for j in 0..self.config.dg_expansion_factor {
                output[i] += separated[i * self.config.dg_expansion_factor + j];
            }
            output[i] /= expansion_factor;
        }

        output
    }

    /// CA3: Autoassociative dynamics with attractor network
    #[cfg(feature = "pattern_completion")]
    fn ca3_dynamics(&mut self, input: DVector<f32>) -> DVector<f32> {
        self.current_state = input;

        for _ in 0..self.config.max_iterations {
            self.previous_state = self.current_state.clone();

            // Hopfield-like update: s(t+1) = sign(W * s(t))
            let activation = &self.ca3_weights * &self.current_state;

            // Apply sigmoid activation with sparsity
            for i in 0..activation.len() {
                self.current_state[i] = 1.0 / (1.0 + (-activation[i]).exp());
            }

            // Apply sparsity constraint
            self.apply_sparsity_constraint();

            // Check convergence
            let diff = (&self.current_state - &self.previous_state).norm();
            if diff < self.config.convergence_threshold {
                self.converged = true;
                break;
            }

            self.iterations += 1;
        }

        self.current_state.clone()
    }

    /// Apply sparsity constraint using k-winner-take-all
    #[cfg(feature = "pattern_completion")]
    fn apply_sparsity_constraint(&mut self) {
        let k = fraction_to_count(self.current_state.len(), self.config.ca3_sparsity);
        let mut activations: Vec<(usize, f32)> = self
            .current_state
            .iter()
            .enumerate()
            .map(|(i, &v)| (i, v))
            .collect();

        activations.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Zero out all but top-k
        for i in k..self.current_state.len() {
            if i < activations.len() {
                self.current_state[activations[i].0] = 0.0;
            }
        }
    }

    /// CA1: Output gating with confidence calibration
    fn ca1_gate(&self, _pattern: &[f32], completion_confidence: Confidence) -> bool {
        completion_confidence.raw() >= self.config.ca1_threshold.raw()
    }

    /// Update CA3 weights using Hebbian learning
    #[cfg(feature = "pattern_completion")]
    fn update_weights(&mut self, pattern: &DVector<f32>) {
        // Hebbian update: ΔW = η * (pattern * pattern^T)
        let learning_rate = 0.01;
        let update = pattern * pattern.transpose() * learning_rate;
        self.ca3_weights += update;

        // Normalize weights to prevent saturation
        let max_weight = self.ca3_weights.max();
        if max_weight > 1.0 {
            self.ca3_weights /= max_weight;
        }
    }

    /// Find best matching stored pattern
    fn find_best_match(&self, completed_pattern: &[f32]) -> Option<&Episode> {
        let mut best_match = None;
        let mut best_similarity = 0.0;

        for episode in &self.stored_patterns {
            let similarity = Self::cosine_similarity(&episode.embedding, completed_pattern);
            if similarity > best_similarity {
                best_similarity = similarity;
                best_match = Some(episode);
            }
        }

        if best_similarity > 0.7 {
            best_match
        } else {
            None
        }
    }

    /// Compute cosine similarity between two vectors
    fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm_a > 0.0 && norm_b > 0.0 {
            dot / (norm_a * norm_b)
        } else {
            0.0
        }
    }

    /// Reconstruct location field from stored patterns using semantic similarity
    fn reconstruct_location_from_patterns(&self, partial: &PartialEpisode) -> Option<String> {
        partial.known_fields.get("what").and_then(|what| {
            let mut location_counts = std::collections::HashMap::new();

            for episode in &self.stored_patterns {
                // Check if this episode is semantically similar
                let what_lower = what.to_lowercase();
                let episode_what_lower = episode.what.to_lowercase();

                let is_similar = episode_what_lower.contains(&what_lower)
                    || what_lower.contains(&episode_what_lower)
                    || Self::word_level_similarity(&what_lower, &episode_what_lower) > 0.0;

                if is_similar {
                    if let Some(ref location) = episode.where_location {
                        *location_counts.entry(location.clone()).or_insert(0) += 1;
                    }
                }
            }

            // Return most common location
            location_counts
                .into_iter()
                .max_by_key(|(_, count)| *count)
                .map(|(location, _)| location)
        })
    }

    /// Reconstruct participants field from stored patterns using semantic similarity
    fn reconstruct_participants_from_patterns(
        &self,
        partial: &PartialEpisode,
    ) -> Option<Vec<String>> {
        partial.known_fields.get("what").and_then(|what| {
            let mut participant_counts = std::collections::HashMap::new();

            for episode in &self.stored_patterns {
                // Check if this episode is semantically similar
                let what_lower = what.to_lowercase();
                let episode_what_lower = episode.what.to_lowercase();

                let is_similar = episode_what_lower.contains(&what_lower)
                    || what_lower.contains(&episode_what_lower)
                    || Self::word_level_similarity(&what_lower, &episode_what_lower) > 0.0;

                if is_similar {
                    if let Some(ref participants) = episode.who {
                        for participant in participants {
                            *participant_counts.entry(participant.clone()).or_insert(0) += 1;
                        }
                    }
                }
            }

            if participant_counts.is_empty() {
                None
            } else {
                // Sort by frequency and take top participants
                let mut sorted_participants: Vec<_> = participant_counts.into_iter().collect();
                sorted_participants.sort_by(|a, b| b.1.cmp(&a.1));

                let participants: Vec<String> = sorted_participants
                    .into_iter()
                    .take(3) // Limit to top 3 most frequent participants
                    .map(|(name, _)| name)
                    .collect();

                if participants.is_empty() {
                    None
                } else {
                    Some(participants)
                }
            }
        })
    }

    /// Calculate word-level similarity score
    fn word_level_similarity(what1: &str, what2: &str) -> f32 {
        let words1: Vec<&str> = what1.split_whitespace().collect();
        let words2: Vec<&str> = what2.split_whitespace().collect();

        let mut matches = 0;
        for word1 in &words1 {
            for word2 in &words2 {
                if word1 == word2 || word1.contains(word2) || word2.contains(word1) {
                    matches += 1;
                    break;
                }
            }
        }

        if !words1.is_empty() && !words2.is_empty() {
            ratio(matches, words1.len().max(words2.len()))
        } else {
            0.0
        }
    }

    /// Prepare input vector from partial episode
    fn prepare_input_vector(
        &self,
        partial: &PartialEpisode,
    ) -> CompletionResult<(DVector<f32>, usize)> {
        let _ = self;
        let mut input = DVector::zeros(768);
        let mut known_count = 0;

        for (i, value) in partial.partial_embedding.iter().enumerate() {
            if i >= 768 {
                break;
            }
            if let Some(v) = value {
                input[i] = *v;
                known_count += 1;
            }
        }

        // Validate we have sufficient information
        if known_count < 100 {
            return Err(CompletionError::InsufficientPattern);
        }

        Ok((input, known_count))
    }

    /// Apply the core pattern completion algorithm
    fn apply_pattern_completion_algorithm(&self, input: &DVector<f32>) -> ([f32; 768], usize) {
        // Clone self for mutable operations
        let mut engine = Self::new(self.config.clone());
        engine.ca3_weights = self.ca3_weights.clone();
        engine.stored_patterns.clone_from(&self.stored_patterns);

        // Pattern separation
        let separated = engine.pattern_separate(input);

        // CA3 attractor dynamics
        let completed = engine.ca3_dynamics(separated);

        // Convert back to array
        let mut completed_embedding = [0.0f32; 768];
        for i in 0..768 {
            completed_embedding[i] = completed[i];
        }

        (completed_embedding, engine.iterations)
    }

    /// Find existing episode or create new one from completed pattern
    fn find_or_create_episode(
        &self,
        partial: &PartialEpisode,
        completed_embedding: &[f32; 768],
    ) -> Episode {
        self.find_best_match(completed_embedding).map_or_else(
            || self.create_new_episode_from_pattern(partial, completed_embedding),
            Clone::clone,
        )
    }

    /// Create a new episode from completed pattern
    fn create_new_episode_from_pattern(
        &self,
        partial: &PartialEpisode,
        completed_embedding: &[f32; 768],
    ) -> Episode {
        let mut episode = Episode::new(
            format!("completed_{}", chrono::Utc::now().timestamp()),
            Utc::now(),
            partial
                .known_fields
                .get("what")
                .cloned()
                .unwrap_or_else(|| "Reconstructed memory".to_string()),
            *completed_embedding,
            partial.cue_strength,
        );

        // Try to reconstruct location from stored patterns
        if let Some(where_location) = self.reconstruct_location_from_patterns(partial) {
            episode.where_location = Some(where_location);
        }

        // Try to reconstruct participants from stored patterns
        if let Some(participants) = self.reconstruct_participants_from_patterns(partial) {
            episode.who = Some(participants);
        }

        episode
    }

    /// Build the final completed episode with metadata
    fn build_completed_episode(
        &self,
        episode: Episode,
        partial: &PartialEpisode,
        completion_confidence: Confidence,
    ) -> CompletedEpisode {
        let _ = self;
        // Build source attribution map
        let source_map = self.build_source_attribution(partial, completion_confidence);

        // Create activation trace
        let activation_trace = ActivationTrace {
            source_memory: episode.id.clone(),
            activation_strength: completion_confidence.raw(),
            pathway: ActivationPathway::Direct,
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

    /// Calculate confidence based on completion quality
    fn calculate_completion_confidence(
        &self,
        _known_count: usize,
        iterations: usize,
    ) -> Confidence {
        // Higher confidence when fewer iterations needed (matches original logic)
        let iteration_ratio = ratio(iterations, self.config.max_iterations);

        Confidence::exact(0.9 * (1.0 - iteration_ratio))
    }

    /// Build source attribution map for completed fields
    fn build_source_attribution(
        &self,
        partial: &PartialEpisode,
        completion_confidence: Confidence,
    ) -> SourceMap {
        let _ = self;
        let mut source_map = SourceMap {
            field_sources: HashMap::new(),
            source_confidence: HashMap::new(),
        };

        // Mark known fields as recalled with high confidence
        for field in partial.known_fields.keys() {
            source_map
                .field_sources
                .insert(field.clone(), MemorySource::Recalled);
            source_map
                .source_confidence
                .insert(field.clone(), Confidence::exact(1.0));
        }

        // Add reconstructed fields with completion confidence
        if !partial.known_fields.contains_key("what") {
            source_map
                .field_sources
                .insert("what".to_string(), MemorySource::Reconstructed);
            source_map
                .source_confidence
                .insert("what".to_string(), completion_confidence);
        }

        source_map
    }
}

impl PatternCompleter for HippocampalCompletion {
    fn complete(&self, partial: &PartialEpisode) -> CompletionResult<CompletedEpisode> {
        #[cfg(feature = "pattern_completion")]
        {
            // 1. Validate and prepare input
            let (input, known_count) = self.prepare_input_vector(partial)?;

            // 2. Apply pattern completion algorithm
            let (completed_embedding, iterations) = self.apply_pattern_completion_algorithm(&input);

            // 3. Find or create episode from completed pattern
            let episode = self.find_or_create_episode(partial, &completed_embedding);

            let completion_confidence =
                self.calculate_completion_confidence(known_count, iterations);

            if !self.ca1_gate(&episode.embedding, completion_confidence) {
                return Err(CompletionError::InsufficientPattern);
            }

            // 4. Build completion metadata
            let completed_episode =
                self.build_completed_episode(episode, partial, completion_confidence);

            Ok(completed_episode)
        }

        #[cfg(not(feature = "pattern_completion"))]
        {
            Err(CompletionError::MatrixError(
                "Pattern completion feature not enabled".to_string(),
            ))
        }
    }

    fn update(&mut self, episodes: &[Episode]) {
        self.stored_patterns.extend_from_slice(episodes);

        #[cfg(feature = "pattern_completion")]
        {
            // Update CA3 weights with new episodes
            for episode in episodes {
                let pattern = DVector::from_row_slice(&episode.embedding);
                self.update_weights(&pattern);
            }
        }
    }

    fn estimate_confidence(&self, partial: &PartialEpisode) -> Confidence {
        // Estimate based on percentage of known information
        let known_count = partial
            .partial_embedding
            .iter()
            .filter(|v| v.is_some())
            .count();
        let total = partial.partial_embedding.len();

        let ratio = ratio(known_count, total);
        Confidence::exact(ratio * partial.cue_strength.raw())
    }
}

#[cfg(feature = "pattern_completion")]
impl BiologicalDynamics for HippocampalCompletion {
    fn step(&mut self, input: &DVector<f32>) -> DVector<f32> {
        self.previous_state = self.current_state.clone();
        self.current_state = input.clone();

        // One step of attractor dynamics
        let activation = &self.ca3_weights * &self.current_state;

        // Apply activation function
        for i in 0..activation.len() {
            self.current_state[i] = 1.0 / (1.0 + (-activation[i]).exp());
        }

        self.apply_sparsity_constraint();
        self.current_state.clone()
    }

    fn has_converged(&self) -> bool {
        self.converged
    }

    fn reset(&mut self) {
        self.current_state = DVector::zeros(768);
        self.previous_state = DVector::zeros(768);
        self.iterations = 0;
        self.converged = false;
    }

    fn energy(&self) -> f32 {
        // Hopfield energy: E = -0.5 * s^T * W * s

        -0.5 * self
            .current_state
            .dot(&(&self.ca3_weights * &self.current_state))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hippocampal_creation() {
        let config = CompletionConfig::default();
        let engine = HippocampalCompletion::new(config);
        assert_eq!(engine.iterations, 0);
        assert!(!engine.converged);
    }

    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert!((HippocampalCompletion::cosine_similarity(&a, &b) - 1.0_f32).abs() <= f32::EPSILON);

        let c = vec![0.0, 1.0, 0.0];
        assert!(HippocampalCompletion::cosine_similarity(&a, &c).abs() <= f32::EPSILON);
    }
}
