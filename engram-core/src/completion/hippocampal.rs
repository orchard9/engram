//! Hippocampal-inspired pattern completion with CA3/CA1/DG dynamics.

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
        let k = (expanded_size as f32 * self.config.ca3_sparsity) as usize;
        let mut activations: Vec<(usize, f32)> = Vec::new();

        // Generate expanded representation
        for i in 0..expanded_size {
            let mut activation = 0.0;
            for j in 0..size {
                // Random projection for expansion
                let weight = ((i * size + j) as f32).sin() * 0.1;
                activation += input[j] * weight;
            }
            activations.push((i, activation));
        }

        // Select top-k neurons
        activations.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        for i in 0..k.min(activations.len()) {
            separated[activations[i].0] = activations[i].1;
        }

        // Compress back to original size
        let mut output = DVector::zeros(size);
        for i in 0..size {
            for j in 0..self.config.dg_expansion_factor {
                output[i] += separated[i * self.config.dg_expansion_factor + j];
            }
            output[i] /= self.config.dg_expansion_factor as f32;
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
        let k = (self.current_state.len() as f32 * self.config.ca3_sparsity) as usize;
        let mut activations: Vec<(usize, f32)> = self
            .current_state
            .iter()
            .enumerate()
            .map(|(i, &v)| (i, v))
            .collect();

        activations.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

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
            let similarity = self.cosine_similarity(&episode.embedding, completed_pattern);
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
    fn cosine_similarity(&self, a: &[f32], b: &[f32]) -> f32 {
        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm_a > 0.0 && norm_b > 0.0 {
            dot / (norm_a * norm_b)
        } else {
            0.0
        }
    }
}

impl PatternCompleter for HippocampalCompletion {
    fn complete(&self, partial: &PartialEpisode) -> CompletionResult<CompletedEpisode> {
        // Convert partial embedding to full vector
        #[cfg(feature = "pattern_completion")]
        {
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

            // Check if we have enough information
            if known_count < 100 {
                return Err(CompletionError::InsufficientPattern);
            }

            // Clone self for mutable operations
            let mut engine = Self::new(self.config.clone());
            engine.ca3_weights = self.ca3_weights.clone();
            engine.stored_patterns = self.stored_patterns.clone();

            // Pattern separation
            let separated = engine.pattern_separate(&input);

            // CA3 attractor dynamics
            let completed = engine.ca3_dynamics(separated);

            // Convert back to array
            let mut completed_embedding = [0.0f32; 768];
            for i in 0..768 {
                completed_embedding[i] = completed[i];
            }

            // Find best matching episode or create new one
            let episode = if let Some(matched) = self.find_best_match(&completed_embedding) {
                matched.clone()
            } else {
                // Create new episode from completed pattern
                Episode::new(
                    format!("completed_{}", chrono::Utc::now().timestamp()),
                    Utc::now(),
                    partial
                        .known_fields
                        .get("what")
                        .cloned()
                        .unwrap_or_else(|| "Reconstructed memory".to_string()),
                    completed_embedding,
                    partial.cue_strength,
                )
            };

            // Calculate completion confidence
            let completion_confidence = Confidence::exact(
                0.9 * (1.0 - (engine.iterations as f32 / self.config.max_iterations as f32)),
            );

            // Build source map
            let mut source_map = SourceMap {
                field_sources: HashMap::new(),
                source_confidence: HashMap::new(),
            };

            for field in partial.known_fields.keys() {
                source_map
                    .field_sources
                    .insert(field.clone(), MemorySource::Recalled);
                source_map
                    .source_confidence
                    .insert(field.clone(), Confidence::exact(1.0));
            }

            // Add reconstructed fields
            if !partial.known_fields.contains_key("what") {
                source_map
                    .field_sources
                    .insert("what".to_string(), MemorySource::Reconstructed);
                source_map
                    .source_confidence
                    .insert("what".to_string(), completion_confidence);
            }

            // Create activation trace
            let activation_trace = ActivationTrace {
                source_memory: episode.id.clone(),
                activation_strength: completion_confidence.raw(),
                pathway: ActivationPathway::Direct,
                decay_factor: 0.1,
            };

            Ok(CompletedEpisode {
                episode,
                completion_confidence,
                source_attribution: source_map,
                alternative_hypotheses: Vec::new(),
                metacognitive_confidence: completion_confidence,
                activation_evidence: vec![activation_trace],
            })
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

        let ratio = known_count as f32 / total as f32;
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
        let config = CompletionConfig::default();
        let engine = HippocampalCompletion::new(config);

        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert_eq!(engine.cosine_similarity(&a, &b), 1.0);

        let c = vec![0.0, 1.0, 0.0];
        assert_eq!(engine.cosine_similarity(&a, &c), 0.0);
    }
}
