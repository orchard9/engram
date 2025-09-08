//! Pattern reconstruction engine for missing information recovery.

use super::{
    ActivationPathway, ActivationTrace, CompletedEpisode, CompletionConfig, CompletionError,
    CompletionResult, MemorySource, PartialEpisode, PatternCompleter, SourceMap,
    hippocampal::HippocampalCompletion,
};
use crate::{Confidence, Episode, Memory};
use chrono::Utc;
use std::collections::{HashMap, HashSet};

/// Pattern reconstruction engine combining multiple strategies
pub struct PatternReconstructor {
    /// Hippocampal completion engine
    hippocampal: HippocampalCompletion,

    /// Memory store for context retrieval
    memory_store: Vec<Memory>,

    /// Episode store for temporal patterns
    episode_store: Vec<Episode>,

    /// Configuration
    config: CompletionConfig,

    /// Cache of reconstruction patterns
    pattern_cache: HashMap<String, Vec<f32>>,
}

impl PatternReconstructor {
    /// Create a new pattern reconstructor
    #[must_use]
    pub fn new(config: CompletionConfig) -> Self {
        Self {
            hippocampal: HippocampalCompletion::new(config.clone()),
            memory_store: Vec::new(),
            episode_store: Vec::new(),
            config,
            pattern_cache: HashMap::new(),
        }
    }

    /// Add memories to the reconstruction context
    pub fn add_memories(&mut self, memories: Vec<Memory>) {
        self.memory_store.extend(memories);
    }

    /// Add episodes for temporal pattern learning
    pub fn add_episodes(&mut self, episodes: Vec<Episode>) {
        self.episode_store.extend(episodes.clone());
        self.hippocampal.update(&episodes);
    }

    /// Reconstruct missing fields using semantic similarity
    fn reconstruct_semantic_fields(&self, partial: &PartialEpisode) -> HashMap<String, String> {
        let mut reconstructed = HashMap::new();

        // Find similar episodes based on known fields
        let similar_episodes = self.find_similar_episodes(partial);

        // Aggregate missing fields from similar episodes
        if !partial.known_fields.contains_key("what") {
            if let Some(what) = self.aggregate_field(&similar_episodes, "what") {
                reconstructed.insert("what".to_string(), what);
            }
        }

        if !partial.known_fields.contains_key("where") {
            if let Some(where_loc) = self.aggregate_field(&similar_episodes, "where") {
                reconstructed.insert("where".to_string(), where_loc);
            }
        }

        if !partial.known_fields.contains_key("who") {
            if let Some(who) = self.aggregate_field(&similar_episodes, "who") {
                reconstructed.insert("who".to_string(), who);
            }
        }

        reconstructed
    }

    /// Find episodes similar to the partial pattern
    fn find_similar_episodes(&self, partial: &PartialEpisode) -> Vec<&Episode> {
        let mut scored_episodes: Vec<(&Episode, f32)> = Vec::new();

        for episode in &self.episode_store {
            let mut score = 0.0;
            let mut count = 0;

            // Score based on known fields matching
            if let Some(what) = partial.known_fields.get("what") {
                if episode.what.contains(what) {
                    score += 1.0;
                }
                count += 1;
            }

            if let Some(where_loc) = partial.known_fields.get("where") {
                if let Some(ref ep_where) = episode.where_location {
                    if ep_where.contains(where_loc) {
                        score += 1.0;
                    }
                }
                count += 1;
            }

            // Score based on temporal context
            for context_id in &partial.temporal_context {
                if episode.id == *context_id {
                    score += 0.5;
                }
            }

            if count > 0 {
                scored_episodes.push((episode, score / count as f32));
            }
        }

        // Sort by score and take top-k
        scored_episodes.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        scored_episodes
            .into_iter()
            .take(5)
            .map(|(ep, _)| ep)
            .collect()
    }

    /// Aggregate a field from similar episodes
    fn aggregate_field(&self, episodes: &[&Episode], field: &str) -> Option<String> {
        match field {
            "what" => {
                // Take most common 'what' field
                let mut counts = HashMap::new();
                for ep in episodes {
                    *counts.entry(ep.what.clone()).or_insert(0) += 1;
                }
                counts
                    .into_iter()
                    .max_by_key(|(_, count)| *count)
                    .map(|(what, _)| what)
            }
            "where" => {
                // Take most common location
                let mut counts = HashMap::new();
                for ep in episodes {
                    if let Some(ref loc) = ep.where_location {
                        *counts.entry(loc.clone()).or_insert(0) += 1;
                    }
                }
                counts
                    .into_iter()
                    .max_by_key(|(_, count)| *count)
                    .map(|(loc, _)| loc)
            }
            "who" => {
                // Aggregate all participants
                let mut participants = HashSet::new();
                for ep in episodes {
                    if let Some(ref who) = ep.who {
                        participants.extend(who.clone());
                    }
                }
                if participants.is_empty() {
                    None
                } else {
                    Some(participants.into_iter().collect::<Vec<_>>().join(", "))
                }
            }
            _ => None,
        }
    }

    /// Reconstruct embedding using weighted average
    fn reconstruct_embedding(
        &self,
        partial: &PartialEpisode,
        similar_episodes: &[&Episode],
    ) -> [f32; 768] {
        let mut embedding = [0.0f32; 768];
        let mut weights = [0.0f32; 768];

        // Start with known dimensions
        for (i, value) in partial.partial_embedding.iter().enumerate() {
            if i >= 768 {
                break;
            }
            if let Some(v) = value {
                embedding[i] = *v;
                weights[i] = 1.0;
            }
        }

        // Fill unknown dimensions from similar episodes
        if !similar_episodes.is_empty() {
            for i in 0..768 {
                if weights[i] == 0.0 {
                    let mut sum = 0.0;
                    for ep in similar_episodes {
                        sum += ep.embedding[i];
                    }
                    embedding[i] = sum / similar_episodes.len() as f32;
                }
            }
        }

        embedding
    }

    /// Calculate reconstruction confidence
    fn calculate_confidence(
        &self,
        partial: &PartialEpisode,
        reconstructed_fields: usize,
    ) -> Confidence {
        let known_ratio = partial.known_fields.len() as f32 / 4.0; // 4 main fields
        let reconstructed_ratio = reconstructed_fields as f32 / 4.0;
        let base_confidence = reconstructed_ratio.mul_add(0.5, known_ratio).min(1.0);

        Confidence::exact(base_confidence * partial.cue_strength.raw())
    }

    /// Build activation evidence from reconstruction process
    fn build_activation_evidence(&self, similar_episodes: &[&Episode]) -> Vec<ActivationTrace> {
        let mut traces = Vec::new();

        for (i, episode) in similar_episodes.iter().enumerate() {
            let pathway = if i == 0 {
                ActivationPathway::Direct
            } else if i < 3 {
                ActivationPathway::Semantic
            } else {
                ActivationPathway::Transitive
            };

            traces.push(ActivationTrace {
                source_memory: episode.id.clone(),
                activation_strength: 1.0 / (i + 1) as f32,
                pathway,
                decay_factor: 0.1 * (i + 1) as f32,
            });
        }

        traces
    }
}

impl PatternCompleter for PatternReconstructor {
    fn complete(&self, partial: &PartialEpisode) -> CompletionResult<CompletedEpisode> {
        // First try hippocampal completion
        let hippocampal_result = self.hippocampal.complete(partial);

        // If hippocampal fails, use semantic reconstruction
        if hippocampal_result.is_err() {
            // Find similar episodes
            let similar_episodes = self.find_similar_episodes(partial);

            if similar_episodes.is_empty() {
                return Err(CompletionError::InsufficientPattern);
            }

            // Reconstruct missing fields
            let reconstructed_fields = self.reconstruct_semantic_fields(partial);

            // Build complete episode
            let mut all_fields = partial.known_fields.clone();
            all_fields.extend(reconstructed_fields.clone());

            // Reconstruct embedding
            let embedding = self.reconstruct_embedding(partial, &similar_episodes);

            // Calculate confidence
            let confidence = self.calculate_confidence(partial, reconstructed_fields.len());

            // Create episode
            let episode = Episode {
                id: format!("reconstructed_{}", Utc::now().timestamp()),
                when: Utc::now(),
                where_location: all_fields.get("where").cloned(),
                who: all_fields.get("who").map(|w| vec![w.clone()]),
                what: all_fields
                    .get("what")
                    .cloned()
                    .unwrap_or_else(|| "Reconstructed memory".to_string()),
                embedding,
                encoding_confidence: confidence,
                vividness_confidence: confidence,
                reliability_confidence: confidence,
                last_recall: Utc::now(),
                recall_count: 0,
                decay_rate: 0.05,
            };

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

            for field in reconstructed_fields.keys() {
                source_map
                    .field_sources
                    .insert(field.clone(), MemorySource::Reconstructed);
                source_map
                    .source_confidence
                    .insert(field.clone(), confidence);
            }

            // Build activation evidence
            let activation_evidence = self.build_activation_evidence(&similar_episodes);

            Ok(CompletedEpisode {
                episode,
                completion_confidence: confidence,
                source_attribution: source_map,
                alternative_hypotheses: Vec::new(),
                metacognitive_confidence: confidence,
                activation_evidence,
            })
        } else {
            hippocampal_result
        }
    }

    fn update(&mut self, episodes: &[Episode]) {
        self.add_episodes(episodes.to_vec());
    }

    fn estimate_confidence(&self, partial: &PartialEpisode) -> Confidence {
        self.hippocampal.estimate_confidence(partial)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pattern_reconstructor_creation() {
        let config = CompletionConfig::default();
        let reconstructor = PatternReconstructor::new(config);
        assert_eq!(reconstructor.memory_store.len(), 0);
        assert_eq!(reconstructor.episode_store.len(), 0);
    }

    #[test]
    fn test_calculate_confidence() {
        let config = CompletionConfig::default();
        let reconstructor = PatternReconstructor::new(config);

        let partial = PartialEpisode {
            known_fields: HashMap::from([
                ("what".to_string(), "test".to_string()),
                ("where".to_string(), "here".to_string()),
            ]),
            partial_embedding: vec![Some(1.0); 768],
            cue_strength: Confidence::exact(0.8),
            temporal_context: Vec::new(),
        };

        let confidence = reconstructor.calculate_confidence(&partial, 1);
        assert!(confidence.raw() > 0.0);
        assert!(confidence.raw() <= 1.0);
    }
}
