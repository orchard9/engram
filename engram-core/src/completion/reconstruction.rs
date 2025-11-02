//! Pattern reconstruction engine for missing information recovery.

use super::numeric::{one_over_usize, ratio, safe_divide, usize_to_f32};
use super::{
    ActivationPathway, ActivationTrace, CompletedEpisode, CompletionConfig, CompletionError,
    CompletionResult, MemorySource, PartialEpisode, PatternCompleter, SourceMap,
    hippocampal::HippocampalCompletion,
};
use crate::{Confidence, Episode, Memory};
use chrono::Utc;
use std::collections::HashMap;
use std::sync::Mutex;

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
    pattern_cache: Mutex<HashMap<String, [f32; 768]>>,
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
            pattern_cache: Mutex::new(HashMap::new()),
        }
    }

    /// Add memories to the reconstruction context
    pub fn add_memories(&mut self, memories: Vec<Memory>) {
        self.memory_store.extend(memories);
    }

    /// Add episodes for temporal pattern learning
    pub fn add_episodes(&mut self, episodes: &[Episode]) {
        self.episode_store.extend(episodes.iter().cloned());
        self.hippocampal.update(episodes);
    }

    /// Reconstruct missing fields using semantic similarity
    fn reconstruct_semantic_fields(&self, partial: &PartialEpisode) -> HashMap<String, String> {
        let mut reconstructed = HashMap::new();

        // Find similar episodes based on known fields
        let similar_episodes = self.find_similar_episodes(partial);

        // Aggregate missing fields from similar episodes
        if !partial.known_fields.contains_key("what")
            && let Some(what) = self.aggregate_field(&similar_episodes, "what")
        {
            reconstructed.insert("what".to_string(), what);
        }

        if !partial.known_fields.contains_key("where")
            && let Some(where_loc) = self.aggregate_field(&similar_episodes, "where")
        {
            reconstructed.insert("where".to_string(), where_loc);
        }

        if !partial.known_fields.contains_key("who")
            && let Some(who) = self.aggregate_field(&similar_episodes, "who")
        {
            reconstructed.insert("who".to_string(), who);
        }

        reconstructed
    }

    /// Find episodes similar to the partial pattern
    fn find_similar_episodes(&self, partial: &PartialEpisode) -> Vec<&Episode> {
        let mut scored_episodes: Vec<(&Episode, f32)> = Vec::new();

        for episode in &self.episode_store {
            let mut score = 0.0;
            let mut count = 0;

            // Score based on known fields matching (bidirectional and case-insensitive)
            if let Some(what) = partial.known_fields.get("what") {
                let what_lower = what.to_lowercase();
                let episode_what_lower = episode.what.to_lowercase();

                // Check both directions for partial matches
                if episode_what_lower.contains(&what_lower)
                    || what_lower.contains(&episode_what_lower)
                {
                    score += 1.0;
                } else {
                    // Check for word-level matches
                    let what_words: Vec<&str> = what_lower.split_whitespace().collect();
                    let episode_words: Vec<&str> = episode_what_lower.split_whitespace().collect();

                    let mut word_matches = 0;
                    for what_word in &what_words {
                        for episode_word in &episode_words {
                            if what_word == episode_word
                                || what_word.contains(episode_word)
                                || episode_word.contains(what_word)
                            {
                                word_matches += 1;
                                break;
                            }
                        }
                    }

                    if word_matches > 0 {
                        score += ratio(word_matches, what_words.len().max(episode_words.len()));
                    }
                }
                count += 1;
            }

            if let Some(where_loc) = partial.known_fields.get("where") {
                if let Some(ref ep_where) = episode.where_location {
                    let where_lower = where_loc.to_lowercase();
                    let ep_where_lower = ep_where.to_lowercase();

                    if ep_where_lower.contains(&where_lower)
                        || where_lower.contains(&ep_where_lower)
                    {
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

            // Always include episodes with non-zero scores
            if score > 0.0 {
                let normalized_score = if count > 0 {
                    safe_divide(score, count)
                } else {
                    score
                };
                scored_episodes.push((episode, normalized_score));
            }
        }

        // Sort by score and take top-k, but lower the threshold
        scored_episodes.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let limit = self.config.working_memory_capacity.max(3);
        scored_episodes
            .into_iter()
            .take(limit)
            .map(|(ep, _)| ep)
            .collect()
    }

    /// Aggregate a field from similar episodes
    fn aggregate_field(&self, episodes: &[&Episode], field: &str) -> Option<String> {
        let _ = self;
        if episodes.is_empty() {
            return None;
        }

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
                // Take most common location - prioritize exact matches
                let mut counts = HashMap::new();
                for ep in episodes {
                    if let Some(ref loc) = ep.where_location {
                        *counts.entry(loc.clone()).or_insert(0) += 1;
                    }
                }

                if counts.is_empty() {
                    None
                } else {
                    // Return the most frequent location
                    counts
                        .into_iter()
                        .max_by_key(|(_, count)| *count)
                        .map(|(loc, _)| loc)
                }
            }
            "who" => {
                // Aggregate all participants, taking most common ones first
                let mut participant_counts = HashMap::new();
                for ep in episodes {
                    if let Some(ref who_list) = ep.who {
                        for person in who_list {
                            *participant_counts.entry(person.clone()).or_insert(0) += 1;
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

                    Some(participants.join(", "))
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
        let cache_key = Self::cache_key(partial);
        if let Ok(cache) = self.pattern_cache.lock()
            && let Some(cached) = cache.get(&cache_key)
        {
            return *cached;
        }

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
                    embedding[i] = safe_divide(sum, similar_episodes.len());
                }
            }
        }

        let result = embedding;

        if let Ok(mut cache) = self.pattern_cache.lock() {
            cache.insert(cache_key, result);
        }

        result
    }

    /// Calculate reconstruction confidence
    fn calculate_confidence(partial: &PartialEpisode, reconstructed_fields: usize) -> Confidence {
        let known_ratio = ratio(partial.known_fields.len(), 4); // 4 main fields
        let reconstructed_ratio = ratio(reconstructed_fields, 4);
        let base_confidence = reconstructed_ratio.mul_add(0.5, known_ratio).min(1.0);

        Confidence::exact(base_confidence * partial.cue_strength.raw())
    }

    /// Build activation evidence from reconstruction process
    fn build_activation_evidence(similar_episodes: &[&Episode]) -> Vec<ActivationTrace> {
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
                activation_strength: one_over_usize(i + 1),
                pathway,
                decay_factor: 0.1 * usize_to_f32(i + 1),
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
            let confidence = Self::calculate_confidence(partial, reconstructed_fields.len());

            // Create episode
            let episode = Episode {
                id: format!("reconstructed_{ts}", ts = Utc::now().timestamp()),
                when: Utc::now(),
                where_location: all_fields.get("where").cloned(),
                who: all_fields.get("who").map(|w| vec![w.clone()]),
                what: all_fields
                    .get("what")
                    .cloned()
                    .unwrap_or_else(|| "Reconstructed memory".to_string()),
                embedding,
                embedding_provenance: None, // Reconstructed episodes don't have provenance
                encoding_confidence: confidence,
                vividness_confidence: confidence,
                reliability_confidence: confidence,
                last_recall: Utc::now(),
                recall_count: 0,
                decay_rate: 0.05,
                decay_function: None, // Use system default for reconstructed episodes
                metadata: std::collections::HashMap::new(),
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
            let activation_evidence = Self::build_activation_evidence(&similar_episodes);

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
        self.add_episodes(episodes);
    }

    fn estimate_confidence(&self, partial: &PartialEpisode) -> Confidence {
        self.hippocampal.estimate_confidence(partial)
    }
}

impl PatternReconstructor {
    fn cache_key(partial: &PartialEpisode) -> String {
        let mut parts: Vec<String> = partial
            .known_fields
            .iter()
            .map(|(key, value)| format!("{key}:{value}"))
            .collect();
        parts.sort();

        let mut contexts: Vec<String> = partial
            .temporal_context
            .iter()
            .map(|ctx| format!("context:{ctx}"))
            .collect();
        contexts.sort();
        parts.extend(contexts);

        parts.join("|")
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
        let partial = PartialEpisode {
            known_fields: HashMap::from([
                ("what".to_string(), "test".to_string()),
                ("where".to_string(), "here".to_string()),
            ]),
            partial_embedding: vec![Some(1.0); 768],
            cue_strength: Confidence::exact(0.8),
            temporal_context: Vec::new(),
        };

        let confidence = PatternReconstructor::calculate_confidence(&partial, 1);
        assert!(confidence.raw() > 0.0);
        assert!(confidence.raw() <= 1.0);
    }
}
