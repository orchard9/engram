//! Memory consolidation for pattern extraction and systems consolidation.

use super::CompletionConfig;
use super::numeric::{i64_to_f32, ratio, safe_divide, usize_to_f32};
use crate::{Confidence, Episode, Memory};
use chrono::{DateTime, Duration, Utc};
use std::collections::{HashMap, VecDeque};

/// Consolidation engine for memory transformation
pub struct ConsolidationEngine {
    /// Configuration
    config: CompletionConfig,

    /// Replay buffer for sharp-wave ripple events
    replay_buffer: VecDeque<ReplayEvent>,

    /// Consolidated semantic patterns
    semantic_patterns: HashMap<String, SemanticPattern>,

    /// Consolidation statistics
    stats: ConsolidationStats,
}

/// A replay event during sharp-wave ripples
#[derive(Debug, Clone)]
struct ReplayEvent {
    /// Episodes being replayed
    episodes: Vec<Episode>,

    /// Replay speed multiplier (typically 8-20x)
    speed_multiplier: f32,

    /// Timestamp of replay
    timestamp: DateTime<Utc>,

    /// Ripple characteristics
    ripple_frequency: f32,
    ripple_duration: f32,
}

/// A consolidated semantic pattern
#[derive(Debug, Clone)]
pub struct SemanticPattern {
    /// Pattern identifier
    pub id: String,

    /// Semantic embedding (averaged across episodes)
    pub embedding: [f32; 768],

    /// Contributing episodes
    pub source_episodes: Vec<String>,

    /// Pattern strength
    pub strength: f32,

    /// Schema confidence
    pub schema_confidence: Confidence,

    /// Last consolidation time
    pub last_consolidated: DateTime<Utc>,
}

/// Statistics for consolidation process
#[derive(Debug, Clone, Default)]
struct ConsolidationStats {
    total_replays: usize,
    successful_consolidations: usize,
    failed_consolidations: usize,
    average_replay_speed: f32,
    total_patterns_extracted: usize,
    avg_ripple_frequency: f32,
    avg_ripple_duration: f32,
    last_replay_timestamp: Option<DateTime<Utc>>,
}

impl ConsolidationEngine {
    /// Create a new consolidation engine
    #[must_use]
    pub fn new(config: CompletionConfig) -> Self {
        Self {
            config,
            replay_buffer: VecDeque::with_capacity(100),
            semantic_patterns: HashMap::new(),
            stats: ConsolidationStats::default(),
        }
    }

    /// Perform sharp-wave ripple replay for consolidation
    pub fn ripple_replay(&mut self, episodes: &[Episode]) {
        // Select episodes for replay (prioritize by importance/recency)
        let replay_episodes = self.select_for_replay(episodes);

        if replay_episodes.is_empty() {
            return;
        }

        // Create replay event
        let replay_event = ReplayEvent {
            episodes: replay_episodes,
            speed_multiplier: 10.0, // 10x speed replay
            timestamp: Utc::now(),
            ripple_frequency: self.config.ripple_frequency,
            ripple_duration: self.config.ripple_duration,
        };

        // Add to replay buffer
        self.replay_buffer.push_back(replay_event.clone());
        if self.replay_buffer.len() > self.config.working_memory_capacity {
            self.replay_buffer.pop_front();
        }

        // Extract patterns during replay
        let extracted = self.extract_patterns(&replay_event);

        // Update statistics
        self.record_replay_metrics(&replay_event, extracted);
    }

    /// Select episodes for replay based on prediction error
    fn select_for_replay(&self, episodes: &[Episode]) -> Vec<Episode> {
        let mut scored_episodes: Vec<(Episode, f32)> = Vec::new();

        for episode in episodes {
            // Calculate prediction error (simplified)
            let error = self.calculate_prediction_error(episode);

            // Prioritize high prediction error (need more consolidation)
            let priority = error * episode.encoding_confidence.raw();

            scored_episodes.push((episode.clone(), priority));
        }

        // Sort by priority
        scored_episodes.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Take top episodes for replay
        scored_episodes
            .into_iter()
            .take(5)
            .map(|(ep, _)| ep)
            .collect()
    }

    /// Calculate prediction error for an episode
    fn calculate_prediction_error(&self, episode: &Episode) -> f32 {
        // Check if we have a semantic pattern for this
        self.find_matching_pattern(episode).map_or(1.0, |pattern| {
            // Calculate difference between episode and pattern
            let diff = Self::embedding_distance(&episode.embedding, &pattern.embedding);
            diff / 768.0 // Normalize
        })
    }

    /// Find matching semantic pattern for an episode
    fn find_matching_pattern(&self, episode: &Episode) -> Option<&SemanticPattern> {
        let mut best_match = None;
        let mut best_similarity = 0.0;

        for pattern in self.semantic_patterns.values() {
            let similarity = Self::embedding_similarity(&episode.embedding, &pattern.embedding);
            if similarity > best_similarity && similarity > 0.7 {
                best_similarity = similarity;
                best_match = Some(pattern);
            }
        }

        best_match
    }

    /// Extract semantic patterns from episodes
    fn extract_patterns(&mut self, event: &ReplayEvent) -> usize {
        // Group episodes by semantic similarity
        let clusters = Self::cluster_episodes(&event.episodes);
        let mut extracted = 0;

        for cluster in clusters {
            if cluster.len() >= 2 {
                // Extract common pattern
                let pattern = Self::extract_common_pattern(&cluster);

                // Store pattern
                self.semantic_patterns.insert(pattern.id.clone(), pattern);
                extracted += 1;
            } else {
                self.stats.failed_consolidations += 1;
            }
        }

        if extracted > 0 {
            self.stats.total_patterns_extracted += extracted;
            self.stats.successful_consolidations += extracted;
        }

        extracted
    }

    fn record_replay_metrics(&mut self, event: &ReplayEvent, extracted_patterns: usize) {
        let previous_total = self.stats.total_replays;
        self.stats.total_replays = previous_total.saturating_add(1);
        let total_replays = self.stats.total_replays;
        let previous_total_f32 = usize_to_f32(previous_total);

        let denominator = event.episodes.len();
        let effectiveness = if denominator == 0 {
            0.0
        } else {
            let successful = extracted_patterns.min(denominator);
            usize_to_f32(successful) / usize_to_f32(denominator)
        };

        let weighted_speed = event
            .speed_multiplier
            .mul_add(0.5 * effectiveness, 0.5 * event.speed_multiplier);

        self.stats.average_replay_speed = safe_divide(
            self.stats
                .average_replay_speed
                .mul_add(previous_total_f32, weighted_speed),
            total_replays,
        );
        self.stats.avg_ripple_frequency = safe_divide(
            self.stats
                .avg_ripple_frequency
                .mul_add(previous_total_f32, event.ripple_frequency),
            total_replays,
        );
        self.stats.avg_ripple_duration = safe_divide(
            self.stats
                .avg_ripple_duration
                .mul_add(previous_total_f32, event.ripple_duration),
            total_replays,
        );
        self.stats.last_replay_timestamp = Some(event.timestamp);
    }

    /// Cluster episodes by semantic similarity
    fn cluster_episodes(episodes: &[Episode]) -> Vec<Vec<Episode>> {
        let mut clusters: Vec<Vec<Episode>> = Vec::new();

        for episode in episodes {
            let mut added = false;

            // Try to add to existing cluster
            for cluster in &mut clusters {
                if !cluster.is_empty() {
                    let similarity =
                        Self::embedding_similarity(&episode.embedding, &cluster[0].embedding);

                    if similarity > 0.8 {
                        cluster.push(episode.clone());
                        added = true;
                        break;
                    }
                }
            }

            // Create new cluster if needed
            if !added {
                clusters.push(vec![episode.clone()]);
            }
        }

        clusters
    }

    /// Extract common pattern from episode cluster
    fn extract_common_pattern(episodes: &[Episode]) -> SemanticPattern {
        // Average embeddings
        let mut avg_embedding = [0.0f32; 768];
        for episode in episodes {
            for (avg_val, episode_val) in avg_embedding.iter_mut().zip(&episode.embedding) {
                *avg_val += episode_val;
            }
        }
        for avg_val in &mut avg_embedding {
            *avg_val = safe_divide(*avg_val, episodes.len());
        }

        // Calculate pattern strength
        let mut strength = 0.0;
        for episode in episodes {
            strength += episode.encoding_confidence.raw();
        }
        strength = safe_divide(strength, episodes.len());

        SemanticPattern {
            id: format!("pattern_{}", Utc::now().timestamp()),
            embedding: avg_embedding,
            source_episodes: episodes.iter().map(|e| e.id.clone()).collect(),
            strength,
            schema_confidence: Confidence::exact(strength),
            last_consolidated: Utc::now(),
        }
    }

    /// Calculate embedding similarity
    fn embedding_similarity(a: &[f32; 768], b: &[f32; 768]) -> f32 {
        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm_a > 0.0 && norm_b > 0.0 {
            dot / (norm_a * norm_b)
        } else {
            0.0
        }
    }

    /// Calculate embedding distance
    fn embedding_distance(a: &[f32; 768], b: &[f32; 768]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f32>()
            .sqrt()
    }

    /// Transform episodic to semantic memory
    pub fn episodic_to_semantic(&mut self, episodes: &[Episode]) -> Vec<Memory> {
        let mut semantic_memories = Vec::new();

        // Perform consolidation
        self.ripple_replay(episodes);

        // Convert patterns to semantic memories
        for pattern in self.semantic_patterns.values() {
            let memory = Memory::new(
                pattern.id.clone(),
                pattern.embedding,
                pattern.schema_confidence,
            );
            semantic_memories.push(memory);
        }

        semantic_memories
    }

    /// Get consolidation progress for an episode
    #[must_use]
    pub fn get_consolidation_progress(&self, episode: &Episode) -> f32 {
        self.find_matching_pattern(episode).map_or(0.0, |_| {
            // Check how many times this episode contributed to patterns
            let contribution_count = self
                .semantic_patterns
                .values()
                .filter(|p| p.source_episodes.contains(&episode.id))
                .count();

            ratio(contribution_count, 5).min(1.0) // Normalize to 0-1
        })
    }

    /// Systems consolidation from hippocampal to neocortical
    #[must_use]
    pub fn systems_consolidation(
        hippocampal_episodes: Vec<Episode>,
        time_delay: Duration,
    ) -> Vec<Memory> {
        let mut neocortical_memories = Vec::new();

        for episode in hippocampal_episodes {
            // Check if enough time has passed for consolidation
            let age = Utc::now() - episode.when;
            if age > time_delay {
                // Transform to semantic memory
                let hours = i64_to_f32(age.num_hours());
                let confidence_decay = (-(hours) / 24.0).exp();
                let semantic_confidence =
                    Confidence::exact(episode.encoding_confidence.raw() * confidence_decay);

                let memory = Memory::new(
                    format!("consolidated_{}", episode.id),
                    episode.embedding,
                    semantic_confidence,
                );

                neocortical_memories.push(memory);
            }
        }

        neocortical_memories
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_consolidation_engine_creation() {
        let config = CompletionConfig::default();
        let engine = ConsolidationEngine::new(config);
        assert_eq!(engine.semantic_patterns.len(), 0);
        assert_eq!(engine.stats.total_replays, 0);
    }

    #[test]
    fn test_embedding_similarity() {
        let a = [1.0; 768];
        let b = [1.0; 768];
        assert!(
            (ConsolidationEngine::embedding_similarity(&a, &b) - 1.0_f32).abs() <= f32::EPSILON
        );

        let c = [0.0; 768];
        assert!(ConsolidationEngine::embedding_similarity(&a, &c).abs() <= f32::EPSILON);
    }

    #[test]
    fn test_cluster_episodes() {
        let episodes = vec![
            Episode::new(
                "ep1".to_string(),
                Utc::now(),
                "test1".to_string(),
                [1.0; 768],
                Confidence::exact(0.9),
            ),
            Episode::new(
                "ep2".to_string(),
                Utc::now(),
                "test2".to_string(),
                [1.0; 768], // Same embedding, should cluster together
                Confidence::exact(0.8),
            ),
            Episode::new(
                "ep3".to_string(),
                Utc::now(),
                "test3".to_string(),
                [0.0; 768], // Different, should be separate cluster
                Confidence::exact(0.7),
            ),
        ];

        let clusters = ConsolidationEngine::cluster_episodes(&episodes);
        assert_eq!(clusters.len(), 2); // Two clusters expected
    }
}
