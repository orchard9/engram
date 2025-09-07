//! Memory consolidation for pattern extraction and systems consolidation.

use crate::{Episode, Memory, Confidence};
use super::CompletionConfig;
use std::collections::{HashMap, VecDeque};
use chrono::{DateTime, Utc, Duration};

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
}

impl ConsolidationEngine {
    /// Create a new consolidation engine
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
        let selected = self.select_for_replay(episodes);
        
        if selected.is_empty() {
            return;
        }
        
        // Create replay event
        let replay_event = ReplayEvent {
            episodes: selected.clone(),
            speed_multiplier: 10.0, // 10x speed replay
            timestamp: Utc::now(),
            ripple_frequency: self.config.ripple_frequency,
            ripple_duration: self.config.ripple_duration,
        };
        
        // Add to replay buffer
        self.replay_buffer.push_back(replay_event.clone());
        if self.replay_buffer.len() > self.config.replay_buffer_size {
            self.replay_buffer.pop_front();
        }
        
        // Extract patterns during replay
        self.extract_patterns(&selected);
        
        // Update statistics
        self.stats.total_replays += 1;
        self.stats.average_replay_speed = 
            (self.stats.average_replay_speed * (self.stats.total_replays - 1) as f32 + 
             replay_event.speed_multiplier) / self.stats.total_replays as f32;
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
        scored_episodes.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        
        // Take top episodes for replay
        scored_episodes.into_iter()
            .take(5)
            .map(|(ep, _)| ep)
            .collect()
    }
    
    /// Calculate prediction error for an episode
    fn calculate_prediction_error(&self, episode: &Episode) -> f32 {
        // Check if we have a semantic pattern for this
        if let Some(pattern) = self.find_matching_pattern(episode) {
            // Calculate difference between episode and pattern
            let diff = self.embedding_distance(&episode.embedding, &pattern.embedding);
            diff / 768.0 // Normalize
        } else {
            1.0 // Maximum error if no pattern exists
        }
    }
    
    /// Find matching semantic pattern for an episode
    fn find_matching_pattern(&self, episode: &Episode) -> Option<&SemanticPattern> {
        let mut best_match = None;
        let mut best_similarity = 0.0;
        
        for pattern in self.semantic_patterns.values() {
            let similarity = self.embedding_similarity(&episode.embedding, &pattern.embedding);
            if similarity > best_similarity && similarity > 0.7 {
                best_similarity = similarity;
                best_match = Some(pattern);
            }
        }
        
        best_match
    }
    
    /// Extract semantic patterns from episodes
    fn extract_patterns(&mut self, episodes: &[Episode]) {
        // Group episodes by semantic similarity
        let clusters = self.cluster_episodes(episodes);
        
        for cluster in clusters {
            if cluster.len() >= 2 {
                // Extract common pattern
                let pattern = self.extract_common_pattern(&cluster);
                
                // Store pattern
                self.semantic_patterns.insert(pattern.id.clone(), pattern);
                self.stats.total_patterns_extracted += 1;
                self.stats.successful_consolidations += 1;
            }
        }
    }
    
    /// Cluster episodes by semantic similarity
    fn cluster_episodes(&self, episodes: &[Episode]) -> Vec<Vec<Episode>> {
        let mut clusters: Vec<Vec<Episode>> = Vec::new();
        
        for episode in episodes {
            let mut added = false;
            
            // Try to add to existing cluster
            for cluster in &mut clusters {
                if !cluster.is_empty() {
                    let similarity = self.embedding_similarity(
                        &episode.embedding, 
                        &cluster[0].embedding
                    );
                    
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
    fn extract_common_pattern(&self, episodes: &[Episode]) -> SemanticPattern {
        // Average embeddings
        let mut avg_embedding = [0.0f32; 768];
        for episode in episodes {
            for i in 0..768 {
                avg_embedding[i] += episode.embedding[i];
            }
        }
        for i in 0..768 {
            avg_embedding[i] /= episodes.len() as f32;
        }
        
        // Calculate pattern strength
        let mut strength = 0.0;
        for episode in episodes {
            strength += episode.encoding_confidence.raw();
        }
        strength /= episodes.len() as f32;
        
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
    fn embedding_similarity(&self, a: &[f32; 768], b: &[f32; 768]) -> f32 {
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
    fn embedding_distance(&self, a: &[f32; 768], b: &[f32; 768]) -> f32 {
        a.iter().zip(b.iter())
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f32>()
            .sqrt()
    }
    
    /// Transform episodic to semantic memory
    pub fn episodic_to_semantic(&mut self, episodes: Vec<Episode>) -> Vec<Memory> {
        let mut semantic_memories = Vec::new();
        
        // Perform consolidation
        self.ripple_replay(&episodes);
        
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
    pub fn get_consolidation_progress(&self, episode: &Episode) -> f32 {
        if let Some(pattern) = self.find_matching_pattern(episode) {
            // Check how many times this episode contributed to patterns
            let contribution_count = self.semantic_patterns.values()
                .filter(|p| p.source_episodes.contains(&episode.id))
                .count();
            
            (contribution_count as f32 / 5.0).min(1.0) // Normalize to 0-1
        } else {
            0.0
        }
    }
    
    /// Systems consolidation from hippocampal to neocortical
    pub fn systems_consolidation(&mut self, hippocampal_episodes: Vec<Episode>, 
                                time_delay: Duration) -> Vec<Memory> {
        let mut neocortical_memories = Vec::new();
        
        for episode in hippocampal_episodes {
            // Check if enough time has passed for consolidation
            let age = Utc::now() - episode.when;
            if age > time_delay {
                // Transform to semantic memory
                let confidence_decay = (-age.num_hours() as f32 / 24.0).exp();
                let semantic_confidence = Confidence::exact(
                    episode.encoding_confidence.raw() * confidence_decay
                );
                
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
        let config = CompletionConfig::default();
        let engine = ConsolidationEngine::new(config);
        
        let a = [1.0; 768];
        let b = [1.0; 768];
        assert_eq!(engine.embedding_similarity(&a, &b), 1.0);
        
        let c = [0.0; 768];
        assert_eq!(engine.embedding_similarity(&a, &c), 0.0);
    }
    
    #[test]
    fn test_cluster_episodes() {
        let config = CompletionConfig::default();
        let engine = ConsolidationEngine::new(config);
        
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
        
        let clusters = engine.cluster_episodes(&episodes);
        assert_eq!(clusters.len(), 2); // Two clusters expected
    }
}