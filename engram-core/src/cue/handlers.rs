//! Concrete cue handler implementations

use super::{CueHandler, CueContext};
use crate::{Cue, CueType, Episode, Confidence};
use crate::error::CueError;

/// Handler for embedding-based cue matching
pub struct EmbeddingCueHandler;

impl CueHandler for EmbeddingCueHandler {
    fn handle(
        &self,
        cue: &Cue,
        context: &dyn CueContext,
    ) -> Result<Vec<(Episode, Confidence)>, CueError> {
        if let CueType::Embedding { vector, threshold } = &cue.cue_type {
            let episodes = context.get_episodes();
            let mut results = Vec::new();
            
            // Use HNSW for efficient similarity search
            let candidates = context.search_similar_episodes(
                vector,
                cue.max_results * 2, // Over-fetch for better results
                *threshold,
            );
            
            for (episode_id, _hnsw_confidence) in candidates {
                if let Some(episode) = episodes.get(&episode_id) {
                    let similarity = context.compute_similarity(vector, &episode.embedding);
                    
                    if similarity >= threshold.raw() {
                        let confidence = Confidence::exact(similarity);
                        let decayed_confidence = context.apply_decay(episode, confidence);
                        results.push((episode.clone(), decayed_confidence));
                    }
                }
            }
            
            results.sort_by(|a, b| b.1.raw().partial_cmp(&a.1.raw()).unwrap_or(std::cmp::Ordering::Equal));
            Ok(results)
        } else {
            Err(CueError::UnsupportedCueType {
                cue_type: format!("{:?}", cue.cue_type),
                operation: "embedding_handler".to_string(),
            })
        }
    }
    
    fn supports_cue_type(&self, cue_type: &CueType) -> bool {
        matches!(cue_type, CueType::Embedding { .. })
    }
    
    fn handler_name(&self) -> &'static str {
        "embedding"
    }
}

/// Handler for context-based cue matching
pub struct ContextCueHandler;

impl CueHandler for ContextCueHandler {
    fn handle(
        &self,
        cue: &Cue,
        context: &dyn CueContext,
    ) -> Result<Vec<(Episode, Confidence)>, CueError> {
        if let CueType::Context { time_range, location, confidence_threshold } = &cue.cue_type {
            let episodes = context.get_episodes();
            let mut results = Vec::new();
            
            for episode in episodes.values() {
                let mut match_score = 0.0f32;
                let mut match_count = 0;
                
                // Check time range match
                if let Some((start, end)) = time_range {
                    if episode.when >= *start && episode.when <= *end {
                        match_score += 1.0;
                    }
                    match_count += 1;
                }
                
                // Check location match
                if let Some(cue_location) = location {
                    if let Some(episode_location) = &episode.where_location {
                        let location_similarity = Self::calculate_location_similarity(cue_location, episode_location);
                        match_score += location_similarity;
                    }
                    match_count += 1;
                }
                
                if match_count > 0 {
                    let final_score = match_score / match_count as f32;
                    if final_score > confidence_threshold.raw() {
                        let confidence = Confidence::exact(final_score);
                        let decayed_confidence = context.apply_decay(&episode, confidence);
                        results.push((episode.clone(), decayed_confidence));
                    }
                }
            }
            
            results.sort_by(|a, b| b.1.raw().partial_cmp(&a.1.raw()).unwrap_or(std::cmp::Ordering::Equal));
            Ok(results)
        } else {
            Err(CueError::UnsupportedCueType {
                cue_type: format!("{:?}", cue.cue_type),
                operation: "context_handler".to_string(),
            })
        }
    }
    
    fn supports_cue_type(&self, cue_type: &CueType) -> bool {
        matches!(cue_type, CueType::Context { .. })
    }
    
    fn handler_name(&self) -> &'static str {
        "context"
    }
}

impl ContextCueHandler {
    fn calculate_location_similarity(cue_location: &str, episode_location: &str) -> f32 {
        if cue_location == episode_location {
            1.0
        } else if cue_location.to_lowercase().contains(&episode_location.to_lowercase()) 
               || episode_location.to_lowercase().contains(&cue_location.to_lowercase()) {
            0.7
        } else {
            0.0
        }
    }
}

/// Handler for semantic cue matching
pub struct SemanticCueHandler;

impl CueHandler for SemanticCueHandler {
    fn handle(
        &self,
        cue: &Cue,
        context: &dyn CueContext,
    ) -> Result<Vec<(Episode, Confidence)>, CueError> {
        if let CueType::Semantic { content, fuzzy_threshold } = &cue.cue_type {
            let episodes = context.get_episodes();
            let mut results = Vec::new();
            
            for episode in episodes.values() {
                let semantic_similarity = Self::calculate_semantic_similarity(content, &episode.what);
                if semantic_similarity > fuzzy_threshold.raw() {
                    let confidence = Confidence::exact(semantic_similarity);
                    let decayed_confidence = context.apply_decay(&episode, confidence);
                    results.push((episode.clone(), decayed_confidence));
                }
            }
            
            results.sort_by(|a, b| b.1.raw().partial_cmp(&a.1.raw()).unwrap_or(std::cmp::Ordering::Equal));
            Ok(results)
        } else {
            Err(CueError::UnsupportedCueType {
                cue_type: format!("{:?}", cue.cue_type),
                operation: "semantic_handler".to_string(),
            })
        }
    }
    
    fn supports_cue_type(&self, cue_type: &CueType) -> bool {
        matches!(cue_type, CueType::Semantic { .. })
    }
    
    fn handler_name(&self) -> &'static str {
        "semantic"
    }
}

impl SemanticCueHandler {
    fn calculate_semantic_similarity(cue_content: &str, episode_content: &str) -> f32 {
        let cue_lower = cue_content.to_lowercase();
        let episode_lower = episode_content.to_lowercase();
        
        let cue_words: std::collections::HashSet<&str> = cue_lower
            .split_whitespace()
            .collect();
        let episode_words: std::collections::HashSet<&str> = episode_lower
            .split_whitespace()
            .collect();
        
        let intersection_size = cue_words.intersection(&episode_words).count();
        let union_size = cue_words.union(&episode_words).count();
        
        if union_size == 0 {
            0.0
        } else {
            let jaccard_score = intersection_size as f32 / union_size as f32;
            
            // Boost score for exact substring matches
            if episode_lower.contains(&cue_lower) {
                (jaccard_score + 0.3).min(1.0)
            } else {
                jaccard_score
            }
        }
    }
}

/// Handler for temporal cue matching
pub struct TemporalCueHandler;

impl CueHandler for TemporalCueHandler {
    fn handle(
        &self,
        cue: &Cue,
        context: &dyn CueContext,
    ) -> Result<Vec<(Episode, Confidence)>, CueError> {
        if let CueType::Temporal { pattern, confidence_threshold } = &cue.cue_type {
            let episodes = context.get_episodes();
            let mut results = Vec::new();
            
            for episode in episodes.values() {
                let temporal_match = Self::matches_temporal_pattern(&episode.when, pattern);
                if temporal_match > confidence_threshold.raw() {
                    let confidence = Confidence::exact(temporal_match);
                    let decayed_confidence = context.apply_decay(&episode, confidence);
                    results.push((episode.clone(), decayed_confidence));
                }
            }
            
            results.sort_by(|a, b| b.1.raw().partial_cmp(&a.1.raw()).unwrap_or(std::cmp::Ordering::Equal));
            Ok(results)
        } else {
            Err(CueError::UnsupportedCueType {
                cue_type: format!("{:?}", cue.cue_type),
                operation: "temporal_handler".to_string(),
            })
        }
    }
    
    fn supports_cue_type(&self, cue_type: &CueType) -> bool {
        matches!(cue_type, CueType::Temporal { .. })
    }
    
    fn handler_name(&self) -> &'static str {
        "temporal"
    }
}

impl TemporalCueHandler {
    fn matches_temporal_pattern(episode_time: &chrono::DateTime<chrono::Utc>, pattern: &crate::TemporalPattern) -> f32 {
        use crate::TemporalPattern;
        
        match pattern {
            TemporalPattern::Recent(duration) => {
                let now = chrono::Utc::now();
                let cutoff_time = now - *duration;
                if *episode_time >= cutoff_time {
                    let time_diff = (now - *episode_time).num_milliseconds() as f32;
                    let total_duration_ms = duration.num_milliseconds() as f32;
                    if total_duration_ms > 0.0 {
                        1.0 - (time_diff / total_duration_ms).min(1.0)
                    } else {
                        1.0
                    }
                } else {
                    0.0
                }
            }
            TemporalPattern::Before(cutoff) => {
                if *episode_time < *cutoff {
                    1.0
                } else {
                    0.0
                }
            }
            TemporalPattern::After(cutoff) => {
                if *episode_time > *cutoff {
                    1.0
                } else {
                    0.0
                }
            }
            TemporalPattern::Between(start, end) => {
                if *episode_time >= *start && *episode_time <= *end {
                    1.0
                } else {
                    0.0
                }
            }
        }
    }
}