//! REMERGE progressive episodic-to-semantic transformation processor.

use crate::{Confidence, Episode};
use chrono::Utc;

/// REMERGE processor implementing progressive episodic-to-semantic transformation
#[derive(Debug, Clone)]
pub struct RemergeProcessor {
    /// Timeline for systems consolidation (default: 3 years)
    pub consolidation_timeline_days: f32,
    /// Semantic extraction rate
    pub extraction_rate: f32,
}

impl Default for RemergeProcessor {
    fn default() -> Self {
        Self {
            consolidation_timeline_days: 1095.0, // 3 years
            extraction_rate: 1.0,
        }
    }
}

impl RemergeProcessor {
    /// Create a new remerge processor with default settings
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Computes semantic extraction progress (0.0-1.0)
    #[must_use]
    pub fn semantic_extraction_progress(&self, age_days: f32) -> f32 {
        let progress = age_days / self.consolidation_timeline_days;
        (progress * self.extraction_rate).min(1.0)
    }

    /// Applies REMERGE transformation to episode confidence
    #[must_use]
    pub fn transform_episode_confidence(&self, episode: &Episode) -> Confidence {
        let age_days = (Utc::now() - episode.when)
            .to_std()
            .map(|duration| duration.as_secs_f32() / 86400.0)
            .unwrap_or(0.0);
        let semantic_progress = self.semantic_extraction_progress(age_days);

        // Progressive transfer from episodic to semantic confidence
        let episodic_weight = 1.0 - semantic_progress;
        let semantic_weight = semantic_progress;

        episode.encoding_confidence.combine_weighted(
            episode.reliability_confidence,
            episodic_weight,
            semantic_weight,
        )
    }
}
