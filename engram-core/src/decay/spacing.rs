//! Spaced repetition optimization with empirical validation.

use std::time::Duration;

/// Spaced repetition optimizer implementing testing effect
#[derive(Debug, Clone)]
pub struct SpacedRepetitionOptimizer {
    /// Optimal spacing multiplier
    pub spacing_multiplier: f32,
    /// Testing effect strength
    pub testing_effect: f32,
}

impl Default for SpacedRepetitionOptimizer {
    fn default() -> Self {
        Self {
            spacing_multiplier: 2.5, // Empirical optimal spacing
            testing_effect: 1.3, // 30% boost from testing
        }
    }
}

impl SpacedRepetitionOptimizer {
    pub fn new() -> Self {
        Self::default()
    }
    
    /// Computes optimal spacing interval
    pub fn optimal_spacing_interval(&self, current_interval: Duration, success: bool) -> Duration {
        if success {
            Duration::from_secs((current_interval.as_secs() as f32 * self.spacing_multiplier) as u64)
        } else {
            Duration::from_secs(86400) // Reset to 1 day on failure
        }
    }
    
    /// Applies testing effect boost to memory strength
    pub fn apply_testing_effect(&self, base_strength: f32, retrieval_success: bool) -> f32 {
        if retrieval_success {
            base_strength * self.testing_effect
        } else {
            base_strength * 0.8 // Slight penalty for failure
        }
    }
}