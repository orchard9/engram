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
            testing_effect: 1.3,     // 30% boost from testing
        }
    }
}

impl SpacedRepetitionOptimizer {
    /// Create a new spaced repetition optimizer with default settings
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Computes optimal spacing interval
    #[must_use]
    pub fn optimal_spacing_interval(&self, current_interval: Duration, success: bool) -> Duration {
        if success {
            let multiplier = self.spacing_multiplier.max(0.1);
            let scaled_seconds = current_interval.as_secs_f32() * multiplier;
            if scaled_seconds.is_finite() {
                Duration::from_secs_f32(scaled_seconds.max(1.0))
            } else {
                Duration::from_secs(u64::MAX)
            }
        } else {
            Duration::from_secs(86400) // Reset to 1 day on failure
        }
    }

    /// Applies testing effect boost to memory strength
    #[must_use]
    pub fn apply_testing_effect(&self, base_strength: f32, retrieval_success: bool) -> f32 {
        if retrieval_success {
            base_strength * self.testing_effect
        } else {
            base_strength * 0.8 // Slight penalty for failure
        }
    }
}
