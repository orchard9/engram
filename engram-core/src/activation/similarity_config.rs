use crate::Confidence;

/// Configuration parameters governing similarity to activation mapping
#[derive(Debug, Clone)]
pub struct SimilarityConfig {
    /// Softmax temperature controlling activation sharpness
    pub temperature: f32,
    /// Baseline similarity threshold for seeding activation
    pub threshold: f32,
    /// Maximum number of HNSW candidates to evaluate per cue
    pub max_candidates: usize,
    /// Beam width for HNSW search
    pub ef_search: usize,
    /// Minimum confidence tolerated from the search backend
    pub min_confidence: Confidence,
}

impl Default for SimilarityConfig {
    fn default() -> Self {
        Self {
            temperature: 0.5,
            threshold: 0.4,
            max_candidates: 100,
            ef_search: 96,
            min_confidence: Confidence::MEDIUM,
        }
    }
}

impl SimilarityConfig {
    /// Create a new builder-style configuration
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Ensure temperature stays within a numerically stable range
    #[must_use]
    pub const fn clamped_temperature(&self) -> f32 {
        self.temperature.clamp(0.05, 2.0)
    }

    /// Adjust the similarity threshold using cue-provided minimums
    #[must_use]
    pub const fn effective_threshold(&self, cue_threshold: Confidence) -> f32 {
        self.threshold.max(cue_threshold.raw())
    }

    /// Limit candidate count to avoid pathological workloads
    #[must_use]
    pub fn candidate_limit(&self) -> usize {
        self.max_candidates.max(1)
    }
}
