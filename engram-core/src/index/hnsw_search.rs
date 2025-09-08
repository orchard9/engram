//! SIMD-optimized search operations for HNSW

use crate::Confidence;

/// Result of a search operation
#[derive(Clone, Debug)]
pub struct SearchResult {
    pub node_id: u32,
    pub distance: f32,
    pub confidence: Confidence,
    pub memory_id: String,
}

impl SearchResult {
    /// Create a new search result
    #[must_use]
    pub const fn new(
        node_id: u32,
        distance: f32,
        confidence: Confidence,
        memory_id: String,
    ) -> Self {
        Self {
            node_id,
            distance,
            confidence,
            memory_id,
        }
    }

    /// Get confidence-weighted distance for ranking
    #[must_use]
    pub fn weighted_distance(&self) -> f32 {
        self.distance * (1.0 - self.confidence.raw())
    }
}
