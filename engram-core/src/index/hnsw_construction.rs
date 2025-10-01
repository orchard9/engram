//! Lock-free construction algorithms for HNSW

use crate::Confidence;
use std::sync::Arc;

/// Builder for HNSW index with cognitive parameters
pub struct HnswBuilder {
    m_max: usize,
    m_l: usize,
    ef_construction: usize,
    ml: f32,
    confidence_threshold: Confidence,
}

impl HnswBuilder {
    /// Create a new builder with default parameters
    #[must_use]
    pub fn new() -> Self {
        Self {
            m_max: 16,
            m_l: 32,
            ef_construction: 200,
            ml: 1.0 / (2.0_f32).ln(),
            confidence_threshold: Confidence::LOW,
        }
    }

    /// Set maximum number of connections
    #[must_use]
    pub const fn m_max(mut self, m: usize) -> Self {
        self.m_max = m;
        self
    }

    /// Set number of connections for layer 0
    #[must_use]
    pub const fn m_l(mut self, m: usize) -> Self {
        self.m_l = m;
        self
    }

    /// Set construction search width
    #[must_use]
    pub const fn ef_construction(mut self, ef: usize) -> Self {
        self.ef_construction = ef;
        self
    }

    /// Set minimum confidence threshold
    #[must_use]
    pub const fn confidence_threshold(mut self, threshold: Confidence) -> Self {
        self.confidence_threshold = threshold;
        self
    }

    /// Build the HNSW index
    #[must_use]
    pub fn build(self) -> super::CognitiveHnswIndex {
        let mut index = super::CognitiveHnswIndex::new();

        index
            .params
            .m_max
            .store(self.m_max.max(2), std::sync::atomic::Ordering::Relaxed);
        index
            .params
            .m_l
            .store(self.m_l.max(1), std::sync::atomic::Ordering::Relaxed);
        index.params.ef_construction.store(
            self.ef_construction.max(1),
            std::sync::atomic::Ordering::Relaxed,
        );
        if let Some(params) = Arc::get_mut(&mut index.params) {
            params.ml = self.ml;
            params.confidence_threshold = self.confidence_threshold;
        }

        index
    }
}

impl Default for HnswBuilder {
    fn default() -> Self {
        Self::new()
    }
}
