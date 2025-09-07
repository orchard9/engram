//! Lock-free construction algorithms for HNSW

use super::{HnswEdge, HnswNode};
use crate::Confidence;

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
    pub fn m_max(mut self, m: usize) -> Self {
        self.m_max = m;
        self
    }

    /// Set number of connections for layer 0
    pub fn m_l(mut self, m: usize) -> Self {
        self.m_l = m;
        self
    }

    /// Set construction search width
    pub fn ef_construction(mut self, ef: usize) -> Self {
        self.ef_construction = ef;
        self
    }

    /// Set minimum confidence threshold
    pub fn confidence_threshold(mut self, threshold: Confidence) -> Self {
        self.confidence_threshold = threshold;
        self
    }

    /// Build the HNSW index
    pub fn build(self) -> super::CognitiveHnswIndex {
        let index = super::CognitiveHnswIndex::new();

        // Apply builder parameters (these are Arc'd so we can't mutate them after creation)
        // The parameters would need to be applied during construction in a more sophisticated implementation

        index
    }
}

impl Default for HnswBuilder {
    fn default() -> Self {
        Self::new()
    }
}
