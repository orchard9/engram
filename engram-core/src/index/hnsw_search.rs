//! SIMD-optimized search operations for HNSW

use crate::Confidence;

/// Result of a search operation
#[derive(Clone, Debug)]
pub struct SearchResult {
    /// Node identifier in the HNSW graph
    pub node_id: u32,
    /// Distance to the query point
    pub distance: f32,
    /// Confidence in this search result
    pub confidence: Confidence,
    /// Memory identifier associated with this node
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

    /// Compute cosine similarity from stored distance (1 - similarity)
    #[must_use]
    pub fn similarity(&self) -> f32 {
        (1.0 - self.distance).clamp(-1.0, 1.0)
    }

    /// Get confidence-weighted distance for ranking
    #[must_use]
    pub fn weighted_distance(&self) -> f32 {
        self.distance * (1.0 - self.confidence.raw())
    }
}

/// Summary statistics describing an HNSW search run
#[derive(Clone, Debug, Default)]
pub struct SearchStats {
    /// Number of nodes visited during search
    pub nodes_visited: usize,
    /// Effective search parameter used
    pub ef_used: usize,
    /// Approximation quality ratio
    pub approximation_ratio: f32,
    /// Search thoroughness metric
    pub thoroughness: f32,
}

impl SearchStats {
    #[must_use]
    /// Create search stats with the given ef parameter
    pub const fn with_ef(ef_used: usize) -> Self {
        Self {
            nodes_visited: 0,
            ef_used,
            approximation_ratio: 1.0,
            thoroughness: 0.0,
        }
    }

    /// Record the number of nodes visited in a layer
    pub const fn record_layer(&mut self, visited: usize) {
        self.nodes_visited += visited;
    }

    /// Finalize search statistics with distance calculations
    pub fn finalize(&mut self, distances: &[f32]) {
        if distances.is_empty() {
            self.approximation_ratio = 1.0;
            if self.ef_used == 0 {
                self.thoroughness = 0.0;
            } else {
                #[allow(clippy::cast_precision_loss)]
                let visited = self.nodes_visited as f64;
                #[allow(clippy::cast_precision_loss)]
                let ef = self.ef_used as f64;
                #[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
                {
                    self.thoroughness = ((visited / ef).min(1.0)) as f32;
                }
            }
            return;
        }

        let (min_dist, max_dist) = distances
            .iter()
            .fold((f32::INFINITY, f32::NEG_INFINITY), |acc, value| {
                (acc.0.min(*value), acc.1.max(*value))
            });

        let epsilon = f32::EPSILON;
        let ratio = (min_dist + epsilon) / (max_dist + epsilon);
        self.approximation_ratio = ratio.clamp(0.0, 1.0);

        if self.ef_used > 0 {
            #[allow(clippy::cast_precision_loss)]
            let target = self.ef_used as f64;
            #[allow(clippy::cast_precision_loss)]
            let visited = self.nodes_visited as f64;
            #[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
            {
                self.thoroughness = ((visited / target).min(1.0)) as f32;
            }
        } else {
            self.thoroughness = 0.0;
        }
    }
}

/// Detailed search output including statistics
#[derive(Clone, Debug, Default)]
pub struct SearchResults {
    /// Search results found
    pub hits: Vec<SearchResult>,
    /// Search performance statistics
    pub stats: SearchStats,
}
