//! Field-level reconstruction engine for completing missing episode details.
//!
//! Implements ensemble-based field completion using temporal neighbors with
//! weighted voting based on similarity and recency. Follows Source Monitoring
//! Framework (Johnson et al., 1993) for precise attribution tracking.

use crate::completion::{MemorySource, PartialEpisode};
use crate::compute::VectorOps;
use crate::compute::dispatch::DispatchVectorOps;
use crate::{Confidence, Episode};
use std::collections::HashMap;
use std::time::Duration;

/// Field-level reconstruction engine for completing missing episode details
pub struct FieldReconstructor {
    /// Temporal window for neighbor retrieval (default: 1 hour)
    temporal_window: Duration,

    /// Minimum similarity for neighbor contribution (default: 0.7)
    similarity_threshold: f32,

    /// Maximum neighbors to consider (default: 5)
    max_neighbors: usize,

    /// Neighbor weighting decay (default: 0.8)
    neighbor_decay: f32,
}

impl FieldReconstructor {
    /// Create new field reconstructor with default parameters
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Create field reconstructor with custom parameters
    #[must_use]
    pub const fn with_params(
        temporal_window: Duration,
        similarity_threshold: f32,
        max_neighbors: usize,
        neighbor_decay: f32,
    ) -> Self {
        Self {
            temporal_window,
            similarity_threshold,
            max_neighbors,
            neighbor_decay,
        }
    }

    /// Reconstruct missing fields using temporal neighbors
    ///
    /// Returns map of field_name -> (reconstructed_value, confidence, source)
    #[must_use]
    pub fn reconstruct_fields(
        &self,
        partial: &PartialEpisode,
        temporal_neighbors: &[Episode],
    ) -> HashMap<String, ReconstructedField> {
        let mut reconstructed = HashMap::new();

        // Filter neighbors by similarity threshold
        let filtered_neighbors = self.filter_neighbors(partial, temporal_neighbors);

        if filtered_neighbors.is_empty() {
            // Graceful degradation: no neighbors available
            return reconstructed;
        }

        // Get all field names from neighbors
        let field_names = Self::collect_field_names(&filtered_neighbors);

        // Reconstruct each missing field
        for field_name in field_names {
            // Skip if field already known in partial episode
            if partial.known_fields.contains_key(&field_name) {
                continue;
            }

            if let Some(reconstructed_field) =
                Self::reconstruct_single_field(&field_name, &filtered_neighbors)
            {
                reconstructed.insert(field_name, reconstructed_field);
            }
        }

        reconstructed
    }

    /// Extract temporal context from episode history
    ///
    /// Returns episodes within temporal window sorted by recency
    #[must_use]
    pub fn extract_temporal_context(
        &self,
        anchor_time: chrono::DateTime<chrono::Utc>,
        episode_store: &[Episode],
    ) -> Vec<Episode> {
        let mut temporal_episodes: Vec<Episode> = episode_store
            .iter()
            .filter(|ep| {
                let time_diff = if ep.when > anchor_time {
                    ep.when.signed_duration_since(anchor_time)
                } else {
                    anchor_time.signed_duration_since(ep.when)
                };

                // Convert to Duration for comparison
                time_diff
                    .to_std()
                    .is_ok_and(|duration| duration <= self.temporal_window)
            })
            .cloned()
            .collect();

        // Sort by recency (closest to anchor time first)
        temporal_episodes.sort_by(|a, b| {
            let dist_a = if a.when > anchor_time {
                a.when.signed_duration_since(anchor_time)
            } else {
                anchor_time.signed_duration_since(a.when)
            };

            let dist_b = if b.when > anchor_time {
                b.when.signed_duration_since(anchor_time)
            } else {
                anchor_time.signed_duration_since(b.when)
            };

            dist_a.cmp(&dist_b)
        });

        temporal_episodes
    }

    /// Compute field confidence based on neighbor consensus
    ///
    /// Higher confidence when multiple neighbors agree
    fn compute_field_confidence(
        field_values: &[(String, f32)], // (value, weight) pairs
    ) -> Confidence {
        if field_values.is_empty() {
            return Confidence::exact(0.0);
        }

        // Find consensus value and compute agreement ratio
        let mut value_weights: HashMap<String, f32> = HashMap::new();
        let mut total_weight = 0.0;

        for (value, weight) in field_values {
            *value_weights.entry(value.clone()).or_insert(0.0) += weight;
            total_weight += weight;
        }

        // Get maximum weight (consensus value)
        let max_weight = value_weights.values().copied().fold(0.0, f32::max);

        // Confidence = weighted agreement ratio
        let confidence = if total_weight > 0.0 {
            max_weight / total_weight
        } else {
            0.0
        };

        Confidence::exact(confidence.clamp(0.0, 1.0))
    }

    /// Filter neighbors by similarity threshold and limit to max_neighbors
    fn filter_neighbors(
        &self,
        partial: &PartialEpisode,
        neighbors: &[Episode],
    ) -> Vec<NeighborEvidence> {
        let vector_ops = DispatchVectorOps::new();

        // Compute partial embedding (fill None with 0.0 for similarity computation)
        let partial_embedding: [f32; 768] = {
            let mut embedding = [0.0f32; 768];
            for (i, opt_val) in partial.partial_embedding.iter().enumerate() {
                if let Some(val) = opt_val {
                    embedding[i] = *val;
                }
            }
            embedding
        };

        let mut evidence: Vec<NeighborEvidence> = neighbors
            .iter()
            .filter_map(|neighbor| {
                let similarity =
                    vector_ops.cosine_similarity_768(&partial_embedding, &neighbor.embedding);

                if similarity >= self.similarity_threshold {
                    Some(NeighborEvidence {
                        episode_id: neighbor.id.clone(),
                        similarity,
                        temporal_distance: 0.0, // Will be computed if needed
                        field_value: neighbor.what.clone(), // Placeholder
                        weight: similarity,
                    })
                } else {
                    None
                }
            })
            .collect();

        // Sort by similarity (descending) and take top max_neighbors
        evidence.sort_by(|a, b| {
            b.similarity
                .partial_cmp(&a.similarity)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        evidence.truncate(self.max_neighbors);

        // Apply rank-based decay to weights
        for (rank, neighbor) in evidence.iter_mut().enumerate() {
            let rank_decay = self.neighbor_decay.powi(rank as i32);
            neighbor.weight = neighbor.similarity * rank_decay;
        }

        evidence
    }

    /// Collect all field names from neighbors
    fn collect_field_names(_neighbors: &[NeighborEvidence]) -> Vec<String> {
        // For now, we support basic fields: "what", "where", "who"
        // In a full implementation, this would extract from episode metadata
        vec!["what".to_string(), "where".to_string(), "who".to_string()]
    }

    /// Reconstruct a single field from neighbors
    fn reconstruct_single_field(
        _field_name: &str,
        neighbors: &[NeighborEvidence],
    ) -> Option<ReconstructedField> {
        if neighbors.is_empty() {
            return None;
        }

        // Collect weighted field values
        let field_values: Vec<(String, f32)> = neighbors
            .iter()
            .map(|n| (n.field_value.clone(), n.weight))
            .collect();

        // Compute consensus value (most weighted value)
        let mut value_weights: HashMap<String, f32> = HashMap::new();
        for (value, weight) in &field_values {
            *value_weights.entry(value.clone()).or_insert(0.0) += weight;
        }

        let consensus_value = value_weights
            .iter()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(val, _)| val.clone())?;

        // Compute confidence
        let confidence = Self::compute_field_confidence(&field_values);

        // Determine source based on confidence
        let source = if confidence.raw() < 0.3 {
            MemorySource::Imagined
        } else {
            MemorySource::Reconstructed
        };

        Some(ReconstructedField {
            value: consensus_value,
            confidence,
            source,
            evidence: neighbors.to_vec(),
        })
    }
}

impl Default for FieldReconstructor {
    fn default() -> Self {
        Self {
            temporal_window: Duration::from_secs(3600), // 1 hour
            similarity_threshold: 0.7,
            max_neighbors: 5,
            neighbor_decay: 0.8,
        }
    }
}

/// Reconstructed field with provenance tracking
#[derive(Debug, Clone)]
pub struct ReconstructedField {
    /// Reconstructed value for the field
    pub value: String,

    /// Confidence in reconstruction (0.0-1.0)
    pub confidence: Confidence,

    /// Source of reconstruction
    pub source: MemorySource,

    /// Evidence from contributing neighbors
    pub evidence: Vec<NeighborEvidence>,
}

/// Evidence from a single temporal neighbor
#[derive(Debug, Clone)]
pub struct NeighborEvidence {
    /// Episode ID contributing evidence
    pub episode_id: String,

    /// Similarity to partial episode (0.0-1.0)
    pub similarity: f32,

    /// Temporal distance (seconds)
    pub temporal_distance: f64,

    /// Field value from this neighbor
    pub field_value: String,

    /// Evidence weight (similarity * temporal_decay)
    pub weight: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_field_reconstructor_default() {
        let reconstructor = FieldReconstructor::new();
        assert_eq!(reconstructor.temporal_window, Duration::from_secs(3600));
        assert!((reconstructor.similarity_threshold - 0.7).abs() < 1e-6);
        assert_eq!(reconstructor.max_neighbors, 5);
        assert!((reconstructor.neighbor_decay - 0.8).abs() < 1e-6);
    }

    #[test]
    fn test_compute_field_confidence_empty() {
        let confidence = FieldReconstructor::compute_field_confidence(&[]);
        assert!((confidence.raw() - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_compute_field_confidence_unanimous() {
        let field_values = vec![
            ("morning".to_string(), 0.9),
            ("morning".to_string(), 0.8),
            ("morning".to_string(), 0.7),
        ];
        let confidence = FieldReconstructor::compute_field_confidence(&field_values);
        // All agree, so confidence should be 1.0
        assert!((confidence.raw() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_compute_field_confidence_split() {
        let field_values = vec![
            ("morning".to_string(), 0.6),
            ("morning".to_string(), 0.6),
            ("evening".to_string(), 0.8),
        ];
        let confidence = FieldReconstructor::compute_field_confidence(&field_values);
        // Split vote: morning has 1.2, evening has 0.8, total 2.0
        // Confidence = 1.2 / 2.0 = 0.6
        assert!((confidence.raw() - 0.6).abs() < 1e-2);
    }
}
