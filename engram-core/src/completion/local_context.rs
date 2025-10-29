//! Local context extractor for temporal and spatial proximity.
//!
//! Implements recency weighting with quadratic decay and spatial proximity
//! filtering for episode neighbor selection. Follows empirical findings
//! that temporal proximity is the strongest cue within 1-hour window.

use crate::Episode;
use chrono::{DateTime, Utc};
use std::time::Duration;

/// Local context extractor for temporal and spatial proximity
pub struct LocalContextExtractor {
    /// Temporal window for context (default: 1 hour before/after)
    temporal_window: Duration,

    /// Spatial radius for proximity (default: 100 meters, if location available)
    #[allow(dead_code)] // Reserved for future spatial filtering implementation
    spatial_radius: f32,

    /// Recency weighting exponent (default: 2.0 for quadratic decay)
    recency_exponent: f32,
}

impl LocalContextExtractor {
    /// Create new local context extractor with default parameters
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Create extractor with custom parameters
    #[must_use]
    pub const fn with_params(
        temporal_window: Duration,
        spatial_radius: f32,
        recency_exponent: f32,
    ) -> Self {
        Self {
            temporal_window,
            spatial_radius,
            recency_exponent,
        }
    }

    /// Extract temporal neighbors within window
    #[must_use]
    pub fn temporal_neighbors(
        &self,
        anchor_time: DateTime<Utc>,
        episodes: &[Episode],
    ) -> Vec<TemporalNeighbor> {
        let mut neighbors: Vec<TemporalNeighbor> = episodes
            .iter()
            .filter_map(|ep| {
                let time_diff = if ep.when > anchor_time {
                    ep.when.signed_duration_since(anchor_time)
                } else {
                    anchor_time.signed_duration_since(ep.when)
                };

                // Convert to Duration for comparison
                if let Ok(duration) = time_diff.to_std() {
                    if duration <= self.temporal_window {
                        let recency_weight = self.recency_weight(duration);
                        return Some(TemporalNeighbor {
                            episode: ep.clone(),
                            temporal_distance: duration,
                            recency_weight,
                        });
                    }
                }
                None
            })
            .collect();

        // Sort by temporal distance (closest first)
        neighbors.sort_by(|a, b| a.temporal_distance.cmp(&b.temporal_distance));

        neighbors
    }

    /// Extract spatially proximate episodes (if location metadata available)
    #[must_use]
    pub fn spatial_neighbors(
        anchor_location: Option<&str>,
        episodes: &[Episode],
    ) -> Vec<SpatialNeighbor> {
        let Some(anchor_loc) = anchor_location else {
            return vec![];
        };

        episodes
            .iter()
            .filter_map(|ep| {
                if let Some(ref ep_loc) = ep.where_location {
                    // Simple string matching for now
                    // In production, would compute actual geographic distance
                    if ep_loc == anchor_loc {
                        return Some(SpatialNeighbor {
                            episode: ep.clone(),
                            spatial_distance: 0.0, // Exact match
                            proximity_weight: 1.0,
                        });
                    }
                }
                None
            })
            .collect()
    }

    /// Compute recency weight for temporal distance
    ///
    /// Weight = (1 - normalized_distance) ^ recency_exponent
    /// Recent episodes have weight near 1.0, distant episodes near 0.0
    #[must_use]
    pub fn recency_weight(&self, temporal_distance: Duration) -> f32 {
        // Normalize distance to [0, 1] based on temporal window
        let normalized_distance =
            temporal_distance.as_secs_f32() / self.temporal_window.as_secs_f32();
        let normalized_distance = normalized_distance.clamp(0.0, 1.0);

        // Apply quadratic decay: weight = (1 - normalized_distance)^2
        let weight = (1.0 - normalized_distance).powf(self.recency_exponent);
        weight.clamp(0.0, 1.0)
    }

    /// Merge temporal and spatial context with adaptive weighting
    #[must_use]
    pub fn merge_contexts(
        temporal: Vec<TemporalNeighbor>,
        spatial: Vec<SpatialNeighbor>,
    ) -> Vec<ContextEvidence> {
        let mut evidence_map: std::collections::HashMap<String, ContextEvidence> =
            std::collections::HashMap::new();

        // Add temporal evidence
        for t_neighbor in temporal {
            evidence_map.insert(
                t_neighbor.episode.id.clone(),
                ContextEvidence {
                    episode: t_neighbor.episode,
                    combined_weight: t_neighbor.recency_weight,
                    temporal_contribution: t_neighbor.recency_weight,
                    spatial_contribution: 0.0,
                },
            );
        }

        // Merge with spatial evidence
        for s_neighbor in spatial {
            evidence_map
                .entry(s_neighbor.episode.id.clone())
                .and_modify(|e| {
                    // Combine weights: average of temporal and spatial
                    e.spatial_contribution = s_neighbor.proximity_weight;
                    e.combined_weight =
                        f32::midpoint(e.temporal_contribution, e.spatial_contribution);
                })
                .or_insert_with(|| ContextEvidence {
                    episode: s_neighbor.episode,
                    combined_weight: s_neighbor.proximity_weight,
                    temporal_contribution: 0.0,
                    spatial_contribution: s_neighbor.proximity_weight,
                });
        }

        // Convert to vector and sort by combined weight
        let mut evidence: Vec<ContextEvidence> = evidence_map.into_values().collect();
        evidence.sort_by(|a, b| {
            b.combined_weight
                .partial_cmp(&a.combined_weight)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        evidence
    }
}

impl Default for LocalContextExtractor {
    fn default() -> Self {
        Self {
            temporal_window: Duration::from_secs(3600), // 1 hour
            spatial_radius: 100.0,                      // 100 meters
            recency_exponent: 2.0,                      // Quadratic decay
        }
    }
}

/// Temporal neighbor with recency weighting
#[derive(Debug, Clone)]
pub struct TemporalNeighbor {
    /// The episode that is a temporal neighbor
    pub episode: Episode,
    /// Temporal distance from anchor time
    pub temporal_distance: Duration,
    /// Computed recency weight based on distance
    pub recency_weight: f32,
}

/// Spatial neighbor with proximity weighting
#[derive(Debug, Clone)]
pub struct SpatialNeighbor {
    /// The episode that is a spatial neighbor
    pub episode: Episode,
    /// Spatial distance from anchor location (meters)
    pub spatial_distance: f32,
    /// Computed proximity weight based on distance
    pub proximity_weight: f32,
}

/// Combined context evidence from temporal and spatial
#[derive(Debug, Clone)]
pub struct ContextEvidence {
    /// The episode providing context evidence
    pub episode: Episode,
    /// Combined weight from temporal and spatial contributions
    pub combined_weight: f32,
    /// Contribution from temporal proximity
    pub temporal_contribution: f32,
    /// Contribution from spatial proximity
    pub spatial_contribution: f32,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Confidence;

    fn create_test_episode(id: &str, when: DateTime<Utc>, what: &str) -> Episode {
        Episode {
            id: id.to_string(),
            when,
            where_location: None,
            who: None,
            what: what.to_string(),
            embedding: [0.0; 768],
            embedding_provenance: None,
            encoding_confidence: Confidence::exact(0.8),
            vividness_confidence: Confidence::exact(0.7),
            reliability_confidence: Confidence::exact(0.8),
            last_recall: when,
            recall_count: 0,
            decay_rate: 0.05,
            decay_function: None,
            metadata: std::collections::HashMap::new(),
        }
    }

    #[test]
    fn test_local_context_extractor_default() {
        let extractor = LocalContextExtractor::new();
        assert_eq!(extractor.temporal_window, Duration::from_secs(3600));
        assert!((extractor.spatial_radius - 100.0).abs() < 1e-6);
        assert!((extractor.recency_exponent - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_recency_weight_at_zero() {
        let extractor = LocalContextExtractor::new();
        let weight = extractor.recency_weight(Duration::from_secs(0));
        // At zero distance, weight should be 1.0
        assert!((weight - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_recency_weight_at_max() {
        let extractor = LocalContextExtractor::new();
        let weight = extractor.recency_weight(Duration::from_secs(3600)); // At window edge
        // At max distance, weight should be near 0.0
        assert!(weight < 0.1);
    }

    #[test]
    fn test_recency_weight_monotonic_decrease() {
        let extractor = LocalContextExtractor::new();
        let weight_10min = extractor.recency_weight(Duration::from_secs(600)); // 10 min
        let weight_30min = extractor.recency_weight(Duration::from_secs(1800)); // 30 min
        let weight_60min = extractor.recency_weight(Duration::from_secs(3600)); // 60 min

        // Weights should decrease monotonically
        assert!(weight_10min > weight_30min);
        assert!(weight_30min > weight_60min);
    }

    #[test]
    fn test_temporal_neighbors_filtering() {
        let extractor = LocalContextExtractor::new();
        let anchor = Utc::now();

        let episodes = vec![
            create_test_episode("ep1", anchor - chrono::Duration::minutes(10), "recent"),
            create_test_episode("ep2", anchor - chrono::Duration::hours(2), "old"),
            create_test_episode("ep3", anchor - chrono::Duration::minutes(30), "medium"),
        ];

        let neighbors = extractor.temporal_neighbors(anchor, &episodes);

        // Only ep1 and ep3 should be included (within 1 hour)
        assert_eq!(neighbors.len(), 2);
        assert_eq!(neighbors[0].episode.id, "ep1"); // Closest first
        assert_eq!(neighbors[1].episode.id, "ep3");
    }

    #[test]
    fn test_temporal_neighbors_sorting() {
        let extractor = LocalContextExtractor::new();
        let anchor = Utc::now();

        let episodes = vec![
            create_test_episode("ep1", anchor - chrono::Duration::minutes(30), "medium"),
            create_test_episode("ep2", anchor - chrono::Duration::minutes(5), "recent"),
            create_test_episode("ep3", anchor - chrono::Duration::minutes(50), "older"),
        ];

        let neighbors = extractor.temporal_neighbors(anchor, &episodes);

        // Should be sorted by distance (closest first)
        assert_eq!(neighbors.len(), 3);
        assert_eq!(neighbors[0].episode.id, "ep2");
        assert_eq!(neighbors[1].episode.id, "ep1");
        assert_eq!(neighbors[2].episode.id, "ep3");
    }
}
