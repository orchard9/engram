//! Confidence aggregation across multiple activation paths.
//!
//! Implements probabilistically-sound combination of confidence values when
//! several spreading paths converge on the same memory. The aggregator applies
//! hop-dependent decay, tier reliability weightings, and maximum-likelihood
//! estimation to maintain mathematically valid probabilities.

use crate::Confidence;
use crate::activation::storage_aware::StorageTier;

/// Aggregates confidence contributions from multiple activation paths.
#[derive(Debug, Clone)]
pub struct ConfidenceAggregator {
    decay_rate: f32,
    min_confidence: Confidence,
    max_paths: usize,
}

impl ConfidenceAggregator {
    /// Create a new aggregator with the provided decay rate, minimum confidence threshold, and maximum paths.
    #[must_use]
    pub const fn new(decay_rate: f32, min_confidence: Confidence, max_paths: usize) -> Self {
        Self {
            decay_rate,
            min_confidence,
            max_paths,
        }
    }

    /// Aggregate the provided paths into a single confidence outcome.
    #[must_use]
    pub fn aggregate_paths(&self, paths: &[ConfidencePath]) -> ConfidenceAggregationOutcome {
        if paths.is_empty() {
            return ConfidenceAggregationOutcome::empty(self.min_confidence);
        }

        let mut evaluated: Vec<PathEvaluation> = paths
            .iter()
            .map(|path| PathEvaluation::new(path.clone(), self.decay_rate))
            .filter(|candidate| candidate.probability.is_finite())
            .collect();

        if evaluated.is_empty() {
            return ConfidenceAggregationOutcome::empty(self.min_confidence);
        }

        evaluated.sort_by(|a, b| b.probability.total_cmp(&a.probability));

        let mut contributions = Vec::new();
        let mut log_one_minus_total = 0.0f64;

        for evaluation in evaluated.into_iter().take(self.max_paths) {
            if evaluation.probability < self.min_confidence.raw() {
                continue;
            }

            let probability = evaluation.probability.clamp(0.0, 1.0);
            let complement = (1.0 - f64::from(probability)).clamp(f64::MIN_POSITIVE, 1.0);
            log_one_minus_total += complement.ln();

            contributions.push(ConfidenceContribution {
                tier: evaluation.path.source_tier,
                hop_count: evaluation.path.hop_count,
                original: evaluation.path.confidence,
                decayed: Confidence::from_raw(probability),
                weight: evaluation.path.path_weight,
            });
        }

        if contributions.is_empty() {
            return ConfidenceAggregationOutcome::empty(self.min_confidence);
        }

        #[allow(clippy::cast_precision_loss)]
        #[allow(clippy::cast_possible_truncation)]
        #[allow(clippy::cast_sign_loss)]
        let combined = (1.0 - log_one_minus_total.exp()).clamp(0.0, 1.0) as f32;
        let aggregate = if combined < self.min_confidence.raw() {
            self.min_confidence
        } else {
            Confidence::from_raw(combined)
        };

        let skipped_paths = paths.len().saturating_sub(contributions.len());

        ConfidenceAggregationOutcome {
            aggregate,
            contributing_paths: contributions,
            skipped_paths,
        }
    }
}

/// Represents a single activation path contributing to the aggregated confidence.
#[derive(Debug, Clone)]
pub struct ConfidencePath {
    /// Confidence value for this path
    pub confidence: Confidence,
    /// Number of hops from the source node in activation spreading
    pub hop_count: u16,
    /// Storage tier where the source memory resides
    pub source_tier: StorageTier,
    /// Weight of this path in the overall confidence calculation
    pub path_weight: f32,
}

impl ConfidencePath {
    /// Create a new confidence path.
    #[must_use]
    pub const fn new(
        confidence: Confidence,
        hop_count: u16,
        source_tier: StorageTier,
        path_weight: f32,
    ) -> Self {
        Self {
            confidence,
            hop_count,
            source_tier,
            path_weight,
        }
    }

    /// Convenience constructor using default path weight of 1.0.
    #[must_use]
    pub const fn with_default_weight(
        confidence: Confidence,
        hop_count: u16,
        source_tier: StorageTier,
    ) -> Self {
        Self::new(confidence, hop_count, source_tier, 1.0)
    }
}

/// Output of the aggregation process containing the combined confidence and contributing paths.
#[derive(Debug, Clone)]
pub struct ConfidenceAggregationOutcome {
    /// Final aggregated confidence value
    pub aggregate: Confidence,
    /// List of paths that contributed to the aggregate confidence
    pub contributing_paths: Vec<ConfidenceContribution>,
    /// Number of paths skipped due to low contribution
    pub skipped_paths: usize,
}

impl ConfidenceAggregationOutcome {
    const fn empty(min_confidence: Confidence) -> Self {
        Self {
            aggregate: min_confidence,
            contributing_paths: Vec::new(),
            skipped_paths: 0,
        }
    }

    /// Whether any paths contributed to the aggregated confidence.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.contributing_paths.is_empty()
    }
}

/// Detailed information about an individual path’s contribution.
#[derive(Debug, Clone)]
pub struct ConfidenceContribution {
    /// Storage tier of the contributing memory
    pub tier: StorageTier,
    /// Number of hops from source in the activation path
    pub hop_count: u16,
    /// Original confidence before decay
    pub original: Confidence,
    /// Confidence after applying decay
    pub decayed: Confidence,
    /// Weight of this contribution in the aggregate
    pub weight: f32,
}

#[derive(Debug, Clone)]
struct PathEvaluation {
    path: ConfidencePath,
    probability: f32,
}

impl PathEvaluation {
    fn new(path: ConfidencePath, decay_rate: f32) -> Self {
        let decay_factor = (-decay_rate * f32::from(path.hop_count)).exp();
        let tier_weight = path.source_tier.confidence_factor();
        let probability =
            (path.confidence.raw() * path.path_weight * tier_weight * decay_factor).clamp(0.0, 1.0);

        Self { path, probability }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    fn aggregator() -> ConfidenceAggregator {
        ConfidenceAggregator::new(0.35, Confidence::from_raw(0.001), 8)
    }

    #[test]
    fn single_path_returns_same_confidence() {
        let aggregator = aggregator();
        let path = ConfidencePath::with_default_weight(Confidence::HIGH, 0, StorageTier::Hot);
        let outcome = aggregator.aggregate_paths(&[path]);

        assert!(!outcome.is_empty());
        assert!((outcome.aggregate.raw() - Confidence::HIGH.raw()).abs() < 1e-6);
        assert_eq!(outcome.contributing_paths.len(), 1);
    }

    #[test]
    fn multiple_paths_increase_confidence() {
        let aggregator = aggregator();
        let path_a =
            ConfidencePath::with_default_weight(Confidence::from_raw(0.6), 1, StorageTier::Hot);
        let path_b =
            ConfidencePath::with_default_weight(Confidence::from_raw(0.5), 1, StorageTier::Warm);

        let outcome = aggregator.aggregate_paths(&[path_a, path_b]);
        let decay = (-aggregator.decay_rate * 1.0).exp();
        let hot_prob = 0.6 * decay * StorageTier::Hot.confidence_factor();
        let warm_prob = 0.5 * decay * StorageTier::Warm.confidence_factor();
        let expected = (1.0 - hot_prob).mul_add(-(1.0 - warm_prob), 1.0);

        assert!((outcome.aggregate.raw() - expected).abs() < 1e-6);
        assert_eq!(outcome.contributing_paths.len(), 2);
    }

    #[test]
    fn hop_decay_reduces_confidence() {
        let aggregator = aggregator();
        let near_path =
            ConfidencePath::with_default_weight(Confidence::from_raw(0.4), 1, StorageTier::Hot);
        let far_path =
            ConfidencePath::with_default_weight(Confidence::from_raw(0.4), 5, StorageTier::Hot);

        let near = aggregator.aggregate_paths(&[near_path]).aggregate.raw();
        let far = aggregator.aggregate_paths(&[far_path]).aggregate.raw();

        assert!(far < near);
    }

    #[test]
    fn respects_max_paths_limit() {
        let aggregator = ConfidenceAggregator::new(0.2, Confidence::from_raw(0.0), 1);
        let paths = vec![
            ConfidencePath::with_default_weight(Confidence::from_raw(0.3), 1, StorageTier::Hot),
            ConfidencePath::with_default_weight(Confidence::from_raw(0.9), 1, StorageTier::Hot),
        ];

        let outcome = aggregator.aggregate_paths(&paths);
        assert_eq!(outcome.contributing_paths.len(), 1);
        assert!(outcome.aggregate.raw() <= 0.9);
    }

    proptest! {
        #[test]
        fn aggregated_confidence_within_bounds(raw in 0.0f32..1.0, hop in 0u16..6, weight in 0.1f32..1.5) {
            let aggregator = aggregator();
            let path = ConfidencePath::new(Confidence::from_raw(raw), hop, StorageTier::Hot, weight);
            let outcome = aggregator.aggregate_paths(&[path]);

            prop_assert!(outcome.aggregate.raw() >= 0.0 && outcome.aggregate.raw() <= 1.0);
        }
    }
}
