use crate::{Cue, CueType, compute};

/// Strategy used to combine multiple cues into seed embeddings
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CueAggregationStrategy {
    /// Simple average of all embeddings
    Average,
    /// Average weighted by confidence values
    WeightedAverage,
    /// Attention-based weighting using centroid similarity
    AttentionWeighted,
}

/// Aggregated embedding ready for seeding along with its weight
#[derive(Debug, Clone)]
pub struct AggregatedCue {
    /// Aggregated embedding vector
    pub embedding: [f32; 768],
    /// Weight of this embedding in seeding
    pub weight: f32,
}

/// Helper that transforms one or more cues into embeddings for seeding
pub struct MultiCueAggregator {
    vector_ops: &'static dyn compute::VectorOps,
}

impl Default for MultiCueAggregator {
    fn default() -> Self {
        Self::new()
    }
}

impl MultiCueAggregator {
    /// Creates a new multi-cue aggregator with default vector operations
    #[must_use]
    pub fn new() -> Self {
        Self {
            vector_ops: compute::get_vector_ops(),
        }
    }

    /// Aggregates multiple cues into embeddings using the specified strategy
    #[must_use]
    pub fn aggregate(&self, cues: &[Cue], strategy: CueAggregationStrategy) -> Vec<AggregatedCue> {
        let mut embeddings = Vec::new();
        let mut weights = Vec::new();

        for cue in cues {
            if let CueType::Embedding { vector, .. } = &cue.cue_type {
                embeddings.push(*vector);
                weights.push(cue.cue_confidence.raw().max(0.01));
            }
        }

        if embeddings.is_empty() {
            return Vec::new();
        }

        match strategy {
            CueAggregationStrategy::Average => {
                let references: Vec<&[f32; 768]> = embeddings.iter().collect();
                #[allow(clippy::cast_precision_loss)]
                let count = references.len() as f32;
                let uniform = vec![1.0 / count; references.len()];
                let centroid = self.vector_ops.weighted_average_768(&references, &uniform);
                vec![AggregatedCue {
                    embedding: centroid,
                    weight: 1.0,
                }]
            }
            CueAggregationStrategy::WeightedAverage => {
                let references: Vec<&[f32; 768]> = embeddings.iter().collect();
                let total = weights.iter().copied().sum::<f32>().max(1.0);
                let normalized: Vec<f32> = weights.iter().map(|w| *w / total).collect();
                let centroid = self
                    .vector_ops
                    .weighted_average_768(&references, &normalized);
                vec![AggregatedCue {
                    embedding: centroid,
                    weight: 1.0,
                }]
            }
            CueAggregationStrategy::AttentionWeighted => {
                let attention = self.attention_weights(&embeddings);
                embeddings
                    .into_iter()
                    .zip(attention)
                    .map(|(embedding, weight)| AggregatedCue { embedding, weight })
                    .collect()
            }
        }
    }

    fn attention_weights(&self, embeddings: &[[f32; 768]]) -> Vec<f32> {
        if embeddings.len() == 1 {
            return vec![1.0];
        }

        let references: Vec<&[f32; 768]> = embeddings.iter().collect();
        #[allow(clippy::cast_precision_loss)]
        let count = references.len() as f32;
        let uniform = vec![1.0 / count; references.len()];
        let centroid = self.vector_ops.weighted_average_768(&references, &uniform);

        let mut scores = Vec::with_capacity(embeddings.len());
        for embedding in embeddings {
            let score = self.vector_ops.dot_product_768(embedding, &centroid);
            scores.push(score);
        }

        softmax(&scores)
    }
}

fn softmax(values: &[f32]) -> Vec<f32> {
    if values.is_empty() {
        return Vec::new();
    }

    let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut exps = Vec::with_capacity(values.len());
    for value in values {
        exps.push((value - max).exp());
    }
    let sum: f32 = exps.iter().sum::<f32>().max(f32::EPSILON);
    exps.into_iter().map(|v| v / sum).collect()
}

#[cfg(test)]
mod tests {
    use super::{AggregatedCue, CueAggregationStrategy, MultiCueAggregator};
    use crate::{
        Confidence, Cue, CueType,
        numeric::{saturating_f32_from_f64, u64_to_f64},
    };
    use std::convert::TryFrom;

    fn sample_cue(id: &str, bias: f32) -> Cue {
        let mut embedding = [0.0f32; 768];
        embedding.iter_mut().enumerate().for_each(|(i, value)| {
            let idx = u64::try_from(i).unwrap_or(u64::MAX);
            let base = u64_to_f64(idx).mul_add(0.001, f64::from(bias));
            *value = saturating_f32_from_f64(base.sin());
        });
        Cue {
            id: id.to_string(),
            cue_type: CueType::Embedding {
                vector: embedding,
                threshold: Confidence::MEDIUM,
            },
            cue_confidence: Confidence::HIGH,
            result_threshold: Confidence::MEDIUM,
            max_results: 32,
        }
    }

    #[test]
    fn average_strategy_produces_single_embedding() {
        let cues = vec![sample_cue("a", 0.0), sample_cue("b", 0.1)];
        let aggregator = MultiCueAggregator::new();
        let aggregated = aggregator.aggregate(&cues, CueAggregationStrategy::Average);
        assert_eq!(aggregated.len(), 1);
        assert!(matches!(aggregated.first(), Some(AggregatedCue { .. })));
    }

    #[test]
    fn attention_strategy_returns_per_cue_weights() {
        let cues = vec![
            sample_cue("a", 0.0),
            sample_cue("b", 0.5),
            sample_cue("c", 1.0),
        ];
        let aggregator = MultiCueAggregator::new();
        let aggregated = aggregator.aggregate(&cues, CueAggregationStrategy::AttentionWeighted);
        assert_eq!(aggregated.len(), cues.len());
        let weight_sum: f32 = aggregated.iter().map(|entry| entry.weight).sum();
        assert!((weight_sum - 1.0).abs() < 1e-3);
    }
}
