//! Confidence-aware distance metrics for cognitive search

use crate::Confidence;

/// Compute confidence-weighted distance between vectors
pub fn confidence_weighted_distance(
    distance: f32,
    confidence_a: Confidence,
    confidence_b: Confidence,
) -> f32 {
    // Combine confidences using logical AND (prevents overconfidence)
    let combined_confidence = confidence_a.and(confidence_b);

    // Weight distance by inverse confidence (high confidence = lower effective distance)
    distance * (1.0 - combined_confidence.raw())
}

/// Apply temporal boost to confidence based on recency
pub fn apply_temporal_boost(
    base_confidence: Confidence,
    age_seconds: f64,
    boost_factor: f32,
) -> Confidence {
    // Exponential decay with configurable boost
    let decay = (-age_seconds / 3600.0).exp() as f32; // Decay over hours
    let boosted = base_confidence.raw() * (1.0 + boost_factor * decay);

    Confidence::exact(boosted)
}

/// Combine multiple confidence scores with diversity weighting
pub fn combine_diverse_confidences(
    confidences: &[Confidence],
    diversity_scores: &[f32],
) -> Confidence {
    if confidences.is_empty() {
        return Confidence::NONE;
    }

    let mut weighted_sum = 0.0;
    let mut weight_sum = 0.0;

    for (confidence, diversity) in confidences.iter().zip(diversity_scores.iter()) {
        let weight = diversity.max(0.1); // Minimum weight to avoid division issues
        weighted_sum += confidence.raw() * weight;
        weight_sum += weight;
    }

    if weight_sum > 0.0 {
        Confidence::exact(weighted_sum / weight_sum)
    } else {
        Confidence::MEDIUM
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_confidence_weighted_distance() {
        let distance = 0.5;
        let high_confidence = Confidence::HIGH;
        let low_confidence = Confidence::LOW;

        // High confidence should reduce effective distance
        let weighted_high =
            confidence_weighted_distance(distance, high_confidence, high_confidence);
        assert!(weighted_high < distance);

        // Low confidence should increase effective distance
        let weighted_low = confidence_weighted_distance(distance, low_confidence, low_confidence);
        assert!(weighted_low > weighted_high);
    }

    #[test]
    fn test_temporal_boost() {
        let base = Confidence::MEDIUM;

        // Recent memories should get boosted
        let recent = apply_temporal_boost(base, 60.0, 0.5); // 1 minute old
        assert!(recent.raw() > base.raw());

        // Old memories should not get boosted
        let old = apply_temporal_boost(base, 86400.0, 0.5); // 1 day old
        assert!(old.raw() <= base.raw());
    }
}
