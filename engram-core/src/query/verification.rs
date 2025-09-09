//! Query verification and validation

use crate::{Confidence, Cue, Episode};

/// Verify query results meet confidence thresholds and constraints
pub struct QueryVerifier;

impl QueryVerifier {
    /// Verify that query results meet the specified confidence threshold
    pub fn verify_confidence_threshold(
        results: &[(Episode, Confidence)],
        threshold: Confidence,
    ) -> bool {
        results
            .iter()
            .all(|(_, confidence)| *confidence >= threshold)
    }

    /// Verify that query results are properly ordered by confidence
    pub fn verify_confidence_ordering(results: &[(Episode, Confidence)]) -> bool {
        results.windows(2).all(|pair| pair[0].1 >= pair[1].1)
    }

    /// Verify that cue constraints are satisfied
    pub fn verify_cue_constraints(cue: &Cue, results: &[(Episode, Confidence)]) -> bool {
        // Check result count limits
        if results.len() > cue.max_results {
            return false;
        }

        // Check confidence threshold
        Self::verify_confidence_threshold(results, cue.result_threshold)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Confidence, Episode};
    use chrono::Utc;

    #[test]
    fn test_confidence_verification() {
        let high_conf_episode = Episode::new(
            "high".to_string(),
            Utc::now(),
            "High confidence test episode".to_string(),
            [0.5f32; 768],
            Confidence::HIGH,
        );

        let results = vec![
            (high_conf_episode.clone(), Confidence::HIGH),
            (high_conf_episode, Confidence::MEDIUM),
        ];

        assert!(QueryVerifier::verify_confidence_threshold(
            &results,
            Confidence::MEDIUM
        ));
        assert!(!QueryVerifier::verify_confidence_threshold(
            &results,
            Confidence::CERTAIN
        ));
    }

    #[test]
    fn test_ordering_verification() {
        let episode = Episode::new(
            "test".to_string(),
            Utc::now(),
            "Test episode for ordering".to_string(),
            [0.5f32; 768],
            Confidence::HIGH,
        );

        let ordered_results = vec![
            (episode.clone(), Confidence::CERTAIN),
            (episode.clone(), Confidence::HIGH),
            (episode, Confidence::MEDIUM),
        ];

        assert!(QueryVerifier::verify_confidence_ordering(&ordered_results));

        let unordered_results = vec![
            (ordered_results[0].0.clone(), Confidence::MEDIUM),
            (ordered_results[1].0.clone(), Confidence::HIGH),
        ];

        assert!(!QueryVerifier::verify_confidence_ordering(
            &unordered_results
        ));
    }
}
