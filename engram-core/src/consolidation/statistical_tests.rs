//! Statistical significance testing for pattern detection.
//!
//! This module implements statistical tests to filter spurious patterns
//! and ensure only statistically significant patterns are retained.

use super::EpisodicPattern;

/// Minimum p-value threshold for pattern significance (p < 0.01)
pub const SIGNIFICANCE_THRESHOLD: f64 = 0.01;

/// Statistical significance filter for patterns
pub struct StatisticalFilter {
    /// P-value threshold
    threshold: f64,
}

impl Default for StatisticalFilter {
    fn default() -> Self {
        Self {
            threshold: SIGNIFICANCE_THRESHOLD,
        }
    }
}

impl StatisticalFilter {
    /// Create a new statistical filter with custom threshold
    #[must_use]
    pub const fn new(threshold: f64) -> Self {
        Self { threshold }
    }

    /// Filter patterns by statistical significance
    ///
    /// Returns only patterns that pass the significance test (p < threshold)
    #[must_use]
    pub fn filter_patterns(&self, patterns: &[EpisodicPattern]) -> Vec<EpisodicPattern> {
        patterns
            .iter()
            .filter(|pattern| self.is_significant(pattern))
            .cloned()
            .collect()
    }

    /// Test if a pattern is statistically significant
    fn is_significant(&self, pattern: &EpisodicPattern) -> bool {
        // Chi-square test for pattern occurrence frequency
        let p_value = Self::chi_square_test(pattern);
        p_value < self.threshold
    }

    /// Chi-square test for pattern significance
    ///
    /// Tests if pattern occurrence is significantly different from random
    fn chi_square_test(pattern: &EpisodicPattern) -> f64 {
        let observed = pattern.occurrence_count as f64;

        // Expected frequency under null hypothesis (random distribution)
        // Using pattern strength as a proxy for expected frequency
        let expected = if pattern.strength > 0.0 {
            // Higher strength = lower expected random occurrence
            2.0 / f64::from(pattern.strength)
        } else {
            10.0 // Default expected frequency
        };

        // Chi-square statistic: Σ((observed - expected)² / expected)
        let chi_square = (observed - expected).powi(2) / expected;

        // Convert chi-square to p-value (simplified)
        // For 1 degree of freedom, approximate p-value calculation
        Self::chi_square_to_p_value(chi_square, 1)
    }

    /// Convert chi-square statistic to p-value
    ///
    /// Simplified approximation for degrees of freedom = 1
    fn chi_square_to_p_value(chi_square: f64, _df: usize) -> f64 {
        // Simplified approximation: p ≈ e^(-χ²/2)
        // This is approximate but sufficient for filtering
        (-chi_square / 2.0).exp()
    }

    /// Compute mutual information between pattern features
    ///
    /// Measures feature dependency strength
    #[must_use]
    pub fn mutual_information(pattern: &EpisodicPattern) -> f64 {
        // Simplified MI calculation based on feature count and pattern strength
        let feature_count = pattern.features.len() as f64;
        let strength = f64::from(pattern.strength);

        if feature_count < 2.0 {
            return 0.0; // No mutual information with < 2 features
        }

        // MI approximation: higher strength and more features = higher MI
        // Use ln_1p for better numerical accuracy
        strength * (feature_count / 10.0).ln_1p()
    }

    /// Compute likelihood ratio vs random baseline
    ///
    /// Returns log likelihood ratio for pattern vs random
    #[must_use]
    pub fn likelihood_ratio(pattern: &EpisodicPattern) -> f64 {
        let n = pattern.occurrence_count as f64;
        let strength = f64::from(pattern.strength);

        if n == 0.0 || strength == 0.0 {
            return 0.0;
        }

        // Log likelihood ratio: log(P(pattern) / P(random))
        // Pattern probability proportional to strength
        // Random probability = 1/n (uniform)
        let log_pattern_prob = strength.ln();
        let log_random_prob = -(n.ln());

        log_pattern_prob - log_random_prob
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;

    fn create_test_pattern(occurrence_count: usize, strength: f32) -> EpisodicPattern {
        EpisodicPattern {
            id: format!("test_pattern_{occurrence_count}"),
            embedding: [0.5; 768],
            source_episodes: (0..occurrence_count).map(|i| format!("ep_{i}")).collect(),
            strength,
            features: Vec::new(),
            first_occurrence: Utc::now(),
            last_occurrence: Utc::now(),
            occurrence_count,
        }
    }

    #[test]
    fn test_statistical_filter_creation() {
        let filter = StatisticalFilter::default();
        assert!((filter.threshold - SIGNIFICANCE_THRESHOLD).abs() < f64::EPSILON);
    }

    #[test]
    fn test_significance_filtering() {
        let filter = StatisticalFilter::default();

        // Strong pattern with many occurrences (should pass)
        let strong_pattern = create_test_pattern(10, 0.9);

        // Weak pattern with few occurrences (may fail)
        let weak_pattern = create_test_pattern(2, 0.3);

        let patterns = vec![strong_pattern, weak_pattern];
        let filtered = filter.filter_patterns(&patterns);

        // At least the strong pattern should pass
        assert!(!filtered.is_empty());
    }

    #[test]
    fn test_chi_square_calculation() {
        let pattern = create_test_pattern(10, 0.8);

        let p_value = StatisticalFilter::chi_square_test(&pattern);

        // P-value should be between 0 and 1
        assert!(p_value >= 0.0);
        assert!(p_value <= 1.0);
    }

    #[test]
    fn test_mutual_information() {
        let mut pattern = create_test_pattern(5, 0.7);

        // No features = 0 MI
        let mi_zero = StatisticalFilter::mutual_information(&pattern);
        assert!((mi_zero - 0.0).abs() < f64::EPSILON);

        // Add features
        pattern
            .features
            .push(crate::consolidation::PatternFeature::TemporalSequence {
                interval: chrono::Duration::seconds(60),
            });
        pattern
            .features
            .push(crate::consolidation::PatternFeature::ConceptualTheme {
                theme: "test".to_string(),
            });

        let mi = StatisticalFilter::mutual_information(&pattern);
        assert!(mi > 0.0); // Should have positive MI with features
    }

    #[test]
    fn test_likelihood_ratio() {
        let strong_pattern = create_test_pattern(10, 0.9);
        let weak_pattern = create_test_pattern(2, 0.3);

        let lr_strong = StatisticalFilter::likelihood_ratio(&strong_pattern);
        let lr_weak = StatisticalFilter::likelihood_ratio(&weak_pattern);

        // Stronger pattern should have higher likelihood ratio
        assert!(lr_strong > lr_weak);
    }
}
