//! Property-based tests for Confidence operations
//!
//! These tests verify invariants that developers naturally expect when working with
//! probabilistic confidence values, following cognitive-friendly property specifications.

#![allow(missing_docs)]
#![allow(clippy::float_cmp)]

use engram_core::Confidence;
use proptest::prelude::*;

/// Generator for valid confidence values [0.0, 1.0]
/// This matches the cognitive expectation that confidence is always a valid probability
fn confidence_strategy() -> impl Strategy<Value = Confidence> {
    (0.0f32..=1.0f32).prop_map(Confidence::exact)
}

/// Generator for arbitrary f32 values (including invalid ones for robustness testing)
/// Tests that the type handles invalid inputs gracefully by clamping to [0,1]
fn arbitrary_f32_strategy() -> impl Strategy<Value = f32> {
    -1000.0f32..1000.0f32
}

/// Generator for weighted combination parameters
/// Tests realistic weighting scenarios developers would encounter
fn weight_strategy() -> impl Strategy<Value = f32> {
    0.0f32..10.0f32
}

/// Generator for frequency-based confidence construction (successes, total)
/// Matches how developers naturally think about confidence: "X out of Y cases"
fn frequency_strategy() -> impl Strategy<Value = (u32, u32)> {
    (1u32..=100u32).prop_flat_map(|total| (0u32..=total, Just(total)))
}

/// Generator for percentage values that developers commonly use
fn percentage_strategy() -> impl Strategy<Value = u8> {
    0u8..=150u8 // Include invalid percentages to test robustness
}

proptest! {
    /// Property: "Confidence values must always be valid probabilities"
    /// Natural expectation: Any confidence value I create should be between 0 and 1
    /// Mathematical invariant: ∀x. 0.0 ≤ Confidence::exact(x).raw() ≤ 1.0
    #[test]
    fn confidence_values_are_always_valid_probabilities(value in arbitrary_f32_strategy()) {
        let confidence = Confidence::exact(value);
        let raw = confidence.raw();

        // Cognitive expectation: "No matter what number I put in, I get a valid confidence"
        prop_assert!(raw >= 0.0, "Confidence must be non-negative: got {}", raw);
        prop_assert!(raw <= 1.0, "Confidence must not exceed certainty: got {}", raw);
        prop_assert!(!raw.is_nan(), "Confidence must be a real number, not NaN");
        prop_assert!(!raw.is_infinite(), "Confidence must be finite, not infinite");
    }

    /// Property: "Frequency-based construction matches intuitive probability"
    /// Natural expectation: "3 successes out of 5 attempts should give me 0.6 confidence"
    /// Mathematical invariant: Confidence::from_successes(s, t) == s/t (when t > 0)
    #[test]
    fn frequency_construction_matches_intuitive_probability((successes, total) in frequency_strategy()) {
        let confidence = Confidence::from_successes(successes, total);
        #[allow(clippy::cast_precision_loss)]
        let expected = successes as f32 / total as f32;

        // Cognitive expectation: "The math should work out like I expect"
        prop_assert!((confidence.raw() - expected).abs() < 1e-6);
        prop_assert!(confidence.raw() >= 0.0);
        prop_assert!(confidence.raw() <= 1.0);
    }

    /// Property: "Zero total cases should give zero confidence"
    /// Natural expectation: "If I have no data, I should have no confidence"
    #[test]
    fn zero_total_gives_zero_confidence(successes in 0u32..1000u32) {
        let confidence = Confidence::from_successes(successes, 0);
        prop_assert!((confidence.raw() - 0.0).abs() < 1e-6, "Zero total should always give zero confidence");
    }

    /// Property: "Percentage construction works like developers expect"
    /// Natural expectation: "80% should become 0.8 confidence"
    /// Edge case handling: Values over 100% should be clamped to 100%
    #[test]
    fn percentage_construction_is_intuitive(percent in percentage_strategy()) {
        let confidence = Confidence::from_percent(percent);
        let expected = f32::from(percent.min(100)) / 100.0;

        // Cognitive expectation: "Percentages should convert naturally to decimals"
        prop_assert!((confidence.raw() - expected).abs() < 1e-6);
        prop_assert!(confidence.raw() >= 0.0);
        prop_assert!(confidence.raw() <= 1.0);
    }

    /// Property: "Logical AND prevents conjunction fallacy"
    /// Natural expectation: "The chance of A AND B can't be higher than either A or B alone"
    /// Cognitive bias prevention: P(A ∧ B) ≤ min(P(A), P(B))
    #[test]
    fn logical_and_prevents_conjunction_fallacy(
        conf_a in confidence_strategy(),
        conf_b in confidence_strategy()
    ) {
        let and_result = conf_a.and(conf_b);
        let min_input = conf_a.raw().min(conf_b.raw());

        // Cognitive expectation: "Both things happening can't be more likely than either alone"
        prop_assert!(
            and_result.raw() <= min_input + f32::EPSILON,
            "AND result {} must not exceed minimum input {} (conjunction fallacy prevention)",
            and_result.raw(), min_input
        );

        // Mathematical expectation: Should equal the product for independent events
        let expected_product = conf_a.raw() * conf_b.raw();
        prop_assert!(
            (and_result.raw() - expected_product).abs() < f32::EPSILON,
            "AND should equal product: expected {}, got {}",
            expected_product, and_result.raw()
        );
    }

    /// Property: "Logical OR follows probability combination rules"
    /// Natural expectation: "A OR B should be at least as likely as either A or B"
    /// Mathematical invariant: P(A ∨ B) = P(A) + P(B) - P(A ∧ B)
    #[test]
    fn logical_or_follows_probability_rules(
        conf_a in confidence_strategy(),
        conf_b in confidence_strategy()
    ) {
        let or_result = conf_a.or(conf_b);
        let max_input = conf_a.raw().max(conf_b.raw());

        // Cognitive expectation: "Either thing happening should be at least as likely as the more likely one"
        prop_assert!(
            or_result.raw() >= max_input - f32::EPSILON,
            "OR result {} must be at least as large as maximum input {}",
            or_result.raw(), max_input
        );

        // Mathematical expectation: P(A ∨ B) = P(A) + P(B) - P(A ∧ B)
        let expected = conf_a.raw().mul_add(-conf_b.raw(), conf_a.raw() + conf_b.raw());
        prop_assert!(
            (or_result.raw() - expected).abs() < f32::EPSILON,
            "OR should follow inclusion-exclusion: expected {}, got {}",
            expected, or_result.raw()
        );

        prop_assert!(or_result.raw() <= 1.0, "OR result must not exceed certainty");
    }

    /// Property: "Negation works like logical complement"
    /// Natural expectation: "If I'm 70% confident, I should be 30% confident in the opposite"
    /// Mathematical invariant: Confidence::exact(p).not().raw() == 1.0 - p
    #[test]
    fn negation_works_like_logical_complement(conf in confidence_strategy()) {
        let negated = conf.not();
        let expected = 1.0 - conf.raw();

        // Cognitive expectation: "Being less confident in something means being more confident in the opposite"
        prop_assert!(
            (negated.raw() - expected).abs() < f32::EPSILON,
            "Negation should be complement: {} -> {}, expected {}",
            conf.raw(), negated.raw(), expected
        );

        // Double negation should return to original (within floating point precision)
        let double_negated = negated.not();
        prop_assert!(
            (double_negated.raw() - conf.raw()).abs() < f32::EPSILON * 2.0,
            "Double negation should return to original: {} -> {} -> {}",
            conf.raw(), negated.raw(), double_negated.raw()
        );
    }

    /// Property: "Weighted combination preserves valid range"
    /// Natural expectation: "Combining two confidence values should give a valid confidence"
    /// Mathematical invariant: Result is weighted average, so must be between inputs
    #[test]
    fn weighted_combination_preserves_range(
        conf_a in confidence_strategy(),
        conf_b in confidence_strategy(),
        weight_a in weight_strategy(),
        weight_b in weight_strategy()
    ) {
        let combined = conf_a.combine_weighted(conf_b, weight_a, weight_b);

        // Cognitive expectation: "The result should be a reasonable middle ground"
        prop_assert!(combined.raw() >= 0.0, "Combined confidence must be non-negative");
        prop_assert!(combined.raw() <= 1.0, "Combined confidence must not exceed certainty");

        // If both weights are positive, result should be between the two inputs
        if weight_a > 0.0 && weight_b > 0.0 {
            let min_input = conf_a.raw().min(conf_b.raw());
            let max_input = conf_a.raw().max(conf_b.raw());
            prop_assert!(
                combined.raw() >= min_input - f32::EPSILON,
                "Combined result {} should be at least minimum input {}",
                combined.raw(), min_input
            );
            prop_assert!(
                combined.raw() <= max_input + f32::EPSILON,
                "Combined result {} should be at most maximum input {}",
                combined.raw(), max_input
            );
        }
    }

    /// Property: "Zero weights in combination return medium confidence"
    /// Natural expectation: "If I have no information, I should be moderately unsure"
    #[test]
    fn zero_weights_return_medium_confidence(
        conf_a in confidence_strategy(),
        conf_b in confidence_strategy()
    ) {
        let combined = conf_a.combine_weighted(conf_b, 0.0, 0.0);

        // Cognitive expectation: "No evidence means middle-ground confidence"
        prop_assert!((combined.raw() - Confidence::MEDIUM.raw()).abs() < 1e-6);
    }

    /// Property: "Overconfidence calibration reduces high confidence appropriately"
    /// Natural expectation: "High confidence should be reduced more than low confidence"
    /// Psychological basis: People tend to be overconfident, especially at high confidence levels
    #[test]
    fn overconfidence_calibration_reduces_high_confidence_more(conf in confidence_strategy()) {
        let calibrated = conf.calibrate_overconfidence();

        // Cognitive expectation: "Very high confidence should be reduced more than moderate confidence"
        if conf.raw() > 0.8 {
            prop_assert!(
                calibrated.raw() < conf.raw(),
                "High confidence {} should be reduced to {}",
                conf.raw(), calibrated.raw()
            );
            prop_assert!(
                calibrated.raw() >= conf.raw() * 0.8, // Shouldn't be reduced too drastically
                "High confidence reduction shouldn't be too extreme: {} -> {}",
                conf.raw(), calibrated.raw()
            );
        } else if conf.raw() > 0.6 {
            prop_assert!(
                calibrated.raw() <= conf.raw() + f32::EPSILON,
                "Medium-high confidence {} should be reduced or unchanged: {}",
                conf.raw(), calibrated.raw()
            );
        } else {
            // Low confidence should be unchanged or slightly increased
            prop_assert!(
                calibrated.raw() >= conf.raw() - f32::EPSILON,
                "Low confidence {} should be unchanged or increased: {}",
                conf.raw(), calibrated.raw()
            );
        }

        // All calibrated values should still be valid probabilities
        prop_assert!(calibrated.raw() >= 0.0);
        prop_assert!(calibrated.raw() <= 1.0);
    }

    /// Property: "High confidence classification matches developer intuition"
    /// Natural expectation: "Values above 0.7 should feel 'high confidence' to developers"
    #[test]
    fn high_confidence_classification_matches_intuition(conf in confidence_strategy()) {
        let is_high = conf.is_high();
        let raw_value = conf.raw();

        // Cognitive expectation: "I should be able to predict when something is 'high confidence'"
        if raw_value >= 0.7 {
            prop_assert!(is_high, "Confidence {} should be classified as high", raw_value);
        } else {
            prop_assert!(!is_high, "Confidence {} should not be classified as high", raw_value);
        }
    }

    /// Property: "Confidence operations never panic"
    /// Critical reliability expectation: "No matter what inputs I give, the code shouldn't crash"
    #[test]
    fn confidence_operations_never_panic(
        conf_a in confidence_strategy(),
        conf_b in confidence_strategy(),
        weight_a in weight_strategy(),
        weight_b in weight_strategy(),
        raw_value in arbitrary_f32_strategy()
    ) {
        // Test all operations with valid inputs - these should never panic
        let _ = conf_a.and(conf_b);
        let _ = conf_a.or(conf_b);
        let _ = conf_a.not();
        let _ = conf_a.combine_weighted(conf_b, weight_a, weight_b);
        let _ = conf_a.calibrate_overconfidence();
        let _ = conf_a.is_high();
        let _ = conf_a.raw();

        // Test construction with potentially invalid inputs
        let _ = Confidence::exact(raw_value);

        // Test percentage construction with any u8 value
        for percent in [0u8, 50u8, 100u8, 200u8, u8::MAX] {
            let _ = Confidence::from_percent(percent);
        }

        // Test frequency construction with edge cases
        let _ = Confidence::from_successes(0, 0);  // Should handle gracefully
        let _ = Confidence::from_successes(u32::MAX, u32::MAX);
        let _ = Confidence::from_successes(1000, 1);  // More successes than total

        // If we reach here without panicking, the test passes
        prop_assert!(true);
    }

    /// Property: "All confidence constants are valid and meaningful"
    /// Natural expectation: "The predefined confidence levels should make intuitive sense"
    #[test]
    fn confidence_constants_are_valid_and_meaningful(_unit in Just(())) {
        // Test that all constants are valid probabilities
        prop_assert!((Confidence::NONE.raw() - 0.0).abs() < 1e-6);
        prop_assert!((Confidence::LOW.raw() - 0.1).abs() < 1e-6);
        prop_assert!((Confidence::MEDIUM.raw() - 0.5).abs() < 1e-6);
        prop_assert!((Confidence::HIGH.raw() - 0.9).abs() < 1e-6);
        prop_assert!((Confidence::CERTAIN.raw() - 1.0).abs() < 1e-6);

        // Test cognitive ordering matches mathematical ordering
        prop_assert!(Confidence::NONE.raw() < Confidence::LOW.raw());
        prop_assert!(Confidence::LOW.raw() < Confidence::MEDIUM.raw());
        prop_assert!(Confidence::MEDIUM.raw() < Confidence::HIGH.raw());
        prop_assert!(Confidence::HIGH.raw() < Confidence::CERTAIN.raw());

        // Test that HIGH constant matches is_high() classification
        prop_assert!(Confidence::HIGH.is_high());
        prop_assert!(Confidence::CERTAIN.is_high());
        prop_assert!(!Confidence::MEDIUM.is_high());
        prop_assert!(!Confidence::LOW.is_high());
        prop_assert!(!Confidence::NONE.is_high());
    }

    /// Property: "Confidence arithmetic maintains closure under valid operations"
    /// Mathematical expectation: Operations on valid confidences should yield valid confidences
    #[test]
    fn confidence_operations_maintain_closure(
        conf_a in confidence_strategy(),
        conf_b in confidence_strategy(),
        conf_c in confidence_strategy()
    ) {
        // Test that chaining operations maintains validity
        let complex_result = conf_a
            .and(conf_b)
            .or(conf_c)
            .not()
            .calibrate_overconfidence();

        prop_assert!(complex_result.raw() >= 0.0);
        prop_assert!(complex_result.raw() <= 1.0);
        prop_assert!(!complex_result.raw().is_nan());
        prop_assert!(!complex_result.raw().is_infinite());
    }

    /// Property: "Idempotent operations behave correctly"
    /// Mathematical expectation: Some operations should be stable when applied multiple times
    #[test]
    fn idempotent_operations_behave_correctly(conf in confidence_strategy()) {
        // Double negation should return to original
        let double_neg = conf.not().not();
        prop_assert!(
            (double_neg.raw() - conf.raw()).abs() < f32::EPSILON * 2.0,
            "Double negation should be identity"
        );

        // AND with self should equal self for mathematical AND (P(A ∧ A) = P(A) for binary logic)
        // But for probabilistic AND it's P(A ∧ A) = P(A) * P(A) = P(A)^2, which is NOT idempotent
        // So we test that AND with self gives the square (for independent events)
        let self_and = conf.and(conf);
        let expected_square = conf.raw() * conf.raw();
        prop_assert!(
            (self_and.raw() - expected_square).abs() < f32::EPSILON,
            "AND with self should equal square for probabilistic independence: {} -> {} (expected {})",
            conf.raw(), self_and.raw(), expected_square
        );

        // OR with self should follow P(A ∨ A) = P(A) + P(A) - P(A ∧ A) = P(A) + P(A) - P(A)^2 = P(A)(2 - P(A))
        // For binary logic it would be idempotent, but for probabilistic OR it's not
        let self_or = conf.or(conf);
        let expected_or = conf.raw().mul_add(-conf.raw(), conf.raw() + conf.raw());
        prop_assert!(
            (self_or.raw() - expected_or).abs() < f32::EPSILON,
            "OR with self should follow probability rules: {} -> {} (expected {})",
            conf.raw(), self_or.raw(), expected_or
        );
    }
}
