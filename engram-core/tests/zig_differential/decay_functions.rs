//! Differential tests for memory decay function kernels.
//!
//! Validates that Zig SIMD-vectorized Ebbinghaus decay produces numerically
//! identical results to the Rust baseline across various age distributions.
//!
//! # Test Coverage
//!
//! 1. **Property-Based Tests** - 10,000 random strength/age combinations
//! 2. **Edge Cases** - Zero age, ancient memories, zero strength, overflow
//! 3. **Boundary Conditions** - Single memory, many memories, extreme values
//!
//! # Current Status (Task 002)
//!
//! Zig kernels are currently stubs (no-op). These tests will FAIL until Task 007
//! implements the actual SIMD Ebbinghaus decay. This validates the differential
//! testing framework correctly detects divergence.

use super::{EPSILON_TRANSCENDENTAL, NUM_PROPTEST_CASES, assert_slices_approx_eq};
use proptest::prelude::*;

// Conditional import - tests work with or without zig-kernels feature
#[cfg(feature = "zig-kernels")]
use engram_core::zig_kernels;

// Mock Zig kernels module when feature is disabled (for testing Rust baseline)
#[cfg(not(feature = "zig-kernels"))]
mod zig_kernels {
    /// Mock apply_decay for testing without zig-kernels feature
    pub const fn apply_decay(_strengths: &mut [f32], _ages_seconds: &[u64]) {
        // Fallback uses stub behavior (no-op) to match Task 002 Zig stubs
    }
}

/// Rust baseline implementation of Ebbinghaus decay
///
/// Uses the forgetting curve: S(t) = S₀ * exp(-t/half_life)
/// where:
/// - S(t) is strength at time t
/// - S₀ is initial strength
/// - half_life is the decay time constant (86400 seconds = 1 day)
/// - t is age in seconds
///
/// IMPORTANT: This uses "half-life" semantics matching Zig's DEFAULT_HALF_LIFE.
/// After time = half_life, strength decays to S₀ * exp(-1) ≈ 0.3679 * S₀
///
/// Note: "half-life" in this context is NOT the radioactive decay half-life
/// (which would use exp(-t*ln(2)/half_life) to decay to 0.5). Instead, it's
/// the time constant τ where strength decays to 1/e.
fn rust_ebbinghaus_decay(strengths: &mut [f32], ages_seconds: &[u64]) {
    assert_eq!(strengths.len(), ages_seconds.len());

    const HALF_LIFE: f64 = 86400.0; // Matches Zig's DEFAULT_HALF_LIFE (1 day in seconds)

    for (strength, &age) in strengths.iter_mut().zip(ages_seconds.iter()) {
        let age_f64 = age as f64;
        let decay_factor = (-age_f64 / HALF_LIFE).exp();
        *strength *= decay_factor as f32;
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(NUM_PROPTEST_CASES))]

    /// Property test: Zig and Rust decay match for random strengths and ages
    #[test]
    #[allow(clippy::cast_precision_loss)]
    fn prop_decay_random_inputs(
        strengths in prop::collection::vec(0.0_f32..1.0, 100..10_000),
        ages in prop::collection::vec(0_u64..1_000_000, 100..10_000)
    ) {
        prop_assume!(strengths.len() == ages.len());

        let mut zig_strengths = strengths.clone();
        let mut rust_strengths = strengths;

        // Call Zig kernel
        zig_kernels::apply_decay(&mut zig_strengths, &ages);

        // Call Rust baseline
        rust_ebbinghaus_decay(&mut rust_strengths, &ages);

        // Verify equivalence (use EPSILON_TRANSCENDENTAL for exp operations)
        assert_slices_approx_eq(&rust_strengths, &zig_strengths, EPSILON_TRANSCENDENTAL);
    }

    /// Property test: Varying batch sizes
    #[test]
    fn prop_decay_batch_sizes(
        batch_size in 1_usize..10_000,
        strength_val in 0.0_f32..1.0,
        age_val in 0_u64..1_000_000
    ) {
        let strengths = vec![strength_val; batch_size];
        let ages = vec![age_val; batch_size];

        let mut zig_strengths = strengths.clone();
        let mut rust_strengths = strengths;

        zig_kernels::apply_decay(&mut zig_strengths, &ages);
        rust_ebbinghaus_decay(&mut rust_strengths, &ages);

        assert_slices_approx_eq(&rust_strengths, &zig_strengths, EPSILON_TRANSCENDENTAL);
    }

    /// Property test: Uniform age distributions
    #[test]
    fn prop_decay_uniform_ages(
        num_memories in 100_usize..1000,
        age in 0_u64..10_000_000
    ) {
        let strengths: Vec<f32> = (0..num_memories)
            .map(|i| (i as f32) / (num_memories as f32))
            .collect();
        let ages = vec![age; num_memories];

        let mut zig_strengths = strengths.clone();
        let mut rust_strengths = strengths;

        zig_kernels::apply_decay(&mut zig_strengths, &ages);
        rust_ebbinghaus_decay(&mut rust_strengths, &ages);

        assert_slices_approx_eq(&rust_strengths, &zig_strengths, EPSILON_TRANSCENDENTAL);
    }

    /// Property test: Varying strength, constant age
    #[test]
    fn prop_decay_varying_strengths(
        strengths in prop::collection::vec(0.0_f32..1.0, 100..1000),
        age in 0_u64..1_000_000
    ) {
        let ages = vec![age; strengths.len()];

        let mut zig_strengths = strengths.clone();
        let mut rust_strengths = strengths;

        zig_kernels::apply_decay(&mut zig_strengths, &ages);
        rust_ebbinghaus_decay(&mut rust_strengths, &ages);

        assert_slices_approx_eq(&rust_strengths, &zig_strengths, EPSILON_TRANSCENDENTAL);
    }

    /// Property test: Linear age progression
    #[test]
    fn prop_decay_linear_ages(
        num_memories in 100_usize..1000,
        initial_strength in 0.1_f32..1.0,
        max_age in 100_u64..1_000_000
    ) {
        let strengths = vec![initial_strength; num_memories];
        let ages: Vec<u64> = (0..num_memories)
            .map(|i| (i as u64 * max_age) / num_memories as u64)
            .collect();

        let mut zig_strengths = strengths.clone();
        let mut rust_strengths = strengths;

        zig_kernels::apply_decay(&mut zig_strengths, &ages);
        rust_ebbinghaus_decay(&mut rust_strengths, &ages);

        assert_slices_approx_eq(&rust_strengths, &zig_strengths, EPSILON_TRANSCENDENTAL);
    }
}

#[cfg(test)]
mod edge_cases {
    use super::*;

    #[test]
    fn test_zero_age() {
        // Zero age should result in no decay (strength unchanged)
        let mut zig_strengths = vec![1.0, 0.8, 0.5, 0.3, 0.1];
        let mut rust_strengths = zig_strengths.clone();
        let ages = vec![0; 5];

        zig_kernels::apply_decay(&mut zig_strengths, &ages);
        rust_ebbinghaus_decay(&mut rust_strengths, &ages);

        assert_slices_approx_eq(&rust_strengths, &zig_strengths, EPSILON_TRANSCENDENTAL);

        // All strengths should be unchanged (decay factor = e^0 = 1)
        for (&rust_val, &expected) in rust_strengths.iter().zip(&[1.0, 0.8, 0.5, 0.3, 0.1]) {
            assert!(
                (rust_val - expected).abs() < EPSILON_TRANSCENDENTAL,
                "Zero age should preserve strength: expected {expected}, got {rust_val}"
            );
        }
    }

    #[test]
    fn test_ancient_memories() {
        // Very old memories should decay significantly
        let mut zig_strengths = vec![1.0; 100];
        let mut rust_strengths = zig_strengths.clone();
        let ages = vec![1_000_000_000_u64; 100]; // ~31 years

        zig_kernels::apply_decay(&mut zig_strengths, &ages);
        rust_ebbinghaus_decay(&mut rust_strengths, &ages);

        assert_slices_approx_eq(&rust_strengths, &zig_strengths, EPSILON_TRANSCENDENTAL);

        // All strengths should be heavily decayed (near zero)
        for &strength in &rust_strengths {
            assert!(
                strength < 0.01,
                "Ancient memories should be heavily decayed: got {strength}"
            );
        }
    }

    #[test]
    fn test_zero_strength() {
        // Zero strength should stay zero
        let mut zig_strengths = vec![0.0; 100];
        let mut rust_strengths = zig_strengths.clone();
        let ages = vec![100_000; 100];

        zig_kernels::apply_decay(&mut zig_strengths, &ages);
        rust_ebbinghaus_decay(&mut rust_strengths, &ages);

        assert_slices_approx_eq(&rust_strengths, &zig_strengths, EPSILON_TRANSCENDENTAL);

        for &strength in &rust_strengths {
            assert!(
                strength.abs() < EPSILON_TRANSCENDENTAL,
                "Zero strength should remain zero: got {strength}"
            );
        }
    }

    #[test]
    fn test_single_memory() {
        let mut zig_strengths = vec![0.8];
        let mut rust_strengths = zig_strengths.clone();
        let ages = vec![3600]; // 1 hour

        zig_kernels::apply_decay(&mut zig_strengths, &ages);
        rust_ebbinghaus_decay(&mut rust_strengths, &ages);

        assert_slices_approx_eq(&rust_strengths, &zig_strengths, EPSILON_TRANSCENDENTAL);
    }

    #[test]
    fn test_many_memories() {
        let num_memories = 100_000;
        let mut zig_strengths = vec![0.5; num_memories];
        let mut rust_strengths = zig_strengths.clone();
        let ages = vec![86400; num_memories]; // 1 day

        zig_kernels::apply_decay(&mut zig_strengths, &ages);
        rust_ebbinghaus_decay(&mut rust_strengths, &ages);

        assert_slices_approx_eq(&rust_strengths, &zig_strengths, EPSILON_TRANSCENDENTAL);
    }

    #[test]
    fn test_one_hour_decay() {
        let mut zig_strengths = vec![1.0; 10];
        let mut rust_strengths = zig_strengths.clone();
        let ages = vec![3600; 10]; // 1 hour

        zig_kernels::apply_decay(&mut zig_strengths, &ages);
        rust_ebbinghaus_decay(&mut rust_strengths, &ages);

        assert_slices_approx_eq(&rust_strengths, &zig_strengths, EPSILON_TRANSCENDENTAL);

        // Should decay slightly from 1.0
        for &strength in &rust_strengths {
            assert!(
                strength < 1.0 && strength > 0.95,
                "1 hour decay should be slight: got {strength}"
            );
        }
    }

    #[test]
    fn test_one_day_decay() {
        let mut zig_strengths = vec![1.0; 10];
        let mut rust_strengths = zig_strengths.clone();
        let ages = vec![86400; 10]; // 1 day

        zig_kernels::apply_decay(&mut zig_strengths, &ages);
        rust_ebbinghaus_decay(&mut rust_strengths, &ages);

        assert_slices_approx_eq(&rust_strengths, &zig_strengths, EPSILON_TRANSCENDENTAL);

        // Should decay to ~0.37 (e^-1)
        for &strength in &rust_strengths {
            assert!(
                (strength - 0.3679).abs() < 0.01,
                "1 day decay should be ~0.37: got {strength}"
            );
        }
    }

    #[test]
    fn test_one_week_decay() {
        let mut zig_strengths = vec![1.0; 10];
        let mut rust_strengths = zig_strengths.clone();
        let ages = vec![604_800_u64; 10]; // 1 week

        zig_kernels::apply_decay(&mut zig_strengths, &ages);
        rust_ebbinghaus_decay(&mut rust_strengths, &ages);

        assert_slices_approx_eq(&rust_strengths, &zig_strengths, EPSILON_TRANSCENDENTAL);

        // Should decay significantly
        for &strength in &rust_strengths {
            assert!(
                strength < 0.1,
                "1 week decay should be significant: got {strength}"
            );
        }
    }

    #[test]
    fn test_mixed_ages() {
        let mut zig_strengths = vec![1.0, 1.0, 1.0, 1.0, 1.0];
        let mut rust_strengths = zig_strengths.clone();
        let ages = vec![
            0_u64,         // Brand new
            3600_u64,      // 1 hour
            86_400_u64,    // 1 day
            604_800_u64,   // 1 week
            2_592_000_u64, // 1 month
        ];

        zig_kernels::apply_decay(&mut zig_strengths, &ages);
        rust_ebbinghaus_decay(&mut rust_strengths, &ages);

        assert_slices_approx_eq(&rust_strengths, &zig_strengths, EPSILON_TRANSCENDENTAL);

        // Strengths should be monotonically decreasing (allow small FP rounding errors)
        for i in 0..rust_strengths.len() - 1 {
            let diff = rust_strengths[i] - rust_strengths[i + 1];
            assert!(
                diff >= -EPSILON_TRANSCENDENTAL,
                "Older memories should have lower strength (modulo FP rounding): \
                 strengths[{}]={} should be >= strengths[{}]={} (diff={}, epsilon={})",
                i,
                rust_strengths[i],
                i + 1,
                rust_strengths[i + 1],
                diff,
                EPSILON_TRANSCENDENTAL
            );
        }
    }

    #[test]
    fn test_mixed_strengths() {
        let mut zig_strengths = vec![0.1, 0.3, 0.5, 0.7, 0.9];
        let mut rust_strengths = zig_strengths.clone();
        let ages = vec![86400; 5]; // All 1 day old

        zig_kernels::apply_decay(&mut zig_strengths, &ages);
        rust_ebbinghaus_decay(&mut rust_strengths, &ages);

        assert_slices_approx_eq(&rust_strengths, &zig_strengths, EPSILON_TRANSCENDENTAL);

        // Relative ordering should be preserved
        for i in 0..rust_strengths.len() - 1 {
            assert!(
                rust_strengths[i] < rust_strengths[i + 1],
                "Relative strength ordering should be preserved"
            );
        }
    }

    #[test]
    fn test_max_strength() {
        let mut zig_strengths = vec![1.0; 10];
        let mut rust_strengths = zig_strengths.clone();
        let ages = vec![1000; 10];

        zig_kernels::apply_decay(&mut zig_strengths, &ages);
        rust_ebbinghaus_decay(&mut rust_strengths, &ages);

        assert_slices_approx_eq(&rust_strengths, &zig_strengths, EPSILON_TRANSCENDENTAL);
    }

    #[test]
    fn test_very_small_strengths() {
        let mut zig_strengths = vec![1e-10; 10];
        let mut rust_strengths = zig_strengths.clone();
        let ages = vec![1000; 10];

        zig_kernels::apply_decay(&mut zig_strengths, &ages);
        rust_ebbinghaus_decay(&mut rust_strengths, &ages);

        assert_slices_approx_eq(&rust_strengths, &zig_strengths, EPSILON_TRANSCENDENTAL);
    }

    #[test]
    fn test_exponential_age_distribution() {
        // Ages distributed exponentially (common in real workloads)
        let mut zig_strengths = vec![1.0; 10];
        let mut rust_strengths = zig_strengths.clone();
        let ages: Vec<u64> = (0..10).map(|i| 2_u64.pow(i)).collect();

        zig_kernels::apply_decay(&mut zig_strengths, &ages);
        rust_ebbinghaus_decay(&mut rust_strengths, &ages);

        assert_slices_approx_eq(&rust_strengths, &zig_strengths, EPSILON_TRANSCENDENTAL);
    }

    #[test]
    fn test_same_age_different_strengths() {
        let mut zig_strengths: Vec<f32> = (0..100).map(|i| (i as f32) / 100.0).collect();
        let mut rust_strengths = zig_strengths.clone();
        let ages = vec![86400; 100]; // All 1 day old

        zig_kernels::apply_decay(&mut zig_strengths, &ages);
        rust_ebbinghaus_decay(&mut rust_strengths, &ages);

        assert_slices_approx_eq(&rust_strengths, &zig_strengths, EPSILON_TRANSCENDENTAL);
    }

    #[test]
    fn test_same_strength_different_ages() {
        let mut zig_strengths = vec![0.8; 100];
        let mut rust_strengths = zig_strengths.clone();
        let ages: Vec<u64> = (0..100).map(|i| i * 1000).collect();

        zig_kernels::apply_decay(&mut zig_strengths, &ages);
        rust_ebbinghaus_decay(&mut rust_strengths, &ages);

        assert_slices_approx_eq(&rust_strengths, &zig_strengths, EPSILON_TRANSCENDENTAL);
    }
}

#[cfg(test)]
mod regression_tests {
    use super::*;

    /// Regression test for Issue H3: Decay constant semantics clarification
    ///
    /// Verifies that Rust and Zig use the same "half-life" semantics.
    /// After time = half_life, strength should decay to exp(-1) ≈ 0.3679
    #[test]
    fn test_regression_decay_constant_semantics() {
        let mut zig_strengths = vec![1.0; 5];
        let mut rust_strengths = vec![1.0; 5];
        let ages = vec![86400; 5]; // Exactly 1 half-life

        zig_kernels::apply_decay(&mut zig_strengths, &ages);
        rust_ebbinghaus_decay(&mut rust_strengths, &ages);

        // Both should decay to exp(-1) ≈ 0.3679
        assert_slices_approx_eq(&rust_strengths, &zig_strengths, EPSILON_TRANSCENDENTAL);
        for &strength in &rust_strengths {
            assert!(
                (strength - 0.3679).abs() < 0.01,
                "After one half-life, strength should be ~0.3679, got {strength}"
            );
        }
    }

    /// Regression test for Issue H4: Denormal float handling
    ///
    /// Previously, test only checked exact zero. This verifies denormal handling.
    #[test]
    fn test_regression_denormal_strengths() {
        let mut zig_strengths = vec![
            f32::MIN_POSITIVE, // Smallest normal float
            1e-38,             // Near denormal threshold
            1e-45,             // Denormal float
            0.0,               // Exact zero
        ];
        let mut rust_strengths = zig_strengths.clone();
        let ages = vec![86400; 4]; // 1 day decay

        zig_kernels::apply_decay(&mut zig_strengths, &ages);
        rust_ebbinghaus_decay(&mut rust_strengths, &ages);

        // Verify no NaN/Inf and consistent handling
        for (zig, rust) in zig_strengths.iter().zip(rust_strengths.iter()) {
            assert!(!zig.is_nan() && !rust.is_nan(), "Should not produce NaN");
            assert!(
                !zig.is_infinite() && !rust.is_infinite(),
                "Should not produce Inf"
            );
        }

        assert_slices_approx_eq(&rust_strengths, &zig_strengths, EPSILON_TRANSCENDENTAL);
    }

    /// Regression test for Issue H7: NaN/Inf numerical stability checks
    ///
    /// Verifies that very large and very small values don't produce NaN/Inf.
    #[test]
    fn test_regression_numerical_stability() {
        let mut zig_strengths = vec![1e10, 1e-10, f32::MAX, f32::MIN_POSITIVE];
        let mut rust_strengths = zig_strengths.clone();
        let ages = vec![1000, 10_000, 100_000, 1_000_000];

        zig_kernels::apply_decay(&mut zig_strengths, &ages);
        rust_ebbinghaus_decay(&mut rust_strengths, &ages);

        // Check for NaN/Inf in results
        for &strength in &rust_strengths {
            assert!(!strength.is_nan(), "Rust produced NaN");
            assert!(!strength.is_infinite(), "Rust produced Inf");
        }
        for &strength in &zig_strengths {
            assert!(!strength.is_nan(), "Zig produced NaN");
            assert!(!strength.is_infinite(), "Zig produced Inf");
        }

        assert_slices_approx_eq(&rust_strengths, &zig_strengths, EPSILON_TRANSCENDENTAL);
    }
}
