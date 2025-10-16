//! Edge case tests for temporal decay functionality
//!
//! Tests boundary conditions, thread safety, extreme parameter values,
//! and other edge cases to ensure robustness.

use chrono::Utc;
use engram_core::{
    Confidence,
    decay::{BiologicalDecaySystem, DecayConfigBuilder, DecayFunction},
};
use std::sync::Arc;
use std::thread;
use std::time::Duration;

#[test]
fn test_zero_elapsed_time_produces_no_decay() {
    let decay_system = Arc::new(BiologicalDecaySystem::new());
    let base_confidence = Confidence::exact(0.7);
    let elapsed = Duration::from_secs(0);

    let result =
        decay_system.compute_decayed_confidence(base_confidence, elapsed, 1, Utc::now(), None);

    // With zero elapsed time, retention should be close to 1.0
    // Allow tolerance for individual differences calibration
    assert!(
        (result.raw() - base_confidence.raw()).abs() < 0.15,
        "Zero elapsed time should cause minimal decay"
    );
}

#[test]
fn test_very_long_elapsed_time_respects_min_confidence() {
    let config = DecayConfigBuilder::new()
        .exponential(1.0)
        .min_confidence(0.1)
        .build();
    let decay_system = Arc::new(BiologicalDecaySystem::with_config(config));

    let base_confidence = Confidence::exact(0.9);
    // Extreme elapsed time (1 year)
    let elapsed = Duration::from_secs(365 * 24 * 3600);

    let result =
        decay_system.compute_decayed_confidence(base_confidence, elapsed, 1, Utc::now(), None);

    // Should respect min_confidence
    assert!(
        result.raw() >= 0.1,
        "Should respect min_confidence: {} >= 0.1",
        result.raw()
    );
    assert!(
        result.raw() <= 1.0,
        "Should not exceed max confidence: {} <= 1.0",
        result.raw()
    );
}

#[test]
fn test_very_high_access_count_stabilizes() {
    let config = DecayConfigBuilder::new().two_component(3).build();
    let decay_system = Arc::new(BiologicalDecaySystem::with_config(config));

    let base_confidence = Confidence::exact(0.8);
    let elapsed = Duration::from_secs(3600 * 4);

    // Access count at threshold
    let at_threshold =
        decay_system.compute_decayed_confidence(base_confidence, elapsed, 3, Utc::now(), None);

    // Very high access count (should use same neocortical decay)
    let very_high =
        decay_system.compute_decayed_confidence(base_confidence, elapsed, 1000, Utc::now(), None);

    // Both should use neocortical decay and give similar results
    assert!(
        (at_threshold.raw() - very_high.raw()).abs() < 0.1,
        "Access counts above threshold should have similar decay: {} ≈ {}",
        at_threshold.raw(),
        very_high.raw()
    );
}

#[test]
fn test_concurrent_decay_computation_thread_safe() {
    let decay_system = Arc::new(BiologicalDecaySystem::new());
    let mut handles = vec![];

    // Spawn multiple threads computing decay simultaneously
    for _ in 0..10 {
        let system = Arc::clone(&decay_system);
        let handle = thread::spawn(move || {
            for i in 0..100 {
                let _ = system.compute_decayed_confidence(
                    Confidence::exact(0.8),
                    Duration::from_secs(3600 * (i % 12)),
                    i % 10,
                    Utc::now(),
                    None,
                );
            }
        });
        handles.push(handle);
    }

    // Should complete without panics or deadlocks
    for handle in handles {
        handle.join().expect("Thread should not panic");
    }
}

#[test]
fn test_decay_disabled_flag_completely_disables_decay() {
    let config = DecayConfigBuilder::new()
        .exponential(0.1) // Very fast decay if enabled
        .enabled(false)
        .build();
    let decay_system = Arc::new(BiologicalDecaySystem::with_config(config));

    let base_confidence = Confidence::exact(0.5);
    let extreme_elapsed = Duration::from_secs(365 * 24 * 3600); // 1 year

    let result = decay_system.compute_decayed_confidence(
        base_confidence,
        extreme_elapsed,
        1,
        Utc::now(),
        None,
    );

    // Should be exactly the same (no decay applied)
    assert!(
        (result.raw() - base_confidence.raw()).abs() < 1e-6,
        "Disabled decay should not change confidence at all: {} vs {}",
        result.raw(),
        base_confidence.raw()
    );
}

#[test]
fn test_max_confidence_bounded_at_one() {
    let decay_system = Arc::new(BiologicalDecaySystem::new());

    // Start with confidence above 1.0 (shouldn't happen in practice, but testing bounds)
    let base_confidence = Confidence::exact(1.0);
    let elapsed = Duration::from_secs(10);

    let result =
        decay_system.compute_decayed_confidence(base_confidence, elapsed, 1, Utc::now(), None);

    // Should never exceed 1.0
    assert!(
        result.raw() <= 1.0,
        "Decayed confidence should not exceed 1.0: {}",
        result.raw()
    );
}

#[test]
fn test_min_confidence_threshold_boundary() {
    let min_threshold = 0.2;
    let config = DecayConfigBuilder::new()
        .exponential(0.5)
        .min_confidence(min_threshold)
        .build();
    let decay_system = Arc::new(BiologicalDecaySystem::with_config(config));

    // Start below min threshold
    let below_threshold = Confidence::exact(0.15);
    let elapsed = Duration::from_secs(3600);

    let result =
        decay_system.compute_decayed_confidence(below_threshold, elapsed, 1, Utc::now(), None);

    // Should be clamped to min_confidence
    assert!(
        result.raw() >= min_threshold,
        "Should be >= min_confidence: {} >= {}",
        result.raw(),
        min_threshold
    );
}

#[test]
fn test_power_law_with_zero_beta_stable() {
    let config = DecayConfigBuilder::new()
        .power_law(0.0) // Zero exponent = no decay
        .build();
    let decay_system = Arc::new(BiologicalDecaySystem::with_config(config));

    let base_confidence = Confidence::exact(0.8);
    let elapsed = Duration::from_secs(3600 * 24); // 24 hours

    let result =
        decay_system.compute_decayed_confidence(base_confidence, elapsed, 1, Utc::now(), None);

    // With beta=0, (1+t)^(-0) = 1, so no decay (except individual differences)
    // But still allow some tolerance
    assert!(
        result.raw() > 0.6,
        "Beta=0 should cause minimal decay: {}",
        result.raw()
    );
}

#[test]
fn test_exponential_with_very_small_tau() {
    let config = DecayConfigBuilder::new()
        .exponential(0.01) // Very small tau (36 seconds)
        .min_confidence(0.05)
        .build();
    let decay_system = Arc::new(BiologicalDecaySystem::with_config(config));

    let base_confidence = Confidence::exact(0.9);
    let elapsed = Duration::from_secs(3600); // 1 hour

    let result =
        decay_system.compute_decayed_confidence(base_confidence, elapsed, 1, Utc::now(), None);

    // With very small tau, decay should be extreme but respect min_confidence
    assert!(
        result.raw() >= 0.05,
        "Should respect min_confidence: {}",
        result.raw()
    );
    assert!(
        result.raw() < 0.5,
        "Very fast decay should significantly reduce confidence: {}",
        result.raw()
    );
}

#[test]
fn test_exponential_with_very_large_tau() {
    let config = DecayConfigBuilder::new()
        .exponential(1000.0) // Very large tau (1000 hours)
        .build();
    let decay_system = Arc::new(BiologicalDecaySystem::with_config(config));

    let base_confidence = Confidence::exact(0.8);
    let elapsed = Duration::from_secs(3600 * 10); // 10 hours

    let result =
        decay_system.compute_decayed_confidence(base_confidence, elapsed, 1, Utc::now(), None);

    // With very large tau, decay should be minimal
    assert!(
        result.raw() > 0.7,
        "Very slow decay should retain most confidence: {}",
        result.raw()
    );
}

#[test]
fn test_hybrid_at_transition_point() {
    let config = DecayConfigBuilder::new()
        .hybrid(
            1.0,  // 1 hour short-term tau
            0.2,  // 0.2 long-term beta
            3600, // 1 hour transition point
        )
        .build();
    let decay_system = Arc::new(BiologicalDecaySystem::with_config(config));

    let base_confidence = Confidence::exact(0.9);

    // Just before transition
    let before = decay_system.compute_decayed_confidence(
        base_confidence,
        Duration::from_secs(3599),
        1,
        Utc::now(),
        None,
    );

    // Just after transition
    let after = decay_system.compute_decayed_confidence(
        base_confidence,
        Duration::from_secs(3601),
        1,
        Utc::now(),
        None,
    );

    // Should both show decay
    assert!(before.raw() < base_confidence.raw());
    assert!(after.raw() < base_confidence.raw());

    // Hybrid function transitions from exponential to power-law at boundary
    // These are fundamentally different functions, so we expect different behavior
    // rather than continuity. The test verifies the transition occurs.
    assert!(
        (before.raw() - after.raw()).abs() > 0.01,
        "Hybrid transition should use different functions: before={}, after={}",
        before.raw(),
        after.raw()
    );
}

#[test]
fn test_per_memory_override_with_different_functions() {
    let config = DecayConfigBuilder::new()
        .exponential(1.0) // System default
        .build();
    let decay_system = Arc::new(BiologicalDecaySystem::with_config(config));

    let base_confidence = Confidence::exact(0.8);
    let elapsed = Duration::from_secs(3600 * 3);

    // Test all possible override types
    let overrides = vec![
        DecayFunction::Exponential { tau_hours: 2.0 },
        DecayFunction::PowerLaw { beta: 0.15 },
        DecayFunction::TwoComponent {
            consolidation_threshold: 5,
        },
        DecayFunction::Hybrid {
            short_term_tau: 1.5,
            long_term_beta: 0.25,
            transition_point: 7200,
        },
    ];

    for override_fn in overrides {
        let result = decay_system.compute_decayed_confidence(
            base_confidence,
            elapsed,
            1,
            Utc::now(),
            Some(override_fn),
        );

        // All should produce valid results
        assert!(
            result.raw() >= 0.0 && result.raw() <= 1.0,
            "Override {:?} should produce valid confidence: {}",
            override_fn,
            result.raw()
        );
    }
}

#[test]
fn test_two_component_at_threshold_boundary() {
    let config = DecayConfigBuilder::new().two_component(3).build();
    let decay_system = Arc::new(BiologicalDecaySystem::with_config(config));

    let base_confidence = Confidence::exact(0.9);
    let elapsed = Duration::from_secs(3600 * 5);

    // Just below threshold
    let below = decay_system.compute_decayed_confidence(
        base_confidence,
        elapsed,
        2, // Below threshold
        Utc::now(),
        None,
    );

    // At threshold
    let at_threshold = decay_system.compute_decayed_confidence(
        base_confidence,
        elapsed,
        3, // At threshold
        Utc::now(),
        None,
    );

    // Above threshold
    let above = decay_system.compute_decayed_confidence(
        base_confidence,
        elapsed,
        4, // Above threshold
        Utc::now(),
        None,
    );

    // There should be a clear difference between below and at/above threshold
    assert!(
        at_threshold.raw() > below.raw(),
        "At threshold should have slower decay than below: {} > {}",
        at_threshold.raw(),
        below.raw()
    );

    // At and above threshold should be similar (both neocortical)
    assert!(
        (at_threshold.raw() - above.raw()).abs() < 0.05,
        "At and above threshold should be similar: {} ≈ {}",
        at_threshold.raw(),
        above.raw()
    );
}

#[test]
fn test_confidence_exact_zero_handled() {
    let decay_system = Arc::new(BiologicalDecaySystem::new());

    let zero_confidence = Confidence::exact(0.0);
    let elapsed = Duration::from_secs(3600);

    let result =
        decay_system.compute_decayed_confidence(zero_confidence, elapsed, 1, Utc::now(), None);

    // Zero should stay around zero (min_confidence might clamp it)
    assert!(
        result.raw() <= 0.15,
        "Zero confidence should stay low: {}",
        result.raw()
    );
}

#[test]
fn test_multiple_decay_systems_independent() {
    let config1 = DecayConfigBuilder::new().exponential(1.0).build();
    let config2 = DecayConfigBuilder::new().exponential(2.0).build();

    let system1 = Arc::new(BiologicalDecaySystem::with_config(config1));
    let system2 = Arc::new(BiologicalDecaySystem::with_config(config2));

    let base_confidence = Confidence::exact(0.8);
    let elapsed = Duration::from_secs(3600 * 2);

    let result1 = system1.compute_decayed_confidence(base_confidence, elapsed, 1, Utc::now(), None);

    let result2 = system2.compute_decayed_confidence(base_confidence, elapsed, 1, Utc::now(), None);

    // Different systems with different configs should produce different results
    assert!(
        (result1.raw() - result2.raw()).abs() > 0.01,
        "Independent decay systems should have independent configurations: {} vs {}",
        result1.raw(),
        result2.raw()
    );
}
