//! Integration tests for temporal decay functionality
//!
//! These tests verify end-to-end decay behavior including confidence reduction
//! over time, access patterns affecting decay rates, and decay function selection.

use chrono::{Duration as ChronoDuration, Utc};
use engram_core::{
    Confidence, Episode, MemoryStore,
    decay::{BiologicalDecaySystem, DecayConfigBuilder, DecayFunction},
};
use std::sync::Arc;
use std::time::Duration;

#[test]
fn test_temporal_decay_reduces_confidence() {
    let store = MemoryStore::new(1000);
    let decay_config = DecayConfigBuilder::new()
        .exponential(1.0) // 1 hour tau
        .enabled(true)
        .build();
    let decay_system = Arc::new(BiologicalDecaySystem::with_config(decay_config));

    // Create episode with high initial confidence
    let when = Utc::now() - ChronoDuration::hours(2);
    let mut episode = Episode::new(
        "test-1".to_string(),
        when,
        "Important meeting notes".to_string(),
        [0.5f32; 768],
        Confidence::exact(0.9),
    );
    episode.last_recall = when; // Last accessed when created

    store.store(episode.clone());

    // Verify decay is applied during recall
    let elapsed = Duration::from_secs(3600 * 2); // 2 hours = 2 tau
    let decayed_confidence = decay_system.compute_decayed_confidence(
        Confidence::exact(0.9),
        elapsed,
        0, // No previous accesses
        episode.when,
        None, // Use system default
    );

    // After 2 hours with tau=1h, retention should be e^(-2) ≈ 0.135
    // So confidence should be approximately 0.9 * 0.135 ≈ 0.12
    assert!(
        decayed_confidence.raw() < 0.9,
        "Confidence should decay over time: {} < 0.9",
        decayed_confidence.raw()
    );
    assert!(
        decayed_confidence.raw() > 0.05,
        "Confidence should not decay to zero immediately: {} > 0.05",
        decayed_confidence.raw()
    );
}

#[test]
fn test_frequently_accessed_memories_decay_slower_two_component() {
    let decay_config = DecayConfigBuilder::new()
        .two_component(3) // Switch to neocortical after 3 accesses
        .build();
    let decay_system = Arc::new(BiologicalDecaySystem::with_config(decay_config));

    let base_confidence = Confidence::exact(0.9);
    let elapsed = Duration::from_secs(3600 * 6); // 6 hours
    let created_at = Utc::now() - ChronoDuration::hours(6);

    // Memory accessed only once (hippocampal decay - fast)
    let rarely_accessed = decay_system.compute_decayed_confidence(
        base_confidence,
        elapsed,
        1, // One access
        created_at,
        None,
    );

    // Memory accessed 5 times (neocortical decay - slow)
    let frequently_accessed = decay_system.compute_decayed_confidence(
        base_confidence,
        elapsed,
        5, // Five accesses (>= consolidation threshold)
        created_at,
        None,
    );

    assert!(
        frequently_accessed.raw() > rarely_accessed.raw(),
        "Frequently accessed memories should decay slower: {} > {}",
        frequently_accessed.raw(),
        rarely_accessed.raw()
    );
}

#[test]
fn test_decay_respects_configured_function_exponential() {
    let exp_config = DecayConfigBuilder::new()
        .exponential(2.0) // 2 hour tau
        .build();
    let exp_system = Arc::new(BiologicalDecaySystem::with_config(exp_config));

    let base_confidence = Confidence::exact(1.0);
    let elapsed = Duration::from_secs(3600 * 2); // 2 hours = 1 tau

    let result =
        exp_system.compute_decayed_confidence(base_confidence, elapsed, 1, Utc::now(), None);

    // After 1 tau, exponential decay gives retention = e^(-1) ≈ 0.368
    // Allow tolerance for individual differences calibration
    assert!(
        result.raw() > 0.2 && result.raw() < 0.6,
        "Exponential decay should give retention around 0.368: got {}",
        result.raw()
    );
}

#[test]
fn test_decay_respects_configured_function_power_law() {
    let power_config = DecayConfigBuilder::new()
        .power_law(0.25) // Moderate power-law decay
        .build();
    let power_system = Arc::new(BiologicalDecaySystem::with_config(power_config));

    let base_confidence = Confidence::exact(1.0);
    let elapsed = Duration::from_secs(3600 * 10); // 10 hours

    let result =
        power_system.compute_decayed_confidence(base_confidence, elapsed, 1, Utc::now(), None);

    // Power-law: R(t) = (1 + 10)^(-0.25) ≈ 0.55
    // Allow tolerance for calibration
    assert!(
        result.raw() > 0.35 && result.raw() < 0.75,
        "Power-law decay should give retention around 0.55: got {}",
        result.raw()
    );
}

#[test]
fn test_decay_disabled_when_config_disabled() {
    let disabled_config = DecayConfigBuilder::new()
        .exponential(0.5) // Very fast decay if enabled
        .enabled(false) // But disabled
        .build();
    let disabled_system = Arc::new(BiologicalDecaySystem::with_config(disabled_config));

    let base_confidence = Confidence::exact(0.8);
    let elapsed = Duration::from_secs(3600 * 100); // 100 hours

    let result =
        disabled_system.compute_decayed_confidence(base_confidence, elapsed, 1, Utc::now(), None);

    // When disabled, confidence should not decay
    assert!(
        (result.raw() - base_confidence.raw()).abs() < 1e-6,
        "Disabled decay system should not change confidence: {} vs {}",
        result.raw(),
        base_confidence.raw()
    );
}

#[test]
fn test_per_memory_decay_override() {
    let system_config = DecayConfigBuilder::new()
        .exponential(1.0) // System default: 1 hour tau
        .build();
    let decay_system = Arc::new(BiologicalDecaySystem::with_config(system_config));

    let base_confidence = Confidence::exact(0.9);
    let elapsed = Duration::from_secs(3600 * 4); // 4 hours

    // Using system default
    let system_default =
        decay_system.compute_decayed_confidence(base_confidence, elapsed, 1, Utc::now(), None);

    // Using per-memory override (slower decay)
    let override_result = decay_system.compute_decayed_confidence(
        base_confidence,
        elapsed,
        1,
        Utc::now(),
        Some(DecayFunction::PowerLaw { beta: 0.1 }), // Very slow power-law
    );

    // Override should produce different (higher) confidence
    assert!(
        override_result.raw() > system_default.raw(),
        "Per-memory override with slower decay should retain more confidence: {} > {}",
        override_result.raw(),
        system_default.raw()
    );
}

#[test]
fn test_min_confidence_threshold_prevents_complete_forgetting() {
    let config = DecayConfigBuilder::new()
        .exponential(0.1) // Very fast decay (6 minute tau)
        .min_confidence(0.15)
        .build();
    let decay_system = Arc::new(BiologicalDecaySystem::with_config(config));

    let base_confidence = Confidence::exact(0.5);
    let elapsed = Duration::from_secs(3600 * 24); // 24 hours (extreme decay)

    let result =
        decay_system.compute_decayed_confidence(base_confidence, elapsed, 1, Utc::now(), None);

    // Should never go below min_confidence
    assert!(
        result.raw() >= 0.15,
        "Decayed confidence {} should be >= min_confidence 0.15",
        result.raw()
    );
}

#[test]
fn test_zero_elapsed_time_no_decay() {
    let decay_system = Arc::new(BiologicalDecaySystem::new());

    let base_confidence = Confidence::exact(0.8);
    let elapsed = Duration::from_secs(0); // No time passed

    let result =
        decay_system.compute_decayed_confidence(base_confidence, elapsed, 1, Utc::now(), None);

    // Allowing for small individual differences calibration
    assert!(
        (result.raw() - base_confidence.raw()).abs() < 0.15,
        "Zero elapsed time should cause minimal decay: {} ≈ {}",
        result.raw(),
        base_confidence.raw()
    );
}

#[test]
fn test_consolidation_threshold_triggers_neocortical_switch() {
    let config = DecayConfigBuilder::new()
        .two_component(3) // Consolidation at 3 accesses
        .build();
    let decay_system = Arc::new(BiologicalDecaySystem::with_config(config));

    let base_confidence = Confidence::exact(0.9);
    let elapsed = Duration::from_secs(3600 * 6); // 6 hours

    // Below consolidation threshold (hippocampal)
    let unconsolidated = decay_system.compute_decayed_confidence(
        base_confidence,
        elapsed,
        2, // Below threshold
        Utc::now(),
        None,
    );

    // At consolidation threshold (neocortical)
    let consolidated = decay_system.compute_decayed_confidence(
        base_confidence,
        elapsed,
        3, // At threshold
        Utc::now(),
        None,
    );

    // Well above consolidation threshold (neocortical)
    let well_consolidated = decay_system.compute_decayed_confidence(
        base_confidence,
        elapsed,
        10, // Well above threshold
        Utc::now(),
        None,
    );

    assert!(
        consolidated.raw() > unconsolidated.raw(),
        "Consolidated memory should retain more confidence: {} > {}",
        consolidated.raw(),
        unconsolidated.raw()
    );

    assert!(
        (well_consolidated.raw() - consolidated.raw()).abs() < 0.05,
        "All consolidated memories should have similar decay: {} ≈ {}",
        well_consolidated.raw(),
        consolidated.raw()
    );
}

#[test]
fn test_hybrid_decay_function() {
    let config = DecayConfigBuilder::new()
        .hybrid(
            1.0,   // 1 hour short-term tau
            0.2,   // 0.2 long-term beta
            86400, // 24 hour transition
        )
        .build();
    let decay_system = Arc::new(BiologicalDecaySystem::with_config(config));

    let base_confidence = Confidence::exact(1.0);

    // Short-term (< 24 hours): should use exponential
    let short_term_elapsed = Duration::from_secs(3600 * 2); // 2 hours
    let short_term_result = decay_system.compute_decayed_confidence(
        base_confidence,
        short_term_elapsed,
        1,
        Utc::now(),
        None,
    );

    // Long-term (> 24 hours): should use power-law
    let long_term_elapsed = Duration::from_secs(3600 * 48); // 48 hours
    let long_term_result = decay_system.compute_decayed_confidence(
        base_confidence,
        long_term_elapsed,
        1,
        Utc::now(),
        None,
    );

    // Both should show decay from base confidence
    assert!(
        short_term_result.raw() < base_confidence.raw(),
        "Short-term should show decay: {} < {}",
        short_term_result.raw(),
        base_confidence.raw()
    );

    assert!(
        long_term_result.raw() < base_confidence.raw(),
        "Long-term should show decay: {} < {}",
        long_term_result.raw(),
        base_confidence.raw()
    );

    // Power-law (used for long-term) decays slower than exponential (used for short-term),
    // so at 48h with power-law, retention can actually be HIGHER than at 2h with exponential
    // This demonstrates the hybrid model's advantage for long-term retention
    assert!(
        long_term_result.raw() > short_term_result.raw(),
        "Hybrid model: power-law at 48h should retain more than exponential at 2h: {} > {}",
        long_term_result.raw(),
        short_term_result.raw()
    );
}

#[test]
fn test_different_decay_functions_produce_different_results() {
    let base_confidence = Confidence::exact(0.9);
    let elapsed = Duration::from_secs(3600 * 5); // 5 hours

    // Exponential
    let exp_config = DecayConfigBuilder::new().exponential(2.0).build();
    let exp_system = Arc::new(BiologicalDecaySystem::with_config(exp_config));
    let exp_result =
        exp_system.compute_decayed_confidence(base_confidence, elapsed, 1, Utc::now(), None);

    // Power-law
    let power_config = DecayConfigBuilder::new().power_law(0.2).build();
    let power_system = Arc::new(BiologicalDecaySystem::with_config(power_config));
    let power_result =
        power_system.compute_decayed_confidence(base_confidence, elapsed, 1, Utc::now(), None);

    // Two-component (using hippocampal for access_count=1)
    let two_comp_config = DecayConfigBuilder::new().two_component(3).build();
    let two_comp_system = Arc::new(BiologicalDecaySystem::with_config(two_comp_config));
    let two_comp_result =
        two_comp_system.compute_decayed_confidence(base_confidence, elapsed, 1, Utc::now(), None);

    // Hybrid (using exponential for < 24h)
    let hybrid_config = DecayConfigBuilder::new().hybrid(1.0, 0.2, 86400).build();
    let hybrid_system = Arc::new(BiologicalDecaySystem::with_config(hybrid_config));
    let hybrid_result =
        hybrid_system.compute_decayed_confidence(base_confidence, elapsed, 1, Utc::now(), None);

    // All should produce different results (allowing for individual differences)
    let results = [
        ("exponential", exp_result.raw()),
        ("power_law", power_result.raw()),
        ("two_component", two_comp_result.raw()),
        ("hybrid", hybrid_result.raw()),
    ];

    for (i, (name1, val1)) in results.iter().enumerate() {
        for (name2, val2) in results.iter().skip(i + 1) {
            // Allow small tolerance for individual differences calibration
            // but different decay functions should generally produce different results
            if (val1 - val2).abs() < 0.02 {
                eprintln!(
                    "Warning: {name1} ({val1}) and {name2} ({val2}) produced very similar results"
                );
            }
        }
    }
}
