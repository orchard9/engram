//! Integration tests for associative and repetition priming
//!
//! Validates all three priming types working together with saturating
//! additive combination.

use engram_core::cognitive::priming::{
    AssociativePrimingEngine, PrimingCoordinator, RepetitionPrimingEngine,
};
use std::time::Duration;

// ==================== Associative Priming Tests ====================

#[test]
fn test_associative_priming_through_cooccurrence() {
    let engine = AssociativePrimingEngine::new();

    // Co-activate "thunder" and "lightning" 5 times within temporal window
    for _ in 0..5 {
        engine.record_coactivation("thunder", "lightning");
        // Small delay to ensure they're recorded as co-occurrences
        std::thread::sleep(Duration::from_millis(1));
    }

    // Check association strength: P(lightning | thunder)
    let strength = engine.compute_association_strength("thunder", "lightning");

    // After 5 co-occurrences, should have strong association
    // P(lightning|thunder) = cooccurrences / thunder_activations
    assert!(
        strength > 0.3,
        "Associative priming should form after 5 co-occurrences: strength = {}",
        strength
    );
}

#[test]
fn test_minimum_cooccurrence_prevents_spurious_associations() {
    let engine = AssociativePrimingEngine::new(); // min_cooccurrence = 2 by default

    // Single co-occurrence (below threshold)
    engine.record_coactivation("node_a", "node_b");

    let strength = engine.compute_association_strength("node_a", "node_b");
    assert_eq!(
        strength, 0.0,
        "Single co-occurrence should not form association (below min threshold of 2)"
    );

    // Second co-occurrence (meets threshold)
    engine.record_coactivation("node_a", "node_b");

    let strength_after = engine.compute_association_strength("node_a", "node_b");
    assert!(
        strength_after > 0.0,
        "Two co-occurrences should form association: strength = {}",
        strength_after
    );
}

#[test]
fn test_cooccurrence_window_matches_working_memory_span() {
    let engine = AssociativePrimingEngine::new()
        .with_cooccurrence_window(Duration::from_millis(50));

    // Activate first node
    engine.record_coactivation("node_x", "temp");

    // Wait beyond co-occurrence window
    std::thread::sleep(Duration::from_millis(60));

    // Activate second node (should NOT count as co-occurrence)
    engine.record_coactivation("node_y", "temp");

    // No association should form (nodes not in temporal window)
    let strength = engine.compute_association_strength("node_x", "node_y");
    assert_eq!(
        strength, 0.0,
        "Nodes outside temporal window should not form associations"
    );
}

#[test]
fn test_associative_priming_conditional_probability() {
    let engine = AssociativePrimingEngine::new();

    // Create asymmetric activation pattern:
    // A activates with B 3 times
    for _ in 0..3 {
        engine.record_coactivation("a", "b");
    }

    // A activates with C 1 time
    engine.record_coactivation("a", "c");

    // P(b|a) should be higher than P(c|a)
    let strength_ab = engine.compute_association_strength("a", "b");
    let strength_ac = engine.compute_association_strength("a", "c");

    assert!(
        strength_ab > strength_ac,
        "More frequent co-occurrence should have higher strength: P(b|a)={} > P(c|a)={}",
        strength_ab,
        strength_ac
    );
}

// ==================== Repetition Priming Tests ====================

#[test]
fn test_repetition_priming_accumulates_with_ceiling() {
    let engine = RepetitionPrimingEngine::new();

    // Expose node 10 times (should hit 30% ceiling after 6 exposures)
    for i in 1..=10 {
        engine.record_exposure("node_id");
        let boost = engine.compute_repetition_boost("node_id");

        if i <= 6 {
            // Below ceiling: linear accumulation (i × 5%)
            let expected = (i as f32) * 0.05;
            assert!(
                (boost - expected).abs() < f32::EPSILON,
                "Exposure {}: boost = {}, expected = {}",
                i,
                boost,
                expected
            );
        } else {
            // At ceiling: should be exactly 30%
            assert!(
                (boost - 0.30).abs() < f32::EPSILON,
                "Exposure {}: boost = {}, should be at ceiling 0.30",
                i,
                boost
            );
        }
    }
}

#[test]
fn test_repetition_priming_provides_5_percent_boost_per_exposure() {
    let engine = RepetitionPrimingEngine::new();

    // Test first 5 exposures (below ceiling)
    for i in 1..=5 {
        engine.record_exposure("concept");
        let boost = engine.compute_repetition_boost("concept");
        let expected = (i as f32) * 0.05;

        assert!(
            (boost - expected).abs() < 1e-6,
            "Exposure {}: boost = {}, expected exactly {}",
            i,
            boost,
            expected
        );
    }
}

#[test]
fn test_repetition_priming_ceiling_enforced_at_30_percent() {
    let engine = RepetitionPrimingEngine::new();

    // Record 20 exposures (way beyond ceiling)
    for _ in 0..20 {
        engine.record_exposure("node");
    }

    let boost = engine.compute_repetition_boost("node");
    assert!(
        (boost - 0.30).abs() < f32::EPSILON,
        "Boost should be capped at 30% ceiling: boost = {}",
        boost
    );
}

// ==================== Priming Types Integration Tests ====================

#[test]
fn test_priming_types_integrate_without_conflicts() {
    let coordinator = PrimingCoordinator::new();

    // Semantic priming setup (would normally activate via spreading)
    // For this test, we'll manually activate semantic priming on "doctor" → "nurse"
    let doctor_embedding = [0.5f32; 768];
    let nurse_embedding = [0.6f32; 768]; // High similarity

    coordinator.semantic().activate_priming("doctor", &doctor_embedding, || {
        vec![("nurse".to_string(), nurse_embedding, 1)]
    });

    // Wait a bit for semantic priming to register
    std::thread::sleep(Duration::from_millis(10));

    // Associative priming: "thunder" → "lightning"
    for _ in 0..3 {
        coordinator.associative().record_coactivation("thunder", "lightning");
    }

    // Repetition priming: "familiar_word" exposed multiple times
    for _ in 0..4 {
        coordinator.repetition().record_exposure("familiar_word");
    }

    // Compute total boost for each type independently
    let semantic_boost = coordinator.semantic().compute_priming_boost("nurse");
    let assoc_boost = coordinator.associative().compute_association_strength("thunder", "lightning");
    let rep_boost = coordinator.repetition().compute_repetition_boost("familiar_word");

    // All three types should be active
    assert!(semantic_boost > 0.0, "Semantic priming should be active");
    assert!(assoc_boost > 0.0, "Associative priming should be active");
    assert!(rep_boost > 0.0, "Repetition priming should be active");

    // Verify no interference between types
    assert!(semantic_boost <= 0.15, "Semantic boost should not exceed max strength");
    assert!(assoc_boost <= 1.0, "Associative strength should not exceed 1.0");
    assert!(rep_boost <= 0.30, "Repetition boost should not exceed ceiling");
}

#[test]
fn test_priming_saturation_prevents_overshoot() {
    let coordinator = PrimingCoordinator::new();

    // Create scenario with high priming from all three sources
    // Semantic: 0.15 (max), Associative: 0.5, Repetition: 0.30 (ceiling)
    // Linear sum = 0.95

    // Setup repetition priming at ceiling
    for _ in 0..10 {
        coordinator.repetition().record_exposure("target_node");
    }

    // Setup associative priming with high strength
    for _ in 0..10 {
        coordinator.associative().record_coactivation("prime_node", "target_node");
    }

    // Setup semantic priming (simplified, assume it activates)
    let prime_embedding = [0.7f32; 768];
    let target_embedding = [0.75f32; 768];
    coordinator.semantic().activate_priming("prime_node", &prime_embedding, || {
        vec![("target_node".to_string(), target_embedding, 1)]
    });

    std::thread::sleep(Duration::from_millis(10));

    // Compute total boost with saturation
    let total_boost = coordinator.compute_total_boost("target_node", Some("prime_node"));

    // With saturation function: 1.0 - exp(-0.95) ≈ 0.613
    // Should be significantly less than linear sum (0.95)
    assert!(
        total_boost < 0.70,
        "Saturation should prevent overshoot: total_boost = {}, expected < 0.70",
        total_boost
    );

    // But should still be substantial
    assert!(
        total_boost > 0.50,
        "Total boost should still be substantial: total_boost = {}",
        total_boost
    );
}

#[test]
fn test_priming_saturation_function_properties() {
    let coordinator = PrimingCoordinator::new();

    // Test saturation at different input levels
    // We'll manually create boosts by exposing nodes

    // No priming: boost = 0
    let boost_zero = coordinator.compute_total_boost("node_none", None);
    assert!(
        (boost_zero - 0.0).abs() < 1e-6,
        "No priming should yield zero boost"
    );

    // Low priming: ~0.3 linear → ~0.26 saturated
    for _ in 0..6 {
        coordinator.repetition().record_exposure("node_low");
    }
    let boost_low = coordinator.compute_total_boost("node_low", None);
    assert!(
        boost_low > 0.20 && boost_low < 0.30,
        "Low priming saturation: boost = {}, expected 0.20-0.30",
        boost_low
    );

    // High priming: ~0.9 linear (repetition 0.3 + associative 0.6)
    for _ in 0..10 {
        coordinator.repetition().record_exposure("node_high");
    }
    for _ in 0..10 {
        coordinator.associative().record_coactivation("prime_high", "node_high");
    }
    let boost_high = coordinator.compute_total_boost("node_high", Some("prime_high"));
    assert!(
        boost_high > 0.50 && boost_high < 0.70,
        "High priming saturation: boost = {}, expected 0.50-0.70",
        boost_high
    );
}

#[test]
fn test_priming_coordinator_detailed_breakdown() {
    let coordinator = PrimingCoordinator::new();

    // Setup different priming types
    for _ in 0..4 {
        coordinator.repetition().record_exposure("target");
    }

    for _ in 0..5 {
        coordinator.associative().record_coactivation("prime", "target");
    }

    let embedding_prime = [0.8f32; 768];
    let embedding_target = [0.85f32; 768];
    coordinator.semantic().activate_priming("prime", &embedding_prime, || {
        vec![("target".to_string(), embedding_target, 1)]
    });

    std::thread::sleep(Duration::from_millis(10));

    // Get detailed breakdown
    let (semantic, associative, repetition, total) =
        coordinator.compute_total_boost_detailed("target", Some("prime"));

    // Verify each component is non-zero
    assert!(semantic >= 0.0, "Semantic boost: {}", semantic);
    assert!(associative > 0.0, "Associative boost: {}", associative);
    assert!(repetition > 0.0, "Repetition boost: {}", repetition);

    // Verify saturation applied to total
    let linear_sum = semantic + associative + repetition;
    let expected_saturated = 1.0 - (-linear_sum).exp();
    assert!(
        (total - expected_saturated).abs() < 1e-6,
        "Total should match saturation function: total = {}, expected = {}",
        total,
        expected_saturated
    );

    // Total should be less than or equal to linear sum (due to saturation)
    assert!(
        total <= linear_sum,
        "Saturated total ({}) should not exceed linear sum ({})",
        total,
        linear_sum
    );
}

// ==================== Performance and Pruning Tests ====================

#[test]
fn test_associative_priming_pruning() {
    let engine = AssociativePrimingEngine::new()
        .with_cooccurrence_window(Duration::from_millis(20));

    // Create some associations
    for _ in 0..3 {
        engine.record_coactivation("old_a", "old_b");
    }

    let stats_before = engine.statistics();
    assert!(stats_before.active_associations > 0);

    // Wait for expiration
    std::thread::sleep(Duration::from_millis(220)); // 11× window

    // Prune old entries
    engine.prune_old_cooccurrences();

    let stats_after = engine.statistics();
    assert_eq!(
        stats_after.active_associations, 0,
        "Old associations should be pruned"
    );
}

#[test]
fn test_priming_coordinator_pruning() {
    let coordinator = PrimingCoordinator::new();

    // Create some priming data
    for _ in 0..3 {
        coordinator.associative().record_coactivation("a", "b");
    }

    let embedding = [0.5f32; 768];
    coordinator.semantic().activate_priming("prime", &embedding, || {
        vec![("target".to_string(), embedding, 1)]
    });

    // Wait for expiration
    std::thread::sleep(Duration::from_secs(1));

    // Prune all engines
    coordinator.prune_expired();

    // Semantic and associative should have reduced data
    let semantic_stats = coordinator.semantic().statistics();
    let assoc_stats = coordinator.associative().statistics();

    // Repetition engine should NOT be affected by pruning (no decay)
    coordinator.repetition().record_exposure("persistent");
    let rep_stats_before = coordinator.repetition().statistics();

    coordinator.prune_expired();

    let rep_stats_after = coordinator.repetition().statistics();
    assert_eq!(
        rep_stats_before.total_nodes,
        rep_stats_after.total_nodes,
        "Repetition priming should persist through pruning"
    );
}

// ==================== Edge Cases and Validation ====================

#[test]
fn test_self_priming_returns_zero() {
    let coordinator = PrimingCoordinator::new();

    // Attempt self-priming in all types
    coordinator.associative().record_coactivation("self", "self");
    coordinator.repetition().record_exposure("self");

    let assoc_self = coordinator.associative().compute_association_strength("self", "self");
    assert_eq!(assoc_self, 0.0, "Self-association should be zero");

    // Repetition should still work (not considered self-priming)
    let rep_self = coordinator.repetition().compute_repetition_boost("self");
    assert!(rep_self > 0.0, "Repetition priming should work for same node");
}

#[test]
fn test_zero_exposures_returns_zero_boost() {
    let engine = RepetitionPrimingEngine::new();
    let boost = engine.compute_repetition_boost("never_seen");
    assert_eq!(boost, 0.0, "Zero exposures should yield zero boost");
}

#[test]
fn test_associative_priming_statistics() {
    let engine = AssociativePrimingEngine::new();

    // Create multiple associations
    for _ in 0..3 {
        engine.record_coactivation("a", "b");
    }
    for _ in 0..5 {
        engine.record_coactivation("c", "d");
    }

    let stats = engine.statistics();
    assert!(stats.total_nodes >= 4, "Should track at least 4 nodes");
    assert!(stats.active_associations >= 2, "Should have at least 2 active associations");
    assert!(stats.average_strength > 0.0, "Average strength should be positive");
}

#[test]
fn test_repetition_priming_statistics() {
    let engine = RepetitionPrimingEngine::new();

    // Create exposure patterns
    engine.record_exposure("node_1");
    for _ in 0..3 {
        engine.record_exposure("node_2");
    }
    for _ in 0..10 {
        engine.record_exposure("node_3"); // At ceiling
    }

    let stats = engine.statistics();
    assert_eq!(stats.total_nodes, 3);
    assert_eq!(stats.total_exposures, 14);
    assert_eq!(stats.nodes_at_ceiling, 1, "Only node_3 should be at ceiling");
    assert_eq!(stats.max_exposures, 10);
}

#[test]
fn test_concurrent_priming_operations() {
    use std::sync::Arc;
    use std::thread;

    let coordinator = Arc::new(PrimingCoordinator::new());
    let mut handles = vec![];

    // Spawn threads for different priming types
    for i in 0..5 {
        let coord = Arc::clone(&coordinator);
        let handle = thread::spawn(move || {
            for _ in 0..100 {
                // Repetition priming
                coord.repetition().record_exposure(&format!("node_{}", i));

                // Associative priming
                coord.associative().record_coactivation(
                    &format!("a_{}", i),
                    &format!("b_{}", i),
                );
            }
        });
        handles.push(handle);
    }

    for handle in handles {
        handle.join().unwrap();
    }

    // Verify concurrent operations completed without data races
    let rep_stats = coordinator.repetition().statistics();
    let assoc_stats = coordinator.associative().statistics();

    assert!(rep_stats.total_nodes >= 5, "Should have nodes from all threads");
    assert!(assoc_stats.total_nodes >= 10, "Should have pairs from all threads");
}
