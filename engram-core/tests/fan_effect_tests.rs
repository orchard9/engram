//! Comprehensive fan effect integration tests
//!
//! Validates Anderson (1974) empirical findings using Engram's memory graph.

#![allow(clippy::float_cmp)]

use engram_core::Confidence;
use engram_core::cognitive::interference::{
    FanEffectDetector, FanEffectResult, FanEffectStatistics,
};
use engram_core::memory::Memory;
use engram_core::memory_graph::UnifiedMemoryGraph;

#[test]
fn test_anderson_1974_retrieval_times_match_empirical_data() {
    let detector = FanEffectDetector::default();

    // Anderson (1974) Table 1 data
    // Fan 1: 1159ms ± 22ms (baseline)
    // Fan 2: 1236ms ± 25ms (+77ms)
    // Fan 3: 1305ms ± 28ms (+69ms average from fan 2)
    let empirical_data = vec![
        (1, 1159.0, 22.0), // (fan, mean_rt, std_dev)
        (2, 1236.0, 25.0),
        (3, 1305.0, 28.0),
    ];

    for (fan, expected_rt, std_dev) in empirical_data {
        let predicted_rt = detector.compute_retrieval_time_ms(fan);

        // Allow ±20ms tolerance (within 1 std dev of empirical data)
        let tolerance = 20.0;
        assert!(
            (predicted_rt - expected_rt).abs() < tolerance,
            "Fan {}: Expected {}ms ± {}ms, got {}ms (diff: {:.1}ms)",
            fan,
            expected_rt,
            std_dev,
            predicted_rt,
            predicted_rt - expected_rt
        );
    }
}

#[test]
fn test_linear_scaling_70ms_per_association() {
    let detector = FanEffectDetector::default();

    // Test linear relationship with 70ms slope
    let rt1 = detector.compute_retrieval_time_ms(1);
    let rt2 = detector.compute_retrieval_time_ms(2);
    let rt3 = detector.compute_retrieval_time_ms(3);
    let rt4 = detector.compute_retrieval_time_ms(4);
    let rt5 = detector.compute_retrieval_time_ms(5);

    // Verify base time
    assert_eq!(rt1, 1150.0, "Base time (fan=1) should be 1150ms");

    // Slope should be constant (~70ms)
    let slope_1_2 = rt2 - rt1;
    let slope_2_3 = rt3 - rt2;
    let slope_3_4 = rt4 - rt3;
    let slope_4_5 = rt5 - rt4;

    assert_eq!(slope_1_2, 70.0, "Slope 1→2 should be exactly 70ms");
    assert_eq!(slope_2_3, 70.0, "Slope 2→3 should be exactly 70ms");
    assert_eq!(slope_3_4, 70.0, "Slope 3→4 should be exactly 70ms");
    assert_eq!(slope_4_5, 70.0, "Slope 4→5 should be exactly 70ms");

    // Verify formula: RT = 1150 + (fan - 1) × 70
    assert_eq!(rt2, 1220.0); // 1150 + 70
    assert_eq!(rt3, 1290.0); // 1150 + 140
    assert_eq!(rt4, 1360.0); // 1150 + 210
    assert_eq!(rt5, 1430.0); // 1150 + 280
}

#[test]
fn test_fan_computation_from_graph_structure() {
    let detector = FanEffectDetector::default();
    let graph = UnifiedMemoryGraph::concurrent();

    // Create Anderson (1974) scenario:
    // The doctor is in the bank, church, park (doctor has fan=3)
    let doctor_id = graph
        .store_memory(Memory::new(
            "The doctor".to_string(),
            [0.1; 768],
            Confidence::HIGH,
        ))
        .unwrap();

    let bank_id = graph
        .store_memory(Memory::new(
            "bank".to_string(),
            [0.2; 768],
            Confidence::HIGH,
        ))
        .unwrap();

    let church_id = graph
        .store_memory(Memory::new(
            "church".to_string(),
            [0.3; 768],
            Confidence::HIGH,
        ))
        .unwrap();

    let park_id = graph
        .store_memory(Memory::new(
            "park".to_string(),
            [0.4; 768],
            Confidence::HIGH,
        ))
        .unwrap();

    // Add edges (associations)
    graph.add_edge(doctor_id, bank_id, 1.0).unwrap();
    graph.add_edge(doctor_id, church_id, 1.0).unwrap();
    graph.add_edge(doctor_id, park_id, 1.0).unwrap();

    // Compute fan
    let fan = detector.compute_fan(&doctor_id, &graph);
    assert_eq!(
        fan, 3,
        "Doctor with 3 location associations should have fan=3"
    );

    // Compute retrieval time
    let fan_effect = detector.detect_fan_effect(&doctor_id, &graph);
    assert_eq!(fan_effect.fan, 3);
    assert_eq!(
        fan_effect.retrieval_time_ms, 1290.0,
        "Fan=3 should predict 1290ms retrieval time"
    );
    assert_eq!(
        fan_effect.activation_divisor, 3.0,
        "Fan=3 should divide activation by 3"
    );
}

#[test]
fn test_activation_division_among_associations() {
    let detector = FanEffectDetector::default();
    let base_activation = 1.0;

    // Helper to create result
    let make_result = |fan: usize| -> FanEffectResult {
        FanEffectResult {
            fan,
            retrieval_time_ms: detector.compute_retrieval_time_ms(fan),
            activation_divisor: detector.compute_activation_divisor(fan),
        }
    };

    // Fan = 1: Full activation (no division)
    let fan1_result = make_result(1);
    let activation_fan1 = detector.apply_to_activation(base_activation, &fan1_result);
    assert_eq!(
        activation_fan1, 1.0,
        "Fan=1 should preserve full activation"
    );

    // Fan = 2: Half activation per edge
    let fan2_result = make_result(2);
    let activation_fan2 = detector.apply_to_activation(base_activation, &fan2_result);
    assert_eq!(activation_fan2, 0.5, "Fan=2 should divide activation by 2");

    // Fan = 3: One-third activation per edge
    let fan3_result = make_result(3);
    let activation_fan3 = detector.apply_to_activation(base_activation, &fan3_result);
    assert!(
        (activation_fan3 - (1.0 / 3.0)).abs() < 0.0001,
        "Fan=3 should divide activation by 3"
    );

    // Fan = 5: One-fifth activation per edge
    let fan5_result = make_result(5);
    let activation_fan5 = detector.apply_to_activation(base_activation, &fan5_result);
    assert_eq!(activation_fan5, 0.2, "Fan=5 should divide activation by 5");
}

#[test]
fn test_sqrt_divisor_mode_softer_falloff() {
    let mut detector = FanEffectDetector::default();

    // Linear mode: fan=9 → divisor=9
    let linear_divisor = detector.compute_activation_divisor(9);
    assert_eq!(linear_divisor, 9.0, "Linear mode should use fan directly");

    // Sqrt mode: fan=9 → divisor=3
    detector.set_sqrt_divisor(true);
    let sqrt_divisor = detector.compute_activation_divisor(9);
    assert_eq!(sqrt_divisor, 3.0, "Sqrt mode should use sqrt(fan)");

    // Application to activation
    let base_activation = 1.0;
    let result = FanEffectResult {
        fan: 9,
        retrieval_time_ms: detector.compute_retrieval_time_ms(9),
        activation_divisor: sqrt_divisor,
    };
    let activation = detector.apply_to_activation(base_activation, &result);
    assert!(
        (activation - (1.0 / 3.0)).abs() < 0.0001,
        "Sqrt mode: activation/sqrt(9) = 1/3"
    );

    // Verify sqrt provides softer falloff for high fan
    detector.set_sqrt_divisor(false);
    let linear_result = FanEffectResult {
        fan: 16,
        retrieval_time_ms: detector.compute_retrieval_time_ms(16),
        activation_divisor: detector.compute_activation_divisor(16),
    };
    let linear_activation = detector.apply_to_activation(base_activation, &linear_result);

    detector.set_sqrt_divisor(true);
    let sqrt_result = FanEffectResult {
        fan: 16,
        retrieval_time_ms: detector.compute_retrieval_time_ms(16),
        activation_divisor: detector.compute_activation_divisor(16),
    };
    let sqrt_activation = detector.apply_to_activation(base_activation, &sqrt_result);

    // linear: 1/16 = 0.0625
    // sqrt: 1/4 = 0.25
    assert_eq!(linear_activation, 1.0 / 16.0);
    assert_eq!(sqrt_activation, 0.25);
    assert!(
        sqrt_activation > linear_activation,
        "Sqrt mode should preserve more activation for high fan"
    );
}

#[test]
fn test_fan_statistics_computation() {
    let detector = FanEffectDetector::default();
    let graph = UnifiedMemoryGraph::concurrent();

    // Create nodes with varying fan counts
    let n1_id = graph
        .store_memory(Memory::new(
            "node1".to_string(),
            [0.1; 768],
            Confidence::HIGH,
        ))
        .unwrap();
    let n2_id = graph
        .store_memory(Memory::new(
            "node2".to_string(),
            [0.2; 768],
            Confidence::HIGH,
        ))
        .unwrap();
    let n3_id = graph
        .store_memory(Memory::new(
            "node3".to_string(),
            [0.3; 768],
            Confidence::HIGH,
        ))
        .unwrap();
    let n4_id = graph
        .store_memory(Memory::new(
            "node4".to_string(),
            [0.4; 768],
            Confidence::HIGH,
        ))
        .unwrap();

    let target1_id = graph
        .store_memory(Memory::new(
            "target1".to_string(),
            [0.5; 768],
            Confidence::HIGH,
        ))
        .unwrap();
    let target2_id = graph
        .store_memory(Memory::new(
            "target2".to_string(),
            [0.6; 768],
            Confidence::HIGH,
        ))
        .unwrap();
    let target3_id = graph
        .store_memory(Memory::new(
            "target3".to_string(),
            [0.7; 768],
            Confidence::HIGH,
        ))
        .unwrap();

    // Create edge structure:
    // n1 → target1 (fan=1)
    // n2 → target1, target2 (fan=2)
    // n3 → target1, target2, target3 (fan=3)
    // n4 → target1 (fan=1)
    graph.add_edge(n1_id, target1_id, 1.0).unwrap();
    graph.add_edge(n2_id, target1_id, 1.0).unwrap();
    graph.add_edge(n2_id, target2_id, 1.0).unwrap();
    graph.add_edge(n3_id, target1_id, 1.0).unwrap();
    graph.add_edge(n3_id, target2_id, 1.0).unwrap();
    graph.add_edge(n3_id, target3_id, 1.0).unwrap();
    graph.add_edge(n4_id, target1_id, 1.0).unwrap();

    // Compute statistics
    let stats = FanEffectStatistics::compute(&graph, &detector);

    // Verify distribution
    // Nodes with edges: n1(1), n2(2), n3(3), n4(1)
    // Nodes without edges: target1(0→1), target2(0→1), target3(0→1)
    assert!(
        stats.fan_distribution.get(&1).unwrap_or(&0) >= &2,
        "At least two nodes with fan=1"
    );
    assert_eq!(
        stats.fan_distribution.get(&2),
        Some(&1),
        "One node with fan=2"
    );
    assert_eq!(
        stats.fan_distribution.get(&3),
        Some(&1),
        "One node with fan=3"
    );

    // Verify max fan
    assert_eq!(stats.max_fan, 3, "Maximum fan should be 3");

    // Verify high fan nodes (fan > 3)
    assert!(
        stats.high_fan_nodes.is_empty(),
        "No nodes should have fan > 3 in this test"
    );

    // Verify average is computed
    assert!(stats.average_fan > 0.0, "Average fan should be positive");
}

#[test]
fn test_retrieval_stage_integration() {
    let detector = FanEffectDetector::default();
    let graph = UnifiedMemoryGraph::concurrent();

    // Create high-fan scenario (Anderson 1974 style)
    let doctor_id = graph
        .store_memory(Memory::new(
            "The doctor".to_string(),
            [0.5; 768],
            Confidence::HIGH,
        ))
        .unwrap();

    let locations = vec![
        graph
            .store_memory(Memory::new(
                "bank".to_string(),
                [0.3; 768],
                Confidence::HIGH,
            ))
            .unwrap(),
        graph
            .store_memory(Memory::new(
                "church".to_string(),
                [0.3; 768],
                Confidence::HIGH,
            ))
            .unwrap(),
        graph
            .store_memory(Memory::new(
                "park".to_string(),
                [0.3; 768],
                Confidence::HIGH,
            ))
            .unwrap(),
    ];

    // Create associations
    for loc_id in &locations {
        graph.add_edge(doctor_id, *loc_id, 0.8).unwrap();
    }

    // Detect fan effect
    let fan_effect = detector.detect_fan_effect(&doctor_id, &graph);

    // Verify fan=3
    assert_eq!(
        fan_effect.fan, 3,
        "Doctor with 3 locations should have fan=3"
    );

    // Verify predicted retrieval time matches Anderson (1974)
    assert_eq!(
        fan_effect.retrieval_time_ms, 1290.0,
        "Fan=3 should predict 1290ms (matches Anderson 1974 within tolerance)"
    );

    // Verify activation division
    assert_eq!(
        fan_effect.activation_divisor, 3.0,
        "Fan=3 should divide activation by 3"
    );

    // Verify activation application
    let base_activation = 1.0;
    let divided_activation = detector.apply_to_activation(base_activation, &fan_effect);
    assert!(
        (divided_activation - (1.0 / 3.0)).abs() < 0.0001,
        "Activation should be divided evenly among 3 associations"
    );
}

#[test]
fn test_high_fan_detection() {
    let detector = FanEffectDetector::default();

    // Test is_high_fan() helper
    let low_fan = FanEffectResult {
        fan: 2,
        retrieval_time_ms: detector.compute_retrieval_time_ms(2),
        activation_divisor: 2.0,
    };
    assert!(!low_fan.is_high_fan(), "Fan=2 should not be high");

    let medium_fan = FanEffectResult {
        fan: 3,
        retrieval_time_ms: detector.compute_retrieval_time_ms(3),
        activation_divisor: 3.0,
    };
    assert!(!medium_fan.is_high_fan(), "Fan=3 should not be high");

    let high_fan = FanEffectResult {
        fan: 4,
        retrieval_time_ms: detector.compute_retrieval_time_ms(4),
        activation_divisor: 4.0,
    };
    assert!(high_fan.is_high_fan(), "Fan=4 should be high");

    let very_high_fan = FanEffectResult {
        fan: 10,
        retrieval_time_ms: detector.compute_retrieval_time_ms(10),
        activation_divisor: 10.0,
    };
    assert!(very_high_fan.is_high_fan(), "Fan=10 should be high");
}

#[test]
fn test_slowdown_calculation() {
    let detector = FanEffectDetector::default();
    let baseline_ms = 1150.0;

    let fan1 = FanEffectResult {
        fan: 1,
        retrieval_time_ms: detector.compute_retrieval_time_ms(1),
        activation_divisor: 1.0,
    };
    assert_eq!(
        fan1.slowdown_ms(baseline_ms),
        0.0,
        "Fan=1 should have no slowdown"
    );

    let fan2 = FanEffectResult {
        fan: 2,
        retrieval_time_ms: detector.compute_retrieval_time_ms(2),
        activation_divisor: 2.0,
    };
    assert_eq!(
        fan2.slowdown_ms(baseline_ms),
        70.0,
        "Fan=2 should have 70ms slowdown"
    );

    let fan3 = FanEffectResult {
        fan: 3,
        retrieval_time_ms: detector.compute_retrieval_time_ms(3),
        activation_divisor: 3.0,
    };
    assert_eq!(
        fan3.slowdown_ms(baseline_ms),
        140.0,
        "Fan=3 should have 140ms slowdown"
    );
}

#[test]
fn test_custom_detector_parameters() {
    // Test with custom base time and time per association
    let custom_detector = FanEffectDetector::new(1000.0, 50.0, false);

    let rt1 = custom_detector.compute_retrieval_time_ms(1);
    let rt2 = custom_detector.compute_retrieval_time_ms(2);
    let rt3 = custom_detector.compute_retrieval_time_ms(3);

    assert_eq!(rt1, 1000.0, "Custom base time should be 1000ms");
    assert_eq!(
        rt2 - rt1,
        50.0,
        "Custom time per association should be 50ms"
    );
    assert_eq!(rt3 - rt2, 50.0, "Custom slope should be consistent");
}

#[test]
fn test_minimum_fan_is_one() {
    let detector = FanEffectDetector::default();
    let graph = UnifiedMemoryGraph::concurrent();

    // Create isolated node with no edges
    let isolated_id = graph
        .store_memory(Memory::new(
            "isolated".to_string(),
            [0.1; 768],
            Confidence::HIGH,
        ))
        .unwrap();

    // Fan should be clamped to minimum of 1
    let fan = detector.compute_fan(&isolated_id, &graph);
    assert_eq!(fan, 1, "Isolated node should have minimum fan of 1");

    let fan_effect = detector.detect_fan_effect(&isolated_id, &graph);
    assert_eq!(
        fan_effect.fan, 1,
        "Fan effect should report minimum fan of 1"
    );
    assert_eq!(
        fan_effect.retrieval_time_ms, 1150.0,
        "Minimum fan should use base retrieval time"
    );
}
