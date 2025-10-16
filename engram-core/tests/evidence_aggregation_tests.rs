//! Integration tests for evidence aggregation
//!
//! Tests end-to-end evidence aggregation with dependency tracking,
//! circular dependency detection, and Bayesian combination.

#![allow(clippy::panic)]
#![allow(clippy::field_reassign_with_default)]
#![allow(clippy::unwrap_used)] // Unwrap is acceptable in tests

use engram_core::Confidence;
use engram_core::query::dependency_graph::DependencyGraph;
use engram_core::query::evidence_aggregator::{EvidenceAggregator, EvidenceInput};
use engram_core::query::{EvidenceSource, MatchType, ProbabilisticError};
use std::time::SystemTime;

fn create_test_evidence(id: u64, strength: f32, dependencies: Vec<u64>) -> EvidenceInput {
    EvidenceInput {
        id,
        source: EvidenceSource::DirectMatch {
            cue_id: format!("test_cue_{id}"),
            similarity_score: strength,
            match_type: MatchType::Semantic,
        },
        strength: Confidence::from_raw(strength),
        timestamp: SystemTime::now(),
        dependencies,
    }
}

#[test]
fn test_end_to_end_independent_evidence_aggregation() {
    let aggregator = EvidenceAggregator::default();

    // Create 3 independent pieces of evidence
    let evidence = vec![
        create_test_evidence(0, 0.7, vec![]),
        create_test_evidence(1, 0.6, vec![]),
        create_test_evidence(2, 0.5, vec![]),
    ];

    let result = aggregator.aggregate_evidence(&evidence).unwrap();

    // All 3 should contribute
    assert_eq!(result.len(), 3);
    assert!(!result.is_empty());
    assert!(!result.has_circular_dependencies);

    // Combined confidence should be higher than any individual
    assert!(result.aggregate_confidence.raw() > 0.7);

    // Verify dependency order contains all evidence
    assert_eq!(result.dependency_order.len(), 3);
}

#[test]
fn test_end_to_end_dependent_evidence_chain() {
    let aggregator = EvidenceAggregator::default();

    // Create a dependency chain: 0 -> 1 -> 2
    let evidence = vec![
        create_test_evidence(0, 0.8, vec![]),  // Independent
        create_test_evidence(1, 0.7, vec![0]), // Depends on 0
        create_test_evidence(2, 0.6, vec![1]), // Depends on 1
    ];

    let result = aggregator.aggregate_evidence(&evidence).unwrap();

    assert_eq!(result.len(), 3);
    assert!(!result.has_circular_dependencies);

    // Verify dependency ordering: 0 must come before 1, 1 before 2
    let order = &result.dependency_order;
    let pos_0 = order.iter().position(|&x| x == 0).unwrap();
    let pos_1 = order.iter().position(|&x| x == 1).unwrap();
    let pos_2 = order.iter().position(|&x| x == 2).unwrap();

    assert!(pos_0 < pos_1, "Evidence 0 must come before 1");
    assert!(pos_1 < pos_2, "Evidence 1 must come before 2");
}

#[test]
fn test_end_to_end_dag_with_multiple_dependencies() {
    let aggregator = EvidenceAggregator::default();

    // Create a diamond-shaped DAG:
    //     0
    //    / \
    //   1   2
    //    \ /
    //     3
    let evidence = vec![
        create_test_evidence(0, 0.9, vec![]),     // Root
        create_test_evidence(1, 0.8, vec![0]),    // Depends on 0
        create_test_evidence(2, 0.7, vec![0]),    // Depends on 0
        create_test_evidence(3, 0.6, vec![1, 2]), // Depends on both 1 and 2
    ];

    let result = aggregator.aggregate_evidence(&evidence).unwrap();

    assert_eq!(result.len(), 4);
    assert!(!result.has_circular_dependencies);

    // Verify topological ordering
    let order = &result.dependency_order;
    let pos_0 = order.iter().position(|&x| x == 0).unwrap();
    let pos_1 = order.iter().position(|&x| x == 1).unwrap();
    let pos_2 = order.iter().position(|&x| x == 2).unwrap();
    let pos_3 = order.iter().position(|&x| x == 3).unwrap();

    assert!(pos_0 < pos_1, "0 must come before 1");
    assert!(pos_0 < pos_2, "0 must come before 2");
    assert!(pos_1 < pos_3, "1 must come before 3");
    assert!(pos_2 < pos_3, "2 must come before 3");
}

#[test]
fn test_circular_dependency_detection_simple_cycle() {
    let aggregator = EvidenceAggregator::default();

    // Create a simple cycle: 0 -> 1 -> 0
    let evidence = vec![
        create_test_evidence(0, 0.8, vec![1]),
        create_test_evidence(1, 0.7, vec![0]),
    ];

    let result = aggregator.aggregate_evidence(&evidence);

    assert!(matches!(
        result,
        Err(ProbabilisticError::CircularDependency)
    ));
}

#[test]
fn test_circular_dependency_detection_three_node_cycle() {
    let aggregator = EvidenceAggregator::default();

    // Create a 3-node cycle: 0 -> 1 -> 2 -> 0
    let evidence = vec![
        create_test_evidence(0, 0.8, vec![1]),
        create_test_evidence(1, 0.7, vec![2]),
        create_test_evidence(2, 0.6, vec![0]),
    ];

    let result = aggregator.aggregate_evidence(&evidence);

    assert!(matches!(
        result,
        Err(ProbabilisticError::CircularDependency)
    ));
}

#[test]
fn test_circular_dependency_detection_self_loop() {
    let aggregator = EvidenceAggregator::default();

    // Create a self-loop
    let evidence = vec![create_test_evidence(0, 0.8, vec![0])];

    let result = aggregator.aggregate_evidence(&evidence);

    assert!(matches!(
        result,
        Err(ProbabilisticError::CircularDependency)
    ));
}

#[test]
fn test_bayesian_combination_increases_confidence() {
    let aggregator = EvidenceAggregator::default();

    // Two independent pieces of moderate evidence
    let evidence = vec![
        create_test_evidence(0, 0.6, vec![]),
        create_test_evidence(1, 0.5, vec![]),
    ];

    let result = aggregator.aggregate_evidence(&evidence).unwrap();

    // Combined confidence should be higher than either individual piece
    assert!(result.aggregate_confidence.raw() > 0.6);
    assert!(result.aggregate_confidence.raw() > 0.5);

    // Should be approximately 1 - (1-0.6)*(1-0.5) = 0.8
    let expected = 0.8;
    assert!((result.aggregate_confidence.raw() - expected).abs() < 1e-5);
}

#[test]
fn test_bayesian_combination_three_independent_sources() {
    let aggregator = EvidenceAggregator::default();

    let evidence = vec![
        create_test_evidence(0, 0.5, vec![]),
        create_test_evidence(1, 0.4, vec![]),
        create_test_evidence(2, 0.3, vec![]),
    ];

    let result = aggregator.aggregate_evidence(&evidence).unwrap();

    // Expected: 1 - (1-0.5)*(1-0.4)*(1-0.3) = 1 - 0.5*0.6*0.7 = 1 - 0.21 = 0.79
    let expected = 0.79;
    assert!((result.aggregate_confidence.raw() - expected).abs() < 1e-5);
}

#[test]
fn test_min_strength_filtering() {
    let aggregator = EvidenceAggregator::new(Confidence::from_raw(0.5), 10);

    let evidence = vec![
        create_test_evidence(0, 0.3, vec![]), // Below threshold
        create_test_evidence(1, 0.4, vec![]), // Below threshold
        create_test_evidence(2, 0.7, vec![]), // Above threshold
        create_test_evidence(3, 0.8, vec![]), // Above threshold
    ];

    let result = aggregator.aggregate_evidence(&evidence).unwrap();

    // Only evidence 2 and 3 should contribute
    assert_eq!(result.len(), 2);

    let contributing_ids: Vec<u64> = result
        .contributing_evidence
        .iter()
        .map(|c| c.evidence_id)
        .collect();

    assert!(contributing_ids.contains(&2));
    assert!(contributing_ids.contains(&3));
    assert!(!contributing_ids.contains(&0));
    assert!(!contributing_ids.contains(&1));
}

#[test]
fn test_max_evidence_limit_enforcement() {
    let aggregator = EvidenceAggregator::new(Confidence::from_raw(0.01), 3);

    let evidence = vec![
        create_test_evidence(0, 0.9, vec![]),
        create_test_evidence(1, 0.8, vec![]),
        create_test_evidence(2, 0.7, vec![]),
        create_test_evidence(3, 0.6, vec![]),
        create_test_evidence(4, 0.5, vec![]),
    ];

    let result = aggregator.aggregate_evidence(&evidence).unwrap();

    // Only 3 evidence sources should contribute
    assert!(result.len() <= 3);
}

#[test]
fn test_mixed_independent_and_dependent_evidence() {
    let aggregator = EvidenceAggregator::default();

    // Mix of independent and dependent evidence
    let evidence = vec![
        create_test_evidence(0, 0.8, vec![]),  // Independent
        create_test_evidence(1, 0.7, vec![]),  // Independent
        create_test_evidence(2, 0.6, vec![0]), // Depends on 0
        create_test_evidence(3, 0.5, vec![]),  // Independent
    ];

    let result = aggregator.aggregate_evidence(&evidence).unwrap();

    assert_eq!(result.len(), 4);
    assert!(!result.has_circular_dependencies);

    // Verify dependency ordering: 0 must come before 2
    let order = &result.dependency_order;
    let pos_0 = order.iter().position(|&x| x == 0).unwrap();
    let pos_2 = order.iter().position(|&x| x == 2).unwrap();
    assert!(pos_0 < pos_2);
}

#[test]
fn test_evidence_contribution_tracking() {
    let aggregator = EvidenceAggregator::default();

    let evidence = vec![
        create_test_evidence(0, 0.8, vec![]),
        create_test_evidence(1, 0.7, vec![0]),
    ];

    let result = aggregator.aggregate_evidence(&evidence).unwrap();

    // Check contribution details
    assert_eq!(result.contributing_evidence.len(), 2);

    for contrib in &result.contributing_evidence {
        assert!(contrib.original_strength.raw() > 0.0);
        assert!(contrib.posterior_strength.raw() > 0.0);
        assert!(contrib.original_strength.raw() <= 1.0);
        assert!(contrib.posterior_strength.raw() <= 1.0);

        // Verify source is correctly tracked
        if let EvidenceSource::DirectMatch {
            similarity_score, ..
        } = contrib.source
        {
            assert!(similarity_score > 0.0);
            assert!(similarity_score <= 1.0);
        } else {
            panic!("Expected DirectMatch evidence source");
        }
    }
}

#[test]
fn test_empty_evidence_returns_error() {
    let aggregator = EvidenceAggregator::default();
    let result = aggregator.aggregate_evidence(&[]);

    assert!(matches!(
        result,
        Err(ProbabilisticError::InsufficientEvidence)
    ));
}

#[test]
fn test_single_evidence_passthrough() {
    let aggregator = EvidenceAggregator::default();
    let evidence = vec![create_test_evidence(0, 0.75, vec![])];

    let result = aggregator.aggregate_evidence(&evidence).unwrap();

    assert_eq!(result.len(), 1);
    assert!((result.aggregate_confidence.raw() - 0.75).abs() < 1e-6);
}

#[test]
fn test_dependency_graph_integration() {
    // Test that the internal dependency graph correctly handles complex dependencies

    let mut graph = DependencyGraph::new();
    graph.add_evidence(0, vec![]);
    graph.add_evidence(1, vec![0]);
    graph.add_evidence(2, vec![1]);
    graph.add_evidence(3, vec![0]);

    // Should not have cycles
    assert!(!graph.has_cycles());

    // Get topological order
    let order = graph.topological_sort().unwrap();

    // Verify ordering constraints
    let pos_0 = order.iter().position(|&x| x == 0).unwrap();
    let pos_1 = order.iter().position(|&x| x == 1).unwrap();
    let pos_2 = order.iter().position(|&x| x == 2).unwrap();
    let pos_3 = order.iter().position(|&x| x == 3).unwrap();

    assert!(pos_0 < pos_1);
    assert!(pos_1 < pos_2);
    assert!(pos_0 < pos_3);
}

#[test]
fn test_very_weak_evidence_filtered() {
    let aggregator = EvidenceAggregator::new(Confidence::from_raw(0.1), 10);

    let evidence = vec![
        create_test_evidence(0, 0.05, vec![]), // Very weak
        create_test_evidence(1, 0.02, vec![]), // Very weak
        create_test_evidence(2, 0.8, vec![]),  // Strong
    ];

    let result = aggregator.aggregate_evidence(&evidence).unwrap();

    // Only strong evidence should contribute
    assert_eq!(result.len(), 1);
    assert_eq!(result.contributing_evidence[0].evidence_id, 2);
}

#[test]
fn test_all_weak_evidence_returns_error() {
    let aggregator = EvidenceAggregator::new(Confidence::from_raw(0.5), 10);

    let evidence = vec![
        create_test_evidence(0, 0.3, vec![]),
        create_test_evidence(1, 0.2, vec![]),
        create_test_evidence(2, 0.1, vec![]),
    ];

    let result = aggregator.aggregate_evidence(&evidence);

    assert!(matches!(
        result,
        Err(ProbabilisticError::InsufficientEvidence)
    ));
}

#[test]
fn test_large_evidence_set() {
    let aggregator = EvidenceAggregator::new(Confidence::from_raw(0.01), 20);

    // Create 15 independent pieces of evidence
    let evidence: Vec<_> = (0..15)
        .map(|i| create_test_evidence(i, 0.3 + (i as f32 * 0.02), vec![]))
        .collect();

    let result = aggregator.aggregate_evidence(&evidence).unwrap();

    assert_eq!(result.len(), 15);

    // Combined confidence should be very high with 15 independent sources
    assert!(result.aggregate_confidence.raw() > 0.9);
}

#[test]
fn test_complex_dag_structure() {
    let aggregator = EvidenceAggregator::default();

    // Complex DAG:
    //       0
    //      /|\
    //     1 2 3
    //      \|/
    //       4
    let evidence = vec![
        create_test_evidence(0, 0.9, vec![]),
        create_test_evidence(1, 0.8, vec![0]),
        create_test_evidence(2, 0.7, vec![0]),
        create_test_evidence(3, 0.6, vec![0]),
        create_test_evidence(4, 0.5, vec![1, 2, 3]),
    ];

    let result = aggregator.aggregate_evidence(&evidence).unwrap();

    assert_eq!(result.len(), 5);
    assert!(!result.has_circular_dependencies);

    // Verify 0 comes before all others
    let order = &result.dependency_order;
    let pos_0 = order.iter().position(|&x| x == 0).unwrap();

    for &id in &[1, 2, 3, 4] {
        let pos = order.iter().position(|&x| x == id).unwrap();
        assert!(pos_0 < pos, "0 must come before {id}");
    }

    // Verify 4 comes after 1, 2, and 3
    let pos_4 = order.iter().position(|&x| x == 4).unwrap();
    for &id in &[1, 2, 3] {
        let pos = order.iter().position(|&x| x == id).unwrap();
        assert!(pos < pos_4, "{id} must come before 4");
    }
}
