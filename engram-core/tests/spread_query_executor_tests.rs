//! Integration tests for SPREAD query executor
//!
//! Tests end-to-end execution of SPREAD queries from AST to results with
//! activation spreading, evidence tracking, and parameter configuration.

#![allow(clippy::panic)]

use chrono::Utc;
use engram_core::activation::{CognitiveRecall, ParallelSpreadingEngine};
use engram_core::memory_graph::{GraphConfig, UnifiedMemoryGraph};
use engram_core::query::executor::{QueryContext, SpreadExecutionError, execute_spread};
use engram_core::query::parser::ast::{NodeIdentifier, SpreadQuery};
use engram_core::{Confidence, Episode, Memory, MemorySpaceId, MemoryStore};
use std::sync::Arc;
use std::time::Duration;

/// Create a test memory store with a simple graph structure for testing
fn create_test_store() -> Arc<MemoryStore> {
    let space_id = MemorySpaceId::from("test_space");
    let store = MemoryStore::new(space_id, 1000);

    // Add some test memories
    let memories = vec![
        create_test_memory("node_a", "Concept A"),
        create_test_memory("node_b", "Concept B related to A"),
        create_test_memory("node_c", "Concept C related to B"),
        create_test_memory("node_d", "Concept D related to A"),
    ];

    for memory in memories {
        store.store(
            memory.clone(),
            Confidence::HIGH,
            #[cfg(feature = "hnsw_index")]
            None,
        );
    }

    // Setup cognitive recall with spreading engine
    let graph_config = GraphConfig::default();
    let graph = UnifiedMemoryGraph::new(graph_config);

    // Add edges to create a connected graph
    // A -> B, A -> D, B -> C
    let _ = graph.add_edge("node_a", "node_b", 0.8);
    let _ = graph.add_edge("node_a", "node_d", 0.7);
    let _ = graph.add_edge("node_b", "node_c", 0.6);

    let spreading_engine = Arc::new(ParallelSpreadingEngine::new(
        Arc::new(graph),
        Default::default(),
        None,
    ));

    let cognitive_recall = CognitiveRecall::builder()
        .spreading_engine(spreading_engine)
        .build();

    store.initialize_cognitive_recall(cognitive_recall);

    Arc::new(store)
}

fn create_test_memory(id: &str, content: &str) -> Memory {
    Memory {
        id: id.to_string(),
        content: content.to_string(),
        embedding: [0.1f32; 768],
        stored_at: Utc::now(),
        confidence: Confidence::HIGH,
        tags: Vec::new(),
    }
}

#[test]
fn test_spread_query_basic_execution() {
    let store = create_test_store();
    let context = QueryContext::without_timeout(MemorySpaceId::from("test_space"));

    let query = SpreadQuery {
        source: NodeIdentifier::from("node_a"),
        max_hops: Some(2),
        decay_rate: Some(0.1),
        activation_threshold: Some(0.01),
        refractory_period: None,
    };

    let result = execute_spread(&query, &context, &store);

    // Should succeed
    assert!(result.is_ok());
    let result = result.unwrap();

    // Should have activated some nodes (B and D at minimum from A)
    assert!(!result.is_empty());
    assert!(result.episodes.len() >= 2);
    assert!(!result.evidence_chain.is_empty());
    assert!(!result.uncertainty_sources.is_empty());
}

#[test]
fn test_spread_query_with_default_parameters() {
    let store = create_test_store();
    let context = QueryContext::without_timeout(MemorySpaceId::from("test_space"));

    let query = SpreadQuery {
        source: NodeIdentifier::from("node_a"),
        max_hops: None,             // Use default
        decay_rate: None,           // Use default
        activation_threshold: None, // Use default
        refractory_period: None,
    };

    let result = execute_spread(&query, &context, &store);

    assert!(result.is_ok());
    let result = result.unwrap();

    // Default parameters should still produce results
    assert!(!result.is_empty());
}

#[test]
fn test_spread_query_respects_max_hops() {
    let store = create_test_store();
    let context = QueryContext::without_timeout(MemorySpaceId::from("test_space"));

    // Spread with max_hops = 1 should only reach direct neighbors
    let query_1_hop = SpreadQuery {
        source: NodeIdentifier::from("node_a"),
        max_hops: Some(1),
        decay_rate: Some(0.1),
        activation_threshold: Some(0.01),
        refractory_period: None,
    };

    let result_1 = execute_spread(&query_1_hop, &context, &store).unwrap();

    // Spread with max_hops = 2 should reach farther
    let query_2_hop = SpreadQuery {
        source: NodeIdentifier::from("node_a"),
        max_hops: Some(2),
        decay_rate: Some(0.1),
        activation_threshold: Some(0.01),
        refractory_period: None,
    };

    let result_2 = execute_spread(&query_2_hop, &context, &store).unwrap();

    // 2-hop spread should potentially activate more nodes than 1-hop
    // (C is reachable from A in 2 hops via B)
    assert!(result_2.episodes.len() >= result_1.episodes.len());
}

#[test]
fn test_spread_query_respects_activation_threshold() {
    let store = create_test_store();
    let context = QueryContext::without_timeout(MemorySpaceId::from("test_space"));

    // High threshold - fewer results
    let query_high_threshold = SpreadQuery {
        source: NodeIdentifier::from("node_a"),
        max_hops: Some(3),
        decay_rate: Some(0.1),
        activation_threshold: Some(0.5), // Very high threshold
        refractory_period: None,
    };

    let result_high = execute_spread(&query_high_threshold, &context, &store).unwrap();

    // Low threshold - more results
    let query_low_threshold = SpreadQuery {
        source: NodeIdentifier::from("node_a"),
        max_hops: Some(3),
        decay_rate: Some(0.1),
        activation_threshold: Some(0.01), // Very low threshold
        refractory_period: None,
    };

    let result_low = execute_spread(&query_low_threshold, &context, &store).unwrap();

    // Lower threshold should include more activated nodes
    assert!(result_low.episodes.len() >= result_high.episodes.len());
}

#[test]
fn test_spread_query_source_not_found() {
    let store = create_test_store();
    let context = QueryContext::without_timeout(MemorySpaceId::from("test_space"));

    let query = SpreadQuery {
        source: NodeIdentifier::from("nonexistent_node"),
        max_hops: Some(2),
        decay_rate: Some(0.1),
        activation_threshold: Some(0.01),
        refractory_period: None,
    };

    let result = execute_spread(&query, &context, &store);

    // Should fail with SourceNodeNotFound error
    assert!(result.is_err());
    match result.unwrap_err() {
        SpreadExecutionError::SourceNodeNotFound(id) => {
            assert_eq!(id, "nonexistent_node");
        }
        other => panic!("Expected SourceNodeNotFound error, got {:?}", other),
    }
}

#[test]
fn test_spread_query_evidence_chain_populated() {
    let store = create_test_store();
    let context = QueryContext::without_timeout(MemorySpaceId::from("test_space"));

    let query = SpreadQuery {
        source: NodeIdentifier::from("node_a"),
        max_hops: Some(2),
        decay_rate: Some(0.1),
        activation_threshold: Some(0.01),
        refractory_period: None,
    };

    let result = execute_spread(&query, &context, &store).unwrap();

    // Evidence chain should be populated
    assert!(!result.evidence_chain.is_empty());

    // All evidence should be from spreading activation
    use engram_core::query::EvidenceSource;
    for evidence in &result.evidence_chain {
        match &evidence.source {
            EvidenceSource::SpreadingActivation {
                source_episode,
                activation_level,
                path_length,
            } => {
                assert_eq!(source_episode, "node_a");
                assert!(activation_level.value() > 0.0);
                assert!(*path_length > 0); // Should have path length > 0
            }
            _ => panic!("Expected SpreadingActivation evidence"),
        }
    }
}

#[test]
fn test_spread_query_uncertainty_sources() {
    let store = create_test_store();
    let context = QueryContext::without_timeout(MemorySpaceId::from("test_space"));

    let query = SpreadQuery {
        source: NodeIdentifier::from("node_a"),
        max_hops: Some(2),
        decay_rate: Some(0.2),
        activation_threshold: Some(0.05),
        refractory_period: None,
    };

    let result = execute_spread(&query, &context, &store).unwrap();

    // Should have uncertainty sources from spreading dynamics
    assert!(!result.uncertainty_sources.is_empty());

    // Verify we have spreading-related uncertainty
    assert!(result.uncertainty_sources.len() >= 2);
}

#[test]
fn test_spread_query_confidence_interval() {
    let store = create_test_store();
    let context = QueryContext::without_timeout(MemorySpaceId::from("test_space"));

    let query = SpreadQuery {
        source: NodeIdentifier::from("node_a"),
        max_hops: Some(2),
        decay_rate: Some(0.1),
        activation_threshold: Some(0.01),
        refractory_period: None,
    };

    let result = execute_spread(&query, &context, &store).unwrap();

    // Confidence interval should be properly constructed
    assert!(result.confidence_interval.lower <= result.confidence_interval.point);
    assert!(result.confidence_interval.point <= result.confidence_interval.upper);
    assert!(result.confidence_interval.width >= 0.0);
}

#[test]
fn test_spread_query_no_spreading_engine() {
    // Create a store without cognitive recall initialized
    let space_id = MemorySpaceId::from("test_space");
    let store = Arc::new(MemoryStore::new(space_id, 1000));

    let context = QueryContext::without_timeout(MemorySpaceId::from("test_space"));

    let query = SpreadQuery {
        source: NodeIdentifier::from("node_a"),
        max_hops: Some(2),
        decay_rate: Some(0.1),
        activation_threshold: Some(0.01),
        refractory_period: None,
    };

    let result = execute_spread(&query, &context, &store);

    // Should fail with NoSpreadingEngine error
    assert!(result.is_err());
    match result.unwrap_err() {
        SpreadExecutionError::NoSpreadingEngine => {}
        other => panic!("Expected NoSpreadingEngine error, got {:?}", other),
    }
}

#[test]
fn test_spread_query_decay_rate_affects_results() {
    let store = create_test_store();
    let context = QueryContext::without_timeout(MemorySpaceId::from("test_space"));

    // Low decay rate - activation persists farther
    let query_low_decay = SpreadQuery {
        source: NodeIdentifier::from("node_a"),
        max_hops: Some(3),
        decay_rate: Some(0.05), // Low decay
        activation_threshold: Some(0.1),
        refractory_period: None,
    };

    let result_low = execute_spread(&query_low_decay, &context, &store).unwrap();

    // High decay rate - activation drops off quickly
    let query_high_decay = SpreadQuery {
        source: NodeIdentifier::from("node_a"),
        max_hops: Some(3),
        decay_rate: Some(0.5), // High decay
        activation_threshold: Some(0.1),
        refractory_period: None,
    };

    let result_high = execute_spread(&query_high_decay, &context, &store).unwrap();

    // Low decay should activate more distant nodes
    assert!(result_low.episodes.len() >= result_high.episodes.len());

    // Uncertainty from low decay should be lower
    assert!(result_low.confidence_interval.width <= result_high.confidence_interval.width);
}
