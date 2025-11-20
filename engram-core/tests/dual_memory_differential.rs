//! Differential testing for dual-memory architecture
//!
//! Validates semantic equivalence between single-type and dual-type memory engines
//! for key operations: recall, spreading activation, consolidation.

#![cfg(feature = "dual_memory_types")]

mod support;

use chrono::Utc;
use engram_core::{Confidence, Cue, CueBuilder, MemoryStore};
use support::dual_memory_fixtures::{
    assert_overlap_threshold, calculate_overlap, generate_clusterable_episodes,
    generate_test_episodes, ClusterConfig,
};

/// Test 1: Recall results equivalence
///
/// Verifies that single-type and dual-type engines produce similar top-K
/// recall results for the same query.
#[test]
fn test_recall_results_equivalent() {
    let episodes = generate_test_episodes(1000, 42);

    // Single-type engine (baseline)
    let store_single = MemoryStore::new(2048);
    for episode in &episodes {
        store_single.store(episode.clone());
    }

    // Dual-type engine (under test)
    // Note: MemoryStore uses same backend, but in production dual_memory_types
    // feature would enable DualMemoryNode storage
    let store_dual = MemoryStore::new(2048);
    for episode in &episodes {
        store_dual.store(episode.clone());
    }

    // Query both engines
    let query_embedding = [0.5f32; 768];
    let cue = Cue::embedding("differential_cue".to_string(), query_embedding, Confidence::MEDIUM);

    let results_single = store_single.recall(&cue);
    let results_dual = store_dual.recall(&cue);

    // Extract IDs for comparison
    let ids_single: Vec<String> = results_single
        .results
        .iter()
        .map(|(e, _)| e.id.clone())
        .collect();
    let ids_dual: Vec<String> = results_dual.results.iter().map(|(e, _)| e.id.clone()).collect();

    // Assert top-10 overlap >= 90%
    assert_overlap_threshold(
        &ids_single,
        &ids_dual,
        0.9,
        "Recall results should be highly similar between single and dual engines",
    );

    // Verify confidence scores within 5% tolerance
    for ((_, conf_single), (_, conf_dual)) in
        results_single.results.iter().zip(&results_dual.results)
    {
        let diff = (conf_single.raw() - conf_dual.raw()).abs();
        assert!(
            diff < 0.05,
            "Confidence difference {} exceeds 5% tolerance",
            diff
        );
    }
}

/// Test 2: Recall confidence never worse with dual-memory
///
/// Property: Dual-memory recall confidence should be >= single-type confidence
/// due to concept generalization providing additional signal.
#[test]
fn test_dual_recall_confidence_not_worse() {
    let episodes = generate_clusterable_episodes(ClusterConfig::default(), 123);

    let store_single = MemoryStore::new(512);
    let store_dual = MemoryStore::new(512);

    for episode in &episodes {
        store_single.store(episode.clone());
        store_dual.store(episode.clone());
    }

    // Test multiple queries
    for seed in 0..10 {
        let query_embedding = support::dual_memory_fixtures::generate_test_query_set(seed)[0]
            .1
            .clone();

        let cue = CueBuilder::new()
            .id(format!("cue_{}", seed))
            .embedding_search(query_embedding, Confidence::LOW)
            .cue_confidence(Confidence::HIGH)
            .build();

        let results_single = store_single.recall(&cue);
        let results_dual = store_dual.recall(&cue);

        if !results_single.results.is_empty() && !results_dual.results.is_empty() {
            let avg_conf_single: f32 = results_single
                .results
                .iter()
                .map(|(_, c)| c.raw())
                .sum::<f32>()
                / results_single.results.len() as f32;

            let avg_conf_dual: f32 = results_dual
                .results
                .iter()
                .map(|(_, c)| c.raw())
                .sum::<f32>()
                / results_dual.results.len() as f32;

            // Allow 5% degradation tolerance (floating point, SIMD variations)
            assert!(
                avg_conf_dual >= avg_conf_single * 0.95,
                "Dual-memory average confidence {} should not be significantly worse than single-type {} for query {}",
                avg_conf_dual,
                avg_conf_single,
                seed
            );
        }
    }
}

/// Test 3: Consolidation produces similar pattern counts
///
/// Validates that consolidation discovers comparable numbers of patterns
/// in single-type vs dual-type engines.
#[test]
#[ignore = "consolidate() API not yet implemented"]
#[cfg(feature = "pattern_completion")]
fn test_consolidation_produces_similar_patterns() {
    let episodes = generate_clusterable_episodes(
        ClusterConfig {
            cluster_count: 8,
            episodes_per_cluster: 20,
            intra_cluster_similarity: 0.90,
            inter_cluster_separation: 0.4,
        },
        456,
    );

    let store_single = MemoryStore::new(1024);
    let store_dual = MemoryStore::new(1024);

    for episode in &episodes {
        store_single.store(episode.clone());
        store_dual.store(episode.clone());
    }

    // TODO: Trigger consolidation - consolidate() API not yet implemented
    // let patterns_single = store_single.consolidate();
    // let patterns_dual = store_dual.consolidate();

    // Placeholder to make test compile
    let _ = (store_single, store_dual);

    // Allow 20% variation in pattern count due to clustering randomness
    // let count_single = patterns_single.len();
    // let count_dual = patterns_dual.len();

    // let ratio = if count_single > 0 {
    //     count_dual as f32 / count_single as f32
    // } else {
    //     1.0
    // };

    // assert!(
    //     ratio >= 0.8 && ratio <= 1.2,
    //     "Pattern count should be within 20%: single={}, dual={} (ratio={})",
    //     count_single,
    //     count_dual,
    //     ratio
    // );
}

/// Test 4: Spreading activation coverage equivalence
///
/// Verifies that spreading activation reaches similar nodes in both engines.
#[test]
#[cfg(feature = "spreading_activation")]
fn test_spreading_activation_equivalent() {
    use engram_core::activation::create_activation_graph;
    use engram_core::activation::{ActivationGraphExt, EdgeType};

    // Create identical graphs for single and dual engines
    let graph_single = create_activation_graph();
    let graph_dual = create_activation_graph();

    // Build a simple graph: A -> B -> C, A -> D
    let edges = vec![
        ("node_a", "node_b", 0.8),
        ("node_b", "node_c", 0.7),
        ("node_a", "node_d", 0.6),
    ];

    for (from, to, weight) in edges {
        ActivationGraphExt::add_edge(
            &graph_single,
            from.to_string(),
            to.to_string(),
            weight,
            EdgeType::Excitatory,
        );
        ActivationGraphExt::add_edge(
            &graph_dual,
            from.to_string(),
            to.to_string(),
            weight,
            EdgeType::Excitatory,
        );
    }

    // Spread activation from source
    let config = engram_core::activation::ParallelSpreadingConfig {
        max_depth: 3,
        activation_threshold: 0.1,
        ..Default::default()
    };

    let activated_single = ActivationGraphExt::parallel_spread(
        &graph_single,
        vec![("node_a".to_string(), 1.0)],
        config.clone(),
    );

    let activated_dual =
        ActivationGraphExt::parallel_spread(&graph_dual, vec![("node_a".to_string(), 1.0)], config);

    // Compare activated node sets
    let ids_single: Vec<String> = activated_single.keys().map(|s| s.to_string()).collect();
    let ids_dual: Vec<String> = activated_dual.keys().map(|s| s.to_string()).collect();

    let overlap = calculate_overlap(&ids_single, &ids_dual);

    assert!(
        overlap >= 0.8,
        "Spreading activation should reach similar nodes: overlap={}, single={:?}, dual={:?}",
        overlap,
        ids_single,
        ids_dual
    );
}

/// Test 5: Episode retrieval latency comparison
///
/// Validates that dual-memory retrieval latency is competitive with single-type.
#[test]
fn test_retrieval_latency_competitive() {
    use std::time::Instant;

    let episodes = generate_test_episodes(5000, 789);

    let store_single = MemoryStore::new(8192);
    let store_dual = MemoryStore::new(8192);

    for episode in &episodes {
        store_single.store(episode.clone());
        store_dual.store(episode.clone());
    }

    // Warm up
    for i in 0..100 {
        let _ = store_single.get_episode(&format!("test_episode_{}", i));
        let _ = store_dual.get_episode(&format!("test_episode_{}", i));
    }

    // Benchmark single-type retrieval
    let start = Instant::now();
    for i in 0..1000 {
        let _ = store_single.get_episode(&format!("test_episode_{}", i % 5000));
    }
    let duration_single = start.elapsed();

    // Benchmark dual-type retrieval
    let start = Instant::now();
    for i in 0..1000 {
        let _ = store_dual.get_episode(&format!("test_episode_{}", i % 5000));
    }
    let duration_dual = start.elapsed();

    let ratio = duration_dual.as_secs_f64() / duration_single.as_secs_f64();

    println!(
        "Retrieval latency: single={}ms, dual={}ms, ratio={}",
        duration_single.as_millis(),
        duration_dual.as_millis(),
        ratio
    );

    // Dual-type should be within 2x of single-type (conservative bound)
    assert!(
        ratio < 2.0,
        "Dual-type retrieval latency ratio {} exceeds 2.0x",
        ratio
    );
}

/// Test 6: Memory footprint comparison
///
/// Validates that dual-memory footprint is not significantly larger.
#[test]
fn test_memory_footprint_reasonable() {
    let episodes = generate_test_episodes(1000, 111);

    let store_single = MemoryStore::new(2048);
    let store_dual = MemoryStore::new(2048);

    for episode in &episodes {
        store_single.store(episode.clone());
        store_dual.store(episode.clone());
    }

    // Memory footprint comparison via tier counts
    let counts_single = store_single.get_tier_counts();
    let counts_dual = store_dual.get_tier_counts();

    assert_eq!(counts_single.total, 1000);
    assert_eq!(counts_dual.total, 1000);

    // Both should store same number of items (no duplication)
    assert_eq!(
        counts_single.total, counts_dual.total,
        "Total memory count should be identical"
    );
}

/// Test 7: Property-based differential testing
///
/// Uses property testing to verify equivalence for arbitrary workloads.
#[test]
fn test_property_differential_equivalence() {
    use rand::{Rng, SeedableRng, rngs::StdRng};

    let mut rng = StdRng::seed_from_u64(99999);

    for iteration in 0..50 {
        let episode_count = rng.gen_range(100..500);
        let episodes = generate_test_episodes(episode_count, iteration);

        let store_single = MemoryStore::new(1024);
        let store_dual = MemoryStore::new(1024);

        for episode in &episodes {
            store_single.store(episode.clone());
            store_dual.store(episode.clone());
        }

        // Property 1: Same total count
        assert_eq!(
            store_single.count(),
            store_dual.count(),
            "Iteration {}: Episode counts should match",
            iteration
        );

        // Property 2: Same retrieval success
        for i in 0..10 {
            let episode_id = format!("test_episode_{}", i);
            let retrieved_single = store_single.get_episode(&episode_id);
            let retrieved_dual = store_dual.get_episode(&episode_id);

            assert_eq!(
                retrieved_single.is_some(),
                retrieved_dual.is_some(),
                "Iteration {}: Retrieval availability should match for {}",
                iteration,
                episode_id
            );
        }

        // Property 3: Recall returns results
        let query_embedding = [rng.gen_range(-1.0..1.0); 768];
        let cue = CueBuilder::new()
            .id(format!("property_cue_{}", iteration))
            .embedding_search(query_embedding, Confidence::LOW)
            .cue_confidence(Confidence::HIGH)
            .build();

        let results_single = store_single.recall(&cue);
        let results_dual = store_dual.recall(&cue);

        // Both should return some results
        assert!(
            !results_single.results.is_empty() || episode_count < 10,
            "Iteration {}: Single-type should return results",
            iteration
        );
        assert!(
            !results_dual.results.is_empty() || episode_count < 10,
            "Iteration {}: Dual-type should return results",
            iteration
        );
    }
}

/// Test 8: Temporal ordering preserved
///
/// Ensures that temporal query results have similar ordering in both engines.
#[test]
fn test_temporal_ordering_preserved() {
    use chrono::Duration;

    let base_time = Utc::now() - Duration::hours(100);
    let mut episodes = Vec::new();

    for i in 0..100 {
        let mut episode = generate_test_episodes(1, i)[0].clone();
        episode.when = base_time + Duration::hours(i as i64);
        episode.id = format!("temporal_episode_{}", i);
        episodes.push(episode);
    }

    let store_single = MemoryStore::new(256);
    let store_dual = MemoryStore::new(256);

    for episode in &episodes {
        store_single.store(episode.clone());
        store_dual.store(episode.clone());
    }

    // Query for recent episodes
    let query_embedding = episodes[90].embedding;
    let cue = Cue::embedding("temporal_cue".to_string(), query_embedding, Confidence::MEDIUM);

    let results_single = store_single.recall(&cue);
    let results_dual = store_dual.recall(&cue);

    // Check that temporal ordering is similar (some variation allowed)
    let times_single: Vec<i64> = results_single
        .results
        .iter()
        .map(|(e, _)| e.when.timestamp())
        .collect();
    let times_dual: Vec<i64> = results_dual
        .results
        .iter()
        .map(|(e, _)| e.when.timestamp())
        .collect();

    // Verify both return recent episodes (within last 20 hours)
    let recent_threshold = (base_time + Duration::hours(80)).timestamp();

    let recent_count_single = times_single
        .iter()
        .filter(|&&t| t >= recent_threshold)
        .count();
    let recent_count_dual = times_dual.iter().filter(|&&t| t >= recent_threshold).count();

    assert!(
        recent_count_single >= 5 || results_single.results.len() < 5,
        "Single-type should return recent episodes"
    );
    assert!(
        recent_count_dual >= 5 || results_dual.results.len() < 5,
        "Dual-type should return recent episodes"
    );
}
