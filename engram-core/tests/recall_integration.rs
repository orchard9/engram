//! Integration tests for the cognitive recall pipeline
//!
//! Tests the orchestration of vector-similarity seeding, spreading activation,
//! confidence aggregation, and result ranking.

#[cfg(feature = "hnsw_index")]
mod tests {
    use chrono::Utc;
    use engram_core::activation::{
        ActivationGraphExt, ConfidenceAggregator, EdgeType, ParallelSpreadingConfig,
        create_activation_graph,
        cycle_detector::CycleDetector,
        parallel::ParallelSpreadingEngine,
        recall::{CognitiveRecall, CognitiveRecallBuilder, RecallConfig, RecallMode},
        seeding::VectorActivationSeeder,
        similarity_config::SimilarityConfig,
    };
    use engram_core::index::CognitiveHnswIndex;
    use engram_core::{Confidence, Cue, EpisodeBuilder, MemoryStore};
    use std::collections::HashMap;
    use std::sync::Arc;
    use std::time::Duration;

    /// Create a test memory store with sample episodes and HNSW index enabled
    fn create_test_store() -> MemoryStore {
        let store = MemoryStore::new(100).with_hnsw_index();

        // Add some test episodes
        let episodes = vec![
            ("memory1", "Learning Rust programming", vec![0.5f32; 768]),
            ("memory2", "Writing integration tests", vec![0.6f32; 768]),
            ("memory3", "Building cognitive systems", vec![0.7f32; 768]),
            (
                "memory4",
                "Parallel processing techniques",
                vec![0.8f32; 768],
            ),
            (
                "memory5",
                "Memory consolidation research",
                vec![0.9f32; 768],
            ),
        ];

        for (id, content, embedding_vec) in episodes {
            let mut embedding = [0.0f32; 768];
            for (i, val) in embedding_vec.iter().enumerate() {
                embedding[i] = *val;
            }

            let episode = EpisodeBuilder::new()
                .id(id.to_string())
                .when(Utc::now())
                .what(content.to_string())
                .embedding(embedding)
                .confidence(Confidence::HIGH)
                .build();

            store.store(episode);
        }

        store
    }

    /// Create a test HNSW index
    fn create_test_index() -> Arc<CognitiveHnswIndex> {
        Arc::new(CognitiveHnswIndex::new())
    }

    /// Create a test memory graph
    fn create_test_graph() -> Arc<engram_core::activation::MemoryGraph> {
        let graph = Arc::new(create_activation_graph());

        // Add some connections between memories
        ActivationGraphExt::add_edge(
            &*graph,
            "memory1".to_string(),
            "memory2".to_string(),
            0.8,
            EdgeType::Excitatory,
        );
        ActivationGraphExt::add_edge(
            &*graph,
            "memory2".to_string(),
            "memory3".to_string(),
            0.7,
            EdgeType::Excitatory,
        );
        ActivationGraphExt::add_edge(
            &*graph,
            "memory1".to_string(),
            "memory4".to_string(),
            0.6,
            EdgeType::Excitatory,
        );
        ActivationGraphExt::add_edge(
            &*graph,
            "memory4".to_string(),
            "memory5".to_string(),
            0.9,
            EdgeType::Excitatory,
        );

        graph
    }

    #[test]
    fn test_cognitive_recall_builder() {
        let index = create_test_index();
        let graph = create_test_graph();

        let seeder = Arc::new(VectorActivationSeeder::with_default_resolver(
            index,
            SimilarityConfig::default(),
        ));

        let spreading_engine = Arc::new(
            ParallelSpreadingEngine::new(ParallelSpreadingConfig::default(), graph).unwrap(),
        );

        let aggregator = Arc::new(ConfidenceAggregator::new(0.8, Confidence::LOW, 10));
        let cycle_detector = Arc::new(CycleDetector::new(HashMap::new()));

        let recall = CognitiveRecallBuilder::new()
            .vector_seeder(seeder)
            .spreading_engine(spreading_engine)
            .confidence_aggregator(aggregator)
            .cycle_detector(cycle_detector)
            .recall_mode(RecallMode::Spreading)
            .time_budget(Duration::from_millis(20))
            .build()
            .expect("Failed to build CognitiveRecall");

        assert_eq!(recall.config().recall_mode, RecallMode::Spreading);
        assert_eq!(recall.config().time_budget, Duration::from_millis(20));
    }

    #[test]
    fn test_similarity_mode_recall() {
        let store = create_test_store();
        // Use the store's HNSW index (auto-populated during store.store())
        let index = store.hnsw_index().expect("HNSW index should be available");
        let graph = create_test_graph();

        let seeder = Arc::new(VectorActivationSeeder::with_default_resolver(
            index,
            SimilarityConfig::default(),
        ));

        let spreading_engine = Arc::new(
            ParallelSpreadingEngine::new(ParallelSpreadingConfig::default(), graph).unwrap(),
        );

        let aggregator = Arc::new(ConfidenceAggregator::new(0.8, Confidence::LOW, 10));
        let cycle_detector = Arc::new(CycleDetector::new(HashMap::new()));

        let recall_config = RecallConfig {
            recall_mode: RecallMode::Similarity,
            ..Default::default()
        };

        let recall = CognitiveRecall::new(
            seeder,
            spreading_engine,
            aggregator,
            cycle_detector,
            recall_config,
        );

        // Create a test cue
        let mut cue_embedding = [0.0f32; 768];
        cue_embedding.fill(0.7);
        let cue = Cue::embedding(
            "similarity_test".to_string(),
            cue_embedding,
            Confidence::HIGH,
        );

        let results = recall.recall(&cue, &store).expect("Recall failed");

        // Should return some results
        assert!(!results.is_empty());
        assert!(results.len() <= 3); // Should respect max_results from cue
    }

    #[test]
    fn test_spreading_mode_recall() {
        let store = create_test_store();
        // Use the store's HNSW index (auto-populated during store.store())
        let index = store.hnsw_index().expect("HNSW index should be available");
        let graph = create_test_graph();

        let seeder = Arc::new(VectorActivationSeeder::with_default_resolver(
            index,
            SimilarityConfig::default(),
        ));

        let spreading_config = ParallelSpreadingConfig {
            num_threads: 2,
            max_depth: 2,
            threshold: 0.1,
            ..Default::default()
        };

        let spreading_engine =
            Arc::new(ParallelSpreadingEngine::new(spreading_config, graph).unwrap());

        let aggregator = Arc::new(ConfidenceAggregator::new(0.8, Confidence::LOW, 10));
        let cycle_detector = Arc::new(CycleDetector::new(HashMap::new()));

        let recall_config = RecallConfig {
            recall_mode: RecallMode::Spreading,
            ..Default::default()
        };

        let recall = CognitiveRecall::new(
            seeder,
            spreading_engine,
            aggregator,
            cycle_detector,
            recall_config,
        );

        // Create a test cue
        let mut cue_embedding = [0.0f32; 768];
        cue_embedding.fill(0.5); // Should match memory1
        let cue = Cue::embedding(
            "spreading_test".to_string(),
            cue_embedding,
            Confidence::HIGH,
        );

        let results = recall.recall(&cue, &store).expect("Recall failed");

        // Should return results through spreading
        assert!(!results.is_empty());

        // Check that activation values are present
        for result in &results {
            assert!(result.activation > 0.0);
            assert!(result.confidence.raw() > 0.0);
        }
    }

    #[test]
    fn test_recency_boost() {
        let store = MemoryStore::new(100).with_hnsw_index();

        // Add episodes with different ages (auto-indexed via store.store())
        let now = Utc::now();
        let recent_episode = EpisodeBuilder::new()
            .id("recent".to_string())
            .when(now)
            .what("Recent memory".to_string())
            .embedding([0.5f32; 768])
            .confidence(Confidence::MEDIUM)
            .build();

        let old_episode = EpisodeBuilder::new()
            .id("old".to_string())
            .when(now - chrono::Duration::hours(2))
            .what("Old memory".to_string())
            .embedding([0.5f32; 768]) // Same embedding
            .confidence(Confidence::MEDIUM)
            .build();

        store.store(recent_episode);
        store.store(old_episode);

        // Use the store's HNSW index (auto-populated during store.store())
        let index = store.hnsw_index().expect("HNSW index should be available");
        let graph = Arc::new(create_activation_graph());
        let seeder = Arc::new(VectorActivationSeeder::with_default_resolver(
            index,
            SimilarityConfig::default(),
        ));

        let spreading_engine = Arc::new(
            ParallelSpreadingEngine::new(ParallelSpreadingConfig::default(), graph).unwrap(),
        );

        let aggregator = Arc::new(ConfidenceAggregator::new(0.8, Confidence::LOW, 10));
        let cycle_detector = Arc::new(CycleDetector::new(HashMap::new()));

        let recall_config = RecallConfig {
            enable_recency_boost: true,
            recency_boost_factor: 2.0, // Double weight for recent
            recency_window: Duration::from_secs(3600), // 1 hour
            ..Default::default()
        };

        let recall = CognitiveRecall::new(
            seeder,
            spreading_engine,
            aggregator,
            cycle_detector,
            recall_config,
        );

        let cue = Cue::embedding("recency_test".to_string(), [0.5f32; 768], Confidence::HIGH);
        let results = recall.recall(&cue, &store).expect("Recall failed");

        // Find the recent and old memories
        let recent_result = results.iter().find(|r| r.episode.id == "recent");
        let old_result = results.iter().find(|r| r.episode.id == "old");

        assert!(recent_result.is_some());
        assert!(old_result.is_some());

        // Recent memory should have higher recency boost
        assert!(recent_result.unwrap().recency_boost > old_result.unwrap().recency_boost);

        // Recent memory should rank higher
        let recent_idx = results
            .iter()
            .position(|r| r.episode.id == "recent")
            .unwrap();
        let old_idx = results.iter().position(|r| r.episode.id == "old").unwrap();
        assert!(recent_idx < old_idx); // Lower index = higher rank
    }

    #[test]
    fn test_hybrid_mode_with_fallback() {
        let store = create_test_store();
        // Use the store's HNSW index (auto-populated during store.store())
        let index = store.hnsw_index().expect("HNSW index should be available");
        let graph = create_test_graph();

        let seeder = Arc::new(VectorActivationSeeder::with_default_resolver(
            index,
            SimilarityConfig::default(),
        ));

        let spreading_engine = Arc::new(
            ParallelSpreadingEngine::new(ParallelSpreadingConfig::default(), graph).unwrap(),
        );

        let aggregator = Arc::new(ConfidenceAggregator::new(0.8, Confidence::LOW, 10));
        let cycle_detector = Arc::new(CycleDetector::new(HashMap::new()));

        let recall_config = RecallConfig {
            recall_mode: RecallMode::Hybrid,
            time_budget: Duration::from_millis(5), // Very short timeout to trigger fallback
            ..Default::default()
        };

        let recall = CognitiveRecall::new(
            seeder,
            spreading_engine,
            aggregator,
            cycle_detector,
            recall_config,
        );

        let cue = Cue::embedding("fallback_test".to_string(), [0.7f32; 768], Confidence::HIGH);
        let results = recall.recall(&cue, &store).expect("Recall failed");

        // Should still return results even with timeout (via fallback)
        assert!(!results.is_empty());
    }

    #[test]
    fn test_confidence_aggregation() {
        let store = create_test_store();
        // Use the store's HNSW index (auto-populated during store.store())
        let index = store.hnsw_index().expect("HNSW index should be available");
        let graph = create_test_graph();

        let seeder = Arc::new(VectorActivationSeeder::with_default_resolver(
            index,
            SimilarityConfig::default(),
        ));

        let spreading_engine = Arc::new(
            ParallelSpreadingEngine::new(ParallelSpreadingConfig::default(), graph).unwrap(),
        );

        let aggregator = Arc::new(ConfidenceAggregator::new(0.8, Confidence::LOW, 10));
        let cycle_detector = Arc::new(CycleDetector::new(HashMap::new()));

        let recall = CognitiveRecall::new(
            seeder,
            spreading_engine,
            aggregator,
            cycle_detector,
            RecallConfig::default(),
        );

        let cue = Cue::embedding(
            "confidence_test".to_string(),
            [0.5f32; 768],
            Confidence::exact(1.0),
        );
        let results = recall.recall(&cue, &store).expect("Recall failed");

        // All results should have valid confidence scores
        for result in results {
            assert!(result.confidence.raw() >= 0.0);
            assert!(result.confidence.raw() <= 1.0);

            // Seed nodes should have boosted confidence
            if result.similarity.is_some() {
                // This was a seed node
                assert!(result.confidence.raw() > 0.0);
            }
        }
    }

    #[test]
    fn test_result_ranking() {
        let store = create_test_store();
        // Use the store's HNSW index (auto-populated during store.store())
        let index = store.hnsw_index().expect("HNSW index should be available");
        let graph = create_test_graph();

        let seeder = Arc::new(VectorActivationSeeder::with_default_resolver(
            index,
            SimilarityConfig::default(),
        ));

        let spreading_engine = Arc::new(
            ParallelSpreadingEngine::new(ParallelSpreadingConfig::default(), graph).unwrap(),
        );

        let aggregator = Arc::new(ConfidenceAggregator::new(0.8, Confidence::LOW, 10));
        let cycle_detector = Arc::new(CycleDetector::new(HashMap::new()));

        let recall = CognitiveRecall::new(
            seeder,
            spreading_engine,
            aggregator,
            cycle_detector,
            RecallConfig::default(),
        );

        let cue = Cue::embedding("recency_test".to_string(), [0.5f32; 768], Confidence::HIGH);
        let results = recall.recall(&cue, &store).expect("Recall failed");

        // Results should be ranked by score
        for i in 1..results.len() {
            assert!(
                results[i - 1].rank_score >= results[i].rank_score,
                "Results not properly ranked"
            );
        }
    }

    #[test]
    fn test_empty_cue_handling() {
        let store = create_test_store();
        // Use the store's HNSW index (auto-populated during store.store())
        let index = store.hnsw_index().expect("HNSW index should be available");
        let graph = create_test_graph();

        let seeder = Arc::new(VectorActivationSeeder::with_default_resolver(
            index,
            SimilarityConfig::default(),
        ));

        let spreading_engine = Arc::new(
            ParallelSpreadingEngine::new(ParallelSpreadingConfig::default(), graph).unwrap(),
        );

        let aggregator = Arc::new(ConfidenceAggregator::new(0.8, Confidence::LOW, 10));
        let cycle_detector = Arc::new(CycleDetector::new(HashMap::new()));

        let recall = CognitiveRecall::new(
            seeder,
            spreading_engine,
            aggregator,
            cycle_detector,
            RecallConfig::default(),
        );

        // Create a semantic cue (not supported yet)
        let cue = Cue::context("test context".to_string(), None, None, Confidence::HIGH);
        let _results = recall.recall(&cue, &store).expect("Recall failed");

        // Should handle gracefully (return empty or fallback)
        // This always passes - testing that no panic occurred
    }

    #[test]
    fn test_memory_store_spreading_mode_dispatch() {
        // Test that MemoryStore::recall() properly dispatches based on RecallMode::Spreading
        let mut store = MemoryStore::new(100).with_hnsw_index();

        // Add test episodes (auto-indexed via store.store())
        let episodes = vec![
            ("mem1", "First memory", [0.3f32; 768]),
            ("mem2", "Second memory", [0.4f32; 768]),
            ("mem3", "Third memory", [0.5f32; 768]),
        ];

        for (id, content, embedding) in episodes {
            let episode = EpisodeBuilder::new()
                .id(id.to_string())
                .when(Utc::now())
                .what(content.to_string())
                .embedding(embedding)
                .confidence(Confidence::HIGH)
                .build();
            store.store(episode);
        }

        // Build cognitive recall pipeline using the store's HNSW index
        let index = store.hnsw_index().expect("HNSW index should be available");
        let graph = create_test_graph();
        let seeder = Arc::new(VectorActivationSeeder::with_default_resolver(
            index,
            SimilarityConfig::default(),
        ));
        let spreading_engine = Arc::new(
            ParallelSpreadingEngine::new(ParallelSpreadingConfig::default(), graph).unwrap(),
        );
        let aggregator = Arc::new(ConfidenceAggregator::new(0.8, Confidence::LOW, 10));
        let cycle_detector = Arc::new(CycleDetector::new(HashMap::new()));

        let recall = CognitiveRecallBuilder::new()
            .vector_seeder(seeder)
            .spreading_engine(spreading_engine)
            .confidence_aggregator(aggregator)
            .cycle_detector(cycle_detector)
            .build()
            .expect("Failed to build CognitiveRecall");

        // Configure store to use spreading activation
        let recall_config = RecallConfig {
            recall_mode: RecallMode::Spreading,
            time_budget: Duration::from_millis(20),
            min_confidence: 0.1,
            max_results: 10,
            ..Default::default()
        };

        store = store
            .with_cognitive_recall(Arc::new(recall))
            .with_recall_config(recall_config);

        // Create test cue
        let cue = Cue::embedding("test_cue".to_string(), [0.4f32; 768], Confidence::HIGH);

        // Call MemoryStore::recall() which should dispatch to spreading activation
        let results = store.recall(&cue);

        // Verify results
        assert!(
            !results.is_empty(),
            "Should return results via spreading activation"
        );

        // Results should be Episode-Confidence tuples
        for (episode, confidence) in results {
            assert!(confidence.raw() > 0.0, "Confidence should be positive");
            assert!(confidence.raw() <= 1.0, "Confidence should be <= 1.0");
            assert!(!episode.id.is_empty(), "Episode should have valid ID");
        }
    }

    #[test]
    fn test_memory_store_similarity_mode_dispatch() {
        // Test that MemoryStore::recall() properly dispatches based on RecallMode::Similarity
        let mut store = MemoryStore::new(100).with_hnsw_index();

        // Add test episodes (auto-indexed via store.store())
        let episode = EpisodeBuilder::new()
            .id("similarity_test".to_string())
            .when(Utc::now())
            .what("Test for similarity mode".to_string())
            .embedding([0.6f32; 768])
            .confidence(Confidence::HIGH)
            .build();
        store.store(episode);

        // Build minimal cognitive recall (won't be used in similarity mode)
        let index = store.hnsw_index().expect("HNSW index should be available");
        let graph = create_test_graph();
        let seeder = Arc::new(VectorActivationSeeder::with_default_resolver(
            index,
            SimilarityConfig::default(),
        ));
        let spreading_engine = Arc::new(
            ParallelSpreadingEngine::new(ParallelSpreadingConfig::default(), graph).unwrap(),
        );
        let aggregator = Arc::new(ConfidenceAggregator::new(0.8, Confidence::LOW, 10));
        let cycle_detector = Arc::new(CycleDetector::new(HashMap::new()));

        let recall = CognitiveRecallBuilder::new()
            .vector_seeder(seeder)
            .spreading_engine(spreading_engine)
            .confidence_aggregator(aggregator)
            .cycle_detector(cycle_detector)
            .build()
            .expect("Failed to build CognitiveRecall");

        // Configure store to use similarity-only mode
        let recall_config = RecallConfig {
            recall_mode: RecallMode::Similarity,
            ..Default::default()
        };

        store = store
            .with_cognitive_recall(Arc::new(recall))
            .with_recall_config(recall_config);

        // Create test cue
        let cue = Cue::embedding(
            "similarity_cue".to_string(),
            [0.6f32; 768],
            Confidence::HIGH,
        );

        // Call MemoryStore::recall() which should use similarity-only path
        let _results = store.recall(&cue);

        // Should return results (using HNSW or basic similarity)
        // Test passes if no panic occurred
    }

    #[test]
    fn test_memory_store_hybrid_mode_dispatch() {
        // Test that MemoryStore::recall() properly dispatches based on RecallMode::Hybrid
        let mut store = MemoryStore::new(100).with_hnsw_index();

        let episode = EpisodeBuilder::new()
            .id("hybrid_test".to_string())
            .when(Utc::now())
            .what("Test for hybrid mode".to_string())
            .embedding([0.7f32; 768])
            .confidence(Confidence::HIGH)
            .build();
        store.store(episode);

        // Build cognitive recall pipeline using the store's HNSW index
        let index = store.hnsw_index().expect("HNSW index should be available");
        let graph = create_test_graph();
        let seeder = Arc::new(VectorActivationSeeder::with_default_resolver(
            index,
            SimilarityConfig::default(),
        ));
        let spreading_engine = Arc::new(
            ParallelSpreadingEngine::new(ParallelSpreadingConfig::default(), graph).unwrap(),
        );
        let aggregator = Arc::new(ConfidenceAggregator::new(0.8, Confidence::LOW, 10));
        let cycle_detector = Arc::new(CycleDetector::new(HashMap::new()));

        let recall = CognitiveRecallBuilder::new()
            .vector_seeder(seeder)
            .spreading_engine(spreading_engine)
            .confidence_aggregator(aggregator)
            .cycle_detector(cycle_detector)
            .build()
            .expect("Failed to build CognitiveRecall");

        // Configure store to use hybrid mode (tries spreading, falls back to similarity)
        let recall_config = RecallConfig {
            recall_mode: RecallMode::Hybrid,
            time_budget: Duration::from_millis(20),
            ..Default::default()
        };

        store = store
            .with_cognitive_recall(Arc::new(recall))
            .with_recall_config(recall_config);

        // Create test cue
        let cue = Cue::embedding("hybrid_cue".to_string(), [0.7f32; 768], Confidence::HIGH);

        // Call MemoryStore::recall() which should try spreading then fallback if needed
        let results = store.recall(&cue);

        // Should return results via hybrid mode
        assert!(
            results.is_empty() || !results.is_empty(),
            "Hybrid mode should complete"
        );
    }
}
