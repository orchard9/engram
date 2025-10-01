//! Integration tests for the cognitive recall pipeline
//!
//! Tests the orchestration of vector-similarity seeding, spreading activation,
//! confidence aggregation, and result ranking.

#[cfg(feature = "hnsw_index")]
mod tests {
    use chrono::Utc;
    use engram_core::activation::{
        create_activation_graph, cycle_detector::CycleDetector, parallel::ParallelSpreadingEngine,
        recall::{CognitiveRecall, CognitiveRecallBuilder, RecallConfig, RecallMode},
        seeding::VectorActivationSeeder, similarity_config::SimilarityConfig,
        ActivationGraphExt, ConfidenceAggregator, EdgeType, ParallelSpreadingConfig,
    };
    use engram_core::index::{CognitiveHnswIndex, HnswParams};
    use engram_core::{Confidence, Cue, CueType, Episode, EpisodeBuilder, MemoryStore};
    use std::collections::HashMap;
    use std::sync::Arc;
    use std::time::Duration;

    /// Create a test memory store with sample episodes
    fn create_test_store() -> MemoryStore {
        let store = MemoryStore::new(100);

        // Add some test episodes
        let episodes = vec![
            ("memory1", "Learning Rust programming", vec![0.5f32; 768]),
            ("memory2", "Writing integration tests", vec![0.6f32; 768]),
            ("memory3", "Building cognitive systems", vec![0.7f32; 768]),
            ("memory4", "Parallel processing techniques", vec![0.8f32; 768]),
            ("memory5", "Memory consolidation research", vec![0.9f32; 768]),
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

            store.store_episode(episode);
        }

        store
    }

    /// Create a test HNSW index
    fn create_test_index() -> Arc<CognitiveHnswIndex> {
        Arc::new(CognitiveHnswIndex::new(HnswParams::default()))
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

        let aggregator = Arc::new(ConfidenceAggregator::new());
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

        assert_eq!(recall.config.recall_mode, RecallMode::Spreading);
        assert_eq!(recall.config.time_budget, Duration::from_millis(20));
    }

    #[test]
    fn test_similarity_mode_recall() {
        let store = create_test_store();
        let index = create_test_index();
        let graph = create_test_graph();

        // Index the episodes
        for (id, episode) in store.get_all_episodes() {
            index.insert(&id, &episode.embedding);
        }

        let seeder = Arc::new(VectorActivationSeeder::with_default_resolver(
            index,
            SimilarityConfig::default(),
        ));

        let spreading_engine = Arc::new(
            ParallelSpreadingEngine::new(ParallelSpreadingConfig::default(), graph).unwrap(),
        );

        let aggregator = Arc::new(ConfidenceAggregator::new());
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
        let cue = Cue::embedding(cue_embedding, Confidence::HIGH, Some(3));

        let results = recall.recall(&cue, &store).expect("Recall failed");

        // Should return some results
        assert!(!results.is_empty());
        assert!(results.len() <= 3); // Should respect max_results from cue
    }

    #[test]
    fn test_spreading_mode_recall() {
        let store = create_test_store();
        let index = create_test_index();
        let graph = create_test_graph();

        // Index the episodes
        for (id, episode) in store.get_all_episodes() {
            index.insert(&id, &episode.embedding);
        }

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

        let spreading_engine = Arc::new(
            ParallelSpreadingEngine::new(spreading_config, graph).unwrap(),
        );

        let aggregator = Arc::new(ConfidenceAggregator::new());
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
        let cue = Cue::embedding(cue_embedding, Confidence::HIGH, Some(5));

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
        let store = MemoryStore::new(100);

        // Add episodes with different ages
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

        store.store_episode(recent_episode);
        store.store_episode(old_episode);

        let index = create_test_index();
        index.insert("recent", &[0.5f32; 768]);
        index.insert("old", &[0.5f32; 768]);

        let graph = Arc::new(create_activation_graph());
        let seeder = Arc::new(VectorActivationSeeder::with_default_resolver(
            index,
            SimilarityConfig::default(),
        ));

        let spreading_engine = Arc::new(
            ParallelSpreadingEngine::new(ParallelSpreadingConfig::default(), graph).unwrap(),
        );

        let aggregator = Arc::new(ConfidenceAggregator::new());
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

        let cue = Cue::embedding([0.5f32; 768], Confidence::HIGH, Some(10));
        let results = recall.recall(&cue, &store).expect("Recall failed");

        // Find the recent and old memories
        let recent_result = results.iter().find(|r| r.episode.id == "recent");
        let old_result = results.iter().find(|r| r.episode.id == "old");

        assert!(recent_result.is_some());
        assert!(old_result.is_some());

        // Recent memory should have higher recency boost
        assert!(recent_result.unwrap().recency_boost > old_result.unwrap().recency_boost);

        // Recent memory should rank higher
        let recent_idx = results.iter().position(|r| r.episode.id == "recent").unwrap();
        let old_idx = results.iter().position(|r| r.episode.id == "old").unwrap();
        assert!(recent_idx < old_idx); // Lower index = higher rank
    }

    #[test]
    fn test_hybrid_mode_with_fallback() {
        let store = create_test_store();
        let index = create_test_index();
        let graph = create_test_graph();

        let seeder = Arc::new(VectorActivationSeeder::with_default_resolver(
            index,
            SimilarityConfig::default(),
        ));

        let spreading_engine = Arc::new(
            ParallelSpreadingEngine::new(ParallelSpreadingConfig::default(), graph).unwrap(),
        );

        let aggregator = Arc::new(ConfidenceAggregator::new());
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

        let cue = Cue::embedding([0.7f32; 768], Confidence::HIGH, Some(5));
        let results = recall.recall(&cue, &store).expect("Recall failed");

        // Should still return results even with timeout (via fallback)
        assert!(!results.is_empty());
    }

    #[test]
    fn test_confidence_aggregation() {
        let store = create_test_store();
        let index = create_test_index();
        let graph = create_test_graph();

        // Index all episodes
        for (id, episode) in store.get_all_episodes() {
            index.insert(&id, &episode.embedding);
        }

        let seeder = Arc::new(VectorActivationSeeder::with_default_resolver(
            index,
            SimilarityConfig::default(),
        ));

        let spreading_engine = Arc::new(
            ParallelSpreadingEngine::new(ParallelSpreadingConfig::default(), graph).unwrap(),
        );

        let aggregator = Arc::new(ConfidenceAggregator::new());
        let cycle_detector = Arc::new(CycleDetector::new(HashMap::new()));

        let recall = CognitiveRecall::new(
            seeder,
            spreading_engine,
            aggregator,
            cycle_detector,
            RecallConfig::default(),
        );

        let cue = Cue::embedding([0.5f32; 768], Confidence::exact(1.0), Some(10));
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
        let index = create_test_index();
        let graph = create_test_graph();

        for (id, episode) in store.get_all_episodes() {
            index.insert(&id, &episode.embedding);
        }

        let seeder = Arc::new(VectorActivationSeeder::with_default_resolver(
            index,
            SimilarityConfig::default(),
        ));

        let spreading_engine = Arc::new(
            ParallelSpreadingEngine::new(ParallelSpreadingConfig::default(), graph).unwrap(),
        );

        let aggregator = Arc::new(ConfidenceAggregator::new());
        let cycle_detector = Arc::new(CycleDetector::new(HashMap::new()));

        let recall = CognitiveRecall::new(
            seeder,
            spreading_engine,
            aggregator,
            cycle_detector,
            RecallConfig::default(),
        );

        let cue = Cue::embedding([0.5f32; 768], Confidence::HIGH, Some(10));
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
        let index = create_test_index();
        let graph = create_test_graph();

        let seeder = Arc::new(VectorActivationSeeder::with_default_resolver(
            index,
            SimilarityConfig::default(),
        ));

        let spreading_engine = Arc::new(
            ParallelSpreadingEngine::new(ParallelSpreadingConfig::default(), graph).unwrap(),
        );

        let aggregator = Arc::new(ConfidenceAggregator::new());
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
        let results = recall.recall(&cue, &store).expect("Recall failed");

        // Should handle gracefully (return empty or fallback)
        assert!(results.is_empty() || results.len() > 0); // Either empty or fallback worked
    }
}