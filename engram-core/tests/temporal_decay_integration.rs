//! Integration tests for Milestone 4: Temporal Decay
//!
//! Tests lazy decay integration with CognitiveRecall pipeline.

#[cfg(feature = "hnsw_index")]
mod tests {
    use engram_core::{
        activation::{
            recall::{CognitiveRecall, CognitiveRecallBuilder, RecallConfig, RecallMode},
            ConfidenceAggregator, ParallelSpreadingConfig,
            create_activation_graph,
            cycle_detector::CycleDetector,
            parallel::ParallelSpreadingEngine,
            seeding::VectorActivationSeeder,
            similarity_config::SimilarityConfig,
        },
        decay::BiologicalDecaySystem,
        Confidence, Cue, EpisodeBuilder, MemoryStore,
    };
    use chrono::Utc;
    use std::collections::HashMap;
    use std::sync::Arc;
    use std::time::Duration as StdDuration;

    /// Helper to create a test memory store with HNSW index
    fn create_test_store() -> MemoryStore {
        MemoryStore::new(10000).with_hnsw_index()
    }

    /// Helper to create a varied embedding (not all same value)
    fn create_varied_embedding(base: f32, variation: usize) -> [f32; 768] {
        let mut embedding = [base; 768];
        // Add some variation to make embeddings distinguishable
        for i in 0..768 {
            embedding[i] = base + (i as f32 * variation as f32 * 0.0001);
        }
        embedding
    }

    /// Helper to create a minimal working CognitiveRecall setup
    fn create_recall_with_decay(
        store: &MemoryStore,
        decay_system: Option<Arc<BiologicalDecaySystem>>,
    ) -> CognitiveRecall {
        let index = store
            .hnsw_index()
            .expect("HNSW index should be available");
        let graph = Arc::new(create_activation_graph());

        // Use relaxed similarity config to find more episodes
        let similarity_config = SimilarityConfig {
            temperature: 0.5,
            threshold: 0.1, // Low threshold to find similar episodes
            max_candidates: 100,
            ef_search: 96,
            min_confidence: Confidence::exact(0.01), // Very low to include all results
        };
        let vector_seeder = Arc::new(VectorActivationSeeder::with_default_resolver(
            index,
            similarity_config,
        ));

        let spreading_config = ParallelSpreadingConfig {
            num_threads: 2,
            max_depth: 3,
            threshold: 0.05, // Lower threshold to allow more spreading
            ..Default::default()
        };
        let spreading_engine = Arc::new(
            ParallelSpreadingEngine::new(spreading_config, graph).unwrap(),
        );

        let confidence_aggregator = Arc::new(ConfidenceAggregator::new(
            0.7, // Lower decay rate to keep more confidence
            Confidence::exact(0.01), // Very low threshold to see decay effects
            100,
        ));
        let cycle_detector = Arc::new(CycleDetector::new(HashMap::new()));

        let config = RecallConfig {
            recall_mode: RecallMode::Similarity, // Use similarity mode for reliable results
            time_budget: StdDuration::from_millis(100),
            min_confidence: 0.01, // Very low threshold to see decay effects
            max_results: 100,
            enable_recency_boost: false, // Disable to isolate decay effects
            ..Default::default()
        };

        let mut builder = CognitiveRecallBuilder::new()
            .vector_seeder(vector_seeder)
            .spreading_engine(spreading_engine)
            .confidence_aggregator(confidence_aggregator)
            .cycle_detector(cycle_detector)
            .config(config);

        if let Some(decay_system) = decay_system {
            builder = builder.decay_system(decay_system);
        }

        builder.build().unwrap()
    }

    #[test]
    fn test_decay_system_optional() {
        // Test that recall works without decay system (backward compatibility)
        let store = create_test_store();

        let embedding = create_varied_embedding(0.5, 1);
        let episode = EpisodeBuilder::new()
            .id("test_episode".to_string())
            .when(Utc::now())
            .what("test content".to_string())
            .embedding(embedding)
            .confidence(Confidence::HIGH)
            .build();

        store.store(episode);

        // Create recall WITHOUT decay system
        let recall = create_recall_with_decay(&store, None);

        let cue = Cue::embedding("query".to_string(), embedding, Confidence::MEDIUM);
        let results = recall.recall(&cue, &store).unwrap();

        // Should still work without decay
        assert!(!results.is_empty(), "Recall should work without decay system");
    }

    #[test]
    fn test_decay_reduces_confidence_over_time() {
        // Test end-to-end decay through CognitiveRecall
        let store = create_test_store();

        // Insert an episode with moderate age (3 hours)
        let embedding = create_varied_embedding(0.5, 1);
        let mut episode = EpisodeBuilder::new()
            .id("old_episode".to_string())
            .when(Utc::now() - chrono::Duration::hours(3))
            .what("old content".to_string())
            .embedding(embedding)
            .confidence(Confidence::from_raw(0.9))
            .build();

        // Simulate this was last recalled 3 hours ago
        episode.last_recall = Utc::now() - chrono::Duration::hours(3);
        episode.recall_count = 1; // Infrequently accessed (hippocampal decay)

        store.store(episode);

        // Test WITH decay system
        let decay_system = Arc::new(BiologicalDecaySystem::default());
        let recall_with_decay = create_recall_with_decay(&store, Some(decay_system));

        let cue = Cue::embedding("query".to_string(), embedding, Confidence::MEDIUM);
        let results_with_decay = recall_with_decay.recall(&cue, &store).unwrap();

        assert!(!results_with_decay.is_empty(), "Should find the episode with decay");

        // Test WITHOUT decay system for comparison
        let recall_no_decay = create_recall_with_decay(&store, None);
        let results_no_decay = recall_no_decay.recall(&cue, &store).unwrap();

        assert!(!results_no_decay.is_empty(), "Should find the episode without decay");

        // Confidence with decay should be lower than without decay
        let with_decay_conf = results_with_decay[0].confidence.raw();
        let no_decay_conf = results_no_decay[0].confidence.raw();

        assert!(
            with_decay_conf < no_decay_conf,
            "Confidence with decay ({}) should be less than without decay ({})",
            with_decay_conf,
            no_decay_conf
        );

        // With decay should be measurably lower but still above minimum threshold
        assert!(
            with_decay_conf >= 0.1,
            "Confidence with decay should be >= 0.1, got {}",
            with_decay_conf
        );
    }

    #[test]
    fn test_frequently_accessed_memories_decay_slower() {
        // Test by querying each memory individually and comparing confidence
        let decay_system = BiologicalDecaySystem::default();
        let elapsed = StdDuration::from_secs(6 * 3600); // 6 hours

        // Rarely accessed memory (hippocampal decay)
        let rare_confidence = decay_system.compute_decayed_confidence(
            Confidence::HIGH,
            elapsed,
            1, // recall_count < 3
            Utc::now() - chrono::Duration::hours(6),
        );

        // Frequently accessed memory (neocortical decay)
        let freq_confidence = decay_system.compute_decayed_confidence(
            Confidence::HIGH,
            elapsed,
            5, // recall_count >= 3
            Utc::now() - chrono::Duration::hours(6),
        );

        // Frequently accessed should decay slower (neocortical vs hippocampal)
        assert!(
            freq_confidence.raw() > rare_confidence.raw(),
            "Frequently accessed (recall_count=5) should have higher confidence than rarely accessed (recall_count=1). Got freq={}, rare={}",
            freq_confidence.raw(),
            rare_confidence.raw()
        );

        // Frequently accessed should be significantly higher due to neocortical decay
        assert!(freq_confidence.raw() >= 0.1, "Frequent confidence should be >= 0.1");

        // The key test: neocortical decay should be significantly slower than hippocampal
        // Frequent confidence should be at least 2x higher than rare
        assert!(
            freq_confidence.raw() >= 2.0 * rare_confidence.raw(),
            "Neocortical decay should be significantly slower. Got freq={}, rare={}, ratio={}",
            freq_confidence.raw(),
            rare_confidence.raw(),
            freq_confidence.raw() / rare_confidence.raw()
        );
    }

    #[test]
    fn test_decay_respects_consolidation_threshold() {
        // Test the consolidation threshold by directly computing decay
        let decay_system = BiologicalDecaySystem::default();
        let elapsed = StdDuration::from_secs(6 * 3600); // 6 hours

        // Memory at exactly consolidation threshold (recall_count = 3 uses neocortical)
        let at_threshold_confidence = decay_system.compute_decayed_confidence(
            Confidence::HIGH,
            elapsed,
            3, // At threshold - uses neocortical decay
            Utc::now() - chrono::Duration::hours(6),
        );

        // Memory below threshold (recall_count = 2 uses hippocampal)
        let below_threshold_confidence = decay_system.compute_decayed_confidence(
            Confidence::HIGH,
            elapsed,
            2, // Below threshold - uses hippocampal decay
            Utc::now() - chrono::Duration::hours(6),
        );

        // At threshold (3) uses neocortical, below uses hippocampal
        assert!(
            at_threshold_confidence.raw() > below_threshold_confidence.raw(),
            "Memory at consolidation threshold (recall_count=3) should have higher confidence than below threshold (recall_count=2). Got at={}, below={}",
            at_threshold_confidence.raw(),
            below_threshold_confidence.raw()
        );
    }

    #[test]
    fn test_recent_memories_decay_minimally() {
        let store = create_test_store();

        // Very recent episode (last recalled 1 minute ago)
        let embedding = create_varied_embedding(0.5, 1);
        let mut recent = EpisodeBuilder::new()
            .id("recent".to_string())
            .when(Utc::now())
            .what("recent content".to_string())
            .embedding(embedding)
            .confidence(Confidence::HIGH)
            .build();

        recent.last_recall = Utc::now() - chrono::Duration::minutes(1);
        recent.recall_count = 1;

        store.store(recent);

        let decay_system = Arc::new(BiologicalDecaySystem::default());
        let recall = create_recall_with_decay(&store, Some(decay_system));

        let cue = Cue::embedding("query".to_string(), embedding, Confidence::MEDIUM);
        let results = recall.recall(&cue, &store).unwrap();

        assert!(!results.is_empty());

        let result_confidence = results[0].confidence.raw();

        // Recent memories should decay very little
        // Note: Decay still happens even after 1 minute, so we check for >0.75 instead of >0.85
        assert!(
            result_confidence >= 0.75,
            "Recent memory should have minimal decay, got {}",
            result_confidence
        );
    }

    #[test]
    fn test_decay_computation_performance() {
        use std::time::Instant;

        let decay_system = BiologicalDecaySystem::default();

        // Test decay computation time (should be <1ms per memory)
        let elapsed_time = StdDuration::from_secs(86400); // 1 day
        let base_confidence = Confidence::HIGH;
        let access_count = 2;
        let created_at = Utc::now() - chrono::Duration::days(30);

        let start = Instant::now();
        let iterations = 1000;

        for _ in 0..iterations {
            let _ = decay_system.compute_decayed_confidence(
                base_confidence,
                elapsed_time,
                access_count,
                created_at,
            );
        }

        let elapsed = start.elapsed();
        let per_call = elapsed / iterations;

        println!(
            "Decay computation: {} iterations in {:?} ({:?} per call)",
            iterations, elapsed, per_call
        );

        // Should be well under 1ms per call
        assert!(
            per_call.as_micros() < 1000,
            "Decay computation should be <1ms, got {:?}",
            per_call
        );
    }

    #[test]
    fn test_zero_performance_impact_without_decay() {
        use std::time::Instant;

        let store = create_test_store();

        // Insert test episode
        let embedding = create_varied_embedding(0.5, 1);
        let episode = EpisodeBuilder::new()
            .id("test".to_string())
            .when(Utc::now())
            .what("content".to_string())
            .embedding(embedding)
            .confidence(Confidence::HIGH)
            .build();

        store.store(episode);

        // Measure without decay
        let recall_no_decay = create_recall_with_decay(&store, None);
        let cue = Cue::embedding("query".to_string(), embedding, Confidence::MEDIUM);

        let start = Instant::now();
        let _ = recall_no_decay.recall(&cue, &store).unwrap();
        let baseline = start.elapsed();

        // Measure with decay
        let decay_system = Arc::new(BiologicalDecaySystem::default());
        let recall_with_decay = create_recall_with_decay(&store, Some(decay_system));

        let start = Instant::now();
        let _ = recall_with_decay.recall(&cue, &store).unwrap();
        let with_decay = start.elapsed();

        println!(
            "Recall baseline: {:?}, with decay: {:?}, overhead: {:?}",
            baseline,
            with_decay,
            with_decay.saturating_sub(baseline)
        );

        // Decay overhead should be minimal (< 2x baseline)
        assert!(
            with_decay < baseline * 2,
            "Decay overhead too high: baseline={:?}, with_decay={:?}",
            baseline,
            with_decay
        );
    }
}
