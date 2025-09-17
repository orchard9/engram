//! Tests for cue handling functionality

#[cfg(test)]
mod tests {
    use super::super::*;
    use crate::{Cue, CueType, Episode, Confidence, TemporalPattern};
    use chrono::{Utc, Duration};

    // Mock context implementation for testing
    struct MockCueContext {
        episodes: std::collections::HashMap<String, Episode>,
    }

    impl MockCueContext {
        fn new(episodes: Vec<Episode>) -> Self {
            let mut episode_map = std::collections::HashMap::new();
            for episode in episodes {
                episode_map.insert(episode.id.clone(), episode);
            }
            Self { 
                episodes: episode_map,
            }
        }
    }

    impl CueContext for MockCueContext {
        fn get_episodes(&self) -> &std::collections::HashMap<String, Episode> {
            &self.episodes
        }

        fn compute_similarity(&self, a: &[f32; 768], b: &[f32; 768]) -> f32 {
            // Simple dot product similarity
            a.iter()
                .zip(b.iter())
                .map(|(x, y)| x * y)
                .sum::<f32>()
                .max(0.0)
                .min(1.0)
        }

        fn apply_decay(&self, _episode: &Episode, confidence: Confidence) -> Confidence {
            // Simple decay - reduce confidence by 10%
            Confidence::exact(confidence.raw() * 0.9)
        }

        fn search_similar_episodes(&self, _vector: &[f32; 768], k: usize, _threshold: Confidence) -> Vec<(String, Confidence)> {
            // Simple mock that returns first k episode IDs from our map
            self.episodes
                .keys()
                .take(k.min(3))
                .map(|id| (id.clone(), Confidence::HIGH))
                .collect()
        }
    }

    fn create_test_episode(id: &str, what: &str, embedding: [f32; 768]) -> Episode {
        use crate::EpisodeBuilder;
        
        EpisodeBuilder::new()
            .id(id.to_string())
            .when(Utc::now())
            .what(what.to_string())
            .embedding(embedding)
            .confidence(Confidence::HIGH)
            .build()
    }

    #[test]
    fn test_embedding_cue_handler() {
        let embedding1 = [0.5; 768];
        let embedding2 = [0.8; 768];
        let embedding3 = [0.1; 768];

        let episodes = vec![
            create_test_episode("1", "test episode 1", embedding1),
            create_test_episode("2", "test episode 2", embedding2),
            create_test_episode("3", "test episode 3", embedding3),
        ];

        let context = MockCueContext::new(episodes);
        let handler = EmbeddingCueHandler;

        let cue = Cue {
            id: "test_cue".to_string(),
            cue_type: CueType::Embedding {
                vector: [0.6; 768],
                threshold: Confidence::exact(0.3),
            },
            cue_confidence: Confidence::HIGH,
            result_threshold: Confidence::MEDIUM,
            max_results: 10,
        };

        let results = handler.handle(&cue, &context).expect("Handler should succeed");
        
        // Should return episodes that meet the similarity threshold
        assert!(!results.is_empty(), "Should find matching episodes");
        
        // Results should be sorted by confidence (highest first)
        for i in 1..results.len() {
            assert!(
                results[i-1].1.raw() >= results[i].1.raw(),
                "Results should be sorted by confidence"
            );
        }
    }

    #[test]
    fn test_semantic_cue_handler() {
        let episodes = vec![
            create_test_episode("1", "walking in the park", [0.1; 768]),
            create_test_episode("2", "running in the garden", [0.2; 768]),
            create_test_episode("3", "cooking dinner", [0.3; 768]),
        ];

        let context = MockCueContext::new(episodes);
        let handler = SemanticCueHandler;

        let cue = Cue {
            id: "test_cue".to_string(),
            cue_type: CueType::Semantic {
                content: "walking park".to_string(),
                fuzzy_threshold: Confidence::exact(0.1),
            },
            cue_confidence: Confidence::HIGH,
            result_threshold: Confidence::MEDIUM,
            max_results: 10,
        };

        let results = handler.handle(&cue, &context).expect("Handler should succeed");
        
        // Should find episodes with semantic similarity
        assert!(!results.is_empty(), "Should find semantically similar episodes");
        
        // Episode 1 should have highest confidence (exact word matches)
        let episode1_result = results.iter().find(|(ep, _)| ep.id == "1");
        assert!(episode1_result.is_some(), "Should find episode 1");
    }

    #[test]
    fn test_temporal_cue_handler() {
        let now = Utc::now();
        let recent_time = now - Duration::hours(1);
        let old_time = now - Duration::days(10);

        let episodes = vec![
            create_test_episode("recent", "recent episode", [0.1; 768]),
            create_test_episode("old", "old episode", [0.2; 768]),
        ];

        // Update the timestamps
        let mut episodes = episodes;
        episodes[0].when = recent_time;
        episodes[1].when = old_time;

        let context = MockCueContext::new(episodes);
        let handler = TemporalCueHandler;

        let cue = Cue {
            id: "test_cue".to_string(),
            cue_type: CueType::Temporal {
                pattern: TemporalPattern::Recent(Duration::days(2)),
                confidence_threshold: Confidence::exact(0.1),
            },
            cue_confidence: Confidence::HIGH,
            result_threshold: Confidence::MEDIUM,
            max_results: 10,
        };

        let results = handler.handle(&cue, &context).expect("Handler should succeed");
        
        // Should only find recent episodes
        assert_eq!(results.len(), 1, "Should find only recent episode");
        assert_eq!(results[0].0.id, "recent", "Should find the recent episode");
    }

    #[test]
    fn test_context_cue_handler() {
        let now = Utc::now();
        let start_time = now - Duration::hours(2);
        let end_time = now - Duration::hours(1);
        let outside_time = now - Duration::days(1);

        let episodes = vec![
            create_test_episode("inside", "episode inside range", [0.1; 768]),
            create_test_episode("outside", "episode outside range", [0.2; 768]),
        ];

        // Update timestamps
        let mut episodes = episodes;
        episodes[0].when = start_time + Duration::minutes(30); // Inside range
        episodes[1].when = outside_time; // Outside range

        let context = MockCueContext::new(episodes);
        let handler = ContextCueHandler;

        let cue = Cue {
            id: "test_cue".to_string(),
            cue_type: CueType::Context {
                time_range: Some((start_time, end_time)),
                location: None,
                confidence_threshold: Confidence::exact(0.5),
            },
            cue_confidence: Confidence::HIGH,
            result_threshold: Confidence::MEDIUM,
            max_results: 10,
        };

        let results = handler.handle(&cue, &context).expect("Handler should succeed");
        
        // Should only find episodes within the time range
        assert_eq!(results.len(), 1, "Should find only episode in time range");
        assert_eq!(results[0].0.id, "inside", "Should find the episode inside range");
    }

    #[test]
    fn test_cue_dispatcher() {
        let episodes = vec![
            create_test_episode("1", "test episode", [0.5; 768]),
        ];

        let context = MockCueContext::new(episodes);
        let dispatcher = CueDispatcher::new();

        let cue = Cue {
            id: "test_cue".to_string(),
            cue_type: CueType::Embedding {
                vector: [0.5; 768],
                threshold: Confidence::exact(0.3),
            },
            cue_confidence: Confidence::HIGH,
            result_threshold: Confidence::MEDIUM,
            max_results: 10,
        };

        let results = dispatcher.handle_cue(&cue, &context).expect("Dispatcher should succeed");
        
        // Should successfully route to the embedding handler
        assert!(!results.is_empty(), "Should find matching episodes");
    }

    #[test]
    fn test_unsupported_cue_type() {
        let episodes = vec![
            create_test_episode("1", "test episode", [0.5; 768]),
        ];

        let context = MockCueContext::new(episodes);
        let handler = EmbeddingCueHandler;

        let cue = Cue {
            id: "test_cue".to_string(),
            cue_type: CueType::Semantic {
                content: "test".to_string(),
                fuzzy_threshold: Confidence::MEDIUM,
            },
            cue_confidence: Confidence::HIGH,
            result_threshold: Confidence::MEDIUM,
            max_results: 10,
        };

        let result = handler.handle(&cue, &context);
        
        // Should return an error for unsupported cue type
        assert!(result.is_err(), "Should return error for unsupported cue type");
        
        if let Err(crate::error::CueError::UnsupportedCueType { cue_type, operation }) = result {
            assert!(cue_type.contains("Semantic"), "Error should mention semantic cue type");
            assert_eq!(operation, "embedding_handler", "Error should mention the handler");
        } else {
            panic!("Should return UnsupportedCueType error");
        }
    }
}