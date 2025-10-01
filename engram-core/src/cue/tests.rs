//! Tests for cue handling functionality

use super::{
    ContextCueHandler, CueContext, CueDispatcher, EmbeddingCueHandler, SemanticCueHandler,
    TemporalCueHandler,
};
use crate::cue::dispatcher::CueHandler;
use crate::error::CueError;
use crate::{Confidence, Cue, CueType, Episode, TemporalPattern};
use chrono::{Duration, Utc};
use std::collections::HashMap;

type TestResult<T = ()> = Result<T, String>;

struct MockCueContext {
    episodes: HashMap<String, Episode>,
}

impl MockCueContext {
    fn new(episodes: Vec<Episode>) -> Self {
        let mut episode_map = HashMap::new();
        for episode in episodes {
            episode_map.insert(episode.id.clone(), episode);
        }
        Self {
            episodes: episode_map,
        }
    }
}

impl CueContext for MockCueContext {
    fn get_episodes(&self) -> &HashMap<String, Episode> {
        &self.episodes
    }

    fn compute_similarity(&self, a: &[f32; 768], b: &[f32; 768]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| x * y)
            .sum::<f32>()
            .clamp(0.0, 1.0)
    }

    fn apply_decay(&self, _episode: &Episode, confidence: Confidence) -> Confidence {
        Confidence::exact(confidence.raw() * 0.9)
    }

    fn search_similar_episodes(
        &self,
        _vector: &[f32; 768],
        k: usize,
        _threshold: Confidence,
    ) -> Vec<(String, Confidence)> {
        self.episodes
            .keys()
            .take(k.min(3))
            .map(|id| (id.clone(), Confidence::HIGH))
            .collect()
    }
}

fn create_test_episode(id: &str, what: &str, embedding: &[f32; 768]) -> Episode {
    use crate::EpisodeBuilder;

    EpisodeBuilder::new()
        .id(id.to_string())
        .when(Utc::now())
        .what(what.to_string())
        .embedding(*embedding)
        .confidence(Confidence::HIGH)
        .build()
}

#[test]
fn test_embedding_cue_handler() -> TestResult {
    let embedding1 = [0.5; 768];
    let embedding2 = [0.8; 768];
    let embedding3 = [0.1; 768];

    let episodes = vec![
        create_test_episode("1", "test episode 1", &embedding1),
        create_test_episode("2", "test episode 2", &embedding2),
        create_test_episode("3", "test episode 3", &embedding3),
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

    let results = handler
        .handle(&cue, &context)
        .map_err(|err| format!("Embedding cue handler should succeed: {err:?}"))?;

    if results.is_empty() {
        return Err("Should find matching episodes".to_string());
    }

    for pair in results.windows(2) {
        if pair[0].1.raw() < pair[1].1.raw() {
            return Err("Results should be sorted by confidence".to_string());
        }
    }
    Ok(())
}

#[test]
fn test_semantic_cue_handler() -> TestResult {
    let episodes = vec![
        create_test_episode("1", "walking in the park", &[0.1; 768]),
        create_test_episode("2", "running in the garden", &[0.2; 768]),
        create_test_episode("3", "cooking dinner", &[0.3; 768]),
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

    let results = handler
        .handle(&cue, &context)
        .map_err(|err| format!("Semantic cue handler should succeed: {err:?}"))?;

    if results.is_empty() {
        return Err("Should find semantically similar episodes".to_string());
    }

    let episode1_result = results.iter().find(|(ep, _)| ep.id == "1");
    if episode1_result.is_none() {
        return Err("Should find episode 1".to_string());
    }

    Ok(())
}

#[test]
fn test_temporal_cue_handler() -> TestResult {
    let now = Utc::now();
    let recent_time = now - Duration::hours(1);
    let old_time = now - Duration::days(10);

    let mut episodes = vec![
        create_test_episode("recent", "recent episode", &[0.1; 768]),
        create_test_episode("old", "old episode", &[0.2; 768]),
    ];

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

    let results = handler
        .handle(&cue, &context)
        .map_err(|err| format!("Temporal cue handler should succeed: {err:?}"))?;

    if results.len() != 1 {
        return Err("Should find only recent episode".to_string());
    }
    if results[0].0.id != "recent" {
        return Err("Should find the recent episode".to_string());
    }
    Ok(())
}

#[test]
fn test_context_cue_handler() -> TestResult {
    let now = Utc::now();
    let start_time = now - Duration::hours(2);
    let end_time = now - Duration::hours(1);
    let outside_time = now - Duration::days(1);

    let mut episodes = vec![
        create_test_episode("inside", "episode inside range", &[0.1; 768]),
        create_test_episode("outside", "episode outside range", &[0.2; 768]),
    ];

    episodes[0].when = start_time + Duration::minutes(30);
    episodes[1].when = outside_time;

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

    let results = handler
        .handle(&cue, &context)
        .map_err(|err| format!("Context cue handler should succeed: {err:?}"))?;

    if results.len() != 1 {
        return Err("Should find only episode in time range".to_string());
    }
    if results[0].0.id != "inside" {
        return Err("Should find the episode inside range".to_string());
    }
    Ok(())
}

#[test]
fn test_cue_dispatcher() -> TestResult {
    let episodes = vec![create_test_episode("1", "test episode", &[0.5; 768])];

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

    let results = dispatcher
        .handle_cue(&cue, &context)
        .map_err(|err| format!("Cue dispatcher should succeed: {err:?}"))?;

    if results.is_empty() {
        return Err("Should find matching episodes".to_string());
    }

    Ok(())
}

#[test]
fn test_unsupported_cue_type() -> TestResult {
    let episodes = vec![create_test_episode("1", "test episode", &[0.5; 768])];

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

    match handler.handle(&cue, &context) {
        Ok(_) => Err("Expected unsupported cue type error".to_string()),
        Err(CueError::UnsupportedCueType {
            cue_type,
            operation,
        }) => {
            if !cue_type.contains("Semantic") {
                return Err("Error should mention semantic cue type".to_string());
            }
            if operation != "embedding_handler" {
                return Err("Error should mention the handler".to_string());
            }
            Ok(())
        }
        Err(other) => Err(format!("Unexpected error variant: {other:?}")),
    }
}
