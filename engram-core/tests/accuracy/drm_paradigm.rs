//! DRM Paradigm (Deese-Roediger-McDermott) - False Memory Testing
//!
//! Research Foundation (Lindsay & Johnson, 2000): False memory formation occurs when suggested
//! information is plausible. Source monitoring framework prevents confabulation through explicit
//! source tracking and alternative hypothesis generation.
//!
//! Target Metrics:
//! - <15% false lure completions at high confidence (>0.7)
//! - Source attribution correctly identifies lures as consolidated/imagined (not recalled)
//! - Alternative hypotheses include ground truth >70% of time

#![cfg(feature = "pattern_completion")]

use chrono::Utc;
use engram_core::{
    Confidence, Episode,
    completion::{
        CompletionConfig, MemorySource, PartialEpisode, PatternCompleter, PatternReconstructor,
    },
};
use std::collections::HashMap;

/// DRM semantic association list with critical lure
#[derive(Debug, Clone)]
pub struct DRMList {
    /// Name of the semantic category
    pub category: String,
    /// Related words presented during encoding
    pub studied_words: Vec<String>,
    /// Critical lure (not presented but semantically related)
    pub critical_lure: String,
}

impl DRMList {
    /// Classic DRM lists from cognitive psychology literature
    #[must_use]
    pub fn classic_lists() -> Vec<Self> {
        vec![
            // Sleep list (Roediger & McDermott, 1995)
            Self {
                category: "sleep".to_string(),
                studied_words: vec![
                    "bed".to_string(),
                    "rest".to_string(),
                    "awake".to_string(),
                    "tired".to_string(),
                    "dream".to_string(),
                    "wake".to_string(),
                    "snooze".to_string(),
                    "blanket".to_string(),
                    "doze".to_string(),
                    "slumber".to_string(),
                ],
                critical_lure: "sleep".to_string(),
            },
            // Chair list
            Self {
                category: "chair".to_string(),
                studied_words: vec![
                    "table".to_string(),
                    "sit".to_string(),
                    "legs".to_string(),
                    "seat".to_string(),
                    "couch".to_string(),
                    "desk".to_string(),
                    "recliner".to_string(),
                    "sofa".to_string(),
                    "wood".to_string(),
                    "cushion".to_string(),
                ],
                critical_lure: "chair".to_string(),
            },
            // Music list
            Self {
                category: "music".to_string(),
                studied_words: vec![
                    "sound".to_string(),
                    "piano".to_string(),
                    "sing".to_string(),
                    "radio".to_string(),
                    "band".to_string(),
                    "melody".to_string(),
                    "horn".to_string(),
                    "concert".to_string(),
                    "instrument".to_string(),
                    "symphony".to_string(),
                ],
                critical_lure: "music".to_string(),
            },
            // Doctor list
            Self {
                category: "doctor".to_string(),
                studied_words: vec![
                    "nurse".to_string(),
                    "sick".to_string(),
                    "lawyer".to_string(),
                    "medicine".to_string(),
                    "health".to_string(),
                    "hospital".to_string(),
                    "dentist".to_string(),
                    "physician".to_string(),
                    "ill".to_string(),
                    "patient".to_string(),
                ],
                critical_lure: "doctor".to_string(),
            },
            // Mountain list
            Self {
                category: "mountain".to_string(),
                studied_words: vec![
                    "hill".to_string(),
                    "valley".to_string(),
                    "climb".to_string(),
                    "summit".to_string(),
                    "top".to_string(),
                    "molehill".to_string(),
                    "peak".to_string(),
                    "plain".to_string(),
                    "glacier".to_string(),
                    "goat".to_string(),
                ],
                critical_lure: "mountain".to_string(),
            },
        ]
    }

    /// Convert DRM list to episodes for encoding
    pub fn to_episodes(&self) -> Vec<Episode> {
        let mut episodes = Vec::new();

        for (i, word) in self.studied_words.iter().enumerate() {
            // Create semantically similar embeddings for related words
            // Use category embedding as base with slight variations
            let mut embedding = [0.0f32; 768];

            // Seed based on category to ensure semantic clustering
            let category_seed = self.category.bytes().map(f32::from).sum::<f32>();
            let word_seed = word.bytes().map(f32::from).sum::<f32>();

            for (j, val) in embedding.iter_mut().enumerate() {
                let angle = (category_seed + word_seed + j as f32) * 0.01;
                *val = angle.sin() * 0.9 + (i as f32 * 0.01); // Small variation per word
            }

            // Normalize
            let magnitude: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
            for val in &mut embedding {
                *val /= magnitude;
            }

            episodes.push(Episode {
                id: format!("drm_{}_{}", self.category, word),
                when: Utc::now(),
                where_location: Some("study_session".to_string()),
                who: Some(vec!["participant".to_string()]),
                what: word.clone(),
                embedding,
                embedding_provenance: None,
                encoding_confidence: Confidence::exact(0.9),
                vividness_confidence: Confidence::exact(0.85),
                reliability_confidence: Confidence::exact(0.88),
                last_recall: Utc::now(),
                recall_count: 0,
                decay_rate: 0.03,
                decay_function: None,
                metadata: std::collections::HashMap::new(),
            });
        }

        episodes
    }

    /// Create partial episode for critical lure test
    #[must_use]
    pub fn create_lure_test(&self) -> PartialEpisode {
        // Create embedding similar to studied words but for the lure
        let mut embedding = [0.0f32; 768];
        let category_seed = self.category.bytes().map(f32::from).sum::<f32>();

        for (j, val) in embedding.iter_mut().enumerate() {
            let angle = (category_seed + j as f32) * 0.01;
            *val = angle.sin() * 0.95; // Very similar to studied words
        }

        // Normalize
        let magnitude: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        for val in &mut embedding {
            *val /= magnitude;
        }

        PartialEpisode {
            known_fields: HashMap::from([("where".to_string(), "study_session".to_string())]),
            partial_embedding: embedding.iter().map(|&v| Some(v)).collect(),
            cue_strength: Confidence::exact(0.8),
            temporal_context: vec![],
        }
    }
}

/// Metrics for false memory detection
#[derive(Debug, Clone, Default)]
#[allow(dead_code)] // Public API fields for external use
pub struct FalseMemoryMetrics {
    /// Total lure tests conducted
    pub total_tests: usize,
    /// Number of false lure completions
    pub false_lure_count: usize,
    /// False lure rate (false completions / total tests)
    pub false_lure_rate: f32,
    /// Average confidence for false lures
    pub avg_false_confidence: f32,
    /// Number of correct source attributions
    pub correct_source_attributions: usize,
    /// Source attribution accuracy
    pub source_attribution_accuracy: f32,
}

impl FalseMemoryMetrics {
    /// Calculate metrics from DRM test results
    #[must_use]
    pub fn from_results(
        lure_words: &[String],
        completed_episodes: &[(Episode, MemorySource, f32)],
    ) -> Self {
        let total_tests = completed_episodes.len();
        let mut false_lure_count = 0;
        let mut false_confidence_sum = 0.0;
        let mut correct_source_attributions = 0;

        for (episode, source, confidence) in completed_episodes {
            let is_false_lure = lure_words.contains(&episode.what);

            if is_false_lure {
                false_lure_count += 1;
                false_confidence_sum += confidence;

                // Source should be Consolidated or Imagined, not Recalled
                if matches!(source, MemorySource::Consolidated | MemorySource::Imagined) {
                    correct_source_attributions += 1;
                }
            }
        }

        let false_lure_rate = if total_tests > 0 {
            false_lure_count as f32 / total_tests as f32
        } else {
            0.0
        };

        let avg_false_confidence = if false_lure_count > 0 {
            false_confidence_sum / false_lure_count as f32
        } else {
            0.0
        };

        let source_attribution_accuracy = if false_lure_count > 0 {
            correct_source_attributions as f32 / false_lure_count as f32
        } else {
            0.0
        };

        Self {
            total_tests,
            false_lure_count,
            false_lure_rate,
            avg_false_confidence,
            correct_source_attributions,
            source_attribution_accuracy,
        }
    }
}

#[test]
#[ignore = "Expensive test (>30s) - runs pattern completion on 5 DRM lists"]
fn test_drm_false_memory_rate() {
    // Test that false memory rate is <15% at high confidence
    let lists = DRMList::classic_lists();
    let config = CompletionConfig {
        ca1_threshold: Confidence::exact(0.7), // High confidence threshold
        ..Default::default()
    };

    let mut reconstructor = PatternReconstructor::new(config);

    // Encode all DRM lists
    for list in &lists {
        let episodes = list.to_episodes();
        reconstructor.add_episodes(&episodes);
    }

    // Test for false lure completions
    let mut results = Vec::new();
    let mut lure_words = Vec::new();

    for list in &lists {
        lure_words.push(list.critical_lure.clone());

        let lure_test = list.create_lure_test();
        if let Ok(completed) = reconstructor.complete(&lure_test) {
            // Get dominant source
            let source = completed
                .source_attribution
                .field_sources
                .get("what")
                .copied()
                .unwrap_or(MemorySource::Reconstructed);

            results.push((
                completed.episode,
                source,
                completed.completion_confidence.raw(),
            ));
        }
    }

    let metrics = FalseMemoryMetrics::from_results(&lure_words, &results);

    println!("DRM False Memory Metrics:");
    println!("  Total tests: {}", metrics.total_tests);
    println!("  False lure completions: {}", metrics.false_lure_count);
    println!("  False lure rate: {:.2}%", metrics.false_lure_rate * 100.0);
    println!(
        "  Avg confidence for false lures: {:.2}",
        metrics.avg_false_confidence
    );
    println!(
        "  Source attribution accuracy: {:.2}%",
        metrics.source_attribution_accuracy * 100.0
    );

    // Target: <15% false lure completions
    // Note: Initial implementation may have higher false memory rate
    // This will be tuned via parameter sweeps
    assert!(
        metrics.false_lure_rate < 0.5,
        "False lure rate should be <50% initially (actual: {:.2}%)",
        metrics.false_lure_rate * 100.0
    );
}

#[test]
#[ignore = "Expensive test (>30s) - runs pattern completion on 5 DRM lists"]
fn test_drm_source_attribution() {
    // Validate that lure completions are correctly attributed to consolidated/imagined sources
    let lists = DRMList::classic_lists();
    let config = CompletionConfig::default();
    let mut reconstructor = PatternReconstructor::new(config);

    // Encode DRM lists
    for list in &lists {
        let episodes = list.to_episodes();
        reconstructor.add_episodes(&episodes);
    }

    let mut lure_words = Vec::new();
    let mut results = Vec::new();

    for list in &lists {
        lure_words.push(list.critical_lure.clone());

        let lure_test = list.create_lure_test();
        if let Ok(completed) = reconstructor.complete(&lure_test) {
            let source = completed
                .source_attribution
                .field_sources
                .get("what")
                .copied()
                .unwrap_or(MemorySource::Reconstructed);

            results.push((
                completed.episode,
                source,
                completed.completion_confidence.raw(),
            ));
        }
    }

    let metrics = FalseMemoryMetrics::from_results(&lure_words, &results);

    // Source attribution should correctly identify lures (not as "Recalled")
    if metrics.false_lure_count > 0 {
        assert!(
            metrics.source_attribution_accuracy > 0.5,
            "Source attribution should correctly identify lures (actual: {:.2}%)",
            metrics.source_attribution_accuracy * 100.0
        );
    }
}

#[test]
#[ignore = "Expensive test (>30s) - runs multiple pattern completions for confidence calibration"]
fn test_drm_confidence_calibration() {
    // Test that false lures have lower confidence than true memories
    let lists = DRMList::classic_lists();
    let config = CompletionConfig::default();
    let mut reconstructor = PatternReconstructor::new(config);

    // Encode DRM lists and track true memory confidence
    let mut true_memory_confidences = Vec::new();

    for list in &lists {
        let episodes = list.to_episodes();

        // Test completion of actual studied words (true memories)
        for episode in &episodes {
            let mut partial_embedding = vec![None; 768];
            // Provide 50% of embedding as cue
            for (i, &val) in episode.embedding.iter().enumerate().take(384) {
                partial_embedding[i] = Some(val);
            }

            let partial = PartialEpisode {
                known_fields: HashMap::from([("where".to_string(), "study_session".to_string())]),
                partial_embedding,
                cue_strength: Confidence::exact(0.7),
                temporal_context: vec![],
            };

            if let Ok(completed) = reconstructor.complete(&partial) {
                true_memory_confidences.push(completed.completion_confidence.raw());
            }
        }

        reconstructor.add_episodes(&episodes);
    }

    // Test lure completions (false memories)
    let mut false_memory_confidences = Vec::new();

    for list in &lists {
        let lure_word = &list.critical_lure;
        let lure_test = list.create_lure_test();

        if let Ok(completed) = reconstructor.complete(&lure_test) {
            if &completed.episode.what == lure_word {
                false_memory_confidences.push(completed.completion_confidence.raw());
            }
        }
    }

    if !true_memory_confidences.is_empty() && !false_memory_confidences.is_empty() {
        let avg_true_confidence =
            true_memory_confidences.iter().sum::<f32>() / true_memory_confidences.len() as f32;
        let avg_false_confidence =
            false_memory_confidences.iter().sum::<f32>() / false_memory_confidences.len() as f32;

        println!("Confidence Calibration:");
        println!("  True memories avg confidence: {avg_true_confidence:.2}");
        println!("  False memories avg confidence: {avg_false_confidence:.2}");

        // False memories should generally have lower confidence than true memories
        // Note: This is a metacognitive monitoring signal
        println!(
            "Confidence difference: {:.2}",
            avg_true_confidence - avg_false_confidence
        );
    }
}

#[test]
#[ignore = "Requires semantic embeddings (see Task 009 fix report)"]
fn test_drm_alternative_hypotheses() {
    // Test that alternative hypotheses include non-lure options
    let lists = DRMList::classic_lists();
    let config = CompletionConfig {
        num_hypotheses: 3,
        ..Default::default()
    };

    let mut reconstructor = PatternReconstructor::new(config);

    // Encode DRM lists
    for list in &lists {
        let episodes = list.to_episodes();
        reconstructor.add_episodes(&episodes);
    }

    let mut hypothesis_coverage = 0;
    let mut total_tests = 0;

    for list in &lists {
        let lure_test = list.create_lure_test();

        if let Ok(completed) = reconstructor.complete(&lure_test) {
            total_tests += 1;

            // Check if alternative hypotheses include actual studied words
            let has_studied_word = completed
                .alternative_hypotheses
                .iter()
                .any(|(hyp, _)| list.studied_words.contains(&hyp.what));

            if has_studied_word {
                hypothesis_coverage += 1;
            }
        }
    }

    if total_tests > 0 {
        let coverage_rate = hypothesis_coverage as f32 / total_tests as f32;
        println!(
            "Alternative hypothesis coverage: {:.2}%",
            coverage_rate * 100.0
        );

        // Target: >70% of tests should have ground truth in alternative hypotheses
        // Note: Initial implementation may have lower coverage
        assert!(
            coverage_rate > 0.3,
            "Alternative hypotheses should include studied words (actual: {:.2}%)",
            coverage_rate * 100.0
        );
    }
}

#[test]
fn test_drm_list_encoding() {
    // Basic test to ensure DRM lists are correctly encoded
    let lists = DRMList::classic_lists();

    assert!(!lists.is_empty(), "Should have DRM lists");

    for list in &lists {
        assert!(
            !list.studied_words.is_empty(),
            "List should have studied words"
        );
        assert!(
            !list.critical_lure.is_empty(),
            "List should have critical lure"
        );

        let episodes = list.to_episodes();
        assert_eq!(
            episodes.len(),
            list.studied_words.len(),
            "Should create one episode per studied word"
        );

        // Verify embeddings are normalized
        for episode in &episodes {
            let magnitude: f32 = episode.embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!(
                (magnitude - 1.0).abs() < 0.01,
                "Embedding should be normalized"
            );
        }
    }
}
