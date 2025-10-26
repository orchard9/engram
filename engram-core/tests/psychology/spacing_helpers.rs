//! Helper functions and test materials for spacing effect validation
//!
//! Provides deterministic test fact generation, retention testing utilities,
//! and statistical analysis functions for validating the spacing effect.

use engram_core::{Confidence, Cue, CueType, EMBEDDING_DIM, Episode, EpisodeBuilder, MemoryStore};

/// Test fact for spacing effect experiments
#[derive(Debug, Clone, PartialEq)]
pub struct TestFact {
    pub id: String,
    pub question: String,
    pub answer: String,
    pub embedding: Vec<f32>,
}

/// Generate random facts for testing
///
/// Creates fact pairs (question → answer) with realistic embeddings
/// that can be used for recall testing. Uses deterministic RNG for reproducibility.
#[must_use]
pub fn generate_random_facts(count: usize, seed: u64) -> Vec<TestFact> {
    use rand::prelude::*;
    use rand_chacha::ChaCha8Rng;

    let mut rng = ChaCha8Rng::seed_from_u64(seed);

    // Fact templates for variety
    let templates = [
        (
            "What is the capital of {}?",
            vec!["France", "Japan", "Brazil", "Egypt", "Kenya"],
        ),
        (
            "Who invented the {}?",
            vec!["telephone", "airplane", "computer", "internet", "lightbulb"],
        ),
        (
            "What year did {} happen?",
            vec!["WWI", "WWII", "moon landing", "fall of Berlin Wall", "9/11"],
        ),
        (
            "What is the chemical symbol for {}?",
            vec!["gold", "silver", "iron", "helium", "carbon"],
        ),
    ];

    let mut facts = Vec::new();

    for i in 0..count {
        let template_idx = i % templates.len();
        let (question_template, answers) = &templates[template_idx];

        let answer_idx = rng.gen_range(0..answers.len());
        let answer = answers[answer_idx];

        let question = question_template.replace("{}", answer);

        // Generate embedding (deterministic based on content)
        #[allow(clippy::cast_possible_truncation)]
        let embedding = generate_fact_embedding(&question, answer, seed + i as u64);

        facts.push(TestFact {
            id: format!("fact_{i}"),
            question: question.clone(),
            answer: answer.to_string(),
            embedding,
        });
    }

    facts
}

/// Generate deterministic embedding for a fact
fn generate_fact_embedding(question: &str, answer: &str, seed: u64) -> Vec<f32> {
    use rand::prelude::*;
    use rand_chacha::ChaCha8Rng;

    let mut rng = ChaCha8Rng::seed_from_u64(seed);

    // Create embedding with controlled properties:
    // 1. Deterministic (same inputs → same embedding)
    // 2. Normalized (unit length)
    // 3. Question and answer embeddings are similar but distinct

    let mut embedding = vec![0.0; EMBEDDING_DIM];

    // Fill with Gaussian noise approximation
    for val in &mut embedding {
        *val = rng.gen_range(-1.0..1.0);
    }

    // Add content-specific features (simple hash-based)
    let content_hash = question.len() + answer.len();
    for (i, val) in embedding.iter_mut().enumerate().take(10) {
        *val += 0.1 * ((content_hash + i) % 7) as f32;
    }

    // Normalize to unit length
    let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for val in &mut embedding {
            *val /= norm;
        }
    }

    embedding
}

/// Test retention for a list of facts
///
/// Returns the proportion of facts successfully recalled
#[must_use]
#[allow(dead_code)] // Used by spacing_effect tests
pub fn test_retention(store: &MemoryStore, facts: &[TestFact]) -> f32 {
    let mut correct = 0;

    for fact in facts {
        // Try to recall the fact by its embedding
        let cue = Cue {
            id: format!("recall_{}", fact.id),
            cue_type: CueType::Embedding {
                vector: fact.embedding.clone().try_into().unwrap_or([0.0; 768]),
                threshold: Confidence::LOW,
            },
            cue_confidence: Confidence::HIGH,
            result_threshold: Confidence::LOW,
            max_results: 10,
            embedding_provenance: None,
        };

        let recall_result = store.recall(&cue);
        let results = recall_result.results;

        // Check if the answer was recalled with sufficient confidence
        let recalled_correctly = results.iter().any(|(episode, conf)| {
            // Check semantic similarity to expected embedding
            let similarity = cosine_similarity(&episode.embedding, &fact.embedding);

            // Must have high similarity and sufficient confidence
            similarity > 0.85 && conf.raw() > 0.3
        });

        if recalled_correctly {
            correct += 1;
        }
    }

    #[allow(clippy::cast_precision_loss)]
    let retention = correct as f32 / facts.len() as f32;
    retention
}

/// Compute cosine similarity between two vectors
#[must_use]
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a > 0.0 && norm_b > 0.0 {
        dot / (norm_a * norm_b)
    } else {
        0.0
    }
}

/// Convert `TestFact` to Episode for storage
#[must_use]
#[allow(dead_code)] // Used by spacing_effect tests
pub fn fact_to_episode(fact: &TestFact, timestamp: chrono::DateTime<chrono::Utc>) -> Episode {
    let embedding_array: [f32; 768] = fact.embedding.clone().try_into().unwrap_or([0.0; 768]);

    EpisodeBuilder::new()
        .id(fact.id.clone())
        .when(timestamp)
        .what(format!("{} → {}", fact.question, fact.answer))
        .embedding(embedding_array)
        .confidence(Confidence::HIGH)
        .build()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_random_facts_determinism() {
        let facts1 = generate_random_facts(10, 42);
        let facts2 = generate_random_facts(10, 42);

        assert_eq!(facts1.len(), facts2.len());
        for (f1, f2) in facts1.iter().zip(facts2.iter()) {
            assert_eq!(f1.question, f2.question);
            assert_eq!(f1.answer, f2.answer);
            assert_eq!(f1.embedding, f2.embedding);
        }
    }

    #[test]
    fn test_fact_embeddings_normalized() {
        let facts = generate_random_facts(10, 42);

        for fact in &facts {
            let norm: f32 = fact.embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!(
                (norm - 1.0).abs() < 0.01,
                "Embedding not normalized: norm = {norm:.3}"
            );
        }
    }

    #[test]
    fn test_different_seeds_produce_different_facts() {
        let facts1 = generate_random_facts(5, 42);
        let facts2 = generate_random_facts(5, 43);

        // Should have different embeddings
        assert_ne!(facts1[0].embedding, facts2[0].embedding);
    }

    #[test]
    fn test_cosine_similarity_identical() {
        let vec = vec![0.5, 0.5, 0.5, 0.5];
        let sim = cosine_similarity(&vec, &vec);
        assert!((sim - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let vec1 = vec![1.0, 0.0, 0.0, 0.0];
        let vec2 = vec![0.0, 1.0, 0.0, 0.0];
        let sim = cosine_similarity(&vec1, &vec2);
        assert!(sim.abs() < 0.001);
    }
}
