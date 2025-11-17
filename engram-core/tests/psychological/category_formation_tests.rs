use engram_core::consolidation::{ConceptFormationEngine, SleepStage};
use engram_core::memory::MemoryNodeType;

use super::test_datasets::{bird_exemplars, create_episode};

pub(crate) fn run_rosch_1975_prototype_effects() {
    let engine = ConceptFormationEngine::default();
    let mut typical_episodes = Vec::new();
    let mut atypical_episodes = Vec::new();

    for (idx, exemplar) in bird_exemplars().into_iter().enumerate() {
        let episode = create_episode(
            exemplar.name,
            exemplar.features,
            exemplar.typicality,
            (idx * 6) as i64,
        );
        if exemplar.typicality >= 0.9 {
            typical_episodes.push(episode);
        } else {
            atypical_episodes.push(episode);
        }
    }

    let mut concepts = Vec::new();
    for _ in 0..8 {
        concepts = engine.form_concepts(&typical_episodes, SleepStage::NREM2);
        if !concepts.is_empty() {
            break;
        }
    }
    assert!(
        !concepts.is_empty(),
        "Concept formation should produce at least one bird prototype"
    );

    let concept = concepts.remove(0);
    let (centroid, coherence) = match concept.concept_node.node_type {
        MemoryNodeType::Concept {
            centroid,
            coherence,
            ..
        } => (centroid, coherence),
        _ => panic!("Expected concept node"),
    };

    let robin_distance = euclidean_distance(&typical_episodes[0].embedding, &centroid);
    let penguin_distance = euclidean_distance(&atypical_episodes[0].embedding, &centroid);

    assert!(
        robin_distance < penguin_distance,
        "Prototype should be closer to centroid than atypical exemplar"
    );

    let typical_scores: Vec<f32> = typical_episodes
        .iter()
        .map(|episode| 1.0 - euclidean_distance(&episode.embedding, &centroid))
        .collect();
    let avg_prototype = typical_scores.iter().sum::<f32>() / typical_scores.len() as f32;
    let avg_atypical = atypical_episodes
        .iter()
        .map(|episode| 1.0 - euclidean_distance(&episode.embedding, &centroid))
        .sum::<f32>()
        / atypical_episodes.len() as f32;

    assert!(
        avg_prototype > avg_atypical + 0.1,
        "Typicality difference should exceed 0.1"
    );

    assert!(
        coherence > 0.65,
        "Concept coherence ({:.3}) must exceed CA3 threshold",
        coherence
    );

    assert!(
        (penguin_distance - robin_distance) > 0.2,
        "Atypical members should sit farther from centroid (Δ {:.3})",
        penguin_distance - robin_distance
    );

    let activation_ratio = penguin_distance / robin_distance.max(1e-3);
    assert!(
        activation_ratio > 1.5,
        "Prototype distance ratio {:.2} should exceed 1.5",
        activation_ratio
    );
}

fn euclidean_distance(
    a: &[f32; engram_core::EMBEDDING_DIM],
    b: &[f32; engram_core::EMBEDDING_DIM],
) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f32>()
        .sqrt()
}
