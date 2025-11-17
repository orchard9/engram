use engram_core::Confidence;
use engram_core::EMBEDDING_DIM;
use engram_core::memory::Memory;
use engram_core::memory_graph::UnifiedMemoryGraph;

use super::statistical_analysis::eta_squared;
use super::test_datasets::{
    blend_embeddings, context_word_list, embedding_for_label, land_context_embedding,
    water_context_embedding,
};

pub(crate) fn run_godden_baddeley_context_effect() {
    let graph = UnifiedMemoryGraph::concurrent();
    let land_context = land_context_embedding();
    let water_context = water_context_embedding();
    let words = context_word_list();

    let land_ids = store_with_context(&graph, &words, &land_context, "land");
    let water_ids = store_with_context(&graph, &words, &water_context, "water");

    let matching_land = recall_with_context(&graph, &land_ids, &land_context);
    let mismatching_land = recall_with_context(&graph, &land_ids, &water_context);
    let matching_water = recall_with_context(&graph, &water_ids, &water_context);
    let mismatching_water = recall_with_context(&graph, &water_ids, &land_context);

    let avg_matching = (matching_land + matching_water) / 2.0;
    let avg_mismatching = (mismatching_land + mismatching_water) / 2.0;

    let advantage = (avg_matching - avg_mismatching) / avg_mismatching.max(0.01);
    assert!(
        advantage > 0.25,
        "Context match improvement should exceed 25%, observed {:.1}%",
        advantage * 100.0
    );

    let groups = vec![matching_land, matching_water];
    let mismatches = vec![mismatching_land, mismatching_water];
    let effect = eta_squared(&[&groups, &mismatches]);
    assert!(
        effect > 0.4,
        "Context effect should be large (η² > 0.4), observed {:.2}",
        effect
    );
}

fn store_with_context(
    graph: &UnifiedMemoryGraph<engram_core::memory_graph::backends::DashMapBackend>,
    words: &[&str],
    context: &[f32; EMBEDDING_DIM],
    suffix: &str,
) -> Vec<uuid::Uuid> {
    words
        .iter()
        .map(|word| {
            let word_embedding = embedding_for_label(word);
            let contextualized = blend_embeddings(&word_embedding, context, 0.3);
            graph
                .store_memory(Memory::new(
                    format!("{word}_{suffix}"),
                    contextualized,
                    Confidence::MEDIUM,
                ))
                .expect("store memory")
        })
        .collect()
}

fn recall_with_context(
    graph: &UnifiedMemoryGraph<engram_core::memory_graph::backends::DashMapBackend>,
    ids: &[uuid::Uuid],
    context: &[f32; EMBEDDING_DIM],
) -> f32 {
    let mut hits = 0.0;
    for id in ids {
        if let Some(memory) = graph.retrieve(id).expect("memory retrieval") {
            let similarity = cosine_similarity(&memory.embedding, context);
            if similarity > 0.72 {
                hits += 1.0;
            }
        }
    }
    hits / ids.len() as f32
}

fn cosine_similarity(a: &[f32; EMBEDDING_DIM], b: &[f32; EMBEDDING_DIM]) -> f32 {
    let dot = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum::<f32>();
    let norm_a = a.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-6);
    let norm_b = b.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-6);
    dot / (norm_a * norm_b)
}
