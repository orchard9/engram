//! Cognitive Recall Patterns Example
//!
//! Demonstrates three recall scenarios using deterministic spreading activation:
//! semantic priming, episodic reconstruction, and confidence-guided exploration.
//! Runs only when the `hnsw_index` feature is enabled.

#[cfg(feature = "hnsw_index")]
use chrono::Utc;
#[cfg(feature = "hnsw_index")]
use engram_core::activation::{
    ActivationGraphExt, ConfidenceAggregator, EdgeType, ParallelSpreadingConfig, RecallMode,
    create_activation_graph,
    cycle_detector::CycleDetector,
    parallel::ParallelSpreadingEngine,
    recall::{CognitiveRecall, CognitiveRecallBuilder, RecallConfig},
    seeding::VectorActivationSeeder,
    similarity_config::SimilarityConfig,
};
#[cfg(feature = "hnsw_index")]
use engram_core::{Confidence, Cue, EpisodeBuilder, MemoryStore};
#[cfg(feature = "hnsw_index")]
use std::collections::HashMap;
#[cfg(feature = "hnsw_index")]
use std::sync::Arc;
#[cfg(feature = "hnsw_index")]
use std::time::Duration;

#[cfg(not(feature = "hnsw_index"))]
fn main() {
    println!("This example requires the 'hnsw_index' feature to be enabled.");
    println!("Run with: cargo run --example cognitive_recall_patterns --features hnsw_index");
}

#[cfg(feature = "hnsw_index")]
fn main() {
    println!("=== Cognitive Recall Patterns ===\n");

    // 1. Seed the memory store with a small clinical scenario graph.
    let store = seed_memory_graph();

    // 2. Build shared spreading infrastructure (deterministic for reproducibility).
    let index = store
        .hnsw_index()
        .expect("HNSW index should be available when feature is enabled");
    let graph = Arc::new(create_activation_graph());
    wire_spreading_edges(&graph);

    let seeder = Arc::new(VectorActivationSeeder::with_default_resolver(
        index,
        SimilarityConfig::default(),
    ));

    let spreading_config = ParallelSpreadingConfig {
        deterministic: true,
        seed: Some(42),
        trace_activation_flow: true,
        max_depth: 3,
        threshold: 0.08,
        ..ParallelSpreadingConfig::default()
    };

    let spreading_engine = Arc::new(
        ParallelSpreadingEngine::new(spreading_config, Arc::clone(&graph))
            .expect("failed to initialise spreading engine"),
    );

    // 3. Demonstrate semantic priming (doctor → nurse).
    println!("-- Semantic Priming (doctor → nurse) --");
    {
        let semantic_recall = build_recall(
            &seeder,
            &spreading_engine,
            RecallMode::Spreading,
            Duration::from_millis(12),
        );
        let doctor_cue = Cue::embedding(
            "doctor_cue".to_string(),
            uniform_embedding(0.90),
            Confidence::from_raw(0.92),
        );
        let semantic_results = semantic_recall
            .recall(&doctor_cue, &store)
            .expect("semantic recall failed");
        display_results("Primed results", &semantic_results);
    }

    // 4. Episodic reconstruction from partial cues.
    println!("\n-- Episodic Reconstruction (clinic follow-up) --");
    {
        let episodic_recall = build_recall(
            &seeder,
            &spreading_engine,
            RecallMode::Hybrid,
            Duration::from_millis(15),
        );
        let clinic_cue = Cue::embedding(
            "clinic_fragment".to_string(),
            uniform_embedding(0.76),
            Confidence::from_raw(0.8),
        );
        let episodic_results = episodic_recall
            .recall(&clinic_cue, &store)
            .expect("episodic recall failed");
        display_results("Reconstructed timeline", &episodic_results);
    }

    // 5. Confidence-guided exploration at a relaxed threshold.
    println!("\n-- Confidence-Guided Exploration --");
    {
        let mut exploratory_config = RecallConfig {
            recall_mode: RecallMode::Spreading,
            min_confidence: 0.25,
            max_results: 8,
            time_budget: Duration::from_millis(18),
            ..RecallConfig::default()
        };
        exploratory_config.enable_recency_boost = false;

        let exploratory_recall =
            build_recall_with_config(&seeder, &spreading_engine, exploratory_config);

        let exploration_cue = Cue::embedding(
            "confidence_probe".to_string(),
            uniform_embedding(0.72),
            Confidence::from_raw(0.72),
        );

        let exploration_results = exploratory_recall
            .recall(&exploration_cue, &store)
            .expect("exploratory recall failed");

        // Sort by confidence to emphasise ranking rationale.
        let mut sorted = exploration_results;
        sorted.sort_by(|a, b| b.confidence.raw().total_cmp(&a.confidence.raw()));
        display_results("Confidence-ranked", &sorted);
    }

    println!(
        "\nTip: enable trace_activation_flow and run \"cargo run -p engram-cli --example spreading_visualizer\" \n     to convert deterministic traces into GraphViz diagrams."
    );

    // Clean shutdown of the spreading engine
    // Try to unwrap the Arc and shutdown cleanly
    match Arc::try_unwrap(spreading_engine) {
        Ok(engine) => {
            engine.shutdown().expect("failed to shutdown spreading engine");
        }
        Err(_) => {
            eprintln!("Warning: Could not shutdown spreading engine (still has active references)");
        }
    }
}

#[cfg(feature = "hnsw_index")]
fn seed_memory_graph() -> MemoryStore {
    let store = MemoryStore::new(128).with_hnsw_index();

    let episodes = vec![
        (
            "doctor_harmon",
            "Dr. Harmon triages patient Lucy",
            0.90,
            0.88,
        ),
        (
            "nurse_lucy",
            "Nurse Lucy schedules the cardiology consult",
            0.88,
            0.86,
        ),
        (
            "cardiology_consult",
            "Cardiology reviews heart rate telemetry",
            0.65,
            0.74,
        ),
        (
            "follow_up_call",
            "Lucy confirms follow-up appointment over the phone",
            0.76,
            0.8,
        ),
        (
            "night_shift",
            "Night shift logs quiet vitals check",
            0.55,
            0.6,
        ),
    ];

    for (id, content, magnitude, confidence) in episodes {
        let episode = EpisodeBuilder::new()
            .id(id.to_string())
            .when(Utc::now())
            .what(content.to_string())
            .embedding(uniform_embedding(magnitude))
            .confidence(Confidence::from_raw(confidence))
            .build();

        store.store(episode);
    }

    store
}

#[cfg(feature = "hnsw_index")]
fn wire_spreading_edges(graph: &Arc<engram_core::activation::MemoryGraph>) {
    let connections = vec![
        ("doctor_harmon", "nurse_lucy", 0.92),
        ("nurse_lucy", "follow_up_call", 0.88),
        ("doctor_harmon", "cardiology_consult", 0.76),
        ("cardiology_consult", "follow_up_call", 0.72),
        ("doctor_harmon", "night_shift", 0.45),
    ];

    for (source, target, weight) in connections {
        ActivationGraphExt::add_edge(
            &**graph,
            source.to_string(),
            target.to_string(),
            weight,
            EdgeType::Excitatory,
        );
        ActivationGraphExt::add_edge(
            &**graph,
            target.to_string(),
            source.to_string(),
            weight * 0.9,
            EdgeType::Excitatory,
        );
    }

    let embeddings = vec![
        ("doctor_harmon", uniform_embedding(0.90)),
        ("nurse_lucy", uniform_embedding(0.88)),
        ("cardiology_consult", uniform_embedding(0.65)),
        ("follow_up_call", uniform_embedding(0.76)),
        ("night_shift", uniform_embedding(0.55)),
    ];

    for (node, embedding) in embeddings {
        ActivationGraphExt::set_embedding(&**graph, &node.to_string(), &embedding);
    }
}

#[cfg(feature = "hnsw_index")]
fn build_recall(
    seeder: &Arc<VectorActivationSeeder>,
    engine: &Arc<ParallelSpreadingEngine>,
    mode: RecallMode,
    budget: Duration,
) -> CognitiveRecall {
    let aggregator = Arc::new(ConfidenceAggregator::new(
        0.78,
        Confidence::from_raw(0.35),
        12,
    ));
    let cycle_detector = Arc::new(CycleDetector::new(HashMap::new()));

    CognitiveRecallBuilder::new()
        .vector_seeder(Arc::clone(seeder))
        .spreading_engine(Arc::clone(engine))
        .confidence_aggregator(aggregator)
        .cycle_detector(cycle_detector)
        .recall_mode(mode)
        .time_budget(budget)
        .build()
        .expect("failed to build cognitive recall")
}

#[cfg(feature = "hnsw_index")]
fn build_recall_with_config(
    seeder: &Arc<VectorActivationSeeder>,
    engine: &Arc<ParallelSpreadingEngine>,
    config: RecallConfig,
) -> CognitiveRecall {
    let aggregator = Arc::new(ConfidenceAggregator::new(
        0.72,
        Confidence::from_raw(0.25),
        16,
    ));
    let cycle_detector = Arc::new(CycleDetector::new(HashMap::new()));

    CognitiveRecallBuilder::new()
        .vector_seeder(Arc::clone(seeder))
        .spreading_engine(Arc::clone(engine))
        .confidence_aggregator(aggregator)
        .cycle_detector(cycle_detector)
        .config(config)
        .build()
        .expect("failed to build configured recall")
}

#[cfg(feature = "hnsw_index")]
fn display_results(title: &str, results: &[engram_core::activation::RankedMemory]) {
    println!("{title}:");
    if results.is_empty() {
        println!("  (no results)\n");
        return;
    }

    for memory in results.iter().take(5) {
        println!(
            "  - {id} | activation={activation:.2} | confidence={confidence:.2} | rank={rank:.2}",
            id = memory.episode.id,
            activation = memory.activation,
            confidence = memory.confidence.raw(),
            rank = memory.rank_score,
        );
    }

    println!("  total results: {}\n", results.len());
}

#[cfg(feature = "hnsw_index")]
fn uniform_embedding(value: f32) -> [f32; 768] {
    let mut embedding = [value; 768];
    embedding[0] = value;
    embedding[1] = value * 0.95;
    embedding[2] = value * 0.9;
    embedding
}
