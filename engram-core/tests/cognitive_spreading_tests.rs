#![allow(missing_docs)]
mod support;

use engram_core::activation::test_support::{deterministic_config, run_spreading};
use engram_core::activation::{ActivationGraphExt, EdgeType, MemoryGraph};
use engram_core::decay::HippocampalDecayFunction;
use std::sync::Arc;
use std::time::Duration;

#[cfg(feature = "hnsw_index")]
use chrono::Utc;
#[cfg(feature = "hnsw_index")]
use engram_core::activation::{
    ConfidenceAggregator,
    cycle_detector::CycleDetector,
    parallel::ParallelSpreadingEngine,
    recall::{CognitiveRecall, RecallConfig, RecallMode},
    seeding::VectorActivationSeeder,
    similarity_config::SimilarityConfig,
};
#[cfg(feature = "hnsw_index")]
use engram_core::{Confidence, Cue, EpisodeBuilder, MemoryStore};
#[cfg(feature = "hnsw_index")]
use std::collections::HashMap;
#[cfg(feature = "hnsw_index")]
use std::sync::atomic::Ordering;

fn build_semantic_priming_graph() -> (Arc<MemoryGraph>, Vec<(String, f32)>) {
    let graph = Arc::new(engram_core::activation::create_activation_graph());

    ActivationGraphExt::add_edge(
        &*graph,
        "doctor".to_string(),
        "nurse".to_string(),
        0.9,
        EdgeType::Excitatory,
    );
    ActivationGraphExt::add_edge(
        &*graph,
        "doctor".to_string(),
        "writer".to_string(),
        0.4,
        EdgeType::Excitatory,
    );
    ActivationGraphExt::add_edge(
        &*graph,
        "mechanic".to_string(),
        "wrench".to_string(),
        0.9,
        EdgeType::Excitatory,
    );

    (graph, vec![("doctor".to_string(), 1.0)])
}

fn activation_for(results: &engram_core::activation::SpreadingResults, node: &str) -> f32 {
    results
        .activations
        .iter()
        .find(|activation| activation.memory_id == node)
        .map_or(0.0, |activation| {
            activation
                .activation_level
                .load(std::sync::atomic::Ordering::Relaxed)
        })
}

#[test]
fn hippocampal_decay_matches_exponential_baseline() {
    let decay = HippocampalDecayFunction::default();
    let tau = decay.tau_base();
    let tolerance = 0.03;

    for hours in [0.5_f32, 1.0, 6.0, 24.0] {
        let elapsed = Duration::from_secs_f32(hours * 3600.0);
        let observed = decay.compute_retention(elapsed);
        let expected = (-hours / tau).exp();
        assert!(
            (observed - expected).abs() <= tolerance,
            "Retention deviated beyond tolerance for {hours}h: observed={observed:.4}, expected={expected:.4}"
        );
    }
}

#[test]
fn semantic_priming_boosts_related_concepts() {
    let (graph, seeds) = build_semantic_priming_graph();
    let mut config = deterministic_config(123);
    config.max_depth = 2;

    let run = run_spreading(&graph, &seeds, config).expect("spreading should succeed");
    let nurse_activation = activation_for(&run.results, "nurse");
    let writer_activation = activation_for(&run.results, "writer");

    assert!(nurse_activation > writer_activation);
    assert!(nurse_activation > 0.0);
}

#[test]
fn fan_effect_reduces_activation_per_association() {
    use support::graph_builders::fan;

    let high_fan = fan("concept_high", 6);
    let low_fan = fan("concept_low", 2);

    let mut config = deterministic_config(456);
    config.max_depth = 2;

    let high_run = run_spreading(high_fan.graph(), &high_fan.seeds, config.clone())
        .expect("high fan spreading");
    let low_run =
        run_spreading(low_fan.graph(), &low_fan.seeds, config).expect("low fan spreading");

    let avg_high = high_run
        .results
        .activations
        .iter()
        .filter(|node| node.memory_id.starts_with("concept_high_spoke"))
        .map(|node| {
            node.activation_level
                .load(std::sync::atomic::Ordering::Relaxed)
        })
        .sum::<f32>()
        / 6.0;

    let avg_low = low_run
        .results
        .activations
        .iter()
        .filter(|node| node.memory_id.starts_with("concept_low_spoke"))
        .map(|node| {
            node.activation_level
                .load(std::sync::atomic::Ordering::Relaxed)
        })
        .sum::<f32>()
        / 2.0;

    assert!(
        avg_low > avg_high,
        "Fan effect should dilute activation per association"
    );
}

#[cfg(feature = "hnsw_index")]
#[test]
#[ignore = "TODO: CognitiveRecall integration with ActivationGraph spreading needs architectural work. The spreading activates nodes in the graph, but mapping those back to episodes in MemoryStore requires the recall system to bridge between graph node IDs and episode IDs. This works for simple spreading tests but needs refinement for full MemoryStore+HNSW+Graph integration."]
fn cognitive_recall_respects_semantic_priming_and_fan_effect() {
    let store = MemoryStore::new(128).with_hnsw_index();

    let mut episodes: Vec<(String, f32, &'static str)> = vec![
        ("doctor".to_string(), 0.92, "Doctor schema"),
        ("nurse".to_string(), 0.91, "Nurse schema"),
        ("writer".to_string(), 0.55, "Writer schema"),
        ("mechanic".to_string(), 0.65, "Mechanic concept"),
        ("wrench".to_string(), 0.63, "Tool association"),
        ("concept_high".to_string(), 0.80, "High fan anchor"),
        ("concept_low".to_string(), 0.75, "Low fan anchor"),
    ];

    for idx in 0..6 {
        episodes.push((
            format!("concept_high_spoke_{idx}"),
            0.6 + (idx as f32) * 0.01,
            "High fan spoke",
        ));
    }
    for idx in 0..2 {
        episodes.push((
            format!("concept_low_spoke_{idx}"),
            0.58 + (idx as f32) * 0.02,
            "Low fan spoke",
        ));
    }

    for (id, value, description) in &episodes {
        let episode = EpisodeBuilder::new()
            .id(id.clone())
            .when(Utc::now())
            .what((*description).to_string())
            .embedding([*value; 768])
            .confidence(Confidence::HIGH)
            .build();
        let _ = store.store(episode);
    }

    let index = store
        .hnsw_index()
        .expect("HNSW index should be available for recall");

    let graph = Arc::new(engram_core::activation::create_activation_graph());
    ActivationGraphExt::add_edge(
        &*graph,
        "doctor".to_string(),
        "nurse".to_string(),
        0.9,
        EdgeType::Excitatory,
    );
    ActivationGraphExt::add_edge(
        &*graph,
        "doctor".to_string(),
        "writer".to_string(),
        0.4,
        EdgeType::Excitatory,
    );
    for idx in 0..6 {
        ActivationGraphExt::add_edge(
            &*graph,
            "concept_high".to_string(),
            format!("concept_high_spoke_{idx}"),
            1.0 / 6.0,
            EdgeType::Excitatory,
        );
    }
    for idx in 0..2 {
        ActivationGraphExt::add_edge(
            &*graph,
            "concept_low".to_string(),
            format!("concept_low_spoke_{idx}"),
            0.5,
            EdgeType::Excitatory,
        );
    }

    let mut config = deterministic_config(2024);
    config.num_threads = 1;
    config.max_depth = 3;
    config.threshold = 0.0;

    let spreading_engine = Arc::new(
        ParallelSpreadingEngine::new(config, Arc::clone(&graph)).expect("engine should construct"),
    );

    let seeder = Arc::new(VectorActivationSeeder::with_default_resolver(
        index,
        SimilarityConfig::default(),
    ));
    let aggregator = Arc::new(ConfidenceAggregator::new(0.8, Confidence::LOW, 16));
    let cycle_detector = Arc::new(CycleDetector::new(HashMap::new()));

    let recall_config = RecallConfig {
        recall_mode: RecallMode::Spreading,
        time_budget: Duration::from_millis(50),
        min_confidence: 0.0,
        max_results: 32,
        enable_recency_boost: false,
        recency_boost_factor: 1.0,
        recency_window: Duration::from_secs(1),
    };

    let recall = CognitiveRecall::new(
        seeder,
        spreading_engine,
        aggregator,
        cycle_detector,
        recall_config,
    );

    let doctor_cue = Cue::embedding("cue_doctor".to_string(), [0.92; 768], Confidence::MEDIUM);
    let doctor_results = recall
        .recall(&doctor_cue, &store)
        .expect("doctor recall should succeed");

    let nurse_pos = doctor_results
        .iter()
        .position(|entry| entry.episode.id == "nurse")
        .expect("nurse should appear in recall results");
    let writer_pos = doctor_results
        .iter()
        .position(|entry| entry.episode.id == "writer")
        .expect("writer should appear in recall results");
    assert!(
        nurse_pos < writer_pos,
        "Semantic priming should elevate nurse above writer"
    );

    let high_cue = Cue::embedding("cue_high".to_string(), [0.80; 768], Confidence::MEDIUM);
    let high_results = recall
        .recall(&high_cue, &store)
        .expect("high fan recall should succeed");
    let low_cue = Cue::embedding("cue_low".to_string(), [0.75; 768], Confidence::MEDIUM);
    let low_results = recall
        .recall(&low_cue, &store)
        .expect("low fan recall should succeed");

    let high_spokes: Vec<f32> = high_results
        .iter()
        .filter(|entry| entry.episode.id.starts_with("concept_high_spoke"))
        .map(|entry| entry.activation)
        .collect();
    assert!(
        !high_spokes.is_empty(),
        "High fan recall should surface spokes"
    );
    let avg_high = high_spokes.iter().sum::<f32>() / high_spokes.len() as f32;

    let low_spokes: Vec<f32> = low_results
        .iter()
        .filter(|entry| entry.episode.id.starts_with("concept_low_spoke"))
        .map(|entry| entry.activation)
        .collect();
    assert!(
        !low_spokes.is_empty(),
        "Low fan recall should surface spokes"
    );
    let avg_low = low_spokes.iter().sum::<f32>() / low_spokes.len() as f32;

    assert!(
        avg_low > avg_high,
        "Fan effect should reduce activation per spoke for high fan concept"
    );

    let metrics = recall.metrics();
    assert_eq!(
        metrics.total_recalls.load(Ordering::Relaxed),
        3,
        "Expected three recall invocations"
    );
    assert_eq!(
        metrics.spreading_mode_count.load(Ordering::Relaxed),
        3,
        "All recalls should run in spreading mode"
    );
    assert_eq!(
        metrics.fallbacks_total.load(Ordering::Relaxed),
        0,
        "Spreading recalls should not fallback"
    );

    store.shutdown_hnsw_worker();
}
