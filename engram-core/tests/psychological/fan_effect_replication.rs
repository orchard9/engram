use std::collections::HashMap;

use engram_core::Confidence;
use engram_core::cognitive::interference::FanEffectDetector;
use engram_core::memory::Memory;
use engram_core::memory_graph::UnifiedMemoryGraph;

use super::statistical_analysis::{linear_regression_slope, pearson_correlation};
use super::test_datasets::{embedding_for_label, fan_effect_stimuli};

pub(crate) fn run_anderson_1974_person_location_paradigm() {
    let graph = UnifiedMemoryGraph::concurrent();
    let detector = FanEffectDetector::default();
    let mut location_cache = HashMap::new();

    let mut predicted = Vec::new();
    let mut empirical = Vec::new();
    let mut fan_counts = Vec::new();

    for stimulus in fan_effect_stimuli() {
        let person_embedding = embedding_for_label(stimulus.person);
        let person_id = graph
            .store_memory(Memory::new(
                stimulus.person.to_string(),
                person_embedding,
                Confidence::HIGH,
            ))
            .expect("person node stored");

        for &location in stimulus.locations {
            let entry = location_cache.entry(location).or_insert_with(|| {
                graph
                    .store_memory(Memory::new(
                        location.to_string(),
                        embedding_for_label(location),
                        Confidence::HIGH,
                    ))
                    .expect("location stored")
            });
            graph.add_edge(person_id, *entry, 1.0).expect("edge stored");
        }

        let fan_effect = detector.detect_fan_effect(&person_id, &graph);
        assert_eq!(fan_effect.fan, stimulus.fan);
        assert!(
            (fan_effect.retrieval_time_ms - stimulus.empirical_rt_ms).abs() < 25.0,
            "Fan {} prediction ({:.1}ms) diverged from empirical {:.1}ms",
            stimulus.fan,
            fan_effect.retrieval_time_ms,
            stimulus.empirical_rt_ms
        );

        predicted.push(fan_effect.retrieval_time_ms);
        empirical.push(stimulus.empirical_rt_ms);
        fan_counts.push(stimulus.fan as f32);
    }

    let correlation = pearson_correlation(&predicted, &empirical);
    assert!(
        correlation > 0.8,
        "Fan effect correlation must exceed 0.8, got {:.3}",
        correlation
    );

    let slope = linear_regression_slope(&fan_counts, &predicted);
    assert!(
        (slope - 70.0).abs() < 10.0,
        "Expected ~70ms slope, observed {:.1}ms",
        slope
    );
}
