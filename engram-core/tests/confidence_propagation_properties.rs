//! Property tests covering dual-memory confidence propagation.

use chrono::Utc;
use engram_core::confidence::dual_memory::DualMemoryConfidence;
use engram_core::query::confidence_calibration::CalibrationTracker;
use engram_core::{Confidence, EMBEDDING_DIM, Episode};
use proptest::prelude::*;

fn episode_with_conf(conf: f32) -> Episode {
    Episode::new(
        format!("episode_{conf}"),
        Utc::now(),
        "test".to_string(),
        [0.0; EMBEDDING_DIM],
        Confidence::exact(conf),
    )
}

fn high_quality_episode() -> Episode {
    let mut episode = episode_with_conf(0.85);
    episode.reliability_confidence = Confidence::exact(0.9);
    episode.vividness_confidence = Confidence::exact(0.8);
    episode
}

fn low_quality_episode() -> Episode {
    let mut episode = episode_with_conf(0.35);
    episode.reliability_confidence = Confidence::exact(0.25);
    episode.vividness_confidence = Confidence::exact(0.3);
    episode
}

fn medium_quality_episode() -> Episode {
    let mut episode = episode_with_conf(0.6);
    episode.reliability_confidence = Confidence::exact(0.55);
    episode.vividness_confidence = Confidence::exact(0.58);
    episode
}

proptest! {
    /// Confidence never increases when propagating through bindings.
    #[test]
    fn propagation_is_monotonic(source in 0.0f32..1.0, binding in 0.0f32..1.0) {
        let rules = DualMemoryConfidence::default();
        let source_conf = Confidence::exact(source);
        let propagated = rules.propagate_through_binding(source_conf, binding);
        prop_assert!(propagated.raw() <= source_conf.raw() + f32::EPSILON);
    }

    /// Propagation always produces valid probability bounds.
    #[test]
    fn propagation_preserves_bounds(source in 0.0f32..1.0, binding in 0.0f32..1.0) {
        let rules = DualMemoryConfidence::default();
        let source_conf = Confidence::exact(source);
        let propagated = rules.propagate_through_binding(source_conf, binding);
        prop_assert!(propagated.raw() >= 0.0);
        prop_assert!(propagated.raw() <= 1.0);
    }

    /// Blend confidence is always bounded by the maximum input plus the bonus multiplier.
    #[test]
    fn blend_confidence_bounded(episodic in 0.0f32..1.0, semantic in 0.0f32..1.0) {
        let rules = DualMemoryConfidence::default();
        let ep_conf = Confidence::exact(episodic);
        let sem_conf = Confidence::exact(semantic);
        let blended = rules.blend_confidence(ep_conf, sem_conf, 0.7, 0.3);
        let max_input = episodic.max(semantic);
        prop_assert!(blended.raw() <= max_input * rules.blend_bonus_multiplier() + f32::EPSILON);
    }

    /// Blend bonus only applies when both sources exceed the convergent-evidence threshold.
    #[test]
    fn blend_bonus_threshold(episodic in 0.0f32..0.5, semantic in 0.0f32..1.0) {
        let rules = DualMemoryConfidence::default();
        let ep_conf = Confidence::exact(episodic);
        let sem_conf = Confidence::exact(semantic);
        let blended = rules.blend_confidence(ep_conf, sem_conf, 0.7, 0.3);
        let weighted = (episodic * 0.7 + semantic * 0.3) / 1.0;
        prop_assert!((blended.raw() - weighted).abs() < 1e-6);
    }

    /// Multi-hop propagation decays exponentially.
    #[test]
    fn multi_hop_exponential_decay(initial in 0.5f32..1.0, hops in 1u32..10) {
        let rules = DualMemoryConfidence::default();
        let mut current = Confidence::exact(initial);
        for _ in 0..hops {
            current = rules.propagate_through_binding(current, 0.8);
        }
        let expected = initial * (0.8 * rules.binding_decay()).powi(hops as i32);
        prop_assert!((current.raw() - expected).abs() < 0.02);
    }
}

#[test]
fn cycle_convergence_reaches_negligible_mass() {
    let rules = DualMemoryConfidence::default();
    let mut current = Confidence::exact(0.9);
    for _ in 0..100 {
        current = rules.propagate_through_binding(current, 0.8);
    }
    assert!(
        current.raw() < 0.001,
        "cycle retained too much mass: {}",
        current.raw()
    );
}

#[test]
fn concept_confidence_respects_penalty() {
    let rules = DualMemoryConfidence::default();
    let episodes = vec![high_quality_episode(), high_quality_episode()];
    let concept = rules.concept_confidence(&episodes, 0.9);
    let max_episode = episodes
        .iter()
        .map(|ep| ep.reliability_confidence.raw())
        .fold(0.0, f32::max);
    assert!(concept.raw() <= max_episode * rules.concept_penalty() * 0.9 + 1e-6);
}

#[test]
fn episodic_vs_semantic_blending_behaves() {
    let rules = DualMemoryConfidence::default();
    let episodic = Confidence::exact(0.8);
    let semantic = Confidence::exact(0.6);
    let blended = rules.blend_confidence(episodic, semantic, 0.7, 0.3);
    assert!(blended.raw() >= semantic.raw());
    assert!(blended.raw() <= episodic.raw() * rules.blend_bonus_multiplier());
}

#[test]
fn dual_memory_calibration_tracks_accuracy() {
    let rules = DualMemoryConfidence::default();
    let mut tracker = CalibrationTracker::new(10);

    let scenarios = vec![
        high_quality_episode(),
        medium_quality_episode(),
        low_quality_episode(),
    ];
    for episode in scenarios {
        let predicted = rules.calculate_episode_confidence(&episode);
        let sample_count = 20;
        let mut success_budget = (predicted.raw() * sample_count as f32)
            .round()
            .clamp(0.0, sample_count as f32) as usize;
        for _ in 0..sample_count {
            let was_correct = if success_budget > 0 {
                success_budget -= 1;
                true
            } else {
                false
            };
            tracker.record_sample(predicted, was_correct);
        }
    }

    let metrics = tracker.compute_metrics();
    assert!(metrics.confidence_accuracy_correlation.unwrap() > 0.7);
    assert!(metrics.expected_calibration_error < 0.1);
}
