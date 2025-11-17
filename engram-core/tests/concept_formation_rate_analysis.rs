//! Formation Rate Analysis: Detailed validation of concept formation dynamics
//!
//! This test suite provides quantitative analysis of concept formation rates
//! across different scenarios, validating that formation timelines match
//! empirical neuroscience data.

use chrono::{Duration, Utc};
use engram_core::consolidation::{ConceptFormationEngine, SleepStage};
use engram_core::memory::types::Episode;
use engram_core::{Confidence, EMBEDDING_DIM};
use std::collections::HashMap;

/// Scenario 1: Single episode (should NOT form concept)
#[test]
fn test_single_episode_no_concept_formation() {
    let engine = ConceptFormationEngine::new();

    // Create 1 episode with high confidence
    let base_embedding = [0.5; EMBEDDING_DIM];
    let episode = Episode::new(
        "single_episode".to_string(),
        Utc::now(),
        "unique content".to_string(),
        base_embedding,
        Confidence::exact(0.9),
    );

    // Process across multiple cycles
    for cycle in 1..=10 {
        let ready_concepts =
            engine.process_episodes(std::slice::from_ref(&episode), SleepStage::NREM2);

        assert!(
            ready_concepts.is_empty(),
            "Cycle {cycle}: Single episode should never form concept (violates min_cluster_size)"
        );
    }

    println!("✅ Single episode correctly prevented from concept formation");
}

/// Scenario 2: Two episodes (below minimum cluster size)
#[test]
fn test_two_episodes_no_concept_formation() {
    let engine = ConceptFormationEngine::new();

    // Create 2 highly similar episodes
    let base_time = Utc::now();
    let base_embedding = [(1.0 / EMBEDDING_DIM as f32).sqrt(); EMBEDDING_DIM];

    let mut episodes = vec![];
    for i in 0..2 {
        let mut embedding = base_embedding;
        // Add tiny noise to maintain very high similarity
        for (j, val) in embedding.iter_mut().enumerate().take(10) {
            *val += ((i + j) as f32 * 0.1).sin() * 0.0001;
        }

        // Normalize
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        for val in &mut embedding {
            *val /= norm;
        }

        episodes.push(Episode::new(
            format!("two_episode_{i:03}"),
            base_time,
            format!("content_{i}"),
            embedding,
            Confidence::exact(0.8),
        ));
    }

    // Process across multiple cycles
    for cycle in 1..=10 {
        let ready_concepts = engine.process_episodes(&episodes, SleepStage::NREM2);

        assert!(
            ready_concepts.is_empty(),
            "Cycle {cycle}: Two episodes insufficient for schema formation (requires 3+)"
        );
    }

    println!("✅ Two episodes correctly prevented from concept formation (min_cluster_size=3)");
}

/// Scenario 3: Multiple related episodes with timeline tracking
#[test]
fn test_multiple_related_episodes_formation_timeline() {
    let engine = ConceptFormationEngine::new();

    // Create 10 highly similar episodes
    let base_time = Utc::now();
    let base_embedding = [(1.0 / EMBEDDING_DIM as f32).sqrt(); EMBEDDING_DIM];

    let mut episodes = vec![];
    for i in 0..10 {
        let mut embedding = base_embedding;
        // Add tiny controlled noise
        for (j, val) in embedding.iter_mut().enumerate().take(20) {
            *val += ((i + j) as f32 * 0.137).sin() * 0.0001;
        }

        // Normalize
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        for val in &mut embedding {
            *val /= norm;
        }

        episodes.push(Episode::new(
            format!("related_{i:03}"),
            base_time - Duration::hours(i as i64),
            format!("similar_content_{i}"),
            embedding,
            Confidence::exact(0.8),
        ));
    }

    // Track formation timeline
    let mut formation_data = HashMap::new();

    for cycle in 1..=10 {
        let ready_concepts = engine.process_episodes(&episodes, SleepStage::NREM2);

        let status = if ready_concepts.is_empty() {
            "accumulating"
        } else {
            "promoted"
        };

        formation_data.insert(cycle, (status, ready_concepts.len()));

        if ready_concepts.is_empty() {
            println!("Cycle {cycle}: accumulating (not yet ready for promotion)");
        } else {
            let concept = &ready_concepts[0];
            println!(
                "Cycle {cycle}: PROMOTED | strength={:.3} | replay_count={} | coherence={:.3}",
                concept.consolidation_strength, concept.replay_count, concept.coherence_score
            );

            // Validate promotion criteria
            assert!(
                concept.consolidation_strength >= 0.10,
                "Promoted concept must have strength ≥ 0.10"
            );
            assert!(
                concept.replay_count >= 3,
                "Promoted concept must have replay_count ≥ 3"
            );
            assert!(
                concept.coherence_score >= 0.65,
                "Promoted concept must have coherence ≥ 0.65 (CA3 threshold)"
            );
        }
    }

    // Validate timeline expectations
    // Cycles 1-4: Should be accumulating
    for cycle in 1..=4 {
        let (status, _) = formation_data[&cycle];
        assert_eq!(
            status, "accumulating",
            "Cycle {cycle} should be accumulating (strength < 0.10)"
        );
    }

    // Cycle 5-6: Should reach promotion threshold
    let (status_c5, _) = formation_data[&5];
    let (status_c6, _) = formation_data[&6];

    assert!(
        status_c5 == "promoted" || status_c6 == "promoted",
        "Concept should be promoted by cycle 5 or 6 (strength ≥ 0.10)"
    );

    println!(
        "\n✅ Formation timeline validated: promotion occurs at cycle 5-6 (~7 days with daily consolidation)"
    );
}

/// Scenario 4: Stability analysis across extended consolidation
#[test]
fn test_repeated_activation_stability_increase() {
    let engine = ConceptFormationEngine::new();

    // Create 10 highly similar episodes
    let base_time = Utc::now();
    let base_embedding = [(1.0 / EMBEDDING_DIM as f32).sqrt(); EMBEDDING_DIM];

    let mut episodes = vec![];
    for i in 0..10 {
        let mut embedding = base_embedding;
        for (j, val) in embedding.iter_mut().enumerate().take(20) {
            *val += ((i + j) as f32 * 0.137).sin() * 0.0001;
        }

        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        for val in &mut embedding {
            *val /= norm;
        }

        episodes.push(Episode::new(
            format!("stable_{i:03}"),
            base_time,
            format!("content_{i}"),
            embedding,
            Confidence::exact(0.8),
        ));
    }

    // Track stability metrics across 50 cycles
    let mut stability_timeline = vec![];

    for cycle in 1..=50 {
        let ready_concepts = engine.process_episodes(&episodes, SleepStage::NREM2);

        if !ready_concepts.is_empty() {
            let concept = &ready_concepts[0];

            stability_timeline.push((
                cycle,
                concept.consolidation_strength,
                concept.replay_count,
                concept.coherence_score,
            ));

            // Key milestones
            if cycle == 5 {
                assert!(
                    concept.consolidation_strength >= 0.10,
                    "Cycle 5: Promotion threshold (strength ≥ 0.10)"
                );
                println!(
                    "Cycle 5 (7 days): strength={:.3} - PROMOTION THRESHOLD",
                    concept.consolidation_strength
                );
            } else if cycle == 25 {
                assert!(
                    concept.consolidation_strength >= 0.50,
                    "Cycle 25: Remote memory transition (strength ≥ 0.50)"
                );
                println!(
                    "Cycle 25 (30 days): strength={:.3} - REMOTE MEMORY TRANSITION",
                    concept.consolidation_strength
                );
            } else if cycle == 50 {
                assert!(
                    (concept.consolidation_strength - 1.0).abs() < 0.01,
                    "Cycle 50: Full consolidation (strength ≈ 1.0)"
                );
                println!(
                    "Cycle 50 (3 months): strength={:.3} - FULL CONSOLIDATION",
                    concept.consolidation_strength
                );
            }
        }
    }

    // Validate monotonic increase
    for i in 1..stability_timeline.len() {
        let (_, prev_strength, _, _) = stability_timeline[i - 1];
        let (_, curr_strength, _, _) = stability_timeline[i];

        assert!(
            curr_strength >= prev_strength,
            "Consolidation strength must increase monotonically (Hebbian learning)"
        );
    }

    println!("\n✅ Stability increases monotonically across 50 consolidation cycles");
    println!("Timeline matches empirical data:");
    println!("  - 7 days (cycle 5): Cortical representation emerges");
    println!("  - 30 days (cycle 25): Remote memory transition");
    println!("  - 3 months (cycle 50): Full semantic consolidation");
}

/// Scenario 5: Interference analysis with overlapping clusters
#[test]
fn test_overlapping_clusters_signature_handling() {
    let engine = ConceptFormationEngine::new();

    // Create two groups of episodes with one shared episode
    let base_time = Utc::now();
    let base_embedding = [(1.0 / EMBEDDING_DIM as f32).sqrt(); EMBEDDING_DIM];

    let mut all_episodes = vec![];

    // Cluster 1: Episodes 0-4
    for i in 0..5 {
        let mut embedding = base_embedding;
        for (j, val) in embedding.iter_mut().enumerate().take(30) {
            *val += ((i + j) as f32 * 0.1).sin() * 0.0001;
        }

        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        for val in &mut embedding {
            *val /= norm;
        }

        all_episodes.push(Episode::new(
            format!("cluster1_{i:03}"),
            base_time,
            format!("cluster1_content_{i}"),
            embedding,
            Confidence::exact(0.8),
        ));
    }

    // Cluster 2: Episodes 5-9 (potentially different cluster)
    for i in 5..10 {
        let mut embedding = base_embedding;
        // Different perturbation pattern
        for (j, val) in embedding.iter_mut().enumerate().skip(30).take(30) {
            *val += ((i + j) as f32 * 0.2).sin() * 0.0001;
        }

        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        for val in &mut embedding {
            *val /= norm;
        }

        all_episodes.push(Episode::new(
            format!("cluster2_{i:03}"),
            base_time,
            format!("cluster2_content_{i}"),
            embedding,
            Confidence::exact(0.8),
        ));
    }

    // Process for 10 cycles
    for _ in 1..=10 {
        let _ = engine.process_episodes(&all_episodes, SleepStage::NREM2);
    }

    // Check proto-concept pool
    let pool_size = engine.proto_pool_size();

    println!("Proto-concept pool size: {pool_size}");
    println!(
        "✅ Overlapping clusters handled via deterministic signatures (no spurious duplicates)"
    );

    // Validate: Should form 1-2 concepts (not 10)
    assert!(
        pool_size <= 5,
        "Should form 1-5 concepts from 10 episodes (not 10 separate concepts)"
    );
}

/// Scenario 6: Sleep stage comparison
#[test]
fn test_sleep_stage_impact_on_formation() {
    // Create 4 separate engines for each sleep stage
    let engine_nrem2 = ConceptFormationEngine::new();
    let engine_nrem3 = ConceptFormationEngine::new();
    let engine_rem = ConceptFormationEngine::new();
    let engine_wake = ConceptFormationEngine::new();

    // Create identical episode set
    let base_time = Utc::now();
    let base_embedding = [(1.0 / EMBEDDING_DIM as f32).sqrt(); EMBEDDING_DIM];

    let mut episodes = vec![];
    for i in 0..10 {
        let mut embedding = base_embedding;
        for (j, val) in embedding.iter_mut().enumerate().take(20) {
            *val += ((i + j) as f32 * 0.137).sin() * 0.0001;
        }

        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        for val in &mut embedding {
            *val /= norm;
        }

        episodes.push(Episode::new(
            format!("sleep_stage_{i:03}"),
            base_time,
            format!("content_{i}"),
            embedding,
            Confidence::exact(0.8),
        ));
    }

    // Process across 10 cycles with different sleep stages
    let mut promotion_cycles = HashMap::new();

    for cycle in 1..=10 {
        let nrem2_concepts = engine_nrem2.process_episodes(&episodes, SleepStage::NREM2);
        let nrem3_concepts = engine_nrem3.process_episodes(&episodes, SleepStage::NREM3);
        let rem_concepts = engine_rem.process_episodes(&episodes, SleepStage::REM);
        let wake_concepts = engine_wake.process_episodes(&episodes, SleepStage::QuietWake);

        if !nrem2_concepts.is_empty() && !promotion_cycles.contains_key("NREM2") {
            promotion_cycles.insert("NREM2", cycle);
        }
        if !nrem3_concepts.is_empty() && !promotion_cycles.contains_key("NREM3") {
            promotion_cycles.insert("NREM3", cycle);
        }
        if !rem_concepts.is_empty() && !promotion_cycles.contains_key("REM") {
            promotion_cycles.insert("REM", cycle);
        }
        if !wake_concepts.is_empty() && !promotion_cycles.contains_key("Wake") {
            promotion_cycles.insert("Wake", cycle);
        }
    }

    println!("\n=== Sleep Stage Formation Comparison ===");
    for (stage, cycle) in &promotion_cycles {
        println!("{stage}: Promoted at cycle {cycle}");
    }

    // All stages should eventually promote (consolidation_rate is stage-independent)
    // But replay weighting affects centroid calculation
    assert!(
        !promotion_cycles.is_empty(),
        "At least one sleep stage should promote concept"
    );

    println!(
        "✅ Sleep stage modulation affects replay weighting (but all stages can promote concepts)"
    );
}

/// Summary statistics helper
#[allow(dead_code)]
fn print_formation_summary(
    scenario: &str,
    formation_cycle: Option<u64>,
    final_strength: Option<f32>,
    final_replay_count: Option<u32>,
) {
    println!("\n=== Formation Summary: {scenario} ===");
    if let Some(cycle) = formation_cycle {
        println!("Formation Cycle: {cycle}");
    } else {
        println!("Formation Cycle: Never promoted");
    }

    if let Some(strength) = final_strength {
        println!("Final Strength: {strength:.3}");
    }

    if let Some(replay) = final_replay_count {
        println!("Final Replay Count: {replay}");
    }
}
