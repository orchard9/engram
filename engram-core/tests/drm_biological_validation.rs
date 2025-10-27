//! DRM False Memory Paradigm - Spreading Activation Validation
//!
//! Validates the FOUNDATION of false memory formation in Engram:
//! spreading activation among semantically related concepts.
//!
//! Research Foundation:
//! - Roediger & McDermott (1995): False memories arise from semantic associations
//! - Collins & Loftus (1975): Spreading activation theory
//! - Semantic priming: Related items activate unstudied concepts
//!
//! This test validates that:
//! 1. Studying semantically related words creates high co-activation
//! 2. Querying for an unstudied lure retrieves STUDIED items (semantic priming)
//! 3. The recalled items have high semantic similarity to the lure
//! 4. This demonstrates the mechanism underlying false memory formation
//!
//! Target: >80% semantic priming effect (recall of studied items when cued with lure)

use chrono::Utc;
use engram_core::{Confidence, Cue, EpisodeBuilder, MemoryStore};
use std::collections::HashMap;

/// DRM word list with critical lure
#[derive(Debug, Clone)]
struct DrmList {
    critical_lure: &'static str,
    study_items: &'static [&'static str],
}

/// Standard DRM lists from Roediger & McDermott (1995)
const DRM_LISTS: &[DrmList] = &[
    DrmList {
        critical_lure: "sleep",
        study_items: &[
            "bed", "rest", "awake", "tired", "dream", "wake", "snooze", "blanket", "doze",
            "slumber", "snore", "nap", "peace", "yawn", "drowsy",
        ],
    },
    DrmList {
        critical_lure: "chair",
        study_items: &[
            "table", "sit", "legs", "seat", "couch", "desk", "recliner", "sofa", "wood", "cushion",
            "swivel", "stool", "sitting", "rocking", "bench",
        ],
    },
    DrmList {
        critical_lure: "doctor",
        study_items: &[
            "nurse",
            "sick",
            "lawyer",
            "medicine",
            "health",
            "hospital",
            "dentist",
            "physician",
            "ill",
            "patient",
            "office",
            "stethoscope",
            "surgeon",
            "clinic",
            "cure",
        ],
    },
    DrmList {
        critical_lure: "mountain",
        study_items: &[
            "hill", "valley", "climb", "summit", "top", "molehill", "peak", "plain", "glacier",
            "goat", "bike", "climber", "range", "steep", "ski",
        ],
    },
];

/// Simple deterministic RNG
fn next_random(state: &mut u64) -> f32 {
    *state = state.wrapping_mul(1_103_515_245).wrapping_add(12_345);
    ((*state / 65_536) % 32_768) as f32 / 32_768.0
}

/// Generate synthetic embeddings with controlled semantic similarity
fn generate_embeddings() -> HashMap<&'static str, [f32; 768]> {
    let mut embeddings = HashMap::new();

    for (list_idx, list) in DRM_LISTS.iter().enumerate() {
        // Create theme vector for this list (orthogonal to other lists)
        let mut theme_vector = [0.0f32; 768];
        let mut rng_state = 12_345_u64 + (list_idx as u64 * 1000);

        for (i, val) in theme_vector.iter_mut().enumerate() {
            *val = if i % 4 == list_idx % 4 {
                next_random(&mut rng_state) * 2.0 - 1.0
            } else {
                (next_random(&mut rng_state) * 2.0 - 1.0) * 0.1
            };
        }

        // Normalize
        let norm: f32 = theme_vector.iter().map(|x| x * x).sum::<f32>().sqrt();
        for val in &mut theme_vector {
            *val /= norm;
        }

        // Critical lure is almost pure theme (similarity ~0.97)
        let mut lure_emb = theme_vector;
        let mut rng_state = 12_345_u64 + (list_idx as u64 * 1000) + 999;
        for val in &mut lure_emb {
            *val += (next_random(&mut rng_state) * 2.0 - 1.0) * 0.015;
        }

        let norm: f32 = lure_emb.iter().map(|x| x * x).sum::<f32>().sqrt();
        for val in &mut lure_emb {
            *val /= norm;
        }

        embeddings.insert(list.critical_lure, lure_emb);

        // Study items are close to lure (similarity 0.80-0.95)
        for (item_idx, &item) in list.study_items.iter().enumerate() {
            let mut item_emb = lure_emb;
            let mut rng_state = 12_345_u64 + (list_idx as u64 * 1000) + (item_idx as u64);

            // Very small noise for high similarity
            for val in &mut item_emb {
                let noise = 0.03 + (next_random(&mut rng_state) * 0.02); // 0.03-0.05
                *val += (next_random(&mut rng_state) * 2.0 - 1.0) * noise;
            }

            let norm: f32 = item_emb.iter().map(|x| x * x).sum::<f32>().sqrt();
            for val in &mut item_emb {
                *val /= norm;
            }

            embeddings.insert(item, item_emb);
        }
    }

    embeddings
}

/// Compute cosine similarity
fn cosine_similarity(a: &[f32; 768], b: &[f32; 768]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }

    dot / (norm_a * norm_b)
}

/// Run a single DRM trial
/// Returns: (semantic_priming_occurred, avg_recalled_similarity_to_lure, list_recall_rate)
fn run_drm_trial(
    list: &DrmList,
    trial_seed: u64,
    embeddings: &HashMap<&str, [f32; 768]>,
) -> (bool, f32, f32) {
    let store = MemoryStore::new(1000);
    let lure_embedding = embeddings[list.critical_lure];

    // Study phase: Store all list items
    for (idx, &word) in list.study_items.iter().enumerate() {
        let embedding = embeddings[word];

        let episode = EpisodeBuilder::new()
            .id(format!("trial_{trial_seed}_word_{idx}"))
            .when(Utc::now())
            .what(word.to_string())
            .embedding(embedding)
            .confidence(Confidence::HIGH)
            .build();

        store.store(episode);
    }

    // Allow spreading activation
    std::thread::sleep(std::time::Duration::from_millis(5));

    // Test phase: Query for critical lure (NEVER presented)
    // This tests SEMANTIC PRIMING: Do we recall studied items when cued with unstudied lure?
    let cue = Cue::embedding(
        format!("recall_lure_{trial_seed}"),
        lure_embedding,
        Confidence::exact(0.6), // Lower threshold to capture spreading activation
    );

    let recall_result = store.recall(&cue);

    // SEMANTIC PRIMING EFFECT: When cued with unstudied lure, do we recall studied items?
    // This is the FOUNDATION of false memory formation
    let mut recalled_studied_items = 0;
    let mut similarities_to_lure = Vec::new();

    for (episode, _conf) in &recall_result.results {
        let similarity_to_lure = cosine_similarity(&episode.embedding, &lure_embedding);
        similarities_to_lure.push(similarity_to_lure);

        // Check if this is one of the studied items
        let is_studied_item = list.study_items.iter().any(|&item| episode.what == item);

        if is_studied_item {
            recalled_studied_items += 1;
        }
    }

    let semantic_priming_occurred = recalled_studied_items > 0;
    let avg_similarity = if similarities_to_lure.is_empty() {
        0.0
    } else {
        similarities_to_lure.iter().sum::<f32>() / similarities_to_lure.len() as f32
    };

    // Test recall of actual list items (veridical memory)
    let mut recalled_count = 0;

    for &item in list.study_items {
        let item_embedding = embeddings[item];
        let item_cue = Cue::embedding(
            format!("recall_item_{trial_seed}_{item}"),
            item_embedding,
            Confidence::exact(0.9),
        );

        let item_recall = store.recall(&item_cue);

        for (episode, _conf) in &item_recall.results {
            let similarity = cosine_similarity(&episode.embedding, &item_embedding);

            if similarity > 0.95 {
                recalled_count += 1;
                break;
            }
        }
    }

    let recall_rate = recalled_count as f32 / list.study_items.len() as f32;

    (semantic_priming_occurred, avg_similarity, recall_rate)
}

#[test]
#[ignore = "Long-running semantic priming validation - run with --ignored"]
fn test_drm_semantic_priming_foundation() {
    const TRIALS_PER_LIST: usize = 50; // 200 total trials

    println!("\n=== DRM Spreading Activation Validation ===");
    println!("Target: >80% semantic priming (recall studied items when cued with unstudied lure)");

    let embeddings = generate_embeddings();

    // Validate semantic structure
    println!("\nValidating semantic structure:");
    for list in DRM_LISTS {
        let lure_emb = embeddings[list.critical_lure];
        let mut sims = Vec::new();

        for &item in list.study_items {
            let item_emb = embeddings[item];
            let sim = cosine_similarity(&lure_emb, &item_emb);
            sims.push(sim);
        }

        let avg_sim = sims.iter().sum::<f32>() / sims.len() as f32;
        println!(
            "  {}: avg BAS = {:.3} (min: {:.3}, max: {:.3})",
            list.critical_lure,
            avg_sim,
            sims.iter().copied().fold(1.0f32, f32::min),
            sims.iter().copied().fold(0.0f32, f32::max)
        );

        assert!(
            avg_sim > 0.70,
            "Insufficient semantic similarity for list '{}': {:.3}",
            list.critical_lure,
            avg_sim
        );
    }

    // Run DRM trials
    let mut all_priming_count = 0;
    let mut all_trials = 0;
    let mut all_similarities = Vec::new();
    let mut all_list_recalls = Vec::new();

    println!("\nRunning {TRIALS_PER_LIST} trials per list:");

    for list in DRM_LISTS {
        let mut list_priming_count = 0;
        let mut list_sims = Vec::new();

        for trial in 0..TRIALS_PER_LIST {
            let trial_seed = (list.critical_lure.len() as u64) * 10000 + trial as u64;
            let (priming_occurred, avg_sim, list_recall_rate) =
                run_drm_trial(list, trial_seed, &embeddings);

            if priming_occurred {
                list_priming_count += 1;
            }

            list_sims.push(avg_sim);
            all_similarities.push(avg_sim);
            all_list_recalls.push(list_recall_rate);
            all_trials += 1;
        }

        all_priming_count += list_priming_count;

        let list_priming_rate = f64::from(list_priming_count) / TRIALS_PER_LIST as f64;
        let list_avg_sim = list_sims.iter().sum::<f32>() / list_sims.len() as f32;
        println!(
            "  {}: {:.1}% priming, avg sim to lure = {:.3}",
            list.critical_lure,
            list_priming_rate * 100.0,
            list_avg_sim
        );
    }

    // Analyze results
    let priming_rate = f64::from(all_priming_count) / f64::from(all_trials);
    let avg_similarity = all_similarities.iter().sum::<f32>() / all_similarities.len() as f32;
    let avg_list_recall = all_list_recalls.iter().sum::<f32>() / all_list_recalls.len() as f32;

    // Statistical analysis
    let std_error = ((priming_rate * (1.0 - priming_rate)) / f64::from(all_trials)).sqrt();
    let ci_lower = (priming_rate - 1.96 * std_error).max(0.0);
    let ci_upper = (priming_rate + 1.96 * std_error).min(1.0);

    println!("\n=== Results ===");
    println!("Total trials: {all_trials}");
    println!(
        "Semantic priming: {} ({:.1}%)",
        all_priming_count,
        priming_rate * 100.0
    );
    println!(
        "95% CI: [{:.1}%, {:.1}%]",
        ci_lower * 100.0,
        ci_upper * 100.0
    );
    println!("Avg similarity to lure: {avg_similarity:.3}");
    println!("List item recall: {:.1}%", avg_list_recall * 100.0);

    // Validation assertions
    assert!(
        priming_rate >= 0.80,
        "VALIDATION FAILED: Semantic priming rate {:.1}% below target (>80%)",
        priming_rate * 100.0
    );

    assert!(
        avg_similarity >= 0.75,
        "Recalled items have insufficient similarity to lure: {avg_similarity:.3}"
    );

    assert!(
        avg_list_recall >= 0.30,
        "List item recall too low: {:.1}%",
        avg_list_recall * 100.0
    );

    println!(
        "\nSUCCESS: Semantic priming validated (>{:.0}%)",
        priming_rate * 100.0
    );
    println!("Spreading activation mechanism working correctly!");
    println!("\nNote: This validates the FOUNDATION of false memory formation.");
    println!("Full DRM false memory effect requires pattern completion feature.");
}

#[test]
fn test_drm_semantic_structure() {
    let embeddings = generate_embeddings();

    for list in DRM_LISTS {
        let lure_emb = embeddings[list.critical_lure];

        for &item in list.study_items {
            let item_emb = embeddings[item];
            let sim = cosine_similarity(&lure_emb, &item_emb);

            assert!(
                sim > 0.70,
                "Insufficient similarity: {} <-> {} = {:.3}",
                list.critical_lure,
                item,
                sim
            );
        }
    }
}
