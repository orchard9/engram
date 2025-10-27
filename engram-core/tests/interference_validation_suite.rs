//! Comprehensive Interference Validation Suite (Milestone 13, Task 010)
//!
//! Validates all three interference types against empirical psychology research:
//! 1. Proactive Interference (Underwood 1957): 20-30% reduction with 5+ prior lists
//! 2. Retroactive Interference (McGeoch 1942): 15-25% reduction with interpolated learning
//! 3. Fan Effect (Anderson 1974): 50-150ms RT increase per association
//!
//! Statistical power: n=200 samples per test for 90% power at α=0.05

#![allow(clippy::cast_lossless)]
#![allow(clippy::uninlined_format_args)]
#![allow(clippy::items_after_statements)]

use chrono::{DateTime, Duration, Utc};
use engram_core::Confidence;
use engram_core::cognitive::interference::{
    FanEffectDetector, ProactiveInterferenceDetector, RetroactiveInterferenceDetector,
};
use engram_core::memory::Episode;
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use std::collections::HashMap;

// ============================================================================
// STIMULUS MATERIALS MODULE (inline)
// ============================================================================

const EMBEDDING_DIM: usize = 768;

/// Paired associate (cue-target pair)
#[derive(Debug, Clone)]
struct WordPair {
    cue: String,
    target: String,
    embedding: [f32; EMBEDDING_DIM],
    category: String,
}

/// Generate paired associate lists with controlled semantic overlap
fn generate_paired_associate_lists(
    num_lists: usize,
    pairs_per_list: usize,
    seed: u64,
    semantic_overlap: f32,
) -> Vec<Vec<WordPair>> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);

    let categories = [
        "fruit",
        "animal",
        "color",
        "vehicle",
        "furniture",
        "tool",
        "clothing",
        "sport",
    ];

    let targets_per_category = [
        vec![
            "apple", "banana", "orange", "grape", "cherry", "lemon", "mango", "peach",
        ],
        vec![
            "dog", "cat", "horse", "lion", "tiger", "elephant", "giraffe", "zebra",
        ],
        vec![
            "red", "blue", "green", "yellow", "purple", "orange", "pink", "brown",
        ],
        vec![
            "car",
            "truck",
            "bicycle",
            "train",
            "airplane",
            "boat",
            "motorcycle",
            "bus",
        ],
        vec![
            "chair", "table", "sofa", "desk", "bed", "lamp", "shelf", "cabinet",
        ],
        vec![
            "hammer",
            "screwdriver",
            "wrench",
            "drill",
            "saw",
            "pliers",
            "axe",
            "chisel",
        ],
        vec![
            "shirt", "pants", "dress", "jacket", "shoes", "hat", "socks", "gloves",
        ],
        vec![
            "soccer",
            "basketball",
            "tennis",
            "baseball",
            "golf",
            "hockey",
            "swimming",
            "running",
        ],
    ];

    let mut lists = Vec::new();

    for list_idx in 0..num_lists {
        let mut pairs = Vec::new();

        for pair_idx in 0..pairs_per_list {
            let cat_idx = pair_idx % categories.len();
            let category = categories[cat_idx];
            let cue = category.to_string();

            let use_same_category = rng.gen_range(0.0..1.0f32) < semantic_overlap;

            let target = if use_same_category && list_idx > 0 {
                let item_idx = (list_idx + pair_idx * 3) % targets_per_category[cat_idx].len();
                targets_per_category[cat_idx][item_idx].to_string()
            } else {
                let item_idx = pair_idx % targets_per_category[cat_idx].len();
                targets_per_category[cat_idx][item_idx].to_string()
            };

            let embedding = generate_embedding(
                &cue,
                &target,
                seed + (list_idx as u64) * 1000 + (pair_idx as u64),
            );

            pairs.push(WordPair {
                cue,
                target,
                embedding,
                category: category.to_string(),
            });
        }

        lists.push(pairs);
    }

    lists
}

/// Fan fact (person-location pair)
#[derive(Debug, Clone)]
struct FanFact {
    #[allow(dead_code)]
    person: String,
    #[allow(dead_code)]
    location: String,
    #[allow(dead_code)]
    embedding: [f32; EMBEDDING_DIM],
    person_fan: usize,
    location_fan: usize,
}

impl FanFact {
    const fn total_fan(&self) -> usize {
        self.person_fan + self.location_fan
    }
}

/// Generate fan effect facts
fn generate_fan_facts(count: usize, seed: u64) -> Vec<FanFact> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);

    let people = [
        "doctor",
        "lawyer",
        "teacher",
        "fireman",
        "artist",
        "scientist",
        "engineer",
        "musician",
    ];
    let locations = [
        "park", "church", "bank", "store", "office", "library", "hospital", "school",
    ];

    let facts_per_fan = count / 4;
    let mut facts = Vec::new();

    for i in 0..facts_per_fan {
        facts.push(FanFact {
            person: people[i % people.len()].to_string(),
            location: locations[i % locations.len()].to_string(),
            embedding: generate_embedding(
                people[i % people.len()],
                locations[i % locations.len()],
                seed + i as u64,
            ),
            person_fan: 1,
            location_fan: 1,
        });
    }

    for i in 0..facts_per_fan {
        let person_idx = i / 2;
        facts.push(FanFact {
            person: people[person_idx % people.len()].to_string(),
            location: locations[(facts_per_fan + i) % locations.len()].to_string(),
            embedding: generate_embedding(
                people[person_idx % people.len()],
                locations[(facts_per_fan + i) % locations.len()],
                seed + (facts_per_fan + i) as u64,
            ),
            person_fan: 2,
            location_fan: 1,
        });
    }

    for i in 0..facts_per_fan {
        let person_idx = i / 3;
        let loc_idx = i / 2;
        facts.push(FanFact {
            person: people[person_idx % people.len()].to_string(),
            location: locations[loc_idx % locations.len()].to_string(),
            embedding: generate_embedding(
                people[person_idx % people.len()],
                locations[loc_idx % locations.len()],
                seed + (facts_per_fan * 2 + i) as u64,
            ),
            person_fan: 2,
            location_fan: 2,
        });
    }

    for i in 0..facts_per_fan {
        let person_idx = i / 4;
        let loc_idx = i / 3;
        facts.push(FanFact {
            person: people[person_idx % people.len()].to_string(),
            location: locations[loc_idx % locations.len()].to_string(),
            embedding: generate_embedding(
                people[person_idx % people.len()],
                locations[loc_idx % locations.len()],
                seed + (facts_per_fan * 3 + i) as u64,
            ),
            person_fan: 3,
            location_fan: 2,
        });
    }

    facts.shuffle(&mut rng);
    facts
}

fn generate_embedding(a: &str, b: &str, seed: u64) -> [f32; EMBEDDING_DIM] {
    // Generate base embedding from category (a) only, so same categories have similar embeddings
    // Then add small perturbation from target (b) and seed
    let category_hash = a.bytes().fold(42u64, |acc, byte| {
        acc.wrapping_mul(31).wrapping_add(byte as u64)
    });

    let mut rng_base = ChaCha8Rng::seed_from_u64(category_hash);
    let mut embedding = [0.0f32; EMBEDDING_DIM];

    // Base embedding from category (90% of magnitude)
    for val in &mut embedding {
        *val = rng_base.gen_range(-0.9..0.9);
    }

    // Add small perturbation from target + seed (10% of magnitude)
    let combined_seed = b.bytes().fold(seed, |acc, byte| {
        acc.wrapping_mul(31).wrapping_add(byte as u64)
    });
    let mut rng_perturb = ChaCha8Rng::seed_from_u64(combined_seed);

    for val in &mut embedding {
        *val += rng_perturb.gen_range(-0.1..0.1);
    }

    // Normalize to unit vector
    let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for val in &mut embedding {
            *val /= norm;
        }
    }

    embedding
}

// ============================================================================
// STATISTICAL ANALYSIS MODULE (inline)
// ============================================================================

#[derive(Debug, Clone)]
struct RegressionResult {
    slope: f64,
    intercept: f64,
    r_squared: f64,
    p_value: f64,
}

fn linear_regression(x: &[f64], y: &[f64]) -> RegressionResult {
    let n = x.len() as f64;
    let mean_x = x.iter().sum::<f64>() / n;
    let mean_y = y.iter().sum::<f64>() / n;

    let numerator: f64 = x
        .iter()
        .zip(y)
        .map(|(xi, yi)| (xi - mean_x) * (yi - mean_y))
        .sum();
    let denominator: f64 = x.iter().map(|xi| (xi - mean_x).powi(2)).sum();
    let slope = numerator / denominator;
    let intercept = mean_y - slope * mean_x;

    let ss_tot: f64 = y.iter().map(|yi| (yi - mean_y).powi(2)).sum();
    let ss_res: f64 = x
        .iter()
        .zip(y)
        .map(|(xi, yi)| {
            let predicted = slope * xi + intercept;
            (yi - predicted).powi(2)
        })
        .sum();
    let r_squared = 1.0 - (ss_res / ss_tot);

    let f_stat = if ss_res > 0.0 {
        (r_squared / 1.0) / ((1.0 - r_squared) / (n - 2.0))
    } else {
        f64::INFINITY
    };

    let p_value = if f_stat > 20.0 {
        0.0001
    } else if f_stat > 10.0 {
        0.001
    } else if f_stat > 5.0 {
        0.01
    } else {
        0.05
    };

    RegressionResult {
        slope,
        intercept,
        r_squared,
        p_value,
    }
}

#[derive(Debug, Clone)]
struct TTestResult {
    t_statistic: f64,
    df: usize,
    p_value: f64,
    effect_size: f64,
}

fn independent_t_test(group1: &[f32], group2: &[f32]) -> TTestResult {
    let n1 = group1.len() as f64;
    let n2 = group2.len() as f64;
    let mean1 = group1.iter().sum::<f32>() as f64 / n1;
    let mean2 = group2.iter().sum::<f32>() as f64 / n2;

    let var1 = group1
        .iter()
        .map(|x| (*x as f64 - mean1).powi(2))
        .sum::<f64>()
        / (n1 - 1.0);
    let var2 = group2
        .iter()
        .map(|x| (*x as f64 - mean2).powi(2))
        .sum::<f64>()
        / (n2 - 1.0);
    let pooled_sd = ((var1 * (n1 - 1.0) + var2 * (n2 - 1.0)) / (n1 + n2 - 2.0)).sqrt();

    let t_statistic = (mean1 - mean2) / (pooled_sd * ((1.0 / n1) + (1.0 / n2)).sqrt());
    let df = (n1 + n2 - 2.0) as usize;
    let p_value = if t_statistic.abs() > 2.6 {
        0.01
    } else if t_statistic.abs() > 2.0 {
        0.05
    } else {
        0.1
    };
    let effect_size = (mean1 - mean2) / pooled_sd;

    TTestResult {
        t_statistic,
        df,
        p_value,
        effect_size,
    }
}

fn cohens_d(group1: &[f32], group2: &[f32]) -> f64 {
    let n1 = group1.len() as f64;
    let n2 = group2.len() as f64;
    let mean1 = group1.iter().sum::<f32>() as f64 / n1;
    let mean2 = group2.iter().sum::<f32>() as f64 / n2;

    let var1 = group1
        .iter()
        .map(|x| (*x as f64 - mean1).powi(2))
        .sum::<f64>()
        / (n1 - 1.0);
    let var2 = group2
        .iter()
        .map(|x| (*x as f64 - mean2).powi(2))
        .sum::<f64>()
        / (n2 - 1.0);
    let pooled_sd = ((var1 * (n1 - 1.0) + var2 * (n2 - 1.0)) / (n1 + n2 - 2.0)).sqrt();

    (mean1 - mean2) / pooled_sd
}

#[derive(Debug, Clone)]
struct ConfidenceInterval {
    #[allow(dead_code)]
    mean: f64,
    lower: f64,
    upper: f64,
}

fn compute_confidence_interval(data: &[f32], confidence_level: f64) -> ConfidenceInterval {
    let n = data.len() as f64;
    let mean = data.iter().sum::<f32>() as f64 / n;
    let variance = data.iter().map(|x| (*x as f64 - mean).powi(2)).sum::<f64>() / (n - 1.0);
    let se = (variance / n).sqrt();

    let z_critical = if confidence_level >= 0.99 {
        2.576
    } else if confidence_level >= 0.95 {
        1.96
    } else {
        1.645
    };
    let margin = z_critical * se;

    ConfidenceInterval {
        mean,
        lower: mean - margin,
        upper: mean + margin,
    }
}

// ============================================================================
// TEST 1: PROACTIVE INTERFERENCE (Underwood 1957)
// ============================================================================

#[test]
fn test_underwood_1957_proactive_interference_validation() {
    println!("\n=== Proactive Interference Validation (Underwood 1957) ===");

    const TRIALS_PER_CONDITION: usize = 33; // 33 × 6 conditions = 198 ≈ 200 samples
    let list_counts = [0, 1, 2, 5, 10, 20];

    let detector = ProactiveInterferenceDetector::default();

    let mut all_results = Vec::new();
    let mut accuracy_by_list_count: HashMap<usize, Vec<f32>> = HashMap::new();

    for &count in &list_counts {
        for trial in 0..TRIALS_PER_CONDITION {
            let seed = (count as u64) * 1000 + (trial as u64);
            let result = run_proactive_interference_trial(&detector, count, seed);

            all_results.push(result.clone());
            accuracy_by_list_count
                .entry(count)
                .or_default()
                .push(result.target_recall_accuracy);
        }
    }

    // Print summary statistics
    println!("\n--- Accuracy by Prior List Count ---");
    for &count in &list_counts {
        let accuracies = &accuracy_by_list_count[&count];
        let mean = accuracies.iter().sum::<f32>() / accuracies.len() as f32;
        let std_dev = compute_std_dev(accuracies);

        println!(
            "{} prior lists: {:.1}% ± {:.1}%",
            count,
            mean * 100.0,
            std_dev * 100.0
        );
    }

    // Statistical Analysis 1: Linear Regression
    println!("\n--- Linear Regression Analysis ---");
    let regression = linear_regression(
        &all_results
            .iter()
            .map(|r| r.prior_list_count as f64)
            .collect::<Vec<_>>(),
        &all_results
            .iter()
            .map(|r| r.target_recall_accuracy as f64)
            .collect::<Vec<_>>(),
    );

    println!(
        "Slope: {:.4} (accuracy reduction per prior list)",
        regression.slope
    );
    println!("Intercept: {:.4}", regression.intercept);
    println!("R² = {:.3}", regression.r_squared);
    println!("p < {:.4}", regression.p_value);

    // Validation 1: Negative slope (more interference → less accuracy)
    assert!(
        regression.slope < 0.0,
        "Slope should be negative (more prior lists → lower accuracy), got {}",
        regression.slope
    );

    // Validation 2: Strong relationship (R² > 0.60)
    assert!(
        regression.r_squared > 0.60,
        "R² = {:.3} too low (need >0.60 for strong linear relationship)",
        regression.r_squared
    );

    // Statistical Analysis 2: Underwood's Key Finding (5 prior lists)
    println!("\n--- Underwood (1957) Benchmark: 5 Prior Lists ---");

    let baseline = &accuracy_by_list_count[&0];
    let five_list = &accuracy_by_list_count[&5];

    let baseline_mean = baseline.iter().sum::<f32>() / baseline.len() as f32;
    let five_list_mean = five_list.iter().sum::<f32>() / five_list.len() as f32;

    let accuracy_reduction = (baseline_mean - five_list_mean) / baseline_mean;

    println!("Baseline (0 lists): {:.1}%", baseline_mean * 100.0);
    println!("With 5 lists: {:.1}%", five_list_mean * 100.0);
    println!("Reduction: {:.1}%", accuracy_reduction * 100.0);
    println!("Target range: 20-30% ± 10% = [10%, 40%]");

    // Confidence interval
    let ci = compute_confidence_interval(five_list, 0.95);
    println!(
        "95% CI for 5-list condition: [{:.1}%, {:.1}%]",
        ci.lower * 100.0,
        ci.upper * 100.0
    );

    // Effect size (Cohen's d)
    let effect_size = cohens_d(baseline, five_list);
    println!("Cohen's d (0 vs 5 lists): {:.3}", effect_size);

    // Validation 3: Accuracy reduction within acceptance range
    assert!(
        (0.10..=0.40).contains(&accuracy_reduction),
        "Proactive interference effect {:.1}% outside acceptance range [10%, 40%]",
        accuracy_reduction * 100.0
    );

    // Validation 4: Statistical significance
    let t_test = independent_t_test(baseline, five_list);
    println!("\nIndependent t-test (0 vs 5 lists):");
    println!("  t({}) = {:.3}", t_test.df, t_test.t_statistic);
    println!("  p = {:.4}", t_test.p_value);
    println!("  Cohen's d = {:.3}", t_test.effect_size);

    assert!(
        t_test.p_value < 0.05,
        "Effect not statistically significant: p = {:.4}",
        t_test.p_value
    );

    assert!(
        t_test.effect_size >= 0.5,
        "Effect size too small: d = {:.3} (expected ≥0.5 for medium effect)",
        t_test.effect_size
    );

    println!("\n✓ Proactive interference validation PASSED");
}

#[derive(Debug, Clone)]
struct ProactiveResult {
    prior_list_count: usize,
    target_recall_accuracy: f32,
    #[allow(dead_code)]
    mean_confidence: f32,
}

fn run_proactive_interference_trial(
    detector: &ProactiveInterferenceDetector,
    prior_list_count: usize,
    seed: u64,
) -> ProactiveResult {
    let now = Utc::now();

    // Generate semantically similar lists (60% overlap)
    let all_lists = generate_paired_associate_lists(prior_list_count + 1, 10, seed, 0.6);

    let prior_lists = &all_lists[0..prior_list_count];
    let target_list = &all_lists[prior_list_count];

    let mut prior_episodes = Vec::new();

    // Study prior lists
    for (list_idx, list) in prior_lists.iter().enumerate() {
        for pair in list {
            let when = now - Duration::hours(3) + Duration::minutes(list_idx as i64 * 5);
            let episode = pair_to_episode(pair, when, list_idx);
            prior_episodes.push(episode);
        }
    }

    // Study target list
    let target_episodes: Vec<Episode> = target_list
        .iter()
        .map(|pair| pair_to_episode(pair, now, 9999))
        .collect();

    // Test target list recall
    let mut correct = 0;
    let mut confidences = Vec::new();

    for target_ep in &target_episodes {
        // Detect interference for this target episode
        let interference = detector.detect_interference(target_ep, &prior_episodes);

        // Debug: Print interference magnitude for first trial of 5-list condition
        if prior_list_count == 5 && seed == 5000 && correct == 0 {
            println!(
                "  DEBUG: Interference magnitude = {:.3}, count = {}",
                interference.magnitude, interference.count
            );
        }

        // Simulate recall success probability based on interference
        // Base recall probability: 0.90 (90% baseline accuracy)
        // Reduce by interference magnitude
        let base_probability = 0.90;
        let recall_probability = base_probability * (1.0 - interference.magnitude);

        // Use deterministic threshold based on episode ID for reproducibility
        let threshold = target_ep
            .id
            .bytes()
            .fold(0u64, |acc, b| acc.wrapping_mul(31).wrapping_add(b as u64))
            as f32
            / u64::MAX as f32;

        if threshold < recall_probability {
            correct += 1;
            confidences.push(recall_probability);
        }
    }

    let accuracy = correct as f32 / target_episodes.len() as f32;
    let mean_conf = if confidences.is_empty() {
        0.0
    } else {
        confidences.iter().sum::<f32>() / confidences.len() as f32
    };

    ProactiveResult {
        prior_list_count,
        target_recall_accuracy: accuracy,
        mean_confidence: mean_conf,
    }
}

// ============================================================================
// TEST 2: RETROACTIVE INTERFERENCE (McGeoch 1942)
// ============================================================================

#[test]
fn test_mcgeoch_1942_retroactive_interference_validation() {
    println!("\n=== Retroactive Interference Validation (McGeoch 1942) ===");

    const TRIALS_PER_CONDITION: usize = 100; // 100 × 2 conditions = 200 samples

    let detector = RetroactiveInterferenceDetector::default();

    let mut control_accuracies = Vec::new();
    let mut experimental_accuracies = Vec::new();

    for trial in 0..TRIALS_PER_CONDITION {
        // Control condition (no interpolated learning)
        let control_result = run_retroactive_interference_trial(&detector, false, trial as u64);
        control_accuracies.push(control_result.list_a_recall_accuracy);

        // Experimental condition (with interpolated List B)
        let exp_result = run_retroactive_interference_trial(&detector, true, trial as u64 + 10000);
        experimental_accuracies.push(exp_result.list_a_recall_accuracy);
    }

    // Summary statistics
    println!("\n--- Accuracy by Condition ---");
    let mean_control = control_accuracies.iter().sum::<f32>() / control_accuracies.len() as f32;
    let mean_exp =
        experimental_accuracies.iter().sum::<f32>() / experimental_accuracies.len() as f32;

    let std_control = compute_std_dev(&control_accuracies);
    let std_exp = compute_std_dev(&experimental_accuracies);

    println!(
        "Control (no interpolation): {:.1}% ± {:.1}%",
        mean_control * 100.0,
        std_control * 100.0
    );
    println!(
        "Experimental (List B interpolated): {:.1}% ± {:.1}%",
        mean_exp * 100.0,
        std_exp * 100.0
    );

    let accuracy_reduction = (mean_control - mean_exp) / mean_control;
    println!("Reduction: {:.1}%", accuracy_reduction * 100.0);
    println!("Target range: 15-25% ± 10% = [5%, 35%]");

    // Confidence intervals
    let ci_control = compute_confidence_interval(&control_accuracies, 0.95);
    let ci_exp = compute_confidence_interval(&experimental_accuracies, 0.95);

    println!(
        "95% CI Control: [{:.1}%, {:.1}%]",
        ci_control.lower * 100.0,
        ci_control.upper * 100.0
    );
    println!(
        "95% CI Experimental: [{:.1}%, {:.1}%]",
        ci_exp.lower * 100.0,
        ci_exp.upper * 100.0
    );

    // Statistical test
    println!("\n--- Independent t-test ---");
    let t_test = independent_t_test(&control_accuracies, &experimental_accuracies);

    println!("  t({}) = {:.3}", t_test.df, t_test.t_statistic);
    println!("  p = {:.4}", t_test.p_value);
    println!("  Cohen's d = {:.3}", t_test.effect_size);

    // Validation 1: Accuracy reduction within acceptance range
    assert!(
        (0.05..=0.35).contains(&accuracy_reduction),
        "Retroactive interference {:.1}% outside [5%, 35%] range",
        accuracy_reduction * 100.0
    );

    // Validation 2: Statistical significance
    assert!(
        t_test.p_value < 0.05,
        "Effect not statistically significant: p = {:.4}",
        t_test.p_value
    );

    // Validation 3: Medium to large effect size
    assert!(
        t_test.effect_size >= 0.5,
        "Effect size too small: d = {:.3} (expected ≥0.5)",
        t_test.effect_size
    );

    println!("\n✓ Retroactive interference validation PASSED");
}

#[derive(Debug, Clone)]
struct RetroactiveResult {
    #[allow(dead_code)]
    condition: String,
    list_a_recall_accuracy: f32,
    #[allow(dead_code)]
    mean_confidence: f32,
}

fn run_retroactive_interference_trial(
    detector: &RetroactiveInterferenceDetector,
    is_experimental: bool,
    seed: u64,
) -> RetroactiveResult {
    // Phase 1: Learn List A at T=0
    let list_a_time = Utc::now();

    // Generate lists with 50% semantic overlap (same cues, different targets)
    let lists = generate_paired_associate_lists(2, 10, seed, 0.5);
    let list_a = &lists[0];
    let list_b = &lists[1];

    // Store List A
    let list_a_episodes: Vec<Episode> = list_a
        .iter()
        .map(|pair| pair_to_episode(pair, list_a_time, 0))
        .collect();

    // Phase 2: Experimental condition learns List B during retention interval
    let mut all_episodes = list_a_episodes.clone();

    if is_experimental {
        let interpolated_time = list_a_time + Duration::minutes(30);
        let interpolated_episodes: Vec<Episode> = list_b
            .iter()
            .map(|pair| pair_to_episode(pair, interpolated_time, 1))
            .collect();
        all_episodes.extend(interpolated_episodes);
    }

    // Phase 3: Test List A recall at T=60min
    let retrieval_time = list_a_time + Duration::minutes(60);

    // Test recall of List A
    let mut correct = 0;
    let mut confidences = Vec::new();

    for list_a_ep in &list_a_episodes {
        // Detect retroactive interference
        let interference = detector.detect_interference(list_a_ep, &all_episodes, retrieval_time);

        // Debug output for first experimental trial
        if is_experimental && seed == 10000 && correct == 0 {
            println!(
                "  DEBUG: Retroactive interference magnitude = {:.3}, count = {}",
                interference.magnitude, interference.count
            );
        }

        // Simulate recall success probability based on interference
        // Base recall probability: 0.90 (90% baseline accuracy)
        // Retroactive interference scales linearly per McGeoch (1942)
        let base_probability = 0.90;
        let recall_probability = base_probability * (1.0 - interference.magnitude);

        // Use deterministic threshold based on episode ID for reproducibility
        let threshold = list_a_ep
            .id
            .bytes()
            .fold(0u64, |acc, b| acc.wrapping_mul(31).wrapping_add(b as u64))
            as f32
            / u64::MAX as f32;

        if threshold < recall_probability {
            correct += 1;
            confidences.push(recall_probability);
        }
    }

    let accuracy = correct as f32 / list_a_episodes.len() as f32;
    let mean_conf = if confidences.is_empty() {
        0.0
    } else {
        confidences.iter().sum::<f32>() / confidences.len() as f32
    };

    RetroactiveResult {
        condition: if is_experimental {
            "experimental".to_string()
        } else {
            "control".to_string()
        },
        list_a_recall_accuracy: accuracy,
        mean_confidence: mean_conf,
    }
}

// ============================================================================
// TEST 3: FAN EFFECT (Anderson 1974)
// ============================================================================

#[test]
fn test_anderson_1974_fan_effect_validation() {
    println!("\n=== Fan Effect Validation (Anderson 1974) ===");

    const TRIALS: usize = 50; // 50 × 4 fan levels = 200 samples

    let detector = FanEffectDetector::default();

    let mut results_by_fan: HashMap<usize, Vec<f32>> = HashMap::new();

    for trial in 0..TRIALS {
        let seed = trial as u64;
        let trial_results = run_fan_effect_trial(&detector, seed);

        for result in trial_results {
            results_by_fan
                .entry(result.fan)
                .or_default()
                .push(result.retrieval_time_ms);
        }
    }

    // Summary statistics
    println!("\n--- Retrieval Time by Fan Level ---");
    let mut sorted_fans: Vec<usize> = results_by_fan.keys().copied().collect();
    sorted_fans.sort_unstable();

    for &fan in &sorted_fans {
        let rts = &results_by_fan[&fan];
        let mean = rts.iter().sum::<f32>() / rts.len() as f32;
        let std_dev = compute_std_dev(rts);

        println!("Fan {}: {:.1} ms ± {:.1} ms", fan, mean, std_dev);
    }

    // Collect all data points for regression
    let mut all_fan_values = Vec::new();
    let mut all_rt_values = Vec::new();

    for (&fan, rts) in &results_by_fan {
        for &rt in rts {
            all_fan_values.push(fan as f64);
            all_rt_values.push(rt as f64);
        }
    }

    // Linear regression: RT ~ Fan
    println!("\n--- Linear Regression Analysis ---");
    let regression = linear_regression(&all_fan_values, &all_rt_values);

    println!("Slope: {:.1} ms per association", regression.slope);
    println!("Intercept: {:.1} ms (baseline RT)", regression.intercept);
    println!("R² = {:.3}", regression.r_squared);
    println!("p < {:.4}", regression.p_value);
    println!("Target slope: 50-150 ms ± 25 ms = [25, 175] ms");

    // Validation 1: Positive slope (more associations → slower retrieval)
    assert!(
        regression.slope > 0.0,
        "Slope should be positive (higher fan → longer RT), got {}",
        regression.slope
    );

    // Validation 2: Slope within acceptance range [25, 175] ms
    assert!(
        (25.0..=175.0).contains(&regression.slope),
        "Fan effect slope {:.1} ms outside [25, 175] ms range",
        regression.slope
    );

    // Validation 3: Strong linear relationship (R² > 0.70)
    assert!(
        regression.r_squared > 0.70,
        "R² = {:.3} too low (need >0.70 for strong relationship)",
        regression.r_squared
    );

    // Validation 4: Statistical significance
    assert!(
        regression.p_value < 0.05,
        "Regression not statistically significant: p = {:.4}",
        regression.p_value
    );

    println!("\n✓ Fan effect validation PASSED");
}

#[derive(Debug, Clone)]
struct FanEffectResult {
    fan: usize,
    retrieval_time_ms: f32,
}

fn run_fan_effect_trial(detector: &FanEffectDetector, seed: u64) -> Vec<FanEffectResult> {
    // Generate fan facts with varying fan levels
    let facts = generate_fan_facts(80, seed); // 20 per fan level (1-4)

    let mut results = Vec::new();

    for fact in &facts {
        let fan = fact.total_fan();

        // Use detector to compute predicted RT
        let rt_ms = detector.compute_retrieval_time_ms(fan);

        results.push(FanEffectResult {
            fan,
            retrieval_time_ms: rt_ms,
        });
    }

    results
}

// ============================================================================
// TEST 4: COMPREHENSIVE INTEGRATION TEST
// ============================================================================

#[test]
fn test_comprehensive_interference_integration() {
    println!("\n=== Comprehensive Interference Integration Test ===");

    // Test that all three interference types can operate together
    let proactive_detector = ProactiveInterferenceDetector::default();
    let retroactive_detector = RetroactiveInterferenceDetector::default();
    let fan_detector = FanEffectDetector::default();

    // Scenario: Multiple lists with varying fan levels
    let now = Utc::now();
    let retrieval_time = now + Duration::hours(1);

    // Create episodes with high fan and temporal overlap
    let lists = generate_paired_associate_lists(3, 10, 42, 0.7);

    let mut all_episodes = Vec::new();

    for (list_idx, list) in lists.iter().enumerate() {
        let when = now + Duration::minutes(list_idx as i64 * 20);
        for pair in list {
            let episode = pair_to_episode(pair, when, list_idx);
            all_episodes.push(episode);
        }
    }

    // Test on final list (should have both proactive and retroactive interference)
    let target_list = &lists[2];
    let target_episodes: Vec<Episode> = target_list
        .iter()
        .map(|pair| pair_to_episode(pair, now + Duration::minutes(40), 2))
        .collect();

    let mut total_proactive = 0.0;
    let mut total_retroactive = 0.0;
    let mut total_fan_slowdown = 0.0;

    for target_ep in &target_episodes {
        // Proactive interference from prior lists
        let proactive = proactive_detector.detect_interference(target_ep, &all_episodes[0..20]);
        total_proactive += proactive.magnitude;

        // Retroactive interference (none expected - no interpolated learning after this list)
        let retroactive =
            retroactive_detector.detect_interference(target_ep, &all_episodes, retrieval_time);
        total_retroactive += retroactive.magnitude;

        // Fan effect (simulated with fan count)
        let fan = 2; // Assume fan=2 for this test
        let fan_result = fan_detector.compute_retrieval_time_ms(fan);
        total_fan_slowdown += fan_result - fan_detector.base_retrieval_time_ms();
    }

    let avg_proactive = total_proactive / target_episodes.len() as f32;
    let avg_retroactive = total_retroactive / target_episodes.len() as f32;
    let avg_fan_slowdown = total_fan_slowdown / target_episodes.len() as f32;

    println!(
        "Average proactive interference: {:.1}%",
        avg_proactive * 100.0
    );
    println!(
        "Average retroactive interference: {:.1}%",
        avg_retroactive * 100.0
    );
    println!("Average fan effect slowdown: {:.1} ms", avg_fan_slowdown);

    // Validation: All three types detected
    assert!(
        avg_proactive > 0.0,
        "Should detect proactive interference from prior lists"
    );
    // Retroactive may be 0 (no interpolated learning after target list)
    assert!(avg_fan_slowdown > 0.0, "Should detect fan effect slowdown");

    println!("\n✓ Comprehensive integration test PASSED");
}

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

fn pair_to_episode(pair: &WordPair, when: DateTime<Utc>, list_idx: usize) -> Episode {
    let mut metadata = HashMap::new();
    metadata.insert("list_idx".to_string(), list_idx.to_string());
    metadata.insert("cue".to_string(), pair.cue.clone());
    metadata.insert("target".to_string(), pair.target.clone());
    metadata.insert("category".to_string(), pair.category.clone());

    Episode {
        id: format!("list{}_cue_{}", list_idx, pair.cue),
        when,
        where_location: None,
        who: None,
        what: format!("{} → {}", pair.cue, pair.target),
        embedding: pair.embedding,
        embedding_provenance: None,
        encoding_confidence: Confidence::HIGH,
        vividness_confidence: Confidence::HIGH,
        reliability_confidence: Confidence::HIGH,
        last_recall: when,
        recall_count: 0,
        decay_rate: 0.05,
        decay_function: None,
        metadata,
    }
}

fn compute_std_dev(values: &[f32]) -> f32 {
    if values.is_empty() {
        return 0.0;
    }

    let mean = values.iter().sum::<f32>() / values.len() as f32;
    let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / values.len() as f32;
    variance.sqrt()
}
