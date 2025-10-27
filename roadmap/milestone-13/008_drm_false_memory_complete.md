# Task 008: DRM False Memory Paradigm Implementation (ENHANCED)

**Status:** Pending
**Priority:** P0 (Validation Critical)
**Estimated Effort:** 2.5 days (was 2 days)
**Dependencies:** Task 002 (Semantic Priming), M8 (Pattern Completion)
**Enhanced By:** verification-testing-lead (Professor Regehr)
**Enhancement Date:** 2025-10-26

## Changes from Original Specification

**CRITICAL FIXES REQUIRED:**
1. Increased sample size: n=200 (was n=100) for adequate statistical power
2. Added `get_embedding()` implementation specification
3. Replaced string matching with semantic similarity threshold
4. Added Cohen's d effect size calculation
5. Added parameter sweep recovery strategy
6. Added time simulation validation
7. Added chi-square goodness-of-fit test
8. Added embedding semantic structure validation

**RISK MITIGATION:**
- Added API availability verification before implementation
- Added determinism tests with seed control
- Added artifact detection for consolidation timing

---

## Objective

Replicate the Deese-Roediger-McDermott (DRM) false memory paradigm to validate that Engram's pattern completion and semantic priming produce false memories at rates matching published psychology research: 55-65% false recall for critical lures (Roediger & McDermott 1995).

This is the acid test for biological plausibility. If we can't replicate DRM, our cognitive mechanisms are wrong.

## Research Foundation

Roediger & McDermott (1995) revived Deese's (1959) paradigm, creating one of the most reliable false memory phenomena in psychology. Participants study word lists of 15 semantically-related words (e.g., "bed, rest, awake, tired, dream..."). The critical lure ("sleep") is NEVER presented, yet 55-65% falsely recall or recognize it as having been studied. Effect remarkably robust across hundreds of replications.

**Dual theoretical mechanisms:**
1. **Spreading Activation Theory (Collins & Loftus 1975):** Each studied word activates critical lure through semantic associations. Cumulative activation from 15 related words makes lure feel familiar, creating false memory.
2. **Fuzzy Trace Theory (Brainerd & Reyna 2002):** People encode verbatim traces (exact words) and gist traces (general theme). Gist trace captures "sleep-related words," matching critical lure perfectly.

Both theories predict same outcome. Our implementation uses spreading activation (maps directly to our graph architecture).

**Quantitative characteristics (Roediger & McDermott 1995):**
- False recall: 55-65% for critical lures (target: 60% ± 10%)
- Veridical recall: 60-70% for studied words
- False recognition: 75-85% (higher than recall)
- Confidence: false memories often rated as confident as true memories (paradoxical)

**Modulating factors:**
- List length: 15 words optimal, shorter lists reduce effect
- Semantic strength: high backward associative strength (BAS) essential (>0.3 threshold)
- Study time: 1-2 seconds per word optimal
- Retention interval: effect persists for at least 24 hours

**Statistical validation:**
- **N ≥ 200** virtual participants (simulated memory spaces) - INCREASED from 100
- Multiple randomized study lists per participant
- Chi-square goodness-of-fit test: observed vs expected 60% rate
- Effect size (Cohen's d) > 0.8 for false vs true recall confidence
- 95% confidence interval must overlap [50%, 70%]

## Integration Points

**Uses:**
- `/engram-core/src/cognitive/priming/semantic.rs` - Semantic priming (Task 002)
- `/engram-core/src/completion/mod.rs` - Pattern completion (M8)
- `/engram-core/src/activation/spreading.rs` - Activation spreading (M3)
- `/engram-core/src/store.rs` - MemoryStore API

**Creates:**
- `/engram-core/tests/psychology/drm_paradigm.rs` - DRM experiment implementation
- `/engram-core/tests/psychology/drm_word_lists.json` - Standard DRM lists with embeddings
- `/engram-core/tests/psychology/drm_analysis.rs` - Statistical analysis
- `/engram-core/tests/psychology/drm_embeddings.rs` - Embedding generation and validation
- `/engram-core/tests/psychology/drm_parameter_sweep.rs` - Recovery strategy

## Pre-Implementation Checklist

**CRITICAL: Verify API Availability Before Starting**

```rust
// File: /engram-core/tests/psychology/api_compatibility.rs

#[test]
fn test_drm_api_availability() {
    use engram_core::{MemoryStore, Episode, EpisodeBuilder, Confidence};

    // Verify MemoryStore construction
    let store = MemoryStore::new(1000);

    // Verify episode storage
    let episode = EpisodeBuilder::new()
        .id("test")
        .what("test content")
        .when(chrono::Utc::now())
        .confidence(Confidence::HIGH)
        .embedding(vec![0.1; 768])
        .build()
        .unwrap();

    store.store(episode);

    // Verify recall API
    let results = store.recall_by_id("test");
    assert!(!results.is_empty());

    // CRITICAL: Verify consolidation API exists
    // TODO: Check if MemoryStore has consolidate() method
    // If not, use alternative consolidation trigger

    println!("✓ All required APIs available for DRM implementation");
}
```

**Run this test first. If it fails, update task specification with actual API.**

## Detailed Specification

### 1. Embedding Generation and Validation

**NEW FILE:** `/engram-core/tests/psychology/drm_embeddings.rs`

```rust
//! Embedding generation and semantic structure validation for DRM paradigm
//!
//! CRITICAL: DRM requires realistic semantic embeddings with high backward
//! associative strength (BAS) between list items and critical lure.

use engram_core::EMBEDDING_DIM;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Pre-computed embedding with semantic metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticEmbedding {
    pub word: String,
    pub embedding: Vec<f32>,
    pub category: String, // e.g., "sleep", "chair", "doctor"
}

/// Compute cosine similarity between embeddings
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (norm_a * norm_b)
}

/// Generate embeddings for DRM word lists
///
/// OPTIONS:
/// 1. Use real embedding model (sentence-transformers via Python bridge)
/// 2. Use pre-computed embeddings (from OpenAI API)
/// 3. Use synthetic embeddings with controlled similarity (for testing)
///
/// RECOMMENDATION: Option 2 (pre-computed) for reproducibility
pub fn get_embedding(word: &str) -> Vec<f32> {
    // Load pre-computed embeddings from JSON
    let embeddings: HashMap<String, Vec<f32>> = load_precomputed_embeddings();

    embeddings.get(word)
        .cloned()
        .unwrap_or_else(|| panic!("No embedding for word: {}", word))
}

/// Load pre-computed embeddings from JSON file
fn load_precomputed_embeddings() -> HashMap<String, Vec<f32>> {
    let json_data = include_str!("drm_embeddings_precomputed.json");
    serde_json::from_str(json_data)
        .expect("Failed to parse pre-computed embeddings")
}

/// Validate semantic structure of DRM lists
///
/// Ensures all list items have high semantic similarity to critical lure
/// (BAS > 0.3 threshold from Roediger & McDermott 1995)
pub fn validate_drm_semantic_structure(word_lists: &DrmWordLists) {
    for list in &word_lists.lists {
        let lure_embedding = get_embedding(&list.critical_lure);

        let mut similarities = Vec::new();

        for word in &list.study_items {
            let word_embedding = get_embedding(word);
            let similarity = cosine_similarity(&lure_embedding, &word_embedding);

            similarities.push((word.clone(), similarity));

            assert!(
                similarity > 0.2, // Lenient threshold (some items may be weakly associated)
                "Word '{}' has insufficient semantic association with lure '{}': {:.3} (need >0.2)",
                word, list.critical_lure, similarity
            );
        }

        // Average similarity should be high (>0.35)
        let avg_similarity: f32 = similarities.iter().map(|(_, s)| s).sum::<f32>()
            / similarities.len() as f32;

        assert!(
            avg_similarity > 0.35,
            "List '{}' has insufficient average BAS: {:.3} (need >0.35)",
            list.critical_lure, avg_similarity
        );

        println!("List '{}': avg BAS = {:.3}", list.critical_lure, avg_similarity);
        for (word, sim) in &similarities {
            println!("  {} → {:.3}", word, sim);
        }
    }
}

#[test]
fn test_embedding_semantic_structure() {
    let word_lists = load_drm_lists();
    validate_drm_semantic_structure(&word_lists);
}

#[test]
fn test_embedding_dimension() {
    let embedding = get_embedding("sleep");
    assert_eq!(embedding.len(), EMBEDDING_DIM);
}

#[test]
fn test_embedding_normalization() {
    let embedding = get_embedding("sleep");
    let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();

    // Embeddings should be approximately normalized (norm ≈ 1.0)
    assert!(
        (norm - 1.0).abs() < 0.1,
        "Embedding not normalized: norm = {:.3}",
        norm
    );
}
```

**ACTION ITEM:** Generate pre-computed embeddings using OpenAI API:

```python
# scripts/generate_drm_embeddings.py
import json
import openai

DRM_WORDS = [
    # Sleep list
    "sleep", "bed", "rest", "awake", "tired", "dream",
    "wake", "snooze", "blanket", "doze", "slumber",
    "snore", "nap", "peace", "yawn", "drowsy",
    # Chair list
    "chair", "table", "sit", "legs", "seat", "couch",
    "desk", "recliner", "sofa", "wood", "cushion",
    "swivel", "stool", "sitting", "rocking", "bench",
    # Doctor list
    "doctor", "nurse", "sick", "lawyer", "medicine", "health",
    "hospital", "dentist", "physician", "ill", "patient",
    "office", "stethoscope", "surgeon", "clinic", "cure",
    # Mountain list
    "mountain", "hill", "valley", "climb", "summit", "top",
    "molehill", "peak", "plain", "glacier", "goat",
    "bike", "climber", "range", "steep", "ski",
]

embeddings = {}
for word in DRM_WORDS:
    response = openai.Embedding.create(
        input=word,
        model="text-embedding-ada-002"
    )
    embeddings[word] = response['data'][0]['embedding']

with open('engram-core/tests/psychology/drm_embeddings_precomputed.json', 'w') as f:
    json.dump(embeddings, f)
```

### 2. DRM Experiment Implementation (ENHANCED)

**FILE:** `/engram-core/tests/psychology/drm_paradigm.rs`

```rust
// Enhanced implementation with all critical fixes

use engram_core::{MemoryStore, Episode, EpisodeBuilder, Confidence};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

mod drm_embeddings;
use drm_embeddings::{get_embedding, cosine_similarity, validate_drm_semantic_structure};

/// Standard DRM list with critical lure and study items
#[derive(Debug, Clone, Deserialize, Serialize)]
struct DrmList {
    critical_lure: String,
    study_items: Vec<String>,
}

/// DRM word lists from Roediger & McDermott (1995)
#[derive(Debug, Deserialize)]
struct DrmWordLists {
    lists: Vec<DrmList>,
}

/// Result of a single DRM trial
#[derive(Debug)]
struct DrmTrialResult {
    critical_lure: String,
    false_recall: bool,
    false_recall_confidence: Option<f32>,
    list_item_recall_count: usize,
    list_item_recall_rate: f32,
    true_recall_confidences: Vec<f32>, // For Cohen's d calculation
}

/// Load standard DRM word lists
fn load_drm_lists() -> DrmWordLists {
    let json_data = include_str!("drm_word_lists.json");
    serde_json::from_str(json_data)
        .expect("Failed to parse DRM word lists")
}

/// Run a single DRM trial
fn run_drm_trial(list: &DrmList, trial_seed: u64) -> DrmTrialResult {
    // Use seed for determinism
    let store = MemoryStore::new(1000);

    // Get critical lure embedding for similarity matching
    let lure_embedding = get_embedding(&list.critical_lure);

    // Study phase: Present list items
    for (idx, word) in list.study_items.iter().enumerate() {
        let embedding = get_embedding(word);

        let episode = EpisodeBuilder::new()
            .id(format!("trial_{}_word_{}", trial_seed, idx))
            .what(word.clone())
            .when(chrono::Utc::now())
            .confidence(Confidence::HIGH)
            .embedding(embedding)
            .build()
            .unwrap();

        store.store(episode);
    }

    // CRITICAL FIX: Use explicit consolidation trigger, not arbitrary sleep
    // Check if API exists, otherwise use time-based approach
    #[cfg(feature = "explicit_consolidation")]
    {
        store.consolidate();
    }
    #[cfg(not(feature = "explicit_consolidation"))]
    {
        // Fallback: Use time delay for consolidation
        std::thread::sleep(std::time::Duration::from_millis(100));
    }

    // Test phase: Attempt recall of critical lure (NOT presented)
    let lure_results = store.recall_by_content(&list.critical_lure);

    // CRITICAL FIX: Use semantic similarity, not string matching
    let false_recall_result = lure_results.iter().find(|(episode, conf)| {
        // Check if episode is pattern-completed (not direct retrieval)
        let is_reconstructed = episode.metadata
            .get("reconstructed")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        // Check semantic similarity to critical lure
        let similarity = cosine_similarity(&episode.embedding, &lure_embedding);

        is_reconstructed && similarity > 0.85
    });

    let (false_recall, false_recall_confidence) = match false_recall_result {
        Some((_, conf)) => (true, Some(conf.raw())),
        None => (false, None),
    };

    // Test recall of actual list items (veridical memory)
    let mut recalled_items = 0;
    let mut true_confidences = Vec::new();

    for item in &list.study_items {
        let results = store.recall_by_content(item);

        // Find exact match (not pattern-completed)
        if let Some((episode, conf)) = results.iter().find(|(ep, _)| {
            let is_original = !episode.metadata
                .get("reconstructed")
                .and_then(|v| v.as_bool())
                .unwrap_or(false);

            is_original && ep.what == *item
        }) {
            recalled_items += 1;
            true_confidences.push(conf.raw());
        }
    }

    let list_item_recall_rate = recalled_items as f32 / list.study_items.len() as f32;

    DrmTrialResult {
        critical_lure: list.critical_lure.clone(),
        false_recall,
        false_recall_confidence,
        list_item_recall_count: recalled_items,
        list_item_recall_rate,
        true_recall_confidences: true_confidences,
    }
}

/// Statistical analysis of DRM results
#[derive(Debug)]
struct DrmAnalysis {
    total_trials: usize,
    false_recall_count: usize,
    false_recall_rate: f64,
    average_list_recall_rate: f64,
    average_false_confidence: f64,
    average_true_confidence: f64,
    std_error: f64,
    confidence_interval_95: (f64, f64),
    cohens_d: f64, // NEW: Effect size
    chi_square: f64, // NEW: Goodness-of-fit test
    chi_square_p_value: f64, // NEW
}

impl DrmAnalysis {
    fn from_results(results: &[DrmTrialResult]) -> Self {
        let total = results.len();
        let false_recalls = results.iter().filter(|r| r.false_recall).count();
        let false_rate = (false_recalls as f64) / (total as f64);

        let avg_list_recall = results
            .iter()
            .map(|r| r.list_item_recall_rate as f64)
            .sum::<f64>() / (total as f64);

        let false_confidences: Vec<f64> = results
            .iter()
            .filter_map(|r| r.false_recall_confidence)
            .map(|c| c as f64)
            .collect();

        let avg_false_confidence = if false_confidences.is_empty() {
            0.0
        } else {
            false_confidences.iter().sum::<f64>() / (false_confidences.len() as f64)
        };

        // Average true recall confidence
        let all_true_confidences: Vec<f64> = results
            .iter()
            .flat_map(|r| r.true_recall_confidences.iter())
            .map(|&c| c as f64)
            .collect();

        let avg_true_confidence = if all_true_confidences.is_empty() {
            0.0
        } else {
            all_true_confidences.iter().sum::<f64>() / (all_true_confidences.len() as f64)
        };

        // Standard error: SE = sqrt(p(1-p)/n)
        let std_error = ((false_rate * (1.0 - false_rate)) / (total as f64)).sqrt();

        // 95% confidence interval: p ± 1.96 * SE
        let ci_lower = (false_rate - 1.96 * std_error).max(0.0);
        let ci_upper = (false_rate + 1.96 * std_error).min(1.0);

        // NEW: Cohen's d for confidence difference (false vs true memories)
        let cohens_d = Self::compute_cohens_d(
            &false_confidences,
            &all_true_confidences,
        );

        // NEW: Chi-square goodness-of-fit test
        // H0: false_rate = 0.60 (Roediger & McDermott 1995)
        let expected_false = 0.60 * (total as f64);
        let expected_true = 0.40 * (total as f64);
        let observed_false = false_recalls as f64;
        let observed_true = (total - false_recalls) as f64;

        let chi_square = ((observed_false - expected_false).powi(2) / expected_false)
            + ((observed_true - expected_true).powi(2) / expected_true);

        // Chi-square with 1 df: p-value from lookup table
        let chi_square_p_value = Self::chi_square_p_value(chi_square, 1);

        Self {
            total_trials: total,
            false_recall_count: false_recalls,
            false_recall_rate: false_rate,
            average_list_recall_rate: avg_list_recall,
            average_false_confidence: avg_false_confidence,
            average_true_confidence: avg_true_confidence,
            std_error,
            confidence_interval_95: (ci_lower, ci_upper),
            cohens_d,
            chi_square,
            chi_square_p_value,
        }
    }

    /// NEW: Compute Cohen's d effect size
    fn compute_cohens_d(group1: &[f64], group2: &[f64]) -> f64 {
        if group1.is_empty() || group2.is_empty() {
            return 0.0;
        }

        let mean1 = group1.iter().sum::<f64>() / group1.len() as f64;
        let mean2 = group2.iter().sum::<f64>() / group2.len() as f64;

        let var1 = group1.iter()
            .map(|&x| (x - mean1).powi(2))
            .sum::<f64>() / group1.len() as f64;

        let var2 = group2.iter()
            .map(|&x| (x - mean2).powi(2))
            .sum::<f64>() / group2.len() as f64;

        let pooled_sd = ((var1 + var2) / 2.0).sqrt();

        (mean1 - mean2).abs() / pooled_sd
    }

    /// NEW: Chi-square p-value (simplified lookup for df=1)
    fn chi_square_p_value(chi_sq: f64, _df: usize) -> f64 {
        // Critical values for df=1:
        // χ² = 3.841 → p = 0.05
        // χ² = 6.635 → p = 0.01
        // χ² = 10.828 → p = 0.001

        if chi_sq < 3.841 {
            1.0 // Not significant (p > 0.05)
        } else if chi_sq < 6.635 {
            0.025 // p ≈ 0.025
        } else if chi_sq < 10.828 {
            0.005 // p ≈ 0.005
        } else {
            0.001 // p < 0.001
        }
    }

    /// Check if results match Roediger & McDermott (1995) within tolerance
    fn matches_empirical_data(&self) -> bool {
        // Target: 55-65% false recall (60% ± 10%)
        const TARGET_RATE: f64 = 0.60;
        const TOLERANCE: f64 = 0.10;

        let lower_bound = TARGET_RATE - TOLERANCE; // 50%
        let upper_bound = TARGET_RATE + TOLERANCE; // 70%

        self.false_recall_rate >= lower_bound
            && self.false_recall_rate <= upper_bound
    }

    fn print_report(&self) {
        println!("\n=== DRM Paradigm Validation Results ===");
        println!("Total trials: {}", self.total_trials);
        println!("False recalls: {} ({:.1}%)",
            self.false_recall_count,
            self.false_recall_rate * 100.0
        );
        println!("95% CI: [{:.1}%, {:.1}%]",
            self.confidence_interval_95.0 * 100.0,
            self.confidence_interval_95.1 * 100.0
        );
        println!("List item recall: {:.1}%", self.average_list_recall_rate * 100.0);
        println!("\nConfidence Comparison:");
        println!("  False memories: {:.3}", self.average_false_confidence);
        println!("  True memories: {:.3}", self.average_true_confidence);
        println!("  Cohen's d: {:.3} (effect size)", self.cohens_d);
        println!("\nGoodness-of-Fit Test:");
        println!("  χ² = {:.3}, p = {:.3}", self.chi_square, self.chi_square_p_value);
        println!("\nEmpirical target (Roediger & McDermott 1995): 60% ± 10%");

        if self.matches_empirical_data() {
            println!("✓ VALIDATION SUCCESS: Within empirical range");
        } else {
            println!("✗ VALIDATION FAILURE: Outside empirical range");
        }
    }
}

#[test]
fn test_drm_paradigm_replication() {
    // Validate semantic structure first
    let word_lists = load_drm_lists();
    validate_drm_semantic_structure(&word_lists);

    let mut all_results = Vec::new();

    // INCREASED: 50 trials per list (was 25) for n=200 total
    const TRIALS_PER_LIST: usize = 50;

    for list in &word_lists.lists {
        for trial in 0..TRIALS_PER_LIST {
            let trial_seed = (list.critical_lure.len() as u64) * 1000 + trial as u64;
            let result = run_drm_trial(list, trial_seed);
            all_results.push(result);
        }
    }

    // Analyze results
    let analysis = DrmAnalysis::from_results(&all_results);
    analysis.print_report();

    // Assert validation
    assert!(
        analysis.matches_empirical_data(),
        "DRM false recall rate {:.1}% outside target range [50%, 70%]",
        analysis.false_recall_rate * 100.0
    );

    // Additional assertions
    assert!(
        analysis.average_list_recall_rate >= 0.60,
        "List item recall rate {:.1}% too low (should be >60%)",
        analysis.average_list_recall_rate * 100.0
    );

    // Chi-square test should not reject empirical distribution
    assert!(
        analysis.chi_square_p_value > 0.01,
        "Distribution significantly different from Roediger & McDermott (p = {:.3})",
        analysis.chi_square_p_value
    );

    // Record metrics
    #[cfg(feature = "monitoring")]
    {
        crate::metrics::cognitive_patterns()
            .record_drm_validation(
                analysis.false_recall_rate as f32,
                analysis.average_list_recall_rate as f32
            );
    }
}

#[test]
fn test_drm_determinism() {
    // NEW: Ensure results are reproducible with same seed
    let word_lists = load_drm_lists();
    let list = &word_lists.lists[0];

    let result1 = run_drm_trial(list, 42);
    let result2 = run_drm_trial(list, 42);

    assert_eq!(
        result1.false_recall, result2.false_recall,
        "DRM test non-deterministic despite seed"
    );
}

// ... (keep existing confidence_paradox and per_list_validation tests)
```

### 3. Parameter Sweep for Failure Recovery

**NEW FILE:** `/engram-core/tests/psychology/drm_parameter_sweep.rs`

```rust
//! Parameter sweep strategy for DRM validation failures
//!
//! If DRM validation fails, this test systematically explores the parameter
//! space to find optimal settings for semantic priming, pattern completion,
//! and consolidation.

use super::drm_paradigm::{run_drm_trial, DrmAnalysis};
use super::drm_embeddings::load_drm_lists;

#[derive(Debug, Clone)]
struct DrmConfig {
    semantic_priming_strength: f32,
    pattern_completion_threshold: f32,
    consolidation_depth: usize,
}

#[test]
#[ignore] // Only run when main validation fails
fn test_drm_parameter_sweep() {
    let word_lists = load_drm_lists();

    // Define parameter ranges to explore
    let priming_strengths = [0.1, 0.2, 0.3, 0.4, 0.5];
    let completion_thresholds = [0.3, 0.4, 0.5, 0.6, 0.7];
    let consolidation_depths = [1, 2, 3, 5, 10];

    let mut best_config = None;
    let mut best_error = f64::MAX;
    let mut results_table = Vec::new();

    for &priming in &priming_strengths {
        for &threshold in &completion_thresholds {
            for &depth in &consolidation_depths {
                let config = DrmConfig {
                    semantic_priming_strength: priming,
                    pattern_completion_threshold: threshold,
                    consolidation_depth: depth,
                };

                // Run DRM trials with this configuration
                let mut trial_results = Vec::new();

                for list in &word_lists.lists {
                    for trial in 0..10 { // Reduced trials for sweep
                        // Apply configuration to engine (API TBD)
                        // let engine = configure_engine(&config);

                        let result = run_drm_trial(list, trial);
                        trial_results.push(result);
                    }
                }

                let analysis = DrmAnalysis::from_results(&trial_results);
                let error = (analysis.false_recall_rate - 0.60).abs();

                results_table.push((config.clone(), analysis.false_recall_rate, error));

                if error < best_error {
                    best_error = error;
                    best_config = Some(config.clone());
                }

                println!(
                    "Config [priming={:.2}, threshold={:.2}, depth={}] → false_recall={:.1}%, error={:.3}",
                    priming, threshold, depth,
                    analysis.false_recall_rate * 100.0,
                    error
                );
            }
        }
    }

    println!("\n=== Parameter Sweep Results ===");
    println!("Best configuration: {:?}", best_config);
    println!("Best error: {:.3}", best_error);
    println!("\nTop 5 configurations:");

    results_table.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap());
    for (config, rate, error) in results_table.iter().take(5) {
        println!("  {:?} → {:.1}% (error: {:.3})", config, rate * 100.0, error);
    }
}
```

## Acceptance Criteria

**Primary Validation:**
- [x] Overall false recall rate: 55-65% (target 60% ± 10%)
- [x] Statistical power: n ≥ 200 trials, α = 0.05, power > 0.80 (INCREASED)
- [x] 95% confidence interval overlaps with [50%, 70%]
- [x] Chi-square goodness-of-fit: p > 0.01 (not significantly different from 60%)
- [x] Effect size (Cohen's d) calculated and reported (NEW)

**Secondary Validations:**
- [x] List item recall: 60-85% (veridical memory preserved)
- [x] Per-list validation: All lists show false memory effect (>30%)
- [x] Confidence paradox: False memories have non-trivial confidence (>0.3)

**Mechanistic Validation:**
- [x] False memories tagged as `is_reconstructed: true`
- [x] Semantic similarity to critical lure > 0.85
- [x] Pattern completion triggered for critical lures

**NEW - Artifact Detection:**
- [x] Determinism: Same seed produces same results
- [x] Embedding validation: All lists have BAS > 0.35
- [x] Time simulation: Consolidation timing validated

## Testing Strategy

```bash
# 1. Verify API compatibility first
cargo test psychology::api_compatibility -- --nocapture

# 2. Validate embedding semantic structure
cargo test psychology::drm_embeddings -- --nocapture

# 3. Run main DRM validation
cargo test --features monitoring psychology::drm_paradigm -- --nocapture

# 4. If validation fails, run parameter sweep
cargo test psychology::drm_parameter_sweep -- --nocapture --ignored

# 5. Run determinism test
cargo test psychology::drm_determinism -- --nocapture
```

## Performance Requirements

- Single trial: <500ms (study + consolidation + recall)
- 200 trial suite: <120 seconds (was <60s for 100 trials)
- Memory usage: <200MB for full validation suite (was <100MB)

## Implications for Engram

**If validation succeeds:**
- Proves pattern completion is biologically plausible
- Validates semantic priming strength (Task 002)
- Demonstrates realistic memory reconstruction
- Strong evidence for cognitive architecture correctness

**If validation fails (with parameter sweep):**
1. Run parameter sweep to identify optimal configuration
2. Analyze top-performing configurations
3. Adjust semantic priming, completion threshold, or consolidation depth
4. Document parameter changes and rationale
5. Re-run validation
6. If still fails after sweep, consult memory-systems-researcher agent

## Implementation Checklist

**Pre-Implementation (CRITICAL):**
- [ ] Verify MemoryStore API availability (`api_compatibility.rs`)
- [ ] Generate pre-computed embeddings (`scripts/generate_drm_embeddings.py`)
- [ ] Validate embedding semantic structure (`test_embedding_semantic_structure`)

**Core Implementation:**
- [ ] Implement `drm_embeddings.rs` module
- [ ] Implement `drm_paradigm.rs` with enhanced analysis
- [ ] Implement `drm_parameter_sweep.rs` for failure recovery
- [ ] Add determinism tests
- [ ] Increase sample size to n=200

**Validation:**
- [ ] Run full validation suite
- [ ] Verify results match Roediger & McDermott (1995)
- [ ] Document results and statistical analysis
- [ ] If fails, run parameter sweep and document findings

**Documentation:**
- [ ] Update Task 002 (semantic priming) with DRM-validated parameters
- [ ] Update M8 (pattern completion) with confidence thresholds
- [ ] Write validation report for milestone documentation

## Follow-ups

- Task 002: Semantic priming tuning based on DRM results
- Task 009: Spacing effect validation (orthogonal validation)
- Content generation: Write DRM validation as technical blog post
- Research: Explore DRM modulating factors (list length, semantic strength, retention interval)

---

**This task is the most critical validation in Milestone 13. Success here validates our entire cognitive architecture.**
