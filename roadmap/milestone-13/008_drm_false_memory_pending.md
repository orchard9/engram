# Task 008: DRM False Memory Paradigm Implementation

**Status:** Pending
**Priority:** P0 (Validation Critical)
**Estimated Effort:** 2 days
**Dependencies:** Task 002 (Semantic Priming), M8 (Pattern Completion)

## Objective

Replicate the Deese-Roediger-McDermott (DRM) false memory paradigm to validate that Engram's pattern completion and semantic priming produce false memories at rates matching published psychology research: 55-65% false recall for critical lures (Roediger & McDermott 1995).

This is the acid test for biological plausibility. If we can't replicate DRM, our cognitive mechanisms are wrong.

## Theoretical Foundation

**DRM Paradigm (Roediger & McDermott 1995):**

1. **Study Phase:** Present semantically related word lists
   - Example: "bed, rest, awake, tired, dream, wake, snooze, blanket..."
   - Critical lure: "sleep" (never presented, but highly associated)

2. **Test Phase:** Test recall/recognition
   - Participants falsely "remember" critical lure at ~60% rate
   - False recall often accompanied by high confidence
   - Demonstrates reconstructive nature of memory

3. **Mechanism:** Semantic activation + pattern completion
   - Studying related words activates critical lure via semantic network
   - During recall, pattern completion fills gap with highly activated lure
   - System "confabulates" plausible but false memory

**Critical Validation:**
- False recall rate: 55-65% (target: 60%)
- Confidence: False memories often high confidence (paradox)
- List items: 70-85% veridical recall

## Integration Points

**Uses:**
- `/engram-core/src/cognitive/priming/semantic.rs` - Semantic priming (Task 002)
- `/engram-core/src/completion/mod.rs` - Pattern completion (M8)
- `/engram-core/src/activation/spreading.rs` - Activation spreading (M3)

**Creates:**
- `/engram-core/tests/psychology/drm_paradigm.rs` - DRM experiment implementation
- `/engram-core/tests/psychology/drm_word_lists.json` - Standard DRM lists
- `/engram-core/tests/psychology/drm_analysis.rs` - Statistical analysis

## Detailed Specification

### 1. Standard DRM Word Lists

```json
// /engram-core/tests/psychology/drm_word_lists.json

{
  "lists": [
    {
      "critical_lure": "sleep",
      "study_items": [
        "bed", "rest", "awake", "tired", "dream",
        "wake", "snooze", "blanket", "doze", "slumber",
        "snore", "nap", "peace", "yawn", "drowsy"
      ]
    },
    {
      "critical_lure": "chair",
      "study_items": [
        "table", "sit", "legs", "seat", "couch",
        "desk", "recliner", "sofa", "wood", "cushion",
        "swivel", "stool", "sitting", "rocking", "bench"
      ]
    },
    {
      "critical_lure": "doctor",
      "study_items": [
        "nurse", "sick", "lawyer", "medicine", "health",
        "hospital", "dentist", "physician", "ill", "patient",
        "office", "stethoscope", "surgeon", "clinic", "cure"
      ]
    },
    {
      "critical_lure": "mountain",
      "study_items": [
        "hill", "valley", "climb", "summit", "top",
        "molehill", "peak", "plain", "glacier", "goat",
        "bike", "climber", "range", "steep", "ski"
      ]
    }
  ]
}
```

### 2. DRM Experiment Implementation

```rust
// /engram-core/tests/psychology/drm_paradigm.rs

use engram_core::{MemoryEngine, Episode, Confidence};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

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
    false_recognition: bool,
    false_confidence: Option<f32>,
    list_item_recall_count: usize,
    list_item_recall_rate: f32,
}

/// Load standard DRM word lists
fn load_drm_lists() -> DrmWordLists {
    let json_data = include_str!("drm_word_lists.json");
    serde_json::from_str(json_data)
        .expect("Failed to parse DRM word lists")
}

/// Run a single DRM trial
///
/// 1. Store study list as episodes
/// 2. Trigger consolidation (semantic pattern extraction)
/// 3. Test recall for critical lure (false memory)
/// 4. Test recall for list items (veridical memory)
fn run_drm_trial(
    engine: &MemoryEngine,
    list: &DrmList
) -> DrmTrialResult {
    // Study phase: Present list items
    for word in &list.study_items {
        let episode = Episode::from_text(
            word,
            get_embedding(word) // Use real embeddings for semantic similarity
        );

        engine.store_episode(episode);
    }

    // Allow semantic priming and activation spreading
    std::thread::sleep(std::time::Duration::from_millis(100));

    // Trigger consolidation to extract semantic patterns
    engine.consolidate();

    // Test phase: Attempt recall of critical lure (NOT presented)
    let lure_results = engine.recall_by_cue(&list.critical_lure);

    // Check if critical lure was falsely "recalled"
    let false_recall = lure_results
        .iter()
        .any(|r| {
            r.is_reconstructed() && // Must be pattern-completed, not retrieved
            r.content.to_lowercase().contains(&list.critical_lure.to_lowercase())
        });

    // Extract confidence if false recall occurred
    let false_confidence = if false_recall {
        lure_results
            .iter()
            .filter(|r| r.is_reconstructed())
            .map(|r| r.confidence.value())
            .max_by(|a, b| a.partial_cmp(b).unwrap())
    } else {
        None
    };

    // Test recall of actual list items (veridical memory)
    let mut recalled_items = 0;
    for item in &list.study_items {
        let results = engine.recall_by_cue(item);
        if results.iter().any(|r| !r.is_reconstructed()) {
            recalled_items += 1;
        }
    }

    let list_item_recall_rate = recalled_items as f32 / list.study_items.len() as f32;

    DrmTrialResult {
        critical_lure: list.critical_lure.clone(),
        false_recall,
        false_recognition: false, // Recognition test not implemented yet
        false_confidence,
        list_item_recall_count: recalled_items,
        list_item_recall_rate,
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
    std_error: f64,
    confidence_interval_95: (f64, f64),
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
            .filter_map(|r| r.false_confidence)
            .map(|c| c as f64)
            .collect();

        let avg_false_confidence = if false_confidences.is_empty() {
            0.0
        } else {
            false_confidences.iter().sum::<f64>() / (false_confidences.len() as f64)
        };

        // Standard error: SE = sqrt(p(1-p)/n)
        let std_error = ((false_rate * (1.0 - false_rate)) / (total as f64)).sqrt();

        // 95% confidence interval: p ± 1.96 * SE
        let ci_lower = (false_rate - 1.96 * std_error).max(0.0);
        let ci_upper = (false_rate + 1.96 * std_error).min(1.0);

        Self {
            total_trials: total,
            false_recall_count: false_recalls,
            false_recall_rate: false_rate,
            average_list_recall_rate: avg_list_recall,
            average_false_confidence: avg_false_confidence,
            std_error,
            confidence_interval_95: (ci_lower, ci_upper),
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
        println!("False memory confidence: {:.3}", self.average_false_confidence);
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
    let word_lists = load_drm_lists();
    let mut all_results = Vec::new();

    // Run 100 trials for statistical power (per list)
    const TRIALS_PER_LIST: usize = 25;

    for list in &word_lists.lists {
        for trial in 0..TRIALS_PER_LIST {
            // Create fresh engine for each trial
            let engine = MemoryEngine::new();

            let result = run_drm_trial(&engine, list);
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
fn test_drm_confidence_paradox() {
    // Validate that false memories often have high confidence
    // This is a key finding from DRM research

    let word_lists = load_drm_lists();
    let engine = MemoryEngine::new();

    let mut false_confidences = Vec::new();
    let mut true_confidences = Vec::new();

    for list in &word_lists.lists {
        let result = run_drm_trial(&engine, list);

        if let Some(conf) = result.false_confidence {
            false_confidences.push(conf);
        }

        // Collect true memory confidences for comparison
        for item in &list.study_items {
            if let Some(recall) = engine.recall_by_cue(item).first() {
                if !recall.is_reconstructed() {
                    true_confidences.push(recall.confidence.value());
                }
            }
        }
    }

    let avg_false_conf: f32 = false_confidences.iter().sum::<f32>()
        / false_confidences.len() as f32;

    let avg_true_conf: f32 = true_confidences.iter().sum::<f32>()
        / true_confidences.len() as f32;

    println!("False memory confidence: {:.3}", avg_false_conf);
    println!("True memory confidence: {:.3}", avg_true_conf);

    // False memories should have non-trivial confidence
    // (exact relationship varies, but should be >0.3)
    assert!(
        avg_false_conf > 0.3,
        "False memory confidence {:.3} too low",
        avg_false_conf
    );
}

#[test]
fn test_drm_per_list_validation() {
    // Validate each DRM list individually to ensure
    // results are not driven by single outlier list

    let word_lists = load_drm_lists();

    for list in &word_lists.lists {
        let mut results = Vec::new();

        for _ in 0..50 { // 50 trials per list
            let engine = MemoryEngine::new();
            let result = run_drm_trial(&engine, list);
            results.push(result);
        }

        let analysis = DrmAnalysis::from_results(&results);

        println!("\nList: {} → False recall: {:.1}%",
            list.critical_lure,
            analysis.false_recall_rate * 100.0
        );

        // Each list should show false memory effect
        // (though some variation is expected)
        assert!(
            analysis.false_recall_rate >= 0.30,
            "List '{}' shows insufficient false memory effect: {:.1}%",
            list.critical_lure,
            analysis.false_recall_rate * 100.0
        );
    }
}
```

### 3. Acceptance Criteria

**Primary Validation:**
- Overall false recall rate: 55-65% (target 60% ± 10%)
- Statistical power: n ≥ 100 trials, α = 0.05, power > 0.80
- 95% confidence interval overlaps with [50%, 70%]

**Secondary Validations:**
- List item recall: 60-85% (veridical memory preserved)
- Per-list validation: All lists show false memory effect (>30%)
- Confidence paradox: False memories have non-trivial confidence (>0.3)

**Mechanistic Validation:**
- False memories tagged as `is_reconstructed: true`
- Semantic priming strength correlates with false recall
- Pattern completion triggered for critical lures

## Testing Strategy

```bash
# Run DRM paradigm validation
cargo test --features monitoring psychology::drm_paradigm -- --nocapture

# Run with detailed statistics
RUST_LOG=debug cargo test psychology::drm_paradigm -- --nocapture

# Run per-list analysis
cargo test psychology::drm_paradigm::test_drm_per_list_validation -- --nocapture
```

## Performance Requirements

- Single trial: <500ms (study + consolidation + recall)
- 100 trial suite: <60 seconds
- Memory usage: <100MB for full validation suite

## Implications for Engram

**If validation succeeds:**
- Proves pattern completion is biologically plausible
- Validates semantic priming strength (Task 002)
- Demonstrates realistic memory reconstruction
- Strong evidence for cognitive architecture correctness

**If validation fails:**
- Re-tune semantic priming parameters
- Adjust pattern completion threshold
- Review consolidation semantic extraction (M6)
- May need memory-systems-researcher agent consultation

This is the most important validation in Milestone 13. If we can replicate DRM, we have real cognitive memory.

## Follow-ups

- Task 002: Semantic priming tuning based on DRM results
- Task 009: Spacing effect validation (orthogonal validation)
- Content generation: Write DRM validation as technical blog post
