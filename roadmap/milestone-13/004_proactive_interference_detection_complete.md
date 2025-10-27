# Task 004: Proactive Interference Detection

**Status:** PENDING
**Priority:** P0 (Critical Path)
**Estimated Duration:** 2 days
**Dependencies:** Task 001 (Zero-Overhead Metrics)
**Agent Review Required:** memory-systems-researcher

## Overview

Implement proactive interference detection and modeling based on Underwood (1957). Proactive interference occurs when old memories interfere with new learning - the classic example is learning List A then List B, where recall of List B is impaired by similar items from List A.

This is a **critical path** task because interference is fundamental to understanding memory failures and must be validated against empirical data.

## Psychology Foundation

### Empirical Basis
**Source:** Underwood, B. J. (1957). Interference and forgetting. *Psychological Review*, 64(1), 49.

**Phenomenon:** Old memories interfere with new learning

**Key Findings:**
- Effect increases with number of prior lists learned
- Similarity between old and new material amplifies interference
- Retention interval modulates interference strength
- **Benchmark:** 20-30% accuracy reduction with 5+ prior similar lists

**Mechanism:**
```
Prior List: [cat, dog, bird, fish, mouse]
New List:   [lion, tiger, bear, wolf, deer]

When recalling New List:
- "cat" from Prior List competes with "lion" from New List
- Both are animals → high similarity → strong interference
- Recall accuracy drops 25% compared to no prior list
```

### Boundary Conditions

1. **Temporal Window:** Only memories within 24 hours prior count as "interfering"
2. **Similarity Threshold:** Embedding similarity ≥ 0.7 required for interference
3. **Linear Scaling:** Interference = 5% per similar prior item, capped at 30%
4. **No Retrograde:** Only forward-in-time interference (old → new)

## Implementation Specifications

### File Structure
```
engram-core/src/cognitive/interference/
├── mod.rs (new, exports all interference types)
├── proactive.rs (new, this task)

engram-core/tests/cognitive/
└── proactive_interference_tests.rs (new)
```

### Proactive Interference Detector

**File:** `/engram-core/src/cognitive/interference/proactive.rs`

```rust
use crate::{Episode, MemoryGraph, NodeId};
use chrono::{DateTime, Utc, Duration};

/// Proactive interference detector and modeler
///
/// Implements Underwood (1957) proactive interference dynamics with
/// exact boundary conditions from empirical research.
pub struct ProactiveInterferenceDetector {
    /// Similarity threshold for interference (default: 0.7)
    similarity_threshold: f32,

    /// Temporal window for "prior" memories (default: 24 hours before)
    prior_memory_window: Duration,

    /// Interference strength per similar prior item (default: 0.05 = 5%)
    interference_per_item: f32,

    /// Maximum interference effect (default: 0.30 = 30% retrieval reduction)
    max_interference: f32,
}

impl Default for ProactiveInterferenceDetector {
    fn default() -> Self {
        Self {
            similarity_threshold: 0.7,
            prior_memory_window: Duration::hours(24),
            interference_per_item: 0.05,
            max_interference: 0.30,
        }
    }
}

impl ProactiveInterferenceDetector {
    /// Detect proactive interference for a new episode
    ///
    /// Searches for similar prior memories and computes interference magnitude
    /// based on Underwood (1957) linear accumulation model.
    ///
    /// Returns interference magnitude in [0, max_interference]
    pub fn detect_interference(
        &self,
        new_episode: &Episode,
        prior_episodes: &[Episode],
        graph: &MemoryGraph
    ) -> ProactiveInterferenceResult {
        // Find similar prior memories within temporal window
        let interfering_episodes = prior_episodes
            .iter()
            .filter(|ep| self.is_interfering(new_episode, ep, graph))
            .collect::<Vec<_>>();

        let interference_count = interfering_episodes.len();

        // Compute interference magnitude (linear in similar items)
        let magnitude = (interference_count as f32 * self.interference_per_item)
            .min(self.max_interference);

        // Record metrics
        #[cfg(feature = "monitoring")]
        {
            crate::metrics::cognitive_patterns()
                .record_interference(InterferenceType::Proactive, magnitude);
        }

        ProactiveInterferenceResult {
            magnitude,
            interfering_episodes: interfering_episodes
                .into_iter()
                .map(|ep| ep.id.clone())
                .collect(),
            count: interference_count,
        }
    }

    /// Check if prior episode interferes with new episode
    ///
    /// Three requirements:
    /// 1. Temporally prior to new episode
    /// 2. Within temporal window (24 hours before)
    /// 3. Semantically similar (≥ threshold)
    fn is_interfering(
        &self,
        new_episode: &Episode,
        prior_episode: &Episode,
        graph: &MemoryGraph
    ) -> bool {
        // Must be temporally prior
        if prior_episode.timestamp >= new_episode.timestamp {
            return false;
        }

        // Must be within temporal window
        let time_diff = new_episode.timestamp - prior_episode.timestamp;
        if time_diff > self.prior_memory_window {
            return false;
        }

        // Must be sufficiently similar (semantic overlap)
        let similarity = graph.compute_embedding_similarity(
            &new_episode.embedding,
            &prior_episode.embedding
        );

        similarity >= self.similarity_threshold
    }

    /// Apply proactive interference to retrieval confidence
    ///
    /// Reduces confidence based on interference magnitude
    pub fn apply_interference(
        &self,
        base_confidence: Confidence,
        interference: &ProactiveInterferenceResult
    ) -> Confidence {
        let reduction_factor = 1.0 - interference.magnitude;
        Confidence::from_value(base_confidence.value() * reduction_factor)
    }
}

/// Result of proactive interference detection
pub struct ProactiveInterferenceResult {
    /// Magnitude of interference in [0, max_interference]
    pub magnitude: f32,

    /// Episode IDs of interfering prior memories
    pub interfering_episodes: Vec<String>,

    /// Count of interfering episodes
    pub count: usize,
}

impl ProactiveInterferenceResult {
    /// Check if interference is significant (>10% magnitude)
    pub fn is_significant(&self) -> bool {
        self.magnitude > 0.10
    }

    /// Predicted accuracy reduction percentage
    pub fn accuracy_reduction_percent(&self) -> f32 {
        self.magnitude * 100.0
    }
}
```

### Integration Module

**File:** `/engram-core/src/cognitive/interference/mod.rs`

```rust
pub mod proactive;
// Future: pub mod retroactive, pub mod fan_effect

pub use proactive::{ProactiveInterferenceDetector, ProactiveInterferenceResult};

#[derive(Debug, Clone, Copy)]
pub enum InterferenceType {
    Proactive,
    // Future: Retroactive, Fan
}
```

## Integration Points

### Existing Systems

**M3 (Activation Spreading):** Apply interference during recall
- **File:** `engram-core/src/activation/recall.rs`
- **Hook:** Reduce activation based on interference magnitude
- **Integration:**
  ```rust
  let interference = proactive_detector.detect_interference(
      &target_episode,
      &prior_episodes,
      &graph
  );
  let adjusted_activation = base_activation * (1.0 - interference.magnitude);
  ```

**M8 (Pattern Completion):** Interference competes with pattern completion
- **File:** `engram-core/src/completion/mod.rs`
- **Hook:** Interfering memories reduce pattern completion confidence
- **Integration:** Similar items from prior lists compete with reconstructed patterns

**Metrics (Task 001):** Record interference events
- **File:** `engram-core/src/metrics/cognitive_patterns.rs`
- **Event:** Proactive interference detected with magnitude and count

### Data Flow
1. New episode learning attempt
2. Query prior episodes within 24-hour window
3. Compute similarity to prior episodes
4. Count interfering episodes (similarity ≥ 0.7)
5. Calculate magnitude = count × 0.05, capped at 0.30
6. Apply interference to retrieval confidence/activation
7. Record metrics if monitoring enabled

## Testing Strategy

### Unit Tests

**File:** `/engram-core/tests/cognitive/proactive_interference_tests.rs`

#### Test 1: Underwood (1957) Replication
```rust
#[test]
fn test_underwood_1957_replication() {
    let detector = ProactiveInterferenceDetector::default();
    let graph = MemoryGraph::new();

    // Simulate learning 5 prior lists of similar items
    let prior_lists = create_five_similar_lists();
    let new_list = create_new_list_similar_to_prior();

    // Store prior lists
    for list in &prior_lists {
        store_list(&graph, list);
    }

    // Attempt to recall new list
    let mut correct_recalls = 0;
    let mut total_recalls = 0;

    for item in &new_list {
        let interference = detector.detect_interference(
            item,
            &flatten_lists(&prior_lists),
            &graph
        );

        // Apply interference to recall
        let recall_success = attempt_recall_with_interference(item, interference);
        if recall_success {
            correct_recalls += 1;
        }
        total_recalls += 1;
    }

    let accuracy = correct_recalls as f32 / total_recalls as f32;

    // Underwood (1957): 20-30% reduction with 5 prior lists
    // Expected baseline: 90% → reduced to 60-70%
    let expected_range = 0.60..=0.75;
    assert!(
        expected_range.contains(&accuracy),
        "Proactive interference accuracy {:.1}% outside expected range [60%, 75%]",
        accuracy * 100.0
    );
}
```

#### Test 2: Linear Accumulation
```rust
#[test]
fn test_linear_accumulation_with_prior_items() {
    let detector = ProactiveInterferenceDetector::default();

    for num_prior in 0..=10 {
        let prior_episodes = create_similar_episodes(num_prior);
        let new_episode = create_new_episode_similar_to_prior();

        let result = detector.detect_interference(&new_episode, &prior_episodes, &graph);

        let expected = (num_prior as f32 * 0.05).min(0.30);
        assert_eq!(
            result.magnitude,
            expected,
            "Interference should scale linearly: {} prior → {}% interference",
            num_prior,
            expected * 100.0
        );
    }
}
```

#### Test 3: Similarity Threshold
```rust
#[test]
fn test_similarity_threshold_enforcement() {
    let detector = ProactiveInterferenceDetector::default();

    // Create episodes with varying similarity
    let new_episode = create_episode("cat");
    let similar_prior = create_episode("dog");     // similarity > 0.7
    let dissimilar_prior = create_episode("car");  // similarity < 0.7

    let result_similar = detector.detect_interference(
        &new_episode,
        &[similar_prior],
        &graph
    );
    assert!(result_similar.magnitude > 0.0);

    let result_dissimilar = detector.detect_interference(
        &new_episode,
        &[dissimilar_prior],
        &graph
    );
    assert_eq!(result_dissimilar.magnitude, 0.0);
}
```

#### Test 4: Temporal Window
```rust
#[test]
fn test_temporal_window_24_hours() {
    let detector = ProactiveInterferenceDetector::default();

    let new_episode = create_episode_at_time(now());
    let recent_prior = create_episode_at_time(now() - Duration::hours(12));  // Within window
    let old_prior = create_episode_at_time(now() - Duration::hours(36));     // Outside window

    let result_recent = detector.detect_interference(&new_episode, &[recent_prior], &graph);
    assert!(result_recent.magnitude > 0.0);

    let result_old = detector.detect_interference(&new_episode, &[old_prior], &graph);
    assert_eq!(result_old.magnitude, 0.0);
}
```

### Integration Tests

**Acceptance Criteria:**
1. Underwood (1957) replication within ±10% (target: 25% reduction)
2. Linear accumulation: 5% per item, capped at 30%
3. Similarity threshold ≥ 0.7 enforced
4. Temporal window 24 hours enforced
5. Metrics record interference magnitude correctly

### Performance Requirements
- **Latency:** Interference detection <100μs for 100 prior episodes
- **Memory:** Bounded by number of episodes within 24-hour window
- **Throughput:** 1K interference checks/sec

## Acceptance Criteria

### Must Have (Blocks Task Completion)
- [ ] Underwood (1957) replication: 20-30% accuracy reduction ±10%
- [ ] Linear accumulation: magnitude = count × 0.05, max 0.30
- [ ] Similarity threshold ≥ 0.7 enforced in `is_interfering()`
- [ ] Temporal window 24 hours enforced in `is_interfering()`
- [ ] Metrics record interference events with correct magnitude
- [ ] All unit tests pass
- [ ] Integration with M3 (activation) works correctly
- [ ] `make quality` passes with zero clippy warnings

### Should Have
- [ ] Performance benchmark validates <100μs latency
- [ ] Interference result includes episode IDs for debugging
- [ ] `is_significant()` helper method for threshold checks

### Nice to Have
- [ ] Configurable temporal window (not just 24h)
- [ ] Visualization of interfering episodes
- [ ] Confidence interval for interference magnitude

## Implementation Checklist

- [ ] Create `engram-core/src/cognitive/interference/mod.rs`
- [ ] Create `engram-core/src/cognitive/interference/proactive.rs`
- [ ] Implement `ProactiveInterferenceDetector` with default parameters
- [ ] Implement `is_interfering()` with three boundary checks
- [ ] Implement linear accumulation with ceiling in `detect_interference()`
- [ ] Implement `apply_interference()` confidence reduction
- [ ] Add metrics recording (conditional on `monitoring` feature)
- [ ] Create unit test file `proactive_interference_tests.rs`
- [ ] Write Underwood (1957) replication test
- [ ] Write linear accumulation test (0-10 prior items)
- [ ] Write similarity threshold enforcement test
- [ ] Write temporal window enforcement test
- [ ] Integrate with M3 activation spreading
- [ ] Run `make quality` and fix all warnings
- [ ] Verify performance benchmark meets <100μs requirement

## Risks and Mitigations

**Risk 1:** Underwood (1957) validation fails (doesn't match 20-30%)
- **Likelihood:** Medium
- **Impact:** High (blocks milestone completion)
- **Mitigation:** Parameter sweep for similarity threshold and interference per item
- **Mitigation:** Consult memory-systems-researcher agent for biological plausibility
- **Mitigation:** Budget +1 day for tuning if initial attempt fails

**Risk 2:** Interference detection too expensive (>100μs)
- **Likelihood:** Low
- **Impact:** Medium (performance degradation)
- **Mitigation:** Index prior episodes by timestamp for efficient window queries
- **Mitigation:** Cache similarity computations for frequently accessed episodes
- **Mitigation:** Sample-based approximation if exact computation too expensive

**Risk 3:** Similarity threshold too restrictive or too permissive
- **Likelihood:** Medium
- **Impact:** Medium (validation accuracy)
- **Mitigation:** 0.7 chosen from empirical research, but validate with real data
- **Mitigation:** Make threshold configurable for A/B testing
- **Mitigation:** Log distribution of similarity scores for tuning

## References

1. Underwood, B. J. (1957). Interference and forgetting. *Psychological Review*, 64(1), 49.
2. Anderson, M. C., & Neely, J. H. (1996). Interference and inhibition in memory retrieval. *Memory*, 125-153.

## Notes

- Proactive interference is "old → new" (prior memories interfere with new learning)
- Retroactive interference is "new → old" (new learning interferes with prior memories)
- Fan effect is structural (more associations slow retrieval)
- All three are distinct phenomena with different empirical signatures
- This task implements only proactive interference; others in Tasks 005
