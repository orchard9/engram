# Task 004: Proactive Interference Detection (CORRECTED)

**Status:** PENDING
**Priority:** P0 (Critical Path)
**Estimated Duration:** 2 days
**Dependencies:** Task 001 (Zero-Overhead Metrics)
**Agent Review Required:** memory-systems-researcher

## Objective

Implement proactive interference detection and modeling based on Underwood (1957). Proactive interference occurs when old memories interfere with new learning - the classic example is learning List A then List B, where recall of List B is impaired by similar items from List A.

This is a **critical path** task because interference is fundamental to understanding memory failures and must be validated against empirical data.

**CRITICAL CORRECTION APPLIED:**
- Prior memory window: 24h → 6h (synaptic consolidation boundary from Dudai et al. 2015)

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

### Boundary Conditions (CORRECTED)

1. **Temporal Window:** Only memories within **6 hours prior** count as "interfering" (CORRECTED from 24h)
   - **Rationale:** Synaptic consolidation completes within ~6 hours (protein synthesis window)
   - **Biology:** After consolidation, memories shift from hippocampal → neocortical representations
   - **CLS Theory:** Cross-consolidation-boundary interference is reduced (McClelland et al. 1995)
   - **Empirical:** Underwood (1957) used session-based interference (minutes to hours), not day-scale

2. **Similarity Threshold:** Embedding similarity ≥ 0.7 required for interference
   - **Validation:** Requires empirical calibration with category structure tests

3. **Linear Scaling:** Interference = 5% per similar prior item, capped at 30%
   - **Validated:** Matches Underwood (1957) empirical data (5 lists × 5% = 25% ✓)

4. **No Retrograde:** Only forward-in-time interference (old → new)
   - **Validated:** Implementation correctly checks temporal direction

## Implementation Specifications

### File Structure
```
engram-core/src/cognitive/interference/
├── mod.rs (new, exports all interference types)
├── proactive.rs (new, this task)

engram-core/tests/cognitive/
└── proactive_interference_tests.rs (new)
```

### Proactive Interference Detector (CORRECTED)

**File:** `/engram-core/src/cognitive/interference/proactive.rs`

```rust
use crate::{Episode, MemoryGraph, NodeId};
use chrono::{DateTime, Utc, Duration};

/// Proactive interference detector and modeler
///
/// Implements Underwood (1957) proactive interference dynamics with
/// exact boundary conditions from empirical research.
///
/// CORRECTED: Temporal window is 6 hours (not 24h) to align with
/// synaptic consolidation timescale per Dudai et al. (2015).
pub struct ProactiveInterferenceDetector {
    /// Similarity threshold for interference (default: 0.7)
    similarity_threshold: f32,

    /// Temporal window for "prior" memories (default: 6 hours before)
    /// CORRECTED: Was 24h, now 6h per synaptic consolidation boundary
    /// Empirical: Underwood (1957) session-based interference
    /// Justification: Synaptic consolidation (~6h) transitions memories from
    /// hippocampal to neocortical representations, reducing interference
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
            prior_memory_window: Duration::hours(6),  // CORRECTED: Was hours(24)
            interference_per_item: 0.05,
            max_interference: 0.30,
        }
    }
}

impl ProactiveInterferenceDetector {
    /// Create detector with custom parameters
    #[must_use]
    pub fn new(
        similarity_threshold: f32,
        prior_memory_window: Duration,
        interference_per_item: f32,
        max_interference: f32
    ) -> Self {
        Self {
            similarity_threshold,
            prior_memory_window,
            interference_per_item,
            max_interference,
        }
    }

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
    /// 2. Within temporal window (6 hours before) - CORRECTED
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

        // Must be within temporal window (6 hours - CORRECTED)
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
2. Query prior episodes within 6-hour window (CORRECTED)
3. Compute similarity to prior episodes
4. Count interfering episodes (similarity ≥ 0.7)
5. Calculate magnitude = count × 0.05, capped at 0.30
6. Apply interference to retrieval confidence/activation
7. Record metrics if monitoring enabled

## Testing Strategy

### Unit Tests (CORRECTED)

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

    // Store prior lists within 6-hour window (CORRECTED)
    let base_time = Utc::now();
    for (i, list) in prior_lists.iter().enumerate() {
        let timestamp = base_time - Duration::hours(5) + Duration::minutes(i as i64 * 30);
        store_list_at_time(&graph, list, timestamp);
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

#### Test 3: Temporal Window (CORRECTED)
```rust
#[test]
fn test_temporal_window_6_hours() {  // CORRECTED: Was 24 hours
    let detector = ProactiveInterferenceDetector::default();

    let new_episode = create_episode_at_time(Utc::now());
    let recent_prior = create_episode_at_time(Utc::now() - Duration::hours(3));  // Within 6h
    let old_prior = create_episode_at_time(Utc::now() - Duration::hours(8));     // Outside 6h

    let result_recent = detector.detect_interference(&new_episode, &[recent_prior], &graph);
    assert!(result_recent.magnitude > 0.0, "Should interfere within 6-hour window");

    let result_old = detector.detect_interference(&new_episode, &[old_prior], &graph);
    assert_eq!(result_old.magnitude, 0.0, "Should NOT interfere outside 6-hour window");
}

#[test]
fn test_consolidation_boundary_reduces_interference() {
    // ADDED: Validates synaptic consolidation effect per CLS theory
    let detector = ProactiveInterferenceDetector::default();

    // Prior memory learned >6 hours ago: no interference (consolidated)
    let consolidated_prior = create_episode_at_time(Utc::now() - Duration::hours(7));

    // Prior memory learned <6 hours ago: full interference (not yet consolidated)
    let unconsolidated_prior = create_episode_at_time(Utc::now() - Duration::hours(2));

    let new_episode = create_episode_at_time(Utc::now());

    let result_consolidated = detector.detect_interference(
        &new_episode,
        &[consolidated_prior],
        &graph
    );
    assert_eq!(result_consolidated.magnitude, 0.0, "Consolidated memory should not interfere");

    let result_unconsolidated = detector.detect_interference(
        &new_episode,
        &[unconsolidated_prior],
        &graph
    );
    assert!(result_unconsolidated.magnitude > 0.0, "Unconsolidated memory should interfere");
}
```

#### Test 4: Similarity Threshold Calibration (NEW)
```rust
#[test]
fn test_similarity_threshold_calibration_with_category_structure() {
    // ADDED: Validates threshold captures semantic category interference
    let detector = ProactiveInterferenceDetector::default();

    // Within-category pairs (dog/cat): similarity should be ≥ 0.7
    let cat_episode = create_episode_with_embedding("cat", ANIMAL_EMBEDDING_1);
    let dog_episode = create_episode_with_embedding("dog", ANIMAL_EMBEDDING_2);

    let similarity_within = graph.compute_embedding_similarity(
        &cat_episode.embedding,
        &dog_episode.embedding
    );

    assert!(
        similarity_within >= 0.7,
        "Within-category similarity should exceed threshold: {:.2}",
        similarity_within
    );

    // Across-category pairs (dog/car): similarity should be < 0.7
    let car_episode = create_episode_with_embedding("car", VEHICLE_EMBEDDING);

    let similarity_across = graph.compute_embedding_similarity(
        &dog_episode.embedding,
        &car_episode.embedding
    );

    assert!(
        similarity_across < 0.7,
        "Across-category similarity should be below threshold: {:.2}",
        similarity_across
    );
}
```

#### Test 5: Temporal Window Exact Boundary (NEW)
```rust
#[test]
fn test_temporal_window_enforced_exactly() {
    // ADDED: Validates exact boundary per is_interfering() logic
    let detector = ProactiveInterferenceDetector::default();

    let new_episode = create_episode_at_time(Utc::now());

    // Prior at 5h 59m: should interfere
    let just_inside = create_episode_at_time(Utc::now() - Duration::hours(5) - Duration::minutes(59));
    let result_inside = detector.detect_interference(&new_episode, &[just_inside], &graph);
    assert!(result_inside.magnitude > 0.0, "Should interfere at 5h 59m (inside window)");

    // Prior at 6h 01m: should NOT interfere
    let just_outside = create_episode_at_time(Utc::now() - Duration::hours(6) - Duration::minutes(1));
    let result_outside = detector.detect_interference(&new_episode, &[just_outside], &graph);
    assert_eq!(result_outside.magnitude, 0.0, "Should NOT interfere at 6h 01m (outside window)");
}
```

### Integration Tests

**Acceptance Criteria:**
1. Underwood (1957) replication within ±10% (target: 25% reduction)
2. Linear accumulation: 5% per item, capped at 30%
3. Similarity threshold ≥ 0.7 enforced
4. Temporal window 6 hours enforced (CORRECTED from 24h)
5. Consolidation boundary effect validated (NEW)
6. Metrics record interference magnitude correctly

### Performance Requirements
- **Latency:** Interference detection <100μs for 100 prior episodes
- **Memory:** Bounded by number of episodes within 6-hour window
- **Throughput:** 1K interference checks/sec

## Acceptance Criteria

### Must Have (Blocks Task Completion)
- [ ] Underwood (1957) replication: 20-30% accuracy reduction ±10%
- [ ] Linear accumulation: magnitude = count × 0.05, max 0.30
- [ ] Similarity threshold ≥ 0.7 enforced in `is_interfering()`
- [ ] Temporal window 6 hours enforced in `is_interfering()` (CORRECTED)
- [ ] Consolidation boundary test validates CLS theory (NEW)
- [ ] Similarity threshold calibration test validates category structure (NEW)
- [ ] Exact boundary test validates temporal precision (NEW)
- [ ] Metrics record interference events with correct magnitude
- [ ] All unit tests pass
- [ ] Integration with M3 (activation) works correctly
- [ ] `make quality` passes with zero clippy warnings

### Should Have
- [ ] Performance benchmark validates <100μs latency
- [ ] Interference result includes episode IDs for debugging
- [ ] `is_significant()` helper method for threshold checks

### Nice to Have
- [ ] Configurable temporal window (not just 6h)
- [ ] Visualization of interfering episodes
- [ ] Confidence interval for interference magnitude

## Implementation Checklist

- [ ] Create `engram-core/src/cognitive/interference/mod.rs`
- [ ] Create `engram-core/src/cognitive/interference/proactive.rs`
- [ ] Implement `ProactiveInterferenceDetector` with default parameters (6h window)
- [ ] Implement `is_interfering()` with three boundary checks
- [ ] Implement linear accumulation with ceiling in `detect_interference()`
- [ ] Implement `apply_interference()` confidence reduction
- [ ] Add metrics recording (conditional on `monitoring` feature)
- [ ] Create unit test file `proactive_interference_tests.rs`
- [ ] Write Underwood (1957) replication test
- [ ] Write linear accumulation test (0-10 prior items)
- [ ] Write similarity threshold enforcement test
- [ ] Write temporal window enforcement test (6h, not 24h)
- [ ] Write consolidation boundary test (NEW)
- [ ] Write similarity calibration test (NEW)
- [ ] Write exact boundary test (NEW)
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
3. McClelland, J. L., et al. (1995). Why there are complementary learning systems. *Psychological Review*, 102(3), 419.
4. Dudai, Y., et al. (2015). The consolidation and transformation of memory. *Neuron*, 88(1), 20-32.

## Notes

- Proactive interference is "old → new" (prior memories interfere with new learning)
- Retroactive interference is "new → old" (new learning interferes with prior memories)
- Fan effect is structural (more associations slow retrieval)
- All three are distinct phenomena with different empirical signatures
- This task implements only proactive interference; others in Tasks 005
- **6-hour window aligns with synaptic consolidation timescale** (CORRECTED)
