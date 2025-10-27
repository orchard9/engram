# Task 005a: Retroactive Interference

**Status:** PENDING
**Priority:** P0 (Critical Path)
**Estimated Duration:** 2 days
**Dependencies:** Task 004 (Proactive Interference Detection)
**Agent Review Required:** memory-systems-researcher

## Overview

Implement retroactive interference detection and modeling based on McGeoch (1942) and Anderson & Neely (1996). Retroactive interference occurs when new learning during the retention interval disrupts consolidation of previously encoded memories.

This is a **critical path** task because it validates Engram's ability to model memory consolidation disruption, a fundamental phenomenon in episodic memory research.

## Psychology Foundation

### Empirical Basis
**Primary Source:** McGeoch, J. A. (1942). The psychology of human learning: An introduction. New York: Longmans, Green.

**Secondary Source:** Anderson, M. C., & Neely, J. H. (1996). Interference and inhibition in memory retrieval. Memory, 125-153.

**Phenomenon:** New learning during retention interval disrupts consolidation of older memories

### Classical Paradigm (Critical for Implementation)

**McGeoch (1942) Three-Phase Design:**
1. **T=0 min:** Learn List A (target material)
2. **T=1-10 min:** Learn List B (interpolated material, DURING retention interval)
3. **T=60 min:** Test recall of List A

**Critical Temporal Logic:**
- List B must be learned AFTER List A encoding (temporal ordering)
- List B must be learned BEFORE List A retrieval (interpolated during retention)
- List B learned AFTER retrieval cannot retroactively interfere (no backwards causation)

**Key Findings:**
- 15-25% reduction in List A recall when List B is similar (McGeoch 1942)
- Interference magnitude scales linearly with similarity (Anderson & Neely 1996)
- Effect strongest during synaptic consolidation window (0-24 hours post-encoding)
- Similarity between materials is the primary moderator

**Benchmark Data:**
```
Control (no interpolated learning):     85% recall accuracy
High similarity interpolation (>0.8):   60% recall accuracy (25% reduction)
Medium similarity interpolation (0.6):  70% recall accuracy (15% reduction)
Low similarity interpolation (<0.5):    82% recall accuracy (3% reduction)
```

### Neuroscience Mechanism

**Synaptic Consolidation Disruption:**
- Newly encoded memories undergo protein synthesis-dependent stabilization (0-24h)
- Interpolated learning during this window recruits overlapping neural populations
- Similar representations compete for limited consolidation resources
- Result: Incomplete stabilization of original memory trace

**Targeted Process:** This implementation focuses on synaptic consolidation (0-24h hippocampal-dependent window), NOT systems consolidation (weeks-months).

### Boundary Conditions

1. **Temporal Window:** Interpolated learning must occur during retention interval (after encoding, before retrieval)
2. **Similarity Threshold:** Embedding similarity >= 0.6 required for interference
3. **Linear Weighting:** Interference scales linearly with similarity (exponent = 1.0, NOT quadratic)
4. **Consolidation Stage:** Applied during background consolidation, NOT during retrieval
5. **Base Interference:** 15% reduction for similarity = 1.0, scales down linearly
6. **Maximum Interference:** Capped at 25% per McGeoch (1942) empirical data

## Implementation Specifications

### File Structure
```
engram-core/src/cognitive/interference/
├── mod.rs (extend from Task 004)
├── proactive.rs (Task 004)
├── retroactive.rs (new, this task)

engram-core/src/consolidation/
├── mod.rs (extend)
├── interference.rs (new, integration point)

engram-core/tests/cognitive/
└── retroactive_interference_tests.rs (new)
```

### Retroactive Interference Detector

**File:** `/engram-core/src/cognitive/interference/retroactive.rs`

```rust
use crate::{Episode, MemoryGraph, NodeId};
use chrono::{DateTime, Utc, Duration};

/// Retroactive interference detector for consolidation-stage disruption
///
/// Implements McGeoch (1942) retroactive interference paradigm with
/// linear similarity weighting from Anderson & Neely (1996).
///
/// CRITICAL: This detector checks for learning that occurred DURING
/// the retention interval (after target encoding, before current retrieval).
pub struct RetroactiveInterferenceDetector {
    /// Base interference magnitude (default: 0.15 = 15% at similarity=1.0)
    /// Empirical basis: McGeoch (1942) 15-25% range
    base_interference: f32,

    /// Similarity threshold for interference (default: 0.6)
    /// Below this threshold, materials are considered dissimilar
    similarity_threshold: f32,

    /// Maximum interference magnitude (default: 0.25 = 25%)
    /// Prevents unrealistic memory obliteration
    max_interference: f32,

    /// Consolidation window (default: 24 hours)
    /// Synaptic consolidation period during which interference operates
    consolidation_window: Duration,
}

impl Default for RetroactiveInterferenceDetector {
    fn default() -> Self {
        Self {
            base_interference: 0.15,       // 15% base reduction
            similarity_threshold: 0.6,      // Anderson & Neely (1996)
            max_interference: 0.25,         // 25% max per McGeoch (1942)
            consolidation_window: Duration::hours(24),
        }
    }
}

impl RetroactiveInterferenceDetector {
    /// Detect retroactive interference for a target episode during consolidation
    ///
    /// Searches for episodes learned DURING the retention interval
    /// (after target encoding, before current time) that may disrupt
    /// target's consolidation.
    ///
    /// # Parameters
    /// - `target_episode`: Episode undergoing consolidation (List A)
    /// - `all_episodes`: All episodes in memory graph
    /// - `retrieval_time`: Current time for determining retention interval
    /// - `graph`: Memory graph for similarity computation
    ///
    /// Returns interference magnitude in [0, max_interference]
    pub fn detect_interference(
        &self,
        target_episode: &Episode,
        all_episodes: &[Episode],
        retrieval_time: DateTime<Utc>,
        graph: &MemoryGraph
    ) -> RetroactiveInterferenceResult {
        // Find interpolated episodes (learned during retention interval)
        let interpolated_episodes: Vec<_> = all_episodes
            .iter()
            .filter(|ep| self.is_retroactively_interfering(
                target_episode,
                ep,
                retrieval_time,
                graph
            ))
            .collect();

        // Compute total interference magnitude
        let magnitude = self.compute_interference_magnitude(
            target_episode,
            &interpolated_episodes,
            graph
        );

        // Record metrics
        #[cfg(feature = "monitoring")]
        {
            crate::metrics::cognitive_patterns()
                .record_interference(
                    crate::metrics::InterferenceType::Retroactive,
                    magnitude
                );
        }

        RetroactiveInterferenceResult {
            magnitude,
            interfering_episodes: interpolated_episodes
                .into_iter()
                .map(|ep| ep.id.clone())
                .collect(),
            count: interpolated_episodes.len(),
        }
    }

    /// Check if an episode is retroactively interfering
    ///
    /// Three critical temporal checks (CORRECTED FROM PREVIOUS SPEC):
    /// 1. Subsequent episode learned AFTER target (temporal ordering)
    /// 2. Subsequent episode learned BEFORE retrieval (interpolated)
    /// 3. Subsequent episode within consolidation window
    /// Plus similarity check.
    fn is_retroactively_interfering(
        &self,
        target_episode: &Episode,
        subsequent_episode: &Episode,
        retrieval_time: DateTime<Utc>,
        graph: &MemoryGraph
    ) -> bool {
        // CHECK 1: Temporal ordering - subsequent must come after target
        if subsequent_episode.timestamp <= target_episode.timestamp {
            return false;  // Not temporally subsequent
        }

        // CHECK 2: Interpolated - subsequent must be learned BEFORE retrieval
        // This is the CRITICAL check that was missing in original spec
        if subsequent_episode.timestamp >= retrieval_time {
            return false;  // Learned after retrieval, cannot interfere retroactively
        }

        // CHECK 3: Within consolidation window
        let time_since_target = retrieval_time - target_episode.timestamp;
        if time_since_target > self.consolidation_window {
            return false;  // Target already consolidated, interference minimal
        }

        // CHECK 4: Semantic similarity
        let similarity = graph.compute_embedding_similarity(
            &target_episode.embedding,
            &subsequent_episode.embedding
        );

        similarity >= self.similarity_threshold
    }

    /// Compute interference magnitude with LINEAR similarity weighting
    ///
    /// Anderson & Neely (1996) show LINEAR relationship between
    /// similarity and interference, NOT quadratic.
    ///
    /// Formula: magnitude = base_interference * avg(similarity) * sqrt(count)
    /// - Linear in similarity (exponent = 1.0)
    /// - Sublinear in count (multiple interfering items, diminishing returns)
    fn compute_interference_magnitude(
        &self,
        target_episode: &Episode,
        interfering_episodes: &[&Episode],
        graph: &MemoryGraph
    ) -> f32 {
        if interfering_episodes.is_empty() {
            return 0.0;
        }

        // Compute average similarity (LINEAR weighting)
        let total_similarity: f32 = interfering_episodes
            .iter()
            .map(|ep| {
                let sim = graph.compute_embedding_similarity(
                    &target_episode.embedding,
                    &ep.embedding
                );
                // Normalize to [0, 1] range relative to threshold
                (sim - self.similarity_threshold) / (1.0 - self.similarity_threshold)
            })
            .sum();

        let avg_similarity = total_similarity / (interfering_episodes.len() as f32);

        // Scale by number of interfering items (sublinear - sqrt)
        let count_factor = (interfering_episodes.len() as f32).sqrt();

        // Final magnitude: base * similarity * sqrt(count)
        let magnitude = self.base_interference * avg_similarity * count_factor;

        // Clamp to maximum
        magnitude.min(self.max_interference)
    }

    /// Apply retroactive interference to consolidation strength
    ///
    /// Reduces the strength at which a memory consolidates based on
    /// interference from interpolated learning.
    ///
    /// Used during background consolidation operations.
    pub fn apply_interference_to_consolidation(
        &self,
        base_strength: f32,
        interference: &RetroactiveInterferenceResult
    ) -> f32 {
        let reduction_factor = 1.0 - interference.magnitude;
        base_strength * reduction_factor
    }
}

/// Result of retroactive interference detection
#[derive(Debug, Clone)]
pub struct RetroactiveInterferenceResult {
    /// Magnitude of interference in [0, max_interference]
    pub magnitude: f32,

    /// Episode IDs of interfering interpolated memories
    pub interfering_episodes: Vec<String>,

    /// Count of interfering episodes
    pub count: usize,
}

impl RetroactiveInterferenceResult {
    /// Check if interference is significant (>10% magnitude)
    pub fn is_significant(&self) -> bool {
        self.magnitude > 0.10
    }

    /// Predicted recall accuracy reduction percentage
    pub fn accuracy_reduction_percent(&self) -> f32 {
        self.magnitude * 100.0
    }

    /// No interference detected
    pub fn none() -> Self {
        Self {
            magnitude: 0.0,
            interfering_episodes: Vec::new(),
            count: 0,
        }
    }
}
```

### Integration with Consolidation System

**File:** `/engram-core/src/consolidation/interference.rs`

```rust
use crate::cognitive::interference::RetroactiveInterferenceDetector;
use crate::{Episode, MemoryGraph};
use chrono::Utc;

/// Apply retroactive interference during consolidation operations
///
/// This is called during background consolidation (e.g., overnight replay)
/// to model disruption from interpolated learning.
pub fn consolidate_with_interference(
    target_episode: &Episode,
    all_episodes: &[Episode],
    graph: &MemoryGraph,
    detector: &RetroactiveInterferenceDetector,
    base_consolidation_strength: f32
) -> f32 {
    // Detect interference from episodes learned during retention interval
    let interference = detector.detect_interference(
        target_episode,
        all_episodes,
        Utc::now(),
        graph
    );

    // Apply interference to consolidation strength
    detector.apply_interference_to_consolidation(
        base_consolidation_strength,
        &interference
    )
}
```

### Module Integration

**Extend:** `/engram-core/src/cognitive/interference/mod.rs`

```rust
pub mod proactive;
pub mod retroactive;  // Add this

pub use proactive::{ProactiveInterferenceDetector, ProactiveInterferenceResult};
pub use retroactive::{RetroactiveInterferenceDetector, RetroactiveInterferenceResult};

#[derive(Debug, Clone, Copy)]
pub enum InterferenceType {
    Proactive,    // Old → New (during encoding)
    Retroactive,  // New → Old (during consolidation)
}
```

## Testing Strategy

### Unit Tests

**File:** `/engram-core/tests/cognitive/retroactive_interference_tests.rs`

#### Test 1: McGeoch (1942) Paradigm Validation - CRITICAL TEMPORAL LOGIC

```rust
#[test]
fn test_mcgeoch_1942_interpolated_learning_paradigm() {
    let detector = RetroactiveInterferenceDetector::default();
    let graph = MemoryGraph::new();

    // PHASE 1: Learn List A at T=0
    let list_a_time = Utc::now();
    let list_a = create_episode_list("list_a", 10, list_a_time);
    store_episodes(&graph, &list_a);

    // PHASE 2: Learn List B at T=30min (DURING retention interval)
    let list_b_time = list_a_time + Duration::minutes(30);
    let list_b = create_similar_episode_list("list_b", 10, list_b_time);
    store_episodes(&graph, &list_b);

    // PHASE 3: Test List A recall at T=60min
    let retrieval_time = list_a_time + Duration::minutes(60);

    // Detect interference on List A episodes
    for target in &list_a {
        let interference = detector.detect_interference(
            target,
            &list_b,
            retrieval_time,
            &graph
        );

        // List B was interpolated (learned during retention interval)
        // Therefore it SHOULD interfere
        assert!(
            interference.magnitude > 0.10,
            "List B was interpolated during retention, should interfere significantly"
        );
    }

    // CRITICAL TEST: Learn List C at T=90min (AFTER retrieval)
    let list_c_time = retrieval_time + Duration::minutes(30);
    let list_c = create_similar_episode_list("list_c", 10, list_c_time);

    for target in &list_a {
        let interference = detector.detect_interference(
            target,
            &list_c,
            retrieval_time,  // Retrieval was at T=60, List C learned at T=90
            &graph
        );

        // List C was learned AFTER retrieval, NOT during retention interval
        // Therefore it CANNOT interfere retroactively
        assert_eq!(
            interference.magnitude, 0.0,
            "List C learned after retrieval cannot retroactively interfere"
        );
    }
}
```

#### Test 2: Linear Similarity Weighting (Anderson & Neely 1996)

```rust
#[test]
fn test_linear_similarity_weighting_not_quadratic() {
    let detector = RetroactiveInterferenceDetector::default();
    let graph = MemoryGraph::new();

    let target_time = Utc::now();
    let target = create_episode("target", target_time);

    let retrieval_time = target_time + Duration::minutes(60);

    // Create interfering episodes with controlled similarity
    let test_cases = vec![
        (0.6, "threshold"),
        (0.7, "medium"),
        (0.8, "high"),
        (0.9, "very_high"),
        (1.0, "identical"),
    ];

    let mut results = Vec::new();

    for (similarity, label) in test_cases {
        let interfering = create_episode_with_similarity(
            label,
            &target,
            similarity,
            target_time + Duration::minutes(30)  // Interpolated
        );

        let interference = detector.detect_interference(
            &target,
            &[interfering],
            retrieval_time,
            &graph
        );

        results.push((similarity, interference.magnitude));
    }

    // Verify LINEAR relationship (not quadratic)
    // Linear: ΔMagnitude / ΔSimilarity should be constant
    // Quadratic: ΔMagnitude / ΔSimilarity would increase with similarity

    for i in 0..(results.len() - 1) {
        let (sim1, mag1) = results[i];
        let (sim2, mag2) = results[i + 1];

        let slope = (mag2 - mag1) / (sim2 - sim1);

        // For LINEAR relationship, slope should be roughly constant
        // Allow 20% tolerance due to normalization effects
        if i > 0 {
            let (sim0, mag0) = results[i - 1];
            let prev_slope = (mag1 - mag0) / (sim1 - sim0);

            let slope_change_ratio = (slope / prev_slope - 1.0).abs();
            assert!(
                slope_change_ratio < 0.20,
                "Slope should be constant for linear relationship, got {:.2}% change",
                slope_change_ratio * 100.0
            );
        }
    }

    // CRITICAL: Verify NOT quadratic
    // If quadratic, similarity=0.9 would give magnitude=0.81 * base
    // With linear, similarity=0.9 gives magnitude=(0.9-0.6)/(1.0-0.6) * base = 0.75 * base
    let sim_09_result = results.iter().find(|(s, _)| *s == 0.9).unwrap();
    let normalized_sim = (0.9 - 0.6) / (1.0 - 0.6);  // = 0.75

    let expected_linear = detector.base_interference * normalized_sim;
    let tolerance = 0.02;

    assert!(
        (sim_09_result.1 - expected_linear).abs() < tolerance,
        "Linear weighting: expected {:.3}, got {:.3}",
        expected_linear,
        sim_09_result.1
    );
}
```

#### Test 3: Consolidation Window Enforcement

```rust
#[test]
fn test_consolidation_window_24_hours() {
    let detector = RetroactiveInterferenceDetector::default();
    let graph = MemoryGraph::new();

    let target_time = Utc::now();
    let target = create_episode("target", target_time);

    // Test retrieval within consolidation window (should interfere)
    let early_retrieval = target_time + Duration::hours(12);
    let early_interfering = create_similar_episode(
        "early",
        &target,
        target_time + Duration::hours(6)  // Interpolated
    );

    let early_interference = detector.detect_interference(
        &target,
        &[early_interfering],
        early_retrieval,
        &graph
    );
    assert!(early_interference.magnitude > 0.10);

    // Test retrieval outside consolidation window (minimal interference)
    let late_retrieval = target_time + Duration::hours(48);
    let late_interfering = create_similar_episode(
        "late",
        &target,
        target_time + Duration::hours(36)  // Still interpolated, but target consolidated
    );

    let late_interference = detector.detect_interference(
        &target,
        &[late_interfering],
        late_retrieval,
        &graph
    );
    assert_eq!(late_interference.magnitude, 0.0);
}
```

#### Test 4: McGeoch (1942) Empirical Benchmarks

```rust
#[test]
fn test_mcgeoch_1942_empirical_benchmarks() {
    let detector = RetroactiveInterferenceDetector::default();
    let graph = MemoryGraph::new();

    // Simulate McGeoch's experimental conditions
    let target_time = Utc::now();
    let retrieval_time = target_time + Duration::hours(1);

    // High similarity condition (should show ~25% interference)
    let target_high_sim = create_episode("target", target_time);
    let interfering_high_sim = create_episode_with_similarity(
        "interfering",
        &target_high_sim,
        0.9,  // High similarity
        target_time + Duration::minutes(10)  // Interpolated early
    );

    let result_high = detector.detect_interference(
        &target_high_sim,
        &[interfering_high_sim],
        retrieval_time,
        &graph
    );

    // McGeoch (1942): High similarity → 25% reduction
    assert!(
        (result_high.magnitude - 0.25).abs() < 0.05,
        "High similarity interference should be ~25%, got {:.1}%",
        result_high.magnitude * 100.0
    );

    // Low similarity condition (should show ~3% interference)
    let target_low_sim = create_episode("target_low", target_time);
    let interfering_low_sim = create_episode_with_similarity(
        "interfering_low",
        &target_low_sim,
        0.65,  // Just above threshold
        target_time + Duration::minutes(10)
    );

    let result_low = detector.detect_interference(
        &target_low_sim,
        &[interfering_low_sim],
        retrieval_time,
        &graph
    );

    // Low similarity → minimal interference
    assert!(
        result_low.magnitude < 0.05,
        "Low similarity interference should be <5%, got {:.1}%",
        result_low.magnitude * 100.0
    );
}
```

### Integration Tests

**Consolidation Stage Integration:**

```rust
#[test]
fn test_consolidation_stage_integration() {
    let detector = RetroactiveInterferenceDetector::default();
    let graph = MemoryGraph::new();

    let target_time = Utc::now();
    let target = create_episode("target", target_time);

    // Learn interfering material during retention interval
    let interfering = create_similar_episode(
        "interfering",
        &target,
        target_time + Duration::minutes(30)
    );

    // Simulate consolidation at T=60min
    let consolidation_time = target_time + Duration::hours(1);
    let base_strength = 1.0;

    let final_strength = consolidate_with_interference(
        &target,
        &[interfering],
        &graph,
        &detector,
        base_strength
    );

    // Consolidation strength should be reduced
    assert!(
        final_strength < base_strength,
        "Interference should reduce consolidation strength"
    );

    // Record that this happened during CONSOLIDATION, not RETRIEVAL
    // (Tested via metrics integration)
}
```

## Integration Points

### Memory Stage Clarification

**CRITICAL:** Retroactive interference operates during CONSOLIDATION stage, NOT retrieval.

```rust
// CORRECT: Applied during consolidation
fn consolidate_episode(&self, episode: Episode) -> Result<()> {
    let interference = self.retroactive_detector.detect_interference(
        &episode,
        &self.get_recent_episodes(),
        Utc::now(),
        &self.graph
    );

    let consolidation_strength = self.base_strength * (1.0 - interference.magnitude);
    self.store_consolidated(episode, consolidation_strength)
}

// INCORRECT: Do NOT apply during retrieval
fn retrieve_episode(&self, cue: Cue) -> Result<Vec<Episode>> {
    // Retroactive interference already happened during consolidation
    // Do NOT apply it again here
}
```

### Existing Systems

**M3 (Activation Spreading):** No direct integration - interference happens earlier
- **Rationale:** Retroactive interference disrupts consolidation, changing the strength of stored memories before retrieval even occurs

**M8 (Pattern Completion):** Indirectly affected via reduced memory strength
- **Rationale:** Poorly consolidated memories are harder to reconstruct

**Consolidation (Background Replay):** Primary integration point
- **File:** `engram-core/src/consolidation/mod.rs`
- **Hook:** Apply interference during offline consolidation operations

**Metrics (Task 001):** Record interference events
- **Event:** Retroactive interference detected with magnitude and count

## Acceptance Criteria

### Must Have (Blocks Task Completion)
- [ ] Temporal logic CORRECT: interpolated learning (after encoding, before retrieval)
- [ ] Linear similarity weighting (exponent = 1.0, NOT quadratic)
- [ ] McGeoch (1942) replication: 15-25% reduction for high similarity ±5%
- [ ] Anderson & Neely (1996) linear relationship validated
- [ ] Consolidation window (24h) enforced
- [ ] Integration with consolidation stage (NOT retrieval stage)
- [ ] All unit tests pass
- [ ] Critical temporal logic test passes (interpolated vs post-retrieval)
- [ ] Linear weighting test passes (not quadratic)
- [ ] `make quality` passes with zero clippy warnings

### Should Have
- [ ] Metrics record interference magnitude correctly
- [ ] Sublinear count scaling (sqrt) for multiple interfering items
- [ ] Performance: interference detection <100μs for 100 episodes

### Nice to Have
- [ ] Configurable consolidation window (not just 24h)
- [ ] Visualization of interference timing
- [ ] Confidence intervals for interference estimates

## Implementation Checklist

- [ ] Create `engram-core/src/cognitive/interference/retroactive.rs`
- [ ] Implement `RetroactiveInterferenceDetector` with correct default parameters
- [ ] Implement `is_retroactively_interfering()` with CORRECT temporal checks
- [ ] Implement `compute_interference_magnitude()` with LINEAR weighting
- [ ] Implement `apply_interference_to_consolidation()`
- [ ] Create `engram-core/src/consolidation/interference.rs`
- [ ] Implement `consolidate_with_interference()` integration function
- [ ] Extend `cognitive/interference/mod.rs` with retroactive exports
- [ ] Create test file `retroactive_interference_tests.rs`
- [ ] Write McGeoch paradigm temporal logic test (CRITICAL)
- [ ] Write linear similarity weighting test (vs quadratic)
- [ ] Write consolidation window enforcement test
- [ ] Write empirical benchmarks test (McGeoch 1942 data)
- [ ] Write consolidation stage integration test
- [ ] Add metrics recording (conditional on `monitoring` feature)
- [ ] Run `make quality` and fix all warnings
- [ ] Verify performance benchmark meets <100μs requirement

## Risks and Mitigations

**Risk 1:** Temporal logic still incorrect (common conceptual error)
- **Likelihood:** Low (explicitly validated in tests)
- **Impact:** Critical (invalidates entire implementation)
- **Mitigation:** Test both interpolated (should interfere) and post-retrieval (should NOT interfere) conditions
- **Mitigation:** Code review by memory-systems-researcher agent
- **Mitigation:** Validate against McGeoch (1942) three-phase paradigm description

**Risk 2:** Linear vs quadratic weighting difficult to validate
- **Likelihood:** Medium
- **Impact:** Medium (affects magnitude accuracy)
- **Mitigation:** Test multiple similarity levels and verify constant slope
- **Mitigation:** Compare predicted vs empirical data from Anderson & Neely (1996)
- **Mitigation:** Make exponent configurable for sensitivity analysis

**Risk 3:** Integration with consolidation system unclear
- **Likelihood:** Medium
- **Impact:** High (wrong memory stage)
- **Mitigation:** Clear documentation of consolidation vs retrieval stages
- **Mitigation:** Integration test validates correct stage application
- **Mitigation:** Consult systems-architecture-optimizer for consolidation pipeline design

## References

1. McGeoch, J. A. (1942). The psychology of human learning: An introduction. New York: Longmans, Green.
2. Anderson, M. C., & Neely, J. H. (1996). Interference and inhibition in memory retrieval. Memory, 125-153.
3. Wixted, J. T. (2004). The psychology and neuroscience of forgetting. Annual Review of Psychology, 55, 235-269.
4. Dudai, Y. (2004). The neurobiology of consolidations, or, how stable is the engram? Annual Review of Psychology, 55, 51-86.

## Notes

- Retroactive interference is "new → old" (new learning disrupts consolidation of old memories)
- Proactive interference is "old → new" (old memories interfere with new learning)
- These operate at DIFFERENT memory stages: consolidation vs encoding
- Fan effect operates at yet another stage: retrieval
- All three are distinct phenomena with different empirical signatures
- This task implements ONLY retroactive interference during consolidation
