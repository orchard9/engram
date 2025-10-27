# Task 005b: Fan Effect

**Status:** PENDING
**Priority:** P0 (Critical Path)
**Estimated Duration:** 1 day
**Dependencies:** Task 004 (Proactive Interference Detection)
**Agent Review Required:** memory-systems-researcher

## Overview

Implement fan effect detection and modeling based on Anderson (1974). The fan effect describes how retrieval time increases linearly with the number of associations to a concept. A person with 3 facts stored about them is retrieved ~140ms slower than a person with only 1 fact.

This is a **critical path** task because it validates Engram's ability to model retrieval-stage interference from associative density.

## Psychology Foundation

### Empirical Basis
**Primary Source:** Anderson, J. R. (1974). Retrieval of propositional information from long-term memory. Cognitive Psychology, 6(4), 451-474.

**Phenomenon:** Retrieval time increases linearly with number of associations (fan) to a concept

**Key Findings (Anderson 1974 Table 1):**
```
Fan = 1:  1159ms ± 22ms  (baseline)
Fan = 2:  1236ms ± 25ms  (+77ms)
Fan = 3:  1305ms ± 28ms  (+69ms average from fan 2)

Average slope: ~70ms per additional association
```

**Classical Example:**
```
Facts learned:
1. The doctor is in the bank.     (doctor: fan=1, bank: fan=1)
2. The doctor is in the church.   (doctor: fan=2, church: fan=1)
3. The doctor is in the park.     (doctor: fan=3, park: fan=1)

Retrieval probe: "Is the doctor in the bank?"
- Must search through 3 associations to doctor
- Retrieval time ≈ 1150ms + (3-1) × 70ms = 1290ms

Retrieval probe: "Is the doctor in the park?"
- Same fan on doctor (3 associations)
- Retrieval time similar: ~1290ms
```

**Mechanism:** Spreading activation must divide among all outgoing associations. Higher fan → lower activation per edge → slower retrieval.

### Boundary Conditions

1. **Fan Computation:** Count of edges (associations) from a node in memory graph
2. **Linear Scaling:** RT = base + (fan - 1) × 70ms per Anderson (1974) data
3. **Base Retrieval Time:** 1150ms for fan=1 (Anderson 1974)
4. **Retrieval Stage:** Applied during recall, NOT during encoding or consolidation
5. **Associative Edges:** All edges count, not just semantic similarity
6. **Minimum Fan:** 1 (a node with zero edges cannot be retrieved)

### Neuroscience Mechanism

**Spreading Activation Competition:**
- Activation spreads from cue node through graph edges
- Total activation is finite (limited by neural resources)
- Activation divides among outgoing edges: A_edge = A_total / fan
- Higher fan → lower per-edge activation → slower to reach threshold → longer RT

**Hippocampal Pattern Separation:**
- Dense connectivity (high fan) requires more pattern separation
- More separation → more disambiguation time → longer retrieval

**NOT an interference effect in traditional sense:**
- No memory degradation (accuracy remains high in Anderson 1974)
- Pure retrieval-time effect from activation dynamics
- Different from proactive/retroactive interference (which affect accuracy)

## Implementation Specifications

### File Structure
```
engram-core/src/cognitive/interference/
├── mod.rs (extend from Task 004)
├── proactive.rs (Task 004)
├── retroactive.rs (Task 005a)
├── fan_effect.rs (new, this task)

engram-core/src/activation/
├── recall.rs (extend with fan effect)

engram-core/tests/cognitive/
└── fan_effect_tests.rs (new)
```

### Fan Effect Detector

**File:** `/engram-core/src/cognitive/interference/fan_effect.rs`

```rust
use crate::{NodeId, MemoryGraph};
use std::collections::HashMap;

/// Fan effect detector for retrieval-stage associative interference
///
/// Implements Anderson (1974) linear RT increase with associative fan.
///
/// CRITICAL: This operates during RETRIEVAL stage, not encoding or consolidation.
/// It models activation competition, not memory degradation.
pub struct FanEffectDetector {
    /// Base retrieval time for fan=1 (default: 1150ms per Anderson 1974)
    base_retrieval_time_ms: f32,

    /// Time per additional association (default: 70ms per Anderson 1974)
    /// CORRECTED from previous spec which used 50ms
    time_per_association_ms: f32,

    /// Activation divisor mode (linear vs sqrt)
    /// Linear: activation / fan (default)
    /// Sqrt: activation / sqrt(fan) (softer falloff)
    use_sqrt_divisor: bool,

    /// Minimum fan value (default: 1, cannot be 0)
    min_fan: usize,
}

impl Default for FanEffectDetector {
    fn default() -> Self {
        Self {
            base_retrieval_time_ms: 1150.0,    // Anderson (1974) fan=1 baseline
            time_per_association_ms: 70.0,     // CORRECTED from 50ms
            use_sqrt_divisor: false,           // Linear divisor by default
            min_fan: 1,
        }
    }
}

impl FanEffectDetector {
    /// Compute fan (number of associations) for a node
    ///
    /// Counts all outgoing edges from the node in the memory graph.
    ///
    /// # Returns
    /// Number of associations, minimum 1
    pub fn compute_fan(&self, node_id: NodeId, graph: &MemoryGraph) -> usize {
        let edge_count = graph.get_outgoing_edge_count(node_id);
        edge_count.max(self.min_fan)
    }

    /// Compute retrieval time based on fan
    ///
    /// Formula: RT = base_time + (fan - 1) × time_per_association
    ///
    /// # Example
    /// ```
    /// Fan 1: 1150ms + (1-1) × 70ms = 1150ms
    /// Fan 2: 1150ms + (2-1) × 70ms = 1220ms
    /// Fan 3: 1150ms + (3-1) × 70ms = 1290ms
    /// ```
    ///
    /// Matches Anderson (1974) empirical data within ±20ms.
    pub fn compute_retrieval_time_ms(&self, fan: usize) -> f32 {
        let fan_clamped = fan.max(self.min_fan);
        self.base_retrieval_time_ms + ((fan_clamped - 1) as f32 * self.time_per_association_ms)
    }

    /// Compute activation divisor based on fan
    ///
    /// Used to model spreading activation competition:
    /// activation_per_edge = total_activation / divisor
    ///
    /// Linear mode (default): divisor = fan
    /// Sqrt mode: divisor = sqrt(fan)
    pub fn compute_activation_divisor(&self, fan: usize) -> f32 {
        let fan_clamped = fan.max(self.min_fan) as f32;

        if self.use_sqrt_divisor {
            fan_clamped.sqrt()
        } else {
            fan_clamped
        }
    }

    /// Detect fan effect for a retrieval operation
    ///
    /// Computes fan, retrieval time, and activation divisor for a node.
    ///
    /// # Returns
    /// FanEffectResult with all fan-related metrics
    pub fn detect_fan_effect(
        &self,
        node_id: NodeId,
        graph: &MemoryGraph
    ) -> FanEffectResult {
        let fan = self.compute_fan(node_id, graph);
        let retrieval_time_ms = self.compute_retrieval_time_ms(fan);
        let activation_divisor = self.compute_activation_divisor(fan);

        // Record metrics
        #[cfg(feature = "monitoring")]
        {
            crate::metrics::cognitive_patterns()
                .record_fan_effect(fan, retrieval_time_ms);
        }

        FanEffectResult {
            fan,
            retrieval_time_ms,
            activation_divisor,
        }
    }

    /// Apply fan effect to activation spreading
    ///
    /// Divides activation among outgoing edges based on fan.
    ///
    /// Used during spreading activation to model competition.
    pub fn apply_to_activation(
        &self,
        base_activation: f32,
        fan_result: &FanEffectResult
    ) -> f32 {
        base_activation / fan_result.activation_divisor
    }
}

/// Result of fan effect detection
#[derive(Debug, Clone, Copy)]
pub struct FanEffectResult {
    /// Number of associations (fan) for the node
    pub fan: usize,

    /// Predicted retrieval time in milliseconds
    pub retrieval_time_ms: f32,

    /// Activation divisor (how much to divide activation by)
    pub activation_divisor: f32,
}

impl FanEffectResult {
    /// Check if fan is high (>3 associations)
    pub fn is_high_fan(&self) -> bool {
        self.fan > 3
    }

    /// Retrieval time slowdown compared to fan=1 baseline
    pub fn slowdown_ms(&self, baseline_ms: f32) -> f32 {
        self.retrieval_time_ms - baseline_ms
    }

    /// No fan effect (single association)
    pub fn single_association(base_time_ms: f32) -> Self {
        Self {
            fan: 1,
            retrieval_time_ms: base_time_ms,
            activation_divisor: 1.0,
        }
    }
}

/// Statistics aggregator for fan effect across memory graph
pub struct FanEffectStatistics {
    /// Distribution of fan values: fan → count
    pub fan_distribution: HashMap<usize, usize>,

    /// Average fan across all nodes
    pub average_fan: f32,

    /// Maximum fan in graph
    pub max_fan: usize,

    /// Median fan
    pub median_fan: usize,

    /// Nodes with high fan (>3)
    pub high_fan_nodes: Vec<NodeId>,
}

impl FanEffectStatistics {
    /// Compute fan statistics across entire memory graph
    pub fn compute(graph: &MemoryGraph, detector: &FanEffectDetector) -> Self {
        let mut fan_distribution = HashMap::new();
        let mut fan_values = Vec::new();
        let mut high_fan_nodes = Vec::new();

        for node_id in graph.all_node_ids() {
            let fan = detector.compute_fan(node_id, graph);

            *fan_distribution.entry(fan).or_insert(0) += 1;
            fan_values.push(fan);

            if fan > 3 {
                high_fan_nodes.push(node_id);
            }
        }

        fan_values.sort_unstable();

        let average_fan = if !fan_values.is_empty() {
            fan_values.iter().sum::<usize>() as f32 / fan_values.len() as f32
        } else {
            0.0
        };

        let max_fan = fan_values.last().copied().unwrap_or(0);

        let median_fan = if !fan_values.is_empty() {
            fan_values[fan_values.len() / 2]
        } else {
            0
        };

        Self {
            fan_distribution,
            average_fan,
            max_fan,
            median_fan,
            high_fan_nodes,
        }
    }

    /// Nodes with unusual fan (outliers > 2 std dev from mean)
    pub fn outlier_nodes(&self) -> Vec<NodeId> {
        // Compute standard deviation
        let mean = self.average_fan;
        let variance: f32 = self.fan_distribution
            .iter()
            .map(|(fan, count)| {
                let diff = *fan as f32 - mean;
                diff * diff * (*count as f32)
            })
            .sum::<f32>() / self.fan_distribution.values().sum::<usize>() as f32;

        let std_dev = variance.sqrt();
        let threshold = mean + 2.0 * std_dev;

        self.high_fan_nodes
            .iter()
            .filter(|node_id| {
                let fan = self.fan_distribution
                    .iter()
                    .find(|(f, _)| **f == self.fan_for_node(node_id))
                    .map(|(f, _)| *f)
                    .unwrap_or(0);
                fan as f32 > threshold
            })
            .copied()
            .collect()
    }

    fn fan_for_node(&self, _node_id: &NodeId) -> usize {
        // Helper to look up fan for a node (implementation detail)
        0  // Placeholder
    }
}
```

### Integration with Recall Engine

**Extend:** `/engram-core/src/activation/recall.rs`

```rust
use crate::cognitive::interference::FanEffectDetector;

impl RecallEngine {
    /// Apply fan effect during spreading activation
    ///
    /// Divides activation among outgoing edges based on fan count.
    fn spread_activation_with_fan(
        &self,
        node_id: NodeId,
        base_activation: f32,
        graph: &MemoryGraph
    ) -> Vec<(NodeId, f32)> {
        // Compute fan effect
        let fan_effect = self.fan_detector.detect_fan_effect(node_id, graph);

        // Get outgoing edges
        let neighbors = graph.get_neighbors(node_id);

        // Divide activation among edges
        let per_edge_activation = self.fan_detector.apply_to_activation(
            base_activation,
            &fan_effect
        );

        // Spread to neighbors
        neighbors
            .into_iter()
            .map(|neighbor_id| (neighbor_id, per_edge_activation))
            .collect()
    }
}
```

### Module Integration

**Extend:** `/engram-core/src/cognitive/interference/mod.rs`

```rust
pub mod proactive;
pub mod retroactive;
pub mod fan_effect;  // Add this

pub use proactive::{ProactiveInterferenceDetector, ProactiveInterferenceResult};
pub use retroactive::{RetroactiveInterferenceDetector, RetroactiveInterferenceResult};
pub use fan_effect::{FanEffectDetector, FanEffectResult, FanEffectStatistics};

#[derive(Debug, Clone, Copy)]
pub enum InterferenceType {
    Proactive,    // Old → New (encoding stage)
    Retroactive,  // New → Old (consolidation stage)
    Fan,          // Associative density (retrieval stage)
}
```

## Testing Strategy

### Unit Tests

**File:** `/engram-core/tests/cognitive/fan_effect_tests.rs`

#### Test 1: Anderson (1974) Empirical Validation

```rust
#[test]
fn test_anderson_1974_retrieval_times() {
    let detector = FanEffectDetector::default();

    // Anderson (1974) Table 1 data
    let empirical_data = vec![
        (1, 1159.0),  // Fan 1: baseline
        (2, 1236.0),  // Fan 2: +77ms
        (3, 1305.0),  // Fan 3: +69ms (average)
    ];

    for (fan, expected_rt) in empirical_data {
        let predicted_rt = detector.compute_retrieval_time_ms(fan);

        // Allow ±20ms tolerance (Anderson's data has ~±25ms std dev)
        let tolerance = 20.0;
        assert!(
            (predicted_rt - expected_rt).abs() < tolerance,
            "Fan {}: Expected {}ms, got {}ms (diff: {:.1}ms)",
            fan,
            expected_rt,
            predicted_rt,
            predicted_rt - expected_rt
        );
    }
}
```

#### Test 2: Linear Scaling Validation

```rust
#[test]
fn test_linear_scaling_70ms_per_association() {
    let detector = FanEffectDetector::default();

    // Test linear relationship
    let rt1 = detector.compute_retrieval_time_ms(1);
    let rt2 = detector.compute_retrieval_time_ms(2);
    let rt3 = detector.compute_retrieval_time_ms(3);
    let rt5 = detector.compute_retrieval_time_ms(5);

    // Slope should be constant (~70ms)
    let slope_1_2 = rt2 - rt1;
    let slope_2_3 = rt3 - rt2;
    let slope_3_5 = (rt5 - rt3) / 2.0;

    assert_eq!(slope_1_2, 70.0, "Slope 1→2 should be 70ms");
    assert_eq!(slope_2_3, 70.0, "Slope 2→3 should be 70ms");
    assert_eq!(slope_3_5, 70.0, "Slope 3→5 should be 70ms");

    // Verify base time
    assert_eq!(rt1, 1150.0, "Base time (fan=1) should be 1150ms");
}
```

#### Test 3: Fan Computation from Graph

```rust
#[test]
fn test_fan_computation_from_graph() {
    let detector = FanEffectDetector::default();
    let mut graph = MemoryGraph::new();

    // Create a node with known associations
    let person_node = graph.add_node("The doctor");

    // Add 3 associations
    let bank = graph.add_node("in the bank");
    let church = graph.add_node("in the church");
    let park = graph.add_node("in the park");

    graph.add_edge(person_node, bank, "location");
    graph.add_edge(person_node, church, "location");
    graph.add_edge(person_node, park, "location");

    // Compute fan
    let fan = detector.compute_fan(person_node, &graph);

    assert_eq!(fan, 3, "Doctor has 3 location associations, fan should be 3");

    // Compute retrieval time
    let rt = detector.compute_retrieval_time_ms(fan);
    let expected_rt = 1150.0 + (3 - 1) as f32 * 70.0;  // 1290ms

    assert_eq!(rt, expected_rt);
}
```

#### Test 4: Activation Division

```rust
#[test]
fn test_activation_division_among_associations() {
    let detector = FanEffectDetector::default();
    let base_activation = 1.0;

    // Fan = 1: Full activation
    let result_fan1 = FanEffectResult {
        fan: 1,
        retrieval_time_ms: 1150.0,
        activation_divisor: 1.0,
    };
    let activation_fan1 = detector.apply_to_activation(base_activation, &result_fan1);
    assert_eq!(activation_fan1, 1.0, "Fan=1 should have full activation");

    // Fan = 3: One-third activation per edge
    let result_fan3 = FanEffectResult {
        fan: 3,
        retrieval_time_ms: 1290.0,
        activation_divisor: 3.0,
    };
    let activation_fan3 = detector.apply_to_activation(base_activation, &result_fan3);
    assert_eq!(activation_fan3, 1.0 / 3.0, "Fan=3 should divide activation by 3");

    // Fan = 5: One-fifth activation per edge
    let result_fan5 = FanEffectResult {
        fan: 5,
        retrieval_time_ms: 1430.0,
        activation_divisor: 5.0,
    };
    let activation_fan5 = detector.apply_to_activation(base_activation, &result_fan5);
    assert_eq!(activation_fan5, 1.0 / 5.0, "Fan=5 should divide activation by 5");
}
```

#### Test 5: Sqrt Divisor Mode (Alternative)

```rust
#[test]
fn test_sqrt_divisor_mode() {
    let mut detector = FanEffectDetector::default();
    detector.use_sqrt_divisor = true;

    // Linear mode: fan=9 → divisor=9
    // Sqrt mode: fan=9 → divisor=3
    let divisor = detector.compute_activation_divisor(9);
    assert_eq!(divisor, 3.0, "Sqrt of 9 should be 3");

    // Application to activation
    let base_activation = 1.0;
    let result = FanEffectResult {
        fan: 9,
        retrieval_time_ms: 0.0,  // Irrelevant for this test
        activation_divisor: divisor,
    };

    let activation = detector.apply_to_activation(base_activation, &result);
    assert_eq!(activation, 1.0 / 3.0, "Activation should be divided by sqrt(9) = 3");
}
```

#### Test 6: Fan Statistics

```rust
#[test]
fn test_fan_statistics_computation() {
    let detector = FanEffectDetector::default();
    let mut graph = MemoryGraph::new();

    // Create nodes with varying fan
    let n1 = graph.add_node("node1");  // fan=1
    let n2 = graph.add_node("node2");  // fan=2
    let n3 = graph.add_node("node3");  // fan=3
    let n4 = graph.add_node("node4");  // fan=1

    let target1 = graph.add_node("target1");
    let target2 = graph.add_node("target2");
    let target3 = graph.add_node("target3");

    graph.add_edge(n1, target1, "edge");
    graph.add_edge(n2, target1, "edge");
    graph.add_edge(n2, target2, "edge");
    graph.add_edge(n3, target1, "edge");
    graph.add_edge(n3, target2, "edge");
    graph.add_edge(n3, target3, "edge");
    graph.add_edge(n4, target1, "edge");

    let stats = FanEffectStatistics::compute(&graph, &detector);

    // Verify distribution
    assert_eq!(stats.fan_distribution.get(&1), Some(&2), "Two nodes with fan=1");
    assert_eq!(stats.fan_distribution.get(&2), Some(&1), "One node with fan=2");
    assert_eq!(stats.fan_distribution.get(&3), Some(&1), "One node with fan=3");

    // Verify average
    let expected_avg = (1.0 + 2.0 + 3.0 + 1.0) / 4.0;  // 1.75
    assert_eq!(stats.average_fan, expected_avg);

    // Verify max
    assert_eq!(stats.max_fan, 3);

    // Verify median (sorted: 1, 1, 2, 3 → median is 1.5, rounds to 1 or 2)
    assert!(stats.median_fan >= 1 && stats.median_fan <= 2);
}
```

### Integration Tests

**Retrieval Stage Integration:**

```rust
#[test]
fn test_retrieval_stage_integration() {
    let detector = FanEffectDetector::default();
    let mut graph = MemoryGraph::new();

    // Create high-fan scenario
    let doctor = graph.add_node("The doctor");
    let bank = graph.add_node("in the bank");
    let church = graph.add_node("in the church");
    let park = graph.add_node("in the park");

    graph.add_edge(doctor, bank, "location");
    graph.add_edge(doctor, church, "location");
    graph.add_edge(doctor, park, "location");

    // Retrieval operation
    let recall_engine = RecallEngine::new_with_fan_detector(detector);

    let start = Instant::now();
    let results = recall_engine.retrieve("The doctor", &graph);
    let elapsed_ms = start.elapsed().as_millis() as f32;

    // Fan=3 should take longer than fan=1 baseline
    // (In practice, actual wall-clock time won't match exactly,
    //  but we can verify the detector predicts slowdown)
    let fan_effect = detector.detect_fan_effect(doctor, &graph);
    assert_eq!(fan_effect.fan, 3);
    assert_eq!(fan_effect.retrieval_time_ms, 1290.0);

    // Verify activation was divided
    assert!(results.iter().all(|r| r.activation < 1.0 / 3.0 * 1.1));
}
```

## Integration Points

### Memory Stage Clarification

**CRITICAL:** Fan effect operates during RETRIEVAL stage, NOT encoding or consolidation.

```rust
// CORRECT: Applied during retrieval
fn retrieve_episode(&self, cue: Cue) -> Result<Vec<(Episode, f32)>> {
    for candidate in candidates {
        let fan_effect = self.fan_detector.detect_fan_effect(candidate.id, &graph);

        // Divide activation based on fan
        candidate.activation = self.fan_detector.apply_to_activation(
            candidate.activation,
            &fan_effect
        );
    }
}

// INCORRECT: Do NOT apply during encoding or consolidation
fn encode_episode(&self, episode: Episode) -> Result<()> {
    // Fan effect is retrieval-stage only
    // Do NOT apply here
}
```

### Existing Systems

**M3 (Activation Spreading):** Primary integration point
- **File:** `engram-core/src/activation/recall.rs`
- **Hook:** Divide activation among outgoing edges based on fan
- **Integration:** Per-edge activation = total_activation / fan

**Metrics (Task 001):** Record fan statistics
- **Event:** Fan effect detected with fan count and predicted RT

**Graph Engine:** Query for outgoing edge count
- **File:** `engram-core/src/graph/mod.rs`
- **API:** `get_outgoing_edge_count(node_id) -> usize`

### Data Flow
1. Retrieval query submitted
2. Candidate nodes identified via spreading activation
3. For each node, compute fan (outgoing edge count)
4. Divide activation: activation_per_edge = total / fan
5. Slower retrieval for high-fan nodes (more edges to search)
6. Record fan statistics for monitoring

## Acceptance Criteria

### Must Have (Blocks Task Completion)
- [ ] Anderson (1974) replication: RT = 1150ms + (fan-1) × 70ms ±20ms
- [ ] Linear scaling validated (70ms per association, NOT 50ms)
- [ ] Fan computed correctly from graph edge count
- [ ] Activation division: activation / fan (linear mode)
- [ ] Integration with M3 spreading activation
- [ ] All unit tests pass
- [ ] Retrieval stage integration test passes
- [ ] `make quality` passes with zero clippy warnings

### Should Have
- [ ] Metrics record fan statistics correctly
- [ ] Fan statistics aggregator (distribution, average, max)
- [ ] Sqrt divisor mode as alternative (softer falloff)
- [ ] Performance: fan detection <10μs per node

### Nice to Have
- [ ] Outlier detection for unusual fan values
- [ ] Visualization of fan distribution
- [ ] Configurable time_per_association parameter

## Implementation Checklist

- [ ] Create `engram-core/src/cognitive/interference/fan_effect.rs`
- [ ] Implement `FanEffectDetector` with corrected default parameters (70ms)
- [ ] Implement `compute_fan()` using graph edge count
- [ ] Implement `compute_retrieval_time_ms()` with linear scaling
- [ ] Implement `compute_activation_divisor()` (linear and sqrt modes)
- [ ] Implement `detect_fan_effect()` main entry point
- [ ] Implement `apply_to_activation()` for spreading activation
- [ ] Implement `FanEffectStatistics` aggregator
- [ ] Extend `cognitive/interference/mod.rs` with fan_effect exports
- [ ] Extend `activation/recall.rs` with fan-aware spreading
- [ ] Create test file `fan_effect_tests.rs`
- [ ] Write Anderson (1974) empirical validation test
- [ ] Write linear scaling test (70ms per association)
- [ ] Write fan computation test (from graph edges)
- [ ] Write activation division test
- [ ] Write sqrt divisor mode test
- [ ] Write fan statistics test
- [ ] Write retrieval stage integration test
- [ ] Add metrics recording (conditional on `monitoring` feature)
- [ ] Run `make quality` and fix all warnings
- [ ] Verify performance benchmark meets <10μs requirement

## Risks and Mitigations

**Risk 1:** Graph API doesn't expose edge count efficiently
- **Likelihood:** Medium
- **Impact:** High (performance)
- **Mitigation:** Add `get_outgoing_edge_count()` method to graph API
- **Mitigation:** Cache edge counts if computation expensive
- **Mitigation:** Use adjacency list representation for O(1) edge count

**Risk 2:** Anderson (1974) parameters don't generalize to Engram's graph structure
- **Likelihood:** Low
- **Impact:** Medium (validation accuracy)
- **Mitigation:** 70ms derived from human studies, may need scaling factor
- **Mitigation:** Make time_per_association configurable for tuning
- **Mitigation:** Parameter sweep to find best fit for Engram's activation dynamics

**Risk 3:** Activation division conflicts with existing spreading activation
- **Likelihood:** Medium
- **Impact:** High (integration)
- **Mitigation:** Consult M3 spreading activation implementation
- **Mitigation:** Ensure division happens BEFORE propagation
- **Mitigation:** Integration test validates correct sequencing

## References

1. Anderson, J. R. (1974). Retrieval of propositional information from long-term memory. Cognitive Psychology, 6(4), 451-474.
2. Anderson, J. R., & Reder, L. M. (1999). The fan effect: New results and new theories. Journal of Experimental Psychology: General, 128(2), 186.
3. Schneider, D. W., & Anderson, J. R. (2012). Modeling fan effects on the time course of associative recognition. Cognitive Psychology, 64(3), 127-160.

## Notes

- Fan effect is a RETRIEVAL phenomenon, not encoding or consolidation
- Unlike proactive/retroactive interference, fan effect doesn't reduce accuracy
- It models activation competition, not memory degradation
- Linear RT increase (not exponential or quadratic)
- All edges count equally (no weighting by edge type in Anderson 1974)
- 70ms per association from empirical data, NOT 50ms (previous spec error)
- This completes the interference trilogy: proactive, retroactive, fan
