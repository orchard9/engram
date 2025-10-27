# Task 003: Associative and Repetition Priming Implementation

**Status:** PENDING
**Priority:** P1
**Estimated Duration:** 2 days
**Dependencies:** Task 002 (Semantic Priming)
**Agent Review Required:** memory-systems-researcher

## Overview

Implement associative priming (co-occurrence learning) and repetition priming (perceptual fluency) to complement semantic priming. These three priming types work together to model how human memory facilitates recall through different mechanisms.

## Psychology Foundation

### Associative Priming
**Source:** McKoon & Ratcliff (1992) - Compound cue theory

- **Mechanism:** Co-occurrence strengthens associative links
- **Example:** "Thunder" primes "lightning" through learned associations, not semantic similarity
- **Temporal Window:** 10 seconds for co-occurrence counting
- **Minimum Evidence:** 3 co-occurrences before association forms
- **Strength Calculation:** Conditional probability P(B|A) = P(A,B) / P(A)

### Repetition Priming
**Source:** Tulving & Schacter (1990) - Priming taxonomy

- **Mechanism:** Repeated exposure facilitates processing (perceptual fluency)
- **Effect Size:** 5% boost per exposure
- **Maximum:** 30% cumulative cap
- **Duration:** Persistent across session
- **Application:** Same stimulus becomes easier to process with each encounter

## Implementation Specifications

### File Structure
```
engram-core/src/cognitive/priming/
├── mod.rs (update to export new types)
├── associative.rs (new)
└── repetition.rs (new)

engram-core/tests/cognitive/
└── priming_integration_tests.rs (new)
```

### Associative Priming Engine

**File:** `/engram-core/src/cognitive/priming/associative.rs`

```rust
use std::sync::atomic::{AtomicU64, Ordering};
use dashmap::DashMap;
use chrono::{DateTime, Utc, Duration};

/// Associative priming through co-occurrence learning
pub struct AssociativePrimingEngine {
    /// Co-occurrence frequency: (node_a, node_b) -> count
    cooccurrence_counts: DashMap<(NodeId, NodeId), AtomicU64>,

    /// Total activations per node (for normalizing probabilities)
    node_activation_counts: DashMap<NodeId, AtomicU64>,

    /// Temporal window for co-occurrence (default: 10 seconds)
    cooccurrence_window: Duration,

    /// Minimum co-occurrence for association (default: 3)
    min_cooccurrence: u64,
}

impl AssociativePrimingEngine {
    /// Record co-activation of nodes within temporal window
    pub fn record_coactivation(&self, node_a: NodeId, node_b: NodeId);

    /// Compute associative priming strength: P(B|A) = P(A,B) / P(A)
    pub fn compute_association_strength(&self, prime: NodeId, target: NodeId) -> f32;

    /// Prune old co-occurrence data (periodic cleanup)
    pub fn prune_old_cooccurrences(&self);
}
```

**Key Implementation Details:**
- Symmetric co-occurrence: (A,B) and (B,A) count as same pair
- Atomic operations for thread-safe counting
- Conditional probability as strength metric
- Memory-bounded: prune entries with count < min_cooccurrence

### Repetition Priming Engine

**File:** `/engram-core/src/cognitive/priming/repetition.rs`

```rust
use dashmap::DashMap;
use std::sync::atomic::{AtomicU64, Ordering};

/// Repetition priming through exposure counting
pub struct RepetitionPrimingEngine {
    /// Exposure counts per node
    exposure_counts: DashMap<NodeId, AtomicU64>,

    /// Activation boost per repetition (default: 0.05 = 5%)
    boost_per_repetition: f32,

    /// Maximum cumulative boost (default: 0.30 = 30% cap)
    max_cumulative_boost: f32,
}

impl RepetitionPrimingEngine {
    /// Record exposure to node (recall, query, consolidation, etc.)
    pub fn record_exposure(&self, node_id: NodeId);

    /// Compute cumulative priming boost from repetitions
    /// Returns value in [0, max_cumulative_boost]
    pub fn compute_repetition_boost(&self, node_id: NodeId) -> f32;

    /// Reset exposure count for node (explicit forgetting)
    pub fn reset_exposures(&self, node_id: NodeId);
}
```

**Key Implementation Details:**
- Linear accumulation: boost = exposures × boost_per_repetition
- Hard ceiling at max_cumulative_boost (30%)
- Lock-free atomic counting
- No decay (persists across session)

### Integration Module

**File:** `/engram-core/src/cognitive/priming/mod.rs`

Export all priming types and provide unified interface:

```rust
pub mod semantic;
pub mod associative;
pub mod repetition;

pub use semantic::SemanticPrimingEngine;
pub use associative::AssociativePrimingEngine;
pub use repetition::RepetitionPrimingEngine;

/// Unified priming coordinator
pub struct PrimingCoordinator {
    semantic: SemanticPrimingEngine,
    associative: AssociativePrimingEngine,
    repetition: RepetitionPrimingEngine,
}

impl PrimingCoordinator {
    /// Compute total priming boost combining all three types
    pub fn compute_total_boost(&self, node_id: NodeId) -> f32 {
        let semantic = self.semantic.compute_priming_boost(node_id);
        let associative = self.associative.compute_association_strength(/* ... */);
        let repetition = self.repetition.compute_repetition_boost(node_id);

        // Additive combination (no multiplicative compounding)
        semantic + associative + repetition
    }
}
```

## Integration Points

### Existing Systems
- **M3 (Activation Spreading):** Priming boosts applied during spreading activation
  - File: `engram-core/src/activation/mod.rs`
  - Hook: Apply priming boost before spreading computation

- **M4 (Temporal Dynamics):** Decay functions for semantic priming only
  - File: `engram-core/src/decay/mod.rs`
  - Note: Associative and repetition priming do not decay within session

- **Metrics (Task 001):** Record all priming events
  - File: `engram-core/src/metrics/cognitive_patterns.rs`
  - Events: Associative priming formed, repetition exposure recorded

### Data Flow
1. Episode recalled → Record co-activation with other active nodes
2. Semantic spreading → Activates related nodes → Record exposures
3. Next recall → Apply combined priming boost from all three types

## Testing Strategy

### Unit Tests

**File:** `/engram-core/tests/cognitive/priming_integration_tests.rs`

#### Test 1: Associative Priming Formation
```rust
#[test]
fn test_associative_priming_through_cooccurrence() {
    let engine = AssociativePrimingEngine::default();

    // Co-activate "thunder" and "lightning" 5 times
    for _ in 0..5 {
        engine.record_coactivation(thunder_id, lightning_id);
    }

    // Check association strength
    let strength = engine.compute_association_strength(thunder_id, lightning_id);

    // Expected: P(lightning|thunder) = 5 / (thunder activations)
    assert!(strength > 0.3, "Associative priming should form after 5 co-occurrences");
}
```

#### Test 2: Repetition Priming Accumulation
```rust
#[test]
fn test_repetition_priming_accumulates_with_ceiling() {
    let engine = RepetitionPrimingEngine::default();

    // Expose node 10 times (should hit 30% ceiling after 6 exposures)
    for i in 0..10 {
        engine.record_exposure(node_id);
        let boost = engine.compute_repetition_boost(node_id);

        if i < 6 {
            assert_eq!(boost, (i + 1) as f32 * 0.05);
        } else {
            assert_eq!(boost, 0.30); // Ceiling reached
        }
    }
}
```

#### Test 3: Priming Types Don't Conflict
```rust
#[test]
fn test_priming_types_integrate_without_conflicts() {
    let coordinator = PrimingCoordinator::default();

    // Activate semantic priming (via spreading)
    coordinator.semantic.activate_priming(&episode, &graph);

    // Activate associative priming (via co-occurrence)
    coordinator.associative.record_coactivation(node_a, node_b);

    // Activate repetition priming (via exposure)
    coordinator.repetition.record_exposure(node_a);

    // Compute total boost (should be additive)
    let total_boost = coordinator.compute_total_boost(node_a);

    // Verify no interference between types
    assert!(total_boost > 0.0 && total_boost <= 1.0);
}
```

### Integration Tests

**Acceptance Criteria:**
1. Associative priming captures co-occurrence within 10-second window
2. Repetition priming provides exactly 5% boost per exposure
3. Repetition priming ceiling enforced at 30%
4. All three priming types integrate additively without conflicts
5. Metrics correctly distinguish between priming types

### Performance Requirements
- **Latency:** Co-occurrence recording <5μs (hot path)
- **Latency:** Repetition boost computation <2μs (hot path)
- **Memory:** Co-occurrence table <10MB for 1M node pairs
- **Memory:** Exposure counts <1MB for 10K nodes

## Acceptance Criteria

### Must Have
- [ ] Associative priming computes conditional probability correctly
- [ ] Repetition priming accumulates linearly with ceiling
- [ ] All three priming types work together without conflicts
- [ ] Metrics record associative and repetition events separately
- [ ] Unit tests pass for co-occurrence formation and repetition accumulation
- [ ] Integration test validates additive combination

### Should Have
- [ ] Co-occurrence pruning prevents unbounded memory growth
- [ ] Performance benchmarks validate <5μs latency for hot paths
- [ ] Lock-free atomic operations verified via loom

### Nice to Have
- [ ] Visualization of co-occurrence network
- [ ] Exposure decay over long timescales (beyond session)
- [ ] Configurable priming boost parameters

## Implementation Checklist

- [ ] Create `engram-core/src/cognitive/priming/associative.rs`
- [ ] Create `engram-core/src/cognitive/priming/repetition.rs`
- [ ] Update `engram-core/src/cognitive/priming/mod.rs` with exports
- [ ] Create `PrimingCoordinator` with additive combination logic
- [ ] Implement co-occurrence counting with atomic operations
- [ ] Implement repetition counting with ceiling enforcement
- [ ] Add metrics recording for associative and repetition events
- [ ] Write unit tests for associative priming formation
- [ ] Write unit tests for repetition priming accumulation
- [ ] Write integration test for all three priming types together
- [ ] Run `make quality` and fix all clippy warnings
- [ ] Verify performance benchmarks meet latency requirements

## Risks and Mitigations

**Risk 1:** Co-occurrence table grows unbounded
- **Mitigation:** Implement periodic pruning of entries below min_cooccurrence threshold
- **Mitigation:** Set maximum table size with LRU eviction

**Risk 2:** Priming types interfere or compound unexpectedly
- **Mitigation:** Use additive combination (not multiplicative)
- **Mitigation:** Integration test validates independent operation

**Risk 3:** Temporal window for co-occurrence too restrictive
- **Mitigation:** Make window configurable (default 10s from research)
- **Mitigation:** Log co-occurrence misses for tuning

## References

1. McKoon, G., & Ratcliff, R. (1992). Spreading activation versus compound cue accounts of priming. *Psychological Review*, 99(1), 177.
2. Tulving, E., & Schacter, D. L. (1990). Priming and human memory systems. *Science*, 247(4940), 301-306.

## Notes

- Associative priming differs from semantic priming: learned co-occurrence vs. inherent similarity
- Repetition priming is the simplest but most persistent form (no decay)
- All three types should combine additively to avoid over-priming
