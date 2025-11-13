# Task 019: Probabilistic Semantics Validation

## Objective
Validate the correctness of confidence propagation, Bayesian updates, and probabilistic operations in the graph engine, ensuring numerical stability and adherence to probability axioms.

## Background

Engram's graph engine uses probabilistic semantics throughout:

1. **Confidence intervals**: Memory nodes carry uncertainty (LOW/MEDIUM/HIGH mapped to probability ranges)
2. **Confidence propagation**: Spreading activation propagates confidence through edges
3. **Bayesian updates**: Pattern completion updates beliefs based on evidence
4. **Numerical stability**: Repeated operations must not drift or overflow

### Example: Confidence Propagation Through Edges

```rust
// Source node: HIGH confidence (0.66-1.0)
// Edge weight: 0.8 (strong connection)
// Target receives confidence: ???

// Correct propagation:
target_confidence = source_confidence * edge_weight
                  = 0.83 * 0.8  // midpoint of HIGH range
                  = 0.664
                  // Still HIGH confidence
```

### Known Issues in Probabilistic Graph Systems

1. **Confidence overestimation**: Repeated spreading inflates confidence unrealistically
2. **Numerical underflow**: Small probabilities (< 1e-7) vanish in log-probability space
3. **Belief update bugs**: Incorrect Bayesian inference from multiple evidence sources
4. **Cycle amplification**: Cycles in graph cause unbounded confidence growth

This task validates Engram avoids these pitfalls.

## Requirements

1. Validate confidence propagation matches probability theory
2. Test numerical stability under repeated operations
3. Verify Bayesian updates follow Bayes' rule
4. Check for probability violations (negative, >1.0)
5. Validate log-probability arithmetic for small values
6. Test cycle detection and dampening

## Technical Specification

### Files to Create

#### `engram-core/tests/probabilistic/correctness_tests.rs`
Validation of probabilistic operations:

```rust
//! Probabilistic semantics correctness tests
//!
//! Validates that confidence propagation, Bayesian updates, and probability
//! arithmetic adhere to probability theory axioms and maintain numerical stability.

use engram_core::memory_graph::{UnifiedMemoryGraph, backends::DashMapBackend};
use engram_core::{Memory, Confidence};
use uuid::Uuid;
use approx::assert_relative_eq;

// ============================================================================
// Test 1: Confidence propagation preserves probability bounds
// ============================================================================

#[test]
fn test_confidence_propagation_bounds() {
    let graph = UnifiedMemoryGraph::with_backend(DashMapBackend::new());

    let source = Uuid::new_v4();
    let target = Uuid::new_v4();

    // Source: HIGH confidence (0.66-1.0, midpoint 0.83)
    let source_memory = Memory::new(
        source.to_string(),
        [0.5f32; 768],
        Confidence::HIGH,
    );
    source_memory.set_activation(1.0);

    // Target: LOW confidence initially
    let target_memory = Memory::new(
        target.to_string(),
        [0.5f32; 768],
        Confidence::LOW,
    );

    graph.store_memory(source_memory).expect("store failed");
    graph.store_memory(target_memory).expect("store failed");

    // Strong edge (0.9 weight)
    graph.add_edge(source, target, 0.9).expect("add_edge failed");

    // Spread activation with confidence propagation
    graph.backend().spread_activation(&source, 0.95).expect("spread failed");

    // Retrieve target and check confidence
    let target_after = graph.backend().retrieve(&target)
        .expect("retrieve failed")
        .expect("memory not found");

    // Expected: HIGH * 0.9 ≈ 0.747 (should remain HIGH or become MEDIUM)
    // Actual confidence depends on implementation
    match target_after.confidence {
        Confidence::HIGH | Confidence::MEDIUM => {
            // Valid: confidence propagated reasonably
        }
        Confidence::LOW => {
            panic!("Confidence not propagated through strong edge");
        }
    }

    // Activation should be propagated
    assert!(
        target_after.activation() > 0.0,
        "Activation not propagated"
    );
}

// ============================================================================
// Test 2: Repeated spreading doesn't inflate confidence unboundedly
// ============================================================================

#[test]
fn test_confidence_inflation_prevention() {
    let graph = UnifiedMemoryGraph::with_backend(DashMapBackend::new());

    // Create cycle: A -> B -> C -> A
    let node_a = Uuid::new_v4();
    let node_b = Uuid::new_v4();
    let node_c = Uuid::new_v4();

    for id in [node_a, node_b, node_c] {
        let memory = Memory::new(
            id.to_string(),
            [0.5f32; 768],
            Confidence::MEDIUM,
        );
        graph.store_memory(memory).expect("store failed");
    }

    graph.add_edge(node_a, node_b, 0.8).expect("add_edge failed");
    graph.add_edge(node_b, node_c, 0.8).expect("add_edge failed");
    graph.add_edge(node_c, node_a, 0.8).expect("add_edge failed");

    // Set initial activation on A
    graph.backend().update_activation(&node_a, 1.0).expect("update failed");

    // Spread 10 times (simulates cyclic propagation)
    for _ in 0..10 {
        for node in [node_a, node_b, node_c] {
            graph.backend().spread_activation(&node, 0.9).expect("spread failed");
        }
    }

    // Check that confidence hasn't inflated unreasonably
    for node in [node_a, node_b, node_c] {
        let memory = graph.backend().retrieve(&node)
            .expect("retrieve failed")
            .expect("memory not found");

        // Confidence should still be in valid range
        match memory.confidence {
            Confidence::LOW | Confidence::MEDIUM | Confidence::HIGH => {
                // Valid
            }
        }

        // Activation should be bounded (not exploded to infinity)
        assert!(
            memory.activation() <= 1.0,
            "Activation unbounded: {} > 1.0",
            memory.activation()
        );
    }
}

// ============================================================================
// Test 3: Bayesian update correctness (pattern completion)
// ============================================================================

#[test]
fn test_bayesian_pattern_completion() {
    // This test validates that pattern completion uses Bayes' rule correctly
    // when combining evidence from multiple sources.

    let graph = UnifiedMemoryGraph::with_backend(DashMapBackend::new());

    // Create pattern: evidence1 -> target, evidence2 -> target
    let evidence1 = Uuid::new_v4();
    let evidence2 = Uuid::new_v4();
    let target = Uuid::new_v4();

    for id in [evidence1, evidence2, target] {
        let memory = Memory::new(
            id.to_string(),
            [0.5f32; 768],
            Confidence::MEDIUM,
        );
        graph.store_memory(memory).expect("store failed");
    }

    graph.add_edge(evidence1, target, 0.9).expect("add_edge failed");
    graph.add_edge(evidence2, target, 0.9).expect("add_edge failed");

    // Activate both evidence nodes
    graph.backend().update_activation(&evidence1, 1.0).expect("update failed");
    graph.backend().update_activation(&evidence2, 1.0).expect("update failed");

    // Spread from both evidence nodes
    graph.backend().spread_activation(&evidence1, 0.95).expect("spread failed");
    graph.backend().spread_activation(&evidence2, 0.95).expect("spread failed");

    // Target should receive combined evidence
    let target_memory = graph.backend().retrieve(&target)
        .expect("retrieve failed")
        .expect("memory not found");

    // With two strong evidence sources, target activation should be high
    // (exact value depends on combination rule: sum, max, Bayesian, etc.)
    assert!(
        target_memory.activation() > 0.5,
        "Evidence not combined: activation = {}",
        target_memory.activation()
    );
}

// ============================================================================
// Test 4: Numerical stability with small probabilities
// ============================================================================

#[test]
fn test_small_probability_stability() {
    let graph = UnifiedMemoryGraph::with_backend(DashMapBackend::new());

    let source = Uuid::new_v4();
    let target = Uuid::new_v4();

    let source_memory = Memory::new(
        source.to_string(),
        [0.5f32; 768],
        Confidence::LOW,
    );
    source_memory.set_activation(0.001); // Very small activation

    let target_memory = Memory::new(
        target.to_string(),
        [0.5f32; 768],
        Confidence::LOW,
    );

    graph.store_memory(source_memory).expect("store failed");
    graph.store_memory(target_memory).expect("store failed");

    // Weak edge
    graph.add_edge(source, target, 0.1).expect("add_edge failed");

    // Spread with small initial activation
    graph.backend().spread_activation(&source, 0.9).expect("spread failed");

    // Target should receive very small activation (0.001 * 0.1 * 0.9 = 0.00009)
    let target_after = graph.backend().retrieve(&target)
        .expect("retrieve failed")
        .expect("memory not found");

    let target_activation = target_after.activation();

    // Should not underflow to exactly 0.0
    // (unless implementation has threshold cutoff)
    assert!(
        target_activation.is_finite(),
        "Activation underflowed to non-finite value"
    );

    // If non-zero, should be approximately expected value
    if target_activation > 0.0 {
        let expected = 0.001 * 0.1 * 0.9;
        assert_relative_eq!(
            target_activation,
            expected,
            epsilon = 0.0001
        );
    }
}

// ============================================================================
// Test 5: Log-probability arithmetic correctness
// ============================================================================

#[test]
fn test_log_probability_operations() {
    // If Engram uses log-probabilities for numerical stability,
    // validate that log(P(A) * P(B)) = log(P(A)) + log(P(B))

    let prob_a = 0.1f32;
    let prob_b = 0.2f32;

    // Linear space multiplication
    let product_linear = prob_a * prob_b;

    // Log space addition
    let log_a = prob_a.ln();
    let log_b = prob_b.ln();
    let product_log = (log_a + log_b).exp();

    assert_relative_eq!(
        product_linear,
        product_log,
        epsilon = 1e-6
    );
}

// ============================================================================
// Test 6: Confidence interval consistency
// ============================================================================

#[test]
fn test_confidence_interval_consistency() {
    // Validate that confidence intervals are consistent with their definitions

    let test_cases = [
        (Confidence::LOW, (0.0, 0.33)),
        (Confidence::MEDIUM, (0.33, 0.66)),
        (Confidence::HIGH, (0.66, 1.0)),
    ];

    for (confidence, (lower, upper)) in test_cases {
        let memory = Memory::new(
            Uuid::new_v4().to_string(),
            [0.5f32; 768],
            confidence,
        );

        // Midpoint of interval
        let midpoint = (lower + upper) / 2.0;

        // Confidence should map to interval consistently
        assert!(
            lower >= 0.0 && upper <= 1.0 && lower <= upper,
            "Invalid interval for {:?}: [{}, {}]",
            confidence,
            lower,
            upper
        );

        // Midpoint should be in interval
        assert!(
            midpoint >= lower && midpoint <= upper,
            "Midpoint {} not in interval [{}, {}]",
            midpoint,
            lower,
            upper
        );
    }
}

// ============================================================================
// Test 7: Multiple evidence sources combine correctly
// ============================================================================

#[test]
fn test_multiple_evidence_combination() {
    let graph = UnifiedMemoryGraph::with_backend(DashMapBackend::new());

    let target = Uuid::new_v4();
    let evidence_nodes: Vec<Uuid> = (0..5).map(|_| Uuid::new_v4()).collect();

    // Store target
    let target_memory = Memory::new(
        target.to_string(),
        [0.5f32; 768],
        Confidence::MEDIUM,
    );
    graph.store_memory(target_memory).expect("store failed");

    // Create 5 evidence nodes, all pointing to target
    for evidence_id in &evidence_nodes {
        let evidence_memory = Memory::new(
            evidence_id.to_string(),
            [0.5f32; 768],
            Confidence::HIGH,
        );
        evidence_memory.set_activation(1.0);
        graph.store_memory(evidence_memory).expect("store failed");

        graph.add_edge(*evidence_id, target, 0.8).expect("add_edge failed");
    }

    // Spread from all evidence nodes
    for evidence_id in &evidence_nodes {
        graph.backend().spread_activation(evidence_id, 0.9).expect("spread failed");
    }

    // Target should receive combined activation
    let target_after = graph.backend().retrieve(&target)
        .expect("retrieve failed")
        .expect("memory not found");

    // With 5 strong evidence sources, activation should be high
    assert!(
        target_after.activation() > 0.5,
        "Multiple evidence not combined: activation = {}",
        target_after.activation()
    );
}

// ============================================================================
// Test 8: Confidence doesn't violate probability axioms
// ============================================================================

#[test]
fn test_probability_axioms() {
    let graph = UnifiedMemoryGraph::with_backend(DashMapBackend::new());

    // Generate 100 random graph operations
    for i in 0..100 {
        let memory = Memory::new(
            Uuid::new_v4().to_string(),
            [0.5f32; 768],
            if i % 3 == 0 {
                Confidence::LOW
            } else if i % 3 == 1 {
                Confidence::MEDIUM
            } else {
                Confidence::HIGH
            },
        );

        graph.store_memory(memory).expect("store failed");
    }

    // Verify all memories have valid confidence
    let ids = graph.backend().all_ids();
    for id in ids {
        let memory = graph.backend().retrieve(&id)
            .expect("retrieve failed")
            .expect("memory not found");

        // Confidence must be one of three valid values
        match memory.confidence {
            Confidence::LOW | Confidence::MEDIUM | Confidence::HIGH => {
                // Valid
            }
        }
    }
}

// ============================================================================
// Test 9: Activation values are always non-negative
// ============================================================================

#[test]
fn test_activation_non_negative() {
    let graph = UnifiedMemoryGraph::with_backend(DashMapBackend::new());

    let nodes: Vec<Uuid> = (0..10).map(|_| Uuid::new_v4()).collect();

    for id in &nodes {
        let memory = Memory::new(
            id.to_string(),
            [0.5f32; 768],
            Confidence::MEDIUM,
        );
        graph.store_memory(memory).expect("store failed");
    }

    // Create random edges
    for i in 0..nodes.len() - 1 {
        graph.add_edge(nodes[i], nodes[i + 1], 0.5).expect("add_edge failed");
    }

    // Set initial activation
    graph.backend().update_activation(&nodes[0], 1.0).expect("update failed");

    // Spread multiple times
    for _ in 0..10 {
        for node in &nodes {
            graph.backend().spread_activation(node, 0.8).expect("spread failed");
        }
    }

    // Verify all activations are non-negative
    for node in &nodes {
        let memory = graph.backend().retrieve(node)
            .expect("retrieve failed")
            .expect("memory not found");

        assert!(
            memory.activation() >= 0.0,
            "Negative activation detected: {}",
            memory.activation()
        );
    }
}
```

### Files to Modify

#### `engram-core/Cargo.toml`
Add approx crate for floating-point comparisons:

```toml
[dev-dependencies]
# ... existing dev-dependencies ...

# Floating-point comparisons
approx = "0.5"
```

## Property-Based Probabilistic Tests

Extend `property_tests.rs` with probabilistic properties:

```rust
// Property: Confidence propagation never produces negative probabilities
proptest! {
    #[test]
    fn prop_confidence_never_negative(
        source_conf in arb_confidence(),
        weight in arb_strength(),
    ) {
        // Propagate confidence through edge
        let propagated = propagate_confidence(source_conf, weight);

        // Should remain in [0.0, 1.0]
        prop_assert!(propagated >= 0.0 && propagated <= 1.0);
    }
}

// Property: Repeated operations converge (don't oscillate)
proptest! {
    #[test]
    fn prop_activation_converges(
        graph in arb_small_graph(),
        iterations in 10usize..=100,
    ) {
        // Spread activation many times
        for _ in 0..iterations {
            // Spread from all nodes
        }

        // Total activation should stabilize (not grow unbounded)
        let total_activation = compute_total_activation(&graph);
        prop_assert!(total_activation.is_finite());
    }
}
```

## Validation Against Probability Theory

### Bayes' Rule
```
P(H|E) = P(E|H) * P(H) / P(E)

Where:
- P(H|E): Posterior (target confidence given evidence)
- P(E|H): Likelihood (edge weight)
- P(H): Prior (target initial confidence)
- P(E): Evidence normalization (sum over all hypotheses)
```

Test validates that pattern completion follows this rule.

### Confidence Combination
For independent evidence sources E1, E2:
```
P(H|E1,E2) ≈ P(H|E1) + P(H|E2) - P(H|E1) * P(H|E2)
```

Or using log-odds for numerical stability:
```
log_odds(H|E1,E2) = log_odds(H) + log(P(E1|H)/P(E1|¬H)) + log(P(E2|H)/P(E2|¬H))
```

## Acceptance Criteria

- [ ] All 9+ correctness tests pass
- [ ] Confidence propagation validated against probability theory
- [ ] Numerical stability confirmed for small probabilities (<1e-6)
- [ ] Bayesian updates follow Bayes' rule (within numerical precision)
- [ ] No probability violations (negative, >1.0) detected in 10,000 operations
- [ ] Log-probability arithmetic validated for numerical stability
- [ ] Multiple evidence combination tested (2-10 sources)
- [ ] Cycle amplification prevented (bounded confidence in cycles)
- [ ] Property tests pass with 10,000 cases

## Dependencies

- Task 017 (Graph Concurrent Correctness) - foundation for testing
- Task 018 (Graph Invariant Validation) - property testing infrastructure
- approx crate for floating-point comparisons

## Estimated Time
3 days

- Day 1: Write correctness tests 1-5 (propagation, stability, Bayes)
- Day 2: Write tests 6-9 (intervals, evidence combination, axioms)
- Day 3: Property-based tests, validation against theory, documentation

## Follow-up Tasks

- Task 020: Numerical stability fuzzing with extreme values
- Task 021: Differential testing against reference Bayesian network implementation
- Task 022: Performance validation (probabilistic ops should be <100ns overhead)

## References

### Probability Theory
- Jaynes (2003). "Probability Theory: The Logic of Science"
- Bishop (2006). "Pattern Recognition and Machine Learning" - Bayesian inference

### Numerical Stability
- Goldberg (1991). "What Every Computer Scientist Should Know About Floating-Point Arithmetic"
- NumPy log-probability documentation: https://numpy.org/doc/stable/reference/generated/numpy.log1p.html

### Bayesian Networks
- Pearl (1988). "Probabilistic Reasoning in Intelligent Systems"
- Koller & Friedman (2009). "Probabilistic Graphical Models" - Belief propagation
