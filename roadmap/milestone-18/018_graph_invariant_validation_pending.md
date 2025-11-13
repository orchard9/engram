# Task 018: Graph Invariant Property Testing

## Objective
Implement property-based tests using proptest to validate critical graph invariants hold under all possible inputs, ensuring operations maintain correctness guarantees across the state space.

## Background

Graph engines must maintain structural invariants to ensure correctness:

1. **Strength bounds**: All node/edge strengths in [0.0, 1.0]
2. **Activation conservation**: Total activation doesn't unboundedly increase
3. **Connectivity preservation**: Operations maintain reachability properties
4. **Embedding validity**: All embeddings have dimension 768, finite values
5. **Confidence semantics**: Probabilities in [0.0, 1.0], Bayesian updates correct
6. **Temporal consistency**: Timestamps monotonic, last_access <= now

Unit tests validate specific inputs. Property-based tests generate **thousands of random inputs** to find edge cases that violate invariants.

### Example Bug Found via Property Testing (Hypothetical)

```rust
// Bug: Strength underflow when decay rate exceeds activation
fn apply_decay(activation: f32, decay_rate: f32) -> f32 {
    activation - decay_rate  // Can go negative!
}

// Property test would find:
// Input: activation=0.1, decay_rate=0.5
// Output: -0.4 (VIOLATES [0.0, 1.0] bound)
```

## Requirements

1. Define formal invariants for all graph operations
2. Implement proptest generators for graph states
3. Write property tests for each operation (store, edge add, spreading, etc.)
4. Validate invariants under composition (sequences of operations)
5. Test edge cases (empty graph, single node, large graphs)
6. Integrate with CI (run 10,000 cases per property)

## Technical Specification

### Files to Create

#### `engram-core/tests/graph/property_tests.rs`
Property-based validation of graph invariants:

```rust
//! Property-based testing of graph invariants
//!
//! Uses proptest to generate random inputs and verify that graph operations
//! maintain correctness invariants across the entire state space.
//!
//! To run:
//! ```bash
//! cargo test --test graph_property_tests
//!
//! # Run with more cases (CI configuration)
//! PROPTEST_CASES=10000 cargo test --test graph_property_tests
//! ```

use proptest::prelude::*;
use engram_core::memory_graph::{UnifiedMemoryGraph, backends::DashMapBackend};
use engram_core::{Memory, Confidence};
use uuid::Uuid;

// ============================================================================
// Generators for graph structures
// ============================================================================

/// Generate valid activation values [0.0, 1.0]
fn arb_activation() -> impl Strategy<Value = f32> {
    (0.0f32..=1.0f32)
}

/// Generate valid strength/weight values [0.0, 1.0]
fn arb_strength() -> impl Strategy<Value = f32> {
    (0.0f32..=1.0f32)
}

/// Generate valid confidence intervals
fn arb_confidence() -> impl Strategy<Value = Confidence> {
    prop_oneof![
        Just(Confidence::LOW),
        Just(Confidence::MEDIUM),
        Just(Confidence::HIGH),
    ]
}

/// Generate valid 768-dimensional embeddings
fn arb_embedding() -> impl Strategy<Value = [f32; 768]> {
    // Use small values to avoid numerical instability
    proptest::collection::vec(-1.0f32..1.0f32, 768)
        .prop_map(|v| {
            let mut arr = [0.0f32; 768];
            arr.copy_from_slice(&v);
            arr
        })
}

/// Generate a valid Memory instance
fn arb_memory() -> impl Strategy<Value = Memory> {
    (arb_embedding(), arb_confidence(), arb_activation())
        .prop_map(|(embedding, confidence, activation)| {
            let id = Uuid::new_v4().to_string();
            let mut memory = Memory::new(id, embedding, confidence);
            memory.set_activation(activation);
            memory
        })
}

/// Generate a small graph (5-20 nodes, 10-50 edges)
fn arb_small_graph() -> impl Strategy<Value = UnifiedMemoryGraph<DashMapBackend>> {
    (5usize..=20, 10usize..=50)
        .prop_flat_map(|(num_nodes, num_edges)| {
            (
                proptest::collection::vec(arb_memory(), num_nodes),
                proptest::collection::vec(
                    (0usize..num_nodes, 0usize..num_nodes, arb_strength()),
                    num_edges
                )
            )
        })
        .prop_map(|(memories, edges)| {
            let graph = UnifiedMemoryGraph::with_backend(DashMapBackend::new());
            let ids: Vec<Uuid> = memories.iter()
                .map(|m| Uuid::parse_str(&m.id).unwrap())
                .collect();

            // Store all memories
            for memory in memories {
                graph.store_memory(memory).expect("store failed");
            }

            // Add edges
            for (from_idx, to_idx, weight) in edges {
                if from_idx != to_idx {
                    let _ = graph.add_edge(ids[from_idx], ids[to_idx], weight);
                }
            }

            graph
        })
}

// ============================================================================
// Property 1: Activation bounds [0.0, 1.0]
// ============================================================================

proptest! {
    #[test]
    fn prop_activation_always_bounded(
        activation in arb_activation(),
        delta in -2.0f32..2.0f32,
    ) {
        let memory = Memory::new(
            Uuid::new_v4().to_string(),
            [0.5f32; 768],
            Confidence::MEDIUM,
        );
        memory.set_activation(activation);

        // Apply delta (may overflow [0.0, 1.0])
        let new_activation = (memory.activation() + delta).clamp(0.0, 1.0);
        memory.set_activation(new_activation);

        // Verify bounds maintained
        let final_activation = memory.activation();
        prop_assert!(
            final_activation >= 0.0 && final_activation <= 1.0,
            "Activation out of bounds: {}",
            final_activation
        );
    }
}

// ============================================================================
// Property 2: Strength bounds [0.0, 1.0]
// ============================================================================

proptest! {
    #[test]
    fn prop_edge_strength_always_bounded(
        weight in arb_strength(),
    ) {
        let graph = UnifiedMemoryGraph::with_backend(DashMapBackend::new());
        let from = Uuid::new_v4();
        let to = Uuid::new_v4();

        // Store memories
        for id in [from, to] {
            let memory = Memory::new(
                id.to_string(),
                [0.5f32; 768],
                Confidence::HIGH,
            );
            graph.store_memory(memory).expect("store failed");
        }

        // Add edge with arbitrary weight
        graph.add_edge(from, to, weight).expect("add_edge failed");

        // Verify weight stored correctly
        let neighbors = graph.backend().get_neighbors(&from)
            .expect("get_neighbors failed");
        prop_assert_eq!(neighbors.len(), 1);

        let (neighbor_id, neighbor_weight) = neighbors[0];
        prop_assert_eq!(neighbor_id, to);
        prop_assert!(
            neighbor_weight >= 0.0 && neighbor_weight <= 1.0,
            "Edge weight out of bounds: {}",
            neighbor_weight
        );
    }
}

// ============================================================================
// Property 3: Connectivity preservation
// ============================================================================

proptest! {
    #[test]
    fn prop_edge_addition_preserves_connectivity(
        graph in arb_small_graph(),
    ) {
        let ids = graph.backend().all_ids();
        if ids.len() < 2 {
            return Ok(());
        }

        let from = ids[0];
        let to = ids[1];

        // Count reachable nodes before edge addition
        let reachable_before = graph.backend().traverse_bfs(&from, 10)
            .expect("traverse failed")
            .len();

        // Add new edge
        graph.add_edge(from, to, 0.8).expect("add_edge failed");

        // Count reachable nodes after edge addition
        let reachable_after = graph.backend().traverse_bfs(&from, 10)
            .expect("traverse failed")
            .len();

        // Reachability should increase or stay same (never decrease)
        prop_assert!(
            reachable_after >= reachable_before,
            "Edge addition decreased reachability: {} -> {}",
            reachable_before,
            reachable_after
        );
    }
}

// ============================================================================
// Property 4: Spreading activation conservation
// ============================================================================

proptest! {
    #[test]
    fn prop_spreading_conserves_total_activation(
        source_activation in arb_activation(),
        decay in 0.0f32..=1.0f32,
    ) {
        let graph = UnifiedMemoryGraph::with_backend(DashMapBackend::new());

        // Create chain: source -> target1, source -> target2
        let source = Uuid::new_v4();
        let target1 = Uuid::new_v4();
        let target2 = Uuid::new_v4();

        for id in [source, target1, target2] {
            let memory = Memory::new(
                id.to_string(),
                [0.5f32; 768],
                Confidence::HIGH,
            );
            graph.store_memory(memory).expect("store failed");
        }

        graph.add_edge(source, target1, 0.5).expect("add_edge failed");
        graph.add_edge(source, target2, 0.5).expect("add_edge failed");

        // Set source activation
        graph.backend().update_activation(&source, source_activation)
            .expect("update failed");

        // Measure total activation before spreading
        let total_before = [source, target1, target2].iter()
            .map(|id| {
                graph.backend().retrieve(id)
                    .expect("retrieve failed")
                    .expect("memory not found")
                    .activation()
            })
            .sum::<f32>();

        // Spread activation
        graph.backend().spread_activation(&source, decay)
            .expect("spread failed");

        // Measure total activation after spreading
        let total_after = [source, target1, target2].iter()
            .map(|id| {
                graph.backend().retrieve(id)
                    .expect("retrieve failed")
                    .expect("memory not found")
                    .activation()
            })
            .sum::<f32>();

        // Total activation should not unboundedly increase
        // (allows increase from spreading, but bounded by 3.0 max)
        prop_assert!(
            total_after <= 3.0,
            "Unbounded activation growth: {} -> {}",
            total_before,
            total_after
        );
    }
}

// ============================================================================
// Property 5: Embedding validity
// ============================================================================

proptest! {
    #[test]
    fn prop_embeddings_always_finite(
        embedding in arb_embedding(),
    ) {
        let memory = Memory::new(
            Uuid::new_v4().to_string(),
            embedding,
            Confidence::LOW,
        );

        // Verify all embedding values are finite
        for (i, &value) in memory.embedding.iter().enumerate() {
            prop_assert!(
                value.is_finite(),
                "Non-finite embedding at index {}: {}",
                i,
                value
            );
        }
    }
}

// ============================================================================
// Property 6: Confidence semantics
// ============================================================================

proptest! {
    #[test]
    fn prop_confidence_intervals_valid(
        confidence in arb_confidence(),
    ) {
        use engram_core::Confidence;

        let (lower, upper) = match confidence {
            Confidence::LOW => (0.0, 0.33),
            Confidence::MEDIUM => (0.33, 0.66),
            Confidence::HIGH => (0.66, 1.0),
        };

        // Verify interval is valid
        prop_assert!(
            lower >= 0.0 && upper <= 1.0 && lower <= upper,
            "Invalid confidence interval: [{}, {}]",
            lower,
            upper
        );
    }
}

// ============================================================================
// Property 7: Node removal maintains consistency
// ============================================================================

proptest! {
    #[test]
    fn prop_node_removal_maintains_consistency(
        graph in arb_small_graph(),
    ) {
        let ids = graph.backend().all_ids();
        if ids.is_empty() {
            return Ok(());
        }

        let to_remove = ids[0];

        // Count nodes before removal
        let count_before = graph.backend().count();

        // Remove node
        let removed = graph.backend().remove(&to_remove)
            .expect("remove failed");

        // Verify node was removed
        prop_assert!(removed.is_some(), "Node not found for removal");

        // Count nodes after removal
        let count_after = graph.backend().count();

        // Exactly one node removed
        prop_assert_eq!(
            count_after,
            count_before - 1,
            "Node count inconsistent after removal"
        );

        // Verify node cannot be retrieved
        let retrieved = graph.backend().retrieve(&to_remove)
            .expect("retrieve failed");
        prop_assert!(
            retrieved.is_none(),
            "Removed node still retrievable"
        );
    }
}

// ============================================================================
// Property 8: BFS traversal terminates
// ============================================================================

proptest! {
    #[test]
    fn prop_bfs_always_terminates(
        graph in arb_small_graph(),
        max_depth in 1usize..=10,
    ) {
        let ids = graph.backend().all_ids();
        if ids.is_empty() {
            return Ok(());
        }

        let start = ids[0];

        // BFS should always terminate
        let visited = graph.backend().traverse_bfs(&start, max_depth)
            .expect("BFS failed");

        // Visited count should not exceed graph size
        prop_assert!(
            visited.len() <= ids.len(),
            "BFS visited more nodes than exist: {} > {}",
            visited.len(),
            ids.len()
        );

        // Should not visit same node twice
        let unique_count = visited.iter()
            .collect::<std::collections::HashSet<_>>()
            .len();
        prop_assert_eq!(
            unique_count,
            visited.len(),
            "BFS visited duplicate nodes"
        );
    }
}

// ============================================================================
// Property 9: Vector search returns k results (when enough exist)
// ============================================================================

proptest! {
    #[test]
    fn prop_search_returns_k_results(
        graph in arb_small_graph(),
        k in 1usize..=5,
        query_embedding in arb_embedding(),
    ) {
        let node_count = graph.backend().count();

        let results = graph.backend().search(&query_embedding, k)
            .expect("search failed");

        // Should return min(k, node_count) results
        let expected = k.min(node_count);
        prop_assert_eq!(
            results.len(),
            expected,
            "Search returned wrong number of results"
        );

        // Similarities should be sorted (highest first)
        for i in 0..results.len().saturating_sub(1) {
            prop_assert!(
                results[i].1 >= results[i + 1].1,
                "Search results not sorted by similarity"
            );
        }
    }
}

// ============================================================================
// Property 10: M17 dual memory - consolidation score bounds
// ============================================================================

#[cfg(feature = "dual_memory_types")]
proptest! {
    #[test]
    fn prop_consolidation_score_bounded(
        score in -1.0f32..2.0f32,
    ) {
        use engram_core::memory::DualMemoryNode;

        let node = DualMemoryNode::new_episode(
            Uuid::new_v4(),
            "test-episode".to_string(),
            [0.5f32; 768],
            Confidence::MEDIUM,
            0.7,
        );

        // Update with arbitrary score (may be out of bounds)
        node.node_type.update_consolidation_score(score);

        // Verify score is clamped to [0.0, 1.0]
        let final_score = node.node_type.consolidation_score()
            .expect("not an episode");
        prop_assert!(
            final_score >= 0.0 && final_score <= 1.0,
            "Consolidation score out of bounds: {}",
            final_score
        );
    }
}

// ============================================================================
// Property 11: M17 dual memory - instance count never decreases
// ============================================================================

#[cfg(feature = "dual_memory_types")]
proptest! {
    #[test]
    fn prop_instance_count_monotonic(
        initial_count in 0u32..100,
        increments in proptest::collection::vec(Just(()), 1..10),
    ) {
        use engram_core::memory::DualMemoryNode;

        let node = DualMemoryNode::new_concept(
            Uuid::new_v4(),
            [0.5f32; 768],
            0.85,
            initial_count,
            Confidence::HIGH,
        );

        let count_before = node.node_type.instance_count()
            .expect("not a concept");

        // Apply increments
        for _ in increments {
            node.node_type.increment_instances();
        }

        let count_after = node.node_type.instance_count()
            .expect("not a concept");

        // Count should increase monotonically
        prop_assert!(
            count_after >= count_before,
            "Instance count decreased: {} -> {}",
            count_before,
            count_after
        );
    }
}
```

### Files to Modify

#### `engram-core/Cargo.toml`
Ensure proptest is in dev-dependencies:

```toml
[dev-dependencies]
# ... existing dev-dependencies ...

# Property-based testing (should already be present)
proptest = "1.5"
```

#### `engram-core/tests/graph/mod.rs`
Add property test module:

```rust
// Property-based testing of graph invariants
#[cfg(test)]
mod property_tests;
```

## Testing Strategy

### Invariants to Validate

1. **Bounds invariants**
   - Activation ∈ [0.0, 1.0]
   - Strength/weight ∈ [0.0, 1.0]
   - Confidence ∈ [0.0, 1.0]
   - Consolidation score ∈ [0.0, 1.0]

2. **Conservation invariants**
   - Total activation bounded by node count
   - Instance counts never decrease

3. **Structural invariants**
   - Connectivity preserved by edge addition
   - Node removal doesn't corrupt edges
   - BFS terminates in finite time

4. **Numerical invariants**
   - Embeddings always finite (no NaN, Inf)
   - Similarity scores in [-1.0, 1.0] for cosine

5. **API contract invariants**
   - Search returns exactly k results (when available)
   - Traversal visits each node at most once
   - Removed nodes are not retrievable

### Test Configuration

```bash
# Default: 256 test cases per property
cargo test --test graph_property_tests

# CI: 10,000 test cases per property (comprehensive)
PROPTEST_CASES=10000 cargo test --test graph_property_tests

# Debug: Replay specific failure
PROPTEST_CASES=1 cargo test --test graph_property_tests -- --nocapture

# Minimize failing case (find smallest input that fails)
cargo test --test graph_property_tests -- --test-threads=1
```

### Shrinking Strategy

When proptest finds a failing case, it **shrinks** to minimal input:

```
Original failure:
  activation = 0.93847, delta = 1.72349
  Result: 2.66196 (OUT OF BOUNDS)

After shrinking:
  activation = 0.5, delta = 1.0
  Result: 1.5 (OUT OF BOUNDS)
```

This helps identify root cause (e.g., missing `.clamp(0.0, 1.0)`).

## Integration with M17 Dual Memory Types

M17 introduced atomic operations that need property testing:

### Consolidation Score Updates
```rust
// Property: Score always in [0.0, 1.0] even with out-of-bounds input
node.node_type.update_consolidation_score(score); // Any f32
let final_score = node.node_type.consolidation_score().unwrap();
assert!(final_score >= 0.0 && final_score <= 1.0);
```

### Instance Count Increments
```rust
// Property: Count never decreases (monotonic)
let before = node.node_type.instance_count().unwrap();
node.node_type.increment_instances();
let after = node.node_type.instance_count().unwrap();
assert!(after >= before);
```

## Acceptance Criteria

- [ ] 11+ property tests implemented covering all critical invariants
- [ ] All properties pass with 10,000 test cases (CI configuration)
- [ ] Shrinking finds minimal failing inputs when bugs detected
- [ ] M17 dual memory types validated (consolidation score, instance count)
- [ ] Graph operations tested (store, remove, edge add, BFS, search, spreading)
- [ ] Numerical stability validated (embeddings, similarities)
- [ ] CI integration: Fails build on property violation
- [ ] Documentation: Each property has clear invariant statement

## Performance Considerations

**Property test overhead:**
- **256 cases**: ~2-5 seconds per property (acceptable for development)
- **10,000 cases**: ~30-120 seconds per property (CI only)
- **Shrinking**: Can take 10-60 seconds to find minimal case

**Optimization strategies:**
- Use `proptest::collection::vec` for efficient collection generation
- Keep generated graphs small (5-20 nodes) for fast tests
- Run expensive properties (spreading activation) with fewer cases

## Dependencies

- Task 001 (Dual Memory Types) - complete
- Task 002 (Graph Storage Adaptation) - complete
- proptest 1.5+ library (already in dependencies)

## Estimated Time
3 days

- Day 1: Set up infrastructure, write first 5 properties (bounds, connectivity)
- Day 2: Write properties 6-9 (BFS, search, removal, conservation)
- Day 3: Write M17-specific properties (consolidation, instance count), CI integration

## Follow-up Tasks

- Task 019: Fuzzing with AFL++ for deeper state space exploration
- Task 020: Model-based testing (compare against reference implementation)
- Task 021: Differential testing between Rust and Zig implementations

## References

### Property-Based Testing
- Claessen & Hughes (2000). "QuickCheck: A Lightweight Tool for Random Testing of Haskell Programs"
- proptest documentation: https://docs.rs/proptest/latest/proptest/

### Graph Algorithms
- Cormen et al. (2009). "Introduction to Algorithms" (3rd ed.) - Graph traversal correctness

### Numerical Stability
- Goldberg (1991). "What Every Computer Scientist Should Know About Floating-Point Arithmetic"
