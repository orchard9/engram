# Task 017: Graph Concurrent Correctness Validation

## Objective
Systematically validate lock-free graph operations using loom to explore all possible thread interleavings, ensuring atomic operations maintain correctness guarantees under concurrent access.

## Background

M17's DashMap-based graph backend uses lock-free operations for high-concurrency scenarios:
- **AtomicF32** for activation updates (Relaxed reads, Release writes)
- **DashMap** sharding for concurrent node/edge access
- **Arc<Memory>** for zero-copy reads during traversal
- **Atomic strength updates** during consolidation (consolidation_score, instance_count)

While unit tests validate sequential correctness, concurrent correctness requires systematic exploration of thread interleavings to find bugs like:
- **Lost updates**: Concurrent activation spreading loses contributions
- **ABA problems**: Node replacement during multi-step operations
- **Memory ordering bugs**: Relaxed atomics allow unexpected reorderings
- **Race conditions**: Offset allocation during consolidation

Task 017 from M17 found 3/7 failing concurrent tests through stress testing. This task uses **loom** to systematically explore interleavings and prove correctness.

## Requirements

1. Implement loom-based tests for all atomic operations in graph backend
2. Validate memory ordering semantics (Relaxed, Release, Acquire, AcqRel)
3. Verify progress guarantees (lock-freedom, wait-freedom)
4. Test ABA problem scenarios (node replacement during traversal)
5. Validate composite operations maintain atomicity
6. Document concurrency contracts for all public APIs

## Technical Specification

### Files to Create

#### `engram-core/tests/graph/loom_tests.rs`
Systematic concurrency testing of graph operations:

```rust
//! Loom-based verification of graph concurrent correctness
//!
//! These tests systematically explore thread interleavings to verify
//! lock-free operations maintain correctness under all possible schedules.
//!
//! To run:
//! ```bash
//! RUSTFLAGS="--cfg loom" cargo test --lib --test graph_loom_tests
//! ```

#![cfg(all(test, loom))]

use loom::sync::Arc;
use loom::thread;
use engram_core::memory_graph::{UnifiedMemoryGraph, backends::DashMapBackend};
use engram_core::{Memory, Confidence};
use uuid::Uuid;

/// Test 1: Concurrent activation updates don't lose contributions
#[test]
fn loom_concurrent_activation_updates() {
    loom::model(|| {
        let graph = Arc::new(UnifiedMemoryGraph::with_backend(
            DashMapBackend::new()
        ));

        // Store initial memory with zero activation
        let id = Uuid::new_v4();
        let memory = Memory::new(
            id.to_string(),
            [0.5f32; 768],
            Confidence::HIGH,
        );
        graph.store_memory(memory).expect("store failed");

        // Two threads concurrently update activation
        let handles: Vec<_> = (0..2)
            .map(|i| {
                let graph = Arc::clone(&graph);
                thread::spawn(move || {
                    // Each thread adds 0.1 to activation
                    let delta = 0.1;
                    let backend = graph.backend();

                    // Read-modify-write cycle (tests atomicity)
                    let memory = backend.retrieve(&id)
                        .expect("retrieve failed")
                        .expect("memory not found");
                    let current = memory.activation();
                    let new_activation = (current + delta).min(1.0);
                    backend.update_activation(&id, new_activation)
                        .expect("update failed");
                })
            })
            .collect();

        for h in handles {
            h.join().unwrap();
        }

        // After 2 threads each add 0.1, activation should be 0.2
        // (starting from 0.0, clamped to [0.0, 1.0])
        let memory = graph.backend().retrieve(&id)
            .expect("retrieve failed")
            .expect("memory not found");

        // CRITICAL: With atomic operations, we should see both contributions
        let final_activation = memory.activation();
        assert!(
            (final_activation - 0.2).abs() < 0.001,
            "Lost update detected: expected 0.2, got {}",
            final_activation
        );
    });
}

/// Test 2: Concurrent edge additions maintain consistency
#[test]
fn loom_concurrent_edge_additions() {
    loom::model(|| {
        let graph = Arc::new(UnifiedMemoryGraph::with_backend(
            DashMapBackend::new()
        ));

        let source = Uuid::new_v4();
        let target1 = Uuid::new_v4();
        let target2 = Uuid::new_v4();

        // Store memories
        for id in [source, target1, target2] {
            let memory = Memory::new(
                id.to_string(),
                [0.5f32; 768],
                Confidence::MEDIUM,
            );
            graph.store_memory(memory).expect("store failed");
        }

        // Two threads add edges from same source
        let handles: Vec<_> = [(target1, 0.7), (target2, 0.8)]
            .into_iter()
            .enumerate()
            .map(|(i, (target, weight))| {
                let graph = Arc::clone(&graph);
                thread::spawn(move || {
                    graph.add_edge(source, target, weight)
                        .expect("add_edge failed");
                })
            })
            .collect();

        for h in handles {
            h.join().unwrap();
        }

        // Both edges should be present
        let neighbors = graph.backend().get_neighbors(&source)
            .expect("get_neighbors failed");
        assert_eq!(neighbors.len(), 2, "Edge lost during concurrent insertion");

        // Check both targets present with correct weights
        let neighbor_ids: Vec<Uuid> = neighbors.iter().map(|(id, _)| *id).collect();
        assert!(neighbor_ids.contains(&target1));
        assert!(neighbor_ids.contains(&target2));
    });
}

/// Test 3: Spreading activation doesn't lose contributions under contention
#[test]
fn loom_spreading_activation_atomicity() {
    loom::model(|| {
        let graph = Arc::new(UnifiedMemoryGraph::with_backend(
            DashMapBackend::new()
        ));

        // Create chain: source -> middle1, source -> middle2
        let source = Uuid::new_v4();
        let middle1 = Uuid::new_v4();
        let middle2 = Uuid::new_v4();

        for id in [source, middle1, middle2] {
            let memory = Memory::new(
                id.to_string(),
                [0.5f32; 768],
                Confidence::HIGH,
            );
            graph.store_memory(memory).expect("store failed");
        }

        graph.add_edge(source, middle1, 0.9).expect("add_edge failed");
        graph.add_edge(source, middle2, 0.9).expect("add_edge failed");

        // Set source activation to 1.0
        graph.backend().update_activation(&source, 1.0)
            .expect("update failed");

        // Two threads spread activation from source concurrently
        let handles: Vec<_> = (0..2)
            .map(|_| {
                let graph = Arc::clone(&graph);
                thread::spawn(move || {
                    graph.backend().spread_activation(&source, 0.9)
                        .expect("spread failed");
                })
            })
            .collect();

        for h in handles {
            h.join().unwrap();
        }

        // Both middle nodes should receive activation
        // Expected: source (1.0) * weight (0.9) * decay (0.9) = 0.81
        // With 2 spreading ops: depends on atomic accumulation semantics
        for middle in [middle1, middle2] {
            let memory = graph.backend().retrieve(&middle)
                .expect("retrieve failed")
                .expect("memory not found");
            let activation = memory.activation();

            // Should receive at least one contribution
            assert!(
                activation > 0.0,
                "Activation not propagated to {}",
                middle
            );
        }
    });
}

/// Test 4: Node removal during traversal doesn't cause use-after-free
#[test]
fn loom_removal_during_traversal() {
    loom::model(|| {
        let graph = Arc::new(UnifiedMemoryGraph::with_backend(
            DashMapBackend::new()
        ));

        let node1 = Uuid::new_v4();
        let node2 = Uuid::new_v4();

        for id in [node1, node2] {
            let memory = Memory::new(
                id.to_string(),
                [0.5f32; 768],
                Confidence::LOW,
            );
            graph.store_memory(memory).expect("store failed");
        }

        graph.add_edge(node1, node2, 0.5).expect("add_edge failed");

        let graph_clone = Arc::clone(&graph);

        // Thread 1: Traverse BFS
        let t1 = thread::spawn(move || {
            let result = graph_clone.backend().traverse_bfs(&node1, 1);
            // Should either succeed or gracefully handle removal
            result.is_ok()
        });

        // Thread 2: Remove node2 during traversal
        let t2 = thread::spawn(move || {
            let _ = graph.backend().remove(&node2);
        });

        let r1 = t1.join().unwrap();
        let r2 = t2.join().unwrap();

        // Both operations should complete without panic
        // Traversal may or may not see node2 depending on interleaving
        assert!(r1, "BFS panicked during concurrent removal");
    });
}

/// Test 5: DualMemoryNode consolidation score updates are atomic
#[cfg(feature = "dual_memory_types")]
#[test]
fn loom_consolidation_score_atomic() {
    use engram_core::memory::DualMemoryNode;

    loom::model(|| {
        let node = Arc::new(DualMemoryNode::new_episode(
            Uuid::new_v4(),
            "test-episode".to_string(),
            [0.5f32; 768],
            Confidence::MEDIUM,
            0.7,
        ));

        // Two consolidation workers update score concurrently
        let handles: Vec<_> = [0.3, 0.5]
            .into_iter()
            .map(|score| {
                let node = Arc::clone(&node);
                thread::spawn(move || {
                    node.node_type.update_consolidation_score(score);
                })
            })
            .collect();

        for h in handles {
            h.join().unwrap();
        }

        // Final score should be one of the two values (last write wins)
        let final_score = node.node_type.consolidation_score()
            .expect("not an episode");
        assert!(
            (final_score - 0.3).abs() < 0.001 || (final_score - 0.5).abs() < 0.001,
            "Invalid consolidation score: {}",
            final_score
        );
    });
}

/// Test 6: Concept instance count increments are atomic
#[cfg(feature = "dual_memory_types")]
#[test]
fn loom_instance_count_atomic() {
    use engram_core::memory::DualMemoryNode;

    loom::model(|| {
        let node = Arc::new(DualMemoryNode::new_concept(
            Uuid::new_v4(),
            [0.5f32; 768],
            0.85,
            0, // Start at zero
            Confidence::HIGH,
        ));

        // Three binding workers increment count concurrently
        let handles: Vec<_> = (0..3)
            .map(|_| {
                let node = Arc::clone(&node);
                thread::spawn(move || {
                    node.node_type.increment_instances();
                })
            })
            .collect();

        for h in handles {
            h.join().unwrap();
        }

        // Final count should be exactly 3 (no lost increments)
        let final_count = node.node_type.instance_count()
            .expect("not a concept");
        assert_eq!(
            final_count, 3,
            "Lost increment detected: expected 3, got {}",
            final_count
        );
    });
}
```

### Files to Modify

#### `engram-core/Cargo.toml`
Add loom dependency for concurrency testing:

```toml
[dev-dependencies]
# ... existing dev-dependencies ...

# Systematic concurrency testing
loom = { version = "0.7", features = ["checkpoint"] }
```

#### `engram-core/tests/graph/mod.rs`
Add loom test module:

```rust
// Concurrency correctness validation
#[cfg(all(test, loom))]
mod loom_tests;
```

### Integration with Existing Tests

The loom tests complement existing concurrent tests:

1. **Task 017 stress tests** (warm_tier_concurrent_tests.rs): Find bugs through random interleavings
2. **Loom tests** (this task): Systematically prove absence of bugs for small thread counts

Strategy:
- Run stress tests (100+ threads) to find empirical bugs
- Write loom tests (2-3 threads) to prove fixes are correct
- Document concurrency contracts discovered through testing

## Memory Ordering Analysis

### Current Memory Ordering in DashMap Backend

```rust
// dashmap.rs:109 - Relaxed ordering for reads
cached.store(activation.clamp(0.0, 1.0), Ordering::Relaxed);

// dashmap.rs:198 - Relaxed ordering in CAS loop
let current = activation.load(Ordering::Relaxed);
// ...
let new_activation = (current + contribution).min(1.0);
```

**Analysis:**
- **Relaxed reads**: Safe for activation reads (no causality required)
- **Relaxed writes**: Safe for activation updates (no synchronization needed)
- **CAS loops**: Should use `compare_exchange` with Acquire/Release for proper synchronization

**Potential Bug:**
The CAS loop in `spread_activation` (lines 197-210) uses `Relaxed` ordering, which could allow reordering of updates. Should use `AcqRel` for the success case:

```rust
loop {
    let current = activation.load(Ordering::Acquire);
    let contribution = source_activation * weight * decay;
    let new_activation = (current + contribution).min(1.0);

    match activation.compare_exchange_weak(
        current,
        new_activation,
        Ordering::AcqRel, // Success: release write, acquire read
        Ordering::Relaxed, // Failure: retry
    ) {
        Ok(_) => break,
        Err(_) => continue,
    }
}
```

**Loom Test Validation:**
The `loom_spreading_activation_atomicity` test will detect this bug by exploring interleavings where Relaxed ordering causes lost updates.

## ABA Problem Analysis

### Scenario: Node Replacement During Traversal

```rust
// Thread 1: Traversing graph
let node = graph.retrieve(&id)?; // Gets Arc<Memory>

// Thread 2: Removes and re-inserts node with same UUID
graph.remove(&id)?;
graph.store_memory(new_memory_with_same_id)?;

// Thread 1: Continues using old Arc<Memory>
// ABA: Node replaced, but Thread 1 still has old pointer
let activation = node.activation(); // Reading stale data
```

**Mitigation:**
DashMap + Arc prevents classic ABA problem:
- `retrieve()` returns `Arc<Memory>`, incrementing refcount
- Even if node removed, Arc keeps data alive until Thread 1 drops it
- Thread 1 reads consistent snapshot, not torn state

**Loom Test:**
`loom_removal_during_traversal` validates that concurrent removal doesn't cause use-after-free.

## Testing Approach

### Loom Configuration

```rust
// Small thread counts for tractable state space
loom::model(|| {
    // Test logic with 2-3 threads
});
```

**State space explosion:**
- 2 threads, 5 operations each: ~3,000 interleavings
- 3 threads, 5 operations each: ~3,000,000 interleavings
- Keep tests small: 2-3 threads, <10 operations per thread

### Test Selection Strategy

1. **Critical operations**: Activation updates, edge additions, node removal
2. **Composite operations**: Spreading activation (multi-step)
3. **M17 additions**: Consolidation score, instance count (dual memory types)
4. **Known failure modes**: Task 017 found 3 concurrent bugs - write loom tests for root causes

### Running Loom Tests

```bash
# Run all loom tests (slow - explores all interleavings)
RUSTFLAGS="--cfg loom" cargo test --lib --test graph_loom_tests

# Run specific test
RUSTFLAGS="--cfg loom" cargo test --lib loom_concurrent_activation_updates

# With checkpointing (faster retries)
RUSTFLAGS="--cfg loom" cargo test --lib --features loom/checkpoint
```

## Acceptance Criteria

- [ ] All 6+ loom tests pass (100% interleaving coverage for 2-3 threads)
- [ ] Memory ordering validated: Relaxed/Release/Acquire/AcqRel used correctly
- [ ] ABA problem mitigated: Arc + DashMap prevents use-after-free
- [ ] Progress guarantees documented: Lock-free for all operations
- [ ] Composite operations validated: Spreading activation maintains atomicity
- [ ] M17 dual memory atomics tested: consolidation_score, instance_count
- [ ] Concurrency contracts documented: Safety conditions for all public APIs
- [ ] CI integration: Loom tests run on every PR (use `cargo nextest` for parallelism)

## Performance Considerations

**Loom overhead:**
- Tests run **1000-10000x slower** than normal tests
- State space grows **exponentially** with thread count
- Use small thread counts (2-3) and short sequences (<10 operations)

**When to use loom vs stress testing:**
- **Loom**: Prove correctness for critical operations (activation updates, edge additions)
- **Stress testing**: Find bugs in complex scenarios (100+ threads, long sequences)
- **Both**: Write stress test first to find bug, then loom test to prove fix

## Dependencies

- Task 001 (Dual Memory Types) - complete
- Task 002 (Graph Storage Adaptation) - complete
- Task 017 (Warm Tier Concurrent Tests) - complete (found bugs to analyze)
- loom 0.7+ library

## Estimated Time
3-4 days

- Day 1: Set up loom infrastructure, write first 3 tests (activation, edges, spreading)
- Day 2: Write M17-specific tests (consolidation, instance count), analyze memory ordering
- Day 3: Write ABA tests, document concurrency contracts
- Day 4: CI integration, fix any bugs found, documentation

## Follow-up Tasks

- Task 018: Property-based testing of graph invariants (strength bounds, connectivity)
- Task 019: ThreadSanitizer validation of memory ordering
- Task 020: Miri validation of unsafe code in atomic operations

## References

### Concurrency Testing
- Kokologiannakis et al. (2019). "Effective Stateless Model Checking for C/C++ Concurrency"
- Loom documentation: https://docs.rs/loom/latest/loom/

### Memory Ordering
- Sewell et al. (2010). "x86-TSO: A Rigorous and Usable Programmer's Model"
- Rust Nomicon: https://doc.rust-lang.org/nomicon/atomics.html

### Lock-Free Algorithms
- Herlihy & Shavit (2008). "The Art of Multiprocessor Programming" - Chapter 3 (Progress guarantees)
- Michael & Scott (1996). "Simple, Fast, and Practical Non-Blocking Algorithms"
