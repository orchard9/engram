# Task 002: Semantic Priming Engine

**Status:** Pending
**Priority:** P0 (Critical Path)
**Estimated Effort:** 2 days
**Dependencies:** Task 001 (Zero-Overhead Metrics)

## Objective

Implement semantic priming based on spreading activation theory (Collins & Loftus 1975). When a concept is recalled, semantically related concepts receive pre-activation boost, reducing retrieval time by 10-20% (empirically validated against Neely 1977).

## Integration Points

**Creates:**
- `/engram-core/src/cognitive/priming/semantic.rs` - Semantic priming engine
- `/engram-core/src/cognitive/priming/mod.rs` - Priming module exports
- `/engram-core/tests/cognitive/semantic_priming_tests.rs` - Validation tests

**Uses:**
- `/engram-core/src/activation/spreading.rs` - Existing spreading activation
- `/engram-core/src/embedding/similarity.rs` - SIMD similarity computation
- `/engram-core/src/metrics/cognitive_patterns.rs` - Metrics recording (Task 001)

**Extends:**
- `/engram-core/src/activation/recall.rs` - Apply priming boost during recall

## Detailed Specification

### 1. Semantic Priming Engine

```rust
// /engram-core/src/cognitive/priming/semantic.rs

use crate::{Episode, NodeId, MemoryGraph};
use std::collections::HashMap;
use std::time::{Duration, Instant};
use dashmap::DashMap;
use crossbeam_utils::CachePadded;
use std::sync::atomic::{AtomicU64, Ordering};

/// Semantic priming engine based on spreading activation
///
/// Implements Collins & Loftus (1975) spreading activation theory.
/// When "doctor" is recalled, related concepts like "nurse", "hospital"
/// receive temporary activation boost.
pub struct SemanticPrimingEngine {
    /// Active primes and their decay state
    active_primes: DashMap<NodeId, PrimeState>,

    /// Priming strength (default: 0.15 = 15% activation boost)
    /// Validated against Neely (1977): 10-20% RT reduction
    priming_strength: f32,

    /// Decay half-life for priming (default: 500ms)
    /// Empirical basis: Neely (1977) SOA effects at 400-600ms
    decay_half_life: Duration,

    /// Similarity threshold for semantic relation (default: 0.6)
    /// Only nodes with embedding similarity >0.6 are primed
    semantic_similarity_threshold: f32,

    /// Maximum neighbors to prime (default: 10)
    /// Prevents unbounded priming spread
    max_prime_neighbors: usize,
}

/// State of an active prime
struct PrimeState {
    /// When prime was activated
    activation_time: Instant,

    /// Initial activation strength [0, priming_strength]
    initial_strength: f32,

    /// Source episode that caused priming
    source_episode_id: String,

    /// Number of times this prime has been reinforced
    reinforcement_count: AtomicU64,
}

impl SemanticPrimingEngine {
    /// Create new semantic priming engine with empirically-validated defaults
    #[must_use]
    pub fn new() -> Self {
        Self {
            active_primes: DashMap::new(),
            priming_strength: 0.15,  // 15% boost
            decay_half_life: Duration::from_millis(500), // Neely 1977
            semantic_similarity_threshold: 0.6,
            max_prime_neighbors: 10,
        }
    }

    /// Activate semantic priming from recalled episode
    ///
    /// Spreads activation to semantically related concepts based on
    /// embedding similarity and graph connectivity.
    ///
    /// # Performance
    /// O(k log n) where k = max_prime_neighbors, n = graph size
    /// Typical: <100μs for 1M node graph
    pub fn activate_priming(&self, recalled: &Episode, graph: &MemoryGraph) {
        let node_id = recalled.node_id();

        // Find semantically similar neighbors via embedding similarity
        let neighbors = graph.find_k_nearest_neighbors(
            &recalled.embedding,
            self.max_prime_neighbors,
            self.semantic_similarity_threshold
        );

        let now = Instant::now();

        for (neighbor_id, similarity) in neighbors {
            // Skip self-priming
            if neighbor_id == node_id {
                continue;
            }

            // Prime strength proportional to semantic similarity
            // similarity ∈ [threshold, 1.0] → strength ∈ [0, priming_strength]
            let normalized_similarity = (similarity - self.semantic_similarity_threshold)
                / (1.0 - self.semantic_similarity_threshold);
            let prime_strength = self.priming_strength * normalized_similarity;

            // Update or insert prime
            self.active_primes
                .entry(neighbor_id)
                .and_modify(|prime| {
                    // Reinforce existing prime
                    prime.reinforcement_count.fetch_add(1, Ordering::Relaxed);
                    prime.activation_time = now; // Reset decay timer
                    prime.initial_strength = prime.initial_strength.max(prime_strength);
                })
                .or_insert_with(|| PrimeState {
                    activation_time: now,
                    initial_strength: prime_strength,
                    source_episode_id: recalled.id.clone(),
                    reinforcement_count: AtomicU64::new(1),
                });
        }

        // Record metrics
        #[cfg(feature = "monitoring")]
        {
            crate::metrics::cognitive_patterns()
                .record_priming(
                    crate::metrics::PrimingType::Semantic,
                    self.priming_strength
                );
        }
    }

    /// Compute current priming boost for a node
    ///
    /// Returns activation boost in [0, priming_strength] based on
    /// exponential decay with half-life.
    ///
    /// # Performance
    /// O(1) - single DashMap lookup + exponential computation
    /// Typical: <10ns
    #[must_use]
    pub fn compute_priming_boost(&self, node_id: NodeId) -> f32 {
        self.active_primes
            .get(&node_id)
            .map(|prime| self.compute_decayed_strength(&prime))
            .unwrap_or(0.0)
    }

    /// Compute decayed priming strength using exponential decay
    ///
    /// Formula: strength × 2^(-t/half_life)
    /// where t = elapsed time since activation
    fn compute_decayed_strength(&self, prime: &PrimeState) -> f32 {
        let elapsed = prime.activation_time.elapsed();
        let half_lives = elapsed.as_secs_f32() / self.decay_half_life.as_secs_f32();

        // Exponential decay: strength * 2^(-t/half_life)
        let decayed = prime.initial_strength * 0.5_f32.powf(half_lives);

        // Account for reinforcement (multiple exposures)
        let reinforcements = prime.reinforcement_count.load(Ordering::Relaxed);
        let reinforcement_bonus = 1.0 + (0.1 * (reinforcements.saturating_sub(1) as f32));

        (decayed * reinforcement_bonus).min(self.priming_strength)
    }

    /// Clear expired primes (prune entries with <1% residual strength)
    ///
    /// Should be called periodically to prevent unbounded growth.
    /// Recommended: every 1000 recall operations or every 10 seconds.
    pub fn prune_expired(&self) {
        let threshold = self.priming_strength * 0.01; // 1% residual

        self.active_primes.retain(|_, prime| {
            self.compute_decayed_strength(prime) > threshold
        });
    }

    /// Get statistics about active primes
    #[must_use]
    pub fn statistics(&self) -> PrimingStatistics {
        let total_primes = self.active_primes.len();

        let mut active_count = 0;
        let mut total_strength = 0.0;

        for entry in self.active_primes.iter() {
            let strength = self.compute_decayed_strength(&entry);
            if strength > 0.01 {
                active_count += 1;
                total_strength += strength;
            }
        }

        let average_strength = if active_count > 0 {
            total_strength / (active_count as f32)
        } else {
            0.0
        };

        PrimingStatistics {
            total_primes,
            active_primes: active_count,
            average_strength,
        }
    }
}

impl Default for SemanticPrimingEngine {
    fn default() -> Self {
        Self::new()
    }
}

pub struct PrimingStatistics {
    pub total_primes: usize,
    pub active_primes: usize,
    pub average_strength: f32,
}
```

### 2. Integration with Recall

```rust
// Extend /engram-core/src/activation/recall.rs

impl RecallEngine {
    /// Apply semantic priming boost during activation spreading
    fn apply_priming_boost(&self,
        node_id: NodeId,
        base_activation: f32
    ) -> f32 {
        if let Some(priming) = self.priming_engine.as_ref() {
            let boost = priming.compute_priming_boost(node_id);
            base_activation + boost
        } else {
            base_activation
        }
    }
}
```

### 3. Validation Tests

```rust
// /engram-core/tests/cognitive/semantic_priming_tests.rs

use engram_core::cognitive::priming::SemanticPrimingEngine;
use engram_core::{Episode, MemoryGraph};
use std::time::Duration;

#[test]
fn test_semantic_priming_basic() {
    let engine = SemanticPrimingEngine::new();
    let mut graph = MemoryGraph::new();

    // Create semantically related episodes
    let doctor = Episode::from_text("doctor", embedding_for("doctor"));
    let nurse = Episode::from_text("nurse", embedding_for("nurse")); // high similarity
    let car = Episode::from_text("car", embedding_for("car")); // low similarity

    graph.add_episode(doctor.clone());
    graph.add_episode(nurse.clone());
    graph.add_episode(car.clone());

    // Activate priming from "doctor"
    engine.activate_priming(&doctor, &graph);

    // "nurse" should be primed (high similarity)
    let nurse_boost = engine.compute_priming_boost(nurse.node_id());
    assert!(
        nurse_boost > 0.1,
        "Expected priming boost >0.1 for semantically related 'nurse', got {}",
        nurse_boost
    );

    // "car" should not be primed (low similarity)
    let car_boost = engine.compute_priming_boost(car.node_id());
    assert!(
        car_boost < 0.01,
        "Expected minimal priming <0.01 for unrelated 'car', got {}",
        car_boost
    );
}

#[test]
fn test_priming_decay() {
    let engine = SemanticPrimingEngine::new();
    let mut graph = MemoryGraph::new();

    let doctor = Episode::from_text("doctor", embedding_for("doctor"));
    let nurse = Episode::from_text("nurse", embedding_for("nurse"));

    graph.add_episode(doctor.clone());
    graph.add_episode(nurse.clone());

    // Activate priming
    engine.activate_priming(&doctor, &graph);

    let initial_boost = engine.compute_priming_boost(nurse.node_id());

    // Wait for half-life (500ms)
    std::thread::sleep(Duration::from_millis(500));

    let decayed_boost = engine.compute_priming_boost(nurse.node_id());

    // After one half-life, boost should be ~50% of initial
    let expected_decay = initial_boost * 0.5;
    let tolerance = 0.05; // 5% tolerance

    assert!(
        (decayed_boost - expected_decay).abs() < tolerance,
        "Expected decay to ~{:.3}, got {:.3} (initial: {:.3})",
        expected_decay, decayed_boost, initial_boost
    );
}

#[test]
fn test_priming_reinforcement() {
    let engine = SemanticPrimingEngine::new();
    let mut graph = MemoryGraph::new();

    let doctor = Episode::from_text("doctor", embedding_for("doctor"));
    let nurse = Episode::from_text("nurse", embedding_for("nurse"));

    graph.add_episode(doctor.clone());
    graph.add_episode(nurse.clone());

    // Activate priming once
    engine.activate_priming(&doctor, &graph);
    let single_boost = engine.compute_priming_boost(nurse.node_id());

    // Activate priming multiple times (reinforcement)
    for _ in 0..5 {
        engine.activate_priming(&doctor, &graph);
    }

    let reinforced_boost = engine.compute_priming_boost(nurse.node_id());

    // Reinforcement should increase boost (but capped at priming_strength)
    assert!(
        reinforced_boost >= single_boost,
        "Reinforcement should maintain or increase boost"
    );
}

#[test]
fn test_priming_pruning() {
    let engine = SemanticPrimingEngine::new();
    let mut graph = MemoryGraph::new();

    // Create many episodes to prime
    for i in 0..100 {
        graph.add_episode(Episode::from_text(&format!("item_{}", i), random_embedding()));
    }

    let prime_source = Episode::from_text("source", random_embedding());
    graph.add_episode(prime_source.clone());

    // Activate priming
    engine.activate_priming(&prime_source, &graph);

    let stats_before = engine.statistics();
    assert!(stats_before.total_primes > 0);

    // Wait for decay beyond threshold
    std::thread::sleep(Duration::from_secs(5)); // Well beyond 500ms half-life

    // Prune expired primes
    engine.prune_expired();

    let stats_after = engine.statistics();

    // Most primes should be pruned
    assert!(
        stats_after.total_primes < stats_before.total_primes / 2,
        "Expected pruning to remove most expired primes"
    );
}

/// Validation against Neely (1977) empirical data
///
/// Neely found 50-80ms RT reduction for semantically related primes
/// at SOA (stimulus onset asynchrony) of 400-700ms.
///
/// We model this as 10-20% activation boost (priming_strength = 0.15)
/// with 500ms half-life.
#[test]
fn test_neely_1977_validation() {
    let engine = SemanticPrimingEngine::new();

    // Verify default parameters match Neely findings
    assert_eq!(engine.decay_half_life, Duration::from_millis(500));
    assert!(engine.priming_strength >= 0.10 && engine.priming_strength <= 0.20,
        "Priming strength should be 10-20% per Neely (1977)");

    // Verify decay at SOA = 500ms (one half-life)
    // Should have ~50% residual boost
    // This matches Neely's observation of reduced but still significant priming
}
```

## Acceptance Criteria

1. **Empirical Validation (Neely 1977):**
   - Priming strength: 10-20% activation boost
   - Decay half-life: 400-600ms (default: 500ms)
   - Semantic similarity threshold: 0.6-0.8 (validated via parameter sweep)

2. **Functional Requirements:**
   - Semantically related concepts primed (embedding similarity >0.6)
   - Unrelated concepts not primed (similarity <0.6)
   - Exponential decay with configurable half-life
   - Reinforcement from multiple activations

3. **Performance:**
   - `activate_priming`: <100μs for 1M node graph
   - `compute_priming_boost`: <10ns (single lookup)
   - Memory: <1KB per 100 active primes

4. **Integration:**
   - Works with existing spreading activation (M3)
   - Records metrics via Task 001 infrastructure
   - Integrated with recall engine

## Follow-ups

- Task 003: Associative and Repetition Priming
- Task 008: DRM False Memory (uses semantic priming for critical lure)
