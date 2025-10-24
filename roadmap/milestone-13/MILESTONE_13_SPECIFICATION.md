# Milestone 13: Cognitive Patterns and Observability - Technical Specification

**Version:** 1.0
**Author:** Systems Product Planner (Bryan Cantrill mode)
**Date:** 2025-10-23
**Status:** Planning
**Dependencies:** M1-M8 (Core memory types through pattern completion)

---

## Executive Summary

Milestone 13 implements rigorously validated cognitive psychology phenomena (priming, interference, reconsolidation) and zero-overhead observability infrastructure. This is NOT about slapping "cognitive" labels on arbitrary heuristics. Every implementation must replicate published empirical data within defined statistical bounds.

**Critical Invariants:**
1. All cognitive phenomena validated against peer-reviewed psychology research
2. Metrics infrastructure: 0% overhead when disabled (compiler removes all instrumentation)
3. Metrics infrastructure: <1% overhead when enabled (measured via production workloads)
4. DRM paradigm replication within 10% of Roediger & McDermott (1995) false recall rates
5. Interference patterns match Anderson & Neely (1996) within 15% for PI/RI effects

**Estimated Duration:** 16-18 days (3.2-3.6 weeks)

---

## 1. Foundational Psychology Research & Validation Framework

### 1.1 Core Research Papers (Required Reading)

**Priming:**
- Tulving & Schacter (1990) - Priming taxonomy (semantic, associative, repetition)
- Collins & Loftus (1975) - Spreading activation and semantic priming
- Neely (1977) - Automatic vs strategic priming (SOA effects)
- McKoon & Ratcliff (1992) - Compound cue theory for associative priming

**Interference:**
- Anderson & Neely (1996) - Interference and forgetting (comprehensive review)
- McGeoch (1942) - Similarity effects in retroactive inhibition
- Underwood (1957) - Proactive inhibition as function of time and list length
- Anderson (1974) - Fan effect (retrieval interference from overlapping associations)

**Reconsolidation:**
- Nader et al. (2000) - Memory reconsolidation boundary conditions
- Lee (2009) - Reconsolidation: maintaining memory relevance
- Schiller et al. (2010) - Preventing return of fear through reconsolidation update

**False Memory:**
- Roediger & McDermott (1995) - DRM paradigm (creating false memories)
- Brainerd & Reyna (2002) - Fuzzy-trace theory of false memory

**Spacing Effect:**
- Cepeda et al. (2006) - Distributed practice meta-analysis
- Bjork & Bjork (1992) - New theory of disuse

### 1.2 Validation Methodology

**Statistical Framework:**
```rust
// /engram-core/src/validation/psychology.rs

/// Acceptance criteria for replicating empirical psychology experiments
pub struct ValidationCriteria {
    /// Maximum relative error from published means (default: 0.15 = 15%)
    pub max_relative_error: f64,

    /// Minimum correlation with published data (default: 0.80)
    pub min_correlation: f64,

    /// Required statistical power for hypothesis tests (default: 0.80)
    pub min_statistical_power: f64,

    /// Alpha level for significance testing (default: 0.05)
    pub alpha: f64,

    /// Minimum sample size for validation runs (default: 100)
    pub min_samples: usize,
}

impl Default for ValidationCriteria {
    fn default() -> Self {
        Self {
            max_relative_error: 0.15,  // 15% tolerance
            min_correlation: 0.80,      // Strong correlation required
            min_statistical_power: 0.80,
            alpha: 0.05,
            min_samples: 100,
        }
    }
}
```

**Acceptance Criteria Per Phenomenon:**

| Phenomenon | Empirical Baseline | Tolerance | Source |
|------------|-------------------|-----------|---------|
| DRM False Recall | 55-65% false "remember" for critical lure | ±10% | Roediger & McDermott 1995 |
| Semantic Priming | 50-80ms RT reduction (related vs unrelated) | ±15ms | Neely 1977 |
| Proactive Interference | 20-30% accuracy reduction (5+ prior lists) | ±10% | Underwood 1957 |
| Retroactive Interference | 15-25% accuracy reduction (1 interpolated list) | ±10% | McGeoch 1942 |
| Fan Effect | 50-150ms RT increase per additional fact | ±25ms | Anderson 1974 |
| Spacing Effect | 20-40% retention improvement (distributed vs massed) | ±10% | Cepeda et al. 2006 |
| Reconsolidation Window | 1-6 hour window for memory modification | Boundaries exact | Nader et al. 2000 |

---

## 2. Zero-Overhead Metrics Infrastructure

### 2.1 Architectural Principles

**Correctness Requirements:**
1. When `monitoring` feature disabled: Zero runtime cost (verified via assembly inspection)
2. When enabled: <1% overhead (measured via production-scale benchmarks)
3. No allocations in hot paths (metrics collection must be lock-free)
4. Metrics never cause query failures (graceful degradation)
5. Sampling must be statistically unbiased (no systematic skew)

**Implementation Strategy:**
- Conditional compilation for coarse-grained elimination
- Const generics for compile-time specialization
- Atomic operations for lock-free collection
- Fixed-size ring buffers to avoid allocation
- Reservoir sampling for bounded memory usage

### 2.2 Core Metrics API

```rust
// /engram-core/src/metrics/cognitive_patterns.rs

use std::sync::atomic::{AtomicU64, AtomicU32, Ordering};
use crossbeam_utils::CachePadded;

/// Zero-overhead cognitive pattern metrics
///
/// When `monitoring` feature is disabled, this compiles to zero-sized type
/// with all methods optimized away entirely.
pub struct CognitivePatternMetrics {
    #[cfg(feature = "monitoring")]
    inner: Arc<CognitivePatternMetricsInner>,
}

#[cfg(feature = "monitoring")]
struct CognitivePatternMetricsInner {
    // Priming metrics
    priming_events_total: CachePadded<AtomicU64>,
    priming_strength_histogram: LockFreeHistogram,
    priming_type_counters: [CachePadded<AtomicU64>; 3], // semantic, associative, repetition

    // Interference metrics
    interference_detections_total: CachePadded<AtomicU64>,
    proactive_interference_strength: LockFreeHistogram,
    retroactive_interference_strength: LockFreeHistogram,
    fan_effect_magnitude: LockFreeHistogram,

    // Reconsolidation metrics
    reconsolidation_events_total: CachePadded<AtomicU64>,
    reconsolidation_modifications: CachePadded<AtomicU64>,
    reconsolidation_window_hits: CachePadded<AtomicU64>,
    reconsolidation_window_misses: CachePadded<AtomicU64>,

    // False memory metrics
    false_memory_generations: CachePadded<AtomicU64>,
    drm_critical_lure_recalls: CachePadded<AtomicU64>,
    drm_list_item_recalls: CachePadded<AtomicU64>,

    // Spacing effect metrics
    massed_practice_events: CachePadded<AtomicU64>,
    distributed_practice_events: CachePadded<AtomicU64>,
    retention_improvement_histogram: LockFreeHistogram,
}

impl CognitivePatternMetrics {
    /// Record priming event with zero overhead when monitoring disabled
    #[inline(always)]
    pub fn record_priming(&self, priming_type: PrimingType, strength: f32) {
        #[cfg(feature = "monitoring")]
        {
            self.inner.priming_events_total.fetch_add(1, Ordering::Relaxed);
            self.inner.priming_strength_histogram.record(strength as f64);

            let idx = priming_type as usize;
            self.inner.priming_type_counters[idx].fetch_add(1, Ordering::Relaxed);
        }

        // When monitoring disabled, this entire function body optimizes away
        #[cfg(not(feature = "monitoring"))]
        {
            let _ = (priming_type, strength); // Suppress unused warnings
        }
    }

    /// Record interference detection with zero overhead when disabled
    #[inline(always)]
    pub fn record_interference(&self,
        interference_type: InterferenceType,
        magnitude: f32
    ) {
        #[cfg(feature = "monitoring")]
        {
            self.inner.interference_detections_total.fetch_add(1, Ordering::Relaxed);

            match interference_type {
                InterferenceType::Proactive => {
                    self.inner.proactive_interference_strength.record(magnitude as f64);
                }
                InterferenceType::Retroactive => {
                    self.inner.retroactive_interference_strength.record(magnitude as f64);
                }
                InterferenceType::Fan => {
                    self.inner.fan_effect_magnitude.record(magnitude as f64);
                }
            }
        }

        #[cfg(not(feature = "monitoring"))]
        {
            let _ = (interference_type, magnitude);
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum PrimingType {
    Semantic = 0,
    Associative = 1,
    Repetition = 2,
}

#[derive(Debug, Clone, Copy)]
pub enum InterferenceType {
    Proactive,
    Retroactive,
    Fan,
}
```

### 2.3 Performance Validation Requirements

**Benchmark Suite:**
```rust
// /engram-core/benches/metrics_overhead.rs

use criterion::{black_box, criterion_group, criterion_main, Criterion};

/// Measure metrics overhead on hot path operations
fn benchmark_recall_with_metrics(c: &mut Criterion) {
    let mut group = c.benchmark_group("metrics_overhead");

    // Baseline: No metrics
    #[cfg(not(feature = "monitoring"))]
    group.bench_function("recall_no_metrics", |b| {
        b.iter(|| {
            // Production recall operation
            black_box(perform_recall());
        });
    });

    // Comparison: With metrics enabled
    #[cfg(feature = "monitoring")]
    group.bench_function("recall_with_metrics", |b| {
        b.iter(|| {
            black_box(perform_recall_instrumented());
        });
    });

    group.finish();
}

// Acceptance criteria: <1% overhead
#[cfg(all(feature = "monitoring", test))]
#[test]
fn verify_metrics_overhead_under_one_percent() {
    let baseline = measure_baseline_throughput();
    let instrumented = measure_instrumented_throughput();

    let overhead = (baseline - instrumented) / baseline;

    assert!(
        overhead < 0.01,
        "Metrics overhead {:.2}% exceeds 1% threshold",
        overhead * 100.0
    );
}
```

---

## 3. Priming Implementation

### 3.1 Semantic Priming

**Biological Model:**
Spreading activation in semantic networks (Collins & Loftus 1975). When "doctor" activates, related concepts like "nurse", "hospital" receive pre-activation, reducing retrieval time.

**Implementation:**

```rust
// /engram-core/src/cognitive/priming/semantic.rs

use crate::{Episode, Confidence, MemoryNode};
use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Semantic priming engine based on spreading activation
pub struct SemanticPrimingEngine {
    /// Active primes and their decay times
    active_primes: dashmap::DashMap<NodeId, PrimeState>,

    /// Priming strength (default: 0.15 = 15% activation boost)
    priming_strength: f32,

    /// Decay half-life for priming (default: 500ms, per Neely 1977)
    decay_half_life: Duration,

    /// Similarity threshold for semantic relation (default: 0.6)
    semantic_similarity_threshold: f32,
}

/// State of an active prime
struct PrimeState {
    /// When prime was activated
    activation_time: Instant,

    /// Initial activation strength
    initial_strength: f32,

    /// Source episode that caused priming
    source_episode_id: String,
}

impl SemanticPrimingEngine {
    /// Activate semantic priming from recalled episode
    ///
    /// Spreads activation to semantically related concepts based on
    /// embedding similarity and graph connectivity.
    pub fn activate_priming(&self, recalled: &Episode, graph: &MemoryGraph) {
        let node_id = recalled.node_id();

        // Find semantically similar neighbors
        let neighbors = graph.find_similar_nodes(
            &recalled.embedding,
            self.semantic_similarity_threshold
        );

        for (neighbor_id, similarity) in neighbors {
            // Prime strength proportional to similarity
            let prime_strength = self.priming_strength * similarity;

            self.active_primes.insert(neighbor_id, PrimeState {
                activation_time: Instant::now(),
                initial_strength: prime_strength,
                source_episode_id: recalled.id.clone(),
            });
        }

        // Record metrics
        #[cfg(feature = "monitoring")]
        {
            crate::metrics::cognitive_patterns()
                .record_priming(PrimingType::Semantic, self.priming_strength);
        }
    }

    /// Compute current priming boost for a node
    ///
    /// Returns activation boost in [0, priming_strength] based on decay
    pub fn compute_priming_boost(&self, node_id: NodeId) -> f32 {
        self.active_primes
            .get(&node_id)
            .map(|prime| self.compute_decayed_strength(&prime))
            .unwrap_or(0.0)
    }

    /// Compute decayed priming strength using exponential decay
    fn compute_decayed_strength(&self, prime: &PrimeState) -> f32 {
        let elapsed = prime.activation_time.elapsed();
        let half_lives = elapsed.as_secs_f32() / self.decay_half_life.as_secs_f32();

        // Exponential decay: strength * 2^(-t/half_life)
        prime.initial_strength * 0.5_f32.powf(half_lives)
    }

    /// Clear expired primes (prune entries with <1% residual strength)
    pub fn prune_expired(&self) {
        self.active_primes.retain(|_, prime| {
            self.compute_decayed_strength(prime) > 0.01
        });
    }
}

impl Default for SemanticPrimingEngine {
    fn default() -> Self {
        Self {
            active_primes: dashmap::DashMap::new(),
            priming_strength: 0.15,  // 15% boost, within empirical 10-20% range
            decay_half_life: Duration::from_millis(500), // Neely 1977: 400-600ms
            semantic_similarity_threshold: 0.6,
        }
    }
}
```

### 3.2 Associative Priming

**Biological Model:**
Co-occurrence strengthens associative links (McKoon & Ratcliff 1992). "Thunder" primes "lightning" not through semantic similarity but through learned associations.

```rust
// /engram-core/src/cognitive/priming/associative.rs

use std::collections::HashMap;
use chrono::{DateTime, Utc};

/// Associative priming through co-occurrence learning
pub struct AssociativePrimingEngine {
    /// Co-occurrence frequency: (node_a, node_b) -> count
    cooccurrence_counts: dashmap::DashMap<(NodeId, NodeId), u64>,

    /// Total activations per node (for normalizing probabilities)
    node_activation_counts: dashmap::DashMap<NodeId, u64>,

    /// Temporal window for co-occurrence (default: 10 seconds)
    cooccurrence_window: Duration,

    /// Minimum co-occurrence for association (default: 3)
    min_cooccurrence: u64,
}

impl AssociativePrimingEngine {
    /// Record co-activation of nodes within temporal window
    pub fn record_coactivation(&self,
        node_a: NodeId,
        node_b: NodeId,
        timestamp: DateTime<Utc>
    ) {
        // Increment co-occurrence count (symmetric)
        let key = if node_a < node_b { (node_a, node_b) } else { (node_b, node_a) };

        self.cooccurrence_counts
            .entry(key)
            .or_insert(0)
            .fetch_add(1, Ordering::Relaxed);

        // Update individual activation counts
        self.node_activation_counts
            .entry(node_a)
            .or_insert(0)
            .fetch_add(1, Ordering::Relaxed);

        self.node_activation_counts
            .entry(node_b)
            .or_insert(0)
            .fetch_add(1, Ordering::Relaxed);
    }

    /// Compute associative priming strength between nodes
    ///
    /// Uses conditional probability: P(B|A) = P(A,B) / P(A)
    pub fn compute_association_strength(&self,
        prime: NodeId,
        target: NodeId
    ) -> f32 {
        let key = if prime < target { (prime, target) } else { (target, prime) };

        let cooccur = self.cooccurrence_counts
            .get(&key)
            .map(|count| *count)
            .unwrap_or(0);

        if cooccur < self.min_cooccurrence {
            return 0.0; // Insufficient evidence
        }

        let prime_count = self.node_activation_counts
            .get(&prime)
            .map(|count| *count)
            .unwrap_or(1); // Avoid division by zero

        // Conditional probability as association strength
        (cooccur as f32) / (prime_count as f32)
    }
}
```

### 3.3 Repetition Priming

**Biological Model:**
Repeated exposure facilitates processing (perceptual fluency). Same stimulus encountered multiple times becomes easier to process.

```rust
// /engram-core/src/cognitive/priming/repetition.rs

/// Repetition priming through exposure counting
pub struct RepetitionPrimingEngine {
    /// Exposure counts per node
    exposure_counts: dashmap::DashMap<NodeId, u64>,

    /// Activation boost per repetition (default: 0.05 = 5% per exposure)
    boost_per_repetition: f32,

    /// Maximum cumulative boost (default: 0.30 = 30% cap)
    max_cumulative_boost: f32,
}

impl RepetitionPrimingEngine {
    /// Record exposure to node (recall, query, consolidation, etc.)
    pub fn record_exposure(&self, node_id: NodeId) {
        self.exposure_counts
            .entry(node_id)
            .or_insert(0)
            .fetch_add(1, Ordering::Relaxed);

        #[cfg(feature = "monitoring")]
        {
            crate::metrics::cognitive_patterns()
                .record_priming(PrimingType::Repetition, self.boost_per_repetition);
        }
    }

    /// Compute cumulative priming boost from repetitions
    pub fn compute_repetition_boost(&self, node_id: NodeId) -> f32 {
        let exposures = self.exposure_counts
            .get(&node_id)
            .map(|count| *count)
            .unwrap_or(0);

        // Linear accumulation with ceiling
        let boost = (exposures as f32) * self.boost_per_repetition;
        boost.min(self.max_cumulative_boost)
    }
}
```

---

## 4. Interference Detection and Modeling

### 4.1 Proactive Interference

**Psychology Model:**
Old memories interfere with new learning (Underwood 1957). Effect increases with:
1. Number of prior lists learned
2. Similarity between old and new material
3. Retention interval

**Implementation:**

```rust
// /engram-core/src/cognitive/interference/proactive.rs

use crate::{Episode, Confidence};
use chrono::{DateTime, Utc, Duration};

/// Proactive interference detector and modeler
pub struct ProactiveInterferenceDetector {
    /// Similarity threshold for interference (default: 0.7)
    similarity_threshold: f32,

    /// Temporal window for "prior" memories (default: 24 hours before)
    prior_memory_window: Duration,

    /// Interference strength per similar prior item (default: 0.05)
    interference_per_item: f32,

    /// Maximum interference effect (default: 0.30 = 30% retrieval reduction)
    max_interference: f32,
}

impl ProactiveInterferenceDetector {
    /// Detect proactive interference for a new episode
    ///
    /// Returns interference magnitude in [0, max_interference]
    pub fn detect_interference(&self,
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
        }
    }

    /// Check if prior episode interferes with new episode
    fn is_interfering(&self,
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
}

pub struct ProactiveInterferenceResult {
    /// Magnitude of interference in [0, max_interference]
    pub magnitude: f32,

    /// Episode IDs of interfering prior memories
    pub interfering_episodes: Vec<String>,
}
```

### 4.2 Retroactive Interference

**Psychology Model:**
New learning interferes with old memories (McGeoch 1942). Most pronounced when:
1. New material is similar to old
2. New material is learned soon after old
3. Testing occurs soon after new learning

```rust
// /engram-core/src/cognitive/interference/retroactive.rs

/// Retroactive interference detector and modeler
pub struct RetroactiveInterferenceDetector {
    /// Similarity threshold (default: 0.7)
    similarity_threshold: f32,

    /// Temporal window for retroactive effect (default: 1 hour after original)
    retroactive_window: Duration,

    /// Base interference strength (default: 0.10 = 10% retrieval reduction)
    base_interference: f32,

    /// Similarity exponent for interference (default: 2.0 for quadratic scaling)
    similarity_exponent: f32,
}

impl RetroactiveInterferenceDetector {
    /// Detect retroactive interference on an old episode
    ///
    /// Returns interference magnitude based on subsequent similar learning
    pub fn detect_interference(&self,
        target_episode: &Episode,
        subsequent_episodes: &[Episode],
        graph: &MemoryGraph
    ) -> RetroactiveInterferenceResult {
        let interfering = subsequent_episodes
            .iter()
            .filter(|ep| self.is_interfering(target_episode, ep, graph))
            .collect::<Vec<_>>();

        // Compute interference using similarity-weighted sum
        let magnitude = interfering
            .iter()
            .map(|ep| {
                let sim = graph.compute_embedding_similarity(
                    &target_episode.embedding,
                    &ep.embedding
                );

                // Interference proportional to similarity^exponent
                self.base_interference * sim.powf(self.similarity_exponent)
            })
            .sum::<f32>()
            .min(0.50); // Cap at 50% retrieval reduction

        #[cfg(feature = "monitoring")]
        {
            crate::metrics::cognitive_patterns()
                .record_interference(InterferenceType::Retroactive, magnitude);
        }

        RetroactiveInterferenceResult {
            magnitude,
            interfering_episodes: interfering
                .into_iter()
                .map(|ep| ep.id.clone())
                .collect(),
        }
    }

    fn is_interfering(&self,
        target: &Episode,
        subsequent: &Episode,
        graph: &MemoryGraph
    ) -> bool {
        // Must be temporally after target
        if subsequent.timestamp <= target.timestamp {
            return false;
        }

        // Must be within retroactive window
        let time_diff = subsequent.timestamp - target.timestamp;
        if time_diff > self.retroactive_window {
            return false;
        }

        // Must be sufficiently similar
        let similarity = graph.compute_embedding_similarity(
            &target.embedding,
            &subsequent.embedding
        );

        similarity >= self.similarity_threshold
    }
}
```

### 4.3 Fan Effect (Retrieval Interference)

**Psychology Model:**
More associations to a concept slow retrieval (Anderson 1974). "The banker is in the park" is retrieved faster if "banker" appears in only one fact versus five facts.

```rust
// /engram-core/src/cognitive/interference/fan_effect.rs

/// Fan effect detector and modeler
pub struct FanEffectDetector {
    /// Base retrieval time (default: 100ms)
    base_retrieval_time_ms: f32,

    /// Time cost per additional association (default: 50ms, per Anderson 1974)
    time_per_association_ms: f32,
}

impl FanEffectDetector {
    /// Compute fan effect slowdown for a node
    ///
    /// Returns predicted retrieval time increase in milliseconds
    pub fn compute_fan_effect(&self,
        node_id: NodeId,
        graph: &MemoryGraph
    ) -> FanEffectResult {
        // Count outgoing edges (associations from this node)
        let fan_count = graph.count_outgoing_edges(node_id);

        // Linear slowdown per association (empirically validated)
        let retrieval_time_ms = self.base_retrieval_time_ms
            + (fan_count as f32 * self.time_per_association_ms);

        let magnitude = (fan_count as f32 * self.time_per_association_ms)
            / self.base_retrieval_time_ms;

        #[cfg(feature = "monitoring")]
        {
            crate::metrics::cognitive_patterns()
                .record_interference(InterferenceType::Fan, magnitude);
        }

        FanEffectResult {
            fan_count,
            predicted_retrieval_time_ms: retrieval_time_ms,
            magnitude,
        }
    }
}

pub struct FanEffectResult {
    /// Number of associations (fan)
    pub fan_count: usize,

    /// Predicted retrieval time in milliseconds
    pub predicted_retrieval_time_ms: f32,

    /// Magnitude relative to baseline
    pub magnitude: f32,
}
```

---

## 5. Reconsolidation System

### 5.1 Theoretical Foundation

**Biology:**
Memories become labile (unstable) during recall, allowing modification before re-stabilization (Nader et al. 2000). Critical window: 1-6 hours post-retrieval.

**Boundary Conditions (Exact, per research):**
1. Reconsolidation window: 1-6 hours post-recall
2. Requires active recall (passive exposure insufficient)
3. Only applies to consolidated memories (>24 hours old)
4. Does not occur if memory is very old (>1 year, boundary fuzzy)

### 5.2 Implementation

```rust
// /engram-core/src/cognitive/reconsolidation/mod.rs

use chrono::{DateTime, Utc, Duration};
use crate::{Episode, Confidence};

/// Memory reconsolidation engine
///
/// Implements Nader et al. (2000) reconsolidation dynamics with exact
/// boundary conditions from empirical research.
pub struct ReconsolidationEngine {
    /// Reconsolidation window start (default: 1 hour post-recall)
    window_start: Duration,

    /// Reconsolidation window end (default: 6 hours post-recall)
    window_end: Duration,

    /// Minimum memory age for reconsolidation (default: 24 hours)
    min_memory_age: Duration,

    /// Maximum memory age for reconsolidation (default: 365 days)
    max_memory_age: Duration,

    /// Plasticity during reconsolidation (default: 0.5 = 50% modifiable)
    reconsolidation_plasticity: f32,

    /// Tracking of recent recalls for reconsolidation opportunities
    recent_recalls: dashmap::DashMap<String, RecallEvent>,
}

struct RecallEvent {
    episode_id: String,
    recall_timestamp: DateTime<Utc>,
    original_episode: Episode,
}

impl ReconsolidationEngine {
    /// Record a recall event for potential reconsolidation
    pub fn record_recall(&self, episode: &Episode, recall_time: DateTime<Utc>) {
        self.recent_recalls.insert(
            episode.id.clone(),
            RecallEvent {
                episode_id: episode.id.clone(),
                recall_timestamp: recall_time,
                original_episode: episode.clone(),
            }
        );

        // Prune old recall events outside reconsolidation window
        self.prune_old_recalls(recall_time);
    }

    /// Attempt to modify a memory through reconsolidation
    ///
    /// Returns None if outside reconsolidation window or boundary conditions not met
    pub fn attempt_reconsolidation(&self,
        episode_id: &str,
        modifications: EpisodeModifications,
        current_time: DateTime<Utc>
    ) -> Option<ReconsolidationResult> {
        let recall_event = self.recent_recalls.get(episode_id)?;

        // Check boundary conditions
        if !self.is_reconsolidation_eligible(&recall_event, current_time) {
            return None;
        }

        // Compute time within window (for plasticity modulation)
        let time_since_recall = current_time - recall_event.recall_timestamp;
        let window_position = self.compute_window_position(time_since_recall);

        // Apply modifications with plasticity factor
        let modified_episode = self.apply_modifications(
            &recall_event.original_episode,
            modifications,
            window_position
        );

        #[cfg(feature = "monitoring")]
        {
            crate::metrics::cognitive_patterns()
                .record_reconsolidation(window_position);
        }

        Some(ReconsolidationResult {
            modified_episode,
            plasticity_factor: window_position,
            modification_confidence: self.compute_modification_confidence(window_position),
        })
    }

    /// Check if memory meets boundary conditions for reconsolidation
    fn is_reconsolidation_eligible(&self,
        recall: &RecallEvent,
        current_time: DateTime<Utc>
    ) -> bool {
        let time_since_recall = current_time - recall.recall_timestamp;

        // Must be within reconsolidation window
        if time_since_recall < self.window_start || time_since_recall > self.window_end {
            return false;
        }

        // Memory must be consolidated (>24 hours old)
        let memory_age = recall.recall_timestamp - recall.original_episode.timestamp;
        if memory_age < self.min_memory_age {
            return false;
        }

        // Memory must not be too old (>1 year)
        if memory_age > self.max_memory_age {
            return false;
        }

        true
    }

    /// Compute position within reconsolidation window [0, 1]
    ///
    /// 0 = start of window (maximum plasticity)
    /// 1 = end of window (minimum plasticity)
    fn compute_window_position(&self, time_since_recall: Duration) -> f32 {
        let window_duration = self.window_end - self.window_start;
        let offset = time_since_recall - self.window_start;

        let position = offset.num_milliseconds() as f32
            / window_duration.num_milliseconds() as f32;

        position.clamp(0.0, 1.0)
    }

    /// Apply modifications during reconsolidation window
    fn apply_modifications(&self,
        original: &Episode,
        modifications: EpisodeModifications,
        window_position: f32
    ) -> Episode {
        // Plasticity decreases linearly across window
        let plasticity = self.reconsolidation_plasticity * (1.0 - window_position);

        let mut modified = original.clone();

        // Apply field modifications weighted by plasticity
        for (field, new_value) in modifications.field_changes {
            // Blend original and new value based on plasticity
            modified.update_field(&field, new_value, plasticity);
        }

        // Update confidence based on modification extent
        modified.confidence = self.compute_modified_confidence(
            original.confidence,
            modifications.modification_extent,
            plasticity
        );

        modified
    }

    fn prune_old_recalls(&self, current_time: DateTime<Utc>) {
        self.recent_recalls.retain(|_, recall| {
            let time_since = current_time - recall.recall_timestamp;
            time_since <= self.window_end
        });
    }
}

pub struct EpisodeModifications {
    field_changes: HashMap<String, String>,
    modification_extent: f32, // [0, 1] how much of episode is modified
}

pub struct ReconsolidationResult {
    pub modified_episode: Episode,
    pub plasticity_factor: f32,
    pub modification_confidence: Confidence,
}
```

---

## 6. Classic Experiment Replication

### 6.1 DRM Paradigm (Deese-Roediger-McDermott)

**Experiment Design:**
1. Present semantically related word lists (e.g., bed, rest, awake, tired, dream...)
2. Test recall of list items plus critical lure ("sleep" - never presented)
3. Measure false recall rate for critical lure

**Expected Result:** 55-65% false recall for critical lure (Roediger & McDermott 1995)

```rust
// /engram-core/tests/psychology/drm_paradigm.rs

use engram_core::{MemoryEngine, Episode};

#[test]
fn test_drm_false_memory_replication() {
    let engine = MemoryEngine::new();

    // DRM list: "sleep" critical lure
    let study_list = vec![
        "bed", "rest", "awake", "tired", "dream",
        "wake", "snooze", "blanket", "doze", "slumber",
        "snore", "nap", "peace", "yawn", "drowsy"
    ];

    // Store study list as episodes
    for word in &study_list {
        engine.store_episode(Episode::from_text(word));
    }

    // Trigger consolidation to form semantic pattern
    engine.consolidate();

    // Recall test with critical lure
    let recall_results = engine.recall_by_cue("sleep");

    // Measure false recall
    let false_recall = recall_results
        .iter()
        .any(|r| r.is_reconstructed() && r.content.contains("sleep"));

    // Run experiment 100 times for statistical power
    let mut false_recall_count = 0;
    for trial in 0..100 {
        let results = run_drm_trial(&engine, &study_list, "sleep");
        if results.false_recall {
            false_recall_count += 1;
        }
    }

    let false_recall_rate = false_recall_count as f64 / 100.0;

    // Acceptance: 55-65% ± 10% = [45%, 75%]
    assert!(
        false_recall_rate >= 0.45 && false_recall_rate <= 0.75,
        "DRM false recall rate {:.1}% outside [45%, 75%] acceptance range",
        false_recall_rate * 100.0
    );

    // Ideal: within 10% of Roediger & McDermott (1995) mean of 60%
    if false_recall_rate >= 0.50 && false_recall_rate <= 0.70 {
        println!("DRM paradigm replication SUCCESS: {:.1}% false recall (target: 60%)",
            false_recall_rate * 100.0);
    }
}
```

### 6.2 Spacing Effect Validation

**Experiment Design:**
1. Massed practice: Study items 3 times consecutively
2. Distributed practice: Study items 3 times with 1-hour spacing
3. Test retention after 24 hours
4. Measure accuracy difference

**Expected Result:** 20-40% better retention for distributed vs massed (Cepeda et al. 2006)

```rust
// /engram-core/tests/psychology/spacing_effect.rs

#[test]
fn test_spacing_effect_replication() {
    let engine = MemoryEngine::new();

    let study_items = generate_random_facts(50);

    // Group 1: Massed practice
    let massed_group = &study_items[0..25];
    for item in massed_group {
        for _ in 0..3 {
            engine.store_episode(item.clone());
        }
    }

    // Group 2: Distributed practice (1 hour spacing)
    let distributed_group = &study_items[25..50];
    for item in distributed_group {
        engine.store_episode(item.clone());
        std::thread::sleep(Duration::from_secs(3600)); // 1 hour
        engine.store_episode(item.clone());
        std::thread::sleep(Duration::from_secs(3600));
        engine.store_episode(item.clone());
    }

    // Retention test after 24 hours
    std::thread::sleep(Duration::from_secs(86400));

    let massed_accuracy = test_retention(&engine, massed_group);
    let distributed_accuracy = test_retention(&engine, distributed_group);

    let improvement = (distributed_accuracy - massed_accuracy) / massed_accuracy;

    // Acceptance: 20-40% ± 10% = [10%, 50%]
    assert!(
        improvement >= 0.10 && improvement <= 0.50,
        "Spacing effect {:.1}% outside [10%, 50%] acceptance range",
        improvement * 100.0
    );
}
```

---

## 7. Task Breakdown (14 Tasks, 16-18 Days)

### Task 001: Zero-Overhead Metrics Infrastructure (2 days)
**Priority:** P0 (Foundation)
**Files:**
- `/engram-core/src/metrics/cognitive_patterns.rs` - New cognitive pattern metrics
- `/engram-core/src/metrics/mod.rs` - Export cognitive pattern types
- `/engram-core/benches/metrics_overhead.rs` - Overhead benchmarks

**Acceptance:**
- `cargo build` with `--no-default-features` produces zero metrics code (assembly inspection)
- Overhead benchmark shows <1% latency increase with monitoring enabled
- Lock-free atomic operations verified via loom

### Task 002: Semantic Priming Engine (2 days)
**Priority:** P0 (Critical Path)
**Dependencies:** Task 001
**Files:**
- `/engram-core/src/cognitive/priming/semantic.rs`
- `/engram-core/src/cognitive/priming/mod.rs`
- `/engram-core/tests/cognitive/semantic_priming_tests.rs`

**Acceptance:**
- Priming decay matches Neely (1977) half-life of 500ms ±100ms
- Semantic priming reduces retrieval time by 10-20% for related concepts
- Integration test: "doctor" primes "nurse" at >0.7 strength

### Task 003: Associative and Repetition Priming (2 days)
**Priority:** P1
**Dependencies:** Task 002
**Files:**
- `/engram-core/src/cognitive/priming/associative.rs`
- `/engram-core/src/cognitive/priming/repetition.rs`
- `/engram-core/tests/cognitive/priming_integration_tests.rs`

**Acceptance:**
- Associative priming captures co-occurrence within 10-second window
- Repetition priming provides 5% boost per exposure, capped at 30%
- All three priming types integrate without conflicts

### Task 004: Proactive Interference Detection (2 days)
**Priority:** P0 (Critical Path)
**Dependencies:** Task 001
**Files:**
- `/engram-core/src/cognitive/interference/proactive.rs`
- `/engram-core/src/cognitive/interference/mod.rs`
- `/engram-core/tests/cognitive/proactive_interference_tests.rs`

**Acceptance:**
- Underwood (1957) replication: 20-30% accuracy reduction with 5+ prior similar lists (±10%)
- Interference magnitude increases linearly with number of similar prior items
- Similarity threshold (0.7) validated through parameter sweep

### Task 005: Retroactive Interference and Fan Effect (2 days)
**Priority:** P1
**Dependencies:** Task 004
**Files:**
- `/engram-core/src/cognitive/interference/retroactive.rs`
- `/engram-core/src/cognitive/interference/fan_effect.rs`
- `/engram-core/tests/cognitive/interference_integration_tests.rs`

**Acceptance:**
- McGeoch (1942) replication: 15-25% accuracy reduction with interpolated list (±10%)
- Anderson (1974) fan effect: 50-150ms RT increase per additional association (±25ms)
- All interference types tracked independently in metrics

### Task 006: Reconsolidation Engine Core (3 days)
**Priority:** P0 (Critical Path)
**Dependencies:** None
**Files:**
- `/engram-core/src/cognitive/reconsolidation/mod.rs`
- `/engram-core/src/cognitive/reconsolidation/window.rs`
- `/engram-core/tests/cognitive/reconsolidation_tests.rs`

**Acceptance:**
- Boundary conditions exactly match Nader et al. (2000):
  - Reconsolidation window: 1-6 hours post-recall (exact)
  - Minimum memory age: 24 hours (exact)
  - Maximum memory age: 365 days (exact)
- Plasticity decreases linearly across window (validated via unit tests)
- Outside window: modifications rejected with clear error

### Task 007: Reconsolidation Integration with Consolidation (2 days)
**Priority:** P1
**Dependencies:** Task 006, M6 (Consolidation)
**Files:**
- `/engram-core/src/cognitive/reconsolidation/consolidation_integration.rs`
- `/engram-core/tests/cognitive/reconsolidation_consolidation_tests.rs`

**Acceptance:**
- Reconsolidated memories re-enter consolidation pipeline
- No conflicts between consolidation and reconsolidation processes
- Metrics distinguish initial consolidation from reconsolidation events

### Task 008: DRM False Memory Paradigm Implementation (2 days)
**Priority:** P0 (Validation Critical)
**Dependencies:** Task 002, M8 (Pattern Completion)
**Files:**
- `/engram-core/tests/psychology/drm_paradigm.rs`
- `/engram-core/tests/psychology/drm_word_lists.json` - Standard DRM lists

**Acceptance:**
- Roediger & McDermott (1995) replication: 55-65% false recall ±10%
- Statistical power: n=100 trials, α=0.05, power>0.80
- False recall tagged with `is_false_memory: true` in reconstruction metadata

### Task 009: Spacing Effect Validation (1 day)
**Priority:** P1 (Validation)
**Dependencies:** M4 (Temporal Dynamics)
**Files:**
- `/engram-core/tests/psychology/spacing_effect.rs`

**Acceptance:**
- Cepeda et al. (2006) meta-analysis: 20-40% retention improvement ±10%
- Massed vs distributed comparison with matched items
- Results logged with statistical significance (p < 0.05)

### Task 010: Interference Validation Suite (1 day)
**Priority:** P1 (Validation)
**Dependencies:** Tasks 004, 005
**Files:**
- `/engram-core/tests/psychology/interference_validation.rs`

**Acceptance:**
- Underwood (1957) PI validation: within ±10%
- McGeoch (1942) RI validation: within ±10%
- Anderson (1974) fan effect: within ±25ms

### Task 011: Tracing Infrastructure for Cognitive Dynamics (2 days)
**Priority:** P1
**Dependencies:** Task 001
**Files:**
- `/engram-core/src/tracing/cognitive_events.rs`
- `/engram-core/src/tracing/visualization.rs`
- `/engram-cli/src/tracing_export.rs`

**Acceptance:**
- Structured tracing for priming events (type, strength, source, target)
- Structured tracing for interference (type, magnitude, competing items)
- Structured tracing for reconsolidation (window position, modifications)
- JSON export for external visualization tools

### Task 012: Prometheus/Grafana Dashboard for Cognitive Metrics (1 day)
**Priority:** P2 (Observability)
**Dependencies:** Task 011
**Files:**
- `/docs/operations/grafana/cognitive_patterns_dashboard.json`

**Deliverables:**
- Grafana dashboard with panels for:
  - Priming event rates by type
  - Interference magnitude histograms (PI, RI, fan)
  - Reconsolidation window hit rate
  - False memory generation rate
- Prometheus queries optimized for <100ms response time

### Task 013: Integration Testing and Performance Validation (2 days)
**Priority:** P0 (Quality Gate)
**Dependencies:** All implementation tasks (001-007, 011)
**Files:**
- `/engram-core/tests/integration/cognitive_patterns_integration.rs`
- `/engram-core/benches/cognitive_patterns_performance.rs`

**Acceptance:**
- All cognitive patterns work together without conflicts
- Metrics overhead <1% on production-scale workload (10K ops/sec)
- No memory leaks during 10-minute soak test
- All psychology validations pass (Tasks 008-010)

### Task 014: Documentation and Operational Runbook (1 day)
**Priority:** P2
**Dependencies:** Task 013
**Files:**
- `/docs/reference/cognitive_patterns.md` - API reference
- `/docs/explanation/psychology_foundations.md` - Biological basis
- `/docs/operations/cognitive_metrics_tuning.md` - Operational guide

**Deliverables:**
- API documentation with psychology citations
- Tuning guide for priming/interference parameters
- Troubleshooting guide for anomalous cognitive metrics

---

## 8. Risk Analysis and Mitigation

### Risk 1: Psychology Validation Failures
**Likelihood:** Medium
**Impact:** High (blocks milestone completion)

**Mitigation:**
1. Parameter sweep during implementation to find empirically-grounded defaults
2. Consult memory-systems-researcher agent for biological plausibility review
3. Budget 2 extra days for tuning if initial validation fails
4. Document parameter sensitivity in explanation docs

### Risk 2: Metrics Overhead Exceeds 1%
**Likelihood:** Low
**Impact:** High (production deployment blocked)

**Mitigation:**
1. Benchmark early (Task 001) before building on top
2. Use assembly inspection to verify zero-cost when disabled
3. Implement sampling if overhead exceeds budget (10% sample rate)
4. Profile with perf/flamegraph to identify hot spots

### Risk 3: Reconsolidation Window Edge Cases
**Likelihood:** Medium
**Impact:** Medium (correctness issues)

**Mitigation:**
1. Implement boundary conditions as exact checks (no fuzzy logic)
2. Comprehensive unit tests for all window boundaries
3. Property-based testing with quickcheck for temporal edge cases
4. Formal specification document with exact inequalities

### Risk 4: Integration with Existing Systems (M6, M8)
**Likelihood:** Medium
**Impact:** Medium (integration complexity)

**Mitigation:**
1. Review existing consolidation (M6) and pattern completion (M8) code early
2. Design reconsolidation as separate module with clear interfaces
3. Integration testing after each task (not just at end)
4. rust-graph-engine-architect agent review for API design

---

## 9. Success Criteria (Milestone Completion)

**Must Have (Blocks Completion):**
1. All 14 tasks marked complete with passing tests
2. DRM paradigm validation within ±10% of published data
3. Metrics overhead benchmark shows <1% when enabled, 0% when disabled
4. All three interference types (PI, RI, fan) validated within acceptance criteria
5. Reconsolidation boundary conditions match Nader et al. (2000) exactly
6. Zero clippy warnings, all tests pass, `make quality` succeeds

**Should Have (Quality Goals):**
1. Spacing effect validation within ±10% of Cepeda et al. (2006)
2. All three priming types (semantic, associative, repetition) validated
3. Grafana dashboard deployed with live metrics
4. Psychology explanation docs cite all primary sources

**Nice to Have (Future Work):**
1. Serial position effects (primacy/recency curves)
2. Deeper DRM analysis (remember vs know judgments)
3. Individual differences in interference susceptibility
4. Reconsolidation interference (new learning during window)

---

## 10. Integration with Existing Codebase

### Extends Existing Systems

**M4 (Temporal Dynamics):**
- Priming decay uses existing decay function infrastructure
- Spacing effect validation uses existing forgetting curves
- File: `/engram-core/src/decay/mod.rs`

**M6 (Consolidation):**
- Reconsolidation re-uses consolidation pipeline
- False memory generation integrates with pattern detection
- File: `/engram-core/src/decay/consolidation.rs`

**M8 (Pattern Completion):**
- DRM paradigm uses pattern completion for critical lure generation
- Integration: `/engram-core/src/completion/mod.rs`

**Existing Metrics (M6, M7):**
- Cognitive pattern metrics extend existing `MetricsRegistry`
- Reuse lock-free infrastructure from `metrics/lockfree.rs`
- File: `/engram-core/src/metrics/mod.rs`

### New Module Structure

```
engram-core/src/
├── cognitive/              # New: Cognitive psychology implementations
│   ├── mod.rs
│   ├── priming/
│   │   ├── mod.rs
│   │   ├── semantic.rs
│   │   ├── associative.rs
│   │   └── repetition.rs
│   ├── interference/
│   │   ├── mod.rs
│   │   ├── proactive.rs
│   │   ├── retroactive.rs
│   │   └── fan_effect.rs
│   └── reconsolidation/
│       ├── mod.rs
│       ├── window.rs
│       └── consolidation_integration.rs
├── metrics/                # Extended
│   ├── cognitive_patterns.rs  # New
│   └── ...existing files
├── tracing/                # New: Structured cognitive event tracing
│   ├── mod.rs
│   ├── cognitive_events.rs
│   └── visualization.rs
└── validation/             # New: Psychology experiment validation
    ├── mod.rs
    └── psychology.rs

engram-core/tests/
├── cognitive/              # New: Unit tests for cognitive modules
│   ├── semantic_priming_tests.rs
│   ├── interference_tests.rs
│   └── reconsolidation_tests.rs
└── psychology/             # New: Empirical validation tests
    ├── drm_paradigm.rs
    ├── spacing_effect.rs
    └── interference_validation.rs
```

---

## 11. Performance Budgets

**Latency:**
- Priming boost computation: <10μs (hot path)
- Interference detection: <100μs (query path)
- Reconsolidation eligibility check: <50μs (recall path)
- Metrics recording: <50ns (atomic operations only)

**Memory:**
- Active primes: <1MB for 10K nodes
- Co-occurrence table: <10MB for 1M node pairs
- Recent recalls: <100KB for 1K tracked episodes
- Metrics buffers: <1MB total

**Throughput:**
- Must sustain 10K recalls/sec with priming enabled
- Must sustain 1K reconsolidation attempts/sec
- Metrics export: 1K events/sec to Prometheus

---

## 12. Academic Citations (Required Reading)

1. Anderson, J. R. (1974). Retrieval of propositional information from long-term memory. *Cognitive Psychology*, 6(4), 451-474.
2. Anderson, M. C., & Neely, J. H. (1996). Interference and inhibition in memory retrieval. *Memory*, 125-153.
3. Bjork, R. A., & Bjork, E. L. (1992). A new theory of disuse and an old theory of stimulus fluctuation. *From Learning Processes to Cognitive Processes*, 2, 35-67.
4. Brainerd, C. J., & Reyna, V. F. (2002). Fuzzy-trace theory and false memory. *Current Directions in Psychological Science*, 11(5), 164-169.
5. Cepeda, N. J., et al. (2006). Distributed practice in verbal recall tasks: A review and quantitative synthesis. *Psychological Bulletin*, 132(3), 354.
6. Collins, A. M., & Loftus, E. F. (1975). A spreading-activation theory of semantic processing. *Psychological Review*, 82(6), 407.
7. Lee, J. L. (2009). Reconsolidation: maintaining memory relevance. *Trends in Neurosciences*, 32(8), 413-420.
8. McGeoch, J. A. (1942). The psychology of human learning: An introduction.
9. McKoon, G., & Ratcliff, R. (1992). Spreading activation versus compound cue accounts of priming. *Psychological Review*, 99(1), 177.
10. Nader, K., Schafe, G. E., & Le Doux, J. E. (2000). Fear memories require protein synthesis in the amygdala for reconsolidation after retrieval. *Nature*, 406(6797), 722-726.
11. Neely, J. H. (1977). Semantic priming and retrieval from lexical memory. *Journal of Experimental Psychology: General*, 106(3), 226.
12. Roediger, H. L., & McDermott, K. B. (1995). Creating false memories: Remembering words not presented in lists. *Journal of Experimental Psychology: Learning, Memory, and Cognition*, 21(4), 803.
13. Schiller, D., et al. (2010). Preventing the return of fear in humans using reconsolidation update mechanisms. *Nature*, 463(7277), 49-53.
14. Tulving, E., & Schacter, D. L. (1990). Priming and human memory systems. *Science*, 247(4940), 301-306.
15. Underwood, B. J. (1957). Interference and forgetting. *Psychological Review*, 64(1), 49.

---

## Appendix A: Validation Test Matrix

| Test | Empirical Target | Acceptance Range | Source | Priority |
|------|------------------|------------------|---------|----------|
| DRM False Recall | 60% | 45-75% | Roediger 1995 | P0 |
| Semantic Priming RT | 65ms reduction | 50-80ms | Neely 1977 | P0 |
| Proactive Interference | 25% accuracy drop | 15-35% | Underwood 1957 | P0 |
| Retroactive Interference | 20% accuracy drop | 10-30% | McGeoch 1942 | P0 |
| Fan Effect | 50ms per association | 25-75ms | Anderson 1974 | P1 |
| Spacing Effect | 30% retention gain | 20-40% | Cepeda 2006 | P1 |
| Reconsolidation Window | 1-6 hours | Exact | Nader 2000 | P0 |

---

## Appendix B: Metrics Overhead Verification

```bash
# Build without monitoring (should produce zero metrics code)
cargo build --release --no-default-features
objdump -d target/release/engram-core | grep -c "cognitive_patterns"
# Expected: 0 (no cognitive pattern symbols in binary)

# Build with monitoring
cargo build --release --features monitoring

# Benchmark overhead
cargo bench --bench metrics_overhead -- --save-baseline baseline
cargo bench --bench metrics_overhead --features monitoring -- --save-baseline instrumented

# Compare baselines
critcmp baseline instrumented
# Expected: <1% regression in P50, P95, P99 latencies
```

---

**End of Specification**

This specification is complete, unambiguous, and directly implementable. Every cognitive phenomenon has precise acceptance criteria tied to published research. Every performance boundary is measurable. Every integration point is specified with exact file paths.

Now ship it.
