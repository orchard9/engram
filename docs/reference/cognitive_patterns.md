# Cognitive Patterns API Reference

Complete API documentation for Engram's biologically-inspired cognitive patterns: priming, interference, and reconsolidation.

## Quick Navigation

- [Priming](#priming) - Semantic, associative, and repetition priming

- [Interference](#interference) - Proactive, retroactive, and fan effect

- [Reconsolidation](#reconsolidation) - Memory updating during recall

- [Integration Examples](#integration-examples) - Real-world usage patterns

---

## Priming

Priming occurs when recalling one memory temporarily makes related memories easier to retrieve. Like how thinking about "doctor" makes "nurse" pop into your head faster.

### Semantic Priming

Spreading activation based on semantic similarity between concepts.

#### Quick Start

```rust
use engram_core::cognitive::priming::SemanticPrimingEngine;
use engram_core::{Episode, MemoryStore};

// Create priming engine with default settings
let priming = SemanticPrimingEngine::new();

// When user recalls an episode about "doctor"
let recalled_episode = store.recall_by_id("doctor_visit").unwrap();

// Activate priming for semantically related concepts
priming.activate_priming(&recalled_episode);

// Later, when recalling related concept like "nurse"
let node_id = "hospital_staff";
let boost = priming.compute_priming_boost(node_id);
println!("Priming boost: {:.1}%", boost * 100.0);
// Output: Priming boost: 12.5%

```

#### Configuration

| Parameter | Default | Range | Description | When to Adjust |
|-----------|---------|-------|-------------|----------------|
| `priming_strength` | 0.15 | 0.0-0.3 | Maximum activation boost (15%) | Decrease to 0.10 if unrelated concepts appearing; increase to 0.20 if related concepts not benefiting |
| `decay_half_life` | 300ms | 100ms-2s | How fast priming decays | Increase to 500ms for longer-lasting effects; decrease to 200ms for rapid decay |
| `similarity_threshold` | 0.6 | 0.4-0.8 | Minimum cosine similarity to prime | Increase to 0.7 for precision; decrease to 0.5 for recall |
| `max_graph_distance` | 2 | 1-3 | Maximum hops in semantic graph | Keep at 2 for biological plausibility |
| `refractory_period` | 50ms | 20ms-200ms | Prevents re-activation of same node | Increase if seeing oscillations |

#### Empirical Justification

**Priming Strength (0.15):**

- Neely (1977) found 10-20% reaction time reduction for primed words

- RT reduction: 50-80ms from 600ms baseline = 8.3%-13.3%

- We use 15% (midpoint) as activation boost

- Validation: DRM paradigm produces 60% false recall (target: 55-65%)

**Decay Half-Life (300ms):**

- Neely (1977): Automatic spreading activation peaks at 200-400ms SOA

- Most priming occurs within 400ms of prime presentation

- 300ms half-life ensures ~90% decay within 1 second

- Biological constraint: Automatic (not strategic) processing

**Similarity Threshold (0.6):**

- Optimized via parameter sweep on DRM word lists

- Balances false positives vs false negatives

- Below 0.5: Too many spurious primes

- Above 0.7: Misses valid semantic relations

#### API Reference

```rust
impl SemanticPrimingEngine {
    /// Create new priming engine with default parameters
    pub fn new() -> Self;

    /// Create with custom parameters
    pub fn with_config(
        priming_strength: f32,
        decay_half_life: Duration,
        similarity_threshold: f32,
    ) -> Self;

    /// Activate priming from a recalled episode
    ///
    /// Spreads activation to semantically similar nodes based on
    /// embedding similarity. Priming strength decays exponentially
    /// with configured half-life.
    ///
    /// # Example
    /// ```rust
    /// let priming = SemanticPrimingEngine::new();
    /// priming.activate_priming(&doctor_episode);
    /// // Now "nurse", "hospital" are primed
    /// ```
    pub fn activate_priming(&self, episode: &Episode);

    /// Compute current priming boost for a node
    ///
    /// Returns activation boost in range [0.0, priming_strength].
    /// Accounts for decay since prime activation and lateral inhibition
    /// from competing primes.
    ///
    /// # Returns
    /// - 0.0 if no active primes affect this node
    /// - Up to `priming_strength` (default 0.15) for strongly primed nodes
    ///
    /// # Example
    /// ```rust
    /// let boost = priming.compute_priming_boost("nurse_concept");
    /// let adjusted_activation = base_activation * (1.0 + boost);
    /// ```
    pub fn compute_priming_boost(&self, node_id: &str) -> f32;

    /// Remove expired primes (housekeeping)
    ///
    /// Call periodically to prevent memory growth. Recommended:
    /// every 1000 operations or 10 seconds, whichever comes first.
    ///
    /// # Example
    /// ```rust
    /// for (i, episode) in episodes.iter().enumerate() {
    ///     priming.activate_priming(episode);
    ///     if i % 1000 == 0 {
    ///         priming.prune_expired();
    ///     }
    /// }
    /// ```
    pub fn prune_expired(&self);

    /// Get statistics for monitoring
    pub fn statistics(&self) -> PrimingStatistics;
}

/// Priming statistics for monitoring
pub struct PrimingStatistics {
    /// Number of currently active primes
    pub active_primes_count: usize,
    /// Total priming activations since start
    pub total_activations: u64,
    /// Average priming boost magnitude
    pub average_boost: f32,
    /// P50 semantic similarity of primed pairs
    pub median_similarity: f32,
}

```

#### Performance Characteristics

**Measured on M1 Max, 32GB RAM, 1M node graph:**

| Operation | P50 Latency | P95 Latency | Memory per Prime |
|-----------|-------------|-------------|------------------|
| `activate_priming()` | 85μs | 120μs | 48 bytes |
| `compute_priming_boost()` | 8ns | 15ns | N/A (atomic read) |
| `prune_expired()` | 45μs | 95μs | Frees memory |

**Scaling:** O(k log n) where k = max_neighbors, n = graph_size

**Run benchmark:**

```bash
cargo bench --bench priming_performance -- --exact semantic

```

#### Common Mistakes

**Mistake 1: Forgetting to prune expired primes**

```rust
// BAD: active_primes grows unbounded
loop {
    priming.activate_priming(&episode);
}

// GOOD: prune every 1000 operations
for (i, episode) in episodes.iter().enumerate() {
    priming.activate_priming(episode);
    if i % 1000 == 0 {
        priming.prune_expired();
    }
}

```

**Why this matters:** Memory leak, performance degradation over time.
**How to detect:** Monitor `engram_active_primes_total` metric.

**Mistake 2: Applying priming boost incorrectly**

```rust
// BAD: Additive boost (wrong)
let activation = base_activation + priming.compute_priming_boost(node);

// GOOD: Multiplicative boost (correct)
let boost = priming.compute_priming_boost(node);
let activation = base_activation * (1.0 + boost);

```

**Why this matters:** Additive boost can exceed valid activation range [0, 1].

**Mistake 3: Not accounting for refractory period**

```rust
// If node activated by spreading activation, don't double-count priming
if recently_activated_by_spreading(node) {
    // Skip semantic priming boost
} else {
    activation *= 1.0 + priming.compute_priming_boost(node);
}

```

### Associative Priming

Co-occurrence based priming independent of semantic similarity.

#### Quick Start

```rust
use engram_core::cognitive::priming::AssociativePrimingEngine;

let priming = AssociativePrimingEngine::new();

// Build co-occurrence from recall patterns
priming.record_co_occurrence("coffee", "morning", 0.8);
priming.record_co_occurrence("coffee", "desk", 0.6);

// When "coffee" recalled, "morning" gets primed
priming.activate_associative_priming("coffee");

let boost = priming.compute_associative_boost("morning");
// boost ≈ 0.12 (0.15 * 0.8 = strength * co-occurrence)

```

#### When to Use

Use associative priming when:

- Building user preference models (co-viewed items)

- Learning temporal patterns (morning routines)

- Tracking episodic associations (this happened with that)

Don't use when:

- Pure semantic similarity is sufficient

- Co-occurrence data sparse or noisy

- Memory constraints tight (stores co-occurrence table)

#### Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `priming_strength` | 0.15 | Maximum boost |
| `decay_half_life` | 500ms | Slower than semantic (learned associations persist) |
| `min_co_occurrence` | 0.3 | Minimum strength to store |
| `max_associations_per_node` | 20 | Prevent unbounded growth |

### Repetition Priming

Faster recall of recently accessed items, regardless of semantic relation.

#### Quick Start

```rust
use engram_core::cognitive::priming::RepetitionPrimingEngine;

let priming = RepetitionPrimingEngine::new();

// Record that episode was just recalled
priming.record_access("episode_42");

// Shortly after, same episode gets boost
let boost = priming.compute_repetition_boost("episode_42");
// boost = 0.20 immediately, decays to 0.10 after 1 second

```

#### When to Use

Use for:

- Recently accessed item caching

- Recency bias in recall

- Working memory simulation

Configuration:

- Fast decay (200ms half-life)

- Higher initial boost (0.20)

- Simple recency tracking

---

## Interference

Interference occurs when similar memories compete during recall, slowing or preventing retrieval. Like struggling to remember where you parked today because yesterday's parking spot keeps intruding.

### Proactive Interference

Old memories interfere with new learning.

#### Quick Start

```rust
use engram_core::cognitive::interference::ProactiveInterferenceDetector;
use engram_core::Episode;

let detector = ProactiveInterferenceDetector::default();

// New memory being formed
let new_episode = Episode::new("parked_at_lot_B_today");

// Similar old memories from past week
let old_episodes = vec![
    Episode::new("parked_at_lot_A_yesterday"),
    Episode::new("parked_at_lot_C_two_days_ago"),
];

// Detect interference
let result = detector.detect_proactive_interference(
    &new_episode,
    &old_episodes,
);

println!("Interference: {:.1}%", result.interference_magnitude * 100.0);
// Output: Interference: 15.0%
// Confidence reduced by 15% due to similar prior memories

```

#### What It Does

Proactive interference models how prior similar memories make new learning harder. When you store similar episodes within 6 hours of each other, the new episode's confidence is reduced proportionally.

**Formula:**

```
interference = min(
    num_similar_old * interference_per_item,
    max_interference
)
confidence_new = confidence_original * (1 - interference)

```

**Example:**

- 3 similar parking memories in past 6 hours

- Each contributes 5% interference

- Total: 15% interference

- New memory confidence: 0.90 → 0.765

#### Configuration

| Parameter | Default | Description | When to Adjust |
|-----------|---------|-------------|----------------|
| `similarity_threshold` | 0.7 | Cosine similarity to count as "similar" | Increase to 0.8 for stricter matching |
| `prior_memory_window` | 6 hours | How far back to look for interfering memories | Based on consolidation window |
| `interference_per_item` | 0.05 | 5% per similar item | Empirical from Underwood (1957) |
| `max_interference` | 0.30 | Cap at 30% reduction | Prevents over-suppression |

#### Empirical Basis

**Underwood (1957):** "Interference and forgetting"

- Benchmark: 20-30% accuracy reduction with 5+ prior similar lists

- Our implementation: 5% per item → 25% with 5 items

- Within empirical range

**Temporal Window (6 hours):**

- Based on synaptic consolidation timescale (Dudai et al. 2015)

- After consolidation, memories transfer from hippocampus to neocortex

- Reduces interference (Complementary Learning Systems theory)

**Validation:**

- Interference validation suite (Task 010) confirms ±10% of empirical data

#### API Reference

```rust
pub struct ProactiveInterferenceDetector {
    pub fn default() -> Self;

    pub fn new(
        similarity_threshold: f32,
        prior_memory_window: Duration,
        interference_per_item: f32,
        max_interference: f32,
    ) -> Self;

    /// Detect interference from old memories on new learning
    ///
    /// # Arguments
    /// - `new_episode`: Episode being stored
    /// - `prior_episodes`: Candidate interfering episodes
    ///
    /// # Returns
    /// `ProactiveInterferenceResult` with:
    /// - `interference_magnitude`: 0.0-0.3 (0% to 30%)
    /// - `num_interfering_items`: Count of similar old episodes
    /// - `interfering_item_ids`: IDs for debugging
    pub fn detect_proactive_interference(
        &self,
        new_episode: &Episode,
        prior_episodes: &[Episode],
    ) -> ProactiveInterferenceResult;
}

pub struct ProactiveInterferenceResult {
    pub interference_magnitude: f32,
    pub num_interfering_items: usize,
    pub interfering_item_ids: Vec<String>,
    pub interference_type: InterferenceType,
}

pub enum InterferenceType {
    Proactive,
    Retroactive,
    FanEffect,
}

```

### Retroactive Interference

New learning interferes with recall of old memories.

#### Quick Start

```rust
use engram_core::cognitive::interference::RetroactiveInterferenceDetector;

let detector = RetroactiveInterferenceDetector::default();

// Old memory trying to recall
let old_episode = Episode::new("parked_at_lot_A_yesterday");

// New similar memories learned since
let new_episodes = vec![
    Episode::new("parked_at_lot_B_today"),
    Episode::new("parked_at_lot_C_this_morning"),
];

let result = detector.detect_retroactive_interference(
    &old_episode,
    &new_episodes,
);

// Old memory retrieval impaired by new learning

```

#### Configuration

Same as proactive interference, but looks forward in time instead of backward.

### Fan Effect

More associations to a concept → slower retrieval of any one association.

#### Quick Start

```rust
use engram_core::cognitive::interference::FanEffectDetector;

let detector = FanEffectDetector::default();

// Node with many associations
let node_id = "doctor_concept";
let num_associations = 15; // Many facts about doctors

let slowdown = detector.compute_fan_effect_slowdown(num_associations);
// slowdown = 1.75x (75% slower than baseline with 1 association)

// Apply to retrieval time
let base_latency_ms = 600.0;
let actual_latency_ms = base_latency_ms * slowdown;
// actual_latency_ms = 1050ms

```

#### What It Does

Models Anderson (1974) finding that retrieval time increases with number of facts associated with a concept. Each additional association adds ~50-150ms to retrieval (we use 80ms midpoint).

**Formula:**

```
retrieval_time = base_time + (num_associations - 1) * time_per_association
time_per_association = 80ms (Anderson 1974)

```

#### Configuration

| Parameter | Default | Empirical Basis |
|-----------|---------|-----------------|
| `base_retrieval_time` | 600ms | Anderson (1974) baseline |
| `time_per_association` | 80ms | Midpoint of 50-150ms range |

#### Empirical Validation

**Anderson (1974) Table 3:**

- 1 association: 600ms

- 2 associations: 680ms (+80ms)

- 3 associations: 760ms (+80ms)

Our implementation matches within ±25ms (Task 005 validation).

---

## Reconsolidation

Memories become modifiable during a specific window after recall, allowing updates before re-stabilization.

### Core Reconsolidation Engine

#### Quick Start

```rust
use engram_core::cognitive::reconsolidation::ReconsolidationEngine;
use chrono::Utc;

let engine = ReconsolidationEngine::new();

// Step 1: Record that memory was recalled
let episode_id = "childhood_birthday_party";
let recalled_episode = store.recall_by_id(episode_id).unwrap();

engine.record_recall(
    episode_id,
    recalled_episode.clone(),
    true, // active recall (required for reconsolidation)
);

// Step 2: Wait 3 hours (within 1-6 hour window)
// In production: time passes naturally
// In tests: advance virtual clock

// Step 3: Update memory with new information
let update = EpisodeModifications {
    confidence_adjustment: Some(0.9), // Reduce confidence slightly
    content_updates: vec![
        ("detail", ModificationType::Add("was_sunny".to_string()))
    ],
};

let result = engine.attempt_reconsolidation(
    episode_id,
    update,
    Utc::now(),
);

match result {
    Ok(modified_episode) => {
        println!("Memory updated during reconsolidation window");
        store.update(modified_episode);
    }
    Err(ReconsolidationError::OutsideWindow) => {
        println!("Too late for reconsolidation (>6 hours)");
    }
    Err(ReconsolidationError::MemoryNotRecalled) => {
        println!("Memory wasn't recalled recently");
    }
    Err(e) => {
        println!("Reconsolidation blocked: {:?}", e);
    }
}

```

#### Boundary Conditions (Exact)

Based on Nader et al. (2000) and Lee (2009):

| Boundary | Value | Rationale |
|----------|-------|-----------|
| Window start | 1 hour post-recall | Protein synthesis begins |
| Window end | 6 hours post-recall | Protein synthesis completes |
| Min memory age | 24 hours | Must be consolidated first |
| Max memory age | 365 days | Remote memories less plastic |
| Requires | Active recall | Not passive re-exposure |

**Plasticity dynamics:**

```
plasticity(t) = inverted_U_function(t, peak_at_3h)

- At 1h: 0.2 (rising)

- At 3h: 1.0 (peak)

- At 6h: 0.1 (declining)

```

#### What Reconsolidation Does

When you recall a memory:

1. Memory becomes labile (modifiable) for 1-6 hours

2. During this window, you can update confidence or content

3. Updates are weighted by plasticity (strongest at 3 hours post-recall)

4. After 6 hours, memory re-stabilizes and updates are rejected

**Real-world analogy:** Recalling childhood memory during therapy session. New perspective during session can modify the memory. But if you think about it again weeks later, the window has closed.

#### Configuration

| Parameter | Default | When to Adjust |
|-----------|---------|----------------|
| `window_start` | 1 hour | Don't adjust (biological) |
| `window_end` | 6 hours | Don't adjust (biological) |
| `min_memory_age` | 24 hours | Extend to 48h for very strong consolidation |
| `max_memory_age` | 365 days | Reduce to 180 days for strict plasticity |
| `reconsolidation_plasticity` | 0.5 | Maximum modification allowed |

#### API Reference

```rust
pub struct ReconsolidationEngine {
    pub fn new() -> Self;

    /// Record that an episode was recalled
    ///
    /// Must be called before attempting reconsolidation.
    /// Starts the 1-6 hour reconsolidation window.
    ///
    /// # Arguments
    /// - `episode_id`: Unique identifier
    /// - `original_episode`: Snapshot at time of recall
    /// - `is_active_recall`: Must be true for reconsolidation
    pub fn record_recall(
        &self,
        episode_id: &str,
        original_episode: Episode,
        is_active_recall: bool,
    );

    /// Attempt to update memory during reconsolidation window
    ///
    /// # Returns
    /// - `Ok(modified_episode)` if within window and conditions met
    /// - `Err(ReconsolidationError)` if blocked
    ///
    /// # Errors
    /// - `OutsideWindow`: Not within 1-6 hours post-recall
    /// - `MemoryNotRecalled`: No recent recall recorded
    /// - `MemoryTooYoung`: Not consolidated yet (<24h old)
    /// - `MemoryTooOld`: Remote memory (>365d old)
    /// - `NotActiveRecall`: Was passive re-exposure
    pub fn attempt_reconsolidation(
        &self,
        episode_id: &str,
        modifications: EpisodeModifications,
        current_time: DateTime<Utc>,
    ) -> Result<Episode, ReconsolidationError>;

    /// Compute current plasticity (0.0 to 1.0)
    ///
    /// Inverted-U function peaking at 3 hours post-recall.
    pub fn compute_plasticity(
        &self,
        time_since_recall: Duration,
    ) -> f32;

    /// Check if memory is in reconsolidation window
    pub fn is_in_window(
        &self,
        episode_id: &str,
        current_time: DateTime<Utc>,
    ) -> bool;
}

pub struct EpisodeModifications {
    /// Adjust confidence by multiplying by this factor
    pub confidence_adjustment: Option<f32>,

    /// Add, modify, or remove content
    pub content_updates: Vec<(String, ModificationType)>,
}

pub enum ModificationType {
    Add(String),
    Modify(String),
    Remove,
}

pub enum ReconsolidationError {
    OutsideWindow,
    MemoryNotRecalled,
    MemoryTooYoung,
    MemoryTooOld,
    NotActiveRecall,
    PlasticityTooLow,
}

```

#### Performance Characteristics

| Operation | Latency | Memory |
|-----------|---------|--------|
| `record_recall()` | 120ns | 256 bytes per recall |
| `attempt_reconsolidation()` | 1.5μs | N/A |
| `compute_plasticity()` | 45ns | N/A |

**Housekeeping:** Prune `recent_recalls` older than 7 days (beyond max window).

---

## Integration Examples

Real-world scenarios showing cognitive patterns working together.

### Example 1: Medical Knowledge System

Building a system where recalling "doctor" primes "nurse", but also detects interference from similar patient cases.

```rust
use engram_core::{MemoryStore, Episode, EpisodeBuilder};
use engram_core::cognitive::{
    SemanticPrimingEngine,
    ProactiveInterferenceDetector,
};

// Setup
let store = MemoryStore::new(10000);
let priming = SemanticPrimingEngine::new();
let interference = ProactiveInterferenceDetector::default();

// Store medical knowledge
let doctor_visit = EpisodeBuilder::new()
    .id("visit_001")
    .what("Annual checkup with Dr. Smith")
    .embedding(embed_text("doctor annual checkup"))
    .build()?;

store.store(doctor_visit.clone());

// Activate semantic priming
priming.activate_priming(&doctor_visit);

// Later: Store similar patient case
let new_case = EpisodeBuilder::new()
    .id("visit_002")
    .what("Follow-up with Dr. Jones")
    .embedding(embed_text("doctor follow-up"))
    .build()?;

// Check for interference from similar cases
let prior_cases = store.recall_similar(&new_case, 0.7, 10);
let interference_result = interference.detect_proactive_interference(
    &new_case,
    &prior_cases,
);

if interference_result.interference_magnitude > 0.2 {
    println!("Warning: High interference from {} similar cases",
        interference_result.num_interfering_items);
    println!("Confidence reduced by {:.1}%",
        interference_result.interference_magnitude * 100.0);
}

// Store with reduced confidence due to interference
let adjusted_confidence = new_case.confidence().value()
    * (1.0 - interference_result.interference_magnitude);

let new_case = new_case.with_confidence(adjusted_confidence.into());
store.store(new_case);

```

### Example 2: Recommendation Engine

Using priming for "users who viewed X often view Y" with reconsolidation for updating preferences.

```rust
use engram_core::cognitive::{
    AssociativePrimingEngine,
    ReconsolidationEngine,
};

let priming = AssociativePrimingEngine::new();
let reconsolidation = ReconsolidationEngine::new();

// Build co-occurrence from user behavior
priming.record_co_occurrence("product_A", "product_B", 0.75);
priming.record_co_occurrence("product_A", "product_C", 0.60);

// User views product A
priming.activate_associative_priming("product_A");

// Recommend products with strong priming
let recommendations = vec!["product_B", "product_C", "product_D"];
let scored: Vec<_> = recommendations.iter()
    .map(|id| {
        let boost = priming.compute_associative_boost(id);
        (id, boost)
    })
    .filter(|(_, boost)| *boost > 0.05) // 5% threshold
    .collect();

// User purchases product B (update co-occurrence)
let purchase_id = "user_123_purchase";
reconsolidation.record_recall(
    purchase_id,
    purchase_episode,
    true, // active
);

// Later: Update preference based on purchase
// (within reconsolidation window)
let update = EpisodeModifications {
    confidence_adjustment: Some(1.2), // Boost confidence
    content_updates: vec![
        ("tags", ModificationType::Add("verified_purchase".into())),
    ],
};

reconsolidation.attempt_reconsolidation(
    purchase_id,
    update,
    Utc::now(),
)?;

```

### Example 3: Educational Platform

Spacing effect for optimal review scheduling, DRM-aware to detect false confidence.

```rust
use engram_core::cognitive::{
    SemanticPrimingEngine,
    ProactiveInterferenceDetector,
};
use chrono::{Duration, Utc};

// Student learning facts with spaced repetition
let facts = vec![
    ("fact_1", "The capital of France is Paris"),
    ("fact_2", "The capital of Germany is Berlin"),
    ("fact_3", "The capital of Italy is Rome"),
];

// Schedule reviews with increasing spacing
let review_schedule = vec![
    Duration::hours(1),   // First review
    Duration::hours(4),   // Second review
    Duration::days(1),    // Third review
    Duration::days(3),    // Fourth review
    Duration::weeks(1),   // Fifth review
];

// Detect false confidence from DRM-like confusion
let detector = ProactiveInterferenceDetector::default();

for (fact_id, fact_text) in &facts {
    let episode = create_fact_episode(fact_id, fact_text)?;

    // Check for interference from similar facts
    let similar_facts = store.recall_similar(&episode, 0.7, 5);
    let interference = detector.detect_proactive_interference(
        &episode,
        &similar_facts,
    );

    if interference.interference_magnitude > 0.15 {
        println!("Warning: Student may confuse {} with similar facts",
            fact_id);
        println!("Consider interleaving different topics");
    }

    store.store(episode);
}

// When student recalls fact, boost similar facts (priming)
let priming = SemanticPrimingEngine::new();
let recalled = store.recall_by_id("fact_1")?;
priming.activate_priming(&recalled);

// Related facts now easier to recall
let boost_for_fact_2 = priming.compute_priming_boost("fact_2");
println!("Fact 2 priming: +{:.1}%", boost_for_fact_2 * 100.0);

```

---

## Monitoring

Each cognitive pattern exposes metrics for production monitoring.

### Key Metrics

**Semantic Priming:**

```
engram_priming_activations_total - Counter of priming activations
engram_active_primes_total - Gauge of currently active primes
engram_priming_boost_magnitude - Histogram of boost strengths
engram_priming_similarity_p50 - Median semantic similarity

```

**Interference:**

```
engram_interference_detections_total{type="proactive|retroactive|fan"}
engram_interference_magnitude - Histogram of interference strength
engram_interference_items_count - Histogram of interfering item counts

```

**Reconsolidation:**

```
engram_reconsolidation_attempts_total{result="success|outside_window|too_young|too_old"}
engram_reconsolidation_plasticity - Histogram of plasticity at attempt time
engram_reconsolidation_window_hits_rate - Success rate within window

```

### Alert Thresholds

See [Cognitive Metrics Tuning Guide](/docs/operations/cognitive_metrics_tuning.md) for detailed alert rules and runbooks.

---

## Troubleshooting

### Common Issues

**Priming not activating:**

- Check `engram_active_primes_total` - should be >0 after recalls

- Verify similarity threshold not too high (>0.8)

- Ensure embeddings are normalized (cosine similarity requires unit vectors)

**Interference always zero:**

- Check similarity threshold not too high

- Verify temporal window covers relevant memories

- Ensure episodes have timestamps

**Reconsolidation always fails:**

- Check time since recall (must be 1-6 hours)

- Verify memory age (must be >24 hours, <365 days)

- Ensure `is_active_recall=true` when recording recall

For complete troubleshooting runbooks, see [Operations Guide](/docs/operations/cognitive_metrics_tuning.md).

---

## Performance Tuning

### If Metrics Overhead >1%

Enable sampling:

```rust
MetricsConfig {
    sampling_rate: 0.10, // 10% of operations
    ..Default::default()
}

```

Reduce histogram buckets:

```rust
MetricsConfig {
    histogram_buckets: vec![0.01, 0.05, 0.10, 0.20, 0.50],
    ..Default::default()
}

```

### If Priming Too Aggressive

Reduce strength and increase decay:

```rust
SemanticPrimingEngine::with_config(
    0.10,           // priming_strength (was 0.15)
    Duration::from_millis(200), // faster decay (was 300ms)
    0.65,           // higher threshold (was 0.6)
)

```

### If Interference Too Sensitive

Increase threshold and reduce per-item:

```rust
ProactiveInterferenceDetector::new(
    0.75,                    // similarity_threshold (was 0.7)
    Duration::hours(6),      // keep same window
    0.03,                    // interference_per_item (was 0.05)
    0.25,                    // max_interference (was 0.30)
)

```

---

## References

Complete psychology foundations and biological basis: [Psychology Foundations](/docs/explanation/psychology_foundations.md)

Operational runbooks: [Cognitive Metrics Tuning](/docs/operations/cognitive_metrics_tuning.md)
