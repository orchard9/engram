# Task 001: Reconstruction Primitives

**Status:** Pending
**Priority:** P0 (Critical Path)
**Estimated Effort:** 3 days
**Dependencies:** None (foundation task)

## Objective

Implement field-level reconstruction primitives for completing missing episode details using local context and temporal neighbors. Create the core API for partial-to-complete episode transformation with explicit source tracking at the field granularity.

## Integration Points

**Extends:**
- `/engram-core/src/completion/reconstruction.rs` - PatternReconstructor trait implementation
- `/engram-core/src/completion/mod.rs` - Export new types and public API

**Uses:**
- `/engram-core/src/memory.rs` - Episode type and field access
- `/engram-core/src/activation/recall.rs` - Temporal neighbor retrieval
- `/engram-core/src/embedding/similarity.rs` - SIMD embedding similarity

**Creates:**
- `/engram-core/src/completion/field_reconstruction.rs` - Field-level completion logic
- `/engram-core/src/completion/local_context.rs` - Temporal/spatial context extraction
- `/engram-core/tests/reconstruction_primitives_tests.rs` - Comprehensive test suite

## Detailed Specification

### 1. Field-Level Reconstruction API

```rust
// /engram-core/src/completion/field_reconstruction.rs

use crate::{Confidence, Episode};
use crate::completion::{MemorySource, PartialEpisode, SourceMap};
use std::collections::HashMap;

/// Field-level reconstruction engine for completing missing episode details
pub struct FieldReconstructor {
    /// Temporal window for neighbor retrieval (default: 1 hour)
    temporal_window: std::time::Duration,

    /// Minimum similarity for neighbor contribution (default: 0.7)
    similarity_threshold: f32,

    /// Maximum neighbors to consider (default: 5)
    max_neighbors: usize,

    /// Neighbor weighting decay (default: 0.8)
    neighbor_decay: f32,
}

impl FieldReconstructor {
    /// Create new field reconstructor with default parameters
    pub fn new() -> Self;

    /// Reconstruct missing fields using temporal neighbors
    ///
    /// Returns map of field_name -> (reconstructed_value, confidence, source)
    pub fn reconstruct_fields(
        &self,
        partial: &PartialEpisode,
        temporal_neighbors: &[Episode],
    ) -> HashMap<String, ReconstructedField>;

    /// Extract temporal context from episode history
    ///
    /// Returns episodes within temporal window sorted by recency
    pub fn extract_temporal_context(
        &self,
        anchor_time: chrono::DateTime<chrono::Utc>,
        episode_store: &[Episode],
    ) -> Vec<Episode>;

    /// Compute field confidence based on neighbor consensus
    ///
    /// Higher confidence when multiple neighbors agree
    fn compute_field_confidence(
        &self,
        field_values: &[String],
        neighbor_similarities: &[f32],
    ) -> Confidence;
}

/// Reconstructed field with provenance tracking
#[derive(Debug, Clone)]
pub struct ReconstructedField {
    /// Reconstructed value for the field
    pub value: String,

    /// Confidence in reconstruction (0.0-1.0)
    pub confidence: Confidence,

    /// Source of reconstruction
    pub source: MemorySource,

    /// Evidence from contributing neighbors
    pub evidence: Vec<NeighborEvidence>,
}

/// Evidence from a single temporal neighbor
#[derive(Debug, Clone)]
pub struct NeighborEvidence {
    /// Episode ID contributing evidence
    pub episode_id: String,

    /// Similarity to partial episode (0.0-1.0)
    pub similarity: f32,

    /// Temporal distance (seconds)
    pub temporal_distance: f64,

    /// Field value from this neighbor
    pub field_value: String,

    /// Evidence weight (similarity * temporal_decay)
    pub weight: f32,
}

impl Default for FieldReconstructor {
    fn default() -> Self {
        Self {
            temporal_window: std::time::Duration::from_secs(3600), // 1 hour
            similarity_threshold: 0.7,
            max_neighbors: 5,
            neighbor_decay: 0.8,
        }
    }
}
```

### 2. Local Context Extraction

```rust
// /engram-core/src/completion/local_context.rs

use crate::{Episode, Confidence};
use chrono::{DateTime, Utc};

/// Local context extractor for temporal and spatial proximity
pub struct LocalContextExtractor {
    /// Temporal window for context (default: 1 hour before/after)
    temporal_window: std::time::Duration,

    /// Spatial radius for proximity (default: 100 meters, if location available)
    spatial_radius: f32,

    /// Recency weighting exponent (default: 2.0 for quadratic decay)
    recency_exponent: f32,
}

impl LocalContextExtractor {
    /// Create new local context extractor
    pub fn new() -> Self;

    /// Extract temporal neighbors within window
    pub fn temporal_neighbors(
        &self,
        anchor_time: DateTime<Utc>,
        episodes: &[Episode],
    ) -> Vec<TemporalNeighbor>;

    /// Extract spatially proximate episodes (if location metadata available)
    pub fn spatial_neighbors(
        &self,
        anchor_location: Option<&str>,
        episodes: &[Episode],
    ) -> Vec<SpatialNeighbor>;

    /// Compute recency weight for temporal distance
    ///
    /// Weight = (1 - normalized_distance) ^ recency_exponent
    /// Recent episodes have weight near 1.0, distant episodes near 0.0
    pub fn recency_weight(&self, temporal_distance: std::time::Duration) -> f32;

    /// Merge temporal and spatial context with adaptive weighting
    pub fn merge_contexts(
        &self,
        temporal: Vec<TemporalNeighbor>,
        spatial: Vec<SpatialNeighbor>,
    ) -> Vec<ContextEvidence>;
}

/// Temporal neighbor with recency weighting
#[derive(Debug, Clone)]
pub struct TemporalNeighbor {
    pub episode: Episode,
    pub temporal_distance: std::time::Duration,
    pub recency_weight: f32,
}

/// Spatial neighbor with proximity weighting
#[derive(Debug, Clone)]
pub struct SpatialNeighbor {
    pub episode: Episode,
    pub spatial_distance: f32,
    pub proximity_weight: f32,
}

/// Combined context evidence from temporal and spatial
#[derive(Debug, Clone)]
pub struct ContextEvidence {
    pub episode: Episode,
    pub combined_weight: f32,
    pub temporal_contribution: f32,
    pub spatial_contribution: f32,
}
```

### 3. Field Consensus Algorithm

**Voting Mechanism:**
1. Retrieve temporal neighbors within window (max 5)
2. Compute embedding similarity between partial and each neighbor
3. Filter neighbors below similarity threshold (0.7)
4. For each missing field, collect values from neighbors
5. Weight each value by: `similarity * recency_weight * neighbor_decay^rank`
6. Compute consensus: Most frequent value weighted by evidence strength
7. Confidence = weighted agreement ratio (0.0-1.0)

**Example:**
```
Partial episode: {what: "breakfast", when: null, where: null}

Temporal neighbors:
  N1 (20 min ago, sim=0.9): {when: "morning", where: "kitchen"}  -> weight = 0.9 * 0.95 = 0.855
  N2 (35 min ago, sim=0.8): {when: "morning", where: "dining"}   -> weight = 0.8 * 0.85 = 0.680
  N3 (50 min ago, sim=0.7): {when: "morning", where: "kitchen"}  -> weight = 0.7 * 0.75 = 0.525

Field "when" reconstruction:
  "morning": total_weight = 0.855 + 0.680 + 0.525 = 2.060
  consensus = "morning", confidence = 2.060 / (3 * max_weight) = 2.060 / 2.565 = 0.803

Field "where" reconstruction:
  "kitchen": weight = 0.855 + 0.525 = 1.380
  "dining": weight = 0.680
  consensus = "kitchen", confidence = 1.380 / 2.060 = 0.670
```

### 4. Source Attribution Rules

**Recalled:** Field present in `partial.known_fields` → `MemorySource::Recalled`, confidence = `partial.cue_strength`

**Reconstructed:** Field completed from temporal neighbors → `MemorySource::Reconstructed`, confidence = consensus ratio

**Imagined:** Field filled with low-confidence placeholder → `MemorySource::Imagined`, confidence < 0.3

**Source Confidence:** Independent measure of attribution accuracy
- High consensus (>80% agreement) → source confidence = 0.95
- Medium consensus (60-80% agreement) → source confidence = 0.75
- Low consensus (<60% agreement) → source confidence = 0.50

## Acceptance Criteria

1. **Field Reconstruction Accuracy:**
   - Reconstruct >85% of missing fields correctly on test dataset with 3+ temporal neighbors
   - Confidence scores correlate >0.80 with reconstruction accuracy (Spearman rank)
   - Consensus algorithm handles conflicting neighbor values gracefully

2. **Source Attribution Precision:**
   - Correctly label recalled vs reconstructed fields with >95% precision
   - Source confidence correlates >0.85 with attribution accuracy
   - Never mark genuinely recalled fields as reconstructed (zero false positives)

3. **Performance:**
   - Field reconstruction <2ms P95 for 5 neighbors (SIMD-optimized similarity)
   - Temporal context extraction <3ms P95 for 100-episode window
   - Zero allocations in consensus hot path (use pre-allocated buffers)

4. **Temporal Context Quality:**
   - Recency weighting produces monotonically decreasing weights with distance
   - Temporal window correctly filters episodes outside range
   - Handles episodes with identical timestamps (use ID tie-breaking)

5. **Edge Case Handling:**
   - Returns empty reconstructions when no neighbors available (graceful degradation)
   - Handles partial episodes with empty `known_fields` (all-reconstructed)
   - Correctly processes episodes with unusual field types (numeric, JSON, binary)

## Testing Strategy

### Unit Tests (`/engram-core/tests/reconstruction_primitives_tests.rs`)

```rust
#[test]
fn test_field_consensus_with_unanimous_agreement() {
    // All neighbors agree on field value
    // Expected: confidence near 1.0, source = Reconstructed
}

#[test]
fn test_field_consensus_with_split_vote() {
    // Neighbors split 60/40 on field value
    // Expected: majority wins, confidence ~0.6
}

#[test]
fn test_recalled_field_preservation() {
    // Field present in known_fields
    // Expected: source = Recalled, original value preserved
}

#[test]
fn test_temporal_neighbor_filtering() {
    // Episodes both inside and outside temporal window
    // Expected: only in-window episodes contribute
}

#[test]
fn test_recency_weighting_decay() {
    // Neighbors at 10min, 30min, 60min distances
    // Expected: weights 0.95, 0.75, 0.50 (quadratic decay)
}

#[test]
fn test_similarity_threshold_filtering() {
    // Neighbors with similarities 0.9, 0.65, 0.5
    // Expected: only 0.9 contributes (threshold 0.7)
}

#[test]
fn test_empty_neighbor_set_graceful_degradation() {
    // No temporal neighbors available
    // Expected: empty reconstructions, confidence = 0.0
}

#[test]
fn test_source_confidence_calibration() {
    // Known ground truth for source attribution
    // Expected: source confidence correlates with accuracy
}
```

### Integration Tests

```rust
#[test]
fn test_end_to_end_field_reconstruction() {
    // Create partial episode with 50% fields missing
    // Store 10 temporal neighbors with varying similarities
    // Reconstruct all fields
    // Expected: >85% accuracy, properly attributed sources
}

#[test]
fn test_temporal_context_with_real_episodes() {
    // Store 100 episodes over 24-hour period
    // Extract context for middle timepoint
    // Expected: correct window, proper recency weights
}
```

### Property-Based Tests (proptest)

```rust
proptest! {
    #[test]
    fn prop_consensus_confidence_bounded(neighbors: Vec<NeighborEvidence>) {
        // Generated: arbitrary neighbor evidence
        // Property: confidence always in [0.0, 1.0]
    }

    #[test]
    fn prop_recency_weight_monotonic_decrease(distances: Vec<Duration>) {
        // Generated: arbitrary temporal distances (sorted)
        // Property: weight(d1) >= weight(d2) when d1 <= d2
    }

    #[test]
    fn prop_source_attribution_consistent(partial: PartialEpisode) {
        // Generated: arbitrary partial episodes
        // Property: recalled fields never marked reconstructed
    }
}
```

## Theoretical Foundations

### Source Monitoring Framework (Johnson et al., 1993)
Reconstruction pathway (Task 005 research) operates when fields are completed from temporal neighbors rather than recalled from cues. Empirical research shows:
- Temporal proximity is a strong cue for related episodes
- Source information must be explicitly tracked (not inferred from confidence alone)
- People struggle to distinguish sources when time delay increases

**Application to Task 001:** NeighborEvidence struct tracks temporal_distance and weight to enable precise source attribution in Task 005.

### Memory Reconstruction Theory (Hemmer & Steyvers, 2009)
Memory reconstructs episodes using hierarchical Bayesian inference where temporal neighbors serve as likelihood evidence. Reconstructions blend specific episodic details with consensus patterns from neighbors.

**Implementation:** Field consensus algorithm implements this through weighted voting where:
```
confidence = weighted_agreement_ratio = total_neighbor_weight / max_possible_weight
```

### Ensemble Methods (Breiman, 1996)
Combining predictions from multiple sources (temporal neighbors) improves accuracy if sources are diverse and better than random baseline.

**Requirements:**
- Diversity: Neighbors at different temporal distances make different errors
- Accuracy: Each neighbor similarity must exceed threshold (0.7 default)

**Validation in Task 009:** Reconstruction accuracy >85% demonstrates ensemble benefit.

## Risk Mitigation

**Risk: Neighbor consensus produces incorrect reconstructions**
- **Mitigation:** Statistical validation on ground truth datasets; adjust similarity threshold
- **Contingency:** Require minimum 3 neighbors for reconstruction; mark low-consensus as "imagined"
- **Research Support:** Ensemble methods (Breiman, 1996) show combining 3+ diverse sources reduces error by 20-30%

**Risk: Temporal window too narrow/wide**
- **Mitigation:** Configurable window per application; adaptive sizing based on episode density
- **Contingency:** Exponential backoff widening when no neighbors found
- **Empirical Basis:** Source monitoring research shows temporal proximity is strongest cue within 1-hour window

**Risk: SIMD similarity computations introduce errors**
- **Mitigation:** Property tests verify SIMD matches scalar; differential testing against reference
- **Contingency:** Feature flag for scalar fallback; numerical tolerance checks

## Implementation Notes

1. Use existing `embedding::similarity::cosine_similarity_simd` for neighbor comparisons
2. Pre-allocate buffers for field consensus to avoid hot-path allocations
3. Store temporal neighbors in ring buffer for efficient sliding window
4. Cache recency weights for common time deltas (memoization)
5. Use `dashmap` for concurrent field reconstructions (if parallelizing)

## Success Criteria Validation

- [ ] Field reconstruction accuracy >85% on test dataset
- [ ] Source attribution precision >95%
- [ ] Confidence correlation >0.80 with accuracy
- [ ] Field reconstruction latency <2ms P95
- [ ] Zero allocations in consensus hot path
- [ ] All unit tests pass
- [ ] All property tests pass (10K runs)
- [ ] Integration tests demonstrate end-to-end correctness
