# Task 005: Source Attribution System

**Status:** Pending
**Priority:** P0 (Critical Path)
**Estimated Effort:** 2 days
**Dependencies:** Task 002 (CA3 Dynamics), Task 004 (Global Integration)

## Objective

Implement source monitoring system that precisely tracks which episode fields are recalled vs reconstructed vs imagined vs consolidated. Provide per-field source confidence scores and alternative hypotheses from System 2 reasoning to prevent false memory formation.

## Integration Points

**Uses:**
- `/engram-core/src/completion/hippocampal.rs` - CA3/CA1 completion from Task 002
- `/engram-core/src/completion/evidence_integration.rs` - Integrated evidence from Task 004
- `/engram-core/src/completion/hypothesis.rs` - System2Reasoner for alternative hypotheses
- `/engram-core/src/completion/confidence.rs` - MetacognitiveConfidence

**Creates:**
- `/engram-core/src/completion/source_monitor.rs` - Source attribution logic
- `/engram-core/src/completion/alternative_hypotheses.rs` - Multiple completion generation
- `/engram-core/tests/source_attribution_tests.rs` - Attribution accuracy tests

## Theoretical Foundations from Research

### Source Monitoring Framework (Johnson, Hashtroudi, & Lindsay, 1993)

Source monitoring is the process of attributing memories to their origins. Three types:

**1. External Source Monitoring:** Which external source provided information
- Engram: Which episode contributed this field (temporal neighbor ID tracking)

**2. Internal Source Monitoring:** Did I imagine this or experience it?
- Engram: Is field recalled (in partial_episode) or reconstructed (from neighbors)?

**3. Reality Monitoring:** Perceived vs imagined memories
- Engram: Reconstructed vs Imagined vs Consolidated source types

**Critical Finding from Johnson et al.:**
> "People struggle to distinguish sources when time delay increases, source information not explicitly encoded, or suggested information is plausible."

**Engram Application:** Explicit source tracking is essential. Confidence alone doesn't indicate source.

**Implementation:** `SourceMap` tracks source per field:
```rust
pub struct SourceMap {
    field_name: String,
    source: MemorySource,           // Explicit source label
    source_confidence: Confidence,   // Independent attribution confidence
    evidence_pathway: EvidencePathway, // How we determined source
}
```

### Reality Monitoring Failures (Lindsay & Johnson, 2000)

Key experiments demonstrating false memory formation:
- Imagined events become "recalled" memories
- Suggested details incorporated into genuine memories
- **High confidence doesn't predict source accuracy**

**Solution for Engram:** Independent source attribution metadata. Don't rely on completion confidence to distinguish recalled vs reconstructed.

**Evidence Pathway Based Attribution:**

**Direct Recall Pathway:**
```
Field present in partial cue → Recalled
Confidence = cue_strength from partial episode
```

**Reconstruction Pathway:**
```
Field completed from temporal neighbors → Reconstructed
Confidence = neighbor consensus (agreement ratio)
Source: NeighborEvidence tracking from Task 001
```

**Consolidation Pathway:**
```
Field derived from semantic patterns → Consolidated
Confidence = pattern_strength (p-value from M6)
Source: RankedPattern tracking from Task 003
```

**Imagination Pathway:**
```
Field speculated with low confidence → Imagined
Confidence <0.3 (below reconstructed_threshold)
```

### Alternative Hypotheses: Preventing Confabulation (Koriat & Goldsmith, 1996)

Metacognitive monitoring framework: Generate multiple alternative answers and choose based on confidence.

**Regulation:** Output can be withheld if confidence doesn't meet threshold (CA1 gating in Task 002).

**Engram Implementation:**
```rust
pub struct AlternativeHypothesisGenerator {
    num_hypotheses: usize,  // Default: 3
    // Generate alternatives by varying pattern weights
    // Ensures diverse completions (>0.3 embedding similarity distance)
}
```

**Benefit:** User sees alternatives, preventing single-path confabulation. System 2 reviews System 1 completions (Kahneman, 2011).

**Acceptance Criterion:** Ground truth in top-3 alternatives >70% of time (Task 009 validation).

### System 2 Reasoning (Kahneman, 2011)

**Fast System 1:** Pattern completion via CA3 dynamics (Task 002)
**Slow System 2:** Deliberative reasoning and checking via alternative hypotheses

**Alternative generation:** Vary parameters (pattern weights, sparsity, threshold) to produce diverse completions. System 2 reviews System 1 output for plausibility and consistency.

## Detailed Specification

### 1. Source Monitoring Engine

```rust
// /engram-core/src/completion/source_monitor.rs

use crate::Confidence;
use crate::completion::{MemorySource, SourceMap, IntegratedField, CompletedEpisode};

pub struct SourceMonitor {
    /// Minimum confidence for "recalled" label (default: 0.85)
    recalled_threshold: Confidence,

    /// Minimum confidence for "consolidated" label (default: 0.70)
    consolidated_threshold: Confidence,

    /// Minimum confidence for "reconstructed" label (default: 0.50)
    reconstructed_threshold: Confidence,
}

impl SourceMonitor {
    pub fn new() -> Self;

    /// Attribute source for each completed field
    pub fn attribute_sources(
        &self,
        partial: &PartialEpisode,
        integrated_fields: &HashMap<String, IntegratedField>,
        ca3_convergence: &ConvergenceStats,
    ) -> SourceMap;

    /// Determine memory source based on evidence pathway
    fn classify_source(
        &self,
        field_name: &str,
        integrated: &IntegratedField,
        in_partial: bool,
    ) -> (MemorySource, Confidence);

    /// Compute source confidence (how sure we are of source attribution)
    fn compute_source_confidence(
        &self,
        source: MemorySource,
        field_confidence: Confidence,
        evidence_consensus: f32,
    ) -> Confidence;
}

impl SourceMonitor {
    fn classify_source(
        &self,
        field_name: &str,
        integrated: &IntegratedField,
        in_partial: bool,
    ) -> (MemorySource, Confidence) {
        // If in partial.known_fields → Recalled
        if in_partial {
            return (MemorySource::Recalled, self.recalled_threshold);
        }

        // If dominated by global patterns → Consolidated
        if integrated.global_contribution > 0.7 {
            if integrated.confidence >= self.consolidated_threshold {
                return (MemorySource::Consolidated, integrated.confidence);
            }
        }

        // If dominated by local context → Reconstructed
        if integrated.local_contribution > 0.7 {
            if integrated.confidence >= self.reconstructed_threshold {
                return (MemorySource::Reconstructed, integrated.confidence);
            }
        }

        // Low confidence → Imagined (speculative)
        (MemorySource::Imagined, integrated.confidence)
    }
}
```

### 2. Alternative Hypotheses Generation

```rust
// /engram-core/src/completion/alternative_hypotheses.rs

use crate::{Episode, Confidence};
use crate::completion::{PartialEpisode, System2Reasoner};

pub struct AlternativeHypothesisGenerator {
    /// System 2 reasoner from existing completion module
    reasoner: System2Reasoner,

    /// Number of hypotheses to generate (default: 3)
    num_hypotheses: usize,

    /// Diversity threshold for hypotheses (default: 0.3 similarity distance)
    diversity_threshold: f32,
}

impl AlternativeHypothesisGenerator {
    pub fn new(num_hypotheses: usize) -> Self;

    /// Generate alternative completions with different pattern weights
    pub fn generate_alternatives(
        &self,
        partial: &PartialEpisode,
        primary_completion: &Episode,
        ranked_patterns: &[RankedPattern],
    ) -> Vec<(Episode, Confidence)>;

    /// Vary pattern weights to produce diverse completions
    fn vary_pattern_weights(
        &self,
        base_weights: &[f32],
        variation: usize,
    ) -> Vec<f32>;

    /// Ensure hypotheses are diverse (min similarity distance)
    fn ensure_diversity(
        &self,
        hypotheses: Vec<(Episode, Confidence)>,
    ) -> Vec<(Episode, Confidence)>;
}
```

## Acceptance Criteria

1. **Source Attribution Precision:** >90% correct classification of recalled vs reconstructed fields on test datasets
2. **Source Confidence Calibration:** Source confidence correlates >0.85 with attribution accuracy
3. **Alternative Hypotheses Quality:** Ground truth in top-3 alternatives >70% of time
4. **Diversity:** Alternative hypotheses differ by >0.3 embedding similarity
5. **Performance:** Source attribution <1ms P95, alternative generation <5ms P95

## Testing Strategy

**Unit Tests:** Source classification rules, confidence thresholds, alternative diversity

**Ground Truth Tests:** Deliberate corruptions with known sources; measure precision/recall per source type

**Property Tests:** Source confidence ∈ [0,1], alternatives contain primary completion, diversity constraints met

## Success Criteria Validation

- [ ] Attribution precision >90%
- [ ] Source confidence correlation >0.85
- [ ] Top-3 alternative coverage >70%
- [ ] Alternative diversity >0.3
- [ ] All tests pass
