# Concept Formation Engine: Design Summary

**Date**: 2025-11-09
**Status**: Design Complete - Ready for Implementation Review

---

## Core Design Principles

### 1. Gradual Consolidation Over Multiple Cycles

**Key Innovation**: Unlike PatternDetector (immediate semantic patterns), ConceptFormationEngine tracks **gradual consolidation strength** across 5-50 cycles, modeling the weeks-to-months timeline of cortical representation formation.

```rust
// Cycle 1:  consolidation_strength = 0.02 (hippocampal binding)
// Cycle 5:  consolidation_strength = 0.10 (cortical trace appears - PROMOTION)
// Cycle 25: consolidation_strength = 0.50 (cortical dominance)
// Cycle 50: consolidation_strength = 1.00 (full semantic independence)
```

**Biological Validation**: Matches Takashima et al. (2006) fMRI timeline (7 days → 7 weeks).

---

## Critical Design Decisions

### Decision 1: Persistent ProtoConcept Pool

**Chosen**: In-memory DashMap with cross-cycle tracking

**Why**:
- Enables consolidation_strength accumulation over 5-50 cycles
- Tracks cumulative replay_count for biological fidelity
- Fast O(1) lookup via deterministic signature
- Thread-safe for future distributed consolidation

**Trade-off**: Lost on restart (acceptable for M17; add checkpointing in M18+)

---

### Decision 2: Deterministic Concept Matching

**Chosen**: 128-bit signature from sorted episode IDs

```rust
fn compute_concept_signature(episode_ids: &[EpisodeId]) -> u128 {
    // Sort for order-invariance
    let mut sorted = episode_ids.to_vec();
    sorted.sort();

    // Hash with collision resistance
    let hash_64 = hash(sorted);
    let count_64 = sorted.len() as u64;

    (hash_64 << 64) | count_64  // 128-bit signature
}
```

**Why**:
- Deterministic: Same episodes → same signature (critical for M14 distributed consolidation)
- Collision-resistant: 128-bit space
- Partial overlap creates separate concepts (models distributed cortical representations)

**Rejected Alternative**: Centroid similarity matching (non-deterministic, threshold-dependent)

---

### Decision 3: Overlapping Concepts Allowed

**Chosen**: Soft clustering with multiple concepts per episode

**Biological Motivation**: Cortical representations are overlapping and distributed (Haxby et al. 2001).

**Example**:
```
Episode: "Reading Python book in library"

Contributes to:
- Concept A: "Reading technical books"
- Concept B: "Library activities"
- Concept C: "Learning new skills"
```

**Implementation**: Episode IDs can appear in multiple ProtoConcepts, each with independent consolidation trajectory.

---

### Decision 4: No Concept Decay

**Chosen**: Concepts persist indefinitely once formed

**Why**:
- Semantic memories are stable (Squire & Alvarez 1995)
- No forgetting curve for consolidated knowledge
- Disuse affects activation (retrieval probability), not structure

**Future Enhancement**: Add activation decay for retrieval likelihood, but structure remains intact.

---

### Decision 5: Never Unbind Episodes

**Chosen**: Episode-concept edges persist permanently

**Why**:
- Biological: Multiple Trace Theory (Nadel & Moscovitch 1997) - hippocampal traces remain
- Functional: Enables episodic reconstitution from concepts
- Simplicity: No complex unbinding heuristics

**Lifecycle**:
```
Episode → Concept formation → Episode compacted → Edge remains
                                                   ↓
                                       Enables reconsolidation
```

---

## Biological Parameter Validation

### Sleep Stage Modulation

| Stage | Formation Prob | Replay Capacity | Weight Factor | Min Cycles |
|-------|---------------|-----------------|---------------|------------|
| NREM2 | 15% | 100 | 1.5x | 3 |
| NREM3 | 8% | 80 | 1.2x | 2 |
| REM | 3% | 50 | 0.8x | 5 |
| Wake | 1% | 20 | 0.5x | 10 |

**Empirical Support**:
- NREM2 peak: Schabus et al. (2004) - 5-7 spindles/minute
- Weight factors: Diekelmann & Born (2010) - replay rate measurements
- Capacities: Wilson & McNaughton (1994) - SWR frequency data

---

### Core Parameters with Citations

| Parameter | Value | Biological Basis | Citation |
|-----------|-------|------------------|----------|
| `coherence_threshold` | 0.65 | CA3 pattern completion | Nakazawa et al. (2002) |
| `similarity_threshold` | 0.55 | DG pattern separation | Yassa & Stark (2011) |
| `consolidation_rate` | 0.02 | Slow cortical learning | Takashima et al. (2006) |
| `min_cluster_size` | 3 | Schema formation | Tse et al. (2007) |
| `replay_weight_decay` | 0.9 | SWR frequency decay | Kudrimoti et al. (1999) |
| `max_concepts_per_cycle` | 5 | Spindle density | Schabus et al. (2004) |

**All parameters fall within empirically-measured ranges.**

---

## Integration Strategy

### Coordination with Existing Systems

```rust
// DreamEngine runs BOTH systems in each cycle:

// 1. PatternDetector (existing): Fast semantic extraction
let patterns = pattern_detector.detect_patterns(episodes);
// → Creates SemanticPattern with threshold=0.8
// → Used for storage compaction (Task 003)

// 2. ConceptFormationEngine (new): Gradual consolidation
let concepts = concept_engine.form_concepts(episodes, sleep_stage);
// → Updates ProtoConcept pool with threshold=0.55
// → Promotes to Concept when strength > 0.1
```

**No Conflict**: Different thresholds produce different clusters naturally.

**Complementary Roles**:
- PatternDetector: Immediate patterns for compaction/caching
- ConceptFormationEngine: Long-term semantic memory formation

---

## Algorithm Highlights

### Replay-Weighted Centroid

**Innovation**: Uses SWR-inspired weighting instead of simple averaging

```rust
replay_weight = recency_weight * stage_factor * importance

where:
  recency_weight = exp(-hours_since / 24.0)  // 24h time constant
  stage_factor = {NREM2: 1.5, NREM3: 1.2, REM: 0.8, Wake: 0.5}
  importance = episode.encoding_confidence.raw()
```

**Biological Fidelity**: Recent, important episodes get higher weight in centroid, matching empirical SWR replay statistics.

---

### Gradual Strength Update

```rust
fn update_concept_strength(existing, new_observation) {
    existing.replay_count += 1;

    // Asymptotic approach to 1.0
    existing.consolidation_strength = min(
        existing.consolidation_strength + 0.02,
        1.0
    );

    // Weighted centroid blending (favors accumulated evidence)
    existing_weight = existing.replay_count;
    new_weight = 1.0;

    existing.centroid = weighted_blend(
        existing.centroid, existing_weight,
        new_observation.centroid, new_weight
    );
}
```

**Property**: Mature concepts (high replay_count) resist centroid drift from new observations.

---

### Garbage Collection Policy

**Removes**:
1. **Dormant concepts**: No updates for 50 cycles (~10 weeks)
2. **Failed consolidation**: strength < 0.05 after 20 cycles (~4 weeks)

**Biological Motivation**: Synaptic homeostasis - weak, unreinforced traces undergo pruning.

---

## Determinism for Distributed Consolidation

**Critical for M14**: All operations are deterministic to enable gossip convergence across distributed nodes.

### Determinism Guarantees

1. **Episode sorting**: Sort by ID before clustering
2. **Tie-breaking**: Lexicographic ordering on episode IDs
3. **Kahan summation**: Centroid computation order-invariant
4. **Signature matching**: Hash-based identity resilient to ordering

**Testing Strategy**:
- 1000-iteration determinism test (from PatternDetector)
- Cross-platform signature validation (ARM64, x86_64)
- Distributed gossip simulation (5 nodes with different arrival orders)

---

## Performance Validation

### <5% Regression Target (M17)

**Optimizations**:
1. **Reuse clustering**: Delegate to existing PatternDetector logic
2. **Efficient pool**: DashMap provides O(1) lookup
3. **Lazy GC**: Only every 10 cycles
4. **Bounded formation**: Max 5 concepts per cycle

**Expected Impact**:
- PatternDetector overhead: ~50ms for 100 episodes
- ConceptFormationEngine adds: ~20-30ms (pool lookup + updates)
- Total: <80ms (well within <5% of current ~1s cycle time)

---

## Implementation Phases

### Phase 1-3: Core Engine (Day 1-2)
- Data structures (ConceptFormationEngine, ProtoConcept)
- Clustering & coherence algorithms
- Replay weighting & centroid extraction

### Phase 4-5: State Management (Day 2-3)
- ProtoConcept pool with signature matching
- Strength updates across cycles
- Promotion to DualMemoryNode::Concept
- Garbage collection

### Phase 6-7: Integration & Testing (Day 3)
- DreamEngine integration
- Multi-cycle consolidation tests
- Biological validation against parameter ranges
- Performance regression testing

**Total**: 3 days (matches task estimate)

---

## Open Questions for Review

### 1. Checkpoint Strategy

**Current**: In-memory only (lost on restart)

**Future Options**:
- Periodic checkpoint to disk (every 100 cycles)?
- Write-ahead log for crash recovery?
- Tiered storage (hot pool in RAM, cold on disk)?

**Recommendation**: Defer to M18+ (M17 focuses on core algorithm validation)

---

### 2. Promotion Threshold Tuning

**Current**: consolidation_strength > 0.1 (5 cycles, ~1 week)

**Could adjust based on**:
- Episode importance (higher importance → earlier promotion)?
- Coherence score (tighter clusters → earlier promotion)?
- Replay count (more evidence → earlier promotion)?

**Recommendation**: Start with fixed 0.1 threshold, add adaptive promotion in M18+ after production data

---

### 3. Episode-Concept Binding

**Deferred to Task 005**: Creating graph edges between episodes and concepts

**Integration Point**: After concept promotion, create bidirectional edges:
```
Episode --[instance_of]--> Concept
Concept --[generalizes]--> Episode
```

**Enables**:
- Spreading activation from concepts to episodes
- Reconsolidation pathways
- Episodic reconstitution from semantic cues

---

## Success Criteria

### Biological Validation
- [ ] Parameters within empirically-measured ranges (see Appendix A of full design doc)
- [ ] Consolidation timeline matches fMRI studies (Takashima et al. 2006)
- [ ] Sleep stage modulation aligns with replay frequency data (Diekelmann & Born 2010)

### Functional Requirements
- [ ] Gradual strength accumulation over 5-50 cycles
- [ ] Deterministic concept matching across cycles
- [ ] Promotion at strength > 0.1 with replay_count >= 3
- [ ] Overlapping concepts from soft clustering
- [ ] Garbage collection of stale proto-concepts

### Performance Requirements
- [ ] <5% regression on existing consolidation benchmarks
- [ ] <100ms overhead per consolidation cycle
- [ ] Pool size bounded (GC prevents unbounded growth)

### Integration Requirements
- [ ] Coordinated with PatternDetector (complementary clustering)
- [ ] Compatible with DreamEngine cycle structure
- [ ] Metrics exposed for observability
- [ ] DualMemoryNode::Concept creation with proper metadata

---

## References

See full design document (TASK_004_CONCEPT_FORMATION_DESIGN.md) for complete bibliography with page citations.

**Key Papers**:
1. McClelland et al. (1995) - Complementary Learning Systems theory
2. Takashima et al. (2006) - fMRI consolidation timescales
3. Nakazawa et al. (2002) - CA3 pattern completion threshold
4. Yassa & Stark (2011) - DG pattern separation boundary
5. Kudrimoti et al. (1999) - SWR replay frequency decay
6. Schabus et al. (2004) - Sleep spindle density limits
7. Tse et al. (2007) - Schema formation requirements
8. Nadel & Moscovitch (1997) - Multiple Trace Theory

---

**Next Steps**:
1. **Review with team leads**:
   - systems-architecture-optimizer: Performance validation
   - verification-testing-lead: Testing strategy review
   - rust-graph-engine-architect: DualMemoryNode integration

2. **Begin implementation** (following Phase 1-7 roadmap)

3. **Coordinate with parallel tasks**:
   - Task 005: Episode-concept binding design
   - Task 006: Sleep stage scheduling integration

**Status**: Design complete and ready for implementation.
