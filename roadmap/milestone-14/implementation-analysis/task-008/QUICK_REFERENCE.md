# Task 008 - Quick Reference Guide

**Status**: Ready for Implementation
**Analysis Location**: `/Users/jordan/Workspace/orchard9/engram/TASK_008_IMPLEMENTATION_ANALYSIS.md`

## Key Findings Summary

### 1. What Exists (No Work Needed)
- Confidence type with arithmetic operations (lib.rs:87-290)
- Episode structure with metadata (memory.rs:206-259)
- SemanticPattern base structure (completion/consolidation.rs:45-65)
- Cosine similarity implementations (cognitive/priming/ and cognitive/interference/)
- Embedding averaging with Kahan summation (consolidation/pattern_detector.rs)
- Pattern strength computation (consolidation/pattern_detector.rs:348+)

### 2. What Needs to Be Created (New Code)
- **VectorClock** - Causality tracking (NOT in codebase)
- **ConflictDetector** - Similarity/overlap analysis
- **PatternMerger** - All 6 merging strategies
- **ConflictResolver** - Orchestrator
- **Tests** - Unit, property-based, integration

### 3. What Needs to Be Modified (Existing Files)
1. `engram-core/src/completion/consolidation.rs`
   - Add: vector_clock, origin_node_id, generation fields to SemanticPattern
   - Change: source_episodes from Vec to HashSet

2. `engram-core/src/consolidation/service.rs`
   - Initialize vector clock on pattern creation

3. `engram-core/src/metrics/cluster_consolidation.rs`
   - Add conflict resolution metrics

4. `engram-core/Cargo.toml`
   - Add proptest dev dependency

## Critical Code Locations

| Component | File | Lines |
|-----------|------|-------|
| Confidence | `lib.rs` | 87-290 |
| Episode | `memory.rs` | 206-259 |
| SemanticPattern | `completion/consolidation.rs` | 45-65 |
| EpisodicPattern | `consolidation/pattern_detector.rs` | 54-73 |
| Cosine Similarity | `cognitive/priming/semantic.rs` | (method) |
| Embedding Average | `consolidation/pattern_detector.rs` | 302-324 |

## Implementation Checklist

### Phase 1: Vector Clock (Day 1)
- [ ] Create `engram-core/src/cluster/conflict/mod.rs`
- [ ] Implement VectorClock struct with increment/merge/compare
- [ ] Define CausalOrder enum
- [ ] Add serde support
- [ ] Unit tests for causality

### Phase 2: Conflict Detection (Day 1-2)
- [ ] Create `engram-core/src/cluster/conflict/detection.rs`
- [ ] Implement cosine_similarity (copy from existing code)
- [ ] Implement Jaccard overlap
- [ ] Implement conflict classification
- [ ] Tests for each conflict type

### Phase 3: Pattern Merging (Day 2-3)
- [ ] Create `engram-core/src/cluster/conflict/merger.rs`
- [ ] Implement confidence-weighted merging
- [ ] Implement conservative dual strategy
- [ ] Implement confidence voting
- [ ] Implement multi-membership
- [ ] Tests for each strategy

### Phase 4: Orchestrator (Day 3)
- [ ] Create `engram-core/src/cluster/conflict/resolver.rs`
- [ ] Implement ConflictResolver
- [ ] Wire up detector + merger
- [ ] Add metrics hooks

### Phase 5: Integration & Testing (Day 4-5)
- [ ] Modify SemanticPattern to include new fields
- [ ] Update consolidation service
- [ ] Add conflict metrics
- [ ] Integration tests with Task 007
- [ ] Property-based tests for determinism
- [ ] Run `make quality` - fix clippy warnings
- [ ] Performance validation

## Key Thresholds

```
Merge Similarity: 0.85     (if similarity > 0.85, merge patterns)
Overlap Threshold: 0.50    (if overlap > 0.50, count as shared)
Confidence Voting: 0.15    (if conf diff > 0.15, voting wins)
Confidence Penalty: 0.30   (30% max penalty for uncertainty)
Min Confidence: 0.10       (floor on all confidence calculations)
```

## Confidence Calculation Formula

```
Merged Confidence = max(
    (base_confidence) × (1 - penalty),
    min_confidence = 0.1
)

where:
  base_confidence = (c1 + c2) / 2
  penalty = (1 - similarity) × 0.3
```

## Example: Two Nodes Merging Patterns

**Node A**: Episodes {e1, e2, e3} → Pattern_A (confidence 0.85, embedding [0.2, 0.8, ...])
**Node B**: Episodes {e1, e2, e4} → Pattern_B (confidence 0.78, embedding [0.25, 0.75, ...])

**Resolution**:
1. Cosine similarity = 0.9
2. Episode overlap (Jaccard) = 2/4 = 0.5
3. Conflict type = DivergentConsolidation (high overlap + similarity)
4. Strategy = ConfidenceWeightedMerge (similarity > 0.85)
5. Merged confidence = max((0.815) × (1 - 0.03), 0.1) = 0.791
6. Merged episodes = {e1, e2, e3, e4}
7. Merged embedding = 0.85×[0.2...] + 0.78×[0.25...] / 1.63 (normalized)

## Testing Strategy

### Unit Tests Required
- Vector clock: causality ordering
- Similarity: cosine computation
- Overlap: Jaccard index
- Conflict classification: all 4 types
- Merge strategies: all 6 approaches
- Confidence adjustments: bounds checking

### Property-Based Tests (proptest)
- **Determinism**: resolve(A,B) produces same output with same inputs
- **Commutativity**: resolve(A,B) ≈ resolve(B,A) (after ordering)
- **Associativity**: resolve(resolve(A,B), C) ≈ resolve(A, resolve(B,C))
- **No inflation**: merged_confidence ≤ max(c1, c2)
- **Information loss**: quantify and limit

### Integration Tests
- Two-node gossip exchange
- Convergence validation
- Pattern propagation timing

## Dependency Status

### Already Available
- serde / chrono / dashmap / tokio / sha2
- All needed for implementation

### New Dependency (for testing)
- proptest = "1.4" (add to [dev-dependencies])

## Performance Targets

| Operation | Target | Notes |
|-----------|--------|-------|
| Conflict detection | <1ms | Per pattern pair |
| Pattern merge | <3ms | 768-dim embeddings |
| Full resolution | <5ms | Detection + merge |
| Gossip convergence | 10 rounds | For 100-node cluster |
| Vector clock overhead | <1KB | Per clock |

## Integration with Task 007 (Gossip)

**Gossip Flow**:
1. Node A sends Merkle root to Node B
2. Roots differ → request patterns
3. Node B sends `remote_pattern: SemanticPattern`
4. Node A calls `resolver.resolve(&local, &remote)`
5. Apply ResolutionResult to local state
6. Increment vector clock, continue gossip

**Convergence**: All nodes reach identical patterns within O(log N) rounds with >99.9% probability

## Quality Checklist

- [ ] All clippy warnings fixed (make quality)
- [ ] Tests passing (cargo test)
- [ ] Performance within targets
- [ ] No unwrap() in library code
- [ ] Large types by reference (&[f32; 768])
- [ ] Determinism proven via property tests
- [ ] Serde support complete
- [ ] Diagnostics run successfully

## References

**Task File**: `roadmap/milestone-14/008_conflict_resolution_divergent_consolidations_pending.md` (1299 lines)

**Key Papers**:
- Fidge (1988) - Vector clocks
- Shapiro et al. (2011) - CRDTs
- Nader & Hardt (2009) - Reconsolidation
- Leutgeb et al. (2007) - Pattern separation
- Yassa & Stark (2011) - CA3 pattern completion

---

**Next Steps**: Start with Phase 1 (Vector Clock) - most foundational component
