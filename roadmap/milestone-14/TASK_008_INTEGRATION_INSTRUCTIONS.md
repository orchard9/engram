# Task 008 Integration Instructions

## Objective

Replace the brief Task 008 summary in `004-012_remaining_tasks_pending.md` with a reference to the comprehensive specification.

## Current State (Lines 120-146)

The current file contains this brief summary:

```markdown
## Task 008: Conflict Resolution for Divergent Consolidations (2 days)

**Objective**: Resolve conflicts when nodes independently consolidate differently.

**Key Components**:
- Vector clock ordering for causality
- Confidence-based voting (higher confidence wins)
- Last-write-wins with timestamp tiebreaker
- Merge strategies for semantic patterns

**Conflict Types**:
1. Same episode consolidated differently -> merge patterns
2. Different episodes consolidated to same semantic -> keep both with reduced confidence
3. Concurrent consolidations -> vector clock ordering

**Files**:
- `engram-core/src/cluster/conflict/mod.rs`
- `engram-core/src/cluster/conflict/strategies.rs`
- `engram-core/src/cluster/conflict/merger.rs`

**Acceptance Criteria**:
- All conflicts resolved deterministically
- No data loss (conservative merging)
- Confidence reflects uncertainty
- Convergence proof via property testing
```

## Replacement Text

Replace lines 120-146 with:

```markdown
## Task 008: Conflict Resolution for Divergent Consolidations (4-5 days)

**COMPREHENSIVE SPECIFICATION**: See dedicated task file
`008_conflict_resolution_divergent_consolidations_pending.md` (1,299 lines)

**Objective**: Implement deterministic, biologically-plausible conflict resolution mechanisms for distributed memory consolidation. Resolution must preserve memory integrity, maintain confidence calibration, and mirror hippocampal-neocortical conflict resolution strategies.

**Duration Updated**: Increased from 2 days to 4-5 days due to comprehensive scope including:
- Vector clock implementation for causality tracking
- Four distinct conflict types with specialized resolution strategies
- Confidence-weighted merging with similarity-based penalties
- Property-based tests proving determinism, commutativity, and associativity
- Biological plausibility validation against hippocampal pattern separation/completion

**Key Components**:
- `VectorClock` for causal ordering (Fidge 1988)
- `ConflictDetector` with similarity and overlap analysis
- `PatternMerger` with 6 resolution strategies
- `ConflictResolver` orchestrator with determinism guarantees

**Conflict Types** (4 types):
1. **Divergent Episode Consolidation**: Same episodes → different patterns
2. **Concurrent Pattern Creation**: Similar semantics, different episodes
3. **Concurrent Updates**: Same pattern updated by multiple nodes
4. **Episode Ownership**: Same episode claimed by multiple patterns

**Resolution Strategies** (6 strategies):
1. **ConfidenceWeightedMerge**: Similarity >0.85 → merge with weighted averaging
2. **ConservativeDual**: Similarity <0.60 → keep both, reduce confidence
3. **ConfidenceVoting**: Large conf difference → higher confidence wins
4. **VectorClockOrdering**: Causal order → use happens-after pattern
5. **MultiMembership**: Episode overlap → allow multi-membership
6. **NoAction**: No conflict detected

**Files to Create** (6 files):
- `engram-core/src/cluster/conflict/mod.rs` - Module exports and types
- `engram-core/src/cluster/conflict/detection.rs` - Conflict detection
- `engram-core/src/cluster/conflict/merger.rs` - Pattern merging algorithms
- `engram-core/src/cluster/conflict/resolver.rs` - Resolution orchestration
- `engram-core/src/cluster/conflict/vector_clock.rs` - Causality tracking
- `engram-core/tests/conflict_resolution_tests.rs` - Comprehensive tests

**Files to Modify** (4 files):
- `engram-core/src/cluster/gossip/consolidation.rs` - Integrate resolver
- `engram-core/src/decay/consolidation.rs` - Add vector clock to patterns
- `engram-core/src/metrics/cluster_consolidation.rs` - Add conflict metrics
- `engram-core/Cargo.toml` - Add proptest dependency

**Acceptance Criteria** (8 criteria):
- **Determinism**: Property tests prove resolution is deterministic
- **Commutativity**: resolve(A, B) ≈ resolve(B, A) (proven via proptest)
- **Associativity**: resolve(resolve(A, B), C) ≈ resolve(A, resolve(B, C))
- **No Data Loss**: Conservative strategies preserve all information
- **Confidence Calibration**: Merged confidence reflects uncertainty
- **Convergence**: 100 nodes converge within 10 gossip rounds (60s)
- **Biological Plausibility**: Strategies mirror DG/CA3 pattern separation/completion
- **Performance**: <5ms per conflict resolution (single-threaded)

**Biological Foundation**:
- DG pattern separation (similarity <0.6) → ConservativeDual strategy
- CA3 pattern completion (similarity >0.85) → ConfidenceWeightedMerge
- Reconsolidation interference → ConcurrentUpdate handling
- Overlapping engrams → MultiMembership strategy

**Performance Targets**:
- Conflict detection: <1ms per pattern pair
- Pattern merge: <3ms for 768-dimensional embeddings
- Full resolution: <5ms per conflict
- Convergence: 60 seconds for 100-node cluster (10 rounds × 6s interval)
- Memory overhead: <1KB per vector clock

**Research Foundation**:
- **Neuroscience**: Hippocampal pattern separation/completion (Leutgeb 2007, Yassa 2011), reconsolidation interference (Nader 2009), schema consistency (Tse 2007)
- **Distributed Systems**: Vector clocks (Fidge 1988), CRDTs (Shapiro 2011), operational transformation (Ellis 1989)

**Testing Strategy**:
- Unit tests: Vector clock causality, conflict detection, merge determinism
- Property-based tests (proptest): Determinism, commutativity, associativity proofs
- Integration tests: Two-node divergent consolidation with convergence validation
- Biological validation: Pattern separation/completion threshold testing

**Dependencies**: Task 007 (Gossip Protocol) must export consolidation state interfaces
```

## Step-by-Step Integration

### Option 1: Keep Original File Structure (Recommended)

1. **Do Not Delete** the original file `004-012_remaining_tasks_pending.md`
2. **Replace** lines 120-146 with the shorter reference above
3. **Keep** the comprehensive specification in the standalone file
4. **Benefit**: Maintains overview while allowing deep dive via dedicated file

### Option 2: Inline Full Specification

1. **Replace** lines 120-146 with the full 1,299-line specification
2. **Delete** the standalone file
3. **Downside**: Makes the combined task file extremely long (2,100+ lines)

### Option 3: Directory Structure (Most Organized)

1. **Create** directory structure:
   ```
   roadmap/milestone-14/
   ├── 004-012_remaining_tasks_pending.md (overview)
   ├── tasks/
   │   ├── 008_conflict_resolution_pending.md (full spec)
   │   └── ... (other future detailed specs)
   ```

2. **Update** line 120 in parent file to reference:
   ```markdown
   See: `tasks/008_conflict_resolution_pending.md`
   ```

## Recommended Approach

Use **Option 1** (Keep Original File Structure):

### Advantages:
- Quick overview preserved in `004-012_remaining_tasks_pending.md`
- Deep technical detail available in standalone file
- Matches pattern of Tasks 001-003 (each has own file)
- Easy to navigate (overview → detailed spec)

### File Structure:
```
roadmap/milestone-14/
├── 001_cluster_membership_swim_pending.md (666 lines - comprehensive)
├── 002_node_discovery_pending.md (standalone when created)
├── 003_network_partition_handling_pending.md (standalone when created)
├── 004-012_remaining_tasks_pending.md (overview of tasks 4-12)
├── 008_conflict_resolution_divergent_consolidations_pending.md (1,299 lines - NEW)
└── ... (other files)
```

### Update Required:

In `004-012_remaining_tasks_pending.md`, change line 121 from:
```markdown
**Objective**: Resolve conflicts when nodes independently consolidate differently.
```

To:
```markdown
**COMPREHENSIVE SPECIFICATION**: See `008_conflict_resolution_divergent_consolidations_pending.md`

**Objective**: Implement deterministic, biologically-plausible conflict resolution mechanisms for distributed memory consolidation.
```

And replace the rest (lines 122-146) with the summary version above.

## Verification Steps

After integration:

1. **File exists**: Confirm `008_conflict_resolution_divergent_consolidations_pending.md` is in milestone-14/
2. **Reference updated**: Line 121 in `004-012_remaining_tasks_pending.md` points to dedicated file
3. **Line count**: Dedicated file should be 1,299 lines
4. **Task dependencies**: Task 008 dependencies remain accurate (after Task 007)
5. **Critical path**: Update task dependency graph if duration changed (2d → 4-5d)

## Timeline Impact

**Original Timeline**: Task 008 allocated 2 days
**Updated Timeline**: Task 008 now requires 4-5 days

**Critical Path Impact**:
- If Task 008 is on critical path: Milestone extends by 2-3 days
- If Task 008 is parallel: May not impact overall timeline

**Updated Dependency Graph** (from original file):
```
007 (Gossip) ──> 008 (Conflict) ──────────────────> 009 (Distributed Query)
                 [2 days → 4-5 days]
```

**Mitigation**: Task 008 is NOT on critical path (Task 009 can start in parallel if needed)

## Next Actions

1. **Review**: Team reviews comprehensive specification for technical correctness
2. **Integrate**: Apply Option 1 changes to `004-012_remaining_tasks_pending.md`
3. **Commit**: Commit both files with message:
   ```
   feat(M14-Task008): Expand conflict resolution specification to 1,299 lines

   - Add comprehensive neuroscience foundation (hippocampal pattern separation/completion)
   - Implement vector clock causality tracking (Fidge 1988)
   - Design 6 resolution strategies with determinism guarantees
   - Specify 4 conflict types with specialized handling
   - Add property-based tests proving commutativity and associativity
   - Increase duration estimate from 2 days to 4-5 days for completeness
   - Maintain biological plausibility throughout design

   File: 008_conflict_resolution_divergent_consolidations_pending.md (1,299 lines)
   ```

4. **Update Timeline**: Adjust Milestone 14 timeline if Task 008 is on critical path

---

**Integration Status**: Ready
**File Location**: `/Users/jordan/Workspace/orchard9/engram/roadmap/milestone-14/008_conflict_resolution_divergent_consolidations_pending.md`
**Recommended Approach**: Option 1 (Keep Original File Structure)
**Action Required**: Update lines 120-146 in `004-012_remaining_tasks_pending.md`
