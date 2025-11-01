# Task 007: Gossip Protocol for Consolidation State - Implementation Guide

**Date**: 2025-11-01  
**Status**: Analysis Complete - Ready for Implementation  
**Task**: Milestone 14, Task 007  
**Owner**: TBD  
**Estimated Duration**: 4 days

## Overview

This directory contains comprehensive analysis and implementation guidance for Task 007: Gossip Protocol for Consolidation State. The task implements anti-entropy gossip for eventual consistency of semantic memory patterns across distributed Engram nodes.

## Documentation Files

### 1. TASK_007_SUMMARY.txt (Quick Start)
**Size**: 4.2 KB  
**Best For**: Quick reference, key decisions, next steps  
**Contains**:
- Executive summary of findings
- 10 critical discoveries with line numbers
- Implementation readiness assessment
- Next steps checklist

**Start here** if you have limited time.

### 2. TASK_007_IMPLEMENTATION_ANALYSIS.md (Complete Guide)
**Size**: 24 KB  
**Best For**: Deep understanding, detailed planning  
**Contains** (726 lines):
1. Executive summary
2. Consolidation implementation location (8 files identified)
3. SemanticPattern structure details
4. Existing SWIM infrastructure analysis
5. Hash functions in current use
6. Consolidation state lifecycle
7. Cluster module structure to create
8. Implementation approach specific to consolidation
9. Detailed file location reference (all with line numbers)
10. Specific code snippet locations
11. Configuration parameters
12. Acceptance criteria mapping
13. Critical integration decisions
14. Performance targets and notes
15. Dependencies and versions
16. Testing strategy summary

**Read this** for complete understanding before implementation.

### 3. TASK_007_FILE_REFERENCE.md (Code Location Index)
**Size**: 8.4 KB  
**Best For**: Navigating the codebase  
**Contains**:
- Absolute paths for all files
- Line numbers for every critical section
- Files to create (8 total)
- Files to modify (4 total)
- Implementation sequence
- Quick reference tables

**Use this** as your navigation guide while coding.

## Quick Facts

### What Exists (Ready to Use)

- **SemanticPattern**: `/engram-core/src/completion/consolidation.rs:45-65`
- **ConsolidationSnapshot**: `/engram-core/src/completion/consolidation.rs:104-113`
- **ConsolidationService trait**: `/engram-core/src/consolidation/service.rs:54-66`
- **Consolidation scheduler**: `/engram-core/src/completion/scheduler.rs:206-216`
- **Pattern hashing examples**: `/engram-core/src/consolidation/pattern_detector.rs:499-506`
- **SWIM spec**: `/roadmap/milestone-14/001_cluster_membership_swim_pending.md`

### What's Missing (To Create)

- Cluster module at `/engram-core/src/cluster/`
- 8 new source files (gossip + conflict modules)
- Integration tests
- Configuration file `engram-cli/config/cluster.toml`

## Critical Integration Points

### Hook Point 1: Consolidation Trigger
**File**: `/engram-core/src/completion/scheduler.rs`  
**Lines**: 206-216  
**Action**: Add gossiper update after snapshot generation

```rust
let snapshot = {
    let mut engine = self.engine.write().await;
    engine.ripple_replay(&episodes);
    engine.snapshot()
};

// ADD HERE: gossiper.rebuild_from_snapshot(&snapshot)
```

### Hook Point 2: Dependency Addition
**File**: `/engram-core/Cargo.toml`  
**Action**: Add `sha2 = "0.10"`

### Hook Point 3: Module Registration
**File**: `/engram-core/src/lib.rs`  
**Action**: Add `pub mod cluster;`

## Implementation Sequence

1. **Add dependency**: `sha2 = "0.10"` to Cargo.toml
2. **Create cluster module**: `/engram-core/src/cluster/` with 8 files
3. **Implement Merkle tree**: Based on spec lines 157-363
4. **Implement gossip messages**: Based on spec lines 365-411
5. **Implement gossiper**: Based on spec lines 415-609
6. **Implement conflict resolution**: Based on spec lines 741-827
7. **Hook to scheduler**: Add gossiper call after consolidation
8. **Write tests**: Unit, property, and integration tests
9. **Verify**: Run acceptance criteria checks

## Key Design Decisions Already Made

### 1. Hash Function
- **Decision**: SHA-256 from `sha2` crate (v0.10)
- **Why**: Cryptographically secure, deterministic, proven
- **Status**: Approved in task spec, not in dependencies yet

### 2. Merkle Tree Structure
- **Depth**: 12 (supports 4096 patterns)
- **Leaf partitioning**: hash(pattern_id) % 4096
- **Incremental updates**: O(log N) recomputation on pattern change

### 3. Pattern Identification
- **Pattern IDs**: Deterministic (hash of sorted source episodes)
- **Hash in gossip**: Metadata only (id, confidence, timestamp), NOT embedding
- **Reason**: Save CPU, sufficient for delta detection

### 4. Conflict Resolution
- **Strategy**: ConfidenceVotingResolver (spec lines 759-827)
- **Causality**: Vector clocks from Task 008
- **Fallback**: Confidence voting when concurrent
- **Merging**: Citations union, embedding averaging

### 5. Gossip Transport
- **Layer**: Piggyback on SWIM (Task 001)
- **Message**: Attach Merkle root to PING messages
- **Metadata field**: "consolidation_root"
- **Interval**: 60 seconds (default)

## File Creation Checklist

- [ ] Create `/engram-core/src/cluster/mod.rs`
- [ ] Create `/engram-core/src/cluster/gossip/mod.rs`
- [ ] Create `/engram-core/src/cluster/gossip/merkle_tree.rs`
- [ ] Create `/engram-core/src/cluster/gossip/consolidation.rs`
- [ ] Create `/engram-core/src/cluster/gossip/messages.rs`
- [ ] Create `/engram-core/src/cluster/conflict/mod.rs`
- [ ] Create `/engram-core/src/cluster/conflict/strategies.rs`
- [ ] Create `/engram-core/src/cluster/conflict/vector_clock.rs`
- [ ] Create `/engram-core/tests/cluster_gossip_integration.rs`

## Modification Checklist

- [ ] Modify `/engram-core/Cargo.toml` - Add sha2 dependency
- [ ] Modify `/engram-core/src/lib.rs` - Add cluster module
- [ ] Modify `/engram-core/src/completion/scheduler.rs` - Add gossiper hook
- [ ] Optionally modify `/engram-core/src/consolidation/service.rs` - Extend trait

## Testing Checklist

- [ ] Unit tests for Merkle tree (4+ tests)
- [ ] Property tests for determinism (2+ tests)
- [ ] Integration tests (3-node cluster, partition healing)
- [ ] Convergence time validation
- [ ] Compression effectiveness
- [ ] Performance benchmarks

## Acceptance Criteria Status

From task spec lines 1147-1157:

- [ ] Convergence within 10 rounds (3-node test)
- [ ] Delta sync <10% bandwidth (measure bytes)
- [ ] Deterministic conflict resolution (unit tests)
- [ ] No lost patterns (partition healing test)
- [ ] Incremental Merkle updates (O(log N) verified)
- [ ] Compression works (output <50% of full state)
- [ ] Vector clock ordering (concurrent detection)
- [ ] Metrics exposed (prometheus integration)

## Performance Targets

From task spec lines 1158-1166:

| Target | Goal | Testing |
|--------|------|---------|
| Merkle update | <1ms | Benchmark incremental |
| Root comparison | <10μs | Direct timing |
| Divergence detection | <10ms | Tree walk benchmark |
| Delta sync bandwidth | <5KB/s avg | Measure over 60s |
| Convergence time | <10min for 100 nodes | 10 rounds test |
| Memory overhead | <50MB per node | Profiling |

## Key Resources

**Task Specification**:
- Full spec: `/roadmap/milestone-14/007_gossip_consolidation_state_pending.md` (1222 lines)
- Task 001 (SWIM): `/roadmap/milestone-14/001_cluster_membership_swim_pending.md` (620+ lines)
- Task 008 (Conflict): `/roadmap/milestone-14/008_conflict_resolution_divergent_consolidations_pending.md`

**Code References**:
- Consolidation: `/engram-core/src/completion/consolidation.rs`
- Scheduler: `/engram-core/src/completion/scheduler.rs`
- Service: `/engram-core/src/consolidation/service.rs`
- Pattern detection: `/engram-core/src/consolidation/pattern_detector.rs`

## Dependency Matrix

Task 007 depends on:
- **Task 001** (SWIM): Not started (spec ready)
- **Task 008** (Conflict): Concurrent (vector clocks needed)

Task 007 is depended on by:
- **Task 009** (Distributed Queries): Needs gossip-synchronized state
- **Task 011** (Jepsen Testing): Will validate gossip convergence

## Success Criteria

Task 007 is complete when:

1. All 8 new files created and implement spec exactly
2. Merkle tree tests pass (unit + property)
3. Gossip integration tests pass (3-node cluster)
4. Partition healing validated
5. All acceptance criteria met
6. Performance targets achieved
7. Zero clippy warnings
8. Fully integrated with consolidation scheduler

## Next Steps (Today)

1. Read TASK_007_SUMMARY.txt (5 min)
2. Read TASK_007_IMPLEMENTATION_ANALYSIS.md (20 min)
3. Review TASK_007_FILE_REFERENCE.md (10 min)
4. Review task spec lines 1-150 (architecture overview)
5. Create /engram-core/src/cluster/ directory
6. Start implementing cluster module structure

## Questions & Decisions

### Q: Should Merkle tree live in ConsolidationService or ConsolidationGossiper?
**A**: ConsolidationGossiper (cluster/gossip/consolidation.rs) - keeps gossip concerns isolated

### Q: How to trigger Merkle updates?
**A**: Hook in scheduler.rs:216 after snapshot generation, call gossiper.rebuild_from_snapshot()

### Q: What to do about embeddings?
**A**: Don't hash them - hash metadata only (id, confidence, timestamp). Saves CPU, still sufficient for delta detection.

### Q: How to ensure determinism?
**A**: Sort patterns by ID before hashing, use stable ordering throughout, property tests verify.

## Contact & Escalation

For questions about this analysis:
1. Review TASK_007_IMPLEMENTATION_ANALYSIS.md (section 12 has critical decisions)
2. Check TASK_007_FILE_REFERENCE.md for exact file locations
3. Review task spec lines 1-150 for architectural overview

---

**Report Generated**: 2025-11-01  
**Thoroughness**: Very Thorough (3 comprehensive documents, 36 KB total)  
**Status**: Ready for Implementation
