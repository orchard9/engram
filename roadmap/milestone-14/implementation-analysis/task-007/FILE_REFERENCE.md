# Task 007: Complete File Reference Guide

## Absolute Paths for All Implementation Resources

### Consolidation Implementation Files

**1. SemanticPattern Definition**
- Path: `/Users/jordan/Workspace/orchard9/engram/engram-core/src/completion/consolidation.rs`
- Lines: 45-65
- Type: Struct definition (core data to be gossipped)
- Key fields: id, embedding [768], source_episodes, strength, schema_confidence, last_consolidated

**2. ConsolidationSnapshot**
- Path: `/Users/jordan/Workspace/orchard9/engram/engram-core/src/completion/consolidation.rs`
- Lines: 104-113
- Type: Struct definition
- Contains: Vec<SemanticPattern>, stats, generated_at timestamp

**3. ConsolidationService Trait**
- Path: `/Users/jordan/Workspace/orchard9/engram/engram-core/src/consolidation/service.rs`
- Lines: 54-66
- Type: Trait definition (abstract interface)
- Methods: cached_snapshot(), update_cache(), alert_log_path(), recent_updates()

**4. InMemoryConsolidationService Implementation**
- Path: `/Users/jordan/Workspace/orchard9/engram/engram-core/src/consolidation/service.rs`
- Lines: 69-83
- Type: Struct implementation
- Storage: RwLock<Option<ConsolidationSnapshot>>, VecDeque<BeliefUpdateRecord>

**5. BeliefUpdateRecord**
- Path: `/Users/jordan/Workspace/orchard9/engram/engram-core/src/consolidation/service.rs`
- Lines: 36-52
- Type: Struct for tracking changes
- Fields: pattern_id, confidence_delta, citation_delta, novelty, timestamps

**6. Consolidation Scheduler (Main Trigger)**
- Path: `/Users/jordan/Workspace/orchard9/engram/engram-core/src/completion/scheduler.rs`
- Lines: 124-126 - Configuration (interval, thresholds)
- Lines: 206-216 - Consolidation execution and snapshot generation
- Lines: 173-246 - run_consolidation() method (PRIMARY HOOK POINT)

**7. ConsolidationEngine**
- Path: `/Users/jordan/Workspace/orchard9/engram/engram-core/src/completion/consolidation.rs`
- Lines: 11-26
- Type: Engine for running consolidation
- Methods: new(), ripple_replay(), snapshot(), patterns(), stats()

**8. Pattern Detector (Hash Pattern Reference)**
- Path: `/Users/jordan/Workspace/orchard9/engram/engram-core/src/consolidation/pattern_detector.rs`
- Lines: 499-506 - compute_pattern_hash() function
- Lines: 547-565 - compute_pattern_set_signature() function
- Usage pattern to follow: DefaultHasher for pattern identification

### Store and Infrastructure Files

**9. MemoryStore Definition**
- Path: `/Users/jordan/Workspace/orchard9/engram/engram-core/src/store.rs`
- Lines: 7-12 - consolidation_service member
- Lines: 100+ - consolidation_service() getter method
- Contains: ConsolidationService trait object

### Task 001 SWIM Reference

**10. SWIM Protocol Specification**
- Path: `/Users/jordan/Workspace/orchard9/engram/roadmap/milestone-14/001_cluster_membership_swim_pending.md`
- Lines: 56-177 - Data structures (NodeInfo, SwimMessage, SwimMembership)
- Lines: 618-654 - Piggyback consolidation gossip approach
- Status: PENDING IMPLEMENTATION (ready for Task 007 to integrate with)

### Current Directory Structure

**11. Module Locations**
- Completion modules: `/Users/jordan/Workspace/orchard9/engram/engram-core/src/completion/`
- Consolidation modules: `/Users/jordan/Workspace/orchard9/engram/engram-core/src/consolidation/`
- Storage modules: `/Users/jordan/Workspace/orchard9/engram/engram-core/src/storage/`
- Query modules: `/Users/jordan/Workspace/orchard9/engram/engram-core/src/query/`
- Store: `/Users/jordan/Workspace/orchard9/engram/engram-core/src/store.rs`
- Lib entry: `/Users/jordan/Workspace/orchard9/engram/engram-core/src/lib.rs`

### Task File Locations

**12. Current Task 007 File**
- Path: `/Users/jordan/Workspace/orchard9/engram/roadmap/milestone-14/007_gossip_consolidation_state_pending.md`
- Lines: Full task specification (1222 lines total)
- Status: PENDING

**13. Milestone 14 Directory**
- Path: `/Users/jordan/Workspace/orchard9/engram/roadmap/milestone-14/`
- Contains: All 12 task files plus supporting documentation

### Configuration Files

**14. Cargo.toml (Dependencies)**
- Path: `/Users/jordan/Workspace/orchard9/engram/engram-core/Cargo.toml`
- Status: Does NOT contain sha2 or cryptographic hashing libraries
- Action: Add `sha2 = "0.10"` before implementing Task 007

**15. Cluster Configuration (To Create)**
- Path: `/Users/jordan/Workspace/orchard9/engram/engram-cli/config/cluster.toml`
- Status: Does not exist yet
- Contents (to add): [gossip] and [merkle_tree] sections

## Files to Create (8 Total)

### Cluster Module Structure

```
/Users/jordan/Workspace/orchard9/engram/engram-core/src/cluster/
├── mod.rs
├── gossip/
│   ├── mod.rs
│   ├── merkle_tree.rs
│   ├── consolidation.rs
│   └── messages.rs
└── conflict/
    ├── mod.rs
    ├── strategies.rs
    └── vector_clock.rs
```

**File Creation Order:**
1. `/engram-core/src/cluster/mod.rs` - Module root
2. `/engram-core/src/cluster/gossip/mod.rs` - Gossip submodule
3. `/engram-core/src/cluster/gossip/merkle_tree.rs` - Merkle tree (lines 157-363 in spec)
4. `/engram-core/src/cluster/gossip/consolidation.rs` - Gossiper (lines 415-609 in spec)
5. `/engram-core/src/cluster/gossip/messages.rs` - Messages (lines 365-411 in spec)
6. `/engram-core/src/cluster/conflict/mod.rs` - Conflict trait
7. `/engram-core/src/cluster/conflict/strategies.rs` - Conflict resolution
8. `/engram-core/src/cluster/conflict/vector_clock.rs` - Vector clocks (from Task 008)

### Test Files to Create

**Test Integration File:**
- Path: `/Users/jordan/Workspace/orchard9/engram/engram-core/tests/cluster_gossip_integration.rs`
- Contents: Integration tests (spec lines 984-1085)

## Files to Modify (4 Total)

**1. Library Root**
- Path: `/Users/jordan/Workspace/orchard9/engram/engram-core/src/lib.rs`
- Action: Add `pub mod cluster;`

**2. Consolidation Service Trait**
- Path: `/Users/jordan/Workspace/orchard9/engram/engram-core/src/consolidation/service.rs`
- Action: Add method to update Merkle tree (optional, or integrate in ConsolidationGossiper)

**3. Consolidation Scheduler**
- Path: `/Users/jordan/Workspace/orchard9/engram/engram-core/src/completion/scheduler.rs`
- Action: Add gossiper update after line 216
- Exact location: After `store.consolidation_service().update_cache(&snapshot, ...)`

**4. Dependencies**
- Path: `/Users/jordan/Workspace/orchard9/engram/engram-core/Cargo.toml`
- Action: Add line: `sha2 = "0.10"`

## Critical Line Numbers for Integration

| File | Line(s) | Action | Purpose |
|------|---------|--------|---------|
| `consolidation.rs` | 45-65 | Reference | Hash this structure |
| `consolidation.rs` | 104-113 | Reference | Input to Merkle tree |
| `scheduler.rs` | 206-216 | HOOK HERE | Add gossiper call |
| `service.rs` | 54-66 | Reference/Extend | Access current snapshot |
| `store.rs` | 7-12 | Reference | Access consolidation_service |
| `pattern_detector.rs` | 499-506 | Reference | Follow hash pattern |

## Implementation Sequence

1. **Create cluster module structure** (files 1-2)
2. **Implement Merkle tree** (file 3, spec lines 157-363)
3. **Implement gossip messages** (file 4, spec lines 365-411)
4. **Implement gossiper** (file 5, spec lines 415-609)
5. **Implement conflict resolver** (files 6-8, spec lines 741-827)
6. **Add dependencies** (Cargo.toml)
7. **Integrate with scheduler** (scheduler.rs, add hook)
8. **Create tests** (test file)
9. **Verify and benchmark** (match acceptance criteria)

## Quick Reference: Where Things Are

| Component | File | Lines | Status |
|-----------|------|-------|--------|
| SemanticPattern | `completion/consolidation.rs` | 45-65 | EXISTS |
| ConsolidationSnapshot | `completion/consolidation.rs` | 104-113 | EXISTS |
| ConsolidationService | `consolidation/service.rs` | 54-66 | EXISTS |
| ConsolidationEngine | `completion/consolidation.rs` | 11-26 | EXISTS |
| Consolidation Scheduler | `completion/scheduler.rs` | 71-83, 206-216 | EXISTS |
| Pattern Hashing | `consolidation/pattern_detector.rs` | 499-506 | EXISTS (follow pattern) |
| SWIM Spec | `roadmap/milestone-14/001_*` | N/A | SPEC ONLY |
| Cluster Module | To create | N/A | MISSING |
| Merkle Tree | To create | N/A | MISSING |
| Conflict Resolver | To create | N/A | MISSING |
| Tests | To create | N/A | MISSING |

## Key Numeric References

- Pattern embedding dimension: 768
- Merkle tree depth: 12 (4096 leaves)
- Gossip interval: 60 seconds
- Consolidation interval: 300 seconds
- Max patterns per sync: 100
- Gossip fanout: 3 nodes
- Default suspect timeout: 5 seconds (SWIM)
- Hash output size: 32 bytes (SHA-256)

