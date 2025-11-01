# Engram Task 007: Implementation Deep Dive Report
## Gossip Protocol for Consolidation State

**Date**: 2025-11-01
**Task**: M14 Task 007 - Gossip Protocol for Consolidation State
**Status**: Pending Implementation
**Thoroughness Level**: Very Thorough

---

## Executive Summary

Task 007 implements anti-entropy gossip protocol for eventual consistency of consolidation state across cluster nodes. This report provides exact file paths, implementation details, and specific integration approach based on current codebase analysis.

**Key Finding**: The codebase is well-structured with consolidation state clearly separated and accessible for gossip implementation. The cluster module does not yet exist and must be created.

---

## 1. CONSOLIDATION IMPLEMENTATION LOCATION

### 1.1 Core Files

**File**: `/Users/jordan/Workspace/orchard9/engram/engram-core/src/completion/consolidation.rs`
- **Lines**: 45-65 - SemanticPattern struct definition
- **Key Structure**:
```rust
pub struct SemanticPattern {
    pub id: String,                          // Pattern identifier
    pub embedding: [f32; 768],              // 768-dim semantic embedding
    pub source_episodes: Vec<String>,       // Contributing episode IDs
    pub strength: f32,                      // Pattern coherence (0.0-1.0)
    pub schema_confidence: Confidence,      // Confidence measure
    pub last_consolidated: DateTime<Utc>,  // Consolidation timestamp
}
```

**File**: `/Users/jordan/Workspace/orchard9/engram/engram-core/src/completion/consolidation.rs`
- **Lines**: 104-113 - ConsolidationSnapshot (immutable output from consolidation runs)
```rust
pub struct ConsolidationSnapshot {
    pub generated_at: DateTime<Utc>,
    pub patterns: Vec<SemanticPattern>,
    pub stats: ConsolidationStats,
}
```

**File**: `/Users/jordan/Workspace/orchard9/engram/engram-core/src/consolidation/service.rs`
- **Lines**: 54-66 - ConsolidationService trait (abstraction for consolidation state management)
- **Key Methods**:
  - `cached_snapshot(&self) -> Option<ConsolidationSnapshot>` - Get current snapshot
  - `update_cache(&self, snapshot: &ConsolidationSnapshot, source: ConsolidationCacheSource)` - Update with new snapshot
  - `recent_updates(&self) -> Vec<BeliefUpdateRecord>` - Get recent changes

**File**: `/Users/jordan/Workspace/orchard9/engram/engram-core/src/consolidation/service.rs`
- **Lines**: 69-83 - InMemoryConsolidationService (current implementation)
  - Caches consolidation snapshots in memory (RwLock<Option<ConsolidationSnapshot>>)
  - Tracks belief update records in VecDeque
  - Maintains alert log path for durability

### 1.2 Consolidation Triggers

**File**: `/Users/jordan/Workspace/orchard9/engram/engram-core/src/completion/scheduler.rs`
- **Lines**: 206-216 - Consolidation execution and snapshot generation
```rust
let snapshot = {
    let mut engine = self.engine.write().await;
    engine.ripple_replay(&episodes);
    engine.snapshot()
};

store
    .consolidation_service()
    .update_cache(&snapshot, ConsolidationCacheSource::Scheduler);
```

**File**: `/Users/jordan/Workspace/orchard9/engram/engram-core/src/completion/scheduler.rs`
- **Lines**: 124-126 - Consolidation interval configuration
  - Default: 300 seconds (5 minutes)
  - Configurable via SchedulerConfig

**Trigger Points for Merkle Tree Updates**:
1. After `engine.snapshot()` generation (line 210)
2. After `store.consolidation_service().update_cache()` call (line 214)
3. Ideal hook: New method `update_merkle_tree(&snapshot)` in gossiper

---

## 2. SEMANTIC PATTERN REPRESENTATION

### 2.1 SemanticPattern Structure Details

**File**: `/Users/jordan/Workspace/orchard9/engram/engram-core/src/completion/consolidation.rs`
- **Lines**: 46-65

**Critical Fields for Merkle Hashing**:
```rust
// What to hash (not embedding to save computation):
- id: String                           // Deterministic identifier
- embedding: [f32; 768]               // Can be hashed or excluded
- source_episodes: Vec<String>        // Episode references
- strength: f32                        // Pattern quality metric
- schema_confidence: Confidence       // Trust measure (Confidence type)
- last_consolidated: DateTime<Utc>    // Temporal ordering
```

**Current Hash Usage Pattern**:
- Files use `std::collections::hash_map::DefaultHasher`
- Located in `/Users/jordan/Workspace/orchard9/engram/engram-core/src/consolidation/pattern_detector.rs`
- **Lines**: 499-506
```rust
fn compute_pattern_hash(source_episodes: &[String]) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    
    let mut hasher = DefaultHasher::new();
    for id in source_episodes {
        id.hash(&mut hasher);
    }
    hasher.finish()
}
```

**Pattern Comparison Functions Already Implemented**:
- File: `/Users/jordan/Workspace/orchard9/engram/engram-core/src/completion/consolidation.rs`
- Lines: 317-336 - Embedding similarity using cosine product
- Lines: 329-336 - Embedding distance using Euclidean norm

### 2.2 Confidence Type

**Current Implementation**: `Confidence` is used throughout, likely with methods like `.raw()` for numeric extraction.

**For Gossip Conflict Resolution**: Can use confidence as tiebreaker when vector clocks are concurrent (higher confidence wins).

---

## 3. EXISTING SWIM GOSSIP INFRASTRUCTURE (Task 001)

### 3.1 Task 001 Status

**File**: `/Users/jordan/Workspace/orchard9/engram/roadmap/milestone-14/001_cluster_membership_swim_pending.md`
- **Status**: PENDING (not yet implemented)
- **Estimated Duration**: 3-4 days
- **Dependencies**: None (foundational)

### 3.2 SWIM Components Defined in Task 001 Spec

**Key Data Structures** (from lines 56-177):

1. **NodeInfo** - Contains node metadata:
   - `id: String` - Unique node identifier
   - `addr: SocketAddr` - SWIM protocol address
   - `api_addr: SocketAddr` - gRPC endpoint
   - `state: NodeState` - Alive/Suspect/Dead/Left
   - `incarnation: u64` - Version counter
   - `spaces: Vec<String>` - Memory spaces hosted

2. **SwimMessage** - Protocol messages:
   - Ping/Ack for failure detection
   - PingReq/ACK for indirect probing
   - Gossip with Vec<MembershipUpdate>

3. **SwimMembership** - Main protocol handler:
   - Uses DashMap for members
   - Configurable with SwimConfig
   - Supports `probe_cycle()` and gossip dissemination

### 3.3 Consolidation Gossip Piggybacking Approach

**From Task 001 spec (lines 618-654)**:
- Piggyback Merkle root on PING messages
- Add metadata field: `"consolidation_root"` containing MerkleHash
- Attach via `attach_consolidation_gossip(&mut msg)`
- Handle received gossip via `handle_consolidation_gossip(from, metadata)`

**Current State**: SWIM membership not yet implemented, but spec is clear and ready.

---

## 4. HASH FUNCTIONS IN USE

### 4.1 Current Hash Implementations

**Standard Library DefaultHasher** (most common):
- File: `/Users/jordan/Workspace/orchard9/engram/engram-core/src/consolidation/pattern_detector.rs`
- Lines: 499-506, 547-565
- Returns u64
- Used for pattern deduplication and signatures

**Pattern Signature Computation**:
```rust
// Lines 547-565
fn compute_pattern_set_signature(patterns: &[EpisodicPattern]) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    
    let mut sorted_ids = pattern_ids.iter().collect::<Vec<_>>();
    sorted_ids.sort();
    
    let mut hasher = DefaultHasher::new();
    sorted_ids.hash(&mut hasher);
    patterns.len().hash(&mut hasher);
    total_occurrences.hash(&mut hasher);
    
    hasher.finish()
}
```

### 4.2 Dependencies Available for SHA256/Merkle

**Current Cargo.toml** (`/Users/jordan/Workspace/orchard9/engram/engram-core/Cargo.toml`):
- Does NOT include `sha2` or cryptographic hash crate
- Must add: `sha2 = "0.10"` (task spec line 1138)

**Crypto Libraries Not Currently Used**:
- No `blake3`, `xxhash`, or other hash crates in dependencies
- `sha2` is recommended in task spec

### 4.3 Pattern to Follow for Task 007

**Use SHA-256 for Merkle tree** (task spec lines 186-206):
```rust
use sha2::{Sha256, Digest};

impl MerkleHash {
    fn hash_pattern(pattern: &SemanticPattern) -> Self {
        let mut hasher = Sha256::new();
        
        // Hash metadata (not embedding for efficiency)
        hasher.update(pattern.id.as_bytes());
        hasher.update(&pattern.confidence.to_le_bytes());
        hasher.update(&pattern.updated_at.timestamp().to_le_bytes());
        
        Self(hasher.finalize().into())
    }
    
    fn combine(left: &MerkleHash, right: &MerkleHash) -> Self {
        let mut hasher = Sha256::new();
        hasher.update(&left.0);
        hasher.update(&right.0);
        Self(hasher.finalize().into())
    }
}
```

---

## 5. CONSOLIDATION STATE LIFECYCLE

### 5.1 Consolidation Runs and State Changes

**Trigger Points** (file: `/Users/jordan/Workspace/orchard9/engram/engram-core/src/completion/scheduler.rs`):

1. **Interval-based** (lines 124-126):
   - Default: Every 300 seconds
   - Configurable via `consolidation_interval_secs`

2. **Threshold-based** (lines 187-199):
   - Minimum 10 episodes required
   - Can be configured via `min_episodes_threshold`

3. **Load-based** (line 267):
   - Max 100 episodes per run (configurable)
   - Prioritizes recent episodes (line 265)

### 5.2 State Storage Points

**File**: `/Users/jordan/Workspace/orchard9/engram/engram-core/src/store.rs`
- **Lines**: 7-12 - MemoryStore has ConsolidationService member
```rust
consolidation_service: Arc<dyn ConsolidationService>,
```

**File**: `/Users/jordan/Workspace/orchard9/engram/engram-core/src/consolidation/service.rs`
- **Lines**: 69-83 - InMemoryConsolidationService implementation
  - `cache: RwLock<Option<ConsolidationSnapshot>>` - Current snapshot
  - `alerts: RwLock<VecDeque<BeliefUpdateRecord>>` - Change history

### 5.3 Where to Hook Merkle Tree Updates

**Primary Hook Point**: `/Users/jordan/Workspace/orchard9/engram/engram-core/src/completion/scheduler.rs`, after line 216

**Current Code**:
```rust
store
    .consolidation_service()
    .update_cache(&snapshot, ConsolidationCacheSource::Scheduler);
```

**Proposed Addition**:
```rust
// Add to consolidation_service trait:
// fn update_merkle_tree(&self, snapshot: &ConsolidationSnapshot);

// Then call it:
store
    .consolidation_service()
    .update_merkle_tree(&snapshot);
```

**Alternative Approach** (cleaner):
- Pass gossiper to ConsolidationService during initialization
- Call gossiper's `rebuild_from_snapshot()` automatically

### 5.4 BeliefUpdateRecord Structure

**File**: `/Users/jordan/Workspace/orchard9/engram/engram-core/src/consolidation/service.rs`
- **Lines**: 36-52
```rust
pub struct BeliefUpdateRecord {
    pub pattern_id: String,
    pub confidence_delta: f32,
    pub citation_delta: i32,
    pub novelty: f32,
    pub generated_at: DateTime<Utc>,
    pub snapshot_generated_at: DateTime<Utc>,
    pub source: ConsolidationCacheSource,
}
```

**For Gossip**: These changes can be extracted and used as audit trail of what patterns changed.

---

## 6. CLUSTER MODULE STRUCTURE (TO BE CREATED)

### 6.1 Directory to Create

**Path**: `/Users/jordan/Workspace/orchard9/engram/engram-core/src/cluster/`

**Files to Create** (from task spec lines 829-837):
1. `mod.rs` - Gossip module exports
2. `gossip/mod.rs` - Gossip submodule
3. `gossip/merkle_tree.rs` - Merkle tree implementation
4. `gossip/consolidation.rs` - ConsolidationGossiper
5. `gossip/messages.rs` - Gossip message types
6. `conflict/mod.rs` - Conflict resolution trait
7. `conflict/strategies.rs` - Conflict resolution implementations
8. `conflict/vector_clock.rs` - Vector clock for causality

**Module Hierarchy**:
```
engram-core/src/cluster/
├── mod.rs
├── gossip/
│   ├── mod.rs              (exports)
│   ├── merkle_tree.rs      (ConsolidationMerkleTree)
│   ├── consolidation.rs    (ConsolidationGossiper)
│   └── messages.rs         (ConsolidationGossip enum)
└── conflict/
    ├── mod.rs              (ConflictResolver trait)
    ├── strategies.rs       (ConfidenceVotingResolver)
    └── vector_clock.rs     (VectorClock)
```

### 6.2 Integration Points

**File to Modify**: `/Users/jordan/Workspace/orchard9/engram/engram-core/src/lib.rs`
- Add: `pub mod cluster;`

**File to Modify**: `/Users/jordan/Workspace/orchard9/engram/engram-core/src/completion/scheduler.rs`
- Import consolidation gossiper
- Call gossiper.update_pattern() or rebuild_from_snapshot() after consolidation

**File to Modify**: `/Users/jordan/Workspace/orchard9/engram/engram-core/src/consolidation/service.rs`
- Trait should optionally accept gossiper
- Call gossiper methods on update_cache()

**File to Modify**: `/Users/jordan/Workspace/orchard9/engram/engram-core/Cargo.toml`
- Add: `sha2 = "0.10"` (for Merkle trees)
- Add: `lz4 = "1.24"` (for pattern compression, if not already present)

---

## 7. IMPLEMENTATION APPROACH SPECIFIC TO CONSOLIDATION

### 7.1 Consolidation-Specific Challenges

**Challenge 1: Pattern Identity**
- Pattern IDs are generated deterministically from source episodes
- **Solution**: Use pattern ID directly as leaf key in Merkle tree
- Located in: `/Users/jordan/Workspace/orchard9/engram/engram-core/src/consolidation/pattern_detector.rs`, line 288

**Challenge 2: Embedding Storage**
- Embeddings are [f32; 768] - expensive to hash in full
- **Solution**: Hash metadata only (id, confidence, timestamp, version)
- Don't hash embedding itself (saves CPU, sufficient for delta detection)
- See task spec lines 189-197

**Challenge 3: State Divergence**
- Different nodes consolidate different episodes
- **Solution**: Use generation counter (incremented per consolidation run)
- Track in gossip exchange (task spec line 381-382)

**Challenge 4: Consolidation Timing**
- Nodes consolidate asynchronously
- **Solution**: Gossip is eventually consistent - eventual convergence acceptable
- Target: <10 minutes for 100-node cluster

### 7.2 Merkle Tree Design for Consolidation

**Tree Structure** (from task spec lines 135-141):
```
Depth 12 binary tree = 4096 leaf slots
Each pattern → deterministic leaf via hash(pattern_id)
Leaf hash = SHA-256(sorted patterns in leaf)
Internal nodes = SHA-256(left_hash || right_hash)
Root = fingerprint of entire consolidation state
```

**Partitioning Strategy**:
```rust
fn pattern_to_leaf(&self, pattern_id: &str) -> usize {
    use std::hash::{Hash, Hasher};
    use std::collections::hash_map::DefaultHasher;
    
    let mut hasher = DefaultHasher::new();
    pattern_id.hash(&mut hasher);
    let hash = hasher.finish();
    
    let leaf_count = 1 << self.depth;  // 2^12 = 4096
    (hash as usize) % leaf_count
}
```

### 7.3 Integration with ConsolidationSnapshot

**Current Data Available**:
- `snapshot.patterns: Vec<SemanticPattern>` - All current patterns
- `snapshot.generated_at: DateTime<Utc>` - When snapshot created
- `snapshot.stats: ConsolidationStats` - Metadata about consolidation run

**Required Additions to Gossip**:
1. Generation counter (incremented each consolidation)
2. Root hash of Merkle tree
3. Leaf contents for delta sync

### 7.4 BeliefUpdateRecord Integration

**Current Usage**:
- File: `/Users/jordan/Workspace/orchard9/engram/engram-core/src/consolidation/service.rs`, lines 36-52
- Tracks pattern confidence deltas, citation deltas, novelty

**For Gossip Protocol**:
- Can extract changed patterns from BeliefUpdateRecord
- Use for efficient gossip (send only recently changed patterns)
- Track pattern versions using timestamps from BeliefUpdateRecord

---

## 8. DETAILED FILE LOCATION REFERENCE

### Critical Existing Files

| File | Lines | Purpose | For Task 007 |
|------|-------|---------|------------|
| `engram-core/src/completion/consolidation.rs` | 45-65 | SemanticPattern definition | Hash this struct |
| `engram-core/src/completion/consolidation.rs` | 104-113 | ConsolidationSnapshot | Input to merkle update |
| `engram-core/src/consolidation/service.rs` | 54-66 | ConsolidationService trait | Extend with gossip methods |
| `engram-core/src/consolidation/service.rs` | 69-83 | InMemoryConsolidationService | Will hold merkle tree |
| `engram-core/src/completion/scheduler.rs` | 206-216 | Consolidation trigger | Hook gossip update here |
| `engram-core/src/store.rs` | 7-12 | MemoryStore.consolidation_service | Access point for gossiper |
| `engram-core/src/consolidation/pattern_detector.rs` | 499-506 | Pattern hashing example | Follow this hash pattern |

### Files to Create

| File | Purpose | Based On |
|------|---------|----------|
| `engram-core/src/cluster/mod.rs` | Cluster module exports | Task 007 spec lines 831 |
| `engram-core/src/cluster/gossip/merkle_tree.rs` | Merkle tree impl | Task 007 spec lines 157-363 |
| `engram-core/src/cluster/gossip/consolidation.rs` | Consolidation gossiper | Task 007 spec lines 415-609 |
| `engram-core/src/cluster/gossip/messages.rs` | Gossip protocol messages | Task 007 spec lines 365-411 |
| `engram-core/src/cluster/conflict/mod.rs` | Conflict resolution trait | Task 007 spec lines 741-827 |

### Test Files

| File | Purpose | Lines |
|------|---------|-------|
| `engram-core/tests/cluster_gossip_integration.rs` | Integration tests | Task 007 spec lines 984-1085 |
| Unit tests within each module | Unit tests | Task 007 spec lines 850-980 |

---

## 9. SPECIFIC CODE SNIPPET LOCATIONS FOR REFERENCE

### 9.1 How to Access Consolidation State in Gossip

**From scheduler.rs (lines 213-215)**:
```rust
store
    .consolidation_service()
    .update_cache(&snapshot, ConsolidationCacheSource::Scheduler);
```

**In gossiper, access via trait**:
```rust
// In ConsolidationGossiper:
let current = self.consolidation_service.cached_snapshot();
if let Some(snapshot) = current {
    // Build merkle tree from snapshot.patterns
}
```

### 9.2 Pattern Struct to Hash

**From consolidation.rs (lines 47-52)**:
```rust
pub struct SemanticPattern {
    pub id: String,
    pub embedding: [f32; 768],
    pub source_episodes: Vec<String>,
    pub strength: f32,
    pub schema_confidence: Confidence,
    pub last_consolidated: DateTime<Utc>,
}
```

**Hash this (excluding embedding)**:
```rust
fn hash_pattern(pattern: &SemanticPattern) -> MerkleHash {
    let mut hasher = Sha256::new();
    hasher.update(pattern.id.as_bytes());
    hasher.update(&pattern.strength.to_le_bytes());
    hasher.update(&pattern.last_consolidated.timestamp().to_le_bytes());
    // Include confidence somehow (Confidence type TBD)
    MerkleHash(hasher.finalize().into())
}
```

### 9.3 Consolidation Trigger Point

**From scheduler.rs (lines 207-211)**:
```rust
let snapshot = {
    let mut engine = self.engine.write().await;
    engine.ripple_replay(&episodes);
    engine.snapshot()
};
```

**Add gossip hook here**:
```rust
let snapshot = {
    let mut engine = self.engine.write().await;
    engine.ripple_replay(&episodes);
    engine.snapshot()
};

// NEW: Update merkle tree for gossip
if let Some(gossiper) = &self.consolidation_gossiper {
    gossiper.rebuild_from_snapshot(&snapshot).await?;
}
```

---

## 10. CONFIGURATION PARAMETERS (From Task Spec)

### Recommended Defaults

| Parameter | Default | Line in Task | Notes |
|-----------|---------|-------------|-------|
| Gossip interval | 60s | 118 | How often to run gossip rounds |
| Fanout | 3 nodes | 448-449 | How many peers to gossip with per round |
| Merkle depth | 12 | 75, 137 | Supports 4096 patterns |
| Max patterns per sync | 100 | 451 | Batch size for delta transfers |
| Enable compression | true | 455 | Use LZ4 for large transfers |
| Suspect timeout | 5s | 135 (SWIM) | From Task 001 |

### Configuration File

**To create**: `engram-cli/config/cluster.toml` (task spec line 844)

```toml
[gossip]
interval_seconds = 60
fanout = 3
max_patterns_per_sync = 100
enable_compression = true

[merkle_tree]
depth = 12
leaf_count = 4096
```

---

## 11. ACCEPTANCE CRITERIA MAPPING

From task spec lines 1147-1157:

| Criterion | Testing Approach | Implementation Location |
|-----------|-----------------|------------------------|
| Convergence <10 rounds | Property tests over 100 rounds | tests/ |
| Delta sync <10% bandwidth | Measure transferred bytes | gossip/consolidation.rs |
| Deterministic conflict resolution | Unit tests with same inputs | conflict/strategies.rs |
| No lost patterns | 3-node partition+heal test | tests/cluster_gossip_integration.rs |
| Incremental Merkle updates | Benchmark merkle update latency | gossip/merkle_tree.rs |
| Compression works | Check output size <50% | gossip/consolidation.rs |
| Vector clock ordering | Unit tests on VectorClock | conflict/vector_clock.rs |
| Metrics exposed | Check prometheus exports | metrics/ |

---

## 12. CRITICAL INTEGRATION DECISIONS

### Decision 1: Hash Function for Merkle Tree

**Recommendation**: Use SHA-256 from `sha2` crate
- **Reason**: Cryptographically secure, deterministic, well-tested
- **Alternative**: Use DefaultHasher from std (but non-cryptographic)
- **Choice**: SHA-256 (matches task spec, more robust)
- **Add to Cargo.toml**: `sha2 = "0.10"`

### Decision 2: Where to Store Merkle Tree

**Option A**: In ConsolidationService (consolidation/service.rs)
- **Pros**: Centralizes consolidation state management
- **Cons**: Mixes consolidation (single-node) with gossip (multi-node)

**Option B**: In ConsolidationGossiper (cluster/gossip/consolidation.rs)
- **Pros**: Keeps gossip concerns separate
- **Cons**: Requires access to current snapshot

**Recommendation**: Option B (gossiper owns merkle tree)
- Store reference to consolidation_service in gossiper
- Update merkle tree when snapshot changes

### Decision 3: Conflict Resolution Strategy

**Recommendation**: ConfidenceVotingResolver (task spec lines 759-827)
- Uses vector clocks for causality (from Task 008)
- Falls back to confidence voting on concurrent updates
- Merges citations (episode references) from both versions

### Decision 4: Pattern Identification in Gossip

**Current**: Pattern ID is deterministic based on source episodes (pattern_detector.rs:288)
- Hash: `pattern_{hash(sorted_source_episodes)}`

**For Merkle**: Use pattern ID directly as key
- No need to recompute hash
- Deterministic ordering guaranteed

---

## 13. PERFORMANCE TARGETS AND IMPLEMENTATION NOTES

From task spec lines 1158-1166:

| Target | Implementation Approach |
|--------|----------------------|
| Merkle update <1ms | Pre-allocate tree nodes, avoid full rebuild |
| Root comparison <10μs | Single hash comparison, no traversal |
| Divergence detection <10ms | Binary tree search, O(log N) comparisons |
| Delta sync <5KB/s avg | Batch patterns, compress if >10KB |
| Convergence <10min for 100 nodes | 10 rounds × 60s interval = 600s OK |
| Memory <50MB per node | 10K patterns × ~5KB each = 50MB |

**Implementation Decision**: Use incremental Merkle updates
- On pattern upsert, recompute only affected path (O(log N) hashes)
- Not full tree rebuild on every pattern change
- Code: Lines 255-288 in task spec

---

## 14. DEPENDENCIES AND VERSIONS

### To Add to Cargo.toml

```toml
# Cryptography for Merkle trees
sha2 = "0.10"

# Compression for large pattern transfers (if not present)
lz4 = "1.24"

# Async trait support (likely already present)
async-trait = "0.1"
```

### Already Available in Cargo.toml

- `dashmap` - For concurrent pattern storage
- `tokio` - For async gossip
- `serde`/`bincode` - For serialization
- `chrono` - For timestamps
- `parking_lot` - For locks

---

## 15. TESTING STRATEGY SUMMARY

### Unit Tests (gossip/merkle_tree.rs)

1. **test_merkle_tree_incremental_update** - Verify root changes on pattern change
2. **test_merkle_divergence_detection** - Verify find_divergence() works
3. **test_pattern_to_leaf_determinism** - Ensure consistent leaf assignment
4. **test_merkle_path_recomputation** - Verify O(log N) recomputation

### Property Tests

1. **test_merkle_tree_determinism** - Same patterns in any order → same root
2. **test_gossip_convergence_property** - N nodes with gossip → all converge

### Integration Tests (tests/cluster_gossip_integration.rs)

1. **test_three_node_consolidation_gossip** - Basic gossip works
2. **test_partition_healing_convergence** - Partition + heal + gossip
3. **test_gossip_convergence_time** - Measure rounds to convergence
4. **test_compression_effectiveness** - Verify <50% bandwidth usage

---

## CONCLUSION

Task 007 implementation is ready to proceed with these specific integration points:

1. **Hook point**: After consolidation snapshot generation (scheduler.rs:216)
2. **State access**: Via consolidation_service trait
3. **Pattern source**: ConsolidationSnapshot.patterns vector
4. **Hash function**: SHA-256 from `sha2` crate
5. **Conflict resolution**: From Task 008 (VectorClock + confidence voting)
6. **SWIM integration**: Ready from Task 001 spec

All critical files are identified with exact line numbers. The codebase is well-structured for integration.

