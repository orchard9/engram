# Task 007 Expansion: Gossip Protocol for Consolidation State

## Status: COMPLETE

**Date**: 2025-11-01
**Expanded From**: 33 lines → 1221 lines (38KB)
**Quality Level**: Matches tasks 001-003 comprehensive specifications

## Deliverables

### Primary File Created
- **Path**: `roadmap/milestone-14/007_gossip_consolidation_state_pending.md`
- **Size**: 1221 lines, 38KB
- **Structure**: 28 major sections with complete implementation details

### Reference Updated
- **File**: `roadmap/milestone-14/004-012_remaining_tasks_pending.md`
- **Change**: Replaced inline summary with pointer to detailed spec
- **Lines**: 86-103 updated

## Research Foundation

### Anti-Entropy Gossip Protocols
- **Demers et al. 1987**: Epidemic algorithms for database replication
- **Amazon Dynamo (2007)**: Merkle tree-based anti-entropy
- **Cassandra**: Per-range Merkle trees with incremental updates
- **Riak**: Active Anti-Entropy as separate subsystem
- **Convergence Theory**: O(log N) rounds, infection-style propagation

### Merkle Tree State Fingerprinting
- **Bitcoin/IPFS**: Content-addressable storage with SHA-256
- **Cassandra Implementation**: Binary trees, MurmurHash3, incremental updates
- **Design Parameters**: Depth 12 (4096 leaves), SHA-256 hashing
- **Performance**: O(log N) updates, O(log N) comparisons
- **Memory**: ~2N hashes for N items (internal nodes)

### Consolidation State Model
- **State Items**: Semantic patterns (embedding, confidence, citations, timestamps)
- **Versioning**: Vector clocks for causality tracking
- **Partitioning**: Consistent hash for deterministic pattern→leaf mapping
- **Divergence Sources**: Independent consolidation, network partitions, timing skew
- **Convergence**: Bounded time guarantee with gossip continuation

## Technical Specification Details

### Core Data Structures (500+ lines)

#### ConsolidationMerkleTree
```rust
pub struct ConsolidationMerkleTree {
    depth: usize,                                    // Tree depth (12 = 4096 leaves)
    nodes: Vec<HashMap<usize, MerkleHash>>,         // Level → index → hash
    patterns: HashMap<String, SemanticPattern>,      // Pattern storage
    leaf_contents: HashMap<usize, Vec<String>>,     // Leaf → pattern IDs
}
```

**Key Operations**:
- `upsert_pattern()`: Insert/update with O(log N) path recomputation
- `root_hash()`: Get state fingerprint (32-byte SHA-256)
- `find_divergence()`: Recursive subtree comparison
- `pattern_to_leaf()`: Consistent hash placement

#### ConsolidationGossiper
```rust
pub struct ConsolidationGossiper {
    merkle_tree: Arc<RwLock<ConsolidationMerkleTree>>,
    generation: Arc<RwLock<u64>>,
    config: GossipConfig,
    conflict_resolver: Arc<dyn ConflictResolver>,
    pending_merges: DashMap<String, Vec<SemanticPattern>>,
}
```

**Key Operations**:
- `gossip_round()`: Select K random peers, exchange with each
- `gossip_with_peer()`: Root exchange → divergence detection → delta sync
- `delta_sync()`: Bi-directional pattern transfer
- `merge_pattern()`: Conflict resolution integration

#### Gossip Messages
- **MerkleRootExchange**: Root hash, generation, pattern count
- **SubtreeRequest/Response**: Drill down into divergent subtrees
- **PatternRequest/Sync**: Transfer patterns for divergent leaves

### Protocol Flow

**Every 60 seconds per node**:
1. Select 3 random peers from SWIM membership
2. For each peer:
   - Exchange Merkle roots (32 bytes)
   - If roots differ:
     - Recursively find divergent subtrees
     - Request missing patterns from peer
     - Send our patterns peer is missing
     - Merge with conflict resolution
   - If roots match: already in sync, skip

**Bandwidth analysis**:
- Root exchange: 32 bytes
- Typical delta: 1-5 patterns × 50KB = 50-250KB
- Gossip interval: 60s
- Average: <5KB/s per node

### Conflict Resolution Integration

**Conflict types**:
1. **Causal ordering** (vector clocks): Remote newer → accept, local newer → keep
2. **Concurrent updates**: Confidence voting (higher confidence wins)
3. **Equal confidence**: Merge citations (union), average embeddings (weighted)

**Resolver interface** (from Task 008):
```rust
#[async_trait]
pub trait ConflictResolver: Send + Sync {
    async fn resolve_pattern_conflict(
        &self,
        local: &SemanticPattern,
        remote: &SemanticPattern,
    ) -> Result<SemanticPattern, ConflictError>;
}
```

### Integration Points

#### SWIM Gossip Transport (Task 001)
- Piggyback Merkle root on SWIM ping messages (metadata field)
- Trigger async delta sync when roots differ
- Reuse UDP transport for consolidation messages

#### Consolidation Service
- Hook `update_cache()` to trigger Merkle tree update
- Rebuild tree from snapshot on consolidation run
- Increment generation counter

#### Metrics
- `consolidation_gossip_rounds_total`: Counter
- `consolidation_patterns_merged_total`: Counter
- `consolidation_gossip_divergent_ranges`: Histogram
- `consolidation_gossip_bandwidth_bytes`: Histogram

## Testing Strategy (450+ lines)

### Unit Tests (150 lines)
1. **Merkle tree incremental updates**: Root changes on pattern update
2. **Divergence detection**: Finds differing subtrees correctly
3. **Gossip convergence**: Two nodes sync via delta transfer

### Property-Based Tests (100 lines)
1. **Merkle determinism**: Same patterns → same root (insertion order independent)
2. **Gossip convergence**: N nodes with random patterns converge in <10 rounds
3. **Conflict resolution**: Same inputs → same merged output (deterministic)

### Integration Tests (200 lines)
1. **3-node consolidation gossip**: Independent consolidation → gossip convergence
2. **Partition healing**: 5-node cluster split 2-3 → heal → converge
3. **Convergence time**: 100 nodes converge in <10 rounds (O(log N) validation)

### Convergence Validation
**Formal proof sketch**:
- Gossip is infection-style: update spreads exponentially
- After k rounds, P(node lacks update) < (1 - K/N)^k
- For K=3, N=100, k=10: P(divergent) < 0.1%

**Property test validation**:
```rust
proptest! {
    fn test_gossip_convergence(
        num_nodes in 3..10,
        patterns_per_node in 5..20,
    ) {
        // Create N nodes with random patterns
        // Run gossip for 10 rounds
        // Assert all nodes have same root hash
    }
}
```

## Performance Targets

| Operation | Target | Rationale |
|-----------|--------|-----------|
| Merkle tree update | <1ms | O(log N) path recomputation |
| Root comparison | <10μs | Single hash comparison |
| Divergence detection | <10ms | Recursive tree walk (10K patterns) |
| Bandwidth per node | <5KB/s | 250KB delta / 60s interval |
| Convergence time | <10 min | 10 rounds × 60s (100 nodes) |
| Memory overhead | <50MB | Merkle tree + patterns (10K) |

## Acceptance Criteria

- [ ] 3-node cluster converges in <10 gossip rounds
- [ ] Bandwidth <10% of full state transfer (delta sync efficiency)
- [ ] Deterministic conflict resolution (property-tested)
- [ ] No lost patterns during partition healing (merge all divergent)
- [ ] Incremental Merkle updates (O(log N), not full rebuild)
- [ ] Compression reduces bandwidth by >50% for large transfers
- [ ] Vector clock causality detection working
- [ ] Metrics exposed (rounds, merges, bandwidth, convergence time)

## Files Specified

### To Create (7 files)
1. `engram-core/src/cluster/gossip/mod.rs` - Module definition
2. `engram-core/src/cluster/gossip/merkle_tree.rs` - Merkle tree implementation
3. `engram-core/src/cluster/gossip/consolidation.rs` - Gossiper with delta sync
4. `engram-core/src/cluster/gossip/messages.rs` - Gossip message types
5. `engram-core/src/cluster/conflict/mod.rs` - Conflict resolver trait
6. `engram-core/src/cluster/conflict/strategies.rs` - Resolution strategies
7. `engram-core/src/cluster/conflict/vector_clock.rs` - Causality tracking

### To Modify (4 files)
1. `engram-core/src/cluster/membership.rs` - Integrate with SWIM
2. `engram-core/src/completion/consolidation.rs` - Hook Merkle updates
3. `engram-core/src/consolidation/service.rs` - Trigger gossip on consolidation
4. `engram-cli/config/cluster.toml` - Add gossip config section

## Dependencies Added

```toml
# Cryptography
sha2 = "0.10"           # SHA-256 for Merkle hashing

# Async traits
async-trait = "0.1"     # ConflictResolver trait

# Compression
lz4 = "1.24"            # Large pattern transfer compression
```

## Integration with Milestone 14

### Task Dependencies
- **Task 001 (SWIM)**: Provides membership and gossip transport layer
- **Task 008 (Conflict)**: Provides ConflictResolver implementation
- **Task 009 (Query)**: Uses gossip-synchronized consolidation state
- **Task 011 (Jepsen)**: Validates convergence properties formally

### Critical Path
Not on critical path - can be implemented in parallel with tasks 005-006 after task 001 completes.

## Implementation Roadmap

### Phase 1: Core Merkle Tree (1 day)
- Implement `ConsolidationMerkleTree` with SHA-256 hashing
- Unit test incremental updates and divergence detection
- Property test determinism (insertion order independence)

### Phase 2: Gossip Protocol (1.5 days)
- Implement `ConsolidationGossiper` with peer selection
- Implement delta sync (root exchange → divergence → transfer)
- Integration test: 3-node convergence

### Phase 3: Integration (1 day)
- Hook consolidation service to update Merkle tree
- Piggyback on SWIM gossip transport
- Add metrics and observability

### Phase 4: Validation (0.5 days)
- Property test: N-node convergence in O(log N) rounds
- Integration test: Partition healing
- Performance validation: bandwidth <5KB/s average

## References

1. **Demers, A., et al. (1987)**. "Epidemic Algorithms for Replicated Database Maintenance". ACM PODC. [Foundational epidemic algorithm theory]

2. **DeCandia, G., et al. (2007)**. "Dynamo: Amazon's Highly Available Key-value Store". SOSP. [Merkle tree anti-entropy design]

3. **Merkle, R. (1987)**. "A Digital Signature Based on a Conventional Encryption Function". CRYPTO. [Merkle tree cryptographic foundations]

4. **Apache Cassandra Documentation**. "Anti-Entropy Repair". https://cassandra.apache.org [Production Merkle tree implementation]

5. **Riak Documentation**. "Active Anti-Entropy". https://docs.riak.com [AAE subsystem design patterns]

## Key Insights

### Why Merkle Trees?
- **Efficient comparison**: O(log N) hash comparisons vs O(N) item checks
- **Incremental updates**: O(log N) path recomputation on single change
- **Bandwidth efficiency**: Only transfer divergent subtrees
- **Tamper evident**: Any change produces different root

### Why Anti-Entropy Gossip?
- **Scalability**: O(1) message load per node (independent of cluster size)
- **Convergence**: O(log N) rounds with high probability
- **Simplicity**: No coordinator, no consensus protocol
- **Robustness**: Self-healing, tolerates arbitrary partitions

### Why Not Alternatives?
- **Broadcast**: O(N^2) network traffic, doesn't scale
- **Consensus (Raft/Paxos)**: Requires quorum, blocks on partitions
- **Vector clocks only**: Can detect divergence but not find changes efficiently
- **Full state transfer**: 500MB per sync for 10K patterns (unacceptable)

## Production Readiness Checklist

- [ ] Merkle tree persistent (WAL integration for crash recovery)
- [ ] Compression configurable (enable/disable via config)
- [ ] Metrics dashboard (Grafana panels for gossip health)
- [ ] Alerting (convergence time >15 minutes)
- [ ] Runbook (partition healing procedure)
- [ ] Load testing (100-node cluster, 10K patterns)

## Known Limitations

1. **Eventual consistency only**: No linearizability guarantees (by design)
2. **Convergence time bounded but not instant**: <10 minutes typical
3. **Conflict resolution is heuristic**: Confidence voting may not always choose "correct" version
4. **Merkle tree memory overhead**: 2N hashes ≈ 64N bytes for N patterns
5. **No cryptographic security**: SHA-256 for efficiency, not collision resistance

## Future Enhancements (Out of Scope)

1. **Adaptive gossip interval**: Faster during active consolidation, slower when stable
2. **Bloom filters**: Quick "no difference" check before Merkle comparison
3. **Partial tree exchange**: Send only hash deltas instead of full subtrees
4. **Compression tuning**: Adaptive compression based on pattern size distribution
5. **Multi-level Merkle trees**: Hierarchical fingerprints for very large clusters (>1000 nodes)

---

**Task 007 expansion is complete and ready for implementation.**
