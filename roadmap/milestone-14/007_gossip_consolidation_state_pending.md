# Task 007: Gossip Protocol for Consolidation State

**Status**: Pending
**Estimated Duration**: 4 days
**Dependencies**: Task 001 (SWIM), Task 008 (Conflict Resolution)
**Owner**: TBD

## Objective

Implement anti-entropy gossip protocol for eventual consistency of consolidation state across cluster nodes. Uses Merkle tree fingerprinting for efficient delta synchronization and integrates with SWIM's gossip transport layer. Ensures semantic patterns converge across all nodes even during network partitions.

## Research Foundation

### Anti-Entropy Gossip Protocols

**The fundamental problem**: In a distributed memory consolidation system, each node independently discovers semantic patterns through background processing. When node A consolidates episodes E1+E2 into pattern P1, and node B consolidates E3+E4 into pattern P2, how do they efficiently discover and merge these updates?

**Naive approach fails**: Broadcasting every consolidation change is O(N^2) network traffic. With 100 nodes consolidating at 1Hz, that's 10,000 messages/second - unacceptable.

**Anti-entropy solution (Demers et al. 1987, "Epidemic Algorithms for Replicated Database Maintenance")**:
Nodes periodically exchange state summaries (Merkle roots) with random peers. Only when summaries differ do they perform delta sync. Expected convergence time is O(log N) gossip rounds with probability approaching 1.

**Dynamo's approach (DeCandia et al. 2007)**:
Amazon Dynamo pioneered Merkle tree-based anti-entropy for distributed key-value stores. Each node maintains a Merkle tree over its data partitions. During gossip:
1. Exchange Merkle roots (32 bytes)
2. If roots differ, recursively compare subtree hashes
3. Identify divergent leaf nodes
4. Transfer only those changed entries

**Key insight**: Merkle trees reduce state comparison from O(N items) to O(log N tree depth). For Engram with 10K semantic patterns, comparison requires ~14 hash comparisons instead of 10K item checks.

**Cassandra's refinement**:
- Merkle trees built per-range (consistent hashing ranges)
- Incremental tree updates on write (not full rebuild)
- Bloom filters for quick "no difference" checks
- Background repair process runs anti-entropy continuously

**Riak's optimization**:
- Lazy Merkle tree construction (build on first comparison)
- Configurable tree depth vs fanout tradeoff
- AAE (Active Anti-Entropy) as separate subsystem from read/write path
- Entropy bounds: quantify max divergence between nodes

### Merkle Trees for State Fingerprinting

**Merkle tree structure (Merkle 1987)**:
Binary tree where:
- Leaf nodes = hash(data item)
- Internal nodes = hash(left_child || right_child)
- Root = fingerprint of entire dataset

**Properties**:
- **Incremental updates**: Changing one item updates O(log N) hashes (path to root)
- **Efficient comparison**: Different roots → descend tree to find divergent subtrees
- **Tamper-evident**: Any data change produces different root hash
- **Parallelizable**: Subtrees can be compared concurrently

**Bitcoin/IPFS approach**:
- Content-addressable storage using SHA-256 Merkle trees
- Persistent immutable trees (functional data structures)
- Each version has unique root hash
- DAG structure allows efficient diff computation

**Cassandra's implementation details**:
- Tree depth: log₂(range_size) - typically 10-15 levels
- Branching factor: Binary (2) for simplicity, though higher fanout possible
- Hash function: MurmurHash3 (fast, good distribution)
- Incremental updates: Rebuild affected path on every write
- Memory overhead: ~2N hashes for N items (parent nodes)

**For Engram consolidation state**:
- Items: Semantic patterns with (pattern_id, confidence, citations, timestamp)
- Partitioning: Hash pattern_id into tree leaves (deterministic placement)
- Update frequency: On every consolidation run (~60s interval)
- Tree parameters: Depth 12 (4096 leaves), SHA-256 hashing
- Persistence: Merkle roots stored in WAL for crash recovery

### State Representation

**Consolidation state to synchronize**:
```rust
struct ConsolidationState {
    patterns: HashMap<PatternId, SemanticPattern>,
    last_consolidation: Timestamp,
    consolidation_generation: u64,
}

struct SemanticPattern {
    id: PatternId,
    embedding: [f32; 768],
    confidence: f32,
    citation_count: u32,
    citations: Vec<EpisodeId>,
    created_at: Timestamp,
    updated_at: Timestamp,
    version: VectorClock,  // For conflict detection
}
```

**State divergence causes**:
1. **Independent consolidation**: Nodes process different episodes locally
2. **Network partitions**: Gossip messages fail to propagate
3. **Timing differences**: Consolidation scheduler runs at different times
4. **Replica lag**: Primary processes consolidation before replicas

**Convergence guarantee**:
If network heals and gossip continues, all nodes converge to same state within bounded time. Proof sketch:
- Gossip is infection-style: once any node has update, it spreads exponentially
- After k rounds, fraction of nodes with update ≥ 1 - (1 - 1/N)^k
- For N=100, k=10: >99.9% probability all nodes have update

### Performance Characteristics

**Bandwidth analysis**:
- Merkle root exchange: 32 bytes per gossip round
- Full state transfer: ~50KB per semantic pattern (embedding + metadata)
- Delta sync (typical): 1-5 patterns changed = 50-250KB
- Gossip interval: 60 seconds
- Per-node bandwidth: <5KB/s average (250KB/60s)

**Comparison with alternatives**:
- **Full state transfer**: 10K patterns × 50KB = 500MB per sync (unacceptable)
- **Vector clocks only**: Can detect divergence but not identify specific changes
- **Merkle trees**: O(log N) comparison + delta transfer (optimal)

**Convergence time**:
- Single pattern update: Reaches all N nodes in O(log N) gossip rounds
- For N=100, interval=60s: ~7 rounds = 7 minutes worst case
- Target: <10 rounds = <10 minutes for 99.9% convergence

## Technical Specification

### Merkle Tree Design

**Tree structure for consolidation state**:
```
Depth 12 binary tree covering 4096 pattern slots
Each pattern hashed into deterministic leaf position
Internal nodes = SHA-256(left || right)
Root = 256-bit state fingerprint
```

**Partitioning strategy**:
Pattern ID → consistent hash → leaf index (0..4095)
Ensures deterministic placement across all nodes
Patterns sorted by hash within each leaf (for stable ordering)

**Update protocol**:
On consolidation run:
1. Insert/update patterns in local state
2. Recompute affected Merkle path (O(log N))
3. Update root hash
4. Increment consolidation generation counter

### Core Data Structures

```rust
// engram-core/src/cluster/gossip/merkle_tree.rs

use sha2::{Sha256, Digest};
use std::collections::HashMap;
use serde::{Serialize, Deserialize};

/// Merkle tree over consolidation state for efficient comparison
pub struct ConsolidationMerkleTree {
    /// Tree depth (log₂ of leaf count)
    depth: usize,

    /// Internal nodes: level -> node_index -> hash
    /// Level 0 = root, level `depth` = leaves
    nodes: Vec<HashMap<usize, MerkleHash>>,

    /// Leaf data: pattern_id -> SemanticPattern
    patterns: HashMap<String, SemanticPattern>,

    /// Mapping: leaf_index -> pattern_ids in that leaf
    leaf_contents: HashMap<usize, Vec<String>>,
}

/// 256-bit Merkle hash (SHA-256 output)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct MerkleHash([u8; 32]);

impl MerkleHash {
    /// Compute hash of a semantic pattern
    fn hash_pattern(pattern: &SemanticPattern) -> Self {
        let mut hasher = Sha256::new();

        // Hash pattern metadata (not embedding for efficiency)
        hasher.update(pattern.id.as_bytes());
        hasher.update(&pattern.confidence.to_le_bytes());
        hasher.update(&pattern.citation_count.to_le_bytes());
        hasher.update(&pattern.updated_at.timestamp().to_le_bytes());
        hasher.update(&pattern.version.serialize());

        Self(hasher.finalize().into())
    }

    /// Combine two child hashes into parent hash
    fn combine(left: &MerkleHash, right: &MerkleHash) -> Self {
        let mut hasher = Sha256::new();
        hasher.update(&left.0);
        hasher.update(&right.0);
        Self(hasher.finalize().into())
    }
}

impl ConsolidationMerkleTree {
    /// Create empty Merkle tree with specified depth
    pub fn new(depth: usize) -> Self {
        Self {
            depth,
            nodes: vec![HashMap::new(); depth + 1],
            patterns: HashMap::new(),
            leaf_contents: HashMap::new(),
        }
    }

    /// Insert or update a semantic pattern
    pub fn upsert_pattern(&mut self, pattern: SemanticPattern) {
        let pattern_id = pattern.id.clone();
        let leaf_index = self.pattern_to_leaf(&pattern_id);

        // Update pattern data
        self.patterns.insert(pattern_id.clone(), pattern);

        // Update leaf contents
        self.leaf_contents
            .entry(leaf_index)
            .or_insert_with(Vec::new)
            .push(pattern_id.clone());

        // Recompute hashes from leaf to root
        self.recompute_path(leaf_index);
    }

    /// Get root hash (state fingerprint)
    pub fn root_hash(&self) -> MerkleHash {
        self.nodes[0].get(&0).copied().unwrap_or(MerkleHash([0; 32]))
    }

    /// Map pattern ID to deterministic leaf index
    fn pattern_to_leaf(&self, pattern_id: &str) -> usize {
        use std::hash::{Hash, Hasher};
        use std::collections::hash_map::DefaultHasher;

        let mut hasher = DefaultHasher::new();
        pattern_id.hash(&mut hasher);
        let hash = hasher.finish();

        let leaf_count = 1 << self.depth;
        (hash as usize) % leaf_count
    }

    /// Recompute Merkle path from leaf to root
    fn recompute_path(&mut self, leaf_index: usize) {
        let mut index = leaf_index;

        // Start at leaf level
        for level in (0..=self.depth).rev() {
            let hash = if level == self.depth {
                // Leaf level: hash all patterns in this leaf
                self.compute_leaf_hash(leaf_index)
            } else {
                // Internal level: combine child hashes
                let left_child = index * 2;
                let right_child = index * 2 + 1;

                let left_hash = self.nodes[level + 1]
                    .get(&left_child)
                    .copied()
                    .unwrap_or(MerkleHash([0; 32]));
                let right_hash = self.nodes[level + 1]
                    .get(&right_child)
                    .copied()
                    .unwrap_or(MerkleHash([0; 32]));

                MerkleHash::combine(&left_hash, &right_hash)
            };

            self.nodes[level].insert(index, hash);

            // Move to parent
            if level > 0 {
                index /= 2;
            }
        }
    }

    /// Compute hash of all patterns in a leaf
    fn compute_leaf_hash(&self, leaf_index: usize) -> MerkleHash {
        let pattern_ids = match self.leaf_contents.get(&leaf_index) {
            Some(ids) if !ids.is_empty() => ids,
            _ => return MerkleHash([0; 32]),
        };

        // Sort pattern IDs for stable ordering
        let mut sorted_ids = pattern_ids.clone();
        sorted_ids.sort();

        // Hash concatenation of all pattern hashes in leaf
        let mut hasher = Sha256::new();
        for id in sorted_ids {
            if let Some(pattern) = self.patterns.get(&id) {
                let pattern_hash = MerkleHash::hash_pattern(pattern);
                hasher.update(&pattern_hash.0);
            }
        }

        MerkleHash(hasher.finalize().into())
    }

    /// Find divergent subtrees between two Merkle trees
    pub fn find_divergence(
        &self,
        other: &ConsolidationMerkleTree,
    ) -> Vec<DivergentRange> {
        let mut divergent = Vec::new();
        self.find_divergence_recursive(other, 0, 0, &mut divergent);
        divergent
    }

    /// Recursive traversal to find divergent subtrees
    fn find_divergence_recursive(
        &self,
        other: &ConsolidationMerkleTree,
        level: usize,
        index: usize,
        divergent: &mut Vec<DivergentRange>,
    ) {
        // Get hashes at this level
        let self_hash = self.nodes[level].get(&index);
        let other_hash = other.nodes[level].get(&index);

        // If hashes match, subtrees are identical
        if self_hash == other_hash {
            return;
        }

        // If at leaf level, record divergence
        if level == self.depth {
            divergent.push(DivergentRange {
                leaf_index: index,
                self_hash: self_hash.copied(),
                other_hash: other_hash.copied(),
            });
            return;
        }

        // Recurse into children
        self.find_divergence_recursive(other, level + 1, index * 2, divergent);
        self.find_divergence_recursive(other, level + 1, index * 2 + 1, divergent);
    }
}

/// Represents a divergent range in Merkle tree comparison
#[derive(Debug, Clone)]
pub struct DivergentRange {
    pub leaf_index: usize,
    pub self_hash: Option<MerkleHash>,
    pub other_hash: Option<MerkleHash>,
}
```

### Gossip Protocol Messages

```rust
// engram-core/src/cluster/gossip/messages.rs

use crate::completion::SemanticPattern;
use crate::cluster::gossip::merkle_tree::MerkleHash;
use serde::{Serialize, Deserialize};

/// Gossip messages for consolidation state synchronization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConsolidationGossip {
    /// Exchange Merkle roots to check for divergence
    MerkleRootExchange {
        from_node: String,
        root_hash: MerkleHash,
        generation: u64,
        pattern_count: usize,
    },

    /// Request specific subtree hashes
    SubtreeRequest {
        from_node: String,
        level: usize,
        node_indices: Vec<usize>,
    },

    /// Response with subtree hashes
    SubtreeResponse {
        from_node: String,
        level: usize,
        hashes: Vec<(usize, MerkleHash)>,
    },

    /// Request full patterns for divergent leaves
    PatternRequest {
        from_node: String,
        leaf_indices: Vec<usize>,
    },

    /// Send patterns for synchronization
    PatternSync {
        from_node: String,
        patterns: Vec<SemanticPattern>,
    },
}
```

### Gossip Exchange Protocol

```rust
// engram-core/src/cluster/gossip/consolidation.rs

use super::merkle_tree::{ConsolidationMerkleTree, MerkleHash};
use super::messages::ConsolidationGossip;
use crate::cluster::membership::NodeInfo;
use crate::completion::SemanticPattern;
use std::sync::Arc;
use tokio::sync::RwLock;
use dashmap::DashMap;

pub struct ConsolidationGossiper {
    /// Local Merkle tree over consolidation state
    merkle_tree: Arc<RwLock<ConsolidationMerkleTree>>,

    /// Current consolidation generation (increments on each run)
    generation: Arc<RwLock<u64>>,

    /// Gossip configuration
    config: GossipConfig,

    /// Conflict resolver (from Task 008)
    conflict_resolver: Arc<dyn ConflictResolver>,

    /// Patterns pending merge
    pending_merges: DashMap<String, Vec<SemanticPattern>>,
}

#[derive(Debug, Clone)]
pub struct GossipConfig {
    /// Interval between gossip rounds (default: 60s)
    pub interval: Duration,

    /// Number of peers to gossip with per round (default: 3)
    pub fanout: usize,

    /// Maximum patterns to transfer in one sync (default: 100)
    pub max_patterns_per_sync: usize,

    /// Enable compression for large pattern transfers
    pub enable_compression: bool,
}

impl Default for GossipConfig {
    fn default() -> Self {
        Self {
            interval: Duration::from_secs(60),
            fanout: 3,
            max_patterns_per_sync: 100,
            enable_compression: true,
        }
    }
}

impl ConsolidationGossiper {
    /// Run one gossip round with random peers
    pub async fn gossip_round(
        &self,
        membership: &SwimMembership,
    ) -> Result<GossipStats, GossipError> {
        let mut stats = GossipStats::default();

        // Select random peers
        let peers = membership.select_random_nodes(self.config.fanout)?;

        for peer in peers {
            let round_stats = self.gossip_with_peer(&peer).await?;
            stats.merge(round_stats);
        }

        Ok(stats)
    }

    /// Execute gossip exchange with a single peer
    async fn gossip_with_peer(
        &self,
        peer: &NodeInfo,
    ) -> Result<GossipStats, GossipError> {
        let mut stats = GossipStats::default();

        // 1. Exchange Merkle roots
        let local_root = {
            let tree = self.merkle_tree.read().await;
            tree.root_hash()
        };
        let local_gen = *self.generation.read().await;

        let exchange_msg = ConsolidationGossip::MerkleRootExchange {
            from_node: membership.local_node_id().to_string(),
            root_hash: local_root,
            generation: local_gen,
            pattern_count: self.merkle_tree.read().await.patterns.len(),
        };

        let peer_root = self.send_and_receive_root(peer, exchange_msg).await?;
        stats.roots_exchanged += 1;

        // 2. If roots match, we're in sync
        if local_root == peer_root.root_hash {
            stats.peers_in_sync += 1;
            return Ok(stats);
        }

        // 3. Roots differ, find divergent subtrees
        let peer_tree = self.fetch_peer_tree_metadata(peer).await?;
        let divergent_ranges = {
            let tree = self.merkle_tree.read().await;
            tree.find_divergence(&peer_tree)
        };

        stats.divergent_ranges += divergent_ranges.len();

        // 4. Request patterns from divergent leaves
        let leaf_indices: Vec<usize> = divergent_ranges
            .iter()
            .map(|r| r.leaf_index)
            .collect();

        let peer_patterns = self.request_patterns(peer, leaf_indices).await?;
        stats.patterns_received += peer_patterns.len();

        // 5. Merge received patterns (with conflict resolution)
        for pattern in peer_patterns {
            self.merge_pattern(pattern).await?;
            stats.patterns_merged += 1;
        }

        Ok(stats)
    }

    /// Send our patterns that peer is missing
    async fn send_missing_patterns(
        &self,
        peer: &NodeInfo,
        divergent_ranges: &[DivergentRange],
    ) -> Result<(), GossipError> {
        let tree = self.merkle_tree.read().await;

        let mut patterns_to_send = Vec::new();
        for range in divergent_ranges {
            if let Some(pattern_ids) = tree.leaf_contents.get(&range.leaf_index) {
                for pattern_id in pattern_ids {
                    if let Some(pattern) = tree.patterns.get(pattern_id) {
                        patterns_to_send.push(pattern.clone());
                    }
                }
            }
        }

        // Send patterns in batches
        for batch in patterns_to_send.chunks(self.config.max_patterns_per_sync) {
            let sync_msg = ConsolidationGossip::PatternSync {
                from_node: membership.local_node_id().to_string(),
                patterns: batch.to_vec(),
            };

            self.send_message(peer, sync_msg).await?;
        }

        Ok(())
    }

    /// Merge received pattern with conflict resolution
    async fn merge_pattern(
        &self,
        remote_pattern: SemanticPattern,
    ) -> Result<(), GossipError> {
        let mut tree = self.merkle_tree.write().await;

        // Check if we have a local version
        if let Some(local_pattern) = tree.patterns.get(&remote_pattern.id) {
            // Conflict: both nodes have different versions
            let merged = self.conflict_resolver
                .resolve_pattern_conflict(local_pattern, &remote_pattern)
                .await?;

            tree.upsert_pattern(merged);
        } else {
            // No conflict: accept remote pattern
            tree.upsert_pattern(remote_pattern);
        }

        Ok(())
    }
}

#[derive(Debug, Default)]
pub struct GossipStats {
    pub roots_exchanged: usize,
    pub peers_in_sync: usize,
    pub divergent_ranges: usize,
    pub patterns_received: usize,
    pub patterns_merged: usize,
}
```

### Integration with SWIM

Piggyback consolidation gossip on SWIM's existing gossip transport:

```rust
// engram-core/src/cluster/gossip/mod.rs

impl SwimMembership {
    /// Attach consolidation gossip to SWIM messages
    pub fn attach_consolidation_gossip(&self, msg: &mut SwimMessage) {
        if let Some(gossiper) = &self.consolidation_gossiper {
            // Piggyback Merkle root on ping messages
            if let SwimMessage::Ping { .. } = msg {
                let root = gossiper.current_root();
                msg.add_metadata("consolidation_root", root);
            }
        }
    }

    /// Handle received consolidation gossip
    pub async fn handle_consolidation_gossip(
        &self,
        from: &NodeInfo,
        metadata: &HashMap<String, Vec<u8>>,
    ) {
        if let Some(root_bytes) = metadata.get("consolidation_root") {
            if let Some(gossiper) = &self.consolidation_gossiper {
                let root: MerkleHash = bincode::deserialize(root_bytes).ok()?;

                // Trigger async sync if roots differ
                if root != gossiper.current_root() {
                    tokio::spawn({
                        let gossiper = gossiper.clone();
                        let from = from.clone();
                        async move {
                            gossiper.gossip_with_peer(&from).await;
                        }
                    });
                }
            }
        }
    }
}
```

## Core Operations

### 1. Build Merkle Tree from Consolidation Snapshot

```rust
impl ConsolidationGossiper {
    /// Rebuild Merkle tree from consolidation snapshot
    pub async fn rebuild_from_snapshot(
        &self,
        snapshot: &ConsolidationSnapshot,
    ) -> Result<(), GossipError> {
        let mut tree = ConsolidationMerkleTree::new(12); // Depth 12 = 4096 leaves

        // Insert all patterns
        for pattern in &snapshot.patterns {
            tree.upsert_pattern(pattern.clone());
        }

        // Update local tree
        *self.merkle_tree.write().await = tree;

        // Increment generation
        *self.generation.write().await += 1;

        Ok(())
    }
}
```

### 2. Incremental Update on Pattern Change

```rust
impl ConsolidationGossiper {
    /// Update Merkle tree when consolidation adds/updates pattern
    pub async fn update_pattern(
        &self,
        pattern: SemanticPattern,
    ) -> Result<(), GossipError> {
        let mut tree = self.merkle_tree.write().await;

        // Upsert triggers incremental path recomputation
        tree.upsert_pattern(pattern);

        Ok(())
    }
}
```

### 3. Delta Sync Between Peers

```rust
impl ConsolidationGossiper {
    /// Perform efficient delta sync with peer
    async fn delta_sync(
        &self,
        peer: &NodeInfo,
        divergent_ranges: Vec<DivergentRange>,
    ) -> Result<DeltaSyncResult, GossipError> {
        let mut result = DeltaSyncResult::default();

        // Bi-directional sync: send ours, receive theirs
        let (send_task, recv_task) = tokio::join!(
            self.send_missing_patterns(peer, &divergent_ranges),
            self.request_patterns(peer, divergent_ranges.iter().map(|r| r.leaf_index).collect()),
        );

        send_task?;
        let received_patterns = recv_task?;

        // Merge received patterns
        for pattern in received_patterns {
            self.merge_pattern(pattern).await?;
            result.merged_count += 1;
        }

        Ok(result)
    }
}

#[derive(Debug, Default)]
struct DeltaSyncResult {
    merged_count: usize,
}
```

### 4. Conflict Resolution Integration

```rust
// engram-core/src/cluster/conflict/mod.rs

use crate::completion::SemanticPattern;
use async_trait::async_trait;

#[async_trait]
pub trait ConflictResolver: Send + Sync {
    /// Resolve conflict between local and remote pattern versions
    async fn resolve_pattern_conflict(
        &self,
        local: &SemanticPattern,
        remote: &SemanticPattern,
    ) -> Result<SemanticPattern, ConflictError>;
}

pub struct ConfidenceVotingResolver;

#[async_trait]
impl ConflictResolver for ConfidenceVotingResolver {
    async fn resolve_pattern_conflict(
        &self,
        local: &SemanticPattern,
        remote: &SemanticPattern,
    ) -> Result<SemanticPattern, ConflictError> {
        // Compare vector clocks for causality
        match local.version.compare(&remote.version) {
            Ordering::Before => {
                // Remote is newer, accept it
                Ok(remote.clone())
            }
            Ordering::After => {
                // Local is newer, keep it
                Ok(local.clone())
            }
            Ordering::Concurrent => {
                // Concurrent updates, use confidence voting
                if remote.confidence > local.confidence {
                    Ok(remote.clone())
                } else if local.confidence > remote.confidence {
                    Ok(local.clone())
                } else {
                    // Equal confidence, merge citations
                    self.merge_patterns(local, remote)
                }
            }
        }
    }

    fn merge_patterns(
        &self,
        local: &SemanticPattern,
        remote: &SemanticPattern,
    ) -> Result<SemanticPattern, ConflictError> {
        // Merge citations from both versions
        let mut citations = local.citations.clone();
        citations.extend(remote.citations.clone());
        citations.sort();
        citations.dedup();

        // Average embeddings with confidence weighting
        let total_conf = local.confidence + remote.confidence;
        let w_local = local.confidence / total_conf;
        let w_remote = remote.confidence / total_conf;

        let merged_embedding: [f32; 768] = std::array::from_fn(|i| {
            w_local * local.embedding[i] + w_remote * remote.embedding[i]
        });

        // Take maximum confidence (conservative)
        let merged_confidence = local.confidence.max(remote.confidence);

        Ok(SemanticPattern {
            id: local.id.clone(),
            embedding: merged_embedding,
            confidence: merged_confidence,
            citation_count: citations.len() as u32,
            citations,
            created_at: local.created_at.min(remote.created_at),
            updated_at: Utc::now(),
            version: local.version.merge(&remote.version),
        })
    }
}
```

## Files to Create

1. `engram-core/src/cluster/gossip/mod.rs` - Gossip module
2. `engram-core/src/cluster/gossip/merkle_tree.rs` - Merkle tree implementation
3. `engram-core/src/cluster/gossip/consolidation.rs` - Consolidation gossiper
4. `engram-core/src/cluster/gossip/messages.rs` - Gossip message types
5. `engram-core/src/cluster/conflict/mod.rs` - Conflict resolution trait
6. `engram-core/src/cluster/conflict/strategies.rs` - Resolution strategies
7. `engram-core/src/cluster/conflict/vector_clock.rs` - Vector clock for causality

## Files to Modify

1. `engram-core/src/cluster/membership.rs` - Integrate consolidation gossip with SWIM
2. `engram-core/src/completion/consolidation.rs` - Trigger Merkle tree updates
3. `engram-core/src/consolidation/service.rs` - Hook gossip on consolidation
4. `engram-cli/config/cluster.toml` - Add gossip configuration

## Testing Strategy

### Unit Tests

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_merkle_tree_incremental_update() {
        let mut tree = ConsolidationMerkleTree::new(8);

        // Insert pattern
        let pattern = test_pattern("pattern1");
        tree.upsert_pattern(pattern.clone());
        let root1 = tree.root_hash();

        // Update pattern (change confidence)
        let mut updated = pattern.clone();
        updated.confidence = 0.95;
        tree.upsert_pattern(updated);
        let root2 = tree.root_hash();

        // Root should change
        assert_ne!(root1, root2);
    }

    #[test]
    fn test_merkle_divergence_detection() {
        let mut tree1 = ConsolidationMerkleTree::new(8);
        let mut tree2 = ConsolidationMerkleTree::new(8);

        // Both have pattern A
        tree1.upsert_pattern(test_pattern("A"));
        tree2.upsert_pattern(test_pattern("A"));

        // Tree1 has pattern B
        tree1.upsert_pattern(test_pattern("B"));

        // Tree2 has pattern C
        tree2.upsert_pattern(test_pattern("C"));

        // Should detect divergence
        let divergent = tree1.find_divergence(&tree2);
        assert!(!divergent.is_empty());
    }

    #[tokio::test]
    async fn test_gossip_convergence() {
        let node1 = ConsolidationGossiper::new_test();
        let node2 = ConsolidationGossiper::new_test();

        // Node1 has pattern A
        node1.update_pattern(test_pattern("A")).await.unwrap();

        // Node2 has pattern B
        node2.update_pattern(test_pattern("B")).await.unwrap();

        // Gossip exchange
        node1.gossip_with_peer(&node2_info).await.unwrap();

        // Both should now have both patterns
        assert!(node1.has_pattern("A").await);
        assert!(node1.has_pattern("B").await);
        assert!(node2.has_pattern("A").await);
        assert!(node2.has_pattern("B").await);

        // Merkle roots should match
        assert_eq!(node1.root_hash().await, node2.root_hash().await);
    }
}
```

### Property-Based Tests

```rust
use proptest::prelude::*;

proptest! {
    #[test]
    fn test_merkle_tree_determinism(
        patterns in prop::collection::vec(arbitrary_pattern(), 1..100)
    ) {
        // Build tree in two different orders
        let mut tree1 = ConsolidationMerkleTree::new(10);
        let mut tree2 = ConsolidationMerkleTree::new(10);

        for pattern in &patterns {
            tree1.upsert_pattern(pattern.clone());
        }

        for pattern in patterns.iter().rev() {
            tree2.upsert_pattern(pattern.clone());
        }

        // Roots should be identical regardless of insertion order
        prop_assert_eq!(tree1.root_hash(), tree2.root_hash());
    }

    #[test]
    fn test_gossip_convergence_property(
        num_nodes in 3..10usize,
        patterns_per_node in 5..20usize,
    ) {
        let runtime = tokio::runtime::Runtime::new().unwrap();

        // Create N nodes with random patterns
        let nodes: Vec<_> = (0..num_nodes)
            .map(|_| {
                let gossiper = ConsolidationGossiper::new_test();
                for _ in 0..patterns_per_node {
                    runtime.block_on(gossiper.update_pattern(random_pattern()));
                }
                gossiper
            })
            .collect();

        // Run gossip for 10 rounds
        for _ in 0..10 {
            for i in 0..num_nodes {
                let peer_idx = (i + 1) % num_nodes;
                runtime.block_on(nodes[i].gossip_with_peer(&nodes[peer_idx].info()));
            }
        }

        // All nodes should have same root hash
        let baseline_root = runtime.block_on(nodes[0].root_hash());
        for node in &nodes[1..] {
            let root = runtime.block_on(node.root_hash());
            prop_assert_eq!(root, baseline_root);
        }
    }
}
```

### Integration Tests

```rust
// engram-core/tests/cluster_gossip_integration.rs

#[tokio::test]
async fn test_three_node_consolidation_gossip() {
    // Start three nodes
    let node1 = start_test_node_with_consolidation(7946).await;
    let node2 = start_test_node_with_consolidation(7947).await;
    let node3 = start_test_node_with_consolidation(7948).await;

    // Each node consolidates independently
    node1.consolidate_episodes(vec!["e1", "e2"]).await;
    node2.consolidate_episodes(vec!["e3", "e4"]).await;
    node3.consolidate_episodes(vec!["e5", "e6"]).await;

    // Merkle roots should differ
    let root1 = node1.merkle_root().await;
    let root2 = node2.merkle_root().await;
    let root3 = node3.merkle_root().await;
    assert_ne!(root1, root2);
    assert_ne!(root2, root3);

    // Run gossip for 10 rounds (60s each in real deployment)
    for _ in 0..10 {
        node1.gossip_round().await.unwrap();
        node2.gossip_round().await.unwrap();
        node3.gossip_round().await.unwrap();
        tokio::time::sleep(Duration::from_millis(100)).await;
    }

    // All nodes should converge to same state
    let final_root1 = node1.merkle_root().await;
    let final_root2 = node2.merkle_root().await;
    let final_root3 = node3.merkle_root().await;
    assert_eq!(final_root1, final_root2);
    assert_eq!(final_root2, final_root3);

    // Each node should have all 6 patterns (or merged versions)
    assert_eq!(node1.pattern_count().await, node2.pattern_count().await);
    assert_eq!(node2.pattern_count().await, node3.pattern_count().await);
}

#[tokio::test]
async fn test_partition_healing_convergence() {
    // Start 5-node cluster
    let nodes = start_test_cluster(5).await;

    // All nodes consolidate shared episodes
    for node in &nodes {
        node.consolidate_episodes(vec!["shared1", "shared2"]).await;
    }

    // Gossip to convergence
    for _ in 0..10 {
        for node in &nodes {
            node.gossip_round().await.unwrap();
        }
    }

    let baseline_root = nodes[0].merkle_root().await;

    // Partition cluster (nodes 0-2 vs 3-4)
    let partition = NetworkPartition::new(&[0, 1, 2], &[3, 4]);
    partition.activate();

    // Each side consolidates independently during partition
    nodes[0].consolidate_episodes(vec!["left1"]).await;
    nodes[3].consolidate_episodes(vec!["right1"]).await;

    // Gossip within each partition
    for _ in 0..5 {
        for i in 0..=2 {
            nodes[i].gossip_round().await.unwrap();
        }
        for i in 3..=4 {
            nodes[i].gossip_round().await.unwrap();
        }
    }

    // Sides should have diverged
    let left_root = nodes[0].merkle_root().await;
    let right_root = nodes[3].merkle_root().await;
    assert_ne!(left_root, right_root);
    assert_ne!(left_root, baseline_root);
    assert_ne!(right_root, baseline_root);

    // Heal partition
    partition.deactivate();

    // Gossip to convergence
    for _ in 0..10 {
        for node in &nodes {
            node.gossip_round().await.unwrap();
        }
    }

    // All nodes should converge (with merged patterns)
    let healed_root = nodes[0].merkle_root().await;
    for node in &nodes[1..] {
        assert_eq!(node.merkle_root().await, healed_root);
    }
}
```

### Convergence Validation

```rust
#[tokio::test]
async fn test_gossip_convergence_time() {
    let num_nodes = 100;
    let nodes = start_test_cluster(num_nodes).await;

    // Node 0 gets a new pattern
    nodes[0].update_pattern(test_pattern("new_pattern")).await;

    // Track how many rounds until all nodes have it
    let mut rounds = 0;
    loop {
        // Run one gossip round
        for node in &nodes {
            node.gossip_round().await.unwrap();
        }
        rounds += 1;

        // Check if all nodes converged
        let baseline_root = nodes[0].merkle_root().await;
        let all_converged = nodes[1..]
            .iter()
            .all(|node| {
                tokio_test::block_on(node.merkle_root()) == baseline_root
            });

        if all_converged {
            break;
        }

        assert!(rounds < 20, "Failed to converge within 20 rounds");
    }

    // Should converge in O(log N) rounds
    // For N=100, log₂(100) ≈ 6.64, expect <10 rounds
    assert!(rounds < 10, "Convergence took {} rounds, expected <10", rounds);
}
```

## Dependencies

Add to `engram-core/Cargo.toml`:

```toml
[dependencies]
# Already have: tokio, dashmap, serde, bincode

# Cryptography for Merkle trees
sha2 = "0.10"

# Async trait support
async-trait = "0.1"

# Compression for large pattern transfers
lz4 = "1.24"
```

## Acceptance Criteria

1. **Convergence within 10 rounds**: 3-node cluster converges in <10 gossip rounds
2. **Bandwidth efficiency**: Delta sync transfers <10% of full state on average
3. **Deterministic conflict resolution**: Same inputs always produce same merged pattern
4. **No lost patterns**: Partition healing merges all patterns from both sides
5. **Incremental Merkle updates**: Pattern update recomputes O(log N) hashes, not full tree
6. **Compression works**: Large pattern transfers use <50% bandwidth with compression
7. **Vector clock ordering**: Concurrent updates detected and resolved deterministically
8. **Metrics exposed**: Gossip rounds, patterns merged, bandwidth used

## Performance Targets

- **Merkle tree update**: <1ms for single pattern insertion (depth 12)
- **Root comparison**: <10μs (single hash comparison)
- **Divergence detection**: <10ms for trees with 10K patterns
- **Delta sync bandwidth**: <5KB/s average per node (gossip interval 60s)
- **Convergence time**: <10 minutes for 100-node cluster (10 rounds × 60s)
- **Memory overhead**: <50MB per node for Merkle tree (10K patterns)

## Convergence Proof Strategy

**Theorem**: If network heals and gossip continues, all nodes converge to same Merkle root with probability approaching 1.

**Proof sketch**:
1. Gossip is symmetric: if A sends update to B, B can send to A
2. Each round, node gossips with K random peers
3. After one round, expected fraction with update = 1/N + K/N
4. After k rounds, probability node lacks update < (1 - K/N)^k
5. For K=3, N=100, k=10: P(divergent) < 0.1%

**Validation via property testing**:
- Generate random patterns across nodes
- Run gossip for bounded rounds
- Assert all nodes have same root hash

## Integration with Task 008 (Conflict Resolution)

Gossip protocol calls conflict resolver when merging patterns:

```rust
// From gossip_with_peer:
for pattern in peer_patterns {
    if let Some(local) = tree.patterns.get(&pattern.id) {
        // Conflict detected, resolve it
        let merged = conflict_resolver.resolve(local, &pattern).await?;
        tree.upsert_pattern(merged);
    } else {
        // No conflict, accept remote
        tree.upsert_pattern(pattern);
    }
}
```

Conflict resolver uses:
- Vector clocks for causality (from Task 008)
- Confidence voting (higher confidence wins)
- Citation merging (union of episode references)
- Embedding averaging (weighted by confidence)

## Next Steps

After completing this task:
- Task 008 will implement detailed conflict resolution strategies
- Task 009 will use gossip-synchronized state for distributed queries
- Task 011 will validate gossip convergence via Jepsen testing
- Consolidation state becomes eventually consistent across cluster

## References

1. Demers, A., et al. (1987). "Epidemic Algorithms for Replicated Database Maintenance"
2. DeCandia, G., et al. (2007). "Dynamo: Amazon's Highly Available Key-value Store"
3. Merkle, R. (1987). "A Digital Signature Based on a Conventional Encryption Function"
4. Cassandra Architecture: Anti-Entropy Repair (https://cassandra.apache.org)
5. Riak Active Anti-Entropy (https://docs.riak.com/riak/kv/latest/learn/concepts/aae/)
