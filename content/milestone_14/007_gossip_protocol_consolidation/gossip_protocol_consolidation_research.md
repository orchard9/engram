# Research: Gossip Protocols for Consolidation Synchronization

## The Consolidation Synchronization Problem

Engram's memory consolidation transforms episodic memories into semantic patterns over hours. In a distributed system, consolidation runs independently on each node. Node A might consolidate episodes {E1, E2, E3} into semantic pattern S1. Node B might consolidate {E2, E3, E4} into pattern S2. These patterns need to sync across the cluster.

Traditional approaches: use a consensus system (Raft/Paxos) to agree on consolidation results. But this sacrifices availability during partitions and adds latency to consolidation.

## Gossip-Based Anti-Entropy

Anti-entropy protocols, introduced by Demers et al. (1987), synchronize state through periodic peer exchange. The algorithm:

1. Select random peer every gossip interval (e.g., 60 seconds)
2. Exchange state digests (Merkle tree roots)
3. If digests differ, identify divergent regions
4. Transfer only the differences
5. Merge with conflict resolution

This provides eventual consistency: all nodes converge to the same state within O(log N) gossip rounds.

## Merkle Trees for Efficient State Comparison

Comparing entire consolidation state would be expensive (gigabytes of semantic patterns). Merkle trees enable efficient comparison:

```rust
struct MerkleTree {
    root: Hash,
    levels: Vec<Vec<Hash>>,
}

impl MerkleTree {
    fn from_consolidation_state(state: &ConsolidationState) -> Self {
        let leaves: Vec<Hash> = state.patterns
            .iter()
            .map(|pattern| hash(pattern))
            .collect();

        let mut levels = vec![leaves];

        while levels.last().unwrap().len() > 1 {
            let prev_level = levels.last().unwrap();
            let next_level = prev_level
                .chunks(2)
                .map(|pair| {
                    if pair.len() == 2 {
                        hash(&(pair[0], pair[1]))
                    } else {
                        pair[0]
                    }
                })
                .collect();
            levels.push(next_level);
        }

        MerkleTree {
            root: levels.last().unwrap()[0],
            levels,
        }
    }

    fn find_diff(&self, other: &MerkleTree) -> Vec<usize> {
        // Recursively compare levels to find divergent leaf indices
        // Returns indices of patterns that differ
    }
}
```

Gossip protocol:
1. Send Merkle root hash (32 bytes)
2. If roots match, state is identical - done
3. If roots differ, recursively compare subtrees
4. Identify leaf indices that differ
5. Transfer only those patterns

For 10,000 patterns with 1% divergence, this transfers ~100 patterns instead of 10,000. 100x bandwidth savings.

## Conflict Resolution for Divergent Patterns

When nodes consolidate independently, they might create conflicting patterns. Conflict types:

**Type 1: Same episodes, different patterns**
Node A consolidates {E1, E2} into "morning routine" with confidence 0.9
Node B consolidates {E1, E2} into "coffee habits" with confidence 0.7
Resolution: Keep both patterns, merge evidence, use higher confidence

**Type 2: Overlapping episodes**
Node A: {E1, E2, E3} -> Pattern P1
Node B: {E2, E3, E4} -> Pattern P2
Resolution: Identify overlap (E2, E3), merge patterns if semantically similar

**Type 3: Concurrent creation of same pattern**
Both nodes create "morning routine" pattern independently
Resolution: Merge via vector clock, combine supporting episodes

Vector clocks track causality:

```rust
struct ConsolidationPattern {
    pattern_id: PatternId,
    episodes: Vec<EpisodeId>,
    confidence: f32,
    vector_clock: VectorClock,
}

fn resolve_conflict(p1: Pattern, p2: Pattern) -> Pattern {
    match p1.vector_clock.compare(&p2.vector_clock) {
        Ordering::Greater => p1,  // P1 happened after P2
        Ordering::Less => p2,     // P2 happened after P1
        Ordering::Concurrent => {
            // Merge: combine episodes, average confidence
            Pattern {
                episodes: p1.episodes.union(p2.episodes),
                confidence: (p1.confidence + p2.confidence) / 2.0,
                vector_clock: p1.vector_clock.merge(p2.vector_clock),
            }
        }
    }
}
```

## Gossip Scheduling

Random peer selection ensures uniform propagation. However, we can optimize:

**Prefer high-lag peers**: If Node A knows Node B is far behind, prioritize gossiping to B
**Cluster-aware selection**: Prefer peers in different racks/zones to spread information faster
**Adaptive intervals**: Increase gossip frequency during high consolidation activity

Implementation:

```rust
struct GossipScheduler {
    peers: Vec<NodeId>,
    last_gossip: HashMap<NodeId, Instant>,
    lag_estimates: HashMap<NodeId, Duration>,
}

impl GossipScheduler {
    fn select_peer(&self) -> NodeId {
        // 80% of time: select random peer
        // 20% of time: select high-lag peer
        if rand::random::<f32>() < 0.8 {
            self.peers.choose(&mut rand::thread_rng()).unwrap()
        } else {
            self.lag_estimates
                .iter()
                .max_by_key(|(_, lag)| lag)
                .map(|(node, _)| node)
                .unwrap_or_else(|| self.peers.choose(&mut rand::thread_rng()).unwrap())
        }
    }
}
```

## Bandwidth Optimization

Gossip can generate significant traffic. Optimizations:

1. **Compression**: Consolidation patterns compress well (repetitive structure). Use zstd for 3-5x reduction.
2. **Delta encoding**: Send only new patterns since last gossip, not entire state.
3. **Bloom filters**: Send Bloom filter of patterns before Merkle tree to quickly identify large differences.

## Academic Foundation

- **Epidemic Algorithms**: Demers et al. (1987) - foundational anti-entropy paper
- **Scuttlebutt**: van Renesse et al. (1998) - efficient gossip for databases
- **Dynamo**: DeCandia et al. (2007) - Merkle trees for state sync
- **Vector Clocks**: Fidge (1988), Mattern (1988) - causality tracking

## Convergence Guarantees

Theorem: Under the gossip protocol with random peer selection, all nodes converge to the same state within O(log N) rounds with high probability, assuming no new consolidations occur.

Proof sketch: Each gossip round halves the number of nodes that don't know about an update (epidemic spreading). After log2(N) rounds, all nodes have seen the update with probability > 1 - ε.

For Engram with 100 nodes and 60-second gossip intervals, convergence time is approximately 7 rounds = 420 seconds (~7 minutes).
