# Research: Network Partition Handling in Distributed Systems

## The Fundamental Challenge

Network partitions are inevitable. A switch fails, a cable gets unplugged, a misconfigured firewall drops packets. Suddenly your five-node cluster splits into two groups that can't communicate. Both groups are alive, both can serve requests, but they can't coordinate.

This is where the CAP theorem bites hard. You can't have Consistency, Availability, and Partition tolerance simultaneously. You must choose two.

## CAP Theorem in Practice

Brewer's CAP theorem (2000), formalized by Gilbert and Lynch (2002), states that in the presence of a network partition, you must choose between:

**Consistency (C)**: Every read receives the most recent write
**Availability (A)**: Every request receives a non-error response

When a partition occurs, CP systems (like most relational databases) sacrifice availability to maintain consistency. They refuse to serve requests that might return stale data. AP systems (like Cassandra, DynamoDB) sacrifice consistency to remain available, accepting that some reads might return old data.

For Engram, the choice is clear: we're an AP system. Why? Because cognitive systems must remain available even with incomplete information. Your brain doesn't freeze when it can't access a memory - it reconstructs what it can and marks uncertainty with lower confidence.

## Detecting Partitions: It's Harder Than You Think

The insidious thing about network partitions is that from each node's perspective, the other side looks dead. Node A can't reach Node B. Is B crashed, or is there a network partition?

This is the fundamental impossibility of distributed consensus in asynchronous networks (Fischer, Lynch, Paterson, 1985): you cannot distinguish a slow node from a dead node with certainty.

SWIM's indirect probing helps but doesn't solve this completely. If Node A can't reach Node B, but Nodes C and D also can't reach B, is B dead or are A, C, and D partitioned from B?

The solution: vector clocks for causality tracking.

## Vector Clocks: Tracking Causality

A vector clock is a data structure that captures the causal relationships between events in a distributed system. Introduced independently by Fidge (1988) and Mattern (1988), vector clocks let us detect when two events are concurrent (happened in different partitions) versus causal (one happened before the other).

Each node maintains a vector of logical timestamps, one per cluster member:

```rust
struct VectorClock {
    // Map from node_id to logical timestamp
    timestamps: HashMap<NodeId, u64>,
}

impl VectorClock {
    // Increment this node's timestamp
    fn tick(&mut self, node_id: NodeId) {
        *self.timestamps.entry(node_id).or_insert(0) += 1;
    }

    // Merge with another vector clock (take max of each element)
    fn merge(&mut self, other: &VectorClock) {
        for (node, timestamp) in &other.timestamps {
            let entry = self.timestamps.entry(*node).or_insert(0);
            *entry = (*entry).max(*timestamp);
        }
    }

    // Compare two vector clocks
    fn compare(&self, other: &VectorClock) -> Ordering {
        let self_ge = self.timestamps.iter()
            .all(|(node, ts)| other.timestamps.get(node).unwrap_or(&0) <= ts);
        let other_ge = other.timestamps.iter()
            .all(|(node, ts)| self.timestamps.get(node).unwrap_or(&0) <= ts);

        match (self_ge, other_ge) {
            (true, true) => Ordering::Equal,      // Identical
            (true, false) => Ordering::Greater,   // Self is later
            (false, true) => Ordering::Less,      // Other is later
            (false, false) => Ordering::Concurrent, // Concurrent!
        }
    }
}
```

When two vector clocks are concurrent, it means the updates happened in different partitions without knowledge of each other. This is our split-brain detector.

## Handling Partitions: Engram's Strategy

Engram's partition handling has three phases: detection, operation during partition, and healing.

### Phase 1: Detection

Use SWIM for failure detection, but augment it with vector clock comparison. When gossip messages arrive, compare vector clocks:

```rust
fn handle_gossip_message(&mut self, msg: GossipMessage) {
    let ordering = self.vector_clock.compare(&msg.vector_clock);

    match ordering {
        Ordering::Greater | Ordering::Less => {
            // Normal case: causal order preserved
            self.vector_clock.merge(&msg.vector_clock);
            self.apply_updates(&msg.updates);
        }
        Ordering::Concurrent => {
            // Partition detected!
            warn!("Concurrent updates detected - possible partition");
            self.partition_state = PartitionState::Suspected;
            self.apply_updates_with_conflict_resolution(&msg.updates);
        }
    }
}
```

### Phase 2: Operation During Partition

Once a partition is suspected, Engram continues operating but marks operations with partition-aware metadata:

```rust
struct PartitionAwareWrite {
    data: MemoryNode,
    vector_clock: VectorClock,
    partition_generation: u64,
}
```

Reads return data but include a confidence penalty:

```rust
fn read_during_partition(&self, query: Query) -> Result {
    let mut results = self.execute_query(query);

    if self.partition_state == PartitionState::Confirmed {
        // Reduce confidence due to possible stale data
        for result in &mut results {
            result.confidence *= 0.8; // 20% penalty
        }
        results.metadata.warnings.push(
            "Results may be incomplete due to network partition"
        );
    }

    Ok(results)
}
```

This follows the biological principle: when your brain can't access all memories (e.g., during stress), it still provides answers but with lower confidence.

### Phase 3: Healing

When the partition heals, nodes need to reconcile divergent state. This is where conflict resolution comes in (covered in Task 008). The key insight: don't discard data from either side. Merge everything and use confidence voting to resolve conflicts.

```rust
async fn heal_partition(&mut self, other_partition: &Membership) {
    info!("Partition healing: merging state");

    // Merge vector clocks
    self.vector_clock.merge(&other_partition.vector_clock);

    // Exchange missing updates
    let our_updates = self.get_updates_since(&other_partition.vector_clock);
    let their_updates = other_partition.get_updates_since(&self.vector_clock);

    // Merge with conflict resolution
    for update in our_updates.chain(their_updates) {
        self.apply_with_conflict_resolution(update).await;
    }

    self.partition_state = PartitionState::Normal;
}
```

## Quorum-Free Operation: Why Engram Doesn't Need Consensus

Traditional CP systems use quorum-based consensus (Raft, Paxos) to ensure consistency. Operations require agreement from a majority of nodes. During a partition, the minority partition becomes unavailable.

Engram deliberately avoids quorums. Why?

1. **Cognitive realism**: Your brain doesn't stop functioning when isolated. It operates with local information and reconciles later.

2. **Availability focus**: Memory retrieval must remain fast and always available. Waiting for quorum would add latency and failure modes.

3. **Eventual consistency suffices**: Memory consolidation operates over hours/days. Immediate consistency isn't necessary.

This is the core tradeoff of AP systems. We accept that different nodes might have slightly different views temporarily, in exchange for always being able to serve requests.

## Split-Brain Prevention via Partition Generation

While Engram remains available during partitions, we need to prevent certain operations that could cause data corruption. The key: partition generation numbers.

Each partition group maintains a generation number that increments when the partition is detected:

```rust
struct PartitionInfo {
    generation: u64,
    members: HashSet<NodeId>,
    detected_at: Instant,
}
```

Operations that require coordination (like schema changes or partition reassignment) are blocked during split-brain:

```rust
fn can_perform_admin_operation(&self) -> bool {
    match &self.partition_state {
        PartitionState::Normal => true,
        PartitionState::Suspected | PartitionState::Confirmed => {
            // Admin operations blocked during partition
            false
        }
    }
}
```

Regular read/write operations continue, but administrative operations wait for healing.

## Testing Partition Handling: Chaos Engineering

The only way to trust partition handling is to test it extensively. Traditional unit tests aren't enough - we need to actually create partitions and verify behavior.

Jepsen (Kingsbury, 2013-2020) pioneered this approach: run a distributed system, inject network partitions using iptables, perform concurrent operations, then verify invariants.

For Engram, the test scenarios include:

1. **Clean split**: Two halves can't communicate at all
2. **Asymmetric partition**: A can send to B, but B can't send to A
3. **Flapping partition**: Partition heals and reforms repeatedly
4. **Minority partition**: 1 node isolated from 4 others

For each scenario, we verify:
- No data loss
- Vector clocks correctly detect concurrency
- Healing converges to consistent state
- Confidence penalties applied appropriately

## Performance Impact of Partition Handling

Vector clocks add overhead to every message. For a cluster of N nodes, each vector clock is N×8 bytes (assuming u64 timestamps). For 100 nodes, that's 800 bytes per message.

The gossip protocol already piggybacks data, so we can include vector clocks in existing messages without adding new network round trips. The CPU cost of vector clock comparison is O(N), which is acceptable for N < 1000.

The real cost is in partition detection and healing. During normal operation (no partitions), overhead is minimal. During partitions, healing can generate significant traffic as nodes exchange missing updates. However, this is acceptable because partitions should be rare in production.

## Academic Foundation

The research on partition handling draws from:

- **CAP Theorem**: Brewer (2000), Gilbert & Lynch (2002)
- **Vector Clocks**: Fidge (1988), Mattern (1988)
- **Eventual Consistency**: Vogels (2009), Dynamo paper
- **Jepsen Testing**: Kingsbury (2013-2020)
- **Byzantine Generals**: Lamport et al. (1982) - though Engram assumes non-Byzantine faults

## Conclusion: Graceful Degradation

Network partitions are not edge cases - they're fundamental realities of distributed systems. The question isn't whether partitions will occur, but how your system handles them.

Engram's approach: continue operating with reduced confidence, detect partitions via vector clocks, heal automatically when connectivity returns. Like a cognitive system that keeps functioning even with incomplete information.

This biological realism - graceful degradation under failure - is what makes Engram viable as a distributed cognitive architecture. The brain doesn't crash during network failures. Neither should Engram.
