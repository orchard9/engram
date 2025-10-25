# Twitter Thread: Network Partition Handling in Engram

## Tweet 1 (Hook)
CAP theorem forces a choice during network partitions: consistency or availability? Traditional databases choose consistency and refuse requests. Engram chooses availability. Why? Because your brain doesn't freeze when it can't access memories.

## Tweet 2 (Detection Problem)
From one node's view, unreachable peers look identical whether they crashed or there's a partition. SWIM's indirect probing helps, but the real solution is vector clocks - they detect when updates happened concurrently in different partitions.

## Tweet 3 (Vector Clocks)
Each node maintains logical timestamps for all cluster members. When comparing vector clocks, three outcomes: equal, ordered (causal), or concurrent (partition!). Concurrent clocks mean updates happened independently without knowledge of each other.

## Tweet 4 (Operation During Partition)
When partitioned, Engram doesn't block operations. Writes continue with vector clock metadata. Reads apply confidence penalty based on cluster visibility. Consolidation runs locally. Graceful degradation, not failure.

## Tweet 5 (Confidence Penalty)
If node sees 4 of 5 cluster members, confidence penalty is small. If it sees only 1 of 5, penalty is large. Results marked with warnings: "Partial cluster visibility - may be incomplete." Like your brain's uncertainty when memories are fuzzy.

## Tweet 6 (Healing Protocol)
When partition heals: exchange Merkle tree roots, identify divergent regions, merge updates with conflict resolution, verify convergence. If 99% of state matches, only transfer the 1% that diverged. Healing completes in seconds.

## Tweet 7 (Split-Brain Prevention)
Partition generation numbers and vector clocks prevent split-brain corruption. Conservative merge: keep updates from both sides, resolve conflicts via causality and confidence, never lose data. Manual review for true conflicts.

## Tweet 8 (Performance)
Benchmarks on 5-node cluster: detection in 2.1s p99, 95% throughput during partition, 4.7s healing time for 10K updates, 0% data loss. Vector clock overhead minimized via delta encoding - 24 bytes vs 800 bytes for full vectors.
