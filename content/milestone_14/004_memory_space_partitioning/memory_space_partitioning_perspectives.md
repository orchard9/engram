# Perspectives: Memory Space Partitioning

## Perspective 1: Systems Architecture Optimizer

Partitioning is fundamentally about cache locality and minimizing cross-node communication. The key insight: partition boundaries should align with query boundaries. If queries span partitions, you pay network latency on every operation.

For Engram, memory spaces are perfect partitioning units because spreading activation never crosses space boundaries. This means 100% of queries hit a single partition (or its replicas). Compare this to hash-partitioning by node ID - activation spreading would constantly jump between nodes, turning every query into distributed joins.

The consistent hashing implementation needs careful engineering. Naive approaches use cryptographic hashes (SHA256), but that's overkill and slow. For partitioning, we need speed and uniform distribution, not cryptographic security. Use xxHash or ahash - 10x faster, equally uniform distribution.

Virtual nodes are critical for balance. With 150 virtual nodes per physical node, statistical variance is tiny. The math: standard deviation of space counts is proportional to sqrt(V) where V is virtual nodes. More virtual nodes = better balance, but costs more ring traversal time. 150 is the sweet spot validated by Cassandra at scale.

Lock-free ring updates are possible using epoch-based reclamation. The hash ring is immutable - when nodes join/leave, create a new ring and atomically swap via ArcSwap. Readers using old rings see consistent snapshots. Once all readers advance epochs, reclaim old ring memory.

## Perspective 2: Rust Graph Engine Architect

From a graph perspective, partitioning is a graph coloring problem: assign colors (nodes) to vertices (spaces) such that related vertices get the same color. The objective function: minimize edge cuts (queries crossing partitions).

Memory spaces make this trivial. Engram's spaces form a forest of disconnected graphs - activation never crosses space boundaries. This means the graph naturally partitions with zero edge cuts. Perfect locality.

The challenge is intra-space partitioning - can we partition a single large space across multiple nodes for parallelism? This requires graph partitioning algorithms (METIS, KaHIP) to minimize cuts. For Milestone 14, we defer this. Spaces are our atomic partitioning units.

Replication adds an interesting graph property: each space exists at 3 locations (1 primary + 2 replicas). From a query routing perspective, this creates a hypergraph where query edges can connect to any replica. The routing algorithm becomes: "Find shortest path to any replica in set {p, r1, r2}."

Implementation-wise, represent the partition map as a specialized graph structure:

```rust
struct PartitionGraph {
    // Map space_id to (primary, replicas)
    assignments: DashMap<SpaceId, (NodeId, SmallVec<[NodeId; 2]>)>,
    // Reverse index: node_id to spaces it owns
    node_spaces: DashMap<NodeId, HashSet<SpaceId>>,
    // Consistent hash ring for efficient lookup
    ring: ArcSwap<ConsistentHashRing>,
}
```

The reverse index enables fast "which spaces does this node own?" queries, needed for rebalancing and failure recovery.

## Perspective 3: Verification Testing Lead

Testing partitioning requires validating both correctness (spaces assigned correctly) and balance (even distribution). My test strategy:

**Correctness Tests:**

Property 1: Every space has exactly 1 primary
Property 2: Every space has exactly N replicas (configurable)
Property 3: Primary and replicas are distinct nodes
Property 4: When nodes join/leave, space assignments converge to new ring state

Use property-based testing to generate arbitrary node sets and verify properties hold:

```rust
proptest! {
    #[test]
    fn space_assignments_valid(
        num_nodes in 1..20usize,
        num_spaces in 1..1000usize,
        replication_factor in 1..5usize,
    ) {
        let nodes = generate_nodes(num_nodes);
        let spaces = generate_spaces(num_spaces);

        let assigner = SpaceAssigner::new(nodes, replication_factor);

        for space in spaces {
            let assignment = assigner.assign(space);

            // Property 1: exactly 1 primary
            assert_eq!(assignment.replicas.len(), replication_factor);

            // Property 2: all distinct
            let mut unique = HashSet::new();
            unique.insert(assignment.primary);
            for r in &assignment.replicas {
                assert!(unique.insert(*r));
            }
        }
    }
}
```

**Balance Tests:**

Measure distribution uniformity using coefficient of variation (std_dev / mean). For good balance, CV should be < 0.05 (5% variation).

```rust
fn test_distribution_balance() {
    let nodes = generate_nodes(10);
    let spaces = generate_spaces(10000);

    let assigner = SpaceAssigner::new(nodes, 3);
    let mut counts = HashMap::new();

    for space in spaces {
        let primary = assigner.assign(space).primary;
        *counts.entry(primary).or_insert(0) += 1;
    }

    let values: Vec<f64> = counts.values().map(|&v| v as f64).collect();
    let cv = coefficient_of_variation(&values);

    assert!(cv < 0.05, "Distribution CV: {}", cv);
}
```

**Rebalancing Tests:**

Verify that when a node joins/leaves, only the minimal number of spaces move. With consistent hashing, expected moves = total_spaces / num_nodes.

Track which spaces actually move and verify it matches theoretical expectation within 10%.

## Perspective 4: Cognitive Architecture Designer

The brain partitions memories spatially, but this isn't arbitrary - it enables specialized processing. Visual memories in visual cortex benefit from specialized visual processing circuits. Linguistic memories in language areas benefit from syntactic processing.

Engram's space-based partitioning mirrors this functional specialization. Each memory space might represent a different cognitive domain (user, application context, conversation). By colocating all memories for a space on one node, we enable that node to specialize its processing for that space's characteristics.

For example, a space containing primarily visual embeddings could use GPU-accelerated similarity search. A space with temporal sequences could use specialized sequence prediction models. Partitioning by space enables this specialization without cross-node coordination.

The replication strategy parallels how the brain has redundancy for critical information. Important memories get encoded in multiple brain regions (episodic memory in hippocampus AND semantic memory in cortex). Engram's 3x replication ensures critical spaces survive node failures.

From a consolidation perspective, space-based partitioning means consolidation is entirely local - no distributed coordination needed. Each node consolidates its own spaces independently, just like different brain regions consolidate independently during sleep. The only distributed coordination is gossip-based synchronization of consolidation results, which happens asynchronously.

This biological realism extends to rebalancing. When brain regions are damaged, remaining regions gradually take over functions through neuroplasticity. Engram's gradual rebalancing (low-priority moves rate-limited) mirrors this slow adaptation rather than sudden reorganization.

The key cognitive principle: partition boundaries should reflect natural boundaries in the information space. Brain regions map to sensory modalities and cognitive functions. Engram spaces map to users and application contexts. Both enable efficient specialized processing within partitions.
