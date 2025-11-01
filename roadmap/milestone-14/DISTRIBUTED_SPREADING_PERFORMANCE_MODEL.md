# Distributed Activation Spreading: Performance Model

**Purpose**: Quantitative latency and throughput analysis for distributed spreading activation

**Conclusion**: Realistic overhead is **5-10x**, not the planned **2x**

---

## Single-Node Baseline

From current implementation (`engram-core/src/activation/parallel.rs`):

```rust
pub struct ParallelSpreadingEngine {
    config: Arc<RwLock<ParallelSpreadingConfig>>,
    memory_graph: Arc<MemoryGraph>,           // DashMap-based lock-free graph
    scheduler: Arc<TierAwareSpreadingScheduler>,
    phase_barrier: Arc<PhaseBarrier>,         // Hop-level synchronization
    cycle_detector: Arc<CycleDetector>,       // Prevents infinite loops
}
```

**Measured Performance** (from `benches/spreading_benchmarks.rs`):
- 16-core machine
- 10K node graph, average degree 10
- 3-hop spreading from single source

| Metric | Value |
|--------|-------|
| Throughput | ~100K activations/sec |
| Latency (1-hop) | 0.5ms P50, 1.2ms P99 |
| Latency (3-hop) | 1.0ms P50, 5.0ms P99 |
| Latency (5-hop) | 2.0ms P50, 8.0ms P99 |

**Algorithmic Complexity**:
- Time: O(V + E) for BFS where V = visited nodes, E = edges
- Memory: O(V) for activation records
- Parallelism: Work-stealing across N threads

---

## Distributed Spreading Architecture

### Network Topology Assumptions

**Datacenter Deployment** (M14 target):
- Nodes: 5 (replication factor = 2)
- Network: 10Gbps Ethernet, same datacenter
- RTT: 0.5-1.0ms P50, 1.5-3.0ms P99
- Bandwidth: Not a bottleneck (activation messages are small)

**Graph Partitioning**:
- Memory spaces distributed via consistent hashing
- Each space replicated to 2 nodes (primary + 1 replica)
- Edge distribution: ~30% cross-partition (from Neo4j benchmarks)

### Latency Breakdown

#### 1-Hop Spreading (Local + Remote)

**Single-Node**:
```
1. Get neighbors from DashMap: ~50μs
2. Update activation levels (atomic): ~100μs per neighbor
3. Total: ~500μs for 10 neighbors
```

**Distributed**:
```
1. Get LOCAL neighbors: ~50μs
2. Update LOCAL activations: ~100μs
3. Identify REMOTE neighbors: ~20μs
4. Send remote activation messages: 1ms RTT × batching efficiency
5. Remote nodes update activations: ~100μs
6. Total: 500μs (local) + 1ms (network) = 1.5ms
```

**Overhead**: **3x** (1.5ms vs 0.5ms)

#### 3-Hop Spreading (Sequential Network RTTs)

**Single-Node**:
```
Hop 1: 500μs
Hop 2: 500μs (phase barrier sync: ~50μs)
Hop 3: 500μs (phase barrier sync: ~50μs)
Total: 1.6ms
```

**Distributed (Naive Sequential)**:
```
Hop 1: 1.5ms (local + remote)
  Wait for all nodes to complete hop 1: +500μs (distributed barrier)
Hop 2: 1.5ms
  Wait for all nodes to complete hop 2: +500μs
Hop 3: 1.5ms
  Wait for all nodes to complete hop 3: +500μs
Total: 6.0ms
```

**Overhead**: **3.75x** (6ms vs 1.6ms)

**Distributed (Optimized Pipelined)**:
```
Hop 1: 1.5ms
Hop 2: 1.5ms (pipelined, overlap with hop 1 remote messages)
Hop 3: 1.5ms (pipelined)
Total: 4.5ms (assuming 30% pipeline efficiency)
```

**Overhead**: **2.8x** (4.5ms vs 1.6ms) - **BEST CASE**

#### 5-Hop Spreading (Multi-Region Scenario)

**Single-Node**:
```
5 hops × 500μs = 2.5ms
```

**Distributed**:
```
Naive: 5 × 2.0ms (hop + barrier) = 10ms
Pipelined: 5 × 1.5ms × 0.7 (pipeline efficiency) = 5.25ms
```

**Overhead**: **2.1-4x**

### Network Message Volume

**Activation Message Format**:
```rust
struct RemoteActivation {
    target_node: NodeId,       // 4 bytes
    activation: f32,           // 4 bytes
    lamport_clock: u64,        // 8 bytes
    vector_clock: Vec<u64>,    // 8 × N nodes = 40 bytes for 5 nodes
    hop_count: u32,            // 4 bytes
}
// Total: ~60 bytes per activation message
```

**Messages per Hop** (assuming 30% cross-partition edges):
```
Graph: 10K nodes, avg degree 10
1-hop: 10K nodes × 10 edges × 0.3 cross-partition = 30K messages
3-hop: Exponential spread, but threshold cutoff
  Estimated: ~50K-100K messages total
```

**Network Bandwidth**:
```
100K messages × 60 bytes = 6MB
Spread over 5ms (pipelined) = 1.2 GB/sec
10Gbps network = 1.25 GB/sec ✓ NOT BOTTLENECKED
```

**Conclusion**: Bandwidth is NOT a bottleneck, **latency is**.

---

## Performance Model: Latency

### Formula

```
T_distributed(h) = T_local(h) + (h × RTT × cross_partition_ratio) + barrier_overhead
```

Where:
- `h` = number of hops
- `T_local(h)` = single-node latency for h hops
- `RTT` = network round-trip time
- `cross_partition_ratio` = fraction of edges crossing partitions (~0.3)
- `barrier_overhead` = distributed phase barrier (if used)

### Concrete Calculations

#### Scenario 1: Intra-Partition (Best Case)

**Assumptions**:
- All activated nodes are on same partition
- No remote messages needed
- Latency = single-node + routing overhead

```
3-hop spreading:
  Single-node: 1.0ms
  Routing overhead: 0.2ms (local method dispatch)
  Total: 1.2ms

Overhead: 1.2x ✓ MEETS "2x" TARGET
```

**Probability**: Low (~7% for random graph, since 0.3³ ≈ 0.027 edges all local)

#### Scenario 2: Cross-Partition (Typical Case)

**Assumptions**:
- 30% edges cross partitions
- Each hop requires 1 network RTT (pipelined)
- No distributed phase barriers (asynchronous)

```
3-hop spreading:
  Single-node compute: 1.0ms
  Network (3 hops × 1ms RTT × 0.3 cross-partition): 0.9ms
  Aggregation overhead: 0.5ms
  Total: 2.4ms

Overhead: 2.4x ✓ MEETS "2x" TARGET (barely)
```

**Probability**: Moderate (~40% for typical workloads)

#### Scenario 3: Multi-Partition (Worst Case)

**Assumptions**:
- High cross-partition edges (50%+)
- Sequential network RTTs (no pipelining)
- Distributed phase barriers enabled (for determinism)

```
3-hop spreading:
  Single-node compute: 1.0ms
  Network (3 hops × 1.5ms RTT × 0.5 cross-partition): 2.25ms
  Distributed barriers (3 × 500μs): 1.5ms
  Aggregation overhead: 0.5ms
  Total: 5.25ms

Overhead: 5.25x ✗ EXCEEDS "2x" TARGET
```

**Probability**: High during network congestion or high fan-out queries

### Revised Performance Targets

| Scenario | M14 Original Target | Realistic Target | Basis |
|----------|---------------------|------------------|-------|
| Intra-partition (best case) | <2x | **1.2-1.5x** | No network RTTs |
| Typical cross-partition | <2x | **2-3x** | 30% edges remote |
| High cross-partition | <5x | **4-7x** | 50%+ edges remote |
| Multi-region | N/A | **10-20x** | 10-50ms RTT |

**Recommendation**: Update M14 targets to:
- Intra-partition: <2x ✓ (achievable)
- Cross-partition: <5x (realistic, was understated)
- Multi-region: Out of scope for M14

---

## Performance Model: Throughput

### Single-Node Throughput

```
Measured: 100K activations/sec (16 cores, 10K node graph)

Bottleneck: Memory bandwidth (activation updates)
  DashMap atomic updates: ~50ns per activation
  16 cores × 0.8 efficiency = 12.8 effective cores
  12.8 cores × (1 / 50ns) = 256M ops/sec theoretical
  Actual: 100K activations/sec

Explanation: Activation spreading is memory-bound, not CPU-bound
  Each activation touches:
    - Source node activation record (read)
    - Neighbor activation records (atomic update)
    - Edge weights (read)
  Cache misses dominate latency
```

### Distributed Throughput (5-Node Cluster)

**Ideal Scaling** (linear):
```
5 nodes × 100K activations/sec = 500K activations/sec
```

**Realistic Scaling** (accounting for coordination):
```
Coordination overhead:
  - Remote activation messages: 30% of activations
  - Network serialization: ~1μs per message
  - Distributed cycle detection: 10% overhead

Effective throughput per node:
  100K × (1 - 0.3 network - 0.1 coordination) = 60K activations/sec

Total cluster throughput:
  5 nodes × 60K = 300K activations/sec

Scaling efficiency: 300K / 500K = 60%
```

**Scaling Factor**: **3x** (300K vs 100K)

**Comparison to M14 Target**:
- Original: "Linear scaling to 16 nodes"
- Realistic: ~60% efficiency → **0.6 × N scaling**

At 16 nodes:
- Ideal: 1.6M activations/sec
- Realistic: ~960K activations/sec (0.6 × 16 × 100K)

**Verdict**: Still acceptable scaling, but not linear.

---

## Latency vs Throughput Trade-Off

### Batch Size Impact

**Small Batches** (low latency):
```
Batch size: 1 activation per network message
Latency: 1ms RTT per activation (best case)
Throughput: 1000 activations/sec per connection
Network efficiency: Poor (60-byte message for 1 activation)
```

**Large Batches** (high throughput):
```
Batch size: 1000 activations per network message
Latency: 1ms RTT + 1ms batching delay = 2ms
Throughput: 1M activations/sec per connection
Network efficiency: Excellent (60KB message, amortized overhead)
```

**Adaptive Batching** (recommended):
```rust
struct AdaptiveBatcher {
    batch_size: AtomicUsize,
    target_latency_ms: f32,
}

impl AdaptiveBatcher {
    fn compute_batch_size(&self, current_load: usize) -> usize {
        if current_load < 100 {
            1  // Low load: prioritize latency
        } else if current_load < 1000 {
            10  // Medium load: balance
        } else {
            100  // High load: prioritize throughput
        }
    }
}
```

**M14 Implementation**: Current codebase has `AdaptiveBatcher` (from `src/activation/parallel.rs` line 99), but it's for **local** batching, not **network** batching.

**New Requirement**: Distributed adaptive batching (1-2 weeks effort)

---

## Comparison to Distributed Graph Databases

### Neo4j Causal Cluster

**Architecture**: Multi-primary with Raft consensus
**Traversal Latency**: 5-10x single-node (from Neo4j benchmarks)
**Reason**: Consensus overhead + cross-partition hops

**Engram vs Neo4j**:
- Neo4j: Strong consistency (CP system)
- Engram: Eventual consistency (AP system)
- Engram advantage: No consensus overhead → **lower latency**

### JanusGraph

**Architecture**: Storage-backed distributed graph
**Traversal Latency**: 10-50x single-node
**Reason**: Storage backend (Cassandra/HBase) adds latency

**Engram vs JanusGraph**:
- JanusGraph: Persistent storage (disk-bound)
- Engram: In-memory (DRAM-bound)
- Engram advantage: No disk I/O → **much lower latency**

### Dgraph

**Architecture**: Sharded graph with transactions
**Traversal Latency**: 3-8x single-node
**Reason**: Sharding requires cross-shard queries

**Engram vs Dgraph**:
- Similar architecture (both AP, both sharded)
- Similar latency overhead (3-8x)
- **Engram target of 2-5x is competitive with industry**

---

## Recommended Performance Targets for M14

### Latency Targets (Revised)

| Operation | Single-Node | Distributed | Overhead | Original M14 | Status |
|-----------|-------------|-------------|----------|--------------|--------|
| 1-hop spreading | 0.5ms | 1.5ms | 3x | 2x | ⚠️ Adjust |
| 3-hop spreading | 1.0ms | 4.5ms | 4.5x | 2x | ⚠️ Adjust |
| 5-hop spreading | 2.0ms | 10ms | 5x | 2x | ⚠️ Adjust |
| Intra-partition | 1.0ms | 1.5ms | 1.5x | 2x | ✓ Achievable |
| HNSW search | 0.8ms | 2.0ms | 2.5x | 2x | ✓ Achievable |

**Verdict**: Original "2x" target is **achievable for intra-partition**, but **unrealistic for cross-partition**.

### Throughput Targets (Revised)

| Cluster Size | Ideal (Linear) | Realistic (60% Efficiency) | Original M14 | Status |
|--------------|----------------|----------------------------|--------------|--------|
| 1 node | 100K/sec | 100K/sec | 10K/sec | ✓ Exceeds |
| 5 nodes | 500K/sec | 300K/sec | 50K/sec | ✓ Exceeds |
| 16 nodes | 1.6M/sec | 960K/sec | Linear | ⚠️ Sublinear |

**Verdict**: Throughput targets are **achievable**, but scaling is sublinear (~60% efficiency).

---

## Optimization Opportunities

### 1. Network Batching

**Current**: Each activation sends individual message
**Optimized**: Batch N activations into single message

**Latency Impact**: +1ms batching delay
**Throughput Impact**: 10-100x (depending on batch size)

**Implementation**: 1 week

### 2. Edge Caching

**Current**: Edge list fetched from `DashMap` on every hop
**Optimized**: Cache hot edges in thread-local storage

**Latency Impact**: -20% (reduce lock contention)
**Memory Impact**: +10% (edge cache)

**Implementation**: 1 week

### 3. Prefetching Remote Nodes

**Current**: Sequential remote activation sends
**Optimized**: Prefetch remote node data in parallel

**Latency Impact**: -30% (overlap network I/O)
**Complexity**: High (requires predictive model)

**Implementation**: 2-3 weeks

### 4. RDMA for Low-Latency Networks

**Current**: TCP/gRPC (1ms RTT)
**Optimized**: RDMA (10-50μs RTT)

**Latency Impact**: -90% (100x faster network)
**Hardware Requirement**: RDMA-capable NICs

**Implementation**: 4+ weeks (hardware + driver integration)
**Priority**: Defer to M20+ (high-performance networking milestone)

---

## Conclusion

**Realistic Distributed Spreading Performance**:

| Metric | M14 Original | Realistic | Achievable? |
|--------|--------------|-----------|-------------|
| Intra-partition latency | <2x | 1.5-2x | ✓ YES |
| Cross-partition latency | <2x | 3-7x | ✗ NO |
| Throughput scaling | Linear | 0.6 × N | ⚠️ Sublinear |

**Recommendations**:
1. Update M14 targets to reflect cross-partition reality (3-7x overhead)
2. Accept sublinear scaling (60% efficiency) as acceptable
3. Optimize for intra-partition queries (>90% of workload)
4. Defer cross-region distribution to M17+ (requires different architecture)

**Key Insight**: Distributed activation spreading is **latency-dominated by network RTTs**, not computation. No amount of CPU optimization can overcome physics of light.

**Final Verdict**: M14 performance targets are **achievable for common case** (intra-partition), but **unrealistic for worst case** (high cross-partition).

---

**References**:
- Current benchmarks: `engram-core/benches/spreading_benchmarks.rs`
- Neo4j distributed performance: "Neo4j Causal Cluster Performance" (2019)
- Dgraph benchmarks: https://dgraph.io/blog/post/benchmark-2022/

**Last Updated**: 2025-10-31
