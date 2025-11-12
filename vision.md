# Vision

## Architecture Principles

1. **Probabilistic Foundation**: Every operation returns probability distributions, not discrete values
2. **Continuous Time**: Timestamps are f64, not discrete transaction IDs
3. **Lazy Reconstruction**: Memories materialize during query, not storage
4. **Local Computation**: Global consistency emerges from local rules
5. **Graceful Degradation**: Partial corruption reduces confidence, not availability

## Core Technical Decisions

### Memory Representation
```rust
struct MemoryNode {
    embedding: [f32; 768],  // Dense vector representation
    activation: AtomicF32,   // Current activation level
    confidence: (f32, f32),  // Interval probability
    last_access: f64,        // Continuous timestamp
    decay_rate: f32,         // Node-specific decay
}
```

Edges carry weights, timestamps, and confidence intervals. Both nodes and edges decay continuously based on configurable functions.

### Query Execution Model

Queries execute via activation spreading:
1. Cue nodes receive initial activation
2. Activation propagates through edges with decay
3. Nodes above threshold enter result set
4. Reconstruction fills gaps via confabulation

No query plan optimization in traditional sense. Instead, spreading parameters tune exploration/exploitation tradeoff.

### Storage Layout

Three-tier architecture:
- **Hot tier**: Lock-free concurrent hashmap for active memories
- **Warm tier**: Append-only log with periodic compression
- **Cold tier**: Columnar storage for embeddings with SIMD operations

Memory migrates between tiers based on activation frequency, not time.

### Concurrency Model

Actor-based memory regions:
- Each region owns a subgraph
- Activation messages pass between regions
- No global locks, only regional coordination
- Eventual consistency through gossip protocols

### GPU Integration

Hybrid CPU/GPU execution:
- CPU: Graph traversal and symbolic operations
- GPU: Embedding operations and batch activation spreading
- Unified memory for zero-copy transfers where available

## Implementation Phases

### Phase 1: Single-Node Foundation ✅ COMPLETE
Rust implementation of core graph with probabilistic operations. Focus on correctness over performance. Validate cognitive principles.

**Status**: Complete (M0-M9)
- Spreading activation with HNSW indexing
- Probabilistic query engine with uncertainty tracking
- Pattern completion with CA3/CA1 dynamics
- Memory consolidation with pattern detection
- SQL-like query language (RECALL, SPREAD, CONSOLIDATE, COMPLETE, IMAGINE)
- Multi-tenant memory spaces

### Phase 2: Performance Optimization ✅ COMPLETE
Profile and rewrite hot paths in Zig. Add SIMD operations. Implement specialized allocators.

**Status**: Complete (M10)
- Zig performance kernels (15-35% speedup)
- SIMD vector operations
- Differential testing between Rust and Zig
- Performance regression framework

### Phase 3: GPU Acceleration ✅ COMPLETE
CUDA/ROCm kernels for embedding operations. Parallel activation spreading on GPU. Maintain CPU fallback.

**Status**: Complete (M12)
- CUDA kernels for vector similarity (10-50x speedup)
- GPU activation spreading
- Hybrid CPU/GPU executor with adaptive switching
- Unified memory allocator
- CPU fallback validated

### Phase 4: Distribution 🔄 IN PROGRESS
Shard graph across nodes. Implement gossip protocols. Design partition-tolerant operations.

**Status**: Prerequisites complete, baseline measurements in progress (M14)
- Consolidation determinism: Fixed for distributed convergence
- Single-node baselines: Weeks 5-7 (in progress)
- 7-day soak test: Weeks 7-10 (pending)
- SWIM membership protocol: M14 Phase 1 (pending)
- Expected completion: 6-9 months after baseline validation

### Phase 5: Production Systems ✅ MOSTLY COMPLETE
Monitoring, debugging tools, query language stabilization. Performance regression framework.

**Status**: Mostly complete (M11, M13, M15, M16)
- Real-time SSE monitoring streams
- Prometheus metrics with Grafana dashboards
- Cognitive pattern validation (semantic priming, interference, reconsolidation)
- HTTP REST + gRPC interfaces with OpenAPI documentation
- Production operations documentation (backup, restore, troubleshooting)
- Remaining: Streaming optimization (4 tasks in M11)

## Design Constraints

- Maximum 10ms P99 latency for single-hop activation
- Support 1M+ nodes with 768-dimensional embeddings
- Sustain 10K activations/second on commodity hardware
- Memory overhead < 2x raw data size
- Zero-copy path from storage to GPU

## Theoretical Foundations

Implementation draws from:
- Kanerva's Sparse Distributed Memory for content addressing
- Hopfield Networks for pattern completion
- Complementary Learning Systems theory for consolidation
- Predictive Coding for reconstruction mechanisms
- Free Energy Principle for activation dynamics

## Success Metrics

- Retrieval follows empirical forgetting curves within 5% error
- Pattern completion accuracy matches human data on standard tasks
- Consolidation produces schemas consistent with psychological findings
- Interference patterns replicate empirical memory research
- Performance scales linearly with number of nodes for read operations

## Competitive Positioning (as of 2025-Q4)

Engram delivers hybrid vector-graph-temporal operations in a unified architecture that outperforms
specialized systems on integrated workloads while maintaining competitive performance on pure operations.

**Performance vs Specialized Systems**:
- Graph traversal: Target <15ms P99 vs Neo4j 27.96ms (46% faster target)
- Vector search: Target <20ms P99 vs Qdrant 22-24ms (competitive parity)
- Hybrid workload: Target <10ms P99 (no direct competitor)

**Key Differentiators**:
1. Only system supporting spreading activation + temporal decay in unified queries
2. Integrated pattern completion using hippocampal-neocortical dynamics
3. Probabilistic confidence propagation across vector, graph, and temporal operations
4. Memory consolidation as first-class operation (not batch ETL)

**Target Markets**:
- Cognitive AI agents requiring human-like memory dynamics
- RAG systems needing temporal context and spreading activation
- Knowledge graphs requiring vector similarity alongside relational queries
- Research platforms studying biologically-plausible memory systems

For detailed competitive baselines and quarterly trends, see `docs/reference/competitive_baselines.md`.
