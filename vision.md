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

### Phase 1: Single-Node Foundation
Rust implementation of core graph with probabilistic operations. Focus on correctness over performance. Validate cognitive principles.

### Phase 2: Performance Optimization  
Profile and rewrite hot paths in Zig. Add SIMD operations. Implement specialized allocators.

### Phase 3: GPU Acceleration
CUDA/ROCm kernels for embedding operations. Parallel activation spreading on GPU. Maintain CPU fallback.

### Phase 4: Distribution
Shard graph across nodes. Implement gossip protocols. Design partition-tolerant operations.

### Phase 5: Production Systems
Monitoring, debugging tools, query language stabilization. Performance regression framework.

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
