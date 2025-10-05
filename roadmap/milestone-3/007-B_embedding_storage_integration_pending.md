# Task 007-B: Embedding Storage Integration for SIMD Batch Spreading

## Objective
Enable embeddings to be stored and retrieved from the memory graph so that SIMD batch spreading can realize its 2-3x performance improvement. Currently, the SIMD infrastructure is complete but always falls back to scalar operations because embeddings are not available.

## Priority
P1 (Performance Critical - Blocks SIMD Performance Gains)

## Effort Estimate
1 day

## Dependencies
- Task 007: SIMD-Optimized Batch Spreading (Complete)

## Current State

The SIMD batch spreading infrastructure is fully implemented and tested in Task 007:
- `ActivationBatch` with AoSoA layout for cache-optimal SIMD access
- `cosine_similarity_batch_768()` for AVX2/FMA batch similarity computation
- Tier-aware SIMD selection (Hot/Warm/Cold)
- Comprehensive test coverage (16/16 tests passing)

**Problem**: The batch spreading code always falls back to scalar because embeddings cannot be retrieved from the graph:

```rust
// engram-core/src/activation/parallel.rs:258-265
let neighbor_embeddings: Vec<_> = neighbors
    .iter()
    .filter_map(|_edge| {
        // Try to get embedding from memory graph
        // This is a placeholder - actual implementation depends on graph storage
        None::<[f32; 768]>
    })
    .collect();
```

## Technical Approach

### Option A: Add Embeddings to WeightedEdge (Recommended)

1. **Extend WeightedEdge Struct**
   ```rust
   pub struct WeightedEdge {
       pub target: NodeId,
       pub weight: f32,
       pub edge_type: EdgeType,
       pub embedding: Option<[f32; 768]>,  // NEW: Store embedding directly
   }
   ```

2. **Update Graph Construction**
   - Modify `ActivationGraphExt::add_edge()` to accept optional embedding
   - Add `add_edge_with_embedding()` method for embedding-aware edges
   - Update all call sites to pass embeddings when available

3. **Update Batch Spreading**
   ```rust
   let neighbor_embeddings: Vec<_> = neighbors
       .iter()
       .filter_map(|edge| edge.embedding)
       .collect();
   ```

**Pros**:
- Simple implementation
- Embeddings stored with edges (locality)
- No additional lookup overhead

**Cons**:
- Increases memory usage per edge (3KB per edge)
- Duplicate embeddings if node has multiple incoming edges

### Option B: Add Embedding Resolver to MemoryGraph

1. **Add Embedding Lookup Method**
   ```rust
   pub trait ActivationGraphExt {
       fn get_embedding(&self, node_id: &NodeId) -> Option<[f32; 768]>;
       fn set_embedding(&self, node_id: &NodeId, embedding: [f32; 768]);
   }
   ```

2. **Implement Storage**
   - Add `DashMap<NodeId, [f32; 768]>` to graph backends
   - Store one embedding per node (deduplicated)

3. **Update Batch Spreading**
   ```rust
   let neighbor_embeddings: Vec<_> = neighbors
       .iter()
       .filter_map(|edge| graph.get_embedding(&edge.target))
       .collect();
   ```

**Pros**:
- Deduplicates embeddings (one per node)
- Lower memory usage
- Clean separation of concerns

**Cons**:
- Additional lookup overhead
- More complex implementation

### Option C: Integration with HNSW Index (Future)

Use existing HNSW index as embedding source:
- Query HNSW index for node embeddings
- Cache results in spreading engine
- Defer until Task 008 (Integrated Recall)

## Implementation Plan (Option B - Recommended)

### 1. Add Embedding Storage to MemoryGraph

**File**: `engram-core/src/memory_graph/mod.rs`

```rust
pub trait ActivationGraphExt {
    // Existing methods...

    /// Store embedding for a node
    fn set_embedding(&self, node_id: &NodeId, embedding: [f32; 768]);

    /// Retrieve embedding for a node
    fn get_embedding(&self, node_id: &NodeId) -> Option<[f32; 768]>;
}
```

### 2. Implement in Graph Backends

**File**: `engram-core/src/memory_graph/backends/dashmap.rs`

```rust
pub struct DashMapBackend {
    graph: DashMap<NodeId, Vec<WeightedEdge>>,
    embeddings: DashMap<NodeId, [f32; 768]>,  // NEW
}

impl ActivationGraphExt for DashMapBackend {
    fn set_embedding(&self, node_id: &NodeId, embedding: [f32; 768]) {
        self.embeddings.insert(node_id.clone(), embedding);
    }

    fn get_embedding(&self, node_id: &NodeId) -> Option<[f32; 768]> {
        self.embeddings.get(node_id).map(|e| *e.value())
    }
}
```

### 3. Update Batch Spreading Integration

**File**: `engram-core/src/activation/parallel.rs`

```rust
fn process_neighbors_batch(
    context: &WorkerContext,
    task: &ActivationTask,
    record: &Arc<ActivationRecord>,
    neighbors: &[WeightedEdge],
    next_tier: StorageTier,
    decay_factor: f32,
) {
    // Get current node's embedding
    let current_embedding = match context.memory_graph.get_embedding(&task.target_node) {
        Some(emb) => emb,
        None => {
            // No embedding available, fallback to scalar
            fallback_to_scalar(context, task, record, neighbors, next_tier, decay_factor);
            return;
        }
    };

    // Collect neighbor embeddings
    let neighbor_embeddings: Vec<[f32; 768]> = neighbors
        .iter()
        .filter_map(|edge| context.memory_graph.get_embedding(&edge.target))
        .collect();

    // Ensure we have all embeddings before using SIMD
    if neighbor_embeddings.len() != neighbors.len() {
        fallback_to_scalar(context, task, record, neighbors, next_tier, decay_factor);
        return;
    }

    // SIMD batch processing (existing code)
    let similarities = cosine_similarity_batch_768(&current_embedding, &neighbor_embeddings);
    // ... rest of SIMD path
}
```

### 4. Add Helper Methods

**File**: `engram-core/src/activation/parallel.rs`

```rust
/// Fallback to scalar processing when embeddings unavailable
fn fallback_to_scalar(
    context: &WorkerContext,
    task: &ActivationTask,
    record: &Arc<ActivationRecord>,
    neighbors: &[WeightedEdge],
    next_tier: StorageTier,
    decay_factor: f32,
) {
    for edge in neighbors {
        let mut next_path = task.path.clone();
        next_path.push(edge.target.clone());
        let new_task = ActivationTask::new(
            edge.target.clone(),
            record.get_activation(),
            edge.weight,
            decay_factor,
            task.depth + 1,
            task.max_depth,
        )
        .with_storage_tier(next_tier)
        .with_path(next_path);

        context.scheduler.enqueue_task(new_task);
    }
}
```

## Acceptance Criteria

- [ ] Embedding storage added to `MemoryGraph` trait
- [ ] Implemented in DashMap and HashMap backends
- [ ] Batch spreading successfully retrieves embeddings
- [ ] SIMD path executes when embeddings available
- [ ] Performance improvement measured: ≥2× speedup on AVX2
- [ ] Tests verify embedding storage and retrieval
- [ ] Integration test shows SIMD path execution
- [ ] Metrics track SIMD vs scalar path usage

## Testing Approach

### Unit Tests
```rust
#[test]
fn test_embedding_storage_and_retrieval() {
    let graph = create_activation_graph();
    let node_id = "test_node".to_string();
    let embedding = [0.5f32; 768];

    graph.set_embedding(&node_id, embedding);
    let retrieved = graph.get_embedding(&node_id).unwrap();

    assert_eq!(embedding, retrieved);
}
```

### Integration Tests
```rust
#[test]
fn test_simd_batch_spreading_with_embeddings() {
    let graph = create_activation_graph();

    // Add nodes with embeddings
    graph.set_embedding("A", [1.0; 768]);
    graph.set_embedding("B1", [0.9; 768]);
    // ... 8 neighbors for batch processing

    // Add edges
    for i in 1..=8 {
        graph.add_edge("A", format!("B{}", i), 0.8, EdgeType::Excitatory);
    }

    let engine = ParallelSpreadingEngine::new(config, graph)?;
    let results = engine.spread_activation(&[("A", 1.0)])?;

    // Verify SIMD path was used (check metrics)
    assert!(engine.metrics.simd_batch_count > 0);
}
```

### Performance Benchmarks
```rust
#[bench]
fn bench_spreading_with_simd_embeddings(b: &mut Bencher) {
    let graph = create_large_graph_with_embeddings(1000);
    let engine = ParallelSpreadingEngine::new(config, graph)?;

    b.iter(|| {
        engine.spread_activation(&[("root", 1.0)])
    });
}
```

## Success Metrics

1. **SIMD Execution Rate**: >50% of neighbor processing uses SIMD batch path
2. **Performance Improvement**: ≥2× speedup on AVX2 hardware for graphs with embeddings
3. **Memory Overhead**: <10% increase in graph memory usage
4. **Test Coverage**: All embedding paths tested (storage, retrieval, fallback)

## Risk Mitigation

- **Memory Pressure** → Monitor memory usage, implement LRU cache if needed
- **Embedding Staleness** → Document update policy for embeddings
- **Partial Embeddings** → Graceful fallback when some nodes lack embeddings
- **Concurrent Updates** → Use DashMap for thread-safe embedding storage

## Notes

- This task unblocks the full 2-3× SIMD performance improvement from Task 007
- Embeddings should be populated from the HNSW index or memory store during graph construction
- Consider lazy loading of embeddings from persistent storage in future optimization
- Integration with Task 008 (Integrated Recall) will provide automatic embedding population

## Related Tasks

- Task 007: SIMD-Optimized Batch Spreading (Complete - provides infrastructure)
- Task 008: Integrated Recall Implementation (Pending - will use embeddings)
- Task 002: Vector Similarity Activation Seeding (Complete - provides embeddings from HNSW)
