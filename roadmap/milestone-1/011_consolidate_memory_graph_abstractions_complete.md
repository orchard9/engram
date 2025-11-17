# Task 011: Consolidate Memory Graph Abstractions

## Status: COMPLETE ✅

## Problem
Multiple overlapping implementations of memory/graph storage exist without clear separation of concerns, causing 30% development velocity loss and high maintenance risk.

## Current State
- `engram-core/src/memory_graph/traits.rs` defines the canonical `MemoryBackend`/`GraphBackend` traits consumed by storage, activation, and recall layers.
- `UnifiedMemoryGraph` lives in `engram-core/src/memory_graph/graph.rs` and exposes helper constructors plus backend-specific impls.
- `engram-core/src/graph.rs`, `engram-core/src/activation/mod.rs`, and `engram-core/src/store.rs` now reuse the unified type with deprecation wrappers for the legacy APIs.

## Progress Update

### Completed (100%):
- ✅ Created unified trait architecture under `engram-core/src/memory_graph/`
- ✅ Implemented `MemoryBackend` and `GraphBackend` traits with `MemoryError` covering storage/index failures
- ✅ Added production-ready backends: `HashMapBackend`, `DashMapBackend`, `InfallibleBackend` (`engram-core/src/memory_graph/backends/`)
- ✅ Implemented `UnifiedMemoryGraph` generic wrapper (single entry point for store/activation/tests)
- ✅ Wired `engram-core/src/graph.rs`, `src/activation/mod.rs`, and `src/store.rs` to use UnifiedMemoryGraph + deprecation shims for the old API
- ✅ Added migration + regression coverage in `engram-core/src/memory_graph/tests.rs`
- ✅ Created helper constructors (`UnifiedMemoryGraph::concurrent()` etc.) to keep call-sites terse during migration

### Consolidation Rollout
All overlapping memory graph implementations now defer to `UnifiedMemoryGraph`:
1. `engram-core/src/graph.rs` – `create_simple_graph`/`create_concurrent_graph` just wrap the new helpers and mark the legacy type as deprecated.
2. `engram-core/src/activation/mod.rs` – Spreading/recall jobs store the `UnifiedMemoryGraph<DashMapBackend>` and leverage trait methods for neighbor traversals.
3. `engram-core/src/store.rs` – `MemoryStore` owns a `UnifiedMemoryGraph<InfallibleBackend>` so API behavior matches storage guarantees.

### Key Improvements:
- Single source of truth for graph operations with backend-specific optimizations
- Backends can be swapped without touching client code (dashmap vs hashmap vs infallible)
- Maintained backward compatibility via deprecated type aliases and helper constructors
- Removed duplicate graph implementations, reducing maintenance debt
- Trait-based API enables additional backends (HNSW, persistence) without churn

## Implementation Plan

### Step 1: Create Unified Memory Trait (src/memory_graph/traits.rs)
```rust
// Create new file: src/memory/traits.rs
pub trait MemoryBackend: Send + Sync {
    type Error: std::error::Error + Send + Sync + 'static;
    
    fn store(&self, id: Uuid, memory: Memory) -> Result<(), Self::Error>;
    fn retrieve(&self, id: &Uuid) -> Result<Option<Memory>, Self::Error>;
    fn search(&self, embedding: &[f32], k: usize) -> Result<Vec<(Uuid, f32)>, Self::Error>;
    fn update_activation(&self, id: &Uuid, activation: f32) -> Result<(), Self::Error>;
    fn spread_activation(&self, source: &Uuid, decay: f32) -> Result<(), Self::Error>;
}

pub trait GraphBackend: MemoryBackend {
    fn add_edge(&self, from: Uuid, to: Uuid, weight: f32) -> Result<(), Self::Error>;
    fn get_neighbors(&self, id: &Uuid) -> Result<Vec<(Uuid, f32)>, Self::Error>;
    fn traverse_bfs(&self, start: &Uuid, max_depth: usize) -> Result<Vec<Uuid>, Self::Error>;
}
```

### Step 2: Implement Backend Adapters (src/memory_graph/backends/)

#### HashMap Backend (src/memory/backends/hashmap.rs)
```rust
pub struct HashMapBackend {
    memories: Arc<RwLock<HashMap<Uuid, Memory>>>,
    edges: Arc<RwLock<HashMap<Uuid, Vec<(Uuid, f32)>>>>,
}

impl MemoryBackend for HashMapBackend {
    type Error = MemoryError;
    
    fn store(&self, id: Uuid, memory: Memory) -> Result<(), Self::Error> {
        self.memories.write()?.insert(id, memory);
        Ok(())
    }
    // ... implement other methods
}
```

#### DashMap Backend (src/memory/backends/dashmap.rs)
```rust
pub struct DashMapBackend {
    memories: Arc<DashMap<Uuid, Memory>>,
    edges: Arc<DashMap<Uuid, Vec<(Uuid, f32)>>>,
    activation_cache: Arc<DashMap<Uuid, AtomicF32>>,
}

impl MemoryBackend for DashMapBackend {
    type Error = MemoryError;
    
    fn store(&self, id: Uuid, memory: Memory) -> Result<(), Self::Error> {
        self.memories.insert(id, memory);
        Ok(())
    }
    
    fn spread_activation(&self, source: &Uuid, decay: f32) -> Result<(), Self::Error> {
        // Parallel activation spreading using rayon
        if let Some(neighbors) = self.edges.get(source) {
            neighbors.par_iter().for_each(|(neighbor_id, weight)| {
                if let Some(activation) = self.activation_cache.get(neighbor_id) {
                    let current = activation.load(Ordering::Relaxed);
                    let new_activation = current + (weight * decay);
                    activation.store(new_activation, Ordering::Relaxed);
                }
            });
        }
        Ok(())
    }
    // ... implement other methods
}
```

### Step 3: Create Unified MemoryGraph (src/memory/graph.rs)
```rust
pub struct UnifiedMemoryGraph<B: GraphBackend> {
    backend: B,
    config: GraphConfig,
}

impl<B: GraphBackend> UnifiedMemoryGraph<B> {
    pub fn new(backend: B, config: GraphConfig) -> Self {
        Self { backend, config }
    }
    
    pub fn store_memory(&self, memory: Memory) -> Result<Uuid, B::Error> {
        let id = Uuid::new_v4();
        self.backend.store(id, memory)?;
        Ok(id)
    }
    
    pub fn recall(&self, cue: &Cue) -> Result<Vec<Memory>, B::Error> {
        match &cue.cue_type {
            CueType::Embedding(vec) => {
                let results = self.backend.search(vec, self.config.max_results)?;
                // Apply spreading activation if enabled
                if self.config.enable_spreading {
                    for (id, _score) in &results {
                        self.backend.spread_activation(id, self.config.decay_rate)?;
                    }
                }
                // Retrieve and return memories
                results.into_iter()
                    .filter_map(|(id, score)| {
                        self.backend.retrieve(&id).ok().flatten()
                            .filter(|m| score >= cue.result_threshold.raw())
                    })
                    .collect()
            }
            // ... handle other cue types
        }
    }
}
```

### Step 4: Migrate Existing Code

#### Update src/graph.rs
```rust
// Replace entire file content with:
pub use crate::memory::{UnifiedMemoryGraph, HashMapBackend};

// Deprecated: Will be removed in next major version
#[deprecated(since = "0.2.0", note = "Use UnifiedMemoryGraph<HashMapBackend> instead")]
pub type MemoryGraph = UnifiedMemoryGraph<HashMapBackend>;
```

#### Update src/activation/mod.rs
```rust
// Remove duplicate MemoryGraph implementation
// Replace with:
use crate::memory::{UnifiedMemoryGraph, DashMapBackend};

pub fn create_parallel_graph(config: GraphConfig) -> UnifiedMemoryGraph<DashMapBackend> {
    UnifiedMemoryGraph::new(DashMapBackend::default(), config)
}
```

#### Update src/store.rs
```rust
// Update MemoryStore to use unified backend
pub struct MemoryStore<B: MemoryBackend = HashMapBackend> {
    graph: UnifiedMemoryGraph<B>,
    wal: Option<WriteAheadLog>,
}

impl<B: MemoryBackend> MemoryStore<B> {
    pub fn new(backend: B) -> Self {
        Self {
            graph: UnifiedMemoryGraph::new(backend, GraphConfig::default()),
            wal: None,
        }
    }
    
    // All existing methods remain but delegate to graph
    pub fn store(&self, memory: Memory) -> Result<Uuid, StoreError> {
        self.graph.store_memory(memory)
            .map_err(|e| StoreError::Backend(e.to_string()))
    }
}
```

### Step 5: Add Migration Tests (src/memory/tests/migration.rs)
```rust
#[cfg(test)]
mod migration_tests {
    use super::*;
    
    #[test]
    fn test_backward_compatibility() {
        // Ensure old API still works during migration period
        #[allow(deprecated)]
        let old_graph = MemoryGraph::default();
        
        let memory = Memory::episodic("test", vec![0.1; 768], Confidence::HIGH);
        let id = old_graph.store_memory(memory).unwrap();
        assert!(old_graph.retrieve(&id).unwrap().is_some());
    }
    
    #[test]
    fn test_backend_equivalence() {
        // Ensure all backends produce same results
        let memory = Memory::episodic("test", vec![0.1; 768], Confidence::HIGH);
        
        let hashmap_graph = UnifiedMemoryGraph::new(
            HashMapBackend::default(),
            GraphConfig::default()
        );
        let dashmap_graph = UnifiedMemoryGraph::new(
            DashMapBackend::default(),
            GraphConfig::default()
        );
        
        let id1 = hashmap_graph.store_memory(memory.clone()).unwrap();
        let id2 = dashmap_graph.store_memory(memory).unwrap();
        
        // Both should successfully store and retrieve
        assert!(hashmap_graph.backend.retrieve(&id1).unwrap().is_some());
        assert!(dashmap_graph.backend.retrieve(&id2).unwrap().is_some());
    }
}
```

## Acceptance Criteria
1. All existing tests pass without modification
2. Single UnifiedMemoryGraph abstraction used throughout codebase
3. Backend can be swapped without changing client code
4. Performance benchmarks show no regression
5. Migration guide documented for external users

## Testing Strategy
1. Run existing test suite to ensure backward compatibility
2. Add property-based tests for backend equivalence
3. Benchmark performance of each backend implementation
4. Test parallel activation spreading with DashMap backend
5. Verify memory usage doesn't increase

## Dependencies
- Must be completed before any new graph features are added
- Blocks milestone-1 tasks that depend on graph operations

## Estimated Effort
2-3 weeks (80 hours)
- Week 1: Create trait abstractions and backend implementations
- Week 2: Migrate existing code and fix tests
- Week 3: Performance optimization and documentation
