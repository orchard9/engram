# Task 002: Graph Storage Adaptation

## Objective
Adapt the memory graph storage layer to efficiently handle dual memory types, including separate indices for episodes and concepts.

## Background
The current storage assumes all nodes are homogeneous memories. We need to support heterogeneous node types with different access patterns and indexing requirements.

## Requirements
1. Extend graph backend trait to support node type queries
2. Implement separate indices for episodes and concepts
3. Add type-aware iteration and filtering
4. Maintain backwards compatibility with existing storage
5. Optimize storage layout for cache efficiency

## Technical Specification

### Files to Modify
- `engram-core/src/memory_graph/mod.rs` - Extend traits
- `engram-core/src/memory_graph/graph.rs` - Update UnifiedMemoryGraph
- `engram-core/src/index/mod.rs` - Type-aware indexing
- `engram-core/src/storage/mod.rs` - Storage layout optimization

### New Traits
```rust
pub trait DualMemoryBackend: MemoryBackend {
    fn add_node_typed(&self, node: DualMemoryNode) -> Result<NodeId>;
    
    fn get_node_typed(&self, id: &NodeId) -> Result<Option<DualMemoryNode>>;
    
    fn iter_episodes(&self) -> Box<dyn Iterator<Item = DualMemoryNode>>;
    
    fn iter_concepts(&self) -> Box<dyn Iterator<Item = DualMemoryNode>>;
    
    fn count_by_type(&self) -> (usize, usize); // (episodes, concepts)
}
```

### Storage Layout
```rust
// Separate storage for better cache locality
pub struct DualMemoryStorage {
    episodes: DashMap<NodeId, Arc<DualMemoryNode>>,
    concepts: DashMap<NodeId, Arc<DualMemoryNode>>,
    type_index: DashMap<NodeId, MemoryNodeType>,
}
```

## Implementation Notes
- Use separate DashMaps to avoid type checking during iteration
- Type index enables quick lookups without full deserialization
- Consider memory alignment for SIMD operations
- Implement lazy migration from existing storage

## Testing Approach
1. Unit tests for CRUD operations on both types
2. Concurrent access tests with mixed types
3. Migration tests from legacy storage
4. Performance benchmarks for type-specific queries

## Acceptance Criteria
- [ ] Separate indices for episodes and concepts
- [ ] Type-aware iteration with zero allocation
- [ ] Backwards compatible with existing Memory storage
- [ ] Query performance within 10% of homogeneous storage
- [ ] Concurrent access maintains type consistency

## Dependencies
- Task 001 (Dual Memory Types)

## Estimated Time
3 days