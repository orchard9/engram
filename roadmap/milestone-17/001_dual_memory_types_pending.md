# Task 001: Dual Memory Type Definitions

## Objective
Define the core type system for dual memory architecture including MemoryNodeType enum, DualMemoryNode struct, and type conversion utilities.

## Background
Currently Engram only supports episodic memories. We need to introduce a type system that distinguishes between episodes (specific events) and concepts (generalized patterns) while maintaining backwards compatibility.

## Requirements
1. Define `MemoryNodeType` enum with Episode and Concept variants
2. Create `DualMemoryNode` wrapper struct that includes node type
3. Implement conversion traits between Memory and DualMemoryNode
4. Add feature flag `dual_memory_types` to gate new functionality
5. Ensure zero-copy conversions where possible

## Technical Specification

### Files to Create
- `engram-core/src/memory/dual_types.rs` - Core type definitions
- `engram-core/src/memory/conversions.rs` - Type conversion implementations

### Files to Modify  
- `engram-core/src/memory/mod.rs` - Add module exports
- `engram-core/src/lib.rs` - Feature flag definition
- `engram-core/Cargo.toml` - Add dual_memory feature

### Type Definitions
```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MemoryNodeType {
    Episode {
        episode_id: EpisodeId,
        strength: f32,
        consolidation_score: AtomicF32,
    },
    Concept {
        centroid: [f32; 768],
        coherence: f32,
        instance_count: AtomicU32,
        formation_time: DateTime<Utc>,
    }
}

#[derive(Debug)]
pub struct DualMemoryNode {
    pub id: NodeId,
    pub node_type: MemoryNodeType,
    pub embedding: [f32; 768],
    pub activation: AtomicF32,
    pub confidence: Confidence,
    pub last_access: DateTime<Utc>,
    pub created_at: DateTime<Utc>,
}
```

## Implementation Notes
- Use atomic types for fields that may be updated concurrently
- Concept centroid is the average embedding of clustered episodes
- Coherence measures how tightly clustered the episodes are (0-1)
- Episode strength represents consolidation progress

## Testing Approach
1. Unit tests for type construction and field access
2. Serialization round-trip tests
3. Property tests for valid value ranges
4. Benchmark memory overhead of new types

## Acceptance Criteria
- [ ] MemoryNodeType enum compiles and serializes correctly
- [ ] DualMemoryNode provides ergonomic API
- [ ] Feature flag properly gates functionality
- [ ] Zero-copy conversion from Memory to DualMemoryNode
- [ ] Memory overhead <10% compared to current Memory type
- [ ] All existing tests pass with feature disabled

## Dependencies
- None

## Estimated Time
2 days