# Milestone 17: Dual Memory Architecture

## Overview
Transform Engram from a pure episodic memory system to a dual memory architecture supporting both episodic (specific events) and semantic (generalized concepts) memory types. This enables more efficient storage, better generalization, and biologically plausible memory consolidation.

## Goals
1. **Introduce dual memory types** - Episode and Concept nodes with distinct properties
2. **Enable concept formation** - Extract semantic patterns from episodic clusters
3. **Implement concept-mediated recall** - Use concepts for generalization and reconstruction
4. **Support fan effects** - Model cognitive interference patterns
5. **Maintain backwards compatibility** - Non-breaking migration path

## Design Principles
- **Type-safe architecture** - Compile-time guarantees for memory type handling
- **Gradual migration** - Feature flags enable incremental rollout
- **Performance conscious** - <5% initial regression, optimization follows correctness
- **Biologically grounded** - Match hippocampal-neocortical consolidation patterns
- **Observable behavior** - Rich metrics for cognitive dynamics

## Technical Approach

### 1. Memory Type System
```rust
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
```

### 2. Consolidation Integration
- Clustering algorithm identifies episodic patterns
- Concepts formed when coherence threshold exceeded
- Bidirectional bindings maintain episode-concept relationships
- Consolidation scores track transformation progress

### 3. Spreading Dynamics
- Fan effect reduces activation with more connections
- Hierarchical spreading through concept-episode bindings
- Confidence propagation respects uncertainty

### 4. Migration Strategy
- Shadow mode: Create concepts without using them
- Hybrid mode: Blend episodic and semantic recall
- Full mode: Concept-first recall with episodic fallback

## Implementation Phases

### Phase 1: Type System Foundation (Tasks 001-003)
Establish core data structures and storage adaptations

### Phase 2: Concept Formation (Tasks 004-006)
Build clustering and binding formation mechanisms

### Phase 3: Spreading & Recall (Tasks 007-009)
Implement cognitive dynamics with dual memory

### Phase 4: Quality & Optimization (Tasks 010-012)
Validate psychology and optimize performance

### Phase 5: Production Readiness (Tasks 013-015)
Monitoring, testing, and migration tooling

## Success Criteria
- Concept formation rate: 1-5% of episodes become concepts
- Fan effect correlation: r > 0.8 with Anderson (1974)
- Storage efficiency: >40% reduction for semantic memories
- Recall quality: <5% degradation with >30% speed improvement
- Migration success: Zero-downtime transition

## Validation Approach
1. Unit tests for type conversions and storage
2. Property tests for confidence propagation
3. Integration tests for consolidation flows
4. Psychological validation against literature
5. Performance benchmarks before/after
6. Production A/B testing with metrics

## Risk Mitigation
- **Performance regression**: Feature flags allow instant rollback
- **Memory bloat**: Budget enforcement and compression
- **Behavioral changes**: Extensive differential testing
- **Migration failures**: Checkpoint-based recovery

## Dependencies
- Milestone 13 (Cognitive Patterns) for psychological validation
- Existing consolidation system for integration points
- SIMD operations for performance-critical paths

## Timeline
15 tasks × 2 days average = 30 days (6 weeks) estimated