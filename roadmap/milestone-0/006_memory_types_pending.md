# Design Memory, Episode, Cue types with probabilistic confidence

## Status: PENDING

## Description
Implement core memory types (Memory, Episode, Cue) with probabilistic confidence as first-class properties, following the architecture defined in vision.md.

## Requirements
- Memory type with embedding, activation, confidence, timestamp, decay
- Episode type capturing temporal, spatial, semantic information
- Cue types for different query patterns (embedding, context, temporal)
- All types must include Confidence, never Option<Confidence>
- Atomic operations for concurrent activation updates
- Builder pattern for complex type construction

## Acceptance Criteria
- [ ] Memory stores 768-dim embeddings with AtomicF32 activation
- [ ] Episode includes when, where, who, what with confidence
- [ ] Cue supports multiple query modalities
- [ ] Types are Send + Sync for concurrent access
- [ ] Builder pattern prevents invalid construction

## Dependencies
- Task 005 (Confidence type)

## Notes
- Reference vision.md for exact struct definitions
- Use AtomicF32 from atomic_float crate
- Consider arena allocation for Episode collections
- Embeddings should use [f32; 768] not Vec<f32>