# Design Memory, Episode, Cue types with probabilistic confidence

## Status: COMPLETE

## Description
Implement core memory types (Memory, Episode, Cue) with cognitive confidence as first-class properties. All types must use the cognitive Confidence type (Task 005) and support confidence propagation through memory operations, following the architecture defined in vision.md.

## Requirements

### Core Memory Types
- Memory type with embedding, activation, confidence, timestamp, decay
- Episode type capturing temporal, spatial, semantic information with confidence
- Cue types for different query patterns (embedding, context, temporal)
- All types must include cognitive Confidence, never Option<Confidence>
- Atomic operations for concurrent activation updates
- Builder pattern for complex type construction using typestate pattern

### Cognitive Confidence Integration
- All confidence operations use cognitive patterns from Task 005
- Confidence propagation through memory consolidation processes
- Confidence decay over time following forgetting curve research
- Confidence combination rules for memory activation spreading
- Natural confidence queries: `memory.seems_reliable()`, `episode.is_vivid()`

### Memory-Specific Confidence Semantics
- Memory activation confidence reflects retrieval certainty
- Episode confidence captures encoding quality and detail richness
- Confidence degradation follows Ebbinghaus forgetting curve patterns
- Spreading activation preserves confidence propagation semantics

## Acceptance Criteria

### Core Functionality
- [ ] Memory stores 768-dim embeddings with AtomicF32 activation
- [ ] Episode includes when, where, who, what with cognitive confidence
- [ ] Cue supports multiple query modalities with confidence thresholds
- [ ] Types are Send + Sync for concurrent access
- [ ] Builder pattern prevents invalid construction at compile time

### Cognitive Confidence Integration
- [ ] All confidence operations use cognitive Confidence type from Task 005
- [ ] Memory operations support natural confidence queries (`is_reliable()`, `seems_accurate()`)
- [ ] Confidence propagation through spreading activation works intuitively
- [ ] Forgetting curve implementation degrades confidence over time naturally
- [ ] Memory consolidation combines confidence values using cognitive-friendly operations

### Memory System Semantics
- [ ] Activation confidence reflects actual retrieval reliability
- [ ] Episode confidence captures subjective memory vividness
- [ ] Confidence decay follows psychologically realistic forgetting patterns
- [ ] Spreading activation preserves confidence semantics across graph traversal

## Dependencies
- Task 005 (Confidence type)

## Notes

### Implementation Details
- Reference vision.md for exact struct definitions
- Use AtomicF32 from atomic_float crate for concurrent activation updates
- Consider arena allocation for Episode collections (memory locality)
- Embeddings should use [f32; 768] not Vec<f32> for cache efficiency

### Cognitive Integration
- All confidence operations must align with research from Task 004 content
- Memory confidence should support frequency-based reasoning where applicable
- Confidence propagation should prevent systematic cognitive biases
- Natural language confidence queries should feel automatic to developers

### Memory System Research Foundation
- Forgetting curves based on Ebbinghaus (1885) and modern memory consolidation research (McGaugh 2000)
- Confidence decay patterns should match empirical memory reliability data following spacing effect research (Ebbinghaus 1885)
- Spreading activation confidence should follow neural network activation patterns with interference theory considerations (Anderson & Neely 1996)
- Episode confidence should reflect psychological research on memory vividness and context-dependent memory (Godden & Badeley 1975)
- Recognition vs recall confidence differences following Mandler (1980) research
- Memory consolidation processes should follow Tulving (1972) episodic to semantic transformation patterns
- Working memory constraints (Baddeley & Hitch 1974) should influence memory type complexity
- See content/0_developer_experience_foundation/005_memory_systems_cognitive_confidence_research.md for comprehensive cognitive architecture research