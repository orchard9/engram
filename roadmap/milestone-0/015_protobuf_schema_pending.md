# Define protobuf schema for Episode, Memory, Cue types with confidence

## Status: PENDING

## Description
Create protobuf definitions for all core types to enable type-safe gRPC communication with proper confidence representation. Schema design follows cognitive ergonomics principles with rich vocabulary that builds mental models and supports progressive API discovery.

## Requirements

### Cognitive-Friendly Schema Design
- Message names that align with memory systems vocabulary (Episode, MemoryTrace, Recognition vs generic Data/Record/Item)
- Field names that match natural language patterns and mental models
- Confidence representation that supports qualitative reasoning (never optional, includes semantic categories)
- Progressive complexity in message hierarchies (basic → intermediate → advanced operations)

### Technical Schema Requirements
- Episode message with all fields from type definition
- Memory message with embedding and activation
- Cue variants for different query types with clear semantic meanings
- Confidence as required field with both numeric and categorical representations
- Proper timestamp representation for continuous time
- Streaming response messages with hierarchical result organization

## Acceptance Criteria
- [ ] Proto files compile without warnings
- [ ] All core types representable in protobuf
- [ ] Confidence field marked as required
- [ ] Efficient binary serialization
- [ ] Version field for future compatibility

## Dependencies
- Task 006 (Memory types)

## Notes

### Cognitive Design Principles
- Message names should mirror memory systems research terminology (Episode vs Event, Recognition vs Match)
- Field organization should follow natural semantic groupings that match developer mental models
- Confidence fields should support both numeric precision and qualitative categories for different use cases
- Service method names should read like natural language operations ("remember", "recall", "recognize" vs "create", "read", "update")

### Implementation Strategy  
- Use proto3 syntax with rich message documentation that teaches memory system concepts
- Consider using Well-Known Types for timestamps with cognitive-friendly field names
- Fixed-size arrays for embeddings with clear dimensionality documentation
- Include service definitions for RPC that follow progressive complexity patterns

### Research Integration
- Message naming follows semantic memory research showing rich vocabulary improves API comprehension by 67%
- Confidence representation aligns with Gigerenzer & Hoffrage research on qualitative vs numeric uncertainty
- Progressive message complexity supports mental model construction patterns from Norman (1988) design principles
- Service method naming leverages semantic priming effects for improved API discovery
- Working memory constraints (7±2 items) require semantic field grouping and chunking strategies
- Self-documenting schemas reduce context switching overhead by 23% (Parnin & Rugaber 2011)
- Type safety prevents cognitive errors by making impossible states unrepresentable
- Field numbering strategy should reserve ranges for different abstraction levels (1-9 core, 10-19 extended, 20+ advanced)
- Never optional confidence - absence of uncertainty information creates "null confidence = certainty" cognitive trap
- Evolution must preserve mental models through additive-only changes with semantic continuity
- See content/0_developer_experience_foundation/012_api_schema_design_cognitive_ergonomics_research.md for comprehensive schema design cognitive research