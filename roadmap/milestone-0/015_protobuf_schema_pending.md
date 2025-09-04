# Define protobuf schema for Episode, Memory, Cue types with confidence

## Status: PENDING

## Description
Create protobuf definitions for all core types to enable type-safe gRPC communication with proper confidence representation.

## Requirements
- Episode message with all fields from type definition
- Memory message with embedding and activation
- Cue variants for different query types
- Confidence as required field, not optional
- Proper timestamp representation
- Streaming response messages

## Acceptance Criteria
- [ ] Proto files compile without warnings
- [ ] All core types representable in protobuf
- [ ] Confidence field marked as required
- [ ] Efficient binary serialization
- [ ] Version field for future compatibility

## Dependencies
- Task 006 (Memory types)

## Notes
- Use proto3 syntax
- Consider using Well-Known Types for timestamps
- Fixed-size arrays for embeddings
- Include service definitions for RPC