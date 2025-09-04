# Implement gRPC MemoryService with Store and Recall operations

## Status: PENDING

## Description
Build gRPC service implementing the MemoryService interface with Store and Recall operations, including streaming support.

## Requirements
- Implement MemoryService trait from protobuf
- Store operation accepting Episode messages
- Recall operation with streaming responses
- Bidirectional streaming for continuous operations
- Error mapping to gRPC status codes
- TLS support for secure communication

## Acceptance Criteria
- [ ] gRPC server starts on configured port
- [ ] Store operation persists episodes
- [ ] Recall streams results as they're found
- [ ] Graceful handling of client disconnects
- [ ] Performance: 10K ops/sec minimum

## Dependencies
- Task 015 (protobuf schema)
- Task 008 (store operation)
- Task 009 (recall operation)

## Notes
- Use tonic for gRPC implementation
- Consider connection pooling
- Implement interceptors for auth/logging
- Support gRPC-Web for browser clients