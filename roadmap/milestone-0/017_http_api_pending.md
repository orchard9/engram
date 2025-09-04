# Build HTTP REST API with POST /api/v1/episodes and /api/v1/recall

## Status: PENDING

## Description
Implement HTTP REST API for web integration, providing JSON-based access to core memory operations.

## Requirements
- POST /api/v1/episodes to store new episodes
- POST /api/v1/recall for memory retrieval
- GET /api/v1/metrics for observability
- Proper HTTP status codes and error responses
- CORS support for browser access
- Request validation with helpful errors

## Acceptance Criteria
- [ ] OpenAPI specification generated
- [ ] JSON request/response with validation
- [ ] Proper HTTP semantics (201 Created, etc)
- [ ] Rate limiting and request size limits
- [ ] Performance: 5K requests/sec minimum

## Dependencies
- Task 008 (store operation)
- Task 009 (recall operation)

## Notes
- Use axum or actix-web
- problem+json format for errors
- Consider ETags for caching
- Support both JSON and MessagePack