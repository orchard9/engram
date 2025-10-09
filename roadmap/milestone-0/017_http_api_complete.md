# Build HTTP REST API with POST /api/v1/episodes and /api/v1/recall

## Status: COMPLETE

## Description
Implement HTTP REST API for web integration, providing JSON-based access to core memory operations. API design follows cognitive ergonomics principles with resource paths and method semantics that align with memory systems mental models rather than generic REST conventions.

## Requirements

### Cognitive-Friendly HTTP API Design
- Resource paths that read like memory operations (/api/v1/memories/remember, /api/v1/memories/recall) vs generic CRUD
- HTTP methods that align with memory system semantics (POST for remembering, GET for recalling with query params)
- JSON response structures that mirror natural memory retrieval patterns (vivid, vague, reconstructed)
- Error responses that teach memory concepts and guide correct usage

### Technical HTTP Requirements
- POST /api/v1/memories/remember to store new episodes (vs generic /episodes)
- GET /api/v1/memories/recall with cognitive query parameters for memory retrieval
- GET /api/v1/system/health for observability with memory system status
- Proper HTTP status codes with cognitive meanings and error responses
- CORS support for browser access with clear security mental models
- Request validation with educational error messages

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

### Cognitive Design Principles
- URL paths should read like natural language memory operations (/memories/remember vs /episodes)
- HTTP status codes should align with memory operation outcomes (202 for "remembering in progress", 200 for successful recall)
- JSON response structure should mirror memory retrieval psychology (immediate, delayed, reconstructed results)
- Error messages should include memory system context and educational guidance

### Implementation Strategy
- Use axum or actix-web with cognitive-friendly route organization and documentation
- Use problem+json format for errors with memory systems terminology and teaching opportunities
- Consider ETags for caching with mental models aligned to memory freshness concepts
- Support both JSON and MessagePack with consistent cognitive vocabulary across formats

### Research Integration
- Resource naming follows semantic memory research showing domain vocabulary improves comprehension by 67% (Myers et al. 2016)
- HTTP method semantics align with natural memory operation mental models vs generic REST patterns (Robillard & DeLine 2011)
- Error response design incorporates teaching opportunities - educational errors improve learning by 34% (Ko & Myers 2005)
- JSON structure mirrors memory retrieval patterns from cognitive psychology (recognition → recall → reconstruction) per Tulving (1985)
- Progressive API complexity follows mental model construction research (60-80% learning improvement)
- Status codes pragmatism: developers reliably know only 5-7 codes, use common ones (Nottingham 2017)
- Rate limiting as conversation with X-Memory-Capacity headers matching cognitive metaphors
- Three-tier response pattern: immediate (high confidence) → associated (spreading activation) → reconstructed (pattern completion)
- See content/0_developer_experience_foundation/014_http_api_cognitive_ergonomics_research.md for comprehensive HTTP API research
- See content/0_developer_experience_foundation/014_http_api_cognitive_ergonomics_perspectives.md for architecture viewpoints
- See content/0_developer_experience_foundation/014_http_api_cognitive_ergonomics_medium.md for implementation examples