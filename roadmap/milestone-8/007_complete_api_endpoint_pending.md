# Task 007: Complete API Endpoint

**Status:** Pending
**Priority:** P1 (Important)
**Estimated Effort:** 2 days
**Dependencies:** Task 006 (Confidence Calibration)

## Objective

Implement production `POST /api/v1/complete` HTTP endpoint and `Complete()` gRPC method exposing pattern completion functionality. Integrate with multi-tenant memory space routing from Milestone 7 and provide comprehensive error handling with actionable messages.

## Integration Points

**Extends:**
- `/engram-cli/src/api.rs` - HTTP REST API server
- `/engram-cli/src/grpc_server.rs` - gRPC service implementation
- `/engram-cli/proto/engram.proto` - Protocol buffer definitions

**Uses:**
- `/engram-core/src/completion/hippocampal.rs` - HippocampalCompletion engine
- `/engram-core/src/registry.rs` - MemorySpaceRegistry for multi-tenancy from M7
- `/engram-core/src/error.rs` - EngramError types

**Creates:**
- `/engram-cli/src/handlers/complete.rs` - HTTP handler for completion
- `/engram-cli/src/grpc/complete.rs` - gRPC completion service
- `/engram-cli/tests/api/complete_tests.rs` - API integration tests

## Theoretical Foundations from Research

### Fielding's REST Constraints (2000)

Representational State Transfer architectural style for distributed systems:

**1. Client-Server Separation:** API server independent of completion engine
**2. Stateless Communication:** Each /complete request is independent (no session state)
**3. Cacheable Responses:** Completion can be cached per partial cue (deterministic)
**4. Uniform Interface:** JSON request/response format consistent across endpoints
**5. Layered System:** API layer, completion layer, storage layer decoupled

**Application to /api/v1/complete:**
```http
POST /api/v1/complete
X-Memory-Space: user_alice  # Stateless tenant identification
Content-Type: application/json
```

Each request is self-contained. Response includes cache hints (completion deterministic given cue).

### HTTP Status Codes (RFC 7231)

Precise status codes enable client decision-making:

**200 OK:** Successful completion
```json
{"completed_episode": {...}, "completion_confidence": 0.82}
```

**422 Unprocessable Entity:** Valid request format, but insufficient evidence for completion
```json
{
  "error": "InsufficientPattern",
  "message": "Pattern completion requires minimum 30% cue overlap",
  "details": {"cue_overlap": 0.15, "required_minimum": 0.30},
  "suggestion": "Provide additional context fields or reduce ca1_threshold"
}
```

**Key Decision (from research):** Use 422 (not 404) for insufficient evidence. Request is valid, but semantically cannot be processed. Distinguishes user error from system error.

**400 Bad Request:** Invalid request format (malformed JSON, missing fields)
**404 Not Found:** Memory space doesn't exist
**500 Internal Server Error:** Unexpected failure (convergence exception, storage error)

### Actionable Error Messages (Nielsen, 1994)

Effective error messages provide:
1. **Diagnosis:** What went wrong
2. **Context:** Why it failed
3. **Action:** How to fix

**Example from research:**
```json
{
  "error": "InsufficientPattern",
  "message": "Pattern completion requires minimum 30% cue overlap",
  "details": {
    "cue_overlap": 0.15,
    "required_minimum": 0.30,
    "missing_critical_fields": ["when", "where"]
  },
  "suggestion": "Provide additional context fields or reduce ca1_threshold to 0.6"
}
```

User knows: **what** failed (insufficient cue), **why** (only 15% overlap), **how to fix** (add fields or adjust threshold).

### Error Recovery Patterns

**Retry-able Errors:**
- 500 Internal Server Error (transient failure)
- 503 Service Unavailable (overload)

**Non-retry-able Errors:**
- 400 Bad Request (client error - fix request)
- 422 Unprocessable Entity (add more evidence)
- 404 Not Found (memory space doesn't exist)

**Client Implementation Guide:**
```rust
match status {
    422 => {
        // Add more fields to partial_episode and retry
        partial.known_fields.insert("where", "kitchen");
        retry_request(partial)
    },
    500..=599 => {
        // Exponential backoff retry (server issue)
        exponential_backoff_retry()
    },
    _ => {
        // Permanent failure, don't retry
        return Err("Request failed permanently")
    }
}
```

### gRPC vs REST Trade-offs

**gRPC Advantages:**
- Binary protocol (Protocol Buffers): ~5x smaller than JSON
- HTTP/2 multiplexing: Multiple concurrent requests on single connection
- Streaming support: Progressive refinement, large result sets

**REST Advantages:**
- Human-readable (debugging, development)
- Browser-friendly (no special client needed)
- Wider tooling ecosystem (Postman, curl, Swagger UI)

**Engram Decision (from research):** Provide both. REST for development/debugging, gRPC for production performance.

**Performance Comparison (Target):**
- REST: 30-40KB JSON, 2ms serialization overhead
- gRPC: 6-8KB protobuf, 0.3ms serialization overhead
- Both: <2ms API overhead (Task 008 monitoring)

### Multi-Tenant API Design

**URL-based routing:**
```
/api/v1/spaces/{space_id}/complete
```
- Pro: RESTful, explicit in URL
- Con: Verbose, harder to route

**Header-based routing:**
```
X-Memory-Space: {space_id}
```
- Pro: Cleaner URLs, easier routing (O(1) hashmap lookup)
- Con: Not visible in URL (less RESTful)

**Engram Choice (from research):** Header-based (following M7 pattern). Matches existing memory space routing.

**Routing Performance:**
- Header extraction: O(1) hashmap lookup
- Space resolution: O(1) registry lookup
- Total overhead: <100μs
- Target: <2ms total API overhead (includes routing, serialization, validation)

### API Versioning

**URL Versioning:** /api/v1/complete, /api/v2/complete
- Pro: Explicit, easy to maintain multiple versions
- Con: URL proliferation

**Header Versioning:** Accept-Version: v1
- Pro: Cleaner URLs
- Con: Less discoverable

**Engram Choice (from research):** URL versioning (/api/v1/). Standard practice, explicit contract.

**Breaking Changes Require New Version:**
- Removing fields from response
- Changing field types
- Renaming fields

**Non-Breaking Changes Allowed:**
- Adding optional request fields
- Adding response fields
- Changing error message text

### OpenAPI/Swagger Documentation

**Schema-Driven Development Benefits:**
- Machine-readable API contract
- Automatic client generation (TypeScript, Python, Rust)
- Interactive documentation (Swagger UI)
- Contract testing (validate requests/responses)

**Example Schema (from research):**
```yaml
paths:
  /api/v1/complete:
    post:
      summary: Complete partial episode using pattern completion
      parameters:
        - name: X-Memory-Space
          in: header
          required: true
          schema:
            type: string
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/CompleteRequest'
      responses:
        '200':
          description: Successful completion
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/CompleteResponse'
        '422':
          description: Insufficient evidence for completion
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/ErrorResponse'
```

**Integration:** Task 027 (OpenAPI Spec) from milestone 0 roadmap implements utoipa annotations.

## API Specification

### HTTP Endpoint

```http
POST /api/v1/complete
Content-Type: application/json
X-Memory-Space: {space_id}

Request:
{
  "partial_episode": {
    "known_fields": {"what": "breakfast", "when": "morning"},
    "partial_embedding": [0.5, null, 0.3, ...],
    "cue_strength": 0.7,
    "temporal_context": ["morning_routine"]
  },
  "config": {
    "ca1_threshold": 0.7,
    "num_hypotheses": 3,
    "max_iterations": 7
  }
}

Response 200 OK:
{
  "completed_episode": {...},
  "completion_confidence": 0.82,
  "source_attribution": {
    "what": {"source": "recalled", "confidence": 0.95},
    "where": {"source": "reconstructed", "confidence": 0.68}
  },
  "alternative_hypotheses": [{...}, {...}],
  "metacognitive_confidence": 0.80,
  "reconstruction_stats": {
    "ca3_iterations": 5,
    "convergence_achieved": true,
    "pattern_sources": ["pattern_breakfast"],
    "plausibility_score": 0.82
  }
}

Response 422 Unprocessable Entity:
{
  "error": "InsufficientPattern",
  "message": "Pattern completion requires minimum 30% cue overlap",
  "details": {
    "cue_overlap": 0.15,
    "required_minimum": 0.30
  },
  "suggestion": "Provide additional context fields or reduce ca1_threshold"
}
```

### gRPC Method

```protobuf
message CompleteRequest {
  string memory_space_id = 1;
  PartialEpisode partial_episode = 2;
  CompletionConfig config = 3;
}

message CompleteResponse {
  Episode completed_episode = 1;
  float completion_confidence = 2;
  map<string, SourceAttribution> source_attribution = 3;
  repeated AlternativeHypothesis alternative_hypotheses = 4;
  float metacognitive_confidence = 5;
  ReconstructionStats reconstruction_stats = 6;
}

message SourceAttribution {
  MemorySource source = 1;
  float confidence = 2;
}

enum MemorySource {
  RECALLED = 0;
  RECONSTRUCTED = 1;
  IMAGINED = 2;
  CONSOLIDATED = 3;
}
```

## Implementation

```rust
// /engram-cli/src/handlers/complete.rs

use axum::{Extension, Json};
use engram_core::{MemorySpaceRegistry, completion::HippocampalCompletion};

pub async fn complete_handler(
    Extension(registry): Extension<Arc<MemorySpaceRegistry>>,
    headers: HeaderMap,
    Json(req): Json<CompleteRequest>,
) -> Result<Json<CompleteResponse>, ApiError> {
    // Extract memory space from header (M7 pattern)
    let space_id = extract_memory_space(&headers, &req)?;

    // Get memory space handle
    let space = registry.get_space(&space_id)
        .map_err(|e| ApiError::MemorySpaceNotFound(space_id.clone()))?;

    // Create completion engine
    let completion = HippocampalCompletion::new(req.config.unwrap_or_default());

    // Perform completion
    let result = completion.complete(&req.partial_episode)
        .map_err(|e| match e {
            CompletionError::InsufficientPattern => ApiError::InsufficientPattern {
                cue_overlap: compute_overlap(&req.partial_episode),
                required: 0.3,
            },
            CompletionError::ConvergenceFailed(iters) => ApiError::ConvergenceFailed(iters),
            CompletionError::LowConfidence(conf) => ApiError::LowConfidence(conf),
            _ => ApiError::Internal(e.to_string()),
        })?;

    Ok(Json(CompleteResponse::from(result)))
}
```

## Acceptance Criteria

1. **HTTP Endpoint:** POST /api/v1/complete returns 200 OK for valid requests, 422 for insufficient evidence
2. **gRPC Method:** Complete() RPC functional with identical semantics to HTTP
3. **Multi-Tenancy:** X-Memory-Space header correctly routes to isolated memory spaces (M7 integration)
4. **Error Handling:** All completion errors map to actionable HTTP responses with suggestions
5. **OpenAPI Docs:** /docs endpoint includes complete API specification with examples
6. **Performance:** API overhead <2ms (completion latency measured separately)

## Testing Strategy

**Unit Tests:** Request validation, error mapping, response serialization

**Integration Tests:** End-to-end HTTP/gRPC completion, multi-tenant isolation, error scenarios

**Contract Tests:** OpenAPI schema validation, gRPC contract compliance

## Success Criteria Validation

- [ ] HTTP endpoint functional and documented
- [ ] gRPC method functional with streaming support
- [ ] Multi-tenant routing correct (M7 integration)
- [ ] Error handling comprehensive with suggestions
- [ ] OpenAPI docs complete
- [ ] API overhead <2ms
