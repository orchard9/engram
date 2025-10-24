# APIs That Remember: Exposing Pattern Completion Over HTTP and gRPC

The hardest part of building a memory system isn't the algorithms. It's the API.

You've implemented hippocampal CA3 dynamics, semantic pattern retrieval, hierarchical evidence integration, source attribution, and confidence calibration. Sub-20ms completion latency. >85% reconstruction accuracy. Biologically validated.

Now you need to expose this complexity through a clean API that developers can actually use.

Task 007 implements production-ready HTTP and gRPC endpoints for pattern completion. Multi-tenant routing. Actionable error messages. OpenAPI documentation. The bridge between cognitive neuroscience and production systems.

## The API Design Problem

Pattern completion has rich outputs:
- Completed episode (reconstructed fields)
- Completion confidence (calibrated probability)
- Source attribution per field (Recalled/Reconstructed/Imagined/Consolidated)
- Alternative hypotheses (competing completions)
- Metacognitive confidence (confidence in confidence)
- Reconstruction stats (CA3 iterations, convergence, pattern sources, plausibility)

Traditional database queries return rows. Memory completion returns epistemology.

How do you expose this over HTTP without overwhelming users?

## REST API Design: Balancing Simplicity and Completeness

**Minimal Request:**
```json
POST /api/v1/complete
X-Memory-Space: user_123

{
  "partial_episode": {
    "known_fields": {
      "what": "breakfast",
      "when": "morning"
    }
  }
}
```

Just the essential: what you know. Everything else defaults.

**Complete Request:**
```json
{
  "partial_episode": {
    "known_fields": {"what": "breakfast", "when": "morning"},
    "partial_embedding": [0.5, null, 0.3, ...],
    "cue_strength": 0.7,
    "temporal_context": ["morning_routine", "kitchen"]
  },
  "config": {
    "ca1_threshold": 0.7,
    "num_hypotheses": 3,
    "max_iterations": 7,
    "pattern_weight": 0.4
  }
}
```

Full control for power users.

**Key Design Choice:** Optional complexity. Simple by default, configurable when needed.

## Response Structure: Progressive Disclosure

**Core Response:**
```json
{
  "completed_episode": {
    "id": "ep_completed_xyz",
    "content": "breakfast with coffee at kitchen table",
    "embedding": [0.52, 0.31, ...],
    "timestamp": "2025-10-23T08:30:00Z"
  },
  "completion_confidence": 0.82
}
```

Enough for most use cases. Episode + confidence.

**Extended Response:**
```json
{
  "completed_episode": {...},
  "completion_confidence": 0.82,
  "source_attribution": {
    "what": {"source": "recalled", "confidence": 0.95},
    "when": {"source": "recalled", "confidence": 0.95},
    "where": {"source": "reconstructed", "confidence": 0.68},
    "details": {"source": "consolidated", "confidence": 0.75}
  },
  "alternative_hypotheses": [
    {"episode": {...}, "confidence": 0.78},
    {"episode": {...}, "confidence": 0.65}
  ],
  "metacognitive_confidence": 0.80,
  "reconstruction_stats": {
    "ca3_iterations": 5,
    "convergence_achieved": true,
    "pattern_sources": ["pattern_breakfast_routine", "pattern_morning_meals"],
    "plausibility_score": 0.82
  }
}
```

Full transparency for debugging, auditing, research.

**Design Principle:** Always include core fields. Include extended fields when config.verbose = true or for debugging.

## Error Handling: Making Failures Actionable

Bad error message:
```json
{
  "error": "Completion failed"
}
```

User thinks: "Why? What do I do?"

Good error message:
```json
HTTP 422 Unprocessable Entity

{
  "error": "InsufficientPattern",
  "message": "Pattern completion requires minimum 30% cue overlap",
  "details": {
    "cue_overlap": 0.15,
    "required_minimum": 0.30,
    "known_fields": ["what"],
    "missing_critical_fields": ["when", "where"]
  },
  "suggestion": "Provide additional context fields (when, where) or reduce ca1_threshold to 0.6 for lower-confidence completions"
}
```

User knows:
- What failed (insufficient cue overlap)
- Why (only 15% vs 30% required)
- How to fix (add fields or adjust threshold)

**Error Categories:**

**Client Errors (4xx):**
- 400 Bad Request: Invalid JSON, missing required fields
- 404 Not Found: Memory space doesn't exist
- 422 Unprocessable Entity: Valid format, insufficient evidence

**Server Errors (5xx):**
- 500 Internal Server Error: CA3 convergence failure, unexpected error
- 503 Service Unavailable: Overloaded, rate limited

**Design Principle:** Use 422 for semantic failures (insufficient evidence). Reserve 400 for syntactic failures (malformed request).

## Multi-Tenant Routing: Memory Space Isolation

Milestone 7 implemented MemorySpaceRegistry for multi-tenancy. Task 007 integrates it.

**Header-Based Routing:**
```http
POST /api/v1/complete
X-Memory-Space: user_alice
```

Why header vs URL path?

**URL-based:** /api/v1/spaces/user_alice/complete
- Pro: RESTful, explicit
- Con: Verbose, harder to route

**Header-based:** X-Memory-Space: user_alice
- Pro: Clean URLs, easy routing, matches M7 pattern
- Con: Less discoverable

Engram chose headers. Consistency with existing memory space routing. Performance (O(1) hashmap lookup).

**Implementation:**
```rust
pub async fn complete_handler(
    Extension(registry): Extension<Arc<MemorySpaceRegistry>>,
    headers: HeaderMap,
    Json(req): Json<CompleteRequest>,
) -> Result<Json<CompleteResponse>, ApiError> {
    // Extract memory space from header
    let space_id = headers
        .get("X-Memory-Space")
        .ok_or(ApiError::MissingMemorySpace)?
        .to_str()
        .map_err(|_| ApiError::InvalidMemorySpace)?;

    // Get isolated memory space
    let space = registry
        .get_space(&space_id)
        .map_err(|_| ApiError::MemorySpaceNotFound(space_id.to_string()))?;

    // Completion operates on isolated space
    let completion = HippocampalCompletion::new_for_space(space);
    // ...
}
```

**Isolation Guarantee:** user_alice's completions never access user_bob's episodes. Memory space enforced at API layer.

## gRPC Implementation: Performance + Streaming

REST is great for development. gRPC is better for production.

**Protocol Buffers:**
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

enum MemorySource {
  RECALLED = 0;
  RECONSTRUCTED = 1;
  IMAGINED = 2;
  CONSOLIDATED = 3;
}
```

**Why gRPC?**

1. **Binary Protocol:** Protocol Buffers ~5x smaller than JSON. Faster serialization.
2. **HTTP/2 Multiplexing:** Multiple requests on single connection. Lower latency.
3. **Streaming:** Progressive refinement of completions.
4. **Type Safety:** Generated clients guarantee type correctness.

**Streaming Completion:**
```protobuf
service EngramService {
  rpc CompleteStream(CompleteRequest) returns (stream CompleteProgress);
}

message CompleteProgress {
  oneof event {
    PatternRetrievalComplete pattern_retrieval = 1;
    CA3ConvergenceProgress ca3_progress = 2;
    CompletionPartial partial_result = 3;
    CompletionFinal final_result = 4;
  }
}
```

Client receives progress updates:
1. Pattern retrieval completed (10 patterns found)
2. CA3 convergence progress (iteration 3/7, energy -4.2)
3. Partial completion (60% fields complete)
4. Final completion (100% complete, confidence 0.85)

**Use Cases:**
- Long-running completions (complex patterns)
- Real-time feedback (UI progress bars)
- Partial result utilization (show what's ready)

## OpenAPI Documentation: Self-Describing API

Machine-readable API specification:

```yaml
openapi: 3.0.0
info:
  title: Engram Pattern Completion API
  version: 1.0.0
  description: Reconstruct missing episode details using hippocampal pattern completion

paths:
  /api/v1/complete:
    post:
      summary: Complete partial episode
      operationId: completeEpisode
      parameters:
        - name: X-Memory-Space
          in: header
          required: true
          description: Isolated memory space identifier
          schema:
            type: string
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/CompleteRequest'
            examples:
              minimal:
                value:
                  partial_episode:
                    known_fields:
                      what: "breakfast"
                      when: "morning"
      responses:
        '200':
          description: Successful completion
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/CompleteResponse'
        '422':
          description: Insufficient evidence
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/Error'

components:
  schemas:
    CompleteRequest: {...}
    CompleteResponse: {...}
    Error: {...}
```

**Benefits:**
- Swagger UI at /docs (interactive testing)
- Client generation (OpenAPI Generator)
- Contract testing (ensure implementation matches spec)
- API evolution tracking (diff specs between versions)

## Performance: <2ms API Overhead

Pattern completion itself: ~14ms (CA3 + integration).
Target total API latency: <16ms (2ms overhead budget).

**Overhead Sources:**
- Request deserialization: ~500μs (serde_json)
- Memory space routing: ~100μs (hashmap lookup)
- Response serialization: ~800μs (nested structures)
- Axum framework: ~300μs (routing, middleware)
- Total: ~1.7ms

Within budget. 10% overhead on top of completion.

**Optimization:** Pre-allocate serialization buffers. Use zero-copy where possible. Profile hot paths.

## Multi-Version Support: /api/v1 and /api/v2

APIs evolve. Breaking changes require new versions.

**v1 Response (current):**
```json
{
  "completed_episode": {...},
  "confidence": 0.82
}
```

**v2 Response (future - breaking):**
```json
{
  "episode": {...},  // Renamed from completed_episode
  "completion": {    // Nested confidence structure
    "value": 0.82,
    "calibrated": true,
    "method": "multi_factor"
  }
}
```

**Routing:**
- /api/v1/complete → v1 handler
- /api/v2/complete → v2 handler

Both versions run simultaneously. Clients migrate at their own pace. Deprecation warnings in v1 headers.

## Testing Strategy

**Unit Tests:** Request validation, error mapping, response serialization
**Integration Tests:** End-to-end HTTP/gRPC, multi-tenant isolation
**Contract Tests:** OpenAPI schema compliance, gRPC protocol correctness
**Load Tests:** Concurrent requests, throughput limits, latency under load

**Example Integration Test:**
```rust
#[tokio::test]
async fn test_complete_api_success() {
    let registry = create_test_registry();
    let app = create_app(registry);

    let request = CompleteRequest {
        partial_episode: PartialEpisode {
            known_fields: hashmap!{"what" => "breakfast"},
            ..Default::default()
        },
        config: None,
    };

    let response = app
        .post("/api/v1/complete")
        .header("X-Memory-Space", "test_space")
        .json(&request)
        .send()
        .await;

    assert_eq!(response.status(), 200);
    let body: CompleteResponse = response.json().await;
    assert!(body.completion_confidence > 0.7);
    assert!(body.source_attribution.contains_key("what"));
}
```

## Conclusion

Pattern completion API transforms cognitive neuroscience into production system. REST for development, gRPC for performance. Multi-tenant routing for isolation. Actionable errors for debugging. OpenAPI for documentation.

The result: A memory system developers can actually use.

Next: Metrics and observability (Task 008) to monitor production completions.

---

**Citations:**
- Fielding, R. T. (2000). Architectural styles and the design of network-based software architectures
- RFC 7231: HTTP/1.1 Semantics and Content
- OpenAPI Specification 3.0
- gRPC Design Guide
