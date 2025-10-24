# Pattern Completion API: Research Foundations

## RESTful API Design Principles

### Fielding's REST Constraints (2000)
Representational State Transfer architectural style for distributed systems:
- Client-server separation
- Stateless communication
- Cacheable responses
- Uniform interface
- Layered system

**Application to Completion API:**
- POST /api/v1/complete is stateless (each request independent)
- Response includes cache hints (completion can be cached per partial cue)
- Uniform JSON interface for requests/responses

### HTTP Status Codes (RFC 7231)
**200 OK:** Successful completion
**422 Unprocessable Entity:** Valid request format, but insufficient evidence for completion
**400 Bad Request:** Invalid request format
**404 Not Found:** Memory space doesn't exist
**500 Internal Server Error:** Unexpected failure

**Key Decision:** Use 422 (not 404) for insufficient evidence. Request is valid, but semantically cannot be processed.

## gRPC vs REST Trade-offs

### Performance Characteristics
**gRPC Advantages:**
- Binary protocol (Protocol Buffers): ~5x smaller than JSON
- HTTP/2 multiplexing: Multiple concurrent requests on single connection
- Streaming support: Progressive refinement, large result sets

**REST Advantages:**
- Human-readable (debugging, development)
- Browser-friendly (no special client needed)
- Wider tooling ecosystem

**Engram Decision:** Provide both. REST for development/debugging, gRPC for production performance.

## Multi-Tenant API Design

### Tenant Isolation Strategies
**URL-based:** /api/v1/spaces/{space_id}/complete
- Pro: RESTful, explicit in URL
- Con: Verbose, harder to route

**Header-based:** X-Memory-Space: {space_id}
- Pro: Cleaner URLs, easier routing
- Con: Not visible in URL (less RESTful)

**Engram Choice:** Header-based (following M7 pattern). Matches existing memory space routing.

### Tenant Routing Performance
Header extraction: O(1) hashmap lookup.
Space resolution: O(1) registry lookup.
Total overhead: <100μs.

Target: <2ms total API overhead (includes routing, serialization, validation).

## Error Handling Best Practices

### Actionable Error Messages (Nielsen 1994)
Effective error messages:
1. State what went wrong (diagnosis)
2. Explain why (context)
3. Suggest how to fix (actionable)

**Example:**
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

User knows: what failed (insufficient cue), why (only 15% overlap), how to fix (add fields or adjust threshold).

### Error Recovery Patterns
**Retry-able Errors:** 500 Internal Server Error, 503 Service Unavailable
**Non-retry-able:** 400 Bad Request, 422 Unprocessable Entity, 404 Not Found

**Client Implementation:**
```
if status == 422:
    # Add more fields and retry
elif status >= 500:
    # Exponential backoff retry
else:
    # Permanent failure, don't retry
```

## API Versioning

### Semantic Versioning for APIs
**URL Versioning:** /api/v1/complete, /api/v2/complete
- Pro: Explicit, easy to maintain multiple versions
- Con: URL proliferation

**Header Versioning:** Accept-Version: v1
- Pro: Cleaner URLs
- Con: Less discoverable

**Engram Choice:** URL versioning (/api/v1/). Standard practice, explicit contract.

### Backward Compatibility
Breaking changes require new version:
- Removing fields from response
- Changing field types
- Renaming fields

Non-breaking changes allowed in same version:
- Adding optional request fields
- Adding response fields
- Changing error message text

## OpenAPI/Swagger Documentation

### Schema-Driven Development
**Benefits:**
- Machine-readable API contract
- Automatic client generation
- Interactive documentation (Swagger UI)
- Contract testing

**Example Schema:**
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
```

## References

1. Fielding, R. T. (2000). Architectural styles and the design of network-based software architectures. Doctoral dissertation, UC Irvine.
2. RFC 7231: Hypertext Transfer Protocol (HTTP/1.1): Semantics and Content
3. Nielsen, J. (1994). Usability engineering. Morgan Kaufmann.
4. OpenAPI Specification 3.0: https://swagger.io/specification/
5. gRPC Design Guide: https://grpc.io/docs/guides/
