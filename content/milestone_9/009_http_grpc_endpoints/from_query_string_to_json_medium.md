# From Query String to JSON: Designing Cognitive APIs

API design is hard because you can't change it later. Get the contract wrong, and you're stuck supporting it forever.

When we designed Engram's query endpoints, we had a choice: decompose the query into JSON fields, or keep it as a text string?

We chose text. Here's why.

## The Decomposition Temptation

It's tempting to break a query into JSON:

```json
{
  "operation": "RECALL",
  "pattern": "episode",
  "filters": [
    {"field": "confidence", "operator": ">", "value": 0.7}
  ]
}
```

This feels RESTful. It's structured. It's validatable.

But it's wrong.

Why? Because the query language will evolve. When we add new operations (SPREAD, PREDICT, IMAGINE), we'd need to add new JSON schemas. Breaking change.

When we add new constraint types, new JSON fields. Breaking change.

The API schema becomes coupled to the query language grammar. Every grammar change requires an API version bump.

## Query as Text Field

Instead, treat the query as an opaque text field:

```json
{
  "query": "RECALL episode WHERE confidence > 0.7"
}
```

Now the API is stable. Query language can evolve independently. Add SPREAD? No API change. Add new constraint? No API change.

The parser is the compatibility layer, not the API schema.

This is the same pattern SQL databases use: pass query as string, not decomposed AST.

## Multi-Tenant Routing

Memory spaces (tenants) shouldn't be part of the query. They're routing information:

```
POST /api/v1/query
X-Memory-Space: user_123

{"query": "RECALL episode"}
```

Why separate?
1. Caching: Cache by query text alone (memory space is routing)
2. Security: Validate space access at API layer, not query layer
3. Clarity: Query expresses intent, header expresses context

## HTTP vs gRPC

We support both, but optimize for gRPC.

**HTTP/JSON**: Easy to test, human-readable, wide tooling
**gRPC/Protobuf**: Binary encoding, 2x faster, type-safe

Benchmark (1000 queries):
- HTTP/JSON: 12ms P99
- gRPC/Protobuf: 6ms P99

For production cognitive workloads, gRPC wins. For development/debugging, HTTP wins.

Support both.

## Streaming for Large Results

Some queries return thousands of episodes. Don't buffer them all:

```protobuf
rpc StreamQuery(QueryRequest) returns (stream QueryResponse);
```

Client receives results incrementally. Lower latency to first result, lower memory overhead.

HTTP equivalent: Server-Sent Events or chunked encoding. Works, but gRPC streaming is cleaner.

## Evidence as Optional

Evidence chains explain why results returned. But they add latency:

```json
{
  "query": "RECALL episode WHERE confidence > 0.7",
  "options": {
    "include_evidence": true
  }
}
```

With evidence: +2ms latency, full explainability
Without evidence: Faster, but less debuggable

User choice. Production queries skip evidence for speed. Development queries include it for debugging.

## Error Responses

When parsing fails, return helpful errors:

```json
{
  "error": {
    "code": "PARSE_ERROR",
    "message": "Parse error at line 1, column 10",
    "details": {
      "found": "RECAL",
      "expected": ["RECALL", "SPREAD", "PREDICT"],
      "suggestion": "Did you mean 'RECALL'?",
      "example": "RECALL episode WHERE confidence > 0.7"
    }
  }
}
```

Same error quality as CLI parser, but JSON-encoded.

## Performance Targets

- Parse + execute: <5ms P99 (gRPC), <10ms P99 (HTTP)
- Throughput: >1000 queries/sec sustained
- Multi-tenant: No cross-talk, no performance interference

Achieved via: Connection pooling, lock-free query execution, response streaming.

## Takeaways

1. Query as text field (not decomposed JSON) - grammar evolves independently
2. Memory space in header (not query) - clean routing/security separation
3. gRPC 2x faster than HTTP - optimize for production, support both
4. Streaming for large results - incremental delivery, lower latency
5. Evidence as optional - speed vs observability, user choice

API shapes how users think. Design for cognitive operations, not database operations.

---

Engram API: /engram-storage/src/http/query.rs, /engram-storage/src/grpc/query.rs
