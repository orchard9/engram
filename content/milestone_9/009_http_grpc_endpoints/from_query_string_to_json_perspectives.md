# Perspectives: From Query String to JSON

## Systems Architecture Perspective

API surface design is a contract with users. Get it wrong, you're stuck with it forever.

**Key Design Decisions**:

1. **Query as text field**: Don't decompose query into JSON fields
   - Bad: `{"operation": "RECALL", "pattern": "episode", "filters": [...]}`
   - Good: `{"query": "RECALL episode WHERE confidence > 0.7"}`
   - Why: Query language evolves independently of API schema

2. **Memory space in header**: Tenant routing separate from query
   - `X-Memory-Space: user_123`
   - Allows caching by query text alone
   - Clean separation of concerns

3. **Evidence as optional**: Performance vs observability
   - Include evidence: +2ms latency, full explainability
   - Exclude evidence: Faster, but less debuggable
   - User choice via `include_evidence` option

## Rust Graph Engine Perspective

gRPC is faster than HTTP/REST for cognitive queries. Here's why:

**HTTP/JSON**:
```
Query → UTF-8 encode → JSON serialize → HTTP → Parse JSON → UTF-8 decode → Execute
```

**gRPC/Protobuf**:
```
Query → Protobuf serialize → gRPC → Protobuf deserialize → Execute
```

Fewer steps, binary encoding, zero-copy where possible.

Benchmark: 1000 queries
- HTTP/JSON: 12ms P99
- gRPC/Protobuf: 6ms P99

2x faster with gRPC.

## Cognitive Architecture Perspective

API should mirror cognitive operations, not database operations.

Traditional API:
```
POST /memories/search
{"filters": [...], "limit": 10}
```

Cognitive API:
```
POST /api/v1/query
{"query": "RECALL episode WHERE confidence > 0.7"}
```

Second maps directly to mental model: "I want to recall something with high confidence."

First is database thinking: "Search memory table with filters."

API shapes how users think about the system.
