# Research: From Query String to JSON - API Surface Design

## Key Findings

### 1. HTTP vs gRPC Trade-offs

**HTTP/REST**:
- Easy to test (curl, Postman)
- Human-readable (JSON)
- Wide tooling support
- Higher latency (text encoding)

**gRPC**:
- Binary encoding (faster)
- Streaming support (real-time)
- Type-safe (protobuf)
- Harder to debug

**Decision**: Support both, optimize for gRPC performance.

### 2. Query Endpoint Design

```protobuf
message QueryRequest {
    string query_text = 1;
    string memory_space_id = 2;
    QueryOptions options = 3;
}

message QueryResponse {
    repeated Episode episodes = 1;
    repeated float confidences = 2;
    ConfidenceInterval aggregate_confidence = 3;
    repeated Evidence evidence_chain = 4;
}
```

**HTTP Equivalent**:
```
POST /api/v1/query
X-Memory-Space: user_123

{
  "query": "RECALL episode WHERE confidence > 0.7",
  "options": {
    "include_evidence": true,
    "max_results": 10
  }
}
```

### 3. Multi-Tenant Routing

Use header-based routing for tenant isolation:
- `X-Memory-Space: user_123`
- Server validates space exists
- Routes query to correct memory partition
- Returns 404 if space invalid

Benefits: No changes to query syntax, clean separation.

### 4. Streaming Responses

For large result sets, use gRPC streaming:
```protobuf
rpc StreamQuery(QueryRequest) returns (stream QueryResponse);
```

Client receives results incrementally, doesn't wait for full set.

### 5. Performance Targets

- HTTP: <10ms P99 latency (parse + execute)
- gRPC: <5ms P99 latency
- Throughput: >1000 queries/sec sustained

Achieved via: Connection pooling, response caching, lock-free execution.

## References

1. gRPC Performance Best Practices (Google)
2. REST API Design (Richardson Maturity Model)
3. Protobuf Encoding Efficiency (Protocol Buffers documentation)
