# Task 009: HTTP/gRPC Query Endpoints

**Status**: Pending
**Duration**: 1.5 days
**Dependencies**: Task 005, 006, 007
**Owner**: TBD

---

## Objective

Add query endpoints to existing HTTP/gRPC services. Support query text input, return ProbabilisticQueryResult with JSON/Protobuf serialization.

---

## Files

1. `engram-proto/engram.proto` - Add QueryRequest/QueryResponse messages
2. `engram-storage/src/http/query.rs` - POST /api/v1/query endpoint
3. `engram-storage/src/grpc/query.rs` - ExecuteQuery RPC

---

## Protocol Definitions

```protobuf
message QueryRequest {
    string query_text = 1;
    string memory_space_id = 2;
}

message QueryResponse {
    repeated Episode episodes = 1;
    repeated float confidences = 2;
    ConfidenceInterval aggregate_confidence = 3;
}
```

---

## Acceptance Criteria

- [ ] POST /api/v1/query returns JSON results
- [ ] gRPC ExecuteQuery returns protobuf results
- [ ] Multi-tenant routing via X-Memory-Space header
- [ ] OpenAPI spec updated
- [ ] Integration tests pass
