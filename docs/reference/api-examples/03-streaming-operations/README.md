# Example 03: Streaming Operations

**Learning Goal**: Use bidirectional streaming for high-throughput memory operations.

**Difficulty**: Intermediate
**Time**: 20 minutes
**Prerequisites**: Completed Example 01, gRPC basics

## Cognitive Concept

Streaming operations mirror how biological memory works - continuous encoding and retrieval rather than discrete transactions:

```
Traditional: remember() -> wait -> remember() -> wait
Streaming:   remember_stream() --> continuous flow
```

Benefits:
- 4-5x higher throughput
- Automatic backpressure handling
- Real-time feedback on storage

## What You'll Learn

- Stream memories with `streaming_remember()`
- Watch consolidation with `dream()` streaming
- Handle backpressure gracefully
- Monitor real-time metrics

## Example Use Cases

- Bulk import: Load 1M memories from another database
- Real-time ingestion: Stream sensor data or logs
- Live monitoring: Watch consolidation create patterns
- Batch operations: Efficient bulk remember/recall

## Code Examples

See language-specific implementations in this directory:

- `rust.rs` - Rust with tokio streams
- `python.py` - Python async generators
- `typescript.ts` - TypeScript async iterators
- `go.go` - Go channels
- `java.java` - Java reactive streams

## Performance Comparison

| Operation | Unary (ops/sec) | Streaming (ops/sec) | Improvement |
|-----------|----------------|-------------------|-------------|
| Remember | 80 | 390 | 4.9x |
| Recall | 45 | 160 | 3.6x |

## Next Steps

- [05-error-handling](../05-error-handling/) - Robust error recovery
- [07-performance-optimization](../07-performance-optimization/) - Advanced tuning
