# Example 07: Performance Optimization

**Learning Goal**: Optimize throughput and latency through batching, connection pooling, and caching.

**Difficulty**: Advanced
**Time**: 25 minutes
**Prerequisites**: Completed Examples 01, 03

## Cognitive Concept

Biological memory is highly optimized - your brain doesn't make one synapse fire at a time. Similarly, Engram performance comes from:

1. **Batching**: Group related operations
2. **Streaming**: Continuous flow vs discrete requests
3. **Caching**: Keep hot memories in RAM
4. **Parallelism**: Leverage multiple cores

## What You'll Learn

- Batch operations for 4-5x throughput
- Connection pooling and keepalive
- Client-side caching with confidence decay
- Spreading activation tuning
- Resource limits and backpressure

## Performance Patterns

### Pattern 1: Batch Instead of Loop
```python
# Slow: 100 requests
for memory in memories:
    client.remember(memory)  # 100 * 12ms = 1200ms

# Fast: 1 batch request
client.batch_remember(memories)  # 180ms
```

### Pattern 2: Connection Pooling
```rust
// Poor: new connection per request
for memory in memories {
    let client = EngramClient::connect("localhost:50051").await?;
    client.remember(memory).await?;
}

// Good: reuse connection pool
let client = EngramClient::connect("localhost:50051")
    .with_pool_size(10)
    .await?;
```

### Pattern 3: Client-Side Caching
```typescript
// Cache high-confidence, consolidated memories
if (memory.confidence > 0.9 && memory.state === 'CONSOLIDATED') {
    cache.set(cue_hash, memory, ttl=3600);
}
```

## Code Examples

See language-specific implementations in this directory:

- `rust.rs` - Rust with connection pooling
- `python.py` - Python with asyncio batching
- `typescript.ts` - TypeScript with LRU cache
- `go.go` - Go with worker pools
- `java.java` - Java with CompletableFuture batching

## Benchmarking Results

| Optimization | Before | After | Improvement |
|--------------|--------|-------|-------------|
| Batching (100 ops) | 1200ms | 180ms | 6.7x |
| Connection pooling | 45ms/op | 12ms/op | 3.8x |
| Client caching (90% hit rate) | 35ms | 5ms | 7x |
| Streaming vs unary | 80 ops/sec | 390 ops/sec | 4.9x |

## Tuning Parameters

**For Low Latency (P99 <10ms)**:
- `max_hops = 3` (reduce depth)
- `threshold = 0.2` (stricter filtering)
- `hot_capacity = 1M` (more RAM caching)

**For High Throughput (>1000 ops/sec)**:
- Use gRPC streaming
- Batch size = 100-500
- Connection pool = 2x CPU cores
- Warm tier on NVMe SSD

## Next Steps

- [Performance Tuning Guide](/operations/performance-tuning.md) - Advanced techniques
- [Monitoring](/operations/monitoring.md) - Track performance metrics
