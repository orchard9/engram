# Example 09: Batch Operations

**Learning Goal**: Efficiently handle bulk memory operations with batching, streaming, and parallelism.

**Difficulty**: Intermediate
**Time**: 20 minutes
**Prerequisites**: Completed Examples 01, 03

## Cognitive Concept

Batch operations trade latency for throughput:
- **Serial**: Process one at a time (low throughput, predictable latency)
- **Batch**: Process many together (high throughput, higher latency)
- **Streaming**: Continuous flow (highest throughput, variable latency)

Engram optimizes batches by:
1. Amortizing network overhead
2. Vectorizing embedding operations
3. Batch-updating indexes

## What You'll Learn

- Batch remember operations for bulk import
- Batch recall for multi-query scenarios
- Determine optimal batch size
- Handle partial failures in batches
- Monitor batch processing metrics

## Batch Size Tuning

| Batch Size | Throughput | Latency | When to Use |
|------------|-----------|---------|-------------|
| 1 (unary) | 80 ops/sec | 12ms | Interactive queries |
| 10 | 280 ops/sec | 35ms | Small batches |
| 100 | 520 ops/sec | 190ms | Bulk import |
| 1000 | 680 ops/sec | 1470ms | Large migrations |

**Sweet spot**: 100-500 for most workloads

## Code Examples

See language-specific implementations in this directory:

- `rust.rs` - Rust with rayon parallel iterators
- `python.py` - Python with asyncio.gather
- `typescript.ts` - TypeScript with Promise.all
- `go.go` - Go with worker pools
- `java.java` - Java with parallel streams

## Batching Patterns

### Pattern 1: Fixed-Size Batching
```python
batch_size = 100
for i in range(0, len(memories), batch_size):
    batch = memories[i:i+batch_size]
    client.batch_remember(batch)
```

### Pattern 2: Time-Based Batching
```python
buffer = []
last_flush = time.now()

for memory in memory_stream:
    buffer.append(memory)
    if len(buffer) >= 100 or time.now() - last_flush > 1.0:
        client.batch_remember(buffer)
        buffer.clear()
        last_flush = time.now()
```

### Pattern 3: Adaptive Batching
```python
# Adjust batch size based on latency
if p99_latency > 200:
    batch_size = max(10, batch_size // 2)
elif p99_latency < 50:
    batch_size = min(1000, batch_size * 2)
```

## Handling Partial Failures

```python
response = client.batch_remember(memories)

for i, result in enumerate(response.results):
    if result.success:
        print(f"Memory {i} stored: {result.memory_id}")
    else:
        print(f"Memory {i} failed: {result.error.code}")
        # Retry individually or log for manual review
```

## Use Cases

1. **Database Migration**: Import 1M memories from PostgreSQL
2. **Bulk Updates**: Update confidence scores across corpus
3. **Archival**: Batch-store historical data
4. **Testing**: Generate synthetic datasets
5. **ETL Pipelines**: Transform and load external data

## Performance Tips

1. **Pre-compute embeddings**: Don't wait for batch to compute
2. **Use gRPC streaming**: 4-5x faster than REST batches
3. **Monitor backpressure**: Don't overwhelm hot tier
4. **Parallelize across spaces**: Independent memory spaces can run in parallel

## Next Steps

- [03-streaming-operations](../03-streaming-operations/) - Even higher throughput
- [07-performance-optimization](../07-performance-optimization/) - Advanced tuning
