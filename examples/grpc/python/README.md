# Engram Python gRPC Client

Cognitive-friendly Python client for Engram memory operations.

## Quick Start (5 minutes)

```bash
# Install dependencies
pip install -r requirements.txt

# Generate protobuf files
python -m grpc_tools.protoc \
    -I../../proto \
    --python_out=. \
    --grpc_python_out=. \
    ../../proto/engram.proto

# Run example
python engram_client.py
```

## Progressive Learning Path

### Level 1: Essential Operations (5 min)
```python
# Store a memory
memory_id = await client.remember("Important fact", confidence=0.9)

# Retrieve memories
memories = await client.recall("related facts", limit=5)
```

### Level 2: Episodic Memory (15 min)
```python
# Record rich experiences
episode_id = await client.experience(
    what="Team discussion about architecture",
    when="Morning standup",
    where="Conference room",
    who=["Alice", "Bob"],
    why="Planning sprint goals"
)
```

### Level 3: Advanced Streaming (45 min)
```python
# Dream consolidation
async for event in client.dream(cycles=10):
    if event['type'] == 'insight':
        print(f"New insight: {event['description']}")
```

## Connection Management

```python
# With connection pooling
channel_options = [
    ('grpc.keepalive_time_ms', 10000),
    ('grpc.keepalive_timeout_ms', 5000),
    ('grpc.keepalive_permit_without_calls', True),
    ('grpc.http2.max_pings_without_data', 0),
]

channel = grpc.insecure_channel(
    'localhost:50051',
    options=channel_options
)
```

## Error Handling

```python
try:
    response = await client.remember(content)
except grpc.RpcError as e:
    if e.code() == grpc.StatusCode.UNAVAILABLE:
        # Retry with exponential backoff
        await asyncio.sleep(2 ** retry_count)
    elif e.code() == grpc.StatusCode.INVALID_ARGUMENT:
        # Log educational error message
        logger.error(f"Invalid input: {e.details()}")
```

## Performance Best Practices

1. **Connection Pooling**: Reuse channels across requests
2. **Streaming for Bulk**: Use streaming APIs for >10 operations
3. **Backpressure**: Respect flow control in bidirectional streams
4. **Timeout Strategy**: Set appropriate deadlines for operations

## Cognitive Design Principles

- Method names use memory metaphors (remember/recall vs store/query)
- Error messages teach memory concepts
- Progressive complexity from basic to advanced
- Confidence scores make uncertainty explicit