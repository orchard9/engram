# Engram TypeScript gRPC Client

Cognitive-friendly TypeScript client with method chaining for Engram memory operations.

## Quick Start (5 minutes)

```bash
# Install dependencies
npm install

# Generate protobuf files
npm run generate

# Run example
npm run example
```

## Progressive Learning Path

### Level 1: Essential Operations (5 min)
```typescript
// Store with method chaining
const memoryId = await client
    .remember("Important fact")
    .withConfidence(0.9)
    .execute();

// Retrieve with options
const memories = await client
    .recall("related facts")
    .limit(5)
    .withTraces()
    .execute();
```

### Level 2: Episodic Memory (15 min)
```typescript
// Record rich experiences with fluent API
const episodeId = await client
    .experience("Team architecture discussion")
    .when("Morning standup")
    .where("Conference room")
    .who(["Alice", "Bob"])
    .why("Planning sprint goals")
    .withEmotion("engaged")
    .execute();
```

### Level 3: Advanced Streaming (45 min)
```typescript
// Dream consolidation with event handlers
await client.dream(10, {
    onReplay: (memories, narrative) => {
        console.log(`Replaying: ${narrative}`);
    },
    onInsight: (insight) => {
        console.log(`New insight: ${insight.description}`);
    },
    onProgress: (progress) => {
        console.log(`Created ${progress.connections} connections`);
    }
});
```

## Connection Management

```typescript
// With keepalive and retry
const channelOptions = {
    'grpc.keepalive_time_ms': 10000,
    'grpc.keepalive_timeout_ms': 5000,
    'grpc.keepalive_permit_without_calls': 1,
    'grpc.http2.max_pings_without_data': 0,
};

const client = new engramProto.EngramService(
    'localhost:50051',
    grpc.credentials.createInsecure(),
    channelOptions
);
```

## Error Handling

```typescript
try {
    await client.remember(content).execute();
} catch (error) {
    if (error.code === grpc.status.UNAVAILABLE) {
        // Retry with exponential backoff
        await delay(Math.pow(2, retryCount) * 1000);
    } else if (error.code === grpc.status.INVALID_ARGUMENT) {
        // Log educational error message
        console.error(`Invalid input: ${error.details}`);
    }
}
```

## TypeScript Features

- **Type Safety**: Full TypeScript support with generated types
- **Method Chaining**: Fluent API for natural code flow
- **Async/Await**: Modern async patterns throughout
- **Event Handlers**: Reactive programming for streams

## Performance Best Practices

1. **Connection Pooling**: Reuse client instances
2. **Streaming for Bulk**: Use streaming APIs for >10 operations
3. **Backpressure**: Handle flow control in streams
4. **Timeout Strategy**: Set appropriate deadlines

## Cognitive Design Principles

- Method chaining matches natural thought flow
- Progressive disclosure of complexity
- Error messages that teach concepts
- Event-driven patterns for real-time updates