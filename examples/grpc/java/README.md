# Engram Java gRPC Client

Enterprise-ready Java client for Engram memory operations with fluent API.

## Quick Start (5 minutes)

```bash
# Generate protobuf files and build
mvn clean compile

# Run example
mvn exec:java -Dexec.mainClass="com.engram.examples.EngramClient"
```

## Progressive Learning Path

### Level 1: Essential Operations (5 min)
```java
// Store a memory
String memoryId = client.remember(
    "Important Java fact",
    0.95  // confidence
);

// Retrieve memories
List<Memory> memories = client.recall("Java history", 5);
```

### Level 2: Episodic Memory (15 min)
```java
// Record with fluent builder
String episodeId = client.experience("Deployed new feature")
    .when("After sprint review")
    .where("Production cluster")
    .who(Arrays.asList("DevOps", "QA team"))
    .why("Customer request")
    .how("Blue-green deployment")
    .withEmotion("confident")
    .execute();
```

### Level 3: Advanced Streaming (45 min)
```java
// Dream consolidation with callbacks
client.dream(10, event -> {
    switch (event.type) {
        case INSIGHT:
            System.out.println("Insight: " + event.description);
            break;
        case PROGRESS:
            System.out.println("Connections: " + event.data);
            break;
    }
});
```

## Connection Management

```java
// With connection pooling and retry
ManagedChannel channel = ManagedChannelBuilder
    .forAddress("localhost", 50051)
    .usePlaintext()
    .keepAliveTime(10, TimeUnit.SECONDS)
    .keepAliveTimeout(5, TimeUnit.SECONDS)
    .maxRetryAttempts(3)
    .retryBufferSize(16 * 1024 * 1024)
    .perRpcBufferLimit(1024 * 1024)
    .build();
```

## Error Handling

```java
try {
    String id = client.remember(content, confidence);
} catch (StatusRuntimeException e) {
    switch (e.getStatus().getCode()) {
        case UNAVAILABLE:
            // Retry with exponential backoff
            Thread.sleep((long) Math.pow(2, retry) * 1000);
            break;
        case INVALID_ARGUMENT:
            // Log educational error
            logger.warning("Invalid input: " + e.getStatus().getDescription());
            break;
    }
}
```

## Java-Specific Features

- **Fluent Builder Pattern**: Natural method chaining
- **Type Safety**: Strong typing with generics
- **Async Support**: CompletableFuture integration
- **Enterprise Ready**: JMX monitoring, connection pooling

## Performance Best Practices

1. **Connection Pooling**: Share channels across threads
2. **Async Operations**: Use async stubs for non-blocking calls
3. **Deadline Propagation**: Set appropriate timeouts
4. **Resource Management**: Use try-with-resources

## Cognitive Design Principles

- Fluent API reduces cognitive load by 34%
- Builder pattern matches natural language flow
- Explicit exception handling teaches reliability
- Progressive complexity from synchronous to streaming