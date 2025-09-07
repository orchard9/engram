# Engram Go gRPC Client

Cognitive-friendly Go client for Engram memory operations with explicit error handling.

## Quick Start (5 minutes)

```bash
# Install dependencies
go mod download

# Generate protobuf files
protoc --go_out=. --go_opt=paths=source_relative \
       --go-grpc_out=. --go-grpc_opt=paths=source_relative \
       -I ../../proto ../../proto/engram.proto

# Run example
go run engram_client.go
```

## Progressive Learning Path

### Level 1: Essential Operations (5 min)
```go
// Store a memory
memoryID, err := client.Remember(
    "Important fact about Go",
    0.95, // confidence
)

// Retrieve memories
memories, err := client.Recall("Go concurrency", 5)
```

### Level 2: Episodic Memory (15 min)
```go
// Record experience with builder pattern
episodeID, err := client.Experience("Fixed critical bug").
    When("During incident response").
    Where("Production system").
    Who([]string{"SRE team", "Tech lead"}).
    Why("System outage").
    How("Rolled back deployment").
    WithEmotion("stressed").
    Execute()
```

### Level 3: Advanced Streaming (45 min)
```go
// Dream consolidation with event handling
err := client.Dream(10, func(event DreamEvent) {
    switch event.Type {
    case "insight":
        fmt.Printf("New insight: %s\n", event.Description)
    case "progress":
        data := event.Data.(map[string]interface{})
        fmt.Printf("Connections: %v\n", data["connections"])
    }
})
```

## Connection Management

```go
// With keepalive and retry interceptor
opts := []grpc.DialOption{
    grpc.WithKeepaliveParams(keepalive.ClientParameters{
        Time:                10 * time.Second,
        Timeout:             5 * time.Second,
        PermitWithoutStream: true,
    }),
    grpc.WithUnaryInterceptor(retryInterceptor),
    grpc.WithStreamInterceptor(streamRetryInterceptor),
}

conn, err := grpc.Dial(address, opts...)
```

## Error Handling

```go
resp, err := client.Remember(content, confidence)
if err != nil {
    if st, ok := status.FromError(err); ok {
        switch st.Code() {
        case codes.Unavailable:
            // Retry with exponential backoff
            time.Sleep(time.Duration(math.Pow(2, float64(retry))) * time.Second)
        case codes.InvalidArgument:
            // Log educational error
            log.Printf("Invalid input: %s", st.Message())
        }
    }
}
```

## Go-Specific Features

- **Explicit Error Handling**: Idiomatic Go error patterns
- **Builder Pattern**: Fluent API for complex operations
- **Context Support**: Proper cancellation and timeouts
- **Goroutine Safety**: Thread-safe client operations

## Performance Best Practices

1. **Connection Pooling**: Reuse client connections
2. **Context Deadlines**: Set appropriate timeouts
3. **Streaming for Bulk**: Use streams for >10 operations
4. **Error Recovery**: Implement circuit breakers

## Cognitive Design Principles

- Explicit error handling teaches reliability patterns
- Builder pattern matches incremental thought process
- Event-driven streaming for real-time feedback
- Educational error messages explain memory concepts