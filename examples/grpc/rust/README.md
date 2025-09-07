# Engram Rust gRPC Client

Type-safe Rust client for Engram memory operations with zero-cost abstractions.

## Quick Start (5 minutes)

```bash
# Build and run
cargo run --release
```

## Progressive Learning Path

### Level 1: Essential Operations (5 min)
```rust
// Store with type safety
let memory_id = client.remember(
    "Important Rust fact".to_string(),
    0.95  // confidence
).await?;

// Retrieve with error handling
let memories = client.recall(
    "Rust memory safety".to_string(),
    5
).await?;
```

### Level 2: Episodic Memory (15 min)
```rust
// Record with builder pattern
let episode_id = client
    .experience("Optimized critical path".to_string())
    .when("During profiling session".to_string())
    .location("Hot loop in parser".to_string())
    .who(vec!["Performance team".to_string()])
    .why("CPU bottleneck".to_string())
    .how("SIMD instructions".to_string())
    .with_emotion("satisfied".to_string())
    .execute()
    .await?;
```

### Level 3: Advanced Streaming (45 min)
```rust
// Dream with pattern matching
client.dream(10, |event| {
    match event {
        DreamEvent::Insight(desc, conf) => {
            println!("Insight: {} ({:.2})", desc, conf);
        }
        DreamEvent::Progress(_, connections, _) => {
            println!("New connections: {}", connections);
        }
        _ => {}
    }
}).await?;
```

## Connection Management

```rust
use tonic::transport::{Channel, ClientTlsConfig};

// With TLS and connection pooling
let channel = Channel::from_static("https://localhost:50051")
    .tls_config(ClientTlsConfig::new())?
    .timeout(Duration::from_secs(5))
    .rate_limit(100, Duration::from_secs(1))
    .concurrency_limit(256)
    .connect()
    .await?;
```

## Error Handling

```rust
match client.remember(content, confidence).await {
    Ok(id) => println!("Success: {}", id),
    Err(e) => {
        if let Some(status) = e.downcast_ref::<Status>() {
            match status.code() {
                Code::Unavailable => {
                    // Retry with exponential backoff
                    tokio::time::sleep(Duration::from_secs(2_u64.pow(retry))).await;
                }
                Code::InvalidArgument => {
                    // Log educational error
                    error!("Invalid input: {}", status.message());
                }
                _ => {}
            }
        }
    }
}
```

## Rust-Specific Features

- **Type Safety**: Compile-time guarantees with strong types
- **Zero-Cost Abstractions**: Builder pattern without runtime overhead
- **Pattern Matching**: Exhaustive handling of dream events
- **Lifetime Management**: Safe memory handling without GC

## Performance Best Practices

1. **Connection Reuse**: Share client across async tasks
2. **Buffered Streaming**: Use `tokio::sync::mpsc` for backpressure
3. **Compile-Time Optimization**: Use `--release` for production
4. **Async Concurrency**: Spawn concurrent operations with `tokio::spawn`

## Cognitive Design Principles

- Type system teaches correctness at compile time
- Result types make error handling explicit
- Builder pattern matches incremental thought
- Pattern matching ensures exhaustive handling