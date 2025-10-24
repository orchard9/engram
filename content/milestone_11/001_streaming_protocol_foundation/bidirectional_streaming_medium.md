# Bidirectional Streaming: Push and Pull in a Single Connection

## The Problem with Batch APIs

Imagine you're building a cognitive AI agent. Your agent is experiencing the world - reading documents, watching videos, having conversations - and needs to remember everything. Traditional batch APIs look like this:

```rust
// Store each observation individually
for experience in experiences {
    memory_system.store(experience).await?;
}

// Query memories when needed
let relevant_memories = memory_system.recall(cue).await?;
```

This works fine for occasional updates. But what if your agent processes 100,000 observations per second? Suddenly you're making 100,000 HTTP requests per second. Even with connection pooling and keep-alive, the overhead crushes you:

- 100K × 500 bytes of HTTP headers = 50 MB/sec just in headers
- 100K round-trips with 1ms latency = 100 seconds of accumulated delay
- TCP slow start on each connection = throughput stuck at 10K/sec

And here's the kicker: while your agent is frantically storing observations, it also needs to recall relevant memories. With a request-response API, you're stuck doing one or the other. Store observations, then query. Query results, then store. Never both at once.

Your brain doesn't work this way. You remember while learning, learn while remembering. The hippocampus encodes new experiences while the prefrontal cortex queries old ones - simultaneously, continuously.

We need a better interface.

## Enter Bidirectional Streaming

Bidirectional streaming means what it sounds like: data flows both ways on a single connection. Client pushes observations to server, server pushes recall results back to client, all at the same time. No separate connections. No request-response roundtrips.

With gRPC bidirectional streaming, the code becomes elegant:

```rust
// Open a single stream
let mut stream = client.observe_and_recall("agent_memory").await?;

// Push observations continuously
tokio::spawn(async move {
    for experience in continuous_observations {
        stream.send(experience).await?;
    }
});

// Pull recall results simultaneously
while let Some(result) = stream.next().await {
    match result {
        Response::Ack(ack) => {
            // Observation indexed, now visible to search
        }
        Response::Memory(memory) => {
            // Relevant memory recalled based on recent observations
        }
    }
}
```

One connection. Two directions. Continuous operation.

## The Protocol: Session Management and Sequence Numbers

Streaming sounds simple until you think about failures. What happens when the network hiccups? How do you resume without losing observations? How do you maintain temporal ordering when observations might arrive out of order?

The answer: session management with monotonic sequence numbers.

### Session Lifecycle

Every stream starts with initialization:

```protobuf
message StreamInit {
  uint32 client_buffer_size = 1;
  bool enable_backpressure = 2;
  uint32 max_batch_size = 3;
}

message StreamInitAck {
  string session_id = 1;
  uint64 initial_sequence = 2;
  StreamCapabilities capabilities = 3;
}
```

Client says: "Here's my buffer capacity, I can handle backpressure, don't send me more than 1000 items at once."

Server responds: "Your session ID is abc-123, start your sequence at 0, and by the way, I can handle 100K observations/sec."

Now the stream is active. Client sends observations with monotonically increasing sequence numbers:

```protobuf
message ObservationRequest {
  string session_id = 1;
  uint64 sequence_number = 2;  // 0, 1, 2, 3, ...
  Episode observation = 3;
}
```

Server echoes the sequence number in the acknowledgment:

```protobuf
message ObservationResponse {
  string session_id = 1;
  uint64 sequence_number = 2;  // Echo: confirms which observation
  ObservationAck ack = 3;
}
```

### Why Sequence Numbers Matter

Sequence numbers provide temporal ordering within a session. If the client sends observations with seq 0, 1, 2, the server knows they happened in that order - even if network delays cause them to arrive 0, 2, 1. The server rejects the out-of-order observation and the client knows to resync.

Here's the client-side implementation:

```rust
pub struct StreamingClient {
    sequence: AtomicU64,
    session_id: String,
}

impl StreamingClient {
    pub async fn observe(&self, episode: Episode) -> Result<ObservationAck> {
        // Atomically increment sequence number
        let seq = self.sequence.fetch_add(1, Ordering::SeqCst);

        let request = ObservationRequest {
            session_id: self.session_id.clone(),
            sequence_number: seq,
            operation: Some(Operation::Observation(episode)),
        };

        let response = self.stream.send(request).await?;

        // Verify server echoed the sequence
        if response.sequence_number != seq {
            return Err(StreamError::SequenceMismatch {
                expected: seq,
                received: response.sequence_number,
            });
        }

        Ok(response.ack)
    }
}
```

And server-side validation:

```rust
fn handle_observation(&self, req: ObservationRequest) -> ObservationResponse {
    let session = self.session_manager.get_session(&req.session_id)?;

    // Validate monotonic sequence
    let expected_seq = session.last_sequence.fetch_add(1, Ordering::SeqCst) + 1;
    if req.sequence_number != expected_seq {
        return ObservationResponse {
            result: Some(Result::Status(StreamStatus {
                state: STATE_ERROR,
                message: format!(
                    "Sequence mismatch: expected {}, got {}",
                    expected_seq, req.sequence_number
                ),
            })),
            session_id: req.session_id,
            sequence_number: req.sequence_number,
            server_timestamp: Some(timestamp_now()),
        };
    }

    // Enqueue observation...
}
```

If the sequence doesn't match, the server rejects it immediately. The client knows something went wrong and can reconnect or resync.

### Graceful Reconnection

Sessions survive network interruptions. If the connection drops, the client can reconnect with the same session ID and resume from the last acknowledged sequence:

```rust
impl StreamingClient {
    pub async fn reconnect(&mut self) -> Result<()> {
        // Keep same session ID and sequence counter
        let last_acked = self.last_acked_sequence.load(Ordering::SeqCst);

        // Reconnect to server
        self.stream = self.connect_with_session(
            &self.session_id,
            last_acked,
        ).await?;

        Ok(())
    }
}
```

The server validates that the resumption point makes sense and allows the stream to continue. No observations lost. No duplicate processing.

## Flow Control: Preventing Queue Overflow

Streaming creates a new problem: what if the client sends observations faster than the server can index them? The server's queue grows unbounded, memory explodes, system crashes.

Traditional solution: block the client until the server catches up. But blocking breaks the streaming abstraction - you're back to request-response semantics.

Better solution: backpressure with flow control messages.

### The Backpressure Protocol

Server monitors its internal queue depth:

```rust
pub fn should_apply_backpressure(&self) -> bool {
    let queue_depth = self.observation_queue.depth();
    let queue_capacity = self.observation_queue.capacity();

    // Backpressure when > 80% full
    queue_depth as f32 / queue_capacity as f32 > 0.8
}
```

When the queue fills up, the server sends a status message:

```protobuf
message StreamStatus {
  State state = 1;  // BACKPRESSURE
  uint32 queue_depth = 3;
  uint32 queue_capacity = 4;
  float pressure = 5;  // 0.85 (85% full)
}
```

Client receives this and adjusts its send rate:

```rust
match status.state {
    STATE_ACTIVE => {
        // Normal operation: 100K obs/sec
        self.send_interval = Duration::from_micros(10);
    }
    STATE_BACKPRESSURE => {
        // Reduce rate by 50%: 50K obs/sec
        self.send_interval = Duration::from_micros(20);
    }
    STATE_OVERLOADED => {
        // Pause and retry after delay
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
}
```

This is adaptive flow control. The server doesn't block the client, it just asks the client to slow down. The client complies (or not - client choice). If the client ignores backpressure, the server has a final defense: admission control.

### Admission Control: The Hard Limit

When the queue reaches 90% capacity, the server stops accepting new observations:

```rust
let result = self.observation_queue.enqueue(episode, priority);

match result {
    Ok(()) => ObservationResponse {
        result: Some(Result::Ack(ObservationAck {
            status: STATUS_ACCEPTED,
            memory_id: episode.id,
            committed_at: Some(timestamp_now()),
        })),
        ...
    },
    Err(QueueError::OverCapacity) => ObservationResponse {
        result: Some(Result::Status(StreamStatus {
            state: STATE_OVERLOADED,
            message: "Queue at capacity, retry later".to_string(),
            queue_depth: self.observation_queue.depth(),
            queue_capacity: self.observation_queue.capacity(),
            pressure: 1.0,
        })),
        ...
    },
}
```

The client receives an error, not an ack. It knows the observation wasn't accepted. It can retry later or reduce its send rate.

Critical invariant: **no silent drops**. The system either accepts the observation (eventual consistency - it will be indexed) or returns an error (client must retry). Never silently discard data.

## Eventual Consistency with Bounded Staleness

Here's where things get cognitively interesting. The server doesn't index observations synchronously. That would be too slow - HNSW insertion takes 100 microseconds per observation. At 100K observations/sec, you'd need 100K × 100μs = 10 seconds of CPU time per second. Math doesn't work.

Instead, the server uses eventual consistency:

1. **Accepted (< 1ms):** Observation queued, client receives ack
2. **Indexed (< 100ms P99):** HNSW insertion complete, visible to search
3. **Consolidated (background):** Pattern completion, spreading activation updates

The observation becomes "queryable" within 100ms of acceptance. This is bounded staleness - not immediate consistency, but fast enough for cognitive workloads.

Why 100ms? Because that's roughly how long it takes your hippocampus to index new episodic memories. The brain has bounded staleness too. When you experience something, you can't immediately recall it with perfect fidelity - there's a brief consolidation period. We're modeling biological reality.

### How Bounded Staleness Works

The server uses a lock-free queue feeding multiple parallel workers:

```
Observation → [SegQueue] → Worker 1 → HNSW index
                         → Worker 2 → HNSW index
                         → Worker 3 → HNSW index
                         → Worker 4 → HNSW index
```

Each worker pulls observations from the queue and indexes them. With 4 workers, throughput quadruples. With adaptive batching (workers batch observations under load), throughput increases further:

- Low load: Process individually, 10ms latency
- Medium load: Batch 100 observations, 30ms latency
- High load: Batch 500 observations, 100ms latency

Latency increases under load, but stays within the 100ms bound. And throughput scales: 4 workers × 25K obs/sec each = 100K obs/sec sustained.

### Cross-Stream Ordering

Important nuance: sequence numbers provide total ordering within a session. Across sessions, ordering is undefined.

If two clients stream simultaneously:

```
Client A: seq 0, 1, 2, 3
Client B: seq 0, 1, 2, 3
```

The server doesn't guarantee global ordering. Client A's observation 2 might get indexed before or after Client B's observation 1. That's fine. The brain has the same property - events from different sensory streams (vision, hearing) don't have guaranteed global ordering. What matters is that events within a stream maintain their sequence.

This is eventual consistency with session-level temporal guarantees. Not linearizable, but biologically realistic.

## Putting It All Together

A complete streaming session looks like this:

```rust
// 1. Initialize stream
let mut client = StreamingClient::connect("localhost:9090").await?;
client.init("agent_memory", StreamInit {
    client_buffer_size: 1000,
    enable_backpressure: true,
    max_batch_size: 100,
}).await?;

// 2. Stream observations
tokio::spawn(async move {
    for i in 0..100_000 {
        let episode = Episode::new(
            format!("experience_{}", i),
            Utc::now(),
            format!("Learned something at step {}", i),
            random_embedding(),
        );

        match client.observe(episode).await {
            Ok(ack) => {
                println!("Observation {} accepted", i);
            }
            Err(StreamError::Backpressure) => {
                // Slow down
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
            Err(e) => {
                eprintln!("Error: {}", e);
                break;
            }
        }
    }
});

// 3. Query memories simultaneously
loop {
    let cue = generate_query_cue();
    let memories = client.recall(cue).await?;

    // Process recalled memories while observations continue
    for memory in memories {
        println!("Recalled: {}", memory.content);
    }

    tokio::time::sleep(Duration::from_secs(1)).await;
}
```

One connection. 100K observations pushed. Queries pulled continuously. Backpressure handled gracefully. Temporal ordering preserved. Eventual consistency with 100ms bounded staleness.

This is how a cognitive memory system should work: continuously observing, continuously recalling, never blocking.

## Why This Matters

Traditional databases separate writes and reads. You insert data, then you query it. Streaming systems like Kafka separate producers and consumers. You write to a log, then you read from it.

Cognitive systems need something different: simultaneous encoding and retrieval. Your brain doesn't stop learning when you try to remember something. An AI agent shouldn't either.

Bidirectional streaming with eventual consistency gives us:

- **High throughput:** 100K observations/sec with 4 cores
- **Low latency:** Observations visible within 100ms
- **Temporal guarantees:** Sequence numbers preserve ordering within sessions
- **Graceful degradation:** Backpressure and admission control prevent cascading failures
- **Biological realism:** Bounded staleness mirrors hippocampal indexing

This isn't just an API design - it's a new way of thinking about memory systems. Not as databases with transactions, but as cognitive processes with bounded uncertainty. Probably right, not definitely consistent.

And that's exactly what we need for building AI that thinks.

---

Generated with Claude Code - https://claude.com/claude-code

*This is part of the Engram project's Milestone 11: Streaming Interface for Real-Time Memory Operations. For implementation details, see the technical specification at github.com/engramhq/engram*
