# Task 001: Streaming Protocol Foundation

**Status:** Pending
**Estimated Effort:** 3 days
**Dependencies:** None
**Priority:** CRITICAL PATH

## Objective

Define protobuf messages for bidirectional streaming memory operations. Establish session management with monotonic sequence numbers for temporal ordering. Implement basic gRPC service stubs.

## Research Foundation

This implementation follows biological memory formation principles where hippocampal indexing happens asynchronously from neocortical consolidation with bounded staleness of 100ms-1s. Human episodic memory is not linearizable - temporal order reconstruction is probabilistic across sensory modalities (Buzsaki 2015, Marr 1971). We accept cross-stream ordering as undefined while guaranteeing intra-stream total ordering via client-generated monotonic sequences.

**Key Design Decisions:**
- Client-generated monotonic sequences (no network round-trip for coordination)
- Server validates monotonicity (rejects gaps/duplicates)
- Bounded staleness target: P99 < 100ms (biologically-inspired consistency model)
- Eventual consistency over linearizability (matches biological memory dynamics)

**Citations:**
- Buzsaki, G. (2015). "Hippocampal sharp wave-ripple: A cognitive biomarker for episodic memory and planning." Hippocampus, 25(10), 1073-1188.
- Marr, D. (1971). "Simple memory: a theory for archicortex." Philosophical Transactions of the Royal Society B, 262(841), 23-81.
- Lamport, L. (1978). "Time, clocks, and the ordering of events in a distributed system." Communications of the ACM, 21(7), 558-565.

## Technical Specifications

### Protobuf Message Design

**New messages in `proto/engram/v1/service.proto`:**

```protobuf
// Observation stream: continuous memory formation
message ObservationRequest {
  string memory_space_id = 1;

  oneof operation {
    StreamInit init = 2;           // Initialize stream session
    Episode observation = 3;        // Store observation
    FlowControl flow = 4;           // Flow control signal
    StreamClose close = 5;          // Graceful shutdown
  }

  string session_id = 10;           // Client-generated session ID
  uint64 sequence_number = 11;      // Monotonic sequence per session
}

message ObservationResponse {
  oneof result {
    StreamInitAck init_ack = 1;     // Session established
    ObservationAck ack = 2;         // Observation accepted
    StreamStatus status = 3;        // Flow control or error
  }

  string session_id = 10;
  uint64 sequence_number = 11;      // Echo client sequence
  google.protobuf.Timestamp server_timestamp = 12;
}

// Stream initialization
message StreamInit {
  uint32 client_buffer_size = 1;   // Client buffer capacity
  bool enable_backpressure = 2;    // Request flow control
  uint32 max_batch_size = 3;       // Max observations per batch
}

message StreamInitAck {
  string session_id = 1;           // Server-assigned session ID
  uint64 initial_sequence = 2;     // Starting sequence number
  StreamCapabilities capabilities = 3;
}

message StreamCapabilities {
  uint32 max_observations_per_second = 1;
  uint32 queue_capacity = 2;
  bool supports_backpressure = 3;
  bool supports_snapshot_isolation = 4;
}

// Observation acknowledgment
message ObservationAck {
  enum Status {
    STATUS_UNSPECIFIED = 0;
    STATUS_ACCEPTED = 1;          // Queued for indexing
    STATUS_INDEXED = 2;           // Visible in HNSW index
    STATUS_REJECTED = 3;          // Admission control reject
  }

  Status status = 1;
  string memory_id = 2;           // Assigned memory ID
  google.protobuf.Timestamp committed_at = 3;
}

// Stream status and flow control
message StreamStatus {
  enum State {
    STATE_UNSPECIFIED = 0;
    STATE_ACTIVE = 1;
    STATE_PAUSED = 2;             // Client requested pause
    STATE_BACKPRESSURE = 3;       // Server backpressure active
    STATE_OVERLOADED = 4;         // Admission control active
    STATE_ERROR = 5;
  }

  State state = 1;
  string message = 2;
  uint32 queue_depth = 3;         // Current queue depth
  uint32 queue_capacity = 4;      // Maximum capacity
  float pressure = 5;             // 0.0 to 1.0
}

// Streaming recall: pull memories while observing
message StreamingRecallRequest {
  string memory_space_id = 1;
  string session_id = 2;          // Link to observation session

  Cue cue = 3;
  uint32 max_results = 4;
  bool snapshot_isolation = 5;    // See only committed observations
  google.protobuf.Timestamp snapshot_time = 6; // For snapshot isolation
}

message StreamingRecallResponse {
  repeated Memory results = 1;    // Incremental result batch
  bool more_results = 2;          // More results pending
  RecallMetadata metadata = 3;
  uint64 snapshot_sequence = 4;   // Last visible observation sequence
}
```

**gRPC service additions:**

```protobuf
service EngramService {
  // Existing methods...

  // Streaming observation (client → server)
  rpc ObserveStream(stream ObservationRequest) returns (stream ObservationResponse);

  // Streaming recall (server → client)
  rpc RecallStream(StreamingRecallRequest) returns (stream StreamingRecallResponse);

  // Bidirectional: observe + recall in same stream
  rpc MemoryStream(stream ObservationRequest) returns (stream ObservationResponse);
}
```

### Session Management

**Session lifecycle:**

1. **Init:** Client sends `StreamInit`, server returns session ID + capabilities
2. **Active:** Client streams observations with monotonic sequence numbers
3. **Pause:** Client sends `FlowControl::ACTION_PAUSE`, server stops processing
4. **Resume:** Client sends `FlowControl::ACTION_RESUME`, server resumes
5. **Close:** Client sends `StreamClose`, server drains queue and closes

**Invariants:**
- Sequence numbers are monotonic per session (no gaps, no duplicates)
- Server echoes client sequence number in ack
- Session survives network interruptions (reconnect with same session ID)
- Graceful close guarantees all acked observations are indexed

**Implementation in `engram-server/src/grpc/streaming.rs`:**

```rust
pub struct StreamingService {
    memory_registry: Arc<MemorySpaceRegistry>,
    session_manager: Arc<SessionManager>,
    observation_queue: Arc<ObservationQueue>,
}

struct SessionManager {
    sessions: DashMap<String, StreamSession>,
}

struct StreamSession {
    session_id: String,
    memory_space_id: MemorySpaceId,
    last_sequence: AtomicU64,
    created_at: Instant,
    last_activity: AtomicU64,  // Unix timestamp
    state: AtomicU8,  // Active, Paused, Closed
}

impl StreamingService {
    pub async fn observe_stream(
        &self,
        request: tonic::Request<tonic::Streaming<ObservationRequest>>,
    ) -> Result<tonic::Response<impl Stream<Item = Result<ObservationResponse, tonic::Status>>>, tonic::Status> {
        let mut stream = request.into_inner();
        let (tx, rx) = mpsc::channel(100);

        tokio::spawn(async move {
            while let Some(req) = stream.next().await {
                match req {
                    Ok(obs_req) => {
                        // Handle observation or flow control
                        let response = self.handle_observation(obs_req).await;
                        tx.send(Ok(response)).await.ok();
                    }
                    Err(e) => {
                        tx.send(Err(e)).await.ok();
                        break;
                    }
                }
            }
        });

        Ok(tonic::Response::new(ReceiverStream::new(rx)))
    }

    async fn handle_observation(
        &self,
        req: ObservationRequest,
    ) -> ObservationResponse {
        match req.operation {
            Some(Operation::Init(init)) => {
                // Create session
                let session = self.session_manager.create_session(req.memory_space_id, init);
                // Return capabilities
                ObservationResponse {
                    result: Some(Result::InitAck(StreamInitAck {
                        session_id: session.session_id.clone(),
                        initial_sequence: 0,
                        capabilities: Some(StreamCapabilities {
                            max_observations_per_second: 100_000,
                            queue_capacity: 100_000,
                            supports_backpressure: true,
                            supports_snapshot_isolation: true,
                        }),
                    })),
                    session_id: session.session_id,
                    sequence_number: req.sequence_number,
                    server_timestamp: Some(timestamp_now()),
                }
            }
            Some(Operation::Observation(episode)) => {
                // Validate sequence number
                let session = self.session_manager.get_session(&req.session_id)?;
                let expected_seq = session.last_sequence.fetch_add(1, Ordering::SeqCst) + 1;
                if req.sequence_number != expected_seq {
                    // Sequence mismatch - protocol error
                    return ObservationResponse {
                        result: Some(Result::Status(StreamStatus {
                            state: StreamStatus::STATE_ERROR,
                            message: format!("Sequence mismatch: expected {}, got {}", expected_seq, req.sequence_number),
                            ..Default::default()
                        })),
                        session_id: req.session_id,
                        sequence_number: req.sequence_number,
                        server_timestamp: Some(timestamp_now()),
                    };
                }

                // Enqueue observation
                let memory_id = episode.id.clone();
                let result = self.observation_queue.enqueue(
                    req.memory_space_id,
                    episode,
                    req.sequence_number,
                );

                match result {
                    Ok(()) => ObservationResponse {
                        result: Some(Result::Ack(ObservationAck {
                            status: ObservationAck::STATUS_ACCEPTED,
                            memory_id,
                            committed_at: Some(timestamp_now()),
                        })),
                        session_id: req.session_id,
                        sequence_number: req.sequence_number,
                        server_timestamp: Some(timestamp_now()),
                    },
                    Err(QueueError::OverCapacity) => ObservationResponse {
                        result: Some(Result::Status(StreamStatus {
                            state: StreamStatus::STATE_OVERLOADED,
                            message: "Queue at capacity, retry later".to_string(),
                            queue_depth: self.observation_queue.depth(),
                            queue_capacity: self.observation_queue.capacity(),
                            pressure: 1.0,
                        })),
                        session_id: req.session_id,
                        sequence_number: req.sequence_number,
                        server_timestamp: Some(timestamp_now()),
                    },
                }
            }
            Some(Operation::Flow(flow)) => {
                // Handle flow control
                self.handle_flow_control(&req.session_id, flow).await
            }
            Some(Operation::Close(_)) => {
                // Graceful shutdown
                self.session_manager.close_session(&req.session_id).await;
                ObservationResponse {
                    result: Some(Result::Status(StreamStatus {
                        state: StreamStatus::STATE_CLOSED,
                        message: "Session closed".to_string(),
                        ..Default::default()
                    })),
                    session_id: req.session_id,
                    sequence_number: req.sequence_number,
                    server_timestamp: Some(timestamp_now()),
                }
            }
            None => {
                // Invalid request
                ObservationResponse {
                    result: Some(Result::Status(StreamStatus {
                        state: StreamStatus::STATE_ERROR,
                        message: "Missing operation".to_string(),
                        ..Default::default()
                    })),
                    session_id: req.session_id,
                    sequence_number: req.sequence_number,
                    server_timestamp: Some(timestamp_now()),
                }
            }
        }
    }
}
```

### Sequence Number Guarantees

**Client-side:**
```rust
pub struct StreamingClient {
    sequence: AtomicU64,
}

impl StreamingClient {
    pub async fn observe(&self, episode: Episode) -> Result<ObservationAck, StreamError> {
        let seq = self.sequence.fetch_add(1, Ordering::SeqCst);
        let request = ObservationRequest {
            session_id: self.session_id.clone(),
            sequence_number: seq,
            operation: Some(Operation::Observation(episode)),
            ..Default::default()
        };

        let response = self.stream.send(request).await?;

        // Verify sequence echo
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

**Server-side validation:**
- Reject out-of-order sequences (gap or duplicate)
- Return error response with expected sequence number
- Client must resync or reconnect

## Files to Create

- `engram-server/src/grpc/streaming.rs` (350 lines)
- `engram-core/src/streaming/mod.rs` (module declaration)
- `engram-core/src/streaming/session.rs` (200 lines)

## Files to Modify

- `proto/engram/v1/service.proto` (add 150 lines)
- `proto/engram/v1/memory.proto` (add StreamStatus enum)
- `engram-proto/src/lib.rs` (add convenience methods for streaming types)
- `engram-server/src/grpc/service.rs` (wire up streaming endpoints)

## Testing Strategy

### Unit Tests

```rust
#[tokio::test]
async fn test_session_creation() {
    let manager = SessionManager::new();
    let init = StreamInit { client_buffer_size: 1000, .. };
    let session = manager.create_session("space1", init);

    assert_eq!(session.last_sequence.load(Ordering::SeqCst), 0);
    assert_eq!(manager.get_session(&session.session_id).unwrap().session_id, session.session_id);
}

#[tokio::test]
async fn test_sequence_validation() {
    let manager = SessionManager::new();
    let session = manager.create_session("space1", StreamInit::default());

    // Valid sequence
    assert!(session.validate_sequence(0));
    assert!(session.validate_sequence(1));
    assert!(session.validate_sequence(2));

    // Invalid: gap
    assert!(!session.validate_sequence(10));

    // Invalid: duplicate
    assert!(!session.validate_sequence(1));
}
```

### Integration Tests

```rust
#[tokio::test]
async fn test_streaming_roundtrip() {
    let server = spawn_test_server().await;
    let client = StreamingClient::connect(server.addr()).await.unwrap();

    // Initialize stream
    client.init("test_space", StreamInit::default()).await.unwrap();

    // Send 10 observations
    for i in 0..10 {
        let episode = Episode::new(
            format!("ep{}", i),
            Utc::now(),
            format!("Observation {}", i),
            vec![0.1; 768],
        );
        let ack = client.observe(episode).await.unwrap();
        assert_eq!(ack.status, ObservationAck::STATUS_ACCEPTED);
    }

    // Close stream
    client.close().await.unwrap();
}

#[tokio::test]
async fn test_session_timeout() {
    let manager = SessionManager::with_timeout(Duration::from_secs(5));
    let session = manager.create_session("space1", StreamInit::default());

    // Session active immediately
    assert!(manager.get_session(&session.session_id).is_some());

    // Wait for timeout
    tokio::time::sleep(Duration::from_secs(6)).await;

    // Session expired
    assert!(manager.get_session(&session.session_id).is_none());
}
```

### Property Tests

```rust
use proptest::prelude::*;

proptest! {
    #[test]
    fn prop_sequence_monotonic(observations in prop::collection::vec(any::<Episode>(), 1..1000)) {
        let runtime = tokio::runtime::Runtime::new().unwrap();
        runtime.block_on(async {
            let client = StreamingClient::test_client().await;

            let mut last_seq = 0u64;
            for episode in observations {
                let ack = client.observe(episode).await.unwrap();
                assert!(ack.sequence > last_seq);
                last_seq = ack.sequence;
            }
        });
    }
}
```

## Acceptance Criteria

1. Client can initialize stream and receive session ID + capabilities
2. Send 1000 observations with monotonic sequence numbers
3. Server echoes sequence numbers in acks
4. Out-of-order observation rejected with error
5. Session survives 60s idle (no timeout)
6. Graceful close returns final sequence number
7. Protobuf schema validated with `buf lint`
8. All tests pass with `cargo test streaming_protocol`

## Performance Targets

Research-validated performance bounds:
- Session creation: < 1ms (protobuf deserialization + DashMap insert)
- Observation ack latency: < 1ms (just enqueue, not index)
- Sequence validation: < 100ns (atomic fetch_add with SeqCst ordering)
- Session lookup: < 500ns (DashMap get with lock-free read path)
- Throughput: 100K+ obs/sec (gRPC bidirectional streaming, binary protobuf)
- Session timeout: 5 minutes idle (matches biological working memory decay)

**Staleness guarantees:**
- Target: P99 staleness < 100ms (time from observation committed to indexed)
- Measurement: staleness = time(observation_indexed) - time(observation_committed)
- Adaptive batching enables 10ms (low load) to 100ms (high load) P99 latency

**Flow control thresholds:**
- ACTIVE: queue depth < 80%
- BACKPRESSURE: queue depth 80-90%
- OVERLOADED: queue depth > 90% (admission control rejects)

## Dependencies

None - this is the foundation task.

## Next Steps

After this task:
- Task 002 implements the `ObservationQueue` that `handle_observation()` uses
- Task 005 completes the gRPC streaming handlers with full flow control
