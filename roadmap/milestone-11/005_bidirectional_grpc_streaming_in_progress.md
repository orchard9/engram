# Task 005: Bidirectional gRPC Streaming

**Status:** pending
**Estimated Effort:** 3 days
**Dependencies:** Task 001 (Protocol Foundation - COMPLETE), Task 002 (Queue - COMPLETE)
**Blocks:** Task 006 (Backpressure), Task 007 (Incremental Recall)

## Objective

Implement gRPC handlers for `ObserveStream` (push observations), `RecallStream` (pull recalls), and `MemoryFlow` (bidirectional) with flow control and session management.

## Background

The streaming protocol foundation is complete (Tasks 001-002):
- Protobuf messages defined in `proto/engram/v1/service.proto` (lines 78-677)
- Session management operational in `engram-core/src/streaming/session.rs`
- Lock-free observation queue ready in `engram-core/src/streaming/observation_queue.rs`

**Critical gap:** No gRPC service handlers connect the protocol to the implementation.

## Current State Analysis

### What Exists (90% foundation):

1. **Protocol Definition** (`proto/engram/v1/service.proto`):
```protobuf
rpc ObserveStream(stream ObservationRequest) returns (stream ObservationResponse);
rpc RecallStream(StreamingRecallRequest) returns (stream StreamingRecallResponse);
rpc MemoryStream(stream ObservationRequest) returns (stream ObservationResponse);
```

2. **Message Types** (lines 574-677):
- `ObservationRequest`: Init, Observation, FlowControl, Close
- `ObservationResponse`: InitAck, ObservationAck, StreamStatus
- `StreamingRecallRequest/Response`: Snapshot isolation params

3. **Session Management** (`engram-core/src/streaming/session.rs`):
- `SessionManager`: Lock-free session storage with DashMap
- `StreamSession`: Monotonic sequence validation, state machine
- Atomic state transitions: Active → Paused → Closed

4. **Observation Queue** (`engram-core/src/streaming/observation_queue.rs`):
- Lock-free `SegQueue` with 3 priority lanes
- Backpressure detection via `should_apply_backpressure()`
- Metrics: enqueue/dequeue counts, queue depths

### What's Missing:

1. **gRPC Handlers:** No implementation in `MemoryService`
2. **Stream Lifecycle:** No init/active/close handling
3. **Error Mapping:** No SessionError → gRPC Status conversion

## Implementation Specification

### Files to Create

1. **`engram-cli/src/handlers/streaming.rs`** (~400 lines)

Core handler struct:
```rust
pub struct StreamingHandlers {
    session_manager: Arc<SessionManager>,
    observation_queue: Arc<ObservationQueue>,
    store: Arc<MemoryStore>,
}
```

Methods:
- `handle_observe_stream()` - Client → server streaming
- `handle_recall_stream()` - Server → client streaming
- `handle_memory_stream()` - Bidirectional streaming

### Files to Modify

1. **`engram-cli/src/grpc.rs`**
   - Add `streaming_handlers: Arc<StreamingHandlers>` to `MemoryService`
   - Implement `EngramService` streaming methods
   - Wire handlers to service methods

2. **`engram-cli/src/handlers/mod.rs`** (create if doesn't exist)
   - Export `streaming` module

3. **`engram-core/src/streaming/mod.rs`**
   - Ensure all types are publicly exported

## Detailed Implementation

### 1. Streaming Handlers Module

**Location:** `engram-cli/src/handlers/streaming.rs`

**Structure:**
```rust
//! Streaming gRPC handlers for continuous memory operations.

use engram_core::{
    MemoryStore, MemorySpaceId, Episode,
    streaming::{
        SessionManager, ObservationQueue, ObservationPriority,
        SessionError, QueueError, SessionState,
    },
};
use engram_proto::{
    ObservationRequest, ObservationResponse,
    observation_request, observation_response,
    StreamInit, StreamInitAck, ObservationAck,
    StreamStatus, StreamCapabilities,
};
use tokio::sync::mpsc;
use tokio_stream::{Stream, StreamExt, wrappers::ReceiverStream};
use tonic::{Request, Response, Status, Streaming};
use std::sync::Arc;

pub struct StreamingHandlers {
    session_manager: Arc<SessionManager>,
    observation_queue: Arc<ObservationQueue>,
    store: Arc<MemoryStore>,
}

impl StreamingHandlers {
    pub fn new(
        session_manager: Arc<SessionManager>,
        observation_queue: Arc<ObservationQueue>,
        store: Arc<MemoryStore>,
    ) -> Self {
        Self {
            session_manager,
            observation_queue,
            store,
        }
    }

    /// Handle ObserveStream RPC
    pub async fn handle_observe_stream(
        &self,
        request: Request<Streaming<ObservationRequest>>,
    ) -> Result<Response<impl Stream<Item = Result<ObservationResponse, Status>>>, Status> {
        // Extract metadata for memory_space_id resolution
        let metadata = request.metadata().clone();
        let mut in_stream = request.into_inner();

        let (tx, rx) = mpsc::channel::<Result<ObservationResponse, Status>>(128);

        // Clone Arc references for spawned task
        let session_manager = Arc::clone(&self.session_manager);
        let observation_queue = Arc::clone(&self.observation_queue);

        // Spawn handler task
        tokio::spawn(async move {
            let mut session_id: Option<String> = None;

            while let Some(result) = in_stream.next().await {
                let req = match result {
                    Ok(r) => r,
                    Err(e) => {
                        let _ = tx.send(Err(Status::internal(format!("Stream error: {e}")))).await;
                        break;
                    }
                };

                // Route based on operation type
                match req.operation {
                    Some(observation_request::Operation::Init(init)) => {
                        // Handle StreamInit
                        if let Err(e) = Self::handle_stream_init(
                            &session_manager,
                            &observation_queue,
                            &tx,
                            &req,
                            init,
                        ).await {
                            let _ = tx.send(Err(e)).await;
                            break;
                        }

                        session_id = Some(
                            // Extract from last sent response or generate
                            uuid::Uuid::new_v4().to_string()
                        );
                    }

                    Some(observation_request::Operation::Observation(episode)) => {
                        // Handle observation
                        if let Err(e) = Self::handle_observation(
                            &session_manager,
                            &observation_queue,
                            &tx,
                            &session_id,
                            &req,
                            episode,
                        ).await {
                            let _ = tx.send(Err(e)).await;
                            // Continue on error - don't break stream
                        }
                    }

                    Some(observation_request::Operation::Flow(flow)) => {
                        // Handle flow control
                        Self::handle_flow_control(
                            &session_manager,
                            &session_id,
                            flow,
                        ).await;
                    }

                    Some(observation_request::Operation::Close(close)) => {
                        // Handle graceful close
                        Self::handle_stream_close(
                            &session_manager,
                            &session_id,
                            close,
                        ).await;
                        break;
                    }

                    None => {
                        let _ = tx.send(Err(Status::invalid_argument(
                            "Empty operation in ObservationRequest"
                        ))).await;
                    }
                }
            }
        });

        Ok(Response::new(ReceiverStream::new(rx)))
    }

    // Helper methods for each operation type
    async fn handle_stream_init(/* ... */) -> Result<(), Status> {
        // Create session
        // Send InitAck with capabilities
        // Return session_id for tracking
    }

    async fn handle_observation(/* ... */) -> Result<(), Status> {
        // Validate session exists
        // Validate sequence number
        // Enqueue observation
        // Send ObservationAck or StreamStatus (backpressure)
    }

    async fn handle_flow_control(/* ... */) {
        // Update session state (pause/resume)
    }

    async fn handle_stream_close(/* ... */) {
        // Mark session closed
        // Optionally drain queue if requested
    }
}
```

**Key Implementation Details:**

1. **Session Lifecycle:**
```rust
// Init: Create session
let session = session_manager.create_session(
    uuid::Uuid::new_v4().to_string(),
    memory_space_id,
);

// Active: Validate sequence on each observation
session.validate_sequence(req.sequence_number)?;

// Close: Mark session closed
session.set_state(SessionState::Closed);
```

2. **Observation Enqueue:**
```rust
let episode_core = Episode::try_from(episode)
    .map_err(|e| Status::invalid_argument(format!("Invalid episode: {e}")))?;

match observation_queue.enqueue(
    session.memory_space_id().clone(),
    episode_core,
    req.sequence_number,
    ObservationPriority::Normal,
) {
    Ok(()) => {
        // Send ObservationAck with ACCEPTED status
    }
    Err(QueueError::OverCapacity { current, limit, .. }) => {
        // Send StreamStatus with BACKPRESSURE state
    }
}
```

3. **Error Mapping:**
```rust
fn map_session_error(err: SessionError) -> Status {
    match err {
        SessionError::NotFound { session_id } => {
            Status::not_found(format!("Session not found: {session_id}"))
        }
        SessionError::SequenceMismatch { expected, received, .. } => {
            Status::invalid_argument(format!(
                "Sequence mismatch: expected {expected}, got {received}"
            ))
        }
        SessionError::InvalidState { reason, .. } => {
            Status::failed_precondition(format!("Invalid state: {reason}"))
        }
    }
}
```

### 2. Extend MemoryService in grpc.rs

**Location:** `engram-cli/src/grpc.rs`

**Modifications:**

1. Add streaming handlers to service:
```rust
pub struct MemoryService {
    store: Arc<MemoryStore>,
    metrics: Arc<MetricsRegistry>,
    registry: Arc<MemorySpaceRegistry>,
    default_space: MemorySpaceId,

    // NEW: Streaming handlers
    streaming_handlers: Arc<StreamingHandlers>,
}

impl MemoryService {
    pub fn new(/* existing params */) -> Self {
        // Create session manager
        let session_manager = Arc::new(SessionManager::new());

        // Get observation queue from store
        let observation_queue = store.observation_queue();

        // Create streaming handlers
        let streaming_handlers = Arc::new(StreamingHandlers::new(
            session_manager,
            observation_queue,
            Arc::clone(&store),
        ));

        Self {
            store,
            metrics,
            registry,
            default_space,
            streaming_handlers,
        }
    }
}
```

2. Implement streaming methods:
```rust
#[tonic::async_trait]
impl EngramService for MemoryService {
    // Existing methods...

    type ObserveStreamStream = Pin<Box<
        dyn Stream<Item = Result<ObservationResponse, Status>> + Send
    >>;

    async fn observe_stream(
        &self,
        request: Request<Streaming<ObservationRequest>>,
    ) -> Result<Response<Self::ObserveStreamStream>, Status> {
        self.streaming_handlers.handle_observe_stream(request).await
            .map(|resp| resp.map(|stream| Box::pin(stream) as Self::ObserveStreamStream))
    }

    type RecallStreamStream = Pin<Box<
        dyn Stream<Item = Result<StreamingRecallResponse, Status>> + Send
    >>;

    async fn recall_stream(
        &self,
        request: Request<StreamingRecallRequest>,
    ) -> Result<Response<Self::RecallStreamStream>, Status> {
        self.streaming_handlers.handle_recall_stream(request).await
            .map(|resp| resp.map(|stream| Box::pin(stream) as Self::RecallStreamStream))
    }

    // Similar for MemoryStream
}
```

### 3. Integration Points

**MemoryStore modifications** (if needed):

Add method to expose observation queue:
```rust
// In engram-core/src/store.rs
impl MemoryStore {
    pub fn observation_queue(&self) -> Arc<ObservationQueue> {
        Arc::clone(&self.observation_queue)
    }
}
```

**Metrics tracking:**

Add streaming metrics to existing registry:
```rust
pub struct StreamingMetrics {
    active_sessions: AtomicUsize,
    observations_per_second: AtomicU64,
    ack_latency_us: AtomicU64,
}
```

## Testing Strategy

### Unit Tests

**File:** `engram-cli/src/handlers/streaming.rs` (test module)

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_stream_init_returns_capabilities() {
        let handlers = setup_test_handlers();

        let init_req = ObservationRequest {
            operation: Some(Operation::Init(StreamInit {
                client_buffer_size: 1000,
                enable_backpressure: true,
                max_batch_size: 100,
            })),
            memory_space_id: "test".to_string(),
            session_id: String::new(),
            sequence_number: 0,
        };

        // Test init response includes session_id and capabilities
    }

    #[tokio::test]
    async fn test_sequence_validation_rejects_gaps() {
        // Send seq 1, 2, 5 (gap at 3-4)
        // Verify error response with expected sequence
    }

    #[tokio::test]
    async fn test_flow_control_pauses_stream() {
        // Send pause
        // Verify session state = Paused
        // Send observation
        // Verify rejected
    }
}
```

### Integration Tests

**File:** `engram-core/tests/integration/streaming_grpc.rs`

```rust
#[tokio::test]
async fn test_observe_stream_end_to_end() {
    // Setup test gRPC server and client
    let (server, client) = setup_test_grpc_service().await;

    // Create observation stream
    let (tx, rx) = mpsc::channel(10);

    // Send init
    tx.send(ObservationRequest {
        operation: Some(Operation::Init(/* ... */)),
        /* ... */
    }).await.unwrap();

    // Send 10 observations with valid sequences
    for i in 1..=10 {
        tx.send(ObservationRequest {
            operation: Some(Operation::Observation(create_test_episode(i))),
            session_id: String::new(),
            sequence_number: i,
        }).await.unwrap();
    }

    // Start stream
    let mut response_stream = client.observe_stream(ReceiverStream::new(rx))
        .await
        .unwrap()
        .into_inner();

    // Verify init ack
    let init_resp = response_stream.next().await.unwrap().unwrap();
    assert!(matches!(init_resp.result, Some(Result::InitAck(_))));

    // Verify 10 observation acks
    for i in 1..=10 {
        let ack_resp = response_stream.next().await.unwrap().unwrap();
        assert!(matches!(ack_resp.result, Some(Result::Ack(_))));
        assert_eq!(ack_resp.sequence_number, i);
    }
}

#[tokio::test]
async fn test_backpressure_triggers_stream_status() {
    // Fill queue to 85% capacity
    // Send observation
    // Verify StreamStatus::BACKPRESSURE received
}

#[tokio::test]
async fn test_graceful_close_drains_queue() {
    // Send observations
    // Send StreamClose with drain_queue=true
    // Verify all observations processed before close confirmation
}
```

### Performance Tests

**File:** `engram-core/benches/streaming_throughput.rs`

```rust
fn bench_observe_stream_throughput(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();

    c.bench_function("observe_stream_10k_ops", |b| {
        b.to_async(&rt).iter(|| async {
            // Stream 10K observations
            // Measure time to receive all acks
            // Target: < 1 second (10K ops/sec)
        });
    });
}

fn bench_observation_ack_latency(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();

    c.bench_function("observation_ack_p99", |b| {
        b.to_async(&rt).iter(|| async {
            // Send single observation
            // Measure time to receive ack
            // Target: P99 < 10ms
        });
    });
}
```

## Acceptance Criteria

### Functional Requirements

- [ ] Client can initialize stream and receive session ID with capabilities
- [ ] Client can send 1000 observations with monotonic sequences and receive acks
- [ ] Sequence validation rejects gaps (expected 5, got 10)
- [ ] Sequence validation rejects duplicates (expected 5, got 3)
- [ ] Flow control pause stops observation processing
- [ ] Flow control resume restores observation processing
- [ ] Graceful close waits for queue drain if requested
- [ ] Immediate close terminates stream without drain

### Performance Requirements

- [ ] **Throughput:** 10K observations/sec sustained for 60 seconds
- [ ] **Latency P99:** < 10ms from observation received to ack sent
- [ ] **Latency P50:** < 5ms from observation received to ack sent
- [ ] **Memory per session:** < 1MB (session state + buffers)
- [ ] **CPU overhead:** < 5% at 10K ops/sec (handler logic only)

### Reliability Requirements

- [ ] No packet loss under normal operation (all acks received)
- [ ] Server crash doesn't corrupt session state (sessions recoverable)
- [ ] Client disconnect cleans up session after timeout (5 minutes)
- [ ] Backpressure activates before OOM (at 80% capacity)
- [ ] Error responses include actionable guidance (expected sequence, etc.)

## Integration Dependencies

### Requires (Completed):
- Task 001: Protocol messages defined
- Task 002: Observation queue operational
- Session manager available
- MemoryStore with observation queue access

### Provides (For downstream tasks):
- Streaming handler infrastructure for Task 006 (Backpressure)
- Session lifecycle management for Task 007 (Incremental Recall)
- Error handling patterns for all streaming tasks

### Blocks:
- Task 006: Backpressure needs handlers to emit flow control
- Task 007: Incremental Recall needs streaming infrastructure
- Task 008: WebSocket can mirror gRPC handler patterns

## Definition of Done

- [ ] All code written following Rust Edition 2024 guidelines
- [ ] `make quality` passes with zero clippy warnings
- [ ] All unit tests passing (test coverage > 80%)
- [ ] All integration tests passing
- [ ] Performance benchmarks meeting targets (10K ops/sec, < 10ms P99)
- [ ] Documentation updated:
  - [ ] Inline rustdoc for all public types/methods
  - [ ] Integration guide in `docs/operations/streaming.md`
  - [ ] Example client in `examples/streaming/rust_client.rs`
- [ ] Diagnostics clean: `./scripts/engram_diagnostics.sh` shows no issues
- [ ] Task file renamed: `005_bidirectional_grpc_streaming_complete.md`
- [ ] Changes committed with descriptive commit message

## Notes

**Architecture Decisions:**

1. **Session ID generation:** Server-generated UUID v4 for uniqueness
2. **Sequence numbering:** Client provides, server validates (catches network issues)
3. **Error handling:** Non-fatal errors continue stream, fatal errors break stream
4. **Memory space resolution:** From request field, fallback to default space

**Performance Optimizations:**

1. Channel buffer size: 128 (balances latency vs memory)
2. Batch responses: Consider batching multiple acks into single response
3. Zero-copy: Use `Arc<Episode>` throughout pipeline

**Future Enhancements (Post-Task):**

- Compression for large episodes (gzip/lz4)
- Client authentication via gRPC metadata
- Multi-space streaming (observations to different spaces in one stream)
- Replay from sequence number (session reconnect)
