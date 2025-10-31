# Milestone 11 Tasks 005-007: Implementation Guidance

**Systems Architect:** Margo Seltzer
**Date:** 2025-10-30
**Status:** Implementation Ready

## Executive Summary

Tasks 005 (Bidirectional gRPC Streaming), 006 (Backpressure), and 007 (Incremental Recall) build on the streaming foundation established in Tasks 001-004. Current analysis reveals:

1. **Protocol foundation EXISTS** - protobuf messages defined, session management complete
2. **Queue infrastructure EXISTS** - lock-free observation queue with priority lanes operational
3. **Critical gap: gRPC handlers** - Service methods defined but handlers not implemented
4. **Critical gap: Worker pool** - Queue exists but no HNSW worker pool consuming it
5. **Critical gap: Snapshot isolation** - Recall exists but not streaming-aware

**Estimated effort:** 8 days total (3 + 2 + 3) for single engineer
**Critical path:** 005 → 006 → 007 (sequential, cannot parallelize)

---

## Task 005: Bidirectional gRPC Streaming

### Current Implementation Status

**PROTOCOL: 90% COMPLETE**
- `/Users/jordan/Workspace/orchard9/engram/proto/engram/v1/service.proto` lines 78-87 define:
  - `rpc ObserveStream(stream ObservationRequest) returns (stream ObservationResponse)`
  - `rpc RecallStream(StreamingRecallRequest) returns (stream StreamingRecallResponse)`
  - `rpc MemoryStream(stream ObservationRequest) returns (stream ObservationResponse)`
- Messages defined lines 574-677: `ObservationRequest/Response`, `StreamInit`, `StreamStatus`, etc.

**SESSION MANAGEMENT: 100% COMPLETE**
- `/Users/jordan/Workspace/orchard9/engram/engram-core/src/streaming/session.rs`
- `SessionManager` with `DashMap` for lock-free concurrent sessions
- Monotonic sequence validation with atomic counters
- State machine: Active → Paused → Closed

**OBSERVATION QUEUE: 100% COMPLETE**
- `/Users/jordan/Workspace/orchard9/engram/engram-core/src/streaming/observation_queue.rs`
- Lock-free `SegQueue` with 3 priority lanes
- Backpressure detection via `should_apply_backpressure()`
- Batch dequeue for efficient worker processing

**CRITICAL GAP: gRPC HANDLERS**
- `/Users/jordan/Workspace/orchard9/engram/engram-cli/src/grpc.rs` implements non-streaming RPCs
- NO IMPLEMENTATION for `ObserveStream`, `RecallStream`, `MemoryStream`
- Need to create streaming handler methods in `MemoryService` impl

### Implementation Approach

**File Structure:**
```
engram-cli/src/
├── grpc.rs                    (extend MemoryService impl)
└── handlers/
    └── streaming.rs          (CREATE - streaming-specific handlers)
```

**Architecture:**
```
Client Stream                    Server Stream
     │                                │
     ├─► ObservationRequest ─────────┤
     │   ├─ StreamInit              │
     │   ├─ Episode                 │
     │   ├─ FlowControl             │
     │   └─ StreamClose             │
     │                              │
     │                              ├─► ObservationResponse
     │                              │   ├─ StreamInitAck
     │                              │   ├─ ObservationAck
     │                              │   └─ StreamStatus
     └──────────────────────────────┘
```

**Implementation Steps:**

1. **Create streaming handlers module** (`engram-cli/src/handlers/streaming.rs`):

```rust
//! Streaming gRPC handlers for continuous memory operations.

use engram_core::streaming::{
    ObservationQueue, SessionManager, ObservationPriority,
    QueueError, SessionError, SessionState
};
use engram_proto::{
    ObservationRequest, ObservationResponse,
    observation_request, observation_response,
    StreamInit, StreamInitAck, ObservationAck,
    StreamStatus, StreamCapabilities,
    observation_ack::Status as AckStatus,
    stream_status::State as StreamState,
};
use tokio_stream::Stream;
use tonic::{Request, Response, Status, Streaming};
use std::pin::Pin;
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

    /// Handle ObserveStream: client → server streaming
    pub async fn handle_observe_stream(
        &self,
        request: Request<Streaming<ObservationRequest>>,
    ) -> Result<Response<Streaming<ObservationResponse>>, Status> {
        let mut in_stream = request.into_inner();
        let (tx, rx) = tokio::sync::mpsc::channel(128);

        let session_manager = Arc::clone(&self.session_manager);
        let observation_queue = Arc::clone(&self.observation_queue);

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

                match req.operation {
                    Some(observation_request::Operation::Init(init)) => {
                        // Initialize session
                        let session = session_manager.create_session(
                            uuid::Uuid::new_v4().to_string(),
                            // TODO: extract memory_space_id from req.memory_space_id
                            MemorySpaceId::default(),
                        );

                        session_id = Some(session.session_id().to_string());

                        let response = ObservationResponse {
                            result: Some(observation_response::Result::InitAck(StreamInitAck {
                                session_id: session.session_id().to_string(),
                                initial_sequence: 0,
                                capabilities: Some(StreamCapabilities {
                                    max_observations_per_second: 100_000,
                                    queue_capacity: observation_queue.total_capacity() as u32,
                                    supports_backpressure: true,
                                    supports_snapshot_isolation: true,
                                }),
                            })),
                            session_id: session.session_id().to_string(),
                            sequence_number: 0,
                            server_timestamp: Some(prost_types::Timestamp::from(std::time::SystemTime::now())),
                        };

                        if tx.send(Ok(response)).await.is_err() {
                            break;
                        }
                    }

                    Some(observation_request::Operation::Observation(episode)) => {
                        // Validate session exists
                        let Some(ref sid) = session_id else {
                            let _ = tx.send(Err(Status::failed_precondition(
                                "Must send StreamInit before observations"
                            ))).await;
                            break;
                        };

                        let session = match session_manager.get_session(sid) {
                            Ok(s) => s,
                            Err(_) => {
                                let _ = tx.send(Err(Status::not_found("Session not found"))).await;
                                break;
                            }
                        };

                        // Validate sequence number
                        if let Err(e) = session.validate_sequence(req.sequence_number) {
                            let _ = tx.send(Err(Status::invalid_argument(
                                format!("Sequence error: {e}")
                            ))).await;
                            break;
                        }

                        // Enqueue observation
                        let episode_core = Episode::try_from(episode).map_err(|e|
                            Status::invalid_argument(format!("Invalid episode: {e}"))
                        )?;

                        match observation_queue.enqueue(
                            session.memory_space_id().clone(),
                            episode_core,
                            req.sequence_number,
                            ObservationPriority::Normal,
                        ) {
                            Ok(()) => {
                                // Accepted
                                let response = ObservationResponse {
                                    result: Some(observation_response::Result::Ack(ObservationAck {
                                        status: AckStatus::Accepted as i32,
                                        memory_id: format!("mem_{}", req.sequence_number),
                                        committed_at: Some(prost_types::Timestamp::from(std::time::SystemTime::now())),
                                    })),
                                    session_id: sid.clone(),
                                    sequence_number: req.sequence_number,
                                    server_timestamp: Some(prost_types::Timestamp::from(std::time::SystemTime::now())),
                                };

                                if tx.send(Ok(response)).await.is_err() {
                                    break;
                                }
                            }
                            Err(QueueError::OverCapacity { .. }) => {
                                // Backpressure
                                let response = ObservationResponse {
                                    result: Some(observation_response::Result::Status(StreamStatus {
                                        state: StreamState::Backpressure as i32,
                                        message: "Queue capacity exceeded - reduce send rate".to_string(),
                                        queue_depth: observation_queue.total_depth() as u32,
                                        queue_capacity: observation_queue.total_capacity() as u32,
                                        pressure: observation_queue.total_depth() as f32
                                                 / observation_queue.total_capacity() as f32,
                                    })),
                                    session_id: sid.clone(),
                                    sequence_number: req.sequence_number,
                                    server_timestamp: Some(prost_types::Timestamp::from(std::time::SystemTime::now())),
                                };

                                if tx.send(Ok(response)).await.is_err() {
                                    break;
                                }
                            }
                        }
                    }

                    Some(observation_request::Operation::Flow(flow)) => {
                        // Handle flow control
                        let Some(ref sid) = session_id else { continue; };
                        let Ok(session) = session_manager.get_session(sid) else { continue; };

                        use engram_proto::flow_control::Action;
                        match flow.action() {
                            Action::Pause => session.set_state(SessionState::Paused),
                            Action::Resume => session.set_state(SessionState::Active),
                            _ => {}
                        }
                    }

                    Some(observation_request::Operation::Close(close)) => {
                        // Graceful close
                        if let Some(ref sid) = session_id {
                            let _ = session_manager.close_session(sid);
                        }
                        break;
                    }

                    None => {}
                }
            }
        });

        Ok(Response::new(ReceiverStream::new(rx)))
    }
}
```

2. **Extend MemoryService in grpc.rs**:

```rust
// Add to MemoryService struct
streaming_handlers: Arc<StreamingHandlers>,

// Implement streaming methods
#[tonic::async_trait]
impl EngramService for MemoryService {
    // ... existing methods ...

    type ObserveStreamStream = Pin<Box<dyn Stream<Item = Result<ObservationResponse, Status>> + Send>>;

    async fn observe_stream(
        &self,
        request: Request<Streaming<ObservationRequest>>,
    ) -> Result<Response<Self::ObserveStreamStream>, Status> {
        self.streaming_handlers.handle_observe_stream(request).await
    }

    // Similar for RecallStream and MemoryStream
}
```

### Integration Points

1. **MemoryStore → ObservationQueue**: Store must provide queue reference
2. **SessionManager**: Create singleton in main.rs, pass to service
3. **Metrics**: Track stream count, observation rate, backpressure events
4. **Error Handling**: Map SessionError/QueueError to gRPC Status codes

### Dependencies

- **Blocks on:** Task 003 (Parallel HNSW Worker Pool) - queue needs consumer
- **Blocked by:** None - protocol and queue are ready

### Testing Strategy

**Unit Tests** (in `streaming.rs`):
- Session initialization: send `StreamInit`, verify `StreamInitAck`
- Observation flow: send 10 episodes, verify 10 acks
- Sequence validation: send out-of-order, verify rejection
- Flow control: send pause, verify observation rejected with error

**Integration Tests** (create `engram-core/tests/integration/streaming_grpc.rs`):
```rust
#[tokio::test]
async fn test_observe_stream_end_to_end() {
    let (service, client) = setup_test_service().await;

    let (tx, rx) = tokio::sync::mpsc::channel(10);

    // Send init
    tx.send(ObservationRequest {
        operation: Some(Operation::Init(StreamInit {
            client_buffer_size: 1000,
            enable_backpressure: true,
            max_batch_size: 100,
        })),
        session_id: String::new(),
        sequence_number: 0,
    }).await.unwrap();

    // Send observations
    for i in 1..=10 {
        tx.send(ObservationRequest {
            operation: Some(Operation::Observation(test_episode())),
            session_id: String::new(),
            sequence_number: i,
        }).await.unwrap();
    }

    // Receive responses
    let mut response_stream = client.observe_stream(ReceiverStream::new(rx)).await?.into_inner();

    // Verify init ack
    let init_ack = response_stream.next().await.unwrap()?;
    assert!(matches!(init_ack.result, Some(Result::InitAck(_))));

    // Verify 10 observation acks
    for _ in 1..=10 {
        let ack = response_stream.next().await.unwrap()?;
        assert!(matches!(ack.result, Some(Result::Ack(_))));
    }
}
```

**Performance Tests** (criterion benchmark):
- Throughput: 10K observations/sec sustained for 60s
- Latency: P99 < 10ms for observation → ack
- Backpressure activation: fill queue to 85%, verify status change

### Acceptance Criteria

1. **Functional:**
   - [ ] Client can initialize stream and receive session ID
   - [ ] Client can send 1000 observations and receive acks
   - [ ] Sequence validation rejects gaps and duplicates
   - [ ] Flow control pause/resume works
   - [ ] Graceful close drains queue

2. **Performance:**
   - [ ] 10K observations/sec sustained throughput
   - [ ] P99 latency < 10ms (observation → ack)
   - [ ] Zero packet loss under normal operation

3. **Reliability:**
   - [ ] Server crash doesn't corrupt session state
   - [ ] Client reconnect restores session (within timeout)
   - [ ] Backpressure activates before OOM

---

## Task 006: Backpressure and Admission Control

### Current Implementation Status

**BACKPRESSURE DETECTION: 80% COMPLETE**
- `ObservationQueue::should_apply_backpressure()` returns bool when > 80% full
- `ObservationQueue::enqueue()` returns `QueueError::OverCapacity` when full
- `StreamStatus` message defined in proto with pressure field

**CRITICAL GAP: Adaptive Batching**
- Queue has `dequeue_batch()` but no logic to adjust batch size based on pressure
- No worker pool to consume batches (Task 003 dependency)

**CRITICAL GAP: Flow Control Emission**
- Server can detect backpressure but doesn't proactively send `StreamStatus::SLOW_DOWN`
- No periodic pressure monitoring loop

### Implementation Approach

**Architecture:**
```
┌─────────────────────────────────────────────────────┐
│  Backpressure Monitor (background task)            │
│  - Check queue depth every 100ms                   │
│  - Emit StreamStatus when pressure > 80%           │
│  - Track backpressure duration for alerting        │
└─────────────────────────────────────────────────────┘
                      │
                      ├──► StreamStatus::SLOW_DOWN
                      │    (to all active sessions)
                      │
                      ▼
┌─────────────────────────────────────────────────────┐
│  Admission Control (enqueue time)                  │
│  - Check queue depth < capacity                    │
│  - Reject with RESOURCE_EXHAUSTED if full          │
│  - Include retry-after based on drain rate         │
└─────────────────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────┐
│  Adaptive Batching (worker pool)                   │
│  - Normal load: batch size = 10 (low latency)      │
│  - Medium load (50-80%): batch size = 100          │
│  - High load (>80%): batch size = 1000 (throughput)│
└─────────────────────────────────────────────────────┘
```

**Implementation Steps:**

1. **Create backpressure module** (`engram-core/src/streaming/backpressure.rs`):

```rust
//! Adaptive backpressure control for streaming observations.

use crate::streaming::{ObservationQueue, SessionManager};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::broadcast;

/// Backpressure state emitted to clients
#[derive(Debug, Clone, Copy)]
pub enum BackpressureState {
    Normal,           // < 50% capacity
    Warning,          // 50-80% capacity
    Critical,         // 80-95% capacity
    Overloaded,       // > 95% capacity
}

impl BackpressureState {
    pub fn from_pressure(pressure: f32) -> Self {
        if pressure < 0.5 {
            Self::Normal
        } else if pressure < 0.8 {
            Self::Warning
        } else if pressure < 0.95 {
            Self::Critical
        } else {
            Self::Overloaded
        }
    }

    pub fn recommended_batch_size(&self) -> usize {
        match self {
            Self::Normal => 10,        // Low latency
            Self::Warning => 100,      // Balanced
            Self::Critical => 500,     // High throughput
            Self::Overloaded => 1000,  // Maximum throughput
        }
    }
}

/// Backpressure monitor that periodically checks queue depth
/// and emits state changes to subscribed sessions
pub struct BackpressureMonitor {
    observation_queue: Arc<ObservationQueue>,
    state_tx: broadcast::Sender<BackpressureState>,
    check_interval: Duration,
}

impl BackpressureMonitor {
    pub fn new(
        observation_queue: Arc<ObservationQueue>,
        check_interval: Duration,
    ) -> Self {
        let (state_tx, _) = broadcast::channel(32);
        Self {
            observation_queue,
            state_tx,
            check_interval,
        }
    }

    /// Subscribe to backpressure state changes
    pub fn subscribe(&self) -> broadcast::Receiver<BackpressureState> {
        self.state_tx.subscribe()
    }

    /// Run monitoring loop (spawn as background task)
    pub async fn run(&self) {
        let mut current_state = BackpressureState::Normal;
        let mut interval = tokio::time::interval(self.check_interval);

        loop {
            interval.tick().await;

            let total_depth = self.observation_queue.total_depth();
            let total_capacity = self.observation_queue.total_capacity();
            let pressure = total_depth as f32 / total_capacity as f32;

            let new_state = BackpressureState::from_pressure(pressure);

            if !matches!((current_state, new_state),
                (BackpressureState::Normal, BackpressureState::Normal) |
                (BackpressureState::Warning, BackpressureState::Warning) |
                (BackpressureState::Critical, BackpressureState::Critical) |
                (BackpressureState::Overloaded, BackpressureState::Overloaded)
            ) {
                // State changed - notify subscribers
                let _ = self.state_tx.send(new_state);
                current_state = new_state;

                tracing::info!(
                    "Backpressure state changed: {:?} (pressure: {:.1}%)",
                    new_state,
                    pressure * 100.0
                );
            }
        }
    }
}

/// Calculate retry-after duration based on current drain rate
pub fn calculate_retry_after(
    queue_depth: usize,
    dequeue_rate: f32, // observations per second
) -> Duration {
    if dequeue_rate < 1.0 {
        return Duration::from_secs(60); // Pessimistic fallback
    }

    // Time to drain excess capacity (above 50% mark)
    let target_depth = queue_depth / 2;
    let excess = queue_depth.saturating_sub(target_depth);
    let drain_seconds = excess as f32 / dequeue_rate;

    Duration::from_secs_f32(drain_seconds.min(300.0)) // Cap at 5 minutes
}
```

2. **Integrate with streaming handlers**:

```rust
// In streaming.rs, add backpressure monitor
pub struct StreamingHandlers {
    // ... existing fields ...
    backpressure_monitor: Arc<BackpressureMonitor>,
}

impl StreamingHandlers {
    pub async fn handle_observe_stream(/* ... */) -> Result</* ... */> {
        // Subscribe to backpressure state changes
        let mut backpressure_rx = self.backpressure_monitor.subscribe();

        // Spawn task to forward backpressure state to client
        let tx_clone = tx.clone();
        let session_id_clone = session_id.clone();
        tokio::spawn(async move {
            while let Ok(state) = backpressure_rx.recv().await {
                let status = StreamStatus {
                    state: match state {
                        BackpressureState::Normal => StreamState::Active,
                        BackpressureState::Warning => StreamState::Active,
                        BackpressureState::Critical => StreamState::Backpressure,
                        BackpressureState::Overloaded => StreamState::Overloaded,
                    } as i32,
                    message: format!("Backpressure: {:?}", state),
                    queue_depth: observation_queue.total_depth() as u32,
                    queue_capacity: observation_queue.total_capacity() as u32,
                    pressure: observation_queue.total_depth() as f32
                             / observation_queue.total_capacity() as f32,
                };

                let response = ObservationResponse {
                    result: Some(observation_response::Result::Status(status)),
                    session_id: session_id_clone.clone(),
                    sequence_number: 0,
                    server_timestamp: Some(prost_types::Timestamp::from(std::time::SystemTime::now())),
                };

                if tx_clone.send(Ok(response)).await.is_err() {
                    break;
                }
            }
        });

        // ... existing stream handling logic ...
    }
}
```

3. **Admission control with retry-after** (in observation enqueue):

```rust
match observation_queue.enqueue(/* ... */) {
    Ok(()) => { /* ... */ }
    Err(QueueError::OverCapacity { current, limit, .. }) => {
        // Calculate retry-after based on drain rate
        let dequeue_rate = metrics.get_dequeue_rate(); // observations/sec
        let retry_after = backpressure::calculate_retry_after(current, dequeue_rate);

        return Err(Status::resource_exhausted(format!(
            "Queue capacity exceeded ({}/{}) - retry after {}s",
            current, limit, retry_after.as_secs()
        )));
    }
}
```

### Integration Points

1. **Worker Pool** (Task 003): Worker pool uses `BackpressureState::recommended_batch_size()` for adaptive batching
2. **Metrics**: Track backpressure duration, admission control rejection rate
3. **Monitoring**: Prometheus metrics for backpressure state histogram

### Dependencies

- **Blocks on:** Task 003 (Worker Pool) - adaptive batching needs consumer
- **Blocked by:** Task 005 - needs streaming handlers to emit flow control

### Testing Strategy

**Unit Tests**:
```rust
#[test]
fn test_backpressure_state_thresholds() {
    assert_eq!(BackpressureState::from_pressure(0.3), BackpressureState::Normal);
    assert_eq!(BackpressureState::from_pressure(0.6), BackpressureState::Warning);
    assert_eq!(BackpressureState::from_pressure(0.85), BackpressureState::Critical);
    assert_eq!(BackpressureState::from_pressure(0.98), BackpressureState::Overloaded);
}

#[test]
fn test_adaptive_batch_sizing() {
    assert_eq!(BackpressureState::Normal.recommended_batch_size(), 10);
    assert_eq!(BackpressureState::Critical.recommended_batch_size(), 500);
}

#[tokio::test]
async fn test_backpressure_monitor_state_changes() {
    let queue = Arc::new(ObservationQueue::new(QueueConfig {
        high_capacity: 100,
        normal_capacity: 100,
        low_capacity: 100,
    }));

    let monitor = BackpressureMonitor::new(queue.clone(), Duration::from_millis(10));
    let mut rx = monitor.subscribe();

    tokio::spawn(async move { monitor.run().await });

    // Fill queue to 60% - should trigger Warning
    for i in 0..180 {
        queue.enqueue(/* ... */, ObservationPriority::Normal).unwrap();
    }

    let state = tokio::time::timeout(Duration::from_secs(1), rx.recv()).await.unwrap().unwrap();
    assert!(matches!(state, BackpressureState::Warning));
}
```

**Integration Tests**:
```rust
#[tokio::test]
async fn test_admission_control_rejects_when_full() {
    // Fill queue to capacity
    // Attempt enqueue
    // Verify RESOURCE_EXHAUSTED error with retry-after
}

#[tokio::test]
async fn test_backpressure_notifies_active_streams() {
    // Start streaming session
    // Fill queue to 85%
    // Verify client receives StreamStatus::BACKPRESSURE
}
```

### Acceptance Criteria

1. **Functional:**
   - [ ] Backpressure monitor detects state changes within 100ms
   - [ ] Active streams receive `StreamStatus` when pressure > 80%
   - [ ] Admission control rejects enqueue when queue full
   - [ ] Retry-after calculation reflects actual drain rate

2. **Performance:**
   - [ ] Monitor overhead < 0.1% CPU
   - [ ] State change notification latency < 10ms
   - [ ] No memory leaks under sustained backpressure

3. **Reliability:**
   - [ ] Queue never exceeds configured capacity
   - [ ] Backpressure doesn't cause session termination
   - [ ] Recovery from overload state within 10s of load reduction

---

## Task 007: Incremental Recall with Snapshot Isolation

### Current Implementation Status

**RECALL EXISTS: 70% COMPLETE**
- `/Users/jordan/Workspace/orchard9/engram/engram-core/src/query/executor.rs` implements recall
- HNSW search in `/Users/jordan/Workspace/orchard9/engram/engram-core/src/index/hnsw_search.rs`
- NOT snapshot-aware - sees all observations regardless of indexing status

**CRITICAL GAP: Snapshot Isolation**
- No mechanism to capture "committed observations before timestamp T"
- HNSW index doesn't track insertion timestamps
- No generation-based visibility control

**CRITICAL GAP: Incremental Streaming**
- Recall returns `Vec<Memory>` - full result set, not streaming
- No way to return results as HNSW search progresses

### Implementation Approach

**Architecture:**
```
Client Request                Server Processing
     │                             │
     ├─► StreamingRecallRequest ───┤
     │   ├─ cue                    │
     │   ├─ snapshot_time          │
     │   └─ snapshot_isolation     │
     │                             │
     │                             ├─► Capture snapshot generation
     │                             │   (last committed sequence)
     │                             │
     │                             ├─► HNSW search with visibility filter
     │                             │   ├─ Search neighbors
     │                             │   ├─ Filter by generation <= snapshot
     │                             │   └─ Yield results incrementally
     │                             │
     │                             ├─► StreamingRecallResponse (batch 1)
     │◄────────────────────────────┤   ├─ results: Vec<Memory>
     │                             │   └─ more_results: true
     │                             │
     │                             ├─► StreamingRecallResponse (batch 2)
     │◄────────────────────────────┤   └─ more_results: false
     └─────────────────────────────┘
```

**Key Insight:** "Snapshot isolation" in eventual consistency means:
- All observations **committed to HNSW index** before snapshot time are visible
- Observations **in queue** may or may not be visible (probabilistic)
- No linearizability guarantee - acceptable for cognitive memory model

**Implementation Steps:**

1. **Add generation tracking to HNSW index** (`engram-core/src/index/hnsw_node.rs`):

```rust
/// HNSW node with generation-based visibility tracking
pub struct HnswNode<T> {
    pub id: String,
    pub data: T,
    pub neighbors: Vec<Vec<usize>>, // neighbors per layer

    /// Generation (sequence number) when this node was inserted
    /// Used for snapshot isolation in streaming recall
    pub generation: u64,

    /// Timestamp when node became visible in index
    pub committed_at: Instant,
}
```

2. **Modify `ObservationQueue` to track current generation**:

```rust
pub struct ObservationQueue {
    // ... existing fields ...

    /// Current generation (last committed observation)
    /// Updated atomically when workers confirm HNSW insertion
    current_generation: AtomicU64,
}

impl ObservationQueue {
    pub fn current_generation(&self) -> u64 {
        self.current_generation.load(Ordering::SeqCst)
    }

    pub fn mark_generation_committed(&self, generation: u64) {
        // Update to max of current and new generation (handles out-of-order commits)
        self.current_generation.fetch_max(generation, Ordering::SeqCst);
    }
}
```

3. **Create incremental recall module** (`engram-core/src/streaming/recall.rs`):

```rust
//! Incremental recall with snapshot isolation for streaming queries.

use crate::index::{HnswGraph, HnswNode};
use crate::memory::Memory;
use crate::query::Cue;
use crate::streaming::ObservationQueue;
use std::sync::Arc;
use tokio_stream::Stream;
use std::pin::Pin;

/// Snapshot-isolated recall configuration
pub struct SnapshotRecallConfig {
    /// Snapshot generation (observations committed before this are visible)
    pub snapshot_generation: u64,

    /// Batch size for incremental results
    pub batch_size: usize,

    /// Include in-flight observations (bounded staleness)
    pub include_recent: bool,
}

/// Incremental recall stream that yields batches of results
pub struct IncrementalRecallStream {
    /// HNSW graph reference
    graph: Arc<HnswGraph>,

    /// Query cue
    cue: Cue,

    /// Snapshot config
    config: SnapshotRecallConfig,

    /// Current position in results
    position: usize,

    /// Cached results from HNSW search
    results: Vec<(usize, f32)>, // (node_id, score)
}

impl IncrementalRecallStream {
    pub fn new(
        graph: Arc<HnswGraph>,
        cue: Cue,
        observation_queue: &ObservationQueue,
        include_recent: bool,
    ) -> Self {
        let snapshot_generation = observation_queue.current_generation();

        Self {
            graph,
            cue,
            config: SnapshotRecallConfig {
                snapshot_generation,
                batch_size: 10,
                include_recent,
            },
            position: 0,
            results: Vec::new(),
        }
    }

    /// Execute HNSW search with snapshot isolation
    pub async fn search(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        // Convert cue to embedding
        let query_embedding = self.cue.to_embedding()?;

        // HNSW search (this is synchronous - run in blocking task)
        let graph = Arc::clone(&self.graph);
        let snapshot_gen = self.config.snapshot_generation;

        self.results = tokio::task::spawn_blocking(move || {
            graph.search_with_filter(
                &query_embedding,
                100, // k neighbors
                |node: &HnswNode<Memory>| {
                    // Visibility filter: only nodes committed before snapshot
                    node.generation <= snapshot_gen
                }
            )
        }).await?;

        Ok(())
    }

    /// Get next batch of results
    pub fn next_batch(&mut self) -> Option<Vec<Memory>> {
        if self.position >= self.results.len() {
            return None;
        }

        let end = (self.position + self.config.batch_size).min(self.results.len());
        let batch: Vec<Memory> = self.results[self.position..end]
            .iter()
            .map(|(node_id, _score)| {
                self.graph.get_node(*node_id).unwrap().data.clone()
            })
            .collect();

        self.position = end;

        Some(batch)
    }

    /// Check if more results are available
    pub fn has_more(&self) -> bool {
        self.position < self.results.len()
    }
}
```

4. **Implement gRPC handler** (extend `streaming.rs`):

```rust
impl StreamingHandlers {
    pub async fn handle_recall_stream(
        &self,
        request: Request<StreamingRecallRequest>,
    ) -> Result<Response<Streaming<StreamingRecallResponse>>, Status> {
        let req = request.into_inner();

        // Convert proto cue to core Cue
        let cue = CoreCue::try_from(req.cue.ok_or_else(||
            Status::invalid_argument("Missing cue"))?
        ).map_err(|e| Status::invalid_argument(format!("Invalid cue: {e}")))?;

        // Create incremental recall stream
        let mut recall_stream = IncrementalRecallStream::new(
            Arc::clone(&self.store.graph()),
            cue,
            &self.observation_queue,
            req.snapshot_isolation,
        );

        // Execute search
        recall_stream.search().await.map_err(|e|
            Status::internal(format!("Search failed: {e}"))
        )?;

        let snapshot_gen = recall_stream.config.snapshot_generation;

        // Create response stream
        let (tx, rx) = tokio::sync::mpsc::channel(10);

        tokio::spawn(async move {
            // Stream results in batches
            while let Some(batch) = recall_stream.next_batch() {
                let has_more = recall_stream.has_more();

                // Convert core Memory to proto Memory
                let proto_memories: Vec<engram_proto::Memory> = batch
                    .into_iter()
                    .map(|m| m.into())
                    .collect();

                let response = StreamingRecallResponse {
                    results: proto_memories,
                    more_results: has_more,
                    metadata: Some(RecallMetadata {
                        total_activated: recall_stream.results.len() as i32,
                        above_threshold: recall_stream.results.len() as i32,
                        avg_activation: 0.8,
                        recall_time_ms: 10,
                        activation_path: vec![],
                    }),
                    snapshot_sequence: snapshot_gen,
                };

                if tx.send(Ok(response)).await.is_err() {
                    break;
                }

                // Yield to allow other tasks to run
                tokio::task::yield_now().await;
            }
        });

        Ok(Response::new(ReceiverStream::new(rx)))
    }
}
```

5. **Modify HNSW search to support visibility filter** (`hnsw_search.rs`):

```rust
impl HnswGraph {
    /// Search with custom node filter (for snapshot isolation)
    pub fn search_with_filter<F>(
        &self,
        query: &[f32],
        k: usize,
        filter: F,
    ) -> Vec<(usize, f32)>
    where
        F: Fn(&HnswNode<Memory>) -> bool,
    {
        let mut candidates = Vec::new();

        // Standard HNSW search, but filter nodes before adding to results
        for layer in (0..self.max_layer).rev() {
            let neighbors = self.get_neighbors_at_layer(entry_point, layer);

            for &neighbor_id in neighbors {
                let node = &self.nodes[neighbor_id];

                // Apply visibility filter
                if !filter(node) {
                    continue; // Skip nodes not visible in snapshot
                }

                let score = self.distance(query, &node.data.embedding);
                candidates.push((neighbor_id, score));
            }
        }

        candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        candidates.truncate(k);
        candidates
    }
}
```

### Integration Points

1. **Worker Pool** (Task 003): Workers call `mark_generation_committed()` after HNSW insertion
2. **ObservationQueue**: Track committed generation for snapshot capture
3. **HNSW Index**: Add generation field to nodes, implement filtered search

### Dependencies

- **Blocks on:** Task 003 (Worker Pool) - workers must commit generations
- **Blocked by:** Task 005 - needs streaming handlers infrastructure

### Testing Strategy

**Unit Tests**:
```rust
#[test]
fn test_snapshot_generation_tracking() {
    let queue = ObservationQueue::new(QueueConfig::default());
    assert_eq!(queue.current_generation(), 0);

    queue.mark_generation_committed(5);
    assert_eq!(queue.current_generation(), 5);

    // Out-of-order commit (common in parallel workers)
    queue.mark_generation_committed(3);
    assert_eq!(queue.current_generation(), 5); // Should not regress
}

#[tokio::test]
async fn test_incremental_recall_batching() {
    let graph = create_test_graph_with_100_nodes();
    let queue = ObservationQueue::new(QueueConfig::default());
    queue.mark_generation_committed(100);

    let mut stream = IncrementalRecallStream::new(
        graph,
        test_cue(),
        &queue,
        false,
    );

    stream.search().await.unwrap();

    // Verify incremental batches
    let batch1 = stream.next_batch().unwrap();
    assert_eq!(batch1.len(), 10);
    assert!(stream.has_more());

    let batch2 = stream.next_batch().unwrap();
    assert_eq!(batch2.len(), 10);
}
```

**Integration Tests**:
```rust
#[tokio::test]
async fn test_snapshot_isolation_consistency() {
    // Store 100 observations (generations 1-100)
    // Mark first 50 as committed
    // Execute snapshot recall
    // Verify only observations 1-50 are visible

    let store = setup_test_store().await;
    for i in 1..=100 {
        store.observe(test_episode(i)).await.unwrap();
    }

    // Simulate only first 50 committed to index
    store.observation_queue().mark_generation_committed(50);

    let recall = store.streaming_recall(test_cue(), true).await.unwrap();
    let all_results: Vec<_> = recall.collect().await;

    // Should only see generations 1-50
    assert!(all_results.iter().all(|m| m.generation <= 50));
}

#[tokio::test]
async fn test_bounded_staleness_p99() {
    // Measure visibility latency: observation → visible in recall
    // Target: P99 < 100ms

    let store = setup_test_store_with_workers().await;
    let mut latencies = Vec::new();

    for i in 0..1000 {
        let start = Instant::now();
        store.observe(test_episode(i)).await.unwrap();

        // Poll until visible in recall
        loop {
            let results = store.recall(test_cue()).await.unwrap();
            if results.iter().any(|m| m.id == format!("test_{}", i)) {
                let latency = start.elapsed();
                latencies.push(latency);
                break;
            }
            tokio::time::sleep(Duration::from_millis(1)).await;
        }
    }

    latencies.sort();
    let p99 = latencies[(latencies.len() * 99) / 100];
    assert!(p99 < Duration::from_millis(100), "P99 latency: {:?}", p99);
}
```

**Performance Tests**:
```rust
#[tokio::test]
async fn test_incremental_streaming_first_result_latency() {
    // First result should be available in < 10ms
    let start = Instant::now();

    let mut recall_stream = client.recall_stream(request).await?.into_inner();
    let first_batch = recall_stream.next().await.unwrap()?;

    let latency = start.elapsed();
    assert!(latency < Duration::from_millis(10), "First result latency: {:?}", latency);
}
```

### Acceptance Criteria

1. **Functional:**
   - [ ] Recall sees all observations committed before snapshot
   - [ ] Recall does NOT see uncommitted observations (when `snapshot_isolation=true`)
   - [ ] Incremental streaming returns results in batches
   - [ ] First result arrives within 10ms of request

2. **Performance:**
   - [ ] Visibility latency P99 < 100ms (observation → recall)
   - [ ] First result latency P99 < 10ms
   - [ ] Full recall latency P99 < 100ms for 10K results

3. **Consistency:**
   - [ ] No phantom reads (observations appear/disappear during recall)
   - [ ] No lost observations (all committed observations eventually visible)
   - [ ] Confidence scores reflect staleness (recent observations = lower confidence)

---

## Critical Path and Dependencies

```
001 Protocol ──┐
002 Queue ─────┼──► 005 gRPC Streaming ──► 006 Backpressure ──► 007 Recall
003 Workers ───┤         │                        │                   │
004 Batch ─────┘         │                        │                   │
                         │                        │                   │
                         └────────────────────────┴───────────────────┘
                                  (all feed into)
                         010 Performance Benchmarking
```

**Parallelization:**
- Tasks 005 and 003 can be developed in parallel by separate engineers
- Task 006 blocks on both 005 (needs handlers) and 003 (needs workers for adaptive batching)
- Task 007 blocks on 005 (needs streaming infrastructure) and 003 (needs generation tracking)

**Recommended Sequencing:**
1. Week 1: Complete Task 003 (Worker Pool) - unblocks everything
2. Week 2: Complete Task 005 (gRPC Streaming) - enables testing
3. Week 2-3: Complete Tasks 006 and 007 in parallel - final integration

---

## Performance Targets Summary

| Metric | Task 005 | Task 006 | Task 007 |
|--------|----------|----------|----------|
| Throughput | 10K obs/sec | 100K obs/sec (with batching) | N/A |
| Latency (P99) | < 10ms (ack) | < 50ms (under backpressure) | < 100ms (visibility) |
| First result | N/A | N/A | < 10ms |
| Memory overhead | < 1MB per session | < 100KB (monitor) | < 1MB (snapshot) |
| CPU overhead | < 5% (handler) | < 1% (monitor) | < 10% (search) |

---

## Files to Create/Modify

### Create:
- `engram-cli/src/handlers/streaming.rs` (Task 005)
- `engram-core/src/streaming/backpressure.rs` (Task 006)
- `engram-core/src/streaming/recall.rs` (Task 007)
- `engram-core/tests/integration/streaming_grpc.rs` (Task 005)
- `engram-core/tests/integration/backpressure.rs` (Task 006)
- `engram-core/tests/integration/snapshot_isolation.rs` (Task 007)

### Modify:
- `engram-cli/src/grpc.rs` - Add streaming method impls
- `engram-core/src/streaming/mod.rs` - Export new modules
- `engram-core/src/index/hnsw_node.rs` - Add generation field
- `engram-core/src/index/hnsw_search.rs` - Add filtered search
- `engram-core/src/streaming/observation_queue.rs` - Add generation tracking
- `engram-core/src/store.rs` - Integrate streaming components

---

## Next Steps

1. **Create individual task files** for 005, 006, 007 with detailed specifications
2. **Review with team** - Verify dependencies and sequencing
3. **Start Task 003** (Worker Pool) - Critical path blocker
4. **Prototype streaming handler** (Task 005) - Validate protocol design
5. **Benchmark visibility latency** - Validate 100ms P99 is achievable

---

**End of Implementation Guidance Document**
