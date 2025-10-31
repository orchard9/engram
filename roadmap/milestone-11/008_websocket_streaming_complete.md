# Task 008: WebSocket Streaming

## Objective

Add WebSocket endpoint for browser clients, mirroring gRPC streaming functionality with same flow control semantics.

## Context

Browsers cannot use gRPC bidirectional streaming directly. WebSocket provides a standard browser-compatible protocol for real-time bidirectional communication. This task implements a WebSocket endpoint that wraps the same session management and observation queue infrastructure used by the gRPC handlers.

## Deliverables

1. **WebSocket Handler** (`engram-cli/src/handlers/websocket.rs`)
   - WebSocket upgrade endpoint at `/v1/stream`
   - JSON message serialization/deserialization (compatible with protobuf schema)
   - Integration with `SessionManager` and `ObservationQueue`
   - Flow control message handling (pause/resume)
   - Heartbeat/keepalive every 30s
   - Session recovery support

2. **Client Example** (`examples/streaming/typescript_client.ts`)
   - Browser-compatible WebSocket client
   - Auto-reconnect with exponential backoff
   - Session recovery on reconnect
   - Flow control implementation

3. **Tests**
   - WebSocket connection establishment
   - Observation streaming with acks
   - Flow control (pause/resume)
   - Auto-reconnect behavior
   - Performance: 10K observations/sec per connection

## Technical Approach

### Message Protocol

Messages are JSON-encoded, compatible with the protobuf schema:

```json
// Client → Server: Stream initialization
{
  "type": "init",
  "memory_space_id": "default",
  "client_buffer_size": 1000,
  "enable_backpressure": true
}

// Server → Client: Initialization acknowledgment
{
  "type": "init_ack",
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "initial_sequence": 0,
  "capabilities": {
    "max_observations_per_second": 100000,
    "queue_capacity": 10000,
    "supports_backpressure": true,
    "supports_snapshot_isolation": true
  }
}

// Client → Server: Observation
{
  "type": "observation",
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "sequence_number": 1,
  "episode": {
    "id": "episode_001",
    "when": "2024-10-30T12:00:00Z",
    "what": "User clicked checkout button",
    "embedding": [0.1, 0.2, ...],  // 768 dimensions
    "encoding_confidence": 0.85
  }
}

// Server → Client: Acknowledgment
{
  "type": "ack",
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "sequence_number": 1,
  "status": "accepted",
  "memory_id": "mem_1",
  "committed_at": "2024-10-30T12:00:00.123Z"
}

// Client → Server: Flow control
{
  "type": "flow_control",
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "action": "pause"  // or "resume"
}

// Server → Client: Heartbeat
{
  "type": "heartbeat",
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "timestamp": "2024-10-30T12:00:30Z"
}
```

### WebSocket Handler Architecture

```rust
use axum::{
    extract::{
        ws::{Message, WebSocket, WebSocketUpgrade},
        State,
    },
    response::IntoResponse,
};
use serde::{Deserialize, Serialize};
use tokio::time::{interval, Duration};

pub async fn websocket_handler(
    ws: WebSocketUpgrade,
    State(state): State<Arc<ApiState>>,
) -> impl IntoResponse {
    ws.on_upgrade(|socket| handle_socket(socket, state))
}

async fn handle_socket(socket: WebSocket, state: Arc<ApiState>) {
    // Split socket into sender/receiver
    let (mut sender, mut receiver) = socket.split();

    // Create heartbeat task
    let heartbeat_task = spawn_heartbeat(sender.clone(), 30);

    // Process incoming messages
    while let Some(msg) = receiver.next().await {
        match msg {
            Ok(Message::Text(text)) => {
                // Parse JSON message
                // Route to appropriate handler
                // Send response
            }
            Ok(Message::Close(_)) => break,
            _ => continue,
        }
    }

    // Cleanup
    heartbeat_task.abort();
}
```

### Session Recovery

When client reconnects after disconnect:

1. Client sends `init` with optional `resume_session_id`
2. Server checks if session still exists in `SessionManager`
3. If yes: return existing session ID and last sequence number
4. If no: create new session and return sequence 0

### Integration Points

Reuse existing infrastructure from Task 005:
- `SessionManager`: Session lifecycle management
- `ObservationQueue`: Lock-free observation queuing
- `StreamingHandlers`: Convert between JSON and core types

## Acceptance Criteria

- [ ] WebSocket endpoint at `/v1/stream` accepts connections
- [ ] Browser client can stream 1K observations, receive acks
- [ ] Flow control: client pauses → server stops processing → client resumes
- [ ] Heartbeat messages sent every 30s
- [ ] Auto-reconnect: disconnect mid-stream → reconnect → resume from last sequence
- [ ] Performance: sustained 10K observations/sec per WebSocket connection
- [ ] No memory leaks: 100 concurrent connections for 60s, graceful cleanup

## Files to Create

- `engram-cli/src/handlers/websocket.rs` - WebSocket handler implementation
- `examples/streaming/typescript_client.ts` - Browser client example
- `engram-cli/tests/websocket_streaming_test.rs` - Integration tests

## Files to Modify

- `engram-cli/Cargo.toml` - Add `axum` WebSocket dependencies
- `engram-cli/src/handlers/mod.rs` - Export `websocket` module
- `engram-cli/src/api.rs` - Add `/v1/stream` route

## Testing Approach

1. **Unit Tests**: Message serialization/deserialization
2. **Integration Tests**:
   - Connection lifecycle (connect → observe → close)
   - Flow control behavior
   - Session recovery
3. **Load Test**: 10 concurrent connections, 10K obs/sec each, 60s duration
4. **Browser Test**: Manual verification with TypeScript client

## Dependencies

- Task 005 (gRPC Streaming): Reuses `SessionManager` and `ObservationQueue`
- Axum WebSocket support (already in dependencies)

## Estimated Effort

2 days (per README.md)

## Notes

- WebSocket is designed for browser clients, not high-throughput scenarios
- gRPC remains the preferred protocol for server-to-server streaming
- Target 100 concurrent WebSocket connections, not 10K
- Document performance characteristics clearly in API docs
