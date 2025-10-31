//! WebSocket streaming handler for browser clients.
//!
//! Provides browser-compatible streaming interface that mirrors gRPC functionality
//! with JSON message serialization. Integrates with the same session management
//! and observation queue infrastructure as gRPC handlers.
//!
//! ## Protocol
//!
//! Messages are JSON-encoded objects compatible with protobuf schema:
//!
//! ```json
//! // Client → Server: Initialize stream
//! {
//!   "type": "init",
//!   "memory_space_id": "default",
//!   "client_buffer_size": 1000,
//!   "enable_backpressure": true
//! }
//!
//! // Server → Client: Initialization ack
//! {
//!   "type": "init_ack",
//!   "session_id": "uuid",
//!   "initial_sequence": 0,
//!   "capabilities": {...}
//! }
//!
//! // Client → Server: Observation
//! {
//!   "type": "observation",
//!   "session_id": "uuid",
//!   "sequence_number": 1,
//!   "episode": {...}
//! }
//!
//! // Server → Client: Acknowledgment
//! {
//!   "type": "ack",
//!   "session_id": "uuid",
//!   "sequence_number": 1,
//!   "status": "accepted",
//!   "memory_id": "mem_1"
//! }
//! ```

use axum::{
    extract::{
        State,
        ws::{Message, WebSocket, WebSocketUpgrade},
    },
    response::IntoResponse,
};
use engram_core::{
    Confidence as CoreConfidence, Episode, MemorySpaceId,
    streaming::{ObservationPriority, SessionState},
};
use futures::{SinkExt, StreamExt};
use serde::{Deserialize, Serialize};
use tokio::time::{Duration, interval};
use tracing::{debug, error, info};

use crate::api::ApiState;

/// WebSocket handler endpoint.
///
/// Upgrades HTTP connection to WebSocket and spawns async handler task.
///
/// # Endpoint
///
/// `GET /v1/stream` with `Upgrade: websocket` header
#[allow(clippy::unused_async)] // Required for axum handler
pub async fn websocket_handler(
    ws: WebSocketUpgrade,
    State(state): State<ApiState>,
) -> impl IntoResponse {
    ws.on_upgrade(move |socket| handle_socket(socket, state))
}

/// Main WebSocket message loop.
///
/// Manages:
/// - Message routing (init, observation, flow control, close)
/// - Heartbeat keepalive (30s interval)
/// - Session lifecycle
/// - Error handling and graceful shutdown
async fn handle_socket(socket: WebSocket, state: ApiState) {
    let (mut sender, mut receiver) = socket.split();
    let mut session_id: Option<String> = None;

    // Spawn heartbeat task
    let (heartbeat_tx, mut heartbeat_rx) = tokio::sync::mpsc::channel::<HeartbeatMessage>(1);
    let heartbeat_handle = tokio::spawn(async move {
        let mut ticker = interval(Duration::from_secs(30));
        ticker.tick().await; // Skip the immediate first tick
        loop {
            ticker.tick().await;
            let msg = HeartbeatMessage {
                r#type: "heartbeat".to_string(),
                session_id: None,
                timestamp: chrono::Utc::now(),
            };
            if heartbeat_tx.send(msg).await.is_err() {
                break;
            }
        }
    });

    // Message processing loop
    loop {
        tokio::select! {
            // Handle incoming WebSocket messages
            msg = receiver.next() => {
                match msg {
                    Some(Ok(Message::Text(text))) => {
                        match handle_text_message(
                            &text,
                            &state,
                            &mut session_id,
                        ) {
                            Ok(Some(response)) => {
                                let json = match serde_json::to_string(&response) {
                                    Ok(j) => j,
                                    Err(e) => {
                                        error!("Failed to serialize response: {}", e);
                                        continue;
                                    }
                                };
                                if sender.send(Message::Text(json.into())).await.is_err() {
                                    debug!("Failed to send response, client disconnected");
                                    break;
                                }
                            }
                            Ok(None) => {
                                // No response needed
                            }
                            Err(error_msg) => {
                                let error_response = ErrorResponse {
                                    r#type: "error".to_string(),
                                    message: error_msg,
                                    session_id: session_id.clone(),
                                };
                                let json = serde_json::to_string(&error_response).unwrap_or_default();
                                let _ = sender.send(Message::Text(json.into())).await;
                            }
                        }
                    }
                    Some(Ok(Message::Close(_))) => {
                        debug!("Client sent close message");
                        break;
                    }
                    Some(Ok(Message::Ping(data))) => {
                        let _ = sender.send(Message::Pong(data)).await;
                    }
                    Some(Ok(_)) => {
                        // Ignore other message types (Binary, Pong)
                    }
                    Some(Err(e)) => {
                        error!("WebSocket error: {}", e);
                        break;
                    }
                    None => {
                        debug!("WebSocket stream ended");
                        break;
                    }
                }
            }

            // Send heartbeat messages
            heartbeat = heartbeat_rx.recv() => {
                if let Some(mut hb) = heartbeat {
                    hb.session_id.clone_from(&session_id);
                    let json = match serde_json::to_string(&hb) {
                        Ok(j) => j,
                        Err(e) => {
                            error!("Failed to serialize heartbeat: {}", e);
                            continue;
                        }
                    };
                    if sender.send(Message::Text(json.into())).await.is_err() {
                        debug!("Failed to send heartbeat, client disconnected");
                        break;
                    }
                }
            }
        }
    }

    // Cleanup: close session in SessionManager
    if let Some(sid) = session_id {
        if let Err(e) = state.session_manager.close_session(&sid) {
            error!("Error closing WebSocket session {}: {}", sid, e);
        } else {
            info!("WebSocket session {} closed", sid);
        }
    }

    // Stop heartbeat task
    heartbeat_handle.abort();
}

/// Handle incoming text message and route to appropriate handler.
#[allow(clippy::too_many_lines)]
fn handle_text_message(
    text: &str,
    state: &ApiState,
    session_id: &mut Option<String>,
) -> Result<Option<WsResponse>, String> {
    // Parse message type first
    let msg_value: serde_json::Value =
        serde_json::from_str(text).map_err(|e| format!("Invalid JSON: {e}"))?;

    let msg_type = msg_value
        .get("type")
        .and_then(|v| v.as_str())
        .ok_or_else(|| "Missing 'type' field".to_string())?;

    match msg_type {
        "init" => {
            let req: InitRequest =
                serde_json::from_str(text).map_err(|e| format!("Invalid init request: {e}"))?;
            handle_init(&req, state, session_id)
        }
        "observation" => {
            let req: ObservationRequest = serde_json::from_str(text)
                .map_err(|e| format!("Invalid observation request: {e}"))?;
            handle_observation(req, state, session_id.as_ref())
        }
        "flow_control" => {
            let req: FlowControlRequest = serde_json::from_str(text)
                .map_err(|e| format!("Invalid flow control request: {e}"))?;
            handle_flow_control(&req, state, session_id.as_ref())
        }
        "close" => Ok(handle_close(state, session_id.as_ref())),
        _ => Err(format!("Unknown message type: {msg_type}")),
    }
}

/// Handle stream initialization.
fn handle_init(
    req: &InitRequest,
    state: &ApiState,
    session_id: &mut Option<String>,
) -> Result<Option<WsResponse>, String> {
    // Validate memory space ID
    let memory_space_id = MemorySpaceId::try_from(req.memory_space_id.as_str())
        .map_err(|e| format!("Invalid memory_space_id: {e}"))?;

    // Create session using SessionManager
    let new_session_id = uuid::Uuid::new_v4().to_string();
    let _session = state
        .session_manager
        .create_session(new_session_id.clone(), memory_space_id);
    *session_id = Some(new_session_id.clone());

    // Get real capabilities from system
    let capabilities = Capabilities {
        max_observations_per_second: 100_000,
        queue_capacity: u32::try_from(state.observation_queue.total_capacity()).unwrap_or(u32::MAX),
        supports_backpressure: true,
        supports_snapshot_isolation: true,
    };

    Ok(Some(WsResponse::InitAck {
        r#type: "init_ack".to_string(),
        session_id: new_session_id,
        initial_sequence: 0,
        capabilities,
    }))
}

/// Handle observation message.
#[allow(clippy::too_many_lines)]
fn handle_observation(
    req: ObservationRequest,
    state: &ApiState,
    session_id: Option<&String>,
) -> Result<Option<WsResponse>, String> {
    // Validate session exists
    let Some(sid) = session_id else {
        return Err("Must send 'init' before observations".to_string());
    };

    // Validate session ID matches
    if req.session_id != *sid {
        return Err(format!(
            "Session ID mismatch: expected {}, got {}",
            sid, req.session_id
        ));
    }

    // Get session from manager and validate sequence
    let session = state
        .session_manager
        .get_session(sid)
        .map_err(|e| format!("Session error: {e}"))?;

    session
        .validate_sequence(req.sequence_number)
        .map_err(|e| format!("Sequence validation failed: {e}"))?;

    // Convert episode to core type
    let embedding: [f32; 768] = req
        .episode
        .embedding
        .try_into()
        .map_err(|_| "Embedding must be exactly 768 dimensions".to_string())?;

    let confidence = CoreConfidence::exact(req.episode.encoding_confidence.unwrap_or(0.7));

    let when = req.episode.when.map_or_else(chrono::Utc::now, |ts| ts);

    let episode_core = Episode::new(
        req.episode.id.clone(),
        when,
        req.episode.what,
        embedding,
        confidence,
    );

    // Enqueue observation to ObservationQueue
    let memory_space_id = session.memory_space_id().clone();

    state
        .observation_queue
        .enqueue(
            memory_space_id,
            episode_core,
            req.sequence_number,
            ObservationPriority::Normal,
        )
        .map_err(|e| format!("Queue error: {e}"))?;

    let ack = AckResponse {
        r#type: "ack".to_string(),
        session_id: sid.clone(),
        sequence_number: req.sequence_number,
        status: "accepted".to_string(),
        memory_id: req.episode.id,
        committed_at: chrono::Utc::now(),
    };

    Ok(Some(WsResponse::Ack(ack)))
}

/// Handle flow control message.
fn handle_flow_control(
    req: &FlowControlRequest,
    state: &ApiState,
    session_id: Option<&String>,
) -> Result<Option<WsResponse>, String> {
    // Validate session exists
    let Some(sid) = session_id else {
        return Err("Must send 'init' before flow control".to_string());
    };

    // Validate session ID matches
    if req.session_id != *sid {
        return Err(format!(
            "Session ID mismatch: expected {}, got {}",
            sid, req.session_id
        ));
    }

    // Get session from manager
    let session = state
        .session_manager
        .get_session(sid)
        .map_err(|e| format!("Session error: {e}"))?;

    // Update session state based on action
    match req.action.as_str() {
        "pause" => {
            session.set_state(SessionState::Paused);
            info!("Session {} paused", sid);
        }
        "resume" => {
            session.set_state(SessionState::Active);
            info!("Session {} resumed", sid);
        }
        _ => {
            return Err(format!("Unknown flow control action: {}", req.action));
        }
    }

    Ok(None) // No response needed for flow control
}

/// Handle stream close.
fn handle_close(state: &ApiState, session_id: Option<&String>) -> Option<WsResponse> {
    if let Some(sid) = session_id {
        // Close session in SessionManager
        if let Err(e) = state.session_manager.close_session(sid) {
            error!("Error closing session {}: {}", sid, e);
        } else {
            info!("Closed session {}", sid);
        }
    }

    None // Connection will be closed by caller
}

// ============================================================================
// Message Types (JSON-serializable)
// ============================================================================

/// Stream initialization request from client.
#[derive(Debug, Deserialize)]
struct InitRequest {
    #[allow(dead_code)]
    r#type: String,
    memory_space_id: String,
    #[allow(dead_code)]
    client_buffer_size: Option<u32>,
    #[allow(dead_code)]
    enable_backpressure: Option<bool>,
}

/// Observation request from client.
#[derive(Debug, Deserialize)]
struct ObservationRequest {
    #[allow(dead_code)]
    r#type: String,
    session_id: String,
    sequence_number: u64,
    episode: EpisodeJson,
}

/// Episode data in JSON format.
#[derive(Debug, Deserialize)]
struct EpisodeJson {
    id: String,
    when: Option<chrono::DateTime<chrono::Utc>>,
    what: String,
    embedding: Vec<f32>,
    encoding_confidence: Option<f32>,
}

/// Flow control request from client.
#[derive(Debug, Deserialize)]
struct FlowControlRequest {
    #[allow(dead_code)]
    r#type: String,
    session_id: String,
    action: String, // "pause" or "resume"
}

/// WebSocket response types.
#[derive(Debug, Serialize)]
#[serde(untagged)]
enum WsResponse {
    InitAck {
        r#type: String,
        session_id: String,
        initial_sequence: u64,
        capabilities: Capabilities,
    },
    Ack(AckResponse),
}

/// Acknowledgment response.
#[derive(Debug, Serialize)]
struct AckResponse {
    r#type: String,
    session_id: String,
    sequence_number: u64,
    status: String,
    memory_id: String,
    committed_at: chrono::DateTime<chrono::Utc>,
}

/// Server capabilities.
#[derive(Debug, Serialize)]
struct Capabilities {
    max_observations_per_second: u32,
    queue_capacity: u32,
    supports_backpressure: bool,
    supports_snapshot_isolation: bool,
}

/// Heartbeat message sent every 30s.
#[derive(Debug, Serialize)]
struct HeartbeatMessage {
    r#type: String,
    session_id: Option<String>,
    timestamp: chrono::DateTime<chrono::Utc>,
}

/// Error response.
#[derive(Debug, Serialize)]
struct ErrorResponse {
    r#type: String,
    message: String,
    session_id: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_init_request_deserialization() {
        let json = r#"{
            "type": "init",
            "memory_space_id": "default",
            "client_buffer_size": 1000,
            "enable_backpressure": true
        }"#;

        let req: InitRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.r#type, "init");
        assert_eq!(req.memory_space_id, "default");
    }

    #[test]
    fn test_observation_request_deserialization() {
        let json = r#"{
            "type": "observation",
            "session_id": "test-session",
            "sequence_number": 1,
            "episode": {
                "id": "ep1",
                "what": "test event",
                "embedding": [0.1, 0.2, 0.3],
                "encoding_confidence": 0.85
            }
        }"#;

        let req: ObservationRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.r#type, "observation");
        assert_eq!(req.session_id, "test-session");
        assert_eq!(req.sequence_number, 1);
        assert_eq!(req.episode.id, "ep1");
    }

    #[test]
    fn test_ack_response_serialization() {
        let ack = AckResponse {
            r#type: "ack".to_string(),
            session_id: "test-session".to_string(),
            sequence_number: 1,
            status: "accepted".to_string(),
            memory_id: "mem_1".to_string(),
            committed_at: chrono::Utc::now(),
        };

        let json = serde_json::to_string(&ack).unwrap();
        assert!(json.contains("\"type\":\"ack\""));
        assert!(json.contains("\"session_id\":\"test-session\""));
    }
}
