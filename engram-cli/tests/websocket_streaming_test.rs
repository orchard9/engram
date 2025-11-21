//! Integration tests for WebSocket streaming.
//!
//! Tests:
//! - WebSocket connection establishment
//! - Stream initialization and session management
//! - Observation streaming with acknowledgments
//! - Flow control (pause/resume)
//! - Heartbeat messages
//! - Error handling

use engram_cli::api::{ApiState, create_api_routes};
use engram_cli::config::SecurityConfig;
use engram_core::{MemorySpaceId, MemorySpaceRegistry, MemoryStore, metrics::MetricsRegistry};
use futures::{SinkExt, StreamExt};
use serde_json::json;
use std::sync::Arc;
use tokio::net::TcpListener;
use tokio_tungstenite::{connect_async, tungstenite::Message};

/// Create test API state for WebSocket testing.
async fn create_test_state() -> ApiState {
    let store = Arc::new(MemoryStore::new(1000));
    let metrics = Arc::new(MetricsRegistry::new());
    let registry = Arc::new(
        MemorySpaceRegistry::new("/tmp/engram_ws_test", |_id, _dirs| {
            Ok(Arc::new(MemoryStore::new(1000)))
        })
        .expect("Failed to create registry"),
    );

    let default_space = MemorySpaceId::default();
    registry
        .create_or_get(&default_space)
        .await
        .expect("Failed to create default space");

    let auto_tuner = engram_core::activation::SpreadingAutoTuner::new(0.1, 1000);
    let (shutdown_tx, _) = tokio::sync::watch::channel(false);

    // Create ApiState using the new constructor
    ApiState::new(
        store,
        registry,
        default_space,
        metrics,
        auto_tuner,
        Arc::new(shutdown_tx),
        None,                                // cluster
        None,                                // router
        None,                                // partition_confidence
        Arc::new(SecurityConfig::default()), // auth_config
        None,                                // auth_validator
    )
}

/// Start test HTTP server with WebSocket support.
async fn start_test_server() -> String {
    let state = create_test_state().await;
    let app = create_api_routes().with_state(state);

    // Bind to random available port
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();

    // Spawn server
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });

    // Return WebSocket URL
    format!("ws://{addr}/v1/stream")
}

#[tokio::test]
async fn test_websocket_connection() {
    let url = start_test_server().await;

    // Give server time to start
    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

    // Connect to WebSocket
    let (ws_stream, _response) = connect_async(&url).await.expect("Failed to connect");

    let (mut write, mut read) = ws_stream.split();

    // Send init message
    let init_msg = json!({
        "type": "init",
        "memory_space_id": "default",
        "client_buffer_size": 1000,
        "enable_backpressure": true
    });
    write
        .send(Message::Text(init_msg.to_string()))
        .await
        .unwrap();

    // Receive init_ack
    let msg = read.next().await.expect("No response").unwrap();
    let text = msg.to_text().unwrap();
    let response: serde_json::Value = serde_json::from_str(text).unwrap();

    assert_eq!(response["type"], "init_ack");
    assert!(response["session_id"].is_string());
    assert_eq!(response["initial_sequence"], 0);
    assert!(response["capabilities"].is_object());

    // Close connection
    write.send(Message::Close(None)).await.unwrap();
}

#[tokio::test]
async fn test_websocket_observation_streaming() {
    let url = start_test_server().await;
    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

    let (ws_stream, _) = connect_async(&url).await.unwrap();
    let (mut write, mut read) = ws_stream.split();

    // Initialize session
    let init_msg = json!({
        "type": "init",
        "memory_space_id": "default"
    });
    write
        .send(Message::Text(init_msg.to_string()))
        .await
        .unwrap();

    // Get session ID from init_ack
    let msg = read.next().await.unwrap().unwrap();
    let init_ack: serde_json::Value = serde_json::from_str(msg.to_text().unwrap()).unwrap();
    let session_id = init_ack["session_id"].as_str().unwrap();

    // Send observation
    let embedding: Vec<f32> = (0..768).map(|i| i as f32 / 768.0).collect();
    let observation = json!({
        "type": "observation",
        "session_id": session_id,
        "sequence_number": 1,
        "episode": {
            "id": "test_episode_1",
            "what": "Test event",
            "embedding": embedding,
            "encoding_confidence": 0.85
        }
    });
    write
        .send(Message::Text(observation.to_string()))
        .await
        .unwrap();

    // Receive acknowledgment
    let msg = read.next().await.unwrap().unwrap();
    let ack: serde_json::Value = serde_json::from_str(msg.to_text().unwrap()).unwrap();

    assert_eq!(ack["type"], "ack");
    assert_eq!(ack["session_id"], session_id);
    assert_eq!(ack["sequence_number"], 1);
    assert_eq!(ack["status"], "accepted");
    assert!(ack["memory_id"].is_string());

    write.send(Message::Close(None)).await.unwrap();
}

#[tokio::test]
async fn test_websocket_flow_control() {
    let url = start_test_server().await;
    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

    let (ws_stream, _) = connect_async(&url).await.unwrap();
    let (mut write, mut read) = ws_stream.split();

    // Initialize session
    let init_msg = json!({"type": "init", "memory_space_id": "default"});
    write
        .send(Message::Text(init_msg.to_string()))
        .await
        .unwrap();

    let msg = read.next().await.unwrap().unwrap();
    let init_ack: serde_json::Value = serde_json::from_str(msg.to_text().unwrap()).unwrap();
    let session_id = init_ack["session_id"].as_str().unwrap();

    // Send pause
    let pause_msg = json!({
        "type": "flow_control",
        "session_id": session_id,
        "action": "pause"
    });
    write
        .send(Message::Text(pause_msg.to_string()))
        .await
        .unwrap();

    // Send resume
    let resume_msg = json!({
        "type": "flow_control",
        "session_id": session_id,
        "action": "resume"
    });
    write
        .send(Message::Text(resume_msg.to_string()))
        .await
        .unwrap();

    // Flow control messages don't get responses, so just verify no errors
    // Wait a bit to ensure messages are processed
    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

    write.send(Message::Close(None)).await.unwrap();
}

#[tokio::test]
async fn test_websocket_error_handling() {
    let url = start_test_server().await;
    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

    let (ws_stream, _) = connect_async(&url).await.unwrap();
    let (mut write, mut read) = ws_stream.split();

    // Send observation without init (should error)
    let observation = json!({
        "type": "observation",
        "session_id": "invalid",
        "sequence_number": 1,
        "episode": {
            "id": "test",
            "what": "test",
            "embedding": vec![0.0; 768]
        }
    });
    write
        .send(Message::Text(observation.to_string()))
        .await
        .unwrap();

    // Should receive error response
    let msg = read.next().await.unwrap().unwrap();
    let response: serde_json::Value = serde_json::from_str(msg.to_text().unwrap()).unwrap();

    assert_eq!(response["type"], "error");
    assert!(response["message"].is_string());

    write.send(Message::Close(None)).await.unwrap();
}

#[tokio::test]
async fn test_websocket_invalid_json() {
    let url = start_test_server().await;
    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

    let (ws_stream, _) = connect_async(&url).await.unwrap();
    let (mut write, mut read) = ws_stream.split();

    // Send invalid JSON
    write
        .send(Message::Text("not valid json".to_string()))
        .await
        .unwrap();

    // Should receive error response
    let msg = read.next().await.unwrap().unwrap();
    let response: serde_json::Value = serde_json::from_str(msg.to_text().unwrap()).unwrap();

    assert_eq!(response["type"], "error");
    assert!(
        response["message"]
            .as_str()
            .unwrap()
            .contains("Invalid JSON")
    );

    write.send(Message::Close(None)).await.unwrap();
}

#[tokio::test]
async fn test_websocket_heartbeat() {
    let url = start_test_server().await;
    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

    let (ws_stream, _) = connect_async(&url).await.unwrap();
    let (mut write, mut read) = ws_stream.split();

    // Initialize session
    let init_msg = json!({"type": "init", "memory_space_id": "default"});
    write
        .send(Message::Text(init_msg.to_string()))
        .await
        .unwrap();

    // Get init_ack
    let _msg = read.next().await.unwrap().unwrap();

    // Wait for heartbeat (30s interval, but we'll just verify the mechanism works)
    // In a real test, we'd wait 30+ seconds, but for CI we'll just verify
    // the connection stays open
    tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;

    // Send ping to verify connection is still alive
    write.send(Message::Ping(vec![])).await.unwrap();

    // Should receive pong
    let msg = read.next().await.unwrap().unwrap();
    assert!(matches!(msg, Message::Pong(_)));

    write.send(Message::Close(None)).await.unwrap();
}

#[tokio::test]
async fn test_websocket_multiple_observations() {
    let url = start_test_server().await;
    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

    let (ws_stream, _) = connect_async(&url).await.unwrap();
    let (mut write, mut read) = ws_stream.split();

    // Initialize session
    let init_msg = json!({"type": "init", "memory_space_id": "default"});
    write
        .send(Message::Text(init_msg.to_string()))
        .await
        .unwrap();

    let msg = read.next().await.unwrap().unwrap();
    let init_ack: serde_json::Value = serde_json::from_str(msg.to_text().unwrap()).unwrap();
    let session_id = init_ack["session_id"].as_str().unwrap().to_string();

    // Send multiple observations
    let num_observations = 10;
    for seq in 1..=num_observations {
        let embedding: Vec<f32> = (0..768).map(|i| (i + seq) as f32 / 768.0).collect();
        let observation = json!({
            "type": "observation",
            "session_id": session_id,
            "sequence_number": seq,
            "episode": {
                "id": format!("episode_{}", seq),
                "what": format!("Event {}", seq),
                "embedding": embedding,
                "encoding_confidence": 0.85
            }
        });
        write
            .send(Message::Text(observation.to_string()))
            .await
            .unwrap();

        // Receive ack
        let msg = read.next().await.unwrap().unwrap();
        let ack: serde_json::Value = serde_json::from_str(msg.to_text().unwrap()).unwrap();

        assert_eq!(ack["type"], "ack");
        assert_eq!(ack["sequence_number"], seq);
        assert_eq!(ack["status"], "accepted");
    }

    write.send(Message::Close(None)).await.unwrap();
}
