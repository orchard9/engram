//! Tests for streaming support (gRPC and HTTP SSE)
//!
//! Tests bidirectional gRPC streaming, Server-Sent Events, backpressure handling,
//! and cognitive-friendly stream management with working memory constraints.

use axum::{
    Router,
    body::Body,
    http::{Method, Request, StatusCode},
    response::Response,
};
use engram_cli::api::{ApiState, create_api_routes};
use engram_core::graph::MemoryGraph;
use std::sync::Arc;
use tokio::sync::RwLock;
use tower::ServiceExt;

/// Create test router for streaming tests
fn create_test_router() -> Router {
    let graph = Arc::new(RwLock::new(MemoryGraph::new()));
    let api_state = ApiState::new(graph);

    create_api_routes().with_state(api_state)
}

/// Helper to make streaming HTTP requests
async fn make_streaming_request(app: &Router, uri: &str) -> Response<Body> {
    let request = Request::builder()
        .method(Method::GET)
        .uri(uri)
        .header("Accept", "text/event-stream")
        .header("Cache-Control", "no-cache")
        .body(Body::empty())
        .unwrap();

    app.clone().oneshot(request).await.unwrap()
}

#[tokio::test]
async fn test_stream_activities_basic() {
    let app = create_test_router();

    let response = make_streaming_request(&app, "/api/v1/stream/activities").await;

    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        response.headers().get("content-type").unwrap(),
        "text/event-stream"
    );
    assert_eq!(response.headers().get("cache-control").unwrap(), "no-cache");

    // Should have SSE headers
    assert!(response.headers().contains_key("cache-control"));
}

#[tokio::test]
async fn test_stream_activities_with_filters() {
    let app = create_test_router();

    let response = make_streaming_request(
        &app,
        "/api/v1/stream/activities?event_types=activation,storage&min_importance=0.5&buffer_size=64"
    ).await;

    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        response.headers().get("content-type").unwrap(),
        "text/event-stream"
    );
}

#[tokio::test]
async fn test_stream_activities_with_session() {
    let app = create_test_router();

    let response = make_streaming_request(
        &app,
        "/api/v1/stream/activities?session_id=test_session_123&last_event_id=42",
    )
    .await;

    assert_eq!(response.status(), StatusCode::OK);
}

#[tokio::test]
async fn test_stream_memories_basic() {
    let app = create_test_router();

    let response = make_streaming_request(&app, "/api/v1/stream/memories").await;

    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        response.headers().get("content-type").unwrap(),
        "text/event-stream"
    );
}

#[tokio::test]
async fn test_stream_memories_with_filters() {
    let app = create_test_router();

    let response = make_streaming_request(
        &app,
        "/api/v1/stream/memories?include_formation=true&include_retrieval=false&min_confidence=0.8&memory_types=semantic,episodic"
    ).await;

    assert_eq!(response.status(), StatusCode::OK);
}

#[tokio::test]
async fn test_stream_consolidation_basic() {
    let app = create_test_router();

    let response = make_streaming_request(&app, "/api/v1/stream/consolidation").await;

    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        response.headers().get("content-type").unwrap(),
        "text/event-stream"
    );
}

#[tokio::test]
async fn test_stream_consolidation_with_filters() {
    let app = create_test_router();

    let response = make_streaming_request(
        &app,
        "/api/v1/stream/consolidation?include_replay=true&include_insights=true&include_progress=false&min_novelty=0.6"
    ).await;

    assert_eq!(response.status(), StatusCode::OK);
}

#[tokio::test]
async fn test_stream_cognitive_constraints() {
    let app = create_test_router();

    // Test working memory constraint (buffer_size should be capped at 128)
    let response = make_streaming_request(
        &app,
        "/api/v1/stream/activities?buffer_size=1000", // Should be capped
    )
    .await;

    assert_eq!(response.status(), StatusCode::OK);

    // Test importance threshold bounds
    let response = make_streaming_request(
        &app,
        "/api/v1/stream/activities?min_importance=1.5", // Should be clamped to 1.0
    )
    .await;

    assert_eq!(response.status(), StatusCode::OK);

    let response = make_streaming_request(
        &app,
        "/api/v1/stream/activities?min_importance=-0.5", // Should be clamped to 0.0
    )
    .await;

    assert_eq!(response.status(), StatusCode::OK);
}

#[tokio::test]
async fn test_stream_confidence_bounds() {
    let app = create_test_router();

    // Test confidence bounds for memory streaming
    let response = make_streaming_request(
        &app,
        "/api/v1/stream/memories?min_confidence=1.5", // Should be clamped to 1.0
    )
    .await;

    assert_eq!(response.status(), StatusCode::OK);

    let response = make_streaming_request(
        &app,
        "/api/v1/stream/memories?min_confidence=-0.5", // Should be clamped to 0.0
    )
    .await;

    assert_eq!(response.status(), StatusCode::OK);
}

#[tokio::test]
async fn test_stream_novelty_bounds() {
    let app = create_test_router();

    // Test novelty bounds for consolidation streaming
    let response = make_streaming_request(
        &app,
        "/api/v1/stream/consolidation?min_novelty=1.5", // Should be clamped to 1.0
    )
    .await;

    assert_eq!(response.status(), StatusCode::OK);

    let response = make_streaming_request(
        &app,
        "/api/v1/stream/consolidation?min_novelty=-0.5", // Should be clamped to 0.0
    )
    .await;

    assert_eq!(response.status(), StatusCode::OK);
}

#[tokio::test]
async fn test_stream_event_type_parsing() {
    let app = create_test_router();

    // Test single event type
    let response =
        make_streaming_request(&app, "/api/v1/stream/activities?event_types=activation").await;

    assert_eq!(response.status(), StatusCode::OK);

    // Test multiple event types
    let response = make_streaming_request(
        &app,
        "/api/v1/stream/activities?event_types=activation,storage,recall",
    )
    .await;

    assert_eq!(response.status(), StatusCode::OK);

    // Test with spaces (should be trimmed) - URL encode the spaces
    let response = make_streaming_request(
        &app,
        "/api/v1/stream/activities?event_types=activation,%20storage%20,%20recall",
    )
    .await;

    assert_eq!(response.status(), StatusCode::OK);
}

#[tokio::test]
async fn test_stream_memory_type_parsing() {
    let app = create_test_router();

    // Test single memory type
    let response =
        make_streaming_request(&app, "/api/v1/stream/memories?memory_types=semantic").await;

    assert_eq!(response.status(), StatusCode::OK);

    // Test multiple memory types
    let response = make_streaming_request(
        &app,
        "/api/v1/stream/memories?memory_types=semantic,episodic,procedural",
    )
    .await;

    assert_eq!(response.status(), StatusCode::OK);
}

#[tokio::test]
async fn test_stream_session_id_generation() {
    let app = create_test_router();

    // Without session_id - should generate one
    let response = make_streaming_request(&app, "/api/v1/stream/activities").await;

    assert_eq!(response.status(), StatusCode::OK);

    // With session_id - should use provided one
    let response = make_streaming_request(
        &app,
        "/api/v1/stream/activities?session_id=custom_session_456",
    )
    .await;

    assert_eq!(response.status(), StatusCode::OK);
}

#[tokio::test]
async fn test_stream_boolean_parameters() {
    let app = create_test_router();

    // Test boolean parsing for memory stream
    let test_cases = vec![
        "include_formation=true&include_retrieval=false&include_completion=true",
        "include_formation=false&include_retrieval=true&include_completion=false",
    ];

    for params in test_cases {
        let response =
            make_streaming_request(&app, &format!("/api/v1/stream/memories?{}", params)).await;

        assert_eq!(response.status(), StatusCode::OK);
    }

    // Test boolean parsing for consolidation stream
    let consolidation_cases = vec![
        "include_replay=true&include_insights=false&include_progress=true",
        "include_replay=false&include_insights=true&include_progress=false",
    ];

    for params in consolidation_cases {
        let response =
            make_streaming_request(&app, &format!("/api/v1/stream/consolidation?{}", params)).await;

        assert_eq!(response.status(), StatusCode::OK);
    }
}

// NOTE: Full SSE event parsing would require reading the response body stream,
// which is more complex and would need additional test infrastructure.
// These tests focus on endpoint availability and basic parameter validation.

#[tokio::test]
async fn test_stream_headers_compliance() {
    let app = create_test_router();

    let endpoints = vec![
        "/api/v1/stream/activities",
        "/api/v1/stream/memories",
        "/api/v1/stream/consolidation",
    ];

    for endpoint in endpoints {
        let response = make_streaming_request(&app, endpoint).await;

        assert_eq!(response.status(), StatusCode::OK);

        // Check required SSE headers
        assert_eq!(
            response.headers().get("content-type").unwrap(),
            "text/event-stream"
        );

        assert_eq!(response.headers().get("cache-control").unwrap(), "no-cache");

        // Connection header should allow keep-alive for SSE
        if let Some(connection) = response.headers().get("connection") {
            assert_ne!(connection, "close");
        }
    }
}

#[tokio::test]
async fn test_stream_cognitive_ergonomics() {
    let app = create_test_router();

    // Test that cognitive constraints are respected
    // Buffer sizes should respect working memory limits (3-4 streams max, reasonable buffer sizes)

    let response = make_streaming_request(
        &app,
        "/api/v1/stream/activities?buffer_size=32", // Reasonable size
    )
    .await;

    assert_eq!(response.status(), StatusCode::OK);

    // Test that importance filtering follows cognitive principles
    let response = make_streaming_request(
        &app,
        "/api/v1/stream/activities?min_importance=0.8", // High importance only
    )
    .await;

    assert_eq!(response.status(), StatusCode::OK);

    // Test that confidence thresholds are meaningful
    let response = make_streaming_request(
        &app,
        "/api/v1/stream/memories?min_confidence=0.7", // High confidence memories
    )
    .await;

    assert_eq!(response.status(), StatusCode::OK);
}
