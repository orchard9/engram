//! Tests for HTTP REST API implementation
//!
//! Tests the cognitive-friendly HTTP API endpoints with proper validation,
//! error handling, and memory operation semantics.

use axum::{
    Router,
    body::Body,
    http::{Method, Request, StatusCode},
};
use chrono::Utc;
use engram_cli::api::{ApiState, create_api_routes};
use engram_core::graph::create_concurrent_graph;
use serde_json::{Value, json};
use std::sync::Arc;
use tokio::sync::RwLock;
use tower::ServiceExt; // for `oneshot`

/// Create test router with API routes
fn create_test_router() -> Router {
    let store = Arc::new(engram_core::MemoryStore::new(100));
    let api_state = ApiState::new(store);

    create_api_routes().with_state(api_state)
}

/// Helper to make HTTP requests
async fn make_request(
    app: &Router,
    method: Method,
    uri: &str,
    body: Option<Value>,
) -> (StatusCode, Value) {
    let request = Request::builder()
        .method(method)
        .uri(uri)
        .header("content-type", "application/json");

    let request = if let Some(body) = body {
        request.body(Body::from(body.to_string()))
    } else {
        request.body(Body::empty())
    }
    .unwrap();

    let response = app.clone().oneshot(request).await.unwrap();
    let status = response.status();

    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let json: Value = serde_json::from_slice(&body).unwrap_or_else(|_| json!({}));

    (status, json)
}

#[tokio::test]
async fn test_remember_memory_success() {
    let app = create_test_router();

    let request_body = json!({
        "id": "test_memory_001",
        "content": "This is a test memory for cognitive graph storage",
        "confidence": 0.8,
        "confidence_reasoning": "High confidence test memory",
        "tags": ["test", "cognitive"],
        "memory_type": "semantic",
        "auto_link": true,
        "link_threshold": 0.7
    });

    let (status, response) = make_request(
        &app,
        Method::POST,
        "/api/v1/memories/remember",
        Some(request_body),
    )
    .await;

    assert_eq!(status, StatusCode::CREATED);
    assert!(response["memory_id"].as_str().is_some());
    assert!(response["storage_confidence"]["value"].as_f64().unwrap() > 0.0);
    assert_eq!(response["consolidation_state"].as_str().unwrap(), "Recent");
    assert!(
        response["system_message"]
            .as_str()
            .unwrap()
            .contains("successfully encoded")
    );

    // Check auto-linking was applied
    assert!(response["auto_links"].is_array());
}

#[tokio::test]
async fn test_remember_memory_validation_errors() {
    let app = create_test_router();

    // Test empty content
    let request_body = json!({
        "content": ""
    });

    let (status, response) = make_request(
        &app,
        Method::POST,
        "/api/v1/memories/remember",
        Some(request_body),
    )
    .await;

    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(
        response["error"]["code"].as_str().unwrap(),
        "INVALID_MEMORY_INPUT"
    );
    assert!(
        response["error"]["message"]
            .as_str()
            .unwrap()
            .contains("cannot be empty")
    );
    assert!(response["error"]["educational_context"].is_string());
    assert!(response["error"]["cognitive_guidance"].is_string());

    // Test content too large
    let large_content = "x".repeat(200_000);
    let request_body = json!({
        "content": large_content
    });

    let (status, response) = make_request(
        &app,
        Method::POST,
        "/api/v1/memories/remember",
        Some(request_body),
    )
    .await;

    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(
        response["error"]["message"]
            .as_str()
            .unwrap()
            .contains("too large")
    );
}

#[tokio::test]
async fn test_remember_episode_success() {
    let app = create_test_router();

    let when = Utc::now();
    let request_body = json!({
        "id": "episode_001",
        "when": when.to_rfc3339(),
        "what": "I attended a fascinating AI conference today",
        "where_location": "San Francisco Convention Center",
        "who": ["Alice", "Bob", "Charlie"],
        "why": "To learn about latest AI developments",
        "how": "Attended keynotes and networking sessions",
        "emotional_valence": 0.7,
        "importance": 0.9,
        "auto_link": false
    });

    let (status, response) = make_request(
        &app,
        Method::POST,
        "/api/v1/episodes/remember",
        Some(request_body),
    )
    .await;

    assert_eq!(status, StatusCode::CREATED);
    assert_eq!(response["memory_id"].as_str().unwrap(), "episode_001");
    assert!(response["storage_confidence"]["value"].as_f64().unwrap() >= 0.8);
    assert!(
        response["system_message"]
            .as_str()
            .unwrap()
            .contains("successfully encoded")
    );
    assert!(
        response["system_message"]
            .as_str()
            .unwrap()
            .contains("contextual details")
    );
}

#[tokio::test]
async fn test_recall_memories_with_query() {
    let app = create_test_router();

    // First store a memory to recall
    let store_request = json!({
        "content": "Machine learning algorithms for pattern recognition",
        "tags": ["ai", "ml", "patterns"]
    });

    let (status, _) = make_request(
        &app,
        Method::POST,
        "/api/v1/memories/remember",
        Some(store_request),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED);

    // Now test recall
    let (status, response) = make_request(
        &app,
        Method::GET,
        "/api/v1/memories/recall?query=machine%20learning&max_results=10&include_metadata=true&trace_activation=true",
        None,
    ).await;

    assert_eq!(status, StatusCode::OK);

    // Check response structure
    assert!(response["memories"]["vivid"].is_array());
    assert!(response["memories"]["associated"].is_array());
    assert!(response["memories"]["reconstructed"].is_array());

    assert!(response["recall_confidence"]["value"].as_f64().unwrap() > 0.0);
    assert!(response["query_analysis"]["understood_intent"].is_string());
    assert!(response["query_analysis"]["search_strategy"].is_string());
    assert!(response["query_analysis"]["cognitive_load"].is_string());
    assert!(response["query_analysis"]["suggestions"].is_array());

    // Check metadata is included
    assert!(response["metadata"].is_object());
    assert!(response["metadata"]["total_memories_searched"].is_number());
    assert!(response["metadata"]["processing_time_ms"].is_number());

    assert!(
        response["system_message"]
            .as_str()
            .unwrap()
            .contains("Recall completed")
    );
}

#[tokio::test]
async fn test_recall_validation_errors() {
    let app = create_test_router();

    // Test recall without query or embedding
    let (status, response) = make_request(&app, Method::GET, "/api/v1/memories/recall", None).await;

    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(
        response["error"]["code"].as_str().unwrap(),
        "INVALID_MEMORY_INPUT"
    );
    assert!(
        response["error"]["message"]
            .as_str()
            .unwrap()
            .contains("requires either a text query")
    );
    assert!(response["error"]["educational_context"].is_string());
}

#[tokio::test]
async fn test_recall_with_embedding() {
    let app = create_test_router();

    let embedding_vec = vec![0.1, 0.2, 0.3, 0.4, 0.5];
    let embedding_json = serde_json::to_string(&embedding_vec).unwrap();
    let query = format!(
        "/api/v1/memories/recall?embedding={}&threshold=0.6",
        urlencoding::encode(&embedding_json)
    );

    let (status, response) = make_request(&app, Method::GET, &query, None).await;

    assert_eq!(status, StatusCode::OK);
    assert!(response["memories"].is_object());
    assert!(
        response["query_analysis"]["understood_intent"]
            .as_str()
            .unwrap()
            .contains("Embedding-based")
    );
}

#[tokio::test]
async fn test_recognize_pattern_success() {
    let app = create_test_router();

    let request_body = json!({
        "input": "This is a recognizable pattern with sufficient length",
        "threshold": 0.6
    });

    let (status, response) = make_request(
        &app,
        Method::POST,
        "/api/v1/memories/recognize",
        Some(request_body),
    )
    .await;

    assert_eq!(status, StatusCode::OK);
    assert!(response["recognized"].is_boolean());
    assert!(response["recognition_confidence"]["value"].is_f64());
    assert!(response["recognition_confidence"]["category"].is_string());
    assert!(response["recognition_confidence"]["reasoning"].is_string());
    assert!(response["similar_patterns"].is_array());
    assert!(response["system_message"].is_string());
}

#[tokio::test]
async fn test_recognize_pattern_validation() {
    let app = create_test_router();

    let request_body = json!({
        "input": ""
    });

    let (status, response) = make_request(
        &app,
        Method::POST,
        "/api/v1/memories/recognize",
        Some(request_body),
    )
    .await;

    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(
        response["error"]["message"]
            .as_str()
            .unwrap()
            .contains("requires input content")
    );
}

#[tokio::test]
async fn test_system_health() {
    let app = create_test_router();

    let (status, response) = make_request(&app, Method::GET, "/api/v1/system/health", None).await;

    assert_eq!(status, StatusCode::OK);
    assert_eq!(response["status"].as_str().unwrap(), "healthy");
    assert!(response["memory_system"]["total_memories"].is_number());
    assert!(response["memory_system"]["consolidation_active"].is_boolean());
    assert!(response["cognitive_load"].is_object());
    assert!(response["system_message"].is_string());
}

#[tokio::test]
async fn test_system_introspect() {
    let app = create_test_router();

    let (status, response) =
        make_request(&app, Method::GET, "/api/v1/system/introspect", None).await;

    assert_eq!(status, StatusCode::OK);
    assert!(response["memory_statistics"].is_object());
    assert!(response["memory_statistics"]["total_nodes"].is_number());
    assert!(response["system_processes"].is_object());
    assert!(response["performance_metrics"].is_object());
}

#[tokio::test]
async fn test_episodes_replay() {
    let app = create_test_router();

    let (status, response) = make_request(
        &app,
        Method::GET,
        "/api/v1/episodes/replay?time_range=recent",
        None,
    )
    .await;

    assert_eq!(status, StatusCode::OK);
    assert!(response["episodes"].is_array());
    assert!(response["replay_confidence"].is_object());
    assert!(response["system_message"].is_string());
}

#[tokio::test]
async fn test_memory_types_and_tags() {
    let app = create_test_router();

    // Test different memory types
    let memory_types = ["semantic", "episodic", "procedural"];

    for memory_type in &memory_types {
        let request_body = json!({
            "content": format!("Test {} memory content", memory_type),
            "memory_type": memory_type,
            "tags": [memory_type, "test"]
        });

        let (status, response) = make_request(
            &app,
            Method::POST,
            "/api/v1/memories/remember",
            Some(request_body),
        )
        .await;

        assert_eq!(status, StatusCode::CREATED);
        assert!(response["memory_id"].is_string());
        assert!(response["storage_confidence"]["value"].is_f64());
    }
}

#[tokio::test]
async fn test_confidence_levels_and_reasoning() {
    let app = create_test_router();

    let confidence_levels = [
        (0.1, "Low confidence test"),
        (0.5, "Medium confidence test"),
        (0.9, "High confidence test"),
        (1.0, "Certain confidence test"),
    ];

    for (confidence, reasoning) in &confidence_levels {
        let request_body = json!({
            "content": format!("Memory with confidence {}", confidence),
            "confidence": confidence,
            "confidence_reasoning": reasoning
        });

        let (status, response) = make_request(
            &app,
            Method::POST,
            "/api/v1/memories/remember",
            Some(request_body),
        )
        .await;

        assert_eq!(status, StatusCode::CREATED);

        let returned_confidence = response["storage_confidence"]["value"].as_f64().unwrap();
        assert!((returned_confidence - confidence).abs() < 0.01);
        assert_eq!(
            response["storage_confidence"]["reasoning"]
                .as_str()
                .unwrap(),
            *reasoning
        );
    }
}

#[tokio::test]
async fn test_auto_linking_behavior() {
    let app = create_test_router();

    // Test with auto-linking enabled
    let request_body = json!({
        "content": "Neural networks for deep learning applications",
        "auto_link": true,
        "link_threshold": 0.6
    });

    let (status, response) = make_request(
        &app,
        Method::POST,
        "/api/v1/memories/remember",
        Some(request_body),
    )
    .await;

    assert_eq!(status, StatusCode::CREATED);
    assert!(response["auto_links"].is_array());
    assert!(
        response["system_message"]
            .as_str()
            .unwrap()
            .contains("Automatic linking enabled")
    );

    // Test with auto-linking disabled
    let request_body = json!({
        "content": "Machine learning algorithms for pattern recognition",
        "auto_link": false
    });

    let (status, response) = make_request(
        &app,
        Method::POST,
        "/api/v1/memories/remember",
        Some(request_body),
    )
    .await;

    assert_eq!(status, StatusCode::CREATED);
    assert_eq!(response["auto_links"].as_array().unwrap().len(), 0);
    assert!(
        response["system_message"]
            .as_str()
            .unwrap()
            .contains("Consider enabling auto-linking")
    );
}

#[tokio::test]
async fn test_recall_query_parameters() {
    let app = create_test_router();

    // Test with various query parameters
    let test_cases = [
        "/api/v1/memories/recall?query=test&max_results=5",
        "/api/v1/memories/recall?query=test&threshold=0.8&include_metadata=true",
        "/api/v1/memories/recall?query=test&trace_activation=true&required_tags=ai,ml",
        "/api/v1/memories/recall?query=test&excluded_tags=test,demo&location=office",
    ];

    for uri in &test_cases {
        let (status, response) = make_request(&app, Method::GET, uri, None).await;

        assert_eq!(status, StatusCode::OK);
        assert!(response["memories"].is_object());
        assert!(response["query_analysis"].is_object());
        assert!(response["system_message"].is_string());
    }
}

#[tokio::test]
async fn test_error_response_structure() {
    let app = create_test_router();

    // Test malformed JSON
    let request = Request::builder()
        .method(Method::POST)
        .uri("/api/v1/memories/remember")
        .header("content-type", "application/json")
        .body(Body::from("invalid json"))
        .unwrap();

    let response = app.clone().oneshot(request).await.unwrap();
    let status = response.status();

    // Should get a JSON parsing error (400 Bad Request from axum)
    assert_eq!(status, StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn test_cors_headers_present() {
    let app = create_test_router();

    // Make a preflight CORS request
    let request = Request::builder()
        .method(Method::OPTIONS)
        .uri("/api/v1/memories/remember")
        .header("Origin", "http://localhost:3000")
        .header("Access-Control-Request-Method", "POST")
        .body(Body::empty())
        .unwrap();

    let response = app.clone().oneshot(request).await.unwrap();

    // CORS should be handled by the CORS layer
    // The exact response depends on axum's CORS implementation
    assert!(response.status().is_success() || response.status() == StatusCode::METHOD_NOT_ALLOWED);
}
