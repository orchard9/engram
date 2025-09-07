//! Tests for real-time monitoring endpoints (SSE)
//!
//! Tests Server-Sent Events endpoints for hierarchical monitoring, causality tracking,
//! and cognitive-friendly event streaming with working memory constraints.

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

/// Create test router for monitoring tests
fn create_test_router() -> Router {
    let graph = Arc::new(RwLock::new(MemoryGraph::new()));
    let api_state = ApiState::new(graph);

    create_api_routes().with_state(api_state)
}

/// Helper to make monitoring SSE requests
async fn make_monitoring_request(app: &Router, uri: &str) -> Response<Body> {
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
async fn test_monitor_events_basic() {
    let app = create_test_router();

    let response = make_monitoring_request(&app, "/api/v1/monitor/events").await;

    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        response.headers().get("content-type").unwrap(),
        "text/event-stream"
    );
    assert_eq!(response.headers().get("cache-control").unwrap(), "no-cache");

    // Drop response body to clean up SSE stream
    drop(response);
}

#[tokio::test]
async fn test_monitor_events_with_level_filtering() {
    let app = create_test_router();

    // Test each hierarchical level
    let levels = vec!["global", "region", "node", "edge"];

    for level in levels {
        let response =
            make_monitoring_request(&app, &format!("/api/v1/monitor/events?level={}", level)).await;

        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(
            response.headers().get("content-type").unwrap(),
            "text/event-stream"
        );

        // Drop response body to clean up SSE stream
        drop(response);
    }
}

#[tokio::test]
async fn test_monitor_events_with_focus_filtering() {
    let app = create_test_router();

    let response =
        make_monitoring_request(&app, "/api/v1/monitor/events?level=node&focus_id=node_123").await;

    assert_eq!(response.status(), StatusCode::OK);
    drop(response);

    let response = make_monitoring_request(
        &app,
        "/api/v1/monitor/events?level=edge&focus_id=edge_abc_def",
    )
    .await;

    assert_eq!(response.status(), StatusCode::OK);
    drop(response);
}

#[tokio::test]
async fn test_monitor_events_frequency_bounds() {
    let app = create_test_router();

    // Test frequency clamping
    let test_cases = vec![
        ("max_frequency=0.05", StatusCode::OK), // Should be clamped to 0.1
        ("max_frequency=100.0", StatusCode::OK), // Should be clamped to 50.0
        ("max_frequency=25.0", StatusCode::OK), // Within bounds
    ];

    for (params, expected_status) in test_cases {
        let response =
            make_monitoring_request(&app, &format!("/api/v1/monitor/events?{}", params)).await;

        assert_eq!(response.status(), expected_status);
        drop(response);
    }
}

#[tokio::test]
async fn test_monitor_events_with_causality() {
    let app = create_test_router();

    let response =
        make_monitoring_request(&app, "/api/v1/monitor/events?include_causality=true").await;

    assert_eq!(response.status(), StatusCode::OK);
    drop(response);

    let response =
        make_monitoring_request(&app, "/api/v1/monitor/events?include_causality=false").await;

    assert_eq!(response.status(), StatusCode::OK);
    drop(response);
}

#[tokio::test]
async fn test_monitor_events_event_type_filtering() {
    let app = create_test_router();

    // Single event type
    let response =
        make_monitoring_request(&app, "/api/v1/monitor/events?event_types=activation").await;

    assert_eq!(response.status(), StatusCode::OK);
    drop(response);

    // Multiple event types
    let response = make_monitoring_request(
        &app,
        "/api/v1/monitor/events?event_types=activation,storage,consolidation",
    )
    .await;

    assert_eq!(response.status(), StatusCode::OK);
    drop(response);

    // Test URL encoding for spaces
    let response = make_monitoring_request(
        &app,
        "/api/v1/monitor/events?event_types=activation,%20storage%20,%20consolidation",
    )
    .await;

    assert_eq!(response.status(), StatusCode::OK);
    drop(response);
}

#[tokio::test]
async fn test_monitor_activations_basic() {
    let app = create_test_router();

    let response = make_monitoring_request(&app, "/api/v1/monitor/activations").await;

    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        response.headers().get("content-type").unwrap(),
        "text/event-stream"
    );
    drop(response);
}

#[tokio::test]
async fn test_monitor_activations_with_thresholds() {
    let app = create_test_router();

    // Test activation threshold bounds
    let test_cases = vec![
        ("min_activation=0.0", StatusCode::OK),
        ("min_activation=0.5", StatusCode::OK),
        ("min_activation=1.0", StatusCode::OK),
        ("min_activation=-0.5", StatusCode::OK), // Should be clamped to 0.0
        ("min_activation=1.5", StatusCode::OK),  // Should be clamped to 1.0
    ];

    for (params, expected_status) in test_cases {
        let response =
            make_monitoring_request(&app, &format!("/api/v1/monitor/activations?{}", params)).await;

        assert_eq!(response.status(), expected_status);
        drop(response);
    }
}

#[tokio::test]
async fn test_monitor_activations_hierarchical_focus() {
    let app = create_test_router();

    // Test hierarchical focusing
    let response = make_monitoring_request(
        &app,
        "/api/v1/monitor/activations?level=region&focus_id=hippocampus",
    )
    .await;

    assert_eq!(response.status(), StatusCode::OK);
    drop(response);

    let response = make_monitoring_request(
        &app,
        "/api/v1/monitor/activations?level=node&focus_id=ca1_pyramidal_123",
    )
    .await;

    assert_eq!(response.status(), StatusCode::OK);
    drop(response);
}

#[tokio::test]
async fn test_monitor_causality_basic() {
    let app = create_test_router();

    let response = make_monitoring_request(&app, "/api/v1/monitor/causality").await;

    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        response.headers().get("content-type").unwrap(),
        "text/event-stream"
    );
    drop(response);
}

#[tokio::test]
async fn test_monitor_causality_with_focus() {
    let app = create_test_router();

    let response =
        make_monitoring_request(&app, "/api/v1/monitor/causality?focus_id=operation_456").await;

    assert_eq!(response.status(), StatusCode::OK);
    drop(response);

    // Test causality inclusion control
    let response = make_monitoring_request(
        &app,
        "/api/v1/monitor/causality?include_causality=true&focus_id=memory_formation_789",
    )
    .await;

    assert_eq!(response.status(), StatusCode::OK);
    drop(response);
}

#[tokio::test]
async fn test_session_management() {
    let app = create_test_router();

    // Without session_id - should generate UUID
    let response = make_monitoring_request(&app, "/api/v1/monitor/events").await;

    assert_eq!(response.status(), StatusCode::OK);
    drop(response);

    // With provided session_id - should use it
    let response = make_monitoring_request(
        &app,
        "/api/v1/monitor/events?session_id=custom_session_abc123",
    )
    .await;

    assert_eq!(response.status(), StatusCode::OK);
    drop(response);

    // With last_sequence for resumption
    let response = make_monitoring_request(
        &app,
        "/api/v1/monitor/events?session_id=resume_session&last_sequence=42",
    )
    .await;

    assert_eq!(response.status(), StatusCode::OK);
    drop(response);
}

#[tokio::test]
async fn test_cognitive_constraints() {
    let app = create_test_router();

    // Test working memory constraint enforcement
    // Max frequency should respect cognitive limits
    let response = make_monitoring_request(
        &app,
        "/api/v1/monitor/events?max_frequency=100.0", // Should be clamped to 50Hz max
    )
    .await;

    assert_eq!(response.status(), StatusCode::OK);
    drop(response);

    // Test activation threshold bounds
    let response = make_monitoring_request(
        &app,
        "/api/v1/monitor/activations?min_activation=2.0", // Should be clamped to 1.0
    )
    .await;

    assert_eq!(response.status(), StatusCode::OK);
    drop(response);

    // Test event type filtering respects working memory (≤4 types)
    let response = make_monitoring_request(
        &app,
        "/api/v1/monitor/events?event_types=activation,storage,consolidation,recall",
    )
    .await;

    assert_eq!(response.status(), StatusCode::OK);
    drop(response);
}

#[tokio::test]
async fn test_hierarchical_event_organization() {
    let app = create_test_router();

    // Test that hierarchical levels are supported
    let hierarchical_levels = vec![
        ("level=global", "Global level monitoring"),
        ("level=region&focus_id=prefrontal_cortex", "Regional focus"),
        ("level=node&focus_id=neuron_123", "Node-specific monitoring"),
        ("level=edge&focus_id=synapse_abc_def", "Edge-level detail"),
    ];

    for (params, _description) in hierarchical_levels {
        let response =
            make_monitoring_request(&app, &format!("/api/v1/monitor/events?{}", params)).await;

        assert_eq!(response.status(), StatusCode::OK);

        // Verify SSE headers are present
        assert_eq!(
            response.headers().get("content-type").unwrap(),
            "text/event-stream"
        );
        drop(response);
    }
}

#[tokio::test]
async fn test_sse_headers_compliance() {
    let app = create_test_router();

    let endpoints = vec![
        "/api/v1/monitor/events",
        "/api/v1/monitor/activations",
        "/api/v1/monitor/causality",
    ];

    for endpoint in endpoints {
        let response = make_monitoring_request(&app, endpoint).await;

        assert_eq!(response.status(), StatusCode::OK);

        // Required SSE headers
        assert_eq!(
            response.headers().get("content-type").unwrap(),
            "text/event-stream"
        );

        assert_eq!(response.headers().get("cache-control").unwrap(), "no-cache");

        // Connection should allow persistent connections
        if let Some(connection) = response.headers().get("connection") {
            assert_ne!(connection, "close");
        }
        drop(response);
    }
}

#[tokio::test]
async fn test_latency_requirements() {
    let app = create_test_router();

    // Test that monitoring endpoints respond quickly (simulating <100ms requirement)
    let start = std::time::Instant::now();

    let response = make_monitoring_request(&app, "/api/v1/monitor/events?max_frequency=10.0").await;

    let duration = start.elapsed();

    assert_eq!(response.status(), StatusCode::OK);
    // Initial connection should be fast (<100ms simulated test)
    assert!(duration < std::time::Duration::from_millis(100));
    drop(response);
}

#[tokio::test]
async fn test_parameter_parsing_edge_cases() {
    let app = create_test_router();

    // Empty event_types - should handle gracefully
    let response = make_monitoring_request(&app, "/api/v1/monitor/events?event_types=").await;

    // Could be OK (empty list) or BAD_REQUEST depending on implementation
    assert!(response.status() == StatusCode::OK || response.status() == StatusCode::BAD_REQUEST);
    drop(response);

    // Invalid level - should still work (implementation may ignore invalid levels)
    let response =
        make_monitoring_request(&app, "/api/v1/monitor/events?level=invalid_level").await;

    assert_eq!(response.status(), StatusCode::OK);
    drop(response);

    // Invalid boolean values - should handle gracefully or return error
    let response =
        make_monitoring_request(&app, "/api/v1/monitor/events?include_causality=maybe").await;

    // Could be OK (default false) or BAD_REQUEST depending on implementation
    assert!(response.status() == StatusCode::OK || response.status() == StatusCode::BAD_REQUEST);
    drop(response);
}

#[tokio::test]
async fn test_concurrent_monitoring_capability() {
    let app = create_test_router();

    // Test that multiple monitoring sessions can be distinguished and handled
    // Uses a rapid create-test-drop pattern to avoid resource buildup
    let session_ids = vec!["concurrent_0", "concurrent_1", "concurrent_2"];

    for session_id in session_ids {
        // Create monitoring request with unique session
        let request = Request::builder()
            .method(Method::GET)
            .uri(&format!(
                "/api/v1/monitor/events?session_id={}&max_frequency=50.0",
                session_id
            ))
            .header("Accept", "text/event-stream")
            .header("Cache-Control", "no-cache")
            .body(Body::empty())
            .unwrap();

        // Test that connection establishes successfully
        let response = app.clone().oneshot(request).await.unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(
            response.headers().get("content-type").unwrap(),
            "text/event-stream"
        );
        assert_eq!(response.headers().get("cache-control").unwrap(), "no-cache");

        // Drop immediately to prevent resource accumulation
        drop(response);
    }
}

#[tokio::test]
async fn test_monitoring_with_complex_queries() {
    let app = create_test_router();

    // Complex monitoring query with multiple parameters
    let complex_query = "/api/v1/monitor/events?level=node&focus_id=ca1_123&max_frequency=25.0&include_causality=true&event_types=activation,consolidation&min_activation=0.7&session_id=complex_test&last_sequence=100";

    let response = make_monitoring_request(&app, complex_query).await;

    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        response.headers().get("content-type").unwrap(),
        "text/event-stream"
    );
    drop(response);
}

// NOTE: Full SSE event stream parsing and real-time event validation
// would require additional test infrastructure with actual event generation.
// These tests focus on endpoint availability, parameter validation,
// and HTTP/SSE compliance.
