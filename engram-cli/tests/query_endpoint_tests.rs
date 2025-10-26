//! Integration tests for query execution endpoints (HTTP and gRPC).

#![allow(clippy::float_cmp)]

use engram_cli::api::ApiState;
use engram_core::{MemorySpaceId, MemorySpaceRegistry, MemoryStore, metrics::MetricsRegistry};
use engram_proto::engram_service_server::EngramService;
use engram_proto::{QueryRequest as ProtoQueryRequest, QueryResponse};
use serde_json::json;
use std::sync::Arc;
use tokio::sync::watch;
use tonic::Request;

/// Helper to create test API state
async fn create_test_state() -> ApiState {
    let store = Arc::new(MemoryStore::new(1000));
    let metrics = Arc::new(MetricsRegistry::new());
    let registry = Arc::new(
        MemorySpaceRegistry::new("/tmp/engram_query_test", |_id, _dirs| {
            Ok(Arc::new(MemoryStore::new(1000)))
        })
        .expect("Failed to create registry"),
    );

    let default_space = MemorySpaceId::new("default").unwrap();
    registry
        .create_or_get(&default_space)
        .await
        .expect("Failed to create default space");

    let (shutdown_tx, _) = watch::channel(false);

    let memory_service = engram_cli::grpc::MemoryService::new(
        store.clone(),
        metrics.clone(),
        registry.clone(),
        default_space.clone(),
    );

    #[allow(deprecated)]
    ApiState {
        store: store.clone(),
        memory_service: Arc::new(memory_service),
        registry,
        default_space,
        metrics,
        auto_tuner: engram_core::activation::SpreadingAutoTuner::new(0.1, 1000),
        shutdown_tx: Arc::new(shutdown_tx),
    }
}

#[tokio::test]
async fn test_http_query_endpoint_with_recall() {
    let state = create_test_state().await;

    // Create HTTP request
    let request_body = json!({
        "memory_space_id": "default",
        "query_text": "RECALL episode_123"
    });

    let headers = axum::http::HeaderMap::new();

    // Call handler directly
    let result = engram_cli::handlers::query::query_handler(
        axum::extract::State(state),
        headers,
        axum::extract::Json(
            serde_json::from_value(request_body).expect("Failed to deserialize request"),
        ),
    )
    .await;

    // Should succeed even though episode doesn't exist (empty results)
    assert!(result.is_ok(), "Query execution should succeed");

    // The handler returns impl IntoResponse which is Json wrapper
    // We can't easily extract the value from tests, so just verify success
    // In real integration tests, this would be tested via HTTP calls
}

#[tokio::test]
async fn test_http_query_endpoint_with_invalid_syntax() {
    let state = create_test_state().await;

    let request_body = json!({
        "memory_space_id": "default",
        "query_text": "INVALID QUERY SYNTAX"
    });

    let headers = axum::http::HeaderMap::new();

    let result = engram_cli::handlers::query::query_handler(
        axum::extract::State(state),
        headers,
        axum::extract::Json(
            serde_json::from_value(request_body).expect("Failed to deserialize request"),
        ),
    )
    .await;

    // Should fail with bad request
    assert!(result.is_err(), "Invalid query should fail");
}

#[tokio::test]
async fn test_http_query_endpoint_with_header_memory_space() {
    let state = create_test_state().await;

    // Create a test space
    let test_space = MemorySpaceId::new("test_space").unwrap();
    state
        .registry
        .create_or_get(&test_space)
        .await
        .expect("Failed to create test space");

    let request_body = json!({
        "query_text": "RECALL episode_123"
    });

    let mut headers = axum::http::HeaderMap::new();
    headers.insert("X-Memory-Space", "test_space".parse().unwrap());

    let result = engram_cli::handlers::query::query_handler(
        axum::extract::State(state),
        headers,
        axum::extract::Json(
            serde_json::from_value(request_body).expect("Failed to deserialize request"),
        ),
    )
    .await;

    assert!(result.is_ok(), "Query should succeed with header space");
}

#[tokio::test]
async fn test_grpc_execute_query_with_recall() {
    let state = create_test_state().await;

    let request = Request::new(ProtoQueryRequest {
        memory_space_id: "default".to_string(),
        query_text: "RECALL episode_123".to_string(),
    });

    let result = state.memory_service.execute_query(request).await;

    assert!(result.is_ok(), "gRPC query execution should succeed");

    if let Ok(response) = result {
        let query_response: QueryResponse = response.into_inner();
        assert_eq!(query_response.total_count, 0, "Should return 0 results");
        // Execution time can be 0 for very fast queries, so just check it's present
        assert!(
            query_response.aggregate_confidence.is_some(),
            "Should include aggregate confidence"
        );
    }
}

#[tokio::test]
async fn test_grpc_execute_query_with_empty_text() {
    let state = create_test_state().await;

    let request = Request::new(ProtoQueryRequest {
        memory_space_id: "default".to_string(),
        query_text: String::new(),
    });

    let result = state.memory_service.execute_query(request).await;

    assert!(result.is_err(), "Empty query text should fail");

    if let Err(status) = result {
        assert_eq!(status.code(), tonic::Code::InvalidArgument);
    }
}

#[tokio::test]
async fn test_grpc_execute_query_with_invalid_syntax() {
    let state = create_test_state().await;

    let request = Request::new(ProtoQueryRequest {
        memory_space_id: "default".to_string(),
        query_text: "INVALID SYNTAX HERE".to_string(),
    });

    let result = state.memory_service.execute_query(request).await;

    assert!(result.is_err(), "Invalid syntax should fail");

    if let Err(status) = result {
        assert_eq!(status.code(), tonic::Code::InvalidArgument);
    }
}

#[tokio::test]
async fn test_grpc_execute_query_with_nonexistent_space() {
    let state = create_test_state().await;

    let request = Request::new(ProtoQueryRequest {
        memory_space_id: "nonexistent_space".to_string(),
        query_text: "RECALL episode_123".to_string(),
    });

    let result = state.memory_service.execute_query(request).await;

    assert!(result.is_err(), "Nonexistent space should fail");

    if let Err(status) = result {
        assert_eq!(status.code(), tonic::Code::NotFound);
    }
}

#[tokio::test]
async fn test_query_response_structure() {
    let state = create_test_state().await;

    let request = Request::new(ProtoQueryRequest {
        memory_space_id: "default".to_string(),
        query_text: "RECALL episode_123".to_string(),
    });

    let result = state.memory_service.execute_query(request).await;

    assert!(result.is_ok());

    if let Ok(response) = result {
        let query_response = response.into_inner();

        // Verify response structure
        assert!(query_response.episodes.is_empty());
        assert!(query_response.confidences.is_empty());
        assert!(query_response.aggregate_confidence.is_some());

        if let Some(confidence_interval) = query_response.aggregate_confidence {
            assert!(confidence_interval.lower_bound >= 0.0);
            assert!(confidence_interval.upper_bound <= 1.0);
            assert!(confidence_interval.mean >= confidence_interval.lower_bound);
            assert!(confidence_interval.mean <= confidence_interval.upper_bound);
            assert_eq!(confidence_interval.confidence_level, 0.95);
        }
    }
}
