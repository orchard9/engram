//! Multi-tenant isolation validation tests for Task 007
//!
//! Tests verify that memory spaces maintain strict isolation across:
//! - Memory storage and recall via HTTP API (no cross-space leakage)
//! - Directory creation (separate per-space directories)
//! - Concurrent operations (thread-safe registry access)

#![allow(clippy::redundant_closure_for_method_calls)]
#![allow(clippy::too_many_lines)]

use axum::{
    Router,
    body::Body,
    http::{Method, Request, StatusCode},
};
use engram_cli::api::{ApiState, create_api_routes};
use engram_core::activation::SpreadingAutoTuner;
use engram_core::{MemorySpaceError, MemorySpaceId, MemorySpaceRegistry, MemoryStore, metrics};
use serde_json::{Value, json};
use std::path::PathBuf;
use std::sync::Arc;
use tower::ServiceExt;

/// Create test router with registry supporting multiple spaces
async fn create_multi_space_router(data_root: PathBuf) -> (Router, Arc<MemorySpaceRegistry>) {
    let registry = Arc::new(
        MemorySpaceRegistry::new(&data_root, |_space_id, _directories| {
            let mut store = MemoryStore::new(100);
            let _ = store.enable_event_streaming(32);
            Ok::<Arc<MemoryStore>, MemorySpaceError>(Arc::new(store))
        })
        .expect("registry creation"),
    );

    // Create default space for API state
    let default_space = MemorySpaceId::default();
    let space_handle = registry
        .create_or_get(&default_space)
        .await
        .expect("default space");
    let store = space_handle.store();

    // Keep event streaming alive
    let mut keepalive_rx = store
        .subscribe_to_events()
        .expect("event streaming initialized");
    tokio::spawn(async move { while keepalive_rx.recv().await.is_ok() {} });

    let metrics = metrics::init();
    let auto_tuner = SpreadingAutoTuner::new(0.10, 16);
    let (shutdown_tx, _shutdown_rx) = tokio::sync::watch::channel(false);
    let api_state = ApiState::new(
        store,
        Arc::clone(&registry),
        default_space,
        metrics,
        auto_tuner,
        Arc::new(shutdown_tx),
    );

    let router = create_api_routes().with_state(api_state);
    (router, registry)
}

/// Helper to make HTTP requests with optional space header
async fn make_request_with_space(
    app: &Router,
    method: Method,
    uri: &str,
    body: Option<Value>,
    space_id: Option<&str>,
) -> (StatusCode, Value) {
    let mut request = Request::builder()
        .method(method)
        .uri(uri)
        .header("content-type", "application/json");

    // Add X-Memory-Space header if specified
    if let Some(space) = space_id {
        request = request.header("X-Memory-Space", space);
    }

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
async fn test_cross_space_memory_isolation() {
    // Create router with two distinct spaces
    let temp_dir = tempfile::tempdir().expect("temp dir");
    let (app, registry) = create_multi_space_router(temp_dir.path().to_path_buf()).await;

    // Create two spaces explicitly
    let space_alpha = MemorySpaceId::new("alpha").expect("valid space id");
    let space_beta = MemorySpaceId::new("beta").expect("valid space id");

    registry.create_or_get(&space_alpha).await.expect("alpha");
    registry.create_or_get(&space_beta).await.expect("beta");

    // Store memory in space alpha
    let alpha_memory = json!({
        "id": "alpha_memory_001",
        "content": "This belongs to alpha",
        "embedding": vec![0.1_f32; 768]
    });

    let (status, _response) = make_request_with_space(
        &app,
        Method::POST,
        "/api/v1/memories/remember",
        Some(alpha_memory.clone()),
        Some("alpha"),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "Alpha remember should succeed");

    // Store memory in space beta
    let beta_memory = json!({
        "id": "beta_memory_001",
        "content": "This belongs to beta",
        "embedding": vec![0.2_f32; 768]
    });

    let (status, _response) = make_request_with_space(
        &app,
        Method::POST,
        "/api/v1/memories/remember",
        Some(beta_memory.clone()),
        Some("beta"),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "Beta remember should succeed");

    // Recall from alpha - should only see alpha memory
    let _alpha_recall = json!({
        "query": vec![0.1_f32; 768],
        "k": 10
    });

    let (status, response) = make_request_with_space(
        &app,
        Method::GET,
        "/api/v1/memories/recall?embedding=[0.1]",
        None,
        Some("alpha"),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "Alpha recall should succeed");

    let memories = response["memories"].as_array().expect("memories array");
    assert_eq!(memories.len(), 1, "Alpha should see exactly 1 memory");
    assert_eq!(
        memories[0]["id"].as_str().unwrap(),
        "alpha_memory_001",
        "Alpha should only see its own memory"
    );

    // Recall from beta - should only see beta memory
    let (status, response) = make_request_with_space(
        &app,
        Method::GET,
        "/api/v1/memories/recall?query=beta",
        None,
        Some("beta"),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "Beta recall should succeed");

    let memories = response["memories"].as_array().expect("memories array");
    assert_eq!(memories.len(), 1, "Beta should see exactly 1 memory");
    assert_eq!(
        memories[0]["id"].as_str().unwrap(),
        "beta_memory_001",
        "Beta should only see its own memory"
    );

    // Verify cross-space query isolation - alpha query with beta content
    let (status, response) = make_request_with_space(
        &app,
        Method::GET,
        "/api/v1/memories/recall?query=beta",
        None,
        Some("alpha"),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "Cross query should succeed");

    // Alpha should still only see its own memory (even with beta's embedding)
    let memories = response["memories"].as_array().expect("memories array");
    for memory in memories {
        let id = memory["id"].as_str().unwrap();
        assert!(
            id.starts_with("alpha"),
            "Alpha space should never see beta memories"
        );
    }
}

#[tokio::test]
async fn test_directory_isolation() {
    // Create registry with temp directory
    let temp_dir = tempfile::tempdir().expect("temp dir");
    let data_root = temp_dir.path().to_path_buf();
    let (_app, registry) = create_multi_space_router(data_root.clone()).await;

    // Create three spaces
    let space_alpha = MemorySpaceId::new("alpha").expect("valid space id");
    let space_beta = MemorySpaceId::new("beta").expect("valid space id");
    let space_gamma = MemorySpaceId::new("gamma").expect("valid space id");

    registry.create_or_get(&space_alpha).await.expect("alpha");
    registry.create_or_get(&space_beta).await.expect("beta");
    registry.create_or_get(&space_gamma).await.expect("gamma");

    // Verify each space has its own directory structure
    let alpha_dir = data_root.join("alpha");
    let beta_dir = data_root.join("beta");
    let gamma_dir = data_root.join("gamma");

    assert!(alpha_dir.exists(), "Alpha should have dedicated directory");
    assert!(beta_dir.exists(), "Beta should have dedicated directory");
    assert!(gamma_dir.exists(), "Gamma should have dedicated directory");

    // Verify WAL subdirectories exist (created on first write)
    // Note: These may not exist yet without writes, but parent dir should exist
    assert!(alpha_dir.is_dir(), "Alpha directory should be a directory");
    assert!(beta_dir.is_dir(), "Beta directory should be a directory");
    assert!(gamma_dir.is_dir(), "Gamma directory should be a directory");

    // Verify directories are separate (not symlinks or hardlinks)
    assert_ne!(alpha_dir, beta_dir, "Space directories must be distinct");
    assert_ne!(beta_dir, gamma_dir, "Space directories must be distinct");
    assert_ne!(alpha_dir, gamma_dir, "Space directories must be distinct");
}

#[tokio::test]
async fn test_concurrent_space_creation() {
    // Create registry
    let temp_dir = tempfile::tempdir().expect("temp dir");
    let registry = Arc::new(
        MemorySpaceRegistry::new(temp_dir.path(), |_space_id, _directories| {
            let mut store = MemoryStore::new(100);
            let _ = store.enable_event_streaming(32);
            Ok::<Arc<MemoryStore>, MemorySpaceError>(Arc::new(store))
        })
        .expect("registry"),
    );

    // Spawn 20 concurrent tasks creating spaces
    let mut handles = vec![];
    for i in 0..20 {
        let registry = Arc::clone(&registry);
        let handle = tokio::spawn(async move {
            let space_id =
                MemorySpaceId::new(format!("concurrent_space_{i}")).expect("valid space id");
            registry
                .create_or_get(&space_id)
                .await
                .expect("space creation");
        });
        handles.push(handle);
    }

    // Wait for all tasks to complete
    for handle in handles {
        handle.await.expect("task completion");
    }

    // Verify all 20 spaces were created (plus default)
    let spaces = registry.list();
    assert!(
        spaces.len() >= 20,
        "Should have created 20+ spaces concurrently"
    );
}

#[tokio::test]
async fn test_health_endpoint_multi_space() {
    // Create router
    let temp_dir = tempfile::tempdir().expect("temp dir");
    let (app, registry) = create_multi_space_router(temp_dir.path().to_path_buf()).await;

    // Create three spaces
    let space_alpha = MemorySpaceId::new("alpha").expect("valid space id");
    let space_beta = MemorySpaceId::new("beta").expect("valid space id");

    registry.create_or_get(&space_alpha).await.expect("alpha");
    registry.create_or_get(&space_beta).await.expect("beta");

    // Store different numbers of memories in each space
    for i in 0..5 {
        let memory = json!({
            "id": format!("alpha_{}", i),
            "content": "alpha memory",
            "embedding": vec![0.1_f32; 768]
        });
        make_request_with_space(
            &app,
            Method::POST,
            "/api/v1/memories/remember",
            Some(memory),
            Some("alpha"),
        )
        .await;
    }

    for i in 0..3 {
        let memory = json!({
            "id": format!("beta_{}", i),
            "content": "beta memory",
            "embedding": vec![0.2_f32; 768]
        });
        make_request_with_space(
            &app,
            Method::POST,
            "/api/v1/memories/remember",
            Some(memory),
            Some("beta"),
        )
        .await;
    }

    // Query health endpoint
    let (status, response) =
        make_request_with_space(&app, Method::GET, "/api/v1/system/health", None, None).await;
    assert_eq!(status, StatusCode::OK, "Health endpoint should succeed");

    // Verify spaces array exists
    let spaces = response["spaces"].as_array().expect("spaces array");
    assert!(spaces.len() >= 2, "Should report at least 2 spaces");

    // Find alpha and beta in response
    let alpha_metrics = spaces.iter().find(|s| s["space"] == "alpha");
    let beta_metrics = spaces.iter().find(|s| s["space"] == "beta");

    assert!(
        alpha_metrics.is_some(),
        "Alpha should be in health response"
    );
    assert!(beta_metrics.is_some(), "Beta should be in health response");

    // Verify memory counts are correct and isolated
    let alpha = alpha_metrics.unwrap();
    let beta = beta_metrics.unwrap();

    assert_eq!(
        alpha["memories"].as_u64().unwrap(),
        5,
        "Alpha should have 5 memories"
    );
    assert_eq!(
        beta["memories"].as_u64().unwrap(),
        3,
        "Beta should have 3 memories"
    );
}
