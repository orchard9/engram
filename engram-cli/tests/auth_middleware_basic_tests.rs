//! Basic integration tests for HTTP authentication middleware.
//!
//! Tests fundamental middleware application and auth-disabled behavior.

#![allow(clippy::uninlined_format_args)]

use axum::{
    Router,
    body::Body,
    http::{Method, Request, StatusCode},
};
use engram_cli::api::{ApiState, create_api_routes};
use engram_cli::config::SecurityConfig;
use engram_core::activation::SpreadingAutoTuner;
use engram_core::{MemorySpaceId, MemorySpaceRegistry, MemoryStore, metrics};
use std::sync::Arc;
use tower::ServiceExt;

/// Create test router with auth disabled
async fn create_test_router_no_auth() -> Router {
    let temp_dir = tempfile::tempdir().unwrap();

    let registry = Arc::new(
        MemorySpaceRegistry::new(temp_dir.path(), |_space_id, _directories| {
            let store = MemoryStore::new(100);
            Ok::<Arc<MemoryStore>, engram_core::MemorySpaceError>(Arc::new(store))
        })
        .unwrap(),
    );

    let default_space = MemorySpaceId::default();
    let space_handle = registry.create_or_get(&default_space).await.unwrap();
    let memory_store = space_handle.store();

    let (shutdown_tx, _shutdown_rx) = tokio::sync::watch::channel(false);

    let api_state = ApiState::new(
        memory_store,
        Arc::clone(&registry),
        default_space,
        metrics::init(),
        SpreadingAutoTuner::new(0.10, 16),
        Arc::new(shutdown_tx),
        None,                                // cluster
        None,                                // router
        None,                                // partition_confidence
        Arc::new(SecurityConfig::default()), // auth_config
        None,                                // No validator - auth disabled
    );

    create_api_routes().with_state(api_state)
}

#[tokio::test]
async fn test_auth_disabled_allows_access_to_protected_routes() {
    let router = create_test_router_no_auth().await;

    // Test that protected routes work without auth when auth is disabled
    let request = Request::builder()
        .method(Method::GET)
        .uri("/api/v1/memories/recall?query=test")
        // No Authorization header
        .body(Body::empty())
        .unwrap();

    let response = router.oneshot(request).await.unwrap();

    // Should not return auth errors when auth is disabled
    assert_ne!(
        response.status(),
        StatusCode::UNAUTHORIZED,
        "Auth disabled should not require API key"
    );
    assert_ne!(
        response.status(),
        StatusCode::FORBIDDEN,
        "Auth disabled should not check permissions"
    );
}

#[tokio::test]
async fn test_public_routes_accessible() {
    let router = create_test_router_no_auth().await;

    // Test various public routes
    let public_routes = vec![
        "/health",
        "/health/alive",
        "/api/v1/system/health",
        "/metrics",
    ];

    for route in public_routes {
        let request = Request::builder()
            .method(Method::GET)
            .uri(route)
            .body(Body::empty())
            .unwrap();

        let response = router.clone().oneshot(request).await.unwrap();

        assert_ne!(
            response.status(),
            StatusCode::UNAUTHORIZED,
            "Public route {} should not require auth",
            route
        );
        assert_ne!(
            response.status(),
            StatusCode::FORBIDDEN,
            "Public route {} should be accessible",
            route
        );
    }
}

#[tokio::test]
async fn test_security_headers_applied() {
    let router = create_test_router_no_auth().await;

    let request = Request::builder()
        .method(Method::GET)
        .uri("/health")
        .body(Body::empty())
        .unwrap();

    let response = router.oneshot(request).await.unwrap();

    // Verify security headers are present
    let headers = response.headers();

    assert_eq!(
        headers
            .get("X-Content-Type-Options")
            .map(|h| h.to_str().unwrap()),
        Some("nosniff"),
        "X-Content-Type-Options header should be set"
    );

    assert_eq!(
        headers.get("X-Frame-Options").map(|h| h.to_str().unwrap()),
        Some("DENY"),
        "X-Frame-Options header should be set"
    );

    assert_eq!(
        headers.get("X-XSS-Protection").map(|h| h.to_str().unwrap()),
        Some("1; mode=block"),
        "X-XSS-Protection header should be set"
    );
}
