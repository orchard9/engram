//! Engram CLI library functions for testing

// Clippy configuration: Only allow specific patterns with justification
#![allow(clippy::multiple_crate_versions)] // Dependencies control their own versions

pub mod api;
pub mod benchmark_simple;
pub mod cli;
pub mod docs;
pub mod grpc;
pub mod openapi;

use anyhow::Result;
use std::time::Duration;
use tokio::net::TcpListener;

/// Check if a port is available
pub async fn is_port_available(port: u16) -> bool {
    TcpListener::bind(format!("127.0.0.1:{port}")).await.is_ok()
}

/// Find an available port starting from the preferred port
///
/// # Errors
///
/// Returns error if no available ports can be found
pub async fn find_available_port(preferred_port: u16) -> Result<u16> {
    // Try the preferred port first
    if is_port_available(preferred_port).await {
        return Ok(preferred_port);
    }

    // If preferred port is taken, try nearby ports
    for offset in 1..=100 {
        let port = preferred_port.saturating_add(offset);
        if port == 0 {
            // Handle wraparound
            break;
        }
        if is_port_available(port).await {
            return Ok(port);
        }
    }

    Err(anyhow::anyhow!(
        "No available ports found near {}",
        preferred_port
    ))
}

/// Start a test server on a specific port for integration tests
///
/// # Errors
///
/// Returns error if server cannot bind to port or start
pub async fn start_test_server(port: u16) -> Result<()> {
    use axum::{Router, response::Json, routing::get};
    use serde_json::json;
    use std::net::SocketAddr;

    let health = || async {
        Json(json!({
            "status": "healthy",
            "service": "engram",
            "version": env!("CARGO_PKG_VERSION"),
            "timestamp": chrono::Utc::now().to_rfc3339()
        }))
    };

    let app = Router::new()
        .route("/health", get(health))
        .route("/", get(|| async { "Engram Test Server" }));

    let addr = SocketAddr::from(([127, 0, 0, 1], port));
    let listener = TcpListener::bind(addr).await?;

    // Run server in background with timeout
    tokio::select! {
        _ = axum::serve(listener, app) => {},
        () = tokio::time::sleep(Duration::from_secs(1)) => {
            // Server started successfully, timeout to prevent hanging
        }
    }

    Ok(())
}
