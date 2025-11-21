//! Integration tests for the `engram space` CLI commands.

use engram_cli::api::{ApiState, create_api_routes};
use engram_cli::config::SecurityConfig;
use engram_core::activation::SpreadingAutoTuner;
use engram_core::{MemorySpaceError, MemorySpaceId, MemorySpaceRegistry, MemoryStore, metrics};
use std::sync::Arc;
use tokio::net::TcpListener;

/// Spin up an HTTP server backed by a real registry and drive the CLI helpers.
#[allow(unsafe_code)]
#[tokio::test]
async fn test_space_list_and_create_commands() {
    // Build registry with an initial default space so list has content.
    let temp_dir = tempfile::tempdir().expect("temp dir");
    let registry = Arc::new(
        MemorySpaceRegistry::new(temp_dir.path(), |_space_id, _directories| {
            let mut store = MemoryStore::new(32);
            let _ = store.enable_event_streaming(16);
            Ok::<Arc<MemoryStore>, MemorySpaceError>(Arc::new(store))
        })
        .expect("registry"),
    );

    let default_space = MemorySpaceId::default();
    registry
        .create_or_get(&default_space)
        .await
        .expect("default space");

    // Prepare API state.
    let metrics = metrics::init();
    let auto_tuner = SpreadingAutoTuner::new(0.10, 16);
    let (shutdown_tx, _shutdown_rx) = tokio::sync::watch::channel(false);
    let api_state = ApiState::new(
        registry
            .get(&default_space)
            .expect("default handle")
            .store(),
        Arc::clone(&registry),
        default_space.clone(),
        metrics,
        auto_tuner,
        Arc::new(shutdown_tx),
        None,                                // cluster
        None,                                // router
        None,                                // partition_confidence
        Arc::new(SecurityConfig::default()), // auth_config
        None,                                // auth_validator
    );

    // Compose router with the control-plane routes and a simple liveness endpoint.
    let router = create_api_routes().with_state(api_state);

    let listener = TcpListener::bind(("127.0.0.1", 0)).await.expect("bind");
    let port = listener.local_addr().expect("addr").port();
    let server_handle = tokio::spawn(async move {
        let _ = axum::serve(listener, router).await;
    });

    // Prepare fake PID file for get_server_connection().
    let pid_path = temp_dir.path().join("cli-space.pid");
    unsafe {
        std::env::set_var("ENGRAM_PID_PATH", &pid_path);
    }
    std::fs::write(&pid_path, format!("1234:{port}")).expect("write pid");

    // Drive the CLI helpers - output is printed to stdout but we only assert success.
    engram_cli::cli::space::list_spaces()
        .await
        .expect("list spaces");
    engram_cli::cli::space::create_space("tenant_integration".to_string())
        .await
        .expect("create space");

    // Shutdown server task.
    server_handle.abort();

    unsafe {
        std::env::remove_var("ENGRAM_PID_PATH");
    }
}
