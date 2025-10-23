//! Tests for gRPC service implementation

#![allow(clippy::uninlined_format_args)]
#![allow(clippy::too_many_lines)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]

use engram_cli::grpc::MemoryService;
use engram_core::{MemorySpaceId, MemorySpaceRegistry, MemoryStore, metrics};
use engram_proto::engram_service_client::EngramServiceClient;
use engram_proto::*;
use std::sync::Arc;
use tonic::Request;

/// Start a test gRPC server and return the port
async fn start_test_grpc_server() -> u16 {
    let mut store_inner = MemoryStore::new(100);
    let _ = store_inner.enable_event_streaming(100);
    let store = Arc::new(store_inner);

    let temp_dir = tempfile::tempdir().expect("temp dir");
    let registry = Arc::new(
        MemorySpaceRegistry::new(temp_dir.path(), {
            let store = Arc::clone(&store);
            move |_space_id, _directories| Ok(Arc::clone(&store))
        })
        .expect("registry"),
    );

    let default_space = MemorySpaceId::default();
    tokio::runtime::Handle::current()
        .block_on(registry.create_or_get(&default_space))
        .expect("default space");

    let metrics = metrics::init();
    let service = MemoryService::new(
        Arc::clone(&store),
        metrics,
        Arc::clone(&registry),
        default_space,
    );

    // Find an available port
    let port = portpicker::pick_unused_port().expect("No available ports");

    // Start server in background
    tokio::spawn(async move {
        service.serve(port).await.ok();
    });

    // Give server time to start
    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

    port
}

#[tokio::test]
#[ignore = "Ignore by default since it requires server startup"]
async fn test_grpc_remember_memory() {
    let port = start_test_grpc_server().await;
    let addr = format!("http://127.0.0.1:{}", port);

    let mut client = EngramServiceClient::connect(addr)
        .await
        .expect("Failed to connect to gRPC server");

    // Create a memory to remember
    let memory = Memory::new("test_memory", vec![0.1; 768]).with_content("Test memory content");

    let request = Request::new(RememberRequest {
        memory_space_id: String::new(), // Empty = use default space
        memory_type: Some(remember_request::MemoryType::Memory(memory)),
        auto_link: false,
        link_threshold: 0.5,
    });

    let response = client.remember(request).await.expect("Remember failed");
    let res = response.into_inner();

    assert_eq!(res.memory_id, "test_memory");
    assert!(res.storage_confidence.is_some());
    assert_eq!(res.initial_state, ConsolidationState::Recent as i32);
}

#[tokio::test]
#[ignore = "Ignore by default since it requires server startup"]
async fn test_grpc_remember_episode() {
    let port = start_test_grpc_server().await;
    let addr = format!("http://127.0.0.1:{}", port);

    let mut client = EngramServiceClient::connect(addr)
        .await
        .expect("Failed to connect to gRPC server");

    // Create an episode to remember
    let episode = Episode::new(
        "episode_test",
        chrono::Utc::now(),
        "Test event occurred",
        vec![0.2; 768],
    );

    let request = Request::new(RememberRequest {
        memory_space_id: String::new(), // Empty = use default space
        memory_type: Some(remember_request::MemoryType::Episode(episode)),
        auto_link: false,
        link_threshold: 0.5,
    });

    let response = client.remember(request).await.expect("Remember failed");
    let res = response.into_inner();

    assert_eq!(res.memory_id, "episode_test");
    assert!(res.storage_confidence.is_some());
}

#[tokio::test]
#[ignore = "Ignore by default since it requires server startup"]
async fn test_grpc_recall() {
    let port = start_test_grpc_server().await;
    let addr = format!("http://127.0.0.1:{}", port);

    let mut client = EngramServiceClient::connect(addr)
        .await
        .expect("Failed to connect to gRPC server");

    let cue = Cue::from_query("test query");

    let request = Request::new(RecallRequest {
        memory_space_id: String::new(), // Empty = use default space
        cue: Some(cue),
        max_results: 10,
        include_metadata: true,
        trace_activation: false,
    });

    let response = client.recall(request).await.expect("Recall failed");
    let res = response.into_inner();

    assert!(res.recall_confidence.is_some());
    assert!(res.metadata.is_some());
}

#[tokio::test]
#[ignore = "Ignore by default since it requires server startup"]
async fn test_grpc_recognize() {
    let port = start_test_grpc_server().await;
    let addr = format!("http://127.0.0.1:{}", port);

    let mut client = EngramServiceClient::connect(addr)
        .await
        .expect("Failed to connect to gRPC server");

    let request = Request::new(RecognizeRequest {
        memory_space_id: String::new(), // Empty = use default space
        input: Some(recognize_request::Input::Content(
            "test content".to_string(),
        )),
        recognition_threshold: 0.5,
    });

    let response = client.recognize(request).await.expect("Recognize failed");
    let res = response.into_inner();

    assert!(!res.recognized); // Should not recognize unknown content
    assert!(res.recognition_confidence.is_some());
}

#[tokio::test]
#[ignore = "Ignore by default since it requires server startup"]
async fn test_grpc_forget() {
    let port = start_test_grpc_server().await;
    let addr = format!("http://127.0.0.1:{}", port);

    let mut client = EngramServiceClient::connect(addr)
        .await
        .expect("Failed to connect to gRPC server");

    let request = Request::new(ForgetRequest {
        memory_space_id: String::new(), // Empty = use default space
        target: Some(forget_request::Target::MemoryId("test_id".to_string())),
        mode: ForgetMode::Suppress as i32,
    });

    let response = client.forget(request).await.expect("Forget failed");
    let res = response.into_inner();

    assert!(res.reversible); // Suppress mode should be reversible
    assert!(res.forget_confidence.is_some());
}

#[tokio::test]
#[ignore = "Ignore by default since it requires server startup"]
async fn test_grpc_introspect() {
    let port = start_test_grpc_server().await;
    let addr = format!("http://127.0.0.1:{}", port);

    let mut client = EngramServiceClient::connect(addr)
        .await
        .expect("Failed to connect to gRPC server");

    let request = Request::new(IntrospectRequest {
        memory_space_id: String::new(), // Empty = use default space
        metrics: vec!["memory_count".to_string()],
        include_health: true,
        include_statistics: true,
    });

    let response = client.introspect(request).await.expect("Introspect failed");
    let res = response.into_inner();

    assert!(res.health.is_some());
    assert!(res.health.as_ref().unwrap().healthy);
    assert!(res.statistics.is_some());
    assert!(!res.active_processes.is_empty());
    assert!(!res.metrics_snapshot_json.is_empty());
}

#[tokio::test]
#[ignore = "Ignore by default since it requires server startup"]
async fn test_grpc_error_handling() {
    let port = start_test_grpc_server().await;
    let addr = format!("http://127.0.0.1:{}", port);

    let mut client = EngramServiceClient::connect(addr)
        .await
        .expect("Failed to connect to gRPC server");

    // Test Remember without memory content
    let request = Request::new(RememberRequest {
        memory_space_id: String::new(), // Empty = use default space
        memory_type: None,
        auto_link: false,
        link_threshold: 0.5,
    });

    let response = client.remember(request).await;
    assert!(response.is_err());
    let error = response.unwrap_err();
    assert_eq!(error.code(), tonic::Code::InvalidArgument);

    // Test Recall without cue
    let request = Request::new(RecallRequest {
        memory_space_id: String::new(), // Empty = use default space
        cue: None,
        max_results: 10,
        include_metadata: false,
        trace_activation: false,
    });

    let response = client.recall(request).await;
    assert!(response.is_err());
    let error = response.unwrap_err();
    assert_eq!(error.code(), tonic::Code::InvalidArgument);
}
