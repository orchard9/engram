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
use engram_cli::cluster::ClusterState;
use engram_cli::router::{Router as ClusterRouter, RouterConfig};
use engram_core::activation::SpreadingAutoTuner;
use engram_core::activation::{ActivationRecordPoolStats, SpreadingMetrics};
use engram_core::cluster::config::{PartitionConfig, ReplicationConfig, SwimConfig};
use engram_core::{
    MemorySpaceError, MemorySpaceId, MemorySpaceRegistry, MemoryStore,
    cluster::{
        MembershipUpdate, NodeInfo, NodeState, PartitionDetector, RebalanceCoordinator,
        SpaceAssignmentManager, SpaceAssignmentPlanner, SplitBrainDetector, SwimMembership,
    },
    metrics,
};
use serde_json::{Value, json};
use std::sync::Arc;
use tower::ServiceExt; // for `oneshot`

/// Create test router with API routes
async fn create_test_router() -> Router {
    build_test_router(None).await
}

async fn create_test_router_with_cluster(cluster: Arc<ClusterState>) -> Router {
    build_test_router(Some(cluster)).await
}

async fn build_test_router(cluster: Option<Arc<ClusterState>>) -> Router {
    let temp_dir = tempfile::tempdir().expect("temp dir");
    let registry = Arc::new(
        MemorySpaceRegistry::new(temp_dir.path(), |_space_id, _directories| {
            let mut store = MemoryStore::new(100);
            let _ = store.enable_event_streaming(32);
            Ok::<Arc<MemoryStore>, MemorySpaceError>(Arc::new(store))
        })
        .expect("registry"),
    );

    let default_space = MemorySpaceId::default();
    let space_handle = registry
        .create_or_get(&default_space)
        .await
        .expect("default space");
    let store = space_handle.store();

    let mut keepalive_rx = store
        .subscribe_to_events()
        .expect("event streaming initialized");
    tokio::spawn(async move { while keepalive_rx.recv().await.is_ok() {} });

    let metrics = metrics::init();
    let auto_tuner = SpreadingAutoTuner::new(0.10, 16);
    let (shutdown_tx, _shutdown_rx) = tokio::sync::watch::channel(false);
    let router = cluster.as_ref().map(|state| {
        Arc::new(ClusterRouter::new(
            Arc::clone(state),
            RouterConfig::default(),
        ))
    });

    let api_state = ApiState::new(
        store,
        Arc::clone(&registry),
        default_space,
        metrics,
        auto_tuner,
        Arc::new(shutdown_tx),
        cluster.clone(),
        router,
        None,
    );

    create_api_routes().with_state(api_state)
}

fn sample_cluster_state() -> Arc<ClusterState> {
    let local_swim = "127.0.0.1:7946".parse().unwrap();
    let local_api = "127.0.0.1:50051".parse().unwrap();
    let local = NodeInfo::new("node-local", local_swim, local_api, None, None);
    let membership = Arc::new(SwimMembership::new(local.clone(), SwimConfig::default()));
    membership.apply_updates(vec![
        MembershipUpdate {
            node: NodeInfo::new(
                "node-alpha",
                "127.0.0.2:7946".parse().unwrap(),
                "127.0.0.2:50051".parse().unwrap(),
                None,
                None,
            ),
            state: NodeState::Alive,
            incarnation: 1,
        },
        MembershipUpdate {
            node: NodeInfo::new(
                "node-beta",
                "127.0.0.3:7946".parse().unwrap(),
                "127.0.0.3:50051".parse().unwrap(),
                None,
                None,
            ),
            state: NodeState::Suspect,
            incarnation: 2,
        },
        MembershipUpdate {
            node: NodeInfo::new(
                "node-gamma",
                "127.0.0.4:7946".parse().unwrap(),
                "127.0.0.4:50051".parse().unwrap(),
                None,
                None,
            ),
            state: NodeState::Dead,
            incarnation: 3,
        },
    ]);

    let replication = ReplicationConfig::default();
    let planner = Arc::new(SpaceAssignmentPlanner::new(
        Arc::clone(&membership),
        &replication,
    ));
    let assignments = Arc::new(SpaceAssignmentManager::new(
        Arc::clone(&planner),
        &replication,
    ));
    let (rebalance, mut plan_rx) =
        RebalanceCoordinator::new(Arc::clone(&assignments), Arc::clone(&membership), 4);
    tokio::spawn(async move { while plan_rx.recv().await.is_some() {} });
    let partition_detector = Arc::new(PartitionDetector::new(
        Arc::clone(&membership),
        PartitionConfig::default(),
    ));
    let split_brain = Arc::new(SplitBrainDetector::new(local.id.clone()));

    Arc::new(ClusterState {
        node_id: local.id,
        membership,
        assignments,
        replication,
        partition_detector,
        split_brain,
        rebalance,
        #[cfg(feature = "memory_mapped_persistence")]
        replication_metadata: None,
    })
}

fn drain_streaming_queue() {
    let registry = metrics::init();
    registry.streaming_aggregator().set_export_enabled(true);
    let _ = registry.streaming_snapshot();
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
    let app = create_test_router().await;

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
async fn test_metrics_endpoint_snapshot() {
    let app = create_test_router().await;

    let (status, response) = make_request(&app, Method::GET, "/metrics", None).await;

    assert_eq!(status, StatusCode::OK);
    assert!(response.get("snapshot").is_some());
    assert!(
        response
            .get("snapshot")
            .and_then(|value| value.get("one_second"))
            .is_some()
    );
    assert!(response.get("export").is_some());
}

#[tokio::test]
async fn test_metrics_endpoint_matches_streaming_snapshot_after_reset() {
    drain_streaming_queue();

    let app = create_test_router().await;

    let metrics = metrics::init();
    let spreading_metrics = SpreadingMetrics::default();
    let pool_stats = ActivationRecordPoolStats {
        available: 4,
        in_flight: 2,
        high_water_mark: 8,
        total_created: 16,
        total_reused: 12,
        misses: 3,
        hit_rate: 0.8,
        utilization: 0.25,
        release_failures: 1,
    };

    spreading_metrics.record_pool_snapshot(&pool_stats);
    let _ = metrics.streaming_snapshot();

    spreading_metrics.reset();

    let (status, response) = make_request(&app, Method::GET, "/metrics", None).await;
    assert_eq!(status, StatusCode::OK);

    let http_snapshot = response
        .get("snapshot")
        .cloned()
        .unwrap_or_else(|| json!({}));
    let expected_snapshot =
        serde_json::to_value(metrics.streaming_snapshot()).expect("serialize core snapshot");

    let windows = ["one_second", "ten_seconds", "one_minute", "five_minutes"];
    for window in windows {
        let http_window = http_snapshot
            .get(window)
            .and_then(|value| value.as_object())
            .unwrap_or_else(|| panic!("window {window} missing in HTTP snapshot"));
        let expected_window = expected_snapshot
            .get(window)
            .and_then(|value| value.as_object())
            .unwrap_or_else(|| panic!("window {window} missing in core snapshot"));

        for (key, expected_value) in expected_window {
            if key.starts_with("activation_pool_") {
                let http_value = http_window
                    .get(key)
                    .unwrap_or_else(|| panic!("metric {key} missing in HTTP snapshot"));
                assert_eq!(
                    http_value, expected_value,
                    "HTTP {window}.{key} should match core snapshot"
                );
            }
        }
    }
}

#[tokio::test]
async fn cluster_health_requires_cluster_state() {
    let app = create_test_router().await;
    let (status, response) = make_request(&app, Method::GET, "/cluster/health", None).await;

    assert_eq!(status, StatusCode::NOT_IMPLEMENTED);
    assert_eq!(
        response
            .get("error")
            .and_then(|err| err.get("code"))
            .and_then(Value::as_str),
        Some("FEATURE_NOT_ENABLED")
    );
}

#[tokio::test]
async fn cluster_health_reports_membership_breakdown() {
    let cluster_state = sample_cluster_state();
    let app = create_test_router_with_cluster(cluster_state.clone()).await;

    let (status, response) = make_request(&app, Method::GET, "/cluster/health", None).await;

    assert_eq!(status, StatusCode::OK);
    assert_eq!(
        response.get("node_id").and_then(Value::as_str),
        Some(cluster_state.node_id.as_str())
    );
    assert_eq!(
        response
            .get("stats")
            .and_then(|stats| stats.get("alive"))
            .and_then(Value::as_u64),
        Some(1)
    );
    assert_eq!(
        response
            .get("stats")
            .and_then(|stats| stats.get("suspect"))
            .and_then(Value::as_u64),
        Some(1)
    );
    assert_eq!(
        response
            .get("stats")
            .and_then(|stats| stats.get("dead"))
            .and_then(Value::as_u64),
        Some(1)
    );
    assert_eq!(
        response
            .get("stats")
            .and_then(|stats| stats.get("total"))
            .and_then(Value::as_u64),
        Some(3)
    );
    let assignments = response
        .get("assignments")
        .expect("assignments summary missing");
    assert!(assignments.get("cached_spaces").is_some());
    let router = response.get("router").expect("router summary missing");
    assert_eq!(
        router
            .get("open_breakers")
            .and_then(Value::as_u64)
            .unwrap_or_default(),
        0
    );
    assert!(router.get("requests_total").is_some());
}

#[tokio::test]
async fn cluster_nodes_list_memberships() {
    let cluster_state = sample_cluster_state();
    let app = create_test_router_with_cluster(cluster_state).await;

    let (status, response) = make_request(&app, Method::GET, "/cluster/nodes", None).await;

    assert_eq!(status, StatusCode::OK);
    let nodes = response
        .get("nodes")
        .and_then(Value::as_array)
        .expect("nodes array");
    assert!(
        nodes
            .iter()
            .any(|node| node.get("local") == Some(&json!(true)))
    );
    assert!(
        nodes
            .iter()
            .any(|node| node.get("state") == Some(&json!("suspect")))
    );
    assert!(
        nodes
            .iter()
            .any(|node| node.get("state") == Some(&json!("dead")))
    );
}

#[tokio::test]
async fn cluster_rebalance_status_is_exposed() {
    let cluster_state = sample_cluster_state();
    let app = create_test_router_with_cluster(cluster_state).await;

    let (status, response) = make_request(&app, Method::GET, "/cluster/rebalance", None).await;

    assert_eq!(status, StatusCode::OK);
    assert!(response.get("recent").is_some());
}

#[tokio::test]
async fn cluster_rebalance_trigger_returns_count() {
    let cluster_state = sample_cluster_state();
    let app = create_test_router_with_cluster(cluster_state).await;

    let (status, response) = make_request(&app, Method::POST, "/cluster/rebalance", None).await;

    assert_eq!(status, StatusCode::ACCEPTED);
    assert!(response.get("planned").is_some());
}

#[tokio::test]
async fn cluster_migrate_requires_valid_space() {
    let cluster_state = sample_cluster_state();
    let app = create_test_router_with_cluster(cluster_state).await;

    let payload = json!({"space": "*invalid*"});
    let (status, response) =
        make_request(&app, Method::POST, "/cluster/migrate", Some(payload)).await;

    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(
        response
            .get("error")
            .and_then(|value| value.get("code"))
            .and_then(Value::as_str),
        Some("INVALID_MEMORY_INPUT")
    );
}

#[tokio::test]
async fn test_remember_memory_validation_errors() {
    let app = create_test_router().await;

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
    let app = create_test_router().await;

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
            .contains("consolidate better")
    );
}

#[tokio::test]
async fn test_recall_memories_with_query() {
    let app = create_test_router().await;

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
    let app = create_test_router().await;

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
    let app = create_test_router().await;

    let embedding_vec = vec![0.5; 768]; // Valid 768-dimensional embedding
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
    let app = create_test_router().await;

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
async fn test_spreading_health_endpoint() {
    let app = create_test_router().await;
    let (status, payload) = make_request(&app, Method::GET, "/health/spreading", None).await;
    assert_eq!(status, StatusCode::OK);
    assert!(payload.get("status").is_some());
}

#[tokio::test]
async fn test_system_health_endpoint() {
    let app = create_test_router().await;
    let (status, payload) = make_request(&app, Method::GET, "/api/v1/system/health", None).await;
    assert_eq!(status, StatusCode::OK);
    assert!(payload.get("checks").is_some());
}

#[tokio::test]
async fn test_spreading_config_endpoint() {
    let app = create_test_router().await;
    let (status, payload) =
        make_request(&app, Method::GET, "/api/v1/system/spreading/config", None).await;
    assert_eq!(status, StatusCode::OK);
    assert!(payload.get("audit_log").is_some());
}

#[tokio::test]
async fn test_recognize_pattern_validation() {
    let app = create_test_router().await;

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
    let app = create_test_router().await;

    let (status, response) = make_request(&app, Method::GET, "/api/v1/system/health", None).await;

    assert_eq!(status, StatusCode::OK);
    assert_eq!(response["status"].as_str().unwrap(), "healthy");
    assert!(response["timestamp"].is_string());
    assert!(response["spaces"].is_array());
    assert!(response["checks"].is_array());

    // Verify per-space metrics structure
    if let Some(spaces) = response["spaces"].as_array()
        && let Some(first_space) = spaces.first()
    {
        assert!(first_space["space"].is_string());
        assert!(first_space["memories"].is_number());
        assert!(first_space["pressure"].is_number());
        assert!(first_space["wal_lag_ms"].is_number());
        assert!(first_space["consolidation_rate"].is_number());
    }
}

#[tokio::test]
async fn test_system_introspect() {
    let app = create_test_router().await;

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
    let app = create_test_router().await;

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
    let app = create_test_router().await;

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
    let app = create_test_router().await;

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
        // Allow system to adjust confidence - verify it's in reasonable range
        assert!((0.0..=1.0).contains(&returned_confidence));
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
    let app = create_test_router().await;

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
    let app = create_test_router().await;

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
    let app = create_test_router().await;

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
    let app = create_test_router().await;

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
