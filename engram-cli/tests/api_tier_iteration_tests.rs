//! Comprehensive tests for tier-aware memory listing API (Phase 1.2)
//!
//! Tests HTTP-level behavior of GET /api/v1/memories endpoint including:
//! - Query parameter validation and parsing
//! - Pagination edge cases
//! - Response format and structure
//! - Backward compatibility
//! - Error handling
//! - Embedding inclusion behavior

use axum::{
    Router,
    body::Body,
    http::{Method, Request, StatusCode},
};
use chrono::Utc;
use engram_cli::api::{ApiState, create_api_routes};
use engram_core::activation::SpreadingAutoTuner;
use engram_core::{
    Confidence, EpisodeBuilder, MemorySpaceError, MemorySpaceId, MemorySpaceRegistry, MemoryStore,
    metrics,
};
use serde_json::{Value, json};
use std::sync::Arc;
use tower::ServiceExt;

/// Create test router with pre-populated memory store
async fn create_test_router_with_memories(num_memories: usize) -> Router {
    let temp_dir = tempfile::tempdir().expect("temp dir");
    let registry = Arc::new(
        MemorySpaceRegistry::new(temp_dir.path(), |_space_id, _directories| {
            let mut store = MemoryStore::new(1000);
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

    // Populate with test memories
    // NOTE: SemanticDeduplicator has default threshold of 0.95, so embeddings must be
    // sufficiently distinct (< 0.95 cosine similarity) to avoid deduplication.
    // We use orthogonal-ish vectors to ensure uniqueness.
    for i in 0..num_memories {
        let mut embedding = [0.0f32; 768];

        // Create sparse, orthogonal-ish embeddings to avoid deduplication
        // Strategy: Each memory "activates" a different subset of dimensions
        let base_offset = (i % 256) * 3; // Spread across embedding space
        embedding[base_offset] = 1.0;
        embedding[(base_offset + 1) % 768] = (i as f32 * 0.1).sin();
        embedding[(base_offset + 2) % 768] = (i as f32 * 0.1).cos();

        // Add some noise to remaining dimensions to ensure distinctness
        for j in 0..10 {
            let idx = (i * 73 + j * 97) % 768;
            embedding[idx] = ((i + j) as f32 * 0.01).sin();
        }

        let ep = EpisodeBuilder::new()
            .id(format!("memory_{i:04}"))
            .when(Utc::now())
            .what(format!("Unique test memory content number {i}"))
            .embedding(embedding)
            .confidence(Confidence::HIGH)
            .build();

        store.store(ep);
    }

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
        None,
        None,
        None,
    );

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

// ============================================================================
// Basic Functionality Tests
// ============================================================================

#[tokio::test]
async fn test_list_memories_default_params() {
    let app = create_test_router_with_memories(150).await;

    let (status, json) = make_request(&app, Method::GET, "/api/v1/memories", None).await;

    assert_eq!(status, StatusCode::OK);

    // Verify response structure
    assert!(json["memories"].is_array());
    assert!(json["count"].is_number());
    assert!(json["pagination"].is_object());
    assert!(json["tier_counts"].is_object());

    // Default behavior: tier=hot, offset=0, limit=100
    let count = json["count"].as_u64().unwrap() as usize;
    assert_eq!(count, 100, "Default limit should be 100");

    let pagination = &json["pagination"];
    assert_eq!(pagination["offset"].as_u64().unwrap(), 0);
    assert_eq!(pagination["limit"].as_u64().unwrap(), 100);
    assert_eq!(pagination["returned"].as_u64().unwrap(), 100);
}

#[tokio::test]
async fn test_list_memories_custom_limit() {
    let app = create_test_router_with_memories(150).await;

    let (status, json) = make_request(&app, Method::GET, "/api/v1/memories?limit=50", None).await;

    assert_eq!(status, StatusCode::OK);

    let count = json["count"].as_u64().unwrap() as usize;
    assert_eq!(count, 50);

    let pagination = &json["pagination"];
    assert_eq!(pagination["returned"].as_u64().unwrap(), 50);
}

#[tokio::test]
async fn test_list_memories_with_offset() {
    let app = create_test_router_with_memories(150).await;

    let (status, json) = make_request(
        &app,
        Method::GET,
        "/api/v1/memories?offset=100&limit=50",
        None,
    )
    .await;

    assert_eq!(status, StatusCode::OK);

    let count = json["count"].as_u64().unwrap() as usize;
    assert_eq!(count, 50, "Should return 50 items from offset 100");

    let pagination = &json["pagination"];
    assert_eq!(pagination["offset"].as_u64().unwrap(), 100);
    assert_eq!(pagination["returned"].as_u64().unwrap(), 50);
}

#[tokio::test]
async fn test_list_memories_empty_store() {
    let app = create_test_router_with_memories(0).await;

    let (status, json) = make_request(&app, Method::GET, "/api/v1/memories", None).await;

    assert_eq!(status, StatusCode::OK);

    let memories = json["memories"].as_array().unwrap();
    assert_eq!(memories.len(), 0);

    let count = json["count"].as_u64().unwrap() as usize;
    assert_eq!(count, 0);

    let tier_counts = &json["tier_counts"];
    assert_eq!(tier_counts["hot"].as_u64().unwrap(), 0);
    assert_eq!(tier_counts["total"].as_u64().unwrap(), 0);
}

// ============================================================================
// Pagination Edge Cases
// ============================================================================

#[tokio::test]
async fn test_pagination_offset_beyond_total() {
    let app = create_test_router_with_memories(50).await;

    let (status, json) = make_request(
        &app,
        Method::GET,
        "/api/v1/memories?offset=100&limit=50",
        None,
    )
    .await;

    assert_eq!(status, StatusCode::OK);

    let memories = json["memories"].as_array().unwrap();
    assert_eq!(
        memories.len(),
        0,
        "Should return empty array when offset > total"
    );

    let count = json["count"].as_u64().unwrap() as usize;
    assert_eq!(count, 0);
}

#[tokio::test]
async fn test_pagination_partial_page() {
    let app = create_test_router_with_memories(75).await;

    let (status, json) = make_request(
        &app,
        Method::GET,
        "/api/v1/memories?offset=50&limit=100",
        None,
    )
    .await;

    assert_eq!(status, StatusCode::OK);

    let count = json["count"].as_u64().unwrap() as usize;
    assert_eq!(count, 25, "Should return remaining 25 items (75 - 50)");

    let pagination = &json["pagination"];
    assert_eq!(pagination["returned"].as_u64().unwrap(), 25);
}

#[tokio::test]
async fn test_pagination_limit_clamping() {
    let app = create_test_router_with_memories(2000).await;

    // Request more than max allowed (1000)
    let (status, json) = make_request(&app, Method::GET, "/api/v1/memories?limit=5000", None).await;

    assert_eq!(status, StatusCode::OK);

    let count = json["count"].as_u64().unwrap() as usize;
    assert_eq!(count, 1000, "Should clamp limit to 1000");

    let pagination = &json["pagination"];
    assert_eq!(pagination["limit"].as_u64().unwrap(), 1000);
    assert_eq!(pagination["returned"].as_u64().unwrap(), 1000);
}

#[tokio::test]
async fn test_pagination_consistency_across_pages() {
    let app = create_test_router_with_memories(150).await;

    // Fetch first page
    let (_status1, json1) = make_request(
        &app,
        Method::GET,
        "/api/v1/memories?offset=0&limit=50",
        None,
    )
    .await;

    // Fetch second page
    let (_status2, json2) = make_request(
        &app,
        Method::GET,
        "/api/v1/memories?offset=50&limit=50",
        None,
    )
    .await;

    // Fetch third page
    let (_status3, json3) = make_request(
        &app,
        Method::GET,
        "/api/v1/memories?offset=100&limit=50",
        None,
    )
    .await;

    let memories1 = json1["memories"].as_array().unwrap();
    let memories2 = json2["memories"].as_array().unwrap();
    let memories3 = json3["memories"].as_array().unwrap();

    assert_eq!(memories1.len(), 50);
    assert_eq!(memories2.len(), 50);
    assert_eq!(memories3.len(), 50);

    // Verify no duplicates across pages
    let ids1: Vec<String> = memories1
        .iter()
        .map(|m| m["id"].as_str().unwrap().to_string())
        .collect();
    let ids2: Vec<String> = memories2
        .iter()
        .map(|m| m["id"].as_str().unwrap().to_string())
        .collect();
    let ids3: Vec<String> = memories3
        .iter()
        .map(|m| m["id"].as_str().unwrap().to_string())
        .collect();

    // No overlaps
    for id in &ids1 {
        assert!(!ids2.contains(id), "Page 1 and 2 should not overlap");
        assert!(!ids3.contains(id), "Page 1 and 3 should not overlap");
    }
    for id in &ids2 {
        assert!(!ids3.contains(id), "Page 2 and 3 should not overlap");
    }
}

// ============================================================================
// Tier Selection Tests
// ============================================================================

#[tokio::test]
async fn test_tier_hot_explicit() {
    let app = create_test_router_with_memories(50).await;

    let (status, json) = make_request(&app, Method::GET, "/api/v1/memories?tier=hot", None).await;

    assert_eq!(status, StatusCode::OK);
    assert!(json["memories"].is_array());
}

#[tokio::test]
async fn test_tier_hot_case_insensitive() {
    let app = create_test_router_with_memories(50).await;

    let (status, json) = make_request(&app, Method::GET, "/api/v1/memories?tier=HOT", None).await;

    assert_eq!(status, StatusCode::OK);
    assert!(json["memories"].is_array());
}

#[tokio::test]
async fn test_tier_warm_not_implemented() {
    let app = create_test_router_with_memories(50).await;

    let (status, json) = make_request(&app, Method::GET, "/api/v1/memories?tier=warm", None).await;

    assert_eq!(status, StatusCode::NOT_IMPLEMENTED);
    assert_eq!(json["error"]["code"], "NOT_IMPLEMENTED");
    assert!(json["error"]["message"].as_str().unwrap().contains("warm"));
}

#[tokio::test]
async fn test_tier_cold_not_implemented() {
    let app = create_test_router_with_memories(50).await;

    let (status, json) = make_request(&app, Method::GET, "/api/v1/memories?tier=cold", None).await;

    assert_eq!(status, StatusCode::NOT_IMPLEMENTED);
    assert_eq!(json["error"]["code"], "NOT_IMPLEMENTED");
}

#[tokio::test]
async fn test_tier_all_not_implemented() {
    let app = create_test_router_with_memories(50).await;

    let (status, json) = make_request(&app, Method::GET, "/api/v1/memories?tier=all", None).await;

    assert_eq!(status, StatusCode::NOT_IMPLEMENTED);
    assert_eq!(json["error"]["code"], "NOT_IMPLEMENTED");
}

#[tokio::test]
async fn test_tier_invalid_returns_400() {
    let app = create_test_router_with_memories(50).await;

    let (status, json) = make_request(
        &app,
        Method::GET,
        "/api/v1/memories?tier=invalid_tier",
        None,
    )
    .await;

    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(json["error"]["code"], "BAD_REQUEST");
    assert!(
        json["error"]["message"]
            .as_str()
            .unwrap()
            .contains("Invalid tier value")
    );
}

// ============================================================================
// Embedding Inclusion Tests
// ============================================================================

#[tokio::test]
async fn test_embedding_excluded_by_default() {
    let app = create_test_router_with_memories(10).await;

    let (status, json) = make_request(&app, Method::GET, "/api/v1/memories", None).await;

    assert_eq!(status, StatusCode::OK);

    let memories = json["memories"].as_array().unwrap();
    assert!(!memories.is_empty());

    // Verify no embeddings present
    for memory in memories {
        assert!(memory["id"].is_string());
        assert!(memory["content"].is_string());
        assert!(memory["confidence"].is_number());
        assert!(memory["timestamp"].is_string());
        assert!(
            memory["embedding"].is_null(),
            "Embeddings should not be included by default"
        );
    }
}

#[tokio::test]
async fn test_embedding_included_when_requested() {
    let app = create_test_router_with_memories(10).await;

    let (status, json) = make_request(
        &app,
        Method::GET,
        "/api/v1/memories?include_embeddings=true&limit=5",
        None,
    )
    .await;

    assert_eq!(status, StatusCode::OK);

    let memories = json["memories"].as_array().unwrap();
    assert_eq!(memories.len(), 5);

    // Verify embeddings present and valid
    for memory in memories {
        assert!(memory["embedding"].is_array(), "Embedding should be array");

        let embedding = memory["embedding"].as_array().unwrap();
        assert_eq!(embedding.len(), 768, "Embedding should have 768 dimensions");

        // Verify all values are numbers
        for val in embedding {
            assert!(val.is_number(), "Embedding values should be numbers");
        }
    }
}

#[tokio::test]
async fn test_embedding_payload_size_difference() {
    let app = create_test_router_with_memories(10).await;

    // Request without embeddings
    let (_status1, json1) = make_request(
        &app,
        Method::GET,
        "/api/v1/memories?limit=10&include_embeddings=false",
        None,
    )
    .await;

    // Request with embeddings
    let (_status2, json2) = make_request(
        &app,
        Method::GET,
        "/api/v1/memories?limit=10&include_embeddings=true",
        None,
    )
    .await;

    let size_without = json1.to_string().len();
    let size_with = json2.to_string().len();

    // Payload with embeddings should be significantly larger
    // 10 memories × 768 floats × ~8 bytes (JSON representation) ≈ 60KB additional
    assert!(
        size_with > size_without * 5,
        "Payload with embeddings should be much larger: {size_with} vs {size_without}"
    );
}

// ============================================================================
// Response Format Tests
// ============================================================================

#[tokio::test]
async fn test_response_has_required_fields() {
    let app = create_test_router_with_memories(50).await;

    let (status, json) = make_request(&app, Method::GET, "/api/v1/memories", None).await;

    assert_eq!(status, StatusCode::OK);

    // Legacy fields
    assert!(json.get("memories").is_some());
    assert!(json.get("count").is_some());

    // New fields
    assert!(json.get("pagination").is_some());
    assert!(json.get("tier_counts").is_some());
}

#[tokio::test]
async fn test_pagination_metadata_format() {
    let app = create_test_router_with_memories(50).await;

    let (status, json) = make_request(
        &app,
        Method::GET,
        "/api/v1/memories?offset=10&limit=20",
        None,
    )
    .await;

    assert_eq!(status, StatusCode::OK);

    let pagination = &json["pagination"];
    assert!(pagination["offset"].is_number());
    assert!(pagination["limit"].is_number());
    assert!(pagination["returned"].is_number());

    assert_eq!(pagination["offset"].as_u64().unwrap(), 10);
    assert_eq!(pagination["limit"].as_u64().unwrap(), 20);
    assert_eq!(pagination["returned"].as_u64().unwrap(), 20);
}

#[tokio::test]
async fn test_tier_counts_format() {
    let app = create_test_router_with_memories(50).await;

    let (status, json) = make_request(&app, Method::GET, "/api/v1/memories", None).await;

    assert_eq!(status, StatusCode::OK);

    let tier_counts = &json["tier_counts"];
    assert!(tier_counts["hot"].is_number());
    assert!(tier_counts["warm"].is_number());
    assert!(tier_counts["cold"].is_number());
    assert!(tier_counts["total"].is_number());

    let hot = tier_counts["hot"].as_u64().unwrap() as usize;
    let warm = tier_counts["warm"].as_u64().unwrap() as usize;
    let cold = tier_counts["cold"].as_u64().unwrap() as usize;
    let total = tier_counts["total"].as_u64().unwrap() as usize;

    assert_eq!(total, hot + warm + cold, "Total should equal sum of tiers");
    assert_eq!(hot, 50, "Should have 50 hot tier memories");
}

#[tokio::test]
async fn test_memory_object_format() {
    let app = create_test_router_with_memories(10).await;

    let (status, json) = make_request(&app, Method::GET, "/api/v1/memories?limit=1", None).await;

    assert_eq!(status, StatusCode::OK);

    let memories = json["memories"].as_array().unwrap();
    assert_eq!(memories.len(), 1);

    let memory = &memories[0];
    assert!(memory["id"].is_string());
    assert!(memory["content"].is_string());
    assert!(memory["confidence"].is_number());
    assert!(memory["timestamp"].is_string());

    // Verify confidence is in valid range [0.0, 1.0]
    let confidence = memory["confidence"].as_f64().unwrap();
    assert!(
        (0.0..=1.0).contains(&confidence),
        "Confidence should be in [0.0, 1.0]"
    );

    // Verify timestamp is valid RFC3339
    let timestamp_str = memory["timestamp"].as_str().unwrap();
    assert!(
        chrono::DateTime::parse_from_rfc3339(timestamp_str).is_ok(),
        "Timestamp should be valid RFC3339"
    );
}

// ============================================================================
// Backward Compatibility Tests
// ============================================================================

#[tokio::test]
async fn test_legacy_client_can_parse_response() {
    let app = create_test_router_with_memories(50).await;

    let (status, json) = make_request(&app, Method::GET, "/api/v1/memories", None).await;

    assert_eq!(status, StatusCode::OK);

    // Simulate legacy client that only knows about memories and count
    let memories = json["memories"].as_array();
    let count = json["count"].as_u64();

    assert!(memories.is_some(), "Legacy 'memories' field should exist");
    assert!(count.is_some(), "Legacy 'count' field should exist");

    // Legacy client should not break on unknown fields
    // (JSON parsers typically ignore unknown fields by default)
}

#[tokio::test]
async fn test_count_matches_pagination_returned() {
    let app = create_test_router_with_memories(150).await;

    let (status, json) = make_request(&app, Method::GET, "/api/v1/memories?limit=75", None).await;

    assert_eq!(status, StatusCode::OK);

    let count = json["count"].as_u64().unwrap();
    let pagination_returned = json["pagination"]["returned"].as_u64().unwrap();

    assert_eq!(
        count, pagination_returned,
        "Legacy 'count' should match pagination.returned"
    );
}

// ============================================================================
// Error Handling Tests
// ============================================================================

#[tokio::test]
async fn test_error_response_format() {
    let app = create_test_router_with_memories(50).await;

    let (status, json) =
        make_request(&app, Method::GET, "/api/v1/memories?tier=invalid", None).await;

    assert_eq!(status, StatusCode::BAD_REQUEST);

    // Verify error structure
    assert!(json["error"].is_object());
    assert!(json["error"]["code"].is_string());
    assert!(json["error"]["message"].is_string());

    // Cognitive-friendly error messages
    let message = json["error"]["message"].as_str().unwrap();
    assert!(!message.is_empty(), "Error message should not be empty");
    assert!(
        message.contains("Invalid tier"),
        "Error message should explain the problem"
    );
}

#[tokio::test]
async fn test_not_implemented_error_has_helpful_message() {
    let app = create_test_router_with_memories(50).await;

    let (status, json) = make_request(&app, Method::GET, "/api/v1/memories?tier=warm", None).await;

    assert_eq!(status, StatusCode::NOT_IMPLEMENTED);

    let message = json["error"]["message"].as_str().unwrap();
    assert!(
        message.contains("warm") || message.contains("not yet available"),
        "Error should mention the tier name or status"
    );
}

// ============================================================================
// Concurrent Access Tests
// ============================================================================

#[tokio::test]
async fn test_concurrent_pagination_requests() {
    let app = Arc::new(create_test_router_with_memories(200).await);

    // Spawn multiple concurrent requests
    let mut handles = vec![];
    for i in 0..10 {
        let app_clone = Arc::clone(&app);
        let handle = tokio::spawn(async move {
            let offset = i * 20;
            let uri = format!("/api/v1/memories?offset={offset}&limit=20");
            make_request(&app_clone, Method::GET, &uri, None).await
        });
        handles.push(handle);
    }

    // All requests should succeed
    for handle in handles {
        let (status, json) = handle.await.unwrap();
        assert_eq!(status, StatusCode::OK);

        let returned = json["pagination"]["returned"].as_u64().unwrap() as usize;
        assert_eq!(returned, 20);
    }
}

// ============================================================================
// Integration Tests with Store State
// ============================================================================

#[tokio::test]
async fn test_pagination_count_consistency_with_store() {
    let app = create_test_router_with_memories(125).await;

    let (status, json) = make_request(&app, Method::GET, "/api/v1/memories", None).await;

    assert_eq!(status, StatusCode::OK);

    let tier_counts = &json["tier_counts"];
    let hot_count = tier_counts["hot"].as_u64().unwrap() as usize;

    // Verify tier counts match actual store state
    assert_eq!(
        hot_count, 125,
        "Tier counts should reflect actual store state"
    );

    // Iterate through all pages and verify total
    let mut total_retrieved = 0;
    let mut offset = 0;
    let limit = 50;

    loop {
        let uri = format!("/api/v1/memories?offset={offset}&limit={limit}");
        let (_status, page_json) = make_request(&app, Method::GET, &uri, None).await;

        let returned = page_json["pagination"]["returned"].as_u64().unwrap() as usize;
        if returned == 0 {
            break;
        }

        total_retrieved += returned;
        offset += returned;

        if returned < limit {
            break;
        }
    }

    assert_eq!(
        total_retrieved, hot_count,
        "Total retrieved across pages should match tier count"
    );
}
