//! Tests for Probabilistic Query API
//!
//! Tests the probabilistic query endpoints with confidence intervals,
//! evidence chains, and uncertainty tracking.

use axum::{
    Router,
    body::Body,
    http::{Method, Request, StatusCode},
};
use chrono::Utc;
use engram_cli::api::{ApiState, create_api_routes};
use engram_core::activation::SpreadingAutoTuner;
use serde_json::{Value, json};
use std::sync::Arc;
use tower::ServiceExt;

/// Create test router with API routes
fn create_test_router() -> Router {
    let store = Arc::new(engram_core::MemoryStore::new(100));
    let metrics = engram_core::metrics::init();
    let auto_tuner = SpreadingAutoTuner::new(0.10, 16);
    let (shutdown_tx, _shutdown_rx) = tokio::sync::watch::channel(false);
    let api_state = ApiState::new(store, metrics, auto_tuner, Arc::new(shutdown_tx));

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

/// Helper to store test episodes
async fn store_test_episodes(app: &Router, count: usize) {
    for i in 0..count {
        let when = Utc::now();
        let request_body = json!({
            "id": format!("episode_{:03}", i),
            "when": when.to_rfc3339(),
            "what": format!("Test episode {} about machine learning", i),
            "where_location": "Conference Room",
            "importance": 0.7 + (i as f64 * 0.01),
            "auto_link": false
        });

        let (status, _) = make_request(
            app,
            Method::POST,
            "/api/v1/episodes/remember",
            Some(request_body),
        )
        .await;

        assert_eq!(status, StatusCode::CREATED);
    }
}

#[tokio::test]
async fn test_probabilistic_query_endpoint() {
    let app = create_test_router();

    // Store test episodes
    store_test_episodes(&app, 10).await;

    // Execute probabilistic query with evidence and uncertainty
    let (status, response) = make_request(
        &app,
        Method::GET,
        "/api/v1/query/probabilistic?query=machine%20learning&limit=5&include_evidence=true&include_uncertainty=true",
        None,
    )
    .await;

    assert_eq!(status, StatusCode::OK);

    // Validate response structure
    assert!(
        response["memories"].is_array(),
        "memories should be an array"
    );
    assert!(
        response["confidence_interval"].is_object(),
        "confidence_interval should be an object"
    );
    assert!(
        response["evidence_chain"].is_array(),
        "evidence_chain should be an array"
    );
    assert!(
        response["uncertainty_sources"].is_array(),
        "uncertainty_sources should be an array"
    );
    assert!(
        response["system_message"].is_string(),
        "system_message should be a string"
    );

    // Validate confidence interval structure
    let confidence_interval = &response["confidence_interval"];
    assert!(
        confidence_interval["point"].is_f64(),
        "point should be a number"
    );
    assert!(
        confidence_interval["lower"].is_f64(),
        "lower should be a number"
    );
    assert!(
        confidence_interval["upper"].is_f64(),
        "upper should be a number"
    );
    assert!(
        confidence_interval["width"].is_f64(),
        "width should be a number"
    );

    // Validate confidence interval bounds
    let point = confidence_interval["point"].as_f64().unwrap();
    let lower = confidence_interval["lower"].as_f64().unwrap();
    let upper = confidence_interval["upper"].as_f64().unwrap();

    assert!(
        (0.0..=1.0).contains(&point),
        "point confidence should be between 0 and 1"
    );
    assert!(
        (0.0..=1.0).contains(&lower),
        "lower confidence should be between 0 and 1"
    );
    assert!(
        (0.0..=1.0).contains(&upper),
        "upper confidence should be between 0 and 1"
    );
    assert!(lower <= point, "lower bound should be <= point");
    assert!(point <= upper, "point should be <= upper bound");

    // Validate memories structure
    let memories = response["memories"].as_array().unwrap();
    assert!(!memories.is_empty(), "should return at least one memory");

    for memory in memories {
        assert!(memory["content"].is_string(), "memory should have content");
        assert!(
            memory["confidence"].is_object(),
            "memory should have confidence object"
        );
        assert!(
            memory["confidence"]["value"].is_f64(),
            "memory confidence should have value"
        );

        let mem_confidence = memory["confidence"]["value"].as_f64().unwrap();
        assert!(
            (0.0..=1.0).contains(&mem_confidence),
            "memory confidence should be between 0 and 1"
        );
    }
}

#[tokio::test]
async fn test_probabilistic_query_with_embedding() {
    let app = create_test_router();

    // Store test episodes
    store_test_episodes(&app, 5).await;

    // Create embedding vector
    let embedding_vec = vec![0.5; 768];
    let embedding_json = serde_json::to_string(&embedding_vec).unwrap();
    let query = format!(
        "/api/v1/query/probabilistic?embedding={}&limit=3",
        urlencoding::encode(&embedding_json)
    );

    // Execute probabilistic query with embedding
    let (status, response) = make_request(&app, Method::GET, &query, None).await;

    assert_eq!(status, StatusCode::OK);
    assert!(response["memories"].is_array());
    assert!(response["confidence_interval"].is_object());
}

#[tokio::test]
async fn test_probabilistic_query_empty_results() {
    let app = create_test_router();

    // Query without storing any episodes
    let (status, response) = make_request(
        &app,
        Method::GET,
        "/api/v1/query/probabilistic?query=nonexistent&limit=5",
        None,
    )
    .await;

    assert_eq!(status, StatusCode::OK);

    // Should still return valid structure with empty results
    assert!(response["memories"].is_array());
    assert!(response["confidence_interval"].is_object());

    let memories = response["memories"].as_array().unwrap();
    assert!(memories.is_empty(), "should return no memories");

    // Confidence interval should still be valid
    let confidence_interval = &response["confidence_interval"];
    let point = confidence_interval["point"].as_f64().unwrap();
    assert!((0.0..=1.0).contains(&point));
}

#[tokio::test]
async fn test_probabilistic_query_validation() {
    let app = create_test_router();

    // Test without query or embedding
    let (status, response) =
        make_request(&app, Method::GET, "/api/v1/query/probabilistic", None).await;

    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(response["error"].is_object());
}

#[tokio::test]
async fn test_probabilistic_query_evidence_chain() {
    let app = create_test_router();

    // Store test episodes with high importance
    for i in 0..3 {
        let when = Utc::now();
        let request_body = json!({
            "id": format!("high_importance_{}", i),
            "when": when.to_rfc3339(),
            "what": format!("Critical machine learning milestone {}", i),
            "importance": 0.95,
            "auto_link": false
        });

        let (status, _) = make_request(
            &app,
            Method::POST,
            "/api/v1/episodes/remember",
            Some(request_body),
        )
        .await;

        assert_eq!(status, StatusCode::CREATED);
    }

    // Execute probabilistic query with evidence enabled
    let (status, response) = make_request(
        &app,
        Method::GET,
        "/api/v1/query/probabilistic?query=machine%20learning&limit=3&include_evidence=true",
        None,
    )
    .await;

    assert_eq!(status, StatusCode::OK);

    // Validate evidence chain
    let evidence_chain = response["evidence_chain"].as_array().unwrap();

    for evidence in evidence_chain {
        assert!(
            evidence["source_type"].is_string(),
            "evidence should have source_type"
        );
        assert!(
            evidence["strength"].is_f64(),
            "evidence should have strength"
        );
        assert!(
            evidence["description"].is_string(),
            "evidence should have description"
        );

        let strength = evidence["strength"].as_f64().unwrap();
        assert!(
            (0.0..=1.0).contains(&strength),
            "evidence strength should be between 0 and 1"
        );
    }
}

#[tokio::test]
async fn test_probabilistic_query_limit_parameter() {
    let app = create_test_router();

    // Store 10 test episodes
    store_test_episodes(&app, 10).await;

    // Test with limit=3
    let (status, response) = make_request(
        &app,
        Method::GET,
        "/api/v1/query/probabilistic?query=machine%20learning&max_results=3",
        None,
    )
    .await;

    assert_eq!(status, StatusCode::OK);

    let memories = response["memories"].as_array().unwrap();
    assert!(memories.len() <= 3, "should return at most 3 memories");

    // Test with limit=5
    let (status, response) = make_request(
        &app,
        Method::GET,
        "/api/v1/query/probabilistic?query=machine%20learning&max_results=5",
        None,
    )
    .await;

    assert_eq!(status, StatusCode::OK);

    let memories = response["memories"].as_array().unwrap();
    assert!(memories.len() <= 5, "should return at most 5 memories");
}

#[tokio::test]
async fn test_probabilistic_query_confidence_ordering() {
    let app = create_test_router();

    // Store test episodes with varying importance
    for i in 0..5 {
        let when = Utc::now();
        let request_body = json!({
            "id": format!("varying_importance_{}", i),
            "when": when.to_rfc3339(),
            "what": format!("Machine learning topic {}", i),
            "importance": 0.5 + (f64::from(i) * 0.1),
            "auto_link": false
        });

        let (status, _) = make_request(
            &app,
            Method::POST,
            "/api/v1/episodes/remember",
            Some(request_body),
        )
        .await;

        assert_eq!(status, StatusCode::CREATED);
    }

    // Execute probabilistic query
    let (status, response) = make_request(
        &app,
        Method::GET,
        "/api/v1/query/probabilistic?query=machine%20learning&max_results=5",
        None,
    )
    .await;

    assert_eq!(status, StatusCode::OK);

    // Validate memories are returned in descending confidence order
    let memories = response["memories"].as_array().unwrap();

    let mut prev_confidence = 1.0;
    for memory in memories {
        let confidence = memory["confidence"]["value"].as_f64().unwrap();
        assert!(
            confidence <= prev_confidence,
            "memories should be ordered by descending confidence"
        );
        prev_confidence = confidence;
    }
}

#[tokio::test]
async fn test_probabilistic_query_uncertainty_sources() {
    let app = create_test_router();

    // Store test episodes
    store_test_episodes(&app, 5).await;

    // Execute probabilistic query with uncertainty enabled
    let (status, response) = make_request(
        &app,
        Method::GET,
        "/api/v1/query/probabilistic?query=machine%20learning&max_results=5&include_uncertainty=true",
        None,
    )
    .await;

    assert_eq!(status, StatusCode::OK);

    // Validate uncertainty sources structure
    let uncertainty_sources = response["uncertainty_sources"].as_array().unwrap();

    for source in uncertainty_sources {
        assert!(
            source["source_type"].is_string(),
            "uncertainty source should have source_type"
        );
        assert!(
            source["explanation"].is_string(),
            "uncertainty source should have explanation"
        );
        assert!(
            source["impact"].is_f64(),
            "uncertainty source should have impact"
        );

        let impact = source["impact"].as_f64().unwrap();
        assert!(impact >= 0.0, "uncertainty impact should be non-negative");
    }
}
