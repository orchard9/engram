//! Query execution HTTP handler exposing the Engram query language.
//!
//! Implements POST /api/v1/query endpoint for executing query strings against
//! the probabilistic graph memory with multi-tenant routing.

use axum::{
    extract::State,
    http::HeaderMap,
    response::{IntoResponse, Json},
};
use serde::{Deserialize, Serialize};
use std::time::Instant;
use utoipa::{IntoParams, ToSchema};

use crate::api::{ApiError, ApiState};
use engram_core::{
    MemorySpaceId,
    query::{
        ProbabilisticQueryResult,
        executor::{AstQueryExecutorConfig, QueryExecutor, context::QueryContext},
        parser::Parser,
    },
};

// ================================================================================================
// Request/Response Types with utoipa Documentation
// ================================================================================================

/// Request for executing a query string against the memory graph
#[derive(Debug, Deserialize, ToSchema, IntoParams)]
pub struct QueryRequest {
    /// Memory space identifier for multi-tenant isolation
    /// If not provided, uses X-Memory-Space header or default space
    pub memory_space_id: Option<String>,

    /// Query text in Engram query language format
    ///
    /// # Examples
    /// - `RECALL episode_123` - Retrieve specific episode
    /// - `SPREAD FROM node_456 HOPS 3` - Spreading activation from node
    /// - `RECALL ContentMatch("breakfast") LIMIT 5` - Semantic search
    pub query_text: String,
}

/// Successful query execution response with probabilistic results
#[derive(Debug, Serialize, ToSchema)]
pub struct QueryResponse {
    /// Retrieved episodes matching the query
    pub episodes: Vec<EpisodeResult>,

    /// Aggregate confidence interval across all results
    pub aggregate_confidence: ConfidenceIntervalResponse,

    /// Total number of results before pagination/limits
    pub total_count: usize,

    /// Query execution time in milliseconds
    pub execution_time_ms: u64,
}

/// Episode result with confidence score
#[derive(Debug, Serialize, ToSchema)]
pub struct EpisodeResult {
    /// Episode identifier
    pub id: String,

    /// When the episode occurred
    pub when: String,

    /// What happened (semantic content)
    pub what: String,

    /// Where it occurred
    #[serde(skip_serializing_if = "Option::is_none")]
    pub where_location: Option<String>,

    /// Who was involved
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub who: Vec<String>,

    /// Confidence score for this episode [0.0, 1.0]
    pub confidence: f32,

    /// Human-readable confidence category
    pub confidence_category: String,
}

/// Confidence interval representation
#[derive(Debug, Serialize, ToSchema)]
pub struct ConfidenceIntervalResponse {
    /// Lower bound of confidence interval [0.0, 1.0]
    pub lower_bound: f32,

    /// Point estimate (mean) confidence [0.0, 1.0]
    pub mean: f32,

    /// Upper bound of confidence interval [0.0, 1.0]
    pub upper_bound: f32,

    /// Confidence interval width (uncertainty measure)
    pub width: f32,
}

// ================================================================================================
// HTTP Handler
// ================================================================================================

/// Execute a query string and return probabilistic results
///
/// # Multi-tenant Routing
///
/// Memory space is determined by (in order of precedence):
/// 1. `memory_space_id` field in request body
/// 2. `X-Memory-Space` header
/// 3. Default space configured at server startup
///
/// # Query Language
///
/// Supports the following query types:
/// - **RECALL**: Retrieve memories matching pattern
/// - **SPREAD**: Activation spreading from source node
/// - **PREDICT**: Predict future states (requires System 2 reasoning)
/// - **IMAGINE**: Pattern completion with creativity
/// - **CONSOLIDATE**: Trigger memory consolidation
///
/// # Errors
///
/// - **400 Bad Request**: Invalid query syntax or parameters
/// - **404 Not Found**: Memory space does not exist
/// - **408 Request Timeout**: Query exceeded time limit
/// - **500 Internal Server Error**: Query execution failed
///
/// # Examples
///
/// ```json
/// {
///   "memory_space_id": "user_123",
///   "query_text": "RECALL episode_456"
/// }
/// ```
#[utoipa::path(
    post,
    path = "/api/v1/query",
    request_body = QueryRequest,
    responses(
        (status = 200, description = "Query executed successfully", body = QueryResponse),
        (status = 400, description = "Invalid query syntax"),
        (status = 404, description = "Memory space not found"),
        (status = 408, description = "Query timeout"),
        (status = 500, description = "Internal server error")
    ),
    tag = "query"
)]
pub async fn query_handler(
    State(state): State<ApiState>,
    headers: HeaderMap,
    Json(request): Json<QueryRequest>,
) -> Result<impl IntoResponse, ApiError> {
    let start = Instant::now();

    // Extract memory space ID from request, header, or default
    let space_id = extract_memory_space_id(
        request.memory_space_id.as_deref(),
        &headers,
        &state.default_space,
    )?;

    // Parse query string
    let query = Parser::parse(&request.query_text).map_err(|e| {
        ApiError::bad_request(
            format!("Invalid query syntax: {}", e),
            "Check query syntax matches Engram query language",
            "Example: RECALL episode_123",
        )
    })?;

    // Create query executor
    let executor = QueryExecutor::new(state.registry.clone(), AstQueryExecutorConfig::default());

    // Create query context
    let context = QueryContext::without_timeout(space_id.clone());

    // Execute query
    let result = executor.execute(query, context).await.map_err(|e| {
        match e {
            engram_core::query::executor::query_executor::QueryExecutionError::MemorySpaceNotFound { .. } => {
                ApiError::not_found(
                    format!("Memory space '{}' not found", space_id),
                    "Verify the memory space ID exists",
                    "Use GET /api/v1/spaces to list available spaces",
                )
            }
            engram_core::query::executor::query_executor::QueryExecutionError::Timeout { .. } => {
                ApiError::timeout(
                    "Query execution timed out",
                    "Simplify query or increase timeout",
                    "Try adding LIMIT clause to reduce result set",
                )
            }
            engram_core::query::executor::query_executor::QueryExecutionError::QueryTooComplex { cost, limit } => {
                ApiError::bad_request(
                    format!("Query too complex: cost={}, limit={}", cost, limit),
                    "Simplify query to reduce computational cost",
                    "Reduce HOPS value or add more constraints",
                )
            }
            engram_core::query::executor::query_executor::QueryExecutionError::NotImplemented { query_type, reason } => {
                ApiError::not_implemented(
                    format!("{} query not implemented: {}", query_type, reason),
                    "Use a different query type",
                    "Try RECALL for basic retrieval",
                )
            }
            _ => {
                ApiError::internal_error(
                    format!("Query execution failed: {}", e),
                    "Contact support if this persists",
                    "",
                )
            }
        }
    })?;

    let execution_time = start.elapsed();

    // Convert to response format
    let response = convert_to_response(result, execution_time.as_millis() as u64);

    Ok(Json(response))
}

// ================================================================================================
// Helper Functions
// ================================================================================================

/// Extract memory space ID from request body, header, or default
fn extract_memory_space_id(
    request_space: Option<&str>,
    headers: &HeaderMap,
    default_space: &MemorySpaceId,
) -> Result<MemorySpaceId, ApiError> {
    // Priority: request body > header > default
    if let Some(space_id) = request_space {
        return MemorySpaceId::new(space_id).map_err(|e| {
            ApiError::bad_request(
                format!("Invalid memory_space_id: {}", e),
                "Use a valid memory space identifier",
                "Example: user_123",
            )
        });
    }

    if let Some(header_value) = headers.get("X-Memory-Space") {
        let space_str = header_value.to_str().map_err(|_| {
            ApiError::bad_request(
                "Invalid X-Memory-Space header encoding",
                "Use UTF-8 string for memory space ID",
                "Example: X-Memory-Space: user_123",
            )
        })?;

        return MemorySpaceId::new(space_str).map_err(|e| {
            ApiError::bad_request(
                format!("Invalid X-Memory-Space header: {}", e),
                "Use a valid memory space identifier",
                "Example: X-Memory-Space: user_123",
            )
        });
    }

    // Use default space
    Ok(default_space.clone())
}

/// Convert ProbabilisticQueryResult to API response format
fn convert_to_response(result: ProbabilisticQueryResult, execution_time_ms: u64) -> QueryResponse {
    let total_count = result.episodes.len();

    let episodes = result
        .episodes
        .into_iter()
        .map(|(episode, confidence)| EpisodeResult {
            id: episode.id,
            when: episode.when.to_rfc3339(),
            what: episode.what,
            where_location: episode.where_location,
            who: episode.who.unwrap_or_default(),
            confidence: confidence.raw(),
            confidence_category: confidence_value_to_category(confidence.raw()),
        })
        .collect();

    let aggregate_confidence = ConfidenceIntervalResponse {
        lower_bound: result.confidence_interval.lower.raw(),
        mean: result.confidence_interval.point.raw(),
        upper_bound: result.confidence_interval.upper.raw(),
        width: result.confidence_interval.width,
    };

    QueryResponse {
        episodes,
        aggregate_confidence,
        total_count,
        execution_time_ms,
    }
}

/// Convert confidence f32 value to category string
fn confidence_value_to_category(value: f32) -> String {
    match value {
        v if v <= 0.0 => "None".to_string(),
        v if v < 0.3 => "Low".to_string(),
        v if v < 0.7 => "Medium".to_string(),
        v if v < 1.0 => "High".to_string(),
        _ => "Certain".to_string(),
    }
}

// ================================================================================================
// Tests
// ================================================================================================

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    #![allow(clippy::unnecessary_to_owned)]

    use super::*;

    #[test]
    fn test_extract_memory_space_from_request_body() {
        let headers = HeaderMap::new();
        let default_space = MemorySpaceId::new("default".to_string()).unwrap();

        let result = extract_memory_space_id(Some("user_123"), &headers, &default_space);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().as_str(), "user_123");
    }

    #[test]
    fn test_extract_memory_space_from_header() {
        let mut headers = HeaderMap::new();
        headers.insert("X-Memory-Space", "user_456".parse().unwrap());
        let default_space = MemorySpaceId::new("default".to_string()).unwrap();

        let result = extract_memory_space_id(None, &headers, &default_space);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().as_str(), "user_456");
    }

    #[test]
    fn test_extract_memory_space_uses_default() {
        let headers = HeaderMap::new();
        let default_space = MemorySpaceId::new("default".to_string()).unwrap();

        let result = extract_memory_space_id(None, &headers, &default_space);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().as_str(), "default");
    }

    #[test]
    fn test_request_body_takes_precedence() {
        let mut headers = HeaderMap::new();
        headers.insert("X-Memory-Space", "header_space".parse().unwrap());
        let default_space = MemorySpaceId::new("default".to_string()).unwrap();

        let result = extract_memory_space_id(Some("body_space"), &headers, &default_space);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().as_str(), "body_space");
    }
}
