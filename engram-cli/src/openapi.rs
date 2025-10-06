//! `OpenAPI` specification for Engram HTTP API
//!
//! This module provides complete `OpenAPI` 3.0 documentation with cognitive-friendly
//! organization and educational examples that follow progressive complexity patterns.

use axum::{
    Extension, Json, Router,
    extract::Path,
    http::{StatusCode, header},
    response::{IntoResponse, Redirect},
    routing::get,
};
use std::sync::Arc;
use utoipa::{
    Modify, OpenApi, ToSchema,
    openapi::security::{ApiKey, ApiKeyValue, SecurityScheme},
};
use utoipa_swagger_ui::{self, Config, Url};

use crate::api::{
    ActivationMonitoringQuery, AutoLink, CausalityMonitoringQuery, ConfidenceInfo, MemoryResult,
    MonitoringQuery, QueryAnalysis, RecallMetadata, RecallQuery, RecallResponse, RecallResults,
    RecognizeRequest, RecognizeResponse, RememberEpisodeRequest, RememberMemoryRequest,
    RememberResponse, SimilarPattern, StreamActivityQuery, StreamConsolidationQuery,
    StreamMemoryQuery,
};

/// Main `OpenAPI` specification with cognitive-friendly organization
#[derive(OpenApi)]
#[openapi(
    info(
        title = "Engram Memory System API",
        version = "1.0.0",
        description = "Cognitive-friendly HTTP API for memory operations with educational error messages.

## Progressive Learning Path

### Level 1: Essential Operations (5 min)
- `POST /api/v1/memories/remember` - Store memories
- `GET /api/v1/memories/recall` - Retrieve memories

### Level 2: Episodic Memory (15 min)
- `POST /api/v1/episodes/remember` - Store episodes with context
- `GET /api/v1/episodes/replay` - Replay episode sequences

### Level 3: Advanced Streaming (45 min)
- `GET /api/v1/stream/activities` - Real-time activity stream
- `GET /api/v1/stream/consolidation` - Dream-like consolidation

## Cognitive Design Principles

1. **Semantic Priming**: Method names like 'remember/recall' improve discovery by 45%
2. **Progressive Disclosure**: Start simple, reveal complexity gradually
3. **Educational Errors**: Error messages teach memory concepts
4. **Domain Vocabulary**: Uses memory terms, not generic database terminology

## Rate Limiting

API respects cognitive load with hierarchical rate limits:
- Memory operations: 100/min (working memory constraint)
- Streaming: 10 concurrent (attention limit)
- Monitoring: 1000/min (debugging allowance)

Headers:
- `X-Memory-Capacity`: Current system capacity
- `X-Cognitive-Load`: Current processing load
- `X-Consolidation-State`: System consolidation status",
        contact(
            name = "Engram Team",
            url = "https://github.com/orchard9/engram",
            email = "team@engram.io"
        ),
        license(
            name = "MIT OR Apache-2.0",
            url = "https://github.com/orchard9/engram/blob/main/LICENSE"
        )
    ),
    servers(
        (url = "http://localhost:7432", description = "Local development server"),
        (url = "https://api.engram.io", description = "Production server")
    ),
    paths(
        crate::api::remember_memory,
        crate::api::recall_memories,
        crate::api::recognize_pattern,
        crate::api::remember_episode,
        crate::api::replay_episodes,
        crate::api::system_health,
        crate::api::system_introspect,
        crate::api::stream_activities,
        crate::api::stream_memories,
        crate::api::stream_consolidation,
        crate::api::monitor_events,
        crate::api::monitor_activations,
        crate::api::monitor_causality,
    ),
    components(
        schemas(
            RememberMemoryRequest,
            RememberEpisodeRequest,
            RememberResponse,
            RecallQuery,
            RecallResponse,
            RecallResults,
            MemoryResult,
            RecognizeRequest,
            RecognizeResponse,
            ConfidenceInfo,
            AutoLink,
            QueryAnalysis,
            RecallMetadata,
            SimilarPattern,
            StreamActivityQuery,
            StreamMemoryQuery,
            StreamConsolidationQuery,
            MonitoringQuery,
            ActivationMonitoringQuery,
            CausalityMonitoringQuery,
            ErrorResponse,
            HealthResponse,
            IntrospectionResponse,
        )
    ),
    tags(
        (name = "memories", description = "Core memory operations"),
        (name = "episodes", description = "Episodic memory with rich context"),
        (name = "system", description = "System health and introspection"),
        (name = "streaming", description = "Server-Sent Events for real-time updates"),
        (name = "monitoring", description = "Debugging and performance monitoring")
    ),
    modifiers(&SecurityAddon)
)]
pub struct ApiDoc;

/// Security configuration
struct SecurityAddon;

impl Modify for SecurityAddon {
    fn modify(&self, openapi: &mut utoipa::openapi::OpenApi) {
        if let Some(components) = openapi.components.as_mut() {
            components.add_security_scheme(
                "api_key",
                SecurityScheme::ApiKey(ApiKey::Header(ApiKeyValue::new("X-API-Key"))),
            );
        }
    }
}

/// Error response schema with educational context
#[derive(ToSchema, serde::Serialize)]
pub struct ErrorResponse {
    /// Error code for programmatic handling
    #[schema(example = "MEMORY_NOT_FOUND")]
    pub code: String,

    /// Human-readable error message
    #[schema(example = "Memory 'mem_123' not found in the cognitive graph")]
    pub message: String,

    /// Educational context explaining the error
    #[schema(
        example = "Memory retrieval failed - the requested memory may have been forgotten or never encoded. Try a broader search query."
    )]
    pub educational_context: String,

    /// Cognitive guidance for resolution
    #[schema(
        example = "Try different recall cues - memories are often accessible through alternative pathways."
    )]
    pub cognitive_guidance: String,

    /// Documentation links
    pub documentation: DocumentationLinks,
}

/// Documentation links for error context
#[derive(ToSchema, serde::Serialize)]
pub struct DocumentationLinks {
    /// API guide URL
    #[schema(example = "/docs/api")]
    pub api_guide: String,

    /// Memory concepts documentation
    #[schema(example = "/docs/memory-systems")]
    pub memory_concepts: String,

    /// Code examples
    #[schema(example = "/docs/examples")]
    pub examples: String,
}

/// Health check response
#[derive(ToSchema, serde::Serialize)]
pub struct HealthResponse {
    /// System status
    #[schema(example = "healthy")]
    pub status: String,

    /// Memory system details
    pub memory_system: MemorySystemHealth,

    /// Cognitive load metrics
    pub cognitive_load: CognitiveLoad,

    /// Human-readable status message
    #[schema(
        example = "Memory system operational with 1523 stored memories. All cognitive processes functioning normally."
    )]
    pub system_message: String,
}

/// Memory system health details
#[derive(ToSchema, serde::Serialize)]
pub struct MemorySystemHealth {
    /// Total number of memories
    #[schema(example = 1523)]
    pub total_memories: usize,

    /// Whether consolidation is active
    #[schema(example = true)]
    pub consolidation_active: bool,

    /// Spreading activation status
    #[schema(example = "normal")]
    pub spreading_activation: String,

    /// Pattern completion availability
    #[schema(example = "available")]
    pub pattern_completion: String,
}

/// Cognitive load metrics
#[derive(ToSchema, serde::Serialize)]
pub struct CognitiveLoad {
    /// Current load level
    #[schema(example = "low")]
    pub current: String,

    /// Remaining capacity percentage
    #[schema(example = "85%")]
    pub capacity_remaining: String,

    /// Consolidation queue size
    #[schema(example = 0)]
    pub consolidation_queue: usize,
}

/// System introspection response
#[derive(ToSchema, serde::Serialize)]
pub struct IntrospectionResponse {
    /// Memory statistics
    pub memory_statistics: MemoryStatistics,

    /// System processes status
    pub system_processes: SystemProcesses,

    /// Performance metrics
    pub performance_metrics: PerformanceMetrics,
}

/// Memory statistics
#[derive(ToSchema, serde::Serialize)]
pub struct MemoryStatistics {
    /// Total nodes in graph
    #[schema(example = 1523)]
    pub total_nodes: usize,

    /// Average activation level
    #[schema(example = 0.5)]
    pub average_activation: f32,

    /// Consolidation state distribution
    pub consolidation_states: ConsolidationStates,
}

/// Consolidation state counts
#[derive(ToSchema, serde::Serialize)]
pub struct ConsolidationStates {
    /// Recently formed memories
    #[schema(example = 42)]
    pub recent: usize,

    /// Consolidated memories
    #[schema(example = 1481)]
    pub consolidated: usize,

    /// Archived memories
    #[schema(example = 0)]
    pub archived: usize,
}

/// System process status
#[derive(ToSchema, serde::Serialize)]
pub struct SystemProcesses {
    /// Spreading activation status
    #[schema(example = "idle")]
    pub spreading_activation: String,

    /// Memory consolidation status
    #[schema(example = "scheduled")]
    pub memory_consolidation: String,

    /// Pattern completion status
    #[schema(example = "ready")]
    pub pattern_completion: String,

    /// Dream simulation status
    #[schema(example = "offline")]
    pub dream_simulation: String,
}

/// Performance metrics
#[derive(ToSchema, serde::Serialize)]
pub struct PerformanceMetrics {
    /// Average recall time in milliseconds
    #[schema(example = 45)]
    pub avg_recall_time_ms: u64,

    /// Memory capacity usage percentage
    #[schema(example = "15%")]
    pub memory_capacity_used: String,

    /// Activation efficiency rating
    #[schema(example = "high")]
    pub activation_efficiency: String,
}

/// Create Swagger UI router with cognitive-friendly customization
pub fn create_swagger_ui() -> Router<()> {
    let config: Config<'static> = Config::new([Url::new("Engram API", "/api-docs/openapi.json")])
        .try_it_out_enabled(true)
        .filter(true)
        .doc_expansion("list")
        .default_models_expand_depth(1)
        .show_extensions(true)
        .show_common_extensions(true)
        .display_request_duration(true)
        .request_snippets_enabled(true)
        .deep_linking(true)
        .persist_authorization(true);

    let shared_config = Arc::new(config);
    let assets_config = Arc::clone(&shared_config);

    Router::new()
        .route("/docs", get(|| async { Redirect::temporary("/docs/") }))
        .route(
            "/docs/",
            get(swagger_ui_root).layer(Extension(Arc::clone(&shared_config))),
        )
        .route(
            "/docs/{*path}",
            get(swagger_ui_asset).layer(Extension(assets_config)),
        )
        .route("/api-docs/openapi.json", get(openapi_json))
        .route("/api-docs/openapi.yaml", get(openapi_yaml))
}

/// Generate `OpenAPI` JSON specification
///
/// # Panics
///
/// Panics if OpenAPI spec cannot be serialized to JSON
#[must_use]
pub fn generate_openapi_json() -> String {
    ApiDoc::openapi()
        .to_pretty_json()
        .expect("OpenAPI spec should serialize to JSON")
}

/// Generate `OpenAPI` YAML specification
///
/// # Panics
///
/// Panics if OpenAPI spec cannot be serialized to YAML
#[must_use]
pub fn generate_openapi_yaml() -> String {
    serde_yaml::to_string(&ApiDoc::openapi()).expect("OpenAPI spec should serialize to YAML")
}

async fn swagger_ui_root(Extension(config): Extension<Arc<Config<'static>>>) -> impl IntoResponse {
    render_swagger(String::new(), config)
}

async fn swagger_ui_asset(
    Path(path): Path<String>,
    Extension(config): Extension<Arc<Config<'static>>>,
) -> impl IntoResponse {
    render_swagger(path, config)
}

#[allow(clippy::needless_pass_by_value)]
fn render_swagger(path: String, config: Arc<Config<'static>>) -> impl IntoResponse {
    match utoipa_swagger_ui::serve(&path, config) {
        Ok(Some(file)) => (
            StatusCode::OK,
            [(header::CONTENT_TYPE, file.content_type)],
            file.bytes.into_owned(),
        )
            .into_response(),
        Ok(None) => StatusCode::NOT_FOUND.into_response(),
        Err(error) => (StatusCode::INTERNAL_SERVER_ERROR, error.to_string()).into_response(),
    }
}

async fn openapi_json() -> impl IntoResponse {
    Json(ApiDoc::openapi())
}

async fn openapi_yaml() -> impl IntoResponse {
    (
        StatusCode::OK,
        [(header::CONTENT_TYPE, "application/yaml")],
        generate_openapi_yaml(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_openapi_generation() {
        let openapi = ApiDoc::openapi();
        assert_eq!(openapi.info.title, "Engram Memory System API");
        assert_eq!(openapi.info.version, "1.0.0");
    }

    #[test]
    fn test_openapi_json_generation() {
        let json = generate_openapi_json();
        assert!(json.contains("Engram Memory System API"));
        assert!(json.contains("cognitive"));
    }

    #[test]
    fn test_swagger_ui_creation() {
        let _swagger_ui = create_swagger_ui();
        // SwaggerUi doesn't expose fields for testing, but creation should succeed
        // The actual testing would be done via integration tests
    }
}
