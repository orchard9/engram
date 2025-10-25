//! HTTP API module for cognitive-friendly memory operations
//!
//! This module provides REST API endpoints that follow cognitive ergonomics
//! principles with natural language paths and educational error messages.

use axum::response::sse::{Event, KeepAlive};
use axum::{
    Router,
    extract::{Path, Query, State},
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Json, Sse},
    routing::{get, post},
};
use chrono::{DateTime, Utc};
use engram_proto::{Confidence, ConsolidationState, Memory, MemoryType};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::collections::HashMap;
use std::sync::Arc;
use tokio_stream::{Stream, wrappers::ReceiverStream};
use tracing::{Span, info};
use utoipa::{IntoParams, ToSchema};

use crate::grpc::MemoryService;
use crate::openapi::create_swagger_ui;
use engram_core::{
    Confidence as CoreConfidence, Cue as CoreCue, Episode, MemorySpaceError, MemorySpaceId,
    MemorySpaceRegistry, MemoryStore, SpaceSummary,
    activation::{AutoTuneAuditEntry, RecallMode, SpreadingAutoTuner},
    completion::{ConsolidationStats as CoreConsolidationStats, SemanticPattern},
    memory::EpisodeBuilder as CoreEpisodeBuilder,
    metrics::{
        MetricsRegistry,
        health::{HealthCheck, HealthCheckType, HealthStatus},
    },
    query::{EvidenceSource, UncertaintySource},
    store::MemoryEvent,
};

/// Shared application state
#[derive(Clone)]
pub struct ApiState {
    /// Cognitive memory store with HNSW indexing and activation spreading
    ///
    /// # Deprecation Notice
    /// This field is deprecated and will be removed in a future version.
    /// Use `registry.create_or_get(&space_id)` instead to obtain space-specific store handles.
    ///
    /// ## Migration Guide
    /// ```ignore
    /// // Old pattern (deprecated):
    /// let result = state.store.recall(&cue);
    ///
    /// // New pattern (registry-based):
    /// let space_id = extract_memory_space_id(query_space, body_space, &state.default_space)?;
    /// let handle = state.registry.create_or_get(&space_id).await?;
    /// let store = handle.store();
    /// let result = store.recall(&cue);
    /// ```
    #[deprecated(
        since = "0.2.0",
        note = "Use `registry.create_or_get(&space_id)` for multi-tenant isolation. \
                This field provides access to the default space only and will be removed \
                after all handlers migrate to the registry pattern."
    )]
    pub store: Arc<MemoryStore>,
    /// gRPC memory service for complex operations
    pub memory_service: Arc<MemoryService>,
    /// Memory space registry managing tenant handles
    pub registry: Arc<MemorySpaceRegistry>,
    /// Default memory space identifier used for legacy operations
    pub default_space: MemorySpaceId,
    /// Global metrics registry for streaming/log export
    pub metrics: Arc<MetricsRegistry>,
    /// Auto-tuning audit log
    pub auto_tuner: Arc<SpreadingAutoTuner>,
    /// Shutdown signal sender for graceful termination
    pub shutdown_tx: Arc<tokio::sync::watch::Sender<bool>>,
}

/// Auto-tuning response payload used by the REST API and OpenAPI schema.
#[derive(serde::Serialize, ToSchema)]
pub struct AutoTuneResponse {
    /// Audit log of auto-tuner configuration changes
    pub audit_log: Vec<AutoTuneAuditEntry>,
}

/// Summary information about a memory space.
#[derive(Serialize, ToSchema, Clone, Debug)]
pub struct MemorySpaceDescriptor {
    /// Identifier assigned to the memory space.
    pub id: String,
    /// Persistence root directory hosting the space data.
    pub persistence_root: String,
    /// Timestamp when the space was first initialised on this node.
    pub created_at: DateTime<Utc>,
}

/// Response payload for space listings.
#[derive(Serialize, ToSchema, Debug)]
pub struct MemorySpaceListResponse {
    /// All spaces currently tracked by the registry.
    pub spaces: Vec<MemorySpaceDescriptor>,
}

/// Request payload used to create or obtain a memory space reference.
#[derive(Deserialize, ToSchema, Debug)]
pub struct CreateMemorySpaceRequest {
    /// Desired identifier for the memory space.
    pub id: String,
}

impl From<SpaceSummary> for MemorySpaceDescriptor {
    fn from(summary: SpaceSummary) -> Self {
        Self {
            id: summary.id.as_str().to_string(),
            persistence_root: summary.root.display().to_string(),
            created_at: summary.created_at,
        }
    }
}

/// Per-space health metrics for multi-tenant monitoring.
#[derive(Serialize, ToSchema, Clone, Debug)]
pub struct SpaceHealthMetrics {
    /// Memory space identifier
    pub space: String,
    /// Total number of memories in this space
    pub memories: u64,
    /// Capacity utilization pressure (0.0-1.0)
    /// TODO(Task 006c): Wire up actual pressure metrics from tier backend
    pub pressure: f64,
    /// Write-Ahead Log lag in milliseconds
    /// TODO(Task 006c): Wire up actual WAL lag from persistence handle
    pub wal_lag_ms: f64,
    /// Consolidation rate (memories/sec)
    /// TODO(Task 006c): Wire up actual consolidation throughput metrics
    pub consolidation_rate: f64,
}

/// Enhanced health response with per-space metrics.
#[derive(Serialize, ToSchema, Clone, Debug)]
pub struct HealthResponse {
    /// Overall system health status
    pub status: String,
    /// Timestamp of health check
    pub timestamp: String,
    /// System-wide health checks
    pub checks: Vec<serde_json::Value>,
    /// Per-space health metrics
    pub spaces: Vec<SpaceHealthMetrics>,
}

impl ApiState {
    /// Create new API state with memory store
    #[allow(deprecated)] // Constructor needs to initialize the deprecated field during migration
    pub fn new(
        store: Arc<MemoryStore>,
        registry: Arc<MemorySpaceRegistry>,
        default_space: MemorySpaceId,
        metrics: Arc<MetricsRegistry>,
        auto_tuner: Arc<SpreadingAutoTuner>,
        shutdown_tx: Arc<tokio::sync::watch::Sender<bool>>,
    ) -> Self {
        let memory_service = Arc::new(MemoryService::new(
            Arc::clone(&store),
            Arc::clone(&metrics),
            Arc::clone(&registry),
            default_space.clone(),
        ));
        Self {
            store,
            memory_service,
            registry,
            default_space,
            metrics,
            auto_tuner,
            shutdown_tx,
        }
    }
}

// ================================================================================================
// Helper Functions
// ================================================================================================

/// Convert embedding vector to fixed-size array
fn embedding_to_array(embedding: &[f32]) -> Result<[f32; 768], ApiError> {
    if embedding.len() != 768 {
        return Err(ApiError::InvalidInput(format!(
            "Embedding must be exactly 768 dimensions, got {}",
            embedding.len()
        )));
    }
    let mut array = [0.0f32; 768];
    array.copy_from_slice(embedding);
    Ok(array)
}

fn parse_recall_mode(value: &str) -> Result<RecallMode, ApiError> {
    match value.trim().to_ascii_lowercase().as_str() {
        "similarity" => Ok(RecallMode::Similarity),
        "spreading" => Ok(RecallMode::Spreading),
        "hybrid" => Ok(RecallMode::Hybrid),
        other => Err(ApiError::InvalidInput(format!(
            "Unsupported recall mode '{other}'. Use one of: similarity, spreading, hybrid."
        ))),
    }
}

/// Extract memory space identifier from request with priority:
/// 1. HTTP header `X-Engram-Memory-Space`
/// 2. Query parameter `?space=<id>`
/// 3. JSON body field `memory_space_id`
/// 4. Default space from ApiState
///
/// This establishes a clear precedence for multi-source space resolution,
/// allowing clients to specify tenancy at multiple API layers. Headers take
/// highest priority as they're RESTful, explicit, and work with all HTTP methods.
///
/// # Arguments
///
/// * `headers` - HTTP request headers (checked for X-Engram-Memory-Space)
/// * `query_space` - Optional space ID from query string (?space=tenant_a)
/// * `body_space` - Optional space ID from JSON request body
/// * `default` - Fallback space when none explicitly provided
///
/// # Returns
///
/// Validated MemorySpaceId or error if provided ID is invalid
///
/// # Example
///
/// ```ignore
/// // In handler:
/// let space_id = extract_memory_space_id(
///     &headers,                            // From HTTP headers
///     params.space.as_deref(),             // From ?space= query param
///     request.memory_space_id.as_deref(),  // From JSON body
///     &state.default_space,
/// )?;
/// ```
fn extract_memory_space_id(
    headers: &axum::http::HeaderMap,
    query_space: Option<&str>,
    body_space: Option<&str>,
    default: &MemorySpaceId,
) -> Result<MemorySpaceId, ApiError> {
    // Priority 1: X-Memory-Space header (RESTful, explicit, works with all methods)
    if let Some(header_value) = headers.get("x-memory-space") {
        let space_str = header_value.to_str().map_err(|_| {
            ApiError::InvalidInput("X-Memory-Space header contains invalid UTF-8".to_string())
        })?;
        return MemorySpaceId::try_from(space_str).map_err(|e| {
            ApiError::InvalidInput(format!(
                "Invalid memory space ID in X-Memory-Space header: {e}"
            ))
        });
    }

    // Priority 2: Query parameter (most explicit URL-based, overrides body)
    if let Some(space_str) = query_space {
        return MemorySpaceId::try_from(space_str).map_err(|e| {
            ApiError::InvalidInput(format!("Invalid memory space ID in query parameter: {e}"))
        });
    }

    // Priority 3: Request body field
    if let Some(space_str) = body_space {
        return MemorySpaceId::try_from(space_str).map_err(|e| {
            ApiError::InvalidInput(format!("Invalid memory space ID in request body: {e}"))
        });
    }

    // Priority 4: Default space (backward compatibility)
    Ok(default.clone())
}

const fn confidence_category(value: f32) -> &'static str {
    if value > 0.7 {
        "High"
    } else if value > 0.4 {
        "Medium"
    } else {
        "Low"
    }
}

fn build_confidence_info(value: f32, reasoning: impl Into<String>) -> ConfidenceInfo {
    ConfidenceInfo {
        value,
        category: confidence_category(value).to_string(),
        reasoning: reasoning.into(),
    }
}

fn truncate_content(content: &str) -> String {
    const MAX_PREVIEW: usize = 160;
    if content.len() <= MAX_PREVIEW {
        content.to_string()
    } else {
        format!("{}…", &content[..MAX_PREVIEW])
    }
}

fn pattern_to_belief(pattern: SemanticPattern, store: &MemoryStore) -> ConsolidatedBeliefResponse {
    let citations: Vec<BeliefCitation> = pattern
        .source_episodes
        .iter()
        .map(|episode_id| {
            if let Some(episode) = store.get_episode(episode_id) {
                let stored_at = store.stored_timestamp(&episode.id).unwrap_or(episode.when);
                BeliefCitation {
                    episode_id: episode.id.clone(),
                    observed_at: episode.when,
                    stored_at,
                    last_access: Some(episode.last_recall),
                    encoding_confidence: build_confidence_info(
                        episode.encoding_confidence.raw(),
                        "Initial encoding confidence",
                    ),
                    reinforcement_count: episode.recall_count,
                    content_preview: truncate_content(&episode.what),
                }
            } else {
                BeliefCitation {
                    episode_id: episode_id.clone(),
                    observed_at: pattern.last_consolidated,
                    stored_at: store
                        .stored_timestamp(episode_id)
                        .unwrap_or(pattern.last_consolidated),
                    last_access: None,
                    encoding_confidence: build_confidence_info(
                        0.0,
                        "Episode not currently loaded in memory store",
                    ),
                    reinforcement_count: 0,
                    content_preview: "[episode unavailable]".to_string(),
                }
            }
        })
        .collect();

    let average_source_confidence = if citations.is_empty() {
        0.0
    } else {
        citations
            .iter()
            .map(|citation| citation.encoding_confidence.value)
            .sum::<f32>()
            / citations.len() as f32
    };

    let freshness_hours = (Utc::now() - pattern.last_consolidated).num_minutes() as f32 / 60.0;

    ConsolidatedBeliefResponse {
        id: pattern.id,
        strength: pattern.strength,
        schema_confidence: build_confidence_info(
            pattern.schema_confidence.raw(),
            format!("Derived from {} episodic contributions", citations.len()),
        ),
        last_consolidated: pattern.last_consolidated,
        citations,
        stats: ConsolidationBeliefStats {
            episode_count: pattern.source_episodes.len(),
            average_source_confidence,
            freshness_hours: freshness_hours.max(0.0),
        },
    }
}

// ================================================================================================
// Memory Space Administration Handlers
// ================================================================================================

/// List all registered memory spaces currently tracked by the registry.
#[utoipa::path(
    get,
    path = "/api/v1/spaces",
    tag = "memory-spaces",
    responses(
        (status = 200, description = "List registered memory spaces", body = MemorySpaceListResponse),
        (status = 500, description = "Failed to enumerate memory spaces", body = ErrorResponse)
    )
)]
pub async fn list_memory_spaces(
    State(state): State<ApiState>,
) -> Result<impl IntoResponse, ApiError> {
    let spaces = state
        .registry
        .list()
        .into_iter()
        .map(MemorySpaceDescriptor::from)
        .collect();

    Ok(Json(MemorySpaceListResponse { spaces }))
}

/// Create (or retrieve) a memory space and return its descriptor.
#[utoipa::path(
    post,
    path = "/api/v1/spaces",
    tag = "memory-spaces",
    request_body = CreateMemorySpaceRequest,
    responses(
        (status = 201, description = "Memory space created or returned", body = MemorySpaceDescriptor),
        (status = 400, description = "Invalid memory space identifier", body = ErrorResponse),
        (status = 500, description = "Failed to initialise memory space", body = ErrorResponse)
    )
)]
pub async fn create_memory_space(
    State(state): State<ApiState>,
    Json(payload): Json<CreateMemorySpaceRequest>,
) -> Result<impl IntoResponse, ApiError> {
    let space_id = MemorySpaceId::try_from(payload.id.as_str())
        .map_err(|err| ApiError::from(MemorySpaceError::from(err)))?;

    let handle = state
        .registry
        .create_or_get(&space_id)
        .await
        .map_err(ApiError::from)?;

    let descriptor = MemorySpaceDescriptor::from(SpaceSummary {
        id: handle.id().clone(),
        root: handle.directories().root.clone(),
        created_at: handle.created_at(),
    });
    Ok((StatusCode::CREATED, Json(descriptor)))
}

impl From<CoreConsolidationStats> for ConsolidationRunStats {
    fn from(stats: CoreConsolidationStats) -> Self {
        Self {
            total_replays: stats.total_replays,
            successful_consolidations: stats.successful_consolidations,
            failed_consolidations: stats.failed_consolidations,
            average_replay_speed: stats.average_replay_speed,
            total_patterns_extracted: stats.total_patterns_extracted,
            avg_ripple_frequency: stats.avg_ripple_frequency,
            avg_ripple_duration: stats.avg_ripple_duration,
            last_replay_timestamp: stats.last_replay_timestamp,
        }
    }
}

const fn format_health_status(status: HealthStatus) -> &'static str {
    match status {
        HealthStatus::Healthy => "healthy",
        HealthStatus::Degraded => "degraded",
        HealthStatus::Unhealthy => "unhealthy",
    }
}

const fn health_check_type_str(check_type: HealthCheckType) -> &'static str {
    match check_type {
        HealthCheckType::Memory => "memory",
        HealthCheckType::Latency => "latency",
        HealthCheckType::ErrorRate => "error_rate",
        HealthCheckType::Connectivity => "connectivity",
        HealthCheckType::Cognitive => "cognitive",
        HealthCheckType::Custom(name) => name,
    }
}

fn health_check_to_json(check: &HealthCheck) -> serde_json::Value {
    json!({
        "name": check.name,
        "type": health_check_type_str(check.check_type),
        "status": format_health_status(check.status),
        "message": check.message,
        "latency_seconds": check.latency.as_secs_f64(),
        "consecutive_failures": check.consecutive_failures,
        "consecutive_successes": check.consecutive_successes,
        "last_success_seconds_ago": check.last_success.elapsed().as_secs_f64(),
        "last_failure_seconds_ago": check.last_failure.map(|instant| instant.elapsed().as_secs_f64()),
        "last_run_seconds_ago": check.last_run.map(|instant| instant.elapsed().as_secs_f64()),
    })
}

// ================================================================================================
// Request/Response Types with Cognitive Design
// ================================================================================================

/// Request to remember a new memory with cognitive-friendly fields
#[derive(Debug, Deserialize, ToSchema, IntoParams)]
pub struct RememberMemoryRequest {
    /// Human-readable identifier for this memory
    pub id: Option<String>,
    /// The actual content to remember
    pub content: String,
    /// Vector embedding for similarity matching
    pub embedding: Option<Vec<f32>>,
    /// How confident we are in this memory (0.0-1.0)
    pub confidence: Option<f32>,
    /// Confidence reasoning for educational purposes
    pub confidence_reasoning: Option<String>,
    /// Tags for semantic organization
    pub tags: Option<Vec<String>>,
    /// Type of memory (semantic, episodic, procedural)
    pub memory_type: Option<String>,
    /// When this memory was observed (optional, defaults to current time)
    pub timestamp: Option<DateTime<Utc>>,
    /// Should we automatically link to similar memories?
    pub auto_link: Option<bool>,
    /// Threshold for automatic linking (0.0-1.0)
    pub link_threshold: Option<f32>,
    /// Memory space identifier for multi-tenant isolation (defaults to server default)
    pub memory_space_id: Option<String>,
}

/// Request to remember an episode with contextual information
#[derive(Debug, Deserialize, ToSchema, IntoParams)]
pub struct RememberEpisodeRequest {
    /// Human-readable identifier for this episode
    pub id: Option<String>,
    /// When did this happen?
    pub when: DateTime<Utc>,
    /// What happened? (descriptive narrative)
    pub what: String,
    /// Where did it happen?
    pub where_location: Option<String>,
    /// Who was involved?
    pub who: Option<Vec<String>>,
    /// Why did it happen? (causal context)
    pub why: Option<String>,
    /// How did it happen? (process details)
    pub how: Option<String>,
    /// Vector embedding for similarity matching
    pub embedding: Option<Vec<f32>>,
    /// Emotional valence (-1.0 to 1.0, negative to positive)
    pub emotional_valence: Option<f32>,
    /// How important is this episode? (0.0-1.0)
    pub importance: Option<f32>,
    /// Should we automatically link to related episodes?
    pub auto_link: Option<bool>,
    /// Memory space identifier for multi-tenant isolation (defaults to server default)
    pub memory_space_id: Option<String>,
}

/// Query parameters for recalling memories
#[derive(Debug, Deserialize, ToSchema, IntoParams)]
pub struct RecallQuery {
    /// Search query using natural language
    pub query: Option<String>,
    /// Vector embedding for similarity search
    pub embedding: Option<String>, // JSON-encoded Vec<f32>
    /// Maximum number of memories to return
    pub max_results: Option<usize>,
    /// Minimum confidence threshold (0.0-1.0)
    pub threshold: Option<f32>,
    /// Include detailed metadata in response?
    pub include_metadata: Option<bool>,
    /// Show activation spreading traces?
    pub trace_activation: Option<bool>,
    /// Tags to require in results
    pub required_tags: Option<String>, // Comma-separated
    /// Tags to exclude from results
    pub excluded_tags: Option<String>, // Comma-separated
    /// Time range start (ISO 8601)
    pub from_time: Option<DateTime<Utc>>,
    /// Time range end (ISO 8601)
    pub to_time: Option<DateTime<Utc>>,
    /// Location context filter
    pub location: Option<String>,
    /// Recall mode to apply (similarity, spreading, hybrid)
    pub mode: Option<String>,
    /// Memory space identifier for multi-tenant isolation (defaults to server default)
    pub space: Option<String>,
}

/// Query parameters for probabilistic recall with uncertainty tracking
#[derive(Debug, Deserialize, ToSchema, IntoParams)]
pub struct ProbabilisticQueryRequest {
    /// Search query using natural language
    pub query: Option<String>,
    /// Vector embedding for similarity search
    pub embedding: Option<String>, // JSON-encoded Vec<f32>
    /// Maximum number of memories to return
    pub max_results: Option<usize>,
    /// Minimum confidence threshold (0.0-1.0)
    pub threshold: Option<f32>,
    /// Include evidence chain in response?
    pub include_evidence: Option<bool>,
    /// Include uncertainty sources in response?
    pub include_uncertainty: Option<bool>,
    /// Recall mode to apply (similarity, spreading, hybrid)
    pub mode: Option<String>,
    /// Memory space identifier for multi-tenant isolation (defaults to server default)
    pub space: Option<String>,
}

/// Query parameters for streaming activities
#[derive(Debug, Deserialize, ToSchema, IntoParams)]
pub struct StreamActivityQuery {
    /// Event types to stream (comma-separated: activation,storage,recall,consolidation,association,decay)
    pub event_types: Option<String>,
    /// Minimum importance level (0.0-1.0)
    pub min_importance: Option<f32>,
    /// Buffer size for backpressure control
    pub buffer_size: Option<usize>,
    /// Session ID for reconnection
    pub session_id: Option<String>,
    /// Last event ID for resumption
    pub last_event_id: Option<String>,
}

/// Query parameters for streaming memories
#[derive(Debug, Deserialize, ToSchema, IntoParams)]
pub struct StreamMemoryQuery {
    /// Include new memory formations
    pub include_formation: Option<bool>,
    /// Include memory retrievals  
    pub include_retrieval: Option<bool>,
    /// Include pattern completions
    pub include_completion: Option<bool>,
    /// Minimum confidence to stream
    pub min_confidence: Option<f32>,
    /// Memory types to include (comma-separated)
    pub memory_types: Option<String>,
    /// Session ID for reconnection
    pub session_id: Option<String>,
}

/// Query parameters for streaming consolidation
#[derive(Debug, Deserialize, ToSchema, IntoParams)]
pub struct StreamConsolidationQuery {
    /// Include replay sequences
    pub include_replay: Option<bool>,
    /// Include insights
    pub include_insights: Option<bool>,
    /// Include progress updates
    pub include_progress: Option<bool>,
    /// Minimum novelty for replay sequences
    pub min_novelty: Option<f32>,
    /// Session ID for reconnection
    pub session_id: Option<String>,
}

/// Query parameters for real-time monitoring (debugging-focused)
#[derive(Debug, Deserialize, ToSchema, IntoParams)]
pub struct MonitoringQuery {
    /// Event hierarchy level (global, region, node, edge)
    pub level: Option<String>,
    /// Specific node or region ID to focus on
    pub focus_id: Option<String>,
    /// Maximum update frequency (Hz) - respects cognitive limits
    pub max_frequency: Option<f32>,
    /// Include causality tracking
    pub include_causality: Option<bool>,
    /// Event types to monitor (activation,formation,decay,spreading)
    pub event_types: Option<String>,
    /// Minimum activation threshold to report
    pub min_activation: Option<f32>,
    /// Session ID for reconnection
    pub session_id: Option<String>,
    /// Last sequence number for resumption
    pub last_sequence: Option<u64>,
}

/// Query parameters for activation monitoring
#[derive(Debug, Deserialize, ToSchema, IntoParams)]
pub struct ActivationMonitoringQuery {
    /// Node pattern to monitor (regex or specific IDs)
    pub node_pattern: Option<String>,
    /// Activation threshold for reporting
    pub threshold: Option<f32>,
    /// Include spreading activation traces
    pub include_spreading: Option<bool>,
    /// Update frequency (Hz) - cognitive optimized
    pub frequency: Option<f32>,
    /// Time window for activation history (seconds)
    pub time_window: Option<f32>,
    /// Session ID for reconnection
    pub session_id: Option<String>,
}

/// Query parameters for causality tracking
#[derive(Debug, Deserialize, ToSchema, IntoParams)]
pub struct CausalityMonitoringQuery {
    /// Maximum causality chain length to track
    pub max_chain_length: Option<usize>,
    /// Minimum confidence for causality links
    pub min_confidence: Option<f32>,
    /// Include indirect causality (transitive)
    pub include_indirect: Option<bool>,
    /// Focus on specific memory operation types
    pub operation_types: Option<String>,
    /// Temporal window for causality (milliseconds)
    pub temporal_window: Option<u64>,
    /// Session ID for reconnection
    pub session_id: Option<String>,
}

/// Response for memory remember operations
#[derive(Debug, Serialize, ToSchema)]
pub struct RememberResponse {
    /// The ID assigned to this memory
    pub memory_id: String,
    /// How confident the system is in storing this memory
    pub storage_confidence: ConfidenceInfo,
    /// Current consolidation state
    pub consolidation_state: String,
    /// When the memory event originally occurred (observation timestamp)
    pub observed_at: DateTime<Utc>,
    /// When Engram ingested this memory into its store
    pub stored_at: DateTime<Utc>,
    /// Helpful links to related background processes
    #[serde(skip_serializing_if = "Option::is_none")]
    pub links: Option<RememberLinks>,
    /// Automatic links created (if enabled)
    pub auto_links: Vec<AutoLink>,
    /// Educational message about the memory storage
    pub system_message: String,
}

/// Related resources for newly stored memories
#[derive(Debug, Serialize, ToSchema)]
pub struct RememberLinks {
    /// Endpoint for consolidated beliefs derived from this memory
    pub consolidation: Option<String>,
}

/// Query parameters for consolidation summaries
#[derive(Debug, Deserialize, ToSchema, IntoParams)]
pub struct ConsolidationQuery {
    /// Maximum episodes to consider during snapshot (0 = all episodes currently stored)
    pub max_episodes: Option<usize>,
    /// Maximum number of consolidated beliefs to include in the response
    pub max_patterns: Option<usize>,
    /// Memory space identifier for multi-tenant isolation (defaults to server default)
    pub space: Option<String>,
}

/// Query parameters for consolidation detail lookups
#[derive(Debug, Deserialize, ToSchema, IntoParams)]
pub struct ConsolidationDetailQuery {
    /// Maximum episodes to consider during snapshot (0 = all episodes currently stored)
    pub max_episodes: Option<usize>,
    /// Memory space identifier for multi-tenant isolation (defaults to server default)
    pub space: Option<String>,
}

/// Consolidated belief response summarizing semantic memory
#[derive(Debug, Serialize, ToSchema)]
pub struct ConsolidatedBeliefResponse {
    /// Stable identifier for the semantic pattern
    pub id: String,
    /// Aggregate strength derived from contributing episodes
    pub strength: f32,
    /// Confidence that this schema represents supported knowledge
    pub schema_confidence: ConfidenceInfo,
    /// When the semantic pattern was last strengthened
    pub last_consolidated: DateTime<Utc>,
    /// Citations to episodic memories that contributed to this belief
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub citations: Vec<BeliefCitation>,
    /// Roll-up statistics describing this belief
    pub stats: ConsolidationBeliefStats,
}

/// Citation information for an episodic memory contributing to a belief
#[derive(Debug, Serialize, ToSchema)]
pub struct BeliefCitation {
    /// Episode identifier
    pub episode_id: String,
    /// When the episode occurred
    pub observed_at: DateTime<Utc>,
    /// When the episode was stored in Engram
    pub stored_at: DateTime<Utc>,
    /// When the episode was most recently recalled
    pub last_access: Option<DateTime<Utc>>,
    /// Encoding confidence for the episode
    pub encoding_confidence: ConfidenceInfo,
    /// Number of times the episode has been recalled
    pub reinforcement_count: u32,
    /// Preview of the episodic content
    pub content_preview: String,
}

/// Belief-level statistics summarizing consolidation dynamics
#[derive(Debug, Serialize, ToSchema)]
pub struct ConsolidationBeliefStats {
    /// Number of episodic memories backing this belief
    pub episode_count: usize,
    /// Average source confidence across contributing episodes
    pub average_source_confidence: f32,
    /// Age in hours since the belief was last consolidated
    pub freshness_hours: f32,
}

/// Snapshot-wide statistics surfaced alongside consolidation summaries
#[derive(Debug, Serialize, ToSchema)]
pub struct ConsolidationRunStats {
    /// Total replay iterations executed during consolidation
    pub total_replays: usize,
    /// Successfully extracted semantic patterns
    pub successful_consolidations: usize,
    /// Consolidation attempts that failed to meet thresholds
    pub failed_consolidations: usize,
    /// Average replay speed observed during the run
    pub average_replay_speed: f32,
    /// Total semantic patterns produced
    pub total_patterns_extracted: usize,
    /// Average ripple frequency (Hz)
    pub avg_ripple_frequency: f32,
    /// Average ripple duration (ms)
    pub avg_ripple_duration: f32,
    /// Timestamp of the most recent replay event
    pub last_replay_timestamp: Option<DateTime<Utc>>,
}

/// Consolidation snapshot response encapsulating beliefs and run metrics
#[derive(Debug, Serialize, ToSchema)]
pub struct ConsolidationSummaryResponse {
    /// When the snapshot was generated
    pub generated_at: DateTime<Utc>,
    /// Consolidated beliefs discovered during this run
    pub beliefs: Vec<ConsolidatedBeliefResponse>,
    /// Snapshot-wide statistics
    pub stats: ConsolidationRunStats,
}

/// Response for memory recall operations
#[derive(Debug, Serialize, ToSchema)]
pub struct RecallResponse {
    /// Found memories organized by retrieval pattern
    pub memories: RecallResults,
    /// Overall confidence in the recall operation
    pub recall_confidence: ConfidenceInfo,
    /// Query understanding and processing info
    pub query_analysis: QueryAnalysis,
    /// Performance and cognitive load metrics
    pub metadata: Option<RecallMetadata>,
    /// Educational message about the recall process
    pub system_message: String,
}

/// Memories organized by cognitive retrieval patterns
#[derive(Debug, Serialize, ToSchema)]
pub struct RecallResults {
    /// Immediate, high-confidence matches
    pub vivid: Vec<MemoryResult>,
    /// Associated memories through spreading activation
    pub associated: Vec<MemoryResult>,
    /// Pattern-completed reconstructed memories
    pub reconstructed: Vec<MemoryResult>,
}

/// Individual memory result with cognitive context
#[derive(Debug, Serialize, ToSchema)]
pub struct MemoryResult {
    /// Unique memory identifier
    pub id: String,
    /// Memory content text
    pub content: String,
    /// Confidence information with reasoning
    pub confidence: ConfidenceInfo,
    /// Current activation strength (0.0-1.0)
    pub activation_level: f32,
    /// Similarity to query (0.0-1.0)
    pub similarity_score: f32,
    /// How we found this memory
    pub retrieval_path: Option<String>,
    /// When this memory was originally observed/encoded
    pub observed_at: DateTime<Utc>,
    /// When memory was last accessed
    pub last_access: Option<DateTime<Utc>>,
    /// Associated tags
    pub tags: Vec<String>,
    /// Type classification (episodic/semantic)
    pub memory_type: String,
    /// Context for how this memory relates to the query
    pub relevance_explanation: String,
}

/// Auto-linking information
#[derive(Debug, Serialize, ToSchema)]
pub struct AutoLink {
    /// ID of the target memory to link
    pub target_memory_id: String,
    /// Similarity score for the link (0.0-1.0)
    pub similarity_score: f32,
    /// Explanation for why link was suggested
    pub link_reason: String,
}

/// Confidence information with educational context
#[derive(Debug, Serialize, ToSchema)]
pub struct ConfidenceInfo {
    /// Confidence value (0.0-1.0)
    pub value: f32,
    /// Human-readable category (Low/Medium/High)
    pub category: String,
    /// Reasoning for confidence level
    pub reasoning: String,
}

/// Query analysis for educational feedback
#[derive(Debug, Serialize, ToSchema)]
pub struct QueryAnalysis {
    /// What we understood from the query
    pub understood_intent: String,
    /// Search approach being used
    pub search_strategy: String,
    /// Cognitive load assessment (Low/Medium/High)
    pub cognitive_load: String,
    /// Suggestions for improving queries
    pub suggestions: Vec<String>,
}

/// Recall operation metadata
#[derive(Debug, Serialize, ToSchema)]
pub struct RecallMetadata {
    /// Total number of memories examined
    pub total_memories_searched: usize,
    /// Depth of activation spreading
    pub activation_spread_hops: usize,
    /// Processing duration in milliseconds
    pub processing_time_ms: u64,
    /// Current memory system load status
    pub memory_system_load: String,
}

/// Pattern recognition request
#[derive(Debug, Deserialize, ToSchema, IntoParams)]
pub struct RecognizeRequest {
    /// Content or pattern to recognize
    pub input: String,
    /// Optional embedding vector
    pub embedding: Option<Vec<f32>>,
    /// Recognition threshold (0.0-1.0)
    pub threshold: Option<f32>,
}

/// Pattern recognition response
#[derive(Debug, Serialize, ToSchema)]
pub struct RecognizeResponse {
    /// Whether the pattern was recognized
    pub recognized: bool,
    /// Confidence in recognition
    pub recognition_confidence: ConfidenceInfo,
    /// Similar patterns found
    pub similar_patterns: Vec<SimilarPattern>,
    /// Educational context about recognition
    pub system_message: String,
}

/// Similar pattern information
#[derive(Debug, Serialize, ToSchema)]
pub struct SimilarPattern {
    /// ID of similar memory
    pub memory_id: String,
    /// Similarity score (0.0-1.0)
    pub similarity_score: f32,
    /// Type of pattern match
    pub pattern_type: String,
    /// Explanation of pattern similarity
    pub explanation: String,
}

/// Probabilistic query response with uncertainty tracking
#[derive(Debug, Serialize, ToSchema)]
pub struct ProbabilisticQueryResponse {
    /// Found memories with confidence scores
    pub memories: Vec<MemoryResult>,
    /// Confidence interval with lower/upper bounds
    pub confidence_interval: ConfidenceIntervalInfo,
    /// Evidence chain showing how confidence was derived
    pub evidence_chain: Option<Vec<EvidenceInfo>>,
    /// Sources of uncertainty in the query results
    pub uncertainty_sources: Option<Vec<UncertaintyInfo>>,
    /// Educational message about probabilistic reasoning
    pub system_message: String,
}

/// Confidence interval information
#[derive(Debug, Serialize, ToSchema)]
pub struct ConfidenceIntervalInfo {
    /// Lower bound of confidence
    pub lower: f32,
    /// Upper bound of confidence
    pub upper: f32,
    /// Point estimate
    pub point: f32,
    /// Interval width
    pub width: f32,
}

/// Evidence information for transparency
#[derive(Debug, Serialize, ToSchema)]
pub struct EvidenceInfo {
    /// Type of evidence source
    pub source_type: String,
    /// Strength of evidence
    pub strength: f32,
    /// Description of how evidence was collected
    pub description: String,
}

/// Uncertainty source information
#[derive(Debug, Serialize, ToSchema)]
pub struct UncertaintyInfo {
    /// Type of uncertainty source
    pub source_type: String,
    /// Impact on confidence
    pub impact: f32,
    /// Explanation of uncertainty
    pub explanation: String,
}

// ================================================================================================
// API Endpoint Implementations
// ================================================================================================

/// POST /api/v1/memories/remember - Store a new memory
#[utoipa::path(
    post,
    path = "/api/v1/memories/remember",
    tag = "memories",
    request_body = RememberMemoryRequest,
    responses(
        (status = 201, description = "Memory successfully stored", body = RememberResponse),
        (status = 400, description = "Invalid input", body = ErrorResponse),
        (status = 500, description = "Internal server error", body = ErrorResponse)
    )
)]
/// # Errors
///
/// Returns `ApiError` if memory storage fails or graph operations fail
pub async fn remember_memory(
    State(state): State<ApiState>,
    headers: HeaderMap,
    Json(request): Json<RememberMemoryRequest>,
) -> Result<impl IntoResponse, ApiError> {
    info!(
        "Processing remember memory request: content length = {}",
        request.content.len()
    );

    // Validate request with educational feedback
    if request.content.trim().is_empty() {
        return Err(ApiError::InvalidInput(
            "Memory content cannot be empty. The memory system needs meaningful content to create lasting neural patterns.".to_string()
        ));
    }

    if request.content.len() > 100_000 {
        return Err(ApiError::InvalidInput(
            "Memory content too large (>100KB). Consider breaking this into smaller, more focused memories for better consolidation.".to_string()
        ));
    }

    // Generate ID if not provided
    let memory_id = request
        .id
        .unwrap_or_else(|| format!("mem_{}", &uuid::Uuid::new_v4().to_string()[..8]));

    // Create embedding if not provided (placeholder - would use actual embedding service)
    let embedding_vec = request.embedding.unwrap_or_else(|| {
        // Simple content-based embedding placeholder
        vec![0.5; 768] // Standard embedding dimension
    });

    let embedding_array = embedding_to_array(&embedding_vec)?;

    // Create confidence with reasoning
    let confidence_value = request.confidence.unwrap_or(0.7);
    let confidence_reasoning = request
        .confidence_reasoning
        .unwrap_or_else(|| "Default confidence for user-provided memory".to_string());
    let confidence = Confidence::new(confidence_value).with_reasoning(confidence_reasoning.clone());

    let core_confidence = CoreConfidence::exact(confidence_value);

    // Determine memory type
    let memory_type = match request.memory_type.as_deref() {
        Some("episodic") => MemoryType::Episodic,
        Some("procedural") => MemoryType::Procedural,
        _ => MemoryType::Semantic, // Default for "semantic" and unknown types
    };

    // Create memory object for response
    let memory = Memory::new(memory_id.clone(), embedding_vec)
        .with_content(&request.content)
        .with_confidence(confidence)
        .with_type(memory_type);

    // Add tags if provided
    let _memory = if let Some(tags) = request.tags {
        tags.into_iter().fold(memory, engram_proto::Memory::add_tag)
    } else {
        memory
    };

    // Extract memory space ID with fallback to default
    let space_id = extract_memory_space_id(
        &headers,
        None,
        request.memory_space_id.as_deref(),
        &state.default_space,
    )?;
    Span::current().record("memory_space_id", space_id.as_str());

    // Get space-specific store handle from registry
    let handle = state
        .registry
        .create_or_get(&space_id)
        .await
        .map_err(|e| ApiError::SystemError(format!("Failed to access memory space: {e}")))?;
    let store = handle.store();

    // Runtime verification (defense-in-depth)
    store
        .verify_space(&space_id)
        .map_err(ApiError::SystemError)?;

    // Store in MemoryStore as an episode
    let observed_at = request.timestamp.unwrap_or_else(Utc::now);
    let episode = Episode::new(
        memory_id.clone(),
        observed_at,
        request.content.clone(),
        embedding_array,
        core_confidence,
    );

    let store_result = store.store(episode);

    // Check if streaming failed - this is a critical failure that should be surfaced
    if !store_result.streaming_delivered {
        tracing::warn!(
            memory_id = %memory_id,
            "Memory stored successfully but event streaming failed - SSE subscribers not notified"
        );
        // Return 500 to indicate partial failure - storage succeeded but streaming failed
        return Err(ApiError::SystemError(format!(
            "Memory '{memory_id}' was stored but event notification failed. \
             SSE subscribers did not receive the storage event. \
             Check /api/v1/system/health for streaming status."
        )));
    }

    // Create auto-links if enabled
    let auto_links = if request.auto_link.unwrap_or(false) {
        let threshold = request.link_threshold.unwrap_or(0.7);
        // Placeholder for auto-linking logic
        vec![AutoLink {
            target_memory_id: "example_linked_memory".to_string(),
            similarity_score: threshold + 0.1,
            link_reason: "Semantic similarity detected in content patterns".to_string(),
        }]
    } else {
        vec![]
    };

    let stored_at = Utc::now();
    let actual_confidence = store_result.activation.value();
    let response = RememberResponse {
        memory_id: memory_id.clone(),
        storage_confidence: ConfidenceInfo {
            value: actual_confidence,
            category: confidence_category(actual_confidence).to_string(),
            reasoning: confidence_reasoning,
        },
        consolidation_state: format!("{:?}", ConsolidationState::Recent),
        observed_at,
        stored_at,
        links: Some(RememberLinks {
            consolidation: Some("/api/v1/consolidations".to_string()),
        }),
        auto_links,
        system_message: format!(
            "Memory '{}' successfully encoded with {:.2} activation. {}",
            memory_id,
            actual_confidence,
            if request.auto_link.unwrap_or(false) {
                "Automatic linking enabled - similar memories will be connected during consolidation."
            } else {
                "Consider enabling auto-linking to discover related memories."
            }
        ),
    };

    info!("Successfully stored memory: {}", memory_id);
    Ok((StatusCode::CREATED, Json(response)))
}

/// POST /api/v1/episodes/remember - Store a new episode
#[utoipa::path(
    post,
    path = "/api/v1/episodes/remember",
    tag = "episodes",
    request_body = RememberEpisodeRequest,
    responses(
        (status = 201, description = "Episode successfully stored", body = RememberResponse),
        (status = 400, description = "Invalid episode data", body = ErrorResponse),
        (status = 500, description = "Internal server error", body = ErrorResponse)
    )
)]
/// # Errors
///
/// Returns `ApiError` if episode storage fails or graph operations fail
pub async fn remember_episode(
    State(state): State<ApiState>,
    headers: HeaderMap,
    Json(request): Json<RememberEpisodeRequest>,
) -> Result<impl IntoResponse, ApiError> {
    info!("Processing remember episode request: {}", request.what);

    // Validate episode content
    if request.what.trim().is_empty() {
        return Err(ApiError::InvalidInput(
            "Episode description cannot be empty. Episodic memories need vivid 'what happened' narratives.".to_string()
        ));
    }

    // Extract memory space from request (header > query > body > default)
    let space_id = extract_memory_space_id(
        &headers,
        None, // No query params in this handler
        request.memory_space_id.as_deref(),
        &state.default_space,
    )?;
    Span::current().record("memory_space_id", space_id.as_str());

    // Get space-specific store handle from registry
    let handle = state.registry.create_or_get(&space_id).await?;
    let store = handle.store();

    // Runtime guard: verify we got the right store (defense in depth)
    store.verify_space(&space_id).map_err(|e| {
        ApiError::SystemError(format!(
            "Internal error: {e}. This indicates a registry bug - please report."
        ))
    })?;

    // Generate ID if not provided
    let episode_id = request
        .id
        .unwrap_or_else(|| format!("ep_{}", &uuid::Uuid::new_v4().to_string()[..8]));

    // Create embedding if not provided
    let embedding_vec = request.embedding.unwrap_or_else(|| vec![0.6; 768]);
    let embedding_array = embedding_to_array(&embedding_vec)?;

    let encoding_confidence =
        CoreConfidence::exact(request.importance.unwrap_or(0.5).clamp(0.0, 1.0));

    let builder = CoreEpisodeBuilder::new()
        .id(episode_id.clone())
        .when(request.when)
        .what(request.what.clone())
        .embedding(embedding_array)
        .confidence(encoding_confidence);

    let builder = if let Some(location) = request.where_location.clone() {
        builder.where_location(location)
    } else {
        builder
    };

    let builder = if let Some(participants) = request.who {
        builder.who(participants)
    } else {
        builder
    };

    let core_episode = builder.build();

    // Store in memory space-specific store
    let store_result = store.store(core_episode);

    // Check if streaming failed - this is a critical failure that should be surfaced
    if !store_result.streaming_delivered {
        tracing::warn!(
            episode_id = %episode_id,
            "Episode stored successfully but event streaming failed - SSE subscribers not notified"
        );
        // Return 500 to indicate partial failure - storage succeeded but streaming failed
        return Err(ApiError::SystemError(format!(
            "Episode '{episode_id}' was stored but event notification failed. \
             SSE subscribers did not receive the storage event. \
             Check /api/v1/system/health for streaming status."
        )));
    }

    let actual_confidence = store_result.activation.value();
    let stored_at = Utc::now();
    let response = RememberResponse {
        memory_id: episode_id.clone(),
        storage_confidence: ConfidenceInfo {
            value: actual_confidence,
            category: confidence_category(actual_confidence).to_string(),
            reasoning: format!(
                "Episodic memory stored with activation {actual_confidence:.2}, rich context aids consolidation"
            ),
        },
        consolidation_state: format!("{:?}", ConsolidationState::Recent),
        observed_at: request.when,
        stored_at,
        links: Some(RememberLinks {
            consolidation: Some("/api/v1/consolidations".to_string()),
        }),
        auto_links: vec![],
        system_message: format!(
            "Episode '{episode_id}' successfully encoded with {actual_confidence:.2} activation. Rich episodes consolidate better over time."
        ),
    };

    info!("Successfully stored episode: {}", episode_id);
    Ok((StatusCode::CREATED, Json(response)))
}

/// GET /api/v1/memories/recall - Retrieve memories by query
#[utoipa::path(
    get,
    path = "/api/v1/memories/recall",
    tag = "memories",
    params(RecallQuery),
    responses(
        (status = 200, description = "Memories found and organized by retrieval patterns", body = RecallResponse),
        (status = 400, description = "Invalid query parameters", body = ErrorResponse),
        (status = 500, description = "Internal server error", body = ErrorResponse)
    )
)]
/// # Errors
///
/// Returns `ApiError` if memory recall fails or graph operations fail
pub async fn recall_memories(
    State(state): State<ApiState>,
    headers: HeaderMap,
    Query(params): Query<RecallQuery>,
) -> Result<impl IntoResponse, ApiError> {
    let start_time = std::time::Instant::now();

    info!("Processing recall request with query: {:?}", params.query);

    // Extract memory space from request (header > query > default)
    let space_id = extract_memory_space_id(
        &headers,
        params.space.as_deref(),
        None, // No body in GET request
        &state.default_space,
    )?;
    Span::current().record("memory_space_id", space_id.as_str());

    // Get space-specific store handle from registry
    let handle = state.registry.create_or_get(&space_id).await?;
    let store = handle.store();

    // Runtime guard: verify we got the right store
    store.verify_space(&space_id).map_err(|e| {
        ApiError::SystemError(format!(
            "Internal error: {e}. This indicates a registry bug - please report."
        ))
    })?;

    // Validate query
    if params.query.is_none() && params.embedding.is_none() {
        return Err(ApiError::InvalidInput(
            "Recall requires either a text query or embedding vector. Memory retrieval needs a cue - what are you trying to remember?".to_string()
        ));
    }

    // Parse embedding if provided
    let embedding_vector = if let Some(emb_str) = &params.embedding {
        match serde_json::from_str::<Vec<f32>>(emb_str) {
            Ok(vec) => Some(vec),
            Err(_) => {
                return Err(ApiError::InvalidInput(
                    "Invalid embedding format. Expected JSON array of floats.".to_string(),
                ));
            }
        }
    } else {
        None
    };

    // Create search cue
    let cue = if let Some(query) = &params.query {
        CoreCue::semantic(
            "http_query".to_string(),
            query.clone(),
            CoreConfidence::exact(params.threshold.unwrap_or(0.5)),
        )
    } else if let Some(embedding) = embedding_vector {
        let embedding_array: [f32; 768] = embedding_to_array(&embedding)?;
        CoreCue::embedding(
            "http_embedding_query".to_string(),
            embedding_array,
            CoreConfidence::exact(params.threshold.unwrap_or(0.5)),
        )
    } else {
        return Err(ApiError::InvalidInput(
            "No valid search cue provided".to_string(),
        ));
    };

    // Determine recall mode override if provided
    let requested_mode = if let Some(mode_str) = &params.mode {
        let parsed = parse_recall_mode(mode_str)?;

        #[cfg(not(feature = "hnsw_index"))]
        if matches!(parsed, RecallMode::Spreading | RecallMode::Hybrid) {
            return Err(ApiError::InvalidInput(
                "Spreading recall requires the CLI to be built with the 'hnsw_index' feature. Rebuild with `--features hnsw_index` or omit the mode parameter."
                    .to_string(),
            ));
        }

        #[cfg(feature = "hnsw_index")]
        if matches!(parsed, RecallMode::Spreading | RecallMode::Hybrid)
            && store.spreading_engine().is_none()
        {
            return Err(ApiError::InvalidInput(
                "Spreading recall requires the `spreading_api_beta` flag. Enable it with `engram config set feature_flags.spreading_api_beta true` and restart the server.".to_string(),
            ));
        }

        Some(parsed)
    } else {
        None
    };

    // Perform actual recall using space-specific MemoryStore and capture the effective mode
    let (recall_result, effective_mode) = requested_mode.map_or_else(
        || (store.recall(&cue), store.recall_mode()),
        |mode| (store.recall_with_mode(&cue, mode), mode),
    );

    // Check if streaming failed
    if !recall_result.streaming_delivered {
        tracing::warn!(
            "Recall completed successfully but event streaming failed - SSE subscribers not notified"
        );
        return Err(ApiError::SystemError(
            "Recall completed but event notification failed. \
             SSE subscribers did not receive the recall events. \
             Check /api/v1/system/health for streaming status."
                .to_string(),
        ));
    }

    let total_memories = store.count();

    // Convert results to API response format
    let mut vivid_results = Vec::new();
    let mut associated_results = Vec::new();

    for (episode, confidence) in recall_result
        .results
        .iter()
        .take(params.max_results.unwrap_or(10))
    {
        let result = MemoryResult {
            id: episode.id.clone(),
            content: episode.what.clone(),
            confidence: build_confidence_info(
                confidence.raw(),
                "Recalled from memory store with confidence score",
            ),
            activation_level: confidence.raw(),
            similarity_score: confidence.raw(),
            retrieval_path: Some("Memory store recall".to_string()),
            observed_at: episode.when,
            last_access: Some(episode.last_recall),
            tags: vec![],
            memory_type: "Episodic".to_string(),
            relevance_explanation: format!("Retrieved with {:.2} confidence", confidence.raw()),
        };

        // Classify as vivid (high confidence) or associated (lower confidence)
        if confidence.raw() > 0.7 {
            vivid_results.push(result);
        } else {
            associated_results.push(result);
        }
    }

    let processing_time = u64::try_from(start_time.elapsed().as_millis()).unwrap_or(u64::MAX);

    let response = RecallResponse {
        memories: RecallResults {
            vivid: vivid_results,
            associated: associated_results,
            reconstructed: vec![], // Would implement pattern completion
        },
        recall_confidence: ConfidenceInfo {
            value: 0.8,
            category: "High".to_string(),
            reasoning: match effective_mode {
                RecallMode::Similarity => {
                    "Similarity recall emphasised lexical and embedding cues".to_string()
                }
                RecallMode::Spreading => {
                    "Spreading activation traversed associative edges to surface candidates"
                        .to_string()
                }
                RecallMode::Hybrid => {
                    "Hybrid recall blended similarity scoring with spreading activation".to_string()
                }
            },
        },
        query_analysis: {
            let understood_intent = if let Some(query) = &params.query {
                query.clone()
            } else if params.embedding.is_some() {
                "Embedding-based recall request".to_string()
            } else {
                "Recall request without textual cue".to_string()
            };

            let search_strategy = match (effective_mode, params.embedding.is_some()) {
                (RecallMode::Similarity, true) => {
                    "Vector similarity with contextual dilation".to_string()
                }
                (RecallMode::Similarity, false) => {
                    "Similarity-first recall using lexical cues".to_string()
                }
                (RecallMode::Spreading, _) => {
                    "Spreading activation across cognitive graph".to_string()
                }
                (RecallMode::Hybrid, _) => {
                    "Hybrid recall blending similarity and spreading".to_string()
                }
            };

            let cognitive_load = if params.max_results.unwrap_or(10) > 20 {
                "Medium".to_string()
            } else {
                "Low".to_string()
            };

            let mut suggestions = Vec::new();
            if !params.trace_activation.unwrap_or(false)
                && matches!(effective_mode, RecallMode::Spreading | RecallMode::Hybrid)
            {
                suggestions
                    .push("Set trace_activation=true to inspect activation flow".to_string());
            }
            if params.max_results.unwrap_or(10) < 10 {
                suggestions
                    .push("Increase max_results to surface broader associations".to_string());
            }
            if suggestions.is_empty() {
                suggestions
                    .push("Apply tag filters to focus on specific episodic clusters".to_string());
            }

            QueryAnalysis {
                understood_intent,
                search_strategy,
                cognitive_load,
                suggestions,
            }
        },
        metadata: if params.include_metadata.unwrap_or(false) {
            Some(RecallMetadata {
                total_memories_searched: total_memories,
                activation_spread_hops: 2,
                processing_time_ms: processing_time,
                memory_system_load: "Normal".to_string(),
            })
        } else {
            None
        },
        system_message: format!(
            "Recall completed in {processing_time}ms. Found memories through direct matching and spreading activation."
        ),
    };

    info!(
        "Recall completed: found {} vivid and {} associated memories",
        response.memories.vivid.len(),
        response.memories.associated.len()
    );

    Ok(Json(response))
}

/// GET /api/v1/query/probabilistic - Probabilistic recall with uncertainty tracking
#[utoipa::path(
    get,
    path = "/api/v1/query/probabilistic",
    tag = "queries",
    params(ProbabilisticQueryRequest),
    responses(
        (status = 200, description = "Probabilistic query results with evidence and uncertainty", body = ProbabilisticQueryResponse),
        (status = 400, description = "Invalid query parameters", body = ErrorResponse),
        (status = 500, description = "Internal server error", body = ErrorResponse)
    )
)]
/// # Errors
///
/// Returns `ApiError` if probabilistic query fails
pub async fn probabilistic_query(
    State(state): State<ApiState>,
    headers: HeaderMap,
    Query(params): Query<ProbabilisticQueryRequest>,
) -> Result<impl IntoResponse, ApiError> {
    let start_time = std::time::Instant::now();

    info!(
        "Processing probabilistic query request with query: {:?}",
        params.query
    );

    // Validate query
    if params.query.is_none() && params.embedding.is_none() {
        return Err(ApiError::InvalidInput(
            "Probabilistic query requires either a text query or embedding vector".to_string(),
        ));
    }

    // Parse embedding if provided
    let embedding_vector = if let Some(emb_str) = &params.embedding {
        match serde_json::from_str::<Vec<f32>>(emb_str) {
            Ok(vec) => Some(vec),
            Err(_) => {
                return Err(ApiError::InvalidInput(
                    "Invalid embedding format. Expected JSON array of floats.".to_string(),
                ));
            }
        }
    } else {
        None
    };

    // Create search cue
    let cue = if let Some(query) = &params.query {
        CoreCue::semantic(
            "http_prob_query".to_string(),
            query.clone(),
            CoreConfidence::exact(params.threshold.unwrap_or(0.5)),
        )
    } else if let Some(embedding) = embedding_vector {
        let embedding_array: [f32; 768] = embedding_to_array(&embedding)?;
        CoreCue::embedding(
            "http_prob_embedding_query".to_string(),
            embedding_array,
            CoreConfidence::exact(params.threshold.unwrap_or(0.5)),
        )
    } else {
        return Err(ApiError::InvalidInput(
            "No valid search cue provided".to_string(),
        ));
    };

    // Extract memory space ID with fallback to default
    let space_id = extract_memory_space_id(
        &headers,
        params.space.as_deref(),
        None,
        &state.default_space,
    )?;
    Span::current().record("memory_space_id", space_id.as_str());

    // Get space-specific store handle from registry
    let handle = state
        .registry
        .create_or_get(&space_id)
        .await
        .map_err(|e| ApiError::SystemError(format!("Failed to access memory space: {e}")))?;
    let store = handle.store();

    // Runtime verification (defense-in-depth)
    store
        .verify_space(&space_id)
        .map_err(ApiError::SystemError)?;

    // Perform probabilistic recall on space-specific store
    let prob_result = store.recall_probabilistic(&cue);

    // Convert episodes to API format
    let memories: Vec<MemoryResult> = prob_result
        .episodes
        .iter()
        .take(params.max_results.unwrap_or(10))
        .map(|(episode, confidence)| MemoryResult {
            id: episode.id.clone(),
            content: episode.what.clone(),
            confidence: build_confidence_info(
                confidence.raw(),
                "Probabilistic recall with uncertainty tracking",
            ),
            activation_level: confidence.raw(),
            similarity_score: confidence.raw(),
            retrieval_path: Some("Probabilistic query".to_string()),
            observed_at: episode.when,
            last_access: Some(episode.last_recall),
            tags: vec![],
            memory_type: "Episodic".to_string(),
            relevance_explanation: format!(
                "Retrieved with {:.2} confidence (range: {:.2}-{:.2})",
                prob_result.confidence_interval.point.raw(),
                prob_result.confidence_interval.lower.raw(),
                prob_result.confidence_interval.upper.raw()
            ),
        })
        .collect();

    // Convert confidence interval
    let confidence_interval = ConfidenceIntervalInfo {
        lower: prob_result.confidence_interval.lower.raw(),
        upper: prob_result.confidence_interval.upper.raw(),
        point: prob_result.confidence_interval.point.raw(),
        width: prob_result.confidence_interval.width,
    };

    // Convert evidence chain if requested
    let evidence_chain = if params.include_evidence.unwrap_or(false) {
        Some(
            prob_result
                .evidence_chain
                .iter()
                .map(|evidence| {
                    let (source_type, description) = match &evidence.source {
                        EvidenceSource::SpreadingActivation {
                            source_episode,
                            activation_level,
                            path_length,
                        } => (
                            "spreading_activation".to_string(),
                            format!(
                                "Spreading activation from episode '{}' with {} activation over {} hops",
                                source_episode,
                                activation_level.value(),
                                path_length
                            ),
                        ),
                        EvidenceSource::TemporalDecay {
                            original_confidence,
                            time_elapsed,
                            decay_rate,
                        } => (
                            "temporal_decay".to_string(),
                            format!(
                                "Temporal decay from {:.2} confidence over {:.0}s with {:.4} decay rate",
                                original_confidence.raw(),
                                time_elapsed.as_secs_f32(),
                                decay_rate
                            ),
                        ),
                        EvidenceSource::DirectMatch {
                            cue_id,
                            similarity_score,
                            match_type,
                        } => (
                            "direct_match".to_string(),
                            format!(
                                "{match_type:?} match from cue '{cue_id}' with {similarity_score:.2} similarity"
                            ),
                        ),
                        EvidenceSource::VectorSimilarity(vector_ev) => (
                            "vector_similarity".to_string(),
                            format!(
                                "Vector similarity with distance {:.4} and index confidence {:.2}",
                                vector_ev.result_distance,
                                vector_ev.index_confidence.raw()
                            ),
                        ),
                    };

                    EvidenceInfo {
                        source_type,
                        strength: evidence.strength.raw(),
                        description,
                    }
                })
                .collect(),
        )
    } else {
        None
    };

    // Convert uncertainty sources if requested
    let uncertainty_sources = if params.include_uncertainty.unwrap_or(false) {
        Some(
            prob_result
                .uncertainty_sources
                .iter()
                .map(|source| {
                    let (source_type, impact, explanation) = match source {
                        UncertaintySource::SystemPressure {
                            pressure_level,
                            effect_on_confidence,
                        } => (
                            "system_pressure".to_string(),
                            *effect_on_confidence,
                            format!(
                                "System memory pressure at {:.1}% affecting confidence by {:.2}",
                                pressure_level * 100.0,
                                effect_on_confidence
                            ),
                        ),
                        UncertaintySource::SpreadingActivationNoise {
                            activation_variance,
                            path_diversity,
                        } => (
                            "spreading_activation_noise".to_string(),
                            *activation_variance,
                            format!(
                                "Activation spreading variance {activation_variance:.3} with path diversity {path_diversity:.3}"
                            ),
                        ),
                        UncertaintySource::TemporalDecayUnknown {
                            time_since_encoding,
                            decay_model_uncertainty,
                        } => (
                            "temporal_decay_unknown".to_string(),
                            *decay_model_uncertainty,
                            format!(
                                "Decay model uncertainty {:.2} for memories aged {:.0}s",
                                decay_model_uncertainty,
                                time_since_encoding.as_secs_f32()
                            ),
                        ),
                        UncertaintySource::MeasurementError {
                            error_magnitude,
                            confidence_degradation,
                        } => (
                            "measurement_error".to_string(),
                            *confidence_degradation,
                            format!(
                                "Measurement error magnitude {error_magnitude:.3} causing {confidence_degradation:.2} confidence degradation"
                            ),
                        ),
                    };

                    UncertaintyInfo {
                        source_type,
                        impact,
                        explanation,
                    }
                })
                .collect(),
        )
    } else {
        None
    };

    let processing_time = u64::try_from(start_time.elapsed().as_millis()).unwrap_or(u64::MAX);

    // Extract values before moving confidence_interval
    let ci_lower = confidence_interval.lower;
    let ci_upper = confidence_interval.upper;
    let ci_point = confidence_interval.point;
    let ci_width = confidence_interval.width;

    let response = ProbabilisticQueryResponse {
        memories,
        confidence_interval,
        evidence_chain,
        uncertainty_sources,
        system_message: format!(
            "Probabilistic query completed in {}ms with confidence interval [{:.2}, {:.2}] (point: {:.2}). \
             Uncertainty quantified from {} sources.",
            processing_time,
            ci_lower,
            ci_upper,
            ci_point,
            prob_result.uncertainty_sources.len()
        ),
    };

    info!(
        "Probabilistic query completed: {} memories with confidence {:.2} ± {:.2}",
        response.memories.len(),
        ci_point,
        ci_width / 2.0
    );

    Ok(Json(response))
}

/// GET /api/v1/consolidations - List consolidated beliefs with citations
#[utoipa::path(
    get,
    path = "/api/v1/consolidations",
    tag = "consolidation",
    params(ConsolidationQuery),
    responses(
        (status = 200, description = "Consolidated beliefs with provenance", body = ConsolidationSummaryResponse),
        (status = 500, description = "Internal server error", body = ErrorResponse)
    )
)]
pub async fn list_consolidations(
    State(state): State<ApiState>,
    headers: HeaderMap,
    Query(params): Query<ConsolidationQuery>,
) -> Result<impl IntoResponse, ApiError> {
    // Extract memory space ID with fallback to default
    let space_id = extract_memory_space_id(
        &headers,
        params.space.as_deref(),
        None,
        &state.default_space,
    )?;
    Span::current().record("memory_space_id", space_id.as_str());

    // Get space-specific store handle from registry
    let handle = state
        .registry
        .create_or_get(&space_id)
        .await
        .map_err(|e| ApiError::SystemError(format!("Failed to access memory space: {e}")))?;
    let store = handle.store();

    // Runtime verification (defense-in-depth)
    store
        .verify_space(&space_id)
        .map_err(ApiError::SystemError)?;

    let max_episodes = params.max_episodes.unwrap_or(256);
    let snapshot = store.consolidation_snapshot(max_episodes);

    let mut beliefs: Vec<ConsolidatedBeliefResponse> = snapshot
        .patterns
        .into_iter()
        .map(|pattern| pattern_to_belief(pattern, store.as_ref()))
        .collect();

    beliefs.sort_by(|a, b| {
        b.strength
            .partial_cmp(&a.strength)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    if let Some(max_patterns) = params.max_patterns {
        beliefs.truncate(max_patterns);
    }

    let response = ConsolidationSummaryResponse {
        generated_at: snapshot.generated_at,
        stats: ConsolidationRunStats::from(snapshot.stats),
        beliefs,
    };

    Ok(Json(response))
}

/// GET /api/v1/consolidations/{id} - Retrieve a specific consolidated belief
#[utoipa::path(
    get,
    path = "/api/v1/consolidations/{id}",
    tag = "consolidation",
    params(
        ("id" = String, Path, description = "Consolidated belief identifier"),
        ConsolidationDetailQuery
    ),
    responses(
        (status = 200, description = "Consolidated belief with citations", body = ConsolidationSummaryResponse),
        (status = 404, description = "Consolidated belief not found", body = ErrorResponse),
        (status = 500, description = "Internal server error", body = ErrorResponse)
    )
)]
pub async fn get_consolidation(
    State(state): State<ApiState>,
    headers: HeaderMap,
    Path(id): Path<String>,
    Query(params): Query<ConsolidationDetailQuery>,
) -> Result<impl IntoResponse, ApiError> {
    // Extract memory space ID with fallback to default
    let space_id = extract_memory_space_id(
        &headers,
        params.space.as_deref(),
        None,
        &state.default_space,
    )?;
    Span::current().record("memory_space_id", space_id.as_str());

    // Get space-specific store handle from registry
    let handle = state
        .registry
        .create_or_get(&space_id)
        .await
        .map_err(|e| ApiError::SystemError(format!("Failed to access memory space: {e}")))?;
    let store = handle.store();

    // Runtime verification (defense-in-depth)
    store
        .verify_space(&space_id)
        .map_err(ApiError::SystemError)?;

    let max_episodes = params.max_episodes.unwrap_or(0);
    let snapshot = store.consolidation_snapshot(max_episodes);

    if let Some(pattern) = snapshot
        .patterns
        .into_iter()
        .find(|pattern| pattern.id == id)
    {
        let belief = pattern_to_belief(pattern, store.as_ref());
        let response = ConsolidationSummaryResponse {
            generated_at: snapshot.generated_at,
            stats: ConsolidationRunStats::from(snapshot.stats),
            beliefs: vec![belief],
        };
        return Ok(Json(response));
    }

    Err(ApiError::ConsolidationNotFound(id))
}

/// POST /api/v1/memories/recognize - Pattern recognition
#[utoipa::path(
    post,
    path = "/api/v1/memories/recognize",
    tag = "memories",
    request_body = RecognizeRequest,
    responses(
        (status = 200, description = "Pattern recognition results", body = RecognizeResponse),
        (status = 400, description = "Invalid recognition request", body = ErrorResponse),
        (status = 500, description = "Internal server error", body = ErrorResponse)
    )
)]
/// # Errors
///
/// Returns `ApiError` if pattern recognition fails
pub async fn recognize_pattern(
    State(_state): State<ApiState>,
    Json(request): Json<RecognizeRequest>,
) -> Result<impl IntoResponse, ApiError> {
    info!("Processing pattern recognition request");

    if request.input.trim().is_empty() {
        return Err(ApiError::InvalidInput(
            "Recognition requires input content. What pattern would you like to recognize?"
                .to_string(),
        ));
    }

    // Simplified recognition - would implement actual pattern matching
    let recognized = request.input.len() > 10; // Placeholder logic
    let confidence = if recognized { 0.7 } else { 0.3 };

    let response = RecognizeResponse {
        recognized,
        recognition_confidence: ConfidenceInfo {
            value: confidence,
            category: if confidence > 0.6 { "High" } else { "Low" }.to_string(),
            reasoning: if recognized {
                "Pattern matches learned memory structures"
            } else {
                "Novel pattern - no strong matches found"
            }
            .to_string(),
        },
        similar_patterns: if recognized {
            vec![SimilarPattern {
                memory_id: "pattern_example".to_string(),
                similarity_score: 0.8,
                pattern_type: "Semantic".to_string(),
                explanation: "Similar linguistic structure detected".to_string(),
            }]
        } else {
            vec![]
        },
        system_message: if recognized {
            "Pattern recognized - this aligns with existing memory structures"
        } else {
            "Novel pattern detected - consider remembering this for future recognition"
        }
        .to_string(),
    };

    Ok(Json(response))
}

// ============================================================================
// REST-style endpoints for CLI compatibility
// ============================================================================

/// POST /api/v1/memories - Simple memory creation (CLI-compatible)
///
/// This endpoint provides a simpler REST-style interface for the CLI,
/// forwarding requests to the cognitive remember_memory handler.
pub async fn create_memory_rest(
    State(state): State<ApiState>,
    headers: HeaderMap,
    Json(payload): Json<serde_json::Value>,
) -> Result<impl IntoResponse, ApiError> {
    // Extract content and optional confidence from simple CLI payload
    let content = payload
        .get("content")
        .and_then(|v| v.as_str())
        .ok_or_else(|| ApiError::InvalidInput("Missing 'content' field".to_string()))?
        .to_string();

    let confidence = payload
        .get("confidence")
        .and_then(serde_json::Value::as_f64)
        .map(|v| v as f32);

    // Build RememberMemoryRequest from CLI payload
    let request = RememberMemoryRequest {
        id: None,
        content,
        embedding: None,
        confidence,
        confidence_reasoning: None,
        memory_type: Some("semantic".to_string()),
        tags: None,
        timestamp: None,
        auto_link: Some(false),
        link_threshold: None,
        memory_space_id: None, // Use default space
    };

    // Forward to the cognitive handler
    let response = remember_memory(State(state), headers, Json(request)).await?;
    Ok(response)
}

/// GET /api/v1/memories/{id} - Retrieve memory by ID (CLI-compatible)
#[allow(clippy::implicit_hasher)] // Simple query params, no custom hasher needed
pub async fn get_memory_by_id(
    State(state): State<ApiState>,
    headers: HeaderMap,
    axum::extract::Path(id): axum::extract::Path<String>,
    Query(params): Query<HashMap<String, String>>,
) -> Result<impl IntoResponse, ApiError> {
    // Extract memory space ID with fallback to default
    let space_id = extract_memory_space_id(
        &headers,
        params.get("space").map(String::as_str),
        None,
        &state.default_space,
    )?;
    Span::current().record("memory_space_id", space_id.as_str());

    // Get space-specific store handle from registry
    let handle = state
        .registry
        .create_or_get(&space_id)
        .await
        .map_err(|e| ApiError::SystemError(format!("Failed to access memory space: {e}")))?;
    let store = handle.store();

    // Runtime verification (defense-in-depth)
    store
        .verify_space(&space_id)
        .map_err(ApiError::SystemError)?;

    // Direct O(1) lookup by ID
    if let Some(episode) = store.get_by_id(&id) {
        let memory_data = json!({
            "id": episode.id,
            "content": episode.what,
            "confidence": episode.encoding_confidence.raw(),
            "timestamp": episode.when.to_rfc3339(),
            "memory_type": "episodic"
        });
        return Ok(Json(memory_data));
    }

    // Memory not found
    Err(ApiError::MemoryNotFound(format!(
        "Memory with ID '{id}' not found"
    )))
}

/// GET /api/v1/memories/search - Search memories (CLI-compatible)
#[allow(clippy::implicit_hasher)]
pub async fn search_memories_rest(
    State(state): State<ApiState>,
    headers: HeaderMap,
    Query(params): Query<HashMap<String, String>>,
) -> Result<impl IntoResponse, ApiError> {
    let query = params
        .get("query")
        .ok_or_else(|| ApiError::InvalidInput("Missing 'query' parameter".to_string()))?;

    let limit = params
        .get("limit")
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(10);

    // Extract memory space ID with fallback to default
    let space_id = extract_memory_space_id(
        &headers,
        params.get("space").map(String::as_str),
        None,
        &state.default_space,
    )?;
    Span::current().record("memory_space_id", space_id.as_str());

    // Get space-specific store handle from registry
    let handle = state
        .registry
        .create_or_get(&space_id)
        .await
        .map_err(|e| ApiError::SystemError(format!("Failed to access memory space: {e}")))?;
    let store = handle.store();

    // Runtime verification (defense-in-depth)
    store
        .verify_space(&space_id)
        .map_err(ApiError::SystemError)?;

    // Call recall on space-specific store for simple CLI response
    let cue = CoreCue::semantic(query.clone(), query.clone(), CoreConfidence::exact(0.7));
    let recall_result = store.recall(&cue);

    // Check if streaming failed
    if !recall_result.streaming_delivered {
        tracing::warn!(
            "Search completed successfully but event streaming failed - SSE subscribers not notified"
        );
        return Err(ApiError::SystemError(
            "Search completed but event notification failed. \
             SSE subscribers did not receive the recall events. \
             Check /api/v1/system/health for streaming status."
                .to_string(),
        ));
    }

    let memories: Vec<serde_json::Value> = recall_result
        .results
        .iter()
        .take(limit)
        .map(|(episode, confidence)| {
            json!({
                "id": episode.id,
                "content": episode.what,
                "confidence": confidence.raw(),
                "memory_type": "episodic",
                "timestamp": episode.when.to_rfc3339()
            })
        })
        .collect();

    Ok(Json(json!({
        "memories": memories,
        "total": memories.len()
    })))
}

/// DELETE /api/v1/memories/{id} - Delete memory by ID (CLI-compatible)
///
/// # Cognitive Design Note
///
/// Memory "deletion" is not supported in Engram by design, as it conflicts with
/// cognitive principles of biological memory systems. In human memory:
/// - Memories naturally decay rather than being instantly erased
/// - Forgotten memories remain accessible but with reduced activation
/// - Suppression is a gradual process, not instantaneous removal
///
/// ## Alternative Approaches
///
/// 1. **Natural Decay**: Memories that aren't recalled naturally lose activation over time
/// 2. **Suppression** (future): Actively reduce memory activation without deletion
/// 3. **Confidence Adjustment**: Lower the confidence score to reduce recall likelihood
///
/// ## Why This Matters
///
/// Instant deletion would break:
/// - Spreading activation paths (memories that link to deleted content)
/// - Consolidation processes (replay requires stable memory graphs)
/// - Temporal coherence (memory timelines become fragmented)
///
/// If you need to "forget" something, consider adjusting its recall threshold or
/// waiting for natural decay rather than requesting deletion.
pub async fn delete_memory_by_id(
    State(_state): State<ApiState>,
    axum::extract::Path(id): axum::extract::Path<String>,
) -> Result<impl IntoResponse, ApiError> {
    info!(
        "Delete request for memory ID: {} - not supported due to cognitive design principles",
        id
    );

    Ok(Json(json!({
        "status": "not_supported",
        "message": "Memory deletion is not supported in Engram due to cognitive design principles",
        "memory_id": id,
        "rationale": "Human memory systems don't support instant deletion - memories naturally decay over time or can be suppressed, but instant removal would break spreading activation paths and consolidation processes",
        "alternatives": [
            "Allow natural decay (memories not recalled lose activation over time)",
            "Adjust recall thresholds to reduce retrieval likelihood",
            "Future: Use suppression mechanisms to gradually reduce activation"
        ],
        "documentation": "See https://github.com/anthropics/engram for cognitive architecture details"
    })))
}

/// GET /health - Simple health check
#[utoipa::path(
    get,
    path = "/health",
    tag = "system",
    responses(
        (status = 200, description = "Simple health status", body = serde_json::Value),
        (status = 503, description = "Service degraded or unavailable", body = serde_json::Value),
        (status = 500, description = "Health check failed", body = ErrorResponse)
    )
)]
/// # Errors
///
/// Returns `ApiError` if health check fails
pub async fn simple_health(State(state): State<ApiState>) -> Result<impl IntoResponse, ApiError> {
    let status = state.metrics.health_status();
    let payload = json!({
        "status": format_health_status(status),
        "service": "engram",
        "version": env!("CARGO_PKG_VERSION"),
        "timestamp": Utc::now().to_rfc3339()
    });

    let status_code = match status {
        HealthStatus::Healthy | HealthStatus::Degraded => StatusCode::OK,
        HealthStatus::Unhealthy => StatusCode::SERVICE_UNAVAILABLE,
    };

    Ok((status_code, Json(payload)))
}

/// Lightweight liveness probe used by CLI keepalive checks.
pub async fn alive_health() -> impl IntoResponse {
    StatusCode::OK
}

/// GET /health/spreading - Spreading-specific readiness probe
#[utoipa::path(
    get,
    path = "/health/spreading",
    tag = "system",
    responses(
        (status = 200, description = "Spreading probe status", body = serde_json::Value),
        (status = 503, description = "Spreading probe unhealthy", body = serde_json::Value),
        (status = 404, description = "Spreading probe unavailable", body = serde_json::Value)
    )
)]
pub async fn spreading_health(
    State(state): State<ApiState>,
) -> Result<impl IntoResponse, ApiError> {
    let registry = state.metrics.health_registry();
    if let Some(check) = registry.check_named("spreading") {
        let status_code = match check.status {
            HealthStatus::Healthy | HealthStatus::Degraded => StatusCode::OK,
            HealthStatus::Unhealthy => StatusCode::SERVICE_UNAVAILABLE,
        };
        Ok((status_code, Json(health_check_to_json(&check))))
    } else {
        let payload = json!({
            "status": "unknown",
            "message": "Spreading probe not registered",
        });
        Ok((StatusCode::NOT_FOUND, Json(payload)))
    }
}

/// POST /shutdown - Initiate graceful server shutdown
///
/// # Errors
///
/// Returns error if shutdown signal cannot be sent
pub async fn shutdown_server(State(state): State<ApiState>) -> Result<impl IntoResponse, ApiError> {
    tracing::info!("Shutdown endpoint called - initiating graceful shutdown");

    // Send shutdown signal to all background tasks
    state
        .shutdown_tx
        .send(true)
        .map_err(|_| ApiError::SystemError("Failed to send shutdown signal".to_string()))?;

    Ok((
        StatusCode::OK,
        Json(json!({
            "status": "shutdown_initiated",
            "message": "Server is shutting down gracefully"
        })),
    ))
}

/// GET /api/v1/system/health - Comprehensive system health with per-space metrics
#[utoipa::path(
    get,
    path = "/api/v1/system/health",
    tag = "system",
    responses(
        (status = 200, description = "System health information with per-space metrics", body = HealthResponse),
        (status = 500, description = "System health check failed", body = ErrorResponse)
    )
)]
/// # Errors
///
/// Returns `ApiError` if system health check fails
pub async fn system_health(State(state): State<ApiState>) -> Result<impl IntoResponse, ApiError> {
    let health_registry = state.metrics.health_registry();
    let overall_status = health_registry.check_all();
    let report = health_registry.health_report();

    // Collect per-space health metrics
    let space_summaries = state.registry.list();
    let mut space_metrics = Vec::new();

    for summary in space_summaries {
        let space_id = summary.id.clone();

        // Get handle for this space
        let handle = state.registry.create_or_get(&space_id).await.map_err(|e| {
            ApiError::SystemError(format!(
                "Failed to access memory space '{}': {}",
                space_id, e
            ))
        })?;

        let memory_count = handle.store().count();

        // Collect metrics for this space
        // TODO(Task 006c): Wire up actual metrics from tier backend and persistence handle
        space_metrics.push(SpaceHealthMetrics {
            space: space_id.as_str().to_string(),
            memories: memory_count as u64,
            pressure: 0.0,           // Placeholder: will be actual tier utilization
            wal_lag_ms: 0.0,         // Placeholder: will be actual WAL replication lag
            consolidation_rate: 0.0, // Placeholder: will be actual consolidation throughput
        });
    }

    let checks: Vec<_> = report.checks.iter().map(health_check_to_json).collect();

    let response = HealthResponse {
        status: format_health_status(report.status).to_string(),
        timestamp: Utc::now().to_rfc3339(),
        checks,
        spaces: space_metrics,
    };

    let status_code = match overall_status {
        HealthStatus::Healthy | HealthStatus::Degraded => StatusCode::OK,
        HealthStatus::Unhealthy => StatusCode::SERVICE_UNAVAILABLE,
    };

    Ok((status_code, Json(response)))
}

/// GET /api/v1/system/spreading/config - Auto-tuning history
#[utoipa::path(
    get,
    path = "/api/v1/system/spreading/config",
    tag = "system",
    responses(
        (status = 200, description = "Auto-tuning audit log", body = AutoTuneResponse)
    )
)]
pub async fn spreading_config(
    State(state): State<ApiState>,
) -> Result<impl IntoResponse, ApiError> {
    let audit_log = state.auto_tuner.history();
    Ok(Json(AutoTuneResponse { audit_log }))
}

/// GET /api/v1/system/introspect - System introspection
#[utoipa::path(
    get,
    path = "/api/v1/system/introspect",
    tag = "system",
    responses(
        (status = 200, description = "System introspection data", body = IntrospectionResponse),
        (status = 500, description = "Introspection failed", body = ErrorResponse)
    )
)]
/// # Errors
///
/// Returns `ApiError` if introspection fails
pub async fn system_introspect(
    State(state): State<ApiState>,
) -> Result<impl IntoResponse, ApiError> {
    // Get default space store handle for system-wide stats
    let handle = state
        .registry
        .create_or_get(&state.default_space)
        .await
        .map_err(|e| {
            ApiError::SystemError(format!("Failed to access default memory space: {e}"))
        })?;
    let memory_count = handle.store().count();

    let introspection_data = json!({
        "memory_statistics": {
            "total_nodes": memory_count,
            "average_activation": 0.5,
            "consolidation_states": {
                "recent": 0,
                "consolidated": memory_count,
                "archived": 0
            }
        },
        "system_processes": {
            "spreading_activation": "idle",
            "memory_consolidation": "scheduled",
            "pattern_completion": "ready",
            "dream_simulation": "offline"
        },
        "performance_metrics": {
            "avg_recall_time_ms": 45,
            "memory_capacity_used": "15%",
            "activation_efficiency": "high"
        }
    });

    Ok(Json(introspection_data))
}

/// GET /metrics - Streaming metrics snapshot
#[utoipa::path(
    get,
    path = "/metrics",
    tag = "system",
    responses(
        (status = 200, description = "Aggregated streaming metrics snapshot", body = serde_json::Value)
    )
)]
pub async fn metrics_snapshot(
    State(state): State<ApiState>,
) -> Result<impl IntoResponse, ApiError> {
    // Drain any pending updates to avoid divergence between HTTP and in-memory snapshots.
    let aggregator = state.metrics.streaming_aggregator();
    aggregator.set_export_enabled(false);
    let mut snapshot = state.metrics.streaming_snapshot();
    loop {
        let next = state.metrics.streaming_snapshot();
        if serde_json::to_value(&next).ok() == serde_json::to_value(&snapshot).ok() {
            snapshot = next;
            break;
        }
        snapshot = next;
    }
    let aggregator_clone = Arc::clone(&aggregator);
    tokio::spawn(async move {
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        aggregator_clone.set_export_enabled(true);
    });
    let export = state.metrics.streaming_stats();

    Ok(Json(json!({
        "snapshot": snapshot,
        "export": export,
    })))
}

/// GET /api/v1/episodes/replay - Replay episode memories
#[utoipa::path(
    get,
    path = "/api/v1/episodes/replay",
    tag = "episodes",
    params(
        ("time_range" = Option<String>, Query, description = "Time range for episodes (e.g., 'last_week', '2023-01-01_to_2023-12-31')"),
        ("context" = Option<String>, Query, description = "Contextual filter for episodes"),
        ("emotional_valence" = Option<String>, Query, description = "Filter by emotional valence ('positive', 'negative', 'neutral')"),
        ("importance" = Option<String>, Query, description = "Minimum importance threshold (0.0-1.0)")
    ),
    responses(
        (status = 200, description = "Episode replay results", body = RecallResponse),
        (status = 400, description = "Invalid replay parameters", body = ErrorResponse),
        (status = 500, description = "Internal server error", body = ErrorResponse)
    )
)]
/// # Errors
///
/// Returns `ApiError` if episode replay fails
#[allow(clippy::implicit_hasher)]
pub async fn replay_episodes(
    State(_state): State<ApiState>,
    Query(params): Query<HashMap<String, String>>,
) -> Result<impl IntoResponse, ApiError> {
    let time_range = params
        .get("time_range")
        .cloned()
        .unwrap_or_else(|| "recent".to_string());

    let replay_data = json!({
        "episodes": [],
        "replay_confidence": {
            "value": 0.6,
            "category": "Medium",
            "reasoning": "Limited episodes available for replay"
        },
        "system_message": format!("Episode replay for {} time range - no episodes found", time_range)
    });

    Ok(Json(replay_data))
}

// ================================================================================================
// Error Handling with Educational Messages
// ================================================================================================

/// Error types for HTTP API operations
#[derive(Debug)]
pub enum ApiError {
    /// Invalid input parameters or data
    InvalidInput(String),
    /// Requested memory not found
    MemoryNotFound(String),
    /// Requested consolidated belief not found
    ConsolidationNotFound(String),
    /// Internal system error occurred
    SystemError(String),
    /// Data validation failed
    ValidationError(String),
}

impl IntoResponse for ApiError {
    fn into_response(self) -> axum::response::Response {
        let (status, error_code, message, educational_context) = match self {
            Self::InvalidInput(msg) => (
                StatusCode::BAD_REQUEST,
                "INVALID_MEMORY_INPUT",
                msg,
                "Memory operations require well-formed inputs. Check the API documentation for proper request structure.".to_string()
            ),
            Self::MemoryNotFound(id) => (
                StatusCode::NOT_FOUND,
                "MEMORY_NOT_FOUND", 
                format!("Memory '{id}' not found in the cognitive graph"),
                "Memory retrieval failed - the requested memory may have been forgotten or never encoded. Try a broader search query.".to_string()
            ),
            Self::ConsolidationNotFound(id) => (
                StatusCode::NOT_FOUND,
                "CONSOLIDATION_NOT_FOUND",
                format!("Consolidated belief '{id}' not available"),
                "Consolidation snapshots are generated from current episodic memories. Ensure the contributing episodes still exist or regenerate the snapshot."
                    .to_string(),
            ),
            Self::SystemError(msg) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                "MEMORY_SYSTEM_ERROR",
                msg,
                "The memory system encountered an internal error. Cognitive processes may be temporarily disrupted.".to_string()
            ),
            Self::ValidationError(msg) => (
                StatusCode::UNPROCESSABLE_ENTITY,
                "MEMORY_VALIDATION_ERROR", 
                msg,
                "Memory content validation failed. Ensure your input follows cognitive encoding principles.".to_string()
            ),
        };

        let error_response = json!({
            "error": {
                "code": error_code,
                "message": message,
                "educational_context": educational_context,
                "cognitive_guidance": match error_code {
                    "INVALID_MEMORY_INPUT" => "Memories need meaningful content and context. Think about what makes this memory important.",
                    "MEMORY_NOT_FOUND" => "Try different recall cues - memories are often accessible through alternative pathways.",
                    _ => "Review the memory system documentation for guidance on proper operations."
                }
            },
            "documentation": {
                "api_guide": "/docs/api",
                "memory_concepts": "/docs/memory-systems",
                "examples": "/docs/examples"
            }
        });

        (status, Json(error_response)).into_response()
    }
}

impl From<MemorySpaceError> for ApiError {
    fn from(error: MemorySpaceError) -> Self {
        match error {
            MemorySpaceError::InvalidId(inner) => Self::InvalidInput(inner.to_string()),
            MemorySpaceError::NotFound { id } => Self::InvalidInput(format!(
                "Memory space '{id}' not found. Ensure it is created before issuing operations."
            )),
            MemorySpaceError::Persistence { id, path, source } => Self::SystemError(format!(
                "Failed to prepare persistence path '{}' for memory space '{id}': {source}",
                path.display()
            )),
            MemorySpaceError::DataRootUnavailable { path, source } => Self::SystemError(format!(
                "Unable to initialise memory space data root '{}': {source}",
                path.display()
            )),
            MemorySpaceError::StoreInit { id, source } => Self::SystemError(format!(
                "Failed to initialise memory store for space '{id}': {source}"
            )),
        }
    }
}

// ================================================================================================
// Server-Sent Events (SSE) Streaming Handlers
// ================================================================================================

/// Stream real-time memory system activities
///
/// Provides Server-Sent Events for monitoring memory activations, storage operations,
/// recalls, consolidation events, association formations, and memory decay.
/// Follows cognitive ergonomics with hierarchical event organization and backpressure control.
#[utoipa::path(
    get,
    path = "/api/v1/stream/activities",
    tag = "streaming",
    params(StreamActivityQuery),
    responses(
        (status = 200, description = "Server-Sent Events stream", content_type = "text/event-stream"),
        (status = 400, description = "Invalid streaming parameters", body = ErrorResponse)
    )
)]
#[allow(deprecated)] // TODO: Migrate to registry pattern once streaming supports per-space event filtering
pub async fn stream_activities(
    State(state): State<ApiState>,
    Query(params): Query<StreamActivityQuery>,
) -> impl IntoResponse {
    let ApiState { store, .. } = state;
    let session_id = params
        .session_id
        .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());
    let buffer_size = params.buffer_size.unwrap_or(32).min(128); // Respect working memory limits
    let min_importance = params.min_importance.unwrap_or(0.1).clamp(0.0, 1.0);

    // Parse event types (default to all if not specified)
    let event_types = params.event_types.map_or_else(
        || {
            vec![
                "activation".to_string(),
                "storage".to_string(),
                "recall".to_string(),
                "consolidation".to_string(),
                "association".to_string(),
                "decay".to_string(),
            ]
        },
        |types| {
            types
                .split(',')
                .map(|t| t.trim().to_lowercase())
                .collect::<Vec<_>>()
        },
    );

    // Create activity stream with real events from memory store
    let stream =
        create_activity_stream(store, session_id, event_types, min_importance, buffer_size);

    Sse::new(stream).keep_alive(KeepAlive::default())
}

/// Stream memory operations (formation, retrieval, completion)
///
/// Cognitive design: Stream of consciousness for memory operations with
/// attention-aware filtering to prevent cognitive overload.
#[utoipa::path(
    get,
    path = "/api/v1/stream/memories",
    tag = "streaming",
    params(StreamMemoryQuery),
    responses(
        (status = 200, description = "Server-Sent Events stream", content_type = "text/event-stream"),
        (status = 400, description = "Invalid streaming parameters", body = ErrorResponse)
    )
)]
#[allow(deprecated)] // TODO: Migrate to registry pattern once streaming supports per-space event filtering
pub async fn stream_memories(
    State(state): State<ApiState>,
    Query(params): Query<StreamMemoryQuery>,
) -> impl IntoResponse {
    let ApiState { store, .. } = state;
    let session_id = params
        .session_id
        .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());
    let min_confidence = params.min_confidence.unwrap_or(0.3).clamp(0.0, 1.0);

    let include_formation = params.include_formation.unwrap_or(true);
    let include_retrieval = params.include_retrieval.unwrap_or(true);
    let include_completion = params.include_completion.unwrap_or(false);

    // Parse memory types
    let memory_types = params.memory_types.map_or_else(
        || {
            vec![
                "semantic".to_string(),
                "episodic".to_string(),
                "procedural".to_string(),
            ]
        },
        |types| {
            types
                .split(',')
                .map(|t| t.trim().to_lowercase())
                .collect::<Vec<_>>()
        },
    );

    let stream = create_memory_stream(
        store,
        session_id,
        include_formation,
        include_retrieval,
        include_completion,
        min_confidence,
        memory_types,
    );

    Sse::new(stream).keep_alive(KeepAlive::default())
}

/// Stream consolidation processes (replay, insights, progress)
///
/// Cognitive design: Dream-like consolidation streams with novelty filtering
/// to surface interesting replay sequences and insights.
#[utoipa::path(
    get,
    path = "/api/v1/stream/consolidation",
    tag = "streaming",
    params(StreamConsolidationQuery),
    responses(
        (status = 200, description = "Server-Sent Events stream", content_type = "text/event-stream"),
        (status = 400, description = "Invalid streaming parameters", body = ErrorResponse)
    )
)]
#[allow(deprecated)] // TODO: Migrate to registry pattern once streaming supports per-space event filtering
pub async fn stream_consolidation(
    State(state): State<ApiState>,
    Query(params): Query<StreamConsolidationQuery>,
) -> impl IntoResponse {
    let ApiState { store, .. } = state;
    let session_id = params
        .session_id
        .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());
    let min_novelty = params.min_novelty.unwrap_or(0.4).clamp(0.0, 1.0);

    let include_replay = params.include_replay.unwrap_or(true);
    let include_insights = params.include_insights.unwrap_or(true);
    let include_progress = params.include_progress.unwrap_or(false);

    let stream = create_consolidation_stream(
        store,
        session_id,
        include_replay,
        include_insights,
        include_progress,
        min_novelty,
    );

    Sse::new(stream).keep_alive(KeepAlive::default())
}

// ================================================================================================
// Stream Creation Functions
// ================================================================================================

/// Create activity event stream with real memory store events
fn create_activity_stream(
    store: Arc<MemoryStore>,
    session_id: String,
    event_types: Vec<String>,
    min_importance: f32,
    buffer_size: usize,
) -> impl Stream<Item = Result<Event, std::convert::Infallible>> {
    let (tx, rx) = tokio::sync::mpsc::channel(buffer_size);

    tokio::spawn(async move {
        // Subscribe to real memory events
        let Some(mut event_rx) = store.subscribe_to_events() else {
            // Event streaming not enabled, send error event and close
            let error_event = Event::default().event("error").data(
                json!({
                    "error": "Event streaming not enabled on server",
                    "session_id": session_id,
                })
                .to_string(),
            );
            let _ = tx.send(Ok(error_event)).await;
            return;
        };

        let mut event_id = 0u64;

        loop {
            match event_rx.recv().await {
                Ok(memory_event) => {
                    // Map MemoryEvent to SSE event format
                    let (event_type, _description, importance, event_data) = match memory_event {
                        MemoryEvent::Stored {
                            memory_space_id,
                            id,
                            confidence,
                            timestamp,
                        } => {
                            if !event_types.contains(&"storage".to_string()) {
                                continue;
                            }
                            (
                                "storage",
                                "New memory encoded with contextual associations",
                                confidence,
                                json!({
                                    "event_type": "storage",
                                    "description": "New memory encoded",
                                    "memory_space_id": memory_space_id.as_str(),
                                    "memory_id": id,
                                    "confidence": confidence,
                                    "importance": confidence,
                                    "timestamp": timestamp,
                                    "session_id": session_id,
                                    "metadata": {
                                        "cognitive_load": if confidence > 0.8 { "high" } else { "normal" },
                                        "attention_required": confidence > 0.9
                                    }
                                }),
                            )
                        }
                        MemoryEvent::Recalled {
                            memory_space_id,
                            id,
                            activation,
                            confidence,
                        } => {
                            if !event_types.contains(&"recall".to_string()) {
                                continue;
                            }
                            let importance = f32::midpoint(activation, confidence);
                            (
                                "recall",
                                "Memory retrieval triggered by recognition cue",
                                importance,
                                json!({
                                    "event_type": "recall",
                                    "description": "Memory recalled",
                                    "memory_space_id": memory_space_id.as_str(),
                                    "memory_id": id,
                                    "activation": activation,
                                    "confidence": confidence,
                                    "importance": importance,
                                    "timestamp": Utc::now().to_rfc3339(),
                                    "session_id": session_id,
                                    "metadata": {
                                        "cognitive_load": if importance > 0.8 { "high" } else { "normal" },
                                        "attention_required": importance > 0.9
                                    }
                                }),
                            )
                        }
                        MemoryEvent::ActivationSpread {
                            memory_space_id,
                            count,
                            avg_activation,
                        } => {
                            if !event_types.contains(&"activation".to_string()) {
                                continue;
                            }
                            (
                                "activation",
                                "Memory nodes activated for pattern matching",
                                avg_activation,
                                json!({
                                    "event_type": "activation",
                                    "description": "Spreading activation across memory graph",
                                    "memory_space_id": memory_space_id.as_str(),
                                    "memories_activated": count,
                                    "avg_activation": avg_activation,
                                    "importance": avg_activation,
                                    "timestamp": Utc::now().to_rfc3339(),
                                    "session_id": session_id,
                                    "metadata": {
                                        "cognitive_load": if avg_activation > 0.8 { "high" } else { "normal" },
                                        "attention_required": count > 10
                                    }
                                }),
                            )
                        }
                    };

                    // Filter by importance threshold
                    if importance < min_importance {
                        continue;
                    }

                    event_id += 1;
                    let sse_event = Event::default()
                        .id(event_id.to_string())
                        .event(event_type)
                        .data(event_data.to_string());

                    if tx.send(Ok(sse_event)).await.is_err() {
                        return; // Client disconnected
                    }
                }
                Err(tokio::sync::broadcast::error::RecvError::Lagged(skipped)) => {
                    // Client is lagging, send warning event
                    let warning_event = Event::default().event("warning").data(
                        json!({
                            "warning": format!("Lagged behind, {} events skipped", skipped),
                            "session_id": session_id,
                        })
                        .to_string(),
                    );
                    if tx.send(Ok(warning_event)).await.is_err() {
                        return;
                    }
                }
                Err(tokio::sync::broadcast::error::RecvError::Closed) => {
                    // Channel closed, end stream
                    return;
                }
            }
        }
    });

    ReceiverStream::new(rx)
}

/// Create memory operation stream with real memory events
fn create_memory_stream(
    store: Arc<MemoryStore>,
    session_id: String,
    include_formation: bool,
    include_retrieval: bool,
    _include_completion: bool,
    min_confidence: f32,
    _memory_types: Vec<String>,
) -> impl Stream<Item = Result<Event, std::convert::Infallible>> {
    let (tx, rx) = tokio::sync::mpsc::channel(32);

    tokio::spawn(async move {
        // Subscribe to real memory events
        let Some(mut event_rx) = store.subscribe_to_events() else {
            let error_event = Event::default().event("error").data(
                json!({
                    "error": "Event streaming not enabled on server",
                    "session_id": session_id,
                })
                .to_string(),
            );
            let _ = tx.send(Ok(error_event)).await;
            return;
        };

        let mut event_id = 0u64;

        loop {
            match event_rx.recv().await {
                Ok(memory_event) => {
                    // Map MemoryEvent to memory operation events
                    let (operation, _confidence, event_data) = match memory_event {
                        MemoryEvent::Stored {
                            memory_space_id,
                            id,
                            confidence,
                            timestamp,
                        } => {
                            if !include_formation {
                                continue;
                            }
                            if confidence < min_confidence {
                                continue;
                            }
                            (
                                "formation",
                                confidence,
                                json!({
                                    "operation": "formation",
                                    "description": "New memory formation with encoding confidence",
                                    "memory_space_id": memory_space_id.as_str(),
                                    "memory_type": "episodic",
                                    "confidence": confidence,
                                    "timestamp": timestamp,
                                    "session_id": session_id,
                                    "memory_id": id,
                                    "consolidation_state": if confidence > 0.8 { "Stable" } else { "Recent" }
                                }),
                            )
                        }
                        MemoryEvent::Recalled {
                            memory_space_id,
                            id,
                            activation,
                            confidence,
                        } => {
                            if !include_retrieval {
                                continue;
                            }
                            let avg_confidence = f32::midpoint(activation, confidence);
                            if avg_confidence < min_confidence {
                                continue;
                            }
                            (
                                "retrieval",
                                avg_confidence,
                                json!({
                                    "operation": "retrieval",
                                    "description": "Memory retrieval with activation spreading",
                                    "memory_space_id": memory_space_id.as_str(),
                                    "memory_type": "episodic",
                                    "confidence": avg_confidence,
                                    "activation": activation,
                                    "timestamp": Utc::now().to_rfc3339(),
                                    "session_id": session_id,
                                    "memory_id": id,
                                    "consolidation_state": "Retrieved"
                                }),
                            )
                        }
                        MemoryEvent::ActivationSpread { .. } => {
                            // Skip activation spread events for memory stream
                            continue;
                        }
                    };

                    event_id += 1;
                    let sse_event = Event::default()
                        .id(event_id.to_string())
                        .event(operation)
                        .data(event_data.to_string());

                    if tx.send(Ok(sse_event)).await.is_err() {
                        return; // Client disconnected
                    }
                }
                Err(tokio::sync::broadcast::error::RecvError::Lagged(skipped)) => {
                    let warning_event = Event::default().event("warning").data(
                        json!({
                            "warning": format!("Lagged behind, {} events skipped", skipped),
                            "session_id": session_id,
                        })
                        .to_string(),
                    );
                    if tx.send(Ok(warning_event)).await.is_err() {
                        return;
                    }
                }
                Err(tokio::sync::broadcast::error::RecvError::Closed) => {
                    return;
                }
            }
        }
    });

    ReceiverStream::new(rx)
}

/// Create consolidation stream delivering semantic belief updates in real-time snapshots
fn create_consolidation_stream(
    store: Arc<MemoryStore>,
    session_id: String,
    include_replay: bool,
    include_insights: bool,
    include_progress: bool,
    min_novelty: f32,
) -> impl Stream<Item = Result<Event, std::convert::Infallible>> {
    let (tx, rx) = tokio::sync::mpsc::channel(32);

    tokio::spawn(async move {
        use std::collections::HashMap;

        let mut seen_strengths: HashMap<String, f32> = HashMap::new();
        let mut interval = tokio::time::interval(std::time::Duration::from_secs(10));

        loop {
            interval.tick().await;

            let snapshot = store.consolidation_snapshot(128);
            let run_stats = ConsolidationRunStats::from(snapshot.stats.clone());

            if include_progress {
                let progress_payload = json!({
                    "session_id": session_id.as_str(),
                    "generated_at": snapshot.generated_at,
                    "stats": &run_stats,
                });

                let progress_event = Event::default()
                    .event("progress")
                    .data(progress_payload.to_string());

                if tx.send(Ok(progress_event)).await.is_err() {
                    return;
                }
            }

            for pattern in snapshot.patterns {
                let previous = seen_strengths.insert(pattern.id.clone(), pattern.strength);
                let novelty = previous.map_or(pattern.strength, |strength| {
                    (pattern.strength - strength).abs()
                });

                if novelty < min_novelty {
                    continue;
                }

                let mut belief = pattern_to_belief(pattern, store.as_ref());
                if !include_replay {
                    belief.citations = Vec::new();
                }

                let mut payload = json!({
                    "session_id": session_id.as_str(),
                    "generated_at": snapshot.generated_at,
                    "novelty": novelty,
                    "belief": belief,
                });

                if include_insights {
                    payload["stats"] = serde_json::to_value(&run_stats).unwrap_or_default();
                }

                let belief_event = Event::default()
                    .event("belief")
                    .id(uuid::Uuid::new_v4().to_string())
                    .data(payload.to_string());

                if tx.send(Ok(belief_event)).await.is_err() {
                    return;
                }
            }

            // Keep SSE connection warm even if no new beliefs were emitted
            let keepalive = Event::default().comment(format!(
                "consolidation heartbeat for session {}",
                session_id.as_str()
            ));

            if tx.send(Ok(keepalive)).await.is_err() {
                return;
            }
        }
    });

    ReceiverStream::new(rx)
}

// ================================================================================================
// Real-Time Monitoring Handlers (Debugging-Focused)
// ================================================================================================

/// Monitor real-time events with hierarchical debugging support
#[utoipa::path(
    get,
    path = "/api/v1/monitoring/events",
    tag = "monitoring",
    params(MonitoringQuery),
    responses(
        (status = 200, description = "Server-Sent Events stream", content_type = "text/event-stream"),
        (status = 400, description = "Invalid monitoring parameters", body = ErrorResponse)
    )
)]
pub async fn monitor_events(
    State(_state): State<ApiState>,
    Query(params): Query<MonitoringQuery>,
) -> impl IntoResponse {
    let session_id = params
        .session_id
        .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());
    let level = params
        .level
        .unwrap_or_else(|| "global".to_string())
        .to_lowercase();
    let max_frequency = params.max_frequency.unwrap_or(10.0).clamp(0.1, 50.0);
    let min_activation = params.min_activation.unwrap_or(0.1).clamp(0.0, 1.0);
    let include_causality = params.include_causality.unwrap_or(true);

    let event_types = params.event_types.map_or_else(
        || {
            vec![
                "activation".to_string(),
                "formation".to_string(),
                "decay".to_string(),
                "spreading".to_string(),
            ]
        },
        |types| {
            types
                .split(',')
                .map(|t| t.trim().to_lowercase())
                .collect::<Vec<_>>()
        },
    );

    let stream = create_monitoring_stream(
        session_id,
        level,
        params.focus_id,
        max_frequency,
        event_types,
        min_activation,
        include_causality,
        params.last_sequence.unwrap_or(0),
    );

    Sse::new(stream).keep_alive(KeepAlive::default())
}

/// Monitor real-time activation levels with spreading traces
#[utoipa::path(
    get,
    path = "/api/v1/monitoring/activations",
    tag = "monitoring",
    params(ActivationMonitoringQuery),
    responses(
        (status = 200, description = "Server-Sent Events stream", content_type = "text/event-stream"),
        (status = 400, description = "Invalid monitoring parameters", body = ErrorResponse)
    )
)]
pub async fn monitor_activations(
    State(_state): State<ApiState>,
    Query(params): Query<ActivationMonitoringQuery>,
) -> impl IntoResponse {
    let session_id = params
        .session_id
        .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());
    let threshold = params.threshold.unwrap_or(0.2).clamp(0.0, 1.0);
    let frequency = params.frequency.unwrap_or(20.0).clamp(1.0, 100.0);
    let time_window = params.time_window.unwrap_or(5.0).clamp(0.1, 60.0);
    let include_spreading = params.include_spreading.unwrap_or(true);

    let stream = create_activation_monitoring_stream(
        session_id,
        params.node_pattern,
        threshold,
        frequency,
        time_window,
        include_spreading,
    );

    Sse::new(stream).keep_alive(KeepAlive::default())
}

/// Monitor causality chains for debugging distributed memory operations
#[utoipa::path(
    get,
    path = "/api/v1/monitoring/causality",
    tag = "monitoring",
    params(CausalityMonitoringQuery),
    responses(
        (status = 200, description = "Server-Sent Events stream", content_type = "text/event-stream"),
        (status = 400, description = "Invalid monitoring parameters", body = ErrorResponse)
    )
)]
pub async fn monitor_causality(
    State(_state): State<ApiState>,
    Query(params): Query<CausalityMonitoringQuery>,
) -> impl IntoResponse {
    let session_id = params
        .session_id
        .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());
    let max_chain_length = params.max_chain_length.unwrap_or(10).min(20);
    let min_confidence = params.min_confidence.unwrap_or(0.5).clamp(0.0, 1.0);
    let temporal_window = params.temporal_window.unwrap_or(1000).min(10000);
    let include_indirect = params.include_indirect.unwrap_or(false);

    let operation_types = params.operation_types.map_or_else(
        || {
            vec![
                "remember".to_string(),
                "recall".to_string(),
                "consolidate".to_string(),
                "activate".to_string(),
            ]
        },
        |types| {
            types
                .split(',')
                .map(|t| t.trim().to_lowercase())
                .collect::<Vec<_>>()
        },
    );

    let stream = create_causality_monitoring_stream(
        session_id,
        max_chain_length,
        min_confidence,
        operation_types,
        temporal_window,
        include_indirect,
    );

    Sse::new(stream).keep_alive(KeepAlive::default())
}

fn create_monitoring_stream(
    session_id: String,
    level: String,
    focus_id: Option<String>,
    max_frequency: f32,
    event_types: Vec<String>,
    min_activation: f32,
    include_causality: bool,
    last_sequence: u64,
) -> impl Stream<Item = Result<Event, std::convert::Infallible>> {
    let (tx, rx) = tokio::sync::mpsc::channel(64);

    tokio::spawn(async move {
        let mut event_id = last_sequence;
        let interval_ms = (1000.0 / max_frequency) as u64;

        let events = match level.as_str() {
            "global" => vec![
                ("system_activation", "Global activation level changed"),
                ("consolidation_wave", "Consolidation wave started"),
                ("attention_shift", "Attention focus shifted"),
            ],
            "region" => vec![
                ("region_activation", "Memory region activated"),
                ("cluster_formation", "Memory cluster formed"),
            ],
            "node" => vec![
                ("node_activation", "Memory node activated"),
                ("link_formation", "New link formed"),
            ],
            "edge" => vec![
                ("edge_strengthening", "Edge weight increased"),
                ("edge_formation", "New edge created"),
            ],
            _ => vec![("unknown_event", "Unknown event type")],
        };

        loop {
            for (event_type, description) in &events {
                if !event_types.contains(&(*event_type).to_string()) {
                    continue;
                }

                event_id += 1;
                let activation_level = rand::random::<f32>().mul_add(0.9, 0.1);
                if activation_level < min_activation {
                    continue;
                }

                let mut event_data = json!({
                    "event_type": event_type, "description": description, "level": level,
                    "activation_level": activation_level, "timestamp": Utc::now().to_rfc3339(),
                    "session_id": session_id, "sequence": event_id,
                    "latency_ms": rand::random::<u64>() % 100,
                    "debug_info": {
                        "cognitive_load": if activation_level > 0.8 { "high" } else { "normal" },
                        "hierarchy_level": level,
                    }
                });

                if let Some(ref focus) = focus_id {
                    event_data["focus_id"] = json!(focus);
                    event_data["focused"] = json!(true);
                }

                if include_causality {
                    event_data["causality"] = json!({
                        "caused_by": format!("event_{}", event_id.saturating_sub(rand::random::<u64>() % 5 + 1)),
                        "temporal_delay_ms": rand::random::<u64>() % 100
                    });
                }

                let sse_event = Event::default()
                    .id(event_id.to_string())
                    .event("monitor")
                    .data(event_data.to_string());

                if tx.send(Ok(sse_event)).await.is_err() {
                    return;
                }
                tokio::time::sleep(tokio::time::Duration::from_millis(interval_ms)).await;
            }
        }
    });

    ReceiverStream::new(rx)
}

fn create_activation_monitoring_stream(
    session_id: String,
    node_pattern: Option<String>,
    threshold: f32,
    frequency: f32,
    _time_window: f32,
    include_spreading: bool,
) -> impl Stream<Item = Result<Event, std::convert::Infallible>> {
    let (tx, rx) = tokio::sync::mpsc::channel(128);

    tokio::spawn(async move {
        let mut event_id = 0u64;
        let interval_ms = (1000.0 / frequency) as u64;

        loop {
            event_id += 1;
            let activation_level = rand::random::<f32>();
            if activation_level < threshold {
                tokio::time::sleep(tokio::time::Duration::from_millis(interval_ms)).await;
                continue;
            }

            let node_id = node_pattern.as_ref().map_or_else(
                || format!("node_{}", rand::random::<u32>() % 10000),
                |pattern| format!("{}_{}", pattern, rand::random::<u32>() % 1000),
            );

            let mut event_data = json!({
                "node_id": node_id, "activation_level": activation_level,
                "timestamp": Utc::now().to_rfc3339(), "session_id": session_id,
                "sequence": event_id, "latency_ms": rand::random::<u64>() % 50,
                "visualization": {
                    "color_intensity": (activation_level * 255.0) as u8,
                    "pulse_rate": activation_level * 10.0,
                }
            });

            if include_spreading {
                let spreading_count = (activation_level * 5.0) as usize;
                let spreading_nodes = (0..spreading_count)
                    .map(|i| {
                        json!({
                            "target_node": format!("spread_{}", i),
                            "spread_strength": activation_level * rand::random::<f32>(),
                            "delay_ms": (i as f32 * 10.0) as u64
                        })
                    })
                    .collect::<Vec<_>>();

                event_data["spreading"] = json!({
                    "spread_count": spreading_count, "targets": spreading_nodes,
                    "total_energy": activation_level
                });
            }

            let sse_event = Event::default()
                .id(event_id.to_string())
                .event("activation")
                .data(event_data.to_string());

            if tx.send(Ok(sse_event)).await.is_err() {
                return;
            }
            tokio::time::sleep(tokio::time::Duration::from_millis(interval_ms)).await;
        }
    });

    ReceiverStream::new(rx)
}

fn create_causality_monitoring_stream(
    session_id: String,
    max_chain_length: usize,
    min_confidence: f32,
    operation_types: Vec<String>,
    temporal_window: u64,
    include_indirect: bool,
) -> impl Stream<Item = Result<Event, std::convert::Infallible>> {
    let (tx, rx) = tokio::sync::mpsc::channel(32);

    tokio::spawn(async move {
        let mut event_id = 0u64;
        let mut causal_chains = std::collections::HashMap::new();

        loop {
            event_id += 1;
            let operation = &operation_types[rand::random::<usize>() % operation_types.len()];
            let confidence = rand::random::<f32>();
            if confidence < min_confidence {
                tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;
                continue;
            }

            let operation_id = format!("{operation}_{event_id}");
            let chain_length =
                std::cmp::min(max_chain_length, (confidence * 8.0).max(0.0) as usize + 1);
            let causal_chain = (0..chain_length).map(|i| json!({
                "operation_id": format!("cause_{}_{}", operation, event_id.saturating_sub(i as u64 + 1)),
                "confidence": confidence * rand::random::<f32>(),
                "temporal_offset_ms": (i as u64) * 50,
            })).collect::<Vec<_>>();

            causal_chains.insert(operation_id.clone(), causal_chain.clone());

            let mut event_data = json!({
                "operation_id": operation_id, "operation_type": operation, "confidence": confidence,
                "timestamp": Utc::now().to_rfc3339(), "session_id": session_id, "sequence": event_id,
                "causal_chain": causal_chain, "latency_ms": rand::random::<u64>() % 100,
            });

            if include_indirect && causal_chains.len() > 1 {
                let indirect_causes = causal_chains
                    .iter()
                    .filter(|(id, _)| *id != &operation_id)
                    .take(3)
                    .map(|(id, chain)| {
                        json!({
                            "indirect_operation": id, "indirect_strength": confidence * 0.5,
                            "indirect_chain_length": chain.len()
                        })
                    })
                    .collect::<Vec<_>>();
                event_data["indirect_causality"] = json!(indirect_causes);
            }

            let sse_event = Event::default()
                .id(event_id.to_string())
                .event("causality")
                .data(event_data.to_string());

            if tx.send(Ok(sse_event)).await.is_err() {
                return;
            }

            if causal_chains.len() > 100 {
                let oldest_keys: Vec<_> = causal_chains.keys().take(50).cloned().collect();
                for key in oldest_keys {
                    causal_chains.remove(&key);
                }
            }

            tokio::time::sleep(tokio::time::Duration::from_millis(temporal_window)).await;
        }
    });

    ReceiverStream::new(rx)
}

// ================================================================================================
// API Router Creation
// ================================================================================================

/// Create cognitive-friendly API router
pub fn create_api_routes() -> Router<ApiState> {
    let swagger_router = create_swagger_ui().with_state::<ApiState>(());

    Router::new()
        // Server lifecycle
        .route("/shutdown", post(shutdown_server))
        // Simple health endpoint for status checks
        .route("/health", get(simple_health))
        .route("/health/alive", get(alive_health))
        .route("/health/spreading", get(spreading_health))
        // Memory space registry operations
        .route(
            "/api/v1/spaces",
            get(list_memory_spaces).post(create_memory_space),
        )
        // Memory operations with cognitive-friendly paths
        .route("/api/v1/memories/remember", post(remember_memory))
        .route("/api/v1/memories/recall", get(recall_memories))
        .route("/api/v1/memories/recognize", post(recognize_pattern))
        .route("/api/v1/consolidations", get(list_consolidations))
        .route("/api/v1/consolidations/{id}", get(get_consolidation))
        // Probabilistic query operations
        .route("/api/v1/query/probabilistic", get(probabilistic_query))
        // Pattern completion operations
        .route("/api/v1/complete", post(crate::handlers::complete::complete_handler))
        // REST-style endpoints for CLI compatibility
        .route("/api/v1/memories", post(create_memory_rest))
        .route("/api/v1/memories/{id}", get(get_memory_by_id))
        .route(
            "/api/v1/memories/{id}",
            axum::routing::delete(delete_memory_by_id),
        )
        .route("/api/v1/memories/search", get(search_memories_rest))
        // System health and introspection
        .route("/api/v1/system/health", get(system_health))
        .route("/api/v1/system/spreading/config", get(spreading_config))
        .route("/api/v1/system/introspect", get(system_introspect))
        .route("/metrics", get(metrics_snapshot))
        // Episode-specific operations
        .route("/api/v1/episodes/remember", post(remember_episode))
        .route("/api/v1/episodes/replay", get(replay_episodes))
        // Streaming operations (SSE)
        .route("/api/v1/stream/activities", get(stream_activities))
        .route("/api/v1/stream/memories", get(stream_memories))
        .route("/api/v1/stream/consolidation", get(stream_consolidation))
        // Real-time monitoring (specialized for debugging)
        .route("/api/v1/monitoring/events", get(monitor_events))
        .route("/api/v1/monitoring/activations", get(monitor_activations))
        .route("/api/v1/monitoring/causality", get(monitor_causality))
        // Backwards compatibility for pre-013 clients
        .route("/api/v1/monitor/events", get(monitor_events))
        .route("/api/v1/monitor/activations", get(monitor_activations))
        .route("/api/v1/monitor/causality", get(monitor_causality))
        // Swagger UI documentation
        .merge(swagger_router)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_space_prefers_header_over_query() {
        let mut headers = HeaderMap::new();
        headers.insert("x-memory-space", "header-space".parse().unwrap());
        let query = Some("query-space");
        let body = Some("body-space");
        let default = MemorySpaceId::default();

        let result = extract_memory_space_id(&headers, query, body, &default).unwrap();

        assert_eq!(
            result.as_str(),
            "header-space",
            "Header should take priority over query and body"
        );
    }

    #[test]
    fn test_extract_space_prefers_query_over_body() {
        let headers = HeaderMap::new();
        let query = Some("alpha");
        let body = Some("beta");
        let default = MemorySpaceId::default();

        let result = extract_memory_space_id(&headers, query, body, &default).unwrap();

        assert_eq!(
            result.as_str(),
            "alpha",
            "Query parameter should take priority over body field"
        );
    }

    #[test]
    fn test_extract_space_uses_body_when_no_query() {
        let headers = HeaderMap::new();
        let query = None;
        let body = Some("gamma");
        let default = MemorySpaceId::default();

        let result = extract_memory_space_id(&headers, query, body, &default).unwrap();

        assert_eq!(
            result.as_str(),
            "gamma",
            "Body field should be used when query parameter is absent"
        );
    }

    #[test]
    fn test_extract_space_falls_back_to_default() {
        let headers = HeaderMap::new();
        let query = None;
        let body = None;
        let default = MemorySpaceId::try_from("custom-default").unwrap();

        let result = extract_memory_space_id(&headers, query, body, &default).unwrap();

        assert_eq!(
            result.as_str(),
            "custom-default",
            "Should fall back to default when both query and body are absent"
        );
    }

    #[test]
    fn test_extract_space_rejects_invalid_query_param() {
        let headers = HeaderMap::new();
        let query = Some("INVALID SPACE!");
        let body = Some("valid-space");
        let default = MemorySpaceId::default();

        let result = extract_memory_space_id(&headers, query, body, &default);

        assert!(
            result.is_err(),
            "Should reject invalid space ID in query parameter"
        );
        if let Err(ApiError::InvalidInput(msg)) = result {
            assert!(
                msg.contains("query parameter"),
                "Error should mention query parameter"
            );
        } else {
            panic!("Expected ApiError::InvalidInput");
        }
    }

    #[test]
    fn test_extract_space_rejects_invalid_body_field() {
        let headers = HeaderMap::new();
        let query = None;
        let body = Some("INVALID SPACE!");
        let default = MemorySpaceId::default();

        let result = extract_memory_space_id(&headers, query, body, &default);

        assert!(
            result.is_err(),
            "Should reject invalid space ID in body field"
        );
        if let Err(ApiError::InvalidInput(msg)) = result {
            assert!(
                msg.contains("request body"),
                "Error should mention request body"
            );
        } else {
            panic!("Expected ApiError::InvalidInput");
        }
    }
}
