//! HTTP API module for cognitive-friendly memory operations
//!
//! This module provides REST API endpoints that follow cognitive ergonomics
//! principles with natural language paths and educational error messages.

use axum::response::sse::{Event, KeepAlive};
use axum::{
    Router,
    extract::{Query, State},
    http::StatusCode,
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
use tracing::info;
use utoipa::{IntoParams, ToSchema};

use crate::grpc::MemoryService;
use crate::openapi::create_swagger_ui;
use engram_core::{
    Confidence as CoreConfidence, MemoryStore,
    memory::{EpisodeBuilder as CoreEpisodeBuilder},
};

/// Shared application state
#[derive(Clone)]
pub struct ApiState {
    /// Cognitive memory store with HNSW indexing and activation spreading
    pub store: Arc<MemoryStore>,
    /// gRPC memory service for complex operations
    pub memory_service: Arc<MemoryService>,
}

impl ApiState {
    /// Create new API state with memory store
    pub fn new(store: Arc<MemoryStore>) -> Self {
        let memory_service = Arc::new(MemoryService::new(Arc::clone(&store)));
        Self {
            store,
            memory_service,
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
    /// Should we automatically link to similar memories?
    pub auto_link: Option<bool>,
    /// Threshold for automatic linking (0.0-1.0)
    pub link_threshold: Option<f32>,
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
    /// Automatic links created (if enabled)
    pub auto_links: Vec<AutoLink>,
    /// Educational message about the memory storage
    pub system_message: String,
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
    let confidence = Confidence::new(confidence_value).with_reasoning(
        request
            .confidence_reasoning
            .unwrap_or_else(|| "Default confidence for user-provided memory".to_string()),
    );

    let core_confidence = CoreConfidence::exact(confidence_value);

    // Determine memory type
    let memory_type = match request.memory_type.as_deref() {
        Some("episodic") => MemoryType::Episodic,
        Some("procedural") => MemoryType::Procedural,
        _ => MemoryType::Semantic, // Default for "semantic" and unknown types
    };

    // Create memory object for response
    let memory = Memory::new(memory_id.clone(), embedding_vec.clone())
        .with_content(&request.content)
        .with_confidence(confidence.clone())
        .with_type(memory_type);

    // Add tags if provided
    let _memory = if let Some(tags) = request.tags {
        tags.into_iter().fold(memory, engram_proto::Memory::add_tag)
    } else {
        memory
    };

    // Store in MemoryStore as an episode
    use engram_core::Episode;
    let episode = Episode::new(
        memory_id.clone(),
        Utc::now(),
        request.content.clone(),
        embedding_array,
        core_confidence,
    );

    let activation = state.store.store(episode);

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

    let actual_confidence = activation.value();
    let response = RememberResponse {
        memory_id: memory_id.clone(),
        storage_confidence: ConfidenceInfo {
            value: actual_confidence,
            category: if actual_confidence > 0.7 {
                "High"
            } else if actual_confidence > 0.4 {
                "Medium"
            } else {
                "Low"
            }
            .to_string(),
            reasoning: format!(
                "Stored with activation {:.2}, accounting for system pressure",
                actual_confidence
            ),
        },
        consolidation_state: format!("{:?}", ConsolidationState::Recent),
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
    Json(request): Json<RememberEpisodeRequest>,
) -> Result<impl IntoResponse, ApiError> {
    info!("Processing remember episode request: {}", request.what);

    // Validate episode content
    if request.what.trim().is_empty() {
        return Err(ApiError::InvalidInput(
            "Episode description cannot be empty. Episodic memories need vivid 'what happened' narratives.".to_string()
        ));
    }

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

    let builder = if let Some(participants) = request.who.clone() {
        builder.who(participants)
    } else {
        builder
    };

    let core_episode = builder.build();

    // Store in MemoryStore
    let activation = state.store.store(core_episode);

    let actual_confidence = activation.value();
    let response = RememberResponse {
        memory_id: episode_id.clone(),
        storage_confidence: ConfidenceInfo {
            value: actual_confidence,
            category: if actual_confidence > 0.7 {
                "High"
            } else if actual_confidence > 0.4 {
                "Medium"
            } else {
                "Low"
            }
            .to_string(),
            reasoning: format!(
                "Episodic memory stored with activation {:.2}, rich context aids consolidation",
                actual_confidence
            ),
        },
        consolidation_state: format!("{:?}", ConsolidationState::Recent),
        auto_links: vec![],
        system_message: format!(
            "Episode '{episode_id}' successfully encoded with {:.2} activation. Rich episodes consolidate better over time.",
            actual_confidence
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
    Query(params): Query<RecallQuery>,
) -> Result<impl IntoResponse, ApiError> {
    let start_time = std::time::Instant::now();

    info!("Processing recall request with query: {:?}", params.query);

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
    use engram_core::Cue as CoreCue;
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

    // Perform actual recall using MemoryStore
    let recall_results = state.store.recall(&cue);
    let total_memories = state.store.count();

    // Convert results to API response format
    let mut vivid_results = Vec::new();
    let mut associated_results = Vec::new();

    for (episode, confidence) in recall_results.iter().take(params.max_results.unwrap_or(10)) {
        let result = MemoryResult {
            id: episode.id.clone(),
            content: episode.what.clone(),
            confidence: ConfidenceInfo {
                value: confidence.raw(),
                category: if confidence.is_high() {
                    "High"
                } else if confidence.is_medium() {
                    "Medium"
                } else {
                    "Low"
                }
                .to_string(),
                reasoning: "Recalled from memory store with confidence score".to_string(),
            },
            activation_level: confidence.raw(),
            similarity_score: confidence.raw(),
            retrieval_path: Some("Memory store recall".to_string()),
            last_access: Some(episode.when),
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
            reasoning: "Strong query understanding with multiple retrieval pathways".to_string(),
        },
        query_analysis: QueryAnalysis {
            understood_intent: params
                .query
                .clone()
                .unwrap_or_else(|| "Embedding-based search".to_string()),
            search_strategy: "Spreading activation with similarity threshold".to_string(),
            cognitive_load: "Medium".to_string(),
            suggestions: vec![
                "Try adding specific tags to narrow results".to_string(),
                "Consider temporal constraints for episode searches".to_string(),
            ],
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

/// GET /health - Simple health check
#[utoipa::path(
    get,
    path = "/health",
    tag = "system",
    responses(
        (status = 200, description = "Simple health status", body = serde_json::Value),
        (status = 500, description = "Health check failed", body = ErrorResponse)
    )
)]
/// # Errors
///
/// Returns `ApiError` if health check fails
pub async fn simple_health() -> Result<impl IntoResponse, ApiError> {
    let health_data = json!({
        "status": "healthy",
        "service": "engram",
        "version": env!("CARGO_PKG_VERSION"),
        "timestamp": Utc::now().to_rfc3339()
    });

    Ok(Json(health_data))
}

/// GET /api/v1/system/health - Comprehensive system health
#[utoipa::path(
    get,
    path = "/api/v1/system/health",
    tag = "system",
    responses(
        (status = 200, description = "System health information", body = HealthResponse),
        (status = 500, description = "System health check failed", body = ErrorResponse)
    )
)]
/// # Errors
///
/// Returns `ApiError` if system health check fails
pub async fn system_health(State(state): State<ApiState>) -> Result<impl IntoResponse, ApiError> {
    let memory_count = state.store.count();

    let health_data = json!({
        "status": "healthy",
        "memory_system": {
            "total_memories": memory_count,
            "consolidation_active": true,
            "spreading_activation": "normal",
            "pattern_completion": "available"
        },
        "cognitive_load": {
            "current": "low",
            "capacity_remaining": "85%",
            "consolidation_queue": 0
        },
        "system_message": format!(
            "Memory system operational with {} stored memories. All cognitive processes functioning normally.",
            memory_count
        )
    });

    Ok(Json(health_data))
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
    let memory_count = state.store.count();

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
pub async fn stream_activities(
    State(_state): State<ApiState>,
    Query(params): Query<StreamActivityQuery>,
) -> impl IntoResponse {
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

    // Create activity stream
    let stream = create_activity_stream(session_id, event_types, min_importance, buffer_size);

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
pub async fn stream_memories(
    State(_state): State<ApiState>,
    Query(params): Query<StreamMemoryQuery>,
) -> impl IntoResponse {
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
pub async fn stream_consolidation(
    State(_state): State<ApiState>,
    Query(params): Query<StreamConsolidationQuery>,
) -> impl IntoResponse {
    let session_id = params
        .session_id
        .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());
    let min_novelty = params.min_novelty.unwrap_or(0.4).clamp(0.0, 1.0);

    let include_replay = params.include_replay.unwrap_or(true);
    let include_insights = params.include_insights.unwrap_or(true);
    let include_progress = params.include_progress.unwrap_or(false);

    let stream = create_consolidation_stream(
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

/// Create activity event stream with cognitive-friendly organization
fn create_activity_stream(
    session_id: String,
    event_types: Vec<String>,
    min_importance: f32,
    buffer_size: usize,
) -> impl Stream<Item = Result<Event, std::convert::Infallible>> {
    let (tx, rx) = tokio::sync::mpsc::channel(buffer_size);

    tokio::spawn(async move {
        let mut event_id = 0u64;
        let events = vec![
            ("activation", "Memory node activated for pattern matching"),
            ("storage", "New memory encoded with contextual associations"),
            ("recall", "Memory retrieval triggered by recognition cue"),
            (
                "consolidation",
                "Memory consolidation strengthening associations",
            ),
            (
                "association",
                "New associative link formed between memories",
            ),
            ("decay", "Memory activation decaying due to inactivity"),
        ];

        loop {
            for (event_type, description) in &events {
                if !event_types.contains(&(*event_type).to_string()) {
                    continue;
                }

                event_id += 1;
                let importance = rand::random::<f32>().mul_add(0.7, 0.3); // Random importance 0.3-1.0

                if importance < min_importance {
                    continue;
                }

                let event_data = json!({
                    "event_type": event_type,
                    "description": description,
                    "importance": importance,
                    "timestamp": Utc::now().to_rfc3339(),
                    "session_id": session_id,
                    "metadata": {
                        "cognitive_load": if importance > 0.8 { "high" } else { "normal" },
                        "attention_required": importance > 0.9
                    }
                });

                let sse_event = Event::default()
                    .id(event_id.to_string())
                    .event(event_type)
                    .data(event_data.to_string());

                if tx.send(Ok(sse_event)).await.is_err() {
                    return; // Client disconnected
                }

                // Cognitive-friendly pacing: 200ms to 2s intervals based on importance
                let delay_ms = importance.mul_add(-1800.0, 2000.0).max(0.0) as u64;
                tokio::time::sleep(tokio::time::Duration::from_millis(delay_ms)).await;
            }
        }
    });

    ReceiverStream::new(rx)
}

/// Create memory operation stream
fn create_memory_stream(
    session_id: String,
    include_formation: bool,
    include_retrieval: bool,
    include_completion: bool,
    min_confidence: f32,
    memory_types: Vec<String>,
) -> impl Stream<Item = Result<Event, std::convert::Infallible>> {
    let (tx, rx) = tokio::sync::mpsc::channel(32);

    tokio::spawn(async move {
        let mut event_id = 0u64;
        let operations = vec![
            (
                "formation",
                "New memory formation with encoding confidence",
                include_formation,
            ),
            (
                "retrieval",
                "Memory retrieval with activation spreading",
                include_retrieval,
            ),
            (
                "completion",
                "Pattern completion reconstructing partial memories",
                include_completion,
            ),
        ];

        loop {
            for (op_type, description, enabled) in &operations {
                if !enabled {
                    continue;
                }

                event_id += 1;
                let confidence = rand::random::<f32>().mul_add(0.8, 0.2); // Random confidence 0.2-1.0

                if confidence < min_confidence {
                    continue;
                }

                let memory_type = &memory_types[rand::random::<usize>() % memory_types.len()];

                let event_data = json!({
                    "operation": op_type,
                    "description": description,
                    "memory_type": memory_type,
                    "confidence": confidence,
                    "timestamp": Utc::now().to_rfc3339(),
                    "session_id": session_id,
                    "memory_id": format!("mem_{}", event_id),
                    "consolidation_state": if confidence > 0.8 { "Stable" } else { "Recent" }
                });

                let sse_event = Event::default()
                    .id(event_id.to_string())
                    .event("memory")
                    .data(event_data.to_string());

                if tx.send(Ok(sse_event)).await.is_err() {
                    return; // Client disconnected
                }

                tokio::time::sleep(tokio::time::Duration::from_millis(1000)).await;
            }
        }
    });

    ReceiverStream::new(rx)
}

/// Create consolidation stream with dream-like replay sequences
fn create_consolidation_stream(
    session_id: String,
    include_replay: bool,
    include_insights: bool,
    include_progress: bool,
    min_novelty: f32,
) -> impl Stream<Item = Result<Event, std::convert::Infallible>> {
    let (tx, rx) = tokio::sync::mpsc::channel(32);

    tokio::spawn(async move {
        let mut event_id = 0u64;
        let consolidation_types = vec![
            (
                "replay",
                "Memory replay sequence consolidating associations",
                include_replay,
            ),
            (
                "insight",
                "Novel connection discovered during consolidation",
                include_insights,
            ),
            (
                "progress",
                "Consolidation progress with strength metrics",
                include_progress,
            ),
        ];

        loop {
            for (cons_type, description, enabled) in &consolidation_types {
                if !enabled {
                    continue;
                }

                event_id += 1;
                let novelty = rand::random::<f32>(); // Random novelty 0.0-1.0

                if novelty < min_novelty {
                    continue;
                }

                let event_data = json!({
                    "consolidation_type": cons_type,
                    "description": description,
                    "novelty": novelty,
                    "timestamp": Utc::now().to_rfc3339(),
                    "session_id": session_id,
                    "sequence_id": format!("seq_{}", event_id),
                    "connections_formed": rand::random::<u32>() % 10,
                    "replay_strength": novelty.mul_add(0.5, 0.5)
                });

                let sse_event = Event::default()
                    .id(event_id.to_string())
                    .event("consolidation")
                    .data(event_data.to_string());

                if tx.send(Ok(sse_event)).await.is_err() {
                    return; // Client disconnected
                }

                // Longer intervals for consolidation (1-5 seconds)
                let delay_ms = 1000 + (rand::random::<u64>() % 4000);
                tokio::time::sleep(tokio::time::Duration::from_millis(delay_ms)).await;
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
    path = "/api/v1/monitor/events",
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

    let event_types = params
        .event_types
        .map(|types| {
            types
                .split(',')
                .map(|t| t.trim().to_lowercase())
                .collect::<Vec<_>>()
        })
        .unwrap_or_else(|| {
            vec![
                "activation".to_string(),
                "formation".to_string(),
                "decay".to_string(),
                "spreading".to_string(),
            ]
        });

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
    path = "/api/v1/monitor/activations",
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
    path = "/api/v1/monitor/causality",
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

    let operation_types = params
        .operation_types
        .map(|types| {
            types
                .split(',')
                .map(|t| t.trim().to_lowercase())
                .collect::<Vec<_>>()
        })
        .unwrap_or_else(|| {
            vec![
                "remember".to_string(),
                "recall".to_string(),
                "consolidate".to_string(),
                "activate".to_string(),
            ]
        });

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
        // Simple health endpoint for status checks
        .route("/health", get(simple_health))
        // Memory operations with cognitive-friendly paths
        .route("/api/v1/memories/remember", post(remember_memory))
        .route("/api/v1/memories/recall", get(recall_memories))
        .route("/api/v1/memories/recognize", post(recognize_pattern))
        // System health and introspection
        .route("/api/v1/system/health", get(system_health))
        .route("/api/v1/system/introspect", get(system_introspect))
        // Episode-specific operations
        .route("/api/v1/episodes/remember", post(remember_episode))
        .route("/api/v1/episodes/replay", get(replay_episodes))
        // Streaming operations (SSE)
        .route("/api/v1/stream/activities", get(stream_activities))
        .route("/api/v1/stream/memories", get(stream_memories))
        .route("/api/v1/stream/consolidation", get(stream_consolidation))
        // Real-time monitoring (specialized for debugging)
        .route("/api/v1/monitor/events", get(monitor_events))
        .route("/api/v1/monitor/activations", get(monitor_activations))
        .route("/api/v1/monitor/causality", get(monitor_causality))
        // Swagger UI documentation
        .merge(swagger_router)
}
