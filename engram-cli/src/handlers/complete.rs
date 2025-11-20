//! Pattern completion HTTP handler with biological plausibility.
//!
//! Implements POST /api/v1/complete endpoint exposing hippocampal pattern completion
//! with comprehensive error handling following Nielsen's actionable error guidelines.

use axum::{
    extract::State,
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Json},
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use utoipa::{IntoParams, ToSchema};

use crate::api::{ApiError, ApiState};
use engram_core::{
    Confidence as CoreConfidence, MemorySpaceId,
    completion::{
        CompletedEpisode, CompletionConfig, CompletionError, HippocampalCompletion, MemorySource,
        PartialEpisode, PatternCompleter,
    },
    metrics::{
        completion_metrics::{
            CA3_CONVERGENCE_ITERATIONS, COMPLETION_CONVERGENCE_FAILURES_TOTAL,
            COMPLETION_DURATION_SECONDS, COMPLETION_INSUFFICIENT_EVIDENCE_TOTAL,
            COMPLETION_OPERATIONS_TOTAL, CompletionTimer,
        },
        with_space,
    },
};

// ================================================================================================
// Request/Response Types with utoipa Documentation
// ================================================================================================

/// Request for pattern completion with partial episode cues
#[derive(Debug, Deserialize, ToSchema, IntoParams)]
pub struct CompleteRequest {
    /// Memory space identifier for multi-tenant isolation
    pub memory_space_id: Option<String>,

    /// Partial episode with known fields to complete
    pub partial_episode: PartialEpisodeRequest,

    /// Optional configuration for completion algorithm
    pub config: Option<CompletionConfigRequest>,
}

/// Partial episode containing known fields and missing information
#[derive(Debug, Deserialize, ToSchema)]
pub struct PartialEpisodeRequest {
    /// Known fields from the episode (e.g., "what": "breakfast")
    pub known_fields: HashMap<String, String>,

    /// Partial embedding vector with masked dimensions
    /// Each element is either Some(value) for known dimensions or None for masked dimensions
    #[serde(default)]
    pub partial_embedding: Vec<Option<f32>>,

    /// Cue strength for pattern completion (0.0-1.0)
    #[serde(default = "default_cue_strength")]
    pub cue_strength: f32,

    /// Temporal context from surrounding episodes
    #[serde(default)]
    pub temporal_context: Vec<String>,
}

const fn default_cue_strength() -> f32 {
    0.7
}

/// Configuration parameters for completion algorithm
#[derive(Debug, Deserialize, ToSchema)]
pub struct CompletionConfigRequest {
    /// CA1 confidence threshold for output gating (0.0-1.0)
    #[serde(default = "default_ca1_threshold")]
    pub ca1_threshold: f32,

    /// Number of alternative hypotheses to generate
    #[serde(default = "default_num_hypotheses")]
    pub num_hypotheses: usize,

    /// Maximum iterations for CA3 convergence
    #[serde(default = "default_max_iterations")]
    pub max_iterations: usize,
}

const fn default_ca1_threshold() -> f32 {
    0.7
}

const fn default_num_hypotheses() -> usize {
    3
}

const fn default_max_iterations() -> usize {
    7
}

/// Successful pattern completion response
#[derive(Debug, Serialize, ToSchema)]
pub struct CompleteResponse {
    /// Completed episode with reconstructed fields
    pub completed_episode: EpisodeResponse,

    /// Pattern completion confidence (CA1 output)
    pub completion_confidence: ConfidenceResponse,

    /// Source attribution: which parts are recalled vs reconstructed
    pub source_attribution: HashMap<String, SourceAttributionResponse>,

    /// Alternative completion hypotheses from System 2 reasoning
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub alternative_hypotheses: Vec<AlternativeHypothesisResponse>,

    /// Metacognitive confidence in completion quality
    pub metacognitive_confidence: ConfidenceResponse,

    /// Reconstruction statistics from completion process
    pub reconstruction_stats: ReconstructionStatsResponse,
}

/// Completed episode representation
#[derive(Debug, Serialize, ToSchema)]
pub struct EpisodeResponse {
    /// Episode identifier
    pub id: String,

    /// When the episode occurred
    pub when: String,

    /// What happened (semantic content)
    pub what: String,

    /// Where it occurred (if reconstructed)
    pub where_location: Option<String>,

    /// Who was involved (if reconstructed)
    pub who: Option<Vec<String>>,

    /// Encoding confidence
    pub encoding_confidence: f32,
}

/// Confidence value with reasoning
#[derive(Debug, Serialize, ToSchema)]
pub struct ConfidenceResponse {
    /// Numeric confidence value [0.0, 1.0]
    pub value: f32,

    /// Human-readable category
    pub category: String,

    /// Reasoning for this confidence level
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<String>,
}

/// Source attribution for a completed field
#[derive(Debug, Serialize, ToSchema)]
pub struct SourceAttributionResponse {
    /// Memory source type
    pub source: String,

    /// Confidence in this field's reconstruction
    pub confidence: f32,
}

/// Alternative completion hypothesis
#[derive(Debug, Serialize, ToSchema)]
pub struct AlternativeHypothesisResponse {
    /// Alternative episode representation
    pub episode: EpisodeResponse,

    /// Confidence in this hypothesis
    pub confidence: f32,
}

/// Statistics from reconstruction process
#[derive(Debug, Serialize, ToSchema)]
pub struct ReconstructionStatsResponse {
    /// Number of CA3 iterations to convergence
    pub ca3_iterations: usize,

    /// Whether attractor dynamics converged
    pub convergence_achieved: bool,

    /// Source patterns used in reconstruction
    pub pattern_sources: Vec<String>,

    /// Overall plausibility score
    pub plausibility_score: f32,
}

/// Error response following Nielsen's actionable error guidelines
#[derive(Debug, Serialize, ToSchema)]
pub struct CompletionErrorResponse {
    /// Error code for programmatic handling
    pub error: String,

    /// Diagnosis: what went wrong
    pub message: String,

    /// Context: why it failed
    pub details: HashMap<String, serde_json::Value>,

    /// Action: how to fix
    pub suggestion: String,
}

// ================================================================================================
// Handler Implementation
// ================================================================================================

/// Complete partial episode using pattern completion
///
/// Implements hippocampal-inspired pattern completion with CA3 autoassociative dynamics,
/// DG pattern separation, and CA1 output gating. Returns completed episode with source
/// attribution and alternative hypotheses.
#[utoipa::path(
    post,
    path = "/api/v1/complete",
    tag = "completion",
    request_body = CompleteRequest,
    responses(
        (status = 200, description = "Successful pattern completion", body = CompleteResponse),
        (status = 400, description = "Invalid request format", body = CompletionErrorResponse),
        (status = 404, description = "Memory space not found", body = CompletionErrorResponse),
        (status = 422, description = "Insufficient evidence for completion", body = CompletionErrorResponse),
        (status = 500, description = "Internal completion error", body = CompletionErrorResponse)
    )
)]
pub async fn complete_handler(
    State(state): State<ApiState>,
    headers: HeaderMap,
    Json(req): Json<CompleteRequest>,
) -> Result<impl IntoResponse, ApiError> {
    // Start metrics timer
    let mut timer = CompletionTimer::new();

    // Extract memory space from header or request body
    let space_id = extract_memory_space(
        &headers,
        req.memory_space_id.as_deref(),
        &state.default_space,
    )?;

    // Prepare labels for metrics
    let labels = with_space(&space_id);

    // Get space-specific store handle from registry
    let handle = state.registry.create_or_get(&space_id).await.map_err(|e| {
        ApiError::SystemError(format!("Failed to access memory space '{space_id}': {e}"))
    })?;

    let _store = handle.store();

    // Validate request
    validate_completion_request(&req)?;

    // Build completion configuration
    let config = build_completion_config(req.config);

    // Convert request to internal partial episode
    let partial = convert_to_partial_episode(&req.partial_episode);

    // Create completion engine
    let completion = HippocampalCompletion::new(config);

    // Perform pattern completion with error tracking
    let result = match completion.complete(&partial) {
        Ok(completed) => {
            // Record success metrics
            let success_labels = [
                ("memory_space", space_id.as_str().to_string()),
                ("result", "success".to_string()),
            ];
            state.metrics.increment_counter_with_labels(
                COMPLETION_OPERATIONS_TOTAL,
                1,
                &success_labels,
            );

            Ok(completed)
        }
        Err(e) => {
            // Record error-specific metrics
            match &e {
                CompletionError::InsufficientPattern => {
                    state.metrics.increment_counter_with_labels(
                        COMPLETION_INSUFFICIENT_EVIDENCE_TOTAL,
                        1,
                        &labels,
                    );
                    let error_labels = [
                        ("memory_space", space_id.as_str().to_string()),
                        ("result", "insufficient_evidence".to_string()),
                    ];
                    state.metrics.increment_counter_with_labels(
                        COMPLETION_OPERATIONS_TOTAL,
                        1,
                        &error_labels,
                    );
                }
                CompletionError::ConvergenceFailed(iterations) => {
                    state.metrics.increment_counter_with_labels(
                        COMPLETION_CONVERGENCE_FAILURES_TOTAL,
                        1,
                        &labels,
                    );
                    let error_labels = [
                        ("memory_space", space_id.as_str().to_string()),
                        ("result", "convergence_failed".to_string()),
                    ];
                    state.metrics.increment_counter_with_labels(
                        COMPLETION_OPERATIONS_TOTAL,
                        1,
                        &error_labels,
                    );

                    // Record iterations histogram even on failure
                    state.metrics.observe_histogram_with_labels(
                        CA3_CONVERGENCE_ITERATIONS,
                        *iterations as f64,
                        &labels,
                    );
                }
                _ => {
                    let error_labels = [
                        ("memory_space", space_id.as_str().to_string()),
                        ("result", "error".to_string()),
                    ];
                    state.metrics.increment_counter_with_labels(
                        COMPLETION_OPERATIONS_TOTAL,
                        1,
                        &error_labels,
                    );
                }
            }

            Err(map_completion_error(e, &req.partial_episode))
        }
    }?;

    // Record completion latency
    let total_duration = timer.finalize();
    state.metrics.observe_histogram_with_labels(
        COMPLETION_DURATION_SECONDS,
        total_duration.as_secs_f64(),
        &labels,
    );

    // Convert to response format
    let response = convert_to_response(&result);

    Ok((StatusCode::OK, Json(response)))
}

// ================================================================================================
// Helper Functions
// ================================================================================================

/// Extract memory space ID with header priority
fn extract_memory_space(
    headers: &HeaderMap,
    body_space: Option<&str>,
    default: &MemorySpaceId,
) -> Result<MemorySpaceId, ApiError> {
    // Priority 1: X-Memory-Space header
    if let Some(header_value) = headers.get("x-memory-space") {
        let space_str = header_value.to_str().map_err(|_| {
            ApiError::InvalidInput("X-Memory-Space header contains invalid UTF-8".to_string())
        })?;
        return MemorySpaceId::try_from(space_str).map_err(|e| {
            ApiError::InvalidInput(format!("Invalid memory space ID in header: {e}"))
        });
    }

    // Priority 2: Request body field
    if let Some(space_str) = body_space {
        return MemorySpaceId::try_from(space_str).map_err(|e| {
            ApiError::InvalidInput(format!("Invalid memory space ID in request body: {e}"))
        });
    }

    // Priority 3: Default space
    Ok(default.clone())
}

/// Validate completion request
fn validate_completion_request(req: &CompleteRequest) -> Result<(), ApiError> {
    // Ensure we have at least some known fields
    if req.partial_episode.known_fields.is_empty()
        && req.partial_episode.partial_embedding.is_empty()
    {
        return Err(ApiError::ValidationError(
            "Pattern completion requires at least some known fields or partial embedding. \
             Provide either known_fields or partial_embedding to reconstruct missing information."
                .to_string(),
        ));
    }

    // Validate cue strength
    if !(0.0..=1.0).contains(&req.partial_episode.cue_strength) {
        return Err(ApiError::ValidationError(format!(
            "Cue strength must be between 0.0 and 1.0, got {}",
            req.partial_episode.cue_strength
        )));
    }

    // Validate embedding dimensions if provided
    if !req.partial_episode.partial_embedding.is_empty()
        && req.partial_episode.partial_embedding.len() != 768
    {
        return Err(ApiError::ValidationError(format!(
            "Partial embedding must have exactly 768 dimensions, got {}",
            req.partial_episode.partial_embedding.len()
        )));
    }

    Ok(())
}

/// Build completion configuration from request
fn build_completion_config(config_req: Option<CompletionConfigRequest>) -> CompletionConfig {
    let config_req = config_req.unwrap_or_else(|| CompletionConfigRequest {
        ca1_threshold: default_ca1_threshold(),
        num_hypotheses: default_num_hypotheses(),
        max_iterations: default_max_iterations(),
    });

    CompletionConfig {
        ca1_threshold: CoreConfidence::exact(config_req.ca1_threshold),
        num_hypotheses: config_req.num_hypotheses,
        max_iterations: config_req.max_iterations,
        ..Default::default()
    }
}

/// Convert request to internal partial episode
fn convert_to_partial_episode(req: &PartialEpisodeRequest) -> PartialEpisode {
    // If partial_embedding is empty, create one filled with None
    let partial_embedding = if req.partial_embedding.is_empty() {
        vec![None; 768]
    } else {
        req.partial_embedding.clone()
    };

    PartialEpisode {
        known_fields: req.known_fields.clone(),
        partial_embedding,
        cue_strength: CoreConfidence::exact(req.cue_strength),
        temporal_context: req.temporal_context.clone(),
    }
}

/// Map completion errors to HTTP errors with actionable messages
fn map_completion_error(error: CompletionError, partial: &PartialEpisodeRequest) -> ApiError {
    match error {
        CompletionError::InsufficientPattern => {
            let known_count = partial
                .partial_embedding
                .iter()
                .filter(|v| v.is_some())
                .count();
            #[allow(clippy::cast_precision_loss)]
            let overlap = if partial.partial_embedding.is_empty() {
                0.0
            } else {
                known_count as f32 / partial.partial_embedding.len() as f32
            };

            ApiError::ValidationError(format!(
                "Pattern completion requires minimum 30% cue overlap. \
                 Current overlap: {:.1}% ({} of {} dimensions known). \
                 Suggestion: Provide additional context fields (e.g., 'when', 'where') \
                 or reduce ca1_threshold to 0.6 for lower-confidence completions.",
                overlap * 100.0,
                known_count,
                partial.partial_embedding.len().max(768)
            ))
        }
        CompletionError::ConvergenceFailed(iterations) => ApiError::SystemError(format!(
            "Pattern completion failed to converge after {iterations} iterations. \
             The CA3 attractor dynamics did not stabilize. \
             This may indicate conflicting evidence in the cue. \
             Try simplifying the partial pattern or increasing max_iterations."
        )),
        CompletionError::InvalidEmbeddingDimension(dim) => ApiError::InvalidInput(format!(
            "Invalid embedding dimension: expected 768, got {dim}"
        )),
        CompletionError::LowConfidence(conf) => ApiError::ValidationError(format!(
            "Completion confidence {conf:.2} below CA1 threshold. \
             The reconstructed episode did not meet quality standards. \
             Suggestion: Provide more cue information or adjust ca1_threshold."
        )),
        CompletionError::MatrixError(msg) => {
            ApiError::SystemError(format!("Pattern completion matrix operation failed: {msg}"))
        }
    }
}

/// Convert completed episode to response format
fn convert_to_response(completed: &CompletedEpisode) -> CompleteResponse {
    let episode = EpisodeResponse {
        id: completed.episode.id.clone(),
        when: completed.episode.when.to_rfc3339(),
        what: completed.episode.what.clone(),
        where_location: completed.episode.where_location.clone(),
        who: completed.episode.who.clone(),
        encoding_confidence: completed.episode.encoding_confidence.raw(),
    };

    let source_attribution = completed
        .source_attribution
        .field_sources
        .iter()
        .map(|(field, source)| {
            let confidence = completed
                .source_attribution
                .source_confidence
                .get(field)
                .map_or(0.0, |c| c.raw());

            (
                field.clone(),
                SourceAttributionResponse {
                    source: source_to_string(*source).to_string(),
                    confidence,
                },
            )
        })
        .collect();

    let alternative_hypotheses = completed
        .alternative_hypotheses
        .iter()
        .map(|(ep, conf)| AlternativeHypothesisResponse {
            episode: EpisodeResponse {
                id: ep.id.clone(),
                when: ep.when.to_rfc3339(),
                what: ep.what.clone(),
                where_location: ep.where_location.clone(),
                who: ep.who.clone(),
                encoding_confidence: ep.encoding_confidence.raw(),
            },
            confidence: conf.raw(),
        })
        .collect();

    CompleteResponse {
        completed_episode: episode,
        completion_confidence: confidence_to_response(
            completed.completion_confidence,
            Some("CA1 output confidence"),
        ),
        source_attribution,
        alternative_hypotheses,
        metacognitive_confidence: confidence_to_response(
            completed.metacognitive_confidence,
            Some("System 2 metacognitive monitoring"),
        ),
        reconstruction_stats: ReconstructionStatsResponse {
            // NOTE: CA3 iteration statistics are not currently exposed by the
            // HippocampalCompletion engine. The engine tracks iterations and convergence
            // internally but doesn't include these in CompletedEpisode.
            // See Technical Debt item in review notes.
            ca3_iterations: 0,
            convergence_achieved: true,
            pattern_sources: vec![],
            plausibility_score: completed.completion_confidence.raw(),
        },
    }
}

/// Convert confidence to response format
fn confidence_to_response(conf: CoreConfidence, reasoning: Option<&str>) -> ConfidenceResponse {
    ConfidenceResponse {
        value: conf.raw(),
        category: confidence_category(conf.raw()),
        reasoning: reasoning.map(ToString::to_string),
    }
}

/// Map confidence value to category
fn confidence_category(value: f32) -> String {
    if value > 0.9 {
        "Certain".to_string()
    } else if value > 0.7 {
        "High".to_string()
    } else if value > 0.4 {
        "Medium".to_string()
    } else if value > 0.1 {
        "Low".to_string()
    } else {
        "None".to_string()
    }
}

/// Convert memory source to string
const fn source_to_string(source: MemorySource) -> &'static str {
    match source {
        MemorySource::Recalled => "recalled",
        MemorySource::Reconstructed => "reconstructed",
        MemorySource::Imagined => "imagined",
        MemorySource::Consolidated => "consolidated",
    }
}
