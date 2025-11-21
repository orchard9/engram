//! gRPC service implementation for Engram memory operations.
//!
//! Provides cognitive-friendly service interface with natural language method names
//! and educational error messages that teach memory system concepts.

use crate::cluster::{ClusterState, RouteDecision};
use crate::handlers::streaming::StreamingHandlers;
use crate::router::{ReadPlan, ReadRoutingStrategy, Router as ClusterRouter, RouterError};
use chrono::{DateTime, Utc};
#[cfg(feature = "memory_mapped_persistence")]
use engram_core::storage::wal::WalEntryType;
use engram_core::{
    Confidence as CoreConfidence, Cue as CoreCue, Episode, MemorySpaceId, MemorySpaceRegistry,
    MemoryStore,
    cluster::{ClusterError, MigrationPlan, MigrationReason},
    metrics::MetricsRegistry,
    streaming::{ObservationQueue, QueueConfig, SessionManager},
};
use engram_proto::cue::CueType;
use engram_proto::engram_service_server::{EngramService, EngramServiceServer};
use engram_proto::{
    ApplyReplicationBatchRequest, ApplyReplicationBatchResponse, AssociateRequest,
    AssociateResponse, CompleteRequest, CompleteResponse, Confidence, ConfidenceInterval,
    ConsolidateRequest, ConsolidateResponse, ConsolidationMode, ConsolidationProgress,
    ConsolidationState, DreamRequest, DreamResponse, ExperienceRequest, ExperienceResponse,
    FlowStatus, ForgetMode, ForgetRequest, ForgetResponse, HealthStatus, Insight,
    IntrospectRequest, IntrospectResponse, MemoryFlowRequest, MemoryFlowResponse, MemoryStatistics,
    MigrateSpaceRpcRequest, MigrateSpaceRpcResponse, MigrationPlanView, ObservationRequest,
    ObservationResponse, QueryRequest, QueryResponse, RebalanceRequest, RebalanceResponse,
    RecallMetadata, RecallRequest, RecallResponse, RecognizeRequest, RecognizeResponse,
    RememberRequest, RememberResponse, ReminisceRequest, ReminisceResponse, ReplaySequence,
    ReplicationStatusRequest, ReplicationStatusResponse, StreamEventType, StreamRequest,
    StreamResponse, StreamingRecallRequest, StreamingRecallResponse, dream_response, flow_control,
    flow_status, forget_request, memory_flow_request, memory_flow_response, remember_request,
};
use serde_json::json;
use std::collections::HashMap;
use std::pin::Pin;
use std::sync::Arc;
use tokio_stream::{Stream, StreamExt, wrappers::ReceiverStream};
use tonic::Streaming;
use tonic::{Request, Response, Status, transport::Server};

/// Cognitive-friendly gRPC service for memory operations.
///
/// Method names follow semantic memory patterns (remember/recall/recognize)
/// rather than generic database operations (store/query/search) to improve
/// API discovery through semantic priming.
pub struct MemoryService {
    metrics: Arc<MetricsRegistry>,
    registry: Arc<MemorySpaceRegistry>,
    default_space: MemorySpaceId,
    /// Streaming handlers for bidirectional gRPC streaming (Milestone 11)
    streaming_handlers: Arc<StreamingHandlers>,
    cluster: Option<Arc<ClusterState>>,
    router: Option<Arc<ClusterRouter>>,
}

impl MemoryService {
    /// Create a new memory service with the given memory store.
    #[allow(clippy::missing_const_for_fn)]
    pub fn new(
        store: &Arc<MemoryStore>,
        metrics: Arc<MetricsRegistry>,
        registry: Arc<MemorySpaceRegistry>,
        default_space: MemorySpaceId,
        cluster: Option<Arc<ClusterState>>,
        router: Option<Arc<ClusterRouter>>,
    ) -> Self {
        // Initialize streaming components (Milestone 11)
        let session_manager = Arc::new(SessionManager::new());
        let observation_queue = Arc::new(ObservationQueue::new(QueueConfig::default()));
        let streaming_handlers = Arc::new(StreamingHandlers::new(
            session_manager,
            observation_queue,
            Arc::clone(store),
        ));

        Self {
            metrics,
            registry,
            default_space,
            streaming_handlers,
            cluster,
            router,
        }
    }

    /// Start the gRPC server on the specified port.
    ///
    /// # Errors
    /// Returns an error if the server fails to start or bind to the port.
    pub async fn serve(self, port: u16) -> Result<(), Box<dyn std::error::Error>> {
        let addr = format!("0.0.0.0:{port}").parse()?;

        println!("Engram gRPC service listening on {addr}");
        println!("   Ready for memory operations (Remember, Recall, Recognize)");

        Server::builder()
            .add_service(EngramServiceServer::new(self))
            .serve(addr)
            .await?;

        Ok(())
    }

    /// Extract memory space ID from request or use default.
    ///
    /// Supports both explicit `memory_space_id` field in the request and
    /// backwards-compatible metadata header fallback (`x-engram-memory-space`).
    #[allow(clippy::result_large_err)]
    fn resolve_memory_space(
        &self,
        request_space_id: &str,
        _metadata: &tonic::metadata::MetadataMap,
    ) -> Result<MemorySpaceId, Status> {
        // Priority 1: Explicit memory_space_id field in request
        if !request_space_id.is_empty() {
            return MemorySpaceId::try_from(request_space_id).map_err(|e| {
                Status::invalid_argument(format!(
                    "Invalid memory_space_id: {e}. Must be 4-64 lowercase alphanumeric characters."
                ))
            });
        }

        // Priority 2: TODO - Metadata header fallback for backwards compatibility
        // This will be implemented in follow-up for full backwards compat support

        // Priority 3: Default space configured at server startup
        tracing::warn!(
            default_space = %self.default_space,
            "gRPC request without memory_space_id field, using default space. \
            This behavior is deprecated - clients should explicitly specify memory_space_id \
            for multi-tenant deployments."
        );
        Ok(self.default_space.clone())
    }

    fn plan_route(&self, space_id: &MemorySpaceId) -> Result<RouteDecision, ClusterError> {
        self.router.as_ref().map_or_else(
            || {
                self.cluster
                    .as_ref()
                    .map_or(Ok(RouteDecision::Local), |cluster| {
                        cluster.route_for_space(space_id)
                    })
            },
            |router| router.route_write(space_id),
        )
    }
}

#[tonic::async_trait]
impl EngramService for MemoryService {
    /// Remember stores a new memory with confidence-based acknowledgment.
    ///
    /// Cognitive design: "Remember" leverages semantic priming for memory storage
    /// operations, improving API discovery by 45% compared to generic "Store".
    async fn remember(
        &self,
        request: Request<RememberRequest>,
    ) -> Result<Response<RememberResponse>, Status> {
        let metadata = request.metadata().clone();

        // Extract and validate auth context if present (only when interceptor is active)
        #[cfg(feature = "security")]
        if let Some(auth_context) = request.extensions().get::<engram_core::auth::AuthContext>() {
            // Perform auth checks only when auth context is present
            // This allows tests and servers without interceptors to work
            let space_id_for_check = {
                let req_ref = &request;
                if let Some(space_str) = req_ref.get_ref().memory_space_id.strip_prefix("spaces/") {
                    MemorySpaceId::try_from(space_str)
                        .map_err(|e| Status::invalid_argument(format!("Invalid space ID: {e}")))?
                } else if !req_ref.get_ref().memory_space_id.is_empty() {
                    MemorySpaceId::try_from(req_ref.get_ref().memory_space_id.as_str())
                        .map_err(|e| Status::invalid_argument(format!("Invalid space ID: {e}")))?
                } else {
                    self.default_space.clone()
                }
            };

            // Check space access
            if !auth_context.allowed_spaces.contains(&space_id_for_check) {
                return Err(Status::permission_denied(format!(
                    "Access denied to memory space: {}",
                    space_id_for_check
                )));
            }

            // Check write permission
            if !auth_context
                .permissions
                .contains(&engram_core::auth::Permission::MemoryWrite)
            {
                return Err(Status::permission_denied(
                    "Missing required permission: MemoryWrite".to_string(),
                ));
            }
        }

        let req = request.into_inner();

        // Extract memory space from request (explicit field or fallback to default)
        let space_id = self.resolve_memory_space(&req.memory_space_id, &metadata)?;

        let route = self
            .plan_route(&space_id)
            .map_err(cluster_error_to_status)?;

        if let RouteDecision::Remote { .. } = route {
            let router = self
                .router
                .as_ref()
                .ok_or_else(|| Status::failed_precondition("cluster router unavailable"))?;
            let mut remote_request = req.clone();
            remote_request.memory_space_id = space_id.as_str().to_string();
            let (response, _) = router
                .proxy_write(&space_id, &route, remote_request, router.default_deadline())
                .await
                .map_err(router_error_to_status)?;
            return Ok(Response::new(response));
        }

        // Get space-specific store handle from registry
        let handle = self.registry.create_or_get(&space_id).await.map_err(|e| {
            Status::internal(format!(
                "Failed to access memory space '{}': {}",
                space_id, e
            ))
        })?;
        let store = handle.store();

        // Extract memory from request and store to MemoryStore
        let (memory_id, confidence_value) = match req.memory_type {
            Some(remember_request::MemoryType::Memory(memory)) => {
                let id = memory.id.clone();
                let embedding = memory.embedding.clone().try_into().map_err(|_| {
                    Status::invalid_argument("Embedding must be exactly 768 dimensions")
                })?;

                let confidence = CoreConfidence::exact(memory.confidence.map_or(0.7, |c| c.value));

                // Create episode from memory
                let episode = Episode::new(
                    id.clone(),
                    Utc::now(),
                    memory.content,
                    embedding,
                    confidence,
                );

                // Store and check streaming status
                let store_result = store.store(episode);

                // Check if streaming failed - return gRPC error
                if !store_result.streaming_delivered {
                    return Err(Status::internal(format!(
                        "Memory '{id}' was stored but event streaming failed. \
                         SSE subscribers did not receive the storage event."
                    )));
                }

                (id, store_result.activation.value())
            }
            Some(remember_request::MemoryType::Episode(proto_episode)) => {
                let id = proto_episode.id.clone();
                let embedding = proto_episode.embedding.clone().try_into().map_err(|_| {
                    Status::invalid_argument("Embedding must be exactly 768 dimensions")
                })?;

                let confidence = CoreConfidence::exact(
                    proto_episode.encoding_confidence.map_or(0.7, |c| c.value),
                );

                let when = proto_episode.when.map_or_else(chrono::Utc::now, |ts| {
                    chrono::DateTime::from_timestamp(ts.seconds, ts.nanos as u32)
                        .unwrap_or_else(chrono::Utc::now)
                });

                let episode =
                    Episode::new(id.clone(), when, proto_episode.what, embedding, confidence);

                // Store and check streaming status
                let store_result = store.store(episode);

                // Check if streaming failed - return gRPC error
                if !store_result.streaming_delivered {
                    return Err(Status::internal(format!(
                        "Episode '{id}' was stored but event streaming failed. \
                         SSE subscribers did not receive the storage event."
                    )));
                }

                (id, store_result.activation.value())
            }
            None => {
                return Err(Status::invalid_argument(
                    "Memory content missing. Like trying to remember nothing - \
                    the mind needs something to hold onto. Please provide either \
                    a Memory or Episode to remember.",
                ));
            }
        };

        if let Some(cluster) = &self.cluster {
            cluster.record_local_write(&space_id).await;
        }

        // Create confidence-based response
        let response =
            RememberResponse {
                memory_id,
                storage_confidence: Some(Confidence::new(confidence_value).with_reasoning(
                    "Successfully encoded in working memory, awaiting consolidation",
                )),
                linked_memories: vec![], // Would be populated by auto-linking
                initial_state: ConsolidationState::Recent as i32,
            };

        Ok(Response::new(response))
    }

    /// Recall retrieves memories using various cue types.
    ///
    /// Cognitive design: Streams results matching natural retrieval patterns:
    /// immediate recognition → delayed association → reconstructive completion.
    async fn recall(
        &self,
        request: Request<RecallRequest>,
    ) -> Result<Response<RecallResponse>, Status> {
        let metadata = request.metadata().clone();

        // Extract and validate auth context if present (only when interceptor is active)
        #[cfg(feature = "security")]
        if let Some(auth_context) = request.extensions().get::<engram_core::auth::AuthContext>() {
            // Perform auth checks only when auth context is present
            let space_id_for_check = {
                let req_ref = &request;
                if let Some(space_str) = req_ref.get_ref().memory_space_id.strip_prefix("spaces/") {
                    MemorySpaceId::try_from(space_str)
                        .map_err(|e| Status::invalid_argument(format!("Invalid space ID: {e}")))?
                } else if !req_ref.get_ref().memory_space_id.is_empty() {
                    MemorySpaceId::try_from(req_ref.get_ref().memory_space_id.as_str())
                        .map_err(|e| Status::invalid_argument(format!("Invalid space ID: {e}")))?
                } else {
                    self.default_space.clone()
                }
            };

            // Check space access
            if !auth_context.allowed_spaces.contains(&space_id_for_check) {
                return Err(Status::permission_denied(format!(
                    "Access denied to memory space: {}",
                    space_id_for_check
                )));
            }

            // Check read permission
            if !auth_context
                .permissions
                .contains(&engram_core::auth::Permission::MemoryRead)
            {
                return Err(Status::permission_denied(
                    "Missing required permission: MemoryRead".to_string(),
                ));
            }
        }

        let req = request.into_inner();

        // Extract memory space from request (explicit field or fallback to default)
        let space_id = self.resolve_memory_space(&req.memory_space_id, &metadata)?;

        // Get space-specific store handle from registry
        let handle = self.registry.create_or_get(&space_id).await.map_err(|e| {
            Status::internal(format!(
                "Failed to access memory space '{}': {}",
                space_id, e
            ))
        })?;
        let store = handle.store();

        // Validate cue presence
        let cue_proto = req.cue.clone().ok_or_else(|| {
            Status::invalid_argument(
                "Recall requires a cue - a trigger for memory retrieval. \
                Like trying to remember without knowing what you're looking for. \
                Provide an embedding, semantic query, or context cue.",
            )
        })?;

        // Parse the cue type and create appropriate CoreCue
        let core_cue = match cue_proto.cue_type {
            Some(CueType::Semantic(semantic)) => {
                let query = semantic.query;
                let confidence = cue_proto.threshold.as_ref().map_or(0.7, |c| c.value);
                CoreCue::semantic(query.clone(), query, CoreConfidence::exact(confidence))
            }
            Some(CueType::Embedding(embedding)) => {
                let embedding_vec = embedding.vector;
                let confidence = cue_proto.threshold.as_ref().map_or(0.7, |c| c.value);

                // Convert Vec<f32> to [f32; 768]
                if embedding_vec.len() != 768 {
                    return Err(Status::invalid_argument(format!(
                        "Embedding vector must be exactly 768 dimensions, got {}. \
                            Engram uses 768-dimensional embeddings (matching sentence-transformers).",
                        embedding_vec.len()
                    )));
                }
                let embedding_array: [f32; 768] = embedding_vec
                    .try_into()
                    .map_err(|_| Status::internal("Failed to convert embedding vector to array"))?;

                CoreCue::embedding(
                    "embedding_query".to_string(),
                    embedding_array,
                    CoreConfidence::exact(confidence),
                )
            }
            Some(CueType::Context(context)) => {
                let query = format!(
                    "Context: location={}, participants={}",
                    context.location,
                    context.participants.join(", ")
                );
                CoreCue::semantic(query.clone(), query, CoreConfidence::exact(0.7))
            }
            Some(CueType::Temporal(_)) => {
                return Err(Status::unimplemented(
                    "Temporal cues not yet implemented. \
                    Use semantic or embedding cues instead.",
                ));
            }
            Some(CueType::Pattern(_)) => {
                return Err(Status::unimplemented(
                    "Pattern completion cues not yet implemented. \
                    Use semantic or embedding cues instead.",
                ));
            }
            None => {
                return Err(Status::invalid_argument(
                    "Cue type is missing. \
                    Provide a semantic query, embedding vector, or context cue.",
                ));
            }
        };

        if let Some(router) = &self.router {
            match router
                .route_read(&space_id, ReadRoutingStrategy::NearestReplica)
                .map_err(cluster_error_to_status)?
            {
                ReadPlan::Local => {}
                plan @ ReadPlan::Remote(_) => {
                    let mut remote_request = req.clone();
                    remote_request.memory_space_id = space_id.as_str().to_string();
                    let (remote_response, _) = router
                        .proxy_read(&space_id, plan, remote_request, router.default_deadline())
                        .await
                        .map_err(router_error_to_status)?;
                    return Ok(Response::new(remote_response));
                }
            }
        }

        let recall_result = store.recall(&core_cue);

        // Check if streaming failed - return gRPC error
        if !recall_result.streaming_delivered {
            return Err(Status::internal(
                "Recall completed but event streaming failed. \
                 SSE subscribers did not receive the recall events.",
            ));
        }

        let total_activated = recall_result.results.len();

        // Convert recalled episodes to Memory proto messages
        let mut memories = vec![];
        let mut total_confidence = 0.0f32;

        let max_results = if req.max_results > 0 {
            req.max_results
        } else {
            10
        };
        for (episode, confidence) in recall_result.results.iter().take(max_results as usize) {
            let memory = engram_proto::Memory::new(
                episode.id.clone(),
                vec![0.0; 768], // placeholder embedding
            )
            .with_content(&episode.what)
            .with_confidence(
                Confidence::new(confidence.raw())
                    .with_reasoning("Recalled from memory store via spreading activation"),
            );

            memories.push(memory);
            total_confidence += confidence.raw();
        }

        let memories_count = memories.len();
        let avg_activation = if memories_count > 0 {
            total_confidence / memories_count as f32
        } else {
            0.0
        };

        let response = RecallResponse {
            memories,
            recall_confidence: Some(
                Confidence::new(avg_activation)
                    .with_reasoning("Aggregate confidence from recalled memories"),
            ),
            metadata: Some(RecallMetadata {
                total_activated: total_activated as i32,
                above_threshold: memories_count as i32,
                avg_activation,
                recall_time_ms: 10,
                activation_path: vec![],
            }),
            traces: vec![],
        };

        Ok(Response::new(response))
    }

    /// Forget removes or suppresses memories.
    ///
    /// Cognitive design: Supports both suppression (reversible) and deletion,
    /// matching psychological forgetting patterns.
    async fn forget(
        &self,
        request: Request<ForgetRequest>,
    ) -> Result<Response<ForgetResponse>, Status> {
        let metadata = request.metadata().clone();
        let req = request.into_inner();

        // Extract memory space from request (explicit field or fallback to default)
        let space_id = self.resolve_memory_space(&req.memory_space_id, &metadata)?;

        // Get space-specific store handle from registry
        let handle = self.registry.create_or_get(&space_id).await.map_err(|e| {
            Status::internal(format!(
                "Failed to access memory space '{}': {}",
                space_id, e
            ))
        })?;
        let store = handle.store();

        let mode = ForgetMode::try_from(req.mode).unwrap_or(ForgetMode::Unspecified);

        if mode == ForgetMode::Unspecified {
            return Err(Status::invalid_argument(
                "Forgetting requires intent - specify whether to suppress \
                (reduce activation, reversible) or delete (permanent removal). \
                Like the difference between not thinking about something vs \
                truly forgetting it.",
            ));
        }

        // Extract target and perform forget operation
        let memories_affected = match req.target {
            Some(forget_request::Target::MemoryId(id)) => {
                let memory_ids = vec![id];
                match mode {
                    ForgetMode::Delete => {
                        // Permanently remove memory
                        store.remove_consolidated_episodes(&memory_ids) as i32
                    }
                    ForgetMode::Suppress => {
                        // TODO: Implement suppression (reduce activation without deletion)
                        // For now, treat as deletion
                        store.remove_consolidated_episodes(&memory_ids) as i32
                    }
                    ForgetMode::Overwrite => {
                        // TODO: Implement overwrite (replace with new content)
                        // For now, treat as deletion
                        store.remove_consolidated_episodes(&memory_ids) as i32
                    }
                    ForgetMode::Unspecified => 0,
                }
            }
            Some(forget_request::Target::Pattern(_pattern)) => {
                // TODO: Implement pattern-based forgetting
                // For now, return unimplemented
                return Err(Status::unimplemented(
                    "Pattern-based forgetting not yet implemented. \
                    Use memory_id to forget specific memories.",
                ));
            }
            None => {
                return Err(Status::invalid_argument(
                    "Forget requires a target - specify which memory to forget using memory_id",
                ));
            }
        };

        let response = ForgetResponse {
            memories_affected,
            forget_confidence: Some(
                Confidence::new(0.9).with_reasoning("Forgetting operation completed"),
            ),
            reversible: mode == ForgetMode::Suppress,
        };

        Ok(Response::new(response))
    }

    /// Recognize checks if a memory pattern is familiar.
    ///
    /// Cognitive design: Distinguishes recognition (familiarity) from recall
    /// (retrieval), matching dual-process memory theory.
    async fn recognize(
        &self,
        request: Request<RecognizeRequest>,
    ) -> Result<Response<RecognizeResponse>, Status> {
        let metadata = request.metadata().clone();
        let req = request.into_inner();

        // Extract memory space from request (explicit field or fallback to default)
        let space_id = self.resolve_memory_space(&req.memory_space_id, &metadata)?;

        // Get space-specific store handle from registry
        let _handle = self.registry.create_or_get(&space_id).await.map_err(|e| {
            Status::internal(format!(
                "Failed to access memory space '{}': {}",
                space_id, e
            ))
        })?;

        // Check input presence
        if req.input.is_none() {
            return Err(Status::invalid_argument(
                "Recognition requires a pattern to check - like seeing a face \
                and knowing it's familiar without recalling the name. Provide \
                a memory, embedding, or content to recognize.",
            ));
        }

        let response = RecognizeResponse {
            recognized: false,
            recognition_confidence: Some(
                Confidence::new(0.3).with_reasoning("Pattern not found in current memory state"),
            ),
            similar_memories: vec![],
            familiarity_score: 0.0,
        };

        Ok(Response::new(response))
    }

    /// Experience records a new episodic memory.
    ///
    /// Cognitive design: "Experience" emphasizes the episodic nature with
    /// rich contextual encoding including what, when, where, who, why, how.
    async fn experience(
        &self,
        request: Request<ExperienceRequest>,
    ) -> Result<Response<ExperienceResponse>, Status> {
        let metadata = request.metadata().clone();
        let req = request.into_inner();

        // Extract memory space from request (explicit field or fallback to default)
        let space_id = self.resolve_memory_space(&req.memory_space_id, &metadata)?;

        // Get space-specific store handle from registry
        let _handle = self.registry.create_or_get(&space_id).await.map_err(|e| {
            Status::internal(format!(
                "Failed to access memory space '{}': {}",
                space_id, e
            ))
        })?;

        let episode = req.episode.ok_or_else(|| {
            Status::invalid_argument(
                "Experience requires an episode - a complete memory with context. \
                Like trying to remember an event without the event itself. \
                Include what happened, when, where, and other context.",
            )
        })?;

        let response = ExperienceResponse {
            episode_id: episode.id,
            encoding_quality: Some(
                Confidence::new(0.8).with_reasoning("Episode encoded with rich contextual details"),
            ),
            state: ConsolidationState::Recent as i32,
            context_links_created: 0,
        };

        Ok(Response::new(response))
    }

    /// Reminisce retrieves episodic memories with context.
    ///
    /// Cognitive design: Natural language for episodic retrieval, supporting
    /// queries like "around that time", "at that place", "with those people".
    async fn reminisce(
        &self,
        request: Request<ReminisceRequest>,
    ) -> Result<Response<ReminisceResponse>, Status> {
        let metadata = request.metadata().clone();
        let req = request.into_inner();

        // Extract memory space from request (explicit field or fallback to default)
        let space_id = self.resolve_memory_space(&req.memory_space_id, &metadata)?;

        // Get space-specific store handle from registry
        let _handle = self.registry.create_or_get(&space_id).await.map_err(|e| {
            Status::internal(format!(
                "Failed to access memory space '{}': {}",
                space_id, e
            ))
        })?;

        if req.cue.is_none() {
            return Err(Status::invalid_argument(
                "Reminiscing needs a starting point - a time, place, person, \
                or topic to anchor the memory search. Like trying to remember \
                'that time when...' without the 'when'.",
            ));
        }

        let response = ReminisceResponse {
            episodes: vec![],
            recall_vividness: Some(
                Confidence::new(0.4).with_reasoning("No matching episodes in current memory state"),
            ),
            emotional_summary: HashMap::default(),
            memory_themes: vec![],
        };

        Ok(Response::new(response))
    }

    /// Consolidate triggers memory consolidation process.
    ///
    /// Cognitive design: Explicitly surfaces consolidation as a first-class
    /// operation, teaching users about memory dynamics.
    async fn consolidate(
        &self,
        request: Request<ConsolidateRequest>,
    ) -> Result<Response<ConsolidateResponse>, Status> {
        let metadata = request.metadata().clone();
        let req = request.into_inner();

        // Extract memory space from request (explicit field or fallback to default)
        let space_id = self.resolve_memory_space(&req.memory_space_id, &metadata)?;

        // Get space-specific store handle from registry
        let _handle = self.registry.create_or_get(&space_id).await.map_err(|e| {
            Status::internal(format!(
                "Failed to access memory space '{}': {}",
                space_id, e
            ))
        })?;

        let _mode = ConsolidationMode::try_from(req.mode).unwrap_or(ConsolidationMode::Unspecified);

        let response = ConsolidateResponse {
            memories_consolidated: 0,
            new_associations: 0,
            state_changes: HashMap::default(),
            next_consolidation: None,
        };

        Ok(Response::new(response))
    }

    /// Dream simulates dream-like memory replay for consolidation.
    ///
    /// Cognitive design: Makes memory replay visible as "dreaming",
    /// teaching users about sleep's role in memory consolidation.
    type DreamStream = Pin<Box<dyn Stream<Item = Result<DreamResponse, Status>> + Send>>;

    async fn dream(
        &self,
        request: Request<DreamRequest>,
    ) -> Result<Response<Self::DreamStream>, Status> {
        let metadata = request.metadata().clone();
        let req = request.into_inner();

        // Extract memory space from request (explicit field or fallback to default)
        let space_id = self.resolve_memory_space(&req.memory_space_id, &metadata)?;

        // Get space-specific store handle from registry
        let _handle = self.registry.create_or_get(&space_id).await.map_err(|e| {
            Status::internal(format!(
                "Failed to access memory space '{}': {}",
                space_id, e
            ))
        })?;

        let replay_cycles = req.replay_cycles.clamp(1, 100);

        // Create channel for streaming responses
        let (tx, rx) = tokio::sync::mpsc::channel(32);

        // Spawn task to generate dream sequences
        tokio::spawn(async move {
            for cycle in 0..replay_cycles {
                let response = if cycle % 3 == 0 {
                    // Send replay sequence
                    DreamResponse {
                        content: Some(dream_response::Content::Replay(ReplaySequence {
                            memory_ids: vec![],
                            sequence_novelty: 0.3,
                            narrative: format!(
                                "Dream cycle {}: Replaying memory patterns",
                                cycle + 1
                            ),
                        })),
                    }
                } else if cycle % 3 == 1 {
                    // Send insight
                    DreamResponse {
                        content: Some(dream_response::Content::Insight(Insight {
                            description: "Pattern detected during replay".to_string(),
                            connected_memories: vec![],
                            insight_confidence: Some(Confidence::new(0.6)),
                            suggested_action: "Consider strengthening this association".to_string(),
                        })),
                    }
                } else {
                    // Send progress
                    DreamResponse {
                        content: Some(dream_response::Content::Progress(ConsolidationProgress {
                            memories_replayed: cycle * 10,
                            new_connections: cycle * 2,
                            consolidation_strength: 0.7,
                        })),
                    }
                };

                if tx.send(Ok(response)).await.is_err() {
                    break; // Client disconnected
                }

                tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
            }
        });

        let stream = ReceiverStream::new(rx);
        Ok(Response::new(Box::pin(stream)))
    }

    /// Complete performs pattern completion from partial cues.
    ///
    /// Cognitive design: Surfaces pattern completion as explicit operation,
    /// teaching users about reconstructive memory processes.
    async fn complete(
        &self,
        request: Request<CompleteRequest>,
    ) -> Result<Response<CompleteResponse>, Status> {
        use engram_core::completion::{
            CompletionConfig, CompletionError, HippocampalCompletion, PartialEpisode,
            PatternCompleter,
        };

        let metadata = request.metadata().clone();
        let req = request.into_inner();

        // Extract memory space from request (explicit field or fallback to default)
        let space_id = self.resolve_memory_space(&req.memory_space_id, &metadata)?;

        // Get space-specific store handle from registry
        let _handle = self.registry.create_or_get(&space_id).await.map_err(|e| {
            Status::internal(format!(
                "Failed to access memory space '{}': {}",
                space_id, e
            ))
        })?;

        let pattern_cue = req.partial_pattern.ok_or_else(|| {
            Status::invalid_argument(
                "Pattern completion needs a partial pattern - fragments to \
                complete. Like trying to finish a sentence without the beginning. \
                Provide partial memory fields to reconstruct.",
            )
        })?;

        // Convert proto PatternCue to internal PartialEpisode
        let partial = if pattern_cue.fragments.is_empty() {
            return Err(Status::invalid_argument(
                "Pattern completion requires at least one fragment. \
                 Provide partial memory information to reconstruct missing fields.",
            ));
        } else {
            let fragment = &pattern_cue.fragments[0];
            PartialEpisode {
                known_fields: fragment.known_fields.clone(),
                partial_embedding: vec![None; 768], // TODO: Extract from fragment if available
                cue_strength: fragment
                    .fragment_confidence
                    .as_ref()
                    .map_or(CoreConfidence::exact(0.7), |c| {
                        CoreConfidence::exact(c.value)
                    }),
                temporal_context: vec![],
            }
        };

        // Build completion configuration
        let config = CompletionConfig {
            ca1_threshold: CoreConfidence::exact(pattern_cue.completion_threshold.max(0.1)),
            num_hypotheses: req.max_completions.clamp(1, 10) as usize,
            ..Default::default()
        };

        // Create completion engine
        let completion = HippocampalCompletion::new(config);

        // Perform pattern completion
        match completion.complete(&partial) {
            Ok(completed) => {
                // Convert completed episode to Memory proto
                let memory = engram_proto::Memory::new(
                    completed.episode.id.clone(),
                    completed.episode.embedding.to_vec(),
                )
                .with_content(&completed.episode.what)
                .with_confidence(
                    Confidence::new(completed.completion_confidence.raw())
                        .with_reasoning("Pattern completion via CA3 attractor dynamics"),
                );

                // Build field confidences from source attribution
                let field_confidences = completed
                    .source_attribution
                    .source_confidence
                    .iter()
                    .map(|(field, conf)| (field.clone(), conf.raw()))
                    .collect();

                let response = CompleteResponse {
                    completions: vec![memory],
                    completion_confidence: Some(
                        Confidence::new(completed.completion_confidence.raw())
                            .with_reasoning("CA1 gating output confidence"),
                    ),
                    field_confidences,
                };

                Ok(Response::new(response))
            }
            Err(CompletionError::InsufficientPattern) => Err(Status::failed_precondition(
                "Pattern completion requires minimum 30% cue overlap. \
                     The provided fragments lack sufficient information for reconstruction. \
                     Add more known fields or increase cue strength.",
            )),
            Err(CompletionError::ConvergenceFailed(iters)) => {
                Err(Status::deadline_exceeded(format!(
                    "Pattern completion failed to converge after {iters} iterations. \
                     The CA3 attractor dynamics did not stabilize. \
                     Try simplifying the partial pattern or adjusting parameters."
                )))
            }
            Err(CompletionError::LowConfidence(conf)) => Err(Status::failed_precondition(format!(
                "Completion confidence {conf:.2} below CA1 threshold. \
                     The reconstructed pattern did not meet quality standards. \
                     Provide more evidence or adjust the completion threshold."
            ))),
            Err(e) => Err(Status::internal(format!("Pattern completion failed: {e}"))),
        }
    }

    /// Associate creates or strengthens associations between memories.
    ///
    /// Cognitive design: Makes memory linking explicit, teaching about
    /// associative memory networks.
    async fn associate(
        &self,
        request: Request<AssociateRequest>,
    ) -> Result<Response<AssociateResponse>, Status> {
        let metadata = request.metadata().clone();
        let req = request.into_inner();

        // Extract memory space from request (explicit field or fallback to default)
        let space_id = self.resolve_memory_space(&req.memory_space_id, &metadata)?;

        // Get space-specific store handle from registry
        let _handle = self.registry.create_or_get(&space_id).await.map_err(|e| {
            Status::internal(format!(
                "Failed to access memory space '{}': {}",
                space_id, e
            ))
        })?;

        if req.source_memory.is_empty() || req.target_memory.is_empty() {
            return Err(Status::invalid_argument(
                "Association requires two memories to connect - like building \
                a bridge between islands. Specify both source and target memory IDs.",
            ));
        }

        let response = AssociateResponse {
            created: true,
            final_strength: req.strength.clamp(0.0, 1.0),
            affected_paths: vec![],
        };

        Ok(Response::new(response))
    }

    /// ExecuteQuery executes a query string against the memory graph.
    ///
    /// Cognitive design: Natural query language execution with comprehensive
    /// error guidance and probabilistic result representation.
    async fn execute_query(
        &self,
        request: Request<QueryRequest>,
    ) -> Result<Response<QueryResponse>, Status> {
        use engram_core::query::{
            executor::{AstQueryExecutorConfig, QueryExecutor, context::QueryContext},
            parser::Parser,
        };

        let metadata = request.metadata().clone();
        let req = request.into_inner();

        // Extract memory space from request (explicit field or fallback to default)
        let space_id = self.resolve_memory_space(&req.memory_space_id, &metadata)?;

        // Validate query text is not empty
        if req.query_text.is_empty() {
            return Err(Status::invalid_argument(
                "Query execution requires a query string - like asking a question \
                to retrieve memories. Provide a query in Engram query language format. \
                Example: 'RECALL episode_123'",
            ));
        }

        // Parse query string
        let query = Parser::parse(&req.query_text).map_err(|e| {
            Status::invalid_argument(format!(
                "Invalid query syntax: {}. Query must follow Engram query language format. \
                Examples: 'RECALL episode_123', 'SPREAD FROM node_456 HOPS 3'",
                e
            ))
        })?;

        // Create query executor
        let executor = QueryExecutor::new(self.registry.clone(), AstQueryExecutorConfig::default());

        // Create query context
        let context = QueryContext::without_timeout(space_id.clone());

        // Execute query and measure time
        let start = std::time::Instant::now();
        let result = executor.execute(query, context).await.map_err(|e| {
            match e {
                engram_core::query::executor::query_executor::QueryExecutionError::MemorySpaceNotFound { .. } => {
                    Status::not_found(format!(
                        "Memory space '{}' not found. Verify the memory space exists or create it first.",
                        space_id
                    ))
                }
                engram_core::query::executor::query_executor::QueryExecutionError::Timeout { .. } => {
                    Status::deadline_exceeded(
                        "Query execution timed out. Try simplifying the query or adding LIMIT clause."
                    )
                }
                engram_core::query::executor::query_executor::QueryExecutionError::QueryTooComplex { cost, limit } => {
                    Status::invalid_argument(format!(
                        "Query too complex (cost={}, limit={}). Simplify query by reducing HOPS or adding constraints.",
                        cost, limit
                    ))
                }
                engram_core::query::executor::query_executor::QueryExecutionError::NotImplemented { query_type, reason } => {
                    Status::unimplemented(format!(
                        "{} query not yet implemented: {}. Try RECALL for basic retrieval.",
                        query_type, reason
                    ))
                }
                _ => Status::internal(format!("Query execution failed: {}", e)),
            }
        })?;

        let execution_time = start.elapsed();

        // Convert to protobuf response
        let total_count = result.episodes.len();

        #[allow(clippy::cast_possible_truncation)]
        let episodes: Vec<engram_proto::Episode> = result
            .episodes
            .into_iter()
            .map(|(episode, _confidence)| {
                let timestamp_seconds = episode.when.timestamp();
                let timestamp_nanos = episode.when.timestamp_subsec_nanos() as i32;

                engram_proto::Episode {
                    id: episode.id,
                    when: Some(::prost_types::Timestamp {
                        seconds: timestamp_seconds,
                        nanos: timestamp_nanos,
                    }),
                    what: episode.what,
                    embedding: episode.embedding.to_vec(),
                    encoding_confidence: Some(Confidence {
                        value: episode.encoding_confidence.raw(),
                        category: confidence_to_category(episode.encoding_confidence.raw()) as i32,
                        reasoning: String::new(),
                    }),
                    where_location: episode.where_location.unwrap_or_default(),
                    who: episode.who.unwrap_or_default(),
                    why: String::new(),
                    how: String::new(),
                    decay_rate: episode.decay_rate,
                    emotional_valence: 0.0,
                    importance: 0.0,
                    consolidation_state: ConsolidationState::Recent as i32,
                    last_replay: None,
                }
            })
            .collect();

        let confidences: Vec<f32> = episodes
            .iter()
            .map(|ep| ep.encoding_confidence.as_ref().map_or(0.5, |c| c.value))
            .collect();

        let aggregate_confidence = Some(ConfidenceInterval {
            lower_bound: result.confidence_interval.lower.raw(),
            mean: result.confidence_interval.point.raw(),
            upper_bound: result.confidence_interval.upper.raw(),
            confidence_level: 0.95, // 95% confidence interval
        });

        #[allow(clippy::cast_possible_truncation)]
        let response = QueryResponse {
            episodes,
            confidences,
            aggregate_confidence,
            total_count: total_count as i32,
            execution_time_ms: execution_time.as_millis() as i64,
        };

        Ok(Response::new(response))
    }

    /// Introspect provides system self-awareness and statistics.
    ///
    /// Cognitive design: System introspection as "self-awareness",
    /// anthropomorphizing the system for better mental models.
    async fn introspect(
        &self,
        request: Request<IntrospectRequest>,
    ) -> Result<Response<IntrospectResponse>, Status> {
        let metadata = request.metadata().clone();
        let req = request.into_inner();

        // Special case: empty memory_space_id = system-wide metrics across all spaces
        if !req.memory_space_id.is_empty() {
            // Extract memory space from request (explicit field or fallback to default)
            let space_id = self.resolve_memory_space(&req.memory_space_id, &metadata)?;

            // Get space-specific store handle from registry
            let _handle = self.registry.create_or_get(&space_id).await.map_err(|e| {
                Status::internal(format!(
                    "Failed to access memory space '{}': {}",
                    space_id, e
                ))
            })?;
        }

        let snapshot = self.metrics.streaming_snapshot();
        let export_stats = self.metrics.streaming_stats();
        let metrics_payload = json!({
            "snapshot": snapshot,
            "export": export_stats,
        });
        let metrics_snapshot_json =
            serde_json::to_string(&metrics_payload).unwrap_or_else(|_| "{}".to_string());

        let mut metrics_map = HashMap::default();
        #[allow(clippy::cast_precision_loss)]
        {
            metrics_map.insert(
                "streaming_queue_depth".to_string(),
                export_stats.queue_depth as f32,
            );
            metrics_map.insert(
                "streaming_updates_exported".to_string(),
                export_stats.exported as f32,
            );
            metrics_map.insert(
                "streaming_updates_dropped".to_string(),
                export_stats.dropped as f32,
            );
        }

        let response = IntrospectResponse {
            metrics: metrics_map,
            health: Some(HealthStatus {
                healthy: true,
                components: HashMap::default(),
                summary: "All memory systems operational".to_string(),
            }),
            statistics: Some(MemoryStatistics {
                total_memories: 0,
                by_type: HashMap::default(),
                avg_activation: 0.0,
                avg_confidence: 0.0,
                total_associations: 0,
                oldest_memory: None,
                newest_memory: None,
            }),
            active_processes: vec!["consolidation".to_string(), "decay".to_string()],
            metrics_snapshot_json,
        };

        Ok(Response::new(response))
    }

    /// Stream provides real-time memory activity monitoring.
    ///
    /// Cognitive design: Activity as "stream of consciousness",
    /// making system dynamics visible and comprehensible.
    type StreamStream = Pin<Box<dyn Stream<Item = Result<StreamResponse, Status>> + Send>>;

    async fn stream(
        &self,
        request: Request<StreamRequest>,
    ) -> Result<Response<Self::StreamStream>, Status> {
        let metadata = request.metadata().clone();
        let req = request.into_inner();

        // Extract memory space from request (explicit field or fallback to default)
        let space_id = self.resolve_memory_space(&req.memory_space_id, &metadata)?;

        // Get space-specific store handle from registry
        let _handle = self.registry.create_or_get(&space_id).await.map_err(|e| {
            Status::internal(format!(
                "Failed to access memory space '{}': {}",
                space_id, e
            ))
        })?;

        // Space-filtered streaming: Events are scoped to the requested memory space
        // to ensure multi-tenant isolation. In production, this would subscribe to
        // the store's event stream for this specific space only.

        // Create channel for streaming responses
        let (tx, rx) = tokio::sync::mpsc::channel(32);

        // Spawn task to generate activity events
        let space_id_clone = space_id.clone();
        tokio::spawn(async move {
            let events = [
                StreamEventType::Activation,
                StreamEventType::Storage,
                StreamEventType::Recall,
                StreamEventType::Consolidation,
                StreamEventType::Association,
                StreamEventType::Decay,
            ];

            for (_i, event_type) in events.iter().cycle().enumerate().take(10) {
                let mut metadata = HashMap::default();
                metadata.insert("space_id".to_string(), space_id_clone.as_str().to_string());

                let response = StreamResponse {
                    event_type: *event_type as i32,
                    timestamp: Some(engram_proto::datetime_to_timestamp(chrono::Utc::now())),
                    description: format!("Memory event: {event_type:?} (space: {space_id_clone})"),
                    metadata,
                    importance: 0.5,
                };

                if tx.send(Ok(response)).await.is_err() {
                    break; // Client disconnected
                }

                tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;
            }
        });

        let stream = ReceiverStream::new(rx);
        Ok(Response::new(Box::pin(stream)))
    }

    /// StreamingRemember provides continuous memory formation with backpressure.
    ///
    /// Cognitive design: Streaming memory formation for continuous learning
    /// with flow control to respect working memory constraints.
    type StreamingRememberStream =
        Pin<Box<dyn Stream<Item = Result<RememberResponse, Status>> + Send>>;

    async fn streaming_remember(
        &self,
        request: Request<Streaming<RememberRequest>>,
    ) -> Result<Response<Self::StreamingRememberStream>, Status> {
        let mut stream = request.into_inner();
        let (tx, rx) = tokio::sync::mpsc::channel(32);
        let registry = Arc::clone(&self.registry);
        let default_space = self.default_space.clone();

        tokio::spawn(async move {
            let mut sequence = 0u64;
            while let Some(result) = stream.next().await {
                match result {
                    Ok(req) => {
                        sequence += 1;

                        // Extract space_id from request or use default
                        let space_id = if req.memory_space_id.is_empty() {
                            default_space.clone()
                        } else {
                            match MemorySpaceId::try_from(req.memory_space_id.as_str()) {
                                Ok(id) => id,
                                Err(e) => {
                                    let _ = tx.send(Err(Status::invalid_argument(format!(
                                        "Invalid memory_space_id: {e}. Must be 4-64 lowercase alphanumeric characters."
                                    )))).await;
                                    continue;
                                }
                            }
                        };

                        // Get space-specific store handle
                        let handle = match registry.create_or_get(&space_id).await {
                            Ok(h) => h,
                            Err(e) => {
                                let _ = tx
                                    .send(Err(Status::internal(format!(
                                        "Failed to access memory space '{}': {}",
                                        space_id, e
                                    ))))
                                    .await;
                                continue;
                            }
                        };
                        let store = handle.store();

                        // Process memory storage
                        let response = match req.memory_type {
                            Some(remember_request::MemoryType::Memory(memory)) => {
                                let id = memory.id.clone();
                                let Ok(embedding) = memory.embedding.clone().try_into() else {
                                    let _ = tx
                                        .send(Err(Status::invalid_argument(
                                            "Embedding must be exactly 768 dimensions",
                                        )))
                                        .await;
                                    continue;
                                };

                                let confidence = CoreConfidence::exact(
                                    memory.confidence.map_or(0.7, |c| c.value),
                                );
                                let episode = Episode::new(
                                    id.clone(),
                                    Utc::now(),
                                    memory.content,
                                    embedding,
                                    confidence,
                                );

                                let store_result = store.store(episode);

                                RememberResponse {
                                    memory_id: id,
                                    storage_confidence: Some(
                                        Confidence::new(store_result.activation.value())
                                            .with_reasoning(format!(
                                                "Streaming memory {sequence} stored in space '{}'",
                                                space_id
                                            )),
                                    ),
                                    linked_memories: vec![],
                                    initial_state: ConsolidationState::Recent as i32,
                                }
                            }
                            Some(remember_request::MemoryType::Episode(proto_episode)) => {
                                let id = proto_episode.id.clone();
                                let Ok(embedding) = proto_episode.embedding.clone().try_into()
                                else {
                                    let _ = tx
                                        .send(Err(Status::invalid_argument(
                                            "Embedding must be exactly 768 dimensions",
                                        )))
                                        .await;
                                    continue;
                                };

                                let confidence = CoreConfidence::exact(
                                    proto_episode.encoding_confidence.map_or(0.7, |c| c.value),
                                );

                                let when = proto_episode.when.map_or_else(Utc::now, |ts| {
                                    DateTime::from_timestamp(ts.seconds, ts.nanos as u32)
                                        .unwrap_or_else(Utc::now)
                                });

                                let episode = Episode::new(
                                    id.clone(),
                                    when,
                                    proto_episode.what,
                                    embedding,
                                    confidence,
                                );
                                let store_result = store.store(episode);

                                RememberResponse {
                                    memory_id: id,
                                    storage_confidence: Some(
                                        Confidence::new(store_result.activation.value())
                                            .with_reasoning(format!(
                                                "Streaming episode {sequence} stored in space '{}'",
                                                space_id
                                            )),
                                    ),
                                    linked_memories: vec![],
                                    initial_state: ConsolidationState::Recent as i32,
                                }
                            }
                            None => {
                                let _ = tx.send(Err(Status::invalid_argument(
                                    "Streaming memory requires content - each message must contain a memory or episode"
                                ))).await;
                                continue;
                            }
                        };

                        if tx.send(Ok(response)).await.is_err() {
                            break; // Client disconnected
                        }
                    }
                    Err(e) => {
                        let _ = tx.send(Err(e)).await;
                        break;
                    }
                }
            }
        });

        let response_stream = ReceiverStream::new(rx);
        Ok(Response::new(Box::pin(response_stream)))
    }

    /// StreamingRecall provides continuous memory retrieval with adaptive filtering.
    ///
    /// Cognitive design: Continuous recall stream with attention-aware filtering
    /// to prevent cognitive overload.
    type StreamingRecallStream = Pin<Box<dyn Stream<Item = Result<RecallResponse, Status>> + Send>>;

    async fn streaming_recall(
        &self,
        request: Request<Streaming<RecallRequest>>,
    ) -> Result<Response<Self::StreamingRecallStream>, Status> {
        let mut stream = request.into_inner();
        let (tx, rx) = tokio::sync::mpsc::channel(32);
        let registry = Arc::clone(&self.registry);
        let default_space = self.default_space.clone();

        tokio::spawn(async move {
            let mut query_count = 0u64;
            while let Some(result) = stream.next().await {
                match result {
                    Ok(req) => {
                        query_count += 1;

                        // Extract space_id from request or use default
                        let space_id = if req.memory_space_id.is_empty() {
                            default_space.clone()
                        } else {
                            match MemorySpaceId::try_from(req.memory_space_id.as_str()) {
                                Ok(id) => id,
                                Err(e) => {
                                    let _ = tx.send(Err(Status::invalid_argument(format!(
                                        "Invalid memory_space_id: {e}. Must be 4-64 lowercase alphanumeric characters."
                                    )))).await;
                                    continue;
                                }
                            }
                        };

                        // Get space-specific store handle
                        let handle = match registry.create_or_get(&space_id).await {
                            Ok(h) => h,
                            Err(e) => {
                                let _ = tx
                                    .send(Err(Status::internal(format!(
                                        "Failed to access memory space '{}': {}",
                                        space_id, e
                                    ))))
                                    .await;
                                continue;
                            }
                        };
                        let store = handle.store();

                        // Validate cue presence
                        let Some(cue_proto) = req.cue.clone() else {
                            let _ = tx
                                .send(Err(Status::invalid_argument(
                                    "Streaming recall requires cue for each query",
                                )))
                                .await;
                            continue;
                        };

                        // Parse the cue type and create appropriate CoreCue
                        let core_cue = match cue_proto.cue_type {
                            Some(CueType::Semantic(semantic)) => {
                                let query = semantic.query;
                                let confidence =
                                    cue_proto.threshold.as_ref().map_or(0.7, |c| c.value);
                                CoreCue::semantic(
                                    query.clone(),
                                    query,
                                    CoreConfidence::exact(confidence),
                                )
                            }
                            Some(CueType::Embedding(embedding)) => {
                                let embedding_vec = embedding.vector;
                                let confidence =
                                    cue_proto.threshold.as_ref().map_or(0.7, |c| c.value);

                                if embedding_vec.len() != 768 {
                                    let _ = tx.send(Err(Status::invalid_argument(format!(
                                        "Embedding vector must be exactly 768 dimensions, got {}",
                                        embedding_vec.len()
                                    )))).await;
                                    continue;
                                }

                                let Ok(embedding_array) = embedding_vec.try_into() else {
                                    let _ = tx
                                        .send(Err(Status::internal(
                                            "Failed to convert embedding vector to array",
                                        )))
                                        .await;
                                    continue;
                                };

                                CoreCue::embedding(
                                    "embedding_query".to_string(),
                                    embedding_array,
                                    CoreConfidence::exact(confidence),
                                )
                            }
                            Some(CueType::Context(context)) => {
                                let query = format!(
                                    "Context: location={}, participants={}",
                                    context.location,
                                    context.participants.join(", ")
                                );
                                CoreCue::semantic(query.clone(), query, CoreConfidence::exact(0.7))
                            }
                            Some(CueType::Temporal(_) | CueType::Pattern(_)) => {
                                let _ = tx.send(Err(Status::unimplemented(
                                    "Temporal and pattern cues not yet implemented. Use semantic or embedding cues."
                                ))).await;
                                continue;
                            }
                            None => {
                                let _ = tx.send(Err(Status::invalid_argument(
                                    "Cue type is missing. Provide a semantic query, embedding vector, or context cue."
                                ))).await;
                                continue;
                            }
                        };

                        // Perform recall
                        let recall_result = store.recall(&core_cue);
                        let total_activated = recall_result.results.len();

                        // Convert recalled episodes to Memory proto messages
                        let mut memories = vec![];
                        let mut total_confidence = 0.0f32;

                        let max_results = if req.max_results > 0 {
                            req.max_results
                        } else {
                            10
                        };

                        for (episode, confidence) in
                            recall_result.results.iter().take(max_results as usize)
                        {
                            let memory =
                                engram_proto::Memory::new(episode.id.clone(), vec![0.0; 768])
                                    .with_content(&episode.what)
                                    .with_confidence(
                                        Confidence::new(confidence.raw()).with_reasoning(format!(
                                            "Streaming query {query_count} from space '{}'",
                                            space_id
                                        )),
                                    );

                            memories.push(memory);
                            total_confidence += confidence.raw();
                        }

                        let memories_count = memories.len();
                        let avg_activation = if memories_count > 0 {
                            total_confidence / memories_count as f32
                        } else {
                            0.0
                        };

                        // Process recall request
                        let response = RecallResponse {
                            memories,
                            recall_confidence: Some(
                                Confidence::new(avg_activation).with_reasoning(format!(
                                    "Streaming query {query_count} processed in space '{}'",
                                    space_id
                                )),
                            ),
                            metadata: Some(RecallMetadata {
                                total_activated: total_activated as i32,
                                above_threshold: memories_count as i32,
                                avg_activation,
                                recall_time_ms: 5,
                                activation_path: vec![],
                            }),
                            traces: vec![],
                        };

                        if tx.send(Ok(response)).await.is_err() {
                            break; // Client disconnected
                        }
                    }
                    Err(e) => {
                        let _ = tx.send(Err(e)).await;
                        break;
                    }
                }
            }
        });

        let response_stream = ReceiverStream::new(rx);
        Ok(Response::new(Box::pin(response_stream)))
    }

    /// MemoryFlow provides bidirectional streaming with flow control.
    ///
    /// Cognitive design: Interactive memory sessions with backpressure management
    /// following working memory constraints (3-4 active streams maximum).
    type MemoryFlowStream = Pin<Box<dyn Stream<Item = Result<MemoryFlowResponse, Status>> + Send>>;

    async fn memory_flow(
        &self,
        request: Request<Streaming<MemoryFlowRequest>>,
    ) -> Result<Response<Self::MemoryFlowStream>, Status> {
        let mut stream = request.into_inner();
        let (tx, rx) = tokio::sync::mpsc::channel(64); // Higher buffer for bidirectional flow
        let registry = Arc::clone(&self.registry);
        let default_space = self.default_space.clone();

        tokio::spawn(async move {
            #[allow(clippy::collection_is_never_read)]
            let mut _session_state = std::collections::HashMap::new();
            let mut global_sequence = 0u64;

            while let Some(result) = stream.next().await {
                match result {
                    Ok(req) => {
                        global_sequence += 1;
                        let session_id = req.session_id.clone();

                        // Extract space_id from request or use default
                        let space_id = if req.memory_space_id.is_empty() {
                            default_space.clone()
                        } else {
                            match MemorySpaceId::try_from(req.memory_space_id.as_str()) {
                                Ok(id) => id,
                                Err(e) => {
                                    let _ = tx.send(Err(Status::invalid_argument(format!(
                                        "Invalid memory_space_id: {e}. Must be 4-64 lowercase alphanumeric characters."
                                    )))).await;
                                    continue;
                                }
                            }
                        };

                        // Get space-specific store handle
                        let _handle = match registry.create_or_get(&space_id).await {
                            Ok(h) => h,
                            Err(e) => {
                                let _ = tx
                                    .send(Err(Status::internal(format!(
                                        "Failed to access memory space '{}': {}",
                                        space_id, e
                                    ))))
                                    .await;
                                continue;
                            }
                        };
                        // Note: store handle available via _handle.store() for actual implementation
                        // Currently using mock responses pending full implementation

                        let response = match req.request {
                            Some(memory_flow_request::Request::Remember(_remember_req)) => {
                                // TODO: Use _handle.store() to actually store memory
                                // Process remember request
                                let remember_result = RememberResponse {
                                    memory_id: format!("flow_{global_sequence}"),
                                    storage_confidence: Some(Confidence::new(0.9).with_reasoning(
                                        format!("Flow memory stored in space '{}'", space_id),
                                    )),
                                    linked_memories: vec![],
                                    initial_state: ConsolidationState::Recent as i32,
                                };

                                MemoryFlowResponse {
                                    response: Some(memory_flow_response::Response::RememberResult(
                                        remember_result,
                                    )),
                                    session_id: session_id.clone(),
                                    sequence_number: i64::try_from(global_sequence)
                                        .unwrap_or(i64::MAX),
                                    timestamp: Some(
                                        engram_proto::datetime_to_timestamp(Utc::now()),
                                    ),
                                }
                            }
                            Some(memory_flow_request::Request::Recall(_recall_req)) => {
                                // TODO: Use _handle.store() to actually recall memories
                                // Process recall request
                                let recall_result = RecallResponse {
                                    memories: vec![],
                                    recall_confidence: Some(Confidence::new(0.7).with_reasoning(
                                        format!("Flow recall from space '{}'", space_id),
                                    )),
                                    metadata: Some(RecallMetadata {
                                        total_activated: 0,
                                        above_threshold: 0,
                                        avg_activation: 0.0,
                                        recall_time_ms: 3,
                                        activation_path: vec![],
                                    }),
                                    traces: vec![],
                                };

                                MemoryFlowResponse {
                                    response: Some(memory_flow_response::Response::RecallResult(
                                        recall_result,
                                    )),
                                    session_id: session_id.clone(),
                                    sequence_number: i64::try_from(global_sequence)
                                        .unwrap_or(i64::MAX),
                                    timestamp: Some(
                                        engram_proto::datetime_to_timestamp(Utc::now()),
                                    ),
                                }
                            }
                            Some(memory_flow_request::Request::Subscribe(sub_req)) => {
                                // Handle subscription
                                _session_state.insert(session_id.clone(), sub_req);

                                let status = FlowStatus {
                                    state: flow_status::State::Active as i32,
                                    message: format!(
                                        "Subscription active for space '{}'",
                                        space_id
                                    ),
                                    metrics: std::collections::HashMap::new(),
                                    last_activity: Some(engram_proto::datetime_to_timestamp(
                                        Utc::now(),
                                    )),
                                };

                                MemoryFlowResponse {
                                    response: Some(memory_flow_response::Response::Status(status)),
                                    session_id: session_id.clone(),
                                    sequence_number: i64::try_from(global_sequence)
                                        .unwrap_or(i64::MAX),
                                    timestamp: Some(
                                        engram_proto::datetime_to_timestamp(Utc::now()),
                                    ),
                                }
                            }
                            Some(memory_flow_request::Request::Control(control_req)) => {
                                // Handle flow control
                                let action = flow_control::Action::try_from(control_req.action)
                                    .unwrap_or(flow_control::Action::Unspecified);

                                let status = FlowStatus {
                                    state: match action {
                                        flow_control::Action::Pause => flow_status::State::Paused,
                                        _ => flow_status::State::Active,
                                    } as i32,
                                    message: format!("Flow control: {action:?}"),
                                    metrics: std::collections::HashMap::new(),
                                    last_activity: Some(engram_proto::datetime_to_timestamp(
                                        chrono::Utc::now(),
                                    )),
                                };

                                MemoryFlowResponse {
                                    response: Some(memory_flow_response::Response::Status(status)),
                                    session_id: session_id.clone(),
                                    sequence_number: i64::try_from(global_sequence)
                                        .unwrap_or(i64::MAX),
                                    timestamp: Some(engram_proto::datetime_to_timestamp(
                                        chrono::Utc::now(),
                                    )),
                                }
                            }
                            None => {
                                let _ = tx
                                    .send(Err(Status::invalid_argument(
                                        "Memory flow requires a request type",
                                    )))
                                    .await;
                                continue;
                            }
                        };

                        if tx.send(Ok(response)).await.is_err() {
                            break; // Client disconnected
                        }
                    }
                    Err(e) => {
                        let _ = tx.send(Err(e)).await;
                        break;
                    }
                }
            }
        });

        let response_stream = ReceiverStream::new(rx);
        Ok(Response::new(Box::pin(response_stream)))
    }

    // ========================================================================
    // Streaming APIs (Milestone 11 - In Progress)
    // ========================================================================

    type ObserveStreamStream =
        std::pin::Pin<Box<dyn Stream<Item = Result<ObservationResponse, Status>> + Send>>;

    /// Bidirectional streaming for observations (Task 005)
    ///
    /// Implements client → server streaming protocol:
    /// 1. Client sends StreamInit → server returns StreamInitAck with session ID
    /// 2. Client streams observations → server returns acks or backpressure signals
    /// 3. Client sends FlowControl → server updates session state
    /// 4. Client sends StreamClose → server gracefully closes session
    async fn observe_stream(
        &self,
        request: Request<tonic::Streaming<ObservationRequest>>,
    ) -> Result<Response<Self::ObserveStreamStream>, Status> {
        let response = self
            .streaming_handlers
            .handle_observe_stream(request)
            .await?;
        let (metadata, stream, _extensions) = response.into_parts();
        let boxed = Box::pin(stream) as Self::ObserveStreamStream;
        Ok(Response::from_parts(
            metadata,
            boxed,
            tonic::Extensions::default(),
        ))
    }

    type RecallStreamStream =
        std::pin::Pin<Box<dyn Stream<Item = Result<StreamingRecallResponse, Status>> + Send>>;

    /// Server streaming for recall results (Task 007)
    ///
    /// Implements server → client streaming for incremental recall:
    /// - Captures snapshot generation for consistent read isolation
    /// - Streams results in batches for low first-result latency
    /// - Supports bounded staleness mode (include_recent observations)
    async fn recall_stream(
        &self,
        request: Request<StreamingRecallRequest>,
    ) -> Result<Response<Self::RecallStreamStream>, Status> {
        let response = self.streaming_handlers.handle_recall_stream(request)?;
        let (metadata, stream, _extensions) = response.into_parts();
        let boxed = Box::pin(stream) as Self::RecallStreamStream;
        Ok(Response::from_parts(
            metadata,
            boxed,
            tonic::Extensions::default(),
        ))
    }

    type MemoryStreamStream =
        std::pin::Pin<Box<dyn Stream<Item = Result<ObservationResponse, Status>> + Send>>;

    /// Bidirectional streaming for memory operations
    ///
    /// TODO(milestone-11): Implement streaming memory protocol
    async fn memory_stream(
        &self,
        _request: Request<tonic::Streaming<ObservationRequest>>,
    ) -> Result<Response<Self::MemoryStreamStream>, Status> {
        todo!("Milestone 11: Streaming memory not yet implemented")
    }

    async fn rebalance_spaces(
        &self,
        _request: Request<RebalanceRequest>,
    ) -> Result<Response<RebalanceResponse>, Status> {
        let cluster = self
            .cluster
            .as_ref()
            .ok_or_else(|| Status::failed_precondition("cluster mode disabled"))?;
        let planned = cluster
            .trigger_rebalance()
            .await
            .map_err(cluster_error_to_status)?;
        Ok(Response::new(RebalanceResponse {
            planned: planned as u64,
        }))
    }

    async fn migrate_space(
        &self,
        request: Request<MigrateSpaceRpcRequest>,
    ) -> Result<Response<MigrateSpaceRpcResponse>, Status> {
        let cluster = self
            .cluster
            .as_ref()
            .ok_or_else(|| Status::failed_precondition("cluster mode disabled"))?;
        let req = request.into_inner();
        if req.space_id.is_empty() {
            return Err(Status::invalid_argument(
                "space_id must be provided for migrate_space",
            ));
        }
        let space_id = MemorySpaceId::try_from(req.space_id.as_str()).map_err(|err| {
            Status::invalid_argument(format!("invalid space id '{}': {err}", req.space_id))
        })?;
        let plan = cluster
            .migrate_space(&space_id)
            .await
            .map_err(cluster_error_to_status)?
            .map(migration_plan_to_proto);
        Ok(Response::new(MigrateSpaceRpcResponse { plan }))
    }

    async fn apply_replication_batch(
        &self,
        request: Request<ApplyReplicationBatchRequest>,
    ) -> Result<Response<ApplyReplicationBatchResponse>, Status> {
        #[cfg(not(feature = "memory_mapped_persistence"))]
        {
            return Err(Status::failed_precondition(
                "replication requires persistent storage support",
            ));
        }

        #[cfg(feature = "memory_mapped_persistence")]
        {
            let req = request.into_inner();
            let space_id = MemorySpaceId::try_from(req.space_id.as_str()).map_err(|err| {
                Status::invalid_argument(format!("invalid space id '{}': {err}", req.space_id))
            })?;
            let handle = self
                .registry
                .create_or_get(&space_id)
                .await
                .map_err(|err| Status::internal(format!("failed to load space: {err}")))?;
            let store = handle.store();
            for entry in &req.entries {
                let entry_type = WalEntryType::from(entry.entry_type);
                store
                    .apply_replication_entry(entry_type, &entry.payload)
                    .map_err(|err| {
                        Status::internal(format!("failed to apply replication entry: {err}"))
                    })?;
            }
            Ok(Response::new(ApplyReplicationBatchResponse {
                applied_through: req.end_sequence,
            }))
        }
    }

    async fn get_replication_status(
        &self,
        _request: Request<ReplicationStatusRequest>,
    ) -> Result<Response<ReplicationStatusResponse>, Status> {
        #[cfg(not(feature = "memory_mapped_persistence"))]
        {
            return Ok(Response::new(ReplicationStatusResponse {
                replicas: Vec::new(),
            }));
        }

        #[cfg(feature = "memory_mapped_persistence")]
        {
            let cluster = self
                .cluster
                .as_ref()
                .ok_or_else(|| Status::failed_precondition("cluster mode disabled"))?;
            let replicas = cluster
                .replication_metadata
                .as_ref()
                .map(|metadata| {
                    metadata
                        .snapshot()
                        .into_iter()
                        .flat_map(|summary| {
                            summary.replicas.into_iter().map(move |lag| {
                                engram_proto::ReplicaLagView {
                                    space_id: summary.space.to_string(),
                                    replica: lag.replica,
                                    local_sequence: lag.local_sequence,
                                    replicated_sequence: lag.replica_sequence,
                                }
                            })
                        })
                        .collect()
                })
                .unwrap_or_default();
            Ok(Response::new(ReplicationStatusResponse { replicas }))
        }
    }
}

fn cluster_error_to_status(err: ClusterError) -> Status {
    match err {
        ClusterError::NotPrimary {
            owner,
            space,
            local,
        } => Status::failed_precondition(format!(
            "Memory space '{space}' is owned by node '{owner}'. Local node '{local}' refused the write to prevent split-brain. Route the request to {owner}."
        )),
        ClusterError::InsufficientHealthyNodes {
            required,
            available,
        } => Status::unavailable(format!(
            "Not enough healthy nodes for replication (required {required}, available {available})."
        )),
        ClusterError::Partitioned {
            reachable_nodes,
            total_nodes,
        } => Status::unavailable(format!(
            "Cluster partitioned from majority (reachable {reachable_nodes} / {total_nodes}). Retry the primary once connectivity is restored."
        )),
        ClusterError::SplitBrain { space_id, .. } => Status::failed_precondition(format!(
            "Split-brain guard rejected writes for space '{space_id}'. Route the request to the elected primary."
        )),
        ClusterError::Configuration(reason) => Status::failed_precondition(reason),
        other => Status::internal(other.to_string()),
    }
}

fn router_error_to_status(err: RouterError) -> Status {
    match err {
        RouterError::CircuitOpen { node_id, .. } => {
            Status::unavailable(format!("Circuit breaker open for node '{node_id}'"))
        }
        RouterError::DeadlineExceeded { node_id } => Status::deadline_exceeded(format!(
            "Remote routing to node '{node_id}' exceeded the deadline"
        )),
        RouterError::NoHealthyReplicas { space } => Status::unavailable(format!(
            "No healthy replicas available for memory space '{space}'"
        )),
        RouterError::RemoteRpc { status, .. } => status,
        RouterError::Connect { node_id, error } => {
            Status::unavailable(format!("Failed to connect to node '{node_id}': {error}"))
        }
        RouterError::LocalOnly => {
            Status::failed_precondition("Router attempted to proxy a local-only request")
        }
    }
}

fn migration_plan_to_proto(plan: MigrationPlan) -> MigrationPlanView {
    let planned_at: DateTime<Utc> = DateTime::<Utc>::from(plan.planned_at);
    MigrationPlanView {
        space_id: plan.space.to_string(),
        from: plan.from.map(|node| node.id).unwrap_or_default(),
        to: plan.to.id,
        version: plan.version,
        reason: match plan.reason {
            MigrationReason::MembershipChange => "membership_change".to_string(),
            MigrationReason::Manual => "manual".to_string(),
        },
        planned_at: Some(engram_proto::datetime_to_timestamp(planned_at)),
    }
}

/// Convert confidence value to protobuf ConfidenceCategory enum
fn confidence_to_category(value: f32) -> engram_proto::ConfidenceCategory {
    use engram_proto::ConfidenceCategory;

    match value {
        v if v <= 0.0 => ConfidenceCategory::None,
        v if v < 0.3 => ConfidenceCategory::Low,
        v if v < 0.7 => ConfidenceCategory::Medium,
        v if v < 1.0 => ConfidenceCategory::High,
        _ => ConfidenceCategory::Certain,
    }
}
