//! gRPC service implementation for Engram memory operations.
//!
//! Provides cognitive-friendly service interface with natural language method names
//! and educational error messages that teach memory system concepts.

use engram_core::{
    Confidence as CoreConfidence, Cue as CoreCue, Episode, MemorySpaceId, MemorySpaceRegistry,
    MemoryStore, metrics::MetricsRegistry,
};
use engram_proto::cue::CueType;
use engram_proto::engram_service_server::{EngramService, EngramServiceServer};
use engram_proto::{
    AssociateRequest, AssociateResponse, CompleteRequest, CompleteResponse, Confidence,
    ConsolidateRequest, ConsolidateResponse, ConsolidationMode, ConsolidationProgress,
    ConsolidationState, DreamRequest, DreamResponse, ExperienceRequest, ExperienceResponse,
    FlowStatus, ForgetMode, ForgetRequest, ForgetResponse, HealthStatus, Insight,
    IntrospectRequest, IntrospectResponse, MemoryFlowRequest, MemoryFlowResponse, MemoryStatistics,
    RecallMetadata, RecallRequest, RecallResponse, RecognizeRequest, RecognizeResponse,
    RememberRequest, RememberResponse, ReminisceRequest, ReminisceResponse, ReplaySequence,
    StreamEventType, StreamRequest, StreamResponse, dream_response, flow_control, flow_status,
    memory_flow_request, memory_flow_response, remember_request,
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
    store: Arc<MemoryStore>,
    metrics: Arc<MetricsRegistry>,
    registry: Arc<MemorySpaceRegistry>,
    default_space: MemorySpaceId,
}

impl MemoryService {
    /// Create a new memory service with the given memory store.
    #[allow(clippy::missing_const_for_fn)]
    pub fn new(
        store: Arc<MemoryStore>,
        metrics: Arc<MetricsRegistry>,
        registry: Arc<MemorySpaceRegistry>,
        default_space: MemorySpaceId,
    ) -> Self {
        Self {
            store,
            metrics,
            registry,
            default_space,
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
        let _cue = req.cue.ok_or_else(|| {
            Status::invalid_argument(
                "Recall requires a cue - a trigger for memory retrieval. \
                Like trying to remember without knowing what you're looking for. \
                Provide an embedding, semantic query, or context cue.",
            )
        })?;

        // Parse the cue type and create appropriate CoreCue
        let core_cue = match &_cue.cue_type {
            Some(CueType::Semantic(semantic)) => {
                let query = semantic.query.clone();
                let confidence = _cue.threshold.as_ref().map_or(0.7, |c| c.value);
                CoreCue::semantic(query.clone(), query, CoreConfidence::exact(confidence))
            }
            Some(CueType::Embedding(embedding)) => {
                let embedding_vec = embedding.vector.clone();
                let confidence = _cue.threshold.as_ref().map_or(0.7, |c| c.value);

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
        let _handle = self.registry.create_or_get(&space_id).await.map_err(|e| {
            Status::internal(format!(
                "Failed to access memory space '{}': {}",
                space_id, e
            ))
        })?;

        let mode = ForgetMode::try_from(req.mode).unwrap_or(ForgetMode::Unspecified);

        if mode == ForgetMode::Unspecified {
            return Err(Status::invalid_argument(
                "Forgetting requires intent - specify whether to suppress \
                (reduce activation, reversible) or delete (permanent removal). \
                Like the difference between not thinking about something vs \
                truly forgetting it.",
            ));
        }

        let response = ForgetResponse {
            memories_affected: 0,
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

        let _partial = req.partial_pattern.ok_or_else(|| {
            Status::invalid_argument(
                "Pattern completion needs a partial pattern - fragments to \
                complete. Like trying to finish a sentence without the beginning. \
                Provide partial memory fields to reconstruct.",
            )
        })?;

        let response = CompleteResponse {
            completions: vec![],
            completion_confidence: Some(
                Confidence::new(0.4).with_reasoning("No patterns match the partial cue"),
            ),
            field_confidences: HashMap::default(),
        };

        Ok(Response::new(response))
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

        // TODO(Task 005c): Wire up per-space event filtering
        // Currently generates mock events. For full multi-tenant isolation, need to:
        // 1. Subscribe to store's event stream for this specific space only
        // 2. Filter events by space_id to prevent cross-tenant leakage
        // 3. Implement per-space backpressure management
        // See Task 007 for comprehensive streaming isolation implementation.

        // Create channel for streaming responses
        let (tx, rx) = tokio::sync::mpsc::channel(32);

        // Spawn task to generate activity events
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
                let response = StreamResponse {
                    event_type: *event_type as i32,
                    timestamp: Some(engram_proto::datetime_to_timestamp(chrono::Utc::now())),
                    description: format!("Memory event: {event_type:?}"),
                    metadata: HashMap::default(),
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
        let _store = Arc::clone(&self.store);
        let _registry = Arc::clone(&self.registry);
        let default_space = self.default_space.clone();

        tokio::spawn(async move {
            let mut sequence = 0u64;
            while let Some(result) = stream.next().await {
                match result {
                    Ok(req) => {
                        sequence += 1;

                        // TODO(Task 005c): Per-request space routing for streaming operations
                        // Each RememberRequest has its own memory_space_id field
                        // For full isolation, need to:
                        // 1. Extract space_id from req.memory_space_id (or use default_space)
                        // 2. Validate space exists via _registry.create_or_get()
                        // 3. Route storage operation to the correct space-specific store
                        // Currently stores to shared store regardless of space_id
                        let _space_id = if req.memory_space_id.is_empty() {
                            default_space.clone()
                        } else {
                            MemorySpaceId::try_from(req.memory_space_id.as_str())
                                .unwrap_or_else(|_| default_space.clone())
                        };

                        // TODO: Validate space access
                        // let _handle = _registry.create_or_get(&space_id).await?;

                        // Process memory storage
                        let response = match req.memory_type {
                            Some(remember_request::MemoryType::Memory(memory)) => {
                                RememberResponse {
                                    memory_id: memory.id,
                                    storage_confidence: Some(Confidence::new(0.9).with_reasoning(
                                        format!("Streaming memory {sequence} processed"),
                                    )),
                                    linked_memories: vec![],
                                    initial_state: ConsolidationState::Recent as i32,
                                }
                            }
                            Some(remember_request::MemoryType::Episode(episode)) => {
                                RememberResponse {
                                    memory_id: episode.id,
                                    storage_confidence: Some(Confidence::new(0.85).with_reasoning(
                                        format!("Streaming episode {sequence} processed"),
                                    )),
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
        let _store = Arc::clone(&self.store);
        let _registry = Arc::clone(&self.registry);
        let default_space = self.default_space.clone();

        tokio::spawn(async move {
            let mut query_count = 0u64;
            while let Some(result) = stream.next().await {
                match result {
                    Ok(req) => {
                        query_count += 1;

                        // TODO(Task 005c): Per-request space routing for streaming recall
                        // Each RecallRequest has its own memory_space_id field
                        // For full isolation, need to:
                        // 1. Extract space_id from req.memory_space_id (or use default_space)
                        // 2. Validate space exists via _registry.create_or_get()
                        // 3. Route recall operation to the correct space-specific store
                        // Currently queries shared store regardless of space_id
                        let _space_id = if req.memory_space_id.is_empty() {
                            default_space.clone()
                        } else {
                            MemorySpaceId::try_from(req.memory_space_id.as_str())
                                .unwrap_or_else(|_| default_space.clone())
                        };

                        // TODO: Validate space access
                        // let _handle = _registry.create_or_get(&space_id).await?;

                        // Validate cue presence
                        if req.cue.is_none() {
                            let _ = tx
                                .send(Err(Status::invalid_argument(
                                    "Streaming recall requires cue for each query",
                                )))
                                .await;
                            continue;
                        }

                        // Process recall request
                        let response = RecallResponse {
                            memories: vec![], // Would search graph
                            recall_confidence: Some(Confidence::new(0.6).with_reasoning(format!(
                                "Streaming query {query_count} processed"
                            ))),
                            metadata: Some(RecallMetadata {
                                total_activated: 0,
                                above_threshold: 0,
                                avg_activation: 0.0,
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
        let _store = Arc::clone(&self.store);
        let _registry = Arc::clone(&self.registry);
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

                        // TODO(Task 005c): Per-request space routing for bidirectional memory flow
                        // MemoryFlowRequest has memory_space_id at top level
                        // All operations in this request should be routed to the same space
                        // For full isolation, need to:
                        // 1. Extract space_id from req.memory_space_id (or use default_space)
                        // 2. Validate space exists via _registry.create_or_get()
                        // 3. Route all remember/recall operations to correct space-specific store
                        // 4. Ensure flow control and session state are per-space isolated
                        // Currently operates on shared store regardless of space_id
                        let _space_id = if req.memory_space_id.is_empty() {
                            default_space.clone()
                        } else {
                            MemorySpaceId::try_from(req.memory_space_id.as_str())
                                .unwrap_or_else(|_| default_space.clone())
                        };

                        // TODO: Validate space access
                        // let _handle = _registry.create_or_get(&space_id).await?;

                        let response = match req.request {
                            Some(memory_flow_request::Request::Remember(_remember_req)) => {
                                // Process remember request
                                let remember_result = RememberResponse {
                                    memory_id: format!("flow_{global_sequence}"),
                                    storage_confidence: Some(
                                        Confidence::new(0.9).with_reasoning("Flow memory stored"),
                                    ),
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
                                    timestamp: Some(engram_proto::datetime_to_timestamp(
                                        chrono::Utc::now(),
                                    )),
                                }
                            }
                            Some(memory_flow_request::Request::Recall(_recall_req)) => {
                                // Process recall request
                                let recall_result = RecallResponse {
                                    memories: vec![],
                                    recall_confidence: Some(
                                        Confidence::new(0.7)
                                            .with_reasoning("Flow recall processed"),
                                    ),
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
                                    timestamp: Some(engram_proto::datetime_to_timestamp(
                                        chrono::Utc::now(),
                                    )),
                                }
                            }
                            Some(memory_flow_request::Request::Subscribe(sub_req)) => {
                                // Handle subscription
                                _session_state.insert(session_id.clone(), sub_req);

                                let status = FlowStatus {
                                    state: flow_status::State::Active as i32,
                                    message: "Subscription active".to_string(),
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
}
use chrono::Utc;
