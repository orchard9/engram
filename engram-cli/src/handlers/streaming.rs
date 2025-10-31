//! Streaming gRPC handlers for continuous memory operations.
//!
//! Implements the core streaming protocol handlers for:
//! - ObserveStream: Client → server streaming for continuous observations
//! - RecallStream: Server → client streaming for incremental recall results
//! - MemoryFlow: Bidirectional streaming for combined observe/recall workflows
//!
//! ## Architecture
//!
//! The streaming handlers act as the gRPC-to-core translation layer,
//! managing session lifecycle, sequence validation, and flow control while
//! delegating actual memory operations to the MemoryStore.
//!
//! ```text
//! gRPC Client
//!      ↓
//! StreamingHandlers (this module)
//!      ↓
//! SessionManager + ObservationQueue
//!      ↓
//! MemoryStore (indexing, recall)
//! ```

use engram_core::{
    Confidence as CoreConfidence, Episode, MemorySpaceId, MemoryStore,
    streaming::{
        ObservationPriority, ObservationQueue, QueueError, SessionError, SessionManager,
        SessionState,
    },
};
use engram_proto::{
    ObservationAck, ObservationRequest, ObservationResponse, StreamCapabilities, StreamInitAck,
    StreamStatus, StreamingRecallRequest, StreamingRecallResponse,
    observation_ack::Status as AckStatus, observation_request, observation_response,
    stream_status::State as StreamState,
};
use std::sync::Arc;
use std::time::SystemTime;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tokio_stream::{Stream, StreamExt};
use tonic::{Request, Response, Status, Streaming};

/// Streaming gRPC handlers with session and queue management.
///
/// Stateless request handler that delegates to:
/// - `SessionManager` for session lifecycle and sequence validation
/// - `ObservationQueue` for lock-free observation queuing
/// - `MemoryStore` for memory operations
#[allow(dead_code)] // store will be used in Task 007 for recall
pub struct StreamingHandlers {
    /// Session manager for tracking active streaming sessions
    session_manager: Arc<SessionManager>,

    /// Observation queue for lock-free enqueue operations
    observation_queue: Arc<ObservationQueue>,

    /// Memory store for recall operations
    store: Arc<MemoryStore>,
}

impl StreamingHandlers {
    /// Create new streaming handlers.
    ///
    /// # Arguments
    ///
    /// * `session_manager` - Session lifecycle manager
    /// * `observation_queue` - Lock-free observation queue
    /// * `store` - Memory store for recall operations
    #[must_use]
    pub const fn new(
        session_manager: Arc<SessionManager>,
        observation_queue: Arc<ObservationQueue>,
        store: Arc<MemoryStore>,
    ) -> Self {
        Self {
            session_manager,
            observation_queue,
            store,
        }
    }

    /// Handle ObserveStream RPC: client → server streaming.
    ///
    /// Protocol flow:
    /// 1. Client sends `StreamInit` → server returns `StreamInitAck`
    /// 2. Client sends `Observation`s → server returns `ObservationAck`s
    /// 3. Client sends `FlowControl` → server updates session state
    /// 4. Client sends `StreamClose` → server gracefully closes
    ///
    /// # Errors
    ///
    /// Returns gRPC `Status` error on:
    /// - Invalid session state
    /// - Sequence validation failure
    /// - Queue capacity exceeded
    #[allow(clippy::unused_async)] // async required for spawned task
    pub async fn handle_observe_stream(
        &self,
        request: Request<Streaming<ObservationRequest>>,
    ) -> Result<
        Response<impl Stream<Item = Result<ObservationResponse, Status>> + Send + 'static>,
        Status,
    > {
        let mut in_stream = request.into_inner();
        let (tx, rx) = mpsc::channel::<Result<ObservationResponse, Status>>(128);

        // Clone Arc references for spawned task
        let session_manager = Arc::clone(&self.session_manager);
        let observation_queue = Arc::clone(&self.observation_queue);

        // Spawn handler task to process incoming stream
        tokio::spawn(async move {
            let mut session_id: Option<String> = None;

            while let Some(result) = in_stream.next().await {
                let req = match result {
                    Ok(r) => r,
                    Err(e) => {
                        let _ = tx
                            .send(Err(Status::internal(format!("Stream error: {e}"))))
                            .await;
                        break;
                    }
                };

                // Route based on operation type
                match req.operation {
                    Some(observation_request::Operation::Init(_init)) => {
                        // Handle stream initialization
                        match Self::handle_stream_init(&session_manager, &observation_queue, &req) {
                            Ok(response) => {
                                // Extract session ID from response for subsequent operations
                                if let Some(observation_response::Result::InitAck(ref ack)) =
                                    response.result
                                {
                                    session_id = Some(ack.session_id.clone());
                                }

                                if tx.send(Ok(response)).await.is_err() {
                                    break;
                                }
                            }
                            Err(e) => {
                                let _ = tx.send(Err(e)).await;
                                break;
                            }
                        }
                    }

                    Some(observation_request::Operation::Observation(ref episode)) => {
                        // Handle observation
                        match Self::handle_observation(
                            &session_manager,
                            &observation_queue,
                            session_id.as_ref(),
                            &req,
                            episode,
                        ) {
                            Ok(response) => {
                                if tx.send(Ok(response)).await.is_err() {
                                    break;
                                }
                            }
                            Err(e) => {
                                // Send error but continue stream (non-fatal)
                                let _ = tx.send(Err(e)).await;
                            }
                        }
                    }

                    Some(observation_request::Operation::Flow(ref flow)) => {
                        // Handle flow control
                        Self::handle_flow_control(&session_manager, session_id.as_ref(), flow);
                    }

                    Some(observation_request::Operation::Close(_close)) => {
                        // Handle graceful close
                        Self::handle_stream_close(&session_manager, session_id.as_ref());
                        break;
                    }

                    None => {
                        let _ = tx
                            .send(Err(Status::invalid_argument(
                                "Empty operation in ObservationRequest",
                            )))
                            .await;
                    }
                }
            }

            // Cleanup: close session on stream end
            if let Some(sid) = session_id {
                let _ = session_manager.close_session(&sid);
            }
        });

        Ok(Response::new(ReceiverStream::new(rx)))
    }

    /// Handle stream initialization: create session and return capabilities.
    ///
    /// # Errors
    ///
    /// Returns `Status` error on invalid memory space ID
    #[allow(clippy::result_large_err)] // tonic::Status is large but idiomatic for gRPC
    fn handle_stream_init(
        session_manager: &SessionManager,
        observation_queue: &ObservationQueue,
        req: &ObservationRequest,
    ) -> Result<ObservationResponse, Status> {
        // Parse memory space ID
        let memory_space_id = MemorySpaceId::try_from(req.memory_space_id.as_str())
            .map_err(|e| Status::invalid_argument(format!("Invalid memory_space_id: {e}")))?;

        // Create session with server-generated ID
        let session_id = uuid::Uuid::new_v4().to_string();
        let session = session_manager.create_session(session_id, memory_space_id);

        // Build capabilities response
        let capabilities = StreamCapabilities {
            max_observations_per_second: 100_000,
            queue_capacity: observation_queue.total_capacity() as u32,
            supports_backpressure: true,
            supports_snapshot_isolation: true,
        };

        let init_ack = StreamInitAck {
            session_id: session.session_id().to_string(),
            initial_sequence: 0,
            capabilities: Some(capabilities),
        };

        Ok(ObservationResponse {
            result: Some(observation_response::Result::InitAck(init_ack)),
            session_id: session.session_id().to_string(),
            sequence_number: 0,
            server_timestamp: Some(prost_types::Timestamp::from(SystemTime::now())),
        })
    }

    /// Handle observation: validate sequence and enqueue for processing.
    ///
    /// # Errors
    ///
    /// Returns `Status` error on:
    /// - No active session
    /// - Sequence validation failure
    /// - Invalid episode data
    #[allow(clippy::too_many_lines, clippy::result_large_err)] // tonic::Status is large but idiomatic for gRPC
    fn handle_observation(
        session_manager: &SessionManager,
        observation_queue: &ObservationQueue,
        session_id: Option<&String>,
        req: &ObservationRequest,
        episode_proto: &engram_proto::Episode,
    ) -> Result<ObservationResponse, Status> {
        // Validate session exists
        let Some(sid) = session_id else {
            return Err(Status::failed_precondition(
                "Must send StreamInit before observations",
            ));
        };

        let session = session_manager
            .get_session(sid)
            .map_err(Self::map_session_error)?;

        // Validate sequence number
        session
            .validate_sequence(req.sequence_number)
            .map_err(Self::map_session_error)?;

        // Convert proto Episode to core Episode
        let embedding =
            episode_proto.embedding.clone().try_into().map_err(|_| {
                Status::invalid_argument("Embedding must be exactly 768 dimensions")
            })?;

        let confidence = CoreConfidence::exact(
            episode_proto
                .encoding_confidence
                .as_ref()
                .map_or(0.7, |c| c.value),
        );

        let when = episode_proto
            .when
            .as_ref()
            .map_or_else(chrono::Utc::now, |ts| {
                chrono::DateTime::from_timestamp(ts.seconds, ts.nanos as u32)
                    .unwrap_or_else(chrono::Utc::now)
            });

        let episode_core = Episode::new(
            episode_proto.id.clone(),
            when,
            episode_proto.what.clone(),
            embedding,
            confidence,
        );

        // Enqueue observation
        match observation_queue.enqueue(
            session.memory_space_id().clone(),
            episode_core,
            req.sequence_number,
            ObservationPriority::Normal,
        ) {
            Ok(()) => {
                // Observation accepted - send ack
                let ack = ObservationAck {
                    status: AckStatus::Accepted as i32,
                    memory_id: format!("mem_{}", req.sequence_number),
                    committed_at: Some(prost_types::Timestamp::from(SystemTime::now())),
                };

                Ok(ObservationResponse {
                    result: Some(observation_response::Result::Ack(ack)),
                    session_id: sid.clone(),
                    sequence_number: req.sequence_number,
                    server_timestamp: Some(prost_types::Timestamp::from(SystemTime::now())),
                })
            }
            Err(QueueError::OverCapacity {
                current,
                limit,
                priority,
            }) => {
                // Queue at capacity - send backpressure status
                let status = StreamStatus {
                    state: StreamState::Backpressure as i32,
                    message: format!(
                        "Queue capacity exceeded for {priority:?} priority: {current}/{limit} items"
                    ),
                    queue_depth: current as u32,
                    queue_capacity: limit as u32,
                    pressure: current as f32 / limit as f32,
                };

                Ok(ObservationResponse {
                    result: Some(observation_response::Result::Status(status)),
                    session_id: sid.clone(),
                    sequence_number: req.sequence_number,
                    server_timestamp: Some(prost_types::Timestamp::from(SystemTime::now())),
                })
            }
        }
    }

    /// Handle flow control: update session state.
    fn handle_flow_control(
        session_manager: &SessionManager,
        session_id: Option<&String>,
        flow: &engram_proto::FlowControl,
    ) {
        use engram_proto::flow_control::Action;

        let Some(sid) = session_id else {
            return;
        };
        let Ok(session) = session_manager.get_session(sid) else {
            return;
        };
        match flow.action() {
            Action::Pause => {
                session.set_state(SessionState::Paused);
                tracing::info!("Session {sid} paused by client");
            }
            Action::Resume => {
                session.set_state(SessionState::Active);
                tracing::info!("Session {sid} resumed by client");
            }
            _ => {
                // Other actions not yet implemented
            }
        }
    }

    /// Handle stream close: mark session as closed.
    fn handle_stream_close(session_manager: &SessionManager, session_id: Option<&String>) {
        if let Some(sid) = session_id {
            let _ = session_manager.close_session(sid);
            tracing::info!("Session {sid} closed gracefully");
        }
    }

    /// Handle RecallStream RPC: server → client streaming (Task 007).
    ///
    /// This is a placeholder for Task 007 implementation.
    #[allow(
        unused_variables,
        clippy::unused_self,
        clippy::unnecessary_wraps,
        clippy::needless_pass_by_value,
        clippy::result_large_err
    )] // tonic::Status is large but idiomatic for gRPC
    pub fn handle_recall_stream(
        &self,
        request: Request<StreamingRecallRequest>,
    ) -> Result<
        Response<impl Stream<Item = Result<StreamingRecallResponse, Status>> + Send + 'static>,
        Status,
    > {
        // TODO: Implement in Task 007
        let (tx, rx) = mpsc::channel::<Result<StreamingRecallResponse, Status>>(1);
        Ok(Response::new(ReceiverStream::new(rx)))
    }

    /// Map `SessionError` to gRPC `Status`.
    fn map_session_error(err: SessionError) -> Status {
        match err {
            SessionError::NotFound { session_id } => {
                Status::not_found(format!("Session not found: {session_id}"))
            }
            SessionError::SequenceMismatch {
                expected,
                received,
                session_id,
            } => Status::invalid_argument(format!(
                "Sequence mismatch in session {session_id}: expected {expected}, got {received}"
            )),
            SessionError::InvalidState { reason, session_id } => Status::failed_precondition(
                format!("Invalid state for session {session_id}: {reason}"),
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use engram_core::streaming::QueueConfig;

    fn setup_test_handlers() -> StreamingHandlers {
        let session_manager = Arc::new(SessionManager::new());
        let observation_queue = Arc::new(ObservationQueue::new(QueueConfig::default()));
        // Create test store with reasonable max_memories limit
        let store = Arc::new(MemoryStore::new(1000));

        StreamingHandlers::new(session_manager, observation_queue, store)
    }

    #[tokio::test]
    async fn test_stream_init_returns_session_id() {
        let handlers = setup_test_handlers();
        let req = ObservationRequest {
            memory_space_id: "default".to_string(),
            operation: Some(observation_request::Operation::Init(
                engram_proto::StreamInit {
                    client_buffer_size: 1000,
                    enable_backpressure: true,
                    max_batch_size: 100,
                },
            )),
            session_id: String::new(),
            sequence_number: 0,
        };

        let response = StreamingHandlers::handle_stream_init(
            &handlers.session_manager,
            &handlers.observation_queue,
            &req,
        )
        .unwrap();

        // Verify InitAck response
        assert!(matches!(
            response.result,
            Some(observation_response::Result::InitAck(_))
        ));

        if let Some(observation_response::Result::InitAck(ack)) = response.result {
            assert!(!ack.session_id.is_empty());
            assert_eq!(ack.initial_sequence, 0);
            assert!(ack.capabilities.is_some());
        }
    }

    #[tokio::test]
    async fn test_observation_requires_init() {
        let handlers = setup_test_handlers();
        let session_id = None; // No session initialized

        let episode = engram_proto::Episode::default();
        let req = ObservationRequest {
            memory_space_id: "default".to_string(),
            operation: Some(observation_request::Operation::Observation(episode.clone())),
            session_id: String::new(),
            sequence_number: 1,
        };

        let result = StreamingHandlers::handle_observation(
            &handlers.session_manager,
            &handlers.observation_queue,
            session_id.as_ref(),
            &req,
            &episode,
        );

        assert!(result.is_err());
        assert_eq!(result.unwrap_err().code(), tonic::Code::FailedPrecondition);
    }
}
