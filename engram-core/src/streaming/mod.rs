//! Streaming protocol for continuous memory observation and recall.
//!
//! This module implements the foundation for Milestone 11's high-performance
//! streaming interface, enabling 100K+ observations/second with bounded staleness
//! consistency.
//!
//! ## Design Principles
//!
//! - **Client-generated monotonic sequences**: No network round-trip for coordination
//! - **Server-validated monotonicity**: Rejects gaps/duplicates for correctness
//! - **Eventual consistency**: Bounded staleness target P99 < 100ms
//! - **Biological inspiration**: Matches hippocampal-neocortical asynchrony
//!
//! ## Session Lifecycle
//!
//! 1. **Init**: Client sends `StreamInit`, server returns session ID + capabilities
//! 2. **Active**: Client streams observations with monotonic sequence numbers
//! 3. **Pause**: Client sends `FlowControl::ACTION_PAUSE`, server stops processing
//! 4. **Resume**: Client sends `FlowControl::ACTION_RESUME`, server resumes
//! 5. **Close**: Client sends `StreamClose`, server drains queue and closes
//!
//! ## Research Foundation
//!
//! Based on:
//! - Buzsaki, G. (2015). Hippocampal sharp wave-ripple: A cognitive biomarker
//! - Marr, D. (1971). Simple memory: a theory for archicortex
//! - Lamport, L. (1978). Time, clocks, and the ordering of events

pub mod backpressure;
pub mod observation_queue;
pub mod queue_metrics;
pub mod recall;
pub mod session;
pub mod space_isolated_hnsw;
pub mod stream_metrics;
pub mod worker_pool;

pub use backpressure::{BackpressureMonitor, BackpressureState, calculate_retry_after};
pub use observation_queue::{
    ObservationPriority, ObservationQueue, QueueConfig, QueueDepths, QueueError, QueueMetrics,
    QueuedObservation,
};
pub use queue_metrics::{QueueMetricsTracker, QueueStatistics};
pub use recall::{IncrementalRecallStream, RecallBatchItem, RecallError, SnapshotRecallConfig};
pub use session::{SessionError, SessionManager, SessionState, StreamSession};
pub use space_isolated_hnsw::{SpaceHnswError, SpaceIsolatedHnsw};
pub use stream_metrics::{
    record_backpressure_activation, record_batch_failure, record_batch_processed,
    record_batch_size, record_observation_latency, record_observation_processed,
    record_observation_rejected, record_queue_wait_time, record_recall_latency, record_work_stolen,
    register_all_metrics, update_active_sessions, update_backpressure_state, update_queue_depth,
    update_worker_utilization,
};
pub use worker_pool::{WorkerPool, WorkerPoolConfig, WorkerPoolError, WorkerStats};
