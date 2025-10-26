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

pub mod session;

pub use session::{SessionError, SessionManager, SessionState, StreamSession};
