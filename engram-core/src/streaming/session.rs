//! Session management for streaming protocol with monotonic sequence validation.
//!
//! Implements the core session lifecycle and sequence number protocol that guarantees
//! intra-stream total ordering while allowing cross-stream undefined ordering
//! (matching biological memory formation dynamics).

use crate::types::MemorySpaceId;
use dashmap::DashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU8, AtomicU64, Ordering};
use std::time::{Duration, Instant};
use thiserror::Error;

/// Session timeout duration (5 minutes idle - matches biological working memory decay)
const DEFAULT_SESSION_TIMEOUT: Duration = Duration::from_secs(300);

/// Session states following the lifecycle: Init → Active → Paused → Closed
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum SessionState {
    /// Session is active and accepting observations
    Active = 0,
    /// Session is paused by client request
    Paused = 1,
    /// Session is closed
    Closed = 2,
}

impl From<u8> for SessionState {
    fn from(value: u8) -> Self {
        match value {
            1 => Self::Paused,
            2 => Self::Closed,
            _ => Self::Active, // Default to active for 0 or invalid values
        }
    }
}

/// Stream session with monotonic sequence tracking and lifecycle management.
///
/// Each session maintains:
/// - Session ID (client-generated or server-assigned)
/// - Memory space ID for tenant isolation
/// - Last sequence number (atomically updated for validation)
/// - Session state (active, paused, closed)
/// - Activity timestamps for timeout detection
#[derive(Debug)]
pub struct StreamSession {
    /// Unique session identifier
    session_id: String,

    /// Memory space this session operates on
    memory_space_id: MemorySpaceId,

    /// Last sequence number received (monotonically increasing)
    last_sequence: AtomicU64,

    /// Session creation time
    created_at: Instant,

    /// Last activity timestamp (as Unix nanos, for atomic operations)
    last_activity: AtomicU64,

    /// Current session state
    state: AtomicU8,
}

impl StreamSession {
    /// Creates a new stream session.
    ///
    /// # Arguments
    ///
    /// * `session_id` - Unique session identifier
    /// * `memory_space_id` - Memory space for this session
    #[must_use]
    pub fn new(session_id: String, memory_space_id: MemorySpaceId) -> Self {
        let now = Instant::now();
        Self {
            session_id,
            memory_space_id,
            last_sequence: AtomicU64::new(0),
            created_at: now,
            last_activity: AtomicU64::new(now.elapsed().as_nanos().try_into().unwrap_or(u64::MAX)),
            state: AtomicU8::new(SessionState::Active as u8),
        }
    }

    /// Returns the session ID.
    #[must_use]
    pub fn session_id(&self) -> &str {
        &self.session_id
    }

    /// Returns the memory space ID.
    #[must_use]
    pub const fn memory_space_id(&self) -> &MemorySpaceId {
        &self.memory_space_id
    }

    /// Returns the last sequence number.
    #[must_use]
    pub fn last_sequence(&self) -> u64 {
        self.last_sequence.load(Ordering::SeqCst)
    }

    /// Returns the current session state.
    #[must_use]
    pub fn state(&self) -> SessionState {
        let state_val = self.state.load(Ordering::SeqCst);
        SessionState::from(state_val)
    }

    /// Sets the session state.
    pub fn set_state(&self, new_state: SessionState) {
        self.state.store(new_state as u8, Ordering::SeqCst);
    }

    /// Updates the last activity timestamp.
    fn update_activity(&self) {
        let elapsed = self.created_at.elapsed().as_nanos();
        let elapsed_u64 = elapsed.try_into().unwrap_or(u64::MAX);
        self.last_activity.store(elapsed_u64, Ordering::SeqCst);
    }

    /// Checks if the session has been idle for longer than the timeout.
    ///
    /// # Arguments
    ///
    /// * `timeout` - Timeout duration
    #[must_use]
    pub fn is_idle(&self, timeout: Duration) -> bool {
        let last_activity_nanos = self.last_activity.load(Ordering::SeqCst);
        let last_activity_duration = Duration::from_nanos(last_activity_nanos);
        let current_elapsed = self.created_at.elapsed();

        current_elapsed.saturating_sub(last_activity_duration) > timeout
    }

    /// Validates and updates the sequence number.
    ///
    /// Returns `Ok(())` if the sequence number is valid (monotonic increment),
    /// or `Err` with the expected sequence number if invalid.
    ///
    /// # Arguments
    ///
    /// * `received_seq` - Sequence number received from client
    ///
    /// # Errors
    ///
    /// Returns [`SessionError::SequenceMismatch`] if the received sequence number
    /// is not exactly `last_sequence + 1`.
    pub fn validate_sequence(&self, received_seq: u64) -> Result<(), SessionError> {
        // Get current last_sequence and compute expected next value
        let current = self.last_sequence.load(Ordering::SeqCst);
        let expected = current.wrapping_add(1);

        if received_seq != expected {
            return Err(SessionError::SequenceMismatch {
                session_id: self.session_id.clone(),
                expected,
                received: received_seq,
            });
        }

        // Update to new sequence number
        self.last_sequence.store(received_seq, Ordering::SeqCst);
        self.update_activity();

        Ok(())
    }
}

/// Session manager for concurrent session storage and lifecycle management.
///
/// Uses `DashMap` for lock-free concurrent access to session storage.
/// Provides methods for session creation, lookup, validation, and timeout cleanup.
pub struct SessionManager {
    /// Concurrent session storage
    sessions: Arc<DashMap<String, Arc<StreamSession>>>,

    /// Session timeout duration
    timeout: Duration,
}

impl SessionManager {
    /// Creates a new session manager with default timeout.
    #[must_use]
    pub fn new() -> Self {
        Self {
            sessions: Arc::new(DashMap::new()),
            timeout: DEFAULT_SESSION_TIMEOUT,
        }
    }

    /// Creates a new session manager with custom timeout.
    ///
    /// # Arguments
    ///
    /// * `timeout` - Custom session timeout duration
    #[must_use]
    pub fn with_timeout(timeout: Duration) -> Self {
        Self {
            sessions: Arc::new(DashMap::new()),
            timeout,
        }
    }

    /// Creates a new session and stores it in the manager.
    ///
    /// # Arguments
    ///
    /// * `session_id` - Unique session identifier
    /// * `memory_space_id` - Memory space for this session
    ///
    /// # Returns
    ///
    /// Arc reference to the created session.
    #[must_use]
    pub fn create_session(
        &self,
        session_id: String,
        memory_space_id: MemorySpaceId,
    ) -> Arc<StreamSession> {
        let session = Arc::new(StreamSession::new(session_id.clone(), memory_space_id));
        self.sessions.insert(session_id, Arc::clone(&session));
        session
    }

    /// Retrieves a session by ID.
    ///
    /// # Arguments
    ///
    /// * `session_id` - Session identifier
    ///
    /// # Errors
    ///
    /// Returns [`SessionError::NotFound`] if session does not exist.
    pub fn get_session(&self, session_id: &str) -> Result<Arc<StreamSession>, SessionError> {
        self.sessions
            .get(session_id)
            .map(|entry| Arc::clone(entry.value()))
            .ok_or_else(|| SessionError::NotFound {
                session_id: session_id.to_string(),
            })
    }

    /// Closes a session by setting its state to Closed.
    ///
    /// The session remains in storage for potential reconnection until timeout.
    ///
    /// # Arguments
    ///
    /// * `session_id` - Session identifier
    ///
    /// # Errors
    ///
    /// Returns [`SessionError::NotFound`] if session does not exist.
    pub fn close_session(&self, session_id: &str) -> Result<(), SessionError> {
        let session = self.get_session(session_id)?;
        session.set_state(SessionState::Closed);
        Ok(())
    }

    /// Removes a session from storage.
    ///
    /// # Arguments
    ///
    /// * `session_id` - Session identifier
    pub fn remove_session(&self, session_id: &str) {
        self.sessions.remove(session_id);
    }

    /// Cleans up idle sessions that have exceeded the timeout.
    ///
    /// Returns the number of sessions removed.
    #[must_use]
    pub fn cleanup_idle_sessions(&self) -> usize {
        let mut removed = 0;

        // Collect session IDs to remove (to avoid holding lock during iteration)
        let to_remove: Vec<String> = self
            .sessions
            .iter()
            .filter(|entry| entry.value().is_idle(self.timeout))
            .map(|entry| entry.key().clone())
            .collect();

        for session_id in to_remove {
            self.sessions.remove(&session_id);
            removed += 1;
        }

        removed
    }

    /// Returns the number of active sessions.
    #[must_use]
    pub fn session_count(&self) -> usize {
        self.sessions.len()
    }
}

impl Default for SessionManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Errors that can occur during session operations.
#[derive(Debug, Error)]
pub enum SessionError {
    /// Session not found
    #[error("Session not found: {session_id}")]
    NotFound {
        /// Session identifier
        session_id: String,
    },

    /// Sequence number mismatch (gap or duplicate)
    #[error("Sequence mismatch for session {session_id}: expected {expected}, got {received}")]
    SequenceMismatch {
        /// Session identifier
        session_id: String,
        /// Expected sequence number
        expected: u64,
        /// Received sequence number
        received: u64,
    },

    /// Session is in invalid state for operation
    #[error("Invalid session state for {session_id}: {reason}")]
    InvalidState {
        /// Session identifier
        session_id: String,
        /// Reason for invalid state
        reason: String,
    },
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic, clippy::expect_used)]
mod tests {
    use super::*;
    use std::thread;
    use std::time::Duration;

    #[test]
    fn test_session_creation() {
        let manager = SessionManager::new();
        let space_id = MemorySpaceId::new("test_space").unwrap();
        let session = manager.create_session("session1".to_string(), space_id.clone());

        assert_eq!(session.session_id(), "session1");
        assert_eq!(session.memory_space_id(), &space_id);
        assert_eq!(session.last_sequence(), 0);
        assert_eq!(session.state(), SessionState::Active);
    }

    #[test]
    fn test_sequence_validation() {
        let space_id = MemorySpaceId::new("test_space").unwrap();
        let session = StreamSession::new("test".to_string(), space_id);

        // Valid sequence progression
        assert!(session.validate_sequence(1).is_ok());
        assert_eq!(session.last_sequence(), 1);

        assert!(session.validate_sequence(2).is_ok());
        assert_eq!(session.last_sequence(), 2);

        assert!(session.validate_sequence(3).is_ok());
        assert_eq!(session.last_sequence(), 3);
    }

    #[test]
    fn test_sequence_gap_detection() {
        let space_id = MemorySpaceId::new("test_space").unwrap();
        let session = StreamSession::new("test".to_string(), space_id);

        assert!(session.validate_sequence(1).is_ok());

        // Gap: expecting 2, got 10
        let result = session.validate_sequence(10);
        assert!(result.is_err());

        if let Err(SessionError::SequenceMismatch {
            expected, received, ..
        }) = result
        {
            assert_eq!(expected, 2);
            assert_eq!(received, 10);
        } else {
            panic!("Expected SequenceMismatch error");
        }
    }

    #[test]
    fn test_sequence_duplicate_detection() {
        let space_id = MemorySpaceId::new("test_space").unwrap();
        let session = StreamSession::new("test".to_string(), space_id);

        assert!(session.validate_sequence(1).is_ok());
        assert!(session.validate_sequence(2).is_ok());

        // Duplicate: expecting 3, got 1
        let result = session.validate_sequence(1);
        assert!(result.is_err());
    }

    #[test]
    fn test_session_state_transitions() {
        let space_id = MemorySpaceId::new("test_space").unwrap();
        let session = StreamSession::new("test".to_string(), space_id);

        assert_eq!(session.state(), SessionState::Active);

        session.set_state(SessionState::Paused);
        assert_eq!(session.state(), SessionState::Paused);

        session.set_state(SessionState::Closed);
        assert_eq!(session.state(), SessionState::Closed);

        session.set_state(SessionState::Active);
        assert_eq!(session.state(), SessionState::Active);
    }

    #[test]
    fn test_session_manager_crud() {
        let manager = SessionManager::new();
        let space_id = MemorySpaceId::new("test_space").unwrap();

        // Create
        let session = manager.create_session("session1".to_string(), space_id);
        assert_eq!(manager.session_count(), 1);

        // Read
        let retrieved = manager.get_session("session1");
        assert!(retrieved.is_ok());
        assert_eq!(retrieved.unwrap().session_id(), session.session_id());

        // Close
        assert!(manager.close_session("session1").is_ok());
        let session = manager.get_session("session1").unwrap();
        assert_eq!(session.state(), SessionState::Closed);

        // Remove
        manager.remove_session("session1");
        assert_eq!(manager.session_count(), 0);
        assert!(manager.get_session("session1").is_err());
    }

    #[test]
    fn test_session_not_found_error() {
        let manager = SessionManager::new();

        let result = manager.get_session("nonexistent");
        assert!(result.is_err());

        if let Err(SessionError::NotFound { session_id }) = result {
            assert_eq!(session_id, "nonexistent");
        } else {
            panic!("Expected NotFound error");
        }
    }

    #[test]
    fn test_session_timeout() {
        let manager = SessionManager::with_timeout(Duration::from_millis(100));
        let space_id = MemorySpaceId::new("test_space").unwrap();
        let session = manager.create_session("session1".to_string(), space_id);

        // Session should not be idle initially
        assert!(!session.is_idle(Duration::from_millis(100)));

        // Wait for timeout
        thread::sleep(Duration::from_millis(150));

        // Session should now be idle
        assert!(session.is_idle(Duration::from_millis(100)));
    }

    #[test]
    fn test_cleanup_idle_sessions() {
        let manager = SessionManager::with_timeout(Duration::from_millis(50));
        let space_id = MemorySpaceId::new("test_space").unwrap();

        // Create multiple sessions
        let _ = manager.create_session("session1".to_string(), space_id.clone());
        let _ = manager.create_session("session2".to_string(), space_id.clone());
        let _ = manager.create_session("session3".to_string(), space_id);

        assert_eq!(manager.session_count(), 3);

        // Wait for timeout
        thread::sleep(Duration::from_millis(100));

        // Cleanup should remove all sessions
        let removed = manager.cleanup_idle_sessions();
        assert_eq!(removed, 3);
        assert_eq!(manager.session_count(), 0);
    }

    #[test]
    fn test_concurrent_session_access() {
        // Test that sessions can be accessed concurrently from multiple threads
        // without data races or corruption
        let manager = Arc::new(SessionManager::new());
        let space_id = MemorySpaceId::new("test_space").unwrap();

        // Create multiple sessions concurrently and verify they succeed
        let success_count = (0..10)
            .map(|i| {
                let manager = Arc::clone(&manager);
                let space_id = space_id.clone();
                thread::spawn(move || {
                    let session_id = format!("session_{i}");
                    let _ = manager.create_session(session_id.clone(), space_id);
                    manager.get_session(&session_id).is_ok()
                })
            })
            .map(|h| h.join().unwrap())
            .filter(|&r| r)
            .count();

        // All session creations and lookups should succeed
        assert_eq!(success_count, 10);
        assert_eq!(manager.session_count(), 10);
    }
}
