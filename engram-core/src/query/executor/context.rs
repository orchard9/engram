//! Query execution context for multi-tenant isolation and timeout management.
//!
//! This module provides the execution context that accompanies every query,
//! ensuring proper multi-tenant isolation through memory space validation
//! and preventing resource exhaustion through timeout enforcement.

use crate::MemorySpaceId;
use std::time::Duration;

/// Query execution context containing multi-tenant isolation and timeout information.
///
/// Every query must be executed within a context that specifies:
/// - Which memory space to operate on (multi-tenant isolation)
/// - Optional timeout for long-running queries
///
/// # Design Philosophy
///
/// The context is designed to be:
/// - **Lightweight**: Small enough to pass by value (24 bytes)
/// - **Explicit**: Requires explicit memory space specification
/// - **Safe**: Enforces multi-tenant boundaries at the type level
///
/// # Memory Layout
///
/// Total size: ~40 bytes
/// - memory_space_id: 24 bytes (String with Arc)
/// - timeout: 16 bytes (Option<Duration>)
///
/// # Example
///
/// ```rust
/// use engram_core::query::executor::QueryContext;
/// use engram_core::MemorySpaceId;
/// use std::time::Duration;
///
/// // Create context for specific memory space with timeout
/// let context = QueryContext::new(
///     MemorySpaceId::new("user_123".to_string()).unwrap(),
///     Some(Duration::from_secs(5)),
/// );
///
/// // Create context without timeout (for fast queries)
/// let context = QueryContext::without_timeout(MemorySpaceId::new("user_123".to_string()).unwrap());
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QueryContext {
    /// Memory space ID for multi-tenant isolation.
    ///
    /// Every query must specify which memory space it operates on.
    /// The executor validates this ID against the registry before
    /// executing any operations.
    pub memory_space_id: MemorySpaceId,

    /// Optional timeout for query execution.
    ///
    /// If specified, the executor will attempt to abort the query
    /// if it exceeds this duration. Note that timeout enforcement
    /// is best-effort and may not be immediate for certain operations.
    ///
    /// Typical values:
    /// - Fast queries (RECALL simple patterns): 100ms - 1s
    /// - Complex queries (SPREAD with many hops): 1s - 5s
    /// - Pattern completion (IMAGINE): 5s - 30s
    pub timeout: Option<Duration>,
}

impl QueryContext {
    /// Create a new query context with memory space and optional timeout.
    ///
    /// # Arguments
    ///
    /// * `memory_space_id` - The memory space to execute the query in
    /// * `timeout` - Optional timeout duration for query execution
    ///
    /// # Example
    ///
    /// ```rust
    /// use engram_core::query::executor::QueryContext;
    /// use engram_core::MemorySpaceId;
    /// use std::time::Duration;
    ///
    /// let context = QueryContext::new(
    ///     MemorySpaceId::new("user_123".to_string()).unwrap(),
    ///     Some(Duration::from_secs(5)),
    /// );
    /// ```
    #[must_use]
    pub const fn new(memory_space_id: MemorySpaceId, timeout: Option<Duration>) -> Self {
        Self {
            memory_space_id,
            timeout,
        }
    }

    /// Create a query context without a timeout.
    ///
    /// Use this for queries expected to complete quickly or when
    /// timeout enforcement is not needed.
    ///
    /// # Example
    ///
    /// ```rust
    /// use engram_core::query::executor::QueryContext;
    /// use engram_core::MemorySpaceId;
    ///
    /// let context = QueryContext::without_timeout(MemorySpaceId::new("user_123".to_string()).unwrap());
    /// ```
    #[must_use]
    pub const fn without_timeout(memory_space_id: MemorySpaceId) -> Self {
        Self::new(memory_space_id, None)
    }

    /// Create a query context with a timeout.
    ///
    /// Convenience method for specifying timeout directly.
    ///
    /// # Example
    ///
    /// ```rust
    /// use engram_core::query::executor::QueryContext;
    /// use engram_core::MemorySpaceId;
    /// use std::time::Duration;
    ///
    /// let context = QueryContext::with_timeout(
    ///     MemorySpaceId::new("user_123".to_string()).unwrap(),
    ///     Duration::from_secs(5),
    /// );
    /// ```
    #[must_use]
    pub const fn with_timeout(memory_space_id: MemorySpaceId, timeout: Duration) -> Self {
        Self::new(memory_space_id, Some(timeout))
    }

    /// Check if this context has a timeout configured.
    #[must_use]
    pub const fn has_timeout(&self) -> bool {
        self.timeout.is_some()
    }

    /// Get the timeout duration if configured.
    #[must_use]
    pub const fn timeout_duration(&self) -> Option<Duration> {
        self.timeout
    }

    /// Create a new context with a different memory space.
    #[must_use]
    pub fn with_memory_space(self, memory_space_id: MemorySpaceId) -> Self {
        Self {
            memory_space_id,
            timeout: self.timeout,
        }
    }

    /// Create a new context with a different timeout.
    #[must_use]
    pub fn with_new_timeout(self, timeout: Option<Duration>) -> Self {
        Self {
            memory_space_id: self.memory_space_id,
            timeout,
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)] // Tests are allowed to use unwrap
#[allow(clippy::unnecessary_to_owned)] // Test readability is more important
mod tests {
    use super::*;

    #[test]
    fn test_context_creation() {
        let space_id = MemorySpaceId::new("test_space".to_string()).unwrap();
        let timeout = Duration::from_secs(5);

        let context = QueryContext::new(space_id.clone(), Some(timeout));

        assert_eq!(context.memory_space_id, space_id);
        assert_eq!(context.timeout, Some(timeout));
        assert!(context.has_timeout());
    }

    #[test]
    fn test_context_without_timeout() {
        let space_id = MemorySpaceId::new("test_space".to_string()).unwrap();
        let context = QueryContext::without_timeout(space_id.clone());

        assert_eq!(context.memory_space_id, space_id);
        assert_eq!(context.timeout, None);
        assert!(!context.has_timeout());
    }

    #[test]
    fn test_context_with_timeout() {
        let space_id = MemorySpaceId::new("test_space".to_string()).unwrap();
        let timeout = Duration::from_millis(500);
        let context = QueryContext::with_timeout(space_id.clone(), timeout);

        assert_eq!(context.memory_space_id, space_id);
        assert_eq!(context.timeout, Some(timeout));
        assert!(context.has_timeout());
        assert_eq!(context.timeout_duration(), Some(timeout));
    }

    #[test]
    fn test_context_modification() {
        let space_id_1 = MemorySpaceId::new("space_1".to_string()).unwrap();
        let space_id_2 = MemorySpaceId::new("space_2".to_string()).unwrap();
        let timeout = Duration::from_secs(10);

        let context = QueryContext::without_timeout(space_id_1);
        let context = context.with_memory_space(space_id_2.clone());
        let context = context.with_new_timeout(Some(timeout));

        assert_eq!(context.memory_space_id, space_id_2);
        assert_eq!(context.timeout, Some(timeout));
    }

    #[test]
    fn test_context_clone() {
        let context = QueryContext::with_timeout(
            MemorySpaceId::new("test".to_string()).unwrap(),
            Duration::from_secs(1),
        );

        let cloned = context.clone();
        assert_eq!(context, cloned);
    }
}
