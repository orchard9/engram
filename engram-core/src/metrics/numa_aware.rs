//! NUMA-aware monitoring for multi-socket systems
//!
//! **STATUS**: Temporarily disabled during dependency migration.
//!
//! The old `hwloc` v0.5 dependency used deprecated `bitflags` v0.7.0 which will be rejected
//! by future Rust versions. NUMA monitoring will be re-enabled in a future milestone using
//! modern `hwlocality` bindings with proper API migration.
//!
//! **TODO**: Re-implement using hwlocality v1.0+ API when NUMA monitoring is needed.

use crossbeam_utils::CachePadded;
use std::sync::atomic::AtomicU64;

/// NUMA-aware metric collectors (currently disabled)
#[derive(Default)]
pub struct NumaCollectors {
    /// Placeholder - feature disabled
    _marker: (),
}

impl NumaCollectors {
    /// Create NUMA-aware collectors if available
    ///
    /// Currently always returns None - NUMA monitoring disabled during dependency migration.
    #[must_use]
    pub const fn new() -> Option<Self> {
        // NUMA monitoring temporarily disabled during hwlocality migration
        // TODO: Re-implement using hwlocality API
        None
    }

    /// Record an activation event (no-op when NUMA disabled)
    /// Keeps `&self` for API compatibility with future implementation
    #[allow(clippy::unused_self)]
    pub const fn record_activation(&self, _node_id: usize) {
        // No-op
    }

    /// Record a memory access (no-op when NUMA disabled)
    /// Keeps `&self` for API compatibility with future implementation
    #[allow(clippy::unused_self)]
    pub const fn record_memory_access(&self, _is_local: bool) {
        // No-op
    }

    /// Get locality ratio (returns 1.0 when NUMA disabled)
    /// Keeps `&self` for API compatibility with future implementation
    #[must_use]
    #[allow(clippy::unused_self)]
    pub const fn locality_ratio(&self) -> f64 {
        1.0 // Perfect locality when not tracking
    }
}

/// Per-socket collector (stub implementation)
#[allow(dead_code)]
struct SocketCollector {
    socket_id: usize,
    activation_count: CachePadded<AtomicU64>,
}

impl SocketCollector {
    #[allow(dead_code)]
    const fn new(socket_id: usize) -> Self {
        Self {
            socket_id,
            activation_count: CachePadded::new(AtomicU64::new(0)),
        }
    }
}
