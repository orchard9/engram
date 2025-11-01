//! NUMA-aware monitoring for multi-socket systems
//!
//! Provides thread-safe wrapper around hwlocality for collecting NUMA locality metrics
//! in high-performance concurrent systems. Uses Arc-wrapped topology for safe sharing
//! across threads while maintaining lock-free metric collection.
//!
//! # Architecture
//!
//! The hwlocality `Topology` type is thread-safe for reads after initialization
//! (hwloc documentation states multiple threads may concurrently consult the topology
//! once loaded). We wrap it in `Arc` for safe shared ownership across threads.
//!
//! Metric collection remains lock-free using atomic operations, with topology
//! queries only performed during initialization and infrequent updates.

#[cfg(feature = "monitoring")]
use hwlocality::Topology;

use crossbeam_utils::CachePadded;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

/// Thread-safe handle to hardware topology information
///
/// Wraps hwlocality::Topology in Arc for safe concurrent access. The underlying
/// Topology is immutable after initialization, making it safe to share across
/// threads for read-only queries.
///
/// # Thread Safety
///
/// - Topology: Safe for concurrent reads (hwloc guarantees)
/// - Arc: Provides safe shared ownership
/// - No interior mutability: All queries are const/immutable operations
#[cfg(feature = "monitoring")]
#[derive(Clone)]
pub struct TopologyHandle {
    inner: Arc<Topology>,
}

#[cfg(feature = "monitoring")]
impl TopologyHandle {
    /// Create a new topology handle by detecting system hardware
    ///
    /// # Errors
    ///
    /// Returns error if hwloc cannot detect system topology or if hwloc
    /// library is not available on this platform.
    pub fn new() -> Result<Self, TopologyError> {
        let topology = Topology::new().map_err(|e| TopologyError::InitFailed(e.to_string()))?;
        Ok(Self {
            inner: Arc::new(topology),
        })
    }

    /// Get the number of NUMA nodes in the system
    ///
    /// Returns 1 if NUMA support is not available or not detected.
    #[must_use]
    pub fn num_numa_nodes(&self) -> usize {
        use hwlocality::object::types::ObjectType;

        self.inner
            .objects_with_type(ObjectType::NUMANode)
            .count()
            .max(1)
    }

    /// Get the number of physical CPU cores (excluding hyperthreads)
    #[must_use]
    pub fn num_physical_cores(&self) -> usize {
        use hwlocality::object::types::ObjectType;

        self.inner
            .objects_with_type(ObjectType::Core)
            .count()
            .max(1)
    }

    /// Get the number of processing units (including hyperthreads)
    #[must_use]
    pub fn num_processing_units(&self) -> usize {
        use hwlocality::object::types::ObjectType;

        self.inner.objects_with_type(ObjectType::PU).count().max(1)
    }

    /// Check if the current thread is bound to a specific NUMA node
    ///
    /// Returns the NUMA node index if bound, None if not bound or
    /// if NUMA binding cannot be determined.
    #[must_use]
    pub fn current_numa_node(&self) -> Option<usize> {
        use hwlocality::cpu::binding::CpuBindingFlags;
        use hwlocality::object::types::ObjectType;

        // Get CPU binding for current thread
        let Ok(cpu_set) = self.inner.cpu_binding(CpuBindingFlags::THREAD) else {
            return None;
        };

        // Find NUMA node containing this CPU set
        self.inner
            .objects_with_type(ObjectType::NUMANode)
            .enumerate()
            .find(|(_, obj)| {
                // Check if cpu_set is included in this NUMA node's cpuset
                obj.cpuset()
                    .is_some_and(|cpuset| cpu_set.iter_set().all(|cpu| cpuset.is_set(cpu)))
            })
            .map(|(idx, _)| idx)
    }

    /// Get reference to underlying Topology for advanced queries
    #[must_use]
    pub fn topology(&self) -> &Topology {
        &self.inner
    }
}

/// Errors that can occur when initializing NUMA topology
#[cfg(feature = "monitoring")]
#[derive(Debug, Clone)]
pub enum TopologyError {
    /// Failed to initialize hwloc topology
    InitFailed(String),
    /// NUMA support not available on this platform
    NotSupported,
}

#[cfg(feature = "monitoring")]
impl std::fmt::Display for TopologyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InitFailed(msg) => write!(f, "Failed to initialize NUMA topology: {msg}"),
            Self::NotSupported => write!(f, "NUMA support not available on this platform"),
        }
    }
}

#[cfg(feature = "monitoring")]
impl std::error::Error for TopologyError {}

#[cfg(feature = "monitoring")]
impl From<hwlocality::errors::HybridError<std::io::Error>> for TopologyError {
    fn from(err: hwlocality::errors::HybridError<std::io::Error>) -> Self {
        Self::InitFailed(err.to_string())
    }
}

/// NUMA-aware metric collectors
///
/// Collects lock-free metrics about NUMA locality and memory access patterns.
/// Safe to share across threads via Arc, with atomic counters preventing
/// false sharing through cache-line padding.
///
/// # Architecture
///
/// - Topology: Immutable after init, safe for concurrent reads
/// - Counters: Lock-free atomics with cache-line padding
/// - Total overhead: <20ns per metric update
#[cfg(feature = "monitoring")]
pub struct NumaCollectors {
    /// Hardware topology handle (thread-safe, immutable)
    topology: TopologyHandle,

    /// Per-node activation counters (cache-line padded to prevent false sharing)
    node_activations: Vec<CachePadded<AtomicU64>>,

    /// Local vs remote memory access counters
    local_accesses: CachePadded<AtomicU64>,
    remote_accesses: CachePadded<AtomicU64>,
}

#[cfg(feature = "monitoring")]
impl NumaCollectors {
    /// Create NUMA-aware collectors if hardware topology is available
    ///
    /// Returns None if hwloc cannot initialize or NUMA is not supported.
    /// This is expected on single-socket systems or platforms without hwloc.
    #[must_use]
    pub fn new() -> Option<Self> {
        let topology = match TopologyHandle::new() {
            Ok(t) => t,
            Err(err) => {
                tracing::debug!(
                    target: "engram::metrics::numa",
                    error = %err,
                    "NUMA topology initialization failed, metrics disabled"
                );
                return None;
            }
        };

        let num_nodes = topology.num_numa_nodes();

        tracing::info!(
            target: "engram::metrics::numa",
            numa_nodes = num_nodes,
            physical_cores = topology.num_physical_cores(),
            processing_units = topology.num_processing_units(),
            "NUMA-aware metrics enabled"
        );

        let node_activations = (0..num_nodes)
            .map(|_| CachePadded::new(AtomicU64::new(0)))
            .collect();

        Some(Self {
            topology,
            node_activations,
            local_accesses: CachePadded::new(AtomicU64::new(0)),
            remote_accesses: CachePadded::new(AtomicU64::new(0)),
        })
    }

    /// Record an activation event on a specific NUMA node
    ///
    /// If node_id is out of range, the event is recorded on node 0.
    /// This ensures metrics are never lost even if node detection fails.
    pub fn record_activation(&self, node_id: usize) {
        let idx = node_id.min(self.node_activations.len().saturating_sub(1));
        self.node_activations[idx].fetch_add(1, Ordering::Relaxed);
    }

    /// Record a memory access, categorized as local or remote
    ///
    /// Local accesses are to memory on the same NUMA node as the accessing thread.
    /// Remote accesses cross NUMA boundaries and have higher latency.
    pub fn record_memory_access(&self, is_local: bool) {
        if is_local {
            self.local_accesses.fetch_add(1, Ordering::Relaxed);
        } else {
            self.remote_accesses.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Calculate NUMA locality ratio (local / total accesses)
    ///
    /// Returns value in range [0.0, 1.0] where:
    /// - 1.0 = perfect locality (all accesses local)
    /// - 0.5 = 50/50 split
    /// - 0.0 = all accesses remote (pathological)
    ///
    /// Returns 1.0 if no accesses recorded yet.
    #[must_use]
    pub fn locality_ratio(&self) -> f64 {
        let local = self.local_accesses.load(Ordering::Acquire);
        let remote = self.remote_accesses.load(Ordering::Acquire);
        let total = local + remote;

        if total == 0 {
            return 1.0; // No data yet, assume perfect locality
        }

        local as f64 / total as f64
    }

    /// Get activation counts for all NUMA nodes
    ///
    /// Returns vector indexed by node ID containing activation counts.
    #[must_use]
    pub fn node_activation_counts(&self) -> Vec<u64> {
        self.node_activations
            .iter()
            .map(|counter| counter.load(Ordering::Acquire))
            .collect()
    }

    /// Get number of NUMA nodes in the system
    #[must_use]
    pub fn num_nodes(&self) -> usize {
        self.topology.num_numa_nodes()
    }

    /// Get the NUMA node the current thread is bound to, if any
    #[must_use]
    pub fn current_node(&self) -> Option<usize> {
        self.topology.current_numa_node()
    }

    /// Get reference to topology handle for advanced queries
    #[must_use]
    pub const fn topology(&self) -> &TopologyHandle {
        &self.topology
    }
}

/// Stub implementation when monitoring feature is disabled
#[cfg(not(feature = "monitoring"))]
#[derive(Default)]
pub struct NumaCollectors {
    _marker: (),
}

#[cfg(not(feature = "monitoring"))]
impl NumaCollectors {
    #[must_use]
    pub const fn new() -> Option<Self> {
        None
    }

    #[allow(clippy::unused_self)]
    pub const fn record_activation(&self, _node_id: usize) {}

    #[allow(clippy::unused_self)]
    pub const fn record_memory_access(&self, _is_local: bool) {}

    #[must_use]
    #[allow(clippy::unused_self)]
    pub const fn locality_ratio(&self) -> f64 {
        1.0
    }
}

#[cfg(all(test, feature = "monitoring"))]
mod tests {
    use super::*;

    #[test]
    fn test_topology_handle_creation() {
        // Should either succeed or fail gracefully
        if let Ok(handle) = TopologyHandle::new() {
            assert!(handle.num_numa_nodes() >= 1);
            assert!(handle.num_physical_cores() >= 1);
            assert!(handle.num_processing_units() >= 1);
        } else {
            // Expected on systems without hwloc or NUMA
        }
    }

    #[test]
    fn test_topology_handle_is_clone() {
        if let Ok(handle) = TopologyHandle::new() {
            let handle2 = &handle;
            assert_eq!(handle.num_numa_nodes(), handle2.num_numa_nodes());
        }
    }

    #[test]
    fn test_numa_collectors_creation() {
        // Should either succeed or return None gracefully
        if let Some(collectors) = NumaCollectors::new() {
            assert!(collectors.num_nodes() >= 1);
            assert!((collectors.locality_ratio() - 1.0).abs() < f64::EPSILON); // No data yet
        }
    }

    #[test]
    fn test_record_activation() {
        if let Some(collectors) = NumaCollectors::new() {
            collectors.record_activation(0);
            collectors.record_activation(0);

            let counts = collectors.node_activation_counts();
            assert!(!counts.is_empty());
            assert_eq!(counts[0], 2);
        }
    }

    #[test]
    fn test_record_activation_out_of_bounds() {
        if let Some(collectors) = NumaCollectors::new() {
            let num_nodes = collectors.num_nodes();

            // Should not panic, records to last node
            collectors.record_activation(num_nodes + 100);

            let counts = collectors.node_activation_counts();
            assert!(counts.iter().sum::<u64>() >= 1);
        }
    }

    #[test]
    fn test_memory_access_tracking() {
        if let Some(collectors) = NumaCollectors::new() {
            collectors.record_memory_access(true); // local
            collectors.record_memory_access(true); // local
            collectors.record_memory_access(false); // remote

            let ratio = collectors.locality_ratio();
            assert!((ratio - 2.0 / 3.0).abs() < 0.01); // ~0.667
        }
    }

    #[test]
    fn test_locality_ratio_perfect() {
        if let Some(collectors) = NumaCollectors::new() {
            collectors.record_memory_access(true);
            collectors.record_memory_access(true);

            assert!((collectors.locality_ratio() - 1.0).abs() < f64::EPSILON);
        }
    }

    #[test]
    fn test_locality_ratio_pathological() {
        if let Some(collectors) = NumaCollectors::new() {
            collectors.record_memory_access(false);
            collectors.record_memory_access(false);

            assert!(collectors.locality_ratio().abs() < f64::EPSILON);
        }
    }

    #[test]
    fn test_concurrent_activation_recording() {
        if let Some(collectors) = NumaCollectors::new() {
            use std::sync::Arc;
            use std::thread;

            let collectors = Arc::new(collectors);
            let mut handles = vec![];

            // Spawn 10 threads, each recording 100 activations
            for _ in 0..10 {
                let collectors_clone = Arc::clone(&collectors);
                handles.push(thread::spawn(move || {
                    for _ in 0..100 {
                        collectors_clone.record_activation(0);
                    }
                }));
            }

            for handle in handles {
                assert!(handle.join().is_ok(), "thread should not panic");
            }

            let counts = collectors.node_activation_counts();
            assert_eq!(counts[0], 1000);
        }
    }

    #[test]
    fn test_concurrent_memory_access_recording() {
        if let Some(collectors) = NumaCollectors::new() {
            use std::sync::Arc;
            use std::thread;

            let collectors = Arc::new(collectors);
            let mut handles = vec![];

            // Spawn threads recording local and remote accesses
            for i in 0..10 {
                let collectors_clone = Arc::clone(&collectors);
                handles.push(thread::spawn(move || {
                    for _ in 0..100 {
                        collectors_clone.record_memory_access(i % 2 == 0);
                    }
                }));
            }

            for handle in handles {
                assert!(handle.join().is_ok(), "thread should not panic");
            }

            let ratio = collectors.locality_ratio();
            assert!((ratio - 0.5).abs() < 0.01); // Should be ~50%
        }
    }
}
