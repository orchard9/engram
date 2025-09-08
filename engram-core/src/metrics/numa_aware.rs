//! NUMA-aware monitoring for multi-socket systems

use crossbeam_utils::CachePadded;
use hwloc::{CpuSet, ObjectType, Topology};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

/// NUMA-aware metric collectors
pub struct NumaCollectors {
    /// Per-socket collectors
    socket_collectors: Vec<SocketCollector>,

    /// NUMA topology information
    topology: Arc<Topology>,

    /// Cross-socket communication metrics
    cross_socket_accesses: CachePadded<AtomicU64>,
    local_accesses: CachePadded<AtomicU64>,
}

impl NumaCollectors {
    /// Create NUMA-aware collectors if available
    #[must_use]
    pub fn new() -> Option<Self> {
        let topology = Topology::new();

        // Check if we have multiple NUMA nodes
        let numa_nodes = topology.objects_with_type(&ObjectType::NUMANode).ok()?;
        if numa_nodes.len() <= 1 {
            return None; // Single socket system, NUMA awareness not needed
        }

        // Create per-socket collectors
        let socket_collectors = numa_nodes
            .iter()
            .enumerate()
            .map(|(id, node)| {
                let cpuset = node.cpuset().unwrap_or_else(CpuSet::new);
                SocketCollector::new(id, cpuset)
            })
            .collect();

        Some(Self {
            socket_collectors,
            topology: Arc::new(topology),
            cross_socket_accesses: CachePadded::new(AtomicU64::new(0)),
            local_accesses: CachePadded::new(AtomicU64::new(0)),
        })
    }

    /// Get the collector for the current CPU
    pub fn current_collector(&self) -> &SocketCollector {
        let cpu_id = self.current_cpu_id();
        self.collector_for_cpu(cpu_id)
    }

    /// Get the collector for a specific CPU
    pub fn collector_for_cpu(&self, cpu_id: usize) -> &SocketCollector {
        // Find which socket this CPU belongs to
        if cpu_id < 64 {
            for collector in &self.socket_collectors {
                if (collector.cpu_mask & (1 << cpu_id)) != 0 {
                    return collector;
                }
            }
        }

        // Fallback to first collector
        &self.socket_collectors[0]
    }

    /// Record a memory access
    #[inline(always)]
    pub fn record_memory_access(&self, is_local: bool) {
        if is_local {
            self.local_accesses.fetch_add(1, Ordering::Relaxed);
        } else {
            self.cross_socket_accesses.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Get NUMA locality ratio
    pub fn locality_ratio(&self) -> f32 {
        let local = self.local_accesses.load(Ordering::Acquire) as f32;
        let remote = self.cross_socket_accesses.load(Ordering::Acquire) as f32;

        let total = local + remote;
        if total > 0.0 {
            local / total
        } else {
            1.0 // Assume perfect locality if no data
        }
    }

    /// Get current CPU ID
    #[cfg(target_os = "linux")]
    fn current_cpu_id(&self) -> usize {
        unsafe { libc::sched_getcpu() as usize }
    }

    #[cfg(not(target_os = "linux"))]
    const fn current_cpu_id(&self) -> usize {
        0 // Fallback for non-Linux systems
    }

    /// Aggregate metrics from all sockets
    pub fn aggregate_metrics(&self) -> NumaMetrics {
        let mut total_operations = 0u64;
        let mut total_cache_hits = 0u64;
        let mut total_cache_misses = 0u64;

        for collector in &self.socket_collectors {
            total_operations += collector.operations.load(Ordering::Acquire);
            total_cache_hits += collector.cache_hits.load(Ordering::Acquire);
            total_cache_misses += collector.cache_misses.load(Ordering::Acquire);
        }

        NumaMetrics {
            total_operations,
            total_cache_hits,
            total_cache_misses,
            locality_ratio: self.locality_ratio(),
            socket_count: self.socket_collectors.len(),
        }
    }
}

/// Per-socket metric collector
pub struct SocketCollector {
    /// Socket ID
    pub socket_id: usize,

    /// CPUs in this socket (stored as bit vector)
    pub cpu_mask: u64,

    /// Socket-local metrics (cache-aligned)
    pub operations: CachePadded<AtomicU64>,
    pub cache_hits: CachePadded<AtomicU64>,
    pub cache_misses: CachePadded<AtomicU64>,
    pub memory_bandwidth: CachePadded<AtomicU64>,
}

impl SocketCollector {
    fn new(socket_id: usize, cpuset: CpuSet) -> Self {
        // Convert CpuSet to a simple bit mask (supports up to 64 CPUs)
        let mut cpu_mask = 0u64;
        for i in 0..64 {
            if cpuset.is_set(i as u32) {
                cpu_mask |= 1 << i;
            }
        }

        Self {
            socket_id,
            cpu_mask,
            operations: CachePadded::new(AtomicU64::new(0)),
            cache_hits: CachePadded::new(AtomicU64::new(0)),
            cache_misses: CachePadded::new(AtomicU64::new(0)),
            memory_bandwidth: CachePadded::new(AtomicU64::new(0)),
        }
    }

    /// Record an operation on this socket
    #[inline(always)]
    pub fn record_operation(&self) {
        self.operations.fetch_add(1, Ordering::Relaxed);
    }

    /// Record cache access
    #[inline(always)]
    pub fn record_cache_access(&self, hit: bool) {
        if hit {
            self.cache_hits.fetch_add(1, Ordering::Relaxed);
        } else {
            self.cache_misses.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Record memory bandwidth usage
    #[inline(always)]
    pub fn record_bandwidth(&self, bytes: u64) {
        self.memory_bandwidth.fetch_add(bytes, Ordering::Relaxed);
    }

    /// Get cache hit ratio for this socket
    pub fn cache_hit_ratio(&self) -> f32 {
        let hits = self.cache_hits.load(Ordering::Acquire) as f32;
        let misses = self.cache_misses.load(Ordering::Acquire) as f32;

        let total = hits + misses;
        if total > 0.0 { hits / total } else { 0.0 }
    }
}

/// Aggregated NUMA metrics
#[derive(Debug, Clone)]
pub struct NumaMetrics {
    pub total_operations: u64,
    pub total_cache_hits: u64,
    pub total_cache_misses: u64,
    pub locality_ratio: f32,
    pub socket_count: usize,
}

/// NUMA memory allocation hints
#[derive(Debug, Clone, Copy)]
pub enum NumaPolicy {
    /// Allocate on local socket
    Local,
    /// Interleave across all sockets
    Interleaved,
    /// Bind to specific socket
    Bind(usize),
    /// Prefer local but allow remote
    Preferred,
}

/// NUMA-aware memory allocator wrapper
pub struct NumaAllocator {
    policy: NumaPolicy,
    collectors: Option<Arc<NumaCollectors>>,
}

impl NumaAllocator {
    pub fn new(policy: NumaPolicy) -> Self {
        Self {
            policy,
            collectors: NumaCollectors::new().map(Arc::new),
        }
    }

    /// Allocate memory with NUMA awareness
    #[cfg(target_os = "linux")]
    pub fn allocate(&self, size: usize) -> Result<*mut u8, std::io::Error> {
        use libc::{MAP_ANONYMOUS, MAP_PRIVATE, PROT_READ, PROT_WRITE, mmap};

        let ptr = unsafe {
            mmap(
                std::ptr::null_mut(),
                size,
                PROT_READ | PROT_WRITE,
                MAP_PRIVATE | MAP_ANONYMOUS,
                -1,
                0,
            )
        };

        if ptr == libc::MAP_FAILED {
            return Err(std::io::Error::last_os_error());
        }

        // Apply NUMA policy
        self.apply_numa_policy(ptr, size)?;

        Ok(ptr as *mut u8)
    }

    #[cfg(not(target_os = "linux"))]
    #[allow(unsafe_code)]
    pub fn allocate(&self, size: usize) -> Result<*mut u8, std::io::Error> {
        // Fallback to standard allocation on non-Linux
        let layout = std::alloc::Layout::from_size_align(size, 64)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidInput, e))?;

        let ptr = unsafe { std::alloc::alloc(layout) };

        if ptr.is_null() {
            Err(std::io::Error::new(
                std::io::ErrorKind::OutOfMemory,
                "Failed to allocate memory",
            ))
        } else {
            Ok(ptr)
        }
    }

    #[cfg(target_os = "linux")]
    fn apply_numa_policy(&self, ptr: *mut libc::c_void, size: usize) -> Result<(), std::io::Error> {
        use libc::{MPOL_BIND, MPOL_DEFAULT, MPOL_INTERLEAVE, MPOL_PREFERRED, mbind};

        let (mode, nodemask) = match self.policy {
            NumaPolicy::Local => (MPOL_DEFAULT, 0),
            NumaPolicy::Interleaved => (MPOL_INTERLEAVE, !0),
            NumaPolicy::Bind(socket) => (MPOL_BIND, 1 << socket),
            NumaPolicy::Preferred => (MPOL_PREFERRED, 0),
        };

        let result = unsafe {
            mbind(
                ptr,
                size,
                mode,
                &nodemask as *const _ as *const libc::c_ulong,
                std::mem::size_of::<libc::c_ulong>() as libc::c_ulong * 8,
                0,
            )
        };

        if result != 0 {
            Err(std::io::Error::last_os_error())
        } else {
            Ok(())
        }
    }

    #[cfg(not(target_os = "linux"))]
    const fn apply_numa_policy(
        &self,
        _ptr: *mut std::ffi::c_void,
        _size: usize,
    ) -> Result<(), std::io::Error> {
        Ok(()) // No-op on non-Linux
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_numa_detection() {
        // This test will only create collectors on multi-socket systems
        let collectors = NumaCollectors::new();

        if let Some(numa) = collectors {
            assert!(numa.socket_collectors.len() > 1);

            // Test metric recording
            numa.record_memory_access(true);
            numa.record_memory_access(false);

            let ratio = numa.locality_ratio();
            assert!(ratio >= 0.0 && ratio <= 1.0);
        }
    }

    #[test]
    fn test_socket_collector() {
        // Create a simple cpuset for testing
        let cpuset = CpuSet::new();
        let collector = SocketCollector::new(0, cpuset);

        collector.record_operation();
        collector.record_cache_access(true);
        collector.record_cache_access(false);
        collector.record_bandwidth(1024);

        assert_eq!(collector.operations.load(Ordering::Acquire), 1);
        assert_eq!(collector.cache_hits.load(Ordering::Acquire), 1);
        assert_eq!(collector.cache_misses.load(Ordering::Acquire), 1);
        assert_eq!(collector.memory_bandwidth.load(Ordering::Acquire), 1024);

        let hit_ratio = collector.cache_hit_ratio();
        assert_eq!(hit_ratio, 0.5);
    }
}
