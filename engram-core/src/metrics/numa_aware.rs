//! NUMA-aware monitoring for multi-socket systems

use crossbeam_utils::CachePadded;
use hwloc::{CpuSet, ObjectType, Topology};
use std::convert::TryFrom;
use std::sync::atomic::{AtomicU64, Ordering};

/// NUMA-aware metric collectors
pub struct NumaCollectors {
    /// Per-socket collectors
    socket_collectors: Vec<SocketCollector>,

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
                SocketCollector::new(id, &cpuset)
            })
            .collect();

        Some(Self {
            socket_collectors,
            cross_socket_accesses: CachePadded::new(AtomicU64::new(0)),
            local_accesses: CachePadded::new(AtomicU64::new(0)),
        })
    }

    /// Get the collector for the current CPU.
    #[must_use]
    pub fn current_collector(&self) -> &SocketCollector {
        let cpu_id = Self::current_cpu_id();
        self.collector_for_cpu(cpu_id)
    }

    /// Get the collector for a specific CPU.
    ///
    /// # Panics
    /// Panics if the collector list is empty, which can only occur when the
    /// type is constructed manually without using `NumaCollectors::new`.
    #[must_use]
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
    pub fn record_memory_access(&self, is_local: bool) {
        if is_local {
            self.local_accesses.fetch_add(1, Ordering::Relaxed);
        } else {
            self.cross_socket_accesses.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Get NUMA locality ratio.
    #[must_use]
    pub fn locality_ratio(&self) -> f32 {
        let local = self.local_accesses.load(Ordering::Acquire);
        let remote = self.cross_socket_accesses.load(Ordering::Acquire);
        let total = local.saturating_add(remote);

        ratio_as_f32(local, total, 1.0)
    }

    /// Get current CPU ID
    #[cfg(target_os = "linux")]
    fn current_cpu_id() -> usize {
        let cpu = unsafe { libc::sched_getcpu() };
        usize::try_from(cpu).unwrap_or(0)
    }

    #[cfg(not(target_os = "linux"))]
    const fn current_cpu_id() -> usize {
        0 // Fallback for non-Linux systems
    }

    /// Aggregate metrics from all sockets.
    #[must_use]
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
    /// Total number of operations recorded on this socket.
    pub operations: CachePadded<AtomicU64>,
    /// Cache hits observed for this socket.
    pub cache_hits: CachePadded<AtomicU64>,
    /// Cache misses observed for this socket.
    pub cache_misses: CachePadded<AtomicU64>,
    /// Accumulated memory bandwidth usage in bytes.
    pub memory_bandwidth: CachePadded<AtomicU64>,
}

impl SocketCollector {
    fn new(socket_id: usize, cpuset: &CpuSet) -> Self {
        // Convert CpuSet to a simple bit mask (supports up to 64 CPUs)
        let mut cpu_mask = 0u64;
        for i in 0..64 {
            if cpuset.is_set(u32::try_from(i).unwrap_or(u32::MAX)) {
                cpu_mask |= 1_u64 << i;
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
    pub fn record_operation(&self) {
        self.operations.fetch_add(1, Ordering::Relaxed);
    }

    /// Record cache access
    pub fn record_cache_access(&self, hit: bool) {
        if hit {
            self.cache_hits.fetch_add(1, Ordering::Relaxed);
        } else {
            self.cache_misses.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Record memory bandwidth usage
    pub fn record_bandwidth(&self, bytes: u64) {
        self.memory_bandwidth.fetch_add(bytes, Ordering::Relaxed);
    }

    /// Get cache hit ratio for this socket.
    #[must_use]
    pub fn cache_hit_ratio(&self) -> f32 {
        let hits = self.cache_hits.load(Ordering::Acquire);
        let misses = self.cache_misses.load(Ordering::Acquire);
        let total = hits.saturating_add(misses);

        ratio_as_f32(hits, total, 0.0)
    }
}

/// Aggregated NUMA metrics
#[derive(Debug, Clone)]
pub struct NumaMetrics {
    /// Total operations observed across all sockets.
    pub total_operations: u64,
    /// Aggregate count of cache hits.
    pub total_cache_hits: u64,
    /// Aggregate count of cache misses.
    pub total_cache_misses: u64,
    /// Ratio of accesses served from local sockets.
    pub locality_ratio: f32,
    /// Number of sockets included in the aggregation.
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
    #[cfg(target_os = "linux")]
    collectors: Option<Arc<NumaCollectors>>,
}

impl NumaAllocator {
    #[cfg(target_os = "linux")]
    /// Create a NUMA-aware allocator using the specified policy.
    #[must_use]
    pub fn new(policy: NumaPolicy) -> Self {
        Self {
            policy,
            collectors: NumaCollectors::new().map(Arc::new),
        }
    }

    #[cfg(not(target_os = "linux"))]
    /// Create a NUMA-aware allocator using the specified policy when NUMA is unsupported.
    #[must_use]
    pub const fn new(policy: NumaPolicy) -> Self {
        Self { policy }
    }

    /// Allocate memory with NUMA awareness
    #[cfg(target_os = "linux")]
    ///
    /// # Errors
    ///
    /// Returns an [`std::io::Error`] when the underlying `mmap` call fails or the
    /// requested NUMA policy cannot be applied to the newly allocated region.
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

        if let Some(collectors) = &self.collectors {
            collectors.record_memory_access(true);
        }

        Ok(ptr.cast::<u8>())
    }

    #[cfg(not(target_os = "linux"))]
    #[allow(unsafe_code)]
    ///
    /// # Errors
    ///
    /// Returns an [`std::io::Error`] if the requested memory layout is
    /// invalid or if the allocator fails to reserve memory from the operating
    /// system.
    pub fn allocate(&self, size: usize) -> Result<*mut u8, std::io::Error> {
        // Fallback to standard allocation on non-Linux with policy-guided alignment
        let alignment = match self.policy {
            NumaPolicy::Interleaved => 128,
            NumaPolicy::Bind(_) | NumaPolicy::Local | NumaPolicy::Preferred => 64,
        };

        let layout = std::alloc::Layout::from_size_align(size, alignment)
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
                std::ptr::addr_of!(nodemask) as *const libc::c_ulong,
                libc::c_ulong::try_from(std::mem::size_of::<libc::c_ulong>() * 8)
                    .unwrap_or(libc::c_ulong::MAX),
                0,
            )
        };

        if result != 0 {
            Err(std::io::Error::last_os_error())
        } else {
            Ok(())
        }
    }
}

fn ratio_as_f32(numerator: u64, denominator: u64, default: f32) -> f32 {
    if denominator == 0 {
        return default;
    }

    let ratio = u64_to_f64(numerator) / u64_to_f64(denominator);
    if !ratio.is_finite() {
        return default;
    }

    clamped_f64_to_f32(ratio, default).clamp(0.0, f32::MAX)
}

fn u64_to_f64(value: u64) -> f64 {
    let high_part = u32::try_from(value >> 32).unwrap_or(u32::MAX);
    let low_part = u32::try_from(value & 0xFFFF_FFFF).unwrap_or(u32::MAX);
    f64::from(high_part).mul_add(4_294_967_296.0, f64::from(low_part))
}

fn clamped_f64_to_f32(value: f64, default: f32) -> f32 {
    if !value.is_finite() {
        return default;
    }

    let clamped = value.clamp(-f64::from(f32::MAX), f64::from(f32::MAX));
    let sign_bit = if clamped.is_sign_negative() {
        1_u32 << 31
    } else {
        0
    };
    let abs = clamped.abs();

    if abs == 0.0 {
        return f32::from_bits(sign_bit);
    }

    let bits = abs.to_bits();
    let exponent_bits = (bits >> 52) & 0x7FF;
    let exponent = i32::try_from(exponent_bits).unwrap_or(0);
    let mut exponent_adjusted = exponent - 1023 + 127;
    if exponent_adjusted <= 0 {
        return f32::from_bits(sign_bit);
    }
    if exponent_adjusted >= 0xFF {
        exponent_adjusted = 0xFE;
    }

    let mantissa = bits & ((1_u64 << 52) - 1);
    let mantissa32 = u32::try_from(mantissa >> (52 - 23)).unwrap_or(0x007F_FFFF);
    let exponent_field = u32::try_from(exponent_adjusted).unwrap_or(0);
    let bits32 = sign_bit | (exponent_field << 23) | mantissa32;
    f32::from_bits(bits32)
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
            assert!((0.0..=1.0).contains(&ratio));
        }
    }

    #[test]
    fn test_socket_collector() {
        // Create a simple cpuset for testing
        let cpuset = CpuSet::new();
        let collector = SocketCollector::new(0, &cpuset);

        collector.record_operation();
        collector.record_cache_access(true);
        collector.record_cache_access(false);
        collector.record_bandwidth(1024);

        assert_eq!(collector.operations.load(Ordering::Acquire), 1);
        assert_eq!(collector.cache_hits.load(Ordering::Acquire), 1);
        assert_eq!(collector.cache_misses.load(Ordering::Acquire), 1);
        assert_eq!(collector.memory_bandwidth.load(Ordering::Acquire), 1024);

        let hit_ratio = collector.cache_hit_ratio();
        assert!((hit_ratio - 0.5).abs() < f32::EPSILON);
    }
}
