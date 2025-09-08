//! NUMA-aware memory allocation and topology detection
//!
//! This module provides NUMA topology detection and socket-local allocation
//! for optimal memory access patterns on multi-socket systems.

use super::{StorageError, StorageResult};
use std::collections::HashMap;
use std::sync::Arc;

#[cfg(unix)]
use libc;

/// NUMA topology information
#[derive(Debug, Clone)]
pub struct NumaTopology {
    pub socket_count: usize,
    pub node_count: usize,
    pub socket_to_nodes: HashMap<usize, Vec<usize>>,
    pub cpu_to_socket: HashMap<usize, usize>,
    pub memory_per_node_mb: Vec<usize>,
}

impl NumaTopology {
    /// Detect NUMA topology from the system
    pub fn detect() -> StorageResult<Self> {
        #[cfg(unix)]
        {
            Self::detect_unix()
        }

        #[cfg(not(unix))]
        {
            // Fallback for non-Unix systems
            Ok(Self {
                socket_count: 1,
                node_count: 1,
                socket_to_nodes: [(0, vec![0])].iter().cloned().collect(),
                cpu_to_socket: [(0, 0)].iter().cloned().collect(),
                memory_per_node_mb: vec![8192], // Assume 8GB
            })
        }
    }

    #[cfg(unix)]
    fn detect_unix() -> StorageResult<Self> {
        // Try to read from /sys/devices/system/node/
        let node_path = std::path::Path::new("/sys/devices/system/node");
        if !node_path.exists() {
            return Self::detect_fallback();
        }

        let mut socket_to_nodes = HashMap::new();
        let mut cpu_to_socket = HashMap::new();
        let mut memory_per_node_mb = Vec::new();
        let mut node_count = 0;

        // Read NUMA nodes
        for entry in std::fs::read_dir(node_path)
            .map_err(|e| StorageError::NumaError(format!("Failed to read NUMA nodes: {}", e)))?
        {
            let entry =
                entry.map_err(|e| StorageError::NumaError(format!("NUMA entry error: {}", e)))?;
            let file_name = entry.file_name();
            let name_str = file_name.to_string_lossy();

            if name_str.starts_with("node") {
                if let Ok(node_id) = name_str[4..].parse::<usize>() {
                    node_count = node_count.max(node_id + 1);

                    // Read memory info for this node
                    let meminfo_path = entry.path().join("meminfo");
                    if let Ok(meminfo) = std::fs::read_to_string(&meminfo_path) {
                        let memory_mb = Self::parse_memory_mb(&meminfo);
                        memory_per_node_mb.push(memory_mb);
                    } else {
                        memory_per_node_mb.push(1024); // Default 1GB
                    }

                    // Map to socket (simplified - assume node == socket for now)
                    socket_to_nodes
                        .entry(node_id)
                        .or_insert_with(Vec::new)
                        .push(node_id);

                    // Read CPU list for this node
                    let cpulist_path = entry.path().join("cpulist");
                    if let Ok(cpulist) = std::fs::read_to_string(&cpulist_path) {
                        let cpus = Self::parse_cpu_list(&cpulist);
                        for cpu in cpus {
                            cpu_to_socket.insert(cpu, node_id);
                        }
                    }
                }
            }
        }

        let socket_count = socket_to_nodes.len().max(1);

        Ok(Self {
            socket_count,
            node_count,
            socket_to_nodes,
            cpu_to_socket,
            memory_per_node_mb,
        })
    }

    #[cfg(unix)]
    fn detect_fallback() -> StorageResult<Self> {
        // Use sysconf to get basic info
        let cpu_count = unsafe { libc::sysconf(libc::_SC_NPROCESSORS_ONLN) } as usize;

        Ok(Self {
            socket_count: 1,
            node_count: 1,
            socket_to_nodes: [(0, vec![0])].iter().cloned().collect(),
            cpu_to_socket: (0..cpu_count).map(|i| (i, 0)).collect(),
            memory_per_node_mb: vec![8192],
        })
    }

    #[cfg(unix)]
    fn parse_memory_mb(meminfo: &str) -> usize {
        // Parse "Node X MemTotal: Y kB" format
        for line in meminfo.lines() {
            if line.contains("MemTotal:") {
                let parts: Vec<&str> = line.split_whitespace().collect();
                if parts.len() >= 3 {
                    if let Ok(kb) = parts[parts.len() - 2].parse::<usize>() {
                        return kb / 1024; // Convert KB to MB
                    }
                }
            }
        }
        1024 // Default 1GB
    }

    #[cfg(unix)]
    fn parse_cpu_list(cpulist: &str) -> Vec<usize> {
        let mut cpus = Vec::new();

        for part in cpulist.trim().split(',') {
            if part.contains('-') {
                // Range like "0-7"
                let range: Vec<&str> = part.split('-').collect();
                if range.len() == 2 {
                    if let (Ok(start), Ok(end)) =
                        (range[0].parse::<usize>(), range[1].parse::<usize>())
                    {
                        for cpu in start..=end {
                            cpus.push(cpu);
                        }
                    }
                }
            } else {
                // Single CPU
                if let Ok(cpu) = part.parse::<usize>() {
                    cpus.push(cpu);
                }
            }
        }

        cpus
    }

    /// Get the socket for the current thread
    pub fn current_socket(&self) -> usize {
        // Try to get current CPU and map to socket (Linux only)
        #[cfg(target_os = "linux")]
        {
            let cpu = unsafe { libc::sched_getcpu() };
            if cpu >= 0 {
                return self
                    .cpu_to_socket
                    .get(&(cpu as usize))
                    .copied()
                    .unwrap_or(0);
            }
        }

        0 // Default to socket 0
    }

    /// Suggest NUMA placement for temporal clustering
    pub fn suggest_socket_for_timestamp(&self, timestamp_ns: u64) -> usize {
        // Hash timestamp to distribute across sockets
        let hash = timestamp_ns.wrapping_mul(0x9e3779b9);
        (hash as usize) % self.socket_count
    }
}

/// NUMA policy for memory mapping
#[derive(Debug, Clone, Copy)]
pub enum NumaPolicy {
    /// Default system policy
    Default,
    /// Allocate on specific node
    Bind(usize),
    /// Interleave across all nodes
    Interleaved,
    /// Prefer specific node but allow fallback
    Preferred(usize),
}

/// NUMA-aware memory mapping wrapper
pub struct NumaMemoryMap {
    mapping: *mut u8,
    size: usize,
    numa_policy: NumaPolicy,
    topology: Arc<NumaTopology>,
}

impl NumaMemoryMap {
    /// Create a new NUMA-aware memory mapping
    pub fn new(
        size: usize,
        policy: NumaPolicy,
        topology: Arc<NumaTopology>,
    ) -> StorageResult<Self> {
        let mapping = Self::allocate_mapping(size, policy)?;

        Ok(Self {
            mapping,
            size,
            numa_policy: policy,
            topology,
        })
    }

    /// Create interleaved mapping for balanced access
    pub fn new_interleaved(size: usize, topology: Arc<NumaTopology>) -> StorageResult<Self> {
        Self::new(size, NumaPolicy::Interleaved, topology)
    }

    /// Create socket-local mapping
    pub fn new_socket_local(
        size: usize,
        socket: usize,
        topology: Arc<NumaTopology>,
    ) -> StorageResult<Self> {
        Self::new(size, NumaPolicy::Bind(socket), topology)
    }

    #[cfg(unix)]
    fn allocate_mapping(size: usize, policy: NumaPolicy) -> StorageResult<*mut u8> {
        // Align size to page boundary
        let page_size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) } as usize;
        let aligned_size = (size + page_size - 1) & !(page_size - 1);

        // Create anonymous mapping
        let mapping = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                aligned_size,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                -1,
                0,
            )
        };

        if mapping == libc::MAP_FAILED {
            return Err(StorageError::mmap_failed("Anonymous mapping failed"));
        }

        // Try to use huge pages if available
        #[cfg(target_os = "linux")]
        {
            unsafe {
                libc::madvise(mapping, aligned_size, libc::MADV_HUGEPAGE);
            }
        }

        // Apply NUMA policy
        match policy {
            NumaPolicy::Default => {
                // No special policy
            }
            NumaPolicy::Bind(node) => {
                #[cfg(target_os = "linux")]
                {
                    let nodemask = 1u64 << node;
                    unsafe {
                        libc::syscall(
                            libc::SYS_mbind,
                            mapping,
                            aligned_size,
                            libc::MPOL_BIND,
                            &nodemask as *const u64,
                            64, // maxnode
                            0,  // flags
                        );
                    }
                }
            }
            NumaPolicy::Interleaved => {
                #[cfg(target_os = "linux")]
                {
                    let nodemask = !0u64; // All nodes
                    unsafe {
                        libc::syscall(
                            libc::SYS_mbind,
                            mapping,
                            aligned_size,
                            libc::MPOL_INTERLEAVE,
                            &nodemask as *const u64,
                            64, // maxnode
                            0,  // flags
                        );
                    }
                }
            }
            NumaPolicy::Preferred(node) => {
                #[cfg(target_os = "linux")]
                {
                    let nodemask = 1u64 << node;
                    unsafe {
                        libc::syscall(
                            libc::SYS_mbind,
                            mapping,
                            aligned_size,
                            libc::MPOL_PREFERRED,
                            &nodemask as *const u64,
                            64, // maxnode
                            0,  // flags
                        );
                    }
                }
            }
        }

        Ok(mapping as *mut u8)
    }

    #[cfg(not(unix))]
    fn allocate_mapping(size: usize, _policy: NumaPolicy) -> StorageResult<*mut u8> {
        // Fallback allocation on non-Unix systems
        use std::alloc::{Layout, alloc};

        let layout = Layout::from_size_align(size, 64)
            .map_err(|e| StorageError::allocation_failed(&format!("Layout error: {}", e)))?;

        let ptr = unsafe { alloc(layout) };

        if ptr.is_null() {
            Err(StorageError::allocation_failed("System allocation failed"))
        } else {
            Ok(ptr)
        }
    }

    /// Get raw pointer to mapped memory
    pub fn as_ptr(&self) -> *mut u8 {
        self.mapping
    }

    /// Get size of mapping
    pub fn size(&self) -> usize {
        self.size
    }

    /// Get slice view of mapped memory
    pub unsafe fn as_slice(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.mapping, self.size) }
    }

    /// Get mutable slice view of mapped memory
    pub unsafe fn as_mut_slice(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.mapping, self.size) }
    }

    /// Prefetch memory pages
    pub fn prefetch(&self, offset: usize, len: usize) {
        if offset + len <= self.size {
            #[cfg(unix)]
            unsafe {
                libc::madvise(
                    self.mapping.add(offset) as *mut libc::c_void,
                    len,
                    libc::MADV_WILLNEED,
                );
            }
        }
    }

    /// Advise on memory access pattern
    pub fn advise_sequential(&self) {
        #[cfg(unix)]
        unsafe {
            libc::madvise(
                self.mapping as *mut libc::c_void,
                self.size,
                libc::MADV_SEQUENTIAL,
            );
        }
    }

    pub fn advise_random(&self) {
        #[cfg(unix)]
        unsafe {
            libc::madvise(
                self.mapping as *mut libc::c_void,
                self.size,
                libc::MADV_RANDOM,
            );
        }
    }

    /// Lock pages in memory to prevent swapping
    pub fn lock_in_memory(&self) -> StorageResult<()> {
        #[cfg(unix)]
        {
            let result = unsafe { libc::mlock(self.mapping as *const libc::c_void, self.size) };

            if result != 0 {
                return Err(StorageError::allocation_failed(
                    "Failed to lock pages in memory",
                ));
            }
        }

        Ok(())
    }
}

impl Drop for NumaMemoryMap {
    fn drop(&mut self) {
        if !self.mapping.is_null() {
            #[cfg(unix)]
            unsafe {
                libc::munmap(self.mapping as *mut libc::c_void, self.size);
            }

            #[cfg(not(unix))]
            unsafe {
                use std::alloc::{Layout, dealloc};
                let layout = Layout::from_size_align_unchecked(self.size, 64);
                dealloc(self.mapping, layout);
            }
        }
    }
}

unsafe impl Send for NumaMemoryMap {}
unsafe impl Sync for NumaMemoryMap {}

/// NUMA-aware allocator for embedding blocks
pub struct NumaAllocator {
    topology: Arc<NumaTopology>,
    socket_pools: Vec<parking_lot::Mutex<Vec<*mut u8>>>,
}

impl NumaAllocator {
    pub fn new(topology: Arc<NumaTopology>) -> Self {
        let socket_pools = (0..topology.socket_count)
            .map(|_| parking_lot::Mutex::new(Vec::new()))
            .collect();

        Self {
            topology,
            socket_pools,
        }
    }

    /// Allocate aligned memory on preferred socket
    pub fn allocate_aligned(
        &self,
        size: usize,
        alignment: usize,
        preferred_socket: usize,
    ) -> StorageResult<*mut u8> {
        let socket = preferred_socket.min(self.topology.socket_count - 1);

        // Try to reuse from pool first
        if let Some(ptr) = self.socket_pools[socket].lock().pop() {
            return Ok(ptr);
        }

        // Allocate new mapping on the socket
        let policy = NumaPolicy::Preferred(socket);
        let mapping = Self::allocate_mapping(size, alignment, policy)?;
        Ok(mapping)
    }

    #[cfg(unix)]
    fn allocate_mapping(
        size: usize,
        alignment: usize,
        policy: NumaPolicy,
    ) -> StorageResult<*mut u8> {
        // Align size to page boundary
        let page_size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) } as usize;
        let aligned_size = (size + page_size - 1) & !(page_size - 1);

        // Allocate with posix_memalign for guaranteed alignment
        let mut ptr: *mut libc::c_void = std::ptr::null_mut();
        let result = unsafe { libc::posix_memalign(&mut ptr, alignment, aligned_size) };

        if result != 0 {
            return Err(StorageError::allocation_failed("posix_memalign failed"));
        }

        // Apply NUMA policy if on Linux
        #[cfg(target_os = "linux")]
        match policy {
            NumaPolicy::Preferred(node) => {
                let nodemask = 1u64 << node;
                unsafe {
                    libc::syscall(
                        libc::SYS_mbind,
                        ptr,
                        aligned_size,
                        libc::MPOL_PREFERRED,
                        &nodemask as *const u64,
                        64, // maxnode
                        0,  // flags
                    );
                }
            }
            _ => {}
        }

        Ok(ptr as *mut u8)
    }

    #[cfg(not(unix))]
    fn allocate_mapping(
        size: usize,
        alignment: usize,
        _policy: NumaPolicy,
    ) -> StorageResult<*mut u8> {
        use std::alloc::{Layout, alloc};

        let layout = Layout::from_size_align(size, alignment)
            .map_err(|e| StorageError::allocation_failed(&format!("Layout error: {}", e)))?;

        let ptr = unsafe { alloc(layout) };

        if ptr.is_null() {
            Err(StorageError::allocation_failed("Aligned allocation failed"))
        } else {
            Ok(ptr)
        }
    }

    /// Deallocate memory back to socket pool
    pub fn deallocate(&self, ptr: *mut u8, socket: usize) {
        if !ptr.is_null() && socket < self.socket_pools.len() {
            self.socket_pools[socket].lock().push(ptr);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_numa_topology_detection() {
        let topology = NumaTopology::detect().unwrap();
        assert!(topology.socket_count > 0);
        assert!(topology.node_count > 0);
    }

    #[test]
    fn test_numa_memory_mapping() {
        let topology = Arc::new(NumaTopology::detect().unwrap());
        let mapping = NumaMemoryMap::new_interleaved(4096, topology).unwrap();

        assert_eq!(mapping.size(), 4096);
        assert!(!mapping.as_ptr().is_null());

        // Test basic memory access
        unsafe {
            let slice = mapping.as_slice();
            assert_eq!(slice.len(), 4096);
        }
    }

    #[test]
    fn test_numa_allocator() {
        let topology = Arc::new(NumaTopology::detect().unwrap());
        let allocator = NumaAllocator::new(topology);

        let ptr = allocator.allocate_aligned(1024, 64, 0).unwrap();
        assert!(!ptr.is_null());

        // Test alignment
        assert_eq!(ptr as usize % 64, 0);

        allocator.deallocate(ptr, 0);
    }
}
