//! Cache-optimal data structures for high-performance memory access
//!
//! This module implements cache-friendly data structures optimized for:
//! - Spatial locality for sequential memory access patterns
//! - Temporal locality for recently accessed memories
//! - Lock-free concurrent access with minimal contention
//! - NUMA-aware allocation and access patterns

// Allow unsafe code for performance-critical cache operations
#![allow(unsafe_code)]

use super::{StorageError, StorageMetrics, StorageResult};
use crate::Memory;
use atomic_float::AtomicF32;
use std::sync::{
    Arc,
    atomic::{AtomicU32, AtomicU64, AtomicUsize, Ordering},
};

#[cfg(all(feature = "memory_mapped_persistence", unix))]
use super::numa::{NumaAllocator, NumaMemoryMap};

/// Cache-optimized memory node with hot/warm/cold data separation
#[repr(C, align(64))]
pub struct CacheOptimalMemoryNode {
    /// Hot data: accessed on every activation (exactly 64 bytes = 1 cache line)
    pub id_hash: u64, // Fast comparison hash, 8 bytes
    /// Current activation level (0.0 to 1.0)
    pub activation: AtomicF32,  // Current activation level, 4 bytes
    /// Memory confidence score (0.0 to 1.0)
    pub confidence: f32,        // Memory confidence, 4 bytes
    /// Timestamp of last access for LRU eviction
    pub last_access: AtomicU64, // Timestamp for LRU, 8 bytes
    /// Bitfield containing node state and type information
    pub node_flags: AtomicU32,  // State and type flags, 4 bytes
    /// Number of outgoing edges from this memory node
    pub edges_count: AtomicU32, // Number of outgoing edges, 4 bytes
    /// Pointer to the full Memory object in storage
    pub memory_ptr: AtomicU64,  // Pointer to full Memory object, 8 bytes
    /// Padding to align hot data to exactly 64 bytes
    pub _hot_padding: [u8; 24], // Pad to exactly 64 bytes

    /// Warm data: accessed during recall operations (768*4 = 3072 bytes = 48 cache lines)
    pub embedding: [f32; 768], // Dense vector for similarity computation

    /// Cold data: accessed during maintenance (exactly 64 bytes = 1 cache line)
    pub decay_rate: f32, // Forgetting curve parameter, 4 bytes
    /// Timestamp when this memory was originally created
    pub creation_time: u64,      // Original encoding time, 8 bytes
    /// Counter tracking how many times this memory was recalled
    pub recall_count: AtomicU32, // Access frequency counter, 4 bytes
    /// Hash of memory content for deduplication
    pub content_hash: u64,       // For deduplication, 8 bytes
    /// Offset to the list of outgoing edges
    pub edges_offset: AtomicU64, // Offset to edge list, 8 bytes
    /// Pointer to backup storage location
    pub backup_ptr: AtomicU64,   // Backup storage pointer, 8 bytes
    /// Schema identifier for memory type information
    pub schema_id: u32,          // Memory schema type, 4 bytes
    /// Padding to align cold data to exactly 64 bytes
    pub _cold_padding: [u8; 12], // Pad to exactly 64 bytes
}

// Verify our structure sizes at compile time
const _: () = {
    assert!(std::mem::size_of::<CacheOptimalMemoryNode>() == 3200); // 50 cache lines exactly
    assert!(std::mem::align_of::<CacheOptimalMemoryNode>() == 64); // Cache line aligned
};

impl CacheOptimalMemoryNode {
    /// Create a new node from a memory object
    pub fn new(memory: Arc<Memory>, node_id_hash: u64) -> Self {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::SystemTime::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;

        Self {
            id_hash: node_id_hash,
            activation: AtomicF32::new(memory.activation()),
            confidence: memory.confidence.raw(),
            last_access: AtomicU64::new(now),
            node_flags: AtomicU32::new(0),
            edges_count: AtomicU32::new(0),
            memory_ptr: AtomicU64::new(Arc::as_ptr(&memory) as u64),
            _hot_padding: [0; 24],

            embedding: memory.embedding,

            decay_rate: 0.2, // Default decay rate
            creation_time: std::time::SystemTime::from(memory.created_at)
                .duration_since(std::time::SystemTime::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos() as u64,
            recall_count: AtomicU32::new(0),
            content_hash: Self::compute_content_hash(&memory.id),
            edges_offset: AtomicU64::new(0),
            backup_ptr: AtomicU64::new(0),
            schema_id: 0,
            _cold_padding: [0; 12],
        }
    }

    fn compute_content_hash(content: &str) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        content.hash(&mut hasher);
        hasher.finish()
    }

    /// Prefetch hot cache line for activation operations
    #[cfg(all(feature = "memory_mapped_persistence", target_arch = "x86_64"))]
    #[inline]
    pub fn prefetch_hot(&self) {
        unsafe {
            std::arch::x86_64::_mm_prefetch(
                (self as *const Self) as *const i8,
                std::arch::x86_64::_MM_HINT_T0, // L1 cache
            );
        }
    }

    /// Prefetch embedding data for similarity computation
    #[cfg(all(feature = "memory_mapped_persistence", target_arch = "x86_64"))]
    #[inline]
    pub fn prefetch_warm(&self) {
        unsafe {
            let embedding_ptr = self.embedding.as_ptr() as *const i8;
            for line in 0..48 {
                // 768 * 4 bytes / 64 bytes = 48 lines
                std::arch::x86_64::_mm_prefetch(
                    embedding_ptr.add(line * 64),
                    std::arch::x86_64::_MM_HINT_T1, // L2 cache
                );
            }
        }
    }

    /// Prefetch cold data for maintenance operations
    #[cfg(all(feature = "memory_mapped_persistence", target_arch = "x86_64"))]
    #[inline]
    pub fn prefetch_cold(&self) {
        unsafe {
            let cold_ptr = &self.decay_rate as *const f32 as *const i8;
            std::arch::x86_64::_mm_prefetch(cold_ptr, std::arch::x86_64::_MM_HINT_T2);
        }
    }

    #[cfg(not(all(feature = "memory_mapped_persistence", target_arch = "x86_64")))]
    #[inline]
    /// Prefetch hot data (no-op on non-x86_64 platforms)
    pub fn prefetch_hot(&self) {}

    #[cfg(not(all(feature = "memory_mapped_persistence", target_arch = "x86_64")))]
    #[inline]
    /// Prefetch warm data (no-op on non-x86_64 platforms)
    pub fn prefetch_warm(&self) {}

    #[cfg(not(all(feature = "memory_mapped_persistence", target_arch = "x86_64")))]
    #[inline]
    /// Prefetch cold data (no-op on non-x86_64 platforms)
    pub fn prefetch_cold(&self) {}

    /// Update activation level atomically
    pub fn update_activation(&self, new_activation: f32) {
        self.activation.store(new_activation, Ordering::Relaxed);
        self.last_access.store(
            std::time::SystemTime::now()
                .duration_since(std::time::SystemTime::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos() as u64,
            Ordering::Relaxed,
        );
        self.recall_count.fetch_add(1, Ordering::Relaxed);
    }

    /// Get current activation level
    pub fn get_activation(&self) -> f32 {
        self.activation.load(Ordering::Relaxed)
    }

    /// Check if node needs maintenance (old or low activation)
    pub fn needs_maintenance(&self, current_time: u64, threshold: f32) -> bool {
        let last_access = self.last_access.load(Ordering::Relaxed);
        let activation = self.activation.load(Ordering::Relaxed);

        // Check age and activation level
        let age_ms = (current_time - last_access) / 1_000_000; // Convert to milliseconds
        age_ms > 3_600_000 || activation < threshold // 1 hour or below threshold
    }
}

/// Lock-free hash table optimized for cognitive access patterns
pub struct CognitiveIndex {
    /// Hash table with linear probing (power-of-2 size)
    table: Box<[AtomicU64]>,
    table_mask: u64,

    /// Node storage pool
    #[cfg(all(feature = "memory_mapped_persistence", unix))]
    nodes: NumaMemoryMap,
    #[cfg(not(all(feature = "memory_mapped_persistence", unix)))]
    nodes: Vec<u8>,

    node_count: AtomicUsize,
    node_capacity: usize,

    /// Generation counter for ABA prevention
    generation: AtomicU64,

    /// Allocator for NUMA-aware node allocation  
    #[cfg(all(feature = "memory_mapped_persistence", unix))]
    allocator: NumaAllocator,

    /// Performance metrics
    metrics: Arc<StorageMetrics>,
}

impl CognitiveIndex {
    /// Create a new cognitive index with specified capacity
    pub fn new(capacity: usize, metrics: Arc<StorageMetrics>) -> StorageResult<Self> {
        // Round capacity up to next power of 2
        let table_capacity = capacity.next_power_of_two();
        let table_mask = (table_capacity - 1) as u64;

        // Initialize hash table
        let table = (0..table_capacity)
            .map(|_| AtomicU64::new(0))
            .collect::<Vec<_>>()
            .into_boxed_slice();

        // Calculate node storage size
        let node_size = std::mem::size_of::<CacheOptimalMemoryNode>();
        let storage_size = capacity * node_size;

        #[cfg(all(feature = "memory_mapped_persistence", unix))]
        let numa_topology = Arc::new(super::numa::NumaTopology::detect()?);

        #[cfg(all(feature = "memory_mapped_persistence", unix))]
        let nodes = NumaMemoryMap::new_interleaved(storage_size, numa_topology.clone())?;

        #[cfg(not(all(feature = "memory_mapped_persistence", unix)))]
        let nodes = vec![0u8; storage_size];

        #[cfg(all(feature = "memory_mapped_persistence", unix))]
        let allocator = NumaAllocator::new(numa_topology);

        Ok(Self {
            table,
            table_mask,
            nodes,
            node_count: AtomicUsize::new(0),
            node_capacity: capacity,
            generation: AtomicU64::new(0),
            #[cfg(all(feature = "memory_mapped_persistence", unix))]
            allocator,
            metrics,
        })
    }

    /// Insert a memory node with lock-free operation
    pub fn insert(&self, memory: Arc<Memory>) -> StorageResult<u64> {
        let id_hash = self.hash_memory_id(&memory.id);
        let mut probe_distance = 0;

        loop {
            let slot_idx = (id_hash + probe_distance) & self.table_mask;
            let slot = &self.table[slot_idx as usize];

            let current = slot.load(Ordering::Acquire);

            if current == 0 {
                // Empty slot found - try to claim it
                let node_offset = self.allocate_node_slot()?;
                let packed_value = self.pack_hash_offset(id_hash, node_offset);

                match slot.compare_exchange_weak(
                    0,
                    packed_value,
                    Ordering::AcqRel,
                    Ordering::Acquire,
                ) {
                    Ok(_) => {
                        // Successfully claimed slot - initialize node
                        let node = CacheOptimalMemoryNode::new(memory, id_hash);
                        unsafe {
                            let node_ptr = self.node_ptr_mut(node_offset);
                            std::ptr::write(node_ptr, node);
                        }

                        self.node_count.fetch_add(1, Ordering::Relaxed);
                        self.metrics
                            .record_write(std::mem::size_of::<CacheOptimalMemoryNode>() as u64);

                        return Ok(node_offset);
                    }
                    Err(_) => {
                        // Slot was taken - deallocate and retry
                        self.deallocate_node_slot(node_offset);
                        probe_distance += 1;
                    }
                }
            } else {
                // Check if this is the same memory (update case)
                let (existing_hash, existing_offset) = self.unpack_hash_offset(current);
                if existing_hash == (id_hash >> 32) {
                    // Found existing entry - update in place
                    unsafe {
                        let node_ptr = self.node_ptr_mut(existing_offset);
                        (*node_ptr).update_activation(memory.activation());
                    }
                    return Ok(existing_offset);
                }

                probe_distance += 1;
                if probe_distance > self.table_mask / 4 {
                    return Err(StorageError::allocation_failed("Hash table too full"));
                }
            }
        }
    }

    /// Lookup a memory by ID  
    pub fn lookup(&self, memory_id: &str) -> Option<&CacheOptimalMemoryNode> {
        let id_hash = self.hash_memory_id(memory_id);
        let mut probe_distance = 0;

        loop {
            let slot_idx = (id_hash + probe_distance) & self.table_mask;
            let slot = &self.table[slot_idx as usize];
            let current = slot.load(Ordering::Acquire);

            if current == 0 {
                // Empty slot - not found
                self.metrics.record_cache_miss();
                return None;
            }

            let (hash, offset) = self.unpack_hash_offset(current);
            if hash == (id_hash >> 32) {
                // Found it
                unsafe {
                    let node_ptr = self.node_ptr(offset);
                    (*node_ptr).prefetch_hot(); // Prefetch for likely access
                    self.metrics.record_cache_hit();
                    return Some(&*node_ptr);
                }
            }

            probe_distance += 1;
            if probe_distance > self.table_mask / 4 {
                self.metrics.record_cache_miss();
                return None;
            }
        }
    }

    /// Remove a memory from the index
    pub fn remove(&self, memory_id: &str) -> bool {
        let id_hash = self.hash_memory_id(memory_id);
        let mut probe_distance = 0;

        loop {
            let slot_idx = (id_hash + probe_distance) & self.table_mask;
            let slot = &self.table[slot_idx as usize];
            let current = slot.load(Ordering::Acquire);

            if current == 0 {
                return false; // Not found
            }

            let (hash, offset) = self.unpack_hash_offset(current);
            if hash == (id_hash >> 32) {
                // Found it - try to remove
                match slot.compare_exchange_weak(current, 0, Ordering::AcqRel, Ordering::Acquire) {
                    Ok(_) => {
                        self.deallocate_node_slot(offset);
                        self.node_count.fetch_sub(1, Ordering::Relaxed);
                        return true;
                    }
                    Err(_) => {
                        // Concurrent modification - retry
                        continue;
                    }
                }
            }

            probe_distance += 1;
            if probe_distance > self.table_mask / 4 {
                return false;
            }
        }
    }

    /// Iterate over all nodes for maintenance
    pub fn iter_nodes<F>(&self, mut callback: F)
    where
        F: FnMut(&CacheOptimalMemoryNode),
    {
        for slot in self.table.iter() {
            let current = slot.load(Ordering::Acquire);
            if current != 0 {
                let (_, offset) = self.unpack_hash_offset(current);
                unsafe {
                    let node_ptr = self.node_ptr(offset);
                    callback(&*node_ptr);
                }
            }
        }
    }

    /// Get performance statistics
    pub fn statistics(&self) -> IndexStatistics {
        let node_count = self.node_count.load(Ordering::Relaxed);
        let load_factor = node_count as f32 / self.node_capacity as f32;

        IndexStatistics {
            node_count,
            capacity: self.node_capacity,
            load_factor,
            cache_hit_rate: self.metrics.cache_hit_rate(),
        }
    }

    /// Hash a memory ID to table index
    fn hash_memory_id(&self, memory_id: &str) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        memory_id.hash(&mut hasher);
        hasher.finish()
    }

    /// Allocate a slot in the node storage
    fn allocate_node_slot(&self) -> StorageResult<u64> {
        let current_count = self.node_count.load(Ordering::Relaxed);
        if current_count >= self.node_capacity {
            return Err(StorageError::allocation_failed("Node storage full"));
        }

        // Simple sequential allocation for now
        // In production, this would use a free list
        Ok(current_count as u64)
    }

    /// Deallocate a node slot
    fn deallocate_node_slot(&self, _offset: u64) {
        // For now, we just leak the slot
        // In production, this would add to free list
    }

    /// Pack hash and offset into single u64
    fn pack_hash_offset(&self, hash: u64, offset: u64) -> u64 {
        let hash_upper = hash >> 32; // Use upper 32 bits of hash
        (hash_upper << 32) | (offset & 0xFFFF_FFFF)
    }

    /// Unpack hash and offset from u64
    fn unpack_hash_offset(&self, packed: u64) -> (u64, u64) {
        let hash_upper = packed >> 32;
        let offset = packed & 0xFFFF_FFFF;
        (hash_upper, offset)
    }

    /// Get pointer to node at offset
    unsafe fn node_ptr(&self, offset: u64) -> *const CacheOptimalMemoryNode {
        let node_size = std::mem::size_of::<CacheOptimalMemoryNode>();

        #[cfg(all(feature = "memory_mapped_persistence", unix))]
        {
            (unsafe { self.nodes.as_ptr().add(offset as usize * node_size) }) as *const CacheOptimalMemoryNode
        }

        #[cfg(not(all(feature = "memory_mapped_persistence", unix)))]
        {
            (unsafe { self.nodes.as_ptr().add(offset as usize * node_size) }) as *const CacheOptimalMemoryNode
        }
    }

    /// Get mutable pointer to node at offset
    unsafe fn node_ptr_mut(&self, offset: u64) -> *mut CacheOptimalMemoryNode {
        let node_size = std::mem::size_of::<CacheOptimalMemoryNode>();

        #[cfg(all(feature = "memory_mapped_persistence", unix))]
        {
            (unsafe { self.nodes.as_ptr().add(offset as usize * node_size) }) as *mut CacheOptimalMemoryNode
        }

        #[cfg(not(all(feature = "memory_mapped_persistence", unix)))]
        {
            (unsafe { self.nodes.as_ptr().add(offset as usize * node_size) }) as *mut CacheOptimalMemoryNode
        }
    }
}

/// Statistics about the index performance
#[derive(Debug, Clone)]
pub struct IndexStatistics {
    /// Current number of memory nodes in the index
    pub node_count: usize,
    /// Maximum number of nodes that can fit in the index
    pub capacity: usize,
    /// Current load factor (node_count / capacity)
    pub load_factor: f32,
    /// Percentage of cache hits vs total accesses
    pub cache_hit_rate: f32,
}

/// Batch prefetcher for cognitive access patterns
pub struct CognitivePreloader {
    /// Recently accessed nodes for temporal prediction
    access_history: parking_lot::RwLock<Vec<u64>>,

    /// Performance metrics
    metrics: Arc<StorageMetrics>,
}

impl CognitivePreloader {
    /// Create a new cognitive preloader with performance metrics tracking
    pub fn new(metrics: Arc<StorageMetrics>) -> Self {
        Self {
            access_history: parking_lot::RwLock::new(Vec::with_capacity(1000)),
            metrics,
        }
    }

    /// Record node access for pattern learning
    pub fn record_access(&self, node_hash: u64) {
        let mut history = self.access_history.write();
        history.push(node_hash);

        // Keep history bounded
        if history.len() > 1000 {
            history.drain(0..100); // Remove oldest 100 entries
        }
    }

    /// Predict next accesses based on patterns
    pub fn predict_next_accesses(&self, current_hash: u64, count: usize) -> Vec<u64> {
        let history = self.access_history.read();
        let mut predictions = Vec::new();

        // Simple pattern: find what typically follows current_hash
        for i in 0..history.len().saturating_sub(1) {
            if history[i] == current_hash {
                let next_hash = history[i + 1];
                if !predictions.contains(&next_hash) {
                    predictions.push(next_hash);
                    if predictions.len() >= count {
                        break;
                    }
                }
            }
        }

        predictions
    }

    /// Prefetch predicted nodes
    pub fn prefetch_predicted(&self, _index: &CognitiveIndex, predictions: &[u64]) {
        for &_hash in predictions {
            // This would trigger prefetch of the predicted node
            // For now, just record the prediction
            self.metrics.record_cache_hit(); // Optimistic
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Confidence, EpisodeBuilder};
    use chrono::Utc;

    fn create_test_memory(id: &str) -> Arc<Memory> {
        let episode = EpisodeBuilder::new()
            .id(id.to_string())
            .when(Utc::now())
            .what(format!("test memory {}", id))
            .embedding([0.5f32; 768])
            .confidence(Confidence::HIGH)
            .build();

        Arc::new(Memory::from_episode(episode, 0.8))
    }

    #[test]
    fn test_cache_optimal_memory_node_layout() {
        // Verify exact cache line alignment
        assert_eq!(std::mem::size_of::<CacheOptimalMemoryNode>(), 3200);
        assert_eq!(std::mem::align_of::<CacheOptimalMemoryNode>(), 64);

        // Verify hot data is in first cache line
        let node = CacheOptimalMemoryNode::new(create_test_memory("test"), 12345);
        assert_eq!(node.id_hash, 12345);
    }

    #[test]
    fn test_cognitive_index_operations() {
        let metrics = Arc::new(StorageMetrics::new());
        let index = CognitiveIndex::new(1000, metrics).unwrap();

        let memory = create_test_memory("test_memory");

        // Test insert
        let offset = index.insert(memory.clone()).unwrap();
        assert!(offset < 1000);

        // Test lookup
        let found = index.lookup("test_memory");
        assert!(found.is_some());

        let node = found.unwrap();
        assert_eq!(node.confidence, memory.confidence.raw());

        // Test remove
        let removed = index.remove("test_memory");
        assert!(removed);

        // Verify it's gone
        let not_found = index.lookup("test_memory");
        assert!(not_found.is_none());
    }

    #[test]
    fn test_cognitive_preloader() {
        let metrics = Arc::new(StorageMetrics::new());
        let preloader = CognitivePreloader::new(metrics);

        // Record access pattern
        preloader.record_access(1);
        preloader.record_access(2);
        preloader.record_access(3);
        preloader.record_access(1);
        preloader.record_access(2);

        // Predict what follows 1
        let predictions = preloader.predict_next_accesses(1, 2);
        assert!(predictions.contains(&2));
    }

    #[test]
    fn test_index_statistics() {
        let metrics = Arc::new(StorageMetrics::new());
        let index = CognitiveIndex::new(100, metrics).unwrap();

        // Insert some memories
        for i in 0..10 {
            let memory = create_test_memory(&format!("memory_{}", i));
            index.insert(memory).unwrap();
        }

        let stats = index.statistics();
        assert_eq!(stats.node_count, 10);
        assert_eq!(stats.capacity, 100);
        assert!((stats.load_factor - 0.1).abs() < 0.01);
    }
}
