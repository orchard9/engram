//! Memory pool primitives for activation spreading
//!
//! Provides legacy byte-oriented arenas as well as the lock-free
//! [`ActivationRecordPool`] used by the spreading engine to recycle
//! `ActivationRecord` instances with minimal contention.

#[cfg(not(loom))]
use crossbeam_queue::SegQueue;
#[cfg(loom)]
use loom::sync::Arc;
#[cfg(loom)]
use loom::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use parking_lot::Mutex;
use std::cell::RefCell;
use std::collections::HashMap;
#[cfg(loom)]
use std::collections::VecDeque;
use std::mem;
#[cfg(not(loom))]
use std::sync::Arc;
#[cfg(not(loom))]
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

use super::{ActivationRecord, NodeId, storage_aware::StorageTier};

#[cfg(loom)]
type GlobalQueue<T> = loom::sync::Mutex<VecDeque<T>>;
#[cfg(not(loom))]
type GlobalQueue<T> = SegQueue<T>;

#[cfg(loom)]
fn global_queue_new<T>() -> GlobalQueue<T> {
    GlobalQueue::new(VecDeque::new())
}

#[cfg(not(loom))]
const fn global_queue_new<T>() -> GlobalQueue<T> {
    GlobalQueue::new()
}

#[cfg(loom)]
fn global_queue_push<T>(queue: &GlobalQueue<T>, value: T) {
    queue.lock().push_back(value);
}

#[cfg(not(loom))]
fn global_queue_push<T>(queue: &GlobalQueue<T>, value: T) {
    queue.push(value);
}

#[cfg(loom)]
fn global_queue_pop<T>(queue: &GlobalQueue<T>) -> Option<T> {
    queue.lock().pop_front()
}

#[cfg(not(loom))]
fn global_queue_pop<T>(queue: &GlobalQueue<T>) -> Option<T> {
    queue.pop()
}

// ---------------------------------------------------------------------------
// Legacy chunk-based arena (kept for backwards compatibility)
// ---------------------------------------------------------------------------

/// Memory pool for activation records and traversal state
pub struct ActivationMemoryPool {
    /// Pre-allocated chunks of memory
    chunks: Arc<Mutex<Vec<MemoryChunk>>>,
    /// Size of each chunk in bytes
    chunk_size: usize,
    /// Maximum number of chunks to keep in pool
    max_chunks: usize,
}

/// A chunk of pre-allocated memory
struct MemoryChunk {
    data: Vec<u8>,
    used: usize,
}

impl MemoryChunk {
    fn new(size: usize) -> Self {
        Self {
            data: vec![0u8; size],
            used: 0,
        }
    }

    const fn reset(&mut self) {
        self.used = 0;
    }

    const fn has_space(&self, size: usize) -> bool {
        self.used + size <= self.data.len()
    }

    fn allocate(&mut self, size: usize) -> Option<&mut [u8]> {
        if !self.has_space(size) {
            return None;
        }

        let start = self.used;
        self.used += size;
        Some(&mut self.data[start..start + size])
    }
}

impl ActivationMemoryPool {
    /// Create a new memory pool
    #[must_use]
    pub fn new(chunk_size: usize, max_chunks: usize) -> Self {
        let initial_chunks = vec![MemoryChunk::new(chunk_size)];
        Self {
            chunks: Arc::new(Mutex::new(initial_chunks)),
            chunk_size,
            max_chunks,
        }
    }

    /// Allocate memory from the pool
    #[must_use]
    pub fn allocate(&self, size: usize) -> PooledAllocation {
        let mut chunks = self.chunks.lock();

        // Try to find a chunk with space
        for chunk in chunks.iter_mut() {
            if let Some(memory) = chunk.allocate(size) {
                return PooledAllocation {
                    data: memory.as_mut_ptr(),
                    size,
                    pool: None,
                };
            }
        }

        // Need a new chunk
        if chunks.len() < self.max_chunks {
            let mut new_chunk = MemoryChunk::new(self.chunk_size.max(size));
            let Some(memory) = new_chunk.allocate(size) else {
                unreachable!("Fresh chunk should have capacity for requested size")
            };
            let ptr = memory.as_mut_ptr();
            chunks.push(new_chunk);
            drop(chunks);
            PooledAllocation {
                data: ptr,
                size,
                pool: None,
            }
        } else {
            // Fall back to regular allocation
            let mut vec = vec![0u8; size];
            PooledAllocation {
                data: vec.as_mut_ptr(),
                size,
                pool: None,
            }
        }
    }

    /// Reset all chunks for reuse
    pub fn reset(&self) {
        let mut chunks = self.chunks.lock();
        for chunk in chunks.iter_mut() {
            chunk.reset();
        }
    }

    /// Get statistics about pool usage
    #[must_use]
    pub fn stats(&self) -> PoolStats {
        let chunks = self.chunks.lock();
        let total_capacity = chunks.len() * self.chunk_size;
        let total_used: usize = chunks.iter().map(|c| c.used).sum();

        PoolStats {
            num_chunks: chunks.len(),
            chunk_size: self.chunk_size,
            total_capacity,
            total_used,
            utilization: if total_capacity > 0 {
                total_used as f32 / total_capacity as f32
            } else {
                0.0
            },
        }
    }
}

/// An allocation from the memory pool
pub struct PooledAllocation {
    data: *mut u8,
    size: usize,
    /// Pool reference reserved for future RAII-based deallocation on drop
    #[allow(dead_code)]
    pool: Option<Arc<ActivationMemoryPool>>,
}

impl PooledAllocation {
    /// Get the allocated memory as a slice
    ///
    /// # Safety
    /// Safe because pointer and size are guaranteed valid by pool allocation
    #[allow(unsafe_code)]
    #[must_use]
    pub const fn as_slice(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.data, self.size) }
    }

    /// Get the allocated memory as a mutable slice
    ///
    /// # Safety
    /// Safe because pointer and size are guaranteed valid by pool allocation
    #[allow(unsafe_code)]
    pub const fn as_mut_slice(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.data, self.size) }
    }
}

/// Statistics about memory pool usage
#[derive(Debug, Clone)]
pub struct PoolStats {
    /// Number of chunks allocated
    pub num_chunks: usize,
    /// Size of each chunk in bytes
    pub chunk_size: usize,
    /// Total capacity across all chunks
    pub total_capacity: usize,
    /// Total memory currently in use
    pub total_used: usize,
    /// Utilization ratio (0.0 to 1.0)
    pub utilization: f32,
}

thread_local! {
    /// Thread-local memory pool for single-threaded access
    static LOCAL_POOL: RefCell<LocalMemoryPool> = RefCell::new(LocalMemoryPool::new(4096));
}

/// Thread-local memory pool without locking overhead
pub struct LocalMemoryPool {
    buffer: Vec<u8>,
    position: usize,
}

impl LocalMemoryPool {
    /// Create a new local memory pool
    #[must_use]
    pub fn new(size: usize) -> Self {
        Self {
            buffer: vec![0u8; size],
            position: 0,
        }
    }

    /// Allocate from the local pool
    pub fn allocate<T>(&mut self) -> Option<&mut T> {
        let size = mem::size_of::<T>();
        let align = mem::align_of::<T>();

        // Align position
        let aligned_pos = (self.position + align - 1) & !(align - 1);

        if aligned_pos + size > self.buffer.len() {
            return None;
        }

        self.position = aligned_pos + size;
        let ptr = (&raw mut self.buffer[aligned_pos]).cast::<T>();
        // SAFETY: ptr is derived from a valid mutable reference to aligned buffer space
        // that we just verified has enough room for T. The lifetime is tied to &mut self.
        #[allow(unsafe_code)]
        Some(unsafe { &mut *ptr })
    }

    /// Reset the pool for reuse
    pub const fn reset(&mut self) {
        self.position = 0;
    }

    /// Allocate from thread-local pool
    pub fn with<F, R>(f: F) -> R
    where
        F: FnOnce(&mut Self) -> R,
    {
        LOCAL_POOL.with(|pool| f(&mut pool.borrow_mut()))
    }
}

// ---------------------------------------------------------------------------
// Lock-free ActivationRecord pool used by the spreading engine
// ---------------------------------------------------------------------------

const DEFAULT_DECAY_RATE: f32 = 0.1;

static NEXT_POOL_ID: AtomicUsize = AtomicUsize::new(1);

thread_local! {
    static THREAD_CACHES: RefCell<HashMap<usize, Vec<Arc<ActivationRecord>>>> =
        RefCell::new(HashMap::new());
}

/// Lock-free pool for recycling `ActivationRecord` instances.
#[derive(Clone)]
pub struct ActivationRecordPool {
    inner: Arc<ActivationRecordPoolInner>,
}

impl ActivationRecordPool {
    /// Create a pool with a default configuration suitable for tests.
    #[must_use]
    pub fn new(initial_records: usize) -> Self {
        Self::with_config(
            initial_records,
            initial_records.max(512),
            64,
            DEFAULT_DECAY_RATE,
        )
    }

    /// Create a pool with explicit limits and decay rate.
    #[must_use]
    pub fn with_config(
        initial_records: usize,
        max_records: usize,
        thread_cache_capacity: usize,
        default_decay_rate: f32,
    ) -> Self {
        let id = NEXT_POOL_ID.fetch_add(1, Ordering::Relaxed);
        let inner = Arc::new(ActivationRecordPoolInner::new(
            id,
            max_records.max(initial_records).max(1),
            thread_cache_capacity.max(1),
            default_decay_rate,
        ));
        inner.prepopulate(initial_records);
        Self { inner }
    }

    /// Acquire an activation record, reusing a pooled instance when available.
    #[must_use]
    pub fn acquire(
        &self,
        node_id: NodeId,
        decay_rate: f32,
        storage_tier: Option<StorageTier>,
    ) -> Arc<ActivationRecord> {
        if let Some(mut record) = self.inner.checkout_from_pool() {
            if let Some(inner) = Arc::get_mut(&mut record) {
                inner.reinitialize(node_id, decay_rate, storage_tier);
            }
            record
        } else {
            self.inner.record_miss();
            let mut record = ActivationRecord::new(node_id, decay_rate);
            if let Some(tier) = storage_tier {
                record.set_storage_tier(tier);
            }
            Arc::new(record)
        }
    }

    /// Acquire a record using the pool's default decay rate.
    #[must_use]
    pub fn acquire_with_defaults(
        &self,
        node_id: NodeId,
        storage_tier: Option<StorageTier>,
    ) -> Arc<ActivationRecord> {
        self.acquire(node_id, self.inner.default_decay_rate, storage_tier)
    }

    /// Return a record to the pool for reuse.
    pub fn release(&self, mut record: Arc<ActivationRecord>) {
        self.inner.mark_return();

        if Arc::strong_count(&record) != 1 {
            self.inner
                .counters
                .release_failures
                .fetch_add(1, Ordering::Relaxed);
            return;
        }

        if !self.inner.should_retain() {
            // Drop record instead of returning to avoid exceeding pool bound.
            return;
        }

        if let Some(inner) = Arc::get_mut(&mut record) {
            inner.prepare_for_pool();
        }

        self.inner
            .counters
            .available
            .fetch_add(1, Ordering::Relaxed);
        self.inner.update_high_water();

        if let Err(record) = self.inner.push_to_thread_cache(record) {
            global_queue_push(&self.inner.global, record);
        }
    }

    /// Snapshot statistics about the pool.
    #[must_use]
    pub fn stats(&self) -> ActivationRecordPoolStats {
        let counters = &self.inner.counters;
        let available = counters.available.load(Ordering::Relaxed);
        let total_checked_out = counters.total_checked_out.load(Ordering::Relaxed);
        let total_returned = counters.total_returned.load(Ordering::Relaxed);
        let local_hits = counters.local_hits.load(Ordering::Relaxed);
        let global_hits = counters.global_hits.load(Ordering::Relaxed);
        let misses = counters.misses.load(Ordering::Relaxed);

        let total_reused = local_hits + global_hits;
        let in_flight = total_checked_out.saturating_sub(total_returned);

        let hit_rate = if total_checked_out > 0 {
            (total_reused as f64 / total_checked_out as f64) as f32
        } else {
            0.0
        };

        let utilization = {
            let total_known = available as u64 + in_flight;
            if total_known > 0 {
                (in_flight as f64 / total_known as f64) as f32
            } else {
                0.0
            }
        };

        ActivationRecordPoolStats {
            available,
            in_flight,
            high_water_mark: counters.high_water_mark.load(Ordering::Relaxed),
            total_created: counters.total_created.load(Ordering::Relaxed),
            total_reused,
            misses,
            hit_rate,
            utilization,
            release_failures: counters.release_failures.load(Ordering::Relaxed),
        }
    }

    /// Return the pool identifier (test-only).
    #[cfg(test)]
    #[must_use]
    pub fn id(&self) -> usize {
        self.inner.id
    }
}

struct ActivationRecordPoolInner {
    id: usize,
    global: GlobalQueue<Arc<ActivationRecord>>,
    counters: PoolCounters,
    max_records: usize,
    thread_cache_capacity: usize,
    default_decay_rate: f32,
}

impl ActivationRecordPoolInner {
    fn new(
        id: usize,
        max_records: usize,
        thread_cache_capacity: usize,
        default_decay_rate: f32,
    ) -> Self {
        Self {
            id,
            global: global_queue_new(),
            counters: PoolCounters::default(),
            max_records,
            thread_cache_capacity,
            default_decay_rate,
        }
    }

    fn prepopulate(&self, count: usize) {
        let to_create = count.min(self.max_records);
        for _ in 0..to_create {
            let record = Arc::new(ActivationRecord::new(
                String::new(),
                self.default_decay_rate,
            ));
            global_queue_push(&self.global, record);
            self.counters.available.fetch_add(1, Ordering::Relaxed);
            self.counters.total_created.fetch_add(1, Ordering::Relaxed);
        }
        self.update_high_water();
    }

    fn checkout_from_pool(&self) -> Option<Arc<ActivationRecord>> {
        self.take_from_thread_cache()
            .inspect(|_| self.after_hit(true))
            .or_else(|| global_queue_pop(&self.global).inspect(|_| self.after_hit(false)))
    }

    fn take_from_thread_cache(&self) -> Option<Arc<ActivationRecord>> {
        THREAD_CACHES.with(|caches| {
            let mut caches = caches.borrow_mut();
            caches.get_mut(&self.id).and_then(Vec::pop)
        })
    }

    fn push_to_thread_cache(
        &self,
        record: Arc<ActivationRecord>,
    ) -> Result<(), Arc<ActivationRecord>> {
        THREAD_CACHES.with(|caches| {
            let mut caches = caches.borrow_mut();
            let cache = caches.entry(self.id).or_default();
            if cache.len() < self.thread_cache_capacity {
                cache.push(record);
                Ok(())
            } else {
                Err(record)
            }
        })
    }

    fn after_hit(&self, local: bool) {
        self.decrement_available();
        if local {
            self.counters.local_hits.fetch_add(1, Ordering::Relaxed);
        } else {
            self.counters.global_hits.fetch_add(1, Ordering::Relaxed);
        }
        self.counters
            .total_checked_out
            .fetch_add(1, Ordering::Relaxed);
    }

    fn record_miss(&self) {
        self.counters
            .total_checked_out
            .fetch_add(1, Ordering::Relaxed);
        self.counters.misses.fetch_add(1, Ordering::Relaxed);
        self.counters.total_created.fetch_add(1, Ordering::Relaxed);
    }

    fn mark_return(&self) {
        self.counters.total_returned.fetch_add(1, Ordering::Relaxed);
    }

    fn should_retain(&self) -> bool {
        let available = self.counters.available.load(Ordering::Relaxed);
        available < self.max_records
    }

    fn update_high_water(&self) {
        let mut current = self.counters.high_water_mark.load(Ordering::Relaxed);
        loop {
            let available = self.counters.available.load(Ordering::Relaxed);
            if available <= current {
                break;
            }
            match self.counters.high_water_mark.compare_exchange(
                current,
                available,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(observed) => current = observed,
            }
        }
    }

    fn decrement_available(&self) {
        let _ =
            self.counters
                .available
                .fetch_update(Ordering::AcqRel, Ordering::Relaxed, |current| {
                    if current == 0 {
                        None
                    } else {
                        Some(current - 1)
                    }
                });
    }
}

#[derive(Default)]
struct PoolCounters {
    available: AtomicUsize,
    high_water_mark: AtomicUsize,
    total_created: AtomicU64,
    total_checked_out: AtomicU64,
    total_returned: AtomicU64,
    local_hits: AtomicU64,
    global_hits: AtomicU64,
    misses: AtomicU64,
    release_failures: AtomicU64,
}

/// Snapshot of activation record pool utilisation.
#[derive(Debug, Clone)]
pub struct ActivationRecordPoolStats {
    /// Records currently available for reuse (global + thread-local caches).
    pub available: usize,
    /// Records currently checked out by workers and not yet returned.
    pub in_flight: u64,
    /// Maximum number of simultaneously available records observed.
    pub high_water_mark: usize,
    /// Total activation records allocated since pool creation.
    pub total_created: u64,
    /// Number of acquisitions served from the pool rather than new allocations.
    pub total_reused: u64,
    /// Number of acquisitions that required allocating a fresh record.
    pub misses: u64,
    /// Ratio of pooled acquisitions to total acquisitions.
    pub hit_rate: f32,
    /// Fraction of records currently in flight relative to tracked pool size.
    pub utilization: f32,
    /// Count of failed release attempts (usually due to outstanding references).
    pub release_failures: u64,
}

#[cfg(test)]
#[allow(clippy::panic)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_pool_allocation() {
        let pool = ActivationMemoryPool::new(1024, 4);

        // Allocate some memory
        let mut alloc1 = pool.allocate(100);
        let slice1 = alloc1.as_mut_slice();
        slice1[0] = 42;
        assert_eq!(slice1.len(), 100);

        // Allocate more
        let alloc2 = pool.allocate(200);
        assert_eq!(alloc2.as_slice().len(), 200);

        // Check stats
        let stats = pool.stats();
        assert_eq!(stats.num_chunks, 1);
        assert_eq!(stats.total_used, 300);
        assert!(stats.utilization > 0.0);
    }

    #[test]
    fn test_local_memory_pool() {
        let mut pool = LocalMemoryPool::new(256);

        // Allocate some values
        let Some(val1) = pool.allocate::<u32>() else {
            panic!("allocation should succeed");
        };
        *val1 = 42;
        assert_eq!(*val1, 42);

        let Some(val2) = pool.allocate::<u64>() else {
            panic!("allocation should succeed");
        };
        *val2 = 100;
        assert_eq!(*val2, 100);

        // Reset and reuse
        pool.reset();
        let Some(val3) = pool.allocate::<u32>() else {
            panic!("allocation should succeed");
        };
        *val3 = 7;
        assert_eq!(*val3, 7);
    }

    #[test]
    fn activation_record_pool_reuses_records() {
        let pool = ActivationRecordPool::with_config(2, 8, 4, DEFAULT_DECAY_RATE);

        let first = pool.acquire("node-a".to_string(), 0.1, Some(StorageTier::Hot));
        assert!(first.get_activation().abs() < f32::EPSILON);
        pool.release(first);

        let stats_after_release = pool.stats();
        assert!(stats_after_release.available >= 1);

        let recycled = pool.acquire("node-b".to_string(), 0.2, Some(StorageTier::Warm));
        assert_eq!(recycled.storage_tier(), Some(StorageTier::Warm));
        assert!(recycled.get_activation().abs() < f32::EPSILON);

        let stats = pool.stats();
        assert!(stats.total_reused >= 1);
        assert!(stats.hit_rate > 0.0);
        // Ensure record returned to pool clears node id when reclaimed.
        assert_eq!(recycled.node_id, "node-b");

        pool.release(recycled);
    }

    #[test]
    fn activation_record_pool_respects_capacity() {
        let pool = ActivationRecordPool::with_config(0, 1, 1, DEFAULT_DECAY_RATE);

        let record1 = pool.acquire("node-a".into(), 0.1, None);
        pool.release(record1);

        // Pool is at capacity, next release should drop instead of retain.
        let record2 = pool.acquire("node-b".into(), 0.1, None);
        pool.release(record2);

        let stats = pool.stats();
        assert!(stats.available <= 1);
    }
}
