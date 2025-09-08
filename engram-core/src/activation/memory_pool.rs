//! Memory pool for efficient allocation during activation spreading
//!
//! This module provides arena-based allocation to reduce allocation overhead
//! during graph traversal and activation spreading operations.

use parking_lot::Mutex;
use std::cell::RefCell;
use std::mem;
use std::sync::Arc;

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

    fn reset(&mut self) {
        self.used = 0;
        // Optionally clear memory for security
        // self.data.fill(0);
    }

    fn has_space(&self, size: usize) -> bool {
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
    pub fn new(chunk_size: usize, max_chunks: usize) -> Self {
        let initial_chunks = vec![MemoryChunk::new(chunk_size)];
        Self {
            chunks: Arc::new(Mutex::new(initial_chunks)),
            chunk_size,
            max_chunks,
        }
    }

    /// Allocate memory from the pool
    pub fn allocate(&self, size: usize) -> PooledAllocation {
        let mut chunks = self.chunks.lock();
        
        // Try to find a chunk with space
        for chunk in chunks.iter_mut() {
            if let Some(memory) = chunk.allocate(size) {
                return PooledAllocation {
                    data: memory.as_mut_ptr(),
                    size,
                    pool: None, // Would need unsafe to store reference
                };
            }
        }
        
        // Need a new chunk
        if chunks.len() < self.max_chunks {
            let mut new_chunk = MemoryChunk::new(self.chunk_size.max(size));
            let memory = new_chunk.allocate(size).unwrap();
            let ptr = memory.as_mut_ptr();
            chunks.push(new_chunk);
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
    pub fn stats(&self) -> PoolStats {
        let chunks = self.chunks.lock();
        let total_capacity = chunks.len() * self.chunk_size;
        let total_used: usize = chunks.iter().map(|c| c.used).sum();
        
        PoolStats {
            num_chunks: chunks.len(),
            chunk_size: self.chunk_size,
            total_capacity,
            total_used,
            utilization: total_used as f32 / total_capacity as f32,
        }
    }
}

/// An allocation from the memory pool
pub struct PooledAllocation {
    data: *mut u8,
    size: usize,
    pool: Option<Arc<ActivationMemoryPool>>,
}

impl PooledAllocation {
    /// Get the allocated memory as a slice
    pub fn as_slice(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.data, self.size) }
    }

    /// Get the allocated memory as a mutable slice
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.data, self.size) }
    }
}

/// Statistics about memory pool usage
#[derive(Debug, Clone)]
pub struct PoolStats {
    pub num_chunks: usize,
    pub chunk_size: usize,
    pub total_capacity: usize,
    pub total_used: usize,
    pub utilization: f32,
}

/// Thread-local memory pool for single-threaded access
thread_local! {
    static LOCAL_POOL: RefCell<LocalMemoryPool> = RefCell::new(LocalMemoryPool::new(4096));
}

/// Thread-local memory pool without locking overhead
pub struct LocalMemoryPool {
    buffer: Vec<u8>,
    position: usize,
}

impl LocalMemoryPool {
    /// Create a new local memory pool
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
        let ptr = &mut self.buffer[aligned_pos] as *mut u8 as *mut T;
        Some(unsafe { &mut *ptr })
    }

    /// Reset the pool for reuse
    pub fn reset(&mut self) {
        self.position = 0;
    }

    /// Allocate from thread-local pool
    pub fn with<T, F, R>(f: F) -> R
    where
        F: FnOnce(&mut LocalMemoryPool) -> R,
    {
        LOCAL_POOL.with(|pool| f(&mut pool.borrow_mut()))
    }
}

#[cfg(test)]
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
    }

    #[test]
    fn test_local_memory_pool() {
        let mut pool = LocalMemoryPool::new(256);
        
        // Allocate some values
        let val1: &mut u32 = pool.allocate().unwrap();
        *val1 = 42;
        
        let val2: &mut u64 = pool.allocate().unwrap();
        *val2 = 100;
        
        assert_eq!(*val1, 42);
        assert_eq!(*val2, 100);
        
        // Reset and reuse
        pool.reset();
        let val3: &mut u32 = pool.allocate().unwrap();
        *val3 = 7;
        assert_eq!(*val3, 7);
    }
}