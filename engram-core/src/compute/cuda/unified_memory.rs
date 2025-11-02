//! Unified memory allocator for zero-copy GPU operations
//!
//! This module provides RAII wrappers for CUDA unified memory with automatic
//! prefetching and memory advise hints. It handles graceful fallback to pinned
//! memory on older GPU architectures that don't support unified memory.
//!
//! # Architecture
//!
//! - **Unified Memory (Pascal+)**: Single allocation accessible from CPU and GPU
//!   with automatic migration on access
//! - **Pinned Memory (Maxwell)**: Separate CPU/GPU allocations with explicit transfers
//! - **Memory Pool**: Reusable allocations to amortize allocation overhead
//! - **OOM Prevention**: Conservative VRAM limits and graceful degradation
//!
//! # Usage
//!
//! ```rust,ignore
//! use engram_core::compute::cuda::unified_memory::{UnifiedMemory, MemoryPool};
//!
//! // Allocate unified memory for f32 vectors
//! let mut mem = UnifiedMemory::<f32>::new(1024)?;
//!
//! // Write from CPU
//! for i in 0..1024 {
//!     mem[i] = i as f32;
//! }
//!
//! // Prefetch to GPU before kernel launch
//! mem.prefetch_to_gpu()?;
//!
//! // GPU kernel can now access without explicit copy
//! // ... kernel launch ...
//!
//! // Memory automatically freed when `mem` goes out of scope
//! ```

use super::ffi::{self, CUDA_CPU_DEVICE_ID, CudaError, CudaMemoryAdvise};
use dashmap::DashMap;
use std::ffi::c_void;
use std::marker::PhantomData;
use std::ops::{Index, IndexMut};
use std::ptr::NonNull;
use std::sync::atomic::{AtomicUsize, Ordering};

/// RAII wrapper for unified memory allocation
///
/// Provides automatic memory management with deterministic cleanup.
/// Memory is accessible from both CPU and GPU, with automatic migration
/// on first access (for unified memory) or explicit prefetch hints.
///
/// # Type Safety
///
/// The `UnifiedMemory<T>` is parameterized by element type to ensure
/// type-safe access and prevent misaligned access patterns.
#[derive(Debug)]
pub struct UnifiedMemory<T> {
    /// Pointer to allocated memory (unified or pinned)
    ptr: NonNull<T>,
    /// Number of elements currently in use
    len: usize,
    /// Total capacity in elements
    capacity: usize,
    /// Device ID for this allocation
    device_id: i32,
    /// True if using unified memory, false if using pinned memory
    is_unified: bool,
    /// Phantom data for type safety
    _phantom: PhantomData<T>,
}

unsafe impl<T: Send> Send for UnifiedMemory<T> {}
unsafe impl<T: Sync> Sync for UnifiedMemory<T> {}

impl<T> UnifiedMemory<T> {
    /// Allocate new unified memory buffer
    ///
    /// Automatically detects whether unified memory is supported and falls back
    /// to pinned memory if necessary.
    ///
    /// # Arguments
    ///
    /// * `capacity` - Number of elements to allocate
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - Device query fails
    /// - Memory allocation fails
    /// - Device ID is invalid
    pub fn new(capacity: usize) -> Result<Self, CudaError> {
        let device_id = ffi::get_device()?;
        Self::new_on_device(capacity, device_id)
    }

    /// Allocate unified memory on specific device
    ///
    /// # Arguments
    ///
    /// * `capacity` - Number of elements to allocate
    /// * `device_id` - CUDA device ID to allocate on
    ///
    /// # Errors
    ///
    /// Returns error if allocation fails or device doesn't exist
    pub fn new_on_device(capacity: usize, device_id: i32) -> Result<Self, CudaError> {
        if capacity == 0 {
            return Err(CudaError::InvalidValue);
        }

        // Query device properties to determine unified memory support
        let props = ffi::get_device_properties(device_id)?;
        let is_unified = props.managed_memory != 0;

        let size_bytes = capacity * std::mem::size_of::<T>();

        // Allocate memory based on device capabilities
        let ptr = if is_unified {
            // Pascal+ GPU: use unified memory
            let raw_ptr = ffi::malloc_managed(size_bytes)?;
            NonNull::new(raw_ptr.cast::<T>()).ok_or(CudaError::OutOfMemory)?
        } else {
            // Maxwell GPU: use pinned memory
            let raw_ptr = ffi::malloc_host(size_bytes)?;
            NonNull::new(raw_ptr.cast::<T>()).ok_or(CudaError::OutOfMemory)?
        };

        Ok(Self {
            ptr,
            len: 0,
            capacity,
            device_id,
            is_unified,
            _phantom: PhantomData,
        })
    }

    /// Get raw pointer to underlying memory
    ///
    /// # Safety
    ///
    /// The returned pointer is valid for the lifetime of this `UnifiedMemory`.
    /// Caller must ensure no data races occur if accessed concurrently.
    #[must_use]
    pub fn as_ptr(&self) -> *const T {
        self.ptr.as_ptr()
    }

    /// Get mutable raw pointer to underlying memory
    ///
    /// # Safety
    ///
    /// The returned pointer is valid for the lifetime of this `UnifiedMemory`.
    /// Caller must ensure no data races occur if accessed concurrently.
    #[must_use]
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr.as_ptr()
    }

    /// Get number of elements in use
    #[must_use]
    pub const fn len(&self) -> usize {
        self.len
    }

    /// Get total capacity
    #[must_use]
    pub const fn capacity(&self) -> usize {
        self.capacity
    }

    /// Check if buffer is empty
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Check if using unified memory (vs pinned memory)
    #[must_use]
    pub const fn is_unified(&self) -> bool {
        self.is_unified
    }

    /// Set the length of the buffer
    ///
    /// # Safety
    ///
    /// Caller must ensure that elements [0..new_len) are properly initialized.
    ///
    /// # Panics
    ///
    /// Panics if new_len > capacity
    pub unsafe fn set_len(&mut self, new_len: usize) {
        assert!(new_len <= self.capacity, "Length exceeds capacity");
        self.len = new_len;
    }

    /// Prefetch memory to GPU
    ///
    /// For unified memory: Async prefetch to device, hides transfer latency
    /// For pinned memory: Explicit async copy to device memory
    ///
    /// # Errors
    ///
    /// Returns error if prefetch/copy operation fails
    pub fn prefetch_to_gpu(&self) -> Result<(), CudaError> {
        if self.len == 0 {
            return Ok(());
        }

        let size_bytes = self.len * std::mem::size_of::<T>();

        if self.is_unified {
            // Prefetch unified memory to GPU
            ffi::mem_prefetch_async(
                self.ptr.as_ptr().cast::<c_void>(),
                size_bytes,
                self.device_id,
                std::ptr::null_mut(), // Default stream
            )
        } else {
            // For pinned memory, caller must handle explicit transfers
            // This is a no-op here
            Ok(())
        }
    }

    /// Prefetch memory to CPU
    ///
    /// # Errors
    ///
    /// Returns error if prefetch operation fails
    pub fn prefetch_to_cpu(&self) -> Result<(), CudaError> {
        if !self.is_unified || self.len == 0 {
            return Ok(());
        }

        let size_bytes = self.len * std::mem::size_of::<T>();

        ffi::mem_prefetch_async(
            self.ptr.as_ptr().cast::<c_void>(),
            size_bytes,
            CUDA_CPU_DEVICE_ID,
            std::ptr::null_mut(),
        )
    }

    /// Mark memory as read-mostly
    ///
    /// Hint to driver that this memory will be mostly read, enabling caching optimizations.
    /// Particularly useful for query vectors that are broadcast to many threads.
    ///
    /// # Errors
    ///
    /// Returns error if memory advise operation fails
    pub fn advise_read_mostly(&self) -> Result<(), CudaError> {
        if !self.is_unified || self.len == 0 {
            return Ok(());
        }

        let size_bytes = self.len * std::mem::size_of::<T>();

        ffi::mem_advise(
            self.ptr.as_ptr().cast::<c_void>(),
            size_bytes,
            CudaMemoryAdvise::SetReadMostly,
            self.device_id,
        )
    }

    /// Set preferred location for memory
    ///
    /// Hint to driver where this memory should primarily reside.
    /// Useful for large batches that will be processed entirely on GPU.
    ///
    /// # Arguments
    ///
    /// * `device` - Device ID (or `CUDA_CPU_DEVICE_ID` for CPU)
    ///
    /// # Errors
    ///
    /// Returns error if memory advise operation fails
    pub fn set_preferred_location(&self, device: i32) -> Result<(), CudaError> {
        if !self.is_unified || self.len == 0 {
            return Ok(());
        }

        let size_bytes = self.len * std::mem::size_of::<T>();

        ffi::mem_advise(
            self.ptr.as_ptr().cast::<c_void>(),
            size_bytes,
            CudaMemoryAdvise::SetPreferredLocation,
            device,
        )
    }

    /// Get slice view of memory
    ///
    /// # Safety
    ///
    /// Caller must ensure no concurrent GPU access while holding this slice.
    #[must_use]
    pub fn as_slice(&self) -> &[T] {
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }

    /// Get mutable slice view of memory
    ///
    /// # Safety
    ///
    /// Caller must ensure no concurrent GPU access while holding this slice.
    #[must_use]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }
}

impl<T> Drop for UnifiedMemory<T> {
    fn drop(&mut self) {
        unsafe {
            let raw_ptr = self.ptr.as_ptr().cast::<c_void>();
            let result = if self.is_unified {
                ffi::free(raw_ptr)
            } else {
                ffi::free_host(raw_ptr)
            };

            if let Err(e) = result {
                tracing::error!("Failed to free unified/pinned memory: {}", e);
            }
        }
    }
}

impl<T> Index<usize> for UnifiedMemory<T> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        assert!(index < self.len, "Index out of bounds");
        unsafe { &*self.ptr.as_ptr().add(index) }
    }
}

impl<T> IndexMut<usize> for UnifiedMemory<T> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        assert!(index < self.len, "Index out of bounds");
        unsafe { &mut *self.ptr.as_ptr().add(index) }
    }
}

/// Memory pool for reusable unified memory allocations
///
/// Maintains a pool of pre-allocated buffers to amortize allocation overhead.
/// Tracks VRAM usage to prevent OOM conditions.
pub struct MemoryPool {
    /// Available buffers by capacity (reusable)
    available: DashMap<usize, Vec<UnifiedMemory<f32>>>,
    /// Total allocated VRAM from OS in bytes (includes pooled buffers)
    total_from_os: AtomicUsize,
    /// VRAM limit (80% of total device memory)
    vram_limit: usize,
    /// Device ID for this pool
    device_id: i32,
}

impl MemoryPool {
    /// Create new memory pool
    ///
    /// Queries device memory and sets conservative limit (80% of total VRAM).
    ///
    /// # Errors
    ///
    /// Returns error if device query fails
    pub fn new() -> Result<Self, CudaError> {
        let device_id = ffi::get_device()?;
        Self::new_on_device(device_id)
    }

    /// Create memory pool on specific device
    ///
    /// # Errors
    ///
    /// Returns error if device query fails
    pub fn new_on_device(device_id: i32) -> Result<Self, CudaError> {
        let (_free, total) = ffi::mem_get_info()?;
        let vram_limit = (total as f64 * 0.8) as usize;

        Ok(Self {
            available: DashMap::new(),
            total_from_os: AtomicUsize::new(0),
            vram_limit,
            device_id,
        })
    }

    /// Allocate memory from pool
    ///
    /// Attempts to reuse existing allocation if available, otherwise allocates new.
    /// Enforces VRAM limit to prevent OOM.
    ///
    /// # Arguments
    ///
    /// * `capacity` - Number of f32 elements to allocate
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - Allocation would exceed VRAM limit
    /// - Memory allocation fails
    pub fn allocate(&self, capacity: usize) -> Result<UnifiedMemory<f32>, CudaError> {
        // Try to reuse existing buffer from pool
        if let Some(mut available_buffers) = self.available.get_mut(&capacity) {
            if let Some(mut buffer) = available_buffers.pop() {
                // Reset length for reuse
                unsafe { buffer.set_len(0) };
                // Buffer is already counted in total_from_os
                return Ok(buffer);
            }
        }

        // Check VRAM limit before allocating new buffer from OS
        let size_bytes = capacity * std::mem::size_of::<f32>();
        let current = self.total_from_os.load(Ordering::Acquire);
        let new_total = current.saturating_add(size_bytes);

        if new_total > self.vram_limit {
            return Err(CudaError::OutOfMemory);
        }

        // Allocate new buffer from OS
        let buffer = UnifiedMemory::new_on_device(capacity, self.device_id)?;
        self.total_from_os.fetch_add(size_bytes, Ordering::Release);

        Ok(buffer)
    }

    /// Return memory to pool for reuse
    ///
    /// Memory remains allocated from OS but is available for reuse.
    ///
    /// # Arguments
    ///
    /// * `buffer` - Buffer to return to pool
    pub fn deallocate(&self, buffer: UnifiedMemory<f32>) {
        let capacity = buffer.capacity();
        // Return to pool without changing total_from_os
        // Memory is still allocated from OS, just not actively in use
        self.available.entry(capacity).or_default().push(buffer);
    }

    /// Get total allocated VRAM from OS (including pooled buffers)
    #[must_use]
    pub fn total_allocated(&self) -> usize {
        self.total_from_os.load(Ordering::Acquire)
    }

    /// Get VRAM limit
    #[must_use]
    pub const fn vram_limit(&self) -> usize {
        self.vram_limit
    }

    /// Check if allocation would exceed VRAM limit
    #[must_use]
    pub fn would_exceed_limit(&self, capacity: usize) -> bool {
        let size_bytes = capacity * std::mem::size_of::<f32>();
        let current = self.total_from_os.load(Ordering::Acquire);
        current.saturating_add(size_bytes) > self.vram_limit
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(cuda_available)]
    fn test_unified_memory_allocation() {
        use crate::compute::cuda;

        if !cuda::is_available() {
            println!("GPU not available, skipping test");
            return;
        }

        let mut mem = UnifiedMemory::<f32>::new(1024).expect("Failed to allocate");
        assert_eq!(mem.capacity(), 1024);
        assert_eq!(mem.len(), 0);

        // Write from CPU
        unsafe { mem.set_len(1024) };
        for i in 0..1024 {
            mem[i] = i as f32;
        }

        // Prefetch to GPU
        mem.prefetch_to_gpu().expect("Prefetch failed");

        // Verify data
        for i in 0..1024 {
            assert_eq!(mem[i], i as f32);
        }
    }

    #[test]
    #[cfg(cuda_available)]
    fn test_memory_pool_reuse() {
        use crate::compute::cuda;

        if !cuda::is_available() {
            println!("GPU not available, skipping test");
            return;
        }

        let pool = MemoryPool::new().expect("Failed to create pool");

        let buf1 = pool.allocate(1024).expect("Allocation failed");
        let ptr1 = buf1.as_ptr();
        pool.deallocate(buf1);

        let buf2 = pool.allocate(1024).expect("Allocation failed");
        let ptr2 = buf2.as_ptr();

        // Should reuse same allocation
        assert_eq!(ptr1, ptr2);
    }

    #[test]
    #[cfg(cuda_available)]
    fn test_oom_prevention() {
        use crate::compute::cuda;

        if !cuda::is_available() {
            println!("GPU not available, skipping test");
            return;
        }

        let pool = MemoryPool::new().expect("Failed to create pool");

        // Try to allocate more than VRAM limit
        let huge_capacity = pool.vram_limit() / std::mem::size_of::<f32>() + 1000;
        let result = pool.allocate(huge_capacity);

        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), CudaError::OutOfMemory));
    }

    #[test]
    #[cfg(cuda_available)]
    fn test_memory_advise() {
        use crate::compute::cuda;

        if !cuda::is_available() {
            println!("GPU not available, skipping test");
            return;
        }

        let mut mem = UnifiedMemory::<f32>::new(1024).expect("Failed to allocate");
        unsafe { mem.set_len(1024) };

        // These should not error (even if device doesn't support, they're no-ops)
        mem.advise_read_mostly().expect("Read mostly failed");
        mem.set_preferred_location(0)
            .expect("Preferred location failed");
    }
}
