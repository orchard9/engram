//! Cache-optimized batch buffer structures

use crate::batch::{BatchError, BatchOperation, BatchResult};
use std::sync::atomic::{AtomicUsize, Ordering};

/// Cache-aligned batch buffer for SIMD operations
#[repr(align(64))]
pub struct BatchBuffer {
    /// Batch operations to process
    operations: Vec<BatchOperation>,
    /// Pre-allocated result buffer
    results: Vec<BatchResult>,
    /// SIMD-aligned embeddings for vectorized similarity
    embedding_buffer: Vec<[f32; 768]>,
    /// Current batch size
    size: AtomicUsize,
    /// Memory usage tracking
    memory_usage: AtomicUsize,
    /// Maximum capacity
    capacity: usize,
}

impl BatchBuffer {
    /// Create a new batch buffer with specified capacity
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        Self {
            operations: Vec::with_capacity(capacity),
            results: Vec::with_capacity(capacity),
            embedding_buffer: Vec::with_capacity(capacity),
            size: AtomicUsize::new(0),
            memory_usage: AtomicUsize::new(0),
            capacity,
        }
    }

    /// Push an operation to the buffer
    ///
    /// # Errors
    ///
    /// Returns `BatchError::CapacityExceeded` if buffer is at capacity
    pub fn push(&mut self, operation: BatchOperation) -> Result<(), BatchError> {
        if self.size.load(Ordering::Relaxed) >= self.capacity {
            return Err(BatchError::CapacityExceeded);
        }

        // Track memory usage
        let op_size = std::mem::size_of_val(&operation);
        self.memory_usage.fetch_add(op_size, Ordering::Relaxed);

        // Extract embedding if present for SIMD operations
        if let BatchOperation::SimilaritySearch { embedding, .. } = &operation {
            self.embedding_buffer.push(*embedding);
        }

        self.operations.push(operation);
        self.size.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    /// Get the current number of operations in the buffer
    pub fn len(&self) -> usize {
        self.size.load(Ordering::Relaxed)
    }

    /// Check if buffer is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get current memory usage in bytes
    pub fn memory_usage(&self) -> usize {
        self.memory_usage.load(Ordering::Relaxed)
    }

    /// Check if memory limit is exceeded
    ///
    /// # Errors
    ///
    /// Returns `BatchError::MemoryLimitExceeded` if memory usage exceeds limit
    pub fn check_memory_limit(&self, limit_bytes: usize) -> Result<(), BatchError> {
        let current = self.memory_usage();
        if current > limit_bytes {
            Err(BatchError::MemoryLimitExceeded {
                current_mb: current / (1024 * 1024),
                limit_mb: limit_bytes / (1024 * 1024),
            })
        } else {
            Ok(())
        }
    }

    /// Clear the buffer
    pub fn clear(&mut self) {
        self.operations.clear();
        self.results.clear();
        self.embedding_buffer.clear();
        self.size.store(0, Ordering::Relaxed);
        self.memory_usage.store(0, Ordering::Relaxed);
    }

    /// Take all operations from the buffer
    pub fn take_operations(&mut self) -> Vec<BatchOperation> {
        self.size.store(0, Ordering::Relaxed);
        self.memory_usage.store(0, Ordering::Relaxed);
        std::mem::take(&mut self.operations)
    }

    /// Get a reference to the embedding buffer for SIMD operations
    pub fn embeddings(&self) -> &[[f32; 768]] {
        &self.embedding_buffer
    }

    /// Reserve additional capacity
    pub fn reserve(&mut self, additional: usize) {
        self.operations.reserve(additional);
        self.results.reserve(additional);
        self.embedding_buffer.reserve(additional);
    }
}

/// Aligned buffer specifically for SIMD vector operations
#[repr(align(64))]
pub struct AlignedVectorBuffer {
    /// Aligned vector data
    data: Vec<[f32; 768]>,
    /// Current count
    count: AtomicUsize,
    /// Maximum capacity
    capacity: usize,
}

impl AlignedVectorBuffer {
    /// Create new aligned buffer
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        Self {
            data: Vec::with_capacity(capacity),
            count: AtomicUsize::new(0),
            capacity,
        }
    }

    /// Push a vector to the buffer
    ///
    /// # Errors
    ///
    /// Returns `BatchError::CapacityExceeded` if buffer is at capacity
    pub fn push(&mut self, vector: &[f32; 768]) -> Result<usize, BatchError> {
        let index = self.count.load(Ordering::Relaxed);
        if index >= self.capacity {
            return Err(BatchError::CapacityExceeded);
        }

        self.data.push(*vector);
        self.count.fetch_add(1, Ordering::Relaxed);
        Ok(index)
    }

    /// Get vectors as a slice
    pub fn as_slice(&self) -> &[[f32; 768]] {
        &self.data[..self.count.load(Ordering::Relaxed)]
    }

    /// Get the number of vectors
    pub fn len(&self) -> usize {
        self.count.load(Ordering::Relaxed)
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Clear the buffer
    pub fn clear(&mut self) {
        self.data.clear();
        self.count.store(0, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::batch::BatchOperation;
    use crate::{Confidence, EpisodeBuilder};
    use chrono::Utc;

    #[test]
    fn test_batch_buffer_capacity() {
        let mut buffer = BatchBuffer::new(10);

        // Fill buffer to capacity
        for i in 0..10 {
            let episode = EpisodeBuilder::new()
                .id(format!("ep{}", i))
                .when(Utc::now())
                .what("test".to_string())
                .embedding([0.1; 768])
                .confidence(Confidence::HIGH)
                .build();

            let op = BatchOperation::Store(episode);
            assert!(buffer.push(op).is_ok());
        }

        assert_eq!(buffer.len(), 10);

        // Try to exceed capacity
        let episode = EpisodeBuilder::new()
            .id("overflow".to_string())
            .when(Utc::now())
            .what("test".to_string())
            .embedding([0.1; 768])
            .confidence(Confidence::HIGH)
            .build();

        let op = BatchOperation::Store(episode);
        assert!(matches!(buffer.push(op), Err(BatchError::CapacityExceeded)));
    }

    #[test]
    fn test_memory_tracking() {
        let mut buffer = BatchBuffer::new(100);

        let episode = EpisodeBuilder::new()
            .id("test".to_string())
            .when(Utc::now())
            .what("test memory".to_string())
            .embedding([0.1; 768])
            .confidence(Confidence::HIGH)
            .build();

        let op = BatchOperation::Store(episode);
        buffer.push(op).unwrap();

        assert!(buffer.memory_usage() > 0);

        // Test memory limit checking
        let tiny_limit = 1; // 1 byte limit
        assert!(buffer.check_memory_limit(tiny_limit).is_err());

        let large_limit = 1024 * 1024; // 1MB limit
        assert!(buffer.check_memory_limit(large_limit).is_ok());
    }

    #[test]
    fn test_aligned_vector_buffer() {
        let mut buffer = AlignedVectorBuffer::new(5);

        // Add vectors
        for i in 0..5 {
            let mut vector = [0.0; 768];
            vector[0] = i as f32;
            assert!(buffer.push(&vector).is_ok());
        }

        assert_eq!(buffer.len(), 5);
        assert!(!buffer.is_empty());

        // Verify data
        let slice = buffer.as_slice();
        assert_eq!(slice.len(), 5);
        assert_eq!(slice[0][0], 0.0);
        assert_eq!(slice[4][0], 4.0);

        // Test capacity exceeded
        let vector = [0.0; 768];
        assert!(matches!(
            buffer.push(&vector),
            Err(BatchError::CapacityExceeded)
        ));
    }
}
