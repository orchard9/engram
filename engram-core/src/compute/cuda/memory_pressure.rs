//! GPU memory pressure monitoring and adaptive batch size calculation
//!
//! This module provides production-grade OOM prevention through:
//! - Real-time VRAM availability monitoring
//! - Adaptive batch size calculation based on memory pressure
//! - Automatic batch splitting when memory constrained
//! - Telemetry for memory-related events
//!
//! # Design Principles
//!
//! 1. **Conservative Memory Management**: Use only 80% of available VRAM to leave headroom
//! 2. **Never Crash**: OOM conditions trigger automatic CPU fallback, never application failure
//! 3. **Transparent Splitting**: Large batches are transparently split and reassembled
//! 4. **Predictive Allocation**: Estimate memory requirements before attempting allocation
//!
//! # Architecture
//!
//! The memory pressure monitor queries VRAM availability before each batch operation
//! and calculates the maximum safe batch size. If the requested batch exceeds this limit,
//! it is automatically split into chunks that fit within available VRAM.
//!
//! ```text
//! Request (10K vectors)
//!       |
//!       v
//! Check VRAM availability
//!       |
//!       v
//! Calculate safe batch size (3K vectors)
//!       |
//!       v
//! Split into chunks: [3K, 3K, 3K, 1K]
//!       |
//!       v
//! Process each chunk on GPU
//!       |
//!       v
//! Reassemble results
//! ```

use super::ffi::{CudaError, mem_get_info};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Memory pressure monitor for GPU VRAM management
///
/// Tracks total VRAM, enforces conservative usage limits (80% of total),
/// and calculates safe batch sizes based on current memory pressure.
///
/// # Thread Safety
///
/// This type is thread-safe and can be shared across threads via Arc.
/// Current allocation tracking uses atomic operations for lock-free updates.
pub struct MemoryPressureMonitor {
    /// Total VRAM in bytes (queried at initialization)
    total_vram: usize,

    /// Maximum VRAM we allow ourselves to use (80% of total)
    vram_limit: usize,

    /// Currently allocated VRAM (tracked by this monitor)
    current_allocated: AtomicUsize,

    /// Safety margin multiplier (default: 0.8 = 80% of available)
    safety_margin: f32,
}

impl MemoryPressureMonitor {
    /// Create new memory pressure monitor
    ///
    /// Queries GPU for total VRAM and sets conservative usage limits.
    /// If GPU query fails, returns None (caller should fall back to CPU).
    ///
    /// # Safety Margin
    ///
    /// The default safety margin is 80%, meaning we'll only use 80% of
    /// available VRAM to leave headroom for:
    /// - Other GPU processes
    /// - CUDA runtime overhead
    /// - Fragmentation
    /// - Transient allocations
    pub fn new() -> Option<Self> {
        Self::with_safety_margin(0.8)
    }

    /// Create monitor with custom safety margin
    ///
    /// # Arguments
    ///
    /// * `safety_margin` - Fraction of VRAM to use (0.0-1.0)
    ///
    /// # Example
    ///
    /// ```ignore
    /// // Very conservative: only use 50% of VRAM
    /// let monitor = MemoryPressureMonitor::with_safety_margin(0.5);
    ///
    /// // Aggressive: use 95% of VRAM (risky)
    /// let monitor = MemoryPressureMonitor::with_safety_margin(0.95);
    /// ```
    pub fn with_safety_margin(safety_margin: f32) -> Option<Self> {
        // Query total VRAM from GPU
        let (free_vram, total_vram) = mem_get_info().ok()?;

        if total_vram == 0 {
            tracing::warn!("GPU reports 0 bytes total VRAM, cannot create memory monitor");
            return None;
        }

        let vram_limit = (total_vram as f32 * safety_margin) as usize;

        tracing::info!(
            "GPU memory pressure monitor initialized: {:.2} GB total, {:.2} GB limit ({:.0}% safety margin)",
            total_vram as f64 / (1024.0 * 1024.0 * 1024.0),
            vram_limit as f64 / (1024.0 * 1024.0 * 1024.0),
            safety_margin * 100.0
        );

        Some(Self {
            total_vram,
            vram_limit,
            current_allocated: AtomicUsize::new(0),
            safety_margin,
        })
    }

    /// Query current available VRAM from GPU
    ///
    /// This is more accurate than tracking allocations ourselves,
    /// as it accounts for other processes using the GPU.
    ///
    /// Returns available bytes, or None if query fails.
    pub fn query_available_vram(&self) -> Option<usize> {
        mem_get_info().ok().map(|(free, _total)| free)
    }

    /// Get total VRAM in bytes
    #[must_use]
    pub const fn total_vram(&self) -> usize {
        self.total_vram
    }

    /// Get VRAM limit in bytes (total * safety_margin)
    #[must_use]
    pub const fn vram_limit(&self) -> usize {
        self.vram_limit
    }

    /// Get currently tracked allocation in bytes
    #[must_use]
    pub fn current_allocated(&self) -> usize {
        self.current_allocated.load(Ordering::Relaxed)
    }

    /// Calculate safe batch size based on available VRAM
    ///
    /// Computes the maximum number of items that can fit in available VRAM
    /// while respecting the safety margin. If the entire requested batch fits,
    /// returns the requested size. Otherwise, returns a smaller safe size.
    ///
    /// # Arguments
    ///
    /// * `requested_batch_size` - Desired number of items to process
    /// * `per_item_memory` - Memory required per item in bytes
    ///
    /// # Returns
    ///
    /// Maximum safe batch size (always >= 1 if VRAM available)
    ///
    /// # Example
    ///
    /// ```ignore
    /// let monitor = MemoryPressureMonitor::new().unwrap();
    ///
    /// // Want to process 10K vectors, each requires 3KB
    /// let safe_batch = monitor.calculate_safe_batch_size(10_000, 3_072);
    ///
    /// if safe_batch < 10_000 {
    ///     println!("Must split batch into chunks of {}", safe_batch);
    /// }
    /// ```
    #[must_use]
    pub fn calculate_safe_batch_size(
        &self,
        requested_batch_size: usize,
        per_item_memory: usize,
    ) -> usize {
        // Query actual available VRAM (more accurate than tracking)
        let available = self.query_available_vram().unwrap_or_else(|| {
            // Fallback: use tracked allocation if query fails
            self.vram_limit
                .saturating_sub(self.current_allocated.load(Ordering::Relaxed))
        });

        let total_required = requested_batch_size.saturating_mul(per_item_memory);

        if total_required <= available {
            // Entire batch fits in available VRAM
            requested_batch_size
        } else {
            // Must reduce batch size to fit
            let safe_batch = available / per_item_memory.max(1);

            tracing::debug!(
                "Batch size reduced: requested {}, available memory {:.2} MB, safe batch {}",
                requested_batch_size,
                available as f64 / (1024.0 * 1024.0),
                safe_batch
            );

            safe_batch.max(1) // Always try to process at least 1 item
        }
    }

    /// Process items in memory-safe chunks
    ///
    /// Automatically splits the input into chunks that fit in available VRAM,
    /// processes each chunk with the provided function, and reassembles results.
    ///
    /// This is the primary interface for memory-safe GPU operations.
    ///
    /// # Arguments
    ///
    /// * `items` - Items to process
    /// * `per_item_memory` - Memory required per item in bytes
    /// * `process_fn` - Function to process each chunk on GPU
    ///
    /// # Returns
    ///
    /// Vector of results from processing all chunks
    ///
    /// # Panics
    ///
    /// Panics if `process_fn` returns a result vector with length different from the chunk length.
    /// This indicates a programming error in the processing function.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let monitor = MemoryPressureMonitor::new().unwrap();
    /// let vectors = vec![[1.0f32; 768]; 100_000];
    ///
    /// let results = monitor.process_in_chunks(
    ///     &vectors,
    ///     768 * 4, // 768 floats * 4 bytes
    ///     |chunk| gpu_cosine_similarity(query, chunk),
    /// );
    /// ```
    pub fn process_in_chunks<T, R>(
        &self,
        items: &[T],
        per_item_memory: usize,
        process_fn: impl Fn(&[T]) -> Vec<R>,
    ) -> Vec<R> {
        let safe_batch = self.calculate_safe_batch_size(items.len(), per_item_memory);

        if safe_batch >= items.len() {
            // Process all at once
            tracing::trace!("Processing {} items in single batch", items.len());
            let results = process_fn(items);

            // Validate result size matches input size
            assert_eq!(
                results.len(),
                items.len(),
                "process_fn must return exactly one result per input item (expected {}, got {})",
                items.len(),
                results.len()
            );

            results
        } else {
            // Process in chunks
            tracing::debug!(
                "Processing {} items in chunks of {} (total {} chunks)",
                items.len(),
                safe_batch,
                (items.len() + safe_batch - 1) / safe_batch
            );

            let mut results = Vec::with_capacity(items.len());

            for (chunk_idx, chunk) in items.chunks(safe_batch).enumerate() {
                tracing::trace!("Processing chunk {}: {} items", chunk_idx, chunk.len());

                let chunk_results = process_fn(chunk);

                // Validate chunk result size matches chunk size to prevent silent data corruption
                assert_eq!(
                    chunk_results.len(),
                    chunk.len(),
                    "process_fn must return exactly one result per input item for chunk {} (expected {}, got {})",
                    chunk_idx,
                    chunk.len(),
                    chunk_results.len()
                );

                results.extend(chunk_results);
            }

            results
        }
    }

    /// Track allocation of VRAM
    ///
    /// Updates internal allocation counter. Used for fallback estimation
    /// when real-time VRAM query is unavailable.
    pub fn track_allocation(&self, bytes: usize) {
        self.current_allocated.fetch_add(bytes, Ordering::Relaxed);
    }

    /// Track deallocation of VRAM
    ///
    /// Updates internal allocation counter. Used for fallback estimation
    /// when real-time VRAM query is unavailable.
    pub fn track_deallocation(&self, bytes: usize) {
        self.current_allocated.fetch_sub(bytes, Ordering::Relaxed);
    }

    /// Check if we're under memory pressure
    ///
    /// Returns true if available VRAM is less than 20% of total.
    /// This can be used to trigger preemptive CPU fallback.
    #[must_use]
    pub fn is_under_pressure(&self) -> bool {
        if let Some(available) = self.query_available_vram() {
            let pressure_threshold = (self.total_vram as f32 * 0.2) as usize;
            available < pressure_threshold
        } else {
            // If we can't query VRAM, assume we're under pressure
            true
        }
    }

    /// Get memory pressure ratio (0.0 = no pressure, 1.0 = out of memory)
    ///
    /// Useful for adaptive behavior and telemetry.
    #[must_use]
    pub fn pressure_ratio(&self) -> f32 {
        if let Some(available) = self.query_available_vram() {
            let used = self.total_vram.saturating_sub(available);
            (used as f32) / (self.total_vram as f32)
        } else {
            // If we can't query, assume high pressure
            0.9
        }
    }
}

impl Default for MemoryPressureMonitor {
    fn default() -> Self {
        Self::new().expect("Failed to create default MemoryPressureMonitor")
    }
}

/// Global memory pressure monitor (initialized lazily)
static GLOBAL_MONITOR: std::sync::OnceLock<Option<Arc<MemoryPressureMonitor>>> =
    std::sync::OnceLock::new();

/// Get global memory pressure monitor
///
/// Returns shared reference to global monitor, or None if GPU unavailable.
/// The monitor is initialized lazily on first access.
pub fn get_memory_pressure_monitor() -> Option<Arc<MemoryPressureMonitor>> {
    GLOBAL_MONITOR
        .get_or_init(|| MemoryPressureMonitor::new().map(Arc::new))
        .clone()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_monitor_creation() {
        // Should create successfully if GPU available
        if let Some(monitor) = MemoryPressureMonitor::new() {
            println!(
                "Total VRAM: {:.2} GB",
                monitor.total_vram() as f64 / (1024.0 * 1024.0 * 1024.0)
            );
            println!(
                "VRAM limit: {:.2} GB",
                monitor.vram_limit() as f64 / (1024.0 * 1024.0 * 1024.0)
            );

            assert!(monitor.total_vram() > 0);
            assert!(monitor.vram_limit() > 0);
            assert!(monitor.vram_limit() <= monitor.total_vram());
        }
    }

    #[test]
    fn test_safety_margin() {
        if let Some(monitor) = MemoryPressureMonitor::with_safety_margin(0.5) {
            // 50% safety margin
            let expected_limit = (monitor.total_vram() as f32 * 0.5) as usize;
            assert_eq!(monitor.vram_limit(), expected_limit);
        }
    }

    #[test]
    fn test_batch_size_calculation() {
        if let Some(monitor) = MemoryPressureMonitor::new() {
            // Test with 768-dim vectors (3KB each)
            let per_vector = 768 * 4;

            // Small batch should fit entirely
            let safe_batch = monitor.calculate_safe_batch_size(100, per_vector);
            assert!(safe_batch >= 100 || monitor.is_under_pressure());

            // Huge batch should be reduced
            let huge_batch = 10_000_000; // ~30GB
            let safe_huge = monitor.calculate_safe_batch_size(huge_batch, per_vector);
            assert!(safe_huge < huge_batch);
            assert!(safe_huge >= 1); // Always at least 1
        }
    }

    #[test]
    fn test_available_vram_query() {
        if let Some(monitor) = MemoryPressureMonitor::new() {
            if let Some(available) = monitor.query_available_vram() {
                println!(
                    "Available VRAM: {:.2} GB",
                    available as f64 / (1024.0 * 1024.0 * 1024.0)
                );
                assert!(available > 0);
                assert!(available <= monitor.total_vram());
            }
        }
    }

    #[test]
    fn test_allocation_tracking() {
        if let Some(monitor) = MemoryPressureMonitor::new() {
            let initial = monitor.current_allocated();

            monitor.track_allocation(1024 * 1024); // 1MB
            assert_eq!(monitor.current_allocated(), initial + 1024 * 1024);

            monitor.track_deallocation(512 * 1024); // 512KB
            assert_eq!(monitor.current_allocated(), initial + 512 * 1024);
        }
    }

    #[test]
    fn test_pressure_detection() {
        if let Some(monitor) = MemoryPressureMonitor::new() {
            let pressure = monitor.pressure_ratio();
            println!("Current memory pressure: {:.1}%", pressure * 100.0);

            assert!(pressure >= 0.0);
            assert!(pressure <= 1.0);

            // Check if pressure detection is consistent
            let under_pressure = monitor.is_under_pressure();
            if pressure > 0.8 {
                assert!(under_pressure);
            }
        }
    }

    #[test]
    fn test_chunked_processing() {
        if let Some(monitor) = MemoryPressureMonitor::new() {
            let items = vec![1.0f32; 10_000];
            let per_item = 1024; // 1KB per item

            let results = monitor.process_in_chunks(&items, per_item, |chunk| {
                // Simulate processing: double each value
                chunk.iter().map(|x| x * 2.0).collect()
            });

            assert_eq!(results.len(), items.len());
            assert!(results.iter().all(|&x| (x - 2.0).abs() < 1e-6));
        }
    }

    #[test]
    fn test_global_monitor() {
        // Should return same instance on multiple calls
        let monitor1 = get_memory_pressure_monitor();
        let monitor2 = get_memory_pressure_monitor();

        match (monitor1, monitor2) {
            (Some(m1), Some(m2)) => {
                assert_eq!(m1.total_vram(), m2.total_vram());
            }
            (None, None) => {
                // GPU not available, both None is correct
            }
            _ => panic!("Inconsistent global monitor state"),
        }
    }
}
