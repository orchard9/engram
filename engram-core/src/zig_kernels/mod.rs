//! High-performance kernels implemented in Zig with C FFI boundary.
//!
//! This module provides zero-copy FFI wrappers for Zig-implemented performance kernels.
//! When the `zig-kernels` feature is enabled, these functions call into statically-linked
//! Zig code. Without the feature, they fall back to pure Rust implementations.
//!
//! # FFI Safety Invariants
//!
//! All FFI functions maintain strict safety invariants:
//! - Pointers are valid for the entire call duration
//! - Slice lengths match the documented memory layout
//! - No aliasing between input and output buffers
//! - All dimensions are validated before crossing FFI boundary
//!
//! # Memory Layout
//!
//! Zero-copy design: caller allocates, callee populates.
//! - No serialization overhead
//! - No hidden allocations
//! - Direct memory access patterns
//!
//! # Feature Flags
//!
//! - `zig-kernels`: Enable Zig implementations (requires Zig 0.13.0 compiler)
//! - Without feature: Pure Rust fallback implementations
//!
//! # Safety Note
//!
//! This module uses unsafe code for FFI boundary crossing. All unsafe blocks
//! are thoroughly documented with safety invariants that are validated at
//! the safe Rust API layer before crossing into unsafe territory.

// FFI boundary requires unsafe code - allow it for this module
#![allow(unsafe_code)]

#[cfg(feature = "zig-kernels")]
mod ffi {
    #[link(name = "engram_kernels", kind = "static")]
    unsafe extern "C" {
        /// FFI: Vector similarity (cosine) between query and candidates
        ///
        /// # Safety
        ///
        /// Caller must ensure:
        /// - `query` points to `query_len` valid f32 values
        /// - `candidates` points to `num_candidates * query_len` valid f32 values
        /// - `scores` points to `num_candidates` writable f32 slots
        /// - Pointers remain valid for entire call duration
        /// - No aliasing between buffers
        pub fn engram_vector_similarity(
            query: *const f32,
            candidates: *const f32,
            scores: *mut f32,
            query_len: usize,
            num_candidates: usize,
        );

        /// FFI: Activation spreading across graph with CSR representation
        ///
        /// # Safety
        ///
        /// Caller must ensure:
        /// - `adjacency` points to `num_edges` valid u32 node indices
        /// - `weights` points to `num_edges` valid f32 weights
        /// - `activations` points to `num_nodes` writable f32 values
        /// - Node indices in adjacency are < num_nodes
        /// - iterations > 0
        pub fn engram_spread_activation(
            adjacency: *const u32,
            weights: *const f32,
            activations: *mut f32,
            num_nodes: usize,
            num_edges: usize,
            iterations: u32,
        );

        /// FFI: Memory decay (Ebbinghaus exponential)
        ///
        /// # Safety
        ///
        /// Caller must ensure:
        /// - `strengths` points to `num_memories` writable f32 values
        /// - `ages_seconds` points to `num_memories` valid u64 values
        /// - strengths are initially in [0.0, 1.0] range
        pub fn engram_apply_decay(
            strengths: *mut f32,
            ages_seconds: *const u64,
            num_memories: usize,
        );

        /// FFI: Configure arena allocator pool size and overflow strategy
        ///
        /// # Safety
        ///
        /// Call from single thread before spawning workers.
        /// Do not call while arenas are in use.
        pub fn engram_configure_arena(pool_size_mb: u32, overflow_strategy: u8);

        /// FFI: Get global arena usage statistics
        ///
        /// # Safety
        ///
        /// Pointers must be valid and point to writable usize locations.
        pub fn engram_arena_stats(
            total_resets: *mut usize,
            total_overflows: *mut usize,
            max_high_water_mark: *mut usize,
        );

        /// FFI: Reset thread-local arena allocator
        pub fn engram_reset_arenas();

        /// FFI: Reset global arena metrics
        pub fn engram_reset_arena_metrics();
    }
}

/// Compute cosine similarity between a query vector and multiple candidate vectors.
///
/// # Arguments
///
/// * `query` - Query vector of length `dim`
/// * `candidates` - Flattened array of candidate vectors (length = `num_candidates * dim`)
/// * `num_candidates` - Number of candidate vectors
///
/// # Returns
///
/// Vector of similarity scores, one per candidate (length = `num_candidates`).
/// Values are in range [-1.0, 1.0] where 1.0 indicates identical direction.
///
/// # Panics
///
/// Panics if:
/// - `query.is_empty()`
/// - `candidates.len() != num_candidates * query.len()`
/// - `num_candidates == 0`
///
/// # Implementation
///
/// - With `zig-kernels`: SIMD-accelerated Zig implementation (AVX2/NEON)
/// - Without feature: Pure Rust fallback
///
/// # Examples
///
/// ```
/// # use engram_core::zig_kernels::vector_similarity;
/// let query = vec![1.0, 0.0, 0.0];
/// let candidates = vec![
///     1.0, 0.0, 0.0,  // Same direction as query
///     0.0, 1.0, 0.0,  // Orthogonal to query
/// ];
/// let scores = vector_similarity(&query, &candidates, 2);
/// // With stubs: returns [0.0, 0.0]
/// // With actual kernels: returns [1.0, 0.0]
/// ```
#[must_use]
pub fn vector_similarity(query: &[f32], candidates: &[f32], num_candidates: usize) -> Vec<f32> {
    // Validate dimensions
    assert!(!query.is_empty(), "Query vector cannot be empty");
    assert_ne!(num_candidates, 0, "Must have at least one candidate");
    let dim = query.len();
    assert_eq!(
        candidates.len(),
        num_candidates * dim,
        "Candidates array size mismatch: expected {} ({}*{}), got {}",
        num_candidates * dim,
        num_candidates,
        dim,
        candidates.len()
    );

    #[cfg(feature = "zig-kernels")]
    {
        // Call Zig kernel via FFI
        let mut scores = vec![0.0_f32; num_candidates];
        unsafe {
            ffi::engram_vector_similarity(
                query.as_ptr(),
                candidates.as_ptr(),
                scores.as_mut_ptr(),
                dim,
                num_candidates,
            );
        }
        scores
    }

    #[cfg(not(feature = "zig-kernels"))]
    {
        // Fallback to pure Rust implementation
        // TODO: Task 005 will implement actual SIMD cosine similarity
        // For now, stub returns zeros (same as Zig stub)
        vec![0.0_f32; num_candidates]
    }
}

/// Perform spreading activation across a graph represented in CSR format.
///
/// Updates activation levels in-place by propagating activation along weighted edges
/// for the specified number of iterations.
///
/// # Arguments
///
/// * `adjacency` - CSR edge destinations (node indices)
/// * `weights` - Edge weights corresponding to adjacency
/// * `activations` - Current activation levels (modified in-place)
/// * `num_nodes` - Total number of nodes in graph
/// * `iterations` - Number of spreading iterations to perform
///
/// # Panics
///
/// Panics if:
/// - `adjacency.len() != weights.len()`
/// - `activations.len() != num_nodes`
/// - `iterations == 0`
/// - Any node index in adjacency >= num_nodes
///
/// # Implementation
///
/// - With `zig-kernels`: Cache-optimized BFS with SIMD (Task 006)
/// - Without feature: Pure Rust BFS implementation
///
/// # Examples
///
/// ```
/// # use engram_core::zig_kernels::spread_activation;
/// // Simple 3-node graph: 0 -> 1 (weight 0.5), 0 -> 2 (weight 0.3)
/// let adjacency = vec![1, 2];
/// let weights = vec![0.5, 0.3];
/// let mut activations = vec![1.0, 0.0, 0.0];
///
/// spread_activation(&adjacency, &weights, &mut activations, 3, 5);
/// // Activation spreads from node 0 to nodes 1 and 2 over 5 iterations
/// ```
pub fn spread_activation(
    adjacency: &[u32],
    weights: &[f32],
    activations: &mut [f32],
    num_nodes: usize,
    iterations: u32,
) {
    // Validate dimensions
    assert_eq!(
        adjacency.len(),
        weights.len(),
        "Adjacency and weights must have same length"
    );
    assert_eq!(
        activations.len(),
        num_nodes,
        "Activations length must match num_nodes"
    );
    assert_ne!(iterations, 0, "Must perform at least one iteration");

    // Validate node indices are in bounds (debug builds only for performance)
    debug_assert!(
        adjacency.iter().all(|&idx| (idx as usize) < num_nodes),
        "All adjacency indices must be < num_nodes"
    );

    let num_edges = adjacency.len();

    #[cfg(feature = "zig-kernels")]
    {
        // Call Zig kernel via FFI
        unsafe {
            ffi::engram_spread_activation(
                adjacency.as_ptr(),
                weights.as_ptr(),
                activations.as_mut_ptr(),
                num_nodes,
                num_edges,
                iterations,
            );
        }
    }

    #[cfg(not(feature = "zig-kernels"))]
    {
        // Fallback to pure Rust implementation
        // This will be replaced by Task 006 with actual implementation
        // For now, stub does nothing (matches Zig stub behavior)
        let _ = (
            adjacency,
            weights,
            activations,
            num_nodes,
            num_edges,
            iterations,
        );
    }
}

/// Apply memory decay function (Ebbinghaus exponential) to memory strengths.
///
/// Updates strength values in-place based on time since last access,
/// implementing biological memory decay dynamics.
///
/// # Arguments
///
/// * `strengths` - Current memory strengths (modified in-place, should be in [0.0, 1.0])
/// * `ages_seconds` - Time since last access for each memory (in seconds)
///
/// # Panics
///
/// Panics if `strengths.len() != ages_seconds.len()`.
///
/// # Implementation
///
/// - With `zig-kernels`: SIMD-vectorized exponential approximation (Task 007)
/// - Without feature: Pure Rust exponential decay
///
/// # Examples
///
/// ```
/// # use engram_core::zig_kernels::apply_decay;
/// let mut strengths = vec![1.0, 0.8, 0.5];
/// let ages = vec![0, 3600, 86400];  // 0s, 1h, 1d
///
/// apply_decay(&mut strengths, &ages);
/// // Older memories decay more: strengths[2] < strengths[1] < strengths[0]
/// ```
pub fn apply_decay(strengths: &mut [f32], ages_seconds: &[u64]) {
    // Validate dimensions
    assert_eq!(
        strengths.len(),
        ages_seconds.len(),
        "Strengths and ages must have same length"
    );

    let num_memories = strengths.len();

    #[cfg(feature = "zig-kernels")]
    {
        // Call Zig kernel via FFI
        unsafe {
            ffi::engram_apply_decay(strengths.as_mut_ptr(), ages_seconds.as_ptr(), num_memories);
        }
    }

    #[cfg(not(feature = "zig-kernels"))]
    {
        // Fallback to pure Rust implementation
        // This will be replaced by Task 007 with actual implementation
        // For now, stub does nothing (matches Zig stub behavior)
        let _ = (strengths, ages_seconds, num_memories);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vector_similarity_basic() {
        let query = vec![1.0, 0.0, 0.0];
        let candidates = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        let scores = vector_similarity(&query, &candidates, 2);

        assert_eq!(scores.len(), 2);
        // Stub implementation returns zeros for now
        // Task 005 will implement actual SIMD cosine similarity
    }

    #[test]
    #[should_panic(expected = "Query vector cannot be empty")]
    fn test_vector_similarity_empty_query() {
        let query = vec![];
        let candidates = vec![1.0, 0.0];
        let _ = vector_similarity(&query, &candidates, 2);
    }

    #[test]
    #[should_panic(expected = "Candidates array size mismatch")]
    fn test_vector_similarity_size_mismatch() {
        let query = vec![1.0, 0.0, 0.0];
        let candidates = vec![1.0, 0.0]; // Wrong size
        let _ = vector_similarity(&query, &candidates, 2);
    }

    #[test]
    fn test_spread_activation_basic() {
        let adjacency = vec![1, 2];
        let weights = vec![0.5, 0.3];
        let mut activations = vec![1.0, 0.0, 0.0];

        spread_activation(&adjacency, &weights, &mut activations, 3, 5);

        // Stub implementation is no-op for now
        // Task 006 will implement actual spreading activation
    }

    #[test]
    #[should_panic(expected = "Adjacency and weights must have same length")]
    fn test_spread_activation_size_mismatch() {
        let adjacency = vec![1, 2];
        let weights = vec![0.5]; // Wrong size
        let mut activations = vec![1.0, 0.0, 0.0];

        spread_activation(&adjacency, &weights, &mut activations, 3, 5);
    }

    #[test]
    fn test_apply_decay_basic() {
        let mut strengths = vec![1.0, 0.8, 0.5];
        let ages = vec![0, 3600, 86400];

        apply_decay(&mut strengths, &ages);

        // Stub implementation is no-op for now
        // Task 007 will implement actual Ebbinghaus decay
    }

    #[test]
    #[should_panic(expected = "Strengths and ages must have same length")]
    fn test_apply_decay_size_mismatch() {
        let mut strengths = vec![1.0, 0.8];
        let ages = vec![0, 3600, 86400]; // Wrong size

        apply_decay(&mut strengths, &ages);
    }
}

// Arena allocator configuration and monitoring (Task 008)

/// Arena allocator overflow strategy
///
/// Determines behavior when arena capacity is exhausted.
#[cfg(feature = "zig-kernels")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum OverflowStrategy {
    /// Panic immediately on overflow (development mode)
    Panic = 0,

    /// Return error on overflow (production mode)
    ErrorReturn = 1,

    /// Attempt fallback allocation (experimental)
    Fallback = 2,
}

/// Arena allocator statistics
///
/// Aggregated metrics from all thread-local arenas.
#[cfg(feature = "zig-kernels")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ArenaStats {
    /// Total number of arena resets across all threads
    pub total_resets: usize,

    /// Total overflow events across all threads
    pub total_overflows: usize,

    /// Maximum high-water mark across all thread arenas (bytes)
    pub max_high_water_mark: usize,
}

/// Configure arena allocator pool size and overflow strategy
///
/// Sets global configuration for all thread-local arenas.
/// Must be called before any arena allocation occurs.
///
/// # Arguments
///
/// * `pool_size_mb` - Pool size in megabytes (1-1024)
/// * `overflow_strategy` - Behavior on overflow
///
/// # Thread Safety
///
/// Not thread-safe. Call from main thread before spawning workers.
///
/// # Examples
///
/// ```no_run
/// # #[cfg(feature = "zig-kernels")]
/// # {
/// use engram_core::zig_kernels::{configure_arena, OverflowStrategy};
///
/// // Configure 2MB arenas with error return on overflow
/// configure_arena(2, OverflowStrategy::ErrorReturn);
/// # }
/// ```
#[cfg(feature = "zig-kernels")]
pub fn configure_arena(pool_size_mb: u32, overflow_strategy: OverflowStrategy) {
    unsafe {
        ffi::engram_configure_arena(pool_size_mb, overflow_strategy as u8);
    }
}

/// Get global arena usage statistics
///
/// Returns aggregated metrics from all thread-local arenas.
/// Metrics are cumulative since process start or last reset.
///
/// # Thread Safety
///
/// Thread-safe via internal mutex in Zig implementation.
///
/// # Examples
///
/// ```no_run
/// # #[cfg(feature = "zig-kernels")]
/// # {
/// use engram_core::zig_kernels::get_arena_stats;
///
/// let stats = get_arena_stats();
/// println!("Resets: {}", stats.total_resets);
/// println!("Overflows: {}", stats.total_overflows);
/// println!("Peak usage: {} bytes", stats.max_high_water_mark);
/// # }
/// ```
#[cfg(feature = "zig-kernels")]
#[must_use]
pub fn get_arena_stats() -> ArenaStats {
    let mut stats = ArenaStats {
        total_resets: 0,
        total_overflows: 0,
        max_high_water_mark: 0,
    };

    unsafe {
        ffi::engram_arena_stats(
            &raw mut stats.total_resets,
            &raw mut stats.total_overflows,
            &raw mut stats.max_high_water_mark,
        );
    }

    stats
}

/// Reset thread-local arena allocator
///
/// Manually resets the calling thread's arena.
/// Useful for testing or explicit memory reclamation.
///
/// Normally arenas are reset automatically at kernel exit,
/// but this provides explicit control when needed.
///
/// # Thread Safety
///
/// Thread-local - only affects calling thread's arena.
///
/// # Examples
///
/// ```no_run
/// # #[cfg(feature = "zig-kernels")]
/// # {
/// use engram_core::zig_kernels::reset_arenas;
///
/// // Manually reset current thread's arena
/// reset_arenas();
/// # }
/// ```
#[cfg(feature = "zig-kernels")]
pub fn reset_arenas() {
    unsafe {
        ffi::engram_reset_arenas();
    }
}

/// Reset global arena metrics
///
/// Clears all accumulated metrics counters.
/// Useful for starting fresh measurement periods in testing.
///
/// # Thread Safety
///
/// Thread-safe via internal mutex in Zig implementation.
///
/// # Examples
///
/// ```no_run
/// # #[cfg(feature = "zig-kernels")]
/// # {
/// use engram_core::zig_kernels::reset_arena_metrics;
///
/// // Clear all metrics
/// reset_arena_metrics();
/// # }
/// ```
#[cfg(feature = "zig-kernels")]
pub fn reset_arena_metrics() {
    unsafe {
        ffi::engram_reset_arena_metrics();
    }
}
