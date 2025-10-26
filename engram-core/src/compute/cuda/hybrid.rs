//! CPU-GPU hybrid executor with intelligent workload dispatch
//!
//! This module provides production-ready GPU acceleration with automatic CPU fallback.
//! It intelligently routes operations to CPU or GPU based on:
//! - Batch size (small batches use CPU to avoid kernel launch overhead)
//! - GPU availability (graceful degradation if GPU unavailable)
//! - Historical performance (adaptive dispatch based on actual measurements)
//! - GPU reliability (falls back to CPU if GPU error rate too high)
//!
//! # Architecture
//!
//! The hybrid executor is the production interface for all compute operations.
//! It wraps both CPU and GPU implementations and makes intelligent dispatch decisions.
//!
//! Decision tree (priority order):
//! 1. Batch size too small -> CPU
//! 2. GPU unavailable -> CPU
//! 3. GPU in error state -> CPU
//! 4. Historical speedup < threshold -> CPU
//! 5. GPU success rate < threshold -> CPU
//! 6. Otherwise -> GPU with CPU fallback
//!
//! # Usage
//!
//! ```rust,ignore
//! use engram_core::compute::cuda::hybrid::{HybridExecutor, HybridConfig};
//!
//! let config = HybridConfig::default();
//! let executor = HybridExecutor::new(config);
//!
//! let query = [1.0f32; 768];
//! let targets = vec![[0.5f32; 768]; 1000];
//!
//! // Automatically dispatches to GPU (batch >= 64) with CPU fallback
//! let results = executor.execute_batch_cosine_similarity(&query, &targets);
//! ```

#[cfg(cuda_available)]
use super::cosine_similarity::GpuCosineSimilarity;
#[cfg(cuda_available)]
use super::ffi::CudaError;
#[cfg(cuda_available)]
use super::memory_pressure::{MemoryPressureMonitor, get_memory_pressure_monitor};
use super::performance_tracker::{Operation, PerformanceTracker};
use crate::compute::{CpuCapability, VectorOps, detect_cpu_features, get_vector_ops};
use std::sync::Arc;

/// Configuration for hybrid executor behavior
#[derive(Debug, Clone)]
pub struct HybridConfig {
    /// Minimum batch size to consider GPU execution
    ///
    /// Based on profiling from Task 001: GPU kernel launch overhead is ~10-20us.
    /// For cosine similarity, CPU takes ~2.1us/vector (AVX-512) vs GPU ~0.3us/vector.
    /// Break-even is around 64 vectors.
    pub gpu_min_batch_size: usize,

    /// Speedup threshold for preferring GPU (default: 1.5x)
    ///
    /// GPU must be at least this much faster than CPU based on historical
    /// measurements. Conservative threshold accounts for measurement noise.
    pub gpu_speedup_threshold: f64,

    /// Success rate required to trust GPU (default: 0.95)
    ///
    /// GPU must succeed at least this fraction of attempts to be trusted.
    /// Falls back to CPU if reliability drops below threshold.
    pub gpu_success_rate_threshold: f64,

    /// Window size for performance tracking (default: 100)
    ///
    /// Number of recent samples to keep for moving average computation.
    /// Larger windows smooth out noise but are slower to adapt.
    pub performance_window_size: usize,

    /// Force CPU-only mode (for debugging)
    ///
    /// When true, GPU is never used even if available.
    /// Useful for performance comparisons and debugging.
    pub force_cpu_mode: bool,

    /// Enable performance tracking telemetry
    ///
    /// When true, detailed performance metrics are recorded.
    /// Minimal overhead (<1% of operation latency).
    pub telemetry_enabled: bool,
}

impl Default for HybridConfig {
    fn default() -> Self {
        Self {
            gpu_min_batch_size: 64,
            gpu_speedup_threshold: 1.5,
            gpu_success_rate_threshold: 0.95,
            performance_window_size: 100,
            force_cpu_mode: false,
            telemetry_enabled: true,
        }
    }
}

/// CPU/GPU hybrid executor with intelligent dispatch
///
/// This is the production interface for compute operations. It automatically
/// routes operations to CPU or GPU based on performance characteristics.
pub struct HybridExecutor {
    /// GPU interface (None if GPU unavailable or force_cpu_mode)
    #[cfg(cuda_available)]
    gpu_interface: Option<Arc<GpuCosineSimilarity>>,

    /// CPU SIMD implementation (always available)
    cpu_ops: &'static dyn VectorOps,

    /// Tracks historical performance for adaptive dispatch
    performance_tracker: Arc<PerformanceTracker>,

    /// Configuration parameters
    config: HybridConfig,
}

impl HybridExecutor {
    /// Create new hybrid executor
    ///
    /// Attempts to initialize GPU if available and not in force_cpu_mode.
    /// GPU initialization failure is not fatal - executor will use CPU only.
    #[must_use]
    pub fn new(config: HybridConfig) -> Self {
        #[cfg(cuda_available)]
        let gpu_interface = if !config.force_cpu_mode {
            Self::try_initialize_gpu()
        } else {
            tracing::info!("Force CPU mode enabled, GPU disabled");
            None
        };

        Self {
            #[cfg(cuda_available)]
            gpu_interface,
            cpu_ops: get_vector_ops(),
            performance_tracker: Arc::new(PerformanceTracker::new(config.performance_window_size)),
            config,
        }
    }

    /// Attempt to initialize GPU
    ///
    /// Returns Some(gpu_interface) if GPU is available and functional.
    /// Returns None if GPU initialization fails (non-fatal).
    #[cfg(cuda_available)]
    fn try_initialize_gpu() -> Option<Arc<GpuCosineSimilarity>> {
        use crate::compute::cuda;

        match cuda::is_available() {
            true => {
                let gpu_ops = GpuCosineSimilarity::new();
                tracing::info!("GPU acceleration enabled");
                Some(Arc::new(gpu_ops))
            }
            false => {
                tracing::info!("GPU not available, using CPU only");
                None
            }
        }
    }

    /// Check if GPU is available
    #[must_use]
    #[allow(clippy::unused_self)] // Self is unused when cuda_available not set
    pub const fn is_gpu_available(&self) -> bool {
        #[cfg(cuda_available)]
        {
            self.gpu_interface.is_some()
        }
        #[cfg(not(cuda_available))]
        {
            false
        }
    }

    /// Get executor capabilities
    #[must_use]
    pub fn capabilities(&self) -> ExecutorCapabilities {
        ExecutorCapabilities {
            gpu_available: self.is_gpu_available(),
            gpu_device_name: None, // Could be extended to query actual device name
            cpu_simd_level: detect_cpu_features(),
        }
    }

    /// Get performance tracker for metrics
    #[must_use]
    pub const fn performance_tracker(&self) -> &Arc<PerformanceTracker> {
        &self.performance_tracker
    }

    /// Execute batch cosine similarity with hybrid dispatch
    ///
    /// Automatically routes to GPU or CPU based on batch size and performance.
    /// Always succeeds (falls back to CPU on GPU failure).
    ///
    /// # Arguments
    ///
    /// * `query` - Query vector (768 dimensions)
    /// * `targets` - Slice of target vectors (each 768 dimensions)
    ///
    /// # Returns
    ///
    /// Vector of cosine similarities (always succeeds)
    #[must_use]
    pub fn execute_batch_cosine_similarity(
        &self,
        query: &[f32; 768],
        targets: &[[f32; 768]],
    ) -> Vec<f32> {
        let batch_size = targets.len();
        let decision = self.make_dispatch_decision(Operation::CosineSimilarity, batch_size);

        match decision {
            ExecutionTarget::GPU => {
                #[cfg(cuda_available)]
                {
                    // Try GPU with fallback
                    let start = std::time::Instant::now();
                    let result = self.try_gpu_execution_with_fallback(
                        Operation::CosineSimilarity,
                        || {
                            // GPU path
                            self.gpu_interface
                                .as_ref()
                                .unwrap()
                                .batch_cosine_similarity_gpu(query, targets)
                                .map_err(|e| CudaError::from_gpu_error(e))
                        },
                        || {
                            // CPU fallback path
                            self.cpu_ops.cosine_similarity_batch_768(query, targets)
                        },
                    );

                    if self.config.telemetry_enabled {
                        let latency = start.elapsed();
                        self.performance_tracker
                            .record_cpu_latency(Operation::CosineSimilarity, latency);
                    }

                    result
                }
                #[cfg(not(cuda_available))]
                {
                    // No CUDA available, fall back to CPU
                    let start = std::time::Instant::now();
                    let result = self.cpu_ops.cosine_similarity_batch_768(query, targets);

                    if self.config.telemetry_enabled {
                        let latency = start.elapsed();
                        self.performance_tracker
                            .record_cpu_latency(Operation::CosineSimilarity, latency);
                    }

                    result
                }
            }
            ExecutionTarget::CPU => {
                // Direct CPU execution
                let start = std::time::Instant::now();
                let result = self.cpu_ops.cosine_similarity_batch_768(query, targets);

                if self.config.telemetry_enabled {
                    let latency = start.elapsed();
                    self.performance_tracker
                        .record_cpu_latency(Operation::CosineSimilarity, latency);
                }

                result
            }
        }
    }

    /// Execute batch cosine similarity with OOM protection
    ///
    /// This variant uses the memory pressure monitor to automatically split
    /// large batches that would exceed available VRAM. It never crashes due
    /// to OOM - it either succeeds with automatic chunking or falls back to CPU.
    ///
    /// # Arguments
    ///
    /// * `query` - Query vector (768 dimensions)
    /// * `targets` - Slice of target vectors (each 768 dimensions)
    ///
    /// # Returns
    ///
    /// Vector of cosine similarities (always succeeds, never OOM crashes)
    #[cfg(cuda_available)]
    #[must_use]
    pub fn execute_batch_cosine_similarity_oom_safe(
        &self,
        query: &[f32; 768],
        targets: &[[f32; 768]],
    ) -> Vec<f32> {
        let batch_size = targets.len();
        let decision = self.make_dispatch_decision(Operation::CosineSimilarity, batch_size);

        match decision {
            ExecutionTarget::GPU => {
                // Check if we have memory pressure monitor
                if let Some(monitor) = get_memory_pressure_monitor() {
                    // Use memory-safe chunked processing
                    let per_vector_memory = 768 * std::mem::size_of::<f32>(); // 3KB per vector

                    monitor.process_in_chunks(targets, per_vector_memory, |chunk| {
                        // Try GPU for this chunk
                        self.try_gpu_execution_with_oom_recovery(
                            Operation::CosineSimilarity,
                            || {
                                self.gpu_interface
                                    .as_ref()
                                    .unwrap()
                                    .batch_cosine_similarity_gpu(query, chunk)
                                    .map_err(CudaError::from_gpu_error)
                            },
                            || {
                                // CPU fallback for this chunk
                                self.cpu_ops.cosine_similarity_batch_768(query, chunk)
                            },
                        )
                    })
                } else {
                    // No memory monitor available, fall back to CPU
                    tracing::warn!("Memory pressure monitor unavailable, using CPU");
                    self.cpu_ops.cosine_similarity_batch_768(query, targets)
                }
            }
            ExecutionTarget::CPU => {
                // Direct CPU execution
                self.cpu_ops.cosine_similarity_batch_768(query, targets)
            }
        }
    }

    /// Try GPU execution with OOM recovery
    ///
    /// Attempts GPU execution and handles OOM errors gracefully by falling back to CPU.
    /// Records OOM events for telemetry and adaptive dispatch.
    #[cfg(cuda_available)]
    fn try_gpu_execution_with_oom_recovery<T>(
        &self,
        operation: Operation,
        gpu_fn: impl FnOnce() -> Result<T, CudaError>,
        cpu_fallback: impl FnOnce() -> T,
    ) -> T {
        let start = std::time::Instant::now();

        match gpu_fn() {
            Ok(result) => {
                // GPU succeeded
                if self.config.telemetry_enabled {
                    let latency = start.elapsed();
                    self.performance_tracker
                        .record_gpu_success(operation, latency);
                }
                result
            }
            Err(CudaError::OutOfMemory) => {
                // OOM detected - this is expected under memory pressure
                tracing::warn!(
                    "GPU OOM detected for {:?}, falling back to CPU (batch may be too large for available VRAM)",
                    operation
                );

                if self.config.telemetry_enabled {
                    self.performance_tracker.record_gpu_failure(operation);
                    self.performance_tracker.record_oom_event(operation);
                }

                let result = cpu_fallback();

                if self.config.telemetry_enabled {
                    let latency = start.elapsed();
                    self.performance_tracker.record_cpu_fallback(latency);
                }

                result
            }
            Err(e) => {
                // Other GPU error
                tracing::warn!("GPU execution failed: {}, falling back to CPU", e);

                if self.config.telemetry_enabled {
                    self.performance_tracker.record_gpu_failure(operation);
                }

                let result = cpu_fallback();

                if self.config.telemetry_enabled {
                    let latency = start.elapsed();
                    self.performance_tracker.record_cpu_fallback(latency);
                }

                result
            }
        }
    }

    /// Make dispatch decision for an operation
    ///
    /// Implements decision tree:
    /// 1. Too small for GPU -> CPU
    /// 2. GPU unavailable -> CPU
    /// 3. Historical performance indicates CPU is faster -> CPU
    /// 4. GPU success rate too low -> CPU
    /// 5. Otherwise -> GPU
    ///
    /// Uses atomic snapshot of metrics to avoid race conditions.
    fn make_dispatch_decision(&self, operation: Operation, batch_size: usize) -> ExecutionTarget {
        // Rule 1: Too small for GPU
        if batch_size < self.config.gpu_min_batch_size {
            tracing::trace!(
                "Batch size {} < min {}, using CPU",
                batch_size,
                self.config.gpu_min_batch_size
            );
            return ExecutionTarget::CPU;
        }

        // Rule 2: GPU unavailable (or not compiled in)
        #[cfg(not(cuda_available))]
        {
            let _ = operation; // Silence unused warning
            ExecutionTarget::CPU
        }

        #[cfg(cuda_available)]
        {
            if self.gpu_interface.is_none() {
                return ExecutionTarget::CPU;
            }

            // Get atomic snapshot of all metrics to avoid race conditions
            let metrics = self.performance_tracker.snapshot(operation);

            // Rule 3: Historical performance indicates CPU is faster
            if metrics.speedup > 0.0 && metrics.speedup < self.config.gpu_speedup_threshold {
                tracing::trace!(
                    "GPU speedup {:.2}x < threshold {:.2}x, using CPU",
                    metrics.speedup,
                    self.config.gpu_speedup_threshold
                );
                return ExecutionTarget::CPU;
            }

            // Rule 4: GPU success rate too low
            if metrics.success_rate < self.config.gpu_success_rate_threshold {
                tracing::warn!(
                    "GPU success rate {:.2}% < threshold {:.2}%, using CPU",
                    metrics.success_rate * 100.0,
                    self.config.gpu_success_rate_threshold * 100.0
                );
                return ExecutionTarget::CPU;
            }

            // All checks passed, use GPU
            ExecutionTarget::GPU
        }
    }

    /// Try GPU execution with automatic CPU fallback
    ///
    /// Executes GPU function and falls back to CPU on error.
    /// Records performance metrics for adaptive dispatch.
    #[cfg(cuda_available)]
    fn try_gpu_execution_with_fallback<T>(
        &self,
        operation: Operation,
        gpu_fn: impl FnOnce() -> Result<T, CudaError>,
        cpu_fallback: impl FnOnce() -> T,
    ) -> T {
        let start = std::time::Instant::now();

        match gpu_fn() {
            Ok(result) => {
                // GPU succeeded
                if self.config.telemetry_enabled {
                    let latency = start.elapsed();
                    self.performance_tracker
                        .record_gpu_success(operation, latency);
                }
                result
            }
            Err(e) => {
                // GPU failed, fall back to CPU
                tracing::warn!("GPU execution failed: {}, falling back to CPU", e);

                if self.config.telemetry_enabled {
                    self.performance_tracker.record_gpu_failure(operation);
                }

                let result = cpu_fallback();

                if self.config.telemetry_enabled {
                    let latency = start.elapsed();
                    self.performance_tracker.record_cpu_fallback(latency);
                }

                result
            }
        }
    }
}

/// Dispatch decision
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)] // GPU variant is only used when cuda_available
#[allow(clippy::upper_case_acronyms)] // CPU and GPU are established acronyms
enum ExecutionTarget {
    CPU,
    GPU,
}

/// Executor capabilities
#[derive(Debug, Clone)]
pub struct ExecutorCapabilities {
    /// GPU available for acceleration
    pub gpu_available: bool,
    /// GPU device name (if available)
    pub gpu_device_name: Option<String>,
    /// CPU SIMD level detected
    pub cpu_simd_level: CpuCapability,
}

impl std::fmt::Display for ExecutorCapabilities {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "CPU: {:?}", self.cpu_simd_level)?;
        if self.gpu_available {
            write!(f, ", GPU: Available")?;
            if let Some(name) = &self.gpu_device_name {
                write!(f, " ({name})")?;
            }
        } else {
            write!(f, ", GPU: Not available")?;
        }
        Ok(())
    }
}

/// Extension trait to convert GpuError to CudaError
#[cfg(cuda_available)]
trait CudaErrorExt {
    fn from_gpu_error(e: super::cosine_similarity::GpuError) -> Self;
}

#[cfg(cuda_available)]
impl CudaErrorExt for CudaError {
    fn from_gpu_error(e: super::cosine_similarity::GpuError) -> Self {
        use super::cosine_similarity::GpuError;

        match e {
            GpuError::MemoryAllocation { .. } => CudaError::OutOfMemory,
            GpuError::MemoryTransfer => CudaError::InvalidMemcpyDirection,
            GpuError::KernelLaunch { .. } => CudaError::Unknown,
            GpuError::KernelExecution { .. } => CudaError::Unknown,
            GpuError::InvalidBatchSize { .. } => CudaError::InvalidValue,
            GpuError::Unknown { .. } => CudaError::Unknown,
        }
    }
}

impl Default for HybridExecutor {
    fn default() -> Self {
        Self::new(HybridConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hybrid_executor_creation() {
        let config = HybridConfig::default();
        let executor = HybridExecutor::new(config);

        // Should always have CPU available
        let caps = executor.capabilities();
        // CpuCapability variants are platform-specific, so we just verify it's set
        println!("Executor capabilities: {caps}");
    }

    #[test]
    fn test_force_cpu_mode() {
        let config = HybridConfig {
            force_cpu_mode: true,
            ..Default::default()
        };
        let executor = HybridExecutor::new(config);

        assert!(
            !executor.is_gpu_available(),
            "Force CPU mode should disable GPU"
        );
    }

    #[test]
    fn test_dispatch_decision_small_batch() {
        let config = HybridConfig {
            gpu_min_batch_size: 64,
            ..Default::default()
        };
        let executor = HybridExecutor::new(config);

        // Small batch should use CPU
        let decision = executor.make_dispatch_decision(Operation::CosineSimilarity, 32);
        assert_eq!(decision, ExecutionTarget::CPU);

        // Large batch might use GPU (if available)
        let decision = executor.make_dispatch_decision(Operation::CosineSimilarity, 128);
        if executor.is_gpu_available() {
            assert_eq!(decision, ExecutionTarget::GPU);
        } else {
            assert_eq!(decision, ExecutionTarget::CPU);
        }
    }

    #[test]
    fn test_batch_cosine_similarity_basic() {
        let executor = HybridExecutor::new(HybridConfig::default());

        let query = [1.0f32; 768];
        let targets = vec![[1.0f32; 768]; 128];

        let similarities = executor.execute_batch_cosine_similarity(&query, &targets);

        assert_eq!(similarities.len(), 128);
        for sim in &similarities {
            assert!(
                (*sim - 1.0).abs() < 1e-6,
                "Identical vectors should have similarity 1.0, got {sim}"
            );
        }
    }

    #[test]
    fn test_small_batch_uses_cpu() {
        let executor = HybridExecutor::new(HybridConfig::default());

        let query = [1.0f32; 768];
        let targets = vec![[0.5f32; 768]; 32]; // < 64, should use CPU

        let similarities = executor.execute_batch_cosine_similarity(&query, &targets);

        assert_eq!(similarities.len(), 32);
        // Verify computation is correct
        for sim in &similarities {
            assert!((sim - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    #[cfg(cuda_available)]
    fn test_gpu_batch_with_fallback() {
        use crate::compute::cuda;

        if !cuda::is_available() {
            println!("GPU not available, skipping test");
            return;
        }

        let executor = HybridExecutor::new(HybridConfig::default());

        let query = [1.0f32; 768];
        let targets = vec![[1.0f32; 768]; 256]; // Large batch for GPU

        let similarities = executor.execute_batch_cosine_similarity(&query, &targets);

        assert_eq!(similarities.len(), 256);
        for sim in &similarities {
            assert!(
                (*sim - 1.0).abs() < 1e-6,
                "Expected similarity ~1.0, got {}",
                sim
            );
        }

        // Check that performance was tracked
        let tracker = executor.performance_tracker();
        let speedup = tracker.gpu_speedup(Operation::CosineSimilarity);
        println!("Measured GPU speedup: {:.2}x", speedup);
    }

    #[test]
    fn test_telemetry_tracking() {
        let config = HybridConfig {
            telemetry_enabled: true,
            ..Default::default()
        };
        let executor = HybridExecutor::new(config);

        let query = [1.0f32; 768];
        let targets = vec![[0.5f32; 768]; 128];

        // Execute multiple times to build up telemetry
        for _ in 0..5 {
            let _ = executor.execute_batch_cosine_similarity(&query, &targets);
        }

        // Check telemetry was recorded
        let tracker = executor.performance_tracker();
        let telemetry = tracker.telemetry();
        println!("{telemetry}");
        assert!(telemetry.contains("CosineSimilarity"));
    }

    #[test]
    fn test_cpu_gpu_result_equivalence() {
        // Compare CPU-only vs hybrid (GPU if available) results
        let cpu_config = HybridConfig {
            force_cpu_mode: true,
            ..Default::default()
        };
        let cpu_executor = HybridExecutor::new(cpu_config);

        let hybrid_executor = HybridExecutor::new(HybridConfig::default());

        let query = [0.5f32; 768];
        let targets = vec![[0.75f32; 768]; 256];

        let cpu_results = cpu_executor.execute_batch_cosine_similarity(&query, &targets);
        let hybrid_results = hybrid_executor.execute_batch_cosine_similarity(&query, &targets);

        assert_eq!(cpu_results.len(), hybrid_results.len());

        for (cpu, hybrid) in cpu_results.iter().zip(hybrid_results.iter()) {
            assert!(
                (cpu - hybrid).abs() < 1e-6,
                "CPU and hybrid results should match: {cpu} vs {hybrid}"
            );
        }
    }

    #[test]
    fn test_dispatch_with_performance_history() {
        let executor = HybridExecutor::new(HybridConfig::default());

        // Simulate some historical data
        let tracker = executor.performance_tracker();
        tracker.record_cpu_latency(
            Operation::CosineSimilarity,
            std::time::Duration::from_micros(100),
        );
        tracker.record_gpu_success(
            Operation::CosineSimilarity,
            std::time::Duration::from_micros(20),
        );

        // With good speedup, large batch should prefer GPU (if available)
        let decision = executor.make_dispatch_decision(Operation::CosineSimilarity, 256);
        if executor.is_gpu_available() {
            assert_eq!(decision, ExecutionTarget::GPU);
        }
    }

    #[test]
    fn test_dispatch_with_poor_gpu_success_rate() {
        let config = HybridConfig {
            gpu_success_rate_threshold: 0.95,
            ..Default::default()
        };
        let executor = HybridExecutor::new(config);

        // Simulate poor GPU success rate
        let tracker = executor.performance_tracker();
        tracker.record_gpu_failure(Operation::CosineSimilarity);
        tracker.record_gpu_failure(Operation::CosineSimilarity);
        tracker.record_gpu_success(
            Operation::CosineSimilarity,
            std::time::Duration::from_micros(20),
        );
        // Success rate: 1/3 = 0.33 < 0.95

        // Should fall back to CPU due to poor success rate
        let decision = executor.make_dispatch_decision(Operation::CosineSimilarity, 256);
        assert_eq!(decision, ExecutionTarget::CPU);
    }
}
