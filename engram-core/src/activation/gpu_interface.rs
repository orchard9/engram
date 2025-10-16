//! GPU abstraction layer for future CUDA/HIP acceleration
//!
//! This module defines the interface for GPU-accelerated activation spreading,
//! allowing CUDA kernels to be integrated without refactoring the spreading engine.

use super::{ActivationResult, SpreadingMetrics, simd_optimization::SimdActivationMapper};
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};
use std::time::Instant;
use tracing::warn;

/// GPU device capabilities for acceleration planning
#[derive(Debug, Clone)]
pub struct GpuCapabilities {
    /// Maximum batch size the GPU can process efficiently
    pub max_batch: usize,
    /// Whether unified memory is available (CPU/GPU shared memory)
    pub unified_memory: bool,
    /// Device name for telemetry and debugging
    pub device_name: String,
}

impl Default for GpuCapabilities {
    fn default() -> Self {
        Self {
            max_batch: 0,
            unified_memory: false,
            device_name: "CPU_FALLBACK".to_string(),
        }
    }
}

/// Metadata for batch processing context
#[derive(Debug, Clone)]
pub struct BatchMetadata {
    /// Depth in the activation spreading graph
    pub depth: u16,
    /// Temperature parameter for sigmoid activation
    pub temperature: f32,
    /// Activation threshold for filtering
    pub threshold: f32,
    /// Timestamp for latency tracking
    pub timestamp: Instant,
}

impl Default for BatchMetadata {
    fn default() -> Self {
        Self {
            depth: 0,
            temperature: 1.0,
            threshold: 0.01,
            timestamp: Instant::now(),
        }
    }
}

/// GPU-compatible batch container for activation spreading
///
/// Memory layout optimized for coalesced GPU access patterns.
/// Uses `repr(C, align(32))` for proper alignment with CUDA memory requirements.
#[repr(C, align(32))]
#[derive(Debug, Clone)]
pub struct GPUActivationBatch {
    /// Source node embedding for similarity computation
    pub source: [f32; 768],
    /// Target node embeddings for batch processing
    pub targets: Vec<[f32; 768]>,
    /// Activation levels for each target
    pub activations: Vec<f32>,
    /// Confidence scores for each activation
    pub confidences: Vec<f32>,
    /// Batch processing metadata
    pub meta: BatchMetadata,
}

impl GPUActivationBatch {
    /// Create a new GPU activation batch
    #[must_use]
    pub fn new(source: &[f32; 768]) -> Self {
        Self {
            source: *source,
            targets: Vec::new(),
            activations: Vec::new(),
            confidences: Vec::new(),
            meta: BatchMetadata::default(),
        }
    }

    /// Add a target node to the batch
    pub fn add_target(&mut self, target: &[f32; 768], activation: f32, confidence: f32) {
        self.targets.push(*target);
        self.activations.push(activation);
        self.confidences.push(confidence);
    }

    /// Get the batch size
    #[must_use]
    pub const fn size(&self) -> usize {
        self.targets.len()
    }

    /// Check if batch is empty
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.targets.is_empty()
    }

    /// Reserve capacity for batch operations
    pub fn reserve(&mut self, additional: usize) {
        self.targets.reserve(additional);
        self.activations.reserve(additional);
        self.confidences.reserve(additional);
    }

    /// Clear the batch for reuse
    pub fn clear(&mut self) {
        self.targets.clear();
        self.activations.clear();
        self.confidences.clear();
    }

    /// Ensure contiguous memory layout for GPU transfer
    pub fn ensure_contiguous(&mut self) {
        // Shrink to fit ensures vectors use contiguous memory
        self.targets.shrink_to_fit();
        self.activations.shrink_to_fit();
        self.confidences.shrink_to_fit();
    }
}

/// Future type for asynchronous GPU operations
pub struct GpuLaunchFuture {
    /// Result of the GPU operation
    result: Option<Vec<f32>>,
    /// Completion flag
    completed: bool,
}

impl GpuLaunchFuture {
    /// Create a completed future with results
    #[must_use]
    pub const fn ready(result: Vec<f32>) -> Self {
        Self {
            result: Some(result),
            completed: true,
        }
    }

    /// Create a pending future
    #[must_use]
    pub const fn pending() -> Self {
        Self {
            result: None,
            completed: false,
        }
    }
}

impl Future for GpuLaunchFuture {
    type Output = ActivationResult<Vec<f32>>;

    fn poll(mut self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Self::Output> {
        if self.completed {
            Poll::Ready(Ok(self.result.take().unwrap_or_default()))
        } else {
            Poll::Pending
        }
    }
}

#[doc = include_str!("doc/gpu_interface.md")]
/// Interface for GPU-accelerated spreading operations
///
/// This trait defines the contract for GPU backends (CUDA, HIP, etc.)
/// to integrate with the spreading engine.
pub trait GPUSpreadingInterface: Send + Sync {
    /// Get device capabilities
    fn capabilities(&self) -> &GpuCapabilities;

    /// Check if GPU acceleration is available
    fn is_available(&self) -> bool;

    /// Launch GPU kernel for batch activation spreading
    ///
    /// # Arguments
    /// * `batch` - GPU-compatible batch of activations to process
    ///
    /// # Returns
    /// Future that resolves to activation results
    fn launch(&self, batch: &GPUActivationBatch) -> GpuLaunchFuture;

    /// Warm up GPU context and allocate resources
    fn warm_up(&self) -> ActivationResult<()> {
        Ok(())
    }

    /// Clean up GPU resources
    fn cleanup(&self) -> ActivationResult<()> {
        Ok(())
    }
}

/// CPU fallback implementation using SIMD optimization
pub struct CpuFallback {
    /// Capabilities for CPU processing
    capabilities: GpuCapabilities,
}

impl CpuFallback {
    /// Create a new CPU fallback processor
    #[must_use]
    pub fn new() -> Self {
        Self {
            capabilities: GpuCapabilities {
                max_batch: 1024,
                unified_memory: true,
                device_name: "CPU_SIMD_FALLBACK".to_string(),
            },
        }
    }

    /// Process batch using SIMD operations
    fn process_batch(batch: &GPUActivationBatch) -> Vec<f32> {
        if batch.is_empty() {
            return Vec::new();
        }

        // Compute similarities between source and targets
        let similarities = batch
            .targets
            .iter()
            .map(|target| compute_cosine_similarity(&batch.source, target))
            .collect::<Vec<_>>();

        // Apply sigmoid activation mapping
        SimdActivationMapper::batch_sigmoid_activation(
            &similarities,
            batch.meta.temperature,
            batch.meta.threshold,
        )
    }
}

impl Default for CpuFallback {
    fn default() -> Self {
        Self::new()
    }
}

impl GPUSpreadingInterface for CpuFallback {
    fn capabilities(&self) -> &GpuCapabilities {
        &self.capabilities
    }

    fn is_available(&self) -> bool {
        true // CPU fallback is always available
    }

    fn launch(&self, batch: &GPUActivationBatch) -> GpuLaunchFuture {
        // Synchronous CPU processing wrapped in future
        let result = Self::process_batch(batch);
        GpuLaunchFuture::ready(result)
    }
}

/// Mock GPU interface for testing
pub struct MockGpuInterface {
    capabilities: GpuCapabilities,
    available: bool,
}

impl MockGpuInterface {
    /// Create a mock GPU with specified availability
    #[must_use]
    pub fn new(available: bool) -> Self {
        Self {
            capabilities: GpuCapabilities {
                max_batch: 2048,
                unified_memory: true,
                device_name: "MOCK_GPU_DEVICE".to_string(),
            },
            available,
        }
    }
}

impl GPUSpreadingInterface for MockGpuInterface {
    fn capabilities(&self) -> &GpuCapabilities {
        &self.capabilities
    }

    fn is_available(&self) -> bool {
        self.available
    }

    fn launch(&self, batch: &GPUActivationBatch) -> GpuLaunchFuture {
        // Mock GPU returns ones for all activations
        let result = vec![1.0; batch.size()];
        GpuLaunchFuture::ready(result)
    }
}

/// Adaptive spreading engine that dispatches to GPU or CPU
pub struct AdaptiveSpreadingEngine {
    /// Optional GPU interface
    gpu_interface: Option<Arc<dyn GPUSpreadingInterface>>,
    /// CPU fallback processor
    cpu_fallback: CpuFallback,
    /// Configuration for adaptive dispatch
    config: AdaptiveConfig,
    /// Cached GPU availability status
    gpu_available_cached: Option<bool>,
    /// Optional spreading metrics for telemetry
    metrics: Option<Arc<SpreadingMetrics>>,
}

/// Configuration for adaptive GPU/CPU dispatch
#[derive(Debug, Clone)]
pub struct AdaptiveConfig {
    /// Minimum batch size to consider GPU acceleration
    pub gpu_threshold: usize,
    /// Enable GPU acceleration
    pub enable_gpu: bool,
}

impl Default for AdaptiveConfig {
    fn default() -> Self {
        Self {
            gpu_threshold: 64,
            enable_gpu: false,
        }
    }
}

impl AdaptiveSpreadingEngine {
    /// Create a new adaptive spreading engine
    #[must_use]
    pub fn new(
        gpu_interface: Option<Arc<dyn GPUSpreadingInterface>>,
        config: AdaptiveConfig,
        metrics: Option<Arc<SpreadingMetrics>>,
    ) -> Self {
        Self {
            gpu_interface,
            cpu_fallback: CpuFallback::new(),
            config,
            gpu_available_cached: None,
            metrics,
        }
    }

    /// Check if GPU should be used for the given batch
    fn should_use_gpu(&mut self, batch_size: usize) -> bool {
        if !self.config.enable_gpu || batch_size < self.config.gpu_threshold {
            return false;
        }

        // Cache GPU availability check
        if self.gpu_available_cached.is_none() {
            self.gpu_available_cached = Some(
                self.gpu_interface
                    .as_ref()
                    .is_some_and(|gpu| gpu.is_available()),
            );

            // Warm up GPU context if available
            if self.gpu_available_cached == Some(true)
                && let Some(ref gpu) = self.gpu_interface
            {
                let _ = gpu.warm_up();
            }
        }

        self.gpu_available_cached.unwrap_or(false)
    }

    /// Process activation batch adaptively
    pub async fn spread(&mut self, batch: &GPUActivationBatch) -> ActivationResult<Vec<f32>> {
        if self.should_use_gpu(batch.size()) {
            // Use GPU acceleration
            if let Some(ref gpu) = self.gpu_interface {
                if let Some(ref metrics) = self.metrics {
                    metrics.record_gpu_launch();
                }
                match gpu.launch(batch).await {
                    Ok(result) => return Ok(result),
                    Err(error) => {
                        if let Some(ref metrics) = self.metrics {
                            metrics.record_gpu_fallback();
                        }
                        warn!(
                            target = "engram::activation::gpu",
                            ?error,
                            "GPU launch failed, falling back to CPU"
                        );
                        return Ok(CpuFallback::process_batch(batch));
                    }
                }
            }

            if let Some(ref metrics) = self.metrics {
                metrics.record_gpu_fallback();
            }
        }

        // Fall back to CPU SIMD
        Ok(CpuFallback::process_batch(batch))
    }

    /// Get the current processing backend name
    #[must_use]
    pub fn backend_name(&self) -> &str {
        if self.gpu_available_cached == Some(true) {
            self.gpu_interface
                .as_ref()
                .map_or("CPU", |gpu| &gpu.capabilities().device_name)
        } else {
            &self.cpu_fallback.capabilities().device_name
        }
    }
}

/// Compute cosine similarity between two embeddings
#[inline]
fn compute_cosine_similarity(a: &[f32; 768], b: &[f32; 768]) -> f32 {
    let mut dot_product = 0.0;
    let mut norm_a = 0.0;
    let mut norm_b = 0.0;

    for i in 0..768 {
        dot_product += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }

    if norm_a > 0.0 && norm_b > 0.0 {
        dot_product / (norm_a.sqrt() * norm_b.sqrt())
    } else {
        0.0
    }
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_capabilities_default() {
        let caps = GpuCapabilities::default();
        assert_eq!(caps.max_batch, 0);
        assert!(!caps.unified_memory);
        assert_eq!(caps.device_name, "CPU_FALLBACK");
    }

    #[test]
    fn test_gpu_batch_operations() {
        let source = [1.0; 768];
        let mut batch = GPUActivationBatch::new(&source);
        assert!(batch.is_empty());
        assert_eq!(batch.size(), 0);

        let target = [0.5; 768];
        batch.add_target(&target, 0.7, 0.8);
        assert!(!batch.is_empty());
        assert_eq!(batch.size(), 1);
        assert!((batch.activations[0] - 0.7).abs() < f32::EPSILON);
        assert!((batch.confidences[0] - 0.8).abs() < f32::EPSILON);

        batch.clear();
        assert!(batch.is_empty());
    }

    #[test]
    fn test_cpu_fallback() {
        let fallback = CpuFallback::new();
        assert!(fallback.is_available());
        assert_eq!(fallback.capabilities().device_name, "CPU_SIMD_FALLBACK");

        let source = [1.0; 768];
        let mut batch = GPUActivationBatch::new(&source);
        let target1 = [1.0; 768];
        let target2 = [0.0; 768];
        batch.add_target(&target1, 0.5, 0.6);
        batch.add_target(&target2, 0.3, 0.4);

        let future = fallback.launch(&batch);
        // In real async context, this would be awaited
        // For now, we just verify it's created successfully
        assert!(future.result.is_some());
    }

    #[test]
    fn test_adaptive_engine_cpu_only() {
        let config = AdaptiveConfig {
            gpu_threshold: 100,
            enable_gpu: false,
        };
        let mut engine = AdaptiveSpreadingEngine::new(None, config, None);

        // Should always use CPU when GPU disabled
        assert!(!engine.should_use_gpu(200));
        assert_eq!(engine.backend_name(), "CPU_SIMD_FALLBACK");
    }

    #[test]
    fn test_adaptive_engine_threshold() {
        let mock_gpu = Arc::new(MockGpuInterface::new(true));
        let config = AdaptiveConfig {
            gpu_threshold: 50,
            enable_gpu: true,
        };
        let mut engine = AdaptiveSpreadingEngine::new(Some(mock_gpu), config, None);

        // Below threshold - use CPU
        assert!(!engine.should_use_gpu(30));

        // Above threshold - use GPU
        assert!(engine.should_use_gpu(100));
        assert_eq!(engine.backend_name(), "MOCK_GPU_DEVICE");
    }

    #[test]
    fn test_cosine_similarity() {
        let a = [1.0; 768];
        let b = [1.0; 768];
        let similarity = compute_cosine_similarity(&a, &b);
        assert!((similarity - 1.0).abs() < 1e-6);

        let c = [0.0; 768];
        let similarity_zero = compute_cosine_similarity(&a, &c);
        assert!((similarity_zero - 0.0).abs() < 1e-6);
    }

    #[tokio::test]
    async fn test_adaptive_spread() {
        let mock_gpu = Arc::new(MockGpuInterface::new(true));
        let config = AdaptiveConfig {
            gpu_threshold: 1,
            enable_gpu: true,
        };
        let mut engine = AdaptiveSpreadingEngine::new(Some(mock_gpu), config, None);

        let source = [1.0; 768];
        let mut batch = GPUActivationBatch::new(&source);
        let target = [0.5; 768];
        batch.add_target(&target, 0.7, 0.8);

        let result = engine.spread(&batch).await.expect("spread should succeed");
        assert_eq!(result.len(), 1);
        assert!((result[0] - 1.0).abs() < f32::EPSILON); // Mock GPU returns 1.0
    }
}
