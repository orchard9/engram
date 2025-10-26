# Task 007: CPU-GPU Hybrid Executor

**Status**: Pending
**Estimated Duration**: 2 days
**Priority**: Critical (enables production GPU usage)
**Owner**: Systems Architect

## Objective

Implement intelligent workload dispatcher that automatically routes operations to CPU or GPU based on performance characteristics, batch size, and historical measurements. Ensures graceful CPU fallback when GPU fails or is unavailable.

## Background

Raw GPU access isn't sufficient for production use. We need a hybrid executor that:
1. Routes small batches to CPU (where they're faster due to no kernel launch overhead)
2. Routes large batches to GPU (where parallelism provides speedup)
3. Tracks actual performance and adapts dispatch decisions
4. Handles GPU failures transparently with CPU fallback

This is the production interface that all Engram operations will use. It must be bulletproof.

## Deliverables

1. **HybridExecutor Implementation**
   - Performance-based dispatch logic
   - Automatic GPU capability detection
   - CPU fallback on GPU failure
   - Feature flag for force-CPU mode

2. **Performance Tracking**
   - Moving average of CPU vs GPU latencies
   - Success rate tracking per backend
   - Adaptive decision making based on history
   - Telemetry export for monitoring

3. **Integration with Existing APIs**
   - Replace direct `VectorOps` calls with hybrid dispatch
   - Integrate with `BatchEngine`
   - Update `ParallelSpreadingEngine` to use hybrid executor
   - Maintain API compatibility

4. **Configuration System**
   - Break-even batch sizes per operation
   - CPU fallback thresholds
   - Performance tracking window sizes
   - Feature flags for debugging

## Technical Specification

### HybridExecutor Architecture

```rust
// engram-core/src/compute/cuda/hybrid.rs

use super::ffi::CudaError;
use super::performance_tracker::PerformanceTracker;
use crate::compute::{VectorOps, get_vector_ops};
use std::sync::Arc;

pub struct HybridExecutor {
    /// GPU interface (None if GPU unavailable)
    gpu_interface: Option<Arc<dyn GPUSpreadingInterface>>,

    /// CPU SIMD implementation (always available)
    cpu_ops: Arc<dyn VectorOps>,

    /// Tracks historical performance for adaptive dispatch
    performance_tracker: Arc<PerformanceTracker>,

    /// Configuration parameters
    config: HybridConfig,
}

impl HybridExecutor {
    pub fn new(config: HybridConfig) -> Self {
        let gpu_interface = if !config.force_cpu_mode {
            Self::try_initialize_gpu()
        } else {
            None
        };

        Self {
            gpu_interface,
            cpu_ops: Arc::new(get_vector_ops()),
            performance_tracker: Arc::new(PerformanceTracker::new(
                config.performance_window_size
            )),
            config,
        }
    }

    fn try_initialize_gpu() -> Option<Arc<dyn GPUSpreadingInterface>> {
        match detect_gpu_capabilities() {
            Some(caps) if caps.is_usable() => {
                match create_gpu_interface(caps) {
                    Ok(interface) => {
                        tracing::info!(
                            "GPU initialized: {}",
                            interface.capabilities().device_name
                        );
                        Some(Arc::new(interface))
                    }
                    Err(e) => {
                        tracing::warn!("GPU initialization failed: {:?}", e);
                        None
                    }
                }
            }
            _ => None,
        }
    }

    pub fn is_gpu_available(&self) -> bool {
        self.gpu_interface.is_some()
    }

    pub fn capabilities(&self) -> ExecutorCapabilities {
        ExecutorCapabilities {
            gpu_available: self.is_gpu_available(),
            gpu_device_name: self.gpu_interface
                .as_ref()
                .map(|gpu| gpu.capabilities().device_name.clone()),
            cpu_simd_level: detect_cpu_features(),
        }
    }
}

/// Configuration for hybrid executor behavior
#[derive(Debug, Clone)]
pub struct HybridConfig {
    /// Minimum batch size to consider GPU execution
    pub gpu_min_batch_size: usize,

    /// Speedup threshold for preferring GPU (default: 1.5x)
    pub gpu_speedup_threshold: f64,

    /// Success rate required to trust GPU (default: 0.95)
    pub gpu_success_rate_threshold: f64,

    /// Window size for performance tracking (default: 100)
    pub performance_window_size: usize,

    /// Force CPU-only mode (for debugging)
    pub force_cpu_mode: bool,

    /// Enable performance tracking telemetry
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

/// Dispatch decision logic
impl HybridExecutor {
    pub fn execute_batch_cosine_similarity(
        &self,
        query: &[f32; 768],
        targets: &[[f32; 768]],
    ) -> Vec<f32> {
        let batch_size = targets.len();
        let decision = self.make_dispatch_decision(
            Operation::CosineSimilarity,
            batch_size,
        );

        match decision {
            ExecutionTarget::GPU => {
                self.try_gpu_execution_with_fallback(
                    || self.gpu_cosine_similarity(query, targets),
                    || self.cpu_ops.cosine_similarity_batch_768(query, targets),
                )
            }
            ExecutionTarget::CPU => {
                self.cpu_ops.cosine_similarity_batch_768(query, targets)
            }
        }
    }

    fn make_dispatch_decision(
        &self,
        operation: Operation,
        batch_size: usize,
    ) -> ExecutionTarget {
        // Decision tree based on profiling data and historical performance

        // Rule 1: Too small for GPU
        let min_batch = self.config.gpu_min_batch_size;
        if batch_size < min_batch {
            tracing::trace!("Batch size {} < min {}, using CPU", batch_size, min_batch);
            return ExecutionTarget::CPU;
        }

        // Rule 2: GPU unavailable
        if self.gpu_interface.is_none() {
            return ExecutionTarget::CPU;
        }

        // Rule 3: GPU in error state
        let gpu = self.gpu_interface.as_ref().unwrap();
        if !gpu.is_available() {
            tracing::warn!("GPU in error state, falling back to CPU");
            self.performance_tracker.record_gpu_unavailable();
            return ExecutionTarget::CPU;
        }

        // Rule 4: Historical performance indicates CPU is faster
        let speedup = self.performance_tracker.gpu_speedup(operation);
        if speedup < self.config.gpu_speedup_threshold {
            tracing::trace!(
                "GPU speedup {:.2}x < threshold {:.2}x, using CPU",
                speedup,
                self.config.gpu_speedup_threshold
            );
            return ExecutionTarget::CPU;
        }

        // Rule 5: GPU success rate too low
        let success_rate = self.performance_tracker.gpu_success_rate(operation);
        if success_rate < self.config.gpu_success_rate_threshold {
            tracing::warn!(
                "GPU success rate {:.2}% < threshold {:.2}%, using CPU",
                success_rate * 100.0,
                self.config.gpu_success_rate_threshold * 100.0
            );
            return ExecutionTarget::CPU;
        }

        // All checks passed, use GPU
        ExecutionTarget::GPU
    }

    fn try_gpu_execution_with_fallback<T>(
        &self,
        gpu_fn: impl FnOnce() -> Result<T, CudaError>,
        cpu_fallback: impl FnOnce() -> T,
    ) -> T {
        let start = std::time::Instant::now();

        match gpu_fn() {
            Ok(result) => {
                let latency = start.elapsed();
                self.performance_tracker.record_gpu_success(latency);
                result
            }
            Err(e) => {
                tracing::warn!("GPU execution failed: {:?}, falling back to CPU", e);
                self.performance_tracker.record_gpu_failure();

                let result = cpu_fallback();
                let latency = start.elapsed();
                self.performance_tracker.record_cpu_fallback(latency);

                result
            }
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum ExecutionTarget {
    CPU,
    GPU,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Operation {
    CosineSimilarity,
    ActivationSpreading,
    HnswSearch,
}

pub struct ExecutorCapabilities {
    pub gpu_available: bool,
    pub gpu_device_name: Option<String>,
    pub cpu_simd_level: CpuCapability,
}
```

### Performance Tracker

```rust
// engram-core/src/compute/cuda/performance_tracker.rs

use std::collections::{HashMap, VecDeque};
use std::sync::Mutex;
use std::time::Duration;
use super::hybrid::Operation;

pub struct PerformanceTracker {
    /// Ring buffer of recent CPU latencies per operation
    cpu_latencies: Mutex<HashMap<Operation, VecDeque<Duration>>>,

    /// Ring buffer of recent GPU latencies per operation
    gpu_latencies: Mutex<HashMap<Operation, VecDeque<Duration>>>,

    /// Count of GPU failures per operation
    gpu_failures: Mutex<HashMap<Operation, usize>>,

    /// Count of successful GPU executions per operation
    gpu_successes: Mutex<HashMap<Operation, usize>>,

    /// Maximum window size for moving averages
    window_size: usize,
}

impl PerformanceTracker {
    pub fn new(window_size: usize) -> Self {
        Self {
            cpu_latencies: Mutex::new(HashMap::new()),
            gpu_latencies: Mutex::new(HashMap::new()),
            gpu_failures: Mutex::new(HashMap::new()),
            gpu_successes: Mutex::new(HashMap::new()),
            window_size,
        }
    }

    pub fn record_cpu_latency(&self, operation: Operation, latency: Duration) {
        let mut latencies = self.cpu_latencies.lock().unwrap();
        let queue = latencies.entry(operation).or_insert_with(VecDeque::new);

        if queue.len() >= self.window_size {
            queue.pop_front();
        }
        queue.push_back(latency);
    }

    pub fn record_gpu_success(&self, latency: Duration) {
        // Similar to record_cpu_latency
    }

    pub fn record_gpu_failure(&self) {
        // Increment failure counter
    }

    pub fn gpu_speedup(&self, operation: Operation) -> f64 {
        let cpu_avg = self.average_cpu_latency(operation);
        let gpu_avg = self.average_gpu_latency(operation);

        if gpu_avg.is_zero() {
            return 0.0; // No GPU data yet
        }

        cpu_avg.as_secs_f64() / gpu_avg.as_secs_f64()
    }

    pub fn gpu_success_rate(&self, operation: Operation) -> f64 {
        let failures = self.gpu_failures.lock().unwrap();
        let successes = self.gpu_successes.lock().unwrap();

        let fail_count = failures.get(&operation).copied().unwrap_or(0);
        let success_count = successes.get(&operation).copied().unwrap_or(0);
        let total = fail_count + success_count;

        if total == 0 {
            return 1.0; // No data, assume success
        }

        success_count as f64 / total as f64
    }

    fn average_cpu_latency(&self, operation: Operation) -> Duration {
        let latencies = self.cpu_latencies.lock().unwrap();
        let queue = match latencies.get(&operation) {
            Some(q) if !q.is_empty() => q,
            _ => return Duration::from_secs(0),
        };

        let sum: Duration = queue.iter().sum();
        sum / queue.len() as u32
    }

    fn average_gpu_latency(&self, operation: Operation) -> Duration {
        // Similar to average_cpu_latency
    }
}
```

## Acceptance Criteria

1. **Dispatch Logic**
   - [ ] Small batches (<64) routed to CPU automatically
   - [ ] Large batches routed to GPU when available
   - [ ] Historical performance influences dispatch decisions
   - [ ] Configuration overrides work correctly

2. **CPU Fallback**
   - [ ] GPU failures don't crash, fall back to CPU
   - [ ] Results identical whether from CPU or GPU
   - [ ] Fallback logged for monitoring
   - [ ] Performance tracker records fallback events

3. **Performance Tracking**
   - [ ] Accurately tracks CPU and GPU latencies
   - [ ] Computes speedup correctly
   - [ ] Success rate reflects actual GPU reliability
   - [ ] Moving averages smooth out noise

4. **Integration**
   - [ ] Works with existing `VectorOps` trait
   - [ ] Integrates with `BatchEngine`
   - [ ] Compatible with `ParallelSpreadingEngine`
   - [ ] No breaking changes to public APIs

## Integration Points

### Existing Code to Modify

1. **`engram-core/src/compute/dispatch.rs`**
   - Replace direct GPU calls with `HybridExecutor`
   - Maintain existing `VectorOps` API compatibility

2. **`engram-core/src/batch/engine.rs`**
   - Use `HybridExecutor` for batch operations
   - Route through hybrid dispatch instead of direct GPU

3. **`engram-core/src/activation/parallel.rs`**
   - Integrate `HybridExecutor` for activation spreading
   - Use performance tracking to optimize dispatch

## Testing Approach

### Test 1: Dispatch Decision Logic

```rust
#[test]
fn test_dispatch_decision_logic() {
    let config = HybridConfig {
        gpu_min_batch_size: 64,
        gpu_speedup_threshold: 1.5,
        ..Default::default()
    };
    let executor = HybridExecutor::new(config);

    // Small batch should use CPU
    let decision = executor.make_dispatch_decision(
        Operation::CosineSimilarity,
        32, // < 64
    );
    assert_eq!(decision, ExecutionTarget::CPU);

    // Large batch should use GPU (if available)
    let decision = executor.make_dispatch_decision(
        Operation::CosineSimilarity,
        1024,
    );
    if executor.is_gpu_available() {
        assert_eq!(decision, ExecutionTarget::GPU);
    } else {
        assert_eq!(decision, ExecutionTarget::CPU);
    }
}
```

### Test 2: CPU Fallback on GPU Failure

```rust
#[test]
fn test_gpu_failure_fallback() {
    let executor = HybridExecutor::new(Default::default());
    let query = random_vector_768();
    let targets = random_vectors_768(1000);

    // Mock GPU failure
    let gpu_fn = || Err(CudaError::OutOfMemory);
    let cpu_fn = || vec![0.5f32; 1000];

    let result = executor.try_gpu_execution_with_fallback(gpu_fn, cpu_fn);

    assert_eq!(result.len(), 1000);
    // Should have recorded failure
    assert!(executor.performance_tracker.gpu_failure_count() > 0);
}
```

### Test 3: Performance Tracking Accuracy

```rust
#[test]
fn test_performance_tracking() {
    let tracker = PerformanceTracker::new(100);

    // Record some measurements
    tracker.record_cpu_latency(
        Operation::CosineSimilarity,
        Duration::from_micros(100),
    );
    tracker.record_gpu_success(Duration::from_micros(20));

    let speedup = tracker.gpu_speedup(Operation::CosineSimilarity);
    assert!((speedup - 5.0).abs() < 0.1); // 100us / 20us = 5x

    let success_rate = tracker.gpu_success_rate(Operation::CosineSimilarity);
    assert!((success_rate - 1.0).abs() < 0.01); // No failures yet
}
```

### Test 4: Integration with Existing APIs

```rust
#[test]
fn test_api_compatibility() {
    let executor = HybridExecutor::new(Default::default());
    let query = random_vector_768();
    let targets = random_vectors_768(1000);

    // Should work identically to direct CPU call
    let hybrid_result = executor.execute_batch_cosine_similarity(&query, &targets);
    let cpu_result = get_vector_ops().cosine_similarity_batch_768(&query, &targets);

    for (h, c) in hybrid_result.iter().zip(cpu_result.iter()) {
        assert!((h - c).abs() < 1e-6);
    }
}
```

## Files to Create/Modify

### New Files
- `/Users/jordanwashburn/Workspace/orchard9/engram/engram-core/src/compute/cuda/hybrid.rs`
- `/Users/jordanwashburn/Workspace/orchard9/engram/engram-core/src/compute/cuda/performance_tracker.rs`
- `/Users/jordanwashburn/Workspace/orchard9/engram/engram-core/tests/hybrid_executor.rs`

### Modified Files
- `/Users/jordanwashburn/Workspace/orchard9/engram/engram-core/src/compute/dispatch.rs`
- `/Users/jordanwashburn/Workspace/orchard9/engram/engram-core/src/batch/engine.rs`
- `/Users/jordanwashburn/Workspace/orchard9/engram/engram-core/src/activation/parallel.rs`
- `/Users/jordanwashburn/Workspace/orchard9/engram/engram-core/src/compute/cuda/mod.rs`

## Dependencies

**Blocking**:
- Task 003 (Cosine Similarity Kernel) - provides GPU implementation
- Task 005 (Activation Spreading Kernel) - provides GPU implementation
- Task 006 (HNSW Kernel) - provides GPU implementation

**Blocked By This Task**:
- Task 008 (Multi-Hardware Testing) - needs hybrid executor
- Task 009 (OOM Handling) - builds on hybrid executor
- Task 012 (Integration Testing) - validates hybrid executor

## Risk Assessment

### Risk: Dispatch Overhead

**Probability**: LOW
**Impact**: MEDIUM
**Description**: Decision logic might add latency to hot path

**Mitigation**:
- Cache dispatch decisions for repeated batch sizes
- Inline critical path decision code
- Profile to ensure overhead <1% of total latency

### Risk: Performance Tracker Memory Growth

**Probability**: LOW
**Impact**: MEDIUM
**Description**: Ring buffers might consume excessive memory

**Mitigation**:
- Bounded ring buffer size (100 samples default)
- Periodically trim old data
- Monitor memory usage in tests

### Risk: Race Conditions in Performance Tracking

**Probability**: LOW
**Impact**: HIGH
**Description**: Concurrent updates might corrupt performance data

**Mitigation**:
- Use `Mutex` for shared state
- Atomic operations for counters
- Extensive concurrency testing

## Success Metrics

1. **Dispatch Accuracy**: 95% of batches routed to optimal backend
2. **Fallback Reliability**: 0 crashes due to GPU failures
3. **Performance Overhead**: <1% latency overhead from dispatch logic
4. **API Compatibility**: 100% of existing tests pass with hybrid executor

## Notes

This is the production interface for GPU acceleration. It must be rock-solid: no crashes, no performance regressions, transparent CPU fallback.

The performance tracker enables adaptive behavior - the system learns from experience which operations benefit from GPU acceleration under actual workload conditions.

Feature flag `force_cpu_mode` is critical for debugging and performance comparisons.
