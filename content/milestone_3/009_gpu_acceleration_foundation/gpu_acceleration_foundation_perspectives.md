# GPU Acceleration Foundation Perspectives

## Multiple Architectural Perspectives on Task 009: GPU Acceleration Foundation

### GPU-Acceleration Architect Perspective

Designing `GPUSpreadingInterface` means committing to stable kernel boundaries: batch inputs (embeddings, activations, confidences) and metadata. We mirror CUDA stream semantics by exposing an async method that returns a future, ready for integration with Rust's async runtime.

```rust
pub trait GPUSpreadingInterface: Send + Sync {
    fn capabilities(&self) -> GpuCapabilities;
    fn launch(&self, batch: &GPUActivationBatch) -> GpuLaunchFuture;
}
```

`GpuCapabilities` includes memory limits, preferred batch size, and unified memory support. Future CUDA engines will map directly onto this trait.

### Systems Architecture Perspective

CPU/GPU dispatch belongs in an `AdaptiveSpreadingEngine`. It caches device availability, warms up contexts lazily, and records telemetry.

```rust
pub async fn spread(&self, batch: &GPUActivationBatch) -> SpreadingResults {
    if self.should_use_gpu(batch) {
        match self.gpu.launch(batch).await {
            Ok(results) => return results,
            Err(err) => tracing::warn!(?err, "gpu_launch_failed"),
        }
    }
    self.cpu_fallback.launch(batch).await
}
```

Dispatch decisions consider batch size, deterministic mode flags, and current GPU queue depth.

### Rust Graph Engine Perspective

We isolate GPU-specific concerns in `engram-core/activation/gpu_interface.rs`, keeping the main spreading engine agnostic. Batches are constructed via builders that ensure alignment and guard against host/device mismatches.

```rust
pub struct GPUActivationBatchBuilder {
    embeddings: Vec<[f32; 768]>,
    activations: Vec<f32>,
    confidences: Vec<f32>,
}
```

Builders enforce invariants (equal lengths, alignment) and provide `into_batch()` that returns an error if alignment requirements fail.

### Systems Product Planner Perspective

GPU support becomes a premium feature. The interface must degrade gracefully on installations without CUDA. Feature flags: `spreading_gpu_enabled` and `spreading_gpu_required`. Documentation highlights hardware prerequisites, configuration steps, and fallback behavior. Observability dashboards show GPU utilization alongside CPU metrics so operators judge ROI.

### Verification & Testing Perspective

Before GPU kernels exist, we implement a `CPUSpreadingFallback` that uses the SIMD kernels from Task 007. Test harnesses inject a mock GPU interface to validate dispatch logic, error handling, and telemetry.

```rust
struct MockGpu {
    pub should_fail: bool,
}
```

Integration tests cover scenarios: no GPU, GPU success, GPU failure fallback, deterministic-mode forcing CPU.

## Key Citations
- NVIDIA. *CUDA C Programming Guide, v12.* (2023).
- Johnson, J., Douze, M., & Jégou, H. "Billion-scale similarity search with GPUs." *IEEE Transactions on Big Data* (2017).
