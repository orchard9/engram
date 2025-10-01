# GPU Acceleration Foundation: Designing the Offload Interface Before the Kernels

*Perspective: GPU-Acceleration Architect*

Ripping out CPU code and bolting on GPU kernels rarely ends well. We need a bridge first—a stable interface that lets us experiment with offload strategies without destabilizing the spreading engine. Task 009 delivers that bridge by defining `GPUSpreadingInterface`, data layouts, and dispatch heuristics that keep semantics intact whether the work runs on host or device.

## Why an Interface Now?
Spreading activation is bandwidth- and compute-heavy. GPUs promise 5×–10× speedups for large batches, but kernel development and cross-device debugging take time. By establishing the interface now, we decouple future CUDA implementation (Milestone 11) from the remaining Milestone 3 tasks. The CPU fallback implements the same trait, so the rest of the engine treats GPU and CPU paths uniformly.

## Trait Design
The trait exposes capability discovery, asynchronous launch, and health status:

```rust
pub trait GPUSpreadingInterface: Send + Sync {
    fn capabilities(&self) -> GpuCapabilities;
    fn is_available(&self) -> bool;
    fn launch(&self, batch: &GPUActivationBatch) -> GpuLaunchFuture;
}
```

`GpuCapabilities` includes fields like `max_batch_size`, `supports_unified_memory`, and `preferred_alignment`. `GpuLaunchFuture` resolves to `GPUSpreadingResults`, mirroring the CPU path output so downstream aggregators remain oblivious to execution venue.

## Data Layout Ready for CUDA
`GPUActivationBatch` packs embeddings, activations, and confidences contiguously with 32-byte alignment. We also include metadata such as `source_count`, `target_count`, and `hop_index`. Layout decisions mirror CUDA's coalesced memory access pattern: each warp reads contiguous 32-byte segments.

```rust
#[repr(C, align(32))]
pub struct GPUActivationBatch {
    pub embeddings: Box<[[f32; 768]]>,
    pub activations: Box<[f32]>,
    pub confidences: Box<[f32]>,
    pub meta: BatchMetadata,
}
```

Builders validate lengths and alignment, so runtime code never handles malformed batches.

## Adaptive Dispatch
`AdaptiveSpreadingEngine` chooses between GPU and CPU fallback:

```rust
pub async fn spread(&self, batch: &GPUActivationBatch) -> SpreadingResults {
    if self.detector.use_gpu(batch) {
        match self.gpu.as_ref().unwrap().launch(batch).await {
            Ok(results) => return results,
            Err(err) => tracing::warn!(?err, "gpu_launch_failed"),
        }
    }
    self.cpu.launch(batch).await
}
```

`detector.use_gpu` considers batch size, deterministic mode, cached GPU health, and active thresholds (default: offload when batch ≥ 64). GPU availability is cached after initial discovery to avoid repeated CUDA runtime calls.

## Observability and Safety
Even without kernels, we wire up telemetry: dispatch counts, fallback occurrences, and batch sizes. When GPU hardware is absent, metrics stay at zero but the code path remains exercised by tests using mock GPU interfaces. Security concerns—like residual sensitive data in device memory—are addressed by zeroing buffers in the CPU fallback so the GPU code will adopt the same practice.

## Preparing for Milestone 11
When CUDA kernels arrive, they will implement `GPUSpreadingInterface`. Because the trait already handles asynchronous launch and error propagation, integration becomes a matter of wiring the kernels into the interface. Meanwhile, CPU-only deployments run unchanged.

## References
- NVIDIA. *CUDA C Programming Guide, v12.* (2023).
- Johnson, J., Douze, M., & Jégou, H. "Billion-scale similarity search with GPUs." *IEEE Transactions on Big Data* (2017).
