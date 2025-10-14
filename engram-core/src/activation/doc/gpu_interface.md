# GPUSpreadingInterface Summary

| Method | Purpose | Notes |
| --- | --- | --- |
| `capabilities(&self)` | Returns `GpuCapabilities` | Inspect max batch size, unified memory support, and device name for telemetry. |
| `is_available(&self)` | Determines whether GPU acceleration can run | Call before scheduling GPU batches. Always `false` for CPU fallback implementations. |
| `launch(&self, batch: &GPUActivationBatch)` | Submit a batch and receive an async `GpuLaunchFuture` | Batch must be contiguous (`ensure_contiguous()`) before invoking the GPU driver. |
| `warm_up(&self)` | Prepare device resources | Optional; override for backends requiring context initialisation. |
| `cleanup(&self)` | Release device resources | Called during graceful shutdown or when disabling GPU acceleration. |

Enable this trait by wiring a backend implementation (CUDA, HIP, or mock). Until Milestone 4 the GPU path is considered beta.
