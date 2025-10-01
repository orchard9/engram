# Task 009: GPU Acceleration Foundation

## Objective
Define the GPU abstraction layer and data layout so future CUDA kernels can drop in without refactoring the spreading engine.

## Priority
P2 (Future-Proofing)

## Effort Estimate
1 day

## Dependencies
- Task 007: SIMD-Optimized Batch Spreading

## Technical Approach

### Existing Hooks
- SIMD code already batches activations via `ParallelSpreadingConfig::simd_batch_size` and `SimdActivationMapper` (Task 007). We can reuse the same AoSoA layout for GPU transfers.
- Metrics and configuration live under `activation/mod.rs` and `activation/simd_optimization.rs`.
- No GPU abstraction exists; this task introduces `GPUSpreadingInterface` and CPU fallback.

### Implementation Details
1. **Trait + Capability Model**
   - Create `engram-core/src/activation/gpu_interface.rs` with:
     ```rust
     pub struct GpuCapabilities { pub max_batch: usize, pub unified_memory: bool, pub device_name: String }
     pub trait GPUSpreadingInterface: Send + Sync {
         fn capabilities(&self) -> &GpuCapabilities;
         fn is_available(&self) -> bool;
         fn launch(&self, batch: &GPUActivationBatch) -> GpuLaunchFuture;
     }
     ```
   - Implement `GPUSpreadingInterface` for `CpuFallback` that wraps SIMD batch spreading (`activation/parallel.rs::process_task`).

2. **Batch Container**
   - Introduce `GPUActivationBatch` adjacent to SIMD structs:
     ```rust
     #[repr(C, align(32))]
     pub struct GPUActivationBatch {
         pub source: [f32; 768],
         pub targets: Vec<[f32; 768]>,
         pub activations: Vec<f32>,
         pub confidences: Vec<f32>,
         pub meta: BatchMetadata,
     }
     ```
   - Provide builders that reuse the AoSoA buffers prepared for SIMD to avoid extra copies. Ensure `targets` memory is contiguous for `cudaMemcpy`.

3. **Adaptive Dispatch**
   - Add `AdaptiveSpreadingEngine` wrapper in `activation/parallel.rs` that holds `Option<Box<dyn GPUSpreadingInterface>>` plus `CpuFallback`.
   - Inside `process_task`, call `adaptive_engine.spread(batch)`; the adaptive layer decides GPU or CPU based on `batch.targets.len()` and `capabilities().is_available`.
   - Cache GPU availability and warm up contexts once to avoid runtime penalties.

4. **Configuration Flags**
   - Extend `ParallelSpreadingConfig` with `gpu_threshold: usize` and `enable_gpu: bool`. Default `enable_gpu = false` until CUDA implementation lands in Milestone 11.
   - Document config via Rustdoc and surface in `SpreadingConfig` so operator config files can set it.

5. **Telemetry**
   - Emit metrics (`metrics::prometheus`) for `engram_spreading_gpu_launch_total`, `engram_spreading_gpu_fallback_total`, and average transfer latency to support Task 012 dashboards.

### Acceptance Criteria
- [ ] `GPUSpreadingInterface` trait and `GpuCapabilities` struct merged
- [ ] `CpuFallback` implementation reuses SIMD code and satisfies trait
- [ ] `AdaptiveSpreadingEngine` chooses GPU/CPU based on thresholds and exposes fallbacks
- [ ] Configurable `enable_gpu`/`gpu_threshold` fields documented in `ParallelSpreadingConfig`
- [ ] Metrics emitted for GPU launches/fallbacks
- [ ] Unit tests mock GPU interface to verify dispatch paths

### Testing Approach
- Mock GPU interface returning canned results to validate dispatch without CUDA
- Integration test toggling `enable_gpu` to ensure CPU fallback remains correct
- Benchmark comparing direct SIMD path vs adaptive wrapper to confirm negligible overhead when GPU disabled

## Risk Mitigation
- **GPU unavailability** → default to CPU fallback, log warning once, metrics track fallback count
- **Interface drift** → keep GPU trait minimal (launch + capabilities) so CUDA and future HIP backends share it
- **Security** → zero buffers after launch by default; enforce in trait contract

## Notes
Relevant modules:
- SIMD batches (`engram-core/src/activation/simd_optimization.rs`)
- Parallel engine (`engram-core/src/activation/parallel.rs`)
- Metrics export (`engram-core/src/metrics/prometheus.rs`)
