# Task 009: GPU Acceleration Foundation

## Objective
Design activation spreading interfaces for future GPU acceleration without implementing full GPU kernels.

## Priority
P2 (Future-Proofing)

## Effort Estimate
1 day

## Dependencies
- Task 007: SIMD-Optimized Batch Spreading

## Technical Approach

### Implementation Details
- Create `GPUSpreadingInterface` trait for future CUDA implementation
- Design memory layout compatible with unified memory
- Add CPU/GPU dispatch based on problem size and hardware availability
- Implement CPU fallback ensuring identical semantics

### Files to Create/Modify
- `engram-core/src/activation/gpu_interface.rs` - New file for GPU interfaces
- `engram-core/src/activation/mod.rs` - Export GPU interface traits
- `engram-core/src/compute/gpu.rs` - GPU dispatch logic (if not exists)

### Integration Points
- Extends SIMD batch spreading from Task 007
- Designs for future CUDA implementation in Milestone 11
- Maintains compatibility with existing CPU-only infrastructure
- Prepares for unified memory integration

## Implementation Details

### GPUSpreadingInterface Trait
```rust
pub trait GPUSpreadingInterface: Send + Sync {
    async fn spread_batch_gpu(
        &self,
        source_embeddings: &[[f32; 768]],
        target_embeddings: &[[f32; 768]],
        activations: &mut [f32],
        confidences: &mut [f32],
    ) -> Result<GPUSpreadingResults, GPUError>;

    fn supports_unified_memory(&self) -> bool;
    fn max_batch_size(&self) -> usize;
    fn is_available(&self) -> bool;
}

// CPU implementation for fallback
pub struct CPUSpreadingFallback;
impl GPUSpreadingInterface for CPUSpreadingFallback { /* ... */ }

// Future CUDA implementation
pub struct CUDASpreadingEngine; // To be implemented in Milestone 11
```

### Memory Layout Design
```rust
// GPU-friendly memory layout
#[repr(C, align(32))]
pub struct GPUActivationBatch {
    embeddings: Vec<[f32; 768]>,     // Aligned for GPU transfer
    activations: Vec<f32>,           // Contiguous activation array
    confidences: Vec<f32>,           // Contiguous confidence array
    batch_metadata: BatchMetadata,
}

pub struct BatchMetadata {
    batch_size: u32,
    source_count: u32,
    target_count: u32,
    padding: u32,  // Ensure GPU alignment
}
```

### CPU/GPU Dispatch Logic
```rust
pub struct AdaptiveSpreadingEngine {
    gpu_interface: Option<Box<dyn GPUSpreadingInterface>>,
    cpu_fallback: CPUSpreadingFallback,
    gpu_threshold: usize,  // Minimum batch size for GPU
}

impl AdaptiveSpreadingEngine {
    pub async fn spread_adaptive(
        &self,
        batch: GPUActivationBatch,
    ) -> SpreadingResults {
        if let Some(gpu) = &self.gpu_interface {
            if batch.batch_size >= self.gpu_threshold && gpu.is_available() {
                match gpu.spread_batch_gpu(&batch).await {
                    Ok(results) => return results,
                    Err(_) => {
                        // Fall back to CPU on GPU error
                        warn!("GPU spreading failed, falling back to CPU");
                    }
                }
            }
        }

        // CPU fallback path
        self.cpu_fallback.spread_batch_gpu(&batch).await
    }
}
```

### Future CUDA Design Considerations
- **Memory Management**: Design for pinned/unified memory
- **Kernel Launch**: Prepare grid/block dimensions for embedding operations
- **Stream Management**: Async GPU execution with CPU overlap
- **Error Handling**: Graceful GPU error recovery

## Acceptance Criteria
- [ ] `GPUSpreadingInterface` trait properly defined for future implementation
- [ ] Memory layout optimized for GPU transfer and computation
- [ ] CPU/GPU dispatch logic correctly selects implementation
- [ ] CPU fallback produces identical results to direct CPU implementation
- [ ] No performance regression when GPU hardware unavailable
- [ ] Memory alignment requirements properly enforced
- [ ] Interface design validated against CUDA programming model

## Testing Approach
- Unit tests for CPU fallback implementation
- Memory layout tests ensuring proper alignment
- Dispatch logic tests with various batch sizes
- Performance tests ensuring no CPU regression
- Interface design review against CUDA best practices

## Risk Mitigation
- **Risk**: GPU interface design incompatible with actual CUDA implementation
- **Mitigation**: Review with GPU experts, validate against CUDA samples
- **Testing**: Prototype simple CUDA kernel to validate interface

- **Risk**: Memory layout inefficient for GPU operations
- **Mitigation**: Research GPU memory access patterns, align with CUDA guidelines
- **Validation**: Measure memory transfer overhead in prototype

- **Risk**: CPU/GPU dispatch overhead negates performance benefits
- **Mitigation**: Minimize dispatch logic, cache GPU availability
- **Testing**: Benchmark dispatch overhead vs batch processing time

## Implementation Strategy

### Phase 1: Interface Definition
- Define `GPUSpreadingInterface` trait
- Implement CPU fallback with identical semantics
- Basic dispatch logic

### Phase 2: Memory Layout Optimization
- Design GPU-friendly data structures
- Implement memory alignment requirements
- Test memory transfer patterns

### Phase 3: Validation and Documentation
- Validate interface against CUDA programming model
- Document GPU preparation for Milestone 11
- Performance baseline for future comparison

## Future Implementation Notes

### Milestone 11 CUDA Implementation
```rust
impl GPUSpreadingInterface for CUDASpreadingEngine {
    async fn spread_batch_gpu(&self, batch: &GPUActivationBatch) -> Result<GPUSpreadingResults> {
        // 1. Transfer embeddings to GPU memory
        // 2. Launch similarity computation kernel
        // 3. Launch activation propagation kernel
        // 4. Transfer results back to CPU
        // 5. Return GPU spreading results
    }
}
```

### Performance Expectations
- **Target Speedup**: 5-10x for large batches (>256 vectors)
- **Overhead Threshold**: GPU beneficial when batch_size > 64
- **Memory Bandwidth**: Utilize >80% of GPU memory bandwidth
- **Latency**: <1ms GPU kernel execution for typical batches

## Notes
This task prepares the cognitive database for massive parallelization while maintaining the CPU-only functionality. The interface design here determines how effectively Milestone 11 can accelerate spreading operations. Unlike traditional databases that rarely benefit from GPU acceleration, cognitive spreading with thousands of vector similarities is ideally suited for GPU parallelization.