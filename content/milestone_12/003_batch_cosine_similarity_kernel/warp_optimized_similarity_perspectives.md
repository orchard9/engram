# Perspectives: Warp-Optimized Cosine Similarity

## GPU-Acceleration-Architect Perspective

Warp-level optimization is where GPU performance lives. A naive kernel that doesn't exploit the warp abstraction leaves 10x performance on the table.

The fundamental insight: warps execute SIMT (Single Instruction Multiple Thread). All 32 threads run the same instruction simultaneously. This means we want thread cooperation, not thread independence. Distribute the 768-dimensional dot product across 32 threads, each computing 24 elements, then reduce via shuffle.

Shuffle instructions are the secret weapon. Traditional reduction uses shared memory - 30 cycle latency. Shuffle exchanges data between threads directly - 3 cycle latency. For warp-level reduction, shuffle is 10x faster. This is why modern kernels avoid shared memory when possible.

Memory coalescing is non-negotiable. When thread 0 reads `targets[0]`, thread 1 reads `targets[1]`, ..., thread 31 reads `targets[31]`, the hardware combines these into a single 128-byte memory transaction. Perfect efficiency. If threads access random addresses, each thread triggers a separate transaction - 32x bandwidth waste.

For batch sizes >= 1024, forget custom kernels. Use cuBLAS with Tensor Cores. We're not smarter than NVIDIA's library team. Their GEMM implementation achieves 90%+ of peak bandwidth through tiling, prefetching, and Tensor Core utilization. Our custom kernel matters for batch sizes < 1024 where cuBLAS overhead dominates.

## Systems-Architecture-Optimizer Perspective

The memory hierarchy determines everything. RTX 3060 has:
- L1 cache: 128 KB per SM (very fast, 30 cycles)
- L2 cache: 3 MB shared (fast, 200 cycles)
- GDDR6 VRAM: 12 GB (slow, 400 cycles)

For 768-dimensional vectors, each comparison reads 6 KB of data. This doesn't fit in L1, so we're hitting L2 or VRAM. Memory bandwidth becomes the bottleneck.

The arithmetic intensity calculation is brutal: 1600 FLOPS / 6 KB = 0.27 FLOPS/byte. The RTX 3060's ridge point (where memory and compute balance) is 36 FLOPS/byte. We're 130x below the ridge - deeply memory-bound.

This is why our theoretical speedup is 7x (bandwidth ratio) not 6.5x (compute ratio). We can't utilize the GPU's 13 TFLOPS of compute because we're starved for data.

Vectorized loads (float4) help marginally. Loading 4 floats per instruction instead of 1 reduces instruction overhead, but we're still bottlenecked on bandwidth. The benefit is 5-10%, not 4x.

Register pressure matters for occupancy. Each thread needs registers for partial sums, loop counters, addresses. If the kernel uses >64 registers per thread, occupancy drops below 50%. Use `-maxrregcount=64` compiler flag to force register limit and maintain high occupancy.

## Rust-Graph-Engine-Architect Perspective

The FFI boundary for this kernel is straightforward:
```rust
extern "C" {
    fn cuda_cosine_similarity_batch(
        query: *const f32,
        targets: *const f32,
        results: *mut f32,
        dim: usize,
        num_targets: usize,
    ) -> i32;
}
```

Type safety across the boundary requires discipline. Rust's `[f32; 768]` is a 3KB contiguous block, same as C's `float[768]`. But Rust's slice `&[f32]` includes length metadata - that doesn't cross FFI. We pass raw pointers and lengths separately.

Memory alignment is critical. Rust's `#[repr(C, align(16))]` ensures 16-byte alignment for float4 vectorization. Without this, unaligned loads incur 3-5x penalty.

The safety abstraction wraps the unsafe FFI:
```rust
pub fn cosine_similarity_batch_gpu(
    query: &[f32; 768],
    targets: &[[f32; 768]],
) -> Result<Vec<f32>, CudaError> {
    let mut results = vec![0.0f32; targets.len()];
    unsafe {
        cuda_check(cuda_cosine_similarity_batch(
            query.as_ptr(),
            targets.as_ptr() as *const f32,
            results.as_mut_ptr(),
            768,
            targets.len(),
        ))?;
    }
    Ok(results)
}
```

Error handling integrates with Rust's `Result` type. CUDA error codes convert to typed enum. Propagation via `?` operator.

## Verification-Testing-Lead Perspective

Differential testing is mandatory. We generate random vectors, compute cosine similarity on both CPU (AVX-512) and GPU, compare results. Divergence must be <1e-6.

The challenge: floating-point non-associativity. CPU computes `(a + b) + (c + d)`, GPU computes `(a + c) + (b + d)` due to different reduction order. Results differ in the least significant bits.

Solution: allow 1e-6 relative error. For cosine similarity in [-1, 1], absolute error <1e-6 is acceptable.

Property-based testing generates edge cases:
- Orthogonal vectors (similarity = 0)
- Parallel vectors (similarity = 1)
- Anti-parallel vectors (similarity = -1)
- Zero vectors (undefined, should error)
- Near-zero vectors (numerical instability)

Performance regression testing benchmarks across batch sizes:
- 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 4096, 16384

We expect break-even at batch size 64, 7x speedup at 1024, saturation at 16384 (bandwidth limited).

Correctness on multiple GPU architectures requires actual hardware testing:
- GTX 1060 (Pascal, sm_60)
- RTX 2060 (Turing, sm_75)
- RTX 3060 (Ampere, sm_86)

Each architecture has subtle differences in warp scheduling, memory coalescing, and Tensor Core availability. Testing all three validates our fat binary approach.
