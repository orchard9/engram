# SIMD-Optimized Batch Spreading Perspectives

## Multiple Architectural Perspectives on Task 007: SIMD-Optimized Batch Spreading

### GPU-Acceleration Architect Perspective

SIMD optimization is the CPU counterpart to the GPU foundations planned in Task 009. We structure activation batches so the same data layout can later feed GPU kernels—AoSoA tiles with 32 embedding elements align with warp-sized processing. Prefetch hints keep CPU pipelines full while we prototype GPU offload APIs.

```rust
pub struct AoSoABatch<'a> {
    pub tile_dims: (usize, usize), // (tile_count, lane_width)
    pub components: &'a [f32],     // contiguous lane-major storage
}
```

By aligning CPU SIMD lanes with GPU warp lanes, we avoid redesigning data paths when migrating hot kernels to GPU.

### Rust Graph Engine Perspective

**SIMD Wrapper Traits:**
We provide a trait abstraction so the spreading engine can switch between scalar and SIMD implementations without branching in hot loops.

```rust
pub trait BatchKernel {
    fn similarity_batch(&self, source: &[f32; 768], targets: &[[f32; 768]]) -> Vec<f32>;
    fn accumulate(&self, activations: &mut [f32], deltas: &[f32]);
}
```

Implementations:
- `ScalarKernel` (baseline)
- `Avx2Kernel`
- `Avx512Kernel`
- `NeonKernel`

Each kernel compiles conditionally using `cfg(target_feature)` and runtime dispatch selects the best available kernel via CPUID.

### Systems Architecture Perspective

**Cache Discipline:**
Process batches sized to fit within 32 KB L1 cache (8 embeddings × 768 × 4 bytes = 24 KB). Use hardware prefetching hints for upcoming tiles and software prefetch for cold-tier data. Activation outputs reside in a separate contiguous buffer to avoid read-after-write hazards.

**Monitoring Hooks:**
Expose perf counters (`simd_cycles`, `simd_bytes_loaded`, `simd_speedup_ratio`) for Task 010's performance dashboard. Logging these metrics per tier helps correlate storage locality with SIMD efficiency.

### Memory Systems Perspective

SIMD batches should respect consolidation state. Hot-tier embeddings likely reside in RAM with high locality; cold-tier embeddings may require staging from SSD. The spreading scheduler can prioritize hot-tier batches for SIMD kernels while cold-tier batches stream through a smaller, scalar-friendly path to avoid loading large tiles from disk.

### Verification & Testing Perspective

**Numerical Parity Harness:**
Compare SIMD outputs with scalar baseline using `assert_ulps_eq!` with tolerance 2 ULPs. For stress cases (very small magnitudes), fall back to absolute error threshold (1e-6).

```rust
for (simd, scalar) in simd_outputs.iter().zip(scalar_outputs.iter()) {
    approx::assert_ulps_eq!(simd, scalar, max_ulps = 2);
}
```

**Performance Regression Check:**
Criterion benchmarks track throughput per batch size; CI fails if speedup dips below 2× for AVX2-capable hosts.

## Key Citations
- Intel. *Intel 64 and IA-32 Architectures Optimization Reference Manual.* (2023).
- Johnson, J., Douze, M., & Jégou, H. "Billion-scale similarity search with GPUs." *IEEE Transactions on Big Data* (2017).
