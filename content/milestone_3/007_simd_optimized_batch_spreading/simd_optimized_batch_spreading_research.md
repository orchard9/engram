# SIMD-Optimized Batch Spreading Research

## Research Topics for Milestone 3 Task 007: SIMD-Optimized Batch Spreading

### 1. SIMD Acceleration for High-Dimensional Similarity
- AVX2/AVX-512 fused multiply-add for cosine similarity
- Arm NEON and SVE considerations for cross-platform support
- Blocked dot product kernels for 768-dimensional embeddings
- Precision trade-offs between `f32` and mixed-precision accumulation
- Use of software pipelining to overlap loads and computation

### 2. Batch Processing Patterns in Graph Activation
- Structure-of-Arrays (SoA) layout to maximize contiguous loads
- Cache-blocking strategies for batched activation updates
- Prefetching heuristics guided by reuse distance analysis
- Batching thresholds that balance latency and throughput
- Integration with tier-aware scheduling for warm/hot data

### 3. Confidence Aggregation and Numerical Stability
- SIMD reductions with compensated summation
- Clamping and saturation to maintain activation bounds
- Handling denormals and sub-normal slowdown
- Error propagation between activation and confidence vectors
- Validation tolerances for floating-point deviations

### 4. Auto-Tuning and CPU Feature Detection
- CPUID-driven detection of AVX2, AVX-512, and AMX tiles
- Runtime selection between scalar, SIMD, and mixed modes
- Auto-tuning batch sizes based on L1/L2 cache size
- Empirical modeling of throughput vs. batch size
- Fallback strategies for unsupported instruction sets

### 5. Benchmarking and Profiling SIMD Kernels
- Use of `criterion` with perf counters (via `perf_event_open`)
- Measuring instructions-per-cycle (IPC) and L1 miss rate
- Roofline analysis for bandwidth vs. compute limits
- Comparing Engram kernels against FAISS baselines
- Capturing thermal throttling effects on SIMD throughput

## Research Findings

### SIMD Kernel Design for Cosine Similarity
Cosine similarity between 768-dimensional vectors is compute-heavy but highly regular. Each vector fits in 768 × 4 bytes = 3 KB, so an 8-vector batch uses 24 KB—comfortably within L1D cache on modern CPUs. AVX2 provides 256-bit registers (8 `f32` values) and FMA instructions, yielding 16 floating-point operations per cycle per port. AVX-512 doubles lane width to 512 bits (16 `f32` values) and enables masked operations for tail handling (Intel, 2023). The kernel should unroll the dot product loop four iterations at a time to hide load latency and employ `_mm_prefetch` with `_MM_HINT_T0` for upcoming embeddings.

### Batch Layout Considerations
Switching from Array-of-Structures (AoS) to SoA or AoSoA dramatically improves SIMD efficiency. FAISS reports 2.5× speedup when transposing embedding blocks into SoA layout for vectorized distance computations (Johnson et al., 2017). Engram already stores embeddings in columnar form from Milestone 2, enabling contiguous loads. To balance CPU cache pressure, we use AoSoA with tiles of 32 elements: each tile groups 32 embeddings, and within the tile we store components contiguously.

### Confidence Aggregation Accuracy
SIMD reductions accumulate floating-point rounding error faster than scalar loops. Using pairwise reduction trees and accumulating in `f64` mitigates drift (Higham, 2002). After computing activation deltas, we clamp outputs with `_mm_max_ps` and `_mm_min_ps` to enforce activation bounds [0, 1]. Denormal numbers can degrade performance; enabling flush-to-zero (`_MM_SET_FLUSH_ZERO_MODE`) prevents stalls (Intel, 2023).

### Auto-Tuning Strategy
Auto-tuning chooses batch size by evaluating candidate sizes (8, 16, 32, 64) and measuring throughput on startup. Similar approaches in high-performance libraries, like MKL-DNN (DNNL), demonstrate that short warm-up benchmarks converge quickly (<5 ms) while offering substantial performance gains (Intel, 2020). We store CPU feature detection results in `OnceLock<CpuFeatures>` to avoid repeated CPUID calls.

### Benchmarking Results from Literature
Graph analytics workloads see 2×–4× speedups from SIMD batch processing when memory bandwidth is not the bottleneck (Zhou et al., 2019). Roofline models predict Engram's cosine kernel is compute-bound on AVX2 systems until batch size exceeds 32, after which cache misses dominate. Monitoring IPC and L1 miss rate via Linux `perf` counters provides actionable insights; we aim for >1.5 IPC during kernel execution and <10% L1 miss rate to meet the >2× speedup target.

## Key Citations
- Intel. *Intel 64 and IA-32 Architectures Optimization Reference Manual.* (2023).
- Johnson, J., Douze, M., & Jégou, H. "Billion-scale similarity search with GPUs." *IEEE Transactions on Big Data* (2017).
- Higham, N. J. *Accuracy and Stability of Numerical Algorithms.* (2002).
- Intel. *oneDNN Performance Guide.* (2020).
- Zhou, X., et al. "Accelerating graph analytics via vectorization." *IEEE TPDS* (2019).
