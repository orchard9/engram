# Perspectives: Profiling GPU Bottlenecks

## GPU-Acceleration-Architect Perspective

Profiling is where GPU acceleration lives or dies. I've seen too many teams waste weeks implementing GPU kernels for operations that contribute <5% to total runtime, or worse, operations that are already faster on CPU due to launch overhead.

The critical insight is break-even batch size. Every kernel launch costs 5-20 microseconds - this is the tax you pay for GPU parallelism. For cosine similarity at 2.1 microseconds per vector on CPU versus 0.3 microseconds on GPU, you need at least 6-10 vectors just to break even with launch overhead. In practice, add safety margin: 64 vectors minimum.

Flamegraphs show the truth. When we profile Engram's batch recall with 1000 queries, cosine similarity is 60% of the wall-clock time. That's the target. Not the HNSW graph traversal at 3%, not the confidence calibration at 2%. Optimize what matters.

The Roofline model separates wishful thinking from reality. RTX 3060 has 360 GB/s memory bandwidth versus CPU's 50 GB/s - that's your ceiling for memory-bound operations like cosine similarity. You can't beat physics. The 7x theoretical speedup from bandwidth ratio is the maximum we'll achieve, and that's assuming perfect memory access patterns.

For Engram, the profiling data drives three decisions: (1) cosine similarity gets GPU implementation first because it's 60% of CPU time, (2) batch size dispatch keeps <64 vectors on CPU, (3) activation spreading is second priority at 25% CPU time. Everything else can wait.

## Systems-Architecture-Optimizer Perspective

Profiling reveals the memory hierarchy's brutality. When I see L3 cache miss rates >40% on cosine similarity workloads, I know we're DRAM-bound. At that point, CPU optimization has hit its ceiling - adding more SIMD lanes doesn't help because we're waiting on memory.

The performance counter analysis is illuminating. High `cycle_activity.stalls_mem_any` with low IPC (instructions per cycle) means the CPU is spending most of its time waiting for DRAM. This is the signature of a GPU-suitable workload - we're not compute-limited, we're bandwidth-limited.

What's interesting about Engram's workload is the data access pattern. Batch recall compares one query vector against thousands of stored vectors. That's a streaming access pattern - sequential reads through the target vector array. GPU memory systems excel at this: coalesced memory access across warp threads, prefetching, high bandwidth.

The break-even analysis has a subtle point people miss: it's not just kernel launch overhead. Memory transfer overhead matters too. With CUDA Unified Memory, we eliminate explicit transfers, but there's still page migration cost. For small batches, that migration overhead dominates. This is why the practical break-even (64 vectors) is much higher than the theoretical minimum (6 vectors).

The decision matrix approach quantifies trade-offs. ROI calculation: (CPU time percentage × theoretical speedup × frequency) / implementation effort. Cosine similarity scores highest not just because of speedup potential, but because it's the dominant operation in production workloads. That's where engineering effort pays off.

## Rust-Graph-Engine-Architect Perspective

Profiling in Rust gives us zero-cost abstractions visibility. When we profile `VectorOps::cosine_similarity_batch_768`, we can see if our trait-based dispatch has overhead. Spoiler: it doesn't. Monomorphization eliminates the abstraction at compile time.

The interesting architectural question is where GPU dispatch fits in our existing trait hierarchy. We have CPU dispatch that selects AVX-512 vs AVX2 vs NEON at runtime based on feature detection. GPU dispatch is the same pattern - runtime selection of the fastest backend.

What I like about the profiling approach is it informs the API design. We know batch sizes <64 should stay on CPU, so the hybrid executor's dispatch logic needs batch size as input. The performance tracker gives us historical data - if GPU is consistently slower than predicted, fall back to CPU automatically.

The FFI boundary is critical for profiling accuracy. We need to measure end-to-end including the Rust-to-CUDA crossing. If the FFI overhead is 5 microseconds and the kernel saves 10 microseconds, net benefit is only 5 microseconds. This affects break-even calculation.

Safety considerations: profiling must not perturb the system under test. `pprof` uses sampling at 100Hz, which introduces <2% overhead. That's acceptable. Instrumentation-based profiling can introduce 10-50% overhead - unacceptable for performance-critical measurements.

## Verification-Testing-Lead Perspective

Profiling correctness requires statistical rigor. I need 100+ samples per measurement to compute confidence intervals. Without this, we're making decisions based on noise.

The coefficient of variation metric is key: standard deviation divided by mean. If CV >5%, the measurement is too noisy to trust. This happens with small batch sizes where cache effects dominate. Solution: warm the cache before timing.

Differential testing between CPU and GPU implementations starts with profiling. We need identical workloads for both backends to validate performance predictions. If profiling predicts 7x GPU speedup but we only see 3x in practice, something's wrong - either the profiling methodology or the GPU implementation.

Profiling overhead validation is a test itself. We measure the same operation with and without profiling enabled. If profiling introduces >5% overhead, we need different profiling tools or sampling rates.

The break-even calculation needs empirical validation. We profile across batch sizes (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024) and find the actual crossover point. This validates the theoretical model and accounts for real-world variance.

For Engram's GPU acceleration, the acceptance criteria are clear: profiling must identify operations representing >70% of CPU time, theoretical speedup must match empirical results within 30%, and break-even batch sizes must be validated with production workloads.
