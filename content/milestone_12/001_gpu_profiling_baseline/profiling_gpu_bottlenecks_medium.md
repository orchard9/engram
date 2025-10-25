# Before You Write a Single GPU Kernel: The Art of Performance Profiling

Let's talk about a mistake I see constantly in high-performance computing: teams rushing to GPU-accelerate their code without understanding where their CPU time actually goes. They spend weeks implementing CUDA kernels for operations that contribute 2% to total runtime, or worse, operations that are already faster on CPU.

Here's the truth: profiling is where GPU acceleration lives or dies.

## The False Promise of "Just Use the GPU"

You've probably heard it. "GPUs are 100x faster than CPUs!" Well, yes and no. A modern datacenter GPU like the A100 has 312 TFLOPS of compute versus a CPU's 2 TFLOPS. That's 150x more compute.

But here's what they don't tell you: every GPU kernel invocation costs 5-20 microseconds in launch overhead. Memory transfers between CPU and GPU can dominate your runtime. And most importantly, not all operations benefit from parallelism.

In Engram, our cognitive memory system, we profile first and optimize second. This is how we avoid wasting engineering effort.

## The Profiling Workflow: Measure Everything

Our profiling workflow for Milestone 12 starts with flamegraph analysis. We run production-representative workloads - batch recall with 1,000 queries, activation spreading across 5,000 nodes, HNSW index search through 100,000 vectors - and measure where CPU time actually goes.

The flamegraph shows us something striking: cosine similarity computation consumes 60% of wall-clock time in batch recall operations. Not the graph traversal. Not the confidence calibration. Not the HNSW search itself. The simple act of computing similarity between a query vector and thousands of stored vectors.

This is your GPU acceleration target.

## Memory-Bound vs Compute-Bound: The Critical Distinction

Understanding whether an operation is memory-bound or compute-bound determines GPU speedup potential.

Cosine similarity is archetypal memory-bound. For 768-dimensional vectors:
- We transfer 6KB of data (3KB query + 3KB target)
- We perform ~800 floating-point operations (multiplies, adds, square roots)
- Arithmetic intensity: 0.13 FLOPS per byte

This is deeply memory-bound. The RTX 3060 has 360 GB/s memory bandwidth versus the CPU's 50 GB/s. That's your theoretical speedup ceiling: 7.2x.

You can't beat physics. No amount of kernel optimization will exceed the bandwidth ratio between GPU and CPU memory systems.

## The Break-Even Batch Size: Where GPU Wins

Here's the critical calculation everyone forgets: kernel launch overhead.

Every CUDA kernel invocation costs ~10 microseconds just to launch. For cosine similarity:
- CPU: 2.1 microseconds per vector (with AVX-512)
- GPU: 0.3 microseconds per vector (RTX 3060)
- Savings per vector: 1.8 microseconds

How many vectors do we need to amortize the 10 microsecond launch cost?

```
Break_Even = Launch_Overhead / Savings_Per_Vector
Break_Even = 10 us / 1.8 us = 5.6 vectors
```

Theoretically, we break even at 6 vectors. In practice? Add safety margin for variance. We use 64 vectors as our practical break-even point.

This is why small batch operations stay on CPU. The GPU is faster per operation, but the fixed cost of kernel launch dominates for small batches.

## The Roofline Model: Your Performance Ceiling

The Roofline model is beautifully simple. It plots attainable performance against arithmetic intensity.

For RTX 3060:
- Peak compute: 13 TFLOPS
- Peak memory bandwidth: 360 GB/s
- Ridge point: 36 FLOPS/byte

Operations with arithmetic intensity below 36 FLOPS/byte are memory-bound. Cosine similarity at 0.13 FLOPS/byte? Deeply memory-bound. Our speedup is limited by bandwidth ratio (7x), not compute ratio (6.5x).

The Roofline model keeps us honest. We can't claim "100x GPU speedup" when the memory system physics won't allow it.

## The Decision Matrix: Where to Invest Engineering Effort

Profiling data drives prioritization. We calculate ROI for each candidate operation:

```
ROI = (CPU_Time_Pct × Theoretical_Speedup × Frequency) / Implementation_Effort
```

For Engram's operations:

1. **Cosine Similarity**: (60% × 7x × 1.0) / 3 days = 1.4 ROI
   - Highest CPU time percentage
   - Good theoretical speedup
   - Appears in every batch recall operation

2. **Activation Spreading**: (25% × 5x × 0.8) / 3 days = 0.33 ROI
   - Second highest CPU time
   - Moderate speedup (sparse matrix ops)
   - Frequent but not universal

3. **HNSW Search**: (10% × 6x × 0.3) / 2 days = 0.09 ROI
   - Lower CPU time percentage
   - Good speedup potential but infrequent

The decision is clear: implement cosine similarity GPU kernel first. It has the highest ROI and represents the largest opportunity for end-to-end speedup.

## Amdahl's Law: The Harsh Reality of Heterogeneous Computing

Even if we achieve 7x speedup on cosine similarity, what's the end-to-end impact?

Amdahl's Law for heterogeneous computing:

```
Overall_Speedup = 1 / (Serial_Fraction + Parallel_Fraction / Parallel_Speedup)
```

If cosine similarity is 60% of runtime and gets 7x speedup:
- Serial work: 40%
- Parallelized work: 60% / 7 = 8.6%
- Overall: 1 / (0.40 + 0.086) = 2.06x end-to-end

This is why profiling the entire system matters. GPU-accelerating a single operation gives you local speedup, but end-to-end performance is constrained by the serial portions.

## Performance Counter Deep Dive: The Memory Hierarchy's Truth

Modern CPUs expose performance counters that reveal the memory hierarchy's brutality.

When we profile cosine similarity on CPU, we see:
- L3 cache miss rate: 42%
- `cycle_activity.stalls_mem_any`: 65% of cycles
- Instructions per cycle: 0.8 (should be 2-4 for well-optimized code)

This is the signature of DRAM-bound code. The CPU spends most of its time waiting for memory. Adding more SIMD lanes won't help - we're bottlenecked on bandwidth, not compute.

This is when GPU acceleration makes sense. The GPU's 360 GB/s bandwidth versus CPU's 50 GB/s means we can keep those compute units fed.

## The Practical Workflow: From Profiling to Implementation

Our profiling workflow for Milestone 12:

1. **Flamegraph Analysis**: Identify hot functions consuming >10% CPU time each
2. **Performance Counter Analysis**: Determine memory-bound vs compute-bound
3. **Theoretical Speedup Calculation**: Roofline model + bandwidth ratios
4. **Break-Even Batch Size**: Account for kernel launch and transfer overhead
5. **ROI Calculation**: Prioritize by impact and effort
6. **Empirical Validation**: Profile across batch sizes to validate theory

The profiling data from step 1-5 predicts performance. Step 6 validates those predictions with real measurements.

For Engram, this workflow identified:
- Cosine similarity: 60% of CPU time, 7x theoretical speedup, batch size >=64
- Activation spreading: 25% of CPU time, 5x theoretical speedup, graph size >=512 nodes
- HNSW search: 10% of CPU time, 6x theoretical speedup, index size >=10,000 vectors

## Statistical Rigor: Trusting Your Measurements

Performance measurements without statistical rigor are worthless. We require:
- Minimum 100 samples per operation
- Coefficient of variation <5%
- Warm cache before timing
- Isolated CPU cores (no system noise)

If CV >5%, the measurement is too noisy. This happens with small batches where cache effects dominate. Solution: warm the cache, increase sample size, or acknowledge the variance.

We report P50, P90, P99 latencies - not just mean. The tail matters in production systems. If P99 latency is 10x the median, your "average" speedup doesn't reflect user experience.

## The GPU Acceleration Anti-Patterns

From years of profiling work, here are the patterns that waste time:

**Anti-Pattern 1: Optimizing the Wrong Thing**
Spending a week GPU-accelerating an operation that's 2% of CPU time. Even 100x speedup gives 1.98% end-to-end improvement.

**Anti-Pattern 2: Ignoring Launch Overhead**
Assuming GPU is always faster. Small batches (<64 items) are often faster on CPU due to kernel launch overhead.

**Anti-Pattern 3: Comparing Against Unoptimized CPU Code**
"Our GPU implementation is 50x faster than naive C!" Cool, now compare it to optimized SIMD code. Suddenly it's 3x, which might not justify the complexity.

**Anti-Pattern 4: Trusting Theoretical Speedup Without Validation**
Roofline model says 10x possible, so we'll achieve 10x! No. Measure actual performance. Theoretical models have assumptions that reality violates.

## The Engram Result: Data-Driven Decisions

Our profiling for Milestone 12 revealed:
- Batch recall is dominated by cosine similarity (60% CPU time)
- Break-even batch size is 64 vectors for RTX 3060
- Theoretical maximum speedup is 7x (memory bandwidth limited)
- Expected end-to-end improvement: 2.1x for batch recall operations

This drives our implementation priorities:
1. Task 003: Implement cosine similarity GPU kernel (highest ROI)
2. Task 005: Implement activation spreading kernel (second priority)
3. Task 006: Implement HNSW search kernel (third priority)

Everything else waits until these three prove valuable in production.

## Conclusion: Profile First, Optimize Second

GPU acceleration is powerful, but it's not magic. Every kernel has launch overhead, memory transfer costs, and implementation complexity.

Profiling separates high-impact optimizations from low-impact busy-work. For Engram, the profiling data is unambiguous: cosine similarity is the target. It's 60% of CPU time, has 7x theoretical speedup, and appears in every batch recall operation.

The break-even analysis keeps us honest about when GPU helps versus hurts. Batches smaller than 64 vectors stay on CPU where they're faster. Larger batches go to GPU where parallelism pays off.

The Roofline model sets realistic expectations. We can't exceed the 7x speedup from bandwidth ratios, no matter how much we optimize the kernel.

This is engineering discipline: measure, analyze, prioritize, implement, validate. The profiling data from Task 001 will guide every GPU implementation decision in Milestone 12.

Before you write a single GPU kernel, profile your code. Understand where time goes. Calculate theoretical speedup. Validate your assumptions. Then, and only then, start writing CUDA.

Your future self will thank you.
