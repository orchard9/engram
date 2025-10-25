# Twitter Thread: Warp-Optimized Cosine Similarity

## Tweet 1 (Hook)
We just got 7x GPU speedup on cosine similarity - the operation that eats 60% of our CPU time.

The secret: warp-level optimization, not just "throw more threads at it."

Thread on making GPUs actually fast:

## Tweet 2 (The Naive Approach)
Naive GPU kernel: one thread per vector, each computing 768-dim dot product sequentially.

Result: 3x SLOWER than CPU AVX-512.

Why? 13,000 CUDA cores sitting idle while each thread does sequential work.

Not exploiting intra-vector parallelism.

## Tweet 3 (The Warp Insight)
GPUs execute warps - groups of 32 threads in lockstep (SIMT).

Key insight: distribute the 768-dim dot product across 32 threads.

Each thread: 24 multiplies
Warp shuffle: combine partial sums

32-way parallelism within each vector comparison.

## Tweet 4 (Warp Shuffle Magic)
Traditional reduction: shared memory, 30 cycle latency
Warp shuffle: thread-to-thread exchange, 3 cycle latency

```cuda
for (int offset = 16; offset > 0; offset /= 2) {
    sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
}
```

10x faster reduction.

## Tweet 5 (Memory Coalescing)
RTX 3060: 360 GB/s bandwidth when threads access consecutive addresses.

Thread 0 reads targets[0]
Thread 1 reads targets[1]
...
Thread 31 reads targets[31]

Hardware combines into single 128-byte transaction. Perfect coalescing.

## Tweet 6 (Vectorized Loads)
float4 loads 4 floats per instruction instead of 1.

For memory-bound operations, fewer instructions = less overhead = more bandwidth.

Requirement: 16-byte alignment
Benefit: 20% throughput improvement

## Tweet 7 (Tensor Cores for Scale)
For batch sizes >= 1024, reformulate as matrix multiply and use Tensor Cores.

cuBLAS with TF32: 10-15x faster than custom kernels.

A100 result: 10,000 vectors in 800us (26x speedup over CPU)

## Tweet 8 (Call to Action)
Warp-level thinking unlocks GPU performance:

1. Distribute work across warp threads
2. Warp shuffle for low-latency reduction
3. Coalesced memory access
4. Vectorized loads
5. Tensor Cores at scale

7x speedup on the operation consuming 60% of our CPU time.

Building: https://github.com/YourOrg/engram
