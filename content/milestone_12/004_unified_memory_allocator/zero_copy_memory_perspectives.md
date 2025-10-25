# Perspectives: Zero-Copy Memory Management

## GPU-Acceleration-Architect Perspective

Unified Memory is one of CUDA's best features - when used correctly. The promise is simple: one pointer accessible from CPU and GPU. No explicit copies. No synchronization headaches.

The reality: page migration overhead can destroy performance if you're not careful. A 4KB page fault costs 5-10 microseconds. For operations that execute in 20 microseconds, migration overhead dominates.

The solution: prefetching. Always prefetch data to GPU before kernel launch. The `cudaMemPrefetchAsync` call is non-blocking and overlaps with CPU work. By the time the kernel launches, data is already resident on GPU. Zero migration overhead.

Memory advise hints are critical for performance. Query vectors are read-only from GPU perspective - use `cudaMemAdviseSetReadMostly` to keep replicas on both CPU and GPU. Target vectors are GPU-resident - use `cudaMemAdviseSetPreferredLocation` GPU. Results need CPU access - use `cudaMemAdviseSetAccessedBy` for both.

The fallback path for pre-Pascal GPUs adds complexity. Unified Memory on Maxwell performs poorly due to inefficient page migration. We need pinned memory fallback with explicit copies. This means dual code paths, but it's necessary for broad hardware support.

## Systems-Architecture-Optimizer Perspective

Unified Memory's page migration mechanism is fascinating. The hardware detects access from CPU or GPU, triggers page fault, and migrates the 64KB page (on modern GPUs). This is implemented via memory management unit (MMU) integration - same mechanism as CPU virtual memory.

The performance characteristics depend on page size. Pascal through Turing use 64KB pages. Ampere and newer support both 64KB and 2MB large pages. Large pages reduce TLB pressure and migration frequency, but increase migration latency.

For Engram's workload, we're transferring multi-megabyte arrays (1000 vectors × 768 dims × 4 bytes = 3MB). With 64KB pages, that's 48 page migrations. With prefetching, all 48 migrations happen asynchronously before kernel launch. No runtime overhead.

The coherence model is weaker than CPU caches. CPU caches provide immediate visibility across cores. Unified Memory only guarantees coherence at kernel boundaries. This means you can't have CPU and GPU simultaneously accessing the same data with strong consistency. Design around this limitation.

What's interesting is the bandwidth achieved. Unified Memory with prefetching achieves 95-98% of explicit `cudaMemcpy` bandwidth for large transfers (>1MB). For small transfers (<64KB), overhead is 20-30% due to page fault handling. This informs our batch size decisions.

## Rust-Graph-Engine-Architect Perspective

The Rust abstraction for Unified Memory needs careful thought. We want safety without runtime overhead. The `UnifiedBuffer<T>` type owns the allocation, implements `Drop` for cleanup, and exposes safe slices.

Type safety across the FFI boundary is critical. Rust's `[f32; 768]` is a contiguous block, same memory layout as C's `float[768]`. We can transmute safely. But `Vec<[f32; 768]>` has heap indirection - that doesn't cross FFI. We need `Box<[[f32; 768]]>` or better, `UnifiedBuffer<[f32; 768]>`.

The lifetime management question: who owns the memory? If we allocate with `cudaMallocManaged`, Rust owns it. But if we receive a pointer from CUDA (e.g., from another library), we don't own it. Use `ManuallyDrop` for non-owned pointers.

Error handling integrates with Rust's `Result` type. Every CUDA API call can fail. Wrap in `cuda_check()` to convert error codes to `CudaError` enum. Propagate with `?` operator. No unwinding across FFI boundary.

The zero-copy pattern simplifies the API. Instead of separate allocate/copy/kernel/copy/free steps, we have allocate/kernel/read. Users don't need to think about memory transfers. The hybrid executor handles prefetching transparently.

## Verification-Testing-Lead Perspective

Unified Memory correctness testing requires validating coherence. We write data from CPU, launch kernel that modifies it, read from CPU. Results must match expected values.

The challenge: debugging page faults. If we forget to prefetch and take page faults during kernel execution, performance degrades but correctness is maintained. How do we detect this? Profile with `nvprof` and count page faults.

Target: zero page faults during kernel execution. All migrations should happen during prefetch.

Property-based testing generates random workloads with varying access patterns:
- CPU write → GPU read (common case)
- GPU write → CPU read (results)
- CPU write → GPU write → CPU read (read-modify-write)
- Concurrent access from multiple streams (stress test)

Memory leak detection is critical. Every `cudaMallocManaged` must have matching `cudaFree`. Use CUDA memory profiler to track allocations. Rust's ownership helps - `Drop` impl ensures cleanup.

Cross-GPU testing validates Unified Memory behavior on different architectures:
- Maxwell: expect fallback to pinned memory (UM performance poor)
- Pascal: expect good UM performance with 64KB pages
- Ampere: expect excellent UM performance with demand paging
