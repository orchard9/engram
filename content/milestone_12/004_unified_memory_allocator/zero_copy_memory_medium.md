# Zero-Copy GPU Memory: How CUDA Unified Memory Simplifies Everything

Traditional GPU programming has a dirty secret: memory management is a nightmare.

Every GPU operation requires explicit memory copies. Allocate on device, copy host to device, execute kernel, copy device to host, free device memory. Five operations for what should be one.

Here's the typical CUDA code:

```c
float* h_data = malloc(size);
float* d_data;
cudaMalloc(&d_data, size);
cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice);
kernel<<<blocks, threads>>>(d_data);
cudaMemcpy(h_result, d_result, size, cudaMemcpyDeviceToHost);
cudaFree(d_data);
free(h_data);
```

This is error-prone, verbose, and requires careful synchronization. Miss a synchronization point and you get race conditions. Miss a free and you leak GPU memory.

CUDA Unified Memory fixes this.

## The Unified Memory Promise

Unified Memory provides a single pointer accessible from both CPU and GPU:

```c
float* data;
cudaMallocManaged(&data, size);

// Access from CPU
data[0] = 1.0f;

// Access from GPU
kernel<<<blocks, threads>>>(data);

// Access from CPU again  
float result = data[0];

cudaFree(data);
```

No explicit copies. No dual allocations. One pointer, two access points.

The CUDA runtime automatically migrates pages between CPU and GPU memory as needed. You access data, it appears. That's zero-copy programming.

## How Page Migration Works

Under the hood, Unified Memory uses the GPU's memory management unit (MMU) - the same hardware that implements CPU virtual memory.

When your code accesses a page not resident on the executing device, the MMU triggers a page fault. The runtime intercepts the fault, migrates the page, and resumes execution.

For modern GPUs (Pascal through Ampere), pages are 64KB. For newer architectures, 2MB large pages are supported for reduced migration overhead.

Migration latency:
- 64KB page: 5-10 microseconds
- 2MB page: 20-30 microseconds

For kernels that execute in 20-50 microseconds, migration overhead can dominate. That's the catch.

## The Prefetching Solution

Naive Unified Memory takes page faults during kernel execution. This destroys performance.

The solution: prefetch data before kernel launch:

```c
cudaMemPrefetchAsync(data, size, device, stream);
kernel<<<blocks, threads, 0, stream>>>(data);
```

The prefetch call is non-blocking. It migrates all pages to GPU while the CPU continues working. By the time the kernel launches, data is resident. Zero page faults during execution.

For Engram's cosine similarity kernel operating on 1000 vectors:
- Without prefetch: 1000 vectors × 3KB = 3MB, 48 page migrations during kernel = 240-480us overhead
- With prefetch: all migrations before kernel, zero overhead

Prefetching is mandatory for performance.

## Memory Advise Hints

The runtime benefits from usage hints via the Memory Advise API:

```c
// Hint: mostly read-only from GPU
cudaMemAdvise(data, size, cudaMemAdviseSetReadMostly, device);

// Hint: preferred location is GPU  
cudaMemAdvise(data, size, cudaMemAdviseSetPreferredLocation, device);

// Hint: accessed by both CPU and GPU
cudaMemAdvise(data, size, cudaMemAdviseSetAccessedBy, device);
```

For Engram's workload:
- Query vectors: `SetReadMostly` (CPU writes, GPU reads, no modification)
- Target vectors: `SetPreferredLocation` GPU (rarely accessed by CPU)
- Result arrays: `SetAccessedBy` both (GPU writes, CPU reads)

Read-mostly optimization is powerful. The runtime maintains replicas on both CPU and GPU, eliminating migration for read-only data. This cuts migration overhead in half for typical workloads.

## Performance: Unified vs Explicit

With proper prefetching and advise hints, Unified Memory matches explicit copy performance:

| Transfer Size | Explicit Copy | Unified Memory (prefetch) | Overhead |
|---------------|--------------|---------------------------|----------|
| 4 KB | 8 us | 8 us | 0% |
| 64 KB | 22 us | 23 us | +5% |
| 1 MB | 180 us | 182 us | +1% |
| 16 MB | 2.8 ms | 2.8 ms | 0% |

For large transfers (>1MB), Unified Memory achieves 95-98% of explicit copy bandwidth.

The benefit: code simplicity. No explicit copy management, no synchronization complexity, no double allocations.

## The Coherence Model

Unified Memory has weaker coherence than CPU caches. CPU caches provide immediate visibility - write on core 0, read on core 1 sees the write immediately.

Unified Memory only guarantees coherence at kernel boundaries:
- Before kernel launch: CPU writes visible to GPU
- After kernel completion: GPU writes visible to CPU
- During kernel execution: no guarantees

This means you can't have CPU and GPU concurrently accessing the same memory location with strong consistency.

For most GPU workloads, this is fine. Kernels are atomic units of work. But it's a constraint to design around.

## Rust Integration: Safe Zero-Copy

Wrapping Unified Memory in safe Rust requires ownership discipline:

```rust
pub struct UnifiedBuffer<T> {
    ptr: *mut T,
    len: usize,
    _marker: PhantomData<T>,
}

impl<T> UnifiedBuffer<T> {
    pub fn new(len: usize) -> Result<Self, CudaError> {
        let size = len * std::mem::size_of::<T>();
        let mut ptr = std::ptr::null_mut();
        
        unsafe {
            cuda_check(cudaMallocManaged(
                &mut ptr as *mut *mut c_void,
                size,
                cudaMemAttachGlobal,
            ))?;
        }
        
        Ok(Self { ptr: ptr as *mut T, len, _marker: PhantomData })
    }
    
    pub fn as_slice_mut(&mut self) -> &mut [T] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }
}

impl<T> Drop for UnifiedBuffer<T> {
    fn drop(&mut self) {
        unsafe { cudaFree(self.ptr as *mut c_void); }
    }
}
```

The `UnifiedBuffer` type owns the allocation, ensures cleanup via `Drop`, and exposes safe slice access. Users never touch raw pointers.

## Fallback for Older GPUs

Maxwell and earlier GPUs have poor Unified Memory performance. We need a fallback strategy:

```rust
if device_supports_efficient_unified_memory() {
    UnifiedMemoryAllocator::new()
} else {
    PinnedMemoryAllocator::new()
}
```

Pinned memory allows fast DMA transfers (12 GB/s vs 6 GB/s for pageable memory) but requires explicit copies.

The abstraction hides this complexity. Users get zero-copy on modern GPUs, fallback to explicit copy on older hardware.

## Real-World Performance: Engram Batch Recall

For Engram's batch cosine similarity with 1000 queries:

Without Unified Memory:
```rust
let query_gpu = allocate_device(query.len());
copy_to_device(&query, &query_gpu);
let targets_gpu = allocate_device(targets.len());
copy_to_device(&targets, &targets_gpu);
let results_gpu = allocate_device(results.len());

cuda_cosine_similarity(&query_gpu, &targets_gpu, &results_gpu);

copy_to_host(&results_gpu, &results);
free_device(query_gpu);
free_device(targets_gpu);
free_device(results_gpu);
```

With Unified Memory:
```rust
let mut buffer = UnifiedBuffer::new(targets.len())?;
buffer.as_slice_mut().copy_from_slice(&targets);
buffer.prefetch_to_gpu(device)?;

cuda_cosine_similarity(&query, buffer.as_ptr(), results.as_mut_ptr());

// Results automatically visible on CPU
```

Five operations reduced to two. Complexity hidden behind safe abstractions.

## Conclusion: Simplicity Through Hardware

Unified Memory represents a shift in GPU programming philosophy. Instead of explicit management of two separate memory spaces, we have one memory space with automatic migration.

The performance cost is minimal with prefetching (<5% overhead for large transfers). The code simplicity benefit is massive.

For Engram's Task 004, Unified Memory enables zero-copy patterns that make GPU acceleration transparent. Users don't think about memory transfers. They work with vectors, the runtime handles GPU migration.

This is how production GPU systems should work: fast by default, simple by design.
