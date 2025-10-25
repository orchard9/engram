# Twitter Thread: Zero-Copy GPU Memory

## Tweet 1 (Hook)
Traditional GPU programming: allocate on device, copy host to device, execute kernel, copy device to host, free.

CUDA Unified Memory: allocate once, access from anywhere.

Thread on zero-copy GPU programming:

## Tweet 2 (The Old Way)
Old CUDA memory management:

```c
cudaMalloc(&d_data, size);
cudaMemcpy(d_data, h_data, size, H2D);
kernel<<<...>>>(d_data);
cudaMemcpy(h_result, d_result, size, D2H);
cudaFree(d_data);
```

5 operations for what should be 1.
Error-prone. Verbose. Synchronization headaches.

## Tweet 3 (Unified Memory)
With Unified Memory:

```c
cudaMallocManaged(&data, size);
data[0] = 1.0f;  // CPU access
kernel<<<...>>>(data);  // GPU access
float result = data[0];  // CPU access again
```

One pointer. Two access points. Runtime handles migration automatically.

## Tweet 4 (Page Faults Are Real)
Under the hood: MMU triggers page fault on access, runtime migrates 64KB page, resumes execution.

Migration: 5-10us per page.

For kernel executing in 20us, page faults during execution = disaster.

Solution: prefetch.

## Tweet 5 (Prefetching is Mandatory)
Always prefetch before kernel launch:

```c
cudaMemPrefetchAsync(data, size, device, stream);
kernel<<<..., stream>>>(data);
```

Non-blocking. All migrations happen before kernel starts. Zero runtime page faults.

Performance: 95-98% of explicit copy bandwidth.

## Tweet 6 (Memory Advise Hints)
Help the runtime optimize:

- SetReadMostly: keep replicas on CPU and GPU (query vectors)
- SetPreferredLocation GPU: rarely accessed by CPU (target vectors)
- SetAccessedBy both: GPU writes, CPU reads (results)

Cuts migration overhead in half.

## Tweet 7 (Safe Rust Wrapper)
```rust
pub struct UnifiedBuffer<T> {
    ptr: *mut T,
    len: usize,
}

impl Drop for UnifiedBuffer<T> {
    fn drop(&mut self) {
        cudaFree(self.ptr);
    }
}
```

Ownership ensures cleanup. Safe slice access. No raw pointers for users.

## Tweet 8 (Call to Action)
Unified Memory: complexity hidden behind hardware.

Minimal performance cost (<5% overhead).
Massive code simplicity benefit.

Zero-copy GPU programming: fast by default, simple by design.

Building production GPU systems: https://github.com/YourOrg/engram
