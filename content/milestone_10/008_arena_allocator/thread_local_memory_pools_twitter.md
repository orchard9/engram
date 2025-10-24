# Thread-Local Memory Pools - Twitter Thread

## Thread: The Fastest Lock is No Lock

**Tweet 1/8:**

We had 32 threads running Zig kernels in parallel. Each kernel took 100μs. Perfect for concurrency, right?

Throughput collapsed under load. Not because kernels were slow, but because they were fighting over a shared memory pool.

The fix: thread-local arenas. Zero contention.

**Tweet 2/8:**

The problem: Every allocation from a shared arena needs synchronization.

```zig
mutex.lock();  // 20ns
allocate();
mutex.unlock();
```

10 allocations × 20ns = 200ns overhead per kernel.

32 threads × lock contention = microseconds of stalls.

**Tweet 3/8:**

Solution: Give each thread its own arena.

```zig
threadlocal var kernel_arena: ArenaAllocator = ...;
```

No sharing = no locks = no contention.

Each thread bumps its own pointer. Pure O(1) allocation.

**Tweet 4/8:**

The tradeoff: memory overhead.

- Shared arena: 2MB total
- Thread-local (32 threads): 64MB total

Is 64MB worth it?

Yes. Memory is cheap. Lock contention costs CPU cycles you can't get back.

**Tweet 5/8:**

Measured impact:

Single thread: 10,000 ops/sec
32 threads (shared arena): 12,000 ops/sec (lock contention kills scaling)
32 threads (thread-local): 320,000 ops/sec (linear scaling)

27x throughput improvement.

**Tweet 6/8:**

How to size arenas? Track high water marks.

```zig
pub fn getStats() ArenaStats {
    return .{
        .capacity = buffer.len,
        .high_water_mark = max_offset_reached,
        .overflow_count = allocation_failures,
    };
}
```

Healthy arena: 50-80% utilization, <1% overflow.

**Tweet 7/8:**

Testing thread isolation:

Spawn 32 threads running kernels simultaneously. If arenas interfere, you'd see:
- Data corruption
- Lock contention
- Crashes

We saw: Linear scaling, zero contention, no corruption.

Thread-local works.

**Tweet 8/8:**

Pattern for concurrent hot paths:

1. Profile under load (find contention)
2. Verify allocations are short-lived
3. Replace shared allocator with thread-local arenas
4. Validate scaling

The fastest lock is the lock you never take.

---

## Thread: Arena Allocation - Simple is Fast

**Tweet 1/5:**

Arena allocation: The simplest allocator that could possibly work.

```zig
fn alloc(size: usize) []u8 {
    ptr = buffer[offset..][0..size];
    offset += size;  // Bump pointer
    return ptr;
}

fn reset() void {
    offset = 0;  // Free everything
}
```

O(1) alloc, O(1) free. No metadata, no free lists.

**Tweet 2/5:**

Perfect for short-lived scratch space:

1. Kernel starts → alloc buffers
2. Kernel computes → use buffers
3. Kernel done → reset arena

All allocations discarded in one instruction. No per-allocation bookkeeping.

**Tweet 3/5:**

Compare to malloc:
- Metadata: 16 bytes per allocation
- Free lists: Pointer chasing, cache misses
- Fragmentation: Mixed with other allocations

Arena: Sequential memory, sequential access, perfect for prefetchers.

**Tweet 4/5:**

Measured cache impact:

malloc: 91% L1 hit rate
arena: 98% L1 hit rate

That 7% improvement = 10-15% runtime reduction.

Locality matters.

**Tweet 5/5:**

When NOT to use arenas:
- Long-lived allocations (can't free individually)
- Unpredictable sizes (hard to size buffer)
- Cross-thread sharing (thread-local is isolated)

For Zig kernels? Perfect fit.
