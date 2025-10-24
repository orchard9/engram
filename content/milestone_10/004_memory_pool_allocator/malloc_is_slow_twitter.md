# Twitter Thread: malloc() Is Slow - Arena Allocation for Performance Kernels

## Tweet 1/8
Your kernel is fast. SIMD optimized, cache-friendly, tight loops.

But profiling shows malloc() consuming 15% of runtime.

Problem: malloc() is designed for safety and flexibility, not speed.

For hot-path kernels with predictable allocation patterns, you can do 15x better: arena allocation.

## Tweet 2/8
The malloc() tax:

Every call does expensive work:
- Search free lists
- Acquire global lock (thread contention)
- Update metadata
- Handle fragmentation

Cost: 50-100ns per allocation

1000 queries/sec × 3 allocations/query × 75ns = 225μs overhead on bookkeeping, not computation.

## Tweet 3/8
Arena allocation: bump the pointer, reset when done.

```zig
pub fn alloc(self: *Arena, size: usize) ![]u8 {
    const ptr = self.buffer[self.offset..][0..size];
    self.offset += size;  // Just pointer arithmetic
    return ptr;
}

pub fn reset(self: *Arena) void {
    self.offset = 0;  // Bulk deallocation in O(1)
}
```

Cost per allocation: ~5ns (15x faster than malloc)

## Tweet 4/8
Thread-local pools eliminate contention:

```zig
threadlocal var arena_buffer: [1MB]u8 = undefined;
threadlocal var arena: Arena = undefined;
```

Each thread has its own scratch space. No locks, no atomics, no contention.

Perfect for parallel kernels.

## Tweet 5/8
Usage in Zig kernels:

```zig
export fn engram_kernel(...) void {
    const arena = getThreadArena();
    defer resetThreadArena();  // Cleanup guaranteed

    const temp = arena.alloc(f32, 768) catch {
        // Fallback to stack if arena exhausted
    };

    // Use temp buffer...
    // Arena resets automatically via defer
}
```

## Tweet 6/8
Allocation strategy hierarchy:

1. Caller-provided buffers (zero overhead)
2. Thread-local arena (fast, no locks)
3. Stack allocation (arena exhausted, size < 1KB)
4. Fail gracefully (return zeros/error)

Fastest path first, fallbacks for robustness.

## Tweet 7/8
Real performance impact on Engram:

Before arena: 2.117ms
After arena: 1.809ms

15% faster (310μs saved per 1000 queries)

The allocation overhead was real. Arena eliminated it.

## Tweet 8/8
When to use arenas:

Good for:
- Short-lived allocations (lifetime of function)
- Predictable sizes
- Thread-local usage

Bad for:
- Long-lived allocations
- Arbitrary deallocation order
- Shared across threads

For performance kernels with predictable lifetimes, arenas turn 15% allocator overhead into <1%.
