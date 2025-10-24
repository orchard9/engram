# malloc() Is Slow: Arena Allocation for Performance Kernels

Your performance kernel is fast. SIMD vectorization, cache-friendly access patterns, tight inner loops. You've done everything right.

But profiling shows malloc() consuming 15% of your runtime.

Here's the problem: general-purpose allocators are designed for safety and flexibility, not speed. They handle arbitrary allocation patterns, thread contention, and fragmentation. That generality costs performance.

For hot-path kernels with predictable allocation patterns, you can do better: arena allocation.

## The malloc() Tax

Every malloc() call does expensive work:
- Search free lists for appropriately-sized block
- Acquire global lock (thread contention)
- Update metadata (size, alignment, freelist pointers)
- Handle fragmentation (coalescing, splitting blocks)

For a single allocation, this might cost 50-100ns. But when your kernel allocates temporary buffers thousands of times per second, it adds up:

```
1000 queries/sec × 3 allocations/query × 75ns/allocation = 225,000ns = 225μs overhead
```

That's 225μs spent on allocator bookkeeping, not actual computation.

## Arena Allocation: Bump the Pointer, Reset When Done

Arena allocators trade generality for speed:

**Key insight:** If you know allocations are short-lived (lifetime of a single kernel call), you can allocate from a contiguous buffer with a bump pointer, then reset when done.

```zig
pub const ArenaAllocator = struct {
    buffer: []u8,
    offset: usize,

    pub fn alloc(self: *ArenaAllocator, size: usize, alignment: usize) ![]u8 {
        // Align offset
        const aligned_offset = std.mem.alignForward(usize, self.offset, alignment);

        // Check if allocation fits
        if (aligned_offset + size > self.buffer.len) {
            return error.OutOfMemory;
        }

        // Bump pointer (no freelist search, no locks)
        const ptr = self.buffer[aligned_offset..][0..size];
        self.offset = aligned_offset + size;

        return ptr;
    }

    pub fn reset(self: *ArenaAllocator) void {
        self.offset = 0;  // Bulk deallocation in O(1)
    }
};
```

Cost per allocation: ~5ns (pointer arithmetic, bounds check). **15x faster than malloc().**

## Thread-Local Pools Eliminate Contention

Even better: thread-local arenas avoid all synchronization overhead.

```zig
threadlocal var kernel_arena_buffer: [1024 * 1024]u8 = undefined;  // 1MB per thread
threadlocal var kernel_arena: ArenaAllocator = undefined;

pub fn getThreadArena() *ArenaAllocator {
    if (!arena_initialized) {
        kernel_arena = ArenaAllocator.init(&kernel_arena_buffer);
        arena_initialized = true;
    }
    return &kernel_arena;
}
```

Each thread has its own 1MB scratch space. No locks, no atomic operations, no contention.

## Usage Pattern in Zig Kernels

```zig
export fn engram_vector_similarity(
    query: [*]const f32,
    candidates: [*]const f32,
    scores: [*]f32,
    query_len: usize,
    num_candidates: usize,
) void {
    const arena = allocator.getThreadArena();
    defer allocator.resetThreadArena();  // Reset when function returns

    // Allocate normalized query vector
    const normalized = arena.allocArray(f32, query_len) catch {
        // Fallback to stack allocation if arena exhausted
        var stack_buffer: [768]f32 = undefined;
        if (query_len <= 768) {
            // Use stack buffer...
            return;
        }
        // Too large, fail gracefully
        @memset(scores[0..num_candidates], 0.0);
        return;
    };

    // Use arena-allocated buffer for computation
    normalize_vector(query, normalized);
    batch_cosine_similarity(normalized, candidates, scores, num_candidates);

    // Arena reset happens automatically via defer
}
```

The defer ensures arena cleanup even if the function returns early.

## Zero-Allocation is Even Better

The fastest allocation is no allocation.

When possible, have the caller provide output buffers:

```zig
// Caller allocates output (Rust side)
let mut scores = vec![0.0_f32; num_candidates];

// Zig writes directly to caller's buffer (no allocation)
engram_vector_similarity(
    query.as_ptr(),
    candidates.as_ptr(),
    scores.as_mut_ptr(),  // Write here
    query.len(),
    num_candidates,
);
```

This is the zero-copy FFI pattern: caller allocates, callee populates. No arena needed.

Reserve arena allocation for internal temporary buffers (normalized vectors, intermediate results).

## Allocation Strategy Hierarchy

For Engram's kernels:

1. **Prefer caller-provided buffers:** Zero overhead (output scores, activation maps)
2. **Use thread-local arena for temporaries:** Fast, no locks (normalized vectors, edge lists)
3. **Fall back to stack allocation:** When arena exhausted and size is small (<1KB)
4. **Fail gracefully:** Return zeros or error codes if all else fails

This hierarchy ensures performance while maintaining robustness.

## Real Performance Impact

Before arena allocator (using malloc for temporaries):

```
vector_similarity_1000
  time:   [2.103 ms 2.117 ms 2.133 ms]
```

After arena allocator:

```
vector_similarity_1000
  time:   [1.793 ms 1.809 ms 1.826 ms]
```

**Improvement: 15% faster** (310μs saved per 1000 queries)

The allocation overhead was real. Arena allocation eliminated it.

## Memory Leak Prevention

Arena allocators are safe because reset is explicit and local:

```zig
export fn engram_kernel(...) void {
    const arena = allocator.getThreadArena();
    defer allocator.resetThreadArena();  // Always runs

    // ... computation ...

    // Arena automatically resets even if we return early
    if (error_condition) return;

    // ... more computation ...
}
```

Zig's defer runs cleanup code when the scope exits (similar to Rust's Drop, but explicit). This prevents memory leaks even if the kernel returns early.

## When Arena Allocation Isn't Appropriate

Arenas work well for:
- Short-lived allocations (lifetime of function call)
- Predictable sizes (bounded buffer requirements)
- Single-threaded or thread-local usage

Arenas don't work well for:
- Long-lived allocations (survive function return)
- Arbitrary deallocation order (need to free individual allocations)
- Shared allocations across threads (need synchronization)

For Engram's performance kernels, allocation patterns fit the arena model perfectly.

## Key Takeaways

1. **malloc() is expensive:** 50-100ns per call, adds up in hot paths
2. **Arena allocation is fast:** ~5ns per allocation, 15x faster than malloc()
3. **Thread-local pools eliminate contention:** No locks, no atomic operations
4. **Defer ensures cleanup:** Arena reset happens automatically
5. **Zero-allocation is fastest:** Caller-provided buffers avoid allocation entirely

For performance kernels with predictable lifetimes, arena allocation turns allocator overhead from 15% of runtime to <1%.

## Try It Yourself

Zig's standard library provides arena allocators:

```zig
const std = @import("std");

pub fn main() !void {
    var buffer: [1024]u8 = undefined;
    var fba = std.heap.FixedBufferAllocator.init(&buffer);
    const allocator = fba.allocator();

    // Allocate from arena
    const data = try allocator.alloc(u32, 256);
    defer allocator.free(data);  // Or just let arena scope end

    // Use data...
}
```

For performance kernels, measure before and after. If allocator overhead is significant (>5% of runtime in profiling), arena allocation pays off.
