# Thread-Local Memory Pools: Zero Lock Contention

The fastest lock is the lock you never take.

When we profiled Engram's Zig kernels under concurrent load, we found something surprising. The kernels themselves were fast - vector similarity in 1.7 microseconds, spreading activation in 95 microseconds. But under load from 32 threads, performance collapsed. Not because the algorithms were slow, but because they were waiting for each other.

The culprit: a shared memory pool with a mutex protecting every allocation.

## The Allocation Problem

Zig kernels need scratch space. Vector similarity allocates temporary buffers for 1000-candidate comparisons (3 MB). Spreading activation needs a priority queue for graph traversal (8 KB). These aren't long-lived allocations - they live for microseconds, then get discarded.

Using Rust's global allocator works but adds overhead:
- Malloc metadata: 16 bytes per allocation
- Thread synchronization: atomics or locks
- Fragmentation: allocations mixed with other heap activity
- Cache pollution: scattered memory regions

For a kernel executing in 100 microseconds, this overhead is unacceptable.

## Arena Allocation: The Simple Solution

Arena allocators use bump-pointer allocation from a contiguous buffer:

```zig
pub const ArenaAllocator = struct {
    buffer: []u8,
    offset: usize,

    pub fn alloc(self: *ArenaAllocator, size: usize) ![]u8 {
        if (self.offset + size > self.buffer.len) {
            return error.OutOfMemory;
        }

        const ptr = self.buffer[self.offset..][0..size];
        self.offset += size;
        return ptr;
    }

    pub fn reset(self: *ArenaAllocator) void {
        self.offset = 0;  // Boom, everything freed
    }
};
```

Allocation is O(1): increment offset pointer. Deallocation is O(1): reset offset to zero. No free lists, no metadata, no complexity.

For kernels with short-lived scratch space, this pattern is perfect:

1. Allocate buffers for computation
2. Run kernel
3. Reset arena

Clean, fast, predictable.

## The Contention Problem

But we have 32 threads running kernels concurrently. If they share a single arena, every allocation needs synchronization:

```zig
pub fn alloc(self: *ArenaAllocator, size: usize) ![]u8 {
    self.mutex.lock();
    defer self.mutex.unlock();

    if (self.offset + size > self.buffer.len) {
        return error.OutOfMemory;
    }

    const ptr = self.buffer[self.offset..][0..size];
    self.offset += size;
    return ptr;
}
```

Mutex acquisition takes ~20 nanoseconds on modern hardware. For a kernel that allocates 10 buffers, that's 200 nanoseconds of pure overhead.

When 32 threads hit the same mutex, cache line bouncing and lock contention amplify the problem. What should take 200 nanoseconds becomes microseconds.

We measured the impact: throughput dropped 40% under 32-thread load.

## Thread-Local Arenas: Zero Contention

The solution is embarrassingly simple: give each thread its own arena.

```zig
threadlocal var kernel_arena: ?ArenaAllocator = null;
threadlocal var kernel_arena_buffer: ?[]u8 = null;

pub fn getThreadArena() *ArenaAllocator {
    if (kernel_arena == null) {
        const buffer = std.heap.page_allocator.alloc(u8, 2 * 1024 * 1024)
            catch @panic("Failed to allocate thread arena");
        kernel_arena_buffer = buffer;
        kernel_arena = ArenaAllocator.init(buffer);
    }
    return &kernel_arena.?;
}
```

Now each thread allocates from its own buffer. No locks, no atomics, no contention.

Measured impact: 32-thread throughput matches single-thread throughput. Linear scaling.

## Memory Overhead vs. Performance

The tradeoff: memory consumption.

- Shared arena: 2 MB total (one buffer for all threads)
- Thread-local arenas: 64 MB total (2 MB per thread × 32 threads)

Is 64 MB worth it?

Absolutely. Memory is cheap. Lock contention is expensive.

Back-of-napkin calculation:
- 32 threads processing kernels at 10,000 ops/sec
- Each kernel allocates 10 buffers
- Shared arena: 10 allocations × 20 ns lock × 10,000 ops = 2 ms/sec of pure lock overhead
- Thread-local arenas: 0 ms/sec lock overhead

We're trading 64 MB of RAM for 2 milliseconds of CPU time per second. On any modern server, that's a steal.

## Sizing Arenas: How Much is Enough?

How big should each arena be?

Too small: Overflows trigger error paths or fallback to heap allocation
Too large: Memory waste

We instrumented arenas to track high water marks (maximum offset reached during lifetime):

| Workload | High Water Mark | Recommended Size |
|----------|----------------|------------------|
| Vector similarity (768d, 1000 candidates) | 3.1 MB | 4 MB |
| Spreading activation (1000 nodes) | 127 KB | 2 MB |
| Decay (10,000 memories) | 24 KB | 1 MB |

Default of 2 MB handles spreading and decay with headroom. Vector similarity with large candidate sets may trigger overflow.

Overflow strategy:

```zig
pub fn alloc(self: *ArenaAllocator, size: usize) ![]u8 {
    if (self.offset + size > self.buffer.len) {
        self.overflow_count += 1;
        return error.OutOfMemory;  // Caller decides fallback
    }
    // ... allocation
}
```

Rust wrapper catches OutOfMemory and falls back to heap allocation for oversized requests. Normal case (fits in arena) stays fast.

## Configuration: One Size Doesn't Fit All

Different workloads need different arena sizes. We expose runtime configuration:

```bash
export ENGRAM_ARENA_SIZE=4194304  # 4 MB in bytes
export ENGRAM_ARENA_OVERFLOW=error  # panic, error, or fallback
```

Production systems can tune based on actual workload:

1. Run with 2 MB default
2. Monitor overflow rates via metrics
3. Increase size if overflow > 1%
4. Decrease size if utilization < 30% (memory waste)

## Metrics: Observability for Allocators

How do you know if arenas are sized correctly?

Track key metrics:

```zig
pub const ArenaStats = struct {
    capacity: usize,           // Total buffer size
    current_usage: usize,      // Current offset
    high_water_mark: usize,    // Max offset ever reached
    overflow_count: usize,     // Times allocation failed
};
```

Healthy arena:
- Utilization (high_water_mark / capacity): 50-80%
- Overflow rate: < 1%

Over-provisioned arena:
- Utilization: < 30%
- Overflow rate: 0%
- Action: Decrease size to save memory

Under-provisioned arena:
- Overflow rate: > 1%
- Action: Increase size or analyze allocation patterns

We expose these metrics via FFI for Rust to query:

```rust
let stats = zig_kernels::get_arena_stats();
println!("Arena utilization: {:.1}%",
    stats.high_water_mark as f32 / stats.capacity as f32 * 100.0);
```

## Testing Thread-Local Isolation

How do you verify thread-local arenas don't interfere?

Stress test: spawn 32 threads, each running kernels simultaneously.

```rust
#[test]
fn test_multithreaded_arena_isolation() {
    let handles: Vec<_> = (0..32)
        .map(|thread_id| {
            std::thread::spawn(move || {
                for _ in 0..1000 {
                    let query = vec![thread_id as f32; 768];
                    let candidates = vec![vec![0.0; 768]; 100];
                    let _scores = batch_cosine_similarity(&query, &candidates);
                }
            })
        })
        .collect();

    for handle in handles {
        handle.join().unwrap();
    }

    let stats = get_arena_stats();
    assert_eq!(stats.total_overflows, 0);
}
```

If arenas were shared, we'd see:
1. Data corruption (allocations overlap)
2. Lock contention (throughput degradation)
3. Crashes (race conditions)

Test passes. No corruption, no contention, linear scaling.

## When Not to Use Thread-Local Arenas

Thread-local storage isn't always the answer:

1. Long-lived allocations: Arenas reset invalidates all pointers
2. Inter-thread sharing: Can't pass arena-allocated data between threads
3. Unpredictable sizes: If allocations vary wildly, sizing is hard

For Zig kernels, none of these apply:
- Allocations live microseconds (reset immediately)
- No cross-thread sharing (kernels are pure functions)
- Sizes are predictable (based on embedding dimensions)

Perfect fit.

## The Big Picture

Thread-local arenas are a pattern, not magic:

1. Identify hot allocation paths
2. Verify allocations are short-lived
3. Measure contention under concurrent load
4. Replace shared allocator with thread-local arenas
5. Validate performance improvement

For Engram's Zig kernels, this pattern eliminated lock contention entirely. 32 threads scale linearly instead of fighting over a shared mutex.

The fastest lock is still the lock you never take.
