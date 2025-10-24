# Task 004: Memory Pool Allocator

**Duration:** 2 days
**Status:** Pending
**Dependencies:** 003 (Differential Testing Harness)

## Objectives

Implement high-performance memory pool allocator for Zig kernels to eliminate allocation overhead in hot paths. The allocator provides bump-pointer allocation for temporary scratch space with bulk deallocation, avoiding per-allocation overhead and memory fragmentation.

1. **Arena allocator** - Bump-pointer allocator with fixed capacity
2. **Thread-local pools** - Per-thread memory pools to avoid contention
3. **Zero-copy operation** - Caller-provided buffers when possible
4. **Overflow detection** - Fail safely when pool exhausted

## Dependencies

- Task 003 (Differential Testing Harness) - Testing infrastructure available

## Deliverables

### Files to Create

1. `/zig/src/allocator.zig` - Memory pool implementation
   - Arena allocator with bump pointer
   - Thread-local storage integration
   - Overflow detection and error handling

2. `/zig/src/allocator_test.zig` - Allocator unit tests
   - Basic allocation and deallocation
   - Alignment requirements
   - Overflow behavior
   - Thread-safety validation

3. `/tests/zig_differential/allocator.rs` - Rust-side allocator tests
   - Memory leak detection (valgrind integration)
   - Stress testing with high allocation rates
   - Thread contention scenarios

### Files to Modify

1. `/zig/src/ffi.zig` - Integrate allocator with kernels
   - Initialize thread-local pools
   - Use arena allocation for temporary buffers
   - Reset pools between kernel invocations

## Acceptance Criteria

1. Arena allocator passes all unit tests
2. Thread-local pools eliminate allocation contention in benchmarks
3. Memory leak tests pass (no leaks detected by valgrind)
4. Overflow detection prevents buffer overruns
5. Allocation overhead <1% of kernel runtime

## Implementation Guidance

### Arena Allocator Design

Use bump-pointer allocation with fixed-size backing buffer:

```zig
const std = @import("std");

/// Fixed-size arena allocator for kernel scratch space
pub const ArenaAllocator = struct {
    buffer: []u8,
    offset: usize,

    pub fn init(buffer: []u8) ArenaAllocator {
        return .{
            .buffer = buffer,
            .offset = 0,
        };
    }

    /// Allocate aligned memory from arena
    pub fn alloc(self: *ArenaAllocator, size: usize, alignment: usize) ![]u8 {
        // Align offset to requested alignment
        const aligned_offset = std.mem.alignForward(usize, self.offset, alignment);

        // Check if allocation fits
        if (aligned_offset + size > self.buffer.len) {
            return error.OutOfMemory;
        }

        // Bump pointer
        const ptr = self.buffer[aligned_offset..][0..size];
        self.offset = aligned_offset + size;

        return ptr;
    }

    /// Allocate typed array from arena
    pub fn allocArray(self: *ArenaAllocator, comptime T: type, n: usize) ![]T {
        const bytes = try self.alloc(n * @sizeOf(T), @alignOf(T));
        return std.mem.bytesAsSlice(T, bytes);
    }

    /// Reset arena to beginning (bulk deallocation)
    pub fn reset(self: *ArenaAllocator) void {
        self.offset = 0;
    }

    /// Get remaining capacity
    pub fn remaining(self: *ArenaAllocator) usize {
        return self.buffer.len - self.offset;
    }
};

// Thread-local arena pools
threadlocal var kernel_arena_buffer: [1024 * 1024]u8 = undefined; // 1MB per thread
threadlocal var kernel_arena: ArenaAllocator = undefined;
threadlocal var arena_initialized: bool = false;

/// Get thread-local arena allocator
pub fn getThreadArena() *ArenaAllocator {
    if (!arena_initialized) {
        kernel_arena = ArenaAllocator.init(&kernel_arena_buffer);
        arena_initialized = true;
    }
    return &kernel_arena;
}

/// Reset thread-local arena (call at kernel exit)
pub fn resetThreadArena() void {
    if (arena_initialized) {
        kernel_arena.reset();
    }
}
```

### Integration with Kernels

Use arena for temporary allocations in hot paths:

```zig
// ffi.zig
const allocator = @import("allocator.zig");

export fn engram_vector_similarity(
    query: [*]const f32,
    candidates: [*]const f32,
    scores: [*]f32,
    query_len: usize,
    num_candidates: usize,
) void {
    // Get thread-local arena
    const arena = allocator.getThreadArena();
    defer allocator.resetThreadArena();

    // Allocate temporary buffer for normalized query
    const normalized_query = arena.allocArray(f32, query_len) catch {
        // Fall back to stack allocation on overflow
        var stack_buffer: [768]f32 = undefined;
        if (query_len <= 768) {
            vector_similarity_impl(
                query,
                candidates,
                scores,
                &stack_buffer,
                query_len,
                num_candidates,
            );
            return;
        }
        // Too large for stack, fail gracefully
        @memset(scores[0..num_candidates], 0.0);
        return;
    };

    // Use arena-allocated buffer
    vector_similarity_impl(
        query,
        candidates,
        scores,
        normalized_query,
        query_len,
        num_candidates,
    );
}
```

### Caller-Provided Buffers

Prefer zero-copy by having caller provide output buffers:

```zig
// Caller allocates output buffer
export fn engram_vector_similarity(
    query: [*]const f32,
    candidates: [*]const f32,
    scores: [*]f32,  // Output buffer allocated by caller
    query_len: usize,
    num_candidates: usize,
) void {
    // No allocation needed - write directly to caller's buffer
    for (0..num_candidates) |i| {
        const candidate_start = i * query_len;
        scores[i] = cosine_similarity(
            query[0..query_len],
            candidates[candidate_start..][0..query_len],
        );
    }
}
```

### Allocation Strategy

1. **Prefer caller-provided buffers** - Zero allocation overhead
2. **Use thread-local arena for temporaries** - Small working sets
3. **Fall back to stack allocation** - When arena exhausted and size is small
4. **Fail gracefully** - Return error codes or zeros on allocation failure

### Unit Tests

```zig
// allocator_test.zig
const std = @import("std");
const ArenaAllocator = @import("allocator.zig").ArenaAllocator;

test "basic allocation" {
    var buffer: [1024]u8 = undefined;
    var arena = ArenaAllocator.init(&buffer);

    const bytes = try arena.alloc(100, 1);
    try std.testing.expectEqual(@as(usize, 100), bytes.len);
    try std.testing.expectEqual(@as(usize, 100), arena.offset);
}

test "alignment" {
    var buffer: [1024]u8 = undefined;
    var arena = ArenaAllocator.init(&buffer);

    // Allocate unaligned
    _ = try arena.alloc(1, 1);

    // Allocate aligned to 8 bytes
    const aligned = try arena.alloc(8, 8);
    const addr = @intFromPtr(aligned.ptr);
    try std.testing.expectEqual(@as(usize, 0), addr % 8);
}

test "overflow detection" {
    var buffer: [100]u8 = undefined;
    var arena = ArenaAllocator.init(&buffer);

    // Fill arena
    _ = try arena.alloc(90, 1);

    // Overflow
    const result = arena.alloc(20, 1);
    try std.testing.expectError(error.OutOfMemory, result);
}

test "reset" {
    var buffer: [1024]u8 = undefined;
    var arena = ArenaAllocator.init(&buffer);

    _ = try arena.alloc(500, 1);
    try std.testing.expectEqual(@as(usize, 500), arena.offset);

    arena.reset();
    try std.testing.expectEqual(@as(usize, 0), arena.offset);

    // Can allocate again after reset
    _ = try arena.alloc(500, 1);
}

test "typed allocation" {
    var buffer: [1024]u8 = undefined;
    var arena = ArenaAllocator.init(&buffer);

    const floats = try arena.allocArray(f32, 100);
    try std.testing.expectEqual(@as(usize, 100), floats.len);

    // Check alignment
    const addr = @intFromPtr(floats.ptr);
    try std.testing.expectEqual(@as(usize, 0), addr % @alignOf(f32));
}
```

### Memory Leak Testing

Use valgrind to detect leaks in Rust integration tests:

```rust
// tests/zig_differential/allocator.rs

#[test]
#[ignore] // Only run manually with valgrind
fn test_no_memory_leaks() {
    // Run kernel repeatedly
    for _ in 0..10_000 {
        let query = vec![1.0; 768];
        let candidates = vec![1.0; 768 * 100];
        let scores = vector_similarity(&query, &candidates, 100);
        drop(scores);
    }
    // Run with: valgrind --leak-check=full cargo test --features zig-kernels test_no_memory_leaks
}

#[test]
fn test_allocation_stress() {
    // Stress test with many threads
    let handles: Vec<_> = (0..8)
        .map(|_| {
            std::thread::spawn(|| {
                for _ in 0..1000 {
                    let query = vec![1.0; 768];
                    let candidates = vec![1.0; 768 * 1000];
                    let _scores = vector_similarity(&query, &candidates, 1000);
                }
            })
        })
        .collect();

    for handle in handles {
        handle.join().unwrap();
    }
}
```

## Testing Approach

1. **Unit tests**
   - Basic allocation and deallocation
   - Alignment correctness
   - Overflow detection
   - Reset behavior

2. **Memory leak testing**
   - Run kernels under valgrind
   - Check for leaked allocations
   - Verify thread-local cleanup

3. **Stress testing**
   - High allocation rates
   - Thread contention scenarios
   - Arena exhaustion recovery

## Integration Points

- **Task 005-007 (Kernels)** - Use arena allocator for temporary buffers
- **Task 008 (Arena Allocator)** - Build on this foundation for larger pools

## Notes

- Thread-local storage may have platform-specific behavior (test on macOS, Linux)
- Consider using std.heap.ArenaAllocator from Zig stdlib as reference
- Document arena size tuning in operational guide
- Add metrics for arena usage high-water mark
