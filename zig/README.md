# Zig Performance Kernels

High-performance computational kernels written in Zig for critical hot paths in Engram's graph memory operations.

## Architecture

- **allocator.zig**: Arena allocator providing O(1) bump-pointer allocation for kernel scratch space
- **ffi.zig**: C-compatible FFI exports for Rust integration
- **allocator_test.zig**: Comprehensive unit tests for arena allocator

## Building

```bash
# Build static library
zig build

# Run all tests
zig build test

# Build with optimizations
zig build -Doptimize=ReleaseFast
```

## Testing

### Unit Tests

The test suite includes comprehensive coverage of the arena allocator:

```bash
# Run all tests
zig build test

# Run with verbose output
zig build test --summary all
```

Test categories:
- Basic allocation and deallocation
- Alignment correctness for all primitive types
- Overflow detection and error handling
- Reset behavior and memory reuse
- Thread-local arena initialization and isolation
- Utilization tracking
- Stress tests with many allocations

### Expected Output

All tests should pass:
```
Test [1/40] allocator_test.allocate zero bytes... PASS
Test [2/40] allocator_test.allocate single byte... PASS
Test [3/40] allocator_test.allocate multiple times... PASS
...
Test [40/40] allocator_test.edge case: allocation at exact remaining capacity... PASS

All 40 tests passed.
```

## Integration with Rust

The Zig kernels are compiled to a static library (libengram_kernels.a) and linked with the Rust codebase via FFI.

```rust
// Rust side (when zig-kernels feature is enabled)
extern "C" {
    fn engram_vector_similarity(
        query: *const f32,
        candidates: *const f32,
        scores: *mut f32,
        query_len: usize,
        num_candidates: usize,
    );
}
```

## Performance Characteristics

### Arena Allocator

- **Allocation**: O(1) - pointer increment with alignment padding
- **Deallocation**: O(1) - bulk reset, offset = 0
- **Space overhead**: 0 bytes per allocation (no metadata)
- **Fragmentation**: None (linear allocation pattern)
- **Thread safety**: Thread-local storage eliminates contention

### Thread-Local Pools

Each thread maintains a 1MB arena:
- First allocation: Lazy initialization
- Subsequent allocations: Direct pointer bumping
- Between kernel calls: Bulk reset via defer pattern

Example usage in future kernels:
```zig
const arena = allocator_mod.getThreadArena();
defer allocator_mod.resetThreadArena();

const temp_buffer = try arena.allocArray(f32, 768);
// Use temp_buffer...
// Automatic cleanup via defer
```

## Design Rationale

### Why Arena Allocation?

1. **Hot path optimization**: Kernel invocations happen millions of times per second
2. **Predictable latency**: No malloc/free overhead or fragmentation
3. **Cache-friendly**: Sequential allocations improve locality
4. **Simple mental model**: Allocate during kernel, reset at exit

### Why Thread-Local Storage?

1. **Zero contention**: Each thread has independent arena
2. **No locking**: Thread-local access is lock-free by design
3. **NUMA-aware**: Memory allocated on local node
4. **Scalability**: Linear scaling with thread count

### Why Fixed-Size Pools?

1. **Bounded memory**: 1MB per thread, easy to reason about
2. **No hidden allocations**: Fixed buffer, no dynamic growth
3. **OOM handling**: Clear overflow detection at allocation
4. **Predictable behavior**: Same semantics on all platforms

## Debugging

### Allocation Tracking

Use `getThreadArenaUtilization()` to monitor high-water marks:

```zig
const arena = allocator_mod.getThreadArena();
defer {
    const util = allocator_mod.getThreadArenaUtilization();
    std.debug.print("Arena utilization: {d:.1}%\n", .{util * 100});
    allocator_mod.resetThreadArena();
}
```

### Overflow Investigation

If kernels hit OutOfMemory errors:

1. Check allocation sizes - are buffers larger than expected?
2. Verify reset is called between kernel invocations
3. Consider increasing arena size (currently 1MB)
4. Profile actual usage with utilization tracking

## Roadmap

- **Task 004** (CURRENT): Memory pool allocator - COMPLETE
- **Task 005**: SIMD vector similarity - will use arena for normalization buffers
- **Task 006**: Cache-optimized spreading activation - will use arena for BFS queues
- **Task 007**: Vectorized decay function - will use arena for temporary calculations
- **Task 008**: Larger arena allocator for batch operations

## References

- Zig Standard Library: std.mem.alignForward for alignment calculations
- Thread-local storage: Platform-specific TLS implementation
- Bump allocator pattern: Classic arena allocation strategy
