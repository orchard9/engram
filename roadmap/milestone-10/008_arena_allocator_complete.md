# Task 008: Arena Allocator

**Duration:** 2 days
**Status:** Pending
**Dependencies:** 007 (Decay Function Kernel)

## Objectives

Extend the basic memory pool allocator from Task 004 to support larger working sets and multi-threaded workloads. The enhanced arena allocator provides per-thread memory pools with configurable capacity and overflow handling for production use.

1. **Configurable pool size** - Runtime-configurable arena capacity
2. **Multi-threaded isolation** - Thread-local pools prevent contention
3. **Overflow strategies** - Graceful degradation when pool exhausted
4. **Monitoring integration** - Track arena usage for capacity planning

## Dependencies

- Task 007 (Decay Function Kernel) - All kernels implemented and ready for production testing

## Deliverables

### Files to Create

1. `/zig/src/arena_config.zig` - Configuration management
   - ArenaConfig struct with size/overflow options
   - Runtime configuration from environment variables
   - Default sizing heuristics

2. `/zig/src/arena_metrics.zig` - Usage tracking
   - High-water mark tracking
   - Allocation statistics
   - Overflow event counting

3. `/tests/arena_stress.rs` - Stress testing
   - Multi-threaded allocation patterns
   - Arena exhaustion scenarios
   - Memory pressure testing

### Files to Modify

1. `/zig/src/allocator.zig` - Enhance with configuration and metrics
   - Add configuration support
   - Integrate metrics tracking
   - Implement overflow strategies

2. `/zig/src/ffi.zig` - Add arena control functions
   - engram_configure_arena: Set pool size
   - engram_arena_stats: Get usage metrics
   - engram_reset_arenas: Manual reset for testing

3. `/src/zig_kernels/mod.rs` - Expose configuration API
   - configure_arena_size
   - get_arena_stats
   - Arena statistics struct

## Acceptance Criteria

1. Thread-local arenas scale to 32+ threads without contention
2. Configurable arena sizes (1MB to 100MB) work correctly
3. Overflow strategies prevent crashes under memory pressure
4. Metrics tracking adds <1% overhead
5. Stress tests pass with high allocation rates

## Implementation Guidance

### Arena Configuration

```zig
// arena_config.zig
const std = @import("std");

pub const ArenaConfig = struct {
    /// Size of thread-local arena in bytes
    pool_size: usize,

    /// Overflow behavior
    overflow_strategy: OverflowStrategy,

    pub const OverflowStrategy = enum {
        /// Panic on overflow (development)
        panic,

        /// Return error (production)
        error_return,

        /// Fall back to system allocator (with warning)
        fallback,
    };

    pub const DEFAULT = ArenaConfig{
        .pool_size = 1024 * 1024, // 1MB
        .overflow_strategy = .error_return,
    };

    /// Load configuration from environment or use defaults
    pub fn fromEnv() ArenaConfig {
        var config = DEFAULT;

        // ENGRAM_ARENA_SIZE in bytes
        if (std.os.getenv("ENGRAM_ARENA_SIZE")) |size_str| {
            config.pool_size = std.fmt.parseInt(usize, size_str, 10) catch DEFAULT.pool_size;
        }

        // ENGRAM_ARENA_OVERFLOW: panic, error, fallback
        if (std.os.getenv("ENGRAM_ARENA_OVERFLOW")) |strategy_str| {
            config.overflow_strategy = std.meta.stringToEnum(OverflowStrategy, strategy_str)
                orelse DEFAULT.overflow_strategy;
        }

        return config;
    }
};

var global_config: ArenaConfig = ArenaConfig.DEFAULT;

pub fn setConfig(config: ArenaConfig) void {
    global_config = config;
}

pub fn getConfig() ArenaConfig {
    return global_config;
}
```

### Enhanced Arena Allocator

```zig
// allocator.zig (enhanced)
const std = @import("std");
const config = @import("arena_config.zig");
const metrics = @import("arena_metrics.zig");

pub const ArenaAllocator = struct {
    buffer: []u8,
    offset: usize,
    high_water_mark: usize,
    overflow_count: usize,

    pub fn init(buffer: []u8) ArenaAllocator {
        return .{
            .buffer = buffer,
            .offset = 0,
            .high_water_mark = 0,
            .overflow_count = 0,
        };
    }

    pub fn alloc(self: *ArenaAllocator, size: usize, alignment: usize) ![]u8 {
        const aligned_offset = std.mem.alignForward(usize, self.offset, alignment);

        if (aligned_offset + size > self.buffer.len) {
            self.overflow_count += 1;

            const strategy = config.getConfig().overflow_strategy;
            switch (strategy) {
                .panic => @panic("Arena allocator overflow"),
                .error_return => return error.OutOfMemory,
                .fallback => {
                    // Log warning
                    std.log.warn("Arena overflow, falling back to system allocator", .{});
                    // Note: Can't easily use system allocator in this context
                    return error.OutOfMemory;
                },
            }
        }

        const ptr = self.buffer[aligned_offset..][0..size];
        self.offset = aligned_offset + size;
        self.high_water_mark = @max(self.high_water_mark, self.offset);

        return ptr;
    }

    pub fn reset(self: *ArenaAllocator) void {
        // Record metrics before reset
        metrics.recordReset(self.high_water_mark, self.overflow_count);

        self.offset = 0;
        // Keep high_water_mark for diagnostics
        self.overflow_count = 0;
    }

    pub fn getStats(self: *const ArenaAllocator) ArenaStats {
        return .{
            .capacity = self.buffer.len,
            .current_usage = self.offset,
            .high_water_mark = self.high_water_mark,
            .overflow_count = self.overflow_count,
        };
    }
};

pub const ArenaStats = struct {
    capacity: usize,
    current_usage: usize,
    high_water_mark: usize,
    overflow_count: usize,
};

// Thread-local storage
threadlocal var kernel_arena_buffer: ?[]u8 = null;
threadlocal var kernel_arena: ?ArenaAllocator = null;

pub fn initThreadArena() !void {
    if (kernel_arena != null) return; // Already initialized

    const cfg = config.getConfig();
    const buffer = try std.heap.page_allocator.alloc(u8, cfg.pool_size);
    kernel_arena_buffer = buffer;
    kernel_arena = ArenaAllocator.init(buffer);
}

pub fn getThreadArena() *ArenaAllocator {
    if (kernel_arena == null) {
        initThreadArena() catch @panic("Failed to initialize thread arena");
    }
    return &kernel_arena.?;
}

pub fn resetThreadArena() void {
    if (kernel_arena) |*arena| {
        arena.reset();
    }
}

pub fn getThreadArenaStats() ?ArenaStats {
    if (kernel_arena) |*arena| {
        return arena.getStats();
    }
    return null;
}
```

### Metrics Tracking

```zig
// arena_metrics.zig
const std = @import("std");

const Metrics = struct {
    total_resets: usize,
    total_overflows: usize,
    max_high_water_mark: usize,

    const Self = @This();

    fn init() Self {
        return .{
            .total_resets = 0,
            .total_overflows = 0,
            .max_high_water_mark = 0,
        };
    }
};

var global_metrics: Metrics = Metrics.init();
var metrics_mutex: std.Thread.Mutex = .{};

pub fn recordReset(high_water_mark: usize, overflow_count: usize) void {
    metrics_mutex.lock();
    defer metrics_mutex.unlock();

    global_metrics.total_resets += 1;
    global_metrics.total_overflows += overflow_count;
    global_metrics.max_high_water_mark = @max(
        global_metrics.max_high_water_mark,
        high_water_mark,
    );
}

pub fn getGlobalMetrics() Metrics {
    metrics_mutex.lock();
    defer metrics_mutex.unlock();
    return global_metrics;
}

pub fn resetGlobalMetrics() void {
    metrics_mutex.lock();
    defer metrics_mutex.unlock();
    global_metrics = Metrics.init();
}
```

### FFI Control Functions

```zig
// ffi.zig (additions)
const arena_config = @import("arena_config.zig");
const allocator = @import("allocator.zig");
const metrics = @import("arena_metrics.zig");

export fn engram_configure_arena(
    pool_size_mb: u32,
    overflow_strategy: u8, // 0=panic, 1=error, 2=fallback
) void {
    const pool_size = @as(usize, pool_size_mb) * 1024 * 1024;
    const strategy: arena_config.ArenaConfig.OverflowStrategy = switch (overflow_strategy) {
        0 => .panic,
        1 => .error_return,
        2 => .fallback,
        else => .error_return,
    };

    arena_config.setConfig(.{
        .pool_size = pool_size,
        .overflow_strategy = strategy,
    });
}

export fn engram_arena_stats(
    total_resets: *usize,
    total_overflows: *usize,
    max_high_water_mark: *usize,
) void {
    const stats = metrics.getGlobalMetrics();
    total_resets.* = stats.total_resets;
    total_overflows.* = stats.total_overflows;
    max_high_water_mark.* = stats.max_high_water_mark;
}

export fn engram_reset_arenas() void {
    allocator.resetThreadArena();
}
```

### Rust API

```rust
// src/zig_kernels/mod.rs

#[cfg(feature = "zig-kernels")]
pub struct ArenaStats {
    pub total_resets: usize,
    pub total_overflows: usize,
    pub max_high_water_mark: usize,
}

#[cfg(feature = "zig-kernels")]
pub fn configure_arena(pool_size_mb: u32, overflow_strategy: OverflowStrategy) {
    unsafe {
        ffi::engram_configure_arena(pool_size_mb, overflow_strategy as u8);
    }
}

#[cfg(feature = "zig-kernels")]
pub fn get_arena_stats() ArenaStats {
    let mut stats = ArenaStats {
        total_resets: 0,
        total_overflows: 0,
        max_high_water_mark: 0,
    };

    unsafe {
        ffi::engram_arena_stats(
            &mut stats.total_resets,
            &mut stats.total_overflows,
            &mut stats.max_high_water_mark,
        );
    }

    stats
}

#[cfg(feature = "zig-kernels")]
pub enum OverflowStrategy {
    Panic = 0,
    ErrorReturn = 1,
    Fallback = 2,
}
```

### Stress Testing

```rust
// tests/arena_stress.rs

#[test]
#[cfg(feature = "zig-kernels")]
fn test_multithreaded_arena_isolation() {
    // Configure arenas
    configure_arena(2, OverflowStrategy::ErrorReturn); // 2MB per thread

    let handles: Vec<_> = (0..32)
        .map(|thread_id| {
            std::thread::spawn(move || {
                for iteration in 0..100 {
                    // Each thread runs kernels independently
                    let query = vec![thread_id as f32; 768];
                    let candidates = vec![vec![iteration as f32; 768]; 100];
                    let _scores = batch_cosine_similarity(&query, &candidates);
                }
            })
        })
        .collect();

    for handle in handles {
        handle.join().unwrap();
    }

    // Check for overflows
    let stats = get_arena_stats();
    assert_eq!(stats.total_overflows, 0, "Arena overflows detected");
}

#[test]
#[cfg(feature = "zig-kernels")]
fn test_arena_overflow_handling() {
    // Configure tiny arenas to force overflow
    configure_arena(1, OverflowStrategy::ErrorReturn); // 1MB

    // Try to allocate more than arena capacity
    let huge_query = vec![1.0; 1_000_000]; // ~4MB
    let huge_candidates = vec![vec![1.0; 1_000_000]; 10]; // ~40MB

    // Should handle gracefully
    let result = std::panic::catch_unwind(|| {
        batch_cosine_similarity(&huge_query, &huge_candidates)
    });

    // Verify overflow was counted
    let stats = get_arena_stats();
    assert!(stats.total_overflows > 0, "Expected overflow events");
}
```

## Testing Approach

1. **Configuration testing**
   - Verify environment variable parsing
   - Test various pool sizes
   - Validate overflow strategies

2. **Multi-threaded stress testing**
   - 32+ threads with independent arenas
   - High allocation rates
   - Verify no contention or corruption

3. **Overflow testing**
   - Force arena exhaustion
   - Verify graceful degradation
   - Check metrics accuracy

## Integration Points

- **Task 010 (Performance Regression)** - Monitor arena overhead in benchmarks
- **Task 011 (Documentation)** - Document arena tuning in operational guide

## Notes

- Consider using huge pages (mmap with MAP_HUGETLB) for large arenas
- Profile metrics overhead (should be negligible)
- Document recommended arena sizes for different workloads
- Add observability hooks for production monitoring
