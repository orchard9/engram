// High-performance memory pool allocator for Zig kernels
// Provides bump-pointer allocation with bulk deallocation for zero overhead
//
// Design Principles:
// - O(1) allocation: Simple pointer increment
// - O(1) bulk deallocation: Reset offset to zero
// - Thread-local pools: No cross-thread contention
// - Fixed-size arenas: Predictable memory usage (1MB per thread)
// - Alignment-aware: Respects type alignment requirements
//
// Architecture:
// Each thread gets a dedicated 1MB arena for kernel scratch space.
// Allocations bump a pointer forward. After kernel execution, bulk reset
// returns the arena to empty state. No per-allocation metadata overhead.

const std = @import("std");
const arena_config = @import("arena_config.zig");
const arena_metrics = @import("arena_metrics.zig");

/// Fixed-size arena allocator for kernel scratch space
///
/// Uses bump-pointer allocation: maintains a single offset into a backing buffer.
/// Allocations increment the offset. Reset clears all allocations at once.
///
/// Performance characteristics:
/// - alloc(): O(1) - pointer increment + alignment padding
/// - allocArray(): O(1) - typed allocation wrapper
/// - reset(): O(1) - offset = 0
/// - Space overhead: 0 bytes per allocation (no metadata)
///
/// Thread safety: Not thread-safe. Use thread-local instances.
pub const ArenaAllocator = struct {
    buffer: []u8,
    offset: usize,
    high_water_mark: usize,
    overflow_count: usize,

    /// Initialize arena with backing buffer
    ///
    /// The buffer must remain valid for the lifetime of the arena.
    /// Typically uses thread-local storage for the buffer.
    pub fn init(buffer: []u8) ArenaAllocator {
        return .{
            .buffer = buffer,
            .offset = 0,
            .high_water_mark = 0,
            .overflow_count = 0,
        };
    }

    /// Allocate aligned memory from arena
    ///
    /// Returns a slice of exactly `size` bytes aligned to `alignment`.
    /// The alignment must be a power of 2.
    ///
    /// Returns error.OutOfMemory if allocation would exceed arena capacity.
    ///
    /// Time complexity: O(1)
    /// Space overhead: Only alignment padding (at most alignment-1 bytes)
    pub fn alloc(self: *ArenaAllocator, size: usize, alignment: usize) ![]u8 {
        // Align offset to requested alignment
        // alignForward rounds up to next multiple of alignment
        const aligned_offset = std.mem.alignForward(usize, self.offset, alignment);

        // Check if allocation fits in remaining space
        if (aligned_offset + size > self.buffer.len) {
            // Record overflow event
            self.overflow_count += 1;

            // Handle according to configured strategy
            const config = arena_config.getConfig();
            switch (config.overflow_strategy) {
                .panic => @panic("Arena allocator overflow"),
                .error_return => return error.OutOfMemory,
                .fallback => {
                    // Log warning (future: could fall back to system allocator)
                    std.log.warn("Arena overflow detected (size={}, remaining={})", .{ size, self.buffer.len - self.offset });
                    return error.OutOfMemory;
                },
            }
        }

        // Bump pointer and return slice
        const ptr = self.buffer[aligned_offset..][0..size];
        self.offset = aligned_offset + size;

        // Track high-water mark
        self.high_water_mark = @max(self.high_water_mark, self.offset);

        return ptr;
    }

    /// Allocate typed array from arena
    ///
    /// Convenience wrapper that handles sizing and alignment for type T.
    /// Returns a slice of exactly `n` elements.
    ///
    /// Example:
    ///   const floats = try arena.allocArray(f32, 100);
    ///   floats[0] = 3.14;
    ///
    /// Returns error.OutOfMemory if allocation would exceed arena capacity.
    pub fn allocArray(self: *ArenaAllocator, comptime T: type, n: usize) ![]T {
        const bytes = try self.alloc(n * @sizeOf(T), @alignOf(T));
        return std.mem.bytesAsSlice(T, bytes);
    }

    /// Reset arena to beginning (bulk deallocation)
    ///
    /// Invalidates all previous allocations from this arena.
    /// Does not zero memory - callers should not rely on cleared state.
    ///
    /// Records metrics before reset for global aggregation.
    ///
    /// Time complexity: O(1)
    pub fn reset(self: *ArenaAllocator) void {
        // Record metrics before reset
        arena_metrics.recordReset(self.high_water_mark, self.overflow_count);

        // Reset allocation state
        self.offset = 0;
        // Keep high_water_mark for local diagnostics
        self.overflow_count = 0;
    }

    /// Get remaining capacity in arena
    ///
    /// Returns number of bytes available before next allocation would fail.
    /// Does not account for alignment padding.
    pub fn remaining(self: *ArenaAllocator) usize {
        return self.buffer.len - self.offset;
    }

    /// Get current utilization as fraction [0.0, 1.0]
    ///
    /// Useful for monitoring arena high-water marks.
    pub fn utilization(self: *ArenaAllocator) f32 {
        return @as(f32, @floatFromInt(self.offset)) / @as(f32, @floatFromInt(self.buffer.len));
    }

    /// Get current arena statistics
    ///
    /// Returns snapshot of arena state including capacity, usage, and overflow counts.
    pub fn getStats(self: *const ArenaAllocator) ArenaStats {
        return .{
            .capacity = self.buffer.len,
            .current_usage = self.offset,
            .high_water_mark = self.high_water_mark,
            .overflow_count = self.overflow_count,
        };
    }
};

/// Arena statistics snapshot
///
/// Provides diagnostic view of arena state for monitoring and capacity planning.
pub const ArenaStats = struct {
    /// Total arena capacity in bytes
    capacity: usize,

    /// Current offset (bytes allocated since last reset)
    current_usage: usize,

    /// Peak usage since arena creation
    high_water_mark: usize,

    /// Number of overflow events since last reset
    overflow_count: usize,
};

// Thread-local arena pools (configurable size per thread)
//
// Each thread gets its own arena to eliminate contention.
// The arena persists across kernel invocations - reset between calls.
//
// Configuration:
// - Pool size determined by arena_config.getConfig().pool_size
// - Allocated from page allocator on first use
// - Persists for lifetime of thread
//
// Platform notes:
// - Linux/macOS: __thread TLS (fast, direct CPU register access)
// - Windows: Zig maps threadlocal to platform-specific TLS
threadlocal var kernel_arena_buffer: ?[]u8 = null;
threadlocal var kernel_arena: ?ArenaAllocator = null;

/// Initialize thread-local arena with configured pool size
///
/// Allocates backing buffer from page allocator on first call per thread.
/// Subsequent calls are no-op.
///
/// Thread safety: Thread-local - safe to call from multiple threads.
///
/// Returns error.OutOfMemory if page allocation fails.
fn initThreadArena() !void {
    if (kernel_arena != null) return; // Already initialized

    const config = arena_config.getConfig();
    const buffer = try std.heap.page_allocator.alloc(u8, config.pool_size);
    kernel_arena_buffer = buffer;
    kernel_arena = ArenaAllocator.init(buffer);
}

/// Get thread-local arena allocator
///
/// Lazily initializes arena on first access per thread.
/// Returns the same arena instance for all calls within a thread.
///
/// Thread safety: Thread-local - each thread has independent arena.
/// Safe to call from multiple threads simultaneously.
///
/// Example:
///   const arena = allocator.getThreadArena();
///   const buffer = try arena.allocArray(f32, 768);
///   defer allocator.resetThreadArena();
pub fn getThreadArena() *ArenaAllocator {
    if (kernel_arena == null) {
        initThreadArena() catch @panic("Failed to initialize thread arena");
    }
    return &kernel_arena.?;
}

/// Reset thread-local arena (call at kernel exit)
///
/// Bulk-deallocates all allocations from the thread's arena.
/// Should be called after each kernel invocation to reclaim scratch space.
///
/// Safe to call even if arena was never initialized (no-op).
///
/// Thread safety: Thread-local - only affects calling thread's arena.
pub fn resetThreadArena() void {
    if (kernel_arena) |*arena| {
        arena.reset();
    }
}

/// Get thread-local arena statistics
///
/// Returns snapshot of current arena state.
/// Useful for diagnostics and capacity planning.
///
/// Returns null if arena never initialized.
pub fn getThreadArenaStats() ?ArenaStats {
    if (kernel_arena) |*arena| {
        return arena.getStats();
    }
    return null;
}

/// Get thread-local arena high-water mark
///
/// Returns peak utilization of arena since last reset.
/// Useful for tuning arena size based on actual workload.
///
/// Returns 0.0 if arena never initialized.
pub fn getThreadArenaUtilization() f32 {
    if (kernel_arena) |*arena| {
        return arena.utilization();
    }
    return 0.0;
}

// Unit tests
test "ArenaAllocator basic allocation" {
    var buffer: [1024]u8 = undefined;
    var arena = ArenaAllocator.init(&buffer);

    const bytes = try arena.alloc(100, 1);
    try std.testing.expectEqual(@as(usize, 100), bytes.len);
    try std.testing.expectEqual(@as(usize, 100), arena.offset);
    try std.testing.expectEqual(@as(usize, 100), arena.high_water_mark);
    try std.testing.expectEqual(@as(usize, 0), arena.overflow_count);
}

test "ArenaAllocator alignment" {
    var buffer: [1024]u8 = undefined;
    var arena = ArenaAllocator.init(&buffer);

    // Allocate unaligned byte
    _ = try arena.alloc(1, 1);
    try std.testing.expectEqual(@as(usize, 1), arena.offset);

    // Allocate aligned to 8 bytes
    const aligned = try arena.alloc(8, 8);
    const addr = @intFromPtr(aligned.ptr);
    try std.testing.expectEqual(@as(usize, 0), addr % 8);

    // Offset should be aligned + size
    try std.testing.expectEqual(@as(usize, 8 + 8), arena.offset);
}

test "ArenaAllocator overflow detection" {
    // Set error_return strategy for this test
    arena_config.setConfig(.{
        .pool_size = 100,
        .overflow_strategy = .error_return,
    });
    defer arena_config.resetConfig();

    var buffer: [100]u8 = undefined;
    var arena = ArenaAllocator.init(&buffer);

    // Fill arena almost to capacity
    _ = try arena.alloc(90, 1);
    try std.testing.expectEqual(@as(usize, 90), arena.offset);
    try std.testing.expectEqual(@as(usize, 0), arena.overflow_count);

    // Attempt allocation that would overflow
    const result = arena.alloc(20, 1);
    try std.testing.expectError(error.OutOfMemory, result);

    // Offset should be unchanged after failed allocation
    try std.testing.expectEqual(@as(usize, 90), arena.offset);
    // Overflow should be recorded
    try std.testing.expectEqual(@as(usize, 1), arena.overflow_count);
}

test "ArenaAllocator reset" {
    arena_metrics.resetGlobalMetrics();

    var buffer: [1024]u8 = undefined;
    var arena = ArenaAllocator.init(&buffer);

    // Allocate some memory
    _ = try arena.alloc(500, 1);
    try std.testing.expectEqual(@as(usize, 500), arena.offset);
    try std.testing.expectEqual(@as(usize, 500), arena.high_water_mark);

    // Reset should clear offset but record metrics
    arena.reset();
    try std.testing.expectEqual(@as(usize, 0), arena.offset);
    try std.testing.expectEqual(@as(usize, 500), arena.high_water_mark); // Preserved
    try std.testing.expectEqual(@as(usize, 0), arena.overflow_count); // Cleared

    // Metrics should be recorded
    const metrics = arena_metrics.getGlobalMetrics();
    try std.testing.expectEqual(@as(usize, 1), metrics.total_resets);
    try std.testing.expectEqual(@as(usize, 500), metrics.max_high_water_mark);

    // Can allocate again after reset
    const bytes = try arena.alloc(500, 1);
    try std.testing.expectEqual(@as(usize, 500), bytes.len);
}

test "ArenaAllocator typed allocation" {
    var buffer: [1024]u8 = undefined;
    var arena = ArenaAllocator.init(&buffer);

    const floats = try arena.allocArray(f32, 100);
    try std.testing.expectEqual(@as(usize, 100), floats.len);

    // Check alignment for f32 (4 bytes)
    const addr = @intFromPtr(floats.ptr);
    try std.testing.expectEqual(@as(usize, 0), addr % @alignOf(f32));

    // Check size: 100 * 4 = 400 bytes
    try std.testing.expectEqual(@as(usize, 400), arena.offset);
}

test "ArenaAllocator remaining capacity" {
    var buffer: [1000]u8 = undefined;
    var arena = ArenaAllocator.init(&buffer);

    try std.testing.expectEqual(@as(usize, 1000), arena.remaining());

    _ = try arena.alloc(300, 1);
    try std.testing.expectEqual(@as(usize, 700), arena.remaining());

    _ = try arena.alloc(700, 1);
    try std.testing.expectEqual(@as(usize, 0), arena.remaining());
}

test "ArenaAllocator utilization" {
    var buffer: [1000]u8 = undefined;
    var arena = ArenaAllocator.init(&buffer);

    try std.testing.expectEqual(@as(f32, 0.0), arena.utilization());

    _ = try arena.alloc(250, 1);
    try std.testing.expectEqual(@as(f32, 0.25), arena.utilization());

    _ = try arena.alloc(250, 1);
    try std.testing.expectEqual(@as(f32, 0.5), arena.utilization());

    arena.reset();
    try std.testing.expectEqual(@as(f32, 0.0), arena.utilization());
}

test "thread-local arena initialization" {
    arena_config.setConfig(.{
        .pool_size = 1024,
        .overflow_strategy = .error_return,
    });
    defer arena_config.resetConfig();

    const arena1 = getThreadArena();
    const arena2 = getThreadArena();

    // Should return same instance
    try std.testing.expectEqual(arena1, arena2);

    // Should be able to allocate
    const bytes = try arena1.alloc(100, 1);
    try std.testing.expectEqual(@as(usize, 100), bytes.len);

    // Reset should clear
    resetThreadArena();
    try std.testing.expectEqual(@as(usize, 0), arena1.offset);
}

test "thread-local arena isolation" {
    arena_config.setConfig(.{
        .pool_size = 1024 * 1024, // 1MB
        .overflow_strategy = .error_return,
    });
    defer arena_config.resetConfig();

    // This test verifies thread-local isolation conceptually
    // Actual multi-threading test would require spawning threads
    const arena = getThreadArena();

    _ = try arena.alloc(100, 1);
    try std.testing.expectEqual(@as(usize, 100), arena.offset);

    resetThreadArena();
    try std.testing.expectEqual(@as(usize, 0), arena.offset);

    // Utilization tracking
    _ = try arena.alloc(512 * 1024, 1); // 512KB of 1MB
    const util = getThreadArenaUtilization();
    try std.testing.expect(util > 0.49 and util < 0.51); // ~0.5

    // Stats tracking
    const stats = getThreadArenaStats();
    try std.testing.expect(stats != null);
    if (stats) |s| {
        try std.testing.expectEqual(@as(usize, 1024 * 1024), s.capacity);
        try std.testing.expect(s.current_usage >= 512 * 1024);
    }

    resetThreadArena();
}
