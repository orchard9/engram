// Comprehensive unit tests for arena allocator
// Tests allocation patterns, alignment, overflow, and thread-local behavior

const std = @import("std");
const allocator = @import("allocator.zig");
const ArenaAllocator = allocator.ArenaAllocator;

// Basic allocation tests

test "allocate zero bytes" {
    var buffer: [1024]u8 = undefined;
    var arena = ArenaAllocator.init(&buffer);

    const bytes = try arena.alloc(0, 1);
    try std.testing.expectEqual(@as(usize, 0), bytes.len);
    try std.testing.expectEqual(@as(usize, 0), arena.offset);
}

test "allocate single byte" {
    var buffer: [1024]u8 = undefined;
    var arena = ArenaAllocator.init(&buffer);

    const bytes = try arena.alloc(1, 1);
    try std.testing.expectEqual(@as(usize, 1), bytes.len);
    try std.testing.expectEqual(@as(usize, 1), arena.offset);
}

test "allocate multiple times" {
    var buffer: [1024]u8 = undefined;
    var arena = ArenaAllocator.init(&buffer);

    const bytes1 = try arena.alloc(100, 1);
    try std.testing.expectEqual(@as(usize, 100), bytes1.len);
    try std.testing.expectEqual(@as(usize, 100), arena.offset);

    const bytes2 = try arena.alloc(200, 1);
    try std.testing.expectEqual(@as(usize, 200), bytes2.len);
    try std.testing.expectEqual(@as(usize, 300), arena.offset);

    const bytes3 = try arena.alloc(300, 1);
    try std.testing.expectEqual(@as(usize, 300), bytes3.len);
    try std.testing.expectEqual(@as(usize, 600), arena.offset);
}

test "allocate entire arena" {
    var buffer: [1024]u8 = undefined;
    var arena = ArenaAllocator.init(&buffer);

    const bytes = try arena.alloc(1024, 1);
    try std.testing.expectEqual(@as(usize, 1024), bytes.len);
    try std.testing.expectEqual(@as(usize, 1024), arena.offset);
    try std.testing.expectEqual(@as(usize, 0), arena.remaining());
}

// Alignment tests

test "alignment power of two validation" {
    var buffer: [1024]u8 = undefined;
    var arena = ArenaAllocator.init(&buffer);

    // Common alignments
    _ = try arena.alloc(8, 1);
    _ = try arena.alloc(8, 2);
    _ = try arena.alloc(8, 4);
    _ = try arena.alloc(8, 8);
    _ = try arena.alloc(8, 16);
    _ = try arena.alloc(8, 32);
    _ = try arena.alloc(8, 64);
}

test "alignment for different types" {
    var buffer: [1024]u8 = undefined;
    var arena = ArenaAllocator.init(&buffer);

    // u8: 1-byte alignment
    const bytes = try arena.alloc(10, @alignOf(u8));
    try std.testing.expectEqual(@as(usize, 0), @intFromPtr(bytes.ptr) % @alignOf(u8));

    // u16: 2-byte alignment
    const u16_bytes = try arena.alloc(10, @alignOf(u16));
    try std.testing.expectEqual(@as(usize, 0), @intFromPtr(u16_bytes.ptr) % @alignOf(u16));

    // u32: 4-byte alignment
    const u32_bytes = try arena.alloc(10, @alignOf(u32));
    try std.testing.expectEqual(@as(usize, 0), @intFromPtr(u32_bytes.ptr) % @alignOf(u32));

    // u64: 8-byte alignment
    const u64_bytes = try arena.alloc(10, @alignOf(u64));
    try std.testing.expectEqual(@as(usize, 0), @intFromPtr(u64_bytes.ptr) % @alignOf(u64));

    // f32: 4-byte alignment
    const f32_bytes = try arena.alloc(10, @alignOf(f32));
    try std.testing.expectEqual(@as(usize, 0), @intFromPtr(f32_bytes.ptr) % @alignOf(f32));

    // f64: 8-byte alignment
    const f64_bytes = try arena.alloc(10, @alignOf(f64));
    try std.testing.expectEqual(@as(usize, 0), @intFromPtr(f64_bytes.ptr) % @alignOf(f64));
}

test "alignment padding calculation" {
    var buffer: [1024]u8 = undefined;
    var arena = ArenaAllocator.init(&buffer);

    // Allocate 1 byte (offset = 1)
    _ = try arena.alloc(1, 1);
    try std.testing.expectEqual(@as(usize, 1), arena.offset);

    // Allocate 8-byte aligned (should pad to offset 8)
    const aligned = try arena.alloc(1, 8);
    try std.testing.expectEqual(@as(usize, 0), @intFromPtr(aligned.ptr) % 8);
    try std.testing.expectEqual(@as(usize, 9), arena.offset); // 8 (aligned) + 1 (size)

    // Allocate 1 byte (offset = 9)
    _ = try arena.alloc(1, 1);
    try std.testing.expectEqual(@as(usize, 10), arena.offset);

    // Allocate 4-byte aligned (should pad to offset 12)
    const aligned4 = try arena.alloc(1, 4);
    try std.testing.expectEqual(@as(usize, 0), @intFromPtr(aligned4.ptr) % 4);
    try std.testing.expectEqual(@as(usize, 13), arena.offset); // 12 (aligned) + 1 (size)
}

// Typed allocation tests

test "allocArray for primitives" {
    var buffer: [1024]u8 = undefined;
    var arena = ArenaAllocator.init(&buffer);

    const u8_array = try arena.allocArray(u8, 100);
    try std.testing.expectEqual(@as(usize, 100), u8_array.len);

    const u32_array = try arena.allocArray(u32, 50);
    try std.testing.expectEqual(@as(usize, 50), u32_array.len);

    const f32_array = try arena.allocArray(f32, 25);
    try std.testing.expectEqual(@as(usize, 25), f32_array.len);

    const f64_array = try arena.allocArray(f64, 10);
    try std.testing.expectEqual(@as(usize, 10), f64_array.len);
}

test "allocArray respects alignment" {
    var buffer: [1024]u8 = undefined;
    var arena = ArenaAllocator.init(&buffer);

    // Misalign the offset
    _ = try arena.alloc(1, 1);

    // f64 requires 8-byte alignment
    const f64_array = try arena.allocArray(f64, 10);
    const addr = @intFromPtr(f64_array.ptr);
    try std.testing.expectEqual(@as(usize, 0), addr % @alignOf(f64));
}

test "allocArray writes and reads correctly" {
    var buffer: [1024]u8 = undefined;
    var arena = ArenaAllocator.init(&buffer);

    const floats = try arena.allocArray(f32, 5);
    floats[0] = 1.0;
    floats[1] = 2.0;
    floats[2] = 3.0;
    floats[3] = 4.0;
    floats[4] = 5.0;

    try std.testing.expectEqual(@as(f32, 1.0), floats[0]);
    try std.testing.expectEqual(@as(f32, 2.0), floats[1]);
    try std.testing.expectEqual(@as(f32, 3.0), floats[2]);
    try std.testing.expectEqual(@as(f32, 4.0), floats[3]);
    try std.testing.expectEqual(@as(f32, 5.0), floats[4]);
}

// Overflow tests

test "overflow on exact capacity" {
    var buffer: [100]u8 = undefined;
    var arena = ArenaAllocator.init(&buffer);

    _ = try arena.alloc(100, 1);
    try std.testing.expectEqual(@as(usize, 0), arena.remaining());

    // Next allocation should fail
    const result = arena.alloc(1, 1);
    try std.testing.expectError(error.OutOfMemory, result);
}

test "overflow with alignment padding" {
    var buffer: [100]u8 = undefined;
    var arena = ArenaAllocator.init(&buffer);

    // Allocate 95 bytes
    _ = try arena.alloc(95, 1);
    try std.testing.expectEqual(@as(usize, 95), arena.offset);

    // Attempt 8-byte aligned allocation
    // Would need: alignForward(95, 8) = 96, plus 8 bytes = 104 > 100
    const result = arena.alloc(8, 8);
    try std.testing.expectError(error.OutOfMemory, result);
}

test "overflow does not modify offset" {
    var buffer: [100]u8 = undefined;
    var arena = ArenaAllocator.init(&buffer);

    _ = try arena.alloc(90, 1);
    const offset_before = arena.offset;

    const result = arena.alloc(20, 1);
    try std.testing.expectError(error.OutOfMemory, result);
    try std.testing.expectEqual(offset_before, arena.offset);
}

// Reset tests

test "reset clears offset" {
    var buffer: [1024]u8 = undefined;
    var arena = ArenaAllocator.init(&buffer);

    _ = try arena.alloc(500, 1);
    try std.testing.expectEqual(@as(usize, 500), arena.offset);

    arena.reset();
    try std.testing.expectEqual(@as(usize, 0), arena.offset);
}

test "reset allows reallocation" {
    var buffer: [1024]u8 = undefined;
    var arena = ArenaAllocator.init(&buffer);

    const bytes1 = try arena.alloc(500, 1);
    try std.testing.expectEqual(@as(usize, 500), bytes1.len);

    arena.reset();

    const bytes2 = try arena.alloc(500, 1);
    try std.testing.expectEqual(@as(usize, 500), bytes2.len);

    // Pointers should be the same (reusing same memory)
    try std.testing.expectEqual(bytes1.ptr, bytes2.ptr);
}

test "multiple reset cycles" {
    var buffer: [1024]u8 = undefined;
    var arena = ArenaAllocator.init(&buffer);

    for (0..10) |_| {
        _ = try arena.alloc(100, 1);
        try std.testing.expectEqual(@as(usize, 100), arena.offset);
        arena.reset();
        try std.testing.expectEqual(@as(usize, 0), arena.offset);
    }
}

// Remaining capacity tests

test "remaining capacity decreases with allocations" {
    var buffer: [1000]u8 = undefined;
    var arena = ArenaAllocator.init(&buffer);

    try std.testing.expectEqual(@as(usize, 1000), arena.remaining());

    _ = try arena.alloc(100, 1);
    try std.testing.expectEqual(@as(usize, 900), arena.remaining());

    _ = try arena.alloc(200, 1);
    try std.testing.expectEqual(@as(usize, 700), arena.remaining());

    _ = try arena.alloc(700, 1);
    try std.testing.expectEqual(@as(usize, 0), arena.remaining());
}

test "remaining capacity with alignment padding" {
    var buffer: [1000]u8 = undefined;
    var arena = ArenaAllocator.init(&buffer);

    // Allocate 1 byte (offset = 1)
    _ = try arena.alloc(1, 1);
    try std.testing.expectEqual(@as(usize, 999), arena.remaining());

    // Allocate 8-byte aligned (offset = 8 + 8 = 16)
    _ = try arena.alloc(8, 8);
    try std.testing.expectEqual(@as(usize, 1000 - 16), arena.remaining());
}

// Utilization tests

test "utilization tracks usage fraction" {
    var buffer: [1000]u8 = undefined;
    var arena = ArenaAllocator.init(&buffer);

    try std.testing.expectEqual(@as(f32, 0.0), arena.utilization());

    _ = try arena.alloc(250, 1);
    try std.testing.expectEqual(@as(f32, 0.25), arena.utilization());

    _ = try arena.alloc(250, 1);
    try std.testing.expectEqual(@as(f32, 0.5), arena.utilization());

    _ = try arena.alloc(500, 1);
    try std.testing.expectEqual(@as(f32, 1.0), arena.utilization());
}

test "utilization after reset" {
    var buffer: [1000]u8 = undefined;
    var arena = ArenaAllocator.init(&buffer);

    _ = try arena.alloc(750, 1);
    try std.testing.expectEqual(@as(f32, 0.75), arena.utilization());

    arena.reset();
    try std.testing.expectEqual(@as(f32, 0.0), arena.utilization());
}

// Thread-local arena tests

test "getThreadArena returns valid arena" {
    const arena = allocator.getThreadArena();
    try std.testing.expect(arena.buffer.len > 0);
}

test "getThreadArena returns same instance" {
    const arena1 = allocator.getThreadArena();
    const arena2 = allocator.getThreadArena();
    try std.testing.expectEqual(arena1, arena2);
}

test "thread-local arena can allocate" {
    const arena = allocator.getThreadArena();
    allocator.resetThreadArena(); // Clear any previous state

    const bytes = try arena.alloc(100, 1);
    try std.testing.expectEqual(@as(usize, 100), bytes.len);
    try std.testing.expectEqual(@as(usize, 100), arena.offset);
}

test "resetThreadArena clears thread-local arena" {
    const arena = allocator.getThreadArena();

    _ = try arena.alloc(500, 1);
    try std.testing.expectEqual(@as(usize, 500), arena.offset);

    allocator.resetThreadArena();
    try std.testing.expectEqual(@as(usize, 0), arena.offset);
}

test "thread-local arena has expected capacity" {
    const arena = allocator.getThreadArena();
    allocator.resetThreadArena();

    // Should be able to allocate 1MB
    const bytes = try arena.alloc(1024 * 1024, 1);
    try std.testing.expectEqual(@as(usize, 1024 * 1024), bytes.len);
}

test "getThreadArenaUtilization tracks usage" {
    const arena = allocator.getThreadArena();
    allocator.resetThreadArena();

    try std.testing.expectEqual(@as(f32, 0.0), allocator.getThreadArenaUtilization());

    // Allocate 512KB of 1MB
    _ = try arena.alloc(512 * 1024, 1);
    const util = allocator.getThreadArenaUtilization();
    try std.testing.expect(util > 0.49 and util < 0.51); // ~0.5

    allocator.resetThreadArena();
    try std.testing.expectEqual(@as(f32, 0.0), allocator.getThreadArenaUtilization());
}

// Stress tests

test "many small allocations" {
    var buffer: [10000]u8 = undefined;
    var arena = ArenaAllocator.init(&buffer);

    for (0..100) |i| {
        const bytes = try arena.alloc(10, 1);
        try std.testing.expectEqual(@as(usize, 10), bytes.len);
        try std.testing.expectEqual(@as(usize, (i + 1) * 10), arena.offset);
    }

    try std.testing.expectEqual(@as(usize, 1000), arena.offset);
}

test "alternating allocation sizes" {
    var buffer: [10000]u8 = undefined;
    var arena = ArenaAllocator.init(&buffer);

    var expected_offset: usize = 0;
    for (0..50) |_| {
        _ = try arena.alloc(10, 1);
        expected_offset += 10;
        _ = try arena.alloc(100, 1);
        expected_offset += 100;
    }

    try std.testing.expectEqual(expected_offset, arena.offset);
}

test "allocation with mixed alignments" {
    var buffer: [10000]u8 = undefined;
    var arena = ArenaAllocator.init(&buffer);

    for (0..100) |_| {
        _ = try arena.alloc(1, 1); // 1-byte
        _ = try arena.alloc(4, 4); // 4-byte aligned
        _ = try arena.alloc(8, 8); // 8-byte aligned
    }

    // Should successfully allocate without overflow
    try std.testing.expect(arena.offset < buffer.len);
}

test "edge case: zero-size arena" {
    var buffer: [0]u8 = undefined;
    var arena = ArenaAllocator.init(&buffer);

    try std.testing.expectEqual(@as(usize, 0), arena.remaining());

    const result = arena.alloc(1, 1);
    try std.testing.expectError(error.OutOfMemory, result);
}

test "edge case: allocation at exact remaining capacity" {
    var buffer: [100]u8 = undefined;
    var arena = ArenaAllocator.init(&buffer);

    _ = try arena.alloc(50, 1);
    try std.testing.expectEqual(@as(usize, 50), arena.remaining());

    // Allocate exactly remaining capacity
    const bytes = try arena.alloc(50, 1);
    try std.testing.expectEqual(@as(usize, 50), bytes.len);
    try std.testing.expectEqual(@as(usize, 0), arena.remaining());
}
