// Arena allocator metrics and monitoring
// Provides global tracking of arena usage patterns
//
// Design Principles:
// - Thread-safe global aggregation via mutex
// - Minimal overhead (<1% target)
// - High-water mark tracking for capacity planning
// - Reset event counting for allocation pattern analysis
// - Overflow detection for production alerting
//
// Architecture:
// Each thread maintains local arena state (offset, high-water mark).
// On reset, local metrics are aggregated into global counters.
// Global metrics use mutex for thread-safety (infrequent updates).

const std = @import("std");

/// Global arena metrics
///
/// Aggregates metrics from all thread-local arenas to provide
/// system-wide visibility into allocation patterns.
pub const Metrics = struct {
    /// Total number of arena resets across all threads
    ///
    /// Each reset indicates completion of a kernel invocation.
    /// High reset rate suggests good arena reuse efficiency.
    /// Low reset rate may indicate long-running operations or leaks.
    total_resets: usize,

    /// Total overflow events across all threads
    ///
    /// Overflow indicates arena capacity was insufficient.
    /// Non-zero value suggests need for larger pool_size.
    /// Production deployments should monitor this metric.
    total_overflows: usize,

    /// Maximum high-water mark across all thread arenas
    ///
    /// Indicates peak utilization of any single thread's arena.
    /// Use for capacity planning: pool_size should exceed this value.
    /// Consider adding 20-50% headroom for safety margin.
    max_high_water_mark: usize,

    /// Initialize metrics to zero state
    pub fn init() Metrics {
        return .{
            .total_resets = 0,
            .total_overflows = 0,
            .max_high_water_mark = 0,
        };
    }
};

// Global metrics storage
// Protected by mutex for thread-safe updates
var global_metrics: Metrics = Metrics.init();
var metrics_mutex: std.Thread.Mutex = .{};

/// Record a reset event from a thread-local arena
///
/// Called automatically when arena.reset() is invoked.
/// Updates global metrics with thread-local statistics.
///
/// Performance: Single mutex acquisition per reset (amortized O(1))
/// Overhead: <1% for typical kernel durations (milliseconds+)
///
/// Arguments:
/// - high_water_mark: Peak usage of arena before reset
/// - overflow_count: Number of overflow events since last reset
///
/// Thread safety: Thread-safe via internal mutex
///
/// Example:
///   const arena = getThreadArena();
///   // ... use arena ...
///   const hwm = arena.offset;
///   const overflows = arena.overflow_count;
///   arena.reset();
///   recordReset(hwm, overflows);
pub fn recordReset(high_water_mark: usize, overflow_count: usize) void {
    metrics_mutex.lock();
    defer metrics_mutex.unlock();

    // Increment reset counter
    global_metrics.total_resets += 1;

    // Accumulate overflow events
    global_metrics.total_overflows += overflow_count;

    // Track maximum high-water mark across all threads
    global_metrics.max_high_water_mark = @max(
        global_metrics.max_high_water_mark,
        high_water_mark,
    );
}

/// Get snapshot of current global metrics
///
/// Returns a copy of metrics at time of call.
/// Metrics may change immediately after return due to concurrent updates.
///
/// Thread safety: Thread-safe via internal mutex
///
/// Example:
///   const metrics = getGlobalMetrics();
///   std.log.info("Resets: {}, Overflows: {}, Peak: {}",
///       .{metrics.total_resets, metrics.total_overflows, metrics.max_high_water_mark});
pub fn getGlobalMetrics() Metrics {
    metrics_mutex.lock();
    defer metrics_mutex.unlock();
    return global_metrics;
}

/// Reset global metrics to zero
///
/// Useful for testing or when starting a new measurement period.
/// Should not be called during active workload (may lose data).
///
/// Thread safety: Thread-safe via internal mutex
///
/// Example:
///   // Start fresh measurement period
///   resetGlobalMetrics();
///   // ... run workload ...
///   const metrics = getGlobalMetrics();
pub fn resetGlobalMetrics() void {
    metrics_mutex.lock();
    defer metrics_mutex.unlock();
    global_metrics = Metrics.init();
}

// Unit tests
test "Metrics initialization" {
    const metrics = Metrics.init();
    try std.testing.expectEqual(@as(usize, 0), metrics.total_resets);
    try std.testing.expectEqual(@as(usize, 0), metrics.total_overflows);
    try std.testing.expectEqual(@as(usize, 0), metrics.max_high_water_mark);
}

test "recordReset updates metrics" {
    resetGlobalMetrics();

    // Record first reset
    recordReset(1000, 0);

    var metrics = getGlobalMetrics();
    try std.testing.expectEqual(@as(usize, 1), metrics.total_resets);
    try std.testing.expectEqual(@as(usize, 0), metrics.total_overflows);
    try std.testing.expectEqual(@as(usize, 1000), metrics.max_high_water_mark);

    // Record second reset with overflow
    recordReset(500, 2);

    metrics = getGlobalMetrics();
    try std.testing.expectEqual(@as(usize, 2), metrics.total_resets);
    try std.testing.expectEqual(@as(usize, 2), metrics.total_overflows);
    try std.testing.expectEqual(@as(usize, 1000), metrics.max_high_water_mark); // Still 1000
}

test "recordReset tracks maximum high-water mark" {
    resetGlobalMetrics();

    // Record resets with increasing high-water marks
    recordReset(100, 0);
    recordReset(500, 0);
    recordReset(300, 0); // Lower than previous, should not update max

    const metrics = getGlobalMetrics();
    try std.testing.expectEqual(@as(usize, 500), metrics.max_high_water_mark);
}

test "recordReset accumulates overflows" {
    resetGlobalMetrics();

    // Record multiple resets with overflows
    recordReset(1000, 1);
    recordReset(1000, 3);
    recordReset(1000, 2);

    const metrics = getGlobalMetrics();
    try std.testing.expectEqual(@as(usize, 6), metrics.total_overflows); // 1 + 3 + 2
}

test "resetGlobalMetrics clears all metrics" {
    // Set up some metrics
    recordReset(1000, 5);
    recordReset(2000, 3);

    // Verify metrics are non-zero
    var metrics = getGlobalMetrics();
    try std.testing.expect(metrics.total_resets > 0);
    try std.testing.expect(metrics.total_overflows > 0);
    try std.testing.expect(metrics.max_high_water_mark > 0);

    // Reset should clear everything
    resetGlobalMetrics();
    metrics = getGlobalMetrics();
    try std.testing.expectEqual(@as(usize, 0), metrics.total_resets);
    try std.testing.expectEqual(@as(usize, 0), metrics.total_overflows);
    try std.testing.expectEqual(@as(usize, 0), metrics.max_high_water_mark);
}

test "getGlobalMetrics returns copy" {
    resetGlobalMetrics();
    recordReset(100, 1);

    // Get two copies
    const metrics1 = getGlobalMetrics();
    const metrics2 = getGlobalMetrics();

    // They should have same values
    try std.testing.expectEqual(metrics1.total_resets, metrics2.total_resets);
    try std.testing.expectEqual(metrics1.total_overflows, metrics2.total_overflows);
    try std.testing.expectEqual(metrics1.max_high_water_mark, metrics2.max_high_water_mark);

    // Modify copy should not affect global
    var metrics_copy = metrics1;
    metrics_copy.total_resets = 9999;

    const metrics3 = getGlobalMetrics();
    try std.testing.expectEqual(@as(usize, 1), metrics3.total_resets); // Still 1, not 9999
}
