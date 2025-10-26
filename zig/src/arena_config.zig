// Arena allocator configuration management
// Supports runtime configuration from environment variables
//
// Design Principles:
// - Environment-based configuration for production flexibility
// - Sensible defaults for development use cases
// - Overflow strategies for different deployment scenarios
// - Zero overhead when using defaults (compile-time constants)
//
// Configuration Sources (priority order):
// 1. Explicit setConfig() calls (for testing)
// 2. Environment variables (for production)
// 3. Compile-time defaults (for development)

const std = @import("std");

/// Overflow behavior when arena capacity is exhausted
///
/// Different strategies optimize for different deployment scenarios:
/// - panic: Development/testing - fail fast on capacity issues
/// - error_return: Production - graceful error handling
/// - fallback: Best-effort - log warning and attempt fallback
pub const OverflowStrategy = enum {
    /// Panic immediately on overflow (development mode)
    ///
    /// Use when: Debugging capacity issues or validating workload sizing
    /// Behavior: @panic("Arena allocator overflow")
    /// Performance: Zero overhead (no runtime checks after overflow detection)
    panic,

    /// Return error.OutOfMemory on overflow (production mode)
    ///
    /// Use when: Graceful degradation required
    /// Behavior: Propagates error up the call stack
    /// Performance: Minimal overhead (error return path)
    error_return,

    /// Fall back to alternative allocation strategy (experimental)
    ///
    /// Use when: Best-effort allocation preferred over hard failure
    /// Behavior: Logs warning and returns error (fallback not yet implemented)
    /// Performance: Same as error_return currently
    /// Future: Could fall back to system allocator with tracking
    fallback,
};

/// Arena allocator configuration
///
/// Controls thread-local arena behavior including size and overflow handling.
/// Configuration is typically loaded once at startup and remains constant.
pub const ArenaConfig = struct {
    /// Size of thread-local arena in bytes
    ///
    /// Recommended ranges:
    /// - Small workloads (embedding only): 512KB - 1MB
    /// - Medium workloads (spreading + embedding): 2MB - 4MB
    /// - Large workloads (batch processing): 8MB - 16MB
    /// - Stress testing: 32MB - 100MB
    ///
    /// Trade-offs:
    /// - Larger pools: Reduce overflow risk, increase memory footprint
    /// - Smaller pools: Reduce memory waste, increase overflow risk
    pool_size: usize,

    /// Overflow behavior when pool exhausted
    ///
    /// Recommended settings:
    /// - Development: .panic (fail fast)
    /// - Testing: .error_return (validate error handling)
    /// - Production: .error_return (graceful degradation)
    /// - Experimental: .fallback (best-effort allocation)
    overflow_strategy: OverflowStrategy,

    /// Default configuration for development
    ///
    /// Conservative settings optimized for typical development workloads:
    /// - 1MB pool: Sufficient for ~262K f32 elements or ~1300 768-dim embeddings
    /// - error_return: Safe default for development and testing
    pub const DEFAULT = ArenaConfig{
        .pool_size = 1024 * 1024, // 1MB
        .overflow_strategy = .error_return,
    };

    /// Load configuration from environment variables or use defaults
    ///
    /// Environment variables:
    /// - ENGRAM_ARENA_SIZE: Pool size in bytes (e.g., "2097152" for 2MB)
    /// - ENGRAM_ARENA_OVERFLOW: Strategy as string ("panic", "error_return", "fallback")
    ///
    /// Invalid values are ignored and defaults are used instead.
    ///
    /// Examples:
    ///   ENGRAM_ARENA_SIZE=4194304 ENGRAM_ARENA_OVERFLOW=panic ./engram
    ///   # Uses 4MB pool with panic-on-overflow
    ///
    ///   ENGRAM_ARENA_SIZE=invalid ./engram
    ///   # Uses default 1MB pool (invalid value ignored)
    pub fn fromEnv() ArenaConfig {
        var config = DEFAULT;

        // Load pool size from ENGRAM_ARENA_SIZE (in bytes)
        if (std.posix.getenv("ENGRAM_ARENA_SIZE")) |size_str| {
            if (std.fmt.parseInt(usize, size_str, 10)) |size| {
                // Validate size is reasonable (at least 64KB, at most 1GB)
                if (size >= 64 * 1024 and size <= 1024 * 1024 * 1024) {
                    config.pool_size = size;
                }
            } else |_| {
                // Parsing failed, use default
            }
        }

        // Load overflow strategy from ENGRAM_ARENA_OVERFLOW
        if (std.posix.getenv("ENGRAM_ARENA_OVERFLOW")) |strategy_str| {
            // Try to parse as enum
            if (std.meta.stringToEnum(OverflowStrategy, strategy_str)) |strategy| {
                config.overflow_strategy = strategy;
            }
            // Invalid values are silently ignored (use default)
        }

        return config;
    }
};

// Global configuration (thread-safe via initialization once)
//
// Design: Configuration is loaded once at first use and then remains constant.
// This avoids the need for synchronization on every access.
//
// Thread safety: Multiple threads may race to initialize, but they'll all
// produce the same result (idempotent fromEnv()). No mutex needed.
var global_config: ArenaConfig = ArenaConfig.DEFAULT;
var config_initialized: bool = false;

/// Set global configuration explicitly
///
/// Useful for testing and programmatic configuration.
/// Should be called before any arena allocation occurs.
///
/// Thread safety: Not thread-safe. Call from main thread before spawning workers.
///
/// Example:
///   arena_config.setConfig(.{
///       .pool_size = 2 * 1024 * 1024,
///       .overflow_strategy = .panic,
///   });
pub fn setConfig(config: ArenaConfig) void {
    global_config = config;
    config_initialized = true;
}

/// Get current global configuration
///
/// Lazily initializes from environment on first call.
/// Subsequent calls return cached value.
///
/// Thread safety: Safe to call from multiple threads.
/// First call may race, but produces identical results.
///
/// Example:
///   const config = arena_config.getConfig();
///   const pool_size = config.pool_size;
pub fn getConfig() ArenaConfig {
    if (!config_initialized) {
        global_config = ArenaConfig.fromEnv();
        config_initialized = true;
    }
    return global_config;
}

/// Reset configuration to uninitialized state
///
/// Useful for testing to ensure clean environment.
/// Not safe to call while arenas are in use.
///
/// Thread safety: Not thread-safe. Use only in single-threaded tests.
pub fn resetConfig() void {
    global_config = ArenaConfig.DEFAULT;
    config_initialized = false;
}

// Unit tests
test "ArenaConfig default values" {
    const config = ArenaConfig.DEFAULT;
    try std.testing.expectEqual(@as(usize, 1024 * 1024), config.pool_size);
    try std.testing.expectEqual(OverflowStrategy.error_return, config.overflow_strategy);
}

test "ArenaConfig fromEnv with no environment variables" {
    // Clear any existing config
    resetConfig();

    const config = ArenaConfig.fromEnv();
    try std.testing.expectEqual(ArenaConfig.DEFAULT.pool_size, config.pool_size);
    try std.testing.expectEqual(ArenaConfig.DEFAULT.overflow_strategy, config.overflow_strategy);
}

test "ArenaConfig setConfig and getConfig" {
    resetConfig();

    const custom = ArenaConfig{
        .pool_size = 2 * 1024 * 1024,
        .overflow_strategy = .panic,
    };

    setConfig(custom);
    const retrieved = getConfig();

    try std.testing.expectEqual(custom.pool_size, retrieved.pool_size);
    try std.testing.expectEqual(custom.overflow_strategy, retrieved.overflow_strategy);
}

test "ArenaConfig resetConfig" {
    // Set custom config
    setConfig(.{
        .pool_size = 100,
        .overflow_strategy = .panic,
    });

    // Reset should restore defaults
    resetConfig();
    const config = getConfig();

    try std.testing.expectEqual(ArenaConfig.DEFAULT.pool_size, config.pool_size);
    try std.testing.expectEqual(ArenaConfig.DEFAULT.overflow_strategy, config.overflow_strategy);
}
