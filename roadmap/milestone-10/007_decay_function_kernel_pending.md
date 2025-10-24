# Task 007: Decay Function Kernel

**Duration:** 2 days
**Status:** Pending
**Dependencies:** 006 (Activation Spreading Kernel)

## Objectives

Implement optimized memory decay kernel in Zig to accelerate Ebbinghaus forgetting curve calculations. This operation accounts for 10-15% of compute time and involves SIMD-friendly mathematical operations (exponentials, divisions) that benefit significantly from vectorization.

1. **SIMD exponential** - Vectorized exp() approximation for decay calculation
2. **Batch processing** - Process thousands of memories in parallel
3. **Numerical accuracy** - Match Rust baseline within acceptable epsilon
4. **Performance validation** - 20-30% improvement over Rust baseline

## Dependencies

- Task 006 (Activation Spreading Kernel) - SIMD patterns and optimization techniques established

## Deliverables

### Files to Create

1. `/zig/src/decay_functions.zig` - Core implementation
   - ebbinghausDecay: Vectorized forgetting curve
   - Fast exp() approximation for SIMD
   - Batch decay application

2. `/zig/src/decay_functions_test.zig` - Unit tests
   - Decay curve correctness
   - Edge cases (zero age, ancient memories)
   - Numerical accuracy validation

3. `/benches/decay_comparison.rs` - Performance benchmarks
   - Rust baseline vs. Zig kernel
   - Various memory counts (100, 1000, 10000)
   - Different age distributions

### Files to Modify

1. `/zig/src/ffi.zig` - Export decay function
   - engram_apply_decay implementation
   - Batch processing integration

2. `/src/zig_kernels/mod.rs` - Rust wrapper
   - Safe wrapper for apply_decay
   - Age calculation from timestamps

3. `/tests/zig_differential/decay_functions.rs` - Extend differential tests
   - Property-based age generation
   - Validate decay curves

## Acceptance Criteria

1. All differential tests pass (Zig matches Rust within epsilon = 1e-6)
2. Performance improvement: 20-30% faster than Rust baseline
3. Numerical accuracy within 0.1% of baseline for all inputs
4. Handles edge cases correctly (zero age, overflow ages)
5. Benchmarks show improvement across varying batch sizes

## Implementation Guidance

### Ebbinghaus Decay Formula

The Ebbinghaus forgetting curve with spacing effect:

```
strength_new = strength_old * exp(-age_seconds / half_life)
where half_life = base_half_life * (1 + repetition_count)
```

For simplicity, assume base_half_life = 86400 (1 day) and no repetition tracking.

### Fast Exponential Approximation

Use polynomial approximation for fast SIMD exp():

```zig
const std = @import("std");

/// Fast exp() approximation using polynomial (accurate to ~1e-5)
fn fastExp(x: f32) f32 {
    // Clamp to avoid overflow
    const clamped = std.math.clamp(x, -10.0, 10.0);

    // Polynomial approximation: exp(x) ≈ 1 + x + x²/2 + x³/6 + x⁴/24
    const x2 = clamped * clamped;
    const x3 = x2 * clamped;
    const x4 = x2 * x2;

    return 1.0 + clamped + x2 * 0.5 + x3 * 0.16666667 + x4 * 0.041666667;
}

/// Vectorized fast exp for SIMD
fn fastExpSimd(comptime width: u32, x: @Vector(width, f32)) @Vector(width, f32) {
    // Clamp
    const clamped = @max(@min(x, @as(@Vector(width, f32), @splat(10.0))),
                          @as(@Vector(width, f32), @splat(-10.0)));

    // Polynomial
    const x2 = clamped * clamped;
    const x3 = x2 * clamped;
    const x4 = x2 * x2;

    const one: @Vector(width, f32) = @splat(1.0);
    const c1: @Vector(width, f32) = @splat(0.5);
    const c2: @Vector(width, f32) = @splat(0.16666667);
    const c3: @Vector(width, f32) = @splat(0.041666667);

    return one + clamped + x2 * c1 + x3 * c2 + x4 * c3;
}
```

Note: For production, consider using std.math.exp() unless profiling shows it's a bottleneck.

### Decay Calculation

```zig
pub fn ebbinghausDecay(strength: f32, age_seconds: u64, half_life_seconds: u64) f32 {
    if (age_seconds == 0) return strength;
    if (strength == 0.0) return 0.0;

    const age_f = @as(f32, @floatFromInt(age_seconds));
    const half_life_f = @as(f32, @floatFromInt(half_life_seconds));

    // decay = exp(-age / half_life)
    const exponent = -age_f / half_life_f;
    const decay_factor = std.math.exp(exponent);

    return strength * decay_factor;
}

/// Batch decay application with SIMD
pub fn batchDecay(
    strengths: []f32,
    ages_seconds: []const u64,
    half_life_seconds: u64,
) void {
    std.debug.assert(strengths.len == ages_seconds.len);

    const half_life_f = @as(f32, @floatFromInt(half_life_seconds));
    const simd_width = 8;
    const simd_len = (strengths.len / simd_width) * simd_width;

    const half_life_vec: @Vector(8, f32) = @splat(half_life_f);

    var i: usize = 0;
    while (i < simd_len) : (i += simd_width) {
        // Load strengths
        const strength_vec: @Vector(8, f32) = strengths[i..][0..simd_width].*;

        // Convert ages to f32
        var age_vec: @Vector(8, f32) = undefined;
        inline for (0..8) |j| {
            age_vec[j] = @floatFromInt(ages_seconds[i + j]);
        }

        // Compute exponents: -age / half_life
        const exponent_vec = -age_vec / half_life_vec;

        // Apply exp (use fast approximation if needed)
        var decay_vec: @Vector(8, f32) = undefined;
        inline for (0..8) |j| {
            decay_vec[j] = std.math.exp(exponent_vec[j]);
        }

        // Apply decay
        const result_vec = strength_vec * decay_vec;

        // Store results
        strengths[i..][0..simd_width].* = result_vec;
    }

    // Handle remaining elements
    while (i < strengths.len) : (i += 1) {
        strengths[i] = ebbinghausDecay(strengths[i], ages_seconds[i], half_life_seconds);
    }
}
```

### FFI Integration

```zig
// ffi.zig
const decay = @import("decay_functions.zig");

const DEFAULT_HALF_LIFE: u64 = 86400; // 1 day in seconds

export fn engram_apply_decay(
    strengths: [*]f32,
    ages_seconds: [*]const u64,
    num_memories: usize,
) void {
    const strengths_slice = strengths[0..num_memories];
    const ages_slice = ages_seconds[0..num_memories];

    decay.batchDecay(strengths_slice, ages_slice, DEFAULT_HALF_LIFE);
}
```

### Rust Wrapper

```rust
// src/zig_kernels/mod.rs

#[cfg(feature = "zig-kernels")]
pub fn apply_decay(memories: &mut [Memory], current_time: Timestamp) {
    let num_memories = memories.len();

    // Extract strengths and ages
    let mut strengths: Vec<f32> = memories.iter()
        .map(|m| m.strength)
        .collect();

    let ages: Vec<u64> = memories.iter()
        .map(|m| (current_time - m.last_accessed).as_secs())
        .collect();

    // Call Zig kernel
    unsafe {
        ffi::engram_apply_decay(
            strengths.as_mut_ptr(),
            ages.as_ptr(),
            num_memories,
        );
    }

    // Write back strengths
    for (memory, strength) in memories.iter_mut().zip(strengths.iter()) {
        memory.strength = *strength;
    }
}
```

### Performance Benchmarks

```rust
// benches/decay_comparison.rs
use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};

fn bench_decay_calculation(c: &mut Criterion) {
    let mut group = c.benchmark_group("decay_calculation");

    for num_memories in [100, 1000, 10_000] {
        let mut strengths: Vec<f32> = (0..num_memories)
            .map(|_| rand::random::<f32>())
            .collect();

        let ages: Vec<u64> = (0..num_memories)
            .map(|_| rand::random::<u64>() % 1_000_000)
            .collect();

        // Rust baseline
        group.bench_with_input(
            BenchmarkId::new("rust", num_memories),
            &(&mut strengths.clone(), &ages),
            |b, (strengths, ages)| {
                b.iter(|| {
                    rust_apply_decay(strengths, ages);
                    black_box(strengths);
                });
            },
        );

        // Zig kernel
        #[cfg(feature = "zig-kernels")]
        group.bench_with_input(
            BenchmarkId::new("zig", num_memories),
            &(&mut strengths.clone(), &ages),
            |b, (strengths, ages)| {
                b.iter(|| {
                    zig_apply_decay(strengths, ages);
                    black_box(strengths);
                });
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_decay_calculation);
criterion_main!(benches);
```

### Numerical Accuracy Tests

```rust
// tests/zig_differential/decay_functions.rs

#[test]
#[cfg(feature = "zig-kernels")]
fn test_decay_numerical_accuracy() {
    let test_cases = vec![
        // (strength, age_seconds, expected_decay_factor)
        (1.0, 0, 1.0),           // No decay
        (1.0, 86400, 0.5),       // Half-life
        (1.0, 172800, 0.25),     // Two half-lives
        (0.8, 43200, 0.8 * 0.707), // Half of half-life
    ];

    for (initial_strength, age, expected_factor) in test_cases {
        let mut strengths = vec![initial_strength];
        let ages = vec![age];

        apply_decay(&mut strengths, &ages);

        let expected = initial_strength * expected_factor;
        assert_relative_eq!(strengths[0], expected, epsilon = 0.001);
    }
}

proptest! {
    #[test]
    fn decay_matches_rust(
        strengths in prop::collection::vec(0.0_f32..1.0_f32, 100..1000),
        ages in prop::collection::vec(0_u64..1_000_000_u64, 100..1000)
    ) {
        let mut zig_strengths = strengths.clone();
        let mut rust_strengths = strengths.clone();

        zig_apply_decay(&mut zig_strengths, &ages);
        rust_apply_decay(&mut rust_strengths, &ages);

        for (zig, rust) in zig_strengths.iter().zip(rust_strengths.iter()) {
            assert_relative_eq!(zig, rust, epsilon = 1e-6);
        }
    }
}
```

## Testing Approach

1. **Unit tests**
   - Decay curve correctness
   - Edge cases (zero age, max age)
   - Fast exp() accuracy

2. **Differential tests**
   - Property-based strength/age generation
   - Verify Zig matches Rust
   - Test numerical stability

3. **Performance validation**
   - Benchmark various batch sizes
   - Verify 20-30% improvement
   - Profile SIMD utilization

## Integration Points

- **Task 006 (Activation Spreading)** - Decay applied during spreading
- **Task 009 (Integration Testing)** - Validate in memory consolidation pipeline
- **Task 010 (Performance Regression)** - Add to regression benchmark suite

## Notes

- Consider using lookup tables for exp() if accuracy allows
- Profile std.math.exp() vs. fast approximation tradeoff
- Document half-life parameter tuning in operational guide
- May benefit from FMA (fused multiply-add) instructions
- Test numerical stability with very old memories (large ages)
