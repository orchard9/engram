# Performance Regression Testing Guide

## Overview

Engram's performance regression framework ensures that code changes do not inadvertently degrade performance. The framework:

- Establishes baseline performance metrics for critical operations
- Runs automated benchmarks on every build
- Fails CI builds if performance regresses >5% from baseline
- Tracks performance trends over time

## Critical Benchmarks

The regression suite measures three core operations:

### 1. Vector Similarity (768d embeddings, 1000 candidates)

**Operation**: Cosine similarity computation for semantic search
**Target**: 1-2ms per batch (typical)
**Hot path**: Vector dot products, SIMD optimizations

### 2. Spreading Activation (1000 nodes, 100 iterations)

**Operation**: Graph traversal and activation propagation
**Target**: 80-100ms per query (typical)
**Hot path**: Concurrent graph operations, cache locality

### 3. Decay Calculation (10,000 memories)

**Operation**: Memory strength decay computation
**Target**: 50-70ms per batch (typical)
**Hot path**: Vectorized exponential calculations

## Baseline Management

### Establishing Baselines

Baselines should be established on representative hardware matching production:

```bash
# Clean build environment
cargo clean

# Run baseline update
./scripts/update_baselines.sh
```

This creates `engram-core/benches/regression/baselines.json` with:

```json
{
  "version": "1.0",
  "platform": "x86_64-apple-darwin",
  "cpu": "Apple M1 Pro",
  "timestamp": "2025-10-25T12:00:00Z",
  "baselines": {
    "vector_similarity_768d_1000c": {
      "mean_ns": 1700000,
      "std_dev_ns": 50000,
      "sample_size": 100
    },
    "spreading_activation_1000n_100i": {
      "mean_ns": 95000000,
      "std_dev_ns": 2000000,
      "sample_size": 100
    },
    "decay_calculation_10000m": {
      "mean_ns": 65000000,
      "std_dev_ns": 1500000,
      "sample_size": 100
    }
  }
}
```

### Platform-Specific Baselines

Different platforms require separate baselines:

- **x86_64 (AVX2)**: Desktop/server with SIMD vector extensions
- **ARM64 (NEON)**: Apple Silicon, mobile, embedded systems

Store baselines for each platform in version control. CI systems should use baselines matching their architecture.

### Updating Baselines

Update baselines after:

1. **Intentional performance improvements**
   ```bash
   ./scripts/update_baselines.sh
   git add engram-core/benches/regression/baselines.json
   git commit -m "chore: update baselines after SIMD optimization"
   ```

2. **API changes affecting benchmark behavior**
   - Review that functionality remains equivalent
   - Document why baseline changed in commit message

3. **Compiler or platform upgrades**
   - Expected after Rust version updates
   - May see improvements or regressions from compiler changes

## Running Regression Tests

### Local Development

Run regression checks before committing:

```bash
./scripts/benchmark_regression.sh
```

Output indicates pass/fail for each benchmark:

```
✓ OK: vector_similarity_768d_1000c within acceptable variance (-2.34% change)
✓ IMPROVEMENT: spreading_activation_1000n_100i is 7.12% faster than baseline
❌ REGRESSION DETECTED: decay_calculation_10000m
   Current: 78000000 ns
   Baseline: 65000000 ns
   Regression: 20.00%
```

### Continuous Integration

**IMPORTANT**: Per CLAUDE.md guidelines, Engram does not use GitHub workflows.

For CI integration, run the regression script in your CI system:

```bash
# Example CI command
./scripts/benchmark_regression.sh
```

The script exits with code 1 if regressions are detected, failing the build.

### Interpreting Results

**Acceptable variance**: ±5% from baseline

Performance can vary due to:
- System load (background processes)
- CPU frequency scaling
- Cache/memory pressure
- Compiler optimizations

**Green (✓)**: Performance within ±5% of baseline or improved >5%

**Yellow (⚠)**: Missing baseline - recorded for future comparisons

**Red (❌)**: Performance regressed >5% - investigate immediately

## Debugging Performance Regressions

When regressions are detected:

### Step 1: Verify Regression is Real

```bash
# Run multiple times to confirm
./scripts/benchmark_regression.sh
./scripts/benchmark_regression.sh
./scripts/benchmark_regression.sh
```

Consistent regressions indicate real issues. Sporadic failures may be system noise.

### Step 2: Isolate the Change

```bash
# Compare against previous commit
git checkout HEAD~1
./scripts/update_baselines.sh  # Establish baseline for old code
git checkout -
./scripts/benchmark_regression.sh  # Check if regression appears
```

### Step 3: Profile Hot Paths

```bash
# Generate flamegraph for regressed operation
cargo bench --bench profiling_harness
./scripts/profile_hotspots.sh
```

Flamegraphs show where CPU time is spent. Compare before/after to identify expensive operations.

### Step 4: Use Differential Profiling

```bash
# Profile with perf (Linux)
perf record cargo bench --bench regression
perf report

# Profile with Instruments (macOS)
cargo instruments --bench regression
```

### Step 5: Microbenchmark Suspects

Create focused benchmarks for suspect operations:

```rust
// In benches/debug_regression.rs
fn bench_suspect_operation(c: &mut Criterion) {
    c.bench_function("suspect", |b| {
        b.iter(|| {
            // Isolated operation from hot path
        });
    });
}
```

## Performance Optimization Workflow

### 1. Establish Baseline

```bash
git checkout main
./scripts/update_baselines.sh
```

### 2. Implement Optimization

Make targeted changes to hot paths identified by profiling.

### 3. Verify Improvement

```bash
UPDATE_BASELINES=1 cargo bench --bench regression
```

Check that optimization improved target metric without regressing others.

### 4. Profile for Side Effects

```bash
./scripts/profile_hotspots.sh
```

Ensure optimization didn't shift bottleneck elsewhere.

### 5. Update Baselines

```bash
./scripts/update_baselines.sh
git add engram-core/benches/regression/baselines.json
git commit -m "perf: optimize vector similarity with AVX2 intrinsics

- Implement SIMD dot product for 768d embeddings
- Reduces vector_similarity_768d_1000c from 1.7ms to 0.9ms (47% faster)
- Updated regression baselines for x86_64-apple-darwin"
```

## Baseline History and Trends

Track performance over time by preserving baseline history:

```bash
# Save historical baseline
cp engram-core/benches/regression/baselines.json \
   engram-core/benches/regression/baselines-$(git rev-parse --short HEAD).json
```

Plot trends with:

```python
# scripts/plot_performance_history.py
import json
import matplotlib.pyplot as plt

# Load historical baselines
# Plot mean_ns over time for each benchmark
# Identify performance trends
```

## Troubleshooting

### Regression tests fail in CI but pass locally

**Cause**: Different hardware, CPU frequency scaling, or system load

**Solution**:
1. Use dedicated CI runners with consistent hardware
2. Disable CPU frequency scaling on CI
3. Establish separate baselines for CI platform
4. Increase regression threshold for noisy CI (e.g., 10% instead of 5%)

### Baselines drift over time

**Cause**: Compiler updates, OS changes, transitive dependencies

**Solution**:
1. Re-establish baselines after major updates
2. Document baseline history in git commits
3. Use relative comparisons (% change) rather than absolute times

### Benchmarks are too slow

**Cause**: Large sample sizes, complex workloads

**Solution**:
1. Reduce sample size in `bench_function` calls
2. Use smaller datasets while maintaining representativeness
3. Run full benchmarks nightly instead of per-commit

### False positives from variance

**Cause**: Insufficient warm-up, system jitter

**Solution**:
1. Increase Criterion warm-up time
2. Increase sample size for more stable measurements
3. Run benchmarks with elevated priority (requires root)

## Best Practices

1. **Establish baselines early** - Before adding features, record baseline

2. **Review baselines in PRs** - Performance changes should be intentional

3. **Document performance changes** - Explain why baselines changed in commits

4. **Profile before optimizing** - Measure to identify real bottlenecks

5. **Optimize hot paths first** - Focus on operations in regression suite

6. **Test on target hardware** - Use same CPU architecture as production

7. **Minimize system noise** - Close applications during baseline updates

8. **Use consistent build flags** - Always benchmark with `--release`

9. **Track historical baselines** - Preserve old baselines for trend analysis

10. **Fail fast on regressions** - Don't merge code that regresses performance

## Future Enhancements

### Planned Improvements

- **Zig kernel baselines**: When Zig kernels are implemented (Milestone 10 Tasks 005-007), add baselines for Zig-optimized operations
- **Platform matrix**: Automated baselines for x86_64, ARM64, with different SIMD capabilities
- **Historical tracking**: Database of baseline history for long-term trend analysis
- **Automated bisection**: Auto-identify regressing commit with `git bisect`
- **Performance budgets**: Per-operation time budgets that enforce latency SLAs

### Integration with Profiling

Regression tests complement profiling infrastructure:

- **Profiling harness** (`benches/profiling_harness.rs`): Identifies hot paths
- **Regression tests** (`benches/regression/mod.rs`): Prevents performance degradation
- **Flamegraphs** (`scripts/profile_hotspots.sh`): Visualizes CPU time distribution

Use profiling to find optimization opportunities, regression tests to prevent backsliding.

## References

- Task 001: Profiling Infrastructure - Flamegraph generation and hotspot analysis
- Task 009: Integration Testing - Functional validation of kernels
- `scripts/profile_hotspots.sh` - Generate flamegraphs for hot path analysis
- Criterion documentation: https://bheisler.github.io/criterion.rs/book/

## Support

For performance regression issues:

1. Review this guide's troubleshooting section
2. Check flamegraphs from profiling harness
3. Compare against historical baselines
4. Profile with platform-specific tools (perf, Instruments)
5. Create focused microbenchmarks for suspect operations
