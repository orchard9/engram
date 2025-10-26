# Zig Performance Kernels - Operations Guide

## Overview

Engram includes optional Zig performance kernels that accelerate compute-intensive operations through SIMD vectorization and optimized memory management. These kernels provide measurable performance improvements for production workloads:

- **Vector similarity**: 15-25% faster cosine similarity calculations for embedding search
- **Activation spreading**: 20-35% faster graph traversal for associative memory retrieval
- **Memory decay**: 20-30% faster Ebbinghaus decay calculations for temporal dynamics

This guide covers deployment, configuration, monitoring, and troubleshooting for production environments.

## Architecture Summary

Zig kernels integrate with Rust through a C-compatible FFI boundary:

- **Language boundary**: Rust manages graph data structures, Zig handles compute kernels
- **Memory model**: Caller-allocated buffers (Rust), zero-copy computation (Zig)
- **Thread safety**: Thread-local arena allocators eliminate contention
- **Fallback**: Runtime feature detection gracefully degrades to Rust implementations

For detailed architecture, see [Architecture Documentation](../internal/zig_architecture.md).

## Prerequisites

### System Requirements

- **Operating System**: Linux (x86_64, ARM64) or macOS (Apple Silicon, Intel)
- **CPU Features**: AVX2 (x86_64) or NEON (ARM64) for SIMD acceleration
- **Memory**: Minimum 1MB per thread for arena allocators (configurable)
- **Zig Compiler**: Version 0.13.0 (required at build time)

### Installing Zig

#### macOS

```bash
# Homebrew installation
brew install zig

# Verify installation
zig version  # Should output: 0.13.0
```

#### Linux

```bash
# Download official release
wget https://ziglang.org/download/0.13.0/zig-linux-x86_64-0.13.0.tar.xz
tar xf zig-linux-x86_64-0.13.0.tar.xz
sudo mv zig-linux-x86_64-0.13.0 /opt/zig

# Add to PATH
echo 'export PATH=/opt/zig:$PATH' >> ~/.bashrc
source ~/.bashrc

# Verify installation
zig version
```

#### ARM64 Linux

```bash
wget https://ziglang.org/download/0.13.0/zig-linux-aarch64-0.13.0.tar.xz
tar xf zig-linux-aarch64-0.13.0.tar.xz
sudo mv zig-linux-aarch64-0.13.0 /opt/zig
# Add to PATH as above
```

### Rust Toolchain

Ensure Rust 1.75+ is installed:

```bash
rustc --version  # Should be 1.75.0 or higher
```

## Building with Zig Kernels

### Development Build

```bash
# Using the build script (recommended)
./scripts/build_with_zig.sh

# Or manually with cargo
cargo build --features zig-kernels

# Verify Zig kernels are linked
nm target/debug/engram-cli | grep engram_vector_similarity
# Should output FFI function symbols
```

### Production Build

```bash
# Release build with optimizations
./scripts/build_with_zig.sh release

# Or manually
cargo build --release --features zig-kernels

# Verify release binary includes Zig kernels
ldd target/release/engram-cli  # Linux
otool -L target/release/engram-cli  # macOS
```

### Build Verification

Run differential tests to ensure Zig kernels produce identical results to Rust baseline:

```bash
# Run all differential tests
cargo test --features zig-kernels --test zig_differential

# Expected output: All tests pass with zero divergence
```

Run performance regression benchmarks:

```bash
# Verify performance improvements
./scripts/benchmark_regression.sh

# Expected: 15-35% improvement on target operations
```

## Configuration

### Arena Allocator Settings

Zig kernels use thread-local arena allocators for scratch space. Configure capacity based on workload:

#### Environment Variables

```bash
# Set arena size per thread (in bytes)
export ENGRAM_ARENA_SIZE=2097152  # 2MB (default: 1MB)

# Set overflow behavior: panic, error, fallback
export ENGRAM_ARENA_OVERFLOW=error  # Default: error
```

#### Overflow Strategies

- **panic**: Abort process on overflow (development/testing only)
- **error**: Return error and log warning (recommended for production)
- **fallback**: Attempt fallback to system allocator (experimental)

#### Runtime Configuration

If using the programmatic API:

```rust
use engram::zig_kernels::{configure_arena, OverflowStrategy};

// Configure arena before first kernel invocation
configure_arena(2, OverflowStrategy::ErrorReturn);  // 2MB per thread
```

### Sizing Guidelines

Choose arena size based on embedding dimensions and query batch sizes:

| Workload Type | Embedding Dim | Batch Size | Recommended Arena |
|---------------|--------------|------------|-------------------|
| Light | 384 | 100 | 1 MB (default) |
| Medium | 768 | 500 | 2 MB |
| Heavy | 768 | 1000 | 4 MB |
| Batch Processing | 1536 | 2000 | 8 MB |

**Formula**: `arena_size >= (embedding_dim * 4 bytes * batch_size * 2) + overhead`

The 2x multiplier accounts for intermediate calculations.

### Thread Configuration

Zig kernels scale linearly with threads (each has an independent arena):

```bash
# Set thread pool size
export RAYON_NUM_THREADS=8  # Match CPU core count

# Verify thread count at runtime
./target/release/engram-cli config get concurrency.num_threads
```

**Best Practice**: Set `RAYON_NUM_THREADS` to physical core count, not hyperthreads.

### CPU Feature Detection

Verify SIMD support on your platform:

#### x86_64 (AVX2 Required)

```bash
# Linux
grep avx2 /proc/cpuinfo

# macOS
sysctl -a | grep machdep.cpu.features | grep AVX2
```

If AVX2 is not available, Zig kernels fall back to scalar implementations (slower).

#### ARM64 (NEON Standard)

```bash
# NEON is standard on all ARMv8+ processors
# Verify ARM architecture version
uname -m  # Should output: aarch64 or arm64
```

## Deployment

### Production Deployment Checklist

Before deploying to production:

- [ ] Zig 0.13.0 installed on all production nodes
- [ ] Build with `--features zig-kernels` succeeds
- [ ] All differential tests pass (zero correctness regressions)
- [ ] Regression benchmarks show expected improvements (15-35%)
- [ ] Arena size configured for expected workload
- [ ] CPU features verified (AVX2 on x86_64, NEON on ARM64)
- [ ] Monitoring and alerting configured (see below)
- [ ] Rollback procedure documented and tested
- [ ] Gradual rollout plan prepared (canary -> production)

### Deployment Strategies

#### Option 1: Canary Deployment

1. Deploy to 5% of production traffic (canary instances)
2. Monitor for 24 hours:
   - Arena overflow rate should be <0.1%
   - Performance improvements match benchmarks
   - No increase in error rates
3. Gradually increase to 25%, 50%, 100%

#### Option 2: Blue-Green Deployment

1. Deploy Zig-enabled build to green environment
2. Run smoke tests and performance validation
3. Switch traffic from blue to green
4. Monitor for 1 hour before decommissioning blue

#### Option 3: Feature Flag

If your deployment supports feature flags:

```rust
// Disable Zig kernels at runtime
if config.get("zig_kernels_enabled") {
    use zig_kernels::vector_similarity;
} else {
    use rust_baseline::vector_similarity;
}
```

## Monitoring

### Key Metrics

Monitor these metrics in production:

#### Arena Utilization

```bash
# Query arena statistics via API (if exposed)
curl http://localhost:7432/internal/zig/arena_stats

# Expected response:
{
  "total_resets": 1234567,
  "total_overflows": 0,
  "max_high_water_mark": 1572864  // bytes
}
```

**Alert if**: `total_overflows > 0` or `max_high_water_mark > arena_size * 0.9`

#### Performance Metrics

Track kernel execution times:

```rust
use std::time::Instant;

let start = Instant::now();
let scores = vector_similarity(&query, &candidates);
let duration = start.elapsed();

// Log or export to metrics system
metrics.record_histogram("zig_kernel.vector_similarity", duration);
```

**Alert if**: p99 latency exceeds baseline expectations:

| Operation | Baseline (Rust) | Target (Zig) | Alert Threshold |
|-----------|----------------|--------------|-----------------|
| Vector Similarity (768d) | 2.3 us | 1.7 us | > 2.0 us |
| Spreading Activation (1000n) | 145 us | 95 us | > 120 us |
| Decay Calculation (10k) | 89 us | 65 us | > 80 us |

### Grafana Dashboards

Example Prometheus queries for monitoring:

```promql
# Arena overflow rate
rate(engram_zig_arena_overflows_total[5m])

# Arena high-water mark (per thread)
engram_zig_arena_high_water_mark_bytes

# Kernel execution time (p99)
histogram_quantile(0.99,
  rate(engram_zig_kernel_duration_seconds_bucket[5m])
)
```

### Recommended Alerts

Configure alerts for production:

1. **Arena overflow rate > 1%**: Increase arena size or investigate large allocations
2. **Similarity query p99 > 3us**: Performance regression or resource contention
3. **Spreading activation p99 > 150us**: Graph size or connectivity issues
4. **Zig kernel errors > 0.1%**: FFI boundary issues or memory corruption

## Performance Tuning

### Identifying Bottlenecks

If performance does not match expectations:

#### 1. Verify SIMD Usage

```bash
# Check that SIMD instructions are used
objdump -d target/release/engram-cli | grep -E 'vfmadd|vmulps|vaddps'  # AVX2
objdump -d target/release/engram-cli | grep -E 'fmla|fmul|fadd'        # NEON

# If no SIMD instructions found, verify CPU features
```

#### 2. Profile Arena Overhead

```bash
# Enable arena metrics tracking
export ENGRAM_ARENA_METRICS=1

# Run workload and check high-water marks
./target/release/engram-cli benchmark --workload vector_similarity

# Output will include arena utilization percentage
```

#### 3. Check Thread Contention

```bash
# Profile with perf (Linux)
perf record -g ./target/release/engram-cli benchmark
perf report

# Look for lock contention or false sharing
```

### Tuning Strategies

#### Arena Size Optimization

If high-water mark approaches arena capacity:

```bash
# Increase arena size incrementally
export ENGRAM_ARENA_SIZE=4194304  # 4MB

# Retest and check overflow rate
```

If high-water mark is much smaller than arena:

```bash
# Decrease to save memory
export ENGRAM_ARENA_SIZE=524288  # 512KB
```

#### Thread Count Optimization

```bash
# Benchmark with different thread counts
for threads in 4 8 16 32; do
  export RAYON_NUM_THREADS=$threads
  ./scripts/benchmark_regression.sh
done

# Use thread count with best throughput
```

#### NUMA Considerations

On multi-socket systems:

```bash
# Pin threads to NUMA nodes
numactl --cpunodebind=0 --membind=0 ./target/release/engram-cli

# Verify arena allocations happen on local node
```

## Troubleshooting

### Build Failures

#### Error: Zig compiler not found

```
error: could not execute process `zig` (never executed)
Caused by: No such file or directory
```

**Solution**: Install Zig 0.13.0 and ensure it's in PATH:

```bash
which zig  # Should output: /usr/local/bin/zig or similar
zig version  # Should output: 0.13.0
```

#### Error: Zig version mismatch

```
WARNING: Recommended Zig version is 0.13.0, found 0.12.0
```

**Solution**: Upgrade to Zig 0.13.0:

```bash
# macOS
brew upgrade zig

# Linux - download from ziglang.org
```

#### Error: cargo build failed in build.rs

```
error: failed to run custom build command for `engram-core`
```

**Solution**: Check Zig build output:

```bash
# Manually build Zig library
cd zig
zig build -Doptimize=ReleaseFast

# Check for errors
```

### Runtime Issues

#### Symptom: Arena overflow warnings in logs

```
WARN Arena overflow, falling back to system allocator
```

**Diagnosis**: Arena capacity insufficient for workload

**Solution**: Increase arena size:

```bash
export ENGRAM_ARENA_SIZE=4194304  # 4MB
./target/release/engram-cli restart
```

**Prevention**: Monitor `max_high_water_mark` and size arenas to 2x peak usage.

#### Symptom: Performance worse than baseline

```
REGRESSION: vector_similarity_768d_1000c is 12.5% slower than baseline
```

**Diagnosis**: Possible causes:
1. SIMD not enabled (missing AVX2/NEON)
2. Arena size too small (frequent overflows)
3. Thread contention or lock thrashing
4. Memory pressure (swapping)

**Solution**:

```bash
# 1. Verify CPU features
cat /proc/cpuinfo | grep avx2  # Should have output

# 2. Check arena overflow rate
curl http://localhost:7432/internal/zig/arena_stats
# total_overflows should be 0

# 3. Profile for contention
perf record -g ./target/release/engram-cli benchmark
perf report  # Look for lock contention

# 4. Check memory pressure
free -h  # Ensure sufficient free memory
```

#### Symptom: Numerical divergence from Rust baseline

```
FAIL: Differential test failed - Zig result differs from Rust baseline
Expected: 0.8765432
Got: 0.8765431
```

**Diagnosis**: Floating-point precision differences (usually acceptable)

**Solution**: Check if divergence is within tolerance:

```bash
# Differential tests allow small epsilon (1e-6)
# If delta > epsilon, file a bug report with reproduction
```

### Validation

#### Test Zig kernels are active

```rust
#[cfg(feature = "zig-kernels")]
{
    println!("Zig kernels enabled");
}

#[cfg(not(feature = "zig-kernels"))]
{
    println!("Using Rust baseline implementations");
}
```

#### Compare performance

```bash
# Benchmark with Zig kernels
cargo bench --features zig-kernels --bench regression

# Benchmark without (Rust baseline)
cargo bench --bench regression

# Compare results
```

#### Verify correctness

```bash
# Run all differential tests
cargo test --features zig-kernels --test zig_differential

# Expected: All tests pass with zero divergence
```

## Production Operations

### Health Checks

```bash
# Basic health check
curl http://localhost:7432/health

# Zig kernel-specific health
curl http://localhost:7432/internal/zig/status

# Expected response:
{
  "status": "healthy",
  "zig_kernels_enabled": true,
  "arena_overflows": 0,
  "last_kernel_invocation": "2025-10-25T10:30:00Z"
}
```

### Log Monitoring

Key log messages to monitor:

```
INFO  Zig kernels initialized: vector_similarity, spreading_activation, decay_calculation
WARN  Arena overflow detected on thread 4, consider increasing ENGRAM_ARENA_SIZE
ERROR Zig kernel invocation failed: OutOfMemory
```

### Incident Response

If Zig kernels cause production issues:

1. **Immediate**: Follow [Rollback Procedures](./zig_rollback_procedures.md)
2. **Investigation**: Capture logs, metrics, and reproduction steps
3. **Escalation**: File incident report with Engram team
4. **Post-mortem**: Schedule review and update documentation

## See Also

- [Rollback Procedures](./zig_rollback_procedures.md) - Emergency and gradual rollback strategies
- [Architecture Documentation](../internal/zig_architecture.md) - Internal design for maintainers
- [Performance Regression Guide](../internal/performance_regression_guide.md) - Benchmarking framework
- [Profiling Results](../internal/profiling_results.md) - Hotspot analysis and kernel selection

## Support

For production support:

- Check troubleshooting guide above
- Review [GitHub Issues](https://github.com/orchard9/engram/issues) for known issues
- File new issues with reproduction steps and system information
- Contact engineering team for critical production incidents
