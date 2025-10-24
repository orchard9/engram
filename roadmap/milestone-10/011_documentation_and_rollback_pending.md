# Task 011: Documentation and Rollback

**Duration:** 1 day
**Status:** Pending
**Dependencies:** 010 (Performance Regression Framework)

## Objectives

Create comprehensive documentation for operating Engram with Zig kernels in production, including deployment procedures, monitoring guidance, performance tuning, and rollback strategies. Documentation must enable operators to confidently deploy, monitor, and troubleshoot Zig kernel integration.

1. **Deployment guide** - Step-by-step Zig kernel activation
2. **Performance tuning** - Arena sizing and optimization strategies
3. **Monitoring integration** - Metrics collection and alerting
4. **Rollback procedures** - Safe degradation to Rust-only mode

## Dependencies

- Task 010 (Performance Regression Framework) - All technical work complete

## Deliverables

### Files to Create

1. `/docs/operations/zig_performance_kernels.md` - Complete operational guide
   - Deployment procedures
   - Configuration reference
   - Performance tuning guidelines
   - Troubleshooting guide

2. `/docs/operations/zig_rollback_procedures.md` - Rollback documentation
   - Emergency rollback steps
   - Gradual rollback strategies
   - Verification procedures

3. `/docs/internal/zig_architecture.md` - Architecture documentation
   - FFI boundary design
   - Memory management
   - SIMD implementation details
   - Performance characteristics

4. `/CHANGELOG.md` - Update with Milestone 10 changes
   - New features (Zig kernels)
   - Performance improvements
   - Breaking changes (if any)

### Files to Modify

1. `/README.md` - Add Zig kernel documentation links
   - Quick start with Zig kernels
   - Link to operations guide
   - Performance benchmarks

2. `/docs/operations/README.md` - Add Zig kernel operations index
   - Link to deployment guide
   - Link to rollback procedures

## Acceptance Criteria

1. Deployment guide enables operator to activate Zig kernels in production
2. Rollback procedures validated by actually performing rollback
3. Configuration reference documents all environment variables and flags
4. Troubleshooting guide covers common issues identified during testing
5. All documentation reviewed for technical accuracy

## Implementation Guidance

### Deployment Guide Structure

```markdown
# Zig Performance Kernels - Operations Guide

## Overview

Engram includes optional Zig performance kernels that accelerate compute-intensive operations:
- **Vector similarity**: 15-25% faster cosine similarity calculations
- **Activation spreading**: 20-35% faster graph traversal
- **Memory decay**: 20-30% faster Ebbinghaus decay calculations

This guide covers deployment, configuration, monitoring, and troubleshooting.

## Prerequisites

- Zig 0.13.0 compiler installed
- Rust toolchain 1.75+
- x86_64 with AVX2 or ARM64 with NEON

### Installing Zig

```bash
# macOS
brew install zig

# Linux
wget https://ziglang.org/download/0.13.0/zig-linux-x86_64-0.13.0.tar.xz
tar xf zig-linux-x86_64-0.13.0.tar.xz
export PATH=$PATH:$PWD/zig-linux-x86_64-0.13.0
```

## Building with Zig Kernels

### Development Build

```bash
# Build with Zig kernels enabled
./scripts/build_with_zig.sh

# Or manually
cargo build --release --features zig-kernels
```

### Verification

```bash
# Run differential tests
cargo test --features zig-kernels

# Run regression benchmarks
./scripts/benchmark_regression.sh
```

## Configuration

### Arena Allocator Settings

Control memory pool size for kernel scratch space:

```bash
# Set arena size (default: 1MB per thread)
export ENGRAM_ARENA_SIZE=2097152  # 2MB in bytes

# Set overflow behavior (panic, error, fallback)
export ENGRAM_ARENA_OVERFLOW=error
```

### Feature Flags

Enable Zig kernels at runtime:

```rust
// In your application
use engram::zig_kernels;

// Configure arena
zig_kernels::configure_arena(2, OverflowStrategy::ErrorReturn);

// Use kernels through standard API
let scores = graph.find_similar(&query, 10); // Uses Zig kernel
```

## Monitoring

### Arena Metrics

Monitor arena usage to detect capacity issues:

```rust
let stats = zig_kernels::get_arena_stats();
println!("Arena overflows: {}", stats.total_overflows);
println!("High water mark: {} MB", stats.max_high_water_mark / 1_048_576);
```

### Performance Metrics

Track kernel execution times:

```rust
// Instrument critical paths
let start = Instant::now();
let scores = graph.find_similar(&query, 10);
let duration = start.elapsed();
metrics.record_similarity_query(duration);
```

### Recommended Alerts

- **Arena overflow rate > 1%**: Increase arena size
- **Similarity query p99 > 3us**: Investigate performance regression
- **Spreading activation p99 > 150us**: Check graph size or connectivity

## Performance Tuning

### Arena Sizing

Choose arena size based on workload:

| Workload | Embedding Dim | Arena Size |
|----------|--------------|------------|
| Light | 384 | 1 MB |
| Medium | 768 | 2 MB |
| Heavy | 1536 | 4 MB |

### Thread Count

Zig kernels scale linearly with threads (each has independent arena):

```bash
# Set thread pool size
export RAYON_NUM_THREADS=8
```

### CPU Features

Verify SIMD support:

```bash
# x86_64: Check for AVX2
cat /proc/cpuinfo | grep avx2

# ARM64: NEON is standard on ARMv8
```

## Troubleshooting

### Build Failures

**Error**: `Zig compiler not found`
**Solution**: Install Zig 0.13.0 and ensure it's in PATH

**Error**: `cargo build failed in build.rs`
**Solution**: Check Zig version matches 0.13.0

### Runtime Issues

**Symptom**: Arena overflow warnings in logs
**Solution**: Increase `ENGRAM_ARENA_SIZE` or reduce embedding dimensions

**Symptom**: Performance worse than baseline
**Solution**: Verify AVX2/NEON support, check arena sizing, profile for contention

### Validation

**Test Zig kernels are active**:

```rust
#[cfg(feature = "zig-kernels")]
{
    println!("Zig kernels enabled");
}
```

**Compare performance**:

```bash
# With Zig kernels
cargo bench --features zig-kernels

# Without (Rust baseline)
cargo bench
```

## Production Deployment Checklist

- [ ] Zig 0.13.0 installed on all production nodes
- [ ] Build with `--features zig-kernels` succeeds
- [ ] All differential tests pass
- [ ] Regression benchmarks show expected improvements
- [ ] Arena size configured for workload
- [ ] Monitoring and alerting configured
- [ ] Rollback procedure documented and tested
- [ ] Gradual rollout plan (canary -> production)

## See Also

- [Rollback Procedures](./zig_rollback_procedures.md)
- [Architecture Documentation](../internal/zig_architecture.md)
- [Performance Benchmarks](../benchmarks/)
```

### Rollback Procedures

```markdown
# Zig Kernels - Rollback Procedures

## Emergency Rollback

If critical issues arise with Zig kernels, follow this procedure for immediate rollback:

### Step 1: Rebuild without Zig feature

```bash
# Stop the service
systemctl stop engram

# Rebuild Rust-only binary
cargo build --release
# Note: omit --features zig-kernels

# Deploy new binary
cp target/release/engram /usr/local/bin/engram

# Restart service
systemctl start engram
```

**Recovery Time Objective (RTO)**: 5-10 minutes

### Step 2: Verify Rollback

```bash
# Check logs for Zig kernel messages (should be absent)
journalctl -u engram | grep -i "zig"

# Run health check
curl http://localhost:8080/health

# Monitor performance (should return to baseline)
```

### Step 3: Incident Response

1. Document the issue (symptoms, logs, metrics)
2. File incident report with reproduction steps
3. Notify team via incident channel
4. Schedule post-mortem

## Gradual Rollback

For non-critical issues, use gradual rollback:

### Option 1: Feature Flag

If feature flags are implemented:

```rust
// Disable Zig kernels at runtime
config.set("zig_kernels_enabled", false);
```

### Option 2: Canary Rollback

1. Roll back canary instances first
2. Monitor for 1 hour
3. Roll back remaining instances in waves

### Option 3: Traffic Shifting

If load balancer supports gradual traffic shifting:

1. Route 90% traffic to Rust-only instances
2. Monitor for stability
3. Gradually increase to 100%

## Verification After Rollback

### Functional Verification

```bash
# Run integration tests against rolled-back system
cargo test --test integration

# Verify API responses
./scripts/smoke_test.sh
```

### Performance Verification

```bash
# Check that performance returns to baseline
cargo bench --bench baseline_performance

# Verify no Zig-related errors in logs
journalctl -u engram --since "10 minutes ago" | grep -i error
```

## Common Rollback Scenarios

### Scenario 1: Arena Overflows Causing Errors

**Symptoms**: High error rate, arena overflow warnings

**Rollback Steps**:
1. Attempt to increase arena size first: `ENGRAM_ARENA_SIZE=8388608`
2. If errors persist, perform emergency rollback
3. Investigate required arena size for workload

### Scenario 2: Performance Regression

**Symptoms**: Increased latency, degraded throughput

**Rollback Steps**:
1. Verify with regression benchmarks
2. Check CPU features (AVX2/NEON availability)
3. If confirmed, perform gradual rollback
4. Investigate performance regression root cause

### Scenario 3: Numerical Divergence

**Symptoms**: Incorrect results, failing differential tests

**Rollback Steps**:
1. Immediate emergency rollback (correctness issue)
2. Capture failing test cases
3. File critical bug report
4. Investigate numerical accuracy issue

## Rollback Testing

Regularly test rollback procedures in staging:

```bash
# Monthly rollback drill
./scripts/deploy_with_zig.sh staging
sleep 300  # Let it run for 5 minutes
./scripts/rollback_to_rust.sh staging

# Verify system health
./scripts/health_check.sh staging
```

## Decision Matrix

| Issue Severity | Action | Timeline |
|---------------|--------|----------|
| Critical (correctness) | Emergency rollback | Immediate |
| High (performance) | Gradual rollback | Within 1 hour |
| Medium (errors <1%) | Investigate, then decide | Within 1 day |
| Low (warnings) | Monitor, tune configuration | Within 1 week |

## Post-Rollback Actions

1. **Root cause analysis**: Identify why rollback was necessary
2. **Fix and validate**: Address issue in development environment
3. **Re-deployment plan**: Plan safe re-introduction of Zig kernels
4. **Documentation update**: Document lessons learned

## Contact

For rollback assistance, contact:
- On-call engineer: [on-call rotation]
- Engram team lead: [team contact]
- Emergency escalation: [escalation path]
```

### Architecture Documentation

Document internal design for maintainers:

```markdown
# Zig Kernels - Architecture

## FFI Boundary Design

### Memory Ownership Model

- **Caller allocates**: Rust allocates all buffers passed to Zig
- **Zig computes**: Zig kernels write results to caller-provided buffers
- **No ownership transfer**: Zig never frees Rust-allocated memory

### Function Signatures

All FFI functions follow this pattern:

```zig
export fn engram_kernel(
    inputs: [*]const InputType,  // Read-only inputs
    outputs: [*]OutputType,      // Write-only outputs
    count: usize,                // Element counts
) void;  // Never return errors (handle gracefully)
```

## Memory Management

### Arena Allocator

Thread-local arena allocators provide scratch space:

- **Allocation**: Bump-pointer (O(1))
- **Deallocation**: Bulk reset after kernel execution
- **Capacity**: Configurable per-workload (default 1MB)
- **Isolation**: Thread-local, no cross-thread access

### Overflow Handling

When arena exhausted:
1. Log warning
2. Return zeros/default values
3. Increment overflow counter

## SIMD Implementation

### Platform Detection

Runtime detection of CPU features:

```zig
if (std.Target.x86.featureSetHas(builtin.cpu.features, .avx2)) {
    // Use AVX2 path
} else {
    // Fall back to scalar
}
```

### Vector Width

- **x86_64 (AVX2)**: Process 8 floats per instruction
- **ARM64 (NEON)**: Process 4 floats per instruction
- **Scalar fallback**: Process 1 float per instruction

## Performance Characteristics

### Expected Improvements

| Operation | Rust Baseline | Zig Kernel | Improvement |
|-----------|--------------|------------|-------------|
| Vector Similarity (768d) | 2.3 us | 1.7 us | 25% |
| Spreading (1000n) | 145 us | 95 us | 35% |
| Decay (10k memories) | 89 us | 65 us | 27% |

### Bottlenecks

- **Vector similarity**: Memory bandwidth (loading embeddings)
- **Spreading activation**: Cache locality (random graph access)
- **Decay calculation**: Exponential function (consider lookup table)

## Testing Strategy

### Differential Testing

Every Zig kernel has corresponding differential test:

```rust
proptest! {
    fn zig_matches_rust(inputs) {
        assert_eq!(zig_kernel(inputs), rust_kernel(inputs));
    }
}
```

### Performance Testing

Regression benchmarks prevent performance degradation:

```rust
fn regression_benchmark() {
    let baseline = load_baseline();
    let current = measure_performance();
    assert!(current < baseline * 1.05);  // Max 5% regression
}
```

## Future Optimizations

Potential improvements for future milestones:

1. **FMA instructions**: Use fused multiply-add for dot products
2. **Lookup tables**: Replace exp() with LUT for decay
3. **Edge reordering**: Sort graph edges for cache locality
4. **GPU offload**: Port kernels to CUDA (Milestone 14)
```

## Testing Approach

1. **Documentation review**
   - Technical accuracy validation
   - Clarity and completeness check
   - Operator walkthrough

2. **Rollback testing**
   - Perform actual rollback in staging
   - Verify procedures work as documented
   - Time each step

3. **Deployment testing**
   - Follow deployment guide in clean environment
   - Verify all prerequisites
   - Validate monitoring integration

## Integration Points

- **Task 012 (Final Validation)** - Documentation reviewed as part of UAT
- **All previous tasks** - Documentation references specific implementation details

## Notes

- Use concrete examples in troubleshooting guide
- Include expected vs. actual outputs for common issues
- Provide decision trees for rollback scenarios
- Document lessons learned from integration testing
- Keep deployment checklist concise and actionable
