# GPU Testing Requirements and Validation Strategy

**Document Version**: 1.0
**Last Updated**: 2025-10-26
**Status**: Production Critical

## Executive Summary

This document defines the requirements, constraints, and procedures for validating Engram's GPU acceleration features. GPU-specific tests cannot run in standard CI environments due to hardware dependencies and must be executed manually on GPU-enabled hardware before production deployment.

## Critical Constraint: GPU Tests Cannot Run in CI

### Why GPU Tests Are Skipped

All GPU-specific integration tests include this guard:

```rust
#[test]
#[cfg(all(feature = "gpu", cuda_available))]
fn test_gpu_feature() {
    if !cuda::is_available() {
        println!("GPU not available, skipping test");
        return;
    }
    // Test code never executes in CI
}

```

**Reality**: Standard CI runners do not have CUDA-capable GPUs. The `cuda_available` flag is never set, and `cuda::is_available()` returns false. This means:

- GPU tests compile but do not execute on GPU hardware in CI

- Test passing in CI does not validate GPU functionality

- Manual validation on GPU hardware is required for production readiness

### CI vs. GPU Hardware Testing

| Test Type | CI Environment | GPU Hardware Required |
|-----------|----------------|----------------------|
| CPU-only tests | ✓ Runs in CI | No |
| GPU fallback tests | ✓ Runs in CI (CPU mode) | No |
| GPU differential tests | ⚠️ Compiles but skips | Yes |
| GPU integration tests | ⚠️ Compiles but skips | Yes |
| Sustained load tests | ❌ Marked #[ignore] | Yes |

## Required GPU Hardware Specifications

### Minimum Requirements

For validation testing:

- **GPU**: NVIDIA Tesla T4, A10, or better

- **CUDA Compute Capability**: 7.0+ (Volta architecture)

- **VRAM**: 16GB minimum

- **CUDA Toolkit**: 11.0 or later

- **Driver Version**: 450.80.02 or later

### Production Recommendations

For production deployment:

- **GPU**: NVIDIA A100, H100, or L40S

- **CUDA Compute Capability**: 8.0+ (Ampere/Hopper)

- **VRAM**: 40GB or more for multi-tenant workloads

- **CUDA Toolkit**: 12.0 or later

- **Driver Version**: 525.60.13 or later

### Cloud GPU Instance Options

| Provider | Instance Type | GPU | VRAM | Suitable For |
|----------|---------------|-----|------|--------------|
| AWS | `p3.2xlarge` | Tesla V100 | 16GB | Validation testing |
| AWS | `p4d.24xlarge` | A100 | 40GB | Production validation |
| GCP | `a2-highgpu-1g` | A100 | 40GB | Production validation |
| Azure | `NC6s_v3` | V100 | 16GB | Validation testing |
| Azure | `ND96asr_v4` | A100 | 40GB | Production validation |

## Manual GPU Test Execution Procedure

### Prerequisites

1. **Verify CUDA Installation**:

   ```bash
   nvidia-smi
   nvcc --version
   ```

2. **Build with GPU Support**:

   ```bash
   cargo build --release --features gpu
   ```

3. **Verify GPU Detection**:

   ```bash
   cargo test --features gpu test_cuda_device_detection -- --nocapture
   ```

### Test Execution Phases

#### Phase 1: Foundation Tests (5-10 minutes)

Execute GPU acceleration and differential tests:

```bash
# Run GPU acceleration foundation tests
cargo test --features gpu gpu_acceleration_test -- --nocapture

# Run CPU-GPU differential tests
cargo test --features gpu gpu_differential_cosine -- --nocapture
cargo test --features gpu gpu_differential_hnsw -- --nocapture
cargo test --features gpu gpu_differential_spreading -- --nocapture

```

**Acceptance Criteria**:

- All differential tests pass with <1e-6 divergence

- GPU memory allocations succeed

- No CUDA errors in logs

#### Phase 2: Integration Tests (15-20 minutes)

Execute main GPU integration test suite:

```bash
cargo test --features gpu gpu_integration -- --nocapture

```

**Acceptance Criteria**:

- All 12 integration tests pass

- Multi-tenant isolation verified

- GPU memory pressure handled gracefully

- CPU fallback works correctly

#### Phase 3: Sustained Load Test (60+ minutes)

Execute long-running throughput validation:

```bash
cargo test --features gpu test_sustained_throughput -- --ignored --nocapture

```

**Acceptance Criteria**:

- Throughput ≥10,000 ops/sec sustained

- Performance degradation <10% over duration

- No memory leaks detected

- GPU temperature remains stable

#### Phase 4: Production Workload Validation (30-60 minutes)

Execute realistic workload tests:

```bash
cargo test --features gpu test_production_workload -- --nocapture
cargo test --features gpu test_multi_tenant_security -- --nocapture
cargo test --features gpu test_confidence_calibration -- --nocapture

```

**Acceptance Criteria**:

- Power-law degree distribution handled efficiently

- GPU speedup >3x on realistic graphs

- Multi-tenant security boundaries enforced

- Confidence scores remain calibrated

### Logging and Diagnostics

After each test phase:

```bash
./scripts/engram_diagnostics.sh
cat tmp/engram_diagnostics.log

```

Monitor GPU-specific metrics:

```bash
nvidia-smi dmon -s pucvmet -i 0 -c 60

```

## Test Coverage Requirements

### P0: Blocking Production Deployment

These tests MUST pass on GPU hardware before production:

1. **GPU Detection and Initialization**
   - Verify CUDA runtime available
   - Detect GPU compute capability
   - Allocate GPU memory successfully

2. **CPU-GPU Differential Correctness**
   - Cosine similarity divergence <1e-6
   - HNSW search results identical
   - Spreading activation within tolerance

3. **Multi-Tenant Security Isolation**
   - Tenant A cannot access Tenant B GPU buffers
   - Resource exhaustion does not affect other tenants
   - GPU memory is zeroed between tenants

4. **Graceful Fallback**
   - CPU fallback on GPU OOM
   - CPU fallback on CUDA errors
   - Performance degradation detected and logged

### P1: Should Validate Before Production

5. **Sustained Throughput**
   - 60-minute sustained load at 10K ops/sec
   - Performance degradation tracking
   - Memory leak detection

6. **Production Workload Patterns**
   - Power-law degree distribution (social graphs)
   - Dense semantic clustering (knowledge graphs)
   - Temporal correlation (financial networks)

7. **Confidence Score Calibration**
   - Statistical validation over 100K+ operations
   - Calibration drift tracking
   - Aggregation mathematics verification

### P2: Can Validate Post-Deployment

8. **Chaos Engineering**
   - GPU OOM injection
   - Driver failure simulation
   - Concurrent access conflicts

9. **Performance Regression**
   - Baseline SLI tracking
   - Percentile measurements (p50, p95, p99)
   - Automated regression detection

## Known Limitations

### GPU Test Execution

1. **CI Environment**: GPU tests do not run in standard CI
   - **Mitigation**: Manual GPU hardware validation required
   - **Risk**: GPU regressions not caught by CI
   - **Action**: Document GPU testing procedure in release checklist

2. **Sustained Load Tests**: Marked `#[ignore]` due to 60-minute runtime
   - **Mitigation**: Execute manually before production deployment
   - **Risk**: Performance degradation over time not validated
   - **Action**: Run 24-hour soak test in staging before production

3. **Mock GPU Interface**: Tests use mocks in CPU-only mode
   - **Mitigation**: Mock behavior validated against real GPU
   - **Risk**: Mock divergence from real GPU behavior
   - **Action**: Periodic validation of mock accuracy

### Production Deployment Constraints

1. **GPU Availability**: Not all deployment environments have GPUs
   - **Solution**: Hybrid executor with automatic CPU fallback
   - **Testing**: Verify CPU fallback maintains correctness

2. **Multi-Tenant Security**: GPU memory isolation requires careful management
   - **Solution**: Per-tenant memory allocation and zeroing
   - **Testing**: Security boundary validation tests

3. **Memory Pressure**: GPU OOM more constrained than CPU
   - **Solution**: Dynamic batch sizing and pressure detection
   - **Testing**: Memory pressure stress tests

## Production Deployment Checklist

Before deploying GPU-accelerated Engram to production:

- [ ] Execute Phase 1 foundation tests on target GPU hardware

- [ ] Execute Phase 2 integration tests successfully

- [ ] Run Phase 3 sustained load test for full 60 minutes

- [ ] Execute Phase 4 production workload validation

- [ ] Run 24-hour soak test in staging environment

- [ ] Validate multi-tenant security boundaries

- [ ] Verify confidence score calibration

- [ ] Document GPU temperature and throttling behavior

- [ ] Establish baseline performance SLIs

- [ ] Configure GPU monitoring and alerting

- [ ] Document fallback behavior and CPU requirements

- [ ] Create GPU operations runbook

## Operational Monitoring

### GPU Health Metrics

Monitor these metrics in production:

- **GPU Utilization**: Target 60-80% for cost efficiency

- **GPU Memory Usage**: Alert at 85% to prevent OOM

- **GPU Temperature**: Alert at 80°C, throttle at 83°C

- **CUDA Errors**: Alert on any error, failover to CPU

- **Fallback Rate**: Alert if >5% of operations fall back to CPU

- **Throughput**: Alert if <10K ops/sec sustained

### Diagnostic Commands

```bash
# GPU utilization monitoring
nvidia-smi dmon -s pucvmet -i 0 -c 60

# Detailed GPU metrics
nvidia-smi --query-gpu=timestamp,name,pci.bus_id,driver_version,utilization.gpu,utilization.memory,memory.total,memory.free,memory.used,temperature.gpu --format=csv -l 5

# CUDA error checking
dmesg | grep -i cuda

# Engram GPU diagnostics
./scripts/engram_diagnostics.sh

```

## References

- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)

- [NVIDIA GPU Architecture Documentation](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)

- [Engram GPU Acceleration Implementation](../explanation/gpu_acceleration.md)

- [Engram Deployment Guide](./deployment.md)

## Document History

| Version | Date | Changes | Author |
|---------|------|---------|--------|
| 1.0 | 2025-10-26 | Initial documentation of GPU testing requirements | Denise Gosnell |

---

**IMPORTANT**: This document must be reviewed and updated after each GPU-related milestone. Manual GPU validation is required before production deployment.
