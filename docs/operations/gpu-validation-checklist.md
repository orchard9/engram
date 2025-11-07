# GPU Validation Checklist

**Document Version**: 1.0  
**Last Updated**: 2024-11-01  
**Status**: Production Critical

## Overview

This checklist provides a systematic approach to validating GPU functionality in Engram. Use this before production deployments and after significant GPU-related changes.

## Quick Start

```bash
# Run quick smoke tests (5 minutes)
./scripts/validate_gpu.sh quick

# Run all validation phases except sustained tests (45-60 minutes)
./scripts/validate_gpu.sh all

# Run specific phase
./scripts/validate_gpu.sh foundation
./scripts/validate_gpu.sh integration
./scripts/validate_gpu.sh production
./scripts/validate_gpu.sh sustained  # 60+ minutes
```

## Pre-Validation Checklist

### Hardware Requirements

- [ ] NVIDIA GPU with CUDA Compute Capability 7.0+ (check with `nvidia-smi`)
- [ ] Minimum 16GB VRAM (production: 40GB+ recommended)
- [ ] CUDA Toolkit 11.0+ installed (`nvcc --version`)
- [ ] NVIDIA Driver 450.80.02+ (`nvidia-smi` shows driver version)

### Environment Setup

- [ ] GPU feature enabled in build: `cargo build --features gpu`
- [ ] No other GPU-intensive processes running
- [ ] Sufficient disk space for logs (1GB+)
- [ ] Temperature monitoring available

## Phase 1: Foundation Tests (5-10 minutes)

### GPU Detection and Initialization
- [ ] CUDA runtime available
- [ ] GPU device enumeration successful
- [ ] Memory allocation tests pass
- [ ] Kernel compilation successful

### Differential Correctness
- [ ] Cosine similarity: CPU-GPU divergence < 1e-6
- [ ] HNSW scoring: Results match exactly
- [ ] Spreading activation: Within tolerance bounds
- [ ] Batch operations: Consistent results

### Command
```bash
./scripts/validate_gpu.sh foundation
```

### Expected Output
```
✓ GPU detection successful
✓ GPU acceleration foundation passed
✓ CPU-GPU differential cosine similarity passed
✓ CPU-GPU differential HNSW passed
✓ CPU-GPU differential spreading passed
```

## Phase 2: Integration Tests (15-20 minutes)

### Core Integration
- [ ] Multi-threaded GPU access safe
- [ ] Memory management correct
- [ ] Error handling graceful
- [ ] Performance tracking accurate

### Hybrid Executor
- [ ] Automatic GPU/CPU dispatch working
- [ ] Fallback on OOM functional
- [ ] Batch size adaptation correct
- [ ] Performance metrics collected

### Command
```bash
./scripts/validate_gpu.sh integration
```

### Expected Results
- All 12 integration tests pass
- No CUDA errors in logs
- Memory usage stable
- Performance metrics reasonable

## Phase 3: Production Workload (30-60 minutes)

### Multi-Tenant Security
- [ ] Resource isolation verified
- [ ] Memory boundaries enforced
- [ ] Fairness maintained under load
- [ ] No cross-tenant data leakage

### Workload Patterns
- [ ] Power-law graphs: 3x+ speedup
- [ ] Dense clustering: Efficient processing
- [ ] Mixed workloads: Stable performance
- [ ] Large batches: No OOM errors

### Confidence Calibration
- [ ] Statistical validation passes
- [ ] Calibration stable over time
- [ ] Aggregation math correct
- [ ] No drift detected

### Command
```bash
./scripts/validate_gpu.sh production
```

### Success Criteria
- GPU speedup > 3x on realistic workloads
- Multi-tenant isolation perfect
- Confidence scores calibrated
- All chaos tests handled gracefully

## Phase 4: Sustained Load Tests (60+ minutes)

### Long-Running Stability
- [ ] 60-minute sustained throughput > 10K ops/sec
- [ ] Performance degradation < 10%
- [ ] No memory leaks detected
- [ ] Temperature stable (< 80°C)

### Monitoring During Test
```bash
# In separate terminal
watch -n 1 nvidia-smi

# Or detailed metrics
nvidia-smi dmon -s pucvmet -i 0
```

### Command
```bash
./scripts/validate_gpu.sh sustained
```

### Expected Metrics
- GPU Utilization: 60-80%
- Memory Usage: < 85%
- Temperature: < 80°C
- Power: Within TDP limits

## Performance Benchmarks

### Run Benchmarks
```bash
./scripts/validate_gpu.sh benchmarks
```

### Expected Performance

| Operation | Batch Size | Expected Throughput |
|-----------|------------|-------------------|
| Cosine Similarity | 10K vectors | > 50K/sec |
| HNSW Scoring | 1K candidates | > 20K/sec |
| Spreading Activation | 100K edges | > 10K/sec |

## Post-Validation Diagnostics

### Generate Report
```bash
# Check diagnostics
./scripts/engram_diagnostics.sh
cat tmp/engram_diagnostics.log

# Review validation logs
ls -la tmp/gpu_validation/
```

### Key Metrics to Review
- Peak GPU utilization
- Maximum memory usage
- Temperature profile
- Error/warning count
- Fallback frequency

## Production Deployment Sign-Off

### Required for Production
- [ ] All foundation tests pass
- [ ] All integration tests pass
- [ ] Production workload validation complete
- [ ] Multi-tenant security verified
- [ ] Sustained load test successful (staging)
- [ ] Performance benchmarks acceptable
- [ ] No critical warnings in logs
- [ ] GPU monitoring configured
- [ ] Fallback behavior documented
- [ ] Operations runbook updated

### GPU-Specific Monitoring Setup
```bash
# Prometheus GPU exporter
docker run -d --gpus all \
  -p 9835:9835 \
  --name gpu-exporter \
  mindprince/nvidia-gpu-prometheus-exporter:0.1.0

# Add to monitoring stack
# gpu_temperature_celsius
# gpu_utilization_percent  
# gpu_memory_utilization_percent
# gpu_power_draw_watts
```

## Troubleshooting Common Issues

### CUDA Not Available
```bash
# Check CUDA installation
ls -la /usr/local/cuda/
ldconfig -p | grep cuda

# Verify GPU visible
lspci | grep -i nvidia
```

### Build Failures
```bash
# Clean and rebuild
cargo clean
rm -rf target/
cargo build --features gpu --verbose
```

### Test Failures
1. Check GPU memory availability: `nvidia-smi`
2. Review detailed logs: `cat tmp/gpu_validation/*.log`
3. Run individual failing test with `--nocapture`
4. Check for thermal throttling

### Performance Issues
1. Verify GPU not throttling: `nvidia-smi -q -d PERFORMANCE`
2. Check for other GPU processes: `nvidia-smi pmon -c 1`
3. Review batch sizes in logs
4. Confirm CUDA optimization level

## Emergency Procedures

### GPU Failure in Production
1. Hybrid executor will automatically fall back to CPU
2. Monitor fallback rate in metrics
3. Alert on > 5% fallback rate
4. Scale CPU resources if needed

### GPU Memory Exhaustion
1. System handles gracefully with CPU fallback
2. Review batch size configuration
3. Consider memory pressure settings
4. Monitor multi-tenant usage patterns

## References

- [GPU Testing Requirements](./gpu_testing_requirements.md)
- [GPU Architecture Guide](../reference/gpu-architecture.md)
- [GPU Deployment Guide](./gpu-deployment.md)
- [GPU Troubleshooting](./gpu-troubleshooting.md)

## Validation History

| Date | Version | Tester | Hardware | Result | Notes |
|------|---------|--------|----------|--------|-------|
| | | | | | |

---

**Remember**: GPU validation cannot be fully automated in CI. Manual validation on GPU hardware is required before each production deployment.
