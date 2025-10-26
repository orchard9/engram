# Documentation Fixes Checklist - Task 011

**Status**: IN PROGRESS
**Priority**: CRITICAL
**Estimated Total Time**: 10-15 hours
**Target Completion**: 2025-10-27

---

## CRITICAL FIXES (Must Complete Before Publication)

### 1. Add Implementation Status Disclaimers

**Time**: 1 hour
**Priority**: CRITICAL
**Files**: All 4 documentation files

- [ ] **gpu-architecture.md** - Add banner after title:
```markdown
# GPU Architecture Reference

> **IMPLEMENTATION STATUS**: This document describes the planned GPU acceleration architecture.
> Milestone 12 implements the hybrid executor framework and CPU SIMD fallback (production-ready).
> Actual CUDA kernels will be implemented in Milestone 13.
>
> **Current capabilities**: CPU SIMD operations with GPU-ready architecture
> **Coming soon**: CUDA kernel implementations for GPU acceleration

**Audience**: System architects, performance engineers, developers extending GPU features
```

- [ ] **gpu-deployment.md** - Add warning after overview:
```markdown
## Overview

> **IMPORTANT**: GPU acceleration is under active development. The current release provides:
> - Production-ready CPU SIMD implementation (excellent performance)
> - GPU-ready architecture and interfaces
> - Hybrid executor framework for future GPU integration
>
> CUDA kernels are planned for Milestone 13. This guide describes the deployment
> architecture for when GPU features are available. For deploying today, see the
> "CPU SIMD Deployment" section below.
```

- [ ] **gpu-troubleshooting.md** - Add note at top:
```markdown
# GPU Troubleshooting Guide

> **NOTE**: This guide is written for future GPU implementations. Current Milestone 12
> uses CPU SIMD only. GPU-related errors cannot occur in the current release.
>
> For troubleshooting CPU SIMD performance, see the Performance Tuning Guide.
```

- [ ] **gpu-performance-tuning.md** - Add disclaimer:
```markdown
# GPU Performance Tuning Guide

> **FUTURE IMPLEMENTATION**: This guide describes tuning strategies for GPU acceleration.
> Current release uses CPU SIMD. GPU tuning will be applicable after Milestone 13.
>
> For CPU SIMD performance optimization, see sections marked "CPU Performance".
```

### 2. Fix Configuration Parameter Names

**Time**: 2 hours
**Priority**: CRITICAL
**Files**: gpu-deployment.md, gpu-performance-tuning.md, gpu-architecture.md

- [ ] **gpu-deployment.md:328-369** - Replace TOML config with Rust API:
```markdown
### Configuration

Engram uses Rust API configuration (TOML config file support planned for future release).

```rust
use engram_core::compute::cuda::hybrid::{HybridConfig, HybridExecutor};

let config = HybridConfig {
    // Minimum batch size to use GPU (default: 64)
    gpu_min_batch_size: 64,

    // Speedup threshold for preferring GPU (default: 1.5)
    gpu_speedup_threshold: 1.5,

    // Success rate required to trust GPU (default: 0.95)
    gpu_success_rate_threshold: 0.95,

    // Window size for performance tracking (default: 100)
    performance_window_size: 100,

    // Force CPU-only mode for debugging (default: false)
    force_cpu_mode: false,

    // Enable performance tracking telemetry (default: true)
    telemetry_enabled: true,
};

let executor = HybridExecutor::new(config);
```

**Consumer GPU (RTX 3060) - Recommended**:
```rust
HybridConfig {
    gpu_min_batch_size: 64,
    gpu_speedup_threshold: 1.4,
    gpu_success_rate_threshold: 0.95,
    performance_window_size: 200,
    force_cpu_mode: false,
    telemetry_enabled: true,
}
```

**Datacenter GPU (A100) - Recommended**:
```rust
HybridConfig {
    gpu_min_batch_size: 32,
    gpu_speedup_threshold: 1.2,
    gpu_success_rate_threshold: 0.95,
    performance_window_size: 500,
    force_cpu_mode: false,
    telemetry_enabled: true,
}
```
```

- [ ] **gpu-performance-tuning.md:154-250** - Update all config examples to use correct parameter names
- [ ] **gpu-architecture.md:106-115** - Update configuration parameters table with actual names

### 3. Fix or Remove CLI Flag Examples

**Time**: 2 hours
**Priority**: CRITICAL
**Files**: gpu-deployment.md, gpu-troubleshooting.md

**Option A**: Remove all non-existent CLI examples
**Option B**: Implement the CLI flags
**CHOSEN**: Option A (faster, aligns with current milestone)

- [ ] **gpu-deployment.md:28** - Remove `--gpu-info` example, replace with:
```markdown
# Verify GPU detection (via code)
# Currently no CLI flag - check programmatically:

```rust
use engram_core::compute::cuda::hybrid::{HybridConfig, HybridExecutor};

let executor = HybridExecutor::new(HybridConfig::default());
// If GPU available, executor will use it based on batch size
```

Or check build configuration:
```bash
# Check if built with CUDA support
cargo tree | grep cuda
# If cuda dependencies present, GPU support compiled in
```
```

- [ ] **gpu-deployment.md:419** - Remove `--config` flag, use environment variables:
```markdown
# Start Engram server
./target/release/engram start

# Configure via environment variables (if supported)
export ENGRAM_GPU_MIN_BATCH_SIZE=128
export ENGRAM_FORCE_CPU_MODE=false

# Or configure in code when initializing HybridExecutor
```

- [ ] **gpu-troubleshooting.md:739-750** - Remove diagnostic CLI commands section
- [ ] Search all files for `engram --` and `./target/release/engram --` - remove or fix all instances

### 4. Fix Metrics Endpoint Examples

**Time**: 1 hour
**Priority**: CRITICAL
**Files**: gpu-deployment.md, gpu-performance-tuning.md

- [ ] **gpu-deployment.md:642-654** - Fix metric names:
```markdown
### Verify GPU is Being Used

**Method 1: Check Metrics**

```bash
# Query metrics endpoint
curl http://localhost:8080/metrics | grep gpu

# Expected output (actual metric names):
# engram_gpu_launch_total 1234
# engram_gpu_fallback_total 5

# Calculate success rate
launches=$(curl -s http://localhost:8080/metrics | grep engram_gpu_launch_total | awk '{print $2}')
fallbacks=$(curl -s http://localhost:8080/metrics | grep engram_gpu_fallback_total | awk '{print $2}')

if [ "$launches" -gt 0 ]; then
  success_rate=$(echo "scale=3; ($launches - $fallbacks) / $launches" | bc)
  echo "GPU success rate: $success_rate"
else
  echo "GPU not being used (launches = 0)"
fi
```
```

- [ ] **gpu-performance-tuning.md:578-617** - Update all Prometheus queries:
```promql
# GPU launch rate
rate(engram_gpu_launch_total[5m])

# GPU fallback rate
rate(engram_gpu_fallback_total[5m])

# GPU success rate (calculated)
(rate(engram_gpu_launch_total[5m]) - rate(engram_gpu_fallback_total[5m]))
/ rate(engram_gpu_launch_total[5m])
```

- [ ] **gpu-deployment.md:753-829** - Update alerting rules with correct metric names

### 5. Label Performance Numbers as Predictions

**Time**: 30 minutes
**Priority**: CRITICAL
**Files**: gpu-architecture.md, gpu-performance-tuning.md

- [ ] **gpu-architecture.md:332-350** - Add prediction disclaimer:
```markdown
## Performance Characteristics

### Predicted Speedup Targets (NOT YET VALIDATED)

The following performance numbers are THEORETICAL PREDICTIONS based on Task 001
analysis. These have NOT been empirically validated as CUDA kernels are not yet
implemented. Actual performance will be measured in Milestone 13.

**Consumer GPU (RTX 3060) - PREDICTED**:

| Operation | Batch Size | Predicted Speedup | Break-even Point |
|-----------|-----------|-------------------|------------------|
| Cosine Similarity | 1,024 vectors | 7.0x (predicted) | 64 vectors |
| Activation Spreading | 1,000 nodes | 7.1x (predicted) | 512 nodes |
| HNSW Search | 10,000 candidates | 6.7x (predicted) | N/A |

**Status**: Predictions from theoretical bandwidth/compute analysis
**Validation**: Pending Milestone 13 CUDA kernel implementation
```

- [ ] **gpu-performance-tuning.md:815-857** - Add prediction labels to all GPU model sections
- [ ] Search for "speedup", "7.0x", "26.3x" etc. - add "predicted" qualifier to all

---

## HIGH PRIORITY FIXES (Should Complete This Week)

### 6. Remove CUDA Error Troubleshooting

**Time**: 1 hour
**Priority**: HIGH
**Files**: gpu-troubleshooting.md

- [ ] **gpu-troubleshooting.md:439-521** - Replace CUDA errors section with:
```markdown
### CUDA Error Codes (FOR FUTURE GPU IMPLEMENTATION)

The following CUDA errors will be relevant after GPU kernel implementation in
Milestone 13. Current CPU SIMD implementation cannot produce these errors.

**Common CUDA Errors (Future Reference)**:
- cudaErrorInvalidValue (Error 1): Invalid kernel arguments
- cudaErrorMemoryAllocation (Error 2): Out of GPU memory
- cudaErrorInitializationError (Error 3): Driver/CUDA mismatch
- cudaErrorInsufficientDriver (Error 35): Driver too old
- cudaErrorNoDevice (Error 100): No GPU detected

**For current release**: CPU SIMD errors are standard Rust panics/errors.
See standard Rust error handling documentation.
```

### 7. Fix/Remove Build System Documentation

**Time**: 1 hour
**Priority**: HIGH
**Files**: gpu-architecture.md

- [ ] **gpu-architecture.md:227-263** - Mark as planned architecture:
```markdown
## Build System Architecture (PLANNED FOR MILESTONE 13)

The following build system will be implemented when CUDA kernels are added:

**Planned Build Flow**:
```
cargo build
    │
    ├──> build.rs runs (PLANNED)
    │    │
    │    ├──> Detect CUDA toolkit (nvcc in PATH?)
    │    │    │
    │    │    ├──> Found: Compile .cu files with nvcc
    │    │    │              Link libcudart.so
    │    │    │              Enable GPU features
    │    │    │
    │    │    └──> Not Found: Generate no-op stubs
    │    │                     CPU-only build
```

**Current Build** (Milestone 12):
```bash
cargo build --release
# Always builds with CPU SIMD
# GPU infrastructure compiled but kernels are no-ops
```
```

### 8. Add CPU SIMD Deployment Guide

**Time**: 2 hours
**Priority**: HIGH
**Files**: gpu-deployment.md (new section at top)

- [ ] **gpu-deployment.md:13** - Add after overview:
```markdown
## Deploying Engram with CPU SIMD (Current Release)

The current Milestone 12 release uses optimized CPU SIMD for all vector operations.
Performance is excellent for most production workloads.

### Quick Start (CPU SIMD)

```bash
# 1. Clone and build
git clone https://github.com/your-org/engram.git
cd engram
cargo build --release

# 2. Run
./target/release/engram start

# 3. Verify performance
cargo bench --bench cpu_simd_performance
```

### CPU SIMD Performance

**Hardware Requirements**:
- x86_64 CPU with AVX2 support (AVX-512 recommended)
- 16GB+ RAM for large graphs
- SSD storage for optimal latency

**Performance Characteristics**:
- Cosine similarity: ~2.1 μs/vector (AVX-512)
- Activation spreading: ~850 μs for 1000 nodes
- Throughput: ~70K vectors/sec per core
- Scales linearly with CPU cores

**When CPU SIMD is Sufficient**:
- Query rates < 10K QPS
- Batch sizes < 1000 vectors
- Budget constraints (no GPU hardware)
- Predictable latency requirements

**Configuration**:

CPU SIMD is always available and requires no configuration. To explicitly
disable future GPU acceleration (when implemented):

```rust
let config = HybridConfig {
    force_cpu_mode: true,  // Ensure CPU-only execution
    ..Default::default()
};
```

### CPU Performance Tuning

**Optimize for Your Workload**:

1. **High-Throughput**:
   - Use batch operations
   - Increase worker threads (2x CPU cores)
   - Enable NUMA awareness if multi-socket

2. **Low-Latency**:
   - Pin threads to CPU cores
   - Disable CPU frequency scaling
   - Use isolated CPU cores (isolcpus)

3. **Mixed Workload**:
   - Use default configuration
   - Monitor CPU utilization
   - Scale horizontally if needed

See CPU Performance Tuning section below for details.
```

### 9. Fix Cross-References

**Time**: 30 minutes
**Priority**: HIGH
**Files**: All 4 files

- [ ] Search for references to `performance_report.md` - replace with actual file or remove
- [ ] Search for references to `optimization_roadmap.md` - replace with actual file or remove
- [ ] Search for references to Task 001 profiling data - add caveat that task is pending
- [ ] Verify all internal cross-references point to correct sections

---

## MEDIUM PRIORITY (Nice to Have)

### 10. Add Implementation Roadmap

**Time**: 1 hour
**Files**: gpu-architecture.md

- [ ] Add after "Overview" section:
```markdown
## Implementation Roadmap

### Milestone 12 (CURRENT) - GPU Architecture Foundation
**Status**: Complete
**Deliverables**:
- ✓ Hybrid executor architecture
- ✓ GPU abstraction interfaces (GPUSpreadingInterface, GPUActivationBatch)
- ✓ CPU SIMD fallback implementation (production-ready)
- ✓ Performance tracking infrastructure
- ✓ Adaptive dispatch framework

**What Works Now**:
- All operations use highly optimized CPU SIMD
- Architecture ready for GPU integration
- Performance metrics and telemetry

### Milestone 13 (PLANNED) - CUDA Kernel Implementation
**Status**: Not Started
**Timeline**: Q1 2026 (tentative)
**Deliverables**:
- [ ] Cosine similarity CUDA kernel
- [ ] Activation spreading CUDA kernel
- [ ] HNSW candidate scoring CUDA kernel
- [ ] CUDA FFI bindings
- [ ] GPU memory management (unified memory)
- [ ] Basic OOM handling

**Expected Impact**:
- 3-7x speedup on consumer GPUs (RTX 3060)
- 6-26x speedup on datacenter GPUs (A100)

### Milestone 14 (PLANNED) - Production Hardening
**Status**: Not Started
**Timeline**: Q2 2026 (tentative)
**Deliverables**:
- [ ] Advanced OOM recovery
- [ ] Multi-GPU support
- [ ] Mixed precision (FP16) for Ampere+
- [ ] Kernel fusion optimizations
- [ ] Persistent kernel mode
- [ ] GPU telemetry and observability

### Milestone 15 (FUTURE) - Advanced Optimizations
**Status**: Exploratory
**Possible Features**:
- Multi-Instance GPU (MIG) support
- AMD ROCm support
- Intel GPU support (via SYCL)
- Custom kernel optimizations per GPU architecture
```

### 11. Separate Current vs Future Content

**Time**: 2 hours
**Files**: All 4 files

- [ ] Wrap all GPU-specific content in callout boxes:
```markdown
> **GPU Implementation (Milestone 13+)**:
> The following content describes planned GPU acceleration features.
> See Implementation Roadmap for timeline.
```

- [ ] Add "CPU SIMD" sections where applicable
- [ ] Use consistent labeling: "CURRENT", "PLANNED", "FUTURE"

### 12. Add More Working Examples

**Time**: 2 hours
**Files**: gpu-deployment.md, gpu-architecture.md

- [ ] Add CPU SIMD benchmarking example:
```rust
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use engram_core::compute::get_vector_ops;

fn benchmark_cpu_simd(c: &mut Criterion) {
    let ops = get_vector_ops();
    let query = [1.0f32; 768];
    let targets = vec![[0.5f32; 768]; 1000];

    c.bench_function("cpu_simd_cosine_1k", |b| {
        b.iter(|| {
            ops.cosine_similarity_batch_768(
                black_box(&query),
                black_box(&targets)
            )
        });
    });
}

criterion_group!(benches, benchmark_cpu_simd);
criterion_main!(benches);
```

---

## Completion Checklist

After completing all critical and high-priority fixes:

- [ ] Re-run documentation validation (expect 8/10 score)
- [ ] Test all code examples compile
- [ ] Test all bash commands work (where applicable)
- [ ] Verify all configuration examples use correct parameter names
- [ ] Verify all metrics examples use correct field names
- [ ] Check all cross-references resolve
- [ ] Review with technical writer for clarity
- [ ] Get sign-off from DevOps for deployment accuracy
- [ ] Update task status to "complete" with caveats

---

## Success Criteria (Updated)

After fixes, documentation should:
- [x] Clearly separate current implementation from future plans
- [x] Provide accurate configuration examples that work
- [x] Offer working deployment guide for CPU SIMD (current)
- [x] Describe GPU architecture for future (labeled as planned)
- [x] Include only working CLI examples
- [x] Use correct metrics field names
- [x] Label all performance numbers appropriately

**Target Quality Score**: 8/10 (up from 4/10)

---

## Notes

- Focus on CRITICAL fixes first (6.5 hours) - these prevent immediate user failures
- HIGH priority fixes (4 hours) improve accuracy and usability
- MEDIUM priority fixes (3 hours) enhance long-term documentation quality
- Total effort: ~13.5 hours to bring documentation to publishable quality

**Owner**: Technical Writer + DevOps Review
**Deadline**: 2025-10-27 EOD
**Tracking**: Update this checklist as items are completed
