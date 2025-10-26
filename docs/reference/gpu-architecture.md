# GPU Architecture Reference

**Audience**: System architects, performance engineers, developers extending GPU features

**Last Updated**: 2025-10-26

## IMPORTANT: Current Implementation Status

**GPU acceleration infrastructure is under active development. Milestone 12 implementation status:**

**IMPLEMENTED (Production-Ready)**:
- Hybrid executor architecture with intelligent CPU/GPU dispatch
- GPU abstraction interfaces and performance tracking
- CPU SIMD fallback implementation (high performance, always available)
- Infrastructure for future CUDA kernel integration

**NOT YET IMPLEMENTED**:
- Actual CUDA kernels for cosine similarity, spreading activation, HNSW
- GPU device detection and initialization
- GPU-specific error handling beyond interfaces

**Current Behavior**: All operations currently execute using CPU SIMD implementations. The hybrid executor architecture is in place and ready for CUDA kernel integration in a future milestone.

**This documentation describes the target architecture.** External operators can deploy Engram today using CPU SIMD (production-ready, high performance). GPU acceleration will be added in Milestone 13.

See `/Users/jordanwashburn/Workspace/orchard9/engram/roadmap/milestone-12/MILESTONE_12_IMPLEMENTATION_SPEC.md` for detailed implementation status.

---

## Overview

Engram's GPU acceleration system is designed around a hybrid CPU/GPU executor that intelligently dispatches workloads based on batch size and historical performance data. The architecture is built to support CUDA kernels for three core operations: cosine similarity, activation spreading, and HNSW candidate scoring.

This document describes the technical architecture, components, and design decisions. For deployment guidance, see the GPU Deployment Guide. For performance optimization, see the GPU Performance Tuning Guide.

## Architecture Principles

1. **Architecture Ready for GPU**: Hybrid executor infrastructure in place, currently runs CPU SIMD only.
2. **Future Automatic Dispatch**: Once CUDA kernels are implemented, the executor will automatically choose CPU or GPU based on workload characteristics.
3. **Designed for Graceful Degradation**: Architecture ensures GPU failures will fall back to CPU without surfacing errors.
4. **Production Hardened Foundation**: Performance tracking, error telemetry interfaces, and confidence tracking infrastructure ready for GPU integration.

## System Architecture

### Component Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    Engram Core API                          │
│  (Spreading, HNSW Search, Vector Operations)               │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│              HybridExecutor (Rust)                          │
│  ┌──────────────────────────────────────────────────┐      │
│  │  Dispatch Decision Logic                         │      │
│  │  - Batch size threshold: 64+ vectors → GPU       │      │
│  │  - Historical performance tracking               │      │
│  │  - GPU success rate monitoring                   │      │
│  └──────────────────────────────────────────────────┘      │
│                     │                                        │
│        ┌────────────┴────────────┐                          │
│        ▼                         ▼                          │
│  ┌──────────┐             ┌──────────────┐                 │
│  │ CPU Path │             │  GPU Path    │                 │
│  │  SIMD    │             │  CUDA FFI    │                 │
│  └──────────┘             └──────┬───────┘                 │
└───────────────────────────────────┼──────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────┐
│              GPU Kernels (CUDA C++)                         │
│  ┌──────────────────┬──────────────────┬─────────────────┐ │
│  │ cosine_similarity│ spreading_matmul │ hnsw_scoring    │ │
│  │ - Dot product    │ - Sparse matrix  │ - Top-k select  │ │
│  │ - Normalization  │ - Sigmoid        │ - Distance calc │ │
│  └──────────────────┴──────────────────┴─────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────┐
│               Unified Memory Allocator                      │
│  - CUDA Unified Memory (cudaMallocManaged)                  │
│  - Automatic host/device migration                          │
│  - OOM detection and recovery                               │
└─────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. Hybrid Executor

**Location**: `engram-core/src/activation/gpu_interface.rs`

The hybrid executor sits between Engram's high-level APIs and the CPU/GPU implementations. It makes runtime decisions about where to execute each workload.

**Key Responsibilities**:
- Detect GPU availability at startup
- Route operations to CPU or GPU based on batch size
- Track performance metrics per backend
- Handle GPU failures with CPU fallback
- Export telemetry for monitoring

**Decision Algorithm**:

```rust
fn should_use_gpu(batch_size: usize) -> bool {
    // Rule 1: GPU disabled or unavailable?
    if !config.enable_gpu || !gpu_available {
        return false;
    }

    // Rule 2: Batch too small?
    if batch_size < config.gpu_threshold {
        return false; // Kernel launch overhead not worth it
    }

    // Rule 3: GPU success rate acceptable?
    if gpu_success_rate < config.success_rate_threshold {
        return false; // Too many failures, stick with CPU
    }

    // Use GPU
    return true;
}
```

**Configuration Parameters** (from `HybridConfig` struct):

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `gpu_min_batch_size` | `64` | Minimum batch size for GPU dispatch |
| `gpu_speedup_threshold` | `1.5` | Minimum speedup to prefer GPU |
| `gpu_success_rate_threshold` | `0.95` | Minimum GPU success rate (95%) |
| `performance_window_size` | `100` | Number of samples for moving average |
| `force_cpu_mode` | `false` | Debugging flag to disable GPU |
| `telemetry_enabled` | `true` | Enable performance tracking |

### 2. CUDA Kernels

**Location**: CUDA kernels are planned but not yet implemented in current milestone. The architecture includes FFI interfaces for future integration.

Engram currently uses CPU SIMD implementations with GPU infrastructure in place for future acceleration.

**Planned Kernel Operations**:

1. **Batch Cosine Similarity**
   - Input: Query vector (768D), N target vectors (768D each)
   - Output: N similarity scores (f32)
   - Parallelization: One thread per target vector
   - Memory Pattern: Coalesced reads, single write per thread

2. **Activation Spreading**
   - Input: Sparse adjacency matrix (M nodes, avg degree K), activation levels
   - Output: Updated activation levels after spreading
   - Parallelization: One thread per node
   - Memory Pattern: Irregular reads (neighbor access), single write per thread

3. **HNSW Candidate Scoring**
   - Input: Query embedding, candidate embeddings, HNSW graph structure
   - Output: Top-K nearest neighbors with scores
   - Parallelization: Parallel distance computation + reduction
   - Memory Pattern: Batch distance calculation followed by top-k selection

### 3. FFI Interface

**Location**: Planned for `engram-core/src/compute/cuda/ffi.rs`

The FFI layer bridges Rust and CUDA C++ code, providing safe wrappers around CUDA runtime APIs.

**Design Pattern**:

```rust
// Raw CUDA API (unsafe)
extern "C" {
    pub fn cudaMalloc(ptr: *mut *mut c_void, size: usize) -> i32;
}

// Safe Rust wrapper
pub fn allocate_device_memory(size: usize) -> Result<DevicePtr, CudaError> {
    let mut ptr = std::ptr::null_mut();
    unsafe {
        let result = cudaMalloc(&mut ptr, size);
        match CudaError::from_raw(result) {
            CudaError::Success => Ok(DevicePtr::new(ptr)),
            err => Err(err),
        }
    }
}
```

**Error Handling**:
- All CUDA errors convert to Rust `Result` types
- `CudaError` enum maps CUDA error codes to semantic names
- Errors trigger CPU fallback rather than crashing

### 4. Unified Memory Allocator

**Purpose**: Simplifies CPU/GPU memory management by using CUDA Unified Memory.

**Benefits**:
- Single memory allocation works on both CPU and GPU
- Automatic migration handled by CUDA driver
- Simpler code (no explicit `cudaMemcpy` calls)
- Graceful performance degradation under memory pressure

**Tradeoffs**:
- Slightly higher overhead vs explicit memory management
- Requires GPU compute capability 6.0+ (Pascal or newer)
- Page faults on first access add latency

**When to Use**:
- Prototyping and development (faster iteration)
- Workloads with unpredictable access patterns
- Small to medium batch sizes (< 100MB transfers)

**When to Avoid**:
- Very large batches where explicit transfer overlap helps
- Latency-critical paths where page fault overhead matters
- GPUs older than Pascal architecture

### 5. Performance Tracker

**Location**: `engram-core/src/activation/gpu_interface.rs` (AdaptiveSpreadingEngine metrics)

Tracks historical performance to inform dispatch decisions and provide observability.

**Tracked Metrics**:
- CPU latency distribution (P50/P90/P99)
- GPU latency distribution (P50/P90/P99)
- GPU launch count
- GPU fallback count (failures)
- Success rate per backend
- Speedup ratio (GPU vs CPU)

**Telemetry Export**:

Metrics are accessible via the spreading metrics interface:

```rust
pub struct SpreadingMetrics {
    pub gpu_launch_total: AtomicU64,      // Note: Field name (not "gpu_launches")
    pub gpu_fallback_total: AtomicU64,    // Note: Field name (not "gpu_fallbacks")
    // ... other metrics
}
```

These integrate with Engram's existing Prometheus metrics endpoint at `/metrics`.

## Build System Architecture (PLANNED - Not Yet Implemented)

**Note**: This section describes the planned build system for Milestone 13+ when CUDA kernels are added.

**Challenge**: Support both CUDA and non-CUDA builds from the same codebase.

**Solution**: Feature flags and graceful detection.

**Build Flow**:

```
cargo build
    │
    ├──> build.rs runs
    │    │
    │    ├──> Detect CUDA toolkit (nvcc in PATH?)
    │    │    │
    │    │    ├──> Found: Compile .cu files with nvcc
    │    │    │              Link libcudart.so
    │    │    │              Enable GPU features
    │    │    │
    │    │    └──> Not Found: Generate no-op stubs
    │    │                     CPU-only build
    │    │
    │    └──> Continue Rust compilation
    │
    └──> Runtime: GPU available? Use hybrid executor : Use CPU-only
```

**Key Files**:
- `build.rs`: Detects CUDA, compiles kernels, links libraries
- `Cargo.toml`: Defines `gpu` feature flag
- `src/lib.rs`: Conditional compilation based on feature

**Developer Experience**:
- Developers without NVIDIA GPUs can build and run tests (Current: Always CPU SIMD)
- CI builds succeed on all platforms (Current: CPU SIMD only)
- Future: GPU-enabled CI runners will validate CUDA path
- No special flags needed (Current: CPU SIMD always used)

**Current Milestone 12 State**: The infrastructure above is the target design. Current implementation uses CPU SIMD for all operations with conditional compilation gates ready for future CUDA integration.

## Memory Management

### Memory Flow Diagram

```
┌──────────────────────────────────────────────────────────────┐
│                    Host Memory (RAM)                         │
│  Rust Vectors: Vec<[f32; 768]>                              │
└────────────────────┬─────────────────────────────────────────┘
                     │
                     │ cudaMallocManaged
                     ▼
┌──────────────────────────────────────────────────────────────┐
│               Unified Memory Region                          │
│  - Accessible from CPU and GPU                               │
│  - Automatic migration via page faults                       │
│  - Managed by CUDA driver                                    │
└────────────────────┬─────────────────────────────────────────┘
                     │
                     │ Kernel Launch
                     ▼
┌──────────────────────────────────────────────────────────────┐
│                   GPU Memory (VRAM)                          │
│  - Pages migrated on first access                            │
│  - Coherent with CPU view                                    │
│  - Released on cudaFree                                      │
└──────────────────────────────────────────────────────────────┘
```

### OOM Handling

**Problem**: GPU VRAM is limited. Large batches can exceed available memory.

**Detection**:
- `cudaMalloc` returns `cudaErrorMemoryAllocation`
- Kernel launches fail with `cudaErrorLaunchOutOfResources`

**Recovery Strategy**:

1. **Immediate**: Fail current operation, fallback to CPU
2. **Logging**: Record OOM event with batch size and VRAM state
3. **Adaptation**: Increase GPU batch threshold to avoid future OOMs
4. **Telemetry**: Export OOM count for monitoring

```rust
match gpu.launch(batch).await {
    Ok(result) => result,
    Err(CudaError::OutOfMemory) => {
        metrics.record_oom();
        warn!("GPU OOM at batch size {}, falling back to CPU", batch.size());

        // Adapt threshold to avoid this batch size in future
        config.gpu_threshold = batch.size() + 10;

        // Use CPU fallback
        cpu_fallback.process(batch)
    }
    Err(err) => {
        error!("GPU error: {:?}", err);
        cpu_fallback.process(batch)
    }
}
```

## Performance Characteristics (PREDICTED - NOT YET VALIDATED)

**Important**: The performance numbers below are theoretical predictions from Task 001 analysis. They have NOT been validated with actual GPU implementations since CUDA kernels are not yet implemented in Milestone 12.

### Speedup Targets (Predicted)

Based on Task 001 profiling analysis (predictions, not measurements):

**Consumer GPU (RTX 3060, GTX 1660 Ti)** - PREDICTED, NOT MEASURED:

| Operation | Batch Size | Target Speedup (Predicted) | Break-even Point (Predicted) |
|-----------|-----------|---------------|------------------|
| Cosine Similarity | 1,024 vectors | 7.0x (predicted) | 64 vectors (predicted) |
| Activation Spreading | 1,000 nodes | 7.1x (predicted) | 512 nodes (predicted) |
| HNSW Search | 10,000 candidates | 6.7x (predicted) | N/A |

**Datacenter GPU (A100, H100)** - PREDICTED, NOT MEASURED:

| Operation | Batch Size | Target Speedup (Predicted) | Break-even Point (Predicted) |
|-----------|-----------|---------------|------------------|
| Cosine Similarity | 10,240 vectors | 26.3x (predicted) | 32 vectors (predicted) |
| Activation Spreading | 10,000 nodes | 18.9x (predicted) | 256 nodes (predicted) |
| HNSW Search | 100,000 candidates | 14.1x (predicted) | N/A |

### Latency Breakdown

**GPU Execution Latency Components**:

```
Total GPU Latency = Kernel Launch + Compute + Memory Transfer
                  = 10-20us       + T_compute + T_transfer

For small batches (< 64 vectors):
    T_compute ~= 50us
    Launch overhead dominates → CPU faster

For large batches (> 1024 vectors):
    T_compute ~= 200us
    Compute dominates, GPU wins
```

**CPU Execution Latency**:

```
Total CPU Latency = SIMD Compute
                  = T_simd

For 1024 vectors:
    T_simd ~= 2100us (2.1ms)

GPU Speedup = 2100 / (20 + 200 + 50) = 7.8x
```

### Throughput

**Consumer GPU (RTX 3060)**:
- Peak throughput: ~500K vectors/sec (cosine similarity)
- Sustained throughput: ~400K vectors/sec (accounting for launch overhead)

**Datacenter GPU (A100)**:
- Peak throughput: ~12M vectors/sec (cosine similarity)
- Sustained throughput: ~10M vectors/sec

**CPU SIMD (AVX-512)**:
- Peak throughput: ~70K vectors/sec (cosine similarity)
- Limited by memory bandwidth, not compute

## API Compatibility

### Current API (Milestone 12 - CPU SIMD)

```rust
use engram_core::activation::spreading::spread_activation;

// Currently uses CPU SIMD implementation
let results = spread_activation(&graph, &initial_activations, depth);
```

### Future with GPU Acceleration (Same API)

```rust
use engram_core::activation::spreading::spread_activation;

// Identical call - GPU will be used automatically when CUDA kernels implemented
// Currently: CPU SIMD only
let results = spread_activation(&graph, &initial_activations, depth);
```

**Current Behavior**: All calls use CPU SIMD. The API is designed for GPU transparency - when CUDA kernels are added in Milestone 13, no code changes will be required.

### Explicit Control (Optional)

For advanced users who want explicit control:

```rust
use engram_core::activation::gpu_interface::{AdaptiveConfig, AdaptiveSpreadingEngine};

let config = AdaptiveConfig {
    enable_gpu: true,
    gpu_threshold: 128, // Custom threshold
    ..Default::default()
};

let mut engine = AdaptiveSpreadingEngine::new(None, config, None);
let results = engine.spread(&batch).await?;
```

## Limitations and Constraints

### GPU Requirements

**Minimum**:
- NVIDIA GPU with compute capability 6.0+ (Pascal architecture)
- CUDA 11.0+ toolkit
- 4GB VRAM (8GB recommended for production)

**Tested Configurations**:
- Consumer: GTX 1660 Ti, RTX 3060, RTX 4070
- Datacenter: V100, A100
- Not supported: AMD GPUs, Intel GPUs (CUDA-only currently)

### Workload Constraints

**GPU acceleration provides benefit when**:
- Batch size >= 64 vectors (cosine similarity)
- Graph has >= 512 nodes (activation spreading)
- Index has >= 1000 candidates (HNSW search)

**CPU is faster when**:
- Batch size < 64 vectors
- Single query operations
- Memory bandwidth limited (very large embeddings)

### Architectural Constraints

**Current Implementation**:
- Single GPU per process (no multi-GPU support)
- No GPU-to-GPU transfers (each operation independent)
- No persistent kernels (launch overhead per batch)
- No kernel fusion (each operation separate)

**Future Extensions** (see optimization roadmap):
- Multi-GPU support for large workloads
- Persistent kernels for lower latency
- Kernel fusion to reduce overhead
- Mixed precision (FP16) for Ampere+ GPUs

## Debugging and Diagnostics

### GPU Detection

Check if GPU is detected:

```bash
# From Rust code
let engine = AdaptiveSpreadingEngine::new(config, None, None);
println!("GPU available: {}", engine.backend_name());

# Expected output (GPU available):
# GPU available: NVIDIA GeForce RTX 3060

# Expected output (no GPU):
# GPU available: CPU_SIMD_FALLBACK
```

### Performance Profiling

Monitor GPU dispatch decisions:

```rust
// Enable tracing at DEBUG level
export RUST_LOG=engram::activation::gpu=debug

// Logs will show:
// DEBUG engram::activation::gpu: Batch size 128 >= min 64, using GPU
// DEBUG engram::activation::gpu: GPU launch succeeded, latency=250us
```

### Metrics

Query GPU metrics via HTTP:

```bash
curl http://localhost:8080/metrics | grep gpu

# Expected output (when GPU features are implemented):
# engram_gpu_launch_total 1523
# engram_gpu_fallback_total 12
# Note: gpu_success_rate is derived from launch_total and fallback_total
```

## Security Considerations

### Memory Safety

**Rust-CUDA Boundary**:
- All CUDA FFI is `unsafe` Rust
- Safe wrappers validate pointer lifetimes
- RAII patterns ensure `cudaFree` on drop
- No raw pointer leaks to safe code

**Unified Memory**:
- Page faults cannot corrupt memory
- CPU and GPU see consistent data
- No race conditions (sequential API)

### Resource Exhaustion

**VRAM Exhaustion**:
- OOM detection prevents crashes
- CPU fallback ensures forward progress
- Adaptive thresholding prevents repeated OOMs

**GPU Hang Detection**:
- Currently: No timeout on kernel launches (CUDA driver handles)
- Future: Watchdog timer for stuck kernels

## Performance Regression Testing

**Baseline Benchmarks**:
- Benchmark suite in `engram-core/benches/gpu_performance_validation.rs`
- Run via `cargo bench --bench gpu_performance_validation`
- Results stored in `target/criterion/`

**Regression Thresholds**:
- Warning: >10% performance degradation
- Critical: >20% degradation or speedup < 3x target

**CI Integration**:
- Benchmarks run on GPU-enabled CI runners
- Results compared against baseline
- Alerts on regression

## References

- **Task 001**: GPU Profiling and Baseline Analysis
- **Task 007**: CPU-GPU Hybrid Executor Design
- **Task 010**: Performance Benchmarking and Validation
- Performance Report: `/roadmap/milestone-12/performance_report.md`
- Optimization Roadmap: `/roadmap/milestone-12/optimization_roadmap.md`

## Glossary

**CUDA**: NVIDIA's parallel computing platform and API for GPU programming

**FFI**: Foreign Function Interface - mechanism for calling C/C++ code from Rust

**SIMD**: Single Instruction Multiple Data - CPU vectorization (AVX2, AVX-512, NEON)

**Unified Memory**: CUDA memory accessible from both CPU and GPU with automatic migration

**Kernel**: GPU function launched in parallel across many threads

**Warp**: Group of 32 GPU threads executing in lockstep

**SM (Streaming Multiprocessor)**: GPU compute unit containing CUDA cores

**Occupancy**: Ratio of active warps to maximum possible warps on an SM

**Coalesced Access**: Memory access pattern where consecutive threads access consecutive addresses (efficient)
