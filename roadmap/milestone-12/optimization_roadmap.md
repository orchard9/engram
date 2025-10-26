# GPU Acceleration Optimization Roadmap

**Status**: Planning Document
**Last Updated**: 2025-10-26
**Target**: Post-Milestone 12 optimization work

## Purpose

This document prioritizes future GPU optimization work based on profiling data and bottleneck analysis from Task 010 performance benchmarking. Each optimization is ranked by ROI (return on investment) considering both performance impact and implementation complexity.

## Optimization Classification

### Priority Levels

- **P0 (Critical)**: Blocking production deployment or causing >50% performance loss
- **P1 (High)**: >20% performance improvement with reasonable effort
- **P2 (Medium)**: 10-20% performance improvement or high-effort high-impact
- **P3 (Low)**: <10% performance improvement or exploratory work

### Effort Levels

- **Low**: 1-3 days of focused work
- **Medium**: 1-2 weeks of implementation and testing
- **High**: 2-4 weeks or requiring significant architectural changes
- **Very High**: >1 month or requiring external dependencies

## High-Priority Optimizations (P1)

### 1. Warp-Level Reduction Optimization

**Status**: Identified from profiling
**Current Bottleneck**: Inefficient warp-level reductions in cosine similarity kernel
**Estimated Impact**: 15-25% speedup for batch cosine similarity
**Effort**: Medium (1-2 weeks)

**Current Implementation**:
```cuda
// Naive reduction with divergent branches
__device__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}
```

**Optimized Implementation**:
```cuda
// Use specialized warp intrinsics for better performance
__device__ float warp_reduce_sum_optimized(float val) {
    // Use warp-level primitives available in CUDA 11+
    val += __shfl_xor_sync(0xffffffff, val, 16);
    val += __shfl_xor_sync(0xffffffff, val, 8);
    val += __shfl_xor_sync(0xffffffff, val, 4);
    val += __shfl_xor_sync(0xffffffff, val, 2);
    val += __shfl_xor_sync(0xffffffff, val, 1);
    return val;
}
```

**Profiling Evidence Needed**:
- Warp divergence metrics from Nsight Compute
- Current warp efficiency percentage
- Reduction kernel timing breakdown

**Success Metrics**:
- Warp efficiency >95%
- 15-25% reduction in kernel execution time
- No increase in register pressure

---

### 2. Coalesced Memory Access Patterns

**Status**: Suspected from memory bandwidth profiling
**Current Bottleneck**: Uncoalesced memory accesses in vector load operations
**Estimated Impact**: 20-30% speedup for memory-bound operations
**Effort**: Medium (1-2 weeks)

**Current Access Pattern**:
```cuda
// Row-major access with poor coalescing
__global__ void cosine_similarity_naive(
    const float* query,         // 768 elements
    const float* targets,       // batch_size × 768 elements
    float* results,             // batch_size elements
    int batch_size
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= batch_size) return;

    float dot = 0.0f;
    for (int i = 0; i < 768; i++) {
        // Stride-1 access for query (good)
        // Stride-768 access for targets (BAD - uncoalesced)
        dot += query[i] * targets[tid * 768 + i];
    }
    results[tid] = dot;
}
```

**Optimized Access Pattern**:
```cuda
// Transpose data for coalesced access
__global__ void cosine_similarity_coalesced(
    const float* query,         // 768 elements
    const float* targets_t,     // 768 × batch_size (transposed)
    float* results,             // batch_size elements
    int batch_size
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= batch_size) return;

    float dot = 0.0f;
    for (int i = 0; i < 768; i++) {
        // Both accesses now stride-batch_size (coalesced when batch large)
        dot += query[i] * targets_t[i * batch_size + tid];
    }
    results[tid] = dot;
}
```

**Implementation Steps**:
1. Add transpose kernel for batch preprocessing
2. Measure transpose overhead vs coalescing benefit
3. Implement adaptive dispatch based on batch size
4. Update unified memory prefetch hints

**Profiling Evidence Needed**:
- Global memory transaction efficiency from Nsight
- Memory throughput utilization percentage
- L1/L2 cache hit rates

**Success Metrics**:
- Global memory transaction efficiency >80%
- 20-30% improvement in memory-bound kernels
- Transpose overhead <5% of total time

---

### 3. Stream-Based Pipelining

**Status**: Not yet implemented
**Current Bottleneck**: Serial execution of kernel launch, execute, synchronize
**Estimated Impact**: 15-20% throughput improvement for continuous workloads
**Effort**: Medium (1-2 weeks)

**Current Implementation**:
```rust
// Sequential execution
pub fn execute_batch(&self, queries: &[[f32; 768]]) -> Vec<Vec<f32>> {
    let mut results = Vec::new();
    for query in queries {
        // Launch kernel
        let result = self.gpu_cosine_similarity(query, &self.targets)?;
        // Implicit synchronization
        results.push(result);
    }
    results
}
```

**Optimized Implementation**:
```rust
// Pipelined execution with CUDA streams
pub fn execute_batch_pipelined(&self, queries: &[[f32; 768]]) -> Vec<Vec<f32>> {
    const NUM_STREAMS: usize = 4;
    let streams = create_cuda_streams(NUM_STREAMS);

    let mut results = vec![Vec::new(); queries.len()];

    for (i, query) in queries.iter().enumerate() {
        let stream_id = i % NUM_STREAMS;
        // Launch asynchronously on stream
        launch_async(&streams[stream_id], query, &self.targets, &mut results[i]);
    }

    // Synchronize all streams at end
    for stream in &streams {
        stream.synchronize();
    }

    results
}
```

**Benefits**:
- Overlaps kernel execution across multiple operations
- Hides kernel launch latency
- Better GPU utilization for multi-query workloads

**Success Metrics**:
- GPU utilization >90% during batch processing
- 15-20% higher throughput for continuous workloads
- No impact on single-query latency

---

### 4. Shared Memory Optimization

**Status**: Identified from profiling
**Current Bottleneck**: Repeated global memory accesses for query vector
**Estimated Impact**: 10-15% speedup for cosine similarity
**Effort**: Low (2-3 days)

**Current Implementation**:
```cuda
__global__ void cosine_similarity(const float* query, ...) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    float dot = 0.0f;
    for (int i = 0; i < 768; i++) {
        // query accessed from global memory repeatedly (slow)
        dot += query[i] * targets[tid * 768 + i];
    }
}
```

**Optimized Implementation**:
```cuda
__global__ void cosine_similarity_shared(const float* query, ...) {
    __shared__ float query_shared[768];

    // Cooperative load into shared memory
    for (int i = threadIdx.x; i < 768; i += blockDim.x) {
        query_shared[i] = query[i];
    }
    __syncthreads();

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float dot = 0.0f;
    for (int i = 0; i < 768; i++) {
        // query accessed from shared memory (fast)
        dot += query_shared[i] * targets[tid * 768 + i];
    }
}
```

**Success Metrics**:
- Reduced global memory transactions
- 10-15% faster kernel execution
- No shared memory bank conflicts

---

## Medium-Priority Optimizations (P2)

### 5. Batch Size Auto-Tuning

**Status**: Manual configuration
**Current Bottleneck**: Fixed break-even batch sizes don't adapt to hardware
**Estimated Impact**: 5-15% throughput improvement via better dispatch
**Effort**: Medium (1 week)

**Proposed Solution**:
- Runtime autotuning on first use
- Measure CPU vs GPU crossover point
- Cache tuning results per GPU model
- Adaptive dispatch based on measurements

**Implementation**:
```rust
pub struct AdaptiveBatchSizer {
    tuning_cache: HashMap<String, usize>,  // GPU model -> optimal batch size
}

impl AdaptiveBatchSizer {
    pub fn optimal_batch_size(&mut self, gpu_model: &str) -> usize {
        if let Some(&size) = self.tuning_cache.get(gpu_model) {
            return size;
        }

        // Run microbenchmarks to find crossover
        let batch_size = self.tune_batch_size();
        self.tuning_cache.insert(gpu_model.to_string(), batch_size);
        batch_size
    }

    fn tune_batch_size(&self) -> usize {
        // Binary search for crossover point
        // Measure CPU and GPU at each batch size
        // Return point where GPU becomes faster
        unimplemented!()
    }
}
```

---

### 6. Kernel Fusion

**Status**: Separate kernel launches
**Current Bottleneck**: Multiple kernel launches for normalize + similarity
**Estimated Impact**: 10-20% reduction in overhead for small batches
**Effort**: Medium (1-2 weeks)

**Current Implementation**:
```cuda
// Three separate kernel launches
normalize_kernel<<<...>>>(query, query_norm);
normalize_kernel<<<...>>>(targets, targets_norm);
cosine_similarity_kernel<<<...>>>(query_norm, targets_norm, results);
```

**Optimized Implementation**:
```cuda
// Single fused kernel
__global__ void fused_normalize_and_similarity(
    const float* query,
    const float* targets,
    float* results,
    int batch_size
) {
    // Compute norms and similarity in one pass
    // Reduces kernel launch overhead from 3× to 1×
    // Reduces global memory roundtrips
}
```

---

### 7. FP16 Mixed Precision

**Status**: FP32 only
**Current Bottleneck**: Memory bandwidth for large batches
**Estimated Impact**: 30-50% speedup for Ampere/Hopper with Tensor Cores
**Effort**: High (2-3 weeks)

**Considerations**:
- Requires Tensor Core utilization
- Need careful accuracy validation (differential testing)
- May require FP32 accumulation to avoid precision loss
- Compute capability 7.0+ (Volta+) for good performance

**Implementation Strategy**:
1. Convert inputs to FP16 on GPU
2. Use Tensor Core matrix operations (wmma API)
3. Accumulate in FP32 for precision
4. Convert outputs back to FP32
5. Extensive differential testing vs FP32

**Risk**: Accuracy degradation in cosine similarity
**Mitigation**: Keep FP32 as default, FP16 as opt-in flag

---

### 8. Persistent Kernels

**Status**: Launch-on-demand
**Current Bottleneck**: Kernel launch overhead for high-throughput scenarios
**Estimated Impact**: 20-40% latency reduction for sustained workloads
**Effort**: High (3-4 weeks)

**Concept**:
- Launch long-running kernel once
- Kernel polls work queue
- Eliminates per-operation launch overhead
- Requires work queue management

**Use Case**: Production servers with continuous query stream

---

## Low-Priority Optimizations (P3)

### 9. Multi-GPU Support

**Status**: Single GPU only
**Estimated Impact**: Linear scaling for batch operations
**Effort**: Very High (4-6 weeks)

**Considerations**:
- Data parallelism across GPUs
- NCCL for inter-GPU communication
- Requires load balancing and orchestration
- Complexity increases significantly

**Deferral Reason**: Single-GPU must be proven in production first

---

### 10. CUDA Graphs

**Status**: Not implemented
**Estimated Impact**: 10-15% reduction in CPU overhead
**Effort**: Medium (2 weeks)

**Benefits**:
- Pre-record kernel launch sequences
- Reduce CPU-side overhead for repeated patterns
- Better for steady-state workloads

**Limitations**:
- Less flexible than dynamic launches
- Requires fixed graph structure

---

### 11. ROCm Support (AMD GPUs)

**Status**: NVIDIA only
**Estimated Impact**: Enables AMD hardware
**Effort**: Very High (6-8 weeks)

**Considerations**:
- Requires HIP API instead of CUDA
- Different performance characteristics
- Smaller deployment base
- Maintenance burden doubles

**Deferral Reason**: Focus on NVIDIA ecosystem first

---

## Profiling Workflow

### Development Profiling

**Step 1: Identify Hotspots**
```bash
# System-wide timeline
nsys profile -o timeline.qdrep ./target/release/benchmark

# Analyze in GUI
nsys-ui timeline.qdrep
```

**Step 2: Kernel-Level Optimization**
```bash
# Detailed kernel metrics
ncu --set full -o kernel_profile.ncu-rep ./target/release/benchmark

# Analyze roofline model, memory throughput, warp efficiency
ncu-ui kernel_profile.ncu-rep
```

**Step 3: Validation**
```bash
# Compare optimized vs baseline
cargo bench --bench gpu_performance_validation
```

### Key Metrics to Track

**Memory Bound Indicators**:
- Global memory throughput < 60%: Poor coalescing or cache misses
- L1/L2 cache hit rate < 80%: Poor data locality
- Memory transaction efficiency < 80%: Uncoalesced accesses

**Compute Bound Indicators**:
- Achieved FLOPS < 50% theoretical: Inefficient computation
- Warp execution efficiency < 90%: Divergent branches
- SM occupancy < 50%: Insufficient parallelism

**Optimization Targets**:
- Memory throughput >80% of peak
- Compute utilization >70% of peak
- Warp efficiency >95%
- SM occupancy >75%

---

## Experimental Optimizations

### E1. Unified Memory with Prefetch Hints

**Status**: Experimental
**Approach**: Use `cudaMemPrefetchAsync` to guide page migration
**Risk**: Complex tuning, platform-dependent behavior

### E2. Dynamic Parallelism

**Status**: Experimental
**Approach**: Launch child kernels from GPU for irregular workloads
**Risk**: High overhead, limited use cases

### E3. Custom Memory Allocator

**Status**: Experimental
**Approach**: GPU memory pool with custom allocation strategy
**Risk**: Debugging complexity, marginal gains

---

## Decision Framework

### When to Optimize

**Optimize if**:
- Profiling shows clear bottleneck (>20% time in one area)
- Speedup estimate >15% for reasonable effort
- Production workload would benefit
- No significant risk to correctness

**Defer if**:
- Speculative optimization without profiling data
- Marginal gains (<10%) with high complexity
- Rare code path with low impact
- Correctness risk outweighs performance gain

### Optimization Checklist

- [ ] Profile to confirm bottleneck hypothesis
- [ ] Estimate theoretical maximum speedup
- [ ] Prototype optimization on simplified test case
- [ ] Measure actual speedup on representative workload
- [ ] Run full differential test suite (CPU vs GPU correctness)
- [ ] Benchmark on multiple GPU architectures
- [ ] Document performance characteristics and trade-offs
- [ ] Update performance regression baselines

---

## Milestone Dependencies

### Prerequisites for Each Priority Level

**P1 Optimizations**:
- Requires: Milestone 12 complete, Task 010 profiling data
- Blocks: None (performance improvements only)

**P2 Optimizations**:
- Requires: P1 optimizations validated in production
- Blocks: None

**P3 Optimizations**:
- Requires: Production deployment at scale
- Blocks: Future milestones (e.g., M13 multi-GPU)

---

## Success Metrics

### Overall Goals

- **Achieve Task 001 predictions within ±30%** for all operations
- **Maintain correctness**: Zero failures in differential testing
- **Improve tail latencies**: P99 < 2× P50 for all operations
- **Sustain performance**: No degradation under continuous load

### Per-Optimization Metrics

Each optimization should demonstrate:
1. Profiling data confirming bottleneck
2. Theoretical speedup calculation
3. Measured speedup on multiple GPU models
4. No regression in correctness tests
5. Updated documentation and regression baselines

---

## Future Research Directions

### Beyond Milestone 12

1. **Sparse Matrix Optimizations**: Leverage sparsity in activation spreading
2. **Graph Compression**: Reduce memory footprint for large graphs
3. **Distributed GPU**: Scale across multiple nodes
4. **Custom CUDA Kernels**: Hand-optimized assembly for critical paths
5. **Hardware-Specific Tuning**: Exploit Hopper/Ada architecture features

---

**Document Status**: Living document, updated based on profiling results
**Next Review**: After Task 010 benchmark execution
**Owner**: Performance Engineering Team
