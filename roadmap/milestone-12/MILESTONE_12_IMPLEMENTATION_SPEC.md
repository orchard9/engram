# Milestone 12: GPU Acceleration for Engram

**Technical Lead**: GPU Acceleration Architect
**Duration**: 16-20 days
**Dependencies**: Milestones 1-8 (core memory operations, SIMD optimization, activation spreading)

## Executive Summary

This milestone adds CUDA GPU acceleration to Engram's memory operations while maintaining strict CPU-only fallback compatibility. The implementation follows a ruthlessly pragmatic approach: GPU acceleration only where profiling demonstrates clear speedup (>3x), with zero-copy unified memory where available, and graceful degradation to CPU SIMD paths when GPUs are unavailable or unsuitable for batch sizes.

This is NOT an academic exercise in GPU programming. This is production-grade acceleration that must work on consumer GPUs (GTX 1660, RTX 3060) through datacenter hardware (A100, H100), with comprehensive testing to ensure CPU-GPU result equivalence and no GPU-only code paths in critical sections.

## Architectural Principles

1. **CPU-First Design**: Every GPU operation has a CPU SIMD fallback with identical semantics
2. **Zero-Copy Where Possible**: Use CUDA Unified Memory to eliminate explicit transfers
3. **Conservative Memory Management**: Prevent OOM by batch size adaptation and memory pressure monitoring
4. **Work-Appropriate Acceleration**: Small batches stay on CPU, large batches go to GPU
5. **Deterministic Fallback**: GPU unavailability never breaks functionality, only reduces throughput

## Technical Foundations

### Memory Consistency Model

GPU acceleration introduces a new consistency boundary:

- **CPU-GPU Coherence**: Unified Memory provides automatic coherence with lazy migration
- **Ordering Guarantees**: GPU kernel completion provides acquire semantics for CPU access
- **Visibility**: CUDA stream synchronization ensures CPU observes GPU writes before next operation
- **Isolation**: Each memory space maintains independent GPU allocation contexts

### Probabilistic Guarantees

GPU floating-point operations differ from CPU:

- **Numerical Stability**: GPU reduction order must match CPU to ensure confidence score equivalence
- **Rounding Modes**: Force IEEE 754 rounding in CUDA kernels (no fast-math)
- **Accumulation**: Use Kahan summation for dot products >1024 dimensions to prevent drift
- **Validation**: Differential testing ensures CPU/GPU divergence <1e-6 for confidence scores

### Performance Boundaries

Based on profiling existing SIMD implementations:

| Operation | CPU (AVX-512) | GPU Target | Break-Even Batch Size |
|-----------|--------------|------------|----------------------|
| Cosine Similarity Batch | 2.1 us/vector | 0.3 us/vector | 64 vectors |
| Dot Product Matrix | 45 us/1K pairs | 8 us/1K pairs | 256 pairs |
| Activation Spreading (1 hop) | 850 us | 120 us | 512 nodes |
| HNSW kNN Search | 1.2 ms | 180 us | 1024 candidates |

Break-even batch sizes account for kernel launch overhead (5-20 us) and memory transfer costs.

## Operation Prioritization

### High-Value GPU Operations (Implement First)

1. **Batch Cosine Similarity** (95% of compute time in recall operations)
   - Current: 2.1 us/vector on AVX-512
   - GPU target: 0.3 us/vector on RTX 3060
   - Break-even: 64 vectors
   - Implementation: Single kernel with warp-level reduction

2. **Activation Spreading Matrix Multiply** (second hottest path)
   - Current: 45 us per 1K node-node similarities
   - GPU target: 8 us on consumer GPUs
   - Break-even: 256 node pairs
   - Implementation: Batched GEMM with cuBLAS

3. **HNSW Candidate Scoring** (during index construction and queries)
   - Current: 1.2 ms for 1K candidate evaluation
   - GPU target: 180 us
   - Break-even: 1024 candidates
   - Implementation: Custom kernel with shared memory tiling

### Medium-Value Operations (Implement If Time Permits)

4. **Pattern Completion Reconstruction** (Milestone 8 integration)
   - Sparse matrix operations for CA3 attractor dynamics
   - Only beneficial for >10K patterns in memory

5. **Consolidation Pattern Detection** (Milestone 6 integration)
   - Covariance matrix computation for semantic extraction
   - Requires large working sets (>100K episodes)

### Explicitly Deferred Operations

- **Graph Traversal**: CPU cache-friendly, not memory-bandwidth bound
- **Decay Function Application**: Trivial compute, dominated by memory access
- **Confidence Propagation**: Small batch sizes, CPU faster due to branching
- **WAL Operations**: I/O bound, GPU provides no benefit

## CUDA Kernel Architecture

### Kernel 1: Batch Cosine Similarity

**Purpose**: Compute similarity between query vector and batch of target vectors

**Memory Layout**:
```c
// Input: query (768 floats), targets (N x 768 floats, row-major)
// Output: similarities (N floats)
__global__ void batch_cosine_similarity_768(
    const float* __restrict__ query,      // 768 elements, read-only
    const float* __restrict__ targets,    // N * 768 elements, read-only
    float* __restrict__ similarities,     // N elements, write-only
    int num_targets
);
```

**Thread Configuration**:
- Block size: 256 threads (2 warps for memory coalescing)
- Grid size: (num_targets + 255) / 256 blocks
- Each thread processes one target vector

**Memory Access Pattern**:
- Query vector: Broadcast read (cached in constant memory)
- Target vectors: Coalesced 128-byte aligned reads
- Shared memory: 768 * sizeof(float) per block for query cache

**Warp-Level Optimizations**:
- Use `__shfl_down_sync` for dot product reduction within warp
- Avoid shared memory bank conflicts by padding to 32-float alignment
- Prefetch next target while computing current (software pipelining)

**Tensor Core Utilization** (Ampere+):
- Convert to FP16 for WMMA (Warp Matrix Multiply Accumulate)
- Requires input alignment to 16-byte boundaries
- Fallback to FP32 for older architectures

### Kernel 2: Activation Spreading Matrix Multiply

**Purpose**: Compute activation propagation through graph edges

**Memory Layout**:
```c
// Compute: activations_out = adjacency * activations_in * decay_factors
__global__ void spreading_matmul(
    const float* __restrict__ adjacency,     // N x N sparse matrix (CSR)
    const int* __restrict__ row_offsets,     // N+1 elements
    const int* __restrict__ col_indices,     // nnz elements
    const float* __restrict__ activations_in, // N elements
    const float* __restrict__ decay_factors,  // N elements
    float* __restrict__ activations_out,      // N elements
    int num_nodes
);
```

**Thread Configuration**:
- Block size: 128 threads per block
- Grid size: (num_nodes + 127) / 128 blocks
- Each thread computes one output activation

**Sparse Matrix Optimization**:
- Use CSR (Compressed Sparse Row) format for memory efficiency
- Coalesced reads for column indices and values
- Thread-local accumulation before global write

**Warp-Level Optimizations**:
- Dynamic parallelism for highly connected nodes (degree >32)
- Warp shuffle for reduction within node neighborhoods
- Avoid divergence by sorting edges by target node

### Kernel 3: HNSW Candidate Scoring

**Purpose**: Evaluate distance between query and candidate embeddings during HNSW search

**Memory Layout**:
```c
// Compute distances for HNSW candidate evaluation
__global__ void hnsw_score_candidates(
    const float* __restrict__ query,        // 768 elements
    const float* __restrict__ candidates,   // K x 768 elements
    float* __restrict__ distances,          // K elements
    int* __restrict__ indices,              // K elements (sorted)
    int num_candidates,
    int k
);
```

**Thread Configuration**:
- Block size: 256 threads
- Grid size: Dynamic based on num_candidates and k
- Use warp-level primitives for top-k selection

**Shared Memory Tiling**:
- Load query into shared memory (768 floats)
- Stream candidates through in tiles of 256 vectors
- Use double buffering to hide memory latency

**Top-K Selection**:
- Use warp-level bitonic sort for k <= 32
- Use shared memory radix sort for k > 32
- Maintain partial results across blocks with atomic operations

## Unified Memory Strategy

### Allocation Model

1. **Managed Memory Pool**: Pre-allocate pool at startup
   - Pool size: 75% of available GPU VRAM (conservative to avoid OOM)
   - Allocation: Fixed-size blocks (4KB, 64KB, 1MB) for different batch sizes
   - Deallocation: Return to pool, no actual free until shutdown

2. **Unified Memory Regions**:
   ```rust
   struct UnifiedMemoryRegion {
       ptr: *mut c_void,               // CUDA managed pointer
       size: usize,                    // Allocation size
       last_access: AtomicInstant,     // For LRU eviction
       preferred_location: MemoryLocation, // CPU or GPU
   }

   enum MemoryLocation {
       CPU,    // Hint: cudaMemAdviseSetPreferredLocation(cudaCpuDeviceId)
       GPU,    // Hint: cudaMemAdviseSetPreferredLocation(device_id)
       Shared, // No hint, let driver decide
   }
   ```

3. **Memory Advise Hints**:
   - `cudaMemAdviseSetReadMostly`: For query embeddings (reused across batches)
   - `cudaMemAdviseSetPreferredLocation`: For large target batches (stay on GPU)
   - `cudaMemAdviseSetAccessedBy`: For dual CPU/GPU access patterns

4. **Prefetch Strategy**:
   ```rust
   // Before kernel launch
   cudaMemPrefetchAsync(targets, size, device_id, stream);

   // After kernel completion
   cudaMemPrefetchAsync(results, size, cudaCpuDeviceId, stream);
   ```

### Fallback for Non-Unified Systems

Systems without unified memory (older GPUs, some cloud instances):

1. **Explicit Transfer Path**:
   - Allocate pinned host memory for zero-copy DMA
   - Use separate device memory allocations
   - Stream transfers asynchronously with kernel execution

2. **Double Buffering**:
   ```rust
   struct DoubleBuffer {
       host_buffer_a: *mut [f32; 768],
       host_buffer_b: *mut [f32; 768],
       device_buffer_a: *mut [f32; 768],
       device_buffer_b: *mut [f32; 768],
       current: AtomicBool,  // Which buffer is active
   }
   ```

3. **Pipeline Execution**:
   - Transfer batch N while processing batch N-1
   - Overlap compute and transfer using CUDA streams
   - Three-stage pipeline: transfer-compute-retrieve

### OOM Prevention

1. **Batch Size Adaptation**:
   ```rust
   fn adapt_batch_size(available_vram: usize, operation: GpuOperation) -> usize {
       let per_item_memory = operation.memory_footprint();
       let safety_margin = 0.8; // Use only 80% of available VRAM
       let max_batch = (available_vram as f32 * safety_margin) / per_item_memory as f32;
       max_batch.floor() as usize
   }
   ```

2. **Memory Pressure Monitoring**:
   - Query `cudaMemGetInfo` before each batch
   - Track allocation failures and reduce batch size
   - Fallback to CPU when VRAM < 20% available

3. **Graceful Degradation**:
   - Split large batches into sub-batches if OOM detected
   - If splitting fails, fallback to CPU SIMD path
   - Never fail operation due to GPU memory pressure

## CPU-GPU Hybrid Execution

### Workload Dispatcher

```rust
pub struct HybridExecutor {
    gpu_interface: Option<Arc<dyn GPUSpreadingInterface>>,
    cpu_simd_ops: Arc<dyn VectorOps>,
    performance_tracker: PerformanceTracker,
    config: HybridConfig,
}

impl HybridExecutor {
    pub fn execute_batch_similarity(
        &self,
        query: &[f32; 768],
        targets: &[[f32; 768]],
    ) -> Vec<f32> {
        let decision = self.make_dispatch_decision(targets.len());

        match decision {
            ExecutionTarget::GPU => {
                self.gpu_interface
                    .as_ref()
                    .and_then(|gpu| self.try_gpu_execution(query, targets))
                    .unwrap_or_else(|| {
                        // GPU execution failed, fallback to CPU
                        self.performance_tracker.record_gpu_fallback();
                        self.cpu_simd_ops.cosine_similarity_batch_768(query, targets)
                    })
            }
            ExecutionTarget::CPU => {
                self.cpu_simd_ops.cosine_similarity_batch_768(query, targets)
            }
        }
    }

    fn make_dispatch_decision(&self, batch_size: usize) -> ExecutionTarget {
        // Decision logic based on profiling
        if batch_size < self.config.gpu_min_batch_size {
            return ExecutionTarget::CPU; // Too small, CPU faster
        }

        if self.gpu_interface.is_none() {
            return ExecutionTarget::CPU; // No GPU available
        }

        let gpu = self.gpu_interface.as_ref().unwrap();
        if !gpu.is_available() {
            return ExecutionTarget::CPU; // GPU in error state
        }

        let historical_perf = self.performance_tracker.gpu_speedup();
        if historical_perf < 1.5 {
            return ExecutionTarget::CPU; // GPU not providing speedup
        }

        ExecutionTarget::GPU
    }
}

pub struct HybridConfig {
    pub gpu_min_batch_size: usize,     // Default: 64 for cosine similarity
    pub cpu_fallback_threshold: f64,   // Fallback if GPU slower than this
    pub performance_window_size: usize, // Moving average window for tracking
    pub force_cpu_mode: bool,          // Feature flag for debugging
}
```

### Performance Tracking

```rust
pub struct PerformanceTracker {
    cpu_latencies: RingBuffer<Duration>,
    gpu_latencies: RingBuffer<Duration>,
    gpu_fallback_count: AtomicUsize,
    last_decision_quality: AtomicF32,
}

impl PerformanceTracker {
    pub fn gpu_speedup(&self) -> f64 {
        let cpu_avg = self.cpu_latencies.average();
        let gpu_avg = self.gpu_latencies.average();
        cpu_avg / gpu_avg
    }

    pub fn should_prefer_gpu(&self, batch_size: usize) -> bool {
        let speedup = self.gpu_speedup();
        let success_rate = self.gpu_success_rate();

        // Require 2x speedup and 95% success rate to prefer GPU
        speedup >= 2.0 && success_rate >= 0.95
    }
}
```

### Automatic GPU Detection

```rust
pub fn detect_gpu_capabilities() -> Option<GpuCapabilities> {
    unsafe {
        let mut device_count: c_int = 0;
        if cudaGetDeviceCount(&mut device_count) != cudaSuccess {
            return None; // No CUDA runtime
        }

        if device_count == 0 {
            return None; // No GPUs
        }

        // Use first available GPU (could be extended for multi-GPU)
        let mut props: cudaDeviceProp = std::mem::zeroed();
        if cudaGetDeviceProperties(&mut props, 0) != cudaSuccess {
            return None;
        }

        Some(GpuCapabilities {
            max_batch: calculate_max_batch(&props),
            unified_memory: props.managedMemory != 0,
            device_name: std::ffi::CStr::from_ptr(props.name.as_ptr())
                .to_string_lossy()
                .to_string(),
            compute_capability: (props.major, props.minor),
            total_memory: props.totalGlobalMem,
            max_threads_per_block: props.maxThreadsPerBlock,
            shared_memory_per_block: props.sharedMemPerBlock,
        })
    }
}

fn calculate_max_batch(props: &cudaDeviceProp) -> usize {
    // Conservative estimation: 80% of VRAM, accounting for 768-dim vectors
    let usable_memory = (props.totalGlobalMem as f64 * 0.8) as usize;
    let bytes_per_vector = 768 * std::mem::size_of::<f32>();
    usable_memory / bytes_per_vector
}
```

## Multi-GPU Support (Deferred to Future Milestone)

Multi-GPU support adds substantial complexity and is NOT part of Milestone 12. Document the extension points for future implementation:

1. **Data Parallelism Model**: Partition memory spaces across GPUs
2. **Work Distribution**: Shard large batches across multiple devices
3. **Communication**: Use NCCL for inter-GPU transfers (not required for M12)
4. **Load Balancing**: Monitor per-GPU utilization and rebalance

This is explicitly OUT OF SCOPE for M12. Single-GPU acceleration must be proven before multi-GPU complexity is justified.

## Comprehensive Testing Strategy

### 1. Differential Testing (CPU vs GPU)

**Objective**: Ensure GPU implementations produce identical results to CPU within floating-point tolerance

```rust
#[test]
fn test_cpu_gpu_equivalence_cosine_similarity() {
    let query = random_vector_768();
    let targets = random_vectors_768(1000);

    let cpu_results = cpu_simd_ops.cosine_similarity_batch_768(&query, &targets);
    let gpu_results = gpu_interface.batch_cosine_similarity(&query, &targets).await;

    for (cpu, gpu) in cpu_results.iter().zip(gpu_results.iter()) {
        assert!((cpu - gpu).abs() < 1e-6,
                "CPU/GPU divergence: cpu={cpu}, gpu={gpu}");
    }
}

#[test]
fn test_confidence_score_stability() {
    // Test that confidence propagation through activation spreading
    // produces identical results on CPU and GPU
    let graph = create_test_graph(1000);
    let cue = create_test_cue();

    let cpu_recall = cpu_spreading_engine.recall(&cue, &graph);
    let gpu_recall = gpu_spreading_engine.recall(&cue, &graph);

    for (cpu_result, gpu_result) in cpu_recall.iter().zip(gpu_recall.iter()) {
        assert_eq!(cpu_result.memory_id, gpu_result.memory_id);
        assert!((cpu_result.confidence.raw() - gpu_result.confidence.raw()).abs() < 1e-6);
    }
}
```

### 2. Multi-Hardware Validation

**Objective**: Test on diverse GPU generations to ensure compatibility

Test matrix:
- **Maxwell** (GTX 1060): No Tensor Cores, no Unified Memory
- **Pascal** (GTX 1080): No Tensor Cores, Unified Memory available
- **Ampere** (RTX 3060): Tensor Cores, Unified Memory, FP32/FP16 mix
- **Hopper** (H100): Advanced Tensor Cores, high memory bandwidth

Validation criteria:
1. All tests pass on all GPUs
2. Performance increases with newer generations
3. Graceful degradation on older GPUs (use FP32 instead of FP16)

### 3. Memory Limit Testing

**Objective**: Ensure OOM prevention and batch adaptation work correctly

```rust
#[test]
fn test_large_batch_adaptation() {
    // Simulate limited VRAM (1GB)
    let gpu_interface = MockGpuInterface::with_vram_limit(1 * 1024 * 1024 * 1024);
    let executor = HybridExecutor::new(Some(gpu_interface), cpu_simd_ops, config);

    // Request batch larger than VRAM can hold
    let query = random_vector_768();
    let targets = random_vectors_768(100_000); // ~300MB

    // Should split into sub-batches or fallback to CPU
    let results = executor.execute_batch_similarity(&query, &targets);

    assert_eq!(results.len(), 100_000);
    // Verify no OOM occurred
    assert!(executor.performance_tracker.oom_count() == 0);
}

#[test]
fn test_oom_recovery() {
    // Simulate sudden VRAM pressure (other process allocating memory)
    let gpu_interface = MockGpuInterface::with_dynamic_vram();

    // First batch succeeds
    let result1 = execute_batch(&gpu_interface, 1000);
    assert!(result1.executed_on_gpu);

    // Simulate VRAM exhaustion
    gpu_interface.consume_vram(90);

    // Second batch should fallback to CPU
    let result2 = execute_batch(&gpu_interface, 1000);
    assert!(!result2.executed_on_gpu);
    assert_eq!(result2.results.len(), 1000); // Still produces correct results
}
```

### 4. Numerical Stability Verification

**Objective**: Ensure GPU reduction order doesn't cause numerical drift

```rust
#[test]
fn test_dot_product_precision() {
    // Test vectors designed to expose floating-point issues
    let mut a = [1.0f32; 768];
    let mut b = [1e-8f32; 768];

    // Add a large value that could cause catastrophic cancellation
    a[0] = 1e8;
    b[0] = 1e-8;

    let cpu_dot = cpu_simd_ops.dot_product_768(&a, &b);
    let gpu_dot = gpu_interface.dot_product_768(&a, &b);

    // GPU should use Kahan summation to maintain precision
    let relative_error = ((cpu_dot - gpu_dot) / cpu_dot).abs();
    assert!(relative_error < 1e-6,
            "Numerical instability: cpu={cpu_dot}, gpu={gpu_dot}");
}
```

### 5. Stress Testing

**Objective**: Validate stability under high load and concurrent access

```rust
#[test]
fn test_concurrent_gpu_access() {
    let gpu_interface = Arc::new(create_gpu_interface());
    let handles: Vec<_> = (0..16)
        .map(|_| {
            let gpu = Arc::clone(&gpu_interface);
            std::thread::spawn(move || {
                for _ in 0..1000 {
                    let query = random_vector_768();
                    let targets = random_vectors_768(100);
                    let results = gpu.batch_cosine_similarity(&query, &targets);
                    assert_eq!(results.len(), 100);
                }
            })
        })
        .collect();

    for handle in handles {
        handle.join().unwrap();
    }

    // Verify no crashes or memory corruption
    assert!(gpu_interface.is_healthy());
}

#[test]
fn test_sustained_throughput() {
    // 10 minute soak test
    let start = Instant::now();
    let duration = Duration::from_secs(600);
    let mut iteration_count = 0;

    while start.elapsed() < duration {
        let query = random_vector_768();
        let targets = random_vectors_768(1000);
        let results = gpu_interface.batch_cosine_similarity(&query, &targets);
        assert_eq!(results.len(), 1000);
        iteration_count += 1;
    }

    // Should sustain >1000 batches/second on RTX 3060
    let batches_per_sec = iteration_count as f64 / duration.as_secs_f64();
    assert!(batches_per_sec > 1000.0,
            "Insufficient throughput: {batches_per_sec} batches/sec");
}
```

## Task Breakdown

### Task 001: GPU Profiling and Baseline Establishment (2 days)

**Objective**: Quantify current CPU SIMD performance and identify GPU-suitable workloads

**Deliverables**:
1. Flamegraph profiling of activation spreading operations
2. Identify top 5 hottest code paths (by CPU time)
3. Measure current throughput and latency for each operation
4. Calculate theoretical GPU speedup based on memory bandwidth
5. Document break-even batch sizes for each operation

**Acceptance Criteria**:
- Profiling data showing where >80% of CPU time is spent
- Quantified speedup predictions for each GPU candidate operation
- Decision matrix: which operations to accelerate first

**Files to Modify/Create**:
- `engram-core/benches/gpu_candidate_profiling.rs` (new)
- `roadmap/milestone-12/profiling_report.md` (new)

**Integration Points**:
- Profile existing `compute::batch_cosine_similarity_768`
- Profile existing `activation::parallel::ParallelSpreadingEngine`
- Profile existing `index::hnsw_search::HnswIndex::search`

**Testing Approach**:
- Run benchmarks on production-like workloads
- Compare against FAISS GPU for validation
- Verify profiling overhead <5%

**Dependencies**: None (start immediately)

---

### Task 002: CUDA Kernel Development Environment Setup (2 days)

**Objective**: Establish build system for CUDA code with Rust FFI integration

**Deliverables**:
1. CUDA compilation integrated into `Cargo.build`
2. FFI bindings for CUDA runtime API
3. Error handling wrapper for CUDA status codes
4. Basic kernel compilation and linking test

**Acceptance Criteria**:
- `cargo build` successfully compiles .cu files
- Can launch trivial CUDA kernel from Rust
- CUDA errors propagate to Rust `Result` types
- Works on systems without CUDA (no-op fallback)

**Files to Modify/Create**:
- `engram-core/build.rs` (modify)
- `engram-core/src/compute/cuda/mod.rs` (new)
- `engram-core/src/compute/cuda/ffi.rs` (new)
- `engram-core/cuda/common.cuh` (new)
- `engram-core/cuda/Makefile` (new)

**Integration Points**:
- Extend existing `compute::CpuCapability` to include GPU detection
- Add `GpuCapability` enum alongside CPU capabilities
- Integrate with existing `compute::create_vector_ops()` dispatch

**Testing Approach**:
- Unit test CUDA initialization and device query
- Test build on CI without CUDA toolkit (should compile)
- Validate FFI bindings with basic kernel

**Dependencies**: None (parallel with Task 001)

---

### Task 003: Batch Cosine Similarity CUDA Kernel (3 days)

**Objective**: Implement and optimize first production CUDA kernel

**Deliverables**:
1. `batch_cosine_similarity_768` kernel implementation
2. Warp-level reduction optimized for 768 dimensions
3. Rust wrapper with CPU fallback
4. Performance benchmarks vs CPU SIMD

**Acceptance Criteria**:
- Achieves >3x speedup over AVX-512 for batches >64 vectors
- CPU-GPU result divergence <1e-6
- Gracefully fallbacks to CPU if GPU unavailable
- Handles batch sizes from 1 to 100,000

**Files to Modify/Create**:
- `engram-core/cuda/kernels/cosine_similarity.cu` (new)
- `engram-core/src/compute/cuda/cosine_similarity.rs` (new)
- `engram-core/benches/gpu_cosine_similarity.rs` (new)
- `engram-core/src/compute/dispatch.rs` (modify to add GPU path)

**Integration Points**:
- Modify `VectorOps::cosine_similarity_batch_768` to dispatch to GPU
- Integrate with existing `BatchEngine` for batch operations
- Update `activation::recall` to use GPU-accelerated similarity

**Testing Approach**:
- Differential testing against scalar CPU implementation
- Property-based testing with random vectors
- Benchmark against cuBLAS `cublasSdot` for validation
- Test on GTX 1060, RTX 3060, and A100

**Dependencies**: Task 002 (CUDA environment setup)

---

### Task 004: Unified Memory Allocator (3 days)

**Objective**: Implement zero-copy memory management for GPU operations

**Deliverables**:
1. Unified memory allocation pool with RAII wrappers
2. Memory advise hints for CPU/GPU locality
3. Prefetch automation based on access patterns
4. Fallback to pinned memory for non-unified systems

**Acceptance Criteria**:
- Zero explicit cudaMemcpy calls in hot paths
- Automatic prefetching hides 80% of transfer latency
- Works on Pascal+ (unified memory) and Maxwell (pinned memory)
- OOM prevention via batch size adaptation

**Files to Modify/Create**:
- `engram-core/src/compute/cuda/unified_memory.rs` (new)
- `engram-core/src/compute/cuda/memory_pool.rs` (new)
- `engram-core/src/compute/cuda/prefetch.rs` (new)

**Integration Points**:
- Integrate with `activation::memory_pool::ActivationMemoryPool`
- Use unified memory for `GPUActivationBatch` allocations
- Update `BatchEngine` to use unified memory regions

**Testing Approach**:
- Test on system with unified memory (Pascal+)
- Test on system without unified memory (fallback to pinned)
- Stress test with concurrent allocations
- Verify no memory leaks with valgrind + CUDA memcheck

**Dependencies**: Task 003 (first kernel operational)

---

### Task 005: Activation Spreading Matrix Multiply Kernel (3 days)

**Objective**: GPU-accelerate the second hottest operation (activation propagation)

**Deliverables**:
1. Sparse matrix multiply kernel (CSR format)
2. Warp-level reduction for node neighborhoods
3. Integration with existing `ParallelSpreadingEngine`
4. Performance benchmarks vs CPU parallel spreading

**Acceptance Criteria**:
- Achieves >5x speedup over CPU for graphs >512 nodes
- Correctly handles sparse graphs (average degree <10)
- Maintains confidence score precision (divergence <1e-6)
- Graceful fallback to CPU for small graphs

**Files to Modify/Create**:
- `engram-core/cuda/kernels/spreading_matmul.cu` (new)
- `engram-core/src/compute/cuda/spreading.rs` (new)
- `engram-core/src/activation/parallel.rs` (modify)

**Integration Points**:
- Modify `ParallelSpreadingEngine::spread_activation` to use GPU
- Convert existing adjacency representation to CSR format
- Update `activation::accumulator` to consume GPU results

**Testing Approach**:
- Differential testing against CPU spreading implementation
- Test on graphs with varying sparsity (dense, sparse, scale-free)
- Validate confidence propagation matches CPU exactly
- Benchmark on production-like graph topologies

**Dependencies**: Task 004 (unified memory for graph data)

---

### Task 006: HNSW Candidate Scoring Kernel (2 days)

**Objective**: Accelerate vector index operations during search

**Deliverables**:
1. Batch distance computation for HNSW candidates
2. Warp-level top-k selection (replaces CPU sort)
3. Integration with existing `HnswIndex`
4. Performance benchmarks vs CPU HNSW search

**Acceptance Criteria**:
- Achieves >4x speedup for candidate set >1024 vectors
- Top-k results identical to CPU implementation
- Maintains HNSW recall accuracy (no degradation)
- Works with both L2 distance and cosine similarity

**Files to Modify/Create**:
- `engram-core/cuda/kernels/hnsw_scoring.cu` (new)
- `engram-core/src/compute/cuda/hnsw.rs` (new)
- `engram-core/src/index/hnsw_search.rs` (modify)

**Integration Points**:
- Modify `HnswIndex::search` to use GPU for candidate scoring
- Use unified memory for candidate embeddings
- Update `index::hnsw_construction` to use GPU during build

**Testing Approach**:
- Differential testing against CPU HNSW implementation
- Validate recall@10 and recall@100 metrics
- Benchmark on 1M vector index
- Test with different distance metrics

**Dependencies**: Task 004 (unified memory for embeddings)

---

### Task 007: CPU-GPU Hybrid Executor (2 days)

**Objective**: Implement intelligent dispatch between CPU and GPU execution

**Deliverables**:
1. `HybridExecutor` with performance-based dispatch
2. Automatic GPU capability detection at runtime
3. Performance tracking and adaptive decision-making
4. Feature flag for forcing CPU-only mode

**Acceptance Criteria**:
- Correctly routes small batches to CPU, large batches to GPU
- Tracks historical performance and adjusts dispatch decisions
- Gracefully handles GPU failures by falling back to CPU
- Works correctly when GPU is unavailable (CPU-only systems)

**Files to Modify/Create**:
- `engram-core/src/compute/cuda/hybrid.rs` (new)
- `engram-core/src/compute/cuda/performance_tracker.rs` (new)
- `engram-core/src/compute/dispatch.rs` (modify)

**Integration Points**:
- Replace direct GPU calls with `HybridExecutor` dispatch
- Integrate with `metrics::hardware::GpuMetrics`
- Update `BatchEngine` to use hybrid execution

**Testing Approach**:
- Test dispatch logic with mock GPU interface
- Verify CPU fallback when GPU returns errors
- Benchmark decision overhead (<1% of total latency)
- Test with `force_cpu_mode` feature flag

**Dependencies**: Tasks 003, 005, 006 (all kernels operational)

---

### Task 008: Multi-Hardware Differential Testing (2 days)

**Objective**: Validate correctness across diverse GPU architectures

**Deliverables**:
1. Test suite running on Maxwell, Pascal, Ampere, Hopper
2. Numerical stability validation across architectures
3. Performance regression tests per GPU generation
4. CI integration for GPU testing (where available)

**Acceptance Criteria**:
- All tests pass on all GPU generations
- CPU-GPU divergence <1e-6 on all architectures
- Performance increases with newer generations
- Older GPUs gracefully degrade (FP32 instead of FP16)

**Files to Modify/Create**:
- `engram-core/tests/gpu_differential.rs` (new)
- `engram-core/tests/gpu_numerical_stability.rs` (new)
- `.github/workflows/gpu-tests.yml` (new)

**Integration Points**:
- Test all GPU kernels against CPU SIMD baselines
- Validate confidence score propagation end-to-end
- Test with production workload patterns

**Testing Approach**:
- Run on physical hardware (GTX 1060, RTX 3060, A100)
- Test with various batch sizes (1, 64, 1024, 100K)
- Validate under memory pressure (limited VRAM)
- Test concurrent access from multiple threads

**Dependencies**: Task 007 (hybrid executor complete)

---

### Task 009: Memory Pressure and OOM Handling (2 days)

**Objective**: Ensure robust operation under VRAM constraints

**Deliverables**:
1. Batch size adaptation based on available VRAM
2. OOM recovery via automatic CPU fallback
3. Memory pressure monitoring and telemetry
4. Graceful degradation under constrained resources

**Acceptance Criteria**:
- Never crashes due to OOM (always falls back to CPU)
- Automatically splits large batches when VRAM insufficient
- Monitors and reports VRAM usage to metrics
- Works correctly on GPUs with 4GB, 8GB, 24GB VRAM

**Files to Modify/Create**:
- `engram-core/src/compute/cuda/memory_pressure.rs` (new)
- `engram-core/src/compute/cuda/oom_recovery.rs` (new)
- `engram-core/tests/gpu_memory_limits.rs` (new)

**Integration Points**:
- Integrate with `BatchEngine` for batch splitting
- Update `HybridExecutor` to query VRAM before dispatch
- Add metrics for OOM events and fallback counts

**Testing Approach**:
- Simulate low VRAM conditions (mock interface)
- Test with deliberately oversized batches
- Stress test with concurrent allocations
- Validate recovery without data loss

**Dependencies**: Task 007 (hybrid executor)

---

### Task 010: Performance Benchmarking and Optimization (2 days)

**Objective**: Validate performance targets and identify optimization opportunities

**Deliverables**:
1. Comprehensive benchmark suite vs CPU SIMD
2. Comparison against FAISS GPU and cuBLAS
3. Performance report with speedup analysis
4. Optimization recommendations for future work

**Acceptance Criteria**:
- Achieves >3x speedup over CPU SIMD for target operations
- Performance meets or exceeds FAISS GPU for similarity search
- Identifies bottlenecks and optimization opportunities
- Provides baseline for future performance regression detection

**Files to Modify/Create**:
- `engram-core/benches/gpu_vs_cpu_comprehensive.rs` (new)
- `roadmap/milestone-12/performance_report.md` (new)
- `roadmap/milestone-12/optimization_opportunities.md` (new)

**Integration Points**:
- Benchmark all GPU-accelerated operations
- Compare against existing CPU SIMD benchmarks
- Validate performance on consumer and datacenter GPUs

**Testing Approach**:
- Benchmark on GTX 1660, RTX 3060, RTX 4090, A100, H100
- Test with batch sizes: 1, 16, 64, 256, 1024, 10K, 100K
- Measure latency (P50, P90, P99) and throughput
- Profile GPU kernel occupancy and memory bandwidth

**Dependencies**: Tasks 003, 005, 006 (all kernels operational)

---

### Task 011: Documentation and Production Readiness (2 days)

**Objective**: Document GPU acceleration and create deployment guides

**Deliverables**:
1. GPU acceleration architecture documentation
2. Deployment guide for GPU-enabled clusters
3. Troubleshooting guide for common GPU issues
4. Performance tuning guide for different GPU types

**Acceptance Criteria**:
- External operator can deploy GPU-accelerated Engram
- Documentation covers consumer and datacenter GPUs
- Troubleshooting guide resolves common CUDA errors
- Tuning guide provides recommended configurations per GPU

**Files to Modify/Create**:
- `docs/operations/gpu-deployment.md` (new)
- `docs/reference/gpu-architecture.md` (new)
- `docs/operations/gpu-troubleshooting.md` (new)
- `docs/operations/gpu-performance-tuning.md` (new)
- `milestones.md` (update M12 status)

**Integration Points**:
- Document integration with existing monitoring stack
- Provide configuration examples for different deployments
- Include CUDA toolkit installation instructions

**Testing Approach**:
- Validate documentation with fresh deployment on clean system
- Test troubleshooting guide against common error scenarios
- Verify tuning recommendations match benchmark results

**Dependencies**: Task 010 (performance validation complete)

---

### Task 012: Integration Testing and Acceptance (1 day)

**Objective**: End-to-end validation of GPU acceleration in production scenarios

**Deliverables**:
1. Integration tests with existing Milestone 1-8 features
2. Multi-tenant GPU isolation validation
3. Production workload stress testing
4. Acceptance criteria validation

**Acceptance Criteria**:
- All existing tests pass with GPU acceleration enabled
- Multi-tenant memory spaces maintain GPU isolation
- Sustained 10K+ operations/second under load
- CPU-only fallback maintains identical behavior

**Files to Modify/Create**:
- `engram-core/tests/integration/gpu_acceleration.rs` (new)
- `roadmap/milestone-12/ACCEPTANCE_REPORT.md` (new)

**Integration Points**:
- Test with Milestone 6 consolidation operations
- Test with Milestone 7 multi-tenant isolation
- Test with Milestone 8 pattern completion (if available)

**Testing Approach**:
- Run full test suite with GPU enabled
- Compare results with CPU-only baseline
- Stress test with production traffic patterns
- Validate performance targets met

**Dependencies**: Task 011 (all features complete)

## Risk Analysis and Mitigation

### Risk 1: Floating-Point Determinism

**Probability**: HIGH
**Impact**: CRITICAL
**Description**: GPU reduction order may differ from CPU, causing confidence score divergence

**Mitigation**:
1. Force IEEE 754 rounding modes in CUDA kernels (no `--use_fast_math`)
2. Use Kahan summation for dot products to maintain precision
3. Implement bit-exact reduction order matching CPU implementation
4. Add differential testing with tolerance <1e-6

**Validation**: Test with adversarial inputs designed to expose numerical instability

---

### Risk 2: OOM on Consumer GPUs

**Probability**: MEDIUM
**Impact**: HIGH
**Description**: Large batches may exhaust VRAM on 4GB-8GB consumer GPUs

**Mitigation**:
1. Conservative batch size calculation (80% of available VRAM)
2. Automatic batch splitting when OOM detected
3. Graceful fallback to CPU when VRAM insufficient
4. Continuous VRAM monitoring before each batch

**Validation**: Test on RTX 3060 (8GB) and GTX 1660 (6GB) with deliberately large batches

---

### Risk 3: GPU Unavailability in Production

**Probability**: HIGH (many deployments are CPU-only)
**Impact**: LOW (CPU fallback prevents functional impact)
**Description**: Many production environments don't have GPUs, or CUDA runtime is unavailable

**Mitigation**:
1. CPU-first design: all operations have SIMD fallback
2. Graceful detection: missing CUDA runtime doesn't crash
3. Feature flag: `force_cpu_mode` for debugging
4. Clear documentation: GPU is optional, not required

**Validation**: Test deployment on system without CUDA toolkit or GPU hardware

---

### Risk 4: Multi-Tenant GPU Contention

**Probability**: MEDIUM
**Impact**: MEDIUM
**Description**: Multiple memory spaces competing for GPU resources may cause latency spikes

**Mitigation**:
1. CUDA stream per memory space for isolation
2. GPU queue depth limiting to prevent starvation
3. Fairness scheduler for GPU access across spaces
4. Fallback to CPU if GPU queue depth exceeds threshold

**Validation**: Stress test with 10+ memory spaces making concurrent GPU requests

---

### Risk 5: Kernel Launch Overhead

**Probability**: LOW
**Impact**: MEDIUM
**Description**: Small batches may be slower on GPU due to kernel launch overhead (5-20 us)

**Mitigation**:
1. Performance tracking with adaptive dispatch decisions
2. Conservative break-even batch sizes (>64 vectors)
3. Batch aggregation: accumulate small requests before GPU launch
4. Hybrid executor prefers CPU for known-small batches

**Validation**: Benchmark with batch sizes 1, 2, 4, 8, 16, 32, 64 to determine break-even

---

### Risk 6: CUDA Version Incompatibility

**Probability**: MEDIUM
**Impact**: MEDIUM
**Description**: Different CUDA toolkit versions may have incompatible APIs

**Mitigation**:
1. Target CUDA 11.0+ (widely available, stable API)
2. Runtime version check with graceful degradation
3. Conditional compilation for newer CUDA features
4. Clear documentation of required CUDA version

**Validation**: Test with CUDA 11.0, 11.8, 12.0, 12.3 on CI

## Performance Targets

### Baseline (CPU SIMD - AVX-512)

| Operation | Batch Size | Latency | Throughput |
|-----------|-----------|---------|------------|
| Cosine Similarity | 1,000 | 2.1 ms | 476K ops/sec |
| Activation Spreading (1 hop) | 1,000 nodes | 850 us | 1.18K graphs/sec |
| HNSW kNN Search | k=10, 10K index | 1.2 ms | 833 queries/sec |

### Target (GPU - RTX 3060)

| Operation | Batch Size | Latency | Speedup | Throughput |
|-----------|-----------|---------|---------|------------|
| Cosine Similarity | 1,000 | 300 us | 7.0x | 3.33M ops/sec |
| Activation Spreading (1 hop) | 1,000 nodes | 120 us | 7.1x | 8.33K graphs/sec |
| HNSW kNN Search | k=10, 10K index | 180 us | 6.7x | 5.56K queries/sec |

### Stretch (GPU - A100)

| Operation | Batch Size | Latency | Speedup | Throughput |
|-----------|-----------|---------|---------|------------|
| Cosine Similarity | 10,000 | 800 us | 26.3x | 12.5M ops/sec |
| Activation Spreading (1 hop) | 10,000 nodes | 450 us | 18.9x | 22.2K graphs/sec |
| HNSW kNN Search | k=10, 100K index | 850 us | 14.1x | 11.8K queries/sec |

**Acceptance Criteria**: All operations achieve >3x speedup on consumer GPUs for production batch sizes

## Dependencies and Critical Path

```
Critical Path (16 days total):

Day 1-2:   Task 001 (Profiling) + Task 002 (CUDA Setup) [parallel]
Day 3-5:   Task 003 (Cosine Similarity Kernel) [depends on 002]
Day 6-8:   Task 004 (Unified Memory) [depends on 003]
Day 9-11:  Task 005 (Spreading Kernel) [depends on 004]
Day 12-13: Task 006 (HNSW Kernel) [depends on 004]
Day 14-15: Task 007 (Hybrid Executor) [depends on 003, 005, 006]
Day 16-17: Task 008 (Multi-Hardware Testing) [depends on 007]
Day 18-19: Task 009 (OOM Handling) [depends on 007]
Day 20:    Task 010 (Performance Benchmarking) [depends on 008, 009]
Day 21-22: Task 011 (Documentation) [depends on 010]
Day 23:    Task 012 (Integration Testing) [depends on 011]
```

Parallel opportunities:
- Tasks 001 and 002 can run in parallel
- Tasks 005 and 006 can partially overlap (both use unified memory from 004)
- Tasks 008 and 009 can partially overlap (different focus areas)

## Acceptance Criteria (Milestone-Level)

1. **Correctness**:
   - [ ] All CPU-GPU differential tests pass (<1e-6 divergence)
   - [ ] All existing tests pass with GPU enabled
   - [ ] Multi-tenant isolation maintained with GPU operations

2. **Performance**:
   - [ ] Achieves >3x speedup over CPU SIMD for target operations
   - [ ] Break-even batch sizes match predictions (±20%)
   - [ ] GPU utilization >70% during batch operations

3. **Robustness**:
   - [ ] Zero crashes due to OOM (graceful fallback)
   - [ ] Works on GPUs with 4GB-80GB VRAM
   - [ ] CPU fallback maintains identical behavior
   - [ ] Sustained 10K+ operations/second under load

4. **Compatibility**:
   - [ ] Tests pass on Maxwell, Pascal, Ampere, Hopper
   - [ ] Works on systems without CUDA toolkit (CPU-only)
   - [ ] Graceful degradation on older GPU architectures

5. **Documentation**:
   - [ ] Deployment guide validated by external operator
   - [ ] Troubleshooting guide resolves common issues
   - [ ] Performance tuning guide tested on all GPU types

6. **Production Readiness**:
   - [ ] GPU metrics integrated with monitoring stack
   - [ ] OOM and fallback events properly logged
   - [ ] Feature flag allows forcing CPU-only mode

## Future Work (Explicitly Out of Scope for M12)

1. **Multi-GPU Support**: Data parallelism across multiple GPUs
2. **Tensor Core Optimization**: FP16/BF16 mixed precision for Ampere+
3. **Custom Memory Allocator**: Replace cudaMalloc with specialized allocator
4. **Persistent Kernels**: Keep kernels resident to eliminate launch overhead
5. **CUDA Graphs**: Pre-record kernel launch sequences for lower latency
6. **ROCm Support**: AMD GPU compatibility via HIP
7. **Distributed GPU**: GPU acceleration across multiple nodes
8. **Dynamic Parallelism**: GPU kernels launching child kernels

These are deferred until single-GPU acceleration is proven in production.

## Conclusion

This milestone establishes GPU acceleration as a production-grade performance enhancement for Engram, not an academic experiment. The implementation prioritizes correctness over raw performance, maintains strict CPU-GPU equivalence, and provides comprehensive fallback mechanisms.

The critical insight: GPU acceleration must be invisible to users. Systems without GPUs work identically to systems with GPUs, just slower. This design philosophy ensures Engram remains deployable in any environment while providing significant performance improvements where GPU resources are available.

Success is measured not by peak GPU throughput, but by the seamlessness of the hybrid CPU-GPU execution model and the robustness of the fallback mechanisms.
