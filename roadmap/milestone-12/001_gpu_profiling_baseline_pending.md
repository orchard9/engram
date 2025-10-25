# Task 001: GPU Profiling and Baseline Establishment

**Status**: Pending
**Estimated Duration**: 2 days
**Priority**: Critical (blocks all GPU work)
**Owner**: Performance Engineer

## Objective

Quantify current CPU SIMD performance and identify GPU-suitable workloads through rigorous profiling. Establish performance baselines and calculate theoretical speedups to prioritize GPU acceleration efforts.

## Background

Before writing any GPU code, we must understand where CPU time is actually spent. Premature GPU optimization wastes engineering effort on operations that aren't bottlenecks. This task establishes data-driven priorities for GPU acceleration.

## Research Foundation

GPU acceleration is NOT free - every kernel launch incurs 5-20μs overhead, memory transfers cost bandwidth, and synchronization delays compound. Research principle: measure first, optimize second. Profile-guided GPU acceleration focuses engineering effort where it matters most.

**Memory vs Compute Bound Classification:**
- **Memory-bound** (cosine similarity): 768-dim vectors = 3KB per vector, arithmetic intensity 0.13 FLOPS/byte, speedup limited by bandwidth ratio (GPU 360 GB/s vs CPU 50 GB/s = 7.2x theoretical)
- **Compute-bound** (sparse matrix multiply): higher arithmetic intensity, benefits more from GPU parallelism (13 TFLOPS GPU vs 2 TFLOPS CPU = 6.5x theoretical)

**Break-even equation:**
```
Total_CPU_Time = Per_Item_CPU_Latency × Batch_Size
Total_GPU_Time = Kernel_Launch_Overhead + Per_Item_GPU_Latency × Batch_Size
Break_Even_Size = Launch_Overhead / (Per_Item_CPU - Per_Item_GPU)
```

**Known baseline targets from SIMD profiling:**
- Cosine similarity: CPU 2.1 μs/vector (AVX-512), GPU target 0.3 μs/vector, break-even 64 vectors
- Activation spreading: CPU 850 μs, GPU target 120 μs, break-even 512 nodes
- HNSW kNN search: CPU 1.2 ms, GPU target 180 μs, break-even 1024 candidates

Statistical rigor requirements: minimum 100 samples, report P50/P90/P99 (not just mean), coefficient of variation < 5%, warm cache measurements (representative of production).

## Deliverables

1. **Flamegraph Profiling Data**
   - Profile activation spreading operations under production load
   - Profile batch recall with 1K-10K queries
   - Profile HNSW index construction and search
   - Identify top 5 hottest code paths by CPU time

2. **Performance Baseline Report**
   - Current throughput for each operation (ops/sec)
   - Current latency distribution (P50, P90, P99)
   - Memory bandwidth utilization
   - CPU utilization per operation

3. **GPU Speedup Predictions**
   - Theoretical speedup based on memory bandwidth
   - Theoretical speedup based on compute intensity
   - Break-even batch sizes accounting for kernel launch overhead
   - Risk assessment for each candidate operation

4. **Decision Matrix**
   - Prioritized list of operations to accelerate
   - Estimated ROI (speedup vs implementation effort)
   - Dependency analysis (which operations to do first)

## Technical Specification

### Profiling Methodology

Use `pprof` with flamegraph generation:

```rust
// engram-core/benches/gpu_candidate_profiling.rs

use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId};
use pprof::criterion::{Output, PProfProfiler};
use engram_core::compute::get_vector_ops;
use engram_core::activation::ParallelSpreadingEngine;

fn profile_batch_cosine_similarity(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_cosine_similarity");

    for batch_size in [16, 64, 256, 1024, 4096, 16384] {
        group.bench_with_input(
            BenchmarkId::from_parameter(batch_size),
            &batch_size,
            |b, &size| {
                let query = random_vector_768();
                let targets = random_vectors_768(size);
                let ops = get_vector_ops();

                b.iter(|| {
                    ops.cosine_similarity_batch_768(&query, &targets)
                });
            },
        );
    }
    group.finish();
}

fn profile_activation_spreading(c: &mut Criterion) {
    let mut group = c.benchmark_group("activation_spreading");

    for node_count in [100, 500, 1000, 5000, 10000] {
        group.bench_with_input(
            BenchmarkId::from_parameter(node_count),
            &node_count,
            |b, &count| {
                let graph = create_test_graph(count);
                let cues = create_test_cues(10);
                let engine = ParallelSpreadingEngine::new(Default::default());

                b.iter(|| {
                    engine.spread_activation(&graph, &cues)
                });
            },
        );
    }
    group.finish();
}

criterion_group! {
    name = benches;
    config = Criterion::default().with_profiler(PProfProfiler::new(100, Output::Flamegraph(None)));
    targets = profile_batch_cosine_similarity, profile_activation_spreading
}
criterion_main!(benches);
```

### Performance Baseline Measurements

For each operation, measure:

1. **Throughput**: Operations per second
2. **Latency**: P50, P90, P99, P99.9 in microseconds
3. **CPU Utilization**: % of CPU time spent in operation
4. **Memory Bandwidth**: Bytes transferred per operation
5. **Cache Performance**: L1/L2/L3 hit rates

### Theoretical Speedup Calculation

GPU speedup is bounded by:

```
Theoretical Speedup = min(
    Memory_BW_GPU / Memory_BW_CPU,
    Compute_FLOPS_GPU / Compute_FLOPS_CPU
)
```

For memory-bound operations (like cosine similarity):
- CPU (DDR4-3200): ~25 GB/s per channel (2-4 channels)
- GPU (RTX 3060): 360 GB/s VRAM bandwidth
- Theoretical speedup: ~5-7x (accounting for overhead)

For compute-bound operations (like matrix multiply):
- CPU (AVX-512): ~2 TFLOPS (8 cores)
- GPU (RTX 3060): 13 TFLOPS (FP32)
- Theoretical speedup: ~6x

### Break-Even Analysis

Account for kernel launch overhead:

```
Break_Even_Batch_Size = Kernel_Launch_Overhead / (CPU_Per_Item - GPU_Per_Item)
```

Typical kernel launch overhead: 5-20 microseconds

Example:
- CPU cosine similarity: 2.1 us/vector
- GPU cosine similarity: 0.3 us/vector
- Launch overhead: 10 us
- Break-even: 10 us / (2.1 - 0.3) = 5.6 vectors

Round up to next power of 2: 8 vectors minimum batch size

But add safety margin: use 64 vectors as practical break-even

## Acceptance Criteria

1. **Profiling Data Quality**
   - [ ] Flamegraphs show function-level CPU time breakdown
   - [ ] Top 5 hottest functions identified with >10% CPU time each
   - [ ] Profiling overhead measured and <5%

2. **Baseline Accuracy**
   - [ ] All measurements repeated 100+ times for statistical significance
   - [ ] Standard deviation <5% of mean for latency measurements
   - [ ] Throughput measurements under sustained load (not cold start)

3. **Speedup Predictions**
   - [ ] Theoretical speedup calculated for each candidate operation
   - [ ] Memory vs compute bound analysis for each operation
   - [ ] Break-even batch sizes calculated with safety margin

4. **Decision Matrix**
   - [ ] Operations ranked by ROI (speedup * frequency)
   - [ ] Top 3 operations identified for M12 GPU acceleration
   - [ ] Dependencies mapped (which operations to implement first)

## Integration Points

### Existing Code to Profile

1. **`engram-core/src/compute/dispatch.rs`**
   - Profile `VectorOps::cosine_similarity_batch_768`
   - Profile `VectorOps::dot_product_768`
   - Profile `VectorOps::weighted_average_768`

2. **`engram-core/src/activation/parallel.rs`**
   - Profile `ParallelSpreadingEngine::spread_activation`
   - Profile `accumulator::ActivationAccumulator::accumulate`
   - Profile `simd_optimization::SimdActivationMapper`

3. **`engram-core/src/index/hnsw_search.rs`**
   - Profile `HnswIndex::search`
   - Profile candidate scoring during search
   - Profile layer traversal operations

4. **`engram-core/src/batch/engine.rs`**
   - Profile `BatchEngine::batch_similarity_search`
   - Profile `BatchEngine::batch_recall`

## Testing Approach

### Test 1: Profiling Overhead Validation

Verify profiling doesn't significantly alter performance:

```rust
#[test]
fn test_profiling_overhead() {
    let query = random_vector_768();
    let targets = random_vectors_768(1000);
    let ops = get_vector_ops();

    // Measure without profiling
    let start = Instant::now();
    for _ in 0..100 {
        ops.cosine_similarity_batch_768(&query, &targets);
    }
    let baseline = start.elapsed();

    // Measure with profiling
    let _guard = pprof::ProfilerGuard::new(100).unwrap();
    let start = Instant::now();
    for _ in 0..100 {
        ops.cosine_similarity_batch_768(&query, &targets);
    }
    let with_profiling = start.elapsed();

    let overhead = (with_profiling.as_secs_f64() / baseline.as_secs_f64()) - 1.0;
    assert!(overhead < 0.05, "Profiling overhead {overhead} exceeds 5%");
}
```

### Test 2: Baseline Repeatability

Ensure measurements are stable:

```rust
#[test]
fn test_baseline_repeatability() {
    let query = random_vector_768();
    let targets = random_vectors_768(1000);
    let ops = get_vector_ops();

    let mut latencies = Vec::new();
    for _ in 0..100 {
        let start = Instant::now();
        ops.cosine_similarity_batch_768(&query, &targets);
        latencies.push(start.elapsed().as_micros());
    }

    let mean = latencies.iter().sum::<u128>() / latencies.len() as u128;
    let variance = latencies.iter()
        .map(|&x| (x as f64 - mean as f64).powi(2))
        .sum::<f64>() / latencies.len() as f64;
    let stddev = variance.sqrt();
    let cv = stddev / mean as f64; // Coefficient of variation

    assert!(cv < 0.05, "Latency CV {cv} exceeds 5%");
}
```

### Test 3: Break-Even Calculation Validation

Verify break-even math matches empirical data:

```rust
#[test]
fn test_break_even_calculation() {
    let cpu_per_item = 2.1; // microseconds
    let gpu_per_item = 0.3; // microseconds
    let launch_overhead = 10.0; // microseconds

    let break_even = launch_overhead / (cpu_per_item - gpu_per_item);
    assert!(break_even < 10.0, "Break-even too high: {break_even}");

    // Practical break-even should be higher due to variance
    let practical_break_even = (break_even * 1.5).ceil().next_power_of_two();
    assert!(practical_break_even >= 16);
}
```

## Files to Create/Modify

### New Files
- `/Users/jordanwashburn/Workspace/orchard9/engram/engram-core/benches/gpu_candidate_profiling.rs`
- `/Users/jordanwashburn/Workspace/orchard9/engram/roadmap/milestone-12/profiling_report.md`
- `/Users/jordanwashburn/Workspace/orchard9/engram/roadmap/milestone-12/gpu_speedup_analysis.md`
- `/Users/jordanwashburn/Workspace/orchard9/engram/roadmap/milestone-12/operation_decision_matrix.md`

### Modified Files
- `/Users/jordanwashburn/Workspace/orchard9/engram/engram-core/Cargo.toml` (add pprof benchmark dependency)

## Dependencies

**Blocking**: None (start immediately)

**Blocked By This Task**:
- Task 003 (Cosine Similarity Kernel) - needs break-even batch sizes
- Task 005 (Activation Spreading Kernel) - needs profiling data
- Task 006 (HNSW Kernel) - needs ROI analysis

## Risk Assessment

### Risk: Profiling Under Unrealistic Workloads

**Mitigation**: Use production-like data distributions and access patterns

### Risk: CPU Optimization Makes GPU Unnecessary

**Mitigation**: Profile both optimized and unoptimized CPU paths to understand ceiling

### Risk: Break-Even Calculations Don't Match Reality

**Mitigation**: Add 50% safety margin to calculated break-even batch sizes

## Success Metrics

1. **Profiling Coverage**: >90% of CPU time attributed to known functions
2. **Measurement Precision**: Coefficient of variation <5% for latency
3. **Speedup Predictions**: Within 30% of actual GPU performance (validated in Task 010)
4. **Decision Confidence**: Top 3 operations account for >70% of total CPU time

## Deliverable Artifacts

1. **Profiling Report** (`profiling_report.md`):
   - Flamegraph analysis
   - Top 5 hottest functions with CPU time percentages
   - Memory bandwidth utilization per operation
   - Recommendations for GPU acceleration

2. **Speedup Analysis** (`gpu_speedup_analysis.md`):
   - Theoretical speedup calculations
   - Memory vs compute bound classification
   - Break-even batch size analysis
   - Expected performance improvements

3. **Decision Matrix** (`operation_decision_matrix.md`):
   - Ranked list of GPU candidate operations
   - ROI analysis (speedup * frequency * implementation effort)
   - Implementation order with dependencies
   - Risk assessment per operation

## Notes

This task is purely analytical - no GPU code is written. The goal is data-driven decision making before committing to GPU implementation. If profiling shows insufficient CPU bottlenecks, GPU acceleration may not be worthwhile.

The profiling data from this task will be referenced throughout M12 to validate that GPU acceleration is providing the predicted speedups.
