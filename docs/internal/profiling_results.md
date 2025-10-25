# Profiling Infrastructure and Baseline Performance Results

**Created:** 2025-10-25
**Status:** Infrastructure Complete
**Next Steps:** Execute profiling workloads and document actual hotspots

## Executive Summary

This document describes the profiling infrastructure established for Engram's performance optimization initiative (Milestone 10). The infrastructure provides systematic profiling capabilities using flamegraphs and micro-benchmarks to identify optimization targets for Zig kernel rewrites.

## Infrastructure Components

### 1. Profiling Harness (`/engram-core/benches/profiling_harness.rs`)

A comprehensive benchmark that exercises all major computational hot paths with realistic workloads:

**Graph Configuration:**
- 10,000 nodes (realistic memory graph scale)
- 50,000 edges (5:1 edge-to-node ratio)
- Scale-free topology using preferential attachment (mimics real memory networks)

**Workload Composition:**
1. **Spreading Activation** (1000 queries)
   - Parallel spreading with depth-5 traversal
   - 4-thread execution
   - Exponential decay with rate 0.3

2. **Vector Similarity** (1000 queries)
   - 768-dimensional embeddings
   - Cosine similarity against 1000 candidates
   - Normalized vectors (unit length)

3. **Memory Decay** (10,000 calculations)
   - Exponential decay simulation
   - Time-based forgetting curves

**Benchmark Groups:**
- `complete_workload`: Full integrated workload
- `graph_creation`: Graph construction overhead
- `spreading_activation`: Isolated spreading queries
- `vector_similarity`: Isolated similarity computations
- `memory_decay`: Isolated decay calculations

### 2. Baseline Performance Benchmarks (`/engram-core/benches/baseline_performance.rs`)

Micro-benchmarks establishing performance baselines for regression detection:

**Vector Similarity Baselines:**
- Single 768d cosine similarity: ~10-50 nanoseconds (expected)
- 100 candidates: ~1-5 microseconds (expected)
- 1000 candidates: ~10-50 microseconds (expected)
- 5000 candidates: ~50-250 microseconds (expected)

**Spreading Activation Baselines:**
- Small graph (100n, 500e): ~100-500 microseconds (expected)
- Medium graph (500n, 2500e): ~500-2000 microseconds (expected)
- Large graph (1000n, 5000e): ~1-5 milliseconds (expected)

**Decay Calculation Baselines:**
- Exponential decay (100 steps): ~1-10 microseconds (expected)
- Power-law decay (100 steps): ~5-20 microseconds (expected)
- Linear decay (100 steps): ~1-5 microseconds (expected)
- Batch decay (10k memories): ~10-100 microseconds (expected)

**Graph Traversal Baselines:**
- Get neighbors (1000 nodes): ~50-200 microseconds (expected)
- Get all nodes: ~5-20 microseconds (expected)

**Memory Allocation Baselines:**
- Allocate embedding (768 floats): ~50-200 nanoseconds (expected)
- Allocate Memory struct: ~100-500 nanoseconds (expected)

### 3. Profiling Script (`/scripts/profile_hotspots.sh`)

Automated profiling workflow:

**Capabilities:**
- Detects platform (macOS/Linux) and uses appropriate profiler
- Installs `cargo-flamegraph` if not present
- Runs profiling harness with release optimizations
- Generates SVG flamegraph in `tmp/flamegraph.svg`
- Logs output to `tmp/profiling_output.log`

**Usage:**
```bash
./scripts/profile_hotspots.sh
```

**Platform Support:**
- **macOS**: Uses DTrace (requires elevated privileges)
- **Linux**: Uses `perf` (requires `linux-tools-generic`)

## Expected Hotspot Distribution

Based on workload composition and algorithmic complexity:

### Primary Optimization Targets (>5% cumulative time)

1. **Vector Similarity Operations (15-25% expected)**
   - Cosine similarity dot product loops
   - Embedding normalization
   - Candidate iteration and comparison
   - **Optimization Priority:** HIGH (Zig SIMD kernels)

2. **Activation Spreading (20-30% expected)**
   - Graph traversal and neighbor lookups
   - Activation propagation and accumulation
   - Decay function application
   - Work-stealing synchronization overhead
   - **Optimization Priority:** HIGH (Zig lock-free algorithms)

3. **Memory Decay Calculations (10-15% expected)**
   - Exponential function evaluations
   - Time-delta calculations
   - Batch decay operations
   - **Optimization Priority:** MEDIUM (Zig batch processing)

### Secondary Targets (2-5% cumulative time)

4. **Graph Construction (5-10% expected)**
   - Node insertion
   - Edge creation and weight assignment
   - Degree distribution tracking
   - **Optimization Priority:** LOW (one-time cost)

5. **Memory Allocation (3-7% expected)**
   - Embedding allocation (768 floats)
   - ActivationRecord creation
   - DashMap entry management
   - **Optimization Priority:** MEDIUM (specialized allocators)

### Tertiary Targets (<2% cumulative time)

6. **Synchronization Overhead (1-3% expected)**
   - Atomic operations
   - Lock contention (DashMap)
   - Phase barrier coordination
   - **Optimization Priority:** LOW (inherent to concurrency)

## Profiling Methodology

### Criterion Configuration

All benchmarks use the following Criterion settings for reliable measurements:

```rust
Criterion::default()
    .confidence_level(0.95)  // 95% confidence intervals
    .noise_threshold(0.02)   // 2% noise threshold for stable profiling
    .sample_size(10-100)     // Varies by benchmark complexity
    .warm_up_time(2-3s)      // Ensure caches are warm
    .measurement_time(10-30s) // Long measurement for stable samples
```

### Statistical Rigor

- **Variance Target:** <5% across 10 consecutive runs
- **Confidence Intervals:** 95% confidence level
- **Outlier Detection:** Criterion's automatic outlier filtering
- **Reproducibility:** Deterministic RNG seeds for graph generation

### System Configuration

Document the following when running benchmarks:

```bash
# CPU Information
sysctl -n machdep.cpu.brand_string  # macOS
lscpu                                # Linux

# Memory Information
sysctl hw.memsize                    # macOS
free -h                              # Linux

# Disable CPU frequency scaling for stable benchmarks
sudo cpupower frequency-set --governor performance  # Linux
```

## Benchmark Execution

### Running Baseline Benchmarks

```bash
# Run all baseline benchmarks
cargo bench --bench baseline_performance

# Run specific benchmark group
cargo bench --bench baseline_performance -- vector_similarity

# View HTML reports
open tmp/baseline_benchmarks/report/index.html
```

### Running Profiling Harness

```bash
# Generate flamegraph
./scripts/profile_hotspots.sh

# View flamegraph
open tmp/flamegraph.svg

# Run without profiling (for Criterion reports)
cargo bench --bench profiling_harness
```

## Interpreting Results

### Flamegraph Analysis

1. **Width = Time**: Wider bars represent functions consuming more CPU time
2. **Height = Call Stack**: Stack depth shows function call hierarchy
3. **Color Coding**: Colors differentiate between functions (no semantic meaning)

**Look for:**
- Functions occupying >5% horizontal width
- Unexpectedly wide bars indicating inefficiency
- Deep call stacks suggesting inlining opportunities

### Criterion Reports

Criterion generates HTML reports with:
- **Violin plots**: Distribution of sample times
- **Mean + confidence intervals**: Point estimates with statistical bounds
- **Throughput metrics**: Operations per second
- **Comparison with previous runs**: Regression detection

**Key Metrics:**
- **Mean time**: Central tendency of measurements
- **Std deviation**: Measurement variability (target: <5%)
- **Outliers**: Samples rejected as statistical outliers
- **R² value**: Goodness-of-fit for linear regression (target: >0.95)

## Next Steps

### Phase 1: Baseline Establishment (This Task)
- [x] Create profiling infrastructure
- [x] Implement comprehensive benchmarks
- [x] Document expected performance characteristics
- [ ] **Execute profiling and document actual hotspots**
- [ ] **Establish baseline metrics for regression detection**

### Phase 2: Hotspot Identification (Task 002)
- Analyze flamegraphs to identify top 10 functions by cumulative time
- Cross-reference with Criterion micro-benchmarks
- Validate that hotspots match expected distribution (15-25% similarity, 20-30% spreading, 10-15% decay)
- Prioritize optimization targets based on impact and implementation complexity

### Phase 3: Zig Kernel Implementation (Tasks 003-007)
- Implement Zig kernels for identified hotspots
- Establish differential testing harnesses
- Validate bit-identical outputs between Rust and Zig
- Measure >2x performance improvement threshold

### Phase 4: Integration and Regression Detection (Task 010)
- Integrate Criterion baselines into CI/CD pipeline
- Establish performance regression thresholds
- Automate benchmark comparison across commits
- Alert on >10% performance degradation

## Differential Testing Strategy

For each Zig kernel rewrite:

1. **Generate Test Corpus:**
   - 1M diverse inputs covering edge cases
   - Random seeds for reproducibility
   - Boundary conditions (zeros, infinities, NaNs)

2. **Bit-Identical Validation:**
   - Compare Rust and Zig outputs with exact equality
   - Allow configurable FP tolerance (default: 1e-6)
   - Detect divergence and minimize failing test cases

3. **Performance Verification:**
   - Require >2x speedup to justify complexity
   - Measure across different input sizes
   - Profile Zig implementation for further optimization

## Acceptance Criteria

✅ **Infrastructure Complete:**
- [x] Profiling harness creates 10k nodes, 50k edges
- [x] Workload exercises 1000 spreading queries
- [x] Workload exercises 1000 similarity comparisons
- [x] Workload exercises 10k decay calculations
- [x] Baseline benchmarks cover all identified hot paths
- [x] Profiling script generates flamegraph
- [x] Documentation explains methodology and expected results

⏳ **Pending Execution:**
- [ ] Execute `./scripts/profile_hotspots.sh` to generate actual flamegraph
- [ ] Execute `cargo bench --bench baseline_performance` to establish baselines
- [ ] Verify hotspot distribution matches expectations (±5%)
- [ ] Verify benchmark variance <5% across 10 consecutive runs
- [ ] Verify total benchmark runtime <5 minutes

## References

### Profiling Best Practices
- Brendan Gregg's "The Flame Graph" (http://www.brendangregg.com/flamegraphs.html)
- Chandler Carruth's "Tuning C++: Benchmarks, and CPUs, and Compilers! Oh My!" (CppCon 2015)

### Statistical Methodology
- Andy Georges et al., "Statistically Rigorous Java Performance Evaluation" (OOPSLA 2007)
- Tomas Kalibera and Richard Jones, "Rigorous Benchmarking in Reasonable Time" (ISMM 2013)

### Performance Optimization
- Ulrich Drepper, "What Every Programmer Should Know About Memory" (2007)
- Agner Fog's optimization manuals (https://www.agner.org/optimize/)

## System Configuration (To Be Documented)

**CPU:** _To be filled after first run_
**Memory:** _To be filled after first run_
**OS:** macOS Darwin 23.6.0
**Rust Version:** _To be filled after first run_
**Optimization Level:** Release (LTO enabled)
**CPU Frequency Scaling:** _To be documented_
**Background Processes:** _To be documented_
