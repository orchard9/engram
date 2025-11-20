# Profiling Infrastructure Issues and Fixes
**Task 001 Review - Implementation Recommendations**

## Issues Requiring Immediate Attention

### Issue #1: Missing Variance Validation (HIGH PRIORITY)
**File:** N/A (new script needed)
**Impact:** Cannot verify "variance <5% across 10 runs" acceptance criteria

**Fix:** Create variance validation script

**File:** `/Users/jordanwashburn/Workspace/orchard9/engram/scripts/validate_benchmark_variance.sh`

```bash
#!/usr/bin/env bash
#
# Validate that benchmark variance is <5% across multiple runs
#
# Usage: ./scripts/validate_benchmark_variance.sh <benchmark> <metric>
# Example: ./scripts/validate_benchmark_variance.sh baseline_performance "vector_similarity"

set -euo pipefail

RUNS=${RUNS:-10}
BENCHMARK=${1:-"baseline_performance"}
METRIC_PATTERN=${2:-"vector_similarity"}
VARIANCE_THRESHOLD=5.0  # 5% coefficient of variation

echo "====================================="
echo "  Benchmark Variance Validation"
echo "====================================="
echo "Benchmark: $BENCHMARK"
echo "Metric: $METRIC_PATTERN"
echo "Runs: $RUNS"
echo "Threshold: ${VARIANCE_THRESHOLD}%"
echo

# Create tmp directory
mkdir -p tmp/variance_validation

echo "Running benchmark $RUNS times..."
for i in $(seq 1 $RUNS); do
    echo -n "  Run $i/$RUNS..."
    cargo bench --bench "$BENCHMARK" -- "$METRIC_PATTERN" \
        --noplot \
        --save-baseline "variance_run_$i" 2>&1 \
        > "tmp/variance_validation/run_${i}.log"
    echo " done"
done

echo
echo "Analyzing results..."

# Extract mean times from each run
python3 << 'EOF'
import re
import statistics
import sys
import glob

# Parse all run logs
run_times = {}
for log_file in sorted(glob.glob('tmp/variance_validation/run_*.log')):
    with open(log_file) as f:
        content = f.read()

    # Find all "time: [x.xxx unit]" lines
    for match in re.finditer(r'time:\s+\[([0-9.]+)\s+([µnm]s)\s+([0-9.]+)\s+([µnm]s)\s+([0-9.]+)\s+([µnm]s)\]', content):
        # Extract middle value (median)
        median_value = float(match.group(3))
        unit = match.group(4)

        # Convert to nanoseconds for consistency
        if unit == 'µs':
            median_value *= 1000
        elif unit == 'ms':
            median_value *= 1_000_000

        # Find the benchmark name (appears before "time:")
        bench_name_match = re.search(r'(\S+)\s+time:', content[max(0, match.start()-200):match.start()])
        if bench_name_match:
            bench_name = bench_name_match.group(1)
            if bench_name not in run_times:
                run_times[bench_name] = []
            run_times[bench_name].append(median_value)

# Analyze variance for each benchmark
print("\nVariance Analysis:")
print("=" * 80)

all_pass = True
for bench_name, times in sorted(run_times.items()):
    if len(times) < 2:
        continue

    mean = statistics.mean(times)
    stdev = statistics.stdev(times)
    cv = (stdev / mean) * 100  # Coefficient of variation (%)

    # Format time nicely
    if mean < 1000:
        mean_str = f"{mean:.2f} ns"
    elif mean < 1_000_000:
        mean_str = f"{mean/1000:.2f} µs"
    else:
        mean_str = f"{mean/1_000_000:.2f} ms"

    status = "PASS" if cv < 5.0 else "FAIL"
    if cv >= 5.0:
        all_pass = False

    print(f"{bench_name:50s}")
    print(f"  Mean: {mean_str:>12s}  | Std Dev: {stdev/mean*100:5.2f}%  | CV: {cv:5.2f}%  | {status}")

print("=" * 80)

if all_pass:
    print("\n✓ All benchmarks have variance < 5%")
    sys.exit(0)
else:
    print("\n✗ Some benchmarks exceed 5% variance threshold")
    sys.exit(1)
EOF
```

**Test Command:**
```bash
chmod +x scripts/validate_benchmark_variance.sh
./scripts/validate_benchmark_variance.sh baseline_performance "vector_similarity"
```

---

### Issue #2: Missing Hotspot Percentage Validation (HIGH PRIORITY)
**File:** docs/internal/profiling_results.md
**Impact:** Cannot verify expected hotspot distribution

**Fix:** Add validation to profiling script

**File:** `/Users/jordanwashburn/Workspace/orchard9/engram/scripts/validate_hotspots.sh`

```bash
#!/usr/bin/env bash
#
# Validate that profiling results match expected hotspot distribution
#
# Expected ranges:
#   - Vector similarity: 15-25% of compute time
#   - Activation spreading: 20-30% of compute time
#   - Memory decay: 10-15% of compute time

set -euo pipefail

FLAMEGRAPH_FILE="${1:-tmp/flamegraph.svg}"

if [[ ! -f "$FLAMEGRAPH_FILE" ]]; then
    echo "Error: Flamegraph not found at $FLAMEGRAPH_FILE"
    echo "Run ./scripts/profile_hotspots.sh first"
    exit 1
fi

echo "====================================="
echo "  Hotspot Validation"
echo "====================================="
echo "Analyzing: $FLAMEGRAPH_FILE"
echo

# Parse flamegraph SVG to extract function percentages
# Note: This is a simplified parser - real implementation would need
# to properly parse SVG structure and calculate cumulative percentages

echo "WARNING: Hotspot validation not yet implemented"
echo "TODO: Parse flamegraph SVG and extract top function percentages"
echo
echo "Manual validation steps:"
echo "1. Open $FLAMEGRAPH_FILE in a browser"
echo "2. Identify the widest bars (most CPU time)"
echo "3. Verify percentages match expected ranges:"
echo "   - Vector similarity (cosine_similarity): 15-25%"
echo "   - Activation spreading (spread_activation, run_spreading): 20-30%"
echo "   - Memory decay (decay, exp): 10-15%"
echo
echo "For automated validation, implement SVG parsing or use perf script output"

# Future implementation:
# - Parse perf.data with perf script
# - Calculate cumulative percentages for each function
# - Validate against expected ranges
# - Exit 1 if outside expected ranges
```

**Long-term Fix:** Use `perf script` output instead of flamegraph SVG:
```bash
cargo flamegraph --bench profiling_harness --output tmp/flamegraph.svg -- --bench
perf script > tmp/perf_output.txt
# Parse perf_output.txt to calculate function percentages
```

---

### Issue #3: Degree Tracking Ambiguity (MEDIUM PRIORITY)
**File:** `/Users/jordanwashburn/Workspace/orchard9/engram/engram-core/benches/profiling_harness.rs`
**Lines:** 28-76
**Impact:** Unclear whether preferential attachment uses in-degree, out-degree, or total degree

**Fix:** Add clarifying comments

```rust
// Line 28 - Add comprehensive explanation
    // Add edges using preferential attachment to create realistic degree distribution.
    // This creates a scale-free graph similar to real memory networks.
    //
    // IMPORTANT: This implementation uses *total degree* (in-degree + out-degree)
    // for preferential attachment, not just out-degree. This means:
    //   - Nodes that are popular targets (high in-degree) are more likely to be chosen as sources
    //   - Nodes that are active sources (high out-degree) are more likely to be chosen again
    //   - This creates bidirectional hub nodes, which is realistic for memory consolidation
    //
    // Alternative approaches:
    //   - Out-degree only: Models information spreading (nodes that send more get more connections)
    //   - In-degree only: Models information receiving (popular nodes get more incoming links)
    //   - Total degree: Models overall connectivity (used here)
    //
    // For memory graphs, total degree is appropriate because:
    //   1. Frequently accessed memories (high in-degree) are more likely to trigger associations
    //   2. Memories with many associations (high out-degree) are more likely to be reused
    //   3. Real hippocampal-neocortical consolidation exhibits both patterns
    let mut node_degrees: Vec<usize> = vec![0; node_count];
```

And update the degree tracking comment:

```rust
// Line 74 - Clarify that this tracks total degree
        // Update total degree for both source and target
        // This creates bidirectional hubs in the network
        node_degrees[source_idx] = node_degrees[source_idx].saturating_add(1);
        node_degrees[target_idx] = node_degrees[target_idx].saturating_add(1);
```

---

### Issue #4: Missing Precondition Documentation (MEDIUM PRIORITY)
**File:** `/Users/jordanwashburn/Workspace/orchard9/engram/engram-core/benches/baseline_performance.rs`
**Line:** 39-43
**Impact:** cosine_similarity assumes normalized vectors but doesn't document this

**Fix:** Add documentation and debug assertions

```rust
/// Compute cosine similarity between two normalized 768-dimensional embeddings.
///
/// # Preconditions
/// - Both input vectors MUST have unit L2 norm (magnitude = 1.0)
/// - For normalized vectors, cosine similarity equals dot product
///
/// # Arguments
/// - `a`: First normalized embedding
/// - `b`: Second normalized embedding
///
/// # Returns
/// Cosine similarity in range [-1.0, 1.0]
///
/// # Panics
/// In debug builds, panics if vectors are not normalized (tolerance: 1%)
#[inline]
fn cosine_similarity(a: &[f32; 768], b: &[f32; 768]) -> f32 {
    #[cfg(debug_assertions)]
    {
        let mag_a = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let mag_b = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        debug_assert!(
            (mag_a - 1.0).abs() < 0.01,
            "Vector a not normalized: magnitude = {mag_a}"
        );
        debug_assert!(
            (mag_b - 1.0).abs() < 0.01,
            "Vector b not normalized: magnitude = {mag_b}"
        );
    }

    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}
```

---

### Issue #5: Warm-up Time Too Short (MEDIUM PRIORITY)
**File:** `/Users/jordanwashburn/Workspace/orchard9/engram/engram-core/benches/profiling_harness.rs`
**Line:** 152
**Impact:** Insufficient warm-up for complex workload (graph creation, thread pool, caches)

**Fix:** Increase warm-up time

```rust
// Line 152-153
    group.warm_up_time(Duration::from_secs(5));  // Increased from 2s to 5s
    group.measurement_time(Duration::from_secs(30));
```

**Rationale:**
- Complete workload takes 30+ seconds to measure
- 2s warm-up is only 6% of measurement time (should be 10-20%)
- Graph construction (10k nodes, 50k edges) needs cache warming
- Thread pool (4 threads) needs initialization
- 5s warm-up provides better statistical stability

---

### Issue #6: Magic Numbers Not Extracted (LOW PRIORITY)
**File:** `/Users/jordanwashburn/Workspace/orchard9/engram/engram-core/benches/profiling_harness.rs`
**Lines:** Throughout
**Impact:** Maintainability - hard to find and change constants

**Fix:** Extract to named constants

```rust
// Add at top of file (after imports)
/// Number of nodes in the profiling graph
const PROFILING_GRAPH_NODES: usize = 10_000;

/// Number of edges in the profiling graph (5:1 edge-to-node ratio)
const PROFILING_GRAPH_EDGES: usize = 50_000;

/// Number of spreading activation queries in profiling workload
const PROFILING_SPREADING_QUERIES: usize = 1_000;

/// Number of vector similarity queries in profiling workload
const PROFILING_SIMILARITY_QUERIES: usize = 1_000;

/// Random seed for graph generation (deterministic)
const SEED_GRAPH: u64 = 0xDEAD_BEEF;

/// Random seed for spreading activation queries
const SEED_SPREADING: u64 = 0x0005_EED1;

/// Random seed for similarity queries
const SEED_SIMILARITY: u64 = 0x0005_EED2;

/// Random seed for decay simulation
const SEED_DECAY: u64 = 0x0005_EED3;
```

Then update usages:

```rust
// Line 22-23
    let node_count = PROFILING_GRAPH_NODES;
    let edge_count = PROFILING_GRAPH_EDGES;

// Line 99
    for _ in 0..PROFILING_SPREADING_QUERIES {

// Line 122
    for _ in 0..PROFILING_SIMILARITY_QUERIES {

// Line 156
    let graph = create_realistic_graph(SEED_GRAPH);

// Line 167
    run_spreading_queries(&graph, &config, SEED_SPREADING);

// Line 170
    run_similarity_queries(SEED_SIMILARITY);
```

---

## Issues Not Requiring Immediate Fixes

### Issue #7: Decay Formula Semantics (DOCUMENTATION ONLY)
**File:** `/Users/jordanwashburn/Workspace/orchard9/engram/engram-core/benches/baseline_performance.rs`
**Line:** 224-237
**Impact:** Unclear whether decay is per-step or cumulative

**Fix:** Add comment explaining the semantics

```rust
// Line 218-220
    for (decay_fn, name) in decay_functions {
        group.bench_with_input(
            BenchmarkId::new("decay_type", name),
            &decay_fn,
            |b, decay_fn| {
                b.iter(|| {
                    let initial_activation = 1.0f32;
                    let mut activation = initial_activation;

                    // Simulate cumulative decay over 100 time steps
                    // Note: This computes total decay from t=0 to t=depth,
                    // not incremental per-step decay. For exponential decay:
                    //   - Cumulative: activation * exp(-rate * depth)
                    //   - Per-step: activation * exp(-rate) applied depth times
                    // These are mathematically equivalent when rate is the same.
                    for depth in 0..100 {
```

---

## Summary of Recommended Actions

### Immediate (Before Using for Regression Detection):
1. ✅ Create variance validation script
2. ⚠️ Create hotspot validation script (or document manual process)
3. ✅ Add degree tracking clarification comments
4. ✅ Add cosine_similarity precondition docs + debug_assert

### Short-term (Before Task 010 - Regression Framework):
5. ✅ Increase warm-up time to 5s
6. ✅ Extract magic numbers to constants
7. ✅ Add decay semantics comment

### Long-term (Future Improvements):
8. Implement SVG/perf parsing for automated hotspot validation
9. Add property-based tests for spreading activation
10. Add hardware counter profiling (cache misses, branch mispredictions)

---

## Implementation Priority

**P0 - Critical (Block further work):**
- None

**P1 - High (Fix before depending on results):**
- Issue #1: Variance validation script
- Issue #2: Hotspot validation (at minimum, document manual process)
- Issue #4: cosine_similarity preconditions

**P2 - Medium (Fix before production):**
- Issue #3: Degree tracking clarification
- Issue #5: Warm-up time increase
- Issue #6: Extract magic numbers

**P3 - Low (Nice to have):**
- Issue #7: Decay semantics documentation

---

## Verification Checklist

After implementing fixes:

- [ ] Run variance validation script on baseline_performance
- [ ] Verify variance <5% for all benchmarks
- [ ] Run profiling harness and generate flamegraph
- [ ] Manually validate hotspot percentages (or wait for automated script)
- [ ] Verify debug assertions trigger for non-normalized vectors
- [ ] Run `make quality` to check for new clippy warnings
- [ ] Update docs/internal/profiling_results.md with actual results
- [ ] Run diagnostics: `./scripts/engram_diagnostics.sh`
