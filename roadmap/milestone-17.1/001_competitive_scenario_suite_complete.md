# Task 001: Competitive Scenario Suite (TOML Definitions)

**Status**: Pending
**Complexity**: Moderate
**Dependencies**: None
**Estimated Effort**: 4 hours

## Objective

Create standardized TOML scenario files that replicate competitor benchmarks for apples-to-apples comparison. These scenarios must produce deterministic, reproducible results suitable for rigorous performance comparison.

## Specifications

Create the following scenario files in `scenarios/competitive/`:

1. **`qdrant_ann_1m_768d.toml`**: ANN search matching Qdrant's published benchmark
   - 1M nodes, 768-dimensional embeddings
   - 100% search operations (no store/recall)
   - Target: 99.5% recall rate
   - Seed: 42 (deterministic)
   - Duration: 60s
   - Batch size: 1 (single query latency)

2. **`neo4j_traversal_100k.toml`**: Graph traversal matching Neo4j benchmark
   - 100K nodes, average degree 10 (1M edges)
   - 80% single-hop traversal, 20% 2-hop traversal
   - Seed: 43 (deterministic)
   - Duration: 60s
   - Embedding dimension: 768 (even if not used in traversal)

3. **`hybrid_production_100k.toml`**: Engram's unique hybrid workload
   - 100K nodes, 768-dimensional embeddings
   - 30% store, 30% recall, 30% search, 10% pattern completion
   - Seed: 44 (deterministic)
   - Duration: 60s
   - Demonstrates competitive advantage (no direct competitor equivalent)

4. **`milvus_ann_10m_768d.toml`**: Large-scale ANN (stretch goal)
   - 10M nodes, 768-dimensional embeddings
   - 100% search operations
   - Seed: 45 (deterministic)
   - Duration: 60s
   - Target: 100% recall rate

## File Paths

```
scenarios/competitive/README.md
scenarios/competitive/qdrant_ann_1m_768d.toml
scenarios/competitive/neo4j_traversal_100k.toml
scenarios/competitive/hybrid_production_100k.toml
scenarios/competitive/milvus_ann_10m_768d.toml
scenarios/competitive/validation/determinism_test.sh
scenarios/competitive/validation/resource_bounds_test.sh
scenarios/competitive/validation/correctness_test.sh
```

## Enhanced Acceptance Criteria

### 1. TOML Parsing and Syntax Validation
- All TOML files parse correctly with existing `loadtest` tool
- Schema validation passes (all required fields present, correct types)
- Comments explain parameter choices and competitor equivalents
- README documents benchmark sources with citations (URLs, paper DOIs, or GitHub commit hashes)

### 2. Determinism Validation (Critical)
**Requirement**: Scenarios must produce bitwise-identical operation sequences and nearly-identical latency distributions across runs.

**Validation Method**: Differential testing with triple-run comparison
- Run each scenario 3 times consecutively with same seed
- Compare operation type sequences (must be 100% identical)
- Compare P99 latency (tolerance: ±0.5ms or ±1%, whichever is larger)
- Compare throughput (tolerance: ±2% to account for OS scheduling variance)
- Compare operation counts (must be exactly identical)

**Expected Invariants**:
- Same seed → Same RNG state → Same operation sequence
- Same operations → Same server behavior → Same latency distribution (within measurement noise)
- If any metric diverges beyond tolerance, scenario is non-deterministic and must be fixed

### 3. Scenario Correctness Validation
**Node Count Verification**:
- For 1M scenario: Memory footprint should be 768MB baseline + metadata (~1.5GB total on 16GB machine)
- For 10M scenario: Memory footprint should be 7.68GB baseline + metadata (~10GB total - requires 16GB+ RAM)
- Use `/usr/bin/time -l` (macOS) or `/usr/bin/time -v` (Linux) to measure max RSS
- Expected bounds: `num_nodes * embedding_dim * 4 bytes * 1.3` (30% overhead for indices/metadata)

**Operation Distribution Verification**:
- Parse loadtest JSON output to count operations by type
- For 100% search scenario: verify `store_count == 0 && recall_count == 0 && pattern_completion_count == 0`
- For mixed scenarios: verify operation ratios within ±3% of configured weights (statistical sampling variance)
- Total operations should equal `target_rate * duration_seconds * (1 - error_rate)`

**Embedding Dimension Validation**:
- All generated embeddings must be exactly 768 dimensions (verify via spot checks in JSON logs)
- Pattern completion operations must generate 384-dim partial inputs (embedding_dim/2)
- No truncation or padding should occur in operation generation

### 4. Cross-Platform Determinism
**Requirement**: Same seed produces same operation sequences on macOS and Linux.

**Validation Method**:
- Run on macOS (Darwin/aarch64 or x86_64)
- Run on Linux (x86_64)
- Compare operation sequence checksums (SHA256 hash of operation types in order)
- Platform-specific timing differences are acceptable, sequence differences are not

**Known Variance Sources**:
- Latency measurements will differ due to OS schedulers and hardware
- Throughput may vary by ±5% due to CPU architecture differences
- Operation sequences must be identical (RNG is platform-independent)

### 5. Failure Mode Testing
**Out-of-Memory Scenarios**:
- 10M scenario should gracefully fail with clear error message on 8GB machines
- Should not cause kernel OOM killer or system hang
- Error message should specify memory requirement: "Requires 16GB+ RAM for 10M nodes"

**Invalid Configuration Detection**:
- Typo in TOML field (e.g., `embeddig_dim` instead of `embedding_dim`) → clear parse error
- Operation weights sum to 0.0 → validation error before test starts
- Negative duration or rate → validation error before test starts

**Binary Build Verification**:
- If loadtest binary not built in release mode, print performance warning
- Debug builds run 5-10x slower → suggest `cargo build --release`

**Resource Exhaustion**:
- If system has <16GB RAM, warn before running 1M+ scenarios
- If system has <4GB available memory, abort 1M scenarios preemptively
- Check available memory using sysinfo crate before test execution

## Testing Approach

### Phase 1: Syntax and Schema Validation

```bash
# Validate all TOML files parse without errors
for scenario in scenarios/competitive/*.toml; do
  echo "Validating $scenario..."
  cargo run --release --bin loadtest -- run --scenario "$scenario" --duration 1 --dry-run || exit 1
done
```

**Expected Output**: Zero parsing errors, configuration values echoed correctly.

### Phase 2: Determinism Verification (Triple-Run Test)

```bash
#!/bin/bash
# scenarios/competitive/validation/determinism_test.sh

SCENARIO="$1"  # e.g., scenarios/competitive/qdrant_ann_1m_768d.toml
DURATION=10    # Short test for faster iteration

echo "=== Determinism Test: $SCENARIO ==="

# Run 3 times and capture detailed metrics
for i in {1..3}; do
  echo "Run $i/3..."
  cargo run --release --bin loadtest -- run \
    --scenario "$SCENARIO" \
    --duration "$DURATION" \
    --output "/tmp/determinism_run_${i}.json" \
    > "/tmp/determinism_run_${i}.txt" 2>&1
done

# Extract and compare key metrics
echo -e "\n=== P99 Latency Comparison ==="
for i in {1..3}; do
  p99=$(jq '.p99_latency_ms' "/tmp/determinism_run_${i}.json")
  echo "Run $i: ${p99}ms"
done

echo -e "\n=== Operation Count Comparison ==="
for i in {1..3}; do
  ops=$(jq '.total_operations' "/tmp/determinism_run_${i}.json")
  echo "Run $i: $ops operations"
done

echo -e "\n=== Operation Distribution Comparison ==="
for i in {1..3}; do
  echo "Run $i:"
  jq '.per_operation_stats | to_entries[] | "\(.key): \(.value.count)"' \
    "/tmp/determinism_run_${i}.json"
done

# Statistical comparison (requires Python/R for proper analysis)
echo -e "\n=== Statistical Validation ==="
python3 <<EOF
import json
import statistics

runs = [json.load(open(f'/tmp/determinism_run_{i}.json')) for i in range(1, 4)]

# Check operation counts are identical
op_counts = [r['total_operations'] for r in runs]
if len(set(op_counts)) != 1:
    print(f"FAIL: Operation counts differ: {op_counts}")
    exit(1)
else:
    print(f"PASS: Operation counts identical: {op_counts[0]}")

# Check P99 latency variance
p99_latencies = [r['p99_latency_ms'] for r in runs]
mean_p99 = statistics.mean(p99_latencies)
max_deviation = max(abs(p - mean_p99) for p in p99_latencies)
tolerance_abs = 0.5  # 0.5ms absolute tolerance
tolerance_rel = mean_p99 * 0.01  # 1% relative tolerance
tolerance = max(tolerance_abs, tolerance_rel)

if max_deviation <= tolerance:
    print(f"PASS: P99 latencies within tolerance ({max_deviation:.3f}ms <= {tolerance:.3f}ms)")
    print(f"  Values: {p99_latencies}")
else:
    print(f"FAIL: P99 latencies exceed tolerance ({max_deviation:.3f}ms > {tolerance:.3f}ms)")
    print(f"  Values: {p99_latencies}")
    exit(1)

print("\nDeterminism test PASSED")
EOF
```

**Expected Behavior**:
- Script exits 0 on success
- All operation counts exactly identical
- P99 latency variance <0.5ms or <1% (whichever larger)
- Clear PASS/FAIL output for each metric

### Phase 3: Resource Bounds Verification

```bash
#!/bin/bash
# scenarios/competitive/validation/resource_bounds_test.sh

SCENARIO="$1"
DURATION=30  # 30s test for memory stability

echo "=== Resource Bounds Test: $SCENARIO ==="

# Extract expected node count from TOML
NUM_NODES=$(grep "^num_nodes" "$SCENARIO" | awk -F'=' '{print $2}' | tr -d ' ')
EMBEDDING_DIM=$(grep "^embedding_dim" "$SCENARIO" | awk -F'=' '{print $2}' | tr -d ' ')

# Calculate expected memory footprint
# Formula: num_nodes * embedding_dim * 4 bytes/float * 1.3 overhead factor
EXPECTED_MB=$(python3 -c "print(int($NUM_NODES * $EMBEDDING_DIM * 4 * 1.3 / 1024 / 1024))")

echo "Expected memory footprint: ~${EXPECTED_MB}MB for ${NUM_NODES} nodes"

# Run with memory tracking
if [[ "$OSTYPE" == "darwin"* ]]; then
  # macOS
  /usr/bin/time -l cargo run --release --bin loadtest -- run \
    --scenario "$SCENARIO" \
    --duration "$DURATION" \
    2>&1 | tee /tmp/resource_test.txt

  # Extract max RSS (in bytes on macOS)
  MAX_RSS_BYTES=$(grep "maximum resident set size" /tmp/resource_test.txt | awk '{print $1}')
  MAX_RSS_MB=$((MAX_RSS_BYTES / 1024 / 1024))
else
  # Linux
  /usr/bin/time -v cargo run --release --bin loadtest -- run \
    --scenario "$SCENARIO" \
    --duration "$DURATION" \
    2>&1 | tee /tmp/resource_test.txt

  # Extract max RSS (in KB on Linux)
  MAX_RSS_KB=$(grep "Maximum resident set size" /tmp/resource_test.txt | awk '{print $6}')
  MAX_RSS_MB=$((MAX_RSS_KB / 1024))
fi

echo "Measured max RSS: ${MAX_RSS_MB}MB"

# Validate memory is within expected bounds (2x tolerance for overhead)
UPPER_BOUND=$((EXPECTED_MB * 2))
LOWER_BOUND=$((EXPECTED_MB / 2))

if [ "$MAX_RSS_MB" -lt "$LOWER_BOUND" ]; then
  echo "FAIL: Memory usage too low (${MAX_RSS_MB}MB < ${LOWER_BOUND}MB)"
  echo "  This may indicate nodes are not being created properly"
  exit 1
elif [ "$MAX_RSS_MB" -gt "$UPPER_BOUND" ]; then
  echo "FAIL: Memory usage too high (${MAX_RSS_MB}MB > ${UPPER_BOUND}MB)"
  echo "  This may indicate memory leaks or inefficient storage"
  exit 1
else
  echo "PASS: Memory usage within expected bounds (${LOWER_BOUND}MB - ${UPPER_BOUND}MB)"
fi
```

**Expected Behavior**:
- 1M scenario: 750MB - 3GB RSS (reasonable overhead for graph structures)
- 10M scenario: 5GB - 20GB RSS (should fail gracefully on 8GB machines)
- Memory stable throughout test (no runaway growth)

### Phase 4: Operation Distribution Correctness

```bash
#!/bin/bash
# scenarios/competitive/validation/correctness_test.sh

SCENARIO="$1"
DURATION=60

echo "=== Correctness Test: $SCENARIO ==="

# Run scenario and capture detailed output
cargo run --release --bin loadtest -- run \
  --scenario "$SCENARIO" \
  --duration "$DURATION" \
  --output /tmp/correctness_test.json

# Validate operation distribution
python3 <<EOF
import json
import sys

with open('/tmp/correctness_test.json') as f:
    report = json.load(f)

# Load scenario config to get expected weights
with open('$SCENARIO') as f:
    import toml
    config = toml.load(f)

ops = config['operations']
total_ops = report['total_operations']

print("=== Operation Distribution Validation ===")
print(f"Total operations: {total_ops}")

# Extract actual counts
actual_counts = {}
for op_type, stats in report['per_operation_stats'].items():
    actual_counts[op_type] = stats['count']

# Map config weights to operation types
weight_map = {
    'Store': ops.get('store_weight', 0.0),
    'Recall': ops.get('recall_weight', 0.0),
    'Search': ops.get('embedding_search_weight', 0.0),
    'PatternCompletion': ops.get('pattern_completion_weight', 0.0),
}

total_weight = sum(weight_map.values())
expected_ratios = {k: v/total_weight for k, v in weight_map.items()}

print("\nExpected vs Actual Ratios:")
all_pass = True
for op_type, expected_ratio in expected_ratios.items():
    if expected_ratio == 0.0:
        # Should have zero operations
        actual_count = actual_counts.get(op_type, 0)
        if actual_count != 0:
            print(f"  FAIL: {op_type} should be 0, got {actual_count}")
            all_pass = False
        else:
            print(f"  PASS: {op_type} = 0 (as expected)")
    else:
        actual_count = actual_counts.get(op_type, 0)
        actual_ratio = actual_count / total_ops if total_ops > 0 else 0.0
        deviation = abs(actual_ratio - expected_ratio)
        tolerance = 0.03  # 3% tolerance for statistical variance

        if deviation <= tolerance:
            print(f"  PASS: {op_type} = {actual_ratio:.3f} (expected {expected_ratio:.3f}, deviation {deviation:.3f})")
        else:
            print(f"  FAIL: {op_type} = {actual_ratio:.3f} (expected {expected_ratio:.3f}, deviation {deviation:.3f} > {tolerance})")
            all_pass = False

if not all_pass:
    print("\nOperation distribution validation FAILED")
    sys.exit(1)
else:
    print("\nOperation distribution validation PASSED")
EOF
```

**Expected Behavior**:
- 100% search scenarios: zero Store/Recall/PatternCompletion operations
- Mixed scenarios: operation ratios within ±3% of configured weights
- Total operations close to `rate * duration` (accounting for error rate)

### Phase 5: Integration Test (End-to-End)

```bash
# Run all scenarios sequentially to verify no cross-contamination
for scenario in scenarios/competitive/*.toml; do
  echo "=== Testing $scenario ==="

  # Determinism test
  ./scenarios/competitive/validation/determinism_test.sh "$scenario" || exit 1

  # Resource bounds test
  ./scenarios/competitive/validation/resource_bounds_test.sh "$scenario" || exit 1

  # Correctness test
  ./scenarios/competitive/validation/correctness_test.sh "$scenario" || exit 1

  echo "✓ $scenario passed all validations"
  echo ""
done

echo "All competitive scenarios validated successfully"
```

**Expected Behavior**:
- All scenarios pass determinism, resource bounds, and correctness tests
- No memory leaks between scenarios
- Clean error messages for any failures

## Integration Points

- Extends existing `scenarios/` directory structure
- Uses existing `loadtest` TOML parser (no code changes required)
- Scenario files referenced in Task 002 baseline documentation
- Validation scripts used in CI/CD pipeline for regression detection

## Performance Expectations

Based on existing `m17_baseline.toml` performance:

| Scenario | Expected P99 Latency | Expected Throughput | Memory Footprint |
|----------|---------------------|---------------------|------------------|
| qdrant_ann_1m_768d | <100ms | >800 ops/sec | ~1.5GB |
| neo4j_traversal_100k | <50ms | >1000 ops/sec | ~200MB |
| hybrid_production_100k | <75ms | >900 ops/sec | ~250MB |
| milvus_ann_10m_768d | <150ms | >600 ops/sec | ~10GB |

These are initial estimates based on similar workloads. Actual numbers will establish baselines for competitive comparison.

## Known Limitations

1. **Determinism tolerance**: OS scheduling introduces ±1-2% latency variance even with same seed
2. **Platform differences**: CPU architecture affects throughput but not operation sequence
3. **Memory measurement**: RSS includes OS buffers and may vary by ±10% run-to-run
4. **Statistical variance**: Operation distribution has ±3% variance due to finite sample size

## Success Criteria Summary

Task is complete when:

1. All 4 TOML files parse correctly and run to completion
2. README.md documents competitor benchmark sources with citations
3. All scenarios pass triple-run determinism test (<1% P99 variance)
4. Memory footprints match expected bounds (0.5x - 2x theoretical minimum)
5. Operation distributions match configured weights (±3% tolerance)
6. Validation scripts execute cleanly with clear PASS/FAIL output
7. Resource exhaustion scenarios fail gracefully with helpful error messages

## References

- Qdrant benchmark: https://qdrant.tech/benchmarks/
- Neo4j performance: https://neo4j.com/developer/graph-data-science/performance/
- Milvus benchmarks: https://milvus.io/docs/benchmark.md
- HdrHistogram for latency measurement: https://github.com/HdrHistogram/HdrHistogram_rust
