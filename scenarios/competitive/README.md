# Competitive Benchmark Scenarios

This directory contains standardized load test scenarios designed for rigorous competitive comparison against leading vector and graph databases. Each scenario replicates published competitor benchmarks to enable apples-to-apples performance validation.

## Scenarios

### 1. Qdrant ANN Search (`qdrant_ann_1m_768d.toml`)

**Competitor**: Qdrant Vector Database
**Reference**: https://qdrant.tech/benchmarks/
**Dataset**: 1M vectors, 768 dimensions (OpenAI ada-002 embedding size)
**Workload**: 100% ANN search operations
**Competitor Baseline**: P99 22-24ms, 626 QPS @ 99.5% recall
**Engram Target**: P99 <20ms, >800 QPS

Validates Engram's vector search performance against specialized vector databases. Tests pure HNSW index performance without graph operations.

**Key Metrics**:
- Search latency distribution (P50, P99, P99.9)
- Throughput (queries per second)
- Recall accuracy (target 99.5%)
- Memory footprint (~1.5GB expected)

### 2. Neo4j Graph Traversal (`neo4j_traversal_100k.toml`)

**Competitor**: Neo4j Graph Database
**Reference**: https://neo4j.com/developer/graph-data-science/performance/
**Dataset**: 100K nodes, average degree 10 (1M edges)
**Workload**: 80% single-hop traversal, 20% 2-hop traversal
**Competitor Baseline**: P99 27.96ms, 280 QPS
**Engram Target**: P99 <15ms, >400 QPS

Validates Engram's graph traversal via activation spreading against specialized graph databases. Tests multi-hop recall without heavy vector operations.

**Key Metrics**:
- Traversal latency (single-hop vs multi-hop)
- Activation spreading efficiency
- Edge traversal throughput
- Memory footprint (~200MB expected)

### 3. Hybrid Production Workload (`hybrid_production_100k.toml`)

**Competitor**: None (Engram unique capability)
**Reference**: N/A - demonstrates differentiation
**Dataset**: 100K nodes, 768 dimensions
**Workload**: 30% store, 30% recall, 30% search, 10% completion
**Competitor Baseline**: N/A
**Engram Target**: P99 <10ms mixed operations

Demonstrates Engram's unique value proposition: unified vector+graph+temporal memory in single system. No competitor offers this hybrid capability without multiple specialized databases.

**Key Metrics**:
- Mixed workload latency
- Operation type breakdown
- Memory efficiency under hybrid load
- Validates no performance sacrifice from unification

### 4. Milvus Large-Scale ANN (`milvus_ann_10m_768d.toml`)

**Competitor**: Milvus Vector Database
**Reference**: https://milvus.io/docs/benchmark.md
**Dataset**: 10M vectors, 768 dimensions (10x scale)
**Workload**: 100% ANN search operations
**Competitor Baseline**: P99 708ms, 2,098 QPS @ 100% recall
**Engram Target**: P99 <100ms, >400 QPS

**STRETCH GOAL**: Validates Engram's scalability to production-scale datasets. Requires 16GB+ RAM and may require GPU acceleration.

**Key Metrics**:
- Large-scale search latency
- Memory usage at 10M scale (~10-15GB expected)
- Index build time
- GPU utilization (if enabled)

## Usage

### Running Individual Scenarios

```bash
cargo build --release --bin loadtest

# Run Qdrant benchmark
cargo run --release --bin loadtest -- run \
  --scenario scenarios/competitive/qdrant_ann_1m_768d.toml \
  --duration 60 \
  --seed 42 \
  --output tmp/qdrant_results.json

# Run Neo4j benchmark
cargo run --release --bin loadtest -- run \
  --scenario scenarios/competitive/neo4j_traversal_100k.toml \
  --duration 60 \
  --seed 43 \
  --output tmp/neo4j_results.json
```

### Running Full Competitive Suite

```bash
# Production-hardened orchestration script with pre-flight checks
./scripts/competitive_benchmark_suite.sh

# Results saved to: tmp/m17_performance/competitive_baseline/
```

### Validation Testing

```bash
# Test determinism (3 consecutive runs must be identical)
./scenarios/competitive/validation/determinism_test.sh \
  scenarios/competitive/qdrant_ann_1m_768d.toml

# Test memory footprint
./scenarios/competitive/validation/resource_bounds_test.sh \
  scenarios/competitive/qdrant_ann_1m_768d.toml

# Test operation distribution correctness
./scenarios/competitive/validation/correctness_test.sh \
  scenarios/competitive/hybrid_production_100k.toml
```

## Determinism Requirements

All scenarios use deterministic seeding to ensure reproducible results:

- **Qdrant scenario**: Seed 42
- **Neo4j scenario**: Seed 43
- **Hybrid scenario**: Seed 44
- **Milvus scenario**: Seed 45

**Validation Criteria**:
- Same seed → Same operation sequence (100% identical)
- P99 latency variance: <0.5ms or <1%, whichever larger (across 3 runs)
- Throughput variance: ±2% (accounts for OS scheduling)
- Cross-platform: macOS and Linux produce identical operation sequences

## Memory Requirements

| Scenario | Expected RSS | Minimum RAM | Recommended RAM |
|----------|--------------|-------------|-----------------|
| Qdrant 1M | ~1.5GB | 4GB | 8GB |
| Neo4j 100K | ~200MB | 2GB | 4GB |
| Hybrid 100K | ~250MB | 2GB | 4GB |
| Milvus 10M | ~10-15GB | 16GB | 32GB |

**Formula**: `num_nodes * embedding_dim * 4 bytes * 1.3 overhead`

The 1.3 overhead factor accounts for HNSW indices, metadata, and internal data structures.

## Performance Targets

Competitive comparison targets derived from published benchmarks:

| Workload | Competitor | Their P99 | Engram Target | Delta | Status |
|----------|------------|-----------|---------------|-------|--------|
| ANN 1M | Qdrant | 22-24ms | <20ms | -10% | To measure |
| Traversal 100K | Neo4j | 27.96ms | <15ms | -46% | To measure |
| Hybrid 100K | None | N/A | <10ms | Unique | To measure |
| ANN 10M | Milvus | 708ms | <100ms | -86% | Stretch goal |

## Quarterly Review Process

These scenarios are executed quarterly (Jan, Apr, Jul, Oct) to track competitive positioning over time:

```bash
# One-command quarterly workflow
./scripts/quarterly_competitive_review.sh

# Generates:
# - tmp/m17_performance/competitive_q1_2025/report.md
# - Historical trend analysis
# - Optimization recommendations
```

## Integration with M17 Regression Framework

Competitive scenarios integrate with existing M17 performance tracking:

```bash
# Run M17 task with competitive validation
./scripts/m17_performance_check.sh 001 before --competitive

# Compare with competitive thresholds
./scripts/compare_m17_performance.sh 001 --competitive
```

**Exit Codes**:
- 0: Success (no regressions)
- 1: Internal regression (M17 5% threshold exceeded)
- 2: Competitive regression (10% threshold exceeded)
- 3: Fatal error

## Validation Scripts

### `determinism_test.sh`

Validates reproducibility via triple-run comparison:
- Runs scenario 3 times with same seed
- Compares operation counts (must be exactly identical)
- Compares P99 latency (tolerance: ±0.5ms or ±1%)
- Compares throughput (tolerance: ±2%)
- Statistical validation via Python script

**Usage**: `./determinism_test.sh scenarios/competitive/qdrant_ann_1m_768d.toml`

### `resource_bounds_test.sh`

Validates memory footprint predictions:
- Measures max RSS using `/usr/bin/time`
- Compares against theoretical bounds (formula-based)
- Validates 0.5x - 2x expected range
- Detects memory leaks (runaway growth)

**Usage**: `./resource_bounds_test.sh scenarios/competitive/neo4j_traversal_100k.toml`

### `correctness_test.sh`

Validates operation distribution accuracy:
- Parses loadtest JSON output
- Counts operations by type
- Validates ratios match TOML weights (±3% tolerance)
- Confirms zero-weight operations have zero count

**Usage**: `./correctness_test.sh scenarios/competitive/hybrid_production_100k.toml`

## Troubleshooting

### Out of Memory (10M scenario)

If the Milvus 10M scenario fails with OOM:

```bash
# Check available memory
free -h  # Linux
vm_stat  # macOS

# Expected requirement: 16GB+ total RAM, 12GB+ available
# If insufficient, skip 10M scenario or enable memory-mapped storage
```

### Non-Deterministic Results

If determinism tests fail (P99 variance >1%):

1. Check CPU throttling: `sudo powermetrics --samplers smc` (macOS)
2. Check background load: `top` or `htop`
3. Check disk I/O: `iostat -x 1`
4. Run during low-activity period (e.g., overnight)

### Slow Performance

If latencies significantly exceed targets:

1. Verify release build: `cargo build --release` (debug is 5-10x slower)
2. Check thermal throttling: monitor CPU temperature
3. Check swap usage: `vmstat 1` (swapping kills performance)
4. Profile hot paths: `cargo flamegraph --bin engram`

## References

### Competitor Benchmarks

- **Qdrant**: https://qdrant.tech/benchmarks/
  - Hardware: 8 vCPU, 16GB RAM, 100GB SSD
  - Commit: Published January 2024

- **Neo4j**: https://neo4j.com/developer/graph-data-science/performance/
  - Hardware: 4 cores, 8GB RAM
  - Version: Neo4j 5.x Community Edition

- **Milvus**: https://milvus.io/docs/benchmark.md
  - Hardware: 16 vCPU, 64GB RAM, GPU optional
  - Version: Milvus 2.3.x

### Internal Documentation

- M17 Performance Workflow: `roadmap/milestone-17/PERFORMANCE_WORKFLOW.md`
- M17 Performance Log: `roadmap/milestone-17/PERFORMANCE_LOG.md`
- Load Test Tool: `tools/loadtest/README.md`
- Vision: `vision.md` (competitive positioning section)

## Contributing

When adding new competitive scenarios:

1. **Document competitor baseline**: Include URL, hardware specs, version
2. **Use deterministic seed**: Increment from last seed (e.g., 46, 47, ...)
3. **Add validation tests**: Extend determinism/resource/correctness scripts
4. **Update this README**: Add row to table with targets and references
5. **Run full validation**: All 3 validation scripts must pass
6. **Update quarterly report**: Add scenario to `generate_competitive_report.py`

## License

Scenarios are part of Engram project under Apache 2.0 license.
Competitor benchmarks are referenced for comparison purposes only.
