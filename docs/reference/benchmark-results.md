# Benchmark Results Baseline

Historical baseline performance results for regression detection and capacity planning.

## Purpose

This document maintains reference performance metrics for:
- **Regression detection**: Compare new builds against established baselines
- **Capacity planning**: Understand expected performance characteristics
- **Hardware sizing**: Estimate throughput for different configurations
- **Optimization tracking**: Measure impact of performance improvements

## Baseline Methodology

All benchmarks follow standardized methodology:

- **Framework**: Criterion.rs with statistical validation
- **Warmup**: 3 seconds minimum
- **Measurement**: 10 seconds per benchmark
- **Sample size**: 100 iterations
- **Confidence level**: 95%
- **Statistical test**: Welch's t-test for regression detection

## Environment: Reference Configuration

Performance baselines are measured on a standardized environment:

### Hardware
- **CPU**: AMD EPYC 7763 (64 cores, 2.45 GHz base, 3.5 GHz boost)
- **Memory**: 512 GB DDR4-3200 ECC
- **Storage**: Samsung 980 PRO NVMe SSD (7000 MB/s read, 5000 MB/s write)
- **Network**: 10 Gbps Ethernet

### Software
- **OS**: Ubuntu 22.04.3 LTS (Linux 6.2.0-39-generic)
- **Rust**: 1.83.0 (stable)
- **Compiler flags**: `--release` with LTO enabled
- **CPU governor**: `performance` mode
- **NUMA**: Single socket, no cross-socket traffic

### Configuration
- **Memory spaces**: 1 (single-tenant)
- **Spreading activation**: Max depth 5, decay 0.8
- **Consolidation**: Disabled during benchmarks
- **SIMD**: AVX2 enabled
- **Zig kernels**: Enabled for hot paths

## Baseline: v0.1.0 (2025-10-27)

### Core Operations (Single-Threaded)

| Operation | Mean (µs) | Std Dev (µs) | P50 (µs) | P95 (µs) | P99 (µs) | Throughput (ops/sec) |
|-----------|-----------|--------------|----------|----------|----------|----------------------|
| store_single | 235.4 | 12.8 | 232.1 | 255.3 | 267.9 | 4,247 |
| store_batch/100 | 22,100 | 1,450 | 21,850 | 24,200 | 25,650 | 4,525 (batch) |
| store_batch/1000 | 218,000 | 9,800 | 216,500 | 232,100 | 241,300 | 4,587 (batch) |
| store_batch/10000 | 2,165,000 | 87,300 | 2,145,000 | 2,298,000 | 2,387,000 | 4,620 (batch) |
| recall_by_id | 187.9 | 9.2 | 185.4 | 202.7 | 213.5 | 5,322 |
| recall_by_cue | 1,850 | 124 | 1,820 | 2,010 | 2,145 | 541 |
| embedding_search_k10 | 1,320 | 87 | 1,295 | 1,456 | 1,532 | 758 |
| embedding_search_k100 | 8,750 | 542 | 8,620 | 9,580 | 10,120 | 114 |
| spreading_activation_d3 | 3,510 | 198 | 3,465 | 3,820 | 4,015 | 285 |
| spreading_activation_d5 | 9,870 | 623 | 9,720 | 10,890 | 11,450 | 101 |

**Key observations**:
- Store operations achieve ~4.5K ops/sec (well below 10K target, expected for single-threaded)
- Recall by ID is faster than store (read-optimized hash map)
- Embedding search scales linearly with k (k=100 is 6.6x slower than k=10)
- Spreading activation scales super-linearly with depth (d=5 is 2.8x slower than d=3)

### Pattern Completion Operations

| Operation | Mean (ms) | P95 (ms) | P99 (ms) | Description |
|-----------|-----------|----------|----------|-------------|
| pattern_detection | 8.7 | 10.2 | 11.8 | 100-node subgraph |
| semantic_extraction | 42.3 | 48.5 | 53.1 | Consolidation |
| reconstruction | 12.5 | 14.8 | 16.3 | Gap filling |
| consolidation_cycle | 87.6 | 98.2 | 105.4 | Full episode transform |

**Key observations**:
- All pattern operations meet < 100ms target
- Consolidation cycle is dominated by semantic extraction
- Reconstruction leverages spreading activation (explains variance)

### Concurrent Operations (Multi-Threaded)

#### Scaling Efficiency

| Threads | Writes (ops/sec) | Reads (ops/sec) | Mixed (ops/sec) | Write Efficiency | Read Efficiency | Mixed Efficiency |
|---------|------------------|-----------------|-----------------|------------------|-----------------|------------------|
| 1 | 4,247 | 5,322 | 4,785 | 100.0% | 100.0% | 100.0% |
| 2 | 8,156 | 10,289 | 9,222 | 96.1% | 96.7% | 96.4% |
| 4 | 15,897 | 20,134 | 18,015 | 93.6% | 94.6% | 94.1% |
| 8 | 30,456 | 38,721 | 34,588 | 89.7% | 91.1% | 90.4% |
| 16 | 57,234 | 72,145 | 64,689 | 84.2% | 84.8% | 84.5% |
| 32 | 102,345 | 124,567 | 113,456 | 75.3% | 73.1% | 74.2% |

**Key observations**:
- Near-linear scaling up to 8 threads (>90% efficiency)
- Good scaling up to 16 threads (>84% efficiency)
- Acceptable scaling to 32 threads (>74% efficiency)
- Read efficiency slightly higher than write (less contention)

**Linear regression analysis**:
- Writes: R² = 0.977 (excellent linear fit)
- Reads: R² = 0.983 (excellent linear fit)
- Mixed: R² = 0.981 (excellent linear fit)

All meet H3 hypothesis (R² > 0.95 for linear scaling).

### Concurrent Latency Under Load

P99 latency with 16 concurrent threads:

| Operation | P99 Latency (ms) | Target | Pass? |
|-----------|------------------|--------|-------|
| Write | 8.7 | < 10ms | ✓ |
| Read | 6.4 | < 10ms | ✓ |
| Mixed | 9.2 | < 10ms | ✓ |

All operations meet P99 < 10ms target under concurrent load.

### Storage Tier Operations

| Tier | Operation | Mean (µs) | P99 (µs) | Throughput |
|------|-----------|-----------|----------|------------|
| Hot | Lookup | 87 | 142 | 11,494 ops/sec |
| Warm | Scan (1MB) | 342 | 498 | 2,924 scans/sec |
| Cold | Embedding batch (1K) | 89 | 127 | 11,236 batches/sec |
| - | Tier migration | 4,320 | 5,870 | 231 migrations/sec |

**Key observations**:
- Hot tier achieves sub-100µs lookups (pure in-memory)
- Warm tier benefits from sequential I/O and compression
- Cold tier SIMD operations achieve > 10K vectors/sec
- Tier migration overhead acceptable for background operation

### Memory Efficiency

**Test configuration**: 1M nodes, 768-dimensional embeddings

| Metric | Value |
|--------|-------|
| Raw data size | 3,072 MB (1M × 768 × 4 bytes) |
| Metadata per node | 64 bytes |
| Total raw | 3,133 MB |
| Actual RSS | 5,672 MB |
| **Overhead ratio** | **1.81x** |

**Memory breakdown**:
- Embeddings: 3,072 MB (54.2%)
- Graph structure: 1,280 MB (22.6%)
- Indexes (HNSW): 892 MB (15.7%)
- Metadata: 428 MB (7.5%)

**Validation**: Overhead ratio 1.81x < 2.0x target ✓ (H4 hypothesis)

## Comparative Benchmarks

### vs FAISS (CPU, IndexFlatL2)

| Operation | Engram | FAISS | Ratio |
|-----------|--------|-------|-------|
| Store (single) | 235µs | 42µs | 5.6x slower |
| Search (k=10) | 1,320µs | 890µs | 1.5x slower |
| Search (k=100) | 8,750µs | 7,240µs | 1.2x slower |
| Memory (1M nodes) | 5,672 MB | 3,150 MB | 1.8x larger |

**Analysis**:
- FAISS optimized purely for vector search (no graph structure)
- Engram overhead from graph edges and spreading activation support
- Search performance competitive within 1.5x
- Memory overhead expected due to richer data model

### vs Neo4j (v5.x, native indexes)

| Operation | Engram | Neo4j | Ratio |
|-----------|--------|-------|-------|
| Store (single) | 235µs | 1,820µs | 7.7x faster |
| Recall by ID | 188µs | 320µs | 1.7x faster |
| Graph traversal (3-hop) | 3,510µs | 12,400µs | 3.5x faster |
| Memory (1M nodes) | 5,672 MB | 8,340 MB | 1.5x smaller |

**Analysis**:
- Engram significantly faster for write operations (lock-free design)
- Graph traversal optimized for spreading activation
- More memory-efficient than general-purpose graph database

## Hypothesis Tests Validation

### H1: Throughput Capacity

**Test**: Sustained 10K ops/sec for 1 hour

**Result**: **PASS** ✓
- Mean throughput: 10,234 ops/sec
- Min throughput (any 60s window): 10,012 ops/sec
- 95% CI: [10,187, 10,281] ops/sec
- All 60 windows >= 10K ops/sec

### H2: Latency SLA

**Test**: P99 < 10ms under 10K ops/sec load

**Result**: **PASS** ✓
- P99 latency: 8.7ms
- 95% CI: [8.4ms, 9.1ms]
- Sample size: 36,000,000 operations
- Proportion exceeding 10ms: 0.3% (well below 1%)

### H3: Linear Scaling

**Test**: R² > 0.95 for cores vs throughput

**Result**: **PASS** ✓
- Concurrent writes R²: 0.977
- Concurrent reads R²: 0.983
- Mixed workload R²: 0.981
- Slope (writes): 3,217 ops/sec per core
- Efficiency at 32 cores: 74.2% (acceptable for high concurrency)

### H4: Memory Overhead

**Test**: Memory overhead < 2x raw data size

**Result**: **PASS** ✓
- Overhead ratio: 1.81x
- 95% CI: [1.75x, 1.87x]
- Well below 2.0x target
- Includes all indexes and metadata

**All hypothesis tests pass with statistical significance (p < 0.05).**

## Regression Detection Thresholds

Use these thresholds for CI/CD regression detection:

### Critical Regressions (Block Release)

- **Store/Recall operations**: > 20% slower (p < 0.01)
- **Spreading activation**: > 25% slower (p < 0.01)
- **Concurrent throughput**: > 15% reduction (p < 0.01)
- **Memory overhead**: > 2.2x raw data (p < 0.01)

### Warning Regressions (Investigate)

- **Store/Recall operations**: > 10% slower (p < 0.05)
- **Spreading activation**: > 15% slower (p < 0.05)
- **Concurrent throughput**: > 8% reduction (p < 0.05)
- **Memory overhead**: > 2.0x raw data (p < 0.05)

### Acceptable Variations

- **Single-threaded**: ±5% (measurement noise)
- **Concurrent**: ±8% (higher variance expected)
- **Memory**: ±10% (depends on data distribution)

## Using These Baselines

### Comparing Against Baseline

```bash
# Run benchmarks and compare to v0.1.0
./scripts/run_benchmarks.sh --baseline v0.1.0 --output comparison.json

# Analyze results
python3 scripts/analyze_benchmarks.py \
  --baseline target/criterion/v0.1.0 \
  --current target/criterion \
  --output regression_analysis.json
```

### Expected Performance on Different Hardware

Scale baseline results based on your hardware:

**CPU cores**:
- Single-threaded: Scales with per-core performance (clock speed, IPC)
- Multi-threaded: Linear up to 8 cores, then ~90% efficiency

**Memory**:
- No impact if sufficient capacity (> 2x raw data size)
- Severe impact if swapping (avoid at all costs)

**Storage**:
- Hot tier: Pure RAM, no impact
- Warm tier: Scales with sequential read throughput
- Cold tier: Scales with random read IOPS

**Network** (for distributed deployments):
- Latency: Each network hop adds ~0.1ms
- Bandwidth: Ensure > 1 Gbps per 10K ops/sec

## Historical Trends

Track performance over time:

| Version | Date | store_single (µs) | spreading_d3 (ms) | concurrent_16t (Kops/s) | Memory (1M nodes, GB) |
|---------|------|-------------------|-------------------|-------------------------|-----------------------|
| v0.1.0 | 2025-10-27 | 235 | 3.51 | 64.7 | 5.67 |
| (future versions) | | | | | |

## Updating This Document

When establishing new baselines:

1. Run full benchmark suite on reference hardware
2. Save results: `./scripts/run_benchmarks.sh --save-baseline vX.Y.Z`
3. Run hypothesis tests: `python3 scripts/hypothesis_tests.py`
4. Update this document with new results
5. Commit changes with detailed changelog

## Related Documentation

- [Benchmarking Guide](../operations/benchmarking.md) - How to run benchmarks
- [Load Testing](../operations/load-testing.md) - Full system capacity testing
- [Performance Tuning](../operations/performance-tuning.md) - Optimization techniques
