# Task 004: Dataset Scaling Tests (100K → 10M nodes)

**Status**: Pending
**Estimated Duration**: 5-6 days
**Priority**: Critical - Validates scalability targets

## Objective

Characterize performance across 100K, 1M, 10M node scales with automated memory/latency curve generation. Identify inflection points where algorithmic complexity dominates. Validate sub-linear throughput degradation target (<20%).

## Problem Analysis

HNSW index performance: O(log N) for search, but constants matter:
- **100K nodes**: ~768KB embedding storage, fits in L3 cache
- **1M nodes**: ~7.7MB embeddings, L3 cache misses begin
- **10M nodes**: ~77MB embeddings, DRAM latency dominates

Spreading activation: O(E) where E = edges touched:
- Fan-out grows with graph density
- Cache locality degrades with scale

Need to measure:
- **Memory scaling**: Should be O(N) linear
- **Latency scaling**: Target <5x from 100K → 10M
- **Throughput degradation**: <20% from peak

## Implementation Highlights

### Automated Scaling Test Suite

```bash
#!/bin/bash
# scripts/run_scaling_tests.sh

for scale in 100000 1000000 10000000; do
    echo "Testing scale: $scale nodes"

    # Generate scenario
    envsubst < scenarios/scaling/template.toml > /tmp/scale_$scale.toml

    # Run test
    ./target/release/loadtest run \
        --scenario /tmp/scale_$scale.toml \
        --duration 300 \
        --output tmp/scaling/scale_${scale}_results.json

    # Analyze
    python3 scripts/analyze_scaling.py \
        --results tmp/scaling/scale_${scale}_results.json \
        --scale $scale
done

# Generate scaling curves
python3 scripts/plot_scaling_curves.py \
    --input tmp/scaling/ \
    --output tmp/scaling/curves.png
```

### Memory Scaling Analysis

```rust
pub struct ScalingMetrics {
    pub node_count: usize,
    pub total_memory_mb: f64,
    pub memory_per_node_bytes: f64,
    pub p50_latency_ms: f64,
    pub p99_latency_ms: f64,
    pub throughput_ops_sec: f64,
}

impl ScalingMetrics {
    pub fn memory_overhead_ratio(&self) -> f64 {
        let theoretical_size = self.node_count * (768 * 4 + 128); // embedding + metadata
        (self.total_memory_mb * 1024.0 * 1024.0) / theoretical_size as f64
    }

    pub fn throughput_efficiency(&self, baseline: &Self) -> f64 {
        // Throughput should be constant ideally, measure degradation
        self.throughput_ops_sec / baseline.throughput_ops_sec
    }
}
```

## Test Matrix

| Scale | Nodes | Memory | P99 Target | Throughput Target |
|-------|-------|--------|------------|-------------------|
| Small | 100K | ~200MB | <1ms | 1000 ops/s |
| Medium | 1M | ~2GB | <5ms | >800 ops/s |
| Large | 10M | ~20GB | <15ms | >500 ops/s |

## Success Criteria

- **Memory**: Linear scaling O(N) with <2x overhead
- **Latency**: P99 <5ms at 1M, <15ms at 10M
- **Throughput**: >50% of baseline at 10M scale
- **Competitive**: Beat Neo4j at all scales (27.96ms baseline)

## Hardware Requirements

- **1M test**: 16GB RAM minimum
- **10M test**: 64GB RAM minimum (Tier 2 hardware)
- SSD required for all scales

## Files

- `scripts/run_scaling_tests.sh` (150 lines)
- `scripts/analyze_scaling.py` (300 lines)
- `scenarios/scaling/template.toml`
- `tools/loadtest/src/scaling/analyzer.rs` (450 lines)
