# Competitive Performance Baselines

This document tracks Engram's performance against leading vector databases (Qdrant, Milvus, Weaviate) and graph databases (Neo4j). Measurements are updated quarterly to validate competitive positioning and identify optimization priorities. All baselines are reproducible using standardized scenarios in `scenarios/competitive/`.

## Quick Reference

- [Competitor Baseline Summary](#competitor-baseline-summary)
- [Qdrant (Vector Database)](#qdrant-vector-database)
- [Neo4j (Graph Database)](#neo4j-graph-database)
- [Milvus (Vector Database)](#milvus-vector-database)
- [Weaviate (Vector Database)](#weaviate-vector-database)
- [Redis (Vector Database)](#redis-vector-database)
- [Scenario Mapping](#scenario-mapping)
- [Engram Performance Targets](#engram-performance-targets)
- [Competitive Positioning](#competitive-positioning)
- [Measurement Methodology](#measurement-methodology)
- [Quarterly Review Process](#quarterly-review-process)
- [Historical Trends](#historical-trends)
- [References](#references)

## Competitor Baseline Summary

| System | Category | Workload | P99 Latency | Throughput (QPS) | Recall | Dataset | Scenario File | Source |
|--------|----------|----------|-------------|------------------|--------|---------|---------------|--------|
| Qdrant | Vector DB | ANN search | 22-24ms | 626 | 99.5% | 1M 768d | [`qdrant_ann_1m_768d.toml`](../../scenarios/competitive/qdrant_ann_1m_768d.toml) | [Qdrant Benchmarks](https://qdrant.tech/benchmarks/) |
| Neo4j | Graph DB | 1-hop traversal | 27.96ms | 280 | N/A | 100K nodes | [`neo4j_traversal_100k.toml`](../../scenarios/competitive/neo4j_traversal_100k.toml) | [Neo4j Performance](https://neo4j.com/developer/graph-data-science/performance/) |
| Milvus | Vector DB | ANN search | 708ms | 2,098 | 100% | 10M | [`milvus_ann_10m_768d.toml`](../../scenarios/competitive/milvus_ann_10m_768d.toml) | [Milvus Benchmarks](https://milvus.io/docs/benchmark.md) |
| Weaviate | Vector DB | ANN search | 70-150ms | 200-400 | N/A | 1M | (pending) | [Weaviate Benchmarks](https://weaviate.io/developers/weaviate/benchmarks) |
| Redis | Vector DB | Vector search | 8ms | N/A | N/A | Unknown | (pending) | [Redis Vectors](https://redis.io/docs/stack/search/reference/vectors/) |

**Note**: All latencies measured on comparable hardware (modern multi-core CPU, 32GB+ RAM, NVMe SSD).
Dataset notation: "1M 768d" = 1 million vectors, 768 dimensions.

## Detailed Baseline Data

### Qdrant (Vector Database)

**Overview**: Qdrant is a specialized vector similarity search engine optimized for high-dimensional embeddings. It uses a custom HNSW implementation with SIMD acceleration and memory-mapped storage for large datasets.

**Benchmark Configuration**:
- **Hardware**: 8-core Intel Xeon, 32GB RAM, NVMe SSD
- **Version**: Qdrant v1.7+
- **Dataset**: 1M vectors, 768 dimensions (OpenAI ada-002 embedding size)
- **Index**: HNSW with M=16, ef_construction=128
- **Search Parameters**: ef_search=64 for 99.5% recall

**Performance Characteristics**:

| Metric | Value | Notes |
|--------|-------|-------|
| P50 Latency | 18ms | Median search time |
| P99 Latency | 22-24ms | Tail latency varies with load |
| Throughput | 626 QPS | Single-threaded client |
| Recall | 99.5% | At ef_search=64 |
| Index Build Time | ~45 min | For 1M vectors |
| Memory Footprint | ~1.5GB | With memory-mapped storage |

**Engram Comparison**:
- **Advantage (Qdrant)**: Pure vector search highly optimized, mature SIMD implementation
- **Advantage (Engram)**: Supports hybrid vector+graph+temporal queries in single system
- **Target**: Engram should achieve P99 <20ms for pure vector search (10% faster than Qdrant)

**Source**: [Qdrant Benchmarks](https://qdrant.tech/benchmarks/) (accessed 2025-11-09)

**Reproduction**:
```bash
cargo run --release --bin loadtest -- run \
  --scenario scenarios/competitive/qdrant_ann_1m_768d.toml \
  --duration 60 \
  --seed 42 \
  --output tmp/qdrant_baseline.json
```

### Neo4j (Graph Database)

**Overview**: Neo4j is an ACID-compliant native graph database written in Java. It uses a property graph model with Cypher query language. Performance is bounded by JVM garbage collection pauses and disk-based transaction logs.

**Benchmark Configuration**:
- **Hardware**: 16-core Xeon Silver, 64GB RAM, SSD
- **Version**: Neo4j 5.14 Enterprise Edition
- **Dataset**: 100K nodes, 1M edges (scale-free distribution, average degree 10)
- **Query**: Single-hop MATCH traversal with property filter
- **JVM**: OpenJDK 17, 32GB heap, G1GC

**Performance Characteristics**:

| Metric | Value | Notes |
|--------|-------|-------|
| P50 Latency | 18ms | Median traversal time |
| P99 Latency | 27.96ms | Includes GC pauses |
| Throughput | 280 QPS | Single client |
| Cache Hit Rate | ~85% | With warmed caches |
| Write Latency | 45ms | Transaction log flush |

**Engram Comparison**:
- **Advantage (Neo4j)**: Mature Cypher ecosystem, ACID guarantees, rich query optimizer
- **Advantage (Engram)**: Lock-free graph navigation avoids GC pauses, probabilistic spreading activation
- **Target**: Engram should achieve P99 <15ms for graph traversal (46% faster than Neo4j)

**Source**: [Neo4j Performance Guide](https://neo4j.com/developer/graph-data-science/performance/) (v5.14 docs, accessed 2025-11-09)

**Reproduction**:
```bash
cargo run --release --bin loadtest -- run \
  --scenario scenarios/competitive/neo4j_traversal_100k.toml \
  --duration 60 \
  --seed 43 \
  --output tmp/neo4j_baseline.json
```

### Milvus (Vector Database)

**Overview**: Milvus is a cloud-native vector database designed for massive-scale similarity search. It supports multiple index types, GPU acceleration, and distributed deployment.

**Benchmark Configuration**:
- **Hardware**: 16-core Intel Xeon, 64GB RAM, GPU optional
- **Version**: Milvus 2.3+
- **Dataset**: 10M vectors, 768 dimensions
- **Index**: IVF_FLAT with nlist=4096
- **Search Parameters**: nprobe=128 for 100% recall

**Performance Characteristics**:

| Metric | Value | Notes |
|--------|-------|-------|
| P50 Latency | 450ms | Median at 10M scale |
| P99 Latency | 708ms | Tail latency at scale |
| Throughput | 2,098 QPS | Multi-threaded |
| Recall | 100% | Exhaustive IVF_FLAT |
| Index Build Time | ~3 hours | For 10M vectors |
| Memory Footprint | ~10-15GB | Full dataset |

**Engram Comparison**:
- **Advantage (Milvus)**: Proven massive-scale (billions of vectors), GPU acceleration, distributed architecture
- **Advantage (Engram)**: Lower latency at moderate scale, unified memory model avoids network hops
- **Target**: Engram should achieve P99 <100ms for 10M vectors (86% faster, stretch goal)

**Source**: [Milvus Benchmarks](https://milvus.io/docs/benchmark.md) (accessed 2025-11-09)

**Reproduction**:
```bash
# WARNING: Requires 16GB+ RAM, ~1 hour test duration
cargo run --release --bin loadtest -- run \
  --scenario scenarios/competitive/milvus_ann_10m_768d.toml \
  --duration 60 \
  --seed 45 \
  --output tmp/milvus_baseline.json
```

### Weaviate (Vector Database)

**Overview**: Weaviate is an open-source vector database with built-in semantic search and hybrid search capabilities. It offers a GraphQL API and supports multiple vectorization modules.

**Benchmark Configuration**:
- **Hardware**: Comparable to Qdrant (8-core, 32GB RAM)
- **Version**: Weaviate 1.22+
- **Dataset**: 1M vectors, estimated 768 dimensions
- **Index**: HNSW implementation

**Performance Characteristics**:

| Metric | Value | Notes |
|--------|-------|-------|
| P99 Latency | 70-150ms | Wide range depending on config |
| Throughput | 200-400 QPS | Varies by query complexity |
| Recall | N/A | Not specified in public benchmarks |

**Status**: Baseline measurement pending. Weaviate provides less detailed public benchmarks compared to Qdrant/Milvus.

**Source**: [Weaviate Benchmarks](https://weaviate.io/developers/weaviate/benchmarks) (accessed 2025-11-09)

### Redis (Vector Database)

**Overview**: Redis Stack includes vector similarity search via the RediSearch module, enabling fast in-memory vector operations alongside traditional Redis data structures.

**Benchmark Configuration**:
- **Hardware**: Not specified in public docs
- **Version**: Redis Stack 7.2+
- **Dataset**: Not specified

**Performance Characteristics**:

| Metric | Value | Notes |
|--------|-------|-------|
| P99 Latency | 8ms | Claimed in-memory performance |
| Details | Limited | Public benchmarks lack detail |

**Status**: Baseline measurement pending. Redis provides minimal public benchmark data for vector operations.

**Source**: [Redis Vector Documentation](https://redis.io/docs/stack/search/reference/vectors/) (accessed 2025-11-09)

## Scenario Mapping

Each competitor baseline corresponds to exactly one standardized TOML scenario file:

| Competitor | Workload | Scenario File | Configuration Highlights |
|------------|----------|---------------|-------------------------|
| Qdrant | ANN search 1M | [`scenarios/competitive/qdrant_ann_1m_768d.toml`](../../scenarios/competitive/qdrant_ann_1m_768d.toml) | 100% embedding_search_weight, 1M nodes, 768 dims, seed=42 |
| Neo4j | Graph traversal 100K | [`scenarios/competitive/neo4j_traversal_100k.toml`](../../scenarios/competitive/neo4j_traversal_100k.toml) | 80% recall_weight (1-hop), 20% 2-hop, 100K nodes, seed=43 |
| Milvus | ANN search 10M | [`scenarios/competitive/milvus_ann_10m_768d.toml`](../../scenarios/competitive/milvus_ann_10m_768d.toml) | 100% embedding_search_weight, 10M nodes, 768 dims, seed=45 (stretch goal) |
| Engram | Hybrid workload | [`scenarios/competitive/hybrid_production_100k.toml`](../../scenarios/competitive/hybrid_production_100k.toml) | 30% store, 30% recall, 30% search, 10% completion, seed=44 (no competitor) |

**Important**: All scenarios use deterministic seeds for reproducibility. Running the same scenario with the same seed produces identical operation sequences.

## Engram Performance Targets

Targets represent aspirational goals derived from competitive analysis and architectural capabilities. Each target includes rationale explaining why the number is achievable or stretch.

### ANN Search (1M vectors, 768d)

**Target**: P99 < 20ms, Throughput > 650 QPS, Recall > 99.5%

**Rationale**:
- **Why achievable**: Engram's HNSW implementation uses SIMD (like Qdrant) and benefits from M17 dual-memory optimizations that reduce pointer chasing
- **Competitive gap**: Current best is Qdrant at 22-24ms; target represents 10-20% improvement
- **Trade-offs**: Pure vector search may sacrifice some latency for hybrid query flexibility

**Dependencies**:
- M17 dual-memory consolidation (complete)
- SIMD vector operations (complete)
- Cache-aware HNSW layout (pending - M18 Task 003)

**Validation**: Success = 3 consecutive benchmark runs within target, <5% variance

---

### Graph Traversal (100K nodes)

**Target**: P99 < 15ms, Throughput > 300 QPS

**Rationale**:
- **Why achievable**: Neo4j's 27.96ms baseline uses JVM with GC pauses; Engram's Rust implementation avoids stop-the-world pauses
- **Competitive gap**: Target represents 46% improvement over Neo4j
- **Trade-offs**: Engram optimizes for probabilistic spreading activation, not just deterministic traversal

**Dependencies**:
- Lock-free graph navigation (complete)
- NUMA-aware node placement (complete)
- Edge cache optimization (pending - M18 Task 005)

**Validation**: Measure on Neo4j-equivalent graph structure (scale-free, avg degree 10)

---

### Hybrid Workload (100K nodes)

**Target**: P99 < 10ms, Throughput > 500 QPS

**Rationale**:
- **No competitor baseline**: Unique to Engram's architecture
- **Target derivation**: 50% of pure vector target + 33% of graph traversal target, accounting for workload diversity
- **Strategic importance**: Demonstrates differentiation in market positioning

**Dependencies**:
- Integrated query execution (complete)
- Workload-aware scheduling (pending - M18 Task 007)

**Validation**: Mix of 30% store, 30% recall, 30% search, 10% pattern completion

## Competitive Positioning

### Engram's Strengths

1. **Unified Architecture**: Only system supporting vector search + graph traversal + temporal decay + pattern completion in a single query
2. **Probabilistic Memory**: Confidence-based operations enable graceful degradation and uncertainty quantification
3. **Memory Consolidation**: Automatic schema extraction and semantic compression reduce storage overhead
4. **Lock-Free Design**: Rust's ownership model enables high-concurrency operations without GC pauses
5. **Cognitive Plausibility**: Architecture inspired by hippocampal-neocortical systems for biologically-inspired AI

### Engram's Weaknesses

1. **Pure Vector Search**: Expected to lag specialized vector DBs (Qdrant, Milvus) initially due to hybrid architecture overhead
2. **Maturity**: Newer system with less production battle-testing than Neo4j or Qdrant
3. **Ecosystem**: Smaller community, fewer integrations, less extensive documentation
4. **Scale Validation**: Not yet tested at Milvus-scale (billions of vectors, petabyte datasets)

### Market Differentiation

**Engram is the only system that:**
- Combines vector similarity, graph traversal, and temporal reasoning in unified queries
- Implements biologically-inspired memory consolidation (episodic → semantic transformation)
- Provides probabilistic confidence scores with cognitive error modeling
- Supports pattern completion using hippocampal-inspired dynamics

**Example differentiating query**:
```rust
// Find memories similar to a cue, spread activation through graph,
// apply temporal decay, and complete partial patterns - all in one query
let results = engram.recall(&partial_cue)
    .with_confidence_threshold(0.7)
    .spread_activation(max_hops: 3, decay: 0.9)
    .apply_temporal_decay(since: 30.days_ago())
    .complete_pattern()
    .execute()?;
```

No competitor (Qdrant, Neo4j, Milvus, Weaviate) supports this query pattern natively.

## Measurement Methodology

All measurements follow this standardized protocol to ensure reproducibility:

### Hardware Specification

**Reference Platform**: Apple M1 Max (10-core CPU, 32GB unified memory)
- **Rationale**: Developer laptop representative of engineering workflow
- **CPU**: 8 performance cores + 2 efficiency cores @ 3.2GHz
- **Memory**: 32GB LPDDR5, 400GB/s bandwidth
- **Storage**: 1TB NVMe SSD, 7.4GB/s read

**Production Platform**: Linux x86_64 (32-core Xeon Gold, 128GB RAM, NVMe SSD)
- **Rationale**: Matches Qdrant/Neo4j published benchmark environments
- **Usage**: Quarterly review measurements use this platform
- **CPU**: Intel Xeon Gold 6258R @ 2.7GHz (turbo 4.0GHz)
- **Memory**: 128GB DDR4-2933, 8-channel
- **Storage**: Samsung PM9A3 2TB NVMe, 6.9GB/s read

### Test Configuration

**Engram Version**: Git commit hash from measurement (e.g., `a1b2c3d4e5f6`)
- **Build**: `cargo build --release` with LTO enabled
- **Config**: Default settings unless noted in scenario file
- **Rust Version**: 1.75+ (Edition 2024)

**Load Test Parameters**:
- **Duration**: 60 seconds per scenario (after warmup)
- **Warmup**: 10 seconds to stabilize caches and JIT
- **Client**: Single-threaded deterministic client
- **Seed**: Fixed per scenario (qdrant=42, neo4j=43, hybrid=44, milvus=45)
- **Output**: JSON with full HdrHistogram data

### Execution Protocol

```bash
# 1. Clean environment (no background processes)
pkill -9 engram || true
sleep 5

# 2. Start Engram with diagnostics
engram start --metrics-port 9090 &
ENGRAM_PID=$!
sleep 2  # Wait for server ready

# 3. Warmup (10s)
cargo run --release --bin loadtest -- run \
  --scenario $SCENARIO \
  --duration 10 \
  --seed $SEED \
  --warmup

# 4. Measurement (60s)
cargo run --release --bin loadtest -- run \
  --scenario $SCENARIO \
  --duration 60 \
  --seed $SEED \
  --output results.json

# 5. Collect diagnostics
./scripts/engram_diagnostics.sh >> diagnostics.log

# 6. Shutdown
kill $ENGRAM_PID
wait
```

### Data Collection

**Metrics Captured**:
- **Latency**: Full HdrHistogram (P50, P90, P95, P99, P99.9, max)
- **Throughput**: Operations per second (mean, min, max)
- **System**: CPU usage, memory RSS, disk I/O (via `/usr/bin/time -v`)
- **Engram**: Cache hit rate, consolidation events, decay calculations, spreading activation depth

**Data Storage**:
- **Raw results**: `tmp/competitive_baselines/{scenario}_{date}.json`
- **Historical archive**: `benchmarks/competitive_history/{YYYY-QN}/` (git-tracked)

### Reproducibility Checklist

Before accepting a measurement as valid:

- [ ] Three consecutive runs show <5% P99 variance
- [ ] No system alerts during test window (check `dmesg`)
- [ ] CPU throttling disabled (check `cpufreq-info` or equivalent)
- [ ] Swap usage = 0 (check `free -h` or `vm_stat`)
- [ ] No competing processes (verify with `top` or `htop`)
- [ ] Diagnostics log shows no errors
- [ ] Engram server logs clean (no warnings/errors)

If any check fails, discard measurement and investigate root cause.

## Automated Enforcement: Pre-Commit Hook

To ensure competitive benchmarks stay current, a git pre-commit hook enforces quarterly measurements. This hook prevents commits if benchmarks haven't been run within the last 90 days.

**Installation**:
```bash
./scripts/install_git_hooks.sh
```

**Behavior**:
- Checks timestamp of most recent benchmark in `tmp/competitive_benchmarks/`
- Blocks commit if >90 days old
- Warns if expiring within 14 days
- Clear error messages with instructions to run benchmarks

**Bypass** (use sparingly):
```bash
SKIP_BENCHMARK_CHECK=1 git commit -m "hotfix: critical bug"
```

**Recommended bypass scenarios**:
- Hotfixes that must deploy immediately
- CI/CD environments (set `SKIP_BENCHMARK_CHECK=1` in CI)
- Automated workflows (release scripts, bots)

**Not recommended**:
- Regular development work
- Feature branches merging to main
- Release branches

**CI/CD Integration**:

For continuous integration systems, set the bypass variable in your environment:

```yaml
# GitHub Actions example
env:
  SKIP_BENCHMARK_CHECK: 1
```

```bash
# GitLab CI example
variables:
  SKIP_BENCHMARK_CHECK: "1"
```

The hook is cross-platform compatible with both macOS (BSD date) and Linux (GNU date).

## Quarterly Review Process

**Schedule**: First week of January, April, July, October

**Owner**: Performance Engineering Lead
- **Primary**: @performance-lead (GitHub handle TBD)
- **Backup**: @systems-architect (GitHub handle TBD)
- **Stakeholders**: Engineering team, product manager

**Time Commitment**: ~8 hours per quarter
- 4 hours: Run benchmark suite + validate results
- 2 hours: Analyze trends, identify optimization opportunities
- 1 hour: Update documentation
- 1 hour: Present findings to team

**Deliverables**:
1. Updated `docs/reference/competitive_baselines.md` (this file)
2. Raw measurement data in `benchmarks/competitive_history/YYYY-QN/`
3. Comparison report: `benchmarks/competitive_history/YYYY-QN/report.md`
4. GitHub issues for identified optimization tasks (labeled `performance`, `competitive`)

### Quarterly Workflow

1. **Week 1 Monday**: Performance lead pulls latest `main` branch, verifies clean build
2. **Week 1 Tuesday-Wednesday**: Run full competitive benchmark suite on production platform (~4 hours compute time)
3. **Week 1 Thursday**: Analyze results, compare to previous quarter, identify regressions/improvements
4. **Week 1 Friday**: Update this document, present findings to engineering team, create optimization tasks

### Quarterly Update Template

Copy this template into the Updates section below for each quarter:

---

#### QN YYYY Update (YYYY-MM-DD)

**Measurement Details**:
- **Engram Version**: v0.X.Y (commit `abc123d`)
- **Platform**: [Linux x86_64 / M1 Max]
- **Scenarios**: [list scenarios run]
- **Duration**: 60s per scenario, 3 runs averaged

**Key Changes Since Last Quarter**:
- [Competitor baseline updates]
- [Engram performance improvements]
- [Significant milestones completed]

**Competitive Gaps**:
1. **[Workload]**: Engram vs [Competitor] - [gap percentage]
2. **[Workload]**: Engram vs [Competitor] - [gap percentage]

**Optimization Priorities for Next Quarter**:
1. [Priority 1] → Target: [metric] (task #NNN)
2. [Priority 2] → Target: [metric] (task #NNN)

**Artifacts**:
- Raw data: [`benchmarks/competitive_history/YYYY-QN/`](../../benchmarks/competitive_history/YYYY-QN/)
- Comparison report: [QN YYYY Report](../../benchmarks/competitive_history/YYYY-QN/report.md)

---

### Immediate Update Triggers

Update outside quarterly schedule when:

1. **Competitor releases major version** with claimed performance improvements >20%
   - Action: Re-run affected scenario, update baseline with version note
2. **Engram optimization achieves >10% improvement** on competitive scenario
   - Action: Update Engram measurement, note milestone achievement
3. **Methodology changes** (new hardware, different measurement tool)
   - Action: Re-baseline ALL competitors for consistency
4. **External source updates** (competitor publishes new benchmarks)
   - Action: Update citation and values, note source date change

## Historical Trends

### Baseline Measurements

*This section will be populated after Task 006 (Initial Baseline Measurement) completes.*

**Planned Structure**:
```markdown
### Engram vs Qdrant ANN Search (P99 Latency)

| Quarter | Qdrant | Engram | Gap | Change |
|---------|--------|--------|-----|--------|
| Q4 2025 | 22ms | TBD | TBD | Initial baseline |

### Engram vs Neo4j Graph Traversal (P99 Latency)

| Quarter | Neo4j | Engram | Gap | Change |
|---------|-------|--------|-----|--------|
| Q4 2025 | 27.96ms | TBD | TBD | Initial baseline |
```

**Visualization**: Future - generate `benchmarks/competitive_history/trends.png` showing quarterly progress.

## References

### Competitor Benchmark Sources

- **Qdrant**: https://qdrant.tech/benchmarks/ (accessed 2025-11-09)
  - Hardware: 8-core Intel Xeon, 32GB RAM, NVMe SSD
  - Published: January 2024

- **Neo4j**: https://neo4j.com/developer/graph-data-science/performance/ (accessed 2025-11-09)
  - Version: Neo4j 5.14 Community Edition documentation
  - Published: 2024

- **Milvus**: https://milvus.io/docs/benchmark.md (accessed 2025-11-09)
  - Version: Milvus 2.3.x
  - Hardware: 16-core Xeon, 64GB RAM, GPU optional

- **Weaviate**: https://weaviate.io/developers/weaviate/benchmarks (accessed 2025-11-09)
  - Limited public benchmark data available

- **Redis**: https://redis.io/docs/stack/search/reference/vectors/ (accessed 2025-11-09)
  - Documentation only, no detailed benchmark results

### Internal Documentation

- [M17 Performance Workflow](../roadmap/milestone-17/PERFORMANCE_WORKFLOW.md) - Internal regression tracking
- [M17 Performance Log](../roadmap/milestone-17/PERFORMANCE_LOG.md) - Historical performance data
- [Load Test Tool](../tools/loadtest/README.md) - Usage and configuration
- [Competitive Scenario Suite](../scenarios/competitive/README.md) - Scenario details and validation

### Related Documentation

- **Tutorials**: [Running Your First Benchmark](../tutorials/first-benchmark.md) (pending) - Learn how to execute competitive scenarios step-by-step
- **How-To Guides**: [Optimize Vector Search Performance](../howto/optimize-vector-search.md) (pending) - Practical steps to close competitive gaps
- **Explanation**: [Why Engram Uses Probabilistic Memory](../explanation/probabilistic-memory.md) (pending) - Understanding architectural trade-offs
- **Operations**: [Quarterly Performance Review](../operations/quarterly-review.md) (pending) - Production workflow for updating baselines

---

**Document Version**: 1.0.0 (Initial baseline, pending first measurement)
**Last Updated**: 2025-11-09 (Task 002 completion)
**Next Review**: 2026-01-06 (Q1 2026 quarterly review)
