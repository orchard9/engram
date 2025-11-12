# Task 002: Competitive Baseline Reference Documentation

**Status**: Pending
**Complexity**: Simple
**Dependencies**: Task 001 (requires scenario definitions)
**Estimated Effort**: 3 hours

## Objective

Create comprehensive reference documentation capturing competitor baselines, Engram's positioning, and performance targets.

## Specifications

Create `docs/reference/competitive_baselines.md` with the following sections:

1. **Competitor Baselines** (table format):
   - System | Workload | P99 Latency | Throughput | Recall | Dataset Size | Source
   - Qdrant | ANN search | 22-24ms | 626 QPS | 99.5% | 1M 768d | [URL]
   - Milvus | ANN search | 708ms | 2,098 QPS | 100% | 10M | [URL]
   - Neo4j | 1-hop traversal | 27.96ms | 280 QPS | N/A | Unknown | [URL]
   - Weaviate | ANN search | 70-150ms | 200-400 QPS | N/A | 1M | [URL]
   - Redis | Vector search | 8ms | N/A | N/A | Unknown | [URL]

2. **Scenario Mapping**: Which `scenarios/competitive/*.toml` file corresponds to each baseline

3. **Performance Targets**:
   - **ANN Search (1M vectors)**: P99 < 20ms, throughput > 650 QPS, recall > 99.5%
   - **Graph Traversal (100K nodes)**: P99 < 15ms, throughput > 300 QPS
   - **Hybrid Workload (100K nodes)**: P99 < 10ms, throughput > 500 QPS (no competitor baseline)

4. **Competitive Positioning**:
   - **Strengths**: Hybrid vector+graph+temporal operations, unified API, memory consolidation
   - **Weaknesses**: Pure vector search (expected to lag specialized vector DBs initially)
   - **Differentiation**: Only system supporting probabilistic spreading activation + temporal decay + pattern completion in a single query

5. **Measurement Methodology**:
   - Hardware spec: M1 Max (10-core CPU, 32GB RAM) or equivalent x86_64
   - Engram version: Git commit hash from baseline measurement
   - Load test parameters: Deterministic seed, 60s duration, single-threaded client
   - Warm-up: 10s pre-test to stabilize caches

6. **Quarterly Review Process**:
   - Cadence: First week of each quarter (Jan, Apr, Jul, Oct)
   - Owner: Performance engineering lead
   - Deliverable: Updated `competitive_baselines.md` with new measurements
   - Trigger: Run `scripts/competitive_benchmark_suite.sh` (Task 003)

## File Paths

```
docs/reference/competitive_baselines.md
```

## Acceptance Criteria

1. All competitor baselines include source citations (published blog posts, benchmarks, papers)
2. Scenario mapping is 1:1 (each baseline has exactly one TOML scenario)
3. Performance targets are 10-20% better than current best-in-class competitors
4. Measurement methodology is reproducible (another engineer can replicate)
5. Document passes markdown linting (no errors)

## Testing Approach

```bash
# Validate markdown syntax
npx markdownlint-cli2 docs/reference/competitive_baselines.md

# Verify all scenario file references exist
grep -o 'scenarios/competitive/.*\.toml' docs/reference/competitive_baselines.md | while read f; do
  [ -f "$f" ] || echo "Missing: $f"
done

# Check all URLs are accessible (manual spot check)
grep -o 'https://[^ ]*' docs/reference/competitive_baselines.md | head -5 | xargs -n1 curl -I
```

## Integration Points

- Referenced by Task 003 benchmark runner script
- Used by Task 005 integration with quarterly review workflow
- Informs Task 006 initial baseline measurement
