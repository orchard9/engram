# Task 002: Competitive Baseline Reference Documentation (ENHANCED)

**Status**: Pending
**Complexity**: Simple
**Dependencies**: Task 001 (requires scenario definitions)
**Estimated Effort**: 3 hours

## Objective

Create comprehensive reference documentation capturing competitor baselines, Engram's positioning, and performance targets. This document serves as the single source of truth for competitive positioning and will be read by engineers, product managers, and external stakeholders.

## Core Specifications

Create `docs/reference/competitive_baselines.md` following the enhanced structure and guidelines below.

## Document Structure and Content Guidelines

### Document Organization

The document should follow this hierarchical structure for progressive disclosure:

```markdown
# Competitive Performance Baselines

[Executive Summary - 3-4 sentences]

## Quick Reference
[Table of Contents with anchor links]

## Competitor Baseline Summary
[High-level comparison table - scannable in 30 seconds]

## Detailed Baseline Data
[One section per competitor with full context]

## Scenario Mapping
[Links between competitor baselines and TOML scenarios]

## Engram Performance Targets
[Aspirational targets with rationale]

## Competitive Positioning
[Strengths, weaknesses, differentiation]

## Measurement Methodology
[How to reproduce measurements]

## Quarterly Review Process
[Update workflow and ownership]

## References
[All source citations]
```

### Writing Guidelines

#### Executive Summary Pattern

Start with a clear, jargon-free overview:

**Good Example:**
```markdown
This document tracks Engram's performance against leading vector databases
(Qdrant, Milvus, Weaviate) and graph databases (Neo4j). Measurements are updated
quarterly to validate competitive positioning and identify optimization priorities.
All baselines are reproducible using standardized scenarios in `scenarios/competitive/`.
```

**Poor Example:**
```markdown
Engram is a high-performance cognitive architecture with vector, graph, and
temporal capabilities. We benchmark against competitors to ensure we're fast.
```

**Why?** The good example sets expectations (what, why, how often, where to find details) without marketing language. The poor example uses vague claims and doesn't orient the reader.

#### Metric Explanation Pattern

Define technical terms inline the first time they appear:

**Good Example:**
```markdown
**P99 Latency**: The 99th percentile response time, meaning 99% of requests
complete faster than this value. Critical for user-facing applications where
tail latency affects perceived responsiveness.
```

**Poor Example:**
```markdown
**P99**: 99th percentile latency
```

**Why?** Non-experts need context about why metrics matter. The good example teaches while defining.

#### Citation Format

Use consistent, verifiable citations:

**Good Example:**
```markdown
| System | Workload | P99 Latency | Source |
|--------|----------|-------------|--------|
| Qdrant | ANN 1M 768d | 22-24ms | [Qdrant Benchmarks](https://qdrant.tech/benchmarks/) (accessed 2025-11-09) |
| Neo4j | 1-hop traversal | 27.96ms | [Neo4j Performance Guide](https://neo4j.com/developer/graph-data-science/performance/) (v5.14) |
```

**Poor Example:**
```markdown
| System | P99 | Source |
|--------|-----|--------|
| Qdrant | 22ms | Website |
| Neo4j | 28ms | Docs |
```

**Why?** Good citations enable verification and show measurement context (dataset size, version). Poor citations can't be validated.

### Content Templates

#### Template: Competitor Baseline Summary Table

```markdown
## Competitor Baseline Summary

| System | Category | Workload | P99 Latency | Throughput (QPS) | Recall | Dataset | Scenario File | Source |
|--------|----------|----------|-------------|------------------|--------|---------|---------------|--------|
| Qdrant | Vector DB | ANN search | 22-24ms | 626 | 99.5% | 1M 768d | [`qdrant_ann_1m_768d.toml`](../../scenarios/competitive/qdrant_ann_1m_768d.toml) | [Qdrant Benchmarks](https://qdrant.tech/benchmarks/) |
| Milvus | Vector DB | ANN search | 708ms | 2,098 | 100% | 10M | [`milvus_ann_10m_768d.toml`](../../scenarios/competitive/milvus_ann_10m_768d.toml) | [Milvus Benchmarks](https://milvus.io/docs/benchmark.md) |
| Neo4j | Graph DB | 1-hop traversal | 27.96ms | 280 | N/A | Unknown | [`neo4j_traversal_100k.toml`](../../scenarios/competitive/neo4j_traversal_100k.toml) | [Neo4j Performance](https://neo4j.com/developer/graph-data-science/performance/) |
| Weaviate | Vector DB | ANN search | 70-150ms | 200-400 | N/A | 1M | (pending) | [Weaviate Docs](https://weaviate.io/developers/weaviate/benchmarks) |
| Redis | Vector DB | Vector search | 8ms | N/A | N/A | Unknown | (pending) | [Redis Benchmarks](https://redis.io/docs/stack/search/reference/vectors/) |

**Note**: All latencies measured on comparable hardware (modern multi-core CPU, 32GB+ RAM, NVMe SSD).
Dataset notation: "1M 768d" = 1 million vectors, 768 dimensions.
```

**Design Rationale:**
- **Category column**: Helps readers quickly identify relevant comparisons
- **Scenario File links**: One-click navigation to exact test configuration
- **Inline dataset notation**: Avoids separate legend table
- **Pending entries**: Shows roadmap transparency

#### Template: Detailed Competitor Profile

For each competitor, provide context before raw numbers:

```markdown
### Qdrant (Vector Database)

**Overview**: Qdrant is a specialized vector similarity search engine optimized
for high-dimensional embeddings. It uses a custom HNSW implementation with SIMD
acceleration and memory-mapped storage for large datasets.

**Benchmark Configuration**:
- **Hardware**: 8-core Intel Xeon, 32GB RAM, NVMe SSD
- **Dataset**: 1M vectors, 768 dimensions (OpenAI embeddings)
- **Index**: HNSW with M=16, ef_construction=128
- **Search Parameters**: ef_search=64 for 99.5% recall

**Performance Characteristics**:
| Metric | Value | Notes |
|--------|-------|-------|
| P50 Latency | 18ms | Median search time |
| P99 Latency | 22-24ms | Tail latency varies with load |
| Throughput | 626 QPS | Single-threaded client |
| Recall | 99.5% | At ef_search=64 |
| Index Build Time | 45 min | For 1M vectors |

**Engram Comparison**:
- **Advantage (Qdrant)**: Pure vector search is highly optimized
- **Advantage (Engram)**: Supports hybrid vector+graph+temporal queries
- **Target**: Engram should achieve P99 <20ms for pure vector search (10% faster)

**Source**: [Qdrant Benchmarks](https://qdrant.tech/benchmarks/) (accessed 2025-11-09)

**Reproduction**: Run `tools/loadtest run --scenario scenarios/competitive/qdrant_ann_1m_768d.toml`
```

**Design Rationale:**
- **Overview**: Sets context for non-experts
- **Configuration**: Enables apples-to-apples hardware comparison
- **Comparison section**: Clear strengths/weaknesses without hype
- **Reproduction command**: Actionable next step for engineers

#### Template: Performance Targets with Rationale

```markdown
## Engram Performance Targets

Targets represent aspirational goals derived from competitive analysis and
architectural capabilities. Each target includes rationale explaining why the
number is achievable or stretch.

### ANN Search (1M vectors, 768d)

**Target**: P99 < 20ms, Throughput > 650 QPS, Recall > 99.5%

**Rationale**:
- **Why achievable**: Engram's HNSW implementation uses SIMD (like Qdrant) and
  benefits from M17 dual-memory optimizations that reduce pointer chasing
- **Competitive gap**: Current best is Qdrant at 22-24ms; target represents
  10-20% improvement
- **Trade-offs**: Pure vector search may sacrifice some latency for hybrid
  query flexibility

**Dependencies**:
- M17 dual-memory consolidation (complete)
- SIMD vector operations (complete)
- Cache-aware HNSW layout (pending)

**Validation**: Success = 3 consecutive benchmark runs within target, <5% variance

---

### Graph Traversal (100K nodes)

**Target**: P99 < 15ms, Throughput > 300 QPS

**Rationale**:
- **Why achievable**: Neo4j's 27.96ms baseline uses JVM with GC pauses; Engram's
  Rust implementation avoids stop-the-world pauses
- **Competitive gap**: Target represents 46% improvement over Neo4j
- **Trade-offs**: Engram optimizes for probabilistic spreading activation, not
  just deterministic traversal

**Dependencies**:
- Lock-free graph navigation (complete)
- NUMA-aware node placement (complete)
- Edge cache optimization (pending)

**Validation**: Measure on Neo4j-equivalent graph structure (scale-free, avg degree 10)

---

### Hybrid Workload (100K nodes)

**Target**: P99 < 10ms, Throughput > 500 QPS

**Rationale**:
- **No competitor baseline**: Unique to Engram's architecture
- **Target derivation**: 50% of pure vector target + 33% of graph traversal target
- **Strategic importance**: Demonstrates differentiation in market positioning

**Dependencies**:
- Integrated query execution (complete)
- Workload-aware scheduling (pending)

**Validation**: Mix of 30% store, 30% recall, 30% search, 10% pattern completion
```

**Design Rationale:**
- **Why section**: Builds confidence that targets are data-driven, not arbitrary
- **Dependencies**: Shows what's blocking achievement
- **Validation criteria**: Makes success measurable and reproducible

#### Template: Measurement Methodology

```markdown
## Measurement Methodology

All measurements follow this standardized protocol to ensure reproducibility:

### Hardware Specification

**Reference Platform**: Apple M1 Max (10-core CPU, 32GB unified memory)
- **Rationale**: Developer laptop representative of engineering workflow
- **Alternatives**: Results on Intel/AMD x86_64 included where available

**Production Platform**: Linux x86_64 (32-core Xeon, 128GB RAM, NVMe SSD)
- **Rationale**: Matches Qdrant/Neo4j published benchmark environments
- **Usage**: Quarterly review measurements use this platform

### Test Configuration

**Engram Version**: Git commit hash from measurement (e.g., `a1b2c3d`)
- **Build**: `cargo build --release` with LTO enabled
- **Config**: Default settings unless noted in scenario file

**Load Test Parameters**:
- **Duration**: 60 seconds per scenario
- **Warmup**: 10 seconds to stabilize caches and JIT
- **Client**: Single-threaded deterministic client
- **Seed**: Fixed per scenario (qdrant=42, neo4j=43, etc.)
- **Output**: JSON with full histogram data

### Execution Protocol

```bash
# 1. Clean environment (no background processes)
pkill -9 engram || true
sleep 5

# 2. Start Engram with diagnostics
engram start --metrics-port 9090 &
ENGRAM_PID=$!

# 3. Warmup (10s)
tools/loadtest run --scenario $SCENARIO --duration 10 --warmup

# 4. Measurement (60s)
tools/loadtest run --scenario $SCENARIO --duration 60 --output results.json

# 5. Collect diagnostics
./scripts/engram_diagnostics.sh >> diagnostics.log

# 6. Shutdown
kill $ENGRAM_PID
```

### Data Collection

**Metrics Captured**:
- **Latency**: Full HdrHistogram (P50, P90, P95, P99, P99.9, max)
- **Throughput**: Operations per second (mean, min, max)
- **System**: CPU usage, memory RSS, disk I/O (via `time -v`)
- **Engram**: Cache hit rate, consolidation events, decay calculations

**Data Storage**:
- **Raw results**: `tmp/competitive_baselines/{scenario}_{date}.json`
- **Historical archive**: `benchmarks/competitive_history/` (git-tracked)

### Reproducibility Checklist

Before accepting a measurement as valid:

- [ ] Three consecutive runs show <5% P99 variance
- [ ] No system alerts during test window (check `dmesg`)
- [ ] CPU throttling disabled (`cpufreq-info` shows max frequency)
- [ ] Swap usage = 0 (check `free -h`)
- [ ] No competing processes (verify with `top`)
- [ ] Diagnostics log shows no errors

If any check fails, discard measurement and investigate.
```

**Design Rationale:**
- **Two platforms**: Balances developer accessibility with production relevance
- **Execution protocol**: Shell script format = copy-paste reproducible
- **Checklist**: Prevents invalid measurements from polluting baselines

### Visual Hierarchy Best Practices

#### Use Tables for Comparison

When readers need to scan multiple data points:

```markdown
| Competitor | Category | Strength | Weakness | Engram Positioning |
|------------|----------|----------|----------|--------------------|
| Qdrant | Vector | Pure ANN speed | No graph traversal | Hybrid queries |
| Neo4j | Graph | Rich query lang | Slow vector ops | Probabilistic spreading |
| Milvus | Vector | Massive scale | High latency | Balanced latency/scale |
```

#### Use Lists for Sequential Steps

When readers need to follow a process:

```markdown
## Quarterly Review Workflow

1. **Week 1 Monday**: Performance lead pulls latest `main` branch
2. **Week 1 Tuesday**: Run full competitive benchmark suite (4 hours)
3. **Week 1 Wednesday**: Analyze results, identify regressions/improvements
4. **Week 1 Thursday**: Update `competitive_baselines.md` with new data
5. **Week 1 Friday**: Present findings to eng team, create optimization tasks
```

#### Use Callout Blocks for Important Context

When readers need to notice critical information:

```markdown
> **Important**: All competitor baselines are external measurements from published
> sources. Engram cannot verify exact hardware configurations or workload
> parameters. Use these baselines as directional guidance, not absolute truth.
```

## Integration with Diátaxis Framework

Position this document correctly within the documentation taxonomy:

**Document Type**: Reference
- **Purpose**: Provide factual information for lookup
- **Audience**: All skill levels (engineers, PMs, external evaluators)
- **Structure**: Topic-based organization (by competitor, by metric)
- **Tone**: Objective, verifiable, neutral

**Cross-References to Other Documentation**:

```markdown
## Related Documentation

- **Tutorials**: [Running Your First Benchmark](../../tutorials/first-benchmark.md) -
  Learn how to execute competitive scenarios step-by-step
- **How-To Guides**: [Optimize Vector Search Performance](../../howto/optimize-vector-search.md) -
  Practical steps to close competitive gaps
- **Explanation**: [Why Engram Uses Probabilistic Memory](../../explanation/probabilistic-memory.md) -
  Understanding architectural trade-offs vs deterministic systems
- **Operations**: [Quarterly Performance Review](../../operations/quarterly-review.md) -
  Production workflow for updating baselines
```

**Navigation Aids**:

```markdown
## Quick Navigation

- [Jump to Qdrant Baseline](#qdrant-vector-database)
- [Jump to Neo4j Baseline](#neo4j-graph-database)
- [Jump to Performance Targets](#engram-performance-targets)
- [Jump to Measurement Methodology](#measurement-methodology)
- [Download Raw Data](../../benchmarks/competitive_history/)
```

## Quarterly Update Workflow

### Ownership and Cadence

```markdown
## Quarterly Review Process

**Schedule**: First week of January, April, July, October

**Owner**: Performance Engineering Lead
- **Primary**: @performance-lead (GitHub handle)
- **Backup**: @systems-architect
- **Stakeholders**: Eng team, product manager

**Time Commitment**: ~8 hours per quarter
- 4 hours: Run benchmark suite + validate results
- 2 hours: Analyze trends, identify optimization opportunities
- 1 hour: Update documentation
- 1 hour: Present findings to team

**Deliverables**:
1. Updated `docs/reference/competitive_baselines.md` (this file)
2. Raw measurement data in `benchmarks/competitive_history/YYYY-QN/`
3. Comparison report: `benchmarks/competitive_history/YYYY-QN/report.md`
4. GitHub issues for identified optimization tasks (labeled `performance`)
```

### Update Process Template

```markdown
### Quarterly Update Template

Copy this template into the Updates section below for each quarter:

---

#### Q4 2025 Update (2025-10-15)

**Measurement Details**:
- **Engram Version**: v0.5.2 (commit `abc123d`)
- **Platform**: Linux x86_64, 32-core Xeon Gold 6258R, 128GB RAM
- **Scenarios**: qdrant_ann_1m_768d, neo4j_traversal_100k, hybrid_production_100k
- **Duration**: 60s per scenario, 3 runs averaged

**Key Changes Since Q3**:
- Qdrant baseline updated: 22-24ms → 20-22ms (index optimization in v1.7)
- Engram improved: ANN search P99 26ms → 23ms (M18 cache improvements)
- Neo4j baseline stable: 27.96ms (no new release)

**Competitive Gaps**:
1. **ANN Search**: Engram now within 10% of Qdrant (was 15% slower in Q3)
2. **Graph Traversal**: Engram 18ms vs Neo4j 27.96ms (maintaining 35% advantage)
3. **Hybrid Workload**: No competitor baseline (unique capability)

**Optimization Priorities for Q1 2026**:
1. Close remaining 10% ANN gap → Target: 20ms (task #234)
2. Improve hybrid workload throughput 500→600 QPS (task #235)
3. Validate Milvus 10M scenario on new hardware (task #236)

**Artifacts**:
- Raw data: [`benchmarks/competitive_history/2025-Q4/`](../../benchmarks/competitive_history/2025-Q4/)
- Comparison report: [Q4 2025 Report](../../benchmarks/competitive_history/2025-Q4/report.md)
- Presentation slides: [Q4 Review.pdf](../../benchmarks/competitive_history/2025-Q4/review.pdf)

---
```

### Version Control Strategy

Track changes over time using git and structured data:

```markdown
## Historical Trends

### Engram vs Qdrant ANN Search (P99 Latency)

| Quarter | Qdrant | Engram | Gap | Change |
|---------|--------|--------|-----|--------|
| Q4 2025 | 22ms | 23ms | +5% | Improved from +15% |
| Q3 2025 | 24ms | 28ms | +15% | Baseline measurement |

### Engram vs Neo4j Graph Traversal (P99 Latency)

| Quarter | Neo4j | Engram | Gap | Change |
|---------|-------|--------|-----|--------|
| Q4 2025 | 27.96ms | 18ms | -35% | Stable |
| Q3 2025 | 27.96ms | 18ms | -35% | Baseline measurement |

**Visualization**: See [`benchmarks/competitive_history/trends.png`](../../benchmarks/competitive_history/trends.png)
for line chart of quarterly progress.
```

## Maintenance Guidelines

### When to Update Outside Quarterly Schedule

Immediate updates required when:

1. **Competitor releases major version** with claimed performance improvements >20%
   - Action: Re-run affected scenario, update baseline with version note
   - Example: "Qdrant v2.0 claims 30% latency reduction"

2. **Engram optimization achieves >10% improvement** on competitive scenario
   - Action: Update Engram measurement, note milestone achievement
   - Example: "M18 Task 005 improved ANN search by 12%"

3. **Methodology changes** (new hardware, different measurement tool)
   - Action: Re-baseline ALL competitors for consistency
   - Example: "Migrated from M1 Max to Xeon platform"

4. **External source updates** (competitor publishes new benchmarks)
   - Action: Update citation and values, note source date change
   - Example: "Milvus benchmark updated 2025-11-01"

### Deprecation Policy

Remove competitor baselines when:

1. Product discontinued or unmaintained >2 years
2. No public benchmarks available (can't verify)
3. Architectural changes make comparison invalid

Deprecated entries move to `docs/reference/competitive_baselines_archived.md`.

## File Paths

```
docs/reference/competitive_baselines.md            # Main document (this task)
benchmarks/competitive_history/                    # Quarterly raw data (git-tracked)
benchmarks/competitive_history/YYYY-QN/            # Per-quarter directory
benchmarks/competitive_history/YYYY-QN/report.md   # Quarterly analysis
benchmarks/competitive_history/trends.png          # Historical visualization
```

## Acceptance Criteria

### Content Quality

1. **Citations**: All competitor baselines include source URLs with access dates
2. **Scenario Mapping**: Each baseline links to exactly one TOML file in `scenarios/competitive/`
3. **Targets**: All performance targets include rationale and dependency list
4. **Methodology**: Reproduction steps enable another engineer to replicate measurements

### Document Structure

5. **Navigation**: Table of contents with anchor links to all major sections
6. **Progressive Disclosure**: Executive summary → tables → detailed profiles
7. **Consistent Formatting**: All tables use same column order, all code blocks use bash syntax
8. **Cross-References**: Links to related tutorials, how-tos, explanations validated

### Technical Quality

9. **Markdown Linting**: Passes `npx markdownlint-cli2` with no errors
10. **Link Validation**: All internal links resolve correctly (test with `markdown-link-check`)
11. **Scenario Files Exist**: All referenced TOML files exist in `scenarios/competitive/`

### Usability

12. **Scannability**: Key information (targets, gaps) readable in <2 minutes
13. **Actionability**: Each section ends with "Next Steps" or "Reproduction" command
14. **Teachability**: Technical terms defined inline for non-expert readers

## Testing Approach

```bash
# Validate markdown syntax
npx markdownlint-cli2 docs/reference/competitive_baselines.md

# Verify all scenario file references exist
echo "Checking scenario file links..."
grep -o 'scenarios/competitive/.*\.toml' docs/reference/competitive_baselines.md | while read f; do
  if [ -f "$f" ]; then
    echo "  ✓ $f"
  else
    echo "  ✗ MISSING: $f"
    exit 1
  fi
done

# Check all external URLs are accessible (sample check)
echo "Validating external citations (first 5)..."
grep -o 'https://[^ ]*' docs/reference/competitive_baselines.md | head -5 | while read url; do
  if curl --head --silent --fail "$url" > /dev/null; then
    echo "  ✓ $url"
  else
    echo "  ✗ UNREACHABLE: $url"
  fi
done

# Validate internal anchor links
echo "Checking table of contents links..."
npx markdown-link-check docs/reference/competitive_baselines.md --config .markdown-link-check.json

# Check document length (target: 800-1200 lines for reference docs)
line_count=$(wc -l < docs/reference/competitive_baselines.md)
echo "Document length: $line_count lines"
if [ $line_count -lt 800 ] || [ $line_count -gt 1500 ]; then
  echo "  ⚠ Warning: Document may be too short (<800) or too long (>1500)"
fi

# Verify all code blocks have language annotations
if grep -n '^```$' docs/reference/competitive_baselines.md; then
  echo "  ✗ FAIL: Found code blocks without language annotations"
  exit 1
else
  echo "  ✓ All code blocks annotated"
fi

echo "All validation checks passed!"
```

## Examples: Good vs Poor Documentation

### Example 1: Performance Target

**Good**:
```markdown
### ANN Search Target: P99 < 20ms

**Current Best**: Qdrant at 22-24ms (1M vectors, HNSW ef_search=64)

**Why Achievable**:
- Engram's SIMD cosine similarity matches Qdrant's throughput (1.6M vectors/sec)
- M17 dual-memory reduces cache misses by 15% (measured in `benchmarks/m17/`)
- Rust's zero-cost abstractions eliminate JVM GC pauses (Neo4j bottleneck)

**Dependencies**: HNSW cache-aware layout (M18 Task 003, 2 weeks estimated)

**Validation**: Run `scenarios/competitive/qdrant_ann_1m_768d.toml` 3x, average P99
```

**Poor**:
```markdown
### ANN Search: <20ms

We should be faster than Qdrant because Rust is fast and we have good optimizations.
Target is 20ms because it's a round number.
```

**Why Good Wins**:
- Specific competitor benchmark identified
- Technical rationale with data (SIMD, cache misses, GC)
- Clear dependencies and validation criteria
- Actionable command for verification

### Example 2: Competitor Baseline

**Good**:
```markdown
### Neo4j (Graph Database)

Neo4j is an ACID-compliant native graph database written in Java. It uses a
property graph model with Cypher query language. Performance is bounded by
JVM garbage collection pauses and disk-based transaction logs.

**Benchmark Configuration**:
- Version: Neo4j 5.14 Enterprise Edition
- Hardware: 16-core Xeon Silver, 64GB RAM, SSD
- Dataset: 100K nodes, 1M edges (scale-free distribution)
- Query: Single-hop MATCH traversal with property filter

**Measured Performance**:
- P99 Latency: 27.96ms (includes GC pauses)
- Throughput: 280 QPS (single client)
- Source: [Neo4j Performance Guide](https://neo4j.com/docs/v5.14/performance/) (v5.14 docs)

**Engram Advantage**: Lock-free graph navigation avoids GC pauses
**Neo4j Advantage**: Mature Cypher ecosystem and query optimization
```

**Poor**:
```markdown
### Neo4j

Neo4j is a graph database. It's slow at graph traversal (28ms).

Source: Neo4j website
```

**Why Good Wins**:
- Context for readers unfamiliar with Neo4j
- Specific version and hardware configuration
- Balanced view of strengths/weaknesses
- Verifiable source citation

## Integration Points

- **Task 001**: Scenario TOML files provide configuration details for baselines
- **Task 003**: Benchmark runner script executes measurements documented here
- **Task 004**: Report generator formats raw data into quarterly update template
- **Task 005**: Quarterly workflow script automates update process
- **Task 006**: Initial baseline measurement populates first version of document

## Success Metrics

Document is complete when:

1. **Completeness**: All sections from structure template present
2. **Quality**: All acceptance criteria pass (14 checkboxes)
3. **Usability**: Engineer unfamiliar with M17.1 can reproduce measurements in <30 min
4. **Maintainability**: Quarterly update takes <2 hours using provided templates
5. **Accuracy**: All competitor data traceable to public sources with access dates

## Notes for Implementation

### Start with Template Structure

Begin by creating the document skeleton with all major sections:

```bash
# Create initial structure
cat > docs/reference/competitive_baselines.md <<'EOF'
# Competitive Performance Baselines

[Executive Summary - TODO]

## Quick Reference
[TOC - TODO]

## Competitor Baseline Summary
[Table - TODO]

## Detailed Baseline Data
### Qdrant (Vector Database)
[TODO]

### Neo4j (Graph Database)
[TODO]

...

EOF
```

Then fill each section incrementally, testing with markdown linter after each section.

### Use Real Data from Task 001

Once scenarios exist, run them to get actual Engram numbers:

```bash
# Measure Engram's current performance on competitive scenarios
for scenario in scenarios/competitive/*.toml; do
  tools/loadtest run --scenario "$scenario" --duration 60 --output "tmp/$(basename $scenario .toml).json"
done

# Extract P99 latencies to populate "Engram Comparison" sections
jq '.p99_latency_ms' tmp/*.json
```

### Review Against Reference Examples

Before marking task complete, compare your document against:

1. `docs/reference/performance-baselines.md` - Structure and table formatting
2. `docs/reference/benchmark-results.md` - Citation style and methodology
3. `docs/reference/system-requirements.md` - Progressive disclosure and clarity

### Get Peer Review

Ask another engineer to:

1. Follow reproduction steps without assistance
2. Identify any jargon that needs definition
3. Verify all external links are accessible
4. Check for consistent terminology (e.g., "P99 latency" vs "99th percentile")

## Related Task Enhancements

If during implementation you identify gaps in this specification:

1. **Missing competitor**: Create follow-up task "Add [System] to competitive baselines"
2. **Automation opportunity**: Create task "Automate quarterly baseline collection"
3. **Visualization need**: Create task "Generate trend charts for competitive history"

Do not expand scope of this task. Keep it focused on documentation quality.
