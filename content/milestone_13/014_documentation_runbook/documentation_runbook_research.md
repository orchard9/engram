# Documentation and Runbook: Research and Technical Foundation

## Documentation Strategy

Milestone 13 adds complex cognitive functionality. Documentation must serve multiple audiences:

**Audience 1: Cognitive Scientists**
Need to understand biological plausibility, empirical validation, and how to interpret psychology metrics. Citations to peer-reviewed research essential.

**Audience 2: Systems Engineers**
Need to understand performance characteristics, configuration options, observability integration, and operational troubleshooting.

**Audience 3: Application Developers**
Need to understand APIs, when to use different priming types, how to configure interference detection, and best practices.

## Documentation Structure

Following Diátaxis framework (per CLAUDE.md):

### Tutorials (Learning-Oriented)

**Tutorial 1: Validating DRM False Memory**
Step-by-step guide to running DRM paradigm validation, interpreting results, tuning parameters to match Roediger & McDermott (1995).

**Tutorial 2: Configuring Semantic Priming**
How to set up semantic priming, measure effects, validate against Neely (1977), and tune spreading parameters.

**Tutorial 3: Setting Up Observability**
Installing Prometheus + Grafana, importing dashboards, interpreting cognitive metrics, setting up alerts.

### How-To Guides (Problem-Solving)

**How-To 1: Debug Low DRM False Recall Rate**
- Check semantic edge weights
- Verify spreading activation depth
- Inspect critical lure selection
- Tune activation threshold
- Compare to control condition

**How-To 2: Reduce Interference False Positives**
- Adjust similarity threshold
- Modify interference time window
- Check for over-aggressive competition
- Validate against published data

**How-To 3: Optimize Priming Performance**
- Cache similarity computations
- Adjust decay computation frequency
- Tune hash table sizes
- Profile with perf
- Check for false sharing

### Explanation (Understanding)

**Explanation 1: Why Reconsolidation Boundary Conditions Matter**
Deep dive into Nader et al. (2000) findings, biological mechanisms, computational implications, why we can't reconsolidate every retrieval.

**Explanation 2: The Three Priming Types and Their Neural Substrates**
Semantic (temporal cortex), repetition (sensory cortices), associative (hippocampus), why they're independent, how they combine.

**Explanation 3: Interference as Competition vs Memory Modification**
Traditional retrieval competition models vs reconsolidation-based modification, when each occurs, implications for system design.

### Reference (Information)

**Reference 1: API Documentation**
```rust
/// Encodes a memory with interference tracking and reconsolidation checking.
///
/// # Arguments
/// * `node_id` - The concept to encode
/// * `associations` - Related concepts and their strengths
///
/// # Cognitive Effects
/// - May trigger proactive interference detection (Anderson 1974)
/// - Updates repetition priming traces (Jacoby & Dallas 1981)
/// - Checks reconsolidation boundary conditions (Nader et al. 2000)
///
/// # Performance
/// - Typical latency: 30-50μs
/// - p95 latency: < 75μs
/// - Throughput: 5K+ ops/sec
///
/// # Examples
/// ```rust
/// memory.encode(doctor, vec![(nurse, 0.8), (hospital, 0.6)]);
/// ```
pub fn encode(&mut self, node_id: NodeId, associations: Vec<(NodeId, f32)>);
```

**Reference 2: Configuration Parameters**
Complete reference of all tunable parameters with:
- Default value
- Empirical basis (which study)
- Acceptable range
- Performance impact
- When to tune

**Reference 3: Metrics Catalog**
Every metric with:
- Name and type
- What it measures
- Expected range (from published data)
- How to interpret
- Related alerts

## Operational Runbook

### Runbook Section 1: Cognitive Pattern Health Checks

**Check 1: DRM False Recall Rate**
```bash
# Query Prometheus
curl 'http://localhost:9090/api/v1/query?query=engram_drm_false_recall_rate'

# Expected: 0.55-0.65
# If <0.45: spreading activation too weak or critical lure selection poor
# If >0.75: spreading too aggressive or threshold too low
```

**Remediation:**
- Check semantic edge weights in graph
- Validate critical lure selection algorithm
- Tune activation spreading depth
- Compare to control condition (random words)

**Check 2: Priming Effect Magnitude**
```bash
# Semantic priming median boost
curl 'http://localhost:9090/api/v1/query?query=histogram_quantile(0.5,engram_priming_boost_magnitude{priming_type="semantic"})'

# Expected: 0.30-0.50 (Neely 1977)
# If <0.20: spreading too conservative or decay too fast
# If >0.60: spreading too aggressive or decay too slow
```

### Runbook Section 2: Performance Troubleshooting

**Problem: High Retrieval Latency**

Diagnostic steps:
```bash
# Check p95 latency by operation
curl 'http://localhost:9090/api/v1/query?query=histogram_quantile(0.95,engram_retrieval_latency_microseconds)'

# Profile with perf
perf record -F 999 -g ./target/release/engram-server
perf report

# Check for specific hotspots:
# - Priming boost computation
# - Interference detection
# - Similarity cache misses
# - Lock contention (should be rare)
```

Remediation:
- Increase similarity cache size
- Adjust priming decay computation frequency
- Check for false sharing in atomic metrics
- Reduce interference detection window

**Problem: Memory Growth**

Diagnostic steps:
```bash
# Track memory by subsystem
curl 'http://localhost:9090/api/v1/query?query=engram_memory_bytes{subsystem="priming"}'

# Profile allocations
heaptrack ./target/release/engram-server
```

Common causes:
- Repetition priming traces not being evicted
- Co-occurrence matrix growing unbounded
- Reconsolidation window cleanup not running
- Metrics buffers not flushing

### Runbook Section 3: Validation Failures

**Validation Failure: Interference Tests**

```bash
# Run interference validation suite
cargo test --features=all interference_validation

# If failing, check specific conditions:
cargo test --features=all test_proactive_high_similarity --verbose

# Inspect detailed output:
# - Actual interference magnitude
# - Expected range
# - Similarity distribution
# - Competitor activation levels
```

**Validation Failure: Spacing Effect**

```bash
# Run spacing effect validation
cargo test --features=all spacing_effect_validation

# Long-running test, check progress:
tail -f spacing_validation.log

# If failing:
# - Check consolidation time windows
# - Verify STM→LTM transfer timing
# - Inspect repetition interval distributions
# - Compare to Cepeda et al. (2006) curves
```

## Documentation Maintenance

**Process:**
1. Every new cognitive feature includes documentation
2. Every parameter change updates reference docs
3. Every validation test includes interpretation guide
4. Every performance optimization updates benchmarks

**Review:**
- Technical accuracy reviewed by memory-systems-researcher agent
- Performance claims verified by benchmarks
- Citations checked against original papers
- Runbook tested on clean system

## Success Criteria

Documentation complete when:
- All tutorials executable without errors
- All how-to guides solve stated problems
- All explanations cite primary sources
- All reference docs match implementation
- Runbook covers all common issues
- Validation guides enable independent replication
