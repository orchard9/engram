# Observability for Semantic Search: Building Production Monitoring That Actually Works

## The Problem: Traditional Metrics Miss What Matters

You've shipped a semantic search system. Users query in Spanish, results come back in English. Synonyms expand queries automatically. Metaphorical language gets interpreted, not just matched. It's working.

Until it isn't.

Your standard monitoring dashboard shows healthy metrics: p95 latency at 80ms, error rate at 0.02%, QPS steady at 5,000. Everything looks green. But user complaints are spiking. "Search stopped understanding my questions." "Results used to be relevant, now they're garbage." "French queries broke last week."

The traditional RED method (Rate, Errors, Duration) can't see semantic failures. A query that returns 10 results in 50ms looks successful to your metrics, even if all 10 results are semantically irrelevant. You need observability that measures semantic correctness, not just operational health.

This is the challenge we faced building Engram, a graph-based memory system with multilingual semantic recall. Here's how we solved it.

## Why Semantic Systems Need Different Observability

Traditional search has clear success metrics: did the query match the document? Yes or no. Binary. Deterministic.

Semantic search is probabilistic:
- A query for "debugging" might match documents about "troubleshooting," "fixing errors," or even "detective work" (metaphorical)
- Multilingual retrieval means "perro" (Spanish) should activate memories encoded as "dog" (English) or "chien" (French)
- Confidence scores replace binary matches: 0.92 is better than 0.67, but is 0.74 good enough?

Your observability must answer:
1. Is semantic expansion helping or hurting result quality?
2. Are all supported languages performing equally?
3. Is figurative language interpretation working correctly?
4. Are we getting worse over time (model drift)?

None of these questions can be answered with traditional uptime monitoring.

## The Architecture: Multi-Tier Semantic Tracing

Our solution extends distributed tracing with semantic-specific context. Standard OpenTelemetry traces show request flow across services. We augment spans with semantic attributes:

```rust
// Semantic span attributes
span.set_attribute("semantic.query.language", "es");
span.set_attribute("semantic.expansion.applied", true);
span.set_attribute("semantic.figurative.detected", false);
span.set_attribute("semantic.tier.hits", vec!["hot", "warm"]);
span.set_attribute("semantic.confidence.p95", 0.83);
```

This enables tier-aware performance analysis. Semantic search in Engram spans three storage tiers:
- **Hot tier** (RAM-backed HNSW index): <10ms target latency
- **Warm tier** (SSD vector store): <100ms target latency
- **Cold tier** (S3 + on-demand embedding): <1s target latency

Each tier has different performance characteristics and semantic quality profiles. Our tracing reveals:
- Which tiers contribute results for each query type
- Whether cross-tier results cluster semantically (coherence check)
- Tier-specific performance degradation

A query trace might show: "Hot tier returned 3 results (8ms, avg confidence 0.91), warm tier added 7 results (47ms, avg confidence 0.78)." Now you can diagnose tier skew: warm tier is slower AND lower quality.

## Evaluation Harness: Beyond nDCG

Information retrieval researchers love nDCG (Normalized Discounted Cumulative Gain). It's ranking-aware and handles graded relevance. We use it too, but it's not enough.

Our benchmark suite adds semantic-specific metrics:

**Cross-Lingual Consistency**:
- Does translating a query change result ranking significantly?
- Measure: Spearman rank correlation between query variants
- Alert threshold: correlation <0.8 indicates language-dependent bugs

**Expansion Effectiveness**:
- Delta nDCG: improvement from semantic expansion vs literal-only search
- Coverage ratio: fraction of results from expanded terms vs original terms
- Semantic drift: cosine similarity between original query embedding and result centroids (catches over-expansion)

**Confidence Calibration**:
- Expected Calibration Error (ECE): gap between predicted confidence and observed relevance
- Reliability diagrams: plot predicted vs actual correctness across confidence bins
- Uncalibrated models say "0.95 confidence" for results that are only 60% relevant

Example from our benchmark corpus:

```
Query: "debugging concurrent code" (English)
Literal search: nDCG@10 = 0.71
Semantic expansion: nDCG@10 = 0.89 (added "race condition", "thread safety")
Delta nDCG: +0.18 (expansion helped)

Query: "depurar codigo concurrente" (Spanish, same semantic meaning)
Literal search: nDCG@10 = 0.34 (few Spanish docs in corpus)
Cross-lingual + expansion: nDCG@10 = 0.86 (retrieved English docs)
Delta nDCG: +0.52 (multilingual critical here)
```

If Spanish queries suddenly drop below 0.75 nDCG while English stays high, we've detected a regression in the multilingual pipeline.

## Streaming Diagnostics: SSE Events with Semantic Context

Engram streams query results via Server-Sent Events (SSE). Users get progressive results as the search executes. We piggyback semantic diagnostics onto this stream:

```json
{
  "event": "semantic_query",
  "query_id": "uuid",
  "original_text": "debugging concurrent code",
  "detected_language": "en",
  "expansion_applied": true,
  "expansion_terms": ["race condition", "thread safety", "mutex"],
  "figurative_detected": false,
  "top_results": [
    {"id": "doc123", "confidence": 0.92, "tier": "hot"},
    {"id": "doc456", "confidence": 0.87, "tier": "warm"}
  ],
  "metrics": {
    "expansion_latency_ms": 12,
    "embedding_latency_ms": 45,
    "total_latency_ms": 234,
    "tiers_accessed": ["hot", "warm"]
  }
}
```

This serves two audiences:

**Operators**: Aggregate these events for real-time dashboards. Plot expansion_latency over time to detect GPU saturation. Count figurative_detected to validate metaphor interpretation is triggering.

**Developers**: Debug individual queries by inspecting expansion decisions. "Why did this query fail?" -> Check if expansion_terms drifted semantically. "Why is this slow?" -> See tiers_accessed hit cold storage.

Backward compatibility is critical. We version the schema and make new fields optional. Old clients ignore semantic diagnostics, new clients opt in.

## Regression Detection: Statistical Process Control

CI pipelines usually check "does the code compile" and "do tests pass." For semantic systems, you also need "did quality regress?"

We run nightly benchmark sweeps on main branch and store results in a time-series database. Alert conditions use statistical process control:

**CUSUM (Cumulative Sum Control Chart)**: Detects gradual drift better than fixed thresholds
- Track cumulative deviation from baseline nDCG
- Alert when cumulative sum exceeds threshold (suggests sustained degradation)
- More sensitive to "quality slowly declining over weeks" than one-off anomalies

**EWMA (Exponentially Weighted Moving Average)**: Time-weighted quality tracking
- Recent benchmarks matter more than old baselines
- Lambda parameter balances responsiveness (catch regressions fast) vs stability (ignore noise)

Example alert rule:
```
If nDCG@10 drops >5% below 7-day EWMA for >3 consecutive runs:
  Severity: P1 (page on-call)
  Runbook: docs/operations/semantic_alerts.md#ndcg-regression
```

PR-level smoke tests run a smaller benchmark subset (5 min runtime) on representative queries. Fast feedback loop without full nightly suite cost.

## NUMA-Aware Benchmarking: Controlling Variance

Early on, our benchmarks were flaky. Same code, same dataset, nDCG varied by 3% run-to-run. Unacceptable for regression detection.

Root cause: CPU frequency scaling and NUMA effects. When CPUs throttle or cache locality varies, HNSW search performance fluctuates.

Fixes:
- Pin benchmark process to specific CPU cores with `numactl`
- Disable frequency scaling during CI runs
- Warm caches before timing measurements (throw away first run)
- Run each benchmark 10 times, report confidence intervals
- Use paired t-tests when comparing versions (accounts for variance)

Now variance is <0.5%. We can confidently detect real regressions.

## Operator Dashboards: Hierarchical Detail

Grafana dashboards for semantic search must balance overview and drill-down.

**Top panel**: Aggregate RED metrics across all queries
- Queries per second (split by: literal, expanded, figurative)
- Error rate (split by: embedding failures, timeout, invalid language)
- p50/p95/p99 latency (split by tier: hot, warm, cold)

**Middle panel**: Per-language breakdown
- Heatmap: nDCG for each query language x result document language pair
- Time series: trend lines for expansion effectiveness (delta nDCG)
- Histogram: confidence score distributions

**Bottom panel**: Individual query traces
- Trace viewer with semantic annotations (expansion terms, tiers accessed)
- Correlated events: related queries with similar patterns
- Suggested actions: "High latency -> check GPU utilization panel"

Dashboard design principle: **hierarchical detail**. Start broad (system healthy?), drill down (which language broken?), isolate (show me the exact query).

## Alert Design: Actionable, Not Noisy

Bad alerts: "nDCG dropped." Okay, what do I do?

Good alerts: "Spanish query nDCG dropped 8% in last hour. Likely cause: embedding service GPU OOM (see correlated alert). Mitigation: restart embedding workers or route Spanish queries to backup model."

Our alert rules are tiered by severity:

**P0 (page immediately)**:
- nDCG drops >10% below baseline
- Embedding service completely down
- Any language returns zero results

**P1 (page during business hours)**:
- nDCG drops 5-10%
- Latency >2x expected for any tier
- Expansion effectiveness negative (expansion hurting quality)

**P2 (ticket, investigate next day)**:
- Elevated error rates (but queries succeeding)
- Cache eviction rate spiking (performance warning)
- Single-language underperformance (others still healthy)

Each alert links to a runbook with:
- Recent correlated alerts (GPU + latency = resource issue)
- Diagnostic queries (check embedding cache hit rate)
- Mitigation steps (rollback, failover, capacity increase)

## Privacy and Fairness: Not Just Performance

Observability for semantic systems has ethical dimensions.

**Dataset licensing**: Our benchmark corpus uses only CC-BY or CC0 data (Wikipedia, Common Crawl). No user-generated content without consent. Provenance documented for every evaluation example.

**Anonymization**: Queries logged for debugging are scrubbed of PII before archival. Retention policy: 7 days detailed logs, 90 days aggregated metrics, permanent summary statistics only.

**Fairness auditing**: Track nDCG gap between high-resource (English, Spanish) and low-resource (Bengali, Swahili) languages. Alert if any language underperforms by >15% relative to English. Sets a floor on cross-lingual equity.

**Bias detection**: Benchmark queries cover diverse domains (technical, conversational, literary) and multiple demographic groups. Prevents optimizing for narrow use cases.

## What We Learned

**1. Semantic observability is not optional**. You cannot debug "search got worse" with QPS and error rate. You need quality metrics in production.

**2. Multi-tier systems need tier-aware traces**. Aggregate latency hides which tier is slow. Semantic coherence across tiers catches data quality issues.

**3. Statistical process control beats fixed thresholds**. CUSUM and EWMA detect gradual drift that simple "nDCG <0.75" alerts miss.

**4. Benchmark stability requires discipline**. Control CPU frequency, NUMA placement, cache warming. Otherwise variance drowns signal.

**5. Operator UX matters**. Dashboards must guide from "something's wrong" to "here's the broken component and how to fix it." Hierarchical detail and actionable alerts are table stakes.

## Open Challenges

We haven't solved everything. Active areas:

**Figurative language coverage**: What fraction of real-world queries benefit from metaphor interpretation? Our benchmark has synthetic examples, but production usage patterns are unclear.

**Metric composability**: nDCG measures ranking quality, ECE measures calibration, expansion effectiveness measures delta from baseline. How do you aggregate these into a single "semantic health score"?

**Real-time adaptation**: Right now we alert on regressions after they happen. Can we detect drift from query logs before quality degrades? Anomaly detection on expansion term distributions?

**Cross-lingual fairness**: We measure performance gaps between languages, but how do you set fair targets? Should low-resource languages aim for parity with English (unrealistic?) or just "good enough for users"?

## Conclusion: Ship Semantic Systems with Confidence

Semantic search is powerful but fragile. Models drift, languages evolve, edge cases emerge. You can't ship production systems on hope.

Observability for semantic search requires:
- Semantic-specific tracing attributes (not just latency)
- Evaluation harnesses with multilingual, calibration, and expansion metrics
- Statistical regression detection (CUSUM, EWMA)
- Tier-aware performance budgets
- Hierarchical dashboards guiding operators from overview to root cause
- Fairness auditing across languages and use cases

At Engram, these patterns let us ship multilingual semantic recall with measurable quality guarantees. When regressions occur, we detect them in CI or production dashboards before users complain. When debugging, traces tell us exactly which semantic component failed.

Your semantic system deserves better than "hope it works." Build observability that measures what matters.

---

**About Engram**: An open-source graph-based memory system with biologically-inspired consolidation, multilingual semantic recall, and probabilistic spreading activation. Built in Rust for production workloads.

**Further Reading**:
- Our evaluation methodology: docs/operations/semantic_alerts.md (in repo)
- MIRACL multilingual benchmark: Zhang et al., NeurIPS 2023
- Calibration in neural networks: Guo et al., ICML 2017
- SRE best practices (RED/USE methods): Beyer et al., "Site Reliability Engineering," O'Reilly 2016
