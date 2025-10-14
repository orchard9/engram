# Twitter Thread: Semantic Search Observability

## Thread

1/12

You shipped semantic search. Users query in Spanish, get English results. Synonyms expand automatically. Metaphors get interpreted.

Your dashboard: p95 latency 80ms, errors 0.02%, QPS 5k. All green.

User complaints: "Search is broken."

Traditional monitoring can't see semantic failures.

---

2/12

A query returning 10 results in 50ms looks successful to RED metrics (Rate, Errors, Duration).

But what if all 10 results are semantically irrelevant?

What if French queries work but Spanish don't?

What if your expansion terms drifted and now "debugging" matches "eating bugs"?

---

3/12

Building Engram (graph-based memory system), we faced this exact problem.

Our solution: semantic-specific observability spanning evaluation, tracing, and regression detection.

Here's the architecture we built for production multilingual semantic recall.

---

4/12

KEY INSIGHT 1: Extend distributed tracing with semantic attributes

Standard OpenTelemetry shows request flow. We add:
- semantic.query.language: "es"
- semantic.expansion.applied: true
- semantic.tier.hits: ["hot", "warm"]
- semantic.confidence.p95: 0.83

Now traces reveal WHAT failed, not just THAT something failed.

---

5/12

Our system has 3 storage tiers:
- Hot (RAM HNSW): <10ms, high confidence
- Warm (SSD vectors): <100ms, medium confidence
- Cold (S3 + embedding): <1s, variable confidence

Semantic tracing shows: "Warm tier slow AND returning low-confidence results" = tier skew detected

---

6/12

KEY INSIGHT 2: Evaluation harness beyond nDCG

We measure:
- Cross-lingual consistency: does translating query change ranking?
- Expansion effectiveness: delta nDCG from expansion vs literal
- Confidence calibration: ECE, reliability diagrams

Example: Spanish "depurar codigo" gets nDCG 0.86 via cross-lingual, only 0.34 literal-only

---

7/12

Streaming diagnostics via SSE events:

Every query result includes:
- Detected language
- Expansion terms applied
- Figurative interpretation (metaphor/idiom)
- Per-result confidence + tier
- Latency breakdown (expansion, embedding, total)

Operators aggregate for dashboards. Devs debug individual queries.

---

8/12

KEY INSIGHT 3: Statistical regression detection beats fixed thresholds

We use:
- CUSUM: catches gradual drift over weeks
- EWMA: time-weighted quality tracking

Alert: "nDCG drops >5% below 7-day EWMA for 3+ consecutive runs"

Way more reliable than "nDCG <0.75" hard cutoffs.

---

9/12

NUMA-aware benchmarking to control variance:

Early on: same code, same data, nDCG varied 3% run-to-run.

Fixes:
- Pin to CPU cores (numactl)
- Disable frequency scaling in CI
- Warm caches before timing
- Run 10x, use paired t-tests

Variance now <0.5%. Can detect real regressions.

---

10/12

Dashboard design: hierarchical detail

Top panel: aggregate RED across all queries
Middle: per-language breakdown, expansion effectiveness heatmap
Bottom: individual query traces with semantic annotations

Guides operators from "something wrong" to "here's the broken component + fix"

---

11/12

Fairness auditing as observability:

Track nDCG gap between high-resource (English) and low-resource (Bengali, Swahili) languages.

Alert if any language underperforms by >15% relative to English.

Sets floor on cross-lingual equity, prevents optimizing only for majority languages.

---

12/12

Key takeaways:

- Semantic systems need quality metrics in production, not just uptime
- Tier-aware tracing catches cross-component issues
- CUSUM/EWMA detect drift better than thresholds
- Benchmark stability requires CPU pinning, cache control
- Dashboards must guide from alert to root cause

Ship semantic search with confidence.

---

BONUS: Open challenges we're tackling

- Figurative language: how often do real users need metaphor interpretation?
- Metric composability: combine nDCG, calibration, expansion into single health score?
- Real-time adaptation: detect drift from query logs BEFORE quality degrades?
- Cross-lingual fairness targets: parity or "good enough"?

---

About Engram: open-source graph memory system with biologically-inspired consolidation, multilingual semantic recall, probabilistic spreading activation. Built in Rust.

Repo: github.com/user/engram
Docs: Our eval methodology at docs/operations/semantic_alerts.md

---

Further reading:

MIRACL multilingual benchmark (Zhang et al, NeurIPS 2023)
Confidence calibration (Guo et al, ICML 2017)
SRE patterns (Beyer et al, "Site Reliability Engineering")

Statistical process control for ML: research.google/pubs/pub45742
