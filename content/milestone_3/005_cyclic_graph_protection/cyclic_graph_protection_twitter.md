# Cyclic Graph Protection Twitter Content

## Thread: Keeping Cognitive Spreading Out of Infinite Loops

**Tweet 1/11**
Ever watched a recommendation graph fall into a loop? The same nodes light up, activation never dies, and latency spikes. Cognitive databases suffer the same fate without explicit cycle protection.

**Tweet 2/11**
Task 005 adds a four-layer guardrail for Engram:
1. Bloom filter catches potential revisits cheaply
2. Sharded visit records track hop depth
3. Adaptive penalties dampen repeat activation
4. Tier-specific hop limits enforce hard stops

All tuned for <2% overhead.

**Tweet 3/11**
Bloom filters excel because spreading is mostly first-touch. Three hash functions, 1% false positives, 512 KB footprint. Most checks never hit the heavier DashMap path (Broder & Mitzenmacher, 2004).

**Tweet 4/11**
When the Bloom filter signals a revisit, the visit table records hop count, tier, and activation snapshot. Everything fits in 32 bytes, aligned to avoid false sharing. Deterministic ordering keeps tests reproducible.

**Tweet 5/11**
Penalty curve mirrors prefrontal inhibition research (Miller & Cohen, 2001). First revisit trims activation by 7%. Each subsequent revisit adds another 2%, capped at 35%. Confidence drops at half the activation penalty to warn downstream ranking.

**Tweet 6/11**
Hot tier hop limit = 3. Warm tier = 5. Cold tier = 7. That matches complementary learning systems theory: working memory is fragile, consolidated schemas can sustain longer exploration (McClelland et al., 1995).

**Tweet 7/11**
Termination guarantee: activation mass decreases monotonically once penalties kick in, even if decay is shallow. Property tests confirm all spreads finish within configured hops on random cluster graphs.

**Tweet 8/11**
Observability matters. We export `cycles_detected_total`, `cycle_penalty_sum`, and `max_cycle_length` so Task 012's monitoring dashboards flag pathological graphs quickly.

**Tweet 9/11**
Performance win: per-hop penalty application runs in parallel (`par_iter`) with per-thread caches. In benchmarks, that kept warm-tier spreading within 1.7% of baseline latency.

**Tweet 10/11**
Cognitive realism: humans notice when thoughts repeat and dampen them. Confidence dips, attention shifts. Engram now mirrors that behavior, signaling uncertain paths to integrated recall.

**Tweet 11/11**
Cycle protection is the difference between a graph that perseverates and a graph that explores productively. Essential groundwork before we layer deterministic spreading and GPU acceleration.

---

## Bonus Thread: Debugging Cycles with Metrics

**Tweet 1/6**
Need to explain a spike in activation time? Start with `max_cycle_length`. If it jumped, a new shortcut edge likely created a tight loop.

**Tweet 2/6**
Check tier distribution. Hot-tier loops hint at overactive working memory edges; cold-tier loops suggest schema consolidation issues.

**Tweet 3/6**
Enable sampled traces: we log memory IDs, tiers, and penalty applications for 0.1% of spreads. Perfect for replaying figure-eight failures.

**Tweet 4/6**
Compare `cycle_penalty_sum` over time. Rising penalties with steady recall quality = healthy inhibition. Rising penalties with falling recall quality = overfitting in activation seeding.

**Tweet 5/6**
Performance baseline: cycle detection adds <5 µs to warm-tier spreads in our benchmarks. Anything higher indicates contention in visit tables or mis-sized Bloom filters.

**Tweet 6/6**
Cycle observability closes the loop (pun intended). You cannot tune cognitive spreading without visibility into where energy gets trapped.
