# Cognitive Patterns Operations Guide

Production deployment, monitoring, and troubleshooting for Engram's cognitive patterns.

## Quick Navigation

- [Enabling/Disabling Cognitive Patterns](#enablingdisabling-cognitive-patterns)
- [Metrics Interpretation](#metrics-interpretation)
- [Performance Tuning](#performance-tuning)
- [Runbooks](#runbooks)
- [Alert Rules](#alert-rules)
- [Capacity Planning](#capacity-planning)

---

## Enabling/Disabling Cognitive Patterns

### Compilation-Time Control

Cognitive patterns can be enabled/disabled at compile time for zero overhead when not needed.

**Default (all features enabled):**
```bash
cargo build --release
```

**With monitoring (adds <1% overhead):**
```bash
cargo build --release --features monitoring
```

**Minimal build (no cognitive patterns):**
```bash
cargo build --release --no-default-features --features core-only
```

**Feature flags:**
```toml
# Cargo.toml
[features]
default = ["priming", "interference", "reconsolidation", "monitoring"]
priming = []
interference = []
reconsolidation = []
monitoring = ["prometheus", "tracing"]
core-only = []
```

### Runtime Control

Even when compiled in, cognitive patterns can be disabled at runtime:

```rust
use engram_core::cognitive::CognitiveConfig;

let config = CognitiveConfig {
    enable_semantic_priming: true,
    enable_associative_priming: false,  // Disabled
    enable_interference_detection: true,
    enable_reconsolidation: true,
    monitoring_sampling_rate: 0.10,  // 10% sampling
};

let store = MemoryStore::with_cognitive_config(config);
```

**Environment variables (production):**
```bash
# .env or systemd environment
ENGRAM_ENABLE_PRIMING=true
ENGRAM_ENABLE_INTERFERENCE=true
ENGRAM_ENABLE_RECONSOLIDATION=true
ENGRAM_MONITORING_SAMPLING=0.10
```

---

## Metrics Interpretation

Cognitive patterns expose Prometheus metrics for monitoring. Access at `http://localhost:9090/metrics`.

### Priming Metrics

#### `engram_priming_activations_total`

**Type:** Counter
**Description:** Total number of priming activations since start
**Labels:** `type` (semantic, associative, repetition)

**Expected values:**
- **Low traffic (100 recalls/sec):** 50-150 activations/sec
- **Medium traffic (1K recalls/sec):** 500-1.5K activations/sec
- **High traffic (10K recalls/sec):** 5K-15K activations/sec

**What to watch:**
- **Sudden drop to 0:** Priming engine crashed or disabled
  - Check logs for panics
  - Verify `enable_semantic_priming=true`
- **Spike >5x baseline:** Possible activation loop
  - Check for cycles in semantic graph
  - Verify `max_graph_distance` parameter
- **Gradual increase over time:** Memory leak in active_primes
  - Check `engram_active_primes_total` growing unbounded
  - Ensure `prune_expired()` being called

#### `engram_active_primes_total`

**Type:** Gauge
**Description:** Current number of active primes in memory
**Labels:** None

**Expected values:**
- Should stabilize after warm-up period
- Typical: 100-1000 active primes (depends on `decay_half_life`)
- Formula: `active_primes ≈ activation_rate × decay_half_life`
  - Example: 500 activations/sec × 0.3 sec = 150 active primes

**What to watch:**
- **Unbounded growth:** Memory leak
  - **Symptom:** Value increases indefinitely
  - **Impact:** Eventually OOM
  - **Fix:** Ensure `prune_expired()` called every 1000 ops
- **Always 0:** Priming not activating
  - **Symptom:** 0 active primes despite recalls
  - **Fix:** Check `similarity_threshold` not too high
- **Oscillation:** Priming + spreading activation loop
  - **Symptom:** Sawtooth pattern
  - **Fix:** Increase `refractory_period`

#### `engram_priming_boost_magnitude`

**Type:** Histogram
**Description:** Distribution of priming boost magnitudes
**Buckets:** 0.01, 0.05, 0.10, 0.15, 0.20, 0.30

**Expected distribution:**
- **P50:** 0.08-0.12 (8-12% boost)
- **P95:** 0.14-0.18 (14-18% boost)
- **Max:** Should not exceed `priming_strength` (default 0.15)

**What to watch:**
- **P50 < 0.05:** Priming too weak
  - Increase `priming_strength` from 0.15 to 0.20
  - Decrease `similarity_threshold` from 0.6 to 0.5
- **P95 > 0.20:** Priming too aggressive
  - Decrease `priming_strength` from 0.15 to 0.10
  - Increase `decay_half_life` for faster decay
- **All values near 0:** Similarity threshold too high
  - Reduce from 0.7 to 0.6

#### `engram_priming_similarity_p50`

**Type:** Gauge
**Description:** Median cosine similarity of primed pairs
**Labels:** None

**Expected value:** 0.65-0.75

**What to watch:**
- **< 0.5:** Priming unrelated concepts
  - Increase `similarity_threshold`
  - Check embeddings are normalized
- **> 0.85:** Only near-duplicates priming
  - Decrease `similarity_threshold`
  - May miss useful semantic relations

### Interference Metrics

#### `engram_interference_detections_total`

**Type:** Counter
**Description:** Total interference detections
**Labels:** `type` (proactive, retroactive, fan)

**Expected rate:**
- Depends heavily on similarity of stored memories
- Typical: 5-20% of storage operations detect interference
- Fan effect: 30-50% of retrievals (common to have >1 association)

**What to watch:**
- **0 detections for >1 hour:** Interference not working
  - Check `similarity_threshold` not too high
  - Verify temporal window covers relevant memories
  - Ensure episodes have timestamps
- **>50% of operations:** Too sensitive
  - Increase `similarity_threshold` from 0.7 to 0.75
  - Reduce `interference_per_item` from 0.05 to 0.03
- **Only one type firing:** Configuration issue
  - Proactive but not retroactive: Check temporal window direction
  - Fan but not others: Check similarity computation

#### `engram_interference_magnitude`

**Type:** Histogram
**Description:** Distribution of interference strengths
**Buckets:** 0.05, 0.10, 0.15, 0.20, 0.25, 0.30

**Expected distribution:**
- **P50:** 0.08-0.15 (8-15% reduction)
- **P95:** 0.20-0.28 (20-28% reduction)
- **Max:** Should not exceed `max_interference` (0.30)

**What to watch:**
- **All values at max (0.30):** Interference saturated
  - Many similar memories competing
  - Consider increasing `max_interference` to 0.40
  - Or increase `similarity_threshold` to be more selective
- **All values < 0.05:** Interference too weak
  - Increase `interference_per_item` from 0.05 to 0.08
  - Decrease `similarity_threshold`

#### `engram_interference_items_count`

**Type:** Histogram
**Description:** Number of interfering items per detection
**Buckets:** 1, 2, 3, 5, 10, 20, 50

**Expected distribution:**
- **P50:** 2-4 interfering items
- **P95:** 8-15 interfering items
- **Max:** Depends on application

**What to watch:**
- **P95 > 50:** Too many similar memories
  - Indicates high memory overlap
  - May need better embedding diversity
  - Or tighter `similarity_threshold`
- **Always 1:** Only finding exact duplicates
  - Decrease `similarity_threshold`
  - Check embeddings have meaningful variance

### Reconsolidation Metrics

#### `engram_reconsolidation_attempts_total`

**Type:** Counter
**Description:** Reconsolidation attempts
**Labels:** `result` (success, outside_window, too_young, too_old, not_recalled, not_active)

**Expected distribution:**
- **success:** 10-30% of attempts (most fail due to timing)
- **outside_window:** 40-60% (most common failure)
- **too_young:** 5-10% (memories not consolidated yet)
- **too_old:** 5-10% (remote memories)
- **not_recalled:** 10-20% (no recent recall recorded)
- **not_active:** 1-5% (passive re-exposure)

**What to watch:**
- **0% success rate:** Window timing broken
  - Check system clock synchronized
  - Verify timestamps in UTC
  - Check `window_start` and `window_end` parameters
- **>50% success rate:** Window too wide
  - Check if `window_start` < 1 hour (should be exactly 1h)
  - Verify memory age checks working
- **High `too_young` rate:** Attempting reconsolidation too soon
  - Application should wait >24h after creation
  - Add validation before calling `attempt_reconsolidation`
- **High `too_old` rate:** Targeting remote memories
  - Expected if working with archival data
  - Consider increasing `max_memory_age` to 730 days (2 years)

#### `engram_reconsolidation_plasticity`

**Type:** Histogram
**Description:** Plasticity at attempt time
**Buckets:** 0.1, 0.2, 0.4, 0.6, 0.8, 1.0

**Expected distribution:**
- **Inverted-U shape peaking at 0.8-1.0 bucket**
- Most attempts should target peak plasticity (3h post-recall)

**What to watch:**
- **Uniform distribution:** Attempts not timed optimally
  - Applications should aim for 3h post-recall
  - Add scheduling to retry at peak plasticity
- **Peak at low plasticity (<0.3):** Attempts near window edges
  - Either too early (<1h) or too late (>5h)
  - Adjust application timing logic

#### `engram_reconsolidation_window_hits_rate`

**Type:** Gauge (computed from counters)
**Description:** Success rate within reconsolidation window
**Formula:** `success / (success + outside_window + too_young + too_old)`

**Expected value:** 0.15-0.35 (15-35%)

**What to watch:**
- **< 0.10:** Window too narrow or application mistimed
  - Check if `window_start/end` configured correctly
  - Review application recall→update timing
- **> 0.50:** Window too wide or insufficient validation
  - May be allowing updates outside biological boundaries
  - Verify age checks working

---

## Performance Tuning

### If Metrics Overhead >1%

**Symptom:** Latency P95 increased after enabling monitoring

**Diagnosis:**
```bash
# Compare with/without monitoring
cargo bench --bench cognitive_performance -- --save-baseline without_monitoring
cargo bench --bench cognitive_performance --features monitoring -- --baseline without_monitoring

# Look for >1% regression
```

**Fixes (in order of impact):**

**1. Enable sampling (biggest impact, ~90% overhead reduction):**
```rust
CognitiveConfig {
    monitoring_sampling_rate: 0.10,  // Only 10% of operations
    ..Default::default()
}
```

**2. Reduce histogram buckets:**
```rust
use prometheus::HistogramOpts;

let opts = HistogramOpts::new("engram_priming_boost_magnitude", "help")
    .buckets(vec![0.05, 0.10, 0.15, 0.20]); // Fewer buckets
```

**3. Disable detailed event tracing:**
```rust
CognitiveConfig {
    enable_event_tracing: false,  // Keep metrics, disable tracing
    ..Default::default()
}
```

**4. Use registry with smaller cardinality:**
```rust
// Avoid high-cardinality labels
// BAD: labels=["node_id"] with millions of nodes
// GOOD: labels=["type"] with 3 types
```

### If Priming Too Aggressive

**Symptom:**
- Unrelated concepts appearing in recommendations
- DRM false recall rate >75%
- Users report "weird suggestions"

**Diagnosis:**
```bash
# Check priming similarity distribution
curl -s localhost:9090/metrics | grep engram_priming_similarity_p50

# Should be 0.65-0.75
# If >0.8, priming is very selective (good)
# If <0.5, priming is promiscuous (bad)
```

**Fixes:**

**1. Reduce priming strength:**
```rust
SemanticPrimingEngine::with_config(
    0.10,           // priming_strength (was 0.15)
    Duration::from_millis(300),
    0.6,
)
```

**2. Increase decay rate (shorter half-life):**
```rust
SemanticPrimingEngine::with_config(
    0.15,
    Duration::from_millis(200), // faster decay (was 300ms)
    0.6,
)
```

**3. Raise similarity threshold:**
```rust
SemanticPrimingEngine::with_config(
    0.15,
    Duration::from_millis(300),
    0.65,           // higher threshold (was 0.6)
)
```

**4. Reduce max graph distance:**
```rust
SemanticPrimingEngine {
    max_graph_distance: 1,  // Only direct neighbors (was 2)
    ..Default::default()
}
```

### If Interference Too Sensitive

**Symptom:**
- Most storage operations showing high interference
- New memories getting very low confidence
- Interference magnitude histogram skewed to max (0.30)

**Diagnosis:**
```bash
# Check interference rate
curl -s localhost:9090/metrics | grep engram_interference_detections_total

# If >50% of storage operations, too sensitive
```

**Fixes:**

**1. Increase similarity threshold:**
```rust
ProactiveInterferenceDetector::new(
    0.75,                    // similarity_threshold (was 0.7)
    Duration::hours(6),
    0.05,
    0.30,
)
```

**2. Reduce interference per item:**
```rust
ProactiveInterferenceDetector::new(
    0.7,
    Duration::hours(6),
    0.03,                    // interference_per_item (was 0.05)
    0.25,                    // max_interference (was 0.30)
)
```

**3. Shorten temporal window:**
```rust
ProactiveInterferenceDetector::new(
    0.7,
    Duration::hours(3),      // 3 hours (was 6)
    0.05,
    0.30,
)
```

**4. Limit interfering items considered:**
```rust
// In application code
let prior_memories: Vec<_> = all_memories.iter()
    .filter(/* temporal + similarity */)
    .take(10)  // Only consider 10 most similar (not all)
    .collect();
```

### If Reconsolidation Not Triggering

**Symptom:**
- 0% success rate
- All attempts fail with `OutsideWindow` or `MemoryNotRecalled`

**Diagnosis:**
```bash
# Check attempt results distribution
curl -s localhost:9090/metrics | grep engram_reconsolidation_attempts_total

# Look at failure modes:
# outside_window = timing issue
# not_recalled = record_recall not called
# too_young/too_old = age validation failing
```

**Fixes by failure mode:**

**OutsideWindow:**
```rust
// Check system time
use chrono::Utc;
println!("Current time: {}", Utc::now());
println!("Recall time: {}", recall_timestamp);
println!("Elapsed: {}", Utc::now() - recall_timestamp);

// Should be between 1-6 hours
```

**MemoryNotRecalled:**
```rust
// Must call record_recall before attempting reconsolidation
reconsolidation.record_recall(
    episode_id,
    original_episode,
    true,  // is_active_recall
);

// Then wait 1-6 hours before:
reconsolidation.attempt_reconsolidation(
    episode_id,
    modifications,
    Utc::now(),
)?;
```

**MemoryTooYoung:**
```rust
// Check memory age
let age = Utc::now() - episode.created_at();
println!("Memory age: {} hours", age.num_hours());

// Must be >24 hours
// If application creates and immediately tries to reconsolidate, will fail
```

**MemoryTooOld:**
```rust
// Increase max age for archival data
ReconsolidationEngine {
    max_memory_age: Duration::days(730),  // 2 years (was 365 days)
    ..Default::default()
}
```

---

## Runbooks

Step-by-step procedures for common operational issues.

### RUNBOOK: Priming Event Rate Spiking (>5000/sec)

**Alert Trigger:** `rate(engram_priming_activations_total[1m]) > 5000`

**Business Impact:**
- Increased latency (P95 may spike)
- Memory pressure from active_primes
- Possible activation loop consuming CPU

**Symptoms:**
- `engram_priming_activations_total` increasing rapidly
- `engram_active_primes_total` growing unbounded
- CPU usage elevated on priming threads
- Recall latency P95 increased

---

**Step 1: Verify it's a real issue**

```bash
# Check current activation rate
curl -s localhost:9090/metrics | grep engram_priming_activations_total

# Compare to baseline (expect 100-1000/sec for typical workload)
# If >5000/sec sustained for >5 minutes, investigate
```

**Expected:** 100-1000 activations/sec
**Problem:** >5000 activations/sec sustained

---

**Step 2: Identify root cause**

```bash
# Check for activation loops
curl -s localhost:9090/metrics | grep engram_active_primes_total

# If growing linearly: memory leak (pruning not called)
# If oscillating: activation loop (spreading + priming feedback)
# If stable but high rate: legitimate traffic spike
```

**Look for:**
- **Pattern A (linear growth):** `active_primes_total` increasing monotonically
  - **Root cause:** `prune_expired()` not being called
  - **Evidence:** Memory usage also increasing

- **Pattern B (oscillation):** `active_primes_total` sawtooth pattern
  - **Root cause:** Spreading activation + priming feedback loop
  - **Evidence:** Same nodes activating repeatedly

- **Pattern C (stable):** `active_primes_total` stable, just high rate
  - **Root cause:** Legitimate traffic increase
  - **Evidence:** `recall_operations_total` also elevated

---

**Step 3: Assess severity**

- **Critical:** P95 latency >500ms or memory >80% (page immediately)
- **High:** P95 latency >200ms or memory >60% (fix within 1 hour)
- **Medium:** Elevated but stable (fix within 1 day)
- **Low:** Brief spike, already recovering (monitor)

---

**Step 4: Apply fix based on root cause**

**Fix for Pattern A (memory leak):**

```rust
// In application code, ensure pruning called regularly
for (i, episode) in episodes.iter().enumerate() {
    priming.activate_priming(episode);

    // Add this:
    if i % 1000 == 0 {
        priming.prune_expired();
    }
}
```

**Pros:** Fixes leak, returns to baseline memory
**Cons:** Adds small overhead every 1000 ops
**Time to apply:** Deploy new code (~15 min)
**Validation:** `active_primes_total` stops growing

---

**Fix for Pattern B (activation loop):**

```rust
// Increase refractory period to dampen oscillations
SemanticPrimingEngine {
    refractory_period: Duration::from_millis(100),  // was 50ms
    ..Default::default()
}
```

**Pros:** Breaks feedback loop
**Cons:** May reduce legitimate priming
**Time to apply:** Config change, restart (~5 min)
**Validation:** Activation rate stabilizes

---

**Fix for Pattern C (traffic spike):**

```rust
// Enable sampling to reduce overhead
CognitiveConfig {
    monitoring_sampling_rate: 0.10,  // 10% sampling
    ..Default::default()
}

// Or reduce priming strength temporarily
SemanticPrimingEngine::with_config(
    0.10,  // Reduced from 0.15
    Duration::from_millis(300),
    0.6,
)
```

**Pros:** Reduces load without dropping priming entirely
**Cons:** Weaker priming effects
**Time to apply:** Config change, rolling restart (~10 min)
**Validation:** Activation rate reduced, latency improved

---

**Step 5: Validation procedure**

**Immediate verification (5 min):**
```bash
# Check metrics return to healthy range
watch -n 1 'curl -s localhost:9090/metrics | grep -E "engram_priming_activations|engram_active_primes"'

# Should see:
# - Activation rate trending down toward baseline
# - Active primes stable or declining
# - No more linear growth
```

**Short-term validation (1 hour):**
```bash
# Run priming integration tests
cargo test --test priming_integration_tests --release

# Should pass with normal activation patterns
```

**Long-term monitoring (24 hours):**
- Alert should not re-fire
- `engram_active_primes_total` stable
- P95 latency back to baseline (<100ms)
- Memory usage stable

---

**Step 6: Prevention**

**Monitoring:**
```yaml
# Prometheus alert rule
- alert: PrimingActivationRateHigh
  expr: rate(engram_priming_activations_total[5m]) > 5000
  for: 5m
  annotations:
    summary: "Priming activation rate abnormally high"
    description: "Rate: {{ $value }}/sec (baseline: 100-1000/sec)"
    runbook_url: "https://docs.engram.ai/operations/cognitive-metrics-tuning#runbook-priming-event-rate-spiking"
```

**Pre-deployment validation:**
```bash
# Load test with priming enabled
cargo test --test load_test_priming --release -- --nocapture

# Verify activation rate scales linearly with load
# Not super-linearly (indicates loop)
```

**Configuration review:**
- Ensure `prune_expired()` in hot path
- Set `max_graph_distance=2` (not unbounded)
- Enable `refractory_period >= 50ms`

---

**Related incidents:**
- 2025-10-15: Activation loop due to cyclic semantic graph (fixed by refractory period)
- 2025-09-22: Memory leak from missing prune calls (fixed by adding to main loop)

---

### RUNBOOK: DRM False Recall Out of Range (<45% or >75%)

**Alert Trigger:** Weekly DRM validation test fails acceptable range

**Business Impact:**
- <45%: Semantic network too sparse, poor generalization
- >75%: Pattern completion too aggressive, hallucination risk
- Users may get incorrect recommendations

**Symptoms:**
- DRM validation test: `cargo test --test drm_paradigm` fails
- `engram_priming_boost_magnitude` distribution shifted
- User reports of "surprising" or "wrong" suggestions

---

**Step 1: Verify it's a real issue**

```bash
# Run DRM validation test
cargo test --test drm_paradigm --release -- --nocapture

# Look for output:
# "False recall rate: X% (target: 55-65%)"
#
# If X < 45% or X > 75%, investigate
```

**Expected:** 55-65% false recall
**Problem:** <45% (too low) or >75% (too high)

---

**Step 2: Identify root cause**

```bash
# Check priming parameters
curl -s localhost:9090/metrics | grep -E "priming_boost|priming_similarity"

# Check pattern completion (from M8)
# Check consolidation strength (from M6)
```

**Common causes:**

**Too low (<45%):**
- Similarity threshold too high (>0.7)
- Priming strength too weak (<0.10)
- Pattern completion threshold too high
- Consolidation not strengthening associations

**Too high (>75%):**
- Similarity threshold too low (<0.5)
- Priming strength too strong (>0.20)
- Pattern completion threshold too low
- Consolidation over-strengthening associations

---

**Step 3: Assess severity**

- **Critical:** >90% or <30% (system behavior very wrong)
- **High:** >80% or <40% (fix before production)
- **Medium:** >70% or <50% (investigate and tune)
- **Low:** 65-70% or 45-55% (acceptable, monitor)

---

**Step 4: Fix options**

**Fix 1: Adjust priming strength (recommended for most cases)**

```rust
// If false recall too low (<45%)
SemanticPrimingEngine::with_config(
    0.18,  // Increase from 0.15
    Duration::from_millis(300),
    0.6,
)

// If false recall too high (>75%)
SemanticPrimingEngine::with_config(
    0.12,  // Decrease from 0.15
    Duration::from_millis(300),
    0.6,
)
```

**Pros:** Direct impact on semantic activation strength
**Cons:** Affects all priming, not just DRM
**Time to apply:** Config change, restart
**Validation:** Re-run DRM test

---

**Fix 2: Adjust similarity threshold**

```rust
// If false recall too low (<45%)
SemanticPrimingEngine::with_config(
    0.15,
    Duration::from_millis(300),
    0.55,  // Lower threshold (was 0.6)
)

// If false recall too high (>75%)
SemanticPrimingEngine::with_config(
    0.15,
    Duration::from_millis(300),
    0.68,  // Higher threshold (was 0.6)
)
```

**Pros:** Adjusts selectivity without changing strength
**Cons:** May miss or over-include semantic relations
**Time to apply:** Config change, restart
**Validation:** Check `priming_similarity_p50` metric

---

**Fix 3: Adjust pattern completion (M8)**

```rust
// Check pattern completion threshold
// Documented in M8 (Pattern Completion)

// If false recall too low, reduce threshold
// If false recall too high, increase threshold
```

**Pros:** Targets reconstruction mechanism specifically
**Cons:** Requires M8 implementation details
**Time to apply:** Config change
**Validation:** DRM test + pattern completion tests

---

**Step 5: Validation procedure**

**Immediate verification:**
```bash
# Re-run DRM test with new config
cargo test --test drm_paradigm --release -- --nocapture

# Should show false recall in 55-65% range
```

**Statistical validation:**
```bash
# Run 100 trials for statistical power
cargo test --test drm_paradigm --release -- --nocapture --test-threads=1 --ignored

# Look for:
# - Mean in 55-65% range
# - 95% CI overlaps target range
# - Chi-square p > 0.05 (not significantly different from 60%)
```

**Short-term validation (1 hour):**
- Check user recommendations quality
- Monitor `priming_boost_magnitude` distribution
- Verify no other cognitive tests regressed

**Long-term monitoring (1 week):**
- Re-run DRM test weekly
- Track user feedback on recommendation quality
- Monitor false positive rates in production

---

**Step 6: Prevention**

**Continuous validation:**
```bash
# Add to CI pipeline
cargo test --test drm_paradigm --release

# Fails build if false recall out of range
```

**Monitoring:**
```yaml
# Weekly cron job
0 0 * * 0 cd /opt/engram && cargo test --test drm_paradigm --release | mail -s "DRM Validation" ops@engram.ai
```

**Parameter review:**
- Review priming config during quarterly tuning
- Document any changes in parameter log
- A/B test changes in staging before production

---

### RUNBOOK: Interference Detection Failures (0 detections for 1 hour)

**Alert Trigger:** `rate(engram_interference_detections_total[1h]) == 0`

**Business Impact:**
- Similar memories not being distinguished
- Users may confuse similar items
- Confidence scores unrealistically high

**Symptoms:**
- `engram_interference_detections_total` counter not incrementing
- No interference warnings in logs
- User reports of confusion between similar items

---

**Step 1: Verify it's a real issue**

```bash
# Check interference detection counter
curl -s localhost:9090/metrics | grep engram_interference_detections_total

# If 0 for >1 hour during normal traffic, investigate

# Also check if interference detection enabled
curl -s localhost:9090/config | grep enable_interference_detection
# Should be: true
```

---

**Step 2: Identify root cause**

```bash
# Check similarity threshold
curl -s localhost:9090/config | grep interference_similarity_threshold

# Check temporal window
curl -s localhost:9090/config | grep prior_memory_window

# Check episode timestamps
curl -s localhost:9090/debug/recent_episodes | jq '.[].timestamp'
# Should have valid timestamps, not all null
```

**Common causes:**
- Similarity threshold too high (>0.8)
- Temporal window misconfigured (0 or negative)
- Episodes missing timestamps
- Embeddings not being computed
- Interference detection disabled in config

---

**Step 3: Fix options**

**Fix 1: Lower similarity threshold**

```rust
ProactiveInterferenceDetector::new(
    0.65,  // Lower from 0.7
    Duration::hours(6),
    0.05,
    0.30,
)
```

**Fix 2: Check episode timestamps**

```rust
// Ensure episodes have timestamps
let episode = EpisodeBuilder::new()
    .id("test")
    .when(Utc::now())  // Must include timestamp
    .build()?;
```

**Fix 3: Verify embeddings computed**

```rust
// Check embedding is not zeros
assert!(!episode.embedding().iter().all(|&x| x == 0.0));

// Check embedding normalized
let norm: f32 = episode.embedding().iter().map(|x| x*x).sum::<f32>().sqrt();
assert!((norm - 1.0).abs() < 0.01);  // Should be unit vector
```

---

### RUNBOOK: Reconsolidation Window Misses (hit rate <5%)

**Alert Trigger:** `engram_reconsolidation_window_hits_rate < 0.05` for >1 hour

**Business Impact:**
- Memory updates not persisting
- User edits being rejected
- Therapeutic applications not working

**Symptoms:**
- Most reconsolidation attempts fail with `OutsideWindow`
- `reconsolidation_plasticity` histogram shows values near 0
- User reports of "changes not saving"

---

**Step 1: Verify timing**

```bash
# Check system time
date -u

# Check recall timestamps
curl -s localhost:9090/debug/recent_recalls | jq '.[].recall_timestamp'

# Check current time vs recall time
# Should be 1-6 hours apart
```

---

**Step 2: Fix timing issues**

**Fix 1: Schedule updates at peak plasticity**

```rust
// Record recall
reconsolidation.record_recall(episode_id, episode, true);

// Schedule update for 3 hours later (peak plasticity)
tokio::spawn(async move {
    tokio::time::sleep(Duration::hours(3)).await;

    let result = reconsolidation.attempt_reconsolidation(
        episode_id,
        modifications,
        Utc::now(),
    );

    // Should succeed (within window, at peak plasticity)
});
```

**Fix 2: Retry logic with window checking**

```rust
// Check if in window before attempting
if reconsolidation.is_in_window(episode_id, Utc::now()) {
    // Compute plasticity
    let plasticity = reconsolidation.compute_plasticity(
        Utc::now() - recall_timestamp
    );

    if plasticity > 0.5 {  // Only attempt if plasticity reasonable
        reconsolidation.attempt_reconsolidation(/* ... */)?;
    }
}
```

---

### RUNBOOK: Memory Leak in Active Primes (unbounded growth)

**Alert Trigger:** `engram_active_primes_total` growing without bound

**Business Impact:**
- Eventual OOM and crash
- Degraded performance as memory pressure increases
- Requires restart to recover

---

**Step 1: Confirm leak**

```bash
# Monitor active primes over time
watch -n 10 'curl -s localhost:9090/metrics | grep engram_active_primes_total'

# If consistently increasing (not oscillating), leak confirmed
# Expected: Stabilize after warm-up period
```

---

**Step 2: Check pruning**

```bash
# Search codebase for prune_expired calls
grep -r "prune_expired" src/

# Should appear in hot paths
# Recommended: Every 1000 operations
```

---

**Step 3: Add pruning**

```rust
// In recall loop
for (i, episode) in episodes.iter().enumerate() {
    priming.activate_priming(&episode);

    // Add periodic pruning
    if i % 1000 == 0 {
        priming.prune_expired();
    }
}

// Or in background task
tokio::spawn(async move {
    let mut interval = tokio::time::interval(Duration::from_secs(10));
    loop {
        interval.tick().await;
        priming.prune_expired();
    }
});
```

---

**Step 4: Deploy fix**

```bash
# Deploy updated code
cargo build --release
systemctl restart engram

# Monitor active_primes_total
# Should stabilize within 5 minutes
```

---

## Alert Rules

Prometheus alert rules for production monitoring.

```yaml
# /etc/prometheus/alerts/engram_cognitive.yml

groups:
  - name: engram_cognitive
    interval: 30s
    rules:
      # Priming alerts
      - alert: PrimingActivationRateHigh
        expr: rate(engram_priming_activations_total[5m]) > 5000
        for: 5m
        labels:
          severity: warning
          component: priming
        annotations:
          summary: "Priming activation rate abnormally high"
          description: "Rate: {{ $value }}/sec (baseline: 100-1000/sec)"
          runbook_url: "https://docs.engram.ai/operations/cognitive-metrics-tuning#runbook-priming-event-rate-spiking"

      - alert: PrimingMemoryLeak
        expr: deriv(engram_active_primes_total[10m]) > 10
        for: 30m
        labels:
          severity: critical
          component: priming
        annotations:
          summary: "Active primes growing unbounded (memory leak)"
          description: "Growth rate: {{ $value }} primes/sec"
          runbook_url: "https://docs.engram.ai/operations/cognitive-metrics-tuning#runbook-memory-leak-in-active-primes"

      - alert: PrimingNotActivating
        expr: rate(engram_priming_activations_total[1h]) == 0
        for: 1h
        labels:
          severity: warning
          component: priming
        annotations:
          summary: "No priming activations for 1 hour"
          description: "Check similarity threshold and configuration"

      # Interference alerts
      - alert: InterferenceNotDetecting
        expr: rate(engram_interference_detections_total[1h]) == 0
        for: 1h
        labels:
          severity: warning
          component: interference
        annotations:
          summary: "No interference detections for 1 hour"
          description: "Check similarity threshold and timestamps"
          runbook_url: "https://docs.engram.ai/operations/cognitive-metrics-tuning#runbook-interference-detection-failures"

      - alert: InterferenceTooAggressive
        expr: rate(engram_interference_detections_total[5m]) / rate(engram_store_operations_total[5m]) > 0.5
        for: 10m
        labels:
          severity: warning
          component: interference
        annotations:
          summary: "Interference detection too aggressive (>50% of operations)"
          description: "Detection rate: {{ $value }}"

      # Reconsolidation alerts
      - alert: ReconsolidationLowSuccessRate
        expr: |
          sum(rate(engram_reconsolidation_attempts_total{result="success"}[1h]))
          /
          sum(rate(engram_reconsolidation_attempts_total[1h]))
          < 0.05
        for: 1h
        labels:
          severity: warning
          component: reconsolidation
        annotations:
          summary: "Reconsolidation success rate <5%"
          description: "Success rate: {{ $value }}"
          runbook_url: "https://docs.engram.ai/operations/cognitive-metrics-tuning#runbook-reconsolidation-window-misses"

      - alert: ReconsolidationTimingBroken
        expr: rate(engram_reconsolidation_attempts_total{result="success"}[1h]) == 0
        for: 2h
        labels:
          severity: critical
          component: reconsolidation
        annotations:
          summary: "Zero successful reconsolidations for 2 hours"
          description: "Check system clock and window parameters"

      # Validation alerts
      - alert: DRMValidationFailed
        expr: engram_drm_false_recall_rate < 0.45 or engram_drm_false_recall_rate > 0.75
        for: 1h
        labels:
          severity: warning
          component: validation
        annotations:
          summary: "DRM false recall out of acceptable range (55-65%)"
          description: "False recall rate: {{ $value }}"
          runbook_url: "https://docs.engram.ai/operations/cognitive-metrics-tuning#runbook-drm-false-recall-out-of-range"
```

---

## Capacity Planning

Estimating resource requirements for cognitive patterns.

### Memory Requirements

**Priming:**
- Active primes: 48 bytes per prime
- Typical: 100-1000 active primes
- Memory: 5-50 KB (negligible)

**Formula:**
```
priming_memory = activation_rate × decay_half_life × 48 bytes
```

**Example:**
- 500 activations/sec
- 300ms half-life
- Memory: 500 × 0.3 × 48 = 7,200 bytes ≈ 7 KB

**Interference:**
- No persistent state
- Temporary allocations during detection
- Memory: <1 KB per detection

**Reconsolidation:**
- Recent recalls: 256 bytes per recall
- Typical: 100-1000 recent recalls (within 7 days)
- Memory: 25-250 KB (negligible)

**Total cognitive patterns overhead: <1 MB typical**

### CPU Requirements

**Priming:**
- `activate_priming()`: 85μs P50
- `compute_priming_boost()`: 8ns P50
- At 1000 ops/sec: ~85ms CPU/sec = 8.5% of 1 core

**Interference:**
- Detection: 150μs per episode (with 10 candidates)
- At 100 detections/sec: 15ms CPU/sec = 1.5% of 1 core

**Reconsolidation:**
- `attempt_reconsolidation()`: 1.5μs P50
- Rare operations (<10/sec typically)
- CPU: <0.01% of 1 core

**Total: <10% of 1 core for typical workloads**

### Scaling Guidelines

**Single node capacity:**
- Up to 10K recall operations/sec
- Up to 1K interference detections/sec
- Cognitive patterns add <10% overhead
- Bottleneck: Main memory operations, not cognitive patterns

**When to scale horizontally:**
- >10K ops/sec sustained
- >80% CPU on memory operations
- Not because of cognitive patterns specifically

**When to disable cognitive patterns:**
- Ultra-low latency requirements (<10μs P99)
- Memory-constrained embedded systems (<100MB RAM)
- Batch processing where cognitive effects unnecessary

---

## Production Deployment Checklist

Before deploying cognitive patterns to production:

### Configuration Review

- [ ] Priming strength appropriate for use case (0.10-0.20 range)
- [ ] Similarity thresholds validated (0.5-0.7 range)
- [ ] Temporal windows match application timing
- [ ] Monitoring enabled with sampling if needed
- [ ] Alert rules configured in Prometheus

### Testing

- [ ] DRM validation test passes (55-65% false recall)
- [ ] Spacing effect test passes (20-40% improvement)
- [ ] Interference validation passes (within ±10%)
- [ ] Load test with cognitive patterns enabled
- [ ] Latency P95 <1% regression vs baseline

### Monitoring

- [ ] Grafana dashboard configured
- [ ] Alert rules tested (trigger alerts manually)
- [ ] Runbook URLs accessible to on-call
- [ ] Metrics scraping interval ≤30s
- [ ] Log aggregation capturing cognitive events

### Operational Readiness

- [ ] On-call team trained on runbooks
- [ ] Rollback procedure tested
- [ ] Capacity planning reviewed
- [ ] Incident response procedures documented
- [ ] Post-deployment validation plan

### Documentation

- [ ] API documentation linked from main docs
- [ ] Psychology foundations accessible
- [ ] Operations guide bookmarked
- [ ] Parameter tuning guide reviewed
- [ ] Example configurations documented

---

## Summary

Cognitive patterns add rich, biologically-plausible memory dynamics to Engram:
- **Priming** makes related concepts easier to recall
- **Interference** models competition between similar memories
- **Reconsolidation** enables memory updates during specific windows

With proper monitoring and tuning, cognitive patterns add <1% overhead while dramatically improving memory realism. Use the runbooks in this guide to operate cognitive patterns successfully in production.

For questions or issues not covered here, see:
- [API Reference](/docs/reference/cognitive_patterns.md)
- [Psychology Foundations](/docs/explanation/psychology_foundations.md)
- [GitHub Issues](https://github.com/engram/engram/issues)
