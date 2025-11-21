# Dual Memory Architecture Production Rollout Runbook

**Version:** 1.0
**Last Updated:** 2025-11-20
**Owner:** Platform Engineering Team
**Review Cycle:** Before each phase transition

## Executive Summary

This runbook provides step-by-step procedures for safely rolling out Engram's dual-memory architecture (Milestone 17) to production. The dual-memory system introduces episodic and semantic memory types with blended recall capabilities, enabling cognitively-plausible memory retrieval that mirrors human complementary learning systems.

**What This Rollout Delivers:**
- Dual memory types: Episodes (specific experiences) + Concepts (generalized patterns)
- Blended recall combining fast episodic and slower semantic pathways
- Fan effect penalties for realistic cognitive dynamics
- Enhanced monitoring with dual-memory-specific metrics

**Rollout Strategy:**
- Phased deployment with feature flags: 5 phases over 3-4 weeks
- Performance regression target: <5% P99 latency increase
- A/B testing to validate improvements in recall robustness
- Graceful rollback capability at each phase

**Critical Success Factors:**
1. Comprehensive pre-deployment validation (24-hour soak test)
2. Tight monitoring at each phase (Grafana dashboards + Prometheus alerts)
3. Abort triggers clearly defined with automated detection
4. Statistical rigor in A/B testing (p<0.05 significance threshold)

## Feature Flags

The dual-memory architecture is controlled by four feature flags in Engram's configuration:

| Flag | Purpose | Shadow Mode | Canary | Full Rollout |
|------|---------|-------------|--------|--------------|
| `dual_memory_types` | Enable episode/concept type system | ON | ON | ON |
| `blended_recall` | Blend episodic + semantic pathways | OFF | ON (5%) | ON (100%) |
| `fan_effect` | Apply fan-out penalties during spreading | OFF | ON (5%) | ON (100%) |
| `monitoring` | Dual-memory metrics export | ON | ON | ON |

**Flag Configuration (engram.toml):**

```toml
[features]
dual_memory_types = true    # Foundation for dual memory
blended_recall = false       # Toggle for blended recall (gradual ramp)
fan_effect = false           # Toggle for fan effect penalties
monitoring = true            # Always enabled for observability
```

**Runtime Flag Toggle (HTTP API):**

```bash
# Enable blended recall for canary cohort
curl -X POST http://localhost:7432/api/v1/admin/features \
  -H "Content-Type: application/json" \
  -d '{"blended_recall": true, "cohort": "canary_5pct"}'

# Disable blended recall (rollback)
curl -X POST http://localhost:7432/api/v1/admin/features \
  -H "Content-Type: application/json" \
  -d '{"blended_recall": false}'
```

## Phase 0: Pre-Deployment Validation

**Duration:** 3-5 days
**Objective:** Establish performance baseline and validate monitoring infrastructure before any production changes.

### 0.1 Integration Test Suite

Run the complete dual-memory integration test suite:

```bash
cd /Users/jordanwashburn/Workspace/orchard9/engram

# Run all integration tests (requires dual_memory_types feature)
cargo test --features dual_memory_types --test dual_memory_migration
cargo test --features dual_memory_types --test dual_memory_feature_flags
cargo test --features dual_memory_types --test dual_memory_differential
cargo test --features dual_memory_types --test dual_memory_load
cargo test --features dual_memory_types --test dual_memory_chaos

# Expected: All tests pass, zero failures
```

**Success Criteria:**
- All 5 test suites pass with zero failures
- Migration tests validate data integrity and recall ordering
- Feature flag matrix tests exercise all flag combinations
- Differential tests show <5% deviation between single-type and dual-type engines

### 0.2 24-Hour Soak Test

Establish performance baseline with extended load testing:

```bash
# Build release binary
cargo build --release --features dual_memory_types,monitoring

# Start Engram with dual_memory_types enabled but blended_recall disabled
./target/release/engram start \
  --config config/production.toml \
  --port 7432 &

ENGRAM_PID=$!

# Wait for health check
sleep 5
curl -f http://localhost:7432/health || exit 1

# Run 24-hour baseline soak test
./target/release/loadtest run \
  --scenario scenarios/m17_baseline.toml \
  --duration 86400 \
  --seed 0xDEADBEEF \
  --endpoint http://localhost:7432 \
  --output tmp/m17_performance/phase0_soak_baseline.json \
  > tmp/m17_performance/phase0_soak.log 2>&1 &

LOADTEST_PID=$!

# Monitor system resources every 15 minutes
watch -n 900 'ps -p $ENGRAM_PID -o rss,vsz,%cpu,%mem >> tmp/m17_performance/phase0_soak_resources.log'
```

**Monitoring During Soak Test:**
- Check Grafana "Engram Overview" dashboard every 4 hours
- Verify no memory leaks: RSS should stabilize within 6 hours
- CPU usage should remain <60% on average
- No spreading circuit breaker trips
- Consolidation runs complete successfully every 5 minutes

**Success Criteria:**
- 24-hour test completes without crashes or panics
- P99 latency remains stable (<10% variance hour-over-hour)
- Error rate <0.1% throughout entire soak period
- Memory footprint increases <5% over 24 hours (GC stability)
- No consolidation staleness alerts (freshness <450s)

**Abort Triggers:**
- Engram process crash or panic
- Error rate >1% sustained for 1 hour
- P99 latency >50ms (2x baseline) for 2 hours
- Memory growth >1GB/hour (memory leak indicator)

### 0.3 Performance Baseline Establishment

After soak test completion, establish quantitative baselines:

```bash
# Extract baseline metrics
jq '.latency.p50_ms, .latency.p95_ms, .latency.p99_ms, .throughput.ops_per_sec, .errors.total' \
  tmp/m17_performance/phase0_soak_baseline.json \
  > tmp/m17_performance/BASELINE_METRICS.txt

# Document in performance log
cat >> PERFORMANCE_LOG.md <<EOF

## Phase 0 Baseline (Pre-Dual-Memory)
- Date: $(date +%Y-%m-%d)
- Configuration: dual_memory_types=ON, blended_recall=OFF
- Duration: 24 hours
- Metrics:
  - P50 latency: $(jq -r '.latency.p50_ms' tmp/m17_performance/phase0_soak_baseline.json) ms
  - P95 latency: $(jq -r '.latency.p95_ms' tmp/m17_performance/phase0_soak_baseline.json) ms
  - P99 latency: $(jq -r '.latency.p99_ms' tmp/m17_performance/phase0_soak_baseline.json) ms
  - Throughput: $(jq -r '.throughput.ops_per_sec' tmp/m17_performance/phase0_soak_baseline.json) ops/s
  - Error rate: $(jq -r '.errors.error_rate' tmp/m17_performance/phase0_soak_baseline.json)
- Status: Baseline established
EOF
```

### 0.4 Monitoring Stack Verification

Verify Grafana dashboards and Prometheus alerts are operational:

```bash
# Check Prometheus targets
curl -s http://localhost:9090/api/v1/targets | jq '.data.activeTargets[] | select(.labels.job=="engram") | .health'
# Expected: "up"

# Verify dual-memory metrics are exported
curl -s http://localhost:7432/metrics | grep -E 'engram_(concepts|bindings|blended)'

# Expected metrics (even with blended_recall=OFF, they should exist with zero values):
# - engram_concepts_formed_total 0
# - engram_bindings_created_total 0
# - engram_blended_recall_total 0

# Check Grafana dashboard accessibility
curl -s -u admin:admin http://localhost:3000/api/dashboards/uid/dual-memory-overview | jq '.dashboard.title'
# Expected: "Dual Memory Architecture Overview"

# Verify alert rules loaded
curl -s http://localhost:9090/api/v1/rules | jq '.data.groups[] | select(.name=="dual_memory") | .rules[].name'
# Expected: ConceptQualityViolation, FanEffectExcessive, BlendedRecallLatency, etc.
```

**Success Criteria:**
- Prometheus successfully scraping Engram /metrics endpoint (15s interval)
- Grafana "Dual Memory Architecture Overview" dashboard displays data
- Grafana "Cognitive Metrics Tuning" dashboard displays data
- All dual-memory alert rules loaded (at least 6 rules in "dual_memory" group)
- Test alert fires correctly (temporarily set threshold to trigger test alert)

### 0.5 Operator Training Checklist

Ensure on-call team is prepared:

- [ ] Reviewed this runbook end-to-end
- [ ] Familiarized with Grafana dual-memory dashboards (location: http://localhost:3000/d/dual-memory-overview)
- [ ] Practiced feature flag toggle commands (enable/disable blended_recall)
- [ ] Tested rollback procedure in staging environment
- [ ] Reviewed abort triggers and escalation procedures
- [ ] Set up PagerDuty/Slack notifications for dual-memory alerts
- [ ] Confirmed backup/restore procedures documented (see docs/operations/backup-restore.md)
- [ ] Identified SMEs for each component (consolidation, activation, binding)

## Phase 1: Shadow Mode Deployment

**Duration:** 1 week
**Objective:** Enable dual memory type system and concept formation WITHOUT affecting recall behavior. Monitor resource overhead.

### 1.1 Configuration Changes

Enable `dual_memory_types` feature in production configuration:

```toml
# config/production.toml

[features]
dual_memory_types = true     # NEW: Enable episode/concept types
blended_recall = false        # Keep OFF - shadow mode only
fan_effect = false            # Keep OFF - shadow mode only
monitoring = true             # Already ON from Phase 0

[consolidation]
enable_concept_formation = true   # NEW: Form concepts during consolidation
concept_min_coherence = 0.6       # Only create high-quality concepts
concept_formation_interval_minutes = 5  # Run every 5min
```

**Deployment:**

```bash
# Kubernetes deployment
kubectl apply -f deployments/kubernetes/engram-shadow-mode.yaml

# Docker deployment
docker stop engram
docker run -d \
  --name engram \
  -p 7432:7432 \
  -v $(pwd)/config/production-shadow.toml:/config/engram.toml:ro \
  -v engram-data:/data \
  engram:milestone-17-shadow

# Verify feature flags
curl http://localhost:7432/api/v1/admin/features | jq '.dual_memory_types, .blended_recall'
# Expected: { "dual_memory_types": true, "blended_recall": false }
```

### 1.2 Monitoring Targets for Shadow Mode

**Key Metrics to Watch:**

1. **Concept Formation Overhead**
   - Metric: `engram_concepts_formed_total`
   - Dashboard: "Dual Memory Architecture Overview" > "Concept Formation Rate"
   - Target: <1000 concepts/hour during consolidation runs
   - Alert: ConceptFormationExcessive if >5000 concepts/hour for 15m

2. **Memory Footprint Increase**
   - Metric: `process_resident_memory_bytes`
   - Dashboard: "Engram Overview" > "Memory Usage"
   - Target: <10% increase vs Phase 0 baseline
   - Alert: MemoryFootprintRegression if >15% increase sustained for 30m

3. **Consolidation Latency**
   - Metric: `engram_consolidation_duration_seconds{quantile="0.99"}`
   - Dashboard: "Cognitive Metrics Tuning" > "Consolidation Performance"
   - Target: P99 <5s (within 20% of Phase 0 baseline)
   - Alert: ConsolidationLatencyRegression if P99 >6s for 10m

4. **Binding Index Size**
   - Metric: `engram_bindings_total`
   - Dashboard: "Dual Memory Architecture Overview" > "Binding Index Stats"
   - Target: Bindings accumulate linearly with episodes (<2x episode count)
   - Alert: BindingIndexBloat if ratio >3.0

**Query Examples:**

```promql
# Concept formation rate (concepts/min)
rate(engram_concepts_formed_total[5m]) * 60

# Memory overhead percentage vs baseline
((process_resident_memory_bytes - <BASELINE_FROM_PHASE_0>) / <BASELINE_FROM_PHASE_0>) * 100

# Consolidation P99 latency trend
engram_consolidation_duration_seconds{quantile="0.99"}

# Binding-to-episode ratio
engram_bindings_total / engram_episodes_total
```

### 1.3 Success Criteria (Shadow Mode)

**Functional:**
- [ ] Concept formation runs successfully every 5 minutes
- [ ] Concepts stored with valid centroids and coherence scores
- [ ] Binding index populated with episode-concept associations
- [ ] Recall behavior unchanged (blended_recall=OFF ensures episodic-only)
- [ ] No panics or crashes related to dual memory code

**Performance:**
- [ ] Memory footprint increase <10% vs Phase 0 baseline
- [ ] Consolidation P99 latency increase <20% vs Phase 0 baseline
- [ ] Recall P99 latency unchanged (<2% variance vs Phase 0)
- [ ] CPU usage increase <15% vs Phase 0 baseline
- [ ] No consolidation staleness alerts

**Data Quality:**
- [ ] Concept coherence scores >0.6 for 95% of concepts
- [ ] No concept quality violation alerts
- [ ] Binding strengths in valid range [0.0, 1.0]
- [ ] Episode-to-concept binding ratio <2.0

### 1.4 Abort Triggers (Shadow Mode)

Immediately rollback to Phase 0 if:

1. **Memory leak detected:** RSS growth >500MB/hour sustained for 3 hours
2. **Consolidation failures:** >10% of consolidation runs fail within 24 hours
3. **Recall regression:** P99 latency increases >5% (blended_recall is OFF, so any increase is a bug)
4. **Crash or panic:** Any dual-memory-related panic in production logs
5. **Data corruption:** Concept coherence scores invalid (NaN, negative, >1.0) for >1% of concepts

**Rollback Procedure:**

```bash
# Disable dual_memory_types feature flag
kubectl set env deployment/engram DUAL_MEMORY_TYPES=false

# OR for Docker
docker stop engram
docker run -d \
  --name engram \
  -v $(pwd)/config/production-baseline.toml:/config/engram.toml:ro \
  engram:milestone-16-stable

# Verify rollback
curl http://localhost:7432/api/v1/admin/features | jq '.dual_memory_types'
# Expected: false

# Monitor for stabilization (15 minutes)
watch -n 60 'curl -s http://localhost:7432/api/v1/system/health | jq ".status, .uptime_seconds"'
```

## Phase 2: Canary Deployment (5% Traffic)

**Duration:** 1 week
**Objective:** Enable blended recall for a small cohort (5% of queries). Validate improvements in recall robustness without performance regression.

### 2.1 Canary Cohort Configuration

Enable `blended_recall` for 5% of traffic using consistent hashing:

```toml
# config/production-canary.toml

[features]
dual_memory_types = true
blended_recall = true          # NEW: Enable blended recall
fan_effect = true              # NEW: Enable fan effect penalties
monitoring = true

[blended_recall]
cohort_sampling_rate = 0.05    # 5% of queries
cohort_seed = 42                # Deterministic cohort assignment
base_episodic_weight = 0.7
base_semantic_weight = 0.3
adaptive_weighting = true
enable_pattern_completion = true
min_concept_coherence = 0.6
semantic_timeout_ms = 8
max_concepts = 20
```

**Cohort Assignment Strategy:**

```python
# Pseudo-code for cohort assignment
def assign_to_canary(user_id: str, query_hash: str) -> bool:
    """Deterministic cohort assignment using consistent hashing."""
    seed = 42
    cohort_key = f"{user_id}:{query_hash}:{seed}"
    hash_value = crc32(cohort_key.encode()) / 2**32  # Normalize to [0, 1)
    return hash_value < 0.05  # 5% sampling rate
```

This ensures:
- Same user/query combination always gets same treatment (A/B test validity)
- ~5% of queries use blended recall, 95% use episodic-only (control group)
- Reproducible cohort assignment for debugging

### 2.2 Canary Deployment Commands

```bash
# Kubernetes: Deploy canary variant alongside stable
kubectl apply -f deployments/kubernetes/engram-canary-5pct.yaml

# Verify both deployments running
kubectl get deployments -l app=engram
# Expected: engram-stable (replicas=3), engram-canary (replicas=1)

# Configure traffic split (95/5) using service mesh or ingress
kubectl apply -f deployments/kubernetes/engram-traffic-split.yaml

# Docker: Run canary container with cohort config
docker run -d \
  --name engram-canary \
  -p 7433:7432 \
  -v $(pwd)/config/production-canary.toml:/config/engram.toml:ro \
  engram:milestone-17-canary

# Update load balancer to send 5% traffic to port 7433
```

### 2.3 Nightly Performance Comparison

Run automated nightly comparison between canary and control:

```bash
# Cron job: Run every night at 2 AM
0 2 * * * /usr/local/bin/compare_canary_control.sh

# compare_canary_control.sh
#!/bin/bash
set -e

DATE=$(date +%Y%m%d)
DURATION=3600  # 1-hour test

# Test control group (blended_recall=OFF)
./scripts/m17_performance_check.sh canary_control before
# ... wait for completion ...

# Test canary group (blended_recall=ON, 5%)
./scripts/m17_performance_check.sh canary_treatment after

# Compare results
./scripts/compare_m17_performance.sh canary > tmp/m17_performance/canary_${DATE}_comparison.txt

# Check for regressions
if [ $? -eq 1 ]; then
    # Regression detected - send alert
    curl -X POST $SLACK_WEBHOOK \
      -d "{\"text\": \"Canary regression detected on ${DATE}. Check tmp/m17_performance/canary_${DATE}_comparison.txt\"}"
fi

# Store results for statistical analysis
cp tmp/m17_performance/canary_${DATE}_comparison.txt \
   metrics/canary_history/
```

### 2.4 Metrics to Track (Canary vs Control)

**Primary Metrics (Regression Detection):**

1. **P99 Recall Latency**
   - Canary target: <15ms (5ms overhead vs control's ~10ms)
   - Query: `histogram_quantile(0.99, rate(engram_recall_duration_seconds_bucket{cohort="canary"}[5m]))`
   - Acceptable increase: <5% vs control group

2. **Recall Accuracy (Top-K Overlap)**
   - Metric: `engram_recall_topk_overlap_ratio{cohort="canary"}`
   - Target: >0.85 overlap with control group for same queries
   - This validates that blended recall doesn't degrade precision

3. **Throughput**
   - Metric: `rate(engram_recall_total{cohort="canary"}[1m])`
   - Target: >95% of control group throughput
   - Acceptable decrease: <5%

**Secondary Metrics (Value Validation):**

4. **Recall Robustness (Partial Cue Performance)**
   - Metric: `engram_blended_recall_pattern_completed_total`
   - Target: Canary returns >20% more results for partial/noisy cues
   - Validates core value proposition of semantic pathway

5. **Convergent Retrieval Rate**
   - Metric: `engram_blended_recall_convergent_ratio`
   - Target: >30% of canary results are convergent (both pathways agree)
   - High convergence = high confidence in blended results

6. **Semantic Pathway Timeout Rate**
   - Metric: `engram_semantic_pathway_timeout_total / engram_blended_recall_total`
   - Target: <10% timeout rate
   - High timeouts suggest semantic pathway is too slow

**Dashboard Panels:**

Create "Canary vs Control" dashboard with side-by-side comparisons:

```yaml
# Grafana dashboard JSON snippet
{
  "panels": [
    {
      "title": "P99 Latency: Canary vs Control",
      "targets": [
        {
          "expr": "histogram_quantile(0.99, rate(engram_recall_duration_seconds_bucket{cohort=\"canary\"}[5m]))",
          "legendFormat": "Canary (blended)"
        },
        {
          "expr": "histogram_quantile(0.99, rate(engram_recall_duration_seconds_bucket{cohort=\"control\"}[5m]))",
          "legendFormat": "Control (episodic-only)"
        }
      ],
      "alert": {
        "conditions": [
          {
            "evaluator": { "type": "gt", "params": [1.05] },
            "query": { "params": ["A", "5m", "now"] },
            "reducer": { "type": "avg" },
            "type": "query"
          }
        ],
        "name": "Canary P99 Latency Regression"
      }
    }
  ]
}
```

### 2.5 Statistical Significance Testing

Run statistical tests weekly to determine if canary improvements are significant:

```python
# scripts/analyze_canary_significance.py
import pandas as pd
from scipy import stats

# Load 7 days of canary vs control data
canary_latencies = pd.read_csv('metrics/canary_p99_latencies.csv')
control_latencies = pd.read_csv('metrics/control_p99_latencies.csv')

# Welch's t-test (unequal variances)
t_stat, p_value = stats.ttest_ind(canary_latencies, control_latencies, equal_var=False)

print(f"T-statistic: {t_stat:.4f}")
print(f"P-value: {p_value:.4f}")

if p_value < 0.05:
    if canary_latencies.mean() > control_latencies.mean():
        print("REGRESSION: Canary is significantly slower (p<0.05)")
        exit(1)
    else:
        print("IMPROVEMENT: Canary is significantly faster (p<0.05)")
else:
    print("NO SIGNIFICANT DIFFERENCE: Continue monitoring")

# Compute effect size (Cohen's d)
pooled_std = np.sqrt((canary_latencies.std()**2 + control_latencies.std()**2) / 2)
cohens_d = (canary_latencies.mean() - control_latencies.mean()) / pooled_std
print(f"Effect size (Cohen's d): {cohens_d:.4f}")
```

**Proceed to Phase 3 If:**
- p-value >0.05 (no significant regression) OR p-value <0.05 AND canary faster
- Effect size |d| <0.3 (small effect) for latency differences
- Canary recall robustness >20% better for partial cues (user value demonstrated)

### 2.6 Success Criteria (Canary)

**Regression Prevention:**
- [ ] P99 latency increase <5% vs control (p>0.05 in t-test)
- [ ] Throughput decrease <5% vs control
- [ ] Error rate increase <0.5 percentage points vs control
- [ ] Memory footprint increase <15% vs Phase 1 shadow mode

**Value Validation:**
- [ ] Recall robustness improvement >20% for partial cues (Cohen's d >0.5)
- [ ] Convergent retrieval rate >30% (both pathways agree)
- [ ] Pattern completion activates for <10% of queries (not overused)
- [ ] Top-K overlap with control >85% (precision maintained)

**Operational:**
- [ ] No canary-specific crashes or panics
- [ ] Semantic pathway timeout rate <10%
- [ ] Concept quality violations <1% of formed concepts
- [ ] Grafana "Canary vs Control" dashboard shows clear comparison

### 2.7 Abort Triggers (Canary)

Immediately rollback to Phase 1 (shadow mode) if:

1. **Latency regression:** Canary P99 latency >10% higher than control for 3 consecutive hours
2. **Accuracy degradation:** Top-K overlap with control <75% (20% precision loss unacceptable)
3. **Crash increase:** Canary crash rate >2x control crash rate
4. **User complaints:** >5 user-reported issues specifically about canary cohort behavior
5. **Statistical significance:** p<0.01 for canary being slower (very strong evidence of regression)

**Rollback Procedure:**

```bash
# Disable blended_recall feature flag for canary cohort
kubectl set env deployment/engram-canary BLENDED_RECALL=false FAN_EFFECT=false

# OR revert to shadow mode configuration
kubectl rollout undo deployment/engram-canary

# Verify rollback
curl http://localhost:7433/api/v1/admin/features | jq '.blended_recall'
# Expected: false

# Monitor canary stabilization (15 minutes)
# Ensure P99 latency drops back to control levels
```

## Phase 3: Gradual Ramp to 25% Traffic

**Duration:** 1 week
**Objective:** Increase blended recall traffic to 25% while maintaining SLOs. Begin chaos engineering validation.

### 3.1 Traffic Ramp Configuration

Update cohort sampling rate from 5% to 25%:

```toml
# config/production-ramp-25pct.toml

[blended_recall]
cohort_sampling_rate = 0.25    # Increase from 0.05 to 0.25
cohort_seed = 42                # Keep same seed for consistency
# ... other settings unchanged ...
```

**Deployment:**

```bash
# Kubernetes: Update traffic split to 25/75
kubectl apply -f deployments/kubernetes/engram-traffic-split-25pct.yaml

# Verify traffic distribution
kubectl get virtualservice engram -o yaml | grep weight
# Expected:
#   weight: 75  # control
#   weight: 25  # canary (blended)

# Monitor during ramp (first 2 hours are critical)
watch -n 60 'curl -s http://localhost:7432/metrics | grep engram_recall_total'
```

### 3.2 Monitoring at Each Ramp Stage

**Critical Period: First 2 Hours After Ramp**

During traffic increase, monitor these metrics every 5 minutes:

1. **P99 Latency Spike Detection**
   - Query: `rate(engram_recall_duration_seconds_bucket{le="0.015"}[1m])`
   - Alert if <95% of requests complete within 15ms for 10 minutes

2. **Error Rate Increase**
   - Query: `rate(engram_recall_errors_total[5m])`
   - Alert if >0.5% error rate sustained for 15 minutes

3. **Circuit Breaker Trips**
   - Query: `engram_spreading_breaker_state`
   - Alert if state=1 (open) for any instance

4. **Memory Pressure**
   - Query: `process_resident_memory_bytes`
   - Alert if >80% of available RAM on any node

**Stabilization Period: 2-24 Hours**

After initial 2 hours, monitor daily:

```bash
# Daily health check script
#!/bin/bash
./scripts/compare_m17_performance.sh ramp25 > tmp/m17_performance/ramp25_$(date +%Y%m%d).txt

# Check for regressions
if [ $? -eq 1 ]; then
    echo "WARNING: Regression detected at 25% traffic"
    # Alert on-call
fi

# Extract key metrics
jq '.p99_latency_ms, .throughput_ops_per_sec, .error_rate' \
  tmp/m17_performance/ramp25_$(date +%Y%m%d).json
```

### 3.3 Chaos Engineering Drills

Validate system resilience under failure conditions:

#### Drill 1: Concept Formation Failure Injection

```bash
# Simulate concept formation failures
kubectl exec -it engram-canary-0 -- /bin/sh -c \
  "curl -X POST http://localhost:7432/api/v1/admin/chaos/consolidation/fail_rate \
   -d '{\"fail_rate\": 0.1}'"  # 10% failure rate

# Monitor fallback behavior
# Expected: System falls back to episodic-only recall, no user-visible errors

# Verify metrics
curl -s http://localhost:7432/metrics | grep engram_consolidation_failures_total

# Restore normal operation
kubectl exec -it engram-canary-0 -- /bin/sh -c \
  "curl -X POST http://localhost:7432/api/v1/admin/chaos/consolidation/fail_rate \
   -d '{\"fail_rate\": 0.0}'"
```

#### Drill 2: Semantic Pathway Timeout Simulation

```bash
# Inject latency into semantic pathway
kubectl exec -it engram-canary-0 -- /bin/sh -c \
  "curl -X POST http://localhost:7432/api/v1/admin/chaos/semantic_pathway/latency \
   -d '{\"latency_ms\": 50}'"  # Force 50ms latency (exceeds 8ms timeout)

# Monitor timeout handling
# Expected: Semantic pathway times out, falls back to episodic-only

# Verify timeout metrics
curl -s http://localhost:7432/metrics | grep engram_semantic_pathway_timeout_total

# Restore normal operation
kubectl exec -it engram-canary-0 -- /bin/sh -c \
  "curl -X POST http://localhost:7432/api/v1/admin/chaos/semantic_pathway/latency \
   -d '{\"latency_ms\": 0}'"
```

#### Drill 3: Binding Index Corruption

```bash
# Simulate binding index corruption (invalid strengths)
kubectl exec -it engram-canary-0 -- /bin/sh -c \
  "curl -X POST http://localhost:7432/api/v1/admin/chaos/binding_index/corrupt \
   -d '{\"corruption_rate\": 0.05}'"  # Corrupt 5% of bindings

# Monitor data quality alerts
# Expected: ConceptQualityViolation alert fires, corrupted bindings filtered out

# Verify quality metrics
curl -s http://localhost:7432/metrics | grep engram_binding_quality_violations_total

# Restore from backup
kubectl exec -it engram-canary-0 -- /bin/sh -c \
  "curl -X POST http://localhost:7432/api/v1/admin/chaos/binding_index/restore"
```

### 3.4 SLO Targets (25% Traffic)

**Service Level Objectives (Must Remain Green):**

| Metric | SLO | Measurement Window | Alert Threshold |
|--------|-----|-------------------|-----------------|
| Recall P99 Latency | <15ms | 5-minute rolling | >15ms for 10min |
| Recall Error Rate | <0.1% | 5-minute rolling | >0.5% for 5min |
| Availability | >99.9% | 24-hour window | <99.5% in 24h |
| Memory Utilization | <80% | 1-hour rolling | >85% for 30min |
| Consolidation Success | >99% | 1-hour rolling | <95% for 1h |

**Grafana SLO Dashboard:**

```promql
# Recall latency SLO (% of requests <15ms)
sum(rate(engram_recall_duration_seconds_bucket{le="0.015"}[5m]))
/
sum(rate(engram_recall_duration_seconds_count[5m]))

# Error rate SLO (% of requests that succeed)
1 - (
  sum(rate(engram_recall_errors_total[5m]))
  /
  sum(rate(engram_recall_total[5m]))
)

# Availability SLO (% uptime in 24h)
avg_over_time(up{job="engram"}[24h])
```

### 3.5 Success Criteria (25% Traffic)

**Performance:**
- [ ] P99 latency <15ms sustained for entire week (SLO met)
- [ ] Error rate <0.1% sustained for entire week
- [ ] Throughput degradation <5% vs Phase 0 baseline
- [ ] Memory footprint <30% increase vs Phase 0 baseline

**Resilience:**
- [ ] All 3 chaos drills passed (graceful degradation, no crashes)
- [ ] System recovered from concept formation failures within 5 minutes
- [ ] Semantic pathway timeouts handled gracefully (fallback to episodic)
- [ ] Binding index corruption detected and filtered automatically

**Operational:**
- [ ] Zero production incidents attributed to dual-memory code
- [ ] On-call pages <2 per week related to dual-memory alerts
- [ ] Grafana SLO dashboard shows >99.9% SLO compliance
- [ ] No user-reported issues specific to dual-memory behavior

### 3.6 Abort Triggers (25% Traffic)

Rollback to Phase 2 (5% canary) if:

1. **SLO breach:** Any SLO violated for >30 minutes during business hours
2. **Chaos drill failure:** System does not recover from any drill within 10 minutes
3. **Incident spike:** >3 production incidents in 1 week related to dual-memory
4. **Memory leak:** RSS growth >200MB/hour sustained for 6 hours
5. **Consolidation degradation:** Success rate <95% for 2 hours

**Rollback Procedure:**

```bash
# Reduce traffic back to 5%
kubectl apply -f deployments/kubernetes/engram-traffic-split-5pct.yaml

# Verify rollback
kubectl get virtualservice engram -o yaml | grep weight
# Expected: weight: 95 (control), weight: 5 (canary)

# Monitor stabilization
watch -n 60 'curl -s http://localhost:7432/api/v1/system/health | jq ".slo_compliance"'
```

## Phase 4: Gradual Ramp to 50% and 75% Traffic

**Duration:** 2 weeks (1 week per increment)
**Objective:** Continue ramping traffic to majority cohorts while maintaining SLOs.

### 4.1 50% Traffic Ramp (Week 1)

**Configuration:**

```toml
[blended_recall]
cohort_sampling_rate = 0.50    # 50% blended, 50% episodic-only
```

**Deployment:**

```bash
kubectl apply -f deployments/kubernetes/engram-traffic-split-50pct.yaml
```

**Monitoring Focus:**
- Same SLOs as Phase 3 (P99 <15ms, error rate <0.1%)
- Increased attention to memory pressure (50% more load on dual-memory paths)
- Daily performance comparisons between blended and episodic cohorts

**Success Criteria:**
- [ ] All Phase 3 SLOs maintained
- [ ] No increase in on-call pages vs Phase 3
- [ ] Memory footprint <40% increase vs Phase 0 baseline
- [ ] User value metrics improving (recall robustness >25% better for partial cues)

**Abort Trigger:**
- Rollback to Phase 3 (25%) if any SLO breached for >30 minutes

### 4.2 75% Traffic Ramp (Week 2)

**Configuration:**

```toml
[blended_recall]
cohort_sampling_rate = 0.75    # 75% blended, 25% episodic-only
```

**Deployment:**

```bash
kubectl apply -f deployments/kubernetes/engram-traffic-split-75pct.yaml
```

**Monitoring Focus:**
- Same SLOs as Phase 3
- Prepare for full rollout by validating 95th percentile of blended recall latency
- Ensure episodic-only control group (25%) remains stable for final comparison

**Chaos Drill (Repeat at 75%):**
- Re-run all 3 chaos drills from Phase 3
- Validate resilience at higher traffic levels
- Expected: Same graceful degradation behavior

**Success Criteria:**
- [ ] All Phase 3 SLOs maintained
- [ ] Chaos drills pass at 75% traffic
- [ ] Statistical analysis shows recall robustness improvement p<0.05
- [ ] Grafana dashboards show stable performance for 1 week

**Abort Trigger:**
- Rollback to Phase 3 (50%) if any SLO breached for >30 minutes

### 4.3 Ramp Schedule Summary

| Phase | Traffic % | Duration | Key Validation |
|-------|-----------|----------|----------------|
| 3 | 25% | 1 week | Chaos drills, SLO compliance |
| 4a | 50% | 1 week | Memory pressure, daily comparisons |
| 4b | 75% | 1 week | Repeat chaos drills, prepare for 100% |

**Progressive Rollout Timeline:**

```
Week 0: Phase 2 (5% canary)
Week 1: Phase 3 (25% ramp + chaos drills)
Week 2: Phase 4a (50% ramp)
Week 3: Phase 4b (75% ramp + final validation)
Week 4: Phase 5 (100% full rollout)
```

## Phase 5: Full Rollout and Post-Mortem

**Duration:** Ongoing (full rollout) + 1 week (post-mortem analysis)
**Objective:** Enable blended recall for 100% of traffic and conduct comprehensive post-rollout analysis.

### 5.1 Full Rollout (100% Traffic)

**Configuration:**

```toml
[blended_recall]
cohort_sampling_rate = 1.0     # 100% blended recall
```

**Deployment:**

```bash
# Remove traffic split - all traffic to dual-memory variant
kubectl apply -f deployments/kubernetes/engram-production-100pct.yaml

# Verify 100% deployment
kubectl get deployments
# Expected: engram-stable (replicas=0), engram-canary (replicas=4)

# Rename canary to production
kubectl label deployment engram-canary variant=production --overwrite

# Monitor first 24 hours closely
./scripts/monitor_full_rollout.sh
```

**First 24 Hours Monitoring:**

```bash
#!/bin/bash
# scripts/monitor_full_rollout.sh

echo "Full rollout monitoring - first 24 hours"

for hour in {1..24}; do
    echo "Hour $hour of 24:"

    # Check P99 latency
    P99=$(curl -s http://localhost:9090/api/v1/query?query=histogram_quantile\(0.99,rate\(engram_recall_duration_seconds_bucket[5m]\)\) | jq -r '.data.result[0].value[1]')
    echo "  P99 latency: ${P99}s"

    # Check error rate
    ERROR_RATE=$(curl -s http://localhost:9090/api/v1/query?query=rate\(engram_recall_errors_total[5m]\)/rate\(engram_recall_total[5m]\) | jq -r '.data.result[0].value[1]')
    echo "  Error rate: ${ERROR_RATE}"

    # Check memory
    MEMORY=$(curl -s http://localhost:9090/api/v1/query?query=process_resident_memory_bytes | jq -r '.data.result[0].value[1]')
    MEMORY_GB=$(echo "scale=2; $MEMORY / 1024 / 1024 / 1024" | bc)
    echo "  Memory: ${MEMORY_GB} GB"

    # Alert if any SLO breached
    if (( $(echo "$P99 > 0.015" | bc -l) )); then
        echo "  ALERT: P99 latency exceeded 15ms!"
    fi

    if (( $(echo "$ERROR_RATE > 0.001" | bc -l) )); then
        echo "  ALERT: Error rate exceeded 0.1%!"
    fi

    sleep 3600  # Wait 1 hour
done

echo "24-hour monitoring complete. Check Grafana for detailed analysis."
```

### 5.2 Final Performance Validation

Run comprehensive performance benchmarks at 100% traffic:

```bash
# 60-second production load test
./scripts/m17_performance_check.sh full_rollout after

# Compare against Phase 0 baseline
./scripts/compare_m17_performance.sh full_rollout

# Expected output:
# Metric               Before     After      Change
# P50 latency (ms)     4.2        4.8        +14.3%
# P95 latency (ms)     8.1        9.5        +17.3%
# P99 latency (ms)     12.3       14.2       +15.4%  # <5% target: FAILED
# Throughput (ops/s)   10500      10200      -2.9%   # <5% target: PASSED
# Errors               0          0          +0
# Error rate           0.0%       0.0%       +0.0pp

# [OK] No significant regressions detected (within 5% threshold)
```

**If Regression >5% Detected:**
- Profile with `cargo flamegraph --bin engram` to identify hot spots
- Review Grafana "Dual Memory Architecture Overview" for bottlenecks
- Consider optimizations (SIMD vectorization, cache tuning)
- Re-validate after optimizations

### 5.3 Post-Rollout Analysis Checklist

Complete within 1 week of 100% rollout:

#### Performance Analysis

- [ ] **Baseline Comparison:** Document final performance vs Phase 0 baseline
  - P99 latency delta: _____ ms (target: <5% increase)
  - Throughput delta: _____ ops/s (target: <5% decrease)
  - Memory footprint delta: _____ MB (document absolute increase)

- [ ] **Latency Distribution Analysis:**
  - Compare P50/P90/P95/P99 percentiles between episodic-only (Phase 0) and blended (Phase 5)
  - Identify which percentiles regressed most
  - Determine if regression is acceptable given value add

- [ ] **User Value Validation:**
  - Recall robustness improvement for partial cues: _____ % (target: >20%)
  - Convergent retrieval rate: _____ % (target: >30%)
  - Pattern completion usage rate: _____ % (target: <10%)
  - User-reported issues: _____ (target: <5 in first week)

#### Operational Analysis

- [ ] **Incident Review:**
  - Total incidents during rollout: _____
  - Incidents attributed to dual-memory: _____
  - Mean time to resolution (MTTR): _____ minutes
  - Lessons learned from each incident

- [ ] **On-Call Impact:**
  - On-call pages during rollout: _____
  - Pages after midnight: _____ (assess operator fatigue)
  - False positive alert rate: _____ %
  - Alert tuning recommendations

- [ ] **Resource Utilization:**
  - Peak CPU usage: _____ % (capacity planning)
  - Peak memory usage: _____ GB
  - Disk I/O increase: _____ %
  - Network bandwidth increase: _____ %

#### Data Quality Analysis

- [ ] **Concept Formation Quality:**
  - Total concepts formed: _____
  - Average concept coherence: _____ (target: >0.7)
  - Concept quality violations: _____ % (target: <1%)
  - Concept lifecycle (formation to GC): _____ days

- [ ] **Binding Index Health:**
  - Total bindings created: _____
  - Binding strength distribution (P50/P90/P99): _____
  - Binding age distribution: _____
  - Binding GC rate: _____ bindings/hour

#### Cost Analysis

- [ ] **Infrastructure Cost:**
  - Compute cost increase: _____ % (due to CPU/memory increase)
  - Storage cost increase: _____ % (due to concepts/bindings)
  - Total cost delta: $ _____ /month

- [ ] **Engineering Cost:**
  - Total engineering hours for rollout: _____ hours
  - On-call hours for incident response: _____ hours
  - Documentation/training hours: _____ hours

### 5.4 Lessons Learned Documentation

Create comprehensive lessons learned document:

```markdown
# Dual Memory Architecture Rollout - Lessons Learned

## Date
2025-11-XX to 2025-12-XX

## Summary
Successfully rolled out dual-memory architecture to 100% of production traffic over 4 weeks.
Final performance: P99 latency +X%, throughput -Y%, recall robustness +Z%.

## What Went Well
1. Phased rollout approach prevented major incidents
2. Feature flags enabled quick rollbacks
3. Grafana dashboards provided excellent visibility
4. Chaos drills caught resilience issues early
5. Statistical A/B testing validated user value

## What Went Wrong
1. [Incident 1]: Brief P99 latency spike during 50% ramp
   - Root cause: Semantic pathway timeout too aggressive
   - Fix: Increased timeout from 8ms to 10ms
   - Prevention: Add latency percentile alerts earlier in rollout

2. [Incident 2]: Binding index bloat at 75% traffic
   - Root cause: GC not running frequently enough
   - Fix: Increased GC frequency from 1h to 30m
   - Prevention: Add binding index size alerts

## Recommendations for Future Rollouts
1. Start with 1% canary (not 5%) for new cognitive features
2. Extend shadow mode to 2 weeks (not 1 week) for better baseline
3. Automate rollback triggers (currently manual decision)
4. Invest in more chaos drills (network partitions, disk failures)
5. Build better load testing scenarios (more realistic query distributions)

## Metrics Archive
- Phase 0 baseline: [link to metrics]
- Phase 2 canary: [link to metrics]
- Phase 5 full rollout: [link to metrics]

## Team Acknowledgments
- Platform team for Kubernetes infrastructure
- SRE team for 24/7 monitoring support
- Data science team for statistical analysis
- Engineering leadership for supporting phased approach
```

### 5.5 Feature Flag Cleanup

After 2 weeks of stable 100% rollout, clean up feature flags:

```bash
# Remove feature flag conditionals from code
# (dual_memory_types becomes always-on, not gated)

# Update Cargo.toml to make dual_memory_types default
[features]
default = ["dual_memory_types", "monitoring"]
dual_memory_types = []  # No longer optional
blended_recall = []      # Keep as optional for future A/B tests
fan_effect = []          # Keep as optional for future tuning

# Remove old episodic-only code paths
git rm engram-core/src/activation/episodic_only_recall.rs

# Update documentation
# Mark dual-memory as production-ready in README
```

## A/B Testing Methodology

### Cohort Assignment

**Deterministic Hashing:**

```rust
// engram-core/src/activation/cohort.rs

use std::hash::{Hash, Hasher};
use std::collections::hash_map::DefaultHasher;

pub fn assign_to_treatment(user_id: &str, query_hash: &str, treatment_rate: f64) -> bool {
    let mut hasher = DefaultHasher::new();
    format!("{}:{}:42", user_id, query_hash).hash(&mut hasher);
    let hash_value = hasher.finish();
    let normalized = (hash_value as f64) / (u64::MAX as f64);
    normalized < treatment_rate
}
```

**Properties:**
- Same user + same query always get same treatment (consistency)
- ~treatment_rate % of queries assigned to treatment
- Deterministic for reproducibility
- Seed=42 for production (change for experiments)

### Metrics to Compare

| Metric | Treatment (Blended) | Control (Episodic) | Target |
|--------|---------------------|-------------------|--------|
| P99 Latency | `histogram_quantile(0.99, rate(engram_recall_duration_seconds_bucket{cohort="treatment"}[5m]))` | `histogram_quantile(0.99, rate(engram_recall_duration_seconds_bucket{cohort="control"}[5m]))` | <5% increase |
| Recall Count | `avg(engram_recall_result_count{cohort="treatment"})` | `avg(engram_recall_result_count{cohort="control"})` | >20% increase for partial cues |
| Top-K Overlap | `engram_recall_topk_overlap{cohort="treatment"}` | N/A (self-comparison) | >0.85 |
| Convergent Rate | `engram_blended_recall_convergent_ratio` | N/A (episodic has no convergence) | >0.30 |

### Statistical Significance Thresholds

**T-Test Parameters:**
- Alpha (significance level): 0.05 (5% chance of false positive)
- Power (1-beta): 0.80 (80% chance of detecting real effect)
- Minimum sample size: 1000 queries per cohort per day
- Test duration: 7 days (minimum) for weekly seasonality

**Effect Size Guidelines (Cohen's d):**
- d <0.2: Small effect (may not be worth rollout cost)
- d 0.2-0.5: Medium effect (proceed if other metrics okay)
- d >0.5: Large effect (strong evidence for rollout)

**Confidence Intervals:**
- Report 95% CI for all metrics
- Example: "P99 latency increased 3.2% (95% CI: [1.8%, 4.6%])"

### Test Duration Requirements

Minimum test duration by cohort size:

| Cohort Size | Min Duration | Rationale |
|-------------|--------------|-----------|
| 5% | 7 days | Weekly seasonality, sufficient sample |
| 25% | 3 days | Larger sample, faster significance |
| 50% | 2 days | Half of traffic, quick validation |
| 75% | 2 days | Majority traffic, final check |

**Early Stopping Criteria:**
- If p<0.001 for regression, stop immediately (very strong evidence of harm)
- If p<0.01 for improvement, can accelerate ramp (strong evidence of benefit)
- If CI upper bound <0% for latency, definitely proceed (confident no regression)

## Rollback Procedures

### Immediate Rollback (Feature Flag Toggle)

**Fastest rollback method** (0-5 minutes):

```bash
# Disable blended_recall feature flag via API
curl -X POST http://localhost:7432/api/v1/admin/features \
  -H "Content-Type: application/json" \
  -d '{"blended_recall": false, "fan_effect": false}'

# Verify rollback
curl http://localhost:7432/api/v1/admin/features | jq '.blended_recall, .fan_effect'
# Expected: false, false

# Monitor recovery (latency should drop within 2 minutes)
watch -n 10 'curl -s http://localhost:9090/api/v1/query?query=histogram_quantile\(0.99,rate\(engram_recall_duration_seconds_bucket[1m]\)\) | jq -r ".data.result[0].value[1]"'
```

**Expected Recovery Time:**
- Latency recovery: <2 minutes (feature flag checked per-request)
- Memory recovery: <15 minutes (semantic pathway state GC'd)
- Full stabilization: <30 minutes

### Partial Rollback (Reduce Traffic Percentage)

**Reduce traffic** instead of full disable (5-15 minutes):

```bash
# Kubernetes: Reduce treatment traffic from X% to Y%
kubectl apply -f deployments/kubernetes/engram-traffic-split-${NEW_PERCENTAGE}pct.yaml

# Example: Rollback from 50% to 25%
kubectl apply -f deployments/kubernetes/engram-traffic-split-25pct.yaml

# Verify traffic distribution
kubectl get virtualservice engram -o yaml | grep weight

# Monitor impact (expect proportional latency reduction)
```

### Full Rollback (Revert to Previous Milestone)

**Complete revert** to pre-dual-memory code (30-60 minutes):

```bash
# Kubernetes: Rollback to previous deployment
kubectl rollout undo deployment/engram

# OR deploy specific stable version
kubectl set image deployment/engram engram=engram:milestone-16-stable

# Verify rollback
kubectl rollout status deployment/engram
curl http://localhost:7432/api/v1/system/info | jq '.milestone'
# Expected: "16" (not "17")

# Run diagnostics
./scripts/engram_diagnostics.sh

# Monitor stabilization (expect return to Phase 0 baseline within 1 hour)
```

### Data Rollback (If Corruption Detected)

**Restore from backup** if dual-memory data is corrupted (1-4 hours):

```bash
# Stop Engram instances
kubectl scale deployment/engram --replicas=0

# Restore from backup (see docs/operations/backup-restore.md)
./scripts/restore_backup.sh \
  --timestamp "2025-11-XX-00-00-00" \
  --data-dir /data/engram

# Verify data integrity
./scripts/verify_data_integrity.sh /data/engram

# Restart Engram with dual-memory disabled
kubectl set env deployment/engram DUAL_MEMORY_TYPES=false
kubectl scale deployment/engram --replicas=4

# Monitor recovery
kubectl logs -f deployment/engram
```

### Rollback Communication Plan

**Internal Notifications:**

```bash
# Slack notification template
curl -X POST $SLACK_WEBHOOK_URL \
  -H 'Content-Type: application/json' \
  -d '{
    "text": "Dual Memory Rollback Initiated",
    "blocks": [
      {
        "type": "section",
        "text": {
          "type": "mrkdwn",
          "text": "*Dual Memory Rollback*\n*Phase:* '${CURRENT_PHASE}'\n*Reason:* '${ROLLBACK_REASON}'\n*Action:* '${ROLLBACK_ACTION}'\n*ETA:* '${RECOVERY_ETA}'"
        }
      }
    ]
  }'
```

**External Notifications (if user-impacting):**

```markdown
# Status page update template

## Service Degradation - Memory Recall
**Status:** Investigating
**Started:** 2025-11-XX HH:MM UTC
**Impact:** Some users may experience slower memory recall (15ms -> 20ms latency)

**Update (HH:MM UTC):** We have identified increased latency related to new dual-memory features and are rolling back to previous version.

**Update (HH:MM UTC):** Rollback complete. Latency has returned to normal (<15ms). We will re-evaluate the dual-memory rollout plan.

**Resolution:** Service fully restored. Post-mortem will be published within 72 hours.
```

### Rollback Testing in Staging

**Before production rollout**, validate rollback procedures:

```bash
# Staging environment
export KUBE_CONTEXT=staging

# Deploy dual-memory to staging
kubectl --context=$KUBE_CONTEXT apply -f deployments/kubernetes/engram-staging.yaml

# Simulate regression scenario
kubectl --context=$KUBE_CONTEXT exec -it engram-0 -- /bin/sh -c \
  "curl -X POST http://localhost:7432/api/v1/admin/chaos/latency -d '{\"latency_ms\": 30}'"

# Verify alert fires
# Expected: Grafana alert "P99 Latency SLO Breach" triggers within 5 minutes

# Practice rollback
time (curl -X POST http://staging.engram:7432/api/v1/admin/features \
  -d '{"blended_recall": false}')

# Verify recovery
# Expected: Latency drops to <15ms within 2 minutes

# Document rollback time
echo "Rollback time: <2 minutes" >> ROLLBACK_VALIDATION.md
```

## Monitoring and Alerting References

### Prometheus Metrics

**Dual-Memory Core Metrics:**

```promql
# Concepts formed per minute
rate(engram_concepts_formed_total[1m]) * 60

# Binding creation rate
rate(engram_bindings_created_total[1m]) * 60

# Blended recall latency P99
histogram_quantile(0.99, rate(engram_blended_recall_duration_seconds_bucket[5m]))

# Convergent retrieval ratio
engram_blended_recall_convergent_total / engram_blended_recall_total

# Semantic pathway timeout rate
rate(engram_semantic_pathway_timeout_total[5m]) / rate(engram_blended_recall_total[5m])

# Concept quality violations
rate(engram_concept_quality_violations_total[5m])

# Fan effect penalties applied
rate(engram_fan_effect_penalties_total[5m])
```

**Memory and Performance Metrics:**

```promql
# Memory footprint
process_resident_memory_bytes

# CPU utilization
rate(process_cpu_seconds_total[1m]) * 100

# Consolidation P99 duration
histogram_quantile(0.99, rate(engram_consolidation_duration_seconds_bucket[5m]))

# Recall throughput
rate(engram_recall_total[1m]) * 60
```

### Grafana Dashboards

**Primary Dashboards for Rollout:**

1. **Dual Memory Architecture Overview**
   - URL: `http://localhost:3000/d/dual-memory-overview`
   - Panels: Concept formation rate, binding dynamics, blended recall latency, convergent retrieval
   - Refresh: 30s

2. **Cognitive Metrics Tuning**
   - URL: `http://localhost:3000/d/cognitive-metrics`
   - Panels: Consolidation performance, fan effect statistics, semantic pathway health
   - Refresh: 1m

3. **Canary vs Control Comparison**
   - URL: `http://localhost:3000/d/canary-comparison`
   - Panels: Side-by-side latency, throughput, error rates, recall quality
   - Refresh: 30s

4. **SLO Compliance Dashboard**
   - URL: `http://localhost:3000/d/slo-compliance`
   - Panels: Latency SLO (% <15ms), error rate SLO (% <0.1%), availability SLO (% uptime)
   - Refresh: 1m

**Dashboard Access:**

```bash
# Port-forward to Grafana
kubectl port-forward -n monitoring svc/grafana 3000:3000

# Open in browser
open http://localhost:3000

# Login credentials (change in production!)
# Username: admin
# Password: admin
```

### Alert Rules

**Critical Alerts (Page Immediately):**

| Alert Name | Condition | Severity | Action |
|------------|-----------|----------|--------|
| `BlendedRecallLatencyP99Breach` | P99 >15ms for 10min | Critical | Investigate semantic pathway, consider rollback |
| `RecallErrorRateHigh` | Error rate >0.5% for 5min | Critical | Check logs, rollback if >1% |
| `ConsolidationFailures` | Success rate <95% for 1h | Critical | Check consolidation logs, restart if stuck |
| `MemoryLeakDetected` | RSS growth >500MB/h for 3h | Critical | Profile memory, prepare rollback |

**Warning Alerts (Slack Only):**

| Alert Name | Condition | Severity | Action |
|------------|-----------|----------|--------|
| `ConceptQualityLow` | Avg coherence <0.5 for 30min | Warning | Review concept formation tuning |
| `SemanticPathwayTimeoutHigh` | Timeout rate >20% for 15min | Warning | Increase timeout or optimize semantic code |
| `BindingIndexBloat` | Binding/episode ratio >3.0 | Warning | Tune GC parameters |
| `FanEffectExcessive` | Avg penalty >0.3 for 1h | Warning | Review fan effect thresholds |

**Alert Configuration (Prometheus):**

```yaml
# prometheus/alerts/dual_memory.yml

groups:
- name: dual_memory
  interval: 30s
  rules:
  - alert: BlendedRecallLatencyP99Breach
    expr: histogram_quantile(0.99, rate(engram_blended_recall_duration_seconds_bucket[5m])) > 0.015
    for: 10m
    labels:
      severity: critical
      component: activation
    annotations:
      summary: "Blended recall P99 latency exceeds 15ms"
      description: "P99 latency is {{ $value | humanizeDuration }}, target <15ms"
      runbook: "docs/operations/dual_memory_rollout.md#abort-triggers-canary"

  - alert: RecallErrorRateHigh
    expr: rate(engram_recall_errors_total[5m]) / rate(engram_recall_total[5m]) > 0.005
    for: 5m
    labels:
      severity: critical
      component: activation
    annotations:
      summary: "Recall error rate exceeds 0.5%"
      description: "Error rate is {{ $value | humanizePercentage }}, target <0.1%"

  - alert: ConceptQualityLow
    expr: avg(engram_concept_coherence) < 0.5
    for: 30m
    labels:
      severity: warning
      component: consolidation
    annotations:
      summary: "Concept coherence score low"
      description: "Average coherence {{ $value }}, target >0.6"

  # ... more alert rules ...
```

**On-Call Escalation:**

```
Level 1 (First 15 min): Primary on-call engineer investigates
Level 2 (15-30 min): Secondary on-call joins, consider rollback
Level 3 (30+ min): Engineering manager paged, initiate rollback
```

## Automation Scripts

### Performance Comparison Automation

**Script:** `/scripts/m17_performance_check.sh`

Already exists in codebase. Usage during rollout:

```bash
# Before rollout (Phase 0 baseline)
./scripts/m17_performance_check.sh 000 before

# After shadow mode (Phase 1)
./scripts/m17_performance_check.sh 001 after
./scripts/compare_m17_performance.sh 001

# After canary (Phase 2)
./scripts/m17_performance_check.sh 002 after
./scripts/compare_m17_performance.sh 002

# After each ramp phase
./scripts/m17_performance_check.sh 003 after  # 25%
./scripts/m17_performance_check.sh 004 after  # 50%
./scripts/m17_performance_check.sh 005 after  # 75%
./scripts/m17_performance_check.sh 006 after  # 100%
```

**Competitive Validation (Optional):**

```bash
# Run competitive scenario (vs Neo4j baseline)
./scripts/m17_performance_check.sh 006 after --competitive
./scripts/compare_m17_performance.sh 006 --competitive

# Expected: Engram P99 <27.96ms (Neo4j baseline)
```

### Soak Test Automation

**Script:** `/scripts/run_soak_test.sh` (create)

```bash
#!/bin/bash
# Automated 24-hour soak test with monitoring

set -e

PHASE=$1
DURATION=${2:-86400}  # Default 24 hours
OUTPUT_DIR="tmp/m17_performance"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

mkdir -p "$OUTPUT_DIR"

echo "=== Starting ${DURATION}s soak test for Phase ${PHASE} ==="

# Start Engram in background
cargo build --release --features dual_memory_types,monitoring
./target/release/engram start --config config/production-${PHASE}.toml &
ENGRAM_PID=$!

# Wait for startup
sleep 10
curl -f http://localhost:7432/health || exit 1

# Start load test
./target/release/loadtest run \
  --scenario scenarios/m17_baseline.toml \
  --duration ${DURATION} \
  --seed 0xDEADBEEF \
  --endpoint http://localhost:7432 \
  --output ${OUTPUT_DIR}/phase${PHASE}_soak_${TIMESTAMP}.json &

LOADTEST_PID=$!

# Monitor resources every 15 minutes
while kill -0 $LOADTEST_PID 2>/dev/null; do
    ps -p $ENGRAM_PID -o rss,vsz,%cpu,%mem >> ${OUTPUT_DIR}/phase${PHASE}_resources_${TIMESTAMP}.log
    ./scripts/engram_diagnostics.sh >> ${OUTPUT_DIR}/phase${PHASE}_diagnostics_${TIMESTAMP}.log
    sleep 900  # 15 minutes
done

# Cleanup
kill $ENGRAM_PID

echo "=== Soak test complete ==="
echo "Results: ${OUTPUT_DIR}/phase${PHASE}_soak_${TIMESTAMP}.json"
echo "Resources: ${OUTPUT_DIR}/phase${PHASE}_resources_${TIMESTAMP}.log"
```

### Feature Flag Toggle Automation

**Script:** `/scripts/toggle_feature_flag.sh` (create)

```bash
#!/bin/bash
# Safe feature flag toggle with validation

set -e

FEATURE=$1
ENABLE=$2
ENDPOINT=${3:-http://localhost:7432}

if [[ "$ENABLE" != "true" && "$ENABLE" != "false" ]]; then
    echo "Usage: $0 <feature> <true|false> [endpoint]"
    echo "Example: $0 blended_recall false"
    exit 1
fi

echo "Toggling ${FEATURE} to ${ENABLE} on ${ENDPOINT}..."

# Toggle feature flag
curl -X POST ${ENDPOINT}/api/v1/admin/features \
  -H "Content-Type: application/json" \
  -d "{\"${FEATURE}\": ${ENABLE}}"

# Verify
sleep 2
ACTUAL=$(curl -s ${ENDPOINT}/api/v1/admin/features | jq -r ".${FEATURE}")

if [[ "$ACTUAL" == "$ENABLE" ]]; then
    echo "[OK] Feature ${FEATURE} successfully set to ${ENABLE}"
else
    echo "[ERROR] Feature toggle failed. Expected ${ENABLE}, got ${ACTUAL}"
    exit 1
fi

# Monitor latency for 2 minutes
echo "Monitoring P99 latency for 2 minutes..."
for i in {1..12}; do
    P99=$(curl -s http://localhost:9090/api/v1/query?query=histogram_quantile\(0.99,rate\(engram_recall_duration_seconds_bucket[1m]\)\) | jq -r '.data.result[0].value[1]')
    echo "  ${i}/12: P99 = ${P99}s"
    sleep 10
done

echo "Feature toggle complete. Check Grafana for detailed monitoring."
```

### Monitoring Stack Validation

**Script:** `/scripts/validate_monitoring.sh` (create)

```bash
#!/bin/bash
# Validate monitoring infrastructure before rollout

set -e

echo "=== Validating Monitoring Stack ==="

# Check Prometheus
echo "1. Checking Prometheus..."
PROM_STATUS=$(curl -s http://localhost:9090/-/healthy)
if [[ "$PROM_STATUS" != "Prometheus is Healthy." ]]; then
    echo "[ERROR] Prometheus unhealthy"
    exit 1
fi
echo "[OK] Prometheus healthy"

# Check Prometheus targets
ENGRAM_UP=$(curl -s http://localhost:9090/api/v1/targets | jq '.data.activeTargets[] | select(.labels.job=="engram") | .health' -r)
if [[ "$ENGRAM_UP" != "up" ]]; then
    echo "[ERROR] Engram target not up in Prometheus"
    exit 1
fi
echo "[OK] Engram target up"

# Check Grafana
echo "2. Checking Grafana..."
GRAFANA_STATUS=$(curl -s -u admin:admin http://localhost:3000/api/health | jq -r '.database')
if [[ "$GRAFANA_STATUS" != "ok" ]]; then
    echo "[ERROR] Grafana unhealthy"
    exit 1
fi
echo "[OK] Grafana healthy"

# Check dual-memory dashboard exists
DASHBOARD=$(curl -s -u admin:admin http://localhost:3000/api/dashboards/uid/dual-memory-overview | jq -r '.dashboard.title')
if [[ "$DASHBOARD" != "Dual Memory Architecture Overview" ]]; then
    echo "[ERROR] Dual-memory dashboard not found"
    exit 1
fi
echo "[OK] Dual-memory dashboard exists"

# Check alert rules loaded
echo "3. Checking alert rules..."
ALERT_COUNT=$(curl -s http://localhost:9090/api/v1/rules | jq '.data.groups[] | select(.name=="dual_memory") | .rules | length')
if [[ "$ALERT_COUNT" -lt 6 ]]; then
    echo "[ERROR] Dual-memory alert rules incomplete (found ${ALERT_COUNT}, expected >=6)"
    exit 1
fi
echo "[OK] ${ALERT_COUNT} alert rules loaded"

# Check dual-memory metrics exist
echo "4. Checking dual-memory metrics..."
CONCEPTS_METRIC=$(curl -s http://localhost:7432/metrics | grep -c engram_concepts_formed_total || true)
if [[ "$CONCEPTS_METRIC" -eq 0 ]]; then
    echo "[ERROR] Dual-memory metrics not exported"
    exit 1
fi
echo "[OK] Dual-memory metrics exported"

echo "=== Monitoring validation complete ==="
```

## Pre-Deployment Checklist

Complete before starting Phase 0:

### Infrastructure

- [ ] Kubernetes cluster v1.28+ with sufficient capacity (4 nodes, 16 CPU, 64GB RAM)
- [ ] Persistent volumes provisioned (20Gi per instance)
- [ ] Monitoring stack deployed (Prometheus, Grafana, Loki)
- [ ] Backup/restore procedures tested and documented
- [ ] Load balancer configured with health checks

### Code and Configuration

- [ ] Dual-memory feature merged to main branch (Tasks 001-014 complete)
- [ ] All integration tests passing (`cargo test --features dual_memory_types`)
- [ ] Clippy warnings resolved (`make quality`)
- [ ] Configuration files prepared for each phase (shadow, canary, ramp)
- [ ] Feature flag defaults set correctly (dual_memory_types=OFF initially)

### Monitoring and Alerting

- [ ] Grafana dashboards imported and accessible
- [ ] Prometheus alert rules loaded and validated
- [ ] PagerDuty/Slack integrations tested
- [ ] On-call rotation confirmed for rollout period
- [ ] Runbook (this document) reviewed by entire team

### Testing and Validation

- [ ] Performance baseline established (Phase 0 metrics captured)
- [ ] Soak test passed (24 hours, zero crashes)
- [ ] Rollback procedures tested in staging
- [ ] Chaos drills designed and ready to execute
- [ ] Statistical analysis scripts validated

### Communication and Documentation

- [ ] Rollout schedule communicated to stakeholders
- [ ] Status page prepared for user communications
- [ ] Internal Slack channel created (#dual-memory-rollout)
- [ ] Post-mortem template prepared
- [ ] Lessons learned document initialized

## Per-Phase Go/No-Go Decision Checklist

Before transitioning to next phase, verify:

### All Phases

- [ ] No critical production incidents in past 48 hours
- [ ] On-call team available for next phase
- [ ] Grafana dashboards showing healthy metrics
- [ ] Backup completed within last 24 hours
- [ ] Stakeholders notified of upcoming phase transition

### Phase 0 -> Phase 1 (Shadow Mode)

- [ ] 24-hour soak test passed
- [ ] Performance baseline documented
- [ ] Integration tests 100% passing
- [ ] Monitoring stack validated

### Phase 1 -> Phase 2 (Canary)

- [ ] Shadow mode ran for 1 week without issues
- [ ] Memory footprint increase <10% vs baseline
- [ ] Consolidation latency increase <20% vs baseline
- [ ] Concept quality >95% coherence >0.6

### Phase 2 -> Phase 3 (25% Ramp)

- [ ] Canary ran for 1 week without regressions
- [ ] Statistical analysis shows p>0.05 (no significant regression)
- [ ] User value demonstrated (recall robustness >20% better)
- [ ] Top-K overlap >85% (precision maintained)

### Phase 3 -> Phase 4a (50% Ramp)

- [ ] 25% traffic ran for 1 week with SLOs met
- [ ] All chaos drills passed
- [ ] Zero production incidents attributed to dual-memory
- [ ] Memory footprint <30% increase vs baseline

### Phase 4a -> Phase 4b (75% Ramp)

- [ ] 50% traffic ran for 1 week with SLOs met
- [ ] Daily performance comparisons stable
- [ ] On-call pages <2 per week related to dual-memory

### Phase 4b -> Phase 5 (100% Rollout)

- [ ] 75% traffic ran for 1 week with SLOs met
- [ ] Chaos drills re-run successfully at 75% traffic
- [ ] Engineering leadership approves full rollout
- [ ] Communication plan ready for 100% announcement

## Rollback Decision Criteria

Trigger rollback if ANY of the following occur:

### Automated Rollback (No Human Decision Needed)

- [ ] P99 latency >20ms sustained for 15 minutes (33% regression)
- [ ] Error rate >1% sustained for 10 minutes (10x target)
- [ ] Engram process crash or panic logged
- [ ] Memory footprint >90% of node capacity

### Manual Rollback (Requires On-Call Decision)

- [ ] P99 latency >15ms sustained for 30 minutes (breach SLO)
- [ ] Error rate >0.5% sustained for 20 minutes (5x target)
- [ ] Consolidation success rate <90% for 1 hour
- [ ] >3 user-reported issues specific to dual-memory in 24 hours
- [ ] On-call pages >5 in 24 hours related to dual-memory

### Strategic Rollback (Requires Engineering Manager Decision)

- [ ] User value not demonstrated after 2 weeks of canary (no improvement in recall robustness)
- [ ] Infrastructure cost increase >50% vs budget
- [ ] Engineering team velocity degraded due to incident load
- [ ] Alternative approach identified that's superior

---

**End of Runbook**

For questions or issues during rollout:
- Slack: #dual-memory-rollout
- On-call: PagerDuty escalation policy
- Engineering lead: [contact info]
- Runbook updates: Submit PR to docs/operations/dual_memory_rollout.md
