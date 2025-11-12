# Task 015: Production Validation and Rollout

## Objective

Validate the dual memory architecture in production through rigorously phased rollout with quantitative success criteria, automated A/B testing, comprehensive monitoring, and documented rollback procedures. Ensure zero data loss and acceptable performance bounds throughout deployment.

## Background

Production validation is the final gate before declaring Milestone 17 complete. This is not a "deploy and monitor" task - it is a multi-phase campaign with explicit success criteria, automated telemetry, and prepared rollback triggers.

The dual memory system introduces:
- Episodic-to-semantic consolidation with concept formation
- Blended recall mixing episode and concept activations
- New storage patterns with binding relationships
- Changed query execution paths with dual memory traversal

Each of these changes represents potential failure modes that must be validated under real production workloads before full deployment. This task specifies exactly how to roll out, what to measure, when to proceed, and when to abort.

## Requirements

1. Design phased rollout plan with explicit duration, success criteria, and rollback triggers for each phase
2. Implement A/B testing framework with statistical rigor for treatment vs control comparison
3. Define quantitative validation checklist with performance budgets and error thresholds
4. Create operational runbook covering common failure modes and recovery procedures
5. Document migration procedures with step-by-step upgrade paths and verification
6. Specify team training requirements for operators and developers
7. Design post-rollout analysis framework for continuous improvement

## Technical Specification

### Architecture Context

This task builds on completed Milestone 17 tasks:
- Task 001: DualMemoryNode type system with episode/concept discrimination
- Task 002: Backend adaptation for type-aware storage
- Task 003: Migration utilities for converting pure-episodic graphs
- Task 004: Concept formation engine with clustering
- Task 005: Binding formation between episodes and concepts
- Task 006: Consolidation integration with background formation
- Task 008: Hierarchical spreading activation across memory types
- Task 009: Blended recall combining episode and concept results
- Task 010: Confidence propagation through dual memory operations
- Task 011: Psychological validation against human memory data

Production validation assumes all prior tasks are complete and passing acceptance tests.

### Files to Create

#### `/Users/jordanwashburn/Workspace/orchard9/engram/docs/operations/dual_memory_rollout.md`

Comprehensive rollout guide following existing operations documentation patterns:

```markdown
# Dual Memory Architecture - Production Rollout

## Overview

This document specifies the phased rollout plan for dual memory architecture (Milestone 17) from shadow mode through full production deployment. Every phase includes explicit success criteria, duration, monitoring dashboards, and rollback triggers.

Recovery Time Objective (RTO): 15 minutes for emergency rollback to pure-episodic mode
Recovery Point Objective (RPO): Zero data loss - all episodes preserved

## Rollout Phases

### Phase 0: Pre-Deployment Validation (Complete Before Phase 1)

**Duration**: 1 week

**Objective**: Verify all prerequisites are met before touching production

**Actions**:

1. Verify all Milestone 17 tasks marked as complete:
   ```bash
   ls roadmap/milestone-17/*_complete.md | wc -l
   # Expected: 14 (tasks 001-014 complete)
   ```

2. Run full integration test suite with dual memory features:
   ```bash
   cargo test --workspace --features dual_memory_types --release
   # Expected: All tests pass, zero failures
   ```

3. Run 24-hour soak test in staging:
   ```bash
   ./scripts/soak_test_dual_memory.sh staging 24h
   # Monitor: Memory leaks, consolidation cycles, binding formation
   ```

4. Benchmark performance vs baseline (from Task 011 validation):
   ```bash
   cargo bench --bench dual_memory_validation
   # Verify: <5% regression on existing operations
   #         Concept formation <100ms P99
   #         Blended recall <10ms P99 delta
   ```

5. Verify monitoring infrastructure is ready:
   - Grafana dashboard for dual memory metrics deployed
   - Prometheus alerts configured (ConceptFormationStuck, BindingLeaks, BlendedRecallSlow)
   - Log aggregation capturing concept formation events
   - Baseline metrics recorded for comparison

6. Train operations team on rollback procedures (see Team Training section)

7. Schedule deployment windows with stakeholder approval

**Success Criteria**:
- [ ] All integration tests pass with dual memory features enabled
- [ ] 24-hour soak test shows no memory leaks (RSS growth <1% per hour)
- [ ] Performance benchmarks within acceptable bounds (<5% regression)
- [ ] Monitoring dashboards operational and showing clean baselines
- [ ] Operations team trained and signed off on runbook
- [ ] Deployment windows scheduled and approved

**Abort Criteria**: Any success criterion fails - block Phase 1 until resolved

### Phase 1: Shadow Mode (2 weeks)

**Duration**: 2 weeks minimum (extend if issues found)

**Objective**: Enable concept formation and binding in background without affecting user-visible behavior. Zero risk to production traffic.

**Configuration**:
```rust
DualMemoryConfig {
    // Enable episode-to-dual-node conversion on store
    enable_dual_types: true,

    // Background concept formation active
    enable_concept_formation: true,
    formation_interval: Duration::from_secs(3600), // Hourly

    // DO NOT use concepts for recall yet
    enable_blended_recall: false,

    // Conservative formation thresholds
    coherence_threshold: 0.85,        // High bar for concept quality
    min_cluster_size: 10,              // Require substantial evidence
    max_concepts_per_cycle: 100,       // Limit resource usage

    // Full observability
    enable_formation_logging: true,
    enable_metrics: true,
}
```

**Deployment**:
1. Deploy feature-flagged binary to production:
   ```bash
   cargo build --release --features dual_memory_types
   ./scripts/deploy_canary.sh production --config shadow_mode.toml
   ```

2. Verify feature flag state:
   ```bash
   curl http://localhost:7432/api/v1/dual_memory/status
   # Expected: {"dual_types": true, "blended_recall": false, "formation": true}
   ```

3. Monitor for 48 hours before declaring stable

**Monitoring Dashboards**:

Primary dashboard: `Dual Memory - Shadow Mode`
- Concept formation rate (concepts/hour)
- Formation latency (P50/P95/P99)
- Memory overhead (RSS delta vs baseline)
- CPU overhead (% utilization increase)
- Binding creation rate (bindings/hour)
- Episode-to-concept ratio (target: 10:1 to 100:1)
- Error rates (formation failures, binding conflicts)

**Metrics to Watch**:

| Metric | Baseline | Warning Threshold | Critical Threshold | Action |
|--------|----------|-------------------|-------------------|--------|
| concepts_formed_total rate | 0 | >1000/hour | >5000/hour | Reduce formation frequency |
| concept_formation_duration_ms P99 | N/A | >100ms | >500ms | Investigate slow clustering |
| memory_rss_bytes delta | Baseline | +10% | +20% | Check concept cache eviction |
| cpu_utilization_percent delta | Baseline | +5% | +10% | Reduce formation frequency |
| concept_formation_errors_total rate | 0 | >10/hour | >100/hour | Investigate error patterns |
| binding_creation_rate | 0 | N/A | >10000/hour | Check runaway binding formation |

**Success Criteria** (must meet ALL for 7 consecutive days):
- [ ] Concept formation runs hourly without errors
- [ ] Concepts formed at reasonable rate (10-1000/hour depending on workload)
- [ ] Memory overhead <10% vs baseline (within acceptable bound)
- [ ] CPU overhead <5% vs baseline (negligible impact)
- [ ] Zero errors related to dual memory types
- [ ] No correlation between formation cycles and user-visible latency
- [ ] Storage growth follows expected pattern (concepts compress episodes)

**Rollback Triggers** (immediate abort if ANY occur):
- Memory growth exceeds +20% vs baseline
- CPU utilization increases >10% vs baseline
- Any errors in concept formation path cause user-visible failures
- Consolidation cycle time increases >50% (interference detected)
- Crash or panic related to dual memory code

**Rollback Procedure**:
```bash
# Emergency rollback: disable formation, keep existing concepts
engram config set dual_memory.enable_concept_formation false
engram config reload

# If concepts causing issues: full rollback
./scripts/rollback_dual_memory.sh --mode shadow --preserve-episodes
# This disables dual_memory_types feature and restarts with pure-episodic mode
```

**Phase 1 Exit Criteria**: All success criteria met for 7 consecutive days AND team approval to proceed

### Phase 2: Internal Testing (2 weeks)

**Duration**: 2 weeks minimum

**Objective**: Enable blended recall for internal test spaces only. Validate recall quality and performance with real queries but limited blast radius.

**Configuration Changes**:
```rust
DualMemoryConfig {
    // Keep shadow mode settings
    enable_dual_types: true,
    enable_concept_formation: true,

    // NEW: Enable blended recall for specific spaces
    enable_blended_recall: true,
    blended_recall_spaces: vec![
        "internal-qa-01".to_string(),
        "internal-qa-02".to_string(),
        "dogfood-team".to_string(),
    ],

    // Conservative blending: 90% episodes, 10% concepts
    semantic_weight: 0.10,
    episodic_weight: 0.90,

    // Safety limits
    max_concept_candidates: 50,  // Cap concept expansion
    blended_timeout_ms: 100,     // Fast fallback if slow
}
```

**Deployment**:
```bash
# Deploy updated config to canary first
./scripts/deploy_canary.sh production --config internal_testing.toml

# Monitor canary for 24 hours
./scripts/monitor_canary.sh 24h

# If stable, roll out to full fleet
./scripts/deploy_fleet.sh production --config internal_testing.toml
```

**Internal Test Spaces**:
- Create dedicated test spaces with known ground truth
- Populate with controlled datasets (similar to Task 011 validation sets)
- Run automated query suites comparing pure-episodic vs blended recall
- Have internal team members use dogfood space for real work

**A/B Testing Setup** (see A/B Testing Framework section):
- Control group: Pure-episodic recall (existing behavior)
- Treatment group: Blended recall with 10% semantic weight
- Assignment: By memory space ID (test spaces in treatment)
- Metrics: Latency, confidence scores, result set size, user satisfaction

**Monitoring Additions**:

New dashboard: `Dual Memory - Blended Recall`
- Blended recall latency vs pure-episodic latency (compare distributions)
- Semantic weight effectiveness (% queries where concepts contributed)
- Result set changes (delta in number of results)
- Confidence score distributions (treatment vs control)
- User feedback scores (if available)

**Metrics to Watch**:

| Metric | Baseline (Pure) | Warning Threshold | Critical Threshold | Action |
|--------|-----------------|-------------------|-------------------|--------|
| recall_latency_ms P99 | Baseline | +10ms | +20ms | Reduce semantic_weight |
| recall_latency_ms P50 | Baseline | +5ms | +10ms | Optimize concept retrieval |
| recall_result_count delta | 0 | ±50% | ±100% | Check concept quality |
| recall_confidence mean | Baseline | -0.05 | -0.10 | Increase coherence_threshold |
| blended_recall_errors_total | 0 | >10/hour | >100/hour | Investigate error patterns |
| blended_recall_timeouts_total | 0 | >1/hour | >10/hour | Increase timeout budget |

**Success Criteria** (must meet ALL for 7 consecutive days):
- [ ] Blended recall latency P99 within +10ms of baseline
- [ ] Blended recall latency P50 within +5ms of baseline
- [ ] Zero timeouts or errors in blended recall path
- [ ] Confidence scores remain stable (±0.05 vs baseline)
- [ ] Internal team reports positive or neutral experience (survey)
- [ ] Result quality metrics equivalent or better (precision/recall if measured)
- [ ] No regressions in pure-episodic recall for other spaces

**Rollback Triggers**:
- Latency P99 exceeds +20ms vs baseline
- Error rate >100/hour in blended recall
- Internal team reports critical quality issues
- Confidence scores drop >0.10 vs baseline
- Any crash or panic in blended recall code path

**Rollback Procedure**:
```bash
# Quick rollback: disable blended recall, keep formation
engram config set dual_memory.enable_blended_recall false
engram config reload

# Full rollback if needed
./scripts/rollback_dual_memory.sh --mode internal_testing
```

**Phase 2 Exit Criteria**: All success criteria met for 7 consecutive days AND internal team approval to proceed

### Phase 3: Canary Deployment (2 weeks)

**Duration**: 2 weeks minimum

**Objective**: Enable blended recall for 1% of production traffic. First real exposure to customer workloads with statistical rigor.

**Traffic Allocation**:
```rust
// Consistent hash-based assignment by space ID
fn assign_to_treatment(space_id: &str) -> bool {
    let hash = hash_space_id(space_id);
    (hash % 100) < 1  // 1% of spaces in treatment
}
```

**Configuration Changes**:
```rust
DualMemoryConfig {
    // Same as Phase 2 but broader scope
    enable_blended_recall: true,
    blended_recall_mode: BlendedRecallMode::Canary,
    canary_percentage: 1,  // 1% of production spaces

    // Slightly higher semantic weight after internal validation
    semantic_weight: 0.15,
    episodic_weight: 0.85,

    // Production safety limits
    max_concept_candidates: 100,
    blended_timeout_ms: 50,  // Tighter timeout in production
}
```

**A/B Testing Framework** (full implementation in next section):
- Control: 99% of spaces using pure-episodic recall
- Treatment: 1% of spaces using blended recall
- Assignment: Deterministic by space_id hash (stable assignment)
- Metrics: All recall metrics labeled with experiment group

**Statistical Analysis Requirements**:
- Minimum sample size: 1000 queries per group (wait for sufficient traffic)
- Significance level: p < 0.05 (95% confidence)
- Effect size: Detect ±5ms latency difference, ±0.05 confidence difference
- Power: 80% (beta = 0.20)
- Methodology: Two-sample t-test (or Mann-Whitney U if non-normal)

**Monitoring**:

New dashboard: `Dual Memory - A/B Test Results`
- Latency comparison (control vs treatment violin plots)
- Confidence score comparison (histograms)
- Result count comparison (box plots)
- Error rate comparison (bar chart)
- Statistical significance indicators (p-values, confidence intervals)

**Automated Analysis**:
```bash
# Run daily statistical analysis
./scripts/analyze_ab_test.sh --experiment dual_memory_canary

# Output example:
# Metric: recall_latency_ms_p99
#   Control:   45.2ms ± 2.1ms (n=12453)
#   Treatment: 47.8ms ± 2.3ms (n=127)
#   Delta:     +2.6ms (+5.8%)
#   p-value:   0.032 (significant at p<0.05)
#   Decision:  PROCEED (within +10ms warning threshold)
```

**Success Criteria** (must meet ALL):
- [ ] Minimum 1000 queries per group collected (sufficient sample)
- [ ] Latency P99 delta <+10ms with statistical significance
- [ ] Latency P50 delta <+5ms with statistical significance
- [ ] Confidence score delta within ±0.05 (not statistically different)
- [ ] Error rate delta <0.1% (not statistically different)
- [ ] Result count delta not causing user-visible quality issues
- [ ] Zero SEV1 or SEV2 incidents attributable to dual memory
- [ ] Customer feedback (if collected) shows no regression

**Rollback Triggers**:
- Latency P99 >+10ms AND statistically significant (p<0.05)
- Error rate increase >0.1% AND statistically significant
- Any SEV1/SEV2 incident related to dual memory
- Customer complaints mentioning poor recall quality (>5 complaints)
- Statistical analysis shows significant negative impact on confidence

**Rollback Procedure**:
```bash
# Immediate feature flag disable (no restart required)
engram feature-flag set dual_memory.blended_recall false

# Verify rollback
curl http://localhost:7432/api/v1/dual_memory/status | jq .blended_recall
# Expected: false

# Monitor for 30 minutes to confirm issues resolved
./scripts/monitor_rollback.sh 30m
```

**Phase 3 Exit Criteria**:
- All success criteria met for 14 consecutive days
- Statistical analysis shows neutral or positive impact
- Product and engineering sign-off to proceed

### Phase 4: Gradual Rollout (4 weeks)

**Duration**: 4 weeks for full rollout

**Objective**: Gradually increase treatment percentage from 1% to 100% with holds at each level for validation.

**Rollout Schedule**:

| Week | Traffic % | Duration at Level | Decision Point | Rollback Trigger |
|------|-----------|-------------------|----------------|------------------|
| 1 | 5% | 48 hours | Automated + manual review | Latency >+10ms P99 |
| 2 | 25% | 48 hours | Automated + manual review | Error rate >+0.1% |
| 3 | 50% | 72 hours | Automated + manual review | Confidence >-0.05 |
| 4 | 100% | Permanent | Final validation | Any regression |

**Automated Decision System**:
```rust
pub struct RolloutDecision {
    current_percentage: u8,
    next_percentage: u8,
    hold_duration: Duration,
    decision: Decision,
    reason: String,
}

pub enum Decision {
    Proceed,        // All metrics green
    Hold,           // Metrics borderline, need more data
    Rollback,       // Critical threshold breached
}

// Evaluated every hour by automated system
pub fn evaluate_rollout_decision(metrics: &ABTestMetrics) -> RolloutDecision {
    // Check critical thresholds first
    if metrics.latency_p99_delta > Duration::from_millis(20) {
        return RolloutDecision {
            decision: Decision::Rollback,
            reason: "Latency P99 exceeds +20ms critical threshold".to_string(),
            ..Default::default()
        };
    }

    if metrics.error_rate_delta > 0.001 {  // 0.1%
        return RolloutDecision {
            decision: Decision::Rollback,
            reason: "Error rate increase >0.1%".to_string(),
            ..Default::default()
        };
    }

    // Check warning thresholds
    if metrics.latency_p99_delta > Duration::from_millis(10) {
        return RolloutDecision {
            decision: Decision::Hold,
            reason: "Latency P99 in warning range (+10-20ms)".to_string(),
            ..Default::default()
        };
    }

    // All green, can proceed
    RolloutDecision {
        decision: Decision::Proceed,
        reason: "All metrics within acceptable bounds".to_string(),
        ..Default::default()
    }
}
```

**Configuration Updates**:
```bash
# Increase canary percentage via config update (no restart)
engram config set dual_memory.canary_percentage 5
engram config reload

# Verify assignment distribution
curl http://localhost:7432/api/v1/dual_memory/assignment_stats | jq .
# Expected: {"control": ~95%, "treatment": ~5%}
```

**Monitoring at Each Level**:
- Run full statistical analysis before proceeding to next level
- Require manual engineering approval at 25%, 50%, 100% gates
- Monitor for 48-72 hours at each level (more time = more risk absorbed)
- Watch for cumulative effects (memory growth, concept count growth)

**Success Criteria at Each Level**:
- [ ] Automated decision system shows "Proceed"
- [ ] Statistical analysis confirms metrics within bounds
- [ ] No increase in customer support tickets
- [ ] Engineering team manually approves progression
- [ ] Hold duration completed without incidents

**Rollback at Any Level**:
```bash
# Immediate rollback to previous percentage
engram config set dual_memory.canary_percentage <previous_level>
engram config reload

# If critical, full rollback
./scripts/rollback_dual_memory.sh --mode production --preserve-data
```

**Phase 4 Exit Criteria**: Successfully at 100% for 7 consecutive days with all metrics stable

### Phase 5: Optimization and Tuning (2 weeks)

**Duration**: 2 weeks post-100% rollout

**Objective**: Now that blended recall is fully deployed, optimize parameters based on production data and enable advanced features.

**Optimization Targets**:

1. **Semantic weight tuning**:
   - Baseline: 0.15 episodic, 0.85 semantic (from rollout)
   - Experiment: Test 0.20, 0.25, 0.30 semantic weights
   - Methodology: A/B test different weights on 10% traffic
   - Goal: Find optimal balance for recall quality

2. **Concept formation tuning**:
   - Baseline: Hourly formation, coherence 0.85
   - Experiment: Test 30-minute cadence, coherence 0.80
   - Methodology: Gradual increase with monitoring
   - Goal: More concepts without quality degradation

3. **Advanced features**:
   - Enable hierarchical spreading (Task 008) if disabled
   - Enable confidence propagation refinements (Task 010)
   - Enable aggressive consolidation mode (higher compression)

**Tuning Procedure**:
```bash
# Set up parameter sweep
./scripts/tune_dual_memory.sh \
  --parameter semantic_weight \
  --values 0.15,0.20,0.25,0.30 \
  --traffic-per-variant 5% \
  --duration 48h

# Analyze results
./scripts/analyze_tuning.sh --parameter semantic_weight

# Apply winning configuration
engram config set dual_memory.semantic_weight 0.25  # Example winner
```

**Success Criteria**:
- [ ] Identified optimal semantic weight with statistical validation
- [ ] Concept formation tuned for maximum compression without quality loss
- [ ] Advanced features enabled without regression
- [ ] Final parameter set documented in production config

**Phase 5 Exit Criteria**: Tuning complete, final configuration documented, Milestone 17 COMPLETE

## Rollout Decision Tree

```
Start: Phase 0 complete?
├─ NO → Block rollout, complete prerequisites
└─ YES → Proceed to Phase 1

Phase 1: Shadow mode metrics green for 7 days?
├─ NO → Investigate issues, extend Phase 1
└─ YES → Proceed to Phase 2

Phase 2: Internal testing successful for 7 days?
├─ NO → Investigate quality issues, extend Phase 2
└─ YES → Proceed to Phase 3

Phase 3: Canary A/B test shows neutral/positive impact?
├─ NO → Rollback and root cause analysis
└─ YES → Proceed to Phase 4

Phase 4: At each traffic level (5%, 25%, 50%, 100%):
├─ Metrics OK after hold duration?
│  ├─ NO → Rollback to previous level or abort
│  └─ YES → Proceed to next level
└─ At 100%: Stable for 7 days?
   ├─ NO → Extend monitoring or rollback
   └─ YES → Proceed to Phase 5

Phase 5: Optimization complete?
├─ NO → Continue tuning
└─ YES → MILESTONE 17 COMPLETE
```

## Emergency Procedures

### Emergency Rollback to Pure-Episodic Mode

**Use when**: SEV1 incident caused by dual memory system

**Procedure**:
```bash
# 1. Disable blended recall immediately (no restart)
engram feature-flag set dual_memory.blended_recall false

# 2. Verify recall is pure-episodic
curl http://localhost:7432/api/v1/dual_memory/status | jq .blended_recall
# Expected: false

# 3. Monitor for 15 minutes
./scripts/monitor_rollback.sh 15m

# 4. If issues persist, disable concept formation
engram feature-flag set dual_memory.enable_concept_formation false

# 5. If STILL persisting, full rollback
./scripts/rollback_dual_memory.sh --mode emergency
# This rebuilds without dual_memory_types feature and restarts
```

**RTO**: 15 minutes from decision to fully rolled back
**RPO**: Zero data loss (all episodes preserved, concepts can be regenerated)

### Partial Rollback (Specific Spaces)

**Use when**: Issues isolated to specific memory spaces

**Procedure**:
```bash
# Disable blended recall for specific space
engram config set dual_memory.blended_recall_exclude_spaces \
  '["problem-space-01", "problem-space-02"]'
engram config reload

# Verify space excluded
curl "http://localhost:7432/api/v1/dual_memory/space_status?space=problem-space-01"
# Expected: {"blended_recall": false}
```

### Data Corruption Recovery

**Use when**: Concepts or bindings corrupted

**Procedure**:
```bash
# 1. Stop concept formation
engram feature-flag set dual_memory.enable_concept_formation false

# 2. Identify corrupted concepts
./scripts/validate_dual_memory_integrity.sh

# 3. Quarantine corrupted data
./scripts/quarantine_concepts.sh --space <space_id> --concept <concept_id>

# 4. Regenerate from episodes
./scripts/regenerate_concepts.sh --space <space_id>

# 5. Resume formation
engram feature-flag set dual_memory.enable_concept_formation true
```

## Success Metrics Summary

### Performance Budgets (must maintain throughout rollout)

| Operation | Baseline | Warning | Critical | Measurement |
|-----------|----------|---------|----------|-------------|
| Recall latency P50 | 5ms | +5ms | +10ms | Per-query Prometheus histogram |
| Recall latency P99 | 45ms | +10ms | +20ms | Per-query Prometheus histogram |
| Concept formation P99 | N/A | 100ms | 500ms | Per-cycle Prometheus histogram |
| Memory overhead | Baseline | +10% | +20% | RSS via node_exporter |
| CPU overhead | Baseline | +5% | +10% | CPU% via node_exporter |

### Quality Metrics (must not regress)

| Metric | Baseline | Acceptable Delta | Measurement |
|--------|----------|------------------|-------------|
| Recall confidence | Baseline mean | ±0.05 | Per-query confidence score |
| Result count | Baseline mean | ±20% | Per-query result size |
| Customer satisfaction | Baseline score | -0 (no regression) | Survey/support tickets |

### Operational Metrics (must remain healthy)

| Metric | Target | Warning | Critical | Measurement |
|--------|--------|---------|----------|-------------|
| Error rate | <0.01% | >0.05% | >0.1% | errors_total / requests_total |
| Availability | >99.9% | <99.9% | <99.5% | Uptime monitoring |
| Data loss | Zero | Zero | Any | WAL integrity checks |

## Timeline and Milestones

**Total Duration**: 10-12 weeks from Phase 0 to Phase 5 complete

Week-by-week breakdown:
- Week 1: Phase 0 pre-deployment validation
- Weeks 2-3: Phase 1 shadow mode (2 weeks)
- Weeks 4-5: Phase 2 internal testing (2 weeks)
- Weeks 6-7: Phase 3 canary deployment (2 weeks)
- Weeks 8-11: Phase 4 gradual rollout (4 weeks)
- Weeks 12-13: Phase 5 optimization (2 weeks)

**Go/No-Go Gates**:
- Gate 1 (before Phase 1): Pre-deployment checklist complete
- Gate 2 (before Phase 2): Shadow mode metrics green for 7 days
- Gate 3 (before Phase 3): Internal testing approved by product team
- Gate 4 (before Phase 4-5%): Canary A/B test shows neutral/positive
- Gate 5 (before Phase 4-25%): 5% stable for 48 hours
- Gate 6 (before Phase 4-50%): 25% stable for 48 hours
- Gate 7 (before Phase 4-100%): 50% stable for 72 hours
- Gate 8 (before Phase 5): 100% stable for 7 days

**Critical Path**:
- Any failed gate blocks progression indefinitely
- Any SEV1/SEV2 incident triggers rollback and root cause requirement
- Statistical analysis showing significant negative impact blocks progression

```

### A/B Testing Framework

Implement rigorous A/B testing infrastructure for comparing treatment (dual memory) vs control (pure-episodic) groups.

#### Assignment Mechanism

```rust
// tools/ab_testing/assignment.rs

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

pub struct ABTestAssignment {
    experiment_name: String,
    treatment_percentage: u8,  // 0-100
}

impl ABTestAssignment {
    pub fn assign(&self, space_id: &str) -> ExperimentGroup {
        // Deterministic assignment based on space ID hash
        // Ensures same space always gets same assignment
        let mut hasher = DefaultHasher::new();
        space_id.hash(&mut hasher);
        let hash_value = hasher.finish();

        let bucket = (hash_value % 100) as u8;

        if bucket < self.treatment_percentage {
            ExperimentGroup::Treatment
        } else {
            ExperimentGroup::Control
        }
    }

    pub fn get_current_assignment(&self, space_id: &str) -> ExperimentAssignment {
        ExperimentAssignment {
            experiment: self.experiment_name.clone(),
            group: self.assign(space_id),
            space_id: space_id.to_string(),
            assigned_at: Utc::now(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExperimentGroup {
    Control,
    Treatment,
}

impl std::fmt::Display for ExperimentGroup {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Self::Control => write!(f, "control"),
            Self::Treatment => write!(f, "treatment"),
        }
    }
}
```

#### Metrics Collection

All recall operations must be instrumented with experiment group labels for A/B comparison.

#### Statistical Analysis

Automated daily analysis comparing control vs treatment groups with proper statistical tests (t-test for normal distributions, Mann-Whitney U for non-normal).

Minimum requirements:
- Sample size: >1000 queries per group
- Significance level: p < 0.05 (95% confidence)
- Effect size detection: ±5ms latency, ±0.05 confidence
- Power: 80% (beta = 0.20)



## Operational Runbook

#### `/Users/jordanwashburn/Workspace/orchard9/engram/docs/operations/dual_memory_runbook.md`

Comprehensive operational procedures for dual memory system following incident-response.md patterns:

```markdown
# Dual Memory Operations Runbook

## Quick Reference

| Issue | Symptoms | Resolution | Escalation |
|-------|----------|------------|------------|
| High concept formation rate | Memory growth, slow consolidation | Reduce formation frequency | Engineer if persists |
| Poor recall quality | Low confidence, user complaints | Check concept quality | Product team review |
| Binding leaks | Unbounded binding growth | Quarantine and regenerate | Engineer immediately |
| Blended recall timeout | Latency spikes, timeouts | Reduce semantic_weight | Disable if critical |
| Concept corruption | NaN values, integrity errors | Stop formation, regenerate | SEV1 incident |

## Common Issues

### Issue 1: High Concept Formation Rate

**Symptoms**:
- concepts_formed_total metric spiking (>1000/hour)
- Memory RSS increasing beyond budget (+10% baseline)
- Consolidation cycles taking longer (>60s)

**Root Cause**: Coherence threshold too low or cluster size too small, causing over-formation

**Diagnosis**:
```bash
# Check formation rate
curl http://localhost:7432/metrics | grep concepts_formed_total

# Check current configuration
curl http://localhost:7432/api/v1/dual_memory/config | jq .

# Review recent concept quality
curl http://localhost:7432/api/v1/dual_memory/concepts/recent?limit=100 | jq '.[] | {coherence, instance_count}'
```

**Resolution Steps**:
1. Increase coherence threshold (more selective):
   ```bash
   engram config set dual_memory.coherence_threshold 0.90
   engram config reload
   ```

2. Increase minimum cluster size (require more evidence):
   ```bash
   engram config set dual_memory.min_cluster_size 15
   engram config reload
   ```

3. Reduce formation frequency (less aggressive):
   ```bash
   engram config set dual_memory.formation_interval 7200  # 2 hours
   engram config reload
   ```

4. Monitor for 30 minutes to verify rate decreases

**Escalation**: If rate does not decrease or memory growth continues, escalate to on-call engineer

### Issue 2: Poor Recall Quality

**Symptoms**:
- User reports of irrelevant results
- Confidence scores dropping (mean <0.60 vs baseline 0.75)
- Customer support tickets mentioning search quality
- A/B test shows statistically significant regression

**Root Cause**: Concepts are low quality, blended recall is retrieving poor semantic matches

**Diagnosis**:
```bash
# Check confidence score distribution
curl http://localhost:7432/metrics | grep recall_confidence

# Sample recent recall results
./scripts/sample_recall_quality.sh --space <affected_space> --n 20

# Check concept coherence distribution
curl http://localhost:7432/api/v1/dual_memory/stats | jq .coherence_histogram
```

**Resolution Steps**:
1. Reduce semantic weight (less concept influence):
   ```bash
   engram config set dual_memory.semantic_weight 0.05
   engram config reload
   ```

2. Increase coherence threshold for NEW concepts:
   ```bash
   engram config set dual_memory.coherence_threshold 0.90
   engram config reload
   ```

3. Quarantine low-quality concepts:
   ```bash
   ./scripts/quarantine_low_coherence_concepts.sh --threshold 0.70
   ```

4. Monitor confidence scores for improvement

**Escalation**: If quality does not improve within 4 hours, involve product team to decide on rollback

### Issue 3: Binding Leak (Runaway Binding Creation)

**Symptoms**:
- binding_creation_rate metric exceeding 10000/hour
- Memory growth accelerating exponentially
- Consolidation taking minutes instead of seconds

**Root Cause**: Bug in binding formation logic causing duplicate or circular bindings

**Diagnosis**:
```bash
# Check binding count per concept
curl http://localhost:7432/api/v1/dual_memory/bindings/stats | jq .

# Identify concepts with excessive bindings
curl http://localhost:7432/api/v1/dual_memory/concepts?sort=binding_count&limit=10 | jq .
```

**Resolution Steps**:
1. Immediately stop concept formation:
   ```bash
   engram feature-flag set dual_memory.enable_concept_formation false
   ```

2. Identify affected concepts:
   ```bash
   ./scripts/identify_binding_leaks.sh > /tmp/binding_leaks.txt
   ```

3. Quarantine affected concepts:
   ```bash
   ./scripts/quarantine_concepts.sh --from-file /tmp/binding_leaks.txt
   ```

4. Clean up bindings:
   ```bash
   ./scripts/clean_bindings.sh --dry-run
   ./scripts/clean_bindings.sh --execute
   ```

5. Resume formation with monitoring:
   ```bash
   engram feature-flag set dual_memory.enable_concept_formation true
   ```

**Escalation**: SEV2 incident - notify on-call engineer immediately. If bindings continue to grow, escalate to SEV1.

### Issue 4: Blended Recall Timeout

**Symptoms**:
- blended_recall_timeouts_total metric increasing (>10/hour)
- Latency P99 spiking (>100ms)
- Some queries returning partial results

**Root Cause**: Semantic search taking too long, hitting timeout budget

**Diagnosis**:
```bash
# Check timeout rate
curl http://localhost:7432/metrics | grep blended_recall_timeouts

# Check concept search latency
curl http://localhost:7432/metrics | grep concept_search_duration

# Profile slow queries
./scripts/profile_blended_recall.sh --duration 60s
```

**Resolution Steps**:
1. Reduce semantic weight (smaller search space):
   ```bash
   engram config set dual_memory.semantic_weight 0.10
   engram config reload
   ```

2. Reduce max concept candidates:
   ```bash
   engram config set dual_memory.max_concept_candidates 50
   engram config reload
   ```

3. Increase timeout budget if latency acceptable:
   ```bash
   engram config set dual_memory.blended_timeout_ms 75
   engram config reload
   ```

**Escalation**: If timeouts exceed 100/hour or latency P99 >150ms, disable blended recall

### Issue 5: Concept Corruption (NaN Values or Integrity Errors)

**Symptoms**:
- Errors in logs mentioning NaN values or invalid embeddings
- Integrity check failures
- Recall returning errors for affected spaces
- Silent degradation of recall quality

**Root Cause**: Data corruption in concept centroids or confidence scores

**Diagnosis**:
```bash
# Run integrity check
./scripts/validate_dual_memory_integrity.sh --full

# Identify corrupted concepts
grep -r "NaN\|invalid embedding" /var/log/engram/

# Check affected spaces
curl http://localhost:7432/api/v1/dual_memory/health | jq .corrupted_concepts
```

**Resolution Steps**:
1. **IMMEDIATE**: Stop concept formation to prevent spread:
   ```bash
   engram feature-flag set dual_memory.enable_concept_formation false
   ```

2. Disable blended recall for affected spaces:
   ```bash
   engram config set dual_memory.blended_recall_exclude_spaces \
     '["corrupted-space-01", "corrupted-space-02"]'
   engram config reload
   ```

3. Quarantine corrupted concepts:
   ```bash
   ./scripts/quarantine_concepts.sh --corrupted
   ```

4. Regenerate from clean episodes:
   ```bash
   ./scripts/regenerate_concepts.sh --space corrupted-space-01
   ```

5. Run integrity check again:
   ```bash
   ./scripts/validate_dual_memory_integrity.sh --space corrupted-space-01
   ```

6. Resume formation after validation:
   ```bash
   engram feature-flag set dual_memory.enable_concept_formation true
   ```

**Escalation**: SEV1 incident - data corruption is critical. Notify on-call immediately and create war room.

## Monitoring and Alerts

### Key Dashboards

1. **Dual Memory - Overview**
   - URL: http://grafana:3000/d/dual-memory-overview
   - Metrics: Formation rate, memory overhead, CPU overhead, error rates
   - Use for: Daily health checks

2. **Dual Memory - A/B Test**
   - URL: http://grafana:3000/d/dual-memory-ab-test
   - Metrics: Latency comparison, confidence comparison, statistical significance
   - Use for: Rollout decision making

3. **Dual Memory - Troubleshooting**
   - URL: http://grafana:3000/d/dual-memory-troubleshooting
   - Metrics: Binding counts, concept quality, formation errors
   - Use for: Issue diagnosis

### Alert Rules

Configured in Prometheus alerting rules:

```yaml
# concepts_formed_total rate too high
- alert: HighConceptFormationRate
  expr: rate(concepts_formed_total[5m]) > 1000
  for: 10m
  labels:
    severity: warning
  annotations:
    summary: Concept formation rate exceeding 1000/hour
    runbook: https://docs/operations/dual_memory_runbook.md#issue-1

# blended_recall_timeouts increasing
- alert: BlendedRecallTimeouts
  expr: rate(blended_recall_timeouts_total[5m]) > 10
  for: 5m
  labels:
    severity: warning
  annotations:
    summary: Blended recall timeouts >10/hour
    runbook: https://docs/operations/dual_memory_runbook.md#issue-4

# binding_creation_rate runaway
- alert: BindingLeak
  expr: rate(binding_creation_rate[5m]) > 10000
  for: 2m
  labels:
    severity: critical
  annotations:
    summary: Binding creation rate exceeding 10000/hour (possible leak)
    runbook: https://docs/operations/dual_memory_runbook.md#issue-3

# concept corruption detected
- alert: ConceptCorruption
  expr: increase(concept_integrity_errors_total[5m]) > 0
  for: 0m
  labels:
    severity: critical
  annotations:
    summary: Concept corruption detected
    runbook: https://docs/operations/dual_memory_runbook.md#issue-5
```

## Routine Maintenance

### Daily Tasks

1. Review Grafana dashboards for anomalies (5 minutes)
2. Check error rates in logs (2 minutes):
   ```bash
   journalctl -u engram --since "24 hours ago" | grep -i "error\|panic" | wc -l
   ```
3. Verify A/B test metrics if in rollout phase (10 minutes):
   ```bash
   ./scripts/analyze_ab_test.sh dual_memory_canary
   ```

### Weekly Tasks

1. Run integrity check on all spaces (30 minutes):
   ```bash
   ./scripts/validate_dual_memory_integrity.sh --all-spaces
   ```
2. Review concept quality metrics:
   ```bash
   curl http://localhost:7432/api/v1/dual_memory/stats/weekly | jq .
   ```
3. Tune parameters if needed based on production data
4. Update team on rollout progress

### Monthly Tasks

1. Full performance benchmark vs baseline:
   ```bash
   cargo bench --bench dual_memory_validation
   ```
2. Review and update operational runbook based on incidents
3. Train new team members on dual memory operations
4. Audit concept formation quality with sample review

## Rollback Procedures

See `/docs/operations/dual_memory_rollout.md` Section "Emergency Procedures" for full rollback instructions.

**Quick Rollback Commands**:
```bash
# Disable blended recall (immediate, no restart)
engram feature-flag set dual_memory.blended_recall false

# Disable concept formation
engram feature-flag set dual_memory.enable_concept_formation false

# Full emergency rollback
./scripts/rollback_dual_memory.sh --mode emergency
```

## Team Contacts

- On-call Engineer: Check PagerDuty rotation
- Dual Memory Tech Lead: [Name, Slack @handle]
- Product Manager: [Name, Slack @handle]
- Incident Commander: Check incident response guide

## Change Log

- 2025-XX-XX: Initial runbook created for Milestone 17 rollout
- YYYY-MM-DD: [Future updates go here]
```

## Team Training Requirements

### Operator Training (Level 1 On-Call)

**Duration**: 2 hours

**Prerequisites**: Completed general Engram operations training

**Curriculum**:

1. **Dual Memory Architecture Overview** (30 minutes)
   - Episodic vs semantic memory distinction
   - Concept formation process
   - Binding relationships
   - Blended recall mechanism
   - Why we're doing this (product value)

2. **Monitoring and Dashboards** (30 minutes)
   - Walk through Grafana dashboards
   - Key metrics to watch
   - How to interpret A/B test results
   - Alert triage procedures

3. **Common Issues and Resolutions** (45 minutes)
   - Work through 5 common issues from runbook
   - Practice diagnosis commands
   - Execute resolution steps in staging
   - Know when to escalate

4. **Rollback Procedures** (15 minutes)
   - Practice emergency rollback in staging
   - Understand RTO/RPO guarantees
   - Verify rollback success

**Hands-On Exercise**:
- Simulated SEV2 incident: High concept formation rate
- Operator must diagnose and resolve using runbook
- Time limit: 30 minutes
- Pass/fail criteria: Issue resolved, metrics return to baseline

**Certification**: Sign-off from senior engineer required before taking on-call shifts

### Engineer Training (Level 2 Subject Matter Expert)

**Duration**: 4 hours

**Prerequisites**: Operator training complete, Rust proficiency

**Curriculum**:

1. **Deep Dive on Dual Memory Implementation** (90 minutes)
   - Code walkthrough: Task 001-014
   - DualMemoryNode type system
   - Concept formation algorithm (clustering, coherence)
   - Blended recall execution path
   - Performance considerations

2. **Debugging and Profiling** (60 minutes)
   - Debug builds with symbols
   - Using perf/flamegraph for profiling
   - Memory leak detection with valgrind
   - Log analysis techniques
   - Reading core dumps

3. **Advanced Troubleshooting** (60 minutes)
   - Data corruption scenarios
   - Performance regression diagnosis
   - Statistical analysis of A/B tests
   - Custom diagnostic tools

4. **Code Changes and Hotfixes** (30 minutes)
   - Emergency patch procedures
   - Testing requirements for hotfixes
   - Deployment verification
   - Post-incident analysis

**Hands-On Exercise**:
- Simulated SEV1 incident: Concept corruption with data loss risk
- Engineer must root cause, apply fix, verify integrity
- Time limit: 2 hours
- Pass/fail: Corruption stopped, clean data regenerated, no recurrence

**Certification**: Code review and incident response simulation with tech lead

### Product Team Training

**Duration**: 1 hour

**Audience**: Product managers, design leads

**Curriculum**:

1. **User-Facing Benefits** (20 minutes)
   - Improved recall through semantic generalization
   - Better results for abstract queries
   - Compression benefits (faster loading)

2. **Rollout Plan and Timeline** (20 minutes)
   - Phase-by-phase explanation
   - Success criteria at each gate
   - How decisions are made
   - When to expect completion

3. **Quality Validation** (15 minutes)
   - How we're measuring recall quality
   - A/B testing methodology
   - What "statistically significant" means
   - How to interpret results

4. **Risk and Rollback** (5 minutes)
   - What could go wrong
   - How we detect issues
   - Rollback procedures and timing
   - User impact during rollback

## Post-Rollout Analysis

### 30-Day Review (after Phase 5 complete)

**Date**: [Schedule 30 days after 100% rollout]

**Attendees**: Engineering team, product team, operations

**Agenda**:

1. **Metrics Review** (30 minutes)
   - Performance vs baseline (latency, memory, CPU)
   - Quality metrics (confidence, result count, user satisfaction)
   - Operational metrics (error rates, availability)
   - Cost metrics (infrastructure, storage)

2. **Incident Review** (20 minutes)
   - All SEV1/SEV2 incidents during rollout
   - Root cause analysis summaries
   - Preventive measures taken
   - Outstanding action items

3. **Parameter Tuning Results** (20 minutes)
   - Optimal semantic weight identified
   - Final concept formation cadence
   - Production configuration documented

4. **Lessons Learned** (20 minutes)
   - What went well in the rollout
   - What could be improved
   - Documentation gaps identified
   - Training effectiveness

5. **Future Improvements** (10 minutes)
   - Feature requests from production data
   - Performance optimization opportunities
   - Scaling considerations

**Outputs**:
- 30-day review document published
- Updated operational documentation
- Roadmap items for future improvements
- Milestone 17 officially declared COMPLETE

### Continuous Monitoring

After 30-day review, ongoing monitoring continues with:

- Monthly concept quality audits
- Quarterly performance benchmarks
- Semi-annual full system validation
- Continuous A/B testing for parameter optimization

## Documentation Files to Create

### Markdown Documentation

1. `/Users/jordanwashburn/Workspace/orchard9/engram/docs/operations/dual_memory_rollout.md`
   - Complete rollout guide (embedded above)
   - ~3000 lines

2. `/Users/jordanwashburn/Workspace/orchard9/engram/docs/operations/dual_memory_runbook.md`
   - Operational procedures (embedded above)
   - ~1000 lines

3. `/Users/jordanwashburn/Workspace/orchard9/engram/docs/operations/dual_memory_migration.md`
   - Step-by-step migration from pure-episodic to dual memory
   - Downtime estimates
   - Backup/restore procedures
   - Verification steps
   - ~500 lines

### Scripts and Tools

1. `/Users/jordanwashburn/Workspace/orchard9/engram/tools/dual_memory_validator.rs`
   - Automated validation tool checking integrity
   - Runs pre-flight checks before rollout phases
   - Validates concept coherence, binding consistency
   - ~300 lines

2. `/Users/jordanwashburn/Workspace/orchard9/engram/scripts/analyze_ab_test.sh`
   - Bash script for quick A/B test analysis
   - ~200 lines

3. `/Users/jordanwashburn/Workspace/orchard9/engram/scripts/statistical_analysis.py`
   - Python script for rigorous statistical testing
   - Generates reports with p-values, confidence intervals
   - ~300 lines

4. `/Users/jordanwashburn/Workspace/orchard9/engram/scripts/rollback_dual_memory.sh`
   - Emergency rollback script
   - ~150 lines

5. `/Users/jordanwashburn/Workspace/orchard9/engram/scripts/validate_dual_memory_integrity.sh`
   - Integrity checking script
   - ~200 lines

### Configuration Files

1. `/Users/jordanwashburn/Workspace/orchard9/engram/configs/dual_memory_shadow_mode.toml`
2. `/Users/jordanwashburn/Workspace/orchard9/engram/configs/dual_memory_internal_testing.toml`
3. `/Users/jordanwashburn/Workspace/orchard9/engram/configs/dual_memory_canary.toml`
4. `/Users/jordanwashburn/Workspace/orchard9/engram/configs/dual_memory_production.toml`

## Implementation Notes

- This task is DOCUMENTATION-HEAVY, not code-heavy
- Much of the infrastructure (feature flags, metrics, configuration) exists from prior tasks
- Focus is on PROCESS: how to roll out safely, what to measure, when to abort
- Operator training is critical - do not skip
- Statistical rigor in A/B testing is non-negotiable
- Document everything - future rollouts will reference this

## Testing Approach

### Pre-Rollout Validation

1. **Staging Environment Validation** (Phase 0)
   - Deploy full rollout plan to staging
   - Run through all 5 phases in compressed timeline (days not weeks)
   - Verify all scripts work
   - Validate monitoring dashboards
   - Practice rollback procedures

2. **Operator Training Validation**
   - Each operator completes training and passes exercise
   - Simulated incident response drills
   - Verify runbook completeness

3. **A/B Testing Infrastructure Validation**
   - Deploy A/B test framework to staging
   - Run synthetic experiments with known outcomes
   - Verify metrics collection and statistical analysis
   - Confirm assignment consistency

### During Rollout

Each phase has explicit success criteria and automated checks. See rollout plan above.

### Post-Rollout

- 30-day review validates overall success
- Continuous monitoring catches regressions
- Quarterly benchmarks track long-term trends

## Acceptance Criteria

- [ ] Comprehensive rollout guide document created and reviewed
- [ ] Operational runbook covering all common issues created
- [ ] Migration procedures documented with step-by-step instructions
- [ ] A/B testing framework implemented and validated
- [ ] Statistical analysis tools created and tested
- [ ] Validation scripts (integrity, quality) implemented
- [ ] Emergency rollback procedures documented and tested
- [ ] Monitoring dashboards created for all phases
- [ ] Prometheus alerts configured with correct thresholds
- [ ] Operator training curriculum created and delivered
- [ ] Engineer training curriculum created and delivered
- [ ] Product team training delivered
- [ ] All training materials validated with practice exercises
- [ ] Phase 0 pre-deployment checklist complete
- [ ] Staging environment successfully completes full rollout simulation
- [ ] Team sign-off obtained for production rollout
- [ ] 30-day post-rollout review scheduled

## Dependencies

- Tasks 001-014 complete and passing acceptance tests
- Prometheus and Grafana monitoring infrastructure operational
- Feature flag system implemented and tested
- Configuration reload mechanism working
- Staging environment available for validation

## Estimated Time

4 weeks:
- Week 1: Create documentation (rollout guide, runbook, migration guide)
- Week 2: Implement tools (validator, A/B testing, statistical analysis)
- Week 3: Create training materials and deliver training
- Week 4: Staging validation and team sign-off

Note: This is PREPARATION time. Actual rollout takes 10-12 weeks (Phases 1-5).

## Follow-Up Tasks

- None (this is final task of Milestone 17)
- After 30-day review, Milestone 17 officially complete
- Future milestones may optimize or extend dual memory features

## Performance Budget

Since this is primarily a documentation and process task, performance budgets are defined for the SYSTEM being rolled out, not this task itself:

- Recall latency P50: <+5ms vs baseline (warning), <+10ms (critical)
- Recall latency P99: <+10ms vs baseline (warning), <+20ms (critical)
- Memory overhead: <+10% vs baseline (warning), <+20% (critical)
- CPU overhead: <+5% vs baseline (warning), <+10% (critical)
- Error rate increase: <0.05% (warning), <0.1% (critical)
- Data loss: Zero tolerance

These budgets are enforced throughout the rollout phases.

## References

- Existing operational documentation patterns:
  - `/Users/jordanwashburn/Workspace/orchard9/engram/docs/operations/production-deployment.md`
  - `/Users/jordanwashburn/Workspace/orchard9/engram/docs/operations/incident-response.md`
  - `/Users/jordanwashburn/Workspace/orchard9/engram/docs/operations/zig_rollback_procedures.md`
- Statistical testing methodology: https://en.wikipedia.org/wiki/Mann%E2%80%93Whitney_U_test
- A/B testing best practices: https://exp-platform.com/
- Milestone 17 task specifications: roadmap/milestone-17/001-014
- Vision document: `/Users/jordanwashburn/Workspace/orchard9/engram/vision.md`
- Milestones document: `/Users/jordanwashburn/Workspace/orchard9/engram/milestones.md`

