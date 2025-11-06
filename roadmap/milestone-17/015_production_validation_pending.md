# Task 015: Production Validation

## Objective
Validate the dual memory system in production through phased rollout, A/B testing, and comprehensive monitoring.

## Background
Production validation ensures the dual memory architecture performs correctly under real-world conditions.

## Requirements
1. Design phased rollout plan
2. Implement A/B testing framework
3. Define success criteria and rollback triggers
4. Create operational runbook
5. Document migration procedures

## Technical Specification

### Files to Create
- `docs/operations/dual_memory_rollout.md` - Rollout guide
- `docs/operations/dual_memory_runbook.md` - Operational procedures
- `tools/dual_memory_validator.rs` - Validation tool

### Rollout Phases
```markdown
## Phase 1: Shadow Mode (Week 1-2)
- Enable dual_memory_types feature flag
- Run concept formation in background
- No user-facing changes
- Monitor: Memory usage, CPU, formation metrics

## Phase 2: Internal Testing (Week 3-4)
- Enable for internal test spaces
- Use blended recall with low weight (10%)
- Monitor: Recall quality, latency
- Rollback trigger: >10% latency increase

## Phase 3: Canary Deployment (Week 5-6)
- 1% of production traffic
- Full dual memory features
- A/B test vs control group
- Monitor: All metrics, user feedback

## Phase 4: Gradual Rollout (Week 7-10)
- Increase to 5%, 25%, 50%, 100%
- Hold at each level for 48 hours
- Monitor: Error rates, performance

## Phase 5: Optimization (Week 11-12)
- Tune parameters based on production data
- Enable advanced features
- Document learnings
```

### A/B Testing Framework
```rust
pub struct DualMemoryABTest {
    control_group: Arc<MemoryGraph>,      // Legacy single-type
    treatment_group: Arc<DualMemoryGraph>, // Dual memory
    assignment_rate: f32,                  // % in treatment
}

impl DualMemoryABTest {
    pub async fn process_request(&self, request: MemoryRequest) -> Response {
        let in_treatment = self.assign_to_treatment(&request.space_id);
        
        let (response, group) = if in_treatment {
            let resp = self.treatment_group.handle(request).await?;
            (resp, "treatment")
        } else {
            let resp = self.control_group.handle(request).await?;
            (resp, "control")
        };
        
        // Record metrics with group label
        AB_TEST_LATENCY
            .with_label_values(&[group])
            .observe(response.latency);
            
        AB_TEST_RECALL_QUALITY
            .with_label_values(&[group])
            .observe(response.confidence);
        
        response
    }
    
    fn assign_to_treatment(&self, space_id: &str) -> bool {
        // Consistent assignment based on space ID
        let hash = hash(space_id) as f32 / u64::MAX as f32;
        hash < self.assignment_rate
    }
}
```

### Validation Checklist
```rust
pub struct ProductionValidator {
    baseline_metrics: BaselineMetrics,
    thresholds: ValidationThresholds,
}

impl ProductionValidator {
    pub async fn validate_deployment(&self) -> ValidationResult {
        let mut checks = Vec::new();
        
        // Performance checks
        checks.push(self.check_latency_regression().await);
        checks.push(self.check_memory_usage().await);
        checks.push(self.check_cpu_utilization().await);
        
        // Quality checks
        checks.push(self.check_recall_quality().await);
        checks.push(self.check_concept_coherence().await);
        checks.push(self.check_fan_effect_distribution().await);
        
        // Operational checks
        checks.push(self.check_error_rates().await);
        checks.push(self.check_consolidation_success().await);
        
        ValidationResult {
            passed: checks.iter().all(|c| c.passed),
            checks,
            recommendation: self.generate_recommendation(&checks),
        }
    }
}
```

### Operational Runbook
```yaml
# Dual Memory Operational Runbook

## Common Issues

### High Concept Formation Rate
Symptoms:
  - concepts_formed metric spike
  - Memory usage increase
  - Consolidation latency increase

Actions:
  1. Check coherence_threshold setting
  2. Reduce concept_sample_rate
  3. Increase min_cluster_size
  4. Monitor for 30 minutes

### Poor Recall Quality
Symptoms:
  - Lower confidence scores
  - User complaints
  - A/B test showing regression

Actions:
  1. Check semantic_weight balance
  2. Verify concept quality metrics
  3. Review recent concept formations
  4. Consider rollback if severe

### Memory Leak
Symptoms:
  - Steady memory growth
  - OOM errors
  - GC pressure

Actions:
  1. Check binding cleanup
  2. Review concept cache size
  3. Analyze heap dump
  4. Emergency: Disable concept formation

## Rollback Procedures

### Feature Flag Rollback
```bash
# Disable specific features
engram config set dual_memory.enable_concepts false
engram config set dual_memory.use_blended_recall false

# Full rollback
engram config set dual_memory.enabled false
```

### Data Rollback
```bash
# Export current state
engram export --format dual_memory > backup.json

# Revert to single-type
engram migrate --to single_memory --backup backup.json
```
```

## Implementation Notes
- Start with conservative thresholds
- Automate rollback triggers
- Keep detailed logs of all changes
- Coordinate with support team

## Testing Approach
1. Staging environment validation
2. Canary deployment monitoring
3. A/B test statistical analysis
4. Production soak test

## Acceptance Criteria
- [ ] Rollout plan reviewed and approved
- [ ] A/B testing shows positive results
- [ ] No critical issues in production
- [ ] Runbook tested and validated
- [ ] Team trained on procedures

## Dependencies
- Tasks 001-014 completed
- Production monitoring in place
- Rollback procedures tested

## Estimated Time
2 weeks (includes monitoring period)