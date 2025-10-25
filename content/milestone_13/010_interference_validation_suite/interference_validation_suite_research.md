# Interference Validation Suite: Research and Technical Foundation

## Comprehensive Interference Testing

Tasks 004-005 implement proactive interference, retroactive interference, and the fan effect. This task creates a comprehensive validation suite ensuring all three match published empirical data with statistical rigor.

## Validation Targets

**Proactive Interference (Underwood, 1957; Anderson, 1974):**
- High similarity: 30-40% recall reduction
- Medium similarity: 15-25% reduction
- Low similarity: 0-10% reduction
- Time course: strongest immediately, fades over 24 hours

**Retroactive Interference (Postman & Underwood, 1973):**
- High similarity: 40-50% recall reduction
- Medium similarity: 20-30% reduction
- Stronger with higher new-learning strength

**Fan Effect (Anderson, 1974):**
- Fan 2 vs Fan 1: +100-150ms
- Fan 3 vs Fan 1: +200-300ms
- Fan 4 vs Fan 1: +300-450ms
- Logarithmic scaling

## Validation Suite Architecture

```rust
pub struct InterferenceValidationSuite {
    proactive_tests: ProactiveInterferenceTests,
    retroactive_tests: RetroactiveInterferenceTests,
    fan_effect_tests: FanEffectTests,
}

impl InterferenceValidationSuite {
    pub fn run_full_validation(&mut self) -> ValidationReport {
        let pi_results = self.proactive_tests.run();
        let ri_results = self.retroactive_tests.run();
        let fan_results = self.fan_effect_tests.run();

        ValidationReport {
            proactive: pi_results,
            retroactive: ri_results,
            fan: fan_results,
            overall_pass: pi_results.pass && ri_results.pass && fan_results.pass,
        }
    }
}

pub struct ProactiveInterferenceTests;

impl ProactiveInterferenceTests {
    pub fn run(&mut self) -> TestResults {
        let high_sim = self.test_high_similarity();
        let med_sim = self.test_medium_similarity();
        let low_sim = self.test_low_similarity();
        let time_course = self.test_time_course();

        TestResults {
            high_similarity: self.validate_range(high_sim, 0.30, 0.40),
            medium_similarity: self.validate_range(med_sim, 0.15, 0.25),
            low_similarity: self.validate_range(low_sim, 0.0, 0.10),
            time_course_pass: time_course,
            pass: /* all sub-tests pass */,
        }
    }

    fn test_high_similarity(&mut self) -> f32 {
        let mut recall_reductions = Vec::new();

        for trial in 0..800 {
            // Learn List A with high similarity to List B
            let list_a = generate_list_with_similarity(0.85);
            memory.encode_list(list_a);
            advance_time(Duration::from_hours(1));

            // Learn List B
            let list_b = generate_similar_list(list_a, 0.85);
            memory.encode_list(list_b);

            // Test recall of List B (interfered by List A)
            let recall_b = memory.recall_list();
            let accuracy = compute_accuracy(recall_b, list_b);

            // Control: recall with no prior learning
            memory.clear();
            memory.encode_list(list_b.clone());
            let control_recall = memory.recall_list();
            let control_accuracy = compute_accuracy(control_recall, list_b);

            // Interference = reduction from control
            let reduction = (control_accuracy - accuracy) / control_accuracy;
            recall_reductions.push(reduction);
        }

        mean(&recall_reductions)
    }
}
```

## Statistical Power Analysis

For each interference type, we need sufficient power to detect published effect sizes:

**Proactive Interference:**
- Expected effect: d = 0.7-1.0 (large)
- Required N: 500-800 per condition
- Alpha: 0.05
- Power: 0.80

**Retroactive Interference:**
- Expected effect: d = 0.8-1.2 (large)
- Required N: 500-800 per condition
- Alpha: 0.05
- Power: 0.80

**Fan Effect:**
- Expected effect: d = 0.6-0.9 (medium to large)
- Required N: 600-1000 per condition
- Alpha: 0.05
- Power: 0.80

Total trials across all tests: ~10,000-15,000

## Acceptance Criteria

**Passing Criteria:**
All sub-tests must pass for validation to succeed:

1. **Proactive Interference:**
   - High similarity: 30-40% ± 5%
   - Medium similarity: 15-25% ± 5%
   - Low similarity: 0-10% ± 5%
   - Statistical significance: p < 0.001

2. **Retroactive Interference:**
   - High similarity: 40-50% ± 5%
   - Medium similarity: 20-30% ± 5%
   - Statistical significance: p < 0.001

3. **Fan Effect:**
   - Fan 2: +100-150ms ± 25ms
   - Fan 3: +200-300ms ± 50ms
   - Fan 4: +300-450ms ± 75ms
   - Logarithmic fit: R² > 0.95

**Reporting:**
```rust
pub struct ValidationReport {
    pub proactive: TestResults,
    pub retroactive: TestResults,
    pub fan: TestResults,
    pub overall_pass: bool,
    pub effect_sizes: HashMap<String, f32>,
    pub confidence_intervals: HashMap<String, (f32, f32)>,
    pub p_values: HashMap<String, f32>,
}
```

## Integration Testing

Beyond individual phenomena, test interactions:

```rust
#[test]
fn test_interference_interactions() {
    // Does proactive interference persist through reconsolidation?
    // Does fan effect modulate interference magnitude?
    // Do spacing effects reduce interference?

    // These integration tests validate the full cognitive architecture
}
```

## Performance Budget

Running full suite:
- 10,000-15,000 total trials
- Each trial: encoding + retrieval = ~2ms
- Total time: 20-30 seconds
- Acceptable for nightly validation, not PR checks

For PR checks, run abbreviated suite:
- 100 trials per condition
- Total time: <1 second
- Provides sanity check, not full validation
