# Spacing Effect Validation: Research and Technical Foundation

## The Robust Phenomenon

The spacing effect is one of the most reliable findings in all of psychology: spaced repetition produces better retention than massed repetition. Cepeda et al. (2006) meta-analyzed 317 experiments and found consistent 20-40% retention improvement from optimal spacing.

Example: studying a concept three times
- Massed: study 3 times in one session → 50% retention at 1 week
- Spaced: study 3 times across 3 days → 70% retention at 1 week
- Improvement: 40% relative gain

The effect holds across:
- Ages (children to elderly)
- Materials (facts, skills, concepts)
- Retention intervals (hours to years)
- Species (humans, pigeons, bees)

## Optimal Spacing Parameters

Cepeda et al. (2006) found the relationship between spacing and retention is an inverted U:

- Too little spacing: approaches massed, minimal benefit
- Optimal spacing: typically 10-20% of retention interval
- Too much spacing: forgetting occurs between repetitions, benefit decreases

For 7-day retention interval:
- 1-day spacing: ~35% improvement
- 2-day spacing: ~40% improvement (optimal)
- 4-day spacing: ~25% improvement
- 7-day spacing: ~10% improvement

The optimal spacing shifts with retention interval:
- 1-day retention: optimal spacing ~2-4 hours
- 7-day retention: optimal spacing ~1-2 days
- 30-day retention: optimal spacing ~5-7 days

## Theoretical Mechanisms

**Encoding Variability (Estes, 1955):**
Each study episode encodes context. Spaced repetitions encode multiple contexts, creating more retrieval cues. Retrieval from varied contexts generalizes better.

**Deficient Processing (Braun & Rubin, 1998):**
Massed repetitions are processed superficially - they feel familiar, so less cognitive effort is applied. Spaced repetitions feel less familiar, triggering deeper encoding.

**Consolidation (Wickelgren, 1972):**
Memory consolidation takes hours. Spacing allows consolidation to complete before next repetition, building on stable foundation rather than interfering with ongoing consolidation.

Our implementation focuses on the consolidation mechanism, as it maps directly to our M6 consolidation pipeline.

## Implementation Architecture

```rust
pub struct SpacingEffectTracker {
    repetitions: HashMap<NodeId, Vec<RepetitionEvent>>,
}

pub struct RepetitionEvent {
    timestamp: Instant,
    strength: f32,
    consolidation_state: ConsolidationState,
}

impl SpacingEffectTracker {
    pub fn record_repetition(&mut self, node_id: NodeId, strength: f32, state: ConsolidationState) {
        self.repetitions.entry(node_id)
            .or_insert_with(Vec::new)
            .push(RepetitionEvent {
                timestamp: Instant::now(),
                strength,
                consolidation_state: state,
            });
    }

    pub fn compute_spacing_benefit(&self, node_id: NodeId) -> f32 {
        let reps = match self.repetitions.get(&node_id) {
            Some(r) if r.len() >= 2 => r,
            _ => return 1.0,  // No spacing benefit with <2 repetitions
        };

        let mut total_benefit = 1.0;

        for window in reps.windows(2) {
            let interval = window[1].timestamp.duration_since(window[0].timestamp);
            let interval_hours = interval.as_secs_f32() / 3600.0;

            // Benefit peaks at ~10-20% of retention interval
            // Assuming 7-day (168h) retention: optimal ~20-40h spacing
            let optimal_hours = 30.0;
            let spacing_quality = (-((interval_hours - optimal_hours).powi(2)) / 200.0).exp();

            // Each well-spaced repetition adds 20-40% benefit
            total_benefit += 0.3 * spacing_quality;
        }

        total_benefit.min(1.8)  // Cap at 80% total improvement
    }

    pub fn apply_spacing_benefit(&self, base_strength: f32, node_id: NodeId) -> f32 {
        let benefit = self.compute_spacing_benefit(node_id);
        base_strength * benefit
    }
}
```

## Validation Criteria

**Target: Cepeda et al. (2006) meta-analytic findings**

**Must Match:**
- Optimal spacing for 7-day retention: 20-40% improvement
- Massed vs spaced (3 reps): 30-50% difference
- Statistical significance: p < 0.001

**Should Match:**
- Inverted-U spacing curve (too little and too much spacing both sub-optimal)
- Optimal spacing ~10-20% of retention interval
- Benefit magnitude 20-40% across multiple retention intervals

**Statistical Requirements:**
- N >= 500 trials per spacing condition
- 3-5 spacing conditions (massed, near-optimal, optimal, far-optimal, very-far)
- Counterbalanced presentation order
- Power = 0.80 for d = 0.5 effects

## Validation Protocol

```rust
#[test]
fn validate_spacing_effect_cepeda2006() {
    let retention_interval = Duration::from_days(7);
    let conditions = vec![
        ("massed", Duration::from_hours(0)),
        ("1-day", Duration::from_hours(24)),
        ("2-day", Duration::from_hours(48)),  // Expected optimal
        ("4-day", Duration::from_hours(96)),
        ("7-day", Duration::from_hours(168)),
    ];

    let mut results = HashMap::new();

    for (condition_name, spacing_interval) in conditions {
        let mut retention_rates = Vec::new();

        for trial in 0..500 {
            let word = random_word();

            // First presentation
            memory.encode(word, strength=1.0);
            advance_time(spacing_interval);

            // Second presentation
            memory.encode(word, strength=1.0);
            advance_time(spacing_interval);

            // Third presentation
            memory.encode(word, strength=1.0);

            // Wait for retention interval
            advance_time(retention_interval);

            // Test retrieval
            let retrieved_strength = memory.retrieve(word).strength;
            retention_rates.push(retrieved_strength);

            memory.clear();
        }

        results.insert(condition_name, mean(&retention_rates));
    }

    // Optimal spacing (2-day) should show highest retention
    let optimal_retention = results["2-day"];
    let massed_retention = results["massed"];

    let improvement = (optimal_retention - massed_retention) / massed_retention;

    // Cepeda et al.: expect 20-40% improvement
    assert!(improvement >= 0.20 && improvement <= 0.40,
        "Spacing improvement {}% outside expected 20-40%", improvement * 100.0);

    // Check inverted-U: 2-day should beat 1-day and 4-day
    assert!(results["2-day"] > results["1-day"]);
    assert!(results["2-day"] > results["4-day"]);
}
```

## Integration with Consolidation

The spacing effect emerges from interaction with M6 consolidation:

1. **First repetition:** Initiates STM→LTM transfer
2. **Spacing interval:** Allows consolidation to complete (6-12 hours)
3. **Second repetition:** Strengthens consolidated trace, re-enters consolidation
4. **Result:** Each repetition builds on consolidated foundation

Massed repetitions interfere with ongoing consolidation, reducing cumulative benefit.

Implementation requires tracking consolidation state:

```rust
pub enum ConsolidationState {
    InSTM,
    Consolidating { progress: f32 },  // 0.0 to 1.0
    InLTM,
}

pub fn optimal_spacing_interval(state: ConsolidationState) -> Duration {
    match state {
        ConsolidationState::InSTM => Duration::from_hours(6),  // Wait for consolidation start
        ConsolidationState::Consolidating { progress } => {
            // Wait for ~75% consolidation
            Duration::from_hours((12.0 * (0.75 - progress)).max(0.0) as u64)
        }
        ConsolidationState::InLTM => Duration::from_hours(24),  // Standard spacing
    }
}
```

## Performance Implications

Spacing effect validation requires:
- Long time scales: 7-day retention × 500 trials = 3500 simulated days
- Can accelerate by scaling time: 1 hour real = 1 day simulated
- Consolidation must scale accordingly
- Approximately 10-20 minutes wall-clock time for full validation

This makes spacing validation one of the longest-running tests, appropriate for nightly CI rather than PR checks.
