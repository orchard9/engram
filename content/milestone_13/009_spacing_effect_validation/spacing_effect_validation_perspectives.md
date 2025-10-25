# Spacing Effect Validation: Architectural Perspectives

## Cognitive Architecture Designer

The spacing effect, first documented by Ebbinghaus (1885) and meta-analyzed comprehensively by Cepeda et al. (2006), demonstrates that distributed practice produces 20-40% better retention than massed practice. Learning "cat-dog" three times with 10-minute intervals yields stronger memory than three repetitions back-to-back.

From a biological perspective, the spacing effect emerges from synaptic tagging and capture mechanisms (Redondo & Morris, 2011). Each learning episode tags synapses for potential strengthening. Spaced repetitions, arriving when tags are still active but initial plasticity has partially decayed, trigger protein synthesis that consolidates the memory. Massed repetitions arrive before decay begins, wasting consolidation potential.

The temporal dynamics are precise: optimal spacing depends on the retention interval. For retention of 1 week, optimal spacing is 12-24 hours (Cepeda et al., 2006). For retention of 1 month, optimal spacing expands to 2-4 days. This reflects that spacing should be proportional to the forgetting curve - repeat when memory has partially decayed but not completely disappeared.

Engram's implementation must capture this decay-sensitive timing. The consolidation pipeline (Milestone 6) already implements exponential decay (tau = 24 hours). Spacing effect validation tests whether repeated encoding at different intervals produces the predicted strength differences, matching the 20-40% improvement from optimal spacing.

## Memory Systems Researcher

Cepeda et al. (2006) conducted a meta-analysis of 317 spacing effect experiments, establishing robust statistical relationships between spacing interval, retention interval, and memory strength. Key findings:

**Optimal Spacing Function:**
For retention interval R, optimal spacing S ≈ 0.1R to 0.2R
- Retain 7 days: space 17-34 hours
- Retain 30 days: space 3-6 days
- Retain 1 year: space 36-72 days

**Effect Size:**
Cohen's d = 0.4-0.8 for optimal spacing vs massed practice (medium to large effect)

**Dose-Response:**
More spacing repetitions increase benefit with diminishing returns (3 spaced > 2 spaced, but increment decreases)

Statistical validation requires within-subjects designs where each participant serves as their own control:
- Condition A: Massed practice (3 repetitions, 0-second spacing)
- Condition B: Short spacing (3 repetitions, 1-hour spacing)
- Condition C: Optimal spacing (3 repetitions, 24-hour spacing)
- Condition D: Long spacing (3 repetitions, 7-day spacing)

Measure retention at 7-day interval. Expected result: C > B > D > A (optimal spacing best, massed worst).

Acceptance criteria:
1. Optimal spacing shows 20-40% higher retention than massed (95% CI)
2. Spacing-retention relationship follows inverted-U curve (quadratic fit R² > 0.7)
3. Effect size d > 0.5 for optimal vs massed comparison (p < 0.001)

## Rust Graph Engine Architect

Implementing spacing effect validation requires tracking repetition timing and measuring final memory strength after controlled retention intervals:

```rust
pub struct SpacingEffectValidator {
    /// Learning episodes with timestamps
    learning_history: HashMap<NodeId, Vec<LearningEpisode>>,

    /// Retention interval for testing
    retention_interval: Duration,

    /// Memory graph
    graph: Arc<MemoryGraph>,
}

#[derive(Clone, Debug)]
struct LearningEpisode {
    timestamp: Timestamp,
    strength: f32,
    spacing_from_previous: Option<Duration>,
}

impl SpacingEffectValidator {
    pub async fn test_spacing_condition(
        &mut self,
        node: NodeId,
        num_repetitions: usize,
        spacing_interval: Duration,
    ) -> Result<SpacingTestResult> {
        let mut episodes = Vec::new();

        // Encode with controlled spacing
        for rep in 0..num_repetitions {
            let timestamp = Timestamp::now();
            self.graph.encode_node(node, 0.8).await?;

            let spacing = if rep > 0 {
                Some(spacing_interval)
            } else {
                None
            };

            episodes.push(LearningEpisode {
                timestamp,
                strength: 0.8,
                spacing_from_previous: spacing,
            });

            // Wait for spacing interval before next repetition
            if rep < num_repetitions - 1 {
                tokio::time::sleep(spacing_interval).await;
            }
        }

        self.learning_history.insert(node, episodes);

        // Wait for retention interval
        tokio::time::sleep(self.retention_interval).await;

        // Measure final memory strength
        let final_strength = self.graph.get_strength(node).await?;

        Ok(SpacingTestResult {
            node,
            num_repetitions,
            spacing_interval,
            retention_interval: self.retention_interval,
            final_strength,
        })
    }
}
```

Performance targets: each test condition runs in real time (waiting for spacing intervals), but measurement overhead is <100μs. Statistical validation requires running 4 conditions × 20 items = 80 tests, parallelizable across items.

## Systems Architecture Optimizer

The spacing effect validation creates opportunities for accelerated testing using time dilation. Rather than waiting 24 hours for optimal spacing, we can scale time in the consolidation dynamics:

```rust
pub struct AcceleratedSpacingValidator {
    /// Time dilation factor (1.0 = real time, 1000.0 = 1000x faster)
    time_dilation: f32,

    /// Consolidation with dilated time
    consolidation: Arc<AcceleratedConsolidation>,
}

impl AcceleratedSpacingValidator {
    pub async fn test_spacing_accelerated(
        &mut self,
        spacing_interval: Duration,
    ) -> Result<SpacingTestResult> {
        // Dilate spacing interval
        let dilated_spacing = Duration::from_millis(
            (spacing_interval.as_millis() as f32 / self.time_dilation) as u64
        );

        // Run test with dilated time
        // Internal consolidation uses dilated clock, so 24 hours becomes 86.4 seconds @ 1000x

        // ...
    }
}
```

This enables running full spacing effect validation (including 7-day retention intervals) in ~10 minutes instead of days, critical for continuous integration testing.
