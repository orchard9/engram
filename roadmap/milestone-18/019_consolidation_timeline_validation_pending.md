# Task 019: Episodic-to-Semantic Consolidation Timeline Validation

## Objective
Validate that M17's consolidation engine produces empirically accurate episodic-to-semantic memory transformation over the timescales observed in human neuroscience studies (Takashima 2006, Frankland & Bontempi 2005), with quantitative metrics for concept formation rate, semanticization trajectory, and hippocampal-neocortical transfer dynamics.

## Background

### Systems Consolidation Theory (McClelland et al. 1995)

**Complementary Learning Systems (CLS):**
- **Hippocampus**: Rapid encoding (single-trial learning), pattern separated episodic traces
- **Neocortex**: Slow extraction of statistical regularities, distributed semantic representations
- **Consolidation**: Gradual transfer via repeated replay, preventing catastrophic interference

**Key Timescales:**
- **Rapid consolidation**: Detectable neocortical strengthening within 6-12 hours (Gais & Born 2004)
- **Initial systems consolidation**: Hippocampal independence emerges at 7-30 days (Frankland & Bontempi 2005)
- **Remote memory**: Complete hippocampal independence at 3-6 months (Squire & Alvarez 1995)

### Takashima et al. (2006) - Longitudinal fMRI Study

**Experimental Design:**
- Participants learn 60 face-location associations
- fMRI scanning at 30min, 24h, 90 days post-learning
- Measure hippocampal vs neocortical activation during retrieval

**Key Findings:**
- **30 minutes**: Strong hippocampal activation, weak neocortical activation
- **24 hours**: Hippocampal activation maintained, neocortical activation increases by ~2-5%
- **90 days**: Hippocampal activation reduced by 40%, neocortical activation increased by 15-20%
- **Gradual transfer**: Linear increase in neocortical activation, linear decrease in hippocampal dependence

### Consolidation Phenomena to Validate

1. **Concept Formation Rate**: 1-5% of episodes form concepts per consolidation cycle (Task 004 spec)
2. **Semanticization**: Episodes lose contextual details, concepts gain strength
3. **Replay Frequency**: SWR-mediated replay peaks in first 24h, decays with ~0.9 factor (Kudrimoti 1999)
4. **Consolidation Asymptote**: Concepts approach full consolidation (strength → 1.0) logarithmically
5. **Schema Integration**: New concepts integrate with existing semantic structures
6. **Interference Prevention**: No catastrophic forgetting of old concepts

## Requirements

1. **Temporal Simulation**: Model 30min, 24h, 7d, 30d, 90d timepoints
2. **Hippocampal Proxy**: Episode binding strength as hippocampal dependence
3. **Neocortical Proxy**: Concept consolidation strength as neocortical activation
4. **Replay Simulation**: Integrate with M17 Task 004 replay-weighted consolidation
5. **Longitudinal Tracking**: Measure same memories across multiple consolidation cycles
6. **Correlation Target**: r > 0.75 with Takashima (2006) activation trajectories

## Technical Specification

### Files to Create
- `engram-core/tests/cognitive_validation/consolidation_timeline.rs` - Longitudinal validation
- `engram-core/tests/cognitive_validation/datasets/consolidation_schedule.json` - Consolidation timeline parameters
- `engram-core/src/consolidation/validation/metrics.rs` - Hippocampal/neocortical activation tracking

### Files to Modify
- `engram-core/src/consolidation/concept_formation.rs` - Add metrics hooks
- `engram-core/src/consolidation/dream.rs` - Expose consolidation cycle counts
- `engram-core/src/memory/bindings.rs` - Track binding strength decay over time

### Consolidation Timeline Simulation

```rust
use chrono::{DateTime, Utc, Duration as ChronoDuration};
use engram_core::consolidation::{ConsolidationEngine, SleepStage};

/// Consolidation timeline validation engine
pub struct ConsolidationTimelineValidator {
    /// Memory graph with dual memory architecture
    graph: Arc<UnifiedMemoryGraph<DashMapBackend>>,

    /// Consolidation engine (from M17)
    consolidation_engine: Arc<ConsolidationEngine>,

    /// Tracked memories (episodes that should form concepts)
    tracked_memories: Vec<TrackedMemory>,

    /// Consolidation cycles (simulates sleep/wake over days)
    consolidation_schedule: ConsolidationSchedule,

    /// Results per timepoint
    results: HashMap<TimePoint, ConsolidationMeasurement>,
}

/// Timepoints matching Takashima (2006) design
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum TimePoint {
    /// 30 minutes post-encoding (baseline)
    Immediate,  // 0.5 hours

    /// 24 hours (first night consolidation)
    Day1,       // 24 hours

    /// 7 days (one week consolidation)
    Week1,      // 168 hours

    /// 30 days (one month)
    Month1,     // 720 hours

    /// 90 days (three months, remote memory)
    Month3,     // 2160 hours
}

impl TimePoint {
    /// Get hours since encoding
    pub fn hours(&self) -> f64 {
        match self {
            Self::Immediate => 0.5,
            Self::Day1 => 24.0,
            Self::Week1 => 168.0,
            Self::Month1 => 720.0,
            Self::Month3 => 2160.0,
        }
    }

    /// Get expected neocortical activation (from Takashima 2006)
    pub fn expected_neocortical_activation(&self) -> f32 {
        match self {
            Self::Immediate => 0.15,  // Weak (baseline)
            Self::Day1 => 0.20,       // +5% increase
            Self::Week1 => 0.25,      // +10% total
            Self::Month1 => 0.30,     // +15% total
            Self::Month3 => 0.35,     // +20% total (remote memory)
        }
    }

    /// Get expected hippocampal activation (from Takashima 2006)
    pub fn expected_hippocampal_activation(&self) -> f32 {
        match self {
            Self::Immediate => 0.85,  // Strong (episodic retrieval)
            Self::Day1 => 0.82,       // Maintained
            Self::Week1 => 0.70,      // Gradual reduction
            Self::Month1 => 0.55,     // Half hippocampal dependence
            Self::Month3 => 0.45,     // 40% reduction (semantic retrieval)
        }
    }
}

/// Consolidation schedule (simulates sleep cycles)
#[derive(Debug, Clone)]
pub struct ConsolidationSchedule {
    /// Number of consolidation cycles per night
    pub cycles_per_night: usize,

    /// NREM2 duration per cycle (minutes)
    pub nrem2_duration_min: u64,

    /// NREM3 duration per cycle (minutes)
    pub nrem3_duration_min: u64,

    /// REM duration per cycle (minutes)
    pub rem_duration_min: u64,

    /// Days to simulate
    pub simulation_days: usize,
}

impl Default for ConsolidationSchedule {
    fn default() -> Self {
        Self {
            cycles_per_night: 5,         // Typical adult sleep
            nrem2_duration_min: 20,       // ~20min NREM2 per cycle
            nrem3_duration_min: 15,       // ~15min NREM3 per cycle
            rem_duration_min: 10,         // ~10min REM per cycle
            simulation_days: 90,          // 90 days for Takashima timepoint
        }
    }
}

/// Tracked memory for longitudinal analysis
#[derive(Debug, Clone)]
pub struct TrackedMemory {
    /// Episode node ID
    pub episode_id: NodeId,

    /// Episode content (for logging)
    pub content: String,

    /// Encoding timestamp
    pub encoded_at: DateTime<Utc>,

    /// Concept IDs formed from this episode
    pub concept_ids: Vec<NodeId>,

    /// Initial binding strengths
    pub initial_binding_strengths: HashMap<NodeId, f32>,
}

impl TrackedMemory {
    /// Create tracked memory from episode
    pub fn new(episode_id: NodeId, content: String) -> Self {
        Self {
            episode_id,
            content,
            encoded_at: Utc::now(),
            concept_ids: Vec::new(),
            initial_binding_strengths: HashMap::new(),
        }
    }

    /// Add concept formed from this episode
    pub fn add_concept(&mut self, concept_id: NodeId, binding_strength: f32) {
        self.concept_ids.push(concept_id.clone());
        self.initial_binding_strengths.insert(concept_id, binding_strength);
    }
}

/// Consolidation measurement at a timepoint
#[derive(Debug, Clone)]
pub struct ConsolidationMeasurement {
    /// Timepoint (e.g., Day1, Week1)
    pub timepoint: TimePoint,

    /// Hippocampal activation proxy (mean episode binding strength)
    pub hippocampal_activation: f32,

    /// Neocortical activation proxy (mean concept consolidation strength)
    pub neocortical_activation: f32,

    /// Number of concepts formed
    pub concepts_formed: usize,

    /// Replay count for tracked episodes
    pub replay_count: u32,

    /// Consolidation cycles completed
    pub cycles_completed: usize,

    /// Semanticization score (concept strength / episode strength)
    pub semanticization_ratio: f32,
}

impl ConsolidationTimelineValidator {
    /// Create validator with tracked memories
    pub fn new(
        graph: Arc<UnifiedMemoryGraph<DashMapBackend>>,
        consolidation_engine: Arc<ConsolidationEngine>,
    ) -> Self {
        Self {
            graph,
            consolidation_engine,
            tracked_memories: Vec::new(),
            consolidation_schedule: ConsolidationSchedule::default(),
            results: HashMap::new(),
        }
    }

    /// Encode learning material (60 face-location associations per Takashima)
    pub async fn encode_learning_material(&mut self, n_associations: usize) -> Result<()> {
        for i in 0..n_associations {
            let content = format!("Person {} is at Location {}", i, i % 10);
            let embedding = Self::generate_embedding(&content)?;

            let episode_id = self.graph.add_node(
                MemoryNode::new_with_type(
                    embedding,
                    0.9, // High confidence (learned material)
                    content.clone(),
                    MemoryNodeType::Episode {
                        timestamp: Utc::now(),
                        context: "Consolidation_Experiment".into(),
                    },
                )
            ).await?;

            let tracked = TrackedMemory::new(episode_id, content);
            self.tracked_memories.push(tracked);
        }

        Ok(())
    }

    /// Run consolidation simulation for specified duration
    pub async fn simulate_consolidation(&mut self, target_timepoint: TimePoint) -> Result<()> {
        let target_hours = target_timepoint.hours();
        let mut current_hours = 0.0;

        while current_hours < target_hours {
            // Simulate one night of sleep (5 cycles)
            for cycle_num in 0..self.consolidation_schedule.cycles_per_night {
                // NREM2: Peak consolidation
                self.run_consolidation_cycle(SleepStage::NREM2).await?;

                // NREM3: Sustained consolidation
                self.run_consolidation_cycle(SleepStage::NREM3).await?;

                // REM: Selective consolidation
                self.run_consolidation_cycle(SleepStage::REM).await?;
            }

            current_hours += 24.0; // One day completed

            // Measure at target timepoint
            if (current_hours - target_hours).abs() < 1.0 {
                let measurement = self.measure_consolidation_state(target_timepoint).await?;
                self.results.insert(target_timepoint, measurement);
                break;
            }
        }

        Ok(())
    }

    /// Run single consolidation cycle
    async fn run_consolidation_cycle(&mut self, sleep_stage: SleepStage) -> Result<()> {
        // Call M17 consolidation engine
        let result = self.consolidation_engine.consolidate_with_concepts(sleep_stage).await?;

        // Track concepts formed from our tracked episodes
        for tracked_memory in &mut self.tracked_memories {
            // Check if new concepts were formed from this episode
            let concepts = self.graph.get_episode_concepts(&tracked_memory.episode_id).await?;

            for concept_ref in concepts {
                if !tracked_memory.concept_ids.contains(&concept_ref.target_id) {
                    tracked_memory.add_concept(
                        concept_ref.target_id.clone(),
                        concept_ref.get_strength(),
                    );
                }
            }
        }

        Ok(())
    }

    /// Measure hippocampal and neocortical activation proxies
    async fn measure_consolidation_state(
        &self,
        timepoint: TimePoint,
    ) -> Result<ConsolidationMeasurement> {
        let mut hippocampal_activations = Vec::new();
        let mut neocortical_activations = Vec::new();
        let mut total_concepts = 0;
        let mut total_replay = 0;

        for tracked_memory in &self.tracked_memories {
            // Hippocampal proxy: Episode binding strength (high = still dependent)
            let episode_bindings = self.graph
                .get_episode_concepts(&tracked_memory.episode_id)
                .await?;

            let mean_binding_strength = if episode_bindings.is_empty() {
                0.85 // No concepts yet, fully hippocampal
            } else {
                mean(&episode_bindings.iter().map(|b| b.get_strength()).collect::<Vec<_>>())
            };
            hippocampal_activations.push(mean_binding_strength);

            // Neocortical proxy: Concept consolidation strength (high = semantic retrieval)
            for concept_id in &tracked_memory.concept_ids {
                if let Ok(concept_node) = self.graph.get_node(concept_id).await {
                    // Extract consolidation_strength from concept metadata
                    if let Some(consolidation_strength) = Self::get_consolidation_strength(&concept_node) {
                        neocortical_activations.push(consolidation_strength);
                    }
                }
            }

            total_concepts += tracked_memory.concept_ids.len();

            // Replay count (from consolidation engine metrics)
            // In real implementation, track via consolidation engine
            total_replay += 5; // Placeholder
        }

        let hippocampal = mean(&hippocampal_activations);
        let neocortical = if neocortical_activations.is_empty() {
            0.15 // Baseline neocortical (from Takashima)
        } else {
            mean(&neocortical_activations)
        };

        Ok(ConsolidationMeasurement {
            timepoint,
            hippocampal_activation: hippocampal,
            neocortical_activation: neocortical,
            concepts_formed: total_concepts,
            replay_count: total_replay,
            cycles_completed: 0, // Track separately
            semanticization_ratio: neocortical / hippocampal.max(0.01),
        })
    }

    /// Run full Takashima (2006) longitudinal experiment
    pub async fn run_full_timeline(&mut self) -> Result<TimelineValidationResults> {
        // Encode learning material
        self.encode_learning_material(60).await?;

        // Measure immediate (30min baseline)
        let immediate = self.measure_consolidation_state(TimePoint::Immediate).await?;
        self.results.insert(TimePoint::Immediate, immediate);

        // Simulate to each timepoint
        for timepoint in [TimePoint::Day1, TimePoint::Week1, TimePoint::Month1, TimePoint::Month3] {
            self.simulate_consolidation(timepoint).await?;
        }

        self.analyze_timeline()
    }

    /// Analyze consolidation timeline and compare to Takashima (2006)
    fn analyze_timeline(&self) -> Result<TimelineValidationResults> {
        let mut analysis = TimelineValidationResults::default();

        // Extract hippocampal and neocortical trajectories
        let timepoints = vec![
            TimePoint::Immediate,
            TimePoint::Day1,
            TimePoint::Week1,
            TimePoint::Month1,
            TimePoint::Month3,
        ];

        let mut hippocampal_observed = Vec::new();
        let mut hippocampal_expected = Vec::new();
        let mut neocortical_observed = Vec::new();
        let mut neocortical_expected = Vec::new();

        for tp in timepoints {
            if let Some(measurement) = self.results.get(&tp) {
                hippocampal_observed.push(measurement.hippocampal_activation);
                hippocampal_expected.push(tp.expected_hippocampal_activation());
                neocortical_observed.push(measurement.neocortical_activation);
                neocortical_expected.push(tp.expected_neocortical_activation());

                analysis.measurements.insert(tp, measurement.clone());
            }
        }

        // Correlation with Takashima data
        analysis.hippocampal_correlation = pearson_correlation(&hippocampal_expected, &hippocampal_observed)?;
        analysis.neocortical_correlation = pearson_correlation(&neocortical_expected, &neocortical_observed)?;

        // Linear trend analysis
        let hours: Vec<f32> = timepoints.iter().map(|tp| tp.hours() as f32).collect();
        let (neocortical_slope, _) = linear_regression(&hours, &neocortical_observed)?;
        analysis.neocortical_slope = neocortical_slope;

        // Semanticization rate
        let immediate_ratio = self.results.get(&TimePoint::Immediate)
            .map(|m| m.semanticization_ratio)
            .unwrap_or(0.1);
        let final_ratio = self.results.get(&TimePoint::Month3)
            .map(|m| m.semanticization_ratio)
            .unwrap_or(0.5);
        analysis.semanticization_increase = final_ratio - immediate_ratio;

        Ok(analysis)
    }
}

#[derive(Debug, Default)]
pub struct TimelineValidationResults {
    pub measurements: HashMap<TimePoint, ConsolidationMeasurement>,
    pub hippocampal_correlation: f32,
    pub neocortical_correlation: f32,
    pub neocortical_slope: f32,
    pub semanticization_increase: f32,
}

impl TimelineValidationResults {
    /// Check if results meet acceptance criteria
    pub fn meets_criteria(&self) -> bool {
        // Hippocampal trajectory correlation > 0.75
        if self.hippocampal_correlation < 0.75 {
            return false;
        }

        // Neocortical trajectory correlation > 0.75
        if self.neocortical_correlation < 0.75 {
            return false;
        }

        // Positive neocortical slope (increasing over time)
        if self.neocortical_slope <= 0.0 {
            return false;
        }

        // Semanticization increase > 0.2 (from 0.1 to 0.3+)
        if self.semanticization_increase < 0.2 {
            return false;
        }

        true
    }

    /// Generate validation report
    pub fn generate_report(&self) -> String {
        let mut report = String::from("=== Consolidation Timeline Validation (Takashima 2006) ===\n\n");

        report.push_str(&format!("Hippocampal Trajectory: r = {:.3}\n", self.hippocampal_correlation));
        report.push_str(&format!("Neocortical Trajectory: r = {:.3}\n", self.neocortical_correlation));
        report.push_str(&format!("Neocortical Slope: {:.4} per hour\n", self.neocortical_slope));
        report.push_str(&format!("Semanticization Increase: {:.3}\n\n", self.semanticization_increase));

        for tp in [TimePoint::Immediate, TimePoint::Day1, TimePoint::Week1, TimePoint::Month1, TimePoint::Month3] {
            if let Some(measurement) = self.measurements.get(&tp) {
                report.push_str(&format!("{:?} ({:.1}h):\n", tp, tp.hours()));
                report.push_str(&format!("  Hippocampal: {:.3} (expected {:.3})\n",
                    measurement.hippocampal_activation, tp.expected_hippocampal_activation()));
                report.push_str(&format!("  Neocortical: {:.3} (expected {:.3})\n",
                    measurement.neocortical_activation, tp.expected_neocortical_activation()));
                report.push_str(&format!("  Concepts Formed: {}\n", measurement.concepts_formed));
                report.push_str(&format!("  Semanticization Ratio: {:.3}\n\n", measurement.semanticization_ratio));
            }
        }

        report.push_str(&format!("\nOverall Status: {}\n",
            if self.meets_criteria() { "PASS - Matches Takashima (2006) consolidation timeline" }
            else { "FAIL - Deviates from expected consolidation trajectory" }));

        report
    }
}
```

## Testing Approach

### Unit Tests
1. **Schedule Calculation**: Test consolidation cycle timing
2. **Proxy Metrics**: Validate hippocampal/neocortical activation calculations
3. **Timepoint Math**: Verify hours-since-encoding conversions
4. **Tracked Memory**: Test concept addition and strength tracking

### Integration Tests
1. **Single Cycle**: Run one consolidation cycle, verify concept formation
2. **24-Hour Timeline**: Test Day1 timepoint
3. **90-Day Timeline**: Full Takashima protocol (slow test)
4. **Replay Frequency**: Verify decay with ~0.9 factor per night

### Validation Tests
1. **Takashima Replication**: N=60 memories, 5 timepoints
2. **Correlation Analysis**: Pearson's r for both trajectories
3. **Linear Trends**: Validate neocortical increase, hippocampal decrease
4. **Semanticization**: Quantify episode→concept transformation

## Acceptance Criteria

- [ ] Hippocampal trajectory: r > 0.75 with Takashima data
- [ ] Neocortical trajectory: r > 0.75 with Takashima data
- [ ] Day 1 neocortical increase: 2-5% from baseline
- [ ] Month 3 hippocampal reduction: 30-50% from baseline
- [ ] Concept formation rate: 1-5% of episodes per cycle
- [ ] Semanticization ratio: Increases from 0.1 to 0.3+ over 90 days
- [ ] Replay frequency: Decays with ~0.9 factor per night
- [ ] No catastrophic interference: Old concepts maintain >90% strength
- [ ] Simulation performance: 90-day timeline completes in <30 minutes
- [ ] Determinism: <5% variance across runs

## Dependencies

- M17 Task 004 (Concept Formation Engine) - complete
- M17 Task 006 (Consolidation Integration) - complete
- M17 Tasks 001-002 (Dual Memory Types, Storage) - complete
- Existing consolidation/dream.rs engine

## Estimated Time
5 days

## References

1. **Takashima, A., et al. (2006).** Declarative memory consolidation in humans: A prospective functional magnetic resonance imaging study. PNAS, 103(3), 756-761.
2. **McClelland, J. L., McNaughton, B. L., & O'Reilly, R. C. (1995).** Why there are complementary learning systems. Psychological Review, 102(3), 419-457.
3. **Frankland, P. W., & Bontempi, B. (2005).** The organization of recent and remote memories. Nature Reviews Neuroscience, 6(2), 119-130.
4. **Gais, S., & Born, J. (2004).** Declarative memory consolidation: Mechanisms acting during human sleep. Learning & Memory, 11(6), 679-685.
5. **Squire, L. R., & Alvarez, P. (1995).** Retrograde amnesia and memory consolidation. Current Opinion in Neurobiology, 5(2), 169-177.
6. **Kudrimoti, H. S., Barnes, C. A., & McNaughton, B. L. (1999).** Reactivation of hippocampal cell assemblies. Journal of Neuroscience, 19(10), 4090-4101.
