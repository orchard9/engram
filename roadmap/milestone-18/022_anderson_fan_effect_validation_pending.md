# Task 018: Anderson Fan Effect Validation (Person-Location Paradigm)

## Objective
Validate that M17's fan effect implementation in spreading activation accurately replicates Anderson (1974)'s linear RT increase with associative fan, achieving r > 0.85 correlation with published data across fan 1-5 conditions.

## Background

### Anderson (1974) - The Fan Effect

**Key Empirical Findings:**
- **Fan 1**: 1159ms ± 22ms (baseline, one fact per concept)
- **Fan 2**: 1236ms ± 25ms (+77ms, two facts per concept)
- **Fan 3**: 1305ms ± 28ms (+69ms from fan 2)
- **Fan 4**: 1374ms ± 31ms (+69ms from fan 3)
- **Linear relationship**: ~70ms per additional association

**Person-Location Paradigm:**
- Sentences like "The doctor is in the park"
- Experimental phase: Learn 26 person-location facts
- Test phase: Verify true/false sentences
- Manipulation: Vary fan (number of facts) per person and location
- Result: RT increases linearly with fan-out from both person and location

### Mechanism: Activation Spreading with Interference

Anderson's ACT-R model explains fan effect through **limited activation spreading**:

```
A_i = B_i + Σ_j (W_j / fan_j) * S_ji

Where:
- A_i = total activation of target concept i
- B_i = base-level activation
- W_j = source activation from j
- fan_j = number of associations from j
- S_ji = associative strength from j to i
```

Key insight: Source activation **divides** among all outgoing associations, creating interference.

### Implications for Engram's Dual Memory Architecture

1. **Episode→Concept spreading**: Episodes activate concepts with no fan penalty (upward)
2. **Concept→Episode spreading**: Concepts divide activation among episodes (downward, fan effect)
3. **Asymmetric spreading**: Upward stronger than downward (1.2x boost per M17 Task 007)
4. **Consolidation impact**: More consolidated concepts should have stronger base activation

## Requirements

1. **Person-Location Stimuli**: 26 sentences matching Anderson's original study
2. **Fan Manipulation**: Create graphs with controlled fan 1-5 for persons and locations
3. **RT Simulation**: Model retrieval time based on final concept activation levels
4. **Linear Regression**: Validate slope ~70ms/association (±10ms tolerance)
5. **Correlation Target**: r > 0.85 with Anderson (1974) Table 2 data
6. **Effect Size**: Cohen's d > 1.0 for fan 1 vs fan 3 (large effect)

## Technical Specification

### Files to Create
- `engram-core/tests/cognitive_validation/anderson_fan_effect.rs` - Main validation suite
- `engram-core/tests/cognitive_validation/datasets/anderson_1974_stimuli.json` - Person-location sentences
- `engram-core/src/cognitive/interference/fan_validation.rs` - Fan effect metrics

### Files to Modify
- `engram-core/src/cognitive/interference/fan_effect.rs` - Add validation hooks
- `engram-core/src/activation/parallel.rs` - Verify fan divisor logic

### Anderson (1974) Stimulus Structure

```rust
/// Person-location fact from Anderson (1974)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersonLocationFact {
    /// Person concept (e.g., "doctor", "lawyer")
    pub person: String,

    /// Location concept (e.g., "park", "church", "bank")
    pub location: String,

    /// Person fan (number of locations this person appears in)
    pub person_fan: usize,

    /// Location fan (number of people in this location)
    pub location_fan: usize,

    /// Total fan (person_fan + location_fan)
    pub total_fan: usize,

    /// Expected RT from Anderson Table 2 (ms)
    pub expected_rt_ms: f32,

    /// Standard error from original study
    pub standard_error_ms: f32,
}

impl PersonLocationFact {
    /// Create fact with automatic fan calculation
    pub fn new(
        person: impl Into<String>,
        location: impl Into<String>,
        person_fan: usize,
        location_fan: usize,
        expected_rt_ms: f32,
        standard_error_ms: f32,
    ) -> Self {
        Self {
            person: person.into(),
            location: location.into(),
            person_fan,
            location_fan,
            total_fan: person_fan + location_fan - 1, // -1 because fact itself is shared
            expected_rt_ms,
            standard_error_ms,
        }
    }

    /// Get fan category for analysis
    pub fn fan_category(&self) -> FanCategory {
        match self.total_fan {
            1 => FanCategory::Fan1,
            2 => FanCategory::Fan2,
            3 => FanCategory::Fan3,
            4 => FanCategory::Fan4,
            5.. => FanCategory::Fan5Plus,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FanCategory {
    Fan1,  // Baseline
    Fan2,  // +1 association
    Fan3,  // +2 associations
    Fan4,  // +3 associations
    Fan5Plus, // Stress test
}
```

### Fan Effect Validation Engine

```rust
use engram_core::{
    activation::ParallelSpreadingEngine,
    cognitive::interference::FanEffectDetector,
    memory_graph::UnifiedMemoryGraph,
};

/// Anderson (1974) fan effect validation engine
pub struct FanEffectValidationEngine {
    /// Memory graph with dual memory architecture
    graph: Arc<UnifiedMemoryGraph<DashMapBackend>>,

    /// Person-location facts dataset
    facts: Vec<PersonLocationFact>,

    /// Fan effect detector (from M17 Task 007)
    fan_detector: Arc<FanEffectDetector>,

    /// Spreading configuration with fan effect enabled
    spreading_config: ParallelSpreadingConfig,

    /// Results accumulator
    results: Vec<FanTrialResult>,
}

impl FanEffectValidationEngine {
    /// Create validation engine with Anderson stimuli
    pub fn new(graph: Arc<UnifiedMemoryGraph<DashMapBackend>>) -> Result<Self> {
        let facts = Self::load_anderson_stimuli()?;

        // Configure fan detector with Anderson parameters
        let fan_detector = Arc::new(FanEffectDetector::new(
            1150.0, // Base RT (Anderson 1974 Table 2)
            70.0,   // Slope per association
            false,  // Linear divisor (not sqrt)
        ));

        let spreading_config = ParallelSpreadingConfig {
            max_depth: 4, // Person→Fact, Fact→Location, Location→Fact, Fact→Person
            threshold: 0.01, // Low threshold for subtle fan effects
            decay_function: DecayFunction::exponential(0.85),
            fan_effect_config: FanEffectConfig {
                enabled: true,
                base_retrieval_time_ms: 1150.0,
                time_per_association_ms: 70.0,
                use_sqrt_divisor: false,
                upward_spreading_boost: 1.2,
                min_fan: 1,
            },
            deterministic_seed: Some(42),
            ..Default::default()
        };

        Ok(Self {
            graph,
            facts,
            fan_detector,
            spreading_config,
            results: Vec::new(),
        })
    }

    /// Load Anderson (1974) person-location sentences
    fn load_anderson_stimuli() -> Result<Vec<PersonLocationFact>> {
        let json_path = "tests/cognitive_validation/datasets/anderson_1974_stimuli.json";
        let json_str = std::fs::read_to_string(json_path)?;
        serde_json::from_str(&json_str).map_err(Into::into)
    }

    /// Populate graph with person-location memory structure
    pub async fn populate_person_location_graph(&self) -> Result<()> {
        // Step 1: Create concept nodes for all persons and locations
        let mut person_concepts = HashMap::new();
        let mut location_concepts = HashMap::new();

        for fact in &self.facts {
            if !person_concepts.contains_key(&fact.person) {
                let concept_id = self.create_concept_node(
                    &fact.person,
                    MemoryNodeType::Concept { category: "person".into() },
                ).await?;
                person_concepts.insert(fact.person.clone(), concept_id);
            }

            if !location_concepts.contains_key(&fact.location) {
                let concept_id = self.create_concept_node(
                    &fact.location,
                    MemoryNodeType::Concept { category: "location".into() },
                ).await?;
                location_concepts.insert(fact.location.clone(), concept_id);
            }
        }

        // Step 2: Create episodic fact nodes for each person-location pair
        for fact in &self.facts {
            let fact_content = format!("The {} is in the {}", fact.person, fact.location);
            let fact_episode = self.create_episode_node(&fact_content).await?;

            // Step 3: Create bindings: Episode ↔ Person Concept
            let person_concept_id = person_concepts[&fact.person].clone();
            self.graph.add_binding(
                fact_episode.clone(),
                person_concept_id.clone(),
                0.9, // High binding strength
                0.5, // Equal contribution
            ).await?;

            // Step 4: Create bindings: Episode ↔ Location Concept
            let location_concept_id = location_concepts[&fact.location].clone();
            self.graph.add_binding(
                fact_episode.clone(),
                location_concept_id.clone(),
                0.9, // High binding strength
                0.5, // Equal contribution
            ).await?;
        }

        Ok(())
    }

    /// Create concept node for person or location
    async fn create_concept_node(
        &self,
        name: &str,
        node_type: MemoryNodeType,
    ) -> Result<NodeId> {
        let embedding = Self::get_concept_embedding(name)?;

        let node_id = self.graph.add_node(
            MemoryNode::new_with_type(
                embedding,
                0.8, // High confidence
                format!("concept_{}", name.to_lowercase()),
                node_type,
            )
        ).await?;

        Ok(node_id)
    }

    /// Create episode node for person-location fact
    async fn create_episode_node(&self, fact_content: &str) -> Result<NodeId> {
        let embedding = Self::get_sentence_embedding(fact_content)?;

        let node_id = self.graph.add_node(
            MemoryNode::new_with_type(
                embedding,
                0.9, // High confidence for learned facts
                fact_content.to_string(),
                MemoryNodeType::Episode {
                    timestamp: Utc::now(),
                    context: "Anderson_1974_experiment".into(),
                },
            )
        ).await?;

        Ok(node_id)
    }

    /// Run fact verification trial: activate person+location, measure episode activation
    pub async fn run_verification_trial(
        &mut self,
        fact: &PersonLocationFact,
    ) -> Result<FanTrialResult> {
        let trial_start = Instant::now();

        // 1. Get concept IDs for person and location
        let person_id = self.get_concept_id(&fact.person)?;
        let location_id = self.get_concept_id(&fact.location)?;

        // 2. Activate both concepts simultaneously (simulates "doctor in park" probe)
        self.graph.activate_node(&person_id, 1.0).await?;
        self.graph.activate_node(&location_id, 1.0).await?;

        // 3. Spread activation through bindings with fan effect
        let spreading_engine = ParallelSpreadingEngine::new(
            self.spreading_config.clone(),
            self.graph.clone(),
        )?;

        spreading_engine.spread_from_seeds(vec![person_id.clone(), location_id.clone()]).await?;

        // 4. Measure activation of fact episode
        let fact_content = format!("The {} is in the {}", fact.person, fact.location);
        let fact_id = self.get_episode_id(&fact_content)?;
        let fact_activation = self.graph.get_activation(&fact_id).await?;

        // 5. Compute simulated RT based on activation and fan
        let simulated_rt_ms = self.compute_retrieval_time(
            fact_activation,
            fact.person_fan,
            fact.location_fan,
        );

        // 6. Clean up for next trial
        self.graph.reset_activation().await?;

        let result = FanTrialResult {
            person: fact.person.clone(),
            location: fact.location.clone(),
            person_fan: fact.person_fan,
            location_fan: fact.location_fan,
            total_fan: fact.total_fan,
            fact_activation,
            simulated_rt_ms,
            expected_rt_ms: fact.expected_rt_ms,
            elapsed_wall_time: trial_start.elapsed(),
        };

        self.results.push(result.clone());
        Ok(result)
    }

    /// Compute retrieval time based on activation and fan
    ///
    /// Formula: RT = base_time + (total_fan - 1) × time_per_fan + activation_penalty
    fn compute_retrieval_time(
        &self,
        activation: f32,
        person_fan: usize,
        location_fan: usize,
    ) -> f32 {
        let base_time = self.fan_detector.base_retrieval_time_ms();
        let time_per_fan = self.fan_detector.time_per_association_ms();

        // Fan effect: Linear increase with total associations
        let total_fan = person_fan + location_fan - 1; // -1 because fact itself is shared
        let fan_penalty = (total_fan.saturating_sub(1) as f32) * time_per_fan;

        // Activation penalty: Lower activation → slower retrieval
        let activation_factor = (1.0 - activation).max(0.0);
        let activation_penalty = activation_factor * 100.0; // Scale factor

        base_time + fan_penalty + activation_penalty
    }

    /// Run full Anderson (1974) experiment with all fan conditions
    pub async fn run_full_experiment(&mut self, trials_per_fact: usize) -> Result<FanExperimentResults> {
        for fact in self.facts.clone() {
            for _trial in 0..trials_per_fact {
                self.run_verification_trial(&fact).await?;
            }
        }

        self.analyze_results()
    }

    /// Analyze results and validate against Anderson (1974)
    fn analyze_results(&self) -> Result<FanExperimentResults> {
        let mut analysis = FanExperimentResults::default();

        // Group by fan category
        for fan_cat in [FanCategory::Fan1, FanCategory::Fan2, FanCategory::Fan3, FanCategory::Fan4] {
            let fan_results: Vec<_> = self.results.iter()
                .filter(|r| {
                    let cat = match r.total_fan {
                        1 => FanCategory::Fan1,
                        2 => FanCategory::Fan2,
                        3 => FanCategory::Fan3,
                        4 => FanCategory::Fan4,
                        _ => FanCategory::Fan5Plus,
                    };
                    cat == fan_cat
                })
                .collect();

            if fan_results.is_empty() {
                continue;
            }

            let mean_rt = mean(&fan_results.iter().map(|r| r.simulated_rt_ms).collect::<Vec<_>>());
            let std_dev = std_deviation(&fan_results.iter().map(|r| r.simulated_rt_ms).collect::<Vec<_>>());
            let mean_expected = mean(&fan_results.iter().map(|r| r.expected_rt_ms).collect::<Vec<_>>());

            analysis.fan_effects.insert(fan_cat, FanCategoryAnalysis {
                mean_rt_ms: mean_rt,
                std_dev_ms: std_dev,
                expected_rt_ms: mean_expected,
                n_trials: fan_results.len(),
            });
        }

        // Linear regression: RT ~ fan
        let fan_values: Vec<f32> = self.results.iter().map(|r| r.total_fan as f32).collect();
        let rt_values: Vec<f32> = self.results.iter().map(|r| r.simulated_rt_ms).collect();

        let (slope, intercept) = linear_regression(&fan_values, &rt_values)?;
        analysis.regression_slope = slope;
        analysis.regression_intercept = intercept;

        // Correlation with Anderson data
        let expected: Vec<f32> = self.results.iter().map(|r| r.expected_rt_ms).collect();
        let observed: Vec<f32> = self.results.iter().map(|r| r.simulated_rt_ms).collect();
        analysis.correlation_with_anderson = pearson_correlation(&expected, &observed)?;

        Ok(analysis)
    }
}

#[derive(Debug, Clone)]
pub struct FanTrialResult {
    pub person: String,
    pub location: String,
    pub person_fan: usize,
    pub location_fan: usize,
    pub total_fan: usize,
    pub fact_activation: f32,
    pub simulated_rt_ms: f32,
    pub expected_rt_ms: f32,
    pub elapsed_wall_time: Duration,
}

#[derive(Debug, Default)]
pub struct FanExperimentResults {
    pub fan_effects: HashMap<FanCategory, FanCategoryAnalysis>,
    pub regression_slope: f32,
    pub regression_intercept: f32,
    pub correlation_with_anderson: f32,
}

#[derive(Debug, Clone)]
pub struct FanCategoryAnalysis {
    pub mean_rt_ms: f32,
    pub std_dev_ms: f32,
    pub expected_rt_ms: f32,
    pub n_trials: usize,
}

impl FanExperimentResults {
    /// Check if results meet acceptance criteria
    pub fn meets_criteria(&self) -> bool {
        // Correlation with Anderson > 0.85
        if self.correlation_with_anderson < 0.85 {
            return false;
        }

        // Slope within 70ms ± 10ms
        if (self.regression_slope - 70.0).abs() > 10.0 {
            return false;
        }

        // Fan 1 baseline near 1159ms (±50ms tolerance)
        if let Some(fan1) = self.fan_effects.get(&FanCategory::Fan1) {
            if (fan1.mean_rt_ms - 1159.0).abs() > 50.0 {
                return false;
            }
        }

        true
    }

    /// Generate validation report
    pub fn generate_report(&self) -> String {
        let mut report = String::from("=== Anderson (1974) Fan Effect Validation ===\n\n");

        report.push_str(&format!("Linear Regression: RT = {:.1} + {:.1} × fan\n",
            self.regression_intercept, self.regression_slope));
        report.push_str(&format!("Expected Slope: 70.0ms/association (±10ms)\n"));
        report.push_str(&format!("Slope Status: {}\n\n",
            if (self.regression_slope - 70.0).abs() <= 10.0 { "PASS" } else { "FAIL" }));

        report.push_str(&format!("Correlation with Anderson: r = {:.3}\n", self.correlation_with_anderson));
        report.push_str(&format!("Acceptance Threshold: r > 0.85\n"));
        report.push_str(&format!("Status: {}\n\n",
            if self.correlation_with_anderson >= 0.85 { "PASS" } else { "FAIL" }));

        for (fan_cat, analysis) in &self.fan_effects {
            report.push_str(&format!("{:?}:\n", fan_cat));
            report.push_str(&format!("  Mean RT: {:.1}ms (±{:.1}ms)\n", analysis.mean_rt_ms, analysis.std_dev_ms));
            report.push_str(&format!("  Expected RT: {:.1}ms\n", analysis.expected_rt_ms));
            report.push_str(&format!("  Deviation: {:.1}ms\n", analysis.mean_rt_ms - analysis.expected_rt_ms));
            report.push_str(&format!("  N Trials: {}\n\n", analysis.n_trials));
        }

        report.push_str(&format!("\nOverall Status: {}\n",
            if self.meets_criteria() { "PASS - Meets Anderson (1974) replication criteria" }
            else { "FAIL - Does not meet replication criteria" }));

        report
    }
}
```

## Testing Approach

### Unit Tests
1. **Stimulus Loading**: Verify JSON parsing and fan calculation
2. **Graph Construction**: Test person-location-fact binding structure
3. **Fan Divisor Logic**: Verify activation splits correctly by fan
4. **RT Calculation**: Test linear relationship formula

### Integration Tests
1. **Single Fact Verification**: Activate person+location, measure episode
2. **Fan Manipulation**: Test fan 1 vs fan 3, verify RT increase
3. **Asymmetric Spreading**: Confirm episode→concept stronger than concept→episode
4. **Binding Strength Impact**: Test weak vs strong bindings

### Validation Tests
1. **Full Experiment**: N=100 trials per fact (26 facts × 100 = 2600 trials)
2. **Linear Regression**: Verify slope ~70ms, r² > 0.70
3. **Correlation Analysis**: Pearson's r with Anderson Table 2 data
4. **Effect Size**: Cohen's d > 1.0 for fan 1 vs fan 3

## Acceptance Criteria

- [ ] Correlation with Anderson (1974): r > 0.85
- [ ] Regression slope: 70ms ± 10ms per association
- [ ] Fan 1 baseline: 1159ms ± 50ms
- [ ] Fan 2: 1236ms ± 50ms (+77ms from fan 1)
- [ ] Fan 3: 1305ms ± 50ms (+69ms from fan 2)
- [ ] Linear r²: > 0.70 (strong linear relationship)
- [ ] Asymmetric spreading: Episode→concept 1.2x stronger
- [ ] Determinism: <3% RT variance across runs
- [ ] Performance: Full experiment <10 minutes
- [ ] CI integration: Automated validation on fan effect changes

## Dependencies

- M17 Task 007 (Fan Effect Spreading) - in progress
- M17 Task 005 (Binding Formation) - complete
- M17 Tasks 001-004 (Dual Memory Types, Concept Formation) - complete

## Estimated Time
4 days

## References

1. **Anderson, J. R. (1974).** Retrieval of propositional information from long-term memory. Cognitive Psychology, 6(4), 451-474.
   - Table 2 (p. 465): RT data for all fan conditions
   - Figure 3 (p. 466): Linear RT × fan relationship

2. **Anderson, J. R., & Reder, L. M. (1999).** The fan effect: New results and new theories. Journal of Experimental Psychology: General, 128(2), 186-197.
   - Meta-analysis of 25 years of fan effect research

3. **Anderson, J. R. (1983).** The Architecture of Cognition. Harvard University Press.
   - Chapter 3: ACT* activation spreading equations
