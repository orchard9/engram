# Task 020: Retrograde Amnesia Gradient Validation (Ribot's Law)

## Objective
Validate that M17's consolidation architecture produces the empirically-observed temporal gradient of retrograde amnesia (Ribot 1881, Squire & Alvarez 1995), where recent episodic memories show greater impairment than remote semantic memories, demonstrating proper hippocampal-neocortical transfer dynamics.

## Background

### Ribot's Law (1881) - Temporal Gradient of Amnesia

**Classical Finding:**
In cases of hippocampal damage or anterograde amnesia, memory loss follows a temporal gradient:
- **Recent memories** (hours to weeks): Severe impairment (70-90% loss)
- **Intermediate memories** (months to years): Moderate impairment (30-50% loss)
- **Remote memories** (years to decades): Minimal impairment (0-20% loss)

**Theoretical Explanation:**
- Recent memories depend on hippocampus for retrieval (episodic)
- Remote memories become hippocampus-independent (semantic)
- Consolidation creates gradient: older = more consolidated = less vulnerable

### Squire & Alvarez (1995) - Quantitative Gradients

**Key Empirical Data:**
- **1 day old**: 85% impairment after hippocampal lesion
- **1 week old**: 70% impairment
- **1 month old**: 50% impairment
- **1 year old**: 20% impairment
- **10 years old**: 5% impairment

**Mathematical Form:**
Impairment follows exponential decay with consolidation time:
```
Impairment(t) = I_max × exp(-t / τ)

Where:
- I_max = 0.85 (maximum impairment for recent memories)
- τ = 60 days (consolidation time constant)
- t = age of memory in days
```

### Nadel & Moscovitch (1997) - Multiple Trace Theory

**Alternative View:**
- Episodic details always hippocampus-dependent (no complete transfer)
- Semantic gist becomes hippocampus-independent
- Gradients reflect episodic→semantic transformation, not complete transfer

**Implications for Engram:**
- Episodes should remain vulnerable even after consolidation
- Concepts should become resistant to "hippocampal lesion" (binding disruption)
- Gradient reflects binding strength decay, not concept strength

## Requirements

1. **Lesion Simulation**: Model hippocampal damage as binding disruption or removal
2. **Temporal Gradient**: Create memories at t=-1d, -1w, -1m, -3m, -1y
3. **Retrieval Testing**: Measure pre-lesion vs post-lesion retrieval accuracy
4. **Gradient Fitting**: Exponential decay fit with r² > 0.80
5. **Episodic vs Semantic**: Test differential vulnerability
6. **Correlation Target**: r > 0.80 with Squire & Alvarez (1995) data

## Technical Specification

### Files to Create
- `engram-core/tests/cognitive_validation/retrograde_amnesia.rs` - Amnesia simulation
- `engram-core/tests/cognitive_validation/datasets/amnesia_gradients.json` - Expected impairment curves
- `engram-core/src/cognitive/validation/lesion_simulator.rs` - Hippocampal lesion model

### Files to Modify
- `engram-core/src/memory/bindings.rs` - Add lesion/disruption methods
- `engram-core/src/consolidation/concept_formation.rs` - Track consolidation age

### Retrograde Amnesia Simulation

```rust
use engram_core::{
    memory_graph::UnifiedMemoryGraph,
    consolidation::ConsolidationEngine,
};
use chrono::{DateTime, Utc, Duration as ChronoDuration};

/// Retrograde amnesia validator
pub struct RetrogradeAmnesiaValidator {
    /// Memory graph with dual memory architecture
    graph: Arc<UnifiedMemoryGraph<DashMapBackend>>,

    /// Consolidation engine for creating memory ages
    consolidation_engine: Arc<ConsolidationEngine>,

    /// Memories at different time points
    temporal_memories: HashMap<MemoryAge, Vec<TestMemory>>,

    /// Pre-lesion retrieval accuracy
    pre_lesion_accuracy: HashMap<MemoryAge, f32>,

    /// Post-lesion retrieval accuracy
    post_lesion_accuracy: HashMap<MemoryAge, f32>,
}

/// Memory age categories (matching Squire & Alvarez 1995)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum MemoryAge {
    /// 1 day old (very recent, highly vulnerable)
    Day1,

    /// 1 week old (recent, vulnerable)
    Week1,

    /// 1 month old (intermediate, moderately vulnerable)
    Month1,

    /// 3 months old (consolidating, less vulnerable)
    Month3,

    /// 1 year old (remote, minimally vulnerable)
    Year1,
}

impl MemoryAge {
    /// Get days since encoding
    pub fn days(&self) -> i64 {
        match self {
            Self::Day1 => 1,
            Self::Week1 => 7,
            Self::Month1 => 30,
            Self::Month3 => 90,
            Self::Year1 => 365,
        }
    }

    /// Get expected impairment (from Squire & Alvarez 1995)
    pub fn expected_impairment(&self) -> f32 {
        match self {
            Self::Day1 => 0.85,   // 85% impairment
            Self::Week1 => 0.70,  // 70% impairment
            Self::Month1 => 0.50, // 50% impairment
            Self::Month3 => 0.30, // 30% impairment
            Self::Year1 => 0.05,  // 5% impairment
        }
    }

    /// Standard error from literature
    pub fn standard_error(&self) -> f32 {
        match self {
            Self::Day1 => 0.08,
            Self::Week1 => 0.09,
            Self::Month1 => 0.10,
            Self::Month3 => 0.08,
            Self::Year1 => 0.03,
        }
    }
}

/// Test memory with known correct answer
#[derive(Debug, Clone)]
pub struct TestMemory {
    /// Episode node ID
    pub episode_id: NodeId,

    /// Content for retrieval cue
    pub content: String,

    /// Encoding timestamp (backdated)
    pub encoded_at: DateTime<Utc>,

    /// Concept IDs formed from this episode
    pub concept_ids: Vec<NodeId>,

    /// Expected answer (for recognition test)
    pub expected_answer: String,

    /// Pre-lesion retrieval success
    pub pre_lesion_retrieved: bool,

    /// Post-lesion retrieval success
    pub post_lesion_retrieved: bool,
}

impl RetrogradeAmnesiaValidator {
    /// Create validator
    pub fn new(
        graph: Arc<UnifiedMemoryGraph<DashMapBackend>>,
        consolidation_engine: Arc<ConsolidationEngine>,
    ) -> Self {
        Self {
            graph,
            consolidation_engine,
            temporal_memories: HashMap::new(),
            pre_lesion_accuracy: HashMap::new(),
            post_lesion_accuracy: HashMap::new(),
        }
    }

    /// Encode memories at different time points
    pub async fn encode_temporal_memories(&mut self, n_per_age: usize) -> Result<()> {
        for age in [MemoryAge::Day1, MemoryAge::Week1, MemoryAge::Month1, MemoryAge::Month3, MemoryAge::Year1] {
            let mut memories = Vec::new();

            for i in 0..n_per_age {
                let content = format!("Memory {} at age {:?}", i, age);
                let encoded_at = Utc::now() - ChronoDuration::days(age.days());

                let episode_id = self.graph.add_node(
                    MemoryNode::new_with_type(
                        Self::generate_embedding(&content)?,
                        0.9,
                        content.clone(),
                        MemoryNodeType::Episode {
                            timestamp: encoded_at, // Backdate timestamp
                            context: "Retrograde_Experiment".into(),
                        },
                    )
                ).await?;

                let memory = TestMemory {
                    episode_id,
                    content: content.clone(),
                    encoded_at,
                    concept_ids: Vec::new(),
                    expected_answer: format!("Answer_{}", i),
                    pre_lesion_retrieved: false,
                    post_lesion_retrieved: false,
                };

                memories.push(memory);
            }

            self.temporal_memories.insert(age, memories);
        }

        Ok(())
    }

    /// Run consolidation cycles to form concepts (respecting memory ages)
    pub async fn consolidate_memories(&mut self) -> Result<()> {
        // For each memory age, run appropriate number of consolidation cycles
        for (age, memories) in &mut self.temporal_memories {
            // Estimate consolidation cycles based on age
            let consolidation_cycles = match age {
                MemoryAge::Day1 => 1,    // 1 night
                MemoryAge::Week1 => 7,   // 7 nights
                MemoryAge::Month1 => 30, // 30 nights
                MemoryAge::Month3 => 90, // 90 nights
                MemoryAge::Year1 => 365, // 365 nights (simulate accelerated)
            };

            // Run consolidation for this memory set
            for _cycle in 0..consolidation_cycles.min(100) { // Cap at 100 for performance
                self.consolidation_engine.consolidate_with_concepts(SleepStage::NREM2).await?;

                // Track concepts formed from these episodes
                for memory in memories.iter_mut() {
                    let concepts = self.graph.get_episode_concepts(&memory.episode_id).await?;
                    for concept_ref in concepts {
                        if !memory.concept_ids.contains(&concept_ref.target_id) {
                            memory.concept_ids.push(concept_ref.target_id.clone());
                        }
                    }
                }
            }
        }

        Ok(())
    }

    /// Test retrieval accuracy (pre-lesion baseline)
    pub async fn test_pre_lesion_retrieval(&mut self) -> Result<()> {
        for (age, memories) in &mut self.temporal_memories {
            let mut correct = 0;
            let total = memories.len();

            for memory in memories.iter_mut() {
                // Attempt retrieval with cue
                let retrieved = self.retrieve_memory(&memory.content).await?;

                if retrieved.contains(&memory.expected_answer) {
                    memory.pre_lesion_retrieved = true;
                    correct += 1;
                }
            }

            let accuracy = correct as f32 / total as f32;
            self.pre_lesion_accuracy.insert(*age, accuracy);
        }

        Ok(())
    }

    /// Simulate hippocampal lesion (disrupt episode-concept bindings)
    pub async fn simulate_hippocampal_lesion(&self) -> Result<()> {
        // Model 1: Complete binding disruption (severe lesion)
        // This simulates removing hippocampal indexing function

        for memories in self.temporal_memories.values() {
            for memory in memories {
                // Remove or severely weaken bindings
                // Concepts remain intact, but episode access is disrupted

                for concept_id in &memory.concept_ids {
                    // Reduce binding strength to simulate lesion
                    self.graph.update_binding_strength(
                        &memory.episode_id,
                        concept_id,
                        |_current| 0.05, // Severe reduction
                    ).await?;
                }
            }
        }

        Ok(())
    }

    /// Test retrieval accuracy (post-lesion)
    pub async fn test_post_lesion_retrieval(&mut self) -> Result<()> {
        for (age, memories) in &mut self.temporal_memories {
            let mut correct = 0;
            let total = memories.len();

            for memory in memories.iter_mut() {
                // Attempt retrieval with disrupted bindings
                let retrieved = self.retrieve_memory(&memory.content).await?;

                if retrieved.contains(&memory.expected_answer) {
                    memory.post_lesion_retrieved = true;
                    correct += 1;
                }
            }

            let accuracy = correct as f32 / total as f32;
            self.post_lesion_accuracy.insert(*age, accuracy);
        }

        Ok(())
    }

    /// Retrieve memory given cue
    async fn retrieve_memory(&self, cue: &str) -> Result<Vec<String>> {
        // Use spreading activation to retrieve
        let cue_embedding = Self::generate_embedding(cue)?;

        // Find similar nodes
        let results = self.graph.vector_search(
            &cue_embedding,
            10, // Top 10
            0.5, // Similarity threshold
        ).await?;

        // Extract answers from retrieved nodes
        let answers: Vec<String> = results.iter()
            .filter_map(|node_id| {
                // Extract expected answer from node metadata
                Some(format!("Answer_placeholder"))
            })
            .collect();

        Ok(answers)
    }

    /// Run full retrograde amnesia experiment
    pub async fn run_full_experiment(&mut self, n_per_age: usize) -> Result<AmnesiaGradientResults> {
        // 1. Encode memories at different ages
        self.encode_temporal_memories(n_per_age).await?;

        // 2. Run consolidation to form concepts
        self.consolidate_memories().await?;

        // 3. Test pre-lesion retrieval (baseline)
        self.test_pre_lesion_retrieval().await?;

        // 4. Simulate hippocampal lesion
        self.simulate_hippocampal_lesion().await?;

        // 5. Test post-lesion retrieval
        self.test_post_lesion_retrieval().await?;

        // 6. Analyze gradient
        self.analyze_gradient()
    }

    /// Analyze retrograde amnesia gradient
    fn analyze_gradient(&self) -> Result<AmnesiaGradientResults> {
        let mut results = AmnesiaGradientResults::default();

        for age in [MemoryAge::Day1, MemoryAge::Week1, MemoryAge::Month1, MemoryAge::Month3, MemoryAge::Year1] {
            let pre_acc = self.pre_lesion_accuracy.get(&age).copied().unwrap_or(0.0);
            let post_acc = self.post_lesion_accuracy.get(&age).copied().unwrap_or(0.0);

            // Impairment = (pre - post) / pre
            let impairment = if pre_acc > 0.0 {
                (pre_acc - post_acc) / pre_acc
            } else {
                0.0
            };

            results.impairments.insert(age, AmnesiaDataPoint {
                memory_age: age,
                pre_lesion_accuracy: pre_acc,
                post_lesion_accuracy: post_acc,
                impairment,
                expected_impairment: age.expected_impairment(),
            });
        }

        // Exponential fit: Impairment(t) = I_max × exp(-t / τ)
        let days: Vec<f32> = results.impairments.keys().map(|age| age.days() as f32).collect();
        let impairments: Vec<f32> = results.impairments.values().map(|dp| dp.impairment).collect();

        let (tau, i_max) = Self::fit_exponential_decay(&days, &impairments)?;
        results.decay_constant_days = tau;
        results.max_impairment = i_max;

        // Correlation with Squire & Alvarez data
        let expected: Vec<f32> = results.impairments.values().map(|dp| dp.expected_impairment).collect();
        let observed: Vec<f32> = results.impairments.values().map(|dp| dp.impairment).collect();
        results.correlation_with_squire = pearson_correlation(&expected, &observed)?;

        Ok(results)
    }

    /// Fit exponential decay: y = I_max × exp(-x / τ)
    fn fit_exponential_decay(x: &[f32], y: &[f32]) -> Result<(f32, f32)> {
        // Linearize: ln(y) = ln(I_max) - x/τ
        // Fit linear: ln(y) = a - b×x, where a = ln(I_max), b = 1/τ

        let ln_y: Vec<f32> = y.iter().map(|&yi| yi.max(0.01).ln()).collect();

        let (slope, intercept) = linear_regression(x, &ln_y)?;

        let tau = -1.0 / slope;  // τ = -1/b
        let i_max = intercept.exp();  // I_max = exp(a)

        Ok((tau, i_max))
    }
}

#[derive(Debug, Clone)]
pub struct AmnesiaDataPoint {
    pub memory_age: MemoryAge,
    pub pre_lesion_accuracy: f32,
    pub post_lesion_accuracy: f32,
    pub impairment: f32,
    pub expected_impairment: f32,
}

#[derive(Debug, Default)]
pub struct AmnesiaGradientResults {
    pub impairments: HashMap<MemoryAge, AmnesiaDataPoint>,
    pub decay_constant_days: f32,
    pub max_impairment: f32,
    pub correlation_with_squire: f32,
}

impl AmnesiaGradientResults {
    /// Check if results meet acceptance criteria
    pub fn meets_criteria(&self) -> bool {
        // Correlation with Squire & Alvarez > 0.80
        if self.correlation_with_squire < 0.80 {
            return false;
        }

        // Decay constant 40-80 days (expected ~60 days)
        if self.decay_constant_days < 40.0 || self.decay_constant_days > 80.0 {
            return false;
        }

        // Max impairment 0.75-0.95 (expected ~0.85)
        if self.max_impairment < 0.75 || self.max_impairment > 0.95 {
            return false;
        }

        // Gradient: Recent > Remote
        let day1_imp = self.impairments.get(&MemoryAge::Day1).map(|d| d.impairment).unwrap_or(0.0);
        let year1_imp = self.impairments.get(&MemoryAge::Year1).map(|d| d.impairment).unwrap_or(1.0);

        if day1_imp <= year1_imp {
            return false; // No gradient
        }

        true
    }

    /// Generate validation report
    pub fn generate_report(&self) -> String {
        let mut report = String::from("=== Retrograde Amnesia Gradient Validation (Ribot's Law) ===\n\n");

        report.push_str(&format!("Exponential Decay Fit:\n"));
        report.push_str(&format!("  τ (time constant): {:.1} days\n", self.decay_constant_days));
        report.push_str(&format!("  I_max (max impairment): {:.3}\n", self.max_impairment));
        report.push_str(&format!("  Expected τ: 60 days (±20 days tolerance)\n\n"));

        report.push_str(&format!("Correlation with Squire & Alvarez (1995): r = {:.3}\n", self.correlation_with_squire));
        report.push_str(&format!("Acceptance Threshold: r > 0.80\n\n"));

        for age in [MemoryAge::Day1, MemoryAge::Week1, MemoryAge::Month1, MemoryAge::Month3, MemoryAge::Year1] {
            if let Some(dp) = self.impairments.get(&age) {
                report.push_str(&format!("{:?} ({}d):\n", age, age.days()));
                report.push_str(&format!("  Pre-lesion Accuracy: {:.3}\n", dp.pre_lesion_accuracy));
                report.push_str(&format!("  Post-lesion Accuracy: {:.3}\n", dp.post_lesion_accuracy));
                report.push_str(&format!("  Impairment: {:.3} (expected {:.3})\n", dp.impairment, dp.expected_impairment));
                report.push_str(&format!("  Deviation: {:.3}\n\n", (dp.impairment - dp.expected_impairment).abs()));
            }
        }

        report.push_str(&format!("\nOverall Status: {}\n",
            if self.meets_criteria() { "PASS - Demonstrates Ribot's Law temporal gradient" }
            else { "FAIL - Gradient does not match expected pattern" }));

        report
    }
}
```

## Testing Approach

### Unit Tests
1. **Memory Dating**: Test backdated timestamps and age calculation
2. **Lesion Simulation**: Verify binding strength reduction
3. **Exponential Fitting**: Test decay parameter estimation
4. **Retrieval Cuing**: Test spreading activation retrieval

### Integration Tests
1. **Single Age**: Test one memory age condition
2. **Gradient Formation**: Verify recent > remote impairment
3. **Concept Resistance**: Confirm concepts survive lesion better than episodes
4. **Multiple Trace**: Test Nadel & Moscovitch alternative model

### Validation Tests
1. **Full Experiment**: N=50 memories per age (5 ages = 250 total)
2. **Gradient Fitting**: Exponential decay with r² > 0.80
3. **Correlation Analysis**: Pearson's r with Squire data
4. **Effect Size**: Large effect (Cohen's d > 1.0) for Day1 vs Year1

## Acceptance Criteria

- [ ] Correlation with Squire & Alvarez (1995): r > 0.80
- [ ] Exponential decay time constant: 40-80 days (expected ~60 days)
- [ ] Maximum impairment: 0.75-0.95 (expected ~0.85 for 1-day memories)
- [ ] Day 1 impairment: 75-95% (recent memory vulnerable)
- [ ] Year 1 impairment: 0-15% (remote memory resistant)
- [ ] Gradient monotonicity: Recent > Week > Month > Year
- [ ] Concept resistance: Concepts show <30% impairment even at Day 1
- [ ] Episode vulnerability: Episodes show >70% impairment at Day 1
- [ ] Exponential fit r²: > 0.80
- [ ] Determinism: <5% variance across runs
- [ ] Performance: Full experiment <15 minutes
- [ ] CI integration: Automated validation on consolidation changes

## Dependencies

- M17 Task 006 (Consolidation Integration) - complete
- M17 Task 005 (Binding Formation) - complete
- M17 Task 004 (Concept Formation) - complete
- Task 019 (Consolidation Timeline Validation) - for temporal simulation infrastructure

## Estimated Time
4 days

## References

1. **Ribot, T. (1881).** Les maladies de la mémoire. Paris: Germer Baillière.
   - Original description of temporal gradient in amnesia

2. **Squire, L. R., & Alvarez, P. (1995).** Retrograde amnesia and memory consolidation: A neurobiological perspective. Current Opinion in Neurobiology, 5(2), 169-177.
   - Quantitative gradient data

3. **Frankland, P. W., & Bontempi, B. (2005).** The organization of recent and remote memories. Nature Reviews Neuroscience, 6(2), 119-130.
   - Rodent model gradients and consolidation timescales

4. **Nadel, L., & Moscovitch, M. (1997).** Memory consolidation, retrograde amnesia and the hippocampal complex. Current Opinion in Neurobiology, 7(2), 217-227.
   - Multiple Trace Theory alternative explanation

5. **Winocur, G., & Moscovitch, M. (2011).** Memory transformation and systems consolidation. Journal of the International Neuropsychological Society, 17(5), 766-780.
   - Episodic-to-semantic transformation accounts

6. **Bontempi, B., et al. (1999).** Time-dependent reorganization of brain circuitry underlying long-term memory storage. Nature, 400(6745), 671-675.
   - C-fos imaging of consolidation gradients in rodents
