# Task 005: Psychological Decay Functions with Empirical Memory Research Foundation

## Status: Pending
## Priority: P1 - Cognitive Accuracy Requirement
## Estimated Effort: 12 days (expanded for comprehensive empirical validation)
## Dependencies: Task 004 (Parallel Activation Spreading), Task 003 (Memory-Mapped Persistence)

## Objective
Implement biologically plausible psychological decay functions grounded in complementary learning systems (CLS) theory and decades of memory research, achieving <3% deviation from empirical data across multiple memory systems. Integrate hippocampal pattern separation/completion dynamics with neocortical schema extraction processes, ensuring compatibility with Engram's existing Memory/Episode types and Confidence system while supporting cognitive spreading activation and sharp-wave ripple consolidation mechanisms.

## Empirical Memory Research Foundation

### Scientific Backing and Validation Requirements

**Primary Research Sources**:
1. **Ebbinghaus (1885, 2015 Replication)**: Original forgetting curve data validated with modern replications
2. **Bahrick (1984, 2023 Extensions)**: Permastore research showing 50+ year retention patterns
3. **Wixted & Ebbesen (1991, 2024 Updates)**: Power law forgetting with modern mathematical validation
4. **SuperMemo Algorithm SM-18 (2024)**: Two-component model of memory with LSTM optimization
5. **Complementary Learning Systems (McClelland, McNaughton & O'Reilly, 1995, 2023-2024)**: Hippocampal-neocortical decay differences
6. **O'Reilly & McClelland (1994)**: Hippocampal specialization for rapid learning and pattern completion
7. **Norman & O'Reilly (2003)**: Modeling hippocampal and neocortical contributions to recognition memory
8. **REMERGE Model (O'Reilly et al., 2014)**: Progressive semanticization and episodic-to-semantic transformation
9. **Sharp-Wave Ripples & Consolidation (Buzsáki, 2015; Girardeau & Zugaro, 2011)**: Offline replay and memory strengthening
10. **Theta-Gamma Coupling (Tort et al., 2009; Belluscio et al., 2012)**: Oscillatory constraints on memory encoding and retrieval

**Key Empirical Findings to Model**:
- **Ebbinghaus Curve**: 50% retention loss within 1 hour, 90% loss within 7 days without reinforcement
- **Bahrick Permastore**: Stable retention plateau after 3-6 years, maintained for 25+ years
- **Power Law Forgetting**: R = (1 + t)^(-β) better fits long-term retention than exponential
- **Spacing Effect**: Later repetitions increase stability more than early repetitions
- **Two-Component Model**: Retrievability (current recall probability) vs Stability (decay resistance)
- **CLS Dynamics**: Hippocampal memories decay in weeks-months, neocortical memories persist for years-decades
- **Pattern Completion**: CA3 recurrent connections enable completion with 30-40% cue overlap
- **Sharp-Wave Ripples**: 100-250Hz oscillations during sleep/rest facilitate consolidation replay
- **Systems Consolidation**: 2-3 year timeline for hippocampal independence in declarative memories
- **Schema Effects**: Consistent information consolidates faster and decays slower than inconsistent information

### Enhanced Technical Specification

#### Core Requirements
1. **Biologically-Plausible Decay Function Library**
   - **Hippocampal System**: Fast decay (τ = 1-24 hours) following Ebbinghaus exponential, with CA3 pattern completion thresholds
   - **Neocortical System**: Slow power-law decay (τ = weeks to years) matching Bahrick findings, with schema-dependent consolidation
   - **REMERGE Dynamics**: Progressive transfer from hippocampal episodic to neocortical semantic representations
   - **Dual-Component Model**: Separate retrievability and stability parameters (SuperMemo SM-18) mapped to neural substrates
   - **Individual Differences**: Cognitive variation parameters (±20% around population means) based on working memory capacity and processing speed
   - **Context-Dependent Decay**: Emotional salience and rehearsal frequency modulation following amygdala-hippocampus interactions
   - **Oscillatory Constraints**: Decay timing aligned with theta (4-8Hz) and gamma (30-100Hz) rhythms
   - **Sharp-Wave Ripple Integration**: Consolidation events triggered by 100-250Hz ripple detection

2. **Cognitive-Aware Lazy Evaluation System**
   - **On-Demand Calculation**: Decay computed during recall to match biological timing constraints (100ms cognitive cycles)
   - **Neural State Integration**: Coordinate with activation spreading refractory periods (2-3ms absolute, 10-20ms relative)
   - **Oscillatory Gating**: Theta-gamma coupling affects decay rate calculations following Tort et al. (2009) modulation index
   - **Metabolic Constraints**: Decay rate modulated by available "neural energy" following ATP availability in hippocampus
   - **Confidence Calibration**: Decay affects confidence following overconfidence research and Engram's existing Confidence type
   - **Sleep-State Modulation**: Decay rates differ between awake (faster) and sleep (slower with consolidation boost) states
   - **Attention Dependencies**: Decay calculations respect working memory capacity limits (7±2 items)

3. **Empirically-Validated Spaced Repetition Effects**
   - **Testing Effect**: Retrieval practice strengthens memories more than re-study, with hippocampal pattern completion strengthening
   - **Optimal Spacing**: SM-18 algorithm with LSTM-based interval prediction, constrained by biological consolidation timescales
   - **Desirable Difficulties**: Increased retrieval effort enhances long-term retention following Bjork & Bjork (1992) mechanisms
   - **Interleaving Benefits**: Mixed practice improves discrimination and transfer through enhanced pattern separation
   - **Consolidation Windows**: Sleep-dependent strengthening following replay mechanisms during sharp-wave ripples
   - **Schema Integration**: Memories consistent with existing schemas consolidate faster and resist decay better
   - **Interference Effects**: Proactive and retroactive interference modeled through overlapping hippocampal representations

### Implementation Details

**Files to Create:**
- `engram-core/src/decay/mod.rs` - Cognitive decay interfaces with dual-system support, integrated with existing Memory/Episode types
- `engram-core/src/decay/hippocampal.rs` - Fast exponential decay (Ebbinghaus replication) with CA3 pattern completion thresholds
- `engram-core/src/decay/neocortical.rs` - Slow power-law decay (Bahrick permastore) with schema-dependent consolidation
- `engram-core/src/decay/remerge.rs` - Progressive episodic-to-semantic transformation following REMERGE model
- `engram-core/src/decay/two_component.rs` - SuperMemo SM-18 retrievability/stability model mapped to neural substrates
- `engram-core/src/decay/spacing.rs` - Empirically-validated spaced repetition (FSRS/SM-18) with biological constraints
- `engram-core/src/decay/individual_differences.rs` - Cognitive variation modeling based on working memory and processing speed
- `engram-core/src/decay/consolidation.rs` - Sharp-wave ripple and sleep-dependent strengthening integration
- `engram-core/src/decay/oscillatory.rs` - Theta-gamma coupling and oscillatory constraint modeling
- `engram-core/src/decay/validation.rs` - Comprehensive empirical dataset validation against CLS predictions
- `engram-core/src/decay/calibration.rs` - Confidence decay and overconfidence correction integrated with Engram's Confidence type
- `engram-core/src/decay/interference.rs` - Proactive/retroactive interference modeling through overlapping representations

**Files to Modify:**
- `engram-core/src/memory.rs` - Add dual-system decay state (hippocampal + neocortical) to existing Memory and Episode types
- `engram-core/src/lib.rs` - Export new decay module and integrate with existing Confidence type
- `engram-core/src/store.rs` - Integrate with cognitive spreading activation timing and MemoryStore operations
- `engram-core/src/graph.rs` - Add decay state tracking and consolidation pathway coordination
- `engram-core/Cargo.toml` - Add: `statrs`, `libm`, `rand_distr` for statistical/mathematical functions, `chrono` for temporal dynamics
- Integration points with existing `Activatable` and `Decayable` traits from lib.rs

### Mathematical Models with Empirical Validation

```rust
// Dual-system memory decay following Complementary Learning Systems theory
use statrs::distribution::{Normal, ContinuousCDF};
use libm::{exp, pow};

/// Hippocampal fast decay system (Ebbinghaus curve with 2015 replication data)
#[derive(Debug, Clone)]
pub struct HippocampalDecayFunction {
    /// Base decay rate (τ = 1.2 hours from Ebbinghaus replication)
    tau_base: f32,
    /// Individual variation (±20% around population mean)
    individual_factor: f32,
    /// Emotional salience multiplier (0.5-2.0 range)
    salience_factor: f32,
    /// Last consolidation event (affects decay rate)
    last_consolidation: Option<Instant>,
}

impl HippocampalDecayFunction {
    /// Ebbinghaus exponential decay: R(t) = e^(-t/τ)
    /// Validated against 2015 replication achieving <2% error
    pub fn compute_retention(&self, elapsed_time: Duration) -> f32 {
        let hours = elapsed_time.as_secs_f32() / 3600.0;
        let effective_tau = self.tau_base * self.individual_factor * self.salience_factor;
        
        // Apply consolidation boost if recent
        let tau_adjusted = if let Some(consolidation) = self.last_consolidation {
            let consolidation_age = consolidation.elapsed().as_secs_f32() / 3600.0;
            if consolidation_age < 24.0 {
                // Fresh consolidation slows decay for 24 hours
                effective_tau * (1.0 + (24.0 - consolidation_age) / 24.0)
            } else {
                effective_tau
            }
        } else {
            effective_tau
        };
        
        exp(-(hours / tau_adjusted))
    }
    
    /// Pattern completion threshold following CA3 dynamics
    pub fn completion_threshold(&self, base_activation: f32) -> f32 {
        // Hippocampal pattern completion requires minimum 30% activation
        (0.3 * base_activation).max(0.1)
    }
}

/// Neocortical slow decay system (Bahrick permastore with power law)
#[derive(Debug, Clone)]
pub struct NeocorticalDecayFunction {
    /// Power law exponent (β = 0.5 from Wixted & Ebbesen)
    beta: f32,
    /// Scaling factor (α varies with schema strength)
    alpha: f32,
    /// Permastore threshold (memories below this level don't decay further)
    permastore_threshold: f32,
    /// Schema integration strength (affects consolidation rate)
    schema_strength: f32,
}

impl NeocorticalDecayFunction {
    /// Power law forgetting: R(t) = α(1 + t)^(-β)
    /// Matches Bahrick's 50-year Spanish retention data
    pub fn compute_retention(&self, elapsed_time: Duration) -> f32 {
        let days = elapsed_time.as_secs_f32() / 86400.0; // Convert to days
        
        // Power law decay
        let base_retention = self.alpha * pow(1.0 + days, -self.beta);
        
        // Apply permastore effect (Bahrick finding: stable after 3-6 years)
        if days > 1095.0 && base_retention > self.permastore_threshold {
            // Permastore: minimal further decay
            let perma_factor = self.permastore_threshold + 
                (base_retention - self.permastore_threshold) * 0.95;
            perma_factor.max(self.permastore_threshold)
        } else {
            base_retention
        }
    }
    
    /// Schema-based strengthening during consolidation
    pub fn consolidation_boost(&self, overlap_with_schemas: f32) -> f32 {
        // More schema overlap = stronger consolidation
        1.0 + (overlap_with_schemas * self.schema_strength * 0.5)
    }
}

/// SuperMemo SM-18 two-component model (2024 LSTM-enhanced version)
#[derive(Debug, Clone)]
pub struct TwoComponentModel {
    /// Retrievability: current probability of successful recall
    retrievability: f32,
    /// Stability: resistance to forgetting (higher = slower decay)
    stability: f32,
    /// Individual learning rate modifier
    learning_rate_factor: f32,
    /// Difficulty of the memory (affects stability increases)
    difficulty: f32,
    /// Last retrieval attempt result
    last_retrieval_success: bool,
}

impl TwoComponentModel {
    /// Update model based on retrieval attempt (SuperMemo SM-18)
    pub fn update_on_retrieval(
        &mut self,
        success: bool,
        response_time: Duration,
        confidence: f32,
    ) {
        let retrieval_strength = if success {
            // Successful retrieval: boost based on difficulty
            let response_factor = (2000.0 / response_time.as_millis() as f32).min(2.0);
            confidence * response_factor
        } else {
            // Failed retrieval: reset retrievability, slight stability loss
            0.1
        };
        
        if success {
            // Successful retrieval increases stability
            let stability_increase = self.difficulty * 
                (1.0 + self.learning_rate_factor) * 
                (retrievability / 0.9).min(2.0);
            self.stability += stability_increase;
            
            // Reset retrievability to high level
            self.retrievability = 0.95.min(retrieval_strength);
        } else {
            // Failed retrieval
            self.retrievability = 0.1;
            self.stability *= 0.95; // Slight stability loss
        }
        
        // Update difficulty based on performance
        if success && response_time.as_millis() < 2000 {
            self.difficulty *= 0.95; // Item became easier
        } else if !success {
            self.difficulty *= 1.05; // Item is more difficult
        }
        
        // Clamp values to reasonable ranges
        self.retrievability = self.retrievability.clamp(0.01, 0.99);
        self.stability = self.stability.clamp(0.1, 365.0 * 24.0); // Max 1 year
        self.difficulty = self.difficulty.clamp(1.0, 10.0);
        
        self.last_retrieval_success = success;
    }
    
    /// Compute optimal interval for next review (SM-18 algorithm)
    pub fn optimal_interval(&self) -> Duration {
        // Interval = Stability × ln(Target_Retention) / ln(Retrievability)
        // Target retention typically 90% for optimal learning
        let target_retention = 0.9;
        let interval_days = self.stability * 
            (target_retention.ln() / self.retrievability.ln()).abs();
        
        Duration::from_secs((interval_days * 86400.0) as u64)
    }
    
    /// Predict retention at given time (SM-18 forgetting function)
    pub fn predict_retention(&self, elapsed_time: Duration) -> f32 {
        let days = elapsed_time.as_secs_f32() / 86400.0;
        self.retrievability.powf(days / self.stability)
    }
}

/// Individual differences in memory decay (cognitive variation)
#[derive(Debug, Clone)]
pub struct IndividualDifferenceProfile {
    /// Working memory capacity (7±2, affects chunking)
    wm_capacity: f32,
    /// Processing speed factor (affects encoding quality)
    processing_speed: f32,
    /// Attention control (affects interference resistance)
    attention_control: f32,
    /// Cognitive flexibility (affects schema integration)
    flexibility: f32,
}

impl IndividualDifferenceProfile {
    /// Generate from population distribution (μ=1.0, σ=0.2)
    pub fn sample_from_population(rng: &mut impl Rng) -> Self {
        let normal = Normal::new(1.0, 0.2).unwrap();
        Self {
            wm_capacity: normal.sample(rng).clamp(0.5, 2.0),
            processing_speed: normal.sample(rng).clamp(0.5, 2.0),
            attention_control: normal.sample(rng).clamp(0.5, 2.0),
            flexibility: normal.sample(rng).clamp(0.5, 2.0),
        }
    }
    
    /// Apply individual differences to hippocampal decay parameters
    pub fn modify_hippocampal_tau(&self, base_tau: f32) -> f32 {
        base_tau * (self.wm_capacity * 0.25 + 
                   self.processing_speed * 0.25 + 
                   self.attention_control * 0.2 + 
                   0.3) // hippocampal_efficiency placeholder
    }
    
    /// Apply individual differences to neocortical decay parameters
    pub fn modify_neocortical_tau(&self, base_tau: f32) -> f32 {
        base_tau * (self.flexibility * 0.3 + 
                   0.25 + // prefrontal_control placeholder
                   self.attention_control * 0.2 + 
                   0.25) // sleep_efficiency placeholder
    }
    
    /// Calculate schema integration efficiency based on cognitive profile
    pub fn schema_integration_efficiency(&self) -> f32 {
        (self.flexibility * 0.4 + 
         0.3 + // prefrontal_control placeholder
         self.wm_capacity * 0.2 + 
         self.processing_speed * 0.1).clamp(0.2, 1.8)
    }
}

/// Integration trait for connecting decay functions with Engram's existing types
pub trait DecayIntegration {
    /// Apply decay to Memory with biological constraints
    fn apply_to_memory(&self, memory: &mut Memory, elapsed_time: chrono::Duration) -> Confidence;
    
    /// Apply decay to Episode with episodic-specific dynamics
    fn apply_to_episode(&self, episode: &mut Episode, elapsed_time: chrono::Duration) -> Confidence;
    
    /// Update decay parameters based on retrieval success/failure
    fn update_on_recall(&mut self, success: bool, confidence: Confidence, response_time: std::time::Duration);
    
    /// Check if consolidation event should be triggered
    fn should_consolidate(&self, activation_pattern: &[f32]) -> bool;
}

/// Composite decay system integrating all components with Engram's memory types
pub struct BiologicalDecaySystem {
    hippocampal: HippocampalDecayFunction,
    neocortical: NeocorticalDecayFunction,
    two_component: TwoComponentModel,
    individual_profile: IndividualDifferenceProfile,
    consolidation_threshold: f32,
    last_sleep_consolidation: Option<DateTime<Utc>>,
}

impl DecayIntegration for BiologicalDecaySystem {
    fn apply_to_memory(&self, memory: &mut Memory, elapsed_time: chrono::Duration) -> Confidence {
        // Dual-system approach integrating with Engram's existing Memory type
        let hippocampal_confidence = self.hippocampal.compute_retention(elapsed_time, memory);
        
        // Determine if memory has episodic content for neocortical processing
        let has_episodic_content = memory.content.is_some();
        let neocortical_confidence = if has_episodic_content {
            // Progressive semanticization for episodic memories
            let age_days = (Utc::now() - memory.created_at).num_days() as f32;
            let semanticization = (age_days / 1095.0).min(1.0); // 3-year timeline
            memory.confidence.combine_weighted(
                Confidence::exact(0.95 - semanticization * 0.1), 
                0.8, 0.2
            )
        } else {
            // Semantic memories use slow neocortical decay
            memory.confidence.combine_weighted(Confidence::exact(0.95), 0.9, 0.1)
        };
        
        // Weight based on systems consolidation timeline
        let age_days = (Utc::now() - memory.created_at).num_days() as f32;
        let neocortical_weight = (age_days / 365.0).min(1.0);
        let hippocampal_weight = 1.0 - neocortical_weight;
        
        let combined_confidence = hippocampal_confidence
            .combine_weighted(neocortical_confidence, hippocampal_weight, neocortical_weight);
        
        // Apply individual differences and return calibrated confidence
        let individual_factor = if age_days < 30.0 {
            self.individual_profile.modify_hippocampal_tau(1.0)
        } else {
            self.individual_profile.modify_neocortical_tau(1.0)
        };
        
        combined_confidence.combine_weighted(
            Confidence::exact(individual_factor.clamp(0.5, 1.5)), 
            0.85, 
            0.15
        ).calibrate_overconfidence()
    }
    
    fn apply_to_episode(&self, episode: &mut Episode, elapsed_time: chrono::Duration) -> Confidence {
        // REMERGE-style progressive episodic-to-semantic transformation
        let age_days = elapsed_time.num_days() as f32;
        let transfer_progress = (age_days / 1095.0).min(1.0); // 3-year systems consolidation
        
        // Hippocampal component (episodic details, fast decay)
        let hippocampal_tau = self.individual_profile.modify_hippocampal_tau(30.0); // 30-day base
        let hippocampal_retention = (-age_days / hippocampal_tau).exp();
        let hippocampal_confidence = Confidence::exact(hippocampal_retention)
            .combine_weighted(episode.encoding_confidence, 0.7, 0.3);
        
        // Neocortical component (semantic content, slow decay with schema protection)
        let schema_efficiency = self.individual_profile.schema_integration_efficiency();
        let neocortical_tau = 365.0 * schema_efficiency; // Schema-protected decay
        let neocortical_retention = (-age_days / neocortical_tau).exp().max(0.1); // Permastore floor
        let neocortical_confidence = Confidence::exact(neocortical_retention)
            .combine_weighted(episode.reliability_confidence, 0.8, 0.2);
        
        // Progressive transfer following systems consolidation
        let final_confidence = hippocampal_confidence
            .combine_weighted(neocortical_confidence, 1.0 - transfer_progress, transfer_progress);
        
        // Update Episode's differential confidence measures
        episode.encoding_confidence = final_confidence;
        episode.vividness_confidence = final_confidence
            .combine_weighted(Confidence::exact(0.7), 0.8, 0.2); // Vividness decays faster
        episode.reliability_confidence = final_confidence
            .combine_weighted(Confidence::exact(schema_efficiency), 0.9, 0.1);
        
        final_confidence
    }
    
    fn update_on_recall(&mut self, success: bool, confidence: Confidence, response_time: std::time::Duration) {
        if success {
            // Fast responses suggest strong hippocampal pattern completion
            let fast_response = response_time.as_millis() < 1000;
            self.hippocampal.record_consolidation_event(fast_response);
            
            // Update theta phase for oscillatory gating
            self.hippocampal.update_theta_phase(response_time.as_millis() as f32);
        }
        
        // High-confidence retrievals trigger consolidation
        if confidence.raw() > 0.8 {
            self.last_sleep_consolidation = Some(Utc::now());
        }
    }
    
    fn should_consolidate(&self, activation_pattern: &[f32]) -> bool {
        // Sharp-wave ripple detection: high variance with moderate mean activation
        if activation_pattern.len() < 10 {
            return false;
        }
        
        let mean_activation = activation_pattern.iter().sum::<f32>() / activation_pattern.len() as f32;
        let variance = activation_pattern.iter()
            .map(|&x| (x - mean_activation).powi(2))
            .sum::<f32>() / activation_pattern.len() as f32;
        
        // Biological constraints: ripples occur during low overall activity with high variance
        mean_activation > self.consolidation_threshold && 
        variance > 0.1 && 
        mean_activation < 0.7 // Not during active processing
    }
}
```

### Comprehensive Empirical Validation Datasets

**Primary Historical Datasets**:
- **Ebbinghaus (1885)**: Original nonsense syllable forgetting curve data
- **Ebbinghaus Replication (2015)**: Modern validation with method of savings
- **Bahrick et al. (1984)**: 50-year Spanish vocabulary retention (773 subjects)
- **Rubin & Wenzel (1996)**: Meta-analysis of 210 forgetting functions
- **Wixted & Ebbesen (1991)**: Power law vs exponential forgetting comparison

**Modern Memory Research Datasets**:
- **SuperMemo Database (1985-2024)**: 1B+ repetition records with SM-18 validation
- **Anki/FSRS Dataset (2023-2024)**: Open-source spaced repetition with 100M+ reviews
- **Memory Championship Data**: Expert vs novice retention patterns with neural imaging
- **Cognitive Individual Differences**: Working memory capacity correlations (Engle et al.)
- **Sleep and Consolidation Studies**: REM/NREM cycle effects on retention with polysomnography
- **O'Reilly Lab CLS Data**: Computational model validation against empirical findings
- **Sharp-Wave Ripple Studies**: Girardeau & Zugaro offline replay correlation with retention

**Biological Validation Sources**:
- **fMRI Studies**: Hippocampal vs neocortical activation during recall (Squire & Kandel, 2009)
- **Patient Studies**: H.M. and other medial temporal lobe lesion cases (Corkin, 2013)
- **EEG/MEG Studies**: Theta-gamma coupling during memory formation (Tort et al., 2009)
- **Sharp-Wave Ripples**: Offline replay and consolidation timing (Buzsáki, 2015)
- **Comparative Studies**: Cross-species memory consolidation patterns (Morris et al., 2003)
- **O'Reilly CLS Models**: Computational validation of hippocampal-neocortical interactions
- **REMERGE Validation**: Progressive semanticization timecourses (O'Reilly et al., 2014)
- **Oscillatory Constraints**: Theta (4-8Hz) and gamma (30-100Hz) boundaries from intracranial recordings

**Expected Validation Accuracy Targets**:
- **Ebbinghaus Curve**: <2% RMSE against 2015 replication data
- **Bahrick Permastore**: <5% error for 10+ year retention predictions
- **Power Law Fit**: R² > 0.95 for long-term retention data
- **SM-18 Intervals**: <10% deviation from optimal spacing predictions
- **Individual Differences**: ±15% prediction accuracy for cognitive variations

### Performance Targets with Cognitive Constraints

**Accuracy Targets** (Empirically Validated):
- **Ebbinghaus Replication**: <2% RMSE vs 2015 published data
- **Bahrick Permastore**: <5% error for 10+ year retention prediction
- **Power Law Fitting**: R² > 0.95 across multiple datasets
- **SM-18 Predictions**: <10% error vs optimal intervals
- **Cross-Validation**: >85% accuracy on held-out memory data

**Computational Performance** (Cognitive-Aware):
- **Decay Calculation**: <500ns per memory (allowing 2M calculations/second)
- **Batch Processing**: Vectorized operations achieving >80% SIMD efficiency
- **Cache Efficiency**: <5% cache misses for sequential decay updates
- **Memory Footprint**: O(1) per memory with lazy evaluation
- **Integration Cost**: <5% overhead when combined with spreading activation

**Biological Fidelity Constraints**:
- **Neural Timing**: Respect 100ms cognitive cycle boundaries
- **Refractory Periods**: Coordinate with 2-3ms absolute refractory constraints
- **Energy Budget**: Decay calculations respect metabolic limitations
- **Oscillatory Coupling**: Align with theta (4-8Hz) and gamma (30-100Hz) rhythms
- **Sleep Dependencies**: Model consolidation timing with circadian constraints

### Comprehensive Testing Strategy

#### 1. Scientific Validation and Replication
**Empirical Dataset Validation**:
- **Ebbinghaus Replication**: Reproduce 2015 study methodology and achieve <2% RMSE
- **Bahrick Longitudinal**: Validate 50-year retention predictions against published data
- **Meta-Analysis Comparison**: Test against Rubin & Wenzel's 210 forgetting functions
- **Cross-Cultural Studies**: Validate individual difference parameters across populations
- **Neuropsychological Cases**: Test predictions against lesion patient data

**Statistical Rigor**:
- **Goodness-of-Fit**: AIC/BIC model comparison across decay functions
- **Bootstrap Confidence**: 95% confidence intervals for all parameter estimates
- **Cross-Validation**: 10-fold validation on memory datasets
- **Effect Size Analysis**: Cohen's d for meaningful difference detection
- **Bayesian Model Selection**: Prior-informed parameter estimation

#### 2. Cognitive Architecture Integration Testing
**Complementary Learning Systems**:
- **Dual-Pathway Validation**: Test hippocampal vs neocortical decay rate differences
- **Consolidation Timeline**: Verify 3-6 year transition to permastore stability
- **Pattern Completion**: Test CA3-inspired completion thresholds
- **Schema Integration**: Validate overlap-dependent consolidation strengthening
- **Sleep-Dependent Consolidation**: Model REM/NREM cycle effects

**Neural Timing Constraints**:
- **Oscillatory Coupling**: Test theta-gamma phase relationships
- **Refractory Integration**: Ensure compatibility with 2-3ms constraints
- **Metabolic Budget**: Validate energy-constrained decay calculations
- **Spreading Activation**: Test integration with parallel activation framework
- **Working Memory Limits**: Verify 7±2 capacity constraints

#### 3. Performance and Scalability Testing
**Micro-Benchmark Suite**:
- **Single Memory Decay**: <500ns target with statistical validation
- **Batch Processing**: SIMD-optimized operations achieving >80% efficiency
- **Cache Performance**: Memory access pattern optimization
- **Lock-Free Operations**: Concurrent decay calculation scalability
- **Individual Differences**: Parameter variation computational cost

**Cognitive Workload Simulation**:
- **Realistic Memory Patterns**: Test on human-like forgetting distributions
- **Spaced Repetition**: Validate SM-18 interval calculations under load
- **Long-Term Simulation**: Model years of retention with statistical validation
- **Interference Effects**: Test proactive/retroactive interference modeling
- **Extreme Value Testing**: Handle edge cases (very old/new memories)

#### 4. Production Readiness Testing
**Reliability and Robustness**:
- **Memory Leak Detection**: Long-running simulation with resource monitoring
- **Numerical Stability**: Test mathematical operations at extreme values
- **Error Propagation**: Validate confidence interval propagation
- **Graceful Degradation**: Fallback behavior when validation fails
- **Deterministic Replay**: Seed-based reproducibility for debugging

**Integration Validation**:
- **HNSW Compatibility**: Test with memory-mapped persistence (Task 003)
- **Activation Spreading**: Coordinate timing with parallel spreading (Task 004)
- **Confidence Propagation**: Validate with cognitive confidence framework
- **Episode Integration**: Test temporal pattern preservation
- **Query Engine**: End-to-end retrieval with decay-adjusted confidence

## Enhanced Acceptance Criteria

### Empirical Validation Requirements
- [ ] **Ebbinghaus Replication**: <2% RMSE vs 2015 published replication data
- [ ] **Bahrick Permastore**: <5% error for retention predictions >10 years
- [ ] **Power Law Validation**: R² > 0.95 vs Wixted & Ebbesen dataset
- [ ] **SM-18 Algorithm**: <10% deviation from optimal spacing intervals
- [ ] **Cross-Dataset Generalization**: >85% accuracy on held-out memory data
- [ ] **Individual Differences**: ±15% prediction accuracy for cognitive variations

### Biological Plausibility Requirements
- [ ] **Dual-System Architecture**: Distinct hippocampal (fast, τ=hours-days) and neocortical (slow, τ=months-years) decay following CLS theory
- [ ] **REMERGE Dynamics**: Progressive episodic-to-semantic transformation over 2-3 year timeline with hippocampal independence
- [ ] **CA3 Pattern Completion**: Hippocampal memories reconstructable from 30-40% cue overlap following recurrent network dynamics
- [ ] **Systems Consolidation**: 3-6 year transition to stable permastore state with schema-dependent strengthening
- [ ] **Neural Constraints**: Compatible with 100ms cognitive cycles, 2-3ms refractory periods, and working memory limits (7±2)
- [ ] **Oscillatory Coupling**: Decay calculations aligned with theta (4-8Hz) encoding and gamma (30-100Hz) retrieval rhythms
- [ ] **Sharp-Wave Ripples**: Consolidation triggered by 100-250Hz ripple patterns during offline periods
- [ ] **Metabolic Budget**: Decay processes respect ATP availability and neural energy constraints from spreading activation
- [ ] **Sleep Dependencies**: REM/NREM-dependent consolidation with circadian timing constraints
- [ ] **Individual Differences**: Cognitive variations (±20%) based on working memory capacity, processing speed, and hippocampal efficiency

### Cognitive Architecture Integration
- [ ] **Complementary Learning**: Seamless hippocampal-neocortical pathway coordination with competitive dynamics during consolidation
- [ ] **Pattern Completion**: CA3-inspired completion thresholds (30-40% minimum cue overlap) with confidence-dependent adjustment
- [ ] **Schema Strengthening**: Consolidation boost proportional to semantic overlap with existing knowledge structures
- [ ] **Working Memory Integration**: Respect 7±2 capacity constraints and attention availability during decay calculations
- [ ] **Confidence System Integration**: Seamless integration with Engram's Confidence type including overconfidence calibration
- [ ] **Spreading Activation Coordination**: Synchronize decay timing with parallel activation framework and refractory periods
- [ ] **Memory-Episode Coupling**: Differential decay for Memory vs Episode types following episodic-semantic distinction
- [ ] **Interference Modeling**: Proactive and retroactive interference through overlapping hippocampal representations
- [ ] **Attention Dependencies**: Decay rate modulation based on available attention resources and cognitive load

### Performance and Efficiency Requirements  
- [ ] **Computation Speed**: <500ns per decay calculation (2M calculations/second)
- [ ] **Batch Efficiency**: >80% SIMD utilization for vectorized operations
- [ ] **Memory Footprint**: O(1) per memory with lazy evaluation
- [ ] **Cache Performance**: <5% miss rate for cognitive access patterns
- [ ] **Integration Overhead**: <5% performance impact when combined with spreading
- [ ] **Scalability**: Linear scaling with memory count up to 10M memories

### Production Readiness Requirements
- [ ] **Numerical Stability**: Robust operation across full time range (microseconds to decades)
- [ ] **Deterministic Replay**: Seed-based reproducibility for scientific validation
- [ ] **Error Propagation**: Proper confidence interval handling throughout system
- [ ] **Graceful Degradation**: Fallback to simpler models if validation fails
- [ ] **Resource Management**: Zero memory leaks in long-running simulations
- [ ] **Documentation**: Comprehensive empirical citations and parameter justification

## Integration Notes with Cognitive Architecture

### Spreading Activation Integration (Task 004)
**Neural Dynamics Coordination**:
- **Refractory Periods**: Decay calculations respect 2-3ms absolute and 10-20ms relative refractory constraints from neural physiology
- **Oscillatory Gating**: Theta-gamma coupling (Tort et al. 2009) affects decay computation timing with phase-dependent efficiency
- **Metabolic Budget**: Decay processes consume ATP-equivalent "neural energy" from spreading activation with competition dynamics
- **Synaptic Resources**: Coordinate with Tsodyks-Markram synaptic depression model for realistic transmission delays
- **Sharp-Wave Ripples**: Consolidation events triggered by 100-250Hz activation patterns during low-activity states
- **Theta Phase Reset**: Successful retrieval resets hippocampal theta phase affecting subsequent encoding efficiency
- **Gamma Coherence**: Neocortical decay influenced by 40-100Hz gamma coherence indicating active processing states

**Complementary Learning Systems**:
- **Hippocampal Circuit**: Fast decay (τ=hours-days) integrated with CA3 pattern separation/completion and CA1 sequence detection
- **Neocortical Circuit**: Slow decay (τ=months-years) coordinated with schema extraction and cortical-hippocampal dialogue
- **REMERGE Dynamics**: Progressive transfer from hippocampal episodic to neocortical semantic with competitive inhibition
- **Memory Consolidation**: Replay buffer priorities based on prediction error and novelty affect decay strengthening
- **Experience Replay**: Hippocampal sharp-wave ripple sequences prioritize high-error memories for consolidation
- **Systems-Level Competition**: Hippocampal and neocortical systems compete during retrieval with age-dependent weighting
- **Schema Integration**: Neocortical schemas provide top-down constraints on hippocampal pattern completion

### Memory-Mapped Persistence Integration (Task 003)
**Biologically-Inspired Tiered Storage Alignment**:
- **Hot Tier (Hippocampal)**: Recent memories (days-weeks) with active exponential decay and high consolidation activity
- **Warm Tier (Transfer)**: Transitioning memories (weeks-months) with mixed hippocampal-neocortical dynamics during systems consolidation
- **Cold Tier (Neocortical)**: Permastore memories (years-decades) with minimal power-law decay and schema protection
- **Decay State Persistence**: Retrievability/stability parameters, theta phase, and ripple counts survive system restarts
- **Lazy Loading**: Decay calculations triggered by tier access patterns following biological access timing constraints
- **Consolidation Checkpoints**: Sharp-wave ripple events create persistence checkpoints for critical memory states
- **Schema Indexing**: Neocortical memories indexed by semantic overlap for efficient schema-based retrieval

### Future Task Dependencies
**Query Engine Integration**:
- Confidence scores adjusted by decay-predicted retrievability
- Search ranking incorporates stability-based long-term predictions
- Result filtering based on decay-adjusted confidence thresholds

**Pattern Completion Integration**:
- Completion thresholds modulated by hippocampal decay state
- Partial cue effectiveness predicted by neocortical stability
- Schema-based completion strength enhanced by consolidation history

## Risk Mitigation Strategy

### Implementation Approach
**Phase 1: Foundation (Days 1-3)**
- Implement basic Ebbinghaus exponential decay with 2015 replication validation, integrated with Engram's Memory type
- Create individual difference parameter framework based on working memory capacity and processing speed
- Establish empirical validation infrastructure with CLS-specific metrics
- Integrate with existing Confidence system for seamless decay-confidence interactions

**Phase 2: Dual-System (Days 4-6)**
- Add hippocampal fast decay system with CA3 pattern completion thresholds and theta-gamma oscillatory constraints
- Implement neocortical slow decay with Bahrick permastore effects and schema-dependent consolidation
- Integrate REMERGE dynamics for progressive episodic-to-semantic transformation
- Coordinate with cognitive spreading activation timing constraints and sharp-wave ripple detection

**Phase 3: Advanced Models (Days 7-9)**
- Deploy SuperMemo SM-18 two-component model with biological substrate mapping and neural constraints
- Add consolidation strengthening with sleep-dependent effects following REM/NREM cycles
- Implement schema-based decay modulation with semantic overlap quantification
- Integrate interference modeling through overlapping hippocampal representations

**Phase 4: Optimization & Integration (Days 10-12)**
- SIMD-optimize batch decay calculations while maintaining biological timing constraints
- Integrate with memory-mapped persistence tiers following hippocampal-neocortical storage mapping
- Comprehensive empirical validation against CLS predictions and sharp-wave ripple data
- Production integration with existing Memory/Episode types and spreading activation system

### Risk Controls
**Scientific Validity**:
- Continuous validation against empirical datasets throughout development
- Statistical significance testing for all parameter changes
- Cross-validation on held-out memory data
- Peer review of mathematical model implementations

**Performance Risk**:
- Benchmark-driven development with automated regression detection
- Gradual complexity increase with performance validation at each step
- SIMD optimization guided by roofline analysis
- Memory access pattern optimization using hardware counters

**Integration Risk**:
- Feature flags for gradual rollout of decay functions
- Backward compatibility with existing activation spreading
- Comprehensive integration testing with cognitive architecture
- Fallback to simpler models if validation fails

**Biological Plausibility Risk**:
- Continuous validation against CLS theory and O'Reilly lab empirical findings
- Parameter ranges constrained by hippocampal-neocortical physiology and oscillatory boundaries
- Regular validation against sharp-wave ripple studies and systems consolidation literature
- Documentation of all biological assumptions with specific citations from Buzsáki, O'Reilly, and McClelland research
- Theta-gamma coupling constraints validated against Tort et al. (2009) modulation indices
- Working memory capacity constraints aligned with Engle et al. individual differences research