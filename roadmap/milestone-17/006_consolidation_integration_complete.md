# Task 006: Consolidation Integration with Biologically-Plausible Timescales

## Objective
Integrate concept formation and binding creation into the existing consolidation scheduler with biologically-plausible timescales, sleep-stage-aware processing, and gradual rollout controls that prevent excessive concept formation while maintaining system stability.

## Background
Memory consolidation operates on multiple timescales in the brain:
- Fast hippocampal encoding: seconds to minutes
- Initial consolidation: hours (first sleep cycle)
- Systems consolidation: days to weeks (repeated sleep cycles)
- Schema formation: weeks to months (asymptotic strengthening)

The existing DreamEngine implements sharp-wave ripple (SWR) replay with configurable sleep parameters. We must integrate concept formation such that:
1. Formation frequency respects biological resource constraints (sleep spindle density, synaptic energy budgets)
2. Timescales align with empirical consolidation data (Rasch & Born 2013, Diekelmann & Born 2010)
3. Sleep stage transitions modulate formation parameters appropriately
4. Gradual rollout enables validation before full deployment

Critical: Concept formation should NOT happen every consolidation cycle. Research shows semantic abstraction requires:
- Multiple replay events across different sleep stages (Mölle & Born 2011)
- Sufficient temporal spacing for interference reduction (Walker 2009)
- Accumulation of statistical evidence across 3+ episodes (Tse et al. 2007)

## Requirements
1. Extend ConsolidationEngine with sleep-stage-aware concept formation
2. Implement biologically-justified formation frequencies and thresholds
3. Add multi-tier rollout controls (shadow → 1% → 10% → 50% → 100%)
4. Track concept formation metrics with biological validation criteria
5. Design circuit breakers to prevent pathological formation rates
6. Ensure consolidation timing remains consistent with existing baselines
7. Validate against empirical memory consolidation timescales

## Technical Specification

### Files to Modify
- `engram-core/src/consolidation/dream.rs` - Add concept formation to DreamEngine
- `engram-core/src/consolidation/mod.rs` - Export concept formation types
- `engram-core/src/consolidation/service.rs` - Track concept formation metrics
- `engram-core/src/metrics/mod.rs` - Add concept formation metrics

### Files to Create
- `engram-core/src/consolidation/concept_integration.rs` - Integration logic with timescale controls

### Sleep Stage Configuration

```rust
use std::time::Duration;

/// Sleep stage modulates consolidation parameters based on neuroscience literature
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SleepStage {
    /// NREM Stage 2: Peak sleep spindle activity
    /// - Highest concept formation capacity (Mölle & Born 2011)
    /// - Strong hippocampal-cortical dialogue
    /// - Optimal for declarative memory consolidation
    NREM2,

    /// NREM Stage 3 (Slow-Wave Sleep): Deep consolidation
    /// - Sustained replay with slow oscillations
    /// - Moderate concept formation (Rasch & Born 2013)
    /// - Long-range cortical connections strengthened
    NREM3,

    /// REM Sleep: Selective consolidation
    /// - Emotional memory processing emphasized
    /// - Lower concept formation rate
    /// - Creative recombination of existing concepts
    REM,

    /// Quiet Waking: Minimal consolidation
    /// - Brief offline replay during rest
    /// - Very low concept formation
    /// - Primarily for immediate memory stabilization
    QuietWake,
}

impl SleepStage {
    /// Get concept formation probability for this sleep stage
    /// Based on empirical replay frequency and spindle density data
    #[must_use]
    pub fn concept_formation_probability(&self) -> f32 {
        match self {
            SleepStage::NREM2 => 0.15,      // 15% (peak spindle-ripple coupling)
            SleepStage::NREM3 => 0.08,      // 8% (sustained but lower density)
            SleepStage::REM => 0.03,        // 3% (selective processing)
            SleepStage::QuietWake => 0.01,  // 1% (minimal offline consolidation)
        }
    }

    /// Get replay capacity (max episodes) for this stage
    /// Derived from SWR frequency data (Wilson & McNaughton 1994)
    #[must_use]
    pub fn replay_capacity(&self) -> usize {
        match self {
            SleepStage::NREM2 => 100,   // High replay during spindles
            SleepStage::NREM3 => 80,    // Sustained during slow waves
            SleepStage::REM => 50,      // Selective replay
            SleepStage::QuietWake => 20, // Brief awake replay
        }
    }

    /// Typical duration of this sleep stage in minutes
    /// Used for consolidation cycle timing (Carskadon & Dement 2011)
    #[must_use]
    pub fn typical_duration_minutes(&self) -> u32 {
        match self {
            SleepStage::NREM2 => 20,    // ~20 min per cycle
            SleepStage::NREM3 => 30,    // ~30 min in early cycles
            SleepStage::REM => 15,      // ~15 min, increases across night
            SleepStage::QuietWake => 5, // Brief rest periods
        }
    }

    /// Minimum consolidation cycles before concept formation
    /// Ensures sufficient statistical evidence accumulation
    #[must_use]
    pub fn min_cycles_before_formation(&self) -> u32 {
        match self {
            SleepStage::NREM2 => 3,     // 3 spindle-coupled replays minimum
            SleepStage::NREM3 => 2,     // 2 deep consolidation passes
            SleepStage::REM => 5,       // More selective, needs more evidence
            SleepStage::QuietWake => 10, // Awake replay very conservative
        }
    }
}

/// Biologically-inspired consolidation cycle tracker
#[derive(Debug, Clone)]
pub struct ConsolidationCycleState {
    /// Current sleep stage
    pub sleep_stage: SleepStage,

    /// Number of consolidation cycles completed in current stage
    pub cycles_in_stage: u32,

    /// Total cycles across all stages in this sleep session
    pub total_cycles: u32,

    /// Last concept formation cycle number
    pub last_formation_cycle: u32,

    /// Episodes accumulated since last formation
    pub episodes_since_formation: usize,

    /// Timestamp of last formation event
    pub last_formation_time: chrono::DateTime<chrono::Utc>,
}

impl ConsolidationCycleState {
    /// Create new cycle state starting in NREM2
    #[must_use]
    pub fn new() -> Self {
        Self {
            sleep_stage: SleepStage::NREM2,
            cycles_in_stage: 0,
            total_cycles: 0,
            last_formation_cycle: 0,
            episodes_since_formation: 0,
            last_formation_time: chrono::Utc::now(),
        }
    }

    /// Check if concept formation should occur this cycle
    ///
    /// Criteria (ALL must be true):
    /// 1. Minimum cycles in stage reached (prevents premature formation)
    /// 2. Sufficient episodes accumulated (≥3 for statistical regularity)
    /// 3. Probabilistic gate based on sleep stage (models resource constraints)
    /// 4. Temporal spacing since last formation (≥30 min, prevents excessive formation)
    #[must_use]
    pub fn should_form_concepts(&self, rollout_sample_rate: f32) -> bool {
        // Criterion 1: Minimum cycles in current stage
        let min_cycles_met = self.cycles_in_stage >= self.sleep_stage.min_cycles_before_formation();

        // Criterion 2: Sufficient episodes (minimum 3 per Tse et al. 2007)
        let sufficient_episodes = self.episodes_since_formation >= 3;

        // Criterion 3: Temporal spacing (30 minutes minimum per Rasch & Born 2013)
        let time_since_formation = chrono::Utc::now()
            .signed_duration_since(self.last_formation_time);
        let sufficient_spacing = time_since_formation.num_minutes() >= 30;

        // Criterion 4: Sleep stage probability gate (models spindle/ripple coupling)
        let stage_probability = self.sleep_stage.concept_formation_probability();
        let probabilistic_gate = rand::random::<f32>() < stage_probability;

        // Criterion 5: Rollout control (gradual deployment)
        let rollout_gate = rand::random::<f32>() < rollout_sample_rate;

        min_cycles_met
            && sufficient_episodes
            && sufficient_spacing
            && probabilistic_gate
            && rollout_gate
    }

    /// Advance to next cycle, potentially transitioning sleep stage
    pub fn advance_cycle(&mut self, episodes_processed: usize) {
        self.cycles_in_stage += 1;
        self.total_cycles += 1;
        self.episodes_since_formation += episodes_processed;

        // Simulate sleep stage transitions (simplified 90-minute ultradian rhythm)
        // Real implementation would use external sleep stage detection
        if self.cycles_in_stage >= self.sleep_stage.typical_duration_minutes() / 10 {
            self.transition_sleep_stage();
        }
    }

    /// Record concept formation event
    pub fn record_formation(&mut self, concepts_formed: usize) {
        self.last_formation_cycle = self.total_cycles;
        self.last_formation_time = chrono::Utc::now();
        self.episodes_since_formation = 0;
    }

    /// Transition to next sleep stage in cycle
    fn transition_sleep_stage(&mut self) {
        self.sleep_stage = match self.sleep_stage {
            SleepStage::NREM2 => SleepStage::NREM3,
            SleepStage::NREM3 => SleepStage::REM,
            SleepStage::REM => SleepStage::NREM2, // Loop back for next cycle
            SleepStage::QuietWake => SleepStage::NREM2, // Enter sleep
        };
        self.cycles_in_stage = 0;
    }
}

impl Default for ConsolidationCycleState {
    fn default() -> Self {
        Self::new()
    }
}
```

### Extended DreamEngine Configuration

```rust
/// Extended dream configuration with concept formation controls
#[derive(Debug, Clone)]
pub struct ConceptFormationConfig {
    /// Enable concept formation (master switch)
    pub enabled: bool,

    /// Rollout sample rate (0.0 = shadow mode, 1.0 = full deployment)
    /// - 0.0: Shadow mode (log decisions but don't create concepts)
    /// - 0.01: 1% rollout (early validation)
    /// - 0.10: 10% rollout (broader testing)
    /// - 0.50: 50% rollout (pre-production)
    /// - 1.0: Full deployment
    pub rollout_sample_rate: f32,

    /// Minimum cluster size for concept formation (default: 3)
    /// Derived from Tse et al. (2007) schema formation requirements
    pub min_cluster_size: usize,

    /// Coherence threshold for viable concepts (default: 0.65)
    /// Matches CA3 pattern completion threshold (Nakazawa et al. 2002)
    pub coherence_threshold: f32,

    /// Maximum concepts per formation event (default: 5)
    /// Limited by sleep spindle density (Mölle & Born 2011)
    pub max_concepts_per_event: usize,

    /// Circuit breaker: max formation rate per hour
    /// Prevents pathological concept explosion (default: 20/hour)
    pub max_formations_per_hour: usize,

    /// Circuit breaker: max concept/episode ratio
    /// Prevents semantic inflation (default: 0.3 = 30% concepts)
    pub max_concept_ratio: f32,
}

impl Default for ConceptFormationConfig {
    fn default() -> Self {
        Self {
            enabled: false, // Disabled by default, explicit opt-in
            rollout_sample_rate: 0.0, // Start in shadow mode
            min_cluster_size: 3,
            coherence_threshold: 0.65,
            max_concepts_per_event: 5,
            max_formations_per_hour: 20,
            max_concept_ratio: 0.3,
        }
    }
}
```

### Integration with DreamEngine

```rust
// In engram-core/src/consolidation/dream.rs

use crate::consolidation::concept_formation::ConceptFormationEngine;
use crate::consolidation::binding_formation::BindingFormationEngine;

impl DreamEngine {
    /// Extended dream cycle with concept formation
    pub fn dream_with_concepts(
        &self,
        store: &MemoryStore,
        cycle_state: &mut ConsolidationCycleState,
        concept_config: &ConceptFormationConfig,
    ) -> Result<ExtendedDreamOutcome, DreamError> {
        let start = Instant::now();

        // Phase 1: Standard dream consolidation (existing)
        let base_outcome = self.dream(store)?;

        // Phase 2: Concept formation (conditionally enabled)
        let concept_stats = if concept_config.enabled
            && cycle_state.should_form_concepts(concept_config.rollout_sample_rate)
        {
            self.form_concepts_from_episodes(store, cycle_state, concept_config)?
        } else {
            ConceptFormationStats::default()
        };

        // Phase 3: Update cycle state
        cycle_state.advance_cycle(base_outcome.episodes_replayed);
        if concept_stats.concepts_formed > 0 {
            cycle_state.record_formation(concept_stats.concepts_formed);
        }

        Ok(ExtendedDreamOutcome {
            base: base_outcome,
            concepts_formed: concept_stats.concepts_formed,
            bindings_created: concept_stats.bindings_created,
            formation_skipped_reason: concept_stats.skip_reason,
            sleep_stage: cycle_state.sleep_stage,
            cycle_number: cycle_state.total_cycles,
        })
    }

    fn form_concepts_from_episodes(
        &self,
        store: &MemoryStore,
        cycle_state: &ConsolidationCycleState,
        config: &ConceptFormationConfig,
    ) -> Result<ConceptFormationStats, DreamError> {
        // Circuit breaker checks
        if !self.check_formation_circuit_breakers(store, config) {
            return Ok(ConceptFormationStats {
                concepts_formed: 0,
                bindings_created: 0,
                skip_reason: Some(SkipReason::CircuitBreakerTripped),
            });
        }

        // Get episodes replayed in this consolidation cycle
        let episodes = self.select_dream_episodes(store)?;

        if episodes.len() < config.min_cluster_size {
            return Ok(ConceptFormationStats {
                concepts_formed: 0,
                bindings_created: 0,
                skip_reason: Some(SkipReason::InsufficientEpisodes),
            });
        }

        // Form concepts with sleep-stage-aware parameters
        let concept_engine = ConceptFormationEngine::new(config.clone());
        let concepts = concept_engine.form_concepts(&episodes, cycle_state.sleep_stage)?;

        // Create bindings between episodes and concepts
        let binding_engine = BindingFormationEngine::new();
        let bindings = binding_engine.create_bindings(&episodes, &concepts)?;

        // Store in graph (shadow mode only logs, doesn't persist)
        if config.rollout_sample_rate > 0.0 {
            for concept in &concepts {
                store.add_concept_node(concept)?;
            }
            for binding in &bindings {
                store.add_concept_binding(binding)?;
            }
        } else {
            tracing::info!(
                concepts_formed = concepts.len(),
                bindings_created = bindings.len(),
                sleep_stage = ?cycle_state.sleep_stage,
                "Shadow mode: concept formation logged but not persisted"
            );
        }

        Ok(ConceptFormationStats {
            concepts_formed: concepts.len(),
            bindings_created: bindings.len(),
            skip_reason: None,
        })
    }

    fn check_formation_circuit_breakers(
        &self,
        store: &MemoryStore,
        config: &ConceptFormationConfig,
    ) -> bool {
        // Circuit breaker 1: Formation rate limit
        let recent_formations = store.count_concepts_formed_since(
            chrono::Utc::now() - chrono::Duration::hours(1)
        );
        if recent_formations >= config.max_formations_per_hour {
            tracing::warn!(
                recent_formations,
                max_allowed = config.max_formations_per_hour,
                "Circuit breaker: formation rate limit exceeded"
            );
            return false;
        }

        // Circuit breaker 2: Concept/episode ratio
        let episode_count = store.episode_count();
        let concept_count = store.concept_count();
        if episode_count > 0 {
            let current_ratio = concept_count as f32 / episode_count as f32;
            if current_ratio >= config.max_concept_ratio {
                tracing::warn!(
                    current_ratio,
                    max_ratio = config.max_concept_ratio,
                    episode_count,
                    concept_count,
                    "Circuit breaker: concept/episode ratio exceeded"
                );
                return false;
            }
        }

        true
    }
}

/// Extended dream outcome with concept formation stats
#[derive(Debug, Clone)]
pub struct ExtendedDreamOutcome {
    /// Base dream consolidation stats
    pub base: DreamOutcome,

    /// Number of concepts formed
    pub concepts_formed: usize,

    /// Number of bindings created
    pub bindings_created: usize,

    /// Reason formation was skipped (if applicable)
    pub formation_skipped_reason: Option<SkipReason>,

    /// Sleep stage during this cycle
    pub sleep_stage: SleepStage,

    /// Cycle number in consolidation sequence
    pub cycle_number: u32,
}

#[derive(Debug, Clone)]
pub struct ConceptFormationStats {
    pub concepts_formed: usize,
    pub bindings_created: usize,
    pub skip_reason: Option<SkipReason>,
}

impl Default for ConceptFormationStats {
    fn default() -> Self {
        Self {
            concepts_formed: 0,
            bindings_created: 0,
            skip_reason: None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SkipReason {
    /// Rollout sample rate excluded this cycle
    RolloutSample,

    /// Minimum cycles in sleep stage not reached
    InsufficientCycles,

    /// Less than 3 episodes accumulated
    InsufficientEpisodes,

    /// Less than 30 minutes since last formation
    InsufficientTemporalSpacing,

    /// Probabilistic sleep stage gate
    SleepStageProbability,

    /// Circuit breaker tripped (rate limit or ratio exceeded)
    CircuitBreakerTripped,
}
```

### Metrics Integration

```rust
// In engram-core/src/metrics/mod.rs

/// Total concept formation events across all consolidation cycles
pub const CONCEPT_FORMATION_TOTAL: &str = "engram_concept_formation_total";

/// Concepts formed per formation event (histogram)
pub const CONCEPT_FORMATION_SIZE: &str = "engram_concept_formation_size";

/// Bindings created per formation event (histogram)
pub const CONCEPT_BINDINGS_CREATED: &str = "engram_concept_bindings_created";

/// Coherence score distribution for formed concepts (histogram)
pub const CONCEPT_COHERENCE_SCORE: &str = "engram_concept_coherence_score";

/// Time since last concept formation (gauge, seconds)
pub const CONCEPT_FORMATION_INTERVAL: &str = "engram_concept_formation_interval_seconds";

/// Current concept/episode ratio (gauge)
pub const CONCEPT_EPISODE_RATIO: &str = "engram_concept_episode_ratio";

/// Concept formation skip reasons (counter by reason)
pub const CONCEPT_FORMATION_SKIPPED: &str = "engram_concept_formation_skipped_total";

/// Circuit breaker trips (counter by type: rate_limit, ratio_exceeded)
pub const CONCEPT_CIRCUIT_BREAKER_TRIPS: &str = "engram_concept_circuit_breaker_trips_total";

/// Sleep stage during concept formation (counter by stage)
pub const CONCEPT_FORMATION_BY_STAGE: &str = "engram_concept_formation_by_stage_total";

// In engram-core/src/consolidation/service.rs

impl ConsolidationService for InMemoryConsolidationService {
    fn update_cache(&self, snapshot: &ConsolidationSnapshot, source: ConsolidationCacheSource) {
        // Existing implementation...

        // Add concept formation metrics
        if let Some(concept_stats) = &snapshot.concept_stats {
            metrics::increment_counter(
                metrics::CONCEPT_FORMATION_TOTAL,
                concept_stats.formations_this_session,
            );

            metrics::record_histogram(
                metrics::CONCEPT_FORMATION_SIZE,
                concept_stats.avg_concepts_per_formation as f64,
            );

            metrics::record_gauge(
                metrics::CONCEPT_EPISODE_RATIO,
                concept_stats.current_ratio as f64,
            );

            metrics::record_gauge(
                metrics::CONCEPT_FORMATION_INTERVAL,
                concept_stats.seconds_since_last_formation as f64,
            );
        }
    }
}
```

### Gradual Rollout Configuration

```toml
# config/consolidation.toml

[consolidation.dream]
dream_duration_secs = 600          # 10 minutes
replay_speed = 15.0                # 15x faster
replay_iterations = 5
ripple_frequency = 200.0           # 200 Hz
min_episode_age_secs = 86400       # 1 day
enable_compaction = true

[consolidation.concepts]
# Master enable (default: false for safety)
enabled = false

# Rollout stages (change gradually):
# - 0.0 = shadow mode (log only, don't create)
# - 0.01 = 1% rollout (A/B test validation)
# - 0.10 = 10% rollout (broader validation)
# - 0.50 = 50% rollout (pre-production)
# - 1.0 = full deployment
rollout_sample_rate = 0.0

# Biological parameters (validated against neuroscience literature)
min_cluster_size = 3               # Tse et al. 2007
coherence_threshold = 0.65         # Nakazawa et al. 2002 (CA3 completion)
max_concepts_per_event = 5         # Mölle & Born 2011 (spindle density)

# Circuit breakers (prevent pathological behavior)
max_formations_per_hour = 20       # Rate limit
max_concept_ratio = 0.3            # 30% max concepts relative to episodes
```

### Rollout Strategy

**Phase 1: Shadow Mode (rollout_sample_rate = 0.0)**
- Duration: 1 week
- Goal: Validate formation logic without side effects
- Metrics: Log all formation decisions, measure skip reasons
- Validation:
  - Formation frequency matches biological expectations (0.5-2 per hour during sleep)
  - Coherence scores distribute in [0.65, 0.95] range
  - Skip reasons are predominantly temporal spacing or sleep stage probability
  - Circuit breakers never trip (indicates healthy parameters)

**Phase 2: 1% Rollout (rollout_sample_rate = 0.01)**
- Duration: 1 week
- Goal: Validate persistence and retrieval correctness
- Metrics:
  - Concept/episode ratio stays below 0.1
  - Formed concepts are retrievable
  - Bindings enable correct bottom-up and top-down traversal
- Validation:
  - No memory leaks (monitor with metrics::CONCEPT_EPISODE_RATIO)
  - Consolidation latency increase <10%
  - Concept coherence distribution matches shadow mode

**Phase 3: 10% Rollout (rollout_sample_rate = 0.10)**
- Duration: 2 weeks
- Goal: Validate at scale with realistic load
- Metrics:
  - Formation rate per sleep stage matches expected distribution
  - Circuit breakers trip <0.1% of cycles (indicates rare edge cases)
  - Concept quality metrics (coherence, cluster size) stable
- Validation:
  - System remains stable over 7-day continuous run
  - No consolidation regressions (compare to baseline)
  - Concept retrieval accuracy >95%

**Phase 4: 50% Rollout (rollout_sample_rate = 0.50)**
- Duration: 1 month
- Goal: Pre-production validation
- Metrics: All metrics stable across diverse workloads
- Validation: Production readiness checklist complete

**Phase 5: Full Deployment (rollout_sample_rate = 1.0)**
- Enabled after successful 50% rollout
- Continuous monitoring with alerts on:
  - Circuit breaker trip rate >1%
  - Concept/episode ratio >0.25
  - Formation rate >25/hour (indicates parameter drift)
  - Consolidation latency increase >15%

## Biological Plausibility Validation

### Formation Frequency
**Expected**: 0.5-2 formations per hour during sleep-like consolidation
**Basis**:
- NREM cycles every 90 minutes with 20-30 min of peak spindle activity
- 15% formation probability during NREM2 → ~1 formation per 90-min cycle
- Matches empirical schema consolidation timescales (Tse et al. 2007)

**Metric**: `CONCEPT_FORMATION_INTERVAL` should show mean ~45-90 minutes

### Coherence Distribution
**Expected**: Right-skewed distribution with mode around 0.70-0.80
**Basis**:
- Threshold at 0.65 (CA3 pattern completion minimum)
- Most viable concepts have strong coherence (>0.70)
- Very high coherence (>0.90) rare (overfitting to specific episodes)

**Metric**: `CONCEPT_COHERENCE_SCORE` histogram should show:
- 10th percentile: ~0.66
- Median: ~0.75
- 90th percentile: ~0.88

### Sleep Stage Distribution
**Expected**:
- 60% formations during NREM2
- 25% during NREM3
- 10% during REM
- 5% during quiet wake

**Basis**: Formation probability × stage duration
- NREM2: 15% probability × 20 min = highest absolute count
- NREM3: 8% probability × 30 min = moderate count
- REM: 3% probability × 15 min = low count

**Metric**: `CONCEPT_FORMATION_BY_STAGE` should match expected distribution ±10%

## Implementation Notes

### Critical Timing Constraints
1. Concept formation must NOT block dream consolidation
2. Formation decision latency <1ms (minimal overhead)
3. Actual formation (clustering + binding) can take up to 100ms
4. Circuit breaker checks must be O(1) (use cached counters)

### Determinism Requirements
For M14 distributed consolidation:
- Formation decisions must be deterministic given same cycle_state and episode set
- Use deterministic RNG seeded by cycle number for reproducibility
- Skip reasons must be consistently logged across nodes

### Error Handling
- Formation failures should log but not crash consolidation
- Circuit breaker trips are warnings, not errors
- Invalid coherence scores (NaN, Inf) trigger fallback to pattern detection

## Testing Approach

### Unit Tests
1. **ConsolidationCycleState**
   - Test sleep stage transitions match 90-minute ultradian rhythm
   - Verify minimum cycle requirements enforced
   - Test temporal spacing calculation accuracy

2. **Circuit Breakers**
   - Test rate limit enforcement
   - Test ratio limit enforcement
   - Verify breakers reset after cooldown period

3. **Rollout Controls**
   - Test shadow mode logs without persisting
   - Verify sample rate affects formation frequency
   - Test 0% and 100% edge cases

### Integration Tests
1. **Multi-Cycle Consolidation**
   - Run 10 consecutive consolidation cycles
   - Verify formation occurs at expected frequency
   - Check consolidation latency stays within budget

2. **Sleep Stage Progression**
   - Simulate full sleep cycle (NREM2 → NREM3 → REM → NREM2)
   - Verify formation frequency matches stage probabilities
   - Test stage-specific parameter modulation

3. **Circuit Breaker Scenarios**
   - Force rapid formation attempts
   - Verify rate limiter prevents excessive formation
   - Test ratio limiter with concept-heavy workload

### Acceptance Tests
1. **Biological Timescale Validation**
   - 7-day continuous run with consolidation every 10 minutes
   - Measure formation frequency: expected ~1-2 per hour
   - Validate coherence distribution matches CA3 dynamics
   - Confirm sleep stage distribution aligns with probabilities

2. **Rollout Progression**
   - Complete shadow mode validation (1 week)
   - Progress through 1%, 10%, 50% rollouts
   - Verify metrics stability at each stage

3. **Performance Baseline**
   - Compare consolidation latency with/without concept formation
   - Ensure <10% increase in P99 latency
   - Verify no regressions in existing consolidation metrics

## Acceptance Criteria
- [ ] Concept formation integrates without breaking existing consolidation
- [ ] Formation frequency matches biological expectations (0.5-2/hour during sleep)
- [ ] Sleep stage transitions modulate formation appropriately
- [ ] Circuit breakers prevent pathological formation rates (<1% trip rate)
- [ ] Rollout controls enable safe progressive deployment
- [ ] Shadow mode logs decisions without side effects
- [ ] Coherence distribution aligns with CA3 pattern completion (mode ~0.75)
- [ ] Temporal spacing enforced (≥30 min between formations)
- [ ] Consolidation latency increase <10% (P99)
- [ ] Metrics properly track all formation events and skip reasons
- [ ] Formation decisions are deterministic (critical for M14)
- [ ] 7-day continuous run remains stable with concept formation enabled

## Dependencies
- Task 004 (Concept Formation Engine)
- Task 005 (Binding Formation)
- Existing DreamEngine (engram-core/src/consolidation/dream.rs)
- Existing ConsolidationService (engram-core/src/consolidation/service.rs)

## Estimated Time
3 days

### Day 1: Core Integration
- Implement ConsolidationCycleState with sleep stage tracking
- Add ConceptFormationConfig with rollout controls
- Integrate with DreamEngine::dream_with_concepts
- Unit tests for cycle state and configuration

### Day 2: Circuit Breakers and Metrics
- Implement formation circuit breakers (rate limit, ratio limit)
- Add concept formation metrics to metrics/mod.rs
- Integrate metrics recording in ConsolidationService
- Unit tests for circuit breakers and metrics

### Day 3: Validation and Rollout
- Implement gradual rollout logic (shadow mode through 100%)
- Add biological plausibility validation tests
- Run 7-day soak test scenario (accelerated)
- Document rollout strategy and validation criteria

## Key References

### Systems Consolidation Timescales
1. **Rasch & Born (2013)** - "About sleep's role in memory." Physiological Reviews 93(2): 681-766. [Consolidation windows: p. 720-725]
2. **Diekelmann & Born (2010)** - "The memory function of sleep." Nature Reviews Neuroscience 11(2): 114-126. [Sleep stage effects: p. 118-120]
3. **Frankland & Bontempi (2005)** - "The organization of recent and remote memories." Nature Reviews Neuroscience 6(2): 119-130. [Timescale gradients: p. 124-126]

### Sleep Spindles and Consolidation Capacity
4. **Mölle & Born (2011)** - "Slow oscillations orchestrating fast oscillations and memory consolidation." Progress in Brain Research 193: 93-110. [Spindle-ripple coupling: p. 99-103]
5. **Schabus et al. (2004)** - "Sleep spindles and their significance for declarative memory consolidation." Sleep 27(8): 1479-1485. [Spindle density limits: p. 1481-1483]

### Schema Formation Requirements
6. **Tse et al. (2007)** - "Schemas and memory consolidation." Science 316(5821): 76-82. [Minimum trial requirements: p. 78-79]
7. **van Kesteren et al. (2012)** - "How schema and novelty augment memory formation." Trends in Neurosciences 35(4): 211-219. [Statistical regularity: p. 214-216]

### Pattern Completion Thresholds
8. **Nakazawa et al. (2002)** - "Requirement for hippocampal CA3 NMDA receptors in associative memory recall." Science 297(5579): 211-218. [CA3 completion threshold: p. 216]

### Interference and Temporal Spacing
9. **Walker (2009)** - "The role of sleep in cognition and emotion." Annals of the New York Academy of Sciences 1156: 168-197. [Temporal spacing effects: p. 178-182]

## Notes
This integration design respects the multi-timescale nature of biological memory consolidation. Unlike immediate pattern detection (which runs every cycle), concept formation is gated by multiple biological constraints: minimum replay iterations, temporal spacing, sleep stage appropriateness, and resource limitations. This mirrors how the brain doesn't form schemas from every experience immediately, but rather extracts them gradually through repeated consolidation cycles spaced across multiple sleep periods.

The circuit breakers are critical for system stability - they prevent pathological scenarios where concept formation runs away (e.g., forming concepts from concepts, leading to exponential growth). The gradual rollout strategy enables careful validation at each stage before broader deployment, following production engineering best practices while maintaining biological plausibility.
