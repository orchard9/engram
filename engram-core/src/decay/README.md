# Decay Module

Biologically-inspired temporal decay system for Engram, implementing complementary learning systems (CLS) theory with empirical validation.

## Overview

This module provides psychological decay functions grounded in decades of memory research. It achieves <3% deviation from empirical data (Ebbinghaus, Bahrick, Wixted) while integrating seamlessly with Engram's existing Memory/Episode types and Confidence system.

**Key features:**
- Dual hippocampal/neocortical decay systems following CLS theory
- Four decay functions validated against psychological research
- Lazy evaluation (compute-on-read, zero write amplification)
- Per-memory configuration with system-wide defaults
- Individual differences calibration
- Thread-safe, lock-free implementation

## Architecture

### Complementary Learning Systems (CLS)

The module implements a dual-system architecture based on McClelland, McNaughton & O'Reilly (1995):

```
┌─────────────────────────────────────────────────────────┐
│              BiologicalDecaySystem                       │
│                                                           │
│  ┌───────────────────┐      ┌───────────────────┐       │
│  │ Hippocampal       │      │ Neocortical       │       │
│  │ System            │      │ System            │       │
│  ├───────────────────┤      ├───────────────────┤       │
│  │ - Fast decay      │      │ - Slow decay      │       │
│  │ - Exponential     │◄─────┤ - Power-law       │       │
│  │ - Pattern sep.    │      │ - Schema extract  │       │
│  │ - τ ≈ hours-days  │      │ - τ ≈ months-years│       │
│  └───────────────────┘      └───────────────────┘       │
│           ▲                          ▲                   │
│           │                          │                   │
│           └──────────┬───────────────┘                   │
│                      │                                   │
│           ┌──────────▼──────────┐                        │
│           │  TwoComponentModel  │                        │
│           │  (Auto-switching)   │                        │
│           └─────────────────────┘                        │
│                                                           │
│  ┌──────────────────────────────────────────┐            │
│  │    Individual Difference Calibration     │            │
│  └──────────────────────────────────────────┘            │
└─────────────────────────────────────────────────────────┘
```

### Design Principles

1. **Lazy Evaluation**: Decay computed during recall, not during storage
   - Zero write amplification
   - Deterministic results (same query at time T always returns same result)
   - Enables time-travel queries

2. **Biological Plausibility**: Matches human memory systems
   - Hippocampal system: fast episodic decay
   - Neocortical system: slow semantic decay
   - Systems consolidation: gradual transfer over time

3. **Empirical Validation**: All functions validated against research
   - Ebbinghaus (1885, 2015 replication): Exponential curve
   - Bahrick (1984): Power-law permastore
   - Wixted & Ebbesen (1991): Function comparison

## Module Organization

### Core Files

```
src/decay/
├── mod.rs                       # Public API and BiologicalDecaySystem
├── hippocampal.rs              # Fast exponential decay (Ebbinghaus)
├── neocortical.rs              # Slow power-law decay (Bahrick)
├── two_component.rs            # Automatic hippocampal ↔ neocortical switching
├── calibration.rs              # Confidence calibration utilities
├── individual_differences.rs   # Cognitive variation modeling
├── consolidation.rs            # Memory consolidation mechanisms
├── remerge.rs                  # Episodic-to-semantic transformation
├── spacing.rs                  # Spaced repetition optimization
├── oscillatory.rs              # Theta/gamma rhythm constraints
├── interference.rs             # Interference modeling
└── validation.rs               # Empirical validation utilities
```

### Key Types

#### `BiologicalDecaySystem`
Main entry point for decay functionality. Integrates all subsystems into a unified biologically plausible system.

**Location:** `mod.rs:104-328`

**Fields:**
- `hippocampal: HippocampalDecayFunction` - Fast episodic decay
- `neocortical: NeocorticalDecayFunction` - Slow semantic decay
- `two_component: TwoComponentModel` - Automatic system switching
- `individual_profile: IndividualDifferenceProfile` - Cognitive variation
- `remerge: RemergeProcessor` - Episodic-to-semantic transformation
- `config: DecayConfig` - System-wide configuration

**Key methods:**
```rust
// Create with default configuration
let system = BiologicalDecaySystem::new();

// Create with custom configuration
let config = DecayConfigBuilder::new().exponential(2.0).build();
let system = BiologicalDecaySystem::with_config(config);

// Compute decayed confidence (lazy evaluation)
let decayed = system.compute_decayed_confidence(
    base_confidence,
    elapsed_time,
    access_count,
    created_at,
    decay_override,
);

// Predict future retention
let retention = system.predict_retention(&memory, future_time);
```

#### `DecayFunction`
Enum specifying which decay model to use.

**Location:** `mod.rs:507-602`

**Variants:**
```rust
DecayFunction::Exponential { tau_hours: f32 }
DecayFunction::PowerLaw { beta: f32 }
DecayFunction::TwoComponent { consolidation_threshold: u64 }
DecayFunction::Hybrid {
    short_term_tau: f32,
    long_term_beta: f32,
    transition_point: u64,
}
```

**Factory methods:**
```rust
DecayFunction::exponential()     // tau=1.96h (Ebbinghaus)
DecayFunction::power_law()       // beta=0.18 (Bahrick)
DecayFunction::two_component()   // threshold=3 (default)
DecayFunction::hybrid()          // Best fit to Ebbinghaus data
```

#### `DecayConfig`
System-wide configuration for decay behavior.

**Location:** `mod.rs:609-628`

**Fields:**
```rust
default_function: DecayFunction  // Default for all memories
enabled: bool                    // Master enable/disable
min_confidence: f32              // Minimum threshold (prevents complete forgetting)
```

#### `DecayConfigBuilder`
Fluent API for constructing decay configurations.

**Location:** `mod.rs:654-734`

**Example:**
```rust
let config = DecayConfigBuilder::new()
    .exponential(2.0)           // Use exponential decay, tau=2h
    .min_confidence(0.15)       // Never go below 15%
    .enabled(true)              // Enable decay
    .build();
```

#### `DecayIntegration` (Trait)
Interface for applying decay to Engram's memory types.

**Location:** `mod.rs:67-96`

**Methods:**
```rust
fn apply_to_memory(&self, memory: &mut Memory, elapsed_time: Duration) -> Confidence;
fn apply_to_episode(&self, episode: &mut Episode, elapsed_time: Duration) -> Confidence;
fn update_on_recall(&mut self, success: bool, confidence: Confidence, response_time: StdDuration);
fn should_consolidate(&self, activation_pattern: &[f32]) -> bool;
```

### Subsystem Types

#### `HippocampalDecayFunction`
Fast exponential decay for episodic memories.

**File:** `hippocampal.rs`

**Formula:** `R(t) = e^(-t/τ)`

**Default tau:** 1.96 hours (Murre & Dros 2015 replication)

#### `NeocorticalDecayFunction`
Slow power-law decay for semantic memories.

**File:** `neocortical.rs`

**Formula:** `R(t) = (1 + t)^(-α)`

**Default alpha:** 0.18 (Bahrick 1984 permastore)

#### `TwoComponentModel`
Automatic switching between hippocampal and neocortical based on access patterns.

**File:** `two_component.rs`

**Behavior:**
- `access_count < threshold`: Hippocampal (fast) decay
- `access_count >= threshold`: Neocortical (slow) decay

**Default threshold:** 3 accesses (matches testing effect research)

#### `IndividualDifferenceProfile`
Models natural variation in human memory performance.

**File:** `individual_differences.rs`

**Purpose:** Adds realistic variance (~10-15%) to prevent overfitting to exact mathematical curves

#### `ConsolidationProcessor`
Handles memory strengthening through retrieval and offline consolidation.

**File:** `consolidation.rs`

**Features:**
- Sharp-wave ripple detection
- Testing effect implementation
- Sleep-dependent consolidation

#### `RemergeProcessor`
Implements progressive episodic-to-semantic transformation.

**File:** `remerge.rs`

**Timeline:** 2-3 years for full systems consolidation (O'Reilly et al. 2014)

## Integration with Engram

### With `Memory` and `Episode` Types

The decay system integrates with existing Engram types:

```rust
use engram_core::{Episode, Memory, Confidence};
use engram_core::decay::{BiologicalDecaySystem, DecayFunction};
use std::sync::Arc;

// System-wide configuration
let decay_system = Arc::new(BiologicalDecaySystem::default());

// Episodes support per-memory overrides
let mut episode = Episode::new(...);
episode.decay_function = Some(DecayFunction::PowerLaw { beta: 0.1 });

// Compute decay during recall
let elapsed = Utc::now() - episode.last_recall;
let decayed_confidence = decay_system.compute_decayed_confidence(
    episode.encoding_confidence,
    elapsed.to_std()?,
    episode.recall_count,
    episode.when,
    episode.decay_function,
);
```

### With `CognitiveRecall`

Decay is applied during the recall pipeline:

```rust
use engram_core::activation::CognitiveRecallBuilder;

let recall = CognitiveRecallBuilder::new()
    .vector_seeder(seeder)
    .spreading_engine(engine)
    .decay_system(decay_system)  // Attach decay system
    .build()?;

// Recall automatically applies decay
let results = recall.recall(&cue, &store)?;
```

### With `MemoryStore`

The store tracks access metadata needed for decay:

```rust
// Store updates last_recall and recall_count
episode.last_recall = Utc::now();
episode.recall_count += 1;
store.store(episode);
```

### With `Confidence` System

Decay integrates with probabilistic confidence:

```rust
// Decay produces confidence values
let decayed: Confidence = decay_system.compute_decayed_confidence(...);

// Respects min_confidence threshold
let clamped = if decayed.raw() < config.min_confidence {
    Confidence::exact(config.min_confidence)
} else {
    decayed
};
```

## Usage Examples

### Example 1: System-Wide Exponential Decay

```rust
use engram_core::decay::{BiologicalDecaySystem, DecayConfigBuilder};
use std::sync::Arc;

let config = DecayConfigBuilder::new()
    .exponential(2.0)      // 2 hour time constant
    .min_confidence(0.1)
    .enabled(true)
    .build();

let decay_system = Arc::new(BiologicalDecaySystem::with_config(config));
```

### Example 2: Per-Memory Override

```rust
// System uses two-component by default
let decay_system = Arc::new(BiologicalDecaySystem::default());

// But critical memories get special slow decay
let mut critical_episode = Episode::new(...);
critical_episode.decay_function = Some(
    DecayFunction::PowerLaw { beta: 0.05 }  // Very slow
);

// And temporary data gets fast decay
let mut temp_episode = Episode::new(...);
temp_episode.decay_function = Some(
    DecayFunction::Exponential { tau_hours: 0.5 }  // 30 minutes
);
```

### Example 3: Spaced Repetition

```rust
// Two-component model perfect for flashcards
let config = DecayConfigBuilder::new()
    .two_component(3)  // Consolidate after 3 correct recalls
    .build();

let decay_system = Arc::new(BiologicalDecaySystem::with_config(config));

// New cards (recall_count < 3) decay quickly → shown frequently
// Mastered cards (recall_count >= 3) decay slowly → shown rarely
```

### Example 4: Lazy Evaluation

```rust
// Decay computed at query-time, not storage-time
let episode = store.get_episode(&id)?;
let elapsed = Utc::now() - episode.last_recall;

// Compute current confidence without mutating episode
let current_confidence = decay_system.compute_decayed_confidence(
    episode.encoding_confidence,
    elapsed.to_std()?,
    episode.recall_count,
    episode.when,
    episode.decay_function,
);

println!("Stored confidence: {:.3}", episode.encoding_confidence.raw());
println!("Current confidence: {:.3}", current_confidence.raw());
println!("Decay: {:.1}%", (1.0 - current_confidence.raw() / episode.encoding_confidence.raw()) * 100.0);
```

### Example 5: Disabling Decay

```rust
// For performance testing or specific use cases
let config = DecayConfigBuilder::new()
    .enabled(false)
    .build();

let decay_system = Arc::new(BiologicalDecaySystem::with_config(config));

// compute_decayed_confidence returns input unchanged (~2μs overhead)
```

## Testing

### Unit Tests

Run module unit tests:
```bash
cargo test --lib --package engram-core decay::
```

Key test files:
- `mod.rs:736-1083` - BiologicalDecaySystem tests
- `hippocampal.rs` - Exponential decay validation
- `neocortical.rs` - Power-law decay validation
- `two_component.rs` - Consolidation threshold tests

### Integration Tests

Run temporal decay integration tests:
```bash
cargo test --test temporal_integration_test
cargo test --test temporal_edge_cases_test
```

Test coverage:
- 11 integration tests (end-to-end behavior)
- 15 edge case tests (boundary conditions, thread safety)
- All 4 decay functions validated

### Benchmarks

Run performance benchmarks:
```bash
cargo bench --bench temporal_performance
```

Validates <1ms P95 target for decay computation.

### Empirical Validation

Run validation against psychological research:
```bash
cargo test --test forgetting_curves_validation
```

Ensures <5% error vs:
- Ebbinghaus (1885) data
- Bahrick (1984) permastore
- Wixted & Ebbesen (1991) comparisons

## Performance Characteristics

### Computation Time

Benchmarked on M1 Mac (single-threaded):

| Function | Mean Time | P95 Time | Operations |
|----------|-----------|----------|------------|
| Disabled | ~2μs | ~3μs | Early return |
| Exponential | ~48μs | ~65μs | One `exp()` |
| Power-Law | ~82μs | ~110μs | One `pow()` |
| Two-Component | ~56μs | ~75μs | `if` + `exp()` |
| Hybrid | ~64μs | ~85μs | `if` + `exp()/pow()` |

**All functions well under 1ms target.**

### Thread Safety

- `BiologicalDecaySystem` is immutable after creation
- Thread-safe via `Arc<BiologicalDecaySystem>`
- No locks, no contention
- Pure computation, no shared mutable state

### Memory Overhead

Per-memory storage:
- `last_recall: DateTime<Utc>` - 12 bytes
- `recall_count: u64` - 8 bytes
- `decay_function: Option<DecayFunction>` - ~16 bytes (when Some)

**Total: 20-36 bytes per memory**

### Scaling

Linear cost with memory count:
- 100 memories: ~5-8ms
- 1,000 memories: ~50-80ms
- 10,000 memories: ~500-800ms

Parallelizable for large batches.

## Scientific Foundation

### Key References

1. **Ebbinghaus, H. (1885)**. Memory: A Contribution to Experimental Psychology
   - Original forgetting curve research

2. **Murre, J. M. J., & Dros, J. (2015)**. Replication and Analysis of Ebbinghaus' Forgetting Curve
   - Modern validation: tau = 1.96 hours ± 0.3

3. **Bahrick, H. P. (1984)**. Semantic Memory Content in Permastore
   - 50-year Spanish language retention study
   - Power-law with alpha ≈ 0.15-0.20

4. **Wickelgren, W. A. (1974)**. Single-trace Fragility Theory of Memory Dynamics
   - Mathematical framework for power-law forgetting

5. **Wixted, J. T., & Ebbesen, E. B. (1991)**. On the Form of Forgetting
   - Comprehensive comparison: power-law superior for long retention intervals

6. **McClelland, J. L., McNaughton, B. L., & O'Reilly, R. C. (1995)**. Why There Are Complementary Learning Systems
   - CLS theory: hippocampal fast learning, neocortical slow learning

7. **O'Reilly, R. C., Bhattacharyya, R., Howard, M. D., & Ketz, N. (2014)**. Complementary Learning Systems
   - REMERGE model of progressive semanticization

8. **SuperMemo Algorithm SM-18 (2024)**. Two-component model with adaptive parameters
   - Spaced repetition optimization

### Validation Results

All decay functions validated to <5% error:

**Exponential vs Ebbinghaus (1885):**
- Mean absolute error: 1.26%
- Max error: 2.2% (at 20 minutes)

**Power-Law vs Bahrick (1984):**
- Mean absolute error: 1.18%
- Max error: 1.8% (at 3 years and 50 years)

## Future Enhancements

Potential additions for future milestones:

1. **Context-dependent decay** - Location/time-of-day influences
2. **Interference modeling** - Competing memories accelerate decay
3. **Sleep consolidation** - Offline strengthening during system idle
4. **Adaptive parameters** - ML-tuned decay rates based on patterns
5. **Oscillatory gating** - Theta/gamma rhythm modulation

## External Documentation

For detailed usage guides and mathematical specifications:

- **[Temporal Dynamics Architecture](../../../docs/temporal-dynamics.md)** - High-level design principles
- **[Decay Functions Reference](../../../docs/decay-functions.md)** - Mathematical specifications and validation
- **[Configuration Tutorial](../../../docs/tutorials/temporal-configuration.md)** - Practical usage examples

## Contributing

When modifying the decay module:

1. **Maintain biological plausibility** - All changes should be grounded in cognitive psychology research
2. **Validate empirically** - Update validation tests if changing decay formulas
3. **Preserve performance** - All functions must stay <1ms P95
4. **Update documentation** - Keep this README and external docs in sync
5. **Add tests** - Integration tests for behavior, unit tests for edge cases

## API Stability

The decay module API is considered **stable** as of Milestone 4. Breaking changes require:
- Strong empirical justification
- Migration guide for users
- Deprecation warnings for one release cycle

**Stable types:**
- `BiologicalDecaySystem`
- `DecayFunction`
- `DecayConfig`
- `DecayConfigBuilder`
- `DecayIntegration` trait

**Internal implementation** (may change):
- Individual subsystem implementations
- Calibration algorithms
- Consolidation heuristics
