# Task 007: Temporal Dynamics Documentation

## Objective
Create comprehensive documentation covering temporal decay architecture, API usage, configuration, and cognitive psychology foundations.

## Priority
P2 (important for adoption and maintainability)

## Effort Estimate
1 day

## Dependencies
- All previous Milestone 4 tasks (002-006)

## Technical Approach

### Files to Create
- `docs/temporal-dynamics.md` - Architecture and design
- `docs/decay-functions.md` - Detailed decay function documentation
- `docs/tutorials/temporal-configuration.md` - Configuration guide
- `engram-core/src/decay/README.md` - Module documentation

### Documentation Structure

**1. Architecture Overview** (`docs/temporal-dynamics.md`):
```markdown
# Temporal Dynamics in Engram

## Overview

Engram implements biologically-inspired temporal decay following cognitive psychology research. Memories naturally decay over time unless reinforced through retrieval, matching human forgetting patterns.

## Key Concepts

### Lazy Decay Evaluation

Unlike traditional databases with background vacuum processes, Engram computes decay **lazily during recall**:

- Decay is a view-time transformation, not storage mutation
- No background threads updating stored values
- Deterministic results for given query time
- Zero overhead when not querying

### Dual-System Architecture

Based on complementary learning systems theory:

**Hippocampal System** (Fast decay):
- New episodic memories
- Rapid initial forgetting
- Exponential decay: `R(t) = e^(-λt)`

**Neocortical System** (Slow decay):
- Consolidated semantic knowledge
- Gradual long-term forgetting
- Power-law decay: `R(t) = (1 + t)^(-α)`

**Consolidation Trigger**: Memories transition from hippocampal to neocortical after ≥3 retrievals.

## How It Works

```rust
// 1. Store memory (confidence = 0.9)
store.insert_episode(Episode::new("meeting notes")
    .with_confidence(Confidence::from_raw(0.9)));

// 2. Time passes... (7 days)

// 3. Recall applies decay lazily
let results = recall.recall(&cue, &store)?;
// -> confidence = 0.6 (decayed based on elapsed time)

// 4. Next access updates last_access
// -> Strengthens memory, slows future decay
```

## Biological Inspiration

Engram's decay models are validated against:
- **Ebbinghaus (1885)**: Exponential forgetting curve
- **Wickelgren (1974)**: Power-law decay for long-term memory
- **McClelland et al. (1995)**: Complementary learning systems

See `docs/decay-functions.md` for mathematical details and validation.

## Performance Characteristics

- Decay computation: <1ms p95 per memory
- No background threads or processes
- Zero overhead when decay disabled
- Lock-free concurrent access

## Design Decisions

**Why lazy evaluation?**
- Avoids write amplification from frequent updates
- Enables deterministic results for testing
- Supports time-travel queries (what was confidence at time T?)
- Simpler than background consolidation processes

**Why multiple decay functions?**
- Different memory types have different forgetting patterns
- Allows tuning for specific use cases
- Matches neuroscience evidence for dual systems

**Why track access_count?**
- Spaced repetition effect (SuperMemo, Anki)
- Triggers consolidation (hippocampal → neocortical)
- Personalizes decay rates
```

**2. Decay Functions Reference** (`docs/decay-functions.md`):
```markdown
# Decay Functions Reference

## Exponential Decay

**Formula**: `R(t) = e^(-λt)`

**When to use**: Short-term episodic memories (hours to days)

**Configuration**:
```rust
let config = DecayConfig::builder()
    .exponential(0.000012)  // λ rate parameter
    .build();
```

**Parameters**:
- `rate` (λ): Decay rate per second
  - 0.000012: Ebbinghaus curve (20% at 31 days)
  - 0.00001: Slower decay (25% at 31 days)
  - 0.00002: Faster decay (12% at 31 days)

**Psychological Basis**: Ebbinghaus (1885) forgetting curve

## Power-Law Decay

**Formula**: `R(t) = (1 + t)^(-α)`

**When to use**: Long-term semantic knowledge (weeks to years)

**Configuration**:
```rust
let config = DecayConfig::builder()
    .power_law(0.3)  // α exponent
    .build();
```

**Parameters**:
- `exponent` (α): Decay exponent
  - 0.2: Very slow decay (semantic facts)
  - 0.3: Moderate decay (skills, procedures)
  - 0.5: Faster decay (contextual knowledge)

**Psychological Basis**: Wickelgren (1974), Wixted & Ebbesen (1991)

**Key Difference from Exponential**: Power-law has slower long-term decay (long tail)

## Two-Component Decay

**Formula**:
```
R(t) = hippocampal_decay(t)  if access_count < 3
R(t) = neocortical_decay(t)  if access_count ≥ 3
```

**When to use**: General-purpose (recommended default)

**Configuration**:
```rust
let config = DecayConfig::builder()
    .two_component(0.0001, 0.00001)  // hippocampal_rate, neocortical_rate
    .build();
```

**Parameters**:
- `hippocampal_rate`: Fast decay for new memories
- `neocortical_rate`: Slow decay for consolidated memories
- `consolidation_threshold`: Access count to trigger consolidation (default: 3)

**Psychological Basis**: Complementary Learning Systems (McClelland et al. 1995)

**Consolidation Effect**: Memories accessed ≥3 times decay 10x slower

## Comparison

| Function | Best For | Decay Pattern | Complexity |
|----------|----------|---------------|------------|
| Exponential | Short-term episodic | Fast initial, exponential tail | Low |
| Power-Law | Long-term semantic | Slow long-tail | Low |
| Two-Component | General purpose | Adaptive based on usage | Medium |

## Performance

All decay functions compute in <100μs:

| Function | Computation Time |
|----------|------------------|
| Exponential | ~50μs |
| Power-Law | ~80μs (pow operation) |
| Two-Component | ~60μs (conditional exponential) |
```

**3. Configuration Tutorial** (`docs/tutorials/temporal-configuration.md`):
```markdown
# Configuring Temporal Decay

## Quick Start

```rust
use engram_core::{
    decay::{BiologicalDecaySystem, DecayConfig, DecayFunction},
    activation::CognitiveRecallBuilder,
};

// 1. Create decay system with default (two-component)
let decay_system = BiologicalDecaySystem::default();

// 2. Add to recall pipeline
let recall = CognitiveRecallBuilder::new()
    .vector_seeder(seeder)
    .spreading_engine(engine)
    .decay_system(Arc::new(decay_system))
    .build()?;

// 3. Recall automatically applies decay
let results = recall.recall(&cue, &store)?;
```

## System-Wide Configuration

```rust
// Exponential decay for all memories
let config = DecayConfig::builder()
    .exponential(0.000012)
    .enabled(true)
    .min_confidence(0.1)  // Forget below this threshold
    .build();

let decay_system = BiologicalDecaySystem::new(config);
```

## Per-Memory Configuration

```rust
// Override decay function for specific memory
let episode = Episode::new("critical information")
    .with_decay_function(DecayFunction::PowerLaw { exponent: 0.2 });  // Very slow decay

store.insert_episode(episode);
```

## Common Scenarios

### Scenario 1: Personal Knowledge Base
```rust
// Slow decay for semantic knowledge
let config = DecayConfig::builder()
    .power_law(0.2)  // Long-term retention
    .build();
```

### Scenario 2: Chat History
```rust
// Fast decay for ephemeral conversations
let config = DecayConfig::builder()
    .exponential(0.0001)  // Forget within weeks
    .min_confidence(0.2)
    .build();
```

### Scenario 3: Spaced Repetition System
```rust
// Two-component for adaptive learning
let config = DecayConfig::builder()
    .two_component(0.0001, 0.00001)
    .build();

// Frequently reviewed items consolidate and decay slower
```

### Scenario 4: Disable Decay
```rust
let config = DecayConfig::builder()
    .enabled(false)
    .build();
```

## Monitoring Decay

```rust
// Check current confidence with decay applied
let results = recall.recall(&cue, &store)?;
for result in results {
    println!("Confidence: {} (with decay)", result.confidence.raw());
    println!("Last accessed: {}", result.episode.last_access);
    println!("Access count: {}", result.episode.access_count);
}
```

## Performance Tuning

```rust
// Reduce computation if latency is critical
let config = DecayConfig::builder()
    .exponential(0.000012)  // Faster than power-law
    .min_confidence(0.3)     // Filter aggressively
    .build();
```
```

**4. Module Documentation** (`engram-core/src/decay/README.md`):
```markdown
# Decay Module

Temporal decay functions for cognitive memory dynamics.

## Architecture

```
decay/
├── mod.rs              - BiologicalDecaySystem, DecayConfig
├── functions.rs        - DecayFunction enum and implementations
├── hippocampal.rs      - Fast exponential decay
├── neocortical.rs      - Slow power-law decay
├── two_component.rs    - Dual-system model
└── validation.rs       - Psychology curve validation
```

## Key Types

- `BiologicalDecaySystem`: Main decay orchestrator
- `DecayFunction`: Enum of decay models
- `DecayConfig`: Configuration builder
- `DecayIntegration`: Trait for applying decay to memories

## Integration Points

- `CognitiveRecall`: Applies decay during ranking
- `Memory`/`Episode`: Store `last_access` and `decay_function`
- `MemoryStore`: Provides decay system to recall

## Testing

Run decay tests:
```bash
cargo test --test forgetting_curves_validation
cargo test --test temporal_integration_test
cargo bench temporal_performance
```

## References

See `docs/decay-functions.md` for psychological foundations.
```

## Acceptance Criteria

- [ ] Architecture documentation explains lazy decay and dual-system design
- [ ] Decay functions documented with formulas, parameters, and use cases
- [ ] Tutorial covers common configuration scenarios with code examples
- [ ] Module README provides navigation and integration points
- [ ] All public APIs have rustdoc comments
- [ ] Documentation includes performance characteristics
- [ ] References to psychology literature included
- [ ] Examples compile and run correctly

## Testing Approach

**Documentation Testing**:
```bash
# Ensure all code examples compile
cargo test --doc

# Check for broken links
mdbook test docs/

# Validate rustdoc builds
cargo doc --no-deps
```

**Review Checklist**:
- [ ] Can a new developer understand temporal decay from docs alone?
- [ ] Are all configuration options explained?
- [ ] Do examples cover common use cases?
- [ ] Is performance guidance clear?
- [ ] Are limitations and trade-offs documented?

## Risk Mitigation

**Risk**: Documentation becomes stale as code evolves
**Mitigation**: Use `cargo test --doc` in CI to ensure examples compile. Link docs to tests.

**Risk**: Too much theory, not enough practical guidance
**Mitigation**: Lead with quick start and common scenarios. Put theory in separate reference docs.

**Risk**: Cognitive psychology jargon confuses developers
**Mitigation**: Define terms clearly. Use analogies to familiar concepts (caching, TTL).

## Notes

Good documentation is critical for temporal dynamics adoption. Developers need to understand:
1. Why decay matters (biological inspiration)
2. How to configure it (practical examples)
3. What trade-offs exist (performance, accuracy)

**Writing Style**: Clear, concise, example-driven. Assume reader is familiar with Rust but not cognitive psychology.
