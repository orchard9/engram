# Task 004: Decay Configuration API

## Objective
Create API for configuring decay functions per memory and system-wide, supporting exponential, power-law, and two-component decay models.

## Priority
P1 (important - required for full Milestone 4 feature set)

## Effort Estimate
1.5 days

## Dependencies
- Task 002: Last Access Tracking
- Task 003: Lazy Decay Integration

## Technical Approach

### Files to Modify
- `engram-core/src/decay/mod.rs` - Add configuration builder
- `engram-core/src/memory.rs` - Add decay configuration to Memory/Episode
- `engram-core/src/store.rs` - Support per-memory decay configuration

### Design

**Decay Configuration Type**:
```rust
// engram-core/src/decay/mod.rs
#[derive(Debug, Clone, Copy)]
pub enum DecayFunction {
    /// Exponential decay: C(t) = C₀ * e^(-λt)
    Exponential { rate: f32 },
    /// Power-law decay: C(t) = C₀ * (1 + t)^(-α)
    PowerLaw { exponent: f32 },
    /// Two-component: hippocampal + neocortical
    TwoComponent {
        hippocampal_rate: f32,
        neocortical_rate: f32,
        consolidation_threshold: f32,
    },
}

impl Default for DecayFunction {
    fn default() -> Self {
        Self::TwoComponent {
            hippocampal_rate: 0.1,  // Fast decay
            neocortical_rate: 0.01, // Slow decay
            consolidation_threshold: 0.7,
        }
    }
}

#[derive(Debug, Clone)]
pub struct DecayConfig {
    /// Default decay function for new memories
    pub default_function: DecayFunction,
    /// Enable automatic decay during recall
    pub enabled: bool,
    /// Minimum confidence threshold (memories below this are forgotten)
    pub min_confidence: f32,
}

impl Default for DecayConfig {
    fn default() -> Self {
        Self {
            default_function: DecayFunction::default(),
            enabled: true,
            min_confidence: 0.1,
        }
    }
}
```

**Memory-Level Decay Configuration**:
```rust
// engram-core/src/memory.rs
pub struct Episode {
    // ... existing fields ...
    pub decay_function: Option<DecayFunction>,  // None = use system default
}

impl Episode {
    pub fn with_decay_function(mut self, function: DecayFunction) -> Self {
        self.decay_function = Some(function);
        self
    }
}
```

**Builder Pattern for Configuration**:
```rust
pub struct DecayConfigBuilder {
    config: DecayConfig,
}

impl DecayConfigBuilder {
    pub fn new() -> Self {
        Self {
            config: DecayConfig::default(),
        }
    }

    pub fn exponential(mut self, rate: f32) -> Self {
        self.config.default_function = DecayFunction::Exponential { rate };
        self
    }

    pub fn power_law(mut self, exponent: f32) -> Self {
        self.config.default_function = DecayFunction::PowerLaw { exponent };
        self
    }

    pub fn two_component(mut self, hippocampal_rate: f32, neocortical_rate: f32) -> Self {
        self.config.default_function = DecayFunction::TwoComponent {
            hippocampal_rate,
            neocortical_rate,
            consolidation_threshold: 0.7,
        };
        self
    }

    pub fn enabled(mut self, enabled: bool) -> Self {
        self.config.enabled = enabled;
        self
    }

    pub fn build(self) -> DecayConfig {
        self.config
    }
}
```

**Decay Function Evaluation**:
```rust
impl DecayFunction {
    pub fn compute_decay(&self, elapsed_time: Duration, access_count: u64) -> f32 {
        let t = elapsed_time.as_secs_f32();

        match self {
            DecayFunction::Exponential { rate } => {
                (-rate * t).exp()
            }
            DecayFunction::PowerLaw { exponent } => {
                (1.0 + t).powf(-exponent)
            }
            DecayFunction::TwoComponent {
                hippocampal_rate,
                neocortical_rate,
                consolidation_threshold
            } => {
                // Use consolidation_threshold and access_count to determine
                // which decay rate to use
                if access_count >= 3 {
                    // Consolidated to neocortical
                    (-neocortical_rate * t).exp()
                } else {
                    // Still in hippocampal
                    (-hippocampal_rate * t).exp()
                }
            }
        }
    }
}
```

## Acceptance Criteria

- [ ] `DecayFunction` enum supports exponential, power-law, two-component
- [ ] `DecayConfig` allows system-wide defaults
- [ ] `Episode` can override system decay function
- [ ] Builder pattern for ergonomic configuration
- [ ] Decay functions mathematically correct (unit tested)
- [ ] Configuration serializable to JSON/TOML
- [ ] Documentation with examples for each decay function
- [ ] Performance: decay computation <100μs per memory

## Testing Approach

**Unit Tests**:
```rust
#[test]
fn test_exponential_decay_correct() {
    let func = DecayFunction::Exponential { rate: 0.1 };
    let decay = func.compute_decay(Duration::from_secs(10), 0);
    let expected = (-0.1 * 10.0_f32).exp();
    assert!((decay - expected).abs() < 1e-6);
}

#[test]
fn test_power_law_decay_correct() {
    let func = DecayFunction::PowerLaw { exponent: 0.5 };
    let decay = func.compute_decay(Duration::from_secs(100), 0);
    let expected = (1.0 + 100.0_f32).powf(-0.5);
    assert!((decay - expected).abs() < 1e-6);
}

#[test]
fn test_two_component_switches_to_neocortical() {
    let func = DecayFunction::TwoComponent {
        hippocampal_rate: 0.1,
        neocortical_rate: 0.01,
        consolidation_threshold: 0.7,
    };

    // With low access_count, uses hippocampal
    let decay1 = func.compute_decay(Duration::from_secs(10), 1);

    // With high access_count, uses neocortical
    let decay2 = func.compute_decay(Duration::from_secs(10), 5);

    assert!(decay2 > decay1, "Neocortical should decay slower");
}

#[test]
fn test_per_memory_decay_override() {
    let episode = Episode::new("test")
        .with_decay_function(DecayFunction::PowerLaw { exponent: 0.8 });

    assert!(episode.decay_function.is_some());
}
```

**Documentation Example**:
```rust
/// Configure exponential decay
let config = DecayConfigBuilder::new()
    .exponential(0.05)  // 5% decay per time unit
    .enabled(true)
    .build();

/// Configure power-law (slower decay over time)
let config = DecayConfigBuilder::new()
    .power_law(0.3)
    .build();

/// Create episode with custom decay
let episode = Episode::new("important meeting notes")
    .with_decay_function(DecayFunction::PowerLaw { exponent: 0.2 });  // Very slow decay
```

## Risk Mitigation

**Risk**: Complex configuration options confuse users
**Mitigation**: Provide sensible defaults. Include examples for common use cases. Document when to use each decay function.

**Risk**: Per-memory decay configuration causes storage bloat
**Mitigation**: Store decay config as enum (8 bytes). Most memories use system default (stored as None).

**Risk**: Decay functions not validated against psychology
**Mitigation**: Next task (005) validates against Ebbinghaus curves. This task focuses on API correctness.

## Notes

This task provides the user-facing API for temporal decay. The three decay functions map to psychological models:
- **Exponential**: Classic Ebbinghaus forgetting curve
- **Power-law**: Long-tail forgetting (Wickelgren model)
- **Two-component**: Hippocampal-neocortical dual system (complementary learning systems)

**Design Principle**: Sensible defaults for most users, fine-grained control for power users.
