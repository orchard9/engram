# Temporal Decay Configuration Tutorial

Complete guide to configuring temporal decay in Engram for different use cases.

## Table of Contents

1. [Quick Start](#quick-start)
2. [System-Wide Configuration](#system-wide-configuration)
3. [Per-Memory Configuration](#per-memory-configuration)
4. [Common Use Cases](#common-use-cases)
5. [Advanced Configuration](#advanced-configuration)
6. [Monitoring Decay](#monitoring-decay)
7. [Performance Tuning](#performance-tuning)
8. [Troubleshooting](#troubleshooting)

---

## Quick Start

### Default Configuration (Recommended)

The simplest approach is to use the default configuration, which provides a two-component decay model suitable for most applications:

```rust
use engram_core::{
    activation::CognitiveRecallBuilder,
    decay::BiologicalDecaySystem,
};
use std::sync::Arc;

// Use default configuration (two-component with threshold=3)
let decay_system = Arc::new(BiologicalDecaySystem::default());

// Attach to recall pipeline
let recall = CognitiveRecallBuilder::new()
    .vector_seeder(seeder)
    .spreading_engine(engine)
    .decay_system(decay_system)
    .build()?;
```

This gives you:

- Hippocampal decay (fast) for new memories (access_count < 3)
- Neocortical decay (slow) for consolidated memories (access_count >= 3)
- Minimum confidence threshold of 0.1
- Enabled by default

### Disable Decay Entirely

If you don't want temporal decay:

```rust
use engram_core::decay::{BiologicalDecaySystem, DecayConfigBuilder};

let config = DecayConfigBuilder::new()
    .enabled(false)
    .build();
let decay_system = Arc::new(BiologicalDecaySystem::with_config(config));
```

---

## System-Wide Configuration

### Exponential Decay (Short-Term Memory)

Best for ephemeral data like chat logs, sessions, or temporary notes:

```rust
use engram_core::decay::DecayConfigBuilder;

let config = DecayConfigBuilder::new()
    .exponential(2.0)  // tau = 2 hours
    .min_confidence(0.15)
    .enabled(true)
    .build();

let decay_system = Arc::new(BiologicalDecaySystem::with_config(config));
```

**Configuration parameters:**

- `tau_hours`: Time constant in hours (smaller = faster decay)
  - `0.5`: Very fast (for temporary UI state)
  - `1.96`: Standard Ebbinghaus (for general episodic memory)
  - `10.0`: Slow exponential (for important short-term data)

**Memory retention examples (tau = 2.0):**

- After 1 hour: ~60% retention
- After 2 hours: ~37% retention (one tau)
- After 6 hours: ~5% retention
- After 24 hours: <1% retention

### Power-Law Decay (Long-Term Knowledge)

Best for permanent knowledge bases, skills, or documentation:

```rust
let config = DecayConfigBuilder::new()
    .power_law(0.15)  // beta = 0.15 (very slow decay)
    .min_confidence(0.1)
    .enabled(true)
    .build();

let decay_system = Arc::new(BiologicalDecaySystem::with_config(config));
```

**Configuration parameters:**

- `beta`: Decay exponent (smaller = slower decay)
  - `0.1`: Permastore memories (extremely slow)
  - `0.18`: Bahrick permastore (default, validated against 50-year data)
  - `0.3`: Moderate long-term decay
  - `0.5`: Fast power-law (still slower tail than exponential)

**Memory retention examples (beta = 0.15):**

- After 1 day: ~77% retention
- After 1 week: ~53% retention
- After 1 month: ~37% retention
- After 1 year: ~17% retention

### Two-Component Decay (Adaptive Learning)

Best for spaced repetition systems, adaptive learning, or general-purpose applications:

```rust
let config = DecayConfigBuilder::new()
    .two_component(3)  // Consolidation threshold = 3 accesses
    .enabled(true)
    .build();

let decay_system = Arc::new(BiologicalDecaySystem::with_config(config));
```

**Configuration parameters:**

- `consolidation_threshold`: Number of accesses before switching to slow decay
  - `2`: Early consolidation (quick learning)
  - `3`: Balanced (recommended default, matches testing effect research)
  - `5`: Late consolidation (more reinforcement required)
  - `10`: Very conservative (only well-practiced memories get slow decay)

**Behavior:**

- `access_count < threshold`: Uses hippocampal decay (exponential, tau=1h)
- `access_count >= threshold`: Uses neocortical decay (exponential, tau=10h)

**Example with threshold=3:**

```rust
// Memory accessed once (unconsolidated)
// 6 hours later: ~0.25% retention (hippocampal, fast decay)

// Memory accessed 5 times (consolidated)
// 6 hours later: ~54.9% retention (neocortical, slow decay)
// That's 218x more retention!
```

### Hybrid Decay (Best Empirical Fit)

Best for academic/research applications requiring mathematical accuracy:

```rust
let config = DecayConfigBuilder::new()
    .hybrid(
        0.8,      // short_term_tau (hours)
        0.25,     // long_term_beta
        86400,    // transition_point (seconds) = 24 hours
    )
    .enabled(true)
    .build();

let decay_system = Arc::new(BiologicalDecaySystem::with_config(config));
```

**Configuration parameters:**

- `short_term_tau`: Exponential tau before transition (typical: 0.5-2.0 hours)
- `long_term_beta`: Power-law beta after transition (typical: 0.15-0.35)
- `transition_point`: When to switch (typical: 3600s=1h, 86400s=24h, 604800s=1week)

**Behavior:**

- Before transition: Exponential decay (fast initial drop)
- After transition: Power-law decay (slow long-term forgetting)

**Example with default parameters:**

- At 2 hours (<24h): Exponential gives ~8.2% retention
- At 48 hours (>24h): Power-law gives ~46.1% retention

---

## Per-Memory Configuration

You can override the system default for individual memories by setting the `decay_function` field on an Episode:

```rust
use engram_core::{Episode, Confidence, decay::DecayFunction};
use chrono::Utc;

// Create episode with custom decay function
let mut episode = Episode::new(
    "critical-data-1".to_string(),
    Utc::now(),
    "Critical system configuration".to_string(),
    embedding,
    Confidence::exact(0.95),
);

// Override: use very slow power-law decay for this critical memory
episode.decay_function = Some(DecayFunction::PowerLaw { beta: 0.1 });

store.store(episode);
```

**Common override patterns:**

1. **Critical memories** (never forget):

```rust
episode.decay_function = Some(DecayFunction::PowerLaw { beta: 0.05 });
```

2. **Temporary cache** (forget quickly):

```rust
episode.decay_function = Some(DecayFunction::Exponential { tau_hours: 0.5 });
```

3. **Disable decay for specific memory**:

```rust
// System has decay enabled, but this memory won't decay
episode.decay_function = Some(DecayFunction::Disabled);
```

---

## Common Use Cases

### Use Case 1: Personal Knowledge Base

**Requirements:**

- Long-term retention (years)
- Slow forgetting of facts
- Frequently accessed items should be very stable

**Configuration:**

```rust
let config = DecayConfigBuilder::new()
    .power_law(0.15)  // Very slow decay (Bahrick permastore)
    .min_confidence(0.2)  // Higher minimum (keep quality memories)
    .enabled(true)
    .build();
```

**Why this works:**

- Power-law decay matches human semantic memory
- Beta=0.15 gives excellent long-term retention
- Higher min_confidence ensures only quality information remains

### Use Case 2: Chat History / Conversation Logs

**Requirements:**

- Short-term retention (hours to days)
- Fast forgetting of old conversations
- Recent chats should be easily accessible

**Configuration:**

```rust
let config = DecayConfigBuilder::new()
    .exponential(2.0)  // 2 hour tau (moderate speed)
    .min_confidence(0.1)  // Low minimum (keep older context if needed)
    .enabled(true)
    .build();
```

**Why this works:**

- Exponential decay matches episodic memory
- Tau=2h gives ~37% retention after 2 hours, ~5% after 6 hours
- Recent messages stay highly confident, old ones fade quickly

### Use Case 3: Spaced Repetition Flashcards

**Requirements:**

- Adaptive to user performance
- Well-practiced cards should be stable
- New cards should be tested frequently

**Configuration:**

```rust
let config = DecayConfigBuilder::new()
    .two_component(3)  // Consolidate after 3 correct recalls
    .enabled(true)
    .build();
```

**Why this works:**

- Two-component models the testing effect from cognitive psychology
- New cards (access_count < 3) decay quickly → shown frequently
- Mastered cards (access_count >= 3) decay slowly → shown rarely
- Matches SuperMemo algorithm principles

### Use Case 4: Temporary Session Cache

**Requirements:**

- Very fast forgetting (minutes)
- No long-term retention needed
- Reduce memory footprint quickly

**Configuration:**

```rust
let config = DecayConfigBuilder::new()
    .exponential(0.25)  // 15 minute tau (very fast)
    .min_confidence(0.05)  // Very low minimum
    .enabled(true)
    .build();
```

**Why this works:**

- Tau=0.25 hours (15 minutes) gives rapid decay
- After 30 minutes: ~13.5% retention
- After 1 hour: ~1.8% retention
- After 2 hours: ~0.03% retention (effectively forgotten)

### Use Case 5: Critical Long-Term Documentation

**Requirements:**

- Never forget important docs
- Extremely slow decay
- High confidence retention

**System-wide configuration:**

```rust
let config = DecayConfigBuilder::new()
    .power_law(0.05)  // Extremely slow decay
    .min_confidence(0.3)
    .enabled(true)
    .build();
```

**Or use per-memory override for critical docs:**

```rust
// System uses default, but critical docs get special treatment
doc_episode.decay_function = Some(DecayFunction::PowerLaw { beta: 0.05 });
```

### Use Case 6: Mixed Content (General Purpose)

**Requirements:**

- Handle both short-term and long-term memories
- Adapt based on usage patterns
- Good default for unknown memory types

**Configuration:**

```rust
// Use the default (two-component)
let decay_system = Arc::new(BiologicalDecaySystem::default());

// Or explicitly:
let config = DecayConfigBuilder::new()
    .two_component(3)
    .min_confidence(0.1)
    .enabled(true)
    .build();
```

**Why this works:**

- Automatically adapts to usage patterns
- New/rarely-used memories decay quickly (don't clutter the system)
- Frequently-accessed memories become stable (important content persists)
- Best general-purpose configuration

---

## Advanced Configuration

### Minimum Confidence Threshold

The `min_confidence` parameter prevents complete forgetting:

```rust
let config = DecayConfigBuilder::new()
    .exponential(1.0)
    .min_confidence(0.2)  // Never go below 20% confidence
    .build();
```

**Choosing min_confidence:**

- `0.05`: Allow very low confidence (keeps more memories, lower quality)
- `0.1`: Default (good balance)
- `0.2`: Higher quality threshold (fewer but more confident memories)
- `0.3`: Very strict (only high-confidence memories remain)

**Use cases:**

- Low (0.05-0.1): Large knowledge bases, exploratory systems
- Medium (0.1-0.2): General applications
- High (0.2-0.3): Critical systems, high-precision requirements

### Individual Differences Calibration

The decay system automatically applies individual differences calibration to account for natural variation in memory performance. This is applied transparently during decay computation.

**Effect:**

- Adds realistic variance to retention (~10-15% variation)
- Prevents overfitting to exact mathematical curves
- Models human memory variability

**No configuration needed** - this happens automatically.

### Combining System and Per-Memory Configuration

You can use a system default and override specific memories:

```rust
// System default: moderate exponential decay
let config = DecayConfigBuilder::new()
    .exponential(2.0)
    .build();
let decay_system = Arc::new(BiologicalDecaySystem::with_config(config));

// Most memories use system default
let normal_episode = Episode::new(...);
store.store(normal_episode);

// But critical memories get special treatment
let mut critical_episode = Episode::new(...);
critical_episode.decay_function = Some(DecayFunction::PowerLaw { beta: 0.1 });
store.store(critical_episode);

// And temporary data gets fast decay
let mut temp_episode = Episode::new(...);
temp_episode.decay_function = Some(DecayFunction::Exponential { tau_hours: 0.5 });
store.store(temp_episode);
```

---

## Monitoring Decay

### Check Current Confidence

To see how much a memory has decayed:

```rust
use std::time::Duration;

let episode = store.get_episode(&episode_id)?;
let elapsed = Utc::now() - episode.last_recall;

let current_confidence = decay_system.compute_decayed_confidence(
    episode.encoding_confidence,
    elapsed.to_std()?,
    episode.recall_count,
    episode.when,
    episode.decay_function,
);

println!("Original: {:.3}", episode.encoding_confidence.raw());
println!("Current:  {:.3}", current_confidence.raw());
println!("Decay:    {:.1}%", (1.0 - current_confidence.raw() / episode.encoding_confidence.raw()) * 100.0);
```

### Monitoring Access Patterns

For two-component decay, track consolidation status:

```rust
let episode = store.get_episode(&episode_id)?;

if let Some(DecayFunction::TwoComponent { consolidation_threshold }) =
    episode.decay_function.or(Some(decay_system.config.default_function))
{
    if episode.recall_count < consolidation_threshold {
        println!("Status: Unconsolidated (hippocampal decay)");
        println!("Accesses until consolidation: {}",
                 consolidation_threshold - episode.recall_count);
    } else {
        println!("Status: Consolidated (neocortical decay)");
        println!("Stable long-term memory");
    }
}
```

### Batch Decay Analysis

Analyze decay across all memories:

```rust
let all_episodes = store.get_all_episodes();
let now = Utc::now();

for episode in all_episodes {
    let elapsed = (now - episode.last_recall).to_std()?;
    let current_conf = decay_system.compute_decayed_confidence(
        episode.encoding_confidence,
        elapsed,
        episode.recall_count,
        episode.when,
        episode.decay_function,
    );

    let decay_pct = (1.0 - current_conf.raw() / episode.encoding_confidence.raw()) * 100.0;

    if decay_pct > 50.0 {
        println!("Memory {} has decayed {:.1}% (consider refreshing)",
                 episode.id, decay_pct);
    }
}
```

---

## Performance Tuning

### Decay Computation Cost

All decay functions are designed to be fast (<100μs per memory):

| Function | Mean Time | Notes |
|----------|-----------|-------|
| Disabled | ~2μs | Early return, no computation |
| Exponential | ~48μs | One `exp()` call |
| Two-Component | ~56μs | One comparison + `exp()` |
| Hybrid | ~64μs | One comparison + `exp()` or `pow()` |
| Power-Law | ~82μs | One `pow()` call (slightly slower) |

**Recommendation:** All functions meet performance targets. Choose based on memory behavior, not performance.

### Disabling Decay for Performance

If you need maximum performance and don't want decay:

```rust
let config = DecayConfigBuilder::new()
    .enabled(false)
    .build();
```

This gives ~2μs overhead (early return check only).

### Batch Processing

When processing many memories, the cost is linear:

- 100 memories: ~5-8ms
- 1,000 memories: ~50-80ms
- 10,000 memories: ~500-800ms

If you need to process very large batches:

1. **Filter before decay**: Only compute decay for top-K results
2. **Parallelize**: Decay computation is thread-safe and parallelizable
3. **Cache recent results**: If querying same memories repeatedly

### Thread Safety

`BiologicalDecaySystem` is thread-safe and can be shared across threads:

```rust
let decay_system = Arc::new(BiologicalDecaySystem::default());

// Clone Arc and use in multiple threads
let system1 = Arc::clone(&decay_system);
let system2 = Arc::clone(&decay_system);

thread::spawn(move || {
    // Use system1 in thread 1
});

thread::spawn(move || {
    // Use system2 in thread 2
});
```

No locks, no contention - pure computation.

---

## Troubleshooting

### Problem: Memories Decaying Too Quickly

**Symptoms:**

- Important memories disappear after a few hours
- Confidence drops too fast

**Solutions:**

1. **Increase tau for exponential decay:**

```rust
let config = DecayConfigBuilder::new()
    .exponential(5.0)  // Increase from default 1.96 to 5.0
    .build();
```

2. **Switch to power-law decay:**

```rust
let config = DecayConfigBuilder::new()
    .power_law(0.18)  // Much slower long-term decay
    .build();
```

3. **Use per-memory override for critical data:**

```rust
episode.decay_function = Some(DecayFunction::PowerLaw { beta: 0.1 });
```

### Problem: Memories Not Decaying At All

**Symptoms:**

- Old, unused memories remain at high confidence
- System doesn't forget anything

**Solutions:**

1. **Check if decay is enabled:**

```rust
// Make sure enabled=true
let config = DecayConfigBuilder::new()
    .exponential(1.0)
    .enabled(true)  // Check this!
    .build();
```

2. **Verify last_recall is being updated:**

```rust
// After retrieval, update last_recall
episode.last_recall = Utc::now();
store.store(episode);
```

3. **Check min_confidence isn't too high:**

```rust
// If min_confidence=0.9, decay can't reduce confidence much
let config = DecayConfigBuilder::new()
    .exponential(1.0)
    .min_confidence(0.1)  // Lower this if needed
    .build();
```

### Problem: Inconsistent Decay Behavior

**Symptoms:**

- Two similar memories decay differently
- Decay seems random

**Solutions:**

1. **Check for per-memory overrides:**

```rust
// One memory might have override, other uses system default
if let Some(override_fn) = episode.decay_function {
    println!("Memory has override: {:?}", override_fn);
} else {
    println!("Memory uses system default");
}
```

2. **Individual differences calibration:**
   - Decay includes realistic variance (~10-15%)
   - This is expected and models human memory
   - If you need deterministic behavior for testing, use fixed seed

3. **Access count differences (two-component):**

```rust
// Check recall_count - this affects two-component decay
println!("Memory A accesses: {}", episode_a.recall_count);
println!("Memory B accesses: {}", episode_b.recall_count);

// If A has recall_count=2 and B has recall_count=4 with threshold=3,
// they use different decay functions!
```

### Problem: Performance Issues

**Symptoms:**

- Slow query times
- High CPU usage during recall

**Solutions:**

1. **Profile decay computation:**

```rust
use std::time::Instant;

let start = Instant::now();
let result = decay_system.compute_decayed_confidence(...);
let duration = start.elapsed();

if duration.as_micros() > 100 {
    println!("Warning: Decay took {}μs (expected <100μs)", duration.as_micros());
}
```

2. **Consider disabling decay temporarily:**

```rust
// For performance testing, disable decay
let config = DecayConfigBuilder::new()
    .enabled(false)
    .build();
```

3. **Verify you're not recomputing unnecessarily:**

```rust
// BAD: Recomputing for same memory multiple times
for _ in 0..100 {
    let conf = decay_system.compute_decayed_confidence(...);  // Same memory!
}

// GOOD: Compute once, reuse
let conf = decay_system.compute_decayed_confidence(...);
for _ in 0..100 {
    use_confidence(conf);  // Reuse computed value
}
```

### Problem: Testing Effect Not Working (Two-Component)

**Symptoms:**

- Memories don't consolidate even after many accesses
- All memories decay quickly

**Solutions:**

1. **Verify recall_count is being incremented:**

```rust
// After each retrieval
episode.recall_count += 1;
episode.last_recall = Utc::now();
store.store(episode);
```

2. **Check consolidation threshold:**

```rust
// Make sure threshold is reasonable
let config = DecayConfigBuilder::new()
    .two_component(3)  // Not too high (e.g., not 100)
    .build();
```

3. **Verify you're using two-component:**

```rust
// Check config has TwoComponent
match decay_system.config.default_function {
    DecayFunction::TwoComponent { consolidation_threshold } => {
        println!("Using two-component with threshold {}", consolidation_threshold);
    }
    other => {
        println!("Warning: Not using two-component! Using {:?}", other);
    }
}
```

---

## See Also

- [Decay Functions Reference](../decay-functions.md) - Mathematical specifications
- [Temporal Dynamics Architecture](../temporal-dynamics.md) - Design principles
- [Module README](../../engram-core/src/decay/README.md) - Code organization
