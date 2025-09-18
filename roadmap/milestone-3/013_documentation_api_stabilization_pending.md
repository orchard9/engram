# Task 013: Documentation and API Stabilization

## Objective
Document the spreading API and create examples demonstrating cognitive recall patterns.

## Priority
P2 (Quality Enhancement)

## Effort Estimate
0.5 days

## Dependencies
- Task 012: Production Integration and Monitoring

## Technical Approach

### Implementation Details
- Document spreading configuration parameters and cognitive implications
- Create spreading visualization tools for debugging
- Write spreading performance tuning guide
- Add spreading examples to existing test suite

### Files to Create/Modify
- `docs/spreading_activation.md` - Comprehensive spreading documentation
- `examples/cognitive_recall_patterns.rs` - Example usage patterns
- `docs/performance_tuning.md` - Performance optimization guide
- `tools/spreading_visualizer.rs` - Debug visualization tool

### Integration Points
- Extends existing documentation structure
- Integrates with example systems
- Connects to monitoring and debugging tools
- Uses configuration management patterns

## Implementation Details

### API Documentation
```rust
/// Cognitive recall using activation spreading through memory networks.
///
/// Unlike traditional similarity search that returns statically ranked results,
/// cognitive recall spreads activation through memory associations to simulate
/// how human memory retrieval works with context and priming effects.
///
/// # Examples
///
/// Basic spreading activation:
/// ```rust
/// use engram_core::{MemoryStore, Cue, SpreadingConfig};
///
/// let config = SpreadingConfig {
///     max_hop_count: 3,
///     activation_threshold: 0.01,
///     time_budget: Duration::from_millis(10),
///     ..Default::default()
/// };
///
/// let memory_store = MemoryStore::with_spreading_config(config);
/// let cue = Cue::from_text("artificial intelligence");
/// let results = memory_store.recall(&cue).await?;
///
/// // Results include both directly similar memories and
/// // associatively related memories discovered through spreading
/// for (episode, confidence) in results {
///     println!("Memory: {}, Confidence: {:.3}", episode.content, confidence);
/// }
/// ```
///
/// # Cognitive Principles
///
/// The spreading activation implementation follows established cognitive science:
///
/// - **Semantic Priming**: Related concepts receive higher activation
/// - **Fan Effect**: Highly connected nodes spread less activation per connection
/// - **Decay Functions**: Activation decreases with distance (hop count)
/// - **Confidence Propagation**: Uncertainty increases through spreading chains
///
/// # Performance Characteristics
///
/// - **Latency Target**: <10ms P95 for single-hop activation
/// - **Scalability**: Linear with CPU cores up to 32 cores
/// - **Memory Overhead**: <20% additional compared to similarity search
/// - **Tier Awareness**: Automatically prioritizes hot tier over cold tier
///
impl MemoryStore {
    /// Perform cognitive recall using activation spreading.
    ///
    /// # Arguments
    ///
    /// * `cue` - The retrieval cue (text, embedding, or structured query)
    ///
    /// # Returns
    ///
    /// Vector of (Episode, Confidence) tuples ranked by final activation level.
    /// Confidence scores reflect both similarity and spreading path reliability.
    ///
    /// # Errors
    ///
    /// Returns `EnggramError::SpreadingFailed` if activation spreading encounters
    /// unrecoverable errors (e.g., graph corruption, tier unavailability).
    /// Falls back to similarity search when possible.
    pub async fn recall(&self, cue: &Cue) -> Result<Vec<(Episode, Confidence)>, EnggramError> {
        // Implementation details...
    }
}
```

### Configuration Documentation
```rust
/// Configuration parameters for cognitive spreading activation.
///
/// These parameters control how activation spreads through memory networks
/// and directly impact both cognitive realism and computational performance.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpreadingConfig {
    /// Maximum number of hops activation can traverse.
    ///
    /// Higher values enable deeper exploration but increase latency.
    /// Cognitive research suggests 2-4 hops optimal for human-like behavior.
    ///
    /// Default: 3
    /// Range: 1-10
    /// Performance impact: O(branching_factor^max_hop_count)
    pub max_hop_count: u16,

    /// Minimum activation level to continue spreading.
    ///
    /// Lower values enable more extensive exploration but may include
    /// spurious associations. Higher values focus on strong associations.
    ///
    /// Default: 0.01 (1% activation)
    /// Range: 0.001-0.1
    /// Cognitive basis: Matches human recognition thresholds
    pub activation_threshold: f32,

    /// Maximum time budget for spreading operation.
    ///
    /// Hard timeout preventing spreading from exceeding latency targets.
    /// When exceeded, returns partial results from completed spreading.
    ///
    /// Default: 10ms
    /// Range: 1ms-100ms
    /// Production requirement: <10ms P95 latency
    pub time_budget: Duration,

    /// Decay rate for activation strength per hop.
    ///
    /// Controls how rapidly activation decreases with distance.
    /// Lower values preserve distant associations, higher values focus locally.
    ///
    /// Default: 0.7 (30% decay per hop)
    /// Range: 0.1-0.9
    /// Cognitive basis: Matches semantic distance effects
    pub decay_rate: f32,

    /// Whether to enable tier-aware spreading prioritization.
    ///
    /// When enabled, hot tier memories processed before warm/cold tiers.
    /// Improves latency but may miss some distant associations.
    ///
    /// Default: true
    /// Recommended: true for production, false for research
    pub tier_aware_prioritization: bool,
}
```

### Example Usage Patterns
```rust
//! # Cognitive Recall Pattern Examples
//!
//! This module demonstrates various cognitive recall patterns achievable
//! with Engram's spreading activation engine.

use engram_core::*;

/// Demonstrates semantic priming effect in recall.
///
/// Shows how priming with related concepts improves recall of target memories.
#[tokio::main]
async fn semantic_priming_example() -> Result<(), Box<dyn std::error::Error>> {
    let memory_store = MemoryStore::new().await?;

    // Store related memories
    memory_store.store(Episode::new("Doctors treat patients in hospitals")).await?;
    memory_store.store(Episode::new("Nurses assist doctors with medical care")).await?;
    memory_store.store(Episode::new("Bread is baked in ovens")).await?;

    // Prime with medical concept
    let medical_cue = Cue::from_text("doctor");
    let results = memory_store.recall(&medical_cue).await?;

    println!("Medical priming results:");
    for (episode, confidence) in results {
        println!("  {:.3}: {}", confidence, episode.content);
    }

    // "Nurse" should rank higher than "Bread" due to semantic relation
    Ok(())
}

/// Demonstrates episodic memory reconstruction.
///
/// Shows how partial cues can retrieve complete episodic memories
/// through spreading activation.
#[tokio::main]
async fn episodic_reconstruction_example() -> Result<(), Box<dyn std::error::Error>> {
    let memory_store = MemoryStore::new().await?;

    // Store episodic memory with multiple components
    let episode = Episode::builder()
        .content("I met Sarah at the coffee shop on Tuesday morning")
        .add_component("person", "Sarah")
        .add_component("location", "coffee shop")
        .add_component("time", "Tuesday morning")
        .build();

    memory_store.store(episode).await?;

    // Partial cue should retrieve complete episode
    let partial_cue = Cue::from_text("Sarah Tuesday");
    let results = memory_store.recall(&partial_cue).await?;

    println!("Reconstructed from partial cue:");
    for (episode, confidence) in results {
        println!("  {:.3}: {}", confidence, episode.content);
    }

    Ok(())
}

/// Demonstrates confidence-guided exploration.
///
/// Shows how spreading activation confidence guides memory exploration
/// and uncertainty quantification.
#[tokio::main]
async fn confidence_guided_exploration() -> Result<(), Box<dyn std::error::Error>> {
    let config = SpreadingConfig {
        max_hop_count: 5,
        activation_threshold: 0.005, // Lower threshold for exploration
        ..Default::default()
    };

    let memory_store = MemoryStore::with_spreading_config(config);

    // Store memories with varying confidence levels
    memory_store.store_with_confidence(
        Episode::new("High confidence memory"),
        Confidence::new(0.95)
    ).await?;

    memory_store.store_with_confidence(
        Episode::new("Uncertain memory"),
        Confidence::new(0.6)
    ).await?;

    let cue = Cue::from_text("memory");
    let results = memory_store.recall(&cue).await?;

    println!("Confidence-guided results:");
    for (episode, confidence) in results {
        if confidence.value() > 0.8 {
            println!("  HIGH: {:.3}: {}", confidence, episode.content);
        } else if confidence.value() > 0.5 {
            println!("  MED:  {:.3}: {}", confidence, episode.content);
        } else {
            println!("  LOW:  {:.3}: {}", confidence, episode.content);
        }
    }

    Ok(())
}
```

### Performance Tuning Guide
```markdown
# Spreading Activation Performance Tuning

## Understanding Performance Characteristics

Spreading activation performance depends on several factors:

1. **Graph Topology**: Dense graphs spread more slowly than sparse graphs
2. **Hop Count**: Exponential growth with branching factor
3. **Storage Tier Distribution**: Cold tier access adds latency
4. **CPU Architecture**: SIMD capabilities affect batch processing

## Tuning Parameters

### For Low Latency (< 5ms)
```rust
SpreadingConfig {
    max_hop_count: 2,
    activation_threshold: 0.05,
    time_budget: Duration::from_millis(5),
    tier_aware_prioritization: true,
}
```

### For High Recall Quality
```rust
SpreadingConfig {
    max_hop_count: 4,
    activation_threshold: 0.001,
    time_budget: Duration::from_millis(20),
    tier_aware_prioritization: false,
}
```

### For Balanced Performance
```rust
SpreadingConfig {
    max_hop_count: 3,
    activation_threshold: 0.01,
    time_budget: Duration::from_millis(10),
    tier_aware_prioritization: true,
}
```

## Monitoring and Optimization

### Key Metrics to Watch
- `engram_spreading_latency_p95`: Should stay under target latency
- `engram_cycle_detection_rate`: High rates indicate dense graphs
- `engram_memory_pool_utilization`: Should stay under 90%
- `engram_tier_access_latency`: Cold tier access patterns

### Auto-Tuning Recommendations
Enable auto-tuning for production workloads:
```rust
let auto_tuner = SpreadingAutoTuner::new(TuningStrategy::Adaptive);
memory_store.enable_auto_tuning(auto_tuner).await?;
```
```

### Visualization Tool
```rust
/// Debug visualization tool for spreading activation patterns.
///
/// Generates GraphViz DOT output showing activation flow through memory networks.
pub struct SpreadingVisualizer {
    output_format: VisualizationFormat,
    max_nodes: usize,
    highlight_threshold: f32,
}

impl SpreadingVisualizer {
    pub fn visualize_spreading(
        &self,
        spreading_result: &SpreadingResult,
    ) -> Result<String, VisualizationError> {
        let mut dot_output = String::new();
        dot_output.push_str("digraph spreading_activation {\n");
        dot_output.push_str("  rankdir=LR;\n");
        dot_output.push_str("  node [shape=circle];\n");

        // Add nodes with activation-based coloring
        for (node_id, activation) in &spreading_result.node_activations {
            let color = self.activation_to_color(activation.level);
            dot_output.push_str(&format!(
                "  \"{}\" [fillcolor=\"{}\", style=filled, label=\"{}\\n{:.3}\"];\n",
                node_id, color, node_id, activation.level
            ));
        }

        // Add edges with confidence-based weights
        for edge in &spreading_result.traversed_edges {
            let weight = edge.confidence.value() * 5.0; // Scale for visibility
            dot_output.push_str(&format!(
                "  \"{}\" -> \"{}\" [weight={:.1}, label=\"{:.3}\"];\n",
                edge.source, edge.target, weight, edge.confidence
            ));
        }

        dot_output.push_str("}\n");
        Ok(dot_output)
    }

    fn activation_to_color(&self, activation: f32) -> &'static str {
        if activation > 0.8 { "red" }
        else if activation > 0.5 { "orange" }
        else if activation > 0.2 { "yellow" }
        else { "lightblue" }
    }
}
```

## Acceptance Criteria
- [ ] Comprehensive API documentation with cognitive science context
- [ ] Example code demonstrating key cognitive recall patterns
- [ ] Performance tuning guide with specific parameter recommendations
- [ ] Visualization tool for debugging spreading patterns
- [ ] Documentation integrated with existing doc structure
- [ ] All examples validated and include in test suite

## Testing Approach
- Documentation review for technical accuracy
- Example code compilation and execution validation
- Performance tuning guide validation against benchmarks
- Visualization tool testing with various graph topologies

## Notes
This task completes Milestone 3 by ensuring the cognitive spreading activation engine is fully documented and accessible to developers. Quality documentation is essential for adoption, debugging, and future development of cognitive database applications.