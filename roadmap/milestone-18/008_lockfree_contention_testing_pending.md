# Task 008: Lock-Free Contention Testing

**Status**: Pending
**Estimated Duration**: 3-4 days
**Priority**: Medium - Validates DashMap hot-spot behavior

## Objective

Create synthetic hot-spot scenarios (Zipf distribution, 80/20 access patterns) to stress DashMap sharding. Measure contention under skewed workloads and validate <10% performance loss at 128 concurrent writers.

## Hot-Spot Patterns

**Zipf Distribution**: 20% of nodes receive 80% of traffic
- Power-law parameter s=1.0 (realistic social networks)
- Tests shard-local contention

**Celebrity Node**: Single node receives 50% of all accesses
- Worst-case scenario
- Tests per-bucket locking

## Implementation

```rust
pub struct HotSpotGenerator {
    distribution: HotSpotDistribution,
}

pub enum HotSpotDistribution {
    Uniform,  // Baseline (no hot spots)
    Zipf { s: f64 }, // Power-law: P(k) ∝ 1/k^s
    Celebrity { node_id: NodeId, concentration: f64 }, // Single hot node
}

impl HotSpotGenerator {
    pub fn sample_node(&self, rng: &mut impl Rng) -> NodeId {
        match &self.distribution {
            HotSpotDistribution::Zipf { s } => {
                // Rejection sampling for Zipf
                let u = rng.gen::<f64>();
                let k = (u.powf(-1.0 / s)).floor() as usize;
                NodeId(k)
            }
            // ...
        }
    }
}
```

## Success Criteria

- **Uniform Load**: Baseline (100% performance)
- **Zipf Load**: >90% of uniform performance
- **Celebrity Load**: >70% of uniform performance
- **No Deadlocks**: Zero hangs under any distribution

## Files

- `tools/loadtest/src/contention/hotspot_generator.rs` (320 lines)
- `scenarios/contention/zipf_hotspots.toml`
- `scenarios/contention/celebrity_node.toml`
