# Cognitive Workflow Scenarios

End-to-end behavior-driven scenarios demonstrating multi-step cognitive operations spanning the entire Engram system.

## Purpose

These scenarios validate complete cognitive workflows rather than individual components:
- Multi-phase operations (formation → activation → recall → consolidation)
- System-wide behavior under realistic usage patterns
- Integration across memory types, indices, and cognitive operations
- Performance characteristics of complete workflows

## Running Scenarios

```bash
# Run individual scenario
cargo run --example cognitive_workflow

# Run all scenarios
for scenario in engram-cli/examples/scenarios/*.rs; do
    cargo run --example $(basename $scenario .rs)
done
```

## Available Scenarios

### `cognitive_workflow.rs`
Complete memory lifecycle demonstration:
1. Memory formation (encoding 5 related memories)
2. Spreading activation (semantic associations)
3. Recall operations (query-based retrieval)
4. Pattern completion (reconstruct partial memories)
5. Consolidation assessment (readiness metrics)

**Expected performance:**
- Formation: < 10ms per memory
- Spreading activation: < 100ms
- Recall: < 50ms
- Recall accuracy: > 90%

## Creating New Scenarios

Scenarios should:
1. Demonstrate realistic usage patterns (not artificial stress tests)
2. Validate end-to-end behavior (not individual components)
3. Output human-readable results (not just pass/fail)
4. Include performance expectations (latency, throughput, accuracy)
5. Be deterministic (use fixed seeds for reproducibility)

Example template:
```rust
//! Scenario Name
//!
//! Description of what this scenario validates
//!
//! Run with: cargo run --example scenario_name

use anyhow::Result;

fn main() -> Result<()> {
    println!("=== Scenario Name ===\n");

    // Phase 1: Setup
    // Phase 2: Execute cognitive operations
    // Phase 3: Validate behavior
    // Phase 4: Report results

    Ok(())
}
```

## Relationship to Tests

- **Unit tests** (`cargo test`): Validate individual component correctness
- **Integration tests** (`cargo test --test *`): Validate component interactions
- **Psychology tests** (`cargo test --test psychology --ignored`): Validate against empirical data
- **Scenarios** (`cargo run --example *`): Demonstrate complete workflows
- **Benchmarks** (`cargo bench`): Measure performance under load

Scenarios complement tests by providing:
1. Human-readable demonstrations of system capabilities
2. Validation of realistic usage patterns
3. Artifacts and visualizations for debugging
4. Performance baselines for cognitive operations
