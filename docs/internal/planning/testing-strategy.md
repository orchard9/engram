# Testing and Validation Strategy

Engram's testing strategy follows a multi-layered approach designed to validate both correctness and biological plausibility.

## Testing Layers

### 1. Unit Tests (`cargo test`)

**Location:** `src/**/*.rs` (inline module tests)

**Purpose:** Validate individual component correctness

**Characteristics:**

- Fast execution (< 1 second total)

- Deterministic results

- Pure function validation

- Edge case coverage

**Example:**

```rust
#[cfg(test)]
mod tests {
    #[test]
    fn test_confidence_bounds() {
        let conf = Confidence::from_raw(0.5);
        assert_eq!(conf.raw(), 0.5);
    }
}

```

### 2. Integration Tests (`cargo test --test <name>`)

**Location:** `engram-core/tests/*.rs`, `engram-cli/tests/*.rs`

**Purpose:** Validate component interactions

**Characteristics:**

- Multiple components working together

- Realistic data flows

- API contract validation

- Deterministic seeds for reproducibility

**Example:**

```rust
#[tokio::test]
async fn test_store_and_recall_integration() {
    let store = MemoryStore::new(1000);
    store.store(episode);
    let results = store.recall(&query, 10);
    assert_eq!(results.len(), 1);
}

```

**Long-running integration tests:**

- Use `#[ignore]` attribute

- Run with `cargo test --ignored`

- Typically simulate server lifecycle (seconds to minutes)

### 3. Property-Based Tests (`cargo test`)

**Location:** `engram-core/tests/confidence_property_tests.rs`

**Purpose:** Validate probabilistic invariants

**Characteristics:**

- Randomized inputs

- Invariant checking

- Shrinking on failure

- Coverage of input space

**Example:**

```rust
proptest! {
    #[test]
    fn confidence_never_exceeds_bounds(raw in 0.0f32..1.0f32) {
        let conf = Confidence::from_raw(raw);
        assert!(conf.raw() >= 0.0 && conf.raw() <= 1.0);
    }
}

```

### 4. Compile-Time Tests

**Location:** `engram-core/tests/compile_fail/*.rs`, `engram-core/tests/compile_pass/*.rs`

**Purpose:** Validate type-state pattern and API ergonomics

**Characteristics:**

- Ensures invalid code doesn't compile

- Documents API usage patterns

- Zero runtime cost validation

**Example:**

```rust
// compile_fail/memory_builder_missing_confidence.rs
// Should fail: confidence is required
let memory = MemoryBuilder::new()
    .id("test")
    .embedding(vec![0.1; 768])
    .build(); // ERROR: missing confidence

```

### 5. Psychology Validation Tests (`cargo test --test psychology --ignored`)

**Location:** `engram-core/tests/psychology/*.rs`

**Purpose:** Validate against empirical psychology research (Milestone 12)

**Characteristics:**

- Long-running (hours to days)

- Empirical data baselines from published research

- Tolerance-based validation (typically ±5%)

- Citations for all empirical baselines

**Example:**

```rust
#[tokio::test]
#[ignore] // Simulates multi-day decay
async fn test_ebbinghaus_forgetting_curve() {
    // Validates against Ebbinghaus (1885) retention rates
    let expected_retention = 0.60; // 60% at 20 minutes
    let tolerance = 0.05; // 5% from Milestone 4
    assert!((actual - expected).abs() <= tolerance);
}

```

**Required validations:**

- DRM paradigm (false memory formation)

- Proactive/retroactive interference

- Forgetting curves (Ebbinghaus replication)

- Semantic priming (spreading activation)

- Memory consolidation patterns

### 6. Benchmarks (`cargo bench`)

**Location:** `engram-core/benches/**/*.rs`

**Purpose:** Performance validation and regression detection

**Characteristics:**

- Criterion.rs framework

- Statistical analysis

- Comparison against baseline

- Tracks allocations and time

**Example:**

```rust
fn benchmark_recall(c: &mut Criterion) {
    c.bench_function("recall_k10", |b| {
        b.iter(|| store.recall(&query, 10));
    });
}

```

**Performance requirements:**

- All merges require benchmark comparison

- P50, P95, P99 latency tracking

- Allocation count monitoring

- Flamegraphs for hot paths

### 7. Cognitive Workflow Scenarios (`cargo run --example scenarios/*`)

**Location:** `engram-cli/examples/scenarios/*.rs`

**Purpose:** End-to-end validation of complete workflows

**Characteristics:**

- Multi-phase operations

- Human-readable output

- Performance expectations documented

- Artifact generation

- Not part of `cargo test`

**Example:**

```rust
// examples/scenarios/cognitive_workflow.rs
fn main() {
    println!("Phase 1: Memory Formation");
    // Store memories
    println!("Phase 2: Spreading Activation");
    // Activate concepts
    println!("Phase 3: Recall");
    // Query memories
    println!("Summary: All phases completed in {:?}", total_time);
}

```

### 8. Soak Tests (`cargo run --example soak/*`)

**Location:** `engram-cli/examples/soak/*.rs`

**Purpose:** Long-running stability and resource validation

**Characteristics:**

- Hours to days of execution

- Memory leak detection

- Performance degradation monitoring

- Snapshot capture at intervals

- Not part of `cargo test`

**Example:**

```rust
// examples/soak/memory_pool_soak.rs
const ITERATIONS: usize = 86400; // 24 hours
for i in 0..ITERATIONS {
    exercise_pool();
    if i % 1000 == 0 {
        capture_snapshot();
    }
}

```

### 9. Differential Testing (`cargo test --test differential_testing`)

**Location:** `engram-core/benches/milestone_1/differential_testing.rs`

**Purpose:** Validate Rust and Zig implementations produce identical results (Milestone 9)

**Characteristics:**

- Bit-identical output validation

- Million-operation traces

- Cross-language correctness

- Performance comparison

**Example:**

```rust
#[test]
fn test_rust_zig_equivalence() {
    let rust_result = rust_impl.compute(&input);
    let zig_result = zig_impl.compute(&input);
    assert_eq!(rust_result, zig_result, "Implementations must be bit-identical");
}

```

## Testing Infrastructure

### Feature Flags

**`long_running_tests`** - Enable multi-hour validation tests:

```bash
cargo test --features long_running_tests

```

**`monitoring`** - Enable metrics collection (default):

```bash
cargo test  # Monitoring enabled
cargo test --no-default-features  # Minimal build without monitoring

```

### Test Organization

```
engram/
├── engram-core/
│   ├── src/             # Unit tests (inline #[cfg(test)])
│   ├── tests/
│   │   ├── *.rs              # Integration tests
│   │   ├── psychology/       # Empirical validation (Milestone 12)
│   │   ├── compile_fail/     # Type-state validation
│   │   └── compile_pass/     # API usage patterns
│   └── benches/
│       └── milestone_*/      # Performance benchmarks
│
└── engram-cli/
    ├── tests/           # CLI integration tests
    └── examples/
        ├── scenarios/   # End-to-end workflows
        └── soak/        # Long-running stability tests

```

### Running All Tests

```bash
# Fast unit + integration tests
cargo test

# Include long-running tests
cargo test --features long_running_tests

# Include ignored tests (psychology, long-running)
cargo test --ignored

# All tests + benchmarks
cargo test && cargo bench

# With code coverage
cargo tarpaulin --all-features --workspace

# Memory leak detection
valgrind --leak-check=full cargo test

# Run specific test category
cargo test --test psychology --ignored  # Psychology validations
cargo run --example cognitive_workflow   # Scenarios
cargo run --example memory_pool_soak     # Soak tests

```

## Milestone-Specific Requirements

Each milestone specifies validation criteria:

- **Milestone 1:** Type-state prevents invalid construction

- **Milestone 2:** 90% recall@10, <1ms query time vs. FAISS

- **Milestone 4:** Forgetting curves within 5% of empirical data

- **Milestone 9:** Bit-identical Rust/Zig outputs

- **Milestone 10:** 100K observations/second sustained

- **Milestone 12:** Replicate DRM, interference patterns

- **Milestone 13:** Jepsen-style distributed consistency testing

## Continuous Integration

### Quick Tests (every commit)

```bash
cargo test --workspace
cargo clippy --all-targets
cargo fmt -- --check

```

### Nightly Tests

```bash
cargo test --ignored
cargo bench
cargo run --example memory_pool_soak

```

### Weekly Tests

```bash
cargo test --features long_running_tests --ignored
# 24-hour soak tests
# Full benchmark suite
# Psychology validations

```

## Debugging Failed Tests

### For unit/integration test failures

```bash
# Verbose output
cargo test -- --nocapture

# Specific test
cargo test test_name -- --exact --nocapture

# With backtraces
RUST_BACKTRACE=1 cargo test test_name

```

### For psychology test failures

1. Check tolerance levels (5% from Milestone 4)

2. Verify empirical baseline is correct

3. Check for randomness/seed issues

4. Compare against cited research

5. Document deviations in test output

### For soak test failures

1. Check for memory leaks with valgrind

2. Monitor resource usage (htop, Activity Monitor)

3. Review snapshot outputs for anomalies

4. Check for performance degradation over time

5. Verify metrics accuracy

## References

- **Coding Guidelines:** See `coding_guidelines.md` for testing conventions

- **Milestones:** See `milestones.md` for validation requirements

- **Psychology Tests:** See `engram-core/tests/psychology/README.md`

- **Scenarios:** See `engram-cli/examples/scenarios/README.md`

- **Soak Tests:** See `engram-cli/examples/soak/README.md`
