# Task 010: Network Partition Testing Framework - Codebase Analysis Report

## Executive Summary

The Engram codebase already has **partial chaos testing infrastructure** in place at `engram-core/tests/chaos/`. This analysis identifies how to integrate the comprehensive Task 010 specification with the existing infrastructure and reveals key abstraction points for network transport simulation.

**Key Finding**: The codebase is well-structured for test infrastructure expansion, with existing async testing patterns, helper utilities, and a pre-existing chaos testing framework that handles streaming memory operations.

---

## 1. Existing Test Infrastructure

### 1.1 Test Directory Structure

```
engram-core/tests/
├── support/                    # Test utilities and fixtures
│   ├── mod.rs                 # Shared test helpers
│   └── graph_builders.rs      # Graph construction for tests (GraphFixture, etc.)
├── helpers/                    # Additional test helpers
│   ├── mod.rs
│   └── embeddings.rs          # Embedding generation utilities
├── common/                     # Common test module
├── integration/               # Integration test subdirectory
│   ├── helpers/
│   └── *.rs                   # Integration tests
├── chaos/                      # EXISTING chaos testing framework
│   ├── mod.rs                 # Framework documentation
│   ├── fault_injector.rs      # DelayInjector, PacketLossSimulator, etc.
│   ├── validators.rs          # Invariant validators
│   └── (missing: streaming_chaos.rs, network_simulator.rs)
├── accuracy/                   # Accuracy validation tests
├── cognitive/                  # Cognitive dynamics tests
├── compile_fail/              # Compile-time error tests (40+ files)
├── compile_pass/              # Compile-time success tests (3 files)
└── *.rs                        # 90+ individual test files
```

**Key Insight**: The chaos directory exists but is incomplete. `streaming_chaos.rs` is referenced in `mod.rs` but not implemented.

### 1.2 Existing Chaos Testing Infrastructure

#### `engram-core/tests/chaos/mod.rs`
- Framework for streaming memory operations
- Comprehensive documentation (55 lines)
- References chaos engineering literature
- Exports: `DelayInjector`, `PacketLossSimulator`, `ChaosScenario`, validators

#### `engram-core/tests/chaos/fault_injector.rs`
- **`DelayInjector`**: Random network delays (min/max ms)
  - Seeded RNG for reproducibility
  - `async fn inject_delay()` - suspends execution
  
- **`PacketLossSimulator`**: Probabilistic packet drops
  - Configurable drop rate (0.0-1.0)
  - Tracks `drops_total` and `attempts_total` for statistics
  
- **`BurstLoadGenerator`**: Queue overflow simulation
  
- **`ClockSkewSimulator`**: Time drift simulation

- **`ChaosScenario`**: Declarative scenario definition
  - Builder pattern support
  - Composes fault injectors

#### `engram-core/tests/chaos/validators.rs`
- **`EventualConsistencyValidator`**: Tracks acked observations
  - `record_ack()` - records acknowledged data
  - `validate_all_present()` - checks recall completeness
  
- **`SequenceValidator`**: Monotonic sequence number checking
  
- **`GraphIntegrityValidator`**: HNSW structure validation
  
- **`ValidationError`** enum with detailed error types

#### Statistics Tracking
- `ChaosTestStats` for collecting chaos test metrics
- Tracks: latency percentiles, throughput, error rates

### 1.3 Test File Organization

The test suite uses **multiple patterns**:

1. **Standalone test files** (most common)
   ```rust
   // error_recovery_integration.rs
   #[tokio::test]
   async fn test_retry_strategy_with_exponential_backoff() { ... }
   ```

2. **Nested test modules** (less common)
   ```rust
   // query_language_corpus.rs
   mod tests {
       #[test]
       fn test_something() { ... }
   }
   ```

3. **Helper modules** (utilities imported by tests)
   ```rust
   // tests/support/graph_builders.rs
   pub struct GraphFixture { ... }
   ```

### 1.4 Async Testing Patterns

The codebase uses **tokio async testing** extensively:

```rust
// Pattern from error_recovery_integration.rs
#[tokio::test]
async fn test_async_operation() {
    use tokio::time::timeout;
    
    let result = timeout(Duration::from_secs(10), async_operation()).await;
    assert!(result.is_ok());
}
```

**Key Dependencies**:
- `tokio = { version = "1.47", features = ["full"] }` (workspace dependency)
- `tokio::test` available in dev-dependencies
- `tokio::time::timeout` for test deadlines
- `Arc<Mutex<T>>` for shared mutable state in tests

### 1.5 Synchronization Patterns

The codebase uses **multiple synchronization primitives**:

- `Arc<Mutex<T>>` - Basic shared mutable state
- `Arc<atomic::AtomicU64>` - Lock-free counters
- `parking_lot::Mutex` - Faster alternative to std Mutex
- `dashmap::DashMap` - Concurrent hash map (workspace dep)
- `crossbeam::*` - Lock-free data structures

**Example from fault_injector.rs**:
```rust
rng: Arc<Mutex<StdRng>>,
drops_total: Arc<std::sync::atomic::AtomicU64>,
```

---

## 2. Analyzing Network Transport Abstraction

### 2.1 Current Clustering/Distribution Status

**Finding**: The codebase does NOT have cluster/distributed memory features yet.

```bash
$ find engram-core/src -name "*cluster*" -o -name "*network*" -o -name "*partition*"
(no results)
```

**Implications for Task 010**:
- Network transport is abstract (will be added in this task)
- No existing distributed components to integrate with
- Clean slate for NetworkTransport trait design

### 2.2 Where Transport Abstraction Fits

Based on Task 010's requirements, we need:

```rust
// Will be created in engram-core/src/cluster/transport.rs
pub trait NetworkTransport: Send + Sync {
    async fn send(&self, from: SocketAddr, to: SocketAddr, data: Vec<u8>) -> Result<()>;
    async fn recv(&self, addr: SocketAddr) -> Option<(Vec<u8>, SocketAddr)>;
}

// Two implementations:
pub struct RealUdpTransport { ... }    // Production
pub struct SimulatedTransport { ... }  // Tests (uses NetworkSimulator)
```

### 2.3 Test Double Pattern

The task specification follows the **Test Double** pattern:

1. **Real Implementation** (`RealUdpTransport`)
   - Uses actual UDP sockets
   - Used in integration tests with real networking

2. **Simulated Implementation** (`SimulatedTransport`)
   - Wraps `NetworkSimulator`
   - Deterministic, reproducible
   - Used in chaos tests

3. **Dependency Injection**
   ```rust
   pub struct SwimMembership {
       transport: Arc<dyn NetworkTransport>,
       // ...
   }
   
   impl SwimMembership {
       pub fn new_with_transport(
           id: String,
           addr: SocketAddr,
           transport: impl NetworkTransport,
       ) -> Self { ... }
   }
   ```

---

## 3. Benchmark and Profiling Infrastructure

### 3.1 Testing Benchmarks

The Makefile provides:
```makefile
test:
	cargo test --workspace -- --test-threads=1

quality: fmt lint test docs-lint example-cognitive
```

**Note**: Single-threaded test execution (important for determinism!)

### 3.2 Criterion Benchmarking

Cargo.toml includes criterion for performance benchmarks:
```toml
[[bench]]
name = "streaming_throughput"
harness = false
```

### 3.3 Performance Testing Infrastructure

Existing performance validation:
- Differential testing between implementations
- GPU acceleration benchmarks
- Streaming throughput tests

---

## 4. CI/CD and Test Execution Strategy

### 4.1 Current CI Structure

**Finding**: No GitHub Actions workflows found

```bash
$ find .github -type f
(no results)
```

**Note from CLAUDE.md**:
> "Never use .github workflows, actions, or CI - all quality checks run via make quality and git hooks."

This means **CI integration for chaos tests should use Make targets**, not GitHub Actions.

### 4.2 Makefile Integration Strategy

Current structure:
```makefile
.PHONY: fmt lint test docs-lint example-cognitive quality
quality: fmt lint test docs-lint example-cognitive
```

**We should add**:
```makefile
.PHONY: chaos-test chaos-test-quick chaos-test-full
chaos-test-quick:
	cargo test --package engram-core --test partition_scenarios test_clean_partition -- --nocapture
chaos-test: # Full chaos test suite
	cargo test --package engram-core --test partition_scenarios -- --nocapture --test-threads=1
chaos-test-full: # Extended soak tests
	CHAOS_TEST_DURATION=300 cargo test --package engram-core --test partition_scenarios -- --nocapture --ignored
```

### 4.3 Test Scripts Directory

Existing scripts in `/scripts/` provide models:

```
scripts/
├── chaos/                    # Existing chaos scripts
│   ├── cleanup_chaos.sh
│   ├── inject_network_latency.sh      # Good model for fault injection
│   ├── inject_packet_loss.sh          # Models for our scenarios
│   └── ...
├── engram_diagnostics.sh     # Good model for health checking
├── analyze_benchmarks.py     # Models for result analysis
└── ... (40+ utility scripts)
```

**Key pattern**: `/scripts/chaos/inject_*.sh` scripts handle OS-level chaos

Our `run_chaos_tests.sh` will complement these with **in-process simulation**.

---

## 5. Dependency Analysis

### 5.1 Already Available Dependencies

From `Cargo.toml` (workspace):
```toml
# Async runtime
tokio = { version = "1.47", features = ["full"] }

# Random number generation (seeded for determinism)
rand = "0.8"

# Serialization (for event recording)
serde = { version = "1.0", features = ["derive"] }
bincode = "1.3"

# Synchronization primitives
parking_lot = "0.12"
crossbeam = "0.8"
dashmap = "5.5"

# Time handling
chrono = { version = "0.4", features = ["serde"] }

# Testing support
proptest = "1.5"
quickcheck = "1.0"
```

### 5.2 Additional Dependencies Needed

From Task 010 specification:
```toml
[dev-dependencies]
# Already included above, but worth checking versions:
proptest = "1.4"          # Property-based testing
quickcheck = "1.0"        # Property testing alternative

# NEW - but low priority (optional for future expansion):
thiserror = "1.0"         # Already in workspace!
```

**Finding**: All required dependencies are already in workspace Cargo.toml!

### 5.3 Feature Flags

We may want to add:
```toml
[features]
# In engram-core Cargo.toml:
chaos_testing = []  # Enable in-process chaos tests
deterministic_rng = []  # Force seeded RNG for reproducibility
```

---

## 6. Test Utility Patterns to Follow

### 6.1 Builder Pattern (from graph_builders.rs)

The existing `GraphFixture` uses a builder-like pattern:

```rust
pub struct GraphFixture {
    pub name: &'static str,
    pub seeds: Vec<(String, f32)>,
    pub graph: Arc<MemoryGraph>,
}

impl GraphFixture {
    pub fn new(name, description, graph, seeds) -> Self { ... }
    pub fn with_config_adjuster<F>(mut self, adjuster: F) -> Self { ... }
}
```

**We should follow this for `ChaosScenarioBuilder`** (already specified in Task 010):

```rust
pub struct ChaosScenarioBuilder {
    scenario: ChaosScenario,
}

impl ChaosScenarioBuilder {
    pub fn new(name: &str) -> Self { ... }
    pub fn nodes(mut self, num: usize) -> Self { ... }
    pub fn duration(mut self, duration: Duration) -> Self { ... }
    pub fn inject_fault(mut self, at: Duration, fault: FaultSpec, duration: Option<Duration>) -> Self { ... }
    pub fn build(self) -> ChaosScenario { ... }
}
```

### 6.2 Arc<Mutex<T>> for Shared Test State

Pattern from existing tests:

```rust
// From error_recovery_integration.rs
let attempts = Arc::new(AtomicU32::new(0));
let attempts_clone = attempts.clone();

// Passed to async closure
let result = ErrorRecovery::with_retry(
    || async {
        let current = attempts_clone.fetch_add(1, Ordering::SeqCst);
        // ...
    },
    // ...
).await;

assert_eq!(attempts.load(Ordering::SeqCst), 3);
```

**We use similar patterns in NetworkSimulator**:
```rust
pub struct NetworkSimulator {
    clock: Arc<Mutex<u64>>,
    message_queues: Arc<Mutex<HashMap<...>>>,
    fault_injectors: Arc<Mutex<HashMap<...>>>,
    rng: Arc<Mutex<StdRng>>,
}
```

### 6.3 Test Module Organization

Two patterns used in codebase:

**Pattern 1: Standalone test file** (PREFERRED for larger tests)
```rust
// error_recovery_integration.rs
#[tokio::test]
async fn test_something() { ... }

#[tokio::test]
async fn test_another() { ... }
```

**Pattern 2: Nested module** (less common)
```rust
// query_language_corpus.rs
mod tests {
    #[test]
    fn test_something() { ... }
}
```

**For Task 010, we'll use Pattern 1**:
- `engram-core/tests/partition_scenarios.rs` - 5 concrete scenarios
- `engram-core/tests/network_simulator.rs` - Unit tests for simulator
- `engram-core/tests/chaos/scenario.rs` - Scenario DSL
- `engram-core/tests/chaos/orchestrator.rs` - Test runner
- `engram-core/tests/chaos/replay.rs` - Deterministic replay

---

## 7. Integration Approach for NetworkSimulator

### 7.1 Module Structure

```
engram-core/
├── src/
│   ├── cluster/                          # NEW
│   │   ├── mod.rs                        # Exports
│   │   ├── transport.rs                  # NetworkTransport trait
│   │   ├── membership.rs                 # SWIM membership with transport
│   │   └── test_transport.rs             # SimulatedTransport impl
│   └── ... (existing modules)
└── tests/
    ├── network_simulator.rs              # NetworkSimulator unit tests
    ├── partition_scenarios.rs            # 5 concrete chaos scenarios
    ├── chaos/
    │   ├── mod.rs                        # Updated to include network tests
    │   ├── fault_injector.rs             # EXISTING - extend as needed
    │   ├── validators.rs                 # EXISTING - extend as needed
    │   ├── scenario.rs                   # NEW - ChaosScenario DSL
    │   ├── orchestrator.rs               # NEW - Test runner
    │   └── replay.rs                     # NEW - Deterministic replay
    └── support/
        └── mod.rs                        # Update imports
```

### 7.2 Integration Points

1. **With existing chaos framework**:
   - Extend `fault_injector.rs` with network-specific faults
   - Extend `validators.rs` with network partition validators
   - Update `mod.rs` to export new network testing module

2. **With testing support**:
   - Add helper in `tests/support/mod.rs` for network setup
   - Reuse `Arc<Mutex<>>` patterns

3. **With cargo features**:
   - `chaos_testing` feature flag (optional, for large test binaries)
   - Default to including chaos tests in dev builds

### 7.3 Deterministic Time Handling

**Key Challenge**: Simulating time in tests while keeping determinism.

**Solution**: Use logical time (simulated clock), not wall-clock time.

```rust
pub struct NetworkSimulator {
    clock: Arc<Mutex<u64>>,  // Logical time in milliseconds
}

impl NetworkSimulator {
    pub fn now(&self) -> u64 {
        *self.clock.lock().unwrap()
    }
    
    pub fn advance_time(&self, duration_ms: u64) {
        let mut clock = self.clock.lock().unwrap();
        *clock += duration_ms;
        drop(clock);  // Release lock before delivering messages
        self.deliver_pending_messages(*self.clock.lock().unwrap());
    }
}
```

**In tests**:
```rust
#[tokio::test]
async fn test_partition_healing() {
    let sim = Arc::new(NetworkSimulator::new(Default::default()));
    
    // Create cluster
    for i in 0..5 {
        let addr = format!("127.0.0.1:{}", 7946 + i).parse().unwrap();
        let node = TestNode::new(i, addr, sim.clone());
        nodes.push(node);
    }
    
    // Let cluster form (simulated time)
    sim.advance_time(5000);
    
    // Partition
    sim.partition(&nodes[0..3], &nodes[3..5]);
    
    // Simulate network communication
    sim.advance_time(2000);
    
    // Verify partition detected
    assert_eq!(nodes[0].members.len(), 3);
    assert_eq!(nodes[3].members.len(), 2);
    
    // Heal
    sim.heal(&nodes[0..3], &nodes[3..5]);
    sim.advance_time(5000);
    
    // Verify healed
    for node in &nodes {
        assert_eq!(node.members.len(), 5);
    }
}
```

---

## 8. Existing Test Patterns We Should Follow

### 8.1 Assertion Patterns

From existing tests:
```rust
// Simple assertions
assert!(result.is_ok());
assert_eq!(count, expected);

// With custom messages
assert!(condition, "Custom error: {:?}", context);

// Closures/filters
assert!(!results.is_empty(), "Should have recall results");
assert!(results.len() <= 2, "Should respect max_results");

// Found in confidence tests
assert!((observed - expected).abs() <= tolerance, "Deviation message");
```

**For Task 010, follow existing patterns**:
```rust
assert!(result.passed(), "Test failed: {:?}", result.invariant_checks);
assert_eq!(nodes[0].members.len(), 3, "Partition not detected");
```

### 8.2 Test Isolation

From existing patterns:
- **Create fresh instances per test** (no shared state)
- **Use #[tokio::test]** for async operations
- **Scope expensive operations** early to ensure cleanup

```rust
#[tokio::test]
async fn test_isolated_operation() {
    let store = MemoryStore::new(1000);  // Fresh per test
    // ... test operations
}
```

### 8.3 Deterministic RNG Usage

From fault_injector.rs:
```rust
pub struct DelayInjector {
    rng: Arc<Mutex<StdRng>>,
}

pub fn new(min_delay_ms: u64, max_delay_ms: u64, seed: u64) -> Self {
    Self {
        rng: Arc::new(Mutex::new(StdRng::seed_from_u64(seed))),
    }
}
```

**This matches NetworkSimulator's approach**:
```rust
pub struct NetworkSimulator {
    rng: Arc<Mutex<StdRng>>,
}

pub fn new(config: SimulatorConfig) -> Self {
    Self {
        rng: Arc::new(Mutex::new(StdRng::seed_from_u64(config.seed))),
    }
}
```

---

## 9. Performance Test Infrastructure

### 9.1 Existing Benchmarking

The codebase has:
- Criterion benchmarks (`.criterion/` exists)
- Divan microbenchmarks
- Profiling harnesses with `pprof`

### 9.2 For Task 010 Performance Validation

We should add:
```rust
// engram-core/benches/chaos_simulator_performance.rs
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn bench_message_simulation(c: &mut Criterion) {
    c.bench_function("simulate_100_node_cluster_60s", |b| {
        b.iter(|| {
            let sim = NetworkSimulator::new(Default::default());
            // Setup 100 nodes
            // Simulate 60s worth of events
            // Measure wall-clock time
        });
    });
}

criterion_group!(benches, bench_message_simulation);
criterion_main!(benches);
```

This validates the **performance targets**:
- 5-node cluster 60s test: <5s wall-clock time
- 100-node cluster 60s test: <30s wall-clock time

---

## 10. Documentation Integration

### 10.1 Documentation Structure

```
docs/
├── guide/                     # Tutorials (learning)
├── howto/                     # How-to guides (problem-solving)
├── explanation/               # Conceptual explanations
├── reference/                 # API reference
├── operations/                # Production operations (sysadmin)
├── internal/                  # Internal planning docs
└── ... (assets, diagrams, etc.)
```

### 10.2 Where to Document Task 010

Following Diátaxis framework:

1. **Tutorial**: `docs/guide/chaos-testing-quickstart.md`
   - How to run the 5 scenarios
   - Interpreting results
   - Debugging a failure

2. **How-to**: `docs/howto/write-new-chaos-scenario.md`
   - Step-by-step guide for adding scenarios
   - Builder API reference
   - Common patterns

3. **Explanation**: `docs/explanation/network-partition-testing.md`
   - Why we test partitions
   - Network simulation approach
   - Deterministic replay explanation

4. **Reference**: `docs/reference/chaos-testing-api.md`
   - `NetworkSimulator` API
   - `ChaosScenario` builder API
   - Fault types and invariants

5. **Operations**: (only if distributed feature ships)
   - Production chaos testing procedures

---

## 11. High-Level Integration Checklist

### Before Implementation Starts

- [ ] Review existing `engram-core/tests/chaos/` structure
- [ ] Understand `test_support` module in `engram-core/src/activation/`
- [ ] Verify all dependencies are in `Cargo.toml` (they are!)
- [ ] Plan module layout in `engram-core/src/cluster/`

### Core Implementation

- [ ] Implement `NetworkTransport` trait in `src/cluster/transport.rs`
- [ ] Implement `SimulatedTransport` in `src/cluster/test_transport.rs`
- [ ] Implement `NetworkSimulator` in `tests/network_simulator.rs`
- [ ] Implement `ChaosScenario` DSL in `tests/chaos/scenario.rs`
- [ ] Implement `ChaosOrchestrator` in `tests/chaos/orchestrator.rs`
- [ ] Implement 5 test scenarios in `tests/partition_scenarios.rs`

### Testing & Validation

- [ ] Run `cargo test network_simulator` - unit tests pass
- [ ] Run `cargo test partition_scenarios` - all 5 scenarios pass
- [ ] Run `cargo test chaos` - integration tests pass
- [ ] Run `make quality` - no clippy warnings
- [ ] Verify determinism: same seed = same results

### Documentation

- [ ] Update `tests/chaos/mod.rs` with new exports
- [ ] Create `docs/guide/chaos-testing-quickstart.md`
- [ ] Create `docs/howto/write-new-chaos-scenario.md`
- [ ] Update root `README.md` with link to chaos testing guide

### CI Integration

- [ ] Add `make chaos-test` target
- [ ] Create `scripts/run_chaos_tests.sh`
- [ ] Update `.pre-commit` hooks if using them

---

## 12. Summary: Key Findings

### What Already Exists

1. **Chaos test framework** at `engram-core/tests/chaos/`
   - Fault injectors (delay, packet loss, clock skew)
   - Validators (eventual consistency, sequence, graph integrity)
   - Partial infrastructure for broader network simulation

2. **Async testing** with tokio
   - `#[tokio::test]` macro available
   - Timeout utilities, Arc<Mutex<T>> patterns established

3. **Test utilities**
   - Builder patterns for test configuration
   - Fixture support in `tests/support/`
   - Shared helpers in `tests/helpers/`

4. **Synchronization primitives**
   - Atomic operations for counters
   - Arc<Mutex<>> for shared mutable state
   - Deterministic seeded RNG (StdRng::seed_from_u64)

5. **Dependencies**
   - All required deps already in workspace Cargo.toml
   - No additional dependencies needed

### What Needs to Be Added

1. **Network Transport Layer**
   - `NetworkTransport` trait
   - `RealUdpTransport` implementation
   - `SimulatedTransport` wrapping NetworkSimulator

2. **NetworkSimulator**
   - Complete in-process network simulation
   - Deterministic message delivery
   - Fault injection for realistic scenarios

3. **Scenario DSL & Orchestrator**
   - Declarative fault injection scheduling
   - Automated test execution and result collection
   - Invariant checking after test completion

4. **5 Concrete Test Scenarios**
   - Clean partition (3|2 split)
   - Asymmetric partition (one-way failure)
   - Flapping partition (rapid cycles)
   - Cascading failures (sequential node death)
   - Network congestion (latency + packet loss)

5. **Documentation**
   - User guide for running tests
   - Developer guide for writing new scenarios
   - Conceptual explanation of approach

### Integration Strategy

1. **Extend existing chaos module** - add scenario.rs, orchestrator.rs, replay.rs
2. **Add cluster module** - new transport.rs, test_transport.rs, membership.rs
3. **Reuse existing patterns** - Arc<Mutex<>>, Arc<StdRng>, async/tokio patterns
4. **Follow established conventions** - builder pattern, test doubles, deterministic RNG
5. **Add Make targets** - chaos-test, chaos-test-quick, chaos-test-full

---

## 13. Specific File Paths for Implementation

### New Files to Create

```
engram-core/src/cluster/
├── mod.rs                       # Module exports
├── transport.rs                 # NetworkTransport trait + RealUdpTransport
├── membership.rs                # SWIM membership protocol (Milestone 14 Task 003)
└── test_transport.rs            # SimulatedTransport implementation

engram-core/tests/
├── network_simulator.rs         # NetworkSimulator + unit tests
└── partition_scenarios.rs       # 5 concrete test scenarios

engram-core/tests/chaos/
├── scenario.rs                  # ChaosScenario DSL + builder
├── orchestrator.rs              # ChaosOrchestrator + TestNode
└── replay.rs                    # Deterministic replay mechanism

scripts/
└── run_chaos_tests.sh           # CI integration script

docs/guide/
└── chaos-testing-quickstart.md # User guide

docs/howto/
└── write-new-chaos-scenario.md # Developer guide
```

### Files to Modify

```
engram-core/Cargo.toml           # No changes needed (all deps present)
engram-core/tests/chaos/mod.rs   # Add network simulator exports
engram-core/src/lib.rs           # Add pub mod cluster
Makefile                         # Add chaos-test targets
```

---

## 14. Critical Implementation Notes

1. **Determinism is key** - Use seeded RNG, logical time, reproducible event ordering
2. **Test isolation** - Each test gets fresh simulator instance, no shared state
3. **Single-threaded tests** - Makefile uses `--test-threads=1` for determinism
4. **Async/await support** - Tokio runtime available, use #[tokio::test]
5. **Arc<Mutex<>> locking** - Remember to drop locks to avoid deadlocks
6. **Builder pattern** - Follow ChaosScenarioBuilder for declarative tests
7. **No external dependencies** - All needed deps already in Cargo.toml
8. **Performance targets** - 5s for 5-node, 30s for 100-node simulations

