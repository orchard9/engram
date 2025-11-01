# Task 010: Network Partition Testing Framework - Implementation Guide

## Quick Start for Implementer

This document provides a condensed guide for implementing Task 010, pulling insights from the comprehensive codebase analysis.

### The Task in One Sentence

Build a **deterministic network simulator** that injects controlled failures (partitions, latency, packet loss) into distributed cluster tests, enabling reproducible chaos testing with the ability to replay failures from a seed.

---

## Phase 1: Foundation (Day 1)

### 1.1 Understand the Existing Chaos Framework

The codebase already has partial chaos testing at `engram-core/tests/chaos/`:

**Existing Components**:
- `fault_injector.rs`: `DelayInjector`, `PacketLossSimulator`, `BurstLoadGenerator`, `ClockSkewSimulator`
- `validators.rs`: `EventualConsistencyValidator`, `SequenceValidator`, `GraphIntegrityValidator`
- `mod.rs`: Module documentation and exports

**What's Missing**:
- `streaming_chaos.rs` (referenced but not implemented)
- `network_simulator.rs` (the core simulation engine)
- Network-specific scenario DSL
- Test orchestrator

### 1.2 Review Key Patterns in Existing Tests

Before writing code, study these patterns:

1. **Async Testing** (`tests/error_recovery_integration.rs`)
   ```rust
   #[tokio::test]
   async fn test_name() {
       let result = timeout(Duration::from_secs(10), async_op()).await;
       assert!(result.is_ok());
   }
   ```

2. **Builder Pattern** (`tests/support/graph_builders.rs`)
   ```rust
   pub struct GraphFixture {
       pub name: &'static str,
       pub graph: Arc<MemoryGraph>,
   }
   impl GraphFixture {
       pub fn with_config_adjuster<F>(mut self, f: F) -> Self { ... }
   }
   ```

3. **Arc<Mutex<>> for Shared State** (existing chaos framework)
   ```rust
   rng: Arc<Mutex<StdRng>>,
   clock: Arc<Mutex<u64>>,
   ```

### 1.3 Create NetworkSimulator Core

**File**: `engram-core/tests/network_simulator.rs`

Start with the data structures from Task 010 spec:

```rust
use std::collections::{HashMap, VecDeque};
use std::net::SocketAddr;
use std::sync::{Arc, Mutex};
use std::time::Duration;
use serde::{Serialize, Deserialize};
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;

pub struct NetworkSimulator {
    clock: Arc<Mutex<u64>>,  // Logical time in ms
    message_queues: Arc<Mutex<HashMap<SocketAddr, VecDeque<PendingMessage>>>>,
    fault_injectors: Arc<Mutex<HashMap<(SocketAddr, SocketAddr), FaultInjector>>>,
    event_log: Arc<Mutex<Vec<NetworkEvent>>>,
    rng: Arc<Mutex<StdRng>>,
    config: SimulatorConfig,
}

#[derive(Clone)]
pub struct SimulatorConfig {
    pub base_latency: u64,
    pub latency_jitter: u64,
    pub default_packet_loss: f64,
    pub enable_recording: bool,
    pub seed: u64,
}

impl Default for SimulatorConfig {
    fn default() -> Self {
        Self {
            base_latency: 10,
            latency_jitter: 5,
            default_packet_loss: 0.0,
            enable_recording: true,
            seed: 42,
        }
    }
}
```

**Key Methods to Implement**:

1. `new(config) -> Self` - Constructor
2. `now() -> u64` - Get simulated time
3. `advance_time(duration_ms)` - Progress simulation
4. `send(from, to, data) -> Result<()>` - Queue message for delivery
5. `recv(addr) -> Option<(Vec<u8>, SocketAddr)>` - Receive delivered message
6. `inject_fault(from, to, fault, duration)` - Add network fault
7. `clear_fault(from, to)` - Remove network fault
8. `partition(&group_a, &group_b)` - Bidirectional partition
9. `heal(&group_a, &group_b)` - Remove partition

**Critical Implementation Detail**:
- Use logical time, NOT wall-clock time
- Message delivery is deterministic based on time advancement
- Same seed = identical packet drops and delivery order

---

## Phase 2: Scenario DSL & Orchestration (Day 1-2)

### 2.1 Create Scenario Definition Language

**File**: `engram-core/tests/chaos/scenario.rs`

```rust
pub struct ChaosScenario {
    pub name: String,
    pub num_nodes: usize,
    pub duration: Duration,
    pub faults: Vec<ScheduledFault>,
    pub operations: Vec<Operation>,
    pub invariants: Vec<Invariant>,
}

pub struct ScheduledFault {
    pub at: Duration,
    pub fault: FaultSpec,
    pub duration: Option<Duration>,
}

pub enum FaultSpec {
    Partition { group_a: Vec<usize>, group_b: Vec<usize> },
    AsymmetricPartition { from_group: Vec<usize>, to_group: Vec<usize> },
    KillNode { node: usize },
    AddLatency { between: Vec<(usize, usize)>, latency_ms: u64, jitter_ms: Option<u64> },
    PacketLoss { between: Vec<(usize, usize)>, rate: f64 },
}

pub enum Invariant {
    EventualConsistency { within: Duration },
    NoDataLoss,
    NoSplitBrain,
    AvailabilityThreshold { min_nodes: usize },
}

pub struct ChaosScenarioBuilder {
    scenario: ChaosScenario,
}

impl ChaosScenarioBuilder {
    pub fn new(name: &str) -> Self { ... }
    pub fn nodes(mut self, num: usize) -> Self { ... }
    pub fn duration(mut self, duration: Duration) -> Self { ... }
    pub fn inject_fault(mut self, at: Duration, fault: FaultSpec, duration: Option<Duration>) -> Self { ... }
    pub fn invariant(mut self, invariant: Invariant) -> Self { ... }
    pub fn build(self) -> ChaosScenario { ... }
}
```

### 2.2 Create Test Orchestrator

**File**: `engram-core/tests/chaos/orchestrator.rs`

```rust
pub struct ChaosOrchestrator {
    simulator: Arc<NetworkSimulator>,
    nodes: HashMap<usize, TestNode>,
}

pub struct TestNode {
    id: usize,
    addr: SocketAddr,
    simulator: Arc<NetworkSimulator>,
    // Will contain SwimMembership when Task 003 is done
}

impl ChaosOrchestrator {
    pub fn new(scenario: &ChaosScenario) -> Self {
        let config = SimulatorConfig { seed: 42, ..Default::default() };
        let simulator = Arc::new(NetworkSimulator::new(config));
        
        let mut nodes = HashMap::new();
        for i in 0..scenario.num_nodes {
            let addr = format!("127.0.0.1:{}", 7946 + i).parse().unwrap();
            let node = TestNode::new(i, addr, simulator.clone());
            nodes.insert(i, node);
        }
        
        Self { simulator, nodes }
    }
    
    pub async fn run(&mut self, scenario: &ChaosScenario) -> ChaosTestResult {
        // 1. Start all nodes
        // 2. Schedule events from scenario
        // 3. Advance time in 100ms increments
        // 4. Execute scheduled faults/operations
        // 5. Verify invariants at end
        // 6. Return results
    }
}

pub struct ChaosTestResult {
    pub scenario_name: String,
    pub events: Vec<String>,
    pub invariant_checks: Vec<InvariantCheckResult>,
}

impl ChaosTestResult {
    pub fn passed(&self) -> bool {
        self.invariant_checks.iter().all(|c| c.passed)
    }
}
```

---

## Phase 3: Concrete Test Scenarios (Day 2)

### 3.1 Implement 5 Test Scenarios

**File**: `engram-core/tests/partition_scenarios.rs`

Each scenario follows the same pattern:

```rust
#[tokio::test]
async fn test_clean_partition_3_2_split() {
    let scenario = ChaosScenario::builder("clean_partition")
        .nodes(5)
        .duration(Duration::from_secs(60))
        .inject_fault(
            Duration::from_secs(10),
            FaultSpec::Partition {
                group_a: vec![0, 1, 2],
                group_b: vec![3, 4],
            },
            Some(Duration::from_secs(20)),
        )
        .invariant(Invariant::EventualConsistency {
            within: Duration::from_secs(30),
        })
        .invariant(Invariant::NoDataLoss)
        .build();
    
    let mut orchestrator = ChaosOrchestrator::new(&scenario);
    let result = orchestrator.run(&scenario).await;
    
    assert!(result.passed(), "Test failed: {:?}", result.invariant_checks);
}
```

**The 5 Scenarios**:

1. **Clean Partition (3|2 split)**
   - At t=10s: Partition into 3-node majority and 2-node minority
   - At t=30s: Heal partition
   - Verify: All nodes converge within 30s after healing

2. **Asymmetric Partition**
   - Node 0 can send to Node 1, but Node 1 cannot send back
   - Verifies heartbeat timeout logic
   - Verifies no data loss despite asymmetry

3. **Flapping Partition**
   - 10 rapid partition/heal cycles (3s apart)
   - Detects race conditions in healing code
   - Verifies eventual convergence

4. **Cascading Failures**
   - Nodes fail sequentially (not simultaneously)
   - Tests graceful degradation as cluster shrinks
   - Verifies cluster remains functional with 4/7 nodes

5. **Network Congestion**
   - 500ms latency + 30% packet loss (10-30s)
   - Tests behavior under realistic WAN conditions
   - Verifies confidence bounds adjust appropriately

---

## Phase 4: Network Transport Abstraction (Day 2-3)

### 4.1 Create Transport Trait

**File**: `engram-core/src/cluster/transport.rs`

```rust
use std::net::SocketAddr;

pub trait NetworkTransport: Send + Sync {
    async fn send(&self, from: SocketAddr, to: SocketAddr, data: Vec<u8>) -> Result<(), TransportError>;
    async fn recv(&self, addr: SocketAddr) -> Option<(Vec<u8>, SocketAddr)>;
}

#[derive(Debug, thiserror::Error)]
pub enum TransportError {
    #[error("Network error: {0}")]
    NetworkError(String),
    
    #[error("Timeout")]
    Timeout,
}

// Real UDP implementation (for future production use)
pub struct RealUdpTransport {
    // Will contain actual UDP socket implementation
}

// Test implementation (uses NetworkSimulator)
pub struct SimulatedTransport {
    addr: SocketAddr,
    simulator: Arc<NetworkSimulator>,
}

impl SimulatedTransport {
    pub fn new(addr: SocketAddr, simulator: Arc<NetworkSimulator>) -> Self {
        Self { addr, simulator }
    }
}

#[async_trait::async_trait]
impl NetworkTransport for SimulatedTransport {
    async fn send(&self, from: SocketAddr, to: SocketAddr, data: Vec<u8>) -> Result<(), TransportError> {
        self.simulator.send(from, to, data)
            .map_err(|e| TransportError::NetworkError(e.to_string()))
    }
    
    async fn recv(&self, addr: SocketAddr) -> Option<(Vec<u8>, SocketAddr)> {
        self.simulator.recv(addr)
    }
}
```

### 4.2 Use Transport in Tests

The NetworkSimulator uses SimulatedTransport to test SWIM membership:

```rust
#[tokio::test]
async fn test_swim_under_partition() {
    let sim = Arc::new(NetworkSimulator::new(Default::default()));
    
    let nodes: Vec<_> = (0..5)
        .map(|i| {
            let addr = format!("127.0.0.1:{}", 7946 + i).parse().unwrap();
            let transport = SimulatedTransport::new(addr, sim.clone());
            // SwimMembership::new_with_transport("node{i}", addr, transport)
            // (This will be implemented in Task 003)
        })
        .collect();
    
    // Let cluster form
    sim.advance_time(5000);
    
    // Test partition behavior
    // ...
}
```

---

## Phase 5: Testing & CI Integration (Day 3)

### 5.1 Unit Tests for Simulator

**File**: `engram-core/tests/network_simulator.rs` (add to existing)

```rust
#[test]
fn test_simulator_determinism() {
    let config1 = SimulatorConfig { seed: 42, ..Default::default() };
    let config2 = SimulatorConfig { seed: 42, ..Default::default() };
    
    let sim1 = NetworkSimulator::new(config1);
    let sim2 = NetworkSimulator::new(config2);
    
    let addr1 = "127.0.0.1:1".parse().unwrap();
    let addr2 = "127.0.0.1:2".parse().unwrap();
    
    // Inject same packet loss
    sim1.inject_fault(addr1, addr2, FaultType::PacketLoss { rate: 0.5 }, None).unwrap();
    sim2.inject_fault(addr1, addr2, FaultType::PacketLoss { rate: 0.5 }, None).unwrap();
    
    // Send 100 packets
    for i in 0..100 {
        sim1.send(addr1, addr2, vec![i]).unwrap();
        sim2.send(addr1, addr2, vec![i]).unwrap();
    }
    
    // Should drop same packets
    let dropped1 = sim1.export_events().iter()
        .filter(|e| matches!(e, NetworkEvent::MessageDropped { .. }))
        .count();
    let dropped2 = sim2.export_events().iter()
        .filter(|e| matches!(e, NetworkEvent::MessageDropped { .. }))
        .count();
    
    assert_eq!(dropped1, dropped2, "Determinism violated!");
}

#[test]
fn test_message_delivery_ordering() {
    let sim = Arc::new(NetworkSimulator::new(Default::default()));
    
    let addr1 = "127.0.0.1:1".parse().unwrap();
    let addr2 = "127.0.0.1:2".parse().unwrap();
    
    // Send 3 messages in order
    sim.send(addr1, addr2, vec![1]).unwrap();
    sim.send(addr1, addr2, vec![2]).unwrap();
    sim.send(addr1, addr2, vec![3]).unwrap();
    
    // Deliver all messages
    sim.advance_time(1000);
    
    // Verify delivery order
    let msg1 = sim.recv(addr2);
    let msg2 = sim.recv(addr2);
    let msg3 = sim.recv(addr2);
    
    assert_eq!(msg1.map(|(d, _)| d[0]), Some(1));
    assert_eq!(msg2.map(|(d, _)| d[0]), Some(2));
    assert_eq!(msg3.map(|(d, _)| d[0]), Some(3));
}
```

### 5.2 Add Makefile Targets

**File**: `Makefile` (add these targets)

```makefile
.PHONY: chaos-test chaos-test-quick chaos-test-full

chaos-test-quick:
	cargo test --package engram-core --test partition_scenarios test_clean_partition -- --nocapture

chaos-test:
	cargo test --package engram-core --test partition_scenarios -- --nocapture --test-threads=1

chaos-test-full:
	CHAOS_TEST_DURATION=300 cargo test --package engram-core --test partition_scenarios -- --nocapture --ignored
```

### 5.3 Create CI Integration Script

**File**: `scripts/run_chaos_tests.sh`

```bash
#!/bin/bash
set -e

echo "=== Engram Chaos Testing Suite ==="

# Run quick chaos tests
echo "Running chaos tests..."
cargo test --package engram-core --test partition_scenarios --release -- --nocapture

# Check for failures
if [ $? -ne 0 ]; then
    echo "ERROR: Chaos tests failed!"
    exit 1
fi

echo "All chaos tests passed!"
```

---

## Phase 6: Documentation

### 6.1 Update Chaos Module Docs

**File**: `engram-core/tests/chaos/mod.rs` (update exports)

```rust
pub mod fault_injector;
pub mod streaming_chaos;    // If implementing streaming tests
pub mod validators;
pub mod scenario;           // NEW
pub mod orchestrator;       // NEW
pub mod replay;             // NEW (optional for deterministic replay)

pub use scenario::{ChaosScenario, ChaosScenarioBuilder, FaultSpec};
pub use orchestrator::{ChaosOrchestrator, ChaosTestResult};
```

### 6.2 Create User Guide

**File**: `docs/guide/chaos-testing-quickstart.md`

```markdown
# Chaos Testing Quick Start

## Running the 5 Scenarios

```bash
# Run all scenarios
cargo test --test partition_scenarios

# Run specific scenario
cargo test test_clean_partition_3_2_split

# Run with output
cargo test --test partition_scenarios -- --nocapture
```

## Writing Your Own Scenario

```rust
let scenario = ChaosScenario::builder("my_scenario")
    .nodes(5)
    .duration(Duration::from_secs(60))
    .inject_fault(
        Duration::from_secs(10),
        FaultSpec::Partition { ... },
        Some(Duration::from_secs(20)),
    )
    .invariant(Invariant::EventualConsistency { ... })
    .build();

let mut orchestrator = ChaosOrchestrator::new(&scenario);
let result = orchestrator.run(&scenario).await;
assert!(result.passed());
```

## Debugging a Failure

1. Get the failure seed: `grep "seed:" test_output.log`
2. Modify scenario to use that seed in config
3. Run scenario again - should reproduce exactly
4. Check event log: `result.simulator.export_events()`
```

---

## Summary: File Checklist

### Create (New Files)
- [ ] `engram-core/tests/network_simulator.rs` (400 lines)
- [ ] `engram-core/tests/chaos/scenario.rs` (200 lines)
- [ ] `engram-core/tests/chaos/orchestrator.rs` (300 lines)
- [ ] `engram-core/tests/partition_scenarios.rs` (300 lines)
- [ ] `engram-core/src/cluster/mod.rs` (20 lines)
- [ ] `engram-core/src/cluster/transport.rs` (100 lines)
- [ ] `scripts/run_chaos_tests.sh` (30 lines)
- [ ] `docs/guide/chaos-testing-quickstart.md` (100 lines)

### Modify (Existing Files)
- [ ] `engram-core/tests/chaos/mod.rs` - Add scenario, orchestrator exports
- [ ] `engram-core/src/lib.rs` - Add `pub mod cluster`
- [ ] `Makefile` - Add chaos-test targets

### Total Implementation Effort
- ~1,500 lines of Rust code
- ~130 lines of shell/docs
- 3-4 days for experienced implementer

---

## Key Success Criteria

1. **All 5 scenarios pass consistently** - Same seed = same results
2. **No data loss** under partition scenarios
3. **Deterministic replay works** - Recorded events replay perfectly
4. **Performance targets met**:
   - 5-node cluster 60s test: <5s wall-clock
   - 100-node cluster 60s test: <30s wall-clock
5. **All clippy warnings removed** - `make quality` passes
6. **Documented** - User guide and API docs complete

---

## Critical Implementation Notes

1. **Logical Time, Not Wall Clock**
   - Use `clock: Arc<Mutex<u64>>` for milliseconds
   - Increment on `advance_time()`, not with real time
   - Ensures determinism

2. **Seeded RNG for Reproducibility**
   - Use `StdRng::seed_from_u64()` 
   - Same seed = same packet drops, same latencies
   - Store seed in `SimulatorConfig`

3. **Message Ordering Matters**
   - Deliver in order when multiple messages ready
   - Track message IDs for debugging
   - Record all events for replay

4. **Lock Management**
   - Always `drop(lock)` before operations that might acquire locks
   - Use scoped blocks: `{ let mut lock = ...; } // lock dropped here`
   - Prevents deadlocks

5. **Test Isolation**
   - Each test gets fresh simulator instance
   - No shared state between tests
   - Use Arc to share within single test

---

## Next Steps After Task 010

- Task 003: SWIM membership protocol with NetworkTransport
- Task 011: Jepsen-style consistency validation using these tests
- Task 012: Runbook with chaos testing procedures for production

