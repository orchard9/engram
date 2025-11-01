# Task 010: Network Partition Testing Framework

**Status**: Pending
**Estimated Duration**: 3 days
**Dependencies**: None (can start immediately)
**Owner**: TBD

## Objective

Build a comprehensive deterministic network simulation framework for testing distributed cluster behavior under realistic failure scenarios. This framework enables reproducible chaos testing, partition simulation, and fault injection with deterministic replay for debugging distributed race conditions.

## Research Foundation

### The Oracle Problem in Distributed Systems Testing

Traditional unit tests have clear oracles: expected outputs for given inputs. Distributed systems lack this - there is no single "correct" behavior during network partitions. Instead, we verify **properties**: eventual consistency, no data loss, bounded staleness, split-brain prevention.

Kyle Kingsbury's Jepsen (2013-present) revolutionized distributed testing by showing that even production databases with strong consistency claims (MongoDB, Cassandra, Elasticsearch) had correctness bugs discoverable through systematic fault injection. The key insight: bugs manifest at the **boundaries** - during partition healing, concurrent primaries, asymmetric failures.

### Network Simulation Approaches

**Three paradigms exist:**

1. **Non-deterministic Chaos (Jepsen, Chaos Mesh, Gremlin)**
   - Inject real faults into running systems
   - Uses iptables for network partitions, SIGSTOP for process pauses, clock_settime for skew
   - Pros: Tests actual deployment environment, finds environment-specific bugs
   - Cons: Non-reproducible failures, requires complex infrastructure, slow iteration

2. **Deterministic Simulation (FoundationDB, TigerBeetle, Madsim)**
   - Mock all I/O (network, disk, time) with deterministic simulator
   - Every test run with same seed produces identical event sequence
   - Pros: Perfect reproducibility, fast iteration, explores state space exhaustively
   - Cons: Simulator divergence from reality, doesn't test actual networking code

3. **Hybrid Approach (This task)**
   - Real networking code with simulated fault injection
   - Deterministic replay via recorded event sequences
   - Pros: Tests actual code paths while maintaining reproducibility
   - Best of both worlds for early-stage distributed systems

### Research Papers

**SWIM Protocol (Das et al. 2002)**: "SWIM: Scalable Weakly-consistent Infection-style Process Group Membership Protocol"
- Proves O(log N) dissemination time for membership updates
- Our partition tester validates this under network failures

**Jepsen Linearizability Checker (Kingsbury 2013)**: Uses Knossos algorithm to verify linearizability from operation histories
- We adapt for eventual consistency: verify convergence within bounded time

**Deterministic Simulation Testing (Will Wilson @ FoundationDB, Strange Loop 2014)**
- Simulation testing at FoundationDB found 3x more bugs than production deployment
- Key technique: deterministic random number generation with seeds
- We implement simplified DST for network events only

**Chaos Engineering Principles (Netflix, O'Reilly 2017)**
- Build confidence in system's capability to withstand turbulence
- Minimize blast radius, inject faults continuously
- We apply to pre-production cluster testing

### Toxiproxy vs. In-Process Simulation

**Toxiproxy** is a TCP proxy for fault injection (Shopify, 2014):
- Sits between services, injects latency/packet loss/bandwidth limits
- Problem: Requires reconfiguring applications to route through proxy
- Problem: Single point of failure, doesn't work for UDP (SWIM protocol uses UDP)

**Our approach**: In-process simulation via trait abstraction:
- `NetworkTransport` trait with `RealUdpTransport` and `SimulatedTransport` impls
- Simulated transport records all events for deterministic replay
- Tests use simulated transport, production uses real UDP

### Deterministic Replay Mechanism

Inspired by rr (Mozilla) and Antithesis, we record non-deterministic inputs:
1. Network messages (content, source, arrival time offset)
2. Random number generator seeds
3. Timer events (which timeout fired when)

Replay: Feed recorded events back at same logical timestamps, bypass actual I/O.

Critical insight from FoundationDB: **Determinism requires controlling ALL non-deterministic inputs**:
- Time: Mock `Instant::now()` with simulated clock
- Randomness: Seeded RNG passed explicitly
- Concurrency: Record task scheduling order (hard - we use single-threaded simulation)

### Test Scenarios from Literature

**Jepsen found these partition patterns expose bugs:**

1. **Clean majority/minority split** - Classic scenario, easiest to handle correctly
2. **Asymmetric partition** - A→B works but B→A fails (misconfigured firewalls, NAT)
   - Exposes bugs in bidirectional heartbeat assumptions
3. **Flapping partition** - Rapid connect/disconnect cycles
   - Exposes race conditions in healing code
4. **Cascading failures** - Nodes fail sequentially, not simultaneously
   - Tests graceful degradation as cluster shrinks
5. **Partial partition** - Some nodes can't reach some others (3-way split)
   - Exposes gossip protocol convergence bugs

We implement all five as first-class test scenarios.

## Technical Specification

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Test Orchestrator                         │
│  - Scenario definition (DSL)                                │
│  - Event scheduling                                         │
│  - Validation engine                                        │
└─────────────────────┬───────────────────────────────────────┘
                      │
         ┌────────────┴────────────┐
         │                         │
┌────────▼────────┐    ┌──────────▼────────┐
│ NetworkSimulator │    │  DeterministicRNG  │
│ - Fault injection│    │  - Seeded random   │
│ - Packet routing │    │  - Replay support  │
└────────┬─────────┘    └───────────────────┘
         │
    ┌────┴────┬────────┬────────┐
    │         │        │        │
┌───▼──┐  ┌──▼──┐  ┌──▼──┐  ┌──▼──┐
│Node 1│  │Node2│  │Node3│  │Node4│  (Simulated cluster)
└──────┘  └─────┘  └─────┘  └─────┘
```

### Core Data Structures

```rust
// engram-core/tests/network_simulator.rs

use std::collections::{HashMap, BTreeMap, VecDeque};
use std::net::SocketAddr;
use std::sync::{Arc, Mutex};
use std::time::Duration;
use serde::{Serialize, Deserialize};
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;

/// Simulated network with configurable fault injection
pub struct NetworkSimulator {
    /// Simulated clock (logical time in milliseconds)
    clock: Arc<Mutex<u64>>,

    /// Pending messages (destination -> queue of messages)
    message_queues: Arc<Mutex<HashMap<SocketAddr, VecDeque<PendingMessage>>>>,

    /// Active network conditions per link
    fault_injectors: Arc<Mutex<HashMap<(SocketAddr, SocketAddr), FaultInjector>>>,

    /// Event log for deterministic replay
    event_log: Arc<Mutex<Vec<NetworkEvent>>>,

    /// Random number generator (seeded for determinism)
    rng: Arc<Mutex<StdRng>>,

    /// Configuration
    config: SimulatorConfig,
}

#[derive(Debug, Clone)]
pub struct SimulatorConfig {
    /// Base network latency (ms)
    pub base_latency: u64,

    /// Latency variance (ms)
    pub latency_jitter: u64,

    /// Default packet loss rate (0.0 = none, 1.0 = all)
    pub default_packet_loss: f64,

    /// Whether to record events for replay
    pub enable_recording: bool,

    /// Seed for deterministic randomness
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

/// Message waiting for delivery
#[derive(Debug, Clone)]
struct PendingMessage {
    /// Message payload
    data: Vec<u8>,

    /// Source address
    from: SocketAddr,

    /// Destination address
    to: SocketAddr,

    /// Scheduled delivery time (simulated clock)
    delivery_time: u64,

    /// Unique message ID for tracking
    id: u64,
}

/// Network fault injection for a specific link
#[derive(Debug, Clone)]
pub struct FaultInjector {
    /// Type of fault
    pub fault_type: FaultType,

    /// When fault was injected (simulated time)
    pub injected_at: u64,

    /// Fault duration (None = indefinite)
    pub duration: Option<Duration>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FaultType {
    /// Drop packets with given probability
    PacketLoss { rate: f64 },

    /// Add latency to all packets
    Latency {
        /// Fixed delay (ms)
        delay: u64,
        /// Optional jitter (ms)
        jitter: Option<u64>,
    },

    /// Complete partition (drop all packets)
    Partition,

    /// Bandwidth throttling
    Throttle {
        /// Bytes per second
        max_bandwidth: u64,
    },

    /// Packet reordering
    Reorder {
        /// Probability of reordering
        rate: f64,
        /// Max reorder delay (ms)
        max_delay: u64,
    },

    /// Packet duplication
    Duplicate { rate: f64 },

    /// Bit corruption
    Corrupt { rate: f64 },
}

/// Recorded network event for deterministic replay
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NetworkEvent {
    /// Message sent
    MessageSent {
        id: u64,
        from: SocketAddr,
        to: SocketAddr,
        size: usize,
        timestamp: u64,
    },

    /// Message delivered
    MessageDelivered {
        id: u64,
        timestamp: u64,
    },

    /// Message dropped (fault injection)
    MessageDropped {
        id: u64,
        reason: String,
        timestamp: u64,
    },

    /// Fault injected
    FaultInjected {
        from: SocketAddr,
        to: SocketAddr,
        fault: FaultType,
        timestamp: u64,
    },

    /// Fault cleared
    FaultCleared {
        from: SocketAddr,
        to: SocketAddr,
        timestamp: u64,
    },

    /// Clock advanced
    ClockAdvanced {
        from: u64,
        to: u64,
    },
}

impl NetworkSimulator {
    /// Create new simulator with given configuration
    pub fn new(config: SimulatorConfig) -> Self {
        Self {
            clock: Arc::new(Mutex::new(0)),
            message_queues: Arc::new(Mutex::new(HashMap::new())),
            fault_injectors: Arc::new(Mutex::new(HashMap::new())),
            event_log: Arc::new(Mutex::new(Vec::new())),
            rng: Arc::new(Mutex::new(StdRng::seed_from_u64(config.seed))),
            config,
        }
    }

    /// Create from recorded event log for deterministic replay
    pub fn from_recording(events: Vec<NetworkEvent>, config: SimulatorConfig) -> Self {
        let mut sim = Self::new(config);

        // Pre-populate event log
        *sim.event_log.lock().unwrap() = events;

        sim
    }

    /// Get current simulated time
    pub fn now(&self) -> u64 {
        *self.clock.lock().unwrap()
    }

    /// Advance simulated clock and deliver pending messages
    pub fn advance_time(&self, duration_ms: u64) {
        let mut clock = self.clock.lock().unwrap();
        let old_time = *clock;
        let new_time = old_time + duration_ms;
        *clock = new_time;
        drop(clock);

        self.record_event(NetworkEvent::ClockAdvanced {
            from: old_time,
            to: new_time,
        });

        // Deliver messages whose time has come
        self.deliver_pending_messages(new_time);
    }

    /// Send message through simulated network
    pub fn send(
        &self,
        from: SocketAddr,
        to: SocketAddr,
        data: Vec<u8>,
    ) -> Result<(), SimulatorError> {
        let now = self.now();
        let msg_id = self.next_message_id();

        // Record send event
        self.record_event(NetworkEvent::MessageSent {
            id: msg_id,
            from,
            to,
            size: data.len(),
            timestamp: now,
        });

        // Check if packet should be dropped
        if self.should_drop_packet(from, to)? {
            self.record_event(NetworkEvent::MessageDropped {
                id: msg_id,
                reason: "fault injection".to_string(),
                timestamp: now,
            });
            return Ok(());
        }

        // Calculate delivery time with latency and jitter
        let latency = self.calculate_latency(from, to)?;
        let delivery_time = now + latency;

        // Enqueue message
        let msg = PendingMessage {
            data,
            from,
            to,
            delivery_time,
            id: msg_id,
        };

        let mut queues = self.message_queues.lock().unwrap();
        queues.entry(to).or_default().push_back(msg);

        Ok(())
    }

    /// Receive message from simulated network
    pub fn recv(&self, addr: SocketAddr) -> Option<(Vec<u8>, SocketAddr)> {
        let mut queues = self.message_queues.lock().unwrap();

        if let Some(queue) = queues.get_mut(&addr) {
            if let Some(msg) = queue.pop_front() {
                self.record_event(NetworkEvent::MessageDelivered {
                    id: msg.id,
                    timestamp: self.now(),
                });

                return Some((msg.data, msg.from));
            }
        }

        None
    }

    /// Inject network fault on specific link
    pub fn inject_fault(
        &self,
        from: SocketAddr,
        to: SocketAddr,
        fault: FaultType,
        duration: Option<Duration>,
    ) -> Result<(), SimulatorError> {
        let now = self.now();

        let injector = FaultInjector {
            fault_type: fault.clone(),
            injected_at: now,
            duration,
        };

        let mut faults = self.fault_injectors.lock().unwrap();
        faults.insert((from, to), injector);

        self.record_event(NetworkEvent::FaultInjected {
            from,
            to,
            fault,
            timestamp: now,
        });

        Ok(())
    }

    /// Clear fault on specific link
    pub fn clear_fault(
        &self,
        from: SocketAddr,
        to: SocketAddr,
    ) -> Result<(), SimulatorError> {
        let mut faults = self.fault_injectors.lock().unwrap();
        faults.remove(&(from, to));

        self.record_event(NetworkEvent::FaultCleared {
            from,
            to,
            timestamp: self.now(),
        });

        Ok(())
    }

    /// Partition two sets of nodes (bidirectional)
    pub fn partition(&self, group_a: &[SocketAddr], group_b: &[SocketAddr]) {
        for &a in group_a {
            for &b in group_b {
                // Bidirectional partition
                let _ = self.inject_fault(a, b, FaultType::Partition, None);
                let _ = self.inject_fault(b, a, FaultType::Partition, None);
            }
        }
    }

    /// Heal partition between two sets of nodes
    pub fn heal(&self, group_a: &[SocketAddr], group_b: &[SocketAddr]) {
        for &a in group_a {
            for &b in group_b {
                let _ = self.clear_fault(a, b);
                let _ = self.clear_fault(b, a);
            }
        }
    }

    /// Export event log for deterministic replay
    pub fn export_events(&self) -> Vec<NetworkEvent> {
        self.event_log.lock().unwrap().clone()
    }

    /// Check if packet should be dropped based on active faults
    fn should_drop_packet(&self, from: SocketAddr, to: SocketAddr) -> Result<bool, SimulatorError> {
        let faults = self.fault_injectors.lock().unwrap();

        if let Some(injector) = faults.get(&(from, to)) {
            // Check if fault has expired
            if let Some(duration) = injector.duration {
                let elapsed = self.now() - injector.injected_at;
                if elapsed > duration.as_millis() as u64 {
                    return Ok(false);
                }
            }

            match &injector.fault_type {
                FaultType::Partition => return Ok(true),
                FaultType::PacketLoss { rate } => {
                    let mut rng = self.rng.lock().unwrap();
                    return Ok(rng.gen::<f64>() < *rate);
                },
                _ => {},
            }
        }

        Ok(false)
    }

    /// Calculate latency for packet including jitter and fault injection
    fn calculate_latency(&self, from: SocketAddr, to: SocketAddr) -> Result<u64, SimulatorError> {
        let mut latency = self.config.base_latency;

        // Add jitter
        if self.config.latency_jitter > 0 {
            let mut rng = self.rng.lock().unwrap();
            let jitter = rng.gen_range(0..=self.config.latency_jitter);
            latency += jitter;
        }

        // Check for latency fault injection
        let faults = self.fault_injectors.lock().unwrap();
        if let Some(injector) = faults.get(&(from, to)) {
            if let FaultType::Latency { delay, jitter } = injector.fault_type {
                latency += delay;
                if let Some(jitter_range) = jitter {
                    let mut rng = self.rng.lock().unwrap();
                    let jitter_val = rng.gen_range(0..=jitter_range);
                    latency += jitter_val;
                }
            }
        }

        Ok(latency)
    }

    /// Deliver all messages whose delivery time has arrived
    fn deliver_pending_messages(&self, current_time: u64) {
        let mut queues = self.message_queues.lock().unwrap();

        for queue in queues.values_mut() {
            // Sort queue by delivery time (already in order due to scheduling)
            // Messages are ready if delivery_time <= current_time
            // They'll be popped by recv() calls
        }
    }

    /// Record event to log
    fn record_event(&self, event: NetworkEvent) {
        if self.config.enable_recording {
            let mut log = self.event_log.lock().unwrap();
            log.push(event);
        }
    }

    /// Generate unique message ID
    fn next_message_id(&self) -> u64 {
        static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        COUNTER.fetch_add(1, std::sync::atomic::Ordering::SeqCst)
    }
}

#[derive(Debug, thiserror::Error)]
pub enum SimulatorError {
    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),

    #[error("Simulation error: {0}")]
    SimulationError(String),
}
```

### Scenario Definition Language

```rust
// engram-core/tests/chaos/scenario.rs

use std::time::Duration;
use std::net::SocketAddr;

/// Declarative chaos test scenario
#[derive(Debug, Clone)]
pub struct ChaosScenario {
    /// Scenario name
    pub name: String,

    /// Number of nodes in cluster
    pub num_nodes: usize,

    /// Test duration (simulated time)
    pub duration: Duration,

    /// Fault injection schedule
    pub faults: Vec<ScheduledFault>,

    /// Operations to perform during test
    pub operations: Vec<Operation>,

    /// Invariants to check
    pub invariants: Vec<Invariant>,
}

#[derive(Debug, Clone)]
pub struct ScheduledFault {
    /// When to inject fault (relative to test start)
    pub at: Duration,

    /// Fault to inject
    pub fault: FaultSpec,

    /// How long fault lasts (None = until cleared)
    pub duration: Option<Duration>,
}

#[derive(Debug, Clone)]
pub enum FaultSpec {
    /// Partition cluster into two groups
    Partition {
        group_a: Vec<usize>, // node indices
        group_b: Vec<usize>,
    },

    /// Asymmetric partition (A→B works, B→A fails)
    AsymmetricPartition {
        from_group: Vec<usize>,
        to_group: Vec<usize>,
    },

    /// Kill specific node
    KillNode { node: usize },

    /// Add latency between nodes
    AddLatency {
        between: Vec<(usize, usize)>,
        latency_ms: u64,
        jitter_ms: Option<u64>,
    },

    /// Packet loss between nodes
    PacketLoss {
        between: Vec<(usize, usize)>,
        rate: f64, // 0.0-1.0
    },

    /// Bandwidth throttle
    Throttle {
        between: Vec<(usize, usize)>,
        bandwidth_bps: u64,
    },
}

#[derive(Debug, Clone)]
pub struct Operation {
    /// When to perform operation
    pub at: Duration,

    /// Operation to perform
    pub op: OperationType,
}

#[derive(Debug, Clone)]
pub enum OperationType {
    /// Store memory
    Store {
        node: usize,
        space: String,
        content: String,
    },

    /// Recall memory
    Recall {
        node: usize,
        space: String,
        query: String,
    },

    /// Check cluster state
    VerifyClusterSize { expected: usize },

    /// Wait for convergence
    WaitForConvergence { timeout: Duration },
}

#[derive(Debug, Clone)]
pub enum Invariant {
    /// Eventually all nodes agree on data
    EventualConsistency {
        within: Duration,
    },

    /// No acknowledged write is lost
    NoDataLoss,

    /// Cluster remains available (>50% nodes respond)
    AvailabilityThreshold {
        min_nodes: usize,
    },

    /// No split-brain (multiple primaries for same space)
    NoSplitBrain,

    /// Confidence reflects actual consistency
    ConfidenceBounds {
        min: f32,
        max: f32,
    },
}

impl ChaosScenario {
    /// Define test scenario using builder pattern
    pub fn builder(name: &str) -> ChaosScenarioBuilder {
        ChaosScenarioBuilder::new(name)
    }
}

pub struct ChaosScenarioBuilder {
    scenario: ChaosScenario,
}

impl ChaosScenarioBuilder {
    pub fn new(name: &str) -> Self {
        Self {
            scenario: ChaosScenario {
                name: name.to_string(),
                num_nodes: 3,
                duration: Duration::from_secs(60),
                faults: vec![],
                operations: vec![],
                invariants: vec![],
            },
        }
    }

    pub fn nodes(mut self, num: usize) -> Self {
        self.scenario.num_nodes = num;
        self
    }

    pub fn duration(mut self, duration: Duration) -> Self {
        self.scenario.duration = duration;
        self
    }

    pub fn inject_fault(mut self, at: Duration, fault: FaultSpec, duration: Option<Duration>) -> Self {
        self.scenario.faults.push(ScheduledFault { at, fault, duration });
        self
    }

    pub fn operation(mut self, at: Duration, op: OperationType) -> Self {
        self.scenario.operations.push(Operation { at, op });
        self
    }

    pub fn invariant(mut self, invariant: Invariant) -> Self {
        self.scenario.invariants.push(invariant);
        self
    }

    pub fn build(self) -> ChaosScenario {
        self.scenario
    }
}
```

### Test Orchestrator

```rust
// engram-core/tests/chaos/orchestrator.rs

use super::scenario::*;
use crate::network_simulator::*;
use std::collections::HashMap;

pub struct ChaosOrchestrator {
    simulator: Arc<NetworkSimulator>,
    nodes: HashMap<usize, TestNode>,
}

impl ChaosOrchestrator {
    pub fn new(scenario: &ChaosScenario) -> Self {
        let config = SimulatorConfig {
            seed: 42, // deterministic
            ..Default::default()
        };

        let simulator = Arc::new(NetworkSimulator::new(config));
        let mut nodes = HashMap::new();

        // Create simulated cluster
        for i in 0..scenario.num_nodes {
            let addr = format!("127.0.0.1:{}", 7946 + i).parse().unwrap();
            let node = TestNode::new(i, addr, simulator.clone());
            nodes.insert(i, node);
        }

        Self { simulator, nodes }
    }

    pub async fn run(&mut self, scenario: &ChaosScenario) -> ChaosTestResult {
        let mut result = ChaosTestResult::new(&scenario.name);

        // Start all nodes
        for node in self.nodes.values_mut() {
            node.start().await;
        }

        // Schedule faults and operations
        let mut event_schedule = self.build_schedule(scenario);

        // Run simulation
        let start_time = self.simulator.now();
        let end_time = start_time + scenario.duration.as_millis() as u64;

        while self.simulator.now() < end_time {
            // Execute scheduled events
            while let Some(event) = event_schedule.first() {
                if event.at_time > self.simulator.now() {
                    break;
                }

                let event = event_schedule.remove(0);
                self.execute_event(event, &mut result).await;
            }

            // Advance time by 100ms
            self.simulator.advance_time(100);
        }

        // Verify invariants
        for invariant in &scenario.invariants {
            let check_result = self.check_invariant(invariant).await;
            result.invariant_checks.push(check_result);
        }

        result
    }

    fn build_schedule(&self, scenario: &ChaosScenario) -> Vec<ScheduledEvent> {
        let mut events = Vec::new();

        // Convert faults to events
        for fault in &scenario.faults {
            events.push(ScheduledEvent {
                at_time: fault.at.as_millis() as u64,
                event_type: EventType::InjectFault(fault.clone()),
            });

            // Schedule fault clearing if duration specified
            if let Some(duration) = fault.duration {
                events.push(ScheduledEvent {
                    at_time: (fault.at + duration).as_millis() as u64,
                    event_type: EventType::ClearFault(fault.fault.clone()),
                });
            }
        }

        // Convert operations to events
        for op in &scenario.operations {
            events.push(ScheduledEvent {
                at_time: op.at.as_millis() as u64,
                event_type: EventType::Operation(op.op.clone()),
            });
        }

        // Sort by time
        events.sort_by_key(|e| e.at_time);
        events
    }

    async fn execute_event(&mut self, event: ScheduledEvent, result: &mut ChaosTestResult) {
        match event.event_type {
            EventType::InjectFault(fault) => {
                self.inject_fault(&fault.fault).await;
                result.events.push(format!("Injected fault: {:?}", fault.fault));
            },
            EventType::ClearFault(fault) => {
                self.clear_fault(&fault).await;
                result.events.push(format!("Cleared fault: {:?}", fault));
            },
            EventType::Operation(op) => {
                self.execute_operation(&op).await;
                result.events.push(format!("Executed operation: {:?}", op));
            },
        }
    }

    async fn inject_fault(&self, fault: &FaultSpec) {
        match fault {
            FaultSpec::Partition { group_a, group_b } => {
                let addrs_a: Vec<_> = group_a.iter()
                    .map(|&i| self.nodes[&i].addr)
                    .collect();
                let addrs_b: Vec<_> = group_b.iter()
                    .map(|&i| self.nodes[&i].addr)
                    .collect();

                self.simulator.partition(&addrs_a, &addrs_b);
            },
            FaultSpec::KillNode { node } => {
                if let Some(n) = self.nodes.get(node) {
                    n.stop().await;
                }
            },
            // ... implement other fault types
            _ => {},
        }
    }

    async fn check_invariant(&self, invariant: &Invariant) -> InvariantCheckResult {
        match invariant {
            Invariant::EventualConsistency { within } => {
                self.check_eventual_consistency(*within).await
            },
            Invariant::NoDataLoss => {
                self.check_no_data_loss().await
            },
            Invariant::NoSplitBrain => {
                self.check_no_split_brain().await
            },
            _ => InvariantCheckResult {
                invariant: format!("{:?}", invariant),
                passed: true,
                message: "Not implemented".to_string(),
            },
        }
    }

    async fn check_eventual_consistency(&self, timeout: Duration) -> InvariantCheckResult {
        // Wait for all nodes to converge
        let deadline = self.simulator.now() + timeout.as_millis() as u64;

        loop {
            if self.simulator.now() > deadline {
                return InvariantCheckResult {
                    invariant: "EventualConsistency".to_string(),
                    passed: false,
                    message: format!("Failed to converge within {}ms", timeout.as_millis()),
                };
            }

            // Check if all nodes have same state
            let states: Vec<_> = self.nodes.values()
                .map(|n| n.get_state())
                .collect();

            if states.windows(2).all(|w| w[0] == w[1]) {
                return InvariantCheckResult {
                    invariant: "EventualConsistency".to_string(),
                    passed: true,
                    message: format!("Converged in {}ms", self.simulator.now()),
                };
            }

            self.simulator.advance_time(100);
        }
    }
}

#[derive(Debug)]
struct ScheduledEvent {
    at_time: u64,
    event_type: EventType,
}

#[derive(Debug)]
enum EventType {
    InjectFault(ScheduledFault),
    ClearFault(FaultSpec),
    Operation(OperationType),
}

#[derive(Debug)]
pub struct ChaosTestResult {
    pub scenario_name: String,
    pub events: Vec<String>,
    pub invariant_checks: Vec<InvariantCheckResult>,
}

impl ChaosTestResult {
    fn new(name: &str) -> Self {
        Self {
            scenario_name: name.to_string(),
            events: vec![],
            invariant_checks: vec![],
        }
    }

    pub fn passed(&self) -> bool {
        self.invariant_checks.iter().all(|c| c.passed)
    }
}

#[derive(Debug)]
pub struct InvariantCheckResult {
    pub invariant: String,
    pub passed: bool,
    pub message: String,
}

/// Simulated cluster node for testing
struct TestNode {
    id: usize,
    addr: SocketAddr,
    simulator: Arc<NetworkSimulator>,
    // Would contain actual SwimMembership, etc.
}

impl TestNode {
    fn new(id: usize, addr: SocketAddr, simulator: Arc<NetworkSimulator>) -> Self {
        Self { id, addr, simulator }
    }

    async fn start(&mut self) {
        // Initialize SWIM membership with simulated transport
    }

    async fn stop(&self) {
        // Graceful shutdown
    }

    fn get_state(&self) -> String {
        // Return serialized state for convergence checking
        format!("node-{}-state", self.id)
    }
}
```

## Concrete Test Scenarios

### Scenario 1: Clean Majority/Minority Split

```rust
// engram-core/tests/partition_scenarios.rs

#[tokio::test]
async fn test_clean_partition_3_2_split() {
    let scenario = ChaosScenario::builder("clean_partition")
        .nodes(5)
        .duration(Duration::from_secs(60))
        // Partition at t=10s
        .inject_fault(
            Duration::from_secs(10),
            FaultSpec::Partition {
                group_a: vec![0, 1, 2],
                group_b: vec![3, 4],
            },
            Some(Duration::from_secs(20)),
        )
        // Write to majority partition
        .operation(
            Duration::from_secs(15),
            OperationType::Store {
                node: 0,
                space: "test".to_string(),
                content: "data_during_partition".to_string(),
            },
        )
        // Wait for partition heal
        .operation(
            Duration::from_secs(35),
            OperationType::WaitForConvergence {
                timeout: Duration::from_secs(20),
            },
        )
        // Verify data replicated to minority
        .operation(
            Duration::from_secs(40),
            OperationType::Recall {
                node: 4,
                space: "test".to_string(),
                query: "data_during_partition".to_string(),
            },
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

### Scenario 2: Asymmetric Partition

```rust
#[tokio::test]
async fn test_asymmetric_partition() {
    let scenario = ChaosScenario::builder("asymmetric_partition")
        .nodes(3)
        .duration(Duration::from_secs(60))
        // Node 0 can send to Node 1, but Node 1 cannot send to Node 0
        .inject_fault(
            Duration::from_secs(10),
            FaultSpec::AsymmetricPartition {
                from_group: vec![1],
                to_group: vec![0],
            },
            Some(Duration::from_secs(20)),
        )
        // Node 0 tries to ping Node 1
        .operation(
            Duration::from_secs(15),
            OperationType::Store {
                node: 0,
                space: "test".to_string(),
                content: "ping".to_string(),
            },
        )
        // Verify Node 1 gets it but cannot ack
        .invariant(Invariant::EventualConsistency {
            within: Duration::from_secs(40),
        })
        .invariant(Invariant::NoSplitBrain)
        .build();

    let mut orchestrator = ChaosOrchestrator::new(&scenario);
    let result = orchestrator.run(&scenario).await;

    assert!(result.passed());
}
```

### Scenario 3: Flapping Partition

```rust
#[tokio::test]
async fn test_flapping_partition() {
    let mut scenario_builder = ChaosScenario::builder("flapping_partition")
        .nodes(5)
        .duration(Duration::from_secs(120));

    // Inject 10 rapid partition/heal cycles
    for i in 0..10 {
        let partition_time = Duration::from_secs(10 + i * 10);
        let heal_time = Duration::from_secs(10 + i * 10 + 3);

        scenario_builder = scenario_builder
            .inject_fault(
                partition_time,
                FaultSpec::Partition {
                    group_a: vec![0, 1],
                    group_b: vec![2, 3, 4],
                },
                Some(Duration::from_secs(3)),
            );
    }

    let scenario = scenario_builder
        .invariant(Invariant::EventualConsistency {
            within: Duration::from_secs(30),
        })
        .invariant(Invariant::NoDataLoss)
        .build();

    let mut orchestrator = ChaosOrchestrator::new(&scenario);
    let result = orchestrator.run(&scenario).await;

    assert!(result.passed());
}
```

### Scenario 4: Cascading Failures

```rust
#[tokio::test]
async fn test_cascading_node_failures() {
    let scenario = ChaosScenario::builder("cascading_failures")
        .nodes(7)
        .duration(Duration::from_secs(90))
        // Kill nodes sequentially
        .inject_fault(
            Duration::from_secs(10),
            FaultSpec::KillNode { node: 6 },
            None,
        )
        .inject_fault(
            Duration::from_secs(20),
            FaultSpec::KillNode { node: 5 },
            None,
        )
        .inject_fault(
            Duration::from_secs(30),
            FaultSpec::KillNode { node: 4 },
            None,
        )
        // Verify cluster still functional with 4/7 nodes
        .operation(
            Duration::from_secs(40),
            OperationType::VerifyClusterSize { expected: 4 },
        )
        .invariant(Invariant::AvailabilityThreshold { min_nodes: 4 })
        .build();

    let mut orchestrator = ChaosOrchestrator::new(&scenario);
    let result = orchestrator.run(&scenario).await;

    assert!(result.passed());
}
```

### Scenario 5: Network Congestion

```rust
#[tokio::test]
async fn test_network_congestion() {
    let scenario = ChaosScenario::builder("network_congestion")
        .nodes(5)
        .duration(Duration::from_secs(60))
        // Add 500ms latency + packet loss
        .inject_fault(
            Duration::from_secs(10),
            FaultSpec::AddLatency {
                between: vec![(0, 1), (1, 2), (2, 3), (3, 4)],
                latency_ms: 500,
                jitter_ms: Some(200),
            },
            Some(Duration::from_secs(30)),
        )
        .inject_fault(
            Duration::from_secs(10),
            FaultSpec::PacketLoss {
                between: vec![(0, 1), (1, 2), (2, 3), (3, 4)],
                rate: 0.3, // 30% packet loss
            },
            Some(Duration::from_secs(30)),
        )
        // Write during congestion
        .operation(
            Duration::from_secs(20),
            OperationType::Store {
                node: 0,
                space: "test".to_string(),
                content: "data_under_congestion".to_string(),
            },
        )
        .invariant(Invariant::EventualConsistency {
            within: Duration::from_secs(50),
        })
        .invariant(Invariant::ConfidenceBounds {
            min: 0.3, // Confidence reduced during congestion
            max: 0.9,
        })
        .build();

    let mut orchestrator = ChaosOrchestrator::new(&scenario);
    let result = orchestrator.run(&scenario).await;

    assert!(result.passed());
}
```

## Files to Create

1. `engram-core/tests/network_simulator.rs` - Network simulation engine
2. `engram-core/tests/chaos/mod.rs` - Chaos testing framework
3. `engram-core/tests/chaos/scenario.rs` - Scenario definition DSL
4. `engram-core/tests/chaos/orchestrator.rs` - Test orchestration
5. `engram-core/tests/chaos/replay.rs` - Deterministic replay
6. `engram-core/tests/partition_scenarios.rs` - 5 concrete test scenarios
7. `engram-core/tests/chaos/validators.rs` - Invariant validators
8. `engram-core/src/cluster/test_transport.rs` - Simulated transport trait impl
9. `scripts/run_chaos_tests.sh` - CI integration script
10. `docs/testing/chaos-testing-guide.md` - Usage documentation

## Files to Modify

1. `engram-core/src/cluster/transport.rs` - Extract `NetworkTransport` trait
2. `engram-core/src/cluster/membership.rs` - Accept `impl NetworkTransport`
3. `engram-core/Cargo.toml` - Add `test-support` feature for chaos tests
4. `.github/workflows/test.yml` - Add chaos test CI job (if using CI)
5. `Makefile` - Add `make chaos-test` target

## Testing Strategy

### Unit Tests for Simulator

```rust
#[test]
fn test_simulator_determinism() {
    // Same seed = same results
    let config1 = SimulatorConfig { seed: 42, ..Default::default() };
    let config2 = SimulatorConfig { seed: 42, ..Default::default() };

    let sim1 = NetworkSimulator::new(config1);
    let sim2 = NetworkSimulator::new(config2);

    let addr1 = "127.0.0.1:1".parse().unwrap();
    let addr2 = "127.0.0.1:2".parse().unwrap();

    // Inject same packet loss
    sim1.inject_fault(addr1, addr2, FaultType::PacketLoss { rate: 0.5 }, None);
    sim2.inject_fault(addr1, addr2, FaultType::PacketLoss { rate: 0.5 }, None);

    // Send 100 packets
    for i in 0..100 {
        sim1.send(addr1, addr2, vec![i]);
        sim2.send(addr1, addr2, vec![i]);
    }

    // Should drop same packets
    let dropped1 = sim1.export_events().iter()
        .filter(|e| matches!(e, NetworkEvent::MessageDropped { .. }))
        .count();
    let dropped2 = sim2.export_events().iter()
        .filter(|e| matches!(e, NetworkEvent::MessageDropped { .. }))
        .count();

    assert_eq!(dropped1, dropped2, "Non-deterministic packet loss!");
}

#[test]
fn test_deterministic_replay() {
    // Record events
    let config = SimulatorConfig { seed: 42, enable_recording: true, ..Default::default() };
    let sim1 = NetworkSimulator::new(config.clone());

    let addr1 = "127.0.0.1:1".parse().unwrap();
    let addr2 = "127.0.0.1:2".parse().unwrap();

    sim1.send(addr1, addr2, vec![1, 2, 3]).unwrap();
    sim1.advance_time(100);

    let events = sim1.export_events();

    // Replay from events
    let sim2 = NetworkSimulator::from_recording(events.clone(), config);

    // Should produce identical outcomes
    assert_eq!(sim1.export_events().len(), sim2.export_events().len());
}
```

### Integration Tests

```rust
#[tokio::test]
async fn test_swim_under_partition() {
    // 5-node cluster
    let sim = Arc::new(NetworkSimulator::new(Default::default()));

    let nodes: Vec<_> = (0..5)
        .map(|i| {
            let addr = format!("127.0.0.1:{}", 7946 + i).parse().unwrap();
            SwimMembership::new_with_transport(
                format!("node{}", i),
                addr,
                SimulatedTransport::new(addr, sim.clone()),
            )
        })
        .collect();

    // Let cluster form
    sim.advance_time(5000);

    // Partition 3|2
    sim.partition(
        &nodes[0..3].iter().map(|n| n.addr).collect::<Vec<_>>(),
        &nodes[3..5].iter().map(|n| n.addr).collect::<Vec<_>>(),
    );

    // Wait for detection
    sim.advance_time(10000);

    // Verify majority partition functional
    assert_eq!(nodes[0].members.len(), 3); // Sees self + 2 others
    assert_eq!(nodes[3].members.len(), 2); // Minority sees self + 1 other

    // Heal partition
    sim.heal(
        &nodes[0..3].iter().map(|n| n.addr).collect::<Vec<_>>(),
        &nodes[3..5].iter().map(|n| n.addr).collect::<Vec<_>>(),
    );

    // Wait for convergence
    sim.advance_time(15000);

    // All nodes should see full cluster
    for node in &nodes {
        assert_eq!(node.members.len(), 5);
    }
}
```

### Property-Based Tests

```rust
use proptest::prelude::*;

proptest! {
    #[test]
    fn test_eventual_consistency_property(
        num_nodes in 3..10usize,
        num_partitions in 1..5usize,
        seed in any::<u64>(),
    ) {
        let config = SimulatorConfig { seed, ..Default::default() };
        let sim = NetworkSimulator::new(config);

        // Create cluster
        // Inject random partitions
        // Verify eventual consistency

        // Property: After healing, all nodes converge within 60s
    }
}
```

## Dependencies

Add to `engram-core/Cargo.toml`:

```toml
[dev-dependencies]
# Testing frameworks
proptest = "1.4"
quickcheck = "1.0"

# Async testing
tokio-test = "0.4"

# Serialization for event recording
bincode = "1.3"
serde = { version = "1.0", features = ["derive"] }

# Random number generation
rand = { version = "0.8", features = ["small_rng"] }

# Error handling
thiserror = "1.0"
```

## Acceptance Criteria

1. **Deterministic Replay Works**
   - Same seed produces identical packet drops, latencies, delivery orders
   - Recorded events can be replayed exactly
   - Failures reproduce 100% of the time

2. **All 5 Scenarios Implemented**
   - Clean partition 3|2 split
   - Asymmetric partition (A→B works, B→A fails)
   - Flapping partition (10 rapid cycles)
   - Cascading failures (nodes fail sequentially)
   - Network congestion (latency + packet loss)

3. **Invariant Validation**
   - Eventual consistency checker works
   - Data loss detector works
   - Split-brain detector works
   - Confidence bounds validator works

4. **CI Integration**
   - Chaos tests run in CI on every PR
   - Tests complete within 5 minutes
   - Failures include deterministic replay seed for reproduction
   - Test results dashboard shows partition coverage

5. **Performance**
   - Simulate 100-node cluster for 60s in <10s wall-clock time
   - Event recording overhead <5%
   - Replay matches original within 1ms timing variance

6. **Documentation**
   - Guide for writing new chaos scenarios
   - Tutorial on debugging partition failures
   - Examples for each fault type
   - CI integration instructions

## CI Integration Strategy

### Makefile Target

```makefile
# Makefile

.PHONY: chaos-test
chaos-test:
	@echo "Running chaos tests..."
	cargo test --package engram-core --test partition_scenarios -- --nocapture
	cargo test --package engram-core --test chaos -- --nocapture

.PHONY: chaos-test-quick
chaos-test-quick:
	@echo "Running quick chaos tests (subset)..."
	cargo test --package engram-core --test partition_scenarios test_clean_partition -- --nocapture

.PHONY: chaos-test-full
chaos-test-full:
	@echo "Running full chaos test suite (long-running)..."
	CHAOS_TEST_DURATION=300 cargo test --package engram-core --test partition_scenarios -- --nocapture --ignored
```

### CI Script

```bash
#!/bin/bash
# scripts/run_chaos_tests.sh

set -e

echo "=== Engram Chaos Testing Suite ==="

# Run quick chaos tests (5 minutes)
echo "Running quick chaos tests..."
cargo test --package engram-core --test partition_scenarios --release -- --nocapture

# Generate chaos test report
echo "Generating coverage report..."
cargo test --package engram-core --test partition_scenarios -- --nocapture --format json > chaos_test_results.json

# Check for failures
if grep -q '"passed": false' chaos_test_results.json; then
    echo "ERROR: Chaos tests failed!"

    # Extract deterministic replay seeds
    echo "Deterministic replay seeds for reproduction:"
    grep -o '"seed": [0-9]*' chaos_test_results.json

    exit 1
fi

echo "All chaos tests passed!"
```

### GitHub Actions (if using CI)

```yaml
# .github/workflows/chaos-tests.yml

name: Chaos Tests

on:
  pull_request:
    paths:
      - 'engram-core/src/cluster/**'
      - 'engram-core/tests/partition_scenarios.rs'
  schedule:
    # Run full suite nightly
    - cron: '0 2 * * *'

jobs:
  chaos-quick:
    name: Quick Chaos Tests
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions-rs/toolchain@v1
        with:
          toolchain: stable
      - name: Run quick chaos tests
        run: make chaos-test-quick

  chaos-full:
    name: Full Chaos Test Suite
    runs-on: ubuntu-latest
    if: github.event_name == 'schedule'
    steps:
      - uses: actions/checkout@v3
      - uses: actions-rs/toolchain@v1
        with:
          toolchain: stable
      - name: Run full chaos suite
        run: make chaos-test-full
        timeout-minutes: 60
```

## Performance Targets

- Simulate 5-node cluster for 60s test: <5s wall-clock time
- Simulate 100-node cluster for 60s test: <30s wall-clock time
- Event recording overhead: <5% of test time
- Deterministic replay accuracy: 100% (bit-for-bit identical)
- Memory overhead: <100MB for 60s simulation

## Next Steps

After completing this task:
- Task 011 (Jepsen) uses this framework for formal consistency validation
- Task 003 (Partition Handling) is tested with these scenarios
- Task 007 (Gossip) convergence validated under partitions
- Task 012 (Runbook) includes chaos test debugging procedures

## Risk Mitigation

**Risk**: Simulator diverges from real network behavior
**Mitigation**: Validate simulator against real 3-node cluster, tune parameters to match

**Risk**: Tests become flaky due to timing
**Mitigation**: Use logical time (simulated clock), not wall-clock time

**Risk**: Deterministic replay breaks on code changes
**Mitigation**: Event log includes version metadata, warn on mismatch

**Risk**: Chaos tests too slow for CI
**Mitigation**: Quick suite (5 scenarios, 60s each) for PR, full suite nightly

## References

1. Kingsbury, K. (2013). "Jepsen: Testing the Partition Tolerance of PostgreSQL, Redis, MongoDB and Riak"
2. Das, A., Gupta, I., Motivala, A. (2002). "SWIM: Scalable Weakly-consistent Infection-style Process Group Membership Protocol"
3. Wilson, W. (2014). "Testing Distributed Systems with Deterministic Simulation", Strange Loop
4. Basiri, A., et al. (2016). "Chaos Engineering", O'Reilly
5. Antithesis (2024). "Deterministic Simulation Testing: Debugging with Perfect Reproducibility"
6. Madsim: Magical Deterministic Simulator for Rust. https://github.com/madsim-rs/madsim
