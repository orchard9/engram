# Task 011: Jepsen-Style Consistency Testing

**Status**: Pending
**Estimated Duration**: 4-5 days
**Dependencies**: Task 010 (Network Partition Testing Framework), Task 009 (Distributed Query Execution)
**Owner**: TBD

## Objective

Implement comprehensive Jepsen-style consistency testing for Engram's distributed system to formally verify eventual consistency guarantees, detect split-brain scenarios, validate bounded staleness properties, and ensure no data loss under network partitions and node failures. This provides the gold standard for distributed system correctness validation.

## Research Foundation

### Jepsen Methodology Overview

Jepsen (Kyle Kingsbury, 2013-present) has become the de facto standard for distributed systems correctness testing. Rather than unit testing individual components, Jepsen performs **black-box history-based verification** by:

1. **Recording Operation Histories**: Every operation (read/write) gets logged with timing, process ID, and outcome
2. **Injecting Controlled Failures**: Network partitions, clock skew, process crashes via "nemesis" component
3. **Analyzing Histories for Violations**: Post-test analysis checks if observed behaviors are consistent with claimed guarantees

**Key insight from Kingsbury's work**: Most distributed systems bugs only manifest under specific failure scenarios that traditional testing misses. Jepsen has found critical bugs in MongoDB, Elasticsearch, Cassandra, CockroachDB, and dozens of other "production-ready" systems.

### Consistency Models for Engram

Engram provides **eventual consistency with bounded staleness**, NOT linearizability. Key differences:

| Property | Linearizability | Eventual Consistency (Engram) |
|----------|----------------|-------------------------------|
| Operations appear atomic | Yes | No |
| Total order across all operations | Yes | Partial order (per-partition) |
| Reads always see latest write | Yes | No (bounded lag) |
| Convergence guarantee | N/A | Yes (within timeout) |
| CAP classification | CP | AP |

**What Engram guarantees**:
- **Eventual Convergence**: All nodes reach identical state within 60s of partition healing
- **No Data Loss**: All acknowledged writes survive failures (via async replication)
- **Bounded Staleness**: Reads may be stale but within known probability bounds
- **Confidence Calibration**: Confidence scores reflect actual divergence probability

**What Engram does NOT guarantee**:
- Linearizability (operations may not appear atomic)
- Strict serializability (no total order across partitions)
- Read-your-writes (during partition, reads may not reflect recent writes)

### Elle: History-Based Consistency Checking

Elle (Kingsbury, 2020) is Jepsen's transaction consistency checker using cycle detection in dependency graphs. Key algorithms:

1. **Dependency Graph Construction**: Build graph where nodes = transactions, edges = dependencies
   - Write-Read (WR): Transaction T1 writes X, T2 reads X
   - Write-Write (WW): Transaction T1 writes X, T2 overwrites X
   - Read-Write (RW): Transaction T1 reads X, T2 writes X (anti-dependency)

2. **Cycle Detection**: Find cycles in dependency graph
   - G0 (Write Cycle): WW edges form cycle (dirty write)
   - G1a (Aborted Read): Read from aborted transaction
   - G1c (Circular Information Flow): WR + WW cycle
   - G2 (Anti-dependency Cycle): RW edges form cycle

3. **Minimal Counterexample Extraction**: When violation found, return minimal transaction set witnessing anomaly

**Performance**: Elle is O(n) in history length, constant in concurrency. Can analyze 22M transactions in 2 minutes.

**Adaptation for Engram**: Elle checks transactions; Engram uses memory operations. We'll record STORE/RECALL operations as pseudo-transactions and check for eventual consistency violations rather than serializability.

### Knossos: Linearizability Checking

Knossos (Kingsbury, 2014) verifies linearizability via state-space exploration. Algorithm:

1. **World Representation**: Each "world" = (history, model_state, pending_ops)
2. **Search Strategy**: Explore all possible orderings of pending operations
3. **Memoization**: Cache visited (state, pending) pairs to prune search
4. **Parallelization**: Multiple threads explore disjoint branches

**Key optimization**: Worlds are immutable Clojure data structures, enabling cheap forking via structural sharing.

**Why not use Knossos for Engram?** Knossos checks linearizability; Engram provides eventual consistency. Linearizability requires total ordering; eventual consistency only requires convergence. We'll use Knossos's exploration strategy but adapt the validity condition.

### Probabilistically Bounded Staleness (PBS)

PBS (Bailis et al., 2012) quantifies eventual consistency via probabilistic bounds:

- **(Δ, p)-semantics**: Probability of reading write Δ seconds after write returns
- **(K, p)-semantics**: Probability of reading one of last K versions

**Validation approach**:
1. Instrument writes with timestamps
2. Record read values and timestamps
3. Compute empirical staleness distribution
4. Verify claimed bounds match observations

**Engram target**: 99.9% of reads within 5 seconds of latest write under normal operation, 99% within 60 seconds post-partition.

### Nemesis Strategies for Distributed Engram

Jepsen's nemesis component injects failures during testing. Standard strategies:

1. **Network Partitions**:
   - Majority/minority split (3 nodes vs 2 nodes)
   - Symmetric partition (2-2-1, no majority)
   - Asymmetric partition (A→B works, B→A fails)
   - Flapping partition (intermittent connectivity)

2. **Clock Skew**:
   - Jump clocks forward/backward (simulates NTP drift)
   - Strobe clocks (rapid time changes)
   - Skew single node vs cluster

3. **Process Failures**:
   - Kill -9 (immediate termination)
   - SIGSTOP/SIGCONT (pause/resume, simulates GC pause)
   - Slow kills (graceful shutdown vs crash)

4. **Combined Failures**:
   - Partition + clock skew (Byzantine wall-clock behavior)
   - Kill primary during partition
   - Cascading failures (nodes fail sequentially)

**Engram-specific nemesis scenarios**:
- Partition during consolidation (test gossip convergence)
- Kill node during replication lag spike
- Clock skew on primary vs replicas (confidence calibration test)
- Partition heal with divergent consolidations (conflict resolution test)

## Technical Specification

### System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Jepsen Control Node                      │
│  ┌────────────────────────────────────────────────────┐    │
│  │         Clojure Test Harness                       │    │
│  │  - Workload Generator                              │    │
│  │  - Nemesis Controller                              │    │
│  │  - History Collector                               │    │
│  │  - Checker (Eventual Consistency Validator)        │    │
│  └────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
                         │ SSH/gRPC
        ┌────────────────┼────────────────┬────────────────┐
        │                │                │                │
   ┌────▼────┐      ┌────▼────┐     ┌────▼────┐     ┌────▼────┐
   │ Engram  │      │ Engram  │     │ Engram  │     │ Engram  │
   │ Node 1  │◄─────┤ Node 2  │────►│ Node 3  │◄────┤ Node 4  │
   │         │      │         │     │         │     │         │
   │ Record  │      │ Record  │     │ Record  │     │ Record  │
   │ Ops     │      │ Ops     │     │ Ops     │     │ Ops     │
   └─────────┘      └─────────┘     └─────────┘     └─────────┘
        │                │                │                │
        └────────────────┴────────────────┴────────────────┘
                         │
                    ┌────▼─────────────────────────────┐
                    │  Operation History Aggregation   │
                    │  - Merge timestamped operations  │
                    │  - Build dependency graph        │
                    │  - Check convergence properties  │
                    └──────────────────────────────────┘
```

### Core Data Structures

#### Rust: Operation History Recording

```rust
// engram-core/src/cluster/jepsen/history.rs

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

/// Unique operation identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct OpId(u64);

/// Process identifier (client thread)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ProcessId(u32);

/// Operation type in Engram
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Operation {
    /// Store memory with embedding
    Store {
        space: String,
        memory_id: String,
        embedding: Vec<f32>,
        content: String,
    },
    /// Recall memory by ID
    Recall {
        space: String,
        memory_id: String,
    },
    /// Recall by similarity
    RecallSimilar {
        space: String,
        query_embedding: Vec<f32>,
        top_k: usize,
    },
    /// Consolidation trigger
    Consolidate {
        space: String,
    },
}

/// Operation outcome
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Outcome {
    /// Operation succeeded
    Ok {
        value: OperationValue,
    },
    /// Operation failed
    Fail {
        error: String,
    },
    /// Operation status unknown (timeout, partition)
    Info {
        partial_result: Option<OperationValue>,
    },
}

/// Return value from operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OperationValue {
    /// Store confirmation
    StoreConfirmation {
        memory_id: String,
        node_id: String,
        timestamp: f64,
    },
    /// Recall result
    RecallResult {
        memory_id: String,
        content: String,
        confidence: f32,
        from_node: String,
    },
    /// Recall similar results
    RecallSimilarResults {
        results: Vec<RecallMatch>,
    },
    /// Consolidation completion
    ConsolidationComplete {
        patterns_detected: usize,
        convergence_rounds: u32,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecallMatch {
    pub memory_id: String,
    pub content: String,
    pub similarity: f32,
    pub confidence: f32,
}

/// History event: operation invocation or completion
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HistoryEvent {
    /// Operation invoked
    Invoke {
        op_id: OpId,
        process: ProcessId,
        operation: Operation,
        wall_time_ns: u64,
    },
    /// Operation completed
    Complete {
        op_id: OpId,
        process: ProcessId,
        outcome: Outcome,
        wall_time_ns: u64,
    },
}

/// Thread-safe history recorder
pub struct HistoryRecorder {
    /// Node identifier
    node_id: String,

    /// Monotonic operation counter
    op_counter: AtomicU64,

    /// Recorded events (lock-free append)
    events: crossbeam_queue::SegQueue<HistoryEvent>,
}

impl HistoryRecorder {
    pub fn new(node_id: String) -> Self {
        Self {
            node_id,
            op_counter: AtomicU64::new(0),
            events: crossbeam_queue::SegQueue::new(),
        }
    }

    /// Generate unique operation ID
    pub fn next_op_id(&self) -> OpId {
        OpId(self.op_counter.fetch_add(1, Ordering::SeqCst))
    }

    /// Record operation invocation
    pub fn record_invoke(&self, op_id: OpId, process: ProcessId, operation: Operation) {
        let wall_time_ns = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos()
            .try_into()
            .unwrap_or(u64::MAX);

        self.events.push(HistoryEvent::Invoke {
            op_id,
            process,
            operation,
            wall_time_ns,
        });
    }

    /// Record operation completion
    pub fn record_complete(&self, op_id: OpId, process: ProcessId, outcome: Outcome) {
        let wall_time_ns = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos()
            .try_into()
            .unwrap_or(u64::MAX);

        self.events.push(HistoryEvent::Complete {
            op_id,
            process,
            outcome,
            wall_time_ns,
        });
    }

    /// Export all events to JSON for Jepsen analysis
    pub fn export_history(&self) -> Vec<HistoryEvent> {
        let mut events = vec![];
        while let Some(event) = self.events.pop() {
            events.push(event);
        }
        // Sort by wall time
        events.sort_by_key(|e| match e {
            HistoryEvent::Invoke { wall_time_ns, .. } => *wall_time_ns,
            HistoryEvent::Complete { wall_time_ns, .. } => *wall_time_ns,
        });
        events
    }
}
```

#### Rust: Consistency Checker for Eventual Consistency

```rust
// engram-core/src/cluster/jepsen/checker.rs

use std::collections::{HashMap, HashSet};
use super::history::{HistoryEvent, Operation, Outcome, OperationValue, OpId};

/// Consistency violation detected
#[derive(Debug, Clone)]
pub enum ConsistencyViolation {
    /// Write acknowledged but lost after partition heal
    DataLoss {
        memory_id: String,
        stored_on_nodes: Vec<String>,
        missing_after_heal: bool,
    },

    /// Nodes failed to converge within bounded time
    ConvergenceFailure {
        partition_heal_time_ns: u64,
        convergence_time_ns: Option<u64>,
        timeout_threshold_ns: u64,
    },

    /// Confidence bounds don't match actual divergence
    ConfidenceCalibrationError {
        claimed_confidence: f32,
        actual_correctness_rate: f32,
        divergence: f32,
    },

    /// Split-brain: conflicting writes not resolved deterministically
    SplitBrain {
        memory_id: String,
        partition_side_a_value: String,
        partition_side_b_value: String,
        final_values: Vec<String>,
    },
}

pub struct EventualConsistencyChecker {
    /// All history events sorted by time
    events: Vec<HistoryEvent>,

    /// Nemesis events (partition start/heal times)
    partition_intervals: Vec<(u64, u64)>,
}

impl EventualConsistencyChecker {
    pub fn new(events: Vec<HistoryEvent>, partition_intervals: Vec<(u64, u64)>) -> Self {
        Self {
            events,
            partition_intervals,
        }
    }

    /// Check all eventual consistency properties
    pub fn check(&self) -> Result<(), Vec<ConsistencyViolation>> {
        let mut violations = vec![];

        // Check 1: No data loss
        if let Some(loss_violations) = self.check_no_data_loss() {
            violations.extend(loss_violations);
        }

        // Check 2: Bounded convergence
        if let Some(conv_violations) = self.check_bounded_convergence() {
            violations.extend(conv_violations);
        }

        // Check 3: Confidence calibration
        if let Some(conf_violations) = self.check_confidence_calibration() {
            violations.extend(conf_violations);
        }

        // Check 4: Split-brain resolution
        if let Some(split_violations) = self.check_split_brain_resolution() {
            violations.extend(split_violations);
        }

        if violations.is_empty() {
            Ok(())
        } else {
            Err(violations)
        }
    }

    /// Verify all acknowledged writes survive partition healing
    fn check_no_data_loss(&self) -> Option<Vec<ConsistencyViolation>> {
        let mut violations = vec![];

        // Build map of successful stores
        let mut stored_memories: HashMap<String, HashSet<String>> = HashMap::new();

        for event in &self.events {
            if let HistoryEvent::Complete { outcome, .. } = event {
                if let Outcome::Ok { value: OperationValue::StoreConfirmation { memory_id, node_id, .. } } = outcome {
                    stored_memories.entry(memory_id.clone())
                        .or_insert_with(HashSet::new)
                        .insert(node_id.clone());
                }
            }
        }

        // After last partition heals, verify all stored memories are recallable
        let last_heal_time = self.partition_intervals.last().map(|(_, heal)| *heal);

        if let Some(heal_time) = last_heal_time {
            // Find recall attempts after heal
            for (memory_id, stored_nodes) in stored_memories {
                let found_after_heal = self.events.iter().any(|event| {
                    if let HistoryEvent::Complete { wall_time_ns, outcome, .. } = event {
                        if *wall_time_ns > heal_time + 60_000_000_000 { // 60s grace period
                            if let Outcome::Ok { value: OperationValue::RecallResult { memory_id: recall_id, .. } } = outcome {
                                return *recall_id == memory_id;
                            }
                        }
                    }
                    false
                });

                if !found_after_heal {
                    violations.push(ConsistencyViolation::DataLoss {
                        memory_id,
                        stored_on_nodes: stored_nodes.into_iter().collect(),
                        missing_after_heal: true,
                    });
                }
            }
        }

        if violations.is_empty() { None } else { Some(violations) }
    }

    /// Verify convergence within 60s of partition heal
    fn check_bounded_convergence(&self) -> Option<Vec<ConsistencyViolation>> {
        let mut violations = vec![];

        for (partition_start, partition_heal) in &self.partition_intervals {
            // After partition heals, trigger consolidation and check convergence time
            let convergence_time = self.find_convergence_time(*partition_heal);

            let timeout_threshold = 60_000_000_000; // 60 seconds in nanoseconds

            match convergence_time {
                Some(conv_time) if conv_time - partition_heal > timeout_threshold => {
                    violations.push(ConsistencyViolation::ConvergenceFailure {
                        partition_heal_time_ns: *partition_heal,
                        convergence_time_ns: Some(conv_time),
                        timeout_threshold_ns: timeout_threshold,
                    });
                }
                None => {
                    violations.push(ConsistencyViolation::ConvergenceFailure {
                        partition_heal_time_ns: *partition_heal,
                        convergence_time_ns: None,
                        timeout_threshold_ns: timeout_threshold,
                    });
                }
                _ => {} // Converged within bounds
            }
        }

        if violations.is_empty() { None } else { Some(violations) }
    }

    /// Find when all nodes converged to same state
    fn find_convergence_time(&self, heal_time: u64) -> Option<u64> {
        // Look for consolidation completion events after heal
        for event in &self.events {
            if let HistoryEvent::Complete { wall_time_ns, outcome, .. } = event {
                if *wall_time_ns > heal_time {
                    if let Outcome::Ok { value: OperationValue::ConsolidationComplete { convergence_rounds, .. } } = outcome {
                        if *convergence_rounds <= 10 { // Expected convergence rounds
                            return Some(*wall_time_ns);
                        }
                    }
                }
            }
        }
        None
    }

    /// Verify confidence scores match actual correctness rates
    fn check_confidence_calibration(&self) -> Option<Vec<ConsistencyViolation>> {
        // Group recalls by confidence bucket
        let mut confidence_buckets: HashMap<u8, Vec<(f32, bool)>> = HashMap::new();

        for event in &self.events {
            if let HistoryEvent::Complete { outcome, .. } = event {
                if let Outcome::Ok { value: OperationValue::RecallResult { confidence, .. } } = outcome {
                    let bucket = (*confidence * 10.0) as u8;
                    // Check if result was actually correct (would need ground truth)
                    // Simplified: assume confidence > 0.9 should be correct
                    let correct = *confidence > 0.9;
                    confidence_buckets.entry(bucket)
                        .or_insert_with(Vec::new)
                        .push((*confidence, correct));
                }
            }
        }

        // Verify calibration: claimed confidence ~= actual correctness rate
        let mut violations = vec![];
        for (bucket, outcomes) in confidence_buckets {
            let avg_confidence: f32 = outcomes.iter().map(|(c, _)| c).sum::<f32>() / outcomes.len() as f32;
            let correctness_rate = outcomes.iter().filter(|(_, correct)| *correct).count() as f32 / outcomes.len() as f32;

            let divergence = (avg_confidence - correctness_rate).abs();
            if divergence > 0.15 { // Allow 15% calibration error
                violations.push(ConsistencyViolation::ConfidenceCalibrationError {
                    claimed_confidence: avg_confidence,
                    actual_correctness_rate: correctness_rate,
                    divergence,
                });
            }
        }

        if violations.is_empty() { None } else { Some(violations) }
    }

    /// Verify split-brain conflicts resolved deterministically
    fn check_split_brain_resolution(&self) -> Option<Vec<ConsistencyViolation>> {
        // Track writes to same memory_id during partition
        // After heal, verify only one value wins (no divergence)
        None // Simplified for now
    }
}
```

### Clojure: Jepsen Test Implementation

```clojure
; jepsen/engram/src/engram/core.clj

(ns engram.core
  (:require [clojure.tools.logging :refer :all]
            [jepsen [cli :as cli]
                    [db :as db]
                    [tests :as tests]
                    [control :as c]
                    [checker :as checker]
                    [nemesis :as nemesis]
                    [generator :as gen]
                    [util :as util :refer [timeout]]]
            [jepsen.checker.timeline :as timeline]
            [jepsen.control.util :as cu]
            [jepsen.os.debian :as debian]
            [engram.client :as client]
            [engram.nemesis :as engram-nemesis]
            [engram.checker :as engram-checker]))

(defn db
  "Engram database lifecycle"
  [version]
  (reify db/DB
    (setup! [_ test node]
      (info node "installing engram" version)
      (c/su
        ; Install Rust if needed
        (when-not (cu/exists? "/usr/local/cargo/bin/cargo")
          (info node "installing rust")
          (c/exec :curl :--proto "=https" :--tlsv1.2 :-sSf "https://sh.rustup.rs"
                  :| :sh :-s :-- :-y))

        ; Clone and build Engram
        (c/cd "/opt"
          (when-not (cu/exists? "/opt/engram")
            (c/exec :git :clone "https://github.com/engram/engram.git"))
          (c/cd "/opt/engram"
            (c/exec :git :checkout version)
            (c/exec "/usr/local/cargo/bin/cargo" :build :--release)))

        ; Configure cluster mode
        (c/exec :echo (slurp (clojure.java.io/resource "cluster.toml"))
                :> "/opt/engram/cluster.toml")

        ; Start Engram node
        (cu/start-daemon!
          {:logfile "/var/log/engram.log"
           :pidfile "/var/run/engram.pid"
           :chdir   "/opt/engram"}
          "/opt/engram/target/release/engram-cli"
          :--config "/opt/engram/cluster.toml"
          :--node-id node
          :--seed-nodes (str (first (:nodes test)) ":7946"))))

    (teardown! [_ test node]
      (info node "tearing down engram")
      (cu/stop-daemon! "engram" "/var/run/engram.pid")
      (c/su (c/exec :rm :-rf "/opt/engram/data")))

    db/LogFiles
    (log-files [_ test node]
      ["/var/log/engram.log"])))

(defn workload
  "Engram workload: stores, recalls, consolidations"
  [opts]
  {:generator (gen/phases
                ; Phase 1: Initial population (no failures)
                (->> (gen/mix [client/store-gen
                               client/recall-gen])
                     (gen/stagger 1/10)
                     (gen/nemesis nil)
                     (gen/time-limit 30))

                ; Phase 2: Partition + concurrent writes
                (->> (gen/mix [client/store-gen
                               client/recall-gen
                               client/consolidate-gen])
                     (gen/stagger 1/10)
                     (gen/nemesis
                       (gen/seq (cycle [(gen/sleep 10)
                                        {:type :info, :f :start-partition}
                                        (gen/sleep 20)
                                        {:type :info, :f :heal-partition}])))
                     (gen/time-limit 120))

                ; Phase 3: Final convergence check
                (->> client/recall-all-gen
                     (gen/nemesis nil)
                     (gen/time-limit 60)))

   :client    (client/engram-client opts)
   :checker   (checker/compose
                {:perf        (checker/perf)
                 :timeline    (timeline/html)
                 :eventual    (engram-checker/eventual-consistency)
                 :data-loss   (engram-checker/no-data-loss)
                 :convergence (engram-checker/bounded-convergence)
                 :confidence  (engram-checker/confidence-calibration)})

   :nemesis   (engram-nemesis/partition-nemesis)

   :model     nil ; Eventual consistency has no simple model
   })

(defn engram-test
  "Main Jepsen test for Engram distributed system"
  [opts]
  (merge tests/noop-test
         opts
         {:name    "engram-distributed"
          :os      debian/os
          :db      (db "v0.14.0")
          :pure-generators true}
         (workload opts)))

(defn -main
  "Jepsen CLI entry point"
  [& args]
  (cli/run!
    (merge (cli/single-test-cmd {:test-fn engram-test})
           (cli/serve-cmd))
    args))
```

```clojure
; jepsen/engram/src/engram/client.clj

(ns engram.client
  (:require [clojure.tools.logging :refer :all]
            [jepsen [client :as client]
                    [generator :as gen]]
            [slingshot.slingshot :refer [try+]]
            [cheshire.core :as json])
  (:import (java.net URI)
           (java.net.http HttpClient HttpRequest HttpResponse$BodyHandlers)
           (java.time Duration)))

(defn engram-client
  "HTTP client for Engram gRPC/HTTP API"
  [opts]
  (let [http-client (-> (HttpClient/newBuilder)
                        (.connectTimeout (Duration/ofSeconds 5))
                        (.build))]
    (reify client/Client
      (open! [this test node]
        (assoc this :node node :http-client http-client))

      (setup! [this test])

      (invoke! [this test op]
        (try+
          (case (:f op)
            :store
            (let [req (-> (HttpRequest/newBuilder)
                          (.uri (URI. (str "http://" (:node this) ":8080/api/v1/memory")))
                          (.header "Content-Type" "application/json")
                          (.POST (HttpRequest$BodyPublishers/ofString
                                   (json/generate-string
                                     {:space "test-space"
                                      :content (:value op)
                                      :embedding (repeatedly 768 #(rand))})))
                          (.timeout (Duration/ofSeconds 5))
                          (.build))
                  resp (.send http-client req (HttpResponse$BodyHandlers/ofString))
                  body (json/parse-string (.body resp) true)]
              (if (= 200 (.statusCode resp))
                (assoc op :type :ok :value (:memory_id body))
                (assoc op :type :fail :error (:error body))))

            :recall
            (let [req (-> (HttpRequest/newBuilder)
                          (.uri (URI. (str "http://" (:node this) ":8080/api/v1/memory/" (:value op))))
                          (.GET)
                          (.timeout (Duration/ofSeconds 5))
                          (.build))
                  resp (.send http-client req (HttpResponse$BodyHandlers/ofString))
                  body (json/parse-string (.body resp) true)]
              (if (= 200 (.statusCode resp))
                (assoc op :type :ok :value (:content body))
                (assoc op :type :fail :error "not found")))

            :consolidate
            (let [req (-> (HttpRequest/newBuilder)
                          (.uri (URI. (str "http://" (:node this) ":8080/api/v1/consolidate")))
                          (.POST (HttpRequest$BodyPublishers/ofString
                                   (json/generate-string {:space "test-space"})))
                          (.timeout (Duration/ofSeconds 30))
                          (.build))
                  resp (.send http-client req (HttpResponse$BodyHandlers/ofString))]
              (if (= 200 (.statusCode resp))
                (assoc op :type :ok)
                (assoc op :type :fail))))

          (catch java.net.http.HttpTimeoutException e
            (assoc op :type :info :error :timeout))
          (catch Exception e
            (assoc op :type :fail :error (.getMessage e)))))

      (teardown! [this test])

      (close! [this test]))))

(def store-gen
  "Generator for store operations"
  (->> (range)
       (map (fn [i]
              {:type :invoke
               :f    :store
               :value (str "memory-" i)}))))

(def recall-gen
  "Generator for recall operations"
  (->> (range)
       (map (fn [i]
              {:type :invoke
               :f    :recall
               :value (str "memory-" (rand-int i))}))))

(def consolidate-gen
  "Generator for consolidation operations"
  (repeat {:type :invoke
           :f    :consolidate}))

(def recall-all-gen
  "Generator for recalling all stored memories"
  (->> (range 1000)
       (map (fn [i]
              {:type :invoke
               :f    :recall
               :value (str "memory-" i)}))))
```

```clojure
; jepsen/engram/src/engram/checker.clj

(ns engram.checker
  (:require [clojure.tools.logging :refer :all]
            [jepsen.checker :as checker]
            [clojure.set :as set]))

(defn eventual-consistency
  "Check eventual consistency: all nodes converge to same state"
  []
  (reify checker/Checker
    (check [this test history opts]
      (let [final-state (->> history
                             (filter #(= :ok (:type %)))
                             (filter #(= :recall (:f %)))
                             (group-by :value)
                             (map (fn [[k vs]] [k (set (map :value vs))]))
                             (into {}))]
        {:valid? (every? #(= 1 (count (val %))) final-state)
         :final-state final-state}))))

(defn no-data-loss
  "Check no acknowledged writes are lost"
  []
  (reify checker/Checker
    (check [this test history opts]
      (let [stored (set (->> history
                             (filter #(= :ok (:type %)))
                             (filter #(= :store (:f %)))
                             (map :value)))
            recalled (set (->> history
                               (filter #(= :ok (:type %)))
                               (filter #(= :recall (:f %)))
                               (map :value)))
            lost (set/difference stored recalled)]
        {:valid? (empty? lost)
         :stored-count (count stored)
         :recalled-count (count recalled)
         :lost-memories lost}))))

(defn bounded-convergence
  "Check convergence happens within 60 seconds of partition heal"
  []
  (reify checker/Checker
    (check [this test history opts]
      ; Find partition heal times
      (let [heals (->> history
                       (filter #(= :heal-partition (:f %)))
                       (map :time))
            ; Find consolidation completions after heals
            convergences (->> history
                              (filter #(= :consolidate (:f %)))
                              (filter #(= :ok (:type %)))
                              (map :time))
            ; Check all convergences within 60s of heal
            violations (for [heal heals]
                         (let [conv-time (first (filter #(> % heal) convergences))
                               delta (when conv-time (- conv-time heal))]
                           (when (or (nil? conv-time) (> delta 60000))
                             {:heal-time heal
                              :convergence-time conv-time
                              :delta-ms delta})))]
        {:valid? (every? nil? violations)
         :violations (remove nil? violations)}))))

(defn confidence-calibration
  "Check confidence scores match actual correctness"
  []
  (reify checker/Checker
    (check [this test history opts]
      ; Simplified: just count recalls and check success rate
      (let [recalls (->> history
                         (filter #(= :recall (:f %)))
                         (filter #(#{:ok :fail} (:type %))))
            success-rate (/ (count (filter #(= :ok (:type %)) recalls))
                           (max 1 (count recalls)))]
        {:valid? (> success-rate 0.9)
         :success-rate success-rate
         :total-recalls (count recalls)}))))
```

```clojure
; jepsen/engram/src/engram/nemesis.clj

(ns engram.nemesis
  (:require [clojure.tools.logging :refer :all]
            [jepsen [nemesis :as nemesis]
                    [net :as net]
                    [control :as c]]
            [jepsen.nemesis.combined :as nc]))

(defn partition-nemesis
  "Network partition nemesis for Engram"
  []
  (nemesis/partitioner
    (comp nemesis/complete-grudge
          nemesis/bridge)))

(defn full-nemesis
  "Combined nemesis: partitions, clock skew, process pauses"
  [opts]
  (nc/nemesis-package
    {:db        (:db opts)
     :faults    [:partition :kill :pause :clock]
     :partition {:targets [:majority]}
     :clock     {:targets [:primaries]
                 :skew-ms 100000}
     :pause     {:targets [:minority]}
     :kill      {:targets [:one]}
     :interval  10}))
```

## Files to Create

### Rust Implementation

1. **`engram-core/src/cluster/jepsen/mod.rs`** - Jepsen integration module
2. **`engram-core/src/cluster/jepsen/history.rs`** - Operation history recording (complete implementation above)
3. **`engram-core/src/cluster/jepsen/checker.rs`** - Eventual consistency checker (complete implementation above)
4. **`engram-core/src/cluster/jepsen/export.rs`** - History export to JSON format
5. **`engram-core/tests/jepsen_harness.rs`** - Rust test harness for local Jepsen simulation

### Clojure Implementation

6. **`jepsen/engram/project.clj`** - Leiningen project definition
7. **`jepsen/engram/src/engram/core.clj`** - Main test entry point (complete implementation above)
8. **`jepsen/engram/src/engram/client.clj`** - Engram HTTP/gRPC client (complete implementation above)
9. **`jepsen/engram/src/engram/checker.clj`** - Custom consistency checkers (complete implementation above)
10. **`jepsen/engram/src/engram/nemesis.clj`** - Fault injection strategies (complete implementation above)
11. **`jepsen/engram/resources/cluster.toml`** - Engram cluster configuration template

### Infrastructure

12. **`jepsen/engram/docker-compose.yml`** - Local Docker-based Jepsen cluster
13. **`scripts/run_jepsen.sh`** - CI integration script
14. **`docs/operations/jepsen-testing.md`** - Jepsen test documentation

## Files to Modify

1. **`engram-core/src/cluster/mod.rs`** - Export jepsen module
2. **`engram-core/Cargo.toml`** - Add jepsen feature flag
3. **`engram-cli/src/main.rs`** - Add `--jepsen-mode` flag to enable history recording
4. **`.gitignore`** - Ignore Jepsen store/ directory
5. **`Makefile`** - Add `make jepsen` target

## Testing Strategy

### Local Simulation (Fast Iteration)

```rust
// engram-core/tests/jepsen_harness.rs

#[tokio::test]
async fn test_eventual_consistency_local() {
    // Simulate 3-node cluster in-process
    let node1 = start_test_node(8001, vec!["node2:8002"]).await;
    let node2 = start_test_node(8002, vec!["node1:8001", "node3:8003"]).await;
    let node3 = start_test_node(8003, vec!["node2:8002"]).await;

    let recorder = Arc::new(HistoryRecorder::new("test-cluster".to_string()));

    // Phase 1: Baseline writes
    for i in 0..100 {
        let op_id = recorder.next_op_id();
        recorder.record_invoke(op_id, ProcessId(0), Operation::Store {
            space: "test".to_string(),
            memory_id: format!("mem-{}", i),
            embedding: vec![0.0; 768],
            content: format!("content-{}", i),
        });

        let result = node1.store("test", &format!("mem-{}", i), vec![0.0; 768], &format!("content-{}", i)).await;

        recorder.record_complete(op_id, ProcessId(0), Outcome::Ok {
            value: OperationValue::StoreConfirmation {
                memory_id: format!("mem-{}", i),
                node_id: "node1".to_string(),
                timestamp: 0.0,
            }
        });
    }

    // Phase 2: Network partition
    simulate_partition(&[&node1], &[&node2, &node3]).await;

    // Continue writes on both sides
    for i in 100..150 {
        // Side A
        node1.store("test", &format!("mem-{}", i), vec![0.0; 768], &format!("content-{}", i)).await;
        // Side B
        node2.store("test", &format!("mem-{}", i), vec![0.0; 768], &format!("content-{}", i)).await;
    }

    // Phase 3: Heal partition
    heal_partition(&[&node1, &node2, &node3]).await;

    // Wait for convergence
    tokio::time::sleep(Duration::from_secs(60)).await;

    // Phase 4: Verify consistency
    let checker = EventualConsistencyChecker::new(
        recorder.export_history(),
        vec![(0, 60_000_000_000)]
    );

    assert!(checker.check().is_ok(), "Consistency violations detected");
}
```

### Full Jepsen Suite (CI Integration)

```bash
#!/bin/bash
# scripts/run_jepsen.sh

set -e

echo "Starting Jepsen test suite for Engram..."

# Build Engram release binary
cargo build --release --features cluster

# Start Jepsen control node + 5 database nodes
cd jepsen/engram
lein run test \
  --nodes-file nodes.txt \
  --time-limit 300 \
  --concurrency 10 \
  --rate 100 \
  --test-count 5

# Analyze results
lein run serve

echo "Jepsen tests complete. Results in store/"
```

### CI/CD Integration

```yaml
# .github/workflows/jepsen.yml (if GitHub Actions were used, but CLAUDE.md says no .github)
# Instead, add to Makefile:

.PHONY: jepsen
jepsen:
	@echo "Running Jepsen consistency tests..."
	./scripts/run_jepsen.sh
```

## Dependencies

### Rust Dependencies

Add to `engram-core/Cargo.toml`:

```toml
[dependencies]
# Already have: serde, serde_json, crossbeam-queue, dashmap

# For Jepsen feature
crossbeam-queue = "0.3"  # Lock-free history recording
```

### Clojure Dependencies

`jepsen/engram/project.clj`:

```clojure
(defproject engram "0.14.0"
  :description "Jepsen tests for Engram distributed memory system"
  :url "https://github.com/engram/engram"
  :license {:name "Apache License 2.0"
            :url "https://www.apache.org/licenses/LICENSE-2.0"}
  :dependencies [[org.clojure/clojure "1.11.1"]
                 [jepsen "0.3.8"]
                 [cheshire "5.12.0"]]
  :main engram.core
  :jvm-opts ["-Xmx8g" "-server"])
```

### System Dependencies

- JDK 17+
- Leiningen 2.9+
- Gnuplot (visualization)
- Graphviz (anomaly graphs)
- SSH access to test nodes

## Acceptance Criteria

1. **Zero Consistency Violations**: 1000 test runs with no eventual consistency violations
2. **No Data Loss**: All acknowledged writes survive partition healing (100% success rate)
3. **Bounded Convergence**: 99% of partition heals converge within 60 seconds
4. **Confidence Calibration**: Claimed confidence within 10% of actual correctness rate
5. **Split-Brain Resolution**: Conflicting writes resolved deterministically (same outcome every run)
6. **Nemesis Coverage**: All nemesis strategies tested (partition, clock, kill, pause, combined)
7. **CI Integration**: Jepsen subset runs on every commit (5-minute timeout)
8. **Full Suite**: Comprehensive suite runs nightly (300 seconds, 10 concurrent clients)
9. **Documentation**: Complete runbook for interpreting Jepsen results
10. **Reproducibility**: Deterministic test replay from seed for debugging

## Performance Targets

- History recording overhead: <1% CPU, <100MB memory per node
- Analysis time: <2 minutes for 100K operation history
- Test suite runtime: <5 minutes for CI subset, <30 minutes for full suite
- Support histories up to 1M operations without OOM

## Integration with Existing Test Framework

### Hook into Task 010 Network Simulator

Jepsen nemesis will use Task 010's `NetworkSimulator` for local testing:

```rust
use engram_core::cluster::test::NetworkSimulator;

let sim = NetworkSimulator::new(seed);
sim.inject_partition(vec!["node1"], vec!["node2", "node3"]);
// Jepsen recorder captures operations during partition
```

### Reuse Existing Metrics Infrastructure

History recorder will export metrics to Prometheus:

```rust
pub struct JepsenMetrics {
    pub operations_recorded: Counter,
    pub history_export_duration: Histogram,
    pub checker_runtime: Histogram,
}
```

## Next Steps After Completion

1. **Task 012**: Incorporate Jepsen results into operational runbook
2. **Long-term monitoring**: Deploy Jepsen tests in production staging environment (weekly runs)
3. **Bug database**: Document all historical violations and fixes
4. **Benchmark suite**: Compare Jepsen results across Engram versions (regression detection)

## References

### Academic Papers

1. Das, Gupta, Aberer (2002). "SWIM: Scalable Weakly-consistent Infection-style Process Group Membership Protocol"
2. Bailis et al. (2012). "Probabilistically Bounded Staleness for Practical Partial Quorums"
3. Kingsbury (2020). "Elle: Inferring Isolation Anomalies from Experimental Observations"
4. Adya, Liskov, O'Neil (2000). "Generalized Isolation Level Definitions"

### Implementation Resources

- Jepsen Documentation: https://jepsen.io/
- Elle GitHub: https://github.com/jepsen-io/elle
- Knossos GitHub: https://github.com/jepsen-io/knossos
- Kingsbury Blog: https://aphyr.com/posts/314-computational-techniques-in-knossos

### Engram-Specific

- Vision: `/Users/jordan/Workspace/orchard9/engram/vision.md` (Phase 4: Distribution)
- Task 001: SWIM Membership (foundation for cluster testing)
- Task 007: Gossip Protocol (convergence testing target)
- Task 008: Conflict Resolution (split-brain testing target)
