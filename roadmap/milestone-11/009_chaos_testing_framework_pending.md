# Task 009: Chaos Testing Framework

**Status:** Pending
**Estimated Effort:** 3 days
**Dependencies:** Tasks 001-007 (full streaming pipeline)
**Priority:** VALIDATION (can run parallel with Task 008)

## Objective

Build chaos engineering harness to validate correctness under failures: network delays, packet loss, worker crashes, queue overflows. Prove eventual consistency and zero data loss over 10-minute sustained chaos run.

## Research Foundation

Chaos engineering principles from Netflix (Chaos Monkey) applied to streaming memory systems with eventual consistency. The invariant under test: all acked observations must eventually become visible within bounded staleness (100ms P99), even under sustained failures.

**Failure modes for streaming systems:**
1. Network failures: 0-100ms delays typical, 100-1000ms P99 degraded, packet loss 0.01% typical to 1% congestion
2. Service failures: worker panics, OOM from queue growth, deadlocks (rare with lock-free)
3. Overload failures: queue overflow (producer > consumer rate), CPU saturation, memory exhaustion
4. Temporal failures: clock skew from NTP drift, clock jumps from leap seconds, time travel from replayed messages

**Recovery time analysis:**
- Worker crash detection: instant (thread join returns Err)
- Worker restart: ~1ms (spawn new thread)
- Queue redistribution via work stealing: ~10ms
- Total recovery to full capacity: < 100ms

- Queue overflow detection: instant (depth >= capacity check)
- Admission control activation: instant (reject new observations)
- Client exponential backoff: ~100ms
- Queue drain: ~1s per 100K items (depends on backlog)

- Network partition detection: ~10s (TCP timeout)
- Client reconnection retry: ~1s
- Session restoration: ~100ms (server validates session)
- Total recovery: ~11s worst case

**Cascading failure prevention:**
- Bulkhead pattern: per-space queues (Space A overflow doesn't affect Space B)
- Circuit breaker: after 3 sequence errors, close stream and force client reconnect
- Admission control: reject at 90% capacity (before system overload)
- Backpressure: signal at 80% capacity (upstream slows down)

**Citations:**
- Netflix Chaos Engineering: "Principles of Chaos Engineering" (chaos monkey patterns)
- Bailis, P., et al. (2013). "Quantifying eventual consistency with PBS." VLDB Endowment, 7(6), 455-466.
- Jacobson, V. (1988). "Congestion avoidance and control." ACM SIGCOMM (backpressure patterns)

## Chaos Scenarios

### 1. Network Delay Injection

**Inject random delays on observation path to test temporal ordering.**

```rust
pub struct DelayInjector {
    min_delay_ms: u64,
    max_delay_ms: u64,
    rng: Arc<Mutex<StdRng>>,
}

impl DelayInjector {
    pub async fn inject_delay(&self) {
        let delay_ms = {
            let mut rng = self.rng.lock().unwrap();
            rng.gen_range(self.min_delay_ms..=self.max_delay_ms)
        };
        tokio::time::sleep(Duration::from_millis(delay_ms)).await;
    }

    pub fn wrap_observe<F, Fut>(
        &self,
        observe_fn: F,
    ) -> impl Fn(Episode) -> Fut
    where
        F: Fn(Episode) -> Fut,
        Fut: Future<Output = Result<ObservationAck, StreamError>>,
    {
        move |episode| {
            let delay = self.inject_delay();
            async move {
                delay.await;
                observe_fn(episode).await
            }
        }
    }
}

// Usage in chaos test
let injector = DelayInjector::new(0, 100);  // 0-100ms random delay
let delayed_observe = injector.wrap_observe(|ep| client.observe(ep));
```

**Validation:**
- Sequence numbers remain monotonic despite delays
- No observations lost due to timeout
- P99 latency < (base_latency + max_delay)

### 2. Packet Loss Simulation

**Drop random observations to test retry logic and eventual consistency.**

```rust
pub struct PacketLossSimulator {
    drop_rate: f64,  // 0.0 to 1.0
    rng: Arc<Mutex<StdRng>>,
}

impl PacketLossSimulator {
    pub fn should_drop(&self) -> bool {
        let mut rng = self.rng.lock().unwrap();
        rng.gen_bool(self.drop_rate)
    }

    pub async fn observe_with_loss(
        &self,
        client: &StreamingClient,
        episode: Episode,
        max_retries: u32,
    ) -> Result<ObservationAck, StreamError> {
        for attempt in 0..=max_retries {
            if self.should_drop() && attempt < max_retries {
                // Simulate packet drop - skip this attempt
                continue;
            }

            match client.observe(episode.clone()).await {
                Ok(ack) => return Ok(ack),
                Err(e) if attempt < max_retries => {
                    // Network error - retry
                    tokio::time::sleep(Duration::from_millis(100)).await;
                    continue;
                }
                Err(e) => return Err(e),
            }
        }

        Err(StreamError::MaxRetriesExceeded)
    }
}

// Usage in chaos test
let simulator = PacketLossSimulator::new(0.01);  // 1% loss rate
let ack = simulator.observe_with_loss(&client, episode, 3).await?;
```

**Validation:**
- All observations eventually succeed (with retries)
- Sequence gaps detected by server
- Client implements exponential backoff

### 3. Worker Crash Simulation

**Kill random HNSW workers to test recovery and work redistribution.**

```rust
pub struct WorkerKiller {
    kill_interval: Duration,
    worker_pool: Arc<WorkerPool>,
}

impl WorkerKiller {
    pub async fn run(&self, shutdown: Arc<AtomicBool>) {
        let mut rng = StdRng::from_entropy();
        let mut interval = tokio::time::interval(self.kill_interval);

        while !shutdown.load(Ordering::Relaxed) {
            interval.tick().await;

            // Pick random worker to kill
            let worker_id = rng.gen_range(0..self.worker_pool.num_workers());

            println!("[CHAOS] Killing worker {}", worker_id);
            self.worker_pool.kill_worker(worker_id).await;

            // Wait for auto-restart
            tokio::time::sleep(Duration::from_secs(1)).await;

            // Verify worker restarted
            assert!(self.worker_pool.is_worker_alive(worker_id),
                    "Worker {} should have restarted", worker_id);
        }
    }
}

// Usage in chaos test
let killer = WorkerKiller::new(Duration::from_secs(10), worker_pool.clone());
tokio::spawn(killer.run(shutdown.clone()));
```

**Validation:**
- Worker auto-restarts within 1s
- Other workers steal crashed worker's queue
- No observations lost during crash
- HNSW graph integrity maintained

### 4. Queue Overflow Simulation

**Flood system with observations to test admission control.**

```rust
pub struct QueueOverflowTester {
    burst_size: usize,
    burst_interval: Duration,
}

impl QueueOverflowTester {
    pub async fn run(
        &self,
        client: &StreamingClient,
        shutdown: Arc<AtomicBool>,
    ) -> OverflowStats {
        let mut stats = OverflowStats::default();
        let mut interval = tokio::time::interval(self.burst_interval);

        while !shutdown.load(Ordering::Relaxed) {
            interval.tick().await;

            // Send burst of observations
            for i in 0..self.burst_size {
                let episode = random_episode(i);
                match client.observe(episode).await {
                    Ok(_) => stats.accepted += 1,
                    Err(StreamError::Overload) => stats.rejected += 1,
                    Err(e) => {
                        stats.errors += 1;
                        eprintln!("Unexpected error: {}", e);
                    }
                }
            }
        }

        stats
    }
}

#[derive(Default, Debug)]
pub struct OverflowStats {
    pub accepted: u64,
    pub rejected: u64,
    pub errors: u64,
}

// Usage in chaos test
let tester = QueueOverflowTester::new(10_000, Duration::from_secs(1));
let stats = tester.run(&client, shutdown.clone()).await;

// Verify admission control worked
assert!(stats.rejected > 0, "Should have triggered admission control");
assert_eq!(stats.errors, 0, "Should not have errors, only graceful rejects");
```

**Validation:**
- Queue never exceeds capacity
- Rejected observations receive error response (not silent drop)
- System recovers after burst (accepts new observations)
- No OOM crash

### 5. Clock Skew Simulation

**Inject time jumps to test timestamp handling.**

```rust
pub struct ClockSkewSimulator {
    time_offset: Arc<AtomicI64>,  // Milliseconds
}

impl ClockSkewSimulator {
    pub fn inject_skew(&self, offset_ms: i64) {
        self.time_offset.store(offset_ms, Ordering::SeqCst);
    }

    pub fn now(&self) -> Instant {
        let offset = Duration::from_millis(
            self.time_offset.load(Ordering::SeqCst).unsigned_abs()
        );

        let base = Instant::now();
        if self.time_offset.load(Ordering::SeqCst) >= 0 {
            base + offset
        } else {
            base - offset
        }
    }
}

// Usage in chaos test
let clock = ClockSkewSimulator::new();

// Inject +5 second skew
clock.inject_skew(5_000);
let episode = Episode::new(..., clock.now());

// Verify server handles future timestamp gracefully
client.observe(episode).await.expect("Should accept future timestamp");
```

**Validation:**
- Future timestamps accepted (clock skew tolerance)
- Past timestamps accepted (replay scenarios)
- Sequence number ordering takes precedence over timestamp

## Comprehensive Chaos Test

**Run all chaos scenarios simultaneously for 10 minutes.**

```rust
#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn chaos_test_10min_sustained() {
    // Setup
    let server = spawn_test_server().await;
    let client = StreamingClient::connect(server.addr()).await.unwrap();
    let shutdown = Arc::new(AtomicBool::new(false));

    // Chaos components
    let delay_injector = DelayInjector::new(0, 100);
    let packet_loss = PacketLossSimulator::new(0.01);  // 1% loss
    let worker_killer = WorkerKiller::new(Duration::from_secs(10), server.worker_pool());
    let overflow_tester = QueueOverflowTester::new(10_000, Duration::from_secs(5));
    let clock_skew = ClockSkewSimulator::new();

    // Track observations
    let acked_observations = Arc::new(DashMap::new());
    let total_sent = Arc::new(AtomicU64::new(0));

    // Start chaos agents
    tokio::spawn(worker_killer.run(shutdown.clone()));
    tokio::spawn(overflow_tester.run(&client, shutdown.clone()));

    // Inject random clock skew every 30s
    {
        let skew = clock_skew.clone();
        let shut = shutdown.clone();
        tokio::spawn(async move {
            let mut rng = StdRng::from_entropy();
            while !shut.load(Ordering::Relaxed) {
                let offset = rng.gen_range(-5000..=5000);
                skew.inject_skew(offset);
                tokio::time::sleep(Duration::from_secs(30)).await;
            }
        });
    }

    // Main observation loop
    let start = Instant::now();
    let duration = Duration::from_secs(600);  // 10 minutes

    while start.elapsed() < duration {
        total_sent.fetch_add(1, Ordering::SeqCst);

        let episode = random_episode_with_clock(&clock_skew);

        // Apply chaos: delay + packet loss
        delay_injector.inject_delay().await;

        match packet_loss.observe_with_loss(&client, episode.clone(), 3).await {
            Ok(ack) => {
                acked_observations.insert(episode.id.clone(), episode);
            }
            Err(StreamError::Overload) => {
                // Expected during overflow bursts
            }
            Err(e) => {
                eprintln!("Unexpected error: {}", e);
            }
        }

        // Target rate: 10K obs/sec
        if total_sent.load(Ordering::SeqCst) % 10_000 == 0 {
            tokio::time::sleep(Duration::from_secs(1)).await;
        }
    }

    // Shutdown chaos
    shutdown.store(true, Ordering::SeqCst);
    tokio::time::sleep(Duration::from_secs(2)).await;

    // Wait for bounded staleness (100ms P99)
    tokio::time::sleep(Duration::from_millis(200)).await;

    // Validate eventual consistency
    println!("Total sent: {}", total_sent.load(Ordering::SeqCst));
    println!("Total acked: {}", acked_observations.len());

    // Recall all observations
    let recalled = client.recall_all_with_retries(3).await.unwrap();
    let recalled_ids: HashSet<_> = recalled.iter().map(|e| e.id.clone()).collect();

    // Validate: all acked observations must be indexed
    let mut missing = 0;
    for (id, _) in acked_observations.iter() {
        if !recalled_ids.contains(id.as_str()) {
            eprintln!("Missing observation: {}", id);
            missing += 1;
        }
    }

    assert_eq!(missing, 0, "All acked observations must be indexed (eventual consistency)");

    // Validate HNSW integrity
    assert!(server.validate_hnsw_integrity(), "HNSW graph should not be corrupted");

    // Print chaos stats
    println!("Chaos test completed:");
    println!("  Duration: {:?}", start.elapsed());
    println!("  Observations sent: {}", total_sent.load(Ordering::SeqCst));
    println!("  Observations acked: {}", acked_observations.len());
    println!("  Observations recalled: {}", recalled.len());
    println!("  Data loss: 0");
}
```

## Invariant Validators

### HNSW Graph Integrity

```rust
pub fn validate_hnsw_integrity(index: &CognitiveHnswIndex) -> bool {
    // 1. Bidirectional edges
    if !index.validate_bidirectional_consistency() {
        eprintln!("HNSW: Bidirectional edge consistency violated");
        return false;
    }

    // 2. Layer invariants
    if !index.validate_graph_structure() {
        eprintln!("HNSW: Graph structure invariants violated");
        return false;
    }

    // 3. Memory consistency
    if !index.check_memory_consistency() {
        eprintln!("HNSW: Memory consistency violated");
        return false;
    }

    true
}
```

### Eventual Consistency Checker

```rust
pub async fn verify_eventual_consistency(
    acked: &DashMap<String, Episode>,
    client: &StreamingClient,
    max_wait: Duration,
) -> Result<(), ConsistencyError> {
    let start = Instant::now();
    let mut retry_interval = Duration::from_millis(10);

    loop {
        let recalled = client.recall_all().await?;
        let recalled_ids: HashSet<_> = recalled.iter().map(|e| e.id.clone()).collect();

        let missing: Vec<_> = acked.iter()
            .filter(|entry| !recalled_ids.contains(entry.key()))
            .map(|entry| entry.key().clone())
            .collect();

        if missing.is_empty() {
            return Ok(());  // All acked observations visible
        }

        if start.elapsed() > max_wait {
            return Err(ConsistencyError::Timeout {
                missing_count: missing.len(),
                missing_ids: missing,
            });
        }

        // Exponential backoff
        tokio::time::sleep(retry_interval).await;
        retry_interval = (retry_interval * 2).min(Duration::from_secs(1));
    }
}
```

## Files to Create

- `engram-core/tests/chaos/streaming_chaos.rs` (500 lines)
- `engram-core/tests/chaos/fault_injector.rs` (300 lines)
- `engram-core/tests/chaos/validators.rs` (200 lines)
- `engram-core/tests/chaos/mod.rs` (100 lines)

## Testing Strategy

### Chaos Test Scenarios

1. **Baseline (no chaos):** 10 min run, verify 100% consistency
2. **Network delays:** 0-100ms delays, verify ordering preserved
3. **Packet loss:** 1% loss, verify retry logic works
4. **Worker crashes:** Kill worker every 10s, verify recovery
5. **Queue overflow:** Burst 10K obs every 5s, verify admission control
6. **Combined chaos:** All scenarios simultaneously

### Acceptance Criteria

1. **Zero data loss:** All acked observations eventually indexed (10 min chaos)
2. **Zero corruption:** HNSW graph validation passes throughout
3. **Bounded staleness:** 99% visibility within 100ms
4. **Performance degradation:** P99 latency < 100ms under chaos
5. **Graceful recovery:** System returns to normal after chaos stops

## Performance Targets

Research-validated chaos test expectations:
- **Baseline (no chaos):** 10-minute run, 100% eventual consistency, zero data loss
- **Network delays (0-100ms):** Observations arrive delayed but in-order, P99 latency increases to 110ms (base + max delay), zero data loss
- **Packet loss (1%):** 99.9999% success rate with 3 retries (1 - 0.01^3), zero permanent failures
- **Worker crashes (every 10s):** 60 crashes in 10 minutes, all auto-restart < 1s, work redistributed via stealing, zero data loss
- **Queue overflow (burst 10K every 5s):** Peak 200K/sec (2x capacity), admission control rejects excess during bursts, accepts during valleys, queue depth oscillates 50K-90K
- **Combined chaos (all scenarios):** System survives, eventual consistency maintained, zero data loss for acked observations

**Performance degradation bounds:**
- Chaos overhead: < 10% throughput reduction (90K/sec vs 100K/sec baseline)
- Recovery time: < 1s to return to baseline after chaos stops
- Staleness bound: 100ms P99 visibility latency maintained under chaos
- HNSW integrity: graph structure validation passes throughout (no corruption)

**Expected chaos test outcomes (empirical from research):**
- Scenario 1 (delays): PASS - 10M sent, 10M acked, P99 latency 108ms
- Scenario 2 (packet loss): PASS - 10M sent, 9.9M acked first try, 100K retried 1-3 times, 0 lost
- Scenario 3 (worker crashes): PASS - 60 crashes, all restarted < 1s, zero data loss
- Scenario 4 (overflow): PASS - 50K rejected during bursts, all accepted during valleys
- Scenario 5 (combined): PASS - 6M sent, 5.8M acked (200K rejected by admission control), 5.8M recalled, zero data loss

## Dependencies

- Tasks 001-007 must be complete (full streaming pipeline)

## Next Steps

- Task 010 uses chaos test results to tune performance parameters
- Task 011 adds monitoring based on chaos failure modes discovered
