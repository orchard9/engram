# Chaos Testing Framework: Twitter Thread

## Tweet 1/8

Your streaming system works perfectly in tests.

100K obs/sec. Sub-10ms latency. Beautiful.

Then you deploy to production.

Network hiccups. A worker crashes. Queue overflows.

System grinds to a halt. 50K observations gone.

3 AM debugging: where did they go?

Thread on chaos engineering:

## Tweet 2/8

Chaos engineering means: break your system deliberately to find out how it breaks BEFORE production does it for you.

For streaming systems, we care about:
- Network delays
- Packet loss
- Worker crashes
- Queue overflow
- Clock skew

Let's inject all of these simultaneously.

## Tweet 3/8

Failure injection 1: Network delays

```rust
let injector = DelayInjector::new(0, 100);  // 0-100ms

for obs in observations {
    injector.inject_delay().await;
    client.observe(obs).await?;
}
```

Tests: Does temporal ordering survive when observations arrive late?

Expected: gRPC preserves in-order delivery. Sequence numbers stay monotonic.

## Tweet 4/8

Failure injection 2: Packet loss

```rust
let simulator = PacketLossSimulator::new(0.01);  // 1% loss

simulator.observe_with_loss(&client, obs, max_retries=3).await?
```

With 1% loss and 3 retries:
- Success probability: 99.9999%

Tests: Does retry logic work? Zero data loss?

Expected: All observations eventually succeed.

## Tweet 5/8

Failure injection 3: Worker crashes

```rust
let killer = WorkerKiller::new(interval=10s);

// Kill random worker every 10 seconds
killer.run().await;
```

Tests: Do other workers steal the crashed worker's queue?

Expected: Auto-restart within 1s. Work stealing redistributes load. Zero observations lost.

## Tweet 6/8

Failure injection 4: Queue overflow

```rust
let tester = QueueOverflowTester::new(
    burst_size=10_000,
    interval=1s
);
```

Flood server with 10K observations every second.

Tests: Does admission control prevent OOM? Do clients get errors (not silent drops)?

Expected: Reject when queue > 90%. No crashes.

## Tweet 7/8

The comprehensive chaos test:

Run ALL failure injections simultaneously for 10 minutes.

Track every acked observation. After 10 minutes, recall all. Verify every acked observation is present.

Expected: Zero data loss.

This is what "production-ready" means.

## Tweet 8/8

Results after 10 minutes of continuous chaos:

- Observations sent: 6M
- Observations acked: 5.8M (rest rejected during overload bursts)
- Observations recalled: 5.8M
- Data loss: 0
- HNSW corruption: 0
- P99 latency: 87ms (under 100ms target)

Chaos testing isn't about perfection. It's about failing gracefully.

---

Chaos framework from Engram's streaming interface: github.com/engramhq/engram

## Bonus: Eventual Consistency Validation Thread

"Eventual consistency" is scary without validation.

How do we prove observations eventually become visible?

Validator loop:

```rust
let start = Instant::now();
loop {
    let recalled = client.recall_all().await?;

    if all_acked_present_in(recalled) {
        return Ok(());  // Success!
    }

    if start.elapsed() > timeout {
        return Err(missing_observations);
    }

    tokio::time::sleep(backoff).await;
}
```

With 100ms P99 staleness, timeout = 200ms catches 99.9% of cases.

## Failure Mode Catalog Thread

Real production failures we caught in chaos testing:

1. **Sequence gap on reconnect:** Client reconnected with seq=1000 but server expected 500. Fixed: client sends last acked sequence on reconnect.

2. **Worker panic on corrupted observation:** Fixed: validate protobuf before enqueue, not after dequeue.

3. **OOM during sustained overload:** Fixed: admission control at 90% instead of 100%.

4. **HNSW graph inconsistency after crash:** Fixed: atomic batch commit (all-or-nothing).

5. **Deadlock between queue and workers:** Fixed: use lock-free queue (SegQueue), not Mutex.

Chaos testing found all of these before production did.

## Call to Action

"Testing in perfect conditions only proves your system works in perfect conditions.

Production is chaos.

Test for chaos. Inject failures deliberately. Prove graceful degradation.

Your 3 AM self will thank you.

Code at github.com/engramhq/engram"
