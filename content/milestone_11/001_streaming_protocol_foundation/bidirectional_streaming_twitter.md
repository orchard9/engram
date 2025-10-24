# Bidirectional Streaming: Twitter Thread

## Tweet 1/8

Building a cognitive AI that learns at 100K observations/sec? Traditional request-response APIs break at that scale.

The problem: your agent needs to store memories AND query them simultaneously. Batch APIs force you to choose.

We built something better: bidirectional streaming.

## Tweet 2/8

With bidirectional streaming, data flows both ways on a single connection.

Client pushes observations → Server
Server pushes recall results ← Client

All at the same time. No roundtrips. No blocking.

Your brain encodes while retrieving. Your AI should too.

## Tweet 3/8

The protocol uses gRPC with session management and sequence numbers.

Every observation gets a monotonic sequence number (0, 1, 2, 3...). Server echoes it in the ack.

Why? Temporal ordering. Even if network delays shuffle observations, sequence numbers preserve "what happened when."

## Tweet 4/8

Streaming creates a new problem: what if the client sends faster than the server can index?

Traditional solution: block the client. But that breaks the streaming abstraction.

Better solution: backpressure with flow control messages.

Server: "I'm at 85% capacity, slow down"
Client: reduces rate by 50%

## Tweet 5/8

But what if the client ignores backpressure?

Final defense: admission control.

At 90% queue capacity, server stops accepting observations. Returns an error. Client must retry later.

Critical invariant: **no silent drops**. System either accepts (eventual consistency) or returns error (retry).

## Tweet 6/8

"Eventual consistency" sounds scary. How do you guarantee correctness?

Bounded staleness: observations become visible in <100ms P99.

Why 100ms? That's how long your hippocampus takes to index new episodic memories.

We're modeling biological reality, not database transactions.

## Tweet 7/8

Implementation uses lock-free queue + parallel HNSW workers.

4 workers × 25K obs/sec each = 100K total throughput

Under load, workers batch observations:
- Low load: 10ms latency (process individually)
- High load: 100ms latency (batch 500 items)

Adaptive batching keeps latency within bounds.

## Tweet 8/8

Bidirectional streaming isn't just an API design - it's a different way of thinking about memory systems.

Not as databases with transactions.
As cognitive processes with bounded uncertainty.

Probably right, not definitely consistent.

Open source at github.com/engramhq/engram

---

Key metrics:
- 100K observations/second sustained
- <10ms P99 latency (low load)
- <100ms P99 latency (high load)
- Zero data loss (eventual consistency guarantee)
- Zero silent drops (admission control)

Built with Rust + gRPC + crossbeam lock-free structures

## Bonus Thread: Technical Deep Dive

**Thread on Sequence Number Protocol:**

Why client-generated sequences?

Alternative: server-generated on commit. Gives linearizable global order. But requires lock/counter on write path.

With client-generated: AtomicU64 fetch_add(1, SeqCst). No network roundtrip. No coordination.

Trade-off: cross-session ordering undefined. But that's fine - your brain has same property. Events from different sensory streams don't have global order.

**Thread on gRPC vs Alternatives:**

Why gRPC over WebSocket?

WebSocket:
- ~50K ops/sec throughput
- Manual flow control (you write the protocol)
- Good browser support

gRPC:
- ~100K+ ops/sec throughput
- Built-in flow control (HTTP/2 backpressure)
- Type safety (protobuf schema)

We use both: gRPC for backends, WebSocket for browsers.

**Thread on Lock-Free Queue:**

How does the queue achieve 5M ops/sec enqueue rate?

crossbeam::queue::SegQueue uses lock-free linked segments.

Push: CAS on tail pointer
Pop: CAS on head pointer

No mutex. No blocking. Scales linearly with cores.

Critical for streaming - any lock on write path would be fatal at 100K ops/sec.

**Thread on Eventual Consistency:**

"Eventual consistency" has bad reputation (thanks, NoSQL hype).

But for cognitive systems, it's *correct* model.

Your brain doesn't crash if you misremember something. Engram doesn't crash if observations arrive out of global order.

Both are "probably right" - and that's a feature, not a bug.

**Thread on Flow Control Math:**

Queue capacity: 100K items
Observation rate: 100K/sec
Index rate: 100K/sec (4 workers × 25K each)

Steady state: queue ~50% full (natural variation)

Spike to 200K/sec:
- t=0s: queue 50K items (50%)
- t=1s: queue 150K items (150% - overflow!)

Backpressure activates at 80K items (80%).
Client reduces to 50K/sec.
Queue drains to equilibrium.

No crashes. Graceful degradation.

## Visual ASCII Diagrams

```
Bidirectional Streaming:

Client                           Server
  |                                |
  |---- StreamInit -------------->|
  |<--- StreamInitAck (session)---|
  |                                |
  |---- Observation (seq=0) ----->|
  |<--- Ack (seq=0) --------------|
  |                                |
  |---- Observation (seq=1) ----->|
  |---- RecallRequest ----------->|
  |<--- Ack (seq=1) --------------|
  |<--- RecallResults ------------|
  |                                |
  |---- Observation (seq=2) ----->|
  |<--- Ack (seq=2) --------------|
  |                                |

One connection, two directions, continuous flow
```

```
Backpressure Activation:

Queue Depth
    |
100%|                    ╱REJECT
 90%|                 ╱
 80%|             ╱BACKPRESSURE
 50%|         ╱
    |     ╱ NORMAL
  0%|─────────────────────────>
        t=0  t=1  t=2  t=3    Time

Soft limit (80%): send backpressure signal
Hard limit (90%): reject new observations
```

```
Worker Pool Architecture:

                     ╔═══════════════╗
                     ║ Observation   ║
                     ║ Queue         ║
                     ║ (SegQueue)    ║
                     ╚═══╦═══════════╝
                         ║
         ╔═══════════════╩═══════════════╗
         ║                               ║
    ╔════▼════╗  ╔════▼════╗  ╔════▼════╗
    ║Worker 1 ║  ║Worker 2 ║  ║Worker 4 ║
    ║25K/sec  ║  ║25K/sec  ║  ║25K/sec  ║
    ╚════╦════╝  ╚════╦════╝  ╚════╦════╝
         ║           ║             ║
         ╚═══════════╩═════════════╝
                     ║
              ╔══════▼═══════╗
              ║ HNSW Index   ║
              ║ 100K obs/sec ║
              ╚══════════════╝
```

## Engagement Hooks

What would make this thread go viral in tech circles?

1. **Performance numbers:** "100K ops/sec" - people love concrete benchmarks
2. **Brain metaphor:** "Your brain encodes while retrieving" - makes it relatable
3. **Controversy:** "Eventual consistency is correct for cognitive systems" - takes a stance
4. **Visual diagrams:** ASCII art for protocol flow
5. **Open source:** github.com/engramhq/engram - people can try it

## Call to Action

"Building AI that needs to remember at scale? Check out Engram's streaming interface.

100K observations/sec with <100ms bounded staleness. Open source, Rust-based, cognitive memory architecture.

Star us on GitHub if this solves a problem you have."
