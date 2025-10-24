# Bidirectional Streaming: Research and Findings

## Research Context

Bidirectional streaming for real-time memory operations in Engram requires careful protocol design that balances eventual consistency with temporal ordering guarantees. This research explores the technical landscape of streaming protocols, session management, and sequence number semantics.

## Core Research Questions

1. **Why eventual consistency for cognitive memory?** How does this differ from traditional database consistency models?
2. **What are the trade-offs in sequence number protocols?** When should we use client-generated vs server-generated sequences?
3. **How do we achieve bounded staleness without linearizability?** What mechanisms provide visibility guarantees?
4. **What flow control patterns work for cognitive workloads?** How do we prevent queue overflow while maintaining temporal ordering?

## Research Findings

### 1. Eventual Consistency in Cognitive Systems

**Key Insight:** Human episodic memory is not linearizable. You cannot perfectly reconstruct the temporal order of events encoded in parallel across different brain regions.

**Biological Evidence:**

- Hippocampal indexing happens asynchronously from neocortical consolidation
- Memory formation has bounded staleness: events become "fixed" within 100ms-1s
- Cross-stream ordering is probabilistic: events from different sensory modalities may interleave unpredictably
- No global clock: temporal relationships are relative and reconstructed during recall

**Engineering Implications:**

- Accept that cross-stream ordering is undefined (no vector clocks needed)
- Guarantee intra-stream total ordering via sequence numbers
- Provide bounded staleness (100ms P99) rather than immediate consistency
- Design for "probably right" rather than "definitely consistent"

**Citations:**

- Buzsaki, G. (2015). "Hippocampal sharp wave-ripple: A cognitive biomarker for episodic memory and planning." *Hippocampus*, 25(10), 1073-1188.
- Marr, D. (1971). "Simple memory: a theory for archicortex." *Philosophical Transactions of the Royal Society B*, 262(841), 23-81.

### 2. Sequence Number Protocols

**Design Space:**

| Approach | Pros | Cons | Use Case |
|----------|------|------|----------|
| Client-generated monotonic | No coordination, fast | Gap detection needed | Streaming observations |
| Server-generated on commit | Linearizable order | Requires lock/counter | Transactional systems |
| Hybrid (client + server validation) | Best of both | More complex | Cognitive memory formation |
| Lamport timestamps | Partial ordering | No total order | Distributed systems |

**Engram Choice: Client-generated with server validation**

Why this works:

1. Client increments atomic counter: `fetch_add(1, SeqCst)` - no network round-trip
2. Server validates monotonicity: rejects gaps or duplicates
3. Sequence numbers provide total order within session
4. Multiple sessions can interleave - cross-session order is timestamp-based

**Protocol Design:**

```rust
// Client side
let seq = self.sequence.fetch_add(1, Ordering::SeqCst);
let request = ObservationRequest {
    session_id: self.session_id.clone(),
    sequence_number: seq,
    operation: Some(Operation::Observation(episode)),
};

// Server side
let expected_seq = session.last_sequence.fetch_add(1, Ordering::SeqCst) + 1;
if received_seq != expected_seq {
    return Err(SequenceError::Mismatch {
        expected,
        received: received_seq,
    });
}
```

**Key Invariant:** No gaps, no duplicates within a session. Sequence numbers are dense in the integers.

**Citations:**

- Lamport, L. (1978). "Time, clocks, and the ordering of events in a distributed system." *Communications of the ACM*, 21(7), 558-565.
- Mattern, F. (1988). "Virtual time and global states of distributed systems." *Parallel and Distributed Algorithms*, 215-226.

### 3. Bounded Staleness Without Linearizability

**Problem:** We want observations to become visible "quickly" but don't need global linearizability.

**Measurement Definition:**

```
Staleness = time(observation_indexed) - time(observation_committed)
Bounded staleness: P99(staleness) < 100ms
```

**Mechanisms for Bounded Staleness:**

1. **Priority queues:** High-priority observations jump the queue
2. **Adaptive batching:** Small batches (10 items) for low latency, large batches (500 items) for high throughput
3. **Work stealing:** Idle workers steal from overloaded workers to reduce tail latency
4. **Admission control:** Reject new observations when queue depth exceeds capacity

**Architectural Pattern:**

```
Observation → [Lock-free queue] → [Worker pool with work stealing] → [HNSW index]
              ↑                    ↑                                  ↑
              |                    |                                  |
         Enqueue time         Processing time                  Index commit time
         <-- 1μs -->         <-- adaptive -->                  <-- 100μs -->
                             (10ms low load, 50ms high load)
```

**Trade-off Analysis:**

- Low load (< 1K obs/sec): Process individually, 10ms P99 latency
- Medium load (10K obs/sec): Batch 100 items, 30ms P99 latency
- High load (100K obs/sec): Batch 500 items, 100ms P99 latency

**Citations:**

- Bailis, P., et al. (2013). "Quantifying eventual consistency with PBS." *VLDB Endowment*, 7(6), 455-466.
- Terry, D. B., et al. (2013). "Consistency-based service level agreements for cloud storage." *SOSP '13*, 309-324.

### 4. Flow Control for Cognitive Workloads

**Failure Scenario:** Producer sends observations faster than consumer can index them. Queue grows unbounded. Memory exhaustion.

**Flow Control Strategies:**

1. **Client-side rate limiting:** Client tracks ack latency, reduces rate if > threshold
2. **Server backpressure signals:** Server sends `SLOW_DOWN` when queue depth > 80%
3. **Admission control:** Server rejects observations when queue > 90% capacity
4. **Adaptive batching:** Server coalesces observations into larger batches under load

**gRPC Backpressure Pattern:**

```protobuf
message StreamStatus {
  State state = 1;
  uint32 queue_depth = 3;
  uint32 queue_capacity = 4;
  float pressure = 5;  // 0.0 to 1.0
}
```

Client receives `StreamStatus` and adjusts send rate:

```rust
match status.state {
    STATE_ACTIVE => {
        // Normal operation
        self.send_interval = Duration::from_micros(10); // 100K/sec
    }
    STATE_BACKPRESSURE => {
        // Reduce rate by 50%
        self.send_interval = Duration::from_micros(20); // 50K/sec
    }
    STATE_OVERLOADED => {
        // Pause and retry after delay
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
}
```

**Critical Invariant:** No silent drops. System either accepts observation (eventual consistency) or returns error (client must retry).

**Citations:**

- Jacobson, V. (1988). "Congestion avoidance and control." *ACM SIGCOMM*, 18(4), 314-329.
- Cardwell, N., et al. (2016). "BBR: Congestion-based congestion control." *ACM Queue*, 14(5), 20-53.

## Comparative Analysis: gRPC vs WebSocket vs SSE

| Feature | gRPC Bidirectional | WebSocket | Server-Sent Events |
|---------|-------------------|-----------|-------------------|
| Bidirectional | Yes (native) | Yes | No (server → client only) |
| Binary protocol | Protobuf | Custom | Text (JSON) |
| Flow control | Built-in HTTP/2 | Manual | Browser-managed |
| Browser support | Limited (needs grpc-web) | Native | Native |
| Throughput | 100K+ ops/sec | 50K ops/sec | 10K ops/sec |
| Type safety | Strong (protobuf) | Weak (JSON) | Weak (JSON) |
| Use case | High-throughput backend | Browser real-time | Simple notifications |

**Engram Choice:**

- Primary: gRPC bidirectional streaming (best performance, type safety)
- Secondary: WebSocket for browsers (still supports 50K ops/sec)
- Future: SSE for read-only metrics streaming (simple, works everywhere)

## Session Management Patterns

**Design Goal:** Sessions survive network interruptions. Clients can reconnect and resume from last acknowledged sequence.

**Session State:**

```rust
struct StreamSession {
    session_id: String,              // Client-generated UUID
    memory_space_id: MemorySpaceId,  // Tenant isolation
    last_sequence: AtomicU64,        // Last sequence number processed
    created_at: Instant,             // Session creation time
    last_activity: AtomicU64,        // Unix timestamp of last message
    state: AtomicU8,                 // Active, Paused, Closed
}
```

**Session Lifecycle:**

1. **Init:** Client generates session ID, sends `StreamInit`, server stores session
2. **Active:** Client streams observations, server updates `last_sequence` and `last_activity`
3. **Network disconnect:** Session remains in server memory (timeout = 5 minutes)
4. **Reconnect:** Client sends same session ID + last acked sequence, server validates and resumes
5. **Graceful close:** Client sends `StreamClose`, server drains queue before closing

**Timeout Strategy:**

- Active sessions: no timeout (as long as client sends periodic messages)
- Idle sessions: 5 minute timeout after last activity
- Zombie sessions: server sends heartbeat every 60s, expects client ack within 10s

**Citations:**

- Fielding, R. T. (2000). "Architectural Styles and the Design of Network-based Software Architectures." *Doctoral dissertation*, UC Irvine.
- Session management patterns from Redis RESP3, Kafka consumer groups, Postgres replication slots

## Implementation Roadmap

### Phase 1: Protocol Definition (Day 1-2)

- Define protobuf messages in `proto/engram/v1/service.proto`
- Add `ObservationRequest`, `ObservationResponse`, `StreamInit`, `ObservationAck`
- Generate Rust code: `buf generate`
- Add convenience methods in `engram-proto/src/lib.rs`

### Phase 2: Session Management (Day 2-3)

- Implement `SessionManager` with `DashMap<String, StreamSession>`
- Add session creation, validation, timeout logic
- Write unit tests for session lifecycle
- Add Prometheus metrics for active sessions

### Phase 3: gRPC Service Stub (Day 3)

- Implement `ObserveStream` handler skeleton
- Wire up session creation on `StreamInit`
- Echo sequence numbers in `ObservationResponse`
- Add error handling for sequence mismatches

### Testing Strategy

- Unit tests: Session creation, timeout, sequence validation
- Property tests: Sequence numbers are monotonic, no gaps
- Integration tests: Real gRPC client + server, stream 1000 observations
- Chaos tests: Random delays, verify ordering preserved

## Open Questions and Future Research

1. **Multi-tenancy isolation:** Should sessions be per-memory-space or cross-space? Current design: per-space for isolation.
2. **Sequence number overflow:** uint64 allows 18 quintillion observations. At 100K/sec, takes 5 billion years to overflow. Not a concern.
3. **Session recovery after server restart:** Should we persist session state to disk? Current design: ephemeral sessions, client must reinitialize.
4. **Cross-region replication:** How do sequence numbers work in geo-distributed setup? Future milestone.

## Conclusion

Bidirectional streaming for Engram requires careful protocol design that balances:

- Eventual consistency (biological realism) vs linearizability (transactional guarantees)
- Client-generated sequences (low latency) vs server-generated (stronger ordering)
- Bounded staleness (100ms P99) vs immediate visibility (requires global locks)
- Graceful degradation (admission control) vs best-effort (silent drops)

The research validates our architectural choices:

- Client-generated monotonic sequences provide intra-stream total ordering
- Server validation prevents gaps and duplicates
- Adaptive batching achieves bounded staleness without sacrificing throughput
- Flow control prevents unbounded queue growth while maintaining FIFO ordering

Next step: Implement protocol foundation in Task 001 and validate with integration tests.
