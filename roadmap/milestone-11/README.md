# Milestone 11: Streaming Interface for Real-Time Memory Operations

## Overview

Implement bidirectional streaming for continuous memory observation and real-time recall with lock-free incremental indexing. Target: sustained 100K observations/second with concurrent recalls while maintaining temporal ordering guarantees and sub-10ms P99 latency.

## Architectural Foundation

### Consistency Model: Eventual with Bounded Staleness

Not linearizable. Not strong consistency. **Eventual consistency with probabilistic temporal bounds**.

**Invariants:**
1. All observations within a stream session are totally ordered by sequence number
2. Cross-stream ordering is undefined - accept probabilistic interleaving
3. Index visibility has bounded staleness: observation visible in recall within 100ms P99
4. Stream failure does not corrupt index state - updates are atomic at item granularity
5. Backpressure preserves FIFO ordering - no reordering under load

**Why this model:** Cognitive memory formation is inherently probabilistic. Human episodic memory does not provide linearizable guarantees - you cannot perfectly reconstruct the temporal order of events encoded in parallel. We model biological memory formation, not transactional databases.

### Temporal Ordering Guarantees

- **Intra-stream**: Sequence numbers provide total ordering within single stream
- **Inter-stream**: Wall-clock timestamps provide partial ordering across streams
- **Recall consistency**: Snapshot isolation - recall sees all observations committed before stream start + probabilistically-available recent observations
- **Under load**: Backpressure blocks writers, maintaining FIFO order. No silent reordering.

### Lock-Free Incremental Indexing

**Critical insight:** HNSW index insertion is inherently lock-free. The existing `CognitiveHnswIndex` uses `crossbeam` concurrent structures. The "lock-free" requirement means:

1. **No mutex on write path** - Use existing `ArrayQueue<HnswUpdate>` pattern
2. **Atomic visibility** - Insertions are atomic, but visibility to reads is eventual
3. **No blocking readers** - Search continues during insertion using consistent snapshots
4. **Bounded wait-freedom** - Writers never starve, even under contention

**Implementation approach:**
- Extend existing async HNSW worker pattern from `MemoryStore`
- Replace `ArrayQueue` with `crossbeam::queue::SegQueue` (unbounded for streaming)
- Add priority lanes for immediate vs batch insertions
- Batch coalescing: merge N observations into single HNSW batch when queue depth > threshold

### Backpressure Mechanism

**Failure mode:** Stream producer faster than index consumer. Queue grows unbounded. Memory exhaustion.

**Solution:** Adaptive backpressure with three strategies:

1. **Flow Control Messages (Primary):**
   - Client sends buffer status via `FlowControl` messages
   - Server tracks per-stream queue depth
   - When queue depth > 0.8 * capacity, send `SLOW_DOWN` status
   - Client reduces send rate or pauses

2. **Server-Side Admission Control:**
   - Reject new observations when global queue depth > 90% capacity
   - Return gRPC error with `RESOURCE_EXHAUSTED` code
   - Include retry-after estimate based on current drain rate

3. **Adaptive Batching:**
   - Under normal load: process observations immediately
   - Under pressure (queue > 50%): coalesce into batches of 100-1000
   - Trade latency (10ms → 50ms) for throughput (10K → 100K ops/sec)

**Critical property:** No silent drops. System either accepts observation (eventual consistency) or returns error (client must retry). Never silently discard data.

### Performance Architecture: 100K ops/sec

**Bottleneck analysis:**
- Single-threaded HNSW insertion: ~10K insertions/sec
- Lock contention on shared index: Fatal for streaming
- Memory allocation per observation: Unsustainable at 100K/sec

**Solution:**

1. **Parallel HNSW Workers (4-8 threads):**
   - Partition index by memory space (natural sharding)
   - Each worker owns subset of spaces, no cross-worker coordination
   - Load balancing via work stealing when worker idle

2. **Zero-Copy Observation Pipeline:**
   ```
   gRPC receive → Arc<Episode> → SegQueue → Worker → HNSW
   ```
   - No intermediate copies
   - `Arc` reference counting, not cloning 768-dim vectors
   - Memory pool for Episode allocation (reuse freed instances)

3. **Batch-Aware HNSW Insertion:**
   - Existing HNSW inserts one node at a time
   - Add `insert_batch(&[Arc<Memory>])` method
   - Amortize graph lock acquisition across batch
   - Pre-sort by expected layer for cache locality

4. **Lock-Free Write Path:**
   ```rust
   fn observe(episode: Episode) -> Result<(), StreamError> {
       // 1. Atomic increment sequence number
       let seq = self.sequence.fetch_add(1, Ordering::SeqCst);

       // 2. Non-blocking enqueue (or fail fast)
       let update = IndexUpdate::Insert {
           memory_id: episode.id.clone(),
           memory: Arc::new(episode.into()),
           generation: seq,
           priority: UpdatePriority::Normal,
       };

       self.update_queue.push(update)?; // SegQueue::push is lock-free

       // 3. Notify SSE subscribers (best-effort, non-blocking)
       let _ = self.event_tx.try_send(MemoryEvent::Stored { .. });

       Ok(())
   }
   ```

**Measurement points:**
- Queue depth histogram (per-space, per-worker)
- Observation latency distribution (gRPC receive → index visible)
- Backpressure activation frequency
- Worker utilization (idle vs busy time)

## Tasks Breakdown

### Task 001: Streaming Protocol Foundation (3 days)

**File:** `001_streaming_protocol_foundation_pending.md`

**Objective:** Define protobuf messages for bidirectional streaming, implement gRPC service handlers, establish session management with sequence numbers.

**Deliverables:**
- Extend `proto/engram/v1/service.proto` with streaming-specific messages
- Add `ObservationRequest/Response` for push operations
- Add `StreamingRecallRequest/Response` for pull operations
- Implement session lifecycle (connect → active → pause → resume → close)
- Sequence number generation (per-session monotonic counter)

**Acceptance criteria:**
- Client can open bidirectional stream
- Send 10 observations, receive ack for each with sequence number
- Interleave recall requests, receive results
- Session survives 60s idle timeout
- Clean shutdown returns final sequence number

**Files to modify:**
- `proto/engram/v1/service.proto`
- `proto/engram/v1/memory.proto` (add streaming-specific fields)
- `engram-proto/src/lib.rs` (add convenience methods)

**Testing approach:**
- Property test: sequence numbers are monotonic
- Chaos test: random interleaving of observe/recall
- Load test: 1K observations/sec for 60s

### Task 002: Lock-Free Observation Queue (2 days)

**File:** `002_lockfree_observation_queue_pending.md`

**Objective:** Replace bounded `ArrayQueue` with unbounded `SegQueue` for streaming observations. Add priority lanes and backpressure detection.

**Deliverables:**
- Extend `IndexUpdate` enum with streaming-specific variants
- Implement `ObservationQueue` wrapper around `SegQueue<IndexUpdate>`
- Add queue depth metrics (per-space histograms)
- Backpressure detector: emit signal when depth > threshold

**Acceptance criteria:**
- Queue accepts 1M insertions without blocking
- Priority insertions (immediate) jump queue
- Queue depth metrics accurate under concurrent access
- Backpressure signal triggers at configured threshold (default 80%)

**Files to create:**
- `engram-core/src/streaming/observation_queue.rs`

**Files to modify:**
- `engram-core/src/index/mod.rs` (export streaming types)
- `engram-core/src/streaming/mod.rs` (create module)

**Testing approach:**
- Concurrent enqueue from 8 threads, verify FIFO order per thread
- Priority test: insert high-priority item, verify it's dequeued before normal
- Backpressure test: fill queue to 85%, verify signal emitted

### Task 003: Parallel HNSW Worker Pool (4 days)

**File:** `003_parallel_hnsw_worker_pool_pending.md`

**Objective:** Implement multi-threaded HNSW update workers with memory-space sharding and work stealing for load balancing.

**Deliverables:**
- `StreamingIndexWorker` struct managing N worker threads
- Per-space queue partitioning (hash memory_space_id to worker)
- Work stealing when worker idle (check other queues)
- Batch coalescing: merge observations into HNSW batch inserts
- Graceful shutdown: drain queues before exit

**Acceptance criteria:**
- 4-worker pool sustains 40K insertions/sec (10K each)
- Load imbalance < 20% (max worker throughput / min worker throughput < 1.2)
- Work stealing activates when one space has 10x more observations
- Clean shutdown: all queued observations processed within 5s

**Files to create:**
- `engram-core/src/streaming/worker_pool.rs`
- `engram-core/src/streaming/work_stealing.rs`

**Files to modify:**
- `engram-core/src/index/hnsw_construction.rs` (add batch insert)
- `engram-core/src/store.rs` (integrate worker pool)

**Testing approach:**
- Benchmark: 100K insertions with 1/2/4/8 workers, measure scaling
- Chaos test: random space distribution, verify work stealing balances load
- Shutdown test: enqueue 10K items, shutdown, verify all processed

### Task 004: Batch-Aware HNSW Insertion (3 days)

**File:** `004_batch_aware_hnsw_insertion_pending.md`

**Objective:** Optimize HNSW for batch insertions by amortizing lock acquisition, pre-sorting by layer, and cache-aware insertion order.

**Deliverables:**
- `HnswGraph::insert_batch(&[Arc<Memory>])` method
- Pre-sort memories by expected layer (reduces layer transitions)
- Amortize entry point lookup across batch
- Batch size tuning: test 10/100/1000, find sweet spot

**Acceptance criteria:**
- `insert_batch(100)` is 3-5x faster than 100x `insert()`
- Batch insertion maintains graph invariants (connectivity, layer distribution)
- No memory leaks (valgrind clean after 1M batch insertions)
- Optimal batch size identified via benchmark (likely 100-500)

**Files to modify:**
- `engram-core/src/index/hnsw_construction.rs`
- `engram-core/src/index/hnsw_graph.rs`

**Testing approach:**
- Correctness: insert 10K nodes batch vs sequential, verify identical graph
- Performance: benchmark batch sizes 1, 10, 100, 1000
- Validation: run graph integrity checks after each batch

### Task 005: Bidirectional gRPC Streaming (3 days)

**File:** `005_bidirectional_grpc_streaming_pending.md`

**Objective:** Implement gRPC handlers for `ObserveStream` (push observations) and `RecallStream` (pull recalls) with flow control.

**Deliverables:**
- `ObserveStream` handler: receive observations, enqueue for indexing
- `RecallStream` handler: execute recalls, stream results incrementally
- `MemoryFlow` handler: bidirectional stream (both observe + recall)
- Flow control: honor client `FlowControl` messages (pause/resume/slow_down)
- Error handling: return gRPC status codes with retry guidance

**Acceptance criteria:**
- Client streams 1K observations, receives acks with sequence numbers
- Client sends recall request mid-stream, receives results while observations continue
- Flow control: client pauses, server stops sending results, client resumes
- Error recovery: server restarts, client reconnects with last sequence number

**Files to create:**
- `engram-server/src/grpc/streaming.rs`

**Files to modify:**
- `engram-server/src/grpc/service.rs` (add streaming endpoints)
- `proto/engram/v1/service.proto` (add streaming RPCs)

**Testing approach:**
- Integration test: real gRPC client + server, stream 10K observations
- Chaos test: randomly inject network delays, verify flow control activates
- Reconnection test: kill server mid-stream, restart, client resumes

### Task 006: Backpressure and Admission Control (2 days)

**File:** `006_backpressure_admission_control_pending.md`

**Objective:** Implement adaptive backpressure with flow control messages, server-side admission control, and adaptive batching under load.

**Deliverables:**
- Queue depth monitoring (per-space, global)
- Admission control: reject observations when queue > 90%
- Adaptive batching: coalesce observations when queue > 50%
- Flow control emitter: send `SLOW_DOWN`/`BUFFER_FULL` to clients
- Metrics: backpressure activation rate, rejected observations count

**Acceptance criteria:**
- Overload test: 200K observations/sec → queue fills → admission control rejects
- Backpressure test: 150K obs/sec → adaptive batching activates → sustains load
- Flow control test: server sends `SLOW_DOWN`, client reduces rate
- No OOM: queue never grows beyond configured limit (default 100K items)

**Files to create:**
- `engram-core/src/streaming/backpressure.rs`

**Files to modify:**
- `engram-core/src/streaming/worker_pool.rs` (integrate adaptive batching)
- `engram-server/src/grpc/streaming.rs` (emit flow control messages)

**Testing approach:**
- Load test: gradually increase observation rate, measure queue depth
- Overload test: sudden 10x spike, verify admission control activates
- Recovery test: overload → back off → verify normal operation resumes

### Task 007: Incremental Recall with Snapshot Isolation (3 days)

**File:** `007_incremental_recall_snapshot_isolation_pending.md`

**Objective:** Implement snapshot-isolated recall that returns committed observations + probabilistically-available recent observations.

**Deliverables:**
- Snapshot timestamp: capture at recall request start
- Index visibility: HNSW nodes committed before snapshot
- Recent observations: best-effort inclusion of in-flight updates
- Incremental streaming: return results as HNSW search progresses
- Confidence scoring: lower confidence for recent observations

**Acceptance criteria:**
- Recall sees all observations committed before request (no phantom reads)
- Recall may see recent observations (bounded staleness: 100ms P99)
- Incremental streaming: first result within 10ms, full results within 100ms
- Confidence scoring: recent observations marked with lower confidence

**Files to modify:**
- `engram-core/src/index/hnsw_search.rs` (snapshot-based search)
- `engram-core/src/streaming/recall.rs` (incremental result streaming)
- `engram-core/src/query/executor.rs` (integrate snapshot isolation)

**Testing approach:**
- Correctness: stream 1000 observations, recall sees all + some in-flight
- Timing test: measure visibility latency (observation → visible in recall)
- Incremental test: recall 10K results, verify first result < 10ms

### Task 008: WebSocket Streaming (2 days)

**File:** `008_websocket_streaming_pending.md`

**Objective:** Add WebSocket endpoint for browser clients, mirroring gRPC streaming functionality with same flow control.

**Deliverables:**
- WebSocket endpoint: `/v1/stream` for observations + recalls
- JSON message format (compatible with protobuf schema)
- Flow control via JSON messages (same semantics as gRPC)
- Heartbeat/keepalive every 30s

**Acceptance criteria:**
- Browser client streams observations via WebSocket
- Same flow control behavior as gRPC (pause/resume)
- Auto-reconnect on disconnect with session recovery
- Performance: 10K observations/sec per WebSocket connection

**Files to create:**
- `engram-server/src/http/websocket.rs`

**Files to modify:**
- `engram-server/src/http/mod.rs` (add WebSocket route)

**Testing approach:**
- Browser integration test: stream 1K observations from JavaScript client
- Reconnection test: disconnect mid-stream, reconnect, verify no data loss
- Load test: 10 concurrent WebSocket connections, 10K obs/sec each

### Task 009: Chaos Testing Framework (3 days)

**File:** `009_chaos_testing_framework_pending.md`

**Objective:** Build chaos engineering harness to validate correctness under failures: delays, packet loss, worker crashes, queue overflows.

**Deliverables:**
- Delay injector: random delays 0-100ms on observation path
- Packet loss simulator: drop random observations (client must retry)
- Worker crash simulator: kill random HNSW worker, verify recovery
- Queue overflow test: exceed queue capacity, verify admission control
- Eventual consistency validator: all observations eventually visible

**Acceptance criteria:**
- Chaos test runs for 10 minutes with continuous failures
- Zero data loss: all accepted observations eventually indexed
- Zero corruption: HNSW graph validation passes throughout
- Performance degradation: P99 latency < 100ms even under chaos

**Files to create:**
- `engram-core/tests/chaos/streaming_chaos.rs`
- `engram-core/tests/chaos/fault_injector.rs`

**Testing approach:**
- 10-minute chaos run: 100K observations with random failures
- Validate: all observations visible in final recall
- Graph integrity: validate HNSW structure after chaos
- Performance: measure P50/P99/P99.9 latency distribution

### Task 010: Performance Benchmarking and Tuning (2 days)

**File:** `010_performance_benchmarking_tuning_pending.md`

**Objective:** Validate 100K observations/sec target with concurrent recalls. Identify bottlenecks, tune parameters, establish production baselines.

**Deliverables:**
- Benchmark suite: 10K, 50K, 100K, 200K observations/sec
- Concurrent recall benchmark: streaming observations + recalls every 100ms
- Bottleneck analysis: flamegraph, perf profiling
- Parameter tuning: worker count, batch size, queue capacity
- Production baselines: P50/P99/P99.9 latency, throughput limits

**Acceptance criteria:**
- Sustained 100K observations/sec for 60s with 4-core CPU
- Concurrent recalls: 10 recalls/sec with < 20ms P99 latency
- Memory usage: < 2GB for 1M observations
- CPU usage: < 80% during sustained load

**Files to create:**
- `engram-core/benches/streaming_throughput.rs`
- `engram-core/benches/concurrent_recall.rs`

**Testing approach:**
- Throughput ramp: 10K → 50K → 100K → 200K obs/sec
- Latency distribution: measure P50/P99/P99.9 at each load level
- Resource usage: track CPU, memory, queue depth
- Tune: adjust worker count, batch size based on results

### Task 011: Production Monitoring and Metrics (2 days)

**File:** `011_production_monitoring_metrics_pending.md`

**Objective:** Add Prometheus metrics for streaming operations, Grafana dashboards, alerting rules for queue depth and backpressure.

**Deliverables:**
- Metrics: observation rate, queue depth, worker utilization, backpressure rate
- Histograms: observation latency, recall latency
- Grafana dashboard: streaming health overview
- Alerts: queue depth > 80%, backpressure active > 5 min, worker crashed

**Acceptance criteria:**
- Metrics exported on `/metrics` endpoint
- Grafana dashboard visualizes streaming health
- Alerts fire when queue depth exceeds threshold
- Metrics have < 1% overhead at 100K obs/sec

**Files to modify:**
- `engram-core/src/streaming/metrics.rs` (add streaming metrics)
- `engram-server/grafana/streaming_dashboard.json` (create dashboard)

**Testing approach:**
- Load test: verify metrics accuracy at 100K obs/sec
- Alert test: trigger queue overflow, verify alert fires
- Overhead test: measure metrics cost (should be < 1% CPU)

### Task 012: Integration Testing and Documentation (2 days)

**File:** `012_integration_testing_documentation_pending.md`

**Objective:** End-to-end integration tests for streaming workflows, client examples (Rust, Python, TypeScript), operational runbook.

**Deliverables:**
- Integration tests: observe → recall → verify results
- Client examples: Rust gRPC client, Python client, TypeScript WebSocket client
- Operational runbook: streaming configuration, monitoring, troubleshooting
- Performance tuning guide: worker count, batch size, queue capacity

**Acceptance criteria:**
- Integration test: stream 10K observations, recall, verify all present
- Client examples: Rust/Python/TypeScript clients successfully stream
- Runbook: operators can configure and monitor streaming
- Tuning guide: provides parameters for different workload profiles

**Files to create:**
- `engram-core/tests/integration/streaming_workflow.rs`
- `examples/streaming/rust_client.rs`
- `examples/streaming/python_client.py`
- `examples/streaming/typescript_client.ts`
- `docs/operations/streaming.md`

**Testing approach:**
- Run all client examples, verify they complete successfully
- Follow runbook, verify streaming can be configured and monitored
- Load test: use runbook tuning guide to optimize for 200K obs/sec

## Critical Path

```
001 (Protocol) → 002 (Queue) → 003 (Workers) → 004 (Batch HNSW)
                                    ↓
005 (gRPC Streaming) → 006 (Backpressure) → 007 (Recall)
                                    ↓
008 (WebSocket) → 009 (Chaos Testing) → 010 (Performance)
                                    ↓
                    011 (Monitoring) → 012 (Integration)
```

**Parallel tracks:**
- Tasks 001-004: Core infrastructure (can parallelize 002 + 004)
- Tasks 005-007: Streaming interfaces (sequential)
- Tasks 008-010: Validation and optimization (can parallelize 008 + 009)
- Tasks 011-012: Production readiness (sequential)

## Risk Analysis and Mitigation

### Risk 1: Lock Contention in HNSW Insert

**Probability:** HIGH
**Impact:** CRITICAL (blocks 100K ops/sec target)

**Mitigation:**
- Validate lock-free assumption with microbenchmark before Task 003
- If contention detected, implement per-layer locks or optimistic concurrency
- Fallback: partition index by memory space (natural sharding)

### Risk 2: Unbounded Queue Growth

**Probability:** MEDIUM
**Impact:** HIGH (OOM crash)

**Mitigation:**
- Admission control is fail-safe (Task 006)
- Queue capacity limit enforced at enqueue time
- Monitoring alerts when queue depth > 80%
- Load shedding: prioritize recalls over observations when overloaded

### Risk 3: Temporal Ordering Violations

**Probability:** MEDIUM
**Impact:** HIGH (correctness bug)

**Mitigation:**
- Sequence numbers provide intra-stream total order
- Property testing: verify no reordering under concurrent load
- Chaos testing: inject delays, verify ordering preserved
- Fallback: if ordering violated, reject stream with error

### Risk 4: Snapshot Isolation Complexity

**Probability:** MEDIUM
**Impact:** MEDIUM (performance degradation)

**Mitigation:**
- Start with simple snapshot: "all committed before timestamp"
- Incremental: add best-effort recent observations
- Measure visibility latency in Task 007
- If visibility > 100ms P99, add fast-path for recent observations

### Risk 5: WebSocket Scalability

**Probability:** LOW
**Impact:** MEDIUM (browser clients limited)

**Mitigation:**
- Design for 100 concurrent WebSocket connections (not 10K)
- Connection pooling for high-concurrency scenarios
- Clear documentation: gRPC preferred for high-throughput clients
- Fallback: SSE for read-only streaming (simpler protocol)

## Success Metrics

**Performance:**
- Sustained 100K observations/sec for 60s with 4-core CPU ✓
- Concurrent recalls: 10 recalls/sec with < 20ms P99 latency ✓
- Index visibility: observation → recall latency < 100ms P99 ✓

**Correctness:**
- Zero data loss in chaos testing (10 min run) ✓
- Zero HNSW graph corruption under concurrent updates ✓
- Temporal ordering: no reordering within stream ✓

**Production Readiness:**
- Grafana dashboard for streaming health ✓
- Alerting for queue depth and backpressure ✓
- Client examples in Rust, Python, TypeScript ✓
- Operational runbook for configuration and tuning ✓

## Estimated Timeline

- Tasks 001-004: Foundation (12 days)
- Tasks 005-007: Streaming (8 days)
- Tasks 008-010: Validation (7 days)
- Tasks 011-012: Production (4 days)

**Total: 31 days (6.2 weeks) with single engineer**

**Parallelization opportunities:**
- 2 engineers: 18 days (3.6 weeks)
- 3 engineers: 14 days (2.8 weeks)

**Recommended:** 2 engineers, 18 days. One focuses on core (001-004), other on interfaces (005-008). Converge for validation (009-012).
