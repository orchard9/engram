# Milestone 11: Streaming Interface - Agent Implementation Plan

**Objective**: Bidirectional streaming for continuous memory observation and real-time recall with lock-free incremental indexing

**Performance Target**: Sustained 100K observations/second with concurrent recalls at <10ms P99 latency

**Consistency Model**: Eventual consistency with bounded staleness (100ms P99 visibility)

**Timeline**: 18 days with 2 engineers (31 days single engineer)

**Validation**: 10-minute chaos test with zero data loss under continuous failures

---

## Table of Contents

1. [Agent Assignment Matrix](#agent-assignment-matrix)
2. [Implementation Schedule](#implementation-schedule)
3. [Task-by-Task Plan](#task-by-task-plan)
4. [Critical Path Analysis](#critical-path-analysis)
5. [Risk Assessment](#risk-assessment)
6. [Success Metrics](#success-metrics)

---

## Agent Assignment Matrix

| Task | Agent(s) | Rationale | Estimated Effort |
|------|----------|-----------|------------------|
| 001: Protocol Foundation | rust-graph-engine-architect | Expert in Rust gRPC, protocol design, session management | 3 days |
| 002: Lock-Free Queue | systems-architecture-optimizer | Expert in lock-free data structures, NUMA, concurrent systems | 2 days |
| 003: Worker Pool | systems-architecture-optimizer + rust-graph-engine-architect | Work stealing (sys-arch), HNSW integration (rust-graph) | 4 days |
| 004: Batch HNSW | rust-graph-engine-architect | Expert in graph algorithms, cache optimization, HNSW internals | 3 days |
| 005: gRPC Streaming | rust-graph-engine-architect | Expert in Tokio async, bidirectional gRPC, flow control | 3 days |
| 006: Backpressure | systems-architecture-optimizer | Expert in admission control, flow control, backpressure mechanisms | 2 days |
| 007: Snapshot Recall | rust-graph-engine-architect | Expert in concurrent reads, snapshot isolation, MVCC patterns | 3 days |
| 008: WebSocket | rust-graph-engine-architect | Protocol implementation, async WebSocket handlers | 2 days |
| 009: Chaos Testing | verification-testing-lead | Expert in chaos engineering, fault injection, correctness validation | 3 days |
| 010: Performance Tuning | verification-testing-lead | Expert in benchmarking, profiling, performance optimization | 2 days |
| 011: Monitoring | technical-communication-lead | Documentation of metrics, dashboards, operational procedures | 2 days |
| 012: Integration & UAT | graph-systems-acceptance-tester | Production readiness validation, UAT execution, sign-off | 2 days |

---

## Implementation Schedule

### Week 1: Foundation (Days 1-5)

**Day 1-3: Task 001 - Protocol Foundation**
- Agent: rust-graph-engine-architect
- Deliverables: Protobuf messages, session management, sequence validation
- Parallel work: None (foundation task)

**Day 4-5: Task 002 - Lock-Free Queue**
- Agent: systems-architecture-optimizer
- Deliverables: SegQueue with priority lanes, backpressure detection
- Dependencies: Task 001 (protocol defines queue semantics)

### Week 2: Parallelization (Days 6-9)

**Days 6-9: Task 003 - Worker Pool (parallel with Task 004)**
- Agent: systems-architecture-optimizer + rust-graph-engine-architect
- Deliverables: Multi-threaded workers, work stealing, space-based sharding
- Dependencies: Task 002 (observation queue)

**Days 6-8: Task 004 - Batch HNSW (parallel with Task 003)**
- Agent: rust-graph-engine-architect
- Deliverables: Batch insertion API, 3-5x speedup
- Dependencies: Task 002 (can develop in parallel)
- Note: Critical path risk - if HNSW contention found, may need fallback

### Week 3: Streaming (Days 9-16)

**Days 9-11: Task 005 - gRPC Streaming**
- Agent: rust-graph-engine-architect
- Deliverables: Bidirectional stream handlers, flow control integration
- Dependencies: Tasks 001-004 (full pipeline)

**Days 12-13: Task 006 - Backpressure**
- Agent: systems-architecture-optimizer
- Deliverables: Adaptive batching, admission control, queue monitoring
- Dependencies: Task 005 (gRPC streaming)

**Days 14-16: Task 007 - Snapshot Recall**
- Agent: rust-graph-engine-architect
- Deliverables: Snapshot-isolated recall, incremental result streaming
- Dependencies: Task 006 (backpressure)

### Week 4: Validation & Production (Days 16-21)

**Days 16-17: Task 008 - WebSocket (parallel with Task 009)**
- Agent: rust-graph-engine-architect
- Deliverables: WebSocket protocol, browser-friendly streaming
- Dependencies: Task 007 (can run parallel)

**Days 16-18: Task 009 - Chaos Testing (parallel with Task 008)**
- Agent: verification-testing-lead
- Deliverables: Chaos harness, fault injection, 10-min sustained test
- Dependencies: Tasks 001-007 (full pipeline)

**Days 18-19: Task 010 - Performance Tuning**
- Agent: verification-testing-lead
- Deliverables: Profiling, optimization, 100K ops/sec validation
- Dependencies: Task 009 (chaos results inform tuning)

**Days 19-20: Task 011 - Monitoring**
- Agent: technical-communication-lead
- Deliverables: Prometheus metrics, Grafana dashboards, operational docs
- Dependencies: Task 010 (performance baselines)

**Days 20-21: Task 012 - Integration & UAT**
- Agent: graph-systems-acceptance-tester
- Deliverables: UAT report, production readiness checklist, sign-off
- Dependencies: All tasks complete

---

## Task-by-Task Plan

### Task 001: Streaming Protocol Foundation (3 days)

**Agent**: rust-graph-engine-architect

**Why This Agent**:
- Expert in gRPC protocol design and Rust async patterns
- Understands high-performance streaming systems
- Can design zero-copy message patterns

**Key Responsibilities**:
1. Design protobuf messages for bidirectional streaming
2. Implement session management with monotonic sequence numbers
3. Create gRPC service stubs with proper lifetime management
4. Validate sequence number protocol correctness

**Deliverables**:
- `proto/engram/v1/service.proto` (150 lines added)
- `engram-server/src/grpc/streaming.rs` (350 lines)
- `engram-core/src/streaming/session.rs` (200 lines)

**Success Criteria**:
- Client can initialize stream and receive session ID
- Send 1000 observations with monotonic sequences
- Out-of-order observation rejected with error
- Session survives 60s idle without timeout
- All tests pass (`cargo test streaming_protocol`)

**Risk**: Low - well-understood gRPC patterns

---

### Task 002: Lock-Free Observation Queue (2 days)

**Agent**: systems-architecture-optimizer

**Why This Agent**:
- Expert in lock-free data structures and concurrent algorithms
- Deep knowledge of crossbeam and atomic operations
- Understands cache behavior and NUMA architectures

**Key Responsibilities**:
1. Replace ArrayQueue with SegQueue for unbounded lock-free operation
2. Implement priority lanes (High/Normal/Low)
3. Add backpressure detection based on queue depth
4. Optimize for 4M+ ops/sec throughput

**Deliverables**:
- `engram-core/src/streaming/observation_queue.rs` (400 lines)
- `engram-core/src/streaming/queue_metrics.rs` (150 lines)

**Success Criteria**:
- Queue accepts 1M enqueues without blocking
- Priority ordering: High → Normal → Low
- Backpressure triggers at 80% capacity
- Concurrent safety: 4 threads enqueue + 2 dequeue, no data loss
- Performance: <500ns per enqueue+dequeue operation

**Risk**: Low - proven SegQueue implementation

---

### Task 003: Parallel HNSW Worker Pool (4 days)

**Agent**: systems-architecture-optimizer + rust-graph-engine-architect

**Why These Agents**:
- systems-architecture-optimizer: Expert in work stealing, load balancing, NUMA
- rust-graph-engine-architect: Expert in HNSW integration and graph algorithms

**Key Responsibilities**:
1. Implement multi-threaded worker pool (4-8 workers)
2. Design space-based sharding for zero contention
3. Implement work stealing for load balancing
4. Integrate adaptive batching based on queue depth

**Deliverables**:
- `engram-core/src/streaming/worker_pool.rs` (600 lines)
- `engram-core/src/streaming/work_stealing.rs` (250 lines)
- `engram-core/src/streaming/worker_stats.rs` (150 lines)

**Success Criteria**:
- 4-worker pool sustains 40K insertions/sec (10K per worker)
- 8-worker pool sustains 80K insertions/sec (linear scaling)
- Load imbalance <20%
- Work stealing activates when one queue >1000
- Graceful shutdown: all queued observations processed within 5s

**Risk**: MEDIUM - depends on HNSW lock contention (mitigation: pre-benchmark in Task 004)

---

### Task 004: Batch HNSW Insertion (3 days)

**Agent**: rust-graph-engine-architect

**Why This Agent**:
- Expert in HNSW algorithms and graph structure optimization
- Understands cache locality for batch operations
- Can optimize for 3-5x speedup through amortization

**Key Responsibilities**:
1. Design batch insertion API for HNSW
2. Amortize graph locks across batch
3. Optimize entry point selection for batch
4. Validate concurrent batch insertions (pre-benchmark for Task 003)

**Deliverables**:
- `engram-core/src/index/hnsw_construction.rs` (80 lines added)
- `engram-core/benches/batch_hnsw_insert.rs` (200 lines)

**Success Criteria**:
- Batch of 100: 3x faster than 100 individual inserts
- Batch of 500: 4x faster (25μs per item vs 100μs)
- Concurrent benchmark: 8 threads achieve 80K ops/sec
- If <60K ops/sec, implement fallback (per-layer locks)

**Risk**: HIGH - critical bottleneck, requires pre-validation

**Mitigation**: Run concurrent HNSW benchmark BEFORE Task 003 starts

---

### Task 005: gRPC Bidirectional Streaming (3 days)

**Agent**: rust-graph-engine-architect

**Why This Agent**:
- Expert in Tokio async runtime and gRPC streaming
- Understands bidirectional stream lifecycle management
- Can design flow control integration

**Key Responsibilities**:
1. Implement `ObserveStream` RPC handler
2. Implement `RecallStream` RPC handler
3. Implement `MemoryStream` bidirectional handler
4. Integrate with session manager and observation queue

**Deliverables**:
- `engram-server/src/grpc/streaming.rs` (500 lines total with Task 001)
- `examples/streaming/rust_client.rs` (200 lines)

**Success Criteria**:
- Stream 10K observations, receive acks for all
- Bidirectional stream: observe + recall simultaneously
- Flow control messages propagate correctly
- Graceful shutdown drains queue before closing

**Risk**: Low - well-understood gRPC patterns

---

### Task 006: Backpressure & Admission Control (2 days)

**Agent**: systems-architecture-optimizer

**Why This Agent**:
- Expert in flow control mechanisms and admission control
- Understands adaptive algorithms for backpressure
- Can design queue monitoring and alerting

**Key Responsibilities**:
1. Implement adaptive batching based on queue depth
2. Design admission control at 90% capacity
3. Integrate backpressure signaling with gRPC responses
4. Create monitoring dashboards for queue health

**Deliverables**:
- `engram-core/src/streaming/backpressure.rs` (250 lines)
- `engram-core/src/streaming/adaptive_batching.rs` (150 lines)

**Success Criteria**:
- Backpressure activates at 80% capacity
- Admission control rejects at 90% capacity
- Adaptive batching: 10 (low load) → 500 (high load)
- Queue never exceeds hard capacity
- Client receives backpressure signals

**Risk**: Low - standard flow control patterns

---

### Task 007: Snapshot-Isolated Recall (3 days)

**Agent**: rust-graph-engine-architect

**Why This Agent**:
- Expert in concurrent data structures and snapshot isolation
- Understands MVCC patterns and lock-free reads
- Can design incremental result streaming

**Key Responsibilities**:
1. Implement snapshot isolation for recalls
2. Design incremental result streaming
3. Ensure bounded staleness (100ms P99 visibility)
4. Handle concurrent reads during writes

**Deliverables**:
- `engram-core/src/index/hnsw_search.rs` (50 lines added)
- `engram-core/src/streaming/recall_stream.rs` (300 lines)

**Success Criteria**:
- Recall sees snapshot at request time
- Recent observations (< 100ms) may not be visible
- Concurrent recalls don't block writers
- Incremental results stream without buffering all

**Risk**: MEDIUM - snapshot isolation complexity

**Fallback**: If too complex, ship without snapshot isolation (document visibility lag)

---

### Task 008: WebSocket Streaming (2 days)

**Agent**: rust-graph-engine-architect

**Why This Agent**:
- Expert in async WebSocket protocol
- Understands browser compatibility requirements
- Can design protocol translation from gRPC

**Key Responsibilities**:
1. Implement WebSocket protocol handler
2. Translate between WebSocket and internal gRPC
3. Handle browser-specific constraints
4. Document WebSocket API for client developers

**Deliverables**:
- `engram-server/src/http/websocket.rs` (350 lines)
- `examples/streaming/typescript_client.ts` (200 lines)

**Success Criteria**:
- Browser client can stream observations via WebSocket
- Protocol compatible with gRPC semantics
- Handles connection drops gracefully
- Performance: >10K obs/sec per connection

**Risk**: Low - standard WebSocket patterns

---

### Task 009: Chaos Testing Framework (3 days)

**Agent**: verification-testing-lead

**Why This Agent**:
- Expert in chaos engineering and fault injection
- Understands eventual consistency validation
- Can design comprehensive failure scenarios

**Key Responsibilities**:
1. Build chaos harness with fault injection
2. Implement 10-minute sustained chaos test
3. Validate zero data loss and eventual consistency
4. Create failure scenario catalog

**Deliverables**:
- `engram-core/tests/chaos/streaming_chaos.rs` (500 lines)
- `engram-core/tests/chaos/fault_injector.rs` (300 lines)
- `engram-core/tests/chaos/validators.rs` (200 lines)

**Success Criteria**:
- 10-min chaos run with zero data loss
- All acked observations eventually indexed
- HNSW graph integrity maintained
- P99 latency <100ms under chaos
- System recovers after chaos stops

**Risk**: Low - chaos patterns well-understood

---

### Task 010: Performance Tuning (2 days)

**Agent**: verification-testing-lead

**Why This Agent**:
- Expert in profiling and performance optimization
- Can identify bottlenecks and validate optimizations
- Understands statistical performance analysis

**Key Responsibilities**:
1. Profile full streaming pipeline
2. Identify and optimize bottlenecks
3. Validate 100K ops/sec sustained throughput
4. Create performance regression tests

**Deliverables**:
- `engram-core/benches/streaming_performance.rs` (400 lines)
- Performance tuning report

**Success Criteria**:
- Sustained 100K obs/sec for 60s
- Concurrent recalls: 10/sec at <20ms P99
- Index visibility: <100ms P99
- CPU usage: <80% during sustained load

**Risk**: Low - performance targets validated in architecture

---

### Task 011: Monitoring & Operations (2 days)

**Agent**: technical-communication-lead

**Why This Agent**:
- Expert in operational documentation
- Understands monitoring and alerting requirements
- Can create user-friendly dashboards and runbooks

**Key Responsibilities**:
1. Design Prometheus metrics for streaming
2. Create Grafana dashboards
3. Write operational runbook
4. Document troubleshooting procedures

**Deliverables**:
- `docs/operations/streaming.md` (200 lines)
- `config/grafana/streaming_dashboard.json`
- `config/prometheus/streaming_alerts.yml`

**Success Criteria**:
- Queue depth, backpressure, latency metrics exported
- Grafana dashboard shows streaming health
- Alerts configured for critical thresholds
- Runbook enables operator to configure and troubleshoot

**Risk**: Low - standard monitoring patterns

---

### Task 012: Integration Testing & UAT (2 days)

**Agent**: graph-systems-acceptance-tester

**Why This Agent**:
- Expert in production readiness validation
- Understands graph system acceptance criteria
- Can execute comprehensive UAT

**Key Responsibilities**:
1. Execute comprehensive UAT test suite
2. Validate all acceptance criteria met
3. Create production readiness checklist
4. Obtain milestone sign-off

**Deliverables**:
- `docs/internal/milestone_11_uat.md`
- `docs/internal/milestone_11_performance_report.md`
- Production readiness sign-off

**Success Criteria**:
- All integration tests pass
- Performance targets validated
- Chaos test passes
- Documentation complete
- Milestone approved for production

**Risk**: Low - validation of completed work

---

## Critical Path Analysis

### Sequential Dependencies

```
001 → 002 → 003 → 005 → 006 → 007 → 009 → 010 → 011 → 012
            ↑
            004 (parallel with 003)

008 (parallel with 009)
```

### Critical Path (18 days)

1. **Task 001**: 3 days (protocol foundation)
2. **Task 002**: 2 days (lock-free queue) → Total: 5 days
3. **Task 003**: 4 days (worker pool) → Total: 9 days
4. **Task 005**: 3 days (gRPC streaming) → Total: 12 days
5. **Task 006**: 2 days (backpressure) → Total: 14 days
6. **Task 007**: 3 days (snapshot recall) → Total: 17 days
7. **Task 009**: 3 days (chaos testing) → Total: 20 days
8. **Task 010**: 2 days (performance) → Total: 22 days
9. **Task 011**: 2 days (monitoring) → Total: 24 days
10. **Task 012**: 2 days (UAT) → Total: 26 days

**Note**: With 2 engineers working in parallel:
- Engineer A: Tasks 001, 002, 003, 006, 007, 009
- Engineer B: Task 004, 005, 008, 010, 011, 012
- **Total duration**: 18 days (per IMPLEMENTATION_SPEC.md)

### Parallel Opportunities

**Week 1 (Days 6-9)**:
- Task 003 (Worker Pool) + Task 004 (Batch HNSW) in parallel
- Both agents can work independently

**Week 4 (Days 16-18)**:
- Task 008 (WebSocket) + Task 009 (Chaos) in parallel
- Different agents, no dependencies

---

## Risk Assessment

### Critical Risks

**Risk 1: HNSW Lock Contention (CRITICAL)**
- **Probability**: HIGH
- **Impact**: CRITICAL (blocks 100K ops/sec target)
- **Mitigation**: Pre-benchmark concurrent HNSW before Task 003
- **Fallback**: Per-layer locks, optimistic concurrency, space partitioning
- **Owner**: rust-graph-engine-architect (Task 004)

**Risk 2: Unbounded Queue Growth (HIGH)**
- **Probability**: MEDIUM
- **Impact**: HIGH (OOM crash)
- **Mitigation**: Soft limits (80%), hard limits (90%), monitoring alerts
- **Owner**: systems-architecture-optimizer (Task 006)

**Risk 3: Temporal Ordering Violations (MEDIUM)**
- **Probability**: MEDIUM
- **Impact**: HIGH (correctness bug)
- **Mitigation**: Sequence number validation, property testing
- **Owner**: rust-graph-engine-architect (Task 001)

**Risk 4: Snapshot Isolation Complexity (MEDIUM)**
- **Probability**: MEDIUM
- **Impact**: MEDIUM (degraded recall quality)
- **Mitigation**: Start simple, measure visibility latency, add fast-path if needed
- **Fallback**: Ship without snapshot isolation, document <100ms visibility lag
- **Owner**: rust-graph-engine-architect (Task 007)

### Schedule Risks

**Risk**: Task 003/004 dependencies critical path
- **Mitigation**: Pre-benchmark HNSW concurrency in Task 004 before Task 003 starts
- **Buffer**: 2-day contingency built into 18-day schedule

**Risk**: Chaos testing may discover blocking bugs
- **Mitigation**: Chaos runs parallel with WebSocket (Task 008), can extend if needed
- **Buffer**: Task 010 tuning can absorb chaos bug fixes

---

## Success Metrics

### Performance (Quantitative)

- **Throughput**: Sustained 100K observations/sec for 60s on 4-core CPU ✓
- **Concurrent recalls**: 10 recalls/sec with <20ms P99 latency ✓
- **Index visibility**: Observation → recall latency <100ms P99 ✓
- **Memory usage**: <2GB for 1M observations ✓
- **CPU usage**: <80% during sustained 100K ops/sec load ✓

### Correctness (Qualitative)

- **No data loss**: Chaos test (10 min) with 0 lost observations ✓
- **No corruption**: HNSW graph validation passes throughout chaos ✓
- **Temporal ordering**: No reordering within stream (property test) ✓
- **Graceful degradation**: Backpressure activates before OOM ✓

### Production Readiness (Operational)

- **Monitoring**: Grafana dashboard for streaming health ✓
- **Alerting**: Queue depth and backpressure alerts configured ✓
- **Client examples**: Rust, Python, TypeScript examples functional ✓
- **Runbook**: Operators can configure and troubleshoot streaming ✓

---

## Implementation Workflow

### For Each Task

1. **Read task specification** thoroughly
2. **Rename** from `_pending` to `_in_progress`
3. **Launch appropriate agent(s)** with detailed prompt
4. **Implement** according to task requirements
5. **Run tests** and validate acceptance criteria
6. **Run** `make quality` and fix ALL clippy warnings
7. **Verify** implementation matches requirements
8. **Rename** to `_complete`
9. **Commit** with detailed message

### Agent Prompts Should Include

- Link to task specification file
- Summary of deliverables
- Acceptance criteria
- Dependencies and integration points
- Testing requirements
- Performance targets
- Risk mitigations

### Quality Gates

Before marking task complete:
- All tests pass
- `make quality` passes with zero warnings
- Performance targets validated (if applicable)
- Documentation updated
- Integration points verified

---

## Appendices

### Appendix A: Agent Expertise Summary

**rust-graph-engine-architect**:
- Tasks: 001, 003 (joint), 004, 005, 007, 008
- Expertise: gRPC, async Rust, HNSW, graph algorithms, concurrent data structures
- Critical for: Protocol design, HNSW optimization, streaming implementation

**systems-architecture-optimizer**:
- Tasks: 002, 003 (joint), 006
- Expertise: Lock-free structures, work stealing, NUMA, admission control
- Critical for: Queue design, worker pool, backpressure

**verification-testing-lead**:
- Tasks: 009, 010
- Expertise: Chaos engineering, profiling, performance optimization
- Critical for: Correctness validation, performance tuning

**technical-communication-lead**:
- Tasks: 011
- Expertise: Documentation, monitoring, operational procedures
- Critical for: Production readiness, operator enablement

**graph-systems-acceptance-tester**:
- Tasks: 012
- Expertise: UAT execution, production validation, sign-off
- Critical for: Final validation and approval

### Appendix B: File Creation Summary

**New Files**: ~20 files, ~4,500 lines
- Protobuf: 1 file (150 lines)
- Streaming core: 8 files (2,000 lines)
- gRPC/WebSocket: 2 files (850 lines)
- Tests/chaos: 7 files (1,600 lines)
- Documentation: 2 files (400 lines)

**Modified Files**: ~6 files, ~300 line changes
- `proto/engram/v1/service.proto` (+150 lines)
- `engram-core/src/store.rs` (~100 line changes)
- `engram-core/src/index/hnsw_construction.rs` (+80 lines)
- `engram-core/src/index/hnsw_search.rs` (+50 lines)
- `engram-server/src/grpc/service.rs` (+20 lines)
- `Cargo.toml` (verify dependencies)

**Total**: ~4,800 new/modified lines

### Appendix C: Testing Strategy

**Unit Tests**: Per-component correctness
- Session management, sequence validation
- Queue priority ordering, backpressure
- Worker assignment, work stealing
- Batch HNSW correctness

**Integration Tests**: End-to-end workflows
- Streaming roundtrip (observe + recall)
- Flow control propagation
- Graceful shutdown

**Property Tests**: Invariant validation
- Sequence monotonicity
- No observation reordering
- Bounded queue capacity

**Chaos Tests**: Failure resilience
- Network delays (0-100ms)
- Packet loss (1%)
- Worker crashes (every 10s)
- Queue overflow (burst 10K)
- Combined chaos (10 minutes)

**Performance Tests**: Throughput and latency
- Sustained 100K ops/sec
- Concurrent recalls
- Latency distribution (P50, P99, P99.9)

### Appendix D: Deployment Considerations

**Hardware Requirements**:
- Minimum: 4-core CPU, 4GB RAM
- Recommended: 8-core CPU, 8GB RAM, NVMe SSD

**Configuration**:
- Workers: 4-8 (num_cpus)
- Queue capacity: 100K (normal), 10K (high), 50K (low)
- Batch sizes: 10 (low), 100 (medium), 500 (high load)

**Monitoring**:
- Queue depth by priority
- Backpressure event rate
- Worker utilization
- Latency distribution
- Visibility lag

---

**Document Version**: 1.0
**Last Updated**: 2025-10-25
**Author**: AI Planning Assistant
**Status**: Ready for Implementation
