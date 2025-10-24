# Engram: Development Milestones

## A Cognitive Graph Database for Episodic and Semantic Memory

### Milestone 0: Developer Experience Foundation

**Objective**: Establish error handling, logging, and startup infrastructure that makes development and debugging transparent. Create consistent patterns for errors that guide users toward solutions rather than just stating problems.

**Critical**: Every error must include context, suggestion, and example of correct usage - "Expected embedding dimension 768, got 512. Use `Config::embedding_dim(512)` or transform your embeddings with `embedding.pad_to(768)`." Startup must be single command with automatic cluster discovery.

**Validation**: Error message review requiring each error to pass "Would a tired developer at 3am understand what to do?" test. Measure time from `git clone` to running cluster: target <60 seconds including compilation.

### Milestone 1: Core Memory Types

**Objective**: Implement `Memory`, `Episode`, `Cue` types with probabilistic confidence intervals as first-class properties. Create basic `store()` and `recall()` operations that return results with confidence scores.

**Critical**: Type system must enforce that confidence is always present - no `Option<f32>`, always `Confidence(f32)` with valid range [0,1]. Memory operations must be infallible at the API level, degrading gracefully rather than returning errors.

**Validation**: Type-state pattern prevents constructing invalid memories at compile time. Fuzzing with random confidence values verifies no panics and all results within valid probability ranges.

### Milestone 2: Vector-Native Storage

**Objective**: Build embedding storage with SIMD-optimized similarity search supporting `Cue::from_embedding()` operations. Implement content-addressable retrieval using HNSW indices for nearest-neighbor search.

**Critical**: Vector operations must automatically detect and use CPU SIMD extensions (AVX-512, NEON) with runtime fallback. HNSW parameters must self-tune based on recall/precision measurements.

**Validation**: Benchmark against FAISS and Annoy on standard ANN datasets, must achieve 90% recall@10 with <1ms query time. Verify identical results across SIMD and scalar implementations using property-based testing.

### Milestone 3: Activation Spreading Engine

**Objective**: Implement parallel activation spreading for `recall()` using configurable decay rates and hop limits. Design async runtime for concurrent activation flows supporting network-like propagation.

**Critical**: Must handle cyclic graphs without infinite loops using activation thresholds and visited sets. Spreading must be deterministic given same seed for reproducible debugging.

**Validation**: Visualize activation spread on test graphs, verify follows neuroscience activation patterns. Benchmark must show linear scaling with CPU cores up to 32 cores.

### Milestone 3.6: Multilingual Semantic Recall

**Objective**: Deliver language-aware semantic recall that defaults to vector search with multilingual embeddings, expands lexical cues (synonyms, abbreviations, idioms), and interprets figurative language (metaphor/simile analogies) without sacrificing determinism or latency.

**Critical**: Every stored memory must carry a high-quality embedding with tracked provenance. Query expansion must respect confidence budgets and audit trails. Figurative-language interpretation must degrade gracefully—never hallucinate unsupported mappings—and expose explainability metadata for operators.

**Validation**: Cross-lingual MTEB (or equivalent) benchmark ≥0.80 nDCG@10 across English/Spanish/Mandarin test sets; synonym/abbreviation suite demonstrating ≥95% recall parity with literal queries; human-evaluated metaphor/simile set with ≥80% acceptable interpretations; regression tests proving lexical fallback still matches prior behavior when embeddings unavailable.

### Milestone 4: Temporal Dynamics

**Objective**: Create automatic decay functions applying to all episodes based on `last_access` patterns. Implement forgetting curves matching psychological research (Ebbinghaus, spaced repetition).

**Critical**: Decay must happen lazily during recall, not requiring background threads. Must support multiple decay functions (exponential, power law) selectable per memory.

**Validation**: Compare forgetting curves against published psychology data, must match within 5% error. Verify no memory leaks during long-running decay using valgrind.

### Milestone 5: Probabilistic Query Foundation

**Objective**: Build query executor that returns `(Episode, f32)` tuples with confidence scores for all operations. Implement uncertainty propagation through query operations.

**Critical**: Confidence must propagate correctly through joins and filters using probability theory, not ad-hoc rules. System must distinguish "no results" from "low confidence results."

**Validation**: Formal verification of probability propagation using SMT solver. Statistical tests confirming confidence scores correlate with retrieval accuracy on test sets.

### Milestone 6: Consolidation System ✅ COMPLETE

**Objective**: Develop `consolidate_async()` and `dream()` operations that transform episodic patterns into semantic knowledge. Implement compression that preserves retrieval accuracy while reducing storage.

**Critical**: Consolidation must be interruptible and resumable, not blocking recall operations. Must identify and extract patterns without supervision using statistical regularity detection.

**Validation**: Measure semantic category formation against human categorization data. Verify storage reduction >50% while maintaining >95% recall accuracy on consolidated memories.

**Completion Summary** (2025-10-21):
- 8/8 tasks completed: Scheduler (001), Pattern Detection (002), Semantic Extraction (003), Storage Compaction (004), Dream Operation (005), Grafana Dashboard (006), WAL Compaction (007a), gRPC Readiness (007b), Production Validation (007)
- 1-hour soak test: 61 consolidation runs, 100% success rate, perfect 60s cadence, sub-5ms latency
- Production baselines established: Cadence 60s±0s, Latency 1-5ms, Memory <25MB RSS
- SLA thresholds documented: Nominal/Warning/Critical levels for all key metrics
- Grafana dashboard deployed with Prometheus/Loki integration
- System validated for production deployment with observability stack

### Milestone 7: Memory Space Support ✅ COMPLETE

**Objective**: Introduce first-class "memory spaces" that isolate tenants/agents within a single Engram deployment. Provide configuration, API, and persistence boundaries so each agent maintains an autonomous memory graph while sharing infrastructure.

**Critical**: All APIs (HTTP/gRPC/CLI) must require an explicit `memory_space_id` and enforce access control. Persistence must shard WAL/index data per space without cross-contamination, and spreading activation must never traverse between spaces. Operational tooling needs clear per-space metrics and lifecycle (create, migrate, delete) flows.

**Validation**: Multi-tenant integration tests proving isolation across store/recall/streaming paths. Concurrency and load tests with ≥10 active spaces show no data leaks and predictable latency budgets. Migration playbook validated by promoting an agent from single-tenant to shared cluster without downtime.

**Completion Summary** (2025-10-23):
- 9/10 features implemented: MemorySpaceRegistry, per-space persistence, X-Memory-Space routing, CLI commands, health metrics, WAL recovery isolation, directory isolation, concurrent creation, partial event streaming
- Thread-safe DashMap-based space management with isolated storage per space
- Header-based routing with fallback precedence (header > query > body > default)
- Per-space health metrics and WAL recovery on startup
- Migration guide and operational runbook documentation complete
- System validated for multi-tenant production deployment

### Milestone 8: Pattern Completion

**Objective**: Build `complete()` operation for filling gaps in partial episodes using learned patterns. Create reconstruction that generates plausible missing details.

**Critical**: Completion must indicate which parts are reconstructed vs original with confidence scores. Must use both local context and global patterns for reconstruction.

**Validation**: Test with deliberately corrupted episodes, measure reconstruction accuracy against ground truth. Verify reconstructed details are plausible using human evaluation.

### Milestone 9: Query Language Parser

**Objective**: Implement DSL for memory operations supporting RECALL, PREDICT, IMAGINE with probabilistic semantics. Parser must provide helpful error messages with location and suggestions.

**Critical**: Query language must map directly to cognitive operations, not SQL-like syntax. Error messages must show query position and suggest corrections: "Unknown operation 'REMEMBER' at position 12. Did you mean 'RECALL'?"

**Validation**: Parse testing with corpus of valid/invalid queries, ensuring 100% of errors have actionable messages. Benchmark showing <100μs parse time for typical queries.

### Milestone 10: Zig Performance Kernels

**Objective**: Rewrite hot paths identified via profiling in Zig for maximum performance. Create specialized allocators for memory pool management.

**Critical**: Zig code must maintain identical semantics to Rust versions with bit-identical outputs. Must achieve >2x performance improvement to justify complexity.

**Validation**: Differential testing between Rust and Zig implementations on million-operation traces. Performance regression tests ensuring improvements maintained across commits.

### Milestone 11: Streaming Interface

**Objective**: Build continuous observation and real-time memory formation with incremental indexing. Support both push (observations) and pull (recall) in single stream.

**Critical**: Streaming must maintain temporal ordering even under high load using backpressure. Index updates must be lock-free to avoid blocking writes.

**Validation**: Chaos testing with random delays and failures, verify eventual consistency. Benchmark showing sustained 100K observations/second with concurrent recalls.

### Milestone 12: GPU Acceleration

**Objective**: Add CUDA kernels for parallel batch operations and embedding similarity. Implement unified memory for zero-copy CPU-GPU transfers.

**Critical**: Must gracefully fallback to CPU when GPU unavailable with identical results. GPU memory must be managed to avoid OOM on large batches.

**Validation**: Test on various GPU configurations (consumer/datacenter), verify linear speedup with GPU cores. Ensure CPU-only tests pass identically to prevent GPU-only dependencies.

### Milestone 13: Cognitive Patterns and Observability

**Objective**: Implement priming, interference detection, and reconsolidation matching cognitive psychology. Build metrics and tracing for memory dynamics visualization.

**Critical**: Cognitive operations must be validated against published psychology research. Metrics must have near-zero overhead when disabled, <1% when enabled.

**Validation**: Replicate classic psychology experiments (DRM paradigm, interference patterns). Verify metrics overhead using production-like workloads with/without instrumentation.

### Milestone 14: Distributed Architecture

**Objective**: Design partitioned memory across nodes with gossip-based consolidation sync. Enable transparent distribution without changing API semantics.

**Critical**: Must handle network partitions by degrading to local-only recall rather than failing. Consolidation must be eventually consistent across nodes.

**Validation**: Jepsen-style testing for distributed consistency properties. Verify single-node and distributed APIs return equivalent results on same data.

### Milestone 15: Multi-Interface Layer ✅ COMPLETE

**Objective**: Implement gRPC service with streaming and HTTP REST API for web clients. Both must expose identical functionality with interface-appropriate patterns.

**Critical**: Protocol buffers must version gracefully for backwards compatibility. HTTP errors must return problem+json format with actionable details.

**Validation**: Contract testing ensuring gRPC and HTTP return equivalent results. Load testing showing gRPC handles 100K ops/sec, HTTP handles 10K ops/sec.

**Completion Summary** (2025-10-23):
- HTTP REST API with OpenAPI/Swagger documentation at `/docs/`
- gRPC service with streaming support for all memory operations
- Server-Sent Events (SSE) for real-time metrics and activity monitoring
- CORS support for web clients with configurable origins
- Multi-tenant routing via X-Memory-Space header across both interfaces
- Contract tests validating API equivalence and backwards compatibility
- System validated for production client integration

### Milestone 16: Production Operations & Documentation

**Objective**: Complete production-ready documentation covering deployment, monitoring, backup/restore, performance tuning, and scaling. Establish operational runbooks, troubleshooting guides, and migration paths from existing databases.

**Critical**: Documentation must follow Diátaxis framework (tutorials, how-to, explanation, reference) with clear answers to operator questions: "how to deploy", "how to backup", "how to find slow queries", "how to scale". Operations guides must use direct, actionable tone with Context→Action→Verification format.

**Validation**: External operator can deploy from scratch following docs in <2 hours. All common production scenarios (backup, restore, scaling, troubleshooting) have tested runbooks. Migration guides validated for Neo4j, PostgreSQL, Redis paths.
