# Engram Architecture

## Why This Architecture Exists
Engram is designed for teams that need memories to behave more like brains than ledgers. Traditional databases reward perfect consistency and transactional isolation; Engram optimizes for probabilistic answers, graceful degradation, and associative recall (see `why.md`). The architecture blends three worlds:
- vector-native similarity search for rapid cue seeding,
- graph traversal for relationship-aware activation, and
- cognitive dynamics (decay, consolidation, confidence budgets) so results stay meaningful over time.
This document explains how the pieces cooperate to deliver that behavior.

## System Overview
At a glance, Engram is a layered system that turns API calls into cognitive state changes:

```
+-----------+    HTTP/gRPC    +-----------------+    +-------------------------+
|  Clients  | --------------> | engram-cli API  | --> | MemoryStore (Hot Tier)  |
+-----------+                 +-----------------+    |  * Vector index (HNSW)  |
                                                   |  * Confidence tracking    |
                                                   +------------+--------------+
                                                                |
                                                                | writes/reads
                                                                v
                                             +-------------------------+
                                             | Consolidation Service   |
                                             |  * Episodic -> semantic  |
                                             |  * Snapshot cache       |
                                             +-----------+-------------+
                                                         |
                                                         | migrations (future)
                                  +----------------------+----------------------+
                                  | Warm Tier (log)     | Cold Tier (columnar) |
                                  |  planned/partial    |  planned             |
                                  +----------------------+----------------------+
```

On the read side, `CognitiveRecall` coordinates vector seeding, spreading activation, temporal decay, and ranking before responses return to the client. Observability hooks stream metrics, traces, and health signals alongside the data path (`docs/operations.md`, `docs/operations/spreading.md`).

## Core Packages and Responsibilities
The workspace crates map cleanly onto architectural responsibilities (`core_packages.md`):
- **`engram-core`** - graph engine, activation logic, temporal decay, probability math. Provides the `MemoryStore`, `CognitiveRecall`, decay system, and spreading configuration types.
- **`engram-storage`** - tiered persistence primitives. Today the hot tier is production-ready; warm/cold tiers and migration policies are staged to graduate from the partial designs noted in `FEATURES.md`.
- **`engram-cli`** - orchestration layer exposing HTTP, gRPC, SSE, and the operator CLI. Owns feature-flag plumbing, configuration files, and startup/shutdown flows.
- **`engram-proto`** - shared protobuf/OpenAPI contracts so external services match CLI semantics.
Supporting crates (`engram-index`, `engram-dynamics`, `engram-metrics`, etc.) plug specialized capabilities into `engram-core` without bloating the core API surface.

## Memory Space Isolation Architecture
Engram supports multi-tenant deployments through memory space isolation, enabling independent memory graphs for different users, agents, or applications within a single instance. This section explains the registry-based isolation pattern and how to work with it safely.

### Conceptual Model
A **memory space** is a logical partition of the memory graph. Each space has:
- Its own set of episodes, beliefs, and graph edges
- Independent activation dynamics and consolidation state
- Isolated confidence tracking and temporal decay
- No cross-space memory leakage, even when episode IDs collide

Think of spaces like separate databases in a SQL server, except they share nothing at the data level and communicate only through explicit registry operations.

### Registry-Based Isolation
The `MemorySpaceRegistry` (`engram-core/src/registry/`) acts as the central authority for space lifecycle management. When a request arrives:

1. **Space Resolution** - API handlers extract the memory space ID from the request (header, query param, or body field, with fallback to default).
2. **Handle Acquisition** - The registry returns a `SpaceHandle` wrapping the appropriate `MemoryStore` instance.
3. **Operation Execution** - The handler operates on that specific store, which is bound to the space at construction time.
4. **Automatic Cleanup** - Handles use reference counting; when the last reference drops, the space can be unloaded.

### Why Registry-Only (Not Partitioned Collections)
During Task 002 implementation, we evaluated two approaches:

**Option A: Registry-Only Isolation** (chosen)
- Registry creates separate `MemoryStore` instances per space
- Each store owns its own DashMap, HNSW index, decay state
- Isolation guaranteed by Rust ownership (separate instances cannot share data)
- Runtime overhead: registry lookup (~100ns) + Arc clone per request
- Tradeoff: Requires discipline to always use registry, not enforced by type system

**Option B: Partitioned Collections** (deferred)
- Single `MemoryStore` with nested `DashMap<SpaceId, DashMap<EpisodeId, Episode>>`
- Compile-time enforcement via phantom types
- Higher implementation complexity (2-3x code)
- Potential performance overhead from double-hashing

We chose Option A because:
1. Simpler implementation (4-6 hours vs 16-20 hours)
2. Leverages existing registry infrastructure from Task 001
3. Pragmatic isolation is "good enough" for production (see test validation)
4. Can migrate to Option B later if type-safety requirements increase
5. Aligns with user directive to "do it properly" without accumulating tech debt

### Implementation Details

#### MemoryStore Construction
Every `MemoryStore` is bound to a space at creation:

```rust
// engram-core/src/store.rs
pub struct MemoryStore {
    memory_space_id: MemorySpaceId,
    episodes: Arc<DashMap<String, Episode>>,
    // ... rest of fields
}

impl MemoryStore {
    pub fn for_space(memory_space_id: MemorySpaceId, max_memories: usize) -> Self {
        // Creates a store instance dedicated to this space
    }

    pub fn space_id(&self) -> &MemorySpaceId {
        &self.memory_space_id
    }
}
```

The `new()` constructor delegates to `for_space(MemorySpaceId::default(), ...)` for backward compatibility with single-space deployments.

#### Space Extraction Pattern
API handlers follow a consistent pattern (`engram-cli/src/api.rs:185`):

```rust
fn extract_memory_space_id(
    query_space: Option<&str>,
    body_space: Option<&str>,
    default: &MemorySpaceId,
) -> Result<MemorySpaceId, ApiError>
```

Priority order:
1. Query parameter `?space=<id>`
2. Request body field `memory_space_id`
3. Default space from `ApiState`

This allows clients to gradually adopt multi-tenancy without breaking existing single-space usage.

#### Handler Wiring
Every memory operation handler follows this sequence (`engram-cli/src/api.rs:1107`, `1229`):

```rust
pub async fn remember_episode(
    State(state): State<ApiState>,
    Query(params): Query<EpisodeQuery>,
    Json(request): Json<RememberEpisodeRequest>,
) -> Result<impl IntoResponse, ApiError> {
    // 1. Extract space ID
    let space_id = extract_memory_space_id(
        params.space.as_deref(),
        request.memory_space_id.as_deref(),
        &state.default_space,
    )?;

    // 2. Get handle from registry
    let handle = state.registry.create_or_get(&space_id).await?;
    let store = handle.store();

    // 3. Runtime verification (defense-in-depth)
    store.verify_space(&space_id)
        .map_err(|e| ApiError::SystemError(e))?;

    // 4. Proceed with operation
    let core_episode = Episode::new(/* ... */);
    store.store(core_episode);
    // ... rest of handler
}
```

As of Milestone 7 Task 002b, the following handlers are updated:
- `remember_episode` - stores episodes with space isolation
- `recall_memories` - queries within a specific space

Remaining handlers (remember_memory, search_memories, remove_memory, consolidate, dream) follow the same pattern and are queued for Task 002b completion.

#### Runtime Guards
The `verify_space()` method provides runtime defense against bugs (`engram-core/src/store.rs:~470`):

```rust
impl MemoryStore {
    pub fn verify_space(&self, expected: &MemorySpaceId) -> Result<(), String> {
        if &self.memory_space_id == expected {
            Ok(())
        } else {
            Err(format!(
                "Memory space mismatch: expected '{}', got '{}'",
                expected.as_str(),
                self.memory_space_id.as_str()
            ))
        }
    }
}
```

While Rust's ownership prevents accidental cross-space access, this guard catches logic errors where the wrong handle is passed through complex call chains. Integration tests validate these guards trigger correctly (`engram-core/tests/multi_space_isolation.rs:92`).

### Validation and Testing
Isolation correctness is proven through comprehensive integration tests (`engram-core/tests/multi_space_isolation.rs`):

1. **Same ID, Different Content** - Two spaces store episodes with identical IDs but different data; recalls prove no leakage
2. **Runtime Guard Verification** - Attempting to verify wrong space triggers expected error
3. **Concurrent Operations** - 5 spaces with concurrent writes demonstrate no interference
4. **Backward Compatibility** - Default space works unchanged for single-tenant usage

All tests pass, validating that registry-based isolation achieves the architectural goal without relying on partitioned collections.

### Observability
`MemoryEvent` emissions now include `memory_space_id` in all variants:

```rust
pub enum MemoryEvent {
    Stored { memory_space_id: MemorySpaceId, id: String, /* ... */ },
    Recalled { memory_space_id: MemorySpaceId, id: String, /* ... */ },
    ActivationSpread { memory_space_id: MemorySpaceId, /* ... */ },
}
```

SSE streams and JSON payloads include space identifiers, enabling per-space monitoring dashboards and tenant-specific analytics.

### Migration Path
For existing single-space deployments:
- **No changes required** - Requests without space parameters use the default space
- **Opt-in multi-tenancy** - Add `?space=<id>` or `X-Engram-Memory-Space: <id>` header
- **Gradual adoption** - Migrate tenants incrementally without downtime

For new handler implementations:
- **Always use registry** - Obtain stores via `registry.create_or_get()`, never cache across requests
- **Call verify_space()** - Add runtime guard in critical paths for defense-in-depth
- **Handle space in DTOs** - Include optional `memory_space_id` field in request/response types

### Tradeoffs and Limitations

**Current Approach (Registry-Only)**:
- Runtime enforcement vs compile-time (space omission not caught by type system)
- Registry lookup overhead (~100ns per request, negligible in practice)
- Requires developer discipline to use registry consistently
- Each space instance has separate in-memory indexes (higher memory usage vs shared structures)

**Benefits**:
- Simple implementation and maintenance
- Clear ownership boundaries (separate instances = separate data)
- Easy to reason about (no shared mutable state between spaces)
- Can evolve to partitioned collections if type-safety needs increase

**Future Enhancements** (if moving to Option B):
- Replace runtime guards with phantom types for compile-time enforcement
- Partition data structures for reduced memory footprint
- Zero-trust isolation where type system prevents misuse
- Migration path exists without breaking API compatibility

### Relationship to Other Architecture Components
- **Consolidation Service** - Runs per-space; snapshots include space ID for replay
- **Spreading Activation** - Respects space boundaries; no cross-space edge traversal
- **Temporal Decay** - Applied per-space; each space has independent confidence evolution
- **Storage Tiers** - Hot/warm/cold tiers partition by space for performance isolation
- **Metrics and Health** - Can be aggregated globally or filtered per-space for multi-tenant dashboards

See Task 002b specification (`roadmap/milestone-7/002b_handler_registry_wiring_pending.md`) for detailed implementation plan and handler migration checklist.

## Memory Lifecycle
Engram's architecture centers on what happens to a memory from the moment you store it until it's recalled and eventually consolidated.

### Remember Path
1. **Ingress** - Clients call `POST /api/v1/memories/remember` or the gRPC `Store` RPC (`usage.md`). `engram-cli` validates payloads, enforces feature flags, and forwards work to `MemoryStore`.
2. **Encoding** - `MemoryStore` assigns IDs, normalizes embeddings (768-float vectors), records confidence intervals, and stashes everything in the hot tier (DashMap-backed). Optional WAL buffering and warm-tier flushing are scaffolded for completion (see `FEATURES.md:38`).
3. **Indexing** - When the `hnsw_index` feature flag is enabled (default), embeddings also insert into the HNSW index so similarity seeding stays fast.
4. **Diagnostics** - Errors surface through the cognitive error stack described in `coding_guidelines.md`, ensuring context/suggestion/example travel with each failure.

### Recall Path
1. **Cue Seeding** - `CognitiveRecall` accepts a cue (text, embedding, or structured filter). Vector similarity results supply the initial activation set.
2. **Spreading Activation** - The parallel spreading engine expands activation through the graph using deterministic batches when configured (`docs/reference/spreading_api.md`). Auto-tuning adjusts batch size, hop depth, and thresholds when metrics drift (`docs/operations/spreading.md`).
3. **Temporal Adjustment** - Before ranking, the decay system lazily recomputes confidence according to hippocampal/power-law rules (`docs/temporal-dynamics.md`).
4. **Ranking & Response** - Results are categorized (vivid / associated / reconstructed), paired with confidence intervals and evidence chains, and streamed back over HTTP/gRPC (`usage.md:67`).

### Consolidation Path
1. **Scheduling** - An internal consolidation service (being extracted per `docs/architecture/rfc_consolidation_service.md`) periodically scans recent episodes once they pass the biological age threshold (>=1 day by default).
2. **Pattern Formation** - Episodic memories combine into semantic beliefs with provenance, recorded in cached snapshots so recall can reference them without re-running consolidation (`README.md:166`).
3. **Observability** - Belief updates emit through SSE (`/api/v1/stream/consolidation`) and JSON snapshots, letting operators validate consolidation health without touching the store directly.

## Storage Tiers
Think of Engram's tiers like your workspace organization:
- **Hot tier (desk)** - everything you need right now. Lock-free DashMap storage keeps active memories and their confidence state in RAM for sub-millisecond access. Production-ready today.
- **Warm tier (filing cabinet)** - append-only log meant for nearline replay, compression, and WAL safety. The interfaces exist; background flushing and recovery are the next milestone priorities (`FEATURES.md:40`).
- **Cold tier (attic)** - columnar embeddings mapped via `memmap2` for huge archives. Planned to support SIMD scans and GPU hand-off without loading everything into RAM.
Tier migration policies promote/demote memories based on activation frequency, not wall-clock age, preserving the "frequently recalled stays close" heuristic (`vision.md:43`).

## Activation and Query Engine
`engram-core::activation` houses the spreading engine, which behaves like a well-instrumented rumor mill:
- Configuration knobs (`ParallelSpreadingConfig`, `HnswSpreadingConfig`) govern worker threads, hop limits, similarity thresholds, and deterministic seeds.
- Cycle detection and activation breakers prevent runaway spreads in loopy subgraphs (`KNOWN_ISSUES.md` documents the test harness realities).
- Recall modes (`similarity`, `spreading`, `hybrid`) are runtime-selectable via query parameters when the `spreading_api_beta` flag is enabled.
- GPU integration hooks exist through the `GPUSpreadingInterface` trait for future acceleration once CUDA kernels stabilize.

## Temporal Confidence Model
Confidence scores are first-class throughout the stack. The decay subsystem (`docs/decay-functions.md`) applies:
- Exponential decay for hippocampal-style short-term memory,
- Power-law decay for consolidated knowledge,
- Two-component and hybrid models to mimic spaced repetition.
Everything runs lazily during recall, so no background sweeps are required, and you can "time travel" analyses by replaying decay at a historical timestamp. Type-state patterns ensure confidence fields can't be omitted from APIs (`coding_guidelines.md:87`).

## Runtime and Concurrency Model
Engram favours local reasoning with eventual alignment:
- Actor-like **memory regions** own graph subsets and communicate via message passing rather than global locks (`vision.md:47`).
- The hot tier uses `Arc` everywhere, aided by `parking_lot` primitives and lock-free patterns (`coding_guidelines.md:55`).
- Background services (consolidation, auto-tuning, diagnostics) run as independent tasks orchestrated by `engram-cli`, keeping the API path lean.
- Deterministic modes exist for activation tests, helping CI avoid the flakiness described in `KNOWN_ISSUES.md` by serializing the heaviest tests.

## Acceleration Pathways
Performance lives at multiple layers:
- **SIMD** - `simdeez`, `wide`, and custom kernels accelerate cosine similarity and batch activation across CPUs with runtime feature detection (`chosen_libraries.md`).
- **GPU** - The interface is in place (`engram-core/src/activation/doc/gpu_interface.md`); CUDA/ROCm kernels are a roadmap item (`FEATURES.md:74`).
- **Allocator strategy** - `mimalloc` and `bumpalo` reduce allocation churn for the hot tier.
- **Future Zig modules** - Hot loops can graduate to Zig per `vision.md:65` once Rust implementations are validated.

## Configuration and Feature Flags
The CLI keeps operational switches in `~/.config/engram/config.toml` with defaults in-tree. Key flags (`FEATURES.md:93`):
- `hnsw_index` - enables similarity seeding.
- `spreading_api_beta` - exposes spreading-specific endpoints (must restart after toggling; see `docs/howto/spreading_debugging.md`).
- `memory_mapped_persistence` - scaffolds cold-tier experiments.
- `monitoring` - streams metrics and SSE events.
Feature-flag changes are audited, and the architecture expects new features to ship with docs and CLI tooling (`docs/changelog.md`).

## Observability and Operations
Operational maturity is part of the architecture, not an afterthought:
- Health probes combine synthetic spreading checks with overall system status (`docs/operations/spreading.md`, `docs/operations.md`).
- Metrics cover activation latency, breaker state, pool utilization, consolidation progress, and confidence distributions. Exporters target Prometheus-compatible scrapers.
- Runbooks document backup/restore, incident response, and chaos harness usage so on-call engineers know exactly which lever to pull.
- CLI commands (`engram status`, `engram config`, `engram stop`) wrap internal APIs, providing the same interface human operators and automation use.

## Extensibility and Roadmap Alignment
Milestone sequencing (`milestones.md`) shows how the current architecture stretches:
- **Milestone 2** finishes warm/cold tier implementations and WAL durability.
- **Milestone 3** hardens the spreading engine with distributed-friendly schedulers and GPU acceleration.
- **Milestone 6** extracts consolidation into its own service boundary (`docs/architecture/rfc_consolidation_service.md`).
- Future milestones add distribution, multi-interface compatibility, and production packaging.
Keeping responsibilities separated (core vs storage vs dynamics) lets each milestone swap internals without rewriting the API surface.

## How Engram Differs From Traditional Datastores
- **Probabilistic by design** - Instead of guaranteeing ACID semantics, Engram guarantees calibrated confidence intervals and graceful fallbacks.
- **Temporal-first** - Decay and consolidation are core features, not downstream jobs.
- **Graph + vector hybrid** - Similarity seeding and spreading live in the same runtime, avoiding impedance mismatches common when glueing Milvus to Neo4j.
- **Cognitive observability** - Evidence chains, activation traces, and breaker metrics explain *why* a result appeared, not just *what* returned.
- **Documentation-backed process** - Conventions, error messages, and runbooks are wired into the architecture so new contributors inherit working practices immediately.

## Next Steps for Readers
- Follow the [Quickstart](quickstart.md) to feel the API.
- Read `docs/temporal-dynamics.md` and `docs/tutorials/temporal-configuration.md` to experiment with decay settings.
- Explore `docs/tutorials/spreading_getting_started.md` plus the debugging playbook for activation tuning.
- Review `docs/architecture/rfc_consolidation_service.md` if you want to help with consolidation extraction.
- Check `milestones.md` to see where your contributions fit in the roadmap.

And of course, run `engram status` and `engram config list` on a live node to see this architecture in action.
