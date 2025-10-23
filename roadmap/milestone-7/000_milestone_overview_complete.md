# Milestone 7: Memory Space Support — Implementation Playbook
_Prepared: October 22, 2025 (United States)_

## Objective
Deliver truly isolated "memory spaces" so multiple agents can share an Engram deployment without ever touching each other’s episodic graph, persistence, or telemetry. This playbook expands the high-level goal into concrete engineering work that a junior developer can execute with confidence across `engram-core`, `engram-storage`, `engram-cli`, and `engram-proto`.

## Current System Snapshot (Oct 22, 2025)
- `engram-core/src/store.rs` instantiates a single `MemoryStore` with global `DashMap` state, a shared WAL buffer (`wal_buffer`), and broadcast channel lacking tenant context.
- Engine modules (`engram-core/src/activation/*`, `engram-core/src/memory_graph/*`, `engram-core/src/streaming_health.rs`) accept no tenant identifiers; spreading activation, consolidation, and SSE streaming all run against the singleton store.
- Persistence code in `engram-core/src/storage/{wal.rs,tiers.rs,recovery.rs}` and `engram-storage/src/lib.rs` writes beneath one root directory and shares compaction workers.
- Public surfaces (`engram-cli/src/api.rs`, `engram-cli/src/grpc.rs`, `engram-cli/src/cli/*`, `engram-cli/tests/*`) provide no hook for `memory_space_id`; all commands resolve to the single in-process store.
- Protocol definitions in `proto/engram/v1/{service.proto,memory.proto}` and generated glue in `engram-proto/src/lib.rs` lack multi-tenant metadata.
- Metrics (`engram-core/src/metrics/*`), health checks, and SSE streams aggregate totals without a `memory_space` label, so operators cannot disambiguate load.
- Docs (`docs/`, `content/`) describe memory spaces aspirationally without outlining registry APIs, data layout, or migration flows.

## Target Outcomes
1. Every store/recall/streaming operation carries a required `MemorySpaceId`; cross-space access is impossible at both type and runtime layers.
2. Persistence shards (WAL, tiered storage, recovery, compaction) write to `<data_root>/<memory_space_id>/…` with per-space workers and lifecycle management.
3. REST, CLI, and gRPC protocols accept & validate space identifiers with backwards-compatible defaults for single-tenant deployments.
4. Metrics, SSE, and status endpoints emit per-space signals without leaking other tenants.
5. Migration path automatically wraps existing single-space nodes into a named default space (`default`), with step-by-step docs.
6. Validation suite includes concurrent multi-space stress, crash-recovery, and spreading activation isolation tests.
7. Documentation and operator runbooks enable safe rollout and ongoing operations.

## Implementation Tracks

### 001 Memory Space Registry (Control Plane)
- **Goal**: Centralize creation, lookup, and lifecycle of `MemorySpaceId → MemorySpaceHandle` mappings while seeding persistence directories.
- **Key Files**: `engram-core/src/types.rs`, `engram-core/src/space/{mod.rs,registry.rs}` (new), `engram-core/src/lib.rs`, `engram-cli/src/config.rs`, `engram-cli/config/default.toml`.
- **Implementation Checklist**
  - Introduce `MemorySpaceId` newtype in `engram-core/src/types.rs` with `serde::{Serialize, Deserialize}`, `Display`, `FromStr`, and `TryFrom<&str>` implementations. Enforce lowercase `a-z0-9_-` with length limits.
  - Add new module `engram-core/src/space/mod.rs` exporting:
    - `MemorySpaceHandle` (wraps `Arc<MemoryStore>` plus metadata),
    - `MemorySpaceRegistry` managing `DashMap<MemorySpaceId, MemorySpaceHandle>` and delegating to persistence seeding hooks.
  - Implement `MemorySpaceRegistry::create_or_get` using lock-free fast path with `parking_lot::RwLock`-guarded slow path for new spaces. Ensure concurrent creation yields a single handle (double-checked pattern).
  - Generate directory roots inside `RegistryConfig { data_root: PathBuf }` passed from CLI configuration (`engram-cli/src/config.rs`). On create, call a persistence hook (added in Track 003) to allocate `<root>/<space>/` directories.
  - Expose list/delete methods with soft-delete states for future cleanup (set tombstone flag before asynchronous purge).
  - Update `engram-core/src/lib.rs` to `pub mod space;` and re-export `MemorySpaceId` for downstream crates.
  - Extend CLI configuration (`engram-cli/config/default.toml`) with `memory_spaces.data_root` and optional `default_space`. Parse into `CliConfig`.
- **Tests & Validation**
  - New async test `engram-core/tests/memory_space_registry.rs` verifying concurrent `create_or_get` calls coalesce and directories appear.
  - Property test using `proptest` to ensure invalid IDs are rejected and serialized form round-trips.

### 002 Engine Isolation (Core Graph)
- **Goal**: Require space context everywhere the engine touches memory, ensuring compile-time isolation.
- **Key Files**: `engram-core/src/store.rs`, `engram-core/src/memory.rs`, `engram-core/src/activation/{mod.rs,recall.rs}`, `engram-core/src/streaming_health.rs`, `engram-core/src/memory_graph/*`.
- **Implementation Checklist**
  - Refactor `MemoryStore` into space-scoped `MemorySpaceStore` held inside `MemorySpaceHandle`; move global statics (e.g., `wal_buffer`, event broadcasters) into the handle.
  - Introduce `struct SpaceContext { id: MemorySpaceId, metrics: Arc<MetricsRegistry>, events: Sender<SpaceEvent> }` to thread through activation and consolidation paths.
  - Update APIs to accept `(space_id: &MemorySpaceId, request: …)` or `SpaceHandle` instead of exposing the raw store globally.
  - Modify `MemoryStore::store`, `recall`, `consolidate`, and SSE publishers to include `space_id`. Make `store` return `Result<StoreOutcome, SpaceError>` where `SpaceError::WrongSpace` is impossible if type system enforced.
  - Update background workers (HNSW updates, tier migration, consolidation) to register under the originating `MemorySpaceId`, storing handles in `MemorySpaceHandle` for clean shutdown.
  - Ensure `MemoryEvent` enum embeds `memory_space_id` for downstream surfaces.
- **Tests & Validation**
  - Unit tests in `engram-core/src/store.rs` to assert operations panic/refuse when called without a space context (use `#[should_panic]` or error).
  - Integration test `engram-core/tests/multi_space_isolation.rs`: create spaces `alpha` and `beta`, store overlapping IDs, confirm cross-space recall returns empty results, SSE streams are segregated.

### 003 Persistence Partitioning & Recovery
- **Goal**: Wire per-space persistence (WAL, tier migration, compaction, recovery) with deterministic layout.
- **Key Files**: `engram-core/src/storage/{wal.rs,recovery.rs,tiers.rs,compact.rs}`, `engram-storage/src/lib.rs`, `engram-cli/src/config.rs`.
- **Implementation Checklist**
  - Define `SpaceStoragePaths` helper returning `{wal_dir, hot_dir, warm_dir, cold_dir}` under `<data_root>/<memory_space_id>/`.
  - Modify WAL writer (`engram-core/src/storage/wal.rs`) to accept `MemorySpaceId` and open log file within that space directory; maintain separate writer task per space stored inside `MemorySpaceHandle`.
  - Update recovery pipeline (`engram-core/src/storage/recovery.rs`) to iterate directories, reconstructing each space lazily when first requested by the registry.
  - Partition compaction and tier migration workers to operate on a single space at a time; expose `spawn_space_workers(&MemorySpaceId, &SpaceStoragePaths)`.
  - Update `engram-storage/src/lib.rs` (hot tier plus placeholders) to parameterize caches by `MemorySpaceId` where persistent state is shared, ensuring caches don't mix data.
  - Add CLI config options for `max_spaces`, `space_bootstrap_strategy` (auto-create default) and propagate to registry.
- **Tests & Validation**
  - Tempdir-based test in `engram-core/tests/persistence_space_layout.rs` verifying new spaces create directories and WAL files with names like `<space>/wal/current.log`.
  - Recovery test simulating crash: write episodes to two spaces, drop store, run recovery, ensure `alpha` entries never appear in `beta`.

### 004 REST/CLI Surface Updates
- **Goal**: Require `memory_space_id` across REST routes, CLI commands, and configuration UX.
- **Key Files**: `engram-cli/src/api.rs`, `engram-cli/src/lib.rs`, `engram-cli/src/cli/{commands.rs,memory.rs,server.rs}`, `engram-cli/src/config.rs`, `engram-cli/tests/{http_api_tests.rs,cli/memory_*}`, `engram-cli/examples/*`.
- **Implementation Checklist**
  - Introduce `axum` extractors capturing `Path<(MemorySpaceParam, ...)>` and apply to all `/api/*` routes. Reject requests without explicit space using 400 with remediation tips.
  - Add header support (`x-memory-space`) for clients; fallback to path parameter to avoid duplication.
  - `ApiState` should hold `Arc<MemorySpaceRegistry>` and retrieve handles per request. Remove global `Arc<MemoryStore>` usage.
  - Extend CLI `memory` subcommands with `--space` flag defaulting to config `default_space`; display helpful error when missing.
  - Update CLI server bootstrap to auto-create `default_space` on start via registry call.
  - Revise CLI docs/help text (`engram-cli/src/cli/memory.rs`) and examples.
- **Tests & Validation**
  - Extend `engram-cli/tests/http_api_tests.rs` to cover explicit space path (`/api/v1/spaces/alpha/memories`), missing space (expect 400), and cross-space rejection.
  - CLI integration test ensuring `engram-cli memory list --space alpha` only returns alpha items even when beta exists.

### 005 gRPC & Proto Evolution
- **Goal**: Embed `memory_space_id` into gRPC contracts with backwards-compatible defaults.
- **Key Files**: `proto/engram/v1/service.proto`, `proto/engram/v1/memory.proto`, `engram-proto/src/lib.rs`, `engram-cli/src/grpc.rs`, `engram-cli/tests/grpc_tests.rs`.
- **Implementation Checklist**
  - Add `string memory_space_id = 1;` to relevant request messages (`RememberRequest`, `RecallRequest`, streaming requests, etc.), shifting existing field numbers while maintaining reserved slots for compatibility.
  - Introduce optional metadata header (`engrams.v1.Memory-Space`) for legacy clients; implement interceptor in `engram-cli/src/grpc.rs` that maps metadata to registry lookup.
  - Regenerate Rust types (`cargo xtask proto` or `cargo build -p engram-proto`) and plumb new fields through `MemoryService` methods, returning `Status::invalid_argument` when missing.
  - Update `engram-proto` builders/helpers to provide ergonomic constructors accepting `MemorySpaceId`.
  - Document version bump in `engram-proto/Cargo.toml` with semver minor increment.
- **Tests & Validation**
  - gRPC test ensuring requests without space fail once feature flag enabled; legacy path validated via metadata.
  - Backward compatibility test using pre-upgrade fixtures (if available) or synthetic old client simulation.

### 006 Observability & Metrics
- **Goal**: Expose per-space telemetry so operators can spot hot tenants without leakage.
- **Key Files**: `engram-core/src/metrics/{mod.rs,streaming.rs}`, `engram-core/src/streaming_health.rs`, `engram-cli/src/status.rs`, `engram-cli/tests/monitoring_tests.rs`.
- **Implementation Checklist**
  - Extend metric registration helpers to accept `memory_space: &MemorySpaceId` label; update counters/histograms to emit via `metrics.increment_counter_with_labels` (add helper if needed).
  - Replace global SSE topics with `spaces/{id}/events`; update clients to subscribe per space.
  - Update health/status endpoints to show partitioned summaries (pressure, WAL lag, consolidation backlog) keyed by space.
  - Guard cardinality: enforce configurable allowlist (`max_metric_spaces`) to prevent explosion; drop/aggregate beyond threshold with warning log.
  - Modify `engram-cli/src/status.rs` to display per-space sections and highlight spaces exceeding thresholds.
- **Tests & Validation**
  - Unit test for `StreamingAggregator` ensuring events carry `memory_space_id` and do not cross streams.
  - CLI monitoring test verifying JSON output contains `spaces: [{id: "alpha", ...}]` and excludes data from other spaces when filtered.

### 007 Validation & Fuzzing
- **Goal**: Prove isolation through automated testing and benchmarking.
- **Key Files**: `engram-core/tests/*`, `engram-core/proptest-regressions/`, `engram-core/fuzz/`, `engram-cli/tests/integration_tests.rs`, `.github/workflows/*` (if CI updates needed).
- **Implementation Checklist**
  - Add concurrent stress test: spawn tasks storing/recalling in 10 spaces with overlapping IDs; assert results remain scoped.
  - Extend persistence recovery test to simulate crash mid-write (`MemorySpaceRegistry` drop) and ensure WAL replay uses correct space directories.
  - Update fuzz harnesses to include `MemorySpaceId` randomization; refresh regression seeds in `engram-core/proptest-regressions` once deterministic behavior confirmed.
  - Benchmark script (`./benchmark-startup.sh`) should record cold-start latency with 5 pre-seeded spaces; capture metrics HTML.
- **Tests & Validation**
  - Hook new tests into `cargo test --workspace`; add `cargo test -p engram-core -- --ignored multi_space_stress` for heavy cases.
  - Document how to run fuzzers and interpret results in `docs/testing/memory_spaces.md`.

### 008 Documentation & Migration Guidance
- **Goal**: Equip operators and developers with clear upgrade instructions and conceptual docs.
- **Key Files**: `docs/operations/memory_spaces.md` (new), `docs/migrations/milestone7_memory_spaces.md`, `README.md`, `content/milestone_7/*` (link back), `engram-cli/docs.rs` sections.
- **Implementation Checklist**
  - Author migration guide covering: enabling registry, auto-creating default space, updating clients, rolling out gRPC changes, cleaning legacy data.
  - Update README multi-tenant section with CLI and API examples showing `memory_space_id` usage.
  - Add troubleshooting playbook (common errors: missing header, directory permissions, exceeding space limits).
  - Cross-link new docs from `milestones.md` and `vision.md` where memory spaces are referenced.
  - Ensure OpenAPI (`engram-cli/src/openapi.rs`) documents the new path/headers and regenerates Utoipa docs.
- **Tests & Validation**
  - Run `cargo doc --workspace --no-deps` to confirm doctests referencing memory spaces compile.
  - Validate doc snippets via `docs/tests` (if present) or add script to CI.

## Migration & Compatibility Strategy
- Autocreate `MemorySpaceId::DEFAULT` on server startup; legacy clients without explicit ID map to this space while emitting deprecation warnings (log + metric `engram_memory_space_default_fallback_total`).
- Provide configuration flag `require_explicit_space` defaulting to `true` in new installs but toggled during migration windows.
- CLI/gRPC respond with actionable errors: "Missing memory_space_id. Set --space or X-Memory-Space header."
- Include step-by-step data migration: rename existing persistence tree to `<root>/default/`, verify structure, run `engram-cli memory migrate --from-default --to=<new>` when splitting tenants.

## Validation Workflow
1. Unit + integration tests per track (`cargo test --workspace`).
2. `cargo clippy --workspace --all-targets --all-features -D warnings` ensures lint cleanliness.
3. `cargo fmt --all` prior to review.
4. Run `./benchmark-startup.sh` capturing cold-start metrics; attach HTML in milestone report.
5. Execute `make quality` (fmt + clippy + docs) before marking tasks complete.

## Definition of Done Checklist
- [ ] Registry module merged with concurrency + persistence hooks.
- [ ] Engine APIs refuse operations without `MemorySpaceId`.
- [ ] Persistence writes/readers scoped per space with tests.
- [ ] REST/CLI/gRPC surfaces require `memory_space_id` and docs updated.
- [ ] Metrics/SSE/health outputs per-space data.
- [ ] Multi-space stress, recovery, and fuzzing suites green with updated seeds.
- [ ] Migration guide published and referenced from `milestones.md`.
- [ ] `milestones.md` + roadmap tasks updated to reflect implementation status.

Follow the CLAUDE workflow: once these items are satisfied, rename this file to `_complete`, capture relevant test output, and stage only intentional changes before committing (`feat: add milestone 7 memory space playbook`).
