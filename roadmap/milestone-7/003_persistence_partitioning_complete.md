# 003: Persistence Partitioning & Recovery Isolation — _90_percent_complete_

## Current Status: 90% Complete

**What's Implemented**:
- ✅ Per-space directory layout: `<data_root>/<space_id>/{wal,hot,warm,cold}` (engram-core/src/registry/memory_space.rs:376)
- ✅ `SpaceDirectories` struct with sanitized paths
- ✅ `MemorySpacePersistence` struct managing WAL writer, tier backend, metrics per space (engram-core/src/storage/persistence.rs, 120+ lines)
- ✅ Registry integration with `persistence_handles` DashMap
- ✅ `recover_all()` method scanning directories and recovering per-space WAL (lines 227-299)
- ✅ Per-space WAL writers, tier backends, and migration workers
- ✅ Space ID sanitization preventing path traversal attacks
- ✅ Concurrent-safe handle creation via registry

**Missing** (10%):
- ❌ Integration test `multi_space_persistence.rs` validating cross-space isolation during writes/recovery
- ❌ Crash recovery tests simulating abrupt shutdown and validating per-space replay
- ❌ Documentation in code comments explaining directory structure and recovery semantics
- ❌ Metrics validation confirming per-space storage metrics are tracked

## Goal
Rework persistence so WAL, tiered storage, and recovery operate on per-space directories with independent workers, eliminating cross-space contamination. The implementation must be explicit enough that any new `MemorySpaceId` automatically provisions its own persistence stack and never touches another tenant's data.

## Deliverables
- Deterministic directory layout: `<data_root>/<sanitized_memory_space_id>/{wal,hot,warm,cold}` implemented via a dedicated helper.
- Per-space persistence handle that owns WAL writer, tier migration worker, storage metrics, and shutdown wiring.
- Registry hooks (`engram-core/src/registry.rs`, new or existing) that lazily construct and cache handles, expose `get_or_create_persistence(space_id)` and `recover_all(data_root)`.
- `MemoryStore` persistence paths updated to accept a `MemorySpaceId` or a handle instead of using global fields; in-memory WAL buffer remains per store instance only.
- CLI bootstrap (`engram-cli/src/main.rs`) updated to initialize persistence for the default space via the registry and pass the handle into `MemoryStore`.
- Configuration knobs for per-space capacity limits (hot/warm/cold) surfaced in `engram-cli/config/default.toml` and deserialized in `engram-cli/src/config.rs`.

## Implementation Plan

1. **Define Persistence Layout Helpers**
   - Add `MemorySpacePath` helper in `engram-core/src/storage/mod.rs` (or create `storage::space.rs`) with:
     - `sanitize_space_id(&MemorySpaceId) -> String` that rejects `..`, path separators, and uppercases if needed.
     - `paths_for(space_id: &MemorySpaceId, root: &Path) -> MemorySpacePaths` returning `wal_dir`, `hot_dir`, `warm_dir`, `cold_dir`.
     - Unit tests in `engram-core/src/storage/tests/space_paths.rs` to ensure sanitization and layout.

2. **Per-Space Persistence Handle**
   - Introduce `MemorySpacePersistence` struct in `engram-core/src/store.rs` (or a new module) containing:
     ```rust
     pub struct MemorySpacePersistence {
         pub wal_writer: Arc<WalWriter>,
         pub storage_metrics: Arc<StorageMetrics>,
         pub tier_backend: Arc<CognitiveTierArchitecture>,
         pub tier_worker: Mutex<Option<std::thread::JoinHandle<()>>>,
         pub tier_shutdown: Arc<AtomicBool>,
     }
     ```
   - Provide `fn new(space_id, config, paths) -> Result<Self>` to encapsulate `WalWriter::new(...)` and `CognitiveTierArchitecture::new(...)`.
   - Implement `fn start_workers(&self)` and `fn shutdown(&self)` to manage threads.

3. **Registry Integration**
   - Extend the registry introduced in Task 001 (`engram-core/src/registry.rs`) with:
     - `HashMap<MemorySpaceId, Arc<MemorySpacePersistence>>` guarded by `RwLock`.
     - `fn persistence_handle(&self, space_id, config) -> Result<Arc<MemorySpacePersistence>>` that lazily constructs handles using the helper paths.
     - `fn recover_all(&self, data_root) -> Result<Vec<RecoveryReport>>` scanning directories and invoking `MemorySpacePersistence::recover()` (see next step).
   - Ensure creation is idempotent and concurrent-safe (use `dashmap::DashMap` or `once_cell` pattern).

4. **MemoryStore Changes**
   - Replace current global persistence fields (`wal_writer`, `persistent_backend`, `tier_migration_worker`, `tier_migration_shutdown`, etc.) with references obtained from `MemorySpacePersistence`.
   - Update `MemoryStore::with_persistence` to accept `Arc<MemorySpacePersistence>` instead of `&Path` and remove directory creation logic from the store.
   - Modify persistence helper methods (`persist_episode`, `initialize_persistence`, `recover_from_wal`) to call through the handle.
   - Ensure compile gate features (`cfg(feature = "memory_mapped_persistence")`) continue to work by keeping definitions behind the same flags.

5. **Recovery Workflow**
   - Implement `MemorySpacePersistence::recover(&self, store: &MemoryStore) -> Result<usize>` that reads from the per-space WAL directory.
   - Update CLI startup (`engram-cli/src/main.rs`) to enumerate existing spaces via the registry and call recovery for each before serving traffic.
   - Surface recovery results in logs (e.g., `info!(space = %space_id, recovered)`).

6. **Tier Worker Partitioning**
   - Ensure tier migration and compaction threads launched in `MemoryStore::initialize_persistence` are now per space by storing the join handle inside `MemorySpacePersistence`.
   - When shutting down a space (future work), call `MemorySpacePersistence::shutdown()` to stop threads and flush WAL.

7. **Configuration Updates**
   - Extend `engram-cli/src/config.rs` with new settings:
     ```toml
     [persistence]
     data_root = "~/.local/share/engram"
     hot_capacity = 100000
     warm_capacity = 1000000
     cold_capacity = 10000000
     ```
   - Provide defaults via `default.toml` and plumb them into registry initialization.

8. **Diagnostics & Metrics Hooks**
   - Ensure `StorageMetrics::record_*` calls include `memory_space_id` labels. Update constructors to accept the ID and push labels into metrics registry (ties into Task 006 but this task should wire the data).

## Integration Points
- `engram-core/src/store.rs` (persistence fields, `with_persistence`, `initialize_persistence`, `recover_from_wal`).
- `engram-core/src/storage/wal.rs` (new constructor arguments for per-space paths, sanitized naming).
- `engram-core/src/storage/tiers.rs` (per-space capacities, initialization via helper paths).
- `engram-core/src/registry.rs` (new persistence map and lifecycle APIs).
- `engram-cli/src/main.rs` and `engram-cli/src/config.rs` (startup wiring and configuration).
- Tests under `engram-core/tests/` for multi-space persistence scenarios.

## Acceptance Criteria

1. ✅ **COMPLETE**: Storing memories in two spaces writes to separate WAL files and tier directories
   - Implementation: `SpaceDirectories::for_space()` creates isolated directory trees
   - Validation: Manual testing confirms separate wal/hot/warm/cold directories
   - **MISSING**: Automated integration test asserting directory isolation

2. ✅ **COMPLETE**: WAL recovery rehydrates only the targeted space
   - Implementation: `recover_all()` method in registry (lines 227-299)
   - Logs: Per-space recovery counts emitted during startup
   - **MISSING**: Integration test verifying isolation during recovery

3. ✅ **COMPLETE**: Compaction and tier migration workers operate on per-space queues
   - Implementation: Each `MemorySpacePersistence` owns tier_worker thread
   - Per-space: `CognitiveTierArchitecture` instance in persistence handle
   - **MISSING**: Instrumentation validation showing space ID in thread logs

4. ⚠️ **DEFERRED**: Space deletion safe cleanup routine
   - Status: Not implemented (future feature)
   - TODO: Document cleanup requirements in operations guide

5. ✅ **COMPLETE**: Configuration defaults documented and validated
   - Implementation: Default paths in `SpaceDirectories::for_space()`
   - Fallback: Registry uses sane defaults when config missing

## Remaining Work

1. **Integration Test: Multi-Space Persistence Isolation** (3-4 hours)
   - File: Create `engram-core/tests/multi_space_persistence.rs`
   - Scenario:
     ```rust
     // 1. Create two spaces (alpha, beta)
     // 2. Write episodes to each space
     // 3. Stop server
     // 4. Inspect filesystem: assert ${tempdir}/alpha/wal/ != ${tempdir}/beta/wal/
     // 5. Restart server with recovery
     // 6. Verify alpha episodes != beta episodes
     ```
   - Validation: Directory isolation + recovery isolation

2. **Crash Recovery Test** (2-3 hours)
   - File: Add to `engram-core/tests/multi_space_persistence.rs`
   - Scenario:
     ```rust
     // 1. Start server, write to space alpha
     // 2. Kill server mid-transaction (drop handle without flush)
     // 3. Restart, run recovery
     // 4. Assert recovered count matches expected
     // 5. Verify space beta unaffected
     ```
   - Validation: Per-space WAL replay correctness

3. **Documentation & Metrics** (1-2 hours)
   - Files:
     - `engram-core/src/storage/persistence.rs` (add module doc)
     - `engram-core/src/registry/memory_space.rs` (document SpaceDirectories)
   - Content:
     - Explain directory layout and sanitization rules
     - Document recovery semantics and ordering guarantees
     - Show example directory tree structure
   - Metrics: Validate `StorageMetrics` includes space_id labels

## Testing Strategy
- **Integration**: Add `tests/multi_space_persistence.rs` that boots two spaces, performs writes, shuts down, inspects `${tempdir}/alpha/` vs `${tempdir}/beta/`, and runs recovery.
- **Crash Simulation**: Write a test that truncates the process after issuing writes (use temp runtime) and reruns recovery to confirm isolated replay counts.
- **Unit**: Path sanitization tests guard against traversal and invalid IDs.
- **Bench hooks**: Reuse Criterion bench (opt-in) to compare single-space baseline vs dual-space overhead; document results.
- Update CI scripts to include new integration test (ensure runtime stays <60s).

## Review Agent
- `systems-architecture-optimizer` to ensure IO scheduling and concurrency align with storage roadmap.
