# Designing Engram’s Memory Space Registry

Engram’s single-tenant assumption leaks all the way from CLI startup (`engram-cli/src/main.rs:174`) into the core store (`engram-core/src/store.rs:405`). To support autonomous agents sharing infrastructure, we need a control plane that narrates *which* memory graph each request should touch. The Memory Space Registry is that control plane.

## Constraints Driving the Design
1. **Runtime isolation** — every store, recall, or consolidation must resolve to a single `MemoryStore` instance keyed by a `MemorySpaceId` newtype. No stringly-typed IDs, no fallbacks that hide misconfiguration.
2. **Deterministic defaults** — existing deployments rely on a single unnamed space. The registry must bootstrap a configurable default space and keep legacy clients functional while emitting deprecation warnings.
3. **Symmetric lifecycle** — operators create spaces via CLI or API, the registry provisions persistence roots, and teardown reclaims resources. The lifecycle must be thread-safe because HTTP, gRPC, and CLI commands run concurrently.
4. **Extensibility** — future milestones introduce per-space quotas, consolidation cadences, or feature flags. The registry should carry metadata alongside store handles, not just raw pointers.

## API Shape
The registry centers around three operations:
- `create(space_id, options)` → returns an `Arc<MemoryStore>` handle (creating or reusing).
- `get(space_id)` → returns the existing store or an error instructing the caller to create the space.
- `list()` → surfaces registered spaces plus metadata (created_at, persistence paths, capacity caps).
All public APIs flow through the registry before touching a store. HTTP middleware, CLI subcommands, and gRPC handlers call `resolve_space(request_space_id)`; if no ID is supplied, the registry injects the configured default and logs the access.

## Concurrency Story
Registry state lives in a concurrent map (`DashMap<MemorySpaceId, Arc<SpaceHandle>>`) guarded by a lightweight RW lock for metadata updates. Each `SpaceHandle` bundles the `MemoryStore`, persistence configuration, and spawn guards for background workers. By caching handles, repeated per-request lookups on hot paths reduce to a single hash-map fetch.

## Operator Experience
We extend the CLI with `engram space list` and `engram space create <id>`. Listing surfaces the space ID, memory count, persistence root, and pressure metrics once Task 006 lands. Creating a space logs the filesystem paths that were provisioned so operators can integrate with backup policies.

## Path to Implementation
1. Introduce `MemorySpaceId` and registry scaffolding in `engram-core` so the core engine can be constructed per space.
2. Update CLI startup to instantiate the registry, bootstrap the default space, and expose handles via dependency injection.
3. Write lifecycle tests spawning concurrent `create/get` calls to prove idempotence and thread safety.
4. Ship new CLI commands and config fields; keep defaults aligned with legacy behavior until higher-level tasks enforce explicit IDs.

The registry unlocks everything else in Milestone 7. Once we can resolve a space-specific store reliably, we can partition persistence, tag metrics, and validate isolation with confidence.
