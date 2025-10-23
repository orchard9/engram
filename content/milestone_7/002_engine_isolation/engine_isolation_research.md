# Engine Isolation & MemorySpace Enforcement — Research Notes

## Single-Space Assumptions Today
- `MemoryStore` holds global structures (`hot_memories`, `wal_buffer`, eviction queue) keyed only by memory ID, with no namespace field (`engram-core/src/store.rs:218-299`).
- HTTP server boots exactly one `MemoryStore` and shares it across routes and streaming channels, so every request mutates the same graph (`engram-cli/src/main.rs:171-236`).
- REST handlers invoke `state.store.store(...)` and `state.store.recall(...)` directly, meaning all tenants would write into the same in-memory map if we multiplexed agents (`engram-cli/src/api.rs:789-904`).
- gRPC service mirrors the REST flow, calling `self.store.store(...)` and `self.store.recall(...)` with no tenant key, reinforcing the single-space design (`engram-cli/src/grpc.rs:38-156`).

## Isolation Gaps
- Deduplication and eviction queues operate on global `DashMap`/`BTreeMap` instances, so cross-tenant collisions would immediately occur without additional partitioning (`engram-core/src/store.rs:223-245`).
- Spreading activation runs through `cognitive_recall.recall(cue, self)` with the store reference, enabling propagation across the entire graph (`engram-core/src/store.rs:1788-1824`).
- Event broadcasting emits `MemoryEvent` without any namespace metadata, making downstream subscribers unable to filter by tenant (`engram-core/src/store.rs:181-206`).

## Required Refactors
- Thread `MemorySpaceId` through store constructors so registries can produce one store per space before persistence attaches (depends on Task 001).
- Rework deduplicator/content index to either partition per space or wrap them in space-aware containers before storing keys (`engram-core/src/store.rs:270-290`).
- Require space handles in public APIs (`store`, `recall`, streaming) and update CLI/HTTP entry points to obtain the correct handle from the registry (ties into Task 004).
- Extend activation pipelines to verify that every neighbor traversal stays inside the originating space; likely by scoping the queue and visited sets per space (`engram-core/src/activation` modules).

## Testing Considerations
- Build multi-tenant integration fixtures that launch two spaces, seed overlapping memory IDs, and ensure recall in one space never surfaces data from the other (feeds Task 007).
- Ensure regression coverage for legacy single-space behavior by defaulting registry lookups to the configured default ID when a request omits explicit space info.
