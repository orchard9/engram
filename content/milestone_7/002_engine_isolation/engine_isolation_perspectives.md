# Engine Isolation: Multi-Perspective Analysis

## Rust-Graph-Engine Perspective
Isolating memory spaces means every lock-free structure the graph engine relies on must become space-aware. The current `MemoryStore` bundles `DashMap` collections, eviction queues, content indexes, and activation workers behind a single instance, so any tenant shares the same concurrent maps (`engram-core/src/store.rs:218-294`). To prevent contention cascades, we need either one store per space or composite containers that shard by `MemorySpaceId`. That keeps memory fences and atomic counters scoped, sustaining deterministic activation ordering when spreading runs in parallel.

## Systems-Architecture Perspective
On startup the CLI allocates a single `MemoryStore`, wires optional persistence, and then shares that `Arc` with both HTTP and gRPC stacks (`engram-cli/src/main.rs:171-236`). Each REST handler and gRPC method calls `store.store(...)` or `store.recall(...)` directly, so tenants collide the moment they hit the API gateway (`engram-cli/src/api.rs:789-904`; `engram-cli/src/grpc.rs:38-156`). Engine isolation therefore requires routing every inbound request through a registry that hands back the correct space-scoped handle before touching the graph.

## Cognitive-Architecture Perspective
Engram’s vision emphasizes local computation—global consistency should emerge from independent regions cooperating, not sharing mutable state (`vision.md:5-52`). Memory spaces strengthen that principle: each agent’s recall dynamics evolve independently, and inter-space interference becomes an explicit design choice rather than an emergent bug. By enforcing space boundaries in the engine, we maintain cognitive plausibility while unlocking collaborative deployments.

## Observability Perspective
Downstream subscribers currently receive `MemoryEvent` payloads without any namespace metadata, making it impossible to build tenant-aware dashboards or alerting (`engram-core/src/store.rs:181-206`). Engine isolation must therefore extend to event emission: every store, recall, and activation broadcast carries the originating `MemorySpaceId`, enabling metrics and SSE clients to filter correctly (feeding Tasks 006 and 007).

### References
- `engram-core/src/store.rs:181-299`
- `engram-cli/src/main.rs:171-236`
- `engram-cli/src/api.rs:789-904`
- `engram-cli/src/grpc.rs:38-156`
- `vision.md:5-52`
