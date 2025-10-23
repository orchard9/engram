# Making Engram's Memory Engine Multi-Tenant Without Losing Its Soul

Engram’s promise is a cognitive database that thinks like a human memory system. That promise breaks the moment two independent agents overwrite each other’s memories or leak recall results across projects. Milestone 7 is our inflection point: we must lift the engine from a single global store to a set of isolated “memory spaces” while keeping latency, probabilistic semantics, and spreading activation intact.

## Why Isolation Is Non-Negotiable
The vision document is clear: global consistency should emerge from local rules, not shared mutable state (`vision.md:5-52`). Right now every HTTP and gRPC request flows into the same `MemoryStore` instance that the CLI boots on startup, so multiple agents would collide immediately (`engram-cli/src/main.rs:171-236`; `engram-cli/src/api.rs:789-904`; `engram-cli/src/grpc.rs:38-156`). Without a boundary, deduplication, eviction, and activation operate on a single global namespace, violating both cognitive plausibility and tenant trust.

## Anatomy of the Current Store
`MemoryStore` bundles all hot-tier data structures—`DashMap` holdings, eviction B-tree, WAL buffer, event broadcaster, HNSW queue—into a single struct keyed only by memory IDs (`engram-core/src/store.rs:218-299`). The API surface (`store`, `recall`, `recall_with_mode`, streaming) assumes implicit access to that singleton. Any multi-tenant solution has to refactor these internals so space awareness happens before we ever touch those collections.

## Designing Space-Aware Stores
The registry arriving in Task 001 will mint `MemorySpaceId`s and provision store handles per space. Engine isolation extends that work: instead of exposing one monolithic `MemoryStore`, we wrap the struct in a `SpaceHandle` that pins all concurrent structures, persistence hooks, and metrics to the chosen ID. Persistence partitioning in Task 003 then becomes a natural extension—each handle points at its own WAL writer and tiered directories (`engram-core/src/store.rs:270-290`). The CLI and server no longer share a single `Arc`; they request the correct handle from the registry before serving a request (`engram-cli/src/main.rs:171-236`).

## Guarding Activation & Recall Boundaries
Spreading activation is where isolation can quietly fail. The cognitive recall pipeline currently receives the entire store reference and walks the global graph (`engram-core/src/store.rs:1788-1833`). Our refactor introduces space-scoped traversal state: visited sets, activation queues, and auto-tuning metrics reside inside the handle. Event streaming also gains the space ID, allowing downstream analytics to filter multi-tenant traffic (`engram-core/src/store.rs:181-206`).

## Implementation Checklist
1. Inject `MemorySpaceId` into every store/recall entry point before compilation succeeds.
2. Partition deduplication, content indexing, and eviction so they never cross spaces.
3. Update HTTP/gRPC/CLI layers to resolve the correct space handle via the registry.
4. Attach space metadata to activation events and metrics for Tasks 006 and 007.
5. Run the new multi-tenant validation suite to prove no leakage under concurrent load.

Each step keeps Engram aligned with its cognitive design goals while unlocking collaborative deployments.

## References
- `vision.md:5-52`
- `engram-cli/src/main.rs:171-236`
- `engram-cli/src/api.rs:789-904`
- `engram-cli/src/grpc.rs:38-156`
- `engram-core/src/store.rs:181-299`
- `engram-core/src/store.rs:1788-1833`
