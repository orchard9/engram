# Memory Space Registry — Architectural Perspectives

## Cognitive-Architecture Perspective
A registry formalizes how agents perceive their own mnemonic boundaries. By forcing every recall/store request to declare a `MemorySpaceId`, we can model separate autobiographical selves without accidental cross-priming. It also gives us a substrate for space-specific decay curves or consolidation cadences later in the roadmap.

## Systems-Architecture Perspective
The registry becomes the authoritative router between user-facing protocols (HTTP, gRPC, CLI) and scoped `MemoryStore` instances. It must manage lifecycle hooks (create, recover, teardown) and supply persistence roots to downstream workers. Thread safety and low-latency lookups are paramount; caching `Arc<MemoryStore>` handles inside a concurrent map keeps per-request overhead minimal.

## Rust-Graph-Engine Perspective
Space isolation lets us evolve `MemoryStore` APIs so they operate behind typed handles rather than being instantiated ad hoc. With a registry controlling creation, we can eventually hide the constructor and expose a `SpaceHandle` that wraps graph operations, preventing misuse and paving the way for per-space activation schedulers.

## Operational Perspective
Operators need visible primitives—listing spaces, creating them with caps, and setting defaults. The registry is the nucleus for future quotas, billing hooks, or tenant-specific monitoring. Surfacing warnings when the default space is missing or when an operation references an unknown space reduces support burden and aligns with Engram’s “educate through errors” principle.
