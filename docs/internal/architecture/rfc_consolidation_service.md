# RFC: Consolidation Service Extraction (Draft)

**Status**: Draft proposal — Phase 2 wiring in progress (in-memory service landed 2025-10-19).

## Context

- Consolidation logic currently lives inside `MemoryStore` with direct access to scheduler/cache implementation details.
- Recent work introduced snapshot caching, metrics, and persistence hooks that benefit from a dedicated service boundary.

## Goals

- Extract a `ConsolidationService` trait that encapsulates scheduling, caching, observability, and snapshot retrieval.
- Allow multiple backends (in-process, distributed, remote cache) without modifying storage internals.
- Improve testability by enabling isolated consolidation harnesses and mocks.

## Non-Goals (for this RFC)

- Redesigning consolidation algorithms (pattern detection/semantic extraction remains as-is).
- Shipping distributed deployment support; this RFC prepares the boundary but doesn't deliver scaling changes.

## Proposed Architecture (High-Level)

1. `engram-core/src/consolidation/service.rs` defines the `ConsolidationService` trait and an `InMemoryConsolidationService` (implemented on 2025-10-19).
2. `MemoryStore` owns an `Arc<dyn ConsolidationService>`, delegating cache updates, belief logging, and metric emission to the service.
3. Background schedulers (`ConsolidationScheduler`) target the service instead of store internals, enabling alternate implementations to be swapped in.
4. Future backends (remote cache, distributed scheduler) implement the trait and can be injected at construction or via a factory.

### Proposed API (Current Draft)

```rust
pub trait ConsolidationService: Send + Sync {
    fn cached_snapshot(&self) -> Option<ConsolidationSnapshot>;
    fn update_cache(&self, snapshot: &ConsolidationSnapshot, source: ConsolidationCacheSource);
    fn set_alert_log_path(&self, path: PathBuf);
    fn alert_log_path(&self) -> PathBuf;
    fn recent_updates(&self) -> Vec<BeliefUpdateRecord>;
}

pub struct InMemoryConsolidationService { /* cache + observability state */ }
```

The in-memory implementation computes belief deltas, persists JSONL logs, and updates gauges/counters (`engram_consolidation_*`). Trait consumers never touch the underlying locks/files directly.

## Migration Plan

- **Phase 1 (Complete)**: Extract cache/metrics helpers into the consolidation module and expose the preview trait.
- **Phase 2 (In Progress)**: Wire `MemoryStore` and `ConsolidationScheduler` through `ConsolidationService` (commit e85a… introduces the trait and service). Remaining work:
  - Allow service injection via builder/DI so CLI can swap implementations.
  - Move belief-update log rotation & retention policy behind the trait.
  - Provide async-aware service adapter for distributed schedulers.
- **Phase 3 (Planned, Milestone 6 → early Milestone 7)**:
  - Delete deprecated `MemoryStore::update_consolidation_cache` helpers.
  - Update API/CLI surfaces to query scheduler status via service facade.
  - Ship remote-cache prototype (e.g., Redis) guarded behind feature flag.
  - Finalize documentation + dashboards around the new boundary.

### Target Timeline

- **2025-10-25**: Service injection & API stabilization (Phase 2 complete).
- **2025-11-08**: Remote cache adapter prototype & distributed scheduler spike.
- **2025-11-22**: Phase 3 clean-up, remove legacy fields, promote RFC to Accepted.

## Implementation Notes

- The soak harness (`cargo run --bin consolidation-soak`) already targets the new service via `MemoryStore::set_consolidation_alert_log_path` and serves as a regression guard for future service implementations.
- `docs/assets/consolidation/baseline/*.jsonl` captures a short-run reference; refresh with a 1h run when validating alternative backends.

## Open Questions

- Configuration boundaries: Should CLI own service selection, or do we expose a builder inside `engram-core`?
- Persistence strategy: keep writing JSONL logs directly from the service, or abstract into a logging subsystem (allows S3/GCS backends)?
- Observability surfacing: ship dashboard definitions alongside the service crate?

> Action: continue Phase 2 work (service injection + alternate implementations) and update this RFC before requesting "Accepted" status.
