# Consolidation Module Boundary

Engram currently anchors consolidation logic inside `MemoryStore`. To support long-term maintainability, we will promote consolidation into its own module/service boundary with the following design principles:

## Goals

- **Separation of concerns**: isolate background scheduling, snapshot caching, and observability from core storage operations.
- **Extensibility**: allow alternate schedulers (distributed, GPU-accelerated) and plug-in pattern detection strategies without refactoring `MemoryStore`.
- **Testability**: enable integration tests that exercise consolidation workflows independent of the entire store.
- **Observability**: centralize metrics/logging so production systems can swap exporters or add pipelines without touching storage code.

## Proposed Structure

```
engram-core/src/
  consolidation/
    mod.rs
    cache.rs          # Cache abstraction backed by RwLock/Arc
    scheduler.rs      # Existing scheduler moved here with trait-based episode sources
    observability.rs  # Metrics + belief-update logging façade
    pipeline.rs       # Pattern detection + semantic extraction orchestration
```

- `MemoryStore` becomes a thin client that publishes episodes into the consolidation pipeline and reads cached snapshots.
- `ConsolidationCache` exposes a trait for alternative backends (in-memory, Redis, distributed store).
- `ConsolidationEngine` focuses on pattern extraction; scheduling, caching, and metrics wrap around it.

## Migration Plan

1. **Phase 1 (Current)**: maintain compatibility using wrapper types (`ConsolidationCacheSource`, `BeliefUpdateRecord`).
2. **Phase 2** *(in progress)*: caching + metrics now live in `consolidation::service` with `MemoryStore` delegating to `InMemoryConsolidationService`. Next steps: expose service injection hooks and migrate remaining helper methods.
3. **Phase 3**: expose a public facade (`ConsolidationService`) that `MemoryStore` depends on. Deprecate direct calls to `MemoryStore::consolidation_snapshot` once the service is stable.
4. **Phase 4**: introduce distributed scheduler support as an optional crate (`engram-consolidation`).

## Open Questions

- Should the cache live in-process only, or support remote persistence (e.g., Redis) out of the box?
- How do we version snapshots for cross-node replication (task follow-up under Milestone 7)?
- What API should CLI/HTTP expose for scheduler administration (pause/resume/status)?

Track progress via Milestone 6 follow-ups and a future architecture RFC before Phase 2 begins.
