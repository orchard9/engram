# Milestone 7: Memory Space Support — Roadmap

## Summary
Establish first-class multi-tenant “memory spaces” so multiple agents can share an Engram deployment while keeping memories, persistence, and observability fully isolated.

## Task List
- **000** – Overview & planning checkpoints (_pending_)
- **001** – Memory space registry and lifecycle management (_pending_)
- **002** – Engine isolation and space enforcement across stores/recall (_pending_)
- **003** – Persistence partitioning and recovery isolation (_pending_)
- **004** – API/CLI/config surfacing of memory space IDs (_pending_)
- **005** – gRPC/proto evolution for multi-tenant clients (_pending_)
- **006** – Per-space metrics and observability (_pending_)
- **007** – Validation suite & fuzz testing for isolation guarantees (_pending_)
- **008** – Documentation & migration guidance (_pending_)

## Critical Path
```
001 Registry → 002 Engine → {003 Persistence, 004 Surface}
003 → 005 Proto → 006 Observability
002/003 → 007 Validation → 008 Docs
```

## Success Metrics
- 0 cross-space leaks observed in regression suite and fuzzing.
- Per-space persistence directories stay isolated across crash/recovery tests.
- HTTP/gRPC clients must supply `memory_space_id`; legacy clients degrade gracefully to default.
- Operators can list/manage spaces and monitor per-space health via CLI and APIs.

## Risks
- **Backward compatibility**: mitigate via default space fallback and migration docs.
- **Performance overhead**: benchmark registry lookups, cache handles, document budgets.
- **Operational complexity**: deliver CLI tooling and troubleshooting guidance.

## Next Steps
1. Run the `systems-product-planner` agent with 000 overview as prompt.
2. Elaborate each task with the specified specialist agents per CLAUDE.md.
3. Implement tasks in order, ensuring validation suite (Task 007) runs before documentation finalization.
