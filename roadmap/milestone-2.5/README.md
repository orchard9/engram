# Milestone 2.5: Integration Completion

## Overview

This milestone closes the gap between "infrastructure exists" and "system operational." Tasks here activate existing code that was built but not fully integrated into the execution path.

## Status: In Progress

**Created**: 2025-10-05
**Priority**: P0 (Critical - blocks production deployment)
**Estimated Duration**: 5-6 days (1 engineer)

## Problem Statement

Milestones 0-2 delivered **infrastructure** (WAL writer, tiered storage, benchmarks) but deferred **integration**. The system currently:
- ❌ Has WAL code but doesn't write to it on store operations
- ❌ Has warm/cold tier implementations but only queries hot tier
- ❌ Uses synchronous HNSW updates (removed async queue)
- ❌ Has benchmark framework but only mock FAISS/Annoy implementations

This milestone makes existing infrastructure **operational**.

## Tasks

### P0 (Critical Path - Must Complete First)
1. **001_activate_wal_persistence_pending.md** - Wire WAL writer into MemoryStore (2 days)
2. **002_activate_tiered_storage_pending.md** - Enable warm/cold tier queries and migration (2 days)

### P1 (Important - Should Complete)
3. **003_hnsw_queue_or_document_sync_pending.md** - Implement async queue consumer OR document sync design (1 day)
4. **004_real_faiss_annoy_integration_pending.md** - Replace mock benchmarks with real libraries (2 days)

## Dependencies

**Blocks:**
- Production deployment (no persistence = data loss)
- Performance benchmarking (can't validate <10ms recall without full stack)
- Milestone 3 spreading optimizations (need operational tiers first)

**Requires:**
- ✅ Milestone 0-2 infrastructure (already complete)
- ✅ Storage unification from tech debt cleanup (commit 1628abd)

## Success Criteria

- [ ] WAL writes to disk on every store operation with configurable fsync
- [ ] Queries search all tiers (hot → warm → cold) with migration
- [ ] HNSW has documented design (async queue OR intentional sync)
- [ ] FAISS/Annoy benchmarks run with real libraries, validate 90% recall@10

## Validation Approach

1. **WAL Recovery Test**: Kill process mid-write, restart, verify data recovered
2. **Tier Migration Test**: Store 1000 memories, verify automatic hot→warm→cold migration
3. **HNSW Performance Test**: Measure store latency with/without queue consumer
4. **Benchmark Validation**: Compare Engram vs FAISS/Annoy on SIFT1M dataset

## Risk Mitigation

- **WAL overhead**: Batch commits to stay <10ms P99 (already implemented in wal.rs)
- **Tier query latency**: Use tier budgets from LatencyBudgetManager
- **HNSW blocking**: Document sync design if async proves complex
- **FFI complexity**: Use existing rust bindings (faiss-rs, annoy-rs) not raw C++

## Notes

This milestone **activates existing code**, not builds new features. All infrastructure is already in codebase:
- `engram-core/src/storage/wal.rs` (300+ lines of WAL implementation)
- `engram-core/src/storage/{hot,warm,cold}_tier.rs` (tier implementations)
- `engram-core/benches/ann_comparison.rs` (benchmark framework)

We just need to **wire them into the execution path**.
