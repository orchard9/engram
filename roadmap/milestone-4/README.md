# Milestone 4: Temporal Dynamics

## Overview

Implement biologically-inspired temporal decay for episodic memories based on psychological forgetting curves. Decay is computed lazily during recall operations, not via background threads, ensuring deterministic results and zero write amplification.

## Objective

Create automatic decay functions applying to all episodes based on `last_access` patterns. Implement forgetting curves matching psychological research (Ebbinghaus, spaced repetition).

## Critical Requirements

- Decay must happen lazily during recall, not requiring background threads
- Must support multiple decay functions (exponential, power law) selectable per memory
- Forgetting curves must match published psychology data within 5% error
- No memory leaks during long-running decay using valgrind validation

## Success Criteria

- Exponential decay matches Ebbinghaus curve within 5% error at all data points
- Power-law decay matches Wickelgren data within 5% error
- Two-component model demonstrates consolidation benefit (frequently accessed memories decay slower)
- Zero memory leaks during continuous recall operations
- Decay computation <1ms p95 per memory
- Comprehensive test coverage (≥90%)

## Architecture

### System Components

```
┌──────────────────────────────────────────────────────┐
│                 CognitiveRecall                       │
│  ┌────────────────────────────────────────────────┐ │
│  │  Lazy Decay Integration                        │ │
│  │  - Apply decay during result ranking          │ │
│  │  - Update last_access on retrieval             │ │
│  │  - Select decay function based on access count │ │
│  └────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────┘
                        ↓
┌──────────────────────────────────────────────────────┐
│           BiologicalDecaySystem (existing)            │
│  ┌────────────────────────────────────────────────┐ │
│  │  HippocampalDecayFunction                      │ │
│  │  - Fast exponential: R(t) = e^(-λt)           │ │
│  │  - For new/infrequent memories                 │ │
│  └────────────────────────────────────────────────┘ │
│  ┌────────────────────────────────────────────────┐ │
│  │  NeocorticalDecayFunction                      │ │
│  │  - Slow power-law: R(t) = (1+t)^(-α)         │ │
│  │  - For consolidated/frequent memories          │ │
│  └────────────────────────────────────────────────┘ │
│  ┌────────────────────────────────────────────────┐ │
│  │  TwoComponentModel                             │ │
│  │  - Adaptive: switches based on access_count   │ │
│  │  - Consolidation threshold: ≥3 retrievals     │ │
│  └────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────┘
                        ↓
┌──────────────────────────────────────────────────────┐
│               Memory / Episode                        │
│  ┌────────────────────────────────────────────────┐ │
│  │  Temporal Metadata (NEW)                       │ │
│  │  - last_access: DateTime<Utc>                 │ │
│  │  - access_count: u64                           │ │
│  │  - decay_function: Option<DecayFunction>      │ │
│  └────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────┘
```

### Core Design Principles

1. **Lazy Evaluation**: Decay computed during recall, not background processes
   - Avoids write amplification from frequent updates
   - Enables deterministic, reproducible results
   - Supports time-travel queries

2. **Dual-System Architecture**: Hippocampal (fast) + Neocortical (slow)
   - Matches complementary learning systems theory
   - Automatically adapts based on retrieval patterns
   - Models consolidation through repeated access

3. **Configurable per Memory**: System-wide defaults + per-memory overrides
   - Critical memories can have slower decay
   - Different memory types use appropriate functions
   - Supports personalization

4. **Psychological Validation**: All functions validated against published research
   - Ebbinghaus (1885) exponential forgetting
   - Wickelgren (1974) power-law decay
   - SuperMemo spaced repetition effects

## Task Breakdown

### Task 001: Content Creation (COMPLETE)
**Status**: ✅ Complete
**Deliverables**: Research, perspectives, Medium article, Twitter thread

### Task 002: Last Access Tracking
**Priority**: P0 (blocking) | **Effort**: 1 day | **Dependencies**: None

Add `last_access` and `access_count` fields to Memory/Episode types. Update automatically during recall operations.

**Key Deliverables**:
- `last_access: DateTime<Utc>` field
- `access_count: u64` field
- Automatic update during recall
- Lazy persistence (no write amplification)

### Task 003: Lazy Decay Integration
**Priority**: P0 (blocking) | **Effort**: 2 days | **Dependencies**: Task 002

Integrate `BiologicalDecaySystem` with `CognitiveRecall` to apply decay lazily during result ranking.

**Key Deliverables**:
- Decay applied during `rank_results()`
- Optional `decay_system` in `CognitiveRecall`
- Selection of hippocampal vs neocortical based on `access_count`
- <1ms p95 decay computation time

### Task 004: Decay Configuration API
**Priority**: P1 (important) | **Effort**: 1.5 days | **Dependencies**: Tasks 002, 003

Create builder API for configuring decay functions system-wide and per-memory.

**Key Deliverables**:
- `DecayFunction` enum (Exponential, PowerLaw, TwoComponent)
- `DecayConfig` with builder pattern
- Per-memory decay function override
- Ergonomic configuration API

### Task 005: Forgetting Curve Validation
**Priority**: P1 (critical) | **Effort**: 2 days | **Dependencies**: Tasks 002-004

Validate decay functions against published psychology forgetting curves with <5% error.

**Key Deliverables**:
- Ebbinghaus curve validation (exponential)
- Wickelgren curve validation (power-law)
- Two-component consolidation effects
- Validation report with statistical analysis

### Task 006: Comprehensive Testing
**Priority**: P1 (critical) | **Effort**: 1.5 days | **Dependencies**: Tasks 002-005

Create comprehensive test suite covering integration, edge cases, performance, and memory safety.

**Key Deliverables**:
- Integration tests (end-to-end decay behavior)
- Edge case tests (boundary conditions, thread safety)
- Performance benchmarks (<1ms p95 target)
- Memory leak tests (long-running operations)

### Task 007: Documentation
**Priority**: P2 (important) | **Effort**: 1 day | **Dependencies**: All previous tasks

Comprehensive documentation covering architecture, API usage, configuration, and psychology foundations.

**Key Deliverables**:
- Architecture documentation
- Decay functions reference
- Configuration tutorial
- Module README

## Critical Path

```
Task 002 (1d) → Task 003 (2d) → Task 004 (1.5d) → Task 005 (2d) → Task 006 (1.5d) → Task 007 (1d)
```

**Total Duration**: ~9 days (critical path)

## Implementation Order

1. **Phase 1: Foundation** (Tasks 002-003, 3 days)
   - Add temporal metadata to memories
   - Integrate decay with recall
   - Validation Gate: End-to-end decay working

2. **Phase 2: Configuration** (Task 004, 1.5 days)
   - Build configuration API
   - Validation Gate: All decay functions configurable

3. **Phase 3: Validation** (Tasks 005-006, 3.5 days)
   - Validate against psychology curves
   - Comprehensive testing
   - Validation Gate: <5% error on forgetting curves, ≥90% test coverage

4. **Phase 4: Documentation** (Task 007, 1 day)
   - Complete documentation
   - Validation Gate: All examples compile and run

## Key Risks

1. **Decay Computation Adds Latency** (Medium probability, Medium impact)
   - Mitigation: Profile early, cache decay factors, use SIMD if needed
   - Fallback: Make decay optional via feature flag

2. **Psychology Curves Don't Generalize** (Low probability, High impact)
   - Mitigation: Validate on multiple datasets, document limitations
   - Fallback: Support custom decay functions

3. **Access Pattern Tracking Causes Write Amplification** (Medium probability, Medium impact)
   - Mitigation: Lazy persistence, batch updates, approximate timestamps acceptable
   - Fallback: Make access tracking optional

4. **Memory Leaks in Long-Running Operations** (Low probability, Critical impact)
   - Mitigation: Comprehensive memory safety tests, use valgrind
   - Fallback: Add periodic decay system reset

## Status

**Current Status**: 🚧 **IN PROGRESS** (28% - 2/7 tasks)

**Completed Tasks**:
- ✅ Task 001: Content Creation (research, Medium article, Twitter thread)
- ✅ Task 002: Last Access Tracking (added access_count to Memory, 7 unit tests)

**In Progress**:
- 🔄 Task 003: Lazy Decay Integration (2 days) ← **NEXT**

**Pending Tasks**:
- ⏸️ Task 004: Decay Configuration API (1.5 days)
- ⏸️ Task 005: Forgetting Curve Validation (2 days)
- ⏸️ Task 006: Comprehensive Testing (1.5 days)
- ⏸️ Task 007: Documentation (1 day)

**Phase 1 Status**: Task 002 complete ✅, Task 003 next (Foundation phase 50% complete)

## Related Documentation

- `milestones.md` - Overall project milestones
- `vision.md` - Architecture vision and cognitive principles
- `chosen_libraries.md` - Approved dependencies
- `coding_guidelines.md` - Rust coding standards
- `docs/temporal-dynamics.md` - Temporal decay architecture (created in Task 007)
