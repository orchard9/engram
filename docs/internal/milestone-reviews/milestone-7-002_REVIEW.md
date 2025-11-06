# Task 002: Engine Isolation - Critical Review

**Date**: 2025-10-22
**Status**: ⚠️ INCOMPLETE - Major gaps in implementation
**Recommendation**: BLOCK completion, requires substantial additional work

---

## Executive Summary

Task 002 was intended to **eliminate cross-tenant leakage** by partitioning MemoryStore state per memory space. The current implementation adds tracking metadata but **does not achieve isolation**. The core deliverables around state partitioning and API refactoring are missing.

**Risk Level**: 🔴 HIGH - Current code creates false sense of security without actual isolation guarantees.

---

## What Was Completed ✅

### 1. Event Stream Metadata (90% complete)
- ✅ Added `memory_space_id: MemorySpaceId` to all three MemoryEvent variants
- ✅ Updated event emission in `store.rs:1220, 1587, 1610` to include space ID
- ✅ SSE streaming endpoints serialize space ID to JSON
- ⚠️ Missing: Per-space broadcast channels (currently all spaces share one channel)

### 2. Store Metadata (50% complete)
- ✅ Added `memory_space_id` field to `MemoryStore` struct
- ✅ Created `for_space(id, capacity)` constructor
- ✅ Updated registry factory to use `for_space()`
- ❌ Missing: Actual state partitioning within the store

### 3. Code Quality (100% complete)
- ✅ Fixed all clippy warnings (const functions, #[must_use], Debug impls)
- ✅ Updated test patterns for new MemoryEvent structure
- ✅ 627 lib tests passing

---

## Critical Gaps ❌

### 1. **State Partitioning NOT Implemented** (0% complete)
**Required**: Partition global collections per space
**Current**: Single shared state across all instances

```rust
// REQUIRED (from task spec line 34-36):
DashMap<MemorySpaceId, DashMap<String, Arc<Memory>>>
// or
HashMap<MemorySpaceId, SpaceHotMemories>

// ACTUAL (engram-core/src/store.rs:232):
hot_memories: DashMap<String, Arc<Memory>>,  // NOT PARTITIONED
wal_buffer: Arc<DashMap<String, Episode>>,   // NOT PARTITIONED
episode_stored_at: DashMap<String, ...>,      // NOT PARTITIONED
eviction_queue: RwLock<BTreeMap<...>>,       // NOT PARTITIONED
```

**Impact**: No actual isolation. Each MemoryStore instance is isolated ONLY because the registry creates separate instances. If a handler accidentally uses the wrong store reference, memories leak across spaces.

**Evidence**:
- `engram-core/src/store.rs:224-301` - all collections are global to the store instance
- No keying by MemorySpaceId anywhere in store.rs

---

### 2. **API Refactor NOT Done** (0% complete)
**Required**: Change signatures to require space parameters
**Current**: All methods still use implicit self state

```rust
// REQUIRED (from task spec line 38):
pub fn store(&self, space: &MemorySpaceId, episode: Episode)

// ACTUAL (engram-core/src/store.rs:1177):
pub fn store(&self, episode: Episode) -> StoreResult  // No space param!
```

**Impact**: Compile-time enforcement of space isolation is missing. Acceptance criterion #1 fails: "Compilation fails if any store/recall path omits a MemorySpaceId."

**Missing Updates**:
- `store()`
- `recall()`
- `recall_with_mode()`
- `recall_probabilistic()`
- `remove_consolidated_episodes()`
- `enable_event_streaming()`

---

### 3. **Activation Boundaries NOT Enforced** (0% complete)
**Required**: Prevent spreading activation from crossing space boundaries
**Current**: No checks or guards added

**Location**: `engram-core/src/activation/*`
**Status**: No changes made to activation pipeline
**Impact**: Spreading activation can traverse across spaces if memory graphs are connected

---

### 4. **Entry Point Wiring NOT Updated** (0% complete)
**Required**: Handlers fetch handles from registry
**Current**: Handlers still use `state.store` directly

```rust
// REQUIRED (from task spec line 49-51):
// In each handler, resolve the caller's space then call space-scoped store

// ACTUAL (engram-cli/src/api.rs:789-904):
let store_result = state.store.store(core_episode);  // Uses ApiState.store directly!
```

**Impact**: All API operations default to single space. Multi-tenancy not functional.

---

### 5. **SpaceStore Handle NOT Created** (0% complete)
**Required**: Introduce lightweight handle type
**Current**: Using MemoryStore directly with added field

```rust
// REQUIRED (from task spec line 30-32):
pub struct SpaceStore {
    space_id: MemorySpaceId,
    inner: Arc<MemoryStoreInner>
}

// ACTUAL: No such type exists
```

**Impact**: Architecture doesn't match spec. Registry pattern works but lacks the clean API boundary.

---

### 6. **Testing NOT Written** (0% complete)
**Required**:
- Unit tests with overlapping memory IDs across two spaces
- Integration tests for concurrent multi-space operations
- Regression tests for single-space compatibility

**Current**: No multi-space isolation tests exist

**Impact**: No validation that isolation actually works. Acceptance criteria #2 cannot be verified.

---

## Technical Debt 🔶

### Immediate
1. **False Security**: Code appears space-aware but doesn't enforce isolation
2. **Incomplete Abstraction**: `memory_space_id` field exists but serves only documentary purpose
3. **Missing Validation**: No assertions that memories belong to expected space
4. **Event Channel Sharing**: All spaces share broadcast channel (potential perf issue at scale)

### Medium-Term
1. **Registry Coupling**: Relies entirely on registry creating separate instances
2. **Error Handling**: No MemorySpaceError variants for cross-space violations
3. **Metrics**: Auto-tuner and recall metrics not scoped per space
4. **Documentation**: Inline docs don't explain multi-space semantics

### Long-Term
1. **Architecture Mismatch**: Implementation diverged from spec (no SpaceStore)
2. **Migration Path**: Unclear how to refactor from current state to partitioned collections
3. **Performance**: Partitioned DashMaps may have overhead vs current design

---

## Acceptance Criteria Status

| # | Criterion | Status | Evidence |
|---|-----------|--------|----------|
| 1 | Compilation fails if space omitted | ❌ FAIL | Methods don't require space params |
| 2 | Activation never crosses spaces | ❌ UNTESTED | No tests, no guards added |
| 3 | Events include memory_space_id | ✅ PASS | All variants updated |
| 4 | Default space works without changes | ✅ PASS | Single-space tests passing |

**Overall**: 2/4 passing (50%)

---

## Recommendations

### 1. **STOP and Reassess** ⛔
Do not mark this task complete. Current state creates technical debt and false assumptions about isolation.

### 2. **Choose Architecture Pattern** 🏗️

**Option A: Registry-Only Isolation** (Simpler, matches current)
- Accept that isolation comes from registry creating separate instances
- Remove `memory_space_id` field from MemoryStore (it's redundant)
- Add defensive assertions that check store instance matches expected space
- Document that stores MUST be obtained via registry
- Add runtime guards in API handlers

**Option B: Partitioned Collections** (Original spec, more robust)
- Implement task as specified with partitioned data structures
- Changes required:
  - Wrap collections in `PerSpaceData<T>` helper
  - Update all 50+ call sites in store.rs to pass space ID
  - Add SpaceStore handle type
  - Refactor API handlers to resolve space

**Recommendation**: Option A for pragmatism, Option B for correctness. Option B aligns with spec but is 3-5x more work.

### 3. **Minimum Viable Isolation** 🛡️

If proceeding with Option A (registry-only):

```rust
// Add to MemoryStore
pub fn verify_space(&self, expected: &MemorySpaceId) -> Result<(), MemorySpaceError> {
    if &self.memory_space_id != expected {
        return Err(MemorySpaceError::WrongStore {
            expected: expected.clone(),
            actual: self.memory_space_id.clone(),
        });
    }
    Ok(())
}

// In API handlers
let space_id = extract_space_from_request(&request)?;
let handle = state.registry.create_or_get(&space_id).await?;
handle.store().verify_space(&space_id)?;  // Runtime guard
```

### 4. **Testing Requirements** ✅

Before marking complete:
- [ ] Write `test_spaces_are_isolated()` that stores ep_001 in alpha and beta, verifies recall in alpha doesn't see beta's ep_001
- [ ] Write `test_concurrent_multi_space_operations()` with 3+ spaces
- [ ] Write `test_activation_boundary_enforcement()` checking spreading doesn't cross
- [ ] Add `#[should_panic]` test for deliberately using wrong store instance

### 5. **Documentation Updates** 📝

Update task file with:
- Architecture decision (registry-only vs partitioned)
- Known limitations section
- Migration notes for future full partitioning
- Runtime vs compile-time isolation tradeoffs

---

## Proposed Next Steps

1. **Immediate** (1-2 hours):
   - Add `verify_space()` method with runtime checks
   - Write minimal isolation test
   - Update task status to reflect true completion %

2. **Short-term** (4-6 hours, blocks Task 004/005):
   - Implement space extraction in API handlers
   - Wire handlers to use registry
   - Add integration tests for multi-space scenarios

3. **Medium-term** (optional, technical debt paydown):
   - Evaluate partitioned collections approach
   - Create migration plan if needed
   - Implement compile-time enforcement

---

## Impact on Downstream Tasks

| Task | Impact | Risk |
|------|--------|------|
| 003 - Persistence Partitioning | 🟡 Medium | Can proceed with registry-based approach |
| 004 - API/CLI Surface | 🔴 High | Blocked until handlers wire to registry |
| 005 - gRPC Proto | 🔴 High | Blocked until space resolution works |
| 006 - Observability | 🟡 Medium | Events have space ID but filtering not impl |
| 007 - Validation | 🔴 High | Cannot validate what doesn't exist |

---

## Conclusion

Task 002 delivered ~30% of required functionality. The work done is valuable (events, constructors, tests) but the core isolation mechanism is missing. The current implementation creates a **false sense of security** - it looks like isolation exists (memory_space_id field) but doesn't actually prevent leakage.

**Critical Action**: Team must decide between:
1. Accept registry-only isolation + add guards (pragmatic, 4-6 hrs)
2. Implement full partitioned collections (spec-compliant, 16-20 hrs)
3. Redesign approach entirely (unknown effort)

Without choosing a path forward, subsequent tasks will build on unstable foundation.

---

**Reviewed by**: Claude (Sonnet 4.5)
**Next Reviewer**: rust-graph-engine-architect (as specified in task)
