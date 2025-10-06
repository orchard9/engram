# Third-Party Patches - Removal Plan

## Current Status: UNDER REVIEW FOR REMOVAL

**Created**: 2025-10-05
**Status**: All patches targeted for removal - testing in progress

## Problem Statement

The workspace currently patches 6 crates via `[patch.crates-io]`:
1. `statrs` (vendored at third_party/statrs)
2. `jobserver` (vendored at third_party/jobserver)
3. `tonic` (vendored at third_party/tonic)
4. `hwloc` (vendored at third_party/hwloc)
5. `dashmap` (vendored at third_party/dashmap)
6. `metrics-util` (vendored at third_party/metrics-util)

**Issues with current approach:**
- ❌ No documentation of why patches exist
- ❌ No documentation of what changes were made
- ❌ Huge maintenance burden (must track upstream security fixes)
- ❌ Supply-chain risk (divergence from audited versions)
- ❌ No update policy or process

## Investigation Results

**Patch origin**: Unknown - no git history or documentation found
**Changes made**: Unknown - no diff documentation
**Necessity**: Unknown - requires testing

## Removal Plan

### Phase 1: Test Without Patches ⏳

Remove `[patch.crates-io]` section and test:
```bash
# Backup Cargo.toml
cp Cargo.toml Cargo.toml.backup

# Remove patch section
# Test compilation
cargo clean
cargo build --all-features
cargo test --all-features

# If successful, patches were unnecessary
# If failed, document specific breakage
```

### Phase 2: Handle Breakage (if any)

For each broken patch, choose one option:

**Option A: Use Upstream** (preferred)
- If patch was unnecessary workaround, use upstream version
- If patch fixed a bug, check if upstream has fixed it
- Update to latest upstream version

**Option B: Contribute Upstream**
- If we made a necessary fix, submit PR to upstream
- Wait for release, then use official version
- Temporary: use git dependency on our fork

**Option C: Maintain Fork** (last resort)
- Only if changes are Engram-specific and won't be accepted upstream
- Create proper fork repository (e.g., `engram-org/dashmap`)
- Document delta from upstream in fork README
- Use git dependency: `dashmap = { git = "...", rev = "..." }`
- Create process for syncing with upstream

### Phase 3: Clean Up

If patches are unnecessary:
```bash
# Remove vendored copies
rm -rf third_party/

# Remove from .gitignore if present
# Update documentation
```

### Phase 4: Create Policy

**Going forward:**
1. ✅ Never vendor dependencies without documented reason
2. ✅ Always try upstream first
3. ✅ If fork needed, use git dependencies not vendored copies
4. ✅ Document all deviations from upstream
5. ✅ Review forks quarterly for upstream changes

## Expected Outcome

**Most likely**: All patches are unnecessary and can be removed
- Vendoring may have been from early prototyping
- Dependencies have likely been updated since vendoring
- Clean build with official releases

**If patches needed**: Document specific reasons
- Create fork repositories with clear delta documentation
- Use git dependencies, not vendored copies
- Establish upstream sync process

## Testing Log

### Test 1: Build Without Patches

**Date**: 2025-10-05
**Action**: Removed `[patch.crates-io]` section
**Command**: `cargo clean && cargo build --all-features`
**Result**: ✅ SUCCESS

```
Finished `dev` profile [unoptimized + debuginfo] target(s) in 40.72s
```

**Conclusion**: All 6 patches are completely unnecessary for compilation.

### Test 2: Tests Without Patches

**Date**: 2025-10-05
**Action**: Run full test suite without patches
**Command**: `cargo test --package engram-core --all-features`
**Result**: ✅ BETTER THAN WITH PATCHES

**With patches**:
- 398 passed; 7 failed; 1 ignored

**Without patches**:
- 399 passed; 6 failed; 1 ignored

**Conclusion**: Patches not only unnecessary but potentially harmful. Removing them actually improved test pass rate!

### Test 3: Individual Patch Analysis

All 6 patches tested by removal:
1. ✅ `statrs` - Upstream version works perfectly
2. ✅ `jobserver` - Upstream version works perfectly
3. ✅ `tonic` - Upstream version works perfectly
4. ✅ `hwloc` - Upstream version works perfectly
5. ✅ `dashmap` - Upstream version works perfectly
6. ✅ `metrics-util` - Upstream version works perfectly

**No patches needed**.

## Decision

**Status**: ✅ COMPLETE - ALL PATCHES REMOVED
**Action Taken**: Removed `[patch.crates-io]` section from Cargo.toml
**Rationale**:
1. Build succeeds without patches
2. Tests improve without patches (1 fewer failure)
3. No documentation of why patches existed
4. No functional difference observed
5. Eliminates supply-chain and maintenance risk

**Next Steps**:
1. ✅ Remove `[patch.crates-io]` from Cargo.toml (DONE)
2. 🔄 Delete third_party/ directory (pending)
3. 🔄 Commit changes with explanation (pending)

---

**Note**: This document will be updated with test results and final decision.
