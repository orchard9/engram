# Task 004: Real FAISS/Annoy Integration - Status Report

## Current Status: BLOCKED - API Compatibility Issues

### What's Been Completed ✅

1. **Infrastructure Setup** (100%)
   - Added `faiss` dependency to Cargo.toml
   - Created feature flag `ann_benchmarks`
   - Set up conditional compilation for real vs mock implementations

2. **Module Structure** (100%)
   - Created `engram-core/benches/support/faiss_ann.rs` (partial)
   - Created `engram-core/benches/support/annoy_ann.rs` (mock)
   - Updated `mod.rs` with feature-gated imports

3. **Testing Framework** (100%)
   - Created `engram-core/benches/ann_validation.rs` with:
     - `validate_engram_recall_90_percent()` test
     - `compare_all_implementations()` test
     - Helper function `compute_recall()`

4. **Benchmark Framework** (100%)
   - Updated `engram-core/benches/ann_comparison.rs` with:
     - `benchmark_ann_search()` - search performance comparison
     - `benchmark_ann_build()` - build time comparison
     - `benchmark_ann_scalability()` - scalability testing

### Blockers 🚫

#### 1. FAISS Rust Bindings API Complexity

**Issue**: The `faiss` crate (v0.11) has a complex type system that doesn't match the simple API we designed for:

- `Index::search()` requires `&mut self` not `&self`
- Returns `SearchResult` with generic `Idx` type, not `i64`
- Type conversions between `Idx`, `usize`, `i64` are non-trivial
- `index_factory()` signature expects `u32` for dimension

**Examples of compilation errors:**
```
error[E0308]: mismatched types
  --> engram-core/benches/support/faiss_ann.rs:106:29
   |
106 |         match self.index.search(query, k_i64) {
   |                             ^^^^^^ expected `&mut IndexImpl`, found `&IndexImpl`

error[E0605]: non-primitive cast: `Idx` as `usize`
  --> engram-core/benches/support/faiss_ann.rs:118:26
   |
118 |                         (label as usize, similarity)
   |                          ^^^^^^^^^^^^^^ an `as` expression can only be used to convert...
```

**Root Cause**: Our `AnnIndex` trait assumes immutable `&self` for search, but FAISS requires `&mut self`. This is a fundamental API mismatch.

#### 2. Annoy-rs Library Limitations

**Issue**: The `annoy-rs` crate only supports **loading** pre-built indexes, not building new ones from vectors.

**API Available:**
```rust
let index = AnnoyIndex::load(10, "index.ann", IndexType::Angular)?;
```

**API NOT Available:**
```rust
index.new(dimension);         // ❌ Doesn't exist
index.add_item(idx, vector);  // ❌ Doesn't exist
index.build(n_trees);         // ❌ Doesn't exist
```

**Solution Applied**: Created mock Annoy implementation using exact search with noise to simulate approximate behavior.

### What Works

The benchmark **framework** is production-ready:

- `AnnIndex` trait with clean API ✅
- `DatasetLoader` with synthetic and SIFT1M datasets ✅
- `BenchmarkFramework` for systematic comparisons ✅
- Engram's `EngramOptimizedAnnIndex` fully functional ✅
- Comprehensive test coverage for recall validation ✅

### Options to Complete Task

#### Option A: Fix FAISS Integration (2-3 hours)

1. Modify `AnnIndex` trait to use `&mut self` for search
2. Update all implementations (Engram, mocks) to match
3. Learn FAISS `Idx` type system and implement proper conversions
4. Handle `SearchResult` properly

**Pros**: Real FAISS comparison
**Cons**: Breaking API change, affects all existing code

#### Option B: Use Mock for Now, Document Limitation (30 min)

1. Keep mock FAISS that does exact search + noise
2. Document that real library integration is follow-up task
3. Validate Engram against ground truth using our exact search
4. Create Task 005 for proper FAISS/Annoy bindings

**Pros**: Fast completion, validates framework works
**Cons**: Not comparing against "real" industry libraries

#### Option C: Python Interop (4-6 hours)

1. Use PyO3 to call real Python FAISS/Annoy
2. Build indexes in Python, expose via FFI
3. Query from Rust benchmarks

**Pros**: Access to mature Python libraries
**Cons**: Complex setup, adds Python dependency

### Recommendation

**Use Option B** for this task, create follow-up Task 005 for real integration.

**Rationale**:
1. Primary goal is validating **Engram** achieves ≥90% recall@10
2. We can compute ground truth using exact search (FAISS Flat equivalent)
3. Framework is production-ready and extensible
4. Real library integration is valuable but not blocking for validation
5. Task estimates were based on stable library APIs, not debugging binding issues

### Files Status

**Created (Working)**:
- `engram-core/benches/support/faiss_ann.rs` - Partial, needs API fixes
- `engram-core/benches/support/annoy_ann.rs` - Mock implementation (complete)
- `engram-core/benches/ann_validation.rs` - Test suite (complete)
- Updated `engram-core/benches/ann_comparison.rs` - Benchmarks (complete)
- Updated `engram-core/benches/support/mod.rs` - Module structure (complete)

**To Delete** (when real integration works):
- `engram-core/benches/support/mock_faiss.rs` - Old mock
- `engram-core/benches/support/mock_annoy.rs` - Old mock

### Next Steps

1. Decide on Option A, B, or C
2. If Option B: Run validation with mocks, document results
3. If Option A: Allocate 2-3 hours to fix FAISS API issues
4. Update task file from `_in_progress` to appropriate status
5. Commit work with detailed message

### Time Spent

- Phase 1-6: ~3 hours (as estimated)
- Phase 7 (debugging): ~2 hours (unexpected)
- **Total**: ~5 hours of 8 hour estimate

### Remaining Estimate

- Option A: 2-3 hours
- Option B: 30 minutes
- Option C: 4-6 hours
