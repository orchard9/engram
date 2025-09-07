# Task 001: SIMD Vector Operations

## Status: Pending
## Priority: P0 - Critical Path
## Estimated Effort: 8 days
## Dependencies: None

## Objective
Implement SIMD-optimized vector operations with runtime CPU feature detection and scalar fallbacks, achieving 10x performance improvement for embedding operations.

## Technical Specification

### Core Requirements
1. **Runtime CPU Feature Detection**
   - Detect AVX-512, AVX2, SSE4.2 on x86_64
   - Detect NEON on ARM64
   - Graceful fallback to scalar operations
   - Zero-cost abstraction via trait dispatch

2. **Vector Operations to Implement**
   - Cosine similarity (768-dimensional embeddings)
   - Dot product
   - L2 norm
   - Element-wise addition/subtraction
   - Scaled vector operations

3. **Correctness Guarantees**
   - Bit-identical results between SIMD and scalar paths
   - Deterministic computation regardless of hardware
   - Proper handling of edge cases (NaN, infinity, denormals)

### Implementation Details

**Files to Create:**
- `engram-core/src/compute/mod.rs` - Module interface and trait definitions
- `engram-core/src/compute/simd.rs` - SIMD implementations
- `engram-core/src/compute/scalar.rs` - Scalar fallback implementations
- `engram-core/src/compute/detect.rs` - CPU feature detection

**Files to Modify:**
- `engram-core/src/store.rs` - Update cosine_similarity to use new compute module
- `engram-core/src/lib.rs` - Export compute module
- `engram-core/Cargo.toml` - Add dependencies: `packed_simd_2`, `cpufeatures`

### Testing Strategy
1. **Differential Testing**
   - Property-based tests comparing SIMD vs scalar
   - Fuzzing with random vectors
   - Edge case validation

2. **Performance Benchmarks**
   - Micro-benchmarks for each operation
   - Various vector sizes (768, 1536, 3072)
   - Performance regression detection

3. **Hardware Matrix Testing**
   - Test on x86_64 with various SIMD levels
   - Test on ARM64 with NEON
   - Test scalar fallback explicitly

## Acceptance Criteria
- [ ] All vector operations produce bit-identical results across implementations
- [ ] 10x performance improvement on AVX2-capable hardware
- [ ] Graceful degradation on unsupported hardware
- [ ] Zero overhead when using scalar fallback
- [ ] Comprehensive test coverage with property-based tests
- [ ] Benchmarks show linear scaling with vector length
- [ ] Documentation includes performance characteristics per operation

## Integration Notes
- This forms the foundation for all similarity computations
- HNSW index (Task 002) will depend on these operations
- Activation spreading (Task 004) will use these for weight calculations
- Must maintain API compatibility with existing cosine_similarity function

## Risk Mitigation
- Start with scalar implementation to establish correctness baseline
- Add SIMD incrementally, one instruction set at a time
- Extensive differential testing before enabling by default
- Feature flag to force scalar mode if issues arise