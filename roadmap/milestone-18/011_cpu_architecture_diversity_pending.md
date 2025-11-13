# Task 011: CPU Architecture Diversity

**Status**: Pending
**Estimated Duration**: 3-4 days
**Priority**: Medium - Multi-platform validation

## Objective

Validate performance on ARM (Apple Silicon), x86 (Intel/AMD), and verify SIMD fallback correctness. Ensure <10% performance variance across architectures for same workload.

## Test Matrix

| Architecture | CPU Model | SIMD Support | Target P99 |
|--------------|-----------|--------------|------------|
| ARM64 | Apple M1/M2 | NEON | <1ms |
| x86-64 | Intel 12th+ | AVX-512 | <1ms |
| x86-64 | AMD Zen 3+ | AVX2 | <1.2ms |
| ARM64 | AWS Graviton | NEON | <1.5ms |

## SIMD Validation

```rust
#[test]
fn simd_fallback_equivalence() {
    let v1 = vec![1.0f32; 768];
    let v2 = vec![2.0f32; 768];

    // Force scalar implementation
    let scalar_result = cosine_similarity_scalar(&v1, &v2);

    // Use SIMD (auto-detected)
    let simd_result = cosine_similarity(&v1, &v2);

    assert!((scalar_result - simd_result).abs() < 1e-6, "SIMD mismatch");
}
```

## Success Criteria

- **Correctness**: SIMD and scalar implementations produce identical results (ε < 1e-6)
- **Performance Parity**: <10% variance across ARM/x86 for same core count
- **Auto-Detection**: SIMD features detected at runtime, no manual flags
- **CI Coverage**: Tests run on both ARM (macOS) and x86 (Linux) in CI

## Files

- `engram-core/tests/simd_equivalence_tests.rs` (320 lines)
- `scripts/cross_platform_benchmark.sh` (150 lines)
- `.github/workflows/cross_platform_tests.yml` (if using CI)
