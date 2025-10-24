# Semantic Pattern Retrieval: Architectural Perspectives

## Cognitive Architecture: Schema-Driven Reconstruction

Bartlett's schema theory: Memory reconstruction blends episodic fragments with schema-based expectations. You don't recall breakfast verbatim - you reconstruct using "breakfast schema" filled with likely details.

Task 003 implements this by retrieving consolidated semantic patterns. These patterns are learned schemas extracted from repeated episodes.

**Key Difference from Traditional Search:**
- Traditional: Find exact or approximate match
- Schema-based: Retrieve general pattern, instantiate with specific details

Example: Query "breakfast yesterday"
- Traditional match: Find episode with exact date
- Schema retrieval: Get "typical breakfast" pattern, combine with temporal context

Result: Robust completion even when exact episode missing.

## Memory Systems: Complementary Learning Systems

Norman & O'Reilly CLS: Brain maintains two memory systems:
- Fast system (hippocampus): Episodic pattern completion
- Slow system (neocortex): Semantic knowledge consolidation

Task 001-002: Fast system (local temporal reconstruction)
Task 003: Slow system (global semantic patterns)
Task 004: Integration (hierarchical evidence)

The slow system provides stability - statistical regularities that transcend individual episodes. The fast system provides specificity - unique details from particular events.

Retrieval blends both: semantic patterns constrain plausible completions, episodic details provide specifics.

## Systems Architecture: Caching for Hot Patterns

Production memory systems access patterns repeatedly. Without caching, every completion hits consolidation storage (expensive).

LRU cache design:
- Pre-computed pattern embeddings (no re-encoding)
- Hash-based lookup (O(1) retrieval)
- Bounded memory (<50MB for 1000 patterns)
- Hit rate monitoring (target >60%)

Cache key = hash(non_null_indices + temporal_context). Ensures same partial cue gets cached result.

Invalidation strategy: Version number on consolidation. When patterns update, increment version. Cache checks version on hit.

## Rust Performance: SIMD for Masked Similarity

Masked similarity challenge: Some dimensions null, can't use standard SIMD (requires full vectors).

Solution: Conditional SIMD with mask registers:
```rust
#[target_feature(enable = "avx512f")]
unsafe fn masked_similarity_simd(
    partial: &[Option<f32>],
    full: &[f32; 768],
) -> f32 {
    // Create mask from non-null dimensions
    let mask = partial.iter()
        .map(|o| o.is_some() as u64)
        .collect::<Vec<_>>();

    // SIMD dot product with mask
    // Only non-null dimensions contribute
    // ...
}
```

AVX-512 mask registers enable conditional operations. Process 16 floats at once, mask controls which contribute to result.

Result: Masked similarity ~2x slower than full similarity, but still 100μs for 768-dim vectors.
