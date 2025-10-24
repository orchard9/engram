# Integration Testing - Multiple Perspectives

## Systems Architecture Perspective

Unit tests verify components work. Integration tests verify the system works.

For Zig kernels, unit tests prove: "cosine similarity calculation matches Rust to 1e-6 epsilon."

Integration tests prove: "Memory retrieval with Zig kernels produces correct ranked results in <2ms."

The difference matters because real systems have layers:
- Rust graph queries Zig kernels via FFI
- Memory gets copied across language boundaries
- Results flow back through multiple abstractions
- Cache effects differ between isolated and composed operations

Integration tests catch the bugs that live in the gaps.

## Testing and Validation Perspective

Differential testing validates correctness at the algorithm level. Integration testing validates correctness at the system level.

Example failure caught by integration testing:
- Unit test: Zig spreading activation matches Rust spreading activation
- Integration test: Zig spreading + Zig decay produces different results than Rust

Why? Zig decay rounded strengths to f32, then Zig spreading used rounded values. Rust kept full precision throughout. Accumulated rounding differed.

Unit tests passed (each kernel correct in isolation). Integration test failed (composition incorrect).

## Memory Systems Perspective

Cognitive operations are pipelines, not isolated steps. Memory consolidation combines:
1. Encoding (vector similarity)
2. Association (graph edge creation)
3. Retrieval (spreading activation)
4. Forgetting (decay)

Testing them separately misses emergent behavior. Does weak activation from decayed memories reach retrieval threshold? Integration test answers.
