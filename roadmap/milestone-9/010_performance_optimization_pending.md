# Task 010: Parser Performance Optimization

**Status**: Pending
**Duration**: 1 day
**Dependencies**: Task 003, 008
**Owner**: TBD

---

## Objective

Optimize parser to meet <100μs parse time target through profiling, inlining hot paths, and benchmark regression tests.

---

## Techniques

1. Criterion benchmarks for all query types
2. Flamegraph profiling to identify hot spots
3. Inline critical path functions (#[inline])
4. Arena allocation for AST nodes
5. Benchmark regression tests in CI

---

## Files

`engram-core/benches/query_parser.rs`

---

## Acceptance Criteria

- [ ] Parse time <100μs P90
- [ ] Parse time <200μs P99
- [ ] CI fails if parse time regresses >10%
- [ ] Flamegraph shows no unexpected allocations
