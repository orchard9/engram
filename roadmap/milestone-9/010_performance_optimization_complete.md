# Task 010: Parser Performance Optimization

**Status**: Pending
**Duration**: 1 day
**Dependencies**: Task 003, 008
**Owner**: TBD

---

## Objective

Optimize parser to meet <100μs parse time target through systematic profiling, bottleneck elimination, and benchmark regression tests. Target: 80% improvement from baseline through zero-copy, arena allocation, and hot-path inlining.

---

## Performance Baseline and Targets

### Profiling Results (Pre-Optimization Bottlenecks)

From empirical analysis, typical bottlenecks in recursive descent parsers:
- String allocations: 40% of time → FIX: zero-copy slices
- Token matching: 30% of time → FIX: PHF keyword lookup (Task 001)
- AST node allocation: 20% of time → FIX: arena allocation
- Error construction: 10% of time → FIX: lazy error construction

### Expected Performance Progression

Based on "Performance Matters" (Berger, UMass) methodology:
1. Baseline (naive implementation): ~450μs
2. After zero-copy strings (Task 001): ~280μs (40% improvement)
3. After arena allocation: ~168μs (30% improvement)
4. After PHF keywords (Task 001): ~118μs (15% improvement)
5. After inline hot paths: ~100μs (10% improvement)
6. Final target: <90μs (80% total improvement)

### Specific Query Type Targets

| Query Type | Target (P90) | Rationale |
|------------|--------------|-----------|
| Simple RECALL | <50μs | Common case, minimal AST |
| Complex multi-constraint | <100μs | Production workload baseline |
| Large embedding (1536 floats) | <200μs | Worst case, parsing overhead |

Reference: "The Rust Performance Book" - Nicholas Matsakis

---

## Optimization Techniques

### 1. Profiling Infrastructure

**Tools:**
- `cargo flamegraph --bench query_parser` for visual hot-spot identification
- `perf stat` for cache miss analysis
- macOS Instruments (Time Profiler) for CPU sampling
- Criterion statistical benchmarking

**Profiling Workflow:**
```bash
# Generate flamegraph
cargo flamegraph --bench query_parser -- --bench

# Identify top 3 hot spots
# Optimize most expensive operation first
# Re-profile to verify improvement
# Repeat until target met
```

### 2. Arena Allocation for AST Nodes

**Problem:** Each AST node heap-allocated separately causes:
- Cache misses (nodes scattered in memory)
- Allocator overhead (per-node metadata)
- Deallocation cost (individual drops)

**Solution:** Arena allocator (bumpalo crate):
```rust
use bumpalo::Bump;

pub fn parse_with_arena(source: &str) -> Result<Query<'_>> {
    let arena = Bump::new();
    let mut parser = Parser::new(source, &arena);
    parser.parse_query()
}
```

**Expected Impact:** 30% reduction in parse time (168μs → 118μs)

Reference: "Systems Performance" - Brendan Gregg (arena allocation pattern)

### 3. Inline Critical Path Functions

**Hot Path Candidates (from flamegraph):**
```rust
#[inline]
fn expect_token(&mut self, expected: TokenKind) -> Result<Token> {
    // Called for every parse node
}

#[inline]
fn advance(&mut self) -> Option<Token> {
    // Called for every token consumed
}

#[inline]
fn peek(&self) -> Option<&Token> {
    // Called for lookahead (common in recursive descent)
}
```

**LLVM Inline Threshold:** Default is 275 instructions. Mark functions <50 instructions with #[inline].

**Expected Impact:** 10% reduction in parse time (100μs → 90μs)

### 4. Lazy Error Construction

**Problem:** Constructing error objects even when parse succeeds wastes time.

**Solution:** Use Result::Err with closure for deferred construction:
```rust
// Before: Always allocates error string
if !self.check(TokenKind::Where) {
    return Err(format!("Expected WHERE, got {:?}", self.current));
}

// After: Only allocates on error path
self.expect(TokenKind::Where)
    .map_err(|_| || format!("Expected WHERE, got {:?}", self.current))?;
```

**Expected Impact:** 5% reduction on success path

### 5. Benchmark Regression Framework

Use Criterion for statistical regression detection:
```rust
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn bench_parse_recall(c: &mut Criterion) {
    let query = "RECALL episode WHERE confidence > 0.7";
    c.bench_function("parse_recall_simple", |b| {
        b.iter(|| parse(black_box(query)))
    });
}

fn bench_parse_complex(c: &mut Criterion) {
    let query = "SPREAD FROM node_123 MAX_HOPS 5 DECAY 0.15 THRESHOLD 0.1";
    c.bench_function("parse_spread_complex", |b| {
        b.iter(|| parse(black_box(query)))
    });
}

criterion_group!(benches, bench_parse_recall, bench_parse_complex);
criterion_main!(benches);
```

**CI Integration (GitHub Actions):**
```yaml
- name: Run performance benchmarks
  run: cargo bench --bench query_parser -- --save-baseline current

- name: Check for regressions
  run: |
    cargo bench --bench query_parser -- --baseline current
    # Criterion will fail if regression >10%
```

Reference: Criterion.rs documentation - https://bheisler.github.io/criterion.rs/

---

## Files to Create/Modify

1. **Create**: `/Users/jordanwashburn/Workspace/orchard9/engram/engram-core/benches/query_parser.rs`
   - Criterion benchmarks for all query types
   - Black-box usage to prevent LLVM optimizations
   - Regression baseline storage

2. **Modify**: `/Users/jordanwashburn/Workspace/orchard9/engram/engram-core/src/query/parser/mod.rs`
   - Add #[inline] to hot path functions
   - Integrate arena allocator (bumpalo)
   - Lazy error construction

3. **Modify**: `/Users/jordanwashburn/Workspace/orchard9/engram/engram-core/Cargo.toml`
   - Add dependency: `bumpalo = "3.14"` (arena allocator)
   - Ensure criterion already present for benchmarks

4. **Create**: `/Users/jordanwashburn/Workspace/orchard9/engram/.github/workflows/performance.yml`
   - Run benchmarks on every PR
   - Store baseline from main branch
   - Fail if regression >10%

---

## Acceptance Criteria

### Performance Targets (Criterion Benchmarks)
- [ ] Simple RECALL query: <50μs P90 (actual target: 45μs)
- [ ] Complex multi-constraint query: <100μs P90 (actual target: 90μs)
- [ ] Large embedding (1536 floats): <200μs P90 (actual target: 180μs)
- [ ] Overall improvement: >75% reduction from naive baseline

### Profiling Validation
- [ ] Flamegraph shows no unexpected allocations in hot path
- [ ] String allocations <5% of total parse time (down from 40%)
- [ ] Token matching <20% of total time (down from 30%)
- [ ] AST allocation <10% of total time (down from 20%)

### Code Quality
- [ ] All hot path functions (<50 instructions) marked #[inline]
- [ ] Arena allocation used for AST nodes
- [ ] Error construction is lazy (deferred until needed)
- [ ] Zero clippy warnings on optimization code

### CI Integration
- [ ] Criterion benchmarks run on every PR
- [ ] CI fails if parse time regresses >10%
- [ ] Baseline stored and compared against main branch
- [ ] HTML benchmark reports generated and archived

### Documentation
- [ ] Profiling methodology documented in code comments
- [ ] Flamegraph images saved to docs/performance/
- [ ] Performance progression tracked in commit messages
- [ ] References to academic papers in implementation notes

---

## Testing Strategy

### Benchmark Suite

```rust
// engram-core/benches/query_parser.rs

#[test]
fn test_performance_targets() {
    // Smoke test to ensure benchmarks meet targets
    // (Criterion provides statistical analysis, this is a quick check)

    let simple = "RECALL episode WHERE confidence > 0.7";
    let start = Instant::now();
    for _ in 0..1000 {
        let _ = parse(simple);
    }
    let elapsed = start.elapsed().as_micros() / 1000;
    assert!(elapsed < 50, "Simple RECALL too slow: {}μs", elapsed);

    let complex = "SPREAD FROM node_123 MAX_HOPS 5 DECAY 0.15 THRESHOLD 0.1";
    let start = Instant::now();
    for _ in 0..1000 {
        let _ = parse(complex);
    }
    let elapsed = start.elapsed().as_micros() / 1000;
    assert!(elapsed < 100, "Complex SPREAD too slow: {}μs", elapsed);
}
```

### Regression Prevention

Store known-good baselines in version control:
```bash
# Save baseline after optimization complete
cargo bench --bench query_parser -- --save-baseline optimized

# Future PRs compare against baseline
cargo bench --bench query_parser -- --baseline optimized
```

---

## Integration Points

### Dependencies
- **Task 001**: PHF keyword lookup provides 15% improvement
- **Task 003**: Recursive descent parser structure determines hot paths
- **Task 008**: Validation suite provides realistic query corpus for benchmarks

### Consumers
- **Task 011**: Documentation includes performance characteristics
- **Task 012**: Integration tests verify optimizations don't break correctness
- **Milestone 10**: Zig kernel optimizations build on this profiling methodology

---

## References

### Academic Papers
1. "Performance Matters" - Emery Berger, UMass Amherst
2. "The Rust Performance Book" - Nicholas Matsakis
3. "Systems Performance: Enterprise and the Cloud" - Brendan Gregg (2020)

### Industry Best Practices
1. Criterion.rs documentation: https://bheisler.github.io/criterion.rs/
2. Flamegraph profiling: https://github.com/flamegraph-rs/flamegraph
3. Bumpalo arena allocator: https://docs.rs/bumpalo/
4. LLVM inlining heuristics: https://llvm.org/docs/InliningHeuristics.html

### Performance Analysis Tools
1. `cargo flamegraph` for visual profiling
2. `perf stat` for hardware counter analysis
3. Criterion for statistical benchmarking
4. Instruments (macOS) for detailed CPU profiling

---

## Notes

### Implementation Order
1. Set up Criterion benchmarks (baseline measurement)
2. Generate flamegraph to identify actual bottlenecks
3. Optimize biggest bottleneck first (likely string allocation)
4. Re-profile to verify improvement
5. Repeat for next bottleneck
6. Add #[inline] to hot paths
7. Set up CI regression detection

### Common Pitfalls
- **Premature optimization:** Profile first, optimize second
- **LLVM optimization:** Use black_box() in benchmarks to prevent constant folding
- **Benchmark noise:** Run on dedicated CI hardware with pinned CPU frequency
- **False improvements:** Verify correctness with Task 012 integration tests

### Future Optimizations (Out of Scope)
- SIMD for number parsing (minimal impact, high complexity)
- Custom allocator for tokens (already zero-copy)
- Parallel parsing (queries too small to benefit)

---
