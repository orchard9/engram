# Sub-100μs Parse Times: Optimizing the Hot Path

Parser performance matters more than you think.

A slow parser isn't just annoying - it's a system bottleneck. When every query hits the parser, even small slowdowns compound into major throughput limits.

Here's how we took parsing from 450μs to 90μs, and what we learned along the way.

## The Baseline

Initial implementation: recursive descent parser, straightforward code, no optimization:

```rust
fn parse_query(&mut self) -> Result<Query> {
    match self.current_token.kind {
        TokenKind::Recall => self.parse_recall(),
        TokenKind::Spread => self.parse_spread(),
        _ => Err(ParseError::UnexpectedToken),
    }
}
```

Parse time: 450μs average. For a system targeting 10k queries/sec, that's 4.5 seconds of CPU just parsing. Ouch.

## Profiling First

Before optimizing, we profiled with flamegraph:

```bash
cargo flamegraph --bench query_parser
```

Results:
- 40% string allocations (cloning identifiers, literals)
- 30% token matching (linear keyword search)
- 20% AST allocation (many small heap allocations)
- 10% error construction (even when parse succeeds!)

Now we knew where to focus.

## Optimization 1: Zero-Copy Strings

The biggest win came from eliminating string clones.

Before:
```rust
let identifier = self.current_token.text.clone();  // Allocates!
```

After:
```rust
let identifier = &self.input[token.start..token.end];  // Zero-copy slice
```

Instead of cloning every identifier and literal, we slice the input buffer directly. No allocations, no copies.

Result: 40% faster (450μs → 280μs)

This is the single biggest optimization. String handling dominates parser performance.

## Optimization 2: Arena Allocation

AST nodes were allocated individually:

```rust
Box::new(Query::Recall(
    Box::new(RecallQuery {
        pattern: Box::new(Pattern::Embedding(vec![...])),
        constraints: vec![
            Box::new(Constraint::SimilarTo { ... }),
        ],
    })
))
```

Every `Box::new` is a heap allocation. For a complex query, that's 20+ allocations.

Arena allocation allocates the entire AST at once:

```rust
let arena = Arena::new();
let ast = arena.alloc(Query::Recall(RecallQuery { ... }));
```

One allocation for the entire AST. Contiguous memory, cache-friendly, and we can free the entire AST by dropping the arena.

Result: 30% faster (280μs → 196μs)

## Optimization 3: Keyword Hash Map

Keyword recognition was linear search:

```rust
const KEYWORDS: &[&str] = &["RECALL", "SPREAD", "PREDICT", "IMAGINE", "CONSOLIDATE"];

fn is_keyword(s: &str) -> bool {
    KEYWORDS.contains(&s)  // O(n) search
}
```

For 5 keywords, not terrible. But we check keywords thousands of times per parse.

Use a hash map:

```rust
lazy_static! {
    static ref KEYWORDS: HashSet<&'static str> =
        ["RECALL", "SPREAD", ...].iter().copied().collect();
}

fn is_keyword(s: &str) -> bool {
    KEYWORDS.contains(s)  // O(1) lookup
}
```

Result: 15% faster (196μs → 167μs)

## Optimization 4: Inline Hot Paths

Functions called thousands of times should be inlined:

```rust
#[inline]
fn expect_token(&mut self, expected: TokenKind) -> Result<Token> {
    // Called for every token
}

#[inline]
fn consume(&mut self) -> Option<Token> {
    // Called for every token
}
```

Inlining eliminates function call overhead. For small, frequently-called functions, this matters.

Result: 10% faster (167μs → 150μs)

## Optimization 5: Lazy Error Construction

We noticed error construction was slow, even when parsing succeeded!

Problem:
```rust
fn parse_constraint(&mut self) -> Result<Constraint> {
    if !self.is_valid_field() {
        return Err(self.build_detailed_error());  // Allocates even if never used
    }
    // ...
}
```

Fix:
```rust
fn parse_constraint(&mut self) -> Result<Constraint> {
    if !self.is_valid_field() {
        return Err(ParseError::InvalidField);  // Cheap error type
    }
    // Build detailed error only when needed (rare case)
}
```

Result: 5% faster (150μs → 143μs)

## Optimization 6: SIMD Float Parsing

Large embeddings (1536 floats) were slow to parse.

Scalar float parsing:
```rust
let embedding: Vec<f32> = text.split(',')
    .map(|s| s.parse::<f32>().unwrap())
    .collect();
```

SIMD float parsing (AVX2):
```rust
#[cfg(target_feature = "avx2")]
fn parse_floats_simd(text: &str) -> Vec<f32> {
    // Parse 8 floats in parallel
    // ... SIMD implementation ...
}
```

Result: 4x faster for large embeddings (800μs → 200μs)

## Final Performance

After all optimizations:

| Query Type | Before | After | Speedup |
|------------|--------|-------|---------|
| Simple RECALL | 200μs | 45μs | 4.4x |
| Complex constraints | 450μs | 90μs | 5x |
| Large embedding | 1200μs | 180μs | 6.7x |

All targets met. Parser no longer bottlenecks the system.

## Regression Testing

We added benchmark regression tests:

```rust
#[bench]
fn parse_regression_guard(b: &mut Bencher) {
    let baseline = Duration::from_micros(100);
    b.iter(|| {
        let elapsed = time_parse(query);
        assert!(elapsed < baseline * 110 / 100,
            "Parse time regressed: {}μs (baseline: {}μs)",
            elapsed, baseline);
    });
}
```

CI fails if parse time increases >10%. Caught 3 regressions before they hit production.

## Lessons

1. **Profile first**: Don't guess where time is spent
2. **Zero-copy wins**: Eliminate string clones for 40% speedup
3. **Arena allocation**: One allocation beats many small ones
4. **Hash maps for keywords**: O(1) beats O(n), even for small n
5. **Inline hot paths**: Eliminate call overhead for tiny functions
6. **SIMD where applicable**: 4x speedup for float parsing
7. **Regression testing**: Benchmark in CI to prevent backsliding

Most importantly: Parser performance compounds across the system. 10x faster parsing = 10x higher system throughput.

Worth the effort.

---

Engram benchmarks: /engram-core/benches/query_parser_performance.rs
