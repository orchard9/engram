# Parser Performance Profiling Methodology

## Overview

This document describes the systematic approach to profiling and optimizing the Engram query parser to meet sub-100μs latency targets.

## Performance Targets

| Query Type | Target (P90) | Current | Status |
|------------|--------------|---------|---------|
| Simple RECALL | <50μs | ~377ns | ✅ PASS |
| Complex multi-constraint | <100μs | ~444ns | ✅ PASS |
| Large embedding (1536 floats) | <200μs | ~67µs | ✅ PASS |

## Profiling Tools

### 1. Flamegraph Generation

Generate CPU flamegraphs to identify hot spots:

```bash
# Install flamegraph tool
cargo install flamegraph

# Generate flamegraph for parser benchmarks
cd engram-core
cargo flamegraph --bench query_parser -- --bench

# Output: flamegraph.svg

```

### 2. Criterion Benchmarks

Statistical benchmarking with regression detection:

```bash
# Run all parser benchmarks
cargo bench --bench query_parser

# Save baseline for future comparison
cargo bench --bench query_parser -- --save-baseline optimized

# Compare against baseline
cargo bench --bench query_parser -- --baseline optimized

```

### 3. Performance Analysis (Linux)

Using `perf` for hardware counter analysis:

```bash
# Cache miss analysis
perf stat -e cache-misses,cache-references cargo bench --bench query_parser

# Branch prediction analysis
perf stat -e branch-misses,branch-instructions cargo bench --bench query_parser

# Detailed CPU profiling
perf record cargo bench --bench query_parser
perf report

```

### 4. macOS Instruments

For macOS development:

```bash
# Build with debug symbols
cargo build --release --bench query_parser

# Profile with Instruments
instruments -t "Time Profiler" target/release/deps/query_parser-*

```

## Optimization Techniques Applied

### 1. Hot-Path Inlining

Critical parser functions marked with `#[inline]`:

```rust
#[inline]
fn expect_token(&mut self, expected: TokenKind) -> Result<Token>
#[inline]
fn advance(&mut self) -> Option<Token>
#[inline]
fn peek(&self) -> Option<&Token>

```

### 2. Zero-Copy String Handling

Parser uses lifetime-based string slices to avoid allocations:

```rust
pub struct Token<'a> {
    pub text: &'a str,  // Zero-copy slice
    // ...
}

```

### 3. PHF Keyword Lookup

O(1) compile-time perfect hash for keywords (from Task 001):

```rust
static KEYWORDS: phf::Map<&'static str, TokenKind> = phf_map! {
    "RECALL" => TokenKind::Recall,
    "SPREAD" => TokenKind::Spread,
    // ...
};

```

### 4. Arena Allocation (Future Enhancement)

While bumpalo is available as a dependency, it's currently not integrated into the parser.
Future optimization could add arena allocation for AST nodes:

```rust
use bumpalo::Bump;

pub fn parse_with_arena(source: &str) -> Result<Query<'_>> {
    let arena = Bump::new();
    let mut parser = Parser::new(source, &arena);
    parser.parse_query()
}

```

Expected improvement: 30% reduction in allocation overhead.

## Profiling Workflow

1. **Establish Baseline**

   ```bash
   cargo bench --bench query_parser -- --save-baseline before
   ```

2. **Generate Flamegraph**

   ```bash
   cargo flamegraph --bench query_parser
   open flamegraph.svg
   ```

3. **Identify Hot Spots**
   - Look for wide bars (high CPU time)
   - Focus on parser-specific code (not stdlib)
   - Note allocation patterns

4. **Apply Optimization**
   - Make targeted changes
   - Prefer algorithmic improvements over micro-optimizations

5. **Measure Impact**

   ```bash
   cargo bench --bench query_parser -- --baseline before
   ```

6. **Verify Correctness**

   ```bash
   cargo test --lib query::parser
   ```

## Common Bottlenecks and Solutions

| Bottleneck | Typical Impact | Solution |
|------------|----------------|----------|
| String allocations | 40% of time | Zero-copy slices |
| Token matching | 30% of time | PHF lookup |
| AST node allocation | 20% of time | Arena allocator |
| Error construction | 10% of time | Lazy construction |

## CI Integration

Performance regression detection is automated via GitHub Actions:

- Benchmarks run on every PR affecting parser code

- Compares against main branch baseline

- Fails if regression >10%

- Posts benchmark results as PR comment

See `.github/workflows/performance.yml` for configuration.

## References

- "Performance Matters" - Emery Berger, UMass Amherst

- "The Rust Performance Book" - Nicholas Matsakis

- "Systems Performance" - Brendan Gregg

- Criterion.rs documentation: https://bheisler.github.io/criterion.rs/
