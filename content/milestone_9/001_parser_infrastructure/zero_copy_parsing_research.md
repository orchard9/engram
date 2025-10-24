# Research: Zero-Copy Parsing and Performance-Critical Tokenization

## Topic Overview

Zero-copy parsing is a technique where tokenizers reference the original source string directly instead of allocating copies for each token. For a cognitive query language parser targeting sub-100 microsecond latency, this approach is essential.

## Key Research Areas

### 1. Zero-Copy String Handling in Rust

**Lifetime-Based Borrowing**

Rust's lifetime system makes zero-copy parsing safe at compile time. By parameterizing tokens with a lifetime `'a`, we guarantee that tokens cannot outlive the source string:

```rust
enum Token<'a> {
    Identifier(&'a str),  // Borrows from source
    Keyword,              // No data to copy
}
```

**Reference**: Rust Book Chapter 10 on Lifetimes - https://doc.rust-lang.org/book/ch10-03-lifetime-syntax.html

**Cow (Clone-on-Write) Pattern**

`Cow<'a, str>` provides flexibility: borrowed when possible, owned when necessary. This is ideal for string literals that may contain escape sequences:

```rust
// Zero-copy for simple strings
let borrowed = Cow::Borrowed("simple_id");

// Allocation only when needed
let escaped = Cow::Owned(process_escapes("with\\nnewline"));
```

**Reference**: std::borrow::Cow documentation - https://doc.rust-lang.org/std/borrow/enum.Cow.html

### 2. Cache-Optimal Data Structures

**Cache Line Awareness**

Modern CPUs fetch memory in 64-byte cache lines. A tokenizer that fits in a single cache line avoids expensive L1 cache misses:

- x86-64: 64-byte cache lines (L1 typically 32KB)
- ARM: 64-byte cache lines (L1 typically 64KB)
- Apple Silicon: 128-byte cache lines (L1 96KB)

Target tokenizer size: 64 bytes or less

**Memory Layout Optimization**

Enum discriminants add overhead. For `Token<'a>`:
- Discriminant: 1 byte (for <256 variants)
- Alignment padding: 7 bytes (to 8-byte boundary)
- Payload: 16 bytes (for &str: pointer + length)
- Total: 24 bytes per token

**Reference**: "What Every Programmer Should Know About Memory" by Ulrich Drepper
https://people.freebsd.org/~lstewart/articles/cpumemory.pdf

### 3. Perfect Hash Functions for Keywords

**Compile-Time Hash Maps**

The `phf` crate generates perfect hash functions at compile time using the CHD algorithm:

- Zero runtime initialization cost
- O(1) lookup with zero collisions
- ~40 bytes per entry in read-only data
- Total size <1KB for ~20 keywords

**Performance Characteristics**:
- Lookup time: <5ns on modern CPUs
- No heap allocation
- Cache-friendly (fits in L1)

**Reference**: phf crate documentation - https://docs.rs/phf/latest/phf/

**CHD Algorithm**: Compress, Hash, and Displace method for minimal perfect hashing
- Paper: "An optimal algorithm for generating minimal perfect hash functions" (1992)

### 4. Position Tracking for Error Messages

**Dual Tracking Strategy**

Track both byte offsets and line/column numbers:

1. **Byte offsets**: O(1) string slicing for error context
2. **Line/column**: Human-readable error messages

```rust
struct Position {
    offset: usize,  // Byte index for slicing
    line: usize,    // 1-indexed for display
    column: usize,  // 1-indexed, counted in code points
}
```

**UTF-8 Handling**

Use `CharIndices` iterator for correct multi-byte handling:
- Iterates over (byte_index, char) pairs
- Correctly handles multi-byte UTF-8 sequences (emoji, Chinese, etc.)
- No manual UTF-8 validation needed

**Reference**: Rust String documentation - https://doc.rust-lang.org/std/string/struct.String.html#utf-8

### 5. Performance Benchmarking Techniques

**Criterion Framework**

Industry-standard Rust benchmarking:
- Statistical analysis of timing variance
- Outlier detection and removal
- HTML report generation with plots
- CI-friendly regression detection

**Black Box Optimization Prevention**

```rust
use criterion::black_box;

b.iter(|| {
    let result = tokenize(black_box(source));
    black_box(result);  // Prevent LLVM from optimizing away
});
```

**Reference**: Criterion.rs documentation - https://docs.rs/criterion/latest/criterion/

**Flamegraph Profiling**

Visualize where CPU time is spent:
- cargo-flamegraph integration
- Identifies hot paths for inlining
- Shows cache miss patterns

**Reference**: flamegraph crate - https://github.com/flamegraph-rs/flamegraph

### 6. Allocation Tracking and Elimination

**Custom Allocator Instrumentation**

```rust
use std::alloc::{GlobalAlloc, System, Layout};

struct CountingAllocator;

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCATION_COUNT.fetch_add(1, Ordering::SeqCst);
        System.alloc(layout)
    }
    // ...
}
```

**Zero-Allocation Verification**

Test that hot paths don't allocate:
```rust
#[test]
fn test_zero_allocations() {
    let count_before = allocation_count();
    tokenize("RECALL episode WHERE confidence > 0.7");
    let count_after = allocation_count();
    assert_eq!(count_before, count_after, "Unexpected allocations");
}
```

**Reference**: Custom allocators in Rust - https://doc.rust-lang.org/std/alloc/trait.GlobalAlloc.html

### 7. Inline Optimization and Branch Prediction

**Strategic Inlining**

Hot path functions should be inlined:
```rust
#[inline]
fn advance(&mut self) -> Option<char> {
    // Called for every character - must be inlined
}
```

**Branch Predictor Optimization**

Modern CPUs predict branches with ~95% accuracy when patterns are consistent:

- Order match arms by frequency (most common first)
- Avoid branches in hot loops when possible
- Use `likely`/`unlikely` intrinsics for asymmetric paths

**Reference**: "Computer Architecture: A Quantitative Approach" by Hennessy & Patterson

### 8. Comparison with Parser Generators

**nom**: Zero-copy combinator library
- Pros: Composable, zero-copy by default
- Cons: Complex error messages, steep learning curve
- Parse speed: ~200-500ns for simple queries

**pest**: PEG parser generator
- Pros: Declarative grammar, good errors
- Cons: Generated code harder to debug, added build step
- Parse speed: ~500-1000ns for simple queries

**lalrpop**: LR(1) parser generator
- Pros: Powerful error recovery, shift-reduce conflicts explicit
- Cons: Complex grammar syntax, slow compile times
- Parse speed: ~300-700ns for simple queries

**Hand-written recursive descent**:
- Pros: Direct control, easy debugging, inline optimizations
- Cons: More initial work, manual error handling
- Parse speed: <100ns for simple queries (when optimized)

**Reference**: Rust parser comparison - https://github.com/Geal/nom/blob/main/doc/compared_parsers.md

## Performance Targets from Literature

**Query Language Parsers in Production**:

1. **SQL parsers**: 10-50μs for simple queries (PostgreSQL, MySQL)
2. **GraphQL parsers**: 50-200μs for typical queries
3. **JSON parsers**: 5-20μs for small documents (simd-json)
4. **Protocol Buffers**: 1-5μs for small messages

**Cognitive Query Target**: <100μs for 90% of queries

This is achievable because:
- Simpler grammar than SQL (no complex JOIN logic)
- Smaller token vocabulary (~20 keywords vs. 100+ in SQL)
- Predictable structure (operation-first syntax)

## Key Insights for Implementation

1. **Lifetime parameters are non-negotiable** for zero-copy performance
2. **Tokenizer must fit in 64 bytes** to stay cache-resident
3. **PHF keyword lookup** is essentially free (<5ns)
4. **Byte offsets** enable O(1) error context extraction
5. **Inline hot paths** for sub-microsecond token iteration
6. **Test allocation behavior** as rigorously as correctness

## Open Questions

1. Should we support Unicode identifiers (beyond ASCII)?
   - Pro: Internationalization
   - Con: Normalization cost, security concerns (confusables)

2. Arena allocation for AST nodes?
   - Pro: Bulk deallocation, improved locality
   - Con: More complex lifetime management

3. SIMD for keyword matching?
   - Pro: 4-8x speedup possible for long identifier comparisons
   - Con: Complexity, portability concerns

## References

### Academic Papers
1. "Fast and Space Efficient Trie Searches" - Askitis & Sinha (2007)
2. "Parsing Gigabytes of JSON per Second" - Langdale & Lemire (2019)
3. "Minimal Perfect Hash Functions" - Czech, Havas, Majewski (1992)

### Industry Best Practices
1. Redis parser design: https://redis.io/topics/protocol
2. HTTP/1.1 parser (httparse): https://github.com/seanmonstar/httparse
3. Rust lexer design patterns: https://matklad.github.io/2020/04/13/simple-but-powerful-pratt-parsing.html

### Performance Guides
1. Rust Performance Book: https://nnethercote.github.io/perf-book/
2. "Systems Performance" by Brendan Gregg
3. Intel Optimization Manual: https://software.intel.com/content/www/us/en/develop/articles/intel-sdm.html
