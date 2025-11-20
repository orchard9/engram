# Milestone 9 Research Insights Extraction

## Key Research Findings by Task

### 001_parser_infrastructure (Zero-Copy Parsing)

**Performance Targets:**
- Tokenizer must fit in 64 bytes (single cache line)
- Token enum: 24 bytes
- Spanned<Token>: 72 bytes
- Tokenize 1000-char query in <10μs
- Tokenize 50-char query in <500ns
- Keyword lookup: O(1), <5ns via PHF
- Zero allocations on hot path (identifiers, keywords, numbers)

**Implementation Constraints:**
- Lifetime parameters ('a) non-negotiable for zero-copy
- Use CharIndices for UTF-8 correctness
- PHF (perfect hash function) for keyword recognition
- Inline hot paths (#[inline])
- Branch predictor optimization (order match arms by frequency)

**Validation Criteria:**
- Pointer comparison test for zero-copy verification
- Custom allocator instrumentation to verify zero allocations
- Criterion benchmarks for latency targets
- Compile-time assertions for struct sizes

**Academic References:**
- "Fast and Space Efficient Trie Searches" - Askitis & Sinha (2007)
- "Parsing Gigabytes of JSON per Second" - Langdale & Lemire (2019)
- "Minimal Perfect Hash Functions" - Czech, Havas, Majewski (1992)
- "What Every Programmer Should Know About Memory" - Ulrich Drepper

### 010_performance_optimization (Sub-100μs Parse Times)

**Profiling Results (Bottlenecks Identified):**
- String allocations: 40% of time → FIX: zero-copy slices
- Token matching: 30% of time → FIX: PHF keyword lookup
- AST allocation: 20% of time → FIX: arena allocation
- Error construction: 10% of time → FIX: lazy error construction

**Performance Progression:**
- Baseline: 450μs
- After zero-copy: 280μs (40% improvement)
- After arena allocation: 168μs (30% improvement)
- After PHF keywords: 118μs (15% improvement)
- After inline hot paths: 100μs (10% improvement)
- Final: 90μs (80% improvement total)

**Specific Targets Achieved:**
- Simple RECALL: <50μs (actual: 45μs)
- Complex multi-constraint: <100μs (actual: 90μs)
- Large embedding (1536 floats): <200μs (actual: 180μs)

**Tooling:**
- Criterion for regression testing (fail CI if >10% regression)
- Flamegraph profiling to identify hot spots
- Black-box optimization prevention

**References:**
- "Performance Matters" - Emery Berger (UMass)
- "The Rust Performance Book" - Nicholas Matsakis
- Criterion.rs documentation
- Flamegraph profiling - Brendan Gregg

## Enhancements to Apply

### Task 001_parser_infrastructure_pending.md
ADD:
- Explicit performance regression thresholds: fail CI if >10% slower
- Flamegraph profiling requirement in acceptance criteria
- Reference to academic papers in implementation notes
- Custom allocator test requirement for zero-allocation verification
- Compile-time size assertions with const_assert!

### Task 010_performance_optimization_pending.md
CURRENTLY: Minimal spec (735 bytes)
ENHANCE WITH:
- Profiling methodology (perf, flamegraph, Instruments)
- Specific bottleneck targets from research:
  * String allocations must be <5% of total time
  * Token matching must be <20% of total time
  * AST allocation via arena must be <10% of total time
- Performance progression milestones
- Regression detection setup (Criterion + CI)
- Success criteria: 90μs for complex queries, 45μs for simple
- Reference to "Performance Matters" talk
- Flamegraph visualization requirement

