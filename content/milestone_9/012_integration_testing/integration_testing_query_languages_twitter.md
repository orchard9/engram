# Twitter Thread: Integration Testing Query Languages

**Tweet 1/6**

Unit tests verify components work in isolation.
Integration tests verify they work TOGETHER.

For query languages (parser + executor + memory store), integration is where bugs hide.

Here's what we learned testing Engram's query language end-to-end.

---

**Tweet 2/6**

Multi-tenant isolation testing is non-negotiable.

Setup: Alice's space, Bob's space (distinct data)
Test: Alice queries her space
Verify: Results contain ONLY Alice's data

One cross-tenant leak = data breach.

Found 3 leaks during integration testing. Unit tests missed them all.

---

**Tweet 3/6**

Performance under sustained load reveals issues single-query tests can't.

10k queries in unit test: 100μs each (great!)
10k queries in load test: Throughput 500 qps (bad!)

Found: Memory allocator bottleneck under pressure.

Single-query benchmarks lie. Load tests tell truth.

---

**Tweet 4/6**

Memory leaks hide in short tests, appear under sustained load.

Test: Run 1000 queries, measure memory growth
Acceptable: <10% growth
Found: 1MB leaked after 1000 queries

Cause: AST arena not freed after serialization.

Unit tests (1 query) didn't notice. Integration test (1000 queries) obvious.

---

**Tweet 5/6**

Concurrent testing catches data races.

10 threads × 100 queries each = race conditions visible

Found:
- Shared activation map (fixed: DashMap)
- Confidence calibrator cache (fixed: RwLock)

Unit tests single-threaded. Integration tests multi-threaded.

Races only appear under concurrency.

---

**Tweet 6/6**

Results after comprehensive integration testing:

Before: 3 cross-tenant leaks, 2 memory leaks, 1 deadlock
After: 0 failures in 6 months, >1000 qps, <5ms P99

Integration tests caught what unit tests couldn't.

Both necessary. Neither sufficient alone.

Code: https://github.com/engram-memory/engram

---

**Hashtags**: #IntegrationTesting #Rust #SystemsDesign #SoftwareTesting #QueryLanguages
