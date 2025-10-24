# What Integration Tests Catch That Unit Tests Miss - Research

## Research Topics

### 1. Integration Testing vs Unit Testing

**Unit Tests:** Test individual components in isolation
- Fast execution (milliseconds)
- Easy to write and maintain
- Catch logic errors in individual functions
- Miss interaction bugs between components

**Integration Tests:** Test components working together
- Slower execution (seconds to minutes)
- More complex setup (databases, services, state)
- Catch interface mismatches, data flow bugs, emergent behavior
- Verify system-level properties

**Citations:**
- Fowler, M. (2014). "Testing Strategies in a Microservice Architecture"
- Humble, J., & Farley, D. (2010). "Continuous Delivery: Reliable Software Releases"

### 2. End-to-End Memory Consolidation Workflow

Integration tests for Engram must validate complete cognitive pipelines:

1. **Add episodic memories** → Verify graph accepts new memories
2. **Compute similarity** → Zig vector kernel finds associations
3. **Build edges** → Graph structure updated
4. **Spreading activation** → Zig spreading kernel traverses graph
5. **Apply decay** → Zig decay kernel weakens old memories
6. **Consolidation** → Weak memories archived or forgotten

**What Unit Tests Miss:**
- Do Zig kernels correctly integrate with Rust graph structures?
- Does spreading activation respect decayed memory strengths?
- Are edge weights maintained across similarity recalculations?

**Citation:** Nygard, M. (2018). "Release It! Design and Deploy Production-Ready Software"

### 3. Real Workload Patterns

Unit tests use synthetic data. Integration tests should use realistic distributions:

**Embedding Distributions:**
- Not uniform random vectors
- Cluster around semantic topics
- Dimensionality-appropriate (768d for modern models)

**Graph Topology:**
- Not fully connected
- Power-law degree distribution (hubs and sparse nodes)
- Community structure (dense clusters with sparse inter-cluster edges)

**Temporal Patterns:**
- Memory creation follows Poisson process (bursty)
- Retrieval follows Zipfian distribution (some memories accessed often)

**Citations:**
- Barabási, A. L., & Albert, R. (1999). "Emergence of Scaling in Random Networks"
- Newman, M. E. (2003). "The Structure and Function of Complex Networks"

### 4. Error Injection and Chaos Testing

Deliberately introduce failures to test resilience:

**Arena Overflow:** Configure tiny arenas, force overflow
**Invalid Inputs:** NaN embeddings, negative strengths, zero-length vectors
**Resource Exhaustion:** Massive graphs exceeding memory limits
**Concurrent Access:** Race conditions in multi-threaded scenarios

**Expected Behavior:**
- Graceful degradation (fall back to Rust kernels)
- Error messages logged
- No data corruption or crashes

**Citation:** Rosenthal, C., et al. (2020). "Chaos Engineering: System Resilience in Practice"

### 5. Performance Regression Detection in Integration Tests

Integration tests measure end-to-end performance, catching regressions that unit tests miss:

**Example:**
- Zig vector kernel: 25% faster (unit test validates)
- But: FFI overhead + memory copying negates gains (integration test catches)

Measure total query latency, not just kernel execution:

```rust
let start = Instant::now();
let results = graph.query_with_zig_kernels(&query);
let latency = start.elapsed();
assert!(latency < Duration::from_micros(2000), "Query too slow: {:?}", latency);
```

**Citation:** Gregg, B. (2013). "Systems Performance: Enterprise and the Cloud"

## References

1. Fowler, M. (2014). "Testing Strategies in a Microservice Architecture"
2. Humble, J., & Farley, D. (2010). "Continuous Delivery"
3. Nygard, M. (2018). "Release It! Design and Deploy Production-Ready Software"
4. Barabási, A. L., & Albert, R. (1999). "Emergence of Scaling in Random Networks", Science
5. Gregg, B. (2013). "Systems Performance: Enterprise and the Cloud"
6. Rosenthal, C., et al. (2020). "Chaos Engineering", O'Reilly
