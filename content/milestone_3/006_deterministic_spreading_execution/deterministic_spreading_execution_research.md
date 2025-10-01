# Deterministic Spreading Execution Research

## Research Topics for Milestone 3 Task 006: Deterministic Spreading Execution

### 1. Deterministic Parallel Execution Models
- Deterministic multithreading frameworks (CoreDet, DThreads)
- Data-race-free execution and happens-before guarantees
- Phase-based execution barriers for SIMD-style workloads
- Work partitioning strategies that preserve order under parallelism
- Impact of memory consistency models on determinism

### 2. Randomness Control and Seed Management
- Counter-based PRNGs for reproducible parallel streams
- Seeding strategies for reproducible batch operations
- Splittable RNGs (PCG, Philox) versus shared RNG state
- Avoiding correlation in pseudo-random sequences across workers
- Determinism in probabilistic graph algorithms

### 3. Tie-Breaking and Ordering Policies
- Stable sorting for floating-point vectors
- Deterministic reduction operations (associativity issues)
- Lexicographic ordering and canonicalization of graph traversals
- Handling equal activation values without bias
- Precision management to avoid non-deterministic rounding

### 4. Testing and Verification of Deterministic Systems
- Differential testing with repeated runs
- Bitwise comparison versus tolerance thresholds
- Instrumenting execution traces for reproducibility checks
- Detecting data races via deterministic replay
- Benchmarks for deterministic overhead

### 5. Cognitive System Requirements for Repeatability
- Scientific reproducibility in cognitive simulations
- Audit trails for emergent behaviors
- Debuggability through deterministic replay
- Metacognitive tracing of activation decisions
- User trust and regulatory compliance

## Research Findings

### Deterministic Multithreading Foundations
Research on deterministic multithreading demonstrates that enforcing a fixed serialization of synchronization points produces repeatable results with acceptable overhead. CoreDet introduces logical parallelism segments with deterministic commits, achieving 1.14×–2.18× overhead across PARSEC benchmarks (Bergan et al., 2010). DThreads uses memory protection to replicate deterministic segments while keeping the programming model unchanged (Liu et al., 2011). For Engram, we adopt a lighter-weight approach: fixed scheduling order within spreading hops plus barriers to align worker progress. Our workload is highly structured—activation spreads in hops—making phase barriers a natural fit.

### Pseudo-Random Number Generation Strategies
Determinism requires every source of randomness to be reproducible. Counter-based generators such as Philox and Threefry allow each worker to compute random numbers independently using the same seed and counter offsets, guaranteeing identical sequences regardless of execution timing (Salmon et al., 2011). PCG-XSL-RR maintains excellent statistical quality with small state and cheap jumping, making it well-suited for seeding per-hop randomness (O'Neill, 2014). We can combine a global seed with `(hop_index, worker_id)` counters to derive reproducible sub-sequences without synchronization.

### Stable Ordering and Tie Resolution
Associativity issues in floating-point reductions can break determinism because addition is not strictly associative, leading to rounding differences based on execution order. Kahan or Neumaier compensated summation restores determinism by enforcing a canonical accumulation order (Higham, 2002). When two activations have equal magnitude, lexicographic ordering by `(activation_level, memory_id, tier)` ensures stable tie-breaking. This approach mirrors deterministic graph algorithms like Kruskal's MST, which rely on ordered edge lists to produce consistent outputs (Cormen et al., 2009).

### Testing Frameworks for Deterministic Guarantees
Deterministic systems benefit from differential testing: run the same workload multiple times, hash key outputs, and compare for bit equality (Chong et al., 2008). Instrumenting execution with trace IDs per phase enables developers to capture divergence quickly. For Engram we will serialize activation snapshots after each hop when running in validation mode, then diff them across iterations to detect non-deterministic behavior. Stress tests should skew activation values to produce many ties, ensuring tie-break code covers edge cases.

### Cognitive and Compliance Considerations
Reproducibility is critical in cognitive simulations, where researchers must replicate findings across machines and time (Gluck & Myers, 2019). In regulated domains (healthcare, finance), deterministic replay of cognitive decisions enables audit trails that justify recommendations. Deterministic execution also enhances trust for engineers: without it, diagnosing spreading anomalies becomes guesswork. Providing deterministic mode as an opt-in path lets users trade modest performance overhead for full traceability during investigation.

## Key Citations
- Bergan, T., Anderson, O., Devietti, J., Ceze, L., & Grossman, D. "CoreDet: A compiler and runtime system for deterministic multithreaded execution." *ASPLOS* (2010).
- Liu, T., Curtsinger, C., & Berger, E. D. "Dthreads: Efficient deterministic multithreading." *SOSP* (2011).
- Salmon, J. K., Moraes, M. A., Dror, R. O., & Shaw, D. E. "Parallel random numbers: As easy as 1, 2, 3." *SC* (2011).
- O'Neill, M. E. "PCG: A family of simple fast space-efficient statistically good algorithms for random number generation." (2014).
- Higham, N. J. *Accuracy and Stability of Numerical Algorithms.* (2002).
- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. *Introduction to Algorithms, 3rd Edition.* (2009).
- Chong, N., et al. "Debugging parallel applications with D2T." *IBM Systems Journal* (2008).
- Gluck, K. A., & Myers, C. W. *Computational Models of Cognitive Processes.* (2019).
