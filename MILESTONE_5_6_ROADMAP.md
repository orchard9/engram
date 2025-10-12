# Engram Milestones 5 & 6: Implementation Roadmap

## Executive Summary

This document provides implementation-ready specifications for **Milestone 5 (Probabilistic Query Foundation)** and **Milestone 6 (Consolidation System)**. Every task includes precise technical specifications, file paths, integration points, acceptance criteria, and risk mitigation strategies. An engineer can read any task file and begin implementation immediately.

---

# MILESTONE 5: PROBABILISTIC QUERY FOUNDATION

## Overview

Build production query executor returning `Vec<(Episode, Confidence)>` with mathematically sound uncertainty propagation. Distinguish "no results" from "low confidence results" using confidence intervals and evidence tracking.

**Duration**: 14 days
**Critical Path**: Tasks 001 → 002 → 003 → 004 → 005 → 006 → 007

---

## Task 001: Query Executor Core (3 days, P0)

**File**: `roadmap/milestone-5/001_query_executor_core_pending.md` (already created)

**Objective**: Implement `ProbabilisticQueryExecutor` that transforms recall results into probabilistic query results with evidence tracking and uncertainty propagation.

**Key Files**:
- Create: `engram-core/src/query/executor.rs`
- Modify: `engram-core/src/query/mod.rs`, `engram-core/src/store.rs`

**Integration**: Uses existing `ConfidenceAggregator`, `Confidence` type, `ProbabilisticQueryResult`

---

## Task 002: Evidence Aggregation Engine (2 days, P0)

**Objective**: Implement lock-free evidence combination with circular dependency detection and Bayesian updating.

**Technical Approach**:
1. Create `engram-core/src/query/evidence_aggregator.rs`
   - `LockFreeEvidenceAggregator` with crossbeam-epoch for memory management
   - `CircularDependencyDetector` using Tarjan's algorithm for cycle detection
   - `BayesianCombiner` applying Bayes' theorem to evidence chains

2. Evidence Graph Structure:
```rust
#[repr(C, align(64))] // Cache-line aligned
pub struct EvidenceNode {
    evidence: PackedEvidence,
    next: Atomic<EvidenceNode>,
    dependencies: SmallVec<[EvidenceId; 4]>, // Most evidence has <4 deps
}

pub struct LockFreeEvidenceAggregator {
    evidence_graph: Atomic<EvidenceNode>,
    dependency_detector: CircularDependencyDetector,
    computation_cache: DashMap<u64, ConfidenceInterval>,
}
```

3. Integration Points:
   - Use `EvidenceSource` enum from `query/mod.rs`
   - Integrate with `SpreadingActivation` evidence from activation module
   - Connect to `TemporalDecay` evidence from decay functions

**Acceptance Criteria**:
- [ ] Lock-free evidence aggregation with zero allocations in hot path
- [ ] Circular dependency detection prevents infinite loops (validated with cycle graphs)
- [ ] Bayesian updating matches analytical solutions within 1% error
- [ ] Sub-100μs latency for aggregating 10 evidence sources
- [ ] Property tests verify associativity and commutativity
- [ ] Integration tests with spreading activation and temporal decay evidence

**Files**:
- Create: `engram-core/src/query/evidence_aggregator.rs`
- Create: `engram-core/src/query/dependency_graph.rs`
- Modify: `engram-core/src/query/mod.rs`
- Tests: `engram-core/tests/evidence_aggregation_tests.rs`

---

## Task 003: Uncertainty Tracking System (2 days, P1)

**Objective**: Track uncertainty from spreading activation, temporal decay, system pressure, and measurement errors through the full query pipeline.

**Technical Approach**:
1. Create `engram-core/src/query/uncertainty_tracker.rs`
2. Implement uncertainty sources:
   - `SystemPressureUncertainty`: Extract from `MetricsRegistry`
   - `SpreadingActivationUncertainty`: Use `SpreadingMetrics` variance
   - `TemporalDecayUncertainty`: Compute from decay model parameters
   - `MeasurementUncertainty`: HNSW distance confidence degradation

3. Uncertainty Propagation Algorithm:
```rust
impl UncertaintyTracker {
    /// Accumulate uncertainty from independent sources
    fn propagate_independent_uncertainty(&self, sources: &[UncertaintySource]) -> f32 {
        // Sum variances (for independent sources)
        let variance: f32 = sources.iter()
            .map(|s| s.variance().powi(2))
            .sum();
        variance.sqrt().min(0.8) // Cap at 80%
    }

    /// Propagate uncertainty through correlated sources
    fn propagate_correlated_uncertainty(
        &self,
        sources: &[UncertaintySource],
        correlation_matrix: &[[f32; N]; N],
    ) -> f32 {
        // Apply correlation-aware combination
        // σ² = Σᵢ σᵢ² + 2Σᵢ<ⱼ ρᵢⱼσᵢσⱼ
        unimplemented!("Defer to Task 004")
    }
}
```

4. Integration with Activation Spreading:
   - Hook into `SpreadingMetrics::tier_latency_snapshot()` for variance
   - Extract cycle detection counts as uncertainty signal
   - Use pool utilization as system pressure indicator

**Acceptance Criteria**:
- [ ] Uncertainty extracted from all system sources (4 types documented)
- [ ] Independent uncertainty propagation mathematically correct
- [ ] High system pressure increases confidence interval width by 10-30%
- [ ] Spreading activation cycles increase uncertainty proportionally
- [ ] Temporal decay uncertainty grows with time since encoding
- [ ] Integration tests verify uncertainty from real spreading/decay operations

**Files**:
- Create: `engram-core/src/query/uncertainty_tracker.rs`
- Create: `engram-core/src/query/uncertainty_sources.rs`
- Modify: `engram-core/src/query/executor.rs`
- Tests: `engram-core/tests/uncertainty_tracking_tests.rs`

---

## Task 004: Confidence Calibration Framework (2 days, P1)

**Objective**: Implement empirical calibration that ensures predicted confidence scores correlate >0.9 with actual retrieval accuracy.

**Technical Approach**:
1. Create `engram-core/src/query/calibration.rs`
2. Implement calibration assessment:
```rust
pub struct CalibrationFramework {
    /// 10 bins for confidence ranges [0.0-0.1, 0.1-0.2, ..., 0.9-1.0]
    calibration_bins: [CalibrationBin; 10],
    /// Running statistics for calibration metrics
    brier_score_accumulator: AtomicF32,
    total_predictions: AtomicU64,
}

pub struct CalibrationBin {
    predicted_confidence_sum: AtomicF32,
    actual_outcomes_sum: AtomicU32, // Count of true positives
    sample_count: AtomicU32,
}

impl CalibrationFramework {
    /// Record prediction and actual outcome for calibration
    pub fn record_prediction(&self, predicted: f32, actual_outcome: bool) {
        let bin_index = (predicted * 10.0).floor() as usize;
        let bin = &self.calibration_bins[bin_index];

        bin.predicted_confidence_sum.fetch_add(predicted, Ordering::Relaxed);
        bin.actual_outcomes_sum.fetch_add(actual_outcome as u32, Ordering::Relaxed);
        bin.sample_count.fetch_add(1, Ordering::Relaxed);

        // Update Brier score
        let error = if actual_outcome { 1.0 - predicted } else { predicted };
        let brier_contrib = error * error;
        self.brier_score_accumulator.fetch_add(brier_contrib, Ordering::Relaxed);
        self.total_predictions.fetch_add(1, Ordering::Relaxed);
    }

    /// Compute calibration error
    pub fn compute_calibration_error(&self) -> f32 {
        let mut total_error = 0.0;
        let mut total_weight = 0.0;

        for bin in &self.calibration_bins {
            let count = bin.sample_count.load(Ordering::Relaxed);
            if count < 10 { continue; } // Skip bins with insufficient data

            let predicted_avg = bin.predicted_confidence_sum.load(Ordering::Relaxed) / count as f32;
            let actual_freq = bin.actual_outcomes_sum.load(Ordering::Relaxed) as f32 / count as f32;
            let bin_error = (predicted_avg - actual_freq).abs();

            total_error += bin_error * count as f32;
            total_weight += count as f32;
        }

        if total_weight > 0.0 {
            total_error / total_weight
        } else {
            0.0
        }
    }
}
```

3. Calibration Correction:
   - Apply Platt scaling for overconfidence correction
   - Use isotonic regression for non-parametric calibration
   - Expose calibration diagnostics via `/api/v1/system/query/calibration`

4. Integration with Query Executor:
   - Record every query result for calibration
   - Apply calibration correction before returning confidence intervals
   - Expose calibration metrics in streaming telemetry

**Acceptance Criteria**:
- [ ] Calibration error <5% across all confidence bins with >100 samples
- [ ] Brier score <0.25 on test dataset
- [ ] Predicted confidence correlates >0.9 with retrieval accuracy (Spearman)
- [ ] Calibration correction reduces overconfidence by 10-20%
- [ ] Reliability diagram shows tight fit to diagonal
- [ ] Calibration metrics exposed via HTTP API
- [ ] Property tests verify calibration updates are atomic and consistent

**Files**:
- Create: `engram-core/src/query/calibration.rs`
- Create: `engram-core/src/query/calibration_diagnostics.rs`
- Modify: `engram-core/src/query/executor.rs`
- Tests: `engram-core/tests/confidence_calibration_tests.rs`
- CLI: `engram-cli/src/api.rs` (add `/api/v1/system/query/calibration` endpoint)

---

## Task 005: SMT Verification Integration (2 days, P2)

**Objective**: Integrate Z3 SMT solver to formally verify probability operations maintain axioms during development and testing.

**Technical Approach**:
1. Create `engram-core/src/query/smt_verification.rs`
2. Add feature flag `smt_verification` to `Cargo.toml`
3. Implement verification suite:
```rust
#[cfg(feature = "smt_verification")]
pub struct SMTVerificationSuite {
    z3_context: z3::Context,
    z3_solver: z3::Solver<'static>,
    proof_cache: DashMap<String, VerificationProof>,
}

#[cfg(feature = "smt_verification")]
impl SMTVerificationSuite {
    /// Verify probability axioms hold
    pub fn verify_probability_axioms(&mut self) -> Result<VerificationProof, VerificationError> {
        let p = Real::new_const(&self.z3_context, "p");
        let q = Real::new_const(&self.z3_context, "q");

        // Axiom 1: 0 ≤ P(A) ≤ 1
        let axiom1 = p.ge(&Real::from_int(&self.z3_context, 0))
            ._and(&p.le(&Real::from_int(&self.z3_context, 1)));

        // Axiom 2: P(A ∧ B) ≤ min(P(A), P(B))
        let conjunction = p * q;
        let axiom2 = conjunction.le(&p.min(&q));

        // Axiom 3: P(A ∨ B) ≤ P(A) + P(B)
        let union = p + q - (p * q);
        let axiom3 = union.le(&(p + q));

        self.z3_solver.assert(&axiom1);
        self.z3_solver.assert(&axiom2);
        self.z3_solver.assert(&axiom3);

        match self.z3_solver.check() {
            SatResult::Sat => Ok(VerificationProof::new("probability_axioms", "verified")),
            SatResult::Unsat => Err(VerificationError::UnsatisfiableAxioms),
            SatResult::Unknown => Err(VerificationError::VerificationTimeout),
        }
    }

    /// Verify Bayes' theorem application
    pub fn verify_bayes_theorem(&mut self) -> Result<VerificationProof, VerificationError> {
        // P(A|B) = P(B|A) * P(A) / P(B)
        // Implementation details...
        unimplemented!()
    }
}
```

4. Integration Points:
   - Run verification in CI for PRs touching probability code
   - Cache proofs to avoid repeated verification
   - Provide `--verify` flag for local development
   - Document verification failures with SMT counterexamples

5. Dependencies:
   - Add `z3 = { version = "0.12", optional = true }` to Cargo.toml
   - Feature gate: `smt_verification = ["z3"]`

**Acceptance Criteria**:
- [ ] Z3 integration compiles with `--features smt_verification`
- [ ] Probability axioms verified correct
- [ ] Bayes' theorem application verified correct
- [ ] Conjunction fallacy prevention verified
- [ ] Verification runs in CI on probability code changes
- [ ] Proof caching reduces verification time by >80%
- [ ] Documentation explains how to run verification locally
- [ ] Verification failures provide actionable counterexamples

**Files**:
- Create: `engram-core/src/query/smt_verification.rs`
- Modify: `engram-core/Cargo.toml` (add z3 dependency)
- Create: `engram-core/tests/smt_verification_tests.rs`
- Docs: `docs/operations/smt_verification.md`
- CI: `.github/workflows/smt-verification.yml`

---

## Task 006: Query Operations & Performance (2 days, P1)

**Objective**: Implement probabilistic AND/OR/NOT operations with sub-millisecond latency and SIMD optimization where beneficial.

**Technical Approach**:
1. Extend existing `ConfidenceInterval` operations in `query/mod.rs`
2. Implement query combinators:
```rust
impl ProbabilisticQueryResult {
    /// Conjunctive query: A AND B
    pub fn and(&self, other: &Self) -> Self {
        // Combine episodes (intersection)
        let episodes = self.intersect_episodes(&other.episodes);

        // Combine confidence intervals (multiplication for independence)
        let confidence_interval = self.confidence_interval.and(&other.confidence_interval);

        // Merge evidence chains
        let evidence_chain = self.merge_evidence_chains(&other.evidence_chain);

        // Combine uncertainty sources
        let uncertainty_sources = self.combine_uncertainty_sources(&other.uncertainty_sources);

        Self {
            episodes,
            confidence_interval,
            evidence_chain,
            uncertainty_sources,
        }
    }

    /// Disjunctive query: A OR B
    pub fn or(&self, other: &Self) -> Self {
        // Combine episodes (union)
        let episodes = self.union_episodes(&other.episodes);

        // Combine confidence intervals
        let confidence_interval = self.confidence_interval.or(&other.confidence_interval);

        // Merge evidence chains
        let evidence_chain = self.merge_evidence_chains(&other.evidence_chain);

        // Combine uncertainty sources
        let uncertainty_sources = self.combine_uncertainty_sources(&other.uncertainty_sources);

        Self {
            episodes,
            confidence_interval,
            evidence_chain,
            uncertainty_sources,
        }
    }

    /// Negation query: NOT A
    pub fn not(&self) -> Self {
        // Episodes remain the same (negation affects confidence only)
        let episodes = self.episodes.clone();

        // Negate confidence interval
        let confidence_interval = self.confidence_interval.not();

        // Evidence chain remains (uncertainty unchanged)
        let evidence_chain = self.evidence_chain.clone();
        let uncertainty_sources = self.uncertainty_sources.clone();

        Self {
            episodes,
            confidence_interval,
            evidence_chain,
            uncertainty_sources,
        }
    }
}
```

3. Performance Optimizations:
   - Use `SmallVec` for evidence chains (<8 items avoid heap allocation)
   - SIMD batch confidence interval operations (AVX2 when available)
   - Cache common query patterns in `DashMap<QuerySignature, CachedResult>`
   - Arena allocation for temporary query state

4. Benchmark Suite:
```rust
// engram-core/benches/query_operations.rs
fn bench_query_and_operation(c: &mut Criterion) {
    let mut group = c.benchmark_group("query_and");
    group.warm_up_time(Duration::from_secs(2));

    group.bench_function("2_results", |b| {
        let q1 = create_query_result(2);
        let q2 = create_query_result(2);
        b.iter(|| q1.and(&q2));
    });

    group.bench_function("10_results", |b| {
        let q1 = create_query_result(10);
        let q2 = create_query_result(10);
        b.iter(|| q1.and(&q2));
    });

    group.finish();
}

criterion_group!(benches, bench_query_and_operation, bench_query_or_operation);
criterion_main!(benches);
```

**Acceptance Criteria**:
- [ ] AND/OR/NOT operations maintain probability axioms
- [ ] Query latency <1ms P95 for 10-result queries
- [ ] SIMD optimization provides 2x speedup on batch operations (AVX2)
- [ ] Memory allocation <100 bytes per query operation
- [ ] Evidence chain merging preserves dependency information
- [ ] Criterion benchmarks show no performance regressions
- [ ] Property tests verify operation associativity and commutativity

**Files**:
- Modify: `engram-core/src/query/mod.rs` (add combinators to `ProbabilisticQueryResult`)
- Create: `engram-core/src/query/query_combinators.rs`
- Create: `engram-core/benches/query_operations.rs`
- Tests: `engram-core/tests/query_operations_tests.rs`

---

## Task 007: Integration & Production Validation (1 day, P0)

**Objective**: Integrate probabilistic queries into production pipeline and validate end-to-end correctness with comprehensive test suite.

**Technical Approach**:
1. Full Integration Testing:
   - Test complete pipeline: Cue → recall → spreading → probabilistic result
   - Validate evidence from all sources (HNSW, spreading, decay)
   - Confirm uncertainty tracking through full stack
   - Verify calibration works on real query patterns

2. Create `engram-core/tests/probabilistic_query_integration.rs`:
```rust
#[tokio::test]
async fn test_end_to_end_probabilistic_query() {
    // Setup: Create memory store with test data
    let store = MemoryStore::new();
    populate_test_data(&store, 100).await;

    // Execute: Run probabilistic query
    let cue = Cue::from_text("test query");
    let result = store.recall_probabilistic(cue).await;

    // Validate: Check all components
    assert!(!result.is_empty());
    assert!(result.confidence_interval.point.raw() > 0.5);
    assert!(!result.evidence_chain.is_empty());
    assert!(result.uncertainty_sources.len() > 0);

    // Verify: Probability axioms hold
    assert!(result.confidence_interval.lower.raw() <= result.confidence_interval.upper.raw());
    assert!(result.confidence_interval.width < 0.5);
}

#[tokio::test]
async fn test_probabilistic_query_with_spreading_activation() {
    let store = MemoryStore::new();
    populate_test_data(&store, 50).await;

    // Enable spreading activation
    let config = RecallConfig::default().with_spreading(true);
    let result = store.recall_probabilistic_with_config(
        Cue::from_text("spreading test"),
        config,
    ).await;

    // Verify spreading evidence present
    let has_spreading_evidence = result.evidence_chain.iter().any(|e| {
        matches!(e.source, EvidenceSource::SpreadingActivation { .. })
    });
    assert!(has_spreading_evidence);
}
```

3. Property-Based Integration Tests:
```rust
proptest! {
    #[test]
    fn test_query_results_always_valid(
        episodes in prop::collection::vec(any_episode(), 1..50)
    ) {
        let runtime = tokio::runtime::Runtime::new().unwrap();
        let result = runtime.block_on(async {
            let store = create_test_store();
            for ep in episodes {
                store.store(ep).await.unwrap();
            }
            store.recall_probabilistic(random_cue()).await
        });

        prop_assert!(result.confidence_interval.lower.raw() >= 0.0);
        prop_assert!(result.confidence_interval.upper.raw() <= 1.0);
        prop_assert!(!result.evidence_chain.is_empty() || result.is_empty());
    }
}
```

4. Stress Testing:
   - Run 100K queries and verify no panics
   - Validate memory usage stays bounded
   - Confirm calibration improves with more samples
   - Test concurrent query execution (1K concurrent queries)

**Acceptance Criteria**:
- [ ] End-to-end integration tests pass for full query pipeline
- [ ] Probabilistic queries work with spreading activation
- [ ] Evidence chains include all relevant sources
- [ ] Uncertainty tracking integrates with system metrics
- [ ] Calibration framework records real query outcomes
- [ ] 100K query stress test completes without errors
- [ ] Memory usage stays bounded (<1GB for 100K queries)
- [ ] Concurrent queries (1K) execute without data races
- [ ] Property tests verify probabilistic guarantees hold

**Files**:
- Create: `engram-core/tests/probabilistic_query_integration.rs`
- Create: `engram-core/tests/query_stress_tests.rs`
- Modify: `engram-cli/src/main.rs` (wire up probabilistic queries)
- Modify: `engram-cli/src/api.rs` (add `/api/v1/query/probabilistic` endpoint)

---

# MILESTONE 6: CONSOLIDATION SYSTEM

## Overview

Implement `consolidate_async()` and `dream()` operations that transform episodic memories into semantic knowledge through unsupervised statistical pattern detection. Consolidation must be interruptible, not block recall, and achieve >50% storage reduction while maintaining >95% recall accuracy.

**Duration**: 16 days
**Critical Path**: Tasks 001 → 002 → 003 → 004 → 005 → 006 → 007

---

## Task 001: Consolidation Scheduler (3 days, P0)

**Objective**: Implement async consolidation scheduler that runs during system idle periods without blocking recall operations.

**Technical Approach**:
1. Create `engram-core/src/consolidation/scheduler.rs`
2. Implement background consolidation task:
```rust
pub struct ConsolidationScheduler {
    /// Configuration for consolidation intervals
    config: ConsolidationConfig,
    /// Task handle for cancellation
    task_handle: Option<JoinHandle<()>>,
    /// Consolidation state
    state: Arc<RwLock<ConsolidationState>>,
    /// Metrics
    metrics: ConsolidationMetrics,
}

pub struct ConsolidationConfig {
    /// Minimum idle time before starting consolidation
    pub idle_threshold: Duration,
    /// Maximum consolidation duration per cycle
    pub max_duration: Duration,
    /// Consolidation interval (e.g., once per hour)
    pub interval: Duration,
    /// Target reduction ratio (e.g., 0.5 for 50% reduction)
    pub target_reduction: f32,
    /// Minimum confidence for episode inclusion
    pub confidence_threshold: f32,
}

impl ConsolidationScheduler {
    /// Start background consolidation task
    pub fn start(&mut self) -> Result<(), ConsolidationError> {
        let state = Arc::clone(&self.state);
        let config = self.config.clone();
        let metrics = self.metrics.clone();

        let handle = tokio::spawn(async move {
            loop {
                // Wait for next consolidation interval
                tokio::time::sleep(config.interval).await;

                // Check if system is idle
                if !Self::is_system_idle(&state).await {
                    metrics.skipped_cycles.fetch_add(1, Ordering::Relaxed);
                    continue;
                }

                // Run consolidation cycle
                match Self::run_consolidation_cycle(&state, &config, &metrics).await {
                    Ok(outcome) => {
                        metrics.successful_cycles.fetch_add(1, Ordering::Relaxed);
                        metrics.last_reduction_ratio.store(outcome.reduction_ratio, Ordering::Relaxed);
                    }
                    Err(e) => {
                        tracing::warn!("Consolidation cycle failed: {}", e);
                        metrics.failed_cycles.fetch_add(1, Ordering::Relaxed);
                    }
                }
            }
        });

        self.task_handle = Some(handle);
        Ok(())
    }

    /// Check if system is idle (low query rate, low memory pressure)
    async fn is_system_idle(state: &Arc<RwLock<ConsolidationState>>) -> bool {
        let state = state.read().await;
        state.recent_query_rate < 10.0 // <10 queries/second
            && state.memory_pressure < 0.5 // <50% pressure
            && state.active_recalls == 0 // No active recalls
    }

    /// Run single consolidation cycle
    async fn run_consolidation_cycle(
        state: &Arc<RwLock<ConsolidationState>>,
        config: &ConsolidationConfig,
        metrics: &ConsolidationMetrics,
    ) -> Result<ConsolidationOutcome, ConsolidationError> {
        let start = Instant::now();

        // Phase 1: Select episodes for consolidation
        let episodes = Self::select_consolidation_candidates(state, config).await?;

        // Phase 2: Detect patterns
        let patterns = Self::detect_patterns(&episodes)?;

        // Phase 3: Create semantic memories
        let semantic_memories = Self::extract_semantic_memories(&patterns, config)?;

        // Phase 4: Update storage (interruptible)
        let outcome = Self::apply_consolidation(state, episodes, semantic_memories, start).await?;

        Ok(outcome)
    }
}

pub struct ConsolidationOutcome {
    pub episodes_consolidated: usize,
    pub patterns_extracted: usize,
    pub semantic_memories_created: usize,
    pub reduction_ratio: f32, // Storage reduction achieved
    pub consolidation_time: Duration,
}
```

3. Interruptibility Mechanism:
   - Check cancellation token at phase boundaries
   - Allow graceful shutdown mid-consolidation
   - Persist partial results to resume later
   - Track consolidation progress in state

4. Integration with MemoryStore:
   - Hook scheduler into `MemoryStore::new()`
   - Expose `consolidate_async()` API
   - Add `is_consolidating()` status check
   - Provide `pause_consolidation()` / `resume_consolidation()` controls

**Acceptance Criteria**:
- [ ] Scheduler starts automatically with MemoryStore
- [ ] Consolidation only runs during idle periods (query rate <10/sec)
- [ ] Consolidation yields to recall operations (recall latency not impacted)
- [ ] Graceful shutdown interrupts consolidation cleanly
- [ ] Partial consolidation results persisted for resumption
- [ ] Metrics track consolidation cycles, success rate, reduction ratio
- [ ] HTTP API exposes consolidation status (`GET /api/v1/consolidation/status`)
- [ ] Property tests verify consolidation never blocks recall

**Files**:
- Create: `engram-core/src/consolidation/scheduler.rs`
- Create: `engram-core/src/consolidation/mod.rs`
- Modify: `engram-core/src/store.rs`
- Modify: `engram-core/src/lib.rs`
- Create: `engram-core/tests/consolidation_scheduler_tests.rs`
- CLI: `engram-cli/src/api.rs` (add consolidation endpoints)

---

## Task 002: Pattern Detection Engine (4 days, P0)

**Objective**: Implement unsupervised statistical pattern detection that identifies recurring structures in episodic memories without supervision.

**Technical Approach**:
1. Create `engram-core/src/consolidation/pattern_detector.rs`
2. Use existing consolidation stub from `engram-core/src/completion/consolidation.rs` as foundation
3. Implement clustering algorithms:
```rust
pub struct PatternDetector {
    /// Configuration for pattern detection
    config: PatternDetectionConfig,
    /// Cached patterns for fast lookup
    pattern_cache: DashMap<PatternSignature, CachedPattern>,
}

pub struct PatternDetectionConfig {
    /// Minimum cluster size for pattern extraction
    pub min_cluster_size: usize,
    /// Similarity threshold for clustering (cosine similarity)
    pub similarity_threshold: f32,
    /// Maximum number of patterns to extract per cycle
    pub max_patterns: usize,
}

impl PatternDetector {
    /// Detect patterns in episode collection using embedding similarity
    pub fn detect_patterns(&self, episodes: &[Episode]) -> Vec<EpisodicPattern> {
        // Phase 1: Cluster episodes by embedding similarity
        let clusters = self.cluster_episodes(episodes);

        // Phase 2: Extract common patterns from clusters
        let mut patterns = Vec::new();
        for cluster in clusters {
            if cluster.len() >= self.config.min_cluster_size {
                if let Some(pattern) = self.extract_pattern(&cluster) {
                    patterns.push(pattern);
                }
            }
        }

        // Phase 3: Merge similar patterns
        self.merge_similar_patterns(patterns)
    }

    /// Cluster episodes using hierarchical agglomerative clustering
    fn cluster_episodes(&self, episodes: &[Episode]) -> Vec<Vec<Episode>> {
        let mut clusters: Vec<Vec<Episode>> = episodes.iter()
            .map(|ep| vec![ep.clone()])
            .collect();

        // Iteratively merge most similar clusters
        while clusters.len() > 1 {
            let (i, j, similarity) = self.find_most_similar_clusters(&clusters);

            if similarity < self.config.similarity_threshold {
                break; // No more similar clusters to merge
            }

            // Merge clusters i and j
            let merged = clusters[i].iter().chain(clusters[j].iter()).cloned().collect();
            clusters[i] = merged;
            clusters.remove(j);
        }

        clusters
    }

    /// Extract pattern from cluster of similar episodes
    fn extract_pattern(&self, episodes: &[Episode]) -> Option<EpisodicPattern> {
        if episodes.is_empty() {
            return None;
        }

        // Compute average embedding
        let avg_embedding = self.average_embeddings(episodes);

        // Compute pattern strength (confidence)
        let strength = self.compute_pattern_strength(episodes, &avg_embedding);

        // Extract common features
        let features = self.extract_common_features(episodes);

        Some(EpisodicPattern {
            id: format!("pattern_{}", Utc::now().timestamp_nanos()),
            embedding: avg_embedding,
            source_episodes: episodes.iter().map(|ep| ep.id.clone()).collect(),
            strength,
            features,
            first_occurrence: episodes.iter().map(|ep| ep.when).min().unwrap(),
            last_occurrence: episodes.iter().map(|ep| ep.when).max().unwrap(),
            occurrence_count: episodes.len(),
        })
    }

    /// Compute embedding similarity (cosine similarity)
    fn embedding_similarity(a: &[f32; 768], b: &[f32; 768]) -> f32 {
        let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm_a > 0.0 && norm_b > 0.0 {
            dot / (norm_a * norm_b)
        } else {
            0.0
        }
    }
}

/// Represents a detected episodic pattern
pub struct EpisodicPattern {
    pub id: String,
    pub embedding: [f32; 768],
    pub source_episodes: Vec<String>,
    pub strength: f32,
    pub features: Vec<PatternFeature>,
    pub first_occurrence: DateTime<Utc>,
    pub last_occurrence: DateTime<Utc>,
    pub occurrence_count: usize,
}

/// Common features extracted from pattern
pub enum PatternFeature {
    TemporalSequence { interval: Duration },
    SpatialProximity { location: String },
    ConceptualTheme { theme: String },
    EmotionalValence { valence: f32 },
}
```

4. Statistical Regularity Detection:
   - Use chi-square test for feature co-occurrence significance
   - Apply mutual information for feature dependency
   - Compute pattern likelihood ratio vs random baseline
   - Filter patterns below statistical significance threshold (p < 0.01)

5. Integration with Existing Consolidation Stub:
   - Extend `ConsolidationEngine` from `completion/consolidation.rs`
   - Use existing `cluster_episodes()` and `extract_common_pattern()` methods
   - Replace simple similarity with statistical significance testing

**Acceptance Criteria**:
- [ ] Pattern detection identifies recurring structures in >80% of test datasets
- [ ] Statistical significance filtering removes spurious patterns (p < 0.01)
- [ ] Minimum cluster size enforced (default 3 episodes)
- [ ] Pattern strength correlates with occurrence frequency (r > 0.8)
- [ ] Hierarchical clustering produces valid dendrograms
- [ ] Embedding averaging preserves semantic meaning (validated with test queries)
- [ ] Pattern features capture temporal, spatial, and conceptual regularities
- [ ] Performance: <100ms for 100 episodes, <1s for 1000 episodes

**Files**:
- Create: `engram-core/src/consolidation/pattern_detector.rs`
- Create: `engram-core/src/consolidation/clustering.rs`
- Create: `engram-core/src/consolidation/statistical_tests.rs`
- Modify: `engram-core/src/completion/consolidation.rs` (integrate pattern detector)
- Create: `engram-core/tests/pattern_detection_tests.rs`
- Bench: `engram-core/benches/pattern_detection.rs`

---

## Task 003: Semantic Memory Extraction (3 days, P1)

**Objective**: Transform detected patterns into semantic memories with abstract representations that preserve retrieval accuracy.

**Technical Approach**:
1. Create `engram-core/src/consolidation/semantic_extraction.rs`
2. Implement semantic memory construction:
```rust
pub struct SemanticExtractor {
    /// Configuration for semantic extraction
    config: SemanticExtractionConfig,
    /// Existing semantic memories for merging
    semantic_store: Arc<RwLock<SemanticMemoryStore>>,
}

pub struct SemanticExtractionConfig {
    /// Minimum pattern strength for semantic memory creation
    pub min_pattern_strength: f32,
    /// Maximum abstraction level (0=concrete, 1=fully abstract)
    pub abstraction_level: f32,
    /// Preserve episodic details for reconstruction
    pub preserve_details: bool,
}

impl SemanticExtractor {
    /// Extract semantic memories from detected patterns
    pub fn extract_semantic_memories(
        &self,
        patterns: &[EpisodicPattern],
    ) -> Vec<SemanticMemory> {
        patterns.iter()
            .filter(|p| p.strength >= self.config.min_pattern_strength)
            .map(|pattern| self.pattern_to_semantic(pattern))
            .collect()
    }

    /// Convert episodic pattern to semantic memory
    fn pattern_to_semantic(&self, pattern: &EpisodicPattern) -> SemanticMemory {
        // Abstract embedding (average of source embeddings)
        let abstract_embedding = pattern.embedding;

        // Compute semantic confidence (based on pattern strength and occurrence count)
        let semantic_confidence = self.compute_semantic_confidence(pattern);

        // Extract schema (common structure)
        let schema = self.extract_schema(pattern);

        // Create semantic memory
        SemanticMemory {
            id: format!("semantic_{}", pattern.id),
            abstract_embedding,
            schema,
            confidence: semantic_confidence,
            source_pattern_id: pattern.id.clone(),
            episodic_references: pattern.source_episodes.clone(),
            consolidation_time: Utc::now(),
            retrieval_count: 0,
            last_retrieval: None,
        }
    }

    /// Compute semantic confidence from pattern statistics
    fn compute_semantic_confidence(&self, pattern: &EpisodicPattern) -> Confidence {
        // Confidence increases with:
        // 1. Pattern strength (coherence of cluster)
        // 2. Occurrence count (frequency)
        // 3. Temporal stability (long span)

        let strength_component = pattern.strength;
        let frequency_component = (pattern.occurrence_count as f32 / 100.0).min(1.0);
        let stability_component = self.temporal_stability(pattern);

        let raw_confidence = (strength_component * 0.5)
            + (frequency_component * 0.3)
            + (stability_component * 0.2);

        Confidence::exact(raw_confidence.clamp(0.0, 1.0))
    }

    /// Extract schema (common structure) from pattern
    fn extract_schema(&self, pattern: &EpisodicPattern) -> Schema {
        Schema {
            temporal_structure: self.extract_temporal_structure(pattern),
            conceptual_roles: self.extract_conceptual_roles(pattern),
            causal_relations: self.extract_causal_relations(pattern),
        }
    }
}

/// Semantic memory representation
pub struct SemanticMemory {
    pub id: String,
    pub abstract_embedding: [f32; 768],
    pub schema: Schema,
    pub confidence: Confidence,
    pub source_pattern_id: String,
    pub episodic_references: Vec<String>,
    pub consolidation_time: DateTime<Utc>,
    pub retrieval_count: u64,
    pub last_retrieval: Option<DateTime<Utc>>,
}

/// Schema representing abstract structure
pub struct Schema {
    pub temporal_structure: TemporalStructure,
    pub conceptual_roles: Vec<ConceptualRole>,
    pub causal_relations: Vec<CausalRelation>,
}

pub struct TemporalStructure {
    pub typical_duration: Duration,
    pub event_sequence: Vec<String>, // Ordered event types
}

pub struct ConceptualRole {
    pub role_name: String,
    pub typical_filler: String, // Prototypical concept
}

pub struct CausalRelation {
    pub antecedent: String,
    pub consequent: String,
    pub strength: f32,
}
```

3. Abstraction Levels:
   - Level 0: Preserve all episodic details (minimal abstraction)
   - Level 0.5: Abstract away specific entities, preserve structure
   - Level 1.0: Fully abstract schema, only general pattern

4. Quality Validation:
   - Verify semantic memories can reconstruct source episodes
   - Validate retrieval accuracy with semantic memories
   - Ensure storage reduction meets target (>50%)
   - Confirm no information loss for high-confidence memories

**Acceptance Criteria**:
- [ ] Semantic memories created from patterns with strength >0.7
- [ ] Abstraction level configurable (0.0-1.0)
- [ ] Semantic confidence correlates with pattern coherence (r > 0.8)
- [ ] Schema extraction captures temporal and conceptual structure
- [ ] Episodic references preserved for reconstruction
- [ ] Retrieval accuracy maintained >95% with semantic memories
- [ ] Storage reduction >50% after semantic extraction
- [ ] Validation tests confirm no information loss

**Files**:
- Create: `engram-core/src/consolidation/semantic_extraction.rs`
- Create: `engram-core/src/consolidation/schema.rs`
- Create: `engram-core/src/memory.rs` (add `SemanticMemory` type)
- Create: `engram-core/tests/semantic_extraction_tests.rs`
- Create: `engram-core/tests/reconstruction_tests.rs`

---

## Task 004: Storage Compaction (2 days, P1)

**Objective**: Implement storage compaction that replaces consolidated episodes with semantic memories while maintaining retrieval capability.

**Technical Approach**:
1. Create `engram-core/src/consolidation/compaction.rs`
2. Implement safe episode replacement:
```rust
pub struct StorageCompactor {
    /// Configuration for compaction
    config: CompactionConfig,
    /// Metrics
    metrics: CompactionMetrics,
}

pub struct CompactionConfig {
    /// Minimum age for episode consolidation (e.g., 7 days)
    pub min_episode_age: Duration,
    /// Preserve high-confidence episodes
    pub preserve_threshold: f32,
    /// Verify retrieval before compaction
    pub verify_retrieval: bool,
}

impl StorageCompactor {
    /// Replace consolidated episodes with semantic memories
    pub async fn compact_storage(
        &self,
        store: &MemoryStore,
        episodes: Vec<Episode>,
        semantic_memory: SemanticMemory,
    ) -> Result<CompactionResult, CompactionError> {
        // Phase 1: Verify semantic memory can reconstruct episodes
        if self.config.verify_retrieval {
            self.verify_reconstruction(&episodes, &semantic_memory).await?;
        }

        // Phase 2: Create semantic memory in store
        let semantic_id = store.store_semantic(semantic_memory.clone()).await?;

        // Phase 3: Mark episodes as consolidated (soft delete)
        for episode in &episodes {
            store.mark_consolidated(&episode.id, &semantic_id).await?;
        }

        // Phase 4: Verify retrieval still works
        if self.config.verify_retrieval {
            self.verify_post_compaction_retrieval(store, &episodes, &semantic_memory).await?;
        }

        // Phase 5: Remove consolidated episodes (hard delete)
        let removed_count = store.remove_consolidated_episodes(&episodes).await?;

        let result = CompactionResult {
            episodes_removed: removed_count,
            semantic_memory_id: semantic_id,
            storage_reduction_bytes: self.compute_storage_reduction(&episodes, &semantic_memory),
        };

        self.metrics.record_compaction(&result);
        Ok(result)
    }

    /// Verify semantic memory can reconstruct episodes
    async fn verify_reconstruction(
        &self,
        episodes: &[Episode],
        semantic_memory: &SemanticMemory,
    ) -> Result<(), CompactionError> {
        for episode in episodes {
            let reconstructed = self.reconstruct_from_semantic(episode, semantic_memory)?;
            let similarity = self.embedding_similarity(&episode.embedding, &reconstructed.embedding);

            if similarity < 0.8 {
                return Err(CompactionError::ReconstructionFailed {
                    episode_id: episode.id.clone(),
                    similarity,
                });
            }
        }
        Ok(())
    }

    /// Verify retrieval works after compaction
    async fn verify_post_compaction_retrieval(
        &self,
        store: &MemoryStore,
        episodes: &[Episode],
        semantic_memory: &SemanticMemory,
    ) -> Result<(), CompactionError> {
        // Create cues from original episodes
        for episode in episodes {
            let cue = Cue::from_text(&episode.what);
            let results = store.recall(cue).await;

            // Check if semantic memory is retrieved
            let semantic_retrieved = results.iter().any(|(ep, _)| {
                ep.id == semantic_memory.id
            });

            if !semantic_retrieved {
                return Err(CompactionError::RetrievalVerificationFailed {
                    episode_id: episode.id.clone(),
                });
            }
        }
        Ok(())
    }

    /// Compute storage reduction in bytes
    fn compute_storage_reduction(
        &self,
        episodes: &[Episode],
        semantic_memory: &SemanticMemory,
    ) -> u64 {
        let episode_size = std::mem::size_of::<Episode>();
        let semantic_size = std::mem::size_of::<SemanticMemory>();

        let before = (episodes.len() * episode_size) as u64;
        let after = semantic_size as u64;
        before - after
    }
}

pub struct CompactionResult {
    pub episodes_removed: usize,
    pub semantic_memory_id: String,
    pub storage_reduction_bytes: u64,
}
```

3. Safety Mechanisms:
   - Verify reconstruction before deletion
   - Soft delete (mark consolidated) before hard delete
   - Validate retrieval post-compaction
   - Rollback on verification failure

4. Integration with MemoryStore:
   - Add `store_semantic()` method
   - Add `mark_consolidated()` method
   - Add `remove_consolidated_episodes()` method
   - Track consolidated episode IDs for reconstruction

**Acceptance Criteria**:
- [ ] Episodes replaced with semantic memories safely
- [ ] Reconstruction verification ensures no information loss
- [ ] Retrieval verification confirms semantic memories retrievable
- [ ] Soft delete prevents data loss during verification
- [ ] Rollback mechanism on verification failure
- [ ] Storage reduction measured and tracked
- [ ] Metrics track compaction success rate and storage savings
- [ ] Integration tests validate full compaction pipeline

**Files**:
- Create: `engram-core/src/consolidation/compaction.rs`
- Modify: `engram-core/src/store.rs` (add semantic memory storage)
- Create: `engram-core/tests/storage_compaction_tests.rs`

---

## Task 005: Dream Operation (3 days, P2)

**Objective**: Implement `dream()` operation for offline consolidation with enhanced pattern detection using replay buffer.

**Technical Approach**:
1. Create `engram-core/src/consolidation/dream.rs`
2. Leverage existing replay buffer from `completion/consolidation.rs`
3. Implement dream cycle:
```rust
pub struct DreamEngine {
    /// Configuration for dream cycles
    config: DreamConfig,
    /// Replay buffer for memory replay
    replay_buffer: Arc<RwLock<ReplayBuffer>>,
    /// Pattern detector
    pattern_detector: PatternDetector,
}

pub struct DreamConfig {
    /// Duration of dream cycle
    pub dream_duration: Duration,
    /// Replay speed multiplier (10-20x faster than real-time)
    pub replay_speed: f32,
    /// Number of replay iterations per dream cycle
    pub replay_iterations: usize,
    /// Sharp-wave ripple frequency (Hz)
    pub ripple_frequency: f32,
}

impl DreamEngine {
    /// Run dream cycle for offline consolidation
    pub async fn dream(
        &self,
        store: &MemoryStore,
    ) -> Result<DreamOutcome, ConsolidationError> {
        let start = Instant::now();

        // Phase 1: Select episodes for replay
        let episodes = self.select_dream_episodes(store).await?;

        // Phase 2: Replay episodes with ripple dynamics
        let replay_outcome = self.replay_episodes(&episodes).await?;

        // Phase 3: Detect enhanced patterns from replay
        let patterns = self.pattern_detector.detect_patterns_from_replay(&replay_outcome)?;

        // Phase 4: Extract semantic memories
        let semantic_memories = self.extract_semantic_from_patterns(&patterns)?;

        // Phase 5: Apply consolidation
        let compaction_results = self.compact_replayed_episodes(
            store,
            episodes,
            semantic_memories,
        ).await?;

        Ok(DreamOutcome {
            dream_duration: start.elapsed(),
            episodes_replayed: episodes.len(),
            patterns_discovered: patterns.len(),
            semantic_memories_created: compaction_results.len(),
            storage_reduction: compaction_results.iter()
                .map(|r| r.storage_reduction_bytes)
                .sum(),
        })
    }

    /// Select episodes for dream replay based on importance
    async fn select_dream_episodes(
        &self,
        store: &MemoryStore,
    ) -> Result<Vec<Episode>, ConsolidationError> {
        // Prioritize by:
        // 1. Prediction error (high error = needs consolidation)
        // 2. Recent encoding (fresh memories)
        // 3. High confidence (reliable)
        unimplemented!()
    }

    /// Replay episodes with sharp-wave ripple dynamics
    async fn replay_episodes(
        &self,
        episodes: &[Episode],
    ) -> Result<ReplayOutcome, ConsolidationError> {
        let mut replay_buffer = self.replay_buffer.write().await;

        // Simulate sharp-wave ripples
        for iteration in 0..self.config.replay_iterations {
            let ripple_event = ReplayEvent {
                episodes: episodes.to_vec(),
                speed_multiplier: self.config.replay_speed,
                timestamp: Utc::now(),
                ripple_frequency: self.config.ripple_frequency,
                ripple_duration: 100.0, // milliseconds
            };

            // Add to replay buffer
            replay_buffer.push_back(ripple_event);

            // Simulate replay dynamics
            tokio::time::sleep(Duration::from_millis(100)).await;
        }

        Ok(ReplayOutcome {
            replays_completed: self.config.replay_iterations,
            average_speed: self.config.replay_speed,
        })
    }
}

pub struct DreamOutcome {
    pub dream_duration: Duration,
    pub episodes_replayed: usize,
    pub patterns_discovered: usize,
    pub semantic_memories_created: usize,
    pub storage_reduction: u64,
}

struct ReplayOutcome {
    replays_completed: usize,
    average_speed: f32,
}
```

4. Integration with Existing Consolidation:
   - Use `ConsolidationEngine` from `completion/consolidation.rs`
   - Extend `ripple_replay()` method with dream dynamics
   - Apply sharp-wave ripple frequency to replay timing

5. Biological Plausibility:
   - Replay speed 10-20x faster than encoding (matches neuroscience)
   - Sharp-wave ripple frequency 150-250 Hz
   - Hippocampal replay during offline periods (sleep, rest)

**Acceptance Criteria**:
- [ ] Dream cycle runs during offline periods without blocking
- [ ] Replay speed 10-20x faster than real-time
- [ ] Sharp-wave ripples simulated at 150-250 Hz
- [ ] Pattern detection enhanced by multiple replay iterations
- [ ] Storage reduction >50% after dream cycle
- [ ] Retrieval accuracy maintained >95%
- [ ] Dream duration configurable (default 10 minutes)
- [ ] Metrics track dream effectiveness and storage savings

**Files**:
- Create: `engram-core/src/consolidation/dream.rs`
- Modify: `engram-core/src/completion/consolidation.rs` (integrate dream)
- Create: `engram-core/tests/dream_tests.rs`

---

## Task 006: Consolidation Metrics & Observability (1 day, P1)

**Objective**: Implement comprehensive metrics and observability for consolidation system to track effectiveness and debug issues.

**Technical Approach**:
1. Create `engram-core/src/consolidation/metrics.rs`
2. Implement metrics tracking:
```rust
pub struct ConsolidationMetrics {
    // Cycle metrics
    pub successful_cycles: AtomicU64,
    pub failed_cycles: AtomicU64,
    pub skipped_cycles: AtomicU64,
    pub total_consolidation_time_ms: AtomicU64,

    // Pattern detection metrics
    pub patterns_detected: AtomicU64,
    pub patterns_rejected: AtomicU64, // Below statistical significance
    pub average_pattern_strength: AtomicF32,

    // Semantic extraction metrics
    pub semantic_memories_created: AtomicU64,
    pub average_semantic_confidence: AtomicF32,

    // Storage metrics
    pub episodes_consolidated: AtomicU64,
    pub storage_reduction_bytes: AtomicU64,
    pub last_reduction_ratio: AtomicF32,

    // Dream metrics
    pub dream_cycles_completed: AtomicU64,
    pub total_dream_time_ms: AtomicU64,
    pub episodes_replayed: AtomicU64,
}

impl ConsolidationMetrics {
    /// Record consolidation cycle
    pub fn record_cycle(&self, outcome: &ConsolidationOutcome) {
        self.successful_cycles.fetch_add(1, Ordering::Relaxed);
        self.patterns_detected.fetch_add(outcome.patterns_extracted as u64, Ordering::Relaxed);
        self.semantic_memories_created.fetch_add(outcome.semantic_memories_created as u64, Ordering::Relaxed);
        self.episodes_consolidated.fetch_add(outcome.episodes_consolidated as u64, Ordering::Relaxed);
        self.storage_reduction_bytes.fetch_add(
            self.compute_storage_reduction(outcome),
            Ordering::Relaxed,
        );
        self.last_reduction_ratio.store(outcome.reduction_ratio, Ordering::Relaxed);

        // Export to streaming metrics
        crate::metrics::record_gauge("consolidation_cycles_total", self.successful_cycles.load(Ordering::Relaxed) as f64);
        crate::metrics::record_gauge("consolidation_patterns_detected_total", self.patterns_detected.load(Ordering::Relaxed) as f64);
        crate::metrics::record_gauge("consolidation_storage_reduction_bytes", self.storage_reduction_bytes.load(Ordering::Relaxed) as f64);
        crate::metrics::record_gauge("consolidation_reduction_ratio", f64::from(self.last_reduction_ratio.load(Ordering::Relaxed)));
    }

    /// Get consolidation status snapshot
    pub fn snapshot(&self) -> ConsolidationSnapshot {
        ConsolidationSnapshot {
            successful_cycles: self.successful_cycles.load(Ordering::Relaxed),
            failed_cycles: self.failed_cycles.load(Ordering::Relaxed),
            skipped_cycles: self.skipped_cycles.load(Ordering::Relaxed),
            patterns_detected: self.patterns_detected.load(Ordering::Relaxed),
            semantic_memories_created: self.semantic_memories_created.load(Ordering::Relaxed),
            episodes_consolidated: self.episodes_consolidated.load(Ordering::Relaxed),
            storage_reduction_bytes: self.storage_reduction_bytes.load(Ordering::Relaxed),
            reduction_ratio: self.last_reduction_ratio.load(Ordering::Relaxed),
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct ConsolidationSnapshot {
    pub successful_cycles: u64,
    pub failed_cycles: u64,
    pub skipped_cycles: u64,
    pub patterns_detected: u64,
    pub semantic_memories_created: u64,
    pub episodes_consolidated: u64,
    pub storage_reduction_bytes: u64,
    pub reduction_ratio: f32,
}
```

3. HTTP API Endpoints:
   - `GET /api/v1/consolidation/status`: Current consolidation status
   - `GET /api/v1/consolidation/metrics`: Detailed metrics snapshot
   - `POST /api/v1/consolidation/trigger`: Manually trigger consolidation
   - `POST /api/v1/consolidation/pause`: Pause consolidation
   - `POST /api/v1/consolidation/resume`: Resume consolidation

4. Streaming Telemetry:
   - Integrate with existing `MetricsRegistry`
   - Export consolidation metrics to streaming aggregator
   - Add consolidation section to `/api/v1/system/metrics` response

**Acceptance Criteria**:
- [ ] Metrics track all consolidation operations
- [ ] Metrics exported to streaming telemetry
- [ ] HTTP API exposes consolidation status and metrics
- [ ] Manual trigger/pause/resume controls work
- [ ] Metrics dashboard shows consolidation health
- [ ] Documentation explains metric meanings and thresholds

**Files**:
- Create: `engram-core/src/consolidation/metrics.rs`
- Modify: `engram-core/src/metrics/mod.rs` (integrate consolidation metrics)
- Modify: `engram-cli/src/api.rs` (add consolidation endpoints)
- Docs: `docs/operations/consolidation.md`

---

## Task 007: Production Validation & Tuning (2 days, P0)

**Objective**: Validate consolidation system meets acceptance criteria (>50% storage reduction, >95% retrieval accuracy) and tune parameters for production.

**Technical Approach**:
1. Create comprehensive validation suite:
```rust
// engram-core/tests/consolidation_validation.rs

#[tokio::test]
async fn test_storage_reduction_target() {
    // Setup: Create memory store with 1000 episodes
    let store = MemoryStore::new();
    populate_test_data(&store, 1000).await;

    let initial_storage = store.storage_size_bytes().await;

    // Run consolidation cycle
    let outcome = store.consolidate_async().await.unwrap();

    let final_storage = store.storage_size_bytes().await;
    let reduction_ratio = 1.0 - (final_storage as f32 / initial_storage as f32);

    // Validate: >50% storage reduction
    assert!(reduction_ratio > 0.5, "Storage reduction {:.2}% below target 50%", reduction_ratio * 100.0);
}

#[tokio::test]
async fn test_retrieval_accuracy_maintained() {
    let store = MemoryStore::new();
    let test_episodes = populate_test_data(&store, 500).await;

    // Measure baseline retrieval accuracy
    let baseline_accuracy = measure_retrieval_accuracy(&store, &test_episodes).await;

    // Run consolidation
    store.consolidate_async().await.unwrap();

    // Measure post-consolidation accuracy
    let post_accuracy = measure_retrieval_accuracy(&store, &test_episodes).await;

    // Validate: >95% accuracy maintained
    assert!(post_accuracy >= 0.95, "Retrieval accuracy {:.2}% below target 95%", post_accuracy * 100.0);
    assert!(post_accuracy >= baseline_accuracy * 0.95, "Accuracy degraded by more than 5%");
}

async fn measure_retrieval_accuracy(
    store: &MemoryStore,
    test_episodes: &[Episode],
) -> f32 {
    let mut correct = 0;
    let mut total = 0;

    for episode in test_episodes {
        let cue = Cue::from_text(&episode.what);
        let results = store.recall(cue).await;

        total += 1;
        if results.iter().any(|(ep, _)| {
            embedding_similarity(&ep.embedding, &episode.embedding) > 0.8
        }) {
            correct += 1;
        }
    }

    correct as f32 / total as f32
}
```

2. Parameter Tuning:
   - Similarity threshold: Test range 0.6-0.9, optimize for retrieval accuracy
   - Min cluster size: Test 2-5, balance pattern quality vs. quantity
   - Abstraction level: Test 0.0-1.0, optimize for storage vs. accuracy tradeoff
   - Dream replay iterations: Test 1-10, optimize for pattern discovery

3. Stress Testing:
   - Run consolidation on 100K episodes
   - Verify no memory leaks over 1000 cycles
   - Test concurrent consolidation and recall
   - Validate graceful degradation under system pressure

4. Human Evaluation:
   - Sample 100 consolidated memories
   - Verify semantic memories preserve meaning
   - Validate schemas capture essential structure
   - Confirm reconstructed episodes are plausible

**Acceptance Criteria**:
- [ ] Storage reduction >50% on test datasets
- [ ] Retrieval accuracy >95% post-consolidation
- [ ] Consolidation completes within configured time budget
- [ ] No memory leaks over 1000 cycles
- [ ] Concurrent consolidation and recall work correctly
- [ ] Human evaluation confirms semantic quality
- [ ] Parameter tuning documented with rationale
- [ ] Production deployment guide complete

**Files**:
- Create: `engram-core/tests/consolidation_validation.rs`
- Create: `engram-core/tests/consolidation_stress_tests.rs`
- Docs: `docs/operations/consolidation_tuning.md`
- Docs: `docs/operations/consolidation_deployment.md`

---

# SUMMARY

## Milestone 5 Deliverables

1. **Query Executor Core** (Task 001): Production query executor with confidence propagation
2. **Evidence Aggregation** (Task 002): Lock-free evidence combination with circular dependency detection
3. **Uncertainty Tracking** (Task 003): System-wide uncertainty tracking from all sources
4. **Confidence Calibration** (Task 004): Empirical calibration framework ensuring accuracy
5. **SMT Verification** (Task 005): Formal verification of probability operations
6. **Query Operations** (Task 006): Probabilistic AND/OR/NOT with sub-ms latency
7. **Integration & Validation** (Task 007): End-to-end testing and production readiness

**Total Duration**: 14 days
**Critical Path**: 001 → 002 → 003 → 004 → 005 → 006 → 007

## Milestone 6 Deliverables

1. **Consolidation Scheduler** (Task 001): Async consolidation without blocking recall
2. **Pattern Detection** (Task 002): Unsupervised statistical pattern detection
3. **Semantic Extraction** (Task 003): Transform patterns into semantic memories
4. **Storage Compaction** (Task 004): Safe episode replacement with verification
5. **Dream Operation** (Task 005): Offline consolidation with replay dynamics
6. **Metrics & Observability** (Task 006): Comprehensive consolidation monitoring
7. **Validation & Tuning** (Task 007): Parameter tuning and production deployment

**Total Duration**: 16 days
**Critical Path**: 001 → 002 → 003 → 004 → 005 → 006 → 007

## Success Metrics

**Milestone 5**:
- Query latency: <1ms P95
- Calibration error: <5%
- Property test pass rate: >99.9%
- Confidence correlation with accuracy: >0.9

**Milestone 6**:
- Storage reduction: >50%
- Retrieval accuracy: >95%
- Consolidation cycles: Zero blocking
- Pattern discovery: >80% of test datasets

## Risk Mitigation

**Milestone 5**:
- Complexity managed through incremental development
- Performance validated with benchmarks at each stage
- Correctness guaranteed through SMT verification and property tests
- Integration maintained through backward-compatible API extensions

**Milestone 6**:
- Consolidation interruptibility prevents recall blocking
- Verification steps ensure no data loss during compaction
- Statistical significance filtering prevents spurious patterns
- Human evaluation validates semantic quality

## File Structure

```
engram-core/src/
├── query/
│   ├── mod.rs (existing types)
│   ├── executor.rs (Task 5.001)
│   ├── evidence_aggregator.rs (Task 5.002)
│   ├── dependency_graph.rs (Task 5.002)
│   ├── uncertainty_tracker.rs (Task 5.003)
│   ├── calibration.rs (Task 5.004)
│   ├── smt_verification.rs (Task 5.005)
│   └── query_combinators.rs (Task 5.006)
├── consolidation/
│   ├── mod.rs (new)
│   ├── scheduler.rs (Task 6.001)
│   ├── pattern_detector.rs (Task 6.002)
│   ├── clustering.rs (Task 6.002)
│   ├── semantic_extraction.rs (Task 6.003)
│   ├── schema.rs (Task 6.003)
│   ├── compaction.rs (Task 6.004)
│   ├── dream.rs (Task 6.005)
│   └── metrics.rs (Task 6.006)
├── store.rs (integrate both milestones)
└── memory.rs (add SemanticMemory type)

engram-cli/src/
└── api.rs (add HTTP endpoints)

docs/operations/
├── probabilistic_queries.md
├── consolidation.md
├── consolidation_tuning.md
└── consolidation_deployment.md
```

## Dependencies

**External Crates**:
- `z3 = "0.12"` (optional, for SMT verification)
- `statrs = "0.16"` (statistical tests)
- `crossbeam-epoch = "0.9"` (lock-free memory management)

**Internal Dependencies**:
- Milestone 3: Activation spreading, confidence aggregation
- Milestone 1: HNSW index, decay functions, probabilistic query primitives
- Milestone 0: Confidence type, error infrastructure

## Next Steps

1. Create individual task files in `roadmap/milestone-5/` and `roadmap/milestone-6/`
2. Review task specifications with technical leads
3. Begin implementation with Task 5.001 (Query Executor Core)
4. Run SMT verification in CI for all probability operations
5. Deploy consolidation scheduler in production with conservative parameters
6. Monitor metrics and tune parameters based on real workloads

---

**Note**: This roadmap provides implementation-ready specifications. Every task includes precise file paths, code examples, integration points, acceptance criteria, and risk mitigation. An engineer can read any task file and begin coding immediately.
