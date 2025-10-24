# Task 002: Hippocampal CA3/CA1 Dynamics

**Status:** Pending
**Priority:** P0 (Critical Path)
**Estimated Effort:** 2 days
**Dependencies:** Task 001 (Reconstruction Primitives)

## Objective

Enhance existing hippocampal CA3 attractor dynamics for production pattern completion, implementing biologically-plausible convergence within theta rhythm constraints (7 iterations max). Add CA1 output gating with confidence thresholds and DG pattern separation validation. Achieve >95% convergence rate and match Hopfield network energy minimization properties.

## Integration Points

**Extends:**
- `/engram-core/src/completion/hippocampal.rs` - Existing HippocampalCompletion implementation
- `/engram-core/src/completion/context.rs` - EntorhinalContext for grid cells

**Uses:**
- `/engram-core/src/completion/field_reconstruction.rs` - Field-level reconstructions from Task 001
- `/engram-core/src/embedding/similarity.rs` - SIMD cosine similarity
- `nalgebra` - Matrix operations for CA3 recurrent weights

**Creates:**
- `/engram-core/src/completion/attractor_dynamics.rs` - CA3 convergence logic separated for testing
- `/engram-core/src/completion/ca1_gating.rs` - Output gating and confidence thresholding
- `/engram-core/tests/hippocampal_dynamics_tests.rs` - Biological plausibility tests

## Detailed Specification

### 1. CA3 Attractor Dynamics Enhancement

```rust
// /engram-core/src/completion/attractor_dynamics.rs

use nalgebra::{DMatrix, DVector};
use crate::Confidence;

/// CA3 attractor network for pattern completion
pub struct CA3Attractor {
    /// Recurrent weight matrix (768x768 for embeddings)
    weights: DMatrix<f32>,

    /// Current activation state
    state: DVector<f32>,

    /// Previous state for convergence detection
    previous_state: DVector<f32>,

    /// Sparsity level (fraction of active neurons)
    sparsity: f32,

    /// Maximum iterations (theta rhythm constraint)
    max_iterations: usize,

    /// Convergence threshold (L2 norm difference)
    convergence_threshold: f32,

    /// Current iteration counter
    iteration: usize,

    /// Convergence achieved flag
    converged: bool,

    /// Attractor energy (for Hopfield validation)
    energy: f32,
}

impl CA3Attractor {
    /// Create new CA3 attractor with Hebbian-initialized weights
    pub fn new(sparsity: f32, max_iterations: usize) -> Self;

    /// Update weights with Hebbian learning from new pattern
    ///
    /// W_new = W_old + (1/N) * (pattern * pattern^T - I)
    /// Ensures weight matrix is symmetric (Hopfield requirement)
    pub fn learn_pattern(&mut self, pattern: &DVector<f32>);

    /// Run attractor dynamics until convergence or max iterations
    ///
    /// Returns (converged_state, num_iterations, converged_flag)
    pub fn converge(&mut self, input: DVector<f32>) -> (DVector<f32>, usize, bool);

    /// Single step of attractor dynamics
    ///
    /// s(t+1) = sigma(W * s(t))
    /// where sigma is sigmoid with k-winner-take-all sparsity
    fn step(&mut self) -> f32;

    /// Apply k-winner-take-all sparsity constraint
    ///
    /// Keeps top k% neurons active, zeros rest
    fn apply_sparsity(&mut self);

    /// Compute Hopfield energy: E = -0.5 * s^T * W * s
    ///
    /// Should decrease monotonically during convergence
    pub fn compute_energy(&self) -> f32;

    /// Check if dynamics have converged
    ///
    /// Converged when ||s(t+1) - s(t)|| < threshold
    pub fn has_converged(&self) -> bool;

    /// Reset attractor to initial state
    pub fn reset(&mut self);

    /// Get convergence statistics
    pub fn convergence_stats(&self) -> ConvergenceStats;
}

/// Statistics from attractor convergence
#[derive(Debug, Clone)]
pub struct ConvergenceStats {
    pub iterations: usize,
    pub converged: bool,
    pub final_energy: f32,
    pub energy_delta: f32,
    pub state_change: f32,
}

impl CA3Attractor {
    pub fn converge(&mut self, input: DVector<f32>) -> (DVector<f32>, usize, bool) {
        self.state = input;
        self.iteration = 0;
        self.converged = false;

        let initial_energy = self.compute_energy();

        for iter in 0..self.max_iterations {
            self.iteration = iter;
            self.previous_state = self.state.clone();

            // Hopfield-like update: s(t+1) = sigma(W * s(t))
            let activation = &self.weights * &self.state;

            // Sigmoid activation
            for i in 0..activation.len() {
                self.state[i] = 1.0 / (1.0 + (-activation[i]).exp());
            }

            // Apply sparsity constraint (k-winner-take-all)
            self.apply_sparsity();

            // Compute energy for monitoring
            self.energy = self.compute_energy();

            // Check convergence
            let state_diff = (&self.state - &self.previous_state).norm();
            if state_diff < self.convergence_threshold {
                self.converged = true;
                break;
            }
        }

        (self.state.clone(), self.iteration + 1, self.converged)
    }

    fn apply_sparsity(&mut self) {
        let k = (self.state.len() as f32 * self.sparsity).round() as usize;
        let mut activations: Vec<(usize, f32)> = self.state
            .iter()
            .enumerate()
            .map(|(i, &v)| (i, v))
            .collect();

        // Sort by activation strength (descending)
        activations.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Zero out all but top-k
        for i in k..activations.len() {
            self.state[activations[i].0] = 0.0;
        }
    }

    fn compute_energy(&self) -> f32 {
        // Hopfield energy: E = -0.5 * s^T * W * s
        let ws = &self.weights * &self.state;
        -0.5 * self.state.dot(&ws)
    }
}
```

### 2. CA1 Output Gating

```rust
// /engram-core/src/completion/ca1_gating.rs

use crate::{Confidence, Episode};
use crate::completion::{CompletedEpisode, CompletionError};
use nalgebra::DVector;

/// CA1 output gating with confidence thresholding
pub struct CA1Gate {
    /// Minimum confidence threshold for output (default: 0.7)
    threshold: Confidence,

    /// Plausibility scoring for hallucination detection
    plausibility_checker: PlausibilityChecker,
}

impl CA1Gate {
    /// Create new CA1 gate with threshold
    pub fn new(threshold: Confidence) -> Self;

    /// Gate CA3 output based on confidence and plausibility
    ///
    /// Returns Ok(completed) if passes threshold, Err(LowConfidence) otherwise
    pub fn gate_output(
        &self,
        ca3_output: &DVector<f32>,
        convergence_stats: &ConvergenceStats,
        field_reconstructions: &HashMap<String, ReconstructedField>,
    ) -> Result<CompletedEpisode, CompletionError>;

    /// Compute completion confidence from multiple factors
    ///
    /// Factors:
    /// 1. CA3 convergence speed (faster = higher confidence)
    /// 2. Energy reduction (deeper attractor = higher confidence)
    /// 3. Field consensus strength (agreement = higher confidence)
    /// 4. Plausibility score (coherent = higher confidence)
    fn compute_completion_confidence(
        &self,
        convergence_stats: &ConvergenceStats,
        field_consensus: f32,
        plausibility: f32,
    ) -> Confidence;

    /// Check if completion passes threshold
    pub fn passes_threshold(&self, confidence: Confidence) -> bool {
        confidence >= self.threshold
    }
}

/// Plausibility checker for detecting implausible reconstructions
pub struct PlausibilityChecker {
    /// HNSW index for neighborhood consistency
    hnsw_index: Option<Arc<HnswIndex>>,

    /// Minimum neighborhood agreement for plausibility (default: 0.6)
    min_neighborhood_agreement: f32,
}

impl PlausibilityChecker {
    /// Score plausibility of reconstructed embedding
    ///
    /// Returns 0.0-1.0 score based on:
    /// 1. Similarity to nearest neighbors in HNSW
    /// 2. Consistency with local embedding manifold
    /// 3. Not in "nowhere" region (isolated point)
    pub fn score_plausibility(&self, embedding: &[f32; 768]) -> f32;

    /// Check if embedding is in sparse region (potential hallucination)
    fn is_isolated(&self, embedding: &[f32; 768]) -> bool;
}
```

### 3. Theta Rhythm Constraint Validation

**Biological Constraint:**
- Theta rhythm: 4-8 Hz (125-250ms period)
- Single theta cycle: ~140ms average
- 7 gamma cycles per theta cycle (40Hz gamma * 140ms ≈ 5-6 cycles)
- Max 7 iterations for convergence

**Implementation:**
```rust
// Configuration validation
impl CA3Attractor {
    pub fn validate_theta_constraint(&self) -> bool {
        // Max iterations must respect theta rhythm
        self.max_iterations <= 7
    }

    pub fn expected_duration_ms(&self) -> f32 {
        // Assuming ~20ms per iteration (gamma cycle)
        self.max_iterations as f32 * 20.0
    }
}

// Production monitoring
pub struct ThetaRhythmMonitor {
    pub iterations_histogram: [usize; 8], // 0-7 iterations
    pub avg_iterations: f32,
    pub convergence_rate: f32,
}

impl ThetaRhythmMonitor {
    pub fn record_completion(&mut self, iterations: usize, converged: bool);

    pub fn violates_theta_constraint(&self) -> bool {
        // Warn if >5% of completions exceed 7 iterations
        let total: usize = self.iterations_histogram.iter().sum();
        let violations = self.iterations_histogram[7];
        violations as f32 / total as f32 > 0.05
    }
}
```

## Acceptance Criteria

1. **Convergence Performance:**
   - Achieve >95% convergence rate within 7 iterations on test datasets
   - Average iterations <5 for well-formed patterns
   - Convergence failure (<7 iterations) returns explicit error with stats

2. **Biological Plausibility:**
   - Hopfield energy decreases monotonically during convergence (>99% of cases)
   - Energy reduction correlates >0.85 with reconstruction accuracy
   - Theta rhythm constraint (max 7 iterations) enforced and validated

3. **CA1 Gating Accuracy:**
   - Threshold correctly filters low-confidence completions (precision >90%)
   - Completion confidence correlates >0.80 with reconstruction accuracy
   - Plausibility check catches >85% of implausible reconstructions

4. **Performance:**
   - Single CA3 iteration <3ms (768-dim embedding, 5% sparsity)
   - Full convergence (7 iterations) <20ms P95
   - Weight update (Hebbian learning) <5ms per pattern

5. **Mathematical Correctness:**
   - Weight matrix remains symmetric after Hebbian updates (Hopfield requirement)
   - Sparsity constraint maintains exact k-winner-take-all (no approximation errors)
   - Energy computation matches analytical Hopfield formula

## Testing Strategy

### Unit Tests

```rust
#[test]
fn test_ca3_convergence_with_learned_pattern() {
    let mut ca3 = CA3Attractor::new(0.05, 7);

    // Learn pattern
    let pattern = DVector::from_fn(768, |i, _| if i < 38 { 1.0 } else { 0.0 }); // 5% sparse
    ca3.learn_pattern(&pattern);

    // Test convergence from partial cue (30% overlap)
    let mut partial = pattern.clone();
    for i in 38..768 {
        partial[i] = rand::random::<f32>() * 0.1; // Noise
    }

    let (converged, iters, success) = ca3.converge(partial);
    assert!(success, "Should converge within 7 iterations");
    assert!(iters <= 7, "Theta rhythm constraint violated");

    // Check converged to original pattern
    let similarity = cosine_similarity(&converged, &pattern);
    assert!(similarity > 0.9, "Should converge to learned pattern");
}

#[test]
fn test_hopfield_energy_decreases() {
    let mut ca3 = CA3Attractor::new(0.05, 7);
    let pattern = random_sparse_vector(768, 0.05);
    ca3.learn_pattern(&pattern);

    let input = noisy_pattern(&pattern, 0.3);
    let mut energies = Vec::new();

    // Track energy at each iteration
    ca3.state = input;
    for _ in 0..7 {
        energies.push(ca3.compute_energy());
        ca3.step();
    }

    // Energy should decrease (or stay same at attractor)
    for i in 1..energies.len() {
        assert!(
            energies[i] <= energies[i - 1] + 1e-6, // Allow small numerical error
            "Energy increased: {} -> {}", energies[i - 1], energies[i]
        );
    }
}

#[test]
fn test_weight_matrix_symmetry() {
    let mut ca3 = CA3Attractor::new(0.05, 7);
    let pattern = random_sparse_vector(768, 0.05);
    ca3.learn_pattern(&pattern);

    // Check symmetry: W[i,j] == W[j,i]
    for i in 0..768 {
        for j in (i+1)..768 {
            assert!((ca3.weights[(i, j)] - ca3.weights[(j, i)]).abs() < 1e-6);
        }
    }
}

#[test]
fn test_sparsity_constraint_exact() {
    let mut ca3 = CA3Attractor::new(0.05, 7);
    let input = DVector::from_fn(768, |_, _| rand::random());
    ca3.state = input;

    ca3.apply_sparsity();

    // Count non-zero activations
    let active = ca3.state.iter().filter(|&&v| v > 0.0).count();
    let expected = (768.0 * 0.05).round() as usize;
    assert_eq!(active, expected, "Sparsity constraint not exact");
}

#[test]
fn test_ca1_gating_threshold() {
    let gate = CA1Gate::new(Confidence::exact(0.7));

    // High-confidence completion should pass
    let high_conf_stats = ConvergenceStats {
        iterations: 3,
        converged: true,
        final_energy: -5.0,
        energy_delta: 3.0,
        state_change: 0.005,
    };
    let high_conf = gate.compute_completion_confidence(&high_conf_stats, 0.9, 0.85);
    assert!(gate.passes_threshold(high_conf));

    // Low-confidence completion should fail
    let low_conf_stats = ConvergenceStats {
        iterations: 7,
        converged: false,
        final_energy: -1.0,
        energy_delta: 0.5,
        state_change: 0.05,
    };
    let low_conf = gate.compute_completion_confidence(&low_conf_stats, 0.4, 0.5);
    assert!(!gate.passes_threshold(low_conf));
}
```

### Integration Tests

```rust
#[test]
fn test_end_to_end_pattern_completion() {
    // Create HippocampalCompletion with CA3 + CA1
    let mut completion = HippocampalCompletion::new(CompletionConfig::default());

    // Learn 10 breakfast patterns
    let breakfast_patterns = generate_breakfast_episodes(10);
    completion.update(&breakfast_patterns);

    // Create partial cue (30% overlap)
    let partial = create_partial_breakfast(0.3);

    // Complete pattern
    let result = completion.complete(&partial);
    assert!(result.is_ok());

    let completed = result.unwrap();
    assert!(completed.completion_confidence.raw() > 0.7);

    // Verify reconstruction accuracy
    let ground_truth = &breakfast_patterns[0];
    let accuracy = field_accuracy(&completed.episode, ground_truth);
    assert!(accuracy > 0.85);
}
```

### Property-Based Tests

```rust
proptest! {
    #[test]
    fn prop_convergence_within_theta_constraint(
        pattern in arbitrary_sparse_vector(768, 0.05),
        noise_level in 0.0f32..0.5
    ) {
        let mut ca3 = CA3Attractor::new(0.05, 7);
        ca3.learn_pattern(&pattern);

        let noisy = add_noise(&pattern, noise_level);
        let (_, iters, _) = ca3.converge(noisy);

        // Property: iterations never exceed theta constraint
        assert!(iters <= 7);
    }

    #[test]
    fn prop_energy_non_increasing(
        pattern in arbitrary_sparse_vector(768, 0.05),
        input in arbitrary_vector(768)
    ) {
        let mut ca3 = CA3Attractor::new(0.05, 7);
        ca3.learn_pattern(&pattern);

        ca3.state = input;
        let mut prev_energy = ca3.compute_energy();

        for _ in 0..7 {
            ca3.step();
            let energy = ca3.compute_energy();
            // Property: energy never increases
            assert!(energy <= prev_energy + 1e-6);
            prev_energy = energy;
        }
    }
}
```

## Biological Validation

**Match Against Literature:**
1. Hopfield (1982): Energy minimization in attractor networks
2. Marr (1971): Sparse coding in CA3 for pattern separation
3. McNaughton & Morris (1987): CA3 as autoassociative memory
4. Buzsáki (2002): Theta rhythm constraints on memory processes

**Empirical Validation:**
- Convergence speed vs cue overlap matches empirical data (Treves & Rolls, 1994)
- Pattern separation capacity scales with sparsity (Leutgeb et al., 2007)
- Energy landscape structure consistent with Hopfield predictions

## Integration with Confidence Calibration (Task 006)

### Multi-Factor Confidence Signals from CA3
Task 006 research (Koriat's Cue-Utilization Framework, 1997) identifies multiple confidence cues that map directly to CA3 dynamics:

**Intrinsic Cues (Pattern Properties):**
- Pattern strength implemented as weight matrix magnitude
- Cue overlap measured by similarity threshold filtering

**Extrinsic Cues (Learning Conditions):**
- Number of source episodes tracked in Hebbian weight updates
- Pattern age implicit in consolidation from M6

**Mnemonic Cues (Retrieval Fluency):**
- **Convergence speed:** Fewer iterations = higher confidence (Task 006: convergence_weight = 0.3)
- **Energy reduction:** Deeper attractor basin = higher confidence (Task 006: energy_weight = 0.25)

### ConvergenceStats → Confidence Computation

The `ConvergenceStats` struct from this task provides critical inputs to Task 006:
```rust
pub struct ConvergenceStats {
    pub iterations: usize,           // → convergence_factor = 1 - (iters / max_iters)
    pub converged: bool,              // → gates CA1 output
    pub final_energy: f32,            // → used in energy_factor
    pub energy_delta: f32,            // → deeper basin = higher confidence
    pub state_change: f32,            // → convergence detection
}
```

**Calibration Target (Task 006):** Multi-factor combination should achieve <8% calibration error with Brier score validation.

## Risk Mitigation

**Risk: Convergence failures exceed 5%**
- **Mitigation:** Empirical tuning of sparsity and learning rate on validation sets
- **Contingency:** Adaptive max_iterations based on pattern complexity

**Risk: Hopfield energy increases (theoretical violation)**
- **Mitigation:** Rigorous unit tests and property-based verification
- **Contingency:** Fall back to fixed-point iteration without energy monitoring

**Risk: CA1 threshold too conservative (many false negatives)**
- **Mitigation:** A/B testing different thresholds; ROC curve analysis
- **Contingency:** Configurable threshold per memory space

## Implementation Notes

1. Use `nalgebra` sparse matrix if memory becomes issue (>100K patterns)
2. Pre-allocate state vectors to avoid iteration allocations
3. SIMD-optimize matrix-vector multiply for CA3 update
4. Cache Hebbian outer products for batch pattern learning
5. Monitor energy landscape in production for drift detection

## Success Criteria Validation

- [ ] Convergence rate >95% within 7 iterations
- [ ] Hopfield energy decreases monotonically >99% of cases
- [ ] Completion confidence correlates >0.80 with accuracy
- [ ] CA1 gating precision >90%
- [ ] Single iteration <3ms, full convergence <20ms P95
- [ ] Weight matrix remains symmetric after all updates
- [ ] All biological validation criteria met
