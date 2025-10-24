//! CA3 attractor network for pattern completion via Hopfield dynamics.
//!
//! Implements autoassociative memory with energy minimization following
//! Hopfield (1982), constrained to biologically-plausible theta rhythm
//! timing (max 7 iterations ≈ 140ms). Uses k-winner-take-all sparsity
//! matching CA3 sparse coding (Marr, 1971; ~5% active neurons).

use crate::Confidence;
use nalgebra::{DMatrix, DVector};
use std::time::Duration;

/// CA3 attractor network for pattern completion via Hopfield dynamics
pub struct CA3Attractor {
    /// Recurrent weight matrix (768x768 for embeddings)
    weights: DMatrix<f32>,

    /// Current activation state
    state: DVector<f32>,

    /// Previous state for convergence detection
    previous_state: DVector<f32>,

    /// Sparsity level (fraction of active neurons, default: 0.05)
    sparsity: f32,

    /// Maximum iterations (theta rhythm constraint, default: 7)
    max_iterations: usize,

    /// Convergence threshold (L2 norm difference, default: 0.01)
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
    ///
    /// # Arguments
    /// * `sparsity` - Fraction of active neurons (e.g., 0.05 for 5%)
    /// * `max_iterations` - Maximum iterations for convergence (theta constraint: ≤7)
    #[must_use]
    pub fn new(sparsity: f32, max_iterations: usize) -> Self {
        let size = 768;
        Self {
            weights: DMatrix::zeros(size, size),
            state: DVector::zeros(size),
            previous_state: DVector::zeros(size),
            sparsity,
            max_iterations,
            convergence_threshold: 0.01,
            iteration: 0,
            converged: false,
            energy: 0.0,
        }
    }

    /// Create CA3 attractor with custom convergence threshold
    #[must_use]
    pub fn with_threshold(
        sparsity: f32,
        max_iterations: usize,
        convergence_threshold: f32,
    ) -> Self {
        let size = 768;
        Self {
            weights: DMatrix::zeros(size, size),
            state: DVector::zeros(size),
            previous_state: DVector::zeros(size),
            sparsity,
            max_iterations,
            convergence_threshold,
            iteration: 0,
            converged: false,
            energy: 0.0,
        }
    }

    /// Update weights with Hebbian learning from new pattern
    ///
    /// Hebbian rule: ΔW = (1/N) * (pattern * pattern^T - I)
    /// Ensures weight matrix is symmetric (Hopfield requirement)
    pub fn learn_pattern(&mut self, pattern: &DVector<f32>) {
        let n = pattern.len() as f32;
        let learning_rate = 1.0 / n;

        // Outer product: pattern * pattern^T
        let update = pattern * pattern.transpose() * learning_rate;

        // Add to weights (Hebbian accumulation)
        self.weights += update;

        // Remove self-connections (diagonal = 0)
        for i in 0..self.weights.nrows() {
            self.weights[(i, i)] = 0.0;
        }

        // Ensure symmetry (Hopfield requirement)
        for i in 0..self.weights.nrows() {
            for j in (i + 1)..self.weights.ncols() {
                let avg = f32::midpoint(self.weights[(i, j)], self.weights[(j, i)]);
                self.weights[(i, j)] = avg;
                self.weights[(j, i)] = avg;
            }
        }
    }

    /// Run attractor dynamics until convergence or max iterations
    ///
    /// Returns (converged_state, num_iterations, converged_flag)
    pub fn converge(&mut self, input: DVector<f32>) -> (DVector<f32>, usize, bool) {
        self.state = input;
        self.iteration = 0;
        self.converged = false;

        for iter in 0..self.max_iterations {
            self.iteration = iter;
            self.previous_state = self.state.clone();

            // Single step of Hopfield dynamics
            self.step_internal();

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

    /// Single step of attractor dynamics
    ///
    /// s(t+1) = sigma(W * s(t))
    /// where sigma is sigmoid with k-winner-take-all sparsity
    fn step_internal(&mut self) {
        // Hopfield-like update: activation = W * s(t)
        let activation = &self.weights * &self.state;

        // Apply sigmoid activation
        for i in 0..activation.len() {
            self.state[i] = 1.0 / (1.0 + (-activation[i]).exp());
        }

        // Apply sparsity constraint (k-winner-take-all)
        self.apply_sparsity();
    }

    /// Apply k-winner-take-all sparsity constraint
    ///
    /// Keeps top k% neurons active, zeros rest
    fn apply_sparsity(&mut self) {
        let k = (self.state.len() as f32 * self.sparsity).round() as usize;
        let mut activations: Vec<(usize, f32)> = self
            .state
            .iter()
            .enumerate()
            .map(|(i, &v)| (i, v))
            .collect();

        // Sort by activation strength (descending)
        activations.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Zero out all but top-k
        for activation in activations.iter().skip(k) {
            self.state[activation.0] = 0.0;
        }
    }

    /// Compute Hopfield energy: E = -0.5 * s^T * W * s
    ///
    /// Should decrease monotonically during convergence
    #[must_use]
    pub fn compute_energy(&self) -> f32 {
        let ws = &self.weights * &self.state;
        -0.5 * self.state.dot(&ws)
    }

    /// Check if dynamics have converged
    #[must_use]
    pub const fn has_converged(&self) -> bool {
        self.converged
    }

    /// Reset attractor to initial state
    pub fn reset(&mut self) {
        self.state = DVector::zeros(768);
        self.previous_state = DVector::zeros(768);
        self.iteration = 0;
        self.converged = false;
        self.energy = 0.0;
    }

    /// Get convergence statistics
    #[must_use]
    pub const fn convergence_stats(&self) -> ConvergenceStats {
        ConvergenceStats {
            iterations: self.iteration,
            converged: self.converged,
            final_energy: self.energy,
            energy_delta: 0.0, // Computed externally if needed
            state_change: 0.0, // Computed externally if needed
        }
    }

    /// Validate theta rhythm constraint (max 7 iterations)
    #[must_use]
    pub const fn validate_theta_constraint(&self) -> bool {
        self.max_iterations <= 7
    }

    /// Expected duration in milliseconds (assuming ~20ms per gamma cycle)
    #[must_use]
    pub const fn expected_duration_ms(&self) -> f32 {
        self.max_iterations as f32 * 20.0
    }

    /// Get current state (for external inspection)
    #[must_use]
    pub const fn get_state(&self) -> &DVector<f32> {
        &self.state
    }

    /// Get weight matrix (for external inspection)
    #[must_use]
    pub const fn get_weights(&self) -> &DMatrix<f32> {
        &self.weights
    }
}

/// Statistics from attractor convergence
#[derive(Debug, Clone, Copy)]
pub struct ConvergenceStats {
    /// Number of iterations taken
    pub iterations: usize,
    /// Whether convergence was achieved
    pub converged: bool,
    /// Final Hopfield energy
    pub final_energy: f32,
    /// Energy reduction from initial to final
    pub energy_delta: f32,
    /// Final state change magnitude
    pub state_change: f32,
}

impl ConvergenceStats {
    /// Convert convergence stats to completion confidence
    ///
    /// Factors (from Task 006 calibration research):
    /// - Convergence speed: fewer iterations = higher confidence (weight: 0.3)
    /// - Energy reduction: deeper basin = higher confidence (weight: 0.25)
    /// - Convergence flag: must converge for high confidence (gate)
    #[must_use]
    pub fn to_completion_confidence(self, max_iterations: usize) -> Confidence {
        if !self.converged {
            return Confidence::exact(0.3); // Low confidence for non-convergence
        }

        // Convergence factor: 1.0 (fast) to 0.0 (slow)
        let convergence_factor = 1.0 - (self.iterations as f32 / max_iterations as f32);

        // Energy factor: normalized energy reduction (0.0 to 1.0)
        // Assuming typical energy range of -10.0 to 0.0
        let energy_factor = (self.energy_delta / 10.0).clamp(0.0, 1.0);

        // Weighted combination (Task 006: convergence_weight=0.3, energy_weight=0.25)
        let confidence = 0.3 * convergence_factor + 0.25 * energy_factor + 0.45; // Base confidence

        Confidence::exact(confidence.clamp(0.0, 1.0))
    }
}

/// Theta rhythm monitoring for production validation
pub struct ThetaRhythmMonitor {
    /// Histogram of iteration counts (0-7)
    pub iterations_histogram: [usize; 8],
    /// Average iterations per completion
    pub avg_iterations: f32,
    /// Convergence success rate
    pub convergence_rate: f32,
    /// Total completions recorded
    pub total_completions: usize,
    /// Successful convergences
    pub successful_convergences: usize,
}

impl ThetaRhythmMonitor {
    /// Create new theta rhythm monitor
    #[must_use]
    pub const fn new() -> Self {
        Self {
            iterations_histogram: [0; 8],
            avg_iterations: 0.0,
            convergence_rate: 0.0,
            total_completions: 0,
            successful_convergences: 0,
        }
    }

    /// Record a completion event
    pub fn record_completion(&mut self, iterations: usize, converged: bool) {
        self.total_completions += 1;
        if converged {
            self.successful_convergences += 1;
        }

        // Update histogram (clamp to 7)
        let idx = iterations.min(7);
        self.iterations_histogram[idx] += 1;

        // Update average iterations
        self.avg_iterations = self
            .iterations_histogram
            .iter()
            .enumerate()
            .map(|(i, &count)| i * count)
            .sum::<usize>() as f32
            / self.total_completions as f32;

        // Update convergence rate
        self.convergence_rate = self.successful_convergences as f32 / self.total_completions as f32;
    }

    /// Check if theta constraint is being violated (>5% exceed 7 iterations)
    #[must_use]
    pub fn violates_theta_constraint(&self) -> bool {
        if self.total_completions == 0 {
            return false;
        }

        let violations = self.iterations_histogram[7];
        violations as f32 / self.total_completions as f32 > 0.05
    }

    /// Get expected duration for average completion (assuming ~20ms per iteration)
    #[must_use]
    pub fn expected_duration(&self) -> Duration {
        let ms = self.avg_iterations * 20.0;
        Duration::from_millis(ms as u64)
    }
}

impl Default for ThetaRhythmMonitor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ca3_attractor_creation() {
        let ca3 = CA3Attractor::new(0.05, 7);
        assert!((ca3.sparsity - 0.05).abs() < 1e-6);
        assert_eq!(ca3.max_iterations, 7);
        assert!(ca3.validate_theta_constraint());
    }

    #[test]
    fn test_weight_matrix_symmetry_after_learning() {
        let mut ca3 = CA3Attractor::new(0.05, 7);
        let pattern = DVector::from_fn(768, |i, _| if i < 38 { 1.0 } else { 0.0 });
        ca3.learn_pattern(&pattern);

        // Check symmetry: W[i,j] == W[j,i]
        for i in 0..768 {
            for j in (i + 1)..768 {
                let diff = (ca3.weights[(i, j)] - ca3.weights[(j, i)]).abs();
                assert!(
                    diff < 1e-6,
                    "Weight asymmetry at ({}, {}): {} vs {}",
                    i,
                    j,
                    ca3.weights[(i, j)],
                    ca3.weights[(j, i)]
                );
            }
        }
    }

    #[test]
    fn test_no_self_connections() {
        let mut ca3 = CA3Attractor::new(0.05, 7);
        let pattern = DVector::from_fn(768, |i, _| if i < 38 { 1.0 } else { 0.0 });
        ca3.learn_pattern(&pattern);

        // Check diagonal is zero
        for i in 0..768 {
            assert!(ca3.weights[(i, i)].abs() < 1e-6, "Self-connection at {i}");
        }
    }

    #[test]
    fn test_sparsity_constraint_exact() {
        let mut ca3 = CA3Attractor::new(0.05, 7);
        let input = DVector::from_fn(768, |_, _| rand::random::<f32>());
        ca3.state = input;

        ca3.apply_sparsity();

        // Count non-zero activations
        let active = ca3.state.iter().filter(|&&v| v > 0.0).count();
        let expected = (768_f32 * 0.05_f32).round() as usize;
        assert_eq!(active, expected, "Sparsity constraint not exact");
    }

    #[test]
    fn test_theta_rhythm_monitor() {
        let mut monitor = ThetaRhythmMonitor::new();

        // Record many completions (20 total, only 1 at max = 5%)
        for _ in 0..5 {
            monitor.record_completion(3, true);
        }
        for _ in 0..5 {
            monitor.record_completion(4, true);
        }
        for _ in 0..5 {
            monitor.record_completion(5, true);
        }
        for _ in 0..4 {
            monitor.record_completion(6, true);
        }
        monitor.record_completion(7, false); // Only 1 out of 20 = 5%

        assert_eq!(monitor.total_completions, 20);
        assert_eq!(monitor.successful_convergences, 19);
        assert!((monitor.convergence_rate - 0.95).abs() < 0.01);
        assert!(!monitor.violates_theta_constraint());
    }
}
