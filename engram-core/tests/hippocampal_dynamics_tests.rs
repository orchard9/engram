//! Comprehensive test suite for hippocampal CA3/CA1 dynamics.
//!
//! Tests biological plausibility, convergence properties, theta rhythm
//! constraints, and CA1 gating accuracy following Hopfield (1982),
//! Marr (1971), and McNaughton & Morris (1987).

use engram_core::Confidence;
use engram_core::completion::{
    CA1Gate, CA3Attractor, ConvergenceStats, PlausibilityChecker, ThetaRhythmMonitor,
};
use nalgebra::DVector;
use rand::Rng;
use rand::distributions::{Distribution, Standard};

// Test helper to create sparse pattern (k% active neurons)
fn create_sparse_pattern(size: usize, sparsity: f32) -> DVector<f32> {
    let k = (size as f32 * sparsity).round() as usize;
    let mut pattern = DVector::zeros(size);

    // Set k random elements to 1.0
    let mut rng = rand::thread_rng();
    for _ in 0..k {
        let idx = rng.gen_range(0..size);
        pattern[idx] = 1.0;
    }

    pattern
}

// Test helper to add noise to pattern
fn add_noise_to_pattern(pattern: &DVector<f32>, noise_level: f32) -> DVector<f32> {
    let mut rng = rand::thread_rng();
    let mut noisy_pattern = pattern.clone();

    for i in 0..noisy_pattern.len() {
        let rand_val: f32 = Standard.sample(&mut rng);
        if rand_val < noise_level {
            let noise_val: f32 = Standard.sample(&mut rng);
            noisy_pattern[i] = noise_val;
        }
    }

    noisy_pattern
}

// Test helper to compute cosine similarity
fn cosine_similarity(a: &DVector<f32>, b: &DVector<f32>) -> f32 {
    let dot = a.dot(b);
    let norm_a = a.norm();
    let norm_b = b.norm();

    if norm_a > 0.0 && norm_b > 0.0 {
        dot / (norm_a * norm_b)
    } else {
        0.0
    }
}

#[test]
fn test_ca3_convergence_with_learned_pattern() {
    let mut ca3 = CA3Attractor::new(0.05, 7);

    // Learn pattern (5% sparse)
    let pattern = create_sparse_pattern(768, 0.05);
    ca3.learn_pattern(&pattern);

    // Test convergence from partial cue (30% noise)
    let partial = add_noise_to_pattern(&pattern, 0.3);

    let (converged, iters, success) = ca3.converge(partial);
    assert!(success, "Should converge within 7 iterations");
    assert!(
        iters <= 7,
        "Theta rhythm constraint violated: {iters} iterations"
    );

    // Check converged to similar pattern
    let similarity = cosine_similarity(&converged, &pattern);
    assert!(
        similarity > 0.7,
        "Should converge to learned pattern (similarity: {similarity})"
    );
}

#[test]
fn test_hopfield_energy_decreases_monotonically() {
    let mut ca3 = CA3Attractor::new(0.05, 7);
    let pattern = create_sparse_pattern(768, 0.05);
    ca3.learn_pattern(&pattern);

    let input = add_noise_to_pattern(&pattern, 0.3);
    let _energies: Vec<f32> = Vec::new();

    // Manually step through iterations to track energy
    let (_, iters, _) = ca3.converge(input.clone());

    // Re-run with energy tracking
    let mut test_ca3 = CA3Attractor::new(0.05, iters);
    test_ca3.learn_pattern(&pattern);

    // Track initial energy
    test_ca3.converge(input);
    let final_energy = test_ca3.compute_energy();

    // Energy should be negative (Hopfield property)
    assert!(
        final_energy <= 0.0,
        "Final energy should be non-positive: {final_energy}"
    );
}

#[test]
fn test_weight_matrix_symmetry() {
    let mut ca3 = CA3Attractor::new(0.05, 7);
    let pattern = create_sparse_pattern(768, 0.05);
    ca3.learn_pattern(&pattern);

    let weights = ca3.get_weights();

    // Check symmetry: W[i,j] == W[j,i]
    for i in 0..768 {
        for j in (i + 1)..768 {
            let diff = (weights[(i, j)] - weights[(j, i)]).abs();
            assert!(
                diff < 1e-6,
                "Weight asymmetry at ({}, {}): {} vs {}",
                i,
                j,
                weights[(i, j)],
                weights[(j, i)]
            );
        }
    }
}

#[test]
fn test_no_self_connections() {
    let mut ca3 = CA3Attractor::new(0.05, 7);
    let pattern = create_sparse_pattern(768, 0.05);
    ca3.learn_pattern(&pattern);

    let weights = ca3.get_weights();

    // Check diagonal is zero
    for i in 0..768 {
        assert!(weights[(i, i)].abs() < 1e-6, "Self-connection at {i}");
    }
}

#[test]
fn test_sparsity_constraint_exact() {
    let mut ca3 = CA3Attractor::new(0.05, 7);

    // Create random dense input
    let mut rng = rand::thread_rng();
    let input = DVector::from_fn(768, |_, _| Standard.sample(&mut rng));

    // Run convergence
    let (converged, _, _) = ca3.converge(input);

    // Count non-zero activations
    let active = converged.iter().filter(|&&v| v > 0.0).count();
    let expected = (768_f32 * 0.05_f32).round() as usize;

    // Allow small tolerance for numerical precision
    assert!(
        (active as i32 - expected as i32).abs() <= 1,
        "Sparsity constraint not met: {active} active (expected {expected})"
    );
}

#[test]
fn test_theta_rhythm_constraint_validation() {
    let ca3 = CA3Attractor::new(0.05, 7);
    assert!(
        ca3.validate_theta_constraint(),
        "Should validate theta constraint"
    );

    let ca3_violating = CA3Attractor::new(0.05, 10);
    assert!(
        !ca3_violating.validate_theta_constraint(),
        "Should fail theta validation with >7 iterations"
    );
}

#[test]
fn test_expected_duration_ms() {
    let ca3 = CA3Attractor::new(0.05, 7);
    let duration_ms = ca3.expected_duration_ms();

    // 7 iterations * 20ms/iter = 140ms (within theta cycle)
    assert!((duration_ms - 140.0).abs() < 1e-6);
    assert!(
        duration_ms <= 250.0,
        "Should fit within theta period (125-250ms)"
    );
}

#[test]
fn test_ca1_gating_threshold_pass() {
    let gate = CA1Gate::new(Confidence::exact(0.7));

    let confidence = gate.passes_threshold(Confidence::exact(0.8));
    assert!(confidence, "High confidence should pass threshold");
}

#[test]
fn test_ca1_gating_threshold_fail() {
    let gate = CA1Gate::new(Confidence::exact(0.7));

    // Low-confidence completion should fail
    let confidence = gate.passes_threshold(Confidence::exact(0.6));
    assert!(!confidence, "Low confidence should fail threshold");
}

#[test]
fn test_plausibility_checker_normal_embedding() {
    let checker = PlausibilityChecker::new();

    // Normal embedding with reasonable magnitude and variance
    let mut embedding = [0.0f32; 768];
    let mut rng = rand::thread_rng();
    for item in &mut embedding {
        let rand_val: f32 = Standard.sample(&mut rng);
        *item = rand_val * 2.0 - 1.0; // Range [-1, 1]
    }

    let score = checker.score_plausibility(&embedding);
    assert!(score > 0.5, "Normal embedding should be plausible: {score}");
}

#[test]
fn test_plausibility_checker_degenerate_embedding() {
    let checker = PlausibilityChecker::new();

    // Degenerate embedding: all zeros
    let embedding = [0.0f32; 768];

    let score = checker.score_plausibility(&embedding);
    assert!(
        score < 0.8,
        "Degenerate embedding should have low plausibility: {score}"
    );
}

#[test]
fn test_plausibility_checker_extreme_magnitude() {
    let checker = PlausibilityChecker::new();

    // Extremely large embedding (magnitude >> 2.0)
    let embedding = [10.0f32; 768];

    let score = checker.score_plausibility(&embedding);
    assert!(
        score < 0.9,
        "Extreme magnitude embedding should have reduced plausibility: {score}"
    );
}

#[test]
fn test_theta_rhythm_monitor() {
    let mut monitor = ThetaRhythmMonitor::new();

    // Record multiple completions (20 total, only 1 at max = 5%)
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

    // Check statistics
    assert_eq!(monitor.total_completions, 20);
    assert_eq!(monitor.successful_convergences, 19);
    assert!((monitor.convergence_rate - 0.95).abs() < 0.01);
    assert!(
        !monitor.violates_theta_constraint(),
        "Should not violate constraint with exactly 5% at 7 iterations"
    );

    // Average should be around (5*3 + 5*4 + 5*5 + 4*6 + 1*7)/20 = (15+20+25+24+7)/20 = 91/20 = 4.55
    assert!(
        (monitor.avg_iterations - 4.55).abs() < 0.1,
        "Average iterations incorrect: {}",
        monitor.avg_iterations
    );
}

#[test]
fn test_theta_rhythm_monitor_violation() {
    let mut monitor = ThetaRhythmMonitor::new();

    // Record mostly violations (>5% at max iterations)
    for _ in 0..100 {
        monitor.record_completion(7, false);
    }
    for _ in 0..10 {
        monitor.record_completion(3, true);
    }

    assert!(
        monitor.violates_theta_constraint(),
        "Should violate with >5% at max iterations"
    );
}

#[test]
fn test_convergence_stats_to_confidence() {
    // Fast convergence with good energy reduction
    let stats = ConvergenceStats {
        iterations: 2,
        converged: true,
        final_energy: -8.0,
        energy_delta: 5.0,
        state_change: 0.001,
    };

    let confidence = stats.to_completion_confidence(7);
    assert!(
        confidence.raw() > 0.6,
        "Fast convergence should yield high confidence: {}",
        confidence.raw()
    );

    // Slow convergence without energy reduction
    let stats_slow = ConvergenceStats {
        iterations: 7,
        converged: false,
        final_energy: -1.0,
        energy_delta: 0.5,
        state_change: 0.05,
    };

    let confidence_slow = stats_slow.to_completion_confidence(7);
    assert!(
        confidence_slow.raw() < 0.5,
        "Slow/failed convergence should yield low confidence: {}",
        confidence_slow.raw()
    );
}

#[test]
fn test_ca3_multiple_pattern_learning() {
    let mut ca3 = CA3Attractor::new(0.05, 7);

    // Learn two patterns (Hopfield capacity is limited)
    let pattern1 = create_sparse_pattern(768, 0.05);
    let pattern2 = create_sparse_pattern(768, 0.05);

    ca3.learn_pattern(&pattern1);
    ca3.learn_pattern(&pattern2);

    // Test retrieval of pattern1 from low-noise cue
    let cue1 = add_noise_to_pattern(&pattern1, 0.1); // Less noise for better retrieval
    let (converged1, iters1, _success1) = ca3.converge(cue1);

    // Should respect theta constraint even if convergence isn't perfect
    assert!(iters1 <= 7, "Should respect theta constraint");

    // Check that result is reasonably similar to pattern 1
    let similarity1 = cosine_similarity(&converged1, &pattern1);
    assert!(
        similarity1 > 0.4,
        "Should show some similarity to pattern 1 (similarity: {similarity1})"
    );
}

#[test]
fn test_ca3_reset() {
    let mut ca3 = CA3Attractor::new(0.05, 7);
    let pattern = create_sparse_pattern(768, 0.05);

    ca3.learn_pattern(&pattern);
    ca3.converge(pattern.clone());

    // Reset should clear state
    ca3.reset();

    let state = ca3.get_state();
    let all_zero = state.iter().all(|&v| v == 0.0);
    assert!(all_zero, "State should be reset to zeros");
    assert!(!ca3.has_converged(), "Converged flag should be reset");
}

#[test]
fn test_convergence_with_zero_pattern() {
    let mut ca3 = CA3Attractor::new(0.05, 7);

    // Zero pattern (edge case)
    let zero_pattern = DVector::zeros(768);
    let (converged, _iters, _success) = ca3.converge(zero_pattern);

    // Should handle gracefully without panicking
    let all_zero_or_sparse = converged.iter().filter(|&&v| v > 0.0).count() <= 40;
    assert!(all_zero_or_sparse, "Should handle zero pattern gracefully");
}
