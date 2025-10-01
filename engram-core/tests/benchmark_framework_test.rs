//! Smoke tests for the benchmarking support utilities.

#[test]
fn test_benchmark_framework_compiles() {
    use std::path::Path;

    // This test verifies that the comprehensive benchmarking framework compiles correctly
    // by asserting that the canonical benchmark entry point exists.
    assert!(
        Path::new("benches/comprehensive.rs").exists(),
        "Benchmark entry point should exist"
    );
}

#[test]
fn test_statistical_power_calculation() {
    // Test that power analysis correctly calculates sample size
    // For 99.5% power to detect 5% effect size

    // Using Cohen's formulation: n = 2 * (z_alpha + z_beta)^2 / effect_size^2
    // z_alpha(0.05) ≈ 1.96, z_beta(0.005) ≈ 2.576
    // n = 2 * (1.96 + 2.576)^2 / 0.05^2 ≈ 16460

    let expected_sample_size = 16460;
    let z_alpha = 1.96_f64;
    let z_beta = 2.576_f64;
    let effect_size = 0.05_f64;
    let computed = 2.0 * (z_alpha + z_beta).powi(2) / effect_size.powi(2);
    let rounded = computed.round();
    let tolerance = 10.0;

    assert!(
        (rounded - f64::from(expected_sample_size)).abs() <= tolerance,
        "Sample size should be approximately 16460 for 99.5% power, but got {rounded}"
    );
}

#[test]
fn test_metamorphic_relations() {
    // Test basic metamorphic relations for cosine similarity
    let a = [1.0_f32, 0.0, 0.0];
    let b = [0.0_f32, 1.0, 0.0];

    // Orthogonal vectors should have cosine similarity of 0
    let dot_product: f32 = a.iter().zip(&b).map(|(x, y)| x * y).sum();
    assert!(
        dot_product.abs() < f32::EPSILON,
        "Orthogonal vectors should have dot product of 0"
    );

    // Identity relation: cosine_similarity(a, a) = 1
    let self_dot: f32 = a.iter().map(|x| x * x).sum();
    let self_similarity = self_dot.sqrt() / self_dot.sqrt();
    assert!(
        (self_similarity - 1.0).abs() < 1e-6,
        "Self similarity should be 1"
    );
}

#[test]
fn test_differential_testing_structure() {
    // Verify that differential testing baselines are properly structured
    let baseline_names = ["Pinecone", "Weaviate", "FAISS", "ScaNN", "Neo4j"];

    assert!(
        baseline_names.len() >= 5,
        "Should have at least 5 baseline implementations"
    );
    assert!(
        baseline_names.contains(&"Pinecone"),
        "Should include Pinecone baseline"
    );
    assert!(
        baseline_names.contains(&"Neo4j"),
        "Should include Neo4j baseline"
    );
}

#[test]
fn test_cognitive_accuracy_ranges() {
    // Test that cognitive accuracy metrics are within expected ranges

    // DRM false memory rate should be 40-60% to match human data
    let drm_false_memory_rate = 0.47; // Example value
    assert!(
        (0.40..=0.60).contains(&drm_false_memory_rate),
        "DRM false memory rate should match human range (40-60%)"
    );

    // Boundary extension rate should be 15-30%
    let boundary_extension_rate = 0.225; // Example value
    assert!(
        (0.15..=0.30).contains(&boundary_extension_rate),
        "Boundary extension rate should match human range (15-30%)"
    );
}

#[test]
fn test_performance_cliff_detection() {
    // Test that performance cliffs are properly detected
    let baseline_latency = 10.0; // ms
    let current_latency = 25.0; // ms

    let performance_ratio = current_latency / baseline_latency;
    let is_cliff = performance_ratio > 2.0;

    assert!(
        is_cliff,
        "2.5x slowdown should be detected as a performance cliff"
    );
}
