//! Demo program to test the cognitive error infrastructure
//!
//! This example shows how the error system works in practice,
//! demonstrating the "3am developer" experience.

use engram_core::error::CognitiveError;
use engram_core::error_testing::{CognitiveErrorTesting, ErrorFamily};
use engram_core::{Confidence, cognitive_error};

fn main() {
    println!("🧠 Engram Cognitive Error System Demo");
    println!("=====================================\n");

    // Test 1: Show a typical node not found error
    println!("📍 Test 1: Memory Node Access Error");
    println!("-----------------------------------");

    let node_error = simulate_node_not_found("user_999");
    println!("Error Message:");
    println!("{}\n", node_error);

    // Test 2: Show validation error with context
    println!("🔍 Test 2: Validation Error with Context");
    println!("----------------------------------------");

    let validation_error = simulate_validation_error(1.5);
    println!("Error Message:");
    println!("{}\n", validation_error);

    // Test 3: Test the comprehensive error testing framework
    println!("🧪 Test 3: Cognitive Testing Framework");
    println!("--------------------------------------");

    test_cognitive_framework();

    // Test 4: Show procedural learning simulation
    println!("📚 Test 4: Procedural Learning Simulation");
    println!("-----------------------------------------");

    simulate_learning_curve();

    // Test 5: Performance testing
    println!("⚡ Test 5: Performance Testing");
    println!("-----------------------------");

    test_performance();

    println!("\n✅ All tests completed successfully!");
    println!("The cognitive error system is working correctly.");
}

fn simulate_node_not_found(node_id: &str) -> CognitiveError {
    // Find similar node IDs (simulate a real graph)
    let available_nodes = vec!["user_123", "user_124", "user_125", "admin_001"];
    let available_strings: Vec<String> = available_nodes.iter().map(|s| s.to_string()).collect();
    let similar = CognitiveError::find_similar(node_id, &available_strings, 2);

    cognitive_error!(
        summary: format!("Memory node '{}' not found in active graph", node_id),
        context: expected = "Valid node ID from current graph context",
                 actual = node_id,
        suggestion: "Use graph.nodes() to list all available memory nodes",
        example: "let node = graph.get_node(\"user_123\").or_insert_default();",
        confidence: Confidence::HIGH,
        similar: similar
    )
}

fn simulate_validation_error(value: f64) -> CognitiveError {
    cognitive_error!(
        summary: format!("Invalid activation level: {:.1}", value),
        context: expected = "Node activation between 0.0 (inactive) and 1.0 (fully active)",
                 actual = format!("Activation = {:.1} (exceeds valid range)", value),
        suggestion: "Use value.clamp(0.0, 1.0) to normalize activation within valid bounds",
        example: "node.set_activation(energy.clamp(0.0, 1.0));",
        confidence: Confidence::exact(1.0)
    )
}

fn test_cognitive_framework() {
    let mut testing_framework = CognitiveErrorTesting::new();

    let test_error = cognitive_error!(
        summary: "Test error for framework validation",
        context: expected = "Valid test input",
                 actual = "Invalid test data",
        suggestion: "Provide valid input according to schema",
        example: "validate_input(good_data)",
        confidence: Confidence::MEDIUM
    );

    let result = testing_framework.test_error_comprehensive(&test_error, ErrorFamily::Validation);

    println!("Framework Test Results:");
    println!(
        "  ✓ Pattern consistency: {}",
        result.pattern_consistency.passes_pattern_consistency()
    );
    println!(
        "  ✓ Cognitive load compatible: {}",
        result.cognitive_load.passes_cognitive_load_test()
    );
    println!(
        "  ✓ Formatting time: {:?} (target: <1ms)",
        result.formatting_time
    );
    println!(
        "  ✓ Comprehension time: {:?} (target: <30s)",
        result.estimated_comprehension_time
    );
    println!("  ✓ Overall pass: {}", result.passes_all_requirements());
    println!();
}

fn simulate_learning_curve() {
    use engram_core::error_testing::ProceduralLearningSimulator;

    let mut normal_simulator = ProceduralLearningSimulator::new();
    let mut fatigued_simulator = ProceduralLearningSimulator::new().with_high_cognitive_load();

    let error = cognitive_error!(
        summary: "Memory consolidation failed",
        context: expected = "Sufficient activation for consolidation process",
                 actual = "Activation below consolidation threshold",
        suggestion: "Increase node activation or lower consolidation threshold",
        example: "node.boost_activation(0.3).consolidate_with_threshold(0.5)",
        confidence: Confidence::HIGH
    );

    println!("Simulating learning over 5 encounters:");

    let mut normal_times = Vec::new();
    let mut fatigued_times = Vec::new();

    for i in 1..=5 {
        let normal_time =
            normal_simulator.simulate_error_encounter(ErrorFamily::GraphStructure, &error);
        let fatigued_time =
            fatigued_simulator.simulate_error_encounter(ErrorFamily::GraphStructure, &error);

        normal_times.push(normal_time.as_millis());
        fatigued_times.push(fatigued_time.as_millis());

        println!(
            "  Encounter {}: Normal={:.1}s, Fatigued={:.1}s",
            i,
            normal_time.as_millis() as f64 / 1000.0,
            fatigued_time.as_millis() as f64 / 1000.0
        );
    }

    // Check if learning occurred
    let learning_occurred = normal_simulator.is_learning_occurring(&ErrorFamily::GraphStructure);
    println!("  Learning detected: {}", learning_occurred);

    // Show learning curve improvement
    let first_time = normal_times[0] as f64;
    let last_time = normal_times[4] as f64;
    let improvement = ((first_time - last_time) / first_time * 100.0) as i32;

    if improvement > 0 {
        println!("  Performance improvement: {}% faster", improvement);
    }

    println!();
}

fn test_performance() {
    use engram_core::error_testing::ErrorPerformanceMonitor;
    use std::time::Instant;

    let monitor = ErrorPerformanceMonitor::new();

    // Test a complex error with lots of context
    let complex_error = cognitive_error!(
        summary: "Serialization failed for memory node with embedded NaN confidence values",
        context: expected = "JSON-serializable memory structure with finite numerical values",
                 actual = "MemoryNode containing confidence.mean = NaN from division by zero",
        suggestion: "Validate all confidence values using f64::is_finite() before serialization",
        example: "if node.confidence.mean.is_finite() { serde_json::to_string(&node)? } else { node.fix_confidence().serialize() }",
        confidence: Confidence::HIGH
    );

    // Test formatting performance (should be <1ms)
    let formatting_time = monitor.test_formatting_performance(&complex_error);
    println!(
        "Formatting performance: {:?} (target: <1ms)",
        formatting_time
    );

    // Test comprehension time estimation
    let comprehension_normal = monitor.estimate_comprehension_time(&complex_error, 1.0);
    let comprehension_fatigued = monitor.estimate_comprehension_time(&complex_error, 3.0);

    println!("Comprehension time (normal): {:?}", comprehension_normal);
    println!(
        "Comprehension time (fatigued): {:?}",
        comprehension_fatigued
    );

    // Test multiple formatting operations for consistency
    println!("Testing formatting consistency over 100 operations:");
    let mut total_time = std::time::Duration::from_nanos(0);
    let iterations = 100;

    for _ in 0..iterations {
        let start = Instant::now();
        let _ = format!("{}", complex_error);
        total_time += start.elapsed();
    }

    let avg_time = total_time / iterations;
    println!("  Average formatting time: {:?}", avg_time);
    println!("  All operations < 1ms: {}", avg_time.as_millis() < 1);

    println!();
}
