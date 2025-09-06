//! Bisection tool for systematic divergence analysis
//!
//! This module provides cognitive-friendly bisection capabilities to help developers
//! understand and fix implementation divergences by automatically narrowing down
//! the root cause through systematic exploration.

use super::*;
use crate::Confidence;
use std::collections::VecDeque;
use std::fmt;

/// Bisection session for investigating a divergence
pub struct BisectionSession {
    /// The original operation that caused divergence
    pub original_operation: DifferentialOperation,
    /// The original divergent results
    pub original_results: HashMap<Implementation, OperationResult>,
    /// Queue of operations to test during bisection
    pub test_queue: VecDeque<DifferentialOperation>,
    /// Results collected during bisection
    pub bisection_results: Vec<BisectionResult>,
    /// Current hypothesis about the divergence cause
    pub current_hypothesis: Option<DivergenceHypothesis>,
    /// Configuration for bisection behavior
    pub config: BisectionConfig,
}

/// Result of a single bisection test
#[derive(Debug, Clone)]
pub struct BisectionResult {
    /// The operation tested
    pub operation: DifferentialOperation,
    /// Results from each implementation
    pub results: HashMap<Implementation, OperationResult>,
    /// Whether this operation shows divergence
    pub divergent: bool,
    /// Simplified explanation of what this test revealed
    pub insight: String,
    /// Distance from original operation (for complexity tracking)
    pub distance_from_original: u32,
}

/// Hypothesis about what might be causing a divergence
#[derive(Debug, Clone)]
pub struct DivergenceHypothesis {
    /// Human-readable description of the hypothesis
    pub description: String,
    /// Confidence in this hypothesis
    pub confidence: Confidence,
    /// Evidence supporting this hypothesis
    pub supporting_evidence: Vec<String>,
    /// Evidence contradicting this hypothesis
    pub contradicting_evidence: Vec<String>,
    /// Suggested next steps to test this hypothesis
    pub next_steps: Vec<String>,
}

/// Configuration for bisection behavior
#[derive(Debug, Clone)]
pub struct BisectionConfig {
    /// Maximum number of bisection steps
    pub max_steps: u32,
    /// Maximum complexity reduction to attempt
    pub max_simplification_depth: u32,
    /// Whether to explore boundary conditions
    pub explore_boundaries: bool,
    /// Whether to test special values (0, 1, infinity, etc.)
    pub test_special_values: bool,
    /// Epsilon for floating-point comparison
    pub epsilon: f64,
}

impl Default for BisectionConfig {
    fn default() -> Self {
        Self {
            max_steps: 50,
            max_simplification_depth: 5,
            explore_boundaries: true,
            test_special_values: true,
            epsilon: f32::EPSILON as f64,
        }
    }
}

/// Types of bisection strategies
#[derive(Debug, Clone, Copy)]
pub enum BisectionStrategy {
    /// Reduce input complexity systematically
    ComplexityReduction,
    /// Test boundary conditions
    BoundaryExploration,
    /// Test special mathematical values
    SpecialValueTesting,
    /// Binary search on continuous parameters
    ParameterBinarySearch,
    /// Test similar operations with slight variations
    OperationVariation,
}

impl BisectionSession {
    /// Create a new bisection session
    pub fn new(
        operation: DifferentialOperation,
        results: HashMap<Implementation, OperationResult>,
    ) -> Self {
        let mut session = Self {
            original_operation: operation.clone(),
            original_results: results,
            test_queue: VecDeque::new(),
            bisection_results: Vec::new(),
            current_hypothesis: None,
            config: BisectionConfig::default(),
        };

        session.initialize_test_queue();
        session
    }

    /// Configure the bisection session
    pub fn with_config(mut self, config: BisectionConfig) -> Self {
        self.config = config;
        self
    }

    /// Run the bisection session to identify the root cause
    pub fn run_bisection<T>(&mut self, tester: &T) -> BisectionReport
    where
        T: DifferentialExecutor,
    {
        println!(
            "🔍 Starting bisection analysis for: {}",
            self.original_operation.description()
        );

        let mut step = 0;
        while step < self.config.max_steps && !self.test_queue.is_empty() {
            if let Some(test_operation) = self.test_queue.pop_front() {
                println!(
                    "   Step {}: Testing {}",
                    step + 1,
                    test_operation.description()
                );

                let results = tester.execute_operation(&test_operation);
                let comparison = tester.compare_results(&results);

                let divergent = !comparison.equivalent;
                let insight = self.generate_insight(&test_operation, &results, divergent);

                let bisection_result = BisectionResult {
                    operation: test_operation.clone(),
                    results: results.clone(),
                    divergent,
                    insight: insight.clone(),
                    distance_from_original: self.calculate_distance(&test_operation),
                };

                println!(
                    "      Result: {} - {}",
                    if divergent { "DIVERGENT" } else { "EQUIVALENT" },
                    insight
                );
                self.bisection_results.push(bisection_result);

                // Update hypothesis based on this result
                self.update_hypothesis(&test_operation, &results, divergent);

                // Generate follow-up tests based on this result
                self.generate_followup_tests(&test_operation, divergent);

                step += 1;
            }
        }

        self.generate_report()
    }

    /// Initialize the test queue with bisection strategies
    fn initialize_test_queue(&mut self) {
        // Strategy 1: Complexity reduction
        if let Some(simplified) = self.original_operation.simplify() {
            self.test_queue.push_back(simplified);
        }

        // Strategy 2: Boundary exploration
        if self.config.explore_boundaries {
            self.add_boundary_tests();
        }

        // Strategy 3: Special value testing
        if self.config.test_special_values {
            self.add_special_value_tests();
        }

        // Strategy 4: Operation variations
        self.add_operation_variations();
    }

    /// Add boundary condition tests
    fn add_boundary_tests(&mut self) {
        match &self.original_operation {
            DifferentialOperation::ConfidenceAnd { .. }
            | DifferentialOperation::ConfidenceOr { .. } => {
                // Test with boundary values
                self.test_queue
                    .push_back(DifferentialOperation::ConfidenceAnd {
                        conf_a: 0.0,
                        conf_b: 0.0,
                    });
                self.test_queue
                    .push_back(DifferentialOperation::ConfidenceAnd {
                        conf_a: 1.0,
                        conf_b: 1.0,
                    });
                self.test_queue
                    .push_back(DifferentialOperation::ConfidenceAnd {
                        conf_a: 0.0,
                        conf_b: 1.0,
                    });
                self.test_queue
                    .push_back(DifferentialOperation::ConfidenceOr {
                        conf_a: 0.0,
                        conf_b: 0.0,
                    });
                self.test_queue
                    .push_back(DifferentialOperation::ConfidenceOr {
                        conf_a: 1.0,
                        conf_b: 1.0,
                    });
            }
            DifferentialOperation::ConfidenceFromSuccesses { .. } => {
                // Test edge cases for frequency-based construction
                self.test_queue
                    .push_back(DifferentialOperation::ConfidenceFromSuccesses {
                        successes: 0,
                        total: 1,
                    });
                self.test_queue
                    .push_back(DifferentialOperation::ConfidenceFromSuccesses {
                        successes: 1,
                        total: 1,
                    });
                self.test_queue
                    .push_back(DifferentialOperation::ConfidenceFromSuccesses {
                        successes: 0,
                        total: 0,
                    });
                self.test_queue
                    .push_back(DifferentialOperation::ConfidenceFromSuccesses {
                        successes: u32::MAX,
                        total: u32::MAX,
                    });
            }
            _ => {}
        }
    }

    /// Add special value tests (NaN, infinity, etc.)
    fn add_special_value_tests(&mut self) {
        match &self.original_operation {
            DifferentialOperation::ConfidenceAnd { .. }
            | DifferentialOperation::ConfidenceOr { .. }
            | DifferentialOperation::ConfidenceNot { .. }
            | DifferentialOperation::ConfidenceCalibration { .. } => {
                // Test with special floating-point values
                self.test_queue
                    .push_back(DifferentialOperation::ConfidenceNot { conf: f32::NAN });
                self.test_queue
                    .push_back(DifferentialOperation::ConfidenceNot {
                        conf: f32::INFINITY,
                    });
                self.test_queue
                    .push_back(DifferentialOperation::ConfidenceNot {
                        conf: f32::NEG_INFINITY,
                    });
                self.test_queue
                    .push_back(DifferentialOperation::ConfidenceNot { conf: f32::MIN });
                self.test_queue
                    .push_back(DifferentialOperation::ConfidenceNot { conf: f32::MAX });
            }
            _ => {}
        }
    }

    /// Add operation variations to test similar scenarios
    fn add_operation_variations(&mut self) {
        match &self.original_operation {
            DifferentialOperation::ConfidenceAnd { conf_a, conf_b } => {
                // Test OR with same values to see if it's specific to AND
                self.test_queue
                    .push_back(DifferentialOperation::ConfidenceOr {
                        conf_a: *conf_a,
                        conf_b: *conf_b,
                    });
                // Test with swapped parameters
                self.test_queue
                    .push_back(DifferentialOperation::ConfidenceAnd {
                        conf_a: *conf_b,
                        conf_b: *conf_a,
                    });
            }
            DifferentialOperation::ConfidenceOr { conf_a, conf_b } => {
                // Test AND with same values
                self.test_queue
                    .push_back(DifferentialOperation::ConfidenceAnd {
                        conf_a: *conf_a,
                        conf_b: *conf_b,
                    });
                // Test with swapped parameters
                self.test_queue
                    .push_back(DifferentialOperation::ConfidenceOr {
                        conf_a: *conf_b,
                        conf_b: *conf_a,
                    });
            }
            _ => {}
        }
    }

    /// Generate insight from a bisection test result
    fn generate_insight(
        &self,
        operation: &DifferentialOperation,
        results: &HashMap<Implementation, OperationResult>,
        divergent: bool,
    ) -> String {
        if !divergent {
            return format!(
                "✓ {} produces equivalent results across implementations",
                operation.description()
            );
        }

        // Analyze the specific type of divergence
        let mut insights = Vec::new();

        // Check for success/failure differences
        let success_patterns: Vec<bool> = results.values().map(|r| r.success).collect();
        if success_patterns.iter().any(|&s| s) && success_patterns.iter().any(|&s| !s) {
            insights.push("Some implementations succeed while others fail".to_string());
        }

        // Check for performance differences
        let execution_times: Vec<u64> = results.values().map(|r| r.execution_time_nanos).collect();
        if let (Some(&min_time), Some(&max_time)) =
            (execution_times.iter().min(), execution_times.iter().max())
        {
            if max_time > min_time * 2 {
                insights.push(format!(
                    "Performance varies significantly ({}ns to {}ns)",
                    min_time, max_time
                ));
            }
        }

        // Try to decode results for value comparison
        if let Some(ref_result) = results.values().next() {
            if ref_result.success && ref_result.result_data.is_some() {
                insights.push("Results differ in computed values".to_string());
            }
        }

        if insights.is_empty() {
            format!(
                "✗ {} shows implementation divergence",
                operation.description()
            )
        } else {
            format!("✗ {}: {}", operation.description(), insights.join("; "))
        }
    }

    /// Update the current hypothesis based on a bisection result
    fn update_hypothesis(
        &mut self,
        operation: &DifferentialOperation,
        results: &HashMap<Implementation, OperationResult>,
        divergent: bool,
    ) {
        // This is where we'd implement more sophisticated hypothesis updating
        // For now, create a simple hypothesis based on patterns observed

        if divergent {
            let description = match operation {
                DifferentialOperation::ConfidenceAnd { .. }
                | DifferentialOperation::ConfidenceOr { .. } => {
                    "Binary confidence operations may have implementation differences".to_string()
                }
                DifferentialOperation::ConfidenceFromSuccesses { .. } => {
                    "Frequency-based confidence construction may handle edge cases differently"
                        .to_string()
                }
                DifferentialOperation::ConfidenceCalibration { .. } => {
                    "Overconfidence calibration algorithms may differ between implementations"
                        .to_string()
                }
                _ => "Implementation-specific behavior detected".to_string(),
            };

            self.current_hypothesis = Some(DivergenceHypothesis {
                description,
                confidence: Confidence::MEDIUM,
                supporting_evidence: vec![format!(
                    "Divergence observed in {}",
                    operation.description()
                )],
                contradicting_evidence: Vec::new(),
                next_steps: vec![
                    "Test simpler variations of the same operation".to_string(),
                    "Compare intermediate computation steps".to_string(),
                    "Check mathematical specifications".to_string(),
                ],
            });
        }
    }

    /// Generate follow-up tests based on a result
    fn generate_followup_tests(&mut self, operation: &DifferentialOperation, divergent: bool) {
        if divergent {
            // If we found divergence, try to narrow it down further
            if let Some(simplified) = operation.simplify() {
                if !self.test_queue.contains(&simplified) {
                    self.test_queue.push_back(simplified);
                }
            }
        } else {
            // If equivalent, we might have narrowed down the issue
            // Generate slightly more complex variations
            match operation {
                DifferentialOperation::ConfidenceAnd { conf_a, conf_b } => {
                    // Try with slightly different values
                    let epsilon = 0.001;
                    self.test_queue
                        .push_back(DifferentialOperation::ConfidenceAnd {
                            conf_a: conf_a + epsilon,
                            conf_b: *conf_b,
                        });
                }
                _ => {}
            }
        }
    }

    /// Calculate the distance between an operation and the original
    fn calculate_distance(&self, operation: &DifferentialOperation) -> u32 {
        // Simple distance metric - in practice, this would be more sophisticated
        if std::mem::discriminant(operation) == std::mem::discriminant(&self.original_operation) {
            1 // Same operation type, different parameters
        } else {
            2 // Different operation type
        }
    }

    /// Generate a comprehensive bisection report
    fn generate_report(&self) -> BisectionReport {
        let total_tests = self.bisection_results.len();
        let divergent_tests = self
            .bisection_results
            .iter()
            .filter(|r| r.divergent)
            .count();
        let equivalent_tests = total_tests - divergent_tests;

        let patterns = self.identify_patterns();
        let recommendations = self.generate_recommendations();

        BisectionReport {
            original_operation: self.original_operation.clone(),
            total_tests_performed: total_tests as u32,
            divergent_tests: divergent_tests as u32,
            equivalent_tests: equivalent_tests as u32,
            identified_patterns: patterns,
            final_hypothesis: self.current_hypothesis.clone(),
            recommendations,
            detailed_results: self.bisection_results.clone(),
        }
    }

    /// Identify patterns in the bisection results
    fn identify_patterns(&self) -> Vec<String> {
        let mut patterns = Vec::new();

        // Pattern: All boundary conditions diverge
        let boundary_tests: Vec<_> = self
            .bisection_results
            .iter()
            .filter(|r| self.is_boundary_test(&r.operation))
            .collect();

        if !boundary_tests.is_empty() {
            let boundary_divergent = boundary_tests.iter().filter(|r| r.divergent).count();
            if boundary_divergent == boundary_tests.len() {
                patterns.push("All boundary condition tests show divergence".to_string());
            } else if boundary_divergent == 0 {
                patterns.push("All boundary condition tests are equivalent".to_string());
            }
        }

        // Pattern: Complexity correlation
        let simple_tests: Vec<_> = self
            .bisection_results
            .iter()
            .filter(|r| r.operation.cognitive_complexity() == CognitiveComplexity::Simple)
            .collect();

        if !simple_tests.is_empty() {
            let simple_divergent = simple_tests.iter().filter(|r| r.divergent).count();
            if simple_divergent == 0 {
                patterns.push(
                    "Simple operations are equivalent - complexity may be the issue".to_string(),
                );
            }
        }

        patterns
    }

    /// Check if an operation is a boundary test
    fn is_boundary_test(&self, operation: &DifferentialOperation) -> bool {
        match operation {
            DifferentialOperation::ConfidenceAnd { conf_a, conf_b }
            | DifferentialOperation::ConfidenceOr { conf_a, conf_b } => {
                *conf_a == 0.0 || *conf_a == 1.0 || *conf_b == 0.0 || *conf_b == 1.0
            }
            DifferentialOperation::ConfidenceFromSuccesses { successes, total } => {
                *successes == 0 || *successes == *total || *total <= 1
            }
            _ => false,
        }
    }

    /// Generate recommendations based on bisection results
    fn generate_recommendations(&self) -> Vec<String> {
        let mut recommendations = Vec::new();

        recommendations
            .push("1. Review the mathematical specifications for divergent operations".to_string());

        if self.bisection_results.iter().any(|r| r.divergent) {
            recommendations.push(
                "2. Focus investigation on the specific operations that show divergence"
                    .to_string(),
            );
            recommendations.push(
                "3. Compare intermediate computation steps between implementations".to_string(),
            );
        }

        if let Some(hypothesis) = &self.current_hypothesis {
            recommendations.extend(hypothesis.next_steps.iter().cloned());
        }

        recommendations
            .push("Final: Document the root cause and establish consistency tests".to_string());
        recommendations
    }
}

/// Report generated by a bisection session
#[derive(Debug)]
pub struct BisectionReport {
    /// The original operation that was investigated
    pub original_operation: DifferentialOperation,
    /// Total number of tests performed during bisection
    pub total_tests_performed: u32,
    /// Number of tests that showed divergence
    pub divergent_tests: u32,
    /// Number of tests that showed equivalence
    pub equivalent_tests: u32,
    /// Patterns identified during bisection
    pub identified_patterns: Vec<String>,
    /// Final hypothesis about the divergence cause
    pub final_hypothesis: Option<DivergenceHypothesis>,
    /// Recommendations for fixing the issue
    pub recommendations: Vec<String>,
    /// Detailed results from all bisection tests
    pub detailed_results: Vec<BisectionResult>,
}

impl fmt::Display for BisectionReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "🔍 Bisection Analysis Report")?;
        writeln!(f, "==========================")?;
        writeln!(f)?;
        writeln!(
            f,
            "Original Operation: {}",
            self.original_operation.description()
        )?;
        writeln!(f, "Tests Performed: {}", self.total_tests_performed)?;
        writeln!(f, "  ✓ Equivalent: {}", self.equivalent_tests)?;
        writeln!(f, "  ✗ Divergent: {}", self.divergent_tests)?;
        writeln!(f)?;

        if !self.identified_patterns.is_empty() {
            writeln!(f, "🔍 Identified Patterns:")?;
            for pattern in &self.identified_patterns {
                writeln!(f, "  • {}", pattern)?;
            }
            writeln!(f)?;
        }

        if let Some(hypothesis) = &self.final_hypothesis {
            writeln!(f, "🎯 Working Hypothesis:")?;
            writeln!(f, "  {}", hypothesis.description)?;
            writeln!(
                f,
                "  Confidence: {:.1}%",
                hypothesis.confidence.raw() * 100.0
            )?;

            if !hypothesis.supporting_evidence.is_empty() {
                writeln!(f, "  Supporting Evidence:")?;
                for evidence in &hypothesis.supporting_evidence {
                    writeln!(f, "    + {}", evidence)?;
                }
            }
            writeln!(f)?;
        }

        writeln!(f, "📋 Recommendations:")?;
        for recommendation in &self.recommendations {
            writeln!(f, "  {}", recommendation)?;
        }

        Ok(())
    }
}
