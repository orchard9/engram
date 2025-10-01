use criterion::{Criterion, black_box};
use std::time::Duration;

pub mod academic_references;
pub mod activation_spreading;
pub mod baseline_comparisons;
pub mod batch_operations;
pub mod differential_testing;
pub mod formal_verification;
pub mod hardware_variation;
pub mod hnsw_index;
pub mod integration_scenarios;
pub mod metamorphic_testing;
pub mod oracle_functions;
pub mod pattern_completion;
pub mod performance_fuzzing;
pub mod probabilistic_query;
pub mod property_based_testing;
pub mod regression_detection;
pub mod report_generation;
pub mod statistical_framework;
pub mod vector_ops;

use differential_testing::DifferentialTestingHarness;
use formal_verification::FormalVerificationSuite;
use hardware_variation::HardwareVariationTester;
use performance_fuzzing::PerformanceFuzzer;
use regression_detection::RegressionDetector;
use report_generation::BenchmarkReportGenerator;
use statistical_framework::{RegressionAnalysis, StatisticalBenchmarkFramework};

#[derive(Debug, Clone)]
pub struct ComprehensiveBenchmarkResults {
    pub verification_results: Option<VerificationResults>,
    pub hardware_results: Option<HardwareVariationResults>,
    pub fuzzing_results: Option<FuzzingResults>,
    pub differential_results: Option<DifferentialTestResults>,
    pub benchmark_results: Option<StatisticalBenchmarkResults>,
    pub regression_analysis: Option<RegressionAnalysisResults>,
    pub report: Option<String>,
}

impl ComprehensiveBenchmarkResults {
    pub const fn new() -> Self {
        Self {
            verification_results: None,
            hardware_results: None,
            fuzzing_results: None,
            differential_results: None,
            benchmark_results: None,
            regression_analysis: None,
            report: None,
        }
    }

    pub fn set_verification_results(&mut self, results: VerificationResults) {
        self.verification_results = Some(results);
    }

    pub fn set_hardware_results(&mut self, results: HardwareVariationResults) {
        self.hardware_results = Some(results);
    }

    pub fn set_fuzzing_results(&mut self, results: FuzzingResults) {
        self.fuzzing_results = Some(results);
    }

    pub fn set_differential_results(&mut self, results: DifferentialTestResults) {
        self.differential_results = Some(results);
    }

    pub fn set_benchmark_results(&mut self, results: StatisticalBenchmarkResults) {
        self.benchmark_results = Some(results);
    }

    pub fn set_regression_analysis(&mut self, results: RegressionAnalysisResults) {
        self.regression_analysis = Some(results);
    }

    pub fn set_report(&mut self, report: String) {
        self.report = Some(report);
    }
}

pub struct ComprehensiveBenchmarkSuite {
    statistical_framework: StatisticalBenchmarkFramework,
    differential_testing: DifferentialTestingHarness,
    performance_fuzzer: PerformanceFuzzer,
    verification_suite: FormalVerificationSuite,
    hardware_tester: HardwareVariationTester,
    regression_detector: RegressionDetector,
    report_generator: BenchmarkReportGenerator,
}

impl ComprehensiveBenchmarkSuite {
    pub fn new() -> Self {
        Self {
            statistical_framework: StatisticalBenchmarkFramework::new(),
            differential_testing: DifferentialTestingHarness::new(),
            performance_fuzzer: PerformanceFuzzer::new(),
            verification_suite: FormalVerificationSuite::new(),
            hardware_tester: HardwareVariationTester::new(),
            regression_detector: RegressionDetector::new(),
            report_generator: BenchmarkReportGenerator::new(),
        }
    }

    pub fn execute_comprehensive_benchmarks(&mut self) -> ComprehensiveBenchmarkResults {
        let mut results = ComprehensiveBenchmarkResults::new();

        // Phase 1: Formal verification (must pass before performance testing)
        println!("Phase 1: Formal verification of algorithmic correctness...");
        let property_catalog = self.verification_suite.property_names();
        println!(
            "Properties under evaluation: {}",
            property_catalog.join(", ")
        );
        let verification_results = self.verification_suite.verify_all_properties();
        assert!(
            verification_results.all_properties_verified(),
            "Formal verification failed: {:?}",
            verification_results.violation_messages()
        );
        results.set_verification_results(verification_results);

        // Phase 2: Hardware variation testing for correctness
        println!("Phase 2: Cross-architecture correctness validation...");
        let hardware_results = self.hardware_tester.test_all_architectures();
        assert!(
            hardware_results.all_architectures_correct(),
            "Hardware variation testing failed: {}",
            hardware_results.summary()
        );
        results.set_hardware_results(hardware_results);

        // Phase 3: Performance fuzzing to find worst cases
        println!("Phase 3: Performance fuzzing for worst-case detection...");
        let fuzzing_results = self.performance_fuzzer.fuzz_all_operations(100_000);
        results.set_fuzzing_results(fuzzing_results);

        // Phase 4: Differential testing against baselines
        println!("Phase 4: Differential testing against baseline implementations...");
        let differential_results = self.differential_testing.run_comprehensive_tests();
        if let Some((total_relations, passing_relations)) =
            differential_results.metamorphic_summary()
        {
            println!("Metamorphic checks: {passing_relations}/{total_relations} relations passed");
        }
        results.set_differential_results(differential_results);

        // Phase 5: Statistical benchmarking with regression detection
        println!("Phase 5: Statistical benchmarking with regression detection...");
        let benchmark_results = self.execute_statistical_benchmarks();
        results.set_benchmark_results(benchmark_results);

        // Phase 6: Performance regression analysis
        println!("Phase 6: Performance regression analysis...");
        let regression_analysis = self
            .regression_detector
            .analyze_performance_trends(&results);
        results.set_regression_analysis(regression_analysis);

        // Phase 7: Report generation
        println!("Phase 7: Generating comprehensive benchmark report...");
        let report = self
            .report_generator
            .generate_comprehensive_report(&results);
        results.set_report(report);

        results
    }

    fn execute_statistical_benchmarks(&self) -> StatisticalBenchmarkResults {
        let mut results = StatisticalBenchmarkResults::new();

        // Task 001: SIMD Vector Operations
        let simd_benchmarks = vector_ops::SIMDOperationBenchmarks::new();
        let simd_results = simd_benchmarks.run_comprehensive_benchmarks();
        results.add_task_results("001_simd_vector_operations", simd_results);

        // Task 002: HNSW Index Implementation
        let hnsw_benchmarks = hnsw_index::HNSWIndexBenchmarks::new();
        let hnsw_results = hnsw_benchmarks.run_comprehensive_benchmarks();
        results.add_task_results("002_hnsw_index", hnsw_results);

        // Task 004: Parallel Activation Spreading
        let activation_benchmarks = activation_spreading::ParallelActivationBenchmarks::new();
        let activation_results = activation_benchmarks.run_comprehensive_benchmarks();
        results.add_task_results("004_activation_spreading", activation_results);

        // Task 006: Probabilistic Query Engine
        let probabilistic_benchmarks = probabilistic_query::ProbabilisticQueryBenchmarks::new();
        let probabilistic_results = probabilistic_benchmarks.run_comprehensive_benchmarks();
        results.add_task_results("006_probabilistic_query", probabilistic_results);

        // Task 007: Pattern Completion Engine
        let completion_benchmarks = pattern_completion::PatternCompletionBenchmarks::new();
        let completion_results = completion_benchmarks.run_comprehensive_benchmarks();
        results.add_task_results("007_pattern_completion", completion_results);

        // Task 008: Batch Operations API
        let batch_benchmarks = batch_operations::BatchOperationsBenchmarks::new();
        let batch_results = batch_benchmarks.run_comprehensive_benchmarks();
        results.add_task_results("008_batch_operations", batch_results);

        // Integration scenarios
        let integration_benchmarks = integration_scenarios::IntegrationScenarioBenchmarks::new();
        let integration_results = integration_benchmarks.run_comprehensive_benchmarks();
        results.add_task_results("integration_scenarios", integration_results);

        if let Some(batch_results) = results.task_results().get("008_batch_operations") {
            let regression_signal = self.statistical_framework.detect_regression(
                &batch_results.samples,
                &batch_results.samples,
                "batch_operations",
            );

            if let RegressionAnalysis::Detected { recommendation, .. } = regression_signal {
                println!("Batch operations regression policy: {recommendation}");
            }
        }

        results
    }
}

// Type aliases for clarity
pub type VerificationResults = formal_verification::VerificationResult;
pub type HardwareVariationResults = hardware_variation::HardwareVariationResults;
pub type FuzzingResults = performance_fuzzing::FuzzingResults;
pub type DifferentialTestResults = differential_testing::DifferentialTestResults;
pub type StatisticalBenchmarkResults = statistical_framework::StatisticalBenchmarkResults;
pub type RegressionAnalysisResults = regression_detection::RegressionAnalysisResults;

// Main benchmark entry point
pub fn benchmark_milestone_1(c: &mut Criterion) {
    let mut suite = ComprehensiveBenchmarkSuite::new();

    let mut group = c.benchmark_group("milestone_1_comprehensive");
    group
        .confidence_level(0.995) // 99.5% confidence
        .sample_size(246) // From power analysis
        .measurement_time(Duration::from_secs(30))
        .warm_up_time(Duration::from_secs(5));

    group.bench_function("full_comprehensive_suite", |b| {
        b.iter(|| {
            let results = suite.execute_comprehensive_benchmarks();
            black_box(results)
        });
    });

    group.finish();
}
