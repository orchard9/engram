//! Integration tests for accuracy validation and production tuning
//!
//! This test file integrates all accuracy validation components:
//! - Corrupted episodes dataset validation
//! - DRM paradigm false memory testing
//! - Serial position curve biological plausibility
//! - Parameter tuning and Pareto frontier analysis
//!
//! ## Test Performance Optimization
//!
//! Pattern completion tests are computationally expensive due to:
//! - CA3 attractor dynamics with Hebbian learning
//! - Multiple HNSW traversals per completion
//! - Parameter sweeps testing multiple configurations
//!
//! Many tests are marked with `#[ignore]` to keep default test runs fast (<2 minutes).
//! These expensive tests (>30s each) should only run during full validation.
//!
//! ### Running Tests
//!
//! **Fast validation** (default, ~2 minutes):
//! ```bash
//! cargo test --test accuracy_validation_tests
//! ```
//!
//! **Full validation suite** (all tests including expensive ones, ~5-6 minutes):
//! ```bash
//! cargo test --test accuracy_validation_tests -- --ignored --nocapture
//! ```
//!
//! **Run both fast and expensive tests**:
//! ```bash
//! cargo test --test accuracy_validation_tests -- --include-ignored --nocapture
//! ```

#![cfg(feature = "pattern_completion")]

mod accuracy;

#[cfg(test)]
mod integration_tests {
    use super::accuracy::*;

    #[test]
    fn test_validation_metrics_integration() {
        // Ensure ValidationMetrics can be constructed and used
        use corrupted_episodes::ValidationMetrics;

        let metrics = ValidationMetrics {
            precision: 0.85,
            recall: 0.80,
            f1_score: ValidationMetrics::compute_f1(0.85, 0.80),
            per_field_accuracy: std::collections::HashMap::new(),
            total_completions: 100,
            successful_completions: 85,
        };

        assert!((metrics.f1_score - 0.824).abs() < 0.01);
    }

    #[test]
    fn test_drm_list_generation() {
        // Ensure DRM lists can be generated
        use drm_paradigm::DRMList;

        let lists = DRMList::classic_lists();
        assert!(!lists.is_empty());

        for list in &lists {
            assert!(!list.studied_words.is_empty());
            assert!(!list.critical_lure.is_empty());
        }
    }

    #[test]
    fn test_serial_position_data_construction() {
        // Ensure SerialPositionData can be constructed
        use serial_position::SerialPositionData;

        let data = SerialPositionData {
            position: 1,
            total_positions: 20,
            accuracy: 0.8,
            successful_recalls: 8,
            attempted_recalls: 10,
        };

        assert!(data.is_primacy());
        assert!(!data.is_middle());
        assert!(!data.is_recency());
    }

    #[test]
    fn test_parameter_sweep_configuration() {
        // Ensure ParameterSweep can be constructed
        use parameter_tuning::ParameterSweep;

        let sweep = ParameterSweep::default();

        assert_eq!(sweep.ca3_sparsity_values.len(), 5);
        assert_eq!(sweep.ca1_threshold_values.len(), 5);
        assert_eq!(sweep.num_hypotheses_values.len(), 5);

        // Verify biological constraints
        for &sparsity in &sweep.ca3_sparsity_values {
            assert!(
                (0.02..=0.10).contains(&sparsity),
                "Sparsity should be within biological range (2-10%)"
            );
        }
    }

    #[test]
    fn test_benchmark_dataset_creation() {
        // Ensure BenchmarkDataset can be created
        use parameter_tuning::BenchmarkDataset;

        let dataset = BenchmarkDataset::standard(42);

        assert_eq!(dataset.train_episodes.len(), 100);
        assert_eq!(dataset.test_episodes.len(), 100);
        assert_eq!(dataset.test_partials.len(), 100);
    }

    #[test]
    fn test_pareto_frontier_empty_set() {
        // Test Pareto frontier with empty set
        use parameter_tuning::find_pareto_frontier;

        let frontier = find_pareto_frontier(&[]);
        assert!(frontier.is_empty());
    }

    #[test]
    fn test_end_to_end_accuracy_validation() {
        // End-to-end test of the accuracy validation pipeline
        use corrupted_episodes::GroundTruthGenerator;
        use drm_paradigm::DRMList;
        use parameter_tuning::BenchmarkDataset;
        use serial_position::SerialPositionData;

        // Generate ground truth data
        let mut generator = GroundTruthGenerator::new(42);
        let episodes = generator.generate_episodes(10);
        assert_eq!(episodes.len(), 10);

        // Create DRM lists
        let drm_lists = DRMList::classic_lists();
        assert!(!drm_lists.is_empty());

        // Create benchmark dataset
        let dataset = BenchmarkDataset::standard(43);
        assert!(!dataset.train_episodes.is_empty());

        // Create serial position data
        let sp_data = SerialPositionData {
            position: 10,
            total_positions: 20,
            accuracy: 0.7,
            successful_recalls: 7,
            attempted_recalls: 10,
        };
        assert!(sp_data.is_middle());

        println!("End-to-end accuracy validation pipeline verified");
    }
}
