//! Cognitive load and memory-systems error testing framework
//!
//! This module implements comprehensive testing to ensure errors support
//! System 1 thinking and build procedural knowledge even under cognitive load.

use crate::error::CognitiveError;
use std::collections::HashMap;
use std::convert::TryFrom;
use std::time::{Duration, Instant};

/// Error families for pattern consistency testing
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ErrorFamily {
    /// Memory node access errors (get, not found, etc.)
    NodeAccess,
    /// Activation and confidence validation errors
    Validation,
    /// Serialization and data format errors
    Serialization,
    /// I/O and storage operation errors
    Storage,
    /// Graph structure and relationship errors
    GraphStructure,
}

/// Cognitive pattern structure that errors should follow
#[derive(Debug, Clone)]
pub struct CognitivePattern {
    /// Expected summary pattern (regex-like)
    pub summary_pattern: String,
    /// Required context structure
    pub context_structure: ContextStructure,
    /// Expected suggestion format
    pub suggestion_format: SuggestionFormat,
    /// Example code characteristics
    pub example_characteristics: ExampleCharacteristics,
}

/// Structure requirements for error context
#[derive(Debug, Clone)]
pub struct ContextStructure {
    /// Expected vs actual should be clearly contrasted
    pub has_clear_contrast: bool,
    /// Context should use domain-specific language
    pub uses_domain_language: bool,
    /// Should be understandable without prior context
    pub context_independent: bool,
}

/// Format requirements for suggestions
#[derive(Debug, Clone)]
pub struct SuggestionFormat {
    /// Should be an actionable imperative
    pub is_actionable: bool,
    /// Should mention specific methods/functions
    pub mentions_specific_methods: bool,
    /// Should be executable without research
    pub executable_directly: bool,
}

/// Characteristics required for example code
#[derive(Debug, Clone)]
pub struct ExampleCharacteristics {
    /// Should be syntactically valid Rust
    pub is_valid_rust: CheckResult,
    /// Should solve the actual problem
    pub solves_problem: CheckResult,
    /// Should be copy-pastable
    pub is_copy_pastable: CheckResult,
    /// Should use realistic variable names
    pub uses_realistic_names: CheckResult,
}

/// Binary outcome representation for cognitive checks
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CheckResult {
    /// Requirement is satisfied
    Passed,
    /// Requirement is not satisfied
    #[default]
    Failed,
}

impl CheckResult {
    /// Create a [`CheckResult`] from a boolean outcome.
    #[must_use]
    pub const fn from_bool(value: bool) -> Self {
        if value { Self::Passed } else { Self::Failed }
    }

    /// Return `true` when the result represents a passing outcome.
    #[must_use]
    pub const fn as_bool(self) -> bool {
        matches!(self, Self::Passed)
    }
}

/// Simulator for testing procedural knowledge building
#[derive(Debug)]
pub struct ProceduralLearningSimulator {
    /// Track resolution times for error families
    resolution_times: HashMap<ErrorFamily, Vec<Duration>>,
    /// Track success rates over time
    success_rates: HashMap<ErrorFamily, Vec<f64>>,
    /// Cognitive load simulation parameters
    cognitive_load_factor: f64,
}

impl ProceduralLearningSimulator {
    /// Create a new procedural learning simulator
    #[must_use]
    pub fn new() -> Self {
        Self {
            resolution_times: HashMap::new(),
            success_rates: HashMap::new(),
            cognitive_load_factor: 1.0,
        }
    }

    /// Simulate high cognitive load (tired developer at 3am)
    #[must_use]
    pub const fn with_high_cognitive_load(mut self) -> Self {
        self.cognitive_load_factor = 3.0; // 3x longer processing under load
        self
    }

    /// Simulate learning from repeated exposure to error family
    pub fn simulate_error_encounter(
        &mut self,
        family: ErrorFamily,
        error: &CognitiveError,
    ) -> Duration {
        let base_resolution_time = Self::estimate_base_resolution_time(error);

        // Apply learning curve: resolution time decreases with exposure
        let exposures = self
            .resolution_times
            .get(&family)
            .map_or(0, std::vec::Vec::len);
        let exposures_f64 = usize_to_f64(exposures);
        let learning_factor = 1.0 / exposures_f64.mul_add(0.1, 1.0);

        // Apply cognitive load
        let base_millis = base_resolution_time.as_secs_f64() * 1_000.0;
        let scaled_time = base_millis * learning_factor * self.cognitive_load_factor;
        let actual_time = duration_from_millis(scaled_time);

        // Record the encounter
        self.resolution_times
            .entry(family.clone())
            .or_default()
            .push(actual_time);

        // Simulate success rate (improves with learning, degrades with cognitive load)
        let base_success_rate = Self::estimate_success_rate(error);
        let learned_success_rate = base_success_rate + (exposures_f64 * 0.05).min(0.3);
        let cognitive_adjusted_rate = learned_success_rate / self.cognitive_load_factor.min(2.0);

        self.success_rates
            .entry(family)
            .or_default()
            .push(cognitive_adjusted_rate.min(1.0));

        actual_time
    }

    /// Estimate how long it would take to resolve this error
    fn estimate_base_resolution_time(error: &CognitiveError) -> Duration {
        let mut base_ms = 30_000; // Start with 30 seconds

        // Reduce time based on error quality
        if error.example.len() > 20 && error.example.contains("let ") {
            base_ms -= 10_000; // Good example reduces time
        }

        if error.suggestion.starts_with(char::is_lowercase) && error.suggestion.contains("()") {
            base_ms -= 5_000; // Specific method suggestions help
        }

        if !error.similar.is_empty() {
            base_ms -= 5_000; // "Did you mean" helps
        }

        Duration::from_millis(base_ms.max(5_000)) // Minimum 5 seconds
    }

    /// Estimate success rate based on error characteristics
    fn estimate_success_rate(error: &CognitiveError) -> f64 {
        let mut rate: f64 = 0.6; // Base 60% success rate

        // Improve based on error quality
        if error.confidence.raw() > 0.8 {
            rate += 0.15;
        }

        if error.example.contains("unwrap_or") || error.example.contains('?') {
            rate += 0.1; // Proper error handling in examples
        }

        if !error.context.expected.is_empty() && !error.context.actual.is_empty() {
            rate += 0.1; // Clear context
        }

        rate.min(1.0)
    }

    /// Get learning curve data for an error family
    #[must_use]
    pub fn get_learning_curve(&self, family: &ErrorFamily) -> Option<Vec<Duration>> {
        self.resolution_times.get(family).cloned()
    }

    /// Check if learning is occurring (resolution times decreasing)
    #[must_use]
    pub fn is_learning_occurring(&self, family: &ErrorFamily) -> bool {
        if let Some(times) = self.resolution_times.get(family) {
            if times.len() < 3 {
                return false; // Need at least 3 data points
            }

            let first_third = times.len() / 3;
            let last_third = times.len() - first_third;

            let early_avg = times[0..first_third]
                .iter()
                .map(std::time::Duration::as_millis)
                .sum::<u128>()
                / u128::try_from(first_third).unwrap_or(1);

            let late_avg = times[last_third..]
                .iter()
                .map(std::time::Duration::as_millis)
                .sum::<u128>()
                / u128::try_from(times.len() - last_third).unwrap_or(1);

            late_avg < early_avg // Learning if later attempts are faster
        } else {
            false
        }
    }
}

/// Test cognitive load handling
#[derive(Debug)]
pub struct CognitiveLoadTester {
    /// Maximum working memory chunks under load
    max_chunks_under_load: usize,
}

impl CognitiveLoadTester {
    /// Create a new cognitive load tester
    #[must_use]
    pub const fn new() -> Self {
        Self {
            max_chunks_under_load: 3,
        }
    }

    /// Test if error message fits within cognitive load constraints
    #[must_use]
    pub fn test_cognitive_load_compatibility(&self, error: &CognitiveError) -> CognitiveLoadResult {
        let mut result = CognitiveLoadResult::new();

        // Test information chunking
        result.chunk_count = Self::count_information_chunks(error);
        result.fits_cognitive_limits =
            CheckResult::from_bool(result.chunk_count <= self.max_chunks_under_load);

        // Test System 1 recognizability
        result.system1_recognizable = CheckResult::from_bool(Self::test_system1_recognition(error));

        // Test visual hierarchy
        result.has_clear_hierarchy = CheckResult::from_bool(Self::test_visual_hierarchy(error));

        // Test context independence
        result.context_independent = CheckResult::from_bool(Self::test_context_independence(error));

        result
    }

    /// Count discrete information chunks in the error
    const fn count_information_chunks(error: &CognitiveError) -> usize {
        let mut chunks = 0;

        // Each main section is a chunk, but count more carefully
        chunks += 1; // Summary (always 1 chunk)

        // Context is 1 chunk (expected vs actual is one comparison)
        if !error.context.expected.is_empty() || !error.context.actual.is_empty() {
            chunks += 1;
        }

        // Suggestion is 1 chunk
        if !error.suggestion.is_empty() {
            chunks += 1;
        }

        // Similar suggestions and details are optional chunks that don't count toward limits
        // since they're only shown when the developer needs them (progressive disclosure)

        chunks
    }

    /// Test if error can be recognized by System 1 (pattern recognition)
    fn test_system1_recognition(error: &CognitiveError) -> bool {
        // Look for recognizable patterns
        let has_recognizable_keywords = error.summary.to_lowercase().contains("not found")
            || error.summary.to_lowercase().contains("invalid")
            || error.summary.to_lowercase().contains("failed")
            || error.summary.to_lowercase().contains("error")
            || error.summary.to_lowercase().contains("memory")
            || error.summary.to_lowercase().contains("node");

        let has_clear_structure =
            !error.context.expected.is_empty() && !error.suggestion.is_empty();

        has_recognizable_keywords && has_clear_structure
    }

    /// Test if error has clear visual hierarchy
    const fn test_visual_hierarchy(error: &CognitiveError) -> bool {
        // Summary should be shorter than details
        let summary_short = error.summary.len() < 100;

        // Should have progressive disclosure
        let has_structure = !error.context.expected.is_empty() && !error.suggestion.is_empty();

        summary_short && has_structure
    }

    /// Test if error is understandable without prior context
    fn test_context_independence(error: &CognitiveError) -> bool {
        // Check if error defines its terms or gives clear context
        let has_clear_context =
            error.context.expected.len() > 10 && !error.context.actual.is_empty();

        // Check if suggestion is self-contained and actionable
        let self_contained_suggestion = error.suggestion.len() > 10
            && (error.suggestion.to_lowercase().contains("use ")
                || error.suggestion.to_lowercase().contains("call ")
                || error.suggestion.to_lowercase().contains("check ")
                || error.suggestion.to_lowercase().contains("try "));

        has_clear_context && self_contained_suggestion
    }
}

/// Results of cognitive load testing
#[derive(Debug)]
pub struct CognitiveLoadResult {
    /// Number of cognitive chunks identified
    pub chunk_count: usize,
    /// Whether content fits within cognitive limits
    pub fits_cognitive_limits: CheckResult,
    /// Whether recognizable by System 1 (fast thinking)
    pub system1_recognizable: CheckResult,
    /// Whether information has clear hierarchical structure
    pub has_clear_hierarchy: CheckResult,
    /// Whether understandable without additional context
    pub context_independent: CheckResult,
}

impl CognitiveLoadResult {
    const fn new() -> Self {
        Self {
            chunk_count: 0,
            fits_cognitive_limits: CheckResult::Failed,
            system1_recognizable: CheckResult::Failed,
            has_clear_hierarchy: CheckResult::Failed,
            context_independent: CheckResult::Failed,
        }
    }

    /// Overall pass/fail for cognitive load compatibility
    #[must_use]
    pub const fn passes_cognitive_load_test(&self) -> bool {
        self.fits_cognitive_limits.as_bool()
            && self.system1_recognizable.as_bool()
            && self.has_clear_hierarchy.as_bool()
            && self.context_independent.as_bool()
    }
}

/// Performance monitor for error formatting and comprehension
#[derive(Debug)]
pub struct ErrorPerformanceMonitor;

impl ErrorPerformanceMonitor {
    /// Create a new error performance monitor
    #[must_use]
    pub const fn new() -> Self {
        Self
    }

    /// Test that error formatting completes within 1ms
    #[must_use]
    pub fn test_formatting_performance(error: &CognitiveError) -> Duration {
        let start = Instant::now();
        let _formatted = format!("{error}");
        start.elapsed()
    }

    /// Simulate comprehension time (heuristic based on content)
    #[must_use]
    pub fn estimate_comprehension_time(error: &CognitiveError, cognitive_load: f64) -> Duration {
        let base_words = Self::count_words(error);
        let base_time_ms = base_words.saturating_mul(250); // ~250ms per word under normal conditions

        // Apply cognitive load multiplier
        let scaled_time = u64_to_f64(base_time_ms) * cognitive_load;

        duration_from_millis(scaled_time)
    }

    fn count_words(error: &CognitiveError) -> u64 {
        let text = format!("{error}");
        u64::try_from(text.split_whitespace().count()).unwrap_or(u64::MAX)
    }
}

/// Main testing framework coordinator
#[derive(Debug)]
pub struct CognitiveErrorTesting {
    pattern_registry: HashMap<ErrorFamily, CognitivePattern>,
    memory_simulator: ProceduralLearningSimulator,
    cognitive_load_tester: CognitiveLoadTester,
}

impl CognitiveErrorTesting {
    /// Create a new cognitive error testing system
    #[must_use]
    pub fn new() -> Self {
        let mut pattern_registry = HashMap::new();

        // Define expected patterns for each error family
        pattern_registry.insert(
            ErrorFamily::NodeAccess,
            CognitivePattern {
                summary_pattern: ".*not found.*".to_string(),
                context_structure: ContextStructure {
                    has_clear_contrast: true,
                    uses_domain_language: true,
                    context_independent: true,
                },
                suggestion_format: SuggestionFormat {
                    is_actionable: true,
                    mentions_specific_methods: true,
                    executable_directly: true,
                },
                example_characteristics: ExampleCharacteristics {
                    is_valid_rust: CheckResult::Passed,
                    solves_problem: CheckResult::Passed,
                    is_copy_pastable: CheckResult::Passed,
                    uses_realistic_names: CheckResult::Passed,
                },
            },
        );

        pattern_registry.insert(
            ErrorFamily::Validation,
            CognitivePattern {
                summary_pattern: ".*(invalid|validation failed).*".to_string(),
                context_structure: ContextStructure {
                    has_clear_contrast: true,
                    uses_domain_language: true,
                    context_independent: true,
                },
                suggestion_format: SuggestionFormat {
                    is_actionable: true,
                    mentions_specific_methods: true,
                    executable_directly: true,
                },
                example_characteristics: ExampleCharacteristics {
                    is_valid_rust: CheckResult::Passed,
                    solves_problem: CheckResult::Passed,
                    is_copy_pastable: CheckResult::Passed,
                    uses_realistic_names: CheckResult::Passed,
                },
            },
        );

        Self {
            pattern_registry,
            memory_simulator: ProceduralLearningSimulator::new(),
            cognitive_load_tester: CognitiveLoadTester::new(),
        }
    }

    /// Comprehensive test suite for a cognitive error
    pub fn test_error_comprehensive(
        &mut self,
        error: &CognitiveError,
        family: ErrorFamily,
    ) -> ErrorTestResult {
        let mut result = ErrorTestResult::new();

        // Pattern consistency test
        result.pattern_consistency = self.test_pattern_consistency(error, &family);

        // Cognitive load test
        result.cognitive_load = self
            .cognitive_load_tester
            .test_cognitive_load_compatibility(error);

        // Performance test
        result.formatting_time = ErrorPerformanceMonitor::test_formatting_performance(error);
        result.estimated_comprehension_time =
            ErrorPerformanceMonitor::estimate_comprehension_time(error, 1.0);

        // Memory system test
        result.resolution_time = self
            .memory_simulator
            .simulate_error_encounter(family, error);

        result
    }

    /// Test if error follows expected patterns for its family
    fn test_pattern_consistency(
        &self,
        error: &CognitiveError,
        family: &ErrorFamily,
    ) -> PatternConsistencyResult {
        let mut result = PatternConsistencyResult::new();

        if let Some(pattern) = self.pattern_registry.get(family) {
            // Test summary pattern
            result.summary_matches = CheckResult::from_bool(Self::test_regex_pattern(
                &error.summary,
                &pattern.summary_pattern,
            ));

            // Test context structure
            result.context_structure_valid = CheckResult::from_bool(Self::test_context_structure(
                error,
                &pattern.context_structure,
            ));

            // Test suggestion format
            result.suggestion_format_valid = CheckResult::from_bool(Self::test_suggestion_format(
                error,
                &pattern.suggestion_format,
            ));

            // Test example characteristics
            result.example_characteristics_valid = CheckResult::from_bool(
                Self::test_example_characteristics(error, &pattern.example_characteristics),
            );
        }

        result
    }

    fn test_regex_pattern(text: &str, pattern: &str) -> bool {
        // Simple regex-like pattern matching
        if pattern.starts_with(".*") && pattern.ends_with(".*") {
            let middle = &pattern[2..pattern.len() - 2];
            text.to_lowercase().contains(&middle.to_lowercase())
        } else {
            text == pattern
        }
    }

    fn test_context_structure(error: &CognitiveError, structure: &ContextStructure) -> bool {
        let has_contrast = !error.context.expected.is_empty() && !error.context.actual.is_empty();
        let uses_domain_terms = error.context.expected.to_lowercase().contains("node")
            || error.context.expected.to_lowercase().contains("activation")
            || error.context.expected.to_lowercase().contains("confidence");

        (!structure.has_clear_contrast || has_contrast)
            && (!structure.uses_domain_language || uses_domain_terms)
            && (!structure.context_independent || Self::is_context_independent(error))
    }

    fn test_suggestion_format(error: &CognitiveError, format: &SuggestionFormat) -> bool {
        let is_actionable = error.suggestion.starts_with(char::is_uppercase)
            && (error.suggestion.contains("Use ")
                || error.suggestion.contains("Check ")
                || error.suggestion.contains("Try "));

        let mentions_methods = error.suggestion.contains("()") || error.suggestion.contains('.');
        let is_executable = !error.suggestion.contains("TODO") && !error.suggestion.contains("...");

        (!format.is_actionable || is_actionable)
            && (!format.mentions_specific_methods || mentions_methods)
            && (!format.executable_directly || is_executable)
    }

    fn test_example_characteristics(
        error: &CognitiveError,
        chars: &ExampleCharacteristics,
    ) -> bool {
        let is_rust = error.example.contains("let ")
            || error.example.contains("();")
            || error.example.contains('?')
            || error.example.contains("unwrap");

        let solves_problem = error.example.len() > 10 && !error.example.contains("TODO");
        let is_copy_pastable =
            !error.example.contains("...") && !error.example.contains("<placeholder>");
        let realistic_names = !error.example.contains("foo") && !error.example.contains("bar");

        (!chars.is_valid_rust.as_bool() || is_rust)
            && (!chars.solves_problem.as_bool() || solves_problem)
            && (!chars.is_copy_pastable.as_bool() || is_copy_pastable)
            && (!chars.uses_realistic_names.as_bool() || realistic_names)
    }

    const fn is_context_independent(error: &CognitiveError) -> bool {
        // Error should be understandable without external context
        error.context.expected.len() > 10 && error.suggestion.len() > 10
    }

    /// Get learning curve for error family
    #[must_use]
    pub fn get_learning_curve(&self, family: &ErrorFamily) -> Option<Vec<Duration>> {
        self.memory_simulator.get_learning_curve(family)
    }

    /// Test if procedural learning is occurring
    #[must_use]
    pub fn is_procedural_learning_occurring(&self, family: &ErrorFamily) -> bool {
        self.memory_simulator.is_learning_occurring(family)
    }
}

/// Results of pattern consistency testing
#[derive(Debug)]
pub struct PatternConsistencyResult {
    /// Whether error summary matches expected pattern
    pub summary_matches: CheckResult,
    /// Whether context structure is valid
    pub context_structure_valid: CheckResult,
    /// Whether suggestion format is valid
    pub suggestion_format_valid: CheckResult,
    /// Whether example characteristics are valid
    pub example_characteristics_valid: CheckResult,
}

impl PatternConsistencyResult {
    const fn new() -> Self {
        Self {
            summary_matches: CheckResult::Failed,
            context_structure_valid: CheckResult::Failed,
            suggestion_format_valid: CheckResult::Failed,
            example_characteristics_valid: CheckResult::Failed,
        }
    }

    /// Check if all pattern consistency tests pass
    #[must_use]
    pub const fn passes_pattern_consistency(&self) -> bool {
        self.summary_matches.as_bool()
            && self.context_structure_valid.as_bool()
            && self.suggestion_format_valid.as_bool()
            && self.example_characteristics_valid.as_bool()
    }
}

/// Comprehensive test results for an error
#[derive(Debug)]
pub struct ErrorTestResult {
    /// Pattern consistency test results
    pub pattern_consistency: PatternConsistencyResult,
    /// Cognitive load test results
    pub cognitive_load: CognitiveLoadResult,
    /// Time taken to format the error
    pub formatting_time: Duration,
    /// Estimated time to comprehend the error
    pub estimated_comprehension_time: Duration,
    /// Estimated time to resolve the error
    pub resolution_time: Duration,
}

impl ErrorTestResult {
    const fn new() -> Self {
        Self {
            pattern_consistency: PatternConsistencyResult::new(),
            cognitive_load: CognitiveLoadResult::new(),
            formatting_time: Duration::from_millis(0),
            estimated_comprehension_time: Duration::from_millis(0),
            resolution_time: Duration::from_millis(0),
        }
    }

    /// Overall pass/fail for all cognitive requirements
    #[must_use]
    pub fn passes_all_requirements(&self) -> bool {
        self.pattern_consistency.passes_pattern_consistency()
            && self.cognitive_load.passes_cognitive_load_test()
            && self.formatting_time < Duration::from_millis(10)
            && self.estimated_comprehension_time < Duration::from_millis(30_000)
    }

    /// Get a detailed report of test results
    #[must_use]
    pub fn detailed_report(&self) -> String {
        format!(
            "Error Test Results:\n\
                Pattern Consistency: {}\n\
                Cognitive Load Compatible: {}\n\
                Formatting Time: {:?} (target: <10ms)\n\
                Comprehension Time: {:?} (target: <30s)\n\
                Resolution Time: {:?}\n\
                Overall Pass: {}",
            self.pattern_consistency.passes_pattern_consistency(),
            self.cognitive_load.passes_cognitive_load_test(),
            self.formatting_time,
            self.estimated_comprehension_time,
            self.resolution_time,
            self.passes_all_requirements()
        )
    }
}

impl Default for CognitiveErrorTesting {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for ProceduralLearningSimulator {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for CognitiveLoadTester {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for ErrorPerformanceMonitor {
    fn default() -> Self {
        Self::new()
    }
}

fn usize_to_f64(value: usize) -> f64 {
    let capped = u64::try_from(value).unwrap_or(u64::MAX);
    u64_to_f64(capped)
}

fn duration_from_millis(millis: f64) -> Duration {
    if !millis.is_finite() {
        return Duration::MAX;
    }
    let seconds = (millis / 1000.0).max(0.0);
    Duration::try_from_secs_f64(seconds).unwrap_or(Duration::MAX)
}

fn u64_to_f64(value: u64) -> f64 {
    const SHIFT: u32 = 32;
    const FACTOR: f64 = 4_294_967_296.0; // 2^32

    let high = value >> SHIFT;
    let low = value & 0xFFFF_FFFF;

    let high_part = u32::try_from(high).map_or_else(|_| f64::from(u32::MAX), f64::from);
    let low_part = u32::try_from(low).map_or_else(|_| f64::from(u32::MAX), f64::from);

    high_part.mul_add(FACTOR, low_part)
}
